#!/usr/bin/env python3
# *******************************************************************************
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************
"""Settle table keys shared by several swept problems.

Part of scripts/sdpa_tuner; README.md there shows where it fits in the
workflow. Sizes are bucketed in the table key (q=197 -> q256,
bh=6144 -> bh8192), so problems swept separately can land on one key, and
bake_fit.py keeps whichever line it merged last. Run this before baking
when combining sweeps of one architecture, or when a new sweep hits a key
the shipped table already has.

For every key shared by two or more problems in the CSVs it takes each
problem's fastest usable config and the shipped one (read from --select),
times every candidate on each problem of the key where the CSVs lack a
timing, picks the config with the lowest geomean regret over them, checks
it for correctness on each, and writes "key config" lines for the keys
whose pick beats the shipped config by more than --min-gain:

    python scripts/sdpa_tuner/resolve_collisions.py \\
        --benchdnn build/tests/benchdnn/benchdnn \\
        --tag a750 --out fixes_a750.txt "sweep_*_a750.csv"
    python scripts/sdpa_tuner/bake_fit.py \\
        --table table_a750.txt --table fixes_a750.txt

Without --benchdnn it uses only the timings already in the CSVs. The runs
it makes go through tune_fwd_config.py into collisions_<tag>_fill.csv and
collisions_<tag>_verify.csv in the current directory and are read back
next time, so an interrupted run resumes. --show REGEX prints the timings
behind the keys of matching problems instead. The library comes from
$DNNL_LIB, else build/src under the repo.
"""

import argparse
import collections
import csv
import glob
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_fwd_model as fm  # noqa: E402
import sdpa_model as sm  # noqa: E402

TUNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tune_fwd_config.py"
)


def load(paths):
    t: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    bad: Dict[str, Set[str]] = collections.defaultdict(set)
    ok: Dict[str, Set[str]] = collections.defaultdict(set)
    key: Dict[str, str] = {}
    legacy: Dict[str, str] = {}
    row: Dict[str, Any] = {}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                p, c = r.get("problem"), r.get("cfg")
                if not p or not c or not r.get("key"):
                    continue
                key[p] = r["key"]
                if r.get("set") == "legacy":
                    legacy[p] = c
                if p not in row:
                    rr = dict(r)
                    rr["ms_min"] = rr.get("ms_min") or "0"
                    row[p] = fm.Row(rr)
                if r["status"] == "mismatch":
                    bad[p].add(c)
                elif r["status"] in ("ok", "verified") and r["ms_min"]:
                    # files come oldest first: the latest timing wins
                    t[p][c] = float(r["ms_min"])
                    if r["status"] == "verified":
                        ok[p].add(c)
    return t, bad, ok, key, row, legacy


def show(pattern, shipped, t, key, legacy):
    keys = sorted({key[p] for p in t if re.search(pattern, p)})
    for k in keys:
        cur = shipped.get(k)
        print("%s\n  shipped %s" % (k, cur))
        for p in sorted(q for q in t if key[q] == k):
            tp = t[p]
            best = min(tp, key=tp.get)
            lg = legacy.get(p)
            print(
                "    shipped %-9s best %-9s %-22s legacy %-9s %s"
                % (
                    "%.4f" % tp[cur] if cur in tp else "-",
                    "%.4f" % tp[best],
                    best,
                    "%.4f" % tp[lg] if lg in tp else "-",
                    p[:90],
                )
            )


def tune(args, problem, cfgs, out, verify):
    cmd = [sys.executable, TUNER, "--benchdnn", args.benchdnn]
    cmd += ["--problem", problem, "--topk", "0", "--random", "0"]
    cmd += ["--no-legacy", "--verify-top", "1" if verify else "0"]
    for c in sorted(cfgs):
        cmd += ["--extra", c]
    subprocess.run(cmd + ["--out", out], stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "csv",
        nargs="+",
        help="tuner CSVs, globs allowed; read oldest first, newer timings win",
    )
    ap.add_argument(
        "--benchdnn",
        help="benchdnn to run missing timings and checks (default: none run)",
    )
    ap.add_argument(
        "--tag", default="dev", help="name in the collisions_<tag>_*.csv files"
    )
    ap.add_argument("--out", help="file to write the table lines to")
    ap.add_argument(
        "--select",
        default="src/gpu/intel/sdpa/select.cpp",
        help="select.cpp to read the shipped table from",
    )
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="regret improvement needed to replace a shipped config",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="rounds of timing and checking before giving up on a key",
    )
    ap.add_argument(
        "--show",
        help="regex over problems: print the timings behind their table keys",
    )
    args = ap.parse_args()
    if not args.show and not args.out:
        ap.error("--out is required")

    fill_csv = "collisions_%s_fill.csv" % args.tag
    verify_csv = "collisions_%s_verify.csv" % args.tag
    paths = sorted(
        {p for g in args.csv for p in glob.glob(g)}, key=os.path.getmtime
    )
    paths = [p for p in paths if p not in (fill_csv, verify_csv)]
    with open(args.select, encoding="utf-8") as f:
        shipped = dict(re.findall(r'"(xe[^" ]+) ([\d,]+)"', f.read()))
    lib = sm.Library(None)
    fm.LIB = lib
    tried: Set[Tuple[str, str]] = set()
    fixes: Dict[str, str] = {}
    report: List[str] = []

    for rnd in range(args.rounds + 1):
        own = [p for p in (fill_csv, verify_csv) if os.path.exists(p)]
        t, bad, ok, key, row, legacy = load(paths + own)
        if args.show:
            show(args.show, shipped, t, key, legacy)
            return
        bykey: Dict[str, List[str]] = collections.defaultdict(list)
        for p in t:
            bykey[key[p]].append(p)

        def usable(p, c):
            return (
                c not in bad[p]
                and lib.estimate(row[p].hw, row[p].prb, c) is not None
            )

        # Callers only pass configs timed on every problem in ps
        def regret(c: str, ps: List[str]) -> float:
            return math.exp(
                sum(
                    math.log(
                        t[p][c]
                        / min(v for k, v in t[p].items() if k not in bad[p])
                    )
                    for p in ps
                )
                / len(ps)
            )

        fill: Dict[str, Set[str]] = collections.defaultdict(set)
        check: List[Tuple[str, str]] = []
        fixes, report = {}, []
        for k, ps in sorted(bykey.items()):
            if len(ps) < 2:
                continue
            cur = shipped.get(k)
            cur_bad = cur is not None and any(not usable(p, cur) for p in ps)
            cands = set()
            for p in ps:
                own_best = [c for c in t[p] if usable(p, c)]
                if own_best:
                    cands.add(min(own_best, key=t[p].__getitem__))
            if cur and not cur_bad:
                cands.add(cur)
            for c in cands:
                for p in ps:
                    if c not in t[p]:
                        fill[p].add(c)
            if any(c not in t[p] for c in cands for p in ps):
                continue
            common = [c for c in cands if all(usable(p, c) for p in ps)]
            if not common:
                report.append("no usable config for %s" % k)
                continue
            best = min(common, key=lambda c: regret(c, ps))
            r_cur = regret(cur, ps) if cur and not cur_bad else None
            if best == cur or (
                r_cur is not None and regret(best, ps) > r_cur - args.min_gain
            ):
                continue
            todo = [
                p for p in ps if best not in ok[p] and (p, best) not in tried
            ]
            check += [(p, best) for p in todo]
            if all(best in ok[p] for p in ps):
                fixes[k] = best
                report.append(
                    "%s %s -> %s  %s -> x%.3f  (%d problems)"
                    % (
                        k,
                        cur,
                        best,
                        (
                            "x%.3f" % r_cur
                            if r_cur
                            else ("fails" if cur_bad else "new")
                        ),
                        regret(best, ps),
                        len(ps),
                    )
                )
            elif not todo:
                report.append(
                    "%s %s: %s could not be verified" % (k, cur, best)
                )

        print(
            "round %d: %d shared keys, %d need timings, %d checks, %d fixes"
            % (
                rnd,
                sum(len(v) > 1 for v in bykey.values()),
                len(fill),
                len(check),
                len(fixes),
            )
        )
        if not args.benchdnn or (not fill and not check) or rnd == args.rounds:
            break
        for p, cs in fill.items():
            tune(args, p, cs, fill_csv, verify=False)
        for p, c in check:
            tried.add((p, c))
            tune(args, p, [c], verify_csv, verify=True)

    for line in report:
        print("  " + line)
    with open(args.out, "w") as f:
        f.writelines("%s %s\n" % kv for kv in sorted(fixes.items()))
    print("%d table lines -> %s" % (len(fixes), args.out))


if __name__ == "__main__":
    main()
