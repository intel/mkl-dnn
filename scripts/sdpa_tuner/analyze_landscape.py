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
"""Inspect a sweep problem by problem.

Part of scripts/sdpa_tuner; see README.md there. Most useful on a sweep run
with tune_fwd_config.py --all, which times every valid config, so the
per-parameter view covers the whole space. For each problem it prints
  - the fastest --top configs with their tile quantities, time relative to
    the best, model rank and estimate;
  - the measured time of the model's eight top-ranked configs;
  - one-factor marginals: for each config parameter (um_kq, un_kq, um_vs,
    un_vs, wm_kq, wn_kq, wm_vs, wn_vs: the eight config integers in order)
    and derived quantity (kv tile, Q tile, subgroups, GRF mode), the best
    and median time over the configs with each value, relative to the best;
  - the log-log correlation of the model's estimate with the time.

    python scripts/sdpa_tuner/analyze_landscape.py landscape.csv \\
        [--top 12] [--match REGEX]
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpa_model as sm  # noqa: E402

FACTORS = [
    "um_kq",
    "un_kq",
    "um_vs",
    "un_vs",
    "wm_kq",
    "wn_kq",
    "wm_vs",
    "wn_vs",
    "kv",
    "q_tile",
    "sg",
    "grf_actual",
]


def load(path, match):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] not in ("ok", "verified") or not r["ms_min"]:
                continue
            if match and not re.search(match, r["problem"]):
                continue
            c = sm.parse_cfg(r["cfg"])
            row = dict(
                cfg=r["cfg"],
                ms=float(r["ms_min"]),
                est=float(r["cost_us"] or "nan"),
                rank=int(r["rank_model"]) if r["rank_model"] else None,
                kv=int(r["kv"] or 0),
                q_tile=int(r["q_tile"] or 0),
                sg=int(r["sg"] or 0),
                grf_actual=r["grf_actual"],
                status=r["status"],
                set=r["set"],
            )
            for name, v in zip(FACTORS[:8], c):
                row[name] = v
            groups[r["problem"]].append(row)
    return groups


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("csv", help="sweep CSV from tune_fwd_config.py")
    ap.add_argument(
        "--top", type=int, default=12, help="fastest configs to list"
    )
    ap.add_argument(
        "--match", default="", help="regex over problems: only matching ones"
    )
    args = ap.parse_args()

    for problem, rows in load(args.csv, args.match).items():
        rows.sort(key=lambda r: r["ms"])
        best = rows[0]["ms"]
        print("=" * 100)
        print(
            "%s   %d configs, best %.4f ms, median %.4f ms, worst %.4f ms"
            % (
                problem,
                len(rows),
                best,
                median([r["ms"] for r in rows]),
                rows[-1]["ms"],
            )
        )
        print("  fastest:")
        for r in rows[: args.top]:
            print(
                "    %-24s kv=%-5d q=%-4d sg=%-3d grf=%-4s %8.4f ms  x%.3f "
                " model rank %-4s est %8.1f us  %s"
                % (
                    r["cfg"],
                    r["kv"],
                    r["q_tile"],
                    r["sg"],
                    r["grf_actual"],
                    r["ms"],
                    r["ms"] / best,
                    r["rank"],
                    r["est"],
                    r["set"] if r["set"] != "all" else "",
                )
            )
        print("  model top ranks as measured:")
        by_rank = sorted(
            [r for r in rows if r["rank"] is not None], key=lambda r: r["rank"]
        )
        for r in by_rank[:8]:
            print(
                "    rank %-3d %-24s %8.4f ms  x%.3f"
                % (r["rank"], r["cfg"], r["ms"], r["ms"] / best)
            )
        print(
            "  one-factor marginals (value: best / median, over configs with"
            " that value):"
        )
        for f in FACTORS:
            vals = defaultdict(list)
            for r in rows:
                vals[r[f]].append(r["ms"])
            if len(vals) < 2:
                continue
            parts = []
            for v in sorted(vals, key=lambda x: (isinstance(x, str), x)):
                parts.append(
                    "%s: x%.2f / x%.2f (%d)"
                    % (
                        v,
                        min(vals[v]) / best,
                        median(vals[v]) / best,
                        len(vals[v]),
                    )
                )
            print("    %-10s %s" % (f, "   ".join(parts)))
        # correlation of log time with log estimate
        pairs = [
            (math.log(r["est"]), math.log(r["ms"]))
            for r in rows
            if r["est"] == r["est"]
        ]
        if len(pairs) > 2:
            n = len(pairs)
            mx = sum(p[0] for p in pairs) / n
            my = sum(p[1] for p in pairs) / n
            cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
            vx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
            vy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
            print(
                "  log-log correlation of estimate with time: %.2f"
                % (cov / (vx * vy) if vx and vy else float("nan"))
            )


if __name__ == "__main__":
    main()
