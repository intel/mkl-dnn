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
"""Sweep forward SDPA tile configs with benchdnn and record the timings.

Part of scripts/sdpa_tuner; README.md there covers the workflow and the
problem file format. Needs a ONEDNN_DEV_MODE build: the config overrides
(SDPA_CONFIG), the fwd_select verbose line and the cost model API are
dev-mode only.

Sweep (default). For every problem the script
  1. runs benchdnn once to read the fwd_select line: the table key, the
     current config and its source, the hand-tuned (legacy) pick, and the
     device and problem descriptions;
  2. asks the library's cost model for every valid config, best first;
  3. times the model's --topk configs, --random others, the hand-tuned pick
     and any --extra configs (or every valid config with --all), forcing
     each one through SDPA_CONFIG;
  4. re-times the first config every --ref-every runs and warns when it
     drifts by more than 5% (the GPU is throttling or busy);
  5. checks the fastest configs for correctness until --verify-top pass.
It appends one row per timed config to --out, the input of
fit_fwd_model.py, report_fails.py, analyze_landscape.py and
resolve_collisions.py, and with --table-out one "key config" line per
problem for its fastest config that passed correctness, the input of
bake_fit.py. A row's status is ok (timed), verified (timed and correct),
mismatch (wrong results), fail (did not build), fallback (benchdnn ran
another implementation) or timeout.

--compare runs no sweep: it times each problem with the library's own
selection and with the hand-tuned config forced, and prints the ratio per
problem and its geomean. This is the acceptance check after a rebuild.

Problems come from --problems files, --problem strings, or --design N, a
random sample over head sizes, sequence lengths, batch, heads and masks.
One problem per line, either key=value tokens for the --sdpa driver or
graph="..." holding verbatim benchdnn --graph arguments (README.md,
Problem files, lists the keys):
    b=2 h=20 q=64 k=77 d=64 dt=f16
    b=1 h=32 hk=8 q=1 k=2049 d=128 mask=causal_bottom_right args=--ktag=abdc
    graph="--reset --dt=0:s8+2:s8+... --in-shapes=... --case=...json"

Examples, from the repository root:
    python scripts/sdpa_tuner/tune_fwd_config.py \\
        --benchdnn build/tests/benchdnn/benchdnn \\
        --problems scripts/sdpa_tuner/problems/kpi.txt \\
        --out sweep.csv --table-out table.txt
    python scripts/sdpa_tuner/tune_fwd_config.py --benchdnn ... \\
        --design 60 --seed 1 --out sweep.csv
    python scripts/sdpa_tuner/tune_fwd_config.py --benchdnn ... \\
        --problems scripts/sdpa_tuner/problems/kpi.txt --compare
"""

import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpa_model as sm  # noqa: E402

SELECT_RE = re.compile(
    r"fwd_select,key:(?P<key>[^,]+),src:(?P<src>\w+),cfg:(?P<cfg>[\d,]+?),"
    r"legacy:(?P<legacy>[\d,]+?|none),kv:(?P<kv>\d+),q:(?P<q>\d+),"
    r"sg:(?P<sg>\d+),slm:(?P<slm>\d+),grf:(?P<grf>\d+),wgss:(?P<wgss>\d+),"
    r"cost_us:(?P<cost>[\d.eE+-]+),hw:(?P<hw>[^,]+),prb:(?P<prb>[^,]+)"
)
GRF_RE = re.compile(r"grf_min: (\d+)/(\d+)")
PERF_RE = re.compile(r"min\(ms\):([\d.eE+-]+) avg\(ms\):([\d.eE+-]+)")
IMPL_RE = re.compile(r"primitive,exec,gpu,sdpa,([^,]+),")
PASSED_RE = re.compile(r"tests:(\d+) passed:(\d+)")
SKIPPED_RE = re.compile(r"skipped:([1-9]\d*)")

SPEC_DEFAULTS: Dict[str, Any] = dict(
    b=1,
    h=1,
    hk=None,
    q=None,
    k=None,
    d=None,
    dv=None,
    dt="f16",
    mask="none",
    mdt="",
    args="",
    graph="",
)

DESIGN_AXES: Dict[str, List[Any]] = dict(
    d=[32, 64, 96, 128, 256],
    k=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    q=[1, 16, 64, 256, 1024, 4096, "k"],
    b=[1, 2, 4, 8],
    h=[1, 8, 16, 32, 64],
    mask=["none", "causal_top_left", "buffer_1d"],
)


def parse_spec(line):
    spec = dict(SPEC_DEFAULTS)
    for tok in shlex.split(line.split("#", 1)[0]):
        if "=" not in tok:
            raise ValueError("bad token %r in %r" % (tok, line))
        k, v = tok.split("=", 1)
        if k not in spec:
            raise ValueError("unknown key %r in %r" % (k, line))
        spec[k] = v
    for k in ("b", "h", "hk", "q", "k", "d", "dv"):
        if spec[k] is not None and spec[k] != "":
            spec[k] = int(spec[k])
    if spec["graph"]:
        # a verbatim benchdnn --graph command line (dt, op-attrs, in-shapes,
        # case); the problem description comes from the library's verbose line
        return spec
    if spec["q"] is None or spec["k"] is None or spec["d"] is None:
        raise ValueError("q, k and d are required: %r" % line)
    if spec["hk"] in (None, ""):
        spec["hk"] = spec["h"]
    if spec["dv"] in (None, ""):
        spec["dv"] = spec["d"]
    return spec


def spec_str(spec):
    if spec["graph"]:
        return "graph=%s" % shlex.quote(spec["graph"])
    parts = ["b=%d" % spec["b"], "h=%d" % spec["h"]]
    if spec["hk"] != spec["h"]:
        parts.append("hk=%d" % spec["hk"])
    parts += ["q=%d" % spec["q"], "k=%d" % spec["k"], "d=%d" % spec["d"]]
    if spec["dv"] != spec["d"]:
        parts.append("dv=%d" % spec["dv"])
    parts += ["dt=%s" % spec["dt"], "mask=%s" % spec["mask"]]
    if spec["mdt"]:
        parts.append("mdt=%s" % spec["mdt"])
    if spec["args"]:
        parts.append("args=%s" % shlex.quote(spec["args"]))
    return " ".join(parts)


def design(n, seed, dtype, mask, max_bytes):
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    tries = 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        d = rng.choice(DESIGN_AXES["d"])
        k = rng.choice(DESIGN_AXES["k"])
        q = rng.choice(DESIGN_AXES["q"])
        q = k if q == "k" else q
        b = rng.choice(DESIGN_AXES["b"])
        h = rng.choice(DESIGN_AXES["h"])
        m = mask or rng.choice(DESIGN_AXES["mask"])
        if m.startswith("causal") and q > k:
            continue
        bytes_per = 4 if dtype == "f32" else 2
        if b * h * max(q, k) * d * bytes_per > max_bytes:
            continue
        spec = dict(
            SPEC_DEFAULTS, b=b, h=h, hk=h, q=q, k=k, d=d, dv=d, dt=dtype, mask=m
        )
        key = spec_str(spec)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def batch1_graph(spec):
    """The graph spec with its batch cut to 1 through --in-shapes, for cases
    too large for benchdnn's correctness check; None when not applicable."""
    if not spec["graph"] or "--in-shapes" in spec["graph"]:
        return None
    toks = shlex.split(spec["graph"])
    ci = next((i for i, t in enumerate(toks) if t.startswith("--case=")), None)
    if ci is None or not os.path.isfile(toks[ci].split("=", 1)[1]):
        return None
    with open(toks[ci].split("=", 1)[1]) as f:
        g = json.load(f)
    mm = next((op for op in g["graph"] if op["kind"] == "MatMul"), None)
    batch = mm["inputs"][0]["shape"][0] if mm else 1
    if batch <= 1:
        return None
    shapes = {}
    for op in g["graph"]:
        for t in op["inputs"]:
            if t["id"] in g["input_ports"] and t["shape"][:1] == [batch]:
                shapes[t["id"]] = [1] + t["shape"][1:]
    toks.insert(
        ci,
        "--in-shapes="
        + "+".join(
            "%d:%s" % (i, "x".join(map(str, s)))
            for i, s in sorted(shapes.items())
        ),
    )
    return dict(spec, graph=shlex.join(toks))


def benchdnn_args(spec, mode, max_ms=0):
    if spec["graph"]:
        args = ["--graph", "--engine=gpu", "--mode=%s" % mode]
        if mode.upper() == "P" and max_ms > 0:
            args.append("--max-ms-per-prb=%d" % max_ms)
        return args + shlex.split(spec["graph"])
    desc = "%dx%dx%dx%d:%dx%dx%dx%d:%dx%dx%dx%d" % (
        spec["b"],
        spec["h"],
        spec["q"],
        spec["d"],
        spec["b"],
        spec["hk"],
        spec["d"],
        spec["k"],
        spec["b"],
        spec["hk"],
        spec["k"],
        spec["dv"],
    )
    args = ["--sdpa", "--engine=gpu", "--mode=%s" % mode]
    if mode.upper() == "P" and max_ms > 0:
        args.append("--max-ms-per-prb=%d" % max_ms)
    args += ["--dt=%s" % spec["dt"], "--mask=%s" % spec["mask"]]
    if spec["mdt"]:
        args.append("--mdt=%s" % spec["mdt"])
    if spec["args"]:
        args += shlex.split(spec["args"])
    args.append(desc)
    return args


class Runner:
    def __init__(self, benchdnn, timeout, dry_run, verbose):
        self.benchdnn = benchdnn
        self.timeout = timeout
        self.dry_run = dry_run
        self.verbose = verbose
        self.base_env = dict(os.environ)
        for k in ("SDPA_CONFIG", "QUANTIZED_SDPA_CONFIG"):
            self.base_env.pop(k, None)
        lib_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(benchdnn)), "..", "..", "src"
            )
        )
        if os.path.isdir(lib_dir):
            self.base_env["PATH"] = (
                lib_dir + os.pathsep + self.base_env.get("PATH", "")
            )

    def run(self, args, env_extra, verbose_flags="debuginfo=4,profile_exec"):
        env = dict(self.base_env)
        env["ONEDNN_VERBOSE"] = verbose_flags
        env.update(env_extra)
        cmd = [self.benchdnn] + args
        if self.verbose or self.dry_run:
            print(
                "  $",
                " ".join(
                    ["%s=%s" % kv for kv in env_extra.items()]
                    + [shlex.quote(a) for a in cmd]
                ),
            )
        if self.dry_run:
            return ""
        try:
            res = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
                universal_newlines=True,
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        return res.stdout


def parse_select(text) -> Optional[Dict[str, Any]]:
    m = SELECT_RE.search(text)
    if not m:
        return None
    d: Dict[str, Any] = m.groupdict()
    d["hw_obj"] = sm.HW.from_verbose(d["hw"])
    d["prb_obj"] = sm.Problem.from_verbose(d["prb"])
    return d


def model_candidates(lib, sel):
    """Every valid config for the selection's hw/prb with the model's
    estimate and derived tile quantities, best estimate first."""
    cands = []
    for i, cfg in enumerate(lib.enumerate(sel["hw"], sel["prb"])):
        cfg_s = sm.cfg_str(cfg)
        est = lib.estimate(sel["hw"], sel["prb"], cfg_s)
        if est is None:
            continue
        cost, (kv, q, sg, slm, grf, wgss) = est
        cands.append(
            dict(
                rank=str(i),
                cfg=cfg_s,
                kv=str(kv),
                q=str(q),
                sg=str(sg),
                slm=str(slm),
                grf=str(grf),
                wgss=str(wgss),
                cost="%.3f" % cost,
            )
        )
    return cands


def parse_perf(text):
    out = dict(ms_min="", ms_avg="", grf_actual="", impl="", status="fail")
    if text == "TIMEOUT":
        out["status"] = "timeout"
        return out
    g = GRF_RE.search(text)
    if g:
        out["grf_actual"] = str(max(int(g.group(1)), int(g.group(2))))
    im = IMPL_RE.search(text)
    if im:
        out["impl"] = im.group(1)
    p = PERF_RE.search(text)
    if p:
        out["ms_min"], out["ms_avg"] = p.group(1), p.group(2)
        if float(out["ms_min"]) <= 0:
            # benchdnn prints 0 when primitive creation failed
            out["ms_min"] = out["ms_avg"] = ""
            out["status"] = "fail"
        elif out["impl"] and "micro" not in out["impl"]:
            out["status"] = "fallback"
        else:
            out["status"] = "ok"
    return out


def compare(specs, runner, args):
    """Time the library's own selection against the hand-tuned config."""
    import math

    ratios = []
    for pi, spec in enumerate(specs):
        pstr = spec_str(spec)
        sel = parse_select(runner.run(benchdnn_args(spec, "I"), {}))
        if sel is None:
            print(
                "[%d/%d] %s: no selection line, skipping"
                % (pi + 1, len(specs), pstr)
            )
            continue
        own = parse_perf(
            runner.run(benchdnn_args(spec, args.perf_mode, args.max_ms), {})
        )
        if sel["legacy"] in ("none", sel["cfg"]):
            print(
                "[%d/%d] %s: %s %s ms (%s), legacy %s"
                % (
                    pi + 1,
                    len(specs),
                    pstr,
                    sel["cfg"],
                    own["ms_min"] or own["status"],
                    sel["src"],
                    "identical" if sel["legacy"] != "none" else "none",
                )
            )
            continue
        env = {
            "SDPA_CONFIG": sel["legacy"],
            "QUANTIZED_SDPA_CONFIG": sel["legacy"],
        }
        leg = parse_perf(
            runner.run(benchdnn_args(spec, args.perf_mode, args.max_ms), env)
        )
        if own["status"] == "ok" and leg["status"] == "ok":
            r = float(own["ms_min"]) / float(leg["ms_min"])
            ratios.append(r)
            print(
                "[%d/%d] %s: %s %.5f ms (%s) vs legacy %s %.5f ms  x%.3f"
                % (
                    pi + 1,
                    len(specs),
                    pstr,
                    sel["cfg"],
                    float(own["ms_min"]),
                    sel["src"],
                    sel["legacy"],
                    float(leg["ms_min"]),
                    r,
                )
            )
        else:
            print(
                "[%d/%d] %s: %s %s, legacy %s %s"
                % (
                    pi + 1,
                    len(specs),
                    pstr,
                    sel["cfg"],
                    own["ms_min"] or own["status"],
                    sel["legacy"],
                    leg["ms_min"] or leg["status"],
                )
            )
    if ratios:
        print(
            "%d problems compared: selected/legacy geomean %.3f, faster by"
            " >5%%: %d, slower by >5%%: %d"
            % (
                len(ratios),
                math.exp(sum(map(math.log, ratios)) / len(ratios)),
                sum(r < 0.95 for r in ratios),
                sum(r > 1.05 for r in ratios),
            )
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--benchdnn", required=True, help="path to benchdnn")
    ap.add_argument(
        "--out", help="CSV to append rows to (required for a sweep)"
    )
    ap.add_argument(
        "--lib",
        default="",
        help=(
            "oneDNN library for the model API (default: $DNNL_LIB, else "
            "build/src under the repo)"
        ),
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help=(
            "time the library's selection against the hand-tuned "
            "config for each problem, no sweep"
        ),
    )
    ap.add_argument(
        "--problems",
        action="append",
        default=[],
        help="file with one problem spec per line (repeatable)",
    )
    ap.add_argument(
        "--problem",
        action="append",
        default=[],
        help="inline problem spec (repeatable)",
    )
    ap.add_argument(
        "--design",
        type=int,
        default=0,
        help=(
            "draw N random problems over head sizes, sequence lengths, "
            "batch, heads and masks"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed for --design and for the --random candidates",
    )
    ap.add_argument("--dtype", default="f16", help="data type for --design")
    ap.add_argument(
        "--mask",
        default="",
        help="mask for every --design problem (default: drawn at random)",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=1 << 30,
        help="largest Q or K/V tensor in a designed problem",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=12,
        help="model-ranked candidates to time per problem",
    )
    ap.add_argument(
        "--random",
        type=int,
        default=8,
        help="additional random candidates per problem",
    )
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help=(
            "config to time on every problem, 8 comma-separated ints "
            "(repeatable)"
        ),
    )
    ap.add_argument(
        "--no-legacy",
        action="store_true",
        help="do not time the hand-tuned (legacy) pick",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="time every valid candidate (ignores --topk/--random)",
    )
    ap.add_argument(
        "--perf-mode",
        default="P",
        help="benchdnn perf mode: P (default) or F (10 ms fast mode)",
    )
    ap.add_argument(
        "--max-ms",
        type=int,
        default=300,
        help=(
            "--max-ms-per-prb for perf mode P; the 10 ms fast "
            "mode leaves the GPU clock unsettled"
        ),
    )
    ap.add_argument(
        "--verify-top",
        type=int,
        default=2,
        help="check the fastest configs for correctness until N pass",
    )
    ap.add_argument(
        "--ref-every",
        type=int,
        default=10,
        help="re-time the first config every N runs to detect drift (0: off)",
    )
    ap.add_argument(
        "--sleep-ms",
        type=int,
        default=200,
        help="pause between timed runs",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="seconds before a benchdnn run is abandoned (status timeout)",
    )
    ap.add_argument(
        "--table-out",
        help=(
            "file to append 'key config' lines to: each problem's fastest "
            "config that passed correctness, for bake_fit.py"
        ),
    )
    ap.add_argument(
        "--table-tag-device",
        action="store_true",
        help=(
            "suffix the table keys with :eu<N> so the lines apply "
            "only to this device (for a second SKU of an arch)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the benchdnn commands without running them",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "print every benchdnn command, and its output when it has no "
            "selection line"
        ),
    )
    args = ap.parse_args()

    specs = []
    for path in args.problems:
        with open(path) as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    specs.append(parse_spec(line))
    for line in args.problem:
        specs.append(parse_spec(line))
    if args.design:
        specs += design(
            args.design, args.seed, args.dtype, args.mask, args.max_bytes
        )
    if not specs:
        ap.error("no problems: use --problems, --problem or --design")
    extra_cfgs = [sm.parse_cfg(c) for c in args.extra]

    runner = Runner(args.benchdnn, args.timeout, args.dry_run, args.verbose)
    if args.compare:
        return compare(specs, runner, args)
    if not args.out:
        ap.error("--out is required for a sweep")
    lib = None if args.dry_run else sm.Library(args.lib or None)
    rng = random.Random(args.seed)

    fields = (
        [
            "problem",
            "key",
            "set",
            "rank_model",
            "cfg",
            "kv",
            "q_tile",
            "sg",
            "slm",
            "grf_pred",
            "grf_actual",
            "cost_us",
            "ms_min",
            "ms_avg",
            "status",
            "impl",
            "drift_pct",
            "ts",
            "hw",
            "prb",
        ]
        + ["hw_" + f for f in sm.HW_FIELDS]
        + ["p_" + f for f in sm.PROBLEM_FIELDS]
    )
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fields)
    if new_file:
        writer.writeheader()
    table_lines = []

    for pi, spec in enumerate(specs):
        pstr = spec_str(spec)
        print("[%d/%d] %s" % (pi + 1, len(specs), pstr))

        # 1. the selection line: key, source, legacy pick, hw and problem
        text = runner.run(benchdnn_args(spec, "I"), {})
        sel = parse_select(text)
        if sel is None:
            if args.dry_run:
                continue
            print(
                "  no fwd_select line; is this a ONEDNN_DEV_MODE build with "
                "the micro SDPA kernel available for this problem? skipping"
            )
            if args.verbose:
                print(text[-2000:])
            continue
        # 2. every valid config with its model estimate
        cands = model_candidates(lib, sel)
        by_cfg = {c["cfg"]: c for c in cands}
        print(
            "  key=%s  %d candidates, selected %s (%s), model pick %s "
            "(%s us est)"
            % (
                sel["key"],
                len(cands),
                sel["cfg"],
                sel["src"],
                cands[0]["cfg"] if cands else "-",
                cands[0]["cost"] if cands else "-",
            )
        )
        legacy_cfg = None
        if not args.no_legacy and sel["legacy"] != "none":
            legacy_cfg = sel["legacy"]
            print("  legacy pick %s" % legacy_cfg)

        # 3. choose what to time
        chosen = []  # (cfg_str, set_name)
        if args.all:
            chosen = [(c["cfg"], "all") for c in cands]
        else:
            for c in cands[: args.topk]:
                chosen.append((c["cfg"], "topk"))
            rest = [c["cfg"] for c in cands[args.topk :]]
            for c in rng.sample(rest, min(args.random, len(rest))):
                chosen.append((c, "random"))
        if legacy_cfg:
            chosen.append((legacy_cfg, "legacy"))
        for c in extra_cfgs:
            chosen.append((sm.cfg_str(c), "extra"))
        seen, uniq = set(), []
        for c, s in chosen:
            if c not in seen:
                seen.add(c)
                uniq.append((c, s))
        chosen = uniq

        hw_row = {"hw_" + k: v for k, v in asdict(sel["hw_obj"]).items()}
        p_row = {"p_" + k: v for k, v in asdict(sel["prb_obj"]).items()}

        # 4. time them
        results = []
        ref_cfg, ref_ms = None, 0.0
        for ci, (cfg, set_name) in enumerate(chosen):
            env = {"SDPA_CONFIG": cfg, "QUANTIZED_SDPA_CONFIG": cfg}
            text = runner.run(
                benchdnn_args(spec, args.perf_mode, args.max_ms), env
            )
            perf = parse_perf(text)
            drift = ""
            if perf["status"] == "ok":
                ms = float(perf["ms_min"])
                if ref_cfg is None:
                    ref_cfg, ref_ms = cfg, ms
                elif args.ref_every > 0 and (ci + 1) % args.ref_every == 0:
                    rtext = runner.run(
                        benchdnn_args(spec, args.perf_mode, args.max_ms),
                        {
                            "SDPA_CONFIG": ref_cfg,
                            "QUANTIZED_SDPA_CONFIG": ref_cfg,
                        },
                    )
                    rperf = parse_perf(rtext)
                    if rperf["status"] == "ok":
                        drift = "%.1f" % (
                            100.0 * (float(rperf["ms_min"]) - ref_ms) / ref_ms
                        )
                        if abs(float(drift)) > 5:
                            print("  WARNING: reference drifted %s%%" % drift)
            cand = by_cfg.get(cfg, {})
            row = dict(
                problem=pstr,
                key=sel["key"],
                set=set_name,
                rank_model=cand.get("rank", ""),
                cfg=cfg,
                kv=cand.get("kv", ""),
                q_tile=cand.get("q", ""),
                sg=cand.get("sg", ""),
                slm=cand.get("slm", ""),
                grf_pred=cand.get("grf", ""),
                grf_actual=perf["grf_actual"],
                cost_us=cand.get("cost", ""),
                ms_min=perf["ms_min"],
                ms_avg=perf["ms_avg"],
                status=perf["status"],
                impl=perf["impl"],
                drift_pct=drift,
                ts="%.0f" % time.time(),
                hw=sel["hw"],
                prb=sel["prb"],
            )
            row.update(hw_row)
            row.update(p_row)
            results.append(row)
            print(
                "  [%2d/%2d] %-24s %-7s rank=%-4s est=%8s us  %s %s"
                % (
                    ci + 1,
                    len(chosen),
                    cfg,
                    set_name,
                    cand.get("rank", "-"),
                    cand.get("cost", "-"),
                    perf["ms_min"] or perf["status"],
                    "ms" if perf["ms_min"] else "",
                )
            )
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

        # 5. verify the fastest ones; a failing config hands its place to the
        # next fastest so the table still gets a verified winner
        ok = [r for r in results if r["status"] == "ok"]
        ok.sort(key=lambda r: float(r["ms_min"]))
        n_verified = 0
        for r in ok[: 2 * args.verify_top + 2]:
            if n_verified >= args.verify_top:
                break
            env = {"SDPA_CONFIG": r["cfg"], "QUANTIZED_SDPA_CONFIG": r["cfg"]}
            text = runner.run(
                benchdnn_args(spec, "C"), env, verbose_flags="debuginfo=4"
            )
            skipped = SKIPPED_RE.search(text or "")
            small = batch1_graph(spec) if skipped else None
            if small:
                # the tile config does not depend on the batch
                text = runner.run(
                    benchdnn_args(small, "C"), env, verbose_flags="debuginfo=4"
                )
            m = PASSED_RE.search(text or "")
            if m and m.group(1) == m.group(2) and int(m.group(1)) > 0:
                r["status"] = "verified"
                n_verified += 1
                if small:
                    print("  verified %s at batch 1" % r["cfg"])
            elif SKIPPED_RE.search(text or ""):
                print("  UNVERIFIED: benchdnn skipped the correctness check")
                break
            elif not args.dry_run:
                r["status"] = "mismatch"
                print("  MISMATCH: %s fails correctness" % r["cfg"])

        for r in results:
            writer.writerow(r)
        out_f.flush()

        if ok and float(ok[0]["ms_min"]) > 0:
            best = next((r for r in ok if r["status"] == "verified"), ok[0])
            legacy = next(
                (
                    r
                    for r in results
                    if r["set"] == "legacy"
                    and r["status"] in ("ok", "verified")
                ),
                None,
            )
            model = next(
                (
                    r
                    for r in results
                    if r["rank_model"] == "0"
                    and r["status"] in ("ok", "verified")
                ),
                None,
            )
            msg = "  best %s %.4f ms" % (best["cfg"], float(best["ms_min"]))
            if legacy:
                msg += "  legacy x%.3f" % (
                    float(legacy["ms_min"]) / float(best["ms_min"])
                )
            if model:
                msg += "  model-top1 x%.3f" % (
                    float(model["ms_min"]) / float(best["ms_min"])
                )
            print(msg)
            if best["status"] == "verified":
                key = sel["key"]
                if args.table_tag_device:
                    key += ":eu%d" % sel["hw_obj"].eu_count
                table_lines.append("%s %s" % (key, best["cfg"]))

    out_f.close()
    if args.table_out and table_lines:
        with open(args.table_out, "a") as f:
            for line in table_lines:
                f.write(line + "\n")
        print("wrote %d table lines to %s" % (len(table_lines), args.table_out))
    print("rows appended to %s" % args.out)


if __name__ == "__main__":
    main()
