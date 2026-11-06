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
"""Score the forward SDPA cost model against sweeps and fit its coefficients.

Part of scripts/sdpa_tuner; README.md there explains the model and the
metrics (Cost model, Fitting metrics, Fitting). Reads the CSVs of
tune_fwd_config.py and evaluates the library's own model (select.cpp,
through the dev-mode C API) on every timed config, so the library must be
a ONEDNN_DEV_MODE build. The rows must come from one architecture, or
--arch picks one.

Report (always): regret = time(config the model ranks first) / time(best
measured config) per problem, the same for the hand-tuned (legacy) pick,
and the rank correlation between estimate and time; summarized on the
"seed:" line ("given:" with --coefs), per problem with -v.

--check-recorded  compare the estimates recorded in the CSV with the
                  loaded library's, to catch a CSV swept with another model
--fit             search new coefficients and print them as a C++ block for
                  seed_coefs(), which bake_fit.py writes; the
                  "fitted/train" line is their in-sample regret
--folds K         with --fit, cross-validate first: K fits, each scored on
                  the problems it did not see; quote this out-of-sample
                  line, not "fitted/train"
--holdout REGEX   leave matching problems out of the fit and score them
                  separately

The fit minimizes a pairwise ranking loss, the mean log regret and a small
log-error term (weights --w-rank, --w-regret, --w-rmse) over
log-coefficients, with SciPy's Nelder-Mead or, without SciPy, a coordinate
search. Some coefficients are held at their seeds (--help lists them);
--unfix releases one, and bake_fit.py needs the same --unfix to write it.

Examples, from the repository root:
    python scripts/sdpa_tuner/fit_fwd_model.py sweep.csv --check-recorded -v
    python scripts/sdpa_tuner/fit_fwd_model.py "sweep_*.csv" --fit \\
        --folds 4 --restarts 1 > fit.txt
"""

import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdpa_model as sm  # noqa: E402

OK_STATUS = ("ok", "verified")
# Set in main()
LIB: sm.Library = None  # type: ignore[assignment]

# Held at their seeds unless --unfix: clock_ghz only rescales the other
# cycle coefficients, grf_k_load drives a step function (the GRF-mode
# estimate) that a continuous search cannot fit, and the rest only act on
# problems a sweep may not cover (FMA, quantized, misaligned, integrated)
DEFAULT_FIXED = (
    "clock_ghz",
    "grf_k_load",
    "fma_cycles_per_flop",
    "unroll_overhead",
    "dequant_cycles_per_elem",
    "mem_bw_integrated",
    "align4_penalty",
    "unaligned_penalty",
)

WEIGHTS = dict(rank=1.0, regret=1.0, rmse=0.1)


class Row:
    __slots__ = (
        "problem",
        "key",
        "set",
        "cfg",
        "ms",
        "recorded",
        "rank",
        "arch",
        "hw",
        "prb",
    )

    def __init__(self, r):
        self.problem = r["problem"]
        self.key = r["key"]
        self.set = r["set"]
        self.cfg = r["cfg"].encode()
        self.ms = float(r["ms_min"])
        self.recorded = float(r["cost_us"]) if r.get("cost_us") else None
        self.rank = int(r["rank_model"]) if r.get("rank_model") else None
        # hw/prb strings as the library printed them; older CSVs only have
        # the expanded columns, rebuild the strings from those
        hw = sm.hw_from_row(
            {k[3:]: v for k, v in r.items() if k.startswith("hw_")}
        )
        self.arch = hw.arch
        self.hw = (r.get("hw") or hw.to_verbose()).encode()
        if r.get("prb"):
            self.prb = r["prb"].encode()
        else:
            p = sm.problem_from_row(
                {k[2:]: v for k, v in r.items() if k.startswith("p_")}
            )
            self.prb = p.to_verbose().encode()


def load(paths, arch):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("status") not in OK_STATUS or not r.get("ms_min"):
                        continue
                    row = Row(r)
                    if arch and row.arch != arch:
                        continue
                    rows.append(row)
    return rows


def group_rows(rows):
    # One shape on two devices of an arch (BMG and LNL) is two problems
    multi = len({r.hw for r in rows}) > 1
    groups: Dict[str, List[Row]] = {}
    for r in rows:
        name = r.problem
        if multi:
            eu = re.search(rb"eu=(\d+)", r.hw)
            name += " @eu" + (eu.group(1).decode() if eu else "?")
        groups.setdefault(name, []).append(r)
    return groups


def estimate(row, vec):
    """Model estimate in us for a row under a ctypes coefficient vector."""
    res = LIB.estimate(row.hw, row.prb, row.cfg, vec)
    return res[0] if res is not None else None


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def evaluate(groups, coefs, verbose=False, weights=WEIGHTS) -> Dict[str, Any]:
    """Per-problem regret of the model's pick; returns a summary dict.

    The objective combines a pairwise ranking hinge (for pairs of timed
    configs of one problem, the faster one should get the lower estimate,
    weighted by the measured gap), the mean log regret of the pick, and a
    small squared-log-error term that keeps the absolute scale meaningful.
    """
    vec = LIB.vector(coefs)
    regrets, legacy_regrets, rhos, sq_log = [], [], [], []
    hinge_sum, hinge_n = 0.0, 0
    lines = []
    for name, rows in sorted(groups.items()):
        est = [(estimate(r, vec), r) for r in rows]
        est = [(e, r) for e, r in est if e is not None]
        if not est:
            continue
        best = min(r.ms for _, r in est)
        pick = min(est, key=lambda t: t[0])[1]
        regret = pick.ms / best
        regrets.append(regret)
        legacy = [r for _, r in est if r.set == "legacy"]
        lr = legacy[0].ms / best if legacy else float("nan")
        if legacy:
            legacy_regrets.append(lr)
        rho = spearman([e for e, _ in est], [r.ms for _, r in est])
        if not math.isnan(rho):
            rhos.append(rho)
        for e, r in est:
            sq_log.append((math.log(e) - math.log(r.ms * 1000.0)) ** 2)
        # Pairwise hinge; subsample pairs on full landscapes to keep the
        # objective cheap (deterministic stride over the pair index)
        n = len(est)
        stride = max(1, (n * n) // 400)
        for idx in range(0, n * n, stride):
            i, j = divmod(idx, n)
            ei, ri = est[i]
            ej, rj = est[j]
            if ri.ms >= rj.ms:
                continue
            gap = math.log(rj.ms / ri.ms)
            hinge_sum += gap * max(0.0, math.log(ei) - math.log(ej))
            hinge_n += 1
        lines.append(
            "%-60s n=%-3d best=%9.4f ms  model x%-6.3f legacy x%-6.3f rho=%5.2f"
            "  pick=%s"
            % (name[:60], len(est), best, regret, lr, rho, pick.cfg.decode())
        )
    if verbose:
        for line in lines:
            print(line)

    def geomean(v):
        return (
            math.exp(sum(math.log(x) for x in v) / len(v))
            if v
            else float("nan")
        )

    rank_loss = hinge_sum / hinge_n if hinge_n else 0.0
    regret_loss = (
        sum(math.log(r) for r in regrets) / len(regrets) if regrets else 0.0
    )
    rmse_loss = sum(sq_log) / len(sq_log) if sq_log else 0.0
    return dict(
        regrets=regrets,
        legacy_regrets=legacy_regrets,
        problems=len(regrets),
        regret_geomean=geomean(regrets),
        regret_worst=max(regrets) if regrets else float("nan"),
        within_5=(
            sum(1 for r in regrets if r <= 1.05) / len(regrets)
            if regrets
            else 0
        ),
        within_15=(
            sum(1 for r in regrets if r <= 1.15) / len(regrets)
            if regrets
            else 0
        ),
        legacy_regret_geomean=geomean(legacy_regrets),
        rho_mean=sum(rhos) / len(rhos) if rhos else float("nan"),
        rmse_log=math.sqrt(rmse_loss),
        rank_loss=rank_loss,
        objective=weights["rank"] * rank_loss
        + weights["regret"] * regret_loss
        + weights["rmse"] * rmse_loss,
    )


def print_summary(tag, s):
    print(
        "%s: %d problems, regret geomean x%.3f worst x%.3f, within 5%%: %.0f%%,"
        " within 15%%: %.0f%%, legacy geomean x%.3f, mean rho %.2f, rank loss"
        " %.4f, log-rmse %.3f, objective %.4f"
        % (
            tag,
            s["problems"],
            s["regret_geomean"],
            s["regret_worst"],
            100 * s["within_5"],
            100 * s["within_15"],
            s["legacy_regret_geomean"],
            s["rho_mean"],
            s["rank_loss"],
            s["rmse_log"],
            s["objective"],
        )
    )


def check_recorded(rows):
    """Recorded sweep estimates vs the loaded library at its seeds."""
    worst, n = 0.0, 0
    seeds = {}
    for r in rows:
        if r.recorded is None:
            continue
        if r.arch not in seeds:
            seeds[r.arch] = LIB.vector(LIB.seed_coefs(r.arch))
        e = estimate(r, seeds[r.arch])
        if e is None:
            print(
                "library rejects a config the sweep timed: %s on %s"
                % (r.cfg.decode(), r.problem)
            )
            continue
        rel = abs(e - r.recorded) / max(r.recorded, 1e-9)
        n += 1
        worst = max(worst, rel)
    print(
        "recorded vs library estimate: %d rows, worst relative difference"
        " %.3e%s"
        % (
            n,
            worst,
            (
                ""
                if worst < 1e-3
                else "  <-- the CSV was produced by a different model or seeds"
            ),
        )
    )
    return worst < 1e-3


def fit(groups, arch, iters, fixed, restarts=0, seed=1, quiet=False, base=None):
    """Fit from base (the library seeds, with --coefs overrides applied)."""
    base = dict(base) if base else LIB.seed_coefs(arch)
    names = [n for n in LIB.coef_names if n not in fixed]
    # Coefficients are searched in log space so they stay positive; a zero
    # seed (e.g. partial_wave_c1) is floored and effectively stays near zero
    x0 = [math.log(max(base[n], 1e-6)) for n in names]

    def coefs_of(x):
        m = dict(base)
        for n, v in zip(names, x):
            m[n] = math.exp(v)
        return m

    evals = [0]

    def objective(x):
        evals[0] += 1
        return evaluate(groups, coefs_of(x))["objective"]

    def search(start):
        try:
            from scipy.optimize import minimize

            res = minimize(
                objective,
                start,
                method="Nelder-Mead",
                options=dict(
                    maxiter=iters, maxfev=iters, xatol=1e-3, fatol=1e-5
                ),
            )
            return list(res.x), res.fun
        except ImportError:
            pass
        x, fx = list(start), objective(start)
        step = 0.5
        it = 0
        while it < iters and step > 1e-3:
            improved = False
            for i in range(len(x)):
                for sgn in (1, -1):
                    trial = list(x)
                    trial[i] += sgn * step
                    ft = objective(trial)
                    it += 1
                    if ft < fx:
                        x, fx, improved = trial, ft, True
                        break
            if not improved:
                step *= 0.5
        return x, fx

    f0 = objective(x0)
    if not quiet:
        print(
            "fitting %d coefficients on %d problems, initial objective %.4f"
            % (len(names), len(groups), f0)
        )
    best_x, best_f = search(x0)
    # The coordinate search is path dependent; restart from perturbed seeds
    # (deterministic) and keep the best training objective
    rng = random.Random(seed)
    for _ in range(restarts):
        start = [v + rng.gauss(0.0, 0.5) for v in x0]
        x, fx = search(start)
        if fx < best_f:
            best_x, best_f = x, fx
    if not quiet:
        print(
            "search: %d evaluations, %d restarts, objective %.4f"
            % (evals[0], restarts, best_f)
        )
    return coefs_of(best_x)


def main():
    global LIB
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("csv", nargs="+", help="tuner CSV files (globs allowed)")
    ap.add_argument(
        "--lib",
        default="",
        help=(
            "oneDNN library to evaluate the model with (default: "
            "$DNNL_LIB, else build/src under the repo)"
        ),
    )
    ap.add_argument(
        "--arch",
        default="",
        help="use only rows of this architecture (needed when CSVs mix them)",
    )
    ap.add_argument(
        "--check-recorded",
        action="store_true",
        help="compare the CSV's recorded estimates with the library's",
    )
    ap.add_argument(
        "--fit",
        action="store_true",
        help="fit the coefficients and print them for bake_fit.py",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="maximum objective evaluations per search",
    )
    ap.add_argument(
        "--fix",
        action="append",
        default=[],
        help="coefficient to hold at its seed during --fit (default: %s)"
        % ", ".join(DEFAULT_FIXED),
    )
    ap.add_argument(
        "--unfix",
        action="append",
        default=[],
        help="release a default-fixed coefficient (repeatable)",
    )
    ap.add_argument(
        "--coefs",
        default="",
        help=(
            'overrides on the library seeds ("name=value,..." '
            "syntax); also the starting point of --fit, so a "
            "term seeded at 0 can be switched on for a fit"
        ),
    )
    ap.add_argument(
        "--w-rank",
        type=float,
        default=WEIGHTS["rank"],
        help="weight of the pairwise ranking loss in the objective",
    )
    ap.add_argument(
        "--w-regret",
        type=float,
        default=WEIGHTS["regret"],
        help="weight of the mean log regret in the objective",
    )
    ap.add_argument(
        "--w-rmse",
        type=float,
        default=WEIGHTS["rmse"],
        help="weight of the log-error term in the objective",
    )
    ap.add_argument(
        "--holdout",
        default="",
        help=(
            "regex over problem names; matching problems are "
            "excluded from --fit and scored separately"
        ),
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=0,
        help=(
            "K-fold cross-validation before the final fit: fit "
            "on K-1 folds of problems, score the held-out fold, "
            "report the out-of-sample regret over all problems"
        ),
    )
    ap.add_argument(
        "--restarts",
        type=int,
        default=0,
        help="extra searches from perturbed starting points; the best is kept",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="per-problem table"
    )
    args = ap.parse_args()

    LIB = sm.Library(args.lib or None)
    rows = load(args.csv, args.arch)
    if not rows:
        sys.exit("no usable rows (status ok/verified) found")
    arches = sorted(set(r.arch for r in rows))
    if len(arches) > 1:
        sys.exit("rows span several architectures %s; pass --arch" % arches)
    arch = arches[0]
    groups = group_rows(rows)
    print(
        "%d rows, %d problems, arch %s, library %s"
        % (len(rows), len(groups), arch, LIB.path)
    )

    if args.check_recorded:
        check_recorded(rows)

    coefs = LIB.seed_coefs(arch)
    for item in args.coefs.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            coefs[k.strip()] = float(v)
    print_summary(
        "seed" if not args.coefs else "given",
        evaluate(groups, coefs, args.verbose),
    )

    train, held = groups, {}
    if args.holdout:
        held = {k: v for k, v in groups.items() if re.search(args.holdout, k)}
        train = {k: v for k, v in groups.items() if k not in held}
        print(
            "holdout: %d problems held out, %d used for fitting"
            % (len(held), len(train))
        )
        print_summary("seed/holdout", evaluate(held, coefs, args.verbose))

    WEIGHTS.update(rank=args.w_rank, regret=args.w_regret, rmse=args.w_rmse)
    fixed = (set(DEFAULT_FIXED) | set(args.fix)) - set(args.unfix)

    if args.fit and args.folds > 1:
        names = sorted(groups)
        folds = [names[i :: args.folds] for i in range(args.folds)]
        oos, oos_legacy = [], []
        for k, fold in enumerate(folds):
            ftrain = {n: groups[n] for n in names if n not in fold}
            fheld = {n: groups[n] for n in fold}
            fitted_k = fit(
                ftrain,
                arch,
                args.iters,
                fixed,
                args.restarts,
                quiet=True,
                base=coefs,
            )
            s = evaluate(fheld, fitted_k, args.verbose)
            oos += s["regrets"]
            oos_legacy += s["legacy_regrets"]
            print_summary("fold %d/%d held out" % (k + 1, args.folds), s)

        def geomean(v):
            return (
                math.exp(sum(math.log(x) for x in v) / len(v))
                if v
                else float("nan")
            )

        print(
            "cross-validated over %d problems: model regret geomean x%.3f "
            "worst x%.3f, within 5%%: %.0f%%, within 15%%: %.0f%%; legacy "
            "geomean x%.3f"
            % (
                len(oos),
                geomean(oos),
                max(oos),
                100.0 * sum(1 for r in oos if r <= 1.05) / len(oos),
                100.0 * sum(1 for r in oos if r <= 1.15) / len(oos),
                geomean(oos_legacy),
            )
        )

    if args.fit:
        fitted = fit(train, arch, args.iters, fixed, args.restarts, base=coefs)
        print_summary("fitted/train", evaluate(train, fitted, args.verbose))
        if held:
            print_summary(
                "fitted/holdout", evaluate(held, fitted, args.verbose)
            )
        print(
            "\n// select.cpp seed_coefs() for %s:\n%s"
            % (arch, sm.coefs_to_cpp(fitted, LIB.coef_names))
        )


if __name__ == "__main__":
    main()
