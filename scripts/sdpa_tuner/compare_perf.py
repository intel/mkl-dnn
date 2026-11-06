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
"""Compare two benchdnn perf runs problem by problem.

Part of scripts/sdpa_tuner; see README.md there. Typically compares the
same batch (problems_to_batch.py) run on two builds:

    python scripts/sdpa_tuner/compare_perf.py before.log after.log [--top=N]
    python scripts/sdpa_tuner/compare_perf.py both.log   # two runs in one log

Each log is benchdnn --mode=P or F output with the default perf template
(perf,engine,impl,name,prb,...,min_time,...). Problems are paired by their
prb string. With one file, the file must hold two runs, each starting with
its own "perf,engine,impl" header line. Prints the geomean of
second/first min_time, how many problems got faster or slower by more
than 5%, and the --top (default 10) largest differences either way.
"""

import math
import re
import sys
from typing import Dict, List, Optional


def parse(path):
    runs: List[Dict[str, float]] = []
    cur: Optional[Dict[str, float]] = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("perf,engine,impl"):
                cur = {}
                runs.append(cur)
                continue
            if not line.startswith("perf,") or cur is None:
                continue
            f_ = line.rstrip("\n").split(",")
            try:
                t = float(f_[7])
            except (IndexError, ValueError):
                continue
            if t > 0:
                cur.setdefault(f_[4], t)
    return [r for r in runs if r]


def short(prb):
    return re.sub(r"--mode=\S+ --engine=\S+ --sdpa ", "", prb)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 10
    for a in sys.argv[1:]:
        if a.startswith("--top="):
            top = int(a.split("=", 1)[1])
    if len(args) == 1:
        runs = parse(args[0])
        if len(runs) < 2:
            sys.exit(
                "%s holds %d run(s); two are needed" % (args[0], len(runs))
            )
        first, second = runs[0], runs[1]
    elif len(args) == 2:
        first, second = parse(args[0])[0], parse(args[1])[0]
    else:
        sys.exit(__doc__)

    common = [p for p in first if p in second]
    if not common:
        sys.exit("no common problems between the two runs")
    ratios = sorted((second[p] / first[p], p) for p in common)
    logs = [math.log(r) for r, _ in ratios]
    print(
        "%d common problems (%d only in first, %d only in second)"
        % (len(common), len(first) - len(common), len(second) - len(common))
    )
    print("geomean second/first: %.3f" % math.exp(sum(logs) / len(logs)))
    print(
        "second faster by >5%%: %d   first faster by >5%%: %d   within 5%%: %d"
        % (
            sum(r < 0.95 for r, _ in ratios),
            sum(r > 1.05 for r, _ in ratios),
            sum(0.95 <= r <= 1.05 for r, _ in ratios),
        )
    )
    print("\nsecond faster (second/first lowest):")
    for r, p in ratios[:top]:
        print(
            "  %.3f  %.5f -> %.5f ms  %s" % (r, first[p], second[p], short(p))
        )
    print("\nfirst faster (second/first highest):")
    for r, p in ratios[-top:][::-1]:
        print(
            "  %.3f  %.5f -> %.5f ms  %s" % (r, first[p], second[p], short(p))
        )


if __name__ == "__main__":
    main()
