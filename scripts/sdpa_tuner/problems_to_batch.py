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
"""Write a benchdnn batch file for the --sdpa lines of problems files.

Part of scripts/sdpa_tuner; see README.md there. Use it to run the same
problems on two builds, before and after a change or on two branches, and
compare the logs with compare_perf.py. The batch goes to stdout; graph=
lines are skipped, run those through the --graph driver separately.

    python scripts/sdpa_tuner/problems_to_batch.py \\
        scripts/sdpa_tuner/problems/kpi.txt > kpi.batch
    benchdnn --sdpa --engine=gpu --mode=P --batch=kpi.batch > after.log
    python scripts/sdpa_tuner/compare_perf.py before.log after.log

To compare the table against the hand-tuned configs within one build, use
tune_fwd_config.py --compare instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_fwd_config import benchdnn_args, parse_spec  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    n = 0
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                spec = parse_spec(line)
                if spec["graph"]:
                    continue
                # drop the driver, engine and mode: the batch inherits them
                args = benchdnn_args(spec, "P")[3:]
                print("--reset " + " ".join(args))
                n += 1
    print("%d problems" % n, file=sys.stderr)


if __name__ == "__main__":
    main()
