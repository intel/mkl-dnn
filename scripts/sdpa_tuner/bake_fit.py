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
"""Write a fit and table lines into select.cpp.

Part of scripts/sdpa_tuner; README.md there shows where it fits in the
workflow. Either option may be used alone, e.g. --table only when tuning a
workload without refitting:

    python scripts/sdpa_tuner/bake_fit.py --arch xe_hpg --device A750 \\
        --fit fit_a750.txt --table table_a750.txt --table fixes_a750.txt
    clang-format -i src/gpu/intel/sdpa/select.cpp

--fit     output of fit_fwd_model.py --fit: its "// select.cpp seed_coefs()"
          block replaces the assignments in the --arch case of
          seed_coefs(), and its problem and config counts, cross-validated
          regret and in-sample regret go into the comment above them
--arch    the seed_coefs() case to write, e.g. xe_hpg; needed with --fit
--device  device name for that comment (default: --arch)
--unfix   a coefficient the fit released from its default-fixed seed, as
          passed to fit_fwd_model.py (repeatable); default-fixed
          coefficients are not written otherwise
--table   "key config" lines from tune_fwd_config.py --table-out or
          resolve_collisions.py --out, merged into fwd_table_data[];
          repeatable, later files and lines win on equal keys, existing
          entries included
--select  path of select.cpp (default: the one in this repository)

Run clang-format on select.cpp afterwards.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_fwd_model import DEFAULT_FIXED  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SELECT = os.path.normpath(
    os.path.join(HERE, "..", "..", "src", "gpu", "intel", "sdpa", "select.cpp")
)


def fmt(v):
    v = float(v)
    if v == 0:
        return "0.0"
    s = "%.4g" % v
    if "e" in s:
        s = "%.6g" % v
    if "." not in s and "e" not in s:
        s += ".0" if abs(v) < 1e15 else ""
    return s


def bake_coefs(src, arch, device, fit_text, unfix=()):
    fixed = set(DEFAULT_FIXED) - set(unfix)
    block = fit_text.split("// select.cpp seed_coefs()", 1)
    if len(block) < 2:
        sys.exit("no '// select.cpp seed_coefs()' block in the fit output")
    coefs = dict(re.findall(r"^\s*m\.(\w+) = ([-+0-9.e]+);", block[1], re.M))
    rows = re.search(r"^(\d+) rows, (\d+) problems", fit_text, re.M)
    cv = re.search(
        r"^cross-validated over \d+ problems: model regret geomean "
        r"x([\d.]+).*legacy geomean x([\d.]+)",
        fit_text,
        re.M,
    )
    fit = re.search(
        r"^fitted/train: \d+ problems, regret geomean x([\d.]+)"
        r".*legacy geomean x([\d.]+)",
        fit_text,
        re.M,
    )
    if not (rows and fit):
        sys.exit("fit output lacks the row count or the fitted/train line")

    case = "        case compute::gpu_arch_t::%s:" % arch
    start = src.find(case)
    if start < 0:
        sys.exit("no case for %s in seed_coefs()" % arch)
    end = src.index("            break;", start)
    body = src[start + len(case) + 1 : end]

    if cv:
        comment = (
            "            // Fitted on %s: %s problems, %s configs, 4-fold CV "
            "regret\n            // %s (legacy table %s), in-sample %s\n"
            % (
                device,
                rows.group(2),
                rows.group(1),
                cv.group(1),
                cv.group(2),
                fit.group(1),
            )
        )
    else:
        comment = (
            "            // Fitted on %s: %s problems, %s configs, in-sample "
            "regret\n            // %s (legacy table %s)\n"
            % (device, rows.group(2), rows.group(1), fit.group(1), fit.group(2))
        )
    body = re.sub(
        r"            // Fitted on [^\n]*\n            // [^\n]*\n", "", body
    )
    present = set()

    def repl(m):
        name = m.group(1)
        present.add(name)
        if name in coefs and name not in fixed:
            return "            m.%s = %s;" % (name, fmt(coefs[name]))
        return m.group(0)

    body = re.sub(
        r"^            m\.(\w+) = [^;]+;[^\n]*", repl, body, flags=re.M
    )
    extra = [n for n in coefs if n not in present and n not in fixed]
    for n in extra:
        body += "            m.%s = %s;\n" % (n, fmt(coefs[n]))
    body = comment + body.lstrip("\n")
    return (
        src[: start + len(case) + 1] + body + src[end:],
        len(coefs),
        len(extra),
    )


def merge_table(src, files):
    tstart = src.index("const char *const fwd_table_data[] = {")
    tend = src.index("};", tstart)
    table = {}
    old = re.findall(r'"([^"]+)"', src[tstart:tend])
    for line in old:
        k, c = line.split()
        table[k] = c
    added = 0
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2 or len(parts[1].split(",")) != 8:
                    sys.exit("bad table line in %s: %r" % (path, line))
                if table.get(parts[0]) != parts[1]:
                    added += 1
                table[parts[0]] = parts[1]
    lines = "".join('        "%s %s",\n' % (k, table[k]) for k in sorted(table))
    src = (
        src[:tstart]
        + "const char *const fwd_table_data[] = {\n"
        + lines
        + src[tend:]
    )
    return src, len(old), len(table), added


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--select", default=DEFAULT_SELECT, help="select.cpp")
    ap.add_argument("--arch", help="architecture case to bake, e.g. xe_hpc")
    ap.add_argument("--device", default="", help="device name for the comment")
    ap.add_argument("--fit", help="fit_fwd_model.py --fit output")
    ap.add_argument(
        "--unfix",
        action="append",
        default=[],
        help="default-fixed coefficient the fit released",
    )
    ap.add_argument(
        "--table",
        action="append",
        default=[],
        help="table file to merge (repeatable)",
    )
    args = ap.parse_args()
    if not args.fit and not args.table:
        ap.error("nothing to do: give --fit and/or --table")
    if args.fit and not args.arch:
        ap.error("--fit needs --arch")

    with open(args.select, encoding="utf-8") as f:
        src = f.read()
    if args.fit:
        with open(args.fit) as f:
            fit_text = f.read()
        src, n, extra = bake_coefs(
            src, args.arch, args.device or args.arch, fit_text, args.unfix
        )
        print(
            "baked %d coefficients into the %s case (%d added)"
            % (n, args.arch, extra)
        )
    if args.table:
        src, old, new, added = merge_table(src, args.table)
        print(
            "table: %d lines before, %d after, %d added or changed"
            % (old, new, added)
        )
    with open(args.select, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("wrote %s; run clang-format on it" % args.select)


if __name__ == "__main__":
    main()
