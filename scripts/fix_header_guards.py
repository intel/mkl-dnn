#! /usr/bin/env python3
################################################################################
# Copyright 2025 Intel Corporation
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
################################################################################

import enum
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, fields
from functools import total_ordering
from typing import Dict, Iterable, List, Optional

# Check these directories...
SOURCE_DIRECTORIES: Iterable[str] = "src", "include"
# ... but not these directories
IGNORED_DIRECTORIES: Iterable[str] = ()


class LicenseState(enum.Enum):
    NOT_SEEN = 0
    INSIDE = 1
    SEEN = 2


@total_ordering
class FileStatus(enum.Enum):
    # ANSI color codes
    OK = 32  # green
    FAIL = 31  # red
    WARN = 33  # yellow
    SKIP = 37  # white

    def color(self, buffer):
        fmtd = self.name.center(4, " ")
        if buffer.isatty():
            fmtd = f"\033[{self.value};1m{fmtd}\033[0m"  # ]]
        return fmtd

    def __lt__(self, other):
        if other is FileStatus.FAIL:
            return self is not FileStatus.FAIL
        if self is FileStatus.FAIL:
            return False
        if other is FileStatus.WARN:
            return self is not FileStatus.WARN
        if self is FileStatus.WARN:
            return False
        return False

    def __eq__(self, other):
        return self is other


@dataclass
class Options:
    inplace: bool = False
    closing_comment: bool = False
    verbose: bool = False
    pedantic: bool = False


class Status:
    def __init__(self):
        self.status = FileStatus.OK
        self.messages = []

    def add(self, status: FileStatus, msg: str):
        if status > self.status:
            self.status = status
        self.messages.append(msg)

    def __bool__(self):
        return self.status is FileStatus.OK or self.status is FileStatus.SKIP

    def color(self, *args, **kwargs):
        return self.status.color(*args, **kwargs)

    def __str__(self):
        return "; ".join(self.messages)

    def ok(self, msg: str):
        return self.add(FileStatus.OK, msg)

    def skip(self, msg: str):
        return self.add(FileStatus.SKIP, msg)

    def warn(self, msg: str):
        return self.add(FileStatus.WARN, msg)

    def fail(self, msg: str):
        return self.add(FileStatus.FAIL, msg)


def ignore(path: str, status: Status):
    for ignored_path in IGNORED_DIRECTORIES:
        if path.startswith(ignored_path):
            status.skip(f"Files in {ignored_path} are skipped")
            return True
    return False


def get_file_guard(path):
    if path.startswith("src/gpu/intel/gemmstone/"):
        path = path.replace("gpu/intel/gemmstone", "gemmstone")
    elif path.startswith("src/gpu/intel/microkernels"):
        path = path.replace("intel/", "")
    guard = path
    for c in "/.":
        guard = guard.replace(c, "_")
    return guard.split("_", 1)[1].upper()


@dataclass
class Directive:
    kind: str
    args: Optional[str]
    start: int
    end: int


@dataclass
class Block:
    open: Directive
    close: Directive
    children: List["Block"] = field(default_factory=list)


def ifndef_argument(directive: Directive):
    if directive.kind != "ifndef":
        return None
    if directive.args is None:
        return None
    return directive.args.split(None, 1)[0]


def find_guard_blocks(guard: str, root: List[Block]):
    guard_blocks = []
    guard_pragmas = []
    alternate = None
    if len(root) == 1 and root[0].open.kind == "ifndef":
        alternate = root[0]
    for block in root:
        if block.open.kind == "pragma":
            guard_pragmas.append(block)
            continue
        arg = ifndef_argument(block.open)
        if arg == guard:
            guard_blocks.append(block)
            continue
        blocks, pragmas, child = find_guard_blocks(guard, block.children)
        guard_blocks += blocks
        guard_pragmas += pragmas
        if len(root) == 1 and alternate is None:
            alternate = child
    return guard_blocks, guard_pragmas, alternate


def insert_guard(lines: List[str], guard: str, line: int, add_comment: bool):
    close = "#endif"
    if add_comment:
        close += f" // {guard}"
    extra_lines = [close]
    replacement = ["", f"#ifndef {guard}", f"#define {guard}"]
    if lines[line].strip():
        # No blank space after the license
        replacement.append("")
    lines[line:line] = replacement
    if lines[-1].startswith("// vim:"):
        extra_lines += ["", lines[-1]]
        lines = lines[:-1]
    if lines[-1].strip():
        lines.append("")  # Insert blank line before #endif
    lines += extra_lines + [""]  # force EOL


def no_code(lines):
    is_multiline_comment = False
    for *_, line in continuations(lines):
        line = line.strip()
        while line:
            if is_multiline_comment:
                if "*/" not in line:
                    break
                line = line.split("*/", 1)[1].lstrip()
                is_multiline_comment = False
            if line.startswith("/*"):
                is_multiline_comment = True
                continue
            if line.startswith("//"):
                break
            if line.startswith("#"):
                break
            return False
    return True


def insert_define(lines: List[str], guard: str, open: Directive):
    index = open.end + 1
    lines[index:index] = [f"#define {guard}"]
    return Directive("define", guard, index, index)


def replace_guard(
    lines: List[str],
    guard: str,
    block: Block,
    defines: List[Directive],
    closing_comment: bool,
):
    open = block.open
    lines[open.start : open.end + 1] = [f"#ifndef {guard}"]
    for define in defines:
        lines[define.start : define.end + 1] = [f"#define {guard}"]
    close = block.close
    if closing_comment:
        lines[close.start : close.end + 1] = [f"#endif // {guard}"]


def continuations(lines):
    continuation = []
    for index, line in enumerate(lines):
        continuation.append(line)
        if line and line[-1] == "\\":
            continue
        combined = "\n".join(continuation)
        start = index + 1 - len(continuation)
        yield start, index, combined
        continuation = []


def get_relative_path(file: str, status: Status):
    my_name = os.path.basename(__file__)
    fullpath = os.path.realpath(file)
    # os.path.join does not respect the empty first entry on Linux, so  we'll
    # just get rid of it and tack on the root directory later.
    *parts, base = fullpath.split(os.sep)[1:]
    for i, part in enumerate(parts):
        if part not in SOURCE_DIRECTORIES:
            continue
        copy_of_me_parts = os.sep, *parts[:i], "scripts", my_name
        copy_of_me = os.path.abspath(os.path.join(*copy_of_me_parts))
        if os.path.isfile(copy_of_me):
            # For our sanity, make the relpath look Unix-y
            return "/".join(parts[i:] + [base])
    status.skip("Could not find DNNL root directory")
    return None


def fix_file(file: str, options: Options):
    status = Status()
    _, ext = os.path.splitext(file)
    relpath = get_relative_path(file, status)
    if relpath is None:
        return False
    if ignore(relpath, status):
        pass
    elif ext in (".h", ".hpp"):
        guard = get_file_guard(relpath)
        adjust_content(file, guard, status)
        warn_repetitive_filename(status, relpath)
    elif ext in (".cpp", ".c", ".cl", ".cxx", ".hxx"):
        warn_repetitive_filename(status, relpath)
    else:
        return False
    if not status or (options.verbose and str(status)):
        print(f"[{status.color(sys.stdout)}] {file} ({status!s})")
    return status.status is FileStatus.FAIL


def warn_repetitive_filename(status: Status, path: str):
    dirname = os.path.dirname(path)
    path_parts = dirname.split("/")
    basename, ext = os.path.splitext(os.path.basename(path))
    name_parts = basename.split("_")

    def last_index_of(haystack: List[str], *needles: str):
        for index, needle in enumerate(haystack[1:][::-1]):
            if needle not in needles:
                continue
            return -index
        return 0

    path_parts = path_parts[last_index_of(path_parts, "src", "include") :]

    common_parts = set(path_parts) & set(name_parts)
    # Allow the external API headers to keep their names:
    #   include/oneapi/dnnl/dnnl_xxx.h"
    if dirname.startswith("include/"):
        common_parts -= {"dnnl"}
    if options.pedantic and common_parts:
        new_name_parts = []
        for part in name_parts:
            if part in common_parts:
                continue
            new_name_parts.append(part)
        new_path = dirname + "/" + "_".join(new_name_parts) + ext
        message = f"consider renaming {path}"
        if new_name_parts:
            message += f" to {new_path}"
        elif path_parts[-1] in common_parts:
            message += f" to {dirname}{ext}"
        status.warn(message)


def adjust_content(file: str, guard: str, status: Status):
    with open(file) as fd:
        lines = fd.read().splitlines()
    state = LicenseState.NOT_SEEN
    license_ends = 0
    if_stack: List[Directive] = []
    blocks: List[List[Block]] = [[]]
    defines: Dict[str, List[Directive]] = defaultdict(list)
    offset = 0  # Line number correction after removing #pragma once
    for start, end, raw_line in continuations(lines):
        line = raw_line.replace("\\\n", "").strip()
        if line.startswith("/*") and state == LicenseState.NOT_SEEN:
            state = LicenseState.INSIDE
            continue
        elif not line.startswith("*") and state == LicenseState.INSIDE:
            state = LicenseState.SEEN
            license_ends = end
        if not line.startswith("#"):
            continue
        rest = line[1:]
        kind, *rest = line[1:].split(None, 1)
        args = rest[0] if rest else None
        directive = Directive(kind, args, start - offset, end - offset)

        if kind == "endif":
            try:
                if_directive = if_stack.pop()
                children = blocks.pop()
            except IndexError:
                return status.fail("mismatched #ifs/#endifs")
            block = Block(if_directive, directive, children)
            blocks[-1].append(block)
        elif kind == "pragma":
            if args is None or args.strip() != "once":
                continue
            block = Block(directive, directive, [])
            blocks[-1].append(block)
            offset += end - start + 1
        elif kind in ("if", "ifdef", "ifndef"):
            blocks.append([])
            if_stack.append(directive)
        elif kind == "define":
            if args is None:
                continue
            name = args.split(None, 1)[0]
            defines[name].append(directive)

    if len(blocks) != 1:
        return status.fail("mismatched #ifs/#endifs")

    guards, pragmas, root = find_guard_blocks(guard, blocks[0])
    if pragmas and not options.inplace:
        status.fail("uses #pragma once")
    for block in pragmas:
        # remove all instances of #pragma once
        # line numbers have already been corrected
        lines[block.open.start : block.close.end + 1] = []

    if len(guards) == 1:
        root = guards[0]
    elif len(guards) > 1:
        return status.fail("too many guards")

    if root is None:
        start = license_ends
        for block in blocks[0]:
            if block.open.start < start:
                continue
            section = lines[start : block.open.start]
            start = block.close.end
            if no_code(section):
                continue
            break
        else:
            if no_code(lines[start:]):
                if not status:
                    return status
                return status.ok("no content")
        if not options.inplace:
            return status.fail("missing guard")
        insert_guard(lines, guard, license_ends, options.closing_comment)
        message = "added missing guard"
    elif root.open.args is not None:
        old_guard = ifndef_argument(root.open)
        if old_guard is None:
            return status.fail("broken guard")
        message = "found correct guard"
        if old_guard not in defines:
            define = insert_define(lines, old_guard, root.open)
            defines[old_guard].append(define)
            if not options.inplace:
                return status.fail("missing define")
            message = "added missing define"
        replace_endif = False
        if root.close.args is not None:
            comment_guard: Optional[str] = None
            arg = root.close.args.rstrip()
            if arg.startswith("//"):
                comment_guard = arg[2:].lstrip()
            elif arg.startswith("/*") and arg.endswith("*/"):
                comment_guard = arg[2:-2].strip()
            if (
                comment_guard is not None
                and comment_guard != guard
                and comment_guard.upper() != "HEADER GUARD"
            ):
                if not options.inplace:
                    status.warn(f"mismatched #endif comment: {comment_guard}")
                message = "fixed incorrect #endif comment"
                replace_endif = True
        elif options.closing_comment:
            replace_endif = True
        if old_guard != guard or replace_endif:
            replace_guard(lines, guard, root, defines[old_guard], replace_endif)
            if not options.inplace:
                if guard != old_guard:
                    fail_message = f"expected guard {guard}, got {old_guard}"
                    status.fail(fail_message)
                if replace_endif:
                    status.warn("wrong or missing #endif comment")
                return status
            if old_guard != guard:
                message = f"fixed incorrect guard {old_guard}"
    else:
        return status.fail("broken top-level guard")
    if options.inplace:
        if pragmas:
            message = "replaced #pragma once"
        with open(file, "w") as fd:
            fd.write("\n".join(lines).rstrip() + "\n")  # force EOL
    if not status:
        return status
    return status.ok(message)


def find_files(basepath, options):
    exit_code = 0
    if os.path.isfile(basepath):
        return fix_file(basepath, options)
    if not os.path.isdir(basepath):
        return exit_code
    for dir, _, filenames in os.walk(basepath):
        for filename in filenames:
            exit_code |= fix_file(os.path.join(dir, filename), options)
    return exit_code


def print_help(prog: str):
    print(f"""usage: {prog} [OPTIONS] files...

description:
    Checks the files (or directories) given for correct header guards in each
    .hpp or .h file.

options:
    --verbose, -v
          print passing in addition to failing files
    --inplace, -i
          modify files in-place
    --closing-comment, -c
          add a comment with the guard name after #endif
    --pedantic, -p
          warn about repetitive parts in file names
    --help, -h
          print this help and exit""")


def to_long_name(field: str):
    field = field.replace("_", "-")
    return f"--{field}"


def to_short_name(field: str):
    return f"-{field[0]}"


if __name__ == "__main__":
    exit_code = 0
    options = Options()
    args = sys.argv[1:]

    while args:
        if args[0].startswith("-h") or args[0] == "--help":
            print_help(sys.argv[0])
            sys.exit(0)
        for f in fields(options):
            long_name = to_long_name(f.name)
            short_name = to_short_name(f.name)
            if args[0].startswith(short_name):
                setattr(options, f.name, True)
                if len(args[0]) > 2:
                    args[0] = f"-{args[0][2:]}"
                else:
                    args.pop(0)
                break
            elif args[0] == long_name:
                setattr(options, f.name, True)
                args.pop(0)
                break
        else:
            break

    for location in args:
        exit_code |= find_files(location, options)
    sys.exit(exit_code)
