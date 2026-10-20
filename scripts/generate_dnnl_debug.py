#!/usr/bin/env python
################################################################################
# Copyright 2018 Intel Corporation
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

from __future__ import print_function

import os
import re
import sys
import textwrap
import xml.etree.ElementTree as ET


def template(body, banner):
    return f"""\
{banner}
// DO NOT EDIT, AUTO-GENERATED
// Use this script to update the file: scripts/{os.path.basename(__file__)}

// clang-format off

{body}"""


def header(body):
    return f"""
#ifndef ONEAPI_DNNL_DNNL_DEBUG_H
#define ONEAPI_DNNL_DNNL_DEBUG_H

/// @file
/// Debug capabilities

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#ifdef __cplusplus
extern "C" {{
#endif

{body}
const char DNNL_API *dnnl_runtime2str(unsigned v);
const char DNNL_API *dnnl_fmt_kind2str(dnnl_format_kind_t v);

#ifdef __cplusplus
}}
#endif

#endif
""".lstrip()


def source(body):
    return f"""
#include <assert.h>

#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"

{body}
""".lstrip()


def header_benchdnn(body):
    return f"""
#ifndef DNNL_DEBUG_HPP
#define DNNL_DEBUG_HPP

#include "oneapi/dnnl/dnnl.h"

{body}
/* status */
const char *status2str(dnnl_status_t status);

/* data type */
const char *dt2str(dnnl_data_type_t dt);

/* format */
const char *fmt_tag2str(dnnl_format_tag_t tag);

/* encoding */
const char *sparse_encoding2str(dnnl_sparse_encoding_t encoding);

/* engine kind */
const char *engine_kind2str(dnnl_engine_kind_t kind);

/* scratchpad mode */
const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode);

/* fpmath mode */
const char *fpmath_mode2str(dnnl_fpmath_mode_t mode);

/* accumulation mode */
const char *accumulation_mode2str(dnnl_accumulation_mode_t mode);

/* rounding mode */
const char *rounding_mode2str(dnnl_rounding_mode_t mode);

/* quantization mode */
const char *quantization_mode2str(dnnl_quantization_mode_t mode);

#endif
""".lstrip()


def source_benchdnn(body):
    return f"""
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "oneapi/dnnl/dnnl_debug.h"

#include "dnnl_debug.hpp"

#include "src/common/z_magic.hpp"

{body.rstrip()}

const char *status2str(dnnl_status_t status) {{
    return dnnl_status2str(status);
}}

const char *dt2str(dnnl_data_type_t dt) {{
    return dnnl_dt2str(dt);
}}

const char *fmt_tag2str(dnnl_format_tag_t tag) {{
    return dnnl_fmt_tag2str(tag);
}}

const char *sparse_encoding2str(dnnl_sparse_encoding_t encoding) {{
    return dnnl_sparse_encoding2str(encoding);
}}

const char *engine_kind2str(dnnl_engine_kind_t kind) {{
    return dnnl_engine_kind2str(kind);
}}

const char *scratchpad_mode2str(dnnl_scratchpad_mode_t mode) {{
    return dnnl_scratchpad_mode2str(mode);
}}

const char *fpmath_mode2str(dnnl_fpmath_mode_t mode) {{
    return dnnl_fpmath_mode2str(mode);
}}

const char *accumulation_mode2str(dnnl_accumulation_mode_t mode) {{
    return dnnl_accumulation_mode2str(mode);
}}

const char *rounding_mode2str(dnnl_rounding_mode_t mode) {{
    return dnnl_rounding_mode2str(mode);
}}

const char *quantization_mode2str(dnnl_quantization_mode_t mode) {{
    return dnnl_quantization_mode2str(mode);
}}
""".lstrip()


def maybe_skip(enum):
    return enum in (
        "dnnl_memory_extra_flags_t",
        "dnnl_normalization_flags_t",
        "dnnl_query_t",
        "dnnl_rnn_cell_flags_t",
        "dnnl_stream_flags_t",
        "dnnl_format_kind_t",
    )


def enum_abbrev(enum):
    def_enum = re.sub(r"^dnnl_", "", enum)
    def_enum = re.sub(r"_t$", "", def_enum)
    return {
        "dnnl_data_type_t": "dt",
        "dnnl_format_tag_t": "fmt_tag",
        "dnnl_primitive_kind_t": "prim_kind",
        "dnnl_engine_kind_t": "engine_kind",
    }.get(enum, def_enum)


# Map (enum_name, enum_value) to macro_name for experimental features
experimental_features = {
    # ~~~~~~~enum name~~~~~~    ~enum value~     ~~~~~~~~~~macro name~~~~~~~~~~
    (
        "dnnl_sparse_encoding_t",
        "dnnl_grouped",
    ): "DNNL_EXPERIMENTAL_GROUPED_MEMORY",
}


def sanitize_value(v):
    if "undef" in v:
        return "undef"
    if "any" in v:
        return "any"
    v = v.split("dnnl_fpmath_mode_")[-1]
    v = v.split("dnnl_accumulation_mode_")[-1]
    v = v.split("dnnl_rounding_mode_")[-1]
    v = v.split("dnnl_scratchpad_mode_")[-1]
    v = v.split("dnnl_quantization_mode_")[-1]
    v = v.split("dnnl_")[-1]
    return v


def func_to_str_decl(enum, is_header=False):
    abbrev = enum_abbrev(enum)
    visibility = "DNNL_API " if is_header else ""
    return f"const char {visibility}*dnnl_{abbrev}2str({enum} v)"


def blocks_to_func_body(blocks, indent="    "):
    def is_preprocessor_stmt(block: str):
        return block.strip()[0] == "#"

    def add_indent(line):
        if not line.strip():
            return ""
        return indent + line

    blocks = blocks[:]
    for i, block in enumerate(blocks):
        dedented = textwrap.dedent(block).strip()
        if is_preprocessor_stmt(block):
            blocks[i] = dedented
            continue
        indented = "\n".join(map(add_indent, dedented.split("\n")))
        blocks[i] = indented
    if not blocks or is_preprocessor_stmt(blocks[0]):
        return "\n".join(blocks).strip()
    return indent + "\n".join(blocks).strip()


def func_to_str(enum, values):
    abbrev = enum_abbrev(enum)
    signature = func_to_str_decl(enum)
    func_blocks = []
    for v in values:
        # Add conditional compilation for experimental features
        if (enum, v) in experimental_features:
            func_blocks.append(f"#if {experimental_features[enum, v]}")
            func_blocks.append(f'if (v == {v}) return "{sanitize_value(v)}";')
            func_blocks.append("#endif")
        else:
            func_blocks.append(f'if (v == {v}) return "{sanitize_value(v)}";')
    if enum == "dnnl_primitive_kind_t":
        func_blocks.append(
            'if (v == dnnl::impl::primitive_kind::sdpa) return "sdpa";'
        )
        func_blocks.append("""
if (v == dnnl::impl::primitive_kind::gated_mlp) return "gated_mlp";
            """)
    if enum == "dnnl_alg_kind_t":
        func_blocks.append("""
if (v == dnnl::impl::alg_kind::softmax_accurate_inf_as_zero)
    return "softmax_accurate_inf_as_zero";
            """)
    func_blocks.append(f'assert(!"unknown {abbrev}");')
    func_blocks.append(f'return "unknown {abbrev}";')

    body = blocks_to_func_body(func_blocks)
    return f"{signature} {{\n{body}\n}}\n"


def str_to_func_decl(enum, is_header=False, is_dnnl=True):
    attr = "DNNL_API " if is_header and is_dnnl else ""
    prefix = "dnnl_" if is_dnnl else ""
    abbrev = enum_abbrev(enum)
    return f"{enum} {attr}{prefix}str2{abbrev}(const char *str)"


def str_to_func(enum, values, is_dnnl=True):
    abbrev = enum_abbrev(enum)
    func_blocks = []
    signature = str_to_func_decl(enum, is_dnnl=is_dnnl)
    func_blocks.append("""
#define CASE(_case) do { \\
    if (!strcmp(STRINGIFY(_case), str) \\
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \\
        return CONCAT2(dnnl_, _case); \\
} while (0)
        """)
    special_values = []
    v_undef = None
    for v in values:
        if "last" in v:
            continue
        if "undef" in v:
            v_undef = v
            special_values.append(v)
            continue
        if "any" in v:
            special_values.append(v)
            continue
        # Add conditional compilation for experimental features
        if (enum, v) in experimental_features:
            func_blocks.append(f"#if {experimental_features[enum, v]}")
            func_blocks.append(f"CASE({sanitize_value(v)});")
            func_blocks.append("#endif")
        else:
            func_blocks.append(f"CASE({sanitize_value(v)});")
    func_blocks.append("#undef CASE")
    for v in special_values:
        match = re.search(r"(any|undef)", v)
        if match is None:
            continue
        v_short = match.group()
        func_blocks.append(f"""
if (!strcmp("{v_short}", str) || !strcmp("{v}", str))
    return {v};
            """)
    if enum != "dnnl_format_tag_t":
        func_blocks.append(f"""
printf("Error: {abbrev} `%s` is not supported.\\n", str);
            """)
        func_blocks.append(f'assert(!"unknown {abbrev}");')
    assert isinstance(v_undef, str)
    default = v_undef
    if enum == "dnnl_format_tag_t":
        default = "dnnl_format_tag_last"
    func_blocks.append(f"return {default};")

    body = blocks_to_func_body(func_blocks)
    return f"{signature} {{\n{body}\n}}\n"


def generate(ifile, banners):
    h_body, s_body = "", ""
    h_benchdnn_body, s_benchdnn_body = "", ""
    root = ET.parse(ifile).getroot()
    for v_enum in root.findall("Enumeration"):
        enum = v_enum.attrib["name"]
        if maybe_skip(enum):
            continue
        values = [
            v_value.attrib["name"] for v_value in v_enum.findall("EnumValue")
        ]

        # Drop the deprecated data type here to affect all functions.
        for v in values:
            if v == "dnnl_f4_e3m0":
                values.remove(v)

        h_body += func_to_str_decl(enum, is_header=True) + ";\n"
        s_body += func_to_str(enum, values) + "\n"

        if enum in [
            "dnnl_format_tag_t",
            "dnnl_data_type_t",
            "dnnl_sparse_encoding_t",
        ]:
            h_benchdnn_body += (
                str_to_func_decl(enum, is_header=True, is_dnnl=False) + ";\n"
            )
            s_benchdnn_body += str_to_func(enum, values, is_dnnl=False) + "\n"

    bodies = [
        header(h_body),
        source(s_body),
        header_benchdnn(h_benchdnn_body),
        source_benchdnn(s_benchdnn_body),
    ]
    return [template(b, y) for b, y in zip(bodies, banners)]


def usage():
    print(f"""{sys.argv[0]} types.xml

Generates oneDNN debug header and source files with enum to string mapping.
Input types.xml file can be obtained with CastXML[1]:
$ castxml --castxml-cc-gnu-c clang --castxml-output=1 \\
        -Iinclude -Ibuild/include include/oneapi/dnnl/dnnl_types.h -o types.xml

[1] https://github.com/CastXML/CastXML""")
    sys.exit(1)


for arg in sys.argv:
    if "-help" in arg:
        usage()

script_root = os.path.dirname(os.path.realpath(__file__))

ifile = sys.argv[1] if len(sys.argv) > 1 else usage()

file_paths = (
    f"{script_root}/../include/oneapi/dnnl/dnnl_debug.h",
    f"{script_root}/../src/common/dnnl_debug_autogenerated.cpp",
    f"{script_root}/../tests/benchdnn/dnnl_debug.hpp",
    f"{script_root}/../tests/benchdnn/dnnl_debug_autogenerated.cpp",
)

banners = []
for file_path in file_paths:
    with open(file_path, "r") as f:
        m = re.match(r"^/\*+\n(\*.*\n)+\*+/\n", f.read())
        banners.append("" if m is None else m.group(0))

for file_path, file_body in zip(file_paths, generate(ifile, banners)):
    with open(file_path, "w") as f:
        f.write(file_body)
