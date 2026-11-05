/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef UTILS_PARSER_HPP
#define UTILS_PARSER_HPP

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

#include "dnn_types.hpp"
#include "dnnl_debug.hpp"
#include "utils/dims.hpp"
#include "utils/execution_mode.hpp"
#include "utils/impl_filter.hpp"
#include "utils/settings.hpp"
#include "utils/stringstream.hpp"

#include "tests/thread_context.hpp"

namespace parser {

extern bool last_parsed_is_problem;
extern stringstream_t help_ss;

namespace utils {

std::string get_pattern(const std::string &option_name, bool with_args = true);

void add_option_to_help(const std::string &option,
        const std::string &help_message, bool with_args = true);

int64_t stoll_safe(const std::string &s);

float stof_safe(const std::string &s);

// Checks if a `string` consists of only digits. Useful for mask_input parsing.
bool has_only_digits(const std::string &s);

// `option_str` is a string in a format `--option-name=`.
inline bool option_matched(const std::string &option_str, const char *str) {
    // [str, str + option_str.size()) must be a valid range.
    return strlen(str) >= option_str.size()
            && option_str.find(str, 0, option_str.size()) != std::string::npos;
}

std::string lowercase(const std::string &s);

} // namespace utils

namespace parsers {

attr_t::post_ops_t str2attr_post_ops(const std::string &s);

bool str2bool(const std::string &s);

execution_mode_t str2execution_mode(const std::string &s);

dnnl_quantization_mode_t str2quantization_mode(const std::string &s);

} // namespace parsers

// `parse_vector_str` is a heart parser routine which splits input string `str`
// into "chunks" separated by `delimiter` and redirect a chunk into
// `process_func` for further parsing.
//
// The content of `vec` output vector is flushed at the start of routine and
// is updated with parsed objects from `process_func`.
//
// If `str` is empty, the `def` object is inserted into `vec` instead. This
// rule doesn't apply for empty chunks - `process_func` should be responsible
// to parse empty objects and return a proper output or throw an error.
//
// When `allow_empty` is set to `false`, a chunk passed to `process_func` can't
// be empty. It saves from potential undesired user inputs in certain cases.
//
// Returns `true` if parsing and insertion completed successfully.
//
// Note: `T` type represents a vector: `T = std::vector<U>` - since `vec` is
// treated as vector object. `process_func` should return an object of type `U`
// in this notation.
template <typename T, typename F>
bool parse_vector_str(T &vec, const T &def, F process_func,
        const std::string &str, char delimiter = ',', bool allow_empty = true) {
    if (str.empty()) return vec = def, true;

    vec.clear();
    int idx = 1;
    for (size_t pos_st = 0, pos_en = str.find_first_of(delimiter, pos_st); true;
            pos_st = pos_en + 1,
                pos_en = str.find_first_of(delimiter, pos_st)) {
        if (!allow_empty && str.size() == pos_st) {
            BENCHDNN_PRINT(0, "%s %d %s \'%s\'\n", "Error: parsed entry", idx,
                    "is not expected to be empty. Given input:", str.c_str());
            SAFE_V(FAIL);
        }
        //NOLINTBEGIN(readability-redundant-string-cstr)
        // parser.hpp: error: no viable conversion from 'string' to 'const char *'
        // note: in instantiation of function template specialization
        // 'parser::parse_vector_option<std::vector<unsigned int>, unsigned int (*)(const char *)>'
        // parse_vector_option(s.flags, def.flags, str2flags, argv[0],
        // ^
        // TODO: move all functions to std::string input type.
        vec.push_back(
                process_func(str.substr(pos_st, pos_en - pos_st).c_str()));
        //NOLINTEND(readability-redundant-string-cstr)
        if (pos_en == std::string::npos) break;
        idx++;
    }
    return true;
}

template <typename T, typename F>
bool parse_multivector_str(std::vector<T> &vec, const std::vector<T> &def,
        F process_func, const std::string &str, char vector_delim = ',',
        char element_delim = ':', bool allow_empty = true) {
    auto process_subword = [&](const char *word) {
        T v, empty_def_v; // defualt value is not expected to be set here
        // parse vector elements separated by @p element_delim
        parse_vector_str(v, empty_def_v, process_func, word, element_delim);
        return v;
    };

    // parse full vector separated by @p vector_delim
    return parse_vector_str(
            vec, def, process_subword, str, vector_delim, allow_empty);
}

template <typename T, typename F>
bool parse_vector_option(T &vec, const T &def, F process_func, const char *str,
        const std::string &option_name, const std::string &help_message = "") {
    utils::add_option_to_help(option_name, help_message);
    const std::string pattern = utils::get_pattern(option_name);
    if (!utils::option_matched(pattern, str)) return false;
    return parse_vector_str(vec, def, process_func, str + pattern.size());
}

template <typename T, typename F>
bool parse_multivector_option(std::vector<T> &vec, const std::vector<T> &def,
        F process_func, const char *str, const std::string &option_name,
        const std::string &help_message = "", char vector_delim = ',',
        char element_delim = ':') {
    utils::add_option_to_help(option_name, help_message);
    const std::string pattern = utils::get_pattern(option_name);
    if (!utils::option_matched(pattern, str)) return false;
    return parse_multivector_str(vec, def, process_func, str + pattern.size(),
            vector_delim, element_delim);
}

template <typename T, typename F>
bool parse_single_value_option(T &val, const T &def_val, F process_func,
        const char *str, const std::string &option_name,
        const std::string &help_message = "") {
    utils::add_option_to_help(option_name, help_message);
    const std::string pattern = utils::get_pattern(option_name);
    if (!utils::option_matched(pattern, str)) return false;
    str = str + pattern.size();
    if (*str == '\0') return val = def_val, true;
    return val = process_func(str), true;
}

template <typename T, typename F>
bool parse_cfg(T &vec, const T &def, F process_func, const char *str,
        const std::string &option_name = "cfg") {
    static const std::string help
            = "CFG    (Default: `f32`)\n    Specifies data types `CFG` for "
              "source, weights (if supported) and destination of operation.\n  "
              "  `CFG` values vary from driver to driver.\n";
    return parse_vector_option(vec, def, process_func, str, option_name, help);
}

template <typename T, typename F>
bool parse_alg(T &vec, const T &def, F process_func, const char *str,
        const std::string &option_name = "alg") {
    static const std::string help
            = "ALG    (Default: depends on driver)\n    Specifies operation "
              "algorithm `ALG`.\n    `ALG` values vary from driver to "
              "driver.\n";
    return parse_vector_option(vec, def, process_func, str, option_name, help);
}

template <typename T>
bool parse_subattr(std::vector<T> &vec, const char *str,
        const std::string &option_name, const std::string &help_message = "") {
    std::vector<T> def {T()};
    auto parse_subattr_func = [](const std::string &s) {
        T v;
        auto st = v.from_str(s);
        if (st != OK) {
            BENCHDNN_PRINT(
                    0, "Error: failed to parse input: \'%s\'\n", s.c_str());
            SAFE_V(FAIL);
        }
        return v;
    };
    return parse_vector_option(
            vec, def, parse_subattr_func, str, option_name, help_message);
}

template <typename S>
bool parse_reset(S &settings, const char *str,
        const std::string &option_name = "reset") {
    static const std::string help
            = "\n    Instructs the driver to reset driver specific options to "
              "their default values.\n    Neither global options nor "
              "`--perf-template` option would be reset.";
    utils::add_option_to_help(option_name, help, false);

    const std::string pattern = utils::get_pattern(option_name, false);
    if (!utils::option_matched(pattern, str)) return false;
    settings.reset();
    return true;
}

// vector types
bool parse_dir(std::vector<dir_t> &dir, const std::vector<dir_t> &def_dir,
        const char *str, const std::string &option_name = "dir");

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name = "dt");

bool parse_multi_dt(std::vector<std::vector<dnnl_data_type_t>> &dt,
        const std::vector<std::vector<dnnl_data_type_t>> &def_dt,
        const char *str, const std::string &option_name = "sdt");

bool parse_tag(std::vector<std::string> &tag,
        const std::vector<std::string> &def_tag, const char *str,
        const std::string &option_name = "tag");

bool parse_encoding(std::vector<sparse_options_t> &sparse_options,
        const char *str, const std::string &option_name = "encoding");

bool parse_grouped(std::vector<sparse_options_t> &sparse_options,
        const char *str, const std::string &option_name = "grouped");

bool parse_multi_tag(std::vector<std::vector<std::string>> &tag,
        const std::vector<std::vector<std::string>> &def_tag, const char *str,
        const std::string &option_name = "stag");

bool parse_mb(std::vector<int64_t> &mb, const std::vector<int64_t> &def_mb,
        const char *str, const std::string &option_name = "mb");

// This is a parsing method for all attributes. It doesn't take the option name
// unlike other methods. It calls every possible parse call to a dedicated
// attribute and checks if current parse token belongs to any of those.
bool parse_attributes(base_settings_t &settings,
        const base_settings_t &def_settings, const char *str);

bool parse_ctx_init(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str);
bool parse_ctx_exe(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str);
bool parse_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name = "impl");
bool parse_skip_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name = "skip-impl");

bool parse_axis(std::vector<int> &axis, const std::vector<int> &def_axis,
        const char *str, const std::string &option_name = "axis");

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name = "match");

bool parse_inplace(std::vector<bool> &inplace,
        const std::vector<bool> &def_inplace, const char *str,
        const std::string &option_name = "inplace");

bool parse_skip_nonlinear(std::vector<bool> &skip,
        const std::vector<bool> &def_skip, const char *str,
        const std::string &option_name = "skip-nonlinear");

bool parse_strides(std::vector<vdims_t> &strides,
        const std::vector<vdims_t> &def_strides, const char *str,
        const std::string &option_name = "strides");

bool parse_trivial_strides(std::vector<bool> &ts,
        const std::vector<bool> &def_ts, const char *str,
        const std::string &option_name = "trivial-strides");

bool parse_scale_policy(std::vector<policy_t> &policy,
        const std::vector<policy_t> &def_policy, const char *str,
        const std::string &option_name = "scaling");

// plain types
bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name = "perf-template");

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name = "batch");

bool parse_help(const char *str, const std::string &option_name = "help");
bool parse_main_help(const char *str, const std::string &option_name = "help");

// prb_dims_t type
// `prb_vdims_t` type is supposed to run on 2+ tensors. However, in rare cases
// like concat, the library allows a single input. To run a single input, it's
// now a user's responsibility to define a minimum number of inputs for the
// driver with `min_inputs` parameter.
void parse_prb_vdims(
        prb_vdims_t &prb_vdims, const std::string &str, size_t min_inputs = 2);
void parse_prb_dims(prb_dims_t &prb_dims, const std::string &str);

// service functions
bool parse_bench_settings(const char *str);

template <typename S>
bool parse_driver_shared_settings(S &s, const S &def, const char *str) {
    return parse_attributes(s, def, str)
            || parse_ctx_init(s.ctx_init, def.ctx_init, str)
            || parse_ctx_exe(s.ctx_exe, def.ctx_exe, str)
            || parse_test_pattern_match(s.pattern, str)
            || parse_impl(s.impl_filter, def.impl_filter, str)
            || parse_skip_impl(s.impl_filter, def.impl_filter, str)
            || parse_perf_template(s.perf_template,
                    base_settings_t::perf_template_def, s.perf_template_csv(),
                    str)
            || parse_reset(s, str) || parse_help(str);
}

void catch_unknown_options(const char *str);

int parse_last_argument();

// Function returns a substring of a given string @p `s`, using @p `start_pos`
// to start a search from this index in string and @p `delim` as a stop symbol
// and sets a @p `start_pos` to the next symbol after `delim` or to `npos`.
// `allow_dangling` skips a check for dangling symbol at the end of the string
// as some inputs ending on a `delim` in rare cases are legit.
// E.g. 1) s=apple:juice, start_pos=0, delim=':'
//         get_substr -> apple && start_pos -> 6
//      2) s=apple:juice, start_pos=6, delim=':'
//         get_substr -> juice && start_pos -> npos
//      3) s=apple:juice, start_pos=0, delim=';'
//         get_substr -> apple:juice && start_pos -> npos
std::string get_substr(const std::string &s, size_t &start_pos, char delim,
        bool allow_dangling = false);

} // namespace parser

#endif
