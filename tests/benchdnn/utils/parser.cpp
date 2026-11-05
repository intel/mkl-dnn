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

#include <algorithm>
#include <cctype>
#include <map>
#include <unordered_map>

#include "utils/check.hpp"
#include "utils/cold_cache.hpp"
#include "utils/fill.hpp"
#include "utils/parser.hpp"
#include "utils/stream_kind.hpp"
#include "utils/stringstream.hpp"
#include "utils/summary.hpp"

#include "dnnl_common.hpp"

namespace parser {

bool last_parsed_is_problem = false;
stringstream_t help_ss;

static const std::string benchdnn_url
        = "https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn";
static const std::string doc_url = benchdnn_url + "/doc/";

namespace utils {

// Current definition works only through the build system. It can be generalized
// through C++11 `__has_feature` macro, but not every sanitizer has a macro
// to check against.
//
// The function disables `no_ref_memory` modifier for sanitizers testing because
// many legit places in the library can't work with completely overflowed
// values, like int32 zero-point values.
bool has_clang_sanitizers() {
#if defined(DNNL_ENABLED_CLANG_SANITIZER)
    return true;
#else
    return false;
#endif
}

std::string get_pattern(const std::string &option_name, bool with_args) {
    std::string s = std::string("--") + option_name;
    if (with_args) s += "=";
    return s;
}

std::vector<std::string> &get_option_names() {
    static std::vector<std::string> options;
    return options;
}

void add_option_to_help(const std::string &option,
        const std::string &help_message, bool with_args) {
    for (const auto &e : get_option_names())
        if (e == option) return;

    std::string option_str = get_pattern(option, with_args);
    help_ss << option_str << help_message << "\n";
    get_option_names().push_back(option);
}

bool has_only_digits(const std::string &s) {
    if (s.empty()) return false;

    return std::all_of(s.cbegin(), s.cend(), [](int c) {
        assert(c < UINT8_MAX);
        return std::isdigit(c);
    });
}

std::string lowercase(const std::string &s) {
    std::string s_low(s);
    std::transform(s.begin(), s.end(), s_low.begin(), ::tolower);
    return s_low;
}

// Covers all integer parsing routines: `atoi`, `atol, `atoll`, `stoi`, `stol`.
int64_t stoll_safe(const std::string &s) {
    int64_t value = 0;
    try {
        value = std::stoll(s);
        if (std::to_string(value).size() != s.size()) {
            BENCHDNN_PRINT(0, "%s \'%s\'\n",
                    "Error: Parsed value is expected to be an integer number, "
                    "not",
                    s.c_str());
            SAFE_V(FAIL);
        }
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: Parsed value is expected to be an integer number, not",
                s.c_str());
        SAFE_V(FAIL);
    }
    return value;
}

// Covers all float parsing routines: `atof`.
float stof_safe(const std::string &s) {
    float value = 0;
    try {
        value = std::stof(s);
        // Can't compare input with output same way. The only way is to check
        // if input has `e`, `f` and digits. Seems an overkill, let function
        // decide on parsed output.
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: Parsed value is expected to be an floating-point "
                "number, not",
                s.c_str());
        SAFE_V(FAIL);
    }
    return value;
}

// Computes Levenshtein distance between two strings, which indicated how many
// symbols should be appended, replaced, or removed to obtain the input string.
// The implementation follows this article:
// https://en.wikipedia.org/wiki/Levenshtein_distance
//
// The function is designed to be called recursively, decreasing symbols in `in`
// and `opt` by one.
//
// Note: to improve the performance of the routine, an internal cache is used.
// Between different options there would be low similarity, thus, `reset_cache`
// instructs to clear its content (called externally, not during recurse).
size_t compute_distance(const std::string &in, const std::string &opt,
        bool reset_cache = false) {
    if (in.empty()) return opt.size();
    if (opt.empty()) return in.size();
    if (in.front() == opt.front())
        return compute_distance(in.substr(1), opt.substr(1));

    static std::map<std::pair<std::string, std::string>, size_t> cache;
    if (reset_cache) cache.clear();
    std::map<std::pair<std::string, std::string>, size_t>::iterator it {};

    auto cache_shift_in_key = std::make_pair(in.substr(1), opt);
    it = cache.find(cache_shift_in_key);
    size_t shift_in_dist = SIZE_MAX;
    if (it != cache.end()) {
        shift_in_dist = it->second;
    } else {
        shift_in_dist = compute_distance(in.substr(1), opt);
        cache.emplace(cache_shift_in_key, shift_in_dist);
    }

    auto cache_shift_opt_key = std::make_pair(in, opt.substr(1));
    it = cache.find(cache_shift_opt_key);
    size_t shift_opt_dist = SIZE_MAX;
    if (it != cache.end()) {
        shift_opt_dist = it->second;
    } else {
        shift_opt_dist = compute_distance(in, opt.substr(1));
        cache.emplace(cache_shift_opt_key, shift_opt_dist);
    }

    size_t shift_both_dist = compute_distance(in.substr(1), opt.substr(1));
    return std::min(std::min(shift_in_dist, shift_opt_dist), shift_both_dist)
            + 1;
}

// A safe version of str2bool that doesn't exit and sets @parsed to `true` if
// the content was valid, and `false`, otherwise.
bool str2bool_safe(const std::string &str, bool &parsed) {
    auto s = utils::lowercase(str);
    if (s == "1" || s == "true") {
        parsed = true;
        return true;
    } else if (s == "0" || s == "false") {
        parsed = true;
        return false;
    }

    parsed = false;
    return false;
}

} // namespace utils

namespace parsers {

attr_t::post_ops_t str2attr_post_ops(const std::string &s) {
    attr_t::post_ops_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto kind
                = attr_t::post_ops_t::str2kind(get_substr(subs, subs_pos, ':'));
        if (kind == attr_t::post_ops_t::kind_t::KIND_TOTAL) SAFE_V(FAIL);

        v.entry.emplace_back(kind);
        if (subs_pos == std::string::npos) {
            if (kind != attr_t::post_ops_t::kind_t::DW) continue;

            BENCHDNN_PRINT(0, "%s\n",
                    "Error: depthwise post-op entry didn't recognize 'k', 's', "
                    "and 'p' values.");
            SAFE_V(FAIL);
        }

        auto &e = v.entry.back();
        if (e.is_sum_kind()) {
            e.sum.scale = utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;

            auto zp_str = get_substr(subs, subs_pos, ':');
            e.sum.zero_point = utils::stoll_safe(zp_str);
            if (subs_pos == std::string::npos) continue;

            const auto dt_str = get_substr(subs, subs_pos, ':');
            e.sum.dt = str2dt(dt_str.c_str());
            // sum dt, if specified, should be defined
            if (e.sum.dt == dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: sum post-op data type", dt_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
        } else if (e.is_convolution_kind()) {
            // `DW` has input of `dw:kXsYpZ`.
            const auto str_dw_params = get_substr(subs, subs_pos, ':');
            size_t pos = 0, idx = 0;

            pos += idx;
            if (str_dw_params[pos] != 'k') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 'k'.");
                SAFE_V(FAIL);
            }
            // TODO: some safe handling would help here
            e.convolution.kernel = std::stoi(&str_dw_params[++pos], &idx);
            if (e.convolution.kernel <= 0) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: depthwise post-op kernel must be greater "
                        "than 0.");
                SAFE_V(FAIL);
            }

            pos += idx;
            if (str_dw_params[pos] != 's') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 's'.");
                SAFE_V(FAIL);
            }
            e.convolution.stride = std::stoi(&str_dw_params[++pos], &idx);
            if (e.convolution.stride <= 0) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: depthwise post-op stride must be greater "
                        "than 0.");
                SAFE_V(FAIL);
            }

            pos += idx;
            if (str_dw_params[pos] != 'p') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 'p'.");
                SAFE_V(FAIL);
            }
            e.convolution.padding = std::stoi(&str_dw_params[++pos]);

            if (subs_pos == std::string::npos) continue;

            const auto dt_str = get_substr(subs, subs_pos, ':');
            e.convolution.dst_dt = str2dt(dt_str.c_str());
            if (e.convolution.dst_dt == dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op data type", dt_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
            if (subs_pos < std::string::npos) {
                const auto leftover = get_substr(subs, subs_pos, '\0');
                BENCHDNN_PRINT(0,
                        "Error: no more inputs are expected. Provided: "
                        "\'%s\'.\n",
                        leftover.c_str());
                SAFE_V(FAIL);
            }
        } else if (e.is_eltwise_kind()) {
            e.eltwise.alpha = utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;

            e.eltwise.beta = utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
        } else if (e.is_binary_kind()) {

            char src_delim = ':';
            bool has_src2_specs = false;
            std::string src1_subs, src2_subs;

            // placeholder data type for the ternary conditional input
            e.binary.src2_dt = dnnl_s8;

            // For binary algorithms with ternary inputs, specifications can
            // be provided for both binary (src1) and ternary (src2) tensors in
            // the form:
            // --attr-post-ops=BINARY:DT[.S1_MASK_INPUT[.S1_TAG]][:S2_MASK_INPUT[.S2_TAG]].
            // In that case, we check for the ':' delimiter that separates src1
            // and src2 args, split the string for the two tensors and parse
            // them individually.

            if (e.is_binary_kind_with_ternary_op()) {
                src_delim = '.';
                auto src2_pos = subs.find_first_of(':', subs_pos);
                has_src2_specs = (src2_pos != std::string::npos);
                src1_subs = get_substr(subs, subs_pos, ':');
                if (has_src2_specs) src2_subs = get_substr(subs, subs_pos, ':');
            } else {
                src1_subs = subs.substr(subs_pos);
            }

            auto parse_src_input_specs = [&](const std::string &s, char delim,
                                                 const bool is_ternary) {
                size_t src_subpos = 0;

                // parse data type for the src tensor - provided only for
                // the src1 tensor
                if (!is_ternary) {
                    const auto dt_str = get_substr(s, src_subpos, delim);
                    e.binary.src1_dt = str2dt(dt_str.c_str());

                    if (e.binary.src1_dt == dnnl_data_type_undef) {
                        BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                                "Error: binary post-op data type",
                                dt_str.c_str(), "is not recognized.");
                        SAFE_V(FAIL);
                    }
                }

                if (src_subpos == std::string::npos) return;

                // an in-place post-op aliases the destination, which settles
                // both the mask and the layout, leaving the data type as the
                // only thing to spell out
                if (e.is_binary_kind_inplace()) {
                    BENCHDNN_PRINT(0, "%s \'%s\'.\n",
                            "Error: an in-place binary post-op takes no "
                            "specification besides the data type. Provided:",
                            s.substr(src_subpos).c_str());
                    SAFE_V(FAIL);
                }

                // parse mask input - processed for both src1/src2 tensors.
                const auto mask_input_str = get_substr(s, src_subpos, delim);
                if (utils::has_only_digits(mask_input_str)) {
                    // If an input consists of digits only, then read it as the
                    // int value.
                    const auto src_mask = utils::stoll_safe(mask_input_str);

                    if (!is_ternary) {
                        e.binary.mask = src_mask;
                        e.binary.mask_input = attr_t::mask_input_t::mask;
                    } else {
                        e.binary.src2_mask = src_mask;
                        e.binary.src2_mask_input = attr_t::mask_input_t::mask;
                    }
                } else {
                    // Otherwise, re-direct to policy parsing.
                    const auto src_policy = attr_t::str2policy(mask_input_str);

                    if (!is_ternary) {
                        e.binary.policy = src_policy;
                        e.binary.mask_input = attr_t::mask_input_t::policy;

                        if (e.binary.policy == attr_t::policy_t::POLICY_TOTAL) {
                            BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                                    "Error: binary post-op policy",
                                    mask_input_str.c_str(),
                                    "is not "
                                    "recognized. Input also is not consisted "
                                    "of only integers to process it as mask "
                                    "directly.");
                            SAFE_V(FAIL);
                        }
                    } else {
                        e.binary.src2_policy = src_policy;
                        e.binary.src2_mask_input = attr_t::mask_input_t::policy;

                        if (e.binary.src2_policy
                                == attr_t::policy_t::POLICY_TOTAL) {
                            BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                                    "Error: binary post-op policy for the "
                                    "src2 tensor",
                                    mask_input_str.c_str(),
                                    "is not recognized.");
                            SAFE_V(FAIL);
                        }
                    }
                }

                if (src_subpos == std::string::npos) return;

                // parse tag input - processed for both src1/2 tensors.
                //
                // WARNING: "grouped" in the tag position means post-op tensor
                // follows dst layout (e.g., offsets, layout, strides),
                // in this case tag is left as default (any)
                const auto tag_str = get_substr(s, src_subpos, delim);
                if (tag_str == "grouped") {
                    e.binary.grouped = true;
                } else {
                    if (check_tag(tag_str) != OK) {
                        BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                                "Error: binary post-op tag", tag_str.c_str(),
                                "is not recognized.");
                        SAFE_V(FAIL);
                    }
                    e.binary.tag = tag_str;
                }

                if (src_subpos == std::string::npos) return;

                const auto strides_str = get_substr(s, src_subpos, delim);
                if (!strides_str.empty()) {
                    parse_vector_str(e.binary.strides, dims_t(),
                            utils::stoll_safe, strides_str, 'x');
                }

                if (src_subpos != std::string::npos) {
                    const auto unknown_str = get_substr(s, src_subpos, delim);
                    BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                            "Warning: Additional unrecognized arguments",
                            unknown_str.c_str(), "are specified.");
                }
            };

            bool is_ternary_input = false;
            for (const auto &s : {src1_subs, src2_subs}) {
                parse_src_input_specs(s, src_delim, is_ternary_input);
                if (!has_src2_specs)
                    break;
                else
                    is_ternary_input = true;
            }

        } else if (e.is_prelu_kind()) {
            const auto mask_input_str = get_substr(subs, subs_pos, ':');
            if (utils::has_only_digits(mask_input_str)) {
                // If an input consists of digits only, then read it as the int
                // value.
                e.prelu.mask_input = attr_t::mask_input_t::mask;
                e.prelu.mask = utils::stoll_safe(mask_input_str);
            } else {
                // Otherwise, re-direct to policy parsing.
                e.prelu.mask_input = attr_t::mask_input_t::policy;
                e.prelu.policy = attr_t::str2policy(mask_input_str);
                if (e.prelu.policy == attr_t::policy_t::POLICY_TOTAL) {
                    BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                            "Error: prelu post-op policy",
                            mask_input_str.c_str(), "is not recognized.");
                    SAFE_V(FAIL);
                }
            }
        }
        if (subs_pos == std::string::npos) continue;
    }

    return v;
}

attr_t::deterministic_t str2attr_deterministic(const std::string &s) {
    attr_t::deterministic_t v;
    if (s.empty()) return v;

    v.enabled = parsers::str2bool(s);
    return v;
}

attr_t::fpmath_mode_t str2attr_fpmath_mode(const std::string &s) {
    attr_t::fpmath_mode_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    auto subs = get_substr(s, start_pos, ':');
    v.mode = str2fpmath_mode(subs.c_str());
    if (start_pos == std::string::npos) return v;

    subs = get_substr(s, start_pos, '\0');
    v.apply_to_int = parsers::str2bool(subs);

    return v;
}

attr_t::rounding_mode_t str2attr_rounding_mode(const std::string &s) {
    attr_t::rounding_mode_t rm;

    if (s.empty()) return rm;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == DNNL_ARG_UNDEF) {
            BENCHDNN_PRINT(0, "%s\n", "Error: undefined argument index");
            SAFE_V(FAIL);
        }
        if (subs_pos != std::string::npos)
            rm.set(arg,
                    str2rounding_mode(parser::get_substr(subs, subs_pos, ':')));
        if (subs_pos != std::string::npos)
            rm.set_seed(utils::stoll_safe(get_substr(subs, subs_pos, ':')));
    }
    return rm;
}

attr_t::dropout_t str2attr_dropout(const std::string &s) {
    attr_t::dropout_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    auto subs = get_substr(s, start_pos, ':');
    v.p = utils::stof_safe(subs);
    if ((v.p < 0.f) || (v.p > 1.f)) {
        BENCHDNN_PRINT(0, "Error: bad dropout probability value: %f\n", v.p);
        SAFE_V(FAIL);
    }
    if (start_pos == std::string::npos) return v;

    subs = get_substr(s, start_pos, ':');
    v.seed = utils::stoll_safe(subs);
    if (start_pos == std::string::npos) return v;

    v.tag = get_substr(s, start_pos, ':');
    if (check_tag(v.tag) != OK) {
        BENCHDNN_PRINT(0, "%s \'%s\' %s\n", "Error: dropout mask tag",
                v.tag.c_str(), "is not recognized.");
        SAFE_V(FAIL);
    }
    if (start_pos == std::string::npos) return v;

    subs = get_substr(s, start_pos, ':');
    v.offset = utils::stoll_safe(subs);
    if (v.offset < 0) {
        BENCHDNN_PRINT(
                0, "Error: bad dropout offset value: %" PRId64 "\n", v.offset);
        SAFE_V(FAIL);
    }
    if (start_pos == std::string::npos) return v;

    subs = get_substr(s, start_pos, '\0');
    v.use_host_scalars = parsers::str2bool(subs);

    return v;
}

bool str2impl_filter(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, bool use_impl, const char *str,
        const std::string &option_name, const std::string &help) {
    const auto chars2chars = [](const char *str) { return str; };
    const auto str2impl_filter = [&](const char *str) {
        std::vector<std::string> v, def;
        parse_vector_str(v, def, chars2chars, str);

        // Remove all quotes from input string since they affect the search.
        for_(auto &e : v)
        for (auto c : {'"', '\''}) {
            size_t start_pos = 0;
            while (start_pos != std::string::npos) {
                start_pos = e.find_first_of(c, start_pos);
                if (start_pos != std::string::npos) e.erase(start_pos, 1);
            }
        }

        return impl_filter_t(v, use_impl, /* respect_global_filter = */ true);
    };
    return parse_single_value_option(impl_filter, def_impl_filter,
            str2impl_filter, str, option_name, help);
}

summary_t str2summary(const std::string &s) {
    // Allowed input: (no-)option+...
    summary_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        bool negate_option = false;
        if (subs.find("no-", 0, 3) != std::string::npos) {
            negate_option = true;
            subs_pos += 3;
        }

        auto option = parser::get_substr(subs, subs_pos, '\0');
        if (option == "failures") {
            v.failed_cases = !negate_option;
        } else if (option == "impl") {
            v.impl_names = !negate_option;
        } else if (option == "impl-csv") {
            v.impl_names_csv = !negate_option;
        } else {
            BENCHDNN_PRINT(0,
                    "Error: unsupported option-value combination "
                    "\'--summary=%s\'\n",
                    option.c_str());
            SAFE_V(FAIL);
        }
    }

    return v;
}

cold_cache_input_t str2cold_cache_input(const std::string &s) {
    // Allowed input: MODE[+EXTENSION[+...]]
    // Allowed extensions: TLB[:SIZE]
    cold_cache_input_t c;

    size_t start_pos = 0;
    std::string mode_str = get_substr(s, start_pos, '+');
    if (mode_str == "none") {
        c.cold_cache_mode_ = cold_cache_mode_t::none;
    } else if (mode_str == "wei") {
        c.cold_cache_mode_ = cold_cache_mode_t::wei;
    } else if (mode_str == "all") {
        c.cold_cache_mode_ = cold_cache_mode_t::all;
    } else if (mode_str == "custom") {
        c.cold_cache_mode_ = cold_cache_mode_t::custom;
    } else {
        BENCHDNN_PRINT(0,
                "Error: unknown cold-cache mode \'%s\'. Supported values are "
                "\'wei\', \'all\', or \'custom\'.\n",
                mode_str.c_str());
        SAFE_V(FAIL);
    }

    if (c.cold_cache_mode_ == cold_cache_mode_t::none
            && start_pos != std::string::npos) {
        BENCHDNN_PRINT(0, "%s\n",
                "Error: cold-cache extensions can't be enabled with cold-cache "
                "disabled");
        SAFE_V(FAIL);
    }

    while (start_pos != std::string::npos) {
        std::string ext_str = get_substr(s, start_pos, '+');

        size_t ext_pos = 0;
        std::string ext_main_str = get_substr(ext_str, ext_pos, ':');
        if (ext_main_str == "tlb") {
            c.cold_tlb_ = true;
            if (ext_pos != std::string::npos) {
                std::string ext_aux_str = get_substr(ext_str, ext_pos, '\0');

                const auto last_char = std::toupper(ext_aux_str.back());
                if (last_char != 'G' && last_char != 'M') {
                    BENCHDNN_PRINT(0,
                            "Error: cold-TLB supports only \'M\' or \'G\' "
                            "values for size modification. Given input: "
                            "\'%c\'.\n",
                            last_char);
                    SAFE_V(FAIL);
                }

                std::string size_str = ext_aux_str;
                // Remove size modifier to feed the rest for value verification.
                size_str.pop_back();
                const float size = utils::stof_safe(size_str);
                c.cold_tlb_size_ = static_cast<size_t>(
                        size * 1024 * 1024 * (last_char == 'G' ? 1024 : 1));

                // Save the input string once all values are verified.
                c.cold_tlb_size_str_ = std::move(ext_aux_str);
            }
        } else {
            BENCHDNN_PRINT(0,
                    "Error: unknown cold-cache extension \'%s\'. Supported "
                    "values are \'tlb\'\n.",
                    ext_main_str.c_str());
            SAFE_V(FAIL);
        }
    }

    return c;
}

bool str2bool(const std::string &str) {
    auto s = utils::lowercase(str);
    if (s == "1" || s == "true") {
        return true;
    } else if (s == "0" || s == "false") {
        return false;
    } else {
        BENCHDNN_PRINT(0,
                "Error: parsed value \'%s\' is expected to be a boolean value "
                "or 0/1 integer value\n",
                s.c_str());
        SAFE_V(FAIL);
    }
    return false;
}

execution_mode_t str2execution_mode(const std::string &str) {
    auto s = utils::lowercase(str);
    if (s == "direct") {
        return execution_mode_t::direct;
    } else if (s == "graph") {
        return execution_mode_t::graph;
    } else if (s == "native_graph") {
        return execution_mode_t::native_graph;
    } else {
        BENCHDNN_PRINT(
                0, "%s", "Error: execution mode value is not recognized.\n");
        SAFE_V(FAIL);
    }
    return execution_mode_t::direct;
}

// Maps a QUANT_MODE token to its library quantization mode. Accepts both the
// library-canonical names emitted by `quantization_mode2str` and the shorter
// aliases. Returns `undef` for an unrecognized token so callers can tell a
// quantization mode apart from a neighbor token (e.g. groups) without failing.
dnnl_quantization_mode_t str2quantization_mode(const std::string &str) {
    auto s = utils::lowercase(str);
    if (s == "sazp" || s == "static_sazp")
        return dnnl_quantization_mode_static_sazp;
    if (s == "sbzp" || s == "static_sbzp")
        return dnnl_quantization_mode_static_sbzp;
    if (s == "mx" || s == "dynamic_mx")
        return dnnl_quantization_mode_dynamic_mx;
    if (s == "dynamic_fp") return dnnl_quantization_mode_dynamic_fp;
    return dnnl_quantization_mode_undef;
}

runtime_kind_t str2runtime_kind(const std::string &str) {
    auto s = utils::lowercase(str);
    if (s == "cpu") {
        return runtime_kind_t::cpu;
    } else if (s == "gpu") {
        return runtime_kind_t::gpu;
    } else {
        bool parsed = false;
        bool val = utils::str2bool_safe(str, parsed);
        if (!parsed) {
            BENCHDNN_PRINTF(0, "%s \'%s\' %s", "Error: parsed value", s.c_str(),
                    "is expected to be a boolean value, or 0/1 integer value, "
                    "or \'cpu|gpu\' string literals");
            SAFE_V(FAIL);
        }
        if (val) return runtime_kind_t::all;
    }
    return runtime_kind_t::undefined;
}

} // namespace parsers

// vector types
bool parse_dir(std::vector<dir_t> &dir, const std::vector<dir_t> &def_dir,
        const char *str, const std::string &option_name /* = "dir"*/) {
    static const std::string help
            = "DIR    (Default: `FWD_B` when bias applicable, `FWD_D` "
              "otherwise)\n    Specifies propagation kind `DIR` for operation. "
              "Has bias support incorporated with `_B` suffix.\n    `DIR` "
              "values can be `FWD_B`, `FWD_D`, `FWD_I`, `BWD_D`, `BWD_WB`, "
              "`BWD_W` and `BWD_DW`.\n    More details at "
            + doc_url + "knobs_dir.md\n";
    return parse_vector_option(dir, def_dir, str2dir, str, option_name, help);
}

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name /* = "dt"*/) {
    static const std::string help
            = "DT    (Default: `f32`)\n    Specifies data type `DT` for source "
              "and/or destination.\n    `DT` values can be `f32`, `bf16`, "
              "`f16`, `s32`, `s8`, `u8`.\n";
    return parse_vector_option(dt, def_dt, str2dt, str, option_name, help);
}

bool parse_multi_dt(std::vector<std::vector<dnnl_data_type_t>> &dt,
        const std::vector<std::vector<dnnl_data_type_t>> &def_dt,
        const char *str, const std::string &option_name /* = "sdt"*/) {
    static const std::string help
            = "DT0:DT1[:DTi]    (Default: `f32` for all)\n    When the driver "
              "supports the notion of multiple sources, the option specifies a "
              "data type `DTi` for a source `i`.\n    When the driver supports "
              "the notion of source, weights (optional), and destination, the "
              "option specifies data types for source, weights (optional) and "
              "destination correspondently.\n    The option may support "
              "broadcast semantics (check the driver online documentation), "
              "when a single value will be used for all inputs.\n    `DT` "
              "values can be `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`.\n";
    return parse_multivector_option(dt, def_dt, str2dt, str, option_name, help);
}

bool parse_tag(std::vector<std::string> &tag,
        const std::vector<std::string> &def_tag, const char *str,
        const std::string &option_name /* = "tag"*/) {
    static const std::string help
            = "TAG    (Default: `any` for compute-bound, `abx` for rest)\n    "
              "Specifies memory format tag `TAG` for source, weights, or "
              "destination.\n    Valid `TAG` values can be found at "
            + doc_url + "knobs_tag.md\n";

    auto ret_string = [](const char *str) { return std::string(str); };
    bool ok = parse_vector_option(
            tag, def_tag, ret_string, str, option_name, help);
    if (!ok) return false;

    for (size_t i = 0; i < tag.size(); i++) {
        if (check_tag(tag[i], allow_enum_tags_only) != OK) {
            if (allow_enum_tags_only && check_tag(tag[i]) == OK) {
                fprintf(stderr,
                        "ERROR: tag `%s` is valid but not found in "
                        "`dnnl::memory::format_tag`. To force the testing with "
                        "this tag, please specify `--allow-enum-tags-only=0` "
                        "prior to any tag option.\n",
                        tag[i].c_str());
            } else {
                fprintf(stderr,
                        "ERROR: unknown or invalid tag: `%s`, exiting...\n",
                        tag[i].c_str());
            }
            exit(2);
        }
    }
    return true;
}

bool parse_encoding(std::vector<sparse_options_t> &sparse_options,
        const char *str, const std::string &option_name /* = "encoding"*/) {
    static const std::string help
            = "ENCODING[+SPARSITY]:ENCODING[+SPARSITY]:ENCODING[+SPARSITY]\n   "
              "Specifies sparse encodings and sparsity.\n    More details at "
            + doc_url + "knobs_encoding.md\n";

    std::vector<sparse_options_t> def {sparse_options_t()};
    auto parse_sparse_options_func = [](const std::string &s) {
        sparse_options_t v;
        SAFE_V(v.from_str(s));
        return v;
    };

    return parse_vector_option(sparse_options, def, parse_sparse_options_func,
            str, option_name, help);
}

// Format: DIM_IDX:NUM_GROUPS:{size0+...+sizeN[:max_size] | PROFILE[.PARAM][:HINT]}
// - DIM_IDX is the variable dim (0=M, 1=K)
// - NUM_GROUPS is the experts/groups count
// - the rest is either explicit '+'-separated sizes or a generated PROFILE,
//   see doc/driver_matmul.md (--grouped) for the full info
//
// TODO: move to parse_single_value_option once feature is moved out of experimental
bool parse_grouped(std::vector<sparse_options_t> &sparse_options,
        const char *str, const std::string &option_name /* = "grouped"*/) {
    static const std::string help
            = "DIM_IDX:NUM_GROUPS:size0+size1+...+sizeN[:max_size]\n   "
              "DIM_IDX:NUM_GROUPS:PROFILE[.PARAM][:HINT]  [,config2,...]\n   "
              "Specifies grouped encoding for MoE workloads.\n"
              "    DIM_IDX is the dimension index (0=M, 1=K)\n"
              "    NUM_GROUPS is the number of expert groups\n"
              "    Explicit sizes: size0+...+sizeN ('+'-separated, sum along "
              "DIM_IDX); max_size (optional) is the max_variable_dim dispatch "
              "hint.\n"
              "    PROFILE synthesizes the per-group sizes from the shape's "
              "variable dim (total_M):\n"
              "      balanced[.seed] - tokens distributed across all "
              "NUM_GROUPS like a balanced router (seed picks the draw, "
              "default 0)\n"
              "      hot[.pct]       - one hot expert absorbs pct% of the "
              "tokens, the rest spread evenly (default 50; pct=100 = "
              "absorb-all; pct in [1, 100])\n"
              "      decode          - total_M active experts, 1 token each, "
              "rest empty (needs total_M <= NUM_GROUPS)\n"
              "    PARAM (optional, attached to PROFILE via '.') tunes the "
              "profile; omit it to keep the default.\n"
              "    HINT (optional, next ':' field) overrides the "
              "max_variable_dim dispatch hint; it defaults to the largest "
              "generated group and must be >= it.\n"
              "    Multiple configs can be comma-separated to iterate over "
              "them.\n"
              "    Example: --grouped=0:8:32+64+32+96+48+80+56+72\n"
              "    Example: --grouped=0:8:32+64+32+96+48+80+56+72:96\n"
              "    Example: --grouped=0:256:balanced\n"
              "    Example: --grouped=0:256:balanced:512\n"
              "    Example: --grouped=0:64:hot.100\n"
              "    Example: --grouped=0:256:decode\n"
              "    Example: --grouped=0:256:decode:512\n";

    utils::add_option_to_help(option_name, help);
    const std::string pattern = utils::get_pattern(option_name);
    if (!utils::option_matched(pattern, str)) return false;

#if !DNNL_EXPERIMENTAL_GROUPED_MEMORY
    BENCHDNN_PRINT(0, "%s\n",
            "\'--grouped\' option requires DNNL_EXPERIMENTAL_GROUPED_MEMORY "
            "build option set to ON.");
    SAFE_V(FAIL);
#endif

    str = str + pattern.size();
    std::string full_str(str);

    if (full_str.empty()) {
        sparse_options.assign({sparse_options_t()});
        return true;
    }

    sparse_options.clear();
    using gd_t = sparse_options_t::grouped_data_t;

    // Split on ',' for multiple grouped configs
    size_t cfg_pos = 0;
    while (cfg_pos <= full_str.size()) {
        size_t cfg_next = full_str.find(',', cfg_pos);
        std::string s = (cfg_next == std::string::npos)
                ? full_str.substr(cfg_pos)
                : full_str.substr(cfg_pos, cfg_next - cfg_pos);

        sparse_options_t v;

        // Parse format: DIM_IDX:NUM_GROUPS:size0+size1+...[:max_size]
        size_t start_pos = 0;

        // Parse dimension index
        const auto dim_idx_str = get_substr(s, start_pos, ':');
        if (start_pos == std::string::npos) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: grouped format requires DIM_IDX:NUM_GROUPS:sizes");
            SAFE_V(FAIL);
        }

        int variable_dim_idx = static_cast<int>(utils::stoll_safe(dim_idx_str));

        // Validate dimension index (0=M, 1=K)
        if (variable_dim_idx < 0 || variable_dim_idx > 1) {
            BENCHDNN_PRINT(0,
                    "Error: dimension index must be 0 (M) or 1 (K) "
                    "got %d\n",
                    variable_dim_idx);
            SAFE_V(FAIL);
        }

        // Parse number of groups
        const auto group_count_str = get_substr(s, start_pos, ':');
        if (start_pos == std::string::npos) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: grouped format requires NUM_GROUPS and sizes");
            SAFE_V(FAIL);
        }

        dnnl_dim_t group_count = utils::stoll_safe(group_count_str);
        if (group_count <= 0) {
            BENCHDNN_PRINT(0,
                    "Error: grouped NUM_GROUPS must be > 0, got %lld\n",
                    (long long)group_count);
            SAFE_V(FAIL);
        }

        // Next field is either an explicit '+'-separated size list or a
        // generated profile PROFILE[.PARAM] (balanced, hot, decode)
        const std::string sizes_str = get_substr(s, start_pos, ':');

        std::vector<dnnl_dim_t> sizes;
        dnnl_dim_t max_variable_dim = 0;
        auto profile = gd_t::profile_none;
        dnnl_dim_t profile_param = -1;

        // Profile starts with a letter, explicit sizes with a digit
        if (!sizes_str.empty()
                && std::isalpha(static_cast<unsigned char>(sizes_str[0]))) {
            const size_t dot = sizes_str.find('.');
            const std::string profile_str = sizes_str.substr(0, dot);

            // optional PARAM attaches to PROFILE via '.'
            std::string param_str;
            if (dot != std::string::npos) param_str = sizes_str.substr(dot + 1);

            if (profile_str == "balanced")
                profile = gd_t::profile_balanced;
            else if (profile_str == "hot")
                profile = gd_t::profile_hot;
            else if (profile_str == "decode")
                profile = gd_t::profile_decode;
            else {
                BENCHDNN_PRINT(0,
                        "Error: unknown grouped profile '%s' (expected "
                        "balanced|hot|decode)\n",
                        profile_str.c_str());
                SAFE_V(FAIL);
            }

            if (!param_str.empty()) {
                if (profile == gd_t::profile_decode) {
                    BENCHDNN_PRINT(0, "%s\n",
                            "Error: 'decode' profile takes no parameter");
                    SAFE_V(FAIL);
                }
                profile_param = utils::stoll_safe(param_str);
                if (profile == gd_t::profile_balanced && profile_param < 0) {
                    BENCHDNN_PRINT(0,
                            "Error: 'balanced' seed must be >= 0, got %lld\n",
                            (long long)profile_param);
                    SAFE_V(FAIL);
                }
                if (profile == gd_t::profile_hot
                        && (profile_param < 1 || profile_param > 100)) {
                    BENCHDNN_PRINT(0,
                            "Error: 'hot' percent must be in [1, 100], got "
                            "%lld\n",
                            (long long)profile_param);
                    SAFE_V(FAIL);
                }
            }

        } else {
            // Explicit '+'-separated sizes
            size_t size_pos = 0;
            while (size_pos != std::string::npos) {
                sizes.push_back(utils::stoll_safe(
                        get_substr(sizes_str, size_pos, '+')));
            }
            if (sizes.size() != (size_t)group_count) {
                BENCHDNN_PRINT(0,
                        "Error: number of sizes (%zu) doesn't match "
                        "group_count (%lld)\n",
                        sizes.size(), (long long)group_count);
                SAFE_V(FAIL);
            }
        }

        // optional HINT is in the next ':' field
        if (start_pos != std::string::npos) {
            const std::string hint_str = get_substr(s, start_pos, ':');
            if (!hint_str.empty())
                max_variable_dim = utils::stoll_safe(hint_str);
        }

        if (start_pos != std::string::npos) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: grouped takes at most "
                    "DIM_IDX:NUM_GROUPS:{PROFILE[.PARAM]|sizes}[:HINT]");
            SAFE_V(FAIL);
        }

        // Explicit sizes: HINT must not be smaller than the largest group
        if (!sizes.empty()) {
            const dnnl_dim_t max_size
                    = *std::max_element(sizes.begin(), sizes.end());
            if (max_variable_dim > 0 && max_variable_dim < max_size) {
                BENCHDNN_PRINT(0,
                        "Error: grouped HINT (%lld) is smaller than the max "
                        "group size (%lld)\n",
                        (long long)max_variable_dim, (long long)max_size);
                SAFE_V(FAIL);
            }
        }

        // Set grouped encoding based on the src varying dim:
        //  - dim 0 (M varies, 2Dx3D variant)
        //    src is [total_M, K] (var_dim_idx=0),  wei is dense [G, K, N],
        //    dst is [total_M, N] (var_dim_idx=0)
        //  - dim 1 (K/contraction varies, 2Dx2D variant)
        //    src is [M, total_K] (var_dim_idx=1), wei is [total_K, N] (var_dim_idx=0),
        //    dst is dense 3D [G, M, N]
        gd_t gd;
        gd.group_count = group_count;
        gd.max_variable_dim = max_variable_dim;
        if (profile != gd_t::profile_none) {
            gd.profile.kind = profile;
            gd.profile.param = profile_param;
        } else {
            gd.group_sizes = sizes;
        }
        gd.variable_dim_idx = variable_dim_idx;
        v.set_grouped(DNNL_ARG_SRC, gd);
        // dst (for 2Dx3D) / weights (for 2Dx2D) vary along dim 0
        gd.variable_dim_idx = 0;
        v.set_grouped(
                variable_dim_idx == 0 ? DNNL_ARG_DST : DNNL_ARG_WEIGHTS, gd);

        sparse_options.push_back(v);

        if (cfg_next == std::string::npos) break;
        cfg_pos = cfg_next + 1;
    }

    return true;
}

bool parse_multi_tag(std::vector<std::vector<std::string>> &tag,
        const std::vector<std::vector<std::string>> &def_tag, const char *str,
        const std::string &option_name /* = "stag"*/) {
    static const std::string help
            = "TAG0:TAG1[:TAGi]    (Default: `any` for compute-bound, `abx` "
              "for rest)\n    Specifies memory format tag `TAGi` for source "
              "i.\n    Valid `TAGi` values can be found at "
            + doc_url + "knobs_tag.md\n";
    auto ret_string = [](const char *str) { return std::string(str); };
    return parse_multivector_option(
            tag, def_tag, ret_string, str, option_name, help);
}

bool parse_mb(std::vector<int64_t> &mb, const std::vector<int64_t> &def_mb,
        const char *str, const std::string &option_name /* = "mb"*/) {
    static const std::string help
            = "UINT    (Default: `0`)\n    Overrides mini-batch value "
              "specified in a problem descriptor with `UINT` value.\n    When "
              "set to `0`, takes no effect.\n";
    return parse_vector_option(
            mb, def_mb, utils::stoll_safe, str, option_name, help);
}

bool parse_attr_post_ops(std::vector<attr_t::post_ops_t> &po, const char *str,
        const std::string &option_name = "attr-post-ops") {
    static const std::string help
            = "POST-OPS\n    Specifies post-ops attribute. `POST-OPS` syntax "
              "is one of those:\n    * SUM[:SCALE[:ZERO_POINT[:DATA_TYPE]]]\n  "
              "  * ELTWISE[:ALPHA[:BETA[:SCALE]]]\n    * DW:KkSsPp[:DST_DT]\n  "
              "  * BINARY:DT[:MASK_INPUT[:TAG]]\n    * PRELU[:MASK_INPUT]\n    "
              "More details at "
            + doc_url + "knobs_attr.md\n";
    std::vector<attr_t::post_ops_t> def {attr_t::post_ops_t()};
    return parse_vector_option(
            po, def, parsers::str2attr_post_ops, str, option_name, help);
}

bool parse_attr_scales(std::vector<attr_t::arg_scales_t> &scales,
        const char *str, const std::string &option_name = "attr-scales") {
    static const std::string help
            = "ARG:MASK_INPUT[:SCALE[:DATA_TYPE[:GROUPS]]][+...]\n"
              "    Specifies input scales attribute.\n"
              "    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_subattr(scales, str, option_name, help);
}

bool parse_attr_zero_points(std::vector<attr_t::zero_points_t> &zp,
        const char *str, const std::string &option_name = "attr-zero-points") {
    static const std::string help
            = "ARG:MASK_INPUT[:ZEROPOINT[:DATA_TYPE[:GROUPS]]][+...]\n"
              "    Specifies zero-points attribute.\n"
              "    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_subattr(zp, str, option_name, help);
}

bool parse_attr_precomputed_reductions(
        std::vector<attr_t::precomputed_reductions_t> &pr, const char *str,
        const std::string &option_name = "attr-precomputed-reductions") {
    static const std::string help
            = "ARG:MASK_INPUT:DATA_TYPE:GROUPS[+...]\n    Specifies "
              "precomputed reductions attribute.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_subattr(pr, str, option_name, help);
}

bool parse_attr_rounding_mode(std::vector<attr_t::rounding_mode_t> &rm,
        const char *str,
        const std::string &option_name = "attr-rounding-mode") {
    static const std::string help
            = "ARG:MODE[:SEED][+...]    (Default: `environment`)\n    "
              "Specifies a "
              "rounding mode MODE to be applied upon conversion of argument "
              "ARG.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(
            rm, {}, parsers::str2attr_rounding_mode, str, option_name, help);
}

bool parse_attr_scratchpad_mode(
        std::vector<dnnl_scratchpad_mode_t> &scratchpad_mode,
        const std::vector<dnnl_scratchpad_mode_t> &def_scratchpad_mode,
        const char *str, const std::string &option_name = "attr-scratchpad") {
    static const std::string help
            = "MODE    (Default: `library`)\n    Specifies scratchpad "
              "attribute. `MODE` values can be `library` or `user`.\n    More "
              "details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(scratchpad_mode, def_scratchpad_mode,
            str2scratchpad_mode, str, option_name, help);
}

bool parse_attr_fpmath_mode(std::vector<attr_t::fpmath_mode_t> &fpmath_mode,
        const std::vector<attr_t::fpmath_mode_t> &def_fpmath_mode,
        const char *str, const std::string &option_name = "attr-fpmath") {
    static const std::string help
            = "MODE[:APPLY_TO_INT]    (Default: `strict[:false]`)\n    "
              "Specifies "
              "fpmath_mode attribute. `MODE` values can be `strict` or "
              "`bf16`. `APPLY_TO_INT` values can be `true` or `false`.\n";
    return parse_vector_option(fpmath_mode, def_fpmath_mode,
            parsers::str2attr_fpmath_mode, str, option_name, help);
}

bool parse_attr_dropout(std::vector<attr_t::dropout_t> &dropout,
        const std::vector<attr_t::dropout_t> &def_dropout, const char *str,
        const std::string &option_name = "attr-dropout") {
    static const std::string help
            = "PROBABILITY[:SEED[:TAG[:OFFSET[:USE_HOST_SCALARS]]]]\n    "
              "Specifies dropout attribute.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(dropout, def_dropout, parsers::str2attr_dropout,
            str, option_name, help);
}

bool parse_attr_acc_mode(std::vector<dnnl_accumulation_mode_t> &acc_mode,
        const std::vector<dnnl_accumulation_mode_t> &def_acc_mode,
        const char *str, const std::string &option_name = "attr-acc-mode") {
    static const std::string help
            = "MODE    (Default: `strict`)\n    Specifies accumulation mode "
              "attribute. `MODE` values can be `strict`, `relaxed`, `any`,"
              "`f32`, `f16` or `s32`.\n";
    return parse_vector_option(acc_mode, def_acc_mode, str2accumulation_mode,
            str, option_name, help);
}

bool parse_attr_deterministic(
        std::vector<attr_t::deterministic_t> &deterministic,
        const std::vector<attr_t::deterministic_t> &def_deterministic,
        const char *str,
        const std::string &option_name = "attr-deterministic") {
    static const std::string help
            = "MODE    (Default: `false`)\n    Specifies deterministic mode "
              "attribute. `MODE` values can be `true`, or `false`.\n";
    return parse_vector_option(deterministic, def_deterministic,
            parsers::str2attr_deterministic, str, option_name, help);
}

bool parse_attributes(
        base_settings_t &s, const base_settings_t &def, const char *str) {
    const bool parsed_attrs = parse_attr_scales(s.scales, str)
            || parse_attr_zero_points(s.zero_points, str)
            || parse_attr_precomputed_reductions(s.precomputed_reductions, str)
            || parse_attr_post_ops(s.post_ops, str)
            || parse_attr_dropout(s.dropout, def.dropout, str)
            || parse_attr_scratchpad_mode(
                    s.scratchpad_mode, def.scratchpad_mode, str)
            || parse_attr_fpmath_mode(s.fpmath_mode, def.fpmath_mode, str)
            || parse_attr_acc_mode(s.acc_mode, def.acc_mode, str)
            || parse_attr_deterministic(s.deterministic, def.deterministic, str)
            || parse_attr_rounding_mode(s.rounding_mode, str);
    return parsed_attrs;
}

bool parse_axis(std::vector<int> &axis, const std::vector<int> &def_axis,
        const char *str, const std::string &option_name /* = "axis"*/) {
    static const std::string help
            = "UINT    (Default: `1`)\n    Specifies axis dimension `UINT` for "
              "an operation.\n";
    return parse_vector_option(
            axis, def_axis, utils::stoll_safe, str, option_name, help);
}

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name /* = "match"*/) {
    static const std::string help
            = "REGEX    (Default: not specified)\n    `REGEX` is a string "
              "literal representing a regular expression that filters problem "
              "descriptors.\n    Matched descriptors are executed, rest are "
              "skipped.\n";
    const char *def_match = "";
    const auto chars2chars = [](const char *str) { return str; };
    return parse_single_value_option(
            match, def_match, chars2chars, str, option_name, help);
}

bool parse_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name) {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Instructs the driver "
              "to fetch the next implementation from the list if fetched "
              "implementation name doesn't match any from the `STRINGS` list "
              "(a.k.a. include-list).\n    `STRINGS` is a comma-separated list "
              "of string literal entries with no spaces.\n    When empty, the "
              "option has no effect. The option is opposite to "
              "`--skip-impl`.\n";

    return parsers::str2impl_filter(impl_filter, def_impl_filter,
            /* use_impl = */ true, str, option_name, help);
}

bool parse_skip_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name) {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Instructs the driver "
              "to fetch the next implementation from the list if fetched "
              "implementation name matches any from the `STRINGS` list (a.k.a. "
              "exclude-list).\n    `STRINGS` is a comma-separated list of "
              "string literal entries with no spaces.\n    When empty, the "
              "option has no effect. The option is opposite to `--impl`.\n";

    return parsers::str2impl_filter(impl_filter, def_impl_filter,
            /* use_impl = */ false, str, option_name, help);
}

bool parse_inplace(std::vector<bool> &inplace,
        const std::vector<bool> &def_inplace, const char *str,
        const std::string &option_name /* = "inplace"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to use "
              "same memory data handle for source and destination when set to "
              "`true`.\n";
    return parse_vector_option(
            inplace, def_inplace, parsers::str2bool, str, option_name, help);
}

bool parse_skip_nonlinear(std::vector<bool> &skip,
        const std::vector<bool> &def_skip, const char *str,
        const std::string &option_name /* = "skip-nonlinear"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to treat "
              "transcendental activations as linear when set to `true`.\n";
    return parse_vector_option(
            skip, def_skip, parsers::str2bool, str, option_name, help);
}

bool parse_strides(std::vector<vdims_t> &strides,
        const std::vector<vdims_t> &def_strides, const char *str,
        const std::string &option_name /* = "strides"*/) {
    static const std::string help
            = "DIMS_SRC[:DIMS_WEI]:DIMS_DST    (Default: not specified)\n    "
              "Specifies strides `DIMS_ARG` for correspondent supported "
              "`ARG`.\n    If correspondent `DIMS_ARG` is empty, it does not "
              "take an effect.\n    More details at "
            + doc_url + "driver_" + driver_name + ".md\n";
    auto str2strides = [&](const char *str) -> vdims_t {
        vdims_t strides(STRIDES_SIZE);
        parse_multivector_str(
                strides, vdims_t(), utils::stoll_safe, str, ':', 'x');
        return strides;
    };
    return parse_vector_option(
            strides, def_strides, str2strides, str, option_name, help);
}

bool parse_trivial_strides(std::vector<bool> &ts,
        const std::vector<bool> &def_ts, const char *str,
        const std::string &option_name /* = "trivial-strides"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to use "
              "dense (trivial) strides when set to `true`.\n";
    return parse_vector_option(
            ts, def_ts, parsers::str2bool, str, option_name, help);
}

bool parse_scale_policy(std::vector<policy_t> &policy,
        const std::vector<policy_t> &def_policy, const char *str,
        const std::string &option_name /*= "scaling"*/) {
    static const std::string help
            = "POLICY    (Default: `common`)\n    Specifies a mask for scales "
              "to be applied.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(
            policy, def_policy, attr_t::str2policy, str, option_name, help);
}

// plain types
bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name /* = "perf-template"*/) {
    static const std::string help
            = "TEMPLATE    (Default: `def`)\n    Specifies performance output "
              "template for perf mode. `TEMPLATE` values can be `def`, `csv` "
              "or customized set.\n    More details at "
            + doc_url + "knobs_perf_report.md\n";
    const auto str2pt = [&pt_def, &pt_csv](const char *str_) {
        const std::string csv_pattern = "csv";
        const std::string def_pattern = "def";
        if (utils::option_matched(csv_pattern, str_))
            return pt_csv;
        else if (utils::option_matched(def_pattern, str_))
            return pt_def;
        else
            return str_;
    };
    return parse_single_value_option(
            pt, pt_def, str2pt, str, option_name, help);
}

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name /* = "batch"*/) {
    static const std::string help
            = "FILE\n    Instructs the driver to take options and problem "
              "descriptors from a `FILE`.\n";
    int status = OK;
    const auto str2batch = [bench](const char *str_) {
        SAFE(batch(str_, bench), CRIT);
        return OK;
    };
    return parse_single_value_option(
            status, FAIL, str2batch, str, option_name, help);
}

bool parse_help(const char *str, const std::string &option_name /* = "help"*/) {
    std::string pattern = utils::get_pattern(option_name, false);
    if (!utils::option_matched(pattern, str)) return false;

    BENCHDNN_PRINT(0, "%s\n", help_ss.str().c_str());
    exit(0);
}

bool parse_main_help(
        const char *str, const std::string &option_name /* = "help"*/) {
    std::string pattern = utils::get_pattern(option_name, false);
    if (!utils::option_matched(pattern, str)) return false;

    static const std::string main_help
            = "Usage:\n    benchdnn --<driver> [global_options] "
              "[driver_options] problem_description\n\nList of supported "
              "<drivers> (lower case accepted only):\n    * binary\n    * "
              "bnorm\n    * concat\n    * conv\n    * deconv\n    * eltwise\n  "
              "  * ip\n    * lnorm\n    * lrn\n    * matmul\n    * pool\n    * "
              "prelu\n    * reduction\n    * reorder\n    * resampling\n    * "
              "rnn\n    * sdpa\n    * shuffle\n    * softmax\n    * sum\n    * "
              "zeropad\n\nFor global and specific driver options, use:\n    "
              "benchdnn --<driver> --help\n\nMore details at "
            + benchdnn_url + "\n";

    BENCHDNN_PRINT(0, "%s\n", main_help.c_str());
    exit(0);
}

// prb_dims_t type
void parse_prb_vdims(
        prb_vdims_t &prb_vdims, const std::string &str, size_t min_inputs) {
    assert(!str.empty());

    size_t start_pos = 0;
    // `n` is an indicator for a name supplied with dims_t object.
    std::string vdims_str = get_substr(str, start_pos, 'n');
    // Potential trailing underscore before `n` shouldn't be parsed as dims.
    if (vdims_str.back() == '_') vdims_str.pop_back();

    // Sanity check that dims start with a digit.
    if (!std::isdigit(vdims_str[0])) {
        BENCHDNN_PRINT(0, "%s\n%s \'%s\'\n",
                "ERROR: dims are expected to start with an integer value.",
                "Given input:", str.c_str());
        SAFE_V(FAIL);
    }

    std::string name;
    if (start_pos != std::string::npos) name = str.substr(start_pos);

    vdims_t vdims;
    parse_multivector_str(vdims, {dims_t()}, utils::stoll_safe, vdims_str, ':',
            'x', /* allow_empty = */ false);
    // Expect at least two inputs provided
    SAFE_V(vdims.size() >= min_inputs ? OK : FAIL);

    prb_vdims = prb_vdims_t(vdims, name);
}

void parse_prb_dims(prb_dims_t &prb_dims, const std::string &str) {
    size_t start_pos = 0;
    // `n` is an indicator for a name supplied with dims_t object.
    std::string dims_str = get_substr(str, start_pos, 'n');
    // Potential trailing underscore before `n` shouldn't be parsed as dims.
    if (dims_str.back() == '_') dims_str.pop_back();

    parse_vector_str(prb_dims.dims, dims_t(), utils::stoll_safe, dims_str, 'x');

    prb_dims.ndims = static_cast<int>(prb_dims.dims.size());

    if (start_pos != std::string::npos) prb_dims.name = str.substr(start_pos);
}

// Global options
static bool parse_allow_enum_tags_only(const char *str,
        const std::string &option_name = "allow-enum-tags-only") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to validate "
              "format tags against the documented tags from "
              "`dnnl_format_tag_t` enumeration only.\n    When set to `true`, "
              "the only allowed format tags are the ones from "
              "`dnnl_format_tag_t` enumeration.\n";
    return parse_single_value_option(allow_enum_tags_only, true,
            parsers::str2bool, str, option_name, help);
}

static bool parse_attr_same_pd_check(const char *str,
        const std::string &option_name = "attr-same-pd-check") {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to compare "
              "two primitive descriptors - one with requested attributes and "
              "one without them.\n    When set to `true`, check would return "
              "an error if attributes caused fallback to a different "
              "implementation.\n";
    return parse_single_value_option(attr_same_pd_check, false,
            parsers::str2bool, str, option_name, help);
}

static bool parse_canonical(
        const char *str, const std::string &option_name = "canonical") {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to print a "
              "canonical form of a reproducer line.\n    When set to `true`, "
              "the driver prints all options and their values, including "
              "default ones.\n";
    return parse_single_value_option(
            canonical, false, parsers::str2bool, str, option_name, help);
}

static bool parse_check_ref_impl(
        const char *str, const std::string &option_name = "check-ref-impl") {
    static const std::string help
            = "STR    (Default: `false`)\n    Instructs the driver to compare "
              "an implementation name against the \'ref\' string pattern.\n    "
              "When `STR` is set to `true`, the check would return an error if "
              "the implementation name contains such pattern for CPU and GPU "
              "backends.\n    To specify the backend to apply the check for, "
              "set `STR` to `cpu` or `gpu`.\n";
    return parse_single_value_option(check_ref_impl, default_runtime_kind,
            parsers::str2runtime_kind, str, option_name, help);
}

static bool parse_cold_cache(
        const char *str, const std::string &option_name = "cold-cache") {
    static const std::string help
            = "MODE[+EXTENSION]    (Default: `empty`)\n    Instructs the "
              "driver to enable a cold-cache feature for the performance "
              "mode.\n    When `MODE` set to `none` (the default), the "
              "cold-cache mode is disabled.\n    When `MODE` set to `wei`, the "
              "cold-cache is enabled for weights argument only. Targets "
              "forward propagation kind.\n    When `MODE` set to `all`, the "
              "cold-cache is enabled for every execution argument.\n    When "
              "`MODE` set to `custom`, the cold-cache is enabled for custom "
              "arguments which should be specified directly in the code. Refer "
              "to doc for more details.\n    Supported `EXTENSION` values:\n   "
              " * `tlb[:SIZE]`, where `SIZE` is a string-literal with "
              "floating-point number followed by `M` (Megabytes) or `G` "
              "(Gigabytes) characters, e.g., `tlb:500M`.\n";

    return parse_single_value_option(cold_cache_input,
            default_cold_cache_input(), parsers::str2cold_cache_input, str,
            option_name, help);
}

static bool parse_cpu_isa_hints(
        const char *str, const std::string &option_name = "cpu-isa-hints") {
    static const std::string help
            = "HINTS    (Default: `none`)\n    Specifies the ISA specific "
              "hints for CPU engine.\n    `HINTS` values can be `none`, "
              "`no_hints` or `prefer_ymm`.\n";
    const bool parsed
            = parse_single_value_option(hints, isa_hints_t {isa_hints_t::none},
                    isa_hints_t::str2hints, str, option_name, help);
    if (parsed) init_isa_settings();
    return parsed;
}

static bool parse_engine(
        const char *str, const std::string &option_name = "engine") {
    static const std::string help
            = "KIND[:INDEX]    (Default: `cpu`)\n    Instructs the driver to "
              "use an engine with requested `KIND`.\n    `KIND` values can be "
              "`cpu` or `gpu`.\n    `INDEX` is an integer value specifying "
              "which engine to use if several were identified.\n";

    // Note: this is a special case because index and engine kind are parsed
    // into separate global objects instead of one under a common parsing
    // function.
    // TODO: fix this.
    //
    // Because of this fact, need to extract kind separated by `:`. `:` can be
    // valid dangling for certain options in the command line (--strides=::).
    // Thus, extract the kind allowing dangling. Verify, it's `--engine` option,
    // and if yes, perform a safe check for dangling after.
    size_t start_pos = 0;
    std::string kind_str = get_substr(str, start_pos, ':', true);

    if (!parse_single_value_option(engine_tgt_kind, dnnl_cpu, str2engine_kind,
                kind_str.c_str(), option_name, help))
        return false;

    // This is to catch a dangling `:` at the end of `--engine`.
    start_pos = 0;
    kind_str = get_substr(str, start_pos, ':');

    if (start_pos != std::string::npos) {
        std::string index_str(str + start_pos);
        // If the index is a valid number, let the library catch potential
        // issues around unavailable devices, etc.
        engine_index = utils::stoll_safe(index_str);
    }

    return true;
}

static bool parse_fast_ref(
        const char *str, const std::string &option_name = "fast-ref") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to use "
              "faster reference path when doing correctness testing for "
              "`--engine=gpu`.\n    When set to `true`, the library best fit "
              "CPU implementation is used to compute the reference path.\n";
    bool parsed = parse_single_value_option(fast_ref, default_fast_ref,
            parsers::str2bool, str, option_name, help);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (parsed && fast_ref) {
        fast_ref = false;
        fprintf(stderr,
                "%s driver: WARNING: option `fast_ref` is not supported "
                "for GPU only configurations.\n",
                driver_name.c_str());
    }
#endif
    return parsed;
}

static bool parse_global_impl(
        const char *str, const std::string &option_name = "global-impl") {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Same as `--impl` but "
              "overrides any values from `--impl` or `--skip-impl` options met "
              "on the way.\n";

    return parsers::str2impl_filter(global_impl_filter, impl_filter_t(),
            /* use_impl = */ true, str, option_name, help);
}

static bool parse_global_skip_impl(
        const char *str, const std::string &option_name = "global-skip-impl") {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Same as `--skip-impl` "
              "but overrides any values from `--impl` or `--skip-impl` options "
              "met on the way.\n";

    return parsers::str2impl_filter(global_impl_filter, impl_filter_t(),
            /* use_impl = */ false, str, option_name, help);
}

bool parse_ctx(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str,
        const std::string &option_name) {
    const std::string name_in_help
            = (option_name == "ctx-init") ? "initialization." : "execution.";
    const std::string help
            = std::string(
                      "MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]] "
                      "(Default:`auto:auto:auto`)\n    Specifies the threading "
                      "context used during primitive ")
            + name_in_help
            + std::string(
                    "\n    MAX_CONCURRENCY is the maximum number of threads.\n "
                    "   CORE_TYPE enables to select big (value 0) or small "
                    "cores (value 1) for hybrid CPUs (TBB runtime only).\n    "
                    "THREADS_PER_CORE allows to enable/disable hyper-threading "
                    "(TBB runtime only).\n");

    auto str2ctx = [&option_name](const char *str) {
        thr_ctx_t result = get_default_thr_ctx();
        try {
            size_t start_pos = 0;
            /* concurrency piece */
            std::string val_str = get_substr(str, start_pos, ':');
            if (val_str != "auto") result.max_concurrency = std::stoll(val_str);
            /* core_type piece */
            val_str = start_pos != std::string::npos
                    ? get_substr(str, start_pos, ':')
                    : "";
            if (val_str != "auto" && !val_str.empty())
                result.core_type = std::stoll(val_str);
            /* nthr_per_core piece */
            val_str = start_pos != std::string::npos
                    ? get_substr(str, start_pos, ':')
                    : "";
            if (val_str != "auto" && !val_str.empty())
                result.nthr_per_core = std::stoll(val_str);
        } catch (const std::invalid_argument &) {
            BENCHDNN_PRINT(0, "%s %s\n", option_name.c_str(),
                    "fields should be 'auto' or integer values");
            exit(1);
        }

        return result;
    };

    return parse_vector_option(ctx, def_ctx, str2ctx, str, option_name, help);
}

bool parse_ctx_init(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str) {
    return parse_ctx(ctx, def_ctx, str, "ctx-init");
}
bool parse_ctx_exe(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str) {
    return parse_ctx(ctx, def_ctx, str, "ctx-exe");
}

static bool parse_fix_times_per_prb(
        const char *str, const std::string &option_name = "fix-times-per-prb") {
    static const std::string help
            = "UINT    (Default: `0`)\n    Specifies the limit in `UINT` "
              "rounds for performance benchmarking per problem.\n    If `UINT` "
              "is greater than `0`, the number of rounds criterion takes place "
              "over the time criterion.\n";
    bool parsed = parse_single_value_option(fix_times_per_prb,
            default_fix_times_per_prb, utils::stoll_safe, str, option_name,
            help);
    if (parsed) {
        if (bench_mode == bench_mode_t::perf_sim) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: mode=S can't be adjusted. It implies "
                    "fix-times-per-prb=1.");
            exit(2);
        }
        fix_times_per_prb = MAX2(0, fix_times_per_prb);
    }
    return parsed;
}

static bool parse_max_ms_per_prb(
        const char *str, const std::string &option_name = "max-ms-per-prb") {
    static const std::string help
            = "MS    (Default: `3000`)\n    Specifies the limit in `MS` "
              "milliseconds for performance benchmarking per problem.\n    "
              "`MS` is a positive integer in a range [10, 60000].\n";
    bool parsed = parse_single_value_option(max_ms_per_prb,
            default_max_ms_per_prb, utils::stof_safe, str, option_name, help);
    if (parsed) {
        if (bench_mode == bench_mode_t::perf_fast) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: mode=F can't be adjusted. Please use full command "
                    "mode=F aliases with custom max-ms-per-prb input.");
            exit(2);
        }

        max_ms_per_prb = MAX2(10, MIN2(max_ms_per_prb, 60e3));
    }
    return parsed;
}

static bool parse_num_streams(
        const char *str, const std::string &option_name = "num-streams") {
    static const std::string help
            = "N    (Default: `1`)\n    Specifies the number `N` of streams "
              "used for performance benchmarking.\n    `N` is a positive "
              "integer.\n";
    bool parsed = parse_single_value_option(num_streams, default_num_streams,
            utils::stoll_safe, str, option_name, help);
    if (parsed) {
        if (num_streams <= 0) {
            BENCHDNN_PRINT(
                    0, "%s\n", "Error: number of streams must be positive.");
            SAFE_V(FAIL);
        }
    }
    return parsed;
}

static bool parse_repeats_per_prb(
        const char *str, const std::string &option_name = "repeats-per-prb") {
    static const std::string help
            = "N    (Default: `1`)\n    Specifies the number of times to "
              "repeat testing of the problem.\n";
    bool parsed = parse_single_value_option(repeats_per_prb,
            default_repeats_per_prb, utils::stoll_safe, str, option_name, help);
    if (parsed) repeats_per_prb = MAX2(1, repeats_per_prb);
    return parsed;
}

static bool parse_mem_check(
        const char *str, const std::string &option_name = "mem-check") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to perform "
              "a device RAM capability check if a problem fits a device, when "
              "set to `true`.\n";
    return parse_single_value_option(
            mem_check, true, parsers::str2bool, str, option_name, help);
}

static bool parse_memory_kind(
        const char *str, const std::string &option_name = "memory-kind") {
    static const std::string help
            = "KIND    (Default: `usm`)\n    Specifies a memory `KIND` to test "
              "with DPC++ and OpenCL engines.\n    `KIND` values are `usm`, "
              "`buffer`, `usm_device` (malloc_device) or `usm_shared` "
              "(malloc_shared).\n";
    bool parsed = parse_single_value_option(memory_kind, default_memory_kind,
            str2memory_kind, str, option_name, help);

#if !defined(DNNL_WITH_SYCL) && DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL \
        && DNNL_GPU_RUNTIME != DNNL_RUNTIME_ZE
    if (parsed) {
        fprintf(stderr,
                "ERROR: option `--%s` is supported with DPC++ and OpenCL "
                "builds only, exiting...\n",
                option_name.c_str());
        exit(2);
    }
#endif
    return parsed;
}

// TODO: remove once the SYCL kernel compiler becomes thread-safe.
static mode_modifier_t drop_par_create(mode_modifier_t modifier) {
#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
    if (static_cast<bool>(modifier & mode_modifier_t::par_create)) {
        static bool reported = false;
        if (!reported) {
            BENCHDNN_PRINT(0, "%s\n",
                    "INFO: `P` mode modifier is ignored in the build with "
                    "DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER=ON.");
            reported = true;
        }
        modifier = static_cast<mode_modifier_t>(static_cast<unsigned>(modifier)
                & ~static_cast<unsigned>(mode_modifier_t::par_create));
    }
#endif
    return modifier;
}

static bool parse_mode(
        const char *str, const std::string &option_name = "mode") {
    static const std::string help
            = "MODE    (Default: `C`)\n"
              "    Specifies a `MODE` for benchmarking.\n"
              "    `MODE` values are:\n"
              "    * `L` for listing mode.\n"
              "    * `I` for initialization mode.\n"
              "    * `R` for execution mode (no correctness validation).\n"
              "    * `C` for correctness testing.\n"
              "    * `P` for performance testing.\n"
              "    * `F` for fast performance testing (GPU only).\n"
              "    * `S` for simulation performance testing.\n"
              "    * `B` for bitwise (numerical determinism) testing.\n"
              "    More details at "
            + doc_url + "benchdnn_general_info.md\n";

    const auto str2bench_mode = [](const std::string &_str) {
        bench_mode_t mode = default_bench_mode;
        if (_str.size() > 1) {
            BENCHDNN_PRINT(
                    0, "%s\n%s", "Error: mode value is invalid.", help.c_str());
            SAFE_V(FAIL);
        } else if (_str.size() == 1) {
            switch (_str[0]) {
                case 'l':
                case 'L': mode = bench_mode_t::list; break;
                case 'i':
                case 'I': mode = bench_mode_t::init; break;
                case 'r':
                case 'R':
                    mode = bench_mode_t::exec;
                    if (!utils::has_clang_sanitizers())
                        bench_mode_modifier |= mode_modifier_t::no_ref_memory;
                    break;
                case 'c':
                case 'C': mode = bench_mode_t::corr; break;
                case 'p':
                case 'P': mode = bench_mode_t::perf; break;
                case 'f':
                case 'F':
                    mode = bench_mode_t::perf_fast;
                    max_ms_per_prb = 10;
                    bench_mode_modifier |= mode_modifier_t::par_create
                            | mode_modifier_t::no_ref_memory;
                    break;
                case 'b':
                case 'B': mode = bench_mode_t::bitwise; break;
                case 's':
                case 'S':
                    mode = bench_mode_t::perf_sim;
                    fix_times_per_prb = 1;
                    bench_mode_modifier |= mode_modifier_t::no_ref_memory;
                    break;
                default:
                    BENCHDNN_PRINT(0, "%s\n%s", "Error: mode value is invalid.",
                            help.c_str());
                    SAFE_V(FAIL);
            }
        }

        return mode;
    };

    return parse_single_value_option(bench_mode, default_bench_mode,
            str2bench_mode, str, option_name, help);
}

static bool parse_mode_modifier(
        const char *str, const std::string &option_name = "mode-modifier") {
    static const std::string help
            = "MODIFIER    (Default: empty)\n"
              "    Specifies a `MODIFIER` for selected benchmarking mode.\n"
              "    `MODIFIER` values are:\n"
              "    * `P` to enable parallel test objects creation.\n"
              "          The flow will create as many primitives in parallel \n"
              "          as number of threads identified on the system \n"
              "          first, then execute them one by one.\n"
              "    * `M` to disable usage of reference memory.\n"
              "          It removes any overheads for mapping, unmapping and \n"
              "          reorders used in filling functions (disabled).\n"
              "    More details at "
            + doc_url + "benchdnn_general_info.md\n";

    const auto str2mode_modifier = [](const std::string &_str) {
        mode_modifier_t modifier = bench_mode_modifier;
        for (auto s : _str) {
            switch (s) {
                case 'p':
                case 'P': modifier |= mode_modifier_t::par_create; break;
                case 'm':
                case 'M': modifier |= mode_modifier_t::no_ref_memory; break;
                default:
                    BENCHDNN_PRINT(0, "%s\n%s",
                            "Error: modifier value is invalid.", help.c_str());
                    exit(2);
            }
        }

        return modifier;
    };

    return parse_single_value_option(bench_mode_modifier,
            default_bench_mode_modifier, str2mode_modifier, str, option_name,
            help);
}

static bool parse_start(
        const char *str, const std::string &option_name = "start") {
    static const std::string help
            = "UINT    (Default: `0`)\n    Specifies the test case index "
              "`UINT` to start execution. All test cases up to `UINT` will be "
              "skipped.\n";
    return parse_single_value_option(
            test_start, 0, utils::stoll_safe, str, option_name, help);
}

static bool parse_buffer_prefix(
        const char *str, const std::string &option_name = "buffer-prefix") {
    static const std::string help
            = "PREFIX    (Default: not specified)\n    Instructs the driver to "
              "fill specified buffers with the data from the files provided by "
              "`PREFIX`.\n    The expected filename format is "
              "\'PREFIX.ID.bin\'; the files must have binary data in them.\n   "
              " More details at "
            + doc_url + "knobs_common.md#--buffer-prefix\n";
    const auto str2self = [](const std::string &str) { return str; };
    return parse_single_value_option(
            buffer_prefix, std::string(), str2self, str, option_name, help);
}

static bool parse_stream_kind(
        const char *str, const std::string &option_name = "stream-kind") {
    static const std::string help
            = "KIND    (Default: `def`)\n    Specifies a stream `KIND` to test "
              "with DPC++ and OpenCL engines through stream flags.\n    "
              "`KIND` values are `def` (the default flags), `in_order`, or "
              "`out_of_order`.\n";
    bool parsed = parse_single_value_option(stream_kind, default_stream_kind,
            str2stream_kind, str, option_name, help);

#if !defined(DNNL_WITH_SYCL) && DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL \
        && DNNL_GPU_RUNTIME != DNNL_RUNTIME_ZE
    if (parsed) {
        BENCHDNN_PRINT(0,
                "Error: option `--%s` is supported with DPC++ and OpenCL "
                "builds only, exiting...\n",
                option_name.c_str());
        SAFE_V(FAIL);
    }
#endif
    return parsed;
}

static bool parse_summary(
        const char *str, const std::string &option_name = "summary") {
    static const std::string help
            = "STRING    (Default: `failures`)\n    Instructs benchdnn to "
              "print additional statistics and information based on the STRING "
              "values.\n";
    return parse_single_value_option(
            summary, summary_t(), parsers::str2summary, str, option_name, help);
}

static bool parse_verbose(
        const char *str, const std::string &option_name = "verbose") {
    static const std::string help
            = "UINT, -vUINT    (Default: `0`)\n    Instructs the driver to "
              "print additional information depending on `UINT`.\n    More "
              "details at "
            + doc_url + "knobs_verbose.md\n";
    bool parsed = parse_single_value_option(
            verbose, 0, utils::stoll_safe, str, option_name, help);
    if (parsed) return parsed;

    const std::string pattern("-v"); // check short option first
    if (utils::option_matched(pattern, str)) {
        verbose = utils::stoll_safe(str + pattern.size());
        return true;
    }
    return false;
}

static bool parse_execution_mode(
        const char *str, const std::string &option_name = "execution-mode") {
    static const std::string help
            = "MODE    (Default: direct)\n"
              "    Specifies a `MODE` of execution.\n"
              "    `MODE` values are:\n"
              "    * `direct` instruction the driver to execute the primitive "
              "directly.\n"
              "    * `graph` to execute the primitive using a graph backend.\n"
              "          Currently limited to the experimental SYCL Graph on "
              "DPC++ builds.\n"
              "    * `native_graph` to execute the primitive using a graph "
              "backend capturing command buffers directly.\n          "
              "Currently limited to the experimental SYCL Graph on DPC++ "
              "builds.\n";
    bool parsed
            = parse_single_value_option(execution_mode, default_execution_mode,
                    parsers::str2execution_mode, str, option_name, help);

#if !defined(DNNL_WITH_SYCL)
    // Default execution mode is legit for any build configuration.
    if (parsed && execution_mode != default_execution_mode) {
        BENCHDNN_PRINT(0,
                "Error: option `--%s` is supported with DPC++ "
                "builds only, exiting...\n",
                option_name.c_str());
        SAFE_V(FAIL);
    }
#endif
    return parsed;
}

bool parse_bench_settings(const char *str) {
    last_parsed_is_problem = false; // if start parsing, expect an option

    static bool start_msg = false, end_msg = false;
    if (!start_msg) {
        help_ss << "===================\n";
        help_ss << "= Global options: =\n";
        help_ss << "===================\n";
        help_ss << "(More technical details available at " + doc_url
                        + "knobs_common.md)\n\n";
        start_msg = true;
    }

    bool parsed = parse_allow_enum_tags_only(str)
            || parse_attr_same_pd_check(str) || parse_canonical(str)
            || parse_check_ref_impl(str) || parse_cold_cache(str)
            || parse_cpu_isa_hints(str) || parse_engine(str)
            || parse_fast_ref(str) || parse_fix_times_per_prb(str)
            || parse_global_impl(str) || parse_global_skip_impl(str)
            || parse_max_ms_per_prb(str) || parse_num_streams(str)
            || parse_repeats_per_prb(str) || parse_mem_check(str)
            || parse_memory_kind(str) || parse_mode(str)
            || parse_mode_modifier(str) || parse_start(str)
            || parse_stream_kind(str) || parse_summary(str)
            || parse_verbose(str) || parse_execution_mode(str)
            || parse_buffer_prefix(str);

    // Last condition makes this help message to be triggered once driver_name
    // is already known.
    if (!parsed && !end_msg && !driver_name.empty()) {
        help_ss << "===================\n";
        help_ss << "= Driver options: =\n";
        help_ss << "===================\n";
        help_ss << "(More technical details available at " + doc_url + "driver_"
                << driver_name << ".md)\n\n";
        end_msg = true;
    }

    if (parsed) bench_mode_modifier = drop_par_create(bench_mode_modifier);

    return parsed;
}

// Service functions
void catch_unknown_options(const char *str) {
    last_parsed_is_problem = true; // if reached, means problem parsing

    std::string pattern = "--";
    if (utils::option_matched(pattern, str)) {
        std::string s(str);
        auto equal_pos = s.find_first_of('=');
        s = s.substr(2, equal_pos - 2);

        // Try to provide similar options in case of minor typos.
        // The search is based on Levenshtein distance between two strings.
        //
        // Note: print options when there's some similarity, but limit it by 6
        // (taken experimentally) for long given options.
        const size_t good_dist = std::min(s.size() - 1, size_t(6));
        std::multimap<size_t, std::string> opt_distances;
        size_t n_candidates = 0;
        for (const auto &opt : utils::get_option_names()) {
            size_t dist
                    = utils::compute_distance(s, opt, /* reset_cache = */ true);
            opt_distances.emplace(dist, opt);
            if (dist <= good_dist) n_candidates++;
        }

        BENCHDNN_PRINT(0, "Error: unknown option for \'%s\' driver: \'%s\'.\n",
                driver_name.c_str(), str);
        if (n_candidates == 0) exit(2);

        BENCHDNN_PRINT(0, "The most similar %s\n",
                n_candidates == 1 ? "option is:" : "options are:");

        for (const auto &e : opt_distances) {
            if (e.first > good_dist) continue;
            BENCHDNN_PRINT(0, "        %s\n", e.second.c_str());
        }

        exit(2);
    }

    // Must stay after `--` check.
    pattern = "-";
    if (utils::option_matched(pattern, str)) {
        BENCHDNN_PRINT(0, "%s\n%s \'%s\'\n",
                "ERROR: options should be passed with `--` prefix.",
                "Given input:", str);
        exit(2);
    }
}

int parse_last_argument() {
    if (!last_parsed_is_problem)
        fprintf(stderr,
                "%s driver: WARNING: No problem found for a given option!\n",
                driver_name.c_str());
    return OK;
}

std::string get_substr(const std::string &s, size_t &start_pos, char delim,
        bool allow_dangling) {
    auto end_pos = s.find_first_of(delim, start_pos);
    auto sub = s.substr(start_pos, end_pos - start_pos);
    start_pos = end_pos + (end_pos != std::string::npos);
    if (!allow_dangling && start_pos == s.size()) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: dangling symbol at the end of input", s.c_str());
        SAFE_V(FAIL);
    }
    return sub;
}

} // namespace parser
