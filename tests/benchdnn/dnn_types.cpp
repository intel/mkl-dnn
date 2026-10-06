/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <assert.h>
#include <cctype>
#include <cmath>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>

#include "oneapi/dnnl/dnnl.h"

#include "src/common/math_utils.hpp"

#include "common.hpp"
#include "conv/conv_dw_fusion.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "utils/check.hpp"
#include "utils/cold_cache.hpp"
#include "utils/dims.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/execution_mode.hpp"
#include "utils/fill.hpp"
#include "utils/parser.hpp"
#include "utils/stream_kind.hpp"

namespace tag {
const char *x {"x"};
const char *abx {"abx"};
const char *axb {"axb"};
const char *any {"any"};
const char *undef {"undef"};
} // namespace tag

std::ostream &operator<<(std::ostream &s, dir_t dir) {
#define CASE(x) \
    if (dir == (x)) return s << STRINGIFY(x)
    CASE(FWD_B);
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(BWD_D);
    CASE(BWD_DW);
    CASE(BWD_W);
    CASE(BWD_WB);
#undef CASE
    SAFE_V(FAIL);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt) {
    s << dt2str(dt);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek) {
    s << engine_kind2str(ek);
    return s;
}

dnnl_prop_kind_t prop2prop_kind(const dir_t dir) {
    if (dir == FWD_I) return dnnl_forward_inference;
    if (dir & FLAG_FWD) return dnnl_forward_training;
    if (dir == BWD_DW) return dnnl_backward;
    assert(!"unknown dir");
    return dnnl_prop_kind_undef;
}

const char *prop2str(dnnl_prop_kind_t prop) {
    if (prop == dnnl_forward_training) return "FWD_D";
    if (prop == dnnl_forward_inference) return "FWD_I";
    if (prop == dnnl_backward) return "BWD_DW";
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

static const std::map<int, std::vector<const char *>> supported_args {
        {DNNL_ARG_SRC, {"src", "src0"}},
        {DNNL_ARG_SRC_1, {"src1"}},
        {DNNL_ARG_WEIGHTS, {"wei"}},
        {DNNL_ARG_DST, {"dst"}},
        {DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST, {"attr_post_op_dw_dst"}},
        {DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, {"attr_post_op_dw_wei"}},
};

int str2arg(const std::string &str) {
    for_(const auto &arg : supported_args)
    for (const auto &s : arg.second)
        if (str.compare(s) == 0) return arg.first;

    // multiple srcs
    std::string msrc = "msrc";
    if (str.compare(0, msrc.size(), msrc) == 0) {
        const auto &str_index = str.substr(msrc.size());
        if (str_index.empty() /* TODO: || non-digit string */) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: \'msrc\' argument requires index to be specified.");
            return DNNL_ARG_UNDEF;
        }
        const auto index = stoul(str_index);
        return DNNL_ARG_MULTIPLE_SRC + index;
    }
    return DNNL_ARG_UNDEF;
}

std::string arg2str(int arg) {
    if (supported_args.find(arg) != supported_args.end())
        return std::string(supported_args.at(arg)[0]);
    if (arg & DNNL_ARG_MULTIPLE_SRC) {
        std::string msrc("msrc");
        const int index = arg - DNNL_ARG_MULTIPLE_SRC;
        return msrc + std::to_string(index);
    }
    assert(!"unknown argument");
    return "unknown argument";
}

policy_t attr_t::str2policy(const std::string &str) {
    std::string s(str);
    // s.compare is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#define CASE(_plc) \
    if (s.compare(STRINGIFY(_plc)) == 0) return _plc
    CASE(COMMON);
    CASE(HOST_SCALAR);
    CASE(MX);
    CASE(DYNAMIC_FP);
    CASE(PER_OC);
    CASE(PER_OCIC);
    CASE(PER_DIM_0);
    CASE(PER_DIM_1);
    CASE(PER_DIM_01);
    CASE(PER_DIM_2);
    CASE(PER_DIM_3);
    CASE(PER_TENSOR);
#undef CASE
    assert(!"unknown attr_t::policy_t policy");
    return POLICY_TOTAL;
}

const char *attr_t::policy2str(policy_t policy) {
    if (policy == COMMON) return "common";
    if (policy == HOST_SCALAR) return "host_scalar";
    if (policy == MX) return "mx";
    if (policy == DYNAMIC_FP) return "dynamic_fp";
    if (policy == PER_OC) return "per_oc";
    if (policy == PER_OCIC) return "per_ocic";
    if (policy == PER_DIM_0) return "per_dim_0";
    if (policy == PER_DIM_1) return "per_dim_1";
    if (policy == PER_DIM_01) return "per_dim_01";
    if (policy == PER_DIM_2) return "per_dim_2";
    if (policy == PER_DIM_3) return "per_dim_3";
    if (policy == PER_TENSOR) return "per_tensor";
    assert(!"unknown attr_t::policy_t policy");
    return "unknown attr_t::policy_t policy";
}

dnnl_rounding_mode_t str2rounding_mode(const std::string &str) {
    // s.compare is lexicographical, case matters
    auto s = parser::utils::lowercase(str);
#define CASE(_rm) \
    if (s.compare(STRINGIFY(_rm)) == 0) return dnnl_rounding_mode_##_rm
    CASE(environment);
    CASE(stochastic);
#undef CASE
    assert(!"unknown attr_t::rounding_mode_t rounding_mode");
    return dnnl_rounding_mode_environment;
}

static int get_default_mask(policy_t policy, int ndims) {
    switch (policy) {
        case attr_t::policy_t::PER_DIM_0: return (1 << 0);
        case attr_t::policy_t::PER_OC:
        case attr_t::policy_t::PER_DIM_1: return (1 << 1);
        case attr_t::policy_t::PER_OCIC:
        case attr_t::policy_t::PER_DIM_01: return (1 << 0) + (1 << 1);
        case attr_t::policy_t::PER_DIM_2: return (1 << 2);
        case attr_t::policy_t::PER_DIM_3: return (1 << 3);
        case attr_t::policy_t::MX:
        case attr_t::policy_t::DYNAMIC_FP:
        case attr_t::policy_t::PER_TENSOR:
            assert(ndims > 0 && ndims <= DNNL_MAX_NDIMS);
            return (1 << ndims) - 1;
        case attr_t::policy_t::COMMON: return 0;
        // Note: use mask=0 for compatibility with reference compute paths.
        case attr_t::policy_t::HOST_SCALAR: return 0;
        default: SAFE(FAIL, CRIT); return INT_MIN;
    }
}

dnnl_quantization_mode_t attr_t::policy2quantization_mode(policy_t policy) {
    if (policy == policy_t::MX) return dnnl_quantization_mode_dynamic_mx;
    if (policy == policy_t::DYNAMIC_FP)
        return dnnl_quantization_mode_dynamic_fp;

    return dnnl_quantization_mode_static_sazp;
}

int attr_t::policy2mask(int arg, policy_t policy, int ndims,
        dnnl_primitive_kind_t prim_kind, bool has_groups) {

    if (policy == policy_t::HOST_SCALAR) { // shortcut
        return get_default_mask(policy, ndims);
    }

    // Handle of weights mask for various primitives.
    if (prim_kind == dnnl_convolution || prim_kind == dnnl_deconvolution
            || prim_kind == dnnl_inner_product) {
        if (arg != DNNL_ARG_WEIGHTS || policy == policy_t::COMMON)
            return get_default_mask(policy, ndims);

        switch (policy) {
            case PER_OC:
                if (has_groups)
                    return get_default_mask(PER_DIM_01, ndims);
                else
                    return get_default_mask(PER_DIM_0, ndims);
            default: SAFE(FAIL, CRIT); return -1;
        }
    } else if (prim_kind == dnnl_matmul) {
        if ((arg != DNNL_ARG_SRC && arg != DNNL_ARG_WEIGHTS
                    && arg != DNNL_ARG_DST)
                || policy == policy_t::COMMON)
            return get_default_mask(policy, ndims);

        if (ndims < 2) SAFE_V(FAIL);
        switch (policy) {
            case PER_DIM_1:
            case PER_OC: return (1 << (ndims - 1));
            case PER_OCIC: return (1 << (ndims - 1)) + (1 << (ndims - 2));
            case MX:
            case DYNAMIC_FP:
            case PER_TENSOR: return get_default_mask(policy, ndims);
            default: SAFE_V(FAIL); return -1;
        }
    } else if (prim_kind == dnnl_layer_normalization) {
        if (arg != DNNL_ARG_SRC_1 || policy != policy_t::PER_OC)
            return get_default_mask(policy, ndims);

        // PER_OC
        assert(policy == policy_t::PER_OC);
        if (ndims < 1) SAFE_V(FAIL);
        return 1 << (ndims - 1);
    } else {
        // Default case
        return get_default_mask(policy, ndims);
    }
}

// This function takes input string, extracts float value and asteriks, if
// present, from the string. Updates @value with extracted values.
int parse_value_and_runtime(float &value, const std::string &s) {
    // process value
    size_t scale_pos = 0;
    try {
        value = std::stof(s, &scale_pos);
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'.\n",
                "Error: scale or zero point input value is expected to be a "
                "real number. Given input:",
                s.c_str());
        SAFE(FAIL, WARN);
    }
    if (scale_pos != s.size()) {
        BENCHDNN_PRINT(0, "%s \'%s\'. %s \'%g\'.\n",
                "Error: not every input symbol was processed. Given input:",
                s.c_str(), "Parsed value:", value);
        SAFE(FAIL, WARN);
    }
    return OK;
}

#define HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING() \
    if (start_pos == std::string::npos) return OK; \
    if (start_pos >= s.size()) { \
        BENCHDNN_PRINT(0, "%s \'%s\'\n", \
                "Error: dangling symbol at the end of input", s.c_str()); \
        SAFE_V(FAIL); \
    }

int attr_t::arg_scales_t::entry_t::from_str(const std::string &s) {
    *this = arg_scales_t::entry_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    const auto mask_input_str = parser::get_substr(s, start_pos, ':');
    if (parser::utils::has_only_digits(mask_input_str)) {
        // If an input consists of digits only, then read it as the int value.
        this->mask_input = attr_t::mask_input_t::mask;
        this->mask = parser::utils::stoll_safe(mask_input_str);
    } else {
        // Otherwise, re-direct to policy parsing.
        this->mask_input = attr_t::mask_input_t::policy;
        this->policy = attr_t::str2policy(mask_input_str);
        if (this->policy == POLICY_TOTAL) {
            BENCHDNN_PRINT(0, "%s \'%s\' %s\n", "Error: Scale entry policy",
                    mask_input_str.c_str(), "is not recognized.");
            SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process scale value if a single value is expected
    if (this->has_single_element()) {
        SAFE(parse_value_and_runtime(
                     this->scale, parser::get_substr(s, start_pos, ':')),
                WARN);
        if (this->scale < 0) {
            BENCHDNN_PRINT(0, "%s \'%g\'\n",
                    "Error: the scale can't be negative:", this->scale);
            SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process data type
    const auto dt_str = parser::get_substr(s, start_pos, ':');
    this->dt = str2dt(dt_str.c_str());
    if (this->dt == dnnl_data_type_undef) {
        BENCHDNN_PRINT(0, "Error: data type \'%s\' was not recognized.\n",
                dt_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process groups
    const auto g_str = parser::get_substr(s, start_pos, ':');
    parser::parse_vector_str(
            this->groups, dims_t(), parser::utils::stoll_safe, g_str, 'x');

    if (!this->groups.empty() && this->groups.size() != 2) {
        BENCHDNN_PRINT(
                0, "%s\n", "Error: the number of groups is expected to be 2.");
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    return OK;
}

int attr_t::zero_points_t::entry_t::from_str(const std::string &s) {
    *this = zero_points_t::entry_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    const auto mask_input_str = parser::get_substr(s, start_pos, ':');
    if (parser::utils::has_only_digits(mask_input_str)) {
        // If an input consists of digits only, then read it as the int value.
        this->mask_input = attr_t::mask_input_t::mask;
        this->mask = parser::utils::stoll_safe(mask_input_str);
    } else {
        // Otherwise, re-direct to policy parsing.
        this->mask_input = attr_t::mask_input_t::policy;
        this->policy = attr_t::str2policy(mask_input_str);
        if (this->policy == POLICY_TOTAL) {
            BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                    "Error: Zero-point entry policy", mask_input_str.c_str(),
                    "is not recognized.");
            SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    if (this->has_single_element()) {
        float value = 0.0f;
        SAFE(parse_value_and_runtime(
                     value, parser::get_substr(s, start_pos, ':')),
                WARN);
        int zp_val = static_cast<int>(value);
        if (static_cast<float>(zp_val) != value) {
            BENCHDNN_PRINT(0, "%s \'%d\'\n",
                    "Error: the zero point is not exact:", zp_val);
            SAFE_V(FAIL);
        }
        this->value = zp_val;
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process data type
    const auto dt_str = parser::get_substr(s, start_pos, ':');
    this->dt = str2dt(dt_str.c_str());
    if (this->dt == dnnl_data_type_undef) {
        BENCHDNN_PRINT(0, "Error: data type \'%s\' was not recognized.\n",
                dt_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process groups
    const auto g_str = parser::get_substr(s, start_pos, ':');
    parser::parse_vector_str(
            this->groups, dims_t(), parser::utils::stoll_safe, g_str, 'x');
    if (!this->groups.empty() && this->groups.size() != 2) {
        BENCHDNN_PRINT(
                0, "%s\n", "Error: the number of groups is expected to be 2.");
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    return OK;
}

int attr_t::precomputed_reductions_t::entry_t::from_str(const std::string &s) {
    *this = precomputed_reductions_t::entry_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    const auto mask_input_str = parser::get_substr(s, start_pos, ':');
    if (parser::utils::has_only_digits(mask_input_str)) {
        // If an input consists of digits only, then read it as the int value.
        this->mask_input = attr_t::mask_input_t::mask;
        this->mask = parser::utils::stoll_safe(mask_input_str);
    } else {
        // Otherwise, re-direct to policy parsing.
        this->mask_input = attr_t::mask_input_t::policy;
        this->policy = attr_t::str2policy(mask_input_str);
        if (this->policy == POLICY_TOTAL
                || this->policy != policy_t::PER_TENSOR) {
            BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                    "Error: Precomputed reductions entry policy",
                    mask_input_str.c_str(), "is not recognized.");
            SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process data type
    const auto dt_str = parser::get_substr(s, start_pos, ':');
    this->dt = str2dt(dt_str.c_str());
    if (this->dt == dnnl_data_type_undef) {
        BENCHDNN_PRINT(0, "Error: data type \'%s\' was not recognized.\n",
                dt_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process groups
    const auto g_str = parser::get_substr(s, start_pos, ':');
    parser::parse_vector_str(
            this->groups, dims_t(), parser::utils::stoll_safe, g_str, 'x');
    if (!this->groups.empty() && this->groups.size() != 2) {
        BENCHDNN_PRINT(
                0, "%s\n", "Error: the number of groups is expected to be 2.");
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    return OK;
}

#undef HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING

int attr_t::zero_points_t::from_str(const std::string &s) {
    *this = zero_points_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        // TODO: split errors into different messages.
        if (arg == DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= subs.size()) {
            BENCHDNN_PRINT(0,
                    "Error: argument name \'%s\' was not recognized.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }

        zero_points_t::entry_t zero_point;
        SAFE(zero_point.from_str(parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, zero_point);
    }
    return OK;
}

int attr_t::precomputed_reductions_t::from_str(const std::string &s) {
    *this = precomputed_reductions_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == DNNL_ARG_UNDEF) {
            BENCHDNN_PRINT(0,
                    "Error: argument name \'%s\' was not recognized.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }
        if (subs_pos == std::string::npos) {
            BENCHDNN_PRINT(0,
                    "Error: not enough arguments were provided for the input "
                    "\'%s\'.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }
        if (subs_pos >= subs.size()) {
            BENCHDNN_PRINT(0,
                    "Error: dangling character in the input \'%s\' was "
                    "identified.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }

        precomputed_reductions_t::entry_t pr;
        SAFE(pr.from_str(parser::get_substr(subs, subs_pos, '\0')), WARN);
        set(arg, pr);
    }
    return OK;
}

int attr_t::arg_scales_t::from_str(const std::string &s) {
    *this = arg_scales_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        // Special handling for really big float values
        if (subs.back() == 'e') {
            auto subs_add = parser::get_substr(s, start_pos, '+');
            subs += subs_add;
        }
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        // TODO: split errors into different messages.
        if (arg == DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= s.size()) {
            BENCHDNN_PRINT(0,
                    "Error: argument name \'%s\' was not recognized.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }

        arg_scales_t::entry_t arg_scale;
        SAFE(arg_scale.from_str(parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, arg_scale);
    }
    return OK;
}

int attr_t::zero_points_t::get_mask(int arg, dnnl_primitive_kind_t prim_kind,
        int ndims, bool has_groups) const {
    const auto &e = get(arg);
    switch (e.mask_input) {
        // `none` is treated as `policy_t::COMMON` for convenience of calling
        // `get_mask` without extra guards.
        case mask_input_t::none: return 0;
        case mask_input_t::mask: return e.mask;
        case mask_input_t::policy:
            return attr_t::policy2mask(
                    arg, e.policy, ndims, prim_kind, has_groups);
        default: assert(!"unknown mask_input value"); break;
    }
    return INT_MIN;
}

int attr_t::precomputed_reductions_t::get_mask(int arg,
        dnnl_primitive_kind_t prim_kind, int ndims, bool has_groups) const {
    const auto &e = get(arg);
    switch (e.mask_input) {
        // `none` is treated as `policy_t::COMMON` for convenience of calling
        // `get_mask` without extra guards.
        case mask_input_t::none: return 0;
        case mask_input_t::mask: return e.mask;
        case mask_input_t::policy:
            return attr_t::policy2mask(
                    arg, e.policy, ndims, prim_kind, has_groups);
        default: assert(!"unknown mask_input value"); break;
    }
    return INT_MIN;
}

int attr_t::arg_scales_t::get_mask(int arg, dnnl_primitive_kind_t prim_kind,
        int ndims, bool has_groups) const {
    const auto &e = get(arg);
    switch (e.mask_input) {
        // `none` is treated as `policy_t::COMMON` for convenience of calling
        // `get_mask` without extra guards.
        case mask_input_t::none: return 0;
        case mask_input_t::mask: return e.mask;
        case mask_input_t::policy:
            return attr_t::policy2mask(
                    arg, e.policy, ndims, prim_kind, has_groups);
        default: assert(!"unknown mask_input value"); break;
    }
    return INT_MIN;
}

using pk_t = attr_t::post_ops_t::kind_t;

struct po_table_entry_t {
    pk_t kind;
    std::vector<std::string> kind_names;
    dnnl_alg_kind_t dnnl_kind;
};

static po_table_entry_t kind_table[] = {
        // sum
        {pk_t::SUM, {"sum"}, dnnl_alg_kind_undef},
        // depthwise convolution
        {pk_t::DW, {"dw"}, dnnl_convolution_direct},
        // eltwise
        {pk_t::ELTWISE_START, {"eltwise_undef"}, dnnl_alg_kind_undef},
        {pk_t::ABS, {"abs", "eltwise_abs"}, dnnl_eltwise_abs},
        {pk_t::CLIP, {"clip", "eltwise_clip"}, dnnl_eltwise_clip},
        {pk_t::CLIP_V2, {"clip_v2", "eltwise_clip_v2"}, dnnl_eltwise_clip_v2},
        {pk_t::CLIP_V2_DST, {"clip_v2_dst", "eltwise_clip_v2_use_dst_for_bwd"},
                dnnl_eltwise_clip_v2_use_dst_for_bwd},
        {pk_t::ELU, {"elu", "eltwise_elu"}, dnnl_eltwise_elu},
        {pk_t::ELU_DST, {"elu_dst", "eltwise_elu_use_dst_for_bwd"},
                dnnl_eltwise_elu_use_dst_for_bwd},
        {pk_t::EXP, {"exp", "eltwise_exp"}, dnnl_eltwise_exp},
        {pk_t::EXP_DST, {"exp_dst", "eltwise_exp_use_dst_for_bwd"},
                dnnl_eltwise_exp_use_dst_for_bwd},
        {pk_t::GELU_ERF, {"gelu_erf", "eltwise_gelu_erf"},
                dnnl_eltwise_gelu_erf},
        {pk_t::GELU_TANH, {"gelu_tanh", "eltwise_gelu_tanh"},
                dnnl_eltwise_gelu_tanh},
        {pk_t::HARDSIGMOID, {"hardsigmoid", "eltwise_hardsigmoid"},
                dnnl_eltwise_hardsigmoid},
        {pk_t::HARDSWISH, {"hardswish", "eltwise_hardswish"},
                dnnl_eltwise_hardswish},
        {pk_t::LINEAR, {"linear", "eltwise_linear"}, dnnl_eltwise_linear},
        {pk_t::LOG, {"log", "eltwise_log"}, dnnl_eltwise_log},
        {pk_t::LOGISTIC, {"logistic", "eltwise_logistic"},
                dnnl_eltwise_logistic},
        {pk_t::LOGISTIC_DST,
                {"logistic_dst", "eltwise_logistic_use_dst_for_bwd"},
                dnnl_eltwise_logistic_use_dst_for_bwd},
        {pk_t::MISH, {"mish", "eltwise_mish"}, dnnl_eltwise_mish},
        {pk_t::POW, {"pow", "eltwise_pow"}, dnnl_eltwise_pow},
        {pk_t::RELU, {"relu", "eltwise_relu"}, dnnl_eltwise_relu},
        {pk_t::RELU_DST, {"relu_dst", "eltwise_relu_use_dst_for_bwd"},
                dnnl_eltwise_relu_use_dst_for_bwd},
        {pk_t::ROUND, {"round", "eltwise_round"}, dnnl_eltwise_round},
        {pk_t::SQRT, {"sqrt", "eltwise_sqrt"}, dnnl_eltwise_sqrt},
        {pk_t::SQRT_DST, {"sqrt_dst", "eltwise_sqrt_use_dst_for_bwd"},
                dnnl_eltwise_sqrt_use_dst_for_bwd},
        {pk_t::SQUARE, {"square", "eltwise_square"}, dnnl_eltwise_square},
        {pk_t::SRELU, {"soft_relu", "eltwise_soft_relu", "srelu"},
                dnnl_eltwise_soft_relu},
        {pk_t::SWISH, {"swish", "eltwise_swish"}, dnnl_eltwise_swish},
        {pk_t::TANH, {"tanh", "eltwise_tanh"}, dnnl_eltwise_tanh},
        {pk_t::TANH_DST, {"tanh_dst", "eltwise_tanh_use_dst_for_bwd"},
                dnnl_eltwise_tanh_use_dst_for_bwd},
        {pk_t::ELTWISE_END, {"eltwise_undef"}, dnnl_alg_kind_undef},
        // binary
        {pk_t::BINARY_START, {"binary_undef"}, dnnl_alg_kind_undef},
        {pk_t::ADD, {"add", "binary_add"}, dnnl_binary_add},
        {pk_t::DIV, {"div", "binary_div"}, dnnl_binary_div},
        {pk_t::EQ, {"eq", "binary_eq"}, dnnl_binary_eq},
        {pk_t::GE, {"ge", "binary_ge"}, dnnl_binary_ge},
        {pk_t::GT, {"gt", "binary_gt"}, dnnl_binary_gt},
        {pk_t::LE, {"le", "binary_le"}, dnnl_binary_le},
        {pk_t::LT, {"lt", "binary_lt"}, dnnl_binary_lt},
        {pk_t::MAX, {"max", "binary_max"}, dnnl_binary_max},
        {pk_t::MIN, {"min", "binary_min"}, dnnl_binary_min},
        {pk_t::MUL, {"mul", "binary_mul"}, dnnl_binary_mul},
        {pk_t::MUL_INPLACE, {"mul_inplace", "binary_mul_inplace"},
                dnnl_binary_mul_inplace},
        {pk_t::NE, {"ne", "binary_ne"}, dnnl_binary_ne},
        {pk_t::SELECT, {"select", "binary_select"}, dnnl_binary_select},
        {pk_t::SUB, {"sub", "binary_sub"}, dnnl_binary_sub},
        {pk_t::BINARY_END, {"binary_undef"}, dnnl_alg_kind_undef},
        // prelu
        {pk_t::PRELU, {"prelu"}, dnnl_alg_kind_undef},
        // guard entry
        {pk_t::KIND_TOTAL, {"kind_undef"}, dnnl_alg_kind_undef}};

pk_t attr_t::post_ops_t::str2kind(const std::string &str) {
    // string::operator== is lexicographical, case matters
    auto s = parser::utils::lowercase(str);
    for (const auto &e : kind_table) {
        for (const auto &name : e.kind_names) {
            if (s == name) return e.kind;
        }
    }
    BENCHDNN_PRINT(0, "%s\'%s\' %s\n", "Error: ", str.c_str(),
            "kind of post operation entry was not recognized.");

    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind;
}

const char *attr_t::post_ops_t::kind2str(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.kind_names[0].c_str();
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind_names[0].c_str();
}

dnnl_alg_kind_t attr_t::post_ops_t::kind2dnnl_kind(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.dnnl_kind;
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].dnnl_kind;
}

std::vector<std::pair<int, int>> attr_t::post_ops_t::get_po_masks(
        int ndims, dnnl_primitive_kind_t prim_kind) const {
    std::vector<std::pair<int, int>> v_masks;
    for (int idx = 0; idx < len(); ++idx) {
        const auto &e = this->entry[idx];
        int mask = INT_MIN;
        int arg = DNNL_ARG_UNDEF;
        if (e.is_binary_kind_inplace()) {
            // No values to prepare: `maybe_post_ops` takes the right-hand side
            // straight from the destination.
            continue;
        } else if (e.is_binary_kind()) {
            switch (e.binary.mask_input) {
                // `none` is treated as `policy_t::COMMON` for convenience of
                // calling reference compute paths.
                case attr_t::mask_input_t::none: mask = 0; break;
                case attr_t::mask_input_t::mask: mask = e.binary.mask; break;
                case attr_t::mask_input_t::policy:
                    mask = attr_t::policy2mask(
                            DNNL_ARG_SRC_1, e.binary.policy, ndims, prim_kind);
                    break;
                default: assert(!"unknown mask_input value"); break;
            }
            arg = DNNL_ARG_SRC_1;
        } else if (e.is_prelu_kind()) {
            switch (e.prelu.mask_input) {
                // `none` is treated as `policy_t::COMMON` for convenience of
                // calling reference compute paths.
                case attr_t::mask_input_t::none: mask = 0; break;
                case attr_t::mask_input_t::mask: mask = e.prelu.mask; break;
                case attr_t::mask_input_t::policy:
                    mask = attr_t::policy2mask(
                            DNNL_ARG_WEIGHTS, e.prelu.policy, ndims, prim_kind);
                    break;
                default: assert(!"unknown mask_input value"); break;
            }
            arg = DNNL_ARG_WEIGHTS;
        } else
            continue;

        assert(mask >= 0);
        v_masks.emplace_back(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | arg, mask);

        // Condition (src2) mask for the ternary select input, derived from its
        // policy so a broadcast condition is filled accordingly. With no
        // explicit src2 spec the condition spans the full dst shape (mask -1),
        // preserving prior behavior.
        if (e.is_binary_kind_with_ternary_op()) {
            int src2_mask = -1;
            switch (e.binary.src2_mask_input) {
                case attr_t::mask_input_t::none: src2_mask = -1; break;
                case attr_t::mask_input_t::mask:
                    src2_mask = e.binary.src2_mask;
                    break;
                case attr_t::mask_input_t::policy:
                    src2_mask = attr_t::policy2mask(DNNL_ARG_SRC_2,
                            e.binary.src2_policy, ndims, prim_kind);
                    break;
                default: assert(!"unknown mask_input value"); break;
            }
            v_masks.emplace_back(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_2,
                    src2_mask);
        }
    }
    return v_masks;
}

bool attr_t::is_def(bool skip_fpmath, bool skip_acc_mode) const {
    return scales.is_def() && zero_points.is_def() && post_ops.is_def()
            && scratchpad_mode == get_default_scratchpad_mode()
            && IMPLICATION(!skip_fpmath, fpmath_mode.is_def())
            && IMPLICATION(
                    !skip_acc_mode, acc_mode == dnnl_accumulation_mode_strict)
            && rounding_mode.is_def() && deterministic.is_def()
            && dropout.is_def();
}

int attr_t::post_ops_t::find(pk_t kind, int start, int stop) const {
    if (stop == -1) stop = len();
    stop = MIN2(stop, len());
    for (int idx = start; idx < stop; ++idx)
        if (entry[idx].kind == kind) return idx;
    return -1;
}

bool attr_t::post_ops_t::entry_t::is_sum_kind() const {
    return kind == SUM;
}
bool attr_t::post_ops_t::entry_t::is_convolution_kind() const {
    return kind == DW;
}
bool attr_t::post_ops_t::entry_t::is_eltwise_kind() const {
    return kind > ELTWISE_START && kind < ELTWISE_END;
}
bool attr_t::post_ops_t::entry_t::is_binary_kind() const {
    return kind > pk_t::BINARY_START && kind < pk_t::BINARY_END;
}
bool attr_t::post_ops_t::entry_t::is_binary_kind_with_ternary_op() const {
    return kind == pk_t::SELECT;
}
bool attr_t::post_ops_t::entry_t::is_binary_kind_inplace() const {
    return kind == pk_t::MUL_INPLACE;
}
bool attr_t::post_ops_t::entry_t::is_prelu_kind() const {
    return kind == PRELU;
}

int attr_t::post_ops_t::convolution_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_convolution_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::eltwise_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_eltwise_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::binary_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_binary_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::prelu_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_prelu_kind()) return i;
    }
    return -1;
}

bool attr_t::post_ops_t::has_inplace_binary() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_binary_kind_inplace()) return true;
    }
    return false;
}

dnnl_scratchpad_mode_t attr_t::get_default_scratchpad_mode() {
    if (has_bench_mode_modifier(mode_modifier_t::par_create)
            || execution_mode == execution_mode_t::native_graph) {
        return dnnl_scratchpad_mode_user;
    }
    return dnnl_scratchpad_mode_library;
}

std::ostream &operator<<(std::ostream &s, const policy_t &policy) {
    s << attr_t::policy2str(policy);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::arg_scales_t::entry_t &scale) {
    using ::operator<<;

    if (scale.mask_input == attr_t::mask_input_t::mask) {
        s << scale.mask;
    } else {
        assert(scale.mask_input == attr_t::mask_input_t::policy);
        s << scale.policy;
    }
    if (scale.has_single_element()) s << ":" << scale.scale;
    if (scale.dt != dnnl_f32 || !scale.groups.empty()) s << ':' << scale.dt;
    if (!scale.groups.empty()) s << ":" << dims2str(scale.groups);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points) {
    using ::operator<<;

    const char *delim = "";
    for (const auto &point : zero_points.points) {
        s << delim << arg2str(point.first);
        if (point.second.mask_input == attr_t::mask_input_t::mask) {
            s << ":" << point.second.mask;
        } else {
            assert(point.second.mask_input == attr_t::mask_input_t::policy);
            s << ":" << point.second.policy;
        }
        if (point.second.has_single_element()) s << ":" << point.second.value;
        if (point.second.dt != dnnl_s32 || !point.second.groups.empty())
            s << ':' << point.second.dt;
        if (!point.second.groups.empty())
            s << ":" << dims2str(point.second.groups);
        delim = "+";
    }

    return s;
}

std::ostream &operator<<(std::ostream &s,
        const attr_t::precomputed_reductions_t &precomputed_reductions) {
    using ::operator<<;

    const char *delim = "";
    for (const auto &entry : precomputed_reductions.entries) {
        s << delim << arg2str(entry.first);
        if (entry.second.mask_input == attr_t::mask_input_t::mask) {
            s << ":" << entry.second.mask;
        } else {
            assert(entry.second.mask_input == attr_t::mask_input_t::policy);
            s << ":" << entry.second.policy;
        }
        if (entry.second.dt != dnnl_s32 || !entry.second.groups.empty())
            s << ':' << entry.second.dt;
        if (!entry.second.groups.empty())
            s << ":" << dims2str(entry.second.groups);
        delim = "+";
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales) {
    const char *delim = "";
    for (const auto &v : scales.scales) {
        if (!v.second.is_def()) {
            s << delim;
            s << arg2str(v.first) << ":" << v.second;
            delim = "+";
        }
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k) {
    s << attr_t::post_ops_t::kind2str(k);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (idx > 0) s << "+";

        const auto &e = post_ops.entry[idx];
        s << e.kind;

        if (e.is_sum_kind()) {
            if (e.sum.scale != 1.0f || e.sum.zero_point != 0
                    || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.scale;
            if (e.sum.zero_point != 0 || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.zero_point;
            if (e.sum.dt != dnnl_data_type_undef) s << ":" << e.sum.dt;
        } else if (e.is_convolution_kind()) {
            s << ":k" << e.convolution.kernel << "s" << e.convolution.stride
              << "p" << e.convolution.padding;
            if (e.convolution.dst_dt != dnnl_f32)
                s << ":" << e.convolution.dst_dt;
        } else if (e.is_eltwise_kind()) {
            if (e.eltwise.beta != 0.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta;
            else if (e.eltwise.alpha != 0.f)
                s << ":" << e.eltwise.alpha;
        } else if (e.is_binary_kind()) {
            s << ":" << e.binary.src1_dt;
            const auto src_delim
                    = e.is_binary_kind_with_ternary_op() ? "." : ":";

            if (e.binary.mask_input != attr_t::mask_input_t::none
                    || e.binary.tag != tag::any) {
                if (e.binary.mask_input == attr_t::mask_input_t::mask) {
                    s << src_delim << e.binary.mask;
                } else {
                    assert(e.binary.mask_input == attr_t::mask_input_t::policy);
                    s << src_delim << e.binary.policy;
                }
            }
            if (e.binary.tag != tag::any) s << src_delim << e.binary.tag;
            if (!e.binary.strides.empty())
                s << src_delim << dims2str(e.binary.strides);
            if (e.binary.grouped) s << src_delim << "grouped";

            if (e.is_binary_kind_with_ternary_op()) {
                bool delim_added = false;

                if (e.binary.src2_mask_input != attr_t::mask_input_t::none
                        || e.binary.src2_tag != tag::any) {
                    s << ":";
                    delim_added = true;
                    if (e.binary.src2_mask_input
                            == attr_t::mask_input_t::mask) {
                        s << e.binary.src2_mask;
                    } else {
                        assert(e.binary.src2_mask_input
                                == attr_t::mask_input_t::policy);
                        s << e.binary.src2_policy;
                    }
                }
                if (e.binary.src2_tag != tag::any) {
                    s << (delim_added ? src_delim : ":") << e.binary.src2_tag;
                }
            }
        } else if (e.is_prelu_kind()) {
            if (e.prelu.mask_input == attr_t::mask_input_t::mask) {
                if (e.prelu.mask > 0) s << ':' << e.prelu.mask;
            } else if (e.prelu.mask_input == attr_t::mask_input_t::policy) {
                if (e.prelu.policy != policy_t::COMMON)
                    s << ':' << e.prelu.policy;
            }
        } else {
            assert(!"unknown kind");
            s << "unknown_kind";
        }
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm) {
    s << scratchpad_mode2str(sm);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::fpmath_mode_t &fm) {
    s << fpmath_mode2str(fm.mode);
    if (fm.apply_to_int) s << ":" << bool2str(fm.apply_to_int);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_accumulation_mode_t am) {
    s << accumulation_mode2str(am);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::rounding_mode_t &rm) {
    std::string sep;
    for (const auto &i : rm.rounding_modes_) {
        s << sep << arg2str(i.first) << ":" << rounding_mode2str(i.second);
        if (rm.is_set_seed) s << ":" << rm.seed;
        sep = "+";
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::deterministic_t &d) {
    s << bool2str(d.enabled);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::dropout_t &drop) {
    // Update the string in backward direction to prevent multiple nesting
    // condition for each next dumping part.
    std::string str, str_tmp;
    if (drop.use_host_scalars) {
        str_tmp = ":" + std::to_string(drop.use_host_scalars);
        str = str_tmp + str;
    }
    if (!str.empty() || drop.offset != 0) {
        str_tmp = ":" + std::to_string(drop.offset);
        str = str_tmp + str;
    }
    if (!str.empty() || drop.tag != tag::any) {
        str_tmp = ":" + drop.tag;
        str = str_tmp + str;
    }
    if (!str.empty() || drop.seed != 0) {
        str_tmp = ":" + std::to_string(drop.seed);
        str = str_tmp + str;
    }
    // Use ostream to dump `p` for nicer output.
    s << drop.p << str;
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.is_def()) {
        if (!attr.scales.is_def()) s << "--attr-scales=" << attr.scales << " ";
        if (!attr.zero_points.is_def())
            s << "--attr-zero-points=" << attr.zero_points << " ";
        if (!attr.precomputed_reductions.is_def())
            s << "--attr-precomputed-reductions=" << attr.precomputed_reductions
              << " ";
        if (!attr.post_ops.is_def())
            s << "--attr-post-ops=" << attr.post_ops << " ";
        if (attr.scratchpad_mode != attr_t::get_default_scratchpad_mode())
            s << "--attr-scratchpad=" << attr.scratchpad_mode << " ";
        if (!attr.fpmath_mode.is_def())
            s << "--attr-fpmath=" << attr.fpmath_mode << " ";
        if (attr.acc_mode != dnnl_accumulation_mode_strict)
            s << "--attr-acc-mode=" << attr.acc_mode << " ";
        if (!attr.rounding_mode.is_def())
            s << "--attr-rounding-mode=" << attr.rounding_mode << " ";
        if (!attr.deterministic.is_def())
            s << "--attr-deterministic=" << attr.deterministic << " ";
        if (!attr.dropout.is_def())
            s << "--attr-dropout=" << attr.dropout << " ";
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_sparse_encoding_t se) {
    s << sparse_encoding2str(se);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const sparse_options_t &sparse_options) {
    if (!sparse_options.is_def()) {
        const std::vector<int> args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};

        bool has_grouped = false;
        for (const int arg : args) {
            if (sparse_options.is_grouped(arg)) {
                has_grouped = true;
                // Output format: --grouped=0:8:32+64+...
                int var_idx = sparse_options.get_variable_dim_idx(arg);
                s << "--grouped=" << var_idx << ":"
                  << sparse_options.get_group_count() << ":";
                using gd_t = sparse_options_t::grouped_data_t;
                const auto profile = sparse_options.get_profile(arg);
                if (profile.kind != gd_t::profile_none) {
                    s << (profile.kind == gd_t::profile_balanced ? "balanced"
                                    : profile.kind == gd_t::profile_hot
                                    ? "hot"
                                    : "decode");
                    // optional param is attached via "."
                    if (profile.param >= 0) s << "." << profile.param;
                } else {
                    const auto &dims = sparse_options.get_group_sizes(arg);
                    for (size_t i = 0; i < dims.size(); i++) {
                        s << dims[i];
                        if (i != dims.size() - 1) s << "+";
                    }
                }
                // optional hint follows after next ":"
                const auto hint = sparse_options.get_max_variable_dim();
                if (hint > 0) s << ":" << hint;
                s << " ";
                break;
            }
        }
        if (!has_grouped) {
            // Rest of sparse options use different format: --encoding=...
            s << "--encoding=";
            for (int i = 0; i < (int)args.size(); i++) {
                const int arg = args[i];
                if (!sparse_options.is_encoding_def(arg)) {
                    s << sparse_options.get_encoding(arg);
                    if (!sparse_options.is_sparsity_def(arg))
                        s << "+" << sparse_options.get_sparsity(arg);
                }
                if (i != (int)args.size() - 1)
                    s << ":";
                else
                    s << " ";
            }
        }
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, memory_kind_ext_t memory_kind) {
    switch (memory_kind) {
        case memory_kind_ext_t::usm: s << "usm"; break;
        case memory_kind_ext_t::buffer: s << "buffer"; break;
        case memory_kind_ext_t::usm_device: s << "usm_device"; break;
        case memory_kind_ext_t::usm_shared: s << "usm_shared"; break;
        default: assert(!"unexpected"); break;
    }
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    // Need to dump mode and modifiers in front of the driver name to make all
    // updated default values (at least, scratchpad) take effect before parsing
    // a state of a problem.
    if (canonical || bench_mode != default_bench_mode)
        s << "--mode=" << bench_mode << " ";
    // Don't dump modifiers if F or S mode is used to keep the repro simple.
    if (canonical
            || (bench_mode != bench_mode_t::perf_fast
                    && bench_mode != bench_mode_t::perf_sim
                    && bench_mode_modifier != default_bench_mode_modifier))
        s << "--mode-modifier=" << bench_mode_modifier << " ";
    // Don't dump max_ms_per_prb if F mode is used to keep the repro simple.
    if (canonical
            || (bench_mode != bench_mode_t::perf_fast
                    && max_ms_per_prb != default_max_ms_per_prb))
        s << "--max-ms-per-prb=" << max_ms_per_prb << " ";
    // Don't dump fix_times_per_prb if S mode is used to keep the repro simple.
    if (canonical
            || (bench_mode != bench_mode_t::perf_sim
                    && fix_times_per_prb != default_fix_times_per_prb))
        s << "--fix-times-per-prb=" << fix_times_per_prb << " ";

    // Need to dump execution_mode in front of the driver name to make all
    // updated default values (scratchpad) take effect before parsing a state of
    // a problem.
    if (canonical || execution_mode != default_execution_mode)
        s << "--execution-mode=" << execution_mode << " ";

    if (canonical) s << "--canonical=" << bool2str(canonical) << " ";
    if (canonical || engine_tgt_kind != dnnl_cpu) {
        s << "--engine=" << engine_tgt_kind;
        if (engine_index != 0) s << ":" << engine_index;
        s << " ";
    }
    if (canonical || fast_ref != default_fast_ref)
        s << "--fast-ref=" << bool2str(fast_ref) << " ";
    if (!skip_impl.empty()) s << "--skip-impl=" << skip_impl << " ";
    if (canonical || mem_check != true)
        s << "--mem-check=" << bool2str(mem_check) << " ";
    if (canonical || allow_enum_tags_only != true)
        s << "--allow-enum-tags-only=" << bool2str(allow_enum_tags_only) << " ";
    if (canonical || hints.get() != isa_hints_t::none)
        s << "--cpu-isa-hints=" << isa_hints_t::hints2str(hints) << " ";
    if (canonical || attr_same_pd_check != false)
        s << "--attr-same-pd-check=" << bool2str(attr_same_pd_check) << " ";
    if (canonical || check_ref_impl != default_runtime_kind)
        s << "--check-ref-impl=" << check_ref_impl << " ";
#if defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    if (canonical || memory_kind != default_memory_kind)
        s << "--memory-kind=" << memory_kind << " ";
    if (canonical || stream_kind != default_stream_kind)
        s << "--stream-kind=" << stream_kind << " ";
#endif
    if (canonical || cold_cache_input != default_cold_cache_input())
        s << "--cold-cache=" << cold_cache_input << " ";
    if (canonical || !buffer_prefix.empty())
        s << "--buffer-prefix=" << buffer_prefix << " ";

    s << "--" << driver_name << " ";

    return s;
}

dnnl_engine_kind_t str2engine_kind(const std::string &s) {
    if (s == "cpu") {
        return dnnl_cpu;
    } else if (s == "gpu") {
        return dnnl_gpu;
    } else {
        BENCHDNN_PRINT(0,
                "Error: engine kind supports values \'cpu\' and \'gpu\' only. "
                "Given input: %s\n",
                s.c_str());
        SAFE_V(FAIL);
    }
    return dnnl_any_engine;
}

dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str) {
    const char *param = "library";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_library;

    param = "user";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_user;

    assert(!"not expected");
    return attr_t::get_default_scratchpad_mode();
}

dnnl_fpmath_mode_t str2fpmath_mode(const char *str) {
    if (std::strcmp(str, "") == 0) {
        dnnl_fpmath_mode_t ret;
        dnnl_get_default_fpmath_mode(&ret);
        return ret;
    }

#define CASE(fpm) \
    param = #fpm; \
    if (!strncasecmp(param, str, strlen(param))) return dnnl_fpmath_mode_##fpm;

    const char *param;

    CASE(strict);
    CASE(bf16);
    CASE(f16);
    CASE(tf32);
    CASE(any);

    assert(!"not expected");
    return dnnl_fpmath_mode_strict;

#undef CASE
}

dnnl_accumulation_mode_t str2accumulation_mode(const char *str) {

#define CASE(am) \
    param = #am; \
    if (!strncasecmp(param, str, strlen(param))) \
        return dnnl_accumulation_mode_##am;

    const char *param;

    CASE(strict);
    CASE(relaxed);
    CASE(any);
    CASE(f32);
    CASE(s32);
    CASE(f16);

    assert(!"not expected");
    return dnnl_accumulation_mode_strict;

#undef CASE
}

struct post_ops_rhs_tensor_entry_t {
    dnnl_data_type_t dt;
    int mask;
    std::string tag;
    dims_t strides;
    int arg_attr_mask;
};

namespace {

post_ops_rhs_tensor_entry_t get_po_rhs_tensor_entry(
        const attr_t::post_ops_t::entry_t &entry, int ndims,
        dnnl_primitive_kind_t prim_kind) {
    if (entry.is_prelu_kind()) {
        const auto &prelu = entry.prelu;
        int mask = INT_MIN;
        switch (prelu.mask_input) {
            // `none` is treated as `policy_t::COMMON` for convenience of
            // calling `get_po_rhs_tensor_entry` without extra guards.
            case attr_t::mask_input_t::none: mask = 0; break;
            case attr_t::mask_input_t::mask: mask = prelu.mask; break;
            case attr_t::mask_input_t::policy:
                mask = attr_t::policy2mask(
                        DNNL_ARG_WEIGHTS, prelu.policy, ndims, prim_kind);
                break;
            default: assert(!"unknown mask_input value"); break;
        }
        return {dnnl_f32, mask, tag::axb, dims_t(), DNNL_ARG_WEIGHTS};
    } else if (entry.is_binary_kind()) {
        if (entry.is_binary_kind_inplace()) {
            // An in-place post-op aliases the destination, so its descriptor
            // spans the whole dst - no broadcasting - and spells no layout,
            // letting it resolve to whatever the primitive picks for dst.
            return {entry.binary.src1_dt, (1 << ndims) - 1, tag::any, dims_t(),
                    DNNL_ARG_SRC_1};
        }
        const auto &binary = entry.binary;
        int mask = INT_MIN;
        switch (binary.mask_input) {
            // `none` is treated as `policy_t::COMMON` for convenience of
            // calling `get_po_rhs_tensor_entry` without extra guards.
            case attr_t::mask_input_t::none: mask = 0; break;
            case attr_t::mask_input_t::mask: mask = binary.mask; break;
            case attr_t::mask_input_t::policy:
                mask = attr_t::policy2mask(
                        DNNL_ARG_SRC_1, binary.policy, ndims, prim_kind);
                break;
            default: assert(!"unknown mask_input value"); break;
        }
        return {binary.src1_dt, mask, binary.tag, binary.strides,
                DNNL_ARG_SRC_1};
    }

    return post_ops_rhs_tensor_entry_t {};
}

} // namespace

dnnl_status_t attr_args_t::prepare_post_ops_mds(const attr_t &attr, int ndims,
        const dnnl_dims_t prb_dims, dnnl_primitive_kind_t prim_kind,
        const sparse_options_t *sparse_options) {
    const auto &po = attr.post_ops;
    dnnl_dims_t dims;
    for (int d = 0; d < ndims; ++d)
        dims[d] = prb_dims[d];

    int grouped_var_dim_idx = -1;
    dnnl_dim_t grouped_count = 0;
    if (sparse_options) {
        grouped_var_dim_idx
                = sparse_options->get_variable_dim_idx(DNNL_ARG_SRC);
        grouped_count = sparse_options->get_group_count();
    }

    // iterate over all post ops and prepare md for each binary
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (e.is_binary_kind()) {
            const auto &po_rhs_tensor_entry
                    = get_po_rhs_tensor_entry(e, ndims, prim_kind);
            const int mask = po_rhs_tensor_entry.mask;

            // deduce binary, prelu dims based on input policy
            dnnl_dims_t rhs_tensor_dims = {};
            // Overload mask 0 meaning for when binary post-op is applied
            // to grouped memory, changing it to be per-group (i.e. one value
            // per concatenated tensor, for instance for NVFP4 global scale).
            // The grouped dst has no explicit num_groups in dims,
            // so form a [group_count, 1, ...] descriptor manually
            if (grouped_count > 0 && mask == 0) {
                for (auto d = 0; d < ndims; ++d)
                    rhs_tensor_dims[d]
                            = (d == grouped_var_dim_idx) ? grouped_count : 1;
            } else {
                for (auto d = 0; d < ndims; ++d)
                    rhs_tensor_dims[d] = (!(mask & (1 << d))) ? 1 : dims[d];
            }

            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> rhs_tensor_desc;
            DNN_SAFE_STATUS(dnn_mem_t::init_md(rhs_tensor_desc, ndims,
                    rhs_tensor_dims, po_rhs_tensor_entry.dt,
                    po_rhs_tensor_entry.tag, po_rhs_tensor_entry.strides));
            if (e.binary.grouped) {
                DNN_SAFE_STATUS(dnn_mem_t::init_grouped_md(rhs_tensor_desc,
                        ndims, rhs_tensor_dims, po_rhs_tensor_entry.dt,
                        grouped_var_dim_idx, grouped_count));
            }
            mds.emplace((DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | po_rhs_tensor_entry.arg_attr_mask),
                    std::move(rhs_tensor_desc));

            if (e.is_binary_kind_with_ternary_op()) {
                // Deduce the condition (src2) dims from its mask/policy so a
                // broadcast condition (e.g. per_tensor, or broadcast on outer
                // dims) is expressed as size-1 dims, mirroring src1 above.
                // With no explicit src2 spec the condition defaults to the full
                // dst shape (all mask bits set) to preserve prior behavior.
                const int full_mask = (1 << ndims) - 1;
                int src2_mask = full_mask;
                switch (e.binary.src2_mask_input) {
                    case attr_t::mask_input_t::none:
                        src2_mask = full_mask;
                        break;
                    case attr_t::mask_input_t::mask:
                        src2_mask = e.binary.src2_mask;
                        break;
                    case attr_t::mask_input_t::policy:
                        src2_mask = attr_t::policy2mask(DNNL_ARG_SRC_2,
                                e.binary.src2_policy, ndims, prim_kind);
                        break;
                    default: assert(!"unknown mask_input value"); break;
                }
                dnnl_dims_t src2_tensor_dims = {};
                for (auto d = 0; d < ndims; ++d)
                    src2_tensor_dims[d]
                            = (!(src2_mask & (1 << d))) ? 1 : dims[d];
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>
                        rhs_src2_tensor_desc;
                DNN_SAFE_STATUS(dnn_mem_t::init_md(rhs_src2_tensor_desc, ndims,
                        src2_tensor_dims, e.binary.src2_dt, tag::any));
                mds.emplace(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_2),
                        std::move(rhs_src2_tensor_desc));
            }
        } else if (e.is_prelu_kind()) {
            // Prelu doesn't rely on md anywhere so far. Treat it as a quant
            // entry.
            const auto &po_rhs_tensor_entry
                    = get_po_rhs_tensor_entry(e, ndims, prim_kind);
            entries.insert(std::make_pair(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_WEIGHTS,
                    po_rhs_tensor_entry.mask));
        } else if (e.is_convolution_kind()) {
            // Update dims for post operations appended after conv_dw
            conv_dw_fusion::get_fused_conv_dst_dims(ndims, e, dims, dims);
        }
    }

    // dropout
    if (!attr.dropout.is_def()) {
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dropout_mask_desc;
        if (attr.dropout.has_output_mask()) {
            DNN_SAFE_STATUS(dnn_mem_t::init_md(
                    dropout_mask_desc, ndims, dims, dnnl_u8, attr.dropout.tag));
        } else {
            DNN_SAFE_STATUS(dnn_mem_t::init_md(dropout_mask_desc));
        }
        mds.emplace(DNNL_ARG_ATTR_DROPOUT_MASK, std::move(dropout_mask_desc));
    }

    return dnnl_success;
}

void attr_args_t::prepare_dw_post_op(
        const attr_t &attr, dnnl_data_type_t wei_dt, dnnl_data_type_t bia_dt) {
    const int dw_idx = attr.post_ops.convolution_index();
    if (dw_idx == -1) return;

    dw_entry.wei_dt = wei_dt;
    dw_entry.bia_dt = bia_dt;
}

dnnl_status_t create_dnnl_attr(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> &dnnl_attr_out,
        const attr_t &attr, const attr_args_t &attr_args, const int ndims) {
    dnnl_primitive_attr_t dnnl_attr = nullptr;
    DNN_SAFE_STATUS(dnnl_primitive_attr_create(&dnnl_attr));
    // Manage the attribute lifetime with an internal wrapper so that any early
    // exit on failure cleans up the object. On success, the ownership is
    // transferred to the user-provided wrapper.
    auto dnnl_attr_wrapper = make_benchdnn_dnnl_wrapper(dnnl_attr);

    if (!attr.scales.is_def()) {
        const auto &as = attr.scales;
        for (const auto &arg : as.scales) {
            const int arg_name = arg.first;
            if (as.is_def(arg_name)) continue;

            const auto &e = arg.second;
            int is_on_host = e.is_host_scalar();

            // Check if there's a arg with pre-defined mask in `attr_args`...
            int args_mask = attr_args.get_mask(DNNL_ARG_ATTR_SCALES | arg_name);
            // If it's non-default, use it, otherwise, deduce it.
            int mask = args_mask != attr_args_t::undefined_mask
                    ? args_mask
                    : as.get_mask(arg_name, dnnl_undefined_primitive, ndims);

            DNN_SAFE_STATUS(dnnl_primitive_attr_set_scales_v3(dnnl_attr,
                    arg_name, mask, static_cast<int>(e.groups.size()),
                    e.groups.data(), e.dt, is_on_host,
                    attr_t::policy2quantization_mode(e.policy)));
        }
    }

    if (!attr.zero_points.is_def()) {
        const auto &zp = attr.zero_points;
        for (const auto &arg : zp.points) {
            const auto arg_name = arg.first;
            if (zp.is_def(arg_name)) continue;

            const auto &e = arg.second;

            if (e.policy == policy_t::HOST_SCALAR) {
                DNN_SAFE_STATUS(dnnl_primitive_attr_set_zero_points_v2(
                        dnnl_attr, arg_name, 0 /* mask */, 0 /* ndims */,
                        nullptr, e.dt, true /* is_on_host = true*/));

                continue;
            }

            // Check if there's a arg with pre-defined mask in `attr_args`...
            int args_mask
                    = attr_args.get_mask(DNNL_ARG_ATTR_ZERO_POINTS | arg_name);
            // If it's non-default, use it, otherwise, deduce it.
            int mask = args_mask != attr_args_t::undefined_mask
                    ? args_mask
                    : zp.get_mask(arg_name, dnnl_undefined_primitive, ndims);

            int group_ndims = static_cast<int>(e.groups.size());
            const auto &groups = e.groups.data();
            const auto dt = e.dt;

            DNN_SAFE_STATUS(dnnl_primitive_attr_set_zero_points(
                    dnnl_attr, arg_name, mask, group_ndims, groups, dt));
        }
    }

    if (!attr.precomputed_reductions.is_def()) {
        const auto &pr = attr.precomputed_reductions;
        for (const auto &arg : pr.entries) {
            const auto arg_name = arg.first;
            if (pr.is_def(arg_name)) continue;

            const auto &e = arg.second;
            // Check if there's an arg with pre-defined mask in `attr_args`...
            int args_mask = attr_args.get_mask(
                    DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | arg_name);
            // If it's non-default, use it, otherwise, deduce it.
            int mask = args_mask != attr_args_t::undefined_mask
                    ? args_mask
                    : pr.get_mask(arg_name, dnnl_undefined_primitive, ndims);

            int group_ndims = static_cast<int>(e.groups.size());
            const auto &groups = e.groups.data();
            const auto dt = e.dt;

            DNN_SAFE_STATUS(dnnl_primitive_attr_set_precomputed_reductions(
                    dnnl_attr, arg_name, mask, group_ndims, groups, dt));
        }
    }

    if (!attr.rounding_mode.is_def()) {
        for (const auto &e : attr.rounding_mode.rounding_modes_) {
            DNN_SAFE_STATUS(dnnl_primitive_attr_set_rounding(
                    dnnl_attr, e.first, e.second));
        }
    }

    if (!attr.post_ops.is_def()) {
        dnnl_post_ops_t ops = nullptr;
        DNN_SAFE_STATUS(dnnl_post_ops_create(&ops));
        // RAII wrapper cleans up the local post-ops on any early exit. The
        // attribute copies the post-ops in `set_post_ops`, so the local copy
        // can be safely destroyed afterwards.
        auto ops_wrapper = make_benchdnn_dnnl_wrapper(ops);

        const auto &po = attr.post_ops;
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                DNN_SAFE_STATUS(dnnl_post_ops_append_sum(
                        ops, e.sum.scale, e.sum.zero_point, e.sum.dt));
            } else if (e.is_convolution_kind()) {
                const auto wei_dt = attr_args.get_dw_arg(DNNL_ARG_WEIGHTS);
                const auto bia_dt = attr_args.get_dw_arg(DNNL_ARG_BIAS);

                DNN_SAFE_STATUS(dnnl_post_ops_append_dw(ops, wei_dt, bia_dt,
                        e.convolution.dst_dt, e.convolution.kernel,
                        e.convolution.stride, e.convolution.padding));
            } else if (e.is_eltwise_kind()) {
                DNN_SAFE_STATUS(dnnl_post_ops_append_eltwise(
                        ops, e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else if (e.is_binary_kind()) {
                const auto &src1_md = attr_args.get_md(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
                const auto &src2_md = attr_args.get_md(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_2));
                assert(query_md_ndims(src1_md) != 0);

                if (e.is_binary_kind_with_ternary_op()) {
                    assert(query_md_ndims(src2_md) != 0);
                }

                DNN_SAFE_STATUS(dnnl_post_ops_append_binary_v2(
                        ops, e.binary.alg, src1_md, src2_md));

            } else if (e.is_prelu_kind()) {
                const auto mask = attr_args.get_mask(
                        DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_WEIGHTS);
                DNN_SAFE_STATUS(dnnl_post_ops_append_prelu(ops, mask));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_STATUS(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));
        auto c_ops = query_post_ops(dnnl_attr);
        if (dnnl_post_ops_len(c_ops) != po.len()) return dnnl_invalid_arguments;
    }

    DNN_SAFE_STATUS(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, attr.scratchpad_mode));

    DNN_SAFE_STATUS(dnnl_primitive_attr_set_fpmath_mode_v2(
            dnnl_attr, attr.fpmath_mode.mode, attr.fpmath_mode.apply_to_int));

    DNN_SAFE_STATUS(dnnl_primitive_attr_set_accumulation_mode(
            dnnl_attr, attr.acc_mode));

    DNN_SAFE_STATUS(dnnl_primitive_attr_set_deterministic(
            dnnl_attr, attr.deterministic.enabled));

    if (!attr.dropout.is_def()) {
        const auto &drop_mask_md = attr_args.get_md(DNNL_ARG_ATTR_DROPOUT_MASK);
        DNN_SAFE_STATUS(dnnl_primitive_attr_set_dropout_v2(dnnl_attr,
                drop_mask_md, dnnl_s64, attr.dropout.offset != 0,
                attr.dropout.use_host_scalars));
    }

    // Transfer the ownership of a successfully created attribute to the caller.
    dnnl_attr_out = std::move(dnnl_attr_wrapper);
    return dnnl_success;
}

// Exception free version of std::stoi, sets idx to 0 and returns 0 in case of
// error.
static int stoi_safe(const std::string &s, size_t *idx) {
    if (s.empty() || !std::isdigit(s[0])) {
        *idx = 0;
        return 0;
    }
    return std::stoi(s, idx);
}

static bool is_abc_tag(const std::string &tag) {
    if (tag == tag::undef || tag == tag::any) return true;

    bool mask[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;
        int idx = std::tolower(c) - 'a';
        if (idx < 0 || idx >= DNNL_MAX_NDIMS) return false;
        mask[idx] = true;
    }
    // Check there are no gaps, e.g. [1 1 1 1 0 0 ...].
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        if (mask[i]) continue;
        for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
            if (mask[j]) return false;
        break;
    }
    return true;
}

bool is_enum_tag(const std::string &tag) {
    // Relies on a hash-set lookup over a linear chain calls for each tag.
    // The set is derived from the contiguous enum range via `fmt_tag2str` so it
    // stays in sync automatically when tags are added.
    static const std::unordered_set<std::string> enum_tags = []() {
        std::unordered_set<std::string> tags;
        for (int t = dnnl_format_tag_undef; t < dnnl_format_tag_last; t++)
            tags.insert(fmt_tag2str(static_cast<dnnl_format_tag_t>(t)));
        return tags;
    }();
    return enum_tags.find(tag) != enum_tags.end();
}

int check_abc_tag(const std::string &tag_, bool check_enum_tags_only) {
    if (tag_.empty()) return FAIL;
    if (!is_abc_tag(tag_)) return FAIL;
    if (check_enum_tags_only) {
        if (!is_enum_tag(tag_)) return FAIL;
        return OK;
    }

    enum class dim_state_t { undef = 0, upper, lower, lower_with_block };
    dim_state_t dim_states[DNNL_MAX_NDIMS] = {};
    bool in_inner_block = false;
    auto tag = tag_;
    while (!tag.empty()) {
        // Parse block size if presented.
        size_t idx;
        int block = stoi_safe(tag, &idx);
        if (block == 0 && idx != 0) return FAIL;
        if (idx == 0) block = 0;
        if (block > 0) in_inner_block = true;

        // Move to the first position after the block.
        tag = tag.substr(idx);
        if (tag.empty()) return FAIL;

        char c = tag[0];
        bool is_lower = ('a' <= c && c <= 'a' + DNNL_MAX_NDIMS - 1);
        bool is_upper = ('A' <= c && c <= 'A' + DNNL_MAX_NDIMS - 1);
        if (!is_lower && !is_upper) return FAIL;

        // Uppercase cannot be with block.
        if (is_upper && block != 0) return FAIL;
        // Block sizes are required within inner block.
        if (block == 0 && in_inner_block) return FAIL;

        // Check rules related to lowercase/uppercase/block order.
        int dim_idx = std::tolower(c) - 'a';
        dim_state_t prev_state = dim_states[dim_idx];
        dim_state_t cur_state = is_upper ? dim_state_t::upper
                : block != 0             ? dim_state_t::lower_with_block
                                         : dim_state_t::lower;

        switch (cur_state) {
            case dim_state_t::upper:
            case dim_state_t::lower:
                // Letter without block must be the first.
                if (prev_state != dim_state_t::undef) return FAIL;
                break;
            case dim_state_t::lower_with_block:
                // Letter with block must be after uppercase or after a letter
                // with block.
                if (prev_state != dim_state_t::upper
                        && prev_state != dim_state_t::lower_with_block)
                    return FAIL;
                break;
            default: assert(!"not expected");
        }

        // Update state, move to the next position.
        dim_states[dim_idx] = cur_state;
        tag = tag.substr(1);
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        // Uppercase letter must be followed by lowercase.
        if (dim_states[i] == dim_state_t::upper) return FAIL;

        // Ensure there are no gaps (e.g. acd).
        if (dim_states[i] == dim_state_t::undef) {
            for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
                if (dim_states[j] != dim_state_t::undef) return FAIL;
            break;
        }
    }

    return OK;
}

static std::string trim_letter(const std::string &tag_, char c) {
    auto tag = tag_;
    for (size_t pos = tag.find(c); pos != std::string::npos;
            pos = tag.find(c)) {
        tag.replace(pos, 1, "");
        if (pos == 0) return tag;

        pos--;
        while (std::isdigit(tag[pos])) {
            tag.replace(pos, 1, "");
            if (pos == 0) break;
            pos--;
        }
    }
    return tag;
}

// Tries to map a tag to an abc-tag according to a logical tag. For example:
// nchw -> abcd.
static std::string try_map_tag(
        const std::string &logical_tag, const std::string &tag, int *nmatched) {
    // Check if all the required letters are presented.
    for (auto &c : logical_tag) {
        if (std::toupper(c) == c
                && tag.find(std::tolower(c)) == std::string::npos)
            return {};
    }

    // Check that all letters are known and assign indices to letters.
    int logical_indices[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;

        auto lower_pos = logical_tag.find(std::tolower(c));
        auto upper_pos = logical_tag.find(std::toupper(c));
        auto pos = (lower_pos == std::string::npos ? upper_pos : lower_pos);
        if (pos == std::string::npos) return {};

        logical_indices[pos] = 1;
    }

    for (int i = 0, idx = 0; i < (int)logical_tag.size(); i++) {
        if (logical_indices[i] == 0) continue;
        logical_indices[i] = idx++;
    }

    (*nmatched)++;
    std::string mapped_tag = tag;
    for (int i = 0; i < (int)tag.size(); i++) {
        char c = tag[i];
        if (!std::isalpha(tag[i])) continue;
        size_t pos = logical_tag.find(std::tolower(c));
        if (pos == std::string::npos) pos = logical_tag.find(std::toupper(c));
        if (pos >= logical_tag.size()) SAFE_V(FAIL);

        mapped_tag[i]
                = (char)(tag[i] - std::tolower(c) + 'a' + logical_indices[pos]);
    }
    return mapped_tag;
}

// Maps a tag to an abc-tag.
static std::string map_tag_letters(const std::string &tag) {
    int nmatched = 0;

    // Mapping rules:
    // - Uppercase letters are mandatory
    // - Lowercase letters are optional
    auto tag_goidhw = try_map_tag("GOIdhw", tag, &nmatched);
    auto tag_oidhw = try_map_tag("OIdhw", tag, &nmatched);
    auto tag_ncdhw = try_map_tag("NCdhw", tag, &nmatched);
    auto tag_tnc = try_map_tag("TNc", tag, &nmatched);
    auto tag_ldnc = try_map_tag("LDNC", tag, &nmatched);
    auto tag_ldigo = try_map_tag("LDigO", tag, &nmatched);

    if (nmatched == 0) return tag;
    if (nmatched > 1) assert(!"Not expected: ambiguous tag.");

    if (!tag_goidhw.empty()) return tag_goidhw;
    if (!tag_oidhw.empty()) return tag_oidhw;
    if (!tag_ncdhw.empty()) return tag_ncdhw;
    if (!tag_tnc.empty()) return tag_tnc;
    if (!tag_ldnc.empty()) return tag_ldnc;
    if (!tag_ldigo.empty()) return tag_ldigo;

    return tag;
}

std::string trim_tag(const std::string &tag, int ndims) {
    int mask = 0;
    for (int d = 0; d < ndims; d++) {
        mask += (1 << d);
    }
    return trim_tag_by_mask(tag, mask);
}

std::string trim_tag_by_mask(const std::string &tag, int mask) {
    std::string trimmed_tag = tag;
    int ndims_saved = 0;
    for (char c = 'a', d = 0; c < 'a' + (char)(DNNL_MAX_NDIMS); c++, d++) {
        if (!(mask & (1 << d))) {
            trimmed_tag = trim_letter(trimmed_tag, c);
            trimmed_tag = trim_letter(trimmed_tag, std::toupper(c));
        } else {
            ndims_saved++;
        }
    }

    // Mask may operate over non-consecutive dimensions. The piece below will
    // make trimmed_tag consist of consecutive dimensions starting from "a" or
    // "A". E.g., mask = 2 + 8 = 10, trimmed_tag will contain "b" and "d"
    // letters, and will be converted into one with "a" and "b".
    int mask_copy = mask;
    for (int i = 0; i < ndims_saved; i++) {
        int dist_to_a = 0;
        while (mask_copy % 2 == 0) {
            mask_copy /= 2;
            dist_to_a++;
        }
        mask_copy /= 2;
        if (dist_to_a == 0) continue;

        for (size_t j = 0; j < trimmed_tag.size(); j++) {
            char str_j = trimmed_tag[j];
            if (std::isalpha(str_j) && std::tolower(str_j) > 'a' + i) {
                std::string rep_str(1, str_j - dist_to_a);
                trimmed_tag.replace(j, 1, rep_str);
            }
        }
    }

    return trimmed_tag;
}

std::string normalize_tag(const std::string &tag_, int ndims) {
    std::string tag = tag_;
    if (tag == tag::undef || tag == tag::any || ndims == 0) return tag;
    if (tag == tag::x) {
        if (ndims >= 0) assert(ndims == 1);
        return "a";
    }

    // Handle meta-tags (abx, axb, aby, etc).
    auto pos_x = tag.find("x");
    auto pos_y = tag.find("y");
    if (pos_x != std::string::npos && pos_y != std::string::npos) {
        BENCHDNN_PRINTF(0, "%s",
                "Error: a tag can't have both \'x\' and \'y\' letters "
                "specified.");
        return tag;
    }

    if (pos_y != std::string::npos && pos_y != tag.size() - 1) {
        BENCHDNN_PRINTF(0, "%s",
                "Error: a tag containing \'y\' must have it at the end.");
        return tag;
    }

    if (pos_x == std::string::npos && pos_y == std::string::npos) {
        return map_tag_letters(tag);
    }

    auto pos = pos_x != std::string::npos ? pos_x : pos_y;
    // Non-grouped tags will start `x` from `c`, but grouped will most of
    // times start `x` from `d`.
    char start_xy = 'c';
    for (char c = 'a' + DNNL_MAX_NDIMS - 1; c >= 'b'; c--) {
        if (tag.find(c) != std::string::npos) {
            start_xy = c + 1;
            break;
        }
    }
    // Adjust ndims if they are not specified.
    int meta_ndims = (ndims == -1 ? (start_xy - 'a' + 1) : ndims);
    std::string tail;
    for (int i = 0; i < meta_ndims - (start_xy - 'a'); i++)
        tail += (start_xy + i);
    auto trimmed_tag = trim_tag(tag.replace(pos, 1, tail), meta_ndims);
    // Transpose last two letters if 'y' was requested.
    if (pos_y != std::string::npos) {
        assert(trimmed_tag.size() >= 2);
        const auto sz = trimmed_tag.size();
        std::swap(trimmed_tag[sz - 1], trimmed_tag[sz - 2]);
    }
    return trimmed_tag;
}

int check_tag(const std::string &tag_, bool check_enum_tags_only) {
    auto tag = normalize_tag(tag_);
    if (tag == tag::undef || tag == tag::any) return OK;
    return check_abc_tag(tag, check_enum_tags_only);
}

void maybe_scale(const attr_t &attr, float &d, const float *scales, int64_t c,
        int arg, bool opposite_scale) {
    if (attr.scales.is_def(arg)) return;

    const auto &e = attr.scales.get(arg);
    if (!e.is_def()) {
        int64_t idx = e.has_single_element() ? 0 : c;
        float s = scales[idx];
        if (opposite_scale) s = 1.f / s;
        d *= s;
    }
}

void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point) {
    if (attr.zero_points.is_def()) return;

    const auto &e = attr.zero_points.get(arg);
    if (!e.is_def()) {
        const int idx = e.has_single_element() ? 0 : c;
        const int zp_sign = opposite_zero_point ? -1 : 1;
        d -= zp_sign * zero_points[idx];
    }
}

float compute_eltwise_fwd(pk_t kind, float src, float alpha, float beta) {
    if (std::isnan(src)) {
        // Don't compute if source is NAN, propagate it, unless a special case.

        // pow(NAN, 0) = 1 according to IEEE Std 754-2008.
        const bool is_pow0 = kind == pk_t::POW && beta == 0;
        const bool propagate_nan = !is_pow0;
        if (propagate_nan) { return NAN; }
    }

    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return relu_fwd(src, alpha);
        case pk_t::TANH: return tanh_fwd(src);
        case pk_t::ELU: return elu_fwd(src, alpha);
        case pk_t::SQUARE: return square_fwd(src);
        case pk_t::ABS: return abs_fwd(src);
        case pk_t::SQRT: return sqrt_fwd(src);
        case pk_t::LINEAR: return linear_fwd(src, alpha, beta);
        case pk_t::SRELU: return soft_relu_fwd(src, alpha);
        case pk_t::MISH: return mish_fwd(src);
        case pk_t::LOGISTIC: return logistic_fwd(src);
        case pk_t::EXP: return exp_fwd(src);
        case pk_t::GELU_TANH: return gelu_tanh_fwd(src);
        case pk_t::SWISH: return swish_fwd(src, alpha);
        case pk_t::LOG: return log_fwd(src);
        case pk_t::CLIP: return clip_fwd(src, alpha, beta);
        case pk_t::CLIP_V2: return clip_v2_fwd(src, alpha, beta);
        case pk_t::POW: return pow_fwd(src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_fwd(src);
        case pk_t::ROUND: return round_fwd(src);
        case pk_t::HARDSWISH: return hardswish_fwd(src, alpha, beta);
        case pk_t::HARDSIGMOID: return hardsigmoid_fwd(src, alpha, beta);
        case pk_t::RELU_DST: return relu_fwd(src, alpha);
        case pk_t::TANH_DST: return tanh_fwd(src);
        case pk_t::ELU_DST: return elu_fwd(src, alpha);
        case pk_t::SQRT_DST: return sqrt_fwd(src);
        case pk_t::LOGISTIC_DST: return logistic_fwd(src);
        case pk_t::EXP_DST: return exp_fwd(src);
        case pk_t::CLIP_V2_DST: return clip_v2_fwd(src, alpha, beta);

        default: assert(!"unknown attr::post_ops::kind");
    };
    return NAN;
}

float compute_eltwise_bwd(
        pk_t kind, float d_dst, float src, float alpha, float beta) {
    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return relu_bwd(d_dst, src, alpha);
        case pk_t::TANH: return tanh_bwd(d_dst, src);
        case pk_t::ELU: return elu_bwd(d_dst, src, alpha);
        case pk_t::SQUARE: return square_bwd(d_dst, src);
        case pk_t::ABS: return abs_bwd(d_dst, src);
        case pk_t::SQRT: return sqrt_bwd(d_dst, src);
        case pk_t::LINEAR: return linear_bwd(d_dst, src, alpha, beta);
        case pk_t::SRELU: return soft_relu_bwd(d_dst, src, alpha);
        case pk_t::MISH: return mish_bwd(d_dst, src);
        case pk_t::LOGISTIC: return logistic_bwd(d_dst, src);
        case pk_t::EXP: return exp_bwd(d_dst, src);
        case pk_t::GELU_TANH: return gelu_tanh_bwd(d_dst, src);
        case pk_t::SWISH: return swish_bwd(d_dst, src, alpha);
        case pk_t::LOG: return log_bwd(d_dst, src);
        case pk_t::CLIP: return clip_bwd(d_dst, src, alpha, beta);
        case pk_t::CLIP_V2: return clip_v2_bwd(d_dst, src, alpha, beta);
        case pk_t::POW: return pow_bwd(d_dst, src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_bwd(d_dst, src);
        case pk_t::HARDSWISH: return hardswish_bwd(d_dst, src, alpha, beta);
        case pk_t::HARDSIGMOID: return hardsigmoid_bwd(d_dst, src, alpha, beta);

        case pk_t::RELU_DST: return relu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::TANH_DST: return tanh_bwd_use_dst(d_dst, src);
        case pk_t::ELU_DST: return elu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::SQRT_DST: return sqrt_bwd_use_dst(d_dst, src);
        case pk_t::LOGISTIC_DST: return logistic_bwd_use_dst(d_dst, src);
        case pk_t::EXP_DST: return exp_bwd_use_dst(d_dst, src);
        case pk_t::CLIP_V2_DST:
            return clip_v2_bwd_use_dst(d_dst, src, alpha, beta);

        default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

float compute_binary(pk_t kind, float src0, float src1, bool src2) {
    // don't compute on nan, propagate it
    if (std::isnan(src0) || std::isnan(src1)) return NAN;

    if (kind == pk_t::ADD) {
        return src0 + src1;
    } else if (kind == pk_t::MUL || kind == pk_t::MUL_INPLACE) {
        return src0 * src1;
    } else if (kind == pk_t::MAX) {
        return MAX2(src0, src1);
    } else if (kind == pk_t::MIN) {
        return MIN2(src0, src1);
    } else if (kind == pk_t::DIV) {
        return src0 / src1;
    } else if (kind == pk_t::SUB) {
        return src0 - src1;
    } else if (kind == pk_t::GE) {
        return src0 >= src1;
    } else if (kind == pk_t::GT) {
        return src0 > src1;
    } else if (kind == pk_t::LE) {
        return src0 <= src1;
    } else if (kind == pk_t::LT) {
        return src0 < src1;
    } else if (kind == pk_t::EQ) {
        return src0 == src1;
    } else if (kind == pk_t::NE) {
        return src0 != src1;
    } else if (kind == pk_t::SELECT) {
        return src2 ? src0 : src1;
    } else {
        assert(!"operation not supported!");
    }
    return NAN;
}

// This function is a full copy of ref_dropout(...) from the library.
void maybe_dropout(const attr_t &attr, float &val, int64_t idx,
        const dnn_mem_t &dropout_mask) {
    if (attr.dropout.is_def()) return;

    float p = attr.dropout.p;
    int64_t seed = attr.dropout.seed;
    int64_t offset = attr.dropout.offset;

    auto philox_bernoulli = [=]() {
        // TODO: keeping int32_t path until all impl migrate to 64 bit
        // generation.
        uint32_t r = (offset == 0)
                ? dnnl::impl::math::philox4x32(uint32_t(idx), uint32_t(seed))
                : dnnl::impl::math::philox4x32(idx, seed, offset);
        return (r > double(std::numeric_limits<uint32_t>::max()) * p);
    };

    float inv_q = (p != 1.f) ? 1.f / (1.f - p) : 0.f;
    uint8_t m = philox_bernoulli();
    if (attr.dropout.has_output_mask()) dropout_mask.set_elem(idx, m);
    val = (m) ? val * inv_q : 0;
}

void maybe_round(const attr_t &attr, int arg, float &val, int64_t offset,
        dnnl_data_type_t dst_dt) {
    uint32_t seed = attr.rounding_mode.seed;
    switch (attr.rounding_mode.get(arg)) {
        case dnnl_rounding_mode_stochastic:
            val = dnnl::impl::math::stochastic_round_fwd(
                    val, offset, seed, dst_dt);
            break;
        case dnnl_rounding_mode_environment: break;
        default: assert(!"unknown rounding mode");
    }
}

void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_po_vals) {
    const auto &po = attr.post_ops;
    if (po.len() == 0) return;

    using namespace dnnl::impl::math;

    auto it_po = v_po_vals.begin();
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];

        if (e.is_sum_kind()) {
            val += e.sum.scale * (sum_val - e.sum.zero_point);
        } else if (e.is_convolution_kind()) {
            continue;
        } else if (e.is_eltwise_kind()) {
            const auto &a = e.eltwise.alpha;
            const auto &b = e.eltwise.beta;
            val = compute_eltwise_fwd(e.kind, val, a, b);
        } else if (e.is_binary_kind()) {
            bool is_inplace = e.is_binary_kind_inplace();
            auto src1_val = (!is_inplace) ? *it_po : sum_val;
            bool src2_val = false;

            if (e.is_binary_kind_with_ternary_op()) {
                it_po++;
                src2_val = static_cast<bool>(*it_po);
            }
            val = compute_binary(e.kind, val, src1_val, src2_val);
            if (!is_inplace) it_po++;
        } else if (e.is_prelu_kind()) {
            val = val > 0 ? val : val * (*it_po);
            it_po++;
        }
    }
}

// This is a part of prim_ref setup. Update binary post-op data types and tags
// to let prim_ref dispatch to the library as GPU may use data types not
// supported natively on CPU, and output format may not be supported as well.
// prim_ref will be configured and correspondent reorders used to a selected by
// the library format.
//
// `dst_dt` helps to decide on sum_dt. When it's not f32, `sum.dt` must be
// preserved, otherwise, updated.
void update_cpu_ref_attrs(attr_t &attr, dnnl_data_type_t dst_dt) {
    auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        auto &e = po.entry[idx];
        if (e.is_binary_kind()) {
            e.binary.src1_dt = dnnl_f32;
            e.binary.tag = tag::any;
        } else if (e.is_sum_kind()) {
            if (dst_dt == dnnl_f32) e.sum.dt = dnnl_data_type_undef;
        }
    }
}

int sparse_options_t::from_str(const std::string &s) {
    *this = sparse_options_t();
    if (s.empty()) return OK;

    const auto get_arg = [](int i) {
        switch (i) {
            case 0: return DNNL_ARG_SRC;
            case 1: return DNNL_ARG_WEIGHTS;
            case 2: return DNNL_ARG_DST;
            default: return -1;
        }
    };

    int options_count = 0;
    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        // Note: `csr+0.99::` is a valid input, dangling `:` is legit.
        auto subs = parser::get_substr(
                s, start_pos, ':', /* allow_dangling = */ true);

        if (subs.empty()) {
            add(get_arg(options_count), sparse_options_t::def_encoding,
                    sparse_options_t::def_sparsity);
            options_count++;
            continue;
        }

        if (subs.find("+") == std::string::npos) {
            add(get_arg(options_count), str2sparse_encoding(subs.c_str()),
                    sparse_options_t::def_sparsity);
        } else {
            size_t subs_pos = 0;
            auto encoding_str = parser::get_substr(subs, subs_pos, '+');
            auto sparsity_str = parser::get_substr(subs, subs_pos, '+');
            if (encoding_str.empty() || sparsity_str.empty()) { return FAIL; }
            add(get_arg(options_count),
                    str2sparse_encoding(encoding_str.c_str()),
                    atof(sparsity_str.c_str()));
        }
        options_count++;
    }
    static const int expected_num_options = 3;
    return options_count == expected_num_options ? OK : FAIL;
}

int sparse_options_t::resolve_profile(dnnl_dim_t total_size) {
    grouped_data_t::profile_spec_t spec;
    dnnl_dim_t G = 0, user_hint = 0;
    for (const auto &kv : grouped_data_) {
        if (kv.second.profile.kind != grouped_data_t::profile_none) {
            spec = kv.second.profile;
            G = kv.second.group_count;
            user_hint = kv.second.max_variable_dim;
            break;
        }
    }

    std::vector<dnnl_dim_t> sizes(G, 0);
    switch (spec.kind) {
        case grouped_data_t::profile_balanced: {
            // expected that total_size tokens distributed over G groups
            // more or less evenly
            const dnnl_dim_t seed = spec.get_param_val();
            std::minstd_rand msr(G * total_size + seed + 1);
            msr.discard(1);
            for (dnnl_dim_t i = 0; i < total_size; i++)
                sizes[msr() % (uint64_t)G]++;
            break;
        }
        case grouped_data_t::profile_hot: {
            // one group takes pctg% of the tokens, the rest spreads evenly
            // over the remaining groups
            // pctg == 100 is the extreme where one
            // group takes all and the rest are 0
            const dnnl_dim_t pctg = spec.get_param_val();
            const dnnl_dim_t hot_total
                    = (G > 1) ? total_size * pctg / 100 : total_size;
            sizes[0] = hot_total;
            const dnnl_dim_t rest = total_size - hot_total, cnt = G - 1;
            if (cnt > 0) {
                const dnnl_dim_t q = rest / cnt, r = rest % cnt;
                for (dnnl_dim_t j = 0; j < cnt; j++)
                    sizes[1 + j] = q + (j < r ? 1 : 0);
            }
            break;
        }
        case grouped_data_t::profile_decode: {
            // each of the total_size tokens gets its own group
            if (total_size > G) {
                BENCHDNN_PRINT(0,
                        "Error: decode profile requires total_M (%lld) <= "
                        "NUM_GROUPS (%lld)\n",
                        (long long)total_size, (long long)G);
                return FAIL;
            }
            for (dnnl_dim_t i = 0; i < total_size; i++)
                sizes[i * G / total_size] = 1;
            break;
        }
        default:
            BENCHDNN_PRINT(0, "%s\n", "Error: unknown grouped profile");
            return FAIL;
    }

    // max_variable_dim dispatcher hint,
    // if not provided, use the largest group size from generated
    const dnnl_dim_t max_gen = *std::max_element(sizes.begin(), sizes.end());
    if (user_hint > 0 && user_hint < max_gen) {
        BENCHDNN_PRINT(0,
                "Error: grouped max_variable dim hint (%lld) is smaller "
                "than the max generated group size (%lld)\n",
                (long long)user_hint, (long long)max_gen);
        return FAIL;
    }
    const dnnl_dim_t chosen_hint = user_hint > 0 ? user_hint : max_gen;

    // Dump generated offsets explicitly in a form of valid `--grouped` line
    if (verbose >= 3) {
        std::string s("[GROUPED] --grouped="
                + std::to_string(get_variable_dim_idx(DNNL_ARG_SRC)) + ":"
                + std::to_string(G) + ":");
        for (size_t i = 0; i < sizes.size(); i++)
            s += (i ? "+" : "") + std::to_string(sizes[i]);
        s += ":" + std::to_string(chosen_hint);
        BENCHDNN_PRINT(3, "%s\n", s.c_str());
    }

    for (auto &kv : grouped_data_) {
        if (kv.second.profile.kind != grouped_data_t::profile_none) {
            kv.second.group_sizes = sizes;
            kv.second.max_variable_dim = chosen_hint;
        }
    }
    return OK;
}
