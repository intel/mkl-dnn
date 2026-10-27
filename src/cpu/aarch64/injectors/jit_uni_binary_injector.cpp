/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2022-2024 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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
#include <cassert>
#include <cmath>

#include "common/primitive_attr.hpp"
#include "common/utils.hpp"
#include "common/verbose_msg.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace binary_injector {

#define VCHECK_BIN_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, binary_injector, cond, false, msg);

static bcast_set_t get_all_strategies_supported_by_injector() {
    return bcast_set_t {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
}

bool is_data_supported(data_type_t data_type) {
    using namespace data_type;

    return utils::one_of(data_type, f32, s32, s8, u8);
}

bool is_alg_supported(alg_kind_t alg) {
    return utils::one_of(alg, alg_kind::binary_add, alg_kind::binary_mul,
            alg_kind::binary_max, alg_kind::binary_min, alg_kind::binary_div,
            alg_kind::binary_sub, alg_kind::binary_ge, alg_kind::binary_gt,
            alg_kind::binary_le, alg_kind::binary_lt, alg_kind::binary_eq,
            alg_kind::binary_ne);
}

bool is_supported(cpu_isa_t isa, alg_kind_t alg,
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    VCHECK_BIN_INJ_BOOL(utils::one_of(isa, sve, asimd), VERBOSE_UNSUPPORTED_ISA)

    VCHECK_BIN_INJ_BOOL(
            is_data_supported(src1_desc.data_type), VERBOSE_ISA_DT_MISMATCH);

    VCHECK_BIN_INJ_BOOL(is_alg_supported(alg), VERBOSE_BAD_ALGORITHM);

    const memory_desc_wrapper src1_mdw(src1_desc);
    VCHECK_BIN_INJ_BOOL(!src1_mdw.format_any(),
            "set the rhs format before checking for support");
    VCHECK_BIN_INJ_BOOL(src1_mdw.is_dense(true), VERBOSE_NONTRIVIAL_STRIDE);

    VCHECK_BIN_INJ_BOOL(
            is_bcast_supported(src1_desc, dst_d, supported_strategy_set),
            "unsupported broadcast");

    return true;
}

static bool src1_desc_layout_same_as_dst_d(
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d) {
    if (dst_d.md_ == nullptr) return false;
    const auto &lhs = src1_desc;
    const auto &rhs = *(dst_d.md_);

    using namespace dnnl::impl::utils;
    const bool is_format_any
            = one_of(format_kind::any, lhs.format_kind, rhs.format_kind);

    return lhs.ndims == rhs.ndims
            && (is_format_any
                    || (lhs.format_kind == rhs.format_kind
                            && array_cmp(lhs.format_desc.blocking.strides,
                                    rhs.format_desc.blocking.strides,
                                    lhs.ndims)))
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0;
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);

    VCHECK_BIN_INJ_BOOL(
            IMPLICATION(bcast_type == broadcasting_strategy_t::no_broadcast,
                    src1_desc_layout_same_as_dst_d(src1_desc, dst_d)),
            "src1 and dst must have the same layout if not broadcasting");

    return bcast_type != broadcasting_strategy_t::unsupported;
}

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_like_binary()) {
            const auto bcast_type
                    = get_rhs_arg_broadcasting_strategy(entry.binary.src1_desc,
                            dst_d, get_all_strategies_supported_by_injector());
            return !utils::one_of(bcast_type, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::unsupported);
        }
        return false;
    });
}

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen,
        const bcast_set_t &supported_strategy_set) {
    const auto channels = dst_d.dims()[1];
    const int vmm_l_len = vlen / 4;

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    entry.binary.src1_desc, dst_d, supported_strategy_set);
            return utils::one_of(bcast_type, broadcasting_strategy_t::per_oc,
                           broadcasting_strategy_t::per_oc_spatial)
                    && (channels % vmm_l_len != 0);
        }
        return false;
    });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
        if (entry.is_binary()) {
            const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
            return rhs_arg_d.matches_tag(tag);
        }
        return true;
    });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return any_binary_postop_rhs_per_oc_broadcast(
            post_ops, dst_d, get_all_strategies_supported_by_injector());
}

bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
        if (entry.is_binary()) {
            const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                    entry.binary.src1_desc, dst_d, supported_strategy_set);
            return bcast_type == broadcasting_strategy_t::per_oc
                    || bcast_type == broadcasting_strategy_t::per_oc_spatial;
        }
        return false;
    });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> &predicate) {
    return true;
}

static_params_t::static_params_t(const Xbyak_aarch64::XReg &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params) {}

static_params_t::static_params_t(const Xbyak_aarch64::XReg &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1, get_all_strategies_supported_by_injector(),
              rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
        std::size_t tail_size, bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, 0, dst_d, tail_size,
              Xbyak_aarch64::PReg(2), use_exact_tail_scalar_bcast,
              rhs_helper_reg, false /*is_opmask_set*/,
              false /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, std::size_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, Xbyak_aarch64::PReg(2), use_exact_tail_scalar_bcast,
              rhs_helper_reg, false /*is_opmask_set*/,
              true /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
        std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, 0, dst_d, tail_size,
              tail_opmask, use_exact_tail_scalar_bcast, rhs_helper_reg,
              true /*is_opmask_set*/, false /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, std::size_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, tail_opmask, use_exact_tail_scalar_bcast,
              rhs_helper_reg, true /*is_opmask_set*/,
              true /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
        std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
        const Xbyak_aarch64::XReg &reg_tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, 0, dst_d, tail_size,
              tail_opmask, use_exact_tail_scalar_bcast, reg_tail_size,
              true /*is_opmask_set*/, false /*is_dst_orig_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, std::size_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        const Xbyak_aarch64::XReg &reg_tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
              rhs_helper_reg, rhs_addr_cache_reg, preserve_gpr_helpers,
              preserve_vmm_helper, abi_param_offset, dst_orig_offset, dst_d,
              tail_size, tail_opmask, use_exact_tail_scalar_bcast,
              reg_tail_size, true /*is_opmask_set*/, true /*is_dst_orig_set*/) {
}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg,
        const Xbyak_aarch64::XReg &rhs_addr_cache_reg,
        bool preserve_gpr_helpers, bool preserve_vmm_helper,
        std::size_t abi_param_offset, std::size_t dst_orig_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast,
        const Xbyak_aarch64::XReg &reg_tail_size, bool is_opmask_set,
        bool is_dst_orig_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , rhs_addr_cache_reg(rhs_addr_cache_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_orig_offset(dst_orig_offset)
    , dst_d(dst_d)
    , tail_size(tail_size)
    , tail_opmask(tail_opmask)
    , use_exact_tail_scalar_bcast(use_exact_tail_scalar_bcast)
    , reg_tail_size(reg_tail_size)
    , is_tail(tail_size)
    , is_opmask_set_(is_opmask_set)
    , is_dst_orig_set_(is_dst_orig_set) {}

template <cpu_isa_t isa>
jit_uni_binary_injector_t<isa>::jit_uni_binary_injector_t(
        jit_generator_t *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , supported_strategy_set_(static_params.supported_strategy_set) {}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool params_differ_xreg(const std::map<int, Xbyak_aarch64::XReg> &params,
        const std::map<int, Xbyak_aarch64::XReg>::key_type key1,
        const std::map<int, Xbyak_aarch64::XReg>::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    const Xbyak_aarch64::XReg &it1_second = it1->second;
    const Xbyak_aarch64::XReg &it2_second = it2->second;
    return it1_second.getIdx() != it2_second.getIdx();
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_addr = rhs_arg_params.vmm_idx_to_out_addr;
    const auto &out_reg = rhs_arg_params.vmm_idx_to_out_reg;

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;
    const auto &sp_off_addr = rhs_arg_params.vmm_idx_to_sp_elem_off_addr;
    const auto &sp_off_val = rhs_arg_params.vmm_idx_to_sp_elem_off_val;
    const auto &sp_off_oprnd = rhs_arg_params.vmm_idx_to_sp_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_xreg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(out_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_xreg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::per_mb_spatial) {
        return params_differ(out_addr, vmm_idx1, vmm_idx2)
                || params_differ_xreg(out_reg, vmm_idx1, vmm_idx2)
                || params_differ(out_elem_off_val, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_val, vmm_idx1, vmm_idx2)
                || params_differ(sp_off_oprnd, vmm_idx1, vmm_idx2);
    }
    return true;
}

template <cpu_isa_t isa>
int jit_uni_binary_injector_t<isa>::adjust_temp_vmm_hint(
        int user_hint, int start_idx, int end_idx, int max_vmm_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_vmm_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        const bool max_vmm_idx_in_vector_range
                = max_vmm_idx >= start_idx && max_vmm_idx <= end_idx;

        if (max_vmm_idx_in_vector_range || user_hint_exceeded_limit
                || user_hint == max_vmm_idx)
            return 0;
        else
            return max_vmm_idx;
    }

    return user_hint;
}

// All stack size calculations need to be rounded up to the nearest multiple
// of 16 to satisfy architecture requirements
template <typename Vmm>
static void push_vmm(jit_generator_t *host, const Vmm &vmm) {
    const int32_t stack_space_needed_bytes
            = utils::rnd_up(vmm.getBit() / 8, 16);

    host->sub_imm(host->sp, host->sp, stack_space_needed_bytes, host->X_TMP_0);
    host->uni_str(vmm, host->sp);
}

template <typename Vmm>
static void pop_vmm(jit_generator_t *host, const Vmm &vmm) {
    const int32_t stack_space_needed_bytes
            = utils::rnd_up(vmm.getBit() / 8, 16);

    host->uni_ldr(vmm, host->sp);
    host->add_imm(host->sp, host->sp, stack_space_needed_bytes, host->X_TMP_0);
}

static void push_opmask(jit_generator_t *host, const Xbyak_aarch64::PReg &k) {
    const int32_t stack_space_needed_bytes = utils::rnd_up(k.getBit() / 8, 16);

    host->sub_imm(host->sp, host->sp, stack_space_needed_bytes, host->X_TMP_0);
    host->str(k, Xbyak_aarch64::ptr(host->sp));
}

static void pop_opmask(jit_generator_t *host, const Xbyak_aarch64::PReg &k) {
    const int32_t stack_space_needed_bytes = utils::rnd_up(k.getBit() / 8, 16);

    host->ldr(k, Xbyak_aarch64::ptr(host->sp));
    host->add_imm(host->sp, host->sp, stack_space_needed_bytes, host->X_TMP_0);
}

template <cpu_isa_t isa>
std::pair<bool, int> jit_uni_binary_injector_t<isa>::should_preserve_vmm(
        int curr_idx, int vmm_hint, int max_vmm_idx,
        bool dt_helper_vmm_needed) const {
    if (dt_helper_vmm_needed && vmm_hint == curr_idx) {
        if (curr_idx == 0)
            return std::make_pair(true, max_vmm_idx);
        else
            return std::make_pair(true, 0);
    }
    return std::make_pair(false, vmm_hint);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    if (vmm_idxs.empty()) return;

    assert(post_op.is_binary() && is_alg_supported(post_op.binary.alg));

    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    static constexpr int max_vmm_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, supported_strategy_set_);
    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const auto &vmm_tail_idx = rhs_arg_params.vmm_tail_idx_;
    const bool tail_exists_in_range = !vmm_tail_idx.empty();
    const bool should_preserve_vmm_tail = tail_exists_in_range
            && (!utils::one_of(rhs_broadcasting_strategy,
                        broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);

    const bool need_tail_handling_in_range = rhs_arg_static_params_.is_tail
            && tail_exists_in_range
            && rhs_arg_static_params_.use_exact_tail_scalar_bcast;

    // When these conditions are met we can avoid repeatedly loading the
    // an unchanging scalar RHS
    const bool can_load_scalar_once
            = rhs_broadcasting_strategy == broadcasting_strategy_t::scalar
            && !need_tail_handling_in_range && vmm_idxs.count(vmm_hint) == 0;

    const std::initializer_list<Xbyak_aarch64::XReg> gpr_maybe_preserve_list {
            rhs_arg_static_params_.rhs_addr_reg,
            rhs_arg_static_params_.rhs_helper_reg};

    const std::initializer_list<Xbyak_aarch64::VReg> vmm_maybe_preserve_list {
            Xbyak_aarch64::VReg(vmm_hint)};

    const bool dt_helper_vmm_needed = can_load_scalar_once
            || !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || should_preserve_vmm_tail;
    const auto tail_load_mode = rhs_arg_params.tail_load_mode;

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t<to_vla_sve(isa)>
            register_guard {host_,
                    (rhs_arg_static_params_.preserve_gpr_helpers
                                    ? gpr_maybe_preserve_list
                                    : std::initializer_list<
                                              Xbyak_aarch64::XReg>()),

                    (rhs_arg_static_params_.preserve_vmm_helper
                                            && dt_helper_vmm_needed
                                    ? vmm_maybe_preserve_list
                                    : std::initializer_list<
                                              Xbyak_aarch64::VReg>())};

    bool vmm0_was_preserved = false;
    static const Vmm zero_vmm(0);

    rhs_address_t rhs_arg_addr(Xbyak_aarch64::XReg(0));

    // Phase 3 Apply binary post-op over all vmms.
    // Phase 3.a: fast path:
    if (can_load_scalar_once) {
        const Vmm rhs_vmm(vmm_hint);
        rhs_arg_addr = prepare_rhs_arg_addr(start_idx, rhs_arg_idx, post_op,
                rhs_arg_params, rhs_broadcasting_strategy,
                rhs_arg_static_params_.rhs_addr_reg);
        execute_broadcast_no_tail(
                rhs_arg_data_type, rhs_vmm, remove_bcast_bit(rhs_arg_addr));

        // The integral types would have been converted to s32 by the broadcast
        // load, so convert them to f32 here.
        if (utils::one_of(rhs_arg_data_type, data_type::s32, data_type::s8,
                    data_type::u8)) {
            host_->uni_scvtf(rhs_vmm.s, rhs_vmm.s);
        }

        for (const auto vmm_idx : vmm_idxs) {
            const Vmm dst_vmm(vmm_idx);
            execute_binary(post_op.binary.alg, dst_vmm, host_->P_ALL_ONE,
                    dst_vmm, rhs_vmm, rhs_vmm);
        }

        return;
    }

    // Phase 3.b: general path:
    for (const auto vmm_idx : vmm_idxs) {
        if (vmm_idx == start_idx
                || rhs_arg_params_differ(vmm_idx, vmm_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy,
                    rhs_arg_static_params_.rhs_addr_reg);
        }

        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const Vmm dst_vmm(vmm_idx);
        const bool with_tail = rhs_arg_static_params_.is_tail
                && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);

        if (vmm_preservation_needed) {
            const Vmm vmm_to_preserve(local_vmm_preservation.second);
            push_vmm(host_, vmm_to_preserve);
            inject_binary(
                    post_op, dst_vmm, rhs_arg_addr, with_tail, tail_load_mode);
            pop_vmm(host_, vmm_to_preserve);
            // in case all Vmm are occupied, Vmm(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_vmm(host_, zero_vmm);
            vmm0_was_preserved = true;
        } else
            inject_binary(
                    post_op, dst_vmm, rhs_arg_addr, with_tail, tail_load_mode);
    }
    // ...and restored afterwards
    if (vmm0_was_preserved) pop_vmm(host_, zero_vmm);
}

template <cpu_isa_t isa>
rhs_address_t jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy,
        const Xbyak_aarch64::XReg &rhs_addr) const {

    const auto &rhs_addr_reg = rhs_addr;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    host_->ldr_imm(rhs_addr_reg, param1_, abi_param_offset);

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto rhs_addr_offset = rhs_arg_idx * rhs_arg_ptr_size;
    host_->ldr_imm(rhs_addr_reg, rhs_addr_reg, rhs_addr_offset);

    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::scalar:
            return rhs_address_t(rhs_addr_reg, 0, true);
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_out_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_out_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_out_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_no_broadcast_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg);
        }
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_oc_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? rhs_address_t(rhs_addr_reg, 0, true)
                    : rhs_address_t(rhs_addr_reg);
        }
        case broadcasting_strategy_t::per_mb_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_sp_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_sp_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_sp_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_mb_sp_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg);
        }
        case broadcasting_strategy_t::per_mb_w: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_mb_w_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_mb_w_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_mb_w_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_mb_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg);
        }
        case broadcasting_strategy_t::per_w: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_w_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_w_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_w_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            append_w_offset(rhs_arg_params.vmm_idx_to_out_addr,
                    rhs_arg_params.vmm_idx_to_out_reg,
                    rhs_arg_params.vmm_idx_to_out_elem_off_val, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);

            return rhs_address_t(rhs_addr_reg);
        }
        default: assert(false && "Broadcasting type not supported");
    }

    return rhs_address_t(rhs_addr_reg, 0, true);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, rhs_operand_t> &vmm_idx_to_elem_operand_off,
        int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
        const Xbyak_aarch64::XReg &tmp, std::size_t elem_size_bytes) const {

    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg,
                    Xbyak_aarch64::XReg(it_operand_off->second.idx_));
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            const auto &op = it_operand_off->second;
            if (it_operand_off->second.isAddress_) {
                host_->ldr(tmp,
                        Xbyak_aarch64::ptr(host_->addr_off(op.address_.base_,
                                op.address_.offt_, host_->X_DEFAULT_ADDR,
                                host_->X_TMP_0)));
            } else {
                host_->mov(tmp, Xbyak_aarch64::XReg(op.idx_));
            }
            host_->lsl(tmp, tmp, shift_val);
            host_->add(addr_reg, addr_reg, tmp);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, rhs_address_t> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
        const Xbyak_aarch64::XReg &tmp, std::size_t elem_size_bytes) const {

    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set()) {
        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg,
                    Xbyak_aarch64::XReg(it_off_addr->second.base_));
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->add_imm(tmp, it_off_addr->second.base_,
                    it_off_addr->second.offt_, host_->X_TMP_0);
            host_->lsl(tmp, tmp, shift_val);
            host_->add(addr_reg, addr_reg, tmp);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, size_t> &vmm_idx_to_elem_val_off, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end()
            && !rhs_arg_static_params_.is_dst_orig_set())
        host_->add_imm(addr_reg, addr_reg, it_off_val->second * elem_size_bytes,
                host_->X_TMP_0);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_no_broadcast_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg, const Xbyak_aarch64::XReg &offset,
        std::size_t elem_size_bytes) const {

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        assert(rhs_arg_static_params_.is_dst_orig_set()
                && "dst base addr offset not set");
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                offset);

        if (elem_size_bytes > 1) {
            const int shift_val = std::log2(elem_size_bytes);
            host_->lsl(offset, offset, shift_val);
        }
        host_->add(addr_reg, addr_reg, offset);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_no_broadcast(rhs_address_t addr,
        std::size_t offset, const Xbyak_aarch64::XReg &out_reg) const {
    const auto &tmp = host_->X_TMP_0;
    const auto &param_addr = host_->X_DEFAULT_ADDR;
    host_->add_imm(out_reg, addr.getBase(), addr.offt_, tmp);
    if (offset > 0) host_->add_imm(out_reg, out_reg, offset, tmp);
    host_->ldr(tmp,
            Xbyak_aarch64::ptr(host_->addr_off(param1_,
                    rhs_arg_static_params_.dst_orig_offset, param_addr, tmp)));
    host_->sub(out_reg, out_reg, tmp);
    host_->lsr(out_reg, out_reg,
            std::log2(types::data_type_size(
                    rhs_arg_static_params_.dst_d.data_type())));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_oc_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg, const Xbyak_aarch64::XReg &offset,
        std::size_t elem_size_bytes) const {
    const auto &result = host_->X_TMP_0;

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        assert(rhs_arg_static_params_.is_dst_orig_set()
                && "dst base addr offset not set");
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                offset);

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        // c = result
        switch (layout) {
            case injector_utils::layout_t::ncsp:
                calculate_oc_ncsp(strides, offset, result);
                break;
            case injector_utils::layout_t::c_blocked:
                calculate_oc_blocked(strides, offset, result);
                break;
            case injector_utils::layout_t::nspc:
                calculate_oc_nspc(strides, offset, result);
                break;
            case injector_utils::layout_t::cspn:
                calculate_oc_cspn(strides, offset, result);
                break;
            default: assert(!"Unknown layout");
        }

        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg, result);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(offset, result);
            host_->lsl(offset, offset, shift_val);
            host_->add(addr_reg, addr_reg, offset);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_ncsp(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset, const Xbyak_aarch64::XReg &result,
        const bool residue) const {
    const auto &channel_remainder = host_->X_TMP_1;
    const auto &spatial_offset = host_->X_TMP_2;
    const auto &divisor = host_->X_TMP_3;
    const auto &channel_divisor = host_->X_TMP_4;
    // c = (offset % strides[0]) / strides[1]
    // output = X_TMP_0

    host_->mov_imm(divisor, strides[0]);
    host_->mov_imm(channel_divisor, strides[1]);

    host_->umod(spatial_offset, offset, divisor);
    if (residue)
        host_->udiv_mod(
                result, channel_remainder, spatial_offset, channel_divisor);
    else
        host_->udiv(result, spatial_offset, channel_divisor);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_blocked(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &channel_remainder = host_->X_TMP_1;
    const auto &blocked_remainder = host_->X_TMP_2;
    const auto &divisor = host_->X_TMP_3;
    // c = ((offset % strides[0]) / strides[1]) * strides[ndims - 1] + offset % blk_size
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const int simd_w = simd_elems(dst_d.data_type(), isa);
    const int blk_size = dst_d.blocking_desc().inner_blks[0];

    calculate_oc_ncsp(strides, offset, result, blk_size > simd_w);

    if (blk_size > simd_w) {
        // extract c % blk_size
        host_->mov_imm(divisor, blk_size);
        host_->umod(blocked_remainder, channel_remainder, divisor);
    }

    host_->mov_imm(offset, blk_size);
    host_->mul(result, result, offset);
    if (blk_size > simd_w) host_->add(result, result, blocked_remainder);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_nspc(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // c = offset % C
    // output = X_TMP_0
    const auto C = rhs_arg_static_params_.dst_d.dims()[1];

    host_->mov_imm(divisor, C);
    host_->umod(result, offset, divisor);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_oc_cspn(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // c = offset / strides[1]
    // output = X_TMP_0

    host_->mov_imm(divisor, strides[1]);
    host_->udiv(result, offset, divisor);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_mb_sp_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg, const Xbyak_aarch64::XReg &offset,
        std::size_t elem_size_bytes) const {
    const auto &result = host_->X_TMP_0;

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        assert(rhs_arg_static_params_.is_dst_orig_set()
                && "dst base addr offset not set");
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);

        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                offset);

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        // c = result
        switch (layout) {
            case injector_utils::layout_t::ncsp:
                calculate_mb_sp_ncsp(strides, offset, result);
                break;
            case injector_utils::layout_t::c_blocked:
                calculate_mb_sp_blocked(strides, offset, result);
                break;
            case injector_utils::layout_t::nspc:
                calculate_mb_sp_nspc(strides, offset, result);
                break;
            case injector_utils::layout_t::cspn:
                calculate_mb_sp_cspn(strides, offset, result);
                break;
            default: assert(!"Unknown layout");
        }

        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg, result);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(offset, result);
            host_->lsl(offset, offset, shift_val);
            host_->add(addr_reg, addr_reg, offset);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_sp_ncsp(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &within_batch = host_->X_TMP_1;
    const auto &batch = host_->X_TMP_2;
    const auto &divisor = host_->X_TMP_3;
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = (n * (stride_n/C)) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW)
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;
    const auto W = (ndims >= 3) ? dst_d.dims()[ndims - 1] : 1;

    host_->mov_imm(divisor, strides[0]);
    host_->udiv_mod(batch, within_batch, offset, divisor);
    // batch = n
    host_->mov_imm(divisor, strides[1]);
    host_->udiv(result, within_batch, divisor);
    host_->mul(result, result, divisor);
    // X_TMP_0 = c * stride_c
    host_->sub(offset, offset, result);
    // tmp_reg = offset - c * stride_c
    host_->mov(result, batch);
    // X_TMP_0 = n
    host_->mov_imm(divisor, (C_padded - 1) * D * H * W);
    // n(C - 1)DHW = nCDHW - nDHW
    host_->mul(result, result, divisor);
    // X_TMP_0 = n(C - 1)DHW
    host_->sub(result, offset, result);
    // X_TMP_0 = offset - (c * stride_c) - (n * (C - 1)DHW)
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_sp_blocked(
        const dim_t *strides, const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // mb_sp_off = offset - (c * stride_c) - (n * (C - 1)DHW) - c % blk_size
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const int simd_w = simd_elems(dst_d.data_type(), isa);
    const int blk_size = dst_d.blocking_desc().inner_blks[0];

    if (blk_size > simd_w) {
        // substract c % blk_size
        host_->mov_imm(divisor, blk_size);
        host_->umod(result, offset, divisor);
        host_->sub(offset, offset, result);
    }

    calculate_mb_sp_ncsp(strides, offset, result);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_sp_nspc(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_sp_off = nDHW + dHW + hW + w
    // mb_sp_off = offset / C
    // output = X_TMP_0
    const auto C = rhs_arg_static_params_.dst_d.padded_dims()[1];

    host_->mov_imm(divisor, C);
    host_->umod(result, offset, divisor);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_sp_cspn(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_sp_off = dHWN + hWN + wN + n
    // mb_sp_off = offset % stride_c
    // output = X_TMP_0

    host_->mov_imm(divisor, strides[1]);
    host_->umod(result, offset, divisor);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_mb_w_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg, const Xbyak_aarch64::XReg &offset,
        std::size_t elem_size_bytes) const {
    const auto &result = host_->X_TMP_0;

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        assert(rhs_arg_static_params_.is_dst_orig_set()
                && "dst base addr offset not set");
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                offset);

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        switch (layout) {
            case injector_utils::layout_t::ncsp:
                calculate_mb_w_ncsp(strides, offset, result);
                break;
            case injector_utils::layout_t::c_blocked:
                calculate_mb_w_blocked(strides, offset, result);
                break;
            case injector_utils::layout_t::nspc:
                calculate_mb_w_nspc(strides, offset, result);
                break;
            case injector_utils::layout_t::cspn:
                calculate_mb_w_cspn(strides, offset, result);
                break;
            default: assert(!"Unknown layout");
        }

        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg, result);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(offset, result);
            host_->lsl(offset, offset, shift_val);
            host_->add(addr_reg, addr_reg, offset);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_w_ncsp(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &component = host_->X_TMP_1;
    const auto &batch_divisor = host_->X_TMP_2;
    const auto &divisor = host_->X_TMP_3;
    const auto &batch = host_->X_TMP_4;
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;

    host_->mov_imm(batch_divisor, strides[0]);
    host_->udiv_mod(batch, divisor, offset, batch_divisor);
    // batch = n

    host_->mov_imm(batch_divisor, strides[1]);
    host_->umod(result, divisor, batch_divisor);

    if (ndims >= 5) {
        host_->mov_imm(divisor, strides[ndims - 3]);
        host_->umod(component, result, divisor);
        host_->mov(result, component);
    }
    if (ndims >= 4) {
        host_->mov_imm(divisor, strides[ndims - 2]);
        host_->umod(component, result, divisor);
        host_->mov(result, component);
    }
    if (ndims >= 3) {
        host_->mov_imm(divisor, strides[ndims - 1]);
        host_->udiv(result, result, divisor);
        host_->mul(offset, result, divisor);
        // tmp_reg = w * stride_w
    }
    // tmp_reg = w * stride_w
    host_->mov_imm(divisor, strides[0] / (C_padded * D * H));
    host_->mul(result, batch, divisor);
    // X_TMP_0 = n * (stride_n/(C*D*H))
    if (ndims >= 3) host_->add(result, result, offset);
    // X_TMP_0 = (n * (stride_n/(C*D*H))) + (w * stride_w)
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_w_blocked(
        const dim_t *strides, const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    // mb_w_off = (n * (stride_n/(C*D*H))) + (w * stride_w)
    // output = X_TMP_0
    calculate_mb_sp_ncsp(strides, offset, result);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_w_nspc(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &component = host_->X_TMP_1;
    const auto &batch_divisor = host_->X_TMP_2;
    const auto &divisor = host_->X_TMP_3;
    const auto &batch = host_->X_TMP_4;
    // offset = nDHWC + dHWC + hWC + wC + c
    // mb_w_off = nW + w
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();
    const auto C_padded = dst_d.padded_dims()[1];
    const auto D = (ndims >= 5) ? dst_d.dims()[ndims - 3] : 1;
    const auto H = (ndims >= 4) ? dst_d.dims()[ndims - 2] : 1;

    host_->mov_imm(batch_divisor, strides[0]);
    host_->udiv_mod(batch, result, offset, batch_divisor);
    // batch = n
    if (ndims >= 5) {
        host_->mov_imm(divisor, strides[ndims - 3]);
        host_->umod(component, result, divisor);
        host_->mov(result, component);
    }
    if (ndims >= 4) {
        host_->mov_imm(divisor, strides[ndims - 2]);
        host_->umod(component, result, divisor);
        host_->mov(result, component);
    }
    if (ndims >= 3) {
        host_->mov_imm(divisor, strides[ndims - 1]);
        host_->udiv(offset, result, divisor);
        // tmp_reg = w
    }
    host_->mov_imm(divisor, strides[0] / (D * H * C_padded));
    host_->mul(result, batch, divisor);
    // X_TMP_0 = nW
    if (ndims >= 3) host_->add(result, result, offset);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_mb_w_cspn(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // offset = cDHWN + dHWN + hWN + wN + n
    // mb_w_off = wN + n
    // output = X_TMP_0
    const auto ndims = rhs_arg_static_params_.dst_d.ndims();

    host_->mov_imm(divisor, strides[1]);
    host_->umod(result, offset, divisor);
    if (ndims >= 5) {
        host_->mov_imm(divisor, strides[ndims - 3]);
        host_->umod(offset, result, divisor);
    }
    if (ndims >= 4) {
        host_->mov_imm(divisor, strides[ndims - 2]);
        host_->udiv(result, offset, divisor);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_w_offset(
        const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
        const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
        const std::map<int, size_t> &vmm_idx_to_out_elem_off_val, int vmm_idx,
        const Xbyak_aarch64::XReg &addr_reg, const Xbyak_aarch64::XReg &offset,
        std::size_t elem_size_bytes) const {
    const auto &result = host_->X_TMP_0;

    const auto it_out_addr = vmm_idx_to_out_addr.find(vmm_idx);
    const auto it_out_reg = vmm_idx_to_out_reg.find(vmm_idx);

    const bool is_out_addr = it_out_addr != vmm_idx_to_out_addr.end();
    const bool is_out_reg = it_out_reg != vmm_idx_to_out_reg.end();

    if (is_out_addr || is_out_reg) {
        assert(rhs_arg_static_params_.is_dst_orig_set()
                && "dst base addr offset not set");
        rhs_address_t out_addr = is_out_addr
                ? it_out_addr->second
                : rhs_address_t(it_out_reg->second);
        const auto it_off_val = vmm_idx_to_out_elem_off_val.find(vmm_idx);
        calculate_no_broadcast(out_addr,
                it_off_val != vmm_idx_to_out_elem_off_val.end()
                        ? it_off_val->second
                        : 0,
                offset);

        const auto dst_d = rhs_arg_static_params_.dst_d;
        const auto strides = dst_d.blocking_desc().strides;
        const auto layout = injector_utils::get_layout_type(dst_d);

        switch (layout) {
            case injector_utils::layout_t::ncsp:
                calculate_w_ncsp(strides, offset, result);
                break;
            case injector_utils::layout_t::c_blocked:
                calculate_w_blocked(strides, offset, result);
                break;
            case injector_utils::layout_t::nspc:
                calculate_w_nspc(strides, offset, result);
                break;
            case injector_utils::layout_t::cspn:
                calculate_w_cspn(strides, offset, result);
                break;
            default: assert(!"Unknown layout");
        }

        if (elem_size_bytes == 1) {
            host_->add(addr_reg, addr_reg, result);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(offset, result);
            host_->lsl(offset, offset, shift_val);
            host_->add(addr_reg, addr_reg, offset);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_w_ncsp(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // offset = (n * stride_n) + (c * stride_c) + (d * stride_d) + (h * stride_h) + (w * stride_w)
    // w_off = w * stride_w
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    assert(ndims >= 3);

    host_->mov_imm(divisor, strides[ndims - 2]);
    host_->umod(result, offset, divisor);

    host_->mov_imm(divisor, strides[ndims - 1]);
    host_->udiv(result, result, divisor);
    host_->mul(result, result, divisor);
    // X_TMP_0 = w * stride_w
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_w_blocked(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    calculate_w_ncsp(strides, offset, result);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_w_nspc(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    const auto &divisor = host_->X_TMP_1;
    // offset = nDHWC + dHWC + hWC + wC + c
    // w_off = w
    // output = X_TMP_0
    const auto dst_d = rhs_arg_static_params_.dst_d;
    const auto ndims = dst_d.ndims();

    assert(ndims >= 3);

    host_->mov_imm(divisor, strides[ndims - 2]);
    host_->umod(result, offset, divisor);

    host_->mov_imm(divisor, strides[ndims - 1]);
    host_->udiv(result, result, divisor);
    // X_TMP_0 = w
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::calculate_w_cspn(const dim_t *strides,
        const Xbyak_aarch64::XReg &offset,
        const Xbyak_aarch64::XReg &result) const {
    // offset = cDHWN + dHWN + hWN + wN + n
    // w_off = w
    // output = X_TMP_0
    calculate_w_nspc(strides, offset, result);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, Vmm dst,
        const rhs_address_t &rhs_addr, bool with_tail,
        const tail_lode_mode_t tail_load_mode) const {

    const auto &alg = post_op.binary.alg;
    const bool div_op = alg == alg_kind::binary_div;
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const bool scalar_f32
            = rhs_addr.isBroadcast() && rhs_arg_data_type == data_type::f32;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && (isa == asimd || !scalar_f32);
    const bool process_rhs_arg_using_tmp_vmm
            = rhs_arg_data_type != data_type::f32
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_ || div_op;

    if (process_rhs_arg_using_tmp_vmm) {

        const Vmm tmp_vmm = Vmm(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (rhs_addr.isBroadcast())
            execute_broadcast(rhs_arg_data_type, tmp_vmm,
                    remove_bcast_bit(rhs_addr), tail_load_mode, with_tail);
        else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, tail_load_mode,
                    with_tail);

        // The integral types would have been converted to s32 by the
        // broadcast/load, so convert them to f32 here.
        if (utils::one_of(rhs_arg_data_type, data_type::s32, data_type::s8,
                    data_type::u8)) {
            host_->uni_scvtf(tmp_vmm.s, tmp_vmm.s);
        }

        execute_binary(alg, dst, host_->P_ALL_ONE, dst, tmp_vmm, tmp_vmm);
    } else {
        const auto lhs = dst;
        const bool with_tail_fusable_to_binary_op = with_tail && scalar_f32;
        Xbyak_aarch64::PReg mask = host_->P_ALL_ONE;
        if (with_tail_fusable_to_binary_op) {
            assert(rhs_arg_static_params_.is_opmask_set()
                    && "Opmask is not set for tail loading on SVE");
            const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
            mask = tail_opmask;
            // zero inactive lanes
            host_->mov(Xbyak_aarch64::ZRegS(dst.getIdx()),
                    mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ZRegS(dst.getIdx()));
        }

        const Vmm rhs_scratch(lhs.getIdx() == 0 ? 1 : 0);
        execute_binary(alg, dst, mask, lhs, rhs_addr, rhs_scratch);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast(
        const data_type_t &data_type, const Vmm &tmp_reg,
        const rhs_address_t &rhs_addr, const tail_lode_mode_t tail_load_mode,
        bool with_tail) const {
    if (with_tail) {
        if (tail_load_mode == tail_lode_mode_t::DYNAMIC
                || (tail_load_mode == tail_lode_mode_t::DEFAULT)) {
            execute_broadcast_tail_with_opmask(data_type, tmp_reg, rhs_addr);
        } else
            assert(!"unsupported mode");
    } else
        execute_broadcast_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const data_type_t &data_type,
        const Vmm &tmp_reg, const rhs_address_t &rhs_addr,
        const tail_lode_mode_t tail_load_mode, bool with_tail) const {
    if (with_tail) {
        if (tail_load_mode == tail_lode_mode_t::DYNAMIC
                || (tail_load_mode == tail_lode_mode_t::DEFAULT)) {
            load_rhs_tail_dynamically_with_opmask(data_type, tmp_reg, rhs_addr);
        } else
            load_rhs_tail_statically(data_type, tmp_reg, rhs_addr);
    } else
        load_rhs_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
rhs_address_t jit_uni_binary_injector_t<isa>::remove_bcast_bit(
        const rhs_address_t &rhs_addr) const {
    return rhs_address_t(rhs_addr.base_, rhs_addr.offt_, false, rhs_addr.bits_);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr.base_,
                    rhs_addr.offt_, host_->X_TMP_0);
            if (is_superset(isa, sve)) {
                host_->ld1rw(Xbyak_aarch64::ZRegS(tmp_vmm.getIdx()),
                        host_->P_ALL_ONE / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            } else {
                host_->ld1r(Xbyak_aarch64::VReg4S {tmp_vmm.getIdx()},
                        Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            }
            break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<asimd>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Xbyak_aarch64::VReg &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    assert(utils::one_of(data_type, data_type::s8, data_type::u8)
            && "unsupported data type");

    const Xbyak_aarch64::XReg x_addr = host_->addr_off(rhs_addr.base_,
            rhs_addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);

    if (data_type == data_type::s8) {
        // load byte from memory into wtmp and sign-extend
        host_->ldrsb(host_->W_TMP_0, Xbyak_aarch64::ptr(x_addr));
    } else if (data_type == data_type::u8) {
        // unsigned load
        host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(x_addr));
    }
    // broadcast into 4x32-bit
    host_->dup(tmp_vmm.s4, host_->W_TMP_0);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    assert(utils::one_of(data_type, data_type::s8, data_type::u8)
            && "unsupported data type");

    const auto &p_all = host_->P_ALL_ONE;
    const Xbyak_aarch64::XReg x_addr = host_->addr_off(rhs_addr.base_,
            rhs_addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);
    if (data_type == data_type::s8)
        host_->ld1rsb(tmp_vmm.s, p_all / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(x_addr));
    else if (data_type == data_type::u8)
        host_->ld1rb(tmp_vmm.s, p_all / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(x_addr));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_tail_with_opmask(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading on SVE");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
    Xbyak_aarch64::PReg mask = tail_opmask;
    const Xbyak_aarch64::XReg x_addr = host_->addr_off(rhs_addr.base_,
            rhs_addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);

    switch (data_type) {
        case data_type::f32:
            host_->ld1rw(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_addr));
            break;
        case data_type::s32:
            host_->ld1rw(Xbyak_aarch64::ZRegS(tmp_vmm.getIdx()),
                    mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(x_addr));
            break;
        case data_type::s8:
            host_->ld1rsb(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_addr));
            break;
        case data_type::u8: {
            host_->ld1rb(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_addr));
            break;
        }
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<asimd>::execute_broadcast_tail_with_opmask(
        const data_type_t &data_type, const Xbyak_aarch64::VReg &tmp_vmm,
        const rhs_address_t &rhs_addr) const {

    const Xbyak_aarch64::XReg x_addr = host_->addr_off(
            rhs_addr.base_, rhs_addr.offt_, host_->X_TMP_1, host_->X_TMP_0);

    host_->uni_clear(tmp_vmm);
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->ld1(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[0],
                    Xbyak_aarch64::ptr(x_addr));
            for (size_t i = 1; i < rhs_arg_static_params_.tail_size; ++i) {
                host_->ins(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[i],
                        Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[0]);
            }
            break;
        case data_type::s8:
        case data_type::u8:;
            if (data_type == data_type::s8) {
                host_->ldrsb(host_->W_TMP_0, Xbyak_aarch64::ptr(x_addr));
            } else {
                host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(x_addr));
            }
            for (size_t i = 0; i < rhs_arg_static_params_.tail_size; ++i) {
                host_->ins(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[i],
                        host_->W_TMP_0);
            }
            break;
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr.base_,
                    rhs_addr.offt_, host_->X_TMP_0);
            host_->uni_ldr(tmp_vmm, host_->X_DEFAULT_ADDR);
            break;
        case data_type::s8:
        case data_type::u8:
            load_rhs_i8_no_tail(data_type, tmp_vmm, rhs_addr);
            break;
        default: assert(!"unsupported data type");
    }
}

template <>
void jit_uni_binary_injector_t<asimd>::load_rhs_i8_no_tail(
        const data_type_t &data_type, const Xbyak_aarch64::VReg &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    assert(utils::one_of(data_type, data_type::s8, data_type::u8)
            && "unsupported data type");

    const Xbyak_aarch64::XReg x_addr = host_->addr_off(rhs_addr.base_,
            rhs_addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);

    host_->ld1(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[0],
            Xbyak_aarch64::ptr(x_addr));
    if (data_type == data_type::u8) {
        host_->uxtl(tmp_vmm.h8,
                tmp_vmm.b8); // zero-extend low 8 bytes to 8 uint16 lanes
        host_->uxtl(tmp_vmm.s4, tmp_vmm.h4); // low four u16 -> u32
    } else if (data_type == data_type::s8) {
        host_->sxtl(tmp_vmm.h8,
                tmp_vmm.b8); // zero-extend low 8 bytes to 8 int16 lanes
        host_->sxtl(tmp_vmm.s4, tmp_vmm.h4); // low four s16 -> s32
    } else
        assert(!"unsupported data type");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_i8_no_tail(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    if (data_type == data_type::s8) {
        host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr.base_, rhs_addr.offt_,
                host_->X_TMP_0);
        host_->ld1sb(tmp_vmm.s, host_->P_ALL_ONE,
                Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
    } else if (data_type == data_type::u8) {
        host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr.base_, rhs_addr.offt_,
                host_->X_TMP_0);
        host_->ld1b(tmp_vmm.s, host_->P_ALL_ONE,
                Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
    } else
        assert(!"unsupported data type");
}

template <>
void jit_uni_binary_injector_t<asimd>::load_rhs_tail_dynamically_with_opmask(
        const data_type_t &data_type, const Xbyak_aarch64::VReg &tmp_vmm,
        const rhs_address_t &rhs_addr) const {

    const Xbyak_aarch64::XReg x_addr = host_->addr_off(
            rhs_addr.base_, rhs_addr.offt_, host_->X_TMP_1, host_->X_TMP_0);

    host_->uni_clear(tmp_vmm);
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            for (size_t i = 0; i < rhs_arg_static_params_.tail_size; ++i) {
                const Xbyak_aarch64::XReg lane_addr
                        = host_->addr_off(x_addr, i * sizeof(float),
                                host_->X_DEFAULT_ADDR, host_->X_TMP_0);
                host_->ld1(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[i],
                        Xbyak_aarch64::ptr(lane_addr));
            }
            break;
        case data_type::s8:
        case data_type::u8:
            for (size_t i = 0; i < rhs_arg_static_params_.tail_size; ++i) {
                const Xbyak_aarch64::XReg lane_addr
                        = host_->addr_off(x_addr, i * sizeof(int8_t),
                                host_->X_DEFAULT_ADDR, host_->X_TMP_0);
                if (data_type == data_type::s8) {
                    host_->ldrsb(host_->W_TMP_0, Xbyak_aarch64::ptr(lane_addr));
                } else {
                    host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(lane_addr));
                }
                host_->ins(Xbyak_aarch64::VReg4S(tmp_vmm.getIdx())[i],
                        host_->W_TMP_0);
            }
            break;
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail_dynamically_with_opmask(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading on SVE");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
    Xbyak_aarch64::PReg mask = tail_opmask;

    host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr.base_, rhs_addr.offt_,
            host_->X_TMP_0);
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->ld1w(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            break;
        case data_type::s8:
            host_->ld1sb(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            break;
        case data_type::u8:
            host_->ld1b(tmp_vmm.s, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            break;
        default: assert(!"unsupported data type");
    }
}

/**
* load_bytes is the utility function to facilitate loading of
* load_size (0 <= load_size <= 32) many contiguous bytes into the ZReg
* register from the memory referenced by ptr[reg + offset] address.
*
* TODO: Add an option to zero-out unloaded bytes in the ZReg register.
* TODO: Add an option for unsafe_load wherein one could read outside the
* provided memory buffer so as to minimize the total number of read
* memory instructions.
*/
static void load_bytes(jit_generator_t *host, const Xbyak_aarch64::ZReg &vmm,
        const Xbyak_aarch64::XReg reg_addr, int load_size) {

    // Ensure data fits completely inside the ZReg register
    assert(load_size >= 0 && load_size <= 32);

    // Ensure that vector register is compatible with the ISA in hand
    assert(mayiuse(sve));

    if (load_size == 0) {
        return;
    } else {
        host->set_preg(host->P_TMP.b, load_size, host->X_TMP_3);
        host->ld1b(vmm.b, host->P_TMP / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(reg_addr));
    }
}

/**
* load_bytes_to_dword_extension is the utility function to facilitate
* loading of load_size (0 <= load_size <= 16) many contiguous bytes in
* the ZReg register from the memory referenced by ptr[reg + offset]
* address and then do signed/zero extension of those to double words.
*
* Valid values for the load_size variable are:
* [0..8] for ZReg version of the function.
* TODO: Implement this routine for every ISA.
*/
static void load_bytes_to_dword_extension(jit_generator_t *host,
        const Xbyak_aarch64::ZReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        bool is_signed, int load_size) {
    if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_256) {
        // Ensure extended double words fit inside ZReg (32 * load_size <= 256)
        assert(load_size >= 0 && load_size <= 8);
    } else if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_128) {
        // For ZReg register, load capacity is halved (32 * load_size <= 128)
        assert(load_size >= 0 && load_size <= 4);
    } else {
        assert(!"routine is not supported for the current isa");
    }
    // For load_size == 8/4, do load/extension in one go
    if (load_size == 8) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(vmm.s, host->P_NOT_256,
                    Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        } else {
            host->ld1b(vmm.s, host->P_NOT_256,
                    Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        }
    } else if (load_size == 4) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(vmm.s, host->P_NOT_128,
                    Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        } else {
            host->ld1b(vmm.s, host->P_NOT_128,
                    Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        }
    } else {
        load_bytes(host, vmm, reg_addr, load_size);
        if (is_signed) {
            host->mov(
                    vmm.d, host->P_ALL_ONE, Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(
                    Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128, vmm.d);
        } else {
            host->mov(
                    vmm.d, host->P_ALL_ONE, Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(
                    Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128, vmm.d);
        }
    }
}

static void load_data(jit_generator_t *host, data_type_t type_in,
        const Xbyak_aarch64::ZReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        int offset, int load_size) {
    host->add_imm(host->X_TMP_4, reg_addr, offset, host->X_TMP_0);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            load_bytes(host, vmm, host->X_TMP_4, sizeof(int32_t) * load_size);
            break;
        case data_type::s8:
        case data_type::u8:
            load_bytes_to_dword_extension(host, vmm, host->X_TMP_4,
                    type_in == data_type::s8, load_size);
            break;
        default: assert(!"unsupported source data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail_dynamically_with_gpr(
        const data_type_t &data_type, const Vmm &tmp_vmm) const {
    const Xbyak_aarch64::XReg &reg_addr = rhs_arg_static_params_.rhs_addr_reg;
    const Xbyak_aarch64::XReg &reg_tmp = rhs_arg_static_params_.rhs_helper_reg;
    const Xbyak_aarch64::XReg &reg_tail_size
            = rhs_arg_static_params_.reg_tail_size;
    const Xbyak_aarch64::ZReg x = Xbyak_aarch64::ZReg(tmp_vmm.getIdx());

    auto runtime_tail_load = [&](int load_size) {
        load_data(host_, data_type, x, reg_addr, 0, load_size);
    };

    host_->runtime_tail_process<isa>(reg_tail_size, reg_tmp, runtime_tail_load);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail_statically(
        const data_type_t &data_type, const Vmm &tmp_vmm,
        const rhs_address_t &rhs_addr) const {
    assert(!"unsupported tail load mode");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_cmp_mask(
        const Xbyak_aarch64::PReg &cmp_dst, const Xbyak_aarch64::PReg &mask,
        const Xbyak_aarch64::ZReg &cmp_src, const Xbyak_aarch64::ZReg &cmp_src2,
        alg_kind_t cmp_predicate) const {

    using namespace alg_kind;
    switch (cmp_predicate) {
        case binary_ge:
            host_->fcmge(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        case binary_gt:
            host_->fcmgt(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        case binary_le:
            host_->fcmle(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        case binary_lt:
            host_->fcmlt(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        case binary_eq:
            host_->fcmeq(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        case binary_ne:
            host_->fcmne(cmp_dst.s, mask / Xbyak_aarch64::T_z, cmp_src.s,
                    cmp_src2.s);
            break;
        default: assert(!"unsupported compare mode"); break;
    }
}

template <typename T>
void get_z_rhs_value(
        jit_generator_t *host, const T &rhs, const Xbyak_aarch64::ZReg &z_tmp) {
    const bool is_reg = std::is_same<T, Xbyak_aarch64::ZReg>::value;
    if (is_reg) {
        const Xbyak_aarch64::ZReg &rst_reg = (Xbyak_aarch64::ZReg &)rhs;
        host->mov(z_tmp.d, rst_reg.d);
    } else {
        const rhs_address_t &rst_adr = (rhs_address_t &)rhs;
        host->add_imm(host->X_DEFAULT_ADDR, rst_adr.base_, rst_adr.offt_,
                host->X_TMP_0);
        if (rst_adr.isBroadcast_)
            host->ld1rw(z_tmp.s, host->P_ALL_ONE, ptr(host->X_DEFAULT_ADDR));
        else
            host->ld1w(z_tmp.s, host->P_ALL_ONE, ptr(host->X_DEFAULT_ADDR));
    }
}
template <typename T>
void get_v_rhs_value(
        jit_generator_t *host, const T &rhs, const Xbyak_aarch64::VReg &v_tmp) {
    const bool is_reg = std::is_same<T, Xbyak_aarch64::VReg>::value;
    if (is_reg) {
        const Xbyak_aarch64::VReg &rst_reg = (Xbyak_aarch64::VReg &)rhs;
        (void)rst_reg;
    } else {
        const rhs_address_t &rst_adr = (rhs_address_t &)rhs;
        host->add_imm(host->X_DEFAULT_ADDR, rst_adr.base_, rst_adr.offt_,
                host->X_TMP_0);
        host->ld1r(v_tmp.s4, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
    }
}

template <>
void jit_uni_binary_injector_t<asimd>::execute_cmp_binary(
        const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::PReg &mask,
        const Xbyak_aarch64::VReg &lhs, const Xbyak_aarch64::VReg &rhs,
        alg_kind_t cmp_predicate) const {
    UNUSED(mask);

    using namespace alg_kind;
    switch (cmp_predicate) {
        case binary_ge: host_->fcmge(dst.s, lhs.s, rhs.s); break;
        case binary_gt: host_->fcmgt(dst.s, lhs.s, rhs.s); break;
        case binary_le: host_->fcmge(dst.s, rhs.s, lhs.s); break;
        case binary_lt: host_->fcmgt(dst.s, rhs.s, lhs.s); break;
        case binary_eq: host_->fcmeq(dst.s, lhs.s, rhs.s); break;
        case binary_ne:
            host_->fcmeq(dst.s, lhs.s, rhs.s);
            host_->not_(dst.b, dst.b);
            break;
        default: assert(!"unsupported compare mode"); break;
    }

    // ASIMD compare instructions produce all-ones/all-zeros integer masks.
    // Convert them to the expected 1.0f / 0.0f compare result in place.
    host_->ushr(dst.s, dst.s, 31);
    host_->scvtf(dst.s, dst.s);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_cmp_binary(const Vmm &dst,
        const Xbyak_aarch64::PReg &mask, const Vmm &lhs, const Vmm &rhs,
        alg_kind_t cmp_predicate) const {
    // For GreaterEqual op, replace 0xFFFFFFFF by 1
    // which was returned by vcmpps.
    const auto &cmp_mask = rhs_arg_static_params_.tail_opmask;

    push_opmask(host_, cmp_mask);
    compute_cmp_mask(cmp_mask, mask, lhs, rhs, cmp_predicate);
    // broadcast 1.0f with mask
    host_->uni_clear(dst);
    host_->fmov(dst.s, cmp_mask / Xbyak_aarch64::T_m, 1.0);
    // pop tail mask from stack
    pop_opmask(host_, cmp_mask);
}

template <>
template <typename T>
void jit_uni_binary_injector_t<asimd>::execute_binary(alg_kind_t binary_alg,
        const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::PReg &mask,
        const Xbyak_aarch64::VReg &lhs, const T &rhs,
        const Xbyak_aarch64::VReg &rhs_scratch) const {
    const int SIMD_SZ = simd_bytes(asimd);
    const bool isAddr = std::is_same<T, rhs_address_t>::value;
    Xbyak_aarch64::VReg v_rhs(0);

    if (isAddr) {
        const rhs_address_t &addr = (rhs_address_t &)rhs;
        v_rhs = rhs_scratch;
        // Allocate space on stack
        host_->sub(host_->sp, host_->sp, SIMD_SZ);
        // Store the scratch register into the stack so it's safe to clobber it.
        host_->str(Xbyak_aarch64::QReg(v_rhs.getIdx()),
                Xbyak_aarch64::ptr(host_->sp));
        // Compute the effective address we want to load from.
        Xbyak_aarch64::XReg x_addr = host_->addr_off(
                addr.base_, addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);
        // Load into the scratch vector, with or without broadcast.
        if (addr.isBroadcast_)
            host_->ld1r(v_rhs.s, Xbyak_aarch64::ptr(x_addr));
        else
            host_->ld1(v_rhs.s, Xbyak_aarch64::ptr(x_addr));
    } else {
        const Xbyak_aarch64::VReg tmp = (Xbyak_aarch64::VReg &)rhs;
        v_rhs = tmp;
    }

    switch (binary_alg) {
        case alg_kind::binary_add:
            host_->uni_fadd(dst.s, lhs.s, v_rhs.s);
            break;
        case alg_kind::binary_mul:
            host_->uni_fmul(dst.s, lhs.s, v_rhs.s);
            break;
        case alg_kind::binary_max:
            host_->uni_fmax(dst.s, lhs.s, v_rhs.s);
            break;
        case alg_kind::binary_min:
            host_->uni_fmin(dst.s, lhs.s, v_rhs.s);
            break;
        case alg_kind::binary_div:
            host_->uni_fdiv(dst.s, lhs.s, v_rhs.s,
                    Xbyak_aarch64::VReg(jit_generator_t::DUMMY_IDX).s, mask);
            break;
        case alg_kind::binary_sub:
            host_->uni_fsub(dst.s, lhs.s, v_rhs.s);
            break;
        case alg_kind::binary_ge:
        case alg_kind::binary_gt:
        case alg_kind::binary_le:
        case alg_kind::binary_lt:
        case alg_kind::binary_eq:
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, mask, lhs, v_rhs, binary_alg);
            break;
        default: assert(!"unsupported algorithm");
    }

    if (isAddr) {
        // Restore the scratch register's initial content from the stack.
        host_->ldr(Xbyak_aarch64::QReg(v_rhs.getIdx()),
                Xbyak_aarch64::ptr(host_->sp));
        // Free allocated stack space.
        host_->add(host_->sp, host_->sp, SIMD_SZ);
    }
}

template <cpu_isa_t isa>
template <typename T>
void jit_uni_binary_injector_t<isa>::execute_binary(alg_kind_t binary_alg,
        const Vmm &dst, const Xbyak_aarch64::PReg &mask, const Vmm &lhs,
        const T &rhs, const Vmm &rhs_scratch) const {
    const bool isAddr = !std::is_same<T, Xbyak_aarch64::ZReg>::value;
    Vmm z_rhs(0);

    if (isAddr) {
        const rhs_address_t &addr = (rhs_address_t &)rhs;
        z_rhs = rhs_scratch;
        // Allocate 1 VL space on stack
        host_->addvl(host_->sp, host_->sp, -1);
        // Store the scratch register into the stack so it's safe to clobber it.
        host_->str(z_rhs, Xbyak_aarch64::ptr(host_->sp));
        // Compute the effective address we want to load from.
        Xbyak_aarch64::XReg x_addr = host_->addr_off(
                addr.base_, addr.offt_, host_->X_DEFAULT_ADDR, host_->X_TMP_0);
        // Load into the scratch vector, with or without broadcast.
        if (addr.isBroadcast_)
            host_->ld1rw(z_rhs.s, mask, Xbyak_aarch64::ptr(x_addr));
        else
            host_->ld1w(z_rhs.s, mask, Xbyak_aarch64::ptr(x_addr));
    } else {
        const Vmm vmm = (Vmm &)rhs;
        z_rhs = vmm;
    }

    switch (binary_alg) {
        case alg_kind::binary_add:
            host_->uni_fadd(dst.s, lhs.s, z_rhs.s);
            break;
        case alg_kind::binary_mul:
            host_->uni_fmul(dst.s, lhs.s, z_rhs.s);
            break;
        case alg_kind::binary_max:
            host_->uni_fmax(dst.s, lhs.s, z_rhs.s);
            break;
        case alg_kind::binary_min:
            host_->uni_fmin(dst.s, lhs.s, z_rhs.s);
            break;
        case alg_kind::binary_div:
            host_->uni_fdiv(dst.s, lhs.s, z_rhs.s,
                    Vmm(jit_generator_t::DUMMY_IDX).s, mask);
            break;
        case alg_kind::binary_sub:
            host_->uni_fsub(dst.s, lhs.s, z_rhs.s);
            break;
        case alg_kind::binary_ge:
        case alg_kind::binary_gt:
        case alg_kind::binary_le:
        case alg_kind::binary_lt:
        case alg_kind::binary_eq:
        case alg_kind::binary_ne:
            execute_cmp_binary(dst, mask, lhs, z_rhs, binary_alg);
            break;
        default: assert(!"unsupported algorithm");
    }

    if (isAddr) {
        // Restore the scratch register's initial content from the stack.
        host_->ldr(z_rhs, Xbyak_aarch64::ptr(host_->sp));
        // Free allocated stack space.
        host_->addvl(host_->sp, host_->sp, 1);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<sve>;
template class jit_uni_binary_injector_t<asimd>;

} // namespace binary_injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
