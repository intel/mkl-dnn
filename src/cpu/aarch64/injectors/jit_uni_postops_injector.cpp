/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2022-2023 FUJITSU LIMITED
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
#include <cassert>
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector {

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, Xbyak_aarch64::RegPoolManager &reg_pool,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const Xbyak_aarch64::XReg &eltwise_table,
        const Xbyak_aarch64::PReg &eltwise_mask, bool is_fwd, bool use_dst,
        data_type_t d_type, const lambda_jit_injectors_t &lambda_jit_injectors)
    : post_ops_(post_ops)
    , host_(host)
    , reg_pool_(reg_pool)
    , binary_injector_(nullptr)
    , lambda_jit_injectors_(lambda_jit_injectors) {

    bool is_binary = false;
    bool is_eltwise = false;

    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            is_eltwise = true;
            alg_to_eltwise_injector_.emplace(i,
                    jit_uni_eltwise_injector_t<to_vla_sve(isa)>(host_, reg_pool,
                            post_op.eltwise, eltwise_table, eltwise_mask,
                            is_fwd, use_dst, d_type));
        } else if (post_op.is_binary()) {
            is_binary = true;
        }
    }

    if (is_superset(isa, sve_128) && is_eltwise && is_binary
            && binary_static_params.rhs_arg_static_params.tail_size)
        assert(eltwise_mask.getIdx()
                        != binary_static_params.rhs_arg_static_params
                                   .tail_opmask.getIdx()
                && "Binary tail opmask should differ from the eltwise mask");

    if (is_binary)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<isa>>(
                host, reg_pool, binary_static_params);
}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, Xbyak_aarch64::RegPoolManager &reg_pool,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const Xbyak_aarch64::XReg &eltwise_table,
        const Xbyak_aarch64::PReg &eltwise_mask, bool is_fwd, bool use_dst,
        data_type_t d_type)
    : jit_uni_postops_injector_t(host, reg_pool, post_ops, binary_static_params,
              eltwise_table, eltwise_mask, is_fwd, use_dst, d_type,
              lambda_jit_injectors_t()) {}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    compute_vector_range(
            start_idx, end_idx, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    injector_utils::compute_vector_range_with_spill<isa>(host_, reg_pool_,
            vmm_idxs, scratch_vmm_count(),
            [&](const injector_utils::vmm_index_set_t &idxs) {
        compute_vector_range_no_spill(idxs, rhs_arg_params);
    });
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range_no_spill(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    std::size_t rhs_arg_idx = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(i).load_table_addr();
            alg_to_eltwise_injector_.at(i).compute_vector_range(vmm_idxs);
        } else if (post_op.is_binary()) {
            binary_injector_->compute_vector_range(
                    vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
            ++rhs_arg_idx;
        } else {
            const auto lam = lambda_jit_injectors_.find(post_op.kind);
            if (lam != lambda_jit_injectors_.end()) lam->second(vmm_idxs);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::prepare_table(bool gen_table) {
    for (auto &alg_elt_inject : alg_to_eltwise_injector_)
        alg_elt_inject.second.prepare_table(gen_table);
}

template <cpu_isa_t isa>
size_t jit_uni_postops_injector_t<isa>::scratch_vmm_count() const {
    size_t count = 0;
    for (int i = 0; i < post_ops_.len(); ++i) {
        const auto &entry = post_ops_.entry_[i];
        if (entry.is_binary())
            count = nstl::max<size_t>(count, 1);
        else if (entry.is_eltwise())
            count = nstl::max(
                    count, alg_to_eltwise_injector_.at(i).scratch_vmm_count());
    }
    return count;
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range({idx}, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(size_t idx) {
    compute_vector_range({idx});
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::set_lambda_injector(
        lambda_jit_injectors_t::key_type kind,
        const lambda_jit_injectors_t::mapped_type &jit_injector) {
    lambda_jit_injectors_[kind] = jit_injector;
}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::initializer_list<post_op_type> accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
        const bool sum_requires_zp_zero, const bool sum_requires_same_params,
        const bcast_set_t &enabled_bcast_strategy)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d)
    , sum_at_pos_0_only(sum_at_pos_0_only)
    , sum_requires_scale_one(sum_requires_scale_one)
    , sum_requires_zp_zero(sum_requires_zp_zero)
    , sum_requires_same_params(sum_requires_same_params)
    , enabled_bcast_strategy(enabled_bcast_strategy) {};

bool post_ops_ok(const post_ops_ok_args_t &post_ops_ok_args) {
    const auto &args = post_ops_ok_args;

    const post_ops_t &post_ops = args.post_ops;

    // Save scale and zero point of first sum postop in order to check that any
    // subsequent sum postops have the same values. This check is necessary
    // because there is only one lambda injector.
    const auto first_sum_idx = post_ops.find(primitive_kind::sum);
    const bool with_sum = first_sum_idx != -1;
    const auto &maybe_sum_entry = with_sum ? post_ops.entry_[first_sum_idx]
                                           : dnnl_post_ops::entry_t();
    const auto first_sum_scale = with_sum ? maybe_sum_entry.sum.scale : 0;
    const auto first_sum_zero_point
            = with_sum ? maybe_sum_entry.sum.zero_point : 0;

    const auto is_accepted_postop = [&](const int idx) {
        for (const auto &post_op : args.accepted_post_op_types) {
            const auto &entry = post_ops.entry_[idx];
            switch (post_op) {
                case sum:
                    if (entry.is_sum(false, false)) {
                        if (args.sum_requires_same_params
                                && entry.sum.scale != first_sum_scale)
                            return false;
                        if (args.sum_requires_same_params
                                && entry.sum.zero_point != first_sum_zero_point)
                            return false;
                        if (args.sum_requires_scale_one && entry.sum.scale != 1)
                            return false;
                        if (args.sum_requires_zp_zero
                                && entry.sum.zero_point != 0)
                            return false;
                        return IMPLICATION(args.sum_at_pos_0_only, idx == 0);
                    }
                    break;
                case eltwise:
                    if (entry.is_eltwise()) {
                        const auto alg = entry.eltwise.alg;
                        return eltwise_injector::is_supported(
                                to_vla_sve(args.isa), alg);
                    }
                    break;
                case binary:
                    // TODO: support alg_kind::prelu with entry.is_like_binary()
                    if (entry.is_binary()) {
                        const auto alg = entry.binary.alg;

                        return binary_injector::is_supported(
                                to_vla_sve(args.isa), alg,
                                entry.binary.src1_desc, *args.dst_d,
                                args.enabled_bcast_strategy);
                    }
                    break;
                default: assert(false && "Unhandled post_op type");
            }
        }
        return false;
    };

    for (int i = 0; i < post_ops.len(); i++) {
        if (!is_accepted_postop(i)) return false;
    }

    return true;
}

template class jit_uni_postops_injector_t<sve>;
template class jit_uni_postops_injector_t<asimd>;

} // namespace injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
