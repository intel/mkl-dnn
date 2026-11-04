/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#include "graph/backend/dnnl/executables/layer_norm.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

layernorm_executable_t::desc_t layernorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::layer_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }

    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    float epsilon = 1e-5f;
    if (op->has_attr(op_attr::epsilon))
        epsilon = op->get_attr<float>(op_attr::epsilon);
    bool keep_stats = true;
    if (op->has_attr(op_attr::keep_stats))
        keep_stats = op->get_attr<bool>(op_attr::keep_stats);
    bool use_affine = true;
    if (op->has_attr(op_attr::use_affine))
        use_affine = op->get_attr<bool>(op_attr::use_affine);
    bool is_rms = false;
    if (op->has_attr(op_attr::is_rms))
        is_rms = op->get_attr<bool>(op_attr::is_rms);

    auto flags = dnnl::normalization_flags::none;
    if (is_rms) flags |= dnnl::normalization_flags::rms_norm;
    if (use_affine) {
        flags |= dnnl::normalization_flags::use_scale;
        // no shift for rms norm
        if (!is_rms) flags |= dnnl::normalization_flags::use_shift;
    }
    prop_kind pkind = keep_stats ? prop_kind::forward_training
                                 : prop_kind::forward_inference;

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    // onednn 3.6 spec: Implementations optimized for memory formats ab, abc,
    // bac, abcd
    src = to_ncx_format(src);
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);
    dnnl::layer_normalization_forward::primitive_desc pd;
    if (use_affine) {
        memory::data_type scale_shift_data_type
                = static_cast<memory::data_type>(
                        op->get_input_logical_tensor(1).data_type);
        pd = dnnl::layer_normalization_forward::primitive_desc(p_engine, pkind,
                src, dst, scale_shift_data_type, epsilon, flags, prm_attr);
    } else {
        pd = dnnl::layer_normalization_forward::primitive_desc(
                p_engine, pkind, src, dst, epsilon, flags, prm_attr);
    }

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

layernorm_bwd_executable_t::desc_t layernorm_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::layer_normalization_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto epsilon = op->get_attr<float>(op_attr::epsilon);
    auto flags = dnnl::normalization_flags::none;
    const bool use_affine = op->get_attr<bool>(op_attr::use_affine);
    if (use_affine) {
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    }

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dnnl::layer_normalization_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, src, diff_dst, epsilon, flags,
            prm_attr);

    dnnl::layer_normalization_backward::primitive_desc pd(p_engine,
            prop_kind::backward, diff_src, diff_dst, src, epsilon, flags,
            fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

arg_indices_t layernorm_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_norm(op);
}

arg_indices_t layernorm_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    // inputs
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 1}});
    args.insert({DNNL_ARG_MEAN, {indices_t::type_t::input, 2}});
    args.insert({DNNL_ARG_VARIANCE, {indices_t::type_t::input, 3}});

    if (op->num_inputs() > 4) {
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, 4}});
        if (op->num_inputs() > 5) {
            args.insert({DNNL_ARG_SHIFT, {indices_t::type_t::input, 5}});
        } else {
            // use scale mem for fake shift
            args.insert({DNNL_ARG_SHIFT, {indices_t::type_t::input, 4}});
        }
    }
    // outputs
    size_t ind = 0;
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, ind++}});
    if (op->get_attr<bool>(op_attr::use_affine)) {
        args.insert({DNNL_ARG_DIFF_SCALE, {indices_t::type_t::output, ind++}});
        args.insert({DNNL_ARG_DIFF_SHIFT, {indices_t::type_t::output, ind++}});
    }
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, ind++}});
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
