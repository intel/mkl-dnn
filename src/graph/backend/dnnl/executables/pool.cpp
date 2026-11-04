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

#include "graph/backend/dnnl/executables/pool.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

pool_executable_t::desc_t pool_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::pooling_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dims strides = op->get_attr<dims>(op_attr::strides);
    dims kernel = op->get_attr<dims>(op_attr::kernel);
    dims pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    dims pads_end = op->get_attr<dims>(op_attr::pads_end);
    dims dilations(strides.size(), 1);
    if (op->has_attr(op_attr::dilations)
            && (op->get_attr<std::string>(op_attr::kind) == "maxpool")) {
        dilations = op->get_attr<dims>(op_attr::dilations);
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);

    // infer dnnl explicit padding
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr(op_attr::rounding_type)) {
        rounding_type = op->get_attr<std::string>(op_attr::rounding_type);
    }

    // oneDNN pooling primitive doesn't support ceil mode, so we need to add
    // additional padding right to simulate the ceil mode by using floor mode,
    // and then exclude those additional paddings when doing average.
    if (rounding_type == "ceil") {
        dims src_sp = src.get_dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = dst.get_dims();
        output_sp.erase(output_sp.begin(), output_sp.begin() + 2);
        for (size_t i = 0; i < kernel.size(); ++i) {
            dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
            // calculate the expected padded input size according to floor mode
            // formula: output = (padded - dilated) / strides + 1
            dim_t expected_padded = (output_sp[i] - 1) * strides[i] + dilated;
            dim_t cur_pads_end = expected_padded - src_sp[i] - pads_begin[i];
            pads_end[i] = cur_pads_end;
        }
        adj_pad = true;
    }

    algorithm algo = algorithm::undef;
    prop_kind prop = prop_kind::forward_inference;
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        algo = algorithm::pooling_max;
        dilations = get_compatible_dilates(dilations, src.get_ndims());
        if (op->num_outputs() == 3) {
            prop = prop_kind::forward_training;
            op->set_attr<bool>(op_attr::is_training, true);
        }
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        dilations = dims(src.get_ndims(), 0);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        assert(!"only int8 MaxPool/AvgPool is supported.");
    }

    dnnl::pooling_forward::primitive_desc pd(p_engine, prop, algo, src, dst,
            strides, kernel, dilations, pads_begin, pads_end, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

pool_bwd_executable_t::desc_t pool_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::pooling_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dims strides = op->get_attr<dims>(op_attr::strides);
    dims kernel = op->get_attr<dims>(op_attr::kernel);
    dims pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    dims pads_end = op->get_attr<dims>(op_attr::pads_end);
    dims dilations(strides.size(), 0);
    if (op->has_attr(op_attr::dilations)) {
        dilations = op->get_attr<dims>(op_attr::dilations);
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    auto src = op->get_attr<std::string>(op_attr::kind) == "maxpool"
            ? make_dnnl_memory_desc(op->get_input_logical_tensor(2))
            : dnnl::memory::desc(diff_src.get_dims(), diff_src.get_data_type(),
                      get_ncx_format(diff_src.get_dims()));

    // infer dnnl explicit pad
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr(op_attr::rounding_type)) {
        rounding_type = op->get_attr<std::string>(op_attr::rounding_type);
    }
    if (rounding_type == "ceil") {
        dims src_sp = src.get_dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = diff_dst.get_dims();
        output_sp.erase(output_sp.begin(), output_sp.begin() + 2);
        for (size_t i = 0; i < kernel.size(); ++i) {
            dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
            if (op->get_attr<std::string>(op_attr::kind) == "avgpool")
                dilated += 1;
            dim_t cur_pads_end = (output_sp[i] - 1) * strides[i] + dilated
                    - src_sp[i] - pads_begin[i];
            pads_end[i] = cur_pads_end;
        }
        adj_pad = true;
    }

    algorithm algo = algorithm::undef;
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        algo = algorithm::pooling_max;
        dilations = get_compatible_dilates(dilations, src.get_ndims());
    } else if (op->get_attr<std::string>(op_attr::kind) == "avgpool") {
        const bool exclude_pad = op->get_attr<bool>(op_attr::exclude_pad);
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        assert(!"only MaxPoolBackprop/AvgPoolBackprop is supported.");
    }

    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        diff_dst = to_format_any(diff_dst);
    }

    dnnl::pooling_forward::primitive_desc forward_hints
            = dnnl::pooling_forward::primitive_desc(p_engine,
                    prop_kind::forward_training, algo, src, diff_dst, strides,
                    kernel, dilations, pads_begin, pads_end, prm_attr);

    dnnl::pooling_backward::primitive_desc pd(p_engine, algo, diff_src,
            diff_dst, strides, kernel, dilations, pads_begin, pads_end,
            forward_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t pool_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

arg_indices_t pool_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;

    // inputs
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 0}});
    if (op->get_attr<std::string>(op_attr::kind) == "maxpool") {
        // maxpool bwd op must need workspace input
        args.insert({DNNL_ARG_WORKSPACE, {indices_t::type_t::input, 1}});
    }
    // outputs
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
