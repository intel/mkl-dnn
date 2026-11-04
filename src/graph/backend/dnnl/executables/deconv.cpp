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

#include "graph/backend/dnnl/executables/deconv.hpp"
#include "graph/backend/dnnl/executables/conv.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// deconv_fwd_executable_t implementations
void deconv_fwd_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            dnnl::reorder(psrc_mem, dst_mem)
                    .execute(stream, const_cast<memory &>(psrc_mem),
                            const_cast<memory &>(dst_mem));
        }
    }

    prim_.execute(stream, args);
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> deconv_fwd_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    auto sycl_deps = deps;
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            auto prim = dnnl::reorder(psrc_mem, dst_mem);
            auto e = dnnl::sycl_interop::execute(prim, stream,
                    {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                            {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                    sycl_deps);
            sycl_deps = {e};
        }
    }
    auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
    if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
    return e;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t deconv_fwd_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<cl_event> ocl_deps(deps.begin(), deps.end());
    ocl_event_t reorder_event;
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            auto prim = dnnl::reorder(psrc_mem, dst_mem);
            reorder_event = ocl_event_t(dnnl::ocl_interop::execute(prim, stream,
                    {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                            {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                    ocl_deps));
            ocl_deps.assign(1, reorder_event.get());
        }
    }
    auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
    return ocl_event_t(e);
}
#endif

deconv_fwd_executable_t::desc_t deconv_fwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    set_graph_attributes(prm_attr, graph_attr);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    src = to_format_any(src);
    auto weight = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    weight = to_format_any(weight);
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);

    dnnl::deconvolution_forward::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
        bias = to_format_any(bias);
        pd = dnnl::deconvolution_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::deconvolution_direct,
                src, weight, bias, dst, strides, dilates, pads_begin, pads_end,
                prm_attr);
    } else {
        pd = dnnl::deconvolution_forward::primitive_desc(p_engine,
                prop_kind::forward_inference, algorithm::deconvolution_direct,
                src, weight, dst, strides, dilates, pads_begin, pads_end,
                prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t deconv_fwd_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_conv_and_matmul(op);
}

// deconv_bwd_data_executable_t implementations
deconv_bwd_data_executable_t::desc_t deconv_bwd_data_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_backward_data::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    set_graph_attributes(prm_attr, graph_attr);

    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    diff_dst = to_format_any(diff_dst);
    auto weight = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    weight = to_format_any(weight);
    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    diff_src = to_format_any(diff_src);

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(p_engine,
            prop_kind::forward_training, algorithm::deconvolution_direct,
            diff_src, weight, diff_dst, strides, dilates, pads_begin, pads_end,
            prm_attr);

    dnnl::deconvolution_backward_data::primitive_desc pd(p_engine,
            dnnl::algorithm::deconvolution_direct, diff_src, weight, diff_dst,
            strides, pads_begin, pads_end, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t deconv_bwd_data_executable_t::get_arg_indices(const op_t *op) {
    return conv_bwd_data_executable_t::get_arg_indices(op);
}

// deconv_bwd_weights_executable_t implementations
deconv_bwd_weights_executable_t::desc_t
deconv_bwd_weights_executable_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::deconvolution_backward_weights::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>(op_attr::strides);
    auto dilates = op->get_attr<dims>(op_attr::dilations);
    auto pads_begin = op->get_attr<dims>(op_attr::pads_begin);
    auto pads_end = op->get_attr<dims>(op_attr::pads_end);
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    set_graph_attributes(prm_attr, graph_attr);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    src = to_format_any(src);
    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    diff_dst = to_format_any(diff_dst);
    auto diff_weight = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    diff_weight = to_format_any(diff_weight);

    auto fwd_hints = dnnl::deconvolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::deconvolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, prm_attr);

    dnnl::deconvolution_backward_weights::primitive_desc pd(p_engine,
            dnnl::algorithm::deconvolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t deconv_bwd_weights_executable_t::get_arg_indices(const op_t *op) {
    return conv_bwd_weights_executable_t::get_arg_indices(op);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
