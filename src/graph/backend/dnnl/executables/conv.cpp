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

#include "graph/backend/dnnl/executables/conv.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// conv_fwd_executable_t implementations
void conv_fwd_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            // psrc_mem and dst_mem may have different data type bug same
            // buffer size(u8 and s8) for such case, need to reorder
            // psrc_mem to dst_mem with original data type
            if (psrc_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::s8
                    && dst_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::u8) {
                dnnl::memory::desc to_desc = dst_mem.get_desc();
                auto format_tag = md2fmt_tag_str(to_desc.get());
                const auto &dims = to_desc.get_dims();
                const auto &dtype = psrc_mem.get_desc().get_data_type();
                dnnl_memory_desc_t new_to_desc_c;
                dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                        static_cast<int>(dims.size()), dims.data(),
                        static_cast<dnnl_data_type_t>(dtype),
                        format_tag.data());
                dnnl::memory::desc new_to_desc;
                new_to_desc.reset(new_to_desc_c);
                const memory to_mem
                        = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                to_mem.set_data_handle(dst_mem.get_data_handle());
                dnnl::reorder(psrc_mem, to_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(to_mem));
            } else {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }
    }

    prim_.execute(stream, args);
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> conv_fwd_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    auto sycl_deps = deps;
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            // psrc_mem and dst_mem may have different data type bug same
            // buffer size(u8 and s8) for such case, need to reorder
            // psrc_mem to dst_mem with original data type
            if (psrc_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::s8
                    && dst_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::u8) {
                dnnl::memory::desc to_desc = dst_mem.get_desc();
                auto format_tag = md2fmt_tag_str(to_desc.get());
                const auto &dims = to_desc.get_dims();
                const auto &dtype = psrc_mem.get_desc().get_data_type();
                dnnl_memory_desc_t new_to_desc_c;
                dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                        static_cast<int>(dims.size()), dims.data(),
                        static_cast<dnnl_data_type_t>(dtype),
                        format_tag.data());
                dnnl::memory::desc new_to_desc;
                new_to_desc.reset(new_to_desc_c);
                const memory to_mem
                        = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                to_mem.set_data_handle(dst_mem.get_data_handle());
                auto prim = dnnl::reorder(psrc_mem, to_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(to_mem)}},
                        sycl_deps);
                sycl_deps = {e};
                if (stream.get_engine().get_kind() == engine::kind::cpu)
                    e.wait();
            } else {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
    }
    auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
    if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
    return e;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t conv_fwd_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<cl_event> ocl_deps(deps.begin(), deps.end());
    ocl_event_t reorder_event;
    if (with_sum_) {
        const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
        const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            // psrc_mem and dst_mem may have different data type bug same
            // buffer size(u8 and s8) for such case, need to reorder
            // psrc_mem to dst_mem with original data type
            if (psrc_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::s8
                    && dst_mem.get_desc().get_data_type()
                            == dnnl::memory::data_type::u8) {
                dnnl::memory::desc to_desc = dst_mem.get_desc();
                auto format_tag = md2fmt_tag_str(to_desc.get());
                const auto &dims = to_desc.get_dims();
                const auto &dtype = psrc_mem.get_desc().get_data_type();
                dnnl_memory_desc_t new_to_desc_c;
                dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                        static_cast<int>(dims.size()), dims.data(),
                        static_cast<dnnl_data_type_t>(dtype),
                        format_tag.data());
                dnnl::memory::desc new_to_desc;
                new_to_desc.reset(new_to_desc_c);

                const memory to_mem
                        = dnnl::ocl_interop::get_memory_kind(dst_mem)
                                == dnnl::ocl_interop::memory_kind::usm
                        ? dnnl::ocl_interop::make_memory(new_to_desc,
                                  psrc_mem.get_engine(),
                                  dnnl::ocl_interop::memory_kind::usm,
                                  dst_mem.get_data_handle())
                        : dnnl::ocl_interop::make_memory(new_to_desc,
                                  psrc_mem.get_engine(),
                                  reinterpret_cast<cl_mem>(
                                          dst_mem.get_data_handle()));

                auto prim = dnnl::reorder(psrc_mem, to_mem);
                reorder_event = ocl_event_t(dnnl::ocl_interop::execute(prim,
                        stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(to_mem)}},
                        ocl_deps));
                ocl_deps = {reorder_event.get()};
            } else {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                reorder_event = ocl_event_t(dnnl::ocl_interop::execute(prim,
                        stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps));
                ocl_deps = {reorder_event.get()};
            }
        }
    }
    auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
    return ocl_event_t(e);
}
#endif

conv_fwd_executable_t::desc_t conv_fwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_forward::primitive_desc>(
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
    fusion_info_t fusion_info;
    if (op->has_attr(op_attr::fusion_info)) {
        fusion_info = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    set_graph_attributes(prm_attr, graph_attr);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));

    // assume constant weight is for inference scenario
    const auto &wei_lt = op->get_input_logical_tensor(1);
    auto pkind = (logical_tensor_wrapper_t(wei_lt).property_type()
                         == property_type::constant)
            ? prop_kind::forward_inference
            : prop_kind::forward_training;
    auto weight = make_dnnl_memory_desc(wei_lt);
    weight = to_format_any(weight);

    auto base_conv_dst_lt = op->get_output_logical_tensor(0);
    if (fusion_info.has_post_dw_conv()) {
        // when fused post depthwise conv, onednn required to use the base conv
        // dst md to create the conv primitive. in the subgraph, the base conv
        // dst is a intermediate output which has been fused away, so here we
        // get it from fusion info
        const auto &dw_conv = fusion_info.get_post_dw_conv();
        base_conv_dst_lt = dw_conv->get_op()->get_input_logical_tensor(0);
    }
    auto dst = make_dnnl_memory_desc(base_conv_dst_lt);
    auto create_pd = [&](const dnnl::memory::desc &src_md,
                             const dnnl::memory::desc &dst_md) {
        if (op->has_attr(op_attr::with_bias)
                && op->get_attr<bool>(op_attr::with_bias)) {
            auto bias = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
            bias = to_format_any(bias);
            return dnnl::convolution_forward::primitive_desc(p_engine, pkind,
                    algorithm::convolution_direct, src_md, weight, bias, dst_md,
                    strides, dilates, pads_begin, pads_end, prm_attr);
        } else {
            return dnnl::convolution_forward::primitive_desc(p_engine, pkind,
                    algorithm::convolution_direct, src_md, weight, dst_md,
                    strides, dilates, pads_begin, pads_end, prm_attr);
        }
    };

    if (!use_block_layout) {
        src = to_nxc_format(src);
        dst = to_nxc_format(dst);
    } else {
        // If the dst has been explicitly set to nxc layout or the data_format
        // has been defined as NXC by users, we prefer to directly use optimal
        // blocked src and plain dst to create conv pd. In the following, we
        // will first query out the optimal src.
        bool permute_nxc_dst = false;
        if (op->get_output_value(0)->get_consumers().size() == 1) {
            const auto &next_op
                    = op->get_output_value(0)->get_consumers()[0].get_op();
            if (next_op.get_kind() == op_kind::_permute) {
                auto permute_dst_lt = next_op.get_output_logical_tensor(0);
                auto perm = get_permutation(permute_dst_lt.ndims, "NCX", "NXC");
                if (next_op.get_attr<std::vector<int64_t>>(op_attr::permutation)
                        == perm) {
                    auto inverse_perm = get_permutation(
                            permute_dst_lt.ndims, "NXC", "NCX");
                    auto perm_dst = make_dnnl_memory_desc(permute_dst_lt);
                    dst = perm_dst.permute_axes(
                            dnnl_impl::utils::cast_to_int32(inverse_perm));
                    permute_nxc_dst = true;
                }
            }
        }
        if (!is_format(dst, "nxc") && !permute_nxc_dst) {
            src = to_format_any(src);
            dst = to_format_any(dst);
        } else {
            auto tmp_src = to_format_any(src);
            auto tmp_dst = to_format_any(dst);
            dnnl::convolution_forward::primitive_desc tmp_pd
                    = create_pd(tmp_src, tmp_dst);
            src = tmp_pd.src_desc();
        }
    }

    dnnl::convolution_forward::primitive_desc pd = create_pd(src, dst);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t conv_fwd_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_conv_and_matmul(op);
}

// conv_bwd_data_executable_t implementations
conv_bwd_data_executable_t::desc_t conv_bwd_data_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_backward_data::primitive_desc>(
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
    if (!use_block_layout)
        diff_dst = to_nxc_format(diff_dst);
    else
        diff_dst = to_format_any(diff_dst);
    auto weight = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    weight = to_format_any(weight);
    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    if (!use_block_layout)
        diff_src = to_nxc_format(diff_src);
    else
        diff_src = to_format_any(diff_src);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
            strides, dilates, pads_begin, pads_end, prm_attr);

    dnnl::convolution_backward_data::primitive_desc pd(p_engine,
            dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t conv_bwd_data_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    // inputs
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, 1}});

    // outputs
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

// conv_bwd_weights_executable_t implementations
conv_bwd_weights_executable_t::desc_t
conv_bwd_weights_executable_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::convolution_backward_weights::primitive_desc>(
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
    if (!use_block_layout)
        src = to_nxc_format(src);
    else
        src = to_format_any(src);
    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    if (!use_block_layout)
        diff_dst = to_nxc_format(diff_dst);
    else
        diff_dst = to_format_any(diff_dst);
    auto diff_weight = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    diff_weight = to_format_any(diff_weight);

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(p_engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, prm_attr);

    dnnl::convolution_backward_weights::primitive_desc pd(p_engine,
            dnnl::algorithm::convolution_direct, src, diff_weight, diff_dst,
            strides, dilates, pads_begin, pads_end, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t conv_bwd_weights_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    // inputs
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 1}});

    // outputs
    args.insert({DNNL_ARG_DIFF_WEIGHTS, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
