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

#include "graph/backend/dnnl/executables/matmul.hpp"

#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

arg_indices_t matmul_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_conv_and_matmul(op);
}

matmul_executable_t::matmul_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout) {
    using ltw = logical_tensor_wrapper_t;
    // if with zero dimension, the matmul op will take no effect, we
    // construct a dummy kernel
    if (ltw(op->get_input_logical_tensor(0)).has_zero_dim()
            || ltw(op->get_input_logical_tensor(1)).has_zero_dim()) {
        is_dummy_ = true;
        return;
    }

    auto desc
            = create_desc(op, p_engine, pd_cache, graph_attr, use_block_layout);
    prim_ = dnnl::matmul(desc);

    // The scratchpad size of pd created by using any format tag may be
    // different from the scratchpad size of pd created by using queried
    // optimal format tag
    dnnl::memory::desc stored
            = make_dnnl_memory_desc(op->get_output_logical_tensor(1));
    dnnl::memory::desc real = desc.scratchpad_desc();
    if (stored != real) {
        auto scratchpad_val = op->get_output_value(1);
        scratchpad_val->set_layout_type(layout_type::any);
        fill_layout_info(scratchpad_val, real);
    }

    if (op->has_attr(op_attr::with_sum))
        with_sum_ = op->get_attr<bool>(op_attr::with_sum);
}

void matmul_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (is_dummy_) {
        dummy_impl_.execute(stream, args);
        return;
    }

    if (with_sum_) {
        auto it_dst = args.find(DNNL_ARG_DST);
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        if (it_dst == args.end() || it_src == args.end()) {
            assert(!("cannot find the required memory"));
            return;
        }

        memory &dst_mem = const_cast<memory &>(it_dst->second);
        memory &psrc_mem = const_cast<memory &>(it_src->second);

        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            dnnl::reorder(psrc_mem, dst_mem).execute(stream, psrc_mem, dst_mem);
        }
    }
    prim_.execute(stream, args);
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> matmul_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    if (is_dummy_) { return dummy_impl_.execute_sycl(stream, args, deps); }

    auto sycl_deps = deps;
    if (with_sum_) {
        auto it_dst = args.find(DNNL_ARG_DST);
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        if (it_dst == args.end() || it_src == args.end()) {
            // TODO(xxx): this case should not happen. We may want to convert it to
            // a verbose error.
            assert(!"cannot find memory for DNNL_ARG_POST_SRC or DNNL_ARG_DST");
            return std::nullopt;
        }

        memory &dst_mem = const_cast<memory &>(it_dst->second);
        memory &psrc_mem = const_cast<memory &>(it_src->second);

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
ocl_event_t matmul_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    if (is_dummy_) { return dummy_impl_.execute_ocl(stream, args, deps); }

    std::vector<cl_event> ocl_deps(deps.begin(), deps.end());
    ocl_event_t reorder_event;
    if (with_sum_) {
        auto it_dst = args.find(DNNL_ARG_DST);
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        if (it_dst == args.end() || it_src == args.end()) {
            assert(!("cannot find the required memory"));
            return {};
        }

        memory &dst_mem = const_cast<memory &>(it_dst->second);
        memory &psrc_mem = const_cast<memory &>(it_src->second);

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

matmul_executable_t::desc_t matmul_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    using ltw = logical_tensor_wrapper_t;

    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::matmul::primitive_desc>(
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
    set_graph_attributes(prm_attr, graph_attr);

    if (op->has_attr(op_attr::accumulation_mode)) {
        const auto acc_mode
                = op->get_attr<std::string>(op_attr::accumulation_mode);
        prm_attr.set_accumulation_mode(str2accumulation_mode(acc_mode));
    }

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    // For non-constant activation and weight, create primitive desc with
    // strided layout
    bool const_activation = ltw(op->get_input_logical_tensor(0)).is_constant()
            && is_constant_cache_enabled(p_engine);
    if (use_block_layout && const_activation) { src = to_format_any(src); }
    auto wei = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    bool const_weight = ltw(op->get_input_logical_tensor(1)).is_constant()
            && is_constant_cache_enabled(p_engine);
    if (use_block_layout && const_weight) { wei = to_format_any(wei); }
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    const bool keep_dst_layout = op->has_attr(op_attr::keep_dst_layout)
            && op->get_attr<bool>(op_attr::keep_dst_layout);
    if (dst.get_format_kind() == dnnl::memory::format_kind::any
            && !keep_dst_layout) {
        // convert to strided for avoiding blocked activation. The format kind
        // of dst is possible to be any when:
        // 1) It is created with internal logical tensor
        // 2) It is the partition output and defined by user
        dst = to_ncx_format(dst);
    }
    dnnl::matmul::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
        bias = to_format_any(bias);
        pd = dnnl::matmul::primitive_desc(
                p_engine, src, wei, bias, dst, prm_attr);
    } else {
        pd = dnnl::matmul::primitive_desc(p_engine, src, wei, dst, prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
