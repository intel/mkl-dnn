/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#include "graph/backend/dnnl/executables/eltwise.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

eltwise_executable_t::desc_t eltwise_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::eltwise_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float alpha = 0.f, beta = 0.f;
    if (op->has_attr(op_attr::alpha)) {
        alpha = op->get_attr<float>(op_attr::alpha);
    }
    if (op->has_attr(op_attr::beta)) {
        beta = op->get_attr<float>(op_attr::beta);
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

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    if (algo == algorithm::undef) { assert(!"unsupported eltwise op."); }

    dnnl::eltwise_forward::primitive_desc pd;
    pd = dnnl::eltwise_forward::primitive_desc(p_engine, prop_kind::forward,
            algo, src, dst, alpha, beta, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

eltwise_bwd_executable_t::desc_t eltwise_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::eltwise_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const float alpha = op->has_attr(op_attr::alpha)
            ? op->get_attr<float>(op_attr::alpha)
            : 0.f;
    const float beta = op->has_attr(op_attr::beta)
            ? op->get_attr<float>(op_attr::beta)
            : 0.f;
    const auto bwd_algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    const auto fwd_algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::fwd_alg_kind));

    auto forward_data = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    dnnl::eltwise_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, fwd_algo, forward_data, forward_data,
            alpha, beta, prm_attr);

    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    diff_dst = to_format_any(diff_dst);
    diff_src = to_format_any(diff_src);
    dnnl::eltwise_backward::primitive_desc pd(p_engine, bwd_algo, diff_src,
            diff_dst, forward_data, alpha, beta, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

void binary_executable_t::execute(const stream &stream,
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
std::optional<::sycl::event> binary_executable_t::execute_sycl(
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
ocl_event_t binary_executable_t::execute_ocl(const stream &stream,
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

binary_executable_t::desc_t binary_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::binary::primitive_desc>(
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

    auto src0 = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto src1 = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto tmp_dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    // For binary, if we set dst memory tag any, it will deduce strange format
    // for dst when src0 shape is 1x1x1x1, such as cdab. It will cause binary
    // performance poor, and the post matmul pattern performance is poor.
    // So we force dst format to src0 format.
    auto format_tag = md2fmt_tag_str(src0.get());
    const auto &dims = tmp_dst.get_dims();
    const auto &dtype = tmp_dst.get_data_type();
    dnnl_memory_desc_t dst_c;
    dnnl_memory_desc_create_with_string_tag(&dst_c,
            static_cast<int>(dims.size()), dims.data(),
            static_cast<dnnl_data_type_t>(dtype), format_tag.data());
    dnnl::memory::desc dst;
    dst.reset(dst_c);

    const algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));

    dnnl::binary::primitive_desc pd;
    if (algo == algorithm::binary_select) {
        auto src2 = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
        pd = dnnl::binary::primitive_desc(
                p_engine, algo, src0, src1, src2, dst, prm_attr);
    } else {
        pd = dnnl::binary::primitive_desc(
                p_engine, algo, src0, src1, dst, prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_executable_t::desc_t prelu_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::prelu_forward::primitive_desc>(
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

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto wei = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    wei = to_format_any(wei);
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);

    dnnl::prelu_forward::primitive_desc pd(
            p_engine, prop_kind::forward, src, wei, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

prelu_bwd_executable_t::desc_t prelu_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::prelu_backward::primitive_desc>(
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

    auto forward_data = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto wei = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    wei = to_format_any(wei);
    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(2));

    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    auto diff_wei = make_dnnl_memory_desc(op->get_output_logical_tensor(1));
    diff_wei = to_format_any(diff_wei);

    auto hint_fwd_pd = dnnl::prelu_forward::primitive_desc(p_engine,
            prop_kind::forward, forward_data, wei, diff_dst, prm_attr);

    dnnl::prelu_backward::primitive_desc pd(p_engine, forward_data, wei,
            diff_src, diff_wei, diff_dst, hint_fwd_pd, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t binary_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    const dnnl::algorithm algo = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));

    // inputs
    size_t index = 0;
    args.insert({DNNL_ARG_SRC_0, {indices_t::type_t::input, index++}});
    args.insert({DNNL_ARG_SRC_1, {indices_t::type_t::input, index++}});
    if (algo == dnnl::algorithm::binary_select) {
        args.insert({DNNL_ARG_SRC_2, {indices_t::type_t::input, index++}});
    }
    get_arg_indices_for_post_ops(op, args, index);

    // outputs
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

arg_indices_t prelu_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    // inputs
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, 1}});

    // outputs
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

arg_indices_t prelu_bwd_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    // inputs
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, 1}});
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 2}});

    // outputs
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_DIFF_WEIGHTS, {indices_t::type_t::output, 1}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 2}});

    return args;
}

arg_indices_t eltwise_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

arg_indices_t eltwise_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;

    if (op->get_attr<bool>(op_attr::use_dst)) {
        args.insert({DNNL_ARG_DST, {indices_t::type_t::input, 0}});
    } else {
        args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    }
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 1}});

    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
