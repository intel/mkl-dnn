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

#include "graph/backend/dnnl/executables/resampling.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void resampling_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (with_sum_) {
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        auto it_dst = args.find(DNNL_ARG_DST);
        if (it_src == args.end() || it_dst == args.end()) {
            assert(!"cannot find src or dst memory");
            return;
        }

        const memory &psrc_mem = it_src->second;
        const memory &dst_mem = it_dst->second;
        if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
            dnnl::reorder(psrc_mem, dst_mem)
                    .execute(stream, const_cast<memory &>(psrc_mem),
                            const_cast<memory &>(dst_mem));
        }
    }
    prim_.execute(stream, args);
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> resampling_executable_t::execute_sycl(
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
ocl_event_t resampling_executable_t::execute_ocl(const stream &stream,
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

resampling_executable_t::desc_t resampling_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_forward::primitive_desc>(
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
    // resampling src doesn't support any
    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);

    std::string mode = op->get_attr<std::string>(op_attr::mode);
    algorithm algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        assert(!"unsupported resampling mode.");
    }

    dnnl::resampling_forward::primitive_desc pd;
    pd = dnnl::resampling_forward::primitive_desc(
            p_engine, prop_kind::forward_inference, algo, src, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_bwd_executable_t::desc_t resampling_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_backward::primitive_desc>(
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

    auto mode = op->get_attr<std::string>(op_attr::mode);
    auto algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        assert(!"unsupported resampling mode.");
    }

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    dnnl::resampling_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, algo, src, to_format_any(diff_dst),
            prm_attr);

    auto diff_src = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    diff_src = to_format_any(diff_src);
    dnnl::resampling_backward::primitive_desc pd(
            p_engine, algo, diff_src, diff_dst, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t resampling_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

arg_indices_t resampling_bwd_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 1}});
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
