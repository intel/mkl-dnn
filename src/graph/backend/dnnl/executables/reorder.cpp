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

#include "graph/backend/dnnl/executables/reorder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void reorder_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (with_sum_) {
        auto it_dst = args.find(DNNL_ARG_DST);
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        if (it_dst == args.end() || it_src == args.end()) {
            assert(!("cannot find the required memory"));
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
std::optional<::sycl::event> reorder_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
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

        const memory &psrc_mem = it_src->second;
        const memory &dst_mem = it_dst->second;
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
ocl_event_t reorder_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<cl_event> ocl_deps(deps.begin(), deps.end());
    ocl_event_t reorder_event;
    if (with_sum_) {
        auto it_dst = args.find(DNNL_ARG_DST);
        auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
        if (it_dst == args.end() || it_src == args.end()) {
            assert(!("cannot find the required memory"));
            return {};
        }

        const memory &psrc_mem = it_src->second;
        const memory &dst_mem = it_dst->second;
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

reorder_executable_t::desc_t reorder_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::reorder::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }

    if (op->has_attr(op_attr::mask)) {
        const int mask = get_quant_mask(op.get());
        const bool is_per_group = utils::has_group_shape(op.get());
        std::vector<int64_t> groups;
        if (is_per_group) {
            const auto &group_shape
                    = op->get_attr<std::vector<int64_t>>(op_attr::group_shape);
            const auto ndims = group_shape.size();
            groups = {group_shape[ndims - 2], group_shape[ndims - 1]};
        }

        bool with_runtime_scales = false;
        if (op->has_attr(op_attr::with_runtime_scales)
                && op->get_attr<bool>(op_attr::with_runtime_scales)) {
            const auto scale_dt = static_cast<dnnl::memory::data_type>(
                    op->get_input_logical_tensor(1).data_type);
            prm_attr.set_scales(DNNL_ARG_FROM, mask, groups, scale_dt);
            with_runtime_scales = true;
        } else if (op->has_attr(op_attr::scales)) {
            assertm(false, "only support runtime arg scales.\n");
        }

        if (op->has_attr(op_attr::with_runtime_src_zps)
                && op->get_attr<bool>(op_attr::with_runtime_src_zps)) {
            const auto zps_dt = static_cast<dnnl::memory::data_type>(
                    op->get_input_logical_tensor(with_runtime_scales ? 2 : 1)
                            .data_type);
            prm_attr.set_zero_points(DNNL_ARG_FROM, mask, groups, zps_dt);
        } else if (op->has_attr(op_attr::src_zps)) {
            assertm(false, "only support runtime src zero points.\n");
        }

        if (op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            prm_attr.set_zero_points_mask(DNNL_ARG_TO, mask);
        } else if (op->has_attr(op_attr::dst_zps)) {
            assertm(false, "only support runtime dst zero points.\n");
        }
    }

    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto in_md = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto out_md = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    auto pd = dnnl::reorder::primitive_desc(
            p_engine, in_md, p_engine, out_md, prm_attr);
    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t reorder_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;

    size_t idx = 0;
    args.insert({DNNL_ARG_FROM, {indices_t::type_t::input, idx++}});

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    if ((op->has_attr(op_attr::with_runtime_scales)
                && op->get_attr<bool>(op_attr::with_runtime_scales))
            || fusion_info.with_runtime_scales(true, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                {indices_t::type_t::input, idx++}});
    }

    if ((op->has_attr(op_attr::with_runtime_src_zps)
                && op->get_attr<bool>(op_attr::with_runtime_src_zps))
            || fusion_info.with_runtime_zero_points(true, 0)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                {indices_t::type_t::input, idx++}});
    }

    get_arg_indices_for_post_ops(op, args, idx);

    if (fusion_info.with_runtime_scales(false, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, idx++}});
    }

    if ((op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps))
            || fusion_info.with_runtime_zero_points(false, 0)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                {indices_t::type_t::input, idx++}});
    }

    args.insert({DNNL_ARG_TO, {indices_t::type_t::output, 0}});
    if (op->num_outputs() > 1) {
        args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    }

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
