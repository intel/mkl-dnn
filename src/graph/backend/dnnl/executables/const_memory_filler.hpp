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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_CONST_MEMORY_FILLER_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_CONST_MEMORY_FILLER_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <op_attr_t attr_name, typename attr_dt, typename target_dt>
struct const_memory_filler_t : public op_executable_t {
    static arg_indices_t get_arg_indices(const op_t *op) {
        UNUSED(op);
        arg_indices_t args;
        // We only set dst argument, to which constant data will be copied
        args.insert({DNNL_ARG_TO, {indices_t::type_t::output, 0}});
        return args;
    }

    const_memory_filler_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        UNUSED(p_engine);
        UNUSED(pd_cache);
        UNUSED(graph_attr);
        UNUSED(use_block_layout);
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        attr_data_
                = get_attr_data(op->get_attr<std::vector<attr_dt>>(attr_name),
                        std::is_same<attr_dt, target_dt>());
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        auto it = args.find(DNNL_ARG_TO);
        if (it == args.end()) {
            // TODO(xxx): we should propagate the error by returning a status.
            assert(!"cannot find memory for DNNL_ARG_TO");
            return;
        }
        const memory &dst_mem = it->second;

        auto is_cpu = dst_mem.get_engine().get_kind() == engine::kind::cpu;
        // handle cross-engine case
        auto src_eng = (is_cpu) ? dst_mem.get_engine()
                                : engine(dflt_eng_kind, dflt_eng_idx);

        const memory src_mem
                = make_dnnl_memory(dst_mem.get_desc(), src_eng, data_handle);
        dnnl::reorder(src_mem, dst_mem)
                .execute(stream, const_cast<memory &>(src_mem),
                        const_cast<memory &>(dst_mem));
    }

#ifdef DNNL_WITH_SYCL
    std::optional<::sycl::event> execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        auto e = sycl_queue.memcpy(dst_mem.get_data_handle(), data_handle,
                dst_mem.get_desc().get_size());
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    ocl_event_t execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;
        assert(deps.size() <= 1);
        // Passing the empty event to memcpy below causes failure.
        std::vector<cl_event> raw_deps(deps.begin(), deps.end());
        const bool empty = raw_deps.empty() || raw_deps[0] == nullptr;
        const cl_uint num = empty ? 0 : static_cast<cl_uint>(raw_deps.size());
        cl_event e;
        UNUSED_STATUS(
                xpu::ocl::usm::memcpy(stream.get(), dst_mem.get_data_handle(),
                        data_handle, dst_mem.get_desc().get_size(), num,
                        empty ? nullptr : raw_deps.data(), &e));
        return ocl_event_t(e);
    }
#endif

    bool is_initialized() const override { return true; }

private:
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::true_type) {
        return orig_data;
    }
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::false_type) {
        std::vector<target_dt> result;
        result.reserve(orig_data.size());
        for (const auto &v : orig_data)
            result.push_back(static_cast<target_dt>(v));
        return result;
    }

    const engine::kind dflt_eng_kind = engine::kind::cpu;
    const size_t dflt_eng_idx = 0;
    std::vector<target_dt> attr_data_;
};

using const_scales_filler
        = const_memory_filler_t<op_attr::scales, float, float>;
using const_zps_filler = const_memory_filler_t<op_attr::zps, int64_t, int32_t>;

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_CONST_MEMORY_FILLER_HPP
