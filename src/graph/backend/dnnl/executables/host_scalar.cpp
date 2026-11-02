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

#include "graph/backend/dnnl/executables/host_scalar.hpp"

#include "common/dnnl_thread.hpp"
#include "cpu/cpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void host_scalar_executable_t::execute_impl(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    auto it_src = args.find(DNNL_ARG_FROM);
    auto it_dst = args.find(DNNL_ARG_TO);

    if (it_src == args.end() || it_dst == args.end()) {
        assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
        return;
    }

    const memory &src_mem = it_src->second;
    const memory &dst_mem = it_dst->second;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    // Chain through TP so async execution order is preserved.
    // Capture by value since the lambda may outlive this scope.
    stream.get()->before_exec_hook();
    dnnl::impl::parallel(1, [src_mem, dst_mem](int, int) {
        DNNL_HOST_SCALAR_TYPE_SWITCH(
                src_mem.get_desc().get_data_type(), DType, {
                    const DType val = src_mem.get_host_scalar_value<DType>();
                    std::memcpy(dst_mem.get_data_handle(), &val, sizeof(DType));
                });
    });
    stream.get()->after_exec_hook();
#else
    DNNL_HOST_SCALAR_TYPE_SWITCH(src_mem.get_desc().get_data_type(), DType, {
        const DType val = src_mem.get_host_scalar_value<DType>();
        std::memcpy(dst_mem.get_data_handle(), &val, sizeof(DType));
    });
#endif
}

void host_scalar_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        bool block_on_wait = true;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        if (stream.get()->engine()->kind() == engine_kind::cpu) {
            dnnl::threadpool_interop::threadpool_iface *tp = nullptr;
            auto *cpu_stream = dnnl::impl::utils::downcast<
                    dnnl::impl::cpu::cpu_stream_t *>(stream.get());
            auto st = cpu_stream->get_threadpool(&tp);
            block_on_wait = st == dnnl::impl::status::success && tp
                    && !(tp->get_flags()
                            & dnnl::threadpool_interop::threadpool_iface::
                                    ASYNCHRONOUS);
        }
#endif
        if (block_on_wait) stream.get()->wait();
        double start_ms = dnnl::impl::get_msec();
        execute_impl(stream, args);
        if (block_on_wait) stream.get()->wait();
        double duration_ms = dnnl::impl::get_msec() - start_ms;
        VPROF(start_ms, graph, exec, VERBOSE_profile, info_.c_str(),
                duration_ms);
    } else {
        execute_impl(stream, args);
    }
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> host_scalar_executable_t::execute_sycl_impl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    auto it_src = args.find(DNNL_ARG_FROM);
    auto it_dst = args.find(DNNL_ARG_TO);

    if (it_src == args.end() || it_dst == args.end()) {
        // TODO(xxx): this case should not happen. We may want to convert it to
        // a verbose error.
        assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
        return std::nullopt;
    }

    const memory &src_mem = it_src->second;
    const memory &dst_mem = it_dst->second;

    double start_ms = dnnl::impl::get_msec();
    auto *strm = stream.get();
    strm->before_exec_hook();

    // get_data_handle() is blocked for host scalar memories at the C API
    // level, so we access the underlying storage pointer directly.
    void *src_ptr = const_cast<memory_t *>(src_mem.get())
                            ->memory_storage()
                            ->data_handle();
    void *dst_ptr = dst_mem.get_data_handle();
    const size_t size = src_mem.get_desc().get_size();

    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
    auto event = sycl_queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(deps);
        cgh.memcpy(dst_ptr, src_ptr, size);
    });

    // SYCL GPU only.
    if (strm->is_verbose_profiler_enabled()) {
        auto *gpu_strm = dnnl::impl::utils::downcast<gpu::stream_t *>(strm);
        auto verbose_event = std::make_shared<xpu::sycl::event_t>(
                std::vector<::sycl::event> {event});
        gpu_strm->verbose_profiler()->register_event(verbose_event);
        strm->run_verbose_profiler(info_, start_ms,
                static_cast<uint64_t>(dnnl::impl::component_t::graph));
    }
    strm->after_exec_hook();
    return event;
}

std::optional<::sycl::event> host_scalar_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        if (!stream.get()->is_verbose_profiler_enabled()) {
            stream.get()->wait();
            double start_ms = dnnl::impl::get_msec();
            execute_sycl_impl(stream, args, deps);
            stream.get()->wait();
            double duration_ms = dnnl::impl::get_msec() - start_ms;
            VPROF(start_ms, graph, exec, VERBOSE_profile, info_.c_str(),
                    duration_ms);
            return std::nullopt;
        } else {
            return execute_sycl_impl(stream, args, deps);
        }
    } else {
        return execute_sycl_impl(stream, args, deps);
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t host_scalar_executable_t::execute_ocl_impl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    auto it_src = args.find(DNNL_ARG_FROM);
    auto it_dst = args.find(DNNL_ARG_TO);

    if (it_src == args.end() || it_dst == args.end()) {
        assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
        return {};
    }

    const memory &src_mem = it_src->second;
    const memory &dst_mem = it_dst->second;

    double start_ms = dnnl::impl::get_msec();
    auto *strm = stream.get();
    strm->before_exec_hook();

    assert(deps.size() <= 1);
    // Passing the empty event to memcpy below causes failure.
    std::vector<cl_event> raw_deps(deps.begin(), deps.end());
    const bool empty = raw_deps.empty() || raw_deps[0] == nullptr;
    const cl_uint num = empty ? 0 : static_cast<cl_uint>(raw_deps.size());
    const size_t size = src_mem.get_desc().get_size();

    // get_data_handle() is blocked for host scalar memories at the C API
    // level, so we access the underlying storage pointer directly.
    void *src_ptr = const_cast<memory_t *>(src_mem.get())
                            ->memory_storage()
                            ->data_handle();

    cl_event e = nullptr;
    UNUSED_STATUS(xpu::ocl::usm::memcpy(strm, dst_mem.get_data_handle(),
            src_ptr, size, num, empty ? nullptr : raw_deps.data(), &e));
    if (strm->is_verbose_profiler_enabled() && e) {
        auto *gpu_strm = dnnl::impl::utils::downcast<gpu::stream_t *>(strm);
        auto verbose_event = std::make_shared<xpu::ocl::event_t>(
                xpu::ocl::wrapper_t<cl_event>(e, true));
        gpu_strm->verbose_profiler()->register_event(verbose_event);
        strm->run_verbose_profiler(info_, start_ms,
                static_cast<uint64_t>(dnnl::impl::component_t::graph));
    }
    strm->after_exec_hook();
    return ocl_event_t(e);
}

ocl_event_t host_scalar_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        if (!stream.get()->is_verbose_profiler_enabled()) {
            stream.get()->wait();
            double start_ms = dnnl::impl::get_msec();
            execute_ocl_impl(stream, args, deps);
            stream.get()->wait();
            double duration_ms = dnnl::impl::get_msec() - start_ms;
            VPROF(start_ms, graph, exec, VERBOSE_profile, info_.c_str(),
                    duration_ms);
            return {};
        } else {
            return execute_ocl_impl(stream, args, deps);
        }
    } else {
        return execute_ocl_impl(stream, args, deps);
    }
}
#endif

arg_indices_t host_scalar_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t args;

    args.insert({DNNL_ARG_FROM, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_TO, {indices_t::type_t::output, 0}});
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
