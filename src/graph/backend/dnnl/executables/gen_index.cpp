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

#include "graph/backend/dnnl/executables/gen_index.hpp"

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

genindex_executable_t::genindex_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout)
    : axis_(op->get_attr<int64_t>(op_attr::axis))
    , nelems_(0)
    , ndims_(0)
    , ekind_(p_engine.get_kind()) {
    using ltw = logical_tensor_wrapper_t;
    const auto &input_lt = op->get_input_logical_tensor(0);
    nelems_ = ltw(input_lt).nelems();
    ndims_ = ltw(input_lt).ndims();
    const auto &output_lt = op->get_output_logical_tensor(0);
    for (int i = 0; i < ndims_; i++) {
        output_dims_[i] = output_lt.dims[i];
        output_strides_[i] = output_lt.layout.strides[i];
    }
    info_ = std::string(dnnl_engine_kind2str(
                    static_cast<dnnl_engine_kind_t>(ekind_)))
            + "," + op->str();
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    if (ekind_ == engine::kind::gpu) {
        compute::kernel_ctx_t kernel_ctx;
        kernel_ctx.define_int("NDIMS", ndims_);
        if (nelems_ > INT32_MAX) kernel_ctx.use_int32_offset(false);
        for (int d = 0; d < MAX_NDIMS; ++d) {
            dim_t dim = (d < ndims_) ? output_dims_[d] : 1;
            dim_t stride = (d < ndims_) ? output_strides_[d] : 0;
            kernel_ctx.define_int(dnnl::impl::utils::format("D%d", d), dim);
            kernel_ctx.define_int(dnnl::impl::utils::format("S%d", d), stride);
        }
        auto *intel_engine
                = dnnl::impl::utils::downcast<gpu::intel::engine_t *>(
                        p_engine.get());
        std::vector<compute::kernel_t> kernels(1);
        intel_engine->create_kernels(&kernels, {"gen_index"}, kernel_ctx);
        kernel_ = kernels[0];
    }
#endif
}

void genindex_executable_t::execute_impl(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    const auto &it_dst = args.find(DNNL_ARG_DST);
    if (it_dst == args.end()) return;

    auto &output = it_dst->second;
    auto output_ptr = static_cast<int32_t *>(output.get_data_handle());

    stream.get()->before_exec_hook();
    dnnl::impl::parallel_nd(nelems_, [= COMPAT_THIS_CAPTURE](dim_t i) {
        dims_t input_dims; // decomposition for physical offsets
        dnnl::impl::utils::l_dims_by_l_offset(
                input_dims, i, output_dims_, ndims_);
        auto offset
                = utils::offset_compute(output_strides_, input_dims, ndims_);
        output_ptr[offset] = static_cast<int32_t>(input_dims[axis_]);
    });
    stream.get()->after_exec_hook();
}

void genindex_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        bool block_on_wait = true;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        dnnl::threadpool_interop::threadpool_iface *tp = nullptr;
        auto st = stream.get()->get_threadpool(&tp);
        block_on_wait = st == dnnl::impl::status::success && tp
                && !(tp->get_flags()
                        & dnnl::threadpool_interop::threadpool_iface::
                                ASYNCHRONOUS);
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
std::optional<::sycl::event> genindex_executable_t::execute_sycl_impl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    if (stream.get_engine().get_kind() == engine::kind::cpu) {
        auto strm_t = stream.get();
        auto *sycl_stream_impl = dnnl::impl::utils::downcast<
                dnnl::impl::xpu::sycl::stream_impl_t *>(strm_t->impl());

        strm_t->before_exec_hook();
        if (!deps.empty()) { sycl_stream_impl->sycl_ctx().set_deps(deps); }

        execute(stream, args);

        // return output event
        ::sycl::event return_event = sycl_stream_impl->get_output_event();
        strm_t->after_exec_hook();
        return return_event;
    }
#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && (DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
    double start_ms = dnnl::impl::get_msec();
    auto compute_stream
            = dnnl::impl::utils::downcast<gpu::intel::stream_t *>(stream.get());
    compute::range_t gws = {static_cast<size_t>(nelems_)};
    auto nd_range = compute::nd_range_t(gws);
    compute::kernel_arg_list_t arg_list;
    const auto &dst = *(args.at(DNNL_ARG_DST).get()->memory_storage());
    arg_list.set(0, dst);
    arg_list.set(1, static_cast<int>(axis_));
    auto *sycl_stream
            = dnnl::impl::utils::downcast<sycl::stream_t *>(compute_stream);
    sycl_stream->before_exec_hook();
    if (!deps.empty()) sycl_stream->sycl_ctx().set_deps(deps);

    kernel_.parallel_for(*compute_stream, nd_range, arg_list,
            sycl_stream->sycl_ctx().get_deps(),
            sycl_stream->sycl_ctx().get_deps());
    auto return_event = sycl_stream->get_output_event();
    if (sycl_stream->is_verbose_profiler_enabled())
        sycl_stream->run_verbose_profiler(info_, start_ms,
                static_cast<uint64_t>(dnnl::impl::component_t::graph));
    sycl_stream->after_exec_hook();
    return return_event;
#else
    assertm(false,
            "genindex opexcutable is only implemented for intel vendor "
            "under SYCL runtime ");
    throw std::runtime_error("Unimplement");
#endif
}

std::optional<::sycl::event> genindex_executable_t::execute_sycl(
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
        } else {
            execute_sycl_impl(stream, args, deps);
        }

        return std::nullopt; // no event returned in profiling mode
    } else {
        return execute_sycl_impl(stream, args, deps);
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t genindex_executable_t::execute_ocl_impl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    double start_ms = dnnl::impl::get_msec();
    auto compute_stream
            = dnnl::impl::utils::downcast<gpu::intel::stream_t *>(stream.get());

    compute::range_t gws = {static_cast<size_t>(nelems_)};

    auto nd_range = compute::nd_range_t(gws);
    compute::kernel_arg_list_t arg_list;
    const auto &dst = *(args.at(DNNL_ARG_DST).get()->memory_storage());
    arg_list.set(0, dst);
    arg_list.set(1, static_cast<int>(axis_));
    auto *ocl_stream = dnnl::impl::utils::downcast<gpu::intel::ocl::stream_t *>(
            compute_stream);

    ocl_stream->before_exec_hook();

    if (!deps.empty()) {
        ocl_stream->ocl_ctx().set_deps(
                std::vector<ocl_event_t>(deps.begin(), deps.end()));
    }

    kernel_.parallel_for(*compute_stream, nd_range, arg_list,
            compute_stream->ctx().get_deps(), compute_stream->ctx().get_deps());

    cl_event return_event = nullptr;
    if ((ocl_stream->flags() & stream_flags::in_order) == 0) {
        auto last = ocl_stream->get_output_event();
        return_event = last.release();
    }
    if (ocl_stream->is_verbose_profiler_enabled())
        ocl_stream->run_verbose_profiler(info_, start_ms,
                static_cast<uint64_t>(dnnl::impl::component_t::graph));
    ocl_stream->after_exec_hook();
    return ocl_event_t(return_event);
#else
    assertm(false,
            "genindex opexcutable is only implemented for intel vendor "
            "under OCL runtime ");
    throw std::runtime_error("Unimplement");
#endif
}

ocl_event_t genindex_executable_t::execute_ocl(const stream &stream,
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
        } else {
            execute_ocl_impl(stream, args, deps);
        }
        return {};
    } else {
        return execute_ocl_impl(stream, args, deps);
    }
}
#endif

arg_indices_t genindex_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);

    arg_indices_t args;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
