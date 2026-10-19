/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <cstring>

#include <CL/cl.h>

#include "common/verbose.hpp"

#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/memory_storage.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/intel/ocl/engine.hpp"
#include "gpu/intel/ocl/stream.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t stream_t::init() {
    if (is_profiling_enabled()) {
        profiler_ = utils::make_unique<xpu::ocl::stream_profiler_t>(this);
        mdapi_helper_ = utils::make_unique<mdapi_helper_t>();
    }

    // Enables profiling capabilities to allow the verbose mode to print
    // profiling info using device measured times.
    // The verbose profiler state is fixed at stream initialization and does
    // not respond to runtime changes made via set_dnnl_verbose().
    // TODO: allow runtime control of the asynchronous verbose mode via
    // set_dnnl_verbose()
    CHECK(impl()->init_verbose_profiler(engine()->kind()));

    // Restore queue on successful exit, otherwise queue may be released
    // without retain
    cl_command_queue queue = impl()->queue();
    CHECK(impl()->set_queue(nullptr));

    assert(engine()->kind() == engine_kind::gpu);

    const auto *ocl_engine_impl
            = utils::downcast<const xpu::ocl::engine_impl_t *>(
                    engine()->impl());

    // Create queue if it is not set
    if (!queue) {
        cl_int err;
        queue = create_queue(
                ocl_engine_impl->context(), ocl_engine_impl->device(), &err);
        OCL_CHECK(err);
    } else {
        // Check that queue is compatible with the engine
        cl_context ocl_ctx;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                sizeof(cl_context), &ocl_ctx, nullptr));

        cl_device_id ocl_dev;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                sizeof(cl_device_id), &ocl_dev, nullptr));

        if (ocl_engine_impl->device() != ocl_dev
                || ocl_engine_impl->context() != ocl_ctx)
            return status::invalid_arguments;

        OCL_CHECK(xpu::ocl::clRetainCommandQueue(queue));
    }
    CHECK(impl()->set_queue(queue));
    if (is_verbose_profiler_enabled()) {
        verbose_profiler_.set(
                utils::make_unique<xpu::ocl::verbose_profiler_t>(this));
        // Check if the queue has profiling enabled and pause the verbose
        // profiler if it does not. Verbose lines are still emitted during
        // logging, but without execution timing information.
        if (!impl()->queue_has_profiling()) {
            VWARN(common, ocl,
                    "OpenCL queue does not have profiling enabled. "
                    "Verbose profiling is paused and execution times "
                    "will not be reported.");
            verbose_profiler()->pause_profiling();
        }
    }

    if (is_profiling_enabled() || is_verbose_profiler_enabled()) {
        cl_command_queue_properties props;
        OCL_CHECK(xpu::ocl::clGetCommandQueueInfo(impl()->queue(),
                CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr));
        bool is_out_of_order
                = (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
        if (is_out_of_order) {
            VERROR(common, ocl,
                    "OpenCL kernel profiling is not "
                    "supported with out-of-order queues");
            return status::invalid_arguments;
        }
    }

    return status::success;
}

cl_command_queue stream_t::create_queue(
        cl_context ctx, cl_device_id dev, cl_int *err) const {
    if (is_profiling_enabled() && mdapi_helper_) {
        auto ret = mdapi_helper_->create_queue(ctx, dev, err);
        if (ret) return ret;
    }

    const bool is_out_of_order = (flags() & stream_flags::out_of_order);

    cl_command_queue_properties queue_props {};
    if (is_profiling_enabled() || is_verbose_profiler_enabled())
        queue_props |= CL_QUEUE_PROFILING_ENABLE;
    if (is_out_of_order) queue_props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, queue_props, 0};
    return xpu::ocl::clCreateCommandQueueWithProperties(ctx, dev, props, err);
#else
    return xpu::ocl::clCreateCommandQueue(ctx, dev, queue_props, err);
#endif
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
    if (is_verbose_profiler_enabled()) {
        auto &verbose_profiler = verbose_profiler_.get_or_set(
                utils::make_unique<xpu::ocl::verbose_profiler_t>(this));
        if (!impl()->queue_has_profiling()) verbose_profiler->pause_profiling();
        verbose_profiler->update_event_list();
    }
}

void stream_t::after_exec_hook() {
    ocl_ctx().set_deps(xpu::ocl::event_t());
    if (is_profiling_enabled()) profiler_->stop_profiling();
    if (auto *vp = verbose_profiler()) { vp->check_for_completed_primitives(); }
}

status_t stream_t::run_verbose_profiler(
        const std::string &pd_info, double start_ms, uint64_t component) {
    if (!is_verbose_profiler_enabled()) return status::success;

    auto *vp = verbose_profiler();
    vp->add_to_pending_primitive_list(start_ms, pd_info, component);
    return status::success;
}

status_t stream_t::copy(const memory_storage_t &src,
        const memory_storage_t &dst, size_t size, const xpu::event_t &deps,
        xpu::event_t &out_dep) {
    return impl()->copy(this, src, dst, size, deps, out_dep, profiler_.get(),
            verbose_profiler());
}

status_t stream_t::fill(const memory_storage_t &dst, uint8_t pattern,
        size_t size, const xpu::event_t &deps, xpu::event_t &out_dep) {
    return impl()->fill(this, dst, pattern, size, deps, out_dep,
            profiler_.get(), verbose_profiler());
}

status_t stream_t::barrier() {
    return impl()->barrier();
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
