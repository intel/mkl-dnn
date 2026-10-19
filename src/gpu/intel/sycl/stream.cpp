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

#include <map>
#include <memory>
#include <CL/cl.h>

#include "common/verbose.hpp"

#include "xpu/sycl/stream_profiler.hpp"

#include "gpu/intel/sycl/stream.hpp"

#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    if (is_profiling_enabled())
        profiler_ = utils::make_unique<xpu::sycl::stream_profiler_t>(this);

    // Enables profiling capabilities to allow the verbose mode to print
    // profiling info using device measured times.
    // The verbose profiler state is fixed at stream initialization and does
    // not respond to runtime changes made via set_dnnl_verbose().
    // TODO: allow runtime control of the asynchronous verbose mode via
    // set_dnnl_verbose()
    CHECK(impl()->init_verbose_profiler(engine()->kind()));

    const auto &sycl_engine_impl
            = *utils::downcast<const xpu::sycl::engine_impl_t *>(
                    engine()->impl());
    auto &sycl_ctx = sycl_engine_impl.context();
    auto &sycl_dev = sycl_engine_impl.device();

    // If queue_ is not set then construct it
    if (!impl()->queue()) {
        ::sycl::property_list props;
        if ((is_profiling_enabled() || is_verbose_profiler_enabled())
                && sycl_dev.is_gpu()) {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::property_list {::sycl::property::queue::
                                                     in_order {},
                              ::sycl::property::queue::enable_profiling {}}
                    : ::sycl::property_list {
                              ::sycl::property::queue::enable_profiling {}};
        } else {
            props = (flags() & stream_flags::in_order)
                    ? ::sycl::property_list {::sycl::property::queue::
                                      in_order {}}
                    : ::sycl::property_list {};
        }
        impl()->set_queue(::sycl::queue(sycl_ctx, sycl_dev, props));

        // Re-initializes verbose profiler after setting the queue to check for
        // supported backends
        CHECK(impl()->init_verbose_profiler(engine()->kind()));

    } else {
        // TODO: Compare device and context of the engine with those of the
        // queue after SYCL adds support for device/context comparison.
        //
        // For now perform some simple checks.
        auto sycl_dev = queue().get_device();
        bool args_ok = true
                && IMPLICATION(
                        engine()->kind() == engine_kind::gpu, sycl_dev.is_gpu())
                && IMPLICATION(engine()->kind() == engine_kind::cpu,
                        (sycl_dev.is_cpu() || xpu::sycl::is_host(sycl_dev)));
        if (!args_ok) return status::invalid_arguments;
    }
    if (is_verbose_profiler_enabled()) {
        verbose_profiler_.set(
                utils::make_unique<xpu::sycl::verbose_profiler_t>(this));

        auto *vp = utils::downcast<xpu::sycl::verbose_profiler_t *>(
                verbose_profiler());

        const bool queue_has_profiling = queue().has_property<
                ::sycl::property::queue::enable_profiling>();
#ifdef DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG
        const bool use_tag = queue().get_device().has(
                ::sycl::aspect::ext_oneapi_queue_profiling_tag);
#else
        const bool use_tag = false;
#endif
        vp->set_use_ext_oneapi_tag(use_tag);
        // Verbose profiling relies on SYCL profiling tags to query profiling
        // info from queued events for exec time computation.
        // If the sycl_ext_oneapi_profiling_tag is not supported, the profiler
        // falls back to using a profiling-enabled queue for the same purpose.
        // If neither are available, the profiling is paused - verbose lines
        // are still emitted during logging, but without execution timing
        // information.
        if (!queue_has_profiling && !use_tag) {
            VWARN(primitive, exec,
                    "SYCL queue does not have profiling enabled and "
                    "sycl_ext_oneapi_profiling_tag is not supported. "
                    "Verbose profiling is paused and execution times "
                    "will not be reported.");
            verbose_profiler()->pause_profiling();
        }
    }
    if (is_profiling_enabled() && sycl_dev.is_gpu() && !queue().is_in_order()) {
        VERROR(common, dpcpp,
                "DPC++ kernel profiling is not supported with out-of-order "
                "queues");
        return status::invalid_arguments;
    }

    return status::success;
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
    if (is_verbose_profiler_enabled()) {
        auto *profiler = static_cast<xpu::sycl::verbose_profiler_t *>(
                verbose_profiler_
                        .get_or_set(utils::make_unique<
                                xpu::sycl::verbose_profiler_t>(this))
                        .get());
        const bool queue_has_profiling = queue().has_property<
                ::sycl::property::queue::enable_profiling>();
#ifdef DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG
        const bool use_tag = queue().get_device().has(
                ::sycl::aspect::ext_oneapi_queue_profiling_tag);
#else
        const bool use_tag = false;
#endif
        profiler->set_use_ext_oneapi_tag(use_tag);
        const bool use_profiler = (queue_has_profiling || use_tag);
        if (!use_profiler) { profiler->pause_profiling(); }

        // Device event profiling and SYCL graph recording are incompatible
        // because the graph execution creates a different execution context
        // with new events that do not inherit the original queue's profiling
        // properties. This causes profiling info queries on graph events to
        // throw exceptions.
        // Switching back to host-side verbose logging is also not viable
        // as the SYCL graph recording breaks on stream.wait() synchronization
        // calls.
        // The current approach is to skip profiling for the primitive whenever
        // graph is recording and resume thereafter to avoid runtime exceptions.
        // In this scenario, the profiler will report zero execution time for
        // the logged primitives.
        if (!recording() && use_profiler) {
            profiler->start_profiling();
        } else {
            if (profiler->is_active() && use_profiler) {
                VWARN(primitive, exec,
                        "SYCL graph recording active - verbose profiling will "
                        "show zero "
                        "execution times until recording completes");
            }
            profiler->pause_profiling();
        }
        profiler->update_event_list();
    }
}

void stream_t::after_exec_hook() {
    sycl_ctx().set_deps(xpu::sycl::event_t());
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

namespace syclex = ::sycl::ext::oneapi::experimental;

bool stream_t::recording() const {
    return impl()->queue()->ext_oneapi_get_state()
            == syclex::queue_state::recording;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
