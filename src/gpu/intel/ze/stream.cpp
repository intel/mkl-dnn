/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/ze/stream.hpp"
#include "gpu/intel/ze/engine.hpp"

#include "xpu/ze/stream_profiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t stream_t::create_stream(impl::stream_t **stream,
        impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
    std::unique_ptr<intel::ze::stream_t> s(new stream_t(engine, stream_impl));
    if (!s) return status::out_of_memory;

    status_t status = s->init();
    if (status != status::success) {
        // Stream owns stream_impl only if it's created successfully
        // (including initialization).
        s->impl_.release();
        return status;
    }

    *stream = s.release();

    return status::success;
}

status_t stream_t::init() {
    if (is_profiling_enabled() || is_verbose_profiler_enabled()) {
        double timer_frequency = 0.0;
        uint64_t max_timestamp_value = 0;
        CHECK(get_device_properties(
                utils::downcast<engine_t *>(engine())->device(),
                timer_frequency, max_timestamp_value));

        if (is_profiling_enabled()) {
            profiler_ = utils::make_unique<xpu::ze::stream_profiler_t>(
                    this, timer_frequency, max_timestamp_value);
        }

        if (is_verbose_profiler_enabled()) {
            verbose_profiler_.set(
                    utils::make_unique<xpu::ze::verbose_profiler_t>(
                            this, timer_frequency, max_timestamp_value));
        }
    }

    return status::success;
}

void stream_t::before_exec_hook() {
    if (is_profiling_enabled()) profiler_->start_profiling();
    if (is_verbose_profiler_enabled()) {
        if (!verbose_profiler_.is_set()) {
            double timer_frequency = 0.0;
            uint64_t max_timestamp_value = 0;
            auto status = get_device_properties(
                    utils::downcast<engine_t *>(engine())->device(),
                    timer_frequency, max_timestamp_value);
            if (status != status::success) return;
            verbose_profiler_.set(
                    utils::make_unique<xpu::ze::verbose_profiler_t>(
                            this, timer_frequency, max_timestamp_value));
        }
        verbose_profiler_.get()->update_event_list();
    }
}

void stream_t::after_exec_hook() {
    ze_ctx().set_deps(xpu::ze::event_t());
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

status_t stream_t::reset_profiling() {
    if (!is_profiling_enabled()) return status::invalid_arguments;

    profiler_->reset();
    if (!is_verbose_profiler_enabled()) impl()->reset_events();

    return status::success;
}

status_t stream_t::get_profiling_data(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    if (!is_profiling_enabled()) return status::invalid_arguments;

    return profiler_->get_info(data_kind, num_entries, data);
}

status_t stream_t::get_device_properties(ze_device_handle_t dev,
        double &timer_frequency, uint64_t &max_timestamp_value) const {
    ze_device_properties_t device_properties {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    ZE_CHECK(xpu::ze::zeDeviceGetProperties(dev, &device_properties));

    timer_frequency = 1e9 / device_properties.timerResolution;
    max_timestamp_value
            = (1ULL << device_properties.kernelTimestampValidBits) - 1;
    return status::success;
}

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
