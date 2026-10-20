/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/intel/ocl/mdapi_utils.hpp"

#include "oneapi/dnnl/dnnl_config.h"

#if defined(__linux__) && (DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL)
#define DNNL_GPU_ENABLE_MDAPI
#endif

#ifdef DNNL_GPU_ENABLE_MDAPI
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "gpu/intel/ocl/utils.hpp"
#include "mdapi/metrics_discovery_api.h"

#ifndef CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL
#define CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL 0x407F
#endif
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

#ifdef DNNL_GPU_ENABLE_MDAPI

static bool open_metrics_device(MetricsDiscovery::IMetricsDeviceLatest **device,
        const std::shared_ptr<void> &lib) {
    static MetricsDiscovery::OpenMetricsDevice_fn func;
    if (!func) { *(void **)(&func) = dlsym(lib.get(), "OpenMetricsDevice"); }
    if (!func) return false;
    auto code = func(device);
    return code == MetricsDiscovery::CC_OK;
}

static bool close_metrics_device(MetricsDiscovery::IMetricsDeviceLatest *device,
        const std::shared_ptr<void> &lib) {
    static MetricsDiscovery::CloseMetricsDevice_fn func;
    if (!func) { *(void **)(&func) = dlsym(lib.get(), "CloseMetricsDevice"); }
    if (!func) return false;
    auto code = func(device);
    return code == MetricsDiscovery::CC_OK;
}

static void open_lib(std::shared_ptr<void> &lib) {
    // The library was renamed from `libmd.so.1` to `libigdmd.so.1`. Try the
    // new name first and fall back to the old for compatibility.
    void *handle = dlopen("libigdmd.so.1", RTLD_LAZY);
    if (!handle) handle = dlopen("libmd.so.1", RTLD_LAZY);
    if (handle) {
        lib = std::shared_ptr<void>(handle, [](void *h) {
            if (h) dlclose(h);
        });
    }
}

class mdapi_helper_impl_t {
public:
    mdapi_helper_impl_t() {
        if (!lib) open_lib(lib);
        if (!open_metrics_device(&metric_device_, lib)) return;
        if (!activate_freq_metric()) return;
        is_initialized_ = true;
    }

    ~mdapi_helper_impl_t() { close_metrics_device(metric_device_, lib); }

    cl_command_queue create_queue(
            cl_context ctx, cl_device_id dev, cl_int *err) const {
        if (!is_initialized_) {
            *err = CL_INVALID_VALUE;
            return nullptr;
        }
        using clCreatePerfCountersCommandQueueINTEL_func_t
                = cl_command_queue (*)(cl_context, cl_device_id,
                        cl_command_queue_properties, cl_uint, cl_int *);
        static xpu::ocl::ext_func_t<
                clCreatePerfCountersCommandQueueINTEL_func_t>
                create_queue_with_perf_counters(
                        "clCreatePerfCountersCommandQueueINTEL");
        auto func = create_queue_with_perf_counters.get_func(
                xpu::ocl::get_platform(dev));
        if (!func) {
            *err = CL_INVALID_VALUE;
            return nullptr;
        }
        auto config = metric_set_->GetParams()->ApiSpecificId.OCL;
        return func(ctx, dev, CL_QUEUE_PROFILING_ENABLE, config, err);
    }

    double get_freq(cl_event event) const {
        if (!is_initialized_) return 0;

        using namespace MetricsDiscovery;
        auto mparams = metric_set_->GetParams();
        auto report_size = mparams->QueryReportSize;
        size_t out_size = report_size;
        std::vector<uint8_t> report(report_size);

        cl_int err;
        err = xpu::ocl::clGetEventProfilingInfo(event,
                CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL, report_size,
                report.data(), &out_size);
        if (err != CL_SUCCESS) return 0;
        if (out_size != report_size) return 0;

        std::vector<TTypedValue_1_0> results(
                mparams->MetricsCount + mparams->InformationCount);

        uint32_t report_count = 0;
        TCompletionCode code;
        code = metric_set_->CalculateMetrics(report.data(), report_size,
                results.data(),
                (uint32_t)(results.size() * sizeof(TTypedValue_1_0)),
                &report_count, false);
        if (code != CC_OK) return 0;
        if (report_count < 1) return 0;

        auto &value = results[freq_metric_idx_];
        assert(value.ValueType == EValueType::VALUE_TYPE_UINT64);
        return value.ValueUInt64 * 1e6;
    }

private:
    bool activate_freq_metric() {
        using namespace MetricsDiscovery;

        auto *params = metric_device_->GetParams();

        int major = params->Version.MajorNumber;
        int minor = params->Version.MinorNumber;
        auto _1 = 1;
        if (std::tie(major, minor) < std::tie(_1, _1)) return false;

        auto api_mask = API_TYPE_OCL;
        for (uint32_t i = 0; i < params->ConcurrentGroupsCount; i++) {
            auto group = metric_device_->GetConcurrentGroup(i);
            auto gparams = group->GetParams();
            for (uint32_t j = 0; j < gparams->MetricSetsCount; j++) {
                auto set = group->GetMetricSet(j);
                auto sparams = set->GetParams();
                if (!(sparams->ApiMask & api_mask)) continue;
                if (!strcmp(sparams->SymbolName, "ComputeBasic")) {
                    metric_set_ = set;

                    for (uint32_t k = 0; k < sparams->MetricsCount; k++) {
                        auto metric = set->GetMetric(k);
                        auto mparams = metric->GetParams();
                        if (!strcmp(mparams->SymbolName,
                                    "AvgGpuCoreFrequencyMHz"))
                            freq_metric_idx_ = k;
                    }
                }
            }
        }

        if (freq_metric_idx_ < 0) return false;

        TCompletionCode code;
        code = metric_set_->SetApiFiltering(api_mask);
        if (code != CC_OK) return false;

        code = metric_set_->Activate();
        if (code != CC_OK) return false;

        return true;
    }

    bool is_initialized_ = false;
    MetricsDiscovery::IMetricsDeviceLatest *metric_device_ = nullptr;
    MetricsDiscovery::IMetricSet_1_1 *metric_set_ = nullptr;
    int freq_metric_idx_ = -1;
    std::shared_ptr<void> lib = nullptr;
};

static std::shared_ptr<mdapi_helper_impl_t> &mdapi_helper_impl() {
    static auto instance = std::make_shared<mdapi_helper_impl_t>();
    return instance;
}

mdapi_helper_t::mdapi_helper_t() : impl_(mdapi_helper_impl()) {}

cl_command_queue mdapi_helper_t::create_queue(
        cl_context ctx, cl_device_id dev, cl_int *err) const {
    return impl_->create_queue(ctx, dev, err);
}

double mdapi_helper_t::get_freq(cl_event event) const {
    return impl_->get_freq(event);
}

#else

mdapi_helper_t::mdapi_helper_t() = default;

cl_command_queue mdapi_helper_t::create_queue(
        cl_context ctx, cl_device_id dev, cl_int *err) const {
    *err = CL_INVALID_VALUE;
    return nullptr;
}

double mdapi_helper_t::get_freq(cl_event event) const {
    return 0;
}

#endif

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
