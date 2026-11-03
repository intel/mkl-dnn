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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "xpu/sycl/compat.hpp"

#include <memory>

using namespace sycl;

namespace dnnl {

enum class dev_kind { cpu, cpu_only, gpu, gpu_only };
enum class ctx_kind { cpu, cpu_only, gpu, gpu_only };

namespace {

std::string to_string(dev_kind kind) {
    static const char *strs[] = {"CPU", "CPU-only", "GPU", "GPU-only"};
    return strs[static_cast<int>(kind)];
}

std::string to_string(ctx_kind kind) {
    static const char *strs[] = {"CPU", "CPU-only", "GPU", "GPU-only"};
    return strs[static_cast<int>(kind)];
}

} // namespace

struct sycl_engine_test_params_t {
    dev_kind adev_kind;
    ctx_kind actx_kind;
    dnnl_status_t expected_status;
};

class sycl_engine_test_t
    : public ::testing::TestWithParam<sycl_engine_test_params_t> {
protected:
    void SetUp() override {
        for (auto &plat : platform::get_platforms()) {
            for (auto &dev : plat.get_devices()) {
                if (dev.is_gpu()) {
                    if (!gpu_dev
                            && dev.get_backend()
                                    == sycl::backend::ext_oneapi_level_zero) {
                        gpu_dev.reset(new device(dev));
                        gpu_ctx.reset(new context(*gpu_dev));
                    }
                    if (!gpu_only_dev && !dev.is_cpu()
                            && dev.get_backend()
                                    == sycl::backend::ext_oneapi_level_zero) {
                        gpu_only_dev.reset(new device(dev));
                        gpu_only_ctx.reset(new context(*gpu_only_dev));
                    }
                } else if (dev.is_cpu()) {
                    if (!cpu_dev) {
                        cpu_dev.reset(new device(dev));
                        cpu_ctx.reset(new context(*cpu_dev));
                    }
                    if (!cpu_only_dev && !dev.is_gpu()) {
                        cpu_only_dev.reset(new device(dev));
                        cpu_only_ctx.reset(new context(*cpu_only_dev));
                    }
                }
            }
        }
    }

    void TearDown() override {}

    std::unique_ptr<device> cpu_dev, cpu_only_dev;
    std::unique_ptr<device> gpu_dev, gpu_only_dev;
    std::unique_ptr<context> cpu_ctx, cpu_only_ctx;
    std::unique_ptr<context> gpu_ctx, gpu_only_ctx;
};

TEST_P(sycl_engine_test_t, BasicInterop) {
    auto param = GetParam();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    SKIP_IF(param.adev_kind != dev_kind::gpu
                    && param.adev_kind != dev_kind::gpu_only,
            "Skip non-GPU engine kinds for GPU only configuration");
#endif

    device *dev_ptr = nullptr;
    switch (param.adev_kind) {
        case dev_kind::cpu: dev_ptr = cpu_dev.get(); break;
        case dev_kind::cpu_only: dev_ptr = cpu_only_dev.get(); break;
        case dev_kind::gpu: dev_ptr = gpu_dev.get(); break;
        case dev_kind::gpu_only: dev_ptr = gpu_only_dev.get(); break;
    }
    context *ctx_ptr = nullptr;
    switch (param.actx_kind) {
        case ctx_kind::cpu: ctx_ptr = cpu_ctx.get(); break;
        case ctx_kind::cpu_only: ctx_ptr = cpu_only_ctx.get(); break;
        case ctx_kind::gpu: ctx_ptr = gpu_ctx.get(); break;
        case ctx_kind::gpu_only: ctx_ptr = gpu_only_ctx.get(); break;
    }

    SKIP_IF(!dev_ptr, to_string(param.adev_kind) + " device not found");
    SKIP_IF(!ctx_ptr, to_string(param.actx_kind) + " context not found");

    auto &dev = *dev_ptr;
    auto &ctx = *ctx_ptr;

    catch_expected_failures([&]() {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
        if (!impl::utils::one_of(
                    param.adev_kind, dev_kind::gpu, dev_kind::gpu_only)) {
            engine eng(engine::kind::cpu, 0);
            EXPECT_ANY_THROW(sycl_interop::make_engine(dev, ctx));
            EXPECT_ANY_THROW(sycl_interop::get_device(eng));
            EXPECT_ANY_THROW(sycl_interop::get_context(eng));

            return;
        }
#endif
        auto eng = sycl_interop::make_engine(dev, ctx);
        if (param.expected_status != dnnl_success) {
            FAIL() << "Success not expected";
        }

        EXPECT_EQ(sycl_interop::get_device(eng), dev);
        EXPECT_EQ(sycl_interop::get_context(eng), ctx);
    }, param.expected_status != dnnl_success, param.expected_status);
}

TEST_P(sycl_engine_test_t, SubDevice) {
    auto param = GetParam();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    SKIP_IF(param.adev_kind != dev_kind::gpu
                    && param.adev_kind != dev_kind::gpu_only,
            "Skip non-GPU engine kinds for GPU only configuration");
#endif

    SKIP_IF(param.expected_status != dnnl_success,
            "Don't test for failed scenarios");
    SKIP_IF(!gpu_dev.get(), "Non GPU doesn't support sub-devices");

    auto &dev = *gpu_dev;
    auto max_sub_devices
            = dev.get_info<info::device::partition_max_sub_devices>();
    SKIP_IF(max_sub_devices < 2, "This GPU doesn't support sub-devices");

    auto sub_devs = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
    context sub_ctx(sub_devs);

    const auto test_subdevice = [&](const engine &eng) {
        memory::dims tz = {2, 3, 4, 5};
        memory::desc mem_d(
                tz, memory::data_type::f32, memory::format_tag::nchw);
        auto mem = test::make_memory(mem_d, eng);

        auto eltwise_pd
                = eltwise_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
        auto eltwise = eltwise_forward(eltwise_pd);

        stream s(eng);
        eltwise.execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
        s.wait();
    };

    catch_expected_failures([&]() {
        for (const auto &sub_dev_i : sub_devs) {
            engine eng;
            // Test case when each sub-device has its own context.
            ASSERT_NO_THROW(
                    eng = sycl_interop::make_engine(sub_dev_i, sub_ctx));
            test_subdevice(eng);

            // Test case when a sub-device is used with the default
            // context.
            const auto get_default_context = [&]() {
#if defined(SYCL_KHR_DEFAULT_CONTEXT)
                return sub_dev_i.get_platform().khr_get_default_context();
#else
                return sub_dev_i.get_platform()
                        .ext_oneapi_get_default_context();
#endif
            };
            ASSERT_NO_THROW(eng = sycl_interop::make_engine(
                                    sub_dev_i, get_default_context()));
            test_subdevice(eng);
        }
    }, param.expected_status != dnnl_success, param.expected_status);
}

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
TEST_P(sycl_engine_test_t, non_sycl_cpu_runtime) {
    try {
        device dev(dnnl::impl::xpu::sycl::compat::cpu_selector_v);
        context ctx(dev);
        EXPECT_ANY_THROW(sycl_interop::make_engine(dev, ctx));
    } catch (::sycl::exception &e) {
        SKIP_IF(true,
                "Skip this test as CPU device couldn't be created, probably "
                "due to a missing CPU OpenCL or TBB");
    }
}
#endif

INSTANTIATE_TEST_SUITE_P(Simple, sycl_engine_test_t,
        ::testing::Values(sycl_engine_test_params_t {dev_kind::gpu,
                                  ctx_kind::gpu, dnnl_success},
                sycl_engine_test_params_t {
                        dev_kind::cpu, ctx_kind::cpu, dnnl_success}));

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
INSTANTIATE_TEST_SUITE_P(InvalidArgs, sycl_engine_test_t,
        ::testing::Values(sycl_engine_test_params_t {dev_kind::cpu_only,
                ctx_kind::gpu_only, dnnl_invalid_arguments}));
#endif

} // namespace dnnl
