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

#ifndef GEMMSTONE_EXAMPLES_UTILS_HPP
#define GEMMSTONE_EXAMPLES_UTILS_HPP

#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

#include "gemmstone/dsl/runtime.hpp"

// Remove once these headers are publicly available

namespace gemmstone {
namespace examples {

template <typename T>
class Array {
    std::vector<T> data_;

public:
    Array() = default;
    explicit Array(size_t n, T val = T {}) : data_(n, val) {}
    explicit Array(std::vector<T> v) : data_(std::move(v)) {}

    static Array zeros(size_t n) { return Array(n, T {0}); }
    static Array ones(size_t n) { return Array(n, T {1}); }
    static Array full(size_t n, T val) { return Array(n, val); }

    static Array arange(T start, T end, T step = T {1}) {
        std::vector<T> v;
        for (T i = start; i < end; i += step)
            v.push_back(i);
        return Array(std::move(v));
    }

    static Array randn(size_t n, float scale = 1.0f, int seed = 42) {
        std::vector<T> v(n);
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto &x : v)
            x = T(dist(gen));
        return Array(std::move(v));
    }

    size_t size() const { return data_.size(); }
    T &operator[](size_t i) { return data_[i]; }
    const T &operator[](size_t i) const { return data_[i]; }
    std::vector<T> &data() { return data_; }
    const std::vector<T> &data() const { return data_; }
    T *ptr() { return data_.data(); }
    const T *ptr() const { return data_.data(); }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    void print(const std::string &name = "") const {
        if (!name.empty()) std::cout << name << ":\n";
        std::cout << *this << std::endl;
    }

    // NumPy-like style printing.
    friend std::ostream &operator<<(std::ostream &out, const Array &arr) {
        if (arr.size() == 0) return out << "[]";

        const size_t threshold = 1000;
        const size_t edgeitems = 3;
        const size_t linewidth = 80;
        const int precision = 4;

        std::string res;
        bool summarizing = arr.size() > threshold;

        auto to_str = [&](size_t i, int pad = 0) {
            std::ostringstream oss;
            if (pad > 0) oss << std::setw(pad);
            oss << std::setprecision(precision);
            oss << arr[i];
            return oss.str();
        };

        size_t width = 0;
        out << "[";
        if (summarizing) {
            for (size_t i = 0; i < arr.size(); i++) {
                if (i == edgeitems) {
                    i = arr.size() - edgeitems - 1;
                    continue;
                }
                width = std::max(width, to_str(i).length());
            }
            width += 1;
            for (size_t i = 0; i < arr.size(); i++) {
                if (i == edgeitems) {
                    std::string gap = "..., ";
                    res += gap;
                    i = arr.size() - edgeitems - 1;
                    continue;
                }
                out << to_str(i, width);
            }
            return out;
        }

        for (size_t i = 0; i < arr.size(); i++)
            width = std::max(width, to_str(i).length());
        width += 1;

        size_t values_per_row = std::max((size_t)1, linewidth / width);
        for (size_t i = 0; i < arr.size(); i++) {
            if (i > 0 && i % values_per_row == 0) out << "\n ";
            out << to_str(i, width);
        }
        out << "]";
        return out;
    }

    friend Array<T> operator*(T a, Array<T> b) {
        for (auto &e : b)
            e *= a;
        return b;
    }
};

template <typename T>
bool allclose(const Array<T> &a, const Array<T> &b, float atol = 1e-5f,
        float rtol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::abs(float(a[i]) - float(b[i]));
        if (diff > atol + rtol * std::abs(float(b[i]))) return false;
    }
    return true;
}

template <typename T>
void assert_close(const Array<T> &a, const Array<T> &b, float atol = 1e-5f,
        float rtol = 1e-5f) {
    if (allclose(a, b, atol, rtol)) return;
    std::cerr << "assert_close failed!"
              << " (atol=" << atol << ", rtol=" << rtol
              << ", size=" << a.size() << ")\n";
    std::exit(1);
}

struct NDRange {
    NDRange(uint32_t global, uint32_t local)
        : ndims(1), global {global, 1, 1}, local {local, 1, 1} {}
    NDRange(std::array<uint32_t, 2> global, std::array<uint32_t, 2> local)
        : ndims(2)
        , global {global[0], global[1], 1}
        , local {local[0], local[1], 1} {}
    NDRange(std::array<uint32_t, 3> global, std::array<uint32_t, 3> local)
        : ndims(3), global {global}, local {local} {}

    int ndims = 0;
    std::array<uint32_t, 3> global = {};
    std::array<uint32_t, 3> local = {};
};

#ifdef GEMMSTONE_WITH_L0_RUNTIME
void handleL0(ze_result_t status) {
    if (status != ZE_RESULT_SUCCESS) {
        std::ostringstream oss;
        oss << "Level Zero error occurred: 0x" << std::hex << status;
        throw std::runtime_error(oss.str());
    }
}

void findGPU(ze_driver_handle_t &out_driver, ze_context_handle_t &out_ctx,
        ze_device_handle_t &out_dev) {
    // Initialize the driver.
    handleL0(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Discover all the driver instances.
    uint32_t driverCount = 0;
    handleL0(zeDriverGet(&driverCount, nullptr));

    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    handleL0(zeDriverGet(&driverCount, allDrivers.data()));

    // Find a driver instance with a GPU device
    std::vector<ze_device_handle_t> allDevices;

    for (uint32_t i = 0; i < driverCount; i++) {
        uint32_t deviceCount = 0;
        handleL0(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));

        allDevices.resize(deviceCount);
        handleL0(zeDeviceGet(allDrivers[i], &deviceCount, allDevices.data()));

        for (auto dev : allDevices) {
            ze_device_properties_t props
                    = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
            handleL0(zeDeviceGetProperties(dev, &props));

            if (props.type == ZE_DEVICE_TYPE_GPU) {
                ze_context_desc_t ctxDesc
                        = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};

                ze_context_handle_t ctx;
                handleL0(zeContextCreate(allDrivers[i], &ctxDesc, &ctx));

                out_driver = allDrivers[i];
                out_dev = dev;
                out_ctx = ctx;
                return;
            }
        }
    }

    std::cerr << "No GPUs found.\n";
    exit(1);
}

ze_command_queue_handle_t makeQueue(
        ze_context_handle_t context, ze_device_handle_t device) {
    ze_command_queue_desc_t queueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            nullptr, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_handle_t queue;
    handleL0(zeCommandQueueCreate(context, device, &queueDesc, &queue));

    return queue;
}

// Only use via l0_driver_context, this structure leaks resources to avoid
// issues with library unload ordering in static objects.
struct LevelZeroDriverContext {
    LevelZeroDriverContext() {
        findGPU(driver, context, device);
        queue = makeQueue(context, device);
    }
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_command_queue_handle_t queue;
};

const LevelZeroDriverContext &l0_driver_context() {
    static LevelZeroDriverContext ctx;
    return ctx;
}

struct LevelZeroNDRange {
    LevelZeroNDRange(const NDRange range)
        : ndims(range.ndims)
        , groups {range.global[0] / range.local[0],
                  range.global[1] / range.local[1],
                  range.global[2] / range.local[2]}
        , local {range.local}
    {
    }

    int ndims;
    ze_group_count_t groups = {};
    std::array<uint32_t, 3> local = {};
};

// Helper buffer for recording L0 buffers required by a command list.
struct VectorBuffers {
    void *vector = nullptr;
    void *host = nullptr;
    void *device = nullptr;
    size_t size = 0;
};

template <typename KernelArg0, typename... KernelArgs>
std::vector<VectorBuffers> build_list(const LevelZeroDriverContext &ctx,
        ze_command_list_handle_t list, ze_kernel_handle_t kernel,
        LevelZeroNDRange range, int idx, KernelArg0 &karg0,
        KernelArgs &...kargs) {
    handleL0(zeKernelSetArgumentValue(kernel, idx, sizeof(karg0), &karg0));
    return build_list(ctx, list, kernel, range, idx + 1, kargs...);
}

template <typename KernelArg0, typename... KernelArgs>
std::vector<VectorBuffers> build_list(const LevelZeroDriverContext &ctx,
        ze_command_list_handle_t list, ze_kernel_handle_t kernel,
        LevelZeroNDRange range, int idx, std::vector<KernelArg0> &karg0,
        KernelArgs &...kargs) {
    size_t size = karg0.size() * sizeof(KernelArg0);

    void *host_buffer;
    {
        ze_host_mem_alloc_desc_t hmaDesc = {
                ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                nullptr,
                0,
        };
        handleL0(zeMemAllocHost(
                ctx.context, &hmaDesc, size, sizeof(KernelArg0), &host_buffer));
    }

    void *device_buffer;
    {
        ze_device_mem_alloc_desc_t dmaDesc
                = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
        handleL0(zeMemAllocDevice(ctx.context, &dmaDesc, size,
                sizeof(KernelArg0), ctx.device, &device_buffer));
    }

    std::memcpy(host_buffer, karg0.data(), size);
    handleL0(zeCommandListAppendMemoryCopy(
            list, device_buffer, host_buffer, size, nullptr, 0, nullptr));
    handleL0(zeKernelSetArgumentValue(
            kernel, idx, sizeof(device_buffer), &device_buffer));

    auto ret = build_list(ctx, list, kernel, range, idx + 1, kargs...);

    handleL0(zeCommandListAppendMemoryCopy(
            list, host_buffer, device_buffer, size, nullptr, 0, nullptr));

    ret.emplace_back(karg0.data(), host_buffer, device_buffer, size);
    return ret;
}

template <typename KernelArg0, typename... KernelArgs>
std::vector<VectorBuffers> build_list(const LevelZeroDriverContext &ctx,
        ze_command_list_handle_t list, ze_kernel_handle_t kernel,
        LevelZeroNDRange range, int idx, Array<KernelArg0> &karg0,
        KernelArgs &...kargs) {
    return build_list(ctx, list, kernel, range, idx, karg0.data(), kargs...);
}

std::vector<VectorBuffers> build_list(const LevelZeroDriverContext &ctx,
        ze_command_list_handle_t list, ze_kernel_handle_t kernel,
        LevelZeroNDRange range, int idx) {
    handleL0(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    handleL0(zeKernelSetGroupSize(
            kernel, range.local[0], range.local[1], range.local[2]));

    handleL0(zeCommandListAppendLaunchKernel(
            list, kernel, &range.groups, nullptr, 0, nullptr));
    handleL0(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    return {};
}

#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
void handleCL(cl_int status) {
    if (status != CL_SUCCESS) {
        throw(std::runtime_error(
                "OpenCL error occurred: " + std::to_string(status)));
    }
}

cl_device_id findGPU() {
    std::vector<cl_platform_id> platforms(16);
    cl_uint count;

    handleCL(clGetPlatformIDs(platforms.size(), platforms.data(), &count));
    platforms.resize(count);

    for (auto platform : platforms) {
        cl_device_id device;
        cl_int status;

        status = clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (status == CL_SUCCESS)
            return device;
        else if (status != CL_DEVICE_NOT_FOUND)
            handleCL(status);
    }

    std::cerr << "No GPUs found.\n";
    exit(1);
}

// Only use via ocl_driver_context, this structure leaks resources to avoid
// issues with library unload ordering in static objects.
struct OpenCLDriverContext {
    OpenCLDriverContext() {
        cl_int status;
        device = findGPU();
        context = clCreateContext(
                nullptr, 1, &device, nullptr, nullptr, &status);
        handleCL(status);
        queue = clCreateCommandQueueWithProperties(
                context, device, nullptr, &status);
        handleCL(status);
    }
    bool is_static = false;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

const OpenCLDriverContext &ocl_driver_context() {
    static OpenCLDriverContext ctx;
    return ctx;
}

struct OpenCLNDRange {
    OpenCLNDRange(const NDRange range)
        : ndims(range.ndims)
        , global {range.global[0], range.global[1], range.global[2]}
        , local {range.local[0], range.local[1], range.local[2]} {
    }
    int ndims = 0;
    std::array<size_t, 3> global = {};
    std::array<size_t, 3> local = {};
};

template <typename KernelArg0, typename... KernelArgs>
void execute(const OpenCLDriverContext &ctx, cl_kernel kernel,
        OpenCLNDRange range, int idx, KernelArg0 &karg0, KernelArgs &...kargs) {
    handleCL(clSetKernelArg(kernel, idx, sizeof(karg0), &karg0));
    execute(ctx, kernel, range, idx + 1, kargs...);
}

template <typename KernelArg0, typename... KernelArgs>
void execute(const OpenCLDriverContext &ctx, cl_kernel kernel,
        OpenCLNDRange range, int idx, std::vector<KernelArg0> &karg0,
        KernelArgs &...kargs) {

    cl_int status;
    auto buffer_bytes = karg0.size() * sizeof(KernelArg0);
    auto device_buffer = clCreateBuffer(ctx.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_bytes,
            karg0.data(), &status);
    handleCL(status);

    handleCL(
            clSetKernelArg(kernel, idx, sizeof(device_buffer), &device_buffer));
    execute(ctx, kernel, range, idx + 1, kargs...);

    handleCL(clEnqueueReadBuffer(ctx.queue, device_buffer, CL_TRUE, 0,
            buffer_bytes, karg0.data(), 0, nullptr, nullptr));
    handleCL(clReleaseMemObject(device_buffer));
}

template <typename KernelArg0, typename... KernelArgs>
void execute(const OpenCLDriverContext &ctx, cl_kernel kernel,
        OpenCLNDRange range, int idx, Array<KernelArg0> &karg0,
        KernelArgs &...kargs) {
    execute(ctx, kernel, range, idx, karg0.data(), kargs...);
}

void execute(const OpenCLDriverContext &ctx, cl_kernel kernel,
        OpenCLNDRange range, int idx) {
    handleCL(clEnqueueNDRangeKernel(ctx.queue, kernel, range.ndims, nullptr,
            range.global.data(), range.local.data(), 0, nullptr, nullptr));
}

#endif

gemmstone::dsl::hw_t get_hardware(const std::string &target) {
    if (target == "opencl") {
#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
        auto &ctx = ocl_driver_context();
        return gemmstone::get_hardware(ctx.device, ctx.context);
#else
        std::cout << "OpenCL Runtime support is disabled\n";
        exit(1);
#endif
    }
    if (target == "l0") {
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        auto &ctx = l0_driver_context();
        return gemmstone::get_hardware(ctx.device, ctx.context);
#else
        std::cout << "Level Zero Runtime support is disabled\n";
        exit(1);
#endif
    }

    using hw_t = gemmstone::dsl::hw_t;
    using family_t = ngen::ProductFamily;
    if (target == "xelp") return hw_t(family_t::GenericXeLP);
    if (target == "dg2") return hw_t(family_t::DG2);
    if (target == "mtl") return hw_t(family_t::MTL);
    if (target == "arl") return hw_t(family_t::ARL);
    if (target == "xe") return hw_t(family_t::GenericXe2);
    if (target == "pvc") return hw_t(family_t::PVC);
    if (target == "xe2") return hw_t(family_t::GenericXe2);
    if (target == "bmg") return hw_t(family_t::BMG);
    if (target == "lnl") return hw_t(family_t::LNL);
    if (target == "xe3") return hw_t(family_t::GenericXe3);
    if (target == "ptl") return hw_t(family_t::GenericXe3);
    if (target == "xe3p") return hw_t(family_t::GenericXe3p);
    if (target == "nvlp") return hw_t(family_t::NVLP);

    std::cout << "Unknown Target, target must be one of: l0, opencl, xelp, "
                 "dg2, mtl, arl, pvc, xe2, bmg, lnl, xe3, ptl";
    std::cout << ", xe3p, nvlp";
    std::cout << "\n";

    exit(1);
}

void handle_hardware_target(
        const std::string &target, const gemmstone::dsl::kernel_t &kernel) {
    if (target == "opencl" || target == "l0") return;
#ifdef GEMMSTONE_WITH_ASM_RUNTIME
    std::cout << make_asm(kernel);
    exit(0);
#else
    std::cout << "Assembly Runtime support is disabled\n";
    exit(1);
#endif
}

template <typename... KernelArgs>
void execute(const std::string &runtime, const gemmstone::dsl::kernel_t &kernel,
        NDRange range, KernelArgs &...kargs) {
    if (runtime == "opencl") {
#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
        auto ctx = ocl_driver_context();
        auto cl_kernel
                = gemmstone::dsl::make_kernel(kernel, ctx.context, ctx.device);
        execute(ctx, cl_kernel, range, 0, kargs...);
        handleCL(clReleaseKernel(cl_kernel));
#else
        std::cout << "OpenCL Runtime support is disabled\n";
        exit(1);
#endif
    }
    if (runtime == "l0") {
#ifdef GEMMSTONE_WITH_L0_RUNTIME
        auto ctx = l0_driver_context();
        auto km = gemmstone::dsl::make_kernel(kernel, ctx.context, ctx.device);
        ze_command_list_desc_t listDesc
                = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
        ze_command_list_handle_t list;
        handleL0(
                zeCommandListCreate(ctx.context, ctx.device, &listDesc, &list));

        auto buffers = build_list(ctx, list, km.kernel, range, 0, kargs...);

        handleL0(zeCommandListClose(list));
        handleL0(zeCommandQueueExecuteCommandLists(
                ctx.queue, 1, &list, nullptr));
        handleL0(zeCommandQueueSynchronize(ctx.queue, UINT64_MAX));
        handleL0(zeCommandListDestroy(list));

        for (auto &b : buffers) {
            std::memcpy(b.vector, b.host, b.size);
            handleL0(zeMemFree(ctx.context, b.host));
            handleL0(zeMemFree(ctx.context, b.device));
        }
        handleL0(zeKernelDestroy(km.kernel));
        handleL0(zeModuleDestroy(km.module));
#else
        std::cout << "Level Zero Runtime support is disabled\n";
        exit(1);
#endif
    }
}

struct bf16_t {
    bf16_t() = default;
    bf16_t(float f) { (*this) = f; }
    bf16_t &operator=(float f) {
        std::array<uint16_t, 2> iraw;
        std::memcpy(&iraw, &f, sizeof(iraw));
        switch (std::fpclassify(f)) {
            case FP_SUBNORMAL:
            case FP_ZERO:
                // sign preserving zero (denormal go to zero)
                raw_bits_ = iraw[1];
                raw_bits_ &= 0x8000;
                break;
            case FP_INFINITE: raw_bits_ = iraw[1]; break;
            case FP_NAN:
                // truncate and set MSB of the mantissa force QNAN
                raw_bits_ = iraw[1];
                raw_bits_ |= 1 << 6;
                break;
            case FP_NORMAL:
                // round to nearest even and truncate
                const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
                uint32_t int_raw;
                std::memcpy(&int_raw, &f, sizeof(f));
                int_raw += rounding_bias;
                raw_bits_ = int_raw >> 16;
                break;
        }
        return *this;
    }
    operator float() const {
        float ret;
        std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
        std::memcpy(&ret, &iraw, sizeof(iraw));
        return ret;
    }

private:
    uint16_t raw_bits_ = 0;
};

} // namespace examples
} // namespace gemmstone

#endif
