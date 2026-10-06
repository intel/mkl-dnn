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

#ifndef _WIN32
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wnarrowing"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#ifdef TARGET_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 210
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "mdapi/metrics_discovery_api.h"
#elif defined(TARGET_LEVEL_ZERO)
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include "gemmstone/runtime.hpp"
#elif defined(TARGET_DPCPP)
#include <CL/sycl.hpp>
#else
#error Runtime must be chosen by defining one of the TARGET_* macros.
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#endif

#include <stdio.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "benchstone.hpp"
#include "benchstone_matrix.hpp"
#include "benchstone_ref.hpp"
#include "kernel_info.hpp"

#include "gemmstone/driver_info.hpp"
#include "generator/pieces/hw_utils.hpp"
#include "flush_kernel_cl.h"
#include "tools/perf_lock.hpp"

#ifndef GEMMSTONE_WITH_MKL
#include "tools/mini_mkl.hpp"
#elif __has_include("mkl/mkl_lapack.h")
#include "mkl/mkl_lapack.h"
#else
#include "mkl_lapack.h"
#endif
using namespace NGEN_NAMESPACE::utils;
using NGEN_NAMESPACE::HW;
using namespace gemmstone;

#ifdef TARGET_OPENCL

#ifndef CL_MEM_UNCOMPRESSED_HINT_INTEL
#define CL_MEM_UNCOMPRESSED_HINT_INTEL (1u << 22)
#endif
#ifndef CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL
#define CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL 0x407F
#endif

constexpr auto Runtime = benchstone::RuntimeBackend::OpenCL;

using rt_context = cl_context;
using rt_device = cl_device_id;
using rt_queue = cl_command_queue;
using rt_list = void *;
using rt_event = cl_event;
using rt_program = cl_program;
using rt_kernel = cl_kernel;

#ifdef TARGET_USM
using rt_buffer = void *;
#else
using rt_buffer = cl_mem;
#endif

#elif defined(TARGET_LEVEL_ZERO)

// Remove once these headers are publicly available

constexpr auto Runtime = benchstone::RuntimeBackend::LevelZero;

using rt_context = ze_context_handle_t;
using rt_device = ze_device_handle_t;
using rt_queue = ze_command_queue_handle_t;
using rt_list = ze_command_list_handle_t;
using rt_event = ze_event_handle_t;
using rt_program = ze_module_handle_t;
using rt_kernel = ze_kernel_handle_t;
using rt_buffer = void *;

#elif defined(TARGET_DPCPP)

using rt_context = sycl::context *;
using rt_device = sycl::device *;
using rt_queue = sycl::queue *;
using rt_list = void *;
using rt_event = sycl::event *;
using rt_program = sycl::program *;
using rt_kernel = sycl::kernel *;

#ifdef TARGET_USM
using rt_buffer = void *;
#else
using rt_buffer = sycl::buffer<char, 1> *;
#endif

#endif

namespace benchstone {

template <RuntimeBackend backend>
class DriverRT;
template <RuntimeBackend runtime_>
struct DriverRTArgs;
template <RuntimeBackend runtime_>
struct PoolBuffer;
template <RuntimeBackend runtime_>
struct DriverStorage;

template <>
struct PoolBuffer<Runtime> {
    rt_buffer handle = nullptr;
    rt_context context = nullptr;
    size_t capacity = 0;
    BufferFlags flags = 0;
    bool hostBound = false;
    size_t flushBytes = 0;

    PoolBuffer() = default;
    PoolBuffer(DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
               void *hostPtr = nullptr, size_t offset = 0, bool coldCache = false);
    ~PoolBuffer();

    PoolBuffer(const PoolBuffer &) = delete;
    PoolBuffer &operator=(const PoolBuffer &) = delete;

    PoolBuffer(PoolBuffer &&other) noexcept
    {
        swap(*this, other);
    }
    PoolBuffer &operator=(PoolBuffer &&other) noexcept
    {
        if (this != &other) {
            PoolBuffer temp(std::move(other));
            swap(*this, temp);
        }
        return *this;
    }
    friend void swap(PoolBuffer &a, PoolBuffer &b) noexcept
    {
        using std::swap;
        swap(a.handle, b.handle);
        swap(a.capacity, b.capacity);
        swap(a.context, b.context);
        swap(a.flags, b.flags);
        swap(a.hostBound, b.hostBound);
        swap(a.flushBytes, b.flushBytes);
    }

    rt_buffer get() const { return handle; }
    operator rt_buffer() const { return get(); }
    bool sized(size_t bytes) const { return handle && capacity >= bytes; }
    bool reusable(DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
                  void *hostPtr) const;
    void setFlushSize(size_t bytes) { flushBytes = bytes; }
    size_t getFlushSize() const { return flushBytes; }
    rt_buffer realloc(DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
                      void *hostPtr = nullptr, size_t offset = 0, bool coldCache = false);
};

template <>
struct DriverStorage<Runtime> {
    TypedData As, Bs, Cs;
    integer trackingStride, trackingWGSize, tempCBytes;
    integer statusElems, trackingElems;
    PoolBuffer<Runtime> As_buffer, Bs_buffer, Cs_buffer, P_buffer, CO_buffer, X_buffer, S_buffer, D_buffer, plan_buffer;
    PoolBuffer<Runtime> tempCBuffer;
    PoolBuffer<Runtime> Aptrs_buffer, Bptrs_buffer, Cptrs_buffer;
    PoolBuffer<Runtime> AO_buffer, BO_buffer;
    PoolBuffer<Runtime> aScale_buffer, bScale_buffer;
    PoolBuffer<Runtime> Ag_buffer, Bg_buffer;
    PoolBuffer<Runtime> alphaBuffer, betaBuffer;
    PoolBuffer<Runtime> statusBuffer, trackingBuffer;

    template <typename Function>
    void forEachBuffer(Function function) {
        for (auto *buffer : { &As_buffer, &Bs_buffer, &Cs_buffer, &P_buffer, &CO_buffer,
                              &X_buffer, &S_buffer, &D_buffer, &plan_buffer, &tempCBuffer,
                              &Aptrs_buffer, &Bptrs_buffer, &Cptrs_buffer, &AO_buffer,
                              &BO_buffer, &aScale_buffer, &bScale_buffer, &Ag_buffer,
                              &Bg_buffer, &alphaBuffer, &betaBuffer, &statusBuffer,
                              &trackingBuffer })
            function(*buffer);
    }

    uint32_t threads;
    std::vector<gemmstone::Timepoint> LastTrackingData;
    int LastTrackingStride = 1;
    int LastTrackingWGSize = 1;
    bool metricsInitialized = false;
    bool initSuccess = false;
#if TARGET_LEVEL_ZERO
    ze_command_list_handle_t list = nullptr;
    ze_event_pool_handle_t eventPool = nullptr;
    std::vector<rt_event> curEvents;
    uint64_t *timestamps = nullptr;
#endif
};

template <>
struct DriverRTArgs<Runtime> {
    DriverStorage<Runtime> storage;
    rt_context context;
    rt_device device;
    rt_queue queue;
    rt_program program;
    rt_kernel kernel;
    rt_program flushProgram, flushICacheProgram, fillCacheProgram;
    rt_kernel flushKernel, flushICacheKernel, fillCacheKernel;
    std::vector<uint8_t> binary;
#ifdef TARGET_LEVEL_ZERO
    rt_list immediateList;
    rt_list copyList;
    rt_queue copyQueue;
    rt_device memDevice;
    ze_fence_handle_t fence;
    rt_event completionEvent;
    ze_event_pool_handle_t completionEventPool;
    zet_metric_group_handle_t metricGroup;
    zet_metric_query_pool_handle_t queryPool;
    zet_metric_query_handle_t query;

    bool timestampPropertiesValid = false;
    uint32_t timestampValidBits = 0;
    double timerResolution = 0;
#endif
#ifdef TARGET_OPENCL
    cl_platform_id platform;
    MetricsDiscovery::IMetricsDevice_1_5 *metricDevice;
    MetricsDiscovery::IMetricSet_1_1 *metricSet;
    MetricsDiscovery::IAdapter_1_6 *metricAdapter;

    std::optional<cl_device_pci_bus_info_khr> BDF;
#endif
    rt_buffer fillBuffer;
};

template <RuntimeBackend backend>
class DriverRT : public Driver {
public:
    DriverRT() : Driver(backend) {}
    ~DriverRT() override;
    DriverRTArgs<backend> rtArgs{};

    void init() override;
    void enableMetrics(bool enabled = true) override;
    bool initKernel(const KernelInfo& info, const std::vector<uint8_t> &binary) override;
    void finishKernel() override;
    void finish() override;
    TestResult testImpl(KernelInfo &info, TestMode mode,
            uint32_t batchSize, int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
            bool conja, bool conjb, bool unit, char cOffType, bool aOffset, bool bOffset) override;
    bool isKernelLoaded() const override;

    std::tuple<std::vector<gemmstone::Timepoint>, int, int> getChromeTrackingData() const override;
};
} // namespace benchstone

using DriverRT = benchstone::DriverRT<Runtime>;
using PoolBuffer = benchstone::PoolBuffer<Runtime>;
using DriverStorage = benchstone::DriverStorage<Runtime>;
using DriverRTArgs = benchstone::DriverRTArgs<Runtime>;

namespace {

rt_buffer rtCreateBackendBuffer(DriverRT &driver, BufferFlags flags, rt_buffer buffer,
                                size_t size, void *host_ptr, size_t offset);
void rtFreeBuffer(rt_context context, rt_buffer buf);
bool rtBindsHostPtr(const DriverRT &driver, BufferFlags flags, const void *host_ptr);

void rtClearColdCache(DriverStorage &storage)
{
    storage.forEachBuffer([](PoolBuffer &buffer) { buffer.setFlushSize(0); });
}

// Global settings.
bool OverrunTest = false;
bool FillCache = false;
integer StripeIndex = -1;
int PersistentGroupCountOverride = 0;
int WGStrideM = 0;
int WGStrideN = 0;
int KSlicedTilesOverride = -1;
int KSyncSlabsOverride = -1;
int KSlicedWaves = 2;
int KSlicedPhases = 1;
int K0Override = 0;
bool ShowBlockInfo = false;
bool ShowDispatchInfo = false;

#if __cplusplus >= 201703L
template <typename T>
static std::string argFmt(const T& val) {
    if constexpr (std::is_pointer_v<T>) {
        char buf[24];
        snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(val)));
        return buf;
    } else if constexpr (std::is_floating_point_v<T>) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%g", static_cast<double>(val));
        return buf;
    } else if constexpr (std::is_same_v<T, superscalar>) {
        char buf[64];
        if (val.imag() != 0.0)
            snprintf(buf, sizeof(buf), "%g+%gi", val.real(), val.imag());
        else
            snprintf(buf, sizeof(buf), "%g", val.real());
        return buf;
    } else if constexpr (std::is_same_v<T, PoolBuffer>) {
        return argFmt(val.get());
    } else {
        return std::to_string(val);
    }
}
#endif

struct ArgInfo {
#if __cplusplus >= 201703L
    struct Entry { int idx; std::string name; std::string value; };
    std::vector<Entry> entries;

    void update(int idx, const char* name, std::string value) {
        for (auto& e : entries)
            if (e.idx == idx) { e.name = name; e.value = std::move(value); return; }
        entries.push_back({idx, std::string(name), std::move(value)});
    }

    template<typename T>
    void update(int idx, const char* name, const T& value) {
        update(idx, name, argFmt(value));
    }

    void reset() { entries.clear(); }

    void print(const char* kernelName) const {
        if (entries.empty()) return;
        auto sorted = entries;
        std::sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b){ return a.idx < b.idx; });
        printf("[INFO] %s(", kernelName);
        for (size_t i = 0; i < sorted.size(); i++) { if (i) printf(", "); printf("%s:%s", sorted[i].name.c_str(), sorted[i].value.c_str()); }
        printf(")\n");
    }
#else
    template<typename T>
    void update(int, const char*, const T&) {}
    void update(int, const char*, std::string) {}
    void reset() {}
    void print(const char*) const {}
#endif
};

struct rtKernelCtx {
    rt_kernel kernel;
    ArgInfo argInfo;
};

static void rtLogDispatch(rtKernelCtx& ctx, const char* kernelName, DriverPhase phase,
                           int dims, const size_t* global, const size_t* local,
                           const int* cluster = nullptr) {
    if (!ShowDispatchInfo) return;
    if (phase != DriverPhase::PrepareList && phase != DriverPhase::Run) return;
    ctx.argInfo.print(kernelName);
    printf("[INFO] [work sizes] ");
    for (int i = 0; i < dims; i++) { if (i) printf("x"); printf("%zu", local[i]); }
    printf(" local, ");
    for (int i = 0; i < dims; i++) { if (i) printf("x"); printf("%zu", global[i]); }
    printf(" global");
    if (cluster && cluster[0] > 0)
        printf(", %dx%d cluster", cluster[0], cluster[1]);
    printf("\n");
}

bool InOrderQueue = false;
bool TestDeterminism = false;
int Tiles = 1;
size_t LLCUpperBound = 0;
size_t PageSize = 0;
size_t AllocSizeAlign = 1;
int MaxInaccuracies = 100;
bool AllowCompression = false;
bool ShowAllTimes = false;
int MaxWGM = 32, MaxWGN = 32;
bool DumpOutput = true;
const char *MetricSetName = "ComputeBasic";
int FreqMetricIndex = -1;
std::vector<int> AdditionalMetricIndices[MetricCount];
bool ShowAdditionalMetrics = false;
int ExecutionSubdevice = -1;

double GPUWaitTime = 0.;

#if TARGET_LEVEL_ZERO
int MemorySubdevice = -1;
uint32_t CmdQueueOrdinal;
uint32_t CmdQueueOrdinalCopy;
uint32_t DeviceMemOrdinal = 0;
uint32_t DeviceMemOrdinals = 0;
bool ImmediateCommandLists = false;
bool TimestampEvents = true;
#endif

void envInit(DriverRT &driver) {
    driver.setSimulation(simulation());

    // Parse env variable knobs
    auto maxInaccStr = getenv("MAX_INACCURACIES");
    if (maxInaccStr) MaxInaccuracies = atoi(maxInaccStr);

    driver.info.UniformWG = (getenv("NONUNIFORM_WG") == nullptr);
    ShowDispatchInfo = (getenv("DISPATCH_INFO") != nullptr);

    auto cfill = getenv("CACHE_FILL");
    if (cfill) FillCache = atoi(cfill);

    auto inOrder = getenv("IN_ORDER");
    InOrderQueue = inOrder && atoi(inOrder);

    auto pageSize = getenv("PAGE_SIZE");
    if (pageSize) PageSize = atoi(pageSize);

    auto allocAlign = getenv("ALLOC_ALIGN");
    if (allocAlign) AllocSizeAlign = atoi(allocAlign);

    auto allowComp = getenv("ALLOW_COMPRESSION");
    if (allowComp) AllowCompression = atoi(allowComp);

    auto checkDeterminism = getenv("CHECK_DETERMINISM");
    TestDeterminism = checkDeterminism && atoi(checkDeterminism);

    auto persistentGC = getenv("PERSISTENT_GROUPS");
    if (persistentGC) PersistentGroupCountOverride = atoi(persistentGC);

    auto wgStrideM = getenv("WG_STRIDE_M");
    if (wgStrideM) WGStrideM = atoi(wgStrideM);

    auto wgStrideN = getenv("WG_STRIDE_N");
    if (wgStrideN) WGStrideN = atoi(wgStrideN);

    auto kSlicedGC = getenv("K_SLICED_TILES");
    if (kSlicedGC) KSlicedTilesOverride = atoi(kSlicedGC);

    auto kSlicedWC = getenv("K_SLICED_WAVES");
    if (kSlicedWC) KSlicedWaves = atoi(kSlicedWC);

    auto kSlicedPC = getenv("K_SLICED_PHASES");
    if (kSlicedPC) KSlicedPhases = atoi(kSlicedPC);

    auto kSyncSlabs = getenv("K_SYNC_SLABS");
    if (kSyncSlabs) KSyncSlabsOverride = atoi(kSyncSlabs);

    auto k0 = getenv("K0");
    if (k0) K0Override = atoi(k0);

    auto index = getenv("STRIPE_INDEX");
    if (index) StripeIndex = atoi(index);

    auto extraMetrics = getenv("EXTRA_METRICS");
    if (extraMetrics) ShowAdditionalMetrics = atoi(extraMetrics);

    if (ShowAdditionalMetrics && driver.Hardware == HW::XeHPG)
        MetricSetName = "ComputeOverview";      /* includes L3 information */
    auto metricSetOverride = getenv("METRIC_SET");
    if (metricSetOverride)
        MetricSetName = metricSetOverride;

    auto showAllTimes = getenv("SHOW_ALL_TIMES");
    if (showAllTimes) ShowAllTimes = atoi(showAllTimes);

#if TARGET_LEVEL_ZERO
    auto deviceMemOrdinal = getenv("DEVICE_MEM");
    if (deviceMemOrdinal) DeviceMemOrdinal = atoi(deviceMemOrdinal);

    auto execSD = getenv("TILE");
    if (execSD) ExecutionSubdevice = MemorySubdevice = atoi(execSD);

    auto memSD = getenv("MEM_TILE");
    if (memSD) MemorySubdevice = atoi(memSD);

    auto immediate = getenv("IMMEDIATE");
    if (immediate) ImmediateCommandLists = atoi(immediate);

    auto tsEvents = getenv("TIMESTAMP_EVENTS");
    if (tsEvents) TimestampEvents = atoi(tsEvents);
#endif
};

double PeakFlops(const DriverRT &driver, Level3KernelInfo info) {
    return peakFlops(driver.HardwareProduct, driver.info.EUs, driver.info.ClockFrequency,
                     info.typeAInt(), info.typeBInt(), info.typeCInt(), info.systolic);
}

void summarize(const DriverRT &driver, TestResult &result, const Level3KernelInfo &info,
               int32_t batchSize, int32_t m, int32_t n, int32_t k, int wgLimit) {
    double avg     = result.reportedDeviceTimeIfAvailable();
    double avgEff  = result.reportedEffectiveDeviceTimeIfAvailable();

    // If the frequency is 0, we can't calculate an effective time, so just use the average time.
    if (result.avgFreq == 0.)
        avgEff = avg;

    double peakFlops = PeakFlops(driver, info);
    double nops = ops(m, n, k, info, wgLimit) * batchSize;
    double actualFlops = nops / avg;
    double efficiency = actualFlops / peakFlops * 100;
    auto memTraffic = traffic(m, n, k, batchSize, info, !(info.beta == 0), wgLimit);
    double actualBW = (memTraffic.read + memTraffic.write) / avg;
    double memEfficiency = 0.;
    double effectiveEff = efficiency * avg / avgEff;
    if (driver.info.PeakBW > 0.) {
        memEfficiency = actualBW / driver.info.PeakBW * 100;
        efficiency = std::max(efficiency, memEfficiency);
        effectiveEff = std::max(effectiveEff, memEfficiency);
    }

    result.time = avg;
    result.effTime = avgEff;
    result.bw = actualBW;
    result.flops = actualFlops;
    result.efficiency = efficiency;
    result.effEfficiency = effectiveEff;
}

#define RECORD(timer, f) do { \
    auto start = std::chrono::steady_clock::now(); \
    f; \
    auto end = std::chrono::steady_clock::now(); \
    timer += ((end - start) / std::chrono::duration<double>(1)); \
} while (false);

#if TARGET_OPENCL
MetricsDiscovery::OpenAdapterGroup_fn  OpenAdapterGroup  = nullptr;
MetricsDiscovery::OpenMetricsDevice_fn  OpenMetricsDevice  = nullptr;
MetricsDiscovery::CloseMetricsDevice_fn CloseMetricsDevice = nullptr;
#endif


void commonDeviceSetup(DriverRT &driver)
{
    if (driver.Hardware >= HW::Gen12LP && driver.Hardware <= HW::XeHPG) {
        const auto fusedStr = getenv("CFEFusedEUDispatch");
        if ((fusedStr != nullptr) && (atoi(fusedStr) != 0))
            driver.info.EUs >>= 1;
    }

    if (auto e = getenv("EU_OVERRIDE"))   driver.info.EUs = atoi(e);
    if (auto e = getenv("FREQ_OVERRIDE")) driver.info.ClockFrequency = atoi(e) * 1.0e6;
    if (auto e = getenv("MEMORY_BW"))     driver.info.PeakBW = atof(e) * 1.0e9;

    if (driver.info.ClockFrequency <= 0 && !driver.isSimulation()) {
        yprintf("Found an invalid clock frequency; skipping performance "
                "benchmarking.\nConsider overriding with the FREQ_OVERRIDE "
                "environment variable.\n");
    }

    driver.setSimulation(driver.isSimulation() || (driver.info.ClockFrequency == 0));

    if (driver.info.ClockFrequency == 0 && driver.isSimulation() && driver.Hardware == HW::Xe3p)
        driver.info.ClockFrequency = 1.6e9;

    driver.info.LLCSize = LLCUpperBound;
    if (LLCUpperBound == 0) {
        auto MB = 1024 * 1024;
        if (driver.Hardware == HW::XeHPC)
            LLCUpperBound = 256 * MB;
        else
            LLCUpperBound = 32 * MB;
    }
}

#define HANDLE_MDAPI(x, loud) if (!handleMetrics(x, loud)) return false;

#ifdef TARGET_LEVEL_ZERO
void handle(ze_result_t result)
{
    if (result != ZE_RESULT_SUCCESS) {
        std::stringstream ss;
        ss << "Level Zero error: " << std::hex << uint32_t(result);
        throw std::runtime_error(ss.str());
    }
}

bool handleMetrics(ze_result_t result, bool loud)
{
    if (result != ZE_RESULT_SUCCESS) {
        if (loud)
            yprintf("Level Zero error 0x%x occurred when initializing metrics\n", (uint32_t) result);
        return false;
    }
    return true;
}

bool rtImmediateExecution()
{
    return ImmediateCommandLists;
}

rt_list rtCreateList(const DriverRTArgs &args)
{
    if (ImmediateCommandLists)
        return args.immediateList;

    ze_command_list_handle_t list;
    ze_command_list_desc_t desc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        nullptr,
        CmdQueueOrdinal,
        0
    };

    handle(zeCommandListCreate(args.context, args.device, &desc, &list));
    return list;
}

void rtCreateEvents(DriverRTArgs &args, uint32_t count)
{
    auto &storage = args.storage;
    ze_event_pool_desc_t poolDesc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        0,
        count
    };
    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        0,
        0,
        0
    };
    if (TimestampEvents)
        poolDesc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    handle(zeEventPoolCreate(args.context, &poolDesc, 1, &args.device, &storage.eventPool));
    storage.curEvents.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        eventDesc.index = i;
        handle(zeEventCreate(storage.eventPool, &eventDesc, &storage.curEvents[i]));
    }
}

void zeWrapCommandListAppendLaunchKernel(rtKernelCtx& ctx, DriverPhase phase, const char* kernelName,
                                         rt_list list, rt_kernel kernel, int dims, size_t *global, size_t *local,
                                         rt_event eventOut = nullptr, uint32_t nEventsIn = 0, rt_event *eventsIn = nullptr)
{
    ze_group_count_t groups = {1, 1, 1};

    if (dims > 0) groups.groupCountX = uint32_t(global[0] / local[0]);
    if (dims > 1) groups.groupCountY = uint32_t(global[1] / local[1]);
    if (dims > 2) groups.groupCountZ = uint32_t(global[2] / local[2]);

    uint32_t wgX = (dims > 0) ? uint32_t(local[0]) : 1;
    uint32_t wgY = (dims > 1) ? uint32_t(local[1]) : 1;
    uint32_t wgZ = (dims > 2) ? uint32_t(local[2]) : 1;

    size_t effectiveLWS[] = {wgX, wgY, wgZ};
    rtLogDispatch(ctx, kernelName, phase, dims, global, effectiveLWS);

    handle(zeKernelSetGroupSize(kernel, wgX, wgY, wgZ));
    handle(zeCommandListAppendLaunchKernel(list, kernel, &groups, eventOut, nEventsIn, eventsIn));
}

double rtProfileDuration(const DriverRTArgs &args, const uint64_t *timestamps)
{
    const uint64_t timestampMask = ~(lshift(~uint64_t(0), args.timestampValidBits));

    uint64_t tsDiff = (timestamps[1] - timestamps[0]) & timestampMask;

    if (args.timestampPropertiesValid)
        return double(tsDiff) * 1e-9 * args.timerResolution;
    else
        return double(tsDiff) / args.timerResolution;
}

double rtProfileDuration(const DriverRTArgs &args, const std::vector<ze_event_handle_t> &events)
{
    uint64_t aggregated[2] = {0, 0};
    uint64_t &start = aggregated[0];
    uint64_t &end = aggregated[1];
    bool first = true;

    for (auto &e : events) {
        if (!e) continue;
        ze_kernel_timestamp_result_t ts;
        auto status = zeEventQueryKernelTimestamp(e, &ts);
        if (status == ZE_RESULT_NOT_READY) continue;
        handle(status);
        if (first) {
            start = ts.context.kernelStart;
            end = ts.context.kernelEnd;
            first = false;
        } else {
            if (int64_t(start - ts.context.kernelStart) > 0)
                start = ts.context.kernelStart;
            if (int64_t(ts.context.kernelEnd - end) > 0)
                end = ts.context.kernelEnd;
        }
    }

    return rtProfileDuration(args, aggregated);
}

void rtAllocTimestamps(DriverRTArgs &args)
{
    auto &storage = args.storage;
    if (TimestampEvents) return;

    ze_device_mem_alloc_desc_t dmemDesc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0
    };
    ze_host_mem_alloc_desc_t hmemDesc = {
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0
    };
    handle(zeMemAllocShared(args.context, &dmemDesc, &hmemDesc, sizeof(uint64_t) * 2, 64, args.device, (void **) &storage.timestamps));
}

void rtStartTimestamps(const DriverRT &driver, rt_list list)
{
    const auto &args = driver.rtArgs;
    auto &storage = args.storage;
    if (driver.getEnableMetrics())
        handle(zetCommandListAppendMetricQueryBegin(list, args.query));
    if (!TimestampEvents)
        handle(zeCommandListAppendWriteGlobalTimestamp(list, storage.timestamps, nullptr, 0, nullptr));
}

void rtEndTimestamps(const DriverRT &driver, rt_list list, int nevents, rt_event *events)
{
    const auto &args = driver.rtArgs;
    auto &storage = args.storage;
    if (!TimestampEvents)
        handle(zeCommandListAppendWriteGlobalTimestamp(list, storage.timestamps + 1, nullptr, nevents, events));
    if (driver.getEnableMetrics())
        handle(zetCommandListAppendMetricQueryEnd(list, args.query, nullptr, 0, nullptr));
}

void rtMakeTimestampsDeviceResident(const DriverRTArgs &args)
{
    const auto &storage = args.storage;
    if (!TimestampEvents)
        handle(zeContextMakeMemoryResident(args.context, args.device, storage.timestamps, sizeof(uint64_t) * 2));
}

template <typename To>
To rtConvert(zet_typed_value_t data)
{
    switch (data.type) {
        case ZET_VALUE_TYPE_BOOL8:
        case ZET_VALUE_TYPE_UINT32:  return To(data.value.ui32);
        case ZET_VALUE_TYPE_UINT64:  return To(data.value.ui64);
        case ZET_VALUE_TYPE_FLOAT32: return To(data.value.fp32);
        case ZET_VALUE_TYPE_FLOAT64: return To(data.value.fp64);
        default: stub();
    }
    return 0.;
}

void rtCollectMetrics(const DriverRTArgs &args, TestResult &result)
{
    size_t rawDataSize = 0;
    std::vector<uint8_t> rawData;
    uint32_t nvalues = 0;
    uint32_t nsets = 0;
    std::vector<zet_typed_value_t> values;

    result.avgFreq = 0;
    for (auto &metric: result.metrics)
        metric = 0.;

    if (zetMetricQueryGetData(args.query, &rawDataSize, nullptr) != ZE_RESULT_SUCCESS) return;
    rawData.resize(rawDataSize);
    if (zetMetricQueryGetData(args.query, &rawDataSize, rawData.data()) != ZE_RESULT_SUCCESS) return;

    auto multiOK = zetMetricGroupCalculateMultipleMetricValuesExp(args.metricGroup, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, rawDataSize, rawData.data(), &nsets, &nvalues, nullptr, nullptr);
    if (multiOK == ZE_RESULT_SUCCESS) {
        values.resize(nvalues);
        std::vector<uint32_t> counts(nsets);
        if (zetMetricGroupCalculateMultipleMetricValuesExp(args.metricGroup, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, rawDataSize, rawData.data(), &nsets, &nvalues, counts.data(), values.data()) != ZE_RESULT_SUCCESS) return;

        double minFreq = std::numeric_limits<double>::infinity();
        for (uint32_t iset = 0, index0 = 0; iset < nsets; index0 += counts[iset++]) {
            if (index0 + FreqMetricIndex < nvalues) {
                auto freq = rtConvert<double>(values[index0 + FreqMetricIndex]) * 1e6;
                result.tileFreqs.push_back(freq);
                minFreq = std::min(minFreq, freq);
            }
            for (int m = 0; m < MetricCount; m++)
                for (auto index: AdditionalMetricIndices[m])
                    if (index0 + index < nvalues)
                        result.metrics[m] += rtConvert<double>(values[index0 + index]);
        }

        if (!result.tileFreqs.empty())
            result.avgFreq = minFreq;
    } else {
        nvalues = 0;
        if (zetMetricGroupCalculateMetricValues(args.metricGroup, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, rawDataSize, rawData.data(), &nvalues, nullptr) != ZE_RESULT_SUCCESS) return;
        values.resize(nvalues);
        if (zetMetricGroupCalculateMetricValues(args.metricGroup, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, rawDataSize, rawData.data(), &nvalues, values.data()) != ZE_RESULT_SUCCESS) return;

        if (uint32_t(FreqMetricIndex) < nvalues)
            result.avgFreq = rtConvert<double>(values[FreqMetricIndex]) * 1e6;
        for (int m = 0; m < MetricCount; m++)
            for (auto index: AdditionalMetricIndices[m])
                if (uint32_t(index) < nvalues)
                    result.metrics[m] = rtConvert<double>(values[index]);
    }
}

void rtRunList(DriverRT &driver, TestResult &result)
{
    auto &args = driver.rtArgs;
    auto &storage = args.storage;
    double start = 0 , end = 0;

    if (ImmediateCommandLists) {
        handle(zeCommandListAppendBarrier(storage.list, args.completionEvent, 0, nullptr));
        RECORD(GPUWaitTime, handle(zeEventHostSynchronize(args.completionEvent, UINT64_MAX)));
        handle(zeEventHostReset(args.completionEvent));
    } else {
        rtMakeTimestampsDeviceResident(args);
        start = dsecnd();
        handle(zeCommandQueueExecuteCommandLists(args.queue, 1, &storage.list, args.fence));
        RECORD(GPUWaitTime, handle(zeFenceHostSynchronize(args.fence, UINT64_MAX)));
        end = dsecnd();
        result.hostTime = end - start;
        handle(zeFenceReset(args.fence));
    }

    if (TimestampEvents) {
        result.deviceTime = rtProfileDuration(args, storage.curEvents);
        for (auto &e: storage.curEvents)
            handle(zeEventHostReset(e));
    } else
        result.deviceTime = rtProfileDuration(args, storage.timestamps);

    result.avgFreq = 0.;
    for (auto &metric: result.metrics)
        metric = 0.;

    if (driver.getEnableMetrics()) {
        rtCollectMetrics(args, result);
        handle(zetMetricQueryReset(args.query));
    }
}

void clearLevelZeroData(DriverRTArgs &args)
{
    auto &storage = args.storage;
    auto context = args.context;
    if (storage.list && !ImmediateCommandLists)
        handle(zeCommandListDestroy(storage.list));
    for (auto &event: storage.curEvents)
        handle(zeEventDestroy(event));
    if (storage.eventPool)
        handle(zeEventPoolDestroy(storage.eventPool));
    if (storage.timestamps)
        handle(zeMemFree(context, storage.timestamps));

    storage.list = nullptr;
    storage.curEvents.clear();
    storage.eventPool = nullptr;
    storage.timestamps = nullptr;
}

void rtInit(DriverRT &driver)
{
    auto &args = driver.rtArgs;

    args.device = nullptr;
    args.context = nullptr;
    args.queue = nullptr;
    args.program = nullptr;
    args.kernel = nullptr;
    args.immediateList = nullptr;
    args.copyList = nullptr;
    args.copyQueue = nullptr;
    args.fence = nullptr;
    args.completionEvent = nullptr;
    args.completionEventPool = nullptr;
    args.queryPool = nullptr;
    args.query = nullptr;
    args.metricGroup = nullptr;
    args.memDevice = nullptr;

    ze_driver_handle_t driver_handle;
    ze_eu_count_ext_t eprop;
    ze_device_properties_t deviceProperties = {};

    // Enable metrics environment var for the purpose of zeInit
    // But we don't define a metric set unless the caller calls
    // initMetrics, so this should have minimal overhead if metrics aren't used.
    if (!driver.isSimulation()) {   // metrics not supported in simulation
#ifdef _WIN32
        _putenv("ZET_ENABLE_METRICS=1");
#else
        (void) setenv("ZET_ENABLE_METRICS", "1", false);
#endif
    }

    handle(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driverCount = 0;
    handle(zeDriverGet(&driverCount, nullptr));

    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    std::vector<ze_device_handle_t> allDevices;

    handle(zeDriverGet(&driverCount, allDrivers.data()));

    // Allow manual selection of platform and device #
    int driverIdx = -1, devIdx = -1;
    if (getenv("DRIVER")) driverIdx = atoi(getenv("DRIVER"));
    if (getenv("DEVICE"))    devIdx = atoi(getenv("DEVICE"));

    for (uint32_t i = 0; i < driverCount; i++) {
        if (driverIdx > 0 && driverIdx != i) continue;

        uint32_t deviceCount = 0;
        handle(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));

        allDevices.resize(deviceCount);
        handle(zeDeviceGet(allDrivers[i], &deviceCount, allDevices.data()));

        for (uint32_t j = 0; j < deviceCount; j++) {
            if (devIdx > 0 && devIdx != j) continue;

            auto dev = allDevices[j];

            uint32_t subdeviceCount = 0;
            if (zeDeviceGetSubDevices(dev, &subdeviceCount, nullptr) == ZE_RESULT_SUCCESS) {
                Tiles = std::max<uint32_t>(1u, subdeviceCount);

                std::vector<ze_device_handle_t> subdevices(subdeviceCount);
                handle(zeDeviceGetSubDevices(dev, &subdeviceCount, subdevices.data()));

                if (ExecutionSubdevice >= 0) {
                    if (uint32_t(ExecutionSubdevice) < subdeviceCount) {
                        dev = subdevices[ExecutionSubdevice];
                        Tiles = 1;
                    } else
                        continue;
                }

                if (MemorySubdevice >= 0) {
                    if (uint32_t(MemorySubdevice) < subdeviceCount)
                        args.memDevice = subdevices[MemorySubdevice];
                    else
                        continue;
                }
            } else if (ExecutionSubdevice >= 0 || MemorySubdevice >= 0)
                continue;


            deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            deviceProperties.pNext = &eprop;
            eprop.stype = ZE_STRUCTURE_TYPE_EU_COUNT_EXT;
            eprop.pNext = nullptr;

            auto result = zeDeviceGetProperties(dev, &deviceProperties);
            if (result != ZE_RESULT_SUCCESS) {
                deviceProperties.pNext = nullptr;
                eprop.numTotalEUs = 0;
                handle(zeDeviceGetProperties(dev, &deviceProperties));
            }

            if (deviceProperties.type == ZE_DEVICE_TYPE_GPU) {
                driver_handle = allDrivers[i];
                args.device = dev;
                goto success;
            }
        }
    }

    if (!args.device) {
        fprintf(stderr, "No GPUs were found.");
        exit(1);
    }

success:
    if (!args.memDevice)
        args.memDevice = args.device;

    if (driver.Hardware != HW::Unknown)
        deviceProperties.numEUsPerSubslice = eusPerSS(driver.Hardware);
    auto computedEUs = deviceProperties.numSlices * deviceProperties.numSubslicesPerSlice * deviceProperties.numEUsPerSubslice;
    driver.info.EUs = eprop.numTotalEUs;
    if (driver.info.EUs == 0)
        driver.info.EUs = computedEUs;
    else if (driver.info.EUs > 1.5 * computedEUs) {
        if (deviceProperties.numSlices > 1)
            driver.info.EUs /= deviceProperties.numSlices;  /* Workaround */
        else
            driver.info.EUs >>= 1;                          /* Workaround for older drivers reporting 2x too many EUs */
    }

    driver.info.ClockFrequency = deviceProperties.coreClockRate * 1e6;

    deviceProperties.name[ZE_MAX_DEVICE_NAME - 1] = '\0';

    args.timestampPropertiesValid = (deviceProperties.stype == ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES);
    args.timestampValidBits = deviceProperties.timestampValidBits;
    args.timerResolution = deviceProperties.timerResolution;

    std::vector<ze_device_cache_properties_t> cprops;
    uint32_t ccount = 0;
    handle(zeDeviceGetCacheProperties(args.device, &ccount, nullptr));

    cprops.resize(ccount);
    for (auto &cprop: cprops) {
        cprop.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
        cprop.pNext = nullptr;
    }

    handle(zeDeviceGetCacheProperties(args.device, &ccount, cprops.data()));

    for (auto &cprop: cprops)
        LLCUpperBound = std::max(LLCUpperBound, cprop.cacheSize);

    commonDeviceSetup(driver);

    if (!driver.Quiet) {
        printf("Using Level Zero device %s (%s, %d tile%s, %dx%dx%d = %d EUs, %d MHz).\n",
               deviceProperties.name, hwName(driver.Hardware), Tiles, (Tiles > 1) ? "s" : "",
               deviceProperties.numSlices,
               deviceProperties.numSubslicesPerSlice, deviceProperties.numEUsPerSubslice,
               driver.info.EUs, int(driver.info.ClockFrequency * 1e-6));
    }

    uint32_t cgpCount = 0;
    handle(zeDeviceGetCommandQueueGroupProperties(args.device, &cgpCount, nullptr));

    std::vector<ze_command_queue_group_properties_t> cgProps(cgpCount);
    for (auto &cgProp: cgProps) {
        cgProp.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        cgProp.pNext = nullptr;
    }

    handle(zeDeviceGetCommandQueueGroupProperties(args.device, &cgpCount, cgProps.data()));
    for (uint32_t i = 0; i < cgpCount; i++) {
        if (cgProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
            CmdQueueOrdinal = i;
        break;
    }
    for (uint32_t i = 0; i < cgpCount; i++) {
        if (cgProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)
            CmdQueueOrdinalCopy = i;
        break;
    }

    uint32_t dmCount = 0;
    handle(zeDeviceGetMemoryProperties(args.device, &dmCount, nullptr));

    DeviceMemOrdinals = dmCount;

    std::vector<ze_device_memory_properties_t> dmProps(dmCount);
    for (auto &dmProp: dmProps) {
        dmProp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        dmProp.pNext = nullptr;
    }

    handle(zeDeviceGetMemoryProperties(args.device, &dmCount, dmProps.data()));
    if (driver.Loud) for (uint32_t i = 0; i < dmCount; i++) {
        auto sz = double(dmProps[i].totalSize);
        const char *szStr = "b";
        if (sz >= 1024) {
            sz /= 1024;
            szStr = "kb";
        }
        if (sz >= 1024) {
            sz /= 1024;
            szStr = "MB";
        }
        if (sz >= 1024) {
            sz /= 1024;
            szStr = "GB";
        }
        if (sz >= 1024) {
            sz /= 1024;
            szStr = "TB";
        }
        printf(" - device memory %d (%s): %.1f%s", i, dmProps[i].name, sz, szStr);
        if (dmProps[i].maxClockRate > 0)
            printf(", %d MHz", int(dmProps[i].maxClockRate));
        printf("\n");
    }

    ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    handle(zeContextCreate(driver_handle, &ctxDesc, &args.context));
//    handle(zeContextCreateEx(driver_handle, &ctxDesc, 1, &args.device, &args.context));

    ze_command_queue_desc_t queueDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        CmdQueueOrdinal,
        0,
        0,  // not explicit
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH
    };
    if (ImmediateCommandLists)
        handle(zeCommandListCreateImmediate(args.context, args.device, &queueDesc, &args.immediateList));
    else
        handle(zeCommandQueueCreate(args.context, args.device, &queueDesc, &args.queue));

    ze_command_queue_desc_t copyQueueDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        CmdQueueOrdinalCopy,
        0,
        0,
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    handle(zeCommandQueueCreate(args.context, args.device, &copyQueueDesc, &args.copyQueue));

    ze_command_list_desc_t copyListDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        nullptr,
        CmdQueueOrdinalCopy,
        0
    };
    handle(zeCommandListCreate(args.context, args.device, &copyListDesc, &args.copyList));

    ze_fence_desc_t fenceDesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    if (!ImmediateCommandLists)
        handle(zeFenceCreate(args.queue, &fenceDesc, &args.fence));

    ze_event_pool_desc_t poolDesc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
        1
    };
    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        0,
        0,
        0
    };
    if (ImmediateCommandLists) {
        handle(zeEventPoolCreate(args.context, &poolDesc, 1, &args.device, &args.completionEventPool));
        handle(zeEventCreate(args.completionEventPool, &eventDesc, &args.completionEvent));
    }

    auto makeKernel = [&](const char *cl, const char *name) {
        ze_module_desc_t fmoduleDesc = {
            ZE_STRUCTURE_TYPE_MODULE_DESC,
            nullptr,
            (ze_module_format_t) 3,     // ZE_MODULE_FORMAT_OCLC
            strlen(cl) + 1,
            (const uint8_t *) cl,
            "",
            nullptr
        };

        ze_kernel_desc_t fkernelDesc = {
            ZE_STRUCTURE_TYPE_KERNEL_DESC,
            nullptr,
            0,
            name
        };

        ze_module_handle_t program;
        ze_kernel_handle_t kernel;
        handle(zeModuleCreate(args.context, args.device, &fmoduleDesc, &program, nullptr));
        handle(zeKernelCreate(program, &fkernelDesc, &kernel));

        return std::make_tuple(program, kernel);
    };

    bool anyColdMat = driver.ColdA || driver.ColdB || driver.ColdC;
    if (anyColdMat) {
        auto fcl   = (driver.Hardware >= HW::XeHPG) ? lsc_flush_cl   : flush_cl;
        auto fname = (driver.Hardware >= HW::XeHPG) ? lsc_flush_name : flush_name;

        std::tie(args.flushProgram, args.flushKernel) = makeKernel(fcl, fname);
        handle(zeKernelSetGroupSize(args.flushKernel, 1, 1, 1));
    } else {
        args.flushProgram = nullptr;
        args.flushKernel = nullptr;
    }

    if (driver.ColdICache) {
        std::tie(args.flushICacheProgram, args.flushICacheKernel) = makeKernel(flush_icache_cl, flush_icache_name);
        handle(zeKernelSetGroupSize(args.flushICacheKernel, 1, 1, 1));
    } else {
        args.flushICacheProgram = nullptr;
        args.flushICacheKernel = nullptr;
    }

    if (FillCache) {
        std::tie(args.fillCacheProgram, args.fillCacheKernel) = makeKernel(cache_fill_cl, cache_fill_name);
        ze_device_mem_alloc_desc_t memDesc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            nullptr,
            ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
            DeviceMemOrdinal
        };
        handle(zeMemAllocDevice(args.context, &memDesc, LLCUpperBound, 4096, args.memDevice, &args.fillBuffer));
    } else {
        args.fillCacheProgram = nullptr;
        args.fillCacheKernel = nullptr;
    }
}

bool mdapiInit(DriverRT &driver)
{
    auto &args = driver.rtArgs;

#ifndef _WIN32
    if (!dlopen("libigdmd.so.1", RTLD_LAZY) && !dlopen("libmd.so.1", RTLD_LAZY)) {
        if (driver.Loud) yprintf("Could not find libigdmd.so.1 or libmd.so.1 -- unable to monitor frequency\n");
        return false;
    }
#endif

    for (auto &indices: AdditionalMetricIndices)
        indices.clear();
    FreqMetricIndex = -1;

    uint32_t nMetricGroups = 0;
    std::vector<zet_metric_group_handle_t> metricGroups;
    zet_metric_group_properties_t mgProps;

    HANDLE_MDAPI(zetMetricGroupGet(args.device, &nMetricGroups, nullptr), driver.Loud);
    metricGroups.resize(nMetricGroups);
    HANDLE_MDAPI(zetMetricGroupGet(args.device, &nMetricGroups, metricGroups.data()), driver.Loud);

    args.metricGroup = nullptr;
    for (auto &mg: metricGroups) {
        mgProps.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        mgProps.pNext = nullptr;
        HANDLE_MDAPI(zetMetricGroupGetProperties(mg, &mgProps), driver.Loud);

        if (!(mgProps.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED))
            continue;
        if (!strcmp(mgProps.name, MetricSetName)) {
            args.metricGroup = mg;
            break;
        }
    }

    if (!args.metricGroup) {
        yprintf("Could not find metric group %s. Effective efficiencies cannot be calculated.\n", MetricSetName);
        return false;
    }

    HANDLE_MDAPI(zetContextActivateMetricGroups(args.context, args.device, 1, &args.metricGroup), driver.Loud);

    uint32_t nmetrics = 0;
    std::vector<zet_metric_handle_t> metrics;
    zet_metric_properties_t mprops;

    HANDLE_MDAPI(zetMetricGet(args.metricGroup, &nmetrics, nullptr), driver.Loud);
    metrics.resize(nmetrics);
    HANDLE_MDAPI(zetMetricGet(args.metricGroup, &nmetrics, metrics.data()), driver.Loud);

    for (size_t i = 0; i < nmetrics; i++) {
        mprops.stype = ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
        mprops.pNext = nullptr;
        HANDLE_MDAPI(zetMetricGetProperties(metrics[i], &mprops), driver.Loud);

        std::string mname(mprops.name);

        if (mname == "AvgGpuCoreFrequencyMHz")
            FreqMetricIndex = int(i);
        else if (auto search = MetricNames.find(mname); search != MetricNames.end())
            AdditionalMetricIndices[search->second].push_back(int(i));
    }

    if (FreqMetricIndex < 0) {
        yprintf("Could not find AvgGpuCoreFrequencyMHz in group %s\n", MetricSetName);
        return false;
    }

    zet_metric_query_pool_desc_t qpDesc;
    qpDesc.stype = ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
    qpDesc.pNext = nullptr;
    qpDesc.type = ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE;
    qpDesc.count = 1;

    HANDLE_MDAPI(zetMetricQueryPoolCreate(args.context, args.device, args.metricGroup, &qpDesc, &args.queryPool), driver.Loud);

    HANDLE_MDAPI(zetMetricQueryCreate(args.queryPool, 0, &args.query), driver.Loud);
    return true;
}

bool rtInitMetrics(DriverRT &driver)
{
    return mdapiInit(driver);
}

bool rtInitKernel(DriverRT &driver, const KernelInfo &info)
{
    auto &args = driver.rtArgs;
    auto &binary = args.binary;
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_NATIVE,
        binary.size(),
        binary.data(),
        "",
        nullptr
    };

    auto moduleStatus = zeModuleCreate(args.context, args.device, &moduleDesc, &args.program, nullptr);
    if (moduleStatus) {
        if (!driver.Quiet) fprintf(stderr, "Module creation failed with error %x.\n", moduleStatus);
        return false;
    }

    ze_kernel_desc_t kernelDesc = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        nullptr,
        0,
        info.kernelName()
    };

    auto kernelStatus = zeKernelCreate(args.program, &kernelDesc, &args.kernel);
    if (kernelStatus) {
        kernelDesc.pKernelName = "__sycl_kernel_gemmstone_kernel";
        auto kernelStatus = zeKernelCreate(args.program, &kernelDesc, &args.kernel);
        if (kernelStatus) {
            if (!driver.Quiet) fprintf(stderr, "Kernel creation failed with error %x.\n", kernelStatus);
            return false;
        }
    }

    return true;
}

void rtFinishKernel(DriverRTArgs &args)
{
    if (args.kernel)
        handle(zeKernelDestroy(args.kernel));
    if (args.program)
        handle(zeModuleDestroy(args.program));
    args.kernel = nullptr;
    args.program = nullptr;
}

void rtFinish(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    if (driver.getEnableMetrics())
        handleMetrics(zetContextActivateMetricGroups(args.context, args.device, 0, nullptr), driver.Loud);
    if (args.query)
        handle(zetMetricQueryDestroy(args.query));
    if (args.queryPool)
        handle(zetMetricQueryPoolDestroy(args.queryPool));
    for (auto kernel: {args.flushKernel, args.flushICacheKernel, args.fillCacheKernel})
        if (kernel)
            handle(zeKernelDestroy(kernel));
    for (auto program: {args.flushProgram, args.flushICacheProgram, args.fillCacheProgram})
        if (program)
            handle(zeModuleDestroy(program));
    if (args.fence)
        handle(zeFenceDestroy(args.fence));
    if (args.completionEvent)
        handle(zeEventDestroy(args.completionEvent));
    if (args.completionEventPool)
        handle(zeEventPoolDestroy(args.completionEventPool));
    if (args.copyList)
        handle(zeCommandListDestroy(args.copyList));
    if (args.copyQueue)
        handle(zeCommandQueueDestroy(args.copyQueue));
    if (args.queue)
        handle(zeCommandQueueDestroy(args.queue));
    if (args.immediateList)
        handle(zeCommandListDestroy(args.immediateList));
    if (args.context)
        handle(zeContextDestroy(args.context));

    args.fence = nullptr;
    args.completionEvent = nullptr;
    args.completionEventPool = nullptr;
    args.copyList = nullptr;
    args.copyQueue = nullptr;
    args.queue = nullptr;
    args.context = nullptr;
    args.queryPool = nullptr;
    args.immediateList = nullptr;
    args.query = nullptr;
    args.queryPool = nullptr;
    args.flushKernel = args.flushICacheKernel = args.fillCacheKernel = nullptr;
    args.flushProgram = args.flushICacheProgram = args.fillCacheProgram = nullptr;
}

template <typename T>
static inline void rtSetArg(rtKernelCtx& ctx, int argnum, const T &arg, const char* name)
{
    ctx.argInfo.update(argnum, name, arg);
    handle(zeKernelSetArgumentValue(ctx.kernel, argnum, sizeof(arg), &arg));
}

static inline void rtSetArg(rtKernelCtx& ctx, int argnum, size_t bytes, const void *arg)
{
    handle(zeKernelSetArgumentValue(ctx.kernel, argnum, bytes, arg));
}

static inline void rtSetLocalArg(rtKernelCtx& ctx, int argnum, size_t bytes, const char* name)
{
    ctx.argInfo.update(argnum, name, "<local:" + std::to_string(bytes) + ">");
    handle(zeKernelSetArgumentValue(ctx.kernel, argnum, bytes, nullptr));
}

bool rtIsUSM(const DriverRTArgs &args, const void *ptr)
{
    bool isUSM = false;

    ze_memory_allocation_properties_t memProps = {ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, nullptr};
    if (ZE_RESULT_SUCCESS == zeMemGetAllocProperties(args.context, ptr, &memProps, nullptr))
        isUSM = (memProps.type != ZE_MEMORY_TYPE_UNKNOWN);

    return isUSM;
}

struct VMapping {
    struct Entry {
        ze_physical_mem_handle_t pmem;
        size_t psize;
        size_t voff;
    };
    void *vptr;
    size_t vsize;
    std::vector<Entry> translations;
};

std::unordered_map<rt_buffer, VMapping> vmappings;

void rtMap(DriverRTArgs &args, const VMapping &mapping)
{
    for (auto &e: mapping.translations)
        handle(zeVirtualMemMap(args.context, (uint8_t *) mapping.vptr + e.voff, e.psize, e.pmem, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
}

void rtUnmap(DriverRTArgs &args, const VMapping &mapping)
{
    handle(zeVirtualMemUnmap(args.context, mapping.vptr, mapping.vsize));
}

void rtRemapAll(DriverRTArgs &args)
{
    for (auto &it: vmappings) rtUnmap(args, it.second);
    for (auto &it: vmappings) rtMap(args, it.second);
}

rt_buffer rtAllocBuffer(DriverRT &driver, BufferFlags flags, size_t size)
{
    auto &args = driver.rtArgs;
    ze_device_mem_alloc_desc_t memDesc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        nullptr,
        ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,
        DeviceMemOrdinal
    };

    void *mem;
    uint64_t _4GB = 1ull << 32;

    if ((flags & BUF_CROSS_4GB) && (size <= _4GB)) {
        size_t pageSize;
        uint8_t *base;
        handle(zeVirtualMemQueryPageSize(args.context, args.device, size, &pageSize));
        auto vsize = align_up(size, pageSize) + _4GB;
        handle(zeVirtualMemReserve(args.context, nullptr, vsize, (void **) &base));
        auto off = 0x1'0000'0000ull - uint32_t(uintptr_t(base)) - align_up(size >> 1, 256);
        auto poff0 = align_down(off, pageSize);
        auto poff1 = align_up(off + size, pageSize);

        ze_physical_mem_desc_t pmemDesc = {ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC};
        pmemDesc.flags = 0x1;   /* ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE */
        pmemDesc.size = poff1 - poff0;
        ze_physical_mem_handle_t pmem;
        handle(zePhysicalMemCreate(args.context, args.device, &pmemDesc, &pmem));

        VMapping mapping{base, vsize, {{pmem, poff1 - poff0, poff0}}};
        rtMap(args, mapping);

        mem = base + off;
        vmappings[mem] = std::move(mapping);
    } else if (PageSize > 0) {
        // Check if we can allocate physical memory with our page size in one piece;
        //   otherwise allocate in multiple small pieces.
        size_t xsize = align_up(size, PageSize);
        size_t chunk = xsize;

        ze_physical_mem_desc_t pmemDesc = {ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC};
        pmemDesc.flags = 0x1;   /* ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE */

        for (;;) {
            size_t psize;
            handle(zeVirtualMemQueryPageSize(args.context, args.device, chunk, &psize));
            if (psize <= PageSize)
                break;
            else if (chunk <= PageSize)
                throw std::runtime_error("Unable to allocate memory with the requested page size");
            chunk = std::min(psize, chunk) >> 1;
        }

        uint8_t *base;
        handle(zeVirtualMemReserve(args.context, nullptr, xsize, (void **) &base));

        VMapping mapping{base, xsize};
        for (size_t off = 0; off < xsize; ) {
            ze_physical_mem_handle_t pmem;
            pmemDesc.size = std::min(xsize - off, chunk);
            handle(zePhysicalMemCreate(args.context, args.device, &pmemDesc, &pmem));
            mapping.translations.push_back({pmem, pmemDesc.size, off});
            off += pmemDesc.size;
        }

        rtMap(args, mapping);
        mem = base;
        vmappings[mem] = std::move(mapping);
    } else if (driver.ColdTLB) {
        size_t psize;
        handle(zeVirtualMemQueryPageSize(args.context, args.device, size, &psize));
        auto xsize = align_up(size, psize);
        handle(zeVirtualMemReserve(args.context, nullptr, xsize, &mem));

        ze_physical_mem_desc_t pmemDesc = {ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC};
        ze_physical_mem_handle_t pmem;
        pmemDesc.flags = 0x1;
        pmemDesc.size = xsize;
        handle(zePhysicalMemCreate(args.context, args.device, &pmemDesc, &pmem));

        VMapping mapping{mem, xsize, {{pmem, xsize, 0}}};
        rtMap(args, mapping);
        vmappings[mem] = std::move(mapping);
    } else
        handle(zeMemAllocDevice(args.context, &memDesc, align_up(size, AllocSizeAlign), std::max<size_t>(AllocSizeAlign, 4096), args.memDevice, &mem));

    return mem;
}

rt_buffer rtCreateBackendBuffer(DriverRT &driver, BufferFlags flags, rt_buffer buffer,
                                size_t size, void *host_ptr, size_t offset)
{
    auto &args = driver.rtArgs;
    auto mem = buffer ? buffer : rtAllocBuffer(driver, flags, size);

    if (flags & BUF_UNINIT)
        host_ptr = nullptr;

    if (host_ptr) {
        bool isUSM = rtIsUSM(args, host_ptr);
        void *hostMem = host_ptr;

        if (!isUSM) {
            ze_host_mem_alloc_desc_t hmemDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
            handle(zeMemAllocHost(args.context, &hmemDesc, size - offset, 4096, &hostMem));
            memcpy(hostMem, host_ptr, size - offset);
        }

        handle(zeCommandListAppendMemoryCopy(args.copyList, ((char *) mem) + offset, hostMem, size - offset, nullptr, 0, nullptr));
        handle(zeCommandListClose(args.copyList));
        handle(zeCommandQueueExecuteCommandLists(args.copyQueue, 1, &args.copyList, nullptr));
        handle(zeCommandQueueSynchronize(args.copyQueue, UINT64_MAX));
        handle(zeCommandListReset(args.copyList));

        if (!isUSM)
            handle(zeMemFree(args.context, hostMem));
    }

    return mem;
}

bool rtBindsHostPtr(const DriverRT &, BufferFlags, const void *)
{
    return false;
}

void rtZeroBuffer(DriverRTArgs &args, rt_buffer buffer, size_t size)
{
    uint8_t zero = 0;
    handle(zeCommandListAppendMemoryFill(args.copyList, buffer, &zero, sizeof(zero), size, nullptr, 0, nullptr));
    handle(zeCommandListClose(args.copyList));
    handle(zeCommandQueueExecuteCommandLists(args.copyQueue, 1, &args.copyList, nullptr));
    handle(zeCommandQueueSynchronize(args.copyQueue, UINT64_MAX));
    handle(zeCommandListReset(args.copyList));
}

void rtReadBuffer(DriverRTArgs &args, rt_buffer buffer, size_t offset, size_t bytes, void *dst)
{
    bool isUSM = rtIsUSM(args, dst);
    void *hostMem = dst;

    if (!isUSM) {
        ze_host_mem_alloc_desc_t hmemDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
        handle(zeMemAllocHost(args.context, &hmemDesc, bytes, 4096, &hostMem));
    }

    buffer = ((char *) buffer) + offset;
    handle(zeCommandListAppendMemoryCopy(args.copyList, hostMem, buffer, bytes, nullptr, 0, nullptr));
    handle(zeCommandListClose(args.copyList));
    handle(zeCommandQueueExecuteCommandLists(args.copyQueue, 1, &args.copyList, nullptr));
    handle(zeCommandQueueSynchronize(args.copyQueue, UINT64_MAX));
    handle(zeCommandListReset(args.copyList));

    if (!isUSM) {
        memcpy(dst, hostMem, bytes);
        handle(zeMemFree(args.context, hostMem));
    }
}

void rtFreeBuffer(rt_context context, rt_buffer buf)
{
    if (auto it = vmappings.find(buf); it != vmappings.end()) {
        auto &mapping = it->second;
        handle(zeVirtualMemUnmap(context, mapping.vptr, mapping.vsize));
        handle(zeVirtualMemFree(context, mapping.vptr, mapping.vsize));
        for (auto &e: mapping.translations)
            handle(zePhysicalMemDestroy(context, e.pmem));
        vmappings.erase(it);
    } else {
        void *base;
        handle(zeMemGetAddressRange(context, buf, &base, nullptr));
        handle(zeMemFree(context, base));
    }
}

rt_buffer rtOffsetBuffer(rt_buffer buffer, ptrdiff_t offset)
{
    return ((char *) buffer + offset);
}
#endif

#ifdef TARGET_OPENCL
void handle(cl_int result)
{
    if (result < 0) {
        std::string msg = "OpenCL error: ";
        msg += std::to_string(int(result));
        throw std::runtime_error(msg);
    }
}

bool rtImmediateExecution()
{
    return true;
}

template <typename T>
static inline void rtSetArg(rtKernelCtx& ctx, int argnum, const T &arg, const char* name)
{
    ctx.argInfo.update(argnum, name, arg);
    handle(clSetKernelArg(ctx.kernel, argnum, sizeof(T), &arg));
}

static inline void rtSetArg(rtKernelCtx& ctx, int argnum, size_t bytes, const void *arg)
{
    handle(clSetKernelArg(ctx.kernel, argnum, bytes, arg));
}

static inline void rtSetLocalArg(rtKernelCtx& ctx, int argnum, size_t bytes, const char* name)
{
    ctx.argInfo.update(argnum, name, "<local:" + std::to_string(bytes) + ">");
    handle(clSetKernelArg(ctx.kernel, argnum, bytes, nullptr));
}

void rtRemapAll(DriverRTArgs &args)
{
    throw std::runtime_error("OpenCL does not support virtual address remapping");
}

rt_buffer rtCreateBackendBuffer(DriverRT &driver, BufferFlags flags, rt_buffer buffer,
                                size_t size, void *host_ptr, size_t offset)
{
    auto &args = driver.rtArgs;
    cl_mem_flags clFlags = 0;

    if (flags & BUF_UNINIT)
        host_ptr = nullptr;

    if (flags & BUF_READ)
        clFlags |= CL_MEM_READ_ONLY;
    else if (flags & BUF_WRITE)
        clFlags |= CL_MEM_WRITE_ONLY;
    else if (flags & BUF_RW)
        clFlags |= CL_MEM_READ_WRITE;

    if (!host_ptr)
        /* noop */;
    else if (flags & BUF_COPY)
        clFlags |= CL_MEM_COPY_HOST_PTR;
    else if (driver.ZeroCopy)
        clFlags |= CL_MEM_USE_HOST_PTR;
    else
        clFlags |= CL_MEM_COPY_HOST_PTR;

    if (driver.Hardware >= HW::XeHP && AllowCompression)
        clFlags |= CL_MEM_UNCOMPRESSED_HINT_INTEL;

    if (buffer) {
        if (clFlags & CL_MEM_USE_HOST_PTR) stub();
        if (host_ptr)
            handle(clEnqueueWriteBuffer(args.queue, buffer, CL_TRUE, offset, size - offset, host_ptr, 0, nullptr, nullptr));
        return buffer;
    }

    cl_int err = CL_SUCCESS;
    cl_mem mem = nullptr;

    auto xsize = align_up(size, AllocSizeAlign);

    bool needCopy = false;
    if (host_ptr) {
        needCopy |= (offset > 0);
        if (xsize > size) {
            if (AllocSizeAlign > 4096)
                needCopy = true;
            else if (AllocSizeAlign > 1)
                needCopy |= ((uintptr_t) host_ptr & (AllocSizeAlign - 1)) != 0;
        }
    }

    if (!needCopy)
        mem = clCreateBuffer(args.context, clFlags, xsize, host_ptr, &err);
    else {
        if (clFlags & CL_MEM_USE_HOST_PTR) stub();
        clFlags &= ~CL_MEM_COPY_HOST_PTR;
        mem = clCreateBuffer(args.context, clFlags, xsize, nullptr, &err);
    }

    if (!mem || err) {
        fprintf(stderr, "OpenCL buffer creation failed with error %d.\n", (int) err);
        exit(1);
    }

    if (needCopy)
        handle(clEnqueueWriteBuffer(args.queue, mem, CL_TRUE, offset, size - offset, host_ptr, 0, nullptr, nullptr));

    return mem;
}

// Matches the CL_MEM_USE_HOST_PTR selection in rtCreateBackendBuffer.
bool rtBindsHostPtr(const DriverRT &driver, BufferFlags flags, const void *host_ptr)
{
    return host_ptr && !(flags & (BUF_UNINIT | BUF_COPY)) && driver.ZeroCopy;
}

void rtFreeBuffer(rt_context context, rt_buffer buffer)
{
    (void) context;
    handle(clReleaseMemObject(buffer));
}

void rtZeroBuffer(DriverRTArgs &args, rt_buffer buffer, size_t size)
{
    uint32_t zero = 0;
    handle(clEnqueueFillBuffer(args.queue, buffer, &zero, sizeof(zero), 0, size, 0, nullptr, nullptr));
    handle(clFinish(args.queue));
}

void rtReadBuffer(const DriverRTArgs &args, rt_buffer buffer, size_t offset, size_t bytes, void *dst)
{
    handle(clEnqueueReadBuffer(args.queue, buffer, CL_TRUE, offset, bytes, dst, 0, nullptr, nullptr));
}

rt_buffer rtOffsetBuffer(rt_buffer buffer, ptrdiff_t offset)
{
    throw std::runtime_error("Cannot use OpenCL buffers for this test.");
}

void clWrapEnqueueNDRangeKernel(rtKernelCtx& ctx, DriverPhase phase, const char* kernelName,
        cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
        const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
{
    rtLogDispatch(ctx, kernelName, phase, work_dim, global_work_size, local_work_size);
    if (InOrderQueue) {
        num_events_in_wait_list = 0;
        event_wait_list = nullptr;
    }
    handle(clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size,
        local_work_size, num_events_in_wait_list, event_wait_list, event));
}

void clWrapWaitForEvents(cl_uint nevents, cl_event *events) {
    auto err = clWaitForEvents(nevents, events);
    if (err < 0) {
        cl_int ev_err;
        for(size_t i = 0; i < nevents; i++) {
            handle(clGetEventInfo(events[i], CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(ev_err), &ev_err, NULL));
            handle(ev_err);
        }
    }
    handle(err);
};

double rtProfileDuration(const std::vector<cl_event> &events)
{
    cl_ulong start = UINT64_MAX, end = 0;

    if (events.empty())
        start = end = 0;
    else for (auto &e : events) {
        if (!e) continue;
        cl_ulong tstart, tend;
        handle(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(tstart), &tstart, nullptr));
        handle(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(tend), &tend, nullptr));
        start = std::min(start, tstart);
        end = std::max(end, tend);
    }

    return double(end - start) / 1e9;
}

void asyncCLNotify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "Asynchronous OpenCL error: %s.\n", errinfo);
}

cl_command_queue clCreatePerfCountersCommandQueueINTEL(cl_platform_id platform, cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_uint configuration, cl_int *errcode_ret)
{
    static cl_command_queue (*fptr)(cl_context, cl_device_id, cl_command_queue_properties, cl_uint, cl_int *) = nullptr;

    if (!fptr)
        fptr = (decltype(fptr)) clGetExtensionFunctionAddressForPlatform(platform, "clCreatePerfCountersCommandQueueINTEL");

    if (fptr)
        return (*fptr)(context, device, properties, configuration, errcode_ret);
    else {
        if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM;
        return nullptr;
    }
}

bool handleMetrics(MetricsDiscovery::TCompletionCode result, bool loud)
{
    if (result != MetricsDiscovery::CC_OK) {
        if (loud)
            yprintf("MDAPI error occurred: %d -- disabling frequency monitoring\n", static_cast<int>(result));
        return false;
    }
    return true;
    
}

bool handleMetrics(cl_int result, bool loud)
{
    if (result != CL_SUCCESS) {
        if (loud)
            yprintf("OpenCL error occurred: %d -- disabling frequency monitoring\n", static_cast<int>(result));
        return false;
    }
    return true;
}

template <typename T>
T *expectNonNull(T *value)
{
    if (!value)
        throw std::runtime_error("Unexpected NULL return value from MDAPI");
    return value;
}

bool mdapiInit(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    using namespace MetricsDiscovery;

#ifdef _WIN32
    auto lib = LoadLibraryA("igdmd64.dll");

    if (!lib) {
        // Look for igdmd64.dll in the same folder as igdrcl64.dll, which should
        //   already be loaded.
        auto rcl = GetModuleHandleA("igdrcl64.dll");
        if (rcl) {
            char pathchars[256];
            auto len = GetModuleFileNameA(rcl, pathchars, sizeof(pathchars));
            if (len > 0 && len < sizeof(pathchars)) {
                for (int i = len - 1; i >= 0; i--) {
                    if (pathchars[i] == '\\') {
                        pathchars[i] = '\0';
                        break;
                    }
                }
                std::string path(pathchars);
                path += "\\igdmd64.dll";
                lib = LoadLibraryA(path.c_str());
            }
            // FreeLibrary(rcl);
        }
    }

    if (!lib) {
        if (driver.Loud)
            yprintf("Could not find igdmd64.dll -- unable to monitor frequency\n");
        return false;
    }

    *(void**)(&OpenMetricsDevice)  = GetProcAddress(lib, "OpenMetricsDevice");
    *(void**)(&CloseMetricsDevice) = GetProcAddress(lib, "CloseMetricsDevice");
    *(void**)(&OpenAdapterGroup)   = GetProcAddress(lib, "OpenAdapterGroup");
#else
    void *lib = dlopen("libigdmd.so.1", RTLD_LAZY);

    if (!lib)
        lib = dlopen("libmd.so.1", RTLD_LAZY);

    if (!lib) {
        if (driver.Loud)
            yprintf("Could not find libigdmd.so.1 or libmd.so.1 -- unable to monitor frequency\n");
        return false;
    }

    *(void**)(&OpenMetricsDevice)  = dlsym(lib, "OpenMetricsDevice");
    *(void**)(&CloseMetricsDevice) = dlsym(lib, "CloseMetricsDevice");
    *(void**)(&OpenAdapterGroup)   = dlsym(lib, "OpenAdapterGroup");
#endif /* !_WIN32 */

    /* First try the MDAPI v1.6 adapter API */
    args.metricAdapter = nullptr;

    MetricsDiscovery::IAdapterGroupLatest *agroup;
    if (OpenAdapterGroup && OpenAdapterGroup(&agroup) == CC_OK) {
        auto agparams = expectNonNull(agroup->GetParams());
        for (uint32_t idx = 0; idx < agparams->AdapterCount; idx++) {
            auto adapter = expectNonNull(agroup->GetAdapter(idx));
            auto aparams = expectNonNull(adapter->GetParams());

            bool match = true;
            if (args.BDF) {
                match = (aparams->BusNumber      == args.BDF->pci_bus
                      && aparams->DeviceNumber   == args.BDF->pci_device
                      && aparams->FunctionNumber == args.BDF->pci_function);
            }
            if (match) {
                args.metricAdapter = adapter;
                HANDLE_MDAPI(adapter->OpenMetricsDevice(&args.metricDevice), driver.Loud);
                break;
            }
        }
    }

    /* Fall back to older API if we couldn't find a matching adapter */
    if (!args.metricAdapter) {
        if (!OpenMetricsDevice || !CloseMetricsDevice) {
            return false;
        }

        MetricsDiscovery::IMetricsDeviceLatest *dev = nullptr;
        HANDLE_MDAPI(OpenMetricsDevice(&dev), driver.Loud);
        args.metricDevice = dev;
    }



    /* Locate metrics of interest and activate */
    auto params = expectNonNull(args.metricDevice->GetParams());

    uint32_t _1 = 1;
    if (std::tie(params->Version.MajorNumber, params->Version.MinorNumber) < std::tie(_1, _1)) {
        return false;
    }

    auto apiMask = API_TYPE_OCL | API_TYPE_OGL4_X;

    for (uint32_t i = 0; i < params->ConcurrentGroupsCount; i++) {
        auto group = expectNonNull(args.metricDevice->GetConcurrentGroup(i));
        auto gparams = expectNonNull(group->GetParams());
        for (uint32_t j = 0; j < gparams->MetricSetsCount; j++) {
            auto set = expectNonNull(group->GetMetricSet(j));
            auto sparams = expectNonNull(set->GetParams());
            if (!(sparams->ApiMask & apiMask))
                continue;
            if (!strcmp(sparams->SymbolName, MetricSetName)) {
                args.metricSet = set;

                for (uint32_t k = 0; k < sparams->MetricsCount; k++) {
                    auto metric = expectNonNull(set->GetMetric(k));
                    auto mparams = expectNonNull(metric->GetParams());
                    std::string mname(mparams->SymbolName);
                    if (mname == "AvgGpuCoreFrequencyMHz")
                        FreqMetricIndex = k;
                    else if (auto search = MetricNames.find(mname); search != MetricNames.end())
                        AdditionalMetricIndices[search->second].push_back(k);
                }
            }
        }
    }

    if (FreqMetricIndex < 0) {
        if (driver.Loud) yprintf("Could not find AvgGpuCoreFrequencyMHz metric in metric group %s.\n", MetricSetName);
        return false;
    }

    args.metricSet->SetApiFiltering(apiMask);
    HANDLE_MDAPI(args.metricSet->Activate(), driver.Loud);
    return true;
}

template <typename To>
To rtConvert(MetricsDiscovery::TTypedValue_1_0 value)
{
    using namespace MetricsDiscovery;

    switch (value.ValueType) {
        case EValueType::VALUE_TYPE_UINT32: return To(value.ValueUInt32);
        case EValueType::VALUE_TYPE_UINT64: return To(value.ValueUInt64);
        case EValueType::VALUE_TYPE_FLOAT:  return To(value.ValueFloat);
        case EValueType::VALUE_TYPE_BOOL:   return To(value.ValueBool);
        default: stub();
    }

    return 0.;
}

bool rtCollectMetrics(const DriverRTArgs &args, cl_event event, TestResult &result, bool accumulate = false)
{
    using namespace MetricsDiscovery;

    if (!accumulate) {
        result.avgFreq = 0.;
        for (auto &metric: result.metrics)
            metric = 0.;
    }

    auto mparams = args.metricSet->GetParams();
    auto reportSize = mparams->QueryReportSize;
    size_t outSize = reportSize;
    std::vector<uint8_t> report(reportSize);

    auto status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_PERFCOUNTERS_INTEL,
                                          reportSize, report.data(), &outSize);

    if ((status != CL_SUCCESS) || (outSize != reportSize)) return false;

    std::vector<MetricsDiscovery::TTypedValue_1_0> results(mparams->MetricsCount + mparams->InformationCount);

    uint32_t reportCount = 0;
    handle(args.metricSet->CalculateMetrics(report.data(), reportSize, results.data(),
                                            (uint32_t)(results.size() * sizeof(TTypedValue_1_0)),
                                            &reportCount, false));

    if (reportCount < 1) return false;

    result.avgFreq += rtConvert<double>(results[FreqMetricIndex]) * 1e6;
    for (int m = 0; m < MetricCount; m++)
        for (auto index: AdditionalMetricIndices[m])
            result.metrics[m] += rtConvert<double>(results[index]);

    return true;
}

void rtCollectMetrics(const DriverRTArgs &args, const std::vector<cl_event> &events, TestResult &result)
{
    int good = 0;

    result.avgFreq = 0.;
    for (auto &metric: result.metrics)
        metric = 0.;

    for (auto &event: events) if (event)
        good += (int) rtCollectMetrics(args, event, result, true);

    if (good > 0) {
        result.avgFreq /= good;
        for (auto &metric: result.metrics)
            metric /= good;
    }
}

#ifndef CL_DEVICE_NUM_SLICES_INTEL
#define CL_DEVICE_NUM_SLICES_INTEL 0x4252
#endif
#ifndef CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL 0x4253
#endif
#ifndef CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL 0x4254
#endif

std::tuple<cl_device_id, cl_platform_id> rtFindDevice(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    static constexpr auto MAX_PLATFORMS = 16;
    static constexpr auto MAX_DEVICES = 16;
    static constexpr auto MAX_NAME_LEN = 256;

    cl_platform_id plat_ids[MAX_PLATFORMS];
    cl_uint nplat;
    cl_device_id dev_ids[MAX_DEVICES];
    cl_uint ndev = 0;

    cl_uint plat_idx = -1, dev_idx = -1;

    // Allow manual selection of platform and device #
    if (getenv("PLATFORM")) plat_idx = atoi(getenv("PLATFORM"));
    if (getenv("DEVICE"))    dev_idx = atoi(getenv("DEVICE"));

    handle(clGetPlatformIDs(MAX_PLATFORMS, plat_ids, &nplat));

    if (plat_idx == -1) {
        // Loop through platforms looking for a GPU.
        for (plat_idx = 0; plat_idx < nplat; plat_idx++) {
            cl_int status = clGetDeviceIDs(plat_ids[plat_idx], CL_DEVICE_TYPE_GPU, MAX_DEVICES, dev_ids, &ndev);
            if ((status == CL_SUCCESS) && (ndev > 0))
                break;
        }
    } else {
        if (plat_idx >= nplat) {
            fprintf(stderr, "Invalid platform index %d: only %d platforms available.\n", plat_idx, nplat);
            exit(1);
        }

        handle(clGetDeviceIDs(plat_ids[plat_idx], CL_DEVICE_TYPE_GPU, MAX_DEVICES, dev_ids, &ndev));
    }

    if (ndev == 0) {
        fprintf(stderr, "No GPUs were found.\n");
        exit(1);
    } else if (dev_idx == -1) {
        dev_idx = 0;
    } else if (dev_idx >= ndev) {
        fprintf(stderr, "Invalid device index %d: only %d devices available.\n", dev_idx, ndev);
        exit(1);
    }

    auto dev = dev_ids[dev_idx];

    cl_uint eus;
    cl_uint slices = 0, subslicesPerSlice = 0, eusPerSubslice = 0;

    handle(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &eus, nullptr));
    cl_int status = clGetDeviceInfo(dev, CL_DEVICE_NUM_SLICES_INTEL, sizeof(slices), &slices, nullptr)
                  | clGetDeviceInfo(dev, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, sizeof(subslicesPerSlice), &subslicesPerSlice, nullptr)
                  | clGetDeviceInfo(dev, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, sizeof(eusPerSubslice), &eusPerSubslice, nullptr);

    if (driver.Hardware != HW::Unknown)
        eusPerSubslice = eusPerSS(driver.Hardware);    /* some Xe2 drivers report this incorrectly */

    if (status == CL_SUCCESS) {
        // Some drivers report 2x too many EUs in CL_DEVICE_MAX_COMPUTE_UNITS.
        // Cross-check against slice/subslice/EU queries.
        auto computedEUs = slices * subslicesPerSlice * eusPerSubslice;
        if (eus >= 1.5 * computedEUs)
            eus >>= 1;
    } else
        slices = subslicesPerSlice = eusPerSubslice = 0;

    driver.info.EUs = eus;

    char name[MAX_NAME_LEN];
    size_t name_len;
    handle(clGetDeviceInfo(dev, CL_DEVICE_NAME, MAX_NAME_LEN, name, &name_len));
    if (name_len >= MAX_NAME_LEN)
        name_len = MAX_NAME_LEN - 1;
    name[name_len] = '\0';

    cl_uint mhz;
    handle(clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mhz, NULL));

    driver.info.ClockFrequency = mhz * 1.0e6;

    cl_device_pci_bus_info_khr bdf = {};
    if (clGetDeviceInfo(dev, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(bdf), &bdf, NULL) == CL_SUCCESS)
        args.BDF = bdf;

    cl_ulong gcache;
    handle(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &gcache, nullptr));
    LLCUpperBound = gcache;

    commonDeviceSetup(driver);

    if (!driver.Quiet) {
        printf("Using OpenCL device %s (%s, ", name, hwName(driver.Hardware));
        if (slices > 0)
            printf("%dx%dx%d = ", slices, subslicesPerSlice, eusPerSubslice);
        printf("%d EUs, %d MHz", driver.info.EUs, int(driver.info.ClockFrequency * 1e-6));
        if (driver.Loud && args.BDF) {
            printf(", ");
            if (args.BDF->pci_domain > 0)
                printf("%04x:", args.BDF->pci_domain);
            printf("%02x:%02x.%d", args.BDF->pci_bus, args.BDF->pci_device, args.BDF->pci_function);
        }
        printf(").\n");
    }

    return {dev, plat_ids[plat_idx]};
}

void rtInit(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    cl_int err;

    std::tie(args.device, args.platform) = rtFindDevice(driver);

    args.context = clCreateContext(NULL, 1, &args.device, &asyncCLNotify, NULL, &err);
    handle(err);

    cl_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (InOrderQueue)
        props = 0;
    if (driver.DeviceTime)
        props |= CL_QUEUE_PROFILING_ENABLE;
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, props, 0};

    args.queue = clCreateCommandQueueWithProperties(args.context, args.device, properties, &err);
    handle(err);

    auto makeProgram = [&](const char *cl) {
        cl_int status;
        auto program = clCreateProgramWithSource(args.context, 1, &cl, nullptr, &status);
        handle(status);
        handle(clBuildProgram(program, 1, &args.device, NULL, NULL, NULL));
        return program;
    };

    bool anyColdMat = driver.ColdA || driver.ColdB || driver.ColdC;
    if (anyColdMat) {
        auto fcl   = (driver.Hardware >= HW::XeHPG) ? lsc_flush_cl   : flush_cl;
        auto fname = (driver.Hardware >= HW::XeHPG) ? lsc_flush_name : flush_name;

        args.flushProgram = makeProgram(fcl);
        args.flushKernel = clCreateKernel(args.flushProgram, fname, &err);
        handle(err);
    } else {
        args.flushProgram = nullptr;
        args.flushKernel = nullptr;
    }

    if (driver.ColdICache) {
        args.flushICacheProgram = makeProgram(flush_icache_cl);
        args.flushICacheKernel = clCreateKernel(args.flushICacheProgram, flush_icache_name, &err);
        handle(err);
    } else {
        args.flushICacheProgram = nullptr;
        args.flushICacheKernel = nullptr;
    }

    if (FillCache) {
        args.fillCacheProgram = makeProgram(cache_fill_cl);
        args.fillCacheKernel = clCreateKernel(args.fillCacheProgram, cache_fill_name, &err);
        handle(err);
        args.fillBuffer = clCreateBuffer(args.context, CL_MEM_READ_WRITE, LLCUpperBound, nullptr, &err);
        handle(err);
    } else {
        args.fillCacheProgram = nullptr;
        args.fillCacheKernel = nullptr;
    }

    args.program = nullptr;
    args.kernel = nullptr;
}

bool rtInitMetrics(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    cl_int err = CL_SUCCESS;

    bool ok = mdapiInit(driver);
    if (!ok) return false;

    cl_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (InOrderQueue)
        props = 0;
    if (driver.DeviceTime)
        props |= CL_QUEUE_PROFILING_ENABLE;

    cl_command_queue perfQueue = nullptr;
    auto initQueue = [&]() {
        auto config = args.metricSet->GetParams()->ApiSpecificId.OCL;
        perfQueue = clCreatePerfCountersCommandQueueINTEL(args.platform, args.context, args.device, props, config, &err);
    };

    initQueue();

    if (!perfQueue || err) {
        if (driver.Loud) yprintf("Could not create MDAPI-enabled command queue. Frequency will not be monitored.\n");
        return false;
    }

    handle(clReleaseCommandQueue(args.queue));
    args.queue = perfQueue;
    return ok;
}

bool rtInitKernel(DriverRT &driver, const KernelInfo &info)
{
    auto &args = driver.rtArgs;
    cl_int binaryStatus, err;
    auto binaryLen = args.binary.size();
    auto binaryData = args.binary.data();
    args.program = clCreateProgramWithBinary(args.context, 1, &args.device, &binaryLen, (const unsigned char **) &binaryData, &binaryStatus, &err);

    if (!args.program || err) {
        if (!driver.Quiet) fprintf(stderr, "Program creation failed with error %d.\n", (int) err);
        return false;
    }
    if (binaryStatus != CL_SUCCESS) {
        if (!driver.Quiet) fprintf(stderr, "Program creation failed with binary status %d\n", (int) binaryStatus);
        return false;
    }

    err = clBuildProgram(args.program, 1, &args.device, NULL, NULL, NULL);
    if (err) {
        if (!driver.Quiet) fprintf(stderr, "Program build failed with error %d\n", err);
        return false;
    }

    args.kernel = clCreateKernel(args.program, info.kernelName(), &err);
    if (!args.kernel || err) {
        err = CL_SUCCESS;
        args.kernel = clCreateKernel(args.program, "__sycl_kernel_gemmstone_kernel", &err);
        if (!args.kernel || err) {
            if (!driver.Quiet) fprintf(stderr, "Kernel creation failed with error %d\n", err);
            return false;
        }
    }

    return true;
}

void rtFinishKernel(DriverRTArgs &args)
{
    if (args.kernel)
        handle(clReleaseKernel(args.kernel));
    if (args.program)
        handle(clReleaseProgram(args.program));

    args.kernel = nullptr;
    args.program = nullptr;
}

void rtFinish(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    if (driver.getEnableMetrics()) {
        args.metricSet->Deactivate();
        if (args.metricAdapter)
            args.metricAdapter->CloseMetricsDevice(args.metricDevice);
        else
            CloseMetricsDevice(static_cast<MetricsDiscovery::IMetricsDeviceLatest *>(args.metricDevice));
    }

    if (args.flushKernel)
        handle(clReleaseKernel(args.flushKernel));
    if (args.flushProgram)
        handle(clReleaseProgram(args.flushProgram));
    if (args.flushICacheKernel)
        handle(clReleaseKernel(args.flushICacheKernel));
    if (args.flushICacheProgram)
        handle(clReleaseProgram(args.flushICacheProgram));
    if (args.fillCacheKernel)
        handle(clReleaseKernel(args.fillCacheKernel));
    if (args.fillCacheProgram)
        handle(clReleaseProgram(args.fillCacheProgram));
    handle(clReleaseCommandQueue(args.queue));
    handle(clReleaseContext(args.context));

    args.queue = nullptr;
    args.context = nullptr;
    args.flushProgram = args.flushICacheProgram = args.fillCacheProgram = nullptr;
    args.flushKernel = args.flushICacheKernel = args.fillCacheKernel = nullptr;
}
#endif /* TARGET_OPENCL */

void rtReadBuffer(DriverRTArgs &args, rt_buffer buffer, TypedData &hostMem, size_t offset = 0)
{
    rtReadBuffer(args, buffer, offset, hostMem.smashedBytes(), hostMem.ptr);
}

rt_buffer rtCreateBuffer(DriverRT &driver, BufferFlags flags, PoolBuffer &buffer,
                         size_t bytes, void *hostPtr = nullptr, size_t offset = 0, bool coldCache = false)
{
    return buffer.realloc(driver, flags, bytes, hostPtr, offset, coldCache);
}

rt_buffer rtCreateBuffer(DriverRT &driver, BufferFlags flags, PoolBuffer &buffer,
                         TypedData &hostMem, size_t offset = 0, bool coldCache = false)
{
    return rtCreateBuffer(driver, flags, buffer, hostMem.smashedBytes(), hostMem.ptr, offset, coldCache);
}

template <typename T>
rt_buffer rtCreateBuffer(DriverRT &driver, BufferFlags flags, PoolBuffer &buffer,
                         std::vector<T> &hostMem, bool coldCache = false)
{
    return rtCreateBuffer(driver, flags, buffer, hostMem.size() * sizeof(T), hostMem.data(), 0, coldCache);
}

inline void rtFreeBuffer(DriverRTArgs &, PoolBuffer &buffer)
{
    buffer = {};
}

static inline void rtSetArg(rtKernelCtx& ctx, int argnum, Type T, superscalar arg, const char* name)
{
    ctx.argInfo.update(argnum, name, arg.real());
    auto M = Matrix::scalar(T, arg);
    rtSetArg(ctx, argnum, M.data.bytes, M.data.ptr);
}

void flushDriver(DriverRT &driver)
{
    auto &args = driver.rtArgs;
    auto &storage = args.storage;
    rtKernelCtx ctx{args.flushKernel};
    bool lsc = (driver.Hardware >= HW::XeHPG);

    bool flushed = false;
    storage.forEachBuffer([&](const PoolBuffer &buffer) {
        auto bytes = buffer.getFlushSize();
        if (!bytes || (!lsc && flushed)) return;
        auto buf = buffer.get();

        if (lsc) {
            rtSetArg(ctx, 0, buf, "buf");
            rtSetArg(ctx, 1, uint64_t(bytes), "bytes");
        }

        size_t lws = lsc ? 16       : 1;
        size_t gws = lsc ? bytes/64 : 1;
        gws = align_up(gws, lws);

#ifdef TARGET_OPENCL
        handle(clEnqueueNDRangeKernel(args.queue, args.flushKernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
        handle(clFinish(args.queue));
#endif

#ifdef TARGET_LEVEL_ZERO
        ze_group_count_t groups = {uint32_t(gws / lws), 1, 1};
        auto list = rtCreateList(args);
        handle(zeKernelSetGroupSize(args.flushKernel, uint32_t(lws), 1, 1));
        handle(zeCommandListAppendLaunchKernel(list, args.flushKernel, &groups, args.completionEvent, 0, nullptr));
        if (ImmediateCommandLists) {
            handle(zeEventHostSynchronize(args.completionEvent, UINT64_MAX));
            handle(zeEventHostReset(args.completionEvent));
        } else {
            handle(zeCommandListClose(list));
            handle(zeCommandQueueExecuteCommandLists(args.queue, 1, &list, args.fence));
            handle(zeFenceHostSynchronize(args.fence, UINT64_MAX));
            handle(zeFenceReset(args.fence));
            handle(zeCommandListDestroy(list));
        }
#endif
        flushed = true;
    });
}


void flushICache(DriverRT &driver, rt_list list)
{
    auto &args = driver.rtArgs;
    int sg = 16;
    auto ngroups = driver.info.EUs * 8;
#ifdef TARGET_OPENCL
    size_t lws = sg;
    size_t gws = sg * ngroups;
    handle(clEnqueueNDRangeKernel(args.queue, args.flushICacheKernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
    handle(clEnqueueMarkerWithWaitList(args.queue, 0, nullptr, nullptr));
#endif
#ifdef TARGET_LEVEL_ZERO
    ze_group_count_t groups = {uint32_t(ngroups), 1, 1};
    handle(zeKernelSetGroupSize(args.flushICacheKernel, uint32_t(sg), 1, 1));
    handle(zeCommandListAppendLaunchKernel(list, args.flushICacheKernel, &groups, nullptr, 0, nullptr));
    handle(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
#endif
}

void fillCache(DriverRTArgs &args, rt_list list)
{
    rtKernelCtx ctx{args.fillCacheKernel};
    rtSetArg(ctx, 0, args.fillBuffer, "fillBuffer");
    size_t gws = LLCUpperBound / 4;

#ifdef TARGET_OPENCL
    handle(clEnqueueNDRangeKernel(args.queue, args.fillCacheKernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr));
    handle(clEnqueueMarkerWithWaitList(args.queue, 0, nullptr, nullptr));
#endif

#ifdef TARGET_LEVEL_ZERO
    uint32_t lws = 64;
    ze_group_count_t groups = {uint32_t(gws / lws), 1, 1};
    handle(zeKernelSetGroupSize(args.fillCacheKernel, lws, 1, 1));
    handle(zeCommandListAppendLaunchKernel(list, args.fillCacheKernel, &groups, nullptr, 0, nullptr));
    handle(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
#endif
}

void prepareCache(DriverRT &driver, rt_list list)
{
    auto &args = driver.rtArgs;
    if (driver.ColdICache) flushICache(driver, list);
    if (FillCache) fillCache(args, list);
    if (driver.ColdTLB) rtRemapAll(args);
}

MatrixName padFill() {
    return OverrunTest ? MatrixName::NANS : MatrixName::ZERO;
}

FillModifiers fillMods(DriverRT &driver) {
    FillModifiers mods;
    mods.ones = driver.OneFill;
    mods.zeros = driver.ZeroFill;
    mods.beta0 = driver.TestBeta0;
    mods.stripeIndex = StripeIndex;
    return mods;
}

TestResult gemmDriver(DriverPhase phase, DriverRT &driver,
                      const KernelInfo &kinfo, bool timeOnly,
                      bool conja, bool conjb, char cOffType,
                      superscalar alpha, superscalar beta,
                      Matrix &A, Matrix &B, Matrix &C, Matrix &P, Matrix &CO,
                      Matrix &AO, Matrix &BO, Matrix &Ascale, Matrix &Bscale, Matrix &Ag, Matrix &Bg,
                      const int (&AddrOffset)[3])
{
    auto &args = driver.rtArgs;
    auto &storage = args.storage;
    TestResult result(driver.info.ClockFrequency);
#if TARGET_OPENCL
    auto queue = args.queue;
#endif
    auto kernel = args.kernel;

    auto m = C.m, n = C.n, k = A.n;
    auto mpad = C.mpad, npad = C.npad, kpad = A.npad;
    auto lda = A.ld, ldb = B.ld, ldc = C.ld, ldp = P.ld, ldco = CO.ld;
    auto transa = A.trans, transb = B.trans, transc = C.trans;
    auto strideA = A.stride, strideB = B.stride, strideC = C.stride;
    auto batchSize = C.batch;

    auto &info = kinfo.l3[0];
    auto Ta = info.typeA(), Tb = info.typeB(), Tc = info.typeC(), Ts = info.typeS();
    auto Ta_scale = info.typeAs(), Tb_scale = info.typeBs();
    bool copyA = info.copyA();
    bool copyB = info.copyB();
    bool copyC = info.copyC();
    bool trmm =  info.trmm;
    bool symm =  info.symm || info.hemm;
    bool left =  info.left;
    bool lower = info.lower;
    bool batchNS = (info.batch && (info.batch != 's'));
    bool batchNV = (info.batch == 'v');
    bool needCO = info.cOffset || info.cBias || info.sumA || info.sumB;
    bool coIsMatrix = (cOffType == 'M');
    bool coIsRow = (cOffType == 'C' || coIsMatrix || info.sumB);
    bool coIsCol = (cOffType == 'R' || coIsMatrix || info.sumA);
    bool C_packedA = (info.layoutC == 'A');

    auto mb = info.unroll[LoopM], nb = info.unroll[LoopN], kb = info.unroll[LoopK];
    auto kRemHandling = info.kRemainderHandling();
    bool mnk = info.isMNK();
    bool fixWGK = driver.FixWGK || info.fixedWGK();

    double start = 0, end = 0;

    // Persistent data across driver calls.
    auto &As = storage.As;
    auto &Bs = storage.Bs;
    auto &Cs = storage.Cs;

    auto &As_buffer = storage.As_buffer;
    auto &Bs_buffer = storage.Bs_buffer;
    auto &Cs_buffer = storage.Cs_buffer;
    auto &P_buffer  = storage.P_buffer;
    auto &CO_buffer = storage.CO_buffer;

    auto &Aptrs_buffer = storage.Aptrs_buffer;
    auto &Bptrs_buffer = storage.Bptrs_buffer;
    auto &Cptrs_buffer = storage.Cptrs_buffer;

    auto &AO_buffer = storage.AO_buffer;
    auto &BO_buffer = storage.BO_buffer;
    auto &Ascale_buffer = storage.aScale_buffer;
    auto &Bscale_buffer = storage.bScale_buffer;
    auto &Ag_buffer = storage.Ag_buffer;
    auto &Bg_buffer = storage.Bg_buffer;

    auto &alphaBuffer = storage.alphaBuffer;
    auto &betaBuffer = storage.betaBuffer;

    auto &statusElems = storage.statusElems;
    auto &statusBuffer = storage.statusBuffer;

    auto &tempCBuffer = storage.tempCBuffer;
    auto &tempCBytes = storage.tempCBytes;

    auto &trackingElems = storage.trackingElems;
    auto &trackingStride = storage.trackingStride;
    auto &trackingWGSize = storage.trackingWGSize;
    auto &trackingBuffer = storage.trackingBuffer;

#if TARGET_LEVEL_ZERO
    auto &list = storage.list;
    auto &curEvents = storage.curEvents;
#else
    rt_list list = nullptr;
#endif

    // Consistency checks.
    if (!copyA && (transa != (info.layoutA == 'T')))
        throw std::runtime_error("transa parameter incorrect in GEMM driver.");
    if (!copyB && (transb != (info.layoutB == 'T')))
        throw std::runtime_error("transb parameter incorrect in GEMM driver.");
    if (!copyC && (transc != (info.layoutC == 'T')))
        throw std::runtime_error("transc parameter incorrect in GEMM driver.");
    if (trmm && (!copyA || !copyB))
        throw std::runtime_error("Only copy-based TRMM supported.");

    // Handle work group.
    int wgX, wgY, wgZ;
    auto &wgM = mnk ? wgX : wgY;
    auto &wgN = mnk ? wgY : wgX;

    if (info.wg[0] > 0) {
        wgM = info.wg[LoopM];
        wgN = info.wg[LoopN];
        wgZ = info.wg[LoopK];
    } else {
        std::tie(wgX, wgY) = getDefaultWGSize(driver.Hardware, copyA, copyB);
        wgZ = 1;
    }

    applyWGOverrides(info, driver.WGOverride, &wgM, &wgN, &wgZ);

    // Handle blocking.
    integer BlockM, BlockN, BlockK;
    std::tie(BlockM, BlockN, BlockK) = driver.getBlocking(m, n, k, info);

    if (batchNV) BlockK = k; /* no k-blocking support */

    if (info.fusedBeta() || info.fusedPostOps()) {
        BlockM = m;         /* no blocking support in this driver: needs separate status */
        BlockN = n;
    }

    BlockM = align_up(BlockM, kinfo.mbAlign() * wgM);
    BlockN = align_up(BlockN, kinfo.nbAlign() * wgN);
    BlockK = std::max<integer>(BlockK, 1);

    integer nblock_m = div_up(m, BlockM);
    integer nblock_n = div_up(n, BlockN);
    integer nblock_k = div_up(k, BlockK);

    nblock_k = std::max<integer>(nblock_k, 1);
    if (info.kParallel() && info.kPadding()) nblock_k += wgZ;

    if (info.kParallelLocal()) {
        integer blockTiles = div_up(BlockM, info.wgTile(LoopM))
                           * div_up(BlockN, info.wgTile(LoopN));
        int wg_size = info.wg[LoopM] * info.wg[LoopN];
        int ssWGs = eusPerSS(driver.Hardware) * threadsPerEU(driver.Hardware, info) / wg_size;

        bool redoBlockK = false;
        if (fixWGK)
            redoBlockK = true;
        else if (info.kParallelVariable()) {
            if (driver.NBlockOverrides[LoopK])
                wgZ = std::min(wgZ, int(driver.NBlockOverrides[LoopK]));
            else if (blockTiles < ssWGs)
                wgZ = std::min(wgZ, roundup_pow2(div_up(ssWGs, int(blockTiles))));
            else
                wgZ = 1;
            redoBlockK = true;
        } else if (info.shrinkWGK()) {
            int goalWGs = int(ssWGs * driver.info.EUs * info.fillGoal() / eusPerSS(driver.Hardware));
            wgZ = std::min(wgZ, roundup_pow2(div_up(goalWGs, int(blockTiles))));
            redoBlockK = true;
        } else
            wgZ = std::min(wgZ, int(nblock_k));

        // Ensure WG size is not too large.
        int maxWGK = std::max<int>(1, maxThreadsPerTG(driver.Hardware, info) / (wgM * wgN * info.wgExpand));
        if (maxWGK < wgZ && !fixWGK) {
            wgZ = maxWGK;
            redoBlockK |= (!info.kParallel() || driver.TestNonKParallelGlobal);
        }

        if (redoBlockK) {
            BlockK = std::max<integer>(div_up(k, wgZ), 1);
            BlockK = align_up(BlockK, info.unroll[LoopK]);
            nblock_k = std::max<integer>(div_up(k, BlockK), 1);
        }
    }

    integer BlockCInner = (C_packedA ? BlockN : BlockM);
    integer BlockCOuter = (C_packedA ? BlockM : BlockN);

    if (ShowBlockInfo && phase == DriverPhase::Init) {
        printf("                Blocking: %dx%dx%d, %dx%dx%d blocks\n", int(BlockM), int(BlockN), int(BlockK), int(nblock_m), int(nblock_n), int(nblock_k));
        printf("            Thread group: %dx%dx%d (%cx%cx%c)\n", int(wgX), int(wgY), int(wgZ), loopChar(info.loopOrder[0], 0), loopChar(info.loopOrder[1], 1), loopChar(info.loopOrder[2], 2));
    }

    size_t globalWSZ = 1;
    uint32_t k0 = 0;
    if (info.kParallel() || info.kParallelLocal()) {
        k0 = uint32_t(BlockK);   BlockK = std::max<integer>(k, 1);
        globalWSZ = nblock_k;    nblock_k = 1;
        if (info.fixedWGK() && !info.kParallel()) globalWSZ = wgZ;
        if (!info.fixedWGK()) wgZ = std::min(wgZ, int(globalWSZ));
    }

    if (kinfo.isBatched())
        globalWSZ *= batchSize;

    // Memory preparations.
    bool upperBound = (phase == DriverPhase::InitMultiple);

    // Tests in a series have no host data and reuse the buffers created by InitMultiple.
    bool reuseSeriesBuffers = (phase == DriverPhase::Init && !A.data.ptr);

    int32_t ldaExtra = (copyA && info.bOffset) ? (Tc.paddedSize() / Ta) : 0;
    int32_t ldbExtra = (copyB && info.aOffset) ? (Tc.paddedSize() / Tb) : 0;
    int32_t ldcExtra = 0;

    int32_t lda_packed = niceLD(driver.Hardware, Ta, int32_t(BlockK + ldaExtra), upperBound);
    int32_t ldb_packed = niceLD(driver.Hardware, Tb, int32_t(BlockK + ldbExtra), upperBound);
    int32_t ldc_packed = niceLD(driver.Hardware, Tc, int32_t(BlockCInner + ldcExtra), upperBound);

    lda_packed = align_up(lda_packed, info.alignA / Ta);
    ldb_packed = align_up(ldb_packed, info.alignB / Tb);
    ldc_packed = align_up(ldc_packed, info.alignC / Tc);

    int32_t strideAp = info.batch ? niceLD(driver.Hardware, Ta, int32_t(BlockM * lda_packed)) : 0;
    int32_t strideBp = info.batch ? niceLD(driver.Hardware, Tb, int32_t(BlockN * ldb_packed)) : 0;
    int32_t strideCp = info.batch ? niceLD(driver.Hardware, Tc, int32_t(BlockCOuter * ldc_packed)) : 0;

    int64_t strideAs = copyA ? strideAp : strideA;      // stride_A0/B0 are DataType::q (64-bit)
    int64_t strideBs = copyB ? strideBp : strideB;
    int32_t strideCs = copyC ? strideCp : strideC;

    auto packUnpackC = [&](bool pack) {
        if (!copyC) return;
        for (int b = 0; b < batchSize; b++) {
            for (int i0b = 0; i0b < nblock_m; i0b++) {
                for (int j0b = 0; j0b < nblock_n; j0b++) {
                    auto i0 = i0b * BlockM, j0 = j0b * BlockN;
                    auto nm = std::min(m - i0, BlockM);
                    auto nn = std::min(n - j0, BlockN);
                    auto nmpad = std::min(mpad - i0, BlockM);
                    auto nnpad = std::min(npad - j0, BlockN);
                    auto ncRounded = C_packedA ? align_up(nn, nb) : align_up(nm, mb);
                    int32_t this_ldc_packed = niceLD(driver.Hardware, Tc, int32_t(ncRounded + ldcExtra), upperBound);

                    auto Cunpacked = C.selectBatch(b).subview(i0, j0, nmpad, nnpad);
                    auto Cpacked = Cs + ldc_packed * BlockCOuter * (i0b + j0b * nblock_m) + b * strideCp + AddrOffset[2];

                    if (pack) {
                        C_packedA ? packC(Cpacked, this_ldc_packed, Cunpacked, mb, info.crosspackC, info.C_tileR, info.C_tileC, false, padFill())
                                  : packR(Cpacked, this_ldc_packed, Cunpacked, nb, info.crosspackC, info.C_tileR, info.C_tileC, false, padFill());
                    } else {
                        C_packedA ? unpackC(Cunpacked, Cpacked, this_ldc_packed, mb, info.crosspackC, info.C_tileR, info.C_tileC)
                                  : unpackR(Cunpacked, Cpacked, this_ldc_packed, nb, info.crosspackC, info.C_tileR, info.C_tileC);
                    }
                }
            }
        }
    };

    if ((phase == DriverPhase::Init && !reuseSeriesBuffers) || phase == DriverPhase::InitMultiple) {
        rtClearColdCache(storage);

        // Allocate packed matrix memory as needed.
        if (copyA) {
            auto As_elems = lda_packed * BlockM * std::max(int(nblock_m * nblock_k), 1);
            As_elems = std::max<integer>(As_elems, strideAp * batchSize);
            As = TypedData(Ta, As_elems);
        } else
            As = A.data.view();

        if (copyB) {
            auto Bs_elems = ldb_packed * BlockN * std::max(int(nblock_n * nblock_k), 1);
            Bs_elems = std::max<integer>(Bs_elems, strideBp * batchSize);
            Bs = TypedData(Tb, Bs_elems);
        } else
            Bs = B.data.view();

        if (copyC) {
            auto Cs_elems = ldc_packed * BlockCOuter * std::max(int(nblock_m * nblock_n), 1);
            Cs_elems = std::max<integer>(Cs_elems, strideCp * batchSize);
            Cs = TypedData(Tc, Cs_elems);
        } else
            Cs = C.data.view();

        integer P_elems  = P.paddedElements();
        integer CO_elems = (coIsRow ? npad : 1) * (coIsCol ? mpad : 1);

        // Pack inputs as needed.
        for (int b = 0; b < batchSize; b++) {
            for (int h0b = 0; h0b < nblock_k; h0b++) {
                auto h0 = h0b * BlockK;
                auto nk = std::min(k - h0, BlockK);
                auto nkRounded = align_up(nk, kb);

                int32_t this_lda_packed = niceLD(driver.Hardware, Ta, int32_t(nkRounded + ldaExtra), upperBound);
                int32_t this_ldb_packed = niceLD(driver.Hardware, Tb, int32_t(nkRounded + ldbExtra), upperBound);

                if (copyA) for (int i0b = 0; i0b < nblock_m; i0b++) {
                    auto i0 = i0b * BlockM;
                    auto nm = std::min(m - i0, BlockM);
                    auto Asrc = A.selectBatch(b).subview(i0, h0, nm, nk);
                    auto Adst = As + lda_packed * BlockM * (i0b + h0b * nblock_m) + b * strideAp + AddrOffset[0];

                    packC(Adst, this_lda_packed, Asrc, info.packSizeA, info.crosspackA, info.A_tileR, info.A_tileC, info.bOffset, padFill());
                }

                if (copyB) for (int j0b = 0; j0b < nblock_n; j0b++) {
                    auto j0 = j0b * BlockN;
                    auto nn = std::min(n - j0, BlockN);
                    auto Bsrc = B.selectBatch(b).subview(h0, j0, nk, nn);
                    auto Bdst = Bs + ldb_packed * BlockN * (j0b + h0b * nblock_n) + b * strideBp + AddrOffset[1];

                    packR(Bdst, this_ldb_packed, Bsrc, info.packSizeB, info.crosspackB, info.B_tileR, info.B_tileC, info.aOffset, padFill());
                }
            }
        }

        if (copyC) packUnpackC(true);

        As.smash();
        Bs.smash();
        Cs.smash();

        // Prepare buffers.
        rtCreateBuffer(driver,  BUF_R  | driver.ExtraFlagsA, storage.As_buffer, As, size_t(AddrOffset[0] * Ta), driver.ColdA);
        rtCreateBuffer(driver,  BUF_R  | driver.ExtraFlagsB, storage.Bs_buffer, Bs, size_t(AddrOffset[1] * Tb), driver.ColdB);
        rtCreateBuffer(driver,  BUF_RW | driver.ExtraFlagsC, storage.Cs_buffer, Cs, size_t(AddrOffset[2] * Tc), driver.ColdC);
        if (info.hpl)
            rtCreateBuffer(driver, BUF_W | driver.ExtraFlagsP, storage.P_buffer, P.data);
        if (needCO)
            rtCreateBuffer(driver, BUF_R | driver.ExtraFlagsCO, storage.CO_buffer, CO.data);

        // Prepare matrix pointers for non-strided batch.
        if (batchNS) {
            std::vector<rt_buffer> Aptrs(batchSize), Bptrs(batchSize), Cptrs(batchSize);
            for (int b = 0; b < batchSize; b++) {
                Aptrs[b] = rtOffsetBuffer(As_buffer, ptrdiff_t(b * strideAs * Ta));
                Bptrs[b] = rtOffsetBuffer(Bs_buffer, ptrdiff_t(b * strideBs * Tb));
                Cptrs[b] = rtOffsetBuffer(Cs_buffer, ptrdiff_t(b * strideCs * Tc));
            }

            rtCreateBuffer(driver, BUF_R, storage.Aptrs_buffer, Aptrs);
            rtCreateBuffer(driver, BUF_R, storage.Bptrs_buffer, Bptrs);
            rtCreateBuffer(driver, BUF_R, storage.Cptrs_buffer, Cptrs);
        }

        if (batchNV) {
            auto calpha = Matrix::scalar(Ts, alpha);
            auto cbeta = Matrix::scalar(Ts, beta);
            rtCreateBuffer(driver, BUF_R, storage.alphaBuffer, calpha.data);
            rtCreateBuffer(driver, BUF_R, storage.betaBuffer, cbeta.data);
        }

        // Prepare buffers for offsets/scales.
        if (info.aOffsetPtr())         rtCreateBuffer(driver, BUF_R, storage.AO_buffer, AO.data);
        if (info.bOffsetPtr())         rtCreateBuffer(driver, BUF_R, storage.BO_buffer, BO.data);
        if (info.scaleA)               rtCreateBuffer(driver, BUF_R, storage.aScale_buffer, Ascale.data);
        if (info.scaleB)               rtCreateBuffer(driver, BUF_R, storage.bScale_buffer, Bscale.data);
        if (info.needsSeparateASums()) rtCreateBuffer(driver, BUF_R, storage.Ag_buffer, Ag.data);
        if (info.needsSeparateBSums()) rtCreateBuffer(driver, BUF_R, storage.Bg_buffer, Bg.data);

        int maxPersistentGroups = 0;
        if (PersistentGroupCountOverride)
            maxPersistentGroups = PersistentGroupCountOverride;
        else {
            int effWG = info.wg[0] * info.wg[1];
            if (!info.kParallelVariable())
                effWG *= info.wg[2];
            maxPersistentGroups = (eusPerSS(driver.Hardware) * threadsPerEU(driver.Hardware, info)) / effWG;   /* TG/SS  */
            maxPersistentGroups *= (driver.info.EUs / eusPerSS(driver.Hardware));
        }
        maxPersistentGroups = std::max(maxPersistentGroups, 1);

        int maxKSlicedTiles = 0;
        if (!info.kParallelVariable())
            maxKSlicedTiles = div_up(m, info.wgTile(LoopM)) * div_up(n, info.wgTile(LoopN));
        else if (KSlicedTilesOverride >= 0)
            maxKSlicedTiles = std::max(1, KSlicedTilesOverride);   /* FIXME */
        else
            maxKSlicedTiles = KSlicedWaves * maxPersistentGroups;

        if (info.fusedBeta() || info.fusedPostOps()) {
            statusElems = maxKSlicedTiles * (int(info.fusedBeta()) + int(info.fusedPostOps()));
            storage.statusBuffer.realloc(driver, BUF_RW, statusElems * 64);
        }

        if (info.needsTempC()) {
            int esize = std::max<int>(Tc.real().paddedSize(), 4);
            if (Tc.isComplex())
                esize *= 2;
            int stride = info.wgTile(LoopM) * info.wgTile(LoopN);
            if (info.sumA) stride += info.wgTile(LoopM);
            if (info.sumB) stride += info.wgTile(LoopN);
            stride *= esize;
            stride = align_up(stride, 64);
            tempCBytes = maxKSlicedTiles * stride;
            storage.tempCBuffer.realloc(driver, BUF_RW, tempCBytes);
        }

        if (info.tracking) {
            int trackingGroups = div_up(m, info.wgTile(LoopM)) * div_up(n, info.wgTile(LoopN));
            trackingStride = 8;

            if (info.isPersistent()) {
                trackingStride *= div_up(trackingGroups, maxPersistentGroups);
                trackingGroups = maxPersistentGroups;
            }

            trackingWGSize = info.wg[LoopM] * info.wg[LoopN] * info.wg[LoopK];
            trackingElems = trackingGroups * trackingWGSize * trackingStride;
            storage.trackingBuffer.realloc(driver, BUF_RW, trackingElems * sizeof(Timepoint));
        }

    } /* end persistent init phase */

    // Update buffer sizes for cold-cache testing.
    if (reuseSeriesBuffers) {
        auto As_elems = copyA ? lda_packed * BlockM * std::max(int(nblock_m * nblock_k), 1)
                              : A.elements();
        auto Bs_elems = copyB ? ldb_packed * BlockN * std::max(int(nblock_n * nblock_k), 1)
                              : B.elements();
        auto Cs_elems = copyC ? ldc_packed * BlockCOuter * std::max(int(nblock_m * nblock_n), 1)
                              : C.elements();
        storage.As_buffer.setFlushSize(driver.ColdA ? As_elems * Ta : 0);
        storage.Bs_buffer.setFlushSize(driver.ColdB ? Bs_elems * Tb : 0);
        storage.Cs_buffer.setFlushSize(driver.ColdC ? Cs_elems * Tc : 0);
    }

#ifdef TARGET_LEVEL_ZERO
    if (phase == DriverPhase::Init) {
        list = rtCreateList(args);
        integer nevents = nblock_m * nblock_n;
        if (TimestampEvents) nevents *= nblock_k;
        rtCreateEvents(args, int(nevents));
        rtAllocTimestamps(args);
    }
#endif

    if (phase == DriverPhase::RunList || phase == DriverPhase::Run) {
        if ((info.fusedBeta() || info.fusedPostOps()) && KSlicedTilesOverride != 0)
            rtZeroBuffer(args, statusBuffer, statusElems * 64);
        if (info.tracking)
            rtZeroBuffer(args, trackingBuffer, trackingElems * sizeof(Timepoint));
        if (info.zeroTempC())
            rtZeroBuffer(args, tempCBuffer, tempCBytes);
    }

    if (phase == DriverPhase::PrepareList || phase == DriverPhase::Run) {
        rtKernelCtx ctx{kernel};
        // Set default arguments.
        bool needFlags = info.canConjAB() || info.cOffset || info.cBias;
        bool needDiag = info.triangular || info.symm || info.trmm;

        uint32_t flags = 0;
        int argOffsetCO = 0, argFlags = 0;
        int argGroupCountM = 0, argGroupCountN = 0;
        int argVD = 0, argUVDRecip = 0, argHilbertBail = 0;
        int argBSlice = 0, argBThresh = 0;
        int argWGStrideM = 0, argWGStrideN = 0;
        int argGCMNRecip = 0, argGCMRecip = 0, argGCNRecip = 0;
        int argGroupStride = 0, argGroupCount = 0;
        int argK0 = 0, argKVConfig = 0, argKUnsyncedRecip = 0;

        int lastArg = 0, batchArgs = 0;
        info.localA ? rtSetLocalArg(ctx, 0, As.smashedBytes(), "A")
                    : rtSetArg(ctx, 0, batchNS ? Aptrs_buffer : As_buffer, "A");
        info.localB ? rtSetLocalArg(ctx, 1, Bs.smashedBytes(), "B")
                    : rtSetArg(ctx, 1, batchNS ? Bptrs_buffer : Bs_buffer, "B");
        info.localC ? rtSetLocalArg(ctx, 2, Cs.smashedBytes(), "C")
                    : rtSetArg(ctx, 2, batchNS ? Cptrs_buffer : Cs_buffer, "C");
        //rtSetArg(ctx, 3, offsetA);
        //rtSetArg(ctx, 4, offsetB);
        //rtSetArg(ctx, 5, offsetC);
        lastArg += 6;
        if (info.batch == 's') {
            rtSetArg(ctx, lastArg++, strideAs, "strideA");
            rtSetArg(ctx, lastArg++, strideBs, "strideB");
            rtSetArg(ctx, lastArg++, strideC, "strideC");
            batchArgs = 3;
        } else if (info.batch == 'n') {
            int32_t offsetBatch = 0;
            rtSetArg(ctx, lastArg++, offsetBatch, "offsetBatch");
            batchArgs = 1;
        } else if (info.batch == 'v') {
            int32_t incrAArray = 1, incrBArray = 1;
            rtSetArg(ctx, lastArg++, incrAArray, "incrAArray");
            rtSetArg(ctx, lastArg++, incrBArray, "incrBArray");
        } else if (info.batch)
            throw std::runtime_error("Unsupported batch type.");
        //rtSetArg(ctx, 6, lda_packed);
        //rtSetArg(ctx, 7, ldb_packed);
        //rtSetArg(ctx, 8, ldc);
        //rtSetArg(ctx, 9, m);
        //rtSetArg(ctx, 10, n);
        //rtSetArg(ctx, 11, k);
        lastArg += 6;
        //rtSetArg(ctx, 12, diag);
        if (needDiag)
            lastArg++;
        if (info.hpl) {
            rtSetArg(ctx, lastArg++, P_buffer, "P");
            lastArg++; //rtSetArg(ctx, 14, offsetP);
            rtSetArg(ctx, lastArg++, ldp, "ldp");
        }
        auto argnScalar = lastArg;
        rtSetArg(ctx, lastArg++, Ts.real(), alpha.real(), "alpha");
        rtSetArg(ctx, lastArg++, Ts.real(), beta.real(), "beta");
        if (Ts.isComplex()) {
            rtSetArg(ctx, lastArg++, Ts.real(), alpha.imag(), "alpha_i");
            rtSetArg(ctx, lastArg++, Ts.real(), beta.imag(), "beta_i");
        }
        if (batchNV) {
            uint32_t incrAlpha = 0, incrBeta = 0;
            rtSetArg(ctx, lastArg++, alphaBuffer, "alphaBuffer");
            rtSetArg(ctx, lastArg++, betaBuffer, "betaBuffer");
            rtSetArg(ctx, lastArg++, incrAlpha, "incrAlpha");
            rtSetArg(ctx, lastArg++, incrBeta, "incrBeta");
        }
        if (info.aOffset == 's' || info.bOffset == 's') {
            uint32_t abo = 0;
            if (info.aOffset == 's') abo |= uint16_t(AO.get<int16_t>(0, 0));
            if (info.bOffset == 's') abo |= uint16_t(BO.get<int16_t>(0, 0)) << 16;
            rtSetArg(ctx, lastArg++, abo, "abo");
        }
        if (info.aOffsetPtr())         rtSetArg(ctx, lastArg++, AO_buffer, "AO");
        if (info.bOffsetPtr())         rtSetArg(ctx, lastArg++, BO_buffer, "BO");
        if (info.scaleA)               rtSetArg(ctx, lastArg++, Ascale_buffer, "Ascale");
        if (info.scaleB)               rtSetArg(ctx, lastArg++, Bscale_buffer, "Bscale");
        if (info.needsSeparateASums()) rtSetArg(ctx, lastArg++, Ag_buffer, "Ag");
        if (info.needsSeparateBSums()) rtSetArg(ctx, lastArg++, Bg_buffer, "Bg");
        if (info.scaleA || info.aOffset == 'g' || info.needsAGroupSums())
            rtSetArg(ctx, lastArg++, div_up(m, info.aqGroup[0]), "aqGroupCount");
        if (info.scaleB || info.bOffset == 'g' || info.needsBGroupSums())
            rtSetArg(ctx, lastArg++, div_up(k, info.bqGroup[0]), "bqGroupCount");
        if (needCO) {
            if (info.cOffset || info.cBias) {
                if (cOffType == 'C' || cOffType == 'M') flags |= FlagCOColumn;
                if (cOffType == 'R' || cOffType == 'M') flags |= FlagCORow;
            }
            rtSetArg(ctx, lastArg++, CO_buffer, "CO");
            argOffsetCO = lastArg++;
            if (info.canCMatrixOffset())
                rtSetArg(ctx, lastArg++, ldco, "ldco");
        }
        if (info.needsTempC())
            rtSetArg(ctx, lastArg++, tempCBuffer, "tempC");
        if (needFlags)
            argFlags = lastArg++;
        if (info.kParallelVariable())
            argK0 = lastArg++;
        else if ((info.kParallel() || info.kParallelLocal()) && !info.kParallelVariable())
            rtSetArg(ctx, lastArg++, k0, "k0");
        if (info.fusedBeta() || info.fusedPostOps())
            rtSetArg(ctx, lastArg++, statusBuffer, "status");
        if (info.isLinearOrder()) {
            argGroupCountM = lastArg++;
            argGroupCountN = lastArg++;
        }
        if (info.kParallel() && info.fusedBeta())
            rtSetArg(ctx, lastArg++, uint32_t(div_up(globalWSZ, wgZ)), "kParallelBlocks");
        if (info.scrambleM())
            argGCMRecip = lastArg++;
        if (info.scrambleN())
            argGCNRecip = lastArg++;
        if (info.isSimpleLinear() || info.isNestedLinear())
            argGCMNRecip = lastArg++;
        else if (info.isHilbert()) {
            argVD = lastArg++;
            argUVDRecip = lastArg++;
            argHilbertBail = lastArg++;
        } else if (info.isBoustrophedon()) {
            argBSlice = lastArg++;
            argBThresh = lastArg++;
        }
        if (info.scrambleM())
            argWGStrideM = lastArg++;
        if (info.scrambleN())
            argWGStrideN = lastArg++;
        if (info.kParallelVariable()) {             // kv block: k0 now lives here (frontend order)
            argK0 = lastArg++;
            argKVConfig = lastArg++;
            argKUnsyncedRecip = lastArg++;
        }
        if (info.isPersistent())                    // group_stride is persistent-only now
            argGroupStride = lastArg++;
        if (info.isPersistent() && info.isHilbert() && info.triangular)
            argGroupCount = lastArg++;
        if (info.perKSLM > 0) {
            rtSetLocalArg(ctx, lastArg++, (wgZ == 1) ? info.slm
                                                        : std::max(info.slm, info.perKSLM * wgZ), "slm");
        }
        if (info.tracking) {
            rtSetArg(ctx, lastArg++, trackingBuffer, "tracking");
            rtSetArg(ctx, lastArg++, uint16_t(trackingStride * sizeof(Timepoint)), "trackingStride");
        }

        // Main driver.
#ifdef TARGET_OPENCL
        auto nDispatch = nblock_m * nblock_n * nblock_k;

        std::vector<rt_event> curEvents(nblock_m * nblock_n);
        std::vector<rt_event> allEvents;
        allEvents.reserve(nDispatch);
#endif

#ifdef TARGET_LEVEL_ZERO
        rtMakeTimestampsDeviceResident(args);
#endif

        // For no-copy, loop ordering is MNK; copy-based GEMM is NMK.
        // Left-side TRMM is NMK; right side is MNK.
        bool nmk = trmm ? left : (copyA || copyB);
        auto nblock_q = nmk ? nblock_n : nblock_m;
        auto nblock_r = nmk ? nblock_m : nblock_n;
        bool reverse_r = trmm && (left == lower);
        bool reverse_k = reverse_r;

        prepareCache(driver, list);

        if (phase == DriverPhase::Run && rtImmediateExecution())
            start = dsecnd();

#ifdef TARGET_LEVEL_ZERO
        rtStartTimestamps(driver, list);
#endif

        integer nk = BlockK;
        for (integer h0b_ = 0; h0b_ < nblock_k; h0b_++) {
            integer h0b = reverse_k ? (nblock_k - 1 - h0b_) : h0b_;
            integer h0 = h0b * BlockK;
            nk = std::min(k - h0, BlockK);
            integer nkRounded = align_up(nk, kb);

            int32_t this_lda_packed = niceLD(driver.Hardware, Ta, int32_t(nkRounded + ldaExtra));
            int32_t this_ldb_packed = niceLD(driver.Hardware, Tb, int32_t(nkRounded + ldbExtra));

            auto thisBeta = beta;
            if (h0b_ > 0) thisBeta = 1;

            rtSetArg(ctx, argnScalar + 1, Ts.real(), thisBeta.real(), "beta");
            if (Ts.isComplex())
                rtSetArg(ctx, argnScalar + 3, Ts.real(), thisBeta.imag(), "beta_i");

            for (integer r0b_ = 0; r0b_ < nblock_r; r0b_++) {
                auto r0b = reverse_r ? (nblock_r - 1 - r0b_) : r0b_;

                for (integer q0b = 0; q0b < nblock_q; q0b++) {
                    auto i0b = nmk ? r0b : q0b;
                    auto j0b = nmk ? q0b : r0b;
                    integer i0 = i0b * BlockM;
                    integer j0 = j0b * BlockN;
                    integer nm = std::min(m - i0, BlockM);
                    integer nn = std::min(n - j0, BlockN);

                    bool transaEff = transa;
                    bool transbEff = transb;
                    if (symm) {
                        if (left && !lower) transaEff = !transaEff;
                        if (!left && lower) transbEff = !transbEff;
                    }

                    int64_t offsetA = copyA ? lda_packed * BlockM * (i0b + h0b * nblock_m)
                                            : offset(transaEff, lda, i0, h0) + AddrOffset[0];
                    int64_t offsetB = copyB ? ldb_packed * BlockN * (j0b + h0b * nblock_n)
                                            : offset(transbEff, ldb, h0, j0) + AddrOffset[1];
                    int64_t offsetC = copyC ? ldc_packed * BlockCOuter * (i0b + j0b * nblock_m)
                                            : C.offset(i0, j0) + AddrOffset[2];
                    int64_t offsetP = P.offset(i0, j0);
                    int64_t offsetCO = coIsMatrix ? (i0 + j0 * ldco) :
                                          coIsRow ? j0 :
                                          coIsCol ? i0 : 0;

                    auto ncRounded = C_packedA ? align_up(nn, nb) : align_up(nm, mb);
                    int32_t this_ldc_packed = niceLD(driver.Hardware, Tc, int32_t(ncRounded + ldcExtra), upperBound);

                    int32_t this_lda = copyA ? this_lda_packed : lda;
                    int32_t this_ldb = copyB ? this_ldb_packed : ldb;
                    int32_t this_ldc = copyC ? this_ldc_packed : ldc;
                    uint32_t this_flags = flags;
                    if (h0b_ < (nblock_k - 1))
                        this_flags |= FlagNonfinalKBlock;

                    int32_t nkKernel = int32_t(kRemHandling ? nk : nkRounded);

                    int32_t diag = int32_t(((trmm || symm) &&  left) ? h0 - i0 :
                                           ((trmm || symm) && !left) ? j0 - h0 :
                                                                       j0 - i0);
                    int argn = 3;
                    rtSetArg(ctx, argn++, offsetA, "offsetA");
                    rtSetArg(ctx, argn++, offsetB, "offsetB");
                    rtSetArg(ctx, argn++, offsetC, "offsetC");
                    argn += batchArgs;
                    rtSetArg(ctx, argn++, this_lda, "lda");
                    rtSetArg(ctx, argn++, this_ldb, "ldb");
                    rtSetArg(ctx, argn++, this_ldc, "ldc");
                    rtSetArg(ctx, argn++, nm, "m");
                    rtSetArg(ctx, argn++, nn, "n");
                    rtSetArg(ctx, argn++, nkKernel, "k");
                    if (needDiag)
                        rtSetArg(ctx, argn++, diag, "diag");
                    if (info.hpl) {
                        argn++;
                        rtSetArg(ctx, argn++, offsetP, "offsetP");
                    }
                    if (needCO)
                        rtSetArg(ctx, argOffsetCO, offsetCO, "offsetCO");
                    if (needFlags)
                        rtSetArg(ctx, argFlags, this_flags, "flags");

                    size_t threadWS[3] = {size_t(info.subgroupSize), 1, 1};
                    size_t localWS[3]  = {size_t(wgX), size_t(wgY), size_t(wgZ)};
                    size_t globalWS[3] = {size_t(div_up(nm, mb)), size_t(div_up(nn, nb)), size_t(globalWSZ)};

                    // C Interleave: pad up gws[N] to a multiple of the chunk size and add to gws[M] if misaligned ldc
                    globalWS[1] = align_up(globalWS[1], info.cInterleaveChunk(Tc) * wgN);
                    if (info.cInterleaveEnabled() && (offsetC % 64 > 0 || ldc * Tc % 64 > 0)) {
                        auto wgTileM = info.wgTile(LoopM);
                        auto maxShift = 64 / Tc - 1;
                        globalWS[0] += wgM * div_up(wgTileM + maxShift, wgTileM);
                    }

                    if (info.isNMK()) std::swap(globalWS[0], globalWS[1]);

                    if (fixWGK)
                        globalWS[2] = align_up(globalWS[2], localWS[2]);

                    adjustWSFusedEUs(localWS, globalWS, info);
                    adjustWSUniform(driver.Hardware, 3, localWS, globalWS, info, driver.info.UniformWG);

                    if (info.isNestedLinear()) {
                        globalWS[0] = align_up(globalWS[0], div_up(info.blockingAlt[0], mb));
                        globalWS[1] = align_up(globalWS[1], div_up(info.blockingAlt[1], nb));
                    }

                    localWS[1]  *= info.wgExpand;
                    globalWS[1] *= info.wgExpand;

                    auto gcM = uint32_t(globalWS[0] / localWS[0]);
                    auto gcN = uint32_t(globalWS[1] / localWS[1]);
                    auto gcInner = gcM;
                    if (info.isNMK()) std::swap(gcM, gcN);
                    uint32_t groups = gcM * gcN;
                    auto wgTileM = mb * wgM;
                    auto wgTileN = nb * wgN;

                    uint32_t tgPerSS = uint32_t((eusPerSS(driver.Hardware) * threadsPerEU(driver.Hardware, info)) / (localWS[0] * localWS[1] * localWS[2]));
                    if (info.slm > 0)
                        tgPerSS = std::min(tgPerSS, uint32_t(slmCapacity(driver.HardwareProduct) / info.slm));
                    uint32_t maxConcurrentGroups = tgPerSS * (driver.info.EUs / eusPerSS(driver.Hardware));

                    if (PersistentGroupCountOverride) maxConcurrentGroups = PersistentGroupCountOverride;
                    uint32_t maxTileGroups = maxConcurrentGroups / Tiles;

                    if (info.isSimpleLinear())
                        rtSetArg(ctx, argGCMNRecip, irecip(gcInner), "gcMNRecip");
                    else if (info.isNestedLinear()) {
                        auto xgcount = info.isMNK() ? (gcInner * div_up(info.blockingAlt[1], info.wgTile(LoopN)))
                                                    : (gcInner * div_up(info.blockingAlt[0], info.wgTile(LoopM)));
                        rtSetArg(ctx, argGCMNRecip, irecip(xgcount), "gcMNRecip");
                    } else if (info.isHilbert()) {
                        int hilbertBail = (driver.HilbertBailOverride > 0) ? driver.HilbertBailOverride :
                                                          (nk < 64) ? 512
                                                                    : 1;
                        uint32_t vd, uvd, uvdRecip;

                        if (info.triangular && (i0 == j0) && (nm == nn)) {
                            auto c = std::max(wgTileM, wgTileN) / std::min(wgTileM, wgTileN);
                            auto gcU = lower ? gcN : gcM;
                            auto gcV = lower ? gcM : gcN;
                            if ((wgTileM < wgTileN) == lower)
                                groups = gcM * gcN - c * (gcU * (gcU - 1)) / 2;
                            else
                                groups = gcU + c * (gcV * (gcV - 1))/2;
                            vd = uvd = uvdRecip = 0;
                        } else {
                            double bias = (driver.BiasOverride > 0) ? driver.BiasOverride : 1.0;
                            double ratio = gcN / (gcM * bias);
                            if (ratio >= 1) {
                                vd = uint32_t(std::ceil(gcN / std::round(2 * ratio)));
                                uvd = gcM * vd;
                            } else {
                                vd = uint32_t(std::ceil(gcM / std::round(2 / ratio)));
                                uvd = gcN * vd;
                                vd |= 0xFFFF0000;
                            }
                            uvdRecip = irecip(uvd);
                            groups = gcM * gcN;
                        }

                        rtSetArg(ctx, argVD, vd, "vd");
                        rtSetArg(ctx, argUVDRecip, uvdRecip, "uvdRecip");
                        rtSetArg(ctx, argHilbertBail, hilbertBail, "hilbertBail");
                    } else if (info.isBoustrophedon()) {
                        double bias = driver.BiasOverride ? driver.BiasOverride
                                                   : double(wgTileM) / double(wgTileN);
                        double sm = std::max(1., sqrt(maxTileGroups / bias));
                        double sn = std::max(1., sqrt(maxTileGroups * bias));

                        int32_t slice, thresh;

                        for (bool nslice : {gcM > gcN, gcM <= gcN}) {
                            double s = nslice ? sn : sm;
                            if (driver.BoustrophedonSliceOverride) {
                                s = abs(driver.BoustrophedonSliceOverride);
                                if ((driver.BoustrophedonSliceOverride < 0) != nslice)
                                    continue;
                            }

                            // Snap up to full rectangular footprint if possible.
                            int sc = int(ceil(s));
                            if (maxTileGroups % sc == 0)
                                s = sc;
                            sc++;
                            if (maxTileGroups % sc == 0)
                                s = sc;

                            auto gc  = nslice ? gcN : gcM;
                            auto gco = nslice ? gcM : gcN;
                            for (int srange : {0, 1, 2}) {
                                int s0;
                                bool up;
                                if (srange == 0) {
                                    s0 = int(ceil(s));
                                    up = false;
                                } else if (srange == 1) {
                                    s0 = int(ceil(s));
                                    up = true;
                                } else {
                                    s0 = int(floor(s));
                                    up = false;
                                }
                                int s1 = up ? s0 + 1 : s0 - 1;

                                if (s1 <= 0)
                                    continue;

                                int rem = gc % s0;

                                if (!rem)
                                    thresh = gc/s0;
                                else if (up)
                                    thresh = gc/s0 - rem;
                                else
                                    thresh = div_up(gc, s0) - (s0 - rem);
                                bool ok = (thresh >= 0) && (int(gco) >= 2 * std::max(s0, s1));
                                if (!ok)
                                    thresh = gc/s0;

                                slice = s0;
                                if (!up) {
                                    if (thresh > 0)
                                        thresh = -thresh;
                                    else {
                                        slice--;
                                        thresh = gc;
                                    }
                                }
                                if (nslice) slice *= -1;

                                if (ok)
                                    goto success;
                            }
                        }

                        bool nslice;
                        double s;
                        uint32_t gc;

                        nslice = (gcM > gcN);
                        s = nslice ? sn : sm;
                        if (driver.BoustrophedonSliceOverride) {
                            s = abs(driver.BoustrophedonSliceOverride);
                            nslice = (driver.BoustrophedonSliceOverride < 0);
                        }
                        gc = nslice ? gcN : gcM;
                        if (gc < s * 1.5) s = gc;
                        slice = gc / div_up(gc, int(round(s)));
                        thresh = std::max(0u, gc/slice - (gc%slice));
                        if (nslice) slice *= -1;
success:
                        if (slice == 0) {
                            slice = 1;
                            thresh = gcM;
                        }

                        if (ShowBlockInfo) {
                            bool sn = slice < 0;
                            int gcS = sn ? gcN : gcM;
                            int sliceW = abs(slice);
                            int threshGC = abs(thresh) * sliceW;

                            printf("                 Slicing: %d WGs in %c dimension sliced into ", gcS, "mn"[sn]);
                            if (sliceW > gcS)
                                printf("1x%d", gcS);
                            else if (threshGC == gcS)
                                printf("%dx%d", abs(thresh), sliceW);
                            else if (threshGC > gcS)
                                printf("%dx%d + 1x%d", gcS/sliceW, sliceW, gcS % sliceW);
                            else {
                                int rem = gcS - threshGC;
                                int remSliceW = sliceW + ((thresh >= 0) ? 1 : -1);
                                printf("%dx%d + %dx%d", abs(thresh), sliceW, rem/remSliceW, remSliceW);
                                if (rem % remSliceW)
                                    printf(" + 1x%d", rem % remSliceW);
                            }
                            printf("-wide slices\n");
                        }

                        rtSetArg(ctx, argBSlice, slice, "bSlice");
                        rtSetArg(ctx, argBThresh, thresh, "bThresh");
                    }

                    if (info.isLinearOrder()) {
                        if (driver.WGLimit > 0)
                            groups = std::min<uint32_t>(groups, driver.WGLimit);

                        if (info.isPersistent() && info.isHilbert() && info.triangular)
                            rtSetArg(ctx, argGroupCount, groups, "groupCount");

                        if (info.kParallelVariable()) {
                            uint32_t kParallelStart = align_down(groups, maxConcurrentGroups);
                            if (kParallelStart != groups) {
                                uint32_t slicedFull = (KSlicedWaves - 1) * maxConcurrentGroups;
                                if (kParallelStart >= slicedFull)
                                    kParallelStart -= slicedFull;
                                else
                                    kParallelStart = 0;
                            }
                            uint32_t kSlicedTiles = groups - kParallelStart;
                            if (KSlicedTilesOverride >= 0) {
                                kSlicedTiles = std::min<uint32_t>(groups, KSlicedTilesOverride);
                                kParallelStart = groups - kSlicedTiles;
                            }

                            uint32_t tilesPerPhase = div_up(kSlicedTiles, KSlicedPhases);
                            if (KSlicedPhases < 1 || KSlicedPhases > 2)
                                throw std::runtime_error("Only 1 or 2 k-sliced phases allowed");

                            int kPadding = info.kPadding();
                            int oldKPadding = kPadding;
                            int nkPadded = nkKernel;
                            int64_t kTotal = 0;
                            k0 = 0;

                            do {
                                nkPadded = align_up(nkKernel + kPadding, info.unroll[LoopK]);
                                kTotal = int64_t(nkPadded) * tilesPerPhase;
                                if (kTotal == 0) break;
                                k0 = uint32_t(div_up(kTotal, maxConcurrentGroups));
                                k0 = std::max<uint32_t>(k0, div_up(nkKernel, driver.info.EUs / eusPerSS(driver.Hardware)));
                                if (K0Override) k0 = K0Override;
                                k0 = align_up(k0, info.unroll[LoopK]);
                                oldKPadding = kPadding;
                                kPadding = std::min<int>(kPadding, 2 * k0);
                            } while (kPadding != oldKPadding);

                            groups = kParallelStart;
                            uint32_t kParallelGroups = 0;
                            uint32_t kSyncSlabs = 0;

                            if (k0 > 0) {
                                kParallelGroups = uint32_t(div_up(kTotal, k0));
                                if ((!info.isPersistent() && kParallelStart > 0) || KSlicedPhases > 1)
                                    kParallelGroups = maxConcurrentGroups;
                                groups += kParallelGroups * KSlicedPhases;

                                kSyncSlabs = (tilesPerPhase > 0) ? (kParallelGroups + (tilesPerPhase >> 1)) / tilesPerPhase : 0;
                                if (kSyncSlabs > 0) kSyncSlabs--;

                                if (KSyncSlabsOverride >= 0) kSyncSlabs = KSyncSlabsOverride;

                                kSyncSlabs = std::min(kSyncSlabs, (nkKernel - 1) / k0);
                            }

                            uint32_t kUnsyncedPadded = nkPadded - kSyncSlabs * k0;

                            auto plural = [](integer n) { return (n == 1) ? "" : "s"; };
#define PLURAL(v) v, plural(v)
                            if (ShowBlockInfo) {
                                printf("      Variable k-slicing: ");
                                if (k0 > 0) {
                                    printf("slicing final %u tile%s", PLURAL(kSlicedTiles));
                                    if (driver.Loud) printf(": %u phase%s, %u slab%s, k chunk size = %u", PLURAL(KSlicedPhases), PLURAL(kSyncSlabs), k0);
                                    printf("\n");
                                } else
                                    printf("disabled\n");
                            }
#undef PLURAL
                            uint32_t kvConfig = kSlicedTiles | (kSyncSlabs << 16) | ((KSlicedPhases - 1) << 31);

                            rtSetArg(ctx, argK0, k0, "k0");
                            rtSetArg(ctx, argKVConfig, kvConfig, "kvConfig");
                            rtSetArg(ctx, argKUnsyncedRecip, irecip(kUnsyncedPadded), "kUnsyncedRecip");
                        }

                        if (info.isPersistent() || info.kParallelVariable()) {
                            auto concurrentGroups = std::min(groups, maxConcurrentGroups);
                            rtSetArg(ctx, argGroupStride, concurrentGroups, "groupStride");
                            if (info.isPersistent())
                                groups = concurrentGroups;
                        }

                        rtSetArg(ctx, argGroupCountM, gcM, "gcM");
                        rtSetArg(ctx, argGroupCountN, gcN, "gcN");

                        if (info.scrambleM()) {
                            uint32_t wgStrideM = WGStrideM;
                            if (wgStrideM == 0)
                                wgStrideM = transa ? div_up(0x18000, lda * wgTileM * info.typeA()) : 1;
                            while (gcd(wgStrideM, gcM) > 1)
                                wgStrideM++;
                            rtSetArg(ctx, argGCMRecip, irecip(gcM), "gcMRecip");
                            rtSetArg(ctx, argWGStrideM, wgStrideM, "wgStrideM");
                        }
                        if (info.scrambleN()) {
                            uint32_t wgStrideN = WGStrideN;
                            if (wgStrideN == 0)
                                wgStrideN = transb ? 1 : div_up(0x18000, ldb * wgTileN * info.typeB());
                            while (gcd(wgStrideN, gcN) > 1)
                                wgStrideN++;
                            rtSetArg(ctx, argGCNRecip, irecip(gcN), "gcNRecip");
                            rtSetArg(ctx, argWGStrideN, wgStrideN, "wgStrideN");
                        }

                        globalWS[0] = localWS[0] * (groups + info.extraWGs());
                        globalWS[1] = localWS[1];
                    }

                    ShowBlockInfo = false;

                    localWS[0]  *= threadWS[0];  localWS[1]  *= threadWS[1];  localWS[2]  *= threadWS[2];
                    globalWS[0] *= threadWS[0];  globalWS[1] *= threadWS[1];  globalWS[2] *= threadWS[2];

                    rt_event *depEvent = &curEvents[i0b + j0b * nblock_m];
                    rt_event *outEvent = depEvent;

#ifdef TARGET_OPENCL
                    bool wait = (h0b_ > 0) && !driver.GenerateAUB;
                    rt_event thisEvent = nullptr;
                    rt_event *inEvents = wait ? depEvent : nullptr;

                    clWrapEnqueueNDRangeKernel(ctx, phase, kinfo.kernelName(), queue, kernel, 3, nullptr, globalWS, localWS, wait, inEvents, driver.GenerateAUB ? nullptr : &thisEvent);
                    if (!driver.GenerateAUB)
                        allEvents.push_back(thisEvent);
                    *outEvent = thisEvent;
#endif

#ifdef TARGET_LEVEL_ZERO
                    bool wait = (h0b_ > 0);
                    if (wait) {
                        if (TimestampEvents)
                            depEvent = &curEvents[i0b + j0b * nblock_m + (h0b_ - 1) * nblock_m * nblock_n];
                        handle(zeCommandListAppendWaitOnEvents(list, 1, depEvent));
                        if (!TimestampEvents)
                            handle(zeCommandListAppendEventReset(list, *depEvent));
                    }
                    if (TimestampEvents)
                        outEvent = &curEvents[i0b + j0b * nblock_m + h0b_ * nblock_m * nblock_n];
                    zeWrapCommandListAppendLaunchKernel(ctx, phase, kinfo.kernelName(), list, kernel, 3, globalWS, localWS, *outEvent);
#endif
                }
            }
        }

#if TARGET_OPENCL
        if (!driver.GenerateAUB)
            RECORD(GPUWaitTime, clWrapWaitForEvents(cl_uint(nblock_m * nblock_n), curEvents.data()););

        end = dsecnd();
        result.hostTime = end - start;
        result.deviceTime = rtProfileDuration(allEvents);
        if (driver.getEnableMetrics())
            rtCollectMetrics(args, allEvents, result);

        for (auto &e: allEvents)
            handle(clReleaseEvent(e));

        handle(clFinish(queue));
#endif

#if TARGET_LEVEL_ZERO
        rtEndTimestamps(driver, list, int(nblock_m * nblock_n), curEvents.data());
        if (!TimestampEvents) for (auto &event: curEvents)
            handle(zeCommandListAppendEventReset(list, event));
        if (!ImmediateCommandLists)
            handle(zeCommandListClose(list));
#endif
    } // end list prepare/run phase

#if TARGET_LEVEL_ZERO
    if (phase == DriverPhase::RunList || phase == DriverPhase::Run) {
        rtRunList(driver, result);
        if (phase == DriverPhase::Run && ImmediateCommandLists) {
            end = dsecnd();
            result.hostTime = end - start;
        }
    }
#endif

    if (phase == DriverPhase::Finalize) {
        if (!driver.ZeroCopy && !timeOnly) {
            auto cOff = size_t(AddrOffset[2] * Tc);
            rtReadBuffer(args, Cs_buffer, Cs, cOff);
            Cs.unsmash();
            if (copyC)
                packUnpackC(false);
            if (info.hpl) {
                rtReadBuffer(args, P_buffer, P.data);
                P.data.unsmash();
            }
            if (info.sumA || info.sumB) {
                rtReadBuffer(args, CO_buffer, CO.data);
                CO.data.unsmash();
            }
            if (info.fusedBeta() || info.fusedPostOps()) {
                auto statusHost = alloc<uint32_t>(statusElems * 16);
                rtReadBuffer(args, statusBuffer, 0, statusElems * 64, statusHost);
                bool ok = true;
                for (integer i = 0; i < statusElems; i++)
                    if (statusHost[i * 16])
                        ok = false;
                if (!ok) {
                    yprintf("Warning: status counters were not reset to zero.\n");
                    for (integer i = 0; i < statusElems; i++)
                        if (statusHost[i * 16])
                            yprintf("status[%d] = %u\n", i, statusHost[i * 16]);
                }
                dealloc(statusHost, statusElems * 16);
            }
            if (info.zeroTempC()) {
                auto tempCHost = alloc<uint8_t>(tempCBytes);
                rtReadBuffer(args, tempCBuffer, 0, tempCBytes, tempCHost);
                integer badIdx = -1;
                for (integer i = 0; i < tempCBytes; i++) {
                    if (tempCHost[i] != 0) {
                        badIdx = i; break;
                    }
                }
                if (badIdx >= 0) {
                    yprintf("Warning: temporary C buffer was not reset to zero.\n");
                    yprintf("First nonzero element at byte [%ld].\n", badIdx);
                }
                dealloc(tempCHost, tempCBytes);
            }
        }

        if (info.tracking) {
            storage.LastTrackingData.resize(trackingElems);
            storage.LastTrackingStride = int(trackingStride);
            storage.LastTrackingWGSize = int(trackingWGSize);
            rtReadBuffer(args, trackingBuffer, 0, trackingElems * sizeof(Timepoint), storage.LastTrackingData.data());
        }

#if TARGET_LEVEL_ZERO
        clearLevelZeroData(args);
#endif
    } // end finalize phase

    result.recordIteration();
    return result;
}

TestResult gemmDriverDispatch(DriverPhase phase, DriverRT &driver,
                              KernelInfo &kinfo, bool timeOnly,
                              bool conja, bool conjb, char cOffType,
                              superscalar alpha, superscalar beta,
                              Matrix &A, Matrix &B, Matrix &C, Matrix &P, Matrix &CO,
                              Matrix &AO, Matrix &BO, Matrix &Ascale, Matrix &Bscale, Matrix &Ag, Matrix &Bg,
                              const int (&AddrOffset)[3])
{
    return gemmDriver(phase, driver, kinfo, timeOnly,
                      conja, conjb, cOffType, alpha, beta,
                      A, B, C, P, CO, AO, BO, Ascale, Bscale, Ag, Bg,
                      AddrOffset);
}

TestResult copyDriver(DriverPhase phase, DriverRT &driver,
                      KernelInfo &kinfo, bool timeOnly,
                      superscalar alpha, Matrix &S, Matrix &D)
{
    auto &args = driver.rtArgs;
    auto &storage = args.storage;
    TestResult result(driver.info.ClockFrequency);
#if TARGET_OPENCL
    auto queue = args.queue;
#endif
    auto kernel = args.kernel;

    auto m = S.m, n = S.n;
    auto lds = S.ld, ldd = D.ld;

    auto &info = kinfo.copy[0];
    auto Ts = info.typeS(), Td = info.typeD();
    bool trans = (info.layoutS == 'T');
    bool inner = info.inner();

    auto unrollX = info.unrollX();
    auto unrollY = info.unrollY();
    bool xLoop = info.xLoop();

    auto x     = inner ? m : n;
    auto y     = inner ? n : m;
    auto ypad  = D.mpad;
    auto y0pad = y;
    auto z     = xLoop ? x : y;

    if (info.trsm && xLoop) y0pad = std::min<integer>(ypad, align_up(y0pad, unrollX));

    double start = 0;

    // Persistent data.
    auto &S_buffer = storage.S_buffer;
    auto &D_buffer = storage.D_buffer;

#if TARGET_LEVEL_ZERO
    auto &list = storage.list;
    auto &curEvents = storage.curEvents;

    double end;
#else
    rt_list list = nullptr;
#endif

    // Handle blocking.
    auto Block = getBlocking(driver.Hardware, driver.info.EUs, x, y0pad, info, driver.NoBlocking, driver.NBlockOverrides);
    auto BlockY = y0pad;
    int nblock_y = 1, nblock_z = 1;

    if (!info.zParallel()) {
        nblock_y = div_up(y0pad, Block);
        BlockY = int32_t(Block);
        if (ShowBlockInfo && phase == DriverPhase::Init)
            std::cout << "            Block size: " << Block << " (" << nblock_y << " blocks)" << std::endl;
    } else {
        if (info.zBlockFixed())
            Block = xLoop ? unrollX : unrollY;
        nblock_z = div_up(z, Block);
        if (ShowBlockInfo && phase == DriverPhase::Init)
            std::cout << "            Block size: " << Block << " (" << nblock_z << " blocks)" << std::endl;
    }

    if (phase == DriverPhase::Init) {
        rtClearColdCache(storage);

        // Prepare buffers.
        rtCreateBuffer(driver, BUF_R  | driver.ExtraFlagsS, S_buffer, S.data);
        rtCreateBuffer(driver, BUF_RW | driver.ExtraFlagsD, D_buffer, D.data);
    } /* end persistent init phase */

#ifdef TARGET_LEVEL_ZERO
    if (phase == DriverPhase::Init) {
        list = rtCreateList(args);
        rtCreateEvents(args, nblock_y);
        rtAllocTimestamps(args);
    }
#endif

    if (phase == DriverPhase::PrepareList || phase == DriverPhase::Run) {
        rtKernelCtx ctx{kernel};
        // Set arguments.
        int64_t offsetS = 0, offsetD = 0;
        int32_t diag = 0;

        rtSetArg(ctx, 0, S_buffer, "S");
        rtSetArg(ctx, 1, D_buffer, "D");
        //rtSetArg(ctx, 2, offsetS);
        //rtSetArg(ctx, 3, offsetD);
        rtSetArg(ctx, 4, lds, "lds");
        rtSetArg(ctx, 5, ldd, "ldd");
        //rtSetArg(ctx, 6, m);
        //rtSetArg(ctx, 7, n);
        rtSetArg(ctx, 8, Ts, alpha.real(), "alpha");
        rtSetArg(ctx, 9, Ts, alpha.imag(), "alpha_i");
        //if (info.nongeneral())
        //    rtSetArg(ctx, 10, diag);
        if (info.zParallel()) {
            auto blockZ = uint32_t(Block);
            rtSetArg(ctx, info.nongeneral() ? 11 : 10, blockZ, "blockZ");
        }

#ifdef TARGET_LEVEL_ZERO
        rtMakeTimestampsDeviceResident(args);
#endif

        prepareCache(driver, list);

        if (phase == DriverPhase::Run && rtImmediateExecution())
            start = dsecnd();

#ifdef TARGET_OPENCL
        // TODO: device timing
        std::vector<rt_event> events(nblock_y, nullptr);
#endif

#ifdef TARGET_LEVEL_ZERO
        rtStartTimestamps(driver, list);
#endif

        for (int h0b = 0; h0b < nblock_y; h0b++) {
            int h0 = int(h0b * Block);
            int ny     = std::min<int>(y     - h0, BlockY);
            int ny0pad = std::min<int>(y0pad - h0, BlockY);
            int32_t nm = m, nn = n;

            offsetS = (inner ^ (info.layoutS == 'N')) ? h0 : (h0 * lds);
            offsetD = info.packSize * h0;
            (inner ? nn : nm) = ny;
            diag = inner ? h0 : -h0;

            rtSetArg(ctx, 2, offsetS, "offsetS");
            rtSetArg(ctx, 3, offsetD, "offsetD");

            rtSetArg(ctx, 6, nm, "m");
            rtSetArg(ctx, 7, nn, "n");

            if (info.nongeneral())
                rtSetArg(ctx, 10, diag, "diag");

            auto nker = xLoop ? div_up(ny0pad, info.unrollY()) : div_up(x, info.unrollX());

            int tgSize = 16;
            applyWGOverrides(info, driver.WGOverride, &tgSize);

            size_t threadWS[2] = {size_t(info.subgroupSize),    1};
            size_t globalWS[2] = {size_t(nker),                 size_t(nblock_z)};
            size_t localWS[2]  = {size_t(tgSize),               1};

            adjustWSFusedEUs(localWS, globalWS, info);
            adjustWSUniform(driver.Hardware, 2, localWS, globalWS, info, driver.info.UniformWG);

            localWS[0]  *= threadWS[0];  localWS[1]  *= threadWS[1];
            globalWS[0] *= threadWS[0];  globalWS[1] *= threadWS[1];

#ifdef TARGET_OPENCL
            clWrapEnqueueNDRangeKernel(ctx, phase, kinfo.kernelName(), queue, kernel, 2, nullptr, globalWS, localWS,
                                       0, nullptr, driver.GenerateAUB ? nullptr : &events[h0b]);
#endif

#ifdef TARGET_LEVEL_ZERO
            zeWrapCommandListAppendLaunchKernel(ctx, phase, kinfo.kernelName(), list, kernel, 2, globalWS, localWS, curEvents[h0b]);
#endif
        }

#ifdef TARGET_OPENCL
        if (!driver.GenerateAUB)
            RECORD(GPUWaitTime, clWrapWaitForEvents(nblock_y, events.data()));

        result.hostTime = dsecnd() - start;
        if (!driver.GenerateAUB) {
            result.deviceTime = rtProfileDuration(events);
            if (driver.getEnableMetrics())
                rtCollectMetrics(args, events, result);
        }

        for (auto &event: events)
            if (event)
                handle(clReleaseEvent(event));

        handle(clFinish(queue));
#endif

#ifdef TARGET_LEVEL_ZERO
        rtEndTimestamps(driver, list, nblock_y, curEvents.data());
        if (!TimestampEvents) for (auto &event: curEvents)
            handle(zeCommandListAppendEventReset(list, event));
        if (!ImmediateCommandLists)
            handle(zeCommandListClose(list));
#endif
    } /* end list/run phase */

#if TARGET_LEVEL_ZERO
    if (phase == DriverPhase::RunList || phase == DriverPhase::Run) {
        rtRunList(driver, result);
        if (phase == DriverPhase::Run && ImmediateCommandLists) {
            end = dsecnd();
            result.hostTime = end - start;
        }
    }
#endif

    if (phase == DriverPhase::Finalize) {
        if (!driver.ZeroCopy && !timeOnly)
            rtReadBuffer(args, D_buffer, D.data);

#if TARGET_LEVEL_ZERO
        clearLevelZeroData(args);
#endif
    }

    result.recordIteration();
    return result;
}

template <typename fp>
void _checkL3(DriverRT &driver, TestResult &result, const KernelInfo &kinfo,
              const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &CO,
              const Matrix &P, const Matrix &C_ref, const Matrix &CO_ref,
              double maxAbsErrBase, double maxAbsErrGEMM, int maxULPErr)
{
    using fp_real = real_t<fp>;
    using fp_print = printable_t<fp>;
    using fp_diff = diff_t<fp>;

    auto &info = kinfo.l3[0];

    bool readCheck = info.readCheck;
    bool hpl = info.hpl, trsm = info.trsm;
    bool left = info.left, lower = info.lower;
    bool transc = C.trans;
    auto batchSize = C.batch;
    auto m = C.m, n = C.n, ldc = C.ld;
    auto mpad = C.mpad, npad = C.npad;

    auto mb = kinfo.mbAlign();
    auto nb = kinfo.nbAlign();

    auto p = C.data.as<fp>();
    auto pref = C_ref.data.as<fp>();

    fp rcOKVal = 0;
    if (info.strongReadCheck) {
        fp srcOKVal;
        std::memset(&srcOKVal, ~0, sizeof(rcOKVal));
        rcOKVal = srcOKVal;
    }

    auto xpad = !transc ? mpad : npad;
    auto ypad = !transc ? npad : mpad;

    int inaccuracies = 0;

    for (int32_t b = 0; b < batchSize; b++) {
        for (integer y = 0; y < ypad; y++) {
            for (integer x = 0; x < xpad; x++) {
                auto i = !transc ? x : y;
                auto j = !transc ? y : x;
                auto idx = C.offset(i, j, b);
                fp cVal = p[idx];
                fp cRefVal = pref[idx];

                // Compute per-element error tolerance.
                double maxAbsErr;
                if (readCheck || TestDeterminism)
                    maxAbsErr = 0;
                else if (!trsm || driver.OneFill)
                    maxAbsErr = maxAbsErrGEMM;
                else {
                    integer a = left ? (lower ? i : (m - 1 - i))
                                     : (lower ? (n - 1 - j) : j);
                    if (a < 0) a = 0;
                    a++;
                    maxAbsErr = maxAbsErrBase * a * a;
                }

                // Check C matrix.
                bool close = checkNearby(cVal, cRefVal, maxAbsErr);
                if (readCheck || TestDeterminism)
                    close |= identicalBits(cVal, rcOKVal);
                if (maxULPErr > 0)
                    close |= checkULPNearby(cVal, cRefVal, maxULPErr);

                if (!close) {
                    bool print = !driver.Quiet && (inaccuracies++ <= MaxInaccuracies);
                    if (print && !driver.Loud && inaccuracies == 1) std::cerr << std::endl;
                    if (i >= m || j >= n || readCheck) {
                        result.markAccuracy(TestAccuracy::Overrun);
                        if (print) std::cerr << "Overrun";
                    } else {
                        result.markAccuracy(TestAccuracy::Inaccurate);
                        if (print) std::cerr << "Inaccuracy";
                    }
                    if (print) {
                        if (readCheck)
                            std::cerr << " in block starting";
                        std::cerr << " at ";
                        if (info.batch)
                            std::cerr << "matrix " << b << ' ';
                        std::cerr << '(' << i << ',' << j << ") : computed " << fp_print(cVal) << " versus " << fp_print(cRefVal);
                        if constexpr (!isIntegral<fp>()) {
                            fp_diff diff = cVal - cRefVal;
                            if (std::abs(diff) < 1e-4)
                                std::cerr << " diff " << diff;
#ifdef COMPARE_NEARBY
                            for (int ii = 0; ii < 32; ii++)
                                for (int jj = 0; jj < 16; jj++) {
                                    auto ni = (i & -32) + ii;
                                    auto nj = (j & -16) + jj;
                                    if (ni != i || nj != j)
                                        if (std::abs(C_ref[ni + nj * ldc] - cVal) < 1e-4)
                                            std::cerr << " cf. [" << ni << ',' << nj << ']';
                                }
#endif
                        }
                        std::cerr << std::endl;
                        if (inaccuracies > MaxInaccuracies) {
                            std::cerr << "MaxInaccuracies limit reached; there may be more inaccurate entries.\n";
                            return;
                        }
                    }
                }

                // Checked extra packed output (P for HPL, A/B for copy-based TRSM).
                if (hpl || (trsm && info.inplace)) {
                    fp pVal = (hpl ? P : left ? B : A).get<fp>(i, j, b);

                    if (!checkNearby(pVal, cVal, 0)) {
                        bool print = !driver.Quiet && (inaccuracies++ <= MaxInaccuracies);
                        if (i >= m || j >= n) {
                            if (hpl || i >= align_up(m, mb) || j >= align_up(n, nb)) {
                                result.markAccuracy(TestAccuracy::Overrun);
                                if (print && !driver.Loud && inaccuracies == 1) std::cerr << std::endl;
                                if (print) std::cerr << "Packed output overrun";
                            } else {
                                inaccuracies--;
                                print = false;
                            }
                        } else {
                            result.markAccuracy(TestAccuracy::Inaccurate);
                            if (print && !driver.Loud && inaccuracies == 1) std::cerr << std::endl;
                            if (print) std::cerr << "Packed output inaccuracy";
                        }
                        if (print) {
                            std::cerr << " at ";
                            if (info.batch)
                                std::cerr << "matrix " << b << ' ';
                            std::cerr << '(' << i << ',' << j << ") : computed ";
                            std::cerr << fp_print(pVal) << " versus " << fp_print(cVal) << std::endl;
                        }
                    }
                }

                // Check A/B sums.
                if (((info.sumA && j == 0) || (info.sumB && i == 0)) && !readCheck) {
                    fp coVal = CO.data.as<fp>()[i + j];
                    fp coRefVal = CO_ref.data.as<fp>()[i + j];
                    bool close = checkNearby(coVal, coRefVal, maxAbsErr);
                    if (maxULPErr > 0)
                        close |= checkULPNearby(coVal, coRefVal, maxULPErr);
                    if (!close) {
                        bool print = !driver.Quiet && (inaccuracies++ <= MaxInaccuracies);
                        if (i >= m || j >= n) {
                            result.markAccuracy(TestAccuracy::Overrun);
                            if (print) std::cerr << "Overrun";
                        } else {
                            result.markAccuracy(TestAccuracy::Inaccurate);
                            if (print) std::cerr << "Inaccuracy";
                        }
                        if (print) {
                            std::cerr << " at " << "AB"[info.sumB] << " sum (" << i+j << ") : computed "
                                      << fp_print(coVal) << " versus " << fp_print(coRefVal) << std::endl;
                        }
                    }
                }
            }
        }
    }
}

TYPE_TEMPLATIFY(checkL3)

void checkL3(DriverRT &driver, TestResult &result, const KernelInfo &kinfo, integer k,
             superscalar alpha, superscalar beta, const Matrix &A, const Matrix &B,
             const Matrix &C, const Matrix &CO, const Matrix &P, const Matrix &C_ref,
             const Matrix &CO_ref)
{
    auto &info = kinfo.l3[0];
    auto Ta = info.typeA(), Tb = info.typeB(), Tc = info.typeC(), Tc_int = info.typeCInt();
    auto Tao = info.typeAO(), Tbo = info.typeBO();
    auto m = C.m, n = C.n;

    int inaccuracies = 0;

    bool nontrivialScalars = (alpha != 1) || ((beta != 1) && (beta != 0));
    char precisionC = info.precisionC, precisionCInt = info.precisionCInt;

    if (info.relaxedAccumulation)
        precisionCInt = precisionC;

    // A/B conversion and FMA errors.
    double maxAbsErrBase = epsilon(precisionCInt);
    if (info.scaleA) maxAbsErrBase = std::max({maxAbsErrBase, epsilon(info.precisionAInt), epsilon(info.precisionAScale)});
    if (info.scaleB) maxAbsErrBase = std::max({maxAbsErrBase, epsilon(info.precisionBInt), epsilon(info.precisionBScale)});
    maxAbsErrBase *= Tc.complexComponents()
                  *  std::max(precisionLoss(info.precisionA, info.precisionAInt, info.A_components),
                              precisionLoss(info.precisionB, info.precisionBInt, info.B_components));

    if (Ta.isInteger()) {
        int range = 1 << (4 * Ta.paddedSize());
        if (info.aOffset)
            range += (1 << (4 * Tao.paddedSize()));
        maxAbsErrBase *= range;
    }
    if (Tb.isInteger()) {
        int range = 1 << (4 * Tb.paddedSize());
        if (info.bOffset)
            range += (1 << (4 * Tbo.paddedSize()));
        maxAbsErrBase *= range;
    }

    if (Ta.isFP() && Tb.isFP() && nontrivialScalars) {
        maxAbsErrBase *= std::max(std::abs(alpha), std::abs(beta)) * 20;
        if (m == 1 || n == 1)
            maxAbsErrBase *= 2;    // accommodate additional gemv roundoff error in MKL
    }

    // Accumulation chain error.
    double maxAbsErrGEMM = 4 * std::max<integer>(k, 1) * maxAbsErrBase;

    // Final downconversion error.
    int maxULPErr = 0;
    if (precisionC != precisionCInt) {
        maxULPErr++;
        if (info.cBias || info.cOffset) {
            maxULPErr++;    /* accommodate second downconversion in reference path */
            if (nontrivialScalars) maxULPErr += 2;
            maxAbsErrGEMM = std::max(maxAbsErrGEMM, epsilon(precisionC));
        }
        if (info.sumA || info.sumB) {
            if (nontrivialScalars) maxULPErr = 4;
            maxAbsErrGEMM = std::max(maxAbsErrGEMM, epsilon(precisionC));
        }
        if (Tc.isInteger() && Tc_int.isInteger())
            maxULPErr = 0;
    }

    if (auto e = getenv("MAX_ABS_ERROR"))
        maxAbsErrGEMM = atof(e);

    // Call type-templated check code.
    result.markAccuracy(TestAccuracy::OK);
    checkL3(Tc, driver, result, kinfo, A, B, C, CO, P, C_ref, CO_ref, maxAbsErrBase, maxAbsErrGEMM, maxULPErr);
}

TestResult testL3(DriverRT &driver, KernelInfo &kinfo, TestMode mode,
                  int32_t batchSize, int32_t m, int32_t n, int32_t k,
                  int32_t lda, int32_t ldb, int32_t ldc,
                  bool conja, bool conjb, bool unit,
                  char cOffType, bool aOffset, bool bOffset)
{
    auto &args = driver.rtArgs;
    TestResult result(driver.info.ClockFrequency);
    bool accuracy = (mode == TestMode::Accuracy);
    bool time = (mode == TestMode::Time || mode == TestMode::TimeOneOfSeries);

    bool forceTestInPlace = false;
    auto forceTIP = getenv("FORCE_INPLACE");
    if (forceTIP) forceTestInPlace = atoi(forceTIP);

    bool inplace = driver.TestInPlace || forceTestInPlace;

    auto &info = kinfo.l3[0];
    bool readCheck = info.readCheck;

    if (!accuracy && readCheck)
        return result;  /* Don't time read check kernels */

    OverrunTest = accuracy;
    ShowBlockInfo = false;

    auto Ta = info.typeA(), Tb = info.typeB(), Tc = info.typeC(), Ts = info.typeS();
    auto Ta_int = info.typeAInt(), Tb_int = info.typeBInt();

    bool transa = (info.layoutA == 'T');
    bool transb = (info.layoutB == 'T');
    bool transc = (info.layoutC == 'T');

    if (driver.DoTransposeOverride) {
        transa = driver.TransposeOverride[0];
        transb = driver.TransposeOverride[1];
    }

    int AddrOffset[3];
    for (int i = 0; i < 3; i++)
        AddrOffset[i] = driver.AddrOffsetOverride[i] ? driver.AddrOffsetOverride[i] : 0;

    bool trsm  = info.trsm;
    bool trmm  = info.trmm;
    bool symm  = info.symm;
    bool hemm  = info.hemm;
    bool left  = info.left;
    bool lower = info.lower;
    unit |= info.unit;

    if (trsm && transc)
        stub();

    /* Size adjustments and padding */
    int32_t strideA = 0, strideB = 0, strideC = 0;
    int32_t mpad, npad, kpad;

    auto mb = kinfo.mbAlign();
    auto nb = kinfo.nbAlign();
    auto kb = kinfo.kbAlign();

    if (trsm || trmm || symm || hemm)
        k = left ? m : n;
    if (transa && Ta.perByte() > 1)
        k = align_up(k, Ta.perByte());
    if (!transb && Tb.perByte() > 1)
        k = align_up(k, Tb.perByte());

    if (readCheck) {
        mpad = align_up(m + 64, mb * MaxWGM);
        npad = align_up(n + 64, nb * MaxWGN);
        kpad = align_up(k + 64, kb);
    } else if (OverrunTest && !driver.SpecifiedLDs) {
        mpad = align_up(m, mb);
        npad = align_up(n, nb);
        kpad = std::max(k, 1);
    } else {
        mpad = m;
        npad = n;
        kpad = std::max(k, 1);
    }

    bool needCO = info.cOffset || info.cBias || info.sumA || info.sumB;
    bool coIsCol = (cOffType == 'R') || (cOffType == 'M') || info.sumA;
    bool coIsRow = (cOffType == 'C') || (cOffType == 'M') || info.sumB;

    int32_t mCO    = coIsCol ? m    : 1;
    int32_t nCO    = coIsRow ? n    : 1;
    int32_t mpadCO = coIsCol ? mpad : 1;
    int32_t npadCO = coIsRow ? npad : 1;

    bool transco = coIsRow && !coIsCol;

    /* Leading dimension setting. */
    int32_t min_lda = transa ? kpad : mpad;
    int32_t min_ldb = transb ? npad : kpad;
    int32_t min_ldc = transc ? npad : mpad;

    if (lda < 0) lda = niceLD(driver.Hardware, Ta, min_lda) + (-1 - lda);
    if (ldb < 0) ldb = niceLD(driver.Hardware, Tb, min_ldb) + (-1 - ldb);
    if (ldc < 0) ldc = niceLD(driver.Hardware, Tc, min_ldc) + (-1 - ldc);

    lda = std::max(lda, min_lda) + driver.OddLD[0];
    ldb = std::max(ldb, min_ldb) + driver.OddLD[1];
    ldc = std::max(ldc, min_ldc) + driver.OddLD[2];

    int32_t ldco = transco ? npadCO : mpadCO;
    if (cOffType == 'M') ldco += driver.OddLD[2];

    int alignAElems = std::max(1, info.alignA / Ta);
    int alignBElems = std::max(1, info.alignB / Tb);
    int alignCElems = std::max(1, info.alignC / Tc);

    lda = align_up(lda, alignAElems);
    ldb = align_up(ldb, alignBElems);
    ldc = align_up(ldc, alignCElems);

    auto ldp = transc ? align_up(npad, nb) : align_up(mpad, mb);

    if (info.batch) {
        strideA = int32_t(Matrix::elements(transa, lda, mpad, kpad)) + driver.OddStride[0];
        strideB = int32_t(Matrix::elements(transb, ldb, kpad, npad)) + driver.OddStride[1];
        strideC = int32_t(Matrix::elements(transc, ldc, mpad, npad)) + driver.OddStride[2];
        strideA = align_up(strideA, alignAElems);
        strideB = align_up(strideB, alignBElems);
        strideC = align_up(strideC, alignCElems);
    } else
        batchSize = 1;

    if (!driver.Quiet && (time || driver.Loud)) {
        std::cout << "Size is ";
        if (info.batch)
            std::cout << batchSize << " x ";
        std::cout << m << " x " << n << " x " << k;
        std::cout << ", leading dimensions " << lda << ", " << ldb << ", " << ldc;
        if (info.hpl)
            std::cout << " (" << ldp << ')';
        if (info.batch && (driver.OddStride[0] || driver.OddStride[1] || driver.OddStride[2]))
            std::cout << ", strides " << strideA << ", " << strideB << ", " << strideC;
        std::cout << std::endl;
    }

    Matrix A, B, C, C_ref, CO, CO_ref, P, AO, A_scale, Ag, BO, B_scale, Bg;

    auto &U = left ? A : B;

    superscalar alpha = info.alpha, beta = info.beta;

    // For fractional igemm scaling, pick alpha and beta to avoid integer + 0.5 values.
    bool f2i = Tc.isInteger() && Ts.isFP();
    if (!info.fixedAlpha()) {
        alpha = f2i ? makeComplex<superscalar>(2.3333333333333333, -0.6666666666666666)
                    : makeComplex<superscalar>(2.3, -0.6);
    }
    if (!info.fixedBeta()) {
        beta = driver.TestBeta0 ? superscalar(0) :
                     f2i ? makeComplex<superscalar>(-1.6666666666666666, 0.6666666666666666)
                         : makeComplex<superscalar>(-1.7, 0.8);
    }

    if ((info.kParallel() || info.kParallelVariable()) && !info.fusedBeta() && !info.zeroTempC() && !driver.TestNonKParallelGlobal)
        beta = 1;

    if (auto e = getenv("ALPHA")) alpha = atof(e);
    if (auto e = getenv("BETA"))  beta  = atof(e);

    downcast(alpha, Ts);
    downcast(beta, Ts);

    // Initialize A/B/C.
    bool doAlloc = (mode == TestMode::Accuracy || mode == TestMode::Time || mode == TestMode::BeginSeries);

    A = Matrix(Ta, transa, m, k, lda, mpad, kpad, batchSize, strideA, doAlloc);
    B = Matrix(Tb, transb, k, n, ldb, kpad, npad, batchSize, strideB, doAlloc);
    C = Matrix(Tc, transc, m, n, ldc, mpad, npad, batchSize, strideC, doAlloc);
    C_ref = Matrix(Tc, C, doAlloc);
    if (needCO) {
        CO = Matrix(Tc, transco, mCO, nCO, ldco, mpadCO, npadCO, doAlloc);
        if (info.sumA || info.sumB)
            CO_ref = Matrix(Tc, CO, doAlloc);
    }
    if (info.scaleA) {
        auto mm = div_up(m, info.aqGroup[0]);
        auto kk = div_up(k, info.aqGroup[1]);
        A_scale = Matrix(info.typeAs(), false, mm, kk, mm, doAlloc);
    }
    if (info.scaleB) {
        auto kk = div_up(k, info.bqGroup[0]);
        auto nn = div_up(n, info.bqGroup[1]);
        B_scale = Matrix(info.typeBs(), false, kk, nn, kk, doAlloc);
    }

    int Ag_groupK = 0, Bg_groupK = 0;
    if (info.needsSeparateASums()) {
        Ag_groupK = info.needsAGroupSums() ? info.aqGroup[1] : k;
        auto kk = div_up(k, Ag_groupK);
        Ag = Matrix(info.typeAg(), false, m, kk, m, doAlloc);
    }
    if (info.needsSeparateBSums()) {
        Bg_groupK = info.needsBGroupSums() ? info.bqGroup[0] : k;
        auto kk = div_up(k, Bg_groupK);
        Bg = Matrix(info.typeBg(), false, kk, n, kk, doAlloc);
    }

    auto makeOffsetMatrix = [doAlloc](bool isA, char type, Type T, int32_t r, int32_t c, std::array<int32_t, 2> group) {
        int ro = 1, co = 1;
        switch (type) {
            case 'v':
                if (isA)
                    ro = r;
                else
                    co = c;
                break;
            case 'g':
                ro = div_up(r, group[0]);
                co = div_up(c, group[1]);
                break;
            default: break;
        }
        return Matrix(T, false, ro, co, ro, doAlloc);
    };

    if (info.aOffset) AO = makeOffsetMatrix(true,  info.aOffset, info.typeAO(), m, k, info.aqGroup);
    if (info.bOffset) BO = makeOffsetMatrix(false, info.bOffset, info.typeBO(), k, n, info.bqGroup);


    if (doAlloc) {
        if (!readCheck) {
            // Regular initialization.
            auto A_name = MatrixName::GEMM_A;
            auto B_name = MatrixName::GEMM_B;
            auto C_name = MatrixName::GEMM_C;
            auto CO_name = MatrixName::GEMM_CO;

            if (info.sumA || info.sumB)
                CO_name = MatrixName::GEMM_C;

            if (mode == TestMode::BeginSeries || mode == TestMode::Time) {
                auto seriesName = driver.ZeroFill ? MatrixName::SERIES_ZERO :
                                   driver.OneFill ? MatrixName::SERIES_ONE
                                           : MatrixName::SERIES_RANDOM;
                A_name = B_name = C_name = CO_name = seriesName;
            }

            if (trsm) {
                if (left) {
                    A_name = unit ? (lower ? MatrixName::TRSMLLU_U : MatrixName::TRSMLUU_U)
                                  : (lower ? MatrixName::TRSMLL_U  : MatrixName::TRSMLU_U);
                    B_name = lower ? MatrixName::TRSMLL_V : MatrixName::TRSMLU_V;
                } else {
                    B_name = unit ? (lower ? MatrixName::TRSMRLU_U : MatrixName::TRSMRUU_U)
                                  : (lower ? MatrixName::TRSMRL_U  : MatrixName::TRSMRU_U);
                    A_name = lower ? MatrixName::TRSMRL_V : MatrixName::TRSMRU_V;
                }
            }

            parseNameOverride(A_name, "A");
            parseNameOverride(B_name, "B");
            parseNameOverride(C_name, "C");
            parseNameOverride(CO_name, "CO");

            A.fill(A_name, fillMods(driver));
            B.fill(B_name, fillMods(driver));

            if (!inplace)
                C.paddedView().fill(C_name, fillMods(driver));
            else
                copy(C, left ? B : A);

            if (needCO)   CO.fill(CO_name, fillMods(driver));
            if (info.hpl) P = C;

            if (trmm)
                U.paddedView().makeTriangular(lower);
            else if (symm)
                U.makeSymmetric(lower);
            else if (hemm)
                U.makeHermitian(lower);

            if (info.aOffset) {
                bool small = (Ta.isSubByteInt() && Ta_int.isInt8()) /* avoid int4 - int8 wraparound */
                          || (Ta_int == Type::bf16);                /* avoid round-off */
                auto vals = !aOffset ? MatrixName::ZERO :
                               small ? MatrixName::AO_SMALL
                                     : MatrixName::AO;
                parseNameOverride(vals, "AO");
                AO.fill(vals, fillMods(driver));
            }
            if (info.bOffset) {
                bool small = (Tb.isSubByteInt() && Tb_int.isInt8())
                          || (Tb_int == Type::bf16);
                auto vals = !bOffset ? MatrixName::ZERO :
                               small ? MatrixName::BO_SMALL
                                     : MatrixName::BO;
                parseNameOverride(vals, "BO");
                BO.fill(vals, fillMods(driver));
            }
            if (info.scaleA) {
                auto vals = MatrixName::A_SCALE;
                parseNameOverride(vals, "A_SCALE");
                A_scale.fill(vals, fillMods(driver));
            }
            if (info.scaleB) {
                auto vals = MatrixName::B_SCALE;
                parseNameOverride(vals, "B_SCALE");
                B_scale.fill(vals, fillMods(driver));
            }
            if (Ag.valid())    rowGroupSum(Ag, A, info.needsAGroupSums() ? AO : Matrix{}, Ag_groupK);
            if (Bg.valid()) columnGroupSum(Bg, B, info.needsBGroupSums() ? BO : Matrix{}, Bg_groupK);
        } else {
            // Read check: zero out A/B/C/C_ref.
            A.fill(MatrixName::ZERO, fillMods(driver));
            A.alignForReadCheck();
            B.fill(MatrixName::ZERO, fillMods(driver));
            B.alignForReadCheck();
            C.fill(MatrixName::ZERO, fillMods(driver));
            C_ref.fill(MatrixName::ZERO, fillMods(driver));
            if (info.hpl)               P.fill(MatrixName::ZERO, fillMods(driver));
            if (needCO)                 CO.fill(MatrixName::ZERO, fillMods(driver));
            if (info.sumA || info.sumB) CO_ref.fill(MatrixName::ZERO, fillMods(driver));

            if (info.strongReadCheck) {
                A.reverseReadCheck();
                B.reverseReadCheck();
                C.reverseReadCheck();
                C_ref.reverseReadCheck();
                if (info.hpl)               P.reverseReadCheck();
                if (needCO)                 CO.reverseReadCheck();
                if (info.sumA || info.sumB) CO_ref.reverseReadCheck();
            }

            if (info.aOffset)              AO.fill(MatrixName::ZERO, fillMods(driver));
            if (info.bOffset)              BO.fill(MatrixName::ZERO, fillMods(driver));
            if (info.scaleA)          A_scale.fill(MatrixName::ZERO, fillMods(driver));
            if (info.scaleB)          B_scale.fill(MatrixName::ZERO, fillMods(driver));
            if (info.needsSeparateASums()) Ag.fill(MatrixName::ZERO, fillMods(driver));
            if (info.needsSeparateBSums()) Bg.fill(MatrixName::ZERO, fillMods(driver));
        }
    }

    auto launchPhase = [&](DriverPhase phase) -> TestResult {
        if (rtImmediateExecution()) {
            if (phase == DriverPhase::PrepareList)
                return TestResult(driver.info.ClockFrequency);
            else if (phase == DriverPhase::RunList)
                phase = DriverPhase::Run;
        }

        TestResult result(driver.info.ClockFrequency);

        if (!trsm) {
            result = gemmDriverDispatch(phase, driver, kinfo, time,
                                        conja, conjb, cOffType,
                                        alpha, beta, A, B, C, P, CO,
                                        AO, BO, A_scale, B_scale, Ag, Bg,
                                        AddrOffset);
        } else {
        }

        return result;
    };

    auto launchOnce = [&]() -> TestResult {
        bool OldEnableMetrics = driver.getEnableMetrics();
        driver.enableMetrics(false);

        launchPhase(DriverPhase::Init);
        if (ShowDispatchInfo) printf("\n");
        auto res = launchPhase(DriverPhase::Run);
        launchPhase(DriverPhase::Finalize);

        driver.enableMetrics(OldEnableMetrics);
        return res;
    };

    if (driver.GenerateAUB) {
        // Run kernel once for AUB output.
        result = launchOnce();
    } else if (mode == TestMode::Accuracy) {
        bool doReference = !readCheck && !TestDeterminism;

        if (info.localA || info.localB || info.localC) {
            result.markAccuracy(TestAccuracy::Unchecked);
            return result;
        }

        if (doReference) {
            // Reference computation.
            if (conja) A.conjugate();   /* Will restore A/B later. */
            if (conjb) B.conjugate();

            if (info.aOffset == 's') AO.get<int16_t>(0, 0) *= -1;
            if (info.bOffset == 's') BO.get<int16_t>(0, 0) *= -1;

            if (!trsm) {
                // GEMM-like kernels.
                copyAll(C_ref, C);

                if (info.copyC()) {
                    int mgran = info.C_tileR;
                    int ngran = info.C_tileC;
                    if (info.layoutC == 'A') {
                        mgran = std::max(mgran, info.unroll[LoopM]);
                        ngran = std::max(ngran, info.crosspackC);
                    } else {
                        ngran = std::max(ngran, info.unroll[LoopN]);
                        mgran = std::max(mgran, info.crosspackC);
                    }
                    auto Cr_border = C_ref.view();
                    Cr_border.mpad = std::min(mpad, align_up(m, mgran));
                    Cr_border.npad = std::min(npad, align_up(n, ngran));
                    Cr_border.zeroPaddedArea();
                }

                if (info.cBias) {
                    auto C_temp = C;
                    ref::gemm(1, A, B, 0, C_temp,
                              info.aqGroup, info.bqGroup, AO, BO, A_scale, B_scale);
                    offset(C_temp, CO, cOffType);
                    linearCombination(C_ref, C_temp, alpha, beta);
                } else {
                    ref::gemm(alpha, A, B, beta, C_ref,
                              info.aqGroup, info.bqGroup, AO, BO, A_scale, B_scale);
                }

                if (info.triangular) copyTriangle(C_ref, C, !info.lower);
                if (info.hermitian)  C_ref.fillImagDiagonal(MatrixName::ZERO);
                if (info.sumA || info.sumB) {
                    copyAll(CO_ref, CO);
                    Matrix CO_temp(Tc, CO);
                    info.sumA ? rowSum(CO_temp, A)
                              : columnSum(CO_temp, B);
                    linearCombination(CO_ref, CO_temp, alpha, beta);
                }
            } else {
                // TRSM reference computation.
                bool trans_trsm = left ? transa : transb;
                bool trans_rect = left ? transb : transa;
                bool left_trsm = left;
                bool lower_trsm = lower;
                trans_trsm ^= trans_rect;
                left_trsm  ^= trans_rect;
                lower_trsm ^= trans_rect;
                lower_trsm ^= trans_trsm;
                Matrix C_temp;

                // Initialize V padding region.
                if (trans_rect == transc)
                    copyAll(C_ref, C);
                else
                    transposeAll(C_ref, C);

                // Copy V matrix data.
                copy(C_ref, left ? B : A);

                // Compute, transposing if transv and transc do not match.
                C_temp = trans_rect ? C_ref : C_ref.view();
                ref::trsm(left_trsm, lower_trsm, unit, alpha, left ? A : B, C_temp);
                if (trans_rect)
                    transpose(C_ref, C_temp);
            }

            if (info.cOffset)
                offset(C_ref, CO, cOffType);

            if (info.aOffset == 's') AO.get<int16_t>(0, 0) *= -1;
            if (info.bOffset == 's') BO.get<int16_t>(0, 0) *= -1;

            if (conja) A.conjugate();   /* Restore A/B for GPU computation. */
            if (conjb) B.conjugate();
        }

        // For SYMM/HEMM, fill the unused triangle with NaNs to catch unwanted accesses.
        if (!readCheck) {
            if (symm || hemm)
                U.paddedView().makeTriangular(lower, true);
            if (hemm)
                U.paddedView().fillImagDiagonal(MatrixName::NANS);
        }

        // First run for determinism checks.
        if (TestDeterminism) {
            copy(C_ref, C);
            std::swap(C, C_ref);
            result = launchOnce();
            std::swap(C, C_ref);
        }

        // GPU computation.
        result = launchOnce();

        // Check accuracy.
        checkL3(driver, result, kinfo, k, alpha, beta, A, B, C, CO, P, C_ref, CO_ref);

        if (result.failed() && DumpOutput) {
            std::ofstream C_file("computed.bin");
            std::ofstream C_ref_file("ref.bin");
            std::ofstream infoFile("info.txt");
            if (C_file && C_ref_file && infoFile) {
                C_file.write((const char *) C.data.ptr, C.data.bytes);
                C_ref_file.write((const char *) C_ref.data.ptr, C.data.bytes);
                infoFile << "Precision,Layout,m,n,ld\n";
                infoFile << info.precisionC << ',' << info.layoutC  << ',';
                infoFile << m << ',' << n << ',' << ldc << std::endl;
            }
        }
    } else if (mode == TestMode::BeginSeries) {
        launchPhase(DriverPhase::InitMultiple);
    } else if (mode == TestMode::EndSeries) {
        launchPhase(DriverPhase::FinalizeMultiple);
    } else {
        // Timing tests.
        TestResult resultWU(driver.info.ClockFrequency);

        ShowBlockInfo = !driver.Quiet;

        launchPhase(DriverPhase::Init);
        launchPhase(DriverPhase::PrepareList);

        bool Cold = driver.ColdA || driver.ColdB || driver.ColdC;
        for (int r = 0; r < driver.WarmupReps; r++) {
            if (Cold) flushDriver(driver);
            resultWU += launchPhase(DriverPhase::RunList);
        }
        for (int r = 0; r < driver.TimeMaxReps; r++) {
            if (Cold) flushDriver(driver);
            result += launchPhase(DriverPhase::RunList);
            if (r > driver.TimeMinReps)
                if (result.meanStddevBestAvailable() <= driver.RelativeMeanStdDevGoal * result.bestTimeAvailable())
                    break;
            if (r > 1 && driver.PerfBailThreshold > 0.)
                if (result.bestTimeAvailable() - result.meanStddevBestAvailable() > driver.PerfBailThreshold)
                    break;
        }

        launchPhase(DriverPhase::Finalize);

        ShowBlockInfo = false;
        summarize(driver, result, info, batchSize, m, n, k, driver.WGLimit);
        summarize(driver, resultWU, info, batchSize, m, n, k, driver.WGLimit);
        double peakFlops = PeakFlops(driver, info);

        if (!driver.Quiet) {
            if (driver.Loud)
                std::cout << "             Warmup time: " << Duration{resultWU.time} << " (" << driver.WarmupReps << " repetitions)\n";
            std::cout << "             " << TestResult::timeLabel() << " time: " << Duration{result.time} << " (" << result.iterations << " repetitions)";
            std::cout << " -- std. deviation: " << Duration{result.stddevDeviceTimeIfAvailable()} << std::endl;
            if (driver.Loud && result.time != result.medianHostTime())
                std::cout << "           Host overhead: " << Duration{result.medianHostTime() - result.time} << std::endl;
            if (driver.Loud && driver.info.PeakBW > 0. && !info.triangular && !info.trxm()) {
                int mCross = int(std::ceil((0.5 * peakFlops / driver.info.PeakBW) * (0.125 * info.typeB().bits())));
                int nCross = int(std::ceil((0.5 * peakFlops / driver.info.PeakBW) * (0.125 * info.typeA().bits())));
                std::cout << "  Memory-bound crossover: ";
                if (mCross == nCross)
                    std::cout << "m = n = " << mCross << std::endl;
                else
                    std::cout << "m = " << mCross << ", n = " << nCross << std::endl;
            }
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "             Actual GB/s: " << result.bw * 1e-9 << std::endl;
            if (driver.info.PeakBW > 0.)
                std::cout << "               Peak GB/s: " << driver.info.PeakBW * 1e-9 << std::endl;
            std::cout << std::setprecision(1);
            std::cout << "            Actual GOp/s: " << result.flops * 1e-9 << std::endl;
            std::cout << "              Peak GOp/s: " << peakFlops * 1e-9 << std::endl;
            std::cout << "      Nominal efficiency: " << result.efficiency << "%\n";
            if (result.avgFreq > 0.) {
                std::cout << "    Effective efficiency: " << result.effEfficiency << "%   (";
                if (result.tileFreqs.size() <= 1)
                    std::cout << int(std::round(result.avgFreq * 1e-6)) << " MHz";
                else for (size_t i = 0; i < result.tileFreqs.size(); i++) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << "tile " << i << ": " << int(std::round(result.tileFreqs[i] * 1e-6)) << " MHz";
                }
                std::cout << ")\n";
            }
            if (ShowAdditionalMetrics) {
                auto memTraffic = traffic(m, n, k, batchSize, info, !(info.beta == 0), driver.WGLimit);
                std::cout << std::setprecision(2);
                if (result.metrics[L3R] == 0 && result.metrics[L3CLR] > 0)
                    result.metrics[L3R] = result.metrics[L3CLR] * 64;
                if (result.metrics[L3W] == 0 && result.metrics[L3CLW] > 0)
                    result.metrics[L3W] = result.metrics[L3CLW] * 64;

                if (result.metrics[L1R] > 0) {
                  std::cout << "                L1 reads:    "
                            << "     (" << result.metrics[L1R] * 1e-6
                            << " MB)\n";
                }
                if (result.metrics[L1W] > 0) {
                  std::cout << "                L1 writes:   "
                            << "    (" << result.metrics[L1W] * 1e-6
                            << " MB)\n";
                }
                if (result.metrics[L3R] > 0) {
                    auto l3RFraction = result.metrics[L3R] / memTraffic.l3Read;
                    std::cout << "                L3 reads: " << l3RFraction  << "x   ("
                              << result.metrics[L3R] * 1e-6 << " MB)\n";
                }
                if (result.metrics[L3W] > 0) {
                    auto l3WFraction = result.metrics[L3W] / memTraffic.write;
                    std::cout << "               L3 writes: " << l3WFraction  << "x   ("
                              << result.metrics[L3W] * 1e-6 << " MB)\n";
                } else if (result.metrics[L3RW] > 0) {
                    auto l3RWFraction = result.metrics[L3RW] / (memTraffic.l3Read + memTraffic.write);
                    std::cout << "         L3 reads/writes: " << l3RWFraction  << "x   ("
                              << result.metrics[L3RW] * 1e-6 << " MB)\n";
                }
                if (result.metrics[L3Hit] && result.metrics[L3Miss]) {
                    auto total = result.metrics[L3Hit] + result.metrics[L3Miss];
                    std::cout << "             L3 hit rate: "
                              << 100. * result.metrics[L3Hit]/total << "%  (" << result.metrics[L3Hit]
                              << " hits, " << result.metrics[L3Miss] << " misses)\n";
                }
                if (result.metrics[L3Busy]) {
                    std::cout << "                 L3 busy: "
                              << result.metrics[L3Busy] << "%\n";
                }
                if (result.metrics[L3InputAvailable]) {
                    std::cout << "      L3 input available: "
                              << result.metrics[L3InputAvailable] << "%\n";
                }
                if (result.metrics[L3OutputReady]) {
                    std::cout << "         L3 output ready: "
                              << result.metrics[L3OutputReady] << "%\n";
                }
                if (result.metrics[L3SuperQFull])
                    std::cout << "          L3 SuperQ full: " << result.metrics[L3SuperQFull] << "%\n";
                if (result.metrics[L3Stall])
                    std::cout << "                L3 stall: " << result.metrics[L3Stall] << "%\n";
                if (result.metrics[MainR] > 0) {
                    auto memRFraction = result.metrics[MainR] / memTraffic.read;
                    std::cout << "        GPU memory reads: " << memRFraction << "x   (";
                    if (result.metrics[MainRRate] > 0)
                        std::cout << result.metrics[MainRRate] << " GB/s, ";
                    if (result.metrics[MainR32B] > 0)
                        std::cout << result.metrics[MainR32B] * 32 * 1e-6 << " MB 32b load, ";
                    if (result.metrics[MainR64B] > 0)
                        std::cout << result.metrics[MainR64B] * 64 * 1e-6 << " MB 64b load, ";
                    if (result.metrics[MainRL3] > 0)
                        std::cout << result.metrics[MainRL3] << " GTI reads, ";
                    std::cout << result.metrics[MainR] * 1e-6 << " MB)\n";
                }
                if (result.metrics[MainW] > 0) {
                    auto memWFraction = result.metrics[MainW] / memTraffic.write;
                    std::cout << "       GPU memory writes: " << memWFraction << "x   (";
                    if(result.metrics[MainWRate] > 0)
                        std::cout << result.metrics[MainWRate] << " GB/s, ";
                    if (result.metrics[MainW32B] > 0)
                        std::cout << result.metrics[MainW32B] * 32 * 1e-6 << " MB 32b load, ";
                    if (result.metrics[MainW64B] > 0)
                        std::cout << result.metrics[MainW64B] * 64 * 1e-6 << " MB 64b load, ";
                    if (result.metrics[MainWL3] > 0)
                        std::cout << result.metrics[MainWL3] << " GTI writes, ";
                    std::cout << result.metrics[MainW] * 1e-6 << " MB)\n";
                }
                if (result.metrics[MainQFull] > 0)
                    std::cout << "   GPU memory queue full: " << result.metrics[MainQFull] << "%\n";
                if (result.metrics[TLBMiss] > 0)
                    std::cout << "              TLB misses: " << result.metrics[TLBMiss] << "\n";
                if (result.metrics[ICacheMiss] > 0)
                    std::cout << "               I$ misses: " << result.metrics[ICacheMiss] << "\n";
                if (result.metrics[ComputeStall] > 0) {
                    std::cout << "           Compute stall: " << result.metrics[ComputeStall] << "%\n";
                    if (result.metrics[ALUWriteBackStall] > 0)
                        std::cout << "         ALU writeback stall: " << result.metrics[ALUWriteBackStall] << "%\n";
                    if (result.metrics[BarrierStall] > 0)
                        std::cout << "               Barrier stall: " << result.metrics[BarrierStall] << "%\n";
                    if (result.metrics[ControlStall] > 0)
                        std::cout << "               Control stall: " << result.metrics[ControlStall] << "%\n";
                    if (result.metrics[InstrFetchStall] > 0)
                        std::cout << "     Instruction fetch stall: " << result.metrics[InstrFetchStall] << "%\n";
                    if (result.metrics[PipeStall] > 0)
                        std::cout << "                  Pipe stall: " << result.metrics[PipeStall] << "%\n";
                    if (result.metrics[SbidStall] > 0)
                        std::cout << "                  SBID stall: " << result.metrics[SbidStall] << "%\n";
                    if (result.metrics[SendStall] > 0)
                        std::cout << "                  Send stall: " << result.metrics[SendStall] << "%\n";
                    if (result.metrics[SharedFunctionStall] > 0)
                        std::cout << "       Shared function stall: " << result.metrics[SharedFunctionStall] << "%\n";
                    if (result.metrics[OtherStall] > 0)
                        std::cout << "                 Other stall: " << result.metrics[OtherStall] << "%\n";
                }
            }
            if (ShowAllTimes) {
                bool haveDTimes = !result.allDTimes.empty() && result.allDTimes[0] > 0;
                auto &allTimes = haveDTimes ? result.allDTimes : result.allHTimes;
                yprintf("All %s times:\n", haveDTimes ? "device" : "host");
                for (auto &t: allTimes)
                    std::cout << Duration{t} << std::endl;
            }
            std::cout << std::defaultfloat << std::setprecision(6);
        }
    }

    return result;
}

template <typename fp>
void _checkCopy(DriverRT &driver, TestResult &result, const CopyKernelInfo &info, const Matrix &D, const Matrix &D_ref, double maxAbsErr)
{
    using fp_sum = sum_t<fp>;
    using fp_print = printable_t<fp>;
    using fp_sum_print = printable_t<fp_sum>;

    auto Ts = info.typeS(), Td = info.typeD(), Tsum = info.typeSum();

    bool inner = info.inner();

    auto x = D.n, y = D.m;
    auto xpad = D.npad, ypad = D.mpad;
    auto ldd = D.ld;

    int ysum = info.sum ? (Tsum.size() / Td) : 0;

    int y0pad = D.m;
    if (info.trsm)
        y0pad = std::min<integer>(ypad, align_up(D.n, info.unrollX()));

    auto D_tileX = inner ? info.D_tileR : info.D_tileC;
    auto D_tileY = inner ? info.D_tileC : info.D_tileR;
    if (D_tileX == 0) D_tileX = info.packSize;
    if (D_tileY == 0) D_tileY = info.crosspackD;

    auto p = D.data.as<fp>();
    auto pref = D_ref.data.as<fp>();

    int inaccuracies = 0;
    for (integer a0 = 0; a0 < xpad; a0 += info.packSize) {
        auto naa = std::min<integer>(info.packSize, xpad - a0);
        for (integer b0 = 0; b0 < ypad - ysum; b0 += info.crosspackD) {
            auto nbb = std::min<integer>(info.crosspackD, ypad - b0);
            for (integer aa = 0; aa < naa; aa++) {
                for (integer bb = 0; bb < nbb; bb++) {
                    auto aat = aa % D_tileX;
                    auto b0t = b0 % D_tileY;
                    auto idx = a0 * ldd + (b0 - b0t) * info.packSize
                                + (aa - aat) * D_tileY + b0t * D_tileX
                                + aat * info.crosspackD + bb;
                    auto a = a0 + aa;
                    auto b = b0 + bb;
                    fp dVal    = p[idx];
                    fp dRefVal = pref[idx];

                    if (!checkNearby(dVal, dRefVal, maxAbsErr)) {
                        bool print = !driver.Quiet && (inaccuracies++ <= MaxInaccuracies);
                        if (a >= x && a0 < x && b0 < y)
                            continue; // Ignore overruns within x unroll.
                        else {
                            if (print && !driver.Loud && inaccuracies == 1) std::cerr << std::endl;
                            if (a0 >= x || b0 >= y0pad) {
                                result.markAccuracy(TestAccuracy::Overrun);
                                if (print) std::cerr << "Overrun";
                            } else {
                                result.markAccuracy(TestAccuracy::Inaccurate);
                                if (print) std::cerr << "Inaccuracy";
                            }
                        }

                        auto i = inner ? a : b;
                        auto j = inner ? b : a;

                        if (print) {
                            std::cerr << " at (" << i << ',' << j << "): computed "
                                      << fp_print(dVal) << " versus " << fp_print(dRefVal) << std::endl;
                        }
                    }
                }
            }
        }
    }

    if (info.sum) for (integer a0 = 0; a0 < xpad; a0 += info.packSize) {
        auto naa = std::min<integer>(info.packSize, xpad - a0);
        for (integer aa = 0; aa < naa; aa++) {
            auto idx = (a0 + info.packSize) * ldd - ysum * info.packSize + aa * ysum;
            auto a = a0 + aa;
            auto sumVal = *((const fp_sum *) (p + idx));
            auto sumRefVal = *((const fp_sum *) (pref + idx));

            if (!checkNearby(sumVal, sumRefVal, maxAbsErr * y)) {
                bool print = !driver.Quiet && (inaccuracies++ <= MaxInaccuracies);
                if (a >= x && a0 < x)
                    continue; // Ignore overruns within x unroll.
                else {
                    result.markAccuracy(TestAccuracy::Inaccurate);
                    if (print) {
                        if (!driver.Loud && inaccuracies == 1) std::cerr << std::endl;
                        std::cerr << "Sum inaccuracy at " << a << ": computed "
                                  << fp_sum_print(sumVal) << " versus " << fp_sum_print(sumRefVal) << std::endl;
                    }
                }
            }
        }
    }
}

TYPE_TEMPLATIFY(checkCopy)

void checkCopy(DriverRT &driver, TestResult &result, const CopyKernelInfo &info, superscalar alpha, const Matrix &D, const Matrix &D_ref)
{
    auto Ts = info.typeS(), Td = info.typeD(), Tsum = info.typeSum();
    auto m = D.m, n = D.n;

    bool nonunitTRSM = info.trsm && !info.unit;

    double eps = std::min(epsilon(info.precisionD), epsilon(info.precisionS));
    double maxAbsErr = 4 * eps * Ts.complexComponents();
    if (nonunitTRSM && Ts.isComplex())
        maxAbsErr *= 4;

    if (info.readCheck || (alpha == 1.0 && Ts == Td && !nonunitTRSM))     // I'm not overthinking it... it must be perfect!
        maxAbsErr = 0;

    result.markAccuracy(TestAccuracy::OK);
    checkCopy(Td, driver, result, info, D, D_ref, maxAbsErr);
}

template <typename fp>
void _tricopyPrefill(Matrix &D, const CopyKernelInfo &info, integer y0pad)
{
    int nfuse = (info.fusedEUs() ? 2 : 1);

    auto bunroll = (info.targetWG > 1) ? info.targetWG : nfuse;
    auto bsize = info.unrollX() * bunroll;

    auto xpadWrite = align_up(D.n, info.packSize);
    auto ypad = D.mpad, ldd = D.ld;

    auto p = D.data.as<fp>();

    for (integer i = 0; i < xpadWrite; i += info.packSize) {
        integer jmin, jmax;

        if (info.lower == info.inner()) {
            jmin = 0;
            jmax = std::min<integer>(y0pad, bsize + align_down(i, bsize));
        } else {
            jmin = align_down(i, bsize);
            jmax = y0pad;
        }
        for (integer j = 0; j < ypad; j++)
            for (integer ii = 0; ii < info.packSize; ii++)
                p[(i * ldd) + (j * info.packSize) + ii] = (j >= jmin && j < jmax) ? qnan<fp>() : fp(0);
    }
}

TYPE_TEMPLATIFY(tricopyPrefill)

TestResult testCopy(DriverRT &driver, KernelInfo &kinfo, TestMode mode,
                    int32_t m, int32_t n, int32_t lds, int32_t ldd)
{
    if (mode == TestMode::BeginSeries || mode == TestMode::EndSeries)
        return TestResult(driver.info.ClockFrequency);

    bool time = (mode == TestMode::Time || mode == TestMode::TimeOneOfSeries);
    auto &info = kinfo.copy[0];

    auto Ts = info.typeS(), Td = info.typeD(), Tsum = info.typeSum();

    if (info.sum && Tsum != sumType(Td))
        throw std::runtime_error("Unsupported sum type");

    auto readCheck = info.readCheck;
    bool trans = (info.layoutS == 'T');
    bool inner = info.inner();

    auto unrollY = info.unrollY();

    TestResult result(driver.info.ClockFrequency);

    if (info.nongeneral())
        n = m;

    auto x = inner ? m : n;
    auto y = inner ? n : m;

    // Don't time read check kernels.
    if (time && readCheck)
        return result;

    OverrunTest = !time && !driver.SpecifiedLDs;

    int nfuse = (info.fusedEUs() ? 2 : 1);
    int extra = readCheck ? 64 : 0;
    if (info.trsm) extra += info.unrollX();
    int xpad = align_up(x + extra, info.packSize * nfuse);
    int ypad = OverrunTest ? align_up(y + extra, unrollY)
                           : align_up(y, info.crosspackD);

    if (OverrunTest)
        ypad += unrollY;

    int ysum = 0;
    if (info.sum) {
        ysum = Tsum.size() / Td;
        ypad = align_up(ypad, ysum) + ysum;
    }

    int y0pad = y;
    if (info.trsm)
        y0pad = std::min(ypad, align_up(x, info.unrollX()));

    auto mpad = inner ? xpad : ypad;
    auto npad = inner ? ypad : xpad;

    ShowBlockInfo = false;

    int32_t min_lds = trans ? npad : mpad;
    int32_t min_ldd = ypad;

    if (lds < 0) lds = niceLD(driver.Hardware, Ts, min_lds) + (-1 - lds);
    if (ldd < 0) ldd = niceLD(driver.Hardware, Td, min_ldd) + (-1 - ldd);

    lds = std::max(lds, min_lds);
    ldd = std::max(ldd, min_ldd);

    if (driver.OddLD[0]) lds++;
    if (driver.OddLD[1]) ldd++;

    if (info.alignS) lds = align_up(lds, info.alignS);
    if (info.alignD) ldd = align_up(ldd, info.alignD);

    if (!driver.Quiet && (time || driver.Loud)) {
        std::cout << "Size is " << m << " x " << n;
        std::cout << ", leading dimensions " << lds << ", " << ldd << std::endl;
    }

    superscalar alpha = info.alpha;
    if (Ts.isFP() && !std::isfinite(alpha))
        alpha = makeComplex<superscalar>(2.3, -1.7);           // Choose an arbitrary alpha.

    // Initialize S/D.
    Matrix S{Ts, trans, m, n, lds, mpad, npad};
    Matrix D{Td, false, y, x, ldd, ypad, xpad};
    Matrix D_ref;

    if (!readCheck) {
        MatrixName S_name;
        if (info.triangular)
            S_name = MatrixName::COPYTRSM_S;
        else if (info.nongeneral())
            S_name = info.lower ? MatrixName::COPYL_S : MatrixName::COPYU_S;
        else
            S_name = MatrixName::COPY_S;
        parseNameOverride(S_name, "S");
        S.fill(S_name, fillMods(driver));
    } else {
        S.fill(MatrixName::ZERO, fillMods(driver));
        S.alignForReadCheck();
        D.data.zero();
    }

    auto launchPhase = [&](DriverPhase phase) -> TestResult {
        if (rtImmediateExecution()) {
            if (phase == DriverPhase::PrepareList)
                return TestResult(driver.info.ClockFrequency);
            else if (phase == DriverPhase::RunList)
                phase = DriverPhase::Run;
        }

        return copyDriver(phase, driver, kinfo, time, alpha, S, D);
    };

    auto launchOnce = [&]() -> TestResult {
        bool OldEnableMetrics = driver.getEnableMetrics();
        driver.enableMetrics(false);

        launchPhase(DriverPhase::Init);
        if (ShowDispatchInfo) printf("\n");
        auto res = launchPhase(DriverPhase::Run);
        launchPhase(DriverPhase::Finalize);

        driver.enableMetrics(OldEnableMetrics);
        return res;
    };

    if (driver.GenerateAUB) {
        // Run kernel once for AUB output.
        result = launchOnce();
    } else if (!time) {
        if (readCheck)
            D_ref = D;
        else {
            // Reference computation.
            D_ref = Matrix(Td, D);

            if (info.triangular) {
                // Triangular copy kernels aren't supposed to fill in all the 0s.
                // Prefill with NaN where values are expected.
                tricopyPrefill(Td, D, info, y0pad);
            }

            Matrix S_eff;
            if (info.nongeneral()) {
                S_eff = S;
                if (info.symmetric)
                    S_eff.makeSymmetric(info.lower);
                else if (info.hermitian)
                    S_eff.makeHermitian(info.lower);
                else if (info.triangular) {
                    S_eff.makeTriangular(info.lower);
                    if (info.unit)
                        S_eff.makeUnitDiagonal();
                    if (info.trsm) {
                        if (!info.unit)
                            S_eff.invertDiagonal();
                        S_eff.negateDiagonal();
                    }
                }
            } else
                S_eff = S.view();

            (inner ? packC : packR)(D_ref.data, ldd, S_eff, info.packSize, info.crosspackD,
                                    info.D_tileR, info.D_tileC, info.sum, MatrixName::UNINIT);

            if (info.conjugate)
                D_ref.conjugate();

            if (alpha != 1) {
                auto Dp_linear = D_ref.linearPaddedView();
                scale(Dp_linear, alpha);
            }
        }

        // GPU computation.
        result = launchOnce();

        // Check accuracy.
        checkCopy(driver, result, info, alpha, D, D_ref);

        if (result.failed() && DumpOutput) {
            std::ofstream D_file("computed.bin");
            std::ofstream Dref_file("ref.bin");
            std::ofstream infoFile("info.txt");
            if (D_file && Dref_file && infoFile) {
                D_file.write((const char *) D.data.ptr, D.data.bytes);
                Dref_file.write((const char *) D_ref.data.ptr, D_ref.data.bytes);
                infoFile << "Precision,Layout,m,n,ld\n";
                infoFile << info.precisionD << ',' << info.layoutD  << ',';
                infoFile << m << ',' << n << ',' << ldd << std::endl;
            }
        }
    } else {
        // Timing tests.
        TestResult resultWU(driver.info.ClockFrequency);

        ShowBlockInfo = !driver.Quiet;

        launchPhase(DriverPhase::Init);
        launchPhase(DriverPhase::PrepareList);

        bool anyColdMat = driver.ColdA || driver.ColdB || driver.ColdC;
        for (int r = 0; r < driver.WarmupReps; r++) {
            if (anyColdMat) flushDriver(driver);
            resultWU += launchPhase(DriverPhase::RunList);
        }
        for (int r = 0; r < driver.TimeMaxReps; r++) {
            if (anyColdMat) flushDriver(driver);
            result += launchPhase(DriverPhase::RunList);
            if (r > driver.TimeMinReps)
                if (result.meanStddevBestAvailable() <= driver.RelativeMeanStdDevGoal * result.bestTimeAvailable())
                    break;
        }

        launchPhase(DriverPhase::Finalize);

        ShowBlockInfo = false;

        double avg = result.reportedDeviceTimeIfAvailable();
        double avgHost = result.hostTime;
        double avgWU = resultWU.reportedDeviceTimeIfAvailable();

        double entries = double(m) * double(n);
        if (info.triangular)
            entries = triangleEntries(m, n, info.lower);

        double bytes = entries * (Ts.bits() + Td.bits()) * 0.125;
        double actualTput = bytes / avg;

        if (!driver.Quiet) {
            if (driver.Loud) std::cout << "           Warmup time: " << Duration{avgWU} << " (" << driver.WarmupReps << " repetitions)\n";
            std::cout << "           " << TestResult::timeLabel() << " time: " << Duration{avg} << " (" << result.iterations << " repetitions)";
            std::cout << " -- std. deviation: " << Duration{result.stddevDeviceTimeIfAvailable()} << std::endl;
            if (driver.Loud && avg != avgHost)
                std::cout << "         Host overhead: " << Duration{avgHost - avg} << std::endl;
            std::cout << "           Actual GB/s: " << actualTput / 1e9 << std::endl;
        }

        result.time = avg;
        result.bw = actualTput;
        if (driver.info.PeakBW > 0.)
            result.efficiency = result.bw / driver.info.PeakBW * 100;
        result.effEfficiency = result.efficiency; // Assume no throttling
        if (result.avgFreq > 0.)
            result.effTime = result.reportedEffectiveDeviceTimeIfAvailable();
        else
            result.effTime = avg;
    }

    return result;
}

TestResult test(DriverRT &driver, KernelInfo &info, TestMode mode,
                uint32_t batchSize, int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
                bool conja, bool conjb, bool unit, char cOffType, bool aOffset, bool bOffset)
{
    TestResult result(driver.info.ClockFrequency);

    if (driver.TestCOffsetType) cOffType = driver.TestCOffsetType;

    if (driver.Progress) {
        std::cout << '.';
        fflush(stdout);
    }

    if (info.isCopy())
        result = testCopy(driver, info, mode, m, n, lda, ldb);
    else {
        result = testL3(driver, info, mode, batchSize, m, n, k, lda, ldb, ldc,
                        conja, conjb, unit, cOffType, aOffset, bOffset);
    }

    if (MaxInaccuracies < 0)
        result.accuracy = TestAccuracy::OK;      /* ignore failures */

    return result;
}

} // namespace

inline PoolBuffer::PoolBuffer(benchstone::DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
                              void *hostPtr, size_t offset, bool coldCache)
    : handle(rtCreateBackendBuffer(driver, flags, nullptr, bytes, hostPtr, offset)),
      context(driver.rtArgs.context), capacity(bytes), flags(flags),
      hostBound(rtBindsHostPtr(driver, flags, hostPtr)),
      flushBytes(coldCache ? bytes : 0) {}

inline bool PoolBuffer::reusable(benchstone::DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
                                 void *hostPtr) const
{
    return context == driver.rtArgs.context && sized(bytes) && this->flags == flags
        && !hostBound && !rtBindsHostPtr(driver, flags, hostPtr);
}

inline PoolBuffer::~PoolBuffer()
{
    if (handle) rtFreeBuffer(context, handle);
}

inline rt_buffer PoolBuffer::realloc(benchstone::DriverRT<Runtime> &driver, BufferFlags flags, size_t bytes,
                                     void *hostPtr, size_t offset, bool coldCache)
{
    if (reusable(driver, flags, bytes + offset, hostPtr)) {
        rtCreateBackendBuffer(driver, flags, handle, bytes + offset, hostPtr, offset);
        flushBytes = coldCache ? bytes + offset : 0;
    }
    else {
        *this = {}; // Free to avoid overlapping allocations.
        *this = PoolBuffer(driver, flags, bytes + offset, hostPtr, offset, coldCache);
    }
    return handle;
}

namespace benchstone {

template <RuntimeBackend runtime_>
std::tuple<std::vector<gemmstone::Timepoint>, int, int> DriverRT<runtime_>::getChromeTrackingData() const {
    const auto &storage = rtArgs.storage;
    return {storage.LastTrackingData, storage.LastTrackingStride, storage.LastTrackingWGSize};
};

template <RuntimeBackend runtime_>
DriverRT<runtime_>::~DriverRT() = default;

template <RuntimeBackend runtime_>
void DriverRT<runtime_>::init() {
    envInit(*this);
    rtInit(*this);
    initialized = true;
}

template <RuntimeBackend runtime_>
void DriverRT<runtime_>::enableMetrics(bool enabled) {
    assert(initialized);
    if (this->isSimulation()) return; // Cannot use metrics in simulation

    if (!enabled) {
        metricsEnabled = false;
        return;
    }

    auto &storage = rtArgs.storage;
    if (!storage.metricsInitialized) {
        storage.initSuccess = rtInitMetrics(*this);
        storage.metricsInitialized = true;
    }

    metricsEnabled = storage.initSuccess;
}

template <RuntimeBackend runtime_>
bool DriverRT<runtime_>::initKernel(const KernelInfo& info, const std::vector<uint8_t> &binary) {
    rtArgs.binary = binary;
    return rtInitKernel(*this, info);
}

template <RuntimeBackend runtime_>
void DriverRT<runtime_>::finishKernel() {
    rtFinishKernel(rtArgs);
}

template <RuntimeBackend runtime_>
void DriverRT<runtime_>::finish() {
    rtArgs.storage.forEachBuffer([](auto &buffer) { buffer = {}; });
    rtFinish(*this);
}

template <RuntimeBackend runtime_>
bool DriverRT<runtime_>::isKernelLoaded() const {
    return rtArgs.kernel != nullptr;
}

template <RuntimeBackend runtime_>
TestResult DriverRT<runtime_>::testImpl(KernelInfo &info, TestMode mode,
                    uint32_t batchSize, int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
                    bool conja, bool conjb, bool unit, char cOffType, bool aOffset, bool bOffset) {
    perf_lock_guard_t gpu_lock(true);
    return ::test(*this, info, mode, batchSize, m, n, k, lda, ldb, ldc, conja, conjb, unit, cOffType, aOffset, bOffset);
}

} // namespace benchstone
