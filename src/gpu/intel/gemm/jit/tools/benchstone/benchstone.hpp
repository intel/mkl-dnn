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

#ifndef GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_HPP
#define GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_HPP

#include "benchstone_common.hpp"
#include "kernel_info.hpp"
#include "tools/device_utils.hpp"

#include <cassert>
#include <exception>
#include <functional>
#include <memory>

namespace benchstone {
    enum class RuntimeBackend {
        OpenCL,
        LevelZero,
    };

    struct DriverInfo {
        int EUs = 0;
        double ClockFrequency = 0;
        double PeakBW = 0;
        size_t LLCSize = 0;
        bool UniformWG = false;
    };

    class Driver {
    public:
        virtual ~Driver() = default;

        virtual void init() = 0;
        virtual void enableMetrics(bool enabled = true) = 0;
        virtual bool initKernel(const KernelInfo& info, const std::vector<uint8_t> &binary) = 0;
        virtual void finishKernel() = 0;
        virtual void finish() = 0;
        TestResult test(KernelInfo &kernelInfo, TestMode mode,
                    uint32_t batchSize, int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
                    bool conja, bool conjb, bool unit, char cOffType, bool aOffset, bool bOffset) {
            try {
                return testImpl(kernelInfo, mode, batchSize, m, n, k, lda, ldb, ldc,
                                conja, conjb, unit, cOffType, aOffset, bOffset);
            } catch (const std::exception &e) {
                TestResult result(info.ClockFrequency);
                result.markAccuracy(TestAccuracy::RuntimeError);
                result.failingCase = e.what();
                return result;
            }
        }
        virtual bool isKernelLoaded() const = 0;

        TestResult runAccuracyTests(KernelInfo &info);

        
        int hwThreads(const gemmstone::CommonDriverInfo &info) const { return this->info.EUs * threadsPerEU(Hardware, info); }

        virtual std::tuple<std::vector<gemmstone::Timepoint>, int, int> getChromeTrackingData() const = 0;
        RuntimeBackend runtime() const { return runtime_; }
        bool isSimulation() const { return simulation; }
        void setSimulation(bool sim) { simulation = sim; }
        bool getEnableMetrics() const {return metricsEnabled; };

        ngen::Product HardwareProduct = {};
        ngen::HW Hardware = ngen::HW::Unknown;

        bool ZeroCopy = false;
        bool ZeroFill = false;
        bool ColdA = false;
        bool ColdB = false;
        bool ColdC = false;
        bool ColdICache = false;
        bool ColdTLB = false;
        bool Loud = false;
        bool Quiet = false;
        bool Progress = false;
        bool SmallTest = false;
        bool NoBlocking = false;
        bool GenerateAUB = false;
        bool TimingOnly = false;
        bool DeviceTime = true;
        bool OneFill = false;
        
        double RelativeMeanStdDevGoal = 0.01;
        double PerfBailThreshold = 0.;
        
        bool FixWGK = false;
        
        BufferFlags ExtraFlagsA = 0, ExtraFlagsB = 0, ExtraFlagsC = 0;
        BufferFlags ExtraFlagsCO = 0, ExtraFlagsP = 0, ExtraFlagsX = 0;
        BufferFlags ExtraFlagsS = 0, ExtraFlagsD = 0;

        int WarmupReps = 2;
        int CopyWarmupReps = 5;
        int TimeMinReps = 4;
        int TimeMaxReps = 8;

        int WGOverride[3] = {0, 0, 0};
        bool SpecifiedLDs = false;
        bool FixedBlocking = false;
        double BiasOverride = 0.0;
        int HilbertBailOverride = 0;
        int BoustrophedonSliceOverride = 0;
        std::vector<BlockingInfo> SKParallelOverride;
        bool DoSKParallelOverride = false;
        FusedDriverParams FusedParamsOverride;
        bool DoFusedParamsOverride = false;
        int AddrOffsetOverride[3] = {0, 0, 0};
        bool OddStride[3] = {false, false, false};
        int WGLimit = 0;
        bool TransposeOverride[2] = {false, false};
        bool DoTransposeOverride = false;
        bool TestBeta0 = false;
        bool TestNonKParallelGlobal = false;
        char TestCOffsetType = '\0';
        bool TestInPlace = false;
        integer NBlockOverrides[3] = {0, 0, 0};
        bool OddLD[3] = {false, false, false};
        
        DriverInfo info;

        std::tuple<integer, integer, integer> getBlocking(integer m, integer n, integer k,  const Level3KernelInfo &info) const {
            return ::getBlocking(Hardware, this->info.EUs, m, n, k, info, NoBlocking, DoSKParallelOverride, SKParallelOverride, FixedBlocking, &NBlockOverrides[0], TestNonKParallelGlobal);
        }

    protected:
        virtual TestResult testImpl(KernelInfo &info, TestMode mode,
                    uint32_t batchSize, int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
                    bool conja, bool conjb, bool unit, char cOffType, bool aOffset, bool bOffset) = 0;
        Driver(RuntimeBackend runtime) : HardwareProduct(detectDevice(true)), Hardware(getCore(HardwareProduct.family)), runtime_(runtime) {}
        const RuntimeBackend runtime_;
        bool simulation = false;
        bool initialized = false;
        bool metricsEnabled = false;
    };

    using DriverFactory = std::function<std::unique_ptr<Driver>()>;
    void registerDriverFactory(RuntimeBackend runtime, DriverFactory factory);
    std::unique_ptr<Driver> createDriver(RuntimeBackend runtime);
}

#endif
