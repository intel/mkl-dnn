/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_DEVICE_UTILS_HPP
#define GEMMSTONE_TOOLS_DEVICE_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "ngen_opencl.hpp"

#include <string_view>

inline cl_device_id findDevice()
{
    static constexpr size_t max_platforms = 16, max_devices = 16;
    cl_platform_id plat_ids[max_platforms];
    cl_uint nplat;
    cl_device_id dev_ids[max_devices];
    cl_uint ndev = 0;

    if (clGetPlatformIDs(max_platforms, plat_ids, &nplat) != CL_SUCCESS)
        throw std::runtime_error("OpenCL failed to find an Intel GPU");

    // Loop through platforms looking for a GPU.
    for (cl_uint plat_idx = 0; plat_idx < nplat; plat_idx++) {
        cl_int status = clGetDeviceIDs(plat_ids[plat_idx], CL_DEVICE_TYPE_GPU,
                                        max_devices, dev_ids, &ndev);
        if ((status == CL_SUCCESS) && (ndev > 0))
            break;
    }

    if (ndev == 0)
       throw std::runtime_error("OpenCL failed to find an Intel GPU");

    return dev_ids[0];
}

inline ngen::HW parseGen(std::string_view gen)
{
    using namespace ngen;
    using namespace std::string_view_literals;

    if (gen == "12LP"sv) return HW::Gen12LP;
    if (gen == "12HP"sv) return HW::XeHP;
    if (gen == "12p7"sv) return HW::XeHPG;
    if (gen == "12p8"sv) return HW::XeHPC;
    if (gen == "Xe2"sv)  return HW::Xe2;
    if (gen == "Xe3"sv)  return HW::Xe3;
    if (gen == "Xe3p"sv) return HW::Xe3p;
    return HW::Unknown;
}

inline ngen::ProductFamily parseFamily(std::string_view family)
{
    using namespace ngen;
    using namespace std::string_view_literals;

    if (family == "DG2"sv)   return ProductFamily::DG2;
    if (family == "MTL"sv)   return ProductFamily::MTL;
    if (family == "ARL"sv)   return ProductFamily::ARL;
    if (family == "PVC"sv)   return ProductFamily::PVC;
    if (family == "PVCVG"sv) return ProductFamily::PVCVG;
    if (family == "BMG"sv)   return ProductFamily::BMG;
    if (family == "LNL"sv)   return ProductFamily::LNL;
    if (family == "NVLP"sv)  return ProductFamily::NVLP;
    if (family == "CRI"sv)   return ProductFamily::CRI;

    // Architecture names are accepted too, for the generic family.
    auto hw = parseGen(family);
    if (hw != HW::Unknown) return genericProductFamily(hw);

    return ProductFamily::Unknown;
}

inline ngen::Product detectDevice(bool allow_emulation, bool *opencl_detected = nullptr)
{
    using namespace ngen;

    Product ret;
    ret.family = ProductFamily::Unknown;
    ret.stepping = 1 << 30; // Arbitrary large value

    if (opencl_detected) *opencl_detected = false;
    if (allow_emulation) {
        if (auto env = getenv("GEN")) {
            auto hw = parseGen(env);
            ret.family = genericProductFamily(hw);
            if (hw == HW::Unknown)
                fprintf(stderr, "WARNING: ignoring unknown GEN override '%s'\n", env);
        }

        if (auto env = getenv("FAMILY")) {
            ret.family = parseFamily(env);
            if (ret.family == ProductFamily::Unknown)
                fprintf(stderr, "WARNING: ignoring unknown product family override '%s'\n", env);
        }

        if (ret.family != ProductFamily::Unknown)
            ret.type = getPlatformType(ret.family);
    }

    // If not overridden, query from opencl
    if (ret.family == ProductFamily::Unknown) {
        cl_device_id dev = findDevice();

        // Detect currect HW info
        cl_int status;
        auto ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
        if (status != CL_SUCCESS)
            throw(std::runtime_error("OpenCL failed to create device context"));
        ret = OpenCLCodeGenerator<HW::Unknown>::detectHWInfo(ctx, dev);
        clReleaseContext(ctx);
        clReleaseDevice(dev);
        if (opencl_detected) *opencl_detected = true;
    }

    auto stepping = getenv("STEPPING");
    if (allow_emulation && stepping) ret.stepping = atoi(stepping);
    return ret;
}

inline int getDeviceEUCount()
{
    cl_device_id dev = findDevice();
    cl_uint eu_count;
    cl_int status = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                                    sizeof(eu_count), &eu_count, nullptr);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL failed to determine EU count");
    return eu_count;
}

inline int getDeviceMHz()
{
    auto envFreq = getenv("FREQ_OVERRIDE");
    if (envFreq) return atoi(envFreq);

    cl_device_id dev = findDevice();
    cl_uint mhz;
    cl_int status = clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                    sizeof(mhz), &mhz, nullptr);
    if (status != CL_SUCCESS || mhz == 0)
        throw std::runtime_error("OpenCL failed to determine clock frequency");
    return mhz;
}

inline bool simulation()
{
    if (getenv("FORCE_SIMULATION")) return true;
    auto scsr = getenv("SetCommandStreamReceiver");
    return scsr && (atoi(scsr) > 0);
}

inline bool aubCaptureEnabled()
{
    auto scsr = getenv("SetCommandStreamReceiver");
    return scsr && (atoi(scsr) & 1);
}

#endif
