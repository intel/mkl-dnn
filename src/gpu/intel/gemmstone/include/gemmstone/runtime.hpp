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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_RUNTIME_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_RUNTIME_HPP

#include "gemmstone/config.hpp"

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
#include <vector>
#endif

#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
#include <sycl/sycl.hpp>
#endif

#ifdef GEMMSTONE_WITH_L0_RUNTIME
#include <level_zero/ze_api.h>
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
#include <CL/cl.h>
#endif

#include "gemmstone/config.hpp"
#include "gemmstone/dsl/kernel.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "ngen_interface.hpp"

GEMMSTONE_NAMESPACE_START

struct GEMMKernelDesc {
  GEMMKernelDesc(const GEMMProblem &problem, const GEMMStrategy &strategy,
                 const ngen::InterfaceHandler &iface,
                 const dsl::kernel::options_t &options)
        : problem(problem) , strategy(strategy) , iface(iface), options(options) {}

    GEMMProblem problem;
    GEMMStrategy strategy;
    ngen::InterfaceHandler iface;
    dsl::kernel::options_t options;
};

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
std::string make_asm(const GEMMKernelDesc &desc);
#endif

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
std::vector<uint8_t> make_binary(const GEMMKernelDesc &desc);
#endif

#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
std::vector<uint8_t> make_binary(const GEMMKernelDesc &desc, sycl::device device, sycl::context context);
sycl::kernel make_kernel(const GEMMKernelDesc &desc, sycl::device device, sycl::context context);
#endif

#ifdef GEMMSTONE_WITH_L0_RUNTIME
struct LevelZeroKernelAndModule {
    ze_kernel_handle_t kernel;
    ze_module_handle_t module;
};
dsl::hw_t get_hardware(ze_device_handle_t device, ze_context_handle_t context = nullptr);
std::vector<uint8_t> make_binary(const GEMMKernelDesc &desc, ze_device_handle_t device, ze_context_handle_t context);
LevelZeroKernelAndModule make_kernel(const GEMMKernelDesc &desc, ze_device_handle_t device, ze_context_handle_t context);
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
dsl::hw_t get_hardware(cl_device_id device, cl_context context = nullptr);
std::vector<uint8_t> make_binary(const GEMMKernelDesc &desc, cl_device_id device, cl_context context);
cl_kernel make_kernel(const GEMMKernelDesc &desc, cl_device_id device, cl_context context);
#endif

GEMMSTONE_NAMESPACE_END
#endif
