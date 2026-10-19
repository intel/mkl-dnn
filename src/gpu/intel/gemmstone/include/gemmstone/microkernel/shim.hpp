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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_SHIM_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_SHIM_HPP

#include <string>

#include "gemmstone/microkernel/package.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {


enum class HostLanguage { None, OpenCL_C, SYCL, vISA };

struct ShimOptions {
    std::string decorator;
    int subgroupSize = 0;
    bool copyScalarArgs = true;
    bool copyTensorArgs = false;
    bool useTileOps = false;
    uint32_t microkernelID = 0;
};

std::string generateShim(const Package &package, HostLanguage language,
        const ShimOptions &options = ShimOptions());

}
GEMMSTONE_NAMESPACE_END

#endif
