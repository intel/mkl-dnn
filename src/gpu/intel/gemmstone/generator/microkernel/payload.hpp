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

#ifndef GEMMSTONE_GENERATOR_MICROKERNEL_PAYLOAD_HPP
#define GEMMSTONE_GENERATOR_MICROKERNEL_PAYLOAD_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gemmstone/config.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

// A thread payload argument, at a byte offset within its payload block.
struct PayloadArgument {
    uint32_t offset = 0;
    uint32_t size = 0;
};

struct KernelInfo {
    std::string name;
    uint32_t inlineDataBytes = 0;
    uint32_t perThreadBytes = 0;
    // Kept per-argument: inline-data padding shifts each offset individually,
    // and .ze_info may list the arguments before inline_data_payload_size.
    std::vector<PayloadArgument> payload;
};

// Reads kernel thread payload layouts from zebin .ze_info metadata.
// Unrecognized keys are skipped; returns false if no kernel list was found.
bool parseZeInfo(const char *text, size_t length, std::vector<KernelInfo> &kernels);

// Thread payload geometry, in GRF bytes relative to r0. `grfBytes` must be
// supplied: .ze_info records grf_count, not the GRF width.
uint32_t crossthreadBase(const KernelInfo &kernel, int grfBytes);
uint32_t payloadEndBytes(const KernelInfo &kernel, int grfBytes);

}
GEMMSTONE_NAMESPACE_END

#endif
