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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_FUSER_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_FUSER_HPP

#include <cstdint>
#include <vector>

#include "gemmstone/config.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

// Markers for patch sections.
static constexpr uint32_t sigilStart = 0xCAFEFADE;
static constexpr uint32_t sigilEnd = 0xFADECAFE;
static constexpr const char *sigilBinary = "@_u_@";

// Fuse source-embedded microkernels into the compiled host kernel's binary.
// Each is checked against the host thread payload; an overlap throws, leaving
// `binary` partially fused. Returns false if none could be checked (fused but
// unvalidated).
bool fuse(std::vector<uint8_t> &binary, const char *source, int grfBytes);
bool hasMicrokernels(const char *source);

// Transitional overload for legacy callers: skips payload validation.
inline bool fuse(std::vector<uint8_t> &binary, const char *source) {
    return fuse(binary, source, 0);
}

}
GEMMSTONE_NAMESPACE_END

#endif
