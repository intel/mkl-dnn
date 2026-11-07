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

#ifndef CPU_X64_BRGEMM_JIT_BRGEMM_AMX_UKER_HPP
#define CPU_X64_BRGEMM_JIT_BRGEMM_AMX_UKER_HPP

// AMX/ACE BRGEMM kernel (fully unrolled).

#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Caller owns the pointer. Callers are expected to check
// `brgemm_desc_t::can_dispatch_uker()` first.
brgemm_kernel_t *create_brgemm_amx_uker_kernel(const brgemm_desc_t &brg);

// Returns true if the ACE uker for `brg` fits the size threshold once all
// the loops it may roll are rolled.
bool ace_uker_small_enough(const brgemm_desc_t &brg);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
