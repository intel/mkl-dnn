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

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pieces/address_setup.cxx"
#include "pieces/asm_helpers.cxx"
#include "pieces/atomic_fusions.cxx"
#include "pieces/c_update.cxx"
#include "pieces/common.cxx"
#include "pieces/copy.cxx"
#include "pieces/driver_info.cxx"
#include "pieces/emulation.cxx"
#include "pieces/gemm_microkernel.cxx"
#include "pieces/gemm_setup.cxx"
#include "pieces/gemm.cxx"
#include "pieces/k_loop_setup.cxx"
#include "pieces/k_loop.cxx"
#include "pieces/layout_setup.cxx"
#include "pieces/l3_prefetch.cxx"
#include "pieces/masks.cxx"
#include "pieces/math_helpers.cxx"
#include "pieces/matrix_access.cxx"
#include "pieces/matrix_multiply.cxx"
#include "pieces/monolithic_k_loop_dpasw.cxx"
#include "pieces/post_ops.cxx"
#include "pieces/register_allocation.cxx"
#include "pieces/remask.cxx"
#include "pieces/row_column_sums.cxx"
#include "pieces/state_utils.cxx"
#include "pieces/stream_k.cxx"
#include "pieces/tlb_warmup.cxx"
#include "pieces/walk_orders.cxx"

#include "pieces/quantization.cxx"

#ifdef STANDALONE
#include "pieces/debug_tools.cxx"
#include "pieces/notes.cxx"
#endif

GEMMSTONE_NAMESPACE_START

template <HW hw>
constexpr typename Generator<hw>::status_stream::Endl Generator<hw>::status_stream::endl;

#include "pieces/hw_template_instantiations.cxx"
GEMMSTONE_NAMESPACE_END
