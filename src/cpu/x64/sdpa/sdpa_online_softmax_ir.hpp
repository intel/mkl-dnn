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

#ifndef CPU_X64_SDPA_SDPA_ONLINE_SOFTMAX_IR_HPP
#define CPU_X64_SDPA_SDPA_ONLINE_SOFTMAX_IR_HPP

// Online-softmax epilogue for the online-softmax (flash) CPU SDPA (sdpa_online_softmax).
// The QK^T / PV matmuls stay on BRGEMM; only scale + select-mask +
// streaming-softmax + accumulator renormalization is JIT-built here with the
// x64 CPU IR (src/cpu/x64/ir). This file holds the SDPA epilogue math/layout;
// the IR framework itself is generic.

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_X64
#include <cstdint>

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdpa_softmax_ir {

// Arguments for the online-softmax tile epilogue kernel. A tile of `seq_q`
// score rows of `w` elements each is updated in place; per row, the running
// max/denominator and the tile's renormalization coefficient are read and
// written through scalar pointers, matching the per-row state the online-softmax SDPA
// driver carries across KV tiles. scores holds seq_q*w floats (row i at i*w);
// m/l/old_coef hold seq_q floats (row i at i); scale is shared by all rows.
// cond/fill drive the optional select mask: a masked-out lane takes fill. cond
// points at this tile's first column; its row stride is the cond_row_stride
// passed to the builder (columns are contiguous, one uint8 byte per score).
struct softmax_row_args_t {
    float *scores; // in: raw scores rows; out: normalized probabilities P
    const float *scale; // scalar softmax scale (shared by all rows)
    float *m; // in: running row max m_old; out: m_new (one per row)
    float *l; // in: running denominator l_old; out: l_new (one per row)
    float *old_coef; // out: corr*l_old/l_new per row (renormalizes running acc)
    const uint8_t *cond; // select condition bytes (tile column 0 of row 0)
    const float *fill; // scalar fill for masked-out lanes (shared by all rows)
};

// Arguments for the accumulator renormalization epilogue. After the softmax
// tile update produces old_coef per row, the running output accumulator is
// rescaled and the new tile's P*V contribution is added in one pass:
// acc = old_coef*acc + pv, over seq_q rows of hs_v head-size columns. acc and
// pv hold seq_q*hs floats (row i at i*hs); old_coef holds seq_q floats.
struct acc_renorm_args_t {
    float *acc; // in: running output acc; out: renormalized acc
    const float *pv; // in: this tile's P*V contribution
    const float *old_coef; // in: per-row renorm coefficient (corr*l_old/l_new)
};

// Creates the online-softmax tile epilogue kernel for a tile of `seq_q` score
// rows of width `w` (the IR builder in the .cpp holds the exact math and the
// optional select-mask handling; `cond_row_stride` < 0 defaults to `w`). Call
// create_kernel() on the result, then invoke via operator()(const
// softmax_row_args_t *). Returns the generic jit_generator_t base so the
// concrete kernel stays private to the .cpp (mirroring brgemm_ir.cpp).
DNNL_API jit_generator_t *create_softmax_tile_ir_kernel(cpu_isa_t isa,
        int seq_q, int w, bool has_select, bool invert_select,
        int cond_row_stride);

// Creates the accumulator renormalization kernel (acc = old_coef*acc + pv) for
// a tile of `seq_q` rows of head size `hs`. Call create_kernel(), then invoke
// via operator()(const acc_renorm_args_t *).
DNNL_API jit_generator_t *create_acc_renorm_ir_kernel(
        cpu_isa_t isa, int seq_q, int hs);

} // namespace sdpa_softmax_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_X64
#endif // CPU_X64_SDPA_SDPA_ONLINE_SOFTMAX_IR_HPP
