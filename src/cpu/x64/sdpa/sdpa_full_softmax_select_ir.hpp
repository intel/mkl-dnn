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

#ifndef CPU_X64_SDPA_SDPA_FULL_SOFTMAX_SELECT_IR_HPP
#define CPU_X64_SDPA_SDPA_FULL_SOFTMAX_SELECT_IR_HPP

// Select-mask pre-pass kernel for the full-softmax CPU SDPA. The full-softmax
// path reuses jit_uni_softmax, which cannot apply a pre-softmax select mask,
// so when a select mask is present and not folded into mm1 the scores tile has
// the mask applied in a separate pass before softmax.

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_X64
#include <cstdint>

#include "common/c_types_map.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdpa_full_softmax_select_ir {

inline cpu_isa_t isa() {
    return mayiuse(avx512_core) ? avx512_core : avx2;
}

inline int simd_w() {
    return isa_max_vlen(isa()) / (int)sizeof(float);
}

// Arguments for the select pre-pass kernel. A tile of `n_rows` score rows of
// `w` elements each (the kernel bakes `w`) is updated in place; a masked-out
// key's score is replaced with `fill`. `scores` points at query 0, key 0.
// Within each query row, the `w` key scores are adjacent floats; the baked-in
// score row stride advances to the next query. `cond` points at the matching
// condition for query 0, key 0; key conditions are adjacent uint8 values, and
// its baked-in row stride advances to the next query (or is zero to reuse the
// same condition row). `fill` is shared by every row and key.
struct select_row_args_t {
    float *scores; // in/out: score rows; masked-out lanes become fill
    const uint8_t *cond; // select condition bytes (row 0, column 0)
    const float *fill; // scalar fill for masked-out lanes
    dim_t n_rows; // number of score rows to process
};

// Creates the select pre-pass kernel for a tile of `n_rows` (runtime) score
// rows of width `w` (the IR builder in the .cpp holds the exact math; a 0
// `cond_row_stride` broadcasts one condition row over all rows). With
// `invert_select` false, keep scores where cond != 0; with it true, where
// cond == 0. Call create_kernel() on the result, then invoke via
// operator()(const select_row_args_t *). Returns the generic jit_generator_t
// base so the concrete kernel stays private to the .cpp (mirroring brgemm_ir.cpp).
jit_generator_t *create_select_ir_kernel(dim_t w, bool invert_select,
        dim_t scores_row_stride, dim_t cond_row_stride);

} // namespace sdpa_full_softmax_select_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_X64
#endif // CPU_X64_SDPA_SDPA_FULL_SOFTMAX_SELECT_IR_HPP
