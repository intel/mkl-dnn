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

#ifndef CPU_X64_SDPA_SDP_BLOCKED_SELECT_IR_HPP
#define CPU_X64_SDPA_SDP_BLOCKED_SELECT_IR_HPP

// Standalone IR-based select-mask pre-pass kernel for the blocked CPU SDPA
// driver (sdp_blocked_driver_t, sdp_blocked_driver.hpp). The blocked driver
// reuses the stock jit_uni_softmax primitive for the max/exp/normalize, but
// that primitive cannot apply a pre-softmax select mask. So, when a select
// mask is present and not already folded into mm1, the scores tile has the
// mask applied in a separate pre-pass before softmax runs.
//
// This file builds that pre-pass as a small, self-contained JIT kernel with the
// generic x64 CPU IR framework (src/cpu/x64/ir). It deliberately does NOT share
// any code with the fused driver's online-softmax epilogue
// (sdp_fused_softmax_ir.hpp): this is a local, select-only kernel with no
// softmax, no scale and no online/streaming recurrence. It targets AVX2 and
// AVX-512 through the IR, so no intrinsics are needed and tails are handled with
// masked loads/stores.

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_X64
#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdp_blocked_select_ir {

using namespace dnnl::impl::cpu::x64::ir;

inline cpu_isa_t isa() {
    return mayiuse(avx512_core) ? avx512_core : avx2;
}

inline int simd_w() {
    return isa_max_vlen(isa()) / (int)sizeof(float);
}

// Arguments for the select pre-pass kernel. A tile of `n_rows` score rows of
// `w` elements each (the kernel bakes `w`) is updated in place; a masked-out
// lane takes `fill`. scores points at the first row (contiguous, row stride
// baked into the kernel); cond points at the matching condition row (one uint8
// byte per score, columns contiguous, row stride baked into the kernel). fill
// is a loop-invariant scalar shared by every row and lane.
struct select_row_args_t {
    float *scores; // in/out: score rows; masked-out lanes become fill
    const uint8_t *cond; // select condition bytes (row 0, column 0)
    const float *fill; // scalar fill for masked-out lanes
    dim_t n_rows; // number of score rows to process
};

// Builds the select pre-pass for a tile of `n_rows` (runtime) score rows, each
// of width `w` (any w >= 1; the ragged tail beyond the last full simd_w block
// uses masked loads/stores). `scores_row_stride` / `cond_row_stride` are the
// score and condition row strides in elements; the kernel advances both one row
// per iteration (a 0 condition stride means every row reads the same condition
// row, i.e. a broadcast-over-rows condition). Per block the op chain is:
// load scores -> widen the uint8 condition and turn it into a lane mask
// (vload_u8 -> vcmp_ne_zero) -> vblend the broadcast `fill` into the masked-out
// lanes -> store scores. Which lanes are masked out follows the driver:
// fusiable keeps the score where cond != 0, non-fusiable where cond == 0.
inline ir_t build_select_ir(
        int w, bool fusiable, dim_t scores_row_stride, dim_t cond_row_stride) {
    const int n_blk = w / simd_w();
    const int tail = w % simd_w();
    const dim_t vbytes = simd_w() * (dim_t)sizeof(float);
    const dim_t tail_off = n_blk * vbytes;

    ir_t ir;

    // Row pointers: advanced one row per loop iteration.
    const vreg_t sc_ptr = ir.new_gpr();
    ir.load_param(sc_ptr, offsetof(select_row_args_t, scores));
    const vreg_t cond_ptr = ir.new_gpr();
    ir.load_param(cond_ptr, offsetof(select_row_args_t, cond));
    // Loop invariant: the fill scalar is shared by every row and lane.
    const vreg_t fill_ptr = ir.new_gpr();
    ir.load_param(fill_ptr, offsetof(select_row_args_t, fill));
    // Runtime row count drives the loop.
    const vreg_t rows = ir.new_gpr();
    ir.load_param(rows, offsetof(select_row_args_t, n_rows));

    // One mask, reused by every masked tail op and every row, active for `tail`.
    vreg_t mask = vreg_t::none;
    if (tail) {
        mask = ir.new_mask();
        ir.set_mask_imm(mask, tail);
    }

    // Apply the select mask to the single row at the current pointers.
    auto row_body = [&]() {
        const vreg_t fill_bc = ir.new_vec(data_type::f32);
        ir.vload_bcast(fill_bc, fill_ptr, 0, data_type::f32);

        // One block of `n` elements at `sc_off` bytes into the row's scores and
        // `cond_off` bytes into the row's condition. `m` is the tail mask vreg,
        // or `none` for a full block.
        auto apply_block = [&](dim_t sc_off, dim_t cond_off, int n, vreg_t m) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            if (m == vreg_t::none)
                ir.vload(blk, sc_ptr, sc_off, data_type::f32);
            else
                ir.vload_masked(blk, sc_ptr, sc_off, m, data_type::f32);

            const vreg_t cond = ir.new_vec(data_type::s32);
            ir.vload_u8(cond, cond_ptr, cond_off, n);
            const vreg_t cmask = ir.new_mask();
            ir.vcmp_ne_zero(cmask, cond);

            vreg_t out;
            if (fusiable) {
                // Keep the score where cond != 0, else fill: start from fill
                // and blend the score into the cond != 0 lanes.
                const vreg_t sel = ir.new_vec(data_type::f32);
                ir.vbcast(sel, fill_bc);
                ir.vblend(sel, blk, cmask);
                out = sel;
            } else {
                // Keep the score where cond == 0, else fill: blend fill into
                // the cond != 0 lanes.
                ir.vblend(blk, fill_bc, cmask);
                out = blk;
            }

            if (m == vreg_t::none)
                ir.vstore(sc_ptr, sc_off, out, data_type::f32);
            else
                ir.vstore_masked(sc_ptr, sc_off, out, m, data_type::f32);
        };

        for (int b = 0; b < n_blk; ++b)
            apply_block(
                    b * vbytes, (dim_t)b * simd_w(), simd_w(), vreg_t::none);
        if (tail) apply_block(tail_off, (dim_t)n_blk * simd_w(), tail, mask);

        // Advance to the next row. The last iteration's advance is harmless
        // (the pointers are not read again).
        ir.add_imm(sc_ptr, scores_row_stride * (dim_t)sizeof(float));
        ir.add_imm(cond_ptr, cond_row_stride);
    };

    emit_loop_reg(ir, rows, row_body);

    return ir;
}

// JIT kernel that runs a select pre-pass IR: allocate registers, emit code and
// finalize. The IR uses no eltwise and no post-ops, so neither an eltwise
// injector nor a post-ops injector is wired up. Construct with an IR from
// build_select_ir(), call create_kernel(), then invoke via
// operator()(const select_row_args_t *).
class select_ir_kernel_t : public jit_generator_t {
public:
    select_ir_kernel_t(ir_t ir)
        : jit_generator_t("sdp_blocked_select_ir", isa()), ir_(std::move(ir)) {}

    const char *name() const override { return "sdp_blocked_select_ir_kernel"; }
    const char *source_file() const override { return __FILE__; }

protected:
    void generate() override {
        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        // Scratch registers the emitter reserves for spill handling. They are
        // not part of the allocatable register pool.
        const int gpr_scratch0 = 10, gpr_scratch1 = 11;
        const int vec_scratch0 = 13, vec_scratch1 = 14, vec_scratch2 = 15;
        // No post-ops injector here, but make_reg_config still reserves an
        // opmask on AVX-512; name one so it is excluded from allocation.
        const int mask_scratch = 1;

        const reg_config_t reg_cfg = make_reg_config(isa(), param_idx, rsp_idx,
                {gpr_scratch0, gpr_scratch1},
                {vec_scratch0, vec_scratch1, vec_scratch2}, {mask_scratch});

        const reg_alloc_result_t alloc = allocate_registers(ir_, reg_cfg.pools);

        preamble();

        const int frame = (int)utils::rnd_up(alloc.frame_bytes, 16);
        if (frame > 0) sub(rsp, frame);

        // No eltwise and no attribute post-ops in this IR.
        data_section_t data;
        emit(*this, ir_, alloc, reg_cfg, data, /*postops=*/nullptr);

        if (frame > 0) add(rsp, frame);

        postamble();

        emit_data_section(*this, data);
    }

private:
    ir_t ir_;
};

} // namespace sdp_blocked_select_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_X64
#endif // CPU_X64_SDPA_SDP_BLOCKED_SELECT_IR_HPP
