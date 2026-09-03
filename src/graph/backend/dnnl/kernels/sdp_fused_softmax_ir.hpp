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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_FUSED_SOFTMAX_IR_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_FUSED_SOFTMAX_IR_HPP

// IR-based online-softmax epilogue for the fused CPU SDPA kernel
// (sdp_fused_brgemm_kernel_t). The QK^T / PV matmuls stay on BRGEMM; only the
// scale + select-mask + streaming-softmax + accumulator renormalization is
// JIT-built here with the x64 CPU IR framework (src/cpu/x64/ir). This is the
// only file that knows the SDPA epilogue math and data layout; everything in
// the IR framework is generic infrastructure.

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_X64
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/x64/ir/eltwise_injector.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace sdp_softmax_ir {

using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::cpu::x64::ir;

// The IR framework has only an AVX2 backend today, so the builders block at
// that ISA's vector width; a wider backend would take the isa (or its vlen) as
// an arg.
constexpr int simd_w = cpu_isa_traits_t<avx2>::vlen / (int)sizeof(float);

// Arguments for the online-softmax tile epilogue kernel. A tile of `seq_q`
// score rows of `w` elements each is updated in place; per row, the running
// max/denominator and the tile's renormalization coefficient are read and
// written through scalar pointers, matching the per-row state the fused SDPA
// kernel carries across KV tiles. scores holds seq_q*w floats (row i at i*w);
// m/l/old_coef hold seq_q floats (row i at i); scale is shared by all rows.
// cond/fill drive the optional select mask: a masked-out lane takes fill.
struct softmax_row_args_t {
    float *scores; // in: raw scores rows; out: normalized probabilities P
    const float *scale; // scalar softmax scale (shared by all rows)
    float *m; // in: running row max m_old; out: m_new (one per row)
    float *l; // in: running denominator l_old; out: l_new (one per row)
    float *old_coef; // out: corr*l_old/l_new per row (renormalizes running acc)
    const uint8_t *cond; // select condition bytes (seq_q*w, row i at i*w)
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

// Builds the online-softmax epilogue for a tile of `seq_q` score rows, each of
// width `w` (any w >= 1; the ragged tail beyond the last full simd_w block is
// handled with masked loads/stores). Mirrors the per-row scalar epilogue in
// sdp_fused_brgemm.cpp for a single KV tile. With `has_select`, each block also
// gets the attention select mask applied right after scaling: uint8 condition
// bytes are widened and turned into a lane mask (vload_u8 -> vcmp_ne_zero),
// then vblend selects the broadcast `fill` scalar into the masked-out lanes.
// Which lanes are masked out follows the fused kernel: fusiable keeps the score
// where cond != 0, non-fusiable where cond == 0.
// The rows are processed by a loop over seq_q; the score and per-row state
// pointers advance one row per iteration. Per row the op chain is: scale ->
// (select) -> running row max -> exp(scaled - m_new) -> running denominator ->
// divide by l_new. The scalar running state is kept in broadcast vectors so
// the per-row arithmetic reuses the vector ops. In the tail block only some
// lanes hold real scores; before the max and sum reductions the padding lanes
// are set to a neutral value (m_old for the max so they cannot exceed the
// running max, 0 for the sum so they add nothing). The first KV tile (m_old ==
// -inf, l_old == 0) needs no explicit guard: the eltwise exp saturates -inf to
// exactly 0, so corr == 0 and clo == l_old*corr == 0 fall out, and m_new is
// just the tile max. The scale, fill and tail mask are loop invariant, set up
// once; the mask vreg stays live across all rows (masked ops assert it is never
// spilled).
inline ir_t build_softmax_tile_ir(
        int seq_q, int w, bool has_select = false, bool fusiable = true) {
    const int n_blk = w / simd_w;
    const int tail = w % simd_w;
    const dim_t vbytes = simd_w * (dim_t)sizeof(float);
    const dim_t tail_off = n_blk * vbytes;
    const dim_t fsz = (dim_t)sizeof(float);

    ir_t ir;

    // Row pointers: advanced one row per loop iteration.
    const vreg_t sc_ptr = ir.new_gpr();
    ir.load_param(sc_ptr, offsetof(softmax_row_args_t, scores));
    const vreg_t m_ptr = ir.new_gpr();
    ir.load_param(m_ptr, offsetof(softmax_row_args_t, m));
    const vreg_t l_ptr = ir.new_gpr();
    ir.load_param(l_ptr, offsetof(softmax_row_args_t, l));
    const vreg_t oc_ptr = ir.new_gpr();
    ir.load_param(oc_ptr, offsetof(softmax_row_args_t, old_coef));
    // Loop invariant: the scale is shared by every row.
    const vreg_t scale_ptr = ir.new_gpr();
    ir.load_param(scale_ptr, offsetof(softmax_row_args_t, scale));

    // Select mask inputs: cond advances one row per iteration (one byte per
    // score), fill is a loop-invariant scalar shared by every row.
    vreg_t cond_ptr = vreg_t::none;
    vreg_t fill_ptr = vreg_t::none;
    if (has_select) {
        cond_ptr = ir.new_gpr();
        ir.load_param(cond_ptr, offsetof(softmax_row_args_t, cond));
        fill_ptr = ir.new_gpr();
        ir.load_param(fill_ptr, offsetof(softmax_row_args_t, fill));
    }

    // One mask, reused by every masked op and every row, active for `tail`.
    vreg_t mask = vreg_t::none;
    if (tail) {
        mask = ir.new_mask();
        ir.set_mask_imm(mask, tail);
    }

    // Online-softmax epilogue for the single row at the current pointers.
    auto row_body = [&]() {
        // Scratch shared by both horizontal reductions (overwritten each time).
        const vreg_t ws = ir.new_vec(data_type::f32);

        // Broadcast the scalar inputs so scalar arithmetic reuses vector ops.
        const vreg_t scale_bc = ir.new_vec(data_type::f32);
        ir.vload_masked(scale_bc, scale_ptr, 0, vreg_t::none, 1);
        ir.vbcast(scale_bc, scale_bc);

        vreg_t fill_bc = vreg_t::none;
        if (has_select) {
            fill_bc = ir.new_vec(data_type::f32);
            ir.vload_masked(fill_bc, fill_ptr, 0, vreg_t::none, 1);
            ir.vbcast(fill_bc, fill_bc);
        }

        // Apply the select mask to one scaled block of `n` elements at byte
        // offset `cond_off` into this row's condition bytes, and return the
        // vreg holding the result (unchanged when there is no select).
        auto apply_select = [&](vreg_t blk, dim_t cond_off, int n) -> vreg_t {
            if (!has_select) return blk;
            const vreg_t cond = ir.new_vec(data_type::s32);
            ir.vload_u8(cond, cond_ptr, cond_off, n);
            const vreg_t cmask = ir.new_vec(data_type::s32);
            ir.vcmp_ne_zero(cmask, cond);
            if (fusiable) {
                // Keep the score where cond != 0: cmask ? score : fill.
                const vreg_t sel = ir.new_vec(data_type::f32);
                ir.vbcast(sel, fill_bc);
                ir.vblend(sel, blk, cmask);
                return sel;
            }
            // Keep the score where cond == 0: cmask ? fill : score.
            ir.vblend(blk, fill_bc, cmask);
            return blk;
        };

        const vreg_t m_old = ir.new_vec(data_type::f32);
        ir.vload_masked(m_old, m_ptr, 0, vreg_t::none, 1);
        ir.vbcast(m_old, m_old);

        const vreg_t l_old = ir.new_vec(data_type::f32);
        ir.vload_masked(l_old, l_ptr, 0, vreg_t::none, 1);
        ir.vbcast(l_old, l_old);

        // Pass 1: scale each block, store it back, and fold it into the running
        // max (seeded with m_old so the reduction yields m_new directly).
        const vreg_t rmax = ir.new_vec(data_type::f32);
        ir.vbcast(rmax, m_old);
        for (int b = 0; b < n_blk; b++) {
            vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vmul(blk, scale_bc);
            blk = apply_select(blk, (dim_t)b * simd_w, simd_w);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
            ir.vmax(rmax, blk);
        }
        if (tail) {
            vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vmul(blk, scale_bc);
            blk = apply_select(blk, (dim_t)n_blk * simd_w, tail);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
            // Unused lanes take m_old so they never win the max.
            const vreg_t tmax = ir.new_vec(data_type::f32);
            ir.vbcast(tmax, m_old);
            ir.vblend(tmax, blk, mask);
            ir.vmax(rmax, tmax);
        }
        ir.vhreduce_max(rmax, ws); // rmax lane 0 = m_new
        const vreg_t m_new = ir.new_vec(data_type::f32);
        ir.vbcast(m_new, rmax);

        // corr = exp(m_old - m_new): rescales previous tiles' contributions.
        const vreg_t corr = ir.new_vec(data_type::f32);
        ir.vbcast(corr, m_old);
        ir.vsub(corr, m_new);
        ir.vexp(corr);

        // Pass 2: P_unnorm = exp(scaled - m_new); accumulate the tile denom.
        const vreg_t rsum = ir.new_vec(data_type::f32);
        ir.vzero(rsum);
        for (int b = 0; b < n_blk; b++) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vsub(blk, m_new);
            ir.vexp(blk);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
            ir.vadd(rsum, blk);
        }
        if (tail) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vsub(blk, m_new);
            ir.vexp(blk);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
            // Unused lanes take 0 so they add nothing to the denominator.
            const vreg_t tsum = ir.new_vec(data_type::f32);
            ir.vzero(tsum);
            ir.vblend(tsum, blk, mask);
            ir.vadd(rsum, tsum);
        }
        ir.vhreduce(rsum, ws); // rsum lane 0 = tile_sum
        const vreg_t tile_sum = ir.new_vec(data_type::f32);
        ir.vbcast(tile_sum, rsum);

        // clo = l_old*corr; l_new = clo + tile_sum; old_coef = clo / l_new.
        const vreg_t clo = ir.new_vec(data_type::f32);
        ir.vbcast(clo, l_old);
        ir.vmul(clo, corr);

        const vreg_t l_new = ir.new_vec(data_type::f32);
        ir.vbcast(l_new, clo);
        ir.vadd(l_new, tile_sum);

        const vreg_t old_coef = ir.new_vec(data_type::f32);
        ir.vbcast(old_coef, clo);
        ir.vdiv(old_coef, l_new);

        // Pass 3: normalize P by the running denominator.
        for (int b = 0; b < n_blk; b++) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vdiv(blk, l_new);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
        }
        if (tail) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vdiv(blk, l_new);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
        }

        // Write back this row's scalar running state (lane 0).
        ir.vstore_masked(m_ptr, 0, m_new, vreg_t::none, 1);
        ir.vstore_masked(l_ptr, 0, l_new, vreg_t::none, 1);
        ir.vstore_masked(oc_ptr, 0, old_coef, vreg_t::none, 1);
    };

    // Advance every row pointer to the next row.
    auto advance_row = [&]() {
        ir.add_imm(sc_ptr, w * fsz);
        ir.add_imm(m_ptr, fsz);
        ir.add_imm(l_ptr, fsz);
        ir.add_imm(oc_ptr, fsz);
        if (has_select) ir.add_imm(cond_ptr, w); // one uint8 byte per score
    };

    emit_loop_imm(ir, seq_q, row_body, advance_row);

    return ir;
}

// Builds the accumulator renormalization for a tile of `seq_q` rows, each of
// head size `hs` (any hs >= 1; the ragged tail beyond the last full simd_w
// block uses masked loads/stores). Mirrors the acc rescale in the fused SDPA
// kernel that follows the softmax epilogue: acc = old_coef*acc + pv. Rows are
// processed by a loop over seq_q; the acc, pv and old_coef pointers advance one
// row per iteration. old_coef is broadcast so the per-row scalar reuses the
// vector ops; no reduction is needed, so the tail needs no lane neutralization
// (masked ld/st touch only the active columns).
inline ir_t build_acc_renorm_ir(int seq_q, int hs) {
    const int n_blk = hs / simd_w;
    const int tail = hs % simd_w;
    const dim_t vbytes = simd_w * (dim_t)sizeof(float);
    const dim_t tail_off = n_blk * vbytes;
    const dim_t fsz = (dim_t)sizeof(float);

    ir_t ir;

    // Row pointers: advanced one row per loop iteration.
    const vreg_t acc_ptr = ir.new_gpr();
    ir.load_param(acc_ptr, offsetof(acc_renorm_args_t, acc));
    const vreg_t pv_ptr = ir.new_gpr();
    ir.load_param(pv_ptr, offsetof(acc_renorm_args_t, pv));
    const vreg_t oc_ptr = ir.new_gpr();
    ir.load_param(oc_ptr, offsetof(acc_renorm_args_t, old_coef));

    // One mask, reused by every masked op and every row, active for `tail`.
    vreg_t mask = vreg_t::none;
    if (tail) {
        mask = ir.new_mask();
        ir.set_mask_imm(mask, tail);
    }

    // acc = old_coef*acc + pv for the single row at the current pointers.
    auto row_body = [&]() {
        const vreg_t oc_bc = ir.new_vec(data_type::f32);
        ir.vload_masked(oc_bc, oc_ptr, 0, vreg_t::none, 1);
        ir.vbcast(oc_bc, oc_bc);

        for (int b = 0; b < n_blk; b++) {
            const vreg_t acc = ir.new_vec(data_type::f32);
            ir.vload(acc, acc_ptr, b * vbytes);
            ir.vmul(acc, oc_bc);
            const vreg_t pv = ir.new_vec(data_type::f32);
            ir.vload(pv, pv_ptr, b * vbytes);
            ir.vadd(acc, pv);
            ir.vstore_masked(acc_ptr, b * vbytes, acc, vreg_t::none, simd_w);
        }
        if (tail) {
            const vreg_t acc = ir.new_vec(data_type::f32);
            ir.vload_masked(acc, acc_ptr, tail_off, mask, tail);
            ir.vmul(acc, oc_bc);
            const vreg_t pv = ir.new_vec(data_type::f32);
            ir.vload_masked(pv, pv_ptr, tail_off, mask, tail);
            ir.vadd(acc, pv);
            ir.vstore_masked(acc_ptr, tail_off, acc, mask, tail);
        }
    };

    // Advance every row pointer to the next row.
    auto advance_row = [&]() {
        ir.add_imm(acc_ptr, hs * fsz);
        ir.add_imm(pv_ptr, hs * fsz);
        ir.add_imm(oc_ptr, fsz);
    };

    emit_loop_imm(ir, seq_q, row_body, advance_row);

    return ir;
}

// JIT kernel that runs the full IR pipeline for an epilogue IR: allocate
// registers, wire an eltwise injector per algorithm the IR uses (softmax needs
// exp), emit code, and finalize. Construct with an IR from one of the builders
// above, call create_kernel(), then invoke via operator()(const args_t *).
// Only AVX2 is supported today; the emitter picks the lowering by that ISA.
class softmax_ir_kernel_t : public jit_generator_t {
public:
    softmax_ir_kernel_t(ir_t ir)
        : jit_generator_t("sdp_softmax_ir", avx2), ir_(std::move(ir)) {}

    const char *name() const override { return "sdp_softmax_ir_kernel"; }
    const char *source_file() const override { return __FILE__; }

protected:
    void generate() override {
        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        // Scratch registers the emitter reserves for spill handling. They are
        // not part of the register pool.
        const int gpr_scratch0 = 10, gpr_scratch1 = 11;
        const int vec_scratch0 = 13, vec_scratch1 = 14, vec_scratch2 = 15;

        const reg_config_t reg_cfg = make_reg_config(avx2, param_idx, rsp_idx,
                {gpr_scratch0, gpr_scratch1},
                {vec_scratch0, vec_scratch1, vec_scratch2});

        const reg_alloc_result_t alloc = allocate_registers(ir_, reg_cfg.pools);

        // One eltwise injector per algorithm the IR uses, discovered by
        // scanning the veltwise ops. Each one saves and restores every register
        // it borrows, so it takes no part in the IR register allocation. A
        // single generic callback dispatches by algorithm.
        std::map<alg_kind_t, std::unique_ptr<eltwise_injector_t>>
                eltwise_injectors;
        for (const auto &op : ir_.ops()) {
            if (op.kind != op_kind_t::veltwise) continue;
            const auto alg = (alg_kind_t)op.imm;
            if (eltwise_injectors.count(alg)) continue;
            eltwise_injectors.emplace(alg,
                    std::unique_ptr<eltwise_injector_t>(
                            new eltwise_injector_t(*this, avx2, alg,
                                    /* alpha = */ 0.f, /* beta = */ 0.f,
                                    /* scale = */ 1.f)));
        }
        eltwise_fn_t emit_eltwise;
        if (!eltwise_injectors.empty()) {
            emit_eltwise = [&](alg_kind_t alg, int vec_phys) {
                eltwise_injectors.at(alg)->apply(vec_phys);
            };
        }

        preamble();

        const int frame = (int)utils::rnd_up(alloc.frame_bytes, 16);
        if (frame > 0) sub(rsp, frame);

        // This epilogue has no attribute post-ops, so no post-ops injector.
        inject_postops_fn_t emit_injector;
        data_section_t data;
        emit(*this, ir_, alloc, reg_cfg, data, emit_injector, emit_eltwise);

        if (frame > 0) add(rsp, frame);

        postamble();

        emit_data_section(*this, data);

        // Each eltwise injector's constant table follows the postamble.
        for (auto &kv : eltwise_injectors)
            kv.second->prepare_table();
    }

private:
    ir_t ir_;
};

} // namespace sdp_softmax_ir
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // DNNL_X64
#endif // GRAPH_BACKEND_DNNL_KERNELS_SDP_FUSED_SOFTMAX_IR_HPP
