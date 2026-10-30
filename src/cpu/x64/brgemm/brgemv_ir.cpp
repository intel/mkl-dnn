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

// The IR-based GEMV kernel converts a brgemm descriptor into IR, then runs the
// register allocator and code emitter.
//
// This is the only file that knows about GEMV-specific math and data layout.
// Everything in IR is generic infrastructure that can be reused for other
// problems.
//
// The brgemv_ir_supported() function checks whether a problem matches the
// current requirements.

#include <cstdint>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemv_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

#define VCONDCHECK_BRGEMV_IR(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemv_ir, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

// Fixed configuration used during IR generation for the GEMV builder.
//
//   m / k        - dimensions of matrix A (GEMV always has n = 1)
//   lda / incy   - leading dimensions of A and stride between elements of `y`
//   max_bs       - maximum batch size known at IR generation time
//   trans_a      - whether A is transposed (K-major). It selects how an M
//                  block is reduced (see `emit_m_block`)
//   m_block      - M rows per full block
//   k_block      - K elements reduced per K block
//   acc_elems    - outputs one accumulator holds: 1 for non-transposed A,
//                  where each accumulator is horizontally reduced to a single
//                  output, and the SIMD width for transposed A, where each
//                  accumulator holds a vector of neighboring outputs
//   dt_sz_a/x/y  - element size in bytes of A, x, and y
//   dt_a/x/y     - element data type of A, x, and y in memory
//   dt_a_reg / dt_x_reg - data type of the vec vreg holding A and x. It is the
//                  memory type when the reduction consumes it as it is, and
//                  f32 when the load converts. Paired with `k_block`, which is
//                  the element count that follows from the same choice.
//   dt_acc       - accumulation data type
//   beta         - output scaling: 0 overwrites y, 1 accumulates into y
//   m_blocks     - number of full M blocks
//   m_tail       - remaining M rows after the full blocks
//   k_blocks     - number of full K blocks
//   k_tail       - remaining K elements after the full blocks
//   acc_tail     - outputs held by the last, partially filled accumulator of
//                  the M tail. 0 when every accumulator is full
//   mblk_*_off   - byte offset to advance A/y pointers between M blocks
//   kblk_*_off   - byte offset to advance A/x pointers between K blocks
//   pf_*_off     - byte distance along K at which A/x are SW-prefetched ahead
//                  of their loads. 0 disables the prefetch
//   with_bias    - whether a bias is added to the output
//   treat_y_as_row - whether the output y is a row. When true the bias is
//                    indexed per M row, otherwise broadcast from bias[0]
//   dt_bias      - element data type of the bias
//   dt_sz_bias   - element size in bytes of the bias
//   mblk_bias_off - byte offset to advance the bias pointer between M blocks
//   with_injector_postops - whether attribute post-ops (eltwise/binary/sum) are
//                    requested
//   with_src_scales - whether a source scale multiplies the output. Always a
//                    single common scalar for this kernel
//   with_wei_scales - whether a weights scale multiplies the output
//   single_wei_scale - whether the weights scale is one scalar for the whole
//                    output. False only for per-N scales with a row output,
//                    where each output element has its own scale
//   dt_src_scales / dt_wei_scales - element data type of each scale
//   dt_sz_wei_scales - element size in bytes of the weights scale
//   mblk_wei_scale_off - byte offset to advance the weights-scale pointer
//                    between M blocks
struct brgemv_ir_conf_t {
    brgemv_ir_conf_t(const brgemm_desc_t &brg)
        : m(brg.bcast_dim)
        , k(brg.reduce_dim)
        , lda(brg.LDA)
        , incy(brg.LDC)
        , max_bs(brg.brgattr.max_bs)
        , trans_a(brg.transA)
        , m_block(brg.bd_block)
        , k_block(brg.rd_block)
        , acc_elems(trans_a ? m_block / brg.gemv_bd_block() : 1)
        , dt_sz_a(brg.typesize_A)
        , dt_sz_x(brg.typesize_B)
        , dt_sz_y(brg.typesize_C)
        , dt_a(brg.dt_a)
        , dt_x(brg.dt_b)
        , dt_y(brg.dt_c)
        , dt_a_reg(brg.gemv_use_vdpbf16ps() ? brg.dt_a : data_type::f32)
        , dt_x_reg(brg.gemv_use_vdpbf16ps() ? brg.dt_b : data_type::f32)
        , dt_acc(data_type::f32)
        , beta(brg.beta)
        , m_blocks(m / m_block)
        , m_tail(m % m_block)
        , k_blocks(k / k_block)
        , k_tail(k % k_block)
        , acc_tail((int)(m_tail % acc_elems))
        // A is K-major when it is transposed, so the next M block is `m_block`
        // contiguous elements away and the next K block is `k_block` rows
        // away. Otherwise it is the other way around.
        , mblk_a_off(trans_a ? dt_sz_a * (dim_t)m_block
                             : dt_sz_a * (dim_t)m_block * lda)
        , mblk_y_off(dt_sz_y * m_block * incy)
        , kblk_a_off(trans_a ? dt_sz_a * (dim_t)k_block * lda
                             : dt_sz_a * (dim_t)k_block)
        , kblk_x_off(dt_sz_x * k_block)
        , pf_a_off(brg.gemv_pf_rd_blocks() * kblk_a_off)
        , pf_x_off(brg.gemv_pf_rd_blocks() * kblk_x_off)
        , with_bias(brg.with_bias)
        , treat_y_as_row(brg.treat_y_as_row)
        , dt_bias(brg.dt_bias)
        , dt_sz_bias(brg.typesize_bias)
        , mblk_bias_off(dt_sz_bias * m_block)
        , with_injector_postops(
                  brg.with_eltwise || brg.with_binary || brg.with_sum)
        , with_src_scales(brg.with_src_scales)
        , with_wei_scales(brg.with_wei_scales)
        , single_wei_scale(brg.gemv_single_wei_scale())
        , dt_src_scales(brg.dt_src_scales)
        , dt_wei_scales(brg.dt_wei_scales)
        , dt_sz_wei_scales(with_wei_scales
                          ? (int)types::data_type_size(dt_wei_scales)
                          : 0)
        , mblk_wei_scale_off(dt_sz_wei_scales * m_block) {}

    const dim_t m, k, lda, incy;
    const dim_t max_bs;
    const bool trans_a;
    const int m_block, k_block, acc_elems;
    const int dt_sz_a, dt_sz_x, dt_sz_y;
    const data_type_t dt_a, dt_x, dt_y;
    const data_type_t dt_a_reg, dt_x_reg;
    const data_type_t dt_acc;
    const float beta;
    const dim_t m_blocks, m_tail, k_blocks, k_tail;
    const int acc_tail;
    const dim_t mblk_a_off, mblk_y_off;
    const dim_t kblk_a_off, kblk_x_off;
    const dim_t pf_a_off, pf_x_off;
    const bool with_bias, treat_y_as_row;
    const data_type_t dt_bias;
    const int dt_sz_bias;
    const dim_t mblk_bias_off;
    const bool with_injector_postops;
    const bool with_src_scales, with_wei_scales, single_wei_scale;
    const data_type_t dt_src_scales, dt_wei_scales;
    const int dt_sz_wei_scales;
    const dim_t mblk_wei_scale_off;

    bool has_post_ops() const {
        return with_bias || with_injector_postops || with_src_scales
                || with_wei_scales;
    }
};

// M-loop input register classification
//
// These structs describe registers used by the M-block loop, split by
// whether their values advance each iteration or remain constant.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside the M-block loop (acc, batch_ptr, ws, a_ptr,
//   x_ptr) are local temporaries and are not included here.

// Registers that advance each M-block iteration by a fixed byte offset.
struct advancing_regs_t {
    // Current `y` pointer, the vector `beta` accumulates into. Advances by
    // `mblk_y_off` per M-block.
    ir::vreg_t y_ptr = ir::vreg_t::none;
    // Pointer used to store the M-block result. Usually this is `y_ptr`, but
    // when post-ops run it holds `ptr_D`, which may be a separate output
    // buffer. The accumulation and output buffers share the same type and
    // stride (enforced by `brgemv_ir_supported()`), so `store_ptr` advances
    // together with `y_ptr`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A for the current M-block. Starts at 0 and advances by
    // `mblk_a_off` each iteration.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Bias base pointer. Advances by `mblk_bias_off` per M-block when
    // `treat_y_as_row` is set, otherwise stays at the first element. `none`
    // unless `with_bias`.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Weights-scale base pointer. Advances by `mblk_wei_scale_off` per M-block
    // for per-N scales, otherwise stays at the first element. `none` unless
    // `with_wei_scales`.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
};

// Registers that remain constant for the entire M-loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // K-tail mask, shared by every masked tail load. `none` when `k_tail` is
    // zero, which is when there is no tail load to mask.
    ir::vreg_t k_tail_mask = ir::vreg_t::none;
    // Mask of the partially filled accumulator, holding `acc_tail` active
    // elements. Shared by every masked access to its outputs and to its slice
    // of A. `none` when `acc_tail` is zero.
    ir::vreg_t acc_tail_mask = ir::vreg_t::none;
    // Post-ops flag (params.do_post_ops). Non-zero applies the post-ops, zero
    // stores the raw accumulator. `none` unless the kernel has a post-op to
    // apply (bias, scales, or injector post-ops).
    ir::vreg_t do_post_ops = ir::vreg_t::none;
    // Source-scale pointer. A single common scalar, so it never advances.
    // `none` unless `with_src_scales`.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
};

// Complete input register set for the M-loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads kernel argument pointers and the batch count, and initializes the
// running A offset to zero.
//
// Optional arguments are only initialized when the configuration specifies
// their presence. Otherwise, they remain `none`.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const brgemv_ir_conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.y_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.y_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    if (cfg.k_tail > 0) {
        // We only need to set the mask once per kernel. It's lifetime is
        // managed automatically by the allocator.
        regs.invariant.k_tail_mask = ir.new_mask();
        ir.set_mask_imm(regs.invariant.k_tail_mask, (int)cfg.k_tail);
    }

    if (cfg.acc_tail > 0) {
        regs.invariant.acc_tail_mask = ir.new_mask();
        ir.set_mask_imm(regs.invariant.acc_tail_mask, cfg.acc_tail);
    }

    if (cfg.with_bias) {
        regs.advancing.bias_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.bias_ptr, GET_OFF(ptr_bias));
    }

    if (cfg.with_src_scales) {
        regs.invariant.src_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.src_scale_ptr, GET_OFF(ptr_src_scales));
    }

    if (cfg.with_wei_scales) {
        regs.advancing.wei_scale_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.wei_scale_ptr, GET_OFF(ptr_wei_scales));
    }

    if (cfg.has_post_ops()) {
        regs.invariant.do_post_ops = ir.new_gpr();
        ir.load_param(regs.invariant.do_post_ops, GET_OFF(do_post_ops));

        regs.advancing.store_ptr = ir.new_gpr();
        ir.mov_reg(regs.advancing.store_ptr, regs.advancing.y_ptr);

        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
        ir.if_end(if_post_ops);
    } else {
        regs.advancing.store_ptr = regs.advancing.y_ptr;
    }

    return regs;
}

// An accumulator and the number of outputs it holds.
//
// Accumulator `acc_idx` of an M block covers the `n_elems` outputs starting
// at output `acc_idx * acc_elems` of the block. Only the last accumulator of
// the M tail can hold fewer than `acc_elems` outputs.
struct acc_t {
    ir::vreg_t reg = ir::vreg_t::none;
    int n_elems = 0;
};

// Byte offset of the first output of accumulator `acc_idx` from the start of
// the current M block, in a buffer of output-shaped elements of `dt_sz` bytes
// placed `inc` elements apart.
dim_t acc_off(
        const brgemv_ir_conf_t &cfg, int acc_idx, int dt_sz, dim_t inc = 1) {
    return dt_sz * (dim_t)acc_idx * cfg.acc_elems * inc;
}

// Loads the elements of an output-shaped operand (`y` or a per-output bias or
// scale) that match an accumulator holding `n_elems` outputs. A single output
// needs only element 0, and a partially filled accumulator is read under the
// accumulator-tail mask.
void emit_acc_load(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, ir::vreg_t dst, ir::vreg_t base,
        dim_t disp, int n_elems, data_type_t mem_dt) {
    if (n_elems == 1)
        ir.vload_scalar(dst, base, disp, mem_dt);
    else if (n_elems == cfg.acc_elems)
        ir.vload(dst, base, disp, mem_dt);
    else
        ir.vload_masked(dst, base, disp, regs.invariant.acc_tail_mask, mem_dt);
}

// Stores the `n_elems` outputs an accumulator holds. The counterpart of
// `emit_acc_load`.
void emit_acc_store(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, ir::vreg_t base, dim_t disp,
        ir::vreg_t src, int n_elems, data_type_t mem_dt) {
    if (n_elems == 1)
        ir.vstore_scalar(base, disp, src, mem_dt);
    else if (n_elems == cfg.acc_elems)
        ir.vstore(base, disp, src, mem_dt);
    else
        ir.vstore_masked(base, disp, src, regs.invariant.acc_tail_mask, mem_dt);
}

// Loads a value shared by every output into the first `n_elems` elements of
// `dst`. A single output needs only element 0, which a plain load fills.
void emit_bcast_load(ir::ir_t &ir, ir::vreg_t dst, ir::vreg_t base, dim_t disp,
        int n_elems, data_type_t mem_dt) {
    if (n_elems == 1)
        ir.vload_scalar(dst, base, disp, mem_dt);
    else
        ir.vload_bcast(dst, base, disp, mem_dt);
}

// Creates the accumulators of an M block covering `rows` outputs and seeds
// them as `beta` requires.
std::vector<acc_t> init_accumulators(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int rows) {
    std::vector<acc_t> acc(utils::div_up(rows, cfg.acc_elems));
    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++) {
        acc[acc_idx].reg = ir.new_vec(cfg.dt_acc);
        acc[acc_idx].n_elems
                = nstl::min(cfg.acc_elems, rows - acc_idx * cfg.acc_elems);

        // The current implementation supports only 0 and 1 for beta so for the
        // case where beta = 1 we load `y` into the accumulator registers and
        // then the reduction adds the results of the multiplication to them.
        //
        // This reads `y_ptr` while a sum post-op reads `store_ptr`, so the two
        // never count the same value twice.
        if (cfg.beta == 0.0f)
            ir.vzero(acc[acc_idx].reg);
        else
            emit_acc_load(ir, cfg, regs, acc[acc_idx].reg, regs.advancing.y_ptr,
                    acc_off(cfg, acc_idx, cfg.dt_sz_y, cfg.incy),
                    acc[acc_idx].n_elems, cfg.dt_y);
    }
    return acc;
}

// Batch loop of an M block.
//
// For each batch element, loads its A and x pointers, moves A to the current
// M block and calls `body(a_ptr, x_ptr)` to reduce the element into the
// accumulators.
template <typename body_t>
void emit_batch_loop(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, body_t body) {
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    auto bs_body = [&]() {
        const ir::vreg_t a_ptr = ir.new_gpr();
        const ir::vreg_t x_ptr = ir.new_gpr();

        ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
        ir.add_reg(a_ptr, regs.advancing.a_off);
        ir.load(x_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
        ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));

        body(a_ptr, x_ptr);
    };

    if (cfg.max_bs > 1)
        ir::emit_loop_reg(ir, regs.invariant.bs, bs_body);
    else
        ir::emit_loop_imm(ir, 1, bs_body);
}

// Post-ops of an M block.
//
// Applies scales, bias and injector post-ops in place to the reduced
// accumulators.
void emit_post_ops(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const std::vector<acc_t> &acc) {
    if (!cfg.has_post_ops()) return;

    const int n_acc = (int)acc.size();
    // No accumulator holds more outputs than the first, so a shared value
    // broadcast to that many elements covers every accumulator.
    const int bcast_elems = acc[0].n_elems;

    // The implemented order conforms to oneDNN's defined semantics for
    // applying scales, bias and post-ops:
    //   src scale -> weights scale -> bias -> post-ops.
    // Scales apply to the accumulated result before the bias add and
    // post-ops apply after bias. The two scales can swap because both are
    // multiplies but a scale must not move past the bias and the bias must
    // not move past the post-ops.
    const int if_post_ops
            = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);

    if (cfg.with_src_scales) {
        // Loaded once and applied to every output.
        const ir::vreg_t sc = ir.new_vec(cfg.dt_src_scales);
        emit_bcast_load(ir, sc, regs.invariant.src_scale_ptr, 0, bcast_elems,
                cfg.dt_src_scales);

        for (int acc_idx = 0; acc_idx < n_acc; acc_idx++)
            ir.vmul(acc[acc_idx].reg, sc);
    }

    if (cfg.with_wei_scales) {
        // The single scale is loop invariant, so load it once above the
        // loop. The per-N case loads a separate scale per output element.
        const ir::vreg_t sc = ir.new_vec(cfg.dt_wei_scales);
        if (cfg.single_wei_scale)
            emit_bcast_load(ir, sc, regs.advancing.wei_scale_ptr, 0,
                    bcast_elems, cfg.dt_wei_scales);

        for (int acc_idx = 0; acc_idx < n_acc; acc_idx++) {
            if (!cfg.single_wei_scale)
                emit_acc_load(ir, cfg, regs, sc, regs.advancing.wei_scale_ptr,
                        acc_off(cfg, acc_idx, cfg.dt_sz_wei_scales),
                        acc[acc_idx].n_elems, cfg.dt_wei_scales);
            ir.vmul(acc[acc_idx].reg, sc);
        }
    }

    if (cfg.with_bias) {
        // The broadcast bias is loop invariant, so load it once outside the
        // loop. A row output loads a separate bias per output element.
        const ir::vreg_t bias = ir.new_vec(cfg.dt_bias);
        if (!cfg.treat_y_as_row)
            emit_bcast_load(ir, bias, regs.advancing.bias_ptr, 0, bcast_elems,
                    cfg.dt_bias);

        for (int acc_idx = 0; acc_idx < n_acc; acc_idx++) {
            if (cfg.treat_y_as_row)
                emit_acc_load(ir, cfg, regs, bias, regs.advancing.bias_ptr,
                        acc_off(cfg, acc_idx, cfg.dt_sz_bias),
                        acc[acc_idx].n_elems, cfg.dt_bias);
            ir.vadd(acc[acc_idx].reg, bias);
        }
    }

    if (cfg.with_injector_postops) {
        // One `inject_postops` operation takes a single valid-element count,
        // so each run of accumulators that hold the same number of outputs
        // goes into its own operation. The output offsets match the store
        // displacements in `emit_m_block`, so a sum post-op reads each element
        // from the address its result is written to.
        for (int beg = 0; beg < n_acc;) {
            std::vector<ir::vreg_t> run_acc;
            std::vector<dim_t> run_off;
            int end = beg;
            for (; end < n_acc && acc[end].n_elems == acc[beg].n_elems; end++) {
                run_acc.push_back(acc[end].reg);
                run_off.push_back(acc_off(cfg, end, cfg.dt_sz_y, cfg.incy));
            }
            ir.inject_postops(run_acc, regs.advancing.store_ptr, run_off,
                    acc[beg].n_elems);
            beg = end;
        }
    }

    ir.if_end(if_post_ops);
}

} // namespace

namespace nontrans {

// Innermost reduction step.
//
// Loads one k_block-wide chunk of `x`, then performs a multiply-add into each
// accumulator in `acc` using the corresponding chunk of its A row.
//
// Each accumulator holds a partial dot product for one output. The number of
// accumulators is the number of rows in the current M block or M tail.
void emit_microkernel(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<acc_t> &acc, ir::vreg_t a_ptr, ir::vreg_t x_ptr) {
    // GEMV is bandwidth-bound, so SW-prefetch ahead of each load: the x vector
    // once and every A row.
    const ir::vreg_t x = ir.new_vec(cfg.dt_x_reg);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a_reg);
    if (cfg.pf_x_off > 0) ir.prefetch(x_ptr, cfg.pf_x_off);
    ir.vload(x, x_ptr, 0, cfg.dt_x);
    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++) {
        const dim_t a_off = cfg.dt_sz_a * (dim_t)acc_idx * cfg.lda;
        if (cfg.pf_a_off > 0) ir.prefetch(a_ptr, a_off + cfg.pf_a_off);
        ir.vload(a, a_ptr, a_off, cfg.dt_a);
        ir.vdot(acc[acc_idx].reg, a, x);
    }
}

// Innermost reduction step for K tail.
//
// Same shape as emit_microkernel, but uses masked loads.
void emit_microkernel_tail(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<acc_t> &acc, ir::vreg_t a_ptr, ir::vreg_t x_ptr,
        ir::vreg_t mask) {
    const ir::vreg_t x = ir.new_vec(cfg.dt_x_reg);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a_reg);
    ir.vload_masked(x, x_ptr, 0, mask, cfg.dt_x);
    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++) {
        ir.vload_masked(a, a_ptr, cfg.dt_sz_a * (dim_t)acc_idx * cfg.lda, mask,
                cfg.dt_a);
        ir.vdot(acc[acc_idx].reg, a, x);
    }
}

// Reduction of one M block with a non-transposed A.
//
// Each accumulator holds the dot product of one A row with x. The rows are
// read in k_block-wide chunks across the batch, and each accumulator is then
// horizontally reduced to the single output it holds.
void emit_reduce(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const std::vector<acc_t> &acc) {
    emit_batch_loop(ir, cfg, regs, [&](ir::vreg_t a_ptr, ir::vreg_t x_ptr) {
        // Advance the A and x pointers by one K block.
        auto advance_ptrs = [&]() {
            ir.add_imm(a_ptr, cfg.kblk_a_off);
            ir.add_imm(x_ptr, cfg.kblk_x_off);
        };

        // Reduce the full K blocks, then the tail if any. Cases by `k_blocks`.
        // k > 0 guarantees k_blocks and k_tail are never both zero.
        //   *  == 0  whole reduction is the tail
        //   *  == 1  one block, advance by hand only if a tail follows
        //   *  >= 2  loop, advancing per iteration
        if (cfg.k_blocks >= 2) {
            ir::emit_loop_imm(ir, cfg.k_blocks, [&]() {
                emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
            }, advance_ptrs);
        } else if (cfg.k_blocks == 1) {
            emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
            if (cfg.k_tail > 0) advance_ptrs();
        }
        if (cfg.k_tail > 0)
            emit_microkernel_tail(
                    ir, cfg, acc, a_ptr, x_ptr, regs.invariant.k_tail_mask);
    });

    // Horizontal reduction
    const ir::vreg_t ws = ir.new_vec(cfg.dt_acc);
    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++)
        ir.vhreduce(acc[acc_idx].reg, ws);
}

} // namespace nontrans

namespace trans {

// Innermost reduction step.
//
// Broadcasts one element of `x`, then performs a multiply-add into every
// accumulator using the matching slice of the current A row.
//
// A is K-major here, so a single K step reads one contiguous A row that spans
// the whole M block. Each accumulator takes the `n_elems` elements of the row
// that match its outputs.
void emit_microkernel(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const std::vector<acc_t> &acc,
        ir::vreg_t a_ptr, ir::vreg_t x_ptr) {
    const ir::vreg_t x = ir.new_vec(cfg.dt_x_reg);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a_reg);
    ir.vload_bcast(x, x_ptr, 0, cfg.dt_x);
    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++) {
        const dim_t a_off = acc_off(cfg, acc_idx, cfg.dt_sz_a);
        if (cfg.pf_a_off > 0) ir.prefetch(a_ptr, a_off + cfg.pf_a_off);
        if (acc[acc_idx].n_elems == cfg.acc_elems)
            ir.vload(a, a_ptr, a_off, cfg.dt_a);
        else
            ir.vload_masked(
                    a, a_ptr, a_off, regs.invariant.acc_tail_mask, cfg.dt_a);
        ir.vdot(acc[acc_idx].reg, a, x);
    }
}

// Reduction of one M block with a transposed A.
//
// Each accumulator holds `acc_elems` neighboring outputs. K is reduced one A
// row at a time across the batch, multiplying the row by a broadcast element
// of x. The outputs already sit in the elements they are stored from, so
// nothing is reduced horizontally.
void emit_reduce(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const std::vector<acc_t> &acc) {
    // The transA blocking reduces one A row per K step (see
    // `brgemm_blocking_vmm_gemv`), so there is no K tail.
    assert(cfg.k_block == 1 && cfg.k_tail == 0);

    emit_batch_loop(ir, cfg, regs, [&](ir::vreg_t a_ptr, ir::vreg_t x_ptr) {
        ir::emit_loop_imm(ir, cfg.k_blocks, [&]() {
            emit_microkernel(ir, cfg, regs, acc, a_ptr, x_ptr);
        }, [&]() {
            ir.add_imm(a_ptr, cfg.kblk_a_off);
            ir.add_imm(x_ptr, cfg.kblk_x_off);
        });
    });
}

} // namespace trans

namespace {

// One M block covering `rows` outputs.
//
// Reduces the block into its accumulators, applies the post-ops and stores the
// result. Finally, advances the M-block pointers to the next block.
void emit_m_block(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int rows) {
    const std::vector<acc_t> acc = init_accumulators(ir, cfg, regs, rows);

    // The layout of A only decides how the block is reduced. Everything else
    // is shared and sees the layout only as how many outputs each accumulator
    // holds.
    if (cfg.trans_a)
        trans::emit_reduce(ir, cfg, regs, acc);
    else
        nontrans::emit_reduce(ir, cfg, regs, acc);

    emit_post_ops(ir, cfg, regs, acc);

    for (int acc_idx = 0; acc_idx < (int)acc.size(); acc_idx++)
        emit_acc_store(ir, cfg, regs, regs.advancing.store_ptr,
                acc_off(cfg, acc_idx, cfg.dt_sz_y, cfg.incy), acc[acc_idx].reg,
                acc[acc_idx].n_elems, cfg.dt_y);

    // Advance to next M block
    ir.add_imm(regs.advancing.a_off, cfg.mblk_a_off);
    ir.add_imm(regs.advancing.y_ptr, cfg.mblk_y_off);
    // When `!has_post_ops()`, `store_ptr` is the same IR vreg as `y_ptr` (see
    // `init_m_loop_input_regs`), so advancing `y_ptr` above covers both.
    if (cfg.has_post_ops())
        ir.add_imm(regs.advancing.store_ptr, cfg.mblk_y_off);

    if (cfg.with_bias && cfg.treat_y_as_row)
        ir.add_imm(regs.advancing.bias_ptr, cfg.mblk_bias_off);
    if (cfg.with_wei_scales && !cfg.single_wei_scale)
        ir.add_imm(regs.advancing.wei_scale_ptr, cfg.mblk_wei_scale_off);
}

// Builds IR for GEMV.
//
// Computes:
//   y[i] = sum_k A[i][k] * x[k]    (non-transposed A)
//   y[i] = sum_k A[k][i] * x[k]    (transposed A)
//   (m = brg.bcast_dim, k = brg.reduce_dim, n = 1)
//
// The output is partitioned into full M blocks of `m_block` rows each, with a
// final partial block of `m_tail` rows when m is not a multiple of m_block.
//
// Each M block:
// - Maintains one accumulator per `acc_elems` neighboring outputs
// - Accumulates over k across the batch:
//   - non-transposed A: in k_block-wide chunks of each A row, then
//     horizontally reduces each accumulator to its single output
//   - transposed A: one A row at a time, multiplied by a broadcast element of
//     x
// - Applies the post-ops and stores the accumulators to memory
void build_gemv(const brgemm_desc_t &brg, ir::ir_t &ir) {
    const brgemv_ir_conf_t cfg(brg);
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    if (cfg.m_blocks > 0)
        ir::emit_loop_imm(ir, cfg.m_blocks,
                [&]() { emit_m_block(ir, cfg, regs, cfg.m_block); });

    if (cfg.m_tail > 0) emit_m_block(ir, cfg, regs, (int)cfg.m_tail);
}

// IR-based BRGEMV kernel.
struct jit_brgemv_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemv_ir_kernel_t)

    jit_brgemv_ir_kernel_t(const brgemm_desc_t &abrg)
        : brgemm_kernel_t(jit_name(), abrg.isa_impl), brg_(abrg) {}

    void generate() override {
        ir::ir_t ir;
        build_gemv(brg_, ir);

        ir::postops_config_t postops;
        if (brg_.with_eltwise || brg_.with_binary || brg_.with_sum) {
            postops.post_ops = &brg_.attr()->post_ops_;
            postops.dst_md = brg_.dst_md();
            // The transA accumulators run along M, the channel dimension of 3D
            // dst {B, M, 1}, so per_oc_spatial and per_mb_spatial don't apply.
            if (brg_.transA && !brg_.treat_y_as_row
                    && brg_.dst_md()->ndims == 3)
                postops.bcast_set = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_mb,
                        broadcasting_strategy_t::no_broadcast};
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
        }

        ir::generate_kernel(*this, ir, postops);
    }

private:
    brgemm_desc_t brg_;
};

} // namespace

// Returns `status::success` if the descriptor is supported by the GEMV IR
// kernel, otherwise `status::unimplemented`.
status_t brgemv_ir_supported(const brgemm_desc_t &brg) {
    using namespace data_type;

    // Accepted input data type and the ISA it is built at:
    //   f32  - avx2
    //   bf16 - avx512_core_bf16
    //   f16  - avx512_core_fp16
    // GEMV blocking permits no other ISA (see `brgemm_blocking_vmm_gemv`).
    const bool dt_isa_ok = (brg.dt_a == f32 && brg.isa_impl == avx2)
            || (brg.dt_a == bf16 && brg.isa_impl == avx512_core_bf16)
            || (brg.dt_a == f16 && brg.isa_impl == avx512_core_fp16);
    VCONDCHECK_BRGEMV_IR(dt_isa_ok, VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMV_IR(brg.dt_b == brg.dt_a, VERBOSE_UNSUPPORTED_DT);
    // The kernel accumulates in f32 and has no conversion on the way out.
    VCONDCHECK_BRGEMV_IR(brg.dt_c == f32, VERBOSE_UNSUPPORTED_DT);

    // With post-ops the kernel stores to D instead of C. It cannot convert the
    // accumulator on the way out, so it takes only `dt_d == dt_c`. The check is
    // unconditional because `dt_c != dt_d` alone already routes the descriptor
    // through the store-to-D path (see `are_post_ops_applicable()`), which the
    // `has_post_ops` below does not cover.
    VCONDCHECK_BRGEMV_IR(brg.dt_d == brg.dt_c, VERBOSE_UNSUPPORTED_DT);

    const bool has_post_ops = brg.with_bias || brg.with_eltwise
            || brg.with_binary || brg.with_sum || brg.with_src_scales
            || brg.with_wei_scales;
    VCONDCHECK_BRGEMV_IR(IMPLICATION(has_post_ops, brg.LDC == brg.LDD),
            VERBOSE_UNSUPPORTED_FEATURE, "LDC != LDD");

    // Post-ops go through the JIT injector. Accept eltwise, binary and sum, and
    // only binary arguments the injector can handle on this ISA. Other kinds
    // fall back.
    if (brg.attr()) {
        const memory_desc_wrapper dst_d(brg.dst_md());
        const std::vector<injector::post_op_type> accepted
                = {injector::eltwise, injector::binary, injector::sum};
        VCONDCHECK_BRGEMV_IR(
                injector::post_ops_ok({brg.isa_impl, accepted,
                        brg.attr()->post_ops_, &dst_d,
                        /*sum_at_pos_0_only=*/false,
                        /*sum_requires_scale_one=*/false,
                        /*sum_requires_zp_zero=*/false,
                        /*sum_requires_same_params=*/false,
                        binary_injector::
                                get_all_strategies_supported_by_injector()}),
                VERBOSE_UNSUPPORTED_POSTOP);
    }
    VCONDCHECK_BRGEMV_IR(!brg.with_dst_scales, VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(!brg.is_per_k_src_scales && !brg.is_per_k_wei_scales,
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMV_IR(!brg.req_s8s8_compensation,
            VERBOSE_UNSUPPORTED_FEATURE, "s8s8 compensation");
    VCONDCHECK_BRGEMV_IR(utils::everyone_is(brgemm_broadcast_t::none,
                                 brg.zp_type_a, brg.zp_type_b, brg.zp_type_c),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VCONDCHECK_BRGEMV_IR(
            !brg.with_bias || brg.dt_bias == f32, VERBOSE_UNSUPPORTED_BIAS_CFG);
    VCONDCHECK_BRGEMV_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMV_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMV_IR(
            !brg.is_runtime_lda, VERBOSE_UNSUPPORTED_FEATURE, "runtime lda");
    VCONDCHECK_BRGEMV_IR(
            !brg.is_runtime_ldc, VERBOSE_UNSUPPORTED_FEATURE, "runtime ldc");

    const brgemv_ir_conf_t cfg(brg);

    // With a transposed A each accumulator holds a vector of neighboring
    // outputs and stores them with a single instruction, so `y` has to be
    // contiguous.
    VCONDCHECK_BRGEMV_IR(IMPLICATION(cfg.trans_a, cfg.incy == 1),
            VERBOSE_UNSUPPORTED_FEATURE, "strided y with transposed A");

    // Ensure indexed displacements fit in 32-bit
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };

    // Every A displacement inside an M block, prefetch included, is below the
    // block offset plus the prefetch distance.
    VCONDCHECK_BRGEMV_IR(fits(cfg.mblk_a_off + cfg.pf_a_off),
            VERBOSE_UNSUPPORTED_FEATURE, "A block offset overflows int32");
    VCONDCHECK_BRGEMV_IR(fits(cfg.kblk_a_off), VERBOSE_UNSUPPORTED_FEATURE,
            "A reduction step overflows int32");
    VCONDCHECK_BRGEMV_IR(fits(cfg.mblk_y_off), VERBOSE_UNSUPPORTED_FEATURE,
            "y block offset overflows int32");

    return status::success;
}

brgemm_kernel_t *create_brgemv_ir_kernel(const brgemm_desc_t &brg) {
    if (brgemv_ir_supported(brg) != status::success) return nullptr;
    return new jit_brgemv_ir_kernel_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
