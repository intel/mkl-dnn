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

// The IR-based BRGEMM kernel converts a brgemm descriptor into IR, then runs
// the register allocator and code emitter.
//
// This kernel uses the classic kernels as a reference: `jit_brgemm_kernel_t`
// for the looped schedule and `jit_brgemm_amx_uker_t` for the unrolled one. It
// is comparable with them instruction for instruction when it comes to compute
// structure.
//
// `init_conf()` is the only code that reads the descriptor for the builder. It
// converts the descriptor into `conf_t`, and the builder reads only `conf_t`.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/brgemm/brgemm_ir_compute_path.hpp"
#include "cpu/x64/brgemm/brgemm_ir_conf.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

#define VCONDCHECK_BRGEMM_IR(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemm_ir, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

namespace {

// M-loop input register classification
//
// The same split the GEMV IR kernel uses (see `brgemv_ir.cpp`), by whether a
// value advances across M-loop iterations. The one difference is that this
// loop nest has two levels over the output, M and N, so the level a register
// advances at is stated per field instead of shared by the struct.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside a loop body (the accumulators, the B vectors, the
//   A broadcast, and the batch, A, and B pointers) are local temporaries and
//   are not included here.

// Registers that advance by a fixed byte offset as the nest walks the output.
// The unrolled schedule does not advance them.
struct advancing_regs_t {
    // Current C pointer. The N loop advances it by one block per iteration,
    // then the M loop rewinds it and steps to the next M block with one
    // constant. The classic kernel keeps a second base pointer for this
    // instead. One running pointer saves the general-purpose register.
    ir::vreg_t c_ptr = ir::vreg_t::none;
    // Pointer the finished block is stored through. It holds the `ptr_D`
    // kernel argument on a post-ops call and `ptr_C` otherwise, and can be the
    // same register as `c_ptr` (see `init_m_loop_input_regs()`). C and D have
    // the same data type and leading dimension (`init_output_conf()`), so it
    // advances like `c_ptr`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A of the current M block. Starts at 0 and advances by
    // one block per iteration of the M loop. `none` in the unrolled schedule.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Byte offset into B of the current N block. Advances by one block per
    // iteration of the N loop and is set back before the next M block. `none`
    // in the unrolled schedule.
    ir::vreg_t b_off = ir::vreg_t::none;
    // Current bias pointer. The bias is indexed by the load dimension only, so
    // it advances with the N loop and is rewound before the next M block.
    // `none` without the `bias` stage.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Current weights-scale pointer. A per-N scale advances and rewinds like
    // `bias_ptr`, a common one stays at its only element.
    // `none` without a weights scale stage.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
};

// Registers that hold the same value for the entire M loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // A and B base pointers, the `ptr_A` and `ptr_B` kernel arguments, which
    // `brgemm_offs` adds the byte offsets of each batch element to. `none` for
    // `brgemm_addr`, where each element holds its own pointers.
    ir::vreg_t a_base = ir::vreg_t::none;
    ir::vreg_t b_base = ir::vreg_t::none;
    // The `do_post_ops` kernel argument. Non-zero applies the post-ops and
    // stores to D, zero stores to C. `none` when the kernel has no post-op.
    ir::vreg_t do_post_ops = ir::vreg_t::none;
    // Common source and destination scale pointers.
    // `none` without the `src_scale` and `dst_scale` stages respectively.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
    ir::vreg_t dst_scale_ptr = ir::vreg_t::none;
};

// Complete input register set for the M loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads the kernel argument pointers and the batch count, and zeroes the
// running A and B offsets.
m_loop_input_regs_t init_m_loop_input_regs(ir::ir_t &ir, const conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.c_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.c_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.loop.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    if (cfg.loop.batch_kind == brgemm_offs) {
        regs.invariant.a_base = ir.new_gpr();
        ir.load_param(regs.invariant.a_base, GET_OFF(ptr_A));
        regs.invariant.b_base = ir.new_gpr();
        ir.load_param(regs.invariant.b_base, GET_OFF(ptr_B));
    }

    if (cfg.loop.schedule == schedule_t::looped) {
        regs.advancing.a_off = ir.new_gpr();
        ir.mov_imm(regs.advancing.a_off, 0);

        regs.advancing.b_off = ir.new_gpr();
        ir.mov_imm(regs.advancing.b_off, 0);
    }

    if (cfg.output.has_stage(output_stage_t::bias)) {
        regs.advancing.bias_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.bias_ptr, GET_OFF(ptr_bias));
    }

    if (cfg.output.has_stage(output_stage_t::src_scale)) {
        regs.invariant.src_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.src_scale_ptr, GET_OFF(ptr_src_scales));
    }

    if (cfg.output.has_stage(output_stage_t::wei_scale)
            || cfg.output.has_stage(output_stage_t::wei_scale_per_n)) {
        regs.advancing.wei_scale_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.wei_scale_ptr, GET_OFF(ptr_wei_scales));
    }

    if (cfg.output.has_stage(output_stage_t::dst_scale)) {
        regs.invariant.dst_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.dst_scale_ptr, GET_OFF(ptr_dst_scales));
    }

    // The block goes to `ptr_D` when `do_post_ops` is set, and to `ptr_C`
    // otherwise. Resolve that once here so that the stores do not branch.
    //
    // A post-ops call with `beta == 1` reads C for the accumulators and writes
    // D, so the two pointers need separate registers. C is never read when
    // `beta == 0`, so one register is enough and it holds `ptr_D` on a post-ops
    // call.
    if (cfg.output.has_post_ops()) {
        regs.invariant.do_post_ops = ir.new_gpr();
        ir.load_param(regs.invariant.do_post_ops, GET_OFF(do_post_ops));

        if (cfg.output.beta == 0.0f) {
            regs.advancing.store_ptr = regs.advancing.c_ptr;
        } else {
            regs.advancing.store_ptr = ir.new_gpr();
            ir.mov_reg(regs.advancing.store_ptr, regs.advancing.c_ptr);
        }

        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
        ir.if_end(if_post_ops);
    } else {
        regs.advancing.store_ptr = regs.advancing.c_ptr;
    }

    return regs;
}

// Iteration lists
//
// Each level of the loop nest (M, N, and K) has a list of its blocks, stored
// as runs of blocks with the same size. The code of a block depends only on
// its size and its position, so the looped schedule emits one body per run,
// and the unrolled schedule one body per block. The batch has no list because
// it is a runtime loop over blocks of one size.

// Consecutive blocks of one loop level with the same size.
//
//   size  - elements each block covers along the level's dimension
//   count - number of blocks, 0 if the level has no such blocks
struct block_run_t {
    dim_t size;
    dim_t count;
};

// Runs of the M, N, and K levels, in the order `jit_brgemm_kernel_t` walks
// them: the full blocks, then each tail. N has two tails: the full registers
// left after the full blocks, then the masked register.
std::vector<block_run_t> m_runs(const loop_conf_t &loop) {
    return {{loop.bd_block, loop.bdb},
            {loop.bdb_tail, loop.bdb_tail > 0 ? 1 : 0}};
}

std::vector<block_run_t> n_runs(const loop_conf_t &loop) {
    return {{(dim_t)loop.ld_block2 * loop.ld_block, loop.ldb2},
            {(dim_t)loop.ldb2_tail * loop.ld_block, loop.ldb2_tail > 0 ? 1 : 0},
            {loop.ldb_tail, loop.ldb_tail > 0 ? 1 : 0}};
}

std::vector<block_run_t> k_runs(const loop_conf_t &loop) {
    return {{loop.rd_block, loop.rdb},
            {loop.rdb_tail, loop.rdb_tail > 0 ? 1 : 0}};
}

// Emits one level of the loop nest from its runs, for the looped schedule.
//
// A run of more than one block becomes a runtime loop whose step moves the
// pointers by one block. A run of one block is emitted once, without a step
// (see `emit_loop_imm()`). Its accesses add its position to their
// displacements, so the pointers do not move to it.
//
//   ptr_pos - where the pointers stand when the level starts, in elements
//             from the first block
//   body    - `body(size, pos)` emits a block of `size` elements at `pos`
//             elements from the pointers
//   step    - `step(n)` moves the pointers by `n` elements
//
// Returns where the pointers stand after the level.
dim_t emit_looped_level(ir::ir_t &ir, const std::vector<block_run_t> &runs,
        dim_t ptr_pos, const std::function<void(dim_t, dim_t)> &body,
        const std::function<void(dim_t)> &step) {
    dim_t start = 0;
    for (const block_run_t &run : runs) {
        const dim_t pos = start - ptr_pos;
        ir::emit_loop_imm(ir, run.count, [&]() { body(run.size, pos); },
                [&]() { step(run.size); });

        // A runtime loop runs its step after every iteration, including the
        // last one.
        if (run.count > 1) ptr_pos += run.count * run.size;
        start += run.count * run.size;
    }
    return ptr_pos;
}

// Returns an upper bound, in elements, on the distance between the pointers
// of a looped level and any element it accesses or any position the pointers
// move to. It walks the runs as `emit_looped_level()` does.
//
// The bound also covers where the level leaves the pointers, which N needs.
// If the first M block has no step, it leaves the N pointers at the end of
// the N level, and the next M block reaches its N blocks at negative
// positions from there.
dim_t max_looped_pos(const std::vector<block_run_t> &runs) {
    dim_t max_pos = 0, start = 0, ptr_pos = 0;
    for (const block_run_t &run : runs) {
        if (run.count == 0) continue;
        max_pos = std::max(max_pos, start - ptr_pos + run.size);
        if (run.count > 1) ptr_pos += run.count * run.size;
        start += run.count * run.size;
    }
    return std::max(max_pos, ptr_pos);
}

// Emits one level of the loop nest from its runs, for the unrolled schedule.
//
// Each block is emitted at its position, and the pointers never move.
//
//   body - `body(size, pos)` emits a block of `size` elements at `pos`
//          elements from the pointers
void emit_unrolled_level(const std::vector<block_run_t> &runs,
        const std::function<void(dim_t, dim_t)> &body) {

    dim_t pos = 0;
    for (const block_run_t &run : runs) {
        for (dim_t i = 0; i < run.count; i++) {
            body(run.size, pos);
            pos += run.size;
        }
    }
}

// Returns the number of elements a level covers. It bounds every position the
// unrolled schedule addresses on the level.
dim_t level_size(const std::vector<block_run_t> &runs) {
    dim_t size = 0;
    for (const block_run_t &run : runs) {
        size += run.count * run.size;
    }
    return size;
}

// Sets `a_ptr` and `b_ptr` to the A and B of the batch element at `batch_ptr`.
// Mirrors `set_A_B_matrices()` in `jit_brgemm_kernel.cpp`.
void emit_batch_element_ptrs(ir::ir_t &ir, const conf_t &cfg,
        const invariant_regs_t &regs, ir::vreg_t batch_ptr, ir::vreg_t a_ptr,
        ir::vreg_t b_ptr) {

    switch (cfg.loop.batch_kind) {
        case brgemm_addr:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
            break;
        case brgemm_offs:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.A));
            ir.add_reg(a_ptr, regs.a_base);
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.B));
            ir.add_reg(b_ptr, regs.b_base);
            break;
        default: assert(!"unsupported batch kind"); break;
    }
}

// One batch element.
//
// Derives the A and B pointers of the element, shifts them to the current M and
// N block, then runs the reduction loop over K.
void emit_bs_body(ir::ir_t &ir, const conf_t &cfg, const compute_path_t &cp,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t batch_ptr) {

    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t b_ptr = ir.new_gpr();
    emit_batch_element_ptrs(ir, cfg, regs.invariant, batch_ptr, a_ptr, b_ptr);

    // Shift to the current M and N block.
    ir.add_reg(a_ptr, regs.advancing.a_off);
    ir.add_reg(b_ptr, regs.advancing.b_off);

    // Advance to the next batch element. A single batch element has no next
    // one, so nothing is emitted then.
    if (cfg.loop.max_bs > 1) {
        ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));
        // The classic kernel prefetches the next element for the address kind
        // only.
        if (cfg.loop.batch_kind == brgemm_addr) ir.prefetch(batch_ptr, 0);
    }

    // Advance the A and B pointers by `rd` elements of K.
    auto step = [&](dim_t rd) {
        ir.add_imm(a_ptr, cfg.a_off(0, rd));
        ir.add_imm(b_ptr, cfg.b_off(rd, 0));
    };

    // K tail needs no mask. A reduction step reads a full B vector along N
    // and one element of A, so a short tail is just fewer steps.
    emit_looped_level(
            ir, k_runs(cfg.loop), 0, [&](dim_t rd_loop, dim_t rd_start) {
        cp.emit_microkernel(
                ir, cfg, blk, acc, a_ptr, b_ptr, rd_start, (int)rd_loop);
    }, step);
}

// One stage of a post-ops call, applied to the accumulators of the block.
void emit_post_ops_stage(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, output_stage_t stage) {

    switch (stage) {
        case output_stage_t::src_scale: {
            const ir::vreg_t sc = ir.new_vec(cfg.output.dt_src_scales);
            ir.vload_bcast(sc, regs.invariant.src_scale_ptr, 0,
                    cfg.output.dt_src_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
            break;
        }
        case output_stage_t::wei_scale: {
            const ir::vreg_t sc = ir.new_vec(cfg.output.dt_wei_scales);
            ir.vload_bcast(sc, regs.advancing.wei_scale_ptr, 0,
                    cfg.output.dt_wei_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
            break;
        }
        case output_stage_t::wei_scale_per_n: {
            // One scale vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t sc = ir.new_vec(cfg.output.dt_wei_scales);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(cfg.ld_pos(blk, ld)),
                            cfg.output.dt_wei_scales);
                else
                    ir.vload_masked(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(cfg.ld_pos(blk, ld)),
                            blk.ld_tail_mask, cfg.output.dt_wei_scales);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vmul(acc[bd * blk.ld_block2 + ld], sc);
            }
            break;
        }
        case output_stage_t::bias: {
            // One bias vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t bias = ir.new_vec(cfg.output.dt_bias);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(cfg.ld_pos(blk, ld)),
                            cfg.output.dt_bias);
                else
                    ir.vload_masked(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(cfg.ld_pos(blk, ld)), blk.ld_tail_mask,
                            cfg.output.dt_bias);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vadd(acc[bd * blk.ld_block2 + ld], bias);
            }
            break;
        }
        case output_stage_t::injector_postops: {
            // The injector gets the address of the block and the offset of
            // each accumulator in the block, as `jit_brgemm_kernel_t` passes
            // `reg_aux_D` and `D_offset()`. A binary post-op computes the
            // output coordinates from the address at run time and from the
            // offset at JIT time (`calculate_*_partial()`), and gets them
            // wrong for an offset outside the block. So a block that is not
            // at the pointers gets its address in its own register. A sum
            // post-op reads the previous value from the same address.
            ir::vreg_t blk_ptr = regs.advancing.store_ptr;
            if (blk.bd_start != 0 || blk.ld_start != 0) {
                blk_ptr = ir.new_gpr();
                ir.mov_reg(blk_ptr, regs.advancing.store_ptr);
                ir.add_imm(blk_ptr, cfg.c_off(blk.bd_start, blk.ld_start));
            }

            std::vector<dim_t> out_byte_off(acc.size());
            for (int bd = 0; bd < blk.bd_block; bd++)
                for (int ld = 0; ld < blk.ld_block2; ld++)
                    out_byte_off[bd * blk.ld_block2 + ld]
                            = cfg.c_off(bd, (dim_t)ld * cfg.loop.ld_block);

            // A block is either all full registers or the one masked
            // register of the N tail, which holds `ldb_tail` elements.
            const int n_elems = blk.ld_tail_mask != ir::vreg_t::none
                    ? cfg.loop.ldb_tail
                    : cfg.loop.ld_block;
            ir.inject_postops(acc, blk_ptr, out_byte_off, n_elems);
            break;
        }
        case output_stage_t::dst_scale: {
            // The caller passes the inverted f32 destination scale, so it is a
            // multiply.
            const ir::vreg_t sc = ir.new_vec(data_type::f32);
            ir.vload_bcast(sc, regs.invariant.dst_scale_ptr, 0, data_type::f32);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
            break;
        }
    }
}

// One N block.
//
// Creates the accumulators of the block with the compute path, reduces them
// over the batch, and stores them to C, or to D on a post-ops call.
void emit_n_block(ir::ir_t &ir, const conf_t &cfg, const compute_path_t &cp,
        const m_loop_input_regs_t &regs, const out_block_t &blk) {

    const std::vector<ir::vreg_t> acc = cp.init_accumulators(ir, cfg, blk);

    // Batch reduction over the bs dimension.
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    auto bs_body
            = [&]() { emit_bs_body(ir, cfg, cp, regs, blk, acc, batch_ptr); };

    if (cfg.loop.max_bs > 1)
        ir::emit_loop_reg(ir, regs.invariant.bs, bs_body);
    else
        ir::emit_loop_imm(ir, 1, bs_body);

    // Only the vector compute path exists so far. Its accumulators are the rows
    // of the output, one register per (M row, N register), so the output path
    // below works on them directly.
    //
    // The kernel takes only 0 and 1 for beta. C is added after the reduction,
    // as in the classic kernel. Loading it into the accumulators up front
    // would make every FMA chain wait for those loads.
    if (cfg.output.beta != 0.0f) {
        for (int bd = 0; bd < blk.bd_block; bd++) {
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t c = ir.new_vec(cfg.output.dt_c);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(c, regs.advancing.c_ptr,
                            cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)),
                            cfg.output.dt_c);
                else
                    ir.vload_masked(c, regs.advancing.c_ptr,
                            cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)),
                            blk.ld_tail_mask, cfg.output.dt_c);
                ir.vadd(acc[bd * blk.ld_block2 + ld], c);
            }
        }
    }

    if (cfg.output.has_post_ops()) {
        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        for (const output_stage_t stage : cfg.output.post_ops_stages)
            emit_post_ops_stage(ir, cfg, regs, blk, acc, stage);
        ir.if_end(if_post_ops);
    }

    // `store_ptr` already points at C or D, so the stores do not branch. Their
    // memory data type is fixed when the IR is built, which is why C and D must
    // share one.
    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const ir::vreg_t src = acc[bd * blk.ld_block2 + ld];

            if (blk.ld_tail_mask == ir::vreg_t::none) {
                ir.vstore(regs.advancing.store_ptr,
                        cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)), src,
                        cfg.output.dt_c);
            } else {
                ir.vstore_masked(regs.advancing.store_ptr,
                        cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)), src,
                        blk.ld_tail_mask, cfg.output.dt_c);
            }
        }
    }
}

// Emits the loop nest of the looped schedule, which walks N inside M, as
// `jit_brgemm_kernel_t` does.
void emit_looped_nest(ir::ir_t &ir, const conf_t &cfg, const compute_path_t &cp,
        const m_loop_input_regs_t &regs) {

    // Every partial N block is the same width, so one mask serves the whole
    // kernel.
    ir::vreg_t ld_tail_mask = ir::vreg_t::none;

    if (cfg.loop.ldb_tail > 0) {
        ld_tail_mask = ir.new_mask();
        ir.set_mask_imm(ld_tail_mask, cfg.loop.ldb_tail);
    }

    // Where the N pointers stand, and where they stood at the start of the
    // current M block, in elements of N.
    dim_t n_ptr_pos = 0, n_start = 0;

    // Advance the N pointers by `ld` elements. `store_ptr` is a separate
    // register only when `beta == 1` and the kernel has a post-op, so it needs
    // its own advance only then.
    auto n_step = [&](dim_t ld) {
        ir.add_imm(regs.advancing.b_off, cfg.b_off(0, ld));
        ir.add_imm(regs.advancing.c_ptr, cfg.c_off(0, ld));
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr, cfg.c_off(0, ld));
        if (cfg.output.has_stage(output_stage_t::bias))
            ir.add_imm(regs.advancing.bias_ptr, cfg.bias_off(ld));
        if (cfg.output.has_stage(output_stage_t::wei_scale_per_n))
            ir.add_imm(regs.advancing.wei_scale_ptr, cfg.wei_scale_off(ld));
    };

    // Advance the M pointers by `bd` rows and move the N pointers back to
    // where they stood at the start of the M block. C moves along M and N
    // with one `add`. The bias and a per-N weights scale do not depend on M,
    // so they only move back. The B offset depends only on N, so it is set
    // instead of moved.
    auto m_step = [&](dim_t bd) {
        const dim_t ld = n_start - n_ptr_pos;
        ir.add_imm(regs.advancing.a_off, cfg.a_off(bd, 0));
        ir.add_imm(regs.advancing.c_ptr, cfg.c_off(bd, ld));
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr, cfg.c_off(bd, ld));
        if (ld != 0) {
            if (cfg.output.has_stage(output_stage_t::bias))
                ir.add_imm(regs.advancing.bias_ptr, cfg.bias_off(ld));
            if (cfg.output.has_stage(output_stage_t::wei_scale_per_n))
                ir.add_imm(regs.advancing.wei_scale_ptr, cfg.wei_scale_off(ld));
            ir.mov_imm(regs.advancing.b_off, cfg.b_off(0, n_start));
        }
        n_ptr_pos = n_start;
    };

    emit_looped_level(
            ir, m_runs(cfg.loop), 0, [&](dim_t bd_block, dim_t bd_start) {
        n_start = n_ptr_pos;
        n_ptr_pos = emit_looped_level(ir, n_runs(cfg.loop), n_start,
                [&](dim_t ld_elems, dim_t ld_start) {
            // An N block is whole registers, except the N tail of `ldb_tail`
            // elements, which is one masked register.
            const bool is_ld_tail = ld_elems % cfg.loop.ld_block != 0;
            const out_block_t blk {(int)bd_block,
                    (int)utils::div_up(ld_elems, cfg.loop.ld_block), bd_start,
                    ld_start, is_ld_tail ? ld_tail_mask : ir::vreg_t::none};
            emit_n_block(ir, cfg, cp, regs, blk);
        }, n_step);
    }, m_step);
}

// Emits the loop nest of the unrolled schedule, as `jit_brgemm_amx_uker_t`
// does.
//
// Every M, N, and K block is emitted at its position, so no pointer moves. N is
// inside M. The kernel has one batch element, and its A and B pointers are set
// once for the whole kernel.
//
// There are no output stages yet. The accumulators of a block start at zero,
// or at C when beta is 1, and are stored to C unchanged. The uker also loads C
// into the accumulators when beta is 1 (`may_load_accumulators_`).
void emit_unrolled_nest(ir::ir_t &ir, const conf_t &cfg,
        const compute_path_t &cp, const m_loop_input_regs_t &regs) {

    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t b_ptr = ir.new_gpr();
    emit_batch_element_ptrs(
            ir, cfg, regs.invariant, regs.invariant.batch, a_ptr, b_ptr);

    const std::vector<block_run_t> k = k_runs(cfg.loop);

    auto emit_block = [&](dim_t bd_block, dim_t bd_start, dim_t ld_elems,
                              dim_t ld_start) {
        // No N tail reaches this schedule, so a block is whole registers.
        const int ld_block2 = (int)(ld_elems / cfg.loop.ld_block);
        const out_block_t blk {
                (int)bd_block, ld_block2, bd_start, ld_start, ir::vreg_t::none};

        const std::vector<ir::vreg_t> acc = cfg.output.beta != 0.0f
                ? cp.load_accumulators(ir, cfg, blk, regs.advancing.c_ptr)
                : cp.init_accumulators(ir, cfg, blk);

        emit_unrolled_level(k, [&](dim_t rd_block, dim_t rd_start) {
            cp.emit_microkernel(
                    ir, cfg, blk, acc, a_ptr, b_ptr, rd_start, (int)rd_block);
        });

        cp.store_accumulators(ir, cfg, blk, acc, regs.advancing.store_ptr);
    };

    const std::vector<block_run_t> m = m_runs(cfg.loop);
    const std::vector<block_run_t> n = n_runs(cfg.loop);

    emit_unrolled_level(m, [&](dim_t bd_block, dim_t bd_start) {
        emit_unrolled_level(n, [&](dim_t ld_elems, dim_t ld_start) {
            emit_block(bd_block, bd_start, ld_elems, ld_start);
        });
    });
}

// Builds IR for BRGEMM.
//
// Computes:
//   C[i][j] = beta * C[i][j] + sum_bs sum_k A[i][k] * B[k][j]
//   (m = `bcast_dim`, n = `load_dim`, k = `reduce_dim` of the descriptor)
//
// A call that asks for post-ops applies the scales, bias and attribute
// post-ops, and stores to D instead of C:
//   D[i][j] = post_ops(C[i][j] * src_scale * wei_scale[j] + bias[j])
//             * dst_scale
//
// The output is partitioned into M blocks of `bd_block` rows, each split into
// N blocks of `ld_block2 * ld_block` columns. Each block keeps its result in
// registers across the whole batch and K reduction, then stores it once. The
// schedule decides how the loop nest walks the blocks.
void build_brgemm(const conf_t &cfg, const compute_path_t &cp, ir::ir_t &ir) {
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    switch (cfg.loop.schedule) {
        case schedule_t::looped: emit_looped_nest(ir, cfg, cp, regs); break;
        case schedule_t::unrolled: emit_unrolled_nest(ir, cfg, cp, regs); break;
    }
}

// Picks the compute path the caller chose when it created the descriptor.
//
// The finalized descriptor already holds the choice (`is_ace()`, `is_tmm`), so
// it is read from there. The ACE compute path does not exist yet and is
// rejected.
status_t init_compute_path(
        compute_path_kind_t &kind, const brgemm_desc_t &brg) {
    VCONDCHECK_BRGEMM_IR(!brg.is_ace(), VERBOSE_UNSUPPORTED_FEATURE, "ACE");

    kind = brg.is_tmm ? compute_path_kind_t::tmul : compute_path_kind_t::vector;
    return status::success;
}

// Fills the loop nest configuration.
//
// A tile descriptor counts its blocks differently from a vector one. `bdb` and
// `ldb` include the tail block, and the uker groups `bd_block2` M blocks and
// `ld_block2` N blocks into one block of the loop nest (`fill_imap()`).
status_t init_loop_conf(
        loop_conf_t &loop, compute_path_kind_t kind, const brgemm_desc_t &brg) {
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(one_of(brg.type, brgemm_addr, brgemm_offs),
            VERBOSE_UNSUPPORTED_FEATURE, "batch kind");
    VCONDCHECK_BRGEMM_IR(!brgattr.var_bs, VERBOSE_UNSUPPORTED_FEATURE,
            "variable batch size");
    VCONDCHECK_BRGEMM_IR(brg.layout == brgemm_row_major,
            VERBOSE_UNSUPPORTED_FEATURE, "column-major layout");
    VCONDCHECK_BRGEMM_IR(
            !brg.is_blocked, VERBOSE_UNSUPPORTED_FEATURE, "blocked layout");
    VCONDCHECK_BRGEMM_IR(everyone_is(false, brg.is_runtime_lda,
                                 brg.is_runtime_ldb, brg.is_runtime_ldc),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_vpad, brgattr.max_bottom_vpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "virtual padding");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_bpad, brgattr.max_bottom_bpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "batch padding");
    VCONDCHECK_BRGEMM_IR(
            brgattr.bd_mask_level == 0, VERBOSE_UNSUPPORTED_FEATURE, "bd_mask");

    // The classic kernels are picked the same way.
    const bool unrolled = brg.can_dispatch_uker();
    loop.schedule = unrolled ? schedule_t::unrolled : schedule_t::looped;

    const bool is_tmul = kind == compute_path_kind_t::tmul;
    VCONDCHECK_BRGEMM_IR(IMPLICATION(is_tmul, unrolled),
            VERBOSE_UNSUPPORTED_FEATURE, "looped schedule with tiles");

    // Reject the uker features that the unrolled schedule does not have yet,
    // under the conditions in which the uker uses them.
    if (unrolled) {
        auto prefetches = [](const brgemm_prf_t &prf) {
            return prf.sprinkled || prf.dist0 >= 0 || prf.dist1 >= 0
                    || prf.dist2 >= 0 || prf.distNTA >= 0;
        };

        VCONDCHECK_BRGEMM_IR(brgattr.max_bs == 1, VERBOSE_UNSUPPORTED_FEATURE,
                "batch of more than one element");
        VCONDCHECK_BRGEMM_IR(brg.innermost_loop == brgemm_ld_loop_innermost,
                VERBOSE_UNSUPPORTED_FEATURE, "M loop innermost");
        VCONDCHECK_BRGEMM_IR(!brg.interleave_tilestores_,
                VERBOSE_UNSUPPORTED_FEATURE, "interleaved tile stores");
        VCONDCHECK_BRGEMM_IR(!prefetches(brg.prfA), VERBOSE_UNSUPPORTED_FEATURE,
                "A prefetch");
        VCONDCHECK_BRGEMM_IR(!prefetches(brg.prfB), VERBOSE_UNSUPPORTED_FEATURE,
                "B prefetch");
        // The uker prefetches C only at the first two distances.
        VCONDCHECK_BRGEMM_IR(brg.prfC.dist0 < 0 && brg.prfC.dist1 < 0,
                VERBOSE_UNSUPPORTED_FEATURE, "C prefetch");
    }

    loop.batch_kind = brg.type;
    loop.max_bs = brgattr.max_bs;
    loop.load_dim = brg.load_dim;
    loop.lda = brg.LDA;
    loop.ldb = brg.LDB;
    loop.ldc = brg.LDC;
    loop.ld_block = brg.ld_block;
    loop.ld_block2 = brg.ld_block2;
    loop.rd_block = brg.rd_block;
    loop.rdb = brg.rdb;
    loop.rdb_tail = brg.rdb_tail;

    switch (kind) {
        case compute_path_kind_t::vector:
            loop.bd_block = brg.bd_block;
            loop.bdb = brg.bdb;
            loop.bdb_tail = brg.bdb_tail;
            loop.ldb2 = brg.ldb2;
            loop.ldb2_tail = brg.ldb2_tail;
            loop.ldb_tail = brg.ldb_tail;
            break;
        case compute_path_kind_t::tmul:
            VCONDCHECK_BRGEMM_IR(
                    everyone_is(0, brg.bdb_tail, brg.ldb_tail, brg.rdb_tail),
                    VERBOSE_UNSUPPORTED_FEATURE, "tile tails");

            // A group with fewer blocks than `bd_block2` or `ld_block2` is the
            // tail of its level.
            loop.bd_block = brg.bd_block2 * brg.bd_block;
            loop.bdb = brg.bdb / brg.bd_block2;
            loop.bdb_tail = (int)(brg.bdb % brg.bd_block2) * brg.bd_block;
            loop.ldb2 = brg.ldb / brg.ld_block2;
            loop.ldb2_tail = (int)(brg.ldb % brg.ld_block2);
            loop.ldb_tail = 0;
            break;
    }
    return status::success;
}

// Fills the configuration of the vector compute path.
status_t init_vector_conf(
        compute_path_conf_t &compute_path, const brgemm_desc_t &brg) {
    using namespace data_type;

    // `avx2` and `avx512_core` compute f32 directly. The other ISAs receive
    // inputs already converted to f32.
    VCONDCHECK_BRGEMM_IR(utils::one_of(brg.isa_impl, avx2, avx512_core,
                                 avx512_core_fp16, avx10_2),
            VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(utils::everyone_is(f32, brg.dt_a, brg.dt_b),
            VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(!brg.n_bcast_1_load, VERBOSE_UNSUPPORTED_FEATURE,
            "one-load microkernel");
    VCONDCHECK_BRGEMM_IR(!brg.embd_bcst, VERBOSE_UNSUPPORTED_FEATURE,
            "embedded broadcast microkernel");

    compute_path.dt_a = brg.dt_a;
    compute_path.dt_b = brg.dt_b;
    compute_path.dt_acc = f32;
    compute_path.vnni_granularity = brg.rd_step;
    return status::success;
}

// Fills the configuration of the TMUL compute path.
//
// The caller loads the tile configuration from `brgemm_init_tiles()` before it
// calls the kernel. The kernel converts the same configuration into `tiles`
// for code generation and takes the shapes of the operand tiles from it.
status_t init_tmul_conf(compute_path_conf_t &compute_path,
        ir::tile_config_t &tiles, const brgemm_desc_t &brg) {
    using namespace data_type;

    VCONDCHECK_BRGEMM_IR(
            is_superset(brg.isa_impl, amx_bf16), VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(utils::everyone_is(bf16, brg.dt_a, brg.dt_b),
            VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(
            !brg.fused_copy_a, VERBOSE_UNSUPPORTED_FEATURE, "fused copy A");
    VCONDCHECK_BRGEMM_IR(brg.brgattr.mem_advice == brgemm_hint_mem_advice_undef,
            VERBOSE_UNSUPPORTED_FEATURE, "memory advice");

    palette_config_t palette;
    const status_t st = brgemm_init_tiles(brg, (char *)&palette);
    VCONDCHECK_BRGEMM_IR(st == status::success, VERBOSE_UNSUPPORTED_FEATURE,
            "tile configuration");

    std::vector<ir::tile_shape_t> &shapes = tiles.shapes;
    for (int t = 0; t < brgemm_desc_t::AMX_TILES_NUM; t++) {
        shapes.push_back({palette.rows[t], palette.cols[t]});
    }

    compute_path.dt_a = brg.dt_a;
    compute_path.dt_b = brg.dt_b;
    compute_path.dt_acc = f32;
    compute_path.vnni_granularity = brg.rd_step;
    compute_path.a_tile = shapes[brg.get_A_tensor(0)];
    compute_path.b_tile = shapes[brg.get_B_tensor(0)];
    compute_path.c_tile = shapes[brg.get_C_tensor(0, 0)];

    auto load_hint = [](bool nt) {
        return nt ? ir::mem_hint_t::non_temporal : ir::mem_hint_t::none;
    };
    compute_path.a_load_hint = load_hint(brg.load_nt_A);
    compute_path.b_load_hint = load_hint(brg.load_nt_B);
    return status::success;
}

// Fills the output path configuration.
status_t init_output_conf(output_conf_t &output, compute_path_kind_t kind,
        const brgemm_desc_t &brg) {
    using namespace data_type;
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(
            everyone_is(f32, brg.dt_c, brg.dt_d), VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMM_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMM_IR(!brg.req_s8s8_compensation,
            VERBOSE_UNSUPPORTED_FEATURE, "s8s8 compensation");
    VCONDCHECK_BRGEMM_IR(!brgattr.use_intermediate_c_buffer,
            VERBOSE_UNSUPPORTED_FEATURE, "intermediate C buffer");
    VCONDCHECK_BRGEMM_IR(everyone_is(brgemm_broadcast_t::none, brg.zp_type_a,
                                 brg.zp_type_b, brg.zp_type_c),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VCONDCHECK_BRGEMM_IR(!brgattr.generate_skip_accumulation,
            VERBOSE_UNSUPPORTED_FEATURE, "skip accumulation");
    VCONDCHECK_BRGEMM_IR(brgattr.hint_prefetchw == brgemm_prfw_default,
            VERBOSE_UNSUPPORTED_FEATURE, "store prefetch hint");

    // Post-ops go through the JIT injector. Accept eltwise, binary and sum, and
    // only binary arguments the injector can handle on this ISA.
    if (brg.attr()) {
        const memory_desc_wrapper dst_d(brg.dst_md());
        const std::vector<injector::post_op_type> accepted
                = {injector::eltwise, injector::binary, injector::sum};
        VCONDCHECK_BRGEMM_IR(
                injector::post_ops_ok({brg.isa_impl, accepted,
                        brg.attr()->post_ops_, &dst_d,
                        /*sum_at_pos_0_only=*/false,
                        /*sum_requires_scale_one=*/false,
                        /*sum_requires_zp_zero=*/false,
                        /*sum_requires_same_params=*/false,
                        binary_injector::
                                get_all_strategies_supported_by_injector()}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // The binary injector loads an fp8 argument through fp8 converters,
        // which `ir::postops_injector_t` does not create.
        const post_ops_t &post_ops = brg.attr()->post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &e = post_ops.entry_[i];
            VCONDCHECK_BRGEMM_IR(!e.is_binary()
                            || !one_of(e.binary.src1_desc.data_type, f8_e5m2,
                                    f8_e4m3),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    // Scales are f32 values the kernel broadcasts or loads per N register:
    // src and dst scales are common, the weights scale common or per-N.
    VCONDCHECK_BRGEMM_IR(
            !brg.has_per_k_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.with_weights_scale_adjust,
            VERBOSE_UNSUPPORTED_FEATURE, "weights scale adjust");

    VCONDCHECK_BRGEMM_IR(IMPLICATION(brg.with_bias, brg.dt_bias == f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.is_runtime_ldd, VERBOSE_UNSUPPORTED_FEATURE,
            "runtime leading dimension");

    output.beta = brg.beta;
    output.dt_c = brg.dt_c;
    output.dt_bias = brg.dt_bias;
    output.dt_src_scales = brg.dt_src_scales;
    output.dt_wei_scales = brg.dt_wei_scales;

    // The order is the one oneDNN defines and the classic kernel follows:
    //   src scale -> weights scale -> bias -> post-ops -> dst scale.
    // The scales are applied one at a time, as in the classic kernel, so the
    // result rounds the same way.
    std::vector<output_stage_t> &stages = output.post_ops_stages;
    if (brg.with_src_scales) stages.push_back(output_stage_t::src_scale);
    if (brg.with_wei_scales)
        stages.push_back(brg.is_single_wei_scale
                        ? output_stage_t::wei_scale
                        : output_stage_t::wei_scale_per_n);
    if (brg.with_bias) stages.push_back(output_stage_t::bias);
    if (brg.with_eltwise || brg.with_binary || brg.with_sum) {
        stages.push_back(output_stage_t::injector_postops);
        output.post_ops = brg.attr()->post_ops_;
        output.dst_md = *brg.dst_md();
    }
    if (brg.with_dst_scales) stages.push_back(output_stage_t::dst_scale);

    // A post-ops call stores to D through the C displacements.
    VCONDCHECK_BRGEMM_IR(IMPLICATION(output.has_post_ops(), brg.LDC == brg.LDD),
            VERBOSE_UNSUPPORTED_FEATURE, "LDC != LDD");

    // The TMUL compute path stores its accumulators only with tile stores so
    // far, so no output stage can act on them.
    const bool is_tmul = kind == compute_path_kind_t::tmul;
    VCONDCHECK_BRGEMM_IR(IMPLICATION(is_tmul, !output.has_post_ops()),
            VERBOSE_UNSUPPORTED_FEATURE, "output stages on tiles");
    return status::success;
}

// Rejects a configuration whose addresses the builder cannot encode.
//
// Every address the builder emits is a base register plus a build-time
// displacement encoded in the instruction, and every pointer move is a
// build-time immediate, so both have to fit in int32. `max_looped_pos()`
// bounds the positions and moves along each level of the looped schedule, and
// `level_size()` the positions of the unrolled one. Checking the offset at the
// bounds of all levels covers every displacement and move, including the M
// step, which moves C along M and N in one `add`. The B prefetch is not
// checked here, see `emit_microkernel()`.
status_t check_displacements(const conf_t &conf) {
    auto fits = [](dim_t v) { return jit_generator_t::imm_fits_int32(v); };
    const bool looped = conf.loop.schedule == schedule_t::looped;
    auto max_pos = [&](const std::vector<block_run_t> &runs) {
        return looped ? max_looped_pos(runs) : level_size(runs);
    };
    const dim_t m = max_pos(m_runs(conf.loop));
    const dim_t n = max_pos(n_runs(conf.loop));
    const dim_t k = max_pos(k_runs(conf.loop));

    VCONDCHECK_BRGEMM_IR(fits(conf.a_off(m, k)), VERBOSE_UNSUPPORTED_FEATURE,
            "A displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.b_off(k, n)), VERBOSE_UNSUPPORTED_FEATURE,
            "B displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.c_off(m, n)), VERBOSE_UNSUPPORTED_FEATURE,
            "C displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.bias_off(n)), VERBOSE_UNSUPPORTED_FEATURE,
            "bias displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.wei_scale_off(n)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale displacement overflows int32");

    return status::success;
}

// Reads the descriptor into `conf` and `tiles`.
//
// Returns `status::success` if the kernel supports the descriptor, otherwise
// `status::unimplemented`. This is the only code that reads the descriptor for
// the builder. Each part of `conf` has its own function, which rejects
// what its module does not support. A descriptor the kernel rejects falls back
// to the classic kernel.
//
// `tiles` is the tile configuration, which only code generation reads. It is
// kept out of `conf`, so the builder never receives it. It stays empty for the
// vector compute path.
//
// TODO: reject `mem_advice` or implement it. For the vector compute path the
// classic kernel loads B with `vmovrsd` when it is set, and matmul sets it on
// avx10_2.
status_t init_conf(
        conf_t &conf, ir::tile_config_t &tiles, const brgemm_desc_t &brg) {
    // GEMV and DGMM have kernels of their own.
    VCONDCHECK_BRGEMM_IR(!brg.is_gemv, VERBOSE_UNSUPPORTED_FEATURE, "gemv");
    VCONDCHECK_BRGEMM_IR(!brg.is_dgmm, VERBOSE_UNSUPPORTED_FEATURE, "dgmm");

    CHECK(init_compute_path(conf.compute_path.kind, brg));
    const compute_path_kind_t kind = conf.compute_path.kind;
    CHECK(init_loop_conf(conf.loop, kind, brg));
    switch (kind) {
        case compute_path_kind_t::vector:
            CHECK(init_vector_conf(conf.compute_path, brg));
            break;
        case compute_path_kind_t::tmul:
            CHECK(init_tmul_conf(conf.compute_path, tiles, brg));
            break;
    }
    CHECK(init_output_conf(conf.output, kind, brg));
    return check_displacements(conf);
}

// IR-based BRGEMM kernel.
struct jit_brgemm_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_ir_kernel_t)

    jit_brgemm_ir_kernel_t(
            cpu_isa_t isa, const conf_t &conf, const ir::tile_config_t &tiles)
        : brgemm_kernel_t(jit_name(), isa), conf_(conf), tiles_(tiles) {}

    void generate() override {
        ir::ir_t ir;

        std::unique_ptr<compute_path_t> cp;
        switch (conf_.compute_path.kind) {
            case compute_path_kind_t::vector:
                cp = make_vector_compute_path();
                break;
            case compute_path_kind_t::tmul:
                cp = make_tmul_compute_path(ir, conf_);
                break;
        }

        build_brgemm(conf_, *cp, ir);

        ir::codegen_config_t codegen_cfg;
        if (conf_.output.has_stage(output_stage_t::injector_postops)) {
            ir::postops_config_t &postops = codegen_cfg.postops;
            postops.post_ops = &conf_.output.post_ops;
            postops.dst_md = &conf_.output.dst_md;
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
        }
        codegen_cfg.tiles = tiles_;

        ir::generate_kernel(*this, ir, codegen_cfg);
    }

private:
    conf_t conf_;
    ir::tile_config_t tiles_;
};

} // namespace
} // namespace brgemm_ir

brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg) {
    brgemm_ir::conf_t conf;
    ir::tile_config_t tiles;
    if (brgemm_ir::init_conf(conf, tiles, brg) != status::success)
        return nullptr;
    return new brgemm_ir::jit_brgemm_ir_kernel_t(brg.isa_impl, conf, tiles);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
