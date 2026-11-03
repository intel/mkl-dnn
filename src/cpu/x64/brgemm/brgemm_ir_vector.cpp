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

// The vector compute path of the IR-based BRGEMM kernel.

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_ir_compute_path.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

namespace {

struct vector_compute_path_t : public compute_path_t {
    // One accumulator per (M row, N register), in row order. They are the rows
    // of the output that the output path works on.
    std::vector<ir::vreg_t> init_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk) const override {
        std::vector<ir::vreg_t> acc(
                blk.bd_block * blk.ld_block2, ir::vreg_t::none);
        for (int bd = 0; bd < blk.bd_block; bd++) {
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const int i = bd * blk.ld_block2 + ld;
                acc[i] = ir.new_vec(cfg.compute_path.dt_acc);
                ir.vzero(acc[i]);
            }
        }
        return acc;
    }

    // Loads one B vector per N register, then for each M row broadcasts one A
    // element and multiply-adds it into that row's accumulators. This is the
    // `!n_bcast_1_load` variant of `gemm_microkernel()` in
    // `jit_brgemm_kernel.cpp`.
    void emit_microkernel(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t a_ptr, ir::vreg_t b_ptr, dim_t rd_start,
            int rd_loop) const override {
        std::vector<ir::vreg_t> b(blk.ld_block2, ir::vreg_t::none);
        for (int ld = 0; ld < blk.ld_block2; ld++)
            b[ld] = ir.new_vec(cfg.compute_path.dt_b);
        const ir::vreg_t a = ir.new_vec(cfg.compute_path.dt_a);

        for (int rd = 0; rd < rd_loop; rd++) {
            const dim_t rd_pos = rd_start + rd;

            // Only the last register of a block can be partial, and a block
            // that has one holds nothing else, so the mask covers every B load
            // here.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(b[ld], b_ptr,
                            cfg.b_off(rd_pos, cfg.ld_pos(blk, ld)),
                            cfg.compute_path.dt_b);
                else
                    ir.vload_masked(b[ld], b_ptr,
                            cfg.b_off(rd_pos, cfg.ld_pos(blk, ld)),
                            blk.ld_tail_mask, cfg.compute_path.dt_b);
            }

            // One prefetch per B register, issued on the first `ld_block2`
            // rows, so each reduction step fetches the next K block of B
            // exactly once.
            int n_pf_b = 0;
            for (int bd = 0; bd < blk.bd_block; bd++) {
                ir.vload_bcast(a, a_ptr, cfg.a_off(blk.bd_start + bd, rd_pos),
                        cfg.compute_path.dt_a);
                if (n_pf_b < blk.ld_block2) {
                    // The prefetch address is one K block past the B loads, so
                    // it can be outside B. `check_displacements()` does not
                    // cover it, and the prefetch is skipped if its displacement
                    // does not fit in int32.
                    const dim_t pf_off
                            = cfg.b_off(rd_pos, cfg.ld_pos(blk, n_pf_b))
                            + cfg.rdb_b_off();
                    if (jit_generator_t::imm_fits_int32(pf_off))
                        ir.prefetch(b_ptr, pf_off);
                    n_pf_b++;
                }

                for (int ld = 0; ld < blk.ld_block2; ld++)
                    ir.vdot(acc[bd * blk.ld_block2 + ld], b[ld], a);
            }
        }
    }
};

} // namespace

std::unique_ptr<compute_path_t> make_vector_compute_path() {
    return utils::make_unique<vector_compute_path_t>();
}

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
