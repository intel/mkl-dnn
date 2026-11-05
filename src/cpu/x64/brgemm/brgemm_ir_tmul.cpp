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

// The TMUL compute path of the IR-based BRGEMM kernel.

#include <cassert>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_ir_compute_path.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

namespace {

struct tmul_compute_path_t : public compute_path_t {
    tmul_compute_path_t(ir::ir_t &ir, const conf_t &cfg)
        : a_stride_(ir.new_gpr())
        , b_stride_(ir.new_gpr())
        , c_stride_(ir.new_gpr()) {
        ir.mov_imm(a_stride_, cfg.a_off(1, 0));
        ir.mov_imm(b_stride_, cfg.b_off(cfg.compute_path.vnni_granularity, 0));
        ir.mov_imm(c_stride_, cfg.c_off(1, 0));
    }

    // One tile per (M tile, N tile) of the block, in row order.
    std::vector<ir::vreg_t> init_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk) const override {

        const compute_path_conf_t &compute_path = cfg.compute_path;
        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        std::vector<ir::vreg_t> acc(n_m * n_n, ir::vreg_t::none);

        for (ir::vreg_t &t : acc) {
            t = ir.new_tile(compute_path.dt_acc, compute_path.c_tile);
            ir.tzero(t);
        }
        return acc;
    }

    std::vector<ir::vreg_t> load_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, ir::vreg_t c_ptr) const override {

        const compute_path_conf_t &compute_path = cfg.compute_path;
        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        std::vector<ir::vreg_t> acc(n_m * n_n, ir::vreg_t::none);

        for (int m = 0; m < n_m; m++) {
            for (int n = 0; n < n_n; n++) {
                ir::vreg_t &t = acc[m * n_n + n];
                t = ir.new_tile(compute_path.dt_acc, compute_path.c_tile);
                ir.tload(t, c_ptr, c_tile_off(cfg, blk, m, n), c_stride_);
            }
        }
        return acc;
    }

    void store_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t c_ptr) const override {

        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;

        for (int m = 0; m < n_m; m++) {
            for (int n = 0; n < n_n; n++) {
                const dim_t off = c_tile_off(cfg, blk, m, n);
                ir.tstore(c_ptr, off, acc[m * n_n + n], c_stride_);
            }
        }
    }

    // Loads an A tile for each M tile and a B tile for each N tile, and
    // multiplies each pair into its accumulator. The order follows
    // `gemm_microkernel_amx()` in `jit_brgemm_amx_uker.cpp`, which issues each
    // multiplication after the next tile load. For 2x2 tiles the order is A0 B0
    // B1 T00 A1 T01 T10 T11, where Tmn multiplies into the accumulator
    // (`m`, `n`).
    void emit_microkernel(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t a_ptr, ir::vreg_t b_ptr, dim_t rd_start,
            int rd_loop) const override {

        const compute_path_conf_t &compute_path = cfg.compute_path;

        // A row of an A tile holds the whole K block.
        assert(rd_loop * cfg.dt_sz_a() == compute_path.a_tile.colsb);

        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        std::vector<ir::vreg_t> a(n_m, ir::vreg_t::none);
        std::vector<ir::vreg_t> b(n_n, ir::vreg_t::none);

        for (int m = 0; m < n_m; m++) {
            const dim_t bd = blk.bd_start + m * compute_path.c_tile.rows;
            a[m] = ir.new_tile(compute_path.dt_a, compute_path.a_tile);
            ir.tload(a[m], a_ptr, cfg.a_off(bd, rd_start), a_stride_,
                    compute_path.a_load_hint);

            for (int n = 0; n < n_n; n++) {
                if (m == 0) {
                    const dim_t ld = cfg.ld_pos(blk, n);
                    b[n] = ir.new_tile(compute_path.dt_b, compute_path.b_tile);
                    ir.tload(b[n], b_ptr, cfg.b_off(rd_start, ld), b_stride_,
                            compute_path.b_load_hint);
                }

                // Multiply the previous pair: (`m`, `n - 1`), or the last pair
                // of the previous M tile when `n` is 0.
                if (n > 0) {
                    ir.tmma(acc[m * n_n + n - 1], a[m], b[n - 1]);
                } else if (m > 0) {
                    ir.tmma(acc[m * n_n - 1], a[m - 1], b[n_n - 1]);
                }
            }
        }
        ir.tmma(acc.back(), a.back(), b.back());
    }

private:
    // Row strides of the A, B, and C tiles in memory, in bytes. They are set
    // once for the whole kernel.
    ir::vreg_t a_stride_, b_stride_, c_stride_;

    // Number of M tiles in `blk`. There is no M tail, so all of them are full.
    static int m_tiles(const conf_t &cfg, const out_block_t &blk) {
        assert(blk.bd_block % cfg.compute_path.c_tile.rows == 0);
        return blk.bd_block / cfg.compute_path.c_tile.rows;
    }

    // Byte offset in C of the accumulator tile (`m`, `n`) of `blk`.
    static dim_t c_tile_off(
            const conf_t &cfg, const out_block_t &blk, int m, int n) {

        const dim_t bd = blk.bd_start + m * cfg.compute_path.c_tile.rows;
        return cfg.c_off(bd, cfg.ld_pos(blk, n));
    }
};

} // namespace

std::unique_ptr<compute_path_t> make_tmul_compute_path(
        ir::ir_t &ir, const conf_t &cfg) {
    return utils::make_unique<tmul_compute_path_t>(ir, cfg);
}

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
