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

#ifndef CPU_X64_BRGEMM_BRGEMM_IR_COMPUTE_PATH_HPP
#define CPU_X64_BRGEMM_BRGEMM_IR_COMPUTE_PATH_HPP

// The interface between the builder of the IR-based BRGEMM kernel
// (`brgemm_ir.cpp`) and its compute paths.

#include <cassert>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"

#include "cpu/x64/brgemm/brgemm_ir_conf.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

// One compute path of the builder: the accumulators of an output block and
// the microkernel that reduces into them.
//
// For each output block the builder calls `init_accumulators()` before the
// batch loop, then `emit_microkernel()` for each K run of each batch element.
// The unrolled schedule calls `load_accumulators()` instead of
// `init_accumulators()` when beta is 1, and `store_accumulators()` at the end
// of the block. The compute path chooses the layout of the accumulators.
struct compute_path_t {
    virtual ~compute_path_t() = default;

    // Creates the accumulators of `blk` and zeroes them.
    virtual std::vector<ir::vreg_t> init_accumulators(
            ir::ir_t &ir, const conf_t &cfg, const out_block_t &blk) const
            = 0;

    // Creates the accumulators of `blk` and loads them from C at `c_ptr`. Like
    // `store_accumulators()`, only the TMUL compute path implements it.
    virtual std::vector<ir::vreg_t> load_accumulators(ir::ir_t &ir,
            const conf_t &cfg, const out_block_t &blk, ir::vreg_t c_ptr) const {
        assert(!"load_accumulators: not supported by the compute path");
        return {};
    }

    // Stores the accumulators of `blk` to C at `c_ptr` unchanged.
    //
    // Only the unrolled schedule calls it, and that schedule runs only the
    // TMUL compute path (`init_conf()`). The other compute paths keep this
    // default.
    virtual void store_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t c_ptr) const {
        assert(!"store_accumulators: not supported by the compute path");
    }

    // Reduces `rd_loop` elements of K into `acc`, starting at K index
    // `rd_start`, counted from where `a_ptr` and `b_ptr` stand.
    virtual void emit_microkernel(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t a_ptr, ir::vreg_t b_ptr, dim_t rd_start,
            int rd_loop) const
            = 0;
};

std::unique_ptr<compute_path_t> make_vector_compute_path();

// The TMUL compute path sets the row strides of its tiles once, at the end of
// `ir`, so it is created before the loop nest is built.
std::unique_ptr<compute_path_t> make_tmul_compute_path(
        ir::ir_t &ir, const conf_t &cfg);

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
