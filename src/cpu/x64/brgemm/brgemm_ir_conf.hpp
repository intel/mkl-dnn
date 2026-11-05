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

#ifndef CPU_X64_BRGEMM_BRGEMM_IR_CONF_HPP
#define CPU_X64_BRGEMM_BRGEMM_IR_CONF_HPP

// The builder configuration and the output block of the IR-based BRGEMM
// kernel, shared by the builder (`brgemm_ir.cpp`) and the code of each compute
// path.

#include <algorithm>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

// The block of the output one N-block emission computes.
//
//   bd_block     - rows of the broadcast dimension the block covers
//   ld_block2    - vector registers or tiles of the load dimension it covers
//   bd_start     - row the block starts at, counted from where the A and C
//                  pointers stand
//   ld_start     - load-dimension element the block starts at, counted from
//                  where the B and C pointers stand. B and C convert it with
//                  their own element size, so one count serves both.
//   ld_tail_mask - live elements of the last register, `none` when every
//                  register of the block is full
//
// `bd_start` and `ld_start` let a block be addressed without moving the
// pointers to it. A block that follows a run of one block is addressed
// this way.
struct out_block_t {
    int bd_block;
    int ld_block2;
    dim_t bd_start;
    dim_t ld_start;
    ir::vreg_t ld_tail_mask;
};

// Builder configuration
//
// `conf_t` holds one configuration per module of the builder:
//   loop         - the loop nest: the batch, the leading dimensions, and the
//                  blocking
//   compute_path - the compute path and the data types of A, B, and the
//                  accumulators
//   output       - the output path: beta, the bias, the scales, and the
//                  attribute post-ops
//
// Every field is a build-time constant.

// How the product is computed. The caller chooses it when it creates the
// descriptor, and the kernel never changes it.
//
//   vector - broadcasts one A element and multiply-adds it with B vectors into
//            vector accumulators
//   tmul   - multiplies tiles of A and B into tile accumulators
enum class compute_path_kind_t { vector, tmul };

// How the loop nest is emitted.
//
//   looped   - a run of more than one block is a runtime loop that moves the
//              pointers, as in `jit_brgemm_kernel_t`
//   unrolled - every block is emitted at its position and no pointer moves, as
//              in `jit_brgemm_amx_uker_t`
enum class schedule_t { looped, unrolled };

// Loop nest configuration.
//
//   schedule    - how the loop nest is emitted
//   batch_kind  - how a batch element gives the address of its A and B
//   max_bs      - maximum batch size known at IR generation time
//   load_dim    - N, the columns of C
//   lda/ldb/ldc - leading dimensions of A, B, and C, in elements
//   bd_block    - M rows per block
//   ld_block    - N elements per vector register or tile
//   ld_block2   - vector registers or tiles of N per block
//   rd_block    - K elements reduced per K block
//   bdb         - number of full M blocks
//   ldb2        - number of full N blocks
//   rdb         - number of full K blocks
//   bdb_tail    - M rows left over after the full M blocks
//   ldb2_tail   - full N registers or tiles left over after the full N blocks
//   ldb_tail    - N columns left over after every full N register
//   rdb_tail    - K elements left over after the full K blocks
struct loop_conf_t {
    schedule_t schedule = schedule_t::looped;
    brgemm_batch_kind_t batch_kind = brgemm_addr;
    dim_t max_bs = 0;
    dim_t load_dim = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;
    int bd_block = 0, ld_block = 0, ld_block2 = 0, rd_block = 0;
    dim_t bdb = 0, ldb2 = 0, rdb = 0;
    int bdb_tail = 0, ldb2_tail = 0, ldb_tail = 0, rdb_tail = 0;
};

// Compute path configuration.
//
//   kind                    - how the product is computed
//   dt_a/dt_b               - element data type of A and B in memory
//   dt_acc                  - accumulation data type
//   vnni_granularity        - K elements of a column that B stores together,
//                             1 for a plain B
//   a_tile/b_tile/c_tile    - shapes of the tiles of A, B, and the accumulators
//   a_load_hint/b_load_hint - hints of the A and B tile loads
struct compute_path_conf_t {
    compute_path_kind_t kind = compute_path_kind_t::vector;
    data_type_t dt_a = data_type::undef;
    data_type_t dt_b = data_type::undef;
    data_type_t dt_acc = data_type::undef;
    int vnni_granularity = 1;
    ir::tile_shape_t a_tile = {}, b_tile = {}, c_tile = {};
    ir::mem_hint_t a_load_hint = ir::mem_hint_t::none;
    ir::mem_hint_t b_load_hint = ir::mem_hint_t::none;
};

// A stage of the output path. Each one acts on the complete sum, so it runs
// only on a post-ops call.
//
//   src_scale        - multiplies by the common source scale
//   wei_scale        - multiplies by the common weights scale
//   wei_scale_per_n  - multiplies by the weights scale of each output column
//   bias             - adds the bias of each output column
//   injector_postops - applies the attribute post-ops (eltwise, binary, sum)
//                      through the post-ops injector
//   dst_scale        - multiplies by the common destination scale
enum class output_stage_t {
    src_scale,
    wei_scale,
    wei_scale_per_n,
    bias,
    injector_postops,
    dst_scale,
};

// Output path configuration.
//
//   beta            - output scaling: 0 overwrites C, 1 accumulates into C
//   dt_c            - element data type of C and D in memory
//   dt_bias         - element data type of the bias
//   post_ops        - the attribute post-ops, applied through the post-ops
//                     injector
//   dst_md          - the destination memory descriptor, which a binary
//                     post-op needs to locate its right-hand-side argument
//   dt_src_scales / dt_wei_scales - element data type of each scale
//   post_ops_stages - the stages of a post-ops call, in the order they run
//
// Adding C and the store are not stages. The builder emits them on every
// call: adding C before the stages when `beta` is 1, and the store after them.
struct output_conf_t {
    float beta = 0.0f;
    data_type_t dt_c = data_type::undef;
    data_type_t dt_bias = data_type::undef;
    post_ops_t post_ops;
    memory_desc_t dst_md;
    data_type_t dt_src_scales = data_type::undef;
    data_type_t dt_wei_scales = data_type::undef;
    std::vector<output_stage_t> post_ops_stages;

    bool has_stage(output_stage_t stage) const {
        return std::find(post_ops_stages.begin(), post_ops_stages.end(), stage)
                != post_ops_stages.end();
    }

    // True if the kernel has anything to apply on a post-ops call.
    bool has_post_ops() const { return !post_ops_stages.empty(); }
};

// Builder configuration.
//
//   loop         - loop nest configuration
//   compute_path - compute path configuration
//   output       - output path configuration
//
// The member functions derive the addressing of the builder from the module
// configurations.
struct conf_t {
    loop_conf_t loop;
    compute_path_conf_t compute_path;
    output_conf_t output;

    // Element sizes in bytes of A, B, C, and the bias. The bias size is 0
    // without a bias.
    int dt_sz_a() const {
        return (int)types::data_type_size(compute_path.dt_a);
    }
    int dt_sz_b() const {
        return (int)types::data_type_size(compute_path.dt_b);
    }
    int dt_sz_c() const { return (int)types::data_type_size(output.dt_c); }
    int dt_sz_bias() const {
        return output.has_stage(output_stage_t::bias)
                ? (int)types::data_type_size(output.dt_bias)
                : 0;
    }
    // Element size in bytes of a per-N weights scale, 0 for a common one,
    // which is never indexed.
    int dt_sz_wei_scales() const {
        return output.has_stage(output_stage_t::wei_scale_per_n)
                ? (int)types::data_type_size(output.dt_wei_scales)
                : 0;
    }

    // Distance in bytes a B load prefetches ahead, one K block of B rows.
    dim_t rdb_b_off() const { return dt_sz_b() * loop.rd_block * loop.ldb; }

    // Byte offset of an element of A, B, C, the bias, or a per-N weights
    // scale. `bd` is the row, `rd` the K index, and `ld` the N index of the
    // element, counted from where the pointers stand. The immediate of a
    // pointer move comes from the same functions. B keeps `vnni_granularity` K
    // elements of a column together, and `rd` of B is a multiple of it.
    //
    // These are the constants the builder encodes into the operation, the
    // `disp` of `ir::mem_t`. They are functions rather than expressions written
    // at each use because `init_conf()` range-checks the largest of them
    // against int32, and the check is only meaningful if there is consistency
    // between the check and the builder.
    dim_t a_off(dim_t bd, dim_t rd) const {
        return dt_sz_a() * (bd * loop.lda + rd);
    }

    dim_t b_off(dim_t rd, dim_t ld) const {
        return dt_sz_b() * (rd * loop.ldb + ld * compute_path.vnni_granularity);
    }

    dim_t c_off(dim_t bd, dim_t ld) const {
        return dt_sz_c() * (bd * loop.ldc + ld);
    }

    // One bias value per output column, so the offset has no row term.
    dim_t bias_off(dim_t ld) const { return dt_sz_bias() * ld; }

    // A per-N weights scale is indexed like the bias.
    dim_t wei_scale_off(dim_t ld) const { return dt_sz_wei_scales() * ld; }

    // Load-dimension element the vector `ld` of the block starts at, counted
    // from where the B and C pointers stand. B and C differ only in the
    // element size they scale it by.
    dim_t ld_pos(const out_block_t &blk, int ld) const {
        return blk.ld_start + (dim_t)ld * loop.ld_block;
    }
};

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
