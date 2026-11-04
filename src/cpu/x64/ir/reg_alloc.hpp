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

#ifndef CPU_X64_IR_REG_ALLOC_HPP
#define CPU_X64_IR_REG_ALLOC_HPP

#include <cstdint>
#include <vector>

#include "common/utils.hpp"

#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Describes where a virtual register was assigned by the allocator.
// The location is determined by `spilled`:
//
// - If `spilled == false`, the value is stored in the physical register
//   `phys`. The number is the register index within its group
//   (for example: gpr 0=rax, ..., 15=r15 and vec 0=ymm0, ..., 15=ymm15).
//   In this case, `slot` is not used.
//
// - If `spilled == true`, the value is stored on the stack at byte offset
//   `slot` in the spill area. In this case, `phys` is not used.
//   The emitter moves the value through a temp register at each operation
//   that reads or writes it (see `temp_reg_t`).
//
// - A vreg that no operation references, and a tile whose shape has no file,
//   have no location. `spilled` is false and `phys` is -1.
struct assignment_t {
    bool spilled = false;
    int phys = -1;
    size_t slot = 0;
};

// Most temps one operation gets, per register kind, indexed by
// `(int)reg_kind_t`. Operations other than `inject_postops` have at most 2 gpr,
// 3 vec and 1 mask operands, so these caps cover every spilled operand. The
// caps are upper bounds that come from the IR alone, the same on every target.
// Whether a value can spill is up to its file (see `reg_file_t::spillable`), so
// AVX-512 masks stay in k-registers despite the mask cap. Tiles are never
// spilled on any target, so they get no temps.
constexpr int max_temps_per_op[] = {2, 3, 1, 0};

// A register that holds a spilled value while one operation executes.
//
// The emitter reloads the value into it before the operation and stores it
// back after. A spilled value thus needs a register only at the operations
// that reference it. The allocator picks one that holds no other value live
// at that operation, spilling another value to free one if needed. The operand
// gets no temp when every register holds an operand of that operation, or when
// the operation already has `max_temps_per_op` temps of its kind. The emitter
// then fails the kernel.
//
//   vreg - the spilled virtual register
//   phys - physical register that holds it during the operation
struct temp_reg_t {
    vreg_t vreg;
    int phys;
};

// The final allocation result.
//
// - `assignments` contains one `assignment_t` for each virtual register,
//   indexed by virtual register id.
// - `temps` contains, for each operation, one `temp_reg_t` per distinct spilled
//   operand, up to `max_temps_per_op` of each kind. It is indexed by operation.
// - `frame_bytes` is the total amount of stack space needed for spilled
//   values. The kernel reserves this space with a single `sub rsp`.
// - `any_spill` is true if any virtual register was spilled to the stack.
struct reg_alloc_result_t {
    std::vector<assignment_t> assignments;
    std::vector<std::vector<temp_reg_t>> temps;
    size_t frame_bytes = 0;
    bool any_spill = false;
};

// A register file contains physical registers the allocator may assign, and the
// stack-slot size used when a spill is needed.
//
// `regs` holds the register indices available for allocation (for example, all
// general-purpose registers except reserved ones such as `rsp` and the argument
// pointer).
//
// `slot_size` is how many bytes a spilled value needs on the stack
// (8 for a GPR, 32 for a YMM, 64 for a ZMM).
//
// `spillable` is false for a file whose values must stay in registers, such as
// AVX-512 k-registers. The allocator never spills such a value to make room.
// If one still finds no free register, it is marked `spilled` only to record
// the failure, and the emitter fails the kernel.
//
// `tile_shape` is the shape of every register in a tile file, and empty in the
// other files.
struct reg_file_t {
    std::vector<int> regs;
    size_t slot_size = 0;
    bool spillable = true;
    tile_shape_t tile_shape = {};
};

// `n` files of `reg_pools_t::files` starting at `first`.
struct file_range_t {
    int first;
    int n;

    bool operator==(const file_range_t &other) const {
        return first == other.first && n == other.n;
    }
};

// The register files and the files each register kind allocates from.
//
// A file is a set of physical registers, and any of them can hold any vreg
// that allocates from the file. `gpr`, `vec`, and `mask` have one file each.
// Two kinds may share one, e.g. on AVX2* a mask is a vector register, so `vec`
// and `mask` map to the same file and compete for the same pool. On AVX-512 a
// mask is a k-register and has its own file. A tile has one file per tile shape
// in the tile configuration of the kernel, so one kind can have several files
// and the number of files depends on the kernel.
//
// `kind_to_file` holds the files of each kind as a range of `files`, indexed by
// `(int)reg_kind_t`. The files are ordered gpr, vec, mask (AVX-512 only), then
// the tile files in the order their shapes first appear in the tile
// configuration. A kind with no file, such as tile in a kernel without tiles,
// has an empty range.
//
// This structure is created and filled by make_reg_config() based on the target
// ISA and the tile configuration, and later used by allocate_registers() during
// allocation.
struct reg_pools_t {
    std::vector<reg_file_t> files;
    std::vector<file_range_t> kind_to_file;

    // Returns the file a vreg with `info` allocates from: the file of its kind
    // whose `tile_shape` is the shape of the vreg, or -1 when there is none.
    // gpr, vec, and mask vregs and their files have the empty shape. The shape
    // is compared in a range of one file too, so a tile of a shape the tile
    // configuration lacks does not land in the only tile file.
    int file_of(const vreg_info_t &info) const {
        const file_range_t &r = kind_to_file[(int)info.kind];
        for (int f = r.first; f < r.first + r.n; f++)
            if (files[f].tile_shape == info.shape) return f;
        return -1;
    }
};

// Compute liveness for each operation `i`.
//
// A value is `live` at operation `i` if some future operation may still
// read it before it is overwritten. In other words, the value must be kept
// available because it might be needed later.
//
// Two values that are live at the same operation cannot share a register.
//
// Backward data-flow to a fixed point. For each operation `i`:
//
//   1. Computing `live_in` at operation `i`:
//      A variable is live before `i` if `i` uses it, or if it is needed
//      later and not overwritten by `i`.
//      Formula: live_in[i] = use[i] U (live_out[i] - def[i])
//
//   2. Computing `live_out` at operation `i`:
//      A variable is live after `i` if any successor may need it.
//      Formula: live_out[i] = union of live_in over all successors of `i`
//
// Scan each `i` from last to first, and repeat the whole scan until nothing
// changes (the fixed point). Successors are `i+1` for a plain operation, `i+1`
// plus the loop body start for loop_end (the back-edge), `i+1` plus the else
// body or the if_end for if_begin, and the if_end for _else.
//
// Small example: `p` is read at the top of a loop body and rewritten at the
// bottom:
//
//   0  loop_begin
//   1    v = load [p]      (p read)
//   2    p = p + stride    (p rewritten)
//   3  loop_end            (back-edge to 0)
//
// `p` must be live across the whole body, because the value written at 2 is
// read at 1 on the next turn. One backward pass finds most of it. The entry to
// loop_end needs the back-edge, so it appears only on the second pass. A third
// pass changes nothing, which is the fixed point.
void compute_liveness(
        const ir_t &ir, std::vector<std::vector<int8_t>> &live_in);

// Export for testing.
reg_alloc_result_t DNNL_API allocate_registers(
        const ir_t &ir, const reg_pools_t &pools);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
