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

#ifndef CPU_X64_IR_IR_HPP
#define CPU_X64_IR_IR_HPP

// Minimal linear IR for JIT kernels.
//
// An `ir_t` is a flat list of operations. Each operation reads and writes
// virtual registers: integer-named placeholders for the values a kernel works
// with, used instead of physical registers until the allocator assigns them.
// Every virtual register belongs to one of four kinds (gpr/vec/mask/tile),
// which defines the kind of physical register it can later land in. Operations
// are target-neutral. The concrete instructions are chosen later by the
// emitter.
//
// A vec vreg also carries the element data type of the values it holds. The
// operations stay data-type generic. The builder tags each vec vreg with a
// dtype and the emitter lowers each op to the instruction for that dtype.
//
// A tile vreg carries a data type and its physical shape (see `tile_shape_t`).
//
// The virtual registers are mutable. A virtual register is a named value that
// may be written more than once. In this IR we deliberately allow reassignment,
// e.g. a pointer register is loaded and then advanced with `add` each loop
// iteration, reusing the same name. Each virtual register occupies one physical
// location (register or spill slot) for the whole of its live range, so a write
// overwrites that location in place. This keeps the builder and the register
// allocator simple. The allocator relies only on liveness.
//
// Memory operands are base register + displacement, where the displacement is
// a single constant offset known at build time and encoded directly into the
// instruction (`[reg + disp]`). There is no second index register and no
// scale. Any distance that is only known at run time is not a displacement, the
// builder instead folds it into the base pointer with an explicit `add` (a
// running pointer that is advanced each iteration instead of recomputing an
// index). Restricting addresses to one register keeps every memory op reading
// at most one pointer, which lowers register pressure and simplifies spilling.
// A tile load or store also reads a stride gpr, the byte distance between rows.
//
// Loops are explicit structured nodes with a runtime counter register
// (loop_begin/loop_end). Compile-time loops are unrolled by the builder.
// Conditionals are explicit structured nodes too (if_begin/_else/if_end).

#include <vector>

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// gpr  - General-purpose register
// vec  - Vector register
// mask - Mask register (emitter decides whether lower it to k or vector
//        register)
// tile - Tile register, `rows` rows of `colsb` bytes (see `tile_shape_t`)

enum class reg_kind_t { gpr, vec, mask, tile };

// Shape of a tile register: `rows` rows of `colsb` bytes each, as the tile
// configuration sets it. A row holds `colsb / sizeof(dt)` elements of the data
// type of the vreg. Tiles with the same shape share a register file whatever
// their data type, and a tile allocates only from the file of its shape.
struct tile_shape_t {
    int rows;
    int colsb;

    bool operator==(const tile_shape_t &other) const {
        return rows == other.rows && colsb == other.colsb;
    }
};

// vreg_t is a strong alias for `int`. It does not implicitly convert to or from
// `int`, so a vreg and a raw int cannot be mixed up by accident. `none` marks
// an unset id. Cast to `int` where a raw index is needed.

// A virtual register id. A placeholder the allocator later maps to a physical
// register or a spill slot. Its kind (gpr/vec/mask) and, for a vec, element
// data type live separately in vreg_info_.
enum class vreg_t : int { none = -1 };

// Integer comparison of a conditional (see `if_begin`). Signed, because gpr
// values in this IR are counts, offsets, and flags.
enum class cmp_t { eq, ne, lt, le, gt, ge };

enum class op_kind_t {
    // General-purpose register operations

    // dst = immediate value
    mov_imm,
    // dst = source register (s0)
    mov_reg,
    // dst += immediate value
    add_imm,
    // dst += source register (s0)
    add_reg,
    // dst = [base + disp], base is the parameter register if is_param
    load,

    // Vector operations
    //
    // dst = 0 (clear vector)
    vzero,
    // dst = [base + disp] (load full vector)
    vload,
    // [base + disp] = s0 (store full vector)
    vstore,
    // dst = [base + disp] (load one element, the rest of dst reads as zero)
    vload_scalar,
    // [base + disp] = s0 (store one element)
    vstore_scalar,
    // dst = broadcast([base + disp]) (load one element, replicated across dst)
    vload_bcast,
    // dst += sum_{i=0}^{N-1} (s0[i] * s1[i]), where N is the dot length
    vdot,
    // dst += s0 (vector add)
    vadd,
    // dst *= s0 (vector multiply)
    vmul,
    // horizontal reduction of dst; result in element 0. s0 is scratch
    // (overwritten).
    vhreduce,

    // Post-ops
    //
    // Apply post-ops via an injector to a set of accumulators. The injector is
    // not IR-based, so lowering to it needs interoperability code (see
    // `postops_injector_t`). Operands are in `inject_postops_args_t`.
    inject_postops,

    // Mask operations
    //
    // set_mask_imm creates a mask for `imm` active elements.
    set_mask_imm,

    // Masked vector load/store
    //
    // dst = [base + disp] under the mask vreg s1. Masked-out elements read as
    // zero.
    vload_masked,
    // [base + disp] = s0 under the mask vreg s1.
    vstore_masked,

    // prefetch [base + disp] into cache. Reads base, writes nothing.
    prefetch,

    // Tile operations
    //
    // dst = 0 (clear tile)
    tzero,
    // row r of dst = [base + disp + r * s1] for every row r. s1 is the stride
    // gpr.
    tload,
    // [base + disp + r * s1] = row r of s0 for every row r. s1 is the stride
    // gpr.
    tstore,

    // Control flow
    //
    // initialize loop counter (dst = imm or s0 if init_is_reg)
    loop_begin,
    // decrement counter and jump to matching loop_begin if not zero
    loop_end,
    // if s0 `cmp` (s1, or imm when s1 is none) holds, fall through into the
    // then body. Otherwise continue after the matching _else, or at the
    // matching if_end when there is no else.
    if_begin,
    // start the else body, which the then body skips
    _else,
    // close the conditional
    if_end,
};

// A memory address of the form base register + displacement.
//   base     - virtual gpr holding the pointer. Runtime offsets are folded into
//              it by the builder (incrementing the pointer), so the only
//              non-register part of an address is a build-time constant.
//   disp     - that constant byte offset, encoded directly in the instruction
//              There is no index register and no scale.
//   is_param - when true the base is not `base` but the kernel's argument
//              pointer (a fixed register set by the ABI). Used to read fields
//              of the kernel-parameter struct. The register allocator does not
//              manage it, so it is not counted as a virtual-register use.
struct mem_t {
    // virtual gpr holding the pointer (ignored when is_param)
    vreg_t base = vreg_t::none;
    // constant byte offset
    dim_t disp = 0;
    // base is the kernel-argument pointer
    bool is_param = false;
};

// Hints of a memory access, combined with `|`. A hint never changes the result
// or the legality of an operation. The emitter uses the hinted instruction
// where the target has it and the plain instruction otherwise.
//
//   non_temporal - the data is not reused soon
enum class mem_hint_t : unsigned {
    none = 0,
    non_temporal = 1u << 0,
};

inline mem_hint_t operator|(mem_hint_t a, mem_hint_t b) {
    return static_cast<mem_hint_t>(
            static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

inline mem_hint_t operator&(mem_hint_t a, mem_hint_t b) {
    return static_cast<mem_hint_t>(
            static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

// An `op_t` is a single IR operation. One fixed struct represents every
// operation and works like a tagged union, where the `kind` field decides which
// operation it is. Each operation only uses the fields it needs and ignores the
// rest, which stay at default values.
//
// The meaning of each field for a specific operation is explained in the
// `op_kind_t` comments. Whether dst, s0, and s1 are used for reading or writing
// is clearly defined in ir_t::def_use().
//
// kind - which operation this is. Determines how other fields are used.
// dst  - virtual register that is written to, or `none` if none is written.
//        Some ops (e.g. vdot/vadd) both read from and write to dst.
// s0,s1 - source virtual registers (inputs), or `none` if not used.
// imm   - immediate value whose meaning depends on the kind:
//         * mov_imm        -> literal constant
//         * loop_begin     -> loop trip count
//         * set_mask_imm   -> active element count
//         * inject_postops -> index into inject_postops_args()
//         * if_begin       -> right-hand side of the comparison when s1 is
//                             none
// mem   - memory address used only by load/store operations.
// mem_dt - element data type in memory, set by every vector load and store. It
//        is the data type of the vreg when the access moves the value as it
//        is, and a different one when the access converts.
// hint  - for tload, the hints of the access (see `mem_hint_t`).
// match - for loop_end, index of matching loop_begin operation. For _else and
//         if_end, index of matching if_begin operation.
// cmp   - for if_begin, how s0 is compared with s1 or imm.
// init_is_reg - for loop_begin, if true, initialize loop counter from s0
//                (runtime value) instead of imm.
struct op_t {
    op_kind_t kind;
    vreg_t dst = vreg_t::none;
    vreg_t s0 = vreg_t::none, s1 = vreg_t::none;
    dim_t imm = 0;
    mem_t mem;
    data_type_t mem_dt = data_type::undef;
    mem_hint_t hint = mem_hint_t::none;
    int match = -1;
    cmp_t cmp = cmp_t::eq;
    bool init_is_reg = false;
};

// Metadata for virtual registers that contains the register kind and the
// data type of the values it holds. `dt` is `undef` for gpr and mask. `shape`
// is the shape of a tile and empty for the other kinds.
struct vreg_info_t {
    reg_kind_t kind;
    data_type_t dt = data_type::undef;
    tile_shape_t shape = {};
};

// Arguments of an `inject_postops` operation. They live in a side table in
// `ir_t` because an `op_t` has a fixed field set and cannot hold the variadic
// `acc` list. Each `inject_postops` op stores only its index into that table in
// `imm`.
//
//   acc          - accumulators transformed in place by the post-ops
//   base_ptr     - gpr holding the output base pointer, used by binary post-ops
//                  to address their right-hand-side argument
//   out_byte_off - output byte offset of each accumulator from `base_ptr`,
//                  parallel to `acc`. Locates each accumulator in the
//                  destination tensor so binary post-ops address the matching
//                  right-hand-side slice
//   n_elems      - number of valid leading elements in each accumulator, the
//                  same for all of them. It ranges from 1 (a scalar) to the
//                  vector length (a full register). Binary and sum post-ops
//                  read no right-hand-side or destination element past it
//
// `base_ptr` and `out_byte_off` are unused by an eltwise-only chain.
struct inject_postops_args_t {
    std::vector<vreg_t> acc;
    vreg_t base_ptr = vreg_t::none;
    std::vector<dim_t> out_byte_off;
    int n_elems = 0;
};

// An `ir_t` is the operation list plus, for each virtual register, its info
// (kind and for a vec its element data type) so the allocator knows which
// physical registers it can use and the emitter knows which dtype-specific
// instruction to lower each op to.
//
// * The builder fills it through the helpers below
// * The allocator and emitter read it
//
// Export for testing.
#ifdef _MSC_VER
// ir_t is exported whole so that new member functions are covered automatically
// without annotating each one. It has std::vector members, which triggers
// C4251. The library and the tests are build by the same compiler, so no
// mismatch is possible. Silence it.
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
struct DNNL_API ir_t {
    const std::vector<vreg_info_t> &vreg_info() const { return vreg_info_; }
    const std::vector<op_t> &ops() const { return ops_; }
    const std::vector<inject_postops_args_t> &inject_postops_args() const {
        return inject_postops_args_;
    }

    // Convenience wrappers around `new_vreg` for the specific kinds.
    vreg_t new_gpr() { return new_vreg(reg_kind_t::gpr); }
    vreg_t new_vec(data_type_t dt) { return new_vreg(reg_kind_t::vec, dt); }
    vreg_t new_mask() { return new_vreg(reg_kind_t::mask); }
    // `shape` must not be empty.
    vreg_t new_tile(data_type_t dt, tile_shape_t shape);

    int n_vregs() const { return (int)vreg_info_.size(); }
    int n_ops() const { return (int)ops_.size(); }

    // Each interface appends one operation and names registers by `id`. The
    // ones returning `int` return the new operation's index.
    //
    // Refer to documentation for `op_kind_t` for each interface. The notes here
    // add only the parameter-level details that are not obvious from the
    // signature.

    // gpr
    int mov_imm(vreg_t dst, dim_t imm);
    void mov_reg(vreg_t dst, vreg_t src);
    void add_imm(vreg_t dst, dim_t imm);
    void add_reg(vreg_t dst, vreg_t src);
    // Like `load`, but reads a field of the kernel-argument struct through the
    // ABI parameter pointer instead of a virtual base register.
    void load_param(vreg_t dst, dim_t disp);
    void load(vreg_t dst, vreg_t base, dim_t disp);

    // vec
    // `mem_dt` is the element data type in memory. Pass the data type of the
    // vreg to move the value as it is, or a different one to convert it.
    void vzero(vreg_t dst);
    void vload(vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt);
    void vstore(vreg_t base, dim_t disp, vreg_t src, data_type_t mem_dt);
    void vload_scalar(vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt);
    void vstore_scalar(vreg_t base, dim_t disp, vreg_t src, data_type_t mem_dt);
    void vload_bcast(vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt);
    void vdot(vreg_t dst, vreg_t a, vreg_t b);
    void vadd(vreg_t dst, vreg_t src);
    void vmul(vreg_t dst, vreg_t src);
    // `workspace` is scratch. It is overwritten by this call, so pass a vreg
    // whose value is not needed afterwards.
    void vhreduce(vreg_t dst, vreg_t workspace);

    // vec (masked)
    // `mask` comes from `set_mask_imm` and is required. Use `vload`/`vstore`
    // for a full vector and `vload_scalar`/`vstore_scalar` for one element,
    // since neither needs a mask register.
    void vload_masked(vreg_t dst, vreg_t base, dim_t disp, vreg_t mask,
            data_type_t mem_dt);
    void vstore_masked(vreg_t base, dim_t disp, vreg_t src, vreg_t mask,
            data_type_t mem_dt);

    void prefetch(vreg_t base, dim_t disp);

    // mask
    void set_mask_imm(vreg_t mask, int n_elems);

    // tile
    void tzero(vreg_t dst);
    // `stride` is a gpr that holds the byte distance between rows in memory.
    void tload(vreg_t dst, vreg_t base, dim_t disp, vreg_t stride,
            mem_hint_t hint = mem_hint_t::none);
    void tstore(vreg_t base, dim_t disp, vreg_t src, vreg_t stride);

    // post-ops
    // Apply post-ops to `acc` in place. `base_ptr` and `out_byte_off` describe
    // where each accumulator lands in the output, and `n_elems` how many of its
    // leading elements are valid (see `inject_postops_args_t`).
    void inject_postops(const std::vector<vreg_t> &acc, vreg_t base_ptr,
            const std::vector<dim_t> &out_byte_off, int n_elems);

    // control flow
    // Pass the returned op index to `loop_end` to close the loop.
    int loop_begin_imm(vreg_t counter, dim_t count);
    // Like `loop_begin_imm`, but the iteration count is the runtime value in
    // `init` rather than the compile-time `count`.
    int loop_begin_reg(vreg_t counter, vreg_t init);
    // Close the loop opened by `loop_begin_imm` or `loop_begin_reg`, passing
    // the op index it returned as `begin_idx`.
    void loop_end(vreg_t counter, int begin_idx);

    // Open a conditional on `lhs cmp imm`, or on `lhs cmp rhs` for a gpr
    // `rhs`. Return the op index, which the `else_if`, `_else`, and `if_end`
    // of the same statement take as `if_idx`. `imm` must fit in 32 bits.
    //
    // Example:
    //   const int if_idx = ir.if_begin(pos, cmp_t::lt, 16);
    //   ...                                   // pos < 16
    //   ir.else_if(if_idx, pos, cmp_t::lt, 32);
    //   ...                                   // 16 <= pos < 32
    //   ir._else(if_idx);
    //   ...                                   // pos >= 32
    //   ir.if_end(if_idx);
    int if_begin(vreg_t lhs, cmp_t cmp, dim_t imm);
    int if_begin(vreg_t lhs, cmp_t cmp, vreg_t rhs);
    // Close the current branch of `if_idx` and open one on a new condition.
    // This appends `_else` followed by a nested `if_begin`.
    void else_if(int if_idx, vreg_t lhs, cmp_t cmp, dim_t imm);
    void else_if(int if_idx, vreg_t lhs, cmp_t cmp, vreg_t rhs);
    // Close the current branch of `if_idx` and open the branch taken when no
    // condition of the statement holds.
    void _else(int if_idx);
    // Close the conditional `if_idx`, including the ifs its `else_if` calls
    // opened.
    void if_end(int if_idx);

    // Fill defs/uses with the vregs this operation writes/reads. Liveness and
    // the allocator depend on it, so it must match what the emitter emits.
    void def_use(const op_t &op, std::vector<int> &defs,
            std::vector<int> &uses) const;

private:
    // Add a vreg of kind `k` and, for a vec, element data type `dt`.
    // Return its id.
    vreg_t new_vreg(reg_kind_t k, data_type_t dt = data_type::undef);

    // Append an `if_begin`.
    int append_if_begin(vreg_t lhs, cmp_t cmp, vreg_t rhs, dim_t imm);

    // Check that `if_idx` is the chain head of the innermost open conditional,
    // the only one `else_if`, `_else`, and `if_end` may continue or close.
    void assert_innermost_if(int if_idx) const;

    // A loop or conditional that has been opened and not yet closed.
    //   begin    - op index of its own loop_begin or if_begin, which its
    //              `_else`, `if_end`, or `loop_end` links to through `match`
    //   head     - op index of the first if_begin of its `else_if` chain (the
    //              chain head), which the builder passes as `if_idx`. It is
    //              equal to `begin` unless `else_if` opened this if. Every
    //              entry with the same head is closed by one `if_end(head)`.
    //   has_else - `_else` was appended for `begin`
    struct open_region_t {
        int begin;
        int head;
        bool has_else;
    };

    // Info for each vreg, indexed by its id
    std::vector<vreg_info_t> vreg_info_;
    // All operations, in order
    std::vector<op_t> ops_;
    // Arguments of each `inject_postops` operation.
    std::vector<inject_postops_args_t> inject_postops_args_;
    // Open loops and conditionals, innermost last. The builder calls use it to
    // check nesting, and `if_end` uses it to find the ifs that `else_if`
    // opened.
    std::vector<open_region_t> open_regions_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// Loop helpers shared by IR builders.
//
// If the loop count is known at build time and is 1, don't generate a loop.
// Just emit `body` once.
//
// These helpers either:
// - generate a runtime loop (creating the loop counter internally), or
// - inline `body` once when only one iteration is needed.
//
// This avoids repeating:
//   if (count > 1) { loop_begin; ...; loop_end; } else { ... }
// in every builder.

// Emit a runtime loop with `count_imm` as the iteration count.
// count_imm == 0 emits nothing, == 1 inlines `body` once, > 1 builds a loop.
template <typename body_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body) {
    if (count_imm == 0) return;
    if (count_imm == 1) {
        body();
        return;
    }
    const vreg_t counter = ir.new_gpr();
    const int begin = ir.loop_begin_imm(counter, count_imm);
    body();
    ir.loop_end(counter, begin);
}

// Like the helper above, but `step` runs after `body` on each loop iteration.
// A single inlined iteration (count_imm == 1) skips `step`. Use this for
// per-iteration pointer updates not needed for a single iteration.
template <typename body_t, typename step_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body, step_t step) {
    if (count_imm == 0) return;
    if (count_imm == 1) {
        body();
        return;
    }
    const vreg_t counter = ir.new_gpr();
    const int begin = ir.loop_begin_imm(counter, count_imm);
    body();
    step();
    ir.loop_end(counter, begin);
}

// Loop with a runtime iteration count held in `count_reg`. Use this when the
// iteration count is not known at IR generation time. A count of 0 or less
// skips the loop.
template <typename body_t>
void emit_loop_reg(ir_t &ir, vreg_t count_reg, body_t body) {
    const vreg_t counter = ir.new_gpr();
    // A negative count gets stuck.
    const int guard = ir.if_begin(count_reg, cmp_t::gt, 0);
    const int begin = ir.loop_begin_reg(counter, count_reg);
    body();
    ir.loop_end(counter, begin);
    ir.if_end(guard);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
