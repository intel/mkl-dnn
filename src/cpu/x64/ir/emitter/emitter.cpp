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

#include <vector>
#include <unordered_map>

#include "cpu/x64/ir/emitter/backend_avx2.hpp"
#include "cpu/x64/ir/emitter/backend_avx512.hpp"
#include "cpu/x64/ir/emitter/backend_tile.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename backend_t>
void emit(backend_t &be, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &rc, data_section_t &data,
        postops_injector_t *postops) {

    // The backend holds the generator.
    jit_generator_t &gen = be.gen();

    // Tile instructions go to a backend of their own, the same for both vector
    // backends.
    tile_backend_t tbe(gen);

    // One label per loop, keyed by the `loop_begin` instruction index. A
    // `loop_end` jumps back to `labels[op.match]`. A map, so only loops get an
    // entry.
    std::unordered_map<int, Xbyak::Label> labels;

    // Labels of one conditional, keyed by its `if_begin` instruction index.
    //   skip     - where a false condition jumps. Bound by `_else`, or by
    //              `if_end` when there is no else.
    //   end      - where the then body jumps over the else body. Bound by
    //              `if_end`.
    //   has_else - `_else` was lowered, so `if_end` binds `end`
    struct if_labels_t {
        Xbyak::Label skip, end;
        bool has_else = false;
    };
    std::unordered_map<int, if_labels_t> if_labels;

    // Jump to `target` unless `cmp` is true, skipping the then body that
    // follows. Each condition thus emits its inverse jump (`lt` -> `jge`).
    auto jump_unless = [&](cmp_t cmp, const Xbyak::Label &target) {
        const auto type = Xbyak::CodeGenerator::T_NEAR;
        switch (cmp) {
            case cmp_t::eq: gen.jne(target, type); break;
            case cmp_t::ne: gen.je(target, type); break;
            case cmp_t::lt: gen.jge(target, type); break;
            case cmp_t::le: gen.jg(target, type); break;
            case cmp_t::gt: gen.jle(target, type); break;
            case cmp_t::ge: gen.jl(target, type); break;
        }
    };

    // if vreg needs to be spilled
    auto spilled
            = [&](vreg_t vr) { return alloc.assignments[(int)vr].spilled; };
    // get a physical register from a virtual one
    auto phys = [&](vreg_t vr) { return alloc.assignments[(int)vr].phys; };
    // get a stack slot for a virtual register
    auto slot = [&](vreg_t vr) {
        return gen.ptr[gen.rsp + (int)alloc.assignments[(int)vr].slot];
    };
    // stack slot byte offset for a vec spill
    auto slot_off
            = [&](vreg_t vr) { return (int)alloc.assignments[(int)vr].slot; };
    // Data type of a vec vreg.
    auto dt_of = [&](vreg_t vr) { return ir.vreg_info()[(int)vr].dt; };

    // Temps of the operation being lowered (see `temp_reg_t`).
    const std::vector<temp_reg_t> *op_temps = nullptr;

    // Temp that holds the spilled `vr` during the current operation. There is
    // none when the operation exceeds `max_temps_per_op` or needs more
    // registers than the file holds. The kernel then cannot be emitted.
    auto temp_of = [&](vreg_t vr) -> int {
        for (const temp_reg_t &t : *op_temps)
            if (t.vreg == vr) return t.phys;
        JIT_ASSERT_RET(!"emit: spilled operand has no temp register", 0);
        return 0;
    };

    // Physical register that holds `vr` during the current operation, without
    // a reload. It is the temp when `vr` is spilled.
    auto reg_of
            = [&](vreg_t vr) { return spilled(vr) ? temp_of(vr) : phys(vr); };

    // Physical register of the tile `vr`. Tiles are never spilled. A tile has
    // no register when the tile configuration has no tile of its shape, or
    // when more tiles of that shape are live at once than the configuration
    // has. The kernel then cannot be emitted.
    auto tile_of = [&](vreg_t vr) -> int {
        JIT_ASSERT_RET(
                !spilled(vr) && phys(vr) >= 0 && "emit: tile has no register",
                0);
        return phys(vr);
    };

    // Move a spilled vec value between its stack slot and a register, as a
    // vector load/store against the stack frame (`rsp`).
    const int rsp_idx = gen.rsp.getIdx();
    auto spill_reload
            = [&](vreg_t vr, int p) { be.vload_raw(p, rsp_idx, slot_off(vr)); };
    auto spill_store = [&](vreg_t vr, int p) {
        be.vstore_raw(rsp_idx, slot_off(vr), p);
    };

    // Same for a spilled mask. The mask type depends on the ISA, so the backend
    // picks the appropriate instruction.
    auto mask_spill_reload = [&](vreg_t vr, int p) {
        be.load_mask_raw(p, rsp_idx, slot_off(vr));
    };
    auto mask_spill_store = [&](vreg_t vr, int p) {
        be.store_mask_raw(rsp_idx, slot_off(vr), p);
    };

    // Resolve a virtual register that an instruction READS (use) to a
    // concrete physical register, hiding whether the allocator spilled it:
    //   - not spilled: the value is already in a physical register, so just
    //     return that register (no extra instruction).
    //   - spilled: the value lives on the stack slot, so emit a reload into its
    //     temp and return the temp.
    // Each spilled operand of an operation has its own temp, so an instruction
    // with several spilled operands reloads each into a different register.
    // These helpers handle only reads. Writing a spilled result back is done
    // by the defining instruction (compute into the temp, then store to the
    // slot).
    //
    // gpr reloads are ISA-neutral (a plain `mov`), so `gpr_use` emits them
    // directly. A spilled vec or mask source is reloaded through the backend,
    // since the reload instruction is ISA-specific. The `vec_use` and
    // `mask_use` return a physical index rather than a typed register.
    auto gpr_use = [&](vreg_t vr) -> Xbyak::Reg64 {
        const Xbyak::Reg64 r(reg_of(vr));
        // reload the spilled gpr from its stack slot
        if (spilled(vr)) gen.mov(r, slot(vr));
        return r;
    };

    auto vec_use = [&](vreg_t vr) -> int {
        const int r = reg_of(vr);
        // reload the spilled vector register from its stack slot
        if (spilled(vr)) spill_reload(vr, r);
        return r;
    };

    auto mask_use = [&](vreg_t vr) -> int {
        const int r = reg_of(vr);
        // reload the spilled mask from its stack slot
        if (spilled(vr)) mask_spill_reload(vr, r);
        return r;
    };

    // Lower each IR instruction. Spilled operands are handled as follows:
    //
    // - Inputs that an instruction reads are accessed through
    //   gpr_use/vec_use/mask_use.
    //   These return the register directly, or reload the value from its spill
    //   slot into its temp if needed.
    //
    // - The output that an instruction writes is handled separately inside each
    //   case. If the destination is spilled, we reload it first (for
    //   read-modify-write operations), perform the operation in its temp, and
    //   then store the result back to its spill slot.
    //
    // The temps of an operation differ from each other and from every register
    // that holds a value live across the operation, so a reload never clobbers
    // another operand.

    // Fixed registers the injector reads are set up once, ahead of the IR,
    // rather than at every `inject_postops` operation. The pattern does not
    // change between operations, and an operation inside a loop would otherwise
    // rebuild it on every iteration.
    if (postops) postops->init(gen);

    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        op_temps = &alloc.temps[i];
        switch (op.kind) {
            // General-purpose register ops. ISA-neutral, emitted directly.
            case op_kind_t::mov_imm: {
                const Xbyak::Reg64 d(reg_of(op.dst));
                gen.mov(d, op.imm);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::mov_reg: {
                Xbyak::Reg64 s = gpr_use(op.s0);
                if (!spilled(op.dst))
                    gen.mov(Xbyak::Reg64(phys(op.dst)), s);
                else
                    gen.mov(slot(op.dst), s);
                break;
            }
            case op_kind_t::add_imm: {
                const Xbyak::Reg64 d = gpr_use(op.dst);
                gen.add(d, op.imm);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::add_reg: {
                const Xbyak::Reg64 s = gpr_use(op.s0);
                const Xbyak::Reg64 d = gpr_use(op.dst);
                gen.add(d, s);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }
            case op_kind_t::load: {
                Xbyak::Reg64 base = op.mem.is_param ? Xbyak::Reg64(rc.param_reg)
                                                    : gpr_use(op.mem.base);
                const Xbyak::Reg64 d(reg_of(op.dst));
                gen.mov(d, gen.ptr[base + (int)op.mem.disp]);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }

            // Vector ops. Emitting the instruction is the backend's job. A
            // spilled dst is always stored back to its slot after the op.
            // A read-modify-write (`rmw`) op also reloads a spilled dst before
            // the op. An op that overwrites dst does not.
            case op_kind_t::vzero: { // overwrites dst
                int d = reg_of(op.dst);
                be.vzero(d);
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vload: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                be.vstore(base, op.mem.disp, s, op.mem_dt, dt_of(op.s0));
                break;
            }
            case op_kind_t::vload_scalar: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload_scalar(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore_scalar: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                be.vstore_scalar(base, op.mem.disp, s, op.mem_dt, dt_of(op.s0));
                break;
            }
            case op_kind_t::vload_bcast: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                be.vload_bcast(d, base, op.mem.disp, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vdot: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int a = vec_use(op.s0);
                int b = vec_use(op.s1);
                be.vdot(d, a, b, dt_of(op.s0));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vadd: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int s = vec_use(op.s0);
                be.vadd(d, s, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vmul: { // rmw: reads and writes dst
                int d = vec_use(op.dst);
                int s = vec_use(op.s0);
                be.vmul(d, s, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vhreduce: { // reads and writes dst, overwrites ws
                int d = vec_use(op.dst);
                int ws = reg_of(op.s0);
                be.vhreduce(d, ws, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                if (spilled(op.s0)) spill_store(op.s0, ws);
                break;
            }

            // Lowers to the external JIT injector (see `postops_injector_t`).
            // Spilled operands go through their temps like those of any other
            // operation. The injector saves and restores every other register
            // it borrows.
            case op_kind_t::inject_postops: {
                const auto &args = ir.inject_postops_args()[(int)op.imm];
                std::vector<int> acc_phys;
                acc_phys.reserve(args.acc.size());
                for (vreg_t v : args.acc)
                    acc_phys.push_back(vec_use(v));
                // An eltwise-only chain has no base pointer.
                const int base_phys = args.base_ptr == vreg_t::none
                        ? -1
                        : gpr_use(args.base_ptr).getIdx();
                JIT_ASSERT(postops && "inject_postops: missing injector");
                postops->inject(
                        acc_phys, base_phys, args.out_byte_off, args.n_elems);
                for (size_t a = 0; a < args.acc.size(); a++)
                    if (spilled(args.acc[a]))
                        spill_store(args.acc[a], acc_phys[a]);
                break;
            }

            // Mask ops. Emitting the instruction is the backend's job,
            // including the reload and store of a spilled mask. Only AVX2*
            // spills masks. The AVX-512 backend rejects a spilled k-register.
            case op_kind_t::set_mask_imm: { // overwrites dst
                int d = reg_of(op.dst);
                be.set_mask_imm(d, (int)op.imm, data);
                if (spilled(op.dst)) mask_spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vload_masked: { // overwrites dst
                int base = gpr_use(op.mem.base).getIdx();
                int d = reg_of(op.dst);
                int m = mask_use(op.s1);
                be.vload_masked(
                        d, base, op.mem.disp, m, op.mem_dt, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore_masked: {
                int base = gpr_use(op.mem.base).getIdx();
                int s = vec_use(op.s0);
                int m = mask_use(op.s1);
                be.vstore_masked(
                        base, op.mem.disp, s, m, op.mem_dt, dt_of(op.s0));
                break;
            }

            // prefetcht0 is base x86-64, so emit it directly.
            case op_kind_t::prefetch: {
                Xbyak::Reg64 base = gpr_use(op.mem.base);
                gen.prefetcht0(gen.ptr[base + (int)op.mem.disp]);
                break;
            }

            // Tile ops. Emitting the instruction is the tile backend's job.
            case op_kind_t::tzero: tbe.tzero(tile_of(op.dst)); break;
            case op_kind_t::tload: {
                int base = gpr_use(op.mem.base).getIdx();
                int stride = gpr_use(op.s1).getIdx();
                tbe.tload(tile_of(op.dst), base, op.mem.disp, stride, op.hint);
                break;
            }
            case op_kind_t::tstore: {
                int base = gpr_use(op.mem.base).getIdx();
                int stride = gpr_use(op.s1).getIdx();
                tbe.tstore(base, op.mem.disp, tile_of(op.s0), stride);
                break;
            }

            // Control flow. ISA-neutral, emitted directly.
            case op_kind_t::loop_begin: {
                const Xbyak::Reg64 c(reg_of(op.dst));
                if (op.init_is_reg) {
                    Xbyak::Reg64 iv = gpr_use(op.s0);
                    gen.mov(c, iv);
                } else {
                    gen.mov(c, op.imm);
                }
                if (spilled(op.dst)) gen.mov(slot(op.dst), c);
                gen.L(labels[i]); // body start
                break;
            }
            case op_kind_t::loop_end: {
                // dec sets ZF, so the back-edge is a plain jnz with no cmp. The
                // counter starts >= 1 and lands on exactly 0, so jnz matches jg.
                const Xbyak::Reg64 c = gpr_use(op.dst);
                gen.dec(c);
                if (spilled(op.dst)) gen.mov(slot(op.dst), c);
                gen.jnz(labels[op.match]); // back-edge to the matching loop_begin
                break;
            }
            case op_kind_t::if_begin: {
                // `test lhs, lhs` sets the same flags as `cmp lhs, 0` with a
                // shorter encoding.
                const Xbyak::Reg64 lhs = gpr_use(op.s0);
                if (op.s1 != vreg_t::none)
                    gen.cmp(lhs, gpr_use(op.s1));
                else if (op.imm == 0)
                    gen.test(lhs, lhs);
                else
                    gen.cmp(lhs, (int)op.imm);
                jump_unless(op.cmp, if_labels[i].skip);
                break;
            }
            case op_kind_t::_else: {
                if_labels_t &l = if_labels[op.match];
                gen.jmp(l.end, Xbyak::CodeGenerator::T_NEAR);
                gen.L(l.skip);
                l.has_else = true;
                break;
            }
            case op_kind_t::if_end: {
                if_labels_t &l = if_labels[op.match];
                gen.L(l.has_else ? l.end : l.skip);
                break;
            }
        }
    }
}

void emit(jit_generator_t &gen, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &reg_cfg, data_section_t &data,
        postops_injector_t *postops) {
    const cpu_isa_t isa = gen.max_cpu_isa();
    if (is_superset(isa, avx512_core)) {
        avx512_backend_t be(gen, isa);
        emit(be, ir, alloc, reg_cfg, data, postops);
    } else {
        avx2_backend_t be(gen, isa);
        emit(be, ir, alloc, reg_cfg, data, postops);
    }
}

void emit_data_section(jit_generator_t &gen, data_section_t &data) {
    data.begin_offset = gen.getSize();
    for (auto &c : data.constants) {
        gen.align(data_section_t::alignment);
        gen.L(c.second);
        for (unsigned char byte : c.first)
            gen.db(byte);
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
