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

#include <cassert>
#include <cstdint>

#include "common/z_magic.hpp"

#include "common/utils.hpp"

#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

vreg_t ir_t::new_vreg(reg_kind_t k, data_type_t dt) {
    vreg_info_t info {};
    info.kind = k;
    info.dt = dt;
    vreg_info_.push_back(info);
    return static_cast<vreg_t>((int)vreg_info_.size() - 1);
}

int ir_t::mov_imm(vreg_t dst, dim_t imm) {
    op_t op;
    op.kind = op_kind_t::mov_imm;
    op.dst = dst;
    op.imm = imm;
    ops_.push_back(op);
    return (int)ops_.size() - 1;
}

void ir_t::mov_reg(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::mov_reg;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::add_imm(vreg_t dst, dim_t imm) {
    op_t op;
    op.kind = op_kind_t::add_imm;
    op.dst = dst;
    op.imm = imm;
    ops_.push_back(op);
}

void ir_t::add_reg(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::add_reg;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::load_param(vreg_t dst, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::load;
    op.dst = dst;
    op.mem.is_param = true;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::load(vreg_t dst, vreg_t base, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::load;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::vzero(vreg_t dst) {
    op_t op;
    op.kind = op_kind_t::vzero;
    op.dst = dst;
    ops_.push_back(op);
}

void ir_t::vload(vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vload;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vstore(vreg_t base, dim_t disp, vreg_t src, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vstore;
    op.s0 = src;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vload_scalar(
        vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vload_scalar;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vstore_scalar(
        vreg_t base, dim_t disp, vreg_t src, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vstore_scalar;
    op.s0 = src;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vload_bcast(
        vreg_t dst, vreg_t base, dim_t disp, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vload_bcast;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vload_widen(
        vreg_t dst, vreg_t base, dim_t disp, int n_elems, data_type_t mem_dt) {
    op_t op;
    op.kind = op_kind_t::vload_widen;
    op.dst = dst;
    op.imm = n_elems;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vdot(vreg_t dst, vreg_t a, vreg_t b) {
    op_t op;
    op.kind = op_kind_t::vdot;
    op.dst = dst;
    op.s0 = a;
    op.s1 = b;
    ops_.push_back(op);
}

void ir_t::vadd(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vadd;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vsub(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vsub;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vmul(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vmul;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vdiv(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vdiv;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vmax(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vmax;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vcmp_ne_zero(vreg_t dst, vreg_t src) {
    assert(vreg_info_[(int)dst].kind == reg_kind_t::mask
            && "vcmp_ne_zero: dst must be a mask vreg");
    op_t op;
    op.kind = op_kind_t::vcmp_ne_zero;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vblend(vreg_t dst, vreg_t src, vreg_t mask) {
    op_t op;
    op.kind = op_kind_t::vblend;
    op.dst = dst;
    op.s0 = src;
    op.s1 = mask;
    ops_.push_back(op);
}

void ir_t::vbcast(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::vbcast;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::vbcast_imm(vreg_t dst, float value) {
    op_t op;
    op.kind = op_kind_t::vbcast_imm;
    op.dst = dst;
    op.imm = float2int(value);
    ops_.push_back(op);
}

void ir_t::vhreduce(vreg_t dst, vreg_t workspace) {
    assert(dst != workspace && "vhreduce: workspace must differ from dst");

    op_t op;
    op.kind = op_kind_t::vhreduce;
    op.dst = dst;
    op.s0 = workspace;
    ops_.push_back(op);
}

void ir_t::vhreduce_max(vreg_t dst, vreg_t workspace) {
    assert(dst != workspace && "vhreduce_max: workspace must differ from dst");

    op_t op;
    op.kind = op_kind_t::vhreduce_max;
    op.dst = dst;
    op.s0 = workspace;
    ops_.push_back(op);
}

void ir_t::veltwise(alg_kind_t alg, vreg_t dst) {
    op_t op;
    op.kind = op_kind_t::veltwise;
    op.dst = dst;
    op.imm = (dim_t)alg;
    ops_.push_back(op);
}

void ir_t::set_mask_imm(vreg_t mask, int n_elems) {
    op_t op;
    op.kind = op_kind_t::set_mask_imm;
    op.dst = mask;
    op.imm = n_elems;
    ops_.push_back(op);
}

void ir_t::vload_masked(
        vreg_t dst, vreg_t base, dim_t disp, vreg_t mask, data_type_t mem_dt) {
    assert(mask != vreg_t::none && "vload_masked: mask is required");

    op_t op;
    op.kind = op_kind_t::vload_masked;
    op.dst = dst;
    op.s1 = mask;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::vstore_masked(
        vreg_t base, dim_t disp, vreg_t src, vreg_t mask, data_type_t mem_dt) {
    assert(mask != vreg_t::none && "vstore_masked: mask is required");

    op_t op;
    op.kind = op_kind_t::vstore_masked;
    op.s0 = src;
    op.s1 = mask;
    op.mem.base = base;
    op.mem.disp = disp;
    op.mem_dt = mem_dt;
    ops_.push_back(op);
}

void ir_t::prefetch(vreg_t base, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::prefetch;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::inject_postops(const std::vector<vreg_t> &acc, vreg_t base_ptr,
        const std::vector<dim_t> &out_byte_off, int n_elems) {
    assert(n_elems > 0 && "inject_postops: no valid elements");
    inject_postops_args_t args;
    args.acc = acc;
    args.base_ptr = base_ptr;
    args.out_byte_off = out_byte_off;
    args.n_elems = n_elems;
    inject_postops_args_.push_back(args);

    op_t op;
    op.kind = op_kind_t::inject_postops;
    op.imm = (dim_t)inject_postops_args_.size() - 1;
    ops_.push_back(op);
}

int ir_t::loop_begin_imm(vreg_t counter, dim_t count) {
    op_t op;
    op.kind = op_kind_t::loop_begin;
    op.dst = counter;
    op.imm = count;
    ops_.push_back(op);
    const int idx = (int)ops_.size() - 1;
    open_regions_.push_back({idx, idx, false});
    return idx;
}

int ir_t::loop_begin_reg(vreg_t counter, vreg_t init) {
    op_t op;
    op.kind = op_kind_t::loop_begin;
    op.dst = counter;
    op.s0 = init;
    op.init_is_reg = true;
    ops_.push_back(op);
    const int idx = (int)ops_.size() - 1;
    open_regions_.push_back({idx, idx, false});
    return idx;
}

void ir_t::loop_end(vreg_t counter, int begin_idx) {
    assert(!open_regions_.empty() && open_regions_.back().head == begin_idx
            && ops_[begin_idx].kind == op_kind_t::loop_begin
            && "loop_end: not the innermost open loop");
    open_regions_.pop_back();

    op_t op;
    op.kind = op_kind_t::loop_end;
    op.dst = counter;
    op.match = begin_idx;
    ops_.push_back(op);
}

int ir_t::if_begin(vreg_t lhs, cmp_t cmp, dim_t imm) {
    // To compare with a constant wider than 32 bits, load it with `mov_imm`
    // and use the gpr overload.
    assert(imm == (int32_t)imm && "if_begin: immediate exceeds 32 bits");
    return append_if_begin(lhs, cmp, vreg_t::none, imm);
}

int ir_t::if_begin(vreg_t lhs, cmp_t cmp, vreg_t rhs) {
    assert(rhs != vreg_t::none && "if_begin: rhs is required");
    return append_if_begin(lhs, cmp, rhs, 0);
}

void ir_t::else_if(int if_idx, vreg_t lhs, cmp_t cmp, dim_t imm) {
    _else(if_idx);
    if_begin(lhs, cmp, imm);
    open_regions_.back().head = if_idx;
}

void ir_t::else_if(int if_idx, vreg_t lhs, cmp_t cmp, vreg_t rhs) {
    _else(if_idx);
    if_begin(lhs, cmp, rhs);
    open_regions_.back().head = if_idx;
}

void ir_t::_else(int if_idx) {
    assert_innermost_if(if_idx);
    open_region_t &top = open_regions_.back();
    assert(!top.has_else && "_else: the conditional already has an else");
    top.has_else = true;

    op_t op;
    op.kind = op_kind_t::_else;
    op.match = top.begin;
    ops_.push_back(op);
}

void ir_t::if_end(int if_idx) {
    assert_innermost_if(if_idx);
    // Close the ifs that `else_if` opened, innermost first, then `if_idx`.
    while (!open_regions_.empty() && open_regions_.back().head == if_idx) {
        op_t op;
        op.kind = op_kind_t::if_end;
        op.match = open_regions_.back().begin;
        ops_.push_back(op);
        open_regions_.pop_back();
    }
}

int ir_t::append_if_begin(vreg_t lhs, cmp_t cmp, vreg_t rhs, dim_t imm) {
    assert(lhs != vreg_t::none && vreg_info_[(int)lhs].kind == reg_kind_t::gpr
            && "if_begin: lhs must be a gpr");
    assert((rhs == vreg_t::none || vreg_info_[(int)rhs].kind == reg_kind_t::gpr)
            && "if_begin: rhs must be a gpr");

    op_t op;
    op.kind = op_kind_t::if_begin;
    op.s0 = lhs;
    op.s1 = rhs;
    op.imm = imm;
    op.cmp = cmp;
    ops_.push_back(op);
    const int idx = (int)ops_.size() - 1;
    open_regions_.push_back({idx, idx, false});
    return idx;
}

void ir_t::assert_innermost_if(int if_idx) const {
    assert(!open_regions_.empty() && open_regions_.back().head == if_idx
            && ops_[open_regions_.back().begin].kind == op_kind_t::if_begin
            && "else_if/_else/if_end: not the innermost open conditional");
    MAYBE_UNUSED(if_idx);
}

// For each operation, we record which virtual registers it reads (uses)
// and which ones it writes (defs). This info is the basis for liveness
// analysis, so it must accurately reflect what the operation really does.
//
// Some tricky cases:
// - Operations that both read and write the same register (like add_imm,
//   vdot) count as both a use and a def, because they read the old value
//   and then overwrite it.
// - vhreduce only writes its workspace (s0). The lowering overwrites the
//   workspace before reading it, so its old value is never needed.
// - A base register used for memory access counts as a read (use),
//   unless it's a fixed parameter register.
// - Control operations like loop_end both read and write the loop counter.
void ir_t::def_use(
        const op_t &op, std::vector<int> &defs, std::vector<int> &uses) const {
    defs.clear();
    uses.clear();

    auto u = [&](vreg_t v) {
        if (v != vreg_t::none) uses.push_back((int)v);
    };
    auto d = [&](vreg_t v) {
        if (v != vreg_t::none) defs.push_back((int)v);
    };

    switch (op.kind) {
        case op_kind_t::mov_imm: d(op.dst); break;
        case op_kind_t::mov_reg:
            d(op.dst);
            u(op.s0);
            break;
        case op_kind_t::add_imm: // read-modify-write
            u(op.dst);
            d(op.dst);
            break;
        case op_kind_t::add_reg:
            u(op.dst);
            u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::load:
            if (!op.mem.is_param) u(op.mem.base);
            d(op.dst);
            break;
        case op_kind_t::vzero: d(op.dst); break;
        case op_kind_t::vload:
        case op_kind_t::vload_scalar:
        case op_kind_t::vload_bcast:
        case op_kind_t::vload_widen:
            u(op.mem.base);
            d(op.dst);
            break;
        case op_kind_t::vstore:
        case op_kind_t::vstore_scalar:
            u(op.s0);
            u(op.mem.base);
            break;
        case op_kind_t::vdot:
            u(op.dst);
            u(op.s0);
            u(op.s1);
            d(op.dst);
            break;
        case op_kind_t::vadd: // read-modify-write
        case op_kind_t::vsub: // read-modify-write
        case op_kind_t::vmul: // read-modify-write
        case op_kind_t::vdiv: // read-modify-write
        case op_kind_t::vmax: // read-modify-write
            u(op.dst);
            u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::vblend: // dst = mask ? s0 : dst; reads dst, s0, mask
            u(op.dst);
            u(op.s0);
            u(op.s1);
            d(op.dst);
            break;
        case op_kind_t::vbcast: // overwrites dst, reads s0
        case op_kind_t::vcmp_ne_zero: // overwrites dst, reads s0
            u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::vbcast_imm: d(op.dst); break;
        case op_kind_t::vhreduce: // the workspace is only written
        case op_kind_t::vhreduce_max:
            u(op.dst);
            d(op.dst);
            d(op.s0);
            break;
        case op_kind_t::veltwise: // read-modify-write in place
            u(op.dst);
            d(op.dst);
            break;
        case op_kind_t::inject_postops: {
            // The injector transforms the accumulators in place (read and
            // written) and reads the base pointer. Report them so liveness is
            // correct across the op.
            const auto &args = inject_postops_args_[(int)op.imm];
            for (vreg_t v : args.acc) {
                u(v);
                d(v);
            }
            u(args.base_ptr);
            break;
        }
        case op_kind_t::set_mask_imm: d(op.dst); break;
        case op_kind_t::vload_masked:
            u(op.mem.base);
            u(op.s1); // mask
            d(op.dst);
            break;
        case op_kind_t::vstore_masked:
            u(op.s0);
            u(op.s1); // mask
            u(op.mem.base);
            break;
        case op_kind_t::prefetch: u(op.mem.base); break;
        case op_kind_t::loop_begin:
            if (op.init_is_reg) u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::loop_end:
            u(op.dst);
            d(op.dst);
            break;
        case op_kind_t::if_begin:
            u(op.s0);
            u(op.s1); // none for an immediate right-hand side
            break;
        case op_kind_t::_else:
        case op_kind_t::if_end: break;
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
