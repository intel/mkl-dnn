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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/utils.hpp"

#include "cpu/x64/ir/dump.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

namespace {

const char *kind_name(op_kind_t kind) {
    switch (kind) {
        case op_kind_t::mov_imm: return "mov_imm";
        case op_kind_t::mov_reg: return "mov_reg";
        case op_kind_t::add_imm: return "add_imm";
        case op_kind_t::add_reg: return "add_reg";
        case op_kind_t::load: return "load";
        case op_kind_t::vzero: return "vzero";
        case op_kind_t::vload: return "vload";
        case op_kind_t::vstore: return "vstore";
        case op_kind_t::vload_scalar: return "vload_scalar";
        case op_kind_t::vstore_scalar: return "vstore_scalar";
        case op_kind_t::vload_bcast: return "vload_bcast";
        case op_kind_t::vdot: return "vdot";
        case op_kind_t::vadd: return "vadd";
        case op_kind_t::vmul: return "vmul";
        case op_kind_t::vhreduce: return "vhreduce";
        case op_kind_t::inject_postops: return "inject_postops";
        case op_kind_t::set_mask_imm: return "set_mask_imm";
        case op_kind_t::vload_masked: return "vload_masked";
        case op_kind_t::vstore_masked: return "vstore_masked";
        case op_kind_t::prefetch: return "prefetch";
        case op_kind_t::tzero: return "tzero";
        case op_kind_t::tload: return "tload";
        case op_kind_t::tstore: return "tstore";
        case op_kind_t::loop_begin: return "loop_begin";
        case op_kind_t::loop_end: return "loop_end";
        case op_kind_t::if_begin: return "if_begin";
        case op_kind_t::_else: return "_else";
        case op_kind_t::if_end: return "if_end";
    }
    assert(!"unknown op kind");

    return "?";
}

// Returns `cmp` as the C++ operator it stands for.
const char *cmp_name(cmp_t cmp) {
    switch (cmp) {
        case cmp_t::eq: return "==";
        case cmp_t::ne: return "!=";
        case cmp_t::lt: return "<";
        case cmp_t::le: return "<=";
        case cmp_t::gt: return ">";
        case cmp_t::ge: return ">=";
    }
    assert(!"unknown comparison");

    return "?";
}

const char *reg_kind_name(reg_kind_t kind) {
    switch (kind) {
        case reg_kind_t::gpr: return "gpr";
        case reg_kind_t::vec: return "vec";
        case reg_kind_t::mask: return "mask";
        case reg_kind_t::tile: return "tile";
    }
    assert(!"unknown reg kind");

    return "?";
}

bool is_valid(const ir_t &ir, vreg_t v) {
    return (int)v >= 0 && (int)v < ir.n_vregs();
}

// Returns the name of the physical register `phys` for a vreg of kind `kind`.
std::string phys_str(cpu_isa_t isa, reg_kind_t kind, int phys) {
    const bool is_avx512 = is_superset(isa, avx512_core);

    switch (kind) {
        case reg_kind_t::gpr: return Xbyak::Reg64(phys).toString();
        case reg_kind_t::vec:
            return is_avx512 ? Xbyak::Zmm(phys).toString()
                             : Xbyak::Ymm(phys).toString();
        case reg_kind_t::mask:
            return is_avx512 ? Xbyak::Opmask(phys).toString()
                             : Xbyak::Ymm(phys).toString();
        case reg_kind_t::tile: return Xbyak::Tmm(phys).toString();
    }
    assert(!"unknown reg kind");

    return "?";
}

// Returns where `alloc` keeps `v`. It's either a physical register or a stack
// slot`[rsp+<off>]`. A spilled operand is followed by its temp: `(temp <reg>)`.
std::string location_str(const ir_t &ir, cpu_isa_t isa,
        const reg_alloc_result_t &alloc, vreg_t v,
        const std::vector<temp_reg_t> *temps) {
    const assignment_t &a = alloc.assignments[(int)v];
    const reg_kind_t kind = ir.vreg_info()[(int)v].kind;

    if (!a.spilled) return a.phys < 0 ? "?" : phys_str(isa, kind, a.phys);

    std::string slot = "[rsp+" + std::to_string(a.slot) + "]";
    if (!temps) return slot;

    for (const temp_reg_t &t : *temps) {
        if (t.vreg == v)
            return slot + "(temp " + phys_str(isa, kind, t.phys) + ")";
    }

    return slot + "(no temp)";
}

// Returns `g<id>`, `<dt>:v<id>`, `m<id>`, or `<dt>:t<id>` by the kind of `v`,
// followed by its location after `@`.
std::string vreg_str(const ir_t &ir, vreg_t v, cpu_isa_t isa,
        const reg_alloc_result_t &alloc, const std::vector<temp_reg_t> *temps) {
    if (v == vreg_t::none) return "none";

    const std::string id = std::to_string((int)v);
    if (!is_valid(ir, v)) return "?" + id;

    const std::string at = "@" + location_str(ir, isa, alloc, v, temps);

    switch (ir.vreg_info()[(int)v].kind) {
        case reg_kind_t::gpr: return "g" + id + at;
        case reg_kind_t::vec:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":v"
                    + id + at;
        case reg_kind_t::mask: return "m" + id + at;
        case reg_kind_t::tile:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":t"
                    + id + at;
    }
    assert(!"unknown reg kind");

    return "?" + id;
}

// Returns the name of register file `f`: the first kind whose files include
// it, followed by `<rows>x<colsb>` for a tile file. On AVX2, a mask is a vector
// register, so its file is named `vec`.
std::string file_name(const reg_pools_t &pools, int f) {
    for (int k = 0; k < (int)pools.kind_to_file.size(); k++) {
        const file_range_t &r = pools.kind_to_file[k];
        if (f < r.first || f >= r.first + r.n) continue;

        std::string kind = reg_kind_name((reg_kind_t)k);
        const tile_shape_t &shape = pools.files[f].tile_shape;
        if (shape == tile_shape_t {}) return kind;
        return kind + std::to_string(shape.rows) + "x"
                + std::to_string(shape.colsb);
    }
    assert(!"file_name: no kind allocates from the file");

    return "?";
}

// Returns the memory operand of `op`. `data` is the vec vreg the access moves,
// or `none` for an access that moves no vec value (e.g. `load`, `prefetch`).
std::string mem_str(const ir_t &ir, const op_t &op, vreg_t data, cpu_isa_t isa,
        const reg_alloc_result_t &alloc, const std::vector<temp_reg_t> *temps) {
    std::string s;
    if (data != vreg_t::none && op.mem_dt != data_type::undef) {
        s += dnnl_dt2str(op.mem_dt);
        s += ":";
    }
    s += "[";
    s += op.mem.is_param ? "param"
                         : vreg_str(ir, op.mem.base, isa, alloc, temps);
    if (op.mem.disp >= 0) s += "+";
    s += std::to_string(op.mem.disp);
    s += "]";

    return s;
}

// Returns `, <name>` for each flag of `hint`, or nothing when it has none.
std::string hint_str(mem_hint_t hint) {
    std::string s;
    if ((hint & mem_hint_t::non_temporal) != mem_hint_t::none)
        s += ", non_temporal";

    return s;
}

// Returns the operands of an `inject_postops` op from its side-table entry:
// the accumulators, then `base=` unless the base pointer is `none`, then
// `off=` unless the offsets are empty, then `n_elems=`.
std::string postops_str(const ir_t &ir, const op_t &op, cpu_isa_t isa,
        const reg_alloc_result_t &alloc, const std::vector<temp_reg_t> *temps) {
    const auto &table = ir.inject_postops_args();
    if (op.imm < 0 || op.imm >= (dim_t)table.size())
        return "args=?" + std::to_string(op.imm);

    const inject_postops_args_t &args = table[(int)op.imm];

    std::string s;
    for (size_t i = 0; i < args.acc.size(); i++) {
        if (i > 0) s += ", ";
        s += vreg_str(ir, args.acc[i], isa, alloc, temps);
    }

    if (args.base_ptr != vreg_t::none)
        s += ", base=" + vreg_str(ir, args.base_ptr, isa, alloc, temps);

    if (!args.out_byte_off.empty()) {
        s += ", off=[";
        for (size_t i = 0; i < args.out_byte_off.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(args.out_byte_off[i]);
        }
        s += "]";
    }

    s += ", n_elems=" + std::to_string(args.n_elems);

    return s;
}

// Returns the text of one operation, without index or indentation. Every kind
// prints as its name followed by its operands, except the control-flow
// markers. A loop or a conditional prints as an opening and a closing brace,
// and an `_else` prints as `} else {`. Each vreg shows its location.
std::string op_str(const ir_t &ir, const op_t &op, cpu_isa_t isa,
        const reg_alloc_result_t &alloc, const std::vector<temp_reg_t> *temps) {
    const auto r = [&](vreg_t v) { return vreg_str(ir, v, isa, alloc, temps); };
    const auto mem = [&](vreg_t data) {
        return mem_str(ir, op, data, isa, alloc, temps);
    };

    std::string k = std::string(kind_name(op.kind)) + " ";

    switch (op.kind) {
        case op_kind_t::mov_imm:
        case op_kind_t::add_imm:
        case op_kind_t::set_mask_imm:
            return k + r(op.dst) + ", " + std::to_string(op.imm);
        case op_kind_t::mov_reg:
        case op_kind_t::add_reg:
        case op_kind_t::vadd:
        case op_kind_t::vmul:
        case op_kind_t::vhreduce: return k + r(op.dst) + ", " + r(op.s0);
        case op_kind_t::load: return k + r(op.dst) + ", " + mem(vreg_t::none);
        case op_kind_t::vzero:
        case op_kind_t::tzero: return k + r(op.dst);
        case op_kind_t::vload:
        case op_kind_t::vload_scalar:
        case op_kind_t::vload_bcast: return k + r(op.dst) + ", " + mem(op.dst);
        case op_kind_t::vstore:
        case op_kind_t::vstore_scalar: return k + mem(op.s0) + ", " + r(op.s0);
        case op_kind_t::vdot:
            return k + r(op.dst) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::inject_postops:
            return k + postops_str(ir, op, isa, alloc, temps);
        case op_kind_t::vload_masked:
            return k + r(op.dst) + ", " + mem(op.dst) + ", " + r(op.s1);
        case op_kind_t::tload:
            return k + r(op.dst) + ", " + mem(op.dst) + ", " + r(op.s1)
                    + hint_str(op.hint);
        case op_kind_t::vstore_masked:
        case op_kind_t::tstore:
            return k + mem(op.s0) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::prefetch: return k + mem(vreg_t::none);
        case op_kind_t::loop_begin:
            return "loop " + r(op.dst) + " = "
                    + (op.init_is_reg ? r(op.s0) : std::to_string(op.imm))
                    + " {";
        case op_kind_t::loop_end:
            return "} // " + r(op.dst) + " -= 1, repeat while > 0";
        case op_kind_t::if_begin:
            return "if (" + r(op.s0) + " " + cmp_name(op.cmp) + " "
                    + (op.s1 != vreg_t::none ? r(op.s1)
                                             : std::to_string(op.imm))
                    + ") {";
        case op_kind_t::_else: return "} else {";
        case op_kind_t::if_end: return "}";
    }
    assert(!"unknown op kind");

    return k;
}

// Returns the maximum register pressure and the first operation where it
// occurs as `<n> at op <i>`, or `0` if the pressure is always 0.
std::string peak_str(const std::vector<int> &pressure) {
    const auto it = std::max_element(pressure.begin(), pressure.end());
    if (it == pressure.end() || *it == 0) return "0";

    return std::to_string(*it) + " at op "
            + std::to_string((int)(it - pressure.begin()));
}

// Returns the vregs that allocate from file `f`.
std::vector<int> file_vregs(const ir_t &ir, const reg_pools_t &pools, int f) {
    std::vector<int> vregs;
    for (int v = 0; v < ir.n_vregs(); v++)
        if (pools.file_of(ir.vreg_info()[v]) == f) vregs.push_back(v);

    return vregs;
}

// Returns the register pressure of each register file at each operation,
// indexed [file][op].
std::vector<std::vector<int>> count_pressure(
        const ir_t &ir, const reg_pools_t &pools) {
    std::vector<std::vector<int8_t>> live_in;
    compute_liveness(ir, live_in);

    std::vector<std::vector<int>> pressure(
            pools.files.size(), std::vector<int>(ir.n_ops(), 0));
    std::vector<int> defs, uses;

    for (int i = 0; i < ir.n_ops(); i++) {
        std::vector<int8_t> needs = live_in[i];
        ir.def_use(ir.ops()[i], defs, uses);

        for (int v : defs)
            needs[v] = 1;

        // A tile whose shape has no file is not counted.
        for (int v = 0; v < ir.n_vregs(); v++) {
            const int f = pools.file_of(ir.vreg_info()[v]);
            if (needs[v] && f >= 0) pressure[f][i]++;
        }
    }

    return pressure;
}

// Returns the lines of the register allocation, one line per register file.
// The peak is the largest register pressure of the file.
std::string alloc_lines(const ir_t &ir, const reg_pools_t &pools,
        const reg_alloc_result_t &alloc,
        const std::vector<std::vector<int>> &pressure) {

    auto n_spilled = [](const ir_t &ir, const reg_pools_t &pools,
                             const reg_alloc_result_t &alloc, int f) {
        int n = 0;
        for (int v : file_vregs(ir, pools, f)) {
            if (alloc.assignments[v].spilled) n++;
        }
        return n;
    };

    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++)
        ss << "alloc " << file_name(pools, f) << ": pool "
           << pools.files[f].regs.size() << ", peak " << peak_str(pressure[f])
           << ", spilled " << n_spilled(ir, pools, alloc, f) << "\n";

    return ss.str();
}

// Returns the indices of the operations that read or write `v`, for example,
// `1, 29`. For a spilled vreg, these are the operations that access its stack
// slot.
std::string accessing_ops_str(const ir_t &ir, int v) {
    const auto has_v = [&](const std::vector<int> &vregs) {
        return std::find(vregs.begin(), vregs.end(), v) != vregs.end();
    };

    std::string s;
    std::vector<int> defs, uses;
    for (int i = 0; i < ir.n_ops(); i++) {
        ir.def_use(ir.ops()[i], defs, uses);
        if (!has_v(defs) && !has_v(uses)) continue;
        if (!s.empty()) s += ", ";
        s += std::to_string(i);
    }

    return s;
}

// Returns one line per spilled vreg, grouped by register file, with the
// operations that access its stack slot.
std::string spill_lines(const ir_t &ir, cpu_isa_t isa, const reg_pools_t &pools,
        const reg_alloc_result_t &alloc) {
    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++)
        for (int v : file_vregs(ir, pools, f))
            if (alloc.assignments[v].spilled)
                ss << "spill " << vreg_str(ir, (vreg_t)v, isa, alloc, nullptr)
                   << ": ops " << accessing_ops_str(ir, v) << "\n";

    return ss.str();
}

// Returns the IR dump: one line per operation, with the location of each vreg
// in `alloc` and the register pressure `pressure` of each register file in
// `pools`.
std::string ir_dump(const ir_t &ir, cpu_isa_t isa, const reg_pools_t &pools,
        const reg_alloc_result_t &alloc,
        const std::vector<std::vector<int>> &pressure) {
    ostringstream_t ss;

    // A header line names the columns. The pressure column has one number per
    // register file.
    std::vector<std::string> names(pools.files.size());
    for (int f = 0; f < (int)pools.files.size(); f++)
        names[f] = file_name(pools, f);

    ss << "index |";
    for (const std::string &name : names)
        ss << " " << name;
    ss << " | operation\n";

    // Indentation follows the nesting of loops and conditionals. A closing
    // line, and the `} else {` line of a conditional, is indented like the
    // opening line, so each body stands out between them.
    int depth = 0;
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        if (utils::one_of(op.kind, op_kind_t::loop_end, op_kind_t::_else,
                    op_kind_t::if_end))
            depth--;

        ss << std::setw(5) << i << " | ";
        // Each number is right-aligned under the name of its file.
        for (int f = 0; f < (int)names.size(); f++)
            ss << std::setw((int)names[f].size()) << pressure[f][i] << " ";
        ss << "| " << std::string(2 * std::max(depth, 0), ' ')
           << op_str(ir, op, isa, alloc, &alloc.temps[i]) << "\n";

        if (utils::one_of(op.kind, op_kind_t::loop_begin, op_kind_t::if_begin,
                    op_kind_t::_else))
            depth++;
    }

    return ss.str();
}

} // namespace

bool has_x64ir_token(const std::string &verbose_value) {
    // Tokens are split on `,` with no trimming, the same way the rest of
    // `ONEDNN_VERBOSE` is parsed.
    size_t pos = 0;
    while (true) {
        const size_t end = verbose_value.find(',', pos);
        if (verbose_value.compare(pos, end - pos, "x64ir") == 0) return true;
        if (end == std::string::npos) return false;
        pos = end + 1;
    }
}

std::string kernel_dump_str(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data, const reg_config_t &reg_cfg,
        const reg_alloc_result_t &alloc) {
    ostringstream_t ss;

    const char *name = gen.name();
    cpu_isa_t isa = gen.max_cpu_isa();
    size_t code_size = gen.getSize();
    size_t data_size = code_size - data.begin_offset;
    const reg_pools_t &pools = reg_cfg.pools;
    const auto pressure = count_pressure(ir, pools);

    ss << "begin x64ir " << name << " isa=" << isa2str(isa) << "\n";
    ss << "code: " << code_size << " bytes (instructions "
       << code_size - data_size << " bytes, static data " << data_size
       << " bytes)\n";
    ss << alloc_lines(ir, pools, alloc, pressure);
    ss << spill_lines(ir, isa, pools, alloc);
    ss << "\n" << ir_dump(ir, isa, pools, alloc, pressure);
    ss << "end x64ir\n";

    return ss.str();
}

void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data, const reg_config_t &reg_cfg,
        const reg_alloc_result_t &alloc) {
    if (!is_dev_mode()) return;

    static const bool enabled = has_x64ir_token(getenv_string_user("VERBOSE"));
    if (!enabled) return;

    printf("%s", kernel_dump_str(gen, ir, data, reg_cfg, alloc).c_str());
    fflush(stdout);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
