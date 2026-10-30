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
#include <cmath>
#include <limits>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"

#include "cpu/x64/ir/dump.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {

using namespace impl;
using namespace impl::cpu::x64;
using namespace impl::cpu::x64::ir;

// Tests that require generating a kernel require AVX2.
#define SKIP_IF_NO_AVX2() \
    do { \
        if (!mayiuse(avx2)) GTEST_SKIP() << "IR emitter require AVX2"; \
    } while (0)

// Helpers shared by the tests

// The opmasks the post-ops injector writes without restoring. They are not
// part of the register pool.
constexpr int eltwise_opmask = 6, binary_tail_opmask = 7;

// ISA the kernel tests generate for. The emitter needs AVX2 as a floor, so the
// tests run on AVX-512 wherever the machine has it and fall back to AVX2
// otherwise.
cpu_isa_t test_isa() {
    return mayiuse(avx512_core) ? avx512_core : avx2;
}

// Vector width in f32 elements on `test_isa()`. The tests size their inputs
// from it.
int simd_w() {
    return isa_max_vlen(test_isa()) / (int)sizeof(float);
}

// Build a register pool for `test_isa()` with the GPR file narrowed to `n_gpr`
// registers. The vector and mask files come from `make_reg_config()`, so the
// allocator tests see the same layout the kernel tests do, with only the GPR
// pressure under the test's control.
reg_pools_t make_pools(int n_gpr) {
    reg_config_t rc = make_reg_config(test_isa(), /*param_reg=*/0,
            /*rsp_reg=*/Xbyak::Operand::RSP, /*reserved_masks=*/ {});

    reg_file_t &gpr_file = rc.pools.files[0];
    gpr_file.regs.clear();
    for (int i = 0; i < n_gpr; i++)
        gpr_file.regs.push_back(i);

    return rc.pools;
}

// A virtual register's (vreg) live interval. Defined as [start, end]
// operation indices in IR.
struct interval_t {
    int start = std::numeric_limits<int>::max();
    int end = -1;
    bool used() const { return end >= 0; }
};

// Reconstruct each vreg's live interval for branch-free IR.
//
// In linear code, a value is considered live from the first time it is
// used until the last time it is used. So, we define its live interval using
// the recorded reads and writes, without running liveness analysis again. This
// allows the allocator tests to check for interference using the same approach
// as the allocator itself.
std::vector<interval_t> linear_code_intervals(const ir_t &ir) {
    std::vector<interval_t> iv(ir.n_vregs());
    std::vector<int> defs, uses;

    for (int i = 0; i < ir.n_ops(); i++) {
        ir.def_use(ir.ops()[i], defs, uses);

        auto update_interval = [&](int v) {
            if (v < 0) return;
            iv[v].start = std::min(iv[v].start, i);
            iv[v].end = std::max(iv[v].end, i);
        };

        for (int v : defs)
            update_interval(v);
        for (int v : uses)
            update_interval(v);
    }
    return iv;
}

// Check the allocator's main rule. Two overlapping vregs targeting the same
// register file must not be assigned the same physical register.
// Spilled values are stored on the stack, so this rule does not apply to them.
void expect_no_reg_conflicts(const ir_t &ir, const reg_pools_t &pools,
        const reg_alloc_result_t &res) {
    const auto iv = linear_code_intervals(ir);
    const int n = ir.n_vregs();

    for (int a = 0; a < n; a++) {
        if (!iv[a].used()) continue;
        for (int b = a + 1; b < n; b++) {
            if (!iv[b].used()) continue;

            const int file_a = pools.kind_to_file[(int)ir.vreg_info()[a].kind];
            const int file_b = pools.kind_to_file[(int)ir.vreg_info()[b].kind];
            if (file_a != file_b) continue;

            const bool overlap
                    = iv[a].start <= iv[b].end && iv[b].start <= iv[a].end;
            if (!overlap) continue;

            const assignment_t &aa = res.assignments[a];
            const assignment_t &ab = res.assignments[b];

            if (!aa.spilled && !ab.spilled) {
                EXPECT_NE(aa.phys, ab.phys)
                        << "vregs " << a << " and " << b
                        << " overlap but share physical register " << aa.phys;
            }
        }
    }
}

// Check the allocator rule for spilled values. At every operation, the temps
// and the registers of the values live there are all distinct, and each
// spilled operand has exactly one temp.
void expect_valid_temps(const ir_t &ir, const reg_pools_t &pools,
        const reg_alloc_result_t &res) {
    const auto iv = linear_code_intervals(ir);
    std::vector<int> defs, uses;
    for (int i = 0; i < ir.n_ops(); i++) {
        std::set<std::pair<int, int>> held; // (file, physical register)
        auto hold = [&](int v, int phys) {
            const int file = pools.kind_to_file[(int)ir.vreg_info()[v].kind];
            EXPECT_TRUE(held.insert({file, phys}).second)
                    << "op " << i << ": register " << phys << " held twice";
        };

        std::set<int> spilled_operands, temp_vregs;
        ir.def_use(ir.ops()[i], defs, uses);
        for (int v : defs)
            if (res.assignments[v].spilled) spilled_operands.insert(v);
        for (int v : uses)
            if (res.assignments[v].spilled) spilled_operands.insert(v);
        for (const temp_reg_t &t : res.temps[i]) {
            temp_vregs.insert((int)t.vreg);
            hold((int)t.vreg, t.phys);
        }
        EXPECT_EQ(temp_vregs, spilled_operands) << "op " << i;
        EXPECT_EQ(temp_vregs.size(), res.temps[i].size()) << "op " << i;

        for (int v = 0; v < ir.n_vregs(); v++)
            if (iv[v].start <= i && i <= iv[v].end
                    && !res.assignments[v].spilled)
                hold(v, res.assignments[v].phys);
    }
}

// Build an IR where all GPRs in the live set are live at the same time.
// Each register is initialized first, then they are all used together by adding
// them into an accumulator. The live set size controls the register pressure to
// exercise register allocation and spilling.
ir_t build_gpr_live_set(int live_set_size) {
    ir_t ir;
    const vreg_t acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    std::vector<vreg_t> live_set(live_set_size, vreg_t::none);
    for (int i = 0; i < live_set_size; i++) {
        live_set[i] = ir.new_gpr();
        ir.mov_imm(live_set[i], i + 1);
    }
    for (int i = 0; i < live_set_size; i++)
        ir.add_reg(acc, live_set[i]);

    return ir;
}

// Find the index of the first operation with the given opcode. Otherwise
// return -1.
int find_op(const ir_t &ir, op_kind_t op) {
    for (int i = 0; i < ir.n_ops(); i++)
        if (ir.ops()[i].kind == op) return i;
    return -1;
}

// Reference dot product over `n` f32 elements.
float ref_dot(const float *a, const float *b, int n) {
    float acc = 0.f;
    for (int i = 0; i < n; i++)
        acc += a[i] * b[i];
    return acc;
}

// IR-based kernel.
class ir_kernel_t : public impl::cpu::x64::jit_generator_t {
public:
    ir_kernel_t(ir_t ir, int vec_regs_limit = -1, int gpr_regs_limit = -1,
            cpu_isa_t isa = test_isa())
        : jit_generator_t("ir_run_kernel", isa)
        , ir_(std::move(ir))
        , vec_regs_limit_(vec_regs_limit)
        , gpr_regs_limit_(gpr_regs_limit) {}

    const char *name() const override { return "ir_kernel"; }
    const char *source_file() const override { return __FILE__; }

    // Build and finalize the code. Return false on any Xbyak error.
    bool run_ir_pipeline() {
        generate();
        this->ready();
        return Xbyak::GetError() == Xbyak::ERR_NONE;
    }

    // Get generated code.
    const uint8_t *code_ptr() { return this->Xbyak::CodeGenerator::getCode(); }
    // Get generated code size.
    size_t code_size() { return this->Xbyak::CodeGenerator::getSize(); }

    // Calls the generated kernel.
    template <typename arg_t>
    void run(const arg_t *args) {
        auto fn = reinterpret_cast<void (*)(const arg_t *)>(
                const_cast<uint8_t *>(code_ptr()));
        fn(args);
    }

    // Optional post-ops injector setup. When `post_ops` is set, generate()
    // creates the JIT post-ops injector and drives the `inject_postops` op
    // through it, the same way a real IR kernel does.
    //   rhs_arg_offset  - byte offset in the argument struct of the binary
    //                     right-hand-side pointer array
    //   dst_orig_offset - byte offset in the argument struct of the
    //                     destination origin pointer
    //   tail_elems      - right-hand-side elements a partial (tail) load reads
    struct postops_cfg_t {
        const impl::post_ops_t *post_ops = nullptr;
        const impl::memory_desc_t *dst_md = nullptr;
        size_t rhs_arg_offset = 0;
        size_t dst_orig_offset = 0;
        int tail_elems = 0;
    };
    void set_postops(const postops_cfg_t &cfg) { postops_cfg_ = cfg; }

    // Replaces the gpr pool with `regs`. The allocator hands out a pool front
    // to back, so the first gprs the IR defines get them in order.
    void set_gpr_pool(const std::vector<int> &regs) { gpr_pool_ = regs; }

    // Allocation outcome. Becomes valid after `run_ir_pipeline()`.
    //
    // True if the generated kernel has any spills.
    bool spilled() const { return spilled_; }
    // The stack size required by the allocator for spilling.
    size_t stack_size() const { return stack_size_; }

protected:
    void generate() override {
        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        reg_config_t reg_cfg = make_reg_config(max_cpu_isa(), param_idx,
                rsp_idx, {eltwise_opmask, binary_tail_opmask});
        if (!gpr_pool_.empty()) reg_cfg.pools.files[0].regs = gpr_pool_;

        // Shrink the register pools to force spills when requested.
        const auto limit = [&](int file, int n) {
            auto &regs = reg_cfg.pools.files[file].regs;
            if (n >= 0 && (int)regs.size() > n) regs.resize(n);
        };
        limit(0, gpr_regs_limit_);
        limit(1, vec_regs_limit_);

        // Run allocator.
        const reg_alloc_result_t alloc = allocate_registers(ir_, reg_cfg.pools);
        spilled_ = alloc.any_spill;
        stack_size_ = alloc.frame_bytes;

        // Create the post-ops injector and the callback the emitter uses to
        // lower the `inject_postops` op, the same as a real IR kernel's
        // generate(). The injector saves and restores every register it borrows,
        // so it takes no part in the IR register allocation.
        std::unique_ptr<postops_injector_t> injector;
        if (postops_cfg_.post_ops) {
            injector.reset(new postops_injector_t(*this, max_cpu_isa(),
                    *postops_cfg_.post_ops, *postops_cfg_.dst_md,
                    default_strategies(), abi_param1,
                    static_cast<int>(postops_cfg_.rhs_arg_offset),
                    postops_cfg_.dst_orig_offset, postops_cfg_.tail_elems,
                    eltwise_opmask, binary_tail_opmask));
        }

        preamble();

        const int frame = (int)utils::rnd_up(alloc.frame_bytes, 16);
        if (frame > 0) sub(rsp, frame);

        data_section_t data;
        emit(*this, ir_, alloc, reg_cfg, data, injector.get());

        if (frame > 0) add(rsp, frame);

        postamble();

        emit_data_section(*this, data);

        // The injector's constant table follows the postamble. A no-op unless
        // the chain has an eltwise post-op.
        if (injector) injector->maybe_prepare_table();
    }

private:
    ir_t ir_ {};
    int vec_regs_limit_ = 0;
    int gpr_regs_limit_ = 0;
    std::vector<int> gpr_pool_;
    bool spilled_ = false;
    size_t stack_size_ = 0;
    postops_cfg_t postops_cfg_ {};
};

// IR builder tests
//
// Checks that the builder records an IR correctly. Operations stay in the right
// order. Each virtual register has the correct kind and data type. Every
// operation has the right inputs and outputs. The fused multiply add is an
// important example. It must treat its accumulator as both read and written or
// the system would incorrectly think a value is no longer needed even though it
// is still in use.
TEST(IRBuilderTests, OperationOrderMetadataAndDefUse) {
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    ir.load_param(ptr, 0);

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, ptr, 0, data_type::f32);

    const vreg_t b = ir.new_vec(data_type::f32);
    // AVX2 only.
    ir.vload(b, ptr, simd_w() * (dim_t)sizeof(float), data_type::f32);

    ir.vdot(acc, a, b);

    // Instructions appear in the exact order they were built.
    ASSERT_EQ(ir.n_ops(), 5);
    EXPECT_EQ(ir.ops()[0].kind, op_kind_t::load);
    EXPECT_EQ(ir.ops()[1].kind, op_kind_t::vzero);
    EXPECT_EQ(ir.ops()[2].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[3].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[4].kind, op_kind_t::vdot);

    // Register kinds and data type are recorded for the allocator and emitter.
    EXPECT_EQ(ir.vreg_info()[(int)ptr].kind, reg_kind_t::gpr);
    EXPECT_EQ(ir.vreg_info()[(int)ptr].dt, data_type::undef);

    EXPECT_EQ(ir.vreg_info()[(int)a].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)a].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[(int)b].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)b].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[(int)acc].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)acc].dt, data_type::f32);

    std::vector<int> defs, uses;

    // vzero writes its destination and reads nothing.
    ir.def_use(ir.ops()[1], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)acc}));
    EXPECT_TRUE(uses.empty());

    // vload reads the base pointer and writes the loaded vector.
    ir.def_use(ir.ops()[2], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)a}));
    EXPECT_EQ(uses, std::vector<int>({(int)ptr}));

    // vdot accumulates in place so the destination is read and written, both
    // sources are read.
    ir.def_use(ir.ops()[4], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)acc}));
    ASSERT_EQ(uses.size(), 3u);
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)a), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)b), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)acc), uses.end());
}

// Validates loop construction. A real loop links its end back to its begin and
// shares one counter register, while a loop that would run only once is inlined
// rather than emitted as a branch that is never taken.
TEST(IRBuilderTests, LoopLinkageAndSingleIterationInlining) {
    {
        ir_t ir;
        const vreg_t acc = ir.new_gpr();
        ir.mov_imm(acc, 0);
        emit_loop_imm(ir, 4, [&]() { ir.add_imm(acc, 1); });

        const int begin = find_op(ir, op_kind_t::loop_begin);
        const int end = find_op(ir, op_kind_t::loop_end);
        ASSERT_NE(begin, -1);
        ASSERT_NE(end, -1);

        // The back-edge is linked and the body sits between the two markers.
        EXPECT_EQ(ir.ops()[end].match, begin);
        EXPECT_LT(begin, end);
        // loop_begin and loop_end operate on the same counter register.
        EXPECT_EQ(ir.ops()[begin].dst, ir.ops()[end].dst);
        // The counter is a general-purpose register created for the loop.
        EXPECT_EQ(
                ir.vreg_info()[(int)ir.ops()[begin].dst].kind, reg_kind_t::gpr);
    }

    {
        // A single-iteration loop is inlined so no loop markers are emitted.
        ir_t ir;
        const vreg_t acc = ir.new_gpr();
        ir.mov_imm(acc, 0);
        emit_loop_imm(ir, 1, [&]() { ir.add_imm(acc, 1); });

        EXPECT_EQ(find_op(ir, op_kind_t::loop_begin), -1);
        EXPECT_EQ(find_op(ir, op_kind_t::loop_end), -1);
        EXPECT_EQ(ir.n_ops(), 2);
    }
}

// Validates that an if/else is well-formed. This proves the forward-edge
// control flow is constructed correctly.
TEST(IRBuilderTests, ForwardEdgeControlFlow) {
    ir_t ir;

    const vreg_t cond = ir.new_gpr();
    ir.load_param(cond, 0);

    const vreg_t a = ir.new_vec(data_type::f32);
    const vreg_t acc = ir.new_vec(data_type::f32);

    const vreg_t base = ir.new_gpr();

    ir.load_param(base, sizeof(int));
    ir.vload(a, base, 0, data_type::f32);

    // Build an IR for the following control flow:
    //
    //     if (cond != 0) { acc += a * a; }  // then block
    //     else           { acc = 0; }       // else block
    //
    // which lowers to a forward-branch skeleton:
    //
    //     test cond, cond
    //     je -> else           ; fall through to 'then' when cond != 0
    //   then:
    //     ...work...
    //     jmp -> end           ; skip the else block
    //   else:
    //     ...
    //   end:

    const int if_idx = ir.if_begin(cond, cmp_t::ne, 0);

    // then:
    ir.vzero(acc);
    ir.vdot(acc, a, a); // arbitrary work, just to populate the block

    // else:
    ir._else(if_idx);
    ir.vzero(acc);

    // end:
    ir.if_end(if_idx);

    const int else_idx = find_op(ir, op_kind_t::_else);
    const int end_idx = find_op(ir, op_kind_t::if_end);
    ASSERT_NE(else_idx, -1);
    ASSERT_NE(end_idx, -1);

    // The returned index is the if_begin, and the _else and the if_end link
    // back to it.
    EXPECT_EQ(ir.ops()[if_idx].kind, op_kind_t::if_begin);
    EXPECT_EQ(ir.ops()[else_idx].match, if_idx);
    EXPECT_EQ(ir.ops()[end_idx].match, if_idx);

    // The then block precedes the _else, which precedes the if_end.
    EXPECT_LT(if_idx, else_idx);
    EXPECT_LT(else_idx, end_idx);

    // The IR that contains branches can be handled by the allocator.
    // It assigns a location to every value that is read somewhere.
    const reg_pools_t pools = make_pools(/*n_gpr=*/8);
    const reg_alloc_result_t res = allocate_registers(ir, pools);
    // All virtual registers have been assigned.
    ASSERT_EQ((int)res.assignments.size(), ir.n_vregs());

    for (vreg_t v : {cond, base, a, acc}) {
        const assignment_t &as = res.assignments[(int)v];
        // Each value is either spilled or has a physical register assigned.
        EXPECT_TRUE(as.spilled || as.phys >= 0);
    }
}

#ifndef NDEBUG
// Validates that malformed loops and conditionals are rejected. The checks are
// asserts, so the test runs only in a debug build.
TEST(IRBuilderTests, RejectsMalformedControlFlow) {
    // A conditional that is never closed.
    {
        ir_t ir;
        ir.if_begin(ir.new_gpr(), cmp_t::ne, 0);
        EXPECT_DEATH(allocate_registers(ir, make_pools(/*n_gpr=*/8)),
                "if_begin without if_end");
    }
    // A second else.
    {
        ir_t ir;
        const int if_idx = ir.if_begin(ir.new_gpr(), cmp_t::ne, 0);
        ir._else(if_idx);
        EXPECT_DEATH(ir._else(if_idx), "already has an else");
    }
    // A conditional that overlaps another conditional.
    {
        ir_t ir;
        const int outer = ir.if_begin(ir.new_gpr(), cmp_t::ne, 0);
        ir.if_begin(ir.new_gpr(), cmp_t::ne, 0);
        EXPECT_DEATH(ir.if_end(outer), "not the innermost open conditional");
    }
    // A conditional that overlaps a loop.
    {
        ir_t ir;
        const vreg_t counter = ir.new_gpr();
        const int begin = ir.loop_begin_imm(counter, 2);
        ir.if_begin(ir.new_gpr(), cmp_t::ne, 0);
        EXPECT_DEATH(
                ir.loop_end(counter, begin), "not the innermost open loop");
    }
}
#endif

// Validates the inject_postops builder and its liveness. The operation stores
// its variable-length operands in a side table and keeps only the table index,
// so the test checks that the side table holds exactly what the builder was
// given. def_use must report each accumulator as read and written in place, and
// the base pointer as read, or liveness across the injected post-ops would be
// wrong.
TEST(IRBuilderTests, InjectPostopsRecordsArgsAndDefUse) {
    ir_t ir;

    const vreg_t base = ir.new_gpr();
    ir.load_param(base, 0);

    constexpr int n = 3;
    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);
    }

    std::vector<dim_t> out_byte_off(n);
    for (int r = 0; r < n; r++)
        out_byte_off[r] = r * (dim_t)sizeof(float);

    ir.inject_postops(acc, base, out_byte_off, /*n_elems=*/1);

    const int idx = find_op(ir, op_kind_t::inject_postops);
    ASSERT_NE(idx, -1);

    // The operation carries only an index into the side table, which holds the
    // full argument set.
    ASSERT_EQ(ir.inject_postops_args().size(), 1u);
    const auto &args = ir.inject_postops_args()[(int)ir.ops()[idx].imm];
    EXPECT_EQ(args.acc, acc);
    EXPECT_EQ(args.base_ptr, base);
    EXPECT_EQ(args.out_byte_off, out_byte_off);
    EXPECT_EQ(args.n_elems, 1);

    // def_use reports every accumulator as read and written, and the base
    // pointer as read.
    std::vector<int> defs, uses;
    ir.def_use(ir.ops()[idx], defs, uses);

    for (int r = 0; r < n; r++) {
        EXPECT_NE(std::find(defs.begin(), defs.end(), (int)acc[r]), defs.end())
                << "acc " << r << " not written";
        EXPECT_NE(std::find(uses.begin(), uses.end(), (int)acc[r]), uses.end())
                << "acc " << r << " not read";
    }
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)base), uses.end());
}

// Register config tests
//
// Checks the register files `make_reg_config()` builds for each ISA. A mask is
// a vector register on AVX2*, so the mask and vec kinds share one file. AVX-512
// masks with k-registers, so the mask kind gets a file of its own.
//
// `make_reg_config()` generates no code, so both layouts are checkable on any
// machine.
TEST(RegConfigTests, MapsMaskKindToFilePerIsa) {
    const int param_reg = 0, rsp_reg = 4;
    const std::vector<int> reserved_masks {6, 7};
    // The GPR file excludes `rsp` and `param_reg`, plus `rbp` where the build
    // reserves it (see `jit_generator_t::may_use_rbp()`).
    const size_t n_reserved_gprs = jit_generator_t::may_use_rbp() ? 2 : 3;

    {
        const reg_config_t rc
                = make_reg_config(avx2, param_reg, rsp_reg, reserved_masks);

        ASSERT_EQ(rc.pools.files.size(), 2u);
        EXPECT_EQ(rc.pools.kind_to_file, std::vector<int>({0, 1, 1}));
        EXPECT_EQ(rc.pools.files[1].slot_size, 32u);
        EXPECT_EQ(rc.pools.files[0].regs.size(), 16u - n_reserved_gprs);
        EXPECT_EQ(rc.pools.files[1].regs.size(), 16u);
    }

    {
        const reg_config_t rc = make_reg_config(
                avx512_core, param_reg, rsp_reg, reserved_masks);

        ASSERT_EQ(rc.pools.files.size(), 3u);
        EXPECT_EQ(rc.pools.kind_to_file, std::vector<int>({0, 1, 2}));
        EXPECT_EQ(rc.pools.files[1].slot_size, 64u);
        // `r16` to `r31` are added where the machine has Intel APX enabled.
        EXPECT_EQ(rc.pools.files[0].regs.size(),
                (mayiuse(avx10_2) ? 32u : 16u) - n_reserved_gprs);
        EXPECT_EQ(rc.pools.files[1].regs.size(), 32u);
        // The mask file is k1..k7 less the reserved ones. k0 cannot encode a
        // write mask.
        EXPECT_EQ(rc.pools.files[2].regs, std::vector<int>({1, 2, 3, 4, 5}));
    }
}

// Allocator tests
//
// Checks that there are no register collisions. If all values are live at the
// same time and there are enough registers, each value gets its own register
// and no values are spilled.
TEST(AllocatorTests, DoesNotDoubleAllocateSimultaneouslyLiveValues) {
    const int live_set_size = 6;
    // Create an IR that uses `live_set_size` GPRs for values plus 1 additional
    // GPR for an accumulator.
    const ir_t ir = build_gpr_live_set(live_set_size);

    // Create a pool for the exact number of required GPRs.
    const reg_pools_t pools = make_pools(/*n_gpr=*/live_set_size + 1);

    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.any_spill);
    EXPECT_EQ(res.frame_bytes, 0u);
    expect_no_reg_conflicts(ir, pools, res);
}

// Checks that registers are reused correctly. A temporary that is no longer
// needed gives up its register, which is then used by a later temporary.
// A value that is still in use keeps its own register.
TEST(AllocatorTests, ReusesRegisterOfExpiredTemporary) {
    ir_t ir;

    const vreg_t acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    const vreg_t t0 = ir.new_gpr();
    ir.mov_imm(t0, 5);
    ir.add_reg(acc, t0); // t0 dies here

    const vreg_t t1 = ir.new_gpr();
    ir.mov_imm(t1, 7);
    ir.add_reg(acc, t1); // t1 is used only after t0 is dead

    // Two registers are sufficient. One for the accumulator and one reused by
    // t0 then t1.
    const reg_pools_t pools = make_pools(/*n_gpr=*/2);
    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.any_spill);
    EXPECT_FALSE(res.assignments[(int)t0].spilled);
    EXPECT_FALSE(res.assignments[(int)t1].spilled);
    // t0 and t1 reuse the same physical register.
    EXPECT_EQ(res.assignments[(int)t0].phys, res.assignments[(int)t1].phys);
    // The accumulator, live across both, does not collide with them.
    EXPECT_NE(res.assignments[(int)acc].phys, res.assignments[(int)t0].phys);
}

// Checks allocator behavior under register pressure. If the pool doesn't have
// enough registers for all active values, the allocator spills some values to
// the stack. It also reserves frame space, keeps the remaining values from
// overlapping, and assigns each spilled value its own slot offset.
TEST(AllocatorTests, SpillsUnderRegisterPressure) {
    const int live_set_size = 6;
    // Create an IR that uses `live_set_size` GPRs for values plus 1 additional
    // GPR for an accumulator.
    const ir_t ir = build_gpr_live_set(live_set_size);

    // Create pool that has fewer registers than live values.
    const int gpr_pool_size = 4;
    const reg_pools_t pools = make_pools(/*n_gpr=*/gpr_pool_size);

    const reg_alloc_result_t res = allocate_registers(ir, pools);

    // Expect some spills and non-zero stack frame size.
    EXPECT_TRUE(res.any_spill);
    EXPECT_GT(res.frame_bytes, 0u);

    // Check that non-spilled registers do not share the same register, and that
    // every spilled operand gets a temp that clobbers nothing.
    expect_no_reg_conflicts(ir, pools, res);
    expect_valid_temps(ir, pools, res);

    // Collect spill slots and check that they are unique, slot-aligned, and
    // inside the reserved frame.
    std::vector<size_t> slots;
    for (const auto &as : res.assignments) {
        if (as.spilled) {
            // Check alignment.
            EXPECT_EQ(as.slot % pools.files[0].slot_size, 0u);
            // The slot is within the frame.
            EXPECT_LT(as.slot, res.frame_bytes);
            slots.push_back(as.slot);
        }
    }

    ASSERT_FALSE(slots.empty());

    // Check that each spilled register has a unique slot.
    std::sort(slots.begin(), slots.end());
    EXPECT_EQ(std::unique(slots.begin(), slots.end()), slots.end())
            << "two spilled values were given the same stack slot";
}

// Checks which value the allocator sends to the stack. A value referenced
// inside a loop weighs more than one referenced only outside it, so it keeps
// its register even when it is the longest-lived candidate. Values of equal
// weight fall back to the end of the interval.
TEST(AllocatorTests, SpillsByWeightAndBreaksTiesByEnd) {
    {
        ir_t ir;
        const vreg_t in_loop = ir.new_gpr();
        ir.mov_imm(in_loop, 1);
        const vreg_t out_of_loop = ir.new_gpr();
        ir.mov_imm(out_of_loop, 2);

        const vreg_t counter = ir.new_gpr();
        const int begin = ir.loop_begin_imm(counter, 4);
        ir.add_imm(in_loop, 1);
        ir.loop_end(counter, begin);

        ir.add_reg(in_loop, out_of_loop); // `out_of_loop` dies here
        ir.add_imm(in_loop, 1); // `in_loop` outlives every other value

        // Two registers for three values live at the loop, so one is spilled.
        const reg_alloc_result_t res
                = allocate_registers(ir, make_pools(/*n_gpr=*/2));

        // `in_loop` has the latest end, so choosing by end alone would pick it.
        EXPECT_TRUE(res.assignments[(int)out_of_loop].spilled);
        EXPECT_FALSE(res.assignments[(int)in_loop].spilled);
        EXPECT_FALSE(res.assignments[(int)counter].spilled);
    }

    {
        ir_t ir;
        // `early` and `late` are each defined once and used once, so they carry
        // the same weight. `late` is used last.
        const vreg_t early = ir.new_gpr();
        ir.mov_imm(early, 1);
        const vreg_t late = ir.new_gpr();
        ir.mov_imm(late, 2);
        const vreg_t acc = ir.new_gpr();
        ir.mov_imm(acc, 3);

        ir.add_reg(acc, early);
        ir.add_reg(acc, late);

        const reg_alloc_result_t res
                = allocate_registers(ir, make_pools(/*n_gpr=*/2));

        EXPECT_TRUE(res.assignments[(int)late].spilled);
        EXPECT_FALSE(res.assignments[(int)early].spilled);
        EXPECT_FALSE(res.assignments[(int)acc].spilled);
    }
}

// Checks that a mask spills like a vector on AVX2*, where it is a vector
// register. Masks share the vector file there, cut here to 3 registers. When
// the vectors run out of registers, the mask and the vectors each have one def
// and one use, so they weigh the same, and the mask ends last. The mask is
// therefore the one spilled, and it gets a temp at its def and its use.
TEST(AllocatorTests, SpillsMasksOnAvx2) {
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    ir.load_param(ptr, 0);

    const vreg_t mask = ir.new_mask();
    ir.set_mask_imm(mask, 3);

    std::vector<vreg_t> v(4, vreg_t::none);
    for (int r = 0; r < 4; r++) {
        v[r] = ir.new_vec(data_type::f32);
        ir.vload(v[r], ptr, 32 * r, data_type::f32);
    }
    for (int r = 0; r < 4; r++)
        ir.vstore(ptr, 32 * r, v[r], data_type::f32);

    const vreg_t x = ir.new_vec(data_type::f32);
    ir.vload_masked(x, ptr, 0, mask, data_type::f32);
    ir.vstore(ptr, 0, x, data_type::f32);

    reg_pools_t pools = make_reg_config(avx2, /*param_reg=*/0,
            /*rsp_reg=*/Xbyak::Operand::RSP, /*reserved_masks=*/ {})
                                .pools;
    pools.files[1].regs.resize(3);
    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_TRUE(res.assignments[(int)mask].spilled);
    expect_valid_temps(ir, pools, res);
}

// Checks that a k-register is never spilled on AVX-512. The mask file is cut
// to 1 register, and `m1` needs it while `m0` is still live. Spilling `m0`
// is required, but the file is not spillable, so `m0` keeps its register and
// `m1` is left without one.
TEST(AllocatorTests, NeverSpillsAvx512Masks) {
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    ir.load_param(ptr, 0);

    const vreg_t m0 = ir.new_mask();
    ir.set_mask_imm(m0, 3);
    const vreg_t m1 = ir.new_mask();
    ir.set_mask_imm(m1, 5);

    const vreg_t x = ir.new_vec(data_type::f32);
    ir.vload_masked(x, ptr, 0, m1, data_type::f32);
    ir.vload_masked(x, ptr, 0, m0, data_type::f32);
    ir.vstore(ptr, 0, x, data_type::f32);

    reg_pools_t pools = make_reg_config(avx512_core, /*param_reg=*/0,
            /*rsp_reg=*/Xbyak::Operand::RSP, /*reserved_masks=*/ {})
                                .pools;
    pools.files[2].regs.resize(1);
    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.assignments[(int)m0].spilled);
    EXPECT_TRUE(res.assignments[(int)m1].spilled);
}

// Checks that allocation depends only on its inputs. The same IR and register
// pool produces identical register assignments, spill decisions, and stack
// size, which makes the emitted code reproducible.
TEST(AllocatorTests, AllocatesDeterministically) {
    const ir_t ir = build_gpr_live_set(6);
    // Create smaller GPR register pool to put pressure.
    const reg_pools_t pools = make_pools(4);

    const reg_alloc_result_t a = allocate_registers(ir, pools);
    const reg_alloc_result_t b = allocate_registers(ir, pools);

    ASSERT_EQ(a.assignments.size(), b.assignments.size());
    EXPECT_EQ(a.frame_bytes, b.frame_bytes);
    EXPECT_EQ(a.any_spill, b.any_spill);

    for (size_t v = 0; v < a.assignments.size(); v++) {
        EXPECT_EQ(a.assignments[v].spilled, b.assignments[v].spilled);
        EXPECT_EQ(a.assignments[v].phys, b.assignments[v].phys);
        EXPECT_EQ(a.assignments[v].slot, b.assignments[v].slot);
    }
}

// Emitter tests
//
// Builds a small reduction IR (dot product of one vector pair) used by the
// emitter and integration tests.
ir_t build_dot_ir() {
    ir_t ir;

    // Load pointers for a, b, c.
    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, 2 * sizeof(void *));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0, data_type::f32);

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0, data_type::f32);

    ir.vdot(acc, a, b);

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);

    // store the reduced scalar
    ir.vstore_scalar(c_ptr, 0, acc, data_type::f32);

    return ir;
}

// Validates emitter determinism. Emitting the same program twice produces
// byte-for-byte identical machine code. The encoding is position independent
// (relative branches, rip-relative constants), so equal bytes is a valid check.
TEST(EmitterTests, EmitsDeterministicCodeForIdenticalIr) {
    SKIP_IF_NO_AVX2();
    ir_kernel_t k1(build_dot_ir());
    ir_kernel_t k2(build_dot_ir());

    ASSERT_TRUE(k1.run_ir_pipeline());
    ASSERT_TRUE(k2.run_ir_pipeline());

    ASSERT_GT(k1.code_size(), 0u);
    ASSERT_EQ(k1.code_size(), k2.code_size());
    EXPECT_EQ(0, std::memcmp(k1.code_ptr(), k2.code_ptr(), k1.code_size()));
}

// Validates the emitter's spill path. When the allocation spills, the
// reload-compute-store sequence must lower with no assembler error, and a stack
// frame must be reserved for the spilled values.
TEST(EmitterTests, EmitsValidCodeForSpilledAllocation) {
    SKIP_IF_NO_AVX2();

    // Six independent accumulators plus temporaries exceed a four-register
    // vector file, so the allocator must spill.
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0, data_type::f32);

    std::vector<vreg_t> acc(6, vreg_t::none);
    for (int r = 0; r < 6; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);
    }

    for (int r = 0; r < 6; r++) {
        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w() * (dim_t)sizeof(float), data_type::f32);
        ir.vdot(acc[r], a, b);
    }

    ir_kernel_t k(ir, /*vec_regs_limit=*/4);
    ASSERT_TRUE(k.run_ir_pipeline());
    EXPECT_TRUE(k.spilled());
    EXPECT_GT(k.stack_size(), 0u);
    EXPECT_GT(k.code_size(), 0u);
}

// Integration tests

// Arguments for the dot-product kernels.
struct dot_args_t {
    const float *a;
    const float *b;
    float *c;
};

// Pipeline test. A dot product over two vectors' worth of elements, expressed
// as a two-iteration loop, is built, allocated, emitted, run, and checked
// against a reference. Passing it means the whole pipeline computes the right
// number, including loop control flow and values kept live across the
// back-edge.
TEST(IntegrationTests, BuildsLoopReduction) {
    SKIP_IF_NO_AVX2();

    constexpr int k_blocks = 2;
    const int k = k_blocks * simd_w();

    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    // Reduce one simd_w-wide chunk per iteration and advance the pointers.
    emit_loop_imm(ir, k_blocks, [&]() {
        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, 0, data_type::f32);
        const vreg_t b = ir.new_vec(data_type::f32);
        ir.vload(b, b_ptr, 0, data_type::f32);
        ir.vdot(acc, a, b);
    }, [&]() {
        ir.add_imm(a_ptr, simd_w() * (dim_t)sizeof(float));
        ir.add_imm(b_ptr, simd_w() * (dim_t)sizeof(float));
    });

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);
    ir.vstore_scalar(c_ptr, 0, acc, data_type::f32);

    ir_kernel_t kernel(ir);
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(k), b(k);
    for (int i = 0; i < k; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(2 * i - 3);
    }

    float c = -12345.f;
    dot_args_t args {a.data(), b.data(), &c};
    kernel.run(&args);

    EXPECT_FLOAT_EQ(c, ref_dot(a.data(), b.data(), k));
}

// Validates masked access end to end. A mask covering `tail` elements is built
// once and drives both a load and a store. The inputs hold large values past
// `tail`, so a load that reads them or a store that writes them changes the
// result.
TEST(IntegrationTests, MaskedAccessCoversActiveElementsOnly) {
    SKIP_IF_NO_AVX2();

    const int tail = simd_w() - 3;
    const float sentinel = -1.f;
    const dim_t dot_off = simd_w() * (dim_t)sizeof(float);

    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t mask = ir.new_mask();
    ir.set_mask_imm(mask, tail);

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload_masked(a, a_ptr, 0, mask, data_type::f32);
    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload_masked(b, b_ptr, 0, mask, data_type::f32);

    // Reduce first. An inactive element has to read as zero, or the values
    // past `tail` would land in the dot product.
    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);
    ir.vdot(acc, a, b);

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);
    ir.vstore_scalar(c_ptr, dot_off, acc, data_type::f32);

    // Then the elementwise product, stored under the same mask.
    ir.vmul(a, b);
    ir.vstore_masked(c_ptr, 0, a, mask, data_type::f32);

    ir_kernel_t kernel(ir);
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a_buf(simd_w()), b_buf(simd_w());
    for (int i = 0; i < simd_w(); i++) {
        const bool active = i < tail;
        a_buf[i] = active ? (float)(i + 1) : 1e6f;
        b_buf[i] = active ? (float)(2 * i - 3) : 1e6f;
    }

    std::vector<float> c_buf(simd_w() + 1, sentinel);
    dot_args_t args {a_buf.data(), b_buf.data(), c_buf.data()};
    kernel.run(&args);

    EXPECT_FLOAT_EQ(c_buf[simd_w()], ref_dot(a_buf.data(), b_buf.data(), tail));
    for (int i = 0; i < tail; i++)
        EXPECT_FLOAT_EQ(c_buf[i], a_buf[i] * b_buf[i]) << " at element " << i;
    for (int i = tail; i < simd_w(); i++)
        EXPECT_FLOAT_EQ(c_buf[i], sentinel) << " at element " << i;
}

// Computes a dot product where one vector is multiplied by n vectors into
// n independent accumulators, each of which is reduced and stored.
ir_t build_shared_vector_dot_ir(int n) {
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t b = ir.new_vec(data_type::f32);
    // Load shared vector.
    ir.vload(b, b_ptr, 0, data_type::f32);

    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);

        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w() * (dim_t)sizeof(float), data_type::f32);

        ir.vdot(acc[r], a, b);
    }

    const vreg_t ws = ir.new_vec(data_type::f32);
    for (int r = 0; r < n; r++)
        ir.vhreduce(acc[r], ws);

    for (int r = 0; r < n; r++)
        ir.vstore_scalar(
                c_ptr, r * (dim_t)sizeof(float), acc[r], data_type::f32);

    return ir;
}

// Validates that register allocator decisions never change results. The same
// computation is run with full register files and with files only as large as
// its widest operations (3 vec for `vdot`, 1 gpr). Spilled values need no
// reserved registers, so both must produce identical results.
TEST(IntegrationTests, SpillProducesEquivalentResults) {
    SKIP_IF_NO_AVX2();

    constexpr int n = 6;

    ir_kernel_t full(build_shared_vector_dot_ir(n));
    ir_kernel_t limited(build_shared_vector_dot_ir(n), /*vec_regs_limit=*/3,
            /*gpr_regs_limit=*/1);

    ASSERT_TRUE(full.run_ir_pipeline());
    ASSERT_TRUE(limited.run_ir_pipeline());

    // The full file fits everything. The limited file must spill.
    EXPECT_FALSE(full.spilled());
    EXPECT_TRUE(limited.spilled());

    std::vector<float> a(n * simd_w()), b(simd_w());
    for (int i = 0; i < n * simd_w(); i++)
        a[i] = (float)(i % 7) - 3.f;

    for (int i = 0; i < simd_w(); i++)
        b[i] = (float)(i - 2);

    std::vector<float> c_full(n, 0.f), c_limited(n, 0.f);

    dot_args_t args_full {a.data(), b.data(), c_full.data()};
    dot_args_t args_limited {a.data(), b.data(), c_limited.data()};

    full.run(&args_full);
    limited.run(&args_limited);

    for (int r = 0; r < n; r++) {
        const float expected = ref_dot(&a[r * simd_w()], b.data(), simd_w());
        EXPECT_FLOAT_EQ(c_full[r], expected) << "row " << r;
        EXPECT_FLOAT_EQ(c_limited[r], expected) << "row " << r;
    }
}

// Arguments for the branch-selection kernel. A runtime flag plus two candidate
// input vectors and one output vector.
struct select_args_t {
    int64_t cond;
    const float *a;
    const float *b;
    float *c;
};

// Validates forward branches end to end. A runtime flag selects which of two
// inputs to store, and each flag value produces the expected output. This check
// that emitted control flow works and both candidates stayed live across the
// branching.
TEST(IntegrationTests, BranchSelectsCorrectValue) {
    SKIP_IF_NO_AVX2();

    ir_t ir;

    const vreg_t cond = ir.new_gpr();
    ir.load_param(cond, offsetof(select_args_t, cond));

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(select_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(select_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(select_args_t, c));

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0, data_type::f32);

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0, data_type::f32);

    //     if (cond != 0) { c = a; }  // then block
    //     else           { c = b; }  // else block
    //
    // which lowers to a forward-branch skeleton:
    //
    //     test cond, cond
    //     je -> else           ; fall through to 'then' when cond != 0
    //   then:
    //     store a -> c
    //     jmp -> end           ; skip the else block
    //   else:
    //     store b -> c
    //   end:
    const int if_idx = ir.if_begin(cond, cmp_t::ne, 0);
    ir.vstore(c_ptr, 0, a, data_type::f32); // then: c = a
    ir._else(if_idx);
    ir.vstore(c_ptr, 0, b, data_type::f32); // else: c = b
    ir.if_end(if_idx);

    ir_kernel_t kernel(ir);
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a_data(simd_w()), b_data(simd_w()),
            c_data(simd_w(), 0.f);
    for (int i = 0; i < simd_w(); i++) {
        a_data[i] = (float)(10 + i);
        b_data[i] = (float)(100 + i);
    }

    // cond != 0 selects a.
    select_args_t args_a {1, a_data.data(), b_data.data(), c_data.data()};
    kernel.run(&args_a);
    for (int i = 0; i < simd_w(); i++)
        EXPECT_FLOAT_EQ(c_data[i], a_data[i]) << "lane " << i;

    // cond == 0 selects b.
    std::fill(c_data.begin(), c_data.end(), 0.f);
    select_args_t args_b {0, a_data.data(), b_data.data(), c_data.data()};
    kernel.run(&args_b);
    for (int i = 0; i < simd_w(); i++)
        EXPECT_FLOAT_EQ(c_data[i], b_data[i]) << "lane " << i;
}

// Arguments for the conditional kernel. The branch that runs writes 0 to its
// own element of `out`.
struct cmp_args_t {
    int64_t lhs;
    int64_t rhs;
    float *out;
};

// Returns `a cmp b` computed in C++.
bool ref_cmp(cmp_t cmp, int64_t a, int64_t b) {
    switch (cmp) {
        case cmp_t::eq: return a == b;
        case cmp_t::ne: return a != b;
        case cmp_t::lt: return a < b;
        case cmp_t::le: return a <= b;
        case cmp_t::gt: return a > b;
        case cmp_t::ge: return a >= b;
    }
    return false;
}

// Validates every comparison and an else_if chain end to end. For each
// comparison the kernel runs
//
//     if (lhs cmp imm)      { out[0] = 0; }
//     else if (lhs cmp rhs) { out[1] = 0; }
//     else                  { out[2] = 0; }
//
// and must take the same branch as C++. The if compares with an immediate and
// the else_if with a gpr. A zero immediate lowers to `test` and a non-zero one
// to `cmp`, so both are used. The negative values check that the comparison is
// signed.
TEST(IntegrationTests, ConditionalTakesSameBranchAsCpp) {
    SKIP_IF_NO_AVX2();

    for (cmp_t cmp : {cmp_t::eq, cmp_t::ne, cmp_t::lt, cmp_t::le, cmp_t::gt,
                 cmp_t::ge}) {
        for (dim_t imm : {0, 1}) {
            ir_t ir;
            const vreg_t lhs = ir.new_gpr();
            ir.load_param(lhs, offsetof(cmp_args_t, lhs));

            const vreg_t rhs = ir.new_gpr();
            ir.load_param(rhs, offsetof(cmp_args_t, rhs));

            const vreg_t out_ptr = ir.new_gpr();
            ir.load_param(out_ptr, offsetof(cmp_args_t, out));

            const vreg_t zero = ir.new_vec(data_type::f32);
            ir.vzero(zero);

            const auto mark = [&](int branch) {
                ir.vstore_scalar(out_ptr, branch * (dim_t)sizeof(float), zero,
                        data_type::f32);
            };

            const int if_idx = ir.if_begin(lhs, cmp, imm);
            mark(0);
            ir.else_if(if_idx, lhs, cmp, rhs);
            mark(1);
            ir._else(if_idx);
            mark(2);
            ir.if_end(if_idx);

            ir_kernel_t kernel(ir);
            ASSERT_TRUE(kernel.run_ir_pipeline());

            for (int64_t l = -1; l <= 2; l++) {
                for (int64_t r = -1; r <= 2; r++) {
                    float out[3] = {-1.f, -1.f, -1.f};
                    cmp_args_t args {l, r, out};
                    kernel.run(&args);

                    const int taken = ref_cmp(cmp, l, imm) ? 0
                            : ref_cmp(cmp, l, r)           ? 1
                                                           : 2;
                    for (int b = 0; b < 3; b++) {
                        EXPECT_EQ(out[b], b == taken ? 0.f : -1.f)
                                << "cmp " << (int)cmp << ", imm " << imm
                                << ", lhs " << l << ", rhs " << r << ", branch "
                                << b;
                    }
                }
            }
        }
    }
}

// Arguments for the binary post-op kernel. `binary_rhs` points to the array of
// right-hand-side base pointers the injector indexes, and `dst_orig` is the
// destination origin it subtracts to locate each accumulator's output element.
struct binary_args_t {
    const float *a;
    const float *b;
    float *c;
    const void **binary_rhs;
    const void *dst_orig;
};

// Validates the binary post-op path end to end. It computes n independent dot
// products, adds a per-element right-hand-side through the JIT post-ops
// injector, and checks each result against a reference. This exercises the
// inject_postops op, the postops_injector_t driver, and the injector's
// destination-relative right-hand-side addressing with a one-element
// (scalar accumulator) tail load.
TEST(IntegrationTests, BinaryPostOpAddsPerElementRhs) {
    SKIP_IF_NO_AVX2();

    constexpr int n = 4;

    // Destination and right-hand-side share the same n x 1 shape, which selects
    // the injector's no-broadcast (per-element) strategy. The descriptors and
    // post-ops are built with the public API, then handed to the injector as the
    // internal types it takes.
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const dnnl::memory::desc dst_desc({n, 1}, dt::f32, tag::ab);
    const dnnl::memory::desc rhs_desc({n, 1}, dt::f32, tag::ab);

    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_add, rhs_desc);

    const impl::memory_desc_t &dst_md = *dst_desc.get();
    const impl::post_ops_t &post_ops = *po.get();

    // Build the dot-product IR that feeds the injector: one accumulator per
    // output element, each reduced to a scalar.
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(binary_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(binary_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(binary_args_t, c));

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0, data_type::f32);

    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);

        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w() * (dim_t)sizeof(float), data_type::f32);
        ir.vdot(acc[r], a, b);
    }

    const vreg_t ws = ir.new_vec(data_type::f32);
    for (int r = 0; r < n; r++)
        ir.vhreduce(acc[r], ws);

    // Each accumulator is a single scalar. Its output offset locates it in the
    // destination so the injector reads the matching right-hand-side element.
    std::vector<dim_t> out_byte_off(n);
    for (int r = 0; r < n; r++)
        out_byte_off[r] = r * (dim_t)sizeof(float);
    ir.inject_postops(acc, c_ptr, out_byte_off, /*n_elems=*/1);

    for (int r = 0; r < n; r++)
        ir.vstore_scalar(
                c_ptr, r * (dim_t)sizeof(float), acc[r], data_type::f32);

    ir_kernel_t kernel(ir);
    ir_kernel_t::postops_cfg_t cfg;
    cfg.post_ops = &post_ops;
    cfg.dst_md = &dst_md;
    cfg.rhs_arg_offset = offsetof(binary_args_t, binary_rhs);
    cfg.dst_orig_offset = offsetof(binary_args_t, dst_orig);
    cfg.tail_elems = 1;
    kernel.set_postops(cfg);

    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(n * simd_w()), b_data(simd_w()), rhs(n), c(n, 0.f);
    for (int i = 0; i < n * simd_w(); i++)
        a[i] = (float)(i % 5) - 2.f;
    for (int i = 0; i < simd_w(); i++)
        b_data[i] = (float)(i - 3);
    for (int r = 0; r < n; r++)
        rhs[r] = (float)(10 * (r + 1));

    const void *rhs_ptrs[1] = {rhs.data()};
    binary_args_t args {a.data(), b_data.data(), c.data(), rhs_ptrs, c.data()};
    kernel.run(&args);

    for (int r = 0; r < n; r++) {
        const float expected
                = ref_dot(&a[r * simd_w()], b_data.data(), simd_w()) + rhs[r];
        EXPECT_FLOAT_EQ(c[r], expected) << "row " << r;
    }
}

// Validates that the post-ops injector works with any register layout the
// allocator produces and leaves the other values intact. The two accumulators
// take the first and the last vec register, a live value fills every register
// between them, and the output pointer sits in r14, a gpr the binary injector
// borrows. The chain goes through the eltwise, binary, and sum injectors. A
// second run with a vec file of two spills the live values, so they stay in
// their stack slots while the injector runs and come back through temps. The
// accumulators outweigh them and keep their registers.
TEST(IntegrationTests, InjectPostopsWorksWithAnyRegisterLayout) {
    SKIP_IF_NO_AVX2();

    const int n_live = isa_num_vregs(test_isa()) - 2;
    const dim_t vlen = simd_w() * (dim_t)sizeof(float);
    const auto f32 = data_type::f32;

    // The allocator hands out vec registers in index order, so defining the
    // vectors in this order produces the layout.
    ir_t ir;
    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(binary_args_t, c));
    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(binary_args_t, a));
    const vreg_t live_ptr = ir.new_gpr();
    ir.load_param(live_ptr, offsetof(binary_args_t, b));

    std::vector<vreg_t> acc(2, vreg_t::none), live(n_live, vreg_t::none);
    acc[0] = ir.new_vec(f32);
    ir.vload(acc[0], a_ptr, 0, f32);
    for (int i = 0; i < n_live; i++) {
        live[i] = ir.new_vec(f32);
        ir.vload(live[i], live_ptr, i * vlen, f32);
    }
    acc[1] = ir.new_vec(f32);
    ir.vload(acc[1], a_ptr, vlen, f32);

    ir.inject_postops(acc, c_ptr, {0, vlen}, simd_w());
    for (int r = 0; r < 2; r++)
        ir.vstore(c_ptr, r * vlen, acc[r], f32);
    for (int i = 0; i < n_live; i++)
        ir.vstore(c_ptr, (2 + i) * vlen, live[i], f32);

    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const dnnl::memory::desc dst_desc({2, simd_w()}, dt::f32, tag::ab);
    const dnnl::memory::desc s8_desc({2, simd_w()}, dt::s8, tag::ab);
    dnnl::post_ops po;
    po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    po.append_binary(dnnl::algorithm::binary_ge, dst_desc);
    po.append_binary(dnnl::algorithm::binary_add, s8_desc);
    po.append_sum();

    ir_kernel_t::postops_cfg_t cfg;
    cfg.post_ops = po.get();
    cfg.dst_md = dst_desc.get();
    cfg.rhs_arg_offset = offsetof(binary_args_t, binary_rhs);
    cfg.dst_orig_offset = offsetof(binary_args_t, dst_orig);

    const int n = 2 * simd_w();
    std::vector<float> a(n), ge_rhs(n), live_data(n_live * simd_w());
    std::vector<int8_t> add_rhs(n);
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i % 9) * 0.5f - 2.f;
        ge_rhs[i] = i % 2 ? 0.5f : -0.5f;
        add_rhs[i] = (int8_t)(i % 7 - 3);
    }
    for (size_t i = 0; i < live_data.size(); i++)
        live_data[i] = (float)i;
    const void *rhs[2] = {ge_rhs.data(), add_rhs.data()};

    ir_kernel_t full(ir), spilling(ir, /*vec_regs_limit=*/2);
    for (ir_kernel_t *kern : {&full, &spilling}) {
        SCOPED_TRACE(kern == &full ? "full vec file" : "vec file of two");
        kern->set_gpr_pool({Xbyak::Operand::R14, Xbyak::Operand::RAX,
                Xbyak::Operand::RDX, Xbyak::Operand::RBX});
        kern->set_postops(cfg);
        ASSERT_TRUE(kern->run_ir_pipeline());

        // `c` starts at 1 in the accumulator rows, which the sum adds.
        std::vector<float> c(n + live_data.size(), 1.f);
        binary_args_t args {
                a.data(), live_data.data(), c.data(), rhs, c.data()};
        kern->run(&args);

        for (int i = 0; i < n; i++) {
            const float ref = (std::tanh(a[i]) >= ge_rhs[i] ? 1.f : 0.f)
                    + add_rhs[i] + 1.f;
            EXPECT_EQ(c[i], ref) << "element " << i;
        }
        for (size_t i = 0; i < live_data.size(); i++)
            EXPECT_EQ(c[n + i], live_data[i]) << "live value " << i;
    }
    EXPECT_FALSE(full.spilled());
    EXPECT_TRUE(spilling.spilled());
}

// Debug output tests
//
// The output is off unless the `x64ir` token asks for it. `all` and
// `debuginfo=` must not enable it.
TEST(DumpTests, OutputIsOffUnlessRequested) {
    EXPECT_FALSE(has_x64ir_token(""));
    EXPECT_FALSE(has_x64ir_token("all"));
    EXPECT_FALSE(has_x64ir_token("debuginfo=255"));
    EXPECT_FALSE(has_x64ir_token("dispatch,profile"));

    EXPECT_TRUE(has_x64ir_token("x64ir"));
    EXPECT_TRUE(has_x64ir_token("dispatch,x64ir,debuginfo=1"));
}

// Checks the whole output for a small IR with a loop and gpr, vec, and mask
// vregs, on AVX2 and AVX-512. The gpr pool has 2 registers, so 2 of the 3
// gprs spill, and the operations that access them show their temps.
TEST(DumpTests, KernelDumpShowsAllocation) {
    using namespace data_type;
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    const vreg_t step = ir.new_gpr();
    const vreg_t cnt = ir.new_gpr();
    const vreg_t x = ir.new_vec(f32);
    const vreg_t mask = ir.new_mask();
    ir.load_param(ptr, 0);
    ir.load_param(step, 8);
    ir.set_mask_imm(mask, 3);
    const int begin = ir.loop_begin_imm(cnt, 2);
    ir.vload_masked(x, ptr, 0, mask, f32);
    ir.add_reg(ptr, step);
    ir.loop_end(cnt, begin);

    const std::pair<cpu_isa_t, const char *> cases[] = {
            {avx2, R"(begin x64ir ir_kernel isa=avx2
code: 0 bytes (instructions 0 bytes, static data 0 bytes)
alloc gpr: pool 2, peak 3 at op 3, spilled 2
alloc vec: pool 16, peak 2 at op 4, spilled 0
spill g1@[rsp+0]: ops 1, 5
spill g2@[rsp+8]: ops 3, 6

index | gpr vec | operation
    0 |   1   0 | load g0@rax, [param+0]
    1 |   2   0 | load g1@[rsp+0](temp rcx), [param+8]
    2 |   2   1 | set_mask_imm m4@ymm0, 3
    3 |   3   1 | loop g2@[rsp+8](temp rcx) = 2 {
    4 |   3   2 |   vload_masked f32:v3@ymm1, f32:[g0@rax+0], m4@ymm0
    5 |   3   1 |   add_reg g0@rax, g1@[rsp+0](temp rcx)
    6 |   3   1 | } // g2@[rsp+8](temp rcx) -= 1, repeat while > 0
end x64ir
)"},
            {avx512_core, R"(begin x64ir ir_kernel isa=avx512_core
code: 0 bytes (instructions 0 bytes, static data 0 bytes)
alloc gpr: pool 2, peak 3 at op 3, spilled 2
alloc vec: pool 32, peak 1 at op 4, spilled 0
alloc mask: pool 7, peak 1 at op 2, spilled 0
spill g1@[rsp+0]: ops 1, 5
spill g2@[rsp+8]: ops 3, 6

index | gpr vec mask | operation
    0 |   1   0    0 | load g0@rax, [param+0]
    1 |   2   0    0 | load g1@[rsp+0](temp rcx), [param+8]
    2 |   2   0    1 | set_mask_imm m4@k1, 3
    3 |   3   0    1 | loop g2@[rsp+8](temp rcx) = 2 {
    4 |   3   1    1 |   vload_masked f32:v3@zmm0, f32:[g0@rax+0], m4@k1
    5 |   3   0    1 |   add_reg g0@rax, g1@[rsp+0](temp rcx)
    6 |   3   0    1 | } // g2@[rsp+8](temp rcx) -= 1, repeat while > 0
end x64ir
)"},
    };

    for (const auto &c : cases) {
        const ir_kernel_t gen(
                ir, /*vec_regs_limit=*/-1, /*gpr_regs_limit=*/-1, c.first);

        reg_config_t reg_cfg = make_reg_config(
                c.first, abi_param1.getIdx(), Xbyak::Operand::RSP, {});
        reg_cfg.pools.files[0].regs
                = {Xbyak::Operand::RAX, Xbyak::Operand::RCX};
        const reg_alloc_result_t alloc = allocate_registers(ir, reg_cfg.pools);

        EXPECT_EQ(kernel_dump_str(gen, ir, data_section_t(), reg_cfg, alloc),
                c.second);
    }
}

} // namespace dnnl
