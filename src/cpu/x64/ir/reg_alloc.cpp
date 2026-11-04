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

// The register allocator.
//
// Turns the IR's unlimited virtual registers into the CPU's real ones, spilling
// to the stack when more values are live at once than there are registers. It
// replaces the original brgemm kernel's hand-managed register numbering. The
// allocator is responsible for deciding WHO to spill, which relies on three
// ideas:
//   1. liveness
//   2. control-flow graph
//   3. linear scan
//
// It is designed to be generic. It knows only register kinds and control
// flow, nothing about the computation (brgemm, brgemv, copy, etc.) or the ISA.
//
// 1. Liveness
//
// A value is `live` at a point if it will be read again before being
// overwritten. Two values that are live at the same point interfere. They
// cannot share a register. So the allocator's input is each value's live span,
// and its job is to pack non-interfering spans onto the same register.
//
// 2. The Control-Flow Graph
//
// To know what is read later we must know what runs after what. In this IR
// that is nearly trivial. Each operation flows to the next, with three
// exceptions:
//   1. loop_end has two successors. It can fall through, or jump back to the
//      body start (`back-edge`).
//   2. if_begin has two successors. It can fall through into the then body,
//      or jump past it to the else body or out of the conditional
//      (`forward-edge`).
//   3. _else has one successor. The end of the then body jumps over the else
//      body to the matching if_end (`forward-edge`).
//
//     Back-edge example:
//
//     0  loop_begin      <- back-edge returns here
//     1    v = load [ptr]
//     2    acc += v * x
//     3    ptr += stride
//     4  loop_end        -> 5, or jump back to 0
//     5  store acc
//
// So both branches and the back-edge depart from a straight list. The back-edge
// is special because it forms a cycle. A cycle is the one thing that forces the
// liveness pass to repeat, as the next section explains. Forward branches do
// not. There are no basic blocks. Every operation is a node.
//
// Backward Liveness Analysis
//
// Liveness is computed backwards and iterated. Needed here depends on uses
// later, so we go through the list back to front. Each operation keeps alive
// whatever its successors need, minus what it overwrites, plus what it reads.
//
// The back-edge makes this take more than one pass. Iterate to a fixed point
// is a static analysis of the program text. We don't need to know the loop
// counter values. We re-iterate over the operation list until a pass changes
// nothing. That stable result is the `fixed point`.
//
// Why a second pass is ever needed: a `ptr` read at the top of the body
// (see the back-edge example) and advanced at the bottom is, through the
// back-edge, the value the next iteration reads at the top. So it must be live
// across the whole body. The first pass does not yet know the loop start needs
// it. The next pass propagates that, and it appears. This settles in a handful
// of passes. The count scales with loop nesting depth, not with how many times
// the loop runs. The result is liveness valid for every possible execution at
// once.
//
// 3. Linear Scan
//
// Collapse each value's liveness to one [start, end] interval, ignoring holes.
// That simplification is what keeps this cheap. If we need, we can enable
// intervals split to improve register allocation under pressure.
//
//     v0 |=================|   (0..9)
//     v1   |===|               (1..3)
//     v2         |=======|     (4..8)
//     v3           |=====|     (5..8)
//        0 1 2 3 4 5 6 7 8 9
//
// Per register file, the scan goes through the operations left to right,
// keeping the intervals that hold a register (`active`). A register is freed
// when its interval ends. When none is free, one of the overlapping intervals
// has to go to the stack. In the picture, with two registers, at t=5 both are
// taken by v0 and v2, so one of them is spilled to make room for v3.
//
// A spilled value still has to be in a register while an operation reads or
// writes it. The allocator hands out such a register for the duration of that
// one operation, called a `temp` (see `temp_reg_t`). The scan maintains this
// rule:
//
//   Rule: at every operation, each operand is in a register, its own or a
//   temp. Every other live value holds a register or sits in its stack slot.
//
// At t=5 the operation defines v3, so v3 is one of its operands. By the rule
// v3 needs a register at 5 either way, its own or a temp. So spilling v3 frees
// nothing, and the choice is between v0 and v2. In general, only a value the
// operation does not touch frees a register by going to the stack.
//
// 4. Spill Weights
//
// Which one to spill is decided by weight. The weight of a value estimates how
// much code it costs to keep that value on the stack instead of in a register.
//
// Every operation that reads or writes a value adds to that value's weight. How
// much it adds depends on how many loops enclose the operation, because the
// operation runs once per iteration of every loop around it. The iteration
// counts are run-time values, so the allocator assumes `loop_weight` iterations
// for every loop. An operation outside every loop then adds 1, one loop deep
// adds `loop_weight`, and two loops deep adds `loop_weight` squared.
//
// `loop_weight` is 10. It does not model the real iteration count. It only has
// to keep a value that the loop nest references ahead of one referenced only
// outside the nest. At 10 it takes ten references outside the nest to match a
// single reference inside it, so a value can be referenced several times
// outside and still rank below a value the nest touches once. 10 is a starting
// point that looks reasonable for that, and it can be adjusted as needed.
//
// Below is a kernel that walks a block row by row, with `m` counting the rows
// and `n` the elements in a row. Both counters are ordinary values that occupy
// registers. The three columns are the operation's loop depth, what it adds
// with `loop_weight` at 10, and the values it reads or writes:
//
//                                       depth  adds  to
//     0  load_param ptr_a                  0      1  ptr_a
//     1  loop_begin m                      0      1  m
//     2    load_param ptr_b                1     10  ptr_b
//     3    loop_begin n                    1     10  n
//     4      v = load [ptr_b]              2    100  ptr_b, v
//     5      store [ptr_a], v              2    100  ptr_a, v
//     6    loop_end n                      2    100  n
//     7    ptr_a += stride                 1     10  ptr_a
//     8  loop_end m                        1     10  m
//
// A `loop_begin` sets its counter once on entry, so it sits outside its own
// loop. A `loop_end` tests the counter on every turn, so it sits inside. That
// is why rows 1 and 8 differ in depth.
//
// The totals are `v` at 200, `ptr_a` at 111, `ptr_b` at 110, `n` at 110, and
// `m` at 11. The interval with the smallest total is the one spilled, so `m`
// goes first. It is the only value here that is never touched two loops deep.
//
// Weight counts references, the reads and writes of a value. It does not count
// live range. A pointer that is live across a loop but never touched inside it
// adds nothing for that loop. A pointer read by every store in the loop body
// adds once per store. Both have the same live range, and only the reference
// count separates them.
//
// When two values have the same weight, the end of the live interval decides.
// The value whose interval ends last is spilled, because among the tied values
// it is the one that would otherwise occupy a register the longest.

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>

#include "cpu/x64/ir/reg_alloc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

void compute_liveness(
        const ir_t &ir, std::vector<std::vector<int8_t>> &live_in) {
    const int n_ops = ir.n_ops();
    const int n_vregs = ir.n_vregs();

    // Phase 1: build per-operation def/use sets and the control-flow graph.
    //
    // def_at[i][v] is 1 if operation `i` defines (writes) vreg `v`.
    // use_at[i][v] is 1 if operation `i` uses (reads) vreg `v`.
    // successors[i] lists all operations that may execute immediately after i.
    std::vector<std::vector<int8_t>> def_at(
            n_ops, std::vector<int8_t>(n_vregs, 0));
    std::vector<std::vector<int8_t>> use_at(
            n_ops, std::vector<int8_t>(n_vregs, 0));

    std::vector<std::vector<int>> successors(n_ops);
    std::vector<int> def_vregs, use_vregs;

    // The _else and if_end of each if_begin, by op index. -1 for a conditional
    // without an else, and at ops that are not an if_begin.
    std::vector<int> else_of(n_ops, -1), end_of(n_ops, -1);
    for (int i = 0; i < n_ops; i++) {
        const op_t &op = ir.ops()[i];
        if (op.kind == op_kind_t::_else) else_of[op.match] = i;
        if (op.kind == op_kind_t::if_end) end_of[op.match] = i;
    }

    for (int i = 0; i < n_ops; i++) {
        const op_t &op = ir.ops()[i];
        ir.def_use(op, def_vregs, use_vregs);

        for (int v : def_vregs)
            def_at[i][v] = 1;
        for (int v : use_vregs)
            use_at[i][v] = 1;

        // Successor edges include fall-through to `i+1` and any explicit
        // control-flow transfers (branches and loop back-edges).
        if (op.kind == op_kind_t::loop_end) {
            if (i + 1 < n_ops) successors[i].push_back(i + 1);
            successors[i].push_back(op.match + 1);
        } else if (op.kind == op_kind_t::if_begin) {
            assert(end_of[i] >= 0 && "if_begin without if_end");
            // A false condition continues at the else body, or at if_end.
            successors[i].push_back(i + 1);
            successors[i].push_back(
                    else_of[i] >= 0 ? else_of[i] + 1 : end_of[i]);
        } else if (op.kind == op_kind_t::_else) {
            successors[i].push_back(end_of[op.match]);
        } else if (i + 1 < n_ops) {
            successors[i].push_back(i + 1);
        }
    }

    // Phase 2: compute `live_in` and `live_out` using backward dataflow
    // analysis. Iterate until reaching a fixed point (no changes in a full
    // pass).
    live_in.assign(n_ops, std::vector<int8_t>(n_vregs, 0));
    std::vector<std::vector<int8_t>> live_out(
            n_ops, std::vector<int8_t>(n_vregs, 0));

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = n_ops - 1; i >= 0; i--) {
            // live_out[i] = OR over successors `s` of live_in[s]
            for (int v = 0; v < n_vregs; v++) {
                int8_t live_out_v = 0;
                for (int s : successors[i])
                    live_out_v |= live_in[s][v];
                if (live_out_v != live_out[i][v]) {
                    live_out[i][v] = live_out_v;
                    changed = true;
                }
            }
            // live_in[i] = use[i] OR (live_out[i] AND NOT def[i])
            for (int v = 0; v < n_vregs; v++) {
                const char live_in_v
                        = use_at[i][v] || (live_out[i][v] && !def_at[i][v]);
                if (live_in_v != live_in[i][v]) {
                    live_in[i][v] = live_in_v;
                    changed = true;
                }
            }
        }
    }
}

namespace {

// Compute the loop nesting depth of each operation. `compute_spill_weights()`
// later turns each depth into a weight.
std::vector<int> compute_loop_depth(const ir_t &ir) {
    const int n_ops = ir.n_ops();
    std::vector<int> depth(n_ops, 0);

    int d = 0;
    for (int i = 0; i < n_ops; i++) {
        switch (ir.ops()[i].kind) {
            case op_kind_t::loop_begin: depth[i] = d++; break;
            case op_kind_t::loop_end:
                assert(d > 0 && "loop_end without loop_begin");
                depth[i] = d--;
                break;
            default: depth[i] = d; break;
        }
    }
    assert(d == 0 && "unbalanced loop nest");

    return depth;
}

// Compute the spill weight of each virtual register. A read and a write in the
// same operation count separately, because a spilled value needs a reload for
// the read and a store for the write.
std::vector<int64_t> compute_spill_weights(
        const ir_t &ir, const std::vector<int> &depth) {
    // Assumed number of iterations of one loop level.
    constexpr int64_t loop_weight = 10;
    // Depth past which the weight stops growing, which keeps the sum in range.
    // References deeper than this all count the same.
    constexpr int max_weighted_depth = 8;

    std::vector<int64_t> weight(ir.n_vregs(), 0);
    std::vector<int> def_vregs, use_vregs;

    for (int i = 0; i < ir.n_ops(); i++) {
        int64_t op_weight = 1;
        for (int d = std::min(depth[i], max_weighted_depth); d > 0; d--)
            op_weight *= loop_weight;

        ir.def_use(ir.ops()[i], def_vregs, use_vregs);
        for (int v : def_vregs)
            weight[v] += op_weight;
        for (int v : use_vregs)
            weight[v] += op_weight;
    }

    return weight;
}

// Assign physical registers within a single physical register file. A file may
// serve more than one register kind (e.g. vec and mask on AVX2*).
//
// A spilled value lives in its stack slot for its whole live range. No
// register holds it between operations. At each operation that references it,
// it gets a temp, a register that holds it for that one operation. The value
// is loaded into the temp before the operation if the operation reads it, and
// stored back to the slot after if the operation writes it.
//
// The scan goes through the operations in order, keeping an `active` set of
// intervals that hold a register. At operation `i`:
//   1. Expire old intervals:
//      Remove every active interval that ended before `i`, freeing its
//      register.
//
//   2. Count the demand:
//      `i` needs a register from the free pool for
//      - each interval that starts at `i`
//      - each operand of `i` that is already spilled, as its temp. At most
//        `max_temps_per_op` temps of each kind are counted.
//
//   3. Spill if necessary:
//      While fewer registers are free than the demand, spill the lightest
//      active interval that is not an operand of `i`, with ties broken on the
//      latest end. A file that is not `spillable` skips this step.
//
//   4. Hand out registers:
//      Intervals that start at `i` take theirs and become active. An interval
//      left without one is spilled. Then each spilled operand counted in step
//      2 takes a temp while registers last. Temps return to the free pool
//      after `i`.
//
// A value spilled in step 3 is on the stack at all its references, including
// those before `i` that the scan has already passed, and they need temps too.
// From its start up to `i` the value held its register alone, so that register
// is free at each earlier reference and becomes its temp there. At each later
// reference, step 2 counts it as a spilled operand and step 4 hands it a temp.
//
// When every register holds an operand of `i`, or the file is not `spillable`,
// step 3 has nothing to spill. An operand left without a register then gets no
// temp, which the emitter reports by failing the kernel.
//
// Spilled values are assigned stack slots starting at `frame`, increasing by
// `slot_size` per spill.
//
// Example with 2 registers, r0 and r1. The diagram shows the final state,
// after the scan has finished. `|` marks each value's live interval. The
// register columns show what each register holds in the final allocation. `.`
// is free.
//
//                 live       register
//                 a  b  c    r0      r1
//   0  a = 1      |          a temp  .
//   1  b = 2      |  |       .       b
//   2  c = 3      |  |  |    c       b
//   3  b += c     |  |  |    c       b
//   4  a += 1     |          a temp  .
//
// How the scan gets there:
//   0: `a` takes r0.
//   1: `b` takes r1.
//   2: `c` needs a register and none is free. `a` and `b` are not operands of
//      2 and weigh 3 each, so the later end spills `a` and `c` takes r0. The
//      spill covers all of `a`, including 0 and 1, which the scan has passed.
//      `a` held r0 alone up to 2, so r0 becomes its temp at 0. At 1 `a` is not
//      an operand, so it needs no register there and r0 is free.
//   3: `b` and `c` are in registers.
//   4: `b` and `c` have ended. `a` is a spilled operand and gets r0 as its
//      temp.
//
// For `a` this becomes, with `[a]` its stack slot:
//   0: mov r0, 1
//      mov [a], r0
//   4: mov r0, [a]
//      add r0, 1
//      mov [a], r0
void alloc_file(const ir_t &ir, int file_idx, const reg_pools_t &pools,
        const std::vector<int> &start, const std::vector<int> &end,
        const std::vector<int64_t> &weight,
        const std::vector<std::vector<int>> &operands,
        const std::vector<std::vector<int>> &refs, reg_alloc_result_t &res,
        size_t &frame) {

    const int n_ops = ir.n_ops();
    const int n_vregs = ir.n_vregs();
    const reg_file_t &file = pools.files[file_idx];

    auto kind_of = [&](int v) { return ir.vreg_info()[v].kind; };

    auto in_file = [&](int v) {
        return pools.file_of(ir.vreg_info()[v]) == file_idx;
    };

    // For each operation, the intervals of this file that start there, by vreg
    // id. Step 4 hands out registers in this order. `end[v] < 0` means no
    // operation references `v`, so it has no interval.
    std::vector<std::vector<int>> starts_at(n_ops);
    for (int v = 0; v < n_vregs; v++)
        if (in_file(v) && end[v] >= 0) starts_at[start[v]].push_back(v);

    // True if `cand` is a better victim than `best`. A lighter weight wins,
    // then the later end, then the lower id. The id only breaks exact ties, so
    // the victim does not depend on the order of `active`.
    auto better_victim = [&](int cand, int best) {
        if (weight[cand] != weight[best]) return weight[cand] < weight[best];
        if (end[cand] != end[best]) return end[cand] > end[best];
        return cand < best;
    };

    // Moves `v` to a new stack slot for its whole live range and resets its
    // `phys`. The caller releases the register `v` held, if any, and must read
    // `phys` before the call.
    auto spill = [&](int v) {
        assignment_t &as = res.assignments[v];
        as.spilled = true;
        as.phys = -1;
        as.slot = frame;
        frame += file.slot_size;
        res.any_spill = true;
    };

    // Number of temps of `kind` that operation `i` has.
    auto n_temps = [&](int i, reg_kind_t kind) {
        int n = 0;
        for (const temp_reg_t &t : res.temps[i])
            if (kind_of((int)t.vreg) == kind) n++;
        return n;
    };

    // free_regs:    available registers, used as a stack. The top one is
    //               handed out next, and a freed register goes on top. It
    //               starts as the pool reversed, so the pool is handed out
    //               front to back.
    // active:       intervals that hold a register.
    // still_active: step 1's buffer for the intervals that stay active.
    // wanted:       spilled operands of the current operation that want a
    //               temp, at most `max_temps_per_op` per kind. Step 4 gives
    //               each one a temp while registers last.
    // last_ref:     latest operation so far that references each vreg.
    std::vector<int> free_regs(file.regs.rbegin(), file.regs.rend());
    std::vector<int> active, still_active, wanted;
    std::vector<int> last_ref(n_vregs, -1);

    for (int i = 0; i < n_ops; i++) {
        // 1. Expire intervals that ended before `i`.
        still_active.clear();
        for (int a : active) {
            if (end[a] < i)
                free_regs.push_back(res.assignments[a].phys);
            else
                still_active.push_back(a);
        }
        active.swap(still_active);

        // 2. Count the demand.
        // Intervals that start at `i`. Step 4 gives them registers, or spills
        // them when none is left.
        const std::vector<int> &pending = starts_at[i];
        wanted.clear();

        // Temps wanted per kind, indexed like `max_temps_per_op`.
        int n_wanted[sizeof(max_temps_per_op) / sizeof(max_temps_per_op[0])]
                = {};

        for (int v : operands[i]) {
            if (!in_file(v)) continue;

            last_ref[v] = i;

            if (!res.assignments[v].spilled) continue;

            const int k = (int)kind_of(v);
            if (n_wanted[k] < max_temps_per_op[k]) {
                n_wanted[k]++;
                wanted.push_back(v);
            }
        }

        // 3. Spill until enough registers are free.
        while (file.spillable
                && free_regs.size() < pending.size() + wanted.size()) {
            int victim = -1;
            for (int a : active) {
                // Spilling an operand of `i` frees nothing, since it needs a
                // temp at `i`.
                const bool is_operand = last_ref[a] == i;
                if (is_operand) continue;
                if (victim < 0 || better_victim(a, victim)) victim = a;
            }
            if (victim < 0) break; // nothing that frees a register at `i`

            // The register was the victim's alone up to `i`, so it is the temp
            // at each earlier reference.
            const int phys = res.assignments[victim].phys;
            const reg_kind_t kind = kind_of(victim);
            for (int j : refs[victim]) {
                if (j >= i) break;
                if (n_temps(j, kind) < max_temps_per_op[(int)kind])
                    res.temps[j].push_back({(vreg_t)victim, phys});
            }
            active.erase(std::find(active.begin(), active.end(), victim));
            free_regs.push_back(phys);
            spill(victim);
        }

        // 4. Hand out registers. They run short only when step 3 had nothing
        // to spill. A value left without a register is spilled. An operand
        // left without a temp is reported by the emitter.
        for (int v : pending) {
            if (free_regs.empty()) {
                spill(v);
                continue;
            }
            res.assignments[v].phys = free_regs.back();
            free_regs.pop_back();
            active.push_back(v);
        }

        const size_t first_temp = res.temps[i].size();
        for (int v : wanted) {
            if (free_regs.empty()) break;
            res.temps[i].push_back({(vreg_t)v, free_regs.back()});
            free_regs.pop_back();
        }
        // Temps return to the pool after `i`. `temps[i]` can already hold the
        // temps of files allocated before this one, so return only the ones
        // handed out here.
        for (size_t t = first_temp; t < res.temps[i].size(); t++)
            free_regs.push_back(res.temps[i][t].phys);
    }
}

} // namespace

// Run full register allocation pipeline.
// Returns, for each virtual register, either a physical register or a spill
// slot, and for each operation the temps of its spilled operands.
//
// The pipeline:
//   1. Compute liveness (live_in for each operation).
//   2. Convert liveness into a single interval per virtual register:
//        start[v] = first operation where `v` is defined, used, or live_in
//        end[v]   = last such operation
//      and record the distinct operands of each operation.
//   3. Weight each virtual register by how often it is referenced, so that the
//      scan spills the lowest-weight value rather than the longest-lived one.
//   4. Run linear-scan allocation per physical register file, sharing a single
//      stack frame across all files.
//
// Interval construction is intentionally conservative. If a value is live in
// disjoint regions (e.g. 3-7 and 20-25), it is merged into [3-25]. This
// ignores holes but guarantees correctness and keeps the algorithm simple.
//
// This approximation is usually acceptable because the most important values
// are hot loop-invariant pointers (e.g. base pointers for C and batch data),
// which are intended to remain in registers for the entire kernel anyway.
// For these cases, merging gaps has no practical cost. The only downside
// appears when a value has a true dead region while registers are scarce,
// where the gap could otherwise be reused.
//
// Hole-aware live-range splitting would recover that reuse. It is not planned,
// because the weights already keep the values that matter in registers, and
// splitting adds interval bookkeeping and reload placement. Recorded here in
// case pressure ever makes it worthwhile. Such splits would have to respect
// the weights and must not split a value inside a loop that carries it.
reg_alloc_result_t allocate_registers(
        const ir_t &ir, const reg_pools_t &pools) {
    const int n_ops = ir.n_ops();
    const int n_vregs = ir.n_vregs();

    // Step 1: compute operation-level liveness.
    // live_in[i][v] is 1 when virtual register `v` is needed on entry to
    // operation `i`.
    std::vector<std::vector<int8_t>> live_in;
    compute_liveness(ir, live_in);

    // Step 2: build a single live interval per virtual register.
    // Each interval [start[v], end[v]] approximates all points where `v` is
    // live.
    //
    // extend_interval(v) expands the interval to include operation `i`
    // whenever `v` is defined, used, or live on entry to `i`.
    //
    // `operands[i]` lists the vregs operation `i` reads or writes, each once
    // even when it is both read and written or passed twice. `refs[v]` lists
    // the operations that read or write `v`, in increasing order.
    std::vector<int> start(n_vregs, INT_MAX), end(n_vregs, -1);
    std::vector<std::vector<int>> operands(n_ops), refs(n_vregs);
    std::vector<int> def_vregs, use_vregs;

    for (int i = 0; i < n_ops; i++) {
        auto extend_interval = [&](int v) {
            start[v] = std::min(start[v], i);
            end[v] = std::max(end[v], i);
        };

        auto add_operand = [&](int v) {
            extend_interval(v);
            // A vreg both read and written, or passed twice, is listed once.
            // `refs[v]` grows in operation order, so `v` is already listed
            // exactly when its last reference is `i`.
            if (!refs[v].empty() && refs[v].back() == i) return;
            operands[i].push_back(v);
            refs[v].push_back(i);
        };

        ir.def_use(ir.ops()[i], def_vregs, use_vregs);
        for (int v : def_vregs)
            add_operand(v);
        for (int v : use_vregs)
            add_operand(v);
        for (int v = 0; v < n_vregs; v++)
            if (live_in[i][v]) extend_interval(v);
    }

    // Step 3: weight each virtual register by how often it is referenced and
    // how deep in the loop nest those references sit.
    const std::vector<int64_t> weight
            = compute_spill_weights(ir, compute_loop_depth(ir));

    // Step 4: run linear-scan register allocation per physical register file,
    // sharing a single stack frame so spill slots do not overlap across files.
    reg_alloc_result_t res;
    res.assignments.assign(n_vregs, assignment_t());
    res.temps.assign(n_ops, std::vector<temp_reg_t>());

    size_t frame = 0;
    for (int f = 0; f < (int)pools.files.size(); f++)
        alloc_file(
                ir, f, pools, start, end, weight, operands, refs, res, frame);

    constexpr size_t stack_alignment = 16;
    res.frame_bytes = utils::rnd_up(frame, stack_alignment);

    return res;
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
