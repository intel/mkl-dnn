/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <set>
#include <stack>
#include <unordered_map>

#include "dsl/dsl_impl.hpp"
#include "dsl/finalize.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/ir/pass/trace.hpp"
#include "dsl/ir/reorder.hpp"
#include "dsl/ir/send.hpp"
#include "dsl/utils/block_2d_utils.hpp"
#include "dsl/utils/logging.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
std::string make_asm(const kernel_t &kernel);
#endif

using builtin_t = ir::builtin_t;
using ir_context_t = ir::ir_context_t;
using send_op_t = ir::send_op_t;
using send_t = ir::send_t;
using stmt_seq_t = ir::stmt_seq_t;
using store_t = ir::store_t;
using var_t = ir::var_t;

struct ctx_t {
    const kernel::iface_t &interface() const { return interface_; }

    expr_t &get_expr(const std::string &name) { return expr_storage_[name]; }

    void declare_kernel(const kernel::iface_t &interface,
            const kernel::options_t &options) {
        ir::trace_start();
        slm_byte_offset_ = 0;
        gm_u64_map_ = {};
        expr_storage_ = {};
        dsl_assert(scope_stack_.empty())
                << "Invalid generation of a kernel within a kernel";
        interface_ = interface;
        ctx_ = ir_context_t(options);

        begin_scope();

        for (int i = 0; i < 3; i++) {
            group_ids_[i] = var_t::make(group_id_type(), ir::group_id_name(i));
            local_ids_[i] = var_t::make(local_id_type(), ir::local_id_name(i));
            local_sizes_[i]
                    = var_t::make(local_size_type(), ir::local_size_name(i));
            subgroup_ids_[i]
                    = var_t::make(subgroup_id_type(), ir::subgroup_id_name(i));
        }
        group_subgroup_count_
                = var_t::make(group_id_type(), ir::group_subgroup_count_name());
        subgroup_local_id_
                = var_t::make(local_id_type(), ir::subgroup_local_id_name());
        subgroup_linear_id_ = var_t::make(
                local_id_type().scalar(), ir::subgroup_linear_id_name());
    }

    kernel_t end_kernel() {
        dsl_assert(scope_stack_.size() == 1)
                << "Invalid end of kernel, imbalanced scopes detected";
        auto body = pop_scope();
        auto prologue = stmt_seq_t::make(generate_prologue(body));
        body = prologue.append(body);
        kernel_t ret {std::move(interface_), body, ctx_.options()};
        ctx_ = {};
        interface_ = {"undefined_dsl_kernel"};
        std::string pass_name = "Generate " + ret.iface.kernel_name();
        ir::trace_pass(pass_name.c_str(), ret);
        return ret;
    }

    int simd() const { return ctx_.options().simd(); }

    expr_t to_u64(const expr_t &gm_var) {
        auto it = gm_u64_map_.find(gm_var);
        if (it != gm_u64_map_.end()) return it->second;
        auto attr = gm_var.type().attr() & ~type::attr_t::gm;
        auto ret = var_t::make(u64.with_attr(attr), gm_var.as<var_t>().name);
        gm_u64_map_[gm_var] = ret;
        return ret;
    }

    const std::array<expr_t, 3> &group_ids() const { return group_ids_; }
    const expr_t &group_id(int idx) const { return group_ids_[idx]; }
    const expr_t &group_subgroup_count() const { return group_subgroup_count_; }
    const std::array<expr_t, 3> &local_ids() const { return local_ids_; }
    const expr_t &local_id(int idx) const { return local_ids_[idx]; }
    const std::array<expr_t, 3> &local_sizes() const { return local_sizes_; }
    const expr_t &local_size(int idx) const { return local_sizes_[idx]; }
    const expr_t &subgroup_id(int idx) const { return subgroup_ids_[idx]; }
    const expr_t &subgroup_local_id() const { return subgroup_local_id_; }
    const expr_t &subgroup_linear_id() const { return subgroup_linear_id_; }

    expr_t arg(const std::string &name, bool allow_empty = false) {
        auto a = interface_.find_arg(name, allow_empty);
        expr_t value;
        if (a && ctx_.cset().is_single_value(a, value)) { return value; }
        return a;
    }

    lval_t def(
            const std::string &name, type_t _type, const expr_t &value = {}) {
        auto type = _type.with_attr(_type.attr() | type::attr_t::mut);
        auto alloc_var = var(type, name);
        if (!alloc_var.type().is_slm()) {
            append(builtin_t::make("alloc")(alloc_var));
            scope_stack_.top().vars.insert(alloc_var);
        }
        if (!value.is_empty()) assign(alloc_var, value);
        return lval_t(alloc_var.as<var_t>());
    }

    lval_t def(const std::string &name, const expr_t &value) {
        return def(name, value.type(), value);
    }

    tensor_t def(const std::string &name, const layout_t &layout,
            type::attr_t attr, const expr_t &value = {}) {
        const auto &blocks = layout.blocks();
        int size = 1;
        if (!blocks.empty()) {
            const auto &b = blocks.back();
            size = into<int>(b.size * (int64_t)b.stride);
        }

        // Padding allocations due to strides with overlapping dimensions have
        // unclear semantics, disallow their use.
        dsl_assert([&]() {
            int64_t max_off = 0;
            for (auto &b : blocks) {
                max_off += (b.size - 1) * int64_t(b.stride);
            }
            return max_off < size;
        }());

        dsl_assert(layout.offset().is(0));
        const auto &type = layout.type();
        auto t = type.with_attr(attr);
        if (any(attr & type::attr_t::slm)) {
            dsl_assert(value.is_empty());
            auto buf = def(name, t[size]);
            auto size_bytes = size * type.size() / type.packing();
            auto off = div_up(slm_byte_offset() * type.packing(), type.size());
            auto off_bytes = off * type.size() / type.packing();
            reserve_slm((off_bytes - slm_byte_offset()) + size_bytes);
            return tensor_t(buf, layout.with_offset(off));
        }

        // Tensors need to be grf-aligned for loading/storing
        // TODO: IR should be modified to enable loading small tensors (such as
        // scalar values) without GRF alignment.
        const bool align_to_grf = true;
        if (align_to_grf)
            size = std::max(size, grf_size() / layout.type().size());
        auto buf = def(name, t[size], value);
        return tensor_t(buf, layout);
    }

    lval_t let(
            const std::string &name, const type_t &type, const expr_t &value) {
        auto alloc_var = var(type, name);
        append(builtin_t::make("alloc")(alloc_var));
        scope_stack_.top().vars.insert(alloc_var);
        assign(alloc_var, value);
        return alloc_var;
    }
    lval_t let(const std::string &name, const expr_t &value) {
        return let(name, value.type(), value);
    }

    int slm_byte_offset() const { return slm_byte_offset_; }

    void reserve_slm(int bytes) { slm_byte_offset_ += bytes; }

    void assume(const expr_t &e) { ctx_.add_constraint(e); }

    void begin_scope() { scope_stack_.emplace(); }

    void end_scope() {
        auto stmt = pop_scope();
        dsl_assert(!scope_stack_.empty());
        append(stmt);
    }

    stmt_t pop_scope() {
        auto stmt = to_stmt();
        scope_stack_.pop();
        return stmt;
    }

    void append(stmt_t stmt) {
        dsl_assert(!scope_stack_.empty())
                << "Cannot instantiate " << stmt << " outside of a kernel";
        stmts().emplace_back(std::move(stmt));
    }

    const ir_context_t &ir_ctx() const { return ctx_; }

private:
    type_t subgroup_id_type() const { return u16; }
    type_t local_id_type() const { return u16[simd()].with_simd(); }
    type_t group_id_type() const { return u32; }
    type_t local_size_type() const { return u16; }

    expr_t var(type_t type, const std::string &name) {
        return var_t::make(type, ctx_.create_tmp_name(name));
    }

    stmt_t to_stmt() {
        dsl_assert(!scope_stack_.empty());
        auto &vars = scope_stack_.top().vars;
        std::vector<stmt_t> scope_stmts(stmts().size() + vars.size());
        auto it = scope_stmts.end();
        auto prepend = [&](const stmt_t &s) {
            dsl_assert(it != scope_stmts.begin());
            *(--it) = s;
        };
        for (size_t i = stmts().size(); i > 0; i--) {
            auto &s = stmts()[i - 1];
            for (auto &var : ir::find_objects<var_t>(s)) {
                if (vars.count(var) > 0) {
                    prepend(builtin_t::make("free")(var));
                    vars.erase(var);
                }
            }
            prepend(s);
        }
        return stmt_seq_t::make(scope_stmts);
    }

    struct stmt_scope_t {
        std::vector<stmt_t> stmts;
        ir::object_set_t<expr_t> vars;
    };

    std::vector<stmt_t> generate_prologue(const stmt_t &body) {
        struct prologue_builder_t : public ir::ir_visitor_t {
            prologue_builder_t(ctx_t &ctx) : ctx(ctx) {}
            void _visit(const var_t &var) {
                for (int i = 0; i < 3; i++) {
                    auto &id = ctx.subgroup_ids_[i];
                    if (var == id.as<var_t>()) {
                        auto it = vars.emplace(var.name);
                        if (it.second) {
                            visit(ctx.local_id(i));
                            prologue.emplace_back(builtin_t::make("alloc")(id));
                            auto value = i == 0
                                    ? extract(ctx.local_id(i), 0) / ctx.simd()
                                    : extract(ctx.local_id(i), 0);
                            prologue.emplace_back(
                                    ir::assign_t::make(id, value));
                        }
                        return;
                    }
                }
                auto &lid = ctx.subgroup_local_id_.as<var_t>();
                if (var == lid) {
                    auto it = vars.emplace(var.name);
                    if (it.second) {
                        visit(ctx.local_id(0));
                        prologue.emplace_back(builtin_t::make("alloc")(lid));
                        // TODO: Use special register.
                        prologue.emplace_back(ir::assign_t::make(
                                lid, ctx.local_id(0) & (ctx.simd() - 1)));
                    }
                    return;
                }

                std::array<const expr_t *, 19> builtin_vars = {
                        &ctx.subgroup_linear_id_,
                        &ctx.group_subgroup_count_,
                        &ctx.group_ids_[0],
                        &ctx.group_ids_[1],
                        &ctx.group_ids_[2],
                        &ctx.local_ids_[0],
                        &ctx.local_ids_[1],
                        &ctx.local_ids_[2],
                        &ctx.local_sizes_[0],
                        &ctx.local_sizes_[1],
                        &ctx.local_sizes_[2],
                };

                for (auto &builtin : builtin_vars) {
                    if (builtin && builtin->is<var_t>()
                            && var == builtin->as<var_t>()) {
                        auto it = vars.emplace(var.name);
                        if (it.second)
                            prologue.emplace_back(ir::let_t::make(var, {}));
                        return;
                    }
                }
            }

            std::vector<stmt_t> get() { return prologue; }
            std::set<std::string> vars;
            std::vector<stmt_t> prologue;
            ctx_t &ctx;
        };

        auto pb = prologue_builder_t(*this);
        pb.visit(body);
        if (slm_byte_offset() > 0) {
            auto slm_buf = var_t::make(u8[slm_byte_offset()].with_slm(), "slm");
            pb.prologue.emplace_back(builtin_t::make("alloc")(slm_buf));
        }
        return pb.prologue;
    }

    std::vector<stmt_t> &stmts() { return scope_stack_.top().stmts; }
    std::stack<stmt_scope_t> scope_stack_;
    kernel::iface_t interface_ = {"undefined_dsl_kernel"};
    ir_context_t ctx_;
    std::array<expr_t, 3> group_ids_;
    expr_t group_subgroup_count_;
    std::array<expr_t, 3> local_ids_;
    std::array<expr_t, 3> local_sizes_;
    std::array<expr_t, 3> subgroup_ids_;
    expr_t subgroup_local_id_;
    expr_t subgroup_linear_id_;
    int slm_byte_offset_ = 0;

    // Mapping from global memory buffers to u64 address variables.
    ir::object_map_t<expr_t, expr_t> gm_u64_map_;
    // Cache for constant-valued expressions, keyed by variable name.
    std::unordered_map<std::string, expr_t> expr_storage_;
};

ctx_t &default_ctx() {
    static thread_local ctx_t ctx;
    return ctx;
}

const hw_t &get_hw() {
    return default_ctx().ir_ctx().hw();
}
int grf_size() {
    return get_hw().grf_size();
}
const kernel::options_t &options() {
    return default_ctx().ir_ctx().options();
}
int min_align_2d() {
    return block_2d_base_alignment(get_hw());
}
int min_pitch_2d() {
    return block_2d_pitch_alignment(get_hw());
}

void declare_kernel(
        const kernel::iface_t &interface, const kernel::options_t &options) {
    default_ctx().declare_kernel(interface, options);
}

kernel_t end_kernel() {
    auto kernel = default_ctx().end_kernel();
    finalize(kernel);
#ifdef GEMMSTONE_WITH_ASM_RUNTIME
    dsl_debug() << make_asm(kernel);
#endif
    return kernel;
}

void begin_scope() {
    default_ctx().begin_scope();
}

void end_scope() {
    default_ctx().end_scope();
}

stmt_t pop_scope() {
    return default_ctx().pop_scope();
}

void append(stmt_t stmt) {
    default_ctx().append(std::move(stmt));
}

void assume(const expr_t &e) {
    default_ctx().assume(e);
}

int simd() {
    return default_ctx().simd();
}

const kernel::iface_t &kernel_iface() {
    return default_ctx().interface();
}

expr_t &get_expr(const std::string &name) {
    return default_ctx().get_expr(name);
}

expr_t to_u64(const expr_t &gm_var, const expr_t &off = {}) {
    auto ret = default_ctx().to_u64(gm_var);
    if (off) ret += off * gm_var.type().scalar().size();
    return ret;
}

expr_t mad(
        const expr_t &dst, const expr_t &a, const expr_t &b, const expr_t &c) {
    append(builtin_t::make("mad")({dst, a, b, c}));
    return dst;
}

const std::array<expr_t, 3> &group_ids() {
    return default_ctx().group_ids();
}

const expr_t &group_id(int idx) {
    return default_ctx().group_id(idx);
}

const expr_t &group_subgroup_count() {
    return default_ctx().group_subgroup_count();
}

const std::array<expr_t, 3> &local_ids() {
    return default_ctx().local_ids();
}

const expr_t &local_id(int idx) {
    return default_ctx().local_id(idx);
}

const std::array<expr_t, 3> &local_sizes() {
    return default_ctx().local_sizes();
}

const expr_t &local_size(int idx) {
    return default_ctx().local_size(idx);
}

const expr_t global_id(int idx, type_t type) {
    return cast(group_id(idx), type) * local_size(idx) + local_id(idx);
}

expr_t subgroup_id(int idx) {
    return default_ctx().subgroup_id(idx);
}

expr_t subgroup_local_id() {
    return default_ctx().subgroup_local_id();
}

expr_t subgroup_linear_id() {
    return default_ctx().subgroup_linear_id();
}

expr_t arg(const std::string &name, bool allow_empty) {
    return default_ctx().arg(name, allow_empty);
}

lval_t def(const std::string &name, const type_t &type, const expr_t &value) {
    return default_ctx().def(name, type, value);
}

lval_t def(const std::string &name, const expr_t &value) {
    return def(name, value.type(), value);
}

tensor_t def(const std::string &name, const layout_t &layout,
        const expr_t &value, type::attr_t attr) {
    return default_ctx().def(name, layout, attr, value);
}

tensor_t def(
        const std::string &name, const layout_t &layout, type::attr_t attr) {
    return def(name, layout, {}, attr);
}

expr_t iif(
        const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr) {
    return ir::iif_t::make(cond, true_expr, false_expr);
}

expr_t extract(const expr_t &expr, int lane) {
    return ir::shuffle_t::make(expr, {lane});
}

expr_t extract(const expr_t &expr, int lane_start, int lane_end) {
    return ir::ref_t::make(expr, lane_start, lane_end - lane_start);
}

uint32_t uint32_reciprocal(uint32_t x) {
    if (x == 0) return 0;
    return (uint32_t)div_up(uint64_t(0x100000000) << ilog2(x), x);
}

expr_t idiv_impl(const expr_t &a, const expr_t &b, const expr_t &recip,
        expr_t *rem = nullptr) {
    dsl_assert(!b.type().is_simd());
    auto quot = def("quot", a.type());
    quot = ir::ternary_op_t::make(ir::op_kind_t::_idiv, a, b, recip);
    if (rem) {
        auto _rem = def("rem", a.type());
        _rem = a - b * quot;
        *rem = (expr_t)_rem;
    }
    return quot;
}

expr_t idiv(const expr_t &a, uint32_t b) {
    return idiv_impl(a, b, uint32_reciprocal(b));
}
expr_t idiv(const expr_t &a, uint32_t b, expr_t &rem) {
    return idiv_impl(a, b, uint32_reciprocal(b), &rem);
}
expr_t idiv(const expr_t &a, const expr_t &b, const expr_t &recip) {
    return idiv_impl(a, b, recip);
}
expr_t idiv(
        const expr_t &a, const expr_t &b, const expr_t &recip, expr_t &rem) {
    return idiv_impl(a, b, recip, &rem);
}

expr_t _divide_up(const expr_t &a, uint32_t b) {
    using ir::operator+;
    if (is_pow2(b)) return (a + b - 1) / b;
    auto tmp = def("div_down", idiv(a, b));
    return iif(expr_t(tmp) * b < a, tmp + 1, tmp);
}

expr_t _round_up(const expr_t &a, uint32_t b) {
    using ir::operator+;
    if (is_pow2(b)) return (a + b - 1) / b * b;
    auto tmp = def("rnd_down", idiv(a, b) * b);
    return iif(tmp < a, tmp + b, tmp);
}

lval_t::lval_t(const type_t &type, const std::string &name)
    : var(var_t::make(type, name)) {}

lval_t &lval_t::operator=(const expr_t &obj) {
    dsl::assign(this->var, obj);
    return *this;
}

lval_t lval_t::sub(int off, int elems) const {
    return lval_t(ir::ref_t::make(var, off, elems));
}

lval_t let(const std::string &name, const type_t &type, const expr_t &value) {
    return default_ctx().let(name, type, value);
}

lval_t let(const std::string &name, const expr_t &value) {
    return default_ctx().let(name, value);
}

void assign(const expr_t &var, const expr_t &value) {
    if (value.is_empty()) return;
    append(ir::assign_t::make(var, value));
}

void scatter_send(const expr_t &value, const expr_t &buf, const expr_t &off,
        const expr_t &mask, send_op_t op_kind, const send_hint_t &hint) {
    dsl_assert(value.type().scalar() == buf.type().scalar())
            << "Invalid operation, incompatible types: " << value << ": "
            << value.type() << ", " << buf << ": " << buf.type();

    auto send_func = send_t::make({}, op_kind, ir::send_address_t::a64,
            value.type().scalar(), value.type().elems(), true, true,
            hint.cache);
    auto &send = send_func.as<send_t>();
    dsl_assert(buf.type().is_gm());
    auto buf_u64 = to_u64(buf);
    int elem_bytes = buf.type().scalar().size();
    auto addr = buf_u64 + elem_bytes * off;
    append(send.as<send_t>()(expr_t(), addr, value, mask));
}

void scatter_send(const tensor_t &t, const global_tensor_t &g,
        send_op_t op_kind, const icoord_t &base, const send_hint_t &hint) {
    const auto layout = t.layout(); // t.layout() is otherwise ephemeral
    dsl_assert(!layout.is_empty());
    if (layout.is_empty()) return;

    const auto &blocks = layout.blocks();

    // Compute the memory storage of t in elements.
    // XXX: This assumes "nice" layouts.
    int64_t elems = 1;
    for (auto &b : blocks) {
        auto stride = (int64_t)b.stride;
        elems = std::max(elems, stride * b.size);
    }
    const auto transposed = !t.type().is_simd();
    const auto width = simd();
    const auto messages = div_up(elems, width);

    // Recast the tensor as a 1D layout with padding for the vectorized case.
    // This is to simplify offset calculations.
    const auto dummy_elems = messages * width;
    const std::vector<layout::block_t> dummy_blocks = {{0, dummy_elems}};
    const layout_t dummy_layout = layout.with(dummy_blocks);
    const tensor_t dummy {t.buf(), dummy_layout};
    const tile_t tile = {{0, width}};

    auto dst = dummy;
    if (transposed) {
        const std::vector<layout::block_t> tmp_blocks = {{0, width}};
        const layout_t tmp_layout = layout.with(tmp_blocks);
        dst = def("tmp", tmp_layout, type::attr_t::simd);
    }

    auto global_mask = [&](const coord_t &offsets) {
        const auto &keys = g.tile().keys();
        const auto coords = g.coord() + offsets;
        const auto &strides = g.strides();
        const auto &sizes = g.sizes();
        expr_t mask;
        for (const auto &key : keys) {
            if (strides[key].is(0)) continue;
            auto dim_mask = coords[key] < sizes[key];
            mask = mask.is_empty() ? dim_mask : mask & dim_mask;
        }
        return mask;
    };

    for (const auto &coord : dummy_layout.iter(tile)) {
        expr_t mask;
        coord_t coords;
        auto idx = subgroup_local_id() + coord[0];
        // Check the tail if the dummy layout is padded
        if (coord[0] + width > elems && elems % width) mask = idx < elems;
        int64_t outer_stride = elems;
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            auto &coord = coords[it->idx];
            auto stride = (int64_t)it->stride;
            auto outer = idx / stride;
            auto inner = idx - outer * stride;
            coord = coord * it->size + outer;
            if (stride * it->size != outer_stride) {
                // Non-dense layout, make sure not to access out-of-bounds
                auto block_mask = outer < it->size;
                mask = mask.is_empty() ? block_mask : mask & block_mask;
            }
            idx = std::move(inner);
            outer_stride = stride;
        }
        auto gmask = elems == 1 ? expr_t() : global_mask(coords);
        mask = mask.is_empty() ? gmask : gmask.is_empty() ? mask : mask & gmask;
        auto tile_elems = into<int>(tile.elems());
        auto buf = transposed ? dst.subvec({}, tile_elems)
                              : dst.subvec(coord, tile_elems);
        scatter_send(buf, g.buf(), g.offset(coords), mask, op_kind, hint);
        if (transposed) {
            auto vec_elems
                    = into<int>(std::min(tile.elems(), elems - coord[0]));
            assign(dummy.subvec(coord, vec_elems), extract(buf, 0, vec_elems));
        }
    }
}

layout_t prefetch_layout(const global_tensor_t &g, const idx_t &w_idx) {
    std::vector<layout::block_t> blocks;
    blocks.reserve(g.tile().size());
    blocks.emplace_back(w_idx, g.tile()[w_idx]);
    for (auto &idx : g.tile()) {
        if (idx != w_idx) blocks.emplace_back(idx, g.tile()[idx]);
    }
    return layout_t(g.scalar_type(), blocks);
}

void block_send(const tensor_t &t, const global_tensor_t &g, send_op_t &op,
        const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.is_empty();
    auto &operation_tile = is_prefetch ? g.tile() : t.tile();

    idx_t w_idx;
    tile_t tile;
    for (auto &var : operation_tile) {
        if (is_const(g.strides()[var])
                && ir::to_cpp<int64_t>(g.strides()[var]) == 1
                && t.elems() != 1) {
            tile[var] = t.layout()[0].size;
            dsl_assert(t.layout()[0].idx == var);
            w_idx = var;
        } else {
            tile[var] = 1;
        }
    }
    auto type = g.scalar_type();
    auto buf_u64 = to_u64(g.buf());

    auto operation_layout
            = is_prefetch ? prefetch_layout(g, w_idx) : t.layout();
    for (auto &coord : operation_layout.iter(tile)) {
        auto buffer = is_prefetch ? expr_t()
                                  : t.subvec(coord, into<int>(tile.elems()));
        auto width = !w_idx.is_undef()
                ? std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx])
                : 1;

        int width_bytes = into<int>(width * type.size());
        auto coord_local = coord;
        while (width_bytes > 0) {
            auto send_type = [&]() {
                if (width_bytes <= 16) { return type_t::byte(width_bytes); }
                auto load_width = rounddown_pow2(std::min(width_bytes, 512));
                return type_t::oword(load_width / 16);
            }();
            auto send_func = send_t::make({}, op, ir::send_address_t::a64,
                    send_type, 1, true, true, hint.cache);
            auto &send = send_func.as<send_t>();

            auto h = def("h", u8[send.header_size()]);
            store_t::make(h, 0,
                    g.offset(base + coord_local) * g.scalar_type().size());
            append(send.create_offset_store(h, buf_u64,
                    g.offset(base + coord_local) * g.scalar_type().size()));

            append(send.as<send_t>()(buf_u64, h, buffer, {}));

            width_bytes -= send_type.size();
            coord_local[w_idx] += send_type.size() / type.size();
        }
    }
}

struct conf_2d_t {
    type_t type;
    idx_t w_idx;
    int pack_size;
    bool is_vnni;
    bool is_transpose_vnni;
    bool is_store;

    int unit_size() const {
        return is_transpose_vnni || is_vnni ? std::max(type.size(), 4)
                                            : type.size();
    }

    // Tile used for 2d Messages
    tile_t get_tile(std::array<idx_t, 2> dims) const {
        auto width = pack_size ? pack_size : grf_size() / unit_size();
        auto height = is_store ? 8 : 32;

        if (is_transpose_vnni) return {{dims[1], width}, {dims[0], height}};
        return {{dims[0], width}, {dims[1], height}};
    }
};

void block_2d_send(const conf_2d_t &conf, const tensor_t &t,
        const global_tensor_t &g, send_op_t op, const icoord_t &base,
        const send_hint_t &hint) {

    bool is_prefetch = t.is_empty();
    auto &operation_tile = is_prefetch ? g.tile() : t.tile();

    idx_t w_idx = conf.w_idx;
    idx_t h_idx;
    for (auto &var : operation_tile) {
        if (var != w_idx) {
            dsl_assert(h_idx.is_undef())
                    << "n-dimensional support unimplemented";
            h_idx = var;
        }
    }

    auto tensor_width = g.sizes()[w_idx];
    auto tensor_height = g.sizes()[h_idx];
    auto tensor_pitch = g.strides()[h_idx];
    auto type = g.scalar_type();
    auto tile = conf.get_tile({w_idx, h_idx});
    auto buf_u64 = to_u64(g.buf());

    auto operation_layout
            = is_prefetch ? prefetch_layout(g, w_idx) : t.layout();
    for (auto &coord : operation_layout.iter(tile)) {
        auto buffer = is_prefetch ? expr_t()
                                  : t.subvec(coord, into<int>(tile.elems()));
        int width = into<int>(
                std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx]));
        int height = into<int>(
                std::min(tile[h_idx], operation_tile[h_idx] - coord[h_idx]));
        int count = std::max(1, into<int>(tile[w_idx] / width));
        auto width_idx = g.coord()[w_idx]
                + static_cast<uint32_t>((base + coord)[w_idx]);
        auto height_idx = g.coord()[h_idx]
                + static_cast<uint32_t>((base + coord)[h_idx]);
        switch (op) {
            case send_op_t::prefetch: op = send_op_t::prefetch_2d; break;
            case send_op_t::load: op = send_op_t::load_2d; break;
            case send_op_t::store: op = send_op_t::store_2d; break;
            default:
                stub();
                op = send_op_t::undef;
                break;
        }

        auto send_func = send_t::make_2d({}, op, type, tensor_width,
                tensor_height, tensor_pitch, width, height, count, conf.is_vnni,
                conf.is_transpose_vnni,
                /*zero_out=*/true, hint.cache);
        auto &send = send_func.as<send_t>();

        auto h = def("h", u8[send.header_size()]);

        auto write = [&](const expr_t &e, int off) {
            append(store_t::make(h, off, e));
        };
        auto write_s32 = [&](const expr_t &value, int off) {
            append(store_t::make(h, off, cast(value, type_t::s32())));
        };
        auto &info = send.block_2d_info;
        int type_size = send.type.size();
        write(buf_u64, 0);
        write_s32(info.surface_width * type_size - 1,
                send_t::header_2d_off_surface_width());
        write_s32(info.surface_height - 1,
                send_t::header_2d_off_surface_height());
        write_s32(info.surface_pitch * type_size - 1,
                send_t::header_2d_off_surface_pitch());
        write(width_idx, send_t::header_2d_off_x());
        write(height_idx, send_t::header_2d_off_y());
        uint32_t w_enc = info.width - 1;
        uint32_t h_enc = info.height - 1;
        uint32_t count_enc = info.count - 1;
        write_s32((count_enc << 16) + (h_enc << 8) + w_enc,
                send_t::header_2d_off_whc());

        append(send(buf_u64, h, buffer, {}));
    }
}

void send(const tensor_t &t, const global_tensor_t &g, send_op_t op,
        const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.is_empty();
    auto &operation_tile = is_prefetch ? g.tile() : t.tile();
    idx_t w_idx;
    for (auto &var : operation_tile) {
        if (is_const(g.strides()[var])
                && ir::to_cpp<int64_t>(g.strides()[var]) == 1) {
            dsl_assert(w_idx.is_undef())
                    << "Could not determine inner dimension";
            w_idx = var;
        }
    }

    auto type = g.scalar_type();
    const bool enable_2d_send = true;
    const bool enable_block_send = true;

    dsl_assert(is_prefetch || type == t.scalar_type());
    if (enable_2d_send && operation_tile.size() >= 2 && !w_idx.is_undef()) {
        auto conf = [&]() -> conf_2d_t {
            if (is_prefetch) {
                return {g.scalar_type(), w_idx, 0, false, false, false};
            }
            auto l = t.layout();
            int pack_idx = l[0].size * l.type().size() == 4;
            int pack_size = into<int>(l[pack_idx].size);
            bool is_transpose_vnni = l[pack_idx].idx != w_idx;
            bool is_vnni = pack_idx == 1 && !is_transpose_vnni;
            bool is_store = op == send_op_t::store;
            return {g.scalar_type(), w_idx, pack_size, is_vnni,
                    is_transpose_vnni, is_store};
        }();

        if (conf.pack_size <= grf_size() / conf.unit_size()) {
            block_2d_send(conf, t, g, op, base, hint);
            return;
        }
    }

    if (enable_block_send
            && (is_prefetch || t.elems() == 1 || t.layout()[0].idx == w_idx)) {
        block_send(t, g, op, base, hint);
    } else {
        scatter_send(t, g, op, base, hint);
    }
}

void prefetch(const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send({}, g, send_op_t::prefetch, base, hint);
}

expr_t load(const expr_t &buf, const expr_t &off, const expr_t &mask,
        const send_hint_t &hint) {
    auto res_type = buf.type().scalar()[off.type().elems()].with_simd();
    expr_t val = def("load_tmp", res_type);
    scatter_send(val, buf, off, mask, send_op_t::load, hint);
    return val;
}

void load(const tensor_t &t, const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_op_t::load, base, hint);
}

void store(const expr_t &buf, const expr_t &off, const expr_t &val,
        const expr_t mask, const send_hint_t &hint) {
    scatter_send(val, buf, off, mask, send_op_t::store, hint);
}

void store(const global_tensor_t &g, const tensor_t &t, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_op_t::store, base, hint);
}

bool is_blocked_by(const tensor_t &T, const idx_t &idx, int block) {
    auto layout = T.layout();
    if (layout.blocks().empty()) return false;
    auto &b = layout.blocks()[0];
    return b.idx == idx && b.size % block == 0;
}

void get_mnk_dims(const tile_t &a, const tile_t &b, const tile_t &c, idx_t &m,
        idx_t &n, idx_t &k) {
    for (auto &d : a) {
        if (b.has(d)) {
            k = d;
        } else {
            m = d;
        }
    }
    for (auto &d : b) {
        if (d == k) continue;
        n = d;
    }
    dsl_assert(a.get(m) == c.get(m))
            << "M dimension mismatch: A[" << m << "]=" << a.get(m) << " vs C["
            << m << "]=" << c.get(m);
    dsl_assert(b.get(n) == c.get(n))
            << "N dimension mismatch: B[" << n << "]=" << b.get(n) << " vs C["
            << n << "]=" << c.get(n);
    dsl_assert(a.get(k) == b.get(k))
            << "K dimension mismatch: A[" << k << "]=" << a.get(k) << " vs B["
            << k << "]=" << b.get(k);
}

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B) {
    idx_t m_dim, n_dim, k_dim;
    get_mnk_dims(A.tile(), B.tile(), C.tile(), m_dim, n_dim, k_dim);
    const int64_t M = C.tile().get(m_dim);
    const int64_t N = C.tile().get(n_dim);
    const int64_t K = A.tile().get(k_dim);
    const int m_blk = simd();
    const int n_blk = simd();
    dsl_assert(is_blocked_by(A, m_dim, m_blk));
    dsl_assert(is_blocked_by(C, m_dim, m_blk));
    for (int64_t n = 0; n < N; n += n_blk) {
        for (int64_t k = 0; k < K; k++) {
            icoord_t b_coord {{{k_dim, k}, {n_dim, n}}};
            auto b_vec = def("b_vec", f32[n_blk], B.subvec(b_coord, n_blk));
            for (int n_inner = 0; n_inner < n_blk; n_inner++) {
                auto b = b_vec[n_inner];
                for (int64_t m = 0; m < M; m += m_blk) {
                    icoord_t coord {
                            {{m_dim, m}, {n_dim, n + n_inner}, {k_dim, k}}};
                    auto a = A.subvec(coord, m_blk);
                    auto c = C.subvec(coord, m_blk);
                    mad(c, a, b, c);
                }
            }
        }
    }
}

void _if_impl(const expr_t &cond, const stmt_t &if_body) {
    append(ir::if_t::make(cond, if_body));
}
void _if_impl(
        const expr_t &cond, const stmt_t &if_body, const stmt_t &else_body) {
    append(ir::if_t::make(cond, if_body, else_body));
}
void _for_impl(const expr_t &var, const expr_t &bound, const expr_t &step,
        const stmt_t &body) {
    append(ir::for_t::make(var, 0, bound, body, step));
}
void _while_impl(const expr_t &cond, const stmt_t &body) {
    append(ir::while_t::make(cond, body));
}

void binary(op::kind_t op, const tensor_t &dst, const tensor_t &src0,
        const tensor_t &src1) {
    const bool is_src0_scalar = src0.elems() == 1;
    const bool is_src1_scalar = src1.elems() == 1;
    const bool is_src0_simd = src0.type().is_simd();
    const bool is_src1_simd = src1.type().is_simd();

    // SIMD-width subtile from first block of dst layout.
    tile_t subtile;
    if (dst.type().is_simd()) {
        auto &b0 = dst.layout().blocks()[0];
        subtile[b0.idx] = simd();
        dsl_assert(b0.size % simd() == 0)
                << "Incompatible dst layout: " << dst.layout().str();
    }

    // TODO: Add validation for compatibility between dst/src0/src1.

    int dst_elems = into<int>(subtile.elems());
    int s0_elems = is_src0_scalar || !is_src0_simd ? 1 : dst_elems;
    int s1_elems = is_src1_scalar || !is_src1_simd ? 1 : dst_elems;

    for (auto &coord : dst.layout().iter(subtile)) {
        auto d = dst.subvec(coord, dst_elems);
        auto s0 = is_src0_scalar ? src0.subvec({}, 1)
                                 : src0.subvec(coord, s0_elems);
        auto s1 = is_src1_scalar ? src1.subvec({}, 1)
                                 : src1.subvec(coord, s1_elems);
        assign(d, ir::binary_op_t::make((ir::op_kind_t)op, s0, s1));
    }
}

static tile_t max(const tile_t &a, const tile_t &b) {
    tile_t tile = a;
    for (auto &d : b)
        tile[d] = std::max(tile.get(d, 1), b[d]);
    return tile;
}

static void check_binary_operand(const tensor_t &tensor) {
    auto type = tensor.type();
    dsl_assert(!type.is_gm())
            << "check_binary_operand: tensor must not be a global memory type";
    dsl_assert(!type.is_slm())
            << "check_binary_operand: tensor must not be an SLM";
    dsl_assert(!type.is_packed())
            << "check_binary_operand: tensor must not be packed";

    // If SIMD: verify all inner blocks up to simd() elements are dense.
    if (type.is_simd()) {
        dsl_assert(tensor.layout().nblocks() > 0)
                << "check_binary_operand: SIMD tensor must have at least one "
                   "block";
        auto b0 = tensor.layout().blocks()[0];
        dsl_assert(b0.stride == 1 && b0.size % simd() == 0)
                << "check_binary_operand: unsupported SIMD tensor";
    }
}

// Deduces a dense output layout from two source layouts with broadcast semantics. For each dimension, the output size is max(a_size, b_size).
static layout_t deduce_binary_layout(const type_t &dst_type,
        const layout_t &a_layout, const layout_t &b_layout, bool a_simd,
        bool b_simd) {
    if (a_layout.nblocks() == 0) return b_layout.with(dst_type);
    if (b_layout.nblocks() == 0) return a_layout.with(dst_type);

    auto &a0 = a_layout.blocks()[0];
    auto &b0 = b_layout.blocks()[0];

    auto dst_tile = max(a_layout.tile(), b_layout.tile());
    auto rem_dst_tile = dst_tile;

    std::vector<layout::block_t> blocks;
    if (a_simd && b_simd) {
        dsl_assert(a0.idx == b0.idx)
                << "deduce_binary_layout: incompatible A/B";
    }
    idx_t simd_idx = (a_simd ? a0.idx : (b_simd ? b0.idx : idx_t()));
    if (!simd_idx.is_undef()) {
        blocks.emplace_back(simd_idx, simd());
        rem_dst_tile[a0.idx] /= simd();
    }

    for (auto &d : rem_dst_tile) {
        blocks.emplace_back(d, rem_dst_tile[d]);
    }

    return layout_t(dst_type, blocks);
}

// Shared implementation for binary ops with tensor and/or scalar operands.
// When a_expr/b_expr is non-empty, the corresponding scalar expr is used instead of the tensor.
// When dst is non-empty, results are written into it instead of a freshly allocated tensor.
static tensor_t binary_impl(op::kind_t op, const tensor_t &a,
        const expr_t &a_expr, const tensor_t &b, const expr_t &b_expr,
        tensor_t dst = {}) {
    bool a_scalar = !a_expr.is_empty();
    bool b_scalar = !b_expr.is_empty();

    if (!a_scalar) check_binary_operand(a);
    if (!b_scalar) check_binary_operand(b);

    bool a_simd = !a_scalar && a.is_simd();
    bool b_simd = !b_scalar && b.is_simd();

    if (dst.is_empty()) {
        type_t a_type = a_scalar ? a_expr.type() : a.scalar_type();
        type_t b_type = b_scalar ? b_expr.type() : b.scalar_type();
        layout_t a_layout = a_scalar ? layout_t(a_type) : a.layout();
        layout_t b_layout = b_scalar ? layout_t(b_type) : b.layout();

        type_t common = ir::common_type(a_type, b_type).with_mut();
        if (a_simd || b_simd) common = common.with_simd();
        tile_t tile = max(a_layout.tile(), b_layout.tile());
        auto dst_buf_type = common[into<int>(tile.elems())];
        auto dst_layout = deduce_binary_layout(
                dst_buf_type.scalar(), a_layout, b_layout, a_simd, b_simd);
        dst = def("res", dst_layout, dst_buf_type.attr());
    }

    tile_t subtile;
    if (dst.type().is_simd()) {
        auto b0 = dst.layout().blocks()[0];
        subtile[b0.idx] = simd();
    }

    int dst_elems = into<int>(subtile.elems());
    int a_elems = (a_simd ? dst_elems : 1);
    int b_elems = (b_simd ? dst_elems : 1);
    for (auto &coord : dst.layout().iter(subtile)) {
        auto d = dst.subvec(coord, dst_elems);
        auto a_val = a_scalar ? a_expr : a.subvec(coord, a_elems);
        auto b_val = b_scalar ? b_expr : b.subvec(coord, b_elems);
        assign(d, ir::binary_op_t::make((ir::op_kind_t)op, a_val, b_val));
    }
    return dst;
}

tensor_t binary(op::kind_t op, const tensor_t &src0, const tensor_t &src1) {
    return binary_impl(op, src0, {}, src1, {});
}

tensor_t binary(op::kind_t op, const tensor_t &src0, const expr_t &src1) {
    return binary_impl(op, src0, {}, {}, src1);
}

tensor_t binary(op::kind_t op, const expr_t &src0, const tensor_t &src1) {
    return binary_impl(op, {}, src0, src1, {});
}

#define DEFINE_TENSOR_BINARY_OPERATOR(cpp_op, op_kind) \
    tensor_t operator cpp_op(const tensor_t &a, const tensor_t &b) { \
        return binary(op_kind, a, b); \
    } \
    tensor_t operator cpp_op(const tensor_t &a, const expr_t &b) { \
        return binary(op_kind, a, b); \
    } \
    tensor_t operator cpp_op(const expr_t &a, const tensor_t &b) { \
        return binary(op_kind, a, b); \
    }

DEFINE_TENSOR_BINARY_OPERATOR(+, op::add)
DEFINE_TENSOR_BINARY_OPERATOR(-, op::sub)
DEFINE_TENSOR_BINARY_OPERATOR(*, op::mul)
DEFINE_TENSOR_BINARY_OPERATOR(/, op::div)

#undef DEFINE_TENSOR_BINARY_OPERATOR

#define DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR(cpp_op, op_kind) \
    void operator cpp_op##=(const tensor_t &a, const tensor_t &b) { \
        binary_impl(op_kind, a, {}, b, {}, a); \
    } \
    void operator cpp_op##=(const tensor_t &a, const expr_t &b) { \
        binary_impl(op_kind, a, {}, {}, b, a); \
    }

DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR(+, op::add)
DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR(-, op::sub)
DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR(*, op::mul)
DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR(/, op::div)

#undef DEFINE_TENSOR_BINARY_ASSIGN_OPERATOR

void barrier() {
    append(builtin_t::make("barrier")());
}

expr_t cast(const type_t &type, const expr_t &src) {
    if (src.type().is_gm()) {
        if (type == u64) return to_u64(src);
        dsl_error() << "Cannot cast gm expression: " << src.str() << " to "
                    << type.str();
    }
    return ir::cast(src, type);
}

tensor_t cast(const type_t &type, const tensor_t &src) {
    const auto &in_type = src.type();
    if (in_type.scalar() == type.scalar() && in_type.is_simd() == type.is_simd()
            && in_type.is_packed() == type.is_packed())
        return src;
    auto dst_layout
            = src.layout().with(type.scalar()).make_dense().with_offset(0);
    auto dst_tensor = def("buf", dst_layout, type.attr());
    auto reorder = ir::reorder_t::make(
            src.layout(), dst_layout, /*do_normalize=*/false);
    append(reorder({dst_tensor.buf(), src.buf()}));
    return dst_tensor;
}

tensor_t exp(const tensor_t &src) {
    auto dst = def("exp", src.layout(), src.type().attr());
    tile_t subtile;
    if (src.is_simd()) {
        auto b0 = src.layout().blocks()[0];
        subtile[b0.idx] = simd();
    }
    int elems = (src.is_simd() ? into<int>(subtile.elems()) : 1);
    for (auto &coord : dst.layout().iter(subtile)) {
        auto d = dst.subvec(coord, elems);
        auto s = src.subvec(coord, elems);
        assign(d, ir::unary_op_t::make(ir::op_kind_t::_exp, s));
    }
    return dst;
}

expr_t exp(const expr_t &src, const expr_t &acc_out, const expr_t &acc_in) {
    if (src.type().elems() > simd()) {
        auto res = def("res", src.type());
        append(builtin_t::make("exp")({res, src, acc_out, acc_in}));
        return res;
    } else {
        return ir::unary_op_t::make(ir::op_kind_t::_exp, src);
    }
}

expr_t exp(const expr_t &src, const type_t &out_type, const expr_t &acc_out,
        const expr_t &acc_in) {
    auto res_type = out_type.with_elems(src.type().elems())
                            .with_attr(src.type().attr());
    auto res = def("res", res_type);
    append(builtin_t::make("exp")({res, src, acc_out, acc_in}));
    return res;
}

expr_t min(const expr_t &a, const expr_t &b) {
    return ir::binary_op_t::make(ir::op_kind_t::_min, a, b);
}

expr_t max(const expr_t &a, const expr_t &b) {
    return ir::binary_op_t::make(ir::op_kind_t::_max, a, b);
}

void slm_fence() {
    append(builtin_t::make("slm_fence")());
}

global_tensor_t def_global_tensor(const std::string &name,
        const std::vector<expr_t> &batch_idxs, const std::vector<idx_t> &dims,
        const std::vector<expr_t> &dim_sizes, std::vector<expr_t> dim_strides,
        bool transpose) {
    auto ndims = dims.size();
    idx_map_t<expr_t> sizes;
    idx_map_t<expr_t> strides;
    if (transpose) std::swap(dim_strides[ndims - 1], dim_strides[ndims - 2]);
    for (size_t i = 0; i < ndims; ++i) {
        const auto &dim = dims[i];
        sizes[dim] = dim_sizes[i];
        strides[dim] = dim_strides[i];
    }

    auto buf = arg(name);
    expr_t batch_off;
    if (!batch_idxs.empty()) {
        batch_off = expr_t(0);
        for (size_t i = 0; i < batch_idxs.size(); i++) {
            batch_off += batch_idxs[i]
                    * arg("stride_" + name + std::to_string(i));
        }
        batch_off = def(name + "_off", batch_off);
    }
    return global_tensor_t(buf, batch_off, sizes, strides);
}

global_tensor_t def_global_tensor(const std::string &name,
        const std::vector<expr_t> &batch_idxs, const std::vector<idx_t> &dims,
        const std::vector<expr_t> &sizes, bool transpose) {
    auto ndims = sizes.size();
    std::vector<expr_t> strides(ndims);
    const auto &inner = sizes[ndims - 1];
    const auto &outer = sizes[ndims - 2];
    strides[ndims - 1] = 1;
    strides[ndims - 2] = transpose ? outer : inner;
    expr_t stride = inner * outer;
    for (size_t i = ndims - 2; i > 0; i--) {
        strides[i - 1] = stride;
        stride *= sizes[i - 1];
    }
    return def_global_tensor(
            name, batch_idxs, dims, sizes, std::move(strides), transpose);
}

#ifndef STANDALONE
namespace op {
kind_t kind(dnnl::impl::alg_kind_t alg) {
    using namespace dnnl::impl::alg_kind;
    switch (alg) {
        case binary_add: return op::add;
        case binary_sub: return op::sub;
        case binary_mul: return op::mul;
        case binary_div: return op::div;
        case binary_max: return op::max;
        case binary_min: return op::min;
        case binary_prelu: return op::prelu;
        case binary_ge:
        case binary_gt:
        case binary_le:
        case binary_lt:
        case binary_eq:
        case binary_ne:
        default: dsl_error() << "Unsupported algorithm: " << alg;
    }
    return op::kind_t::undef;
}
} // namespace op
#endif

} // namespace dsl
GEMMSTONE_NAMESPACE_END
