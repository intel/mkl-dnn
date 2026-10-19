/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dsl/ir/core.hpp"

#include <algorithm>

#include "dsl/ir/pass/simplify.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

expr_t const_fold_non_recursive(const expr_t &expr);
object_t const_fold(const object_t &obj);

std::string to_string(op_kind_t kind) {
    switch (kind) {
        case op_kind_t::_minus: return "-";
        case op_kind_t::_exp: return "exp";

        case op_kind_t::_add: return "+";
        case op_kind_t::_sub: return "-";
        case op_kind_t::_mul: return "*";
        case op_kind_t::_div: return "/";
        case op_kind_t::_mod: return "%";
        case op_kind_t::_shl: return "<<";
        case op_kind_t::_shr: return ">>";
        case op_kind_t::_min: return "min";
        case op_kind_t::_max: return "max";

        case op_kind_t::_lt: return "<";
        case op_kind_t::_le: return "<=";
        case op_kind_t::_gt: return ">";
        case op_kind_t::_ge: return ">=";
        case op_kind_t::_eq: return "==";
        case op_kind_t::_ne: return "!=";

        case op_kind_t::_and: return "&&";
        case op_kind_t::_or: return "||";
        case op_kind_t::_xor: return "^";

        case op_kind_t::_add3: return "add3";
        case op_kind_t::_mad: return "mad";
        case op_kind_t::_div_up: return "div_up";
        case op_kind_t::_prelu: return "prelu";
        case op_kind_t::_idiv: return "idiv";
        case op_kind_t::_imod: return "imod";

        default: dsl_error() << "Unknown op_kind_t value.";
    }
    return "";
}

bool is_cmp_op(op_kind_t op_kind) {
    switch (op_kind) {
        case op_kind_t::_ge:
        case op_kind_t::_gt:
        case op_kind_t::_le:
        case op_kind_t::_lt:
        case op_kind_t::_eq:
        case op_kind_t::_ne: return true;
        default: return false;
    }
}

bool is_commutative_op(op_kind_t op_kind) {
    switch (op_kind) {
        case op_kind_t::_add:
        case op_kind_t::_mul:
        case op_kind_t::_min:
        case op_kind_t::_max:
        case op_kind_t::_eq:
        case op_kind_t::_ne:
        case op_kind_t::_and:
        case op_kind_t::_or:
        case op_kind_t::_xor:
        case op_kind_t::_add3: return true;
        default: return false;
    }
}

op_kind_t negate_cmp_op(op_kind_t op_kind) {
    switch (op_kind) {
        case op_kind_t::_ge: return op_kind_t::_le;
        case op_kind_t::_gt: return op_kind_t::_lt;
        case op_kind_t::_le: return op_kind_t::_ge;
        case op_kind_t::_lt: return op_kind_t::_gt;
        case op_kind_t::_eq: return op_kind_t::_eq;
        case op_kind_t::_ne: return op_kind_t::_ne;
        default: stub();
    }
    return op_kind_t::undef;
}

type_t unary_op_type(op_kind_t op_kind, const expr_t &a) {
    switch (op_kind) {
        case op_kind_t::_minus: {
            auto &t = a.type();
            if (!t.is_int()) return t;
            if (t.size() < int(sizeof(int32_t)))
                return type_t::s32(t.elems(), a.type().attr());
            return t;
        }
        case op_kind_t::_exp: return a.type();
        default: dsl_error() << "Unknown op_kind_t value: " << op_kind;
    }
    return type_t::undef();
}

type::attr_t common_attr(const type_t &a, const type_t &b) {
    dsl_assert(!(a.is_ptr() && b.is_ptr()));
    return (a.attr() | b.attr()) & ~type::attr_t::mut;
}

type_t common_type(const type_t &base, const type_t &a, const type_t &b) {
    auto attr = common_attr(a, b);
    int elems = std::max(a.elems(), b.elems());
    return base.with_attr(attr)[elems];
}

type_t common_int_type_impl(const type_t &_a, const type_t &_b) {
    dsl_assert(_a.is_int() && _b.is_int()) << "Unexpected types.";

    // Promote to s32 first.
    type_t a = _a.size() < int(sizeof(int32_t)) ? type_t::s32() : _a;
    type_t b = _b.size() < int(sizeof(int32_t)) ? type_t::s32() : _b;
    a = a.scalar();
    b = b.scalar();

    // Integer promotion, follow C++ rules.
    int common_bits = 8 * std::max(a.size(), b.size());
    if (a.is_signed() == b.is_signed()) {
        if (a.is_signed()) return type_t::s(common_bits);
        return type_t::u(common_bits);
    }

    if (a.size() >= b.size() && a.is_unsigned()) return type_t::u(common_bits);
    if (b.size() >= a.size() && b.is_unsigned()) return type_t::u(common_bits);
    if (a.size() > b.size() && a.is_signed()) return type_t::s(common_bits);
    if (b.size() > a.size() && b.is_signed()) return type_t::s(common_bits);

    return type_t::u(common_bits);
}

type_t common_int_type(const type_t &a, const type_t &b) {
    return common_type(common_int_type_impl(a, b), a, b);
}

type_t common_type_impl(const type_t &a, const type_t &b) {
    if (a.is_undef() || b.is_undef()) return type_t::undef();
    if (a.is_fp() && !b.is_fp()) return a;
    if (!a.is_fp() && b.is_fp()) return b;
    if (a.is_fp() && b.is_fp()) return (a.size() > b.size() ? a : b);
    if (a.is_bool() && b.is_bool()) return a;
    return common_int_type(a, b);
}

type_t common_type(const type_t &a, const type_t &b) {
    return common_type(common_type_impl(a, b), a, b);
}

type_t common_type(const expr_t &a, const expr_t &b) {
    return common_type(a.type(), b.type());
}

type_t binary_op_type(op_kind_t op_kind, const type_t &a, const type_t &b,
        const expr_t &a_expr = expr_t(), const expr_t &b_expr = expr_t()) {
    if (a.is_undef() || b.is_undef()) return type_t::undef();
    int elems = std::max(a.elems(), b.elems());

    type::attr_t attr = common_attr(a, b);
    if (is_cmp_op(op_kind)) return type_t::_bool(elems, attr);
    if (one_of(op_kind, {op_kind_t::_shl, op_kind_t::_shr})) {
        return a[elems].with_attr(attr);
    }

    if (one_of(op_kind, {op_kind_t::_and, op_kind_t::_or, op_kind_t::_xor})) {
        if (a == b) return a;
        if (is_const(a_expr)) return b;
        if (is_const(b_expr)) return a;
        return (a.size() >= b.size()) ? a : b;
    }
    if (one_of(op_kind, {op_kind_t::_div, op_kind_t::_mod}) && a.is_int()
            && b.is_int()) {
        return (a.is_signed() ? type_t::s32() : type_t::u32())[elems].with_attr(
                attr);
    }
    return common_type(a, b);
}

type_t binary_op_type(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
    return binary_op_type(op_kind, a.type(), b.type(), a, b);
}

type_t ternary_op_type(
        op_kind_t op_kind, const expr_t &a, const expr_t &b, const expr_t &c) {
    switch (op_kind) {
        case op_kind_t::_add3:
            return binary_op_type(op_kind_t::_add, a.type(),
                    binary_op_type(op_kind_t::_add, b, c));
        case op_kind_t::_mad:
            return binary_op_type(op_kind_t::_add, a.type(),
                    binary_op_type(op_kind_t::_mul, b, c));
        case op_kind_t::_idiv:
        case op_kind_t::_imod: return binary_op_type(op_kind_t::_idiv, a, b);
        default: stub();
    }
    return type_t::undef();
}

type_t nary_op_type(op_kind_t op_kind, const std::vector<expr_t> &args) {
    dsl_assert(!args.empty());
    if (args.size() == 1) return args[0].type();

    auto type = args[0].type();
    for (size_t i = 1; i < args.size(); i++)
        type = common_type(type, args[i].type());

    return type;
}

void ptr_t::normalize(expr_t &base, expr_t &off, op_kind_t op_kind) {
    // Normalize (base + off1) + off2 -> base + (off1 + off2)
    dsl_assert(off.type().is_int()) << "off is not an integer: " << off;
    dsl_assert(one_of(op_kind, {op_kind_t::_add, op_kind_t::_sub}))
            << "Can't apply this operation to pointer: " << to_string(op_kind);

    if (!base.is<ptr_t>()) {
        if (op_kind == op_kind_t::_sub) off = const_fold(-off);
        return;
    }

    off = const_fold_non_recursive(
            binary_op_t::make(op_kind, base.as<ptr_t>().off, off));
    base = base.as<ptr_t>().base;
}

expr_t shift_ptr(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
    expr_t base = a;
    expr_t off = b;
    ptr_t::normalize(base, off, op_kind);
    return ptr_t::make(base, off);
}

void normalize_ptr(const type_t &type, expr_t &base_expr, expr_t &off) {
    if (base_expr.is<ptr_t>()) {
        off = const_fold_non_recursive(base_expr.as<ptr_t>().off + off);
        base_expr = base_expr.as<ptr_t>().base;
        return;
    }
    dsl_assert(is_const(off)) << "var/ref requires constant offset.";
    if (auto *ref = base_expr.as_ptr<ref_t>()) {
        off = const_fold_non_recursive(ref->off + off);
        base_expr = ref->var;
        return;
    }
    if (base_expr.is<var_t>()) return;
    dsl_error() << "Unexpected expression: " << base_expr;
}

expr_t linear_t::to_expr() const {
    auto ret = c;
    for (int i = 0; i < nargs(); i++) {
        ret += u_vec[i] * v_vec[i];
    }
    return simplify_rewrite(ret);
}

static void stmt_seq_flatten(std::vector<stmt_t> &out, const stmt_t &s) {
    if (auto *seq = s.as_ptr<stmt_seq_t>()) {
        out.insert(out.end(), seq->vec.begin(), seq->vec.end());
        return;
    }
    out.push_back(s);
}

stmt_t stmt_seq_t::make(const std::vector<stmt_t> &_vec) {
    std::vector<stmt_t> vec;
    for (auto &s : _vec)
        stmt_seq_flatten(vec, s);
    return stmt_t(new stmt_seq_t(vec));
}

#define DECL_TRAVERSE_LEAF(name) \
    object_t ir_mutator_t::_mutate(const name &obj) { \
        return obj; \
    } \
    void ir_visitor_t::_visit(const name &obj) {}

DECL_TRAVERSE_LEAF(bool_imm_t)
DECL_TRAVERSE_LEAF(const_var_t)
DECL_TRAVERSE_LEAF(float_imm_t)
DECL_TRAVERSE_LEAF(int_imm_t)
DECL_TRAVERSE_LEAF(var_t)

#undef DECL_TRAVERSE_LEAF

object_t ir_mutator_t::_mutate(const alloc_t &obj) {
    auto buf = mutate(obj.buf);
    auto body = mutate(obj.body);

    if (buf.is_same(obj.buf) && body.is_same(obj.body)) return obj;

    return alloc_t::make(buf, obj.size, obj.kind, obj.attrs, body);
}

void ir_visitor_t::_visit(const alloc_t &obj) {
    visit(obj.buf);
    visit(obj.body);
}

object_t ir_mutator_t::_mutate(const assign_t &obj) {
    auto var = mutate(obj.var);
    auto value = mutate(obj.value);

    if (var.is_same(obj.var) && value.is_same(obj.value)) return obj;

    return assign_t::make(var, value);
}

void ir_visitor_t::_visit(const assign_t &obj) {
    visit(obj.var);
    visit(obj.value);
}

object_t ir_mutator_t::_mutate(const ref_t &obj) {
    auto var = mutate(obj.var);
    if (var.impl() == obj.var.impl()) return obj;
    return ref_t::make(var, obj.off, obj.elems);
}

void ir_visitor_t::_visit(const ref_t &obj) {
    visit(obj.var);
}

object_t ir_mutator_t::_mutate(const binary_op_t &obj) {
    auto a = mutate(obj.a);
    auto b = mutate(obj.b);

    if (a.is_same(obj.a) && b.is_same(obj.b)) return obj;

    return binary_op_t::make(obj.op_kind, a, b);
}

void ir_visitor_t::_visit(const binary_op_t &obj) {
    visit(obj.a);
    visit(obj.b);
}

object_t ir_mutator_t::_mutate(const cast_t &obj) {
    auto expr = mutate(obj.expr);

    if (expr.is_same(obj.expr)) return obj;

    return cast_t::make(obj.type, expr, obj.saturate);
}

void ir_visitor_t::_visit(const cast_t &obj) {
    visit(obj.expr);
}

object_t ir_mutator_t::_mutate(const for_t &obj) {
    auto var = mutate(obj.var);
    auto init = mutate(obj.init);
    auto bound = mutate(obj.bound);
    auto body = mutate(obj.body);
    auto step = mutate(obj.step);

    if (var.is_same(obj.var) && init.is_same(obj.init)
            && bound.is_same(obj.bound) && body.is_same(obj.body))
        return obj;

    return for_t::make(var, init, bound, body, step, obj.unroll);
}

void ir_visitor_t::_visit(const for_t &obj) {
    visit(obj.var);
    visit(obj.init);
    visit(obj.bound);
    visit(obj.body);
}

object_t ir_mutator_t::_mutate(const func_call_t &obj) {
    auto args = mutate(obj.args);

    if (utils::is_same(args, obj.args)) return obj;

    return func_call_t::make(obj.func, args, obj.attr);
}

void ir_visitor_t::_visit(const func_call_t &obj) {
    visit(obj.args);
}

object_t ir_mutator_t::_mutate(const if_t &obj) {
    auto cond = mutate(obj.cond);
    auto body = mutate(obj.body);
    auto else_body = mutate(obj.else_body);

    if (cond.is_same(obj.cond) && body.is_same(obj.body)
            && else_body.is_same(obj.else_body))
        return obj;

    return if_t::make(cond, body, else_body);
}

void ir_visitor_t::_visit(const if_t &obj) {
    visit(obj.cond);
    visit(obj.body);
    visit(obj.else_body);
}

object_t ir_mutator_t::_mutate(const iif_t &obj) {
    auto cond = mutate(obj.cond);
    auto true_expr = mutate(obj.true_expr);
    auto false_expr = mutate(obj.false_expr);

    if (cond.is_same(obj.cond) && true_expr.is_same(obj.true_expr)
            && false_expr.is_same(obj.false_expr))
        return obj;

    return iif_t::make(cond, true_expr, false_expr);
}

void ir_visitor_t::_visit(const iif_t &obj) {
    visit(obj.cond);
    visit(obj.true_expr);
    visit(obj.false_expr);
}

object_t ir_mutator_t::_mutate(const let_t &obj) {
    auto var = mutate(obj.var);
    auto value = mutate(obj.value);
    auto body = mutate(obj.body);

    if (var.is_same(obj.var) && value.is_same(obj.value)
            && body.is_same(obj.body))
        return obj;

    return let_t::make(var, value, body);
}

void ir_visitor_t::_visit(const let_t &obj) {
    visit(obj.var);
    visit(obj.value);
    visit(obj.body);
}

object_t ir_mutator_t::_mutate(const linear_t &obj) {
    stub();
    return obj;
}

void ir_visitor_t::_visit(const linear_t &obj) {
    stub();
}

object_t ir_mutator_t::_mutate(const load_t &obj) {
    auto buf = mutate(obj.buf);
    auto off = mutate(obj.off);

    if (buf.is_same(obj.buf) && off.is_same(obj.off)) return obj;

    return load_t::make(obj.type, buf, off, obj.stride);
}

void ir_visitor_t::_visit(const load_t &obj) {
    visit(obj.buf);
    visit(obj.off);
}

object_t ir_mutator_t::_mutate(const ptr_t &obj) {
    auto base = mutate(obj.base);
    auto off = mutate(obj.off);

    if (base.is_same(obj.base) && off.is_same(obj.off)) return obj;

    return ptr_t::make(base, off);
}

void ir_visitor_t::_visit(const ptr_t &obj) {
    visit(obj.base);
    visit(obj.off);
}

object_t ir_mutator_t::_mutate(const shuffle_t &obj) {
    auto vec = mutate(obj.vec);

    if (utils::is_same(vec, obj.vec)) return obj;

    return shuffle_t::make(vec, obj.idx);
}

void ir_visitor_t::_visit(const shuffle_t &obj) {
    visit(obj.vec);
}

object_t ir_mutator_t::_mutate(const stmt_group_t &obj) {
    auto body = mutate(obj.body);

    if (body.is_same(obj.body)) return obj;

    return stmt_group_t::make(obj.label, body);
}

void ir_visitor_t::_visit(const stmt_group_t &obj) {
    visit(obj.body);
}

object_t ir_mutator_t::_mutate(const stmt_seq_t &obj) {
    auto vec = mutate(obj.vec);
    if (utils::is_same(vec, obj.vec)) return obj;
    return stmt_seq_t::make(vec);
}

void ir_visitor_t::_visit(const stmt_seq_t &obj) {
    visit(obj.vec);
}

object_t ir_mutator_t::_mutate(const store_t &obj) {
    auto buf = mutate(obj.buf);
    auto off = mutate(obj.off);
    auto value = mutate(obj.value);
    auto mask = mutate(obj.mask);

    if (buf.is_same(obj.buf) && off.is_same(obj.off) && value.is_same(obj.value)
            && mask.is_same(obj.mask))
        return obj;

    return store_t::make(buf, off, value, obj.stride, mask, obj.fill_mask0);
}

void ir_visitor_t::_visit(const store_t &obj) {
    visit(obj.buf);
    visit(obj.off);
    visit(obj.value);
    visit(obj.mask);
}

object_t ir_mutator_t::_mutate(const ternary_op_t &obj) {
    auto a = mutate(obj.a);
    auto b = mutate(obj.b);
    auto c = mutate(obj.c);

    if (a.is_same(obj.a) && b.is_same(obj.b) && c.is_same(obj.c)) return obj;

    return ternary_op_t::make(obj.op_kind, a, b, c);
}

void ir_visitor_t::_visit(const ternary_op_t &obj) {
    visit(obj.a);
    visit(obj.b);
    visit(obj.c);
}

object_t ir_mutator_t::_mutate(const unary_op_t &obj) {
    auto a = mutate(obj.a);
    if (a.is_same(obj.a)) return obj;
    return unary_op_t::make(obj.op_kind, a);
}

void ir_visitor_t::_visit(const unary_op_t &obj) {
    visit(obj.a);
}

object_t ir_mutator_t::_mutate(const while_t &obj) {
    auto cond = mutate(obj.cond);
    auto body = mutate(obj.body);

    if (cond.is_same(obj.cond) && body.is_same(obj.body)) return obj;

    return while_t::make(cond, body);
}

void ir_visitor_t::_visit(const while_t &obj) {
    visit(obj.cond);
    visit(obj.body);
}

const std::string &local_id_name(int idx) {
    static const std::array<std::string, 3> names
            = {"local_id0", "local_id1", "local_id2"};
    return names[idx];
}
const std::string &local_size_name(int idx) {
    static const std::array<std::string, 3> names
            = {"local_size0", "local_size1", "local_size2"};
    return names[idx];
}
const std::string &group_id_name(int idx) {
    static const std::array<std::string, 3> names
            = {"group_id0", "group_id1", "group_id2"};
    return names[idx];
}
const std::string &group_subgroup_count_name() {
    static const std::string name = "group_subgroup_count";
    return name;
}
const std::string &subgroup_id_name(int idx) {
    static const std::array<std::string, 3> names
            = {"subgroup_id0", "subgroup_id1", "subgroup_id2"};
    return names[idx];
}
const std::string &subgroup_local_id_name() {
    static const std::string name = "subgroup_local_id";
    return name;
}
const std::string &subgroup_linear_id_name() {
    static const std::string name = "subgroup_linear_id";
    return name;
}
const std::string &tg_idx_name(int idx) {
    return group_id_name(idx);
}
const std::string &thr_idx_name(int idx) {
    return subgroup_id_name(idx);
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
