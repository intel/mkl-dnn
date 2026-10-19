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

#include "dsl/ir/ir.hpp"

#include "dsl/ir/codegen/allocation_size.hpp"
#include "dsl/ir/core.hpp"
#include "dsl/ir/pass/simplify.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

using namespace utils;

namespace {

// Helper class to print IR objects.
class ir_printer_t : public ir_visitor_t {
public:
    ir_printer_t(std::ostream &out, int indent = 0)
        : out_(out), indent_(indent) {}

    void _visit(const assign_t &obj) override {
        print_indent();
        out_ << obj << "\n";
    }

    void _visit(const alloc_t &obj) override {
        auto grf_size = 1; // Assume all objects are grf aligned
        auto guard = mem_usage_guard(register_size(obj, grf_size));
        print_indent();
        out_ << obj.line_str() << "(mem_usage: " << mem_usage_bytes_ << ")\n";
        visit(obj.body);
    }

    void _visit(const binary_op_t &obj) override {
        switch (obj.op_kind) {
            case op_kind_t::_div_up:
            case op_kind_t::_idiv:
            case op_kind_t::_imod:
            case op_kind_t::_max:
            case op_kind_t::_min:
                out_ << to_string(obj.op_kind) << "(" << obj.a << ", " << obj.b
                     << ")";
                return;
            case op_kind_t::_prelu:
                out_ << "(" << obj.a << " > 0 ? " << obj.a << " : " << obj.a
                     << " * " << obj.b << ")";
                break;
            default:
                out_ << "(";
                visit(obj.a);
                out_ << " " << to_string(obj.op_kind) << " ";
                visit(obj.b);
                out_ << ")";
                break;
        }
    }

    void _visit(const bool_imm_t &obj) override {
        out_ << (obj.value ? "true" : "false");
    }

    void _visit(const cast_t &obj) override {
        out_ << obj.type;
        if (obj.saturate) out_ << ".sat";
        out_ << "(" << obj.expr << ")";
    }

    void _visit(const const_var_t &obj) override { out_ << obj.name; }

    void _visit(const float_imm_t &obj) override { out_ << obj.value; }

    void _visit(const for_t &obj) override {
        print_indent();
        out_ << "for (" << obj.var << " = " << obj.init << "; " << obj.var
             << " < " << obj.bound << "; " << obj.var << " += " << obj.step
             << ") ";
        if (obj.unroll != 1) out_ << "[unroll: " << obj.unroll << "] ";
        out_ << "{\n";
        add_indent();
        visit(obj.body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

    void _visit(const func_call_t &obj) override {
        print_indent();
        out_ << obj.line_str() << "\n";
    }

    void _visit(const if_t &obj) override {
        print_indent();
        out_ << obj.line_str() << " {\n";
        add_indent();
        visit(obj.body);
        remove_indent();
        print_indent();
        if (obj.else_body.is_empty()) {
            out_ << "}\n";
            return;
        }
        out_ << "} else {\n";
        add_indent();
        visit(obj.else_body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

    void _visit(const iif_t &obj) override {
        out_ << "(" << obj.cond << " ? " << obj.true_expr << " : "
             << obj.false_expr << ")";
    }

    void _visit(const int_imm_t &obj) override {
        out_ << std::to_string(obj.value);
    }

    void _visit(const let_t &obj) override {
        int size = register_size(obj);
        auto guard = mem_usage_guard(size);
        print_indent();
        out_ << obj.line_str() << "\n";
        visit(obj.body);
    }

    void _visit(const linear_t &obj) override {
        if (obj.nargs() == 0 && obj.c.is(0)) {
            out_ << "0";
            return;
        }
        out_ << "(";
        for (int i = 0; i < obj.nargs(); i++) {
            if (i > 0) out_ << " + ";
            if (obj.u_vec[i].is(1)) {
                out_ << obj.v_vec[i];
            } else {
                out_ << obj.u_vec[i] << " * " << obj.v_vec[i];
            }
        }
        if (!obj.c.is(0)) {
            if (obj.nargs() != 0) out_ << " + ";
            out_ << obj.c;
        }
        out_ << ")";
    }

    void _visit(const load_t &obj) override {
        out_ << obj.buf;
        if (obj.has_default_stride()) {
            out_ << "." << obj.type << "(" << obj.off / obj.type.size() << ")";
        } else {
            out_ << "[" << obj.off << "]." << obj.type;
            out_ << "<" << obj.stride << ">";
        }
    }

    void _visit(const ptr_t &obj) override {
        out_ << obj.base << "[" << obj.off << "]";
    }

    void _visit(const shuffle_t &obj) override {
        if (obj.is_broadcast()) {
            out_ << "bcast" << obj.elems() << "(" << obj.vec[0] << ")";
            return;
        }
        std::vector<expr_t> vec_all;
        for (auto &v : obj.vec) {
            for (int i = 0; i < v.type().elems(); i++)
                vec_all.push_back(v);
        }
        int elems = obj.type.elems();
        out_ << "(";
        for (int i = 0; i < elems; i++) {
            int idx = obj.idx[i];
            auto &v = vec_all[idx];
            int v_elems = v.type().elems();
            out_ << v;
            if (v_elems != 1) out_ << "[" << idx << "]";
            if (i != elems - 1) out_ << ", ";
        }
        out_ << ")";
    }

    void _visit(const stmt_group_t &obj) override {
        print_indent();
        out_ << obj.label.str() << " {\n";
        add_indent();
        visit(obj.body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

    void _visit(const stmt_seq_t &obj) override {
        for (auto &s : obj.vec)
            visit(s);
    }

    void _visit(const store_t &obj) override {
        print_indent();
        out_ << obj.line_str() << "\n";
    }

    void _visit(const ternary_op_t &obj) override {
        out_ << to_string(obj.op_kind) << "(" << obj.a << ", " << obj.b << ", "
             << obj.c << ")";
    }

    void _visit(const unary_op_t &obj) override {
        out_ << to_string(obj.op_kind);
        visit(obj.a);
    }

    void _visit(const var_t &obj) override { out_ << obj.name; }

    void _visit(const ref_t &obj) override { out_ << obj; }

    void _visit(const while_t &obj) override {
        print_indent();
        out_ << obj.line_str() << " {\n";
        add_indent();
        visit(obj.body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

private:
    mem_usage_guard_t mem_usage_guard(int size) {
        return mem_usage_guard_t(&mem_usage_bytes_, size);
    }

    static std::string strip_parens(const std::string &s) {
        if (s.size() < 2 || s[0] != '(' || s[s.size() - 1] != ')') return s;
        auto ret = s;
        ret.resize(s.size() - 1);
        return ret.substr(1);
    }

    void print_indent() {
        for (int i = 0; i < indent_; i++)
            out_ << prefix_;
    }

    void add_indent() { indent_++; }
    void remove_indent() { indent_--; }

    std::ostream &out_;
    int indent_ = 0;

    std::string prefix_ = "  ";

    // Size required for all enclosed let/alloc statements. The value is
    // updated during traversal.
    int mem_usage_bytes_ = 0;
};

class substitute_mutator_t : public ir_mutator_t {
public:
    substitute_mutator_t(const object_t &from, const object_t &to)
        : from_(from), to_(to) {}

    int substitutions() const { return substitutions_; }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type &obj) override { \
        if (from_.impl() == (const impl_t *)&obj) { \
            substitutions_++; \
            return to_; \
        } \
        return ir_mutator_t::_mutate(obj); \
    }

    HANDLE_CORE_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    object_t from_;
    object_t to_;

    int substitutions_ = 0;
};

class substitute_and_type_mutator_t : public ir_mutator_t {
public:
    substitute_and_type_mutator_t(const object_t &from, const object_t &to) {
        substitutes_[from] = to;
    }

    int substitutions() const { return substitutions_; }

    template <typename T>
    object_t _mutate_after(const T &obj) {
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate_after(const let_t &obj) {
        auto var = mutate(obj.var);
        auto value = mutate(obj.value);

        // Allow changing variable types when performing substitutions. Avoids
        // the following invalid substitute transformation sequence:
        //
        // tmp0.s32            -> tmp0_0.u64
        // tmp1.s32 = tmp0.s32 -> tmp1.s32 = tmp0_0.u64
        if (value) {
            auto value_type = expr_t(value).type();
            if (var.as<var_t>().type != value_type) {
                auto var_old = var;
                var = var_t::make(value_type, var.as<var_t>().name);

                substitutes_[var_old] = var;
            }
        }

        auto body = mutate(obj.body);

        if (var.is_same(obj.var) && value.is_same(obj.value)
                && body.is_same(obj.body))
            return obj;

        return let_t::make(var, value, body);
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type &obj) override { \
        auto it = substitutes_.find(obj); \
        if (it != substitutes_.end()) { \
            substitutions_++; \
            return it->second; \
        } \
        return _mutate_after(obj); \
    }

    HANDLE_CORE_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    object_eq_map_t<object_t, object_t> substitutes_;

    int substitutions_ = 0;
};

class stmt_flattener_t : public ir_visitor_t {
public:
#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) { \
        size_t old_size = stmts.size(); \
        ir_visitor_t::_visit(obj); \
        if (stmts.size() > old_size) return; \
        if (obj.is_stmt()) stmts.emplace_back(obj); \
    }

    HANDLE_CORE_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    std::vector<stmt_t> stmts;
};

} // namespace

std::string to_string(const object_t &object, int indent) {
    ostringstream_t oss;
    ir_printer_t printer(oss, indent);
    printer.visit(object);
    return oss.str();
}
std::string object::impl_t::str() const {
    ostringstream_t oss;
    ir_printer_t printer(oss);
    printer.visit(this);
    return oss.str();
}

object_t substitute(const object_t &root, const object_t &from,
        const object_t &to, int max_substitutions) {
    if (to.is_same(from)) return root;
    substitute_mutator_t sm(from, to);
    auto ret = sm.mutate(root);
    dsl_assert(sm.substitutions() <= max_substitutions)
            << "Unexpected number of substitutions.";
    return ret;
}

object_t substitute_with_different_type(const object_t &root,
        const object_t &from, const object_t &to, int max_substitutions) {
    if (to.is_same(from)) return root;
    substitute_and_type_mutator_t sm(from, to);
    auto ret = sm.mutate(root);
    dsl_assert(sm.substitutions() <= max_substitutions)
            << "Unexpected number of substitutions.";
    return ret;
}

std::vector<stmt_t> flatten_statements(const stmt_t &root) {
    stmt_flattener_t f;
    f.visit(root);
    return f.stmts;
}

std::vector<expr_t> split_by_and(const expr_t &e) {
    auto *binary = e.as_ptr<binary_op_t>();
    if (!binary || binary->op_kind != op_kind_t::_and) return {e};
    auto a = split_by_and(binary->a);
    auto b = split_by_and(binary->b);
    auto ret = std::move(a);
    ret.insert(ret.end(), b.begin(), b.end());
    return ret;
}

expr_t abs(const expr_t &e) {
    dsl_assert(is_const(e)) << e;
    if (to_cpp<bool>(e >= 0)) return e;
    return -e;
}

expr_t max(const expr_t &a, const expr_t &b) {
    return binary_op_t::make(op_kind_t::_max, a, b);
}

expr_t min(const expr_t &a, const expr_t &b) {
    return binary_op_t::make(op_kind_t::_min, a, b);
}

expr_t cast(const expr_t &e, const type_t &type, bool saturate) {
    return const_fold(cast_t::make(type, e, saturate));
}

bool is_const_broadcast(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return false;
    if (!shuffle->is_broadcast()) return false;
    return is_const(shuffle->vec[0]);
}

bool is_const_broadcast(const expr_t &e, const expr_t &value) {
    if (!is_const_broadcast(e)) return false;
    return e.as<shuffle_t>().vec[0].is_equal(value);
}

expr_t make_buffer(const std::string &name) {
    return var_t::make(type_t::byte(type::attr_t::ptr), name);
}

// Returns number of occurrences of `obj` in `root` (based on identity equality).
int count_object(const object_t &root, const object_t &obj) {
    if (!obj) return 0;

    std::vector<object_t> found;
    do {
#define HANDLE_IR_OBJECT(type) \
    if (obj.is<type>()) { \
        found = find_objects<type>(root); \
        break; \
    }

        HANDLE_CORE_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

        dsl_error() << obj;
    } while (false);

    int ret = 0;
    for (auto &f : found)
        if (f.is_equal(obj)) ret++;
    return ret;
}

bool contains_object(const object_t &root, const object_t &obj) {
    dsl_assert(is_var(obj)) << obj;
    return count_object(root, obj) > 0;
}

std::vector<stmt_t> find_stmt_groups(
        const object_t &root, const stmt_label_t &label) {
    auto groups = find_objects<stmt_group_t>(root);
    std::vector<stmt_t> ret;
    for (auto &g : groups) {
        if (g.as<stmt_group_t>().label == label) ret.emplace_back(g);
    }
    return ret;
}

stmt_t find_stmt_group(const object_t &root, const stmt_label_t &label) {
    auto groups = find_stmt_groups(root, label);
    if (groups.size() == 1)
        return groups[0];
    else
        return {};
}

class stmt_group_remover_t : public ir_mutator_t {
public:
    stmt_group_remover_t(stmt_label_t label) : label_(label) {}
    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.label == label_) return stmt_t();
        return ir_mutator_t::_mutate(obj);
    }
    stmt_label_t label_;
};

object_t remove_stmt_group(const object_t &root, stmt_label_t label) {
    stmt_group_remover_t remover(label);
    return remover.mutate(root);
}

stmt_t get_stmt_body(const stmt_t &stmt) {
    auto *alloc = stmt.as_ptr<alloc_t>();
    if (alloc) return alloc->body;

    auto *_for = stmt.as_ptr<for_t>();
    if (_for) return _for->body;

    auto *_while = stmt.as_ptr<while_t>();
    if (_while) return _while->body;

    auto *let = stmt.as_ptr<let_t>();
    if (let) return let->body;

    auto *group = stmt.as_ptr<stmt_group_t>();
    if (group) return group->body;

    return stmt;
}

stmt_t replace_stmt_body(const stmt_t &stmt, const stmt_t &new_body) {
    auto *alloc = stmt.as_ptr<alloc_t>();
    if (alloc) {
        return alloc_t::make(
                alloc->buf, alloc->size, alloc->kind, alloc->attrs, new_body);
    }

    auto *_for = stmt.as_ptr<for_t>();
    if (_for) {
        return for_t::make(_for->var, _for->init, _for->bound, new_body,
                _for->step, _for->unroll);
    }

    auto *_while = stmt.as_ptr<while_t>();
    if (_while) { return while_t::make(_while->cond, new_body); }

    auto *let = stmt.as_ptr<let_t>();
    if (let) { return let_t::make(let->var, let->value, new_body); }

    auto *group = stmt.as_ptr<stmt_group_t>();
    if (group) { return stmt_group_t::make(group->label, new_body); }

    return new_body;
}

class grf_usage_visitor_t : public ir_visitor_t {
public:
    grf_usage_visitor_t(int grf_size, int external_regs, bool skip_let)
        : grf_size_(grf_size), skip_let_(skip_let), regs_(external_regs) {}

    void _visit(const alloc_t &obj) override {
        auto guard = grf_usage_guard(register_size(obj, grf_size_));
        ir_visitor_t::_visit(obj);
    }

    void _visit(const let_t &obj) override {
        int size = skip_let_ ? 0 : register_size(obj);
        auto guard = grf_usage_guard(size);
        ir_visitor_t::_visit(obj);
    }

    int peak_regs() const { return peak_regs_; }

private:
    mem_usage_guard_t grf_usage_guard(int size) {
        auto ret = mem_usage_guard_t(&regs_, size);
        peak_regs_ = std::max(peak_regs_, regs_);
        return ret;
    }

    int grf_size_ = 0;
    bool skip_let_ = false;
    int regs_ = 0;
    int peak_regs_ = 0;
};

int get_peak_regs(
        const stmt_t &stmt, int grf_size, int external_regs, bool skip_let) {
    grf_usage_visitor_t visitor(grf_size, external_regs, skip_let);
    visitor.visit(stmt);
    return div_up(visitor.peak_regs(), grf_size);
}

bool relation_t::implies(const relation_t &other) const {
    dsl_assert(var().is_same(other.var()));

    if (op_kind() != other.op_kind()) return false;

    auto A = to_cpp<int64_t>(rhs());
    auto B = to_cpp<int64_t>(other.rhs());

    switch (op_kind()) {
        // (x > A) && (A >= B) => (x > B)
        // (x >= A) && (A >= B) => (x >= B)
        case op_kind_t::_gt:
        case op_kind_t::_ge: return A >= B;
        // (x < A) && (A <= B) => (x < B)
        // (x <= A) && (A <= B) => (x <= B)
        case op_kind_t::_lt:
        case op_kind_t::_le: return A <= B;
        default: dsl_error() << "Not implemented: " << expr_;
    }
    return false;
}

relation_t relation_t::transform(
        const linear_transform_t &t, const expr_t &new_var) const {
    dsl_assert(t.a == 1) << "Not implemented.";
    return relation_t(binary_op_t::make(op_kind(), new_var, rhs() + t.b));
}

expr_t relation_t::normalize(const expr_t &e) {
    dsl_assert(is_relation_constraint(e)) << e;
    auto &op = e.as<binary_op_t>();

    auto op_kind = op.op_kind;
    auto a = op.a;
    auto b = op.b;

    switch (op_kind) {
        case op_kind_t::_lt:
            op_kind = op_kind_t::_le;
            b -= 1;
            break;
        case op_kind_t::_gt:
            op_kind = op_kind_t::_ge;
            b += 1;
            break;
        default: return e;
    }
    return binary_op_t::make(op_kind, a, b);
}

bool modulus_info_t::is_modulus_constraint(const expr_t &e) {
    auto *binary_op = e.as_ptr<binary_op_t>();
    if (!binary_op) return false;
    if (!binary_op->b.is(0)) return false;
    if (binary_op->op_kind != op_kind_t::_eq) return false;

    auto *mod_op = binary_op->a.as_ptr<binary_op_t>();
    if (!mod_op) return false;
    if (mod_op->op_kind != op_kind_t::_mod) return false;
    if (!is_var(mod_op->a)) return false;
    if (!is_const(mod_op->b)) return false;

    return true;
}

int64_t bound_finder_base_t::find_bound_impl(
        const expr_t &e, bool is_low) const {
    int64_t def_bound = unlimited_bound(is_low);
    if (is_const(e)) return to_cpp<int64_t>(e);
    if (is_var(e)) return get_var_bound(e, is_low);

    auto *unary = e.as_ptr<unary_op_t>();
    if (unary) {
        dsl_assert(unary->op_kind == op_kind_t::_minus) << e;
        auto a = find_bound_impl(unary->a, !is_low);
        if (!is_good_bound(a)) return def_bound;
        return -a;
    }

    auto *binary = e.as_ptr<binary_op_t>();
    if (binary) {
        switch (binary->op_kind) {
            case op_kind_t::_add: {
                auto a = find_bound_impl(binary->a, is_low);
                auto b = find_bound_impl(binary->b, is_low);
                if (!is_good_bound(a) || !is_good_bound(b)) return def_bound;
                return a + b;
            }
            case op_kind_t::_sub: {
                auto a = find_bound_impl(binary->a, is_low);
                auto b = find_bound_impl(binary->b, !is_low);
                if (!is_good_bound(a) || !is_good_bound(b)) return def_bound;
                return a - b;
            }
            case op_kind_t::_mul: {
                auto a = binary->a;
                auto b = binary->b;
                if (!is_const(a) && is_const(b)) std::swap(a, b);
                if (!is_const(a)) return def_bound;

                auto a_const = to_cpp<int64_t>(a);
                if (a_const == 0) return 0;

                auto b_lo = find_low_bound(b);
                auto b_hi = find_high_bound(b);
                auto b_lo_ok = is_good_bound(b_lo);
                auto b_hi_ok = is_good_bound(b_hi);

                if ((a_const > 0) == is_low && b_lo_ok) return a_const * b_lo;
                if ((a_const > 0) != is_low && b_hi_ok) return a_const * b_hi;

                break;
            }
            case op_kind_t::_div: {
                if (!is_const(binary->b)) return def_bound;

                auto b = to_cpp<int64_t>(binary->b);
                dsl_assert(b != 0);

                auto a = find_bound_impl(binary->a, b > 0 ? is_low : !is_low);
                if (!is_good_bound(a)) return def_bound;

                bool is_neg = ((a > 0) && (b < 0)) || ((a < 0) && (b > 0));

                int64_t div_bound;
                if (is_low != is_neg) {
                    // Truncate away from zero.
                    div_bound = div_up(std::abs(a), std::abs(b));
                } else {
                    // Truncate towards zero.
                    div_bound = std::abs(a) / std::abs(b);
                }
                if (is_neg) div_bound *= -1;
                return div_bound;
            }
            case op_kind_t::_mod: {
                if (is_low) return 0;
                auto max_mod = find_bound_impl(binary->b, /*is_low=*/false);
                if (!is_good_bound(max_mod)) return def_bound;
                return max_mod - 1;
            }
            case op_kind_t::_and: {
                if (e.type().is_u16()) {
                    return is_low ? e.type().min<int64_t>()
                                  : e.type().max<int64_t>();
                }
                break;
            }
            case op_kind_t::_min:
            case op_kind_t::_max: {
                auto a = find_bound_impl(binary->a, is_low);
                auto b = find_bound_impl(binary->b, is_low);
                if (!is_good_bound(a) || !is_good_bound(b)) return def_bound;
                auto a_const = to_cpp<int64_t>(a);
                auto b_const = to_cpp<int64_t>(a);
                return binary->op_kind == op_kind_t::_min
                        ? std::min(a_const, b_const)
                        : std::max(a_const, b_const);
            }
            default: break;
        }
    }

    if (e.type().is_bool()) return is_low ? 0 : 1;
    auto *cast = e.as_ptr<cast_t>();
    if (cast) {
        // Saturate if needed, otherwise assume the same bounds.
        if (!cast->is_bool_vec_u16() && !cast->saturate)
            return find_bound_impl(cast->expr, is_low);

        if (is_low) {
            auto type_lo = cast->type.min<int64_t>();
            auto lo = find_low_bound(cast->expr);
            return std::max(type_lo, lo);
        }
        // Check u64 explicitly as its max doesn't fit into int64_t.
        if (cast->type.is_u64()) return find_bound_impl(cast->expr, is_low);
        auto type_hi = cast->type.max<int64_t>();
        auto hi = find_high_bound(cast->expr);
        return std::min(type_hi, hi);
    }

    return def_bound;
}

bool is_linear_var_transform(const expr_t &e, linear_transform_t &t) {
    if (is_var(e)) {
        t.x = e;
        t.a = 1;
        t.b = 0;
        return true;
    }

    auto *binary_op = e.as_ptr<binary_op_t>();
    if (!binary_op) return false;

    auto vars = find_objects<var_t>(e);
    if (vars.size() != 1) return false;

    auto &var = vars[0];

    // TODO: Extend to match multiplication: (a * var).
    if (!one_of(binary_op->op_kind, {op_kind_t::_add, op_kind_t::_sub}))
        return false;

    auto &a = binary_op->a;
    auto &b = binary_op->b;

    bool is_sub = (binary_op->op_kind == op_kind_t::_sub);

    // var op b -> (t.a = 1, t.b = +/-b)
    if (a.is_same(var) && is_const(b)) {
        t.x = var;
        t.a = 1;
        t.b = (is_sub ? -1 : 1) * to_cpp<int64_t>(b);
        return true;
    }

    // a op var -> (t.a = +/-1, t.b = a)
    if (is_const(a) && b.is_same(var)) {
        t.x = var;
        t.a = (is_sub ? -1 : 1);
        t.b = to_cpp<int64_t>(a);
        return true;
    }

    return false;
}

ir_context_t::ir_context_t(
        const kernel::options_t &options, const constraint_set_t &cset)
    : options_(options), cset_(cset) {
    for (auto &a : options_.assumptions()) {
        add_constraint(a);
    }
}

ir_context_t::ir_context_t(const kernel_t &kernel) : options_(kernel.options) {
    for (auto &a : options_.assumptions()) {
        add_constraint(a);
    }
    for (auto &var : find_objects<var_t>(kernel.body)) {
        all_names_.insert(var.as<var_t>().name);
    }
}

void ir_context_t::add_constraint(const expr_t &e) {
    cset_.add_constraint(e);
}

void constraint_set_t::add_constraint(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (shuffle) {
        if (shuffle->is_broadcast()) add_constraint(shuffle->vec[0]);
        return;
    }

    if (modulus_info_t::is_modulus_constraint(e)) {
        modulus_info_t mi(e);
        modulus_infos_[mi.var()].push_back(mi);
        return;
    }

    if (relation_t::is_relation_constraint(e)) {
        relation_t rel(e);
        relations_[rel.var()].push_back(rel);
        return;
    }

    auto *binary_op = e.as_ptr<binary_op_t>();

    // Propagate constraints from chained ands like a & b & c
    if (binary_op && binary_op->op_kind == op_kind_t::_and) {
        add_constraint(binary_op->a);
        add_constraint(binary_op->b);
        return;
    }

    // Propagate constraints from y for (x == y) equalities.
    if (binary_op && binary_op->op_kind == op_kind_t::_eq) {
        auto &a = binary_op->a;
        auto &b = binary_op->b;
        linear_transform_t t;
        if (is_var(a) && is_linear_var_transform(b, t)) {
            // Relations.
            auto r_it = relations_.find(t.x);
            if (r_it != relations_.end()) {
                for (auto &c : r_it->second) {
                    add_constraint(c.transform(t, a).expr());
                }
            }
            // Modulus.
            if (t.is_identity()) {
                auto m_it = modulus_infos_.find(t.x);
                if (m_it != modulus_infos_.end()) {
                    for (auto &c : m_it->second) {
                        add_constraint(substitute(c.expr(), b, a));
                    }
                }
            }
            return;
        }
    }
}

bool constraint_set_t::is_single_value(const expr_t &e, expr_t &value) const {
    dsl_assert(is_var(e)) << e;
    auto it = relations_.find(e);
    if (it == relations_.end()) return false;

    expr_t lo;
    expr_t hi;
    for (auto &rel : it->second) {
        dsl_assert(is_const(rel.rhs())) << rel.str();
        bool do_break = false;
        switch (rel.op_kind()) {
            case op_kind_t::_eq:
                lo = hi = rel.rhs();
                do_break = true;
                break;
            case op_kind_t::_ne:
                // Ignore for now as there is little benefit to implementing a
                // proper search.
                break;
            case op_kind_t::_ge:
            case op_kind_t::_gt: {
                auto cur_lo = (rel.op_kind() == op_kind_t::_ge ? rel.rhs()
                                                               : rel.rhs() + 1);
                if (lo.is_empty() || to_cpp<bool>(cur_lo > lo)) {
                    lo = std::move(cur_lo);
                }
                break;
            }
            case op_kind_t::_le:
            case op_kind_t::_lt: {
                auto cur_hi = (rel.op_kind() == op_kind_t::_le ? rel.rhs()
                                                               : rel.rhs() - 1);
                if (hi.is_empty() || to_cpp<bool>(cur_hi < hi)) {
                    hi = std::move(cur_hi);
                }
                break;
            }
            default: dsl_error() << rel.str();
        }
        if (do_break) break;
    }
    bool ret = lo && lo.is_equal(hi);
    if (ret) value = std::move(lo);
    return ret;
}

bool constraint_set_t::can_prove_impl(
        const expr_t &_e, bool do_simplify) const {
    auto e = _e;
    if (is_const(e)) {
        dsl_assert(e.type() == type_t::_bool()) << e;
        return to_cpp<bool>(e);
    }

    if (do_simplify) {
        // These passes for comparison help to prove more inequalities.
        e = simplify_cmp_move_const_to_rhs(e);
        e = simplify_cmp_reduce_lhs_rhs(e);
        e = simplify(e);
        if (is_const(e)) {
            dsl_assert(e.type() == type_t::_bool()) << e;
            return to_cpp<bool>(e);
        }
    }

    if (modulus_info_t::is_modulus_constraint(e)) return can_prove_modulus(e);
    if (relation_t::is_relation_constraint(e)) return can_prove_relation(e);

    // Try to estimate bounds for compound relation.
    if (try_prove_compound_relation(e)) return true;

    // Can't prove.
    return false;
}

int constraint_set_t::max_proven_gcd(const expr_t &var) const {
    auto it = modulus_infos_.find(var);
    if (it == modulus_infos_.end()) return 1;
    int ret = 1;
    for (auto &c : it->second) {
        ret = lcm(ret, to_cpp<int>(c.mod()));
    }
    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
