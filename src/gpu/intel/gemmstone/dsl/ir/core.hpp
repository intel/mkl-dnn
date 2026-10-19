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

#ifndef GEMMSTONE_DSL_IR_CORE_HPP
#define GEMMSTONE_DSL_IR_CORE_HPP

#include <algorithm>
#include <cstdio>
#include <string>

#include "gemmstone/../../dsl/dsl_impl.hpp"
#include "gemmstone/../../dsl/utils/utils.hpp"

// All IR expression objects.
#define HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_IR_OBJECT(binary_op_t) \
    HANDLE_IR_OBJECT(bool_imm_t) \
    HANDLE_IR_OBJECT(cast_t) \
    HANDLE_IR_OBJECT(const_var_t) \
    HANDLE_IR_OBJECT(float_imm_t) \
    HANDLE_IR_OBJECT(iif_t) \
    HANDLE_IR_OBJECT(int_imm_t) \
    HANDLE_IR_OBJECT(linear_t) \
    HANDLE_IR_OBJECT(load_t) \
    HANDLE_IR_OBJECT(ptr_t) \
    HANDLE_IR_OBJECT(shuffle_t) \
    HANDLE_IR_OBJECT(ternary_op_t) \
    HANDLE_IR_OBJECT(unary_op_t) \
    HANDLE_IR_OBJECT(var_t) \
    HANDLE_IR_OBJECT(ref_t)

// All IR statement objects.
#define HANDLE_STMT_IR_OBJECTS() \
    HANDLE_IR_OBJECT(alloc_t) \
    HANDLE_IR_OBJECT(assign_t) \
    HANDLE_IR_OBJECT(for_t) \
    HANDLE_IR_OBJECT(func_call_t) \
    HANDLE_IR_OBJECT(if_t) \
    HANDLE_IR_OBJECT(let_t) \
    HANDLE_IR_OBJECT(stmt_group_t) \
    HANDLE_IR_OBJECT(stmt_seq_t) \
    HANDLE_IR_OBJECT(store_t) \
    HANDLE_IR_OBJECT(while_t)

#define HANDLE_CORE_IR_OBJECTS() \
    HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_STMT_IR_OBJECTS()

// Defines getter for a function argument.
#define IR_DEFINE_ARG_GET(name, index) \
    static const expr_t &arg_##name(const func_call_t &c) { \
        dsl_assert(c.func.is<self_type>()) << c; \
        return c.args[index]; \
    } \
    static const expr_t &arg_##name(const stmt_t &s) { \
        dsl_assert(s.is<func_call_t>()) << s; \
        auto &c = s.as<func_call_t>(); \
        return arg_##name(c); \
    } \
    template <typename T> \
    static T &arg_##name(std::vector<T> &args) { \
        return args[index]; \
    } \
    template <typename T> \
    static const T &arg_##name(const std::vector<T> &args) { \
        return args[index]; \
    }

GEMMSTONE_NAMESPACE_START
namespace dsl {

bool is_subset(const type_t &a, const type_t &b);

namespace ir {

template <typename T>
type_t from_cpp() {
#define CASE(cpp_type, type) \
    if (std::is_same<T, cpp_type>::value) return type_t::type()

    CASE(bool, _bool);
    CASE(float, f32);
    CASE(double, f64);
    CASE(int16_t, s16);
    CASE(int32_t, s32);
    CASE(int64_t, s64);
    CASE(uint16_t, u16);
    CASE(uint32_t, u32);
    CASE(uint64_t, u64);

#undef CASE

    stub();

    return type_t::undef();
}

template <typename T>
bool is_cpp(const type_t &t) {
    return t == from_cpp<T>();
}

// clang-tidy doesn't like the semicolon next to the class name.
#define CLASS_DECLARATION(name) class name
#define HANDLE_IR_OBJECT(type) CLASS_DECLARATION(type);
HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
#undef CLASS_DECLARATION

// Helper classes for containers to store object_t.
struct object_id_hash_t {
    size_t operator()(const object_t &obj) const {
        return std::hash<const object::impl_t *>()(obj.impl());
    }
};

struct object_eq_hash_t {
    size_t operator()(const object_t &obj) const { return obj.get_hash(); }
};

struct object_id_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_same(b);
    }
};

struct object_eq_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_equal(b);
    }
};

// Containers to store object_t.

// Unordered set, uses identity comparison for keys.
template <typename KeyT>
using object_set_t
        = std::unordered_set<KeyT, object_id_hash_t, object_id_equal_t>;

// Unordered set, uses equality comparison for keys.
template <typename KeyT>
using object_eq_set_t
        = std::unordered_set<KeyT, object_eq_hash_t, object_eq_equal_t>;

// Unordered map, uses identity comparison for keys.
template <typename KeyT, typename ValueT>
using object_map_t
        = std::unordered_map<KeyT, ValueT, object_id_hash_t, object_id_equal_t>;

// Unordered map, uses equality comparison for keys.
template <typename KeyT, typename ValueT>
using object_eq_map_t
        = std::unordered_map<KeyT, ValueT, object_eq_hash_t, object_eq_equal_t>;

// Helper class to mutate IR tree.
class ir_mutator_t {
public:
    using impl_t = object::impl_t;
    virtual ~ir_mutator_t() = default;

    object_t mutate(const object_t &obj) {
        auto impl = obj.impl();
        if (!impl) return impl;
        return impl->_mutate(*this);
    }

    template <typename T>
    std::vector<T> mutate(const std::vector<T> &v) {
        std::vector<T> new_v;
        new_v.reserve(v.size());
        for (auto &e : v)
            new_v.push_back(mutate(e));
        return new_v;
    }

    // To catch missing _mutate() handlers in ir_mutator_t.
    object_t _mutate(const impl_t &obj) {
        dsl_error() << "Can't handle type: " << object_t(&obj);
        return {};
    }

#define HANDLE_IR_OBJECT(type) virtual object_t _mutate(const type &obj);
    HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
};

// Helper class to walk through IR tree.
class ir_visitor_t {
public:
    using impl_t = object::impl_t;
    virtual ~ir_visitor_t() = default;

    void visit(const object_t &obj) {
        const impl_t *impl = obj.impl();
        if (impl) {
            pre_visit(*impl);
            impl->_visit(*this);
            post_visit(*impl);
        };
    }

    template <typename T>
    void visit(const std::vector<T> &v) {
        for (auto &e : v)
            visit(e);
    }

    virtual void pre_visit(const impl_t &obj) {}
    virtual void post_visit(const impl_t &obj) {}

    // To catch missing _visit() handlers in ir_visitor_t.
    void _visit(const impl_t &obj) {
        dsl_error() << "Can't handle type: " << object_t(obj);
    }

#define HANDLE_IR_OBJECT(type) virtual void _visit(const type &obj);
    HANDLE_CORE_IR_OBJECTS()
#undef HANDLE_IR_OBJECT
};

// Base class for IR expression objects.
class expr_impl_t : public object::impl_t {
public:
    expr_impl_t(object::impl_t::info_t type_info, const type_t &type)
        : object::impl_t(type_info), type(type) {}

    type_t type;
};

template <typename T>
struct expr_iface_t : public expr_impl_t, public object::info_t<T> {
    expr_iface_t(const type_t &type) : expr_impl_t(T::get_info(), type) {}

    bool is_equal(const object::impl_t &obj) const override {
        if (!obj.is<T>()) return false;
        return (*static_cast<const T *>(this) == obj.as<T>());
    }

    object_t _mutate(ir_mutator_t &mutator) const override {
        return mutator._mutate(*static_cast<const T *>(this));
    }
    void _visit(ir_visitor_t &visitor) const override {
        visitor._visit(*static_cast<const T *>(this));
    }
};

inline const type_t &expr_t::type() const {
    dsl_assert(!is_empty());
    return ((const expr_impl_t *)impl())->type;
}

// Helper functions.
inline bool is_var(const expr_t &e);
inline bool is_ref(const expr_t &e);
inline bool all_of(const expr_t &e, const expr_t &value);

// Unary and binary operators.
enum class op_kind_t : int {
    undef = (int)op::kind_t::undef,
    _add = (int)op::add,
    _sub = (int)op::sub,
    _mul = (int)op::mul,
    _div = (int)op::div,
    _mod = (int)op::mod,
    _min = (int)op::min,
    _max = (int)op::max,

    // Parametric ReLU.
    // if (a > 0) op = a
    // else       op = a * b
    _prelu = (int)op::prelu,

    // Operations internal to the IR (not exposed in op::kind_t)
    _minus,
    _exp,
    _shl,
    _shr,

    _lt,
    _le,
    _gt,
    _ge,
    _ne,
    _eq,

    _and,
    _or,
    _xor,

    // Ternary operations.
    // Ternary add.
    // op = a + b + c
    _add3,
    // Multiply-accumulate.
    // op = a + b * c
    _mad,
    // Integer division by a constant with rounding up.
    // if (a >= 1) op = 1 + (a - 1) / b
    // else        op = 0
    _div_up,
    // Integer division by a non-constant (rounding down behavior).
    // if (a % b < 0) op = a / b - 1
    // else           op = a / b
    // This is ternary operation, c is a pre-computed value.
    _idiv,
    // Integer modulus by a non-constant (rounding down behavior).
    // if (a % b < 0) op = a % b + b
    // else           op = a % b
    // This is ternary operation, c is a pre-computed value.
    _imod,
};

std::string to_string(op_kind_t kind);

inline std::ostream &operator<<(std::ostream &out, op_kind_t kind) {
    out << to_string(kind);
    return out;
}

bool is_cmp_op(op_kind_t op_kind);

bool is_commutative_op(op_kind_t op_kind);

op_kind_t negate_cmp_op(op_kind_t op_kind);

type_t unary_op_type(op_kind_t op_kind, const expr_t &a);

type_t common_int_type(const type_t &_a, const type_t &_b);

type_t common_type(const type_t &base, const type_t &a, const type_t &b);

type_t common_type(const type_t &a, const type_t &b);

type_t common_type(const expr_t &a, const expr_t &b);

type_t binary_op_type(op_kind_t op_kind, const expr_t &a, const expr_t &b);

type_t ternary_op_type(
        op_kind_t op_kind, const expr_t &a, const expr_t &b, const expr_t &c);

type_t nary_op_type(op_kind_t op_kind, const std::vector<expr_t> &args);

// Binary operation: (a op b).
class binary_op_t : public expr_iface_t<binary_op_t> {
public:
    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
        return expr_t(new binary_op_t(op_kind, a, b));
    }

    bool operator==(const binary_op_t &other) const {
        return (op_kind == other.op_kind)
                && ((a.is_equal(other.a) && b.is_equal(other.b))
                        || (is_commutative_op(op_kind) && b.is_equal(other.a)
                                && a.is_equal(other.b)));
    }

    size_t compute_hash() const override {
        if (is_commutative_op(op_kind)) {
            size_t a_hash = hash(a);
            size_t b_hash = hash(b);
            return hash(op_kind, a_hash ^ b_hash);
        }
        return hash(op_kind, a, b);
    }

    op_kind_t op_kind;
    expr_t a;
    expr_t b;

private:
    binary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b)
        : expr_iface_t(binary_op_type(op_kind, a, b))
        , op_kind(op_kind)
        , a(a)
        , b(b) {}
};

// Boolean immediate value.
class bool_imm_t : public expr_iface_t<bool_imm_t> {
public:
    friend class expr_t;

    static expr_t make(bool value) { return expr_t(new bool_imm_t(value)); }

    static type_t get_packed_type(int elems) {
        return type_t::u(std::max(elems, 16));
    }

    bool operator==(const bool_imm_t &other) const {
        return value == other.value;
    }

    size_t compute_hash() const override { return hash(value); }

    bool value;

private:
    bool_imm_t(bool value) : expr_iface_t(type_t::_bool()), value(value) {}
};

// Cast between data types. In general conversion follows the C++ casting
// rules. Several modes/scenarios are supported:
// - Cast with saturation: cast(T, e) = max(T_min, min(T_max, e))
//   By default saturation is disabled and any underflow/overflow is unhandled.
// - Bitwise cast from bool vector to u16 (boolxN -> u16, 2 <= N <= 16):
//   In this case the lower N bits of the resulting value are initialized based
//   on the boolean elements. The upper (16 - N) bits are uninitialized.
class cast_t : public expr_iface_t<cast_t> {
public:
    static expr_t make(
            const type_t &type, const expr_t &expr, bool saturate = false) {
        if (expr.type() == type) return expr;
        if (!saturate) {
            auto *expr_cast = expr.as_ptr<cast_t>();
            if (expr_cast && !expr_cast->saturate
                    && type == expr_cast->expr.type())
                return expr_cast->expr;
        }
        return expr_t(new cast_t(type, expr, saturate));
    }

    bool operator==(const cast_t &other) const {
        return type == other.type && expr.is_equal(other.expr)
                && (saturate == other.saturate);
    }

    size_t compute_hash() const override { return hash(type, expr, saturate); }

    bool is_bool_vec_u16() const {
        if (is_bool_vec(expr.type()) && is_u16_or_u32_scalar(type)) return true;
        if (is_bool_vec(type) && is_u16_or_u32_scalar(expr.type())) return true;
        return false;
    }

    expr_t expr;
    bool saturate;

private:
    cast_t(const type_t &type, const expr_t &expr, bool saturate)
        : expr_iface_t(type), expr(expr), saturate(saturate) {
        if (!is_bool_vec_u16()) {
            dsl_assert(type.elems() == expr.type().elems())
                    << "Number of elements must match.";
        }
    }

    static bool is_bool_vec(const type_t &type) {
        return type.is_bool() && type.elems() > 1;
    }

    static bool is_u16_or_u32_scalar(const type_t &type) {
        return (type.is_u16() || type.is_u32()) && type.is_scalar();
    }
};

// Constant variable, used as a coefficient in a linear expression.
class const_var_t : public expr_iface_t<const_var_t> {
public:
    static expr_t make(const type_t &type, const std::string &name) {
        return expr_t(new const_var_t(type, name));
    }

    bool operator==(const const_var_t &other) const { return this == &other; }

    size_t compute_hash() const override { return hash(name); }

    std::string name;

private:
    const_var_t(const type_t &type, const std::string &name)
        : expr_iface_t(type), name(name) {}
};

// Floating-point immediate value.
class float_imm_t : public expr_iface_t<float_imm_t> {
public:
    friend class expr_t;

    static expr_t make(double value, const type_t &type = type_t::undef()) {
        return expr_t(new float_imm_t(value, type));
    }

    bool operator==(const float_imm_t &other) const {
        return type == other.type && (value == other.value);
    }

    size_t compute_hash() const override { return hash(value); }

    double value;

private:
    float_imm_t(double value, const type_t &type = type_t::undef())
        : expr_iface_t(type.is_undef() ? type_t::f32() : type), value(value) {}
};

// Integer immediate value.
class int_imm_t : public expr_iface_t<int_imm_t> {
public:
    friend class expr_t;

    template <typename T>
    static expr_t make(T value, const type_t &type = type_t::undef()) {
        return expr_t(new int_imm_t(value, type));
    }

    bool operator==(const int_imm_t &other) const {
        return type == other.type && (value == other.value);
    }

    size_t compute_hash() const override { return hash(value); }

    static expr_t shrink_type(const expr_t &e) {
        auto &imm = e.as<int_imm_t>();
        type_t new_type = shrink_type(imm.value);
        if (new_type == imm.type) return e;
        return make(imm.value, new_type);
    }

    template <typename T>
    static bool try_shrink_type(int64_t v) {
        if ((v >= 0 && (uint64_t)v <= (uint64_t)std::numeric_limits<T>::max())
                || (v < 0
                        && (int64_t)v
                                >= (int64_t)std::numeric_limits<T>::min()))
            return true;
        return false;
    }

    int64_t value;

private:
    int_imm_t(int64_t value, const type_t &type = type_t::undef())
        : expr_iface_t(type.is_undef() ? shrink_type(value) : type)
        , value(value) {}

    static type_t shrink_type(int64_t v) {
        if (try_shrink_type<int32_t>(v)) return type_t::s32();
        return type_t::s64();
    }
};

// Immediate if or the conditional (ternary) operator.
// C++ equivalent: (cond ? true_expr : false_expr).
class iif_t : public expr_iface_t<iif_t> {
public:
    static expr_t make(const expr_t &cond, const expr_t &true_expr,
            const expr_t &false_expr) {
        return expr_t(new iif_t(cond, true_expr, false_expr));
    }

    bool operator==(const iif_t &other) const {
        return cond.is_equal(other.cond) && true_expr.is_equal(other.true_expr)
                && false_expr.is_equal(other.false_expr);
    }

    size_t compute_hash() const override {
        return hash(cond, true_expr, false_expr);
    }

    expr_t cond;
    expr_t true_expr;
    expr_t false_expr;

private:
    iif_t(const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr)
        : expr_iface_t(
                  iif_type(cond.type(), true_expr.type(), false_expr.type()))
        , cond(cond)
        , true_expr(true_expr)
        , false_expr(false_expr) {}

    static type_t iif_type(
            const type_t &cond, const type_t &T, const type_t &F) {
        auto type = common_type(T, F);
        return common_type(type, type, cond);
    }
};

// Linear combination expression:
//   u[0] * v[0] + u[1] * v[1] + ... u[n - 1] * v[n - 1] + c,
// where:
// - c/u[i] is either an integer immediate (int_imm_t) or a constant variable
//  (const_var_t)
// - v[i] is a non-constant variable (var_t)
class linear_t : public expr_iface_t<linear_t> {
public:
    static expr_t make(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec) {
        return expr_t(new linear_t(c, u_vec, v_vec));
    }
    static expr_t make(const expr_t &c) { return make(c, {}, {}); }
    static expr_t make(const expr_t &c, const std::vector<expr_t> &v_vec) {
        std::vector<expr_t> ones(v_vec.size(), expr_t(1));
        return make(c, ones, v_vec);
    }
    static expr_t to_expr(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec) {
        auto e = linear_t::make(c, u_vec, v_vec);
        return e.as<linear_t>().to_expr();
    }
    int nargs() const { return int(v_vec.size()); }
    expr_t to_expr() const;

    bool operator==(const linear_t &other) const {
        return c.is_equal(other.c) && utils::is_equal(u_vec, other.u_vec)
                && utils::is_equal(v_vec, other.v_vec);
    }

    size_t compute_hash() const override { return hash(c, u_vec, v_vec); }

    expr_t c;
    std::vector<expr_t> u_vec;
    std::vector<expr_t> v_vec;

private:
    linear_t(const expr_t &c, const std::vector<expr_t> &u_vec,
            const std::vector<expr_t> &v_vec)
        : expr_iface_t(type_t::s32()), c(c), u_vec(u_vec), v_vec(v_vec) {}
};

// Updates `base_expr` and `off` so that after return:
// - base_expr contains a variable of a pointer type
// - off contains an offset
void normalize_ptr(const type_t &type, expr_t &base, expr_t &off);

// Load from a GRF buffer.
// C++ equivalent (when type is scalar):
//     load = *(type *)(&buf[off]);
// C++ equivalent (when type is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         load[i] = *(scalar_type *)(&buf[off + i * _stride]);
//     }
class load_t : public expr_iface_t<load_t> {
public:
    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of type.scalar() elements).
    static expr_t make(const type_t &type, const expr_t &buf, const expr_t &off,
            int stride = default_stride) {
        return expr_t(new load_t(type, buf, off, stride));
    }

    bool operator==(const load_t &other) const {
        return type == other.type && buf.is_equal(other.buf)
                && off.is_equal(other.off) && (stride == other.stride);
    }

    size_t compute_hash() const override {
        return hash(type, buf, off, stride);
    }

    bool has_default_stride() const { return stride == default_stride; }

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    int stride;

private:
    load_t(const type_t &_type, const expr_t &_buf, const expr_t &_off,
            int _stride)
        : expr_iface_t(_type), buf(_buf), off(_off), stride(_stride) {
        normalize_ptr(type, buf, off);
        dsl_assert(is_var(buf) || is_ref(buf)) << buf;
        if (stride == type.scalar().size()) stride = default_stride;
    }
};

// Pointer expression: (base_ptr + off).
class ptr_t : public expr_iface_t<ptr_t> {
public:
    // off - offset in elements of the base type.
    static expr_t make(const expr_t &base, const expr_t &off) {
        return expr_t(new ptr_t(base, off));
    }

    bool operator==(const ptr_t &other) const {
        return base.is_equal(other.base) && off.is_equal(other.off);
    }

    size_t compute_hash() const override { return hash(base, off); }

    // Normalizes (base op off) pointer so that the new base is a variable and
    // off is an offset expression.
    // Example:
    //     Before call: base = (base0 + off0), off = off1
    //     After call:  base = base0, off = off0 + off1
    static void normalize(
            expr_t &base, expr_t &off, op_kind_t op_kind = op_kind_t::_add);

    expr_t base;
    expr_t off;

private:
    ptr_t(const expr_t &base, const expr_t &off)
        : expr_iface_t(ptr_type(base.type(), off.type()))
        , base(base)
        , off(off) {
        normalize(this->base, this->off);
    }

    static type_t ptr_type(type_t base, const type_t &off) {
        if (!base.is_ptr()) {
            if (!base.is_slm())
                dsl_assert(off.is_scalar())
                        << "Pointer to register buffer can be scalar only.";
            // base is not a pointer -> treat it as a pointer to the base.
            base = base.scalar().with_ptr();
        }
        int elems = std::max(base.elems(), off.elems());
        dsl_assert(one_of(base.elems(), {1, elems}));
        dsl_assert(one_of(off.elems(), {1, elems}));
        return common_type(base, base, off);
    }
};

inline const expr_t &get_base(const expr_t &e) {
    if (e.is_empty()) return e;
    if (e.is<var_t>()) return e;
    if (e.is<ptr_t>()) return e.as<ptr_t>().base;
    dsl_error() << e;
    return e;
}

class shuffle_t : public expr_iface_t<shuffle_t> {
public:
    static expr_t make(const expr_t &vec_expr, const std::vector<int> &idx) {
        return make(std::vector<expr_t> {vec_expr}, idx);
    }

    static expr_t make(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        check_indices(vec, idx);
        if (!vec[0].type().is_simd() && idx.size() == 1) return vec[idx[0]];
        return expr_t(new shuffle_t(vec, idx));
    }

    static expr_t make(
            const std::vector<expr_t> &_vec, bool find_equal = true) {
        std::vector<expr_t> vec;
        std::vector<int> idx;
        for (auto &v : _vec) {
            bool found = false;
            int size = int(vec.size());
            if (find_equal) {
                for (int i = 0; i < size; i++) {
                    if (v.is_equal(vec[i])) {
                        idx.push_back(i);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                vec.push_back(v);
                idx.push_back(size);
            }
        }
        return make(vec, idx);
    }

    static expr_t make_broadcast(const expr_t &expr, int64_t elems) {
        if (elems == 1) return expr;
        dsl_assert(expr.type().is_scalar()) << expr;
        return make({expr}, std::vector<int>(elems, 0));
    }

    // Slices the existing shuffle expression. For inputs (S, beg, end) returns
    // (S[beg], S[beg + 1], ..., S[end - 1]) vector.
    static expr_t make(const expr_t &_shuffle, int beg, int end) {
        auto &shuffle = _shuffle.as<shuffle_t>();
        dsl_assert(beg >= 0 && beg <= shuffle.elems());
        dsl_assert(end >= 0 && end <= shuffle.elems());
        dsl_assert(beg < end);
        std::vector<expr_t> vec;
        std::vector<int> idx(end - beg, -1);
        for (int i = beg; i < end; i++) {
            if (idx[i - beg] != -1) continue;
            int old_idx = shuffle.idx[i];
            vec.push_back(shuffle.vec[old_idx]);
            for (int j = i; j < end; j++) {
                if (shuffle.idx[j] == old_idx)
                    idx[j - beg] = int(vec.size()) - 1;
            }
        }
        return make(vec, idx);
    }

    bool operator==(const shuffle_t &other) const {
        return utils::is_equal(vec, other.vec)
                && utils::is_equal(idx, other.idx);
    }

    size_t compute_hash() const override { return hash(vec, idx); }

    int elems() const { return int(idx.size()); }

    bool is_vector() const {
        for (int i = 0; i < elems(); i++)
            if (idx[i] != i) return false;
        return true;
    }

    bool is_broadcast() const {
        return !vec[0].type().is_simd() && vec.size() == 1;
    }

    std::vector<expr_t> vec;
    std::vector<int> idx;

private:
    shuffle_t(const std::vector<expr_t> &vec, const std::vector<int> &idx)
        : expr_iface_t(shuffle_type(vec, idx)), vec(vec), idx(idx) {}

    static void check_indices(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        dsl_assert(!vec.empty() && !idx.empty());
        bool is_simd = (vec.size() == 1 && vec[0].type().is_simd());
        for (auto &v : vec) {
            dsl_assert(v.type().is_simd() == is_simd);
        }
        int elems = (is_simd ? vec[0].type().elems() : (int)vec.size());
        for (int i : idx) {
            dsl_assert(i >= 0 && i < elems);
        }
    }

    static type_t shuffle_type(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        auto elem_type = vec[0].type();
        if (vec.size() == 1 && elem_type.is_simd()) {
            dsl_assert(idx.size() == 1);
            return elem_type.scalar();
        }

        for (auto &v : vec)
            elem_type = common_type(elem_type, v.type());

        for (size_t i = 0; i < idx.size(); i++) {
            dsl_assert(idx[i] >= 0 && idx[i] < int(vec.size()))
                    << "Incorrect index.";
            maybe_unused(i);
        }

        int elems = int(idx.size());
        return elem_type.with_elems(elems);
    }
};

// Ternary operation: op(a, b, c).
class ternary_op_t : public expr_iface_t<ternary_op_t> {
public:
    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c) {
        return expr_t(new ternary_op_t(op_kind, a, b, c));
    }

    bool operator==(const ternary_op_t &other) const {
        return (op_kind == other.op_kind) && a.is_equal(other.a)
                && b.is_equal(other.b) && c.is_equal(other.c);
    }

    size_t compute_hash() const override { return hash(op_kind, a, b, c); }

    op_kind_t op_kind;
    expr_t a;
    expr_t b;
    expr_t c;

private:
    ternary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c)
        : expr_iface_t(ternary_op_type(op_kind, a, b, c))
        , op_kind(op_kind)
        , a(a)
        , b(b)
        , c(c) {}
};

inline expr_t ternary_mad(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_mad, a, b, c);
}

inline expr_t ternary_add3(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_add3, a, b, c);
}

inline expr_t ternary_idiv(
        const expr_t &a, const expr_t &b, const expr_t &magic) {
    return ternary_op_t::make(op_kind_t::_idiv, a, b, magic);
}

// Unary operation: (op a).
class unary_op_t : public expr_iface_t<unary_op_t> {
public:
    static expr_t make(op_kind_t op_kind, const expr_t &a) {
        return expr_t(new unary_op_t(op_kind, a));
    }

    bool operator==(const unary_op_t &other) const {
        return (op_kind == other.op_kind) && a.is_equal(other.a);
    }

    size_t compute_hash() const override { return hash(op_kind, a); }

    op_kind_t op_kind;
    expr_t a;

private:
    unary_op_t(op_kind_t op_kind, const expr_t &a)
        : expr_iface_t(unary_op_type(op_kind, a)), op_kind(op_kind), a(a) {}
};

class var_t : public expr_iface_t<var_t> {
public:
    static expr_t make(const type_t &type, const std::string &name,
            bool is_mutable = false) {
        return expr_t(new var_t(type, name, is_mutable));
    }

    bool operator==(const var_t &other) const {
        // Do not allow variable cloning.
        return this == &other;
    }

    size_t compute_hash() const override {
        return std::hash<std::string> {}(name);
    }

    std::string name;
    bool is_mutable = false;

private:
    var_t(const type_t &type, const std::string &name, bool is_mutable)
        : expr_iface_t(type), name(name), is_mutable(is_mutable) {}
};

const std::string &local_id_name(int idx);
const std::string &local_size_name(int idx);
const std::string &group_id_name(int idx);
const std::string &group_subgroup_count_name();
const std::string &subgroup_id_name(int idx);
const std::string &subgroup_local_id_name();
const std::string &subgroup_linear_id_name();

// Deprecated
const std::string &tg_idx_name(int idx);
const std::string &thr_idx_name(int idx);

// Index into a buffer
// off is offset in number of elements
// elems is number of consecutive elements to access starting from off
// off and elems must be GRF aligned
class ref_t : public expr_iface_t<ref_t> {
public:
    static expr_t make(const expr_t &var, int off, int elems) {
        return expr_t(new ref_t(var, off, elems));
    }

    bool operator==(const ref_t &other) const {
        return other.var.is_equal(var) && other.off == off
                && other.elems == elems;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << var << "[" << off;
        if (elems > 1) oss << ":" << off + elems;
        oss << "]";
        return oss.str();
    }

    size_t compute_hash() const override { return hash(var, off, elems); }

    expr_t var;
    int off;
    int elems;

private:
    ref_t(const expr_t &var, int off, int elems)
        : expr_iface_t(var.type().with_elems(elems))
        , var(var)
        , off(off)
        , elems(elems) {
        dsl_assert(off >= 0) << "Invalid offset: " << off;
        dsl_assert(elems > 0) << "Invalid elems: " << elems;
        dsl_assert(off + elems <= var.type().elems())
                << "Incompatible (off, elems): (" << off << ", " << elems
                << "), the base type: " << var.type();
        normalize();
    }

    void normalize() {
        if (var.is<var_t>()) return;
        auto *ref = var.as_ptr<ref_t>();
        dsl_assert(ref) << "Expected var or ref, got: " << var;
        var = ref->var;
        off += ref->off;
    }
};

// Convertor from C++ type to IR expression.
template <typename T>
expr_t to_expr(T value, const type_t &type) {
#define CASE(ir_type, cpp_type) \
    if (type == type_t::ir_type()) return expr_t(static_cast<cpp_type>(value))

    CASE(_bool, bool);
    CASE(f32, float);
    CASE(f64, double);
    CASE(s16, int16_t);
    CASE(s32, int32_t);
    CASE(s64, int64_t);
    CASE(u16, uint16_t);
    CASE(u32, uint32_t);
    CASE(u64, uint64_t);

#undef CASE

    dsl_error() << type;

    return expr_t();
}

template <typename T>
expr_t to_expr(T value) {
    return to_expr(value, from_cpp<T>());
}

inline bool is_binary_op(const expr_t &e) {
    return e.is<binary_op_t>();
}

inline bool is_binary_op(const expr_t &e, op_kind_t op_kind) {
    if (!is_binary_op(e)) return false;
    return e.as<binary_op_t>().op_kind == op_kind;
}

inline bool is_binary_cmp_op(const expr_t &e) {
    if (!is_binary_op(e)) return false;
    return is_cmp_op(e.as<binary_op_t>().op_kind);
}

inline bool all_of(const expr_t &e, const expr_t &value) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return e.is_equal(value);
    for (auto &i : shuffle->idx) {
        if (!shuffle->vec[i].is_equal(value)) return false;
    }
    return true;
}

inline bool is_shuffle_const(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return false;
    for (auto &v : shuffle->vec)
        if (!is_const(v)) return false;
    return true;
}

inline bool is_var(const expr_t &e) {
    return e.is<var_t>();
}

inline bool is_ref(const expr_t &e) {
    return e.is<ref_t>();
}

// Convertor from IR expression to C++ constant.
template <typename T>
T to_cpp(const expr_t &e) {
    dsl_assert(is_const(e)) << "Expression must be constant.";

    if (e.is<int_imm_t>()) return (T)e.as<int_imm_t>().value;
    if (e.is<float_imm_t>()) return (T)e.as<float_imm_t>().value;
    if (e.is<bool_imm_t>()) return (T)e.as<bool_imm_t>().value;

    stub();
    return 0;
}

inline int to_int(const expr_t &e) {
    return to_cpp<int>(e);
}

// Returns a shifted pointer with base `a` (pointer) and offset `b` (in elements).
// shift_ptr(op, a, b) returns &(a op b) in C++ terms (op is either addition or
// subtraction).
expr_t shift_ptr(op_kind_t op_kind, const expr_t &a, const expr_t &b);

// Base class for IR statement objects.
class stmt_impl_t : public object::impl_t {
public:
    stmt_impl_t(object::impl_t::info_t type_info) : object::impl_t(type_info) {}
};
template <typename T>
struct stmt_iface_t : public stmt_impl_t, public object::info_t<T> {
    using self_type = T;
    stmt_iface_t() : stmt_impl_t(T::get_info()) {}

    bool is_equal(const object::impl_t &obj) const override {
        if (!obj.is<T>()) return false;
        return (*static_cast<const T *>(this) == obj.as<T>());
    }

    object_t _mutate(ir_mutator_t &mutator) const override {
        return mutator._mutate(*static_cast<const T *>(this));
    }
    void _visit(ir_visitor_t &visitor) const override {
        visitor._visit(*static_cast<const T *>(this));
    }
};

enum class alloc_kind_t {
    undef,
    grf, // GRF - general register file.
    slm, // SLM - shared local memory.
    global, // Global memory.
};

class alloc_attr_impl_t : public object::impl_t {
public:
    alloc_attr_impl_t(object::impl_t::info_t type_info)
        : object::impl_t(type_info) {}
};

class alloc_attr_t : public object_t {
public:
    using object_t::object_t;

    alloc_attr_t() = default;
    alloc_attr_t(const object_t &obj) : object_t(obj) {}
    alloc_attr_t(object_t &&obj) : object_t(obj) {}
    alloc_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    alloc_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }
};

// Allocation attribute to store extra information to avoid bank conflicts.
class bank_conflict_attr_t : public alloc_attr_impl_t,
                             public object::info_t<bank_conflict_attr_t> {
public:
    static alloc_attr_t make(const std::vector<expr_t> &bufs,
            const std::vector<int> &buf_sizes,
            const std::vector<int> &buf_min_block_sizes,
            const std::vector<stmt_t> &instructions) {
        return alloc_attr_t(new bank_conflict_attr_t(
                bufs, buf_sizes, buf_min_block_sizes, instructions));
    }

    bool is_equal(const object::impl_t &obj) const override {
        return this == &obj;
    }

    size_t compute_hash() const override { return hash(buf_sizes); }

    // List of buffers accessed from instructions.
    std::vector<expr_t> bufs;
    // Buffer sizes in bytes.
    std::vector<int> buf_sizes;
    // Minimum power-of-two block sizes for each buffer to avoid unhandled
    // cross-boundary accesses. A buffer may be allocated in fixed-size blocks
    // to avoid bank conflicts however the block size can't be arbitrary - we
    // need to avoid unhandled boundary crossings (e.g. in memory loads).
    std::vector<int> buf_min_block_sizes;
    // List of instructions whose bank conflicts are to be avoided.
    std::vector<stmt_t> instructions;

private:
    bank_conflict_attr_t(const std::vector<expr_t> &bufs,
            const std::vector<int> &buf_sizes,
            const std::vector<int> &buf_min_block_sizes,
            const std::vector<stmt_t> &instructions)
        : alloc_attr_impl_t(get_info())
        , bufs(bufs)
        , buf_sizes(buf_sizes)
        , buf_min_block_sizes(buf_min_block_sizes)
        , instructions(instructions) {}
};

// Allocation for SLM and GRF buffers.
// C++ equivalent:
//     {
//         byte *buf = new byte[size];
//         body;
//      }
class alloc_t : public stmt_iface_t<alloc_t> {
public:
    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const std::vector<alloc_attr_t> &attrs, const stmt_t &body = {}) {
        return stmt_t(new alloc_t(buf, size, kind, attrs, body));
    }

    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const alloc_attr_t &attr, const stmt_t &body = {}) {
        std::vector<alloc_attr_t> attrs = {attr};
        return make(buf, size, kind, attrs, body);
    }

    static stmt_t make(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const stmt_t &body = {}) {
        return make(buf, size, kind, std::vector<alloc_attr_t>(), body);
    }

    static stmt_t make(const expr_t &buf, const stmt_t &body = {}) {
        return stmt_t(new alloc_t(buf, body));
    }

    bool operator==(const alloc_t &other) const {
        return buf.is_equal(other.buf) && (size == other.size)
                && (kind == other.kind) && utils::is_equal(attrs, other.attrs)
                && body.is_equal(other.body);
    }

    size_t compute_hash() const override {
        return hash(buf, size, kind, attrs, body);
    }

    template <typename T>
    bool has_attr() const {
        for (auto &a : attrs)
            if (a.is<T>()) return true;
        return false;
    }

    template <typename T>
    const T &get_attr() const {
        for (auto &a : attrs)
            if (a.is<T>()) return a.as<T>();
        dsl_error() << "Can't find attribute.";
        return attrs[0].as<T>();
    }

    std::string line_str() const {
        ostringstream_t out;
        out << "alloc " << buf.as<var_t>().name << "[" << size << "]";
        return out.str();
    }

    expr_t buf;
    uint32_t size;
    alloc_kind_t kind;
    std::vector<alloc_attr_t> attrs;
    stmt_t body;

private:
    alloc_t(const expr_t &buf, uint32_t size, alloc_kind_t kind,
            const std::vector<alloc_attr_t> &attrs, const stmt_t &body)
        : buf(buf), size(size), kind(kind), attrs(attrs), body(body) {
        dsl_assert(buf.type().is_ptr()
                || into<uint32_t>(buf.type().size()) == size)
                << buf;
    }

    alloc_t(const expr_t &buf, const stmt_t &body)
        : buf(buf)
        , size(buf.type().size())
        , kind(alloc_kind_t::grf)
        , body(body) {
        dsl_assert(!buf.type().is_ptr()) << buf;
    }
};

// Assignment of a value to a variable.
// C++ equivalent:
//    var = value;
class assign_t : public stmt_iface_t<assign_t> {
public:
    static stmt_t make(const expr_t &var, const expr_t &value) {
        return stmt_t(new assign_t(var, value));
    }

    bool operator==(const assign_t &other) const {
        return var.is_equal(other.var) && value.is_equal(other.value);
    }

    size_t compute_hash() const override { return hash(var, value); }

    std::string str() const override {
        std::ostringstream oss;
        oss << var << "." << var.type();
        oss << " = " << value;
        return oss.str();
    }

    expr_t var;
    expr_t value;

private:
    assign_t(const expr_t &var, const expr_t &value) : var(var), value(value) {}
};

// Store to a GRF buffer.
// C++ equivalent (when value is scalar):
//     *(value_type *)(&buf[off]) = value;
// C++ equivalent (when value is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         *(scalar_type *)(&buf[off + i * _stride]) = value[i];
//     }
class store_t : public stmt_iface_t<store_t> {
public:
    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of value.type().scalar()
    // elements).
    static stmt_t make(const expr_t &buf, const expr_t &off,
            const expr_t &_value, int stride = default_stride,
            const expr_t &_mask = expr_t(), bool fill_mask0 = false) {
        auto mask = _mask;
        auto value = _value;
        if (mask) {
            if (all_of(mask, expr_t(true))) {
                mask = expr_t();
            } else if (all_of(mask, expr_t(false))) {
                // No need to store anything with a false mask,
                // unless explicitly asked to zero-fill the rest.
                if (!fill_mask0) return stmt_t();
                auto type = value.type();
                value = shuffle_t::make_broadcast(
                        cast_t::make(type.scalar(), 0), type.elems());
                mask = expr_t();
            }
        }
        return stmt_t(
                new store_t(buf, off, value, stride, mask, fill_mask0 && mask));
    }

    bool operator==(const store_t &other) const {
        return buf.is_equal(other.buf) && off.is_equal(other.off)
                && value.is_equal(other.value) && mask.is_equal(other.mask)
                && (stride == other.stride) && (fill_mask0 == other.fill_mask0);
    }

    size_t compute_hash() const override {
        return hash(buf, off, value, stride, mask, fill_mask0);
    }

    bool has_default_stride() const { return stride == default_stride; }

    std::string line_str() const {
        ostringstream_t out;
        out << load_t::make(value.type(), buf, off, stride);
        out << " = " << value;
        if (mask) {
            out << ", mask = " << mask;
            if (fill_mask0) out << " [FILL]";
        }
        return out.str();
    }

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    expr_t value;
    int stride;
    expr_t mask;
    bool fill_mask0;

private:
    store_t(const expr_t &_buf, const expr_t &_off, const expr_t &_value,
            int _stride, const expr_t &_mask, bool _fill_mask0)
        : buf(_buf)
        , off(_off)
        , value(_value)
        , stride(_stride)
        , mask(_mask)
        , fill_mask0(_fill_mask0) {
        normalize_ptr(value.type(), buf, off);
        dsl_assert(is_var(buf) || is_ref(buf)) << buf;
        if (stride == value.type().scalar().size()) stride = default_stride;
        if (mask)
            dsl_assert(mask.type() == type_t::_bool(value.type().elems()));
    }
};

// Loop statement with unit increment.
// C++ equivalent:
//    for (var = init; var < bound; var++) {
//        body;
//    }
// unroll specifies the unroll factor, unroll = 1 means no unrolling.
class for_t : public stmt_iface_t<for_t> {
public:
    static stmt_t make(const expr_t &var, const expr_t &init,
            const expr_t &bound, const stmt_t &body = {},
            const expr_t &step = expr_t(1), int unroll = 1) {
        return stmt_t(new for_t(var, init, bound, body, step, unroll));
    }

    bool operator==(const for_t &other) const {
        return var.is_equal(other.var) && init.is_equal(other.init)
                && bound.is_equal(other.bound) && body.is_equal(other.body)
                && step.is_equal(other.step) && (unroll == other.unroll);
    }

    size_t compute_hash() const override {
        return hash(var, init, bound, body, step, unroll);
    }

    std::string line_str() const {
        ostringstream_t out;
        out << "for (" << var << " = " << init << "; " << var << " < " << bound
            << "; " << var << " += " << step << ") ";
        if (unroll != 1) out << "[unroll: " << unroll << "] ";
        return out.str();
    }

    expr_t var;
    expr_t init;
    expr_t bound;
    stmt_t body;
    expr_t step;
    int unroll;

private:
    for_t(const expr_t &var, const expr_t &init, const expr_t &bound,
            const stmt_t &body, const expr_t &step, int unroll)
        : var(var)
        , init(init)
        , bound(bound)
        , body(body)
        , step(step)
        , unroll(unroll) {}
};

// If-else statement.
// C++ equivalent:
//     if (cond) {
//         body;
//     } else {
//         else_body;
//     }
class if_t : public stmt_iface_t<if_t> {
public:
    static stmt_t make(const expr_t &cond, const stmt_t &body,
            const stmt_t &else_body = stmt_t()) {
        return stmt_t(new if_t(cond, body, else_body));
    }

    bool operator==(const if_t &other) const {
        return cond.is_equal(other.cond) && body.is_equal(other.body)
                && else_body.is_equal(other.else_body);
    }

    size_t compute_hash() const override { return hash(cond, body, else_body); }

    std::string line_str() const {
        ostringstream_t oss;
        oss << "if (" << cond << ")";
        return oss.str();
    }

    expr_t cond;
    stmt_t body;
    stmt_t else_body;

private:
    if_t(const expr_t &cond, const stmt_t &body, const stmt_t &else_body)
        : cond(cond), body(body), else_body(else_body) {}
};

// Let statement, used to bind a variable to a value within a scope.
// C++ equivalent:
//     {
//         var = value;
//         body;
//     }
class let_t : public stmt_iface_t<let_t> {
public:
    static stmt_t make(
            const expr_t &var, const expr_t &value, const stmt_t &body = {}) {
        return stmt_t(new let_t(var, value, body));
    }

    bool operator==(const let_t &other) const {
        return var.is_equal(other.var) && value.is_equal(other.value)
                && body.is_equal(other.body);
    }

    size_t compute_hash() const override { return hash(var, value, body); }

    std::string line_str() const {
        ostringstream_t out;
        out << var << "." << var.type() << " = " << value;
        return out.str();
    }

    expr_t var;
    expr_t value;
    stmt_t body;

private:
    let_t(const expr_t &var, const expr_t &value, const stmt_t &body)
        : var(var), value(value), body(body) {
        if (value && !is_const(value))
            dsl_assert(var.type() == value.type())
                    << "Variable " << var << " and  value " << value
                    << "have different types. " << var.type()
                    << " != " << value.type() << "\n";
    }
};

// Statement label, specific to GEMM/convolution.
class stmt_label_t {
public:
    static stmt_label_t kernel(int index = -1) {
        return stmt_label_t(kind_t::_kernel, index);
    }
    static stmt_label_t compute_loop(int index = -1) {
        return stmt_label_t(kind_t::_compute_loop, index);
    }
    static stmt_label_t c_store(int index = -1) {
        return stmt_label_t(kind_t::_c_store, index);
    }
    static stmt_label_t c_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_c_zero_out, index);
    }
    static stmt_label_t b_reduced_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_b_reduced_zero_out, index);
    }
    static stmt_label_t g2s_load(int index = -1) {
        return stmt_label_t(kind_t::_g2s_load, index);
    }
    static stmt_label_t g2s_store(int index = -1) {
        return stmt_label_t(kind_t::_g2s_store, index);
    }
    static stmt_label_t g2r_load(int index = -1) {
        return stmt_label_t(kind_t::_g2r_load, index);
    }
    static stmt_label_t s2r_load(int index = -1) {
        return stmt_label_t(kind_t::_s2r_load, index);
    }
    static stmt_label_t prefetch(int index = -1) {
        return stmt_label_t(kind_t::_prefetch, index);
    }
    static stmt_label_t mul(int index = -1) {
        return stmt_label_t(kind_t::_mul, index);
    }

    bool operator==(const stmt_label_t &other) const {
        if (kind_ != other.kind_) return false;
        if (index_ == -1 || other.index_ == -1) return true;
        return index_ == other.index_;
    }

    bool operator!=(const stmt_label_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const { return hash(kind_, index_); }

    std::string str() const {
        switch (kind_) {
#define CASE(kind) \
    case kind_t::_##kind: return #kind
            CASE(kernel);
            CASE(compute_loop);
            CASE(c_store);
            CASE(c_zero_out);
            CASE(g2r_load);
            CASE(g2s_load);
            CASE(g2s_store);
            CASE(s2r_load);
            CASE(prefetch);
            CASE(mul);
#undef CASE
            default: stub();
        }
        return {};
    }

private:
    enum class kind_t {
        _undef,
        _kernel, // All kernel.
        _compute_loop, // Compute loop.
        _c_store, // GRF to GMEM store of C.
        _c_zero_out, // Zeroing-out of C.
        _b_reduced_zero_out, // Zeroing-out of B reduced buffer.
        _g2r_load, // GMEM to GRF load for further multiplication.
        _g2s_load, // GMEM to GRF load for GMEM -> SLM copy.
        _g2s_store, // GRF to SLM store for GMEM -> SLM copy.
        _s2r_load, // SLM to GRF load for further multiplication.
        _prefetch, // GMEM prefetch.
        _mul, // Multiplication.
    };

    stmt_label_t() : kind_(kind_t::_undef), index_(-1) {}
    stmt_label_t(kind_t kind, int index) : kind_(kind), index_(index) {}

    kind_t kind_;
    int index_; // Used to differentiate groups with the same kind.
};

// Statement group, used to assign a label to a group of statements.
class stmt_group_t : public stmt_iface_t<stmt_group_t> {
public:
    static stmt_t make(const stmt_label_t &label, const stmt_t &body) {
        return stmt_t(new stmt_group_t(label, body));
    }

    bool operator==(const stmt_group_t &other) const {
        return (label == other.label) && body.is_equal(other.body);
    }

    size_t compute_hash() const override { return hash(label, body); }

    stmt_label_t label;
    stmt_t body;

private:
    stmt_group_t(const stmt_label_t &label, const stmt_t &body)
        : label(label), body(body) {}
};

// Statement sequence, allows combining multiple statements.
// C++ equivalent:
//     {
//         vec[0];
//         vec[1];
//         ...
//     }
class stmt_seq_t : public stmt_iface_t<stmt_seq_t> {
public:
    static stmt_t make(const std::vector<stmt_t> &vec);

    static stmt_t make(const stmt_t &head, const stmt_t &tail) {
        return head.append(tail);
    }

    bool operator==(const stmt_seq_t &other) const {
        return utils::is_equal(vec, other.vec);
    }

    size_t compute_hash() const override { return hash(vec); }

    std::vector<stmt_t> vec;

private:
    stmt_seq_t(const std::vector<stmt_t> &vec) : vec(vec) {}
};

// While loop statement with a condition.
// C++ equivalent:
//    while (cond) {
//        body;
//    }
class while_t : public stmt_iface_t<while_t> {
public:
    static stmt_t make(const expr_t &cond, const stmt_t &body = {}) {
        return stmt_t(new while_t(cond, body));
    }

    bool operator==(const while_t &other) const {
        return cond.is_equal(other.cond) && body.is_equal(other.body);
    }

    size_t compute_hash() const override { return hash(cond, body); }

    std::string line_str() const {
        ostringstream_t out;
        out << "while (" << cond << ")";
        return out.str();
    }

    expr_t cond;
    stmt_t body;

private:
    while_t(const expr_t &cond, const stmt_t &body) : cond(cond), body(body) {}
};

// Function call attribute.
class func_call_attr_impl_t : public object::impl_t {
public:
    func_call_attr_impl_t(object::impl_t::info_t type_info)
        : object::impl_t(type_info) {}
};

class func_call_attr_t : public object_t {
public:
    using object_t::object_t;

    func_call_attr_t() = default;
    func_call_attr_t(const object_t &obj) : object_t(obj) {}
    func_call_attr_t(object_t &&obj) : object_t(obj) {}
    func_call_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_call_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    // Returns a function call with the attribute applied. The input statement
    // must be a function call.
    stmt_t apply_to(const stmt_t &s) const;

private:
    func_call_attr_t merge(const func_call_attr_t &other) const;
};

// Instruction modifier, relies on nGEN API.
class instruction_modifier_attr_t
    : public func_call_attr_impl_t,
      public object::info_t<instruction_modifier_attr_t> {
public:
    static func_call_attr_t make(const ngen::InstructionModifier &mod) {
        return func_call_attr_t(new instruction_modifier_attr_t(mod));
    }

    bool is_equal(const object::impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return mod.getAll() == other.mod.getAll();
    }

    size_t compute_hash() const override { return hash(mod.getAll()); }

    std::string str() const override {
        ostringstream_t oss;
        oss << "{";
        bool is_first = true;
        auto append = [&](const std::string &s) {
            if (!is_first) oss << ", ";
            oss << s;
            is_first = false;
        };
        if (mod.isAtomic()) append("Atomic");
        if (mod.isFwd()) append("Fwd");
        for (auto item : mod.getSWSB()) {
            if (item.hasTokenSet()) {
                append(std::string("$")
                        + std::to_string(mod.getSWSB()[0].getToken()));
            }
        }
        oss << "}";
        return oss.str();
    }

    ngen::InstructionModifier mod;

private:
    instruction_modifier_attr_t(const ngen::InstructionModifier &mod)
        : func_call_attr_impl_t(get_info()), mod(mod) {}
};

inline func_call_attr_t func_call_attr_t::merge(
        const func_call_attr_t &other) const {
    if (other.is_empty()) return *this;
    if (is_empty()) return other;
    auto *mod_attr1 = this->as_ptr<instruction_modifier_attr_t>();
    auto *mod_attr2 = other.as_ptr<instruction_modifier_attr_t>();
    dsl_assert(mod_attr1 && mod_attr2)
            << "Merging of attributes is not supported: " << *this << " and "
            << other;
    return instruction_modifier_attr_t::make(mod_attr1->mod | mod_attr2->mod);
}

// Base class for function IR objects.
class func_impl_t : public object::impl_t {
public:
    func_impl_t(object::impl_t::info_t type_info) : object::impl_t(type_info) {}

    size_t compute_hash() const override {
        dsl_error() << "compute_hash() is not implemented.";
        return 0;
    }

    bool is_equal(const object::impl_t &obj) const override {
        dsl_error() << "is_equal() is not implemented.";
        return false;
    }

    stmt_t call(const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) const;

    object_t _mutate(ir_mutator_t &mutator) const override {
        return mutator._mutate(*this);
    }
    void _visit(ir_visitor_t &visitor) const override { visitor._visit(*this); }
};

// Wrapper for IR function objects.
class func_t : public object_t {
public:
    using object_t::object_t;

    func_t() = default;
    func_t(const object_t &obj) : object_t(obj) {}
    func_t(object_t &&obj) : object_t(obj) {}
    func_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    const func_impl_t &func_impl() const {
        return *static_cast<func_impl_t *>(impl());
    }

    stmt_t call(const std::vector<expr_t> &args = {},
            const func_call_attr_t &attr = {}) const {
        return func_impl().call(args, attr);
    }

    stmt_t operator()(const std::vector<expr_t> &args = {}) const {
        return call(args);
    }

    stmt_t operator()(const expr_t &arg) const { return call({arg}); }
};

// Function call.
class func_call_t : public stmt_iface_t<func_call_t> {
public:
    static stmt_t make(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) {
        return stmt_t(new func_call_t(func, args, attr));
    }

    bool operator==(const func_call_t &other) const {
        return func.is_equal(other.func) && utils::is_equal(args, other.args)
                && attr.is_equal(other.attr);
    }

    size_t compute_hash() const override { return hash(args, attr); }

    std::string line_str() const {
        ostringstream_t out;
        out << func << "(" << utils::make_seq_print_helper(args) << ")";
        if (attr) out << " " << attr;
        return out.str();
    }

    func_t func;
    std::vector<expr_t> args;
    func_call_attr_t attr;

private:
    func_call_t(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr)
        : func(func), args(args), attr(attr) {
        dsl_assert(func);
    }
};

inline stmt_t func_impl_t::call(
        const std::vector<expr_t> &args, const func_call_attr_t &attr) const {
    return func_call_t::make(this, args, attr);
}

inline stmt_t func_call_attr_t::apply_to(const stmt_t &s) const {
    auto &c = s.as<func_call_t>();
    return func_call_t::make(c.func, c.args, c.attr.merge(*this));
}

template <typename F>
inline bool is_func_call(const stmt_t &s) {
    auto *c = s.as_ptr<func_call_t>();
    if (!c) return false;
    return c->func.is<F>();
}

// Generic function with a name.
class builtin_t : public func_impl_t, public object::info_t<builtin_t> {
public:
    static func_t make(const std::string &name) {
        return func_t(new builtin_t(name));
    }

    bool is_equal(const object::impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return name == other.name;
    }

    std::string str() const override { return name; }

    std::string name;

private:
    builtin_t(const std::string &name) : func_impl_t(get_info()), name(name) {}
};

// The following types are intrusive pointers and, as such, should have the same
// size as a pointer.
static_assert(sizeof(object_t) <= sizeof(void *),
        "intrusive pointer type object_t size is greater than void * "
        "size.");
static_assert(sizeof(expr_t) <= sizeof(void *),
        "intrusive pointer type expr_t size is greater than void * size.");
static_assert(sizeof(stmt_t) <= sizeof(void *),
        "intrusive pointer type stmt_t size is greater than void * size.");

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
