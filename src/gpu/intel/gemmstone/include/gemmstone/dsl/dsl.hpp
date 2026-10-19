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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_DSL_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_DSL_HPP

#include "gemmstone/dsl/decl.hpp"
#include "gemmstone/dsl/hw.hpp"
#include "gemmstone/dsl/kernel.hpp"
#include "gemmstone/dsl/tensor.hpp"
#include "gemmstone/dsl/type.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

int grf_size();
int min_align_2d();
int min_pitch_2d();
const hw_t &get_hw();
const kernel::options_t &options();

struct send_hint_t {
    send_cache_hint_t cache;
};

void declare_kernel(
        const kernel::iface_t &interface, const kernel::options_t &ctx);
kernel_t end_kernel();

void begin_scope();
void end_scope();
stmt_t pop_scope(); // Ends current scope and removes it from the kernel
void append(stmt_t stmt); // Adds statement to the current scope

void assume(const expr_t &e);

// Multiply and add: dst = a * b + c.
expr_t mad(
        const expr_t &dst, const expr_t &a, const expr_t &b, const expr_t &c);

const std::array<expr_t, 3> &group_ids();
const expr_t &group_id(int idx);
const expr_t &group_subgroup_count();
const std::array<expr_t, 3> &local_ids();
const expr_t &local_id(int idx);
const std::array<expr_t, 3> &local_sizes();
const expr_t &local_size(int idx);
const expr_t global_id(int idx, type_t type = u64);

class lval_t : public stringify_t<lval_t> {
public:
    lval_t() = default;
    lval_t(const lval_t &) = default;
    lval_t(lval_t &&) = default;
    lval_t(const type_t &type, const std::string &name);
    lval_t(const expr_t &v) : var(v) {}
    // Deleted to prevent accidental copies that bypass DSL assign tracking.
    lval_t &operator=(const lval_t &) = delete;
    lval_t &operator=(lval_t &&) = default;
    lval_t &operator=(const expr_t &obj);
    lval_t &assign(const expr_t &rhs) { return *this = rhs; }

    explicit operator bool() const { return !var.is_empty(); }
    lval_t sub(int off, int elems) const;
    expr_t ptr(int off = 0) const { return var.ptr(off); }
    lval_t operator[](int off) const { return sub(off, 1); }
    operator expr_t() const { return var; }

#define DEFINE_BINARY_ASSIGN_OPERATOR(op) \
    lval_t &operator op##=(const expr_t & rhs) { \
        (*this) = (*this)op rhs; \
        return *this; \
    }

    DEFINE_BINARY_ASSIGN_OPERATOR(+)
    DEFINE_BINARY_ASSIGN_OPERATOR(-)
    DEFINE_BINARY_ASSIGN_OPERATOR(*)
    DEFINE_BINARY_ASSIGN_OPERATOR(/)
    DEFINE_BINARY_ASSIGN_OPERATOR(%)
    DEFINE_BINARY_ASSIGN_OPERATOR(|)
    DEFINE_BINARY_ASSIGN_OPERATOR(&)
    DEFINE_BINARY_ASSIGN_OPERATOR(^)

#undef DEFINE_BINARY_ASSIGN_OPERATOR

    std::string str() const {
        std::ostringstream oss;
        oss << "lval->var: " << var.str();
        return oss.str();
    }

    expr_t var;
};

#define DECLARE_LVAL_BINARY_OPERATOR(op) \
    inline expr_t operator op(const lval_t &a, const lval_t &b) { \
        return expr_t(a) op expr_t(b); \
    } \
    inline expr_t operator op(const lval_t &a, const expr_t &b) { \
        return expr_t(a) op b; \
    } \
    inline expr_t operator op(const expr_t &a, const lval_t &b) { \
        return a op expr_t(b); \
    }

DECLARE_LVAL_BINARY_OPERATOR(+)
DECLARE_LVAL_BINARY_OPERATOR(-)
DECLARE_LVAL_BINARY_OPERATOR(*)
DECLARE_LVAL_BINARY_OPERATOR(/)
DECLARE_LVAL_BINARY_OPERATOR(%)
DECLARE_LVAL_BINARY_OPERATOR(<<)
DECLARE_LVAL_BINARY_OPERATOR(>>)
DECLARE_LVAL_BINARY_OPERATOR(==)
DECLARE_LVAL_BINARY_OPERATOR(!=)
DECLARE_LVAL_BINARY_OPERATOR(>)
DECLARE_LVAL_BINARY_OPERATOR(>=)
DECLARE_LVAL_BINARY_OPERATOR(<)
DECLARE_LVAL_BINARY_OPERATOR(<=)
DECLARE_LVAL_BINARY_OPERATOR(&)
DECLARE_LVAL_BINARY_OPERATOR(|)
DECLARE_LVAL_BINARY_OPERATOR(^)

#undef DECLARE_LVAL_BINARY_OPERATOR

inline expr_t operator-(const lval_t &a) {
    return -expr_t(a);
}

expr_t subgroup_id(int idx = 0);
expr_t subgroup_local_id();
expr_t subgroup_linear_id();

expr_t arg(const std::string &name, bool allow_empty = false);
lval_t def(
        const std::string &name, const type_t &type, const expr_t &value = {});
lval_t def(const std::string &name, const expr_t &value);
tensor_t def(const std::string &name, const layout_t &layout,
        type::attr_t attr = {});
tensor_t def(const std::string &name, const layout_t &layout,
        const expr_t &value, type::attr_t attr = {});
lval_t let(const std::string &name, const type_t &type, const expr_t &value);
lval_t let(const std::string &name, const expr_t &value);

expr_t iif(
        const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr);
expr_t extract(const expr_t &expr, int lane);
expr_t extract(const expr_t &expr, int lane_start, int lane_end);

uint32_t uint32_reciprocal(uint32_t x);
expr_t idiv(const expr_t &a, uint32_t b);
expr_t idiv(const expr_t &a, uint32_t b, expr_t &rem);
expr_t idiv(const expr_t &a, const expr_t &b, const expr_t &recip);
expr_t idiv(const expr_t &a, const expr_t &b, const expr_t &recip, expr_t &rem);

// Use leading underscore to avoid conflict with include/internal/utils.hpp.
// TODO: Switch to utils namespace for utils.hpp.
expr_t _divide_up(const expr_t &a, uint32_t b);
expr_t _round_up(const expr_t &a, uint32_t b);

void assign(const expr_t &var, const expr_t &value);

void prefetch(const global_tensor_t &g, const icoord_t &base = {},
        const send_hint_t &hint = {});

expr_t load(const expr_t &buf, const expr_t &off, const expr_t &mask = {},
        const send_hint_t &hint = {});

void load(const tensor_t &t, const global_tensor_t &g,
        const icoord_t &base = {}, const send_hint_t &hint = {});

void store(const expr_t &buf, const expr_t &off, const expr_t &val,
        const expr_t mask = {}, const send_hint_t &hint = {});

void store(const global_tensor_t &g, const tensor_t &t,
        const icoord_t &base = {}, const send_hint_t &hint = {});

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B);

expr_t cast(const type_t &type, const expr_t &src);
tensor_t cast(const type_t &type, const tensor_t &src);

void slm_fence();

void _if_impl(const expr_t &cond, const stmt_t &body);
template <typename F>
void _if(const expr_t &cond, const F &if_body) {
    if (is_const(cond)) {
        if (to_bool(cond)) {
            begin_scope();
            if_body();
            end_scope();
        }
    } else {
        begin_scope();
        if_body();
        _if_impl(cond, pop_scope());
    }
}

void _if_impl(
        const expr_t &cond, const stmt_t &if_body, const stmt_t &else_body);
template <typename F, typename G>
void _if(const expr_t &cond, const F &if_body, const G &else_body) {
    if (is_const(cond)) {
        begin_scope();
        if (to_bool(cond)) {
            if_body();
        } else {
            else_body();
        }
        end_scope();
    } else {
        begin_scope();
        if_body();
        auto if_body_stmt = pop_scope();

        begin_scope();
        else_body();
        _if_impl(cond, if_body_stmt, pop_scope());
    }
}

void _for_impl(const expr_t &var, const expr_t &bound, const expr_t &step,
        const stmt_t &body);
template <typename F>
void _for(const expr_t &var, const expr_t &bound, const expr_t &step,
        const F &body) {
    begin_scope();
    body();
    _for_impl(var, bound, step, pop_scope());
}

template <typename F>
void _for(const expr_t &var, const expr_t &bound, const F &body) {
    _for(var, bound, 1, body);
}

void _while_impl(const expr_t &cond, const stmt_t &body);
template <typename F>
void _while(const expr_t &cond, const F &body) {
    if (is_const(cond) && !to_bool(cond)) return;
    begin_scope();
    body();
    _while_impl(cond, pop_scope());
}

#define DECLARE_TENSOR_BINARY_OPERATOR(cpp_op) \
    tensor_t operator cpp_op(const tensor_t &a, const tensor_t &b); \
    tensor_t operator cpp_op(const tensor_t &a, const expr_t &b); \
    tensor_t operator cpp_op(const expr_t &a, const tensor_t &b);

DECLARE_TENSOR_BINARY_OPERATOR(+)
DECLARE_TENSOR_BINARY_OPERATOR(-)
DECLARE_TENSOR_BINARY_OPERATOR(*)
DECLARE_TENSOR_BINARY_OPERATOR(/)

#undef DECLARE_TENSOR_BINARY_OPERATOR

#define DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR(cpp_op) \
    void operator cpp_op##=(const tensor_t &a, const tensor_t &b); \
    void operator cpp_op##=(const tensor_t &a, const expr_t &b);

DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR(+)
DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR(-)
DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR(*)
DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR(/)

#undef DECLARE_TENSOR_BINARY_ASSIGN_OPERATOR

tensor_t exp(const tensor_t &src);
expr_t exp(const expr_t &src, const expr_t &acc_out = {},
        const expr_t &acc_in = {});
expr_t exp(const expr_t &src, const type_t &out_type,
        const expr_t &acc_out = {}, const expr_t &acc_in = {});
expr_t min(const expr_t &a, const expr_t &b);
expr_t max(const expr_t &a, const expr_t &b);

void barrier();

// Creates a global tensor with explicit strides. batch_idxs are indices into
// batch dimensions, used to compute base pointer offset. transpose swaps
// strides of the last two dimensions.
global_tensor_t def_global_tensor(const std::string &name,
        const std::vector<expr_t> &batch_idxs, const std::vector<idx_t> &dims,
        const std::vector<expr_t> &dim_sizes, std::vector<expr_t> dim_strides,
        bool transpose = false);

// Creates a global tensor with strides computed from sizes (last dim has
// stride 1, earlier dims have product of later sizes). transpose swaps strides
// of the last two dimensions.
global_tensor_t def_global_tensor(const std::string &name,
        const std::vector<expr_t> &batch_idxs, const std::vector<idx_t> &dims,
        const std::vector<expr_t> &sizes, bool transpose = false);

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
