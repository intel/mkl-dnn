/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "dsl/ir/pass/pass.hpp"

#include "dsl/ir/pass/simplify.hpp"
#include "dsl/ir/pass/trace.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class if_condition_fixer_t : public ir_mutator_t {
public:
    if_condition_fixer_t(int simd_size)
        : simd_size_(simd_size), in_cond_(false) {}

    object_t _mutate(const if_t &obj) override {
        auto _new_obj = ir_mutator_t::_mutate(obj);
        auto &new_obj = _new_obj.as<if_t>();
        flag_setter_t in_cond(&in_cond_, true);
        auto cond = mutate(new_obj.cond);
        return if_t::make(cond, new_obj.body, new_obj.else_body);
    }

    object_t _mutate(const binary_op_t &obj) override {
        if (!in_cond_) return obj;
        auto broadcast = [&](const expr_t &operand) {
            object_t ret;
            if (is_cmp_op(obj.op_kind) && obj.type.elems() == 1)
                ret = shuffle_t::make_broadcast(operand, simd_size_);
            else
                ret = mutate(operand);
            return ret;
        };
        auto a = broadcast(obj.a);
        auto b = broadcast(obj.b);
        return binary_op_t::make(obj.op_kind, a, b);
    }

private:
    struct flag_setter_t {
        flag_setter_t(bool *flag, bool value) : flag(flag), old(*flag) {
            *flag = value;
        }
        ~flag_setter_t() { *flag = old; }

        bool *flag;
        bool old;
    };

    int simd_size_;
    bool in_cond_;
};

void fixup_if_conditions(kernel_t &k) {
    trace_start();
    k.body = if_condition_fixer_t(k.options.simd()).mutate(k.body);
    trace_pass("fixup_if_conditions", k);
}

stmt_t fixup_if_conditions(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = if_condition_fixer_t(ir_ctx.options().simd()).mutate(s);
    trace_pass("fixup_if_conditions", ret, ir_ctx);
    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
