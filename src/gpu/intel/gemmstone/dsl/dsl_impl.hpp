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

#ifndef GEMMSTONE_DSL_DSL_IMPL_HPP
#define GEMMSTONE_DSL_DSL_IMPL_HPP

#include "gemmstone/dsl/decl.hpp"
#include "gemmstone/dsl/dsl.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

namespace op {

enum class kind_t {
    undef,
    add,
    sub,
    mul,
    div,
    mod,
    min,
    max,
    prelu,
};

constexpr kind_t add = kind_t::add;
constexpr kind_t sub = kind_t::sub;
constexpr kind_t mul = kind_t::mul;
constexpr kind_t div = kind_t::div;
constexpr kind_t mod = kind_t::mod;
constexpr kind_t min = kind_t::min;
constexpr kind_t max = kind_t::max;
constexpr kind_t prelu = kind_t::prelu;

#ifndef STANDALONE
kind_t kind(dnnl::impl::alg_kind_t alg);
#endif

} // namespace op

void binary(op::kind_t op, const tensor_t &dst, const tensor_t &src0,
        const tensor_t &src1);
tensor_t binary(op::kind_t op, const tensor_t &src0, const tensor_t &src1);
tensor_t binary(op::kind_t op, const tensor_t &src0, const expr_t &src1);
tensor_t binary(op::kind_t op, const expr_t &src0, const tensor_t &src1);

// Returns SIMD size of the kernel being generated.
int simd();

// Returns the interface of the kernel currently being generated.
const kernel::iface_t &kernel_iface();

// Context storage for an expression with the given name. Initialization is
// handled by the user.
expr_t &get_expr(const std::string &name);

// Extracts M, N, K dimension indices from A, B, C tiles for matrix multiplication.
// Validates that tile sizes match between A, B, C.
void get_mnk_dims(const tile_t &a, const tile_t &b, const tile_t &c, idx_t &m,
        idx_t &n, idx_t &k);

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
