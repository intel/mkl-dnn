/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_PLAN_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_PLAN_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/parameter.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

struct postop_t;

/// Features compiled into the generated code, and the bits for runtime checks
///
/// @note `plan_t::flags` holds the codegen flags; the runtime flag word
///     `plan_t::rt_flags` uses the same bit layout
enum codegen_flags_t : int {
    with_alpha = 1, // C = alpha * A * B ...
    with_beta = 2, // ... + beta * C ...
    with_bias = 4, // ... + bias (an array along M)
    with_postops = 8, // ... then the plan's post-op chain
};

/// How the bias array spans M
enum class bias_kind_t {
    per_m, // One element per M, added to every column
    scalar, // One element, added to every M and column
};

/// Block sizes a client fixes ahead of the implementation's own tuning
///
/// @note A zero leaves the decision to the implementation, which honors a
///     positive value or rejects the plan
/// @note The sizes are the implementation's dispatch widths: M granules
///     (`impl_t::mb`), N columns and the K unroll
struct blocks_t {
    int mb = 0;
    int nb = 0;
    int kb = 0;
};

/// Implementation-agnostic matrix multiplication plan
///
/// @note A plain data object passed into `matmul_t::plan()`
struct plan_t {
    /// A's type in memory and in the registers the multiply-accumulate reads
    type_parameter_t dta;

    /// B's type in memory and in its register
    ///
    /// @note The memory type sets B's strides and the width of its scalar
    ///     load; a wider integer register type extends each element as it is
    ///     loaded, so the multiply-accumulate sees it as that type
    type_parameter_t dtb;

    /// C's type in memory and the type the products accumulate in
    ///
    /// @note The accumulators are converted to the memory type as they are
    ///     stored, and the previous C converted from it when it is read
    type_parameter_t dtc;

    /// The type the epilogue works in; none is the accumulators'
    ///
    /// @note An integer accumulation may scale, accumulate and add its bias in
    ///     a float of the same width, which the accumulators are converted to
    ///     in place before the epilogue
    type_t dtepi = type_t::none;

    /// The bias array's type in memory; none is the epilogue's
    type_t dtbias = type_t::none;

    /// How the bias spans M
    bias_kind_t bias_kind = bias_kind_t::per_m;

    bool is_transa = false;
    bool is_transb = false;

    /// The type the epilogue works in
    type_t epi_type() const { return dtepi == type_t::none ? dtc.reg : dtepi; }

    /// The bias array's type in memory
    type_t bias_type() const {
        return dtbias == type_t::none ? epi_type() : dtbias;
    }

    /// What is known about each dimension of the call
    size_parameter_t m, n, k;

    /// The block sizes a client pins, bypassing the implementation's tuning
    blocks_t blocks;

    parameter_t ptra, ptrb, ptrc;
    parameter_t lda, ldb, ldc;

    /// Features compiled into the code: `codegen_flags_t` entries
    int flags = 0;

    /// How to obtain the runtime flag word, a subset of `flags` per call
    ///
    /// @note No source: every feature in `flags` is always applied
    /// @note A literal: the word is known at codegen time, emitting no branch
    parameter_t rt_flags;

    /// Scalars of the epilogue's type, for `with_alpha` and `with_beta`
    scalar_parameter_t alpha, beta;

    /// Address of the bias array, of `bias_type()` elements, for `with_bias`
    parameter_t bias;

    /// Opaque pointer handed to the post-ops, for `with_postops`
    parameter_t postop_args;

    /// Post-ops applied in order after bias, for `with_postops`
    static constexpr int MAX_POSTOPS = 8;
    postop_t *postops[MAX_POSTOPS] = {};
    int n_postops = 0;

    /// Appends a post-op to the chain
    ///
    /// @param p The post-op, which must outlive the plan
    ///
    /// @note Returns false when the chain is full
    bool add_postop(postop_t *p) {
        if (n_postops >= MAX_POSTOPS) return false;
        postops[n_postops++] = p;
        return true;
    }
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#endif
