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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_STRIDES_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_STRIDES_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

/// Bytes between two consecutive granules along an axis of a matrix
///
/// @details Either a number of bytes known at codegen, or a leading dimension
///     (in bytes) times a factor
struct granule_stride_t {
    enum class base_t { bytes, lda, ldb, ldc };

    base_t base = base_t::bytes;

    /// The bytes, or the factor the leading dimension is multiplied by
    int factor = 0;

    granule_stride_t() = default;

    /// Creates a stride
    ///
    /// @param base What the factor scales
    /// @param factor The bytes, or the multiplier of the leading dimension
    granule_stride_t(base_t base, int factor) : base(base), factor(factor) {}

    /// Whether the stride scales a leading dimension
    bool is_ld() const { return base != base_t::bytes; }

    bool operator==(const granule_stride_t &o) const {
        return base == o.base && factor == o.factor;
    }
};

/// The strides of the A, B and C streams a matmul walks, per granule step
///
/// @details The granularity of each dimension, which the implementation
///     decides, is how many elements one step along it covers: the stride
///     along a dimension is the stride of one element times that grain. A
///     step along K or N on a transposed operand jumps whole leading
///     dimensions, so the grain multiplies them
struct strides_t {
    granule_stride_t a_k, a_m; // A is `{k, m}`
    granule_stride_t b_k, b_n; // B is `{k, n}`
    granule_stride_t c_n, c_m; // C is `{n, m}`

    /// The strides of `p` for the given granularities
    ///
    /// @param p The plan
    /// @param mg The elements of one M granule
    /// @param ng The elements of one N granule
    /// @param kg The elements of one K granule
    static strides_t of(const plan_t &p, int mg, int ng, int kg) {
        using base_t = granule_stride_t::base_t;
        const int sewb_a = rvjit::sewb(p.dta.mem);
        const int sewb_b = rvjit::sewb(p.dtb.mem);
        const int sewb_c = rvjit::sewb(p.dtc.mem);

        strides_t s;
        s.a_k = p.is_transa ? granule_stride_t {base_t::bytes, sewb_a * kg}
                            : granule_stride_t {base_t::lda, kg};
        s.a_m = p.is_transa ? granule_stride_t {base_t::lda, mg}
                            : granule_stride_t {base_t::bytes, sewb_a * mg};
        s.b_k = p.is_transb ? granule_stride_t {base_t::ldb, kg}
                            : granule_stride_t {base_t::bytes, sewb_b * kg};
        s.b_n = p.is_transb ? granule_stride_t {base_t::bytes, sewb_b * ng}
                            : granule_stride_t {base_t::ldb, ng};
        s.c_n = {base_t::ldc, ng};
        s.c_m = {base_t::bytes, sewb_c * mg};
        return s;
    }
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#endif
