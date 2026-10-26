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

#ifndef CPU_RV64_RVJIT_RVJIT_MATMUL_HPP
#define CPU_RV64_RVJIT_RVJIT_MATMUL_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"

// The access every RV64 client has to rvjit: a matmul micro-kernel described
// with oneDNN types, tuned once, and generated pinned to the tuned blocks.

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// A value the kernel knows when it is generated, or reads from its arguments.
struct rvjit_value_t {
    static constexpr ptrdiff_t literal_value = -1;

    // The literal value, or the bound of a value (0 is unbounded)
    dim_t value = 0;
    // The byte offset of the value in the arguments, or `literal_value`
    ptrdiff_t arg = literal_value;

    static rvjit_value_t literal(dim_t v) {
        rvjit_value_t r;
        r.value = v;
        return r;
    }
    static rvjit_value_t at(size_t offset, dim_t bound = 0) {
        rvjit_value_t r;
        r.value = bound;
        r.arg = static_cast<ptrdiff_t>(offset);
        return r;
    }
    bool is_literal() const { return arg == literal_value; }
};

/// The type of an operand in memory and in the registers computing on it
///
/// @note An undef register type is the memory type
struct rvjit_type_t {
    data_type_t mem = data_type::f32;
    data_type_t reg = data_type::undef;

    rvjit_type_t() = default;
    rvjit_type_t(data_type_t m, data_type_t r = data_type::undef)
        : mem(m), reg(r) {}
};

// The epilogue features a kernel is generated with; each call enables a
// subset of them through its runtime flag word.
enum rvjit_feature_t : unsigned {
    rvjit_alpha = 1u << 0,
    rvjit_beta = 1u << 1,
    rvjit_bias = 1u << 2,
};

enum class rvjit_bias_kind_t { per_m, scalar };

// C (M x N, column-major) = alpha * A (M x K, column-major unless trans_a)
// * B (K x N, column-major unless trans_b) + beta * C + bias, the vectors
// running along M. The pointers, and every value not literal, are read from
// the client's argument structure at the given offsets.
struct rvjit_matmul_desc_t {
    rvjit_type_t a, b, c;
    // The type the epilogue computes in (undef: the register type of c)
    data_type_t epilogue = data_type::undef;
    // The type of the bias array (undef: the epilogue's)
    data_type_t bias = data_type::undef;
    rvjit_bias_kind_t bias_kind = rvjit_bias_kind_t::per_m;

    bool trans_a = false, trans_b = false;

    rvjit_value_t m, n, k; // The shape, read from the arguments when bounded
    rvjit_value_t lda, ldb, ldc; // In bytes
    size_t ptr_a = 0, ptr_b = 0, ptr_c = 0;
    ptrdiff_t ptr_bias = rvjit_value_t::literal_value;

    unsigned features = 0;
    // Literal scalars, or the offsets of their bit patterns
    double alpha = 1.0, beta = 0.0;
    ptrdiff_t alpha_arg = rvjit_value_t::literal_value;
    ptrdiff_t beta_arg = rvjit_value_t::literal_value;
    // The offset of the 64-bit runtime flag word, or every feature applies
    ptrdiff_t rt_flags = rvjit_value_t::literal_value;

    // The register grouping of the inputs (1, 2, 4 or 8), for a client whose
    // memory layout is blocked by it; 0 leaves it to rvjit
    int lmul = 0;
};

// The blocks of a tuned kernel: each loop is unrolled by a number of
// granules of `{m,n,k}_grain` elements (an M granule is one vector, N and K
// granules are one element), covering `{m,n,k}_block()` elements.
struct rvjit_matmul_blocks_t {
    int m_unroll = 0, m_grain = 0;
    int n_unroll = 0, n_grain = 0;
    int k_unroll = 0, k_grain = 0;

    int m_block() const { return m_unroll * m_grain; }
    int n_block() const { return n_unroll * n_grain; }
    int k_block() const { return k_unroll * k_grain; }

    bool operator==(const rvjit_matmul_blocks_t &o) const {
        return m_unroll == o.m_unroll && m_grain == o.m_grain
                && n_unroll == o.n_unroll && n_grain == o.n_grain
                && k_unroll == o.k_unroll && k_grain == o.k_grain;
    }
    bool operator!=(const rvjit_matmul_blocks_t &o) const {
        return !(*this == o);
    }
};

// Decides the blocks of `desc`, or returns status::unimplemented when rvjit
// cannot take it, so a primitive is skipped at descriptor initialization.
status_t rvjit_matmul_tune(
        const rvjit_matmul_desc_t &desc, rvjit_matmul_blocks_t &blocks);

// The runtime flag word enabling the given features of a call.
uint64_t rvjit_runtime_flags(bool alpha, bool beta, bool bias);

struct jit_rvjit_matmul_t;

// The kernel of `desc`, pinned to `blocks` (from rvjit_matmul_tune).
struct rvjit_matmul_kernel_t {
    rvjit_matmul_kernel_t(const rvjit_matmul_desc_t &desc,
            const rvjit_matmul_blocks_t &blocks);
    ~rvjit_matmul_kernel_t();
    rvjit_matmul_kernel_t(const rvjit_matmul_kernel_t &) = delete;
    rvjit_matmul_kernel_t &operator=(const rvjit_matmul_kernel_t &) = delete;

    status_t create_kernel();

    // Runs the kernel on the client's argument structure
    void operator()(const void *args) const;

private:
    jit_rvjit_matmul_t *jit_ = nullptr;
};

// Generates the kernel of `desc` pinned to `blocks` into `kernel`.
status_t rvjit_matmul_kernel_create(const rvjit_matmul_desc_t &desc,
        const rvjit_matmul_blocks_t &blocks,
        std::unique_ptr<rvjit_matmul_kernel_t> &kernel);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVJIT_RVJIT_MATMUL_HPP
