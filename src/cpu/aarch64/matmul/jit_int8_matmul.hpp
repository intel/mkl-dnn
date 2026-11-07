/*******************************************************************************
* Copyright 2025-2026 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
* Copyright 2026 Intel Corporation
* Copyright 2026 Ampere Computing LLC

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

#ifndef CPU_AARCH64_MATMUL_JIT_INT8_MATMUL_HPP
#define CPU_AARCH64_MATMUL_JIT_INT8_MATMUL_HPP

#include <array>
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/matmul/jit_int8_kernel_types.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

template <cpu_isa_t isa>
struct jit_int8_matmul_kernel_t;
struct jit_int8_matmul_utils_kernel_t;
template <cpu_isa_t isa>
struct jit_int8_matmul_t : public primitive_t {
    struct pd_t : public cpu::matmul::cpu_matmul_pd_t {
    public:
        using cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("jit:int8", jit_int8_matmul_t);

        status_t init(const engine_t *engine);

        int get_idx(bool is_zp_cal, bool is_m_tail, bool is_k_tail,
                bool is_n_tail, const brg_int8_t &brg) const;

        int mtile_sz, ntile_sz, mm_parallel_work, thr_M, thr_N, res_thr_M,
                res_thr_N;

        brg_int8_t brg_int8_conf;
        dyn_vals_t dyn_vals;

    private:
        bool formats_ok() const;
        bool post_ops_ok() const;
    };

    jit_int8_matmul_t(const pd_t *apd);

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    ~jit_int8_matmul_t() override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // 16 possible kernels from each potential combination of true or false over
    // 4 parameters: is_m_tail, is_n_tail, is_k_tail, is_zp_cal
    static constexpr int num_jit_kernels = 16;
    std::array<std::unique_ptr<jit_int8_matmul_kernel_t<isa>>, num_jit_kernels>
            int8_kernels_;
    std::unique_ptr<jit_int8_matmul_utils_kernel_t> reo_ker_a_;
    std::unique_ptr<jit_int8_matmul_utils_kernel_t> reo_ker_b_;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
