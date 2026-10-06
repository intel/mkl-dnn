/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_SDPA_BRGEMM_SDPA_HPP
#define CPU_X64_SDPA_BRGEMM_SDPA_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"

#include "cpu/platform.hpp"

#include "cpu/x64/sdpa/sdpa_full_softmax.hpp"
#include "cpu/x64/sdpa/sdpa_online_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Two interchangeable compute strategies back brgemm_sdpa_fwd_t, both folded in
// as free functions over the pd's conf + the primitive's kernels:
// `full_softmax` (namespace sdpa_full_softmax -- query-axis blocked, two-pass
// softmax; bf16/f16 and an additive attention mask) and `online_softmax`
// (namespace sdpa_online_softmax -- online/flash softmax; f32-only).
// pd_t::init() picks one by shape/dtype capability, or a forced choice via
// ONEDNN_SDPA_IMPL={online_softmax,full_softmax,auto}.
enum class sdpa_impl_kind_t { full_softmax, online_softmax };

struct brgemm_sdpa_fwd_t : public primitive_t {
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T("brg_sdpa:any", brgemm_sdpa_fwd_t);

        status_t init(const engine_t *engine);

        sdpa_impl_kind_t impl_kind() const { return kind_; }
        const sdpa_full_softmax_conf_t &full_softmax_conf() const {
            return full_conf_;
        }
        const sdpa_full_softmax_descs_t &full_softmax_descs() const {
            return full_descs_;
        }
        const sdpa_online_softmax_conf_t &online_softmax_conf() const {
            return online_conf_;
        }
        int nthr() const { return nthr_; }

    private:
        friend struct brgemm_sdpa_fwd_t;

        sdpa_impl_kind_t kind_ = sdpa_impl_kind_t::full_softmax;
        // Derived compute configuration computed once in init() (via the
        // strategy's JIT-free configure()); only the struct matching kind_ is
        // populated. It owns the KV/query tiling and per-thread scratch layout,
        // so the pd sizes its scratchpad from it. For full_softmax, the finalized
        // BRGEMM descriptors live alongside it in full_descs_ (like brgemm_matmul's
        // brg_descs_); the primitive JIT-compiles the kernels from conf + descs.
        sdpa_full_softmax_conf_t full_conf_;
        sdpa_full_softmax_descs_t full_descs_;
        sdpa_online_softmax_conf_t online_conf_;
        int nthr_ = 1;
    };

    brgemm_sdpa_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    // Compute state owned by the primitive, built in init() from the pd's conf;
    // only the one matching pd()->impl_kind() is populated. Both strategies
    // are folded in as free functions (namespace sdpa_online_softmax / sdpa_full_softmax) over
    // the pd conf (+ descs) and these kernels.
    sdpa_online_softmax_kernels_t online_kernels_;
    sdpa_full_softmax_kernels_t full_kernels_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
