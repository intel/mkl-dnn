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

#include "cpu/x64/sdpa/sdp_blocked_driver.hpp"
#include "cpu/x64/sdpa/sdp_fused_driver.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Two interchangeable compute drivers back brgemm_sdpa_fwd_t: sdp_blocked_driver_t
// (query-axis blocked, two-pass softmax; bf16/f16 and an additive attention
// mask) and sdp_fused_driver_t (online/flash softmax; f32-only, no mask yet).
// pd_t::init() picks one by shape/dtype capability, or a forced choice via
// ONEDNN_SDPA_IMPL={blocked,fused,auto} for debugging/perf comparisons.
enum class sdpa_driver_kind_t { blocked, fused };

struct brgemm_sdpa_fwd_t : public primitive_t {
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T("brg_sdpa:any", brgemm_sdpa_fwd_t);

        status_t init(const engine_t *engine);

        sdpa_driver_kind_t driver_kind() const { return kind_; }
        const sdp_blocked_params_t &blocked_params() const { return bp_; }
        const sdp_fused_params_t &fused_params() const { return fp_; }
        int nthr() const { return nthr_; }

    private:
        friend struct brgemm_sdpa_fwd_t;

        sdpa_driver_kind_t kind_ = sdpa_driver_kind_t::blocked;
        // Plain compute parameters derived from this pd's memory descriptors
        // (see init()); only the struct matching kind_ is populated. The
        // driver is JIT-compiled by the primitive (brgemm_sdpa_fwd_t::init),
        // keeping the pd kernel-free; configure() is used here only to size the
        // scratchpad.
        sdp_blocked_params_t bp_;
        sdp_fused_params_t fp_;
        int nthr_ = 1;
    };

    brgemm_sdpa_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    // Owned compute driver, JIT-compiled in init(); only the one matching
    // pd()->driver_kind() is created.
    std::shared_ptr<sdp_blocked_driver_t> blocked_driver_;
    std::shared_ptr<sdp_fused_driver_t> fused_driver_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
