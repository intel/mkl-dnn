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

#include "common/verbose.hpp"

#include "cpu/x64/sdpa/brgemm_sdpa.hpp"

#define VCONDCHECK_BRGEMM_SDPA(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemm_sdpa, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

status_t brgemm_sdpa_fwd_t::pd_t::init(const engine_t *engine) {
    UNUSED(engine);
    VCONDCHECK_BRGEMM_SDPA(
            false, "CPU SDPA BRGEMM implementation is currently unavailable");
    return status::unimplemented;
}

status_t brgemm_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    UNUSED(ctx);
    VCONDCHECK(primitive, exec, check, brgemm_sdpa, false,
            status::unimplemented,
            "CPU SDPA BRGEMM implementation is currently unavailable");
    return status::unimplemented;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
