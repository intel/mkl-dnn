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

// Dev-mode C entry points so scripts/sdpa_tuner can evaluate the forward
// cost model through ctypes instead of keeping a copy of it in Python.
// Strings use the formats of the fwd_select verbose line (hw:, prb:, cfg:)

#include <algorithm>

#include "oneapi/dnnl/dnnl_config.h"

#include "gpu/intel/sdpa/select.hpp"

#ifdef DNNL_DEV_MODE

namespace {

namespace sdpa = dnnl::impl::gpu::intel::sdpa;
namespace compute = dnnl::impl::gpu::intel::compute;

// coefs == nullptr means the architecture's seeds
bool coefs_from(const double *coefs, int count, compute::gpu_arch_t arch,
        sdpa::fwd_model_coefs_t &m) {
    m = sdpa::fwd_model_coefs(arch);
    if (!coefs) return true;
    if (count != (int)sdpa::fwd_model_coef_names().size()) return false;
    for (int i = 0; i < count; i++)
        sdpa::fwd_model_coef_set(m, i, coefs[i]);
    return true;
}

} // namespace

extern "C" {

DNNL_API int dnnl_impl_sdpa_fwd_coef_count(void) {
    return (int)sdpa::fwd_model_coef_names().size();
}

DNNL_API const char *dnnl_impl_sdpa_fwd_coef_name(int idx) {
    const auto &names = sdpa::fwd_model_coef_names();
    if (idx < 0 || idx >= (int)names.size()) return nullptr;
    return names[idx].c_str();
}

// Seeds of the architecture named as device_info prints it ("xe_hpg")
DNNL_API int dnnl_impl_sdpa_fwd_coef_seeds(
        const char *arch, double *coefs, int count) {
    if (!arch || !coefs) return -1;
    const auto a = compute::str2gpu_arch(arch);
    if (a == compute::gpu_arch_t::unknown
            || count != (int)sdpa::fwd_model_coef_names().size())
        return -1;
    const sdpa::fwd_model_coefs_t m = sdpa::fwd_model_coefs(a);
    for (int i = 0; i < count; i++)
        coefs[i] = sdpa::fwd_model_coef_get(m, i);
    return 0;
}

// Estimates one config. Returns 0 and fills cost_us and, when given,
// derived[6] = {kv_tile, q_tile, sg_per_wg, slm_bytes, grfs,
// wg_per_subslice}; 1 when the config is invalid for the problem; -1 when a
// string does not parse
DNNL_API int dnnl_impl_sdpa_fwd_estimate(const char *hw_str,
        const char *prb_str, const char *cfg_str, const double *coefs,
        int count, double *cost_us, int *derived) {
    if (!hw_str || !prb_str || !cfg_str) return -1;
    sdpa::fwd_hw_t hw;
    sdpa::fwd_problem_t p;
    sdpa::fwd_config_t c;
    if (!sdpa::fwd_parse_hw(hw_str, hw) || !sdpa::fwd_parse_problem(prb_str, p)
            || !sdpa::fwd_parse_config(cfg_str, c))
        return -1;
    sdpa::fwd_model_coefs_t m;
    if (!coefs_from(coefs, count, hw.arch, m)) return -1;
    sdpa::fwd_candidate_t cand;
    if (!sdpa::fwd_describe(c, p, hw, m, cand)) return 1;
    if (cost_us) *cost_us = cand.cost_us;
    if (derived) {
        derived[0] = cand.kv_tile;
        derived[1] = cand.q_tile;
        derived[2] = cand.sg_per_wg;
        derived[3] = cand.slm_bytes;
        derived[4] = cand.grfs;
        derived[5] = cand.wg_per_subslice;
    }
    return 0;
}

// Valid configs, best estimate first, 8 ints each. Writes up to max_configs
// of them into configs (may be null) and returns the total count, or -1
// when a string does not parse
DNNL_API int dnnl_impl_sdpa_fwd_enumerate(const char *hw_str,
        const char *prb_str, const double *coefs, int count, int *configs,
        int max_configs) {
    if (!hw_str || !prb_str) return -1;
    sdpa::fwd_hw_t hw;
    sdpa::fwd_problem_t p;
    if (!sdpa::fwd_parse_hw(hw_str, hw) || !sdpa::fwd_parse_problem(prb_str, p))
        return -1;
    sdpa::fwd_model_coefs_t m;
    if (!coefs_from(coefs, count, hw.arch, m)) return -1;
    const auto ranked = sdpa::fwd_enumerate(p, hw, m);
    const int n = configs ? std::min((int)ranked.size(), max_configs) : 0;
    for (int i = 0; i < n; i++) {
        const auto &c = ranked[i].config;
        const int v[8] = {c.unroll_m_kq, c.unroll_n_kq, c.unroll_m_vs,
                c.unroll_n_vs, c.wg_m_kq, c.wg_n_kq, c.wg_m_vs, c.wg_n_vs};
        for (int j = 0; j < 8; j++)
            configs[8 * i + j] = v[j];
    }
    return (int)ranked.size();
}

} // extern "C"

#endif // DNNL_DEV_MODE
