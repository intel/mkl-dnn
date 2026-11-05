/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "reduction.hpp"
#include "utils/stringstream.hpp"

namespace reduction {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(max);
    CASE(reduction_max);
    CASE(min);
    CASE(reduction_min);
    CASE(sum);
    CASE(reduction_sum);
    CASE(mul);
    CASE(reduction_mul);
    CASE(mean);
    CASE(reduction_mean);
    CASE(norm_lp_max);
    CASE(reduction_norm_lp_max);
    CASE(norm_lp_sum);
    CASE(reduction_norm_lp_sum);
    CASE(norm_lp_power_p_max);
    CASE(reduction_norm_lp_power_p_max);
    CASE(norm_lp_power_p_sum);
    CASE(reduction_norm_lp_power_p_sum);

#undef CASE
    assert(!"unknown algorithm");
    return undef;
}

const char *alg2str(alg_t alg) {
    if (alg == max) return "max";
    if (alg == min) return "min";
    if (alg == sum) return "sum";
    if (alg == mul) return "mul";
    if (alg == mean) return "mean";
    if (alg == norm_lp_max) return "norm_lp_max";
    if (alg == norm_lp_sum) return "norm_lp_sum";
    if (alg == norm_lp_power_p_max) return "norm_lp_power_p_max";
    if (alg == norm_lp_power_p_sum) return "norm_lp_power_p_sum";
    assert(!"unknown algorithm");
    return "undef";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == max) return dnnl_reduction_max;
    if (alg == min) return dnnl_reduction_min;
    if (alg == sum) return dnnl_reduction_sum;
    if (alg == mul) return dnnl_reduction_mul;
    if (alg == mean) return dnnl_reduction_mean;
    if (alg == norm_lp_max) return dnnl_reduction_norm_lp_max;
    if (alg == norm_lp_sum) return dnnl_reduction_norm_lp_sum;
    if (alg == norm_lp_power_p_max) return dnnl_reduction_norm_lp_power_p_max;
    if (alg == norm_lp_power_p_sum) return dnnl_reduction_norm_lp_power_p_sum;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || ddt != def.ddt[0]) s << "--ddt=" << ddt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || alg != def.alg[0]) s << "--alg=" << alg2str(alg) << " ";
    if (canonical || p != def.p[0]) s << "--p=" << p << " ";
    if (canonical || eps != def.eps[0]) s << "--eps=" << eps << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<prb_vdims_t>(*this);

    return s.str();
}

} // namespace reduction
