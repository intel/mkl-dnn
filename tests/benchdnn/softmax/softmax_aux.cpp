/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/stringstream.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(SOFTMAX);
    CASE(softmax_accurate);
    CASE(LOGSOFTMAX);
    CASE(softmax_log);
    CASE(SOFTMAX_INF_AS_ZERO);
    CASE(softmax_accurate_inf_as_zero);
#undef CASE
    assert(!"unknown algorithm");
    return UNDEF;
}

const char *alg2str(alg_t alg) {
    if (alg == SOFTMAX) return "SOFTMAX";
    if (alg == LOGSOFTMAX) return "LOGSOFTMAX";
    if (alg == SOFTMAX_INF_AS_ZERO) return "SOFTMAX_INF_AS_ZERO";
    assert(!"unknown algorithm");
    return "UNDEF";
}

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || ddt != def.ddt[0]) s << "--ddt=" << ddt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || alg != def.alg[0]) s << "--alg=" << alg2str(alg) << " ";
    if (canonical || axis != def.axis[0]) s << "--axis=" << axis << " ";
    if (canonical || inplace != def.inplace[0])
        s << "--inplace=" << bool2str(inplace) << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<prb_dims_t>(*this);

    return s.str();
}

} // namespace softmax
