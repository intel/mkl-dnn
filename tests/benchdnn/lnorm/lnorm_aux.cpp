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

#include <assert.h>
#include <stdlib.h>
#include "lnorm/lnorm.hpp"
#include "utils/stringstream.hpp"

namespace lnorm {

flags_t str2flags(const char *str) {
    flags_t flags = NONE;
    // Iterate with a separate cursor so `str` still points at the whole value and the
    // error below can report what the user actually passed rather than its tail.
    for (const char *s = str; s && *s; s++) {
        if (*s == 'G') {
            flags |= GLOB_STATS;
        } else if (*s == 'C') {
            flags |= USE_SCALE;
        } else if (*s == 'H') {
            flags |= USE_SHIFT;
        } else if (*s == 'M') {
            flags |= USE_RMS_NORM;
        } else {
            BENCHDNN_PRINT(0, "%s \'%s\'\n",
                    "Error: --flags option doesn't support value", str);
            SAFE_V(FAIL);
        }
    }
    return flags;
}

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if (canonical || tag != def.tag[0]) {
        s << "--tag=";
        if (tag[1] != def.tag[0][1])
            s << tag[0] << ":" << tag[1] << " ";
        else
            s << tag[0] << " ";
    }
    if (canonical || stat_tag != def.stat_tag[0])
        s << "--stat_tag=" << stat_tag << " ";
    if (canonical || ss_dt != def.ss_dt[0]) s << "--ss_dt=" << ss_dt << " ";
    if (canonical || flags != def.flags[0])
        s << "--flags=" << flags2str(flags) << " ";
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
} // namespace lnorm
