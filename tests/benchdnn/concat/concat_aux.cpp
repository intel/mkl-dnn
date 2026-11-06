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

#include "concat/concat.hpp"
#include "dnnl_debug.hpp"
#include "utils/stringstream.hpp"

namespace concat {

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    using ::operator<<;

    stringstream_t s;
    settings_t def;

    bool has_default_tags = true;
    for (const auto &i_stag : stag)
        has_default_tags = has_default_tags && i_stag == tag::abx;

    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || (dtag != def.dtag[0] && ddt != def.ddt[0]))
        s << "--ddt=" << ddt << " ";
    if (canonical || !has_default_tags) s << "--stag=" << stag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || axis != def.axis[0]) s << "--axis=" << axis << " ";

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

} // namespace concat
