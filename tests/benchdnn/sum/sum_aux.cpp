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

#include "dnnl_debug.hpp"
#include "sum/sum.hpp"
#include "utils/stringstream.hpp"

namespace sum {

std::ostream &operator<<(
        std::ostream &s, const std::vector<float> &input_scales) {
    bool has_single_scale = true;
    for (size_t d = 0; d < input_scales.size() - 1; ++d)
        has_single_scale
                = has_single_scale && input_scales[d] == input_scales[d + 1];

    s << input_scales[0];
    if (!has_single_scale)
        for (size_t d = 1; d < input_scales.size(); ++d)
            s << ":" << input_scales[d];
    return s;
}

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    using ::operator<<;
    using sum::operator<<;

    stringstream_t s;
    settings_t def;

    bool has_default_tags = true;
    for (const auto &i_stag : stag)
        has_default_tags = has_default_tags && i_stag == tag::abx;

    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || ddt != def.ddt[0]) s << "--ddt=" << ddt << " ";
    if (canonical || !has_default_tags) s << "--stag=" << stag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    s << "--scales=" << input_scales << " ";
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

} // namespace sum
