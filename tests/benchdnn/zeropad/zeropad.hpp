/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef ZEROPAD_HPP
#define ZEROPAD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace zeropad {

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_dims_t prb_dims;

    std::vector<dnnl_data_type_t> dt {dnnl_f32};
    std::vector<std::string> tag {tag::abx};

    const char *perf_template_csv() const {
        static const std::string args = "%dt%,%tag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dt.size() == 1 && tag.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_dims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s) : prb_t(s.prb_dims, s.dt[0], s.tag[0]) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_dims_t &prb_dims, dnnl_data_type_t dt,
            const std::string &tag)
        : prb_dims_t(prb_dims), dt(dt), tag(tag) {
        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    dnnl_data_type_t dt;
    std::string tag;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;

    std::unique_ptr<base_perf_report_t> create_perf_report(
            const char *perf_template) const override;

private:
    std::string set_repro_line() override;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , tag_(normalize_tag(p_->tag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    const std::string *name() const override { return &p_->name; }
    const dnnl_data_type_t *dt() const override { return &p_->dt; }
    const std::string *tag() const override { return &tag_; }

private:
    const prb_t *p_;
    std::string tag_;
};

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace zeropad

#endif
