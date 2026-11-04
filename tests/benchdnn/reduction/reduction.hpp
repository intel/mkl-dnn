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

#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace reduction {

enum alg_t {
    undef,
    min,
    max,
    mul,
    sum,
    mean,
    norm_lp_max,
    norm_lp_sum,
    norm_lp_power_p_max,
    norm_lp_power_p_sum,
    reduction_min = min,
    reduction_max = max,
    reduction_mul = mul,
    reduction_sum = sum,
    reduction_mean = mean,
    reduction_norm_lp_max = norm_lp_max,
    reduction_norm_lp_sum = norm_lp_sum,
    reduction_norm_lp_power_p_max = norm_lp_power_p_max,
    reduction_norm_lp_power_p_sum = norm_lp_power_p_sum,
};

alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    std::vector<dnnl_data_type_t> sdt {dnnl_f32};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx};
    std::vector<std::string> dtag {tag::any};
    std::vector<alg_t> alg {alg_t::sum};
    std::vector<float> p {1.0f}, eps {0.0f};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%ddt%,%stag%,%dtag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return sdt.size() == 1 && ddt.size() == 1 && stag.size() == 1
                && dtag.size() == 1 && alg.size() == 1 && p.size() == 1
                && eps.size() == 1 && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_vdims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.sdt[0], s.ddt[0], s.stag[0], s.dtag[0], s.alg[0],
                  s.p[0], s.eps[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::string &stag,
            const std::string &dtag, alg_t alg, float p, float eps,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , base_prb_t(FLAG_FWD, false, attr, ctx_init, ctx_exe, impl_filter)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , p(p)
        , eps(eps) {
        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    dnnl_data_type_t sdt, ddt;
    std::string stag, dtag;
    alg_t alg;
    float p, eps;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

private:
    std::string set_repro_line() override;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , sdt_({p_->sdt})
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

dnnl_status_t init_pd(init_pd_args_t &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int bench(int argc, char **argv);

} // namespace reduction

#endif
