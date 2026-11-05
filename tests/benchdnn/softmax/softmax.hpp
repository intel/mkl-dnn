/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace softmax {

enum alg_t {
    UNDEF,
    SOFTMAX,
    LOGSOFTMAX,
    SOFTMAX_INF_AS_ZERO,
    softmax_accurate = SOFTMAX,
    softmax_log = LOGSOFTMAX,
    softmax_accurate_inf_as_zero = SOFTMAX_INF_AS_ZERO,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_dims_t prb_dims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx}, dtag {tag::any};
    std::vector<alg_t> alg {SOFTMAX};
    std::vector<int> axis {1};
    std::vector<bool> has_stats {false};

    const char *perf_template_csv() const {
        static const std::string args
                = "%dir%,%sdt%,%ddt%,%stag%,%dtag%,%alg%,%axis%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && sdt.size() == 1 && ddt.size() == 1
                && stag.size() == 1 && dtag.size() == 1 && alg.size() == 1
                && axis.size() == 1 && has_stats.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_dims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_dims, s.dir[0], s.sdt[0], s.ddt[0], s.stag[0], s.dtag[0],
                  s.alg[0], s.axis[0], s.has_stats[0], s.mb[0], s.inplace[0],
                  s.attributes.front(), s.ctx_init[0], s.ctx_exe[0],
                  s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_dims_t &prb_dims, dir_t dir, dnnl_data_type_t sdt,
            dnnl_data_type_t ddt, const std::string &stag,
            const std::string &dtag, alg_t alg, int axis, bool has_stats,
            int64_t mb, bool inplace, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : prb_dims_t(prb_dims)
        , base_prb_t(dir, inplace, attr, ctx_init, ctx_exe, impl_filter)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , axis(axis)
        , has_stats(has_stats)
        , user_mb(mb) {
        if (mb) dims[0] = mb;
        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    dnnl_data_type_t sdt, ddt;
    std::string stag, dtag;
    alg_t alg;
    int axis;
    bool has_stats;
    int64_t user_mb;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

    std::unique_ptr<base_perf_report_t> create_perf_report(
            const char *perf_template) const override;

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
        s << static_cast<const prb_dims_t &>(*p_);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const int *axis() const override { return &p_->axis; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

inline void map_off_to_mb_ic(
        const prb_t *prb, int64_t off, int64_t &mb, int64_t &ic) {
    for (int i = prb->ndims - 1; i > 1; i--)
        off /= prb->dims[i];

    ic = off % prb->dims[1];
    off /= prb->dims[1];
    mb = off % prb->dims[0];
    off /= prb->dims[0];
    assert(off == 0);
}

inline void get_sizes(const prb_t *prb, int64_t &outer_size,
        int64_t &inner_size, int64_t &axis_size) {
    outer_size = inner_size = axis_size = 1;
    for (int i = 0; i < prb->axis; i++)
        outer_size *= prb->dims[i];
    for (int i = prb->axis + 1; i < prb->ndims; i++)
        inner_size *= prb->dims[i];
    axis_size = prb->dims[prb->axis];
}

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

} // namespace softmax

#endif
