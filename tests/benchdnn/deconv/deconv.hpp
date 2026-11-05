/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef DECONV_HPP
#define DECONV_HPP

#include <iostream>
#include <map>

#include <assert.h>
#include <stdint.h>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/compare.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace deconv {

enum alg_t {
    UNDEF,
    DIRECT,
    WINO,
    AUTO,
    deconvolution_direct = DIRECT,
    deconvolution_wino = WINO,
    deconvolution_auto = AUTO,
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
alg_t alg_kind2alg(dnnl_alg_kind_t alg);

struct desc_t {
    int64_t g, mb;
    int64_t ic, id, ih, iw;
    int64_t oc, od, oh, ow;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension
    int64_t dd, dh, dw;
    bool has_groups;

    std::string name;
    int ndims;

    // Initialize dependent opposite-side paddings values from the shape
    // parameters.
    void init_pad_r() {
        pw_r = opp_pad(iw, ow, kw, sw, pw, dw);
        ph_r = opp_pad(ih, oh, kh, sh, ph, dh);
        pd_r = opp_pad(id, od, kd, sd, pd, dd);
    }

    int64_t desc_nelems(int arg, int mask) const;

    dims_t src_dims() const;
    dims_t wei_dims() const;
    dims_t bia_dims() const;
    dims_t dst_dims() const;
    dims_t strides() const;
    dims_t dilations() const;
    dims_t padding() const;
    dims_t padding_r() const;

private:
    int64_t opp_pad(int64_t i, int64_t o, int64_t k, int64_t s, int64_t p,
            int64_t d) const {
        return (i - 1) * s - o + ((k - 1) * (d + 1) + 1) - p;
    }
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    desc_t desc {};

    std::vector<dir_t> dir {FWD_B};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};
    std::vector<alg_t> alg {DIRECT};

    const char *perf_template_csv() const {
        static const std::string args
                = "%dir%,%sdt%,%bia_dt%,%stag%,%wtag%,%dtag%,%alg%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && bia_dt.size() == 1
                && stag.size() == 1 && wtag.size() == 1 && dtag.size() == 1
                && alg.size() == 1 && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.dir[0], s.dt[0], s.bia_dt[0], s.stag[0], s.wtag[0],
                  s.dtag[0], s.alg[0], s.mb[0], s.attributes.front(),
                  s.ctx_init[0], s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt, dnnl_data_type_t bia_dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, alg_t alg, int64_t mb, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : desc_t(desc)
        , base_prb_t(dir, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , bia_dt_(bia_dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , alg(alg)
        , user_mb(mb)
        , ops(0) {
        if (mb) this->mb = mb;

        // Broadcast data types if needed
        broadcast_vector(this->dt, dt[0], 3);

        count_ops();
        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    std::vector<dnnl_data_type_t> dt;
    dnnl_data_type_t bia_dt_; // `_` to avoid conflicting name with bia_dt().
    std::string stag, wtag, dtag;
    alg_t alg;
    int64_t user_mb;
    double ops;
    void count_ops();
    int64_t count_n_acc() const {
        return (dir & FLAG_FWD)    ? kd * kh * kw * ic / g
                : (dir & FLAG_WEI) ? od * oh * ow * mb
                                   : kd * kh * kw * oc / g;
    }

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t wei_dt() const { return dt[1]; }
    dnnl_data_type_t bia_dt() const { return bia_dt_; }
    dnnl_data_type_t dst_dt() const { return dt[2]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

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
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    double ops() const override { return p_->ops; }
    const attr_t *attr() const override { return &p_->attr; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const dnnl_data_type_t *bia_dt() const override { return &p_->bia_dt_; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }
    const std::string *wtag() const override { return &wtag_; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;

    float get_density(const density_args_t &density_args) const override;
};

int transpose_data_wei(
        const prb_t *prb, const dnn_mem_t &wei, const dnn_mem_t &wei_tr);

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t ic,
        int64_t id, int64_t ih, int64_t iw) {
    return (((mb * prb->ic + g * prb->ic / prb->g + ic) * prb->id + id)
                           * prb->ih
                   + ih)
            * prb->iw
            + iw;
}

inline int64_t wei_off_f(const prb_t *prb, int64_t g, int64_t oc, int64_t ic,
        int64_t kd, int64_t kh, int64_t kw) {
    return ((((g * prb->oc / prb->g + oc) * prb->ic / prb->g + ic) * prb->kd
                    + kd) * prb->kh
                   + kh)
            * prb->kw
            + kw;
}

inline int64_t bia_off_f(const prb_t *prb, int64_t g, int64_t oc) {
    return g * prb->oc / prb->g + oc;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t g, int64_t oc,
        int64_t od, int64_t oh, int64_t ow) {
    return (((mb * prb->oc + g * prb->oc / prb->g + oc) * prb->od + od)
                           * prb->oh
                   + oh)
            * prb->ow
            + ow;
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

} // namespace deconv

#endif
