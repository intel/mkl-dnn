/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef IP_HPP
#define IP_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace ip {

struct desc_t {
    int64_t mb, oc, ic, id, ih, iw;
    std::string name;
    int ndims;

    dims_t src_dims() const;
    dims_t wei_dims() const;
    dims_t bia_dims() const;
    dims_t dst_dims() const;
    int64_t desc_nelems(int arg, int mask) const;
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

    const char *perf_template_csv() const {
        static const std::string args
                = "%dir%,%sdt%,%bia_dt%,%stag%,%wtag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && bia_dt.size() == 1
                && stag.size() == 1 && wtag.size() == 1 && dtag.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.dir[0], s.dt[0], s.bia_dt[0], s.stag[0], s.wtag[0],
                  s.dtag[0], s.mb[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt, dnnl_data_type_t bia_dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, int64_t mb, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : desc_t(desc)
        , base_prb_t(dir, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , bia_dt_(bia_dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
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
    int64_t user_mb;
    double ops;
    void count_ops() {
        if (ops > 0) return;
        ops = 2. * mb * ic * oc * id * ih * iw;
    }
    int64_t count_n_acc() const {
        return (dir & FLAG_WEI) ? mb : id * ih * iw * ic;
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

inline size_t src_off_f(const prb_t *prb, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline size_t wei_off_f(const prb_t *prb, int64_t oc, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((oc * prb->ic + ic) * prb->id + id) * prb->ih + ih) * prb->iw + iw;
}

inline size_t bia_off_f(const prb_t *prb, int64_t oc) {
    return oc;
}

inline size_t dst_off_f(const prb_t *prb, int64_t mb, int64_t oc) {
    return mb * prb->oc + oc;
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
} // namespace ip

#endif
