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

#ifndef REORDER_HPP
#define REORDER_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace reorder {

enum flag_bit_t {
    FLAG_NONE = 0x0U,
    FLAG_S8S8_COMP = 0x1U,
    FLAG_ZP_COMP = 0x2U,
    FLAG_ANY = ~FLAG_NONE, // For internal use only.
};
using flag_t = std::pair<flag_bit_t, int>;
flag_t str2flag(const char *str);
std::string flag2str(flag_bit_t flag);
std::ostream &operator<<(std::ostream &s, const std::vector<flag_t> &oflag);

struct dt_conf_t {
    dnnl_data_type_t dt;
    float min;
    float max;
};
const dt_conf_t *dt2cfg(dnnl_data_type_t dt);
dnnl_data_type_t cfg2dt(const dt_conf_t *cfg);

enum cross_engine_t { NONE, CPU2GPU, GPU2CPU };
cross_engine_t str2cross_engine(const char *str);
const char *cross_engine2str(cross_engine_t cross_engine);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_dims_t prb_dims;

    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
    std::vector<std::string> stag {tag::undef}, dtag {tag::undef};
    std::vector<vdims_t> strides {vdims_t(2)};
    std::vector<std::vector<flag_t>> oflag {{{FLAG_NONE, 0}}};
    std::vector<unsigned> runtime_dim_mask {0};
    std::vector<cross_engine_t> cross_engine {NONE};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%ddt%,%stag%,%dtag%,%flags%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dt.size() == 1 && stag.size() == 1 && dtag.size() == 1
                && strides.size() == 1 && oflag.size() == 1
                && runtime_dim_mask.size() == 1 && cross_engine.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_dims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_dims, s.dt[0], s.stag[0], s.dtag[0], s.strides[0],
                  s.oflag[0], s.cross_engine[0], s.runtime_dim_mask[0],
                  s.attributes.front(), s.ctx_init[0], s.ctx_exe[0],
                  s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_dims_t &prb_dims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &dtag,
            const vdims_t &strides, const std::vector<flag_t> &oflag,
            cross_engine_t cross_engine, unsigned runtime_dim_mask,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, const impl_filter_t &impl_filter)
        : prb_dims_t(prb_dims)
        , base_prb_t(FLAG_FWD, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , stag(stag)
        , dtag(dtag)
        , strides(strides)
        , oflag(oflag)
        , cross_engine(cross_engine)
        , runtime_dim_mask(runtime_dim_mask) {

        broadcast_vector(this->dt, dt[0], 2);

        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    std::vector<dnnl_data_type_t> dt;
    std::string stag, dtag;
    vdims_t strides;
    std::vector<flag_t> oflag;
    cross_engine_t cross_engine;
    unsigned runtime_dim_mask;

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t dst_dt() const { return dt[1]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    bool is_reorder_with_compensation(flag_bit_t flag) const;
    dims_t get_compensation_dims(flag_bit_t flag) const;
    int get_compensation_mask(flag_bit_t flag) const;
    const dt_conf_t *get_conf(data_kind_t kind) const;

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const override;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

private:
    std::string set_repro_line() override;

    void get_compensation_parameters(
            dims_t &comp_dims, int &mask, flag_bit_t flag) const;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_engine(std::ostream &s) const override {
        if (p_->cross_engine == CPU2GPU)
            s << "cpu2gpu";
        else if (p_->cross_engine == GPU2CPU)
            s << "gpu2cpu";
        else
            base_perf_report_t::dump_engine(s);
    }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    void dump_flags(std::ostream &s) const override { s << p_->oflag; }

    const attr_t *attr() const override { return &p_->attr; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }

private:
    const prb_t *p_;
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

} // namespace reorder

#endif
