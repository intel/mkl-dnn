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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <algorithm>
#include <bitset>
#include <iostream>
#include <map>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace matmul {

using dims_mask_t = std::bitset<DNNL_MAX_NDIMS>;

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> stag {tag::any}, wtag {tag::any}, dtag {tag::any};
    std::vector<sparse_options_t> sparse_options {{DNNL_ARG_SRC,
            sparse_options_t::def_encoding, sparse_options_t::def_sparsity}};
    std::vector<vdims_t> strides {vdims_t(STRIDES_SIZE)};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> bia_mask {2};
    std::vector<std::vector<dims_mask_t>> rt_dims_masks {{}};
    std::vector<std::string> gpu_kernel {""};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%bia_dt%,%stag%,%wtag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dt.size() == 1 && stag.size() == 1 && wtag.size() == 1
                && dtag.size() == 1 && strides.size() == 1 && bia_dt.size() == 1
                && bia_mask.size() == 1 && rt_dims_masks.size() == 1
                && gpu_kernel.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_vdims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.dt[0], s.stag[0], s.wtag[0], s.dtag[0],
                  s.strides[0], s.bia_dt[0], s.bia_mask[0], s.rt_dims_masks[0],
                  s.sparse_options[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter, s.gpu_kernel[0]) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const vdims_t &strides,
            dnnl_data_type_t bia_dt, int bia_mask,
            const std::vector<dims_mask_t> &rt_dims_masks,
            const sparse_options_t &sparse_options, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter,
            const std::string &gpu_kernel = {})
        : prb_vdims_t(prb_vdims)
        , base_prb_t(FLAG_FWD, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , strides(strides)
        , bia_dt(bia_dt)
        , bia_mask(bia_mask)
        , rt_dims_masks(rt_dims_masks)
        , sparse_options(sparse_options)
        , gpu_kernel(gpu_kernel) {

        // Broadcast data types if needed
        broadcast_vector(this->dt, dt[0], 3);

        this->rt_dims_masks.resize(2);
        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();

        // Generate group_sizes according to the profile, if they are not read
        if (this->sparse_options.has_deferred_profile()) {
            const int vdi
                    = this->sparse_options.get_variable_dim_idx(DNNL_ARG_SRC);
            SAFE_V(this->sparse_options.resolve_profile(srcdims[vdi]));
        }

        dst_dims[ndims - 2] = m;
        dst_dims[ndims - 1] = n;

        init_dst_rt_dims_mask();
        init_bias_rt_dims_mask();
        mb = std::accumulate(dst_dims.begin(), dst_dims.end() - 2,
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        const auto nelems = std::accumulate(dst_dims.begin(), dst_dims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k;
        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    int64_t m, n, k, mb;
    std::vector<dnnl_data_type_t> dt;
    std::string stag, wtag, dtag;
    vdims_t strides;
    dnnl_data_type_t bia_dt;
    int bia_mask;
    std::vector<dims_mask_t> rt_dims_masks;
    sparse_options_t sparse_options;
    std::string gpu_kernel;
    double ops;

    const dims_t &src_dims() const { return vdims[0]; }
    const dims_t &weights_dims() const { return vdims[1]; }
    dims_t bia_dims() const {
        dims_t dims(ndims, 1);
        for (int d = 0; d < ndims; ++d)
            if (bia_mask & (1 << d)) dims[d] = dst_dims[d];
        return dims;
    }
    // dims_t prb_vdims_t::dst_dims; // A member in `prb_vdims_t`.

    const dims_mask_t &src_runtime_dim_mask() const { return rt_dims_masks[0]; }
    const dims_mask_t &weights_runtime_dim_mask() const {
        return rt_dims_masks[1];
    }
    const dims_mask_t &dst_runtime_dim_mask() const { return rt_dims_masks[2]; }
    const dims_mask_t &bias_runtime_dim_mask() const {
        return rt_dims_masks[3];
    }

    int src_broadcast_mask() const {
        return prb_vdims_t::get_broadcast_mask(0);
    }
    int weights_broadcast_mask() const {
        return prb_vdims_t::get_broadcast_mask(1);
    }
    int bias_broadcast_mask() const { return bia_mask; }

    bool weights_decompression() const {
        return src_dt() != dnnl_s8 && src_dt() != dnnl_u8
                && (wei_dt() == dnnl_s8 || wei_dt() == dnnl_u8
                        || wei_dt() == dnnl_s4 || wei_dt() == dnnl_u4)
                && attr.fpmath_mode.apply_to_int;
    }

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t wei_dt() const { return dt[1]; }
    dnnl_data_type_t dst_dt() const { return dt[2]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

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

    void init_dst_rt_dims_mask() {
        if (rt_dims_masks.size() > 2) return;

        const auto &src_rt_dim_mask = src_runtime_dim_mask();
        const auto &wei_rt_dim_mask = weights_runtime_dim_mask();
        dims_mask_t dst_rt_dim_mask;

        for (int i = 0; i < ndims - 2; ++i) {
            dst_rt_dim_mask[i] = src_rt_dim_mask[i] || wei_rt_dim_mask[i];
        }

        // m, n mask
        dst_rt_dim_mask[ndims - 2] = src_rt_dim_mask[ndims - 2];
        dst_rt_dim_mask[ndims - 1] = wei_rt_dim_mask[ndims - 1];

        rt_dims_masks.push_back(dst_rt_dim_mask);
    }

    void init_bias_rt_dims_mask() {
        if (rt_dims_masks.size() != 3) return;

        const auto &dst_rt_dim_mask = dst_runtime_dim_mask();
        dims_mask_t bias_rt_dim_mask;

        for (int d = 0; d < ndims; ++d)
            bias_rt_dim_mask[d] = (bia_mask & (1 << d)) && dst_rt_dim_mask[d];

        rt_dims_masks.push_back(bias_rt_dim_mask);
    }
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }
    double ops() const override { return p_->ops; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const dnnl_data_type_t *bia_dt() const override { return &p_->bia_dt; }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

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

inline int64_t src_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t k) {
    return (mb * prb->m + m) * prb->k + k;
}

inline int64_t dst_off_f(const prb_t *prb, int64_t mb, int64_t m, int64_t n) {
    return (mb * prb->m + m) * prb->n + n;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args);
#ifdef DNNL_DEV_MODE
// Number of jit:gemm kernel candidates for `prb`, or -1 if unavailable.
int query_gpu_kernel_count(const prb_t *prb);
#endif
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

} // namespace matmul

#endif
