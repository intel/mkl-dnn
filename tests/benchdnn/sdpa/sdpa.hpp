/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef SDPA_HPP
#define SDPA_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

// C API for SDPA primitive creation (forward and backward overloads).
#include "common/sdpa_test_iface.hpp"

// Semantic argument aliases for SDPA tensors.
// Mirrors the macros in src/common/sdpa_types.hpp without pulling in internal
// dependencies (op_desc_t, etc.) that benchdnn cannot resolve.
#ifndef DNNL_ARG_QUERIES
#define DNNL_ARG_QUERIES DNNL_ARG_SRC_0
#define DNNL_ARG_KEYS DNNL_ARG_SRC_1
#define DNNL_ARG_VALUES DNNL_ARG_SRC_2
#define DNNL_ARG_ATTN_MASK DNNL_ARG_SHIFT
#define DNNL_ARG_DIFF_QUERIES DNNL_ARG_DIFF_SRC_0
#define DNNL_ARG_DIFF_KEYS DNNL_ARG_DIFF_SRC_1
#define DNNL_ARG_DIFF_VALUES DNNL_ARG_DIFF_SRC_2
#define DNNL_ARG_DS DNNL_ARG_DIFF_SRC_3
#endif

namespace sdpa {

// Internal (reference-only) arg key for the per-element conditioning magnitude
// sum_k prob_k*|V_k|, filled by compute_ref and read by setup_cmp to size a
// tighter per-element DST threshold. Negative so init_ref_memory_args skips it
// (it only fills positive args) and it is never treated as a compared kind.
static constexpr int SDPA_REF_ARG_OUT_ABSMAG = -1000;
// Backward counterparts: the dQ/dK/dV contractions over absolute values.
static constexpr int SDPA_REF_ARG_DQ_ABSMAG = -1001;
static constexpr int SDPA_REF_ARG_DK_ABSMAG = -1002;
static constexpr int SDPA_REF_ARG_DV_ABSMAG = -1003;

enum mask_type_t {
    MASK_NONE = 0,
    MASK_BUFFER = 1,
    MASK_BUFFER_1D = 2,
    MASK_BUFFER_2D = 3,
    MASK_CAUSAL_TOP_LEFT = 4,
    MASK_CAUSAL_BOTTOM_RIGHT = 5,
};
mask_type_t str2mask_type(const char *str);
const char *mask_type2str(mask_type_t mt);

enum scale_type_t {
    SCALE_LIBRARY = 0,
    SCALE_MUL = 1,
    SCALE_DIV = 2,
};
scale_type_t str2scale_type(const char *str);
const char *scale_type2str(scale_type_t st);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    std::vector<dir_t> dir {FWD_I};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> qtag {tag::abx}, ktag {tag::abx}, vtag {tag::abx},
            dtag {tag::abx};
    std::vector<dnnl_data_type_t> mdt {dnnl_f32};
    std::vector<mask_type_t> mask_type {MASK_NONE};
    std::vector<scale_type_t> scale_type {SCALE_LIBRARY};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%stag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && qtag.size() == 1
                && ktag.size() == 1 && vtag.size() == 1 && dtag.size() == 1
                && mdt.size() == 1 && mask_type.size() == 1
                && scale_type.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_vdims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.dir[0], s.dt[0], s.qtag[0], s.ktag[0], s.vtag[0],
                  s.dtag[0], s.mdt[0], s.mask_type[0], s.scale_type[0],
                  s.attributes.front(), s.ctx_init[0], s.ctx_exe[0],
                  s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt, const std::string &qtag,
            const std::string &ktag, const std::string &vtag,
            const std::string &dtag, dnnl_data_type_t mdt,
            mask_type_t mask_type, scale_type_t scale_type, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , base_prb_t(dir, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , qtag(qtag)
        , ktag(ktag)
        , vtag(vtag)
        , dtag(dtag)
        , mdt(mdt)
        , mask_type(mask_type)
        , scale_type(scale_type) {

        // Broadcast data types if needed: Q,K,V,DST
        broadcast_vector(this->dt, dt[0], 4);

        const auto &qdims = q_dims();
        const auto &kdims = k_dims();
        const auto &vdims_ref = v_dims();
        n_queries = qdims[ndims - 2];
        head_size = qdims[ndims - 1];
        n_keys = kdims[ndims - 1];
        n_values = vdims_ref[ndims - 1];

        // Compute dst_dims from Q and V dims.
        dst_dims.resize(ndims);
        for (int i = 0; i < ndims - 2; i++)
            dst_dims[i] = qdims[i];
        dst_dims[ndims - 2] = n_queries;
        dst_dims[ndims - 1] = n_values;

        // Score dims (batch x heads x seq_q x seq_kv) — used for mask,
        // dropout, and backward's dS tensor.
        score_dims.resize(ndims);
        for (int i = 0; i < ndims - 2; i++)
            score_dims[i] = qdims[i];
        score_dims[ndims - 2] = n_queries;
        score_dims[ndims - 1] = n_keys;

        // Compute mask dims based on mask_type and broadcasting.
        if (with_mask()) {
            msk_dims.resize(ndims);
            // batch dims: 1 for all broadcast variants.
            for (int i = 0; i < ndims - 2; i++)
                msk_dims[i] = (mask_type == MASK_BUFFER) ? qdims[i] : 1;
            // S_q: 1 for 1D, full for 2D and buffer.
            msk_dims[ndims - 2] = (mask_type == MASK_BUFFER_1D) ? 1 : n_queries;
            // S_kv: always full.
            msk_dims[ndims - 1] = n_keys;
        }

        mb = 1;
        for (int i = 0; i < ndims - 2; i++)
            mb *= qdims[i];

        // Forward: 2 matmuls (Q*K^T and prob*V).
        // Backward: 5 matmuls (2.5x forward, cf. test_sdpa.cpp flash_flops).
        // Causal masks halve the effective flop count.
        double fwd_ops = 2.0 * mb * n_queries * n_keys * head_size
                + 2.0 * mb * n_queries * n_values * n_keys;
        if (with_causal_mask()) fwd_ops /= 2.0;
        ops = (dir & FLAG_BWD) ? 2.5 * fwd_ops : fwd_ops;

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    int64_t n_queries, head_size, n_keys, n_values, mb;
    std::vector<dnnl_data_type_t> dt;
    std::string qtag, ktag, vtag, dtag;
    dnnl_data_type_t mdt;
    mask_type_t mask_type;
    scale_type_t scale_type;

    double ops;
    dims_t dst_dims;
    dims_t msk_dims;
    dims_t score_dims;

    const dims_t &q_dims() const { return vdims[0]; }
    const dims_t &k_dims() const { return vdims[1]; }
    const dims_t &v_dims() const { return vdims[2]; }

    bool with_mask() const {
        return mask_type == MASK_BUFFER || mask_type == MASK_BUFFER_1D
                || mask_type == MASK_BUFFER_2D;
    }
    bool with_causal_mask() const {
        return mask_type == MASK_CAUSAL_TOP_LEFT
                || mask_type == MASK_CAUSAL_BOTTOM_RIGHT;
    }
    bool with_scale() const { return scale_type != SCALE_LIBRARY; }
    bool invert_scale() const { return scale_type == SCALE_DIV; }

    dnnl_data_type_t q_dt() const { return dt[0]; }
    dnnl_data_type_t k_dt() const { return dt[1]; }
    dnnl_data_type_t v_dt() const { return dt[2]; }
    dnnl_data_type_t dst_dt() const { return dt[3]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    // Required by init_memory_args template (for runtime dims support).
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
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , stag_({normalize_tag(p_->qtag, p_->ndims),
                  normalize_tag(p_->ktag, p_->ndims),
                  normalize_tag(p_->vtag, p_->ndims)})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    double ops() const override { return p_->ops; }
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

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;

    float get_density(const density_args_t &density_args) const override;
};

dnnl_status_t init_pd(init_pd_args_t &init_pd_args);
float intermediate_min_normal(const prb_t *prb);
void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);
size_t get_ref_extra_size(const prb_t *prb, dir_t dir);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);

int bench(int argc, char **argv);

} // namespace sdpa

#endif
