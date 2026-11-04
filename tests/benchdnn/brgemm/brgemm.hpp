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

#ifndef BRGEMM_HPP
#define BRGEMM_HPP

#include <algorithm>
#include <iostream>
#include <numeric>

#include "oneapi/dnnl/dnnl.h"

#if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)

#if defined(DNNL_X64) && DNNL_X64 == 1
#include "src/cpu/x64/brgemm/brgemm.hpp"
#elif defined(DNNL_AARCH64) && DNNL_AARCH64 == 1
#include "src/cpu/aarch64/brgemm/brgemm.hpp"
#endif

#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)

#include "oneapi/dnnl/dnnl_ukernel.h"

#endif // !defined(DNNL_EXPERIMENTAL_UKERNEL)
#endif // (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace brgemm {

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> stag {tag::abx}, wtag {tag::abx}, dtag {tag::abx};
    std::vector<vdims_t> strides {vdims_t(STRIDES_SIZE)};
    std::vector<std::vector<int64_t>> ld {{}};
    std::vector<dnnl_data_type_t> bia_dt {dnnl_data_type_undef};
    std::vector<int> batch_size {1};
    std::vector<float> alpha {1.f}, beta {0.f};
    std::vector<std::string> brgemm_attr {std::string()};
    std::vector<std::string> batch_kind {"addr"};

    const char *perf_template_csv() const {
        static const std::string args;
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t, public base_prb_t {
    prb_t(const prb_vdims_t &prb_vdims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, const vdims_t &strides,
            const std::vector<int64_t> &ld, dnnl_data_type_t bia_dt,
            float alpha, float beta, int batch_size,
            const std::string &brgemm_attr, const std::string &batch_kind,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , base_prb_t(FWD_I, false, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , strides(strides)
        , ld(ld)
        , bia_dt(bia_dt)
        , alpha(alpha)
        , beta(beta)
        , batch_size(batch_size)
        , brgemm_attr(brgemm_attr)
        , batch_kind(batch_kind) {

        // Broadcast data types if needed
        broadcast_vector(this->dt, dt[0], 3);

        const auto &srcdims = src_dims();
        const auto &weidims = weights_dims();
        m = srcdims[ndims - 2];
        k = srcdims.back();
        n = weidims.back();
        dst_dims[ndims - 2] = m;
        dst_dims[ndims - 1] = n;

        const auto nelems = std::accumulate(dst_dims.begin(), dst_dims.end(),
                (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
        ops = 2. * nelems * k * batch_size;

        check_block_size();

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    int64_t m, n, k;
    // brgemm does not have any propagation kind. Treat as forward inference to
    // have correct check for data type support.
    std::vector<dnnl_data_type_t> dt;
    std::string stag, wtag, dtag;
    vdims_t strides;
    std::vector<int64_t> ld;
    dnnl_data_type_t bia_dt;
    float alpha, beta;
    int64_t batch_size;
    std::string brgemm_attr;
    std::string batch_kind;
    double ops;

    const dims_t &src_dims() const { return vdims[0]; }
    const dims_t &weights_dims() const { return vdims[1]; }
    // const dims_t &prb_vdims_t::dst_dims() const;

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t wei_dt() const { return dt[1]; }
    dnnl_data_type_t acc_dt() const {
        return is_integral_dt(wei_dt()) ? dnnl_s32 : dnnl_f32;
    }
    dnnl_data_type_t dst_dt() const { return dt[2]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    int64_t get_lda() const {
        if (!ld.empty() && ld[0] != 0) {
            assert(ld[0] >= batch_size * k);
            return ld[0];
        }
        return batch_size * k;
    }
    int64_t get_ldb() const {
        if (!ld.empty() && ld[1] != 0) {
            assert(ld[1] >= n);
            return ld[1];
        }
        // by default, use blocked weights w/ blocksize 16
        const int64_t ldb = rnd_up(n, 16);
        return ldb;
    }
    int64_t get_ldc() const {
        if (use_dst_as_acc()) return get_ldd();
        return n;
    }
    int64_t get_ldd() const {
        if (!ld.empty() && ld[2] != 0) {
            assert(ld[2] >= n);
            return ld[2];
        }
        return n;
    }

    int64_t get_src_batch_offset() const {
        return k * dnnl_data_type_size(src_dt());
    }
    int64_t get_wei_batch_offset() const {
        return get_ldb() * k * dnnl_data_type_size(wei_dt());
    }

    bool use_dst_as_acc() const {
        if (bia_dt == dnnl_data_type_undef && acc_dt() == dst_dt()
                && attr.is_def(/* skip_fmpath = */ true))
            return true;
        return false;
    }

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;

private:
    std::string set_repro_line() override;

    void check_block_size() const;
};

// TODO: not supported as of now.
struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
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
    const std::string *wtag() const override { return &wtag_; }
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

// See the description next to the definition.
int brgemm_finalize();

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int doit(const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace brgemm

#endif
