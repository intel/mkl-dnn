/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GNORM_HPP
#define GNORM_HPP

#include <stdint.h>
#include <string>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/perf_report.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

#include "bnorm/bnorm.hpp"

namespace gnorm {

using check_alg_t = bnorm::check_alg_t;
using flags_t = bnorm::flags_t;
const flags_t NONE = bnorm::NONE;
const flags_t GLOB_STATS = bnorm::GLOB_STATS;
const flags_t USE_SCALE = bnorm::USE_SCALE;
const flags_t USE_SHIFT = bnorm::USE_SHIFT;
const auto flags2str = bnorm::flags2str;
flags_t str2flags(const char *str);

struct desc_t {
    int64_t g, mb, ic, id, ih, iw;
    float eps;
    std::string name;
    int ndims;

    dims_t data_dims() const;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    desc_t desc {};

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::vector<std::string>> tag {{tag::abx, tag::any}};
    std::vector<flags_t> flags {NONE};
    check_alg_t check_alg = bnorm::ALG_AUTO;

    const char *perf_template_csv() const {
        static const std::string args = "%dir%,%dt%,%tag%,%flags%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dir.size() == 1 && dt.size() == 1 && tag.size() == 1
                && flags.size() == 1 && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc, s.dir[0], s.dt[0], s.tag[0], s.flags[0], s.check_alg,
                  s.mb[0], s.inplace[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const desc_t &desc, dir_t dir,
            const std::vector<dnnl_data_type_t> &dt,
            const std::vector<std::string> &tag, flags_t flags,
            check_alg_t check_alg, int64_t mb, bool inplace, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : desc_t(desc)
        , base_prb_t(dir, inplace, attr, ctx_init, ctx_exe, impl_filter)
        , dt(dt)
        , tag(tag)
        , flags(flags)
        , check_alg(check_alg)
        , user_mb(mb) {

        if (mb) this->mb = mb;

        broadcast_vector(this->dt, dt[0], 2);
        broadcast_vector(this->tag, tag[0], 2);

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    std::string stat_tag;
    std::vector<dnnl_data_type_t> dt;
    std::vector<std::string> tag;
    flags_t flags;
    check_alg_t check_alg;
    int64_t user_mb;
    bool use_stats() const { return flags & GLOB_STATS; }
    bool use_sc() const { return flags & USE_SCALE; }
    bool use_sh() const { return flags & USE_SHIFT; }

    int64_t get_c_start(int64_t _g) const { return _g * this->ic / this->g; }

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

struct cfg_t {
    // The idea of data filling is to choose source values the way both mean
    // and variance values are computed exactly. Exactness must hold for any
    // order of the computations. It is achieved when the following equation
    // holds: src[i] + src[i + 1] = 2 * mean.
    //
    // The data variation in source values is allowed in the last `flex_bits_`
    // bits. If the sequence `L_` is too big, e.g.,
    // `flex_bits_ <= min_flex_bits_`, the mean value is set to `0.f` and source
    // is partially filled with zeros according to `density_`. In such case at
    // least `want_flex_bits_` is reserved for source values variation.
    // Once source values are set, the variance value is computed.
    //
    // ALG_0: mean value is set to 0.
    // ALG_1: mean value is set to 2^x, where `x` \in {-2, -1, ..., 4}.
    // ALG_2: same as ALG_1 for mean and some more variation in src.
    // ALG_AUTO: choose between algorithms automatically.
    //
    // `density_` for ALG_0 is filled according to the following inequation:
    //     (exact_bits - log_2(L * density)) / 2 >= flex_bits
    // For ALG_2 use ~100k non-zero elements. With more elements, rounding issues
    // are popping up for large spatial due to very long accumulation chains
    // and potential different order of accumulation.
    cfg_t(const prb_t *prb)
        : exact_bits_(digits_dt(prb->dt[0]))
        , L_(prb->ic / prb->g * prb->id * prb->ih * prb->iw)
        , logL_(static_cast<int64_t>(
                  std::ceil(std::log2(static_cast<float>(L_)))))
        , free_bits_((exact_bits_ - logL_) / 2 - 1)
        , want_flex_bits_(MIN2(6, exact_bits_ / 2))
        , check_alg_(prb->check_alg == bnorm::ALG_AUTO
                          ? (free_bits_ >= min_flex_bits_
                                            ? bnorm::ALG_1
                                            : (want_flex_bits_ == exact_bits_ / 2
                                                              ? bnorm::ALG_2
                                                              : bnorm::ALG_0))
                          : prb->check_alg)
        , flex_bits_(check_alg_ == bnorm::ALG_1 ? MIN2(exact_bits_, free_bits_)
                                                : want_flex_bits_)
        , flex_mask_((1LL << flex_bits_) - 1)
        , density_(check_alg_ == bnorm::ALG_0
                          ? 1.f * (1LL << (exact_bits_ - 2 * flex_bits_)) / L_
                          : check_alg_ == bnorm::ALG_2
                          ? MIN2(100000.f / L_, 1.f)
                          : 1.f) {
        assert(logL_ <= 0 || (1LL << (logL_ - 1)) < L_);
        assert(L_ <= (1LL << logL_));
        assert(flex_bits_ >= min_flex_bits_);
        BENCHDNN_PRINT(6,
                "[CFG]: check_alg:%s; L:%zu; density:%g; flex_bits:" IFMT "\n",
                check_alg2str(check_alg_), (size_t)L_, density_, flex_bits_);
    }

    int64_t exact_bits_;
    int64_t L_;
    int64_t logL_;
    int64_t free_bits_; // Helper value holder.
    int64_t want_flex_bits_;
    check_alg_t check_alg_;
    int64_t flex_bits_;
    int64_t flex_mask_;
    float density_;

    static constexpr int64_t min_flex_bits_ = 3;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , stag_(normalize_tags(p_->tag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_flags(std::ostream &s) const override {
        s << flags2str(p_->flags);
    }

    const attr_t *attr() const override { return &p_->attr; }
    const std::string *name() const override { return &p_->name; }
    const dir_t *dir() const override { return &p_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
};

/* some extra control parameters which shouldn't be placed in prb_t */

inline size_t data_off(const prb_t *prb, int64_t mb, int64_t c, int64_t d,
        int64_t h, int64_t w) {
    return (((mb * prb->ic + c) * prb->id + d) * prb->ih + h) * prb->iw + w;
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

} // namespace gnorm

#endif
