/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/stringstream.hpp"

#include "deconv/deconv.hpp"

namespace deconv {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(DIRECT);
    CASE(deconvolution_direct);
    CASE(WINO);
    CASE(deconvolution_wino);
#undef CASE
    assert(!"unknown algorithm");
    return UNDEF;
}

const char *alg2str(alg_t alg) {
    if (alg == DIRECT) return "direct";
    if (alg == WINO) return "wino";
    assert(!"unknown algorithm");
    return "undef";
}

alg_t alg_kind2alg(dnnl_alg_kind_t alg) {
    if (alg == dnnl_deconvolution_direct) return DIRECT;
    if (alg == dnnl_deconvolution_winograd) return WINO;
    assert(!"unknown algorithm");
    return DIRECT;
}

int str2desc(desc_t *desc, const char *str) {
    /* canonical form:
     * gXmbX_icXidXihXiwX_ocXodXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_ddXdhXdwXnS
     *
     * where X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - if smaller dimensions are not specified => square or cubic form;
     *  - if output is undefined => compute output;
     *  - if padding is undefined => compute trivial padding;
     */

    desc_t d {0};
    d.g = -1;
    d.has_groups = false;
    d.mb = 2;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

    const char *s = str;
    assert(s);

#define CASE_NN(prb, c) \
    do { \
        if (!strncmp(prb, s, strlen(prb))) { \
            ok = 1; \
            s += strlen(prb); \
            char *end_s; \
            d.c = strtoll(s, &end_s, 10); \
            if (end_s == s) { \
                BENCHDNN_PRINT(0, \
                        "ERROR: No value found for `%s` setting. Full " \
                        "descriptor input: `%s`.\n", \
                        prb, str); \
                return FAIL; \
            } \
            s += (end_s - s); \
            if (d.c < 0) { \
                BENCHDNN_PRINT(0, \
                        "ERROR: `%s` must be positive. Full descriptor " \
                        "input: `%s`.\n", \
                        prb, str); \
                return FAIL; \
            } \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(g);
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(id);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(oc);
        CASE_N(od);
        CASE_N(oh);
        CASE_N(ow);
        CASE_N(kd);
        CASE_N(kh);
        CASE_N(kw);
        CASE_N(sd);
        CASE_N(sh);
        CASE_N(sw);
        CASE_N(pd);
        CASE_N(ph);
        CASE_N(pw);
        CASE_N(dd);
        CASE_N(dh);
        CASE_N(dw);
        if (*s == 'n') {
            d.name = s + 1;
            break;
        }
        if (*s == '_') ++s;
        if (!ok) {
            BENCHDNN_PRINT(0,
                    "ERROR: Unrecognized pattern in `%s` descriptor starting "
                    "from `%s` entry.\n",
                    str, s);
            return FAIL;
        }
    }
#undef CASE_NN
#undef CASE_N

#define CHECK_SET_OR_ZERO_VAL(val_str, val) \
    if ((val) <= 0) { \
        assert((val_str)[0] == 'd' && (val_str)[1] == '.'); \
        const char *val_str__ = &(val_str)[2]; \
        BENCHDNN_PRINT(0, \
                "ERROR: setting `%s` was not specified or set to 0. Full " \
                "descriptor input: `%s`.\n", \
                val_str__, str); \
        return FAIL; \
    }

#define CHECK_SET_OR_ZERO(val) CHECK_SET_OR_ZERO_VAL(#val, val)

    // Check any number of groups, including one, works correctly.
    if (d.g >= 0) {
        CHECK_SET_OR_ZERO(d.g);
        d.has_groups = true;
    } else {
        // Rest of driver pieces rely on the logic that `g` is positive.
        d.g = 1;
    }

    CHECK_SET_OR_ZERO(d.ic);
    CHECK_SET_OR_ZERO(d.oc);
    CHECK_SET_OR_ZERO(d.sd);
    CHECK_SET_OR_ZERO(d.sh);
    CHECK_SET_OR_ZERO(d.sw);

#define CHECK_DEDUCED_ZERO_VAL(val_str, val) \
    if ((val) <= 0) { \
        assert((val_str)[0] == 'd' && (val_str)[1] == '.'); \
        const char *val_str__ = &(val_str)[2]; \
        BENCHDNN_PRINT(0, \
                "ERROR: `%s` was not specified but rest provided dimensions " \
                "result in negative or zero value. Full descriptor input: " \
                "`%s`.\n", \
                val_str__, str); \
        return FAIL; \
    }

#define CHECK_DEDUCED_ZERO(val) CHECK_DEDUCED_ZERO_VAL(#val, val)

    auto compute_out
            = [](int64_t i, int64_t k, int64_t s, int64_t p, int64_t d) {
        return (i - 1) * s + (k - 1) * (d + 1) - 2 * p + 1;
    };
    auto compute_pad
            = [](int64_t o, int64_t i, int64_t k, int64_t s, int64_t d) {
        return ((i - 1) * s - o + ((k - 1) * (d + 1) + 1)) / 2;
    };

    const bool no_d = (d.id | d.kd | d.od | d.dd) == 0 && d.sd == 1 && d.pd < 1;
    const bool no_h = (d.ih | d.kh | d.oh | d.dh) == 0 && d.sh == 1 && d.ph < 1;
    const bool no_w = (d.iw | d.kw | d.ow | d.dw) == 0 && d.sw == 1 && d.pw < 1;

    if (!no_d) {
        CHECK_SET_OR_ZERO(d.id);
        CHECK_SET_OR_ZERO(d.kd);
        if (!d.od) {
            if (d.pd < 0) d.pd = 0;
            d.od = compute_out(d.id, d.kd, d.sd, d.pd, d.dd);
            CHECK_DEDUCED_ZERO(d.od);
        } else if (d.pd < 0)
            d.pd = compute_pad(d.od, d.id, d.kd, d.sd, d.dd);
    }

    if (!no_h) {
        CHECK_SET_OR_ZERO(d.ih);
        CHECK_SET_OR_ZERO(d.kh);
        if (!d.oh) {
            if (d.ph < 0) d.ph = 0;
            d.oh = compute_out(d.ih, d.kh, d.sh, d.ph, d.dh);
            CHECK_DEDUCED_ZERO(d.oh);
        } else if (d.ph < 0)
            d.ph = compute_pad(d.oh, d.ih, d.kh, d.sh, d.dh);
    }

    if (!no_w) {
        CHECK_SET_OR_ZERO(d.iw);
        CHECK_SET_OR_ZERO(d.kw);
        if (!d.ow) {
            if (d.pw < 0) d.pw = 0;
            d.ow = compute_out(d.iw, d.kw, d.sw, d.pw, d.dw);
            CHECK_DEDUCED_ZERO(d.ow);
        } else if (d.pw < 0)
            d.pw = compute_pad(d.ow, d.iw, d.kw, d.sw, d.dw);
    }

    if (sanitize_desc(d.ndims, {d.od, d.id, d.kd, d.sd, d.pd, d.dd},
                {d.oh, d.ih, d.kh, d.sh, d.ph, d.dh},
                {d.ow, d.iw, d.kw, d.sw, d.pw, d.dw}, {1, 1, 1, 1, 0, 0}, str,
                true)
            != OK)
        return FAIL;

    d.init_pad_r();
    *desc = d;

    return OK;
}

dnnl_data_type_t prb_t::get_dt(data_kind_t data_kind) const {
    switch (data_kind) {
        case SRC: return src_dt();
        case WEI: return wei_dt();
        case BIA: return bia_dt();
        case DST: return dst_dt();
        default: assert(!"unexpected data_kind"); return dnnl_data_type_undef;
    }
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims,
            {d.od, d.id, d.kd, d.sd, d.pd, d.dd},
            {d.oh, d.ih, d.kh, d.sh, d.ph, d.dh},
            {d.ow, d.iw, d.kw, d.sw, d.pw, d.dw});

    auto print_spatial
            = [&](const char *d_str, int64_t d_val, const char *h_str,
                      int64_t h_val, const char *w_str, int64_t w_val) {
        if (print_d) s << d_str << d_val;
        if (print_h) s << h_str << h_val;
        if (print_w) s << w_str << w_val;
    };

    if (canonical || d.has_groups) s << "g" << d.g;
    if (canonical || d.mb != 2) s << "mb" << d.mb;
    s << "ic" << d.ic;
    print_spatial("id", d.id, "ih", d.ih, "iw", d.iw);
    s << "oc" << d.oc;
    print_spatial("od", d.od, "oh", d.oh, "ow", d.ow);
    print_spatial("kd", d.kd, "kh", d.kh, "kw", d.kw);

    if (canonical || d.sh != 1 || d.sw != 1 || d.sd != 1)
        print_spatial("sd", d.sd, "sh", d.sh, "sw", d.sw);

    print_spatial("pd", d.pd, "ph", d.ph, "pw", d.pw);

    if (canonical || d.dh != 0 || d.dw != 0 || d.dd != 0)
        print_spatial("dd", d.dd, "dh", d.dh, "dw", d.dw);

    if (!d.name.empty()) s << "n" << d.name;

    return s;
}

dims_t desc_t::src_dims() const {
    dims_t src_dims {mb, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        src_dims.erase(src_dims.begin() + 2);
    }

    return src_dims;
}

dims_t desc_t::wei_dims() const {
    dims_t wei_dims {g, oc / g, ic / g, kd, kh, kw};
    if (!has_groups) { wei_dims.erase(wei_dims.begin()); }
    for (int d = 0; d < 5 - ndims; ++d) {
        wei_dims.erase(wei_dims.begin() + 2 + has_groups);
    }

    return wei_dims;
}

dims_t desc_t::bia_dims() const {
    dims_t bia_dims {oc};
    return bia_dims;
}

dims_t desc_t::dst_dims() const {
    dims_t dst_dims {mb, oc, od, oh, ow};
    for (int d = 0; d < 5 - ndims; ++d) {
        dst_dims.erase(dst_dims.begin() + 2);
    }

    return dst_dims;
}

dims_t desc_t::strides() const {
    dims_t strides {sd, sh, sw};
    return dims_t(strides.begin() + (5 - ndims), strides.end());
}

dims_t desc_t::dilations() const {
    dims_t dilations {dd, dh, dw};
    return dims_t(dilations.begin() + (5 - ndims), dilations.end());
}

dims_t desc_t::padding() const {
    dims_t padding {pd, ph, pw};
    return dims_t(padding.begin() + (5 - ndims), padding.end());
}

dims_t desc_t::padding_r() const {
    dims_t padding_r {pd_r, ph_r, pw_r};
    return dims_t(padding_r.begin() + (5 - ndims), padding_r.end());
}

int64_t desc_t::desc_nelems(int arg, int mask) const {
    std::vector<int64_t> src {mb, ic, id, ih, iw};
    std::vector<int64_t> wei {g, oc, ic, kd, kh, kw};
    std::vector<int64_t> dst {mb, oc, od, oh, ow};
    std::vector<int64_t> dummy;

    if (!has_groups) wei.erase(wei.begin());

    for (int d = 0; d < 5 - ndims; d++) {
        src.erase(src.begin() + 2);
        wei.erase(wei.begin() + 2);
        dst.erase(dst.begin() + 2);
    }

    std::vector<int64_t> &dims = dummy;
    switch (arg) {
        case DNNL_ARG_SRC: dims = src; break;
        case DNNL_ARG_WEIGHTS: dims = wei; break;
        case DNNL_ARG_DST: dims = dst; break;
        default: assert(!"unsupported arg");
    }

    int64_t nelems = 1;
    for (int d = 0; d < ndims; d++) {
        nelems *= (mask & (1 << d)) ? dims[d] : 1;
    }
    return nelems;
}

void prb_t::count_ops() {
    if (ops > 0) return;

    int64_t od_t = this->id;
    int64_t oh_t = this->ih;
    int64_t ow_t = this->iw;
    int64_t id_t = this->od;
    int64_t ih_t = this->oh;
    int64_t iw_t = this->ow;
    double sp_ops = 0;
    for_(int64_t od = 0; od < od_t; ++od)
    for_(int64_t oh = 0; oh < oh_t; ++oh)
    for (int64_t ow = 0; ow < ow_t; ++ow) {
        for (int64_t kd = 0; kd < this->kd; ++kd) {
            const int64_t id = od * this->sd - this->pd + kd * (this->dd + 1);
            if (id < 0 || id >= id_t) continue;
            for (int64_t kh = 0; kh < this->kh; ++kh) {
                const int64_t ih
                        = oh * this->sh - this->ph + kh * (this->dh + 1);
                if (ih < 0 || ih >= ih_t) continue;
                for (int64_t kw = 0; kw < this->kw; ++kw) {
                    const int64_t iw
                            = ow * this->sw - this->pw + kw * (this->dw + 1);
                    if (iw < 0 || iw >= iw_t) continue;
                    sp_ops += 1;
                }
            }
        }
    }

    ops = 2 * this->mb * this->oc * this->ic / this->g * sp_ops;
}

std::unique_ptr<base_perf_report_t> prb_t::create_perf_report(
        const char *perf_template) const {
    return std::unique_ptr<base_perf_report_t>(
            new perf_report_t(this, perf_template));
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if ((canonical || bia_dt_ != def.bia_dt[0]) && !(dir & FLAG_BIA))
        s << "--bia-dt=" << bia_dt_ << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || wtag != def.wtag[0]) s << "--wtag=" << wtag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || alg != def.alg[0]) s << "--alg=" << alg2str(alg) << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<const desc_t &>(*this);

    return s.str();
}

} // namespace deconv
