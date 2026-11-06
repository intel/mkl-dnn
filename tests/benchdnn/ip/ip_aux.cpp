/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/stringstream.hpp"

#include "ip/ip.hpp"

namespace ip {

int str2desc(desc_t *desc, const char *str) {
    // Canonical form: mbXicXidXihXiwXocXnS,
    // where
    //     X is integer
    //     S is string
    // note: symbol `_` is ignored.
    // Cubic/square shapes are supported by specifying just highest dimension.

    desc_t d {0};
    d.mb = 2;

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
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(id);
        CASE_N(oc);
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

    CHECK_SET_OR_ZERO(d.ic);
    CHECK_SET_OR_ZERO(d.oc);

    if (sanitize_desc(d.ndims, {d.id}, {d.ih}, {d.iw}, {1}, str) != OK)
        return FAIL;

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
    print_dhw(print_d, print_h, print_w, d.ndims, {d.id}, {d.ih}, {d.iw});

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    s << "oc" << d.oc;

    if (!d.name.empty()) s << "n" << d.name;

    return s;
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

dims_t desc_t::src_dims() const {
    dims_t src_dims {mb, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        src_dims.erase(src_dims.begin() + 2);
    }

    return src_dims;
}

dims_t desc_t::wei_dims() const {
    dims_t wei_dims {oc, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        wei_dims.erase(wei_dims.begin() + 2);
    }

    return wei_dims;
}

dims_t desc_t::bia_dims() const {
    dims_t bia_dims {oc};
    return bia_dims;
}

dims_t desc_t::dst_dims() const {
    dims_t dst_dims {mb, oc};
    return dst_dims;
}

int64_t desc_t::desc_nelems(int arg, int mask) const {
    dims_t dims;
    switch (arg) {
        case DNNL_ARG_SRC: dims = src_dims(); break;
        case DNNL_ARG_WEIGHTS: dims = wei_dims(); break;
        case DNNL_ARG_DST: dims = dst_dims(); break;
        default: assert(!"unsupported arg");
    }

    int64_t nelems = 1;
    for (int d = 0; d < ndims; d++) {
        nelems *= (mask & (1 << d)) ? dims[d] : 1;
    }
    return nelems;
}

} // namespace ip
