/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "resampling/resampling.hpp"
#include "utils/stringstream.hpp"

namespace resampling {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(nearest);
    CASE(resampling_nearest);
    CASE(linear);
    CASE(resampling_linear);
#undef CASE
    assert(!"unknown algorithm");
    return undef;
}

const char *alg2str(alg_t alg) {
    if (alg == nearest) return "nearest";
    if (alg == linear) return "linear";
    assert(!"unknown algorithm");
    return "undef";
}

dnnl_alg_kind_t alg2alg_kind(alg_t alg) {
    if (alg == nearest) return dnnl_resampling_nearest;
    if (alg == linear) return dnnl_resampling_linear;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

int str2desc(desc_t *desc, const char *str) {
    /* canonical form:
     * mbXicXidXihXiwXodXohXowXnS
     *
     * where: Y = {fd, fi, bd}, X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      mb = 2, ih = oh = id = od = 1, S="wip"
     */

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
        CASE_N(id);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(od);
        CASE_N(oh);
        CASE_N(ow);
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

#define CHECK_BOTH_SET_VAL(val1_str, val1, val2_str, val2) \
    if (((val1) > 0 && (val2) <= 0) || ((val1) <= 0 && (val2) > 0)) { \
        assert((val1_str)[0] == 'd' && (val1_str)[1] == '.'); \
        assert((val2_str)[0] == 'd' && (val2_str)[1] == '.'); \
        const char *val1_str__ \
                = ((val1) > 0) ? &(val1_str)[2] : &(val2_str)[2]; \
        const char *val2_str__ \
                = ((val1) > 0) ? &(val2_str)[2] : &(val1_str)[2]; \
        BENCHDNN_PRINT(0, \
                "ERROR: setting `%s` was specified but paired setting `%s` " \
                "was not specified or set to 0. Full descriptor input: " \
                "`%s`.\n", \
                val1_str__, val2_str__, str); \
        return FAIL; \
    }

#define CHECK_BOTH_SET(val1, val2) CHECK_BOTH_SET_VAL(#val1, val1, #val2, val2)

    CHECK_BOTH_SET(d.id, d.od);
    CHECK_BOTH_SET(d.ih, d.oh);
    CHECK_BOTH_SET(d.iw, d.ow);

    if (sanitize_desc(d.ndims, {d.od, d.id}, {d.oh, d.ih}, {d.ow, d.iw}, {1, 1},
                str, true)
            != OK)
        return FAIL;

    *desc = d;

    return OK;
}

dims_t desc_t::src_dims() const {
    dims_t src_dims {mb, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        src_dims.erase(src_dims.begin() + 2);
    }

    return src_dims;
}

dims_t desc_t::dst_dims() const {
    dims_t dst_dims {mb, ic, od, oh, ow};
    for (int d = 0; d < 5 - ndims; ++d) {
        dst_dims.erase(dst_dims.begin() + 2);
    }

    return dst_dims;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims, {d.od, d.id}, {d.oh, d.ih},
            {d.ow, d.iw});

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    if (print_d) s << "od" << d.od;
    if (print_h) s << "oh" << d.oh;
    if (print_w) s << "ow" << d.ow;

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

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";
    if (canonical || sdt != def.sdt[0]) s << "--sdt=" << sdt << " ";
    if (canonical || ddt != def.ddt[0]) s << "--ddt=" << ddt << " ";
    if (canonical || tag != def.tag[0]) s << "--tag=" << tag << " ";
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

} // namespace resampling
