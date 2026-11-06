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

#include <assert.h>
#include <stdlib.h>

#include "bnorm/bnorm.hpp"
#include "utils/stringstream.hpp"

namespace bnorm {

check_alg_t str2check_alg(const char *str) {
    if (!strcasecmp("alg_0", str)) return ALG_0;
    if (!strcasecmp("alg_1", str)) return ALG_1;
    if (!strcasecmp("alg_2", str)) return ALG_2;
    return ALG_AUTO;
}

const char *check_alg2str(check_alg_t alg) {
    switch (alg) {
        case ALG_0: return "alg_0";
        case ALG_1: return "alg_1";
        case ALG_2: return "alg_2";
        case ALG_AUTO: return "alg_auto";
    }
    return "alg_auto";
}

flags_t str2flags(const char *str) {
    flags_t flags = NONE;
    // Iterate with a separate cursor so `str` still points at the whole value and the
    // error below can report what the user actually passed rather than its tail.
    for (const char *s = str; s && *s; s++) {
        if (*s == 'G') {
            flags |= GLOB_STATS;
        } else if (*s == 'C') {
            flags |= USE_SCALE;
        } else if (*s == 'H') {
            flags |= USE_SHIFT;
        } else if (*s == 'R') {
            flags |= FUSE_NORM_RELU;
        } else if (*s == 'A') {
            flags |= FUSE_NORM_ADD_RELU;
        } else {
            BENCHDNN_PRINT(0, "%s \'%s\'\n",
                    "Error: --flags option doesn't support value", str);
            SAFE_V(FAIL);
        }
    }
    return flags;
}

std::string flags2str(flags_t flags) {
    std::string str;
    if (flags & GLOB_STATS) str += "G";
    if (flags & USE_SCALE) str += "C";
    if (flags & USE_SHIFT) str += "H";
    if (flags & FUSE_NORM_RELU) str += "R";
    if (flags & FUSE_NORM_ADD_RELU) str += "A";
    if (flags & USE_RMS_NORM) str += "M";
    return str;
}

int str2desc(desc_t *desc, const char *str) {
    // Canonical form: mbXicXihXiwXidXepsYnS,
    // where
    //     X is integer
    //     Y is float
    //     S is string
    // note: symbol `_` is ignored.
    // Cubic/square shapes are supported by specifying just highest dimension.

    desc_t d {0};
    d.mb = 2;
    d.eps = 1.f / 16;

    const char *s = str;
    assert(s);

    auto mstrtoll = [](const char *nptr, char **endptr) {
        return strtoll(nptr, endptr, 10);
    };

#define CASE_NN(prb, c, cvfunc) \
    do { \
        if (!strncmp(prb, s, strlen(prb))) { \
            ok = 1; \
            s += strlen(prb); \
            char *end_s; \
            d.c = cvfunc(s, &end_s); \
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
#define CASE_N(c, cvfunc) CASE_NN(#c, c, cvfunc)
    while (*s) {
        int ok = 0;
        CASE_N(mb, mstrtoll);
        CASE_N(ic, mstrtoll);
        CASE_N(id, mstrtoll);
        CASE_N(ih, mstrtoll);
        CASE_N(iw, mstrtoll);
        CASE_N(eps, strtof);
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

    if (sanitize_desc(d.ndims, {d.id}, {d.ih}, {d.iw}, {1}, str) != OK)
        return FAIL;

    *desc = d;

    return OK;
}

dims_t desc_t::data_dims() const {
    dims_t data_dims {mb, ic, id, ih, iw};
    for (int d = 0; d < 5 - ndims; ++d) {
        data_dims.erase(data_dims.begin() + 2);
    }

    return data_dims;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims, {d.id}, {d.ih}, {d.iw});

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    if (canonical || d.eps != 1.f / 16) s << "eps" << d.eps;

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
    if (canonical || dt != def.dt[0]) s << "--dt=" << dt << " ";
    if (canonical || tag != def.tag[0]) s << "--tag=" << tag << " ";
    if (canonical || strides != def.strides[0])
        s << "--strides=" << vdims2str(strides) << " ";
    if (canonical || flags != def.flags[0])
        s << "--flags=" << flags2str(flags) << " ";
    if (canonical || check_alg != def.check_alg)
        s << "--check-alg=" << check_alg2str(check_alg) << " ";
    if (canonical || inplace != def.inplace[0])
        s << "--inplace=" << bool2str(inplace) << " ";
    if (canonical || debug_check_ws != def.debug_check_ws)
        s << "--debug-check-ws=" << bool2str(debug_check_ws) << " ";

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

} // namespace bnorm
