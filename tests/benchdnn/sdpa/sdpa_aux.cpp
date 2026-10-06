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

#include <string.h>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/stringstream.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

mask_type_t str2mask_type(const char *str) {
    if (!strcmp(str, "none")) return MASK_NONE;
    if (!strcmp(str, "buffer")) return MASK_BUFFER;
    if (!strcmp(str, "buffer_1d")) return MASK_BUFFER_1D;
    if (!strcmp(str, "buffer_2d")) return MASK_BUFFER_2D;
    if (!strcmp(str, "causal_top_left")) return MASK_CAUSAL_TOP_LEFT;
    if (!strcmp(str, "causal_bottom_right")) return MASK_CAUSAL_BOTTOM_RIGHT;
    assert(!"unknown mask type");
    return MASK_NONE;
}

const char *mask_type2str(mask_type_t mt) {
    switch (mt) {
        case MASK_NONE: return "none";
        case MASK_BUFFER: return "buffer";
        case MASK_BUFFER_1D: return "buffer_1d";
        case MASK_BUFFER_2D: return "buffer_2d";
        case MASK_CAUSAL_TOP_LEFT: return "causal_top_left";
        case MASK_CAUSAL_BOTTOM_RIGHT: return "causal_bottom_right";
        default: assert(!"unknown mask type"); return "unknown";
    }
}

scale_type_t str2scale_type(const char *str) {
    if (!strcmp(str, "library")) return SCALE_LIBRARY;
    if (!strcmp(str, "mul")) return SCALE_MUL;
    if (!strcmp(str, "div")) return SCALE_DIV;
    assert(!"unknown scale type");
    return SCALE_LIBRARY;
}

const char *scale_type2str(scale_type_t st) {
    switch (st) {
        case SCALE_LIBRARY: return "library";
        case SCALE_MUL: return "mul";
        case SCALE_DIV: return "div";
        default: assert(!"unknown scale type"); return "unknown";
    }
}

dnnl_data_type_t prb_t::get_dt(data_kind_t data_kind) const {
    switch (data_kind) {
        case SRC: return q_dt();
        case SRC_1: return k_dt();
        case SRC_2: return v_dt();
        case WEI: return k_dt(); // K and V share WEI kind
        case DST: return dst_dt();
        default: assert(!"unexpected"); return dnnl_data_type_undef;
    }
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> prb_t::get_md(int arg) const {
    // The same tags were already used to create descriptors in `init_pd`, so a
    // failure is not expected.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
    switch (arg) {
        case DNNL_ARG_QUERIES:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, q_dims().data(), q_dt(), qtag));
            return md;
        case DNNL_ARG_KEYS:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, k_dims().data(), k_dt(), ktag));
            return md;
        case DNNL_ARG_VALUES:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, v_dims().data(), v_dt(), vtag));
            return md;
        case DNNL_ARG_DST:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, dst_dims.data(), dst_dt(), dtag));
            return md;
        case DNNL_ARG_ATTN_MASK:
            if (with_mask()) {
                DNN_SAFE_V(dnn_mem_t::init_md(
                        md, ndims, msk_dims.data(), mdt, tag::abx));
                return md;
            }
            DNN_SAFE_V(dnn_mem_t::init_md(md));
            return md;
        case DNNL_ARG_DIFF_QUERIES:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, q_dims().data(), q_dt(), qtag));
            return md;
        case DNNL_ARG_DIFF_KEYS:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, k_dims().data(), k_dt(), ktag));
            return md;
        case DNNL_ARG_DIFF_VALUES:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, v_dims().data(), v_dt(), vtag));
            return md;
        case DNNL_ARG_DIFF_DST:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, dst_dims.data(), dst_dt(), dtag));
            return md;
        case DNNL_ARG_DS:
            DNN_SAFE_V(dnn_mem_t::init_md(
                    md, ndims, score_dims.data(), dnnl_f32, tag::abx));
            return md;
        default:
            assert(!"unsupported arg");
            return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    if (canonical || dir != def.dir[0]) s << "--dir=" << dir << " ";

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if (canonical || qtag != def.qtag[0]) s << "--qtag=" << qtag << " ";
    if (canonical || ktag != def.ktag[0]) s << "--ktag=" << ktag << " ";
    if (canonical || vtag != def.vtag[0]) s << "--vtag=" << vtag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || mask_type != def.mask_type[0])
        s << "--mask=" << mask_type2str(mask_type) << " ";
    if (canonical || mdt != def.mdt[0]) s << "--mdt=" << mdt << " ";
    if (canonical || scale_type != def.scale_type[0])
        s << "--scale=" << scale_type2str(scale_type) << " ";

    s << attr;
    if (canonical || ctx_init != def.ctx_init[0])
        s << "--ctx-init=" << ctx_init << " ";
    if (canonical || ctx_exe != def.ctx_exe[0])
        s << "--ctx-exe=" << ctx_exe << " ";
    if (canonical || !impl_filter.is_def() || !global_impl_filter.is_def())
        s << impl_filter;

    s << static_cast<const prb_vdims_t &>(*this);

    return s.str();
}

} // namespace sdpa
