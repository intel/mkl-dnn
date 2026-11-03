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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/stringstream.hpp"

#include "brgemm/brgemm.hpp"

namespace brgemm {

dnnl_data_type_t prb_t::get_dt(data_kind_t data_kind) const {
    switch (data_kind) {
        case SRC: return src_dt();
        case WEI: return wei_dt();
        case BIA: return bia_dt;
        case DST: return dst_dt();
        default: assert(!"unexpected"); return dnnl_data_type_undef;
    }
}

void prb_t::check_block_size() const {
    // Note: batch_size is incorporated into K dimension.
    // That's why each source batch has an offset of `k`.
    // Weights have more complicated case. Weights are in double-blocked format,
    // which becomes triple-blocked for bf16 and int8 to become VNNI-friendly.
    // Because of this and batch_size incorporation, offsets below DO NOT work
    // with K not divisible by K block size and batch_size > 1.
    // The problem is it can't be handled properly when batch size is fused,
    // but this allows enable s8s8 and zero-points compensation cases easier.
    int block_size = 0;
    switch (wei_dt()) {
        case dnnl_f32: block_size = 16; break;
        case dnnl_f16: block_size = 16; break;
        case dnnl_bf16: block_size = 32; break;
        case dnnl_u8:
        case dnnl_f8_e5m2:
        case dnnl_f8_e4m3:
        case dnnl_s8: block_size = 64; break;
        default: break;
    }
    (void)block_size;
    assert(block_size > 1);
    assert(IMPLICATION(batch_size > 1, k % block_size == 0));
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    auto is_xf16 = [](dnnl_data_type_t dt) {
        return dt == dnnl_bf16 || dt == dnnl_f16;
    };
    if (!IMPLICATION(is_xf16(prb->bia_dt) || is_xf16(prb->dst_dt()),
                is_xf16(prb->wei_dt()))) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->wei_dt(), prb->bia_dt, prb->dst_dt()},
            prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_gemm, prb->src_dt(), prb->dst_dt());
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_gemm);

    // Unconditionally skip remaining unimplemented cases.
    // TODO: stop doing it.
    BENCHDNN_PRINT(
            2, "%s\n", "The kernel return unimplemented by some reason.");
    res->state = SKIPPED;
    res->reason = reason_t::skip_not_supported;
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    if (!prb->attr.zero_points.is_def(DNNL_ARG_WEIGHTS)) {
        // TODO: weights zero point is not supported yet.
        // It requires enabling f32 -> u8 reorder with compensation on the
        // library side. When enabled, it produces incorrect results for cases
        // with K=1. Likely there's a bug inside. Postpone supporting it.
        BENCHDNN_PRINT(2, "%s\n",
                "Reorder with compensation is not supported for u8 destination "
                "data type");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    if (prb->wtag != tag::abx) {
        BENCHDNN_PRINT(
                2, "%s\n", "`wtag` option is supported for ukernel API only.");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    if (!prb->strides[STRIDES_WEI].empty()) {
        BENCHDNN_PRINT(2, "%s\n",
                "`strides` option for weights is supported for ukernel API "
                "only.");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
#else
    if (!prb->attr.is_def()) {
        bool non_def_zps = !prb->attr.zero_points.is_def();
        bool non_def_fpmath = !prb->attr.fpmath_mode.is_def();
        if (non_def_zps || non_def_fpmath) {
            BENCHDNN_PRINT(2, "%s\n",
                    "Non-default scales/zero-points/fpmath attributes are not "
                    "supported");
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }

        bool non_def_po = !prb->attr.post_ops.is_def();
        if (non_def_po) {
            const auto &po = prb->attr.post_ops;
            bool has_sum = po.find(attr_t::post_ops_t::kind_t::SUM) != -1;
            if (has_sum) {
                BENCHDNN_PRINT(2, "%s\n", "Sum post-op is not supported");
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return;
            }
        }
    }

    const bool ldb_ok = prb->get_ldb() == 16 || prb->get_ldb() == 32
            || prb->get_ldb() == 48 || prb->get_ldb() == 64;
    if (!ldb_ok) {
        BENCHDNN_PRINT(2,
                "Unsupported leading B dimension. Only 16, 32, 48, and 64 are "
                "supported. Actual value is \'%zu\'.\n",
                (size_t)prb->get_ldb());
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    if (prb->bia_dt != dnnl_data_type_undef) {
        BENCHDNN_PRINT(2, "%s\n", "Bias is not supported");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    if (prb->src_dt() == dnnl_s8 && prb->wei_dt() == dnnl_s8) {
        // Pre-AMX ISAs require s8s8 compensation buffer passed. The internals
        // should check if it was supplied and don't blow up if it wasn't
        // provided.
        BENCHDNN_PRINT(2, "%s\n", "s8s8 support is temporary disabled");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    if (prb->alpha != 1.f) {
        BENCHDNN_PRINT(2, "%s\n", "Alpha is purposely not supported");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
#endif
}

std::string prb_t::set_repro_line() {
    stringstream_t s;
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || wtag != def.wtag[0]) s << "--wtag=" << wtag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || strides != def.strides[0])
        s << "--strides=" << vdims2str(strides) << " ";
    if (canonical || ld != def.ld[0]) {
        s << "--ld=";
        if (ld[0] != 0) s << ld[0];
        s << ":";
        if (ld[1] != 0) s << ld[1];
        s << ":";
        if (ld[2] != 0) s << ld[2];
        s << " ";
    }

    if (canonical || bia_dt != def.bia_dt[0]) s << "--bia_dt=" << bia_dt << " ";

    if (canonical || alpha != def.alpha[0]) s << "--alpha=" << alpha << " ";
    if (canonical || beta != def.beta[0]) s << "--beta=" << beta << " ";
    if (canonical || batch_size != def.batch_size[0])
        s << "--bs=" << batch_size << " ";
    if (canonical || brgemm_attr != def.brgemm_attr[0])
        s << "--brgemm-attr=" << brgemm_attr << " ";
    if (canonical || batch_kind != def.batch_kind[0])
        s << "--batch-kind=" << batch_kind << " ";

    s << attr;
    s << static_cast<const prb_vdims_t &>(*this);

    return s.str();
}

} // namespace brgemm
