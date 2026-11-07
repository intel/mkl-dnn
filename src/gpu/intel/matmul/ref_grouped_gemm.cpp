/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/matmul/ref_grouped_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <algorithm>

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

namespace {

// Strides of a [G, rows, cols] view of a tensor: a 2D tensor has no group dim,
// and dims of size 1 get a zero stride, e.g., the ones out of a quantization
// mask
void get_strides(dims_t strides, const memory_desc_wrapper &mdw) {
    for (int d = 0; d < mdw.ndims(); d++)
        strides[3 - mdw.ndims() + d] = mdw.dims()[d] > 1 ? mdw.strides()[d] : 0;
}

// Strides of the dense row-major scales or zero points of an argument
status_t get_quant_strides(dims_t strides, const quant_entry_t &entry,
        const memory_desc_wrapper &mdw) {
    memory_desc_t quant_md;
    CHECK(entry.get_md(quant_md, *mdw.md_));
    get_strides(strides, memory_desc_wrapper(quant_md));
    return status::success;
}

} // namespace

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
status_t ref_grouped_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper bia_d(pd()->weights_md(1));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const bool is_2dby2d = pd()->is_2dby2d();

    const dim_t group_count = pd()->group_count_;
    const dim_t N = pd()->N();
    const dim_t total_tokens = src_d.dims()[0];

    // Note, that below are constants for specific patterns
    const dim_t M_fixed = is_2dby2d ? src_d.dims()[0] : 0; // 2Dx2D
    // K for 2Dx3D or total_K for 2Dx2D, the extent the K-groups divide
    const dim_t K_total = pd()->K();

    dims_t src_strides {}, wei_strides {}, bia_strides {}, dst_strides {};
    get_strides(src_strides, src_d);
    get_strides(wei_strides, wei_d);
    get_strides(bia_strides, bia_d);
    get_strides(dst_strides, dst_d);

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const dim_t src_scale_group_k = attr_scales.get_group(DNNL_ARG_SRC, -1);
    const dim_t wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
    dims_t src_scale_strides {}, wei_scale_strides {};
    CHECK(get_quant_strides(
            src_scale_strides, attr_scales.get(DNNL_ARG_SRC), src_d));
    CHECK(get_quant_strides(
            wei_scale_strides, attr_scales.get(DNNL_ARG_WEIGHTS), wei_d));

    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_src_zp = !attr_zps.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_zp = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    const dim_t src_zp_group_k = attr_zps.get_group(DNNL_ARG_SRC, -1);
    const dim_t wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
    dims_t src_zp_strides {}, wei_zp_strides {};
    CHECK(get_quant_strides(src_zp_strides, attr_zps.get(DNNL_ARG_SRC), src_d));
    CHECK(get_quant_strides(
            wei_zp_strides, attr_zps.get(DNNL_ARG_WEIGHTS), wei_d));

    // Finest K-group of scales and zero points, K when there are none
    dim_t k_block = K_total;
    for (const dim_t group_k : {src_scale_group_k, wei_scale_group_k,
                 src_zp_group_k, wei_zp_group_k})
        if (group_k > 1) k_block = math::gcd(k_block, group_k);

    // Runtime group offsets are checked in the kernel against the src total
    // size along the variable dim, and 2Dx2D groups must align with the
    // coarsest scales K-group
    const dim_t src_total
            = src_d.dims()[src_d.sparse_desc().grouped_desc.variable_dim_idx];
    const dim_t k_align
            = is_2dby2d ? std::max(src_scale_group_k, wei_scale_group_k) : 1;

    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);

    compute::kernel_arg_list_t arg_list;
    int arg_idx = 0;
    arg_list.set(arg_idx++, src_data);
    arg_list.set(arg_idx++, src_offsets);
    arg_list.set(arg_idx++, wei_data);
    // Tensor sharing src split: wei (2Dx2D) or dst (2Dx3D)
    if (is_2dby2d) {
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 1));
    } else {
        arg_list.set(arg_idx++, CTX_OUT_STORAGE(DNNL_ARG_DST, 1));
    }
    arg_list.set(arg_idx++, dst_data);
    arg_list.set(arg_idx++, (int)group_count);
    arg_list.set(arg_idx++, (int)is_2dby2d);
    arg_list.set(arg_idx++, M_fixed);
    arg_list.set(arg_idx++, K_total);
    arg_list.set(arg_idx++, N);
    arg_list.set(arg_idx++, k_block);
    arg_list.set(arg_idx++, src_total);
    arg_list.set(arg_idx++, k_align);
    arg_list.set(arg_idx++, src_strides[1]);
    arg_list.set(arg_idx++, src_strides[2]);
    arg_list.set(arg_idx++, wei_strides[0]);
    arg_list.set(arg_idx++, wei_strides[1]);
    arg_list.set(arg_idx++, wei_strides[2]);
    arg_list.set(arg_idx++, dst_strides[0]);
    arg_list.set(arg_idx++, dst_strides[1]);
    arg_list.set(arg_idx++, dst_strides[2]);
    if (pd()->with_bias()) {
        // Bias is [G, N]
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_BIAS));
        arg_list.set(arg_idx++, bia_strides[1]);
        arg_list.set(arg_idx++, bia_strides[2]);
    }
    if (with_src_scales) {
        arg_list.set(
                arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC));
        arg_list.set(arg_idx++, src_scale_strides[1]);
        arg_list.set(arg_idx++, src_scale_strides[2]);
        arg_list.set(arg_idx++, src_scale_group_k);
    }
    if (with_wei_scales) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS));
        arg_list.set(arg_idx++, wei_scale_strides[0]);
        arg_list.set(arg_idx++, wei_scale_strides[1]);
        arg_list.set(arg_idx++, wei_scale_strides[2]);
        arg_list.set(arg_idx++, wei_scale_group_k);
        arg_list.set(arg_idx++, attr_scales.get_group(DNNL_ARG_WEIGHTS, -1));
    }
    if (with_src_zp) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC));
        arg_list.set(arg_idx++, src_zp_strides[1]);
        arg_list.set(arg_idx++, src_zp_strides[2]);
        arg_list.set(arg_idx++, src_zp_group_k);
    }
    if (with_wei_zp) {
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS));
        arg_list.set(arg_idx++, wei_zp_strides[0]);
        arg_list.set(arg_idx++, wei_zp_strides[1]);
        arg_list.set(arg_idx++, wei_zp_strides[2]);
        arg_list.set(arg_idx++, wei_zp_group_k);
        arg_list.set(arg_idx++, attr_zps.get_group(DNNL_ARG_WEIGHTS, -1));
    }

    // Post-ops apply to the 2Dx3D pattern only (grouped dst)
    arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, arg_idx, pd()->generic_po_, pd()->group_po_dst_md_);

    // Simple 3D dispatch for ref impl clarity
    compute::range_t gws
            = {(size_t)group_count, (size_t)total_tokens, (size_t)N};

    return parallel_for(ctx, compute::nd_range_t(gws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
