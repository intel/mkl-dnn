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

#include "utils/dnnl_query.hpp"
#include "utils/parallel.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

int64_t wei_ab_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->k + k) * prb->n + n;
}
int64_t wei_ba_off_f(const prb_t *prb, int64_t mb, int64_t k, int64_t n) {
    return (mb * prb->n + n) * prb->k + k;
}

// Stores parameters that are invariant across different (mc, nc) chunks
// Precomputed once per problem to reduce overhead
struct chunk_params_t {
    // Pointers to memory objects (no ownership here)
    const dnn_mem_t *src_m = nullptr, *wei_m = nullptr, *bia_m = nullptr,
                    *dst_m = nullptr;
    const dnn_mem_t *src_scales = nullptr, *wei_scales = nullptr,
                    *dst_scales = nullptr;
    const dnn_mem_t *src_zps = nullptr, *wei_zps = nullptr, *dst_zps = nullptr;
    const dnn_mem_t *dropout_mask = nullptr;

    // Problem dims
    int64_t N = 0, K = 0;
    int64_t dst_M_group = 1, dst_N_group = 1;

    // Weights strides are invariant across chunks
    int64_t wei_k_stride = 0, wei_n_stride = 0;

    // Feature flags
    bool has_src_scale = false, has_wei_scale = false, has_dst_scale = false;
    bool has_dst_dynamic = false, has_dst_mx = false,
         has_dst_dynamic_fp = false;
    bool has_src_zp = false, has_wei_zp = false, has_dst_zp = false;
    bool has_wei_sbzp = false;
    bool has_src_single_scale = false, has_wei_single_scale = false;
    bool has_src_single_zp = false, has_wei_single_zp = false;

    // Quantization masks
    int src_scale_mask = 0, wei_scale_mask = 0, dst_scale_mask = 0;
    int src_zp_mask = 0, wei_zp_mask = 0, dst_zp_mask = 0;

    // Scale/zp group vectors
    std::vector<int64_t> src_scale_groups, wei_scale_groups;
    std::vector<int64_t> src_zp_groups, wei_zp_groups;
    std::vector<int64_t> dst_scale_groups;

    // Smallest quantization group
    int64_t smallest_k_group = 0;

    // Dst scale storage type
    dnnl_data_type_t dst_scale_dt;

    // Post-ops element masks (one pair per post-op entry).
    std::vector<std::pair<int, int>> v_po_masks;

    // Pre-fetched single-value quant params (valid when has_*_single_* is true).
    int src_zp_single = 0;
    float wei_zp_single = 0.f;
    float src_scale_single = 1.f, wei_scale_single = 1.f;

    // Data types
    dnnl_data_type_t bia_dt, dst_dt;
};

// Precompute parameters for compute_ref_matmul_chunk
// based on the problem definition and the provided arguments
static chunk_params_t make_chunk_params(const prb_t *prb, const args_t &args) {
    chunk_params_t p;

    p.src_m = &args.find(DNNL_ARG_SRC);
    p.wei_m = &args.find(DNNL_ARG_WEIGHTS);
    p.bia_m = &args.find(DNNL_ARG_BIAS);
    p.dst_m = &args.find(DNNL_ARG_DST);
    p.src_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    p.wei_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    p.dst_scales = &args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    p.src_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    p.wei_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    p.dst_zps = &args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    p.dropout_mask = &args.find(DNNL_ARG_ATTR_DROPOUT_MASK);

    p.has_src_scale = !prb->attr.scales.get(DNNL_ARG_SRC).is_def();
    p.has_wei_scale = !prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_def();
    p.has_dst_scale = !prb->attr.scales.get(DNNL_ARG_DST).is_def();
    p.has_dst_dynamic = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic();
    p.has_dst_mx = prb->attr.scales.get(DNNL_ARG_DST).is_mx();
    p.has_dst_dynamic_fp = prb->attr.scales.get(DNNL_ARG_DST).is_dynamic_fp();

    p.src_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, p.src_m->ndims());
    p.wei_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, p.wei_m->ndims());
    p.dst_scale_mask = prb->attr.scales.get_mask(
            DNNL_ARG_DST, dnnl_matmul, p.dst_m->ndims());

    p.has_src_single_scale = p.has_src_scale && p.src_scale_mask == 0;
    p.has_wei_single_scale = p.has_wei_scale && p.wei_scale_mask == 0;

    p.has_src_zp = !prb->attr.zero_points.get(DNNL_ARG_SRC).is_def();
    p.has_wei_zp = !prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).is_def();
    p.has_dst_zp = !prb->attr.zero_points.get(DNNL_ARG_DST).is_def();

    p.has_wei_sbzp = prb->attr.scales.get(DNNL_ARG_WEIGHTS).is_sbzp();

    p.src_zp_mask = p.has_src_zp ? prb->attr.zero_points.get_mask(DNNL_ARG_SRC,
                                           dnnl_matmul, p.src_m->ndims())
                                 : 0;
    p.wei_zp_mask = p.has_wei_zp
            ? prb->attr.zero_points.get_mask(
                      DNNL_ARG_WEIGHTS, dnnl_matmul, p.wei_m->ndims())
            : 0;
    p.dst_zp_mask = p.has_dst_zp ? prb->attr.zero_points.get_mask(DNNL_ARG_DST,
                                           dnnl_matmul, p.dst_m->ndims())
                                 : 0;

    p.has_src_single_zp = p.has_src_zp && p.src_zp_mask == 0;
    p.has_wei_single_zp = p.has_wei_zp && p.wei_zp_mask == 0;

    p.src_scale_groups = prb->attr.scales.get(DNNL_ARG_SRC).groups;
    p.wei_scale_groups = prb->attr.scales.get(DNNL_ARG_WEIGHTS).groups;
    p.src_zp_groups = prb->attr.zero_points.get(DNNL_ARG_SRC).groups;
    p.wei_zp_groups = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).groups;

    const int64_t src_scale_group = !p.src_scale_groups.empty()
            ? p.src_scale_groups[1]
            : ((p.src_scale_mask >> (p.src_m->ndims() - 1)) % 2) > 0 ? 1
                                                                     : prb->k;
    const int64_t wei_scale_group = !p.wei_scale_groups.empty()
            ? p.wei_scale_groups[0]
            : ((p.wei_scale_mask >> (p.wei_m->ndims() - 2)) % 2) > 0 ? 1
                                                                     : prb->k;
    const int64_t src_zp_group = !p.src_zp_groups.empty() ? p.src_zp_groups[1]
            : ((p.src_zp_mask >> (p.src_m->ndims() - 1)) % 2) > 0 ? 1
                                                                  : prb->k;
    const int64_t wei_zp_group = !p.wei_zp_groups.empty() ? p.wei_zp_groups[0]
            : ((p.wei_zp_mask >> (p.wei_m->ndims() - 2)) % 2) > 0 ? 1
                                                                  : prb->k;
    p.smallest_k_group = gcd<int64_t>(
            {src_scale_group, wei_scale_group, src_zp_group, wei_zp_group});

    p.dst_scale_dt = prb->attr.scales.get(DNNL_ARG_DST).dt;

    const auto &dsg = prb->attr.scales.get(DNNL_ARG_DST).groups;
    p.dst_M_group = !dsg.empty() ? dsg[0] : 1;
    p.dst_N_group = !dsg.empty() ? dsg[1] : 1;
    p.dst_scale_groups = dsg;

    p.v_po_masks = prb->attr.post_ops.get_po_masks(prb->ndims);

    p.src_zp_single = p.has_src_single_zp ? p.src_zps->get_elem(0) : 0;
    p.wei_zp_single = p.has_wei_single_zp ? p.wei_zps->get_f32_elem(0) : 0.f;
    p.src_scale_single
            = p.has_src_single_scale ? p.src_scales->get_f32_elem(0) : 1.f;
    p.wei_scale_single
            = p.has_wei_single_scale ? p.wei_scales->get_f32_elem(0) : 1.f;

    p.bia_dt = prb->bia_dt;
    p.dst_dt = prb->dst_dt();

    p.N = prb->n;
    p.K = prb->k;
    const int wei_ndims = p.wei_m->ndims();
    p.wei_k_stride = p.wei_m->strides()[wei_ndims - 2];
    p.wei_n_stride = p.wei_m->strides()[wei_ndims - 1];

    return p;
}

// Computational kernel for a single (mc, nc) chunk of output
//
// _base and _stride are used to compute the actual offsets as follows:
//   src(m, k)    = src_base + m * K + k (src is always row-major,
//                  for grouped matmul, src_base is the start of the group)
//   wei_ab(k, n) = wei_base + k * N + n (for scales and zps)
//   wei(k, n)    = wei_base + k * wei_k_stride + n * wei_n_stride
//   dst(m, n)    = (dst_row_base + m) * N + n
//   bia(m, n)    = bia_base + m * bia_m_stride + n * bia_n_stride
static void compute_ref_matmul_chunk(const chunk_params_t &p, int64_t M,
        int64_t Kg, int64_t mc, int64_t nc, int64_t src_base, int64_t wei_base,
        int64_t dst_row_base, int64_t bia_base, int64_t bia_m_stride,
        int64_t bia_n_stride, const attr_t &attr, const args_t &args,
        int64_t group_id = 0) {
    const int64_t N = p.N, K = p.K;
    const int64_t wei_k_stride = p.wei_k_stride, wei_n_stride = p.wei_n_stride;

    // Mutable per-element quant params; initialised to the single value when
    // applicable and overwritten per K-group otherwise.
    int src_zp = p.src_zp_single;
    float wei_zp = p.wei_zp_single;
    float src_scale = p.src_scale_single;
    float wei_scale = p.wei_scale_single;

    // Adjustment for K block to be processed
    // smallest_k_group is either the per-problem granularity or full K when unquantized
    // for varK grouped matmul, we should process up to Kg instead of K
    const int64_t k_block = MIN2(p.smallest_k_group, Kg);
    const int64_t n_k_groups = Kg /* empty K group */ ? Kg / k_block : 0;

    for_(int64_t m = mc * p.dst_M_group; m < MIN2((mc + 1) * p.dst_M_group, M);
            ++m)
    for_(int64_t n = nc * p.dst_N_group; n < MIN2((nc + 1) * p.dst_N_group, N);
            ++n)
    {
        float dst = 0;
        for (int64_t gK = 0; gK < n_k_groups; gK++) {
            const auto src_gK_off = src_base + m * K + gK * k_block;
            // Note: scales/zero-points are still always in `tag::abx` format.
            const auto wei_gK_off = wei_base + gK * k_block * N + n;

            if (p.has_src_zp && !p.has_src_single_zp) {
                const auto src_zp_idx = p.src_m->get_idx(src_gK_off,
                        p.src_zp_mask, p.src_m->ndims(), p.src_zp_groups);
                src_zp = p.src_zps->get_elem(src_zp_idx);
            }
            if (p.has_wei_zp && !p.has_wei_single_zp) {
                const auto wei_zp_idx = p.wei_m->get_idx(wei_gK_off,
                        p.wei_zp_mask, p.wei_m->ndims(), p.wei_zp_groups);
                wei_zp = p.wei_zps->get_f32_elem(wei_zp_idx);
            }

            if (p.has_src_scale && !p.has_src_single_scale) {
                const auto src_scale_idx = p.src_m->get_idx(src_gK_off,
                        p.src_scale_mask, p.src_m->ndims(), p.src_scale_groups);
                src_scale = p.src_scales->get_f32_elem(src_scale_idx);
            }
            if (p.has_wei_scale && !p.has_wei_single_scale) {
                const auto wei_scale_idx = p.wei_m->get_idx(wei_gK_off,
                        p.wei_scale_mask, p.wei_m->ndims(), p.wei_scale_groups);
                wei_scale = p.wei_scales->get_f32_elem(wei_scale_idx);
            }

            for (int64_t k = 0; k < k_block; ++k) {
                const auto kk = gK * k_block + k;
                const auto src_off = src_base + m * K + kk;
                const auto wei_off
                        = wei_base + kk * wei_k_stride + n * wei_n_stride;

                auto s = src_scale * (p.src_m->get_f32_elem(src_off) - src_zp);
                const auto wei_val = p.wei_m->get_f32_elem(wei_off);
                // Scales-before-zero-points scales the weight first, so the
                // zero-point is not folded into the weight value.
                auto w = p.has_wei_sbzp ? wei_scale * wei_val - wei_zp
                                        : wei_scale * (wei_val - wei_zp);

                dst += s * w;
            }
        }

        const auto dst_off = (dst_row_base + m) * N + n;
        if (p.bia_dt != dnnl_data_type_undef) {
            const auto bia_idx = bia_base + m * bia_m_stride + n * bia_n_stride;
            dst += p.bia_m->get_f32_elem(bia_idx);
        }

        auto v_po_vals = prepare_po_vals(
                *p.dst_m, args, p.v_po_masks, dst_off, group_id);
        maybe_dropout(attr, dst, dst_off, *p.dropout_mask);
        const auto sum_val = p.dst_m->get_f32_elem(dst_off);
        maybe_post_ops(attr, dst, sum_val, v_po_vals);

        // We use dst as temporary storage
        p.dst_m->set_f32_elem(dst_off, dst);
    }

    // Now we can do downconversion and write back to dst.
    // Compute scales if dyn_quant.
    float dst_scale = 1.f;
    if (p.has_dst_dynamic) {
        // Note: Mantissa-less dt would round-up zero to min normal.
        // Note: Mantissa-ed dt needs initial value to be zero to properly
        // handle the final value if the block is full of zero values.
        dst_scale = 0.f;
        for_(int64_t m = mc * p.dst_M_group;
                m < MIN2((mc + 1) * p.dst_M_group, M); ++m)
        for (int64_t n = nc * p.dst_N_group;
                n < MIN2((nc + 1) * p.dst_N_group, N); ++n) {
            const auto dst_off = (dst_row_base + m) * N + n;
            const float dst = fabsf(p.dst_m->get_f32_elem(dst_off));
            // The following check and explicit set of scale to NAN is done on
            // purpose to highlight the test case to adjust it as there's no
            // practical application of NAN scales.
            if (std::isnan(dst) || std::isnan(dst_scale)) {
                dst_scale = NAN;
                continue;
            }
            dst_scale = MAX2(dst, dst_scale);
        }
        if (p.has_dst_mx) {
            dst_scale
                    = round_to_nearest_representable(p.dst_scale_dt, dst_scale)
                    / round_to_nearest_representable(
                            p.dst_scale_dt, max_dt(p.dst_dt));
            dst_scale
                    = round_to_nearest_representable(p.dst_scale_dt, dst_scale);
        } else if (p.has_dst_dynamic_fp) {
            // Clamp the scale value to epsilon(dst_scale_dt) from the down and
            // to max(dst_scale_dt) from the up.
            // This is done to deal with zero block and small values obtained
            // after dividing a group max value.
            // See the following link for more details:
            // https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py
            dst_scale = MAX2(
                    MIN2(dst_scale / max_dt(p.dst_dt), max_dt(p.dst_scale_dt)),
                    epsilon_dt(p.dst_scale_dt));
            dst_scale
                    = round_to_nearest_representable(p.dst_scale_dt, dst_scale);
        }
        const auto dst_off
                = (dst_row_base + mc * p.dst_M_group) * N + nc * p.dst_N_group;
        const auto dscale_idx = p.dst_m->get_idx(dst_off, p.dst_scale_mask,
                p.dst_m->ndims(), p.dst_scale_groups);
        p.dst_scales->set_f32_elem(dscale_idx, dst_scale);
        // Pre-invert the scale to apply it as a multiplier for the group.
        // Note, that it can't be done upfront, as it must be written to
        // the memory before. Can't be zero.
        dst_scale = 1.f / dst_scale;
    }

    // Apply scales and downconvert.
    for_(int64_t m = mc * p.dst_M_group; m < MIN2((mc + 1) * p.dst_M_group, M);
            ++m)
    for_(int64_t n = nc * p.dst_N_group; n < MIN2((nc + 1) * p.dst_N_group, N);
            ++n)
    {
        int dst_zp = 0;
        const auto dst_off = (dst_row_base + m) * N + n;

        if (p.has_dst_zp) {
            const auto dst_zp_idx = p.dst_m->get_idx(dst_off, p.dst_zp_mask);
            dst_zp = p.dst_zps->get_elem(dst_zp_idx);
        }
        if (p.has_dst_scale && !p.has_dst_dynamic) {
            const auto dscale_idx = p.dst_m->get_idx(dst_off, p.dst_scale_mask,
                    p.dst_m->ndims(), p.dst_scale_groups);
            dst_scale = 1.f / p.dst_scales->get_f32_elem(dscale_idx);
        }
        float dst = p.dst_m->get_f32_elem(dst_off);
        float dst_val = dst_scale * dst + dst_zp;
        if (p.has_dst_dynamic) {
            // When the computed dst_scale is < 1.f, its inverse is > 1.f and
            // may push the value beyond the representable range of the dst dt,
            // including NaN or inf.
            // Saturate to the max/min representable value preserving sign
            // instead of overflowing according to spec.
            if (std::isnan(dst_val) || std::isinf(dst_val)) {
                float dst_val_sign = std::signbit(dst_val) ? -1 : 1;
                dst_val = max_dt(p.dst_dt) * dst_val_sign;
            } else {
                dst_val = MAX2(
                        MIN2(dst_val, max_dt(p.dst_dt)), lowest_dt(p.dst_dt));
            }
        }
        maybe_round(attr, DNNL_ARG_DST, dst_val, dst_off, p.dst_dt);
        p.dst_m->set_f32_elem(dst_off, dst_val);
    }
}

// Reference implementation for grouped matmul.
//
// Per group g, computes dst[g] = src[g] * wei[g]. Two layouts are supported,
// selected by which argument is grouped alongside src:
//
//   variable M (2Dx3D): src+dst grouped. M_g varies per group; K, N fixed.
//     src/dst are concatenated row-major buffers [total_M, K] / [total_M, N];
//     wei is per-group dense [G, K, N]. Supports a per-group bias [G, N].
//
//   variable K (2Dx2D): src+wei grouped. K_g varies per group; M, N fixed.
//     src is row-major [M, total_K] in the reference, wei is row-major
//     [total_K, N], dst is dense [G, M, N]. No bias.
//
// Per-group ranges are read from the grouped memory descriptor offsets.
void compute_ref_grouped_matmul(const prb_t *prb, const args_t &args) {
    const bool var_M = prb->sparse_options.is_grouped(DNNL_ARG_DST);

    const int64_t group_count = prb->sparse_options.get_group_count();
    const auto &group_sizes = prb->sparse_options.get_group_sizes(DNNL_ARG_SRC);

    std::vector<int64_t> group_offsets(group_count + 1);
    group_offsets[0] = 0;
    int64_t max_group_size = 0;
    for (int64_t g = 0; g < group_count; g++) {
        group_offsets[g + 1] = group_offsets[g] + group_sizes[g];
        max_group_size = MAX2(max_group_size, group_sizes[g]);
    }

    // Precompute common parameters for the different chunks computations
    const chunk_params_t params = make_chunk_params(prb, args);

    // For var_M, M differs per group; for var_K, M is fixed
    const int64_t M_chunks
            = div_up(var_M ? max_group_size : prb->m, params.dst_M_group);
    const int64_t N_chunks = div_up(prb->n, params.dst_N_group);

    // Parallelize over groups and (mc, nc) chunks within each group
    benchdnn_parallel_nd(group_count, M_chunks, N_chunks,
            [&](int64_t g, int64_t mc, int64_t nc) {
        const int64_t off = group_offsets[g];
        const int64_t M = var_M ? group_sizes[g] : prb->m;
        if (M == 0) return;
        if (mc * params.dst_M_group >= M) return;

        // Per-group base offsets:
        //   src(m, k) = src_base + m * K + k
        //   wei(k, n) = wei_base + k * wei_k_stride + n * wei_n_stride
        //   dst(m, n) = (dst_row_base + m) * N + n
        //   bia(m, n) = bia_base + m * bia_m_stride + n * bia_n_stride
        //
        // Note, that var_M offsets groups along rows (off*K),
        // var_K along K-columns (off)
        const int64_t src_base = var_M ? off * prb->k : off;
        const int64_t wei_base
                = var_M ? g * prb->k * prb->n : off * params.wei_k_stride;
        const int64_t dst_row_base = var_M ? off : g * prb->m;

        // Row stride is total_K, however var_K reduces over the group K_g
        const int64_t Kg = var_M ? prb->k : group_sizes[g];

        int64_t bia_base = 0, bia_m_stride = 0, bia_n_stride = 0;
        if (var_M && params.bia_dt != dnnl_data_type_undef) {
            bia_base = g * prb->n;
            bia_n_stride = 1; // bias is per-group, not per-row
        }

        compute_ref_matmul_chunk(params, M, Kg, mc, nc, src_base, wei_base,
                dst_row_base, bia_base, bia_m_stride, bia_n_stride, prb->attr,
                args, /* group_id = */ g);
    });
}

void compute_ref_matmul(const prb_t *prb, const args_t &args) {
    // Fast return if any dim is zero. Common logic doesn't apply because of
    // broadcast semantics.
    {
        const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);
        for (int d = 0; d < dst_m.ndims(); d++) {
            if (prb->src_dims()[d] == 0 || prb->weights_dims()[d] == 0) return;
        }
    }

    // Precompute common parameters for the different chunks computations
    const chunk_params_t params = make_chunk_params(prb, args);

    // Compute output in chunks of dst_M_group x dst_N_group
    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;
    const int64_t MB = prb->mb;

    const int64_t M_chunks = div_up(M, params.dst_M_group);
    const int64_t N_chunks = div_up(N, params.dst_N_group);

    const dnn_mem_t &dst_m = *params.dst_m;
    const int batch_ndims = dst_m.ndims() - 2;

    const auto src_broadcast_mask = prb->src_broadcast_mask();
    const auto wei_broadcast_mask = prb->weights_broadcast_mask();
    const auto bias_broadcast_mask = prb->bias_broadcast_mask();

    benchdnn_parallel_nd(
            MB, M_chunks, N_chunks, [&](int64_t mb, int64_t mc, int64_t nc) {
        int64_t src_mb = 0;
        int64_t wei_mb = 0;
        if (MB > 1) {
            src_mb = dst_m.get_idx(mb, src_broadcast_mask, batch_ndims);
            wei_mb = dst_m.get_idx(mb, wei_broadcast_mask, batch_ndims);
        }

        int64_t bia_base = 0, bia_m_stride = 0, bia_n_stride = 0;
        if (params.bia_dt != dnnl_data_type_undef) {
            bia_base = dst_m.get_idx(
                    dst_off_f(prb, mb, 0, 0), bias_broadcast_mask);
            bia_m_stride = dst_m.get_idx(dst_off_f(prb, mb, 1, 0),
                                   bias_broadcast_mask)
                    - bia_base;
            bia_n_stride = dst_m.get_idx(dst_off_f(prb, mb, 0, 1),
                                   bias_broadcast_mask)
                    - bia_base;
        }

        // Precompute offsets for this chunk
        const int64_t src_row_base = src_mb * M;
        const int64_t wei_base = wei_mb * K * N;
        const int64_t dst_row_base = mb * M;

        compute_ref_matmul_chunk(params, M, /* Kg = */ K, mc, nc,
                /* src_base = */ src_row_base * K, wei_base, dst_row_base,
                bia_base, bia_m_stride, bia_n_stride, prb->attr, args);
    });
}

void cvt_coo_indices_to_csr_pointers(const int32_t *indices, int32_t *pointers,
        const int nnz, const int nrows) {
    for (int i = 0; i < nnz; ++i) {
        ++pointers[indices[i] + 1];
    }
    for (int i = 0; i < nrows; ++i) {
        pointers[i + 1] += pointers[i];
    }
}

void compute_ref_sparse_matmul(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src_m = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &wei_m = args.find(DNNL_ARG_WEIGHTS);
    const dnn_mem_t &dst_m = args.find(DNNL_ARG_DST);

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    const bool is_src_sparse
            = src_encoding == dnnl_csr || src_encoding == dnnl_coo;
    const bool is_wei_sparse
            = wei_encoding == dnnl_csr || wei_encoding == dnnl_coo;
    auto encoding = is_src_sparse ? src_encoding : wei_encoding;

    const int64_t M = prb->m;
    const int64_t N = prb->n;
    const int64_t K = prb->k;

    // TODO: Depending on the matrix dimensions the pointer buffer may take
    // up a significant amount of memory. This wil require a mechanism to
    // register the memory needed for the current scratchpad during
    // COO-to-CSR format conversion.
    std::vector<int32_t> pointer_buffer(1 + (is_src_sparse ? M : K), 0);

    // Batch is not supported.
    const int64_t mb = 0;
    benchdnn_parallel_nd(M, N, [&](int64_t m, int64_t n) {
        dst_m.set_f32_elem(dst_off_f(prb, mb, m, n), 0.0f);
    });

    if (is_wei_sparse) {
        int32_t *wei_indices = wei_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *wei_pointers = wei_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *wei_row_indices = wei_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(wei_m.md_);

            benchdnn_parallel_nd(
                    K + 1, [&](int64_t i) { pointer_buffer[i] = 0; });
            cvt_coo_indices_to_csr_pointers(
                    wei_row_indices, pointer_buffer.data(), nnz, K);
            wei_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            for (int64_t k = 0; k < K; k++) {
                const int64_t row_start = wei_pointers[k];
                const int64_t row_end = wei_pointers[k + 1];
                for (int64_t n = row_start; n < row_end; n++) {
                    const int64_t src_idx = src_off_f(prb, mb, m, k);
                    const int64_t dst_idx
                            = dst_off_f(prb, mb, m, wei_indices[n]);
                    const float src_val = src_m.get_f32_elem(src_idx);
                    const float wei_val = wei_m.get_elem(n, 0);
                    float dst_val = dst_m.get_f32_elem(dst_idx);
                    dst_val += src_val * wei_val;
                    dst_m.set_f32_elem(dst_idx, dst_val);
                }
            }
        });
    } else if (is_src_sparse) {
        int32_t *src_indices = src_m.get_mapped_pointer<int32_t>(
                encoding == dnnl_csr ? 1 : 2);
        int32_t *src_pointers = src_m.get_mapped_pointer<int32_t>(2);

        if (encoding == dnnl_coo) {
            int32_t *src_row_indices = src_m.get_mapped_pointer<int32_t>(1);
            const int64_t nnz = query_md_nnz(src_m.md_);
            cvt_coo_indices_to_csr_pointers(
                    src_row_indices, pointer_buffer.data(), nnz, M);
            src_pointers = pointer_buffer.data();
        }

        benchdnn_parallel_nd(M, [&](int64_t m) {
            const int64_t row_start = src_pointers[m];
            const int64_t row_end = src_pointers[m + 1];
            for (int64_t n = 0; n < N; n++) {
                const int64_t dst_idx = dst_off_f(prb, mb, m, n);
                float dst_val = dst_m.get_f32_elem(dst_idx);

                for (int64_t k = row_start; k < row_end; k++) {
                    const int64_t wei_idx
                            = wei_ba_off_f(prb, mb, src_indices[k], n);
                    const float src_val = src_m.get_elem(k, 0);
                    const float wei_val = wei_m.get_f32_elem(wei_idx);
                    dst_val += src_val * wei_val;
                }
                dst_m.set_f32_elem(dst_idx, dst_val);
            }
        });
    }
}

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    const prb_t *prb = prb_t::from(base_prb);
    if (prim_ref) {
        SAFE_V(execute_and_wait(prim_ref, args));
        return;
    }

    const auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    if (prb->sparse_options.is_grouped(DNNL_ARG_SRC)
            && (prb->sparse_options.is_grouped(DNNL_ARG_DST)
                    || prb->sparse_options.is_grouped(DNNL_ARG_WEIGHTS))) {
        compute_ref_grouped_matmul(prb, args);
    } else if (src_encoding == dnnl_csr || wei_encoding == dnnl_csr
            || src_encoding == dnnl_coo || wei_encoding == dnnl_coo) {
        compute_ref_sparse_matmul(prb, args);
    } else {
        compute_ref_matmul(prb, args);
    }
}

} // namespace matmul
