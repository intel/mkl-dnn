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

#include "gpu/intel/include/philox.h"
#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/sdpa/utils.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_kq.h"
#include "gemm_ktq.h"
#include "gemm_qdSt.h"
#include "gemm_vs.h"
#include "gemm_vtdA.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define sg_per_wg_BcBr \
    (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n) // same for kq, vtdA
#define sg_per_wg_BcD \
    (ugemm_vs_sg_per_wg_m * ugemm_vs_sg_per_wg_n) // same for qdSt and vs
#define sg_per_wg_BrD (ugemm_ktq_sg_per_wg_m * ugemm_ktq_sg_per_wg_n)
#define sg_per_wg MAX(sg_per_wg_BcBr, MAX(sg_per_wg_BcD, sg_per_wg_BrD))

#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type; // Bc*Br tile
typedef ugemm_qdSt_c_type a_tile_type; // Bc*D tile
typedef ugemm_vtdA_c_type p_tile_type; // Br*Bc tile (.T)
typedef ugemm_vs_c_type dv_tile_type; // D*Bc tile
typedef ugemm_ktq_c_type ktq_tile_type; // D*Br tile

#if WITH_DROPOUT
#define dropout_mul(x, y) ((x) * (y))
#define dropout_keep(_ph, _orow, _ocol) \
    (((_orow) < max_r && (_ocol) < max_c) && ((_ph) > threshold) ? inv_q : 0.f)

/* Consecutive indices run across lanes here: each lane makes one Philox block
   per four columns and the group's words are exchanged with shuffles. */
#define DROPOUT_GROUPED_SHAPE_OK(br, bc, nbr, nbc) \
    (SUBGROUP_SIZE == 16 && (((br) * (nbr)) % 16) == 0 \
            && (((bc) * (nbc)) & 3) == 0)
#define DROPOUT_GROUPED_RT_OK \
    (use_dropout_offset && (k_stride & 3) == 0 \
            && (((batch_head_base + (ulong)tile_offset_r) & 3UL) == 0))

#define DROPOUT_FILL_GROUPED(scale, br, bc, nbr, nbc) \
    do { \
        const int _l = get_sub_group_local_id(); \
        const int _w = _l & 3; \
        for (int j0 = 0; j0 < (bc) * (nbc); j0 += 4) { \
            for (int i0 = 0; i0 < (br) * (nbr); i0 += SUBGROUP_SIZE) { \
                const ulong _idx = batch_head_base \
                        + (ulong)(tile_offset_c + j0 + _w) * (ulong)k_stride \
                        + (ulong)(tile_offset_r + i0 + (_l & ~3)); \
                const uint4 _b = philox_4x32_vec4_w_offset(_idx, seed, offset); \
                for (int jj = 0; jj < 4; jj++) { \
                    const int _src = (_l & ~3) | jj; \
                    const uint _w0 = intel_sub_group_shuffle(_b.s0, _src); \
                    const uint _w1 = intel_sub_group_shuffle(_b.s1, _src); \
                    const uint _w2 = intel_sub_group_shuffle(_b.s2, _src); \
                    const uint _w3 = intel_sub_group_shuffle(_b.s3, _src); \
                    const uint _v = _w == 0 ? _w0 \
                            : _w == 1       ? _w1 \
                            : _w == 2       ? _w2 \
                                            : _w3; \
                    tile_access(scale, i0, j0 + jj, SUBGROUP_SIZE, br, bc, nbr) \
                            = dropout_keep(_v, tile_offset_r + i0 + _l, \
                                    tile_offset_c + j0 + jj); \
                } \
            } \
        } \
    } while (0)

#define dropout_predicate(offset_r, offset_c) \
    ({ \
        ulong _goff = batch_head_base + (ulong)offset_c * (ulong)k_stride \
                + (ulong)offset_r; \
        uint _philox = use_dropout_offset \
                ? philox_4x32_w_offset(_goff, seed, offset) \
                : philox_4x32(_goff, seed); \
        (offset_r < max_r && offset_c < max_c) && (_philox > threshold); \
    })

/*
    Apply inverted dropout in-place to an S tile (s_tile_type = ugemm_kq_c_type
*/
inline void apply_dropout_s_tile(s_tile_type *tile, int tile_offset_r,
        int tile_offset_c, int max_r, int max_c, ulong batch_head_base,
        int k_stride, int use_dropout_offset, long seed, long offset,
        uint threshold, float inv_q) {

    volatile s_tile_type scale_tile;
#if DROPOUT_GROUPED_SHAPE_OK(ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, \
        ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1)
    if (DROPOUT_GROUPED_RT_OK) {
        DROPOUT_FILL_GROUPED(scale_tile, ugemm_kq_c_type_block0,
                ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                ugemm_kq_c_type_nblock1);
    } else
#endif
    {
        tile_predicated_select(scale_tile, tile_offset_r, tile_offset_c,
                dropout_predicate, inv_q, 0.f, SUBGROUP_SIZE,
                ugemm_kq_c_type_block0, ugemm_kq_c_type_block1,
                ugemm_kq_c_type_nblock0, ugemm_kq_c_type_nblock1);
    }

    s_tile_type tmp = *tile;
    tile_binary(tmp, scale_tile, dropout_mul);
    *tile = tmp;
}

/* Apply inverted dropout in-place to a dP tile (p_tile_type = ugemm_vtdA_c_type).
 * the dropout Jacobian: dP = dP_raw * Z / q. */
inline void apply_dropout_dP_tile(p_tile_type *tile, int tile_offset_r,
        int tile_offset_c, int max_r, int max_c, ulong batch_head_base,
        int k_stride, int use_dropout_offset, long seed, long offset,
        uint threshold, float inv_q) {

    volatile p_tile_type scale_p_tile;
#if DROPOUT_GROUPED_SHAPE_OK(ugemm_vtdA_c_type_block0, \
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0, \
        ugemm_vtdA_c_type_nblock1)
    if (DROPOUT_GROUPED_RT_OK) {
        DROPOUT_FILL_GROUPED(scale_p_tile, ugemm_vtdA_c_type_block0,
                ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
                ugemm_vtdA_c_type_nblock1);
    } else
#endif
    {
        tile_predicated_select(scale_p_tile, tile_offset_r, tile_offset_c,
                dropout_predicate, inv_q, 0.f, SUBGROUP_SIZE,
                ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
                ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1);
    }

    p_tile_type tmp = *tile;
    tile_binary(tmp, scale_p_tile, dropout_mul);
    *tile = tmp;
}

#undef dropout_mul
#undef dropout_predicate
#undef dropout_keep
#undef DROPOUT_GROUPED_SHAPE_OK
#undef DROPOUT_GROUPED_RT_OK
#undef DROPOUT_FILL_GROUPED
#endif

#ifdef QRY_DT_F16
#define VEC_TYPE2 half2
#elif defined(QRY_DT_BF16)
#define VEC_TYPE2 ushort2
#elif !defined(QRY_DT_F32)
#error "Data type not supported for VEC_TYPE2"
#endif

typedef NATIVE_LAYOUT_TYPE(DST_DATA_T) dst_tile_data_t;
typedef NATIVE_LAYOUT_TYPE(MSK_DATA_T) msk_tile_data_t;
#define FMA_TYPE NATIVE_LAYOUT_TYPE(QRY_DATA_T)

#define CONVERT_TILE_DATA_T(v) as_native_layout(CONVERT_DATA_T(v))

#define CONVERT_TILE_FLOAT_MSK_T(v) \
    into_float(AS_NATIVE_LAYOUT_TYPE(MSK_DATA_T, v))

/* FMA tile storage, which follows the query type rather than the ambient dst
 * type -- see FMA_TYPE above. */
#define CONVERT_TILE_FMA_T(v) \
    as_native_layout(CONCAT2(into_, QRY_DATA_T)(convert_float(v)))
#define CONVERT_TILE_FLOAT_FMA_T(v) \
    into_float(AS_NATIVE_LAYOUT_TYPE(QRY_DATA_T, v))

DECLARE_2D_TILE(q_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)

DECLARE_2D_TILE(dq_tile_type, float, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
DECLARE_2D_TILE_BLOCK_OPS(
        dq_tile_type, float, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
DECLARE_2D_TILE_COPY_REBLOCK(q_tile_type, SUBGROUP_SIZE, D_MAX, 1, 1,
        q_tile_sg_n, dq_tile_type, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n,
        CONVERT_TILE_FLOAT_FMA_T)

#if K_IN_SLM
#if TRANSPOSE_K

#define k_tile_t_sg_n DIV_UP(ugemm_kq_wg_tile_m, sg_per_wg)
DECLARE_2D_TILE(
        k_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, k_tile_t_sg_n)
#if BLOCK_K
DECLARE_2D_TILE_BLOCK_OPS(
        k_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, k_tile_t_sg_n)
#endif

#else

#define dmax_tile_sg_n DIV_UP(D_MAX, sg_per_wg)
DECLARE_2D_TILE(k_tile_type, FMA_TYPE, SUBGROUP_SIZE, ugemm_kq_wg_tile_m, 1, 1,
        dmax_tile_sg_n)
#if BLOCK_K
DECLARE_2D_TILE_BLOCK_OPS(k_tile_type, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_wg_tile_m, 1, 1, dmax_tile_sg_n)
#endif
#endif
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(s_tile_type_packed_t, uint, SUBGROUP_SIZE,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_block0 / 2,
        ugemm_kq_c_type_nblock1, ugemm_kq_c_type_nblock0)

DECLARE_2D_TILE(p_tile_type_packed, uint, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1 / 2,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1)

DECLARE_2D_TILE(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n)

DECLARE_2D_TILE(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1)
DECLARE_2D_TILE_BLOCK_OPS(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
        p_sum_tile_type, float, SUBGROUP_SIZE, ugemm_vtdA_sg_tile_n, 1, 1, 1)

#if BROADCAST_MASK_Q
#define mask_br ugemm_kq_sg_tile_m
#define mask_bc 1
#define mask_nbr 1
#define mask_nbc 1
#else
#define mask_br ugemm_kq_c_type_block0
#define mask_bc ugemm_kq_c_type_block1
#define mask_nbr ugemm_kq_c_type_nblock0
#define mask_nbc ugemm_kq_c_type_nblock1
#endif

DECLARE_2D_TILE(qmask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n,
        1, 1, 1)
DECLARE_2D_TILE(kmask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_m,
        1, 1, 1)

#if WITH_ATTN_MASK
DECLARE_2D_TILE(mask_tile_type, msk_tile_data_t, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)

#if BROADCAST_MASK_Q
DECLARE_2D_TILE_BLOCK_OPS(mask_tile_type, msk_tile_data_t, SUBGROUP_SIZE,
        mask_br, mask_bc, mask_nbr, mask_nbc)
#endif
DECLARE_2D_TILE(mask_tile_type_float, float, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc)
DECLARE_2D_TILE_COPY_REBLOCK(mask_tile_type, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc, CONVERT_TILE_FLOAT_MSK_T)
#endif

DECLARE_2D_TILE(a_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE,
        ugemm_qdSt_c_type_block0, ugemm_qdSt_c_type_block1,
        ugemm_qdSt_c_type_nblock0, ugemm_qdSt_c_type_nblock1, a_tile_type_dst,
        SUBGROUP_SIZE, ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n,
        CONVERT_TILE_DATA_T)

DECLARE_2D_TILE(dv_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(dv_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(dv_tile_type, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, dv_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n, CONVERT_TILE_DATA_T)

DECLARE_2D_TILE(dq_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(dq_tile_type_dst, dst_tile_data_t, SUBGROUP_SIZE,
        ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(ktq_tile_type, SUBGROUP_SIZE,
        ugemm_ktq_c_type_block0, ugemm_ktq_c_type_block1,
        ugemm_ktq_c_type_nblock0, ugemm_ktq_c_type_nblock1, dq_tile_type_dst,
        SUBGROUP_SIZE, ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n,
        CONVERT_TILE_DATA_T)

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n, CONVERT_TILE_FMA_T)

#define CONVERT_TILE_IDENT(v) (v)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, p_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1,
        CONVERT_TILE_IDENT)
#undef CONVERT_TILE_IDENT

DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1,
        p_tile_type_reblock, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0, 1,
        ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1, CONVERT_TILE_FMA_T)
DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1, p_tile_type,
        SUBGROUP_SIZE, ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1,
        CONVERT_TILE_FLOAT_FMA_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, qmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, qmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_m, 1, 1, 1)
DECLARE_2D_TILE_VREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, p_sum_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
#if WITH_ATTN_MASK
DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif

DECLARE_2D_TILE_SLM_ADD(dv_tile_type, float, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1)
#if (ugemm_qdSt_c_type_block0 != ugemm_vs_c_type_block0) \
        || (ugemm_qdSt_c_type_block1 != ugemm_vs_c_type_block1) \
        || (ugemm_qdSt_c_type_nblock0 != ugemm_vs_c_type_nblock0) \
        || (ugemm_qdSt_c_type_nblock1 != ugemm_vs_c_type_nblock1)
DECLARE_2D_TILE_SLM_ADD(a_tile_type, float, SUBGROUP_SIZE,
        ugemm_qdSt_c_type_block0, ugemm_qdSt_c_type_block1,
        ugemm_qdSt_c_type_nblock0, ugemm_qdSt_c_type_nblock1)
#endif
DECLARE_2D_TILE_SLM_ADD_T(a_tile_type, float, SUBGROUP_SIZE,
        ugemm_qdSt_c_type_block0, ugemm_qdSt_c_type_block1,
        ugemm_qdSt_c_type_nblock0, ugemm_qdSt_c_type_nblock1)

#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c, load_rem) \
    if (load_rem) { \
        tile_load_block(t, ptr, n, ld, off_r, off_c); \
    } else { \
        tile_load_block(t, ptr, ld, off_r, off_c); \
    }

#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c, store_rem) \
    if (store_rem) { \
        tile_store_block(t, ptr, n, ld, off_r, off_c); \
    } else { \
        tile_store_block(t, ptr, ld, off_r, off_c); \
    }

#define binary_add(x, y) ((x) + (y))

#if K_IN_SLM
inline void tile_load_k(k_tile_type *K_tile, const global KEY_DATA_T *K,
        int seq_len, int head_size, int ldk, int seq_off, int sg_ij,
        int load_rem) {

#if TRANSPOSE_K
    // Bc / n_sg -- each sg loads k_tile_t_sg_n k-columns
    uint k0_copy = k_tile_t_sg_n * sg_ij;
    // Coalesced load from d×k column-major memory (d contiguous, k strided)
#if BLOCK_K
    tile_load_block(K_tile, K, ldk, 0, seq_off + k0_copy);
#else
    tile_load(K_tile, K, head_size, seq_len, ldk, 0, seq_off + k0_copy);
#endif

#else
    // D_MAX / n_sg
    uint k0_copy = dmax_tile_sg_n * sg_ij;
#if BLOCK_K
    // can ignore load_rem due to d_full requirement
    tile_load_block(K_tile, K, ldk, seq_off, k0_copy);
#else
    tile_load(K_tile, K, seq_len, head_size, ldk, seq_off, k0_copy);
#endif

#endif
}

inline void tile_store_k_slm(
        k_tile_type *K_tile, local KEY_DATA_T *K_slm, int sg_ij) {

#if TRANSPOSE_K
    // Bc / n_sg -- tile is D*Bc, write transposed to SLM (Bc*D)
    uint k0_copy = k_tile_t_sg_n * sg_ij;
#if USE_SYSTOLIC_UKERNEL
    tile_store_t_sys_src11(*K_tile, K_slm, SUBGROUP_SIZE, D_MAX, D_MAX,
            ugemm_kq_wg_tile_m, 0, k0_copy);
#else
    tile_store_t_packed_src1(
            *K_tile, K_slm, ugemm_kq_sg_tile_m, D_MAX, k0_copy, 0);
#endif

#else

    uint k0_copy = dmax_tile_sg_n * sg_ij;
#if USE_SYSTOLIC_UKERNEL
    tile_store_sys_src1(*K_tile, K_slm, SUBGROUP_SIZE, D_MAX,
            ugemm_kq_wg_tile_m, D_MAX, 0, k0_copy);
#else
    tile_store_packed_src1(
            *K_tile, K_slm, ugemm_kq_sg_tile_m, D_MAX, 0, k0_copy);
#endif

#endif
}
#endif

#if KV_GROUP_SIZE > 1
#define IS_GQA 1
#if defined(DST_DT_F16) || defined(DST_DT_BF16)
#define NEEDS_INTERMEDIATE_DKV 1
#else
#define NEEDS_INTERMEDIATE_DKV 0
#endif
#endif

#if QRY_DATA_T != float
#define NEEDS_INTERMEDIATE_DQ 1
#endif

#if IS_GQA
#define DST_DATA_T_DKDV float
#else
#define DST_DATA_T_DKDV DST_DATA_T
#endif

#if DIRECT_DQ
#define DST_DATA_T_DQ DST_DATA_T
#else
// needs intermediate f32 output before postprocess_dQ
#define DST_DATA_T_DQ float
#endif

// round f32 intermediate values to DST_DATA_T precision before GQA atomic
// accumulation. Although less accurate, it matches the unfused path
// where each query group matmul output passes through DST_DATA_T
// storage before the reduction
inline float round_to_dst(float v) {
    volatile DST_DATA_T tmp = CONVERT_DATA_T(v);
    return CONVERT_FLOAT_T(tmp);
}

inline void tile_store_dV(dv_tile_type *dV_tile_slm, global DST_DATA_T_DKDV *dV,
        int m, int n, int ld, int offset_r, int offset_c, int rem) {

#if IS_GQA
    tile_elementwise_s(*dV_tile_slm, round_to_dst);
    tile_atomic_add(*dV_tile_slm, dV, m, n, ld, offset_r, offset_c);

#else // MHA update

    dv_tile_type_dst dV_tile_dst; // convert to half
    tile_copy_reblock(*dV_tile_slm, &dV_tile_dst);
#if BLOCK_DV
    tile_store_block_rem_q(dV_tile_dst, (global dst_tile_data_t *)dV, n, ld,
            offset_r, offset_c, rem)
#else
    tile_store(dV_tile_dst, (global dst_tile_data_t *)dV, m, n, ld, offset_r,
            offset_c);
#endif
#endif
}

#if TRANSPOSE_K
// uses transposed dv_tile_type (D*Bc) for dK update
inline void tile_store_dK_t(dv_tile_type *dK_tile, global DST_DATA_T_DKDV *dK,
        int m, int n, int ld, int offset_r, int offset_c, int rem) {

#if IS_GQA
    tile_elementwise_s(*dK_tile, round_to_dst);
    tile_atomic_add(*dK_tile, dK, m, n, ld, offset_r, offset_c);
#else // MHA update
    dv_tile_type_dst dK_tile_dst;
    tile_copy_reblock(*dK_tile, &dK_tile_dst);
#if BLOCK_DK
    tile_store_block_rem_q(dK_tile_dst, (global dst_tile_data_t *)dK, n, ld,
            offset_r, offset_c, rem)
#else
    tile_store(dK_tile_dst, (global dst_tile_data_t *)dK, m, n, ld, offset_r,
            offset_c);
#endif
#endif
}

#else

// uses qdSt tile (Bc*D) for dK update
inline void tile_store_dK(a_tile_type *dK_tile, global DST_DATA_T_DKDV *dK,
        int m, int n, int ld, int offset_r, int offset_c) {

#if IS_GQA
    tile_elementwise_s(*dK_tile, round_to_dst);
    tile_atomic_add(*dK_tile, dK, m, n, ld, offset_r, offset_c);
#else // MHA update

    a_tile_type_dst dK_tile_dst;
    tile_copy_reblock(*dK_tile, &dK_tile_dst);
#if BLOCK_DK
    tile_store_block(
            dK_tile_dst, (global dst_tile_data_t *)dK, ld, offset_r, offset_c);
#else
    tile_store(dK_tile_dst, (global dst_tile_data_t *)dK, m, n, ld, offset_r,
            offset_c);
#endif

#endif
}

#endif

#define DO_MM 1

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa_bwd(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, const global float *ws,
        const global float *Di, const global DST_DATA_T *A,
        const global DST_DATA_T *dA,
#if WITH_DS
        global DST_DATA_T *dS, // expensive, optional intermediate
#endif
        global DST_DATA_T_DKDV *dK, global DST_DATA_T_DQ *dQ,
        global DST_DATA_T_DKDV *dV,
#if WITH_HOST_SCALE
        float scalar_scale, float inv_scalar_scale,
#else
        const global SCALE_DATA_T *scale_ptr,
#endif
#if WITH_DROPOUT
        int use_dropout_offset,
#if DROPOUT_HOST_SCALARS
        long dropout_seed, long dropout_offset, float dropout_p,
#else
        const global long *dropout_seed_buf,
        const global long *dropout_offset_buf,
        const global float *dropout_p_buf,
#endif
#endif
        int d, int k, int q, const int attn_mask_type
#if WITH_ATTN_MASK
        ,
        const global MSK_DATA_T *msk
#endif
        ,
        constant long *stride_params, const int remainder_k,
        const int remainder_q) {

    BWD_UNPACK_STRIDE_PARAMS(stride_params)
#if WITH_ATTN_MASK
    BWD_UNPACK_MSK_PARAMS(stride_params)
#endif

    uint wg_k = get_group_id(0);

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint b1 = get_group_id(2);

    // TODO: batch q=1 cases to KV_GROUP_SIZE
    uint b0, b0_kv;
    b0 = get_group_id(1);
    b0_kv = b0 / KV_GROUP_SIZE;

    uint wg_i0 = wg_k * ugemm_kq_wg_tile_m;

    const uint preprocess_batch = b1 * (DST_D1 * q) + b0 * q;
    const global float *ws_logsumexp = ws + preprocess_batch;
    Di += preprocess_batch;

    /* Calculate the number of keys to process */
    int q0end = q;
    int qdiag0 = 0; // potentially offset starting idx in causal mask cases
#if WITH_CAUSAL_MASK
    if (attn_mask_type == ATTN_MASK_TOP_LEFT) {
        qdiag0 = max(0, (int)(wg_i0));
    } else {
        qdiag0 = max(0, (int)(wg_i0 + (q - k)));
    }
#endif

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;
    uint ldda = DA_S2;

    /* leading dimensions for gradient outputs */
#if NEEDS_INTERMEDIATE_DKV
#if TRANSPOSE_K
    uint lddk = (uint)d;
#else
    uint lddk = (uint)k;
#endif
    uint lddv = (uint)d;
#else
    /* diff_key_md may not share key_md's transpose, use max to get the
     * sequence stride regardless of dK orientation */
    uint lddk = TRANSPOSE_K ? MAX(DK_S2, DK_S3) : DK_S2;
    uint lddv = DV_S2;
#endif

#if NEEDS_INTERMEDIATE_DQ
    uint lddq = (uint)d;
#else
    uint lddq = DQ_S2;
#endif

    /* Subgroup IDs for each GEMM, although total number of
     * sg per wg may be shared
     * ordering may differ due to transposes */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vtdA = sg_ij % ugemm_vtdA_sg_per_wg_m;
    uint sg_j_vtdA = sg_ij / ugemm_vtdA_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    uint sg_i_qdSt = sg_ij % ugemm_qdSt_sg_per_wg_m;
    uint sg_j_qdSt = sg_ij / ugemm_qdSt_sg_per_wg_m;

    uint sg_i_ktq = sg_ij % ugemm_ktq_sg_per_wg_m;
    uint sg_j_ktq = sg_ij / ugemm_ktq_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#if K_IN_SLM
#define K_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(KEY_DATA_T))
#else
#define K_slm_size 0
#endif
#define S_slm_size (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(FMA_TYPE))
#if USE_SYSTOLIC_UKERNEL
#define dSt_slm_size \
    (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(FMA_TYPE))
#else
#define dSt_slm_size 0
#endif

#define dK_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))

#define ugemm_slm_size \
    MAX(MAX(MAX(MAX(ugemm_kq_slm_size, ugemm_vs_slm_size), \
                    ugemm_vtdA_slm_size), \
                ugemm_qdSt_slm_size), \
            ugemm_ktq_slm_size)

    local char slm[K_slm_size + S_slm_size + dSt_slm_size + ugemm_slm_size
            + dK_slm_size];

#if K_IN_SLM
    local KEY_DATA_T *K_slm = (local KEY_DATA_T *)&slm[0];
#endif

    // S_slm, softmax for ugemm_vs also reused for dS
    local QRY_DATA_T *S_slm = (local QRY_DATA_T *)&slm[K_slm_size];
#if USE_SYSTOLIC_UKERNEL
    local QRY_DATA_T *dSt_slm
            = (local QRY_DATA_T *)&slm[K_slm_size + S_slm_size];
#endif

    // ugemm scratch space
    local char *ugemm_slm
            = (local char *)(&slm[K_slm_size + S_slm_size + dSt_slm_size]);

    // used for accumulation of dK across q-loop (dV accumulates in registers)
    local float *dK_slm = (local float *)&slm[K_slm_size + S_slm_size
            + dSt_slm_size + ugemm_slm_size];

    const size_t k_offset = KEY_BATCH(b1, b0_kv);
    const size_t v_offset = VAL_BATCH(b1, b0_kv);
    const size_t q_offset = QRY_BATCH(b1, b0);
    const size_t a_offset = DST_BATCH(b1, b0);
    const size_t da_offset = DA_BATCH(b1, b0);

    const size_t dk_offset = DK_BATCH(b1, b0_kv);
    const size_t dv_offset = DV_BATCH(b1, b0_kv);
    const size_t dq_offset = DQ_BATCH(b1, b0);

    /* Locate K/Q/V/A matrices within batch */
    K += k_offset;
    Q += q_offset;
    V += v_offset;
    A += a_offset;

    dK += dk_offset;
    dQ += dq_offset;
    dV += dv_offset;
    dA += da_offset;

#if WITH_DS
    dS += b1 * (DST_D1 * q * k) + b0 * (q * k);
#endif

#if WITH_ATTN_MASK
    msk += MSK_BATCH(b1 % MSK_D0, b0 % MSK_D1);
    int mask_aligned = (((size_t)msk) % 4) == 0;
    bool block_msk = (b1 < MSK_D0 - ceil((float)ugemm_kq_wg_tile_m / MSK_S2))
            && mask_aligned;
#endif

#if K_IN_SLM
    if (qdiag0 < q0end) {
        /* Load K tile, destined for SLM */
        k_tile_type K_tile;
        tile_fill(K_tile, CONVERT_TILE_FMA_T(0.f));

        tile_load_k(&K_tile, K, k, d, ldk, wg_i0, sg_ij, remainder_k);

        /* Store K tile to SLM */
        tile_store_k_slm(&K_tile, K_slm, sg_ij);
    }
#endif

    /* Load scale */
    float scale = 1.f;
    float iscale = 1.f;
    if (qdiag0 < q0end) {
#if WITH_ATTN_SCALE
#if WITH_HOST_SCALE
#if INVERT_SCALE
        iscale = scalar_scale;
        scale = inv_scalar_scale;
#else
        scale = scalar_scale;
        iscale = inv_scalar_scale;
#endif
#else
#if INVERT_SCALE
        iscale = into_float(*scale_ptr);
        scale = native_recip(iscale);
#else
        scale = into_float(*scale_ptr);
        iscale = native_recip(scale);
#endif
#endif
#endif
    }

#if WITH_DROPOUT
#if !DROPOUT_HOST_SCALARS
    long dropout_seed = dropout_seed_buf[0];
    long dropout_offset = use_dropout_offset ? dropout_offset_buf[0] : 0;
    float dropout_p = dropout_p_buf[0];
#endif
    uint dropout_threshold = get_dropout_threshold(dropout_p);
    float dropout_inv_q = (dropout_p != 1.f) ? 1.f / (1.f - dropout_p) : 0.f;
    const ulong dropout_batch_head_idx = (ulong)b1 * (ulong)DST_D1 + (ulong)b0;
    const ulong dropout_batch_head_base
            = dropout_batch_head_idx * (ulong)q * (ulong)k;
#endif

    /* Initialize dV, dK to zero */
#pragma unroll
    for (int i = get_local_id(0); i < ugemm_kq_wg_tile_m * D_MAX;
            i += get_local_size(0)) {
        dK_slm[i] = 0.f;
    }

    dv_tile_type dV_acc;
    tile_fill(dV_acc, 0.0f);

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    const int k0 = wg_i0;

    // make sure K_tile in SLM
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Main loop over k blocks */
    for (int q0 = qdiag0; q0 < q0end; q0 += ugemm_kq_wg_tile_n) {
        const bool first = (q0 == qdiag0);
        const int qnext = q0 + ugemm_kq_wg_tile_n;
        const bool last = (qnext >= q0end);

        int k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
        int q_nchunk = min(q0end - q0, ugemm_kq_wg_tile_n);
        /* Calculate S = (K^T) * Q */
#if DO_MM
#if K_IN_SLM
        s_tile_type S_tile = ugemm_kq(K_slm, D_MAX, Q + q0 * ldq, ldq, k_chunk,
                q_nchunk, d, 0, 0, 0, sg_i_kq, sg_j_kq, ugemm_slm);
#else
        s_tile_type S_tile = ugemm_kq(
#if TRANSPOSE_K
                K + k0 * ldk,
#else
                K + k0,
#endif
                ldk, Q + q0 * ldq, ldq, k_chunk, q_nchunk, d, 0, 0, 0, sg_i_kq,
                sg_j_kq, ugemm_slm);
#endif
#else
        s_tile_type S_tile;
#endif
        uint sg_i0_s2 = sg_i_kq * ugemm_kq_sg_tile_m + k0;
        uint sg_j0_s2 = sg_j_kq * ugemm_kq_sg_tile_n + q0;

        /* Apply attention mask */
#if WITH_ATTN_MASK
        mask_tile_type mask_tile;
#if BROADCAST_MASK_Q
        if (block_msk) {
            tile_load_block(&mask_tile, (const global msk_tile_data_t *)msk,
                    MSK_S2, 0, k0 + sg_i0_kq, 0);
        } else {
            tile_load(&mask_tile, (const global msk_tile_data_t *)msk, k, 1,
                    MSK_S2, k0 + sg_i0_kq, 0);
        }
#else
        tile_load(&mask_tile, (const global msk_tile_data_t *)msk, k, q, MSK_S2,
                k0 + sg_i0_kq, q0 + sg_j0_kq);
#endif

#define unscale(x) ((x) * iscale)
        mask_tile_type_float mask_tile_float;
        tile_copy_reblock(mask_tile, &mask_tile_float);
#if WITH_ATTN_SCALE
        tile_elementwise(mask_tile_float, unscale);
#endif
#undef unscale
#if BROADCAST_MASK_Q
        tile_vbroadcast_add(&S_tile, mask_tile_float);
#else
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif
#endif

        /* Apply q mask */
        if (remainder_q) {
#define gte_q(offset_k, offset_q) (offset_q >= q)
            tile_predicated_assignment(S_tile, k0 + sg_i0_kq, q0 + sg_j0_kq,
                    gte_q, -INFINITY, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                    ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                    ugemm_kq_c_type_nblock1);
#undef gte_q
        }

#if WITH_CAUSAL_MASK
#define less_than(offset_k, offset_q) (offset_q < offset_k)

        int col_offset = q0 + sg_j0_kq;
        if (q == 1) col_offset = 0;
        if (attn_mask_type == ATTN_MASK_BOTTOM_RIGHT) col_offset += k - q;

        /* Apply causal mask */
        const bool is_diag = (q0
                == qdiag0); // first iteration will be on diagonal, requiring partial masking
        if (is_diag) {
            tile_predicated_assignment(S_tile, k0 + sg_i0_kq, col_offset,
                    less_than, -INFINITY, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                    ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                    ugemm_kq_c_type_nblock1);
        }
#undef less_than
#endif

        s_sum_tile_type S_logsumexp_tile;
        tile_fill(S_logsumexp_tile, 0.f);
        tile_load(&S_logsumexp_tile, ws_logsumexp, q, 1, ugemm_kq_wg_tile_n,
                sg_j0_kq + q0, 0);
#define mulscale(x) (x * scale)
        tile_elementwise(S_tile, mulscale);
#undef mulscale
        tile_hbroadcast_sub(&S_tile, S_logsumexp_tile); //layout.N

        /* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x * 1.44269504089f)
        tile_elementwise(S_tile, scaled_exp);
#undef scaled_exp

        s_tile_type S_stash;
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            S_stash = S_tile;

#if WITH_DROPOUT
            /* P_dropped = P (dot) Z, used for dV GEMM */
            apply_dropout_s_tile(&S_tile, k0 + sg_i0_kq, q0 + sg_j0_kq, k, q,
                    dropout_batch_head_base, k, use_dropout_offset,
                    dropout_seed, dropout_offset, dropout_threshold,
                    dropout_inv_q);
#endif

            // Store softmax for ugemm_vs B-operand
#if USE_SYSTOLIC_UKERNEL
            s_tile_type_packed S_tile_packed;
            tile_copy_to_vec2_cvt(
                    S_tile, S_tile_packed, VEC_TYPE2, CONVERT_TILE_FMA_T);
            tile_store_t_sys_src2(S_tile_packed, (local uint *)S_slm,
                    ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_n / 2, sg_j0_kq / 2,
                    sg_i0_kq);
#else
            s_tile_type_reblock S_tile_reblock;
            tile_copy_reblock(S_tile, &S_tile_reblock);
            tile_store_packed_src1(S_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
                    ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        {
#if DO_MM
            dv_tile_type dV_tile1;
            dV_tile1 = ugemm_vs(dA + q0 * ldda, ldda, S_slm, ugemm_kq_wg_tile_n,
                    d, k_chunk, q_nchunk, 0, 0, 0, sg_i_vs, sg_j_vs, ugemm_slm);
#else
            dv_tile_type dV_tile1;
#endif
            uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
            uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

            // accumulate dv tile to slm
            if (sg_ij < sg_per_wg_BcD) {
                tile_binary(dV_acc, dV_tile1, binary_add);
            }
        }

#if DO_MM
        p_tile_type dP_tile
                = ugemm_vtdA(V + k0 * ldv, ldv, dA + q0 * ldda, ldda, k_chunk,
                        q_nchunk, d, 0, 0, 0, sg_i_kq, sg_j_kq, ugemm_slm);
#else
        p_tile_type dP_tile;
#endif

#if WITH_DROPOUT
        /* Backprop through dropout Jacobian: dP = dP_raw * Z / q. */
        apply_dropout_dP_tile(&dP_tile, k0 + sg_i0_kq, q0 + sg_j0_kq, k, q,
                dropout_batch_head_base, k, use_dropout_offset, dropout_seed,
                dropout_offset, dropout_threshold, dropout_inv_q);
#endif

        p_sum_tile_type D_i;
        tile_fill(D_i, 0.0f);
        tile_load(&D_i, Di, q0end, 1, q0end, q0 + sg_j0_kq, 0);
        tile_hbroadcast_sub(&dP_tile,
                D_i); // needs output to be transposed from vtdA layout.C = N

        {
            p_tile_type S2_tile;
            tile_copy_reblock(S_stash, &S2_tile);
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

#define binary_mul_scale(x, y) ((x) * (y) * scale)
            tile_binary(dP_tile, S2_tile, binary_mul_scale);
        }

        if (remainder_k) {
#define gte_k(offset_k, offset_q) (offset_k >= k)
            tile_predicated_assignment(S_tile, k0 + sg_i0_kq, q0 + sg_j0_kq,
                    gte_k, 0, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                    ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                    ugemm_kq_c_type_nblock1);
#undef gte_k
        }

        {
            p_tile_type_reblock P_tile_reblock;
            tile_copy_reblock(dP_tile, &P_tile_reblock);
#if WITH_DS
            tile_store(P_tile_reblock, dS, k_chunk, q_nchunk, k, k0 + sg_i0_kq,
                    q0 + sg_j0_kq);
#endif

            intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#if USE_SYSTOLIC_UKERNEL
            // softmax no longer needed, use slm to cache dS
            tile_store_sys_src22(P_tile_reblock, (local FMA_TYPE *)dSt_slm,
                    ugemm_ktq_sg_tile_n, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n,
                    sg_i0_kq, sg_j0_kq);
            p_tile_type_packed dP_tile_packed;
            tile_copy_to_vec2_cvt(
                    dP_tile, dP_tile_packed, VEC_TYPE2, CONVERT_TILE_FMA_T);
            tile_store_sys_src1(dP_tile_packed, (local uint *)S_slm,
                    SUBGROUP_SIZE, ugemm_kq_wg_tile_n / 2, ugemm_kq_wg_tile_m,
                    ugemm_kq_wg_tile_n / 2, sg_i0_kq, sg_j0_kq / 2);
#else
            // Store dS to S_slm for ugemm_qdSt
            tile_store_packed_src1(P_tile_reblock, S_slm, ugemm_qdSt_sg_tile_m,
                    ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        {
#if DO_MM
            a_tile_type dK_tile1;
            dK_tile1 = ugemm_qdSt(S_slm, ugemm_kq_wg_tile_n, Q + q0 * ldq, ldq,
                    k_chunk, d, q_nchunk, 0, 0, 0, sg_i_qdSt, sg_j_qdSt,
                    ugemm_slm); // dS^t * Q -> Bc x d
#else
            a_tile_type dK_tile1;
#endif
            uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
            uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

            // dk slm tile
            if (sg_ij < sg_per_wg_BcD) {
#if TRANSPOSE_K
                tile_slm_add_t(dK_tile1, dK_slm, D_MAX, sg_i0_dk, sg_j0_dk);
#else
                tile_slm_add(dK_tile1, dK_slm, ugemm_kq_wg_tile_m, sg_i0_dk,
                        sg_j0_dk);
#endif
            }
        }

#if !USE_SYSTOLIC_UKERNEL
        // re-read dS from S_slm and re-store transposed for ugemm_ktq
        {
            p_tile_type_reblock dS_reblock;
            tile_load_packed_src1(&dS_reblock, S_slm, ugemm_qdSt_sg_tile_m,
                    ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
            barrier(CLK_LOCAL_MEM_FENCE);
            tile_store_t_packed_src1(dS_reblock, S_slm, ugemm_ktq_sg_tile_n,
                    ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        local QRY_DATA_T *dSt_slm = S_slm;
#endif

        {
            ktq_tile_type dQ_tile;
#if DO_MM
            dQ_tile = ugemm_ktq(
#if TRANSPOSE_K
                    K + k0 * ldk,
#else
                    K + k0,
#endif
                    ldk, dSt_slm, ugemm_kq_wg_tile_m, d, q_nchunk, k_chunk, 0,
                    0, 0, sg_i_ktq, sg_j_ktq, ugemm_slm);
#endif
            uint sg_i0_dq = sg_i_ktq * ugemm_ktq_sg_tile_m;
            uint sg_j0_dq = sg_j_ktq * ugemm_ktq_sg_tile_n + q0;

#if DIRECT_DQ
            dq_tile_type_dst dQ_tile_dst;
            tile_copy_reblock(dQ_tile, &dQ_tile_dst);
            if (sg_ij < sg_per_wg_BrD)
                tile_store(dQ_tile_dst, (global dst_tile_data_t *)dQ, d, q,
                        lddq, sg_i0_dq, sg_j0_dq);
#else
            if (sg_ij < sg_per_wg_BrD)
                tile_atomic_add(dQ_tile, dQ, d, q, lddq, sg_i0_dq, sg_j0_dq);
#endif
        }
    }

    //////// update dV
    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

    // ensure all loops done writing to SLM
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg_ij < sg_per_wg_BcD) {
        tile_store_dV(&dV_acc, dV, d, k, lddv, sg_i0_vs, wg_i0 + sg_j0_vs,
                remainder_k);
    }
    // /update dV

    //////// update dK
#if TRANSPOSE_K
    // transposed dK_slm (D*Bc) matches dV tile layout
    dv_tile_type dK_tile_t;

    if (sg_ij < sg_per_wg_BcD) {
        tile_load(&dK_tile_t, dK_slm, D_MAX, ugemm_kq_wg_tile_m, D_MAX,
                sg_i0_vs, sg_j0_vs);
        tile_store_dK_t(&dK_tile_t, dK, d, k, lddk, sg_i0_vs, wg_i0 + sg_j0_vs,
                remainder_k);
    }
#else
    // non-transposed dK_slm uses qdSt layout (Bc*D) and indexing
    uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
    uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

    a_tile_type dK_tile_slm;
    int wg_k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
    if (sg_ij < sg_per_wg_BcD) {
        tile_load(&dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, D_MAX,
                ugemm_kq_wg_tile_m, sg_i0_dk, sg_j0_dk);
        tile_store_dK(&dK_tile_slm, dK + wg_i0, wg_k_chunk, d, lddk, sg_i0_dk,
                sg_j0_dk);
    }
#endif
    // /update dK
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
preprocess_Di(global float *Di, const global DST_DATA_T *A,
        const global DST_DATA_T *dA, int d, int k, int q, BWD_QRY_OFFSETS,
        BWD_DST_OFFSETS, BWD_DA_OFFSETS) {

    uint lda = DST_S2;
    uint ldda = DA_S2;
    uint ldq = QRY_S2;

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint b0, b1;
    b0 = get_group_id(1);
    b1 = get_group_id(2);

    const uint preprocess_batch = b1 * (DST_D1 * q) + b0 * q;

    const size_t a_offset = DST_BATCH(b1, b0);
    const size_t da_offset = DA_BATCH(b1, b0);

    /* Locate A/dA matrices within batch */
    A += a_offset;
    dA += da_offset;

    Di += preprocess_batch;

    uint wg_q = get_group_id(0);
    uint wg_j0 = wg_q * ugemm_kq_wg_tile_n;

#define Di_slm_size (ugemm_kq_wg_tile_n * sizeof(float))
    local char slm[Di_slm_size];

    local float *Di_slm = (local float *)&slm[0];

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    uint q0_copy = q_tile_sg_n * sg_ij;

    if (q > 0) {
        // D_i calculation
#if QRY_DT_F32
        dq_tile_type dA_tile, A_tile;
        tile_fill(A_tile, 0.f);
        tile_fill(dA_tile, 0.f);
        tile_load(&dA_tile, (global FMA_TYPE *)dA, d, q, ldda, 0,
                wg_j0 + q0_copy);
        tile_load(&A_tile, (global FMA_TYPE *)A, d, q, lda, 0, wg_j0 + q0_copy);
#else
        dq_tile_type dA_tile, A_tile;
        q_tile_type dA_tile_reblock, A_tile_reblock; // load native type
        tile_fill(A_tile_reblock, CONVERT_TILE_FMA_T(0.f));
        tile_fill(dA_tile_reblock, CONVERT_TILE_FMA_T(0.f));

        tile_load(&dA_tile_reblock, (global FMA_TYPE *)dA, d, q, ldda, 0,
                wg_j0 + q0_copy);
        tile_load(&A_tile_reblock, (global FMA_TYPE *)A, d, q, lda, 0,
                wg_j0 + q0_copy);

        // convert to float for calculation
        tile_copy_reblock(dA_tile_reblock, &dA_tile);
        tile_copy_reblock(A_tile_reblock, &A_tile);
#endif

#define binary_mul(x, y) ((x) * (y))
        tile_binary(A_tile, dA_tile, binary_mul);

        // reduce tile across D_MAX
        for (int j = 0; j < q_tile_sg_n; j++) {
            float r = 0.f;
            for (int i0 = 0; i0 < D_MAX; i0 += SUBGROUP_SIZE) {
                r += sub_group_reduce_add(
                        tile_access(A_tile, i0, j, SUBGROUP_SIZE, D_MAX, 1, 1));
            }
            Di_slm[j + q0_copy] = r;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = get_local_id(0); i < ugemm_kq_wg_tile_n;
                i += get_local_size(0)) {
            if (get_local_id(1) == 0 && (wg_j0 + i) < q) {
                Di[wg_j0 + i] = Di_slm[i];
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
postprocess_dQ(global DST_DATA_T *dst, global const float *src, int nelems,
        DQ_STRIDES, FULL_QRY_OFFSETS) {
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);

    const size_t src_offset = DQ_BATCH(b1, b0);
    const size_t dst_offset = QRY_BATCH(b1, b0);

    /* Locate dQ matrices within batch */
    src += src_offset;
    dst += dst_offset;
    size_t idx = get_global_id(0);
    if (idx >= (size_t)nelems) return;

    const bool dense = (DQ_S3 == 1) && (QRY_S3 == 1) && (DQ_S2 == QRY_D3)
            && (QRY_S2 == QRY_D3);
    if (dense) {
        dst[idx] = TO_DATA_T(src[idx]);
    } else {
        uint lin = (uint)idx;
        uint ncols = (uint)QRY_D3;
        uint row = lin / ncols;
        uint col = lin - row * ncols;
        size_t src_idx = (size_t)row * DQ_S2 + (size_t)col * DQ_S3;
        size_t dst_idx = (size_t)row * QRY_S2 + (size_t)col * QRY_S3;
        dst[dst_idx] = TO_DATA_T(src[src_idx]);
    }
}
