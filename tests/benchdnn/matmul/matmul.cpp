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
#include <random>
#include <set>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

// Helper to create grouped memory descriptor
//
// Supported configurations:
//  2Dx3D:
//   --grouped=0:G:size_0+...+size_{G-1} total_MxK:GxKxN
//   src [total_M, K] (dim 0 variable)
//   wei is dense 3D [G, K, N]
//   dst [total_M, N] (dim 0 variable)
//
//  2Dx2D:
//   --grouped=1:G:size_0+...+size_{G-1} Mxtotal_K:total_KxN
//   src [M, total_K] (dim 1 variable, col-major implicit)
//   wei [total_K, N] (dim 0 variable, row-major)
//   dst is dense 3D [G, M, N]
static dnnl_status_t create_grouped_md(
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md, const prb_t *prb,
        data_kind_t kind, dnnl_data_type_t dt) {
    int arg = (kind == SRC) ? DNNL_ARG_SRC
            : (kind == WEI) ? DNNL_ARG_WEIGHTS
            : (kind == DST) ? DNNL_ARG_DST
                            : DNNL_ARG_UNDEF;
    if (arg == DNNL_ARG_UNDEF) return dnnl_invalid_arguments;
    if (!prb->sparse_options.is_grouped(arg)) return dnnl_invalid_arguments;

    const int variable_dim_idx = prb->sparse_options.get_variable_dim_idx(arg);
    const int64_t group_count = prb->sparse_options.get_group_count();

    dnnl_dims_t dims_2d;
    // 2Dx3D: src [m, k], wei dense, dst [m, n]
    // 2Dx2D: src [m, k], wei [k, n], dst dense
    if (arg == DNNL_ARG_SRC) {
        dims_2d[0] = prb->m;
        dims_2d[1] = prb->k;
    } else if (arg == DNNL_ARG_WEIGHTS) {
        dims_2d[0] = prb->k;
        dims_2d[1] = prb->n;
    } else { // DNNL_ARG_DST
        dims_2d[0] = prb->m;
        dims_2d[1] = prb->n;
    }

    return dnn_mem_t::init_grouped_md(
            md, 2, dims_2d, dt, variable_dim_idx, group_count, dnnl_s32);
}

dims_t get_runtime_dims(const dims_t &dims, const dims_mask_t &mask) {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? DNNL_RUNTIME_DIM_VAL : dims[i];
    }
    return runtime_dims;
}

// TODO: Generalize md creation for sparse data when other primitives
// start supporting it.
dnnl_status_t create_md(benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> &md,
        const prb_t *prb, data_kind_t kind,
        dnnl_data_type_t dt = dnnl_data_type_undef) {
    if (kind == SRC) {
        if (dt == dnnl_data_type_undef) dt = prb->src_dt();
        const auto &src_rt_dims = get_runtime_dims(
                prb->src_dims(), prb->src_runtime_dim_mask());
        auto src_encoding = prb->sparse_options.get_encoding(DNNL_ARG_SRC);
        auto src_sparsity = prb->sparse_options.get_sparsity(DNNL_ARG_SRC);

        if (prb->sparse_options.is_grouped(DNNL_ARG_SRC))
            return create_grouped_md(md, prb, SRC, dt);
        if (src_encoding != dnnl_sparse_encoding_undef) {
            const dnnl_dim_t nnz
                    = std::max(prb->m * prb->k * (1.0f - src_sparsity), 1.0f);
            switch (src_encoding) {
                case dnnl_csr:
                    return dnn_mem_t::init_csr_md(md, prb->ndims,
                            src_rt_dims.data(), dt, nnz, dnnl_s32, dnnl_s32);
                    break;
                case dnnl_coo:
                    return dnn_mem_t::init_coo_md(md, prb->ndims,
                            src_rt_dims.data(), dt, nnz, dnnl_s32);
                    break;
                default:
                    assert(!"unsupported encoding");
                    return dnnl_invalid_arguments;
            }
        }
        return dnn_mem_t::init_md(md, prb->ndims, src_rt_dims.data(), dt,
                prb->stag, prb->strides[STRIDES_SRC]);
    }

    if (kind == WEI) {
        if (dt == dnnl_data_type_undef) dt = prb->wei_dt();
        const auto &weights_rt_dims = get_runtime_dims(
                prb->weights_dims(), prb->weights_runtime_dim_mask());
        auto wei_encoding = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);
        auto wei_sparsity = prb->sparse_options.get_sparsity(DNNL_ARG_WEIGHTS);

        if (prb->sparse_options.is_grouped(DNNL_ARG_WEIGHTS))
            return create_grouped_md(md, prb, WEI, dt);
        if (wei_encoding != dnnl_sparse_encoding_undef) {
            const dnnl_dim_t nnz
                    = std::max(prb->k * prb->n * (1.0f - wei_sparsity), 1.0f);
            switch (wei_encoding) {
                case dnnl_csr:
                    return dnn_mem_t::init_csr_md(md, prb->ndims,
                            weights_rt_dims.data(), dt, nnz, dnnl_s32,
                            dnnl_s32);
                case dnnl_coo:
                    return dnn_mem_t::init_coo_md(md, prb->ndims,
                            weights_rt_dims.data(), dt, nnz, dnnl_s32);
                case dnnl_packed:
                    return dnn_mem_t::init_sparse_packed_md(
                            md, prb->ndims, weights_rt_dims.data(), dt, nnz);
                    break;
                default:
                    assert(!"unsupported encoding");
                    return dnnl_invalid_arguments;
            }
        }
        // for grouped matmul, prb->ndims is not equal to the actual number
        // of dims in weights_rt_dims, so use weights_rt_dims.size() instead
        return dnn_mem_t::init_md(md, (int)weights_rt_dims.size(),
                weights_rt_dims.data(), dt, prb->wtag,
                prb->strides[STRIDES_WEI]);
    }

    if (kind == DST) {
        if (dt == dnnl_data_type_undef) dt = prb->dst_dt();
        const auto &dst_rt_dims
                = get_runtime_dims(prb->dst_dims, prb->dst_runtime_dim_mask());

        // Special grouped matmul handling for dst
        if (prb->sparse_options.is_grouped(DNNL_ARG_DST)) {
            // For 2Dx3D, dst is 2D grouped [total_M, N]
            return create_grouped_md(md, prb, DST, dt);
        } else if (prb->sparse_options.is_2dby2d()) {
            // For 2Dx2D, dst is dense 3D [G, M, N]
            const int64_t group_count = prb->sparse_options.get_group_count();
            dnnl_dims_t dims_3d = {group_count, prb->m, prb->n};
            return dnn_mem_t::init_md(
                    md, 3, dims_3d, dt, prb->dtag, prb->strides[STRIDES_DST]);
        }

        return dnn_mem_t::init_md(md, prb->ndims, dst_rt_dims.data(), dt,
                prb->dtag, prb->strides[STRIDES_DST]);
    }
    return dnnl_invalid_arguments;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(create_md(
            src_d, prb, SRC, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> wei_d;
    DNN_SAFE_STATUS(create_md(
            wei_d, prb, WEI, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
    DNN_SAFE_STATUS(create_md(
            dst_d, prb, DST, force_f32_dt ? dnnl_f32 : dnnl_data_type_undef));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bia_d {};
    if (prb->bia_dt != dnnl_data_type_undef) {
        auto bia_dims = get_runtime_dims(
                prb->bia_dims(), prb->bias_runtime_dim_mask());

        if (prb->sparse_options.is_grouped(DNNL_ARG_SRC)
                && prb->sparse_options.is_grouped(DNNL_ARG_DST)) {
            // verify_grouped_input() enforces bia_mask=2 (N-only)
            const int64_t group_count = prb->sparse_options.get_group_count();
            const dnnl_dim_t grouped_bias_dims[2] = {group_count, prb->n};
            DNN_SAFE_STATUS(dnn_mem_t::init_md(bia_d, 2, grouped_bias_dims,
                    force_f32_dt ? dnnl_f32 : prb->bia_dt, tag::abx));
        } else {
            DNN_SAFE_STATUS(dnn_mem_t::init_md(bia_d, prb->ndims,
                    bia_dims.data(), force_f32_dt ? dnnl_f32 : prb->bia_dt,
                    prb->dst_runtime_dim_mask() != 0 ? tag::abx : tag::any));
        }
    }

    attr_args_t attr_args;
    DNN_SAFE_STATUS(attr_args.prepare_post_ops_mds(prb->attr, prb->ndims,
            prb->dst_dims.data(), dnnl_undefined_primitive,
            &prb->sparse_options));

    const auto overload_quant_mask = [&](policy_t policy, int arg) {
        // Overload PER_OC/PER_OCIC mask definition for batched cases.
        if (policy == policy_t::PER_OC || policy == policy_t::PER_OCIC) {
            int mask = 1 << (prb->ndims - 1);
            if (policy == policy_t::PER_OCIC) mask += 1 << (prb->ndims - 2);
            attr_args.prepare_quant(prb->attr, arg, mask);
        }
    };

    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_SRC).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_WEIGHTS).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    overload_quant_mask(prb->attr.scales.get(DNNL_ARG_DST).policy,
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    overload_quant_mask(prb->attr.zero_points.get(DNNL_ARG_SRC).policy,
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    overload_quant_mask(prb->attr.zero_points.get(DNNL_ARG_WEIGHTS).policy,
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    overload_quant_mask(
            prb->attr.precomputed_reductions.get(DNNL_ARG_SRC).policy,
            DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_matmul_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine,
            init_pd_args.src_md ? init_pd_args.src_md : src_d, wei_d, bia_d,
            dst_d, dnnl_attr)));

    return dnnl_success;
}

int init_prim_ref(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const prb_t *prb, res_t *res) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && fast_ref)) return OK;
    // Create prim_ref if only original prim was successfully created.
    if (res->state != INITIALIZED) return OK;

    // f32 cases should go through reference no matter what.
    if (is_cpu() && (prb->src_dt() == dnnl_f32 && prb->wei_dt() == dnnl_f32))
        return OK;

    if (!prb->sparse_options.is_encoding_def(DNNL_ARG_SRC)
            || !prb->sparse_options.is_encoding_def(DNNL_ARG_WEIGHTS))
        return OK;

    std::vector<std::vector<dnnl_data_type_t>> prim_ref_dt {
            prb->dt, {dnnl_f32}};
    // If there's no bias, undef data type should be used for prim_ref as well.
    dnnl_data_type_t cpu_bia_dt
            = prb->bia_dt == dnnl_data_type_undef ? prb->bia_dt : dnnl_f32;
    std::vector<dnnl_data_type_t> prim_ref_bia_dt {prb->bia_dt, cpu_bia_dt};
    if (is_cpu()) {
        prim_ref_dt.erase(prim_ref_dt.begin());
        prim_ref_bia_dt.erase(prim_ref_bia_dt.begin());
    }

    for_(const auto &prim_ref_dt_i : prim_ref_dt)
    for (const auto &prim_ref_bia_dt_i : prim_ref_bia_dt) {
        auto cpu_attr = prb->attr;
        update_cpu_ref_attrs(cpu_attr, prim_ref_dt_i.back());

        // Create a new copy of prb to avoid potentially corrupting the test by
        // modifying prb in place.
        prb_t prb_cpu {*prb, prim_ref_dt_i, tag::any, tag::any, tag::any,
                {vdims_t(STRIDES_SIZE)}, prim_ref_bia_dt_i, prb->bia_mask,
                {0, 0, 0}, sparse_options_t(), cpu_attr, prb->ctx_init,
                prb->ctx_exe, prb->impl_filter};

        auto st = init_prim_ref_common(prim_ref, &prb_cpu, res, init_pd);
        if (st == OK) return OK;
    }

    prim_ref.reset(nullptr);
    return OK;
}

// The main idea is to generate values and metadata directly without generating
// the dense matrix to avoid excessive memory consumption for large problem
// sizes.
int fill_sparse_data(data_kind_t kind, const prb_t *prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res, dnnl_sparse_encoding_t encoding) {
    if (query_md_num_handles(mem_dt.md_) != 3) return FAIL;

    if (kind != SRC && kind != WEI) return FAIL;

    const int64_t dim0 = kind == SRC ? prb->m : prb->k;
    const int64_t dim1 = kind == SRC ? prb->k : prb->n;

    // Coefficient for distribution of nnz per row.
    const int64_t coef = 3;
    const int64_t nnz = query_md_nnz(mem_fp.md_);
    const int64_t avg_nnz_per_row = nnz / dim0;

    int64_t distributed_nnz_cnt = 0;

    std::uniform_int_distribution<> pointers_gen(0, avg_nnz_per_row * coef);
    std::minstd_rand pointers_seed;

    // Distribute nnz across all rows.
    std::vector<int64_t> distributed_nnz(dim0);
    for (int64_t i = 0; i < dim0; i++) {
        int64_t nnz_per_row = std::min(pointers_gen(pointers_seed), (int)dim1);
        nnz_per_row = std::min(nnz_per_row, (nnz - distributed_nnz_cnt));
        distributed_nnz[i] = nnz_per_row;
        distributed_nnz_cnt += nnz_per_row;
    }

    // Distribute remaining nnz.
    int64_t remaining_nnz_cnt = nnz - distributed_nnz_cnt;
    while (remaining_nnz_cnt > 0) {
        const int64_t remaining_nnz_per_row
                = std::max((int)(remaining_nnz_cnt / dim0), 1);
        for (int64_t i = 0; i < dim0; i++) {
            int64_t nnz_to_add = std::min(
                    remaining_nnz_per_row, (dim1 - distributed_nnz[i]));
            nnz_to_add = std::min(nnz_to_add, remaining_nnz_cnt);
            distributed_nnz[i] += nnz_to_add;
            remaining_nnz_cnt -= nnz_to_add;
            distributed_nnz_cnt += nnz_to_add;

            if (remaining_nnz_cnt == 0) break;
        }
    }

    if (remaining_nnz_cnt != 0) return FAIL;

    int values_idx = 0;
    int indices_idx = 1;
    const int pointers_idx = 2;

    if (encoding == dnnl_csr) {
        // fill pointers for CSR encoding
        mem_fp.set_elem(0, 0, pointers_idx);
        mem_dt.set_elem(0, 0, pointers_idx);

        for (int64_t i = 0; i < dim0; i++) {
            const int32_t pointer
                    = mem_fp.get_elem(i, pointers_idx) + distributed_nnz[i];
            mem_fp.set_elem(i + 1, pointer, pointers_idx);
            mem_dt.set_elem(i + 1, pointer, pointers_idx);
        }
    } else if (encoding == dnnl_coo) {
        values_idx = 0;
        indices_idx = 2;
        const int row_indices_idx = 1;

        // fill row indices for COO encoding
        int32_t row_ptr = 0;

        for (int64_t i = 0; i < dim0; i++) {
            for (int32_t j = 0; j < distributed_nnz[i]; j++) {
                mem_fp.set_elem(row_ptr + j, i, row_indices_idx);
                mem_dt.set_elem(row_ptr + j, i, row_indices_idx);
            }
            row_ptr = row_ptr + distributed_nnz[i];
        }
    }

    std::uniform_int_distribution<> indices_gen(0, dim1 - 1);
    std::minstd_rand indices_seed;

    // Generate indices.
    std::vector<int32_t> indices;
    std::set<int32_t> indices_set;
    for (int64_t i = 0; i < dim0; i++) {
        while ((int64_t)indices_set.size() != distributed_nnz[i]) {
            int index = indices_gen(indices_seed);
            if (indices_set.count(index)) continue;
            indices_set.insert(index);
        }
        indices.insert(indices.end(), indices_set.begin(), indices_set.end());
        indices_set.clear();
    }

    benchdnn_parallel_nd((int)indices.size(), [&](int64_t i) {
        const int32_t index = indices[i];
        mem_fp.set_elem(i, index, indices_idx);
        mem_dt.set_elem(i, index, indices_idx);
    });

    // Don't fill data for `no_ref_memory` as it will be filled by benchdnn.
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    // Generate values.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nnz, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nnz);

        std::uniform_int_distribution<> values_gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::minstd_rand values_seed(kind * nnz + idx_start + 1);
        values_seed.discard(1);

        for (int64_t i = idx_start; i < idx_end; i++) {
            float val = values_gen(values_seed);
            mem_fp.set_elem(i,
                    round_to_nearest_representable(cfg.get_dt(kind), val),
                    values_idx);
            mem_dt.set_elem(i,
                    round_to_nearest_representable(cfg.get_dt(kind), val),
                    values_idx);
        }
    });

    return OK;
}

// Filling for mem_fp that takes into account density and zero points
static void fill_dense_fp_values(data_kind_t kind, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_fp) {
    const int64_t nelems = mem_fp.nelems();

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->k;
    const auto density = cfg.get_density(density_args);

    const auto &e_zp_src = prb->attr.zero_points.get(DNNL_ARG_SRC);
    const bool has_src_zp = !e_zp_src.is_def();
    const int src_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_SRC, dnnl_matmul, prb->ndims);
    // Apply src_zp for source tensor only.
    int src_zp = kind == SRC && has_src_zp && src_zp_mask == 0 ? e_zp_src.value
                                                               : 0;

    const auto &e_zp_wei = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS);
    const bool has_wei_zp = !e_zp_wei.is_def();
    const int wei_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_matmul, prb->ndims);
    // Apply wei_zp for weights tensor only.
    int wei_zp = kind == WEI && has_wei_zp && wei_zp_mask == 0 ? e_zp_wei.value
                                                               : 0;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            val += src_zp + wei_zp; // Add zp so that it will be subtracted.
            mem_fp.set_f32_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            if (!is_one) {
                mem_fp.set_f32_elem(idx, 0.f);
                continue;
            }
            float val = gen(int_seed);
            val += src_zp + wei_zp; // Add zp so that it will be subtracted.
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });
}

// Fill offsets buffer for grouped memory with cumulative sums
static int fill_grouped_offsets(
        dnn_mem_t &mem, const sparse_options_t &sparse_options) {
    const int64_t group_count = sparse_options.get_group_count();
    const auto &group_sizes = sparse_options.get_group_sizes(DNNL_ARG_SRC);
    // Note, that we usually rely on `set_elem` for setting buffer elements, but
    // it operates with floats, while offsets are int32
    int32_t *offsets = mem.get_mapped_pointer<int32_t>(
            sparse_options_t::grouped_data_t::grouped_offsets_idx);

    int64_t cumulative = 0;
    for (int64_t g = 0; g < group_count; g++) {
        if (cumulative > INT32_MAX - group_sizes[g]) {
            BENCHDNN_PRINT(0,
                    "Error: cumulative offset would exceed INT32_MAX at "
                    "group %lld\n",
                    (long long)g);
            return FAIL;
        }
        cumulative += group_sizes[g];
        offsets[g] = static_cast<int32_t>(cumulative);
    }
    return OK;
}

// Fill grouped values for SRC (2Dx3D and 2Dx2D variants) or WEI (2Dx2D variant
// only)
static int fill_grouped_data(data_kind_t kind, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    if (kind != SRC && kind != WEI) {
        BENCHDNN_PRINT(0,
                "Error: grouped filling only supports SRC or WEI, got "
                "kind=%d\n",
                (int)kind);
        return FAIL;
    }

    const int nhandles = query_md_num_handles(mem_dt.md_);
    if (nhandles != 2) {
        BENCHDNN_PRINT(0, "Error: grouped memory requires 2 handles, got %d\n",
                nhandles);
        return FAIL;
    }

    fill_dense_fp_values(kind, prb, cfg, mem_fp);

    // Ref memory is grouped with the same layout, so values are reordered as
    // flat buffers
    const dnnl_dims_t vals_dims = {mem_dt.nelems()};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> fp_vals_md;
    DNN_SAFE(dnn_mem_t::init_md(fp_vals_md, 1, vals_dims, dnnl_f32, tag::abx),
            WARN);
    auto fp_vals = dnn_mem_t::create_from_host_ptr(
            fp_vals_md, mem_fp.engine(), mem_fp.get_mapped_pointer<void>(0));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dt_vals_md;
    DNN_SAFE(
            dnn_mem_t::init_md(dt_vals_md, 1, vals_dims, mem_dt.dt(), tag::abx),
            WARN);
    auto dt_vals = dnn_mem_t::create_from_host_ptr(
            dt_vals_md, mem_fp.engine(), mem_dt.get_mapped_pointer<void>(0));
    SAFE(dt_vals.reorder(fp_vals, res), WARN);

    return OK;
}

int fill_data(data_kind_t kind, int exec_arg, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    bool is_sparse_packed = false;
    bool is_any_sparse = false;
    std::vector<bool> nnz_mask;
    const auto sparse_encoding = prb->sparse_options.get_encoding(kind);
    const bool is_sparse_csr_coo
            = sparse_encoding == dnnl_csr || sparse_encoding == dnnl_coo;
    is_sparse_packed = sparse_encoding == dnnl_packed;
    is_any_sparse = sparse_encoding != sparse_options_t::def_encoding;

    if (is_sparse_csr_coo) {
        return fill_sparse_data(
                kind, prb, mem_dt, mem_fp, res, sparse_encoding);
    }

    if (prb->sparse_options.is_grouped(exec_arg)) {
        SAFE(fill_grouped_data(kind, prb, cfg, mem_dt, mem_fp, res), WARN);
        return OK;
    }

    if (is_sparse_packed) {
        nnz_mask.resize(nelems, false);
        const dnnl_dim_t nnz = query_md_nnz(mem_dt.md_);
        assert(nnz > 0);
        for (int i = 0; i < nnz; i++)
            nnz_mask[i] = true;
        std::default_random_engine rng(nnz);
        std::shuffle(nnz_mask.begin(), nnz_mask.end(), rng);
    }

    // Refer to modes documentation for filling principles.
    // Note: sparse filling is more complex than a general one in a sense that
    // it requires metadata in addition to data. To have reasonable bitwise
    // validation for sparse, only data must be random and indices should remain
    // identical between runs. So far, simply don't support bitwise mode for
    // sparse problems. `CSR`/`COO` will utilize their `fill_sparse_data`
    // function, `packed` will fall back into a regular filling as it involves
    // `nnz_mask`.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && !is_any_sparse) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf) && !is_any_sparse) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    if (is_sparse_packed) {
        /* Do fixed partitioning to have same filling for any number of threads */
        const int64_t chunk_size = 64;
        const int64_t n_chunks = div_up(nelems, chunk_size);

        benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            std::minstd_rand int_seed(kind * nelems + idx_start + 1);
            int_seed.discard(1);
            std::uniform_int_distribution<> gen(
                    cfg.get_range_min(kind), cfg.get_range_max(kind));

            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                const bool is_one = nnz_mask[idx];
                if (!is_one) {
                    mem_fp.set_f32_elem(idx, 0.f);
                    continue;
                }
                float val = 0.f;
                while (val == 0.f)
                    val = gen(int_seed);
                mem_fp.set_f32_elem(idx,
                        round_to_nearest_representable(cfg.get_dt(kind), val));
            }
        });
    } else {
        fill_dense_fp_values(kind, prb, cfg, mem_fp);
    }

    SAFE(mem_dt.reorder(mem_fp, res, cfg.get_swapped_dt(kind)), WARN);

    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->wei_dt(), prb->bia_dt, prb->dst_dt()},
            prb->dir, res);
    skip_unimplemented_binary_po(prb->attr, res);

    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);
    if (wei_encoding == dnnl_packed) {
        // This support is defined on a single CPU platform, skip it
        // unconditionally.
        BENCHDNN_PRINTF(2, "%s",
                "[SKIP]: Weights argument doesn't support packed encoding");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    auto src_rt_mask = prb->src_runtime_dim_mask();
    auto wei_rt_mask = prb->weights_runtime_dim_mask();
    auto dst_rt_mask = prb->dst_runtime_dim_mask();

    // Memory layouts must be defined when some dimensions are unknown at pd
    // creation time.
    //
    // Note: this check must be removed from here, and the check should be
    // delegated to the library API checks, but since it doesn't articulate
    // what's the exact problem, keep it here until it does.
    //
    // Note: runtime_dims get initialized when prb is created which is past
    // input verification point, that's why this and the check below live here,
    // but not there.
    const auto &rt_dims_are_consistent
            = [](const dims_mask_t &rt_dims, const std::string &tag,
                      const dims_t &strides) {
        if (rt_dims.none()) return true;
        return tag != tag::any || !strides.empty();
    };

    if (!rt_dims_are_consistent(src_rt_mask, stag, strides[STRIDES_SRC])
            || !rt_dims_are_consistent(wei_rt_mask, wtag, strides[STRIDES_WEI])
            || !rt_dims_are_consistent(
                    dst_rt_mask, dtag, strides[STRIDES_DST])) {
        BENCHDNN_PRINTF(1, "%s",
                "[INVALID]: Runtime dimensions require user to specify a "
                "memory format for affected arguments. Consider specifying "
                "`--stag`, `--wtag`, and/or `--dtag` or correspondent "
                "`--strides` values.");
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    const int m_idx = prb->ndims - 2;
    const int k_idx_src = prb->ndims - 1;
    const int k_idx_wei = prb->ndims - 2;
    const int n_idx = prb->ndims - 1;
    if (src_rt_mask[m_idx] != dst_rt_mask[m_idx]
            || src_rt_mask[k_idx_src] != wei_rt_mask[k_idx_wei]
            || wei_rt_mask[n_idx] != dst_rt_mask[n_idx]) {
        BENCHDNN_PRINT(2,
                "[INVALID][%s:%d]: Runtime masks for `m`, `k`, and `n` "
                "dimensions must be consistent.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    if (prb->ndims > 2) {
        dims_mask_t batch_rt_mask;
        for (int i = 0; i < prb->ndims - 2; ++i)
            batch_rt_mask[i] = true;
        src_rt_mask &= batch_rt_mask;
        wei_rt_mask &= batch_rt_mask;
        dst_rt_mask &= batch_rt_mask;
        if (src_rt_mask != wei_rt_mask || src_rt_mask != dst_rt_mask) {
            BENCHDNN_PRINT(2,
                    "[INVALID][%s:%d]: Runtime masks for batch dimensions must "
                    "be consistent.\n",
                    __FILE__, __LINE__);
            res->state = SKIPPED;
            res->reason = reason_t::invalid;
            return;
        }
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
}

std::vector<int> prb_t::supported_exec_args(bool) const {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
    };
    return exec_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    // Both sparse and grouped functionality relies on indirect mnemory access.
    // While the data itself can be anything for `no_ref_memory` modifier,
    // metadata (indices/pointers for sparse; offsets for grouped) must be
    // valid before the library executes, otherwise a jump to a random memory
    // location outside of allocated bytes will happen.
    // Note, that `has_sparse_md` covers grouped mds as well: `dnnl_grouped`
    // is a member of `dnnl_sparse_encoding_t`.
    //
    // If there's a sparse memory, non-sparse memory and non-metadata handles
    // will not reach the filling, unless it's a `packed` encoding. In such case
    // the reference f32 counterpart must be mapped and allowed to reach the
    // reorder because metadata needed will be filled as a part of the reorder.
    const bool map_has_sparse_mem = has_sparse_md(mem_map);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
            && !map_has_sparse_mem)
        return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    const auto wei_encoding
            = prb->sparse_options.get_encoding(DNNL_ARG_WEIGHTS);

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Route grouped to the generic else branch below
        const bool is_sparse_src = exec_arg == DNNL_ARG_SRC
                && !prb->sparse_options.is_encoding_def(DNNL_ARG_SRC)
                && !prb->sparse_options.is_grouped(DNNL_ARG_SRC);
        const bool is_sparse_wei = exec_arg == DNNL_ARG_WEIGHTS
                && wei_encoding != dnnl_sparse_encoding_undef
                && !prb->sparse_options.is_grouped(DNNL_ARG_WEIGHTS);
        const bool is_sparse_dst = exec_arg == DNNL_ARG_DST
                && !prb->sparse_options.is_encoding_def(DNNL_ARG_DST)
                && !prb->sparse_options.is_grouped(DNNL_ARG_DST);
        const bool is_sparse = is_sparse_src || is_sparse_wei || is_sparse_dst;
        const bool is_sparse_wei_packed
                = is_sparse_wei && wei_encoding == dnnl_packed;

        // Offsets are read by the library, fill them in any mode
        if (has_grouped_encoding(mem.md_)) {
            SAFE(fill_grouped_offsets(mem, prb->sparse_options), WARN);
            if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory))
                continue;
        }

        // See the comment at the beginning of the function.
        if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
                && exec_arg != DNNL_ARG_HINT_MAX_GROUP_SIZE
#endif
                && !is_sparse)
            continue;

        if (is_sparse && !is_sparse_wei_packed) {
            if (is_sparse_src) {
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_fp_d;
                DNN_SAFE(create_md(src_fp_d, prb, SRC), WARN);
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(src_fp_d, ref_engine, /* prefill = */ false));
            }

            if (is_sparse_wei) {
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> wei_fp_d;
                DNN_SAFE(create_md(wei_fp_d, prb, WEI), WARN);
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(wei_fp_d, ref_engine, /* prefill = */ false));
            }
            if (is_sparse_dst) {
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_fp_d;
                DNN_SAFE(create_md(dst_fp_d, prb, DST), WARN);
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(dst_fp_d, ref_engine,
                                /* prefill = */ false));
            }
        } else {
            if (prb->sparse_options.is_grouped(exec_arg)) {
                // Grouped args (SRC, DST for 2Dx3D; SRC, WEI for 2Dx2D) use
                // grouped f32 ref memory with the same layout
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> grouped_fp_md;
                DNN_SAFE(create_md(grouped_fp_md, prb,
                                 exec_arg2data_kind(exec_arg), dnnl_f32),
                        WARN);
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(grouped_fp_md, ref_engine,
                                /* prefill = */ false));
                SAFE(fill_grouped_offsets(
                             ref_mem_map.at(exec_arg), prb->sparse_options),
                        WARN);
            } else if (exec_arg == DNNL_ARG_WEIGHTS) {
                const auto ndims = mem.ndims();
                const auto &dims = mem.dims();
                // Switch the format tag from "ab" to "ba" but to handle batched
                // cases, use strides instead.
                dnnl_dims_t strides {};
                dnnl_dim_t stride = 1;
                for (int d = ndims - 2; d >= 0; d--) {
                    strides[d] = stride * dims[d + 1];
                    stride = strides[d];
                }
                strides[ndims - 2] = 1;
                strides[ndims - 1] = dims[ndims - 2];
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(mem.md_, dnnl_f32, strides, ref_engine,
                                /* prefill = */ false));
            } else if (exec_arg != DNNL_ARG_SCRATCHPAD) {
                // Scratchpad memory relates to a primitive. If reference needs
                // it, use switch below to define a memory desc for it.
                ref_mem_map.emplace(exec_arg,
                        dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                                /* prefill = */ false));
            }
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        // See the comment at the beginning of the function.
        if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
                && is_sparse_wei_packed) {
            ref_mem.map();
        }

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST: {
                const auto &po = prb->attr.post_ops;
                const int sum_idx = po.find(attr_t::post_ops_t::SUM);
                if ((sum_idx >= 0) || po.has_inplace_binary()) {
                    SAFE(fill_data(DST, exec_arg, prb, cfg, mem, ref_mem, res),
                            WARN);
                    // Bitwise mode for sum and in-place binary post-ops
                    // requires a copy due to data for post-op will be
                    // overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem, res), WARN);
                    }
                }
            } break;
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
            case DNNL_ARG_HINT_MAX_GROUP_SIZE:
                // The hint is for the library to optimize execution,
                // it doesn't affect reference.
                mem.set_elem(0,
                        static_cast<float>(
                                prb->sparse_options.get_max_variable_dim()));
                break;
#endif
            case DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS:
                // Fill it separately down below.
                // TODO: introduce an order of processing arguments to avoid
                // post filling manipulations.
                break;
            default: {
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
            } break;
        }

        update_ref_mem_map_from_prim(prim_ref, mem, ref_mem_map, exec_arg,
                cfg.get_swapped_dt(exec_arg2data_kind(exec_arg)), res);

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    if (ref_mem_map.count(
                DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC)) {
        auto &mem = mem_map.at(
                DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC);
        auto &ref_mem = ref_mem_map.at(
                DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC);
        // TODO: will be handled by `init_ref_memory_args_default_case` once
        // memory argument dependency is resolved.
        if (fill_from_file(DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC,
                    mem, ref_mem, res))
            return OK;
        const auto &ref_mem_src = ref_mem_map.at(DNNL_ARG_SRC);
        const auto src_precomputed_reductions_gs
                = prb->attr.precomputed_reductions.get(DNNL_ARG_SRC).groups[1];

        // Reducing original `ref_src` by group size specified.
        //
        // Assumption that `ref_src` didn't change its `abx` format which can be
        // changed in `update_ref_mem_map_from_prim`.
        for (int64_t i = 0;
                i < ref_mem_src.nelems() / src_precomputed_reductions_gs; i++) {
            float val = 0;
            for (int64_t k = 0; k < src_precomputed_reductions_gs; k++) {
                const auto offset = i * src_precomputed_reductions_gs + k;
                const auto s = ref_mem_src.get_elem(offset);
                val += s;
            }
            ref_mem.set_elem(i, val);
        }

        SAFE(mem.reorder(ref_mem, res), WARN);
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(2); // regular + cpu_ref
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    // Use CPU prim as the reference in GPU testing to reduce testing time.
    SAFE(init_prim_ref(v_prim[1], prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        const auto &prim_ref = v_prim[1];
        if (prim_ref) {
            // Copy res to avoid save/restore state and reason.
            res_t res_copy = *res;
            SAFE(check_total_size(&res_copy, prim_ref), WARN);
            if (res_copy.state == SKIPPED) {
                v_prim[1].reset(nullptr);
                SAFE(check_total_size(res), WARN);
            } else {
                // Copy estimations back to original `res`.
                *res = res_copy;
            }
        } else {
            SAFE(check_total_size(res), WARN);
        }
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
        // Don't check caches for CPU prim as the reference.
    }
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds = {DST};
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];
    const auto &prim_ref = v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(mem_map, prb, prim, res), WARN);
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prim_ref),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, compute_ref,
            setup_cmp, res, prb->dir, prim_ref);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace matmul
