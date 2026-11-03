/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2023-2025 FUJITSU LIMITED
* Copyright 2024-2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

enum {
    decomposition_2x2 = 101,
    decomposition_3x1_3,
    decomposition_3x1_2,
    undefined,
};

impl::data_type_t get_accum_datatype(brgemm_desc_t *brg) {
    // this assert should check if 'init_kernel_datatype()' was previously
    // called.
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

status_t init_kernel_datatype(
        brgemm_desc_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    if (!(dt_a != data_type::undef && dt_b != data_type::undef))
        return status::unimplemented;
    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && utils::one_of(dt_b, data_type::u8, data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    brg->is_f32 = (dt_a == data_type::f32)
            && utils::one_of(dt_b, data_type::f32, data_type::bf16);
    brg->is_f16 = utils::one_of(data_type::f16, dt_a, dt_b);
    if (!(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16))
        return status::unimplemented;
    return status::success;
}

void init_common_conf(brgemm_desc_t *brg, brgemm_batch_kind_t type, float alpha,
        float beta, const brgemm_strides_t *strides) {
    brg->beta = beta;
    brg->alpha = alpha;
    brg->type = type;
    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->with_sum = false;
    brg->with_weights_scale_adjust = false;
    brg->sum_scale = 0;
    brg->sum_zp = 0;
    brg->with_scales = false;

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }
}

namespace brgemm_utils {

bool can_dispatch_uker(const brgemm_desc_t *brg) {
    return false;
}
void maybe_try_bf32(brgemm_desc_t *brg) {
    //
}

// bf16 (BFMMLA) and int8 (SMMLA / UMMLA / USMMLA) mmla are supported
status_t validate_mmla_compute(const brgemm_desc_t &brg) {
    using namespace data_type;
    const bool common_ok = !brg.is_dgmm && brg.is_row_major()
            && utils::one_of(brg.isa_impl, sve_128, sve_256);
    if (!common_ok) return status::unimplemented;

    if (brg.is_bf16)
        return brg.is_bf16_emu ? status::unimplemented : status::success;
    if (brg.is_int8) {
        const bool dt_ok = (brg.dt_a == s8 && brg.dt_b == s8)
                || (brg.dt_a == u8 && brg.dt_b == u8)
                || (brg.dt_a == u8 && brg.dt_b == s8);
        const bool comp_ok
                = !brg.req_s8s8_compensation && !brg.req_cal_comp_pads;
        // int8 mmla is currently tested on sve_256 only
        const bool isa_ok = brg.isa_impl == sve_256;
        return (dt_ok && comp_ok && isa_ok) ? status::success
                                            : status::unimplemented;
    }
    return status::unimplemented;
}

status_t validate_mmla_zero_points(const brgemm_desc_t &brg) {
    if (!brg.brgattr.use_mmla || !brg.is_int8) return status::success;
    // Weight zero points would need a per-row (A) sum inside the kernel.
    if (brg.zp_type_b != brgemm_broadcast_t::none) return status::unimplemented;
    // Rows hidden by virtual padding are fed as zeros, which is only correct
    // when the src zero point is zero. With a src zero point the caller must
    // materialize padding itself (e.g. zp-filled packed A) and disable vpad.
    const bool has_vpad
            = brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0;
    if (brg.zp_type_a != brgemm_broadcast_t::none
            && (has_vpad || brg.req_cal_comp_pads))
        return status::unimplemented;
    return status::success;
}

status_t set_isa_impl(brgemm_desc_t *brg) {
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) &&
                // maybe IMPLICATION(brg->isa_user != isa_undef,
                //  is_superset(brg->isa_user, isa)), but the API is not clear.
                one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_bf32 || brg->is_f16) {
        return status::unimplemented;
    } else if (brg->is_bf16 && !mayiuse_bf16()) {
        return status::unimplemented;
    } else if (brg->is_f32 || brg->is_bf16 || brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(sme), sme,
                is_isa_ok(sve_512), sve_512, is_isa_ok(sve_256), sve_256,
                is_isa_ok(sve_128), sve_128);
        return status::success;
    }
    return status::success;
}

void set_brg_vmm(brgemm_desc_t *brg) {

    brg->is_zmm = mayiuse(sve_512) && is_superset(brg->isa_impl, sve_512);
    brg->is_ymm = !brg->is_zmm && mayiuse(sve_256)
            && is_superset(brg->isa_impl, sve_256);
}

int calculate_ldb_params(brgemm_desc_t *brg, const int try_ld_block2) {
    brg->ld_block2 = try_ld_block2;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;

    if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);

    const int adj_ld_block2
            = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
    return nstl::max(1, adj_ld_block2);
}

int calculate_max_bcast_block(brgemm_desc_t *brg, const int adj_ld_block2) {

    constexpr int max_bcst_regs = 1;
    const bool req_compensation = brg->req_s8s8_compensation
            || brg->zp_type_a != brgemm_broadcast_t::none;
    const bool req_zp_a_comp_pads
            = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                      || brg->brgattr.max_bottom_vpad > 0)
            && brg->zp_type_a != brgemm_broadcast_t::none;
    const int beta_regs = !one_of(brg->beta, 1.f, 0.f);

    const int max_isa_regs = isa_num_vregs(brg->isa_impl);
    auto max_reg_count = max_isa_regs - max_bcst_regs - beta_regs
            - req_compensation - req_zp_a_comp_pads;
    if (req_zp_a_comp_pads)
        max_reg_count
                = nstl::min(max_reg_count, max_isa_regs - max_bcst_regs - 5);

    int max_bcast_block = max_reg_count - adj_ld_block2;

    if (brg->is_bf16_emu) {
        // in theory, vmm bf16_emu register indices overlap with other vmm
        // registers related to 'max_bcast_block'
        assert(is_superset(brg->isa_impl, sve_512));
        constexpr int bf16_emu_reg_count = 28;
        max_bcast_block = nstl::min(max_bcast_block, bf16_emu_reg_count);
    }

    if (brg->is_int8 && !brg->has_int8_vnni) max_bcast_block -= 2;

    max_bcast_block /= adj_ld_block2;

    return max_bcast_block;
}

int default_mmla_ld_block2(const brgemm_desc_t *brg) {
    constexpr int heur_max_acc_regs = 24;
    constexpr int heur_min_ld_block2 = 3;
    constexpr int heur_max_ld_block2 = 6;

    if (brg->bcast_dim <= 2) return 12;
    if (!brg->brgattr.use_mmla_packed_a && brg->bcast_dim > 8) return 4;

    const int bd_block = nstl::min(brg->bcast_dim, 8);
    const int bd_pairs = utils::div_up(bd_block, mmla_bd_blk());
    const int max_ld_block2 = heur_max_acc_regs / (2 * bd_pairs);
    return nstl::min(
            heur_max_ld_block2, nstl::max(heur_min_ld_block2, max_ld_block2));
}

inline size_t data_type_vnni_granularity(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f32:
        case s32: return size_t(1);
        case f16:
        case bf16: return size_t(2);
        case s8:
        case u8: return size_t(4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return size_t(0); /* should not be reachable */
}

status_t brgemm_blocking(brgemm_desc_t *brg) {

    CHECK(set_isa_impl(brg));
    if (brg->isa_impl == isa_undef) return status::unimplemented;
    assert(!brg->is_dgmm); // should not be called from brdgmm
    set_brg_vmm(brg);

    const bool use_mmla = brg->brgattr.use_mmla;
    brg->ld_block = simd_elems(brg->dt_c, brg->isa_impl);
    brg->ldb = use_mmla ? utils::div_up(brg->load_dim, brg->ld_block)
                        : brg->load_dim / brg->ld_block;
    brg->ldb_tail = brg->load_dim % brg->ld_block;

    const int min_block = 1;
    const int max_vpad = nstl::max(
            brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);
    const int mmla_max_bd_block = 8;

    auto calc_max_bcast_block = [&](int try_ld_block2) {
        const int adj_ld_block2 = calculate_ldb_params(brg, try_ld_block2);
        return calculate_max_bcast_block(brg, adj_ld_block2);
    };

    if (use_mmla) {
        const int ld_block2 = brg->brgattr.hint_ld_block2 != 0
                ? brg->brgattr.hint_ld_block2
                : default_mmla_ld_block2(brg);
        calculate_ldb_params(brg, ld_block2);
        auto set_mmla_blocking = [&]() {
            brg->bdb = brg->bcast_dim / brg->bd_block;
            brg->bdb_tail = brg->bcast_dim % brg->bd_block;
            brg->rd_block = mmla_rd_block();
            brg->rdb = brg->reduce_dim / brg->rd_block;
            brg->rdb_tail = brg->reduce_dim % brg->rd_block;
            brg->is_M_tail = false;
        };
        brg->bd_block = brg->brgattr.hint_bd_block != 0
                ? brg->brgattr.hint_bd_block
                : (brg->brgattr.use_mmla_packed_a
                                          || brg->bcast_dim <= mmla_max_bd_block
                                  ? nstl::min(brg->bcast_dim, mmla_max_bd_block)
                                  : nstl::min(brg->bcast_dim, 6));
        set_mmla_blocking();
        return status::success;
    }

    int max_bcast_block = calc_max_bcast_block(4);
    if (max_bcast_block < max_vpad) {
        max_bcast_block = calc_max_bcast_block(2);
    }

    float best_bd_block_eff = 0.f;
    brg->bd_block = 1;
    for (int bd_block = max_bcast_block; bd_block >= min_block; bd_block--) {
        const int min_bd_block = brg->bcast_dim % bd_block != 0
                ? brg->bcast_dim % bd_block
                : bd_block;
        if (max_vpad > min_bd_block) continue;

        const auto bd_block_disb = static_cast<float>(brg->bcast_dim)
                / rnd_up(brg->bcast_dim, bd_block);
        const auto brgemm_microkernel_eff
                = (static_cast<float>(brg->ld_block2) * bd_block)
                / (((brg->ld_block2) + bd_block) * max_bcast_block);
        const auto bd_block_eff = bd_block_disb * brgemm_microkernel_eff;

        float block_foot_print = static_cast<float>(brg->typesize_A)
                * (bd_block * brg->reduce_dim);
        if (block_foot_print <= static_cast<float>(
                    platform::get_per_core_cache_size(1))
                && bd_block_eff > best_bd_block_eff) {
            brg->bd_block = bd_block;
            best_bd_block_eff = bd_block_eff;
        }
    }
    brg->bdb = brg->bcast_dim / brg->bd_block;
    brg->bdb_tail = brg->bcast_dim % brg->bd_block;

    const int rd_unroll = 4;
    const int vnni_granularity = data_type_vnni_granularity(brg->dt_a);
    brg->rd_block = rd_unroll * vnni_granularity;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    brg->is_M_tail = false;
    return status::success;
}

status_t brdgmm_blocking(brgemm_desc_t *brg) {

    if (brg->isa_impl == isa_undef) return status::unimplemented;

    const int requires_permute_dst_vmm = brg->isa_impl == sve_512
            && jit_brdgmm_kernel_base_t::is_fast_vnni_int8(*brg);
    const int max_vregs = isa_num_vregs(brg->isa_impl);
    const int compute_vregs = 2; // b_vmm + a_vmm
    const int aux_vregs
            = nstl::max(brg->is_bf16_emu * 4, 2) + requires_permute_dst_vmm;
    const int max_acc_vmms = max_vregs - aux_vregs - compute_vregs;
    const int simd_w = isa_max_vlen(brg->isa_impl) / brg->typesize_C;

    auto &M = brg->bcast_dim;
    auto &N = brg->load_dim;

    // In current implementation of dgmm, there is no reduce dim.
    auto &m_block1 = brg->bd_block;
    auto &nb_m_block1 = brg->bdb;
    auto &m_block1_tail = brg->bdb_tail;
    auto &m_block2 = brg->bd_block2;
    auto &nb_m_block2 = brg->bdb2;
    auto &m_block2_tail = brg->bdb2_tail;

    auto &n_block1 = brg->ld_block;
    auto &nb_n_block1 = brg->ldb;
    auto &n_block1_tail = brg->ldb_tail;
    auto &n_block2 = brg->ld_block2;
    auto &nb_n_block2 = brg->ldb2;
    auto &n_block2_tail = brg->ldb2_tail;

    // begin blocking
    const int n_block1_num_steps = 1;
    n_block1 = n_block1_num_steps * simd_w;
    nb_n_block1 = div_up(N, n_block1);
    n_block1_tail = N % n_block1;

    const int max_n_block2_vmms = 4;
    const int max_n_block2 = max_n_block2_vmms / n_block1_num_steps;
    n_block2 = nstl::min(max_n_block2, nb_n_block1);
    nb_n_block2 = div_up(nb_n_block1, n_block2);
    n_block2_tail = nb_n_block1 % n_block2;

    m_block1 = 1;
    nb_m_block1 = M / m_block1;
    m_block1_tail = M % m_block1;
    m_block2 = nstl::min(
            nb_m_block1, max_acc_vmms / (n_block2 * n_block1_num_steps));
    nb_m_block2 = div_up(nb_m_block1, m_block2);
    m_block2_tail = nb_m_block1 % m_block2;

    return status::success;
}

status_t init_brgemm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides, bool is_bf32) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = brg->is_row_major() ? dt_a : dt_b;
    brg->dt_b = brg->is_row_major() ? dt_b : dt_a;
    CHECK(init_kernel_datatype(brg, brg->dt_a, brg->dt_b));

    if (brg->is_f32 && (dt_b == data_type::bf16)) return status::unimplemented;

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;
    CHECK(set_isa_impl(brg));
    brg->is_bf32 = false;

    brg->has_int8_vnni = true;

    set_brg_vmm(brg); // TODO: Investigate if it is really needed here.
    brg->req_s8s8_compensation = (brg->is_int8 && (brg->dt_a == data_type::s8)
            && !isa_has_s8s8(brg->isa_impl));

    // For gemv, we need both A and B to have contiguous elements in memory.
    // Therefore, they cannot be blocked, and checking the column order ensures
    // this for A, as it is only column major when wtag=ba. For B, LDB=1
    // guarantees contiguous elements for B as LDB > 1 means that B is blocked.
    brg->is_gemv = (M == 1 && brg->is_col_major()) || (N == 1 && LDB == 1);

    if (brg->is_gemv) {
        brg->LDA = static_cast<int>(LDA);
        brg->LDB = static_cast<int>(LDB);
    } else {
        brg->LDA = (brg->is_row_major()) ? static_cast<int>(LDA)
                                         : static_cast<int>(LDB);
        brg->LDB = (brg->is_row_major()) ? static_cast<int>(LDB)
                                         : static_cast<int>(LDA);
    }

    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim
            = (brg->is_row_major()) ? static_cast<int>(M) : static_cast<int>(N);
    brg->load_dim
            = (brg->is_row_major()) ? static_cast<int>(N) : static_cast<int>(M);
    brg->reduce_dim = static_cast<int>(K);

    brg->bd_block2 = 0;
    brg->bdb2 = 0;
    brg->bdb2_tail = 0;

    brg->ld_step = data_type_vnni_granularity(brg->dt_b);

    const bool has_no_vnni_compute_instruction = false;
    brg->rd_step = has_no_vnni_compute_instruction
            ? 1
            : data_type_vnni_granularity(brg->dt_b);

    if ((sme == brg->isa_impl)
            && (!brg->is_f32 || (data_type::f32 != dt_b)
                    || (data_type::f32 != dt_a)
                    || ((0.f != brg->beta) && (1.f != brg->beta))
                    || (1.f != brg->alpha) || (brgemm_addr != brg->type)
                    || (brgemm_row_major != layout)))
        return status::unimplemented;
    return status::success;
}

status_t init_brdgmm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;
    CHECK(init_kernel_datatype(brg, brg->dt_a, brg->dt_b));

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) && one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_f32 || brg->is_bf16 || brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(sve_512), sve_512,
                is_isa_ok(sve_256), sve_256, is_isa_ok(sve_128), sve_128);
    }

    if (!IMPLICATION(brg->is_bf16, mayiuse_bf16())) {
        return status::unimplemented;
    }

    brg->is_dgmm = true;

    brg->LDA = static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim = M;
    brg->load_dim = N;
    return status::success;
}

} // namespace brgemm_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
