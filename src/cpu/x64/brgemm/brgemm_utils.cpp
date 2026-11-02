/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "cpu/x64/brgemm/brgemm_utils.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

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
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16
            || brg->is_fp8);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

status_t init_kernel_datatype(
        brgemm_desc_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    if (utils::one_of(data_type::undef, dt_a, dt_b)) {
        assert(!"Unsupported data type");
        return status::unimplemented;
    }

    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && utils::one_of(dt_b, data_type::u8, data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    brg->is_f32 = (dt_a == data_type::f32)
            && utils::one_of(
                    dt_b, data_type::f32, data_type::bf16, data_type::f16);
    brg->is_f16 = (dt_a == data_type::f16)
            && utils::one_of(dt_b, data_type::f32, data_type::f16);
    brg->is_fp8 = one_of(dt_a, data_type::f8_e5m2, data_type::f8_e4m3)
            && one_of(dt_b, data_type::f8_e5m2, data_type::f8_e4m3);
    if (utils::everyone_is(false, brg->is_int8, brg->is_bf16, brg->is_f32,
                brg->is_f16, brg->is_fp8)) {
        assert(!"Unsupported data type");
        return status::unimplemented;
    }
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
    brg->with_src_scales = false;
    brg->with_wei_scales = false;
    brg->with_dst_scales = false;
    brg->dt_wei_scales = data_type::undef;

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }
}

namespace brgemm_utils {

void maybe_try_bf32(brgemm_desc_t *brg) {
    const bool try_bf32 = brg->is_f32
            && one_of(brg->brgattr.fpmath_mode, fpmath_mode::bf16,
                    fpmath_mode::any)
            && one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);
    if (try_bf32) {
        const bool is_tmm = brg->is_tmm;
        brg->is_tmm = true;
        if (brg->can_dispatch_uker() /*Requires is_tmm to be true*/) {
            brg->is_bf32 = true;
        } else {
            brg->is_bf32 = false;
            //  Restore
            brg->is_tmm = is_tmm;
        }
    }
}

void set_isa_impl(brgemm_desc_t *brg) {
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) &&
                // maybe IMPLICATION(brg->isa_user != isa_undef,
                //  is_superset(brg->isa_user, isa)), but the API is not clear.
                one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_gemv) {
        if (everyone_is(data_type::f32, brg->dt_a, brg->dt_b)) {
            brg->isa_impl = is_isa_ok(avx2) ? avx2 : isa_undef;
        } else if (everyone_is(data_type::bf16, brg->dt_a, brg->dt_b)) {
            brg->isa_impl = is_isa_ok(avx512_core_bf16) ? avx512_core_bf16
                                                        : isa_undef;
        } else if (everyone_is(data_type::f16, brg->dt_a, brg->dt_b)) {
            brg->isa_impl = is_isa_ok(avx512_core_fp16) ? avx512_core_fp16
                                                        : isa_undef;
        }
        return;
    }

    if (brg->is_bf32) {
        brg->isa_impl = avx512_core_amx;
    } else if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef,
                is_isa_ok(avx512_core) || is_isa_ok(avx512_core_amx) /*bf32*/,
                avx512_core, is_isa_ok(avx2), avx2,
                // Allow avx512_core_fp16 isa in case of a f16 primitive that
                // is implemented using pre-conversion of inputs to f32.
                // This is needed to support f16 binary post-ops.
                is_isa_ok(avx512_core_fp16), avx512_core_fp16, is_isa_ok(avx2),
                avx2, is_isa_ok(avx10_2), avx10_2);
    } else if (brg->is_bf16) {
        if (brg->dt_a == data_type::f32 && brg->dt_b == data_type::bf16) {
            // Distinguish f32:bf16 case upconversion for bf16 on AVX512_CORE
            // and AVX2.
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_amx), avx512_core_amx,
                    is_isa_ok(avx512_core_bf16), avx512_core_bf16,
                    is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2_vnni_2),
                    avx2_vnni_2, is_isa_ok(avx2), avx2);
        } else {
            brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2_ace),
                    avx10_2_ace, is_isa_ok(avx10_2_amx_2), avx10_2_amx_2,
                    is_isa_ok(avx512_core_amx), avx512_core_amx,
                    is_isa_ok(avx10_2), avx10_2, is_isa_ok(avx512_core_bf16),
                    avx512_core_bf16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
        }
    } else if (brg->is_f16) {
        if (everyone_is(data_type::f16, brg->dt_a, brg->dt_b)) {
            // ACE is deliberately absent: ACE defines no FP16 outer product,
            // so a descriptor resolving to the ACE ISA could not be computed
            // and would only lose the TMUL FP16 path.
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx10_2_amx_2), avx10_2_amx_2,
                    is_isa_ok(avx512_core_amx_fp16), avx512_core_amx_fp16,
                    is_isa_ok(avx10_2), avx10_2, is_isa_ok(avx512_core_fp16),
                    avx512_core_fp16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
        } else if (brg->dt_a == data_type::f32 && brg->dt_b == data_type::f16) {
            // Distinguish f32:f16 case upconversion for f16 on AVX512_CORE and
            // AVX2.
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16,
                    is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2), avx2);
        } else {
            brg->isa_impl = utils::map(true, isa_undef,
                    is_isa_ok(avx512_core_fp16), avx512_core_fp16);
        }
    } else if (brg->is_int8) {
        // ACE is listed first, matching the bf16 branch above: utils::map
        // returns the first match, so the ACE entry has to precede the TMUL
        // ones for an ACE descriptor to win whenever both are usable.
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2_ace),
                avx10_2_ace, is_isa_ok(avx10_2_amx_2), avx10_2_amx_2,
                is_isa_ok(avx512_core_amx_fp16), avx512_core_amx_fp16,
                is_isa_ok(avx512_core_amx), avx512_core_amx, is_isa_ok(avx10_2),
                avx10_2, is_isa_ok(avx512_core_fp16), avx512_core_fp16,
                is_isa_ok(avx512_core_vnni), avx512_core_vnni,
                is_isa_ok(avx512_core), avx512_core, is_isa_ok(avx2_vnni_2),
                avx2_vnni_2, is_isa_ok(avx2_vnni), avx2_vnni, is_isa_ok(avx2),
                avx2);
    } else if (brg->is_fp8) {
        // ACE omits fp8 outer products; its kernels do not emit MX-block
        // scaled TOP4MX*PS forms.
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2_amx_2),
                avx10_2_amx_2, is_isa_ok(avx10_1_512_amx_fp16),
                avx10_1_512_amx_fp16, is_isa_ok(avx10_2), avx10_2);
    }
}

void set_brg_vmm(brgemm_desc_t *brg) {
    // is_tmm means "accumulates into tile registers", which ACE does as well,
    // so it is listed here next to the TMUL (is_*_tmm) flags. See the note on
    // avx10_2_ace in cpu_isa_traits.hpp.
    brg->is_tmm = brg->is_int8_tmm || brg->is_bf16_tmm || brg->is_f16_tmm
            || brg->is_bf32 || brg->is_fp8_tmm || brg->is_ace();
    brg->is_zmm = !brg->is_tmm && mayiuse(avx512_core)
            && is_superset(brg->isa_impl, avx512_core);
    brg->is_ymm
            = !brg->is_zmm && mayiuse(avx2) && is_superset(brg->isa_impl, avx2);
}

int calculate_ldb_params(brgemm_desc_t *brg, const int try_ld_block2) {
    brg->ld_block2 = try_ld_block2;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;

    if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);
    brg->embd_bcst = brg->is_f32
            && (brg->ldb2_tail <= 1 && brg->ldb2 == 0)
            /*only avx512 or more can bcast*/
            && is_superset(brg->isa_impl, avx512_core);

    const int adj_ld_block2
            = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
    return nstl::max(1, adj_ld_block2);
}

int calculate_max_bcast_block(brgemm_desc_t *brg, const int adj_ld_block2) {

    // TODO: Calculating the number of available registers should be re-factored
    // to use one code here and in brgemm kernel generator on
    // "max_effective_vregs" calculation
    int max_isa_regs = isa_num_vregs(brg->isa_impl);
    const int max_bcst_regs = brg->n_bcast_1_load ? 0 : 1;
    const int load_regs = brg->n_bcast_1_load ? 1 : adj_ld_block2;
    const bool req_zp_a_comp_pads
            = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                      || brg->brgattr.max_bottom_vpad > 0)
            && brg->zp_type_a != brgemm_broadcast_t::none;

    // --------------  whole kernel --------------
    // To support the f16 vnni B matrix on non-AMX we need to use two Vmm
    // registers for permutation in brgemm kernel:
    // see f16_perm_even_vreg_ and f16_perm_odd_vreg_ in brgemm kernel
    const int b_vnni_regs = brg->is_f16_b_non_amx_vnni() ? 2 : 0;

    // non-VNNI INT8 dot product required 2 temp vectors:
    // see int8_ones_words() and int8_dot_product_temp() in brgemm kernel
    const int non_int8_vnni_regs
            = (brg->is_int8 && !brg->has_int8_vnni) ? 2 : 0;

    // FP8 input emulation and FP8 post-ops require five conversion registers.
    // see vmm_fp8_emu_aux* in brgemm kernel
    const int fp8_emu_regs
            = brg->is_fp8_via_convert_non_amx() || brg->has_fp8_postops() ? 5
                                                                          : 0;

    max_isa_regs -= b_vnni_regs + non_int8_vnni_regs + fp8_emu_regs;

    // --------------- microkernel ---------------
    // see vmm_inp_shift() in brgemm kernel
    const int compensation_regs = brg->req_src_s8_shift
                    || brg->zp_type_a != brgemm_broadcast_t::none
            ? 1
            : 0;

    // see vmm_zp_a_shift(), vmm_one_bytes() in brgemm kernel
    const int zp_a_comp_pads_regs = req_zp_a_comp_pads ? 2 : 0;

    const int microkernel_regs = zp_a_comp_pads_regs + compensation_regs;

    const auto microkernel_max_reg_count
            = max_isa_regs - microkernel_regs - load_regs - max_bcst_regs;

    auto microkernel_max_bcast_block
            = microkernel_max_reg_count / (adj_ld_block2 + brg->n_bcast_1_load);

    // ----- post-ops and store accumulators -----
    const int beta_regs = !one_of(brg->beta, 1.f, 0.f);

    const int postops_regs = brg->attr()
            ? injector::aux_vec_count(
                      brg->attr()->post_ops_, brg->isa_impl, true)
            : 0;

    // Emulators: fp8 emulation are supported for amx only
    // In theory, vmm bf16_emu register indices overlap with other vmm
    // registers related to 'max_bcast_block'
    assert(IMPLICATION(
            brg->is_bf16_emu, is_superset(brg->isa_impl, avx512_core)));
    const int bf16_emu_regs = brg->is_bf16_emu ? 4 : 0;

    const auto store_regs = nstl::max(beta_regs,
            nstl::max(
                    postops_regs, nstl::max(compensation_regs, bf16_emu_regs)));

    const auto store_max_reg_count = max_isa_regs - store_regs;

    auto store_max_bcast_block = store_max_reg_count / adj_ld_block2;

    // ------------ final calculation ------------
    const auto max_bcast_block
            = nstl::min(microkernel_max_bcast_block, store_max_bcast_block);

    return max_bcast_block;
}

// Helper function to find bdb and bdb_tail given bd_block and taking into
// account bd_mask
static void find_bdb_bd_mask(
        const brgemm_desc_t *brg, int bd_block, dim_t &bdb, int &bdb_tail) {
    const auto BD = brg->bcast_dim;

    if (brg->brgattr.bd_mask_level != 2 || BD == 0) {
        bdb = div_up(BD, bd_block);
        bdb_tail = BD % bd_block;
        return;
    }

    bdb = 0;
    bdb_tail = 0;
    for (int i = 0; i < BD;) {
        if (brg->brgattr.bd_mask_level == 2 && brg->brgattr.bd_mask[i] == 0) {
            i++;
        } else {
            i += bd_block;
            if (i > BD) {
                // Remainder bounded by bd_block, safe to narrow.
                bdb_tail = static_cast<int>(BD - i + bd_block);
                if (brg->brgattr.use_uker) bdb++;
            } else
                bdb++;
        }
    }
}

// Helper function to find bdb2 and ldb2 by given blocking
static void recalc_blocking(brgemm_desc_t *brg, int new_bd_block,
        int new_ld_block, int new_bd_block2, int new_ld_block2) {
    const auto LD = brg->load_dim;

    if (new_bd_block != 0) {
        brg->bd_block = new_bd_block;
        find_bdb_bd_mask(brg, brg->bd_block, brg->bdb, brg->bdb_tail);
        brg->is_M_tail = (brg->bdb_tail != 0);
    }

    if (new_ld_block != 0) {
        brg->ld_block = new_ld_block;
        brg->ldb = div_up(LD, brg->ld_block);
        brg->ldb_tail = LD % brg->ld_block;
    }

    if (new_bd_block2 != 0) {
        brg->bd_block2 = new_bd_block2;
        if (brg->can_dispatch_uker()) {
            brg->bdb2 = div_up(brg->bdb, brg->bd_block2);
            brg->bdb2_tail = 0;
        } else {
            if (brg->bdb_tail && brg->bd_block2 > 1) brg->bd_block2--;
            auto full_bd_blocks = brg->bdb - (brg->bdb_tail != 0 ? 1 : 0);
            brg->bdb2 = full_bd_blocks / brg->bd_block2;
            brg->bdb2_tail = full_bd_blocks % brg->bd_block2;
        }
    }

    if (new_ld_block2 != 0) {
        brg->ld_block2 = new_ld_block2;
        if (brg->can_dispatch_uker()) {
            brg->ldb2 = div_up(brg->ldb, brg->ld_block2);
            brg->ldb2_tail = 0;
        } else {
            if (brg->ldb_tail && brg->ld_block2 > 1) brg->ld_block2--;
            auto full_ld_blocks = brg->ldb - (brg->ldb_tail != 0 ? 1 : 0);
            brg->ldb2 = full_ld_blocks / brg->ld_block2;
            brg->ldb2_tail = full_ld_blocks % brg->ld_block2;
        }
    }
}

status_t brgemm_blocking_tmm(brgemm_desc_t *brg) {
    const auto L1 = platform::get_per_core_cache_size(1);

    // Blocking configuration for AMX
    const auto BD = brg->bcast_dim;
    const auto BD_R16 = rnd_up(BD, 16);
    const auto LD = brg->load_dim;
    const auto LD_R16 = rnd_up(LD, 16);

    const int max_width = 16, min_width = 1;
    brg->ld_block = 16;
    brg->ldb = LD / brg->ld_block;
    brg->ldb_tail = LD % brg->ld_block;

    auto find_bd_block_for_bd_mask = [&]() {
        if (brg->brgattr.bd_mask_level != 2 || BD == 0) return false;

        dim_t min_bdb = INT_MAX;
        const int start_bd_block
                = static_cast<int>(nstl::min<dim_t>(max_width, BD));
        auto best_bd_block = start_bd_block;
        for (auto bd_block = start_bd_block; bd_block > 0; bd_block--) {
            dim_t bdb = 0;
            int bdb_tail = 0;
            find_bdb_bd_mask(brg, bd_block, bdb, bdb_tail);
            // bcast_dim should be divided by bd_block
            if (bdb < min_bdb && bdb_tail == 0) {
                min_bdb = bdb;
                best_bd_block = bd_block;
            }
        }
        brg->bd_block = best_bd_block;
        brg->bdb_tail = 0;
        brg->bdb = min_bdb;
        return true;
    };

    auto set_decomposition_by_ld = [&]() {
        if (brg->bd_block2 == 1 && brg->ldb > 0 && brg->ldb_tail == 0) {
            if (brg->ldb % 3 == 0)
                brg->ld_block2 = 3;
            else if (brg->ldb % 2 == 0)
                brg->ld_block2 = 2;
            else
                brg->ld_block2 = 1;
        } else {
            brg->ld_block2
                    = (brg->ldb > 0 && brg->ldb % 2 == 0 && brg->ldb_tail == 0
                              && brg->bd_block2 < 3)
                    ? 2
                    : 1;
        }
        brg->ldb2 = brg->ldb / brg->ld_block2;
        brg->ldb2_tail = brg->ldb % brg->ld_block2;

        // Re-adjust the bd_block2 if possible
        if (brg->ld_block2 == 1 && !brg->is_M_tail && brg->ldb_tail == 0) {
            brg->bd_block2 = (brg->bdb >= 3) ? 3 : (brg->bdb >= 2) ? 2 : 1;
            brg->bdb2 = brg->bdb / brg->bd_block2;
            brg->bdb2_tail = static_cast<int>((brg->bd_block2 == 1)
                            ? brg->bdb
                            : brg->bdb % brg->bd_block2);
        }
    };

    auto try_3x1_decomposition = [&](int width_step) {
        brg->is_M_tail = false;
        if (BD > (width_step - 1) * max_width && BD < width_step * max_width
                && brg->ldb_tail == 0) {
            if (!find_bd_block_for_bd_mask()) {
                brg->bd_block = max_width;
                brg->bdb = div_up(BD, brg->bd_block);
                brg->bdb_tail = BD % brg->bd_block;
                brg->is_M_tail = true;
            }
            brg->bd_block2 = width_step;
            brg->bdb2 = brg->bdb / brg->bd_block2;
            brg->bdb2_tail = brg->bdb % brg->bd_block2;
            set_decomposition_by_ld();
            return true;
        }
        return false;
    };

    auto try_2x2_decomposition = [&]() {
        if (!find_bd_block_for_bd_mask()) {
            for (int m_block = max_width; m_block >= min_width; m_block--) {
                if (BD % m_block == 0) {
                    brg->bd_block = m_block;
                    break;
                }
            }
            if (brg->bd_block == 1) {
                brg->bd_block
                        = static_cast<int>(nstl::min<dim_t>(max_width, BD));
                brg->bdb_tail = BD % max_width;
                for (int i = max_width; i >= min_width; i--) {
                    const int i_tail = BD % i;
                    if (i_tail > brg->bdb_tail || i_tail == 0) {
                        brg->bd_block = i;
                        brg->bdb_tail = i_tail;
                        if (i_tail == 0) break;
                    }
                }
            }
            brg->bdb = BD / brg->bd_block;
            brg->bdb_tail = BD % brg->bd_block;
        }

        brg->bd_block2 = (brg->bdb >= 2) ? 2 : 1;
        brg->bdb2 = brg->bdb / brg->bd_block2;
        brg->bdb2_tail = static_cast<int>(
                (brg->bd_block2 == 1) ? brg->bdb : brg->bdb % brg->bd_block2);

        brg->is_M_tail = false;

        set_decomposition_by_ld();

        return !(brg->ld_block2 == 1 || brg->bd_block2 == 1
                || brg->bd_block < 8);
    };

    auto recalc_blocking_ext
            = [&](int new_bd_block, int new_ld_block, int new_bd_block2,
                      int new_ld_block2, bool load_nt_A, bool load_nt_B,
                      brgemm_kernel_innermost_loop_t innermost_loop) {
        recalc_blocking(
                brg, new_bd_block, new_ld_block, new_bd_block2, new_ld_block2);
        brg->load_nt_A = load_nt_A;
        brg->load_nt_B = load_nt_B;
        brg->innermost_loop = innermost_loop;
    };

    bool is_decomposition_defined = false;
    for (int i = decomposition_2x2; i != undefined; i++) {
        switch (i) {
            case decomposition_2x2:
                is_decomposition_defined = try_2x2_decomposition();
                break;
            case decomposition_3x1_3:
                is_decomposition_defined = try_3x1_decomposition(3);
                break;
            case decomposition_3x1_2:
                is_decomposition_defined = try_3x1_decomposition(2);
                break;
            default: assert(!"invalid value"); break;
        };
        if (is_decomposition_defined) break;
    }
    if (!is_decomposition_defined) try_2x2_decomposition();

    const bool try_load_nt_A
            = (brg->innermost_loop == brgemm_bd_loop_innermost);
    const bool try_load_nt_B
            = (brg->innermost_loop == brgemm_ld_loop_innermost);
    const bool try_load_nt
            = (static_cast<size_t>(brg->typesize_A)
                              * brg->brgattr.hint_expected_A_size
                      + static_cast<size_t>(brg->typesize_B)
                              * brg->brgattr.hint_expected_B_size
                      + static_cast<size_t>(brg->typesize_C)
                              * brg->brgattr.hint_expected_C_size)
            >= L1;
    brg->load_nt_A = try_load_nt_A && try_load_nt;
    brg->load_nt_B = try_load_nt_B && try_load_nt;

    recalc_blocking(
            brg, brg->bd_block, brg->ld_block, brg->bd_block2, brg->ld_block2);

    if (brg->can_dispatch_uker()) {
        // Blocking heuristics for some shapes
        // TODO: Review these criteria
        const size_t eff_K = static_cast<size_t>(
                static_cast<float>(brg->reduce_dim * brg->typesize_A)
                * brg->brgattr.K_koef);
        const auto low_K = (L1 - 4 * 1024) / (6 * 16);

        // TODO: if rdb_tail != 0 then we should limit
        // blocking because we need extra tiles for A and B to load rdb_tail
        // if bd_mask_level != 0 it means it aligned to 16

        const bool bdb_block_tail = !(brg->bd_block > 12
                && (BD % brg->bd_block == 0
                        && brg->brgattr.bd_mask_level == 0));
        const bool ldb_tail_16 = (LD % 16 != 0);
        if (everyone_is(false, bdb_block_tail, ldb_tail_16)) {
            // try to use 1x(4|5) or (4|5)x1 decomposition for specific
            // range of K
            const auto upper_K5 = (L1 - 5 * 1024) / (5 * 16);
            const auto upper_K4 = (L1 - 4 * 1024) / (4 * 16);
            const bool K5_fit_L1 = (low_K <= eff_K && eff_K < upper_K5);
            const bool K4_fit_L1 = (low_K <= eff_K && eff_K < upper_K4);
            const bool bd_big = (BD > 32);
            const bool ld_big = (LD > 32);
            const bool aligned_bd_mask
                    = brg->brgattr.bd_mask_level != 0 && brg->bdb % 4 == 0;
            if (LD % 80 == 0 && K5_fit_L1 && bd_big) {
                recalc_blocking_ext(
                        0, 16, 1, 5, true, false, brgemm_bd_loop_innermost);
            } else if (LD % 64 == 0 && K4_fit_L1 && bd_big) {
                recalc_blocking_ext(
                        0, 16, 1, 4, true, false, brgemm_bd_loop_innermost);
            } else if ((BD % 80 == 0 || aligned_bd_mask) && K5_fit_L1
                    && ld_big) {

                recalc_blocking_ext(
                        0, 16, 5, 1, false, true, brgemm_ld_loop_innermost);
            } else if ((BD % 64 == 0 || aligned_bd_mask) && K4_fit_L1
                    && ld_big) {
                recalc_blocking_ext(
                        16, 16, 4, 1, false, true, brgemm_ld_loop_innermost);
            }
        }
        // Tile decomposition for shapes with small dimensions
        // or dimensions with tails
        const bool weak_ldb = brg->ld_block < 8 || brg->ldb_tail > 0;
        const bool weak_bdb = brg->bd_block < 8 || brg->bdb_tail > 0;
        const bool ldb_tail_only = ldb_tail_16 && !bdb_block_tail;
        const bool bdb_tail_only = bdb_block_tail && !ldb_tail_16;
        if (ldb_tail_only && LD > 64 && brg->ld_block < 8) {
            recalc_blocking(brg, 0, 16, 2, 1);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 64) {
            recalc_blocking(brg, 0, 16, 1, 4);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 48) {
            recalc_blocking(brg, 0, 16, 1, 3);
        } else if (ldb_tail_only && weak_ldb && LD_R16 == 32) {
            recalc_blocking(brg, 0, 16, 2, 2);
        } else if (BD <= 16) {
            // Have to call recalc_blocking twice to calculate ldb
            // BD <= 16 here, safe to narrow into the bd_block parameter.
            recalc_blocking(brg, static_cast<int>(BD), 16, 0, 0);
            const int ld_block2 = static_cast<int>(
                    nstl::min<dim_t>(ldb_tail_16 ? ((brg->ldb > 4) ? 3 : 4) : 5,
                            div_up(LD, 16)));
            recalc_blocking(brg, 0, 0, 1, ld_block2);
        } else if (bdb_tail_only && weak_bdb && BD > 64) {
            recalc_blocking(brg, 16, 16, 1, 2);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 64) {
            recalc_blocking(brg, 16, 16, 4, 1);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 48) {
            recalc_blocking(brg, 16, 16, 3, 1);
        } else if (bdb_tail_only && weak_bdb && BD_R16 == 32
                && (LD % 32 == 0)) {
            recalc_blocking(brg, 16, 16, 2, 2);
        } else if (LD <= 16) {
            // Have to call recalc_blocking twice to calculate bdb
            // we can't use ld_block other than 16
            recalc_blocking(brg, 16, 16, 0, 0);
            const int bd_block2 = static_cast<int>(
                    nstl::min<dim_t>(brg->bdb_tail ? (brg->bdb > 4 ? 3 : 4) : 5,
                            div_up(BD, 16)));
            recalc_blocking(brg, 0, 0, bd_block2, 1);
        } else if (bdb_block_tail && ldb_tail_16 && BD_R16 == 32 && LD_R16 == 32
                && (weak_ldb || weak_bdb)) {
            recalc_blocking(brg, 16, 16, 2, 2);
        }

        // The code below is a draft for the future optimization of interleave
        // stores and small number of iterations.
        // TODO: review and enable if needed
#if 0
        // if interleave stores and small number of iterations then
        // try to increase them
        const auto n_iterations = brg->bdb2 * brg->bdb2;
        if (brg->brgattr.use_interleave_stores && n_iterations < 4) {
            int k_it = div_up(4, n_iterations);
            if (brg->bdb2 > brg->ldb2)
                recalc_blocking(0, 0, div_up(brg->bdb2, k_it), 0);
            else
                recalc_blocking(0, 0, 0, div_up(brg->ldb2, k_it));
        }
#endif
    }

    if (brg->get_num_A_tiles() + brg->get_num_B_tiles() + brg->get_num_C_tiles()
            > brgemm_desc_t::AMX_TILES_NUM) {
        assert(!"brgemm internal error: invalid blocking");
        return status::runtime_error;
    }

    // check hints for blocking parameters
    recalc_blocking(brg, brg->brgattr.hint_bd_block, brg->brgattr.hint_ld_block,
            brg->brgattr.hint_bd_block2 ? brg->brgattr.hint_bd_block2
                                        : brg->bd_block2,
            brg->brgattr.hint_ld_block2 ? brg->brgattr.hint_ld_block2
                                        : brg->ld_block2);

    if (brg->brgattr.hint_load_nt_A != brgemm_hint_nt_undef)
        brg->load_nt_A = (brg->brgattr.hint_load_nt_A == brgemm_hint_nt_true);
    if (brg->brgattr.hint_load_nt_B != brgemm_hint_nt_undef)
        brg->load_nt_B = (brg->brgattr.hint_load_nt_B == brgemm_hint_nt_true);

    // TODO: if rd_block calculated is very small then maybe it makes
    // sense to use 1x2 or 2x1 blocking with supporting rd_block
    // and rdb_tail
    const auto rd_block_step = brg->rd_block_step();
    const auto max_rd_block = brg->max_rd_block();
    if (brg->amx_may_extend_k()) {
        brg->rd_block = static_cast<int>(nstl::min<dim_t>(
                rnd_up(brg->reduce_dim, brg->rd_step), max_rd_block));
    } else if (brg->fused_copy_a) {
        brg->rd_block = max_rd_block;
    } else {
        brg->rd_block = rd_block_step;
        for (int i = max_rd_block; i > 0; i -= rd_block_step) {
            if (brg->reduce_dim % i == 0) {
                brg->rd_block = i;
                break;
            }
        }
    }

    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    // Remove these guards in the future (add tail processing by reduction
    // dimension)
    // TODO: these checks do not work for fp8-f16 and f16-fp8 cfgs
    if (!IMPLICATION(brg->rdb > 0 && brg->rdb_tail,
                brg->is_input_convert() || brg->amx_wary_k_tail()
                        || brg->fused_copy_a)) {
        return status::unimplemented;
    }

    if (!IMPLICATION((brg->rdb_tail
                             % ((brg->is_bf16_tmm || brg->is_f16_tmm) ? 2 : 4))
                        != 0,
                brg->is_input_convert() || brg->amx_wary_k_tail()
                        || brg->fused_copy_a)) {
        return status::unimplemented;
    }

    //TODO: check this condition
    brg->interleave_tilestores_ = brg->beta == 0
                    && (brg->brgattr.use_interleave_stores
                            && (brg->bd_block2 * brg->ld_block2 == 4)
                            && !brg->brgattr.var_bs)
            ? true
            : false;
    return status::success;
}

/**
 * GEMV Register Blocking Strategy (Row-Major A)
 * ============================================
 *
 * Assumptions:
 * - Target ISA: AVX2 (for now)
 * - Only N = 1 is supported (no blocking along load dimension)
 * - A is in row-major layout
 *
 * ------------------------------------------------------------
 * not transposed (non-transA)
 * ------------------------------------------------------------
 *
 * Blocking scheme:
 * bd_block = 8:
 *   - Uses 8 independent accumulators: acc0 to acc7
 *
 * rd_block = simd_w:
 *   - 1 vector register for x
 *   - 1 vector register (a0_reg), reused to load one A micro-row at a time
 *
 * Register usage:
 *   [x_reg]   = vector register holding x[j .. j + rd_block - 1]
 *   [a0_reg]  = vector register reused for each A micro-row
 *   [acc0..7] = accumulators for bd_block rows
 *
 * Microkernel loop:
 *
 *   Load x_reg = x[0 .. rd_block - 1]          // Load entire vector block once
 *
 *   for bd in 0 .. bd_block - 1:
 *     Load a0_reg = A[bd][0 .. rd_block - 1]   // Load A micro-row
 *     acc[bd] += dot(a0_reg, x_reg)            // Accumulate partial results
 *
 * ---------------------------------------------------
 * transposed (transA)
 * ---------------------------------------------------
 *
 * In the transposed GEMV case one broadcasted scalar from B is reused across
 * several independent accumulators to increase ILP.
 *
 * 1. First-level register blocking
 *    - one accumulator is one vector register
 *    - for AVX f32, simd_w = 8 (hardcoded for now)
 *
 * 2. Second-level register blocking
 *    - `gemv_transa_bd_unroll` is the number of such accumulators
 *      updated per reduction step
 *    - `gemv_bd_block()` returns this logical GEMV block size
 *
 * To preserve the existing brgemm harness and outer bdb counting logic
 * `bd_block` is encoded as:
 *
 *     bd_block = simd_w * gemv_transa_bd_unroll
 *
 * This encoding is used only for GEMV + transA.
 *
 * Tail handling in GEMV + transA follows the same two-level structure:
 *
 * 1. Second-level tail
 *    - number of full accumulators in the tail block
 *    - returned by `gemv_bdb_tail()`
 *
 * 2. First-level tail
 *    - remaining valid elements in the final accumulator
 *    - stored in `gemv_tail`
 *
 * So for GEMV + transA:
 *
 *     bdb_tail = gemv_bdb_tail() * simd_w + gemv_tail
 *
 * Microkernel structure:
 *
 * for rd:
 *     broadcast b
 *     for bd in 0 .. gemv_bd_block() - 1:
 *         load one vector a
 *         acc[bd] += a * b
 *
 * Tail block:
 *   - process `gemv_bdb_tail()` full accumulators
 *   - if `gemv_tail > 0` process one final masked vector
 *
 */
status_t brgemm_blocking_vmm_gemv(brgemm_desc_t *brg) {
    assert(utils::one_of(
            brg->isa_impl, avx2, avx512_core_bf16, avx512_core_fp16));
    assert(brg->load_dim == 1);

    const int simd_w = is_superset(brg->isa_impl, avx512_core) ? 16 : 8;

    // Blocking parameters for the non-transposed case.
    if (!brg->transA) {
        brg->ld_block = 1;
        brg->ldb = brg->load_dim / brg->ld_block;
        brg->ldb_tail = brg->load_dim % brg->ld_block;
        assert(brg->ldb_tail == 0);

        brg->ld_block2 = 1;
        brg->ldb2 = brg->ldb / brg->ld_block2;
        brg->ldb2_tail = brg->ldb % brg->ld_block2;
        assert(brg->ldb2_tail == 0);

        brg->bd_block = 8;
        brg->bdb = brg->bcast_dim / brg->bd_block;
        brg->bdb_tail = brg->bcast_dim % brg->bd_block;

        brg->rd_block = brg->gemv_use_vdpbf16ps() ? 2 * simd_w : simd_w;
        brg->rdb = brg->reduce_dim / brg->rd_block;
        brg->rdb_tail = brg->reduce_dim % brg->rd_block;

        brg->gemv_tail = brg->rdb_tail;

        return status::success;
    }

    // Blocking parameters for the transposed case.
    brg->ld_block = 1;
    brg->ldb = brg->load_dim / brg->ld_block;
    brg->ldb_tail = brg->load_dim % brg->ld_block;
    assert(brg->ldb_tail == 0);

    brg->ld_block2 = 1;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;
    assert(brg->ldb2_tail == 0);

    brg->gemv_transa_bd_unroll = 8;
    brg->bd_block = simd_w * brg->gemv_transa_bd_unroll;
    brg->bdb = brg->bcast_dim / brg->bd_block;
    brg->bdb_tail = brg->bcast_dim % brg->bd_block;

    brg->rd_block = 1;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    brg->gemv_tail = brg->bdb_tail % simd_w;

    return status::success;
}

status_t brgemm_blocking_ace(brgemm_desc_t *brg) {
    const auto LD = brg->load_dim;
    const auto BD = brg->bcast_dim;

    brg->ld_block = 16;
    // Use actual M as bd_block for M < 16.
    brg->bd_block = static_cast<int>(nstl::min<dim_t>(BD, 16));

    // recalc_blocking() below recomputes ldb/bdb and the tails; use the same
    // convention here so that the search picks the block2 values for the
    // blocking the kernel is actually built with.
    const auto ldb = div_up(LD, brg->ld_block);
    dim_t bdb = 0;
    int bdb_tail = 0;
    find_bdb_bd_mask(brg, brg->bd_block, bdb, bdb_tail);

    const auto ntiles = brgemm_desc_t::AMX_TILES_NUM;

    // ACE post-op zmm layout: bias uses zmm10-17, scales use zmm18-23.
    // Active scales cap ld_block2 at 6 to avoid overlap with accm zmm24-31.
    const bool needs_scales = brg->with_src_scales || brg->with_wei_scales;
    const int max_postop_ld_block2 = needs_scales ? 6 : 8;

    dim_t best_loads_number = LLONG_MAX;
    auto best_bd_block2 = 1;
    auto best_ld_block2 = 1;
    const int max_bd_block2 = static_cast<int>(nstl::min<dim_t>(ntiles, bdb));
    for (int bd_block2 = 1; bd_block2 <= max_bd_block2; bd_block2++) {
        const int ld_block2 = static_cast<int>(nstl::min<dim_t>(
                nstl::min<dim_t>(nstl::max<dim_t>(1, ldb), ntiles / bd_block2),
                max_postop_ld_block2));

        // Calculate the number of loads for one iteration by reduce_dim
        const auto loads_number
                = static_cast<dim_t>(ldb) * div_up(bdb, bd_block2)
                + bdb * div_up(ldb, ld_block2);
        if (loads_number < best_loads_number) {
            best_loads_number = loads_number;
            best_bd_block2 = bd_block2;
            best_ld_block2 = ld_block2;
        }
    }
    brg->bd_block2 = best_bd_block2;
    brg->ld_block2 = best_ld_block2;

    recalc_blocking(
            brg, brg->bd_block, brg->ld_block, brg->bd_block2, brg->ld_block2);

    // check hints for blocking parameters
    recalc_blocking(brg, brg->brgattr.hint_bd_block, brg->brgattr.hint_ld_block,
            brg->brgattr.hint_bd_block2 ? brg->brgattr.hint_bd_block2
                                        : brg->bd_block2,
            brg->brgattr.hint_ld_block2 ? brg->brgattr.hint_ld_block2
                                        : brg->ld_block2);

    // The hints above are unclamped, and brgemm_init_tiles() returns early on
    // the ACE palette without reaching its AMX_TILES_NUM check, so the budget
    // has to be enforced here. ACE keeps A and B in zmms, only C is tiled.
    if (brg->get_num_C_tiles() > brgemm_desc_t::AMX_TILES_NUM)
        return status::unimplemented;

    // A reduction block fills the four ZMMs used by the ACE A transform.
    brg->rd_block = brg->rd_step * brgemm_desc_t::ace_zmms_per_bd_block;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    // To use fewer registers in the ACE micro-kernel: load several blocks of A
    // and keep them in registers if bd_block2 < ld_block2, otherwise load
    // several blocks of B and keep them in registers.
    brg->n_bcast_1_load = brg->bd_block2 < brg->ld_block2;

    return status::success;
}

status_t brgemm_blocking_vmm(brgemm_desc_t *brg) {
    if (brg->is_gemv) return brgemm_blocking_vmm_gemv(brg);
    const auto L1 = platform::get_per_core_cache_size(1);

    const int simd_w = is_superset(brg->isa_impl, avx512_core) ? 16 : 8;
    brg->ld_block = simd_w;
    brg->ldb = brg->load_dim / brg->ld_block;
    brg->ldb_tail = brg->load_dim % brg->ld_block;

    const int max_vpad = nstl::max(
            brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);

    // iterate ld_block2 starting from 4 to allow bd_block larger than
    // virtual padding
    int max_bcast_block {0}, min_bcast_block {0}, adj_ld_block2 {0};
    bool few_regs = utils::one_of(brg->isa_impl, avx2, avx2_vnni, avx2_vnni_2);
    bool hint_n_bcast_1_load
            = brg->brgattr.hint_loop_order == brgemm_lo_bl_1load;
    for (int try_ld_block2 = 4; try_ld_block2 > 0; --try_ld_block2) {
        adj_ld_block2 = calculate_ldb_params(brg, try_ld_block2);
        brg->n_bcast_1_load
                = (few_regs && adj_ld_block2 == 4) || hint_n_bcast_1_load;
        max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);
        const int bdb_tail = brg->bcast_dim % max_bcast_block;
        min_bcast_block = bdb_tail > 0 ? bdb_tail : max_bcast_block;
        if (min_bcast_block >= max_vpad) break;
    }
    // bcast block in brgemm kernel should be greater than virtual
    // padding to avoid possible functional issues
    if (min_bcast_block < max_vpad) return status::unimplemented;

    const int min_block = nstl::max(1, max_vpad);

    float best_bd_block_eff = 0.f;
    brg->bd_block = max_bcast_block;
    for (int bd_block = max_bcast_block; bd_block >= min_block; bd_block--) {
        // avoid msvc warning 'potential divide by zero'
        const auto bd_block_disb = (brg->bcast_dim <= 0 || bd_block == 0)
                ? 0.f
                : static_cast<float>(brg->bcast_dim)
                        / static_cast<float>(rnd_up(brg->bcast_dim, bd_block));
        const auto brgemm_microkernel_eff
                = (static_cast<float>(adj_ld_block2)
                          * static_cast<float>(bd_block))
                / static_cast<float>(
                        ((adj_ld_block2) + bd_block) * max_bcast_block);
        const auto bd_block_eff = bd_block_disb * brgemm_microkernel_eff;

        float block_foot_print = static_cast<float>(brg->typesize_A)
                * static_cast<float>(bd_block)
                * static_cast<float>(brg->reduce_dim);
        if (block_foot_print <= static_cast<float>(L1)
                && (bd_block_eff > best_bd_block_eff)) {
            brg->bd_block = bd_block;
            best_bd_block_eff = bd_block_eff;
        }
    }
    brg->bdb = brg->bcast_dim / brg->bd_block;
    brg->bdb_tail = brg->bcast_dim % brg->bd_block;

    const int rd_unroll = 4;
    const bool req_emulation = brg->isa_impl != avx2_vnni_2
            && IMPLICATION(brg->is_fp8, brg->fp8_with_f16_vnni_block);
    const data_type_t rd_block_dt
            = get_mac_emu_data_type(brg->dt_a, brg->isa_impl, req_emulation);
    if (rd_block_dt == dnnl_data_type_undef) return status::unimplemented;
    const int vnni_granularity
            = static_cast<int>(data_type_vnni_granularity(rd_block_dt));
    brg->rd_block = rd_unroll * vnni_granularity;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    brg->is_M_tail = false;
    // avx2_vnni_2 kernel with xf16 data type requires blocked weights.
    if (brg->isa_impl == avx2_vnni_2 && brg->is_xf16()
            && brg->LDB % brg->ld_block > 0)
        return status::unimplemented;

    return status::success;
}

status_t brgemm_blocking(brgemm_desc_t *brg) {
    const bool req_emulation = brg->isa_impl != avx2_vnni_2
            && IMPLICATION(brg->is_fp8, brg->fp8_with_f16_vnni_block);
    const data_type_t ld_step_compute_dt
            = get_mac_emu_data_type(brg->dt_b, brg->isa_impl, req_emulation);
    brg->ld_step = brg->is_f16_b_non_amx_vnni()
            ? 2
            : static_cast<int>(data_type_vnni_granularity(ld_step_compute_dt));
    const data_type_t rd_step_compute_dt
            = get_mac_emu_data_type(brg->dt_b, brg->isa_impl,
                    IMPLICATION(brg->is_fp8, brg->fp8_with_f16_vnni_block));
    brg->rd_step
            = static_cast<int>(data_type_vnni_granularity(rd_step_compute_dt));

    set_isa_impl(brg);
    if (brg->isa_impl == isa_undef) return status::unimplemented;
    assert(!brg->is_dgmm); // should not be called from brdgmm
    if (brg->is_dgmm) return status::unimplemented;
    set_brg_vmm(brg);
    if (!(brg->is_tmm || brg->is_zmm || brg->is_ymm))
        return status::unimplemented;

    if (brg->is_ace())
        CHECK(brgemm_blocking_ace(brg));
    else if (brg->is_tmm)
        CHECK(brgemm_blocking_tmm(brg));
    else
        CHECK(brgemm_blocking_vmm(brg));

    if (!IMPLICATION(brg->brgattr.LDB2 == 0, brg->load_dim <= brg->LDB))
        return status::invalid_arguments;

    brg->LDA2 = (brg->brgattr.LDA2 != 0) ? brg->brgattr.LDA2 : brg->LDA;
    brg->LDB2 = (brg->brgattr.LDB2 != 0) ? brg->brgattr.LDB2 : brg->LDB;
    brg->LDC2_M = (brg->brgattr.LDC2_M != 0) ? brg->brgattr.LDC2_M : brg->LDC;
    brg->LDC2_N
            = (brg->brgattr.LDC2_N != 0) ? brg->brgattr.LDC2_N : brg->ld_block;

    brg->is_blocked = (brg->LDA2 != brg->LDA || brg->LDB2 != brg->LDB
            || brg->LDC2_M != brg->LDC || brg->LDC2_N != brg->ld_block);

    if (!IMPLICATION(brg->is_blocked, brg->layout == brgemm_row_major))
        return status::invalid_arguments;

    return status::success;
}

status_t brdgmm_blocking(brgemm_desc_t *brg) {

    if (brg->isa_impl == isa_undef) return status::unimplemented;

    set_brg_vmm(brg); // Needed to dispatch into the right kernel later.
    const int max_vregs = isa_num_vregs(brg->isa_impl);

    const int simd_w = isa_max_vlen(brg->isa_impl) / brg->typesize_C;
    const bool is_avx2_vnni_2_xf16
            = brg->is_xf16() && brg->isa_impl == avx2_vnni_2;

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
    // for avx2_vnni_2_xf16, instead of processing a n_block1 at once, it is
    // processed as even/odd pair.
    const int n_block1_num_steps = is_avx2_vnni_2_xf16 ? 2 : 1;
    n_block1 = n_block1_num_steps * simd_w;
    nb_n_block1 = div_up(N, n_block1);
    n_block1_tail = N % n_block1;

    const int max_n_block2_vmms = 4;
    const int max_n_block2 = max_n_block2_vmms / n_block1_num_steps;
    n_block2 = static_cast<int>(nstl::min<dim_t>(max_n_block2, nb_n_block1));

    const int aux_vregs
            = jit_brdgmm_kernel_t<Xbyak::Zmm>::get_aux_vmm_count(*brg);
    const int compute_vregs
            = jit_brdgmm_kernel_t<Xbyak::Zmm>::get_compute_vmm_count(*brg);
    const int bf16_emu_vregs = brg->is_bf16_emu * 4;
    const int postops_regs = brg->attr()
            ? injector::aux_vec_count(
                      brg->attr()->post_ops_, brg->isa_impl, true)
            : 0;

    const int max_acc_vmms = max_vregs
            - nstl::max(postops_regs,
                    nstl::max(compute_vregs + aux_vregs, bf16_emu_vregs));

    if (brg->brgattr.hint_bs_group > 1) {
        // Check if we can actually apply bs grouping
        const auto min_possible_m_block2
                = (max_acc_vmms / (2 * n_block1_num_steps)
                          - brg->brgattr.hint_bs_group + 1)
                / 2;
        if (min_possible_m_block2 < 1) brg->bs_group = 1;
    }

    if (brg->bs_group > 1) n_block2 = n_block2 % 2 == 0 ? 2 : 1;

    nb_n_block2 = div_up(nb_n_block1, n_block2);
    n_block2_tail = nb_n_block1 % n_block2;

    m_block1 = 1;
    nb_m_block1 = M / m_block1;
    m_block1_tail = M % m_block1;

    m_block2 = static_cast<int>(nstl::min<dim_t>(nb_m_block1,
            brg->bs_group > 1
                    ? (max_acc_vmms / (n_block2 * n_block1_num_steps)
                              - brg->bs_group + 1)
                            / 2
                    : max_acc_vmms / (n_block2 * n_block1_num_steps)));
    assert(m_block2 > 0);
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

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = static_cast<int>(types::data_type_size(brg->dt_a));
    brg->typesize_B = static_cast<int>(types::data_type_size(brg->dt_b));
    brg->typesize_C = static_cast<int>(types::data_type_size(brg->dt_c));
    brg->typesize_D = static_cast<int>(types::data_type_size(brg->dt_d));

    brg->isa_user = isa;

    brg->is_bf32 = is_bf32
            && utils::one_of(brg->isa_user, isa_undef, avx512_core_amx)
            && mayiuse(avx512_core_amx);

    set_isa_impl(brg);
    // ACE hardware can resolve bf16/int8 to ACE even without TMUL.
    // Query the hardware instead of assuming ACE implies no TMUL.
    const auto has_tmul = [&](cpu_isa_t amx_isa) {
        return is_superset(brg->isa_impl, amx_isa)
                || (is_superset(brg->isa_impl, avx10_2_ace)
                        && mayiuse(amx_isa));
    };
    brg->is_int8_tmm = brg->is_int8 && has_tmul(avx512_core_amx);
    brg->is_bf16_tmm = brg->is_bf16 && has_tmul(avx512_core_amx);
    brg->is_f16_tmm = brg->is_f16 && has_tmul(avx512_core_amx_fp16);
    brg->is_fp8_tmm = brg->is_fp8 && has_tmul(avx512_core_amx_fp16);

    brg->has_int8_vnni = isa_has_int8_vnni(brg->isa_impl);

    set_brg_vmm(brg); // TODO: Investigate if it is really needed here.
    brg->req_src_s8_shift = brg->is_int8 && brg->dt_a == data_type::s8
            && !isa_has_s8s8(brg->isa_impl);
    // s8s8 compensation could be applied in per_mn_compensation kernel,
    // in that case brgemm kernel should supress this flag
    // to avoid double compensation.
    brg->req_s8s8_compensation
            = brg->req_src_s8_shift && !brg->with_per_mn_compensation;

    brg->LDA = (brg->is_row_major()) ? LDA : LDB;
    brg->is_runtime_lda = (brg->is_row_major()) ? is_runtime_value(LDA)
                                                : is_runtime_value(LDB);
    brg->LDB = (brg->is_row_major()) ? LDB : LDA;
    brg->is_runtime_ldb = (brg->is_row_major()) ? is_runtime_value(LDB)
                                                : is_runtime_value(LDA);
    brg->LDC = LDC;
    brg->LDD = LDC;
    brg->is_runtime_ldc = brg->is_runtime_ldd = is_runtime_value(LDC);

    brg->bcast_dim = (brg->is_row_major()) ? M : N;
    brg->load_dim = (brg->is_row_major()) ? N : M;
    brg->reduce_dim = K;

    brg->bd_block2 = 0;
    brg->bdb2 = 0;
    brg->bdb2_tail = 0;

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

    brg->typesize_A = static_cast<int>(types::data_type_size(brg->dt_a));
    brg->typesize_B = static_cast<int>(types::data_type_size(brg->dt_b));
    brg->typesize_C = static_cast<int>(types::data_type_size(brg->dt_c));
    brg->typesize_D = static_cast<int>(types::data_type_size(brg->dt_d));

    brg->isa_user = isa;
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) && one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core),
                avx512_core, is_isa_ok(avx2), avx2);
    } else if (brg->is_bf16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_bf16),
                avx512_core_bf16, is_isa_ok(avx2_vnni_2), avx2_vnni_2);
    } else if (brg->is_f16) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx512_core_fp16),
                avx512_core_fp16, is_isa_ok(avx2_vnni_2), avx2_vnni_2,
                is_isa_ok(avx10_2), avx10_2);
    } else if (brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(avx10_2), avx10_2,
                is_isa_ok(avx512_core_vnni), avx512_core_vnni,
                is_isa_ok(avx2_vnni_2), avx2_vnni_2, is_isa_ok(avx2_vnni),
                avx2_vnni);
    }

    brg->req_src_s8_shift = brg->is_int8 && brg->dt_a == data_type::s8
            && !isa_has_s8s8(brg->isa_impl);
    brg->req_s8s8_compensation = brg->req_src_s8_shift;

    brg->is_dgmm = true;

    brg->LDA = LDA;
    brg->LDC = LDC;
    brg->LDD = LDC;

    brg->bcast_dim = M;
    brg->load_dim = N;

    return status::success;
}

} // namespace brgemm_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
