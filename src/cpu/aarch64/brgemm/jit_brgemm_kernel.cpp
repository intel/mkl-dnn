/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024-2025 FUJITSU LIMITED
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
#include <memory>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/platform.hpp"

#define GET_OFF(field) (uint32_t) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) \
    (uint32_t) offsetof(brgemm_batch_element_t, field)

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

struct jit_brgemm_kernel_t : public jit_generator_t {
    jit_brgemm_kernel_t(const brgemm_desc_t &abrg)
        : jit_generator_t(nullptr, MAX_CODE_SIZE, true, sve_512)
        , brg(abrg)
        , postops_injector_(nullptr)
        , use_mmla(abrg.brgattr.use_mmla)
        , use_mmla_packed_a(abrg.brgattr.use_mmla_packed_a)
        , use_gemv_path(abrg.is_gemv && !abrg.brgattr.use_mmla)
        , max_effective_vregs(
                  max_vregs - ((brg.is_int8 && !brg.has_int8_vnni) ? 2 : 0)) {

        // The implementation uses is_superset(), is_subset() utilities.
        // So avoid isa_all, isa_undef in these comparisions.
        assert(!utils::one_of(brg.isa_impl, isa_all, isa_undef));
        const int is_ldb2_tail = brg.ldb2_tail ? 1 : 0;
        const int is_ldb_tail = brg.ldb_tail ? 1 : 0;
        const int n_ldb_loops = use_mmla
                ? brg.ldb2 + is_ldb2_tail
                : brg.ldb2 + is_ldb2_tail + is_ldb_tail;
        is_ldb_loop_ = n_ldb_loops > 1;

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md);

            static const bcast_set_t enabled_bcast_strategy
                    = {broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::per_mb_w,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::no_broadcast};
            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    static_cast<size_t>(Xbyak_aarch64::ZReg(1).getIdx()),
                    XReg(14), XReg(15), XReg(13), preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, static_cast<size_t>(brg.ldb_tail),
                    PReg(ld_tail_mask.getIdx()), use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    XReg(this->param1.getIdx()), enabled_bcast_strategy,
                    rhs_sp};

            postops_injector_ = utils::make_unique<po_injector_t>(
                    this, brg.attr->post_ops_, bsp);

            with_binary_non_scalar_bcast_ = binary_injector::
                    any_binary_postop_rhs_non_scalar_broadcast(
                            brg.attr->post_ops_, dst_md_wrapper);
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    brgemm_desc_t brg;

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<sve>;
    std::unique_ptr<po_injector_t> postops_injector_;

    // Register decomposition
    const XReg param1 = XReg(7); //abi_param1_x64=r7

    const XReg reg_C = x15;
    const XReg reg_aux_C = x14;

    const XReg reg_addr_batch = x13;
    const XReg reg_A = x13;
    const XReg reg_B = x12;

    const XReg reg_aux_A = x11;
    const XReg reg_aux_B = x10;
    const XReg reg_aux_A_vpad = reg_aux_A;

    const XReg reg_bdb_loop = x9;
    const XReg reg_ldb_loop = x8;

    const XReg reg_stride_lda = reg_bdb_loop;
    const XReg reg_stride_ldb = reg_ldb_loop;
    const XReg reg_stride_ld_block = reg_ldb_loop;
    const XReg reg_s8_input_shift = reg_bdb_loop;
    const XReg reg_zp_a_input_shift = reg_bdb_loop;

    const XReg reg_BS_loop = x0;
    const XReg reg_rdb_loop = x3;
    const XReg reg_BS = x1; //from jit_generator_t.hpp in x64

    const XReg reg_a_offset = x2;
    const XReg reg_b_offset = x6;

    const XReg reg_aux1_A = x4;
    const XReg reg_aux1_batch = reg_aux1_A;

    const XReg reg_aux1_B = x7; //from jit_generator_t.hpp in x64

    const XReg reg_offs_batch = x5;
    const XReg reg_strd_batch = reg_rdb_loop;

    const XReg reg_bias = reg_rdb_loop;
    const XReg reg_scales = reg_rdb_loop;
    const XReg reg_aux_bias = reg_rdb_loop;
    const XReg reg_dst_scales = reg_rdb_loop;
    const XReg reg_zp_comp_a = reg_rdb_loop;
    const XReg reg_aux_zp_comp_a = reg_rdb_loop;
    const XReg reg_zp_comp_b = reg_rdb_loop;
    const XReg reg_aux_zp_comp_b = reg_rdb_loop;
    const XReg reg_zp_c_values = reg_rdb_loop;
    const XReg reg_aux_zp_c_values = reg_rdb_loop;

    const XReg reg_aux_scales = reg_aux_B;
    const XReg reg_aux_dst_scales = reg_aux_B;
    const XReg reg_do_post_ops = reg_rdb_loop;
    const XReg reg_do_comp = reg_rdb_loop;
    const XReg reg_skip_accm = reg_rdb_loop;
    const XReg reg_tmp_gpr = reg_rdb_loop;
    const XReg reg_ptr_sum_scale = reg_rdb_loop;
    const XReg reg_ptr_sum_zp = reg_bdb_loop;
    const XReg reg_zp_a_val = reg_rdb_loop;

    const XReg reg_buf = reg_rdb_loop;
    const XReg reg_compensation = reg_bias;
    const XReg reg_aux_compensation = reg_aux_bias;

    const XReg reg_D = reg_aux_A;
    const XReg reg_aux_D = reg_BS_loop;

    const XReg reg_tmp_ = x16;

    const XReg reg_stride_bytes_A = x17;

    constexpr static int origin_offs_batch_offs_ = 0;
    constexpr static int origin_strd_batch_offs_ = 0;
    constexpr static int reg_bias_offs_ = 8;
    constexpr static int reg_aux_bias_offs_ = 16;
    constexpr static int reg_do_post_ops_offs_ = 24;
    constexpr static int reg_D_offs_ = 32;
    constexpr static int reg_aux_D_offs_ = 40;
    constexpr static int reg_scales_offs_ = 48;
    constexpr static int reg_aux_scales_offs_ = 56;
    constexpr static int reg_bdb_loop_offs_ = 64;
    constexpr static int reg_ldb_loop_offs_ = 72;
    constexpr static int reg_buf_offs_ = 80;
    constexpr static int reg_comp_offs_ = reg_buf_offs_;
    constexpr static int reg_aux_comp_offs_ = 88;
    constexpr static int abi_param1_offs_ = 96;
    constexpr static int reg_zp_comp_a_offs_ = 104;
    constexpr static int reg_aux_zp_comp_a_offs_ = 112;
    constexpr static int reg_zp_comp_b_offs_ = 120;
    constexpr static int reg_aux_zp_comp_b_offs_ = 128;
    constexpr static int reg_zp_c_values_offs_ = 136;
    constexpr static int reg_aux_zp_c_values_offs_ = 144;
    constexpr static int reg_data_C_ptr_ = 152;
    constexpr static int reg_skip_accm_offs_ = 160;
    constexpr static int reg_zp_a_val_offs_ = 168;
    constexpr static int reg_do_comp_offs_ = 176;
    constexpr static int reg_dst_scales_offs_ = 184;
    constexpr static int stack_space_needed_ = 192;

    bool is_ldb_loop_ = false;
    bool with_binary_non_scalar_bcast_ = false;
    constexpr static int max_vregs = 32;
    const bool use_mmla;
    const bool use_mmla_packed_a;
    const bool use_gemv_path;
    const int max_effective_vregs;

    int mmla_bd_blk() const { return brgemm_utils::mmla_bd_blk(); }
    int mmla_ld_regs_per_block() const { return brgemm_utils::mmla_ld_blk(); }
    int mmla_rd_chunk_elems() const {
        return brgemm_utils::mmla_rd_chunk_elems(brg.typesize_A);
    }
    int mmla_rd_chunks_per_block() const {
        return brgemm_utils::mmla_rd_chunks_per_block(brg.typesize_A);
    }
    int mmla_rd_block() const { return brgemm_utils::mmla_rd_block(); }
    PReg rd_tail_mask = PReg(2);
    PReg ld_tail_mask = PReg(3);

    ZReg accm(int ld_block, int bd, int ld) const {
        // Starts at the highest (e.g. 31), descending and using ld_block * bd_block registers as accumulators
        return ZReg(max_effective_vregs - 1 - (bd * ld_block + ld));
    }
    ZReg accm_mmla(int ld_block, int bd_pair, int ld4) const {
        return accm(ld_block, mmla_bd_blk() * bd_pair + (ld4 % mmla_bd_blk()),
                ld4 / mmla_bd_blk());
    }
    int mmla_bd_pairs(int bd_block) const {
        return div_up(bd_block, mmla_bd_blk());
    }

    ZReg bcst(int bd = 0) const {
        if (n_bcast_1_load) {
            // Indexed FMLA/DOT instructions are only defined in the architecture for z0-z7
            // https://developer.arm.com/documentation/111108/2025-12/SVE-Instructions/FMLA--indexed---Floating-point-fused-multiply-add-by-indexed-element-
            assert(bd <= 7);
            return ZReg(bd);
        } else {
            return ZReg(0);
        }
    }

    ZReg load(int ld = 0) const {
        if (n_bcast_1_load) {
            // For odd ld, B aliases z_inp_shift() after compensation is complete.
            return ZReg(brg.bd_block + ld % 2);
        } else {
            // Starts off from lowest accm register, and continues descending
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - ld;
            assert(idx > 0);
            return ZReg(idx);
        }
    }
    const ZReg &z_tmp_1() const noexcept { return this->z0; }
    const ZReg &z_tmp_2() const noexcept { return this->z1; }
    const ZReg &z_tmp_3() const noexcept { return this->z2; }
    const ZReg &z_tail_mask() const noexcept { return this->z1; }
    ZReg z_one_bytes() const noexcept {
        if (n_bcast_1_load) {
            return ZReg(brg.bd_block + 3);
        } else {
            return this->z3;
        }
    }
    ZReg z_zp_a_shift() const noexcept {
        if (n_bcast_1_load) {
            return ZReg(brg.bd_block + 2);
        } else {
            return this->z2;
        }
    }
    ZReg z_inp_shift() const noexcept {
        if (n_bcast_1_load) {
            return ZReg(brg.bd_block + 1);
        } else {
            return this->z1;
        }
    }

    ZReg int8_ones_words() const noexcept { return ZReg(max_vregs - 1); }
    ZReg int8_dot_product_temp() const noexcept { return ZReg(max_vregs - 2); }

    void load_data(data_type_t type_in, const Xbyak_aarch64::ZReg &vmm,
            const Xbyak_aarch64::XReg &reg_addr, int load_size);

    void cvt2ps(data_type_t type_in, const ZReg zmm_in, const XReg &addr,
            bool mask_flag, bool store, PReg ktail_mask, const int offset,
            const int base_offset); //for only memory operand

    void ldb_regs_shift(int ld_block2, bool is_tail = false);
    void advance_bd_block2_post_op_regs(int bd_block2);

    void copy_post_ops_stack_values_to_aux(bool is_reg_tail);
    void read_params();
    void zero_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);

    void store_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);
    void store_accumulators_without_post_ops(int bd_block, int ld_block,
            bool is_ld_tail, bool mmla_deinterleave_on_store = false);
    void store_accumulators_apply_post_ops(int bd_block, int ld_block,
            int ldb_and_bdb_offset, bool is_ld_tail);
    void apply_compensation(int bd_block, int ld_block, bool is_ld_tail);
    void apply_alpha_beta(int bd_block, int ld_block, bool is_ld_tail);
    void apply_post_ops(int bd_block, int ld_block2, int ldb_and_bdb_offset,
            bool is_ld_tail);
    void sum_into_one_lane(int bd_block, int ld_block2, bool is_ld_tail);
    void restore_A_B_matrices();
    void set_A_B_matrices();

    void compute_int8_compensation(int rd_loop, int bd_b, int bd_e,
            int bd_block, int ld_block2, bool is_ld_tail, int vpad);

    // Load at most a word from the left hand side for broadcasting
    void load_word_for_bcast(const ZReg &dst, bool is_tail, const XReg &base,
            const int32_t offset_elements, const data_type_t dt,
            const XReg &tmp);

    // Note that we load and bcast words because they are our granule size
    void load_A_word_for_bcast(int32_t &base_offset, const ZReg &dst,
            bool is_tail, const XReg &reg_A_ptr, const int32_t bd,
            const int32_t rd, const XReg &tmp);

    // Load at most a quadword from the left hand side for broadcasting
    void load_quadword_for_bcast(const ZReg &dst, const XReg &base,
            const PReg &mask, const XReg &reg_stride_bytes,
            const int32_t stride_bytes, const int32_t n, const data_type_t dt);

    void mmla_deinterleave_accumulators(int bd_block, int ld_block2);
    int mmla_rd_chunk_B_offset(int ld_blocks) const noexcept;
    int mmla_rdb_B_offset(int ld_blocks) const noexcept;
    void mmla_load_packed_a(int rd_chunk, int bd_pair, const ZReg &z_a,
            const ZReg &z_a_tmp, int bd_block, bool row0_valid,
            bool row1_valid);
    void mmla_load_plain_a(int rd_chunk, int bd_pair, const ZReg &z_a,
            const ZReg &z_a_tmp, int rd_tail, bool row0_valid, bool row1_valid);
    void mmla_load_b(const XReg &b_base, int ld, int z_idx);

    // FMLA/DOT
    void dot_product(ZReg v_acc, ZReg v_a, ZReg v_b);
    // FMLA/DOT with indexed b vector
    void dot_product(ZReg v_acc, ZReg v_a, ZReg v_b, const int16_t index);

    // MMLA
    void mmla(ZReg v_acc, ZReg v_a, ZReg v_b);

    void gemm_microkernel(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail, int vpad, int rows_for_rd_tail);
    void gemm_microkernel_mmla(bool is_bdb_tail, int ld_block,
            int b_panel_ld_block, bool is_rd_tail, bool advance_a_b, int vpad);
    // GEMV microkernel is distinct from GEMM in that it loads vectors from A and B
    // and assumes that we will sum all the elements after the microkernel
    void gemv_microkernel(
            bool is_bdb_tail, int ld_block2, bool is_rd_tail, int vpad);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
            bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
            bool skip_accumulation);
    void bdb_loop();

    void generate() override;

    int A_offset(int bd, int rd) const noexcept;
    int B_offset(int ld, int rd) const noexcept;
    int C_offset(int bd, int ld) const noexcept;
    int D_offset(int bd, int ld) const noexcept;
    int po_offset(int bd, int ld) const noexcept;

    int rdb_A_offset(int bd_block) const noexcept;
    int rdb_B_offset(int ld_block2) const noexcept;

    int ldb_B_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_C_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_D_offset(int ld_block2, bool is_tail = false) const noexcept;
    int ldb_po_offset(int ld_block2, bool is_tail = false) const noexcept;

    int bdb_A_offset(int bd_block2) const noexcept;
    int bdb_C_offset(int bd_block2) const noexcept;
    int bdb_D_offset(int bd_block2) const noexcept;
    int bdb_po_offset(int bd_block2) const noexcept;

    int bias_offset(int ld, bool is_tail = false) const noexcept;
    int oc_logical_offset(int ld, bool is_tail = false) const noexcept;

    int compensations_offset(int ld, bool is_tail = false) const noexcept;
    int bdb_compensation_offset(int bd_block2) const noexcept;
    int compensation_vpad_offset(int ld, int bd) const noexcept;
    int scales_offset(int ld, bool is_tail = false) const noexcept;
    int zp_comp_a_offset(int ld, bool is_tail = false) const noexcept;
    int zp_comp_a_vpad_offset(int ld, int bd) const noexcept;
    int bdb_zp_comp_a_offset(int bd_block2) const noexcept;
    int zp_comp_b_offset(int bd) const noexcept;
    int bdb_zp_comp_b_offset(int bd_block2) const noexcept;
    int zp_c_values_offset(int ld, bool is_tail = false) const noexcept;

    bool n_bcast_1_load = false;
    bool vpad_exist = false;
    bool need_comp_pads = false;
};

int jit_brgemm_kernel_t::A_offset(int bd, int rd) const noexcept {
    return brg.typesize_A * (bd * brg.LDA + rd);
}
int jit_brgemm_kernel_t::B_offset(int ld, int rd) const noexcept {
    if (use_gemv_path) {
        return (ld + rd * brg.ld_block2) * brg.typesize_B;
    } else {
        const int data_vnni_granularity = brg.ld_step;
        const int rdb0 = rd / data_vnni_granularity;
        return brg.typesize_B
                * (rdb0 * data_vnni_granularity * brg.LDB
                        + data_vnni_granularity * ld * brg.ld_block);
    }
}
int jit_brgemm_kernel_t::C_offset(int bd, int ld) const noexcept {
    if (use_gemv_path) {
        return (ld + bd * brg.ld_block2) * brg.typesize_C;
    } else {
        return brg.typesize_C * (bd * brg.LDC + ld * brg.ld_block);
    }
}
int jit_brgemm_kernel_t::D_offset(int bd, int ld) const noexcept {
    if (use_gemv_path) {
        return (ld + bd * brg.ld_block2) * brg.typesize_D;
    } else {
        return brg.typesize_D * (bd * brg.LDD + ld * brg.ld_block);
    }
}
int jit_brgemm_kernel_t::po_offset(int bd, int ld) const noexcept {
    return bd * brg.LDD + ld * brg.ld_block;
}

int jit_brgemm_kernel_t::mmla_rd_chunk_B_offset(int ld_blocks) const noexcept {
    return ld_blocks * mmla_ld_regs_per_block()
            * static_cast<int>(simd_bytes(brg.isa_impl));
}

int jit_brgemm_kernel_t::mmla_rdb_B_offset(int ld_blocks) const noexcept {
    return mmla_rd_chunk_B_offset(ld_blocks) * mmla_rd_chunks_per_block();
}

int jit_brgemm_kernel_t::rdb_A_offset(int bd_block) const noexcept {
    if (use_mmla) {
        if (use_mmla_packed_a)
            return brg.typesize_A * mmla_bd_blk() * mmla_rd_block();
        return brg.typesize_A * mmla_rd_block();
    }

    return brg.typesize_A * brg.rd_block;
}
int jit_brgemm_kernel_t::rdb_B_offset(int ld_block2) const noexcept {
    if (use_mmla) return mmla_rdb_B_offset(ld_block2);

    return brg.typesize_B * brg.rd_block * brg.LDB;
}

int jit_brgemm_kernel_t::ldb_B_offset(
        int ld_block2, bool is_tail) const noexcept {
    if (use_mmla) {
        return mmla_rdb_B_offset(brgemm_utils::mmla_b_panel_ld_block2(brg))
                * div_up(brg.reduce_dim, mmla_rd_block());
    }

    return (is_tail) ? brg.typesize_B * brg.ldb_tail * brg.ld_step
                     : brg.typesize_B * ld_block2 * brg.ld_block * brg.ld_step;
}
int jit_brgemm_kernel_t::ldb_C_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_t::ldb_D_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}
int jit_brgemm_kernel_t::ldb_po_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld_block2 * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_A_offset(int bd_block2) const noexcept {
    if (use_mmla && use_mmla_packed_a)
        return bd_block2 * rnd_up(brg.reduce_dim, mmla_rd_chunk_elems())
                * brg.bd_block * brg.typesize_A;

    return brg.typesize_A * bd_block2 * brg.bd_block * brg.LDA;
}
int jit_brgemm_kernel_t::bdb_C_offset(int bd_block2) const noexcept {
    return brg.typesize_C * bd_block2 * brg.bd_block * brg.LDC;
}
int jit_brgemm_kernel_t::bdb_D_offset(int bd_block2) const noexcept {
    return brg.typesize_D * bd_block2 * brg.bd_block * brg.LDD;
}
int jit_brgemm_kernel_t::bdb_po_offset(int bd_block2) const noexcept {
    return bd_block2 * brg.bd_block * brg.LDD;
}

int jit_brgemm_kernel_t::bias_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::oc_logical_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld * brg.ld_block;
}

int jit_brgemm_kernel_t::compensations_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_compensation_offset(int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

int jit_brgemm_kernel_t::compensation_vpad_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

int jit_brgemm_kernel_t::scales_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.is_oc_scale * sizeof(float) * brg.ldb_tail
                     : brg.is_oc_scale * sizeof(float) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::zp_comp_a_offset(int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

int jit_brgemm_kernel_t::bdb_zp_comp_a_offset(int bd_block2) const noexcept {
    return sizeof(int32_t) * bd_block2 * brg.bd_block * brg.LDB;
}

int jit_brgemm_kernel_t::zp_comp_a_vpad_offset(int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

int jit_brgemm_kernel_t::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

int jit_brgemm_kernel_t::bdb_zp_comp_b_offset(int bd_block2) const noexcept {
    return zp_comp_b_offset(bd_block2 * brg.bd_block);
}

int jit_brgemm_kernel_t::zp_c_values_offset(
        int ld, bool is_tail) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                         : sizeof(int32_t) * ld * brg.ld_block;
    }

    return 0;
}

void jit_brgemm_kernel_t::load_data(data_type_t type_in,
        const Xbyak_aarch64::ZReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        int load_size) {
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: ld1w(vmm.s, P_ALL_ONE / T_z, ptr(reg_addr)); break;
        case data_type::s8:
        case data_type::u8: assert(!"unsupported\n"); break;
        case data_type::bf16: assert(!"unsupported\n"); break;
        case data_type::f16: assert(!"unsupported\n"); break;
        default: assert(!"unsupported source data type");
    }
}

void jit_brgemm_kernel_t::cvt2ps(data_type_t type_in, const ZReg zmm_in,
        const XReg &addr, bool mask_flag, bool store, PReg ktail_mask,
        const int offset, const int base_offset) {
    const auto mask = mask_flag ? ktail_mask : P_ALL_ONE;
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: {
            LD_MUL_VL(ld1w, z_tmp_1().s, mask, addr, offset - base_offset, 4);
            if (store) //Merging
                mov(zmm_in.s, ktail_mask / T_m, z_tmp_1().s);
            break;
        }
        case data_type::bf16: {
            LD_MUL_VL(ld1h, z_tmp_1().s, mask, addr, offset - base_offset, 2);
            lsl(z_tmp_1().s, z_tmp_1().s, 16);
            if (store) //Merging
                mov(zmm_in.s, ktail_mask / T_m, z_tmp_1().s);
            break;
        }
        case data_type::s8:
            LD_MUL_VL(ld1sb, z_tmp_1().s, mask, addr, offset - base_offset, 1);
            if (store) // Merging
                mov(zmm_in.s, ktail_mask / T_m, z_tmp_1().s);
            break;
        case data_type::u8: {
            LD_MUL_VL(ld1b, z_tmp_1().s, mask, addr, offset - base_offset, 1);
            if (store) // Merging
                mov(zmm_in.s, ktail_mask / T_m, z_tmp_1().s);
            break;
        }
        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) {
        scvtf(zmm_in.s, P_ALL_ONE / T_m, zmm_in.s);
    }
}

void jit_brgemm_kernel_t::ldb_regs_shift(int ld_block2, bool is_tail) {
    int C_offset = ldb_C_offset(ld_block2, is_tail);
    int D_offset = ldb_D_offset(ld_block2, is_tail);

    add_imm(reg_aux_C, reg_aux_C, C_offset, X_TMP_0);
    add_imm(reg_aux_D, reg_aux_D, D_offset, X_TMP_0);

    add_imm(reg_b_offset, reg_b_offset, ldb_B_offset(ld_block2, is_tail),
            X_TMP_0);

    if (brg.with_bias) {
        ldr_imm(reg_aux_bias, sp, reg_aux_bias_offs_);
        add_imm(reg_aux_bias, reg_aux_bias, bias_offset(ld_block2, is_tail),
                X_TMP_0);
        str_imm(reg_aux_bias, sp, reg_aux_bias_offs_);
    }
    if (brg.req_s8s8_compensation) {
        ldr_imm(reg_aux_compensation, sp, reg_aux_comp_offs_);
        add_imm(reg_aux_compensation, reg_aux_compensation,
                compensations_offset(ld_block2, is_tail), X_TMP_0);
        str_imm(reg_aux_compensation, sp, reg_aux_comp_offs_);
    }
    if (brg.with_scales) {
        ldr_imm(reg_aux_scales, sp, reg_aux_scales_offs_);
        add_imm(reg_aux_scales, reg_aux_scales,
                scales_offset(ld_block2, is_tail), X_TMP_0);
        str_imm(reg_aux_scales, sp, reg_aux_scales_offs_);
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        ldr_imm(reg_aux_zp_comp_a, sp, reg_aux_zp_comp_a_offs_);
        add_imm(reg_aux_zp_comp_a, reg_aux_zp_comp_a,
                zp_comp_a_offset(ld_block2, is_tail), X_TMP_0);
        str_imm(reg_aux_zp_comp_a, sp, reg_aux_zp_comp_a_offs_);
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        ldr_imm(reg_aux_zp_c_values, sp, reg_aux_zp_c_values_offs_);
        add_imm(reg_aux_zp_c_values, reg_aux_zp_c_values,
                zp_c_values_offset(ld_block2, is_tail), X_TMP_0);
        str_imm(reg_aux_zp_c_values, sp, reg_aux_zp_c_values_offs_);
    }
}

void jit_brgemm_kernel_t::advance_bd_block2_post_op_regs(int bd_block2) {
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        ldr_imm(reg_zp_comp_b, sp, reg_zp_comp_b_offs_);
        add_imm(reg_zp_comp_b, reg_zp_comp_b, bdb_zp_comp_b_offset(bd_block2),
                X_TMP_0);
        str_imm(reg_zp_comp_b, sp, reg_zp_comp_b_offs_);
    }
}

void jit_brgemm_kernel_t::copy_post_ops_stack_values_to_aux(bool is_reg_tail) {
    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        eor(reg_b_offset, reg_b_offset, reg_b_offset);
        if (brg.with_bias) {
            ldr_imm(reg_bias, sp, reg_bias_offs_);
            str_imm(reg_bias, sp, reg_aux_bias_offs_);
        }
        if (brg.req_s8s8_compensation) {
            ldr_imm(reg_compensation, sp, reg_comp_offs_);
            str_imm(reg_compensation, sp, reg_aux_comp_offs_);
        }
        if (brg.with_scales) {
            ldr_imm(reg_scales, sp, reg_scales_offs_);
            str_imm(reg_scales, sp, reg_aux_scales_offs_);
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            ldr_imm(reg_zp_comp_a, sp, reg_zp_comp_a_offs_);
            str_imm(reg_zp_comp_a, sp, reg_aux_zp_comp_a_offs_);
        }

        if (brg.zp_type_c != brgemm_broadcast_t::none) {
            ldr_imm(reg_zp_c_values, sp, reg_zp_c_values_offs_);
            str_imm(reg_zp_c_values, sp, reg_aux_zp_c_values_offs_);
        }
    }
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        ldr_imm(reg_zp_comp_b, sp, reg_zp_comp_b_offs_);
        str_imm(reg_zp_comp_b, sp, reg_aux_zp_comp_b_offs_);
    }
}

void jit_brgemm_kernel_t::read_params() {
    Label label_done;

    if (brg.with_binary) { str_imm(param1, sp, abi_param1_offs_); }

    if (brg.type == brgemm_addr) {
        ldr_imm(reg_addr_batch, param1, GET_OFF(batch));
    } else {
        if (brg.layout == brgemm_row_major) {
            ldr_imm(reg_A, param1, GET_OFF(ptr_A));
            ldr_imm(reg_B, param1, GET_OFF(ptr_B));
        } else {
            ldr_imm(reg_A, param1, GET_OFF(ptr_B));
            ldr_imm(reg_B, param1, GET_OFF(ptr_A));
        }

        if (brg.type == brgemm_offs) {
            ldr_imm(reg_offs_batch, param1, GET_OFF(batch));
            str_imm(reg_offs_batch, sp, origin_offs_batch_offs_);
        } else {
            ldr_imm(reg_strd_batch, param1, GET_OFF(batch));
            str_imm(reg_strd_batch, sp, origin_strd_batch_offs_);
        }
    }

    ldr(reg_C, ptr(param1, GET_OFF(ptr_C)));
    ldr(reg_D, ptr(param1, GET_OFF(ptr_D)));
    if (brg.brgattr.max_bs > 1) ldr(reg_BS, ptr(param1, GET_OFF(BS)));

    if (!use_mmla) mov_imm(reg_stride_bytes_A, brg.typesize_A * brg.LDA);

    // ptr_buf is re-used for passing compensations for
    // brg.req_s8s8_compensation case
    if (brg.req_s8s8_compensation) {
        ldr(reg_buf, ptr(param1, GET_OFF(ptr_buf)));
        str(reg_buf, ptr(sp, reg_buf_offs_));
    }

    if (brg.with_bias) {
        ldr(reg_bias, ptr(param1, GET_OFF(ptr_bias)));
        str(reg_bias, ptr(sp, reg_bias_offs_));
    }
    if (brg.with_scales) {
        ldr(reg_scales, ptr(param1, GET_OFF(ptr_scales)));
        str(reg_scales, ptr(sp, reg_scales_offs_));
    }

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        ldr(reg_zp_comp_a, ptr(param1, GET_OFF(a_zp_compensations)));
        str(reg_zp_comp_a, ptr(sp, reg_zp_comp_a_offs_));
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        ldr(reg_zp_comp_b, ptr(param1, GET_OFF(b_zp_compensations)));
        str(reg_zp_comp_b, ptr(sp, reg_zp_comp_b_offs_));
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        ldr(reg_zp_c_values, ptr(param1, GET_OFF(c_zp_values)));
        str(reg_zp_c_values, ptr(sp, reg_zp_c_values_offs_));
    }

    if (brg.with_dst_scales) {
        ldr(reg_dst_scales, ptr(param1, GET_OFF(ptr_dst_scales)));
        str(reg_dst_scales, ptr(sp, reg_dst_scales_offs_));
    }

    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.with_dst_scales,
            brg.req_s8s8_compensation, has_zero_points);
    if (are_post_ops_applicable) {
        ldr(reg_do_post_ops, ptr(param1, GET_OFF(do_post_ops)));
        str(reg_do_post_ops, ptr(sp, reg_do_post_ops_offs_));
    }

    if (brg.brgattr.generate_skip_accumulation) {
        ldr(reg_skip_accm, ptr(param1, GET_OFF(skip_accm)));
        str(reg_skip_accm, ptr(sp, reg_skip_accm_offs_));
    }

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        ldr(reg_zp_a_val, ptr(param1, GET_OFF(zp_a_val)));
        str(reg_zp_a_val, ptr(sp, reg_zp_a_val_offs_));
    }

    if (brg.is_int8 && (brg.req_s8s8_compensation || has_zero_points)) {
        ldr(reg_do_comp, ptr(param1, GET_OFF(do_apply_comp)));
        str(reg_do_comp, ptr(sp, reg_do_comp_offs_));
    }
}

void jit_brgemm_kernel_t::zero_accumulators(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_ld_tail, bool skip_accumulation) {
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const int acc_bd_block
            = use_mmla ? rnd_up(bd_block, mmla_bd_blk()) : bd_block;
    const bool need_to_apply_beta = brg.beta != 0.f;
    int base_offset = 0;
    auto x_addr = reg_aux_C;
    for_(int bd = 0; bd < acc_bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto zmm = accm(ld_block2, bd, ld);
        // This part is moved here from apply_alpha_beta function so that fadd instruction can be avoided.
        // This is also required only when K is blocked.
        if (need_to_apply_beta && !brg.is_gemv && !use_mmla) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;

            const int offset = C_offset(bd, ld);

            if (!use_mul_vl(offset - base_offset, 4, cpu_sveLen)) {
                add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            LD_MUL_VL(ld1w, zmm.s, k_mask, x_addr, offset - base_offset, 4);

            const bool need_init_beta_vmm = brg.beta != 1.f;
            auto vmm_beta = z_tail_mask();
            if (need_init_beta_vmm) {
                auto wreg_tmp = WReg(reg_tmp_gpr.getIdx());
                mov_imm(wreg_tmp, float2int(static_cast<float>(brg.beta)));
                dup(vmm_beta.s, wreg_tmp);
                fmul(zmm.s, zmm.s, vmm_beta.s);
            }
        } else {
            uni_clear(zmm);
        }
    }
}

void jit_brgemm_kernel_t::apply_alpha_beta(
        int bd_block, int ld_block2, bool is_ld_tail) {
    const bool apply_alpha = brg.alpha != 1.f;
    const bool apply_beta = brg.beta != 0.f;
    const bool dq2ps_required = brg.is_int8 && (apply_alpha || brg.beta != 1.f);
    const bool apply_beta_after_norm = apply_beta && use_mmla;

    auto vmm_alpha = z_tmp_1();
    if (apply_alpha) {
        auto wreg_tmp = WReg(reg_tmp_gpr.getIdx());
        mov_imm(wreg_tmp, float2int(static_cast<float>(brg.alpha)));
        dup(vmm_alpha.s, wreg_tmp);
    }
    auto vmm_beta = z_tmp_3();
    if (apply_beta_after_norm && brg.beta != 1.f) {
        auto wreg_tmp = WReg(reg_tmp_gpr.getIdx());
        mov_imm(wreg_tmp, float2int(static_cast<float>(brg.beta)));
        dup(vmm_beta.s, wreg_tmp);
    }
    int base_offset = 0;
    auto x_addr = reg_aux_C;
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto vmm = accm(ld_block2, bd, ld);
        if (dq2ps_required) { scvtf(vmm.s, P_ALL_ONE / T_m, vmm.s); }
        if (apply_alpha) { fmul(vmm.s, vmm.s, vmm_alpha.s); }
        if (apply_beta_after_norm) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            const int offset = C_offset(bd, ld);
            auto vmm_c = z_tmp_2();

            if (!use_mul_vl(offset - base_offset, 4, cpu_sveLen)) {
                add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            LD_MUL_VL(ld1w, vmm_c.s, k_mask, x_addr, offset - base_offset, 4);

            if (brg.is_int8 && !dq2ps_required) {
                add(vmm.s, vmm.s, vmm_c.s);
            } else {
                if (brg.is_int8) scvtf(vmm_c.s, P_ALL_ONE / T_m, vmm_c.s);
                if (brg.beta != 1.f) { fmul(vmm_c.s, vmm_c.s, vmm_beta.s); }
                fadd(vmm.s, vmm.s, vmm_c.s);
            }
        }
    }
}

void jit_brgemm_kernel_t::apply_post_ops(
        int bd_block, int ld_block2, int ldb_and_bdb_offset, bool is_ld_tail) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    const injector_utils::conditional_register_preserve_guard_t<sve>
            register_guard(brg.with_binary, this, {param1});
    const auto guard_space = register_guard.stack_space_occupied();
    if (brg.with_binary) {
        add_imm(X_DEFAULT_ADDR, sp, abi_param1_offs_ + guard_space, X_TMP_0);
        ldr(param1, ptr(X_DEFAULT_ADDR));

        if (with_binary_non_scalar_bcast_) {
            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm_idx = accm(ld_block2, bd, ld).getIdx();

                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_aux_D);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, D_offset(bd, ld));
                if (is_ld_tail && ld + 1 == ld_block2)
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        {
            const injector_utils::conditional_register_preserve_guard_t<sve>
                    register_guard_sum_scale((with_binary_non_scalar_bcast_)
                                    && p_sum_scale_reg_set,
                            this, {reg_ptr_sum_scale});
            const injector_utils::conditional_register_preserve_guard_t<sve>
                    register_guard_sum_zp(
                            p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

            const auto &vmm_sum_scale = z_tmp_2();
            const auto &vmm_sum_zp = z_tmp_3();

            if (p_sum_zp_reg_set) {
                mov_imm(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
                ld1rw(vmm_sum_zp.s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_zp));
                scvtf(vmm_sum_zp.s, P_ALL_ONE / T_m, vmm_sum_zp.s);
            }

            if (p_sum_scale_reg_set) {
                mov_imm(reg_ptr_sum_scale,
                        reinterpret_cast<size_t>(p_sum_scale));
                ld1rw(vmm_sum_scale.s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_scale));
            }

            for_(int bd = 0; bd < bd_block; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm = accm(ld_block2, bd, ld);
                const auto vmm_prev_dst = z_tmp_1();
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                const int offset = D_offset(bd, ld);
                add_imm(X_DEFAULT_ADDR, reg_aux_D, offset, X_TMP_0);
                cvt2ps(brg.sum_dt, vmm_prev_dst, X_DEFAULT_ADDR, is_tail, false,
                        k_mask, 0, 0);
                if (p_sum_zp_reg_set)
                    fsub(vmm_prev_dst.s, vmm_prev_dst.s, vmm_sum_zp.s);
                if (p_sum_scale_reg_set) {
                    fmla(vmm.s, P_ALL_ONE / T_m, vmm_prev_dst.s,
                            vmm_sum_scale.s);
                } else
                    fadd(vmm.s, vmm.s, vmm_prev_dst.s);
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(
            max_effective_vregs - bd_block * ld_block2, max_effective_vregs,
            rhs_arg_params);
}

static inline bool isa_has_masks(cpu_isa_t isa) {
    return is_superset(isa, sve_128);
}

void jit_brgemm_kernel_t::store_accumulators_apply_post_ops(
        int bd_block, int ld_block2, int ldb_and_bdb_offset, bool is_ld_tail) {

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    // This flag tracks whether the conversion has happened, since it must be
    // done only once despite all scales and bias applications requiring it.
    bool dq2ps_cvt_done = false;

    if (brg.with_scales) {
        // gemv is true for 1xK and Kx1 cases, but here we are only interested
        // in 1xK with per_oc scales, so the leading dimension B (LDB) must not be 1,
        // otherwise it would be a Kx1 case.
        if (brg.is_gemv && brg.is_oc_scale && brg.LDB != 1) {
            int offset = 0;
            add_imm(X_DEFAULT_ADDR, sp, reg_aux_scales_offs_, X_TMP_0);
            ldr(reg_aux_scales, ptr(X_DEFAULT_ADDR));
            for (int ld = 0; ld < ld_block2; ld++) {
                for (int bd = 0; bd < bd_block; bd++) {
                    const auto addr = X_DEFAULT_ADDR;
                    offset = (ld + bd * ld_block2) * sizeof(float);
                    add_imm(addr, reg_aux_scales, offset, X_TMP_0);
                    const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                    const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                    auto vmm_scales = z_tmp_1();
                    if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                        ld1w(vmm_scales.s, k_mask / T_z, ptr(addr));
                    } else {
                        assert(!"Unreachable\n");
                    }
                    auto vmm = accm(ld_block2, bd, ld);
                    fmul(vmm.s, vmm.s, vmm_scales.s);
                }
            }

            // update the scale pointer
            ldr_imm(reg_aux_scales, sp, reg_aux_scales_offs_);
            add_imm(reg_aux_scales, reg_aux_scales, offset + sizeof(float),
                    X_TMP_0);
            str_imm(reg_aux_scales, sp, reg_scales_offs_);
        } else {
            add_imm(X_DEFAULT_ADDR, sp, reg_aux_scales_offs_, X_TMP_0);
            ldr(reg_aux_scales, ptr(X_DEFAULT_ADDR));
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto addr = X_DEFAULT_ADDR;
                add_imm(addr, reg_aux_scales, scales_offset(ld), X_TMP_0);
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                auto vmm_scales = z_tmp_1();
                if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                    ld1w(vmm_scales.s, k_mask / T_z, ptr(addr));
                } else {
                    assert(!"Unreachable\n");
                }
                for (int bd = 0; bd < bd_block; bd++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (dq2ps_required && !dq2ps_cvt_done) {
                        scvtf(vmm.s, P_ALL_ONE / T_m, vmm.s);
                    }
                    fmul(vmm.s, vmm.s, vmm_scales.s);
                }
            }
        }
        dq2ps_cvt_done = true;
    }

    if (dq2ps_required && !dq2ps_cvt_done) {
        for_(int ld = 0; ld < ld_block2; ld++)
        {
            for (int bd = 0; bd < bd_block; bd++) {
                auto zmm = accm(ld_block2, bd, ld);
                scvtf(zmm.s, P_ALL_ONE / T_m, zmm.s);
            }
        }
    }

    if (brg.with_bias) {
        ldr_imm(reg_aux_bias, sp, reg_aux_bias_offs_);

        auto x_addr = reg_aux_bias;
        auto zmm_bias = z_tmp_1();
        int base_offset = 0;

        // gemv is true for 1xK and Kx1 cases, but here we are only interested
        // in 1xK with, so the leading dimension B (LDB) must not be 1,
        // otherwise it would be a Kx1 case.
        if (brg.is_gemv && brg.LDB != 1) {
            int offset = 0;
            for_(int ld = 0; ld < ld_block2; ld++)
            {
                for (int bd = 0; bd < bd_block; bd++) {
                    offset = (ld + bd * ld_block2) * brg.typesize_bias;
                    if (!use_mul_vl(offset - base_offset,
                                types::data_type_size(brg.dt_bias),
                                cpu_sveLen)) {
                        add_imm(reg_tmp_, x_addr, offset - base_offset,
                                X_TMP_0);
                        base_offset = offset;
                        x_addr = reg_tmp_;
                    }
                    const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                    const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                    cvt2ps(brg.dt_bias, zmm_bias, x_addr, is_tail, false,
                            k_mask, offset, base_offset);
                    auto zmm = accm(ld_block2, bd, ld);
                    fadd(zmm.s, zmm.s, zmm_bias.s);
                }
            }

            // update the bias pointer
            ldr_imm(reg_aux_bias, sp, reg_aux_bias_offs_);
            add_imm(reg_aux_bias, reg_aux_bias, offset + 4, X_TMP_0);
            str_imm(reg_aux_bias, sp, reg_bias_offs_);

        } else {
            for_(int ld = 0; ld < ld_block2; ld++)
            {
                const int offset = bias_offset(ld);
                if (!use_mul_vl(offset - base_offset,
                            types::data_type_size(brg.dt_bias), cpu_sveLen)) {
                    add_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                    base_offset = offset;
                    x_addr = reg_tmp_;
                }
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                cvt2ps(brg.dt_bias, zmm_bias, x_addr, is_tail, false, k_mask,
                        offset, base_offset);

                for (int bd = 0; bd < bd_block; bd++) {
                    auto zmm = accm(ld_block2, bd, ld);
                    fadd(zmm.s, zmm.s, zmm_bias.s);
                }
            }
        }
    }

    if (postops_injector_)
        apply_post_ops(bd_block, ld_block2, ldb_and_bdb_offset, is_ld_tail);

    if (brg.with_dst_scales) {
        add_imm(X_DEFAULT_ADDR, sp, reg_dst_scales_offs_, X_TMP_0);
        ldr(reg_aux_dst_scales, ptr(X_DEFAULT_ADDR));
        auto vmm_dst_scales = z_tmp_1();
        ld1rw(vmm_dst_scales.s, P_ALL_ONE / T_z, ptr(reg_aux_dst_scales));

        for (int ld = 0; ld < ld_block2; ld++) {
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                fmul(vmm.s, vmm.s, vmm_dst_scales.s);
            }
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        add_imm(X_DEFAULT_ADDR, sp, reg_aux_zp_c_values_offs_, X_TMP_0);
        ldr(reg_aux_zp_c_values, ptr(X_DEFAULT_ADDR));
        auto vmm_zp_c = z_tmp_1();
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            add_imm(X_DEFAULT_ADDR, reg_aux_zp_c_values, 0, X_TMP_0);
            ld1rw(z_tmp_2().s, P_ALL_ONE, ptr(X_DEFAULT_ADDR));
            scvtf(vmm_zp_c.s, P_ALL_ONE / T_m, z_tmp_2().s);
        }
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                int zp_c_off = zp_c_values_offset(ld);
                add_imm(X_DEFAULT_ADDR, reg_aux_zp_c_values, zp_c_off, X_TMP_0);
                cvt2ps(data_type::s32, vmm_zp_c, X_DEFAULT_ADDR,
                        is_tail ? brg.ldb_tail : brg.ld_block, false, k_mask, 0,
                        0);
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                fadd(vmm.s, vmm.s, vmm_zp_c.s);
            }
        }
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto zmm_lbound = z_tmp_1();
    auto zmm_ubound = z_tmp_2();
    if (dt_requires_saturation) {
        init_saturate_f32(
                zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
                auto zmm = accm(ld_block2, bd, ld);
                saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d, k_mask);
                frinti(zmm.s, k_mask, zmm.s);
                fcvtzs(zmm.s, k_mask, zmm.s);
            }
        }
    }

    auto x_addr = reg_aux_D;
    auto base_offset = 0;

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            const int offset = D_offset(bd, ld);
            if (!use_mul_vl(offset - base_offset,
                        types::data_type_size(brg.dt_d), cpu_sveLen)) {
                add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32:
                    ST_MUL_VL(st1w, zmm.s, k_mask, x_addr, offset - base_offset,
                            4);
                    break;
                case data_type::bf16: {
                    bfcvt(zmm.h, k_mask / T_m, zmm.s);
                    ST_MUL_VL(st1h, zmm.s, k_mask, x_addr, offset - base_offset,
                            2);
                    break;
                }
                case data_type::s8:
                    smin(zmm.s, std::numeric_limits<int8_t>::max());
                    smax(zmm.s, std::numeric_limits<int8_t>::min());
                    ST_MUL_VL(st1b, zmm.s, k_mask, x_addr, offset - base_offset,
                            1);
                    break;
                case data_type::u8:
                    umin(zmm.s, std::numeric_limits<uint8_t>::max());
                    ST_MUL_VL(st1b, zmm.s, k_mask, x_addr, offset - base_offset,
                            1);
                    break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brgemm_kernel_t::apply_compensation(
        int bd_block, int ld_block2, bool is_ld_tail) {
    // apply compensation to accumulated values
    // to avoid the loss of accuracy when converting s32 to f32

    if (!brg.req_cal_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = z_tmp_2();
        add_imm(X_DEFAULT_ADDR, sp, reg_zp_a_val_offs_, X_TMP_0);
        add_imm(reg_zp_a_val, sp, reg_zp_a_val_offs_, X_TMP_0);
        ldr(W_TMP_0, ptr(reg_zp_a_val));
        dup(vmm_zp_a_val.s, W_TMP_0);

        add_imm(X_DEFAULT_ADDR, sp, reg_aux_zp_comp_a_offs_, X_TMP_1);
        ldr(reg_aux_zp_comp_a, ptr(X_DEFAULT_ADDR));
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            auto vmm_zp_comp_a = z_tmp_1();
            int zp_comp_a_off = zp_comp_a_offset(ld);
            // apply src zero points value to the accumulated values
            if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                add_imm(X_DEFAULT_ADDR, reg_aux_zp_comp_a, zp_comp_a_off,
                        X_TMP_1);
                ld1w(vmm_zp_comp_a.s, k_mask / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                assert(!"Unreachable\n");
            }
            mul(vmm_zp_comp_a.s, P_ALL_ONE / T_m, vmm_zp_a_val.s);

            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                add(vmm.s, vmm.s, vmm_zp_comp_a.s);
            }
        }
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        add_imm(X_DEFAULT_ADDR, sp, reg_aux_zp_comp_b_offs_, X_TMP_0);
        ldr(reg_aux_zp_comp_b, ptr(X_DEFAULT_ADDR));
        for (int bd = 0; bd < bd_block; bd++) {
            int zp_comp_b_off = zp_comp_b_offset(bd);
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                ld1rw(vmm.s, P_ALL_ONE / T_z,
                        ptr(reg_aux_zp_comp_b, zp_comp_b_off));
            }
        }
    }

    if (!brg.req_cal_comp_pads && brg.req_s8s8_compensation) {
        ldr(reg_aux_compensation, ptr(sp, reg_aux_comp_offs_));
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm_comp = z_tmp_1();
            int comp_offset = compensations_offset(ld);
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto k_mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            if (IMPLICATION(is_tail, is_superset(brg.isa_impl, sve_512))) {
                add_imm(X_DEFAULT_ADDR, reg_aux_compensation, comp_offset,
                        X_TMP_1);
                ld1w(vmm_comp.s, k_mask / T_z, ptr(X_DEFAULT_ADDR));
            } else {
                not_(P_TMP.b, P_ALL_ONE, P_NOT_256.b);
                cmplt(P_TMP.s, P_TMP / T_z, z_tail_mask().s, 0);
                add_imm(X_DEFAULT_ADDR, reg_aux_compensation, comp_offset,
                        X_TMP_1);
                ld1w(vmm_comp.s, P_TMP / T_z, ptr(X_DEFAULT_ADDR));
            }

            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                add(vmm.s, vmm.s, vmm_comp.s);
            }
        }
    }
}

void jit_brgemm_kernel_t::store_accumulators_without_post_ops(int bd_block,
        int ld_block2, bool is_ld_tail, bool mmla_deinterleave_on_store) {

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dt_requires_saturation = brg.is_int8
            && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    auto zmm_lbound = z_tmp_1();
    auto zmm_ubound = z_tmp_2();
    assert(zmm_lbound.getIdx() != zmm_ubound.getIdx());
    if (dt_requires_saturation) {
        init_saturate_f32(
                zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32, brg.dt_d);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm = accm(ld_block2, bd, ld);
                saturate_f32(zmm, zmm_lbound, zmm_ubound, brg.dt_d, P_ALL_ONE);
                frinti(zmm.s, P_ALL_ONE, zmm.s);
                fcvtzs(zmm.s, P_ALL_ONE, zmm.s);
            }
        }
    }
    auto x_addr = reg_aux_C;
    int base_offset = 0;

    if (mmla_deinterleave_on_store) {
        const auto store_acc = [&](const ZReg &zmm, int bd, int ld) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            const int offset = C_offset(bd, ld);
            if (!use_mul_vl(offset - base_offset, 4, cpu_sveLen)) {
                add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            ST_MUL_VL(st1w, zmm.s, mask, x_addr, offset - base_offset, 4);
        };

        for (int bd_pair = 0; bd_pair < mmla_bd_pairs(bd_block); bd_pair++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto zmm_even = accm_mmla(ld_block2, bd_pair, 2 * ld);
                auto zmm_odd = accm_mmla(ld_block2, bd_pair, 2 * ld + 1);
                uzp1(ZReg(ld).d, zmm_even.d, zmm_odd.d);
                uzp2(zmm_odd.d, zmm_even.d, zmm_odd.d);
            }
            const int bd = mmla_bd_blk() * bd_pair;
            for (int ld = 0; ld < ld_block2; ld++)
                store_acc(ZReg(ld), bd, ld);
            if ((bd + 1) < bd_block) {
                for (int ld = 0; ld < ld_block2; ld++)
                    store_acc(accm_mmla(ld_block2, bd_pair, 2 * ld + 1), bd + 1,
                            ld);
            }
        }
        return;
    }

    auto scalar_reg = SReg(0);

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            const auto mask = is_tail ? ld_tail_mask : P_ALL_ONE;
            const int offset = C_offset(bd, ld);
            if (!use_mul_vl(offset - base_offset, 4, cpu_sveLen)) {
                add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }
            if (use_gemv_path && brg.beta == 0.f) {
                faddv(scalar_reg, P_ALL_ONE, zmm.s);
                str_imm(scalar_reg, x_addr, (offset - base_offset));
            } else if (use_gemv_path && brg.beta != 0.f) {
                ldr_imm(scalar_reg, x_addr, (offset - base_offset));
                fadda(scalar_reg, P_ALL_ONE, zmm.s);
                str_imm(scalar_reg, x_addr, (offset - base_offset));
            } else {
                ST_MUL_VL(st1w, zmm.s, mask, x_addr, offset - base_offset, 4);
            }
        }
    }
}

void jit_brgemm_kernel_t::store_accumulators(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_ld_tail, bool skip_accumulation) {
    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.with_dst_scales,
            brg.req_s8s8_compensation, has_zero_points);
    const bool need_to_apply_alpha_beta = brg.beta != 0.f || brg.alpha != 1.f;
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const bool mmla_deinterleave_on_store = use_mmla
            && brg.dt_c == data_type::f32 && brg.dt_d == data_type::f32
            && ld_block2 <= 6 && !need_to_apply_alpha_beta
            && !are_post_ops_applicable;

    if (use_mmla && !mmla_deinterleave_on_store)
        mmla_deinterleave_accumulators(bd_block, ld_block2);

    if (brg.is_int8 && (brg.req_s8s8_compensation || has_zero_points)) {
        Label label_store_without_comp;
        ldr(reg_do_comp, ptr(sp, reg_do_comp_offs_));
        cmp_imm(reg_do_comp, 0, X_TMP_0);
        b(EQ, label_store_without_comp);
        apply_compensation(bd_block, ld_block2, is_ld_tail);
        L(label_store_without_comp);
    }

    if (need_to_apply_alpha_beta)
        apply_alpha_beta(bd_block, ld_block2, is_ld_tail);

    Label label_done;
    if (are_post_ops_applicable) {
        Label label_store_without_post_ops;

        ldr_imm(reg_do_post_ops, sp, reg_do_post_ops_offs_);
        cmp_imm(reg_do_post_ops, 0, X_TMP_0);
        b(EQ, label_store_without_post_ops);
        if (use_gemv_path) {
            sum_into_one_lane(bd_block, ld_block2, is_ld_tail);
        }
        store_accumulators_apply_post_ops(bd_block, ld_block2, 0, is_ld_tail);
        bl(label_done);

        L(label_store_without_post_ops);
    }
    store_accumulators_without_post_ops(
            bd_block, ld_block2, is_ld_tail, mmla_deinterleave_on_store);
    L(label_done);
}

void jit_brgemm_kernel_t::sum_into_one_lane(
        int bd_block, int ld_block2, bool is_ld_tail) {
    auto x_addr = reg_aux_C;
    int base_offset = 0;

    auto scalar_reg = SReg(0);

    for (int bd = 0; bd < bd_block; bd++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm = accm(ld_block2, bd, ld);
            const int offset = C_offset(bd, ld);
            if ((unsigned)(offset - base_offset) > cpu_sveLen * 7) {
                add_imm(reg_tmp_, reg_aux_C, offset, X_TMP_0);
                base_offset = offset;
                x_addr = reg_tmp_;
            }

            if (brg.beta == 0.f) {
                faddv(scalar_reg, P_ALL_ONE, zmm.s);
            } else {
                ldr_imm(scalar_reg, x_addr, (offset - base_offset));
                fadda(scalar_reg, P_ALL_ONE, zmm.s);
            }
            uni_clear(zmm);
            mov(zmm.s, ld_tail_mask, scalar_reg);
        }
    }
}

void jit_brgemm_kernel_t::restore_A_B_matrices() {
    auto restore_reg_batch = brg.brgattr.max_bs > 1 || vpad_exist;
    if (brg.type == brgemm_addr) {
        if (restore_reg_batch) mov(reg_aux1_batch, reg_addr_batch);
    } else {
        mov(reg_aux1_A, reg_A);
        mov(reg_aux1_B, reg_B);

        if (restore_reg_batch) {
            if (brg.type == brgemm_offs) {
                ldr(reg_offs_batch, ptr(sp, origin_offs_batch_offs_));
            } else {
                ldr(reg_offs_batch, ptr(sp, origin_strd_batch_offs_));
            }
        }
    }
}

void jit_brgemm_kernel_t::set_A_B_matrices() {
    if (brg.type == brgemm_addr) {
        if (brg.brgattr.max_bs > 1) {
            if (brg.layout == brgemm_row_major) {
                ldr(reg_aux_A,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
                ldr(reg_aux_B,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
            } else {
                // In case of a col_major layout, the pointers A and B are
                // swapped due to the equation A@B = (B.T@A.T).T
                ldr(reg_aux_A,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
                ldr(reg_aux_B,
                        ptr(reg_aux1_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            }
        } else {
            // for max_batch == 1 we stored A and B pointers at the beginning
            // of kernel in reg_aux1_A and reg_aux1_B
            if (brg.layout == brgemm_row_major) {
                mov(reg_aux_A, reg_aux1_A);
                mov(reg_aux_B, reg_aux1_B);
            } else {
                mov(reg_aux_A, reg_aux1_B);
                mov(reg_aux_B, reg_aux1_A);
            }
        }

        if (brg.brgattr.max_bs > 1) {
            add_imm(reg_aux1_batch, reg_aux1_batch,
                    sizeof(brgemm_batch_element_t), X_TMP_0);
            prfm(PLDL1KEEP, ptr(reg_aux1_batch));
        }
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);
        ldr(X_TMP_0, ptr(reg_offs_batch, GET_OFF_BATCH_ELEMENT(offset.A)));
        add(reg_aux_A, reg_aux_A, X_TMP_0);
        ldr(X_TMP_1, ptr(reg_offs_batch, GET_OFF_BATCH_ELEMENT(offset.B)));
        add(reg_aux_B, reg_aux_B, X_TMP_1);
        if (brg.brgattr.max_bs > 1)
            add_imm(reg_offs_batch, reg_offs_batch,
                    sizeof(brgemm_batch_element_t), X_TMP_2);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        mov_imm(reg_tmp_gpr, brg.stride_a);
        add(reg_aux1_A, reg_aux1_A, reg_tmp_gpr);
        mov_imm(reg_tmp_gpr, brg.stride_b);
        add(reg_aux1_B, reg_aux1_B, reg_tmp_gpr);
        if (vpad_exist) {
            ldr(reg_strd_batch, ptr(sp, origin_strd_batch_offs_));
            mov_imm(reg_strd_batch, sizeof(brgemm_batch_element_t));
            str(reg_strd_batch, ptr(sp, origin_strd_batch_offs_));
        }
    }
    add(reg_aux_A, reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_aux_B, reg_b_offset);
}

void jit_brgemm_kernel_t::mmla_deinterleave_accumulators(
        int bd_block, int ld_block2) {
    for (int bd_pair = 0; bd_pair < mmla_bd_pairs(bd_block); bd_pair++) {
        for (int ld = 0; ld < ld_block2; ld++) {
            auto zmm_even = accm_mmla(ld_block2, bd_pair, 2 * ld);
            auto zmm_odd = accm_mmla(ld_block2, bd_pair, 2 * ld + 1);
            uzp1(z_tmp_1().d, zmm_even.d, zmm_odd.d);
            uzp2(zmm_odd.d, zmm_even.d, zmm_odd.d);
            mov(zmm_even.d, z_tmp_1().d);
        }
    }
}

void jit_brgemm_kernel_t::dot_product(ZReg v_acc, ZReg v_a, ZReg v_b) {
    if (brg.is_f32) {
        fmla(v_acc.s, P_ALL_ONE / T_m, v_a.s, v_b.s);
    } else if (brg.is_bf16) {
        bfdot(v_acc.s, v_b.h, v_a.h);
    } else if (brg.is_int8 && isa_has_s8s8(brg.isa_impl)) {
        // SDOT/USDOT/UDOT implicitly produce int32 output.
        // we reorder RHS to align for SDOT lane-wise ops.
        if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8)
            udot(v_acc.s, v_a.b, v_b.b);
        else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8)
            sdot(v_acc.s, v_a.b, v_b.b);
        else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8)
            usdot(v_acc.s, v_b.b, v_a.b);
        // TODO: Add support for s8u8 once zero point handling can be properly tested.
        // Currently excluded as we are unable to test zero point compensation.
        else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::u8)
            assert(!"unsupported\n");
    } else {
        assert(!"unsupported\n");
    }
}

void jit_brgemm_kernel_t::dot_product(
        ZReg v_acc, ZReg v_a, ZReg v_b, const int16_t index) {
    if (brg.is_f32) {
        fmla(v_acc.s, v_a.s,
                ZRegSElem(v_b.getIdx(), static_cast<uint32_t>(index)));
    } else if (brg.is_bf16) {
        bfdot(v_acc.s, v_a.h,
                ZRegHElem(v_b.getIdx(), static_cast<uint32_t>(index)));
    } else if (brg.is_int8 && isa_has_s8s8(brg.isa_impl)) {
        // SDOT/USDOT/UDOT implicitly produce int32 output.
        // we reorder RHS to align for SDOT index-wise ops.
        auto v_b_index = ZRegBElem(v_b.getIdx(), static_cast<uint32_t>(index));
        if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8)
            udot(v_acc.s, v_a.b, v_b_index);
        else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8)
            sdot(v_acc.s, v_a.b, v_b_index);
        else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8)
            sudot(v_acc.s, v_a.b, v_b_index);
        // TODO: Add support for s8u8 once zero point handling can be properly tested.
        // Currently excluded as we are unable to test zero point compensation.
        else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::u8)
            assert(!"unsupported\n");
    } else {
        assert(!"unsupported\n");
    }
}

void jit_brgemm_kernel_t::mmla(ZReg v_acc, ZReg v_a, ZReg v_b) {
    if (brg.is_bf16) {
        bfmmla(v_acc.s, v_a.h, v_b.h);
    } else if (brg.is_int8) {
        if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8)
            smmla(v_acc.s, v_a.b, v_b.b);
        else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8)
            ummla(v_acc.s, v_a.b, v_b.b);
        else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8)
            usmmla(v_acc.s, v_a.b, v_b.b);
        else
            assert(!"unsupported int8 mmla data type combination");
    } else {
        assert(!"unsupported");
    }
}

void jit_brgemm_kernel_t::compute_int8_compensation(int rd_loop, int bd_b,
        int bd_e, int bd_block, int ld_block2, bool is_ld_tail, int vpad) {
    assert(brg.is_int8);

    auto compensation_padding = [this, ld_block2](ZReg vmm_load, ZReg vmm_tmp,
                                        int ld, int bd_b, int bd_e) {
        // req_cal_comp_pads -> only calculate compensation along with
        // computation and do not use pre-calculated compensation.
        // Calculate comp padding as:
        // accum - inp_shift * conv(1, wei_s32)
        if (brg.req_s8s8_compensation) {
            if (brg.req_cal_comp_pads) {
                uni_clear(vmm_tmp);
                dot_product(vmm_tmp, vmm_load, z_inp_shift());
            }

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    sub(vmm.s, vmm.s, vmm_tmp.s);
                } else {
                    dot_product(vmm, vmm_load, z_inp_shift());
                }
            }
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            uni_clear(vmm_tmp);
            dot_product(vmm_tmp, vmm_load, z_one_bytes());
            mul(vmm_tmp.s, P_ALL_ONE / T_m, z_zp_a_shift().s);

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    sub(vmm.s, vmm.s, vmm_tmp.s);
                } else {
                    add(vmm.s, vmm.s, vmm_tmp.s);
                }
            }
        }
    };

    if (n_bcast_1_load && brg.zp_type_a != brgemm_broadcast_t::none) {
        str(reg_bdb_loop, ptr(sp, reg_bdb_loop_offs_));
        const auto reg32_scratch = WReg(reg_zp_a_input_shift.getIdx());
        mov_imm(reg32_scratch, 0x1010101);
        dup(z_one_bytes().s, reg32_scratch);
        ldr(reg32_scratch, ptr(sp, reg_zp_a_val_offs_));
        dup(z_zp_a_shift().s, reg32_scratch);
        ldr(reg_bdb_loop, ptr(sp, reg_bdb_loop_offs_));
    }

    for_(int rd = 0; rd < rd_loop; rd += brg.rd_step)
    for (int ld = 0; ld < ld_block2; ++ld) {
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        const auto mask = is_tail ? ld_tail_mask : P_ALL_ONE;
        add_imm(X_DEFAULT_ADDR, reg_aux_B, B_offset(ld, rd), X_TMP_0);
        ld1w(load().s, mask / T_z, ptr(X_DEFAULT_ADDR));

        if (brg.req_cal_comp_pads) {
            compensation_padding(load(), bcst(), ld, bd_b, bd_e);
        } else if (vpad != 0) {
            if (bd_b > 0) compensation_padding(load(), bcst(), ld, 0, bd_b);
            if (bd_e < bd_block)
                compensation_padding(load(), bcst(), ld, bd_e, bd_block);
        }
    }
}

void jit_brgemm_kernel_t::load_A_word_for_bcast(int32_t &base_offset,
        const ZReg &dst, bool is_tail, const XReg &reg_A_ptr, const int32_t bd,
        const int32_t rd, const XReg &tmp) {
    auto dt_bytes = dnnl_data_type_size(brg.dt_a);

    int64_t offset_bytes = A_offset(bd, rd) - base_offset;
    // Stride between bd
    const auto A_stride_bytes = brg.typesize_A * brg.LDA;

    // Tail here means fewer elements than needed to make up a word
    if (!is_tail || brg.typesize_A == 4) {
        // If the offset is out of range of LD1RW, add strides so it isn't
        // https://developer.arm.com/documentation/ddi0596/2021-03/SVE-Instructions/LD1RW--Load-and-broadcast-unsigned-word-to-vector-
        if (offset_bytes > 252 || offset_bytes < 0 || (offset_bytes % 4) != 0) {
            auto num_strides_to_increment = offset_bytes / A_stride_bytes;
            base_offset += num_strides_to_increment * A_stride_bytes;
            offset_bytes -= num_strides_to_increment * A_stride_bytes;
            strided_addr(reg_A_ptr, reg_A_ptr, reg_stride_bytes_A,
                    A_stride_bytes, num_strides_to_increment, tmp);
        }
        // This would require rd > 63 (=252/4), which _should_ be impossible, or we messed up the above logic
        assert(!(offset_bytes > 252 || offset_bytes < 0
                || (offset_bytes % 4) != 0));

        auto addr = ptr(reg_A_ptr, static_cast<int32_t>(offset_bytes));
        ld1rw(dst.s, P_ALL_ONE / T_z, addr);
    } else {
        const int64_t mul_vl = offset_bytes / simd_bytes(brg.isa_impl);
        if (offset_bytes % simd_bytes(brg.isa_impl) == 0 && mul_vl >= -8
                && mul_vl <= 7) {
            auto addr = ptr(reg_A_ptr, mul_vl, MUL_VL);
            if (dt_bytes == 1) ld1b(dst.b, rd_tail_mask / T_z, addr);
            if (dt_bytes == 2) ld1h(dst.h, rd_tail_mask / T_z, addr);
        } else {
            add_imm(X_DEFAULT_ADDR, reg_A_ptr, offset_bytes, tmp);
            auto addr = ptr(X_DEFAULT_ADDR);
            if (dt_bytes == 1) ld1b(dst.b, rd_tail_mask / T_z, addr);
            if (dt_bytes == 2) ld1h(dst.h, rd_tail_mask / T_z, addr);
        }
        // For VL=128, we could elide this dup by using  indexed DOT/FMLA, but
        // we would need to justify the extra logic, especially given
        // that it is only used for tails
        dup(dst.s, dst.s[0]);
    }
}

void jit_brgemm_kernel_t::load_quadword_for_bcast(const ZReg &dst,
        const XReg &base, const PReg &mask, const XReg &reg_stride_bytes,
        const int32_t stride_bytes, const int32_t n, const data_type_t dt) {
    auto dt_bytes = dnnl_data_type_size(dt);
    if (!utils::one_of(dt_bytes, 1ul, 2ul, 4ul)) {
        assert("Unsupported data type size");
    }
    int32_t offset_bytes = stride_bytes * n;
    if (offset_bytes >= -128 && offset_bytes <= 112
            && (offset_bytes % 16) == 0) {
        // (1 instruction) We can use immediate version, includes n == 0 case
        auto addr = AdrScImm(ptr(base, offset_bytes));
        if (dt_bytes == 1) ld1rqb(dst.b, mask / T_z, addr);
        if (dt_bytes == 2) ld1rqh(dst.h, mask / T_z, addr);
        if (dt_bytes == 4) ld1rqw(dst.s, mask / T_z, addr);
    } else {
        // (2 or 3 instructions) [mov_imm +] add + ld
        XReg reg_addr = strided_addr(X_DEFAULT_ADDR, base, reg_stride_bytes,
                stride_bytes, n, X_TMP_3);
        auto addr = ptr(reg_addr);
        if (dt_bytes == 1) ld1rqb(dst.b, mask / T_z, addr);
        if (dt_bytes == 2) ld1rqh(dst.h, mask / T_z, addr);
        if (dt_bytes == 4) ld1rqw(dst.s, mask / T_z, addr);
    }
    // Note that we could use ld1rq* register + register, but it is quite complicated and
    // slower on V1. Perf uplift (~0.5%) on V2 did not justify complexity
}

void jit_brgemm_kernel_t::mmla_load_packed_a(int rd_chunk, int bd_pair,
        const ZReg &z_a, const ZReg &z_a_tmp, int bd_block, bool row0_valid,
        bool row1_valid) {
    const XReg base = (rd_chunk == 0 && bd_pair == 0) ? reg_aux_A : reg_tmp_;
    if (row0_valid && row1_valid) {
        ld1rqh(z_a.h, P_ALL_ONE / T_z, ptr(base));
    } else if (row0_valid) {
        ld1rd(z_a.d, P_ALL_ONE / T_z, ptr(base));
        eor(z_a_tmp.d, z_a_tmp.d, z_a_tmp.d);
        zip1(z_a.d, z_a.d, z_a_tmp.d);
    } else if (row1_valid) {
        eor(z_a.d, z_a.d, z_a.d);
        ld1rd(z_a_tmp.d, P_ALL_ONE / T_z,
                ptr(base, mmla_rd_chunk_elems() * brg.typesize_A));
        zip1(z_a.d, z_a.d, z_a_tmp.d);
    } else {
        eor(z_a.d, z_a.d, z_a.d);
    }
    if (bd_pair != mmla_bd_pairs(bd_block) - 1) {
        const int packed_a_pair_stride
                = div_up(brg.reduce_dim, mmla_rd_chunk_elems()) * mmla_bd_blk()
                * mmla_rd_chunk_elems() * brg.typesize_A;
        add_imm(reg_tmp_, base, packed_a_pair_stride, X_TMP_0);
    }
}

void jit_brgemm_kernel_t::mmla_load_plain_a(int rd_chunk, int bd_pair,
        const ZReg &z_a, const ZReg &z_a_tmp, int rd_tail, bool row0_valid,
        bool row1_valid) {
    const int bd = mmla_bd_blk() * bd_pair;
    const int rd_off = rd_chunk * mmla_rd_chunk_elems();
    const int valid_rd_elems
            = nstl::min(mmla_rd_chunk_elems(), rd_tail - rd_off);
    const bool is_full_rd_chunk = valid_rd_elems == mmla_rd_chunk_elems();

    const auto add_a_offset = [&](const XReg &dst, const XReg &src, int off) {
        constexpr int add_imm12_max = (1 << 12) - 1;
        if (off >= 0 && off <= add_imm12_max) {
            add(dst, src, off);
        } else if (off >= 0 && off % (1 << 12) == 0
                && (off >> 12) <= add_imm12_max) {
            add(dst, src, off >> 12, 12);
        } else {
            add_imm(dst, src, off, X_TMP_0);
        }
    };

    if (!is_full_rd_chunk) {
        if (brg.typesize_A == 1)
            set_preg(rd_tail_mask.b, valid_rd_elems, X_TMP_0);
        else
            set_preg(rd_tail_mask.h, valid_rd_elems, X_TMP_0);
    }
    const auto load_a_row = [&](const ZReg &z, bool row_valid, int row) {
        if (!row_valid) {
            eor(z.d, z.d, z.d);
            return;
        }

        const int off = A_offset(row, rd_off);
        if (is_full_rd_chunk && off <= 504 && off % 8 == 0) {
            ld1rd(z.d, P_ALL_ONE / T_z, ptr(reg_aux_A, off));
        } else {
            add_a_offset(X_TMP_1, reg_aux_A, off);
            if (is_full_rd_chunk) {
                ld1rd(z.d, P_ALL_ONE / T_z, ptr(X_TMP_1));
            } else {
                if (brg.typesize_A == 1)
                    ld1b(z.b, rd_tail_mask / T_z, ptr(X_TMP_1));
                else
                    ld1h(z.h, rd_tail_mask / T_z, ptr(X_TMP_1));
                dup(z.d, z.d[0]);
            }
        }
    };

    load_a_row(z_a, row0_valid, bd);
    load_a_row(z_a_tmp, row1_valid, bd + 1);
    zip1(z_a.d, z_a.d, z_a_tmp.d);
}

void jit_brgemm_kernel_t::mmla_load_b(const XReg &b_base, int ld, int z_idx) {
    const int ld_off = ld * mmla_ld_regs_per_block();
    if (ld_off + 1 <= 7) {
        ld1h(ZReg(z_idx).h, P_ALL_ONE / T_z, ptr(b_base, ld_off, MUL_VL));
        ld1h(ZReg(z_idx + 1).h, P_ALL_ONE / T_z,
                ptr(b_base, ld_off + 1, MUL_VL));
    } else {
        const int simd_size = static_cast<int>(simd_bytes(brg.isa_impl));
        add_vl_or_imm(reg_tmp_, b_base, ld_off * simd_size, X_TMP_0);
        ld1h(ZReg(z_idx).h, P_ALL_ONE / T_z, ptr(reg_tmp_));
        ld1h(ZReg(z_idx + 1).h, P_ALL_ONE / T_z, ptr(reg_tmp_, 1, MUL_VL));
    }
}

void jit_brgemm_kernel_t::gemm_microkernel_mmla(bool is_bdb_tail, int ld_block2,
        int b_panel_ld_block2, bool is_rd_tail, bool advance_a_b, int vpad) {
    const int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const int bd_b = nstl::max(0, vpad);
    const int bd_e = nstl::min(bd_block, bd_block + vpad);
    if (bd_b >= bd_e) return;

    const int bd_pair_count = mmla_bd_pairs(bd_block);
    const int rd_tail = is_rd_tail ? brg.rdb_tail : mmla_rd_block();
    // mmla traverses the rd dimension in 64-bit chunks
    const int rd_chunks = is_rd_tail ? div_up(rd_tail, mmla_rd_chunk_elems())
                                     : mmla_rd_chunks_per_block();

    const int tmp_reg_count
            = max_effective_vregs - mmla_bd_blk() * bd_pair_count * ld_block2;
    const int all_a_reg_count = rd_chunks * bd_pair_count;
    const bool all_a_in_regs
            = all_a_reg_count + mmla_ld_regs_per_block() <= tmp_reg_count;
    const int tmp_reg_0 = all_a_in_regs ? all_a_reg_count : bd_pair_count;
    const int ld_blocks = nstl::min(
            ld_block2, (tmp_reg_count - tmp_reg_0) / mmla_ld_regs_per_block());

    const auto prepare_packed_a_base = [&](int rd_chunk) {
        if (rd_chunk == 0) return;
        const int a_offset = rd_chunk * mmla_bd_blk() * mmla_rd_chunk_elems()
                * brg.typesize_A;
        add_imm(reg_tmp_, reg_aux_A, a_offset, X_TMP_0);
    };

    const auto load_a = [&](int rd_chunk, int bd_pair, const ZReg &z_a,
                                const ZReg &z_a_tmp) {
        const int bd = mmla_bd_blk() * bd_pair;
        const bool row0_valid = bd_b <= bd && bd < bd_e;
        const bool row1_valid = bd_b <= bd + 1 && bd + 1 < bd_e;
        if (!use_mmla_packed_a) {
            mmla_load_plain_a(rd_chunk, bd_pair, z_a, z_a_tmp, rd_tail,
                    row0_valid, row1_valid);
        } else {
            mmla_load_packed_a(rd_chunk, bd_pair, z_a, z_a_tmp, bd_block,
                    row0_valid, row1_valid);
        }
    };

    const auto advance_after_rd_chunk = [&](int rd_chunk) {
        const bool is_last_rd_chunk = (rd_chunk == rd_chunks - 1);
        if (is_last_rd_chunk)
            add_vl_or_imm(reg_aux_A, reg_aux_A,
                    rdb_A_offset(bd_pair_count * mmla_bd_blk()), X_TMP_0);

        const int b_rd_chunk_offset = mmla_rd_chunk_B_offset(b_panel_ld_block2);
        const int b_advance = is_last_rd_chunk
                ? mmla_rdb_B_offset(b_panel_ld_block2)
                        - rd_chunk * b_rd_chunk_offset
                : b_rd_chunk_offset;
        add_vl_or_imm(reg_aux_B, reg_aux_B, b_advance, X_TMP_0);
    };

    if (all_a_in_regs) {
        for (int rd_chunk = 0; rd_chunk < rd_chunks; ++rd_chunk) {
            if (use_mmla_packed_a) prepare_packed_a_base(rd_chunk);
            for (int bd_pair = 0; bd_pair < bd_pair_count; ++bd_pair) {
                const int a_reg_idx = rd_chunk * bd_pair_count + bd_pair;
                load_a(rd_chunk, bd_pair, ZReg(a_reg_idx), ZReg(tmp_reg_0));
            }
        }
    }

    for (int rd_chunk = 0; rd_chunk < rd_chunks; ++rd_chunk) {
        const XReg b_base
                = (rd_chunk == 0 || advance_a_b) ? reg_aux_B : X_TMP_4;
        if (rd_chunk > 0 && !advance_a_b) {
            add_vl_or_imm(X_TMP_4, reg_aux_B,
                    rd_chunk * mmla_rd_chunk_B_offset(b_panel_ld_block2),
                    X_TMP_0);
        }

        if (!all_a_in_regs) {
            if (use_mmla_packed_a) prepare_packed_a_base(rd_chunk);
            for (int bd_pair = 0; bd_pair < bd_pair_count; ++bd_pair)
                load_a(rd_chunk, bd_pair, ZReg(bd_pair), ZReg(tmp_reg_0));
        }

        for (int ld_reg_idx = 0; ld_reg_idx < ld_blocks; ++ld_reg_idx) {
            mmla_load_b(b_base, ld_reg_idx,
                    tmp_reg_0 + mmla_ld_regs_per_block() * ld_reg_idx);
        }

        if (advance_a_b && ld_blocks == ld_block2)
            advance_after_rd_chunk(rd_chunk);

        for (int ld_idx = 0; ld_idx < ld_block2; ++ld_idx) {
            const int ld_reg_idx = ld_idx % ld_blocks;
            const int z_b = tmp_reg_0 + mmla_ld_regs_per_block() * ld_reg_idx;
            for (int bd_pair = 0; bd_pair < bd_pair_count; ++bd_pair) {
                const ZReg z_a = all_a_in_regs
                        ? ZReg(rd_chunk * bd_pair_count + bd_pair)
                        : ZReg(bd_pair);
                mmla(accm_mmla(ld_block2, bd_pair, 2 * ld_idx), z_a, ZReg(z_b));
                mmla(accm_mmla(ld_block2, bd_pair, 2 * ld_idx + 1), z_a,
                        ZReg(z_b + 1));
            }
            const int next_ld = ld_idx + ld_blocks;
            if (next_ld < ld_block2) {
                mmla_load_b(b_base, next_ld, z_b);
                if (advance_a_b && next_ld + 1 == ld_block2)
                    advance_after_rd_chunk(rd_chunk);
            }
        }
    }
}

void jit_brgemm_kernel_t::gemm_microkernel(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_rd_tail, bool is_ld_tail, int vpad,
        int rows_for_rd_tail) {
    MAYBE_UNUSED(bd_block2);
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const auto bd_b = nstl::max(0, vpad);
    const auto bd_e = nstl::min(bd_block, bd_block + vpad);
    const auto is_valid_bd
            = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
    if (!is_valid_bd) return;

    int rd_loop = 0, rd_tail_size = 0;
    if (is_rd_tail) {
        rd_tail_size = brg.rdb_tail % brg.rd_step;
        if (brg.is_bf16 || brg.is_int8) {
            rd_loop = (rd_tail_size != 0)
                    ? ((brg.rdb_tail / brg.rd_step) + 1) * brg.rd_step
                    : brg.rdb_tail;
        } else
            rd_loop = brg.rdb_tail;
    } else {
        rd_loop = brg.rd_block;
    }

    const bool comp_vpad = vpad != 0
            && (brg.req_s8s8_compensation
                    || brg.zp_type_a != brgemm_broadcast_t::none);
    if (brg.req_cal_comp_pads || comp_vpad) {
        compute_int8_compensation(
                rd_loop, bd_b, bd_e, bd_block, ld_block2, is_ld_tail, vpad);
    }

    auto A_stride_bytes = static_cast<int32_t>(brg.typesize_A * brg.LDA);

    if (n_bcast_1_load) {
        const bool load_b_early = ld_block2 > 1;
        // Use a tmp to store the pointer to the first element of A in the tile
        // We can't use reg_aux_A directly because we increment this by a quadword when needed
        XReg reg_A_ptr = X_TMP_4;
        mov(reg_A_ptr, reg_aux_A);
        // We need to bump the ptr forward if we will not be starting at 0 (vpad)
        if (bd_b != 0) {
            reg_A_ptr = strided_addr(reg_A_ptr, reg_A_ptr, reg_stride_bytes_A,
                    A_stride_bytes, bd_b, X_TMP_3);
        }
        auto x_addr = reg_aux_B;
        unsigned long b_base_offset = 0;
        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            // Load a quadword and broadcast the words using an indexed FMLA/DOT instead of
            // broadcasting one element at a time. We can only do this if there is no
            // per-DOT overhead, and our elements are contiguous
            auto quadword_index = rd / data_type_vnni_granularity(brg.dt_a) % 4;
            if (quadword_index == 0 && rd != 0) {
                // Bump by quadword
                add(reg_A_ptr, reg_A_ptr, 16);
            }
            auto load_b = [&](const int ld) {
                const auto mask = is_ld_tail ? ld_tail_mask : P_ALL_ONE;
                const int b_offset = B_offset(ld, rd);
                if (!use_mul_vl(b_offset - b_base_offset, 4, cpu_sveLen)) {
                    add_vl_or_imm(reg_tmp_, x_addr, b_offset - b_base_offset,
                            X_TMP_0);
                    b_base_offset = b_offset;
                    x_addr = reg_tmp_;
                }
                const auto b_mul_vl = compute_off_mul_vl(
                        b_offset - b_base_offset, 4, cpu_sveLen);
                ld1w(load(ld).s, mask / T_z, ptr(x_addr, b_mul_vl, MUL_VL));
            };
            if (load_b_early) load_b(0);
            if (quadword_index == 0) {
                // If loading a quadword would take us past the end of rd_loop, we need to use a mask
                bool quadword_is_too_much = (rd + 4 * brg.rd_step) > rd_loop;
                PReg rd_mask = (quadword_is_too_much || is_rd_tail)
                        ? rd_tail_mask
                        : P_ALL_ONE;
                for (int bd = bd_b; bd < bd_e; bd++) {
                    // We have already incremented reg_A_ptr to bd_b
                    auto n_stride = bd - bd_b;
                    load_quadword_for_bcast(bcst(bd), reg_A_ptr, rd_mask,
                            reg_stride_bytes_A, A_stride_bytes, n_stride,
                            brg.dt_a);
                }
            }
            for (int ld = 0; ld < ld_block2; ld++) {
                const int next_ld = ld + load_b_early;
                if (next_ld < ld_block2) load_b(next_ld);
                for (int bd = bd_b; bd < bd_e; bd++) {
                    dot_product(accm(ld_block2, bd, ld), load(ld), bcst(bd),
                            quadword_index);
                }
            }
        }
    } else {
        auto x_addr = reg_aux_B;
        int b_base_offset = 0;

        bool maybe_load_bytes
                = (rows_for_rd_tail > 0 || brg.brgattr.wary_A_k_tail_read)
                && is_rd_tail && rd_tail_size != 0
                && (brg.is_bf16 || brg.is_int8);

        for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
            // Pointer to A we will increment within this microkernel
            int a_base_offset = 0;
            const XReg reg_A_ptr = X_TMP_4;
            mov(reg_A_ptr, reg_aux_A);
            for (int ld = 0; ld < ld_block2; ld++) {
                auto mask = is_ld_tail ? ld_tail_mask : P_ALL_ONE;
                const int offset = B_offset(ld, rd);
                if (!use_mul_vl(offset - b_base_offset, 4, cpu_sveLen)) {
                    add_vl_or_imm(
                            reg_tmp_, x_addr, offset - b_base_offset, X_TMP_0);
                    b_base_offset = offset;
                    x_addr = reg_tmp_;
                }
                auto b_off_mul_vl = (offset - b_base_offset) / cpu_sveLen;
                ld1w(load(ld).s, mask / T_z, ptr(x_addr, b_off_mul_vl, MUL_VL));
            }
            bool have_to_load_bytes
                    = maybe_load_bytes && (rd == rd_loop - brg.rd_step);

            auto rows_by_load_bytes = have_to_load_bytes ? rows_for_rd_tail : 0;
            for (int bd = bd_b; bd < bd_e; bd++) {
                const auto bd_by_load_bytes = (bd >= bd_e - rows_by_load_bytes
                        || brg.brgattr.wary_A_k_tail_read);
                const auto need_rd_mask
                        = (have_to_load_bytes && bd_by_load_bytes);

                load_A_word_for_bcast(a_base_offset, bcst(bd), need_rd_mask,
                        reg_A_ptr, bd, rd, X_TMP_3);

                for (int ld = 0; ld < ld_block2; ld++) {
                    dot_product(accm(ld_block2, bd, ld), load(ld), bcst(bd));
                }
            }
        }
    }
}

void jit_brgemm_kernel_t::gemv_microkernel(
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, int vpad) {

    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const auto bd_b = nstl::max(0, vpad);
    const auto bd_e = nstl::min(bd_block, bd_block + vpad);
    const auto is_valid_bd
            = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
    if (!is_valid_bd) return;

    // GEMV has no rd loop
    int rd = 0;

    auto x_addr = reg_aux_B;
    int base_offset = 0;

    auto mask = is_rd_tail ? rd_tail_mask : P_ALL_ONE;
    for (int ld = 0; ld < ld_block2; ld++) {
        const int offset = B_offset(ld, rd);
        if (!use_mul_vl(offset - base_offset, 4, cpu_sveLen)) {
            add_vl_or_imm(reg_tmp_, x_addr, offset - base_offset, X_TMP_0);
            base_offset = offset;
            x_addr = reg_tmp_;
        }
        LD_MUL_VL(ld1w, load(ld).s, mask, x_addr, offset - base_offset, 4);
    }
    for (int bd = bd_b; bd < bd_e; bd++) {
        LD_MUL_VL(ld1w, bcst().s, mask, reg_aux_A, A_offset(bd, rd), 4);
        for (int ld = 0; ld < ld_block2; ld++) {
            dot_product(accm(ld_block2, bd, ld), load(ld), bcst());
        }
    }
}

void jit_brgemm_kernel_t::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, int ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
        bool check_top_vpad, bool check_bottom_vpad, int rows_for_rd_tail,
        bool skip_accumulation) {

    Label ldb_loop_label;
    Label BS_loop_label;

    copy_post_ops_stack_values_to_aux(is_reg_tail);
    const int b_panel_ld_block2
            = use_mmla ? brgemm_utils::mmla_b_panel_ld_block2(brg) : ld_block2;

    auto ld_loop_body = [=](int vpad) {
        set_A_B_matrices();

        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        const auto bd_b = nstl::max(0, vpad);
        const auto bd_e = nstl::min(bd_block, bd_block + vpad);
        const auto is_valid_bd
                = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
        if (!is_valid_bd) return;

        int rdb_val = brg.rdb;
        int rdb_tail = brg.rdb_tail;
        if (use_gemv_path) {
            rdb_val = (brg.rd_block * brg.rdb + brg.rdb_tail)
                    / (cpu_sveLen / sizeof(float));
            rdb_tail = (brg.rd_block * brg.rdb + brg.rdb_tail)
                    % (cpu_sveLen / sizeof(float));
        }

        if (rdb_val > 0) {
            Label rdb_loop_label;
            mov(reg_rdb_loop, rdb_val);
            L(rdb_loop_label);
            {
                const bool is_rd_tail = false;

                if (use_mmla) {
                    gemm_microkernel_mmla(is_bdb_tail, ld_block2,
                            b_panel_ld_block2, is_rd_tail, true, vpad);
                } else if (use_gemv_path) {
                    gemv_microkernel(is_bdb_tail, ld_block2, is_rd_tail, vpad);
                } else {
                    if (n_bcast_1_load
                            && (brg.rd_block * brg.typesize_A) != 16) {
                        // GEMM n_bcast_1_load loads quadwords from left hand side when it can
                        // But we need to make sure we don't load more than one rd_block when loading quadwords
                        const int rd_block_quadtail
                                = brg.rd_block % (16 / brg.typesize_A);
                        if (brg.typesize_A == 1)
                            set_preg(
                                    rd_tail_mask.b, rd_block_quadtail, X_TMP_0);
                        if (brg.typesize_A == 2)
                            set_preg(
                                    rd_tail_mask.h, rd_block_quadtail, X_TMP_0);
                        if (brg.typesize_A == 4)
                            set_preg(
                                    rd_tail_mask.s, rd_block_quadtail, X_TMP_0);
                    }

                    gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);
                }

                if (!use_mmla) {
                    int A_rdb_off = use_gemv_path ? cpu_sveLen
                                                  : rdb_A_offset(bd_block);
                    add_vl_or_imm(reg_aux_A, reg_aux_A, A_rdb_off, X_TMP_0);
                    int B_rdb_off = use_gemv_path ? cpu_sveLen
                                                  : rdb_B_offset(ld_block2);
                    add_vl_or_imm(reg_aux_B, reg_aux_B, B_rdb_off, X_TMP_0);
                }

                subs(reg_rdb_loop, reg_rdb_loop, 1);
            }
            bne(rdb_loop_label);
        }
        if (rdb_tail != 0) {
            const bool is_rd_tail = true;

            if (use_mmla) {
                gemm_microkernel_mmla(is_bdb_tail, ld_block2, b_panel_ld_block2,
                        is_rd_tail, false, vpad);
            } else if (use_gemv_path) {
                // GEMV loads whole vector from left hand side
                const int k_tail
                        = brg.LDA % simd_elems(data_type::f32, brg.isa_impl);
                set_preg(rd_tail_mask.s, k_tail, X_TMP_0);

                gemv_microkernel(is_bdb_tail, ld_block2, is_rd_tail, vpad);
            } else {
                if (n_bcast_1_load) {
                    // GEMM n_bcast_1_load loads quadwords from left hand side when it can
                    // But we need to make sure we don't load more than one rd_tail when loading quadwords
                    const int rd_tail_quadtail
                            = brg.rdb_tail == 16 ? 16 : (brg.rdb_tail % 16);
                    if (brg.typesize_A == 1)
                        set_preg(rd_tail_mask.b, rd_tail_quadtail, X_TMP_0);
                    if (brg.typesize_A == 2)
                        set_preg(rd_tail_mask.h, rd_tail_quadtail, X_TMP_0);
                    if (brg.typesize_A == 4)
                        set_preg(rd_tail_mask.s, rd_tail_quadtail, X_TMP_0);
                    // Note that n_bcast_1_load may still need to deal with the tail
                } else {
                    // n_load_1_bcast loads at most a word, but we may need to predicate the
                    // last word load to avoid overstepping the A buffer
                    const auto rd_tail_size = brg.rdb_tail % brg.rd_step;
                    if (brg.typesize_A == 1)
                        set_preg(rd_tail_mask.b, rd_tail_size, X_TMP_0);
                    if (brg.typesize_A == 2)
                        set_preg(rd_tail_mask.h, rd_tail_size, X_TMP_0);
                    if (brg.typesize_A == 4)
                        set_preg(rd_tail_mask.s, rd_tail_size, X_TMP_0);
                }

                gemm_microkernel(bd_block2, is_bdb_tail, ld_block2, is_rd_tail,
                        is_ld_tail, vpad, rows_for_rd_tail);
            }
        }
    };
    if (is_ldb_loop_) { mov_imm(reg_ldb_loop, ldb_loop_length); }

    L(ldb_loop_label);
    {
        zero_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);

        if (is_ldb_loop_) {
            str_imm(reg_D, sp, reg_D_offs_);
        } else {
            mov(reg_ldb_loop, reg_D);
        }
        if (brg.brgattr.max_bs > 1) { str_imm(reg_aux_D, sp, reg_aux_D_offs_); }

        if (brg.alpha != 0.f && !skip_accumulation) {
            restore_A_B_matrices();

            if (brg.req_s8s8_compensation) { assert(!"unsupported\n"); }
            if (need_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
                str(reg_bdb_loop, ptr(sp, reg_bdb_loop_offs_));
                const auto reg32_scratch = WReg(reg_zp_a_input_shift.getIdx());
                mov(z_one_bytes().b, 1);
                ldr(reg32_scratch, ptr(sp, reg_zp_a_val_offs_));
                dup(z_zp_a_shift().s, reg32_scratch);
                ldr(reg_bdb_loop, ptr(sp, reg_bdb_loop_offs_));
            }

            if (brg.brgattr.max_bs > 1) { mov(reg_BS_loop, reg_BS); }
            L(BS_loop_label);
            {
                if (check_top_vpad || check_bottom_vpad) {
                    const auto vpad_first = -brg.brgattr.max_bottom_vpad;
                    const auto vpad_last = brg.brgattr.max_top_vpad;
                    const auto n_vpads = vpad_last - vpad_first + 2;
                    constexpr auto MAX_N_VPADS = 2 * brgemm_desc_t::MAX_VPAD;
                    assert(n_vpads < MAX_N_VPADS);

                    Label Vpad_loop_end_label;
                    std::vector<Label> Vpad_loop_iter_label(MAX_N_VPADS);
                    if (vpad_exist) {
                        XReg reg_batch = (brg.type == brgemm_addr)
                                ? reg_aux1_batch
                                : ((brg.type == brgemm_offs) ? reg_offs_batch
                                                             : reg_strd_batch);
                        if (brg.type == brgemm_strd) {
                            ldr_imm(reg_strd_batch, sp,
                                    origin_strd_batch_offs_);
                        }
                        ldr(reg_aux_A_vpad,
                                ptr(reg_batch,
                                        GET_OFF_BATCH_ELEMENT(vvpad.top)));

                        ldr(X_TMP_0,
                                ptr(reg_batch,
                                        GET_OFF_BATCH_ELEMENT(vvpad.bottom)));
                        sub(reg_aux_A_vpad, reg_aux_A_vpad, X_TMP_0);
                    } else {
                        eor(reg_aux_A_vpad, reg_aux_A_vpad, reg_aux_A_vpad);
                    }

                    for (int vpad = vpad_first; vpad <= vpad_last; vpad++) {
                        const auto label_vpad = vpad - vpad_first;
                        L(Vpad_loop_iter_label[label_vpad]);
                        if (!check_top_vpad && vpad > 0) continue;
                        if (!check_bottom_vpad && vpad < 0) continue;
                        auto real_vpad = vpad;
                        if (check_bottom_vpad && brg.bdb_tail) {
                            if (!is_bdb_tail) {
                                // for last full block before
                                // bdb_tail && -vpad greater than bdb_tail
                                if (brg.bdb_tail < -vpad)
                                    real_vpad += brg.bdb_tail;
                                else
                                    continue;
                            } else {
                                // for block with tail, call ldb_loop()
                                // to only calculate compensation for
                                // padding area when bdb_tail < -vpad for
                                // the cases using pre-cal compensation
                                if (brg.bdb_tail < -vpad && need_comp_pads
                                        && !brg.req_cal_comp_pads)
                                    real_vpad = -brg.bdb_tail;
                            }
                        }
                        cmp_imm(reg_aux_A_vpad, vpad, X_TMP_0);
                        b(NE, Vpad_loop_iter_label[label_vpad + 1]);
                        ld_loop_body(real_vpad);
                        b(Vpad_loop_end_label);
                    }
                    L(Vpad_loop_iter_label[n_vpads - 1]);
                    ld_loop_body(0);
                    L(Vpad_loop_end_label);
                } else {
                    ld_loop_body(0);
                }
                if (brg.brgattr.max_bs > 1) {
                    sub(reg_BS_loop, reg_BS_loop, 1);
                    cmp_imm(reg_BS_loop, 0, X_TMP_0);
                    b(GT, BS_loop_label);
                }
            }
        }

        if (is_ldb_loop_) {
            ldr_imm(reg_D, sp, reg_D_offs_);
        } else {
            mov(reg_D, reg_ldb_loop);
        }
        if (brg.brgattr.max_bs > 1) { ldr_imm(reg_aux_D, sp, reg_aux_D_offs_); }

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);
        if (is_ldb_loop_) {
            if (!is_ld_tail || use_mmla) {
                ldb_regs_shift(ld_block2);
            } else {
                ldb_regs_shift(1, true);
            }
            sub(reg_ldb_loop, reg_ldb_loop, 1);
            cmp_imm(reg_ldb_loop, 0, X_TMP_0);
            b(GT, ldb_loop_label);
        }
    }
}

void jit_brgemm_kernel_t::bdb_loop() {
    auto do_ldb_loop = [=](int bd_block2, bool is_bdb_tail, bool check_top_vpad,
                               bool check_bottom_vpad, int rows_for_rd_tail,
                               bool skip_accumulation) {
        auto call_ldb_loop = [&](int ld_block2, int ldb_loop_length,
                                     bool is_ld_reg_tail, bool is_ld_tail) {
            ldb_loop(bd_block2, is_bdb_tail, ld_block2, ldb_loop_length,
                    is_ld_reg_tail, is_ld_tail, check_top_vpad,
                    check_bottom_vpad, rows_for_rd_tail, skip_accumulation);
        };

        if (use_mmla) {
            if (brg.ldb2_tail > 0) {
                if (brg.ldb2 > 0) {
                    call_ldb_loop(brg.ld_block2, brg.ldb2, false, false);
                }
                const bool is_ld_reg_tail = brg.ldb2 > 0;
                const bool is_ld_tail = brg.ldb_tail > 0;
                call_ldb_loop(brg.ldb2_tail, 1, is_ld_reg_tail, is_ld_tail);
            } else if (brg.ldb2 > 0) {
                if (brg.ldb_tail > 0 && brg.ldb2 > 1) {
                    call_ldb_loop(brg.ld_block2, brg.ldb2 - 1, false, false);
                }
                const bool is_ld_reg_tail = brg.ldb_tail > 0 && brg.ldb2 > 1;
                const bool is_ld_tail = brg.ldb_tail > 0;
                const int ldb_loop_length = is_ld_tail ? 1 : brg.ldb2;
                call_ldb_loop(brg.ld_block2, ldb_loop_length, is_ld_reg_tail,
                        is_ld_tail);
            }
            return;
        }

        if (brg.ldb2 > 0) {
            call_ldb_loop(brg.ld_block2, brg.ldb2, false, false);
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = brg.ldb2 != 0;
            call_ldb_loop(brg.ldb2_tail, 1, is_ld_reg_tail, false);
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail = brg.ldb2 != 0 || brg.ldb2_tail != 0;
            call_ldb_loop(1, 1, is_ld_reg_tail, true);
        }
    };

    auto bdb_loop_body = [=](int bd_block2, bool is_bdb_tail,
                                 bool check_top_vpad, bool check_bottom_vpad,
                                 int rows_for_rd_tail, bool skip_accumulation) {
        do_ldb_loop(bd_block2, is_bdb_tail, check_top_vpad, check_bottom_vpad,
                rows_for_rd_tail, skip_accumulation);

        add_imm(reg_C, reg_C,
                use_gemv_path ? brg.bd_block * brg.typesize_C
                              : bdb_C_offset(bd_block2),
                X_TMP_0);
        add_imm(reg_D, reg_D,
                use_gemv_path ? brg.bd_block * brg.typesize_D
                              : bdb_D_offset(bd_block2),
                X_TMP_0);
        add_imm(reg_a_offset, reg_a_offset, bdb_A_offset(bd_block2), X_TMP_0);
    };

    int rows_for_rd_tail = 0;
    if (!use_mmla && brg.rdb_tail != 0 && (brg.is_bf16 || brg.is_int8)) {
        const auto rd_tail_size = brg.rdb_tail % brg.rd_step;
        rows_for_rd_tail = rd_tail_size
                ? div_up(brg.rd_step - rd_tail_size, brg.reduce_dim)
                : 0;
    }
    const int bd_blocks_for_rd_tail
            = div_up(nstl::max(0,
                             rows_for_rd_tail - brg.bdb_tail
                                     + brg.brgattr.max_bottom_vpad),
                    brg.bd_block);

    auto ld_block2 = (brg.ldb2 > 0) ? brg.ld_block2
                                    : ((brg.ldb2_tail > 0) ? brg.ldb2_tail : 1);
    const int free_vregs = max_effective_vregs - brg.req_s8s8_compensation;
    // For SVE 128 the shorter vector length seems to benefit from this approach
    n_bcast_1_load = (brg.is_int8 || simd_bytes(brg.isa_impl) == 16)
            // On A64FX, indexed DOT instructions use 2 uops, so we avoid them
            && simd_bytes(brg.isa_impl) != 64
            // we must use z0-z7 for indexed FMLAs, so we cannot do n_bcast_1_load if bd_block > 8
            && brg.bd_block <= 8
            && ((brg.bd_block * (ld_block2 + 1) < free_vregs)
                    && (bd_blocks_for_rd_tail == 0) && (rows_for_rd_tail == 0));

    // loop order may be specified in brgemm attributes
    if (brg.brgattr.hint_loop_order != brgemm_lo_default)
        n_bcast_1_load = (brg.brgattr.hint_loop_order == brgemm_lo_bl_1load)
                ? true
                : false;

    auto bdb_loop_sve512 = [=](bool skip_accumulation) {
        Label bdb_loop_end_label, no_vpad_label;
        if (vpad_exist) {
            // max_top_vp is restricted by bd_block due to
            // brgemm_kernel implementation. TODO: remove this restriction
            assert(brg.brgattr.max_top_vpad <= brg.bd_block
                    && brg.brgattr.max_bottom_vpad <= brg.bd_block);

            if (brg.type == brgemm_strd) {
                // if batch is nullptr then it means no vpadding in this call
                cmp_imm(reg_offs_batch, 0, X_TMP_0);
                b(EQ, no_vpad_label);
            }

            // first bd_block --------------
            auto bdblocks = brg.bdb;
            if (bdblocks >= 1) {
                bdb_loop_body(1, false, true,
                        (brg.bcast_dim - brg.brgattr.max_bottom_vpad)
                                < brg.bd_block,
                        brg.bdb - bd_blocks_for_rd_tail > 0 ? 0
                                                            : rows_for_rd_tail,
                        skip_accumulation);
                bdblocks--;
            }
            if (bdblocks > 1) {
                // middle bd_blocks -----------
                Label bdb_loop_label;
                mov_imm(reg_bdb_loop, bdblocks);
                L(bdb_loop_label);
                {
                    bdb_loop_body(1, false, false, false,
                            bd_blocks_for_rd_tail <= 1 ? 0 : rows_for_rd_tail,
                            skip_accumulation);

                    sub(reg_bdb_loop, reg_bdb_loop, 1);
                    cmp_imm(reg_bdb_loop, 1, X_TMP_0);
                    b(GT, bdb_loop_label);
                }
                bdblocks = 1;
            }
            if (bdblocks == 1) {
                // last bd_block ------------
                bdb_loop_body(1, false, false, true,
                        bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail,
                        skip_accumulation);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, brg.bdb < 1, true, rows_for_rd_tail,
                        skip_accumulation);
            // for brgemm_strd "no vpadding" case may be implemented, so skip it
            if (brg.type == brgemm_strd) /*jmp(bdb_loop_end_label);*/
                b(bdb_loop_end_label);
        }
        if (!vpad_exist || brg.type == brgemm_strd) {
            // for brgemm_strd batch may be null so we need this code path
            L(no_vpad_label);
            if (brg.bdb > 0) {
                mov_imm(reg_bdb_loop, brg.bdb);
                if (brg.bdb > (rows_for_rd_tail ? 1 : 0)) {
                    Label bdb_loop_label;
                    L(bdb_loop_label);
                    {
                        bdb_loop_body(1, false, false, false,
                                bd_blocks_for_rd_tail <= 1 ? 0
                                                           : rows_for_rd_tail,
                                skip_accumulation);
                        if (rows_for_rd_tail) {
                            sub(reg_bdb_loop, reg_bdb_loop, 1);
                            cmp_imm(reg_bdb_loop, 1, X_TMP_0);
                            b(GT, bdb_loop_label);
                        } else {
                            subs(reg_bdb_loop, reg_bdb_loop, 1);
                            b(GT, bdb_loop_label);
                        }
                    }
                }

                if (rows_for_rd_tail)
                    bdb_loop_body(1, false, false, true,
                            bd_blocks_for_rd_tail == 0 ? 0 : rows_for_rd_tail,
                            skip_accumulation);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, false, false, rows_for_rd_tail,
                        skip_accumulation);
        }
        L(bdb_loop_end_label);
    };

    auto bdb_loop_general = [=](bool skip_accumulation) {
        if (brg.type == brgemm_addr && brg.brgattr.max_bs == 1 && !vpad_exist
                && !skip_accumulation) {
            ldr(reg_aux1_A, ptr(reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            ldr(reg_aux1_B, ptr(reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
        }

        eor(reg_a_offset, reg_a_offset, reg_a_offset);
        bdb_loop_sve512(skip_accumulation);
    };

    if (brg.brgattr.generate_skip_accumulation) {
        Label bdb_loop_skip_acc_label, bdb_loop_done_label;
        ldr_imm(reg_skip_accm, sp, reg_skip_accm_offs_);
        cmp_imm(reg_skip_accm, 0, X_TMP_0);
        b(NE, bdb_loop_skip_acc_label);

        bdb_loop_general(false);
        b(bdb_loop_done_label);

        L(bdb_loop_skip_acc_label);
        bdb_loop_general(true);

        L(bdb_loop_done_label);
    } else
        bdb_loop_general(false);
}

void jit_brgemm_kernel_t::generate() {

    if (!one_of(brg.isa_impl, sve_512, sve_256, sve_128)) {
        assert(!"unsupported isa: jit_brgemm_kernel_t only supports SVE 512, "
                "256 and 128, this should have been checked earlier in the "
                "implementation");
    }

    size_t simd_w_ = simd_elems(data_type::f32, brg.isa_impl);

    preamble();
    sub_imm(sp, sp, utils::rnd_up(stack_space_needed_, 16), X_TMP_0);

    // Can we remove this?
    if (simd_w_ != cpu_sveLen / sizeof(float)) {
        set_preg(P_ALL_ONE.b, simd_w_ * 4, X_TMP_0);
    }

    mov(x7, x0);
    mov(x6, x1);
    mov(x1, x3);
    mov(x8, x4);
    mov(x9, x5);

    vpad_exist
            = brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0;
    need_comp_pads = IMPLICATION(brg.zp_type_a == brgemm_broadcast_t::none,
                             brg.req_s8s8_compensation)
            && IMPLICATION(!vpad_exist, brg.req_cal_comp_pads);

    set_preg(ld_tail_mask.s, brg.ldb_tail, X_TMP_0);
    if (brg.is_int8 && !brg.has_int8_vnni) { assert(!"unsupported\n"); }

    read_params();

    bdb_loop();

    add_imm(sp, sp, utils::rnd_up(stack_space_needed_, 16), X_TMP_0);
    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();
}

brgemm_attr_t::brgemm_attr_t()
    : max_bs(INT_MAX)
    , max_top_vpad(0)
    , max_bottom_vpad(0)
    , hint_expected_A_size(platform::get_per_core_cache_size(1))
    , hint_expected_B_size(platform::get_per_core_cache_size(1))
    , hint_expected_C_size(platform::get_per_core_cache_size(1))
    , hint_innermost_loop(brgemm_ld_loop_innermost)
    , hint_loop_order(brgemm_kernel_loop_order_t::brgemm_lo_default)
    , hint_prefetching(brgemm_kernel_prefetching_t::brgemm_prf_default)
    , wary_A_k_tail_read(true)
    , extendable_k(false)
    , generate_skip_accumulation(false)
    , bd_mask_level(0)
    , use_uker(false)
    , use_interleave_stores(false)
    , use_mmla(false)
    , use_mmla_packed_a(false)
    , LDA2(0)
    , LDB2(0)
    , LDC2_M(0)
    , LDC2_N(0)
    , bd_mask(nullptr)
    , static_offsets(nullptr) {}

brgemm_kernel_common_t::brgemm_kernel_common_t(const brgemm_desc_t abrd)
    : brgemm_kernel_(new jit_brgemm_kernel_t(abrd)) {}

status_t brgemm_kernel_common_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_kernel_common_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

const jit_generator_t *brgemm_kernel_common_t::get_jit_generator() const {
    return brgemm_kernel_;
}

brgemm_kernel_common_t::~brgemm_kernel_common_t() {
    delete brgemm_kernel_;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
