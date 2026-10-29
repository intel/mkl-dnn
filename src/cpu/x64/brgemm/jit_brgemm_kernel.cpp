/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/brgemm/jit_brgemm_kernel.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;
using namespace injector_utils;

template <typename Wmm>
struct jit_brgemm_kernel_t : public brgemm_kernel_t {
    jit_brgemm_kernel_t(const brgemm_desc_t &abrg)
        : brgemm_kernel_t(jit_name(), abrg.isa_impl)
        , brg(abrg)
        , postops_injector_(nullptr)
        , max_effective_vregs(get_max_effective_vregs(brg)) {

        // The implementation uses is_superset(), is_subset() utilities.
        // So avoid isa_all, isa_undef in these comparisons.
        assert(!utils::one_of(brg.isa_impl, isa_all, isa_undef));
        const dim_t is_ldb2_tail = brg.ldb2_tail ? 1 : 0;
        const dim_t is_ldb_tail = brg.ldb_tail ? 1 : 0;
        is_ldb_loop_ = brg.ldb2 + is_ldb2_tail + is_ldb_tail > 1;

        bool has_f8_e5m2_binary_postops = false;
        bool has_f8_e4m3_binary_postops = false;
        bool has_f8_dt = one_of(data_type::f8_e5m2, brg.dt_d, brg.dt_bias)
                || one_of(data_type::f8_e4m3, brg.dt_d, brg.dt_bias);
        if (brg.with_binary) {
            const auto &post_ops = brg.attr()->post_ops_;
            for (int i = 0; i < post_ops.len(); i++) {
                const auto &entry = post_ops.entry_[i];
                if (!entry.is_binary()) continue;
                has_f8_e5m2_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e5m2
                        || has_f8_e5m2_binary_postops;
                has_f8_e4m3_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e4m3
                        || has_f8_e4m3_binary_postops;
            }
        }

        if (brg.is_fp8 || has_f8_e5m2_binary_postops
                || has_f8_e4m3_binary_postops || has_f8_dt) {
            if (one_of(data_type::f8_e5m2, brg.dt_a, brg.dt_b, brg.dt_c,
                        brg.dt_d, brg.dt_bias)
                    || has_f8_e5m2_binary_postops)
                // Note: avoid using 'vmm0' since it is used as
                // 'fp8_to_f16_upconvert()' param and would collision with these
                // emulation vmms
                f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(this,
                        vmm_fp8_emu_aux1(), vmm_fp8_emu_aux2(),
                        vmm_fp8_emu_aux3(), kmask_fp8_aux, reg64_fp8_aux);
            if (one_of(data_type::f8_e4m3, brg.dt_a, brg.dt_b, brg.dt_c,
                        brg.dt_d, brg.dt_bias)
                    || has_f8_e4m3_binary_postops)
                f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this,
                        vmm_fp8_emu_aux1(), vmm_fp8_emu_aux2(),
                        vmm_fp8_emu_aux3(), vmm_fp8_emu_aux4(),
                        vmm_fp8_emu_aux5(), reg64_fp8_aux);
        }

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md());

            // The binary injector implicitly assumes it is safe to load
            // the entire vector along the load dimension when ldb_tail
            // is 0. In the case of GEMV, ldb_tail is always 0, but only
            // one element can be safely loaded. Therefore, we need to
            // provide information about what the tail size would have
            // been in a non-GEMV case.
            const int tail_size = !brg.is_gemv
                    ? brg.ldb_tail
                    : (brg.gemv_acc_is_vector() ? brg.gemv_tail : 1);
            const auto k_mask = !brg.is_gemv ? ld_tail_mask : gemv_partial_mask;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    vmm_tmp(0).getIdx(), this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, tail_size, k_mask,
                    use_exact_tail_scalar_bcast};

            // The transA GEMV kernel with `y` as a column vectorizes the
            // accumulators along M. For 3D dst {B, M, 1}, M is the channel
            // dimension for the injector, and rhs can only be {1, 1, 1},
            // {1, M, 1}, {B, 1, 1} or {B, M, 1}. The strategies below cover
            // these shapes without assuming vectorization along a spatial
            // dimension, unlike per_oc_spatial and per_mb_spatial. Other dst
            // need no changes, as M is the minibatch dimension for the
            // injector in 2D dst and a spatial one in 4D and higher dst.
            const bool is_transa_gemv_3d = brg.is_gemv
                    && brg.gemv_acc_is_vector() && !brg.treat_y_as_row
                    && brg.dst_md()->ndims == 3;
            const auto bcast_set = is_transa_gemv_3d
                    ? bcast_set_t {broadcasting_strategy_t::scalar,
                              broadcasting_strategy_t::per_oc,
                              broadcasting_strategy_t::per_mb,
                              broadcasting_strategy_t::no_broadcast}
                    : binary_injector::
                              get_all_strategies_supported_by_injector();
            const binary_injector::static_params_t bsp {this->param1, bcast_set,
                    rhs_sp, f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get()};

            postops_injector_ = utils::make_unique<po_injector_t>(this,
                    brg.attr()->post_ops_, bsp,
                    /* inject_sum = */ brg.with_sum);

            with_binary_non_scalar_bcast_ = binary_injector::
                    any_binary_postop_rhs_non_scalar_broadcast(
                            brg.attr()->post_ops_, dst_md_wrapper);
        }
        if (brg.is_bf16_emu)
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1(), bf16_emu_reserv_2(),
                    bf16_emu_reserv_3(), bf16_emu_scratch, bf16_emu_reserv_4(),
                    bf16_emu_reserv_4());
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

private:
    brgemm_desc_t brg;

    // The packed ACE B tensor is a sequence of N panels of brg.LDB columns,
    // consecutive panels being brgattr.LDB2 elements apart.
    bool ace_has_b_panels() const {
        return brg.is_ace() && brg.brgattr.LDB2 != 0 && brg.load_dim > brg.LDB;
    }

    enum matrix_kind_t { matrix_A, matrix_B };
    static constexpr int zmm_width_in_bytes_
            = cpu_isa_traits_t<avx512_core>::vlen;
    using Vmm =
            typename utils::conditional<std::is_same<Wmm, Xbyak::Tmm>::value,
                    Xbyak::Zmm, Wmm>::type;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
    using po_injector_t = injector::jit_uni_postops_injector_t<Vmm>;
    std::unique_ptr<po_injector_t> postops_injector_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;

    Xbyak::Label avx_tail_mask_;
    Xbyak::Label f16_perm_even_table_;
    Xbyak::Label f16_perm_odd_table_;
    using reg64_t = const Xbyak::Reg64;

    registry_scratchpad_t regscratchpad_ {*this, brg.isa_impl};

    // Register decomposition
    const reg64_savable_t param1 {regscratchpad_, abi_param1};
    const reg64_savable_backup_t param1_backup {param1};

    const reg64_savable_t reg_C {regscratchpad_, r15};
    const reg64_savable_t reg_aux_C {regscratchpad_, r14};
    const reg64_savable_backup_t reg_C_backup {reg_aux_C, r19};

    // r14 is used to work with C (via reg_aux_C alias) that happens outside of
    // the microkernel so using r14 (via reg_tmp_microkernel alias) inside the
    // microkernel is safe as long as its content is preserved after exiting
    // the microkernel.
    const reg64_t reg_tmp_microkernel = r14;

    const reg64_t reg_A = r13;
    const reg64_t reg_B = r12;

    const reg64_t reg_aux_A = r11;
    const reg64_t reg_aux_B = r10;
    const reg64_t reg_aux_A_vpad = r11;

    const reg64_savable_t reg_bdb_loop {regscratchpad_, r9, r16};
    const reg64_savable_t reg_ldb_loop {regscratchpad_, r8, r17};

    const reg64_t reg_stride_lda = r9;
    const reg64_t reg_stride_ldb = r8;
    const reg64_savable_t reg_stride_ld_block {regscratchpad_, r8};
    const reg64_t reg_s8_input_shift = r9;
    const reg64_t reg_zp_a_input_shift = r9;

    const reg64_t reg_BS_loop = rax;
    const reg64_t reg_rdb_loop = rbx;
    const reg64_t reg_BS = abi_not_param1;

    const reg64_t reg_a_offset = rdx;
    const reg64_t reg_b_offset = rsi;

    const reg64_savable_t reg_aux1_A {regscratchpad_, rbx, rbp, may_use_rbp()};
    const reg64_t reg_aux1_B = abi_param1;

    const reg64_savable_t reg_batch {regscratchpad_, r10};
    const reg64_savable_t reg_aux_batch {regscratchpad_, rbx};

    const reg64_savable_t reg_bias {regscratchpad_, rbx, r24};
    const reg64_savable_t reg_src_scales {regscratchpad_, rbx, r23};
    const reg64_savable_t reg_wei_scales {regscratchpad_, rbx, r22};
    const reg64_savable_t reg_dst_scales {regscratchpad_, rbx, r20};
    const reg64_savable_t reg_aux_bias {regscratchpad_, rbx, r18};
    const reg64_savable_t reg_zp_comp_a {regscratchpad_, rbx};
    const reg64_savable_t reg_aux_zp_comp_a {regscratchpad_, rbx};
    const reg64_savable_t reg_zp_comp_b {regscratchpad_, rbx, r25};
    const reg64_savable_t reg_aux_zp_comp_b {regscratchpad_, rbx, r30};
    const reg64_savable_t reg_zp_c_values {regscratchpad_, rbx, r31};
    const reg64_savable_t reg_aux_zp_c_values {regscratchpad_, rbx};
    const reg64_savable_t reg_D_shift_bytes {regscratchpad_, rbx};

    const reg64_savable_t reg_aux_src_scales {regscratchpad_, r10};
    const reg64_savable_t reg_aux_wei_scales {regscratchpad_, r10};
    const reg64_savable_t reg_aux_scale_adjust {regscratchpad_, r10};
    const reg64_savable_t reg_do_post_ops {regscratchpad_, rbx};
    const reg64_savable_t reg_do_comp {regscratchpad_, rbx};
    const reg64_savable_t reg_skip_accm {regscratchpad_, rbx};
    const reg64_t reg_tmp_gpr = rbx;
    const reg64_savable_t reg_zp_a_val {regscratchpad_, rbx};
    const reg64_savable_t reg_buf {regscratchpad_, rbx, r26};
    const reg64_savable_t reg_dynamic_C_offset {regscratchpad_, rbx};
    const reg64_savable_t reg_buf_aux {regscratchpad_, abi_param1};
    const reg64_savable_backup_t reg_buf_aux_backup {reg_buf_aux};
    const reg64_savable_t reg_aux_compensation {regscratchpad_, rbx, r29};
    const reg64_savable_t reg_buf_A {regscratchpad_, rbx, r26};

    // Base / working pointers for the per-(M,N) f32 compensation buffer used
    // by apply_per_mn_compensation. Lifecycle mirrors reg_C / reg_aux_C.
    const reg64_savable_t reg_per_mn_comp {regscratchpad_, rbx};
    const reg64_savable_t reg_aux_per_mn_comp {regscratchpad_, rbx};

    const reg64_savable_t reg_D {regscratchpad_, r11};
    const reg64_savable_t reg_aux_D {regscratchpad_, rax, r27};
    const reg64_savable_backup_t reg_aux_D_backup {reg_aux_D, r28};
    const reg64_savable_backup_t reg_aux_D_bdb_loop_backup {reg_aux_D, r19};
    const reg64_savable_t reg_D_bdb_loop_shift {regscratchpad_, rbx, r21};

    // ld columns already consumed in the current N panel of packed ACE B.
    // Only used when the ld loop can cross a panel, see ldb_regs_shift().
    const reg64_savable_t reg_ace_ldb_pos {
            regscratchpad_, rbx, ace_has_b_panels()};

    /* bf16 emulation */
    const reg64_t bf16_emu_scratch = rbx;

    // FP8 Convert
    // Note: registers (GPR and VMM) used in the fp8 emulator should not
    // intersect with the set of registers used in binary injector because fp8
    // emulator may be called from injector
    const reg64_t reg_converted_stride = rbx;
    const reg64_savable_t reg64_fp8_aux {regscratchpad_, r13};

    bool is_ldb_loop_ = false;
    bool with_binary_non_scalar_bcast_ = false;
    const int max_effective_vregs;

    // Number of ZMM registers per bd block for ACE microkernel.
    static constexpr int ace_zmms_per_bd_block
            = brgemm_desc_t::ace_zmms_per_bd_block;

    // Mask register allocation:
    // k1: ld_full_mask (B matrix full blocks)
    // k2: ace_load_A_mask (A matrix loading)
    // k7: ld_tail_mask (B matrix tail blocks)
    //
    // GEMM:
    //   ld_full_mask - all bits in the mask are set; used along the load
    //                  dimension.
    //   ld_tail_mask - only bits corresponding to the load dimension tail
    //                  are set.
    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(1);
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(7);

    // GEMV:
    //   gemv_full_mask    - all bits in the mask are set; used along the
    //                       broadcast dimension.
    //   gemv_partial_mask - only bits corresponding to the broadcast-dimension
    //                       tail are set when transA is true. When transA is
    //                       false, only one bit is set because we store one
    //                       element.
    Xbyak::Opmask gemv_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask gemv_partial_mask = Xbyak::Opmask(3);

    Xbyak::Opmask fp8_col_mask = Xbyak::Opmask(4);
    Xbyak::Opmask kmask_fp8_aux = Xbyak::Opmask(5);
    // Used for both AMX GEMM and GEMV code paths.
    Xbyak::Opmask rd_tail_mask = Xbyak::Opmask(6);
    Xbyak::Opmask fp8_tail_mask = Xbyak::Opmask(7);

    // Aliases gemv_full_mask. That is only safe because ACE implements GEMM
    // only, so the gemv path is never generated together with it.
    Xbyak::Opmask ace_load_A_mask = Xbyak::Opmask(2);

    static int get_max_effective_vregs(const brgemm_desc_t &brg) {
        auto used_vregs = 0;
        if (brg.is_int8 && !brg.has_int8_vnni)
            used_vregs = 2;
        else if (brg.is_fp8_via_convert() || brg.is_fp8_post_ops_emu())
            used_vregs = 5;
        else if (brg.is_f16_b_non_amx_vnni())
            used_vregs = 2;
        return isa_num_vregs(brg.isa_impl) - used_vregs;
    }

    Vmm accm(int ld_block, int bd, int ld) const noexcept {
        return Vmm(max_effective_vregs - 1 - (bd * ld_block + ld));
    }

    Vmm bcst(int bd = 0) const noexcept {
        if (brg.n_bcast_1_load) {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - bd;
            assert(idx > 0);
            return Vmm(idx);
        } else
            return Vmm(0);
    }

    Vmm load(int ld = 0) const noexcept {
        if (brg.n_bcast_1_load) {
            return Vmm(0);
        } else {
            int idx = max_effective_vregs - 1 - (brg.ld_block2 * brg.bd_block)
                    - ld;
            assert(idx > 0);
            return Vmm(idx);
        }
    }

    Vmm gemv_load_a() const noexcept {
        assert(brg.is_gemv);
        const int idx = max_effective_vregs - 1 - brg.gemv_bd_block() - 0;
        return Vmm(idx);
    }

    Vmm gemv_load_b() const noexcept {
        assert(brg.is_gemv);
        const int idx = max_effective_vregs - 1 - brg.gemv_bd_block() - 1;
        return Vmm(idx);
    }

    Vmm vmm_tmp(dim_t i) const noexcept {
        const int bd_block = brg.is_gemv ? brg.gemv_bd_block() : brg.bd_block;
        MAYBE_UNUSED(bd_block);
        assert(IMPLICATION(!brg.is_tmm,
                i >= 0 && i < max_effective_vregs - bd_block * brg.ld_block2));
        return Vmm(static_cast<int>(i));
    }

    Vmm vmm_tail_mask() const noexcept { return vmm_tmp(1); }
    Vmm vmm_beta() const noexcept { return vmm_tmp(1); }
    Vmm vmm_lbound() const noexcept { return vmm_tmp(1); }
    Vmm vmm_ubound() const noexcept { return vmm_tmp(0); }

    // ace_rd_steps is the number of rd steps in the ACE micro-kernel loop.
    // It is the number of ZMM registers used for each block of the A and B
    // matrices.
    int ace_rd_steps(int rd_block) const noexcept {
        return div_up(rd_block, brg.rd_step);
    }

    Xbyak::Zmm ace_zmm_tmp(int i) {
        assert(1 <= i && i <= 4);
        return Xbyak::Zmm(i);
    }

    // Returns the Zmm register for matrix A for ACE microkernel.
    // bdb: block index in bd dimension
    // rds: register index within the block
    Xbyak::Zmm ace_zmm_A(int bdb, int rds) {
        const int base_idx
                = ace_rd_steps(brg.rd_block) * (brg.n_bcast_1_load ? bdb : 0);
        const int idx = 5 + base_idx + rds;
        assert(idx < 32 && "ZMM register index overflow in ace_zmm_A");
        return Xbyak::Zmm(idx);
    }

    // Returns the Zmm register for matrix B for ACE microkernel.
    // bd_block2: number of bd blocks the caller is currently generating for
    // ldb: block index in ld dimension
    // rds: register index within the block
    Xbyak::Zmm ace_zmm_B(int bd_block2, int ldb, int rds) {
        // With n_bcast_1_load the A blocks sit below and every ldb is loaded
        // and consumed before the next one, so all ldb reuse one B slot.
        // Indexing by ldb here would run past zmm31 once ld_block2 >= 6.
        const int base_idx = ace_rd_steps(brg.rd_block)
                * (brg.n_bcast_1_load ? bd_block2 : (1 + ldb));
        const int idx = 5 + base_idx + rds;
        assert(idx < 32 && "ZMM register index overflow in ace_zmm_B");
        return Xbyak::Zmm(idx);
    }

    Vmm vmm_one_bytes() const noexcept { return Vmm(3); }
    Vmm vmm_zp_a_shift() const noexcept { return Vmm(2); }
    Vmm vmm_inp_shift() const noexcept { return Vmm(1); }

    /* bf16 emulation */
    Zmm bf16_emu_reserv_1() const noexcept { return Zmm(0); }
    Zmm bf16_emu_reserv_2() const noexcept { return Zmm(1); }
    Zmm bf16_emu_reserv_3() const noexcept { return Zmm(2); }
    Zmm bf16_emu_reserv_4() const noexcept { return Zmm(3); }
    // note: zmm reserv_5 is not necessary since it's only used for 'vdpbf16ps'

    // fp8 emulation convert
    Vmm vmm_fp8_emu_aux1() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 1);
    }
    Vmm vmm_fp8_emu_aux2() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 2);
    }
    Vmm vmm_fp8_emu_aux3() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 3);
    }
    Vmm vmm_fp8_emu_aux4() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 4);
    }
    Vmm vmm_fp8_emu_aux5() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 5);
    }
    Vmm vmm_fp8_load() const noexcept {
        // Re-use it as output when converting fp8 to f16 vnni
        return Vmm(isa_num_vregs(brg.isa_impl) - 5);
    }
    Vmm vmm_fp8_bcst() const noexcept {
        // Re-use it as output when converting fp8 to f16 vnni
        return Vmm(isa_num_vregs(brg.isa_impl) - 4);
    }

    Zmm zmm_tmp_1() const noexcept { return Zmm(1); }

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm int8_ones_words() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 1);
    }
    Vmm int8_dot_product_temp() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 2);
    }

    Vmm f16_perm_even_vreg() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 1);
    }
    Vmm f16_perm_odd_vreg() const noexcept {
        return Vmm(isa_num_vregs(brg.isa_impl) - 2);
    }

    template <typename U>
    U vmm_mask(const U umm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;
    void maybe_set_avx_mask(bool is_ld_tail);

    void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
            bool mask_flag, bool store, Xbyak::Opmask ktail_mask,
            int tail_size);

    void advance_ldb_post_op_regs();
    void restore_ldb_post_op_regs(int ld_block2);
    void advance_bdb_post_op_regs(int adj_bd_block);
    void restore_bdb_post_op_regs(int bd_block2);
    void ldb_regs_shift(int ld_block2, bool is_tail = false);
    void advance_bd_block2_post_op_regs(int bd_block2, bool is_tail = false);

    void copy_post_ops_stack_values_to_aux(bool is_reg_tail);
    void read_params();
    void zero_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);

    void fp8_to_f16_upconvert(int num_rows, int tile_num_col_bytes,
            reg64_t reg_base, dim_t offset, reg64_t reg_data_stride,
            data_type_t dt, bool is_rd_tail);
    void fp8_to_f16_upconvert(
            data_type_t dt, const Vmm &vmm_out, const Operand &op_in);
    void fp8_to_f16_upconvert_to_vnni(int num_rows, int tile_num_col_bytes,
            reg64_t reg_base, dim_t offset, reg64_t reg_data_stride,
            data_type_t dt, bool is_rd_tail);
    void reduce_gemv_accumulators(int bd_block);
    void store_accumulators(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_ld_tail, bool skip_accumulation);
    void store_accumulators_without_post_ops(
            int bd_block, int ld_block, bool is_ld_tail, bool is_bdb_tail);
    void store_accumulators_apply_post_ops(int bd_block, int ld_block,
            dim_t ldb_and_bdb_offset, bool is_ld_tail, bool is_bdb_tail);
    void apply_zp_s8s8_compensation(
            int bd_block, int ld_block, bool is_ld_tail);
    void apply_per_mn_compensation(int bd_block, int ld_block, bool is_ld_tail);
    void apply_alpha_beta(
            int bd_block, int ld_block, bool is_ld_tail, bool is_bdb_tail);
    void apply_wei_scales(int bd_block, int ld_block2, bool is_ld_tail,
            bool dq2ps_required, bool &dq2ps_cvt_done);
    void apply_src_scales_per_k(int bd_block, int ld_block2,
            bool dq2ps_required, bool &dq2ps_cvt_done);
    void apply_scales(int bd_block, int ld_block2, bool is_ld_tail);
    void apply_post_ops(int bd_block, int ld_block2, dim_t ldb_and_bdb_offset,
            bool is_ld_tail, bool is_bdb_tail);
    void gemv_apply_bias(int bd_block, bool is_bdb_tail);
    void gemv_apply_wei_scales(int bd_block, bool is_bdb_tail);

    void restore_A_B_matrices();
    void set_A_B_matrices();

    void compute_int8_compensation(int rd_loop, int bd_b, int bd_e,
            int bd_block, int ld_block2, bool is_ld_tail, int vpad);
    void maybe_pre_process_data(
            data_type_t dt, const Vmm &vmm_out1, const Vmm &vmm_out2);
    void maybe_pre_process_data(matrix_kind_t matrix_kind, const Tmm &t1,
            reg64_t reg_base, dim_t offset, reg64_t reg_stride, int num_rows,
            int num_col_bytes, bool is_rd_tail);
    bool maybe_pre_process_k_tail(bool is_rd_tail, const Tmm &t1,
            reg64_t reg_base, dim_t offset, reg64_t reg_stride,
            matrix_kind_t mk);
    void maybe_tileloadd_nt(matrix_kind_t matrix_kind, int idx, dim_t offset,
            bool is_rd_tail, bool is_tail);
    void maybe_pre_process_buf_A(reg64_t reg_base, const int bd_b,
            const int bd_e, const int rd_block);
    void load_scales_to_vmm(const data_type_t type_in, const Vmm &scales,
            const Xbyak::Operand &op, bool is_ld_tail, bool is_single_scale);
    void dot_product(Vmm v1, Vmm v2, Vmm v3);
    void gemm_microkernel(int bd_block2, bool is_bdb_tail, int ld_block,
            bool is_rd_tail, bool is_ld_tail, int vpad, int rows_for_rd_tail);
    void gemv_microkernel(bool is_bdb_tail, int ld_block, bool is_rd_tail);
    void gemm_microkernel_amx(int bd_block2, bool is_bdb_tail, int ld_block2,
            bool is_rd_tail, bool is_ld_tail);
    void gemm_microkernel_ace(int bd_block2, bool is_bdb_tail, int ld_block2,
            bool is_rd_tail, bool is_ld_tail);

    void ace_load_A_4x16bytes(
            const Zmm &zmm, size_t mask, const Reg64 &reg_A, dim_t offset);
    void ace_load_A(int bdb, dim_t offset, bool is_bdb_tail, int bd_block,
            bool is_rd_tail = false);
    void ace_load_B(int ldb, dim_t offset, int rdstep, bool is_ld_tail,
            int rd_block, int bd_block2);
    void outer_product(const Zmm &zmm_a, const Zmm &zmm_b, const Tmm &accm);

    void bs_loop(int bd_block2, bool is_bdb_tail, int ld_block, bool is_ld_tail,
            bool first_bdb, bool last_bdb, int rows_for_rd_tail,
            bool skip_accumulation);

    void ldb_loop(int bd_block2, bool is_bdb_tail, int ld_block,
            dim_t ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
            bool first_bdb, bool last_bdb, int rows_for_rd_tail,
            bool skip_accumulation);
    void bdb_loop();

    void generate() override;

    bool gemv_is_tail_acc(int bd, int bd_end, bool is_bdb_tail) const noexcept;
    int gemv_elems_per_vector_acc() const noexcept;

    dim_t A_offset(int bd, int rd, bool is_amx = false) const noexcept;
    dim_t B_offset(int ld, int rd, bool is_amx = false) const noexcept;
    dim_t C_offset(int bd, int ld) const noexcept;
    dim_t D_offset(int bd, int ld) const noexcept;

    dim_t rdb_A_offset() const noexcept;
    dim_t rdb_B_offset() const noexcept;

    dim_t ldb_B_offset(int ld_block2, bool is_tail = false) const noexcept;
    dim_t ldb_C_offset(int ld_block2, bool is_tail = false) const noexcept;
    dim_t ldb_D_offset(int ld_block2, bool is_tail = false) const noexcept;
    dim_t ldb_po_offset(int ld_block2, bool is_tail = false) const noexcept;

    dim_t bdb_A_offset(int bd_block2, bool is_tail = false) const noexcept;
    dim_t bdb_C_offset(int bd_block2, bool is_tail = false) const noexcept;
    dim_t bdb_D_offset(int bd_block2, bool is_tail = false) const noexcept;
    dim_t bdb_po_offset(int bd_block2, bool is_tail = false) const noexcept;

    dim_t bias_offset(int ld, bool is_tail = false) const noexcept;
    dim_t oc_logical_offset(int ld, bool is_tail = false) const noexcept;

    dim_t compensations_offset(int ld, bool is_tail = false) const noexcept;
    dim_t bdb_compensation_offset(
            int bd_block2, bool is_tail = false) const noexcept;
    dim_t bd_compensation_offset(int ld, int bd) const noexcept;
    dim_t wei_scales_offset(int ld, bool is_tail = false) const noexcept;
    dim_t zp_comp_a_offset(int ld, bool is_tail = false) const noexcept;
    dim_t bd_zp_comp_a_offset(int ld, int bd) const noexcept;
    dim_t bdb_zp_comp_a_offset(
            int bd_block2, bool is_tail = false) const noexcept;
    dim_t zp_comp_b_offset(int bd) const noexcept;
    dim_t bdb_zp_comp_b_offset(
            int bd_block2, bool is_tail = false) const noexcept;
    dim_t zp_c_values_offset(int ld, bool is_tail = false) const noexcept;

    // Offsets into the per-(M,N) f32 compensation buffer. The buffer mirrors
    // buf_C layout but with f32 stride (= brg.LDC elements).
    dim_t per_mn_comp_offset(int bd, int ld) const noexcept;
    dim_t ldb_per_mn_comp_offset(
            int ld_block2, bool is_tail = false) const noexcept;
    dim_t bdb_per_mn_comp_offset(int bd_block2) const noexcept;
    dim_t buf_A_offset(int bd, int rd) const noexcept;

    bool vpad_exist = false;
    bool need_comp_pads = false;
    palette_config_t palette_ {};
};

/**
 * Checks whether the current accumulator is the final partial vector
 * accumulator in a GEMV tail block.
 *
 * Applies to vector-accumulator GEMV path.
 *
 * @param bd           Index of the current accumulator.
 * @param bd_end       Number of accumulator indices in the current block.
 *                     Includes the final tail accumulator when gemv_tail > 0.
 * @param is_bdb_tail  Indicates that the current block is a tail block.
 */
template <typename Wmm>
bool jit_brgemm_kernel_t<Wmm>::gemv_is_tail_acc(
        int bd, int bd_end, bool is_bdb_tail) const noexcept {
    assert(brg.is_gemv);
    if (!brg.gemv_acc_is_vector()) return false;
    return (bd + 1) == bd_end && is_bdb_tail && brg.gemv_tail > 0;
}

/**
 * Returns the number of output elements stored in a single GEMV
 * vector accumulator.
 *
 * In transA GEMV, `bd_block` encodes both:
 *   - the SIMD width (number of elements per accumulator)
 *   - the number of accumulators (gemv_bd_block)
 *
 * This helper extracts the SIMD width by dividing the total bd_block
 * by the number of accumulator indices.
 */
template <typename Wmm>
int jit_brgemm_kernel_t<Wmm>::gemv_elems_per_vector_acc() const noexcept {
    assert(brg.is_gemv && brg.gemv_acc_is_vector());
    if (!brg.is_gemv || !brg.gemv_acc_is_vector()) return 1;
    return brg.bd_block / brg.gemv_bd_block();
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::A_offset(
        int bd, int rd, bool is_amx) const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_A * bd * gemv_elems_per_vector_acc();

    return (is_amx) ? brg.typesize_A * (bd * brg.bd_block * brg.LDA)
                    : brg.typesize_A * (bd * brg.LDA + rd);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::B_offset(
        int ld, int rd, bool is_amx) const noexcept {
    if (is_amx) {
        return brg.typesize_B * (brg.rd_step * ld * brg.ld_block);
    } else {
        const dim_t rdb0 = rd / brg.ld_step;
        // Note: Offsets for elements within vnni_granularity are expected to be
        // handled within gemm_microkernel (for ex: odd-even converts).
        // hence no `rd % brg.ld_step`
        return brg.typesize_B
                * (rdb0 * brg.ld_step * brg.LDB
                        + brg.ld_step * ld * brg.ld_block);
    }
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::C_offset(int bd, int ld) const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_C * bd * gemv_elems_per_vector_acc() * brg.LDC;

    const auto bd_shift = brg.is_runtime_ldc ? 0 : bd * brg.LDC;
    return brg.typesize_C * (bd_shift + ld * brg.ld_block);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::D_offset(int bd, int ld) const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_D * bd * gemv_elems_per_vector_acc() * brg.LDD;

    const auto bd_shift = brg.is_runtime_ldd ? 0 : bd * brg.LDD;
    return brg.typesize_D * (bd_shift + ld * brg.ld_block);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::rdb_A_offset() const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return static_cast<dim_t>(brg.rd_block) * brg.LDA * brg.typesize_A;
    return brg.rd_block_A_size();
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::rdb_B_offset() const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return static_cast<dim_t>(brg.rd_block) * brg.typesize_B;
    return brg.rd_block_B_size() * brg.LDB;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::ldb_B_offset(
        int ld_block2, bool is_tail) const noexcept {
    // For TMM/AMX with blocked format, use rd_step instead of ld_step
    const auto step = brg.is_tmm ? brg.rd_step : brg.ld_step;
    return (is_tail) ? brg.typesize_B * brg.ldb_tail * step
                     : brg.typesize_B * ld_block2 * brg.ld_block * step;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::ldb_C_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_C * brg.ldb_tail
                     : brg.typesize_C * ld_block2 * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::ldb_D_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.typesize_D * brg.ldb_tail
                     : brg.typesize_D * ld_block2 * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::ldb_po_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld_block2 * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_A_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_A * bd_block2 * bd_block;
    return brg.typesize_A * bd_block2 * bd_block * brg.LDA;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_C_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_C * bd_block2 * bd_block * brg.LDC;
    return bd_block2 * bd_block
            * (brg.is_runtime_ldc ? 1 : brg.typesize_C * brg.LDC);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_D_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return brg.typesize_D * bd_block2 * bd_block * brg.LDD;
    return bd_block2 * bd_block
            * (brg.is_runtime_ldd ? 1 : brg.typesize_D * brg.LDD);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_po_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    return bd_block2 * bd_block * brg.LDD;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bias_offset(
        int bias_dim, bool is_tail) const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return bias_dim * gemv_elems_per_vector_acc() * brg.typesize_bias;

    return (is_tail) ? brg.typesize_bias * brg.ldb_tail
                     : brg.typesize_bias * bias_dim * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::oc_logical_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? brg.ldb_tail : ld * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::compensations_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_compensation_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    return sizeof(int32_t) * bd_block2 * bd_block * brg.LDB;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bd_compensation_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::wei_scales_offset(
        int ld, bool is_tail) const noexcept {
    if (brg.is_gemv && brg.gemv_acc_is_vector())
        return ld * (brg.bd_block / brg.gemv_bd_block())
                * brg.is_per_n_wei_scales
                * types::data_type_size(brg.dt_wei_scales);

    const dim_t ld_offset = is_tail ? brg.ldb_tail : ld * brg.ld_block;
    return ld_offset * brg.is_per_n_wei_scales
            * types::data_type_size(brg.dt_wei_scales);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::zp_comp_a_offset(
        int ld, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                     : sizeof(int32_t) * ld * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_zp_comp_a_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    return sizeof(int32_t) * bd_block2 * bd_block * brg.LDB;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bd_zp_comp_a_offset(
        int ld, int bd) const noexcept {
    return sizeof(int32_t) * (ld * brg.ld_block + bd * brg.LDB);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::zp_comp_b_offset(int bd) const noexcept {
    return sizeof(int32_t) * bd;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_zp_comp_b_offset(
        int bd_block2, bool is_tail) const noexcept {
    const auto bd_block = is_tail ? brg.bdb_tail : brg.bd_block;
    return zp_comp_b_offset(bd_block2 * bd_block);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::zp_c_values_offset(
        int ld, bool is_tail) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg.ldb_tail
                         : sizeof(int32_t) * ld * brg.ld_block;
    }

    return 0;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::per_mn_comp_offset(
        int bd, int ld) const noexcept {
    return sizeof(float) * (bd * brg.LDC + ld * brg.ld_block);
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::ldb_per_mn_comp_offset(
        int ld_block2, bool is_tail) const noexcept {
    return (is_tail) ? sizeof(float) * brg.ldb_tail
                     : sizeof(float) * ld_block2 * brg.ld_block;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::bdb_per_mn_comp_offset(
        int bd_block2) const noexcept {
    return sizeof(float) * bd_block2 * brg.bd_block * brg.LDC;
}

template <typename Wmm>
dim_t jit_brgemm_kernel_t<Wmm>::buf_A_offset(int bd, int rd) const noexcept {
    return bd * zmm_width_in_bytes_ + sizeof(float16_t) * rd;
}

template <typename Wmm>
template <typename U>
U jit_brgemm_kernel_t<Wmm>::vmm_mask(const U vmm_in, bool mask_flag, bool store,
        Xbyak::Opmask ktail_mask) const {
    return mask_flag && isa_has_masks(brg.isa_impl)
            ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
            : vmm_in;
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::maybe_set_avx_mask(bool is_tail) {
    if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) return;
    // Note: `is_tail` may be true even when `gemv_tail == 0`.
    // In this case, the block is a tail block, but there is no partial
    // accumulator (only full accumulators are present).
    if (brg.is_gemv && brg.gemv_tail == 0) return;
    vmovups(vmm_tail_mask(), ptr[rip + avx_tail_mask_]);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::cvt2ps(data_type_t type_in, const Vmm vmm_in,
        const Xbyak::Operand &op, bool mask_flag, bool store,
        Xbyak::Opmask ktail_mask, int tail_size) {
    Vmm vmm = vmm_in;
    const bool has_tail = op.isMEM()
            && tail_size != vreg_traits_t<Vmm>::vlen / sizeof(float);
    if (IMPLICATION(has_tail, is_superset(brg.isa_impl, avx512_core))) {
        vmm = vmm_mask(vmm_in, mask_flag, store, ktail_mask);
    } else {
        load_data(type_in, vmm_in, op.getAddress(), tail_size);
        if (types::is_integral_dt(type_in)) uni_vcvtdq2ps(vmm_in, vmm_in);
        return;
    }
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: uni_vmovups(vmm, op); break;
        case data_type::bf16:
            uni_vpmovzxwd(vmm, op);
            uni_vpslld(vmm, vmm, 16);
            break;
        case data_type::f16: vcvtph2ps(vmm, op); break;
        case data_type::s8: uni_vpmovsxbd(vmm, op); break;
        case data_type::u8: uni_vpmovzxbd(vmm, op); break;
        case data_type::f8_e5m2: f8_e5m2_cvt_->vcvt_f8_to_f32(vmm, op); break;
        case data_type::f8_e4m3: f8_e4m3_cvt_->vcvt_f8_to_f32(vmm, op); break;

        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::advance_ldb_post_op_regs() {
    if (brg.with_bias) {
        reg_aux_bias.restore();
        add(reg_aux_bias, bias_offset(1));
        reg_aux_bias.save();
    }
    if (brg.with_wei_scales) {
        reg_aux_wei_scales.restore();
        add(reg_aux_wei_scales, wei_scales_offset(1));
        reg_aux_wei_scales.save();
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_a.restore();
        add(reg_aux_zp_comp_a, zp_comp_a_offset(1));
        reg_aux_zp_comp_a.save();
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        reg_aux_zp_c_values.restore();
        add(reg_aux_zp_c_values, zp_c_values_offset(1));
        reg_aux_zp_c_values.save();
    }
    if (brg.with_per_mn_compensation) {
        reg_aux_per_mn_comp.restore();
        add(reg_aux_per_mn_comp, ldb_per_mn_comp_offset(1));
        reg_aux_per_mn_comp.save();
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::restore_ldb_post_op_regs(int ld_block2) {
    if (brg.with_bias) {
        reg_aux_bias.restore();
        sub(reg_aux_bias, bias_offset(ld_block2 - 1));
        reg_aux_bias.save();
    }
    if (brg.with_wei_scales) {
        reg_aux_wei_scales.restore();
        sub(reg_aux_wei_scales, wei_scales_offset(ld_block2 - 1));
        reg_aux_wei_scales.save();
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_a.restore();
        sub(reg_aux_zp_comp_a, zp_comp_a_offset(ld_block2 - 1));
        reg_aux_zp_comp_a.save();
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        reg_aux_zp_c_values.restore();
        sub(reg_aux_zp_c_values, zp_c_values_offset(ld_block2 - 1));
        reg_aux_zp_c_values.save();
    }
    if (brg.with_per_mn_compensation) {
        reg_aux_per_mn_comp.restore();
        sub(reg_aux_per_mn_comp, ldb_per_mn_comp_offset(ld_block2 - 1));
        reg_aux_per_mn_comp.save();
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::advance_bdb_post_op_regs(int adj_bd_block) {
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_b.restore();
        add(reg_aux_zp_comp_b, bdb_zp_comp_b_offset(1));
        reg_aux_zp_comp_b.save();
    }
    if (brg.req_comp_pads_with_bcast
            && brg.zp_type_a != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_a.restore();
        add(reg_aux_zp_comp_a, bdb_compensation_offset(1));
        reg_aux_zp_comp_a.save();
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::restore_bdb_post_op_regs(int bd_block2) {
    bool post_processed = false;
    if (bd_block2 > 1) {
        if (brg.zp_type_b != brgemm_broadcast_t::none) {
            post_processed = true;
            reg_aux_zp_comp_b.restore();
            sub(reg_aux_zp_comp_b, bdb_zp_comp_b_offset(bd_block2 - 1));
            reg_aux_zp_comp_b.save();
        }
        if (brg.req_comp_pads_with_bcast
                && brg.zp_type_a != brgemm_broadcast_t::none) {
            reg_aux_zp_comp_a.restore();
            sub(reg_aux_zp_comp_a, bdb_compensation_offset(bd_block2 - 1));
            reg_aux_zp_comp_a.save();
        }
    }
    if (post_processed) reg_buf.restore();
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::ldb_regs_shift(int ld_block2, bool is_tail) {
    dim_t C_offset
            = (is_tail) ? ldb_C_offset(1, true) : ldb_C_offset(ld_block2);
    dim_t D_offset
            = (is_tail) ? ldb_D_offset(1, true) : ldb_D_offset(ld_block2);
    add(reg_aux_C, C_offset);
    add(reg_aux_D, D_offset);

    // Packed ACE B shifts non-linearly: N panels of brg.LDB columns sit
    // brgattr.LDB2 elements apart, columns inside a panel brg.rd_step apart.
    // The ld tail shift is unused, so only full blocks need the panel jump.
    if (ace_has_b_panels() && !is_tail) {
        const dim_t col_bytes = brg.typesize_B * brg.rd_step;
        const dim_t ld_cols = ld_block2 * brg.ld_block;
        const dim_t panel_cols = brg.LDB;
        const dim_t panel_bytes = brg.typesize_B * brg.brgattr.LDB2;

        if (ld_cols % panel_cols == 0) {
            // Whole panels per step: the position in the panel never changes.
            add(reg_b_offset, (ld_cols / panel_cols) * panel_bytes);
        } else {
            assert(panel_cols % ld_cols == 0);
            Xbyak::Label ld_no_panel_crossing, ld_shift_done;
            reg_ace_ldb_pos.restore();
            add(reg_ace_ldb_pos, ld_cols);
            cmp(reg_ace_ldb_pos, panel_cols);
            jl(ld_no_panel_crossing, T_NEAR);
            sub(reg_ace_ldb_pos, panel_cols);
            add(reg_b_offset, panel_bytes - (panel_cols - ld_cols) * col_bytes);
            jmp(ld_shift_done, T_NEAR);
            L(ld_no_panel_crossing);
            add(reg_b_offset, ld_cols * col_bytes);
            L(ld_shift_done);
            reg_ace_ldb_pos.save();
        }
    } else {
        add(reg_b_offset,
                (is_tail) ? ldb_B_offset(0, true) : ldb_B_offset(ld_block2));
    }

    if (brg.with_bias) {
        reg_aux_bias.restore();
        add(reg_aux_bias,
                (is_tail) ? bias_offset(1, true) : bias_offset(ld_block2));
        reg_aux_bias.save();
    }
    if (brg.req_s8s8_compensation) {
        reg_aux_compensation.restore();
        add(reg_aux_compensation,
                (is_tail) ? compensations_offset(1, true)
                          : compensations_offset(ld_block2));
        reg_aux_compensation.save();
    }
    if (brg.with_wei_scales) {
        reg_aux_wei_scales.restore();
        add(reg_aux_wei_scales,
                (is_tail) ? wei_scales_offset(1, true)
                          : wei_scales_offset(ld_block2));
        reg_aux_wei_scales.save();
    }
    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_a.restore();
        add(reg_aux_zp_comp_a,
                (is_tail) ? zp_comp_a_offset(1, true)
                          : zp_comp_a_offset(ld_block2));
        reg_aux_zp_comp_a.save();
    }
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        reg_aux_zp_c_values.restore();
        add(reg_aux_zp_c_values,
                (is_tail) ? zp_c_values_offset(1, true)
                          : zp_c_values_offset(ld_block2));
        reg_aux_zp_c_values.save();
    }
    if (brg.with_per_mn_compensation) {
        reg_aux_per_mn_comp.restore();
        add(reg_aux_per_mn_comp,
                (is_tail) ? ldb_per_mn_comp_offset(1, true)
                          : ldb_per_mn_comp_offset(ld_block2));
        reg_aux_per_mn_comp.save();
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::advance_bd_block2_post_op_regs(
        int bd_block2, bool is_tail) {
    if (brg.req_comp_pads_with_bcast && brg.req_s8s8_compensation) {
        reg_buf.restore();
        add(reg_buf, bdb_compensation_offset(bd_block2, is_tail));
        reg_buf.save();
    }

    if (brg.req_comp_pads_with_bcast
            && brg.zp_type_a != brgemm_broadcast_t::none) {
        reg_zp_comp_a.restore();
        add(reg_zp_comp_a, bdb_zp_comp_a_offset(bd_block2, is_tail));
        reg_zp_comp_a.save();
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        reg_zp_comp_b.restore();
        add(reg_zp_comp_b, bdb_zp_comp_b_offset(bd_block2, is_tail));
        reg_zp_comp_b.save();
    }
    if (brg.with_per_mn_compensation) {
        reg_per_mn_comp.restore();
        add(reg_per_mn_comp, bdb_per_mn_comp_offset(bd_block2));
        reg_per_mn_comp.save();
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::copy_post_ops_stack_values_to_aux(
        bool is_reg_tail) {
    if (!is_reg_tail) {
        mov(reg_aux_C, reg_C);
        mov(reg_aux_D, reg_D);
        xor_(reg_b_offset, reg_b_offset);
        if (ace_has_b_panels()) {
            xor_(reg_ace_ldb_pos, reg_ace_ldb_pos);
            reg_ace_ldb_pos.save();
        }
        if (brg.with_bias) {
            reg_bias.restore();
            reg_bias.saveTo(reg_aux_bias);
        }
        if (brg.req_s8s8_compensation) {
            reg_buf.restore();
            reg_buf.saveTo(reg_aux_compensation);
        }
        if (brg.with_wei_scales) {
            reg_wei_scales.restore();
            reg_wei_scales.saveTo(reg_aux_wei_scales);
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            reg_zp_comp_a.restore();
            reg_zp_comp_a.saveTo(reg_aux_zp_comp_a);
        }

        if (brg.zp_type_c != brgemm_broadcast_t::none) {
            reg_zp_c_values.restore();
            reg_zp_c_values.saveTo(reg_aux_zp_c_values);
        }
        if (brg.with_per_mn_compensation) {
            reg_per_mn_comp.restore();
            reg_per_mn_comp.saveTo(reg_aux_per_mn_comp);
        }
    }
    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        reg_zp_comp_b.restore();
        reg_zp_comp_b.saveTo(reg_aux_zp_comp_b);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::read_params() {
    Label label_done;

    if (brg.with_binary) param1.save();

    if (brg.type != brgemm_addr) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
        }
    }
    mov(reg_batch, ptr[param1 + GET_OFF(batch)]);
    reg_batch.save();

    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    // ptr_buf is re-used for passing compensations for
    // brg.req_s8s8_compensation case
    if (brg.is_tmm || brg.req_s8s8_compensation) {
        mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);
        reg_buf.save();
    }

    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        reg_bias.save();
    }
    if (brg.with_src_scales) {
        mov(reg_src_scales, ptr[param1 + GET_OFF(ptr_src_scales)]);
        reg_src_scales.save();
    }

    if (brg.with_wei_scales) {
        mov(reg_wei_scales, ptr[param1 + GET_OFF(ptr_wei_scales)]);
        reg_wei_scales.save();
    }

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
        reg_zp_comp_a.save();
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[param1 + GET_OFF(b_zp_compensations)]);
        reg_zp_comp_b.save();
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
        reg_zp_c_values.save();
    }

    if (brg.with_per_mn_compensation) {
        mov(reg_per_mn_comp, ptr[param1 + GET_OFF(ptr_per_mn_compensation)]);
        reg_per_mn_comp.save();
    }

    if (brg.with_dst_scales) {
        mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
        reg_dst_scales.save();
    }

    if (brg.is_runtime_ldc) {
        mov(reg_stride_ld_block, ptr[param1 + GET_OFF(dynamic_LDC)]);
        // typesize_C is always 1/2/4/8, so this shift amount is provably
        // bounded regardless of typesize_C's declared (dim_t) type.
        if (brg.typesize_C > 1) shl(reg_stride_ld_block, brg.typesize_C >> 1);
        reg_stride_ld_block.save();
    }

    if (brg.is_runtime_ldd) {
        mov(reg_D_shift_bytes, ptr[param1 + GET_OFF(dynamic_LDD)]);
        // typesize_D is always 1/2/4/8, so this shift amount is provably
        // bounded regardless of typesize_D's declared (dim_t) type.
        if (brg.typesize_D > 1) shl(reg_D_shift_bytes, brg.typesize_D >> 1);
        reg_D_shift_bytes.save();
    }

    if (brg.is_fp8_via_convert_non_amx() && brg.fp8_with_f16_vnni_block) {
        mov(reg_buf_A, ptr[param1 + GET_OFF(ptr_buf)]);
        reg_buf_A.save();
    }

    mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
    reg_do_post_ops.save();

    mov(reg_skip_accm, ptr[param1 + GET_OFF(skip_accm)]);
    reg_skip_accm.save();

    mov(reg_zp_a_val, ptr[param1 + GET_OFF(zp_a_val)]);
    reg_zp_a_val.save();

    mov(reg_do_comp, ptr[param1 + GET_OFF(do_apply_comp)]);
    reg_do_comp.save();
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::zero_accumulators(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_ld_tail,
        bool skip_accumulation) {
    if (brg.is_tmm) {
        // avoid usage of tile registers if there is no accumulation
        if (skip_accumulation) return;
        for_(int bdb = 0; bdb < bd_block2; bdb++)
        for (int ldb = 0; ldb < ld_block2; ldb++) {
            int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            tilezero(Tmm(brg.get_C_tensor(bdb, idx, is_bdb_tail, is_ld_tail)));
        }
    } else if (brg.is_gemv) {
        for (int bd = 0; bd < brg.gemv_bd_block(); bd++) {
            auto vmm = accm(1, bd, 0);
            uni_vpxor(vmm, vmm, vmm);
        }
    } else {
        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            uni_vpxor(vmm, vmm, vmm);
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::fp8_to_f16_upconvert(
        data_type_t dt, const Vmm &vmm_out, const Operand &op_in) {
    if (!one_of(dt, data_type::f8_e4m3, data_type::f8_e5m2)) {
        assert(!"unsupported data type");
        return;
    }

    if (dt == data_type::f8_e4m3)
        f8_e4m3_cvt_->vcvt_f8_to_f16(vmm_out, op_in);
    else if (dt == data_type::f8_e5m2)
        // skip setting q nan bit for performance purpose
        f8_e5m2_cvt_->vcvt_f8_to_f16_skip_q_nan(vmm_out, op_in);
}

// This method up-converts the data from bf8 to f16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::fp8_to_f16_upconvert(int num_rows,
        int tile_num_col_bytes, reg64_t reg_base, dim_t offset,
        reg64_t reg_data_stride, data_type_t dt, bool is_rd_tail) {

    int rd_block = is_rd_tail ? brg.rdb_tail : brg.rd_block;

    const dim_t max_num_cols
            = rd_block; //tile_num_col_bytes / sizeof(float16_t);
    const dim_t col_tail = max_num_cols % 32;
    auto zmm_1 = vmm_tmp(0);
    auto zmm_1_masked = col_tail ? zmm_1 | fp8_col_mask | T_z : zmm_1;

    assert(max_num_cols > 0);

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp8_col_mask, reg_tmp_gpr);
    }
    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_base + offset]);

    for (dim_t r = 0; r < num_rows; ++r) {
        if (dt == data_type::f8_e5m2)
            f8_e5m2_cvt_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else if (dt == data_type::f8_e4m3)
            f8_e4m3_cvt_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else
            assert(!"unsupported data type");

        vmovups(ptr[reg_buf_aux + r * zmm_width_in_bytes_], zmm_1);
        add(reg_data_aux, reg_data_stride);
    }
}

// This method up-converts and transforms the data from fp8_vnni to f16_vnni
// format. Generally used by matrix_B.
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::fp8_to_f16_upconvert_to_vnni(int num_rows,
        int tile_num_col_bytes, reg64_t reg_base, dim_t offset,
        reg64_t reg_data_stride, data_type_t dt, bool is_rd_tail) {
    const dim_t num_cols_ele = tile_num_col_bytes / 2; // 32 for full tile
    const dim_t num_N = num_cols_ele / 2; // 16 for full tile
    const auto zmm_2 = vmm_tmp(2);

    assert(num_N > 0 && "bad tile parameters");
    MAYBE_UNUSED(num_N);

    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_base + offset]);

    int rd_block = is_rd_tail ? brg.rdb_tail : brg.rd_block;
    const dim_t vnni_granularity = data_type_vnni_granularity(data_type::f16);
    const dim_t r_end = utils::div_up(rd_block, vnni_granularity);
    assert(r_end <= num_rows && "bad tile parameters");

    if (dt == data_type::f8_e5m2)
        f8_e5m2_cvt_->vcvt_f8_to_f16_vnni_block(static_cast<int>(r_end),
                reg_data_aux, reg_data_stride, reg_buf_aux);
    else if (dt == data_type::f8_e4m3)
        f8_e4m3_cvt_->vcvt_f8_to_f16_vnni_block(static_cast<int>(r_end),
                reg_data_aux, reg_data_stride, reg_buf_aux);
    else
        assert(!"unsupported data type");

    // zero rest of the tile data
    if (r_end < num_rows) {
        vpxord(zmm_2, zmm_2, zmm_2);
        for (dim_t r = r_end; r < num_rows; ++r)
            vmovups(ptr[reg_buf_aux + r * zmm_width_in_bytes_], zmm_2);
    }
}

/**
 * Apply bias for GEMV paths.
 *
 * Bias handling in GEMV depends on two orthogonal properties:
 *
 *   1. Accumulator shape
 *      - Scalar accumulator:
 *          each accumulator register holds a single output element
 *      - Vector accumulator:
 *          each accumulator register holds a SIMD-width vector of output
 *          elements
 *
 *   2. Output layout (treat_y_as_row)
 *      - false: bias is a single scalar (broadcast across all outputs)
 *      - true : bias is laid out along the output dimension
 *
 * In the current implementation:
 *   - gemv_acc_is_vector() == true  <=>  transA GEMV path
 *   - gemv_acc_is_vector() == false <=>  non-transA GEMV path
 *
 * These properties define 4 bias application scenarios:
 *
 * 1) Scalar accumulator, scalar bias
 *    (treat_y_as_row = false, gemv_acc_is_vector() = false)
 *    ------------------------------------------------------
 *    - Each accumulator holds 1 output element
 *    - Bias is a single scalar
 *    - Behavior:
 *        Load one scalar bias value and apply it to every accumulator.
 *
 * 2) Scalar accumulator, vector bias
 *    (treat_y_as_row = true, gemv_acc_is_vector() = false)
 *    ------------------------------------------------------
 *    - Each accumulator holds 1 output element
 *    - Bias is indexed per output element
 *    - Behavior:
 *        Each accumulator corresponds to a different output index.
 *        Load one scalar bias value for the current accumulator from
 *            bias_offset(bd) and apply it to that accumulator.
 *
 * 3) Vector accumulator, scalar bias
 *    (treat_y_as_row = false, gemv_acc_is_vector() = true)
 *    ------------------------------------------------------
 *    - Each accumulator holds simd_w output elements
 *    - Bias is a single scalar
 *    - Behavior:
 *        Load one scalar bias value, broadcast it to a SIMD vector
 *        and apply it to the entire accumulator register.
 *
 * 4) Vector accumulator, vector bias
 *    (treat_y_as_row = true, gemv_acc_is_vector() = true)
 *    ------------------------------------------------------
 *    - Each accumulator holds simd_w output elements
 *    - Bias is contiguous along the output dimension
 *    - Behavior:
 *        Each accumulator corresponds to a SIMD-width chunk of outputs.
 *        Load a vector of bias values for the current accumulator from
 *            bias_offset(bd) and apply it to that accumulator.
 *
 *        For the final accumulator (tail case), perform a masked load
 *        using gemv_tail to avoid reading past valid elements.
 *
 * @param bd_block
 *   Number of accumulator indices (`bd`) to process in the current block.
 *
 *   - For scalar accumulators, each `bd` corresponds to one output element.
 *   - For vector accumulators, each `bd` corresponds to one SIMD-width chunk
 *     of output elements.
 *   - For vector accumulators, `bd_block` includes the final accumulator
 *     used for the masked load when gemv_tail > 0.
 *
 * @param is_bdb_tail
 *   Indicates that the current block is the tail block along the GEMV
 *   broadcast dimension.
 *
 * Notes:
 * - 'bd' indexes accumulator registers. For vector accumulators, it also
 *   corresponds to second-level register blocking.
 * - Tail handling applies only to vector accumulators and only for the final
 *   accumulator when gemv_tail > 0.
 *
 * This helper assumes that accumulator layout, offsets and tail handling
 * are consistent with the GEMV blocking model used in the kernel.
 */
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemv_apply_bias(int bd_block, bool is_bdb_tail) {

    reg_aux_bias.restore();

    const bool is_bias_scalar = !brg.treat_y_as_row;
    auto bias = vmm_tmp(0);

    if (is_bias_scalar) {
        const auto addr = ptr[reg_aux_bias + bias_offset(0)];
        switch (brg.dt_bias) {
            case data_type::f32: vbroadcastss(bias, addr); break;
            case data_type::bf16:
                vpbroadcastw(bias, addr);
                uni_vpslld(bias, bias, 16);
                break;
            case data_type::f16:
                vpbroadcastw(Xmm(bias.getIdx()), addr);
                vcvtph2ps(Xmm(bias.getIdx()), Xmm(bias.getIdx()));
                vbroadcastss(bias, Xmm(bias.getIdx()));
                break;
            default: assert(!"unsupported bias data type");
        }
    }

    for (int bd = 0; bd < bd_block; bd++) {
        auto acc = accm(1, bd, 0);

        if (is_bias_scalar) {
            uni_vaddps(acc, acc, bias);
            continue;
        }

        const auto addr = ptr[reg_aux_bias + bias_offset(bd)];

        if (is_superset(brg.isa_impl, avx512_core)) {
            const bool use_partial_mask = !brg.gemv_acc_is_vector()
                    || gemv_is_tail_acc(bd, bd_block, is_bdb_tail);
            const auto k_mask
                    = use_partial_mask ? gemv_partial_mask : gemv_full_mask;

            switch (brg.dt_bias) {
                case data_type::f32:
                    uni_vmovups(bias | k_mask | T_z, addr);
                    break;
                case data_type::bf16:
                    uni_vpmovzxwd(bias | k_mask | T_z, addr);
                    uni_vpslld(bias, bias, 16);
                    break;
                case data_type::f16:
                    vcvtph2ps(bias | k_mask | T_z, addr);
                    break;
                default: assert(!"unsupported bias data type");
            }
        } else {
            if (brg.gemv_acc_is_vector()) {
                if (gemv_is_tail_acc(bd, bd_block, is_bdb_tail))
                    vmaskmovps(bias, vmm_tail_mask(), addr);
                else
                    uni_vmovups(bias, addr);
            } else {
                uni_vmovss(Xmm(bias.getIdx()), addr);
            }
        }

        if (brg.gemv_acc_is_vector())
            uni_vaddps(acc, acc, bias);
        else
            uni_vaddss(
                    Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(bias.getIdx()));
    }
}

/**
 * Apply weight scales for GEMV accumulators.
 *
 * This helper follows the same GEMV accumulator model as gemv_apply_bias():
 * scales are applied to either scalar or vector accumulators depending on
 * gemv_acc_is_vector(), and tail handling for vector accumulators is handled
 * via gemv_is_tail_acc(...).
 *
 * Unlike bias, weight scales have two scale modes:
 *   - common: one scalar scale for the whole operation
 *   - per-N (`is_per_n_wei_scales`): one scale value per element along N
 *
 * In GEMV, per-N scales collapse to a single scalar only when `y` is treated
 * as a column vector.
 *
 * When treat_y_as_row == true, `y` is treated as a row vector instead.
 * In that case, per-N scales remain a vector.
 *
 * Behavior:
 *   - common scale case:
 *       one scalar scale is loaded once and reused for all accumulators
 *
 *   - per-N scale case:
 *       scales are loaded for each accumulator from `wei_scales_offset(bd)`
 *       as either:
 *         - a scalar, for scalar accumulators
 *         - a vector, for vector accumulators
 *
 * For vector accumulators, the final tail accumulator uses a masked load when
 * gemv_tail > 0.
 *
 * @param bd_block
 *   Number of accumulator indices (`bd`) to process in the current block.
 *     - For scalar accumulators, each `bd` corresponds to one output element.
 *     - For vector accumulators, each `bd` corresponds to one SIMD-width chunk
 *       of output elements.
 *     - For vector accumulators, `bd_block` includes the final accumulator
 *       used for the masked load when gemv_tail > 0.
 *
 * @param is_bdb_tail
 *   Indicates that the current block is the tail block along the GEMV
 *   broadcast dimension.
 */
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemv_apply_wei_scales(
        int bd_block, bool is_bdb_tail) {

    reg_aux_wei_scales.restore();

    const Vmm vmm_wei_scales = vmm_tmp(0);

    if (brg.gemv_single_wei_scale()) {
        const auto addr = ptr[reg_aux_wei_scales + wei_scales_offset(0)];

        switch (brg.dt_wei_scales) {
            case data_type::f32: vbroadcastss(vmm_wei_scales, addr); break;
            case data_type::bf16:
                vpbroadcastw(vmm_wei_scales, addr);
                uni_vpslld(vmm_wei_scales, vmm_wei_scales, 16);
                break;
            case data_type::f16:
                vpbroadcastw(Xmm(vmm_wei_scales.getIdx()), addr);
                vcvtph2ps(Xmm(vmm_wei_scales.getIdx()),
                        Xmm(vmm_wei_scales.getIdx()));
                vbroadcastss(vmm_wei_scales, Xmm(vmm_wei_scales.getIdx()));
                break;
            default: assert(!"unsupported wei scales data type");
        }
    }

    for (int bd = 0; bd < bd_block; bd++) {
        auto acc = accm(1, bd, 0);

        if (brg.gemv_single_wei_scale()) {
            uni_vmulps(acc, acc, vmm_wei_scales);
            continue;
        }

        const auto addr = ptr[reg_aux_wei_scales + wei_scales_offset(bd)];

        if (is_superset(brg.isa_impl, avx512_core)) {
            const bool use_partial_mask = !brg.gemv_acc_is_vector()
                    || gemv_is_tail_acc(bd, bd_block, is_bdb_tail);
            const auto k_mask
                    = use_partial_mask ? gemv_partial_mask : gemv_full_mask;

            switch (brg.dt_wei_scales) {
                case data_type::f32:
                    uni_vmovups(vmm_wei_scales | k_mask | T_z, addr);
                    break;
                case data_type::bf16:
                    uni_vpmovzxwd(vmm_wei_scales | k_mask | T_z, addr);
                    uni_vpslld(vmm_wei_scales, vmm_wei_scales, 16);
                    break;
                case data_type::f16:
                    vcvtph2ps(vmm_wei_scales | k_mask | T_z, addr);
                    break;
                default: assert(!"unsupported wei scales data type");
            }
        } else {
            if (brg.gemv_acc_is_vector()) {
                if (gemv_is_tail_acc(bd, bd_block, is_bdb_tail))
                    vmaskmovps(vmm_wei_scales, vmm_tail_mask(), addr);
                else
                    uni_vmovups(vmm_wei_scales, addr);
            } else {
                uni_vmovss(Xmm(vmm_wei_scales.getIdx()), addr);
            }
        }

        if (brg.gemv_acc_is_vector())
            uni_vmulps(acc, acc, vmm_wei_scales);
        else
            uni_vmulss(Xmm(acc.getIdx()), Xmm(acc.getIdx()),
                    Xmm(vmm_wei_scales.getIdx()));
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_alpha_beta(
        int bd_block, int ld_block2, bool is_ld_tail, bool is_bdb_tail) {
    const bool apply_alpha = brg.alpha != 1.f;

    auto vmm_alpha = vmm_tmp(0);
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.alpha)));
        uni_vmovq(Xmm(vmm_alpha.getIdx()), reg_tmp_gpr);
        uni_vbroadcastss(vmm_alpha, Xmm(vmm_alpha.getIdx()));
    }

    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto vmm = accm(ld_block2, bd, ld);
        if (brg.do_dq2ps_cvt()) uni_vcvtdq2ps(vmm, vmm);
        if (apply_alpha) uni_vmulps(vmm, vmm, vmm_alpha);
    }

    if (brg.beta == 0.f) return;
    const bool use_vadd_for_beta = brg.beta == 1.f && !brg.do_dq2ps_cvt();
    const bool need_init_beta_vmm = brg.beta != 1.f;
    auto vmm_prev_dst = vmm_tmp(0);
    if (need_init_beta_vmm) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.beta)));
        uni_vmovq(Xmm(vmm_beta().getIdx()), reg_tmp_gpr);
        uni_vbroadcastss(vmm_beta(), Xmm(vmm_beta().getIdx()));
    }

    reg64_savable_guard_t reg_aux_guard(
            {{{&reg_aux_C}, brg.is_runtime_ldc && bd_block > 1},
                    {{&reg64_fp8_aux}, brg.is_fp8_via_convert()}});

    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        const auto k_mask = is_tail ? ld_tail_mask : ld_full_mask;
        auto vmm = accm(ld_block2, bd, ld);
        auto ptr_C = ptr[reg_aux_C + C_offset(bd, ld)];
        if (use_vadd_for_beta) {
            if (brg.is_gemv) {
                if (brg.gemv_acc_is_vector()) {
                    if (!gemv_is_tail_acc(bd, bd_block, is_bdb_tail)) {
                        uni_vaddps(vmm, vmm, ptr_C);
                    } else if (is_superset(brg.isa_impl, avx512_core)) {
                        uni_vmovups(
                                vmm_prev_dst | gemv_partial_mask | T_z, ptr_C);
                        uni_vaddps(vmm, vmm, vmm_prev_dst);
                    } else {
                        vmaskmovps(vmm_prev_dst, vmm_tail_mask(), ptr_C);
                        uni_vaddps(vmm, vmm, vmm_prev_dst);
                    }
                } else {
                    uni_vaddss(Xmm(vmm.getIdx()), Xmm(vmm.getIdx()), ptr_C);
                }
            } else if (IMPLICATION(is_tail,
                               is_superset(brg.isa_impl, avx512_core))) {
                auto vmm_masked = vmm_mask(vmm, is_tail, false, k_mask);
                if (brg.is_integer_acc())
                    uni_vpaddd(vmm_masked, vmm, ptr_C);
                else
                    uni_vaddps(vmm_masked, vmm, ptr_C);
            } else {
                vmaskmovps(vmm_prev_dst, vmm_tail_mask(), ptr_C);
                if (brg.is_integer_acc())
                    uni_vpaddd(vmm, vmm, vmm_prev_dst);
                else
                    uni_vaddps(vmm, vmm, vmm_prev_dst);
            }
        } else {
            assert(!brg.is_gemv && "int8 data type is not supported for gemv");
            const int ld_size = is_tail ? brg.ldb_tail : brg.ld_block;
            cvt2ps(brg.dt_c, vmm_prev_dst, ptr_C, is_tail, false, k_mask,
                    ld_size);
            if (brg.beta == 1.f)
                uni_vaddps(vmm, vmm, vmm_prev_dst);
            else
                uni_vfmadd231ps(vmm, vmm_prev_dst, vmm_beta());
        }
        if (brg.is_runtime_ldc && bd_block > 1 && ld == ld_block2 - 1)
            reg_stride_ld_block.addTo(reg_aux_C);
    }

    if (need_init_beta_vmm) maybe_set_avx_mask(is_ld_tail);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_post_ops(int bd_block, int ld_block2,
        dim_t ldb_and_bdb_offset, bool is_ld_tail, bool is_bdb_tail) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    reg64_savable_guard_t registers_guard({{{&param1_backup}, true},
            {{&reg_aux_D_backup}, brg.is_runtime_ldd && bd_block > 1}});

    if (brg.with_binary) param1.restore();

    const int bd_block_shift = brg.is_runtime_ldd ? 1 : bd_block;
    for (int bd_block_idx = 0; bd_block_idx < bd_block;
            bd_block_idx += bd_block_shift) {
        int bd_start = bd_block_idx;
        int bd_end = bd_start + bd_block_shift;

        const auto set_binary_injector_params = [&] {
            // Sum post-op requires binary parameters to be set.
            const bool set_for_binary
                    = brg.with_binary && with_binary_non_scalar_bcast_;
            if (!set_for_binary && !brg.with_sum) return;
            for_(int bd = bd_start; bd < bd_end; bd++)
            for (int ld = 0; ld < ld_block2; ld++) {
                const auto vmm_idx = accm(ld_block2, bd, ld).getIdx();

                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_aux_D);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, D_offset(bd, ld));

                // Due to the binary injector's assumptions (see the comment for
                // `binary_injector::rhs_arg_static_params_t`), we need to
                // provide accumulator registers as if this were a non-GEMV
                // case and a tail existed.
                const bool has_tail = !brg.is_gemv
                        ? is_ld_tail
                        : gemv_is_tail_acc(bd, bd_end, is_bdb_tail)
                                || !brg.gemv_acc_is_vector();
                if (has_tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        };

        set_binary_injector_params();

        postops_injector_->compute_vector_range(
                max_effective_vregs - bd_end * ld_block2,
                max_effective_vregs - bd_start * ld_block2, rhs_arg_params);

        if (brg.is_runtime_ldd && bd_block > 1)
            reg_D_shift_bytes.addTo(reg_aux_D);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::reduce_gemv_accumulators(int bd_block) {
    // At this point the broadcast registers are not used.
    auto workspace = bcst();
    for (int bd = 0; bd < bd_block; bd++) {
        auto acc = accm(1, bd, 0);
        regops::horizontal_add_ps(this, acc, workspace);
    }
}

/** Loads scales to a given vmm, applying masking if needed.
* Used in multiple places for loading scales for variety of supported data types.
* @param type_in data type of the scales to be loaded.
* @param vmm_scales vmm register to load the scales to.
* @param op an operand to load the scales from, must point to memory.
* @param is_ld_tail tail flag used for applying tail mask
* @param is_single_scale single scale flag, broadcasting if it's true.
*/
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::load_scales_to_vmm(const data_type_t type_in,
        const Vmm &vmm_scales, const Xbyak::Operand &op, bool is_ld_tail,
        bool is_single_scale) {
    const auto k_mask = is_ld_tail ? ld_tail_mask : ld_full_mask;
    if (is_single_scale) {
        switch (type_in) {
            case data_type::f32: vbroadcastss(vmm_scales, op); break;
            case data_type::bf16:
                vpbroadcastw(vmm_scales, op);
                uni_vpslld(vmm_scales, vmm_scales, 16);
                break;
            case data_type::f16:
                vpbroadcastw(vmm_scales, op);
                vcvtph2ps(Xmm(vmm_scales.getIdx()), Xmm(vmm_scales.getIdx()));
                vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
                break;
            default: assert(!"unsupported scales data type");
        }
    } else if (IMPLICATION(is_ld_tail, isa_has_masks(brg.isa_impl))) {
        const Vmm vmm_scales_masked
                = vmm_mask(vmm_scales, is_ld_tail, false, k_mask);
        const auto addr = op.getAddress();
        switch (type_in) {
            case data_type::f32:
                if (brg.is_gemv) {
                    if (!brg.treat_y_as_row)
                        uni_vmovss(vmm_scales_masked, addr);
                } else {
                    uni_vmovups(vmm_scales_masked, addr);
                }
                break;
            case data_type::bf16:
                uni_vpmovzxwd(vmm_scales_masked, addr);
                uni_vpslld(vmm_scales, vmm_scales, 16);
                break;
            case data_type::f16: vcvtph2ps(vmm_scales_masked, addr); break;
            default: assert(!"unsupported scales data type");
        }
    } else {
        assert(one_of(
                type_in, data_type::f32, data_type::bf16, data_type::f16));
        load_data(type_in, vmm_scales, op.getAddress(), brg.ldb_tail);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_wei_scales(int bd_block, int ld_block2,
        bool is_ld_tail, bool dq2ps_required, bool &dq2ps_cvt_done) {
    reg_aux_wei_scales.restore();
    for (int ld = 0; ld < ld_block2; ld++) {
        const auto addr = ptr[reg_aux_wei_scales + wei_scales_offset(ld)];
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;

        const Vmm vmm_wei_scales = vmm_tmp(0);
        load_scales_to_vmm(brg.dt_wei_scales, vmm_wei_scales, addr, is_tail,
                brg.is_single_wei_scale);

        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            if (dq2ps_required && !dq2ps_cvt_done) uni_vcvtdq2ps(vmm, vmm);
            uni_vmulps(vmm, vmm, vmm_wei_scales);
        }
    }
    dq2ps_cvt_done = true;
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_src_scales_per_k(int bd_block,
        int ld_block2, bool dq2ps_required, bool &dq2ps_cvt_done) {
    reg_src_scales.restoreTo(reg_aux_src_scales);
    const auto vmm_src_sc = vmm_tmp(0);
    for (int bd = 0; bd < bd_block; bd++) {
        const auto src_sc_addr
                = ptr[reg_aux_src_scales + bd * brg.src_scale_m_stride];
        load_scales_to_vmm(brg.dt_src_scales, vmm_src_sc, src_sc_addr,
                /*is_ld_tail=*/false, /*is_single_scale=*/true);
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            if (dq2ps_required && !dq2ps_cvt_done) uni_vcvtdq2ps(vmm, vmm);
            uni_vmulps(vmm, vmm, vmm_src_sc);
        }
    }
    dq2ps_cvt_done = true;
}

// Sibling stage to post-ops; called every brgemm K-call when per-K scales
// are active. Driver shifts scale pointers per call. Leaves accumulator as
// f32 so apply_alpha_beta uses vaddps against f32 buffer C.
template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_scales(
        int bd_block, int ld_block2, bool is_ld_tail) {
    assert(brg.has_per_k_scales());
    const bool dq2ps_required = brg.is_int8;
    // When per-MN compensation was applied before this call, dq2ps is
    // already done by apply_per_mn_compensation.
    bool dq2ps_cvt_done = brg.with_per_mn_compensation;

    if (brg.is_per_k_wei_scales)
        apply_wei_scales(bd_block, ld_block2, is_ld_tail, dq2ps_required,
                dq2ps_cvt_done);
    if (brg.is_per_k_src_scales)
        apply_src_scales_per_k(
                bd_block, ld_block2, dq2ps_required, dq2ps_cvt_done);
    if (dq2ps_required && !dq2ps_cvt_done) {
        for_(int ld = 0; ld < ld_block2; ld++)
        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            uni_vcvtdq2ps(vmm, vmm);
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::store_accumulators_apply_post_ops(int bd_block,
        int ld_block2, dim_t ldb_and_bdb_offset, bool is_ld_tail,
        bool is_bdb_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd)
            && !brg.has_per_k_scales();
    // vmulps with memory op is supported only with avx512 + f32 dt.
    const bool has_ptr_b_support = is_superset(brg.isa_impl, avx512_core)
            && brg.dt_src_scales == data_type::f32;

    // This flag tracks whether the conversion has happened, since it must be
    // done only once despite all scales and bias applications requiring it.
    // TODO: perform conversion in a dedicated loop if it is required as it is
    // done in brgemm_post_ops kernel?
    bool dq2ps_cvt_done
            = brg.has_per_k_scales() || brg.with_per_mn_compensation;

    if (brg.with_src_scales && !brg.is_per_k_src_scales) {
        reg_src_scales.restoreTo(reg_aux_src_scales);
        auto vmm_src_scales = vmm_tmp(0);
        if (!has_ptr_b_support)
            load_scales_to_vmm(brg.dt_src_scales, vmm_src_scales,
                    ptr[reg_aux_src_scales],
                    /*is_ld_tail=*/false, /*is_single_scale=*/true);

        for_(int ld = 0; ld < ld_block2; ld++)
        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            if (dq2ps_required && !dq2ps_cvt_done) uni_vcvtdq2ps(vmm, vmm);

            if (has_ptr_b_support) {
                vmulps(vmm, vmm, ptr_b[reg_aux_src_scales]);
            } else {
                vmulps(vmm, vmm, vmm_src_scales);
            }
        }
        dq2ps_cvt_done = true;
    }

    if (brg.with_wei_scales && !brg.is_per_k_wei_scales) {
        if (!brg.is_gemv) {
            apply_wei_scales(bd_block, ld_block2, is_ld_tail, dq2ps_required,
                    dq2ps_cvt_done);
        } else {
            gemv_apply_wei_scales(bd_block, is_bdb_tail);
        }
    }

    if (brg.with_weights_scale_adjust) {
        assert(!brg.is_gemv && "scale_adjust is not supported for gemv");
        // It's the only value that can be used for scale adjust so far.
        mov(reg_aux_scale_adjust, float2int(1.f / 0.5f));
        auto vmm_scale_adjust = vmm_tmp(0);
        auto xmm_scale_adjust = Xmm(vmm_scale_adjust.getIdx());
        uni_vmovq(xmm_scale_adjust, reg_aux_scale_adjust);
        uni_vbroadcastss(vmm_scale_adjust, xmm_scale_adjust);

        for_(int ld = 0; ld < ld_block2; ld++)
        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            if (dq2ps_required && !dq2ps_cvt_done) uni_vcvtdq2ps(vmm, vmm);
            uni_vmulps(vmm, vmm, vmm_scale_adjust);
        }
        dq2ps_cvt_done = true;
    }

    if (brg.is_fp8_via_convert()) reg64_fp8_aux.save();

    if (!brg.is_gemv) {
        reg_aux_bias.restore();

        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm_bias = vmm_tmp(0);
            if (brg.with_bias) {
                auto ptr_bias = ptr[reg_aux_bias + bias_offset(ld)];
                const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
                cvt2ps(brg.dt_bias, vmm_bias, ptr_bias, is_tail, false, k_mask,
                        is_tail ? brg.ldb_tail : brg.ld_block);
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (dq2ps_required && !dq2ps_cvt_done) uni_vcvtdq2ps(vmm, vmm);
                if (brg.with_bias) uni_vaddps(vmm, vmm, vmm_bias);
            }
        }
    } else if (brg.with_bias)
        gemv_apply_bias(bd_block, is_bdb_tail);

    if (brg.is_fp8_via_convert()) reg64_fp8_aux.restore();

    if (postops_injector_)
        apply_post_ops(bd_block, ld_block2, ldb_and_bdb_offset, is_ld_tail,
                is_bdb_tail);

    if (brg.with_dst_scales) {
        reg_dst_scales.restore();
        auto vmm_dst_scales = vmm_tmp(0);
        vbroadcastss(vmm_dst_scales, ptr[reg_dst_scales]);

        for_(int ld = 0; ld < ld_block2; ld++)
        for (int bd = 0; bd < bd_block; bd++) {
            auto vmm = accm(ld_block2, bd, ld);
            vmulps(vmm, vmm, vmm_dst_scales);
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        assert(!brg.is_gemv && "zp is not supported for gemv");
        reg_aux_zp_c_values.restore();
        auto vmm_zp_c = vmm_tmp(0);
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            if (is_superset(brg.isa_impl, avx512_core)) {
                uni_vcvtdq2ps(vmm_zp_c,
                        EVEX_compress_addr(reg_aux_zp_c_values, 0, true));
            } else {
                uni_vpbroadcastd(vmm_zp_c, ptr[reg_aux_zp_c_values]);
                uni_vcvtdq2ps(vmm_zp_c, vmm_zp_c);
            }
        }
        if (brg.is_fp8_via_convert()) reg64_fp8_aux.save();

        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
                dim_t zp_c_off = zp_c_values_offset(ld);
                if (is_superset(brg.isa_impl, avx512_core)) {
                    auto zp_c_addr
                            = EVEX_compress_addr(reg_aux_zp_c_values, zp_c_off);
                    cvt2ps(data_type::s32, vmm_zp_c, zp_c_addr, is_tail, false,
                            k_mask, is_tail ? brg.ldb_tail : brg.ld_block);
                } else {
                    cvt2ps(data_type::s32, vmm_zp_c,
                            ptr[reg_aux_zp_c_values + zp_c_off], is_tail, false,
                            k_mask, is_tail ? brg.ldb_tail : brg.ld_block);
                }
            }
            for (int bd = 0; bd < bd_block; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                uni_vaddps(vmm, vmm, vmm_zp_c);
            }
        }
        if (brg.is_fp8_via_convert()) reg64_fp8_aux.restore();
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    const bool use_sat_cvt
            = dt_requires_saturation && isa_has_sat_cvt(brg.isa_impl, brg.dt_d);
    assert(vmm_lbound().getIdx() != vmm_ubound().getIdx());
    if (dt_requires_saturation) {
        init_saturate_f32(vmm_lbound(), vmm_ubound(), reg_tmp_gpr,
                data_type::f32, brg.dt_d, false, use_sat_cvt);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                saturate_cvt_f32(vmm, vmm_lbound(), vmm_ubound(), brg.dt_d,
                        false, use_sat_cvt);
            }
        }
        // below call is not required as s32 doesn't use vmm_lbound
        // maybe_set_avx_mask(is_ld_tail);
    }

    if (brg.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

    reg64_savable_guard_t reg_aux_D_backup_guard(
            {&reg_aux_D_backup}, brg.is_runtime_ldd && bd_block > 1);

    if (brg.is_fp8_via_convert()) reg64_fp8_aux.save();

    if (is_superset(brg.isa_impl, avx10_2))
        prefetchrst2(ptr[reg_aux_D]);
    else if (brg.brgattr.hint_prefetchw == brgemm_prfw_store)
        prefetchw(ptr[reg_aux_D]);

    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        auto addr = ptr[reg_aux_D + D_offset(bd, ld)];
        auto vmm = accm(ld_block2, bd, ld);
        auto vmm_lower = Vmm_lower_t(vmm.getIdx());
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        if (brg.brgattr.hint_prefetchw == brgemm_prfw_loop_store
                && !is_superset(brg.isa_impl, avx10_2))
            prefetchw(addr);

        if (brg.is_gemv) {
            if (is_superset(brg.isa_impl, avx512_core)) {
                const bool use_partial_mask = !brg.gemv_acc_is_vector()
                        || gemv_is_tail_acc(bd, bd_block, is_bdb_tail);
                const auto k_mask
                        = use_partial_mask ? gemv_partial_mask : gemv_full_mask;

                switch (brg.dt_d) {
                    case data_type::f32: uni_vmovups(addr, vmm | k_mask); break;
                    case data_type::bf16:
                        vcvtneps2bf16(vmm_lower, vmm, get_encoding());
                        vmovdqu16(addr, vmm_lower | k_mask);
                        break;
                    case data_type::f16:
                        vcvtps2ph(vmm_lower, vmm, _op_mxcsr);
                        vmovdqu16(addr, vmm_lower | k_mask);
                        break;
                    default: assert(!"unsupported gemv dst data type");
                }
            } else {
                assert(brg.dt_d == data_type::f32);
                if (brg.gemv_acc_is_vector()) {
                    if (gemv_is_tail_acc(bd, bd_block, is_bdb_tail))
                        vmaskmovps(addr, vmm_tail_mask(), vmm);
                    else
                        uni_vmovups(addr, vmm);
                } else {
                    uni_vmovss(addr, vmm);
                }
            }
            // Non-gemv path.
        } else if (is_superset(brg.isa_impl, avx512_core)) {
            const Vmm r_vmm = vmm_mask(vmm, is_tail, true, k_mask);
            const Vmm_lower_t r_ymm
                    = vmm_mask(vmm_lower, is_tail, true, k_mask);
            const Xmm xmm = Xmm(vmm.getIdx());
            const Xmm r_xmm = vmm_mask(xmm, is_tail, true, k_mask);
            if (use_sat_cvt) {
                assert(one_of(brg.dt_d, data_type::s8, data_type::u8));
                auto vmm_perm = vmm_ubound();
                vpermb(vmm, vmm_perm, vmm);
                vmovdqu8(addr, r_xmm);
            } else {
                switch (brg.dt_d) {
                    case data_type::f32:
                    case data_type::s32: uni_vmovups(addr, r_vmm); break;
                    case data_type::bf16: // TODO - clean
                        if (brg.is_bf16_emu) {
                            bf16_emu_->vcvtneps2bf16(vmm_lower, vmm);
                        } else {
                            vcvtneps2bf16(vmm_lower, vmm);
                        }
                        vmovdqu16(addr, r_ymm);
                        break;
                    case data_type::f16:
                        vcvtps2ph(vmm_lower, vmm, _op_mxcsr);
                        vmovdqu16(addr, r_ymm);
                        break;
                    case data_type::f8_e5m2:
                        f8_e5m2_cvt_->vcvt_f32_to_f8(xmm, vmm);
                        vmovdqu8(addr, r_xmm);
                        break;
                    case data_type::f8_e4m3:
                        f8_e4m3_cvt_->vcvt_f32_to_f8(xmm, vmm);
                        vmovdqu8(addr, r_xmm);
                        break;
                    case data_type::s8: vpmovsdb(addr, r_vmm); break;
                    case data_type::u8: vpmovusdb(addr, r_vmm); break;
                    default: assert(!"unknown dst_dt");
                }
            }
        } else {
            const int ld_block = is_tail ? brg.ldb_tail : brg.ld_block;
            if (is_tail && types::data_type_size(brg.dt_b) == sizeof(float))
                vmaskmovps(addr, vmm_tail_mask(), vmm);
            else
                store_data(
                        brg.dt_d, vmm, reg_aux_D, D_offset(bd, ld), ld_block);
        }
        if (brg.is_runtime_ldd && bd_block > 1 && ld == ld_block2 - 1)
            reg_D_shift_bytes.addTo(reg_aux_D);
    }
    if (brg.is_fp8_via_convert()) reg64_fp8_aux.restore();
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_zp_s8s8_compensation(
        int bd_block, int ld_block2, bool is_ld_tail) {
    assert(!brg.is_gemv && "compensation is not supported for gemv");
    // apply compensation to accumulated values
    // to avoid the loss of accuracy when converting s32 to f32
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;

    if (!brg.req_cal_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = vmm_tmp(1);
        reg_zp_a_val.restore();
        uni_vpbroadcastd(vmm_zp_a_val, reg_zp_a_val.cvt32());

        reg_aux_zp_comp_a.restore();
        const auto vmm_zp_comp_a = vmm_tmp(0);
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            for (int bd = 0; bd < bd_block; bd++) {
                if (IMPLICATION(!brg.req_comp_pads_with_bcast, bd == 0)) {
                    const auto zp_comp_a_addr = ptr[reg_aux_zp_comp_a
                            + bd_zp_comp_a_offset(ld, bd)];
                    if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
                        auto vmm_zp_comp_a_masked = vmm_mask(
                                vmm_zp_comp_a, is_tail, false, k_mask);
                        uni_vmovups(vmm_zp_comp_a_masked, zp_comp_a_addr);
                    } else {
                        // cannot use vmaskmovps as vmm_zp_a_val clashes with
                        // vmm_tail_mask
                        load_data(data_type::s32, vmm_zp_comp_a, zp_comp_a_addr,
                                brg.ldb_tail);
                    }
                    uni_vpmulld(vmm_zp_comp_a, vmm_zp_comp_a, vmm_zp_a_val);
                }
                auto vmm = accm(ld_block2, bd, ld);
                uni_vpaddd(vmm, vmm, vmm_zp_comp_a);
            }
        }
        maybe_set_avx_mask(is_ld_tail);
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        reg_aux_zp_comp_b.restore();
        for (int bd = 0; bd < bd_block; bd++) {
            dim_t zp_comp_b_off = zp_comp_b_offset(bd);
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (is_superset(brg.isa_impl, avx512_core)) {
                    const auto zp_comp_b_addr = EVEX_compress_addr(
                            reg_aux_zp_comp_b, zp_comp_b_off, true);
                    uni_vpaddd(vmm, vmm, zp_comp_b_addr);
                } else {
                    const auto vmm_zp_comp_b = vmm_tmp(0);
                    uni_vpbroadcastd(vmm_zp_comp_b,
                            ptr[reg_aux_zp_comp_b + zp_comp_b_off]);
                    uni_vpaddd(vmm, vmm, vmm_zp_comp_b);
                }
            }
        }
    }

    if (!brg.req_cal_comp_pads && brg.req_s8s8_compensation) {
        reg_aux_compensation.restore();
        auto vmm_comp = vmm_tmp(0);
        for (int ld = 0; ld < ld_block2; ld++) {
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            for (int bd = 0; bd < bd_block; bd++) {
                if (IMPLICATION(!brg.req_comp_pads_with_bcast, bd == 0)) {
                    const auto comp_addr = ptr[reg_aux_compensation
                            + bd_compensation_offset(ld, bd)];
                    if (IMPLICATION(is_tail,
                                is_superset(brg.isa_impl, avx512_core))) {
                        auto vmm_comp_masked
                                = vmm_mask(vmm_comp, is_tail, false, k_mask);
                        uni_vmovups(vmm_comp_masked, comp_addr);
                    } else
                        vmaskmovps(vmm_comp, vmm_tail_mask(), comp_addr);
                }
                auto vmm = accm(ld_block2, bd, ld);
                uni_vpaddd(vmm, vmm, vmm_comp);
            }
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::apply_per_mn_compensation(
        int bd_block, int ld_block2, bool is_ld_tail) {
    assert(brg.with_per_mn_compensation);
    assert(!brg.is_gemv && "per-(M,N) compensation is not supported for gemv");
    assert(!brg.is_integer_acc()
            && "per-(M,N) compensation is not supported for int accumulation");

    if (brg.is_int8) {
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            uni_vcvtdq2ps(vmm, vmm);
        }
    }

    const auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    reg_aux_per_mn_comp.restore();
    const auto vmm_delta = vmm_tmp(0);
    for_(int bd = 0; bd < bd_block; bd++)
    for (int ld = 0; ld < ld_block2; ld++) {
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        const auto delta_addr
                = ptr[reg_aux_per_mn_comp + per_mn_comp_offset(bd, ld)];
        if (IMPLICATION(is_tail, isa_has_masks(brg.isa_impl))) {
            auto vmm_delta_masked = vmm_mask(vmm_delta, is_tail, false, k_mask);
            uni_vmovups(vmm_delta_masked, delta_addr);
        } else {
            vmaskmovps(vmm_delta, vmm_tail_mask(), delta_addr);
        }
        auto vmm = accm(ld_block2, bd, ld);
        uni_vsubps(vmm, vmm, vmm_delta);
    }
    maybe_set_avx_mask(is_ld_tail);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::store_accumulators_without_post_ops(
        int bd_block, int ld_block2, bool is_ld_tail, bool is_bdb_tail) {
    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dt_requires_saturation = brg.is_int8
            && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    const bool use_sat_cvt
            = dt_requires_saturation && isa_has_sat_cvt(brg.isa_impl, brg.dt_d);

    if (dt_requires_saturation) {
        init_saturate_f32(vmm_lbound(), vmm_ubound(), reg_tmp_gpr,
                data_type::f32, brg.dt_d, false, use_sat_cvt);
        for (int bd = 0; bd < bd_block; bd++) {
            for (int ld = 0; ld < ld_block2; ld++) {
                auto vmm = accm(ld_block2, bd, ld);
                saturate_cvt_f32(vmm, vmm_lbound(), vmm_ubound(), brg.dt_d,
                        false, use_sat_cvt);
            }
        }
        // below call is not required as s32 doesn't use vmm_lbound
        // maybe_set_avx_mask(is_ld_tail);
    }

    reg64_savable_guard_t reg_aux_C_guard(
            {&reg_aux_C}, brg.is_runtime_ldc && bd_block > 1);

    if (is_superset(brg.isa_impl, avx10_2))
        prefetchrst2(ptr[reg_aux_C]);
    else if (brg.brgattr.hint_prefetchw == brgemm_prfw_store)
        prefetchw(ptr[reg_aux_C]);

    if (brg.is_gemv) {
        for (int bd = 0; bd < bd_block; bd++) {
            const bool need_prefetch
                    = brg.brgattr.hint_prefetchw == brgemm_prfw_loop_store
                    && !is_superset(brg.isa_impl, avx10_2);

            const auto addr_c = ptr[reg_aux_C + C_offset(bd, 0)];
            if (need_prefetch) prefetchw(addr_c);

            auto vmm = accm(1, bd, 0);

            if (is_superset(brg.isa_impl, avx512_core)) {
                const bool use_partial_mask = !brg.gemv_acc_is_vector()
                        || gemv_is_tail_acc(bd, bd_block, is_bdb_tail);
                const auto k_mask
                        = use_partial_mask ? gemv_partial_mask : gemv_full_mask;
                uni_vmovups(addr_c | k_mask, vmm);
            } else {
                assert(brg.dt_d == data_type::f32);
                if (brg.gemv_acc_is_vector()) {
                    if (gemv_is_tail_acc(bd, bd_block, is_bdb_tail))
                        vmaskmovps(addr_c, vmm_tail_mask(), vmm);
                    else
                        uni_vmovups(addr_c, vmm);
                } else {
                    uni_vmovss(addr_c, Xmm(vmm.getIdx()));
                }
            }
        }
    } else {
        for_(int bd = 0; bd < bd_block; bd++)
        for (int ld = 0; ld < ld_block2; ld++) {
            auto vmm = accm(ld_block2, bd, ld);
            const auto addr_c = ptr[reg_aux_C + C_offset(bd, ld)];
            const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
            if (brg.brgattr.hint_prefetchw == brgemm_prfw_loop_store
                    && !is_superset(brg.isa_impl, avx10_2))
                prefetchw(addr_c);
            if (!is_tail)
                uni_vmovups(addr_c, vmm);
            else if (isa_has_masks(brg.isa_impl)) { // is_tail
                uni_vmovups(addr_c | ld_tail_mask | T_z, vmm);
            } else {
                vmaskmovps(addr_c, vmm_tail_mask(), vmm);
            }
            if (brg.is_runtime_ldc && bd_block > 1 && ld == ld_block2 - 1)
                reg_stride_ld_block.addTo(reg_aux_C);
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::store_accumulators(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_ld_tail,
        bool skip_accumulation) {
    const bool has_zero_points = !everyone_is(brgemm_broadcast_t::none,
            brg.zp_type_a, brg.zp_type_b, brg.zp_type_c);
    const bool are_post_ops_applicable = brg.are_post_ops_applicable();
    const bool need_to_apply_alpha_beta = brg.beta != 0.f || brg.alpha != 1.f;
    const bool need_generate_zp_a_compensation
            = brg.is_int8 && (brg.req_s8s8_compensation || has_zero_points);

    if (!brg.is_gemv)
        maybe_set_avx_mask(is_ld_tail);
    else if (brg.gemv_acc_is_vector())
        maybe_set_avx_mask(is_bdb_tail);

    if (brg.is_tmm) {
        if (need_to_apply_alpha_beta || are_post_ops_applicable
                || need_generate_zp_a_compensation)
            mov(reg_stride_ld_block, brg.ld_block * brg.typesize_C);
        else if (brg.is_runtime_ldc)
            reg_stride_ld_block.restore();
        else
            mov(reg_stride_ld_block, brg.LDC * brg.typesize_C);

        auto store_accumulators_amx
                = [&](const bool apply_post_ops,
                          const bool apply_zp_a_compensation = false) {
            if (brg.brgattr.max_bs > 1) reg_aux_D.restore();

            reg64_savable_guard_t reg_aux_D_bdb_loop_backup_guard(
                    {{{&reg_C_backup}, true},
                            {{&reg_aux_D_bdb_loop_backup}, apply_post_ops}});

            const bool do_accum_ops = need_to_apply_alpha_beta
                    || are_post_ops_applicable || apply_zp_a_compensation;
            const int adj_bd_block = (brg.is_M_tail && is_bdb_tail)
                    ? brg.bdb_tail
                    : brg.bd_block;

            if (brg.is_runtime_ldc && bd_block2 > 1) {
                xor_(reg_dynamic_C_offset, reg_dynamic_C_offset);
                reg_stride_ld_block.imulTo(
                        reg_dynamic_C_offset, bdb_C_offset(1));
                reg_dynamic_C_offset.save();
            }

            if (apply_post_ops && brg.is_runtime_ldd && bd_block2 > 1) {
                xor_(reg_D_bdb_loop_shift, reg_D_bdb_loop_shift);
                reg_D_shift_bytes.imulTo(reg_D_bdb_loop_shift, bdb_D_offset(1));
                reg_D_bdb_loop_shift.save();
            }

            reg_buf.restore();

            for (int bdb = 0; bdb < bd_block2; bdb++) {
                for (int ldb = 0; ldb < ld_block2; ldb++) {
                    const int idx = is_ld_tail ? brg.ld_block2 : ldb;
                    const int c_tensor = brg.get_C_tensor(
                            bdb, idx, is_bdb_tail, is_ld_tail);
                    if (do_accum_ops) {
                        if (skip_accumulation) {
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                auto vreg_acc = accm(1, bd, 0);
                                uni_vpxor(vreg_acc, vreg_acc, vreg_acc);
                            }
                        } else if (brg.is_ace()) {
                            // For ACE, extract the rows directly from the
                            // tile with tilemovrow.
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                const auto vreg_idx = accm(1, bd, 0).getIdx();
                                const auto vreg_zmm = Zmm(vreg_idx);
                                tilemovrow(vreg_zmm, Tmm(c_tensor), bd);
                            }
                        } else {
                            tilestored(ptr[reg_buf + reg_stride_ld_block],
                                    Tmm(c_tensor));
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                const size_t buf_offset
                                        = (bd * brg.ld_block) * brg.typesize_C;
                                auto vreg_acc = is_ld_tail
                                        ? accm(1, bd, 0) | ld_tail_mask | T_z
                                        : accm(1, bd, 0);
                                uni_vmovups(
                                        vreg_acc, ptr[reg_buf + buf_offset]);
                            }
                        }

                        if (apply_zp_a_compensation)
                            apply_zp_s8s8_compensation(
                                    adj_bd_block, 1, is_ld_tail);

                        if (need_to_apply_alpha_beta)
                            apply_alpha_beta(
                                    adj_bd_block, 1, is_ld_tail, is_bdb_tail);

                        if (apply_post_ops) {
                            const size_t ldb_and_bdb_offset
                                    = ldb_po_offset(ldb) + bdb_po_offset(bdb);
                            store_accumulators_apply_post_ops(adj_bd_block, 1,
                                    ldb_and_bdb_offset, is_ld_tail,
                                    is_bdb_tail);
                            if (ldb < ld_block2 - 1) {
                                advance_ldb_post_op_regs();
                                add(reg_aux_D, ldb_D_offset(1));
                            }
                        } else {
                            store_accumulators_without_post_ops(
                                    adj_bd_block, 1, is_ld_tail, is_bdb_tail);
                        }

                        if (ldb < ld_block2 - 1)
                            add(reg_aux_C, ldb_C_offset(1));

                        reg_buf.restore();
                    } else {
                        auto tmm = Tmm(c_tensor);
                        if (skip_accumulation) tilezero(tmm);
                        if (brg.is_ace()) {
                            // For ACE, extract and store rows using tilemovrow.
                            // tilemovrow returns one dword per element (f32 for
                            // bf16, s32 for int8), so a dword-granular vmovups
                            // stores a full row; the tail uses ld_tail_mask.
                            const size_t ldb_offset = ldb * ldb_C_offset(1);
                            for (int bd = 0; bd < adj_bd_block; bd++) {
                                const size_t c_offset = ldb_offset
                                        + (bd * brg.LDC) * brg.typesize_C;
                                const auto vreg_zmm
                                        = Zmm(31); // Use temp register
                                tilemovrow(vreg_zmm, tmm, bd);
                                if (is_ld_tail) {
                                    uni_vmovups(ptr[reg_aux_C + c_offset]
                                                    | ld_tail_mask | T_z,
                                            vreg_zmm);
                                } else {
                                    uni_vmovups(ptr[reg_aux_C + c_offset],
                                            vreg_zmm);
                                }
                            }
                        } else {
                            tilestored(
                                    ptr[reg_aux_C + reg_stride_ld_block], tmm);
                            if (ldb < ld_block2 - 1)
                                add(reg_aux_C, ldb_C_offset(1));
                        }
                    }
                }
                // Reset reg_aux_C back to column 0 after ldb loop.
                // The do_accum_ops path and the non-ACE tilestored path
                // advance reg_aux_C by ldb_C_offset(1) per ldb iteration.
                // The ACE do_accum_ops=false path uses compile-time offsets
                // and does not modify reg_aux_C.
                if (ld_block2 > 1 && (do_accum_ops || !brg.is_ace()))
                    sub(reg_aux_C, ldb_C_offset(ld_block2 - 1));
                if (bdb < bd_block2 - 1) {
                    if (brg.is_runtime_ldc)
                        reg_dynamic_C_offset.addTo(reg_aux_C);
                    else
                        add(reg_aux_C, bdb_C_offset(1));
                }

                if (apply_post_ops) {
                    bool post_processed = false;
                    if (ld_block2 > 1) {
                        sub(reg_aux_D, ldb_D_offset(ld_block2 - 1));
                        restore_ldb_post_op_regs(ld_block2);
                        post_processed |= utils::one_of(true, brg.with_bias,
                                brg.zp_type_a != brgemm_broadcast_t::none,
                                brg.zp_type_c == brgemm_broadcast_t::per_n,
                                brg.with_src_scales, brg.with_wei_scales,
                                brg.with_dst_scales);
                    }
                    if (bdb < bd_block2 - 1) {
                        if (brg.is_runtime_ldd)
                            reg_D_bdb_loop_shift.addTo(reg_aux_D);
                        else
                            add(reg_aux_D, bdb_D_offset(1));

                        advance_bdb_post_op_regs(adj_bd_block);
                        post_processed |= utils::one_of(true,
                                brg.zp_type_b != brgemm_broadcast_t::none,
                                brg.req_comp_pads_with_bcast
                                        && brg.zp_type_a
                                                != brgemm_broadcast_t::none);
                    }
                    if (post_processed) reg_buf.restore();
                }
            }
            if (apply_post_ops) { restore_bdb_post_op_regs(bd_block2); }
        };

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_skip_post_ops;
            reg_do_post_ops.restore();
            cmp(reg_do_post_ops, 0);
            jz(label_skip_post_ops, T_NEAR);
            if (need_generate_zp_a_compensation) {
                Label label_skip_zp_comp_with_postops;
                reg_do_comp.restore();
                cmp(reg_do_comp, 0);
                jz(label_skip_zp_comp_with_postops, T_NEAR);
                store_accumulators_amx(true, true);
                jmp(label_done, T_NEAR);

                L_aligned(label_skip_zp_comp_with_postops);
            }
            store_accumulators_amx(true);

            jmp(label_done, T_NEAR);

            L_aligned(label_skip_post_ops);
        }

        if (need_generate_zp_a_compensation) {
            Label label_skip_zp_comp;
            reg_do_comp.restore();
            cmp(reg_do_comp, 0);
            jz(label_skip_zp_comp, T_NEAR);
            store_accumulators_amx(false, true);
            jmp(label_done, T_NEAR);

            L_aligned(label_skip_zp_comp);
        }

        store_accumulators_amx(false);
        L_aligned(label_done);
    } else {
        int bd_block = 0;
        if (brg.is_gemv)
            bd_block = brg.gemv_num_acc_blocks(is_bdb_tail);
        else
            bd_block = is_bdb_tail ? brg.bdb_tail : brg.bd_block;
        assert(bd_block > 0);
        if (brg.is_gemv && !brg.gemv_acc_is_vector())
            reduce_gemv_accumulators(bd_block);

        if (need_generate_zp_a_compensation) {
            Label label_store_without_comp;
            reg_do_comp.restore();
            cmp(reg_do_comp, 0);
            jz(label_store_without_comp, T_NEAR);
            apply_zp_s8s8_compensation(bd_block, ld_block2, is_ld_tail);

            L_aligned(label_store_without_comp);
        }

        if (brg.with_per_mn_compensation)
            apply_per_mn_compensation(bd_block, ld_block2, is_ld_tail);
        if (brg.has_per_k_scales())
            apply_scales(bd_block, ld_block2, is_ld_tail);

        if (need_to_apply_alpha_beta)
            apply_alpha_beta(bd_block, ld_block2, is_ld_tail, is_bdb_tail);

        Label label_done;
        if (are_post_ops_applicable) {
            Label label_skip_post_ops;
            reg_do_post_ops.restore();
            cmp(reg_do_post_ops, 0);
            jz(label_skip_post_ops, T_NEAR);
            store_accumulators_apply_post_ops(
                    bd_block, ld_block2, 0, is_ld_tail, is_bdb_tail);
            jmp(label_done, T_NEAR);

            L_aligned(label_skip_post_ops);
        }
        store_accumulators_without_post_ops(
                bd_block, ld_block2, is_ld_tail, is_bdb_tail);
        L_aligned(label_done);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::restore_A_B_matrices() {
    // called at the start of bs loop
    // restore batch pointer
    reg_batch.restore();
    // save batch pointer in reg_aux_batch
    reg_batch.saveTo(reg_aux_batch);

    if (brg.type != brgemm_addr) {
        mov(reg_aux1_A, reg_A);
        reg_aux1_A.save();
        mov(reg_aux1_B, reg_B);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::set_A_B_matrices() {
    // called on each iteration of bs loop
    if (brg.type == brgemm_addr) {
        const bool is_row_major = (brg.layout == brgemm_row_major);
        if (brg.brgattr.max_bs > 1) {
            reg_aux_batch.restore();
            const size_t offA = is_row_major ? GET_OFF_BATCH_ELEMENT(ptr.A)
                                             : GET_OFF_BATCH_ELEMENT(ptr.B);
            const size_t offB = is_row_major ? GET_OFF_BATCH_ELEMENT(ptr.B)
                                             : GET_OFF_BATCH_ELEMENT(ptr.A);

            mov(reg_aux_A, ptr[reg_aux_batch + offA]);
            mov(reg_aux_B, ptr[reg_aux_batch + offB]);

            // Advance to next batch element and prefetch
            add(reg_aux_batch, sizeof(brgemm_batch_element_t));
            reg_aux_batch.save();
            prefetcht0(ptr[reg_aux_batch]);
        } else {
            // max_bs == 1: reuse cached A/B saved at kernel entry
            reg_aux1_A.restore();
            if (is_row_major) {
                mov(reg_aux_A, reg_aux1_A);
                mov(reg_aux_B, reg_aux1_B);
            } else {
                mov(reg_aux_A, reg_aux1_B);
                mov(reg_aux_B, reg_aux1_A);
            }
        }
    } else if (brg.type == brgemm_offs) {
        // Base pointers
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);

        reg_aux_batch.restore();
        // Apply per-batch offsets
        add(reg_aux_A, ptr[reg_aux_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
        add(reg_aux_B, ptr[reg_aux_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
        // Advance batch descriptor
        add(reg_aux_batch, sizeof(brgemm_batch_element_t));
        reg_aux_batch.save();
    } else if (brg.type == brgemm_strd) {
        // Restore original bases saved in restore_A_B_matrices()
        reg_aux1_A.restore();
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);

        // Add strides (safe_add handles large immediates)
        safe_add(reg_aux1_A, brg.stride_a, reg_tmp_gpr);
        reg_aux1_A.save();
        safe_add(reg_aux1_B, brg.stride_b, reg_tmp_gpr);
        if (vpad_exist) {
            reg_aux_batch.restore();
            add(reg_aux_batch, sizeof(brgemm_batch_element_t));
            reg_aux_batch.save();
        }
    }

    add(reg_aux_A, reg_a_offset);
    add(reg_aux_B, reg_b_offset);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::maybe_pre_process_data(
        data_type_t dt, const Vmm &vmm_out1, const Vmm &vmm_out2) {
    assert(brg.is_fp8_via_convert_non_amx());
    const Zmm zmm_out1 = Zmm(vmm_out1.getIdx());
    const Zmm zmm_out2 = Zmm(vmm_out2.getIdx());

    if (dt == data_type::f8_e5m2)
        f8_e5m2_cvt_->vcvt_f8_to_f16_vnni(zmm_out1, zmm_out2, zmm_out1);
    else if (dt == data_type::f8_e4m3)
        f8_e4m3_cvt_->vcvt_f8_to_f16_vnni(zmm_out1, zmm_out2, zmm_out1);
    else
        assert(!"unsupported data type.");
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::maybe_pre_process_data(matrix_kind_t matrix_kind,
        const Tmm &t1, reg64_t reg_base, dim_t offset, reg64_t reg_stride,
        int num_rows, int num_col_bytes, bool is_rd_tail) {
    const auto transform_offset = brg.brgattr.use_interleave_stores
            ? brg.get_num_C_tiles() * brgemm_desc_t::tilesize
            : 0;
    add(reg_buf_aux, transform_offset);

    switch (matrix_kind) {
        case matrix_A:
            fp8_to_f16_upconvert(num_rows, num_col_bytes, reg_base, offset,
                    reg_stride, brg.dt_a, is_rd_tail);
            break;
        case matrix_B:
            fp8_to_f16_upconvert_to_vnni(num_rows, num_col_bytes, reg_base,
                    offset, reg_stride, brg.dt_b, is_rd_tail);
            break;
        default: assert(!"Wrong Matrix");
    }

    // load into tmm from the transformed data.
    mov(reg_converted_stride, zmm_width_in_bytes_);
    tileloadd(t1, ptr[reg_buf_aux + reg_converted_stride]);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::maybe_tileloadd_nt(matrix_kind_t matrix_kind,
        int idx, dim_t offset, bool is_rd_tail, bool is_tail) {

    const bool is_A = matrix_kind == matrix_kind_t::matrix_A;

    const int tmm_idx = is_A ? brg.get_A_tensor(idx, is_tail)
                             : brg.get_B_tensor(idx, is_tail);
    auto t1 = Tmm(tmm_idx);

    auto reg_base = is_A ? reg_aux_A : reg_aux_B;

    auto reg_stride = is_A ? reg_stride_lda : reg_stride_ldb;
    bool try_load_nt = brg.innermost_loop
            == (is_A ? brgemm_bd_loop_innermost : brgemm_ld_loop_innermost);

    if (brg.is_fp8_via_convert()) {
        const int typesize_A
                = brg.is_input_convert() ? sizeof(int16_t) : brg.typesize_A;
        const int typesize_B
                = brg.is_input_convert() ? sizeof(int16_t) : brg.typesize_B;
        int rd_step = 4 / typesize_A;
        int rd_block = (!brg.rdb && brg.rdb_tail) ? brg.rdb_tail : brg.rd_block;
        if (brg.is_input_convert()) {
            const auto vnni_granularity
                    = data_type_vnni_granularity(data_type::f16);
            rd_block = utils::rnd_up(rd_block, vnni_granularity);
        }

        int A_col = typesize_A * rd_block;
        int A_row = is_tail ? brg.bdb_tail : brg.bd_block;

        int B_col = (is_tail ? brg.ldb_tail : brg.ld_block) * typesize_B
                * rd_step;
        int B_row = brg.typesize_C != 0 ? A_col / brg.typesize_C : 0;
        reg64_savable_guard_t reg_fp8_buf_guard({&reg64_fp8_aux, &reg_buf_aux});

        reg_buf.restoreTo(reg_buf_aux);
        maybe_pre_process_data(matrix_kind, t1, reg_base, offset, reg_stride,
                is_A ? A_row : B_row, is_A ? A_col : B_col, is_rd_tail);

    } else {
        if (maybe_pre_process_k_tail(
                    is_rd_tail, t1, reg_base, offset, reg_stride, matrix_kind))
            return;

        const size_t cache_footprint = static_cast<size_t>(brg.typesize_A)
                        * brg.brgattr.hint_expected_A_size
                + static_cast<size_t>(brg.typesize_B)
                        * brg.brgattr.hint_expected_B_size
                + static_cast<size_t>(brg.typesize_C)
                        * brg.brgattr.hint_expected_C_size;
        if (try_load_nt
                && cache_footprint >= platform::get_per_core_cache_size(1))
            tileloaddt1(t1, ptr[reg_base + offset + reg_stride]);
        else
            tileloadd(t1, ptr[reg_base + offset + reg_stride]);
    }
}

template <typename Wmm>
bool jit_brgemm_kernel_t<Wmm>::maybe_pre_process_k_tail(bool is_rd_tail,
        const Tmm &t1, reg64_t reg_base, dim_t offset, reg64_t reg_stride,
        matrix_kind_t mk) {

    // With the extendable_k strategy matrix B is zero-padded along the K
    // dimension, so brgemm reads the full K-tail tile from matrix A and relies
    // on B's zeros to cancel the columns beyond the valid K range. Reading past
    // the valid K columns of A, however, pulls in bytes that belong to adjacent
    // rows of A. If those bytes contain NaN/Inf the identity 'NaN * 0 == 0' no
    // longer holds and the garbage leaks into otherwise clean output rows. To
    // avoid this the K-tail of A is copied into a zero-padded scratch buffer
    // for every M tile (not only the last one) before it is loaded into the
    // AMX tile.
    const auto need_k_tail_processing = mk == matrix_A && brg.amx_wary_k_tail()
            && brg.rdb_tail != 0 && is_rd_tail;
    if (!need_k_tail_processing) return false;

    const auto zmm_width_in_bytes = cpu_isa_traits_t<avx512_core>::vlen;

    auto transform_offset = brg.get_num_C_tiles() * brgemm_desc_t::tilesize
            + brg.get_convert_wsp_buffer_size();

    //TODO: reuse transformed data from matrix A for ldi > 0
    const int max_tiles = amx::get_max_palette_size();
    JIT_ASSERT_RET(t1.getIdx() >= 0 && t1.getIdx() < max_tiles, false);
    const int num_rows = palette_.rows[t1.getIdx()];
    const int num_col_bytes = palette_.cols[t1.getIdx()];

    const auto max_num_cols
            = nstl::min<dim_t>(num_col_bytes / brg.typesize_A, brg.rdb_tail);
    const size_t col_tail
            = max_num_cols % (zmm_width_in_bytes / brg.typesize_A);
    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(rd_tail_mask, reg_tmp_gpr);
    }
    auto zmm_1 = zmm_tmp_1();
    auto zmm_1_masked = col_tail ? zmm_1 | rd_tail_mask | T_z : zmm_1;

    assert(max_num_cols > 0);

    reg_buf_aux.saveTo(reg_buf_aux_backup);
    reg_buf.restoreTo(reg_buf_aux);

    if (transform_offset) add(reg_buf_aux, transform_offset);

    for (dim_t r = 0; r < num_rows; ++r) {
        const auto row_offset = offset + r * brg.typesize_A * brg.LDA;
        switch (brg.dt_a) {
            case data_type::bf16:
            case data_type::f16:
                vmovdqu16(zmm_1_masked, ptr[reg_base + row_offset]);
                break;
            case data_type::f8_e5m2:
            case data_type::f8_e4m3:
            case data_type::s8:
            case data_type::u8:
                vmovdqu8(zmm_1_masked, ptr[reg_base + row_offset]);
                break;
            default: assert(!"unsupported data type");
        }
        vmovups(ptr[reg_buf_aux + r * zmm_width_in_bytes], zmm_1);
    }
    // load into tmm from the transformed data.
    mov(reg_converted_stride, zmm_width_in_bytes);
    tileloadd(t1, ptr[reg_buf_aux + reg_converted_stride]);
    reg_buf_aux_backup.restoreTo(reg_buf_aux);

    return true;
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::maybe_pre_process_buf_A(
        reg64_t reg_base, const int bd_b, const int bd_e, const int rd_block) {
    if (!(brg.is_fp8_via_convert_non_amx() && brg.fp8_with_f16_vnni_block))
        return;

    reg64_savable_guard_t reg_fp8_buf_guard({&reg64_fp8_aux, &reg_buf_A});

    reg_buf_A.restore();

    for (int bd = bd_b; bd < bd_e; bd++) {
        const auto offset = A_offset(bd, 0);
        auto vmm = vmm_tmp(0);
        auto xmm_tmp = Xmm(vmm.getIdx());
        load_bytes(xmm_tmp, reg_base, offset, rd_block);
        fp8_to_f16_upconvert(brg.dt_a, vmm, xmm_tmp);
        vmovdqu16(ptr[reg_buf_A + bd * zmm_width_in_bytes_], vmm);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemm_microkernel_amx(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    auto tdpbxxd = [this](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        using namespace data_type;
        if (brg.is_fp8 && brg.is_fp8_via_convert()) {
            tdpfp16ps(x1, x2, x3);
        } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e5m2) {
            tdpbf8ps(x1, x2, x3);
        } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e4m3) {
            tdpbhf8ps(x1, x2, x3);
        } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e4m3) {
            tdphf8ps(x1, x2, x3);
        } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e5m2) {
            tdphbf8ps(x1, x2, x3);
        } else if (brg.dt_a == bf16 && brg.dt_b == bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (brg.dt_a == f16 && brg.dt_b == f16) {
            tdpfp16ps(x1, x2, x3);
        } else if (brg.dt_a == u8 && brg.dt_b == u8) {
            tdpbuud(x1, x2, x3);
        } else if (brg.dt_a == u8 && brg.dt_b == s8) {
            tdpbusd(x1, x2, x3);
        } else if (brg.dt_a == s8 && brg.dt_b == u8) {
            tdpbsud(x1, x2, x3);
        } else if (brg.dt_a == s8 && brg.dt_b == s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };
    dim_t rbd_block = (is_rd_tail) ? 1 : brg.rdb;
    for (dim_t rdb = 0; rdb < rbd_block; rdb++) {
        for (int bdb = 0; bdb < bd_block2; bdb++) {
            maybe_tileloadd_nt(matrix_kind_t::matrix_A, bdb,
                    rdb * rdb_A_offset() + A_offset(bdb, 0, true), is_rd_tail,
                    is_bdb_tail);
        }
        for (int ldb = 0; ldb < ld_block2; ldb++) {

            const int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
            maybe_tileloadd_nt(matrix_kind_t::matrix_B, idx,
                    rdb * rdb_B_offset() + B_offset(ldb, 0, true), is_rd_tail,
                    is_ld_tail);
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                tdpbxxd(Tmm(brg.get_C_tensor(
                                bdb, idx, is_bdb_tail, is_ld_tail)),
                        Tmm(brg.get_A_tensor(bdb, is_bdb_tail)),
                        Tmm(brg.get_B_tensor(idx, is_ld_tail)));
            }
        }
    }
    if (!is_rd_tail) {
        add(reg_aux_A, brg.rdb * rdb_A_offset());
        add(reg_aux_B, brg.rdb * rdb_B_offset());
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::ace_load_A_4x16bytes(
        const Zmm &zmm, size_t mask, const Reg64 &reg_A, dim_t offset) {
    constexpr size_t full_mask = 0xFFFF;
    constexpr size_t all_full_mask = 0xFFFFFFFFFFFFFFFF;

    const auto zmm_tmp1 = ace_zmm_tmp(1);
    const auto xmm_tmp = Xmm(zmm_tmp1.getIdx());
    if (mask != all_full_mask) vpxord(zmm, zmm, zmm);

    for (int i = 0; i < 4; ++i) {
        const auto cur_offset
                = offset + i * brg.LDA * ace_zmms_per_bd_block * brg.typesize_A;
        const size_t cur_mask = (mask >> (16 * i)) & full_mask;
        if (cur_mask == 0) continue;

        // Always use masked load to handle M < 16 safely
        mov(reg_tmp_gpr, cur_mask);
        kmovq(ace_load_A_mask, reg_tmp_gpr);
        vmovdqu8(xmm_tmp | ace_load_A_mask | T_z, ptr[reg_A + cur_offset]);
        vinserti64x2(zmm, zmm, xmm_tmp, i);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::ace_load_A(int bdb, dim_t offset,
        bool is_bdb_tail, int bd_block, bool is_rd_tail) {
    const auto rd_block = is_rd_tail ? brg.rdb_tail : brg.rd_block;
    const int mask_stride = 16; // 16 bytes per row

    for (int rb = 0; rb < ace_zmms_per_bd_block; ++rb) {
        auto zmm = ace_zmm_A(bdb, rb);
        if (rb >= bd_block) {
            vpxord(zmm, zmm, zmm);
            continue;
        }
        size_t mask = 0;
        int mask_row_offs = 0;
        for (int row = rb; row < bd_block; row += ace_zmms_per_bd_block) {
            for (int col = 0; col < rd_block; ++col) {
                if (brg.typesize_A == 2) {
                    // bf16, f16
                    mask |= (size_t)0b11 << (mask_row_offs + col * 2);
                } else if (brg.typesize_A == 1) {
                    // f8_e5m2, f8_e4m3, s8, u8
                    mask |= (size_t)0b1 << (mask_row_offs + col);
                } else {
                    assert(!"Unsupported data type for ace_load_A");
                }
            }
            mask_row_offs += mask_stride;
        }
        ace_load_A_4x16bytes(
                zmm, mask, reg_aux_A, offset + rb * brg.LDA * brg.typesize_A);
    }
    // transpose four 4x4 blocks
    const auto base_zmm_idx = ace_zmm_A(bdb, 0).getIdx();
    const auto a_zmm1 = Zmm(base_zmm_idx + 0);
    const auto a_zmm2 = Zmm(base_zmm_idx + 1);
    const auto a_zmm3 = Zmm(base_zmm_idx + 2);
    const auto a_zmm4 = Zmm(base_zmm_idx + 3);
    const auto tmp_zmm1 = ace_zmm_tmp(1);
    const auto tmp_zmm2 = ace_zmm_tmp(2);
    const auto tmp_zmm3 = ace_zmm_tmp(3);
    const auto tmp_zmm4 = ace_zmm_tmp(4);
    vpunpckldq(tmp_zmm1, a_zmm1, a_zmm2);
    vpunpckhdq(tmp_zmm2, a_zmm1, a_zmm2);
    vpunpckldq(tmp_zmm3, a_zmm3, a_zmm4);
    vpunpckhdq(tmp_zmm4, a_zmm3, a_zmm4);
    vpunpcklqdq(a_zmm1, tmp_zmm1, tmp_zmm3);
    vpunpckhqdq(a_zmm2, tmp_zmm1, tmp_zmm3);
    vpunpcklqdq(a_zmm3, tmp_zmm2, tmp_zmm4);
    vpunpckhqdq(a_zmm4, tmp_zmm2, tmp_zmm4);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::ace_load_B(int ldb, dim_t offset, int rdstep,
        bool is_ld_tail, int rd_block, int bd_block2) {
    const auto start_rdstep = (rdstep == -1) ? 0 : rdstep;
    const auto finish_rdstep
            = (rdstep == -1) ? ace_rd_steps(rd_block) : rdstep + 1;

    // Use ukernel formula: rd_step * LDB * typesize
    const auto rds_stride = brg.rd_step * brg.LDB * brg.typesize_B;

    for (int rds = start_rdstep; rds < finish_rdstep; rds++) {
        auto zmm = ace_zmm_B(bd_block2, ldb, rds);
        const auto total_offset = offset + rds * rds_stride;
        const auto zmm_B = is_ld_tail ? (zmm | ld_tail_mask | T_z) : zmm;
        // Use dword-granular vmovups: each ld element occupies one dword of
        // the B register, and ld_tail_mask has one bit per element. A
        // qword-granular load would fetch twice the tail bytes and can read
        // past the end of B.
        vmovups(zmm_B, ptr[reg_aux_B + total_offset]);
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::outer_product(
        const Zmm &zmm_a, const Zmm &zmm_b, const Tmm &accm) {
    using namespace data_type;
    if (brg.dt_a == bf16 && brg.dt_b == bf16) {
        top2bf16ps(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == u8 && brg.dt_b == u8) {
        top4buud(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == u8 && brg.dt_b == s8) {
        top4busd(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == s8 && brg.dt_b == u8) {
        top4bsud(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == s8 && brg.dt_b == s8) {
        top4bssd(accm, zmm_a, zmm_b);
    } else {
        assert(!"Unsupported data type for outer product");
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemm_microkernel_ace(int bd_block2,
        bool is_bdb_tail, int ld_block2, bool is_rd_tail, bool is_ld_tail) {
    dim_t rbd_block = (is_rd_tail) ? 1 : brg.rdb;
    for (dim_t rdb = 0; rdb < rbd_block; rdb++) {
        if (brg.n_bcast_1_load) {
            // Load A once for all bdb
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                int bd_block = (is_bdb_tail && bdb == bd_block2 - 1)
                        ? brg.bdb_tail
                        : brg.bd_block;
                // The tail uses rdb=0, and reg_aux_A has already
                // advanced past the full blocks.
                const auto rd_offset = rdb * rdb_A_offset();
                ace_load_A(bdb, rd_offset + A_offset(bdb, 0, true),
                        is_bdb_tail && bdb == bd_block2 - 1, bd_block,
                        is_rd_tail);
            }

            // Load B one rds at a time, compute with all bdb
            for (int ldb = 0; ldb < ld_block2; ldb++) {
                // The offset is relative to reg_aux_B, which has already
                // been advanced for the tail.
                const auto rd_pos = rdb;
                const auto ldb_offs = ldb * brg.ld_block;
                auto b_offset_base
                        = rd_pos * brg.rd_block * brg.LDB * brg.typesize_B
                        + brg.typesize_B
                                * ((ldb_offs / brg.LDB) * brg.brgattr.LDB2
                                        + (ldb_offs % brg.LDB) * brg.rd_step);

                const auto rd_block_cur
                        = is_rd_tail ? brg.rdb_tail : brg.rd_block;
                for (int rds = 0; rds < ace_rd_steps(rd_block_cur); rds++) {
                    // Load one rds line from B into ace_zmm_B(bd_block2, ldb,
                    // rds); ace_load_B adds rds * rds_stride internally.
                    ace_load_B(ldb, b_offset_base, rds, is_ld_tail,
                            rd_block_cur, bd_block2);

                    // Compute with all bdb
                    for (int bdb = 0; bdb < bd_block2; bdb++) {
                        const int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
                        const auto &accm = Tmm(brg.get_C_tensor(
                                bdb, idx, is_bdb_tail, is_ld_tail));
                        outer_product(ace_zmm_A(bdb, rds),
                                ace_zmm_B(bd_block2, ldb, rds), accm);
                    }
                }
            }
        } else {
            // For the non-n_bcast path, load all the B matrices first and
            // then compute with A.
            for (int ldb = 0; ldb < ld_block2; ldb++) {
                // The offset is relative to reg_aux_B, which has already
                // been advanced for the tail.
                const auto rd_pos = rdb;
                const auto ldb_offs = ldb * brg.ld_block;
                const auto b_offset
                        = rd_pos * brg.rd_block * brg.LDB * brg.typesize_B
                        + brg.typesize_B
                                * ((ldb_offs / brg.LDB) * brg.brgattr.LDB2
                                        + (ldb_offs % brg.LDB) * brg.rd_step);
                const auto rd_block_cur
                        = is_rd_tail ? brg.rdb_tail : brg.rd_block;
                ace_load_B(
                        ldb, b_offset, -1, is_ld_tail, rd_block_cur, bd_block2);
            }

            // Compute with loaded B
            for (int bdb = 0; bdb < bd_block2; bdb++) {
                int bd_block = (is_bdb_tail && bdb == bd_block2 - 1)
                        ? brg.bdb_tail
                        : brg.bd_block;
                // The tail uses rdb=0, and reg_aux_A has already
                // advanced past the full blocks.
                const auto rd_offset = rdb * rdb_A_offset();
                ace_load_A(bdb, rd_offset + A_offset(bdb, 0, true),
                        is_bdb_tail && bdb == bd_block2 - 1, bd_block,
                        is_rd_tail);
                for (int ldb = 0; ldb < ld_block2; ldb++) {
                    const int idx = (is_ld_tail) ? brg.ld_block2 : ldb;
                    const auto &accm = Tmm(brg.get_C_tensor(
                            bdb, idx, is_bdb_tail, is_ld_tail));
                    const auto rd_block_cur
                            = is_rd_tail ? brg.rdb_tail : brg.rd_block;
                    for (int rds = 0; rds < ace_rd_steps(rd_block_cur); rds++) {
                        outer_product(ace_zmm_A(bdb, rds),
                                ace_zmm_B(bd_block2, ldb, rds), accm);
                    }
                }
            }
        }
    }
    if (!is_rd_tail) {
        add(reg_aux_A, brg.rdb * rdb_A_offset());
        // For ACE with BA16a64b2a: The rdb loop runs brg.rdb times,
        // each iteration processes rd_block elements.
        // So advance by brg.rdb * rd_block * LDB * typesize
        if (brg.is_ace()) {
            const auto rdb_B_stride
                    = brg.rdb * brg.rd_block * brg.LDB * brg.typesize_B;
            add(reg_aux_B, rdb_B_stride);
        } else {
            add(reg_aux_B, brg.rdb * rdb_B_offset());
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::dot_product(Vmm v1, Vmm v2, Vmm v3) {
    if (brg.is_f16 && brg.isa_impl == avx10_2)
        vdpphps(v1, v2, v3);
    else if (brg.is_fp8 && brg.is_fp8_via_convert_non_amx())
        vdpphps(v1, v2, v3);
    else if (brg.is_f32 || brg.is_f16
            || (brg.is_bf16 && brg.isa_impl == avx2_vnni_2))
        uni_vfmadd231ps(v1, v2, v3);
    else if (brg.is_bf16)
        vdpbf16ps(v1, v2, v3);
    else if (brg.is_int8) {
        if (brg.dt_a == data_type::s8 && isa_has_s8s8(brg.isa_impl))
            vpdpbssd(v1, v3, v2);
        else if (brg.has_int8_vnni)
            vpdpbusd(v1, v3, v2, get_encoding());
        else {
            vpmaddubsw(int8_dot_product_temp(), v3, v2);
            vpmaddwd(int8_dot_product_temp(), int8_dot_product_temp(),
                    int8_ones_words());
            vpaddd(v1, v1, int8_dot_product_temp());
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::compute_int8_compensation(int rd_loop, int bd_b,
        int bd_e, int bd_block, int ld_block2, bool is_ld_tail, int vpad) {
    assert(brg.is_int8);

    auto compensation_padding = [this, ld_block2](Vmm vmm_load, Vmm vmm_tmp,
                                        int ld, int bd_b, int bd_e) {
        // req_cal_comp_pads -> only calculate compensation along with
        // computation and do not use pre-calculated compensation.
        // Calculate comp padding as:
        // accum - inp_shift * conv(1, wei_s32)
        if (brg.req_s8s8_compensation) {
            if (brg.req_cal_comp_pads) {
                uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
                dot_product(vmm_tmp, vmm_load, vmm_inp_shift());
            }

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    uni_vpsubd(vmm, vmm, vmm_tmp);
                } else {
                    dot_product(vmm, vmm_load, vmm_inp_shift());
                }
            }
        }

        if (brg.zp_type_a != brgemm_broadcast_t::none) {
            uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
            dot_product(vmm_tmp, vmm_load, vmm_one_bytes());
            uni_vpmulld(vmm_tmp, vmm_tmp, vmm_zp_a_shift());

            for (int bd = bd_b; bd < bd_e; bd++) {
                auto vmm = accm(ld_block2, bd, ld);
                if (brg.req_cal_comp_pads) {
                    uni_vpsubd(vmm, vmm, vmm_tmp);
                } else {
                    uni_vpaddd(vmm, vmm, vmm_tmp);
                }
            }
        }
    };

    if (need_comp_pads && brg.zp_type_a != brgemm_broadcast_t::none) {
        reg64_savable_guard_t reg_bdb_loop_guard({&reg_bdb_loop});
        const auto reg32_scratch = reg_zp_a_input_shift.cvt32();
        mov(reg32_scratch, 0x1010101);
        uni_vpbroadcastd(vmm_one_bytes(), reg32_scratch);
        reg_zp_a_val.restoreTo(reg32_scratch);
        uni_vpbroadcastd(vmm_zp_a_shift(), reg32_scratch);
    }

    for_(int rd = 0; rd < rd_loop; rd += brg.rd_step)
    for (int ld = 0; ld < ld_block2; ++ld) {
        const auto addr = ptr[reg_aux_B + B_offset(ld, rd)];
        const bool is_tail = is_ld_tail && ld + 1 == ld_block2;
        if (IMPLICATION(is_tail, is_superset(brg.isa_impl, avx512_core))) {
            auto vmm_store = vmm_mask(load(), is_tail, false, ld_tail_mask);
            uni_vmovups(vmm_store, addr);
        } else {
            load_bytes(load(), addr, static_cast<int>(ldb_B_offset(0, true)));
        }

        if (brg.req_cal_comp_pads) {
            compensation_padding(load(), bcst(), ld, bd_b, bd_e);
        } else if (vpad != 0) {
            if (bd_b > 0) compensation_padding(load(), bcst(), ld, 0, bd_b);
            if (bd_e < bd_block)
                compensation_padding(load(), bcst(), ld, bd_e, bd_block);
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemv_microkernel(
        bool is_bdb_tail, int ld_block2, bool is_rd_tail) {

    assert(brg.is_gemv);
    assert(brg.ld_block == 1);
    assert(ld_block2 == 1);

    const int bd_block = brg.gemv_num_acc_blocks(is_bdb_tail);

    const auto load_and_convert_to_f32
            = [this](const Vmm vmm, const Xbyak::Address addr,
                      const data_type_t dt, bool is_tail) {
        if (is_superset(brg.isa_impl, avx512_core)) {
            assert(one_of(brg.dt_a, data_type::bf16, data_type::f16));
            assert(one_of(brg.dt_b, data_type::bf16, data_type::f16));

            const auto tail_mask
                    = brg.transA ? gemv_partial_mask : rd_tail_mask;
            const auto kmask = is_tail ? tail_mask : gemv_full_mask;

            if (dt == data_type::bf16) {
                uni_vpmovzxwd(vmm | kmask | T_z, addr);
                uni_vpslld(vmm, vmm, 16);
            } else if (dt == data_type::f16) {
                vmovdqu16(Vmm_lower_t(vmm.getIdx()) | kmask | T_z, addr);
                vcvtph2ps(vmm, Vmm_lower_t(vmm.getIdx()));
            } else
                assert(!"unsupported data type");
        } else {
            assert(brg.isa_impl == avx2);
            if (is_tail)
                vmaskmovps(vmm, vmm_tail_mask(), addr);
            else
                uni_vmovups(vmm, addr);
        }
    };

    const auto bcast_and_convert_to_f32
            = [this](const Vmm vmm, const Xbyak::Address addr,
                      const data_type_t dt) {
        if (dt == data_type::f32) {
            vbroadcastss(vmm, addr);
        } else if (dt == data_type::bf16) {
            vpbroadcastw(vmm, addr);
            uni_vpslld(vmm, vmm, 16);
        } else if (dt == data_type::f16) {
            vpbroadcastw(Xmm(vmm.getIdx()), addr);
            vcvtph2ps(Xmm(vmm.getIdx()), Xmm(vmm.getIdx()));
            vbroadcastss(vmm, Xmm(vmm.getIdx()));
        } else {
            assert(!"unsupported GEMV input data type");
        }
    };

    if (!brg.transA) {
        // GEMV is bandwidth-bound and the HW prefetcher doesn't keep enough
        // loads in flight for these streams so SW-prefetch ahead of the load:
        // the B vector once and each of the A rows. The distance is empirically
        // tuned.
        constexpr dim_t gemv_pf_dist = 512; // bytes = 8 cache lines

        maybe_set_avx_mask(is_rd_tail);

        const auto a_vmm = gemv_load_a();
        const auto b_vmm = gemv_load_b();

        const auto b_addr = ptr[reg_aux_B];

        if (!is_rd_tail) prefetcht0(ptr[reg_aux_B + gemv_pf_dist]);

        if (brg.gemv_use_vdpbf16ps()) {
            if (is_rd_tail)
                vmovdqu16(b_vmm | rd_tail_mask | T_z, b_addr);
            else
                uni_vmovups(b_vmm, b_addr);
        } else {
            load_and_convert_to_f32(b_vmm, b_addr, brg.dt_b, is_rd_tail);
        }

        for (int bd = 0; bd < bd_block; bd++) {
            const auto acc = accm(1, bd, 0);
            const auto a_addr = ptr[reg_aux_A + A_offset(bd, 0)];

            if (!is_rd_tail)
                prefetcht0(ptr[reg_aux_A + A_offset(bd, 0) + gemv_pf_dist]);

            if (brg.gemv_use_vdpbf16ps()) {
                if (is_rd_tail) {
                    vmovdqu16(a_vmm | rd_tail_mask | T_z, a_addr);
                    vdpbf16ps(acc, b_vmm, a_vmm);
                } else {
                    vdpbf16ps(acc, b_vmm, a_addr);
                }
            } else {
                load_and_convert_to_f32(a_vmm, a_addr, brg.dt_a, is_rd_tail);
                uni_vfmadd231ps(acc, a_vmm, b_vmm);
            }
        }
    } else {
        // Same idea as the non-transA path but here each A row is strided by
        // LDA per reduce step so the prefetch distance is in reduce steps not
        // bytes. Only enabled for the avx512 path (bf16/f16): it moves half the
        // bytes of f32 so it's latency-bound (stalls on strided misses) and
        // benefits from prefetch. f32/avx2 is normally BW-bound so the HW
        // prefetcher is usually enough, but for large-stride, long-running
        // loops (row_bytes >= 2048, rdb >= 32) the stride is wide enough that
        // SW prefetch still helps; gate it on both so short/narrow loops keep
        // relying on the HW prefetcher.
        constexpr dim_t gemv_pf_rd_steps_ahead = 8;
        const dim_t row_bytes = rdb_A_offset();
        const dim_t pf_offset = gemv_pf_rd_steps_ahead * row_bytes;

        const bool do_f32_pf = brg.is_f32 && brg.isa_impl == avx2
                && brg.rdb >= 32 && row_bytes >= 2048;
        const bool do_pf = is_superset(brg.isa_impl, avx512_core) || do_f32_pf;
        bcast_and_convert_to_f32(gemv_load_b(), ptr[reg_aux_B], brg.dt_b);
        for (int bd = 0; bd < bd_block; bd++) {
            const bool is_tail_acc
                    = gemv_is_tail_acc(bd, bd_block, is_bdb_tail);
            const auto a_addr = ptr[reg_aux_A + A_offset(bd, 0)];
            if (do_pf) {
                prefetcht0(ptr[reg_aux_A + A_offset(bd, 0) + pf_offset]);
            }
            maybe_set_avx_mask(is_tail_acc);
            load_and_convert_to_f32(
                    gemv_load_a(), a_addr, brg.dt_a, is_tail_acc);
            uni_vfmadd231ps(accm(1, bd, 0), gemv_load_a(), gemv_load_b());
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::gemm_microkernel(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_rd_tail, bool is_ld_tail, int vpad,
        int rows_for_rd_tail) {

    MAYBE_UNUSED(bd_block2);
    int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
    const auto bd_b = nstl::max(0, vpad);
    const auto bd_e = nstl::min(bd_block, bd_block + vpad);
    const auto is_valid_bd
            = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
    if (!is_valid_bd) return;

    bool is_emdbd = brg.embd_bcst;

    int rd_loop = 0, rd_tail_size = 0;
    if (is_rd_tail) {
        if (brg.is_bf16 || brg.is_f16 || brg.is_int8 || brg.is_fp8) {
            rd_tail_size = brg.rdb_tail % brg.rd_step;
            rd_loop = (rd_tail_size != 0)
                    ? ((brg.rdb_tail / brg.rd_step) + 1) * brg.rd_step
                    : brg.rdb_tail;
        } else
            rd_loop = brg.rdb_tail;
    } else
        rd_loop = brg.rd_block;

    if (brg.req_src_s8_shift) {
        reg_bdb_loop.save();
        mov(reg_s8_input_shift, 128);
        uni_vpbroadcastb(vmm_inp_shift(), reg_s8_input_shift.cvt8());
        reg_bdb_loop.restore();
    }

    auto broadcast_A
            = [this, rd_tail_size, is_rd_tail, rd_loop, rows_for_rd_tail, bd_e](
                      Vmm vmm_bcast, int bd, int rd) {
        const auto offset = A_offset(bd, rd);
        const auto dt = brg.dt_a;
        const bool maybe_load_bytes
                = (rows_for_rd_tail > 0 || brg.brgattr.wary_A_k_tail_read)
                && is_rd_tail && rd_tail_size != 0
                && (brg.is_bf16 || brg.is_f16 || brg.is_int8 || brg.is_fp8);
        const bool have_to_load_bytes
                = maybe_load_bytes && (rd == rd_loop - brg.rd_step);
        const auto rows_by_load_bytes
                = have_to_load_bytes ? rows_for_rd_tail : 0;
        const auto bd_by_load_bytes = (bd >= bd_e - rows_by_load_bytes
                || brg.brgattr.wary_A_k_tail_read);
        const auto is_pre_process_fp8
                = one_of(dt, data_type::f8_e5m2, data_type::f8_e4m3)
                && brg.fp8_with_f16_vnni_block;
        const auto is_tail
                = have_to_load_bytes && bd_by_load_bytes && !is_pre_process_fp8;
        if (is_tail) {
            Xmm xmm_tmp = Xmm(vmm_bcast.getIdx());
            load_bytes(
                    xmm_tmp, reg_aux_A, offset, rd_tail_size * brg.typesize_A);
            uni_vpbroadcastd(vmm_bcast, xmm_tmp);
        } else {
            if (dt == data_type::f32) {
                uni_vbroadcastss(vmm_bcast, ptr[reg_aux_A + offset]);
            } else if (dt == data_type::bf16) {
                if (brg.isa_impl == avx2_vnni_2)
                    vbcstnebf162ps(vmm_bcast, ptr[reg_aux_A + offset]);
                else
                    uni_vpbroadcastd(vmm_bcast, ptr[reg_aux_A + offset]);
            } else if (one_of(dt, data_type::s8, data_type::u8)) {
                uni_vpbroadcastd(vmm_bcast, ptr[reg_aux_A + offset]);
            } else if (one_of(dt, data_type::f8_e5m2, data_type::f8_e4m3)) {
                if (brg.fp8_with_f16_vnni_block) {
                    reg_buf_A.restore();
                    const auto buf_offset = buf_A_offset(bd, rd);
                    uni_vpbroadcastd(vmm_bcast, ptr[reg_buf_A + buf_offset]);
                } else
                    uni_vpbroadcastd(vmm_bcast, ptr[reg_aux_A + offset]);
            } else if (dt == data_type::f16) {
                if (brg.isa_impl == avx10_2) {
                    uni_vpbroadcastd(vmm_bcast, ptr[reg_aux_A + offset]);
                } else if (brg.isa_impl == avx2_vnni_2) {
                    vbcstnesh2ps(vmm_bcast, ptr[reg_aux_A + offset]);
                } else if (is_superset(brg.isa_impl, avx512_core_fp16)) {
                    // Broadcast is not supported for legacy f16-conversions.
                    vcvtph2psx(vmm_bcast, ptr_b[reg_aux_A + offset]);
                }
            }
        }

        if (brg.req_src_s8_shift)
            uni_vpaddb(vmm_bcast, vmm_bcast, vmm_inp_shift());
    };

    auto load_B = [this, is_ld_tail](int vmm_load_idx, int rd, int ld) {
        const bool mem_advice_B
                = utils::one_of(brg.brgattr.mem_advice,
                          brgemm_hint_mem_advice_B, brgemm_hint_mem_advice_A_B)
                && IMPLICATION(
                        brg.dt_b == data_type::f16, brg.isa_impl == avx10_2)
                && IMPLICATION(brg.dt_b == data_type::bf16,
                        brg.isa_impl != avx2_vnni_2);
        const auto tail_mask
                = brg.fp8_with_f16_vnni_block ? fp8_tail_mask : ld_tail_mask;
        const Vmm vmm_load
                = vmm_mask(load(vmm_load_idx), is_ld_tail, false, tail_mask);
        const auto addr = ptr[reg_aux_B + B_offset(ld, rd)];
        // Note: Assuming the tails are properly padded/blocked for
        // avx2_vnni_2 with xf16 data type, as the B matrix is generally
        // at least double-blocked.
        if (mem_advice_B) {
            vmovrsd(vmm_load, addr);
        } else if (brg.dt_b == data_type::f16) {
            if (brg.isa_impl == avx10_2) {
                uni_vmovups(vmm_load, addr);
            } else if (brg.isa_impl == avx2_vnni_2) {
                if (rd % 2 == 0)
                    vcvtneeph2ps(vmm_load, addr);
                else
                    vcvtneoph2ps(vmm_load, addr);
            } else if (brg.is_f16_b_non_amx_vnni()) {
                const auto actual_B_offset = B_offset(ld, utils::rnd_dn(rd, 2));
                const auto vnni_addr = ptr[reg_aux_B + actual_B_offset];
                vmovups(vmm_load, vnni_addr);
                if (rd % 2 == 0)
                    vpermw(vmm_load, f16_perm_even_vreg(), vmm_load);
                else
                    vpermw(vmm_load, f16_perm_odd_vreg(), vmm_load);
                vcvtph2psx(vmm_load, Vmm_lower_t(vmm_load.getIdx()));
            } else if (is_ld_tail && !is_superset(brg.isa_impl, avx512_core)) {
                load_bytes(vmm_load, addr,
                        static_cast<int>(ldb_B_offset(0, true)));
                vcvtph2ps(vmm_load, Xmm(vmm_load.getIdx()));
            } else {
                uni_vcvtph2psx(vmm_load, addr);
            }
        } else if (brg.dt_b == data_type::bf16) {
            if (brg.isa_impl == avx2_vnni_2) {
                if (rd % 2 == 0)
                    vcvtneebf162ps(vmm_load, addr);
                else
                    vcvtneobf162ps(vmm_load, addr);
            } else if (utils::one_of(brg.isa_impl, avx512_core, avx2)) {
                // Upconvert: load 16 bits and move them 16 bits left.
                uni_vpmovzxwd(vmm_load, addr);
                uni_vpslld(vmm_load, vmm_load, 16);
            } else if (is_ld_tail && !is_superset(brg.isa_impl, avx512_core)) {
                load_bytes(vmm_load, addr,
                        static_cast<int>(ldb_B_offset(0, true)));
            } else {
                uni_vmovups(vmm_load, addr);
            }
        } else if (one_of(brg.dt_b, data_type::f8_e5m2, data_type::f8_e4m3)) {
            if (brg.fp8_with_f16_vnni_block)
                fp8_to_f16_upconvert(brg.dt_b, vmm_load, addr);
            else
                uni_vmovups(vmm_load, addr);
        } else if (is_ld_tail) {
            if (is_superset(brg.isa_impl, avx512_core)) {
                uni_vmovups(vmm_load, addr);
            } else {
                load_bytes(vmm_load, addr,
                        static_cast<int>(ldb_B_offset(0, true)));
            }
        } else {
            uni_vmovups(vmm_load, addr);
        }
    };

    const bool comp_vpad = vpad != 0
            && (brg.req_s8s8_compensation
                    || brg.zp_type_a != brgemm_broadcast_t::none);
    if (brg.req_cal_comp_pads || comp_vpad)
        compute_int8_compensation(
                rd_loop, bd_b, bd_e, bd_block, ld_block2, is_ld_tail, vpad);

    // Sometimes the offset used for prefetching is too big and needs to be
    // handled with an additional temporary register.
    // `reg_aux_C` and `reg_tmp_microkernel` are aliases for `r14` so we need to
    // save its content.
    const dim_t max_prefetch_offset = B_offset(ld_block2 - 1, rd_loop - 1)
            + static_cast<dim_t>(brg.LDB) * brg.rd_block_B_size();
    if (max_prefetch_offset > INT_MAX) reg_aux_C.save();

    if (brg.is_fp8_via_convert()) reg64_fp8_aux.save();

    maybe_pre_process_buf_A(
            reg_aux_A, bd_b, bd_e, is_rd_tail ? brg.rdb_tail : rd_loop);

    for (int rd = 0; rd < rd_loop; rd += brg.rd_step) {
        if (brg.n_bcast_1_load) {
            for (int bd = bd_b; bd < bd_e && !is_emdbd; bd++)
                broadcast_A(bcst(bd), bd, rd);
            for (int ld = 0; ld < ld_block2; ld++) {
                load_B(0, rd, ld);
                if (brg.is_fp8_via_convert_non_amx()
                        && !brg.fp8_with_f16_vnni_block)
                    maybe_pre_process_data(brg.dt_b, load(), vmm_fp8_load());
                for (int bd = bd_b; bd < bd_e; bd++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        uni_vfmadd231ps(vmm, load(),
                                ptr_b[reg_aux_A + A_offset(bd, rd)]);
                    else {
                        if (brg.is_fp8_via_convert_non_amx()
                                && !brg.fp8_with_f16_vnni_block) {
                            broadcast_A(bcst(bd), bd, rd);
                            maybe_pre_process_data(
                                    brg.dt_a, bcst(bd), vmm_fp8_bcst());
                            dot_product(vmm, load(), bcst(bd));
                            dot_product(vmm, vmm_fp8_load(), vmm_fp8_bcst());
                        } else
                            dot_product(vmm, load(), bcst(bd));
                    }
                }
            }

        } else {
            int prefetch_count_B = 0;
            for (int ld = 0; ld < ld_block2; ld++) {
                load_B(ld, rd, ld);
            }

            for (int bd = bd_b; bd < bd_e; bd++) {
                if (!is_emdbd) broadcast_A(bcst(), bd, rd);
                if (brg.is_fp8_via_convert_non_amx()
                        && !brg.fp8_with_f16_vnni_block)
                    maybe_pre_process_data(brg.dt_a, bcst(), vmm_fp8_bcst());
                if (prefetch_count_B < ld_block2) {
                    const dim_t prefetch_offset
                            = B_offset(prefetch_count_B++, rd)
                            + static_cast<dim_t>(brg.LDB)
                                    * brg.rd_block_B_size();
                    // Only use EVEX_compress_addr_safe/make_safe_addr
                    // when prefetch_offset > INT_MAX forr perf purpose
                    if (prefetch_offset <= INT_MAX) {
                        prefetcht0(ptr[reg_aux_B + prefetch_offset]);
                    } else {
                        if (is_superset(brg.isa_impl, avx512_core)) {
                            prefetcht0(EVEX_compress_addr_safe(reg_aux_B,
                                    prefetch_offset, reg_tmp_microkernel));
                        } else {
                            prefetcht0(make_safe_addr(reg_aux_B,
                                    prefetch_offset, reg_tmp_microkernel));
                        }
                    }
                }
                for (int ld = 0; ld < ld_block2; ld++) {
                    auto vmm = accm(ld_block2, bd, ld);
                    if (is_emdbd)
                        uni_vfmadd231ps(vmm, load(ld),
                                ptr_b[reg_aux_A + A_offset(bd, rd)]);
                    else {
                        if (brg.is_fp8_via_convert_non_amx()
                                && !brg.fp8_with_f16_vnni_block) {
                            load_B(ld, rd, ld);
                            maybe_pre_process_data(
                                    brg.dt_b, load(ld), vmm_fp8_load());
                            dot_product(vmm, load(ld), bcst());
                            dot_product(vmm, vmm_fp8_load(), vmm_fp8_bcst());
                        } else
                            dot_product(vmm, load(ld), bcst());
                    }
                }
            }
        }
    }

    if (brg.is_fp8_via_convert()) reg64_fp8_aux.restore();

    if (max_prefetch_offset > INT_MAX) reg_aux_C.restore();
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::bs_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, bool is_ld_tail, bool first_bdb, bool last_bdb,
        int rows_for_rd_tail, bool skip_accumulation) {

    auto bs_loop_body = [&](int vpad) {
        set_A_B_matrices();

        int bd_block = (is_bdb_tail) ? brg.bdb_tail : brg.bd_block;
        const auto bd_b = nstl::max(0, vpad);
        const auto bd_e = nstl::min(bd_block, bd_block + vpad);
        const auto is_valid_bd
                = need_comp_pads && vpad != 0 ? bd_b <= bd_e : bd_b < bd_e;
        if (!is_valid_bd) return;

        if (brg.is_tmm) {
            const bool is_rd_tail = false;
            if (brg.is_ace())
                gemm_microkernel_ace(bd_block2, is_bdb_tail, ld_block2,
                        is_rd_tail, is_ld_tail);
            else
                gemm_microkernel_amx(bd_block2, is_bdb_tail, ld_block2,
                        is_rd_tail, is_ld_tail);
        } else {
            if (brg.rdb > 0) {
                Label rdb_loop_label;
                mov(reg_rdb_loop, brg.rdb);
                L_aligned(rdb_loop_label, 64);
                {
                    const bool is_rd_tail = false;
                    if (brg.is_gemv)
                        gemv_microkernel(is_bdb_tail, ld_block2, is_rd_tail);
                    else
                        gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                                is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);

                    add(reg_aux_A, rdb_A_offset());
                    add(reg_aux_B, rdb_B_offset());

                    dec(reg_rdb_loop);
                    cmp(reg_rdb_loop, 0);
                    jg(rdb_loop_label, T_NEAR);
                }
            }
        }
        if (brg.rdb_tail != 0) {
            const bool is_rd_tail = true;
            if (brg.is_tmm) {
                if (brg.is_ace())
                    gemm_microkernel_ace(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail);
                else
                    gemm_microkernel_amx(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail);
            } else {
                if (brg.is_gemv)
                    gemv_microkernel(is_bdb_tail, ld_block2, is_rd_tail);
                else
                    gemm_microkernel(bd_block2, is_bdb_tail, ld_block2,
                            is_rd_tail, is_ld_tail, vpad, rows_for_rd_tail);
            }
        }
    };

    Label BS_loop_label;

    reg64_savable_guard_t reg_aux_D_guard({&reg_aux_D}, brg.brgattr.max_bs > 1);

    if (brg.alpha != 0.f && !skip_accumulation) {
        restore_A_B_matrices();
        if (brg.is_tmm) {
            mov(reg_stride_lda, brg.typesize_A * brg.LDA);
            mov(reg_stride_ldb, brg.rd_step * brg.typesize_B * brg.LDB);
        }

        if (brg.brgattr.max_bs > 1) mov(reg_BS_loop, reg_BS);
        L_aligned(BS_loop_label, 64);
        {
            if (first_bdb || last_bdb) {
                const auto vpad_first
                        = last_bdb ? (-brg.brgattr.max_bottom_vpad) : 1;
                const auto vpad_last
                        = first_bdb ? brg.brgattr.max_top_vpad : -1;
                const auto n_vpads = vpad_last - vpad_first + 2;
                constexpr auto MAX_N_VPADS = 2 * brgemm_desc_t::MAX_VPAD;
                assert(n_vpads < MAX_N_VPADS);

                Label Vpad_loop_end_label;
                std::vector<Label> Vpad_loop_iter_label(MAX_N_VPADS);
                if (vpad_exist) {
                    if (brg.type == brgemm_addr || brg.type == brgemm_strd)
                        reg_aux_batch.restore();

                    mov(reg_aux_A_vpad,
                            ptr[reg_aux_batch
                                    + GET_OFF_BATCH_ELEMENT(vvpad.top)]);
                    sub(reg_aux_A_vpad,
                            ptr[reg_aux_batch
                                    + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                } else
                    xor_(reg_aux_A_vpad, reg_aux_A_vpad);

                for (int vpad = vpad_first; vpad <= vpad_last; vpad++) {
                    const auto label_vpad = vpad - vpad_first;
                    L(Vpad_loop_iter_label[label_vpad]);
                    if (!first_bdb && vpad > 0) continue;
                    if (!last_bdb && vpad < 0) continue;
                    auto real_vpad = vpad;
                    if (last_bdb && brg.bdb_tail && vpad < 0) {
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
                    cmp(reg_aux_A_vpad, vpad);
                    jne(Vpad_loop_iter_label[label_vpad + 1], T_NEAR);
                    bs_loop_body(real_vpad);
                    jmp(Vpad_loop_end_label, T_NEAR);
                }
                L(Vpad_loop_iter_label[n_vpads - 1]);
                bs_loop_body(0);
                L(Vpad_loop_end_label);
            } else {
                bs_loop_body(0);
            }
            if (brg.brgattr.max_bs > 1) {
                dec(reg_BS_loop);
                cmp(reg_BS_loop, 0);
                jg(BS_loop_label, T_NEAR);
            }
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::ldb_loop(int bd_block2, bool is_bdb_tail,
        int ld_block2, dim_t ldb_loop_length, bool is_reg_tail, bool is_ld_tail,
        bool first_bdb, bool last_bdb, int rows_for_rd_tail,
        bool skip_accumulation) {

    Label ldb_loop_label;

    copy_post_ops_stack_values_to_aux(is_reg_tail);

    if (is_ldb_loop_) {
        mov(reg_ldb_loop, ldb_loop_length);
        if (brg.is_tmm) reg_ldb_loop.save();
    }

    L_aligned(ldb_loop_label, 64);
    {
        zero_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);

        if (is_ldb_loop_)
            reg_D.save();
        else {
            mov(reg_ldb_loop, reg_D);
            if (brg.is_tmm) reg_ldb_loop.save();
        }

        bs_loop(bd_block2, is_bdb_tail, ld_block2, is_ld_tail, first_bdb,
                last_bdb, rows_for_rd_tail, skip_accumulation);

        if (is_ldb_loop_)
            reg_D.restore();
        else {
            if (brg.is_tmm) reg_ldb_loop.restore();
            mov(reg_D, reg_ldb_loop);
        }

        store_accumulators(bd_block2, is_bdb_tail, ld_block2, is_ld_tail,
                skip_accumulation);
        if (is_ldb_loop_) {
            if (brg.is_tmm) reg_ldb_loop.restore();
            if (!is_ld_tail)
                ldb_regs_shift(ld_block2);
            else
                ldb_regs_shift(1, true);
            dec(reg_ldb_loop);
            if (brg.is_tmm) reg_ldb_loop.save();
            cmp(reg_ldb_loop, 0);
            jg(ldb_loop_label, T_NEAR);
        }
    }
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::bdb_loop() {
    auto do_ldb_loop = [this](int bd_block2, bool is_bdb_tail, bool first_bdb,
                               bool last_bdb, int rows_for_rd_tail,
                               bool skip_accumulation) {
        // For the ACE rd tail every bdb iteration starts fresh with
        // reg_b_offset = 0. No restore is needed here because
        // copy_post_ops_stack_values_to_aux already zeroes it.

        if (brg.ldb2 > 0) {
            const bool is_ld_reg_tail = false;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ld_block2, brg.ldb2,
                    is_ld_reg_tail, is_ld_tail, first_bdb, last_bdb,
                    rows_for_rd_tail, skip_accumulation);
        }
        if (brg.ldb2_tail > 0) {
            const bool is_ld_reg_tail = (brg.ldb2 == 0) ? false : true;
            const bool is_ld_tail = false;
            ldb_loop(bd_block2, is_bdb_tail, brg.ldb2_tail, 1, is_ld_reg_tail,
                    is_ld_tail, first_bdb, last_bdb, rows_for_rd_tail,
                    skip_accumulation);
        }
        if (brg.ldb_tail > 0) {
            const bool is_ld_reg_tail
                    = (brg.ldb2 == 0 && brg.ldb2_tail == 0) ? false : true;
            const bool is_ld_tail = true;
            ldb_loop(bd_block2, is_bdb_tail, 1, 1, is_ld_reg_tail, is_ld_tail,
                    first_bdb, last_bdb, rows_for_rd_tail, skip_accumulation);
        }
    };

    auto bdb_loop_body = [this, do_ldb_loop](int bd_block2, bool is_bdb_tail,
                                 bool first_bdb, bool last_bdb,
                                 int rows_for_rd_tail, bool skip_accumulation) {
        do_ldb_loop(bd_block2, is_bdb_tail, first_bdb, last_bdb,
                rows_for_rd_tail, skip_accumulation);

        if (brg.is_runtime_ldc) {
            reg_C.saveTo(reg_C_backup);
            xor_(reg_C, reg_C);
            reg_stride_ld_block.imulTo(
                    reg_C, bdb_C_offset(bd_block2, is_bdb_tail));
            reg_C_backup.addTo(reg_C);
        } else {
            add(reg_C, bdb_C_offset(bd_block2, is_bdb_tail));
        }
        if (brg.is_runtime_ldd) {
            reg_D.saveTo(reg_aux_D_backup);
            xor_(reg_D, reg_D);
            reg_D_shift_bytes.imulTo(
                    reg_D, bdb_D_offset(bd_block2, is_bdb_tail));
            reg_aux_D_backup.addTo(reg_D);
        } else {
            add(reg_D, bdb_D_offset(bd_block2, is_bdb_tail));
        }
        add(reg_a_offset, bdb_A_offset(bd_block2, is_bdb_tail));

        if (brg.is_per_k_src_scales) {
            const auto adj_bd_block = is_bdb_tail ? brg.bdb_tail : brg.bd_block;
            const auto src_scales_offset
                    = adj_bd_block * bd_block2 * brg.src_scale_m_stride;
            if (reg_src_scales.savable())
                add(qword[reg_src_scales.getStoragePtr().getRegExp()],
                        src_scales_offset);
            else
                add(reg_src_scales, src_scales_offset);
        }

        if (brg.is_gemv && brg.treat_y_as_row) {
            if (brg.with_bias) {
                reg_bias.restore();
                add(reg_bias, bias_offset(brg.gemv_bd_block()));
                reg_bias.save();
            }

            if (brg.with_wei_scales && !brg.gemv_single_wei_scale()) {
                reg_wei_scales.restore();
                add(reg_wei_scales, wei_scales_offset(brg.gemv_bd_block()));
                reg_wei_scales.save();
            }
        }

        advance_bd_block2_post_op_regs(bd_block2, is_bdb_tail);
    };

    int rows_for_rd_tail, bd_blocks_for_rd_tail;

    if (brg.is_tmm) {
        rows_for_rd_tail = 0;
        bd_blocks_for_rd_tail = 0;
    } else {
        rows_for_rd_tail = 0;
        if (brg.rdb_tail != 0
                && (brg.is_bf16 || brg.is_f16 || brg.is_int8 || brg.is_fp8)) {
            const auto rd_tail_size = brg.rdb_tail % brg.rd_step;
            rows_for_rd_tail = rd_tail_size
                    ? div_up(brg.rd_step - rd_tail_size, brg.reduce_dim)
                    : 0;
        }
        bd_blocks_for_rd_tail
                = div_up(nstl::max(0,
                                 rows_for_rd_tail - brg.bdb_tail
                                         + brg.brgattr.max_bottom_vpad),
                        brg.bd_block);
    }

    auto bdb_loop_avx512 = [&](bool skip_accumulation) {
        Label bdb_loop_end_label, no_vpad_label;
        if (vpad_exist) {
            // max_top_vp is restricted by bd_block due to
            // brgemm_kernel implementation. TODO: remove this restriction
            assert(brg.brgattr.max_top_vpad <= brg.bd_block
                    && brg.brgattr.max_bottom_vpad <= brg.bd_block);

            if (brg.type == brgemm_strd) {
                // if batch is nullptr then it means no vpadding in this call
                reg_aux_batch.restore();
                cmp(reg_aux_batch, 0);
                je(no_vpad_label, T_NEAR);
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
                mov(reg_bdb_loop, bdblocks);
                L_aligned(bdb_loop_label, 64);
                {
                    bdb_loop_body(1, false, false, false,
                            bd_blocks_for_rd_tail <= 1 ? 0 : rows_for_rd_tail,
                            skip_accumulation);
                    dec(reg_bdb_loop);
                    cmp(reg_bdb_loop, 1);
                    jg(bdb_loop_label, T_NEAR);
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
            if (brg.type == brgemm_strd) jmp(bdb_loop_end_label);
        }
        if (!vpad_exist || brg.type == brgemm_strd) {
            // for brgemm_strd batch may be null so we need this code path
            L_aligned(no_vpad_label, 64);
            if (brg.bdb > 0) {
                mov(reg_bdb_loop, brg.bdb);
                if (brg.bdb > (rows_for_rd_tail ? 1 : 0)) {
                    Label bdb_loop_label;
                    L_aligned(bdb_loop_label, 64);
                    {
                        bdb_loop_body(1, false, false, false,
                                bd_blocks_for_rd_tail <= 1 ? 0
                                                           : rows_for_rd_tail,
                                skip_accumulation);
                        dec(reg_bdb_loop);
                        cmp(reg_bdb_loop, rows_for_rd_tail ? 1 : 0);
                        jg(bdb_loop_label, T_NEAR);
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
        L_aligned(bdb_loop_end_label, 64);
    };
    auto bdb_loop_amx = [&](bool skip_accumulation) {
        if (brg.amx_wary_k_tail()) {
            Label bdb_loop_label;
            auto bdblocks = brg.bdb2;
            if (bdblocks > 1) {
                mov(reg_bdb_loop, brg.bdb2);
                L_aligned(bdb_loop_label, 64);
                {
                    reg_bdb_loop.save();
                    bdb_loop_body(brg.bd_block2, false, false, false, 0,
                            skip_accumulation);
                    reg_bdb_loop.restore();
                    dec(reg_bdb_loop);
                    cmp(reg_bdb_loop, 1);
                    jg(bdb_loop_label, T_NEAR);
                }
                bdblocks = 1;
            }
            if (bdblocks == 1) {
                const bool last_bdb = brg.bdb2_tail == 0 && brg.bdb_tail == 0;
                bdb_loop_body(brg.bd_block2, false, false, last_bdb, 0,
                        skip_accumulation);
            }

            if (brg.bdb2_tail > 0) {
                const bool last_bdb = brg.bdb_tail == 0;
                bdb_loop_body(brg.bdb2_tail, false, false, last_bdb, 0,
                        skip_accumulation);
            }
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, false, false, 0, skip_accumulation);

        } else {
            Label bdb_loop_label;
            if (brg.bd_block2 >= 1) {
                mov(reg_bdb_loop, brg.bdb2);
                L_aligned(bdb_loop_label, 64);
                {
                    reg_bdb_loop.save();
                    bdb_loop_body(brg.bd_block2, false, false, false, 0,
                            skip_accumulation);
                    reg_bdb_loop.restore();
                    dec(reg_bdb_loop);
                    cmp(reg_bdb_loop, 0);
                    jg(bdb_loop_label, T_NEAR);
                }
            }
            if (brg.bdb2_tail > 0)
                bdb_loop_body(brg.bdb2_tail, false, false, false, 0,
                        skip_accumulation);
            if (brg.bdb_tail > 0)
                do_ldb_loop(1, true, false, false, 0, skip_accumulation);
        }
    };

    auto bdb_loop_general = [&](bool skip_accumulation) {
        if (brg.type == brgemm_addr && brg.brgattr.max_bs == 1 && !vpad_exist
                && !skip_accumulation) {
            reg_batch.restore();
            mov(reg_aux1_A, ptr[reg_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
            reg_aux1_A.save();
            mov(reg_aux1_B, ptr[reg_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
        }

        xor_(reg_a_offset, reg_a_offset);
        if (brg.is_tmm)
            bdb_loop_amx(skip_accumulation);
        else
            bdb_loop_avx512(skip_accumulation);
    };

    if (brg.brgattr.generate_skip_accumulation) {
        Label bdb_loop_skip_acc_label, bdb_loop_done_label;
        reg_skip_accm.restore();
        cmp(reg_skip_accm, 0);
        jnz(bdb_loop_skip_acc_label, T_NEAR);

        bdb_loop_general(false);
        jmp(bdb_loop_done_label, T_NEAR);

        L_aligned(bdb_loop_skip_acc_label, 64);
        bdb_loop_general(true);

        L_aligned(bdb_loop_done_label, 64);
    } else
        bdb_loop_general(false);
}

template <typename Wmm>
void jit_brgemm_kernel_t<Wmm>::generate() {
    preamble();

    sub(rsp, regscratchpad_.Size());

    vpad_exist
            = (brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0)
            ? true
            : false;
    need_comp_pads = IMPLICATION(brg.zp_type_a == brgemm_broadcast_t::none,
                             brg.req_s8s8_compensation)
            && IMPLICATION(!vpad_exist, brg.req_cal_comp_pads);

    if (is_superset(brg.isa_impl, avx512_core)) {
        const auto full_mask = size_t {0xffffffffffffffff};

        reg64_t reg_mask = rax;

        if (!brg.is_gemv) {
            const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);

            mov(reg_mask, full_mask);
            kmovq(ld_full_mask, reg_mask);
            mov(reg_mask, tail_mask);
            kmovq(ld_tail_mask, reg_mask);

            if (brg.fp8_with_f16_vnni_block) {
                const auto f8_tail = size_t((1 << (brg.ldb_tail * 2)) - 1);
                mov(reg_mask.cvt32(), f8_tail);
                kmovd(fp8_tail_mask, reg_mask.cvt32());
            }
        } else {
            const auto partial_mask
                    = brg.transA ? size_t((1 << brg.gemv_tail) - 1) : 1;

            mov(reg_mask, full_mask);
            kmovq(gemv_full_mask, reg_mask);
            mov(reg_mask, partial_mask);
            kmovq(gemv_partial_mask, reg_mask);

            // non-transA GEMV code path may require a mask for the reduce
            // dimension.
            if (!brg.transA && brg.gemv_tail > 0) {
                mov(reg_mask, static_cast<size_t>((1 << brg.gemv_tail) - 1));
                kmovq(rd_tail_mask, reg_mask);
            }
        }
    }

    if (brg.is_int8 && !brg.has_int8_vnni) {
        mov(reg_tmp_gpr.cvt16(), 0x1);

        if (is_superset(brg.isa_impl, avx512_core))
            vpbroadcastw(int8_ones_words(), reg_tmp_gpr.cvt16());
        else if (is_superset(brg.isa_impl, avx2)) {
            movq(Xmm(int8_ones_words().getIdx()), reg_tmp_gpr);
            vpbroadcastw(int8_ones_words(), Xmm(int8_ones_words().getIdx()));
        } else
            assert(!"unsupported isa");
    }

    if (brg.is_f16_b_non_amx_vnni()) {
        mov(reg_tmp_gpr, f16_perm_even_table_);
        vmovups(f16_perm_even_vreg(), ptr[reg_tmp_gpr]);
        mov(reg_tmp_gpr, f16_perm_odd_table_);
        vmovups(f16_perm_odd_vreg(), ptr[reg_tmp_gpr]);
    }

    if (brg.is_tmm && brg.amx_wary_k_tail()) {
        // save tiles description for later use
        brgemm_init_tiles(brg, (char *)(&palette_));
    }

    read_params();

    bdb_loop();

    add(rsp, regscratchpad_.Size());

    postamble();

    align(32);

    const int simd = vreg_traits_t<Vmm>::vlen / sizeof(float);
    if (!isa_has_masks(brg.isa_impl) && (brg.ldb_tail || brg.gemv_tail)) {
        const int tail_size = brg.is_gemv ? brg.gemv_tail : brg.ldb_tail;
        L(avx_tail_mask_);
        for (dim_t i = 0; i < tail_size; ++i)
            dd(0xffffffff);
        for (dim_t i = tail_size; i < simd; ++i)
            dd(0);
    }

    if (brg.is_fp8_via_convert()) {
        if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
        if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
    }

    if (brg.with_eltwise || brg.with_sum)
        postops_injector_->prepare_table(/* generate = */ true);

    if (brg.is_f16_b_non_amx_vnni()) {
        // convert interleaved vnni data with holes to packed.
        align(64);
        L(f16_perm_even_table_);
        for (dim_t i = 0; i < 32; ++i) {
            if (i < 16)
                dw(uint16_t(2 * i));
            else
                dw(uint16_t(0));
        }
        align(64);
        L(f16_perm_odd_table_);
        for (dim_t i = 0; i < 32; ++i)
            if (i < 16)
                dw(uint16_t(2 * i + 1));
            else
                dw(uint16_t(0));
    }
}

brgemm_attr_t::brgemm_attr_t()
    : max_bs(INT_MAX)
    , max_top_vpad(0)
    , max_bottom_vpad(0)
    , max_top_bpad(0)
    , max_bottom_bpad(0)
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
    , LDA2(0)
    , LDB2(0)
    , LDC2_M(0)
    , LDC2_N(0)
    , bd_mask(nullptr)
    , static_offsets(nullptr) {}

brgemm_kernel_t *create_brgemm_kernel(const brgemm_desc_t &brg) {
    if (brg.type == brgemm_static_offs) return nullptr;

    if (brg.is_tmm) return new jit_brgemm_kernel_t<Xbyak::Tmm>(brg);
    if (brg.is_zmm) return new jit_brgemm_kernel_t<Xbyak::Zmm>(brg);
    if (brg.is_ymm) return new jit_brgemm_kernel_t<Xbyak::Ymm>(brg);
    return nullptr;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
