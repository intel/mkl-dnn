/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "cpu/x64/jit_uni_i8i8_pooling.hpp"
#include <math.h>

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc};
}

static inline dim_t get_offset(const memory_desc_wrapper &mdw, dim_t n, dim_t c,
        dim_t d, dim_t h, dim_t w) {
    switch (mdw.ndims()) {
        case 3: return mdw.blk_off(n, c, w);
        case 4: return mdw.blk_off(n, c, h, w);
        case 5: return mdw.blk_off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace Xbyak;

using namespace dnnl::impl::utils;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_uni_i8i8_pool_call_params_t, field)

struct jit_uni_i8i8_pool_call_params_t {
    const char *src_i8;
    const char *dst_i8;
    const char *dst_orig;
    const void *post_ops_binary_rhs_arg_vec;
    size_t kd_range;
    size_t kh_range;
    size_t kw_range;
    float idivider;
    const char *src_safe_access;
    const char *dst_safe_access;
};

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_pooling_fwd_ker_t)

    using Vmm = typename cpu_isa_traits_t<isa>::Vmm;
    Xmm xreg(int idx) const { return Xmm(idx); }
    Ymm yreg(int idx) const { return Ymm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) const { return Vmm(xreg(idx).getIdx()); }

    // In case of avx2 with data type i8 we need to use
    // maskmovdqu and maskmovq instructions which has its destination hardcoded in rdi.
    // Windows ABI: abi_param1 is rcx - nothing to do else
    // Unix ABI: abi_param1 is rdi - copy it to rcx and use it as abi_param1
    Reg64 reg_param = rcx; // Our "unified abi_param1"
    Reg64 reg_ptr_src_i8 = r8;
    Reg64 reg_ptr_dst_i8 = r9;
    Reg64 reg_ptr_maskmovdqu_dst = rdi; // store destination - must be rdi

    Reg64 reg_kd_index
            = rdi; // shared with reg_ptr_maskmovdqu_dst; only used before store
    Reg64 reg_kh_index = r11;
    Reg64 reg_kw_index = r10;
    Reg64 reg_kd = r14;
    Reg64 reg_kh = r13;
    Reg64 reg_kw = r12;
    Reg64 c_iter = r15; // shared with reg_mask; only used after mask init

    Reg64 aux_reg_src_d
            = rdx; // shared with reg_tmp; loaded before each accum loop, unused during store
    Reg64 aux_reg_src_h = rax;
    Reg64 aux_reg_src_w = rbx;

    Reg64 reg_tmp = rdx; // only used during mask init and store
    Reg64 reg_src_safe_access = rbp;
    Reg64 reg_dst_safe_access = rsi;

    Reg64 reg_mask = r15; // only used during mask init

    Opmask k_cmp_mask = Opmask(7);

    Opmask mask(int idx) { return Opmask(6 - idx); }

    // ref to any of XYZ-regs via xreg/yreg/vreg functions
    Xmm xmm_tmp = xreg(0); // temp to init vreg_tmp
    Vmm vreg_tmp = vreg(0); // max pooling : holds minimum values for data_type
    Vmm vreg_zeros = vreg(1);
    Vmm vreg_ubound = vreg(2);

    // only in case of <isa> == avx2
    Vmm vreg_mask = vreg(3); // full byte-mask
    Xmm xreg_mask_lo = xreg(
            3); // low 128-bits part of byte-mask (alias for xmm part of vreg_mask)
    Xmm xreg_mask_hi = xreg(
            4); // "max" - high 128-bits part of byte-mask (stored separately)
    Vmm vreg_tail = vreg(5);

    // vreg_mask shifted left (aligned left) to be used in tail processing.
    // Example:       idx [31..0]
    //          vreg_mask = [0,0,0,0,0,.....,0,x,x,x,x,x] ; x => byte mask (msb set)
    //          vreg_mask_2 = [x,x,x,x,x,0,0,0,0,0,.....,0]
    Vmm vreg_mask_2 = vreg(6);
    Xmm xreg_mask_2_lo = xreg(6); // similar to xreg_mask_lo
    Xmm xreg_mask_2_hi = xreg(7); // similar to xreg_mask_hi

    Vmm vreg_mask_q = vreg(4); // "avg" - 1/4 part for non-zero tails
    Mmx mmx_dst_i8 = Mmx(
            0); // "avg" - Mmx reg for masked store results of s8/u8 operations
    Mmx mmx_full_msk = Mmx(
            1); // "avg" - Mmx reg for full mask (all 8 bytes) - used until not in tail
    Mmx mmx_tmp = Mmx(2);
    int post_op_tail_opmask_idx_ = -1;
    jit_pool_conf_t jpp;
    std::unique_ptr<injector::jit_uni_postops_injector_t<Vmm>>
            postops_injector_;

    enum : int { max_vidx_base = utils::one_of(isa, avx2) ? 8 : 3 };
    //"avg" pool uses more registers for unrolling.
    enum : int { avg_vidx_base = utils::one_of(isa, avx2) ? 5 : 3 };

    Vmm max_base_vr(int idx) const { return vreg(max_vidx_base + idx); }
    Vmm avg_base_vr(int idx) const { return vreg(avg_vidx_base + idx); }

    int sizeof_src_dt() const {
        return static_cast<int>(data_type_size(jpp.src_dt));
    }
    int sizeof_dst_dt() const {
        return static_cast<int>(data_type_size(jpp.dst_dt));
    }

    static constexpr data_type_t post_ops_proc_dt = data_type::f32;
    static constexpr int post_ops_simd_w = cpu_isa_traits_t<isa>::vlen
            / sizeof(typename prec_traits_t<post_ops_proc_dt>::type);

    // s32 (in case of avg) or f32 (in case of post ops) used for processing of
    // s8/u8 data, thus we need to take into account ratio of sizes f32/u8 = 4
    enum : int {
        max_num_ll = sizeof(typename prec_traits_t<post_ops_proc_dt>::type)
                / sizeof(typename prec_traits_t<data_type::u8>::type)
    };

    inline size_t get_offset_dst(int jj, int ll) const {
        return (ll * jpp.c_block / max_num_ll + jj * jpp.c_block)
                * sizeof_dst_dt();
    }

    /* max pooling */
    Vmm vreg_src(int idx) const {
        return max_base_vr(idx);
    } // [0      .. ur_c-1]
    Vmm vreg_dst(int idx) const {
        return max_base_vr(jpp.ur_c + idx);
    } // [ur_c   .. 2*ur_c-1]

    Vmm vreg_dst_f32_max(int ll) const {
        return max_base_vr(2 * jpp.ur_c + ll);
    } // [2*ur_c .. 2*ur_c+max_num_ll-1]

    Vmm vreg_idx() const { return max_base_vr(2 * jpp.ur_c + max_num_ll); }

    /* avg pooling */
    static constexpr data_type_t avg_proc_dt = data_type::s32;
    enum : int { mmx_msk_base_reg = 3 };

    Vmm vreg_src_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 0 * max_num_ll);
    } // ll: 0..4 [0..3]

    Vmm vreg_dst_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 1 * max_num_ll);
    } // ll: 0..4 [4..7]

    Vmm vreg_dst_f32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 1 * max_num_ll);
    } // ll: 0..4 [4..7]

    Mmx mmx_mask(int ll) {
        return Mmx(mmx_msk_base_reg + ll);
    } // ll: 0..4 [Mmx(2)...Mmx(5)]

    static status_t init_post_ops_conf(jit_pool_conf_t &jpp,
            const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);

    void init_tmp_reg();
    void init_mask();

    void prepare_idx_table(Label &idx_table);

    // Narrows a small, fixed-range value (SIMD lane index or blend mask) to the
    // Xbyak 8-bit immediate at the single instruction-emit boundary.
    static uint8_t to_imm_uint8_t(int v) noexcept {
        assert(v >= 0 && v <= UINT8_MAX);
        return static_cast<uint8_t>(v);
    }

    // Helper for unpack_dst(). Brings the ll-th part of vreg_dst(jj) into an
    // XMM. For ll == 0 the data is already in place; otherwise the part is
    // extracted into vreg_dst_f32_max(ll).
    Xmm xreg_dst_max(int jj, int ll);

    void unpack_dst(int jj, int ll, bool is_signed);
    void pack_dst(int jj, int num_active_ll, bool is_signed);

    void load_vreg_mask_q(int ll) {};

    void load_src_max_op(int jj, int ll, int offset, bool masked, uint64_t msk);
    void load_src_avg_op(int jj, int ll, int offset, bool masked, uint64_t msk);
    void load_src(int jj, int ll, int c_tail);

    void store_dst_max_op(
            int jj, int ll, int offset, bool masked, uint64_t msk);
    void store_dst_avg_op(
            int jj, int ll, int offset, bool masked, uint64_t msk);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_op(const int jj);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate() override;

    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    jit_uni_i8i8_pooling_fwd_ker_t(
            const jit_pool_conf_t &jpp_, const memory_desc_t *dst_md)
        : jit_generator_t(jit_name(), isa)
        , jpp(jpp_)
        , postops_injector_(nullptr) {

        if (jpp.with_postops) {

            const int post_op_tail = jpp.c_tail % post_ops_simd_w;
            switch (jpp.alg) {
                case pooling_max:
                    // For "max" post_op_tail_opmask_idx_ is set to 1, because
                    // only tail[0] is used for loading and storing.
                    post_op_tail_opmask_idx_ = 1;
                    break;
                case pooling_avg_include_padding:
                case pooling_avg_exclude_padding:
                    post_op_tail_opmask_idx_
                            = post_op_tail ? jpp.c_tail / post_ops_simd_w : 0;
                    break;
                default: assert(!"unsupported pooling algorithm");
            }

            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            static constexpr int tmp_vmm_injector = 0;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, r14, r15, r13, preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                    memory_desc_wrapper(*dst_md), post_op_tail,
                    mask(post_op_tail_opmask_idx_),
                    use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<Vmm>>(
                    this, jpp.post_ops, bsp, /* inject_sum = */ false);
        }
    }
};

template <>
Xmm jit_uni_i8i8_pooling_fwd_ker_t<avx2>::xreg_dst_max(int jj, int ll) {
    if (ll == 0) return Xmm(vreg_dst(jj).getIdx());

    vpermq(vreg_dst_f32_max(ll), vreg_dst(jj), to_imm_uint8_t(ll));
    return Xmm(vreg_dst_f32_max(ll).getIdx());
}

template <>
Xmm jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::xreg_dst_max(int jj, int ll) {
    if (ll == 0) return Xmm(vreg_dst(jj).getIdx());

    const Xmm xreg_dst = Xmm(vreg_dst_f32_max(ll).getIdx());
    vextracti32x4(xreg_dst, vreg_dst(jj), to_imm_uint8_t(ll));
    return xreg_dst;
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::unpack_dst(
        int jj, int ll, bool is_signed) {
    const Xmm xreg_dst = xreg_dst_max(jj, ll);

    if (is_signed)
        uni_vpmovsxbd(vreg_dst_f32_max(ll), xreg_dst);
    else
        uni_vpmovzxbd(vreg_dst_f32_max(ll), xreg_dst);

    uni_vcvtdq2ps(vreg_dst_f32_max(ll), vreg_dst_f32_max(ll));
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::pack_dst(
        int jj, int num_active_ll, bool is_signed) {
    auto vreg_po = [&](int ll) {
        return ll < num_active_ll ? vreg_dst_f32_max(ll) : vreg_zeros;
    };

    for (int ll = 0; ll < num_active_ll; ll++)
        saturate_cvt_f32(
                vreg_dst_f32_max(ll), vreg_zeros, vreg_ubound, jpp.dst_dt);

    // s32 -> s16 always saturates as signed; the final uni_vpackuswb clamps to
    // [0, 255] for u8, so an unsigned first stage is not needed.
    uni_vpackssdw(vreg_dst(jj), vreg_dst_f32_max(0), vreg_po(1));
    if (num_active_ll >= 3)
        uni_vpackssdw(vreg_dst_f32_max(2), vreg_dst_f32_max(2), vreg_po(3));

    if (is_signed)
        uni_vpacksswb(vreg_dst(jj), vreg_dst(jj), vreg_po(2));
    else
        uni_vpackuswb(vreg_dst(jj), vreg_dst(jj), vreg_po(2));

    // The result of previous pack* instructions is reordered, so it needs to be
    // permuted using indices stored in vreg_idx.
    vpermd(vreg_dst(jj), vreg_idx(), vreg_dst(jj));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_vreg_mask_q(int ll) {

    // extract ll-th part of mask (ll-th QWORD)
    vpblendd(vreg_mask_q, vreg_zeros, vreg_mask,
            to_imm_uint8_t(0x3 << 2 * ll)); // 0x3 - mask for 2 x DWORD

    // Move mask from ll-th pos to 0-th pos
    if (ll > 0) vpermq(vreg_mask_q, vreg_mask_q, to_imm_uint8_t(ll));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_src_max_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32) {
            vpmaskmovd(vreg_src(jj), vreg_mask, ptr[aux_reg_src_w + offset]);
        } else {
            // Steps to access 'tail' section:
            // 1) First load all data from the shifted src ptr
            // 2) Now bring the required data from the end of reg to beginning.
            // Example:  idx=[31..0]
            //    vreg_src = [x,x,x,x,.....,x,-,-,-,-,-] ; x => byte data
            //    shift to transform vreg_src = [-,-,-,-,-,x,..,x,x,x,x,]
            const uint8_t shift = static_cast<uint8_t>(
                    cpu_isa_traits_t<avx2>::vlen - jpp.c_tail);

            if (jpp.safe_c_tail) {

                /* load src_tail at 'src_address - shift' so that it does not
                 * spill over the memory boundary */
                vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset - shift]);

                vperm2i128(vreg_tmp, vreg_src(jj), vreg_src(jj), 0x81);
                vpalignr(vreg_src(jj), vreg_tmp, vreg_src(jj), shift);

            } else {
                Label load_data_safely, done;
                add(aux_reg_src_w, offset);

                // Check if mask crosses page boundary
                cmp(aux_reg_src_w, reg_src_safe_access);
                ja(load_data_safely, T_NEAR);

                vpblendvb(
                        vreg_src(jj), vreg_tmp, byte[aux_reg_src_w], vreg_mask);
                jmp(done, T_NEAR);

                L(load_data_safely);

                /* load src_tail at 'src_address - shift' so that it does not
                 * spill over the memory boundary */
                vmovups(vreg_src(jj), ptr[aux_reg_src_w - shift]);

                vperm2i128(vreg_tmp, vreg_src(jj), vreg_src(jj), 0x81);
                vpalignr(vreg_src(jj), vreg_tmp, vreg_src(jj), shift);

                L(done);
                sub(aux_reg_src_w, offset);
            }
        }

    } else
        vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::load_src_max_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            vmovups(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
        else
            vmovdqu8(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
    } else
        vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_src_avg_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    auto load_i8 = [&](bool is_signed, const Vmm &vr_src) {
        // Need to use mask of tail?
        if (masked) {

            // load ll-th part of mask into vreg_mask_q
            load_vreg_mask_q(ll);

            // Steps to access 'tail' section:
            // 1) First load all data from the shifted src ptr
            // 2) Now bring the required data from the end of reg to begining.
            // Example:  idx=[31..0]
            //    vreg_src = [x,x,x,x,.....,x,-,-,-,-,-] ; x => byte data
            //    shift to transform vreg_src = [-,-,-,-,-,x,..,x,x,x,x,]
            // Re-purposing vreg_zeros here. Set it back to zero immmediately.
            const int msk_gran = static_cast<int>(
                    cpu_isa_traits_t<avx2>::vlen / data_type_size(avg_proc_dt));

            const uint8_t shift
                    = static_cast<uint8_t>(cpu_isa_traits_t<avx2>::vlen
                            - (jpp.c_tail > (ll + 1) * msk_gran
                                            ? msk_gran
                                            : jpp.c_tail - (ll * msk_gran)));
            if (jpp.safe_c_tail) {
                /* load src_tail at 'src_address - shift' so that it does not
                 * spill over the memory boundary */
                vmovups(vr_src, ptr[aux_reg_src_w + offset - shift]);

                vperm2i128(vreg_zeros, vr_src, vr_src, 0x81);
                vpalignr(vr_src, vreg_zeros, vr_src, shift);
                uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);
            } else {
                Label load_data_safely, done;
                // assume that it is not safe to load the src_tail

                add(aux_reg_src_w, offset);

                // Check if load crosses the memory boundary
                cmp(aux_reg_src_w, reg_src_safe_access);
                ja(load_data_safely, T_NEAR);

                vpblendvb(vr_src, vreg_zeros, ptr[aux_reg_src_w], vreg_mask_q);
                jmp(done, T_NEAR);

                L(load_data_safely);

                /* load src_tail at 'src_address - shift' so that it does not
                 * spill over the memory boundary */
                vmovups(vr_src, ptr[aux_reg_src_w - shift]);

                vperm2i128(vreg_zeros, vr_src, vr_src, 0x81);
                vpalignr(vr_src, vreg_zeros, vr_src, shift);
                uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

                L(done);
                sub(aux_reg_src_w, offset);
            }

            // Conversion s8/u8 -> s32
            if (is_signed)
                vpmovsxbd(vr_src, vr_src);
            else
                vpmovzxbd(vr_src, vr_src);
        } else {

            // Load from mem into vr_src with conversion
            if (is_signed)
                vpmovsxbd(vr_src, ptr[aux_reg_src_w + offset]);
            else
                vpmovzxbd(vr_src, ptr[aux_reg_src_w + offset]);
        }
    };

    switch (jpp.src_dt) {
        case s32:
            if (masked)
                vpmaskmovd(vreg_src_s32(jj, ll), vreg_mask,
                        ptr[aux_reg_src_w + offset]);
            else
                vmovups(vreg_src_s32(jj, ll), ptr[aux_reg_src_w + offset]);
            break;
        case s8: load_i8(true, vreg_src_s32(jj, ll)); break;
        case u8: load_i8(false, vreg_src_s32(jj, ll)); break;
        default: assert(!"unsupported src data type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::load_src_avg_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    const Vmm &vr_src
            = masked ? vreg_src_s32(jj, ll) | mask(ll) : vreg_src_s32(jj, ll);

    switch (jpp.src_dt) {
        case s32: vmovups(vr_src, ptr[aux_reg_src_w + offset]); break;
        case s8: vpmovsxbd(vr_src, ptr[aux_reg_src_w + offset]); break;
        case u8: vpmovzxbd(vr_src, ptr[aux_reg_src_w + offset]); break;
        default: assert(!"unsupported src data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_max_op(jj, ll, offset, masked, jpp.tail[0]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::store_dst_max_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    Label store_data_safely, done;

    int c_block = jpp.c_block;

    const uint64_t low_mask = (1ULL << (c_block / 2)) - 1;
    const uint8_t shift
            = static_cast<uint8_t>(cpu_isa_traits_t<avx2>::vlen - jpp.c_tail);

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                vpmaskmovd(
                        ptr[reg_ptr_dst_i8 + offset], vreg_mask, vreg_dst(jj));
                break;
            case s8:
            case u8: {

                lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);

                if (!jpp.safe_c_tail) {
                    Xmm xreg_dst = Xmm(vreg_dst(jj).getIdx());

                    cmp(reg_ptr_maskmovdqu_dst, reg_dst_safe_access);
                    ja(store_data_safely, T_NEAR);

                    // Store low half by mask (bytes 0...15)
                    vmaskmovdqu(xreg_dst, xreg_mask_lo);

                    // Do we need to store high half (bytes 16...31) ?
                    if (msk & ~low_mask) {
                        vextracti128(xreg_dst, vreg_dst(jj), 1);
                        add(reg_ptr_maskmovdqu_dst, c_block / 2);
                        vmaskmovdqu(xreg_dst, xreg_mask_hi);
                    }
                    jmp(done, T_NEAR);
                }

                L(store_data_safely);

                vperm2i128(vreg_tail, vreg_dst(jj), vreg_dst(jj), 0x08);
                if (shift <= 16) {
                    vpalignr(vreg_tail, vreg_dst(jj), vreg_tail, 16 - shift);
                } else {
                    vpalignr(vreg_tail, vreg_tail, vreg_zeros, 32 - shift);
                }

                Xmm xreg_tail = Xmm(vreg_tail.getIdx());
                // Do we need to store low half (bytes 0...15) ?
                if (msk & ~low_mask) {
                    sub(reg_ptr_maskmovdqu_dst, shift);
                    vmaskmovdqu(xreg_tail, xreg_mask_2_lo);
                    add(reg_ptr_maskmovdqu_dst, c_block / 2);
                } else {
                    add(reg_ptr_maskmovdqu_dst, (c_block / 2) - shift);
                }

                // Store high half by mask (bytes 16..31)
                vextracti128(xreg_tail, vreg_tail, 1);
                vmaskmovdqu(xreg_tail, xreg_mask_2_hi);

                L(done);
            } break;
            default: assert(!"unsupported src data type");
        }
    } else
        vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::store_dst_max_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                break;
            case s8:
            case u8:
                vmovdqu8(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                break;
            default: assert(!"unsupported src data type");
        }
    } else
        vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::store_dst_avg_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    auto s32_to_i8 = [&](bool is_signed, const Vmm &vr_dst) {
        // conversion: s32 -> s16/u16 : {8 x s32}{8 x 0} -> {16 x s16/u16}
        // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
        if (is_signed)
            vpackssdw(vr_dst, vr_dst, vreg_zeros);
        else
            vpackusdw(vr_dst, vr_dst, vreg_zeros);

        // Permute qwords to restore original order
        // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
        vpermq(vr_dst, vr_dst, 0x58);

        // conversion: s16/u16 -> s8/u8 : {16 x s16/u16}{16 x 0} -> {32 x s8/u8}
        // Target QWORD qw = {8 x s8/u8} has proper position: {qw, xx, xx, xx}
        if (is_signed)
            vpacksswb(vr_dst, vr_dst, vreg_zeros);
        else
            vpackuswb(vr_dst, vr_dst, vreg_zeros);
    };

    auto store_i8 = [&](bool is_signed, bool is_masked, const Vmm &vr_dst) {
        // Conversion s32 -> s8/u8
        s32_to_i8(is_signed, vr_dst);

        // early-out for non-masked cases
        if (!is_masked) {
            vmovlps(ptr[reg_ptr_dst_i8 + offset], Xmm(vr_dst.getIdx()));
            return;
        }
        // store 8 bytes
        lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);

        // Need to use mmx 8-bytes operation to avoid memory violations.
        // NOTICE: it was discovered that Intel SSE and Intel AVX instructions
        //         maskmovdqu/vmaskmovdqu
        //         with low 8-bytes mask throws exception if high 8-bytes belongs write-protected page.
        // NOTE: use indirect move via gpr to avoid transition penalty
        vmovq(reg_tmp, Xmm(vr_dst.getIdx()));
        movq(mmx_dst_i8, reg_tmp);

        // mmx_full_msk - mask for all 8 bytes in zero-tail case
        // mmx_mask(ll) - ll-th mask of tail in non-zero-tail case

        const int msk_gran = static_cast<int>(
                cpu_isa_traits_t<avx2>::vlen / data_type_size(avg_proc_dt));

        const int ll_end = (ll + 1) * msk_gran; // ((ll + 1) * 8)

        if (is_masked && (ll_end > jpp.c_tail)) { //implies this tail not full.
            Label store_data_safely, done;
            const uint8_t shift
                    = static_cast<uint8_t>(msk_gran - jpp.c_tail % msk_gran);

            if (!jpp.safe_c_tail) {
                cmp(reg_ptr_maskmovdqu_dst, reg_dst_safe_access);
                ja(store_data_safely, T_NEAR);

                /* store dst_tail with overlap outside the channel dimension,
                 * but assume it's within the memory boundary. */
                maskmovq(mmx_dst_i8, mmx_mask(ll));
                jmp(done, T_NEAR);
            }

            L(store_data_safely);

            /* store dst_tail at 'dst_address - shift' so that it does not
             * spill over the memory boundary */
            movq(mmx_tmp, mmx_mask(ll));
            psllq(mmx_tmp, shift * 8); // multiply with 8 (bits/byte)
            psllq(mmx_dst_i8, shift * 8);
            sub(reg_ptr_maskmovdqu_dst, shift);
            maskmovq(mmx_dst_i8, mmx_tmp);

            L(done);
        } else {
            maskmovq(mmx_dst_i8, mmx_full_msk);
        }
    };

    switch (jpp.dst_dt) {
        case s32:
            if (masked) {
                vpmaskmovd(ptr[reg_ptr_dst_i8 + offset], vreg_mask,
                        vreg_dst_s32(jj, ll));
            } else
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst_s32(jj, ll));
            break;
        case s8: store_i8(true, masked, vreg_dst_s32(jj, ll)); break;
        case u8: store_i8(false, masked, vreg_dst_s32(jj, ll)); break;
        default: assert(!"unsuppotred dst data_type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::store_dst_avg_op(
        int jj, int ll, int offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    const Vmm &vr_dst
            = masked ? vreg_dst_s32(jj, ll) | mask(ll) : vreg_dst_s32(jj, ll);

    switch (jpp.dst_dt) {
        case s32: vmovups(ptr[reg_ptr_dst_i8 + offset], vr_dst); break;
        case s8: vpmovsdb(ptr[reg_ptr_dst_i8 + offset], vr_dst); break;
        case u8: vpmovusdb(ptr[reg_ptr_dst_i8 + offset], vr_dst); break;
        default: assert(!"unsupported dst data_type");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::store_dst(
        int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_max_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::compute_max_op(const int jj) {
    using namespace data_type;
    switch (jpp.src_dt) {
        case s32: vpmaxsd(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        case s8: vpmaxsb(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        case u8: vpmaxub(vreg_dst(jj), vreg_dst(jj), vreg_src(jj)); break;
        default: assert(!"unsupported src data type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::compute_max_op(const int jj) {
    using namespace data_type;

    // Compare
    switch (jpp.src_dt) {
        case s32:
            vpcmpd(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        case s8:
            vpcmpb(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        case u8:
            vpcmpub(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        default: assert(!"unsupported src data type");
    }

    // move max values into vreg_dst
    if (jpp.src_dt == s32)
        vpblendmd(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj), vreg_src(jj));
    else
        vpblendmb(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj), vreg_src(jj));
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_max_step(
        int ur_c, int c_tail) {
    using namespace data_type;

    Label l_kd, l_kh, l_kw;

    auto ih = jpp.ih;
    auto iw = jpp.iw;
    auto c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++)
        uni_vmovups(vreg_dst(jj), vreg_tmp);

    mov(aux_reg_src_d, reg_ptr_src_i8);
    xor_(reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        mov(aux_reg_src_h, aux_reg_src_d);
        xor_(reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            mov(aux_reg_src_w, aux_reg_src_h);
            xor_(reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    load_src(jj, 0, c_tail);
                    compute_max_op(jj);
                }
                add(aux_reg_src_w, c * sizeof_src_dt());
                inc(reg_kw_index);
                cmp(reg_kw_index, reg_kw);
                jl(l_kw, T_NEAR);
            }
            add(aux_reg_src_h, iw * c * sizeof_src_dt());
            inc(reg_kh_index);
            cmp(reg_kh_index, reg_kh);
            jl(l_kh, T_NEAR);
        }
        add(aux_reg_src_d, ih * iw * c * sizeof_src_dt());
        inc(reg_kd_index);
        cmp(reg_kd_index, reg_kd);
        jl(l_kd, T_NEAR);
    }

    // s32: convert vreg_dst to f32, apply post ops, convert back to s32.
    // s8/u8: unpack vreg_dst up to max_num_ll f32 vregs, apply post ops, repack
    // to vreg_dst.
    if (jpp.with_postops) {
        const int num_ll = static_cast<int>(
                data_type_size(post_ops_proc_dt) / data_type_size(jpp.src_dt));
        const int post_op_tail = c_tail % post_ops_simd_w;

        for (int jj = 0; jj < ur_c; jj++) {
            const bool masked = jj == ur_c - 1 && c_tail;
            const int num_active_ll
                    = masked ? utils::div_up(c_tail, post_ops_simd_w) : num_ll;

            injector_utils::vmm_index_set_t vmm_idxs;
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

            for (int ll = 0; ll < num_active_ll; ll++) {
                const int vmm_idx = jpp.src_dt == s32
                        ? vreg_dst(jj).getIdx()
                        : vreg_dst_f32_max(ll).getIdx();
                vmm_idxs.emplace(vmm_idx);

                switch (jpp.src_dt) {
                    case s32: uni_vcvtdq2ps(vreg_dst(jj), vreg_dst(jj)); break;
                    case s8: unpack_dst(jj, ll, true); break;
                    case u8: unpack_dst(jj, ll, false); break;
                    default: assert(!"unsupported src data type");
                }

                if (jpp.with_binary) {
                    rhs_arg_params.vmm_idx_to_out_reg.emplace(
                            vmm_idx, reg_ptr_dst_i8);
                    rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                            vmm_idx, get_offset_dst(jj, ll));

                    // Registering vmm_idx as a tail is only needed when c_tail
                    // is not a multiple of post_ops_simd_w.
                    const bool tail = ll == num_active_ll - 1 && post_op_tail;
                    if (masked && tail)
                        rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
                }
            }

            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);

            switch (jpp.src_dt) {
                case s32:
                    saturate_cvt_f32(
                            vreg_dst(jj), vreg_zeros, vreg_ubound, jpp.dst_dt);
                    break;
                case s8: pack_dst(jj, num_active_ll, true); break;
                case u8: pack_dst(jj, num_active_ll, false); break;
                default: assert(!"unsupported src data type");
            }
        }
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_avg_step(
        int ur_c, int c_tail) {
    using namespace data_type;

    Label l_kd, l_kh, l_kw;

    auto ih = jpp.ih;
    auto iw = jpp.iw;
    auto c = jpp.c;

    const int num_ll = static_cast<int>(
            data_type_size(avg_proc_dt) / data_type_size(jpp.src_dt));

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                // Clearing of src reg is not needed as they are written before read
                uni_vpxor(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll),
                        vreg_dst_s32(jj, ll));
            }
        }
    }

    mov(aux_reg_src_d, reg_ptr_src_i8);
    xor_(reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        mov(aux_reg_src_h, aux_reg_src_d);
        xor_(reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            mov(aux_reg_src_w, aux_reg_src_h);
            xor_(reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    for (int ll = 0; ll < num_ll; ll++) {
                        bool masked = jj == ur_c - 1 && c_tail;
                        size_t msk = jpp.tail[ll];
                        if (!(masked && !msk)) {
                            load_src(jj, ll, c_tail);
                            uni_vpaddd(vreg_dst_s32(jj, ll),
                                    vreg_dst_s32(jj, ll), vreg_src_s32(jj, ll));
                        }
                    }
                }
                add(aux_reg_src_w, c * sizeof_src_dt());
                inc(reg_kw_index);
                cmp(reg_kw_index, reg_kw);
                jl(l_kw, T_NEAR);
            }
            add(aux_reg_src_h, iw * c * sizeof_src_dt());
            inc(reg_kh_index);
            cmp(reg_kh_index, reg_kh);
            jl(l_kh, T_NEAR);
        }
        add(aux_reg_src_d, ih * iw * c * sizeof_src_dt());
        inc(reg_kd_index);
        cmp(reg_kd_index, reg_kd);
        jl(l_kd, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            const bool masked = jj == ur_c - 1 && c_tail;
            const size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                const auto &reg_dst_f32 = vreg_dst_f32(jj, ll);
                const auto &reg_dst_s32 = vreg_dst_s32(jj, ll);
                uni_vcvtdq2ps(reg_dst_f32, reg_dst_s32);
                uni_vfmadd132ps(reg_dst_f32, vreg_zeros, vreg_tmp);

                if (jpp.with_postops) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (jpp.with_binary) {
                        rhs_arg_params.vmm_idx_to_out_reg.emplace(
                                reg_dst_f32.getIdx(), reg_ptr_dst_i8);
                        rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                reg_dst_f32.getIdx(), get_offset_dst(jj, ll));
                        const bool tail = ll == post_op_tail_opmask_idx_;
                        if (tail && masked)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    reg_dst_f32.getIdx());
                    }
                    postops_injector_->compute_vector(
                            reg_dst_f32.getIdx(), rhs_arg_params);
                }

                saturate_cvt_f32(
                        reg_dst_f32, vreg_zeros, vreg_ubound, jpp.dst_dt);

                if (jpp.with_postops && jpp.dst_dt == u8)
                    uni_vpmaxsd(reg_dst_s32, reg_dst_s32, vreg_zeros);

                store_dst(jj, ll, c_tail);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max: compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_c_block() {
    Label l_main_loop;

    auto nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    auto c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop);
        {
            compute_step(ur_c, 0);
            add(reg_ptr_src_i8, ur_c * c_block * sizeof_src_dt());
            add(reg_ptr_dst_i8, ur_c * c_block * sizeof_dst_dt());
            inc(c_iter);
            cmp(c_iter, c_steps);
            jl(l_main_loop, T_NEAR);
        }
    }

    if (ur_c_tail != 0) { compute_step(ur_c_tail, c_tail); }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::init_mask() {
    using namespace data_type;
    using cpu_isa = cpu_isa_traits_t<avx2>;

    // AVX2 mask initialization: mask stored in Ymm-regs
    auto init = [&](uint64_t bit_mask, bool need_ymm_mask = true,
                        bool need_mmx_mask = false) {
        const int QW_PER_VREG = cpu_isa::vlen / sizeof(uint64_t);

        const size_t DBITS = 8 * sizeof_src_dt();
        const uint64_t VMSK = 1ULL << (DBITS - 1);
        const size_t D_PER_QW = (8 * sizeof(uint64_t)) / DBITS;
        uint64_t vmask[QW_PER_VREG];
        for (int i = 0; i < QW_PER_VREG; i++) {
            uint64_t qw_vmask = 0ULL;
            for (size_t j = 0; j < D_PER_QW; j++) {
                if (bit_mask & 1) qw_vmask |= VMSK << DBITS * j;
                bit_mask >>= 1;
            }
            vmask[i] = qw_vmask;
        }

        // Need mask in Ymm regs ?
        if (need_ymm_mask) {

            // Put QWORDS with target mask into xmm regs
            const int xdst_i[QW_PER_VREG]
                    = {xreg_mask_lo.getIdx(), xreg_mask_lo.getIdx(),
                            xreg_mask_hi.getIdx(), xreg_mask_hi.getIdx()};
            const int xsrc_i[QW_PER_VREG] = {
                    vreg_zeros
                            .getIdx(), // 0-th qword insert in zeros -> {qw0,  0}
                    xreg_mask_lo
                            .getIdx(), // 1-st and 0-th merge        -> {qw0,qw1}
                    vreg_zeros.getIdx(), xreg_mask_hi.getIdx()};
            const uint8 qw_dst_idx[QW_PER_VREG]
                    = {0, 1, 0, 1}; // qword index in 128-bit xreg

            for (int i = 0; i < QW_PER_VREG; i++) {
                mov(reg_mask, vmask[i]);
                vpinsrq(Xmm(xdst_i[i]), Xmm(xsrc_i[i]), reg_mask,
                        qw_dst_idx[i]);

                // Need mask in MMX regs also?
                if (need_mmx_mask)
                    movq(mmx_mask(i), reg_mask); // reuse value in reg_mask
            }

            // Merge Low (xreg_mask_lo alias for vreg_mask.xreg)
            // and High (xreg_mask_hi) into full vreg_mask
            // vreg_mask -> {xreg_mask_hi, vreg_mask.xreg}
            vinserti128(vreg_mask, vreg_mask, xreg_mask_hi, 1);

            // Compute mask algned to left from vreg_mask and store it in vreg_mask_2 to be use for tail processing.
            const uint8_t shift = static_cast<uint8_t>(32 - jpp.c_tail);
            vperm2i128(vreg_mask_2, vreg_mask, vreg_mask, 0x08);
            if (shift <= 16) {
                vpalignr(vreg_mask_2, vreg_mask, vreg_mask_2, 16 - shift);
            } else {
                vpalignr(vreg_mask_2, vreg_mask_2, vreg_zeros, 32 - shift);
            }
            vextracti128(xreg_mask_2_hi, vreg_mask_2, 0x1);
        }

        // Need mask in MMX regs ?
        if (need_mmx_mask) {

            // Only in MMX regs ?
            if (!need_ymm_mask)
                for (int i = 0; i < QW_PER_VREG; i++) {
                    mov(reg_mask, vmask[i]);
                    movq(mmx_mask(i), reg_mask);
                }

            // Form full mask for one QWORD
            uint64_t qw_full_vmask = 0ULL;
            for (size_t i = 0; i < D_PER_QW; i++)
                qw_full_vmask |= VMSK << DBITS * i;

            mov(reg_mask, qw_full_vmask);
            movq(mmx_full_msk, reg_mask);
        }
    };

    uint64_t tail_mask = (1ULL << jpp.c_tail) - 1;
    switch (jpp.alg) {
        case pooling_max:
            // For "max" we need mask only in case of non-zero tail
            if (tail_mask) init(tail_mask);
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            // For "avg" we need mask:
            // - s32   - in case of the non-zero tail
            // - s8/u8 - irrespective of the tail in MMX regs (always store by mask)
            //         - for non-zero tail in Ymm regs (for load)
            switch (jpp.src_dt) {
                case s32:
                    if (tail_mask) init(tail_mask);
                    break;
                case s8:
                case u8:
                    init(tail_mask ? tail_mask : ~0ULL, tail_mask != 0, true);
                    break;
                default: assert(!"unsupported src data type");
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::init_mask() {

    switch (jpp.alg) {
        case pooling_max: {
            mov(reg_mask, jpp.tail[0]);
            kmovq(mask(0), reg_mask);

            // Initialize mask(post_op_tail_opmask_idx_) only when needed.
            const int post_op_tail = jpp.c_tail % post_ops_simd_w;
            if (jpp.with_postops && post_op_tail) {
                mov(reg_mask, (1ULL << post_op_tail) - 1);
                kmovq(mask(post_op_tail_opmask_idx_), reg_mask);
            }
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            for (int ll = 0; ll < max_num_ll; ll++) {
                const size_t msk = jpp.tail[ll];
                if (msk) {
                    mov(reg_mask, msk);
                    kmovq(mask(ll), reg_mask);
                }
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            mov(reg_tmp,
                    ptr[reg_param
                            + offsetof(jit_uni_i8i8_pool_call_params_t,
                                    idivider)]);
            uni_vmovq(xmm_tmp, reg_tmp);
            uni_vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case s8:
                    mov(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            uni_vmovq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == s32)
                uni_vpbroadcastd(vreg_tmp, xmm_tmp);
            else if (mayiuse(avx2))
                vpbroadcastb(vreg_tmp, xmm_tmp);
            else
                pshufb(xmm_tmp, vreg_zeros);
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::prepare_idx_table(Label &idx_table) {
    align(64);
    L(idx_table);
    const uint32_t _idx[] = {0, 4, 1, 5, 2, 6, 3, 7};
    for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
        dd(_idx[i]);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::prepare_idx_table(
        Label &idx_table) {
    align(64);
    L(idx_table);
    const uint32_t _idx[]
            = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
        dd(_idx[i]);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::generate() {
    using namespace data_type;

    Label idx_table;
    const bool use_idx_table = jpp.with_postops && jpp.alg == pooling_max
            && utils::one_of(jpp.src_dt, s8, u8);

    preamble();

#if !defined(_WIN32)
    // Always use rcx as abi_param1 -
    // see the note about maskmovdqu/maskmovq near reg_param.
    mov(rcx, rdi);
#endif

#define READ_PARAM(reg, field) \
    mov(reg, ptr[reg_param + offsetof(jit_uni_i8i8_pool_call_params_t, field)])
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kd, kd_range);
    READ_PARAM(reg_kh, kh_range);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_src_safe_access, src_safe_access);
    READ_PARAM(reg_dst_safe_access, dst_safe_access);

#undef READ_PARAM

    init_saturate_f32(vreg_zeros, vreg_ubound, reg_tmp, f32, jpp.dst_dt);

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    init_mask();

    init_tmp_reg();

    if (use_idx_table) uni_vmovups(vreg_idx(), ptr[rip + idx_table]);

    compute_c_block();

    emms();
    postamble();

    if (jpp.with_eltwise && postops_injector_)
        postops_injector_->prepare_table(/* generate = */ true);

    if (use_idx_table) prepare_idx_table(idx_table);
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(
        jit_pool_conf_t &jpp, const pooling_pd_t *ppd) {
    // disabling verbose dispatch messages for unsupported isa for
    // better readability
    if (!mayiuse(isa)) return status::unimplemented;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];

    jpp.id = is_3d ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];

    jpp.od = is_3d ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = is_3d ? pd.strides[ndims - 5] : 1;
    jpp.stride_h = is_1d ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];

    jpp.kd = is_3d ? pd.kernel[ndims - 5] : 1;
    jpp.kh = is_1d ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = is_3d ? pd.padding[0][ndims - 5] : 0;
    jpp.t_pad = is_1d ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const auto back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const auto bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const auto right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    VDISPATCH_POOLING_IC(
            !(jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
                    || back_pad >= jpp.kd || bottom_pad >= jpp.kh
                    || right_pad >= jpp.kw),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "");

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == avx2    : 32 bytes -> 32 for s8/u8, 8 for s32
    //     isa == avx512* : 64 bytes -> 64 for s8/u8, 16 for s32
    int simd_w = static_cast<int>(
            cpu_isa_traits_t<isa>::vlen / data_type_size(jpp.src_dt));

    /* Verify that vlen-sized memory access happens within the tensor's
     * size, otherwise load/store will always spill outside the memory
     * boundary.*/
    const bool safe_load_n_store = IMPLICATION(utils::one_of(isa, avx2),
            jpp.mb * jpp.c * nstl::min(jpp.id, jpp.od)
                            * nstl::min(jpp.ih, jpp.oh)
                            * nstl::min(jpp.iw, jpp.ow)
                    >= simd_w);
    VDISPATCH_POOLING_IC(safe_load_n_store, "safe load-and-store not possible");

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.c_tail != 0;

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    /* If channel_size is bigger than vlen, we can safely assume there is no
     * underflow of memory boundary, so always perform c_tail and save
     * a couple of compute cycles*/
    jpp.safe_c_tail = jpp.c_tail > 0 && jpp.c >= simd_w;

    switch (jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            // avg_proc_dt (s32) defines granularity (because u8/s8 processed as s32)
            // sse : 4, avx2 : 8, avx512 : 16
            const size_t msk_gran
                    = cpu_isa_traits_t<isa>::vlen / data_type_size(avg_proc_dt);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }
            break;
        }
        default: return status::unimplemented;
    }

    CHECK(init_post_ops_conf(jpp, *ppd->attr(), dst_d));

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_post_ops_conf(
        jit_pool_conf_t &jpp, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (post_ops.len() == 0) return status::success;

    jpp.with_eltwise = post_ops.find(primitive_kind::eltwise) != -1;
    jpp.with_binary = post_ops.find(primitive_kind::binary) != -1;
    jpp.with_postops = jpp.with_eltwise || jpp.with_binary;

    VDISPATCH_POOLING_IC(jpp.with_postops, VERBOSE_UNSUPPORTED_POSTOP);

    jpp.post_ops = post_ops;

    using namespace injector;
    const bool po_ok = post_ops_ok(post_ops_ok_args_t(isa, {binary, eltwise},
            post_ops, &dst_d, false /*sum_at_pos_0_only*/,
            false /*sum_requires_scale_one*/, false /*sum_requires_zp_zero*/,
            false /*sum_requires_same_params*/,
            get_supported_bcast_strategies()));

    // Verbose is reported inside `post_ops_ok`.
    if (!po_ok) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jpp_, this);
}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::jit_uni_i8i8_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ker_(nullptr) {}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::~jit_uni_i8i8_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            new jit_uni_i8i8_pooling_fwd_ker_t<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));
    return ker_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_i8 = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst_i8 = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &jpp = pd()->jpp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
    /* Calculate when the memory-access will happen outisde of the memory
     * boundary, if so, compute a safe memory access. */
    const auto src_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(src_i8 + src_d.size() - 1)
            - (cpu_isa_traits_t<isa>::vlen - 1));

    const auto dst_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(dst_i8 + dst_d.size() - 1)
            - (cpu_isa_traits_t<isa>::vlen - 1));

    parallel_nd(jpp.mb, jpp.od, jpp.oh, jpp.ow,
            [= COMPAT_THIS_CAPTURE](dim_t n, dim_t od, dim_t oh, dim_t ow) {
        dim_t id = nstl::max(od * jpp.stride_d - jpp.f_pad, dim_t(0));
        dim_t ih = nstl::max(oh * jpp.stride_h - jpp.t_pad, dim_t(0));
        dim_t iw = nstl::max(ow * jpp.stride_w - jpp.l_pad, dim_t(0));

        dim_t kd_start = nstl::max(dim_t(0), jpp.f_pad - od * jpp.stride_d);
        dim_t kd_end = nstl::min(
                dim_t(jpp.kd), jpp.id + jpp.f_pad - od * jpp.stride_d);
        dim_t kh_start = nstl::max(dim_t(0), jpp.t_pad - oh * jpp.stride_h);
        dim_t kh_end = nstl::min(
                dim_t(jpp.kh), jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        dim_t kw_start = nstl::max(dim_t(0), jpp.l_pad - ow * jpp.stride_w);
        dim_t kw_end = nstl::min(
                dim_t(jpp.kw), jpp.iw + jpp.l_pad - ow * jpp.stride_w);

        auto p = jit_uni_i8i8_pool_call_params_t();
        p.src_i8 = &src_i8[get_offset(src_d, n, 0, id, ih, iw)
                * src_d.data_type_size()];
        p.dst_i8 = &dst_i8[get_offset(dst_d, n, 0, od, oh, ow)
                * dst_d.data_type_size()];
        p.dst_orig = dst_i8;
        p.kd_range = kd_end - kd_start;
        p.kh_range = kh_end - kh_start;
        p.kw_range = kw_end - kw_start;
        p.idivider = 1.0f
                / ((jpp.alg == pooling_avg_exclude_padding)
                                ? p.kd_range * p.kh_range * p.kw_range
                                : jpp.kd * jpp.kh * jpp.kw);
        p.src_safe_access = src_safe_access;
        p.dst_safe_access = dst_safe_access;
        p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        (*ker_)(&p);
    });
    return status::success;
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>;
template struct jit_uni_i8i8_pooling_fwd_t<avx512_core>;

template struct jit_uni_i8i8_pooling_fwd_ker_t<avx2>;
template struct jit_uni_i8i8_pooling_fwd_t<avx2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
