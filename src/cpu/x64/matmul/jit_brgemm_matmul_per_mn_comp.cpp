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

#include "cpu/x64/matmul/jit_brgemm_matmul_per_mn_comp.hpp"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/matmul/jit_int4_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace Xbyak;

namespace {

alignas(64) constexpr uint32_t int4_interleave_dw[16]
        = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};

// JIT generator: fills one stripe of the per-(M, N) compensation tile for one
// brgemm K block of length G (= k_len). The brgemm post-ops epilogue subtracts
// the tile from the int accumulator.
//
//   wei, G x stripe_w              (B) S-reduce, once per call:
//   +----------------------+       S[n] = sum_k wei[k][n]
//   | plain s8/u8/s4/u4 or | ----> S[0 .. stripe_w) ------------+
//   | VNNI-blocked s4/u4   |                                     |
//   +----------------------+                                     v
//   src, M_blk x G                 delta, M_blk x stripe_w (f32)
//   +-------------+ (D) T-reduce,  +-------------------------------------+
//   | row m       | per row: ----> | (C) m-loop, combine:                |
//   |             | T[m] =         | delta[m][n] = zs' * S[n]            |
//   +-------------+ sum_k src[m][k]|             + (T[m] - G * zs) * zw  |
//                                  +-------------------------------------+
//
//   zs:  src zero point, scalar per call (per-M zero points: one row per
//        call).
//   zs': zs, plus 128 when an s8 src is shifted to u8 (no native s8 x s8).
//   zw:  wei zero point, scalar or per N (vector load, s4 / u4 unpacked).
//   The S term is emitted only with a src zero point or the shift, the T term
//   only with a wei zero point.
//
// The driver (fill_per_mn_compensation()) splits the N block into LDC wide
// stripes, one call per stripe (and per row for per-M src zero points).
// Scalar parameters are converted to f32 on the C++ side.
template <typename Vmm>
struct jit_per_mn_comp_kernel_t : public per_mn_comp_kernel_t,
                                  public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_per_mn_comp_kernel_t)

    using ctx_t = per_mn_comp_kernel_t::ctx_t;

    jit_per_mn_comp_kernel_t(
            const brgemm_matmul_conf_t *bgmmc, bool wei_zp_per_n)
        : per_mn_comp_kernel_t(bgmmc, wei_zp_per_n)
        , jit_generator_t(jit_name())
        , isa_(bgmmc->isa)
        , has_src_zp_(bgmmc->has_zero_point_a)
        , has_wei_zp_(bgmmc->has_zero_point_b)
        , wei_zp_dt_(bgmmc->wei_zp_dt)
        , wei_zp_is_int4_(
                  utils::one_of(bgmmc->wei_zp_dt, data_type::s4, data_type::u4))
        , src_is_signed_(bgmmc->orig_src_dt == data_type::s8)
        , wei_is_signed_(bgmmc->orig_wei_dt == data_type::s8
                  || bgmmc->orig_wei_dt == data_type::s4)
        , wei_is_subbyte_(utils::one_of(
                  bgmmc->orig_wei_dt, data_type::s4, data_type::u4))
        , has_vnni_(is_superset(bgmmc->isa, avx512_core_vnni)
                  || is_superset(bgmmc->isa, avx2_vnni))
        // When per-mn-compensation is applied, buffer C is f32 dt required
        // and to avoid f32=>s32=>apply_s8s8_comp=>dst_dt conversion
        // per-mn-compensation kernel should handle the shift
        // brgemm kernel's req_s8s8_compensation flag is
        // suppressed to avoid double compensation.
        , need_s8s8_shift_(bgmmc->s8s8_compensation_required)
        , src_m_stride_bytes_(bgmmc->A_strides[1])
        , src_k_stride_bytes_(bgmmc->A_strides[0])
        , wei_k_stride_bytes_(wei_is_subbyte_ ? bgmmc->B_strides[1] / 2
                                              : bgmmc->B_strides[1])
        , wei_is_blocked_(bgmmc->blocked_B)
        , wei_n_blk_(bgmmc->wei_n_blk)
        , wei_n_blk_stride_bytes_(wei_is_subbyte_ ? bgmmc->B_strides[0] / 2
                                                  : bgmmc->B_strides[0])
        , LDC_(bgmmc->LDC)
        , n_blk_w_(static_cast<int>(nstl::min<dim_t>(bgmmc->LDC, bgmmc->N_blk)))
        , n_tail_w_(bgmmc->N_tail > 0 ? static_cast<int>(nstl::min<dim_t>(
                                                bgmmc->LDC, bgmmc->N_tail))
                                      : 0)
        , stripe_w_(n_blk_w_)
        , chunks_(static_cast<int>(utils::div_up(stripe_w_, simd_w))) {}

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }
    void operator()(const ctx_t *ctx) const override;

private:
    using reg64_t = const Xbyak::Reg64;

#define GET_OFF(x) offsetof(ctx_t, x)

    const cpu_isa_t isa_;
    const bool has_src_zp_;
    const bool has_wei_zp_;
    const data_type_t wei_zp_dt_;
    const bool wei_zp_is_int4_;
    const bool src_is_signed_;
    const bool wei_is_signed_;
    const bool wei_is_subbyte_;
    const bool has_vnni_;
    const bool need_s8s8_shift_;
    const dim_t src_m_stride_bytes_;
    const dim_t src_k_stride_bytes_;
    const dim_t wei_k_stride_bytes_;
    // VNNI-blocked weights: [n / wei_n_blk][k / vnni][wei_n_blk][vnni].
    const bool wei_is_blocked_;
    const dim_t wei_n_blk_;
    const dim_t wei_n_blk_stride_bytes_;
    const dim_t LDC_;
    const int n_blk_w_; // full block width  = min(LDC, N_blk)
    const int n_tail_w_; // N-tail block width = min(LDC, N_tail), 0 if none
    int stripe_w_;
    int chunks_;

    static constexpr int f32_sz = sizeof(float);
    static constexpr int i32_sz = sizeof(int32_t);
    static constexpr int simd_w = vreg_traits_t<Vmm>::vlen / i32_sz;
    static constexpr int vmm_bytes = vreg_traits_t<Vmm>::vlen;
    static constexpr bool is_zmm = std::is_same<Vmm, Xbyak::Zmm>::value;
    // K rows per VNNI group of blocked weights (int8 VNNI granularity).
    static constexpr int blocked_vnni = 4;

    reg64_t reg_param = abi_param1;

    reg64_t reg_M_blk = r12;
    reg64_t reg_stripe_w = r13;
    reg64_t reg_k_len = r14;

    reg64_t reg_a = r8;
    reg64_t reg_b = r9;
    reg64_t reg_iter = r10;
    reg64_t reg_k_iter = r11;
    reg64_t reg_m = rdx;
    reg64_t reg_n = rax;
    reg64_t reg_delta = r15;
    reg64_t reg_tmp = rsi;
    reg64_t reg_T = rbx;

    Opmask k_mask = k1;
    Opmask k_byte_mask = k2;

    // [0 .. chunks_-1]   S_v[chunks_] (lives across the whole m-loop, f32)
    // [chunks_ + 0..+7]  8 reserved slots for persistents + per-chunk temps:
    //                      +0 zs_bc        persistent f32 broadcast
    //                      +1 zw           persistent OR per-chunk
    //                      +2 Gzs_bc       persistent f32 broadcast
    //                      +3 T_bc         per-m f32 broadcast (= T_eff)
    //                      +4 int4_perm    persistent (wei_zp s4/u4 vector)
    //                      +5 int4_0F      persistent (wei_zp s4/u4 vector)
    //                      +6 term         per-chunk temp
    //                      +7 delta        per-chunk temp
    // top                AVX2 mask aux (Vmm(14)/Vmm(15))
    Vmm vmm_S(int c) const { return Vmm(c); }
    Vmm vmm_zs_bc() const { return Vmm(chunks_ + 0); }
    Vmm vmm_zw() const { return Vmm(chunks_ + 1); }
    Vmm vmm_g_sz_bc() const { return Vmm(chunks_ + 2); }
    Vmm vmm_T_bc() const { return Vmm(chunks_ + 3); }
    Vmm vmm_int4_perm() const { return Vmm(chunks_ + 4); }
    Vmm vmm_int4_0F() const { return Vmm(chunks_ + 5); }
    Vmm vmm_term() const { return Vmm(chunks_ + 6); }
    Vmm vmm_delta() const { return Vmm(chunks_ + 7); }
    // Aliases used inside S-reduce only (persistents not yet loaded).
    Vmm vmm_acc_sr() const { return Vmm(chunks_ + 0); }
    Vmm vmm_data_sr() const { return Vmm(chunks_ + 1); }
    Vmm vmm_t1_sr() const { return Vmm(chunks_ + 2); }
    Vmm vmm_t2_sr() const { return Vmm(chunks_ + 3); }
    Vmm vmm_perm_sr() const { return Vmm(chunks_ + 4); }
    Vmm vmm_0f_sr() const { return Vmm(chunks_ + 5); }
    // Aliases used inside T-reduce only (T_bc reloaded right after).
    Vmm vmm_acc_tr() const { return Vmm(chunks_ + 6); } // shares with term
    Vmm vmm_data_tr() const { return Vmm(chunks_ + 7); } // shares with delta
    Vmm vmm_ones_tr() const { return Vmm(chunks_ + 3); } // shares with T_bc

    // AVX2-only auxiliaries.
    Vmm vmm_idx_ = Vmm(15);
    Vmm vmm_tail_ = Vmm(14);

    Xbyak::Label idx_table_;

    void load_param(reg64_t r, int off) { mov(r, ptr[reg_param + off]); }

    // Mask primitives
    void set_full_mask() {
        if (isa_has_masks(isa_)) {
            mov(reg_tmp, simd_w == 16 ? 0xFFFF : 0xFF);
            kmovw(k_mask, reg_tmp.cvt32());
        } else {
            vpcmpeqd(vmm_tail_, vmm_tail_, vmm_tail_);
        }
    }

    void build_low_bits_opmask(Opmask kdst, reg64_t count_reg) {
        assert(isa_has_masks(isa_));
        mov(reg_tmp, -1);
        bzhi(reg_tmp, reg_tmp, count_reg);
        kmovq(kdst, reg_tmp);
    }

    void build_low_bits_mask(reg64_t count_reg) {
        if (isa_has_masks(isa_)) {
            build_low_bits_opmask(k_mask, count_reg);
        } else {
            vmovd(Xbyak::Xmm(vmm_tail_.getIdx()), count_reg.cvt32());
            vpbroadcastd(vmm_tail_, Xbyak::Xmm(vmm_tail_.getIdx()));
            vpcmpgtd(vmm_tail_, vmm_tail_, vmm_idx_);
        }
    }

    void build_kbyte_mask() {
        cmp(reg_iter, simd_w);
        Label cap_ok;
        jle(cap_ok, T_NEAR);
        mov(reg_iter, simd_w);
        L(cap_ok);
        inc(reg_iter);
        shr(reg_iter, 1);
        mov(reg_n, -1);
        bzhi(reg_n, reg_n, reg_iter);
        kmovw(k_byte_mask, reg_n.cvt32());
    }

    void init_int4_helpers() {
        mov(reg_tmp,
                reinterpret_cast<uintptr_t>(
                        static_cast<const void *>(int4_interleave_dw)));
        vmovdqu32(vmm_int4_perm(), ptr[reg_tmp]);
        mov(reg_tmp.cvt32(), 0x0F);
        vpbroadcastd(vmm_int4_0F(), reg_tmp.cvt32());
    }

    void masked_load_f32(Vmm dst, const Xbyak::Address &addr) {
        if (isa_has_masks(isa_))
            vmovups(dst | k_mask | T_z, addr);
        else
            vmaskmovps(dst, vmm_tail_, addr);
    }
    void masked_store_f32(const Xbyak::Address &addr, Vmm src) {
        if (isa_has_masks(isa_))
            vmovups(addr | k_mask, src);
        else
            vmaskmovps(addr, vmm_tail_, src);
    }

    // Build a mask covering `count` low lanes, capped at simd_w.
    // Caller must ensure `count_reg` > 0.
    void build_chunk_mask(reg64_t count_reg) {
        Label use_full, done;
        cmp(count_reg, simd_w);
        jge(use_full, T_NEAR);
        build_low_bits_mask(count_reg);
        jmp(done, T_NEAR);
        L(use_full);
        set_full_mask();
        L(done);
    }

    // VNNI helpers (T-reduce)
    void vpdpbusd_ones(Vmm acc, Vmm data, bool data_is_signed) {
        const auto enc = is_zmm ? Xbyak::EvexEncoding : Xbyak::VexEncoding;
        if (data_is_signed)
            vpdpbusd(acc, vmm_ones_tr(), data, enc);
        else
            vpdpbusd(acc, data, vmm_ones_tr(), enc);
    }

    void broadcast_byte_ones(Vmm v) {
        mov(reg_tmp, 1);
        if (isa_has_masks(isa_)) {
            vpbroadcastb(v, reg_tmp.cvt8());
        } else {
            const Xbyak::Xmm ones_x(v.getIdx());
            vmovd(ones_x, reg_tmp.cvt32());
            vpbroadcastb(v, ones_x);
        }
    }

    void hreduce_dword(Vmm src) {
        const Xbyak::Ymm src_y(src.getIdx());
        const Xbyak::Ymm tmp_y(vmm_data_tr().getIdx());
        const Xbyak::Xmm src_x(src.getIdx());
        const Xbyak::Xmm tmp_x(vmm_data_tr().getIdx());
        if (is_zmm) {
            vextracti64x4(tmp_y, Xbyak::Zmm(src.getIdx()), 1);
            vpaddd(src_y, src_y, tmp_y);
            vextracti32x4(tmp_x, src_y, 1);
        } else {
            vextracti128(tmp_x, src_y, 1);
        }
        vpaddd(src_x, src_x, tmp_x);
        vpshufd(tmp_x, src_x, 0x4E);
        vpaddd(src_x, src_x, tmp_x);
        vpshufd(tmp_x, src_x, 0xB1);
        vpaddd(src_x, src_x, tmp_x);
    }

    // Broadcast f32 scalar from call_params field
    void bcast_f32_param(Vmm dst, int off) {
        vbroadcastss(dst, ptr[reg_param + off]);
    }

    // dst += 128.0f (broadcast). Uses reg_tmp + vmm_term() as scratch; only
    // called at persistent-load time where vmm_term() is free.
    void add_128_f32(Vmm dst) {
        const auto vmm_c = vmm_term();
        mov(reg_tmp.cvt32(), 0x43000000); // 128.0f
        if (isa_has_masks(isa_)) {
            vpbroadcastd(vmm_c, reg_tmp.cvt32());
        } else {
            vmovd(Xbyak::Xmm(vmm_c.getIdx()), reg_tmp.cvt32());
            vpbroadcastd(vmm_c, Xbyak::Xmm(vmm_c.getIdx()));
        }
        vaddps(dst, dst, vmm_c);
    }

    // Loads simd_w int values (zps) of dt from addr, converts to f32.
    // Supports s8 / u8 / s32 / s4 / u4 (int4 path is AVX-512 only).
    // tail_size_bytes: exact byte count for AVX2 byte tails (0 means full load).
    void load_vec_zps_f32(Vmm dst, data_type_t dt, const Xbyak::Address &addr,
            bool tail, int tail_size_bytes = 0) {
        using namespace data_type;
        switch (dt) {
            case s32:
                if (isa_has_masks(isa_)) {
                    if (tail)
                        vmovdqu32(dst | k_mask | T_z, addr);
                    else
                        vmovdqu32(dst, addr);
                } else {
                    if (tail)
                        vpmaskmovd(dst, vmm_tail_, addr);
                    else
                        vmovdqu(dst, addr);
                }
                vcvtdq2ps(dst, dst);
                break;
            case s8:
            case u8: {
                if (isa_has_masks(isa_) && tail) {
                    // simd_w bytes, mask is on dword lanes.
                    if (dt == s8)
                        vpmovsxbd(dst | k_mask | T_z, addr);
                    else
                        vpmovzxbd(dst | k_mask | T_z, addr);
                } else if (!isa_has_masks(isa_) && tail) {
                    // AVX2: avoid OOB read on tails by loading exact byte count first.
                    assert(tail_size_bytes > 0 && tail_size_bytes < simd_w);
                    const Xbyak::Xmm xmm_data(dst.getIdx());
                    load_bytes(xmm_data, addr, tail_size_bytes);
                    if (dt == s8)
                        vpmovsxbd(dst, xmm_data);
                    else
                        vpmovzxbd(dst, xmm_data);
                } else {
                    if (dt == s8)
                        vpmovsxbd(dst, addr);
                    else
                        vpmovzxbd(dst, addr);
                }
                vcvtdq2ps(dst, dst);
                break;
            }
            case s4:
            case u4: {
                assert(is_zmm); // sub-byte vector load is AVX-512 only.
                // Load (simd_w/2) bytes (zero-extended) -> 8 dwords (Ymm).
                const Xbyak::Ymm ymm_low(vmm_delta().getIdx());
                if (tail) {
                    // k_byte_mask is set up by the caller.
                    vpmovzxbd(ymm_low | k_byte_mask | T_z, addr);
                } else {
                    vpmovzxbd(ymm_low, addr);
                }
                // Promote to Zmm view; ymm_low has high half == 0.
                const Vmm v_low = vmm_delta();
                // Low nibbles into v_low's low 4 bits of each dword,
                // high nibbles into vmm_term().
                vpandd(vmm_term(), v_low, vmm_int4_0F());
                vpsrld(v_low, v_low, 4);
                vpandd(v_low, v_low, vmm_int4_0F());
                if (dt == s4) {
                    vpslld(vmm_term(), vmm_term(), 28);
                    vpsrad(vmm_term(), vmm_term(), 28);
                    vpslld(v_low, v_low, 28);
                    vpsrad(v_low, v_low, 28);
                }
                // Interleave: dst = {term[0], v_low[0], term[1], v_low[1],...}
                vmovdqu32(dst, vmm_term());
                vpermt2d(dst, vmm_int4_perm(), v_low);
                vcvtdq2ps(dst, dst);
                break;
            }
            default: assert(!"unsupported wei zp dt"); break;
        }
    }

    // Phase B: S-reduce (vectorized integer, then convert to f32)
    void emit_s_reduce() {
        if (wei_is_blocked_) {
            emit_s_reduce_blocked_subbyte();
            return;
        }
        if (wei_is_subbyte_) {
            emit_s_reduce_subbyte();
            return;
        }
        for (int c = 0; c < chunks_; ++c)
            uni_vpxor(vmm_S(c), vmm_S(c), vmm_S(c));

        load_param(reg_a, GET_OFF(wei_batch_ptr));
        load_param(reg_tmp, GET_OFF(n_base));
        add(reg_a, reg_tmp);

        mov(reg_k_iter, reg_k_len);
        Label k_loop, k_done;
        L(k_loop);
        test(reg_k_iter, reg_k_iter);
        jz(k_done, T_NEAR);
        for (int c = 0; c < chunks_; ++c) {
            Label skip_chunk;
            mov(reg_iter, reg_stripe_w);
            if (c > 0) {
                sub(reg_iter, c * simd_w);
                jle(skip_chunk, T_NEAR);
            }
            const auto addr = ptr[reg_a + c * simd_w];
            if (isa_has_masks(isa_)) {
                build_chunk_mask(reg_iter);
                if (wei_is_signed_)
                    vpmovsxbd(vmm_data_sr() | k_mask | T_z, addr);
                else
                    vpmovzxbd(vmm_data_sr() | k_mask | T_z, addr);
            } else {
                // AVX2: only the final chunk may need a short-byte load.
                const int rem = stripe_w_ - c * simd_w;
                const bool is_last_chunk = (c == chunks_ - 1);
                const bool need_preload
                        = is_last_chunk && rem > 0 && rem < simd_w;
                if (need_preload) {
                    const Xbyak::Xmm xmm_data(vmm_data_sr().getIdx());
                    load_bytes(xmm_data, addr, rem);
                }
                const Vmm src_vmm = vmm_data_sr();
                const auto &src_op = need_preload
                        ? static_cast<const Xbyak::Operand &>(src_vmm)
                        : static_cast<const Xbyak::Operand &>(addr);
                if (wei_is_signed_)
                    vpmovsxbd(vmm_data_sr(), src_op);
                else
                    vpmovzxbd(vmm_data_sr(), src_op);
            }
            vpaddd(vmm_S(c), vmm_S(c), vmm_data_sr());
            L(skip_chunk);
        }
        add(reg_a, wei_k_stride_bytes_);
        dec(reg_k_iter);
        jmp(k_loop, T_NEAR);
        L(k_done);

        // Integer -> f32 once per accumulator.
        for (int c = 0; c < chunks_; ++c)
            vcvtdq2ps(vmm_S(c), vmm_S(c));
    }

    // Sub-byte (s4/u4) S-reduce. AVX-512 only (factory enforces).
    void emit_s_reduce_subbyte() {
        assert(is_zmm);
        for (int c = 0; c < chunks_; ++c)
            uni_vpxor(vmm_S(c), vmm_S(c), vmm_S(c));

        init_int4_helpers();

        load_param(reg_a, GET_OFF(wei_batch_ptr));
        load_param(reg_tmp, GET_OFF(n_base));
        sar(reg_tmp, 1);
        add(reg_a, reg_tmp);

        mov(reg_k_iter, reg_k_len);
        Label k_loop, k_done;
        L(k_loop);
        test(reg_k_iter, reg_k_iter);
        jz(k_done, T_NEAR);
        for (int c = 0; c < chunks_; ++c) {
            Label skip_chunk;
            mov(reg_iter, reg_stripe_w);
            if (c > 0) {
                sub(reg_iter, c * simd_w);
                jle(skip_chunk, T_NEAR);
            }
            // Cap at simd_w, then byte_count = (rem + 1) >> 1.
            build_kbyte_mask();

            const Xbyak::Ymm ymm_data(vmm_data_sr().getIdx());
            vpmovzxbd(ymm_data | k_byte_mask | T_z,
                    ptr[reg_a + c * (simd_w / 2)]);

            vpandd(vmm_t1_sr(), vmm_data_sr(), vmm_0f_sr());
            vpsrld(vmm_t2_sr(), vmm_data_sr(), 4);
            vpandd(vmm_t2_sr(), vmm_t2_sr(), vmm_0f_sr());
            if (wei_is_signed_) {
                vpslld(vmm_t1_sr(), vmm_t1_sr(), 28);
                vpsrad(vmm_t1_sr(), vmm_t1_sr(), 28);
                vpslld(vmm_t2_sr(), vmm_t2_sr(), 28);
                vpsrad(vmm_t2_sr(), vmm_t2_sr(), 28);
            }
            vpermt2d(vmm_t1_sr(), vmm_perm_sr(), vmm_t2_sr());
            vpaddd(vmm_S(c), vmm_S(c), vmm_t1_sr());
            L(skip_chunk);
        }
        add(reg_a, wei_k_stride_bytes_);
        dec(reg_k_iter);
        jmp(k_loop, T_NEAR);
        L(k_done);

        for (int c = 0; c < chunks_; ++c)
            vcvtdq2ps(vmm_S(c), vmm_S(c));
    }

    // (B) S-reduce for VNNI-blocked s4 / u4 weights (see the kernel diagram).
    // AVX-512 only (factory enforces).
    //
    // `BA16a<nb>b4a` (nb = wei_n_blk) is [N / nb][K / 4][nb][4] nibbles. Per
    // 16-column chunk and VNNI row group (4 K rows, 32 bytes):
    //   32 bytes, column c = [k0 k1][k2 k3]
    //     -> int4_utils::expand_to_int8() -> 64 bytes, column c = [k0..k3]
    //     -> dot product with byte ones   -> S[c] += k0 + k1 + k2 + k3
    // The chunk pointer starts at (k_start, n_base) and moves by 16 columns,
    // jumping to the next N block when the current one ends.
    void emit_s_reduce_blocked_subbyte() {
        assert(is_zmm);
        const Vmm vmm_data = vmm_data_sr();
        const Vmm vmm_tmp = vmm_t1_sr();
        const Vmm vmm_ones_w = vmm_t2_sr();
        const Vmm vmm_ones_b = vmm_perm_sr();
        const Vmm vmm_0f = vmm_0f_sr();
        const Vmm vmm_8 = vmm_acc_sr();
        const reg64_t reg_chunk = reg_b;
        const reg64_t reg_chunk_col = reg_n;

        for (int c = 0; c < chunks_; ++c)
            uni_vpxor(vmm_S(c), vmm_S(c), vmm_S(c));

        broadcast_byte_ones(vmm_ones_b);
        int4_utils::init_expand_to_int8_consts(
                this, vmm_0f, vmm_8, wei_is_signed_, reg_tmp);
        if (!has_vnni_) {
            mov(reg_tmp.cvt32(), 1);
            vpbroadcastw(vmm_ones_w, reg_tmp.cvt16());
        }

        const dim_t vnni_row_bytes = wei_n_blk_ * blocked_vnni / 2;
        const int chunk_bytes = simd_w * blocked_vnni / 2;
        load_param(reg_chunk, GET_OFF(wei_batch_ptr));
        load_param(reg_chunk_col, GET_OFF(wei_n_blk_off));
        for (int c = 0; c < chunks_; ++c) {
            Label skip_chunk, k_loop, k_done;
            if (c > 0) {
                mov(reg_iter, reg_stripe_w);
                sub(reg_iter, c * simd_w);
                jle(skip_chunk, T_NEAR);
            }

            // Number of VNNI row groups; padded rows of the blocked layout
            // are zeros.
            mov(reg_a, reg_chunk);
            mov(reg_k_iter, reg_k_len);
            add(reg_k_iter, blocked_vnni - 1);
            shr(reg_k_iter, math::ilog2q(blocked_vnni));

            L(k_loop);
            test(reg_k_iter, reg_k_iter);
            jz(k_done, T_NEAR);
            {
                int4_utils::expand_to_int8(this, vmm_data, vmm_tmp, ptr[reg_a],
                        vmm_0f, vmm_8, wei_is_signed_);
                if (has_vnni_) {
                    vpdpbusd(vmm_S(c), vmm_ones_b, vmm_data,
                            Xbyak::EvexEncoding);
                } else {
                    vpmaddubsw(vmm_tmp, vmm_ones_b, vmm_data);
                    vpmaddwd(vmm_tmp, vmm_tmp, vmm_ones_w);
                    vpaddd(vmm_S(c), vmm_S(c), vmm_tmp);
                }
            }
            add(reg_a, vnni_row_bytes);
            dec(reg_k_iter);
            jmp(k_loop, T_NEAR);
            L(k_done);

            // Next chunk, in the next N block once this one ends.
            if (c + 1 < chunks_) {
                Label same_n_blk;
                add(reg_chunk, chunk_bytes);
                add(reg_chunk_col, simd_w);
                cmp(reg_chunk_col, static_cast<int>(wei_n_blk_));
                jl(same_n_blk, T_NEAR);
                xor_(reg_chunk_col, reg_chunk_col);
                mov(reg_tmp, wei_n_blk_stride_bytes_ - vnni_row_bytes);
                add(reg_chunk, reg_tmp);
                L(same_n_blk);
            }
            L(skip_chunk);
        }

        for (int c = 0; c < chunks_; ++c)
            vcvtdq2ps(vmm_S(c), vmm_S(c));
    }

    // Phase D (inside m-loop): scalar T_m -> f32 broadcast in vmm_T_bc
    // Caller has set `reg_a` to src + (m_base + m) * src_m_stride.
    // After return: vmm_T_bc holds the f32 broadcast of T_eff (= T_m on src-zp-only path).
    void emit_t_reduce_into_T_bc(bool subtract_Gzs) {
        if (has_vnni_) broadcast_byte_ones(vmm_ones_tr());
        uni_vpxor(vmm_acc_tr(), vmm_acc_tr(), vmm_acc_tr());
        mov(reg_tmp, reg_a);
        mov(reg_k_iter, reg_k_len);

        Label k_bulk, k_bulk_done;
        L(k_bulk);
        cmp(reg_k_iter, vmm_bytes);
        jl(k_bulk_done, T_NEAR);
        if (has_vnni_) {
            uni_vmovdqu(vmm_data_tr(), ptr[reg_tmp]);
            vpdpbusd_ones(vmm_acc_tr(), vmm_data_tr(), src_is_signed_);
        } else {
            for (int off = 0; off < vmm_bytes; off += simd_w) {
                if (src_is_signed_)
                    vpmovsxbd(vmm_data_tr(), ptr[reg_tmp + off]);
                else
                    vpmovzxbd(vmm_data_tr(), ptr[reg_tmp + off]);
                vpaddd(vmm_acc_tr(), vmm_acc_tr(), vmm_data_tr());
            }
        }
        add(reg_tmp, vmm_bytes);
        sub(reg_k_iter, vmm_bytes);
        jmp(k_bulk, T_NEAR);
        L(k_bulk_done);

        const bool scalar_tail = !(has_vnni_ && isa_has_masks(isa_));
        if (scalar_tail) xor_(reg_iter, reg_iter);

        Label k_tail_done;
        test(reg_k_iter, reg_k_iter);
        jz(k_tail_done, T_NEAR);
        if (!scalar_tail) {
            push(reg_tmp);
            build_low_bits_mask(reg_k_iter);
            pop(reg_tmp);
            vmovdqu8(Xbyak::Zmm(vmm_data_tr().getIdx()) | k_mask | T_z,
                    ptr[reg_tmp]);
            vpdpbusd_ones(vmm_acc_tr(), vmm_data_tr(), src_is_signed_);
        } else {
            xor_(reg_n, reg_n);
            Label scalar_loop;
            L(scalar_loop);
            if (src_is_signed_)
                movsx(reg_b.cvt32(), byte[reg_tmp + reg_n]);
            else
                movzx(reg_b.cvt32(), byte[reg_tmp + reg_n]);
            add(reg_iter.cvt32(), reg_b.cvt32());
            inc(reg_n);
            cmp(reg_n, reg_k_iter);
            jl(scalar_loop, T_NEAR);
        }
        L(k_tail_done);

        hreduce_dword(vmm_acc_tr());
        vmovd(reg_T.cvt32(), Xmm(vmm_acc_tr().getIdx()));
        if (scalar_tail) add(reg_T.cvt32(), reg_iter.cvt32());

        // Broadcast int32 -> vmm_T_bc; convert to f32.
        if (isa_has_masks(isa_)) {
            vpbroadcastd(vmm_T_bc(), reg_T.cvt32());
        } else {
            const Xbyak::Xmm x(vmm_T_bc().getIdx());
            vmovd(x, reg_T.cvt32());
            vpbroadcastd(vmm_T_bc(), x);
        }
        vcvtdq2ps(vmm_T_bc(), vmm_T_bc());
        if (subtract_Gzs) vsubps(vmm_T_bc(), vmm_T_bc(), vmm_g_sz_bc());
    }

    // Phase C: m-loop -- T_m, combine, write delta
    void emit_m_combine_loop() {
        load_param(reg_delta, GET_OFF(delta_ptr));

        const bool emit_T = has_wei_zp_;
        const bool subtract_Gzs = has_src_zp_ && has_wei_zp_;

        // Per-mn delta is computed afresh for one brgemm K-call;
        // the brgemm post-ops epilogue subtracts the resulting tile
        // each call and the C-buffer accumulates the per-K-block deltas
        // across calls. The JIT always overwrites — no accumulate path.
        emit_m_loop_body(emit_T, subtract_Gzs);
    }

    void emit_m_loop_body(bool emit_T, bool subtract_Gzs) {
        xor_(reg_m, reg_m);
        Label m_loop, m_done;
        L(m_loop);
        cmp(reg_m, reg_M_blk);
        jge(m_done, T_NEAR);

        if (emit_T) {
            load_param(reg_a, GET_OFF(src_batch_ptr));
            load_param(reg_tmp, GET_OFF(m_base));
            add(reg_tmp, reg_m);
            imul(reg_tmp, reg_tmp, src_m_stride_bytes_);
            add(reg_a, reg_tmp);
            emit_t_reduce_into_T_bc(subtract_Gzs);
        }

        // delta row base = delta + m * LDC * 4.
        push(reg_m);
        imul(reg_m, reg_m, LDC_ * f32_sz);
        add(reg_m, reg_delta);

        for (int c = 0; c < chunks_; ++c) {
            mov(reg_iter, reg_stripe_w);
            sub(reg_iter, c * simd_w);
            Label skip;
            cmp(reg_iter, 0);
            jle(skip, T_NEAR);
            build_chunk_mask(reg_iter);

            // Load (or use persistent) per-N wei_zp input.
            if (has_wei_zp_ && wei_zp_per_n_) {
                load_param(reg_tmp, GET_OFF(wei_zp_ptr));
                if (wei_zp_is_int4_) {
                    push(reg_iter);
                    build_kbyte_mask();
                    pop(reg_iter);
                    const dim_t byte_off = c * (simd_w / 2);
                    load_vec_zps_f32(vmm_zw(), wei_zp_dt_,
                            ptr[reg_tmp + byte_off], /* tail = */ true);
                } else if (isa_has_masks(isa_)) {
                    const dim_t off
                            = c * simd_w * types::data_type_size(wei_zp_dt_);
                    load_vec_zps_f32(vmm_zw(), wei_zp_dt_, ptr[reg_tmp + off],
                            /* tail = */ true);
                } else {
                    // AVX2
                    const int rem = stripe_w_ - c * simd_w;
                    const bool has_tail = rem > 0 && rem < simd_w;
                    const dim_t off
                            = c * simd_w * types::data_type_size(wei_zp_dt_);
                    const int tail_bytes = has_tail
                            ? static_cast<int>(
                                      rem * types::data_type_size(wei_zp_dt_))
                            : 0;
                    load_vec_zps_f32(vmm_zw(), wei_zp_dt_, ptr[reg_tmp + off],
                            has_tail, tail_bytes);
                }
            }

            // term = zs_shifted*S[c] + T_eff*zw      (S-term and/or wei-zp term)
            //   zs_shifted = zs (+128 if s8s8 shift folded in); the S-term is also
            //   emitted on the s8s8-shift path even without src zp.
            const bool emit_S_term = has_src_zp_ || need_s8s8_shift_;
            if (emit_S_term && has_wei_zp_) {
                vmulps(vmm_term(), vmm_zs_bc(), vmm_S(c));
                vfmadd231ps(vmm_term(), vmm_T_bc(), vmm_zw());
            } else if (emit_S_term) {
                vmulps(vmm_term(), vmm_zs_bc(), vmm_S(c));
            } else { // has_wei_zp_ only
                vmulps(vmm_term(), vmm_T_bc(), vmm_zw());
            }

            masked_store_f32(ptr[reg_m + c * simd_w * f32_sz], vmm_term());

            L(skip);
        }

        pop(reg_m);
        inc(reg_m);
        jmp(m_loop, T_NEAR);
        L(m_done);
    }

    void emit_compute() {
        // (B) S-reduce (lives across the m-loop). Uses slots chunks_+0..+5
        //     as scratch; persistents loaded AFTER S-reduce.
        //     S[n]=Sum_K wei is needed for the src-zp term and, on non-AMX s8
        //     src, for the +128*S s8->u8 shift correction added into delta.
        const bool emit_S = has_src_zp_ || need_s8s8_shift_;
        if (emit_S) emit_s_reduce();

        // Persistent f32 broadcasts loaded once for the whole call.
        // Shifted src-zp = src_zp (if any) + 128 (if s8s8 shift is active),
        // so zs_shifted*S = zs*S + 128*S. The G*zs*zw cross-term uses the real zs
        // via the separate g_sz_f32 param and is unaffected by the +128.
        if (emit_S) {
            if (has_src_zp_)
                bcast_f32_param(vmm_zs_bc(), GET_OFF(src_zp_f32));
            else
                uni_vpxor(vmm_zs_bc(), vmm_zs_bc(), vmm_zs_bc());
            if (need_s8s8_shift_) add_128_f32(vmm_zs_bc());
        }
        if (has_src_zp_ && has_wei_zp_)
            bcast_f32_param(vmm_g_sz_bc(), GET_OFF(g_sz_f32));
        if (has_wei_zp_ && !wei_zp_per_n_)
            bcast_f32_param(vmm_zw(), GET_OFF(wei_zp_f32));

        // int4 wei_zp helpers: load perm table & 0x0F broadcast.
        if (wei_zp_per_n_ && wei_zp_is_int4_) init_int4_helpers();

        // (C) m-loop with per-m T-reduce and combine.
        emit_m_combine_loop();
    }

    // Select and emit the compute body for a given N-block width.
    void emit_body_for_width(int width) {
        stripe_w_ = width;
        chunks_ = static_cast<int>(utils::div_up(width, simd_w));
        emit_compute();
    }

    void generate() override {
        preamble();

        load_param(reg_M_blk, GET_OFF(M_blk));
        load_param(reg_stripe_w, GET_OFF(stripe_w));
        load_param(reg_k_len, GET_OFF(k_len));

        if (!isa_has_masks(isa_)) { vmovups(vmm_idx_, ptr[rip + idx_table_]); }
        if (!isa_has_masks(isa_) && n_tail_w_ > 0) {
            Label main_blk, done;
            cmp(reg_stripe_w, n_blk_w_);
            je(main_blk, T_NEAR);
            emit_body_for_width(n_tail_w_); // N-tail block
            jmp(done, T_NEAR);
            L(main_blk);
            emit_body_for_width(n_blk_w_); // full N_blk block
            L(done);
        } else {
            emit_body_for_width(n_blk_w_);
        }

        postamble();

        align(64);
        L(idx_table_);
        for (int i = 0; i < simd_w; ++i)
            dd(i);
    }

#undef GET_OFF
};

template <typename Vmm>
void jit_per_mn_comp_kernel_t<Vmm>::operator()(const ctx_t *ctx) const {
    assert(ctx != nullptr);
    assert(ctx->delta_ptr != nullptr);
    assert(ctx->M_blk > 0 && ctx->stripe_w > 0);
    assert(ctx->k_len > 0);

    jit_generator_t::operator()(const_cast<ctx_t *>(ctx));
}

} // namespace

// Factory
status_t per_mn_comp_kernel_t::is_applicable(
        const brgemm_matmul_conf_t *bgmmc) {
    const cpu_isa_t isa = bgmmc->isa;
    if (!is_superset(isa, avx2)) return status::unimplemented;

    // Source / weights dtypes (byte / sub-byte int).
    if (!utils::one_of(bgmmc->orig_src_dt, data_type::s8, data_type::u8))
        return status::unimplemented;
    if (!utils::one_of(bgmmc->orig_wei_dt, data_type::s8, data_type::u8,
                data_type::s4, data_type::u4))
        return status::unimplemented;
    if (bgmmc->a_dt_sz != 1 || bgmmc->b_dt_sz != 1)
        return status::unimplemented;

    if (bgmmc->A_strides[0] != 1) return status::unimplemented;

    const bool wei_is_subbyte
            = utils::one_of(bgmmc->orig_wei_dt, data_type::s4, data_type::u4);
    if (bgmmc->blocked_B) {
        // VNNI-blocked int4 weights (`BA16a<n>b4a`, `aCB16b<n>c4b`): the
        // S-reduce reads 16-column chunks, each of which has to lie within a
        // single N block, and walks K row groups contiguously within it.
        constexpr int chunk_w = 16;
        if (!wei_is_subbyte || !isa_has_masks(isa))
            return status::unimplemented;
        if (data_type_vnni_granularity(bgmmc->wei_dt) != 4)
            return status::unimplemented;
        if (bgmmc->wei_n_blk % chunk_w != 0) return status::unimplemented;
        if (bgmmc->B_strides[1]
                != bgmmc->b_dt_sz * bgmmc->wei_k_blk * bgmmc->wei_n_blk)
            return status::unimplemented;
        const bool n_base_aligned
                = (bgmmc->N_blk % chunk_w == 0 || bgmmc->N_blk >= bgmmc->N)
                && (bgmmc->LDC % chunk_w == 0 || bgmmc->LDC >= bgmmc->N_blk);
        if (!n_base_aligned) return status::unimplemented;
    } else {
        if (bgmmc->B_strides[0] != 1) return status::unimplemented;
        if (wei_is_subbyte) {
            if (!isa_has_masks(isa)) return status::unimplemented;
            if (bgmmc->N_blk % 2 != 0) return status::unimplemented;
            if (bgmmc->B_strides[1] % 2 != 0) return status::unimplemented;
        }
    }

    if (bgmmc->LDC <= 0 || bgmmc->M_blk <= 0 || bgmmc->N_blk <= 0)
        return status::unimplemented;

    // Vmm map: chunks_ + 8. AVX-512 -> chunks < 24; AVX2 -> chunks < 6.
    // Current limitation. TODO.
    const bool zmm_path = isa_has_masks(isa);
    const int simd_w_local = zmm_path ? 16 : 8;
    const dim_t max_stripe_w = nstl::min<dim_t>(bgmmc->LDC, bgmmc->N_blk);
    const int chunks
            = static_cast<int>(utils::div_up(max_stripe_w, simd_w_local));
    const int chunks_cap = zmm_path ? 24 : 6;
    if (chunks > chunks_cap) return status::unimplemented;

    // per-mn compensation is only needed when ZP is present or s8s8 compensation
    // is required for each K-block(grouped scales are present).
    if (!bgmmc->has_zero_point_a && !bgmmc->has_zero_point_b
            && !bgmmc->s8s8_compensation_required)
        return status::unimplemented;

    // Per-axis activation:
    const bool wei_zp_per_n = bgmmc->has_zero_point_b && bgmmc->is_wei_zp_per_n;

    // Vector-load dtype constraints (broadcast scalars accept any dtype via
    // ref_io_helper; vector paths require the JIT to implement the load).
    if (wei_zp_per_n) {
        if (!utils::one_of(bgmmc->wei_zp_dt, data_type::s8, data_type::u8,
                    data_type::s32, data_type::s4, data_type::u4))
            return status::unimplemented;
        const bool wei_zp_is_int4
                = utils::one_of(bgmmc->wei_zp_dt, data_type::s4, data_type::u4);
        if (wei_zp_is_int4 && !zmm_path) return status::unimplemented;
    }

    return status::success;
}

status_t per_mn_comp_kernel_t::create(
        std::unique_ptr<per_mn_comp_kernel_t> &kernel,
        const brgemm_matmul_conf_t *bgmmc) {
    CHECK(is_applicable(bgmmc));

    const cpu_isa_t isa = bgmmc->isa;
    const bool zmm_path = isa_has_masks(isa);

    // Re-derive per-axis activation flags consumed by the JIT impl ctor.
    const bool wei_zp_per_n = bgmmc->has_zero_point_b && bgmmc->is_wei_zp_per_n;

    if (zmm_path) {
        auto jit = utils::make_unique<jit_per_mn_comp_kernel_t<Xbyak::Zmm>>(
                bgmmc, wei_zp_per_n);
        if (!jit) return status::out_of_memory;
        CHECK(jit->create_kernel());
        kernel = std::move(jit);
    } else {
        auto jit = utils::make_unique<jit_per_mn_comp_kernel_t<Xbyak::Ymm>>(
                bgmmc, wei_zp_per_n);
        if (!jit) return status::out_of_memory;
        CHECK(jit->create_kernel());
        kernel = std::move(jit);
    }
    return status::success;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
