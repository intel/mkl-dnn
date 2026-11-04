/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#include <cassert>
#include <cstddef>
#include <functional>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose_msg.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_uni_batch_normalization.hpp"

#define IDX(a) static_cast<uint32_t>((a).getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace memory_tracking::names;

using namespace Xbyak_aarch64;
namespace barrier = simple_barrier;

using acc_data_t = float;

namespace {
dim_t get_c_padded(const batch_normalization_pd_t *pd) {
    return pd->src_md()->padded_dims[1];
}

bool is_nspc(const memory_desc_wrapper &d) {
    using namespace format_tag;
    const bool is_nspc = d.matches_one_of_tag(nc, nwc, nhwc, ndhwc);
    return is_nspc;
}
} // namespace

struct jit_bnorm_conf_t {
    // TODO: put all needed info here to avoid duplicate work and potentially
    // diverging definitions of derived parameters
    const batch_normalization_pd_t *pd_;

    int simd_w_ {0};
    size_t dt_size_ {0};
    bool is_nspc_ {false};

    // thread partition info
    bool do_blocking_ {false};
    bool is_spatial_thr_ {false};
    dim_t C_blks_per_iter_ {0};
    int C_nthr_ {0};
    int N_nthr_ {0};
    int S_nthr_ {0};
    int64_t iters_ {0};
    // C_blks and thread partition can change for last iteration
    dim_t C_blks_last_iter_ {0};
    int C_nthr_last_iter_ {0};
    int N_nthr_last_iter_ {0};
    int S_nthr_last_iter_ {0};

    jit_bnorm_conf_t(const batch_normalization_pd_t *pd, int nthr, int simd_w)
        : pd_(pd)
        , simd_w_(simd_w)
        , dt_size_(types::data_type_size(pd_->src_md()->data_type)) {

        const dim_t N = pd_->MB();
        const dim_t C_PADDED = get_c_padded(pd_);
        const dim_t D = pd_->D();
        const dim_t H = pd_->H();
        const dim_t W = pd_->W();
        const dim_t SP = D * H * W;

        const memory_desc_wrapper src_d(pd_->src_md());
        is_nspc_ = is_nspc(src_d);

        size_t data_size = dt_size_ * N * C_PADDED * SP;
        const size_t l3_size = platform::get_per_core_cache_size(3) * nthr;
        // TODO: cache balancing for nspc
        const size_t l3_filling_factor = 4;
        do_blocking_ = !is_nspc_ && data_size >= l3_size / l3_filling_factor;

        // find thread partition over N, C_blks and SP

        const dim_t C_blks = C_PADDED / simd_w_;

        if (do_blocking_) {
            const int num_tensors = pd_->is_fwd() ? 1 : 2;
            const size_t working_set_size
                    = dt_size_ * (N * SP * simd_w_) * num_tensors;
            bnorm_utils::cache_balance(working_set_size, C_blks, N, nthr,
                    C_blks_per_iter_, iters_);
            C_blks_last_iter_ = C_blks - (iters_ - 1) * C_blks_per_iter_;
        } else {
            C_blks_per_iter_ = C_blks;
            iters_ = 1;
        }

        is_spatial_thr_ = thread_partition(/* is_spatial_thr_ = */ true, nthr,
                /* dimensions */
                N, C_blks_per_iter_, SP,
                /* outputs */
                C_nthr_, N_nthr_, S_nthr_);

        if (iters_ > 1)
            thread_partition(is_spatial_thr_, nthr,
                    /* dimensions */
                    N, C_blks_last_iter_, SP,
                    /* outputs */
                    C_nthr_last_iter_, N_nthr_last_iter_, S_nthr_last_iter_);
    }

    // given nthr and shape of problem, choose the thread partition
    // to use (ie set N_nthr, C_nthr, and S_nthr)
    bool thread_partition(bool spatial_thr_allowed, int nthr, dim_t N,
            dim_t C_blks, dim_t SP, int &C_nthr, int &N_nthr,
            int &S_nthr) const {
        // For SVE128 NSPC with MB=1 and SP >= 225, force spatial threading
        // instead of channel threading (currently yields better performance
        // on certain bnorm shapes)
        const bool force_sve128_spatial
                = is_nspc_ && simd_w_ < 8 && N == 1 && SP >= 225;

        if ((((nthr <= C_blks) && IMPLICATION(is_nspc_, N == 1)
                     && !force_sve128_spatial)
                    || !dnnl_thr_syncable())) {
            C_nthr = nthr;
            N_nthr = 1;
            S_nthr = 1;
        } else {
            if (is_nspc_) {
                const int min_c_blks_per_thr = nstl::max(1, 16 / simd_w_);
                if (C_blks <= 8) {
                    C_nthr = 1;
                } else if (simd_w_ < 16 && C_blks % min_c_blks_per_thr == 0) {
                    // Keep each channel thread on a 16-channel boundary.
                    const dim_t C_chunks = C_blks / min_c_blks_per_thr;
                    C_nthr = (int)nstl::min<dim_t>(
                            nstl::min<dim_t>(8, nthr), C_chunks);
                    while (C_nthr > 1 && C_chunks % C_nthr != 0)
                        --C_nthr;
                } else if (nthr >= 8 && C_blks <= 32) {
                    C_nthr = 8;
                } else if (simd_w_ < 8) {
                    // For SVE128 FP32 (simd_w_=4), keep channel threading disabled.
                    // The plain-layout path still assumes 16-channel grouping:
                    // plain layouts require padded C % 16 == 0, and channel partitioning
                    // still assigns work on 16-channel boundaries rather than 4-channel SVE128 lanes.
                    // This has caused some performance regressions (in cases like mb1ic160ih29iw29)
                    C_nthr = 1;
                } else {
                    C_nthr = (int)math::gcd((dim_t)nthr, C_blks);
                    // Unroll by channels in JIT kernel
                    if ((C_nthr == C_blks) || (C_nthr == nthr)) C_nthr = 1;
                }
                if (force_sve128_spatial) C_nthr = 1;
                N_nthr = (int)nstl::min<dim_t>(N, nthr / C_nthr);
                S_nthr = (int)nstl::min<dim_t>(SP, nthr / (C_nthr * N_nthr));
            } else {
                if (do_blocking_) {
                    N_nthr = (int)nstl::min<dim_t>(N, nthr);
                    C_nthr = (int)nstl::min<dim_t>(C_blks, nthr / N_nthr);
                    S_nthr = (int)nstl::min<dim_t>(
                            SP, nthr / (C_nthr * N_nthr));
                } else {
                    C_nthr = (int)math::gcd((dim_t)nthr, C_blks);
                    N_nthr = (int)nstl::min<dim_t>(N, nthr / C_nthr);
                    S_nthr = (int)nstl::min<dim_t>(
                            SP, nthr / (C_nthr * N_nthr));
                }
            }

            if (!spatial_thr_allowed) S_nthr = 1;

            if (S_nthr < 1) S_nthr = 1;
        }

        // spatial_thr_allowed is meant to help maintain
        // consistent decisions about spatial threading
        // between mutiple invocations of this routine.
        // It is caller's responsibility to check the
        // return value and pass it as a flag to the
        // next call if needed.
        if (S_nthr == 1) spatial_thr_allowed = false;

        return spatial_thr_allowed;
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_t : public jit_generator_t {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size, spat_size_loc;
        size_t S_s, S_tail;
        size_t is_cblk_tail;
        acc_data_t chan_size, eps, one;
        const acc_data_t *scale;
        const acc_data_t *shift;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_scale;
        const acc_data_t *diff_shift;
        const void *src, *dst;
        const void *diff_src, *diff_dst;
        const acc_data_t *rbuf1, *rbuf2;
        const uint8_t *ws;
        barrier::ctx_64_t *barrier;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    /* cpu specific part */
    using TReg = typename utils::conditional<isa == asimd, VReg, ZReg>::type;
    using TRegS =
            typename utils::conditional<isa == asimd, VReg4S, ZRegS>::type;

    const batch_normalization_pd_t *pd_ = nullptr;
    const jit_bnorm_conf_t *jbp_ = nullptr;
    bool is_bf16_ = false;
    bool is_f16_ = false;

    const int vlen = simd_bytes(isa);
    int vlen_spat_data_; // set by ctor depending on data type (BF16 or FP32);

    XReg reg_param = abi_param1;

    XReg reg_scale = x3;
    XReg reg_rbuf1 = x1;
    XReg reg_rbuf2 = x2;
    XReg reg_coff_max_fwd_copy = reg_rbuf2;

    XReg reg_mean = x5;
    XReg reg_var = reg_param;
    XReg reg_diff_scale = x7;
    XReg reg_coff_max_bwd_copy = reg_diff_scale;
    XReg reg_shift = reg_rbuf1;

    XReg reg_coff = x8;
    XReg reg_coff_max = x9;
    XReg reg_soff = x10;
    XReg reg_soff_max = x11;
    XReg reg_diff_shift = reg_soff_max;
    XReg reg_ctr = x12;
    XReg reg_roff = x13;

    XReg reg_mb_stride_Bc = x14;
    XReg reg_soff_nspc = reg_mb_stride_Bc;

    XReg reg_src = x15;
    XReg reg_diff_src = reg_rbuf1;
    XReg reg_dst = x6;
    XReg reg_diff_dst = reg_dst;

    XReg reg_tmp_off = reg_roff;

    // Reuse loop counters
    XReg reg_bar = reg_coff;
    XReg reg_nnthr = reg_soff; // must be usable w/ loops over coff
    XReg reg_tmp = reg_ctr;

    // Relu section
    bool with_relu = false, with_relu_inf_only = false;
    XReg reg_ws = reg_roff;
    PReg kstore_mask = PReg(1);

    // channel tail processing
    PReg ktail_mask = PReg(2);

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_tmp0 = p4;

    size_t unroll_blocks;
    size_t unroll_regs;
    // Index 0 is temporal value. is_fwd() ? vdiff_beta : vbeta
    TReg vzero = TReg(0);
    TReg vacc = TReg(1);
    TReg vdiff_beta = TReg(21);
    TReg vdiff_gamma = TReg(22);
    TReg vsqrtvar = TReg(23);
    TReg vone = TReg(24);
    TReg vmean = TReg(25);
    TReg vgamma = TReg(26);
    TReg vbeta = TReg(27);
    TReg veps = TReg(28);
    TReg vchan_size = TReg(29);
    TReg vtail_mask = TReg(30);
    TReg t_tmp0 = TReg(31);
    TReg t_tmp1 = TReg(20);

    size_t spat_size;
    size_t chan_data_offt;
    size_t spat_step;
    size_t mb_offt;
    size_t ws_mb_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_diff_scale = 48,
        stack_off_ws = 56,
        stack_off_barrier = 64,
        stack_off_spat_size_loc = 72,
        stack_off_s_s = 80,
        stack_off_s_tail = 88,
        stack_off_is_cblk_tail = 96,
        stack_off_ws_off_copy = 104,
        stack_off_shift = 112,
        stack_off_diff_shift = 120,
        stack_off_soff_max = 128,
        stack_off_relu_alpha = 136,
        stack_size_required = 144,
    };

    bool is_xf16() { return is_bf16_ || is_f16_; }
    int bit_shift() { return 5 - is_xf16(); }

    bool stream_store_supported() {
        // keep original behavior for f32
        if (!is_xf16()) return true;
        return false;
    }

    bool is_c_padded() const {
        const memory_desc_wrapper data_d(pd_->src_md());
        return pd_->C() != data_d.padded_dims()[1];
    }

    void compute_static_strides() {
        spat_size = pd_->D() * pd_->W() * pd_->H();
        chan_data_offt = pd_->C() * sizeof(acc_data_t);
        spat_step = jbp_->is_nspc_ ? chan_data_offt / (1 + is_xf16())
                                   : vlen_spat_data_;
        mb_offt = spat_step * spat_size;
        ws_mb_offt = (spat_step / (is_xf16() ? 16 : 32)) * spat_size;
    }

    void load_common_params() {
        ldr_imm(reg_rbuf1, reg_param, offsetof(call_params_t, rbuf1));
        if (!pd_->is_fwd())
            ldr_imm(reg_rbuf2, reg_param, offsetof(call_params_t, rbuf2));

        ldr_imm(reg_coff_max, reg_param, offsetof(call_params_t, coff_max));
        ldr_imm(reg_soff_max, reg_param, offsetof(call_params_t, soff_max));
        ldr_imm(reg_mb_stride_Bc, reg_param,
                offsetof(call_params_t, mb_stride_Bc));
        lsl(reg_coff_max, reg_coff_max, 2);

        ldr_imm(reg_mean, reg_param, offsetof(call_params_t, mean));
        ldr_imm(reg_scale, reg_param, offsetof(call_params_t, scale));

        ldr_imm(W_TMP_1, reg_param, offsetof(call_params_t, chan_size));
        ldr_imm(W_TMP_2, reg_param, offsetof(call_params_t, one));
        ldr_imm(W_TMP_3, reg_param, offsetof(call_params_t, eps));

        dup(vchan_size.s, W_TMP_1);
        dup(vone.s, W_TMP_2);
        dup(veps.s, W_TMP_3);

        mov(X_DEFAULT_ADDR, sp);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, N_nthr));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_N_nthr);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, N_ithr));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_N_ithr);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, src));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_src);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, dst));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_dst);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, diff_src));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_diff_src);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, diff_dst));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_diff_dst);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, ws));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_ws);
        ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, barrier));
        str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_barrier);
        if (jbp_->is_spatial_thr_) {
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, spat_size_loc));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_spat_size_loc);
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, S_s));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_s_s);
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, S_tail));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_s_tail);
        }
        if (is_c_padded()) {
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, is_cblk_tail));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_is_cblk_tail);
        }

        if (pd_->is_fwd()) {
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, shift));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_shift);
        } else {
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, diff_scale));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_diff_scale);
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, diff_shift));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_diff_shift);
            ldr_imm(X_TMP_0, reg_param, offsetof(call_params_t, soff_max));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_soff_max);
        }
        ldr_imm(reg_var, reg_param, offsetof(call_params_t, var));
        if (with_relu_inf_only && pd_->alpha() != 0.f) {
            mov_imm(X_TMP_0, float2int(pd_->alpha()));
            str_imm(X_TMP_0, X_DEFAULT_ADDR, stack_off_relu_alpha);
        }
    }

    void prepare_tail_mask() {
        if (!is_c_padded()) return;
        const int tail = pd_->C() % (int)(vlen / sizeof(float));
        set_preg(ktail_mask.s, tail, X_TMP_0);
    }

    void prepare_relu() {
        with_relu = pd_->is_fwd() ? pd_->with_relu_post_op(pd_->is_training())
                        || pd_->fuse_norm_relu()
                                  : pd_->fuse_norm_relu();
        with_relu_inf_only = with_relu && pd_->is_fwd()
                && !(pd_->fuse_norm_relu() && pd_->is_training());

        vzero = pd_->is_fwd() ? vdiff_beta : vbeta;
        if (with_relu) uni_clear(vzero);
    }

    void fwd_process_relu(ZRegS vdst, int offt = 0) {
        const int bits = bit_shift();
        const int offset = offt / (1 << bits);
        XReg r = jbp_->is_nspc_ ? reg_soff_nspc : reg_soff;
        ZRegS zzero = ZRegS(vzero.getIdx());

        assert(is_superset(isa, sve));

        assert(bits < 64);
        lsr(r, r, bits);
        fcmlt(kstore_mask.s, P_ALL_ONE / T_z, zzero, vdst);
        sub(X_DEFAULT_ADDR, sp, 8);
        uzp1(p_tmp0.b, kstore_mask.b, kstore_mask.b);
        uzp1(p_tmp0.b, p_tmp0.b, p_tmp0.b);
        str(p_tmp0, ptr(X_DEFAULT_ADDR));
        ldrh(W_TMP_0, ptr(X_DEFAULT_ADDR));

        add(X_DEFAULT_ADDR, reg_ws, r);
        if (offset) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offset, X_TMP_0);
        strh(W_TMP_0, ptr(X_DEFAULT_ADDR));

        sel(vdst, kstore_mask, vdst, zzero);
        lsl(r, r, bit_shift());
    }

    void fwd_process_relu_alpha(TRegS vmm_dst) {
        ZRegS dst = ZRegS(vmm_dst.getIdx());
        ZRegS z_tmp0 = ZRegS(t_tmp0.getIdx());

        assert(is_superset(isa, sve));

        add_imm(X_DEFAULT_ADDR, sp, (int)stack_off_relu_alpha, X_TMP_0);
        ld1rw(ZRegS(t_tmp0.getIdx()), P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));

        fcmge(kstore_mask.s, P_ALL_ONE / T_z, dst, 0.0);
        fmul(z_tmp0, dst, z_tmp0);
        sel(dst, kstore_mask, dst, z_tmp0);
    }

    void bwd_process_relu(ZRegS vdiff_dst, int offt = 0) {
        const int bits = bit_shift();
        const int offset = offt / (1 << bits);
        XReg r = jbp_->is_nspc_ ? reg_soff_nspc : reg_soff;

        assert(bits < 64);
        lsr(r, r, bits);

        add(X_DEFAULT_ADDR, reg_ws, r);
        if (offset) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offset, X_TMP_0);

        ldrh(W_TMP_0, ptr(X_DEFAULT_ADDR));
        sub(X_DEFAULT_ADDR, sp, 8);
        str(X_TMP_0, ptr(X_DEFAULT_ADDR));
        ldr(kstore_mask, ptr(X_DEFAULT_ADDR));
        zip1(kstore_mask.b, kstore_mask.b, kstore_mask.b);
        zip1(kstore_mask.b, kstore_mask.b, kstore_mask.b);

        movprfx(vdiff_dst, kstore_mask / T_z, vdiff_dst);
        lsl(r, r, bits);
    }

    void uni_load_spat_data(const VReg &v, const XReg &x) {
        ldr(QReg(IDX(v)), ptr(x));
    }

    void uni_load_spat_data(const ZReg &z, const XReg &x) {
        ld1w(z.s, P_ALL_ONE / T_z, ptr(x));
    }

    void uni_store_spat_data(
            const XReg &x, const VReg &v, bool is_nt_store = false) {
        UNUSED(is_nt_store);
        str(QReg(IDX(v)), ptr(x));
    }

    void uni_store_spat_data(
            const XReg &x, const ZReg &z, bool is_nt_store = false) {
        if (is_nt_store) {
            stnt1w(z.s, P_ALL_ONE / T_z, ptr(x));
        } else {
            st1w(z.s, P_ALL_ONE / T_z, ptr(x));
        }
    }

    void jump_check(const Label &l_no_mask) {
        ldr_imm(reg_tmp, sp, stack_off_is_cblk_tail);
        cmp(reg_tmp, 0);
        b(EQ, l_no_mask);

        add_imm(reg_tmp, reg_coff, vlen, X_TMP_1);
        cmp(reg_tmp, reg_coff_max);
        b(LT, l_no_mask);
    }

    void uni_load_maybe_tail(const TReg &t, const XReg &x) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            jump_check(l_no_mask);
            assert(is_superset(isa, sve));
            ld1w(ZRegS(IDX(t)), ktail_mask / T_z, ptr(x));
            b(l_ret);
        }
        L(l_no_mask);
        uni_ldr(t, x);
        L(l_ret);
    }

    void uni_store_maybe_tail(const XReg &x, const TReg &t) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            jump_check(l_no_mask);
            assert(is_superset(isa, sve));
            st1w(ZRegS(IDX(t)), ktail_mask / T_z, ptr(x));
            b(l_ret);
        }
        L(l_no_mask);
        uni_str(t, x);
        L(l_ret);
    }

    void uni_fmls(const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
        fmls(dst, src, src2);
    }

    void uni_fmls(const ZRegS &dst, const ZRegS &src, const ZRegS &src2) {
        fmls(dst, P_ALL_ONE / T_m, src, src2);
    }

    void uni_fmla(const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
        fmla(dst, src, src2);
    }

    void uni_fmla(const ZRegS &dst, const ZRegS &src, const ZRegS &src2) {
        fmla(dst, P_ALL_ONE / T_m, src, src2);
    }

    void uni_str(const VReg &v, const XReg &base,
            const XReg &off = XReg(DUMMY_IDX), const int disp = 0) {
        str(QReg(IDX(v)), ptr(xreg_addr(base, off, disp)));
    }

    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0) {
        XReg x_addr = base;
        uint32_t offIdx = off.getIdx();

        if (offIdx <= SP_IDX) {
            add(X_DEFAULT_ADDR, base, off);
            x_addr = X_DEFAULT_ADDR;
        }
        if (disp) {
            add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
            x_addr = X_DEFAULT_ADDR;
        }

        return x_addr;
    }

    void uni_str(const ZReg &z, const XReg &base,
            const XReg &off = XReg(DUMMY_IDX), const int disp = 0) {
        st1w(z.s, P_ALL_ONE / T_z, ptr(xreg_addr(base, off, disp)));
    }

    void uni_stnt1w(const ZReg &z, const XReg &base,
            const XReg &off = XReg(DUMMY_IDX), const int disp = 0) {
        stnt1w(z.s, P_ALL_ONE, ptr(xreg_addr(base, off, disp)));
    }

    void uni_fmax(const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
        fmaxnm(dst, src, src2);
        fmax(dst, dst, src2);
    }

    void uni_fmax(const ZRegS &dst, const ZRegS &src, const ZRegS &src2) {
        mov(t_tmp0.s, P_ALL_ONE / T_m, src2);
        fmaxnm(t_tmp0.s, P_ALL_ONE, src);
        fmax(t_tmp0.s, P_ALL_ONE, src);
        mov(dst, P_ALL_ONE / T_m, t_tmp0.s);
    }

    void barrier() {
        ldr_imm(reg_nnthr, sp, stack_off_N_nthr);
        ldr_imm(reg_bar, sp, stack_off_barrier);
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    XReg mean_ptr(size_t offt = 0) {
        return xreg_addr(reg_mean, reg_coff, offt);
    }

    XReg var_ptr(size_t offt = 0) { return xreg_addr(reg_var, reg_coff, offt); }

    XReg diff_gamma_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_scale, reg_coff, offt);
    }

    XReg diff_beta_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_shift, reg_coff, offt);
    }

    XReg gamma_ptr(size_t offt = 0) {
        return xreg_addr(reg_scale, reg_coff, offt);
    }

    XReg beta_ptr(size_t offt = 0) {
        return xreg_addr(reg_shift, reg_coff, offt);
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs, init_t init,
            body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;
        for (size_t i = 0; i < num_active_regs; i++)
            init(i);
        if (loop_unroll) {
            if (jbp_->is_spatial_thr_) {
                ldr_imm(reg_ctr, sp, stack_off_spat_size_loc);
                ldr_imm(X_TMP_0, sp, stack_off_s_s);
                add(reg_soff, reg_soff, X_TMP_0);
            } else {
                mov_imm(reg_ctr, (int)loop_unroll);
            }
            Label label;
            L(label);
            {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }
                add_imm(reg_soff, reg_soff, (int)factor * spat_step, X_TMP_0);
                subs_imm(reg_ctr, reg_ctr, (int)factor, X_TMP_0);
                b(NE, label);
            }
            if (jbp_->is_spatial_thr_) {
                ldr_imm(X_TMP_0, sp, stack_off_s_tail);
                add(reg_soff, reg_soff, X_TMP_0);
            }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail)
            add_imm(reg_soff, reg_soff, (int)loop_tail * spat_step, X_TMP_0);

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label);
        {
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_ldr(TReg(0), X_TMP_0);
            spat_loop(spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                TReg v = TReg(base_reg * 2);
                if (base_reg) uni_clear(v);
            }, [=](size_t base_reg, size_t i) {
                TRegS v0 = TRegS(base_reg * 2 + 0);
                TReg v1 = TReg(base_reg * 2 + 1);
                size_t offt = i * vlen_spat_data_;
                add(X_TMP_0, reg_src, reg_soff);
                if (offt) add_imm(X_TMP_0, X_TMP_0, offt, X_TMP_1);
                uni_load_spat_data(v1, X_TMP_0);
                fadd(v0, v0, v1.s);
            }, [=](size_t base_reg) {
                TRegS b = TRegS(0);
                TRegS v = TRegS(base_reg * 2);
                if (base_reg) fadd(b, b, v);
            });
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_str(TReg(0), X_TMP_0);

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(LT, ch_label);
        }
    }

    void mean_variance_nspc(
            const int num_ch_blks, int num_spat_pts, bool compute_mean) {

        auto mean_compute = [=](int num_ch_blks, int num_spat_pts) {
            const TReg vsrc = t_tmp0;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                add(X_TMP_0, reg_src, reg_soff_nspc);
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    if (ch_idx)
                        add_imm(X_TMP_0, X_TMP_0, vlen_spat_data_, X_TMP_1);
                    uni_load_spat_data(vsrc, X_TMP_0);
                    fadd(TRegS(ch_idx), TRegS(ch_idx), vsrc.s);
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, (int)spat_step, X_TMP_0);
            }
        };

        auto variance_compute = [=](int num_ch_blks, int num_spat_pts) {
            const TRegS vsrc = t_tmp0.s;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                add(X_TMP_0, reg_src, reg_soff_nspc);
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    const TRegS vmean_ch = TRegS(ch_idx + num_ch_blks);
                    if (ch_idx)
                        add_imm(X_TMP_0, X_TMP_0, vlen_spat_data_, X_TMP_1);
                    uni_load_spat_data(TReg(vsrc.getIdx()), X_TMP_0);
                    uni_fsub(vsrc, vsrc, vmean_ch);
                    uni_fmla(TRegS(ch_idx), vsrc, vsrc);
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, (int)spat_step, X_TMP_0);
            }
        };

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int coff = idx * vlen;
            add(X_TMP_0, reg_rbuf1, reg_coff);
            if (coff) add_imm(X_TMP_0, X_TMP_0, coff, X_TMP_1);
            uni_ldr(TReg(idx), X_TMP_0);
            if (!compute_mean) {
                // pre-load mean to avoid extra data movement during variance
                const TReg vmean_ch = TReg(idx + num_ch_blks);
                uni_load_maybe_tail(vmean_ch, mean_ptr(coff));
            }
        }

        eor(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

        if (jbp_->is_spatial_thr_) {
            ldr_imm(reg_ctr, sp, stack_off_spat_size_loc);
            ldr_imm(X_TMP_0, sp, stack_off_s_s);
            add(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
            // TODO: need a better heuristic for num_spat_pts
            num_spat_pts = 1;
        } else {
            mov_imm(reg_ctr, (int)spat_size);
            num_spat_pts = nstl::min((size_t)num_spat_pts, spat_size);
            // TODO: unroll by spatial
            if (spat_size % num_spat_pts != 0) num_spat_pts = 1;
        }

        Label spatial;
        L(spatial);
        {
            compute_mean ? mean_compute(num_ch_blks, num_spat_pts)
                         : variance_compute(num_ch_blks, num_spat_pts);
            subs_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
            b(NE, spatial);
        }

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int coff = idx * vlen;
            add(X_TMP_0, reg_rbuf1, reg_coff);
            if (coff) add_imm(X_TMP_0, X_TMP_0, coff, X_TMP_1);
            uni_str(TReg(idx), X_TMP_0);
        }
    }

    void forward_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            // Overwritten during mean and variance computation
            uni_clear(vzero);

            eor(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

            if (jbp_->is_spatial_thr_) {
                ldr_imm(reg_ctr, sp, stack_off_spat_size_loc);
                ldr_imm(X_TMP_0, sp, stack_off_s_s);
                add(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
            } else {
                mov_imm(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            // pre-compute scale for each channel to avoid costly div and sqrt
            for (int idx = 0; idx < num_ch_blks; ++idx) {
                const int coff = idx * vlen;
                const TRegS vscale = TRegS(idx + num_ch_blks);
                uni_load_maybe_tail(vsqrtvar, var_ptr(coff));
                fadd(vsqrtvar.s, vsqrtvar.s, veps.s);
                uni_fsqrt(vsqrtvar.s, vsqrtvar.s);

                if (pd_->use_scale()) {
                    uni_load_maybe_tail(vgamma, gamma_ptr(coff));
                    uni_fdiv(vscale, vgamma.s, vsqrtvar.s, t_tmp0.s, P_ALL_ONE);
                } else {
                    uni_fdiv(vscale, vone.s, vsqrtvar.s, t_tmp0.s, P_ALL_ONE);
                }
            }

            Label spatial;
            L(spatial);
            {
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    const int coff = idx * vlen;
                    const int offt = idx * vlen_spat_data_;
                    const TRegS vdata = TRegS(idx);
                    const TRegS vscale = TRegS(idx + num_ch_blks);
                    uni_load_maybe_tail(vmean, mean_ptr(coff));

                    if (pd_->use_shift()) {
                        uni_load_maybe_tail(vbeta, beta_ptr(coff));
                    }

                    add(X_DEFAULT_ADDR, reg_src, reg_soff_nspc);
                    if (offt)
                        add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                    uni_load_spat_data(TReg(vdata.getIdx()), X_DEFAULT_ADDR);

                    uni_fsub(vdata, vdata, vmean.s);

                    if (pd_->use_shift()) {
                        // --flags=S,CH,H
                        uni_fmad(vdata, vscale, vbeta.s);
                    } else {
                        // --flags=,C
                        fmul(vdata, vdata, vscale);
                    }

                    if (with_relu_inf_only) { // --attr=post_ops='relu'
                        if (pd_->alpha() != 0.f)
                            fwd_process_relu_alpha(vdata);
                        else
                            uni_fmaxnm(vdata, vdata, vzero.s);
                    } else if (with_relu) { // --flags=R
                        assert(is_superset(isa, sve));
                        fwd_process_relu(
                                ZRegS(vdata.getIdx()), idx * vlen_spat_data_);
                    }
                    add(X_DEFAULT_ADDR, reg_dst, reg_soff_nspc);
                    if (offt)
                        add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                    uni_store_spat_data(X_DEFAULT_ADDR, TReg(vdata.getIdx()),
                            stream_store_allowed);
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
                subs_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
                b(NE, spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            assert(vlen_spat_data_ - 1 < 2048);
            cmp(reg_dst, vlen_spat_data_ - 1);
            b(NE, normal_store);
            compute(true);
            b(end_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // disabled for bf16 when data fits in cache
        }
    }

    void compute_mean_variance_nspc(bool compute_mean = true) {
        eor(reg_coff, reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll, sp_idx = 1; ch_idx > 0;
                --ch_idx, ++sp_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                assert(vlen * ch_blk_size < 1024);
                cmp(reg_coff_max, vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                const int spat_blk_size = (1 << sp_idx);
                mean_variance_nspc(ch_blk_size, spat_blk_size, compute_mean);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size,
                        X_TMP_0);
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size,
                        X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_xf16()) lsr(reg_coff_max, reg_coff_max, 1);
        sub(reg_src, reg_src, reg_coff_max);
        if (is_xf16()) lsl(reg_coff_max, reg_coff_max, 1);
    }

    void var_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_ldr(TReg(0), X_TMP_0);
            spat_loop(spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                TReg v = TReg(3 * base_reg);
                if (base_reg > 0) uni_clear(v);
            }, [=](size_t base_reg, size_t i) {
                TRegS v = TRegS(3 * base_reg);
                TRegS vtmp0 = TRegS(3 * base_reg + 1);
                TRegS vtmp1 = TRegS(3 * base_reg + 2);
                size_t offt = i * vlen_spat_data_;
                add(X_TMP_0, reg_src, reg_soff);
                if (offt) add_imm(X_TMP_0, X_TMP_0, offt, X_TMP_1);
                uni_load_spat_data(TReg(IDX(vtmp0)), X_TMP_0);
                fsub(vtmp1, vmean.s, vtmp0);
                uni_fmla(v, vtmp1, vtmp1);
            }, [=](size_t base_reg) {
                TRegS b = TRegS(0);
                TRegS v = TRegS(base_reg * 3);
                if (base_reg) fadd(b, b, v);
            });
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_str(TReg(0), X_TMP_0);
            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(LT, ch_label);
        }
    }

    void compute_mean_variance() {
        uni_eor(TReg(0), TReg(0), TReg(0));
        eor(reg_coff, reg_coff, reg_coff);
        Label zero_rbuf;
        L(zero_rbuf);
        {
            uni_str(TReg(0), reg_rbuf1, reg_coff);
            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(NE, zero_rbuf);
        }

        ldr_imm(reg_src, sp, stack_off_src);

        eor(reg_soff, reg_soff, reg_soff);
        Label mean_spatial;
        L(mean_spatial);
        {
            eor(reg_coff, reg_coff, reg_coff);

            jbp_->is_nspc_ ? compute_mean_variance_nspc() : mean_channels();

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                    add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                }
            } else {
                add(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            b(LT, mean_spatial);
        }

        if (jbp_->is_nspc_) {
            ldr_imm(reg_src, sp, stack_off_src); // comeback
        }

        Label no_mean_reduction;
        barrier();
        {
            assert(stack_off_N_ithr < 256);
            ldr(reg_tmp, ptr(sp, (int)stack_off_N_ithr));
            cmp(reg_tmp, 0);
            b(NE, no_mean_reduction);
            assert(stack_off_N_nthr < 256);
            ldr(reg_nnthr, ptr(sp, (int)stack_off_N_nthr));
            eor(reg_coff, reg_coff, reg_coff);
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_eor(TReg(0), TReg(0), TReg(0));
                uni_clear(vacc);
                mov(reg_ctr, reg_nnthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs);
                {
                    add(X_TMP_0, reg_rbuf1, reg_roff);
                    uni_ldr(t_tmp0, X_TMP_0);
                    fadd(vacc.s, vacc.s, t_tmp0.s);

                    uni_str(TReg(0), X_TMP_0);
                    add(reg_roff, reg_roff, reg_coff_max);
                    subs_imm(reg_ctr, reg_ctr, 1, X_TMP_0);
                    b(NE, mean_reduction_thrs);
                }

                uni_fdiv(vacc.s, vacc.s, TRegS(vchan_size.getIdx()), t_tmp0.s,
                        P_ALL_ONE);

                uni_store_maybe_tail(mean_ptr(), vacc);

                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);

                cmp(reg_coff, reg_coff_max);
                b(LT, mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        eor(reg_soff, reg_soff, reg_soff);
        Label var_spatial;
        L(var_spatial);
        {
            eor(reg_coff, reg_coff, reg_coff);

            jbp_->is_nspc_ ? compute_mean_variance_nspc(false) : var_channels();

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                    add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                }
            } else {
                add(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            b(LT, var_spatial);
        }

        if (jbp_->is_nspc_) {
            assert(stack_off_src < 256);
            ldr(reg_src, ptr(sp, (int)stack_off_src)); // comeback
        }

        Label no_var_reduction;
        barrier();
        {
            ldr_imm(reg_tmp, sp, stack_off_N_ithr);
            cmp(reg_tmp, 0);
            b(NE, no_var_reduction);

            ldr_imm(reg_nnthr, sp, stack_off_N_nthr);
            eor(reg_coff, reg_coff, reg_coff);
            Label var_reduction_channels;
            L(var_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_clear(vacc);
                mov(reg_ctr, reg_nnthr);
                Label var_reduction_thrs;
                L(var_reduction_thrs);
                { // TODO: unroll (?)
                    add(X_TMP_0, reg_rbuf1, reg_roff);
                    uni_ldr(t_tmp0, X_TMP_0);
                    fadd(vacc.s, vacc.s, t_tmp0.s);
                    add(reg_roff, reg_roff, reg_coff_max);
                    subs(reg_ctr, reg_ctr, 1);
                    b(NE, var_reduction_thrs);
                }

                uni_fdiv(vacc.s, vacc.s, TRegS(vchan_size.getIdx()), t_tmp0.s,
                        P_ALL_ONE);

                uni_store_maybe_tail(var_ptr(), vacc);

                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);

                cmp(reg_coff, reg_coff_max);
                b(NE, var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            uni_load_maybe_tail(vsqrtvar, var_ptr());
            fadd(vsqrtvar.s, vsqrtvar.s, veps.s);
            uni_fsqrt(vsqrtvar.s, vsqrtvar.s);

            if (pd_->use_scale()) { uni_load_maybe_tail(vgamma, gamma_ptr()); }
            if (pd_->use_shift()) { uni_load_maybe_tail(vbeta, beta_ptr()); }

            TReg vscale = (pd_->use_scale()) ? vgamma : vone;
            TReg vdiv = (pd_->use_scale()) ? vgamma : vsqrtvar;

            uni_fdiv(vdiv.s, vscale.s, vsqrtvar.s, t_tmp0.s, P_ALL_ONE);

            const auto spat_loop_init_fin
                    = [](size_t base_reg) { UNUSED(base_reg); };

            const auto spat_loop_body = [=](size_t base_reg, size_t i,
                                                bool stream_store_allowed) {
                const TRegS v = TRegS(base_reg);
                const size_t offt = i * vlen_spat_data_;
                add(X_DEFAULT_ADDR, reg_src, reg_soff);
                add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                uni_load_spat_data(TReg(v.getIdx()), X_DEFAULT_ADDR);
                fsub(v, v, vmean.s);
                if ((pd_->use_scale() && pd_->use_shift())) {
                    // --flags=CH
                    uni_fmad(v, vgamma.s, vbeta.s);
                } else if (pd_->use_scale()) {
                    // --flags=C
                    fmul(v, v, vgamma.s);
                } else if (pd_->use_shift()) {
                    // --flags=H
                    uni_fmad(v, vsqrtvar.s, vbeta.s);
                } else {
                    fmul(v, v, vsqrtvar.s);
                }
                if (with_relu_inf_only) { // --attr=post_ops='relu'
                    if (pd_->alpha() != 0.f) {
                        fwd_process_relu_alpha(v);
                    } else
                        uni_fmaxnm(v, v, vzero.s);
                } else if (with_relu) { // --flags=R
                    assert(is_superset(isa, sve));
                    fwd_process_relu(ZRegS(v.getIdx()), offt);
                }
                add(X_DEFAULT_ADDR, reg_dst, reg_soff);
                if (offt)
                    add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                if (stream_store_allowed) {
                    uni_str(ZReg(v.getIdx()), X_DEFAULT_ADDR);
                } else {
                    uni_store_spat_data(X_DEFAULT_ADDR, TReg(v.getIdx()));
                }
            };

            const auto compute = [=](bool stream_store_allowed) {
                using namespace std::placeholders;
                spat_loop(spat_size, unroll_blocks, unroll_regs,
                        spat_loop_init_fin,
                        std::bind(spat_loop_body, _1, _2, stream_store_allowed),
                        spat_loop_init_fin);
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                assert(vlen - 1 < 2048);
                cmp(reg_dst, vlen - 1);
                b(NE, normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add(reg_coff, reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            b(LT, ch_label);
        }
    }

    void forward_channels_nspc() {
        eor(reg_coff, reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = is_bf16_ ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                assert(vlen * ch_blk_size < 2048);
                cmp(reg_coff_max, vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                forward_channels_nspc_compute(ch_blk_size);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size,
                        X_TMP_0);
                add_imm(reg_dst, reg_dst, vlen_spat_data_ * ch_blk_size,
                        X_TMP_0);

                // advance mean_ptr() and var_ptr()
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);

                add_imm(reg_ws, reg_ws, 2 * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size,
                        X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_xf16()) lsr(reg_coff_max, reg_coff_max, 1);
        sub(reg_src, reg_src, reg_coff_max);
        sub(reg_dst, reg_dst, reg_coff_max);
        if (is_xf16()) lsl(reg_coff_max, reg_coff_max, 1);

        lsr(reg_coff_max, reg_coff_max, 5 % 64);
        sub(reg_ws, reg_ws, reg_coff_max);
        lsl(reg_coff_max, reg_coff_max, 5);
    }

    void forward() {
        ldr_imm(reg_src, sp, stack_off_src);
        ldr_imm(reg_dst, sp, stack_off_dst);
        ldr_imm(reg_ws, sp, stack_off_ws);
        ldr_imm(reg_shift, sp, stack_off_shift);

        eor(reg_soff, reg_soff, reg_soff);
        Label dst_spatial;
        L(dst_spatial);
        {
            eor(reg_coff, reg_coff, reg_coff);

            jbp_->is_nspc_ ? forward_channels_nspc() : forward_channels();

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                    add_imm(reg_dst, reg_dst, mb_offt, X_TMP_0);
                    add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                }
                if (ws_mb_offt) {
                    add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
                }
            } else {
                add(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            b(LT, dst_spatial);
        }

        if (jbp_->is_nspc_) {
            // comeback
            ldr_imm(reg_src, sp, stack_off_src);
            ldr_imm(reg_dst, sp, stack_off_dst);
            ldr_imm(reg_ws, sp, stack_off_ws);
        }
    }

    void backward_sh_channels() {
        Label sh_channels;
        L(sh_channels);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_ldr(TReg(0), X_TMP_0);
            add(X_TMP_0, reg_rbuf2, reg_coff);
            uni_ldr(TReg(1), X_TMP_0);
            spat_loop(spat_size, 1, 1, [=](size_t base_reg) {
                if (base_reg > 0) {
                    for (int i = 0; i < 2; i++) {
                        TReg v(base_reg * 5 + i);
                        uni_clear(v);
                    }
                }
            }, [=](size_t base_reg, size_t i) {
                TReg o0 = TReg(base_reg * 5 + 0);
                TReg o1 = TReg(base_reg * 5 + 1);
                TReg t1 = TReg(base_reg * 5 + 2);
                TReg t2 = TReg(base_reg * 5 + 3);
                TReg t3 = TReg(base_reg * 5 + 4);
                size_t offt = i * vlen_spat_data_;
                add(X_TMP_0, reg_src, reg_soff);
                if (offt) add_imm(X_TMP_0, X_TMP_0, offt, X_TMP_1);
                uni_load_spat_data(t1, X_TMP_0);
                add(X_TMP_0, reg_diff_dst, reg_soff);
                if (offt) add_imm(X_TMP_0, X_TMP_0, offt, X_TMP_1);
                uni_load_spat_data(t2, X_TMP_0);
                if (with_relu) {
                    assert(is_superset(isa, sve));
                    bwd_process_relu(ZRegS(t2.getIdx()), offt);
                }
                fsub(t3.s, vmean.s, t1.s);
                if (isa == asimd) {
                    fmul(t3.s, t3.s, t2.s);
                    uni_fsub(o0.s, o0.s, t3.s);
                } else {
                    uni_fmls(o0.s, t3.s, t2.s);
                }
                fadd(o1.s, o1.s, t2.s);
            }, [=](size_t base_reg) {
                TReg b0 = TReg(0);
                TReg b1 = TReg(1);
                if (base_reg) {
                    fadd(b0.s, b0.s, TRegS(base_reg * 5 + 0));
                    fadd(b1.s, b1.s, TRegS(base_reg * 5 + 1));
                }
            });
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_str(TReg(0), X_TMP_0);
            add(X_TMP_0, reg_rbuf2, reg_coff);
            uni_str(TReg(1), X_TMP_0);
            if (vlen) add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(LT, sh_channels);
        }
    }

    void backward_sh_channels_nspc_compute(const int num_ch_blks) {
        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int offt = idx * vlen;
            const TReg vdiff_gamma_ch = TReg(idx);
            const TReg vdiff_beta_ch = TReg(idx + num_ch_blks);
            if (offt) {
                add_imm(X_TMP_0, reg_coff, offt, X_TMP_1);
                add(X_TMP_2, X_TMP_0, reg_rbuf1);
                add(X_TMP_3, X_TMP_0, reg_rbuf2);
            } else {
                add(X_TMP_2, reg_rbuf1, reg_coff);
                add(X_TMP_3, reg_rbuf2, reg_coff);
            }
            uni_ldr(vdiff_gamma_ch, X_TMP_2);
            uni_ldr(vdiff_beta_ch, X_TMP_3);
        }

        eor(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);

        if (jbp_->is_spatial_thr_) {
            ldr_imm(reg_ctr, sp, stack_off_spat_size_loc);
            ldr_imm(X_TMP_0, sp, stack_off_s_s);
            add(reg_soff_nspc, reg_soff_nspc, X_TMP_0);
        } else {
            mov_imm(reg_ctr, spat_size);
        }

        // TODO: spatial blocking
        const int num_spat_pts = 1;

        Label spatial;
        L(spatial);
        {
            for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                const int coff = ch_idx * vlen;
                const int offt = ch_idx * vlen_spat_data_;
                const TRegS vdiff_gamma_ch = TRegS(ch_idx);
                const TRegS vdiff_beta_ch = TRegS(ch_idx + num_ch_blks);
                // vdiff_beta and vdiff_gamma are free registers for nspc
                const TReg vsrc = vdiff_gamma;
                const TReg vdiff_dst = vdiff_beta;
                uni_load_maybe_tail(vmean, mean_ptr(coff));

                if (offt) {
                    add_imm(X_TMP_0, reg_soff_nspc, offt, X_TMP_1);
                    add(X_TMP_2, X_TMP_0, reg_src);
                    add(X_TMP_3, X_TMP_0, reg_diff_dst);
                } else {
                    add(X_TMP_2, reg_src, reg_soff_nspc);
                    add(X_TMP_3, reg_diff_dst, reg_soff_nspc);
                }
                uni_load_spat_data(vsrc, X_TMP_2);
                uni_load_spat_data(vdiff_dst, X_TMP_3);

                if (with_relu) {
                    assert(is_superset(isa, sve));
                    bwd_process_relu(ZRegS(vdiff_dst.getIdx()), offt);
                }

                fsub(vsrc.s, vsrc.s, vmean.s);
                uni_fmla(vdiff_gamma_ch, vsrc.s, vdiff_dst.s);
                fadd(vdiff_beta_ch, vdiff_beta_ch, vdiff_dst.s);
            }
            add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
            subs_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
            b(NE, spatial);
        }

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const TReg vdiff_gamma_ch = TReg(idx);
            const TReg vdiff_beta_ch = TReg(idx + num_ch_blks);
            const int offt = idx * vlen;
            if (offt) {
                add_imm(X_TMP_0, reg_coff, offt, X_TMP_1);
                add(X_TMP_2, X_TMP_0, reg_rbuf1);
                add(X_TMP_3, X_TMP_0, reg_rbuf2);
            } else {
                add(X_TMP_2, reg_rbuf1, reg_coff);
                add(X_TMP_3, reg_rbuf2, reg_coff);
            }
            uni_str(vdiff_gamma_ch, X_TMP_2);
            uni_str(vdiff_beta_ch, X_TMP_3);
        }
    }

    void backward_sh_channels_nspc() {
        eor(reg_coff, reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                backward_sh_channels_nspc_compute(ch_blk_size);

                add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size,
                        X_TMP_0);
                add_imm(reg_diff_dst, reg_diff_dst,
                        vlen_spat_data_ * ch_blk_size, X_TMP_0);

                // advance mean_ptr() and var_ptr()
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);

                add_imm(reg_ws, reg_ws, 2 * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size,
                        X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        ldr_imm(reg_diff_scale, sp, stack_off_diff_scale);

        if (is_xf16()) lsr(reg_coff_max, reg_coff_max, 1);
        sub(reg_src, reg_src, reg_coff_max);
        sub(reg_diff_dst, reg_diff_dst, reg_coff_max);
        if (is_xf16()) lsl(reg_coff_max, reg_coff_max, 1);

        if (with_relu) {
            lsr(reg_coff_max, reg_coff_max, 5);
            sub(reg_ws, reg_ws, reg_coff_max);
            lsl(reg_coff_max, reg_coff_max, 5);
        }
    }

    void backward_diff_channels() {
        Label diff_channels;
        L(diff_channels);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            uni_load_maybe_tail(vsqrtvar, var_ptr());
            fadd(vsqrtvar.s, vsqrtvar.s, veps.s);
            uni_fsqrt(vsqrtvar.s, vsqrtvar.s);
            uni_fdiv(vsqrtvar.s, vone.s, vsqrtvar.s, t_tmp0.s, P_ALL_ONE);
            if (pd_->use_scale()) uni_load_maybe_tail(vgamma, gamma_ptr());

            uni_load_maybe_tail(vdiff_gamma, diff_gamma_ptr());
            uni_load_maybe_tail(vdiff_beta, diff_beta_ptr());
            fmul(vdiff_gamma.s, vdiff_gamma.s, vsqrtvar.s);
            uni_fdiv(vdiff_beta.s, vdiff_beta.s, vchan_size.s, t_tmp0.s,
                    P_ALL_ONE);
            uni_fdiv(vdiff_gamma.s, vdiff_gamma.s, vchan_size.s, t_tmp0.s,
                    P_ALL_ONE);

            const auto spat_loop_init_fin
                    = [=](size_t base_reg) { UNUSED(base_reg); };
            const auto spat_loop_body = [=](size_t base_reg, size_t i,
                                                bool stream_store_allowed) {
                const TRegS v(base_reg * 2 + 0);
                const TRegS t(base_reg * 2 + 1);
                const TRegS t1(base_reg * 2 + 2);
                const size_t offt = i * vlen_spat_data_;
                add(X_DEFAULT_ADDR, reg_diff_dst, reg_soff);
                if (offt)
                    add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                uni_load_spat_data(TReg(v.getIdx()), X_DEFAULT_ADDR);
                if (with_relu) {
                    assert(is_superset(isa, sve));
                    bwd_process_relu(ZRegS(v.getIdx()), offt);
                }
                if (!pd_->use_global_stats()) {
                    fsub(v, v, vdiff_beta.s);
                    add(X_DEFAULT_ADDR, reg_src, reg_soff);
                    if (offt)
                        add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                    uni_load_spat_data(TReg(t.getIdx()), X_DEFAULT_ADDR);
                    fsub(t, vmean.s, t);
                    fmul(t, t, vdiff_gamma.s);
                    fadd(v, v, t);
                }
                fmul(v, v, vsqrtvar.s);
                if (pd_->use_scale()) { fmul(v, v, vgamma.s); }
                add(X_DEFAULT_ADDR, reg_diff_src, reg_soff);
                if (offt)
                    add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                uni_str(TReg(v.getIdx()), X_DEFAULT_ADDR);
            };

            const auto compute = [=](bool stream_store_allowed) {
                using namespace std::placeholders;
                spat_loop(spat_size, unroll_blocks, unroll_regs,
                        spat_loop_init_fin,
                        std::bind(spat_loop_body, _1, _2, stream_store_allowed),
                        spat_loop_init_fin);
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                assert(vlen - 1 < 2048);
                cmp(reg_diff_src, vlen - 1);
                b(NE, normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(LT, diff_channels);
        }
    }

    void backward_diff_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            eor(reg_soff_nspc, reg_soff_nspc, reg_soff_nspc);
            if (jbp_->is_spatial_thr_) {
                ldr_imm(reg_ctr, sp, stack_off_spat_size_loc);
                ldr_imm(reg_soff_nspc, sp, stack_off_s_s);
            } else {
                mov_imm(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            // pre-compute scale for each channel to avoid costly div and sqrt
            if (!pd_->use_global_stats()) {
                str_imm(reg_ws, sp, stack_off_ws_off_copy);
                ldr_imm(reg_ws, sp, stack_off_diff_scale);
            }
            for (int idx = 0; idx < num_ch_blks; ++idx) {
                const int coff = idx * vlen;
                const TRegS vsqrtvar_ch = TRegS(idx);
                uni_load_maybe_tail(TReg(vsqrtvar_ch.getIdx()), var_ptr(coff));
                fadd(vsqrtvar_ch, vsqrtvar_ch, veps.s);
                uni_fsqrt(vsqrtvar_ch, vsqrtvar_ch);
                uni_fdiv(vsqrtvar_ch, vone.s, vsqrtvar_ch, t_tmp0.s, P_ALL_ONE);
                if (!pd_->use_global_stats()) {
                    const TReg vdiff_beta_ch = TReg(idx + num_ch_blks);
                    const TReg vdiff_gamma_ch = TReg(idx + 2 * num_ch_blks);
                    if (coff) {
                        add_imm(X_TMP_0, reg_coff, coff, X_TMP_1);
                        add(X_TMP_2, X_TMP_0, reg_diff_shift);
                        add(X_TMP_3, X_TMP_0, reg_ws);
                    } else {
                        add(X_TMP_2, reg_diff_shift, reg_coff);
                        add(X_TMP_3, reg_ws, reg_coff);
                    }
                    uni_load_maybe_tail(vdiff_beta_ch, X_TMP_2);
                    uni_load_maybe_tail(vdiff_gamma_ch, X_TMP_3);
                    uni_fdiv(vdiff_beta_ch.s, vdiff_beta_ch.s, vchan_size.s,
                            t_tmp0.s, P_ALL_ONE);
                    fmul(vdiff_gamma_ch.s, vdiff_gamma_ch.s, vsqrtvar_ch);
                    uni_fdiv(vdiff_gamma_ch.s, vdiff_gamma_ch.s, vchan_size.s,
                            t_tmp0.s, P_ALL_ONE);
                }
            }
            if (!pd_->use_global_stats()) {
                ldr_imm(reg_ws, sp, stack_off_ws_off_copy);
            }

            Label spatial;
            L(spatial);
            {
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    const int coff = idx * vlen;
                    const int offt = idx * vlen_spat_data_;
                    // vdiff_beta and vdiff_gamma are free registers for nspc
                    const TRegS vdiff_data = vdiff_beta.s;
                    const TRegS vdata = vdiff_gamma.s;
                    const TRegS vsqrtvar_ch = TRegS(idx);
                    uni_load_maybe_tail(vmean, mean_ptr(coff));

                    if (pd_->use_scale())
                        uni_load_maybe_tail(vgamma, gamma_ptr(coff));

                    add(X_DEFAULT_ADDR, reg_diff_dst, reg_soff_nspc);
                    if (offt)
                        add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                    uni_load_spat_data(
                            TReg(vdiff_data.getIdx()), X_DEFAULT_ADDR);

                    if (with_relu) {
                        assert(is_superset(isa, sve));
                        bwd_process_relu(ZRegS(vdiff_data.getIdx()), offt);
                    }

                    if (!pd_->use_global_stats()) {
                        const TRegS vdiff_beta_ch = TRegS(idx + num_ch_blks);
                        const TRegS vdiff_gamma_ch
                                = TRegS(idx + 2 * num_ch_blks);
                        fsub(vdiff_data, vdiff_data, vdiff_beta_ch);
                        add(X_DEFAULT_ADDR, reg_src, reg_soff_nspc);
                        if (offt)
                            add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt,
                                    X_TMP_0);
                        uni_load_spat_data(
                                TReg(vdata.getIdx()), X_DEFAULT_ADDR);
                        fsub(vdata, vmean.s, vdata);
                        fmul(vdata, vdata, vdiff_gamma_ch);
                        fadd(vdiff_data, vdiff_data, vdata);
                    }

                    fmul(vdiff_data, vdiff_data, vsqrtvar_ch);

                    if (pd_->use_scale()) {
                        fmul(vdiff_data, vdiff_data, vgamma.s);
                    }

                    add(X_DEFAULT_ADDR, reg_diff_src, reg_soff_nspc);
                    if (offt)
                        add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
                    uni_store_spat_data(X_DEFAULT_ADDR,
                            TReg(vdiff_data.getIdx()), stream_store_allowed);
                }
                add_imm(reg_soff_nspc, reg_soff_nspc, spat_step, X_TMP_0);
                subs_imm(reg_ctr, reg_ctr, num_spat_pts, X_TMP_0);
                b(NE, spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            assert(vlen - 1 < 2048);
            cmp(reg_diff_src, vlen - 1);
            b(NE, normal_store);
            compute(true);
            b(end_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // disabled for bf16 when data fits in cache
        }
    }

    void backward_diff_channels_nspc() {
        eor(reg_coff, reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                backward_diff_channels_nspc_compute(ch_blk_size);

                add_imm(reg_diff_dst, reg_diff_dst,
                        vlen_spat_data_ * ch_blk_size, X_TMP_0);
                if (!pd_->use_global_stats())
                    add_imm(reg_src, reg_src, vlen_spat_data_ * ch_blk_size,
                            X_TMP_0);
                add_imm(reg_diff_src, reg_diff_src,
                        vlen_spat_data_ * ch_blk_size, X_TMP_0);

                // advance mean_ptr() and var_ptr()
                add_imm(reg_coff, reg_coff, vlen * ch_blk_size, X_TMP_0);

                add_imm(reg_ws, reg_ws, 2 * ch_blk_size, X_TMP_0);

                sub_imm(reg_coff_max, reg_coff_max, vlen * ch_blk_size,
                        X_TMP_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        ldr_imm(reg_diff_scale, sp, stack_off_diff_scale);

        if (is_xf16()) lsr(reg_coff_max, reg_coff_max, 1);
        sub(reg_diff_dst, reg_diff_dst, reg_coff_max);
        if (!pd_->use_global_stats()) sub(reg_src, reg_src, reg_coff_max);
        sub(reg_diff_src, reg_diff_src, reg_coff_max);
        if (is_xf16()) lsl(reg_coff_max, reg_coff_max, 1);

        lsr(reg_coff_max, reg_coff_max, 5);
        sub(reg_ws, reg_ws, reg_coff_max);
        lsl(reg_coff_max, reg_coff_max, 5);
    }

    void backward() {
        uni_eor(TReg(0), TReg(0), TReg(0));
        eor(reg_coff, reg_coff, reg_coff);
        Label zero_rbuf, sh_spatial;

        L(zero_rbuf);
        {
            add(X_TMP_0, reg_rbuf1, reg_coff);
            uni_str(TReg(0), X_TMP_0);
            add(X_TMP_0, reg_rbuf2, reg_coff);
            uni_str(TReg(0), X_TMP_0);
            add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
            cmp(reg_coff, reg_coff_max);
            b(NE, zero_rbuf);
        }

        ldr_imm(reg_src, sp, stack_off_src);
        ldr_imm(reg_diff_dst, sp, stack_off_diff_dst);
        if (with_relu) {
            assert(is_superset(isa, sve));
            ldr_imm(reg_ws, sp, stack_off_ws);
        }

        eor(reg_soff, reg_soff, reg_soff);
        L(sh_spatial);
        {
            eor(reg_coff, reg_coff, reg_coff);

            jbp_->is_nspc_ ? backward_sh_channels_nspc()
                           : backward_sh_channels();

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                    add_imm(reg_diff_dst, reg_diff_dst, mb_offt, X_TMP_0);
                    add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                }
                if (ws_mb_offt) {
                    add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
                }
            } else {
                add(reg_soff, reg_soff, reg_mb_stride_Bc);
            }
            cmp(reg_soff, reg_soff_max);
            b(LT, sh_spatial);
        }

        if (jbp_->is_nspc_) {
            // comeback
            ldr_imm(reg_src, sp, stack_off_src);
            ldr_imm(reg_diff_dst, sp, stack_off_diff_dst);
        }

        ldr_imm(reg_diff_scale, sp, stack_off_diff_scale);
        ldr_imm(reg_diff_shift, sp, stack_off_diff_shift);

        Label no_sh_reduction;
        barrier();
        {
            ldr_imm(reg_tmp, sp, stack_off_N_ithr);
            cmp(reg_tmp, 0);
            Label sh_reduction_channels;
            b(NE, no_sh_reduction);

            ldr_imm(reg_nnthr, sp, stack_off_N_nthr);
            eor(reg_coff, reg_coff, reg_coff);
            L(sh_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_eor(TReg(0), TReg(0), TReg(0));
                uni_eor(TReg(1), TReg(1), TReg(1));
                uni_load_maybe_tail(vsqrtvar, var_ptr());
                fadd(vsqrtvar.s, vsqrtvar.s, veps.s);
                uni_fsqrt(vsqrtvar.s, vsqrtvar.s);
                uni_fdiv(vsqrtvar.s, vone.s, vsqrtvar.s, t_tmp0.s, P_ALL_ONE);
                mov(reg_ctr, reg_nnthr);
                Label sh_reduction_thrs;
                L(sh_reduction_thrs);
                { // TODO: unroll (?)
                    add(X_TMP_0, reg_rbuf1, reg_roff);
                    add(X_TMP_1, reg_rbuf2, reg_roff);
                    uni_ldr(t_tmp0, X_TMP_0);
                    uni_ldr(t_tmp1, X_TMP_1);
                    fadd(TRegS(0), TRegS(0), t_tmp0.s);
                    fadd(TRegS(1), TRegS(1), t_tmp1.s);
                    add(reg_roff, reg_roff, reg_coff_max);
                    subs(reg_ctr, reg_ctr, 1);
                    b(NE, sh_reduction_thrs);
                }
                fmul(TRegS(0), TRegS(0), vsqrtvar.s);
                uni_store_maybe_tail(diff_gamma_ptr(), TReg(0));
                uni_store_maybe_tail(diff_beta_ptr(), TReg(1));
                add_imm(reg_coff, reg_coff, vlen, X_TMP_0);
                cmp(reg_coff, reg_coff_max);
                b(NE, sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        ldr_imm(reg_diff_src, sp, stack_off_diff_src);
        if (with_relu) {
            assert(is_superset(isa, sve));
            ldr_imm(reg_ws, sp, stack_off_ws);
        }

        eor(reg_soff, reg_soff, reg_soff);
        Label diff_spatial;
        L(diff_spatial);
        {
            eor(reg_coff, reg_coff, reg_coff);
            // diff_shift is shared with soff_max.
            ldr_imm(reg_diff_shift, sp, stack_off_diff_shift);

            jbp_->is_nspc_ ? backward_diff_channels_nspc()
                           : backward_diff_channels();

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    if (!pd_->use_global_stats())
                        add_imm(reg_src, reg_src, mb_offt, X_TMP_0);
                    add_imm(reg_diff_dst, reg_diff_dst, mb_offt, X_TMP_0);
                    add_imm(reg_diff_src, reg_diff_src, mb_offt, X_TMP_0);
                    add_imm(reg_soff, reg_soff, mb_offt, X_TMP_0);
                }
                if (ws_mb_offt) add_imm(reg_ws, reg_ws, ws_mb_offt, X_TMP_0);
            } else {
                add(reg_soff, reg_soff, reg_mb_stride_Bc);
            }

            // comeback soff_max. Shared with diff_shift.
            ldr_imm(reg_soff_max, sp, stack_off_soff_max);
            cmp(reg_soff, reg_soff_max);
            b(LT, diff_spatial);
        }
        if (jbp_->is_nspc_) {
            // comeback
            if (!pd_->use_global_stats()) {
                ldr_imm(reg_src, sp, stack_off_src);
            }

            ldr_imm(reg_diff_dst, sp, stack_off_diff_dst);
            ldr_imm(reg_diff_src, sp, stack_off_diff_src);
            if (with_relu) { ldr_imm(reg_ws, sp, stack_off_ws); }
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd, const jit_bnorm_conf_t *jbp)
        : pd_(pd)
        , jbp_(jbp)
        , is_bf16_(pd_->src_md()->data_type == data_type::bf16)
        , is_f16_(pd_->src_md()->data_type == data_type::f16)
        , vlen_spat_data_(vlen / (1 + is_xf16())) // 32B of xF16 -> 64B of FP32
        , unroll_blocks(simd_bytes(isa) >= 32 && !jbp_->is_spatial_thr_ ? 4 : 1)
        , unroll_regs(simd_bytes(isa) >= 32 && !jbp_->is_spatial_thr_ ? 4 : 1) {
        static_assert(isa == asimd || isa == sve, "unsupported isa");
    }

    void generate() override {
        preamble();

        if (is_superset(isa, sve)) { prepare_tail_mask(); }

        compute_static_strides();

        prepare_relu();

        sub_imm(sp, sp, (int)stack_size_required, X_TMP_0);
        load_common_params();

        if (pd_->is_fwd()) {
            if (!pd_->stats_is_src()) { compute_mean_variance(); }
            forward();
        } else {
            backward();
        }
        add_imm(sp, sp, (int)stack_size_required, X_TMP_0);
        postamble();
    }

    void operator()(const call_params_t *p) { jit_generator_t::operator()(p); }

    ~jit_bnorm_t() override = default;
};

namespace bnorm_impl {

template <cpu_isa_t isa>
struct driver_t {
    driver_t(const batch_normalization_pd_t *pd, int nthr)
        : pd_(pd), jbp_(pd_, nthr, simd_w_), ker_(pd_, &jbp_) {}

    ~driver_t() = default;

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *pd, int nthr) {
        dim_t C_PADDED = get_c_padded(pd);

        auto sbuf_sz = use_tmp_stats(pd) * 2 * C_PADDED;
        auto pbuf_sz
                = (use_tmp_diff_scale(pd) + use_tmp_diff_shift(pd)) * C_PADDED;
        auto rbuf_sz = (pd->is_fwd() ? 1 : 2) * C_PADDED * nthr;

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);

        if (dnnl_thr_syncable()) {
            auto n_barriers = C_PADDED / simd_elems(data_type::f32, isa);
            scratchpad.book<barrier::ctx_64_t>(key_barrier, n_barriers);
        }
    }

    // given nthr, shape of problem, and thread partition,
    // balance work among the threads
    void thread_balance(int ithr, int nthr, dim_t N, dim_t C_blks, dim_t SP,
            int &C_ithr, int C_nthr, dim_t &C_blk_s, dim_t &C_blk_e,
            int &N_ithr, int N_nthr, dim_t &N_s, dim_t &N_e, int &S_ithr,
            int S_nthr, dim_t &S_s, dim_t &S_e) {
        if (ithr < C_nthr * N_nthr * S_nthr) {
            utils::nd_iterator_init(
                    ithr, C_ithr, C_nthr, N_ithr, N_nthr, S_ithr, S_nthr);
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
            balance211(N, N_nthr, N_ithr, N_s, N_e);
            balance211(SP, S_nthr, S_ithr, S_s, S_e);
        } else {
            S_ithr = N_ithr = C_ithr = -ithr;
            S_s = S_e = N_s = N_e = C_blk_s = C_blk_e = -1;
        }
    }

    void exec(int ithr, int nthr, const void *src, void *diff_src, void *dst,
            const void *diff_dst, const acc_data_t *scale,
            acc_data_t *diff_scale, const acc_data_t *shift,
            acc_data_t *diff_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
        auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);

        dim_t N = pd_->MB();
        dim_t C = pd_->C();
        dim_t C_PADDED = get_c_padded(pd_);
        dim_t D = pd_->D();
        dim_t H = pd_->H();
        dim_t W = pd_->W();
        dim_t SP = D * H * W;
        dim_t img_size = C_PADDED * SP;
        const int vlen_spat_data = ker_.spat_step;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = pd_->desc()->batch_norm_epsilon;
        p.one = 1.0f;
        p.spat_size = SP;
        p.chan_size = 1.0f * N * p.spat_size;

        int C_ithr {0}, N_ithr {0}, S_ithr {0};
        dim_t C_blk_s {0}, C_blk_e {0}, N_s {0}, N_e {0}, S_s {0}, S_e {0};

        thread_balance(ithr, nthr, N, jbp_.C_blks_per_iter_, SP, C_ithr,
                jbp_.C_nthr_, C_blk_s, C_blk_e, N_ithr, jbp_.N_nthr_, N_s, N_e,
                S_ithr, jbp_.S_nthr_, S_s, S_e);

        int SP_N_ithr = N_ithr * jbp_.S_nthr_ + S_ithr;
        int SP_N_nthr = jbp_.N_nthr_ * jbp_.S_nthr_;
        assert(IMPLICATION(!dnnl_thr_syncable(), SP_N_nthr == 1));

        p.N_ithr = SP_N_ithr;
        p.N_nthr = SP_N_nthr;

        int global_C_blk_s;
        int global_barriers_per_iter = jbp_.C_nthr_;

        for (int64_t it = 0; it < jbp_.iters_; it++) {
            if (it == jbp_.iters_ - 1 && jbp_.iters_ > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                thread_balance(ithr, nthr, N, jbp_.C_blks_last_iter_, SP,
                        C_ithr, jbp_.C_nthr_last_iter_, C_blk_s, C_blk_e,
                        N_ithr, jbp_.N_nthr_last_iter_, N_s, N_e, S_ithr,
                        jbp_.S_nthr_last_iter_, S_s, S_e);

                // Update call parameters for JIT, last iteration
                p.N_ithr = N_ithr * jbp_.S_nthr_last_iter_ + S_ithr;
                p.N_nthr = jbp_.N_nthr_last_iter_ * jbp_.S_nthr_last_iter_;
            }

            global_C_blk_s = jbp_.do_blocking_ ? (C_blk_s == -1)
                            ? -1
                            : it * jbp_.C_blks_per_iter_ + C_blk_s
                                               : C_blk_s;

            int C_blks_thr = C_blk_e - C_blk_s;
            int N_thr = N_e - N_s;

            if (C_blks_thr == 0 || N_thr == 0) continue;

            size_t coff_base = global_C_blk_s * simd_w_;
            size_t soff_base = jbp_.is_nspc_
                    ? coff_base + N_s * img_size
                    : global_C_blk_s * p.spat_size * simd_w_ + N_s * img_size;
            size_t shift_off = use_tmp_diff_scale(pd_) ? pd_->C() : 0;

            p.spat_size_loc = S_e - S_s;
            p.S_s = S_s * vlen_spat_data;
            p.S_tail = (p.spat_size - S_e) * vlen_spat_data;
            p.coff_max = C_blks_thr * simd_w_;
            const auto tmp_mean = use_tmp_stats(pd_) ? sbuf : mean;
            if (tmp_mean != nullptr) p.mean = tmp_mean + coff_base;
            const auto tmp_var = use_tmp_stats(pd_) ? sbuf + C_PADDED : var;
            if (tmp_var != nullptr) p.var = tmp_var + coff_base;
            if (scale != nullptr) p.scale = scale + coff_base;
            if (shift != nullptr) p.shift = shift + coff_base;
            const auto tmp_diff_scale
                    = use_tmp_diff_scale(pd_) ? pbuf : diff_scale;
            if (tmp_diff_scale != nullptr)
                p.diff_scale = tmp_diff_scale + coff_base;
            const auto tmp_diff_shift
                    = use_tmp_diff_shift(pd_) ? &pbuf[shift_off] : diff_shift;
            if (tmp_diff_shift != nullptr)
                p.diff_shift = tmp_diff_shift + coff_base;

            p.soff_max = jbp_.dt_size_ * N_thr * img_size;
            if (src != nullptr)
                p.src = (void *)((char *)src + soff_base * jbp_.dt_size_);
            if (dst != nullptr)
                p.dst = (void *)((char *)dst + soff_base * jbp_.dt_size_);
            if (diff_src != nullptr)
                p.diff_src = (void *)((char *)diff_src
                        + soff_base * jbp_.dt_size_);
            if (diff_dst != nullptr)
                p.diff_dst = (void *)((char *)diff_dst
                        + soff_base * jbp_.dt_size_);
            if (ws != nullptr) p.ws = ws + soff_base / 8;

            p.mb_stride_Bc
                    = jbp_.dt_size_ * (img_size - p.coff_max * p.spat_size);

            // use SP_N_nthr which is the same as p.N_nthr except maybe for
            // the last iteration.
            p.rbuf1 = rbuf
                    + ((it * jbp_.C_blks_per_iter_) * SP_N_nthr
                              + C_blk_s * p.N_nthr + p.N_ithr * C_blks_thr)
                            * simd_w_;
            // rbuf1 and rbuf2 have to be disjoint
            p.rbuf2 = p.rbuf1 + C_PADDED * nthr;
            p.is_cblk_tail
                    = (it * jbp_.C_blks_per_iter_ + C_blk_e) * simd_w_ > C;

            size_t iter_barriers
                    = jbp_.do_blocking_ ? it * global_barriers_per_iter : 0;
            p.barrier = barriers + C_ithr + iter_barriers;
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    void init_barriers(const memory_tracking::grantor_t &scratchpad) {
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);
        if (barriers) {
            const int n_barriers = get_c_padded(pd_) / simd_w_;
            for (int i = 0; i < n_barriers; ++i)
                barrier::ctx_init(&barriers[i]);
        }
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    int simd_w_
            = simd_bytes(isa) / sizeof(acc_data_t); // BF16 will expand to FP32

    static bool use_tmp_stats(const batch_normalization_pd_t *pd) {
        return !pd->stats_is_src()
                && pd->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale(const batch_normalization_pd_t *pd) {
        return (!pd->is_fwd() && !pd->use_scale())
                || pd->desc()->prop_kind == prop_kind::backward_data;
    }

    static bool use_tmp_diff_shift(const batch_normalization_pd_t *pd) {
        return (!pd->is_fwd() && !pd->use_shift())
                || pd->desc()->prop_kind == prop_kind::backward_data;
    }

    const batch_normalization_pd_t *pd_;
    jit_bnorm_conf_t jbp_;
    jit_bnorm_t<isa> ker_;
};
} // namespace bnorm_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::pd_t::init(
        const engine_t *engine) {
    VDISPATCH_BNORM(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL,
            VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME);
    VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_BNORM(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_BNORM(!has_zero_dim_memory(), "zero dims are not supported");
    VDISPATCH_BNORM(
            dnnl_thr_syncable(), "implementation requires DNNL_THR_SYNC");
    VDISPATCH_BNORM(src_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BNORM(src_md()->data_type == dst_md()->data_type,
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_BNORM(
            check_scale_shift_data_type(), VERBOSE_INCONSISTENT_ALPHA_BETA);
    VDISPATCH_BNORM(
            attr()->has_default_values() || with_relu_post_op(is_training()),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_BNORM(set_default_formats_common(),
            "setting dst memory descriptor failed");
    VDISPATCH_BNORM(
            memory_desc_wrapper(src_md()) == memory_desc_wrapper(dst_md()),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_BNORM(is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // BN+Add+Relu fusion is not currently implemented
    VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_BAD_FLAGS);

    const memory_desc_wrapper src_d(src_md());
    format_tag_t src_tag = format_tag::undef;
    switch (simd_bytes(isa)) {
        case 64:
            src_tag = src_d.matches_one_of_tag(
                    nCw16c, nChw16c, nCdhw16c, nc, nwc, nhwc, ndhwc);
            break;
        case 32:
            src_tag = src_d.matches_one_of_tag(
                    nCw8c, nChw8c, nCdhw8c, nc, nwc, nhwc, ndhwc);
            break;
        case 16:
            src_tag = src_d.matches_one_of_tag(
                    nCw4c, nChw4c, nCdhw4c, nc, nwc, nhwc, ndhwc);
            break;

        default: return status::unimplemented;
    }

    VDISPATCH_BNORM(src_tag != format_tag::undef, VERBOSE_UNSUPPORTED_TAG);

    const bool need_relu_support
            = fuse_norm_relu() || with_relu_post_op(is_training());

    if (need_relu_support) {
        const bool is_sve = is_superset(isa, sve);
        const bool is_sve_512 = is_sve && simd_bytes(isa) == 64;

        VDISPATCH_BNORM(
                is_sve, "relu fusion/post-op is unsupported for this isa");

        if (is_training()) {
            // Training with relu is only supported on SVE-512.
            VDISPATCH_BNORM(is_sve_512,
                    "training with relu fusion/post-op is not supported for "
                    "this isa");

            // Fused relu needs to output a workspace for training
            if (fuse_norm_relu()) { init_default_ws(1); }
        }
    }

    VDISPATCH_BNORM(
            IMPLICATION(memory_desc_wrapper(src_md()).padded_dims()[1] != C(),
                    is_superset(isa, sve)),
            "padded dimensions are not supported on this isa");

    // Only IC % 16 == 0 is supported for now
    VDISPATCH_BNORM(IMPLICATION(src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc),
                            src_d.padded_dims()[1] % 16 == 0),
            "padded input channels must be divisible by 16");

    nthr_ = dnnl_get_max_threads();
    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this, nthr_);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_impl::driver_t<isa>(pd(), pd()->nthr_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SHIFT);

    auto mean = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                      CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
    auto var = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                      CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);
    const int nthr = pd()->nthr_;

    parallel(nthr, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, nullptr, dst, nullptr, scale,
                nullptr, shift, nullptr, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::pd_t::init(
        const engine_t *engine) {
    VDISPATCH_BNORM(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL,
            VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME);
    VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_BNORM(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_BNORM(!has_zero_dim_memory(), "zero dims are not supported");
    VDISPATCH_BNORM(
            dnnl_thr_syncable(), "implementation requires DNNL_THR_SYNC");
    VDISPATCH_BNORM(src_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BNORM(utils::everyone_is(src_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_BNORM(
            check_scale_shift_data_type(), VERBOSE_INCONSISTENT_ALPHA_BETA);
    VDISPATCH_BNORM(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_BNORM(set_default_formats_common(),
            "setting dst memory descriptor failed");
    VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
    VDISPATCH_BNORM(is_dense_format_kind(
                            {src_md(), dst_md(), diff_src_md(), diff_dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // BN+Add+Relu fusion is not currently implemented
    VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_BAD_FLAGS);

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());
    auto src_tag = format_tag::undef;
    auto diff_src_tag = format_tag::undef;
    switch (simd_bytes(isa)) {
        case 64:
            src_tag = src_d.matches_one_of_tag(
                    nCw16c, nChw16c, nCdhw16c, nc, nwc, nhwc, ndhwc);
            diff_src_tag = diff_src_d.matches_one_of_tag(
                    nc, nwc, nCw16c, nhwc, nChw16c, ndhwc, nCdhw16c);
            break;
        case 32:
            src_tag = src_d.matches_one_of_tag(
                    nCw8c, nChw8c, nCdhw8c, nc, nwc, nhwc, ndhwc);
            diff_src_tag = diff_src_d.matches_one_of_tag(
                    nc, nwc, nCw8c, nhwc, nChw8c, ndhwc, nCdhw8c);
            break;
        case 16:
            src_tag = src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc);
            diff_src_tag = diff_src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc);
            break;

        default: return status::unimplemented;
    }

    VDISPATCH_BNORM(src_tag != format_tag::undef
                    && diff_src_tag != format_tag::undef
                    && src_tag == diff_src_tag,
            VERBOSE_UNSUPPORTED_TAG);

    VDISPATCH_BNORM(
            IMPLICATION(memory_desc_wrapper(src_md()).padded_dims()[1] != C(),
                    is_superset(isa, sve)),
            "padded dimensions are not supported on this isa");

    // Only IC % 16 == 0 is supported for now
    VDISPATCH_BNORM(IMPLICATION(src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc),
                            src_d.padded_dims()[1] % 16 == 0),
            VERBOSE_SHAPE_RESTRICTION);

    if (fuse_norm_relu()) {
        VDISPATCH_BNORM(is_superset(isa, sve) && simd_bytes(isa) == 64,
                "relu fusion is not supported on this isa");
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    /* TODO: extra checks required */

    nthr_ = dnnl_get_max_threads();
    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this, nthr_);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_impl::driver_t<isa>(pd(), pd()->nthr_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto var = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    auto diff_scale = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE);
    auto diff_shift = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SHIFT);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);
    const int nthr = pd()->nthr_;

    parallel(nthr, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, diff_src, nullptr, diff_dst, scale,
                diff_scale, nullptr, diff_shift, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<asimd>;
template struct jit_uni_batch_normalization_fwd_t<sve>;

template struct jit_uni_batch_normalization_bwd_t<asimd>;
template struct jit_uni_batch_normalization_bwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
