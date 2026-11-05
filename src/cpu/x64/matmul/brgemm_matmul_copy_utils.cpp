/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

template <typename Vmm>
struct jit_brgemm_matmul_copy_a_impl_t : public jit_brgemm_matmul_copy_a_t,
                                         public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_impl_t)

    jit_brgemm_matmul_copy_a_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator_t(jit_name())
        , typesize_(static_cast<int>(conf_->a_dt_sz))
        , tr_typesize_(static_cast<int>(conf_->tr_a_dt_sz))
        , vnni_granularity_(
                  static_cast<int>(data_type_vnni_granularity(conf_->src_dt)))
        , k_step_(vlen_ / nstl::max(typesize_, tr_typesize_))
        , src_stride_(conf_->copy_A_src_stride)
        , tr_src_stride_((conf_->use_buffer_a_tail_only
                                         ? static_cast<dim_t>(conf_->wei_k_blk)
                                         : conf_->LDA)
                  * tr_typesize_)
        , do_compute_compensation_(
                  conf_->has_zero_point_b && !conf_->with_wei_decompression)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        // See the note in `create_brgemm_matmul_copy_b` why `orig_src_dt` used.
        , use_fp16_instructions_(conf_->isa == avx512_core_fp16
                  && conf_->orig_src_dt == data_type::f16
                  && conf_->src_dt == data_type::f32)
        , req_cvtf82f16_(conf->isa == avx10_2 && conf->src_dt == data_type::f16
                  && utils::one_of(conf->orig_src_dt, data_type::f8_e4m3,
                          data_type::f8_e5m2))
        , k_loop_unroll_(is_ymm_ ? 7 : 16)
        , vmm_copy_idx_(is_ymm_                      ? 13
                          : avx512_core_dot_product_ ? 27
                                                     : 29) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    static constexpr int vlen_ = vreg_traits_t<Vmm>::vlen;
    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr int num_comp_acc_ = is_ymm_ ? 7 : 8;

    const int typesize_;
    const int tr_typesize_;
    const int vnni_granularity_;
    const int k_step_;
    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool do_compute_compensation_;
    const bool avx512_core_dot_product_;
    const bool use_fp16_instructions_;
    const bool req_cvtf82f16_;

    const int k_loop_unroll_;
    const int vmm_copy_idx_;

    opmask_t kTail_load = k7;
    opmask_t kTail_store = k6;
    opmask_t kTail_comp = k5;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_K_start = abi_not_param1;

    reg64_t reg_zp_comp_buf_ptr = rdx;
    reg64_t reg_zp_comp_res_ptr = rsi;

    reg64_t reg_M_blk = r9;
    reg64_t reg_K_blk = r10;
    reg64_t reg_batch = r11;
    reg64_t reg_aux_src = r12;
    reg64_t reg_aux_tr_src = r13;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_zp_ab_comp_ptr = imm_addr64;
    reg64_t reg_zp_b_neg_val_ptr = reg_K_blk;

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(28);
    Vmm vmm_dot_product_temp = Vmm(29);

    Vmm vmm_comp_mul = Vmm(is_ymm_ ? 14 : 30); // 1s
    Vmm vmm_comp_add = Vmm(is_ymm_ ? 15 : 31); // 128

    // Allows to shift A data by 128 for s8s8 problem for AVX512 in copy
    // routine, not in compute kernel. It's disabled for now, as it
    // requires setting some hint to brgemm kernel to avoid double shifting
    const bool allow_input_shift_for_s8s8 = false;

    Vmm get_vmm_comp_acc(int i) {
        assert(i >= 0 && i < num_comp_acc_);
        return Vmm(i);
    }

    Vmm get_vmm_copy(int i) {
        assert(i >= 0 && i < k_loop_unroll_);
        return Vmm(vmm_copy_idx_ - i);
    }

    void load_vmm(int idx, int offset) {}
    void store_vmm(int idx, int offset) {}
    void load_tail(int k_tail, size_t offset) {}
    void store_tail(int k_tail, size_t offset) {}
    void reduce_compensation_across_accumulators(int num_accumulators);
    void copy_K_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    void copy_M_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3, get_encoding());
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }
    void generate() override;
};

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::load_vmm(int idx, int offset) {
    const auto addr = EVEX_compress_addr(reg_src, offset);
    const auto vmm_copy = get_vmm_copy(idx);
    if (use_fp16_instructions_) {
        vcvtph2psx(vmm_copy, addr);
    } else if (req_cvtf82f16_) {
        if (conf_->orig_src_dt == data_type::f8_e4m3)
            vcvthf82ph(vmm_copy, addr);
        else if (conf_->orig_src_dt == data_type::f8_e5m2) {
            vpmovzxbw(vmm_copy, addr);
            vpsllw(vmm_copy, vmm_copy, 8);
        } else {
            assert(!"unsupported data type.");
        }
    } else {
        vmovdqu8(get_vmm_copy(idx), addr);
    }
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::load_vmm(int idx, int offset) {
    uni_vmovups(get_vmm_copy(idx), ptr[reg_src + offset]);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::store_vmm(int idx, int offset) {
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);
    vmovdqu8(tr_src_addr, get_vmm_copy(idx));
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::store_vmm(int idx, int offset) {
    uni_vmovups(ptr[reg_tr_src + offset], get_vmm_copy(idx));
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::load_tail(
        int k_tail, size_t offset) {
    const auto kmovx = [this](Opmask k, size_t q) {
        if (conf_->is_bf32) {
            mov(regq_tmp.cvt32(), q);
            jit_generator_t::kmovw(k, regq_tmp.cvt32());
        } else {
            mov(regq_tmp, q);
            jit_generator_t::kmovq(k, regq_tmp);
        }
    };

    const size_t dt_step
            = conf_->is_bf32 || use_fp16_instructions_ ? 1 : typesize_;
    const size_t tail_mask_load = size_t(((size_t)1 << (dt_step * k_tail)) - 1);
    kmovx(kTail_load, tail_mask_load);
    const int k_tail_st = rnd_up(k_tail, vnni_granularity_);
    const size_t full_mask
            = conf_->is_bf32 ? ((size_t)1 << 16) - 1 : 0xffffffffffffffff;
    const size_t tail_mask_store = k_tail_st == k_step_
            ? full_mask
            : size_t(((size_t)1 << (dt_step * k_tail_st)) - 1);
    kmovx(kTail_store, tail_mask_store);

    auto zmm_tail = get_vmm_copy(0) | kTail_load | T_z;
    auto load_addr = EVEX_compress_addr(reg_src, offset * typesize_);
    if (conf_->is_bf32)
        vmovups(zmm_tail, load_addr);
    else if (use_fp16_instructions_)
        vcvtph2psx(zmm_tail, load_addr);
    else if (req_cvtf82f16_) {
        if (conf_->orig_src_dt == data_type::f8_e4m3)
            vcvthf82ph(zmm_tail, load_addr);
        else if (conf_->orig_src_dt == data_type::f8_e5m2) {
            vpmovzxbw(zmm_tail, load_addr);
            vpsllw(zmm_tail, zmm_tail, 8);
        } else
            assert(!"unsupported data type.");
    } else
        vmovdqu8(zmm_tail, load_addr);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::load_tail(
        int k_tail, size_t offset) {
    const auto vmm_tail = get_vmm_copy(0);
    load_bytes(vmm_tail, reg_src, offset * typesize_, k_tail * typesize_);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Zmm>::store_tail(
        int k_tail, size_t offset) {
    auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset * tr_typesize_);
    if (conf_->is_bf32) {
        Ymm ymm_downcvt_bf16 = Ymm(get_vmm_copy(0).getIdx());
        vcvtneps2bf16(ymm_downcvt_bf16, get_vmm_copy(0));
        vmovdqu16(tr_src_addr, ymm_downcvt_bf16 | kTail_store);
    } else if (use_fp16_instructions_) {
        vmovups(tr_src_addr, get_vmm_copy(0) | kTail_store);
    } else if (req_cvtf82f16_) {
        vmovdqu16(tr_src_addr, get_vmm_copy(0) | kTail_store);
    } else
        vmovdqu8(tr_src_addr, get_vmm_copy(0) | kTail_store);
}

template <>
void jit_brgemm_matmul_copy_a_impl_t<Ymm>::store_tail(
        int k_tail, size_t offset) {
    const int k_tail_st = rnd_up(k_tail, vnni_granularity_);
    const auto vmm_tail = get_vmm_copy(0);
    store_bytes(
            vmm_tail, reg_tr_src, offset * tr_typesize_, k_tail_st * typesize_);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<
        Vmm>::reduce_compensation_across_accumulators(int num_accumulators) {
    int num = num_accumulators;
    while (num > 1) {
        for (int i = 0; i < num / 2; i++) {
            const auto vmm_acc0 = get_vmm_comp_acc(i);
            const auto vmm_acc1 = get_vmm_comp_acc(div_up(num, 2) + i);
            uni_vpaddd(vmm_acc0, vmm_acc0, vmm_acc1);
        }
        num = div_up(num, 2);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::copy_K_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    const int K_blk = static_cast<int>(is_K_tail
                    ? conf_->K % conf_->K_blk
                    : nstl::min<dim_t>(conf_->K, conf_->K_blk));
    const int k_tail = K_blk % k_step_;
    const int num_k_iters = K_blk / k_step_;
    const int num_acc = utils::saturate(1, (int)num_comp_acc_, num_k_iters);

    if (do_compute_compensation_) {
        for (int i = 0; i < num_acc; i++) {
            const auto vmm_acc = get_vmm_comp_acc(i);
            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
        }
    }

    auto maybe_compute_compensation = [this, num_acc](int k_idx, Vmm vmm_copy) {
        if (do_compute_compensation_) {
            const auto vmm_comp_acc = get_vmm_comp_acc(k_idx % num_acc);
            if (conf_->src_dt == data_type::s8)
                dot_product(vmm_comp_acc, vmm_comp_mul, vmm_copy);
            else
                dot_product(vmm_comp_acc, vmm_copy, vmm_comp_mul);
        }
    };

    for (int kb = 0; kb < div_up(num_k_iters, k_loop_unroll_); kb++) {
        const int k_end
                = nstl::min(k_loop_unroll_, num_k_iters - kb * k_loop_unroll_);
        for (int k = 0; k < k_end; k++) {
            const int k_idx = kb * k_loop_unroll_ + k;
            const size_t offset
                    = static_cast<size_t>(k_idx) * k_step_ * typesize_;
            load_vmm(k, static_cast<int>(offset));
            maybe_compute_compensation(k_idx, get_vmm_copy(k));
        }
        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
            for (int k = 0; k < k_end; k++)
                vpaddb(get_vmm_copy(k), get_vmm_copy(k), vmm_comp_add);
        }
        if (conf_->is_bf32) {
            assert(typesize_ != tr_typesize_);
            int k = 0;
            const int k_end_2 = rnd_dn(k_end, 2);
            for (; k < k_end_2; k += 2) {
                const size_t offset = ((size_t)kb * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);

                auto zmm_src = get_vmm_copy(k);
                auto zmm_src_next = get_vmm_copy(k + 1);

                vcvtne2ps2bf16(zmm_src, zmm_src_next, zmm_src);
                vmovups(tr_src_addr, zmm_src);
            }
            if (k < k_end) {
                const size_t offset = ((size_t)kb * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                auto tr_src_addr = EVEX_compress_addr(reg_tr_src, offset);
                Ymm ymm_downcvt_bf16 = Ymm(get_vmm_copy(k).getIdx());
                vcvtneps2bf16(ymm_downcvt_bf16, get_vmm_copy(k));
                vmovdqu16(tr_src_addr, ymm_downcvt_bf16);
            }
        } else {
            for (int k = 0; k < k_end; k++) {
                const size_t offset
                        = (static_cast<size_t>(kb) * k_loop_unroll_ + k)
                        * k_step_ * tr_typesize_;
                store_vmm(k, static_cast<int>(offset));
            }
        }
    }

    if (k_tail > 0) {
        load_tail(k_tail, num_k_iters * k_step_);
        maybe_compute_compensation(0, get_vmm_copy(0));

        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required)
            vpaddb(get_vmm_copy(0), get_vmm_copy(0), vmm_comp_add);

        store_tail(k_tail, num_k_iters * k_step_);
    }

    if (do_compute_compensation_) {
        reduce_compensation_across_accumulators(num_acc);

        const auto addr_buf = ptr[reg_zp_comp_buf_ptr];
        if (!is_first_K_iter)
            uni_vpaddd(get_vmm_comp_acc(0), get_vmm_comp_acc(0), addr_buf);
        if (!is_last_K_iter) {
            uni_vmovups(addr_buf, get_vmm_comp_acc(0));
            return;
        }

        // is_last_K_iter == true: we need to reduce values within acc
        // register, add mixed ab_compensation component if any, multiply
        // it by negative zp_b_value and finally store the result

        // step 1: reduce values within acc register
        const auto ymm_red0 = Ymm(get_vmm_comp_acc(0).getIdx());
        const auto ymm_red1 = Ymm(get_vmm_comp_acc(1).getIdx());
        if (!is_ymm_) {
            vextracti64x4(ymm_red1, Zmm(get_vmm_comp_acc(0).getIdx()), 1);
            vphaddd(ymm_red0, ymm_red0, ymm_red1);
        }
        uni_vpxor(ymm_red1, ymm_red1, ymm_red1);
        uni_vphaddd(ymm_red0, ymm_red0, ymm_red1);
        uni_vphaddd(ymm_red0, ymm_red0, ymm_red1);
        const auto xmm_red1 = Xmm(ymm_red1.getIdx());
        vextractf128(xmm_red1, ymm_red0, 1);
        uni_vpaddd(ymm_red0, ymm_red0, ymm_red1);

        const auto vmm_in_mask = get_vmm_comp_acc(1);
        if (is_ymm_) {
            static const uint32_t mask_in[8]
                    = {0xffffffff, 0, 0, 0, 0, 0, 0, 0};
            mov(regq_tmp, reinterpret_cast<size_t>(mask_in));
            vmovups(vmm_in_mask, ptr[regq_tmp]);
        }

        // step 2: add -K * zp_a_val as mixed ab_compensation component
        if (conf_->src_zp_type != brgemm_broadcast_t::none) {
            assert(conf_->src_zp_type == brgemm_broadcast_t::per_tensor);
            reg64_t reg_zp_ab_comp_ptr = imm_addr64;
            mov(reg_zp_ab_comp_ptr, ptr[param1 + GET_OFF(zp_ab_comp_ptr)]);
            if (is_ymm_) {
                const auto vmm_zp = get_vmm_comp_acc(2);
                vmaskmovps(vmm_zp, vmm_in_mask, ptr[reg_zp_ab_comp_ptr]);
                uni_vpaddd(ymm_red0, ymm_red0, vmm_zp);
            } else {
                const auto addr_ab_comp = zword_b[reg_zp_ab_comp_ptr];
                const auto zmm_res = get_vmm_comp_acc(0) | kTail_comp;
                vpaddd(zmm_res, get_vmm_comp_acc(0), addr_ab_comp);
            }
        }

        // step 3: multiply by zp_b_val
        mov(reg_zp_b_neg_val_ptr, ptr[param1 + GET_OFF(zp_b_neg_val_ptr)]);
        const auto vmm_zp_b_neg_val = get_vmm_comp_acc(is_ymm_ ? 2 : 1);
        uni_vbroadcastss(vmm_zp_b_neg_val, ptr[reg_zp_b_neg_val_ptr]);
        uni_vpmulld(get_vmm_comp_acc(0), get_vmm_comp_acc(0), vmm_zp_b_neg_val);

        // step 4: store the final result value
        if (is_ymm_)
            vmaskmovps(ptr[reg_zp_comp_res_ptr], vmm_in_mask, ymm_red0);
        else
            vmovups(ptr[reg_zp_comp_res_ptr], get_vmm_comp_acc(0) | kTail_comp);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::copy_M_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {

    if (do_compute_compensation_) {
        mov(imm_addr64, 1);
        uni_vpbroadcastb(vmm_comp_mul, imm_addr64.cvt8());
        if (!(is_first_K_iter && is_last_K_iter))
            mov(reg_zp_comp_buf_ptr,
                    ptr[param1 + GET_OFF(zp_b_compensation_buffer_ptr)]);

        if (is_last_K_iter) {
            mov(reg_zp_comp_res_ptr,
                    ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);
            if (!is_ymm_) {
                mov(regq_tmp, 1);
                jit_generator_t::kmovw(kTail_comp, imm_addr64.cvt32());
            }
        }
    }

    Label loop_M;
    L(loop_M);

    copy_K_loop(is_K_tail, is_first_K_iter, is_last_K_iter);

    add(reg_src, src_stride_);
    add(reg_tr_src, tr_src_stride_);
    if (do_compute_compensation_) {
        // shift comp pointers
        if (!(is_first_K_iter && is_last_K_iter))
            add(reg_zp_comp_buf_ptr, sizeof(int32_t) * 16);
        if (is_last_K_iter) add(reg_zp_comp_res_ptr, sizeof(int32_t));
    }

    dec(reg_M_blk);
    jnz(loop_M, T_NEAR);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_impl_t<Vmm>::generate() {
    preamble();

    if (avx512_core_dot_product_) {
        mov(regq_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, regq_tmp.cvt16());
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_blk, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(reg_M_blk, ptr[param1 + GET_OFF(current_M_blk)]);

    if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
        mov(imm_addr64, 128);
        uni_vpbroadcastb(vmm_comp_add, imm_addr64.cvt8());
    }

    auto copy_body = [this](bool is_first_K_iter, bool is_last_K_iter) {
        Label copy_body_done;
        // might be different from conf_->K_tail
        const dim_t K_blk_tail
                = conf_->K_tail > 0 ? conf_->K % conf_->K_blk : 0;
        if (K_blk_tail > 0) {
            Label not_K_tail;
            cmp(reg_K_blk, K_blk_tail);
            jne(not_K_tail, T_NEAR);
            copy_M_loop(true, is_first_K_iter, is_last_K_iter);
            jmp(copy_body_done, T_NEAR);

            L(not_K_tail);
        }

        copy_M_loop(false, is_first_K_iter, is_last_K_iter);
        L(copy_body_done);
    };

    Label done;
    if (do_compute_compensation_) {
        assert(conf_->wei_zp_type == brgemm_broadcast_t::per_tensor);

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            copy_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            copy_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        copy_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }
    copy_body(false, false);
    L(done);

    postamble();
}

template struct jit_brgemm_matmul_copy_a_impl_t<Zmm>;
template struct jit_brgemm_matmul_copy_a_impl_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_a_transposed_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_transposed_impl_t)

    jit_brgemm_matmul_copy_a_transposed_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator_t(jit_name())
        , typesize(conf_->a_dt_sz)
        , tr_typesize(conf_->tr_a_dt_sz)
        , rows_step(16)
        , columns_step(rows_step)
        , src_stride(conf_->copy_A_src_stride)
        , dst_stride(conf_->LDA * tr_typesize)
        , m_loop_src_shift(columns_step * typesize)
        , m_loop_dst_shift(columns_step * dst_stride)
        , k_loop_src_shift(rows_step * src_stride)
        , k_loop_dst_shift(rows_step * tr_typesize)
        , is_f32(conf_->src_dt == data_type::f32)
        , is_bf16(conf_->src_dt == data_type::bf16)
        , is_f16(conf_->src_dt == data_type::f16)
        , is_bf32(conf_->is_bf32)
        , is_f8(utils::one_of(
                  conf_->src_dt, data_type::f8_e4m3, data_type::f8_e5m2))
        , is_dynamic_src_ld(conf_->is_runtime_M)
        // See the note in `create_brgemm_matmul_copy_b` why `orig_src_dt` used.
        , req_cvtf82f16_(utils::one_of(conf_->orig_src_dt, data_type::f8_e4m3,
                                 data_type::f8_e5m2)
                  && conf_->src_dt == data_type::f16 && conf_->isa == avx10_2)
        , use_fp16_instructions_(conf_->isa == avx512_core_fp16
                  && conf_->orig_src_dt == data_type::f16
                  && conf_->src_dt == data_type::f32) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    const size_t typesize;
    const size_t tr_typesize;
    const int rows_step;
    const int columns_step;
    const dim_t src_stride, dst_stride;
    const dim_t m_loop_src_shift;
    const dim_t m_loop_dst_shift;
    const dim_t k_loop_src_shift;
    const dim_t k_loop_dst_shift;
    const bool is_f32;
    const bool is_bf16;
    const bool is_f16;
    const bool is_bf32;
    const bool is_f8;
    const bool is_dynamic_src_ld;
    const bool req_cvtf82f16_;
    const bool use_fp16_instructions_;

    opmask_t kFFFF = k1;
    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t kCCCC = k4;
    opmask_t k55 = k5;
    opmask_t k0F0F = k5;
    opmask_t kCC = k6;
    opmask_t kF0F0 = k6;
    opmask_t k33 = k7;
    opmask_t kTail = is_f32 ? k7 : k1;

    reg64_t regq_tmp = r15;
    reg32_t regw_tmp = regq_tmp.cvt32();
    reg64_t reg_k_src = r14;
    reg64_t reg_k_dst = r13;
    reg64_t reg_m_src = r12;
    reg64_t reg_m_dst = r11;
    reg64_t reg_aux_src0 = r10;
    reg64_t reg_aux_src1 = r9;
    reg64_t reg_loop_k = rax;
    reg64_t reg_loop_m = rbx;
    reg64_t imm_addr64 = rdx;
    // Note: this must be assigned to rcx as it's used in shl instruction,
    // clashes with abi_param1 on Windows OS
    reg64_t reg_opmask_shift_compute = rcx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    constexpr static int current_M_blk_offt_ = 0;
    constexpr static int src_offt_ = 8;
    constexpr static int tr_src_offt_ = 16;
    constexpr static int current_K_blk_offt_ = 24;
    constexpr static int dynamic_src_ld_offt_ = 32;
    constexpr static int dynamic_src_ld_x_2_offt_ = 40;
    constexpr static int dynamic_src_ld_x_kstep_offt_ = 48;
    constexpr static int stack_space_needed_ = 56;

    void vmovdqa64(Vmm v, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(v, ptr[imm_addr64]);
    }

    void vmovdqa32(Vmm v, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa32(v, ptr[imm_addr64]);
    }

    void kmovw(Opmask mask_reg, size_t mask) {
        mov(regw_tmp, mask);
        jit_generator_t::kmovw(mask_reg, regw_tmp);
    }

    void f8_to_f16_upconvert(const data_type_t dt, const Vmm &vmm_out,
            const Xbyak::Address &addr);
    void transpose_f32(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void transpose_bf16(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void transpose_f16(reg64_t dst, reg64_t src, int nrows, int ncolumns);

    void transpose_f8(reg64_t dst, reg64_t src, int nrows, int ncolumns);

    // Transpose an up-to-8x8 block using AVX2 fp32 transpose algorithm.
    //
    // For bf16 inputs we first upconvert values to fp32 then run the same
    // transpose algorithm and then convert back to bf16 on store.
    //
    // Layout:
    // - loads read rows from the source matrix
    // - stores write columns to the destination matrix (transposed)
    //
    // The kernel processes columns in groups of 4 using two 128-bit halves
    // of a YMM register (lower rows 0..3, upper rows 4..7).
    template <typename T = Vmm,
            typename = typename utils::enable_if<
                    std::is_same<T, Xbyak::Ymm>::value>::type>
    void transpose_common_ymm(
            data_type_t dt, reg64_t dst, reg64_t src, int nrows, int ncolumns);

    void deploy_transpose(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void init_masks();
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::f8_to_f16_upconvert(
        const data_type_t dt, const Vmm &vmm_out, const Xbyak::Address &addr) {
    if (!utils::one_of(dt, data_type::f8_e4m3, data_type::f8_e5m2)) {
        assert(!"unsupported data type");
        return;
    }

    if (dt == data_type::f8_e4m3)
        vcvthf82ph(vmm_out, addr);
    else if (dt == data_type::f8_e5m2) {
        vpmovzxbw(vmm_out, addr);
        vpsllw(vmm_out, vmm_out, 8);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_f8(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_f8 copy_a_transposed_impl");
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_f16(
        reg64_t reg_dst, reg64_t reg_src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_f16 copy_a_transposed_impl");
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_bf16(
        reg64_t reg_dst, reg64_t reg_src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_bf16 copy_a_transposed_impl");
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_f32(
        reg64_t reg_dst, reg64_t reg_src, int nrows, int ncolumns) {
    assert(!"unsupported transpose_f32 copy_a_transposed_impl");
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Zmm>::transpose_f8(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;

    // This kernel handles one 16x16 tile of fp8 values.
    //
    // The data is transposed directly as raw bytes:
    // - no conversions
    // - no numeric interpretation
    //
    // rows_step / columns_step are expected to match this 16x16 tile shape.
    assert(rows_step == 16);
    assert(columns_step == 16);

    auto src_xmm = [](int i) {
        assert(i >= 0 && i < 32);
        return Xmm(i);
    };

    Label transpose_f8_done;

    // ncolumns == 0 together with dynamic src ld means the effective number
    // of columns is driven by reg_loop_m and handled through a mask.
    const bool dynamic_columns_size = ncolumns == 0 && is_dynamic_src_ld;

    auto kmovx = [this, dynamic_columns_size](
                         Opmask k, unsigned w, bool load_mask_stage = false) {
        if (dynamic_columns_size && load_mask_stage) {
            // Build a mask of the form (1 << reg_loop_m) - 1.
            //
            // reg_opmask_shift_compute is rcx so cl can be used as the shift
            // count. The resulting value is written into the destination mask.
            mov(reg_opmask_shift_compute, reg_loop_m);
            mov(regq_tmp, 1);
            shl(regq_tmp, cl);
            sub(regq_tmp, 1);
        } else {
            mov(regw_tmp, w);
        }
        jit_generator_t::kmovw(k, regw_tmp);
    };

    auto store = [this, dst](Xmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovdqu8(addr, r);
    };

    // The load mask is in units of bytes / f8 elements.
    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovx(kFFFF, load_mask, true);

    // Load up to 16 source rows.
    //
    // Register layout after load:
    //   xmm0 .. xmm(nrows-1) = source rows
    //   xmm(nrows) .. xmm15  = zero-filled rows
    //
    // Each row occupies 16 bytes.
    if (is_dynamic_src_ld) {
        mov(reg_aux_src0, src);
        for (int i = 0; i < nrows; i++) {
            vmovdqu8(src_xmm(i) | kFFFF | T_z, ptr[reg_aux_src0]);
            if (i + 1 < nrows)
                add(reg_aux_src0, ptr[rsp + dynamic_src_ld_offt_]);
        }
    } else {
        for (int i = 0; i < nrows; i++) {
            auto src_addr = EVEX_compress_addr(src, i * src_stride);
            vmovdqu8(src_xmm(i) | kFFFF | T_z, src_addr);
        }
    }

    // For tail tiles (nrows < 16), zero rows beyond nrows so that the transpose
    // produces correctly zero-padded output. Store the filled rows to avoid
    // leaving garbage in dst.
    //
    // This is required because matmul may read that data (currently the case
    // for the f8 data type).
    for (int i = nrows; i < rows_step; i++) {
        vpxord(src_xmm(i), src_xmm(i), src_xmm(i));
    }

    // 16x16 byte transpose algorithm.
    //
    // The algorithm uses staged unpack:
    //   stage 1: bytes  -> combine row pairs
    //   stage 2: words  -> combine 2-row groups into 4-row groups
    //   stage 3: dwords -> combine 4-row groups into 8-row groups
    //   stage 4: qwords -> combine 8-row groups into 16-row groups
    //
    // Temporary register usage:
    // xmm16 .. xmm31 are used as intermediates.
    //
    // Important:
    // After the final stage the output columns are NOT in natural order.
    // The algorithm produces:
    //   xmm0..7 = even columns (0, 2, 4, ..., 14)
    //   xmm8..15 = odd columns (1, 3, 5, ..., 15)
    //
    // A small remap is therefore needed at store time.

    // Stage 1: interleave bytes from pairs of rows.
    for (int i = 0; i < 8; i++) {
        const int idx0 = 2 * i;
        const int idx1 = 2 * i + 1;
        const int tmp0 = 16 + 2 * i;
        const int tmp1 = 16 + 2 * i + 1;

        vpunpcklbw(src_xmm(tmp0), src_xmm(idx0), src_xmm(idx1));
        vpunpckhbw(src_xmm(tmp1), src_xmm(idx0), src_xmm(idx1));
    }

    // Stage 2: interleave 2-byte groups.
    for (int i = 0; i < 4; i++) {
        const int base_tmp = 16 + 4 * i;
        const int base_dst = 4 * i;

        vpunpcklwd(src_xmm(base_dst + 0), src_xmm(base_tmp + 0),
                src_xmm(base_tmp + 2));
        vpunpckhwd(src_xmm(base_dst + 1), src_xmm(base_tmp + 0),
                src_xmm(base_tmp + 2));
        vpunpcklwd(src_xmm(base_dst + 2), src_xmm(base_tmp + 1),
                src_xmm(base_tmp + 3));
        vpunpckhwd(src_xmm(base_dst + 3), src_xmm(base_tmp + 1),
                src_xmm(base_tmp + 3));
    }

    // Stage 3: interleave 4-byte groups.
    for (int i = 0; i < 2; i++) {
        const int base_src = 8 * i;
        const int base_tmp = 16 + 8 * i;

        vpunpckldq(src_xmm(base_tmp + 0), src_xmm(base_src + 0),
                src_xmm(base_src + 4));
        vpunpckhdq(src_xmm(base_tmp + 1), src_xmm(base_src + 0),
                src_xmm(base_src + 4));

        vpunpckldq(src_xmm(base_tmp + 2), src_xmm(base_src + 1),
                src_xmm(base_src + 5));
        vpunpckhdq(src_xmm(base_tmp + 3), src_xmm(base_src + 1),
                src_xmm(base_src + 5));

        vpunpckldq(src_xmm(base_tmp + 4), src_xmm(base_src + 2),
                src_xmm(base_src + 6));
        vpunpckhdq(src_xmm(base_tmp + 5), src_xmm(base_src + 2),
                src_xmm(base_src + 6));

        vpunpckldq(src_xmm(base_tmp + 6), src_xmm(base_src + 3),
                src_xmm(base_src + 7));
        vpunpckhdq(src_xmm(base_tmp + 7), src_xmm(base_src + 3),
                src_xmm(base_src + 7));
    }

    // Stage 4: interleave 8-byte groups.
    for (int i = 0; i < 8; i++) {
        vpunpcklqdq(src_xmm(i), src_xmm(16 + i), src_xmm(24 + i));
        vpunpckhqdq(src_xmm(i + 8), src_xmm(16 + i), src_xmm(24 + i));
    }

    // The transpose algorithm leaves columns grouped as:
    //   xmm0..7 = even columns
    //   xmm8..15 = odd columns
    //
    // Mapping:
    //   col 0 -> xmm0
    //   col 1 -> xmm8
    //   col 2 -> xmm1
    //   col 3 -> xmm9
    //   ...
    //   col 14 -> xmm7
    //   col 15 -> xmm15
    //
    // We need to map logical columns index -> register index.
    auto get_vec_idx = [](int col_idx) {
        assert(col_idx >= 0 && col_idx < 16);
        const bool is_odd = col_idx % 2;
        // index within even/odd group
        const int position = col_idx / 2;
        return position + (is_odd ? 8 : 0);
    };

    const int columns_to_store = dynamic_columns_size ? columns_step : ncolumns;

    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(src_xmm(get_vec_idx(col_idx)), col_idx);
        if (dynamic_columns_size) {
            dec(reg_opmask_shift_compute);
            jz(transpose_f8_done, T_NEAR);
        }
    }

    L(transpose_f8_done);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Zmm>::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;
    auto src_zmm = [](int i) { return Zmm(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    Label transpose_bf16_done;
    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld;
    auto kmovx
            = [this, dynamic_column_size](Opmask k, unsigned w,
                      bool load_mask_stage = false, bool use_word_sz = false) {
        if (dynamic_column_size && load_mask_stage) {
            // reg_opmask_shift_compute is rcx, and we need cl for the shift
            mov(reg_opmask_shift_compute, reg_loop_m);
            mov(regq_tmp, 1);
            shl(regq_tmp, cl);
            sub(regq_tmp, 1);
        } else
            mov(regw_tmp, w);
        if (use_word_sz)
            jit_generator_t::kmovw(k, regw_tmp);
        else
            jit_generator_t::kmovd(k, regw_tmp);
    };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovdqu16(addr, r | kTail);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovx(kFFFF, load_mask, true, is_bf32);

    for (int i = 0; i < nrows / 2; i++) {
        auto idx0 = 2 * i;
        auto idx1 = 2 * i + 1;
        auto zmm_src0 = src_zmm(idx0);
        auto zmm_src1 = src_zmm(idx1);
        if (is_dynamic_src_ld) {
            if (i == 0) {
                mov(reg_aux_src0, src);
                mov(reg_aux_src1, src);
                add(reg_aux_src1, ptr[rsp + dynamic_src_ld_offt_]);
            } else {
                add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                add(reg_aux_src1, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }
        auto src_addr_0 = is_dynamic_src_ld
                ? ptr[reg_aux_src0]
                : EVEX_compress_addr(src, idx0 * src_stride);
        auto src_addr_1 = is_dynamic_src_ld
                ? ptr[reg_aux_src1]
                : EVEX_compress_addr(src, idx1 * src_stride);
        if (is_bf32) {
            vmovups(zmm_src0 | kFFFF | T_z, src_addr_0);
            vmovups(zmm_src1 | kFFFF | T_z, src_addr_1);
            vcvtne2ps2bf16(zmm_src0, zmm_src1, zmm_src0);
        } else if (req_cvtf82f16_) {
            auto src1 = src_ymm(idx1);
            f8_to_f16_upconvert(
                    conf_->orig_src_dt, zmm_src0 | kFFFF | T_z, src_addr_0);
            f8_to_f16_upconvert(
                    conf_->orig_src_dt, zmm_src1 | kFFFF | T_z, src_addr_1);
            vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        } else {
            auto src1 = src_ymm(idx1);
            vmovdqu16(zmm_src0 | kFFFF | T_z, src_addr_0);
            vmovdqu16(zmm_src1 | kFFFF | T_z, src_addr_1);
            vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        }
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = src_zmm(2 * i);
        if (is_dynamic_src_ld) {
            if (i == 0) {
                mov(reg_aux_src0, src);
            } else {
                add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }
        auto src_addr = is_dynamic_src_ld
                ? ptr[reg_aux_src0]
                : EVEX_compress_addr(src, 2 * i * src_stride);
        if (is_bf32) {
            vmovups(zmm_src0 | kFFFF | T_z, src_addr);
            vcvtneps2bf16(Ymm(zmm_src0.getIdx()), zmm_src0);
        } else if (req_cvtf82f16_) {
            f8_to_f16_upconvert(
                    conf_->orig_src_dt, zmm_src0 | kFFFF | T_z, src_addr);
        } else
            vmovdqu16(zmm_src0 | kFFFF | T_z, src_addr);
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < rows_step; i += 2) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [this](int col_idx) {
        MAYBE_UNUSED(this);
        assert(col_idx < columns_step && col_idx >= 0);
        const int blk_sz = 4;
        const int blk_idx = col_idx / blk_sz;
        const int idx_within_blk = col_idx % blk_sz;

        // 0 1 2 3 -> 0 2 1 3
        const int mapped_blk_idx = 2 * blk_idx - (blk_idx / 2) * 3;
        // 0 1 2 3 -> 1 0 3 2
        const int mapped_idx_within_blk
                = idx_within_blk + 1 - 2 * (idx_within_blk % 2);
        return blk_sz * mapped_blk_idx + mapped_idx_within_blk;
    };
    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step ? (1 << rows_to_store) - 1 : 0xffff;
    kmovx(kTail, store_mask);

    const int columns_to_store = dynamic_column_size ? columns_step : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(src_zmm(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute);
            jz(transpose_bf16_done, T_NEAR);
        }
    }

    L(transpose_bf16_done);
}

template <typename Vmm>
template <typename T, typename>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::transpose_common_ymm(
        data_type_t dt, reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    Ymm ymm_tail_mask = ymm15;
    Ymm ymm_upper_tail_mask = ymm14;
    Xmm xmm_upper_tail_mask = xmm14;
    Ymm ymm_tmp = ymm13;

    // avx2 transpose is 8x8, but we need 16x16 transpose. We use four 8x8
    // transposes as below.
    // _    _T       _      _
    // |A, B|   =>   |At, Ct|
    // |C, D|        |Bt, Dt|

    constexpr int avx2_transpose_size = 8;
    // Tails masks are only needed for f32.
    // The bf16 path handles tails directly while loading values.
    if (dt == data_type::f32) {
        const int tail_size = ncolumns % avx2_transpose_size;
        if (tail_size > 0) {
            Xbyak::Reg64 reg_tmp = regq_tmp;
            init_f32_avx2_mask_ymm(ymm_tail_mask, reg_tmp, tail_size);
            const int upper_xmm_tail_size = tail_size - 4;
            if (upper_xmm_tail_size > 0)
                init_f32_avx2_mask_ymm(
                        ymm_upper_tail_mask, reg_tmp, upper_xmm_tail_size);
        }
    }

    // Source element size used for calculating block offsets.
    const dim_t dt_size = types::data_type_size(dt);

    const int A_rows = nstl::min(avx2_transpose_size, nrows);
    const int A_columns = nstl::min(avx2_transpose_size, ncolumns);
    jit_generator_t::transpose(src, dst, src_stride, dst_stride, A_rows,
            A_columns, dt, ymm_tmp, ymm_tail_mask, xmm_upper_tail_mask);
    if (rows_step <= 8) return;

    const dim_t src_B_offset = dt_size * avx2_transpose_size;
    const dim_t dst_B_offset = dst_stride * avx2_transpose_size;
    const int B_rows = nstl::min(avx2_transpose_size, nrows);
    const int B_columns = nstl::max(ncolumns - avx2_transpose_size, 0);
    add(src, src_B_offset);
    add(dst, dst_B_offset);
    jit_generator_t::transpose(src, dst, src_stride, dst_stride, B_rows,
            B_columns, dt, ymm_tmp, ymm_tail_mask, xmm_upper_tail_mask);

    const dim_t src_C_offset = src_stride * avx2_transpose_size;
    const dim_t dst_C_offset = dt_size * avx2_transpose_size;
    const int C_rows = nstl::max(nrows - avx2_transpose_size, 0);
    const int C_columns = nstl::min(avx2_transpose_size, ncolumns);
    add(src, -src_B_offset + src_C_offset);
    add(dst, -dst_B_offset + dst_C_offset);
    jit_generator_t::transpose(src, dst, src_stride, dst_stride, C_rows,
            C_columns, dt, ymm_tmp, ymm_tail_mask, xmm_upper_tail_mask);

    const dim_t src_D_offset
            = src_stride * avx2_transpose_size + dt_size * avx2_transpose_size;
    const dim_t dst_D_offset
            = dst_stride * avx2_transpose_size + dt_size * avx2_transpose_size;
    const int D_rows = nstl::max(nrows - avx2_transpose_size, 0);
    const int D_columns = nstl::max(ncolumns - avx2_transpose_size, 0);
    add(src, -src_C_offset + src_D_offset);
    add(dst, -dst_C_offset + dst_D_offset);
    jit_generator_t::transpose(src, dst, src_stride, dst_stride, D_rows,
            D_columns, dt, ymm_tmp, ymm_tail_mask, xmm_upper_tail_mask);
    sub(src, src_D_offset);
    sub(dst, dst_D_offset);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Ymm>::transpose_f16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    transpose_common_ymm(data_type::f16, dst, src, nrows, ncolumns);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Ymm>::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    transpose_common_ymm(data_type::bf16, dst, src, nrows, ncolumns);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Ymm>::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    transpose_common_ymm(data_type::f32, dst, src, nrows, ncolumns);
}

template <>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Xbyak::Zmm>::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;
    Label transpose_f32_done;
    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld;
    auto kmovw = [this, dynamic_column_size](
                         Opmask k, size_t q, bool load_mask = false) {
        if (dynamic_column_size && load_mask) {
            // reg_opmask_shift_compute is rcx, and we need cl for the shift
            mov(reg_opmask_shift_compute, reg_loop_m);
            mov(regq_tmp, 1);
            shl(regq_tmp, cl);
            sub(regq_tmp, 1);
        } else
            mov(regw_tmp, q);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovw(kTail, load_mask, true);

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [this, src, nrows, src_zmm](int i) {
        const auto addr = is_dynamic_src_ld
                ? ptr[i % 2 == 0 ? reg_aux_src0 : reg_aux_src1]
                : EVEX_compress_addr(src, i * src_stride);
        if (i < nrows) {
            if (use_fp16_instructions_)
                vcvtph2psx(src_zmm(i) | kTail | T_z, addr);
            else
                vmovups(src_zmm(i) | kTail | T_z, addr);
        } else {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }
    };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovups(addr, r | kTail);
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                if (is_dynamic_src_ld) {
                    mov(reg_aux_src0, src);
                    mov(reg_aux_src1, src);
                    add(reg_aux_src1, ptr[rsp + dynamic_src_ld_offt_]);
                }
                load(src_idx0);
                load(src_idx1);
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) {
                if (is_dynamic_src_ld)
                    add(reg_aux_src0, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                load(next_src_idx0);
            }
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) {
                if (is_dynamic_src_ld)
                    add(reg_aux_src1, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                load(next_src_idx1);
            }
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }

        // swap 2
        for (int i = 0; i < 4; i++) {
            int select_half = (i < 2) ? 0 : 2;
            int src_idx0 = base_idx + i + select_half + 0;
            int src_idx2 = src_idx0 + 2;

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx2);
            auto src0 = src_zmm(src_idx0);
            auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i;
            int src_idx4 = src_idx0 + 4;

            auto tmp0 = tmp_zmm(src_idx0);
            auto src0 = src_zmm(src_idx0);
            auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        const int store_mask = nrows < rows_step ? (1 << nrows) - 1 : 0xffff;
        kmovw(kTail, store_mask);

        // swap 8
        int columns_to_store = dynamic_column_size ? 8 : nstl::min(8, ncolumns);
        for (int i = 0; i < columns_to_store; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
            if (dynamic_column_size) {
                dec(reg_opmask_shift_compute);
                jz(transpose_f32_done, T_NEAR);
            }
        }

        columns_to_store = dynamic_column_size ? 8 : nstl::max(0, ncolumns - 8);
        for (int i = 0; i < columns_to_store; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
            if (dynamic_column_size) {
                dec(reg_opmask_shift_compute);
                jz(transpose_f32_done, T_NEAR);
            }
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
    L(transpose_f32_done);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::deploy_transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {

    if (std::is_same<Vmm, Xbyak::Ymm>::value) {
        if (is_f32 || use_fp16_instructions_)
            transpose_f32(dst, src, nrows, ncolumns);
        else if (is_bf16)
            transpose_bf16(dst, src, nrows, ncolumns);
        else if (is_f16)
            transpose_f16(dst, src, nrows, ncolumns);
        else
            assert(!"unsupported data type");
    } else {
        if (is_f32 || use_fp16_instructions_)
            transpose_f32(dst, src, nrows, ncolumns);
        else if (is_f8)
            transpose_f8(dst, src, nrows, ncolumns);
        else
            transpose_bf16(dst, src, nrows, ncolumns);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::init_masks() {
    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};
    if (is_superset(conf_->isa, avx512_core)) {
        if (is_f32) {
            kmovw(k3333, 0x3333); // 0011001100110011
            kmovw(k5555, 0x5555); // 0101010101010101
            kmovw(kAAAA, 0xaaaa); // 1010101010101010
            kmovw(kCCCC, 0xcccc); // 1100110011001100
            kmovw(k0F0F, 0x0f0f); // 0000111100001111
            kmovw(kF0F0, 0xf0f0); // 1111000011110000
        } else {
            kmovw(kFFFF, 0xffff);
            kmovw(k5555, 0x5555);
            kmovw(kAAAA, 0xaaaa);
            kmovw(kAA, 0xaa);
            kmovw(k55, 0x55);
            kmovw(kCC, 0xcc);
            kmovw(k33, 0x33);
        }
        if (!is_f32) {
            vmovdqa64(vidx1, idx1);
            vmovdqa64(vidx2, idx2);
            vmovdqa32(vidx3, idx3);
            vmovdqa32(vidx4, idx4);
            vmovdqa32(vidx5, (const int32_t *)idx5);
        }
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_a_transposed_impl_t<Vmm>::generate() {

    // only bf16, f16 and f32 supported for now
    if (!one_of(conf_->src_dt, data_type::bf16, data_type::f32, data_type::f16,
                data_type::f8_e5m2, data_type::f8_e4m3))
        return;
    preamble();
    sub(rsp, stack_space_needed_);
    mov(regq_tmp, ptr[param1 + GET_OFF(current_M_blk)]);
    mov(ptr[rsp + current_M_blk_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(src)]);
    mov(ptr[rsp + src_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(tr_src)]);
    mov(ptr[rsp + tr_src_offt_], regq_tmp);
    mov(regq_tmp, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(ptr[rsp + current_K_blk_offt_], regq_tmp);
    if (is_dynamic_src_ld) {
        // dynamic src_stride
        mov(regq_tmp, ptr[param1 + GET_OFF(dynamic_src_ld)]);
        mov(ptr[rsp + dynamic_src_ld_offt_], regq_tmp);

        // src_stride * 2
        shl(regq_tmp, 1);
        mov(ptr[rsp + dynamic_src_ld_x_2_offt_], regq_tmp);

        // src_stride * rows_step
        assert(rows_step == 16);
        shl(regq_tmp, 3);
        mov(ptr[rsp + dynamic_src_ld_x_kstep_offt_], regq_tmp);
    }

    if (!is_f8) init_masks();

    const int k_block_tail = conf_->K_blk % rows_step;
    const int last_k_block_tail = (conf_->K % conf_->K_blk) % rows_step;
    const int m_block_tail = conf_->M_blk % columns_step;
    const int last_m_block_tail = conf_->M_tail % columns_step;

    auto compute_m_loop
            = [&](reg64_t &reg_base, reg64_t &reg_tr_base, int nrows) {
        mov(reg_loop_m, ptr[rsp + current_M_blk_offt_]);
        mov(reg_m_src, reg_base);
        mov(reg_m_dst, reg_tr_base);

        Label m_loop_tail_or_done, m_loop, compute_m_loop_done;
        cmp(reg_loop_m, columns_step);
        jl(m_loop_tail_or_done, T_NEAR);

        L(m_loop);
        {
            deploy_transpose(reg_m_dst, reg_m_src, nrows, columns_step);
            add(reg_m_src, m_loop_src_shift);
            add(reg_m_dst, m_loop_dst_shift);
        }
        sub(reg_loop_m, columns_step);
        cmp(reg_loop_m, columns_step);
        jge(m_loop, T_NEAR);

        if (m_block_tail > 0 || last_m_block_tail > 0 || is_dynamic_src_ld)
            jz(compute_m_loop_done, T_NEAR);

        L(m_loop_tail_or_done);

        if (m_block_tail > 0) {
            Label m_block_tail_done;
            cmp(reg_loop_m, m_block_tail);
            jne(m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows, m_block_tail);
            jmp(compute_m_loop_done, T_NEAR);

            L(m_block_tail_done);
        }
        if (IMPLICATION(
                    last_m_block_tail <= 0 || last_m_block_tail == m_block_tail,
                    is_dynamic_src_ld)) {
            Label last_m_block_tail_done;
            cmp(reg_loop_m, 0);
            jle(last_m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows,
                    is_dynamic_src_ld ? 0 : last_m_block_tail);

            L(last_m_block_tail_done);
        }

        L(compute_m_loop_done);
    };

    auto compute_k_loop = [&]() {
        mov(reg_k_src, ptr[rsp + src_offt_]);
        mov(reg_k_dst, ptr[rsp + tr_src_offt_]);
        mov(reg_loop_k, ptr[rsp + current_K_blk_offt_]);

        Label k_tail_or_done, k_loop, compute_k_loop_done;
        cmp(reg_loop_k, rows_step);
        jl(k_tail_or_done, T_NEAR);

        L(k_loop);
        {
            compute_m_loop(reg_k_src, reg_k_dst, rows_step);
            if (is_dynamic_src_ld)
                add(reg_k_src, ptr[rsp + dynamic_src_ld_x_kstep_offt_]);
            else
                add(reg_k_src, k_loop_src_shift);
            add(reg_k_dst, k_loop_dst_shift);
        }
        sub(reg_loop_k, rows_step);
        cmp(reg_loop_k, rows_step);
        jge(k_loop, T_NEAR);

        if (k_block_tail > 0 || last_k_block_tail > 0)
            jz(compute_k_loop_done, T_NEAR);

        L(k_tail_or_done);

        if (k_block_tail > 0) {
            Label k_block_tail_done;
            cmp(reg_loop_k, k_block_tail);
            jne(k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(k_block_tail_done);
        }
        if (last_k_block_tail > 0 && last_k_block_tail != k_block_tail) {
            Label last_k_block_tail_done;
            cmp(reg_loop_k, last_k_block_tail);
            jne(last_k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, last_k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(last_k_block_tail_done);
        }

        L(compute_k_loop_done);
    };

    compute_k_loop();

    add(rsp, stack_space_needed_);
    postamble();
}

struct jit_brgemm_matmul_copy_a_transposed_int8_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_brgemm_matmul_copy_a_transposed_int8_impl_t)

    jit_brgemm_matmul_copy_a_transposed_int8_impl_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , jit_generator_t(jit_name())
        , src_stride_(conf_->copy_A_src_stride)
        , dst_stride_(conf_->LDA * conf_->tr_a_dt_sz)
        , m_loop_src_shift_(columns_step_ * conf_->a_dt_sz)
        , m_loop_dst_shift_(columns_step_ * dst_stride_)
        , k_loop_src_shift_(rows_step_ * src_stride_)
        , k_loop_dst_shift_(rows_step_ * conf_->tr_a_dt_sz)
        , has_vpermb_(cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
        , is_dynamic_src_ld_(conf_->is_runtime_M)
        , k_block_tail_(conf_->K_blk % rows_step_)
        , last_k_block_tail_((conf_->K % conf_->K_blk) % rows_step_)
        , m_block_tail_(conf_->M_blk % columns_step_)
        , last_m_block_tail_(conf_->M_tail % columns_step_)
        , do_compute_compensation_(conf_->has_zero_point_b) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    constexpr static int rows_step_ = 16;
    constexpr static int columns_step_ = 16;
    constexpr static int current_M_blk_offt_ = 0;
    constexpr static int current_K_blk_offt_ = 8;
    constexpr static int src_offt_ = 16;
    constexpr static int tr_src_offt_ = 24;
    constexpr static int dynamic_src_ld_offt_ = 32;
    constexpr static int dynamic_src_ld_x_2_offt_ = 40;
    constexpr static int dynamic_src_ld_x_kstep_offt_ = 48;
    constexpr static int stack_space_needed_ = 56;

    const dim_t src_stride_, dst_stride_;
    const dim_t m_loop_src_shift_, m_loop_dst_shift_;
    const dim_t k_loop_src_shift_, k_loop_dst_shift_;
    const bool has_vpermb_;
    const bool is_dynamic_src_ld_;

    const int k_block_tail_, last_k_block_tail_;
    const int m_block_tail_, last_m_block_tail_;

    const bool do_compute_compensation_;

    Opmask kFFFF_ = k1;
    Opmask k5555_ = k2;
    Opmask kAAAA_ = k3;
    Opmask kAA_ = k4;
    Opmask k55_ = k5;
    Opmask kCC_ = k6;
    Opmask k33_ = k7;
    Opmask kTail_ = k1;

    Reg64 reg_tmp_ = r15;
    Reg64 reg_k_src_ = r14;
    Reg64 reg_k_dst_ = r13;
    Reg64 reg_m_src_ = r12;
    Reg64 reg_m_dst_ = r11;
    Reg64 reg_aux_src0_ = r10;
    Reg64 reg_aux_src1_ = r9;
    Reg64 reg_zp_comp_res_ptr_ = r8;
    Reg64 reg_zp_comp_prev_data_ = rdx;
    Reg64 reg_loop_m_ = rbx;
    Reg64 reg_loop_k_ = rax;
    // Note: this must be assigned to rcx as it's used in shl instruction,
    // clashes with abi_param1 on Windows OS
    Reg64 reg_opmask_shift_compute_ = rcx;

    // Indices used in permutations
    Zmm zmm_idx_1_ = zmm31;
    Zmm zmm_idx_2_ = zmm30;
    Zmm zmm_idx_3_ = zmm29;
    Zmm zmm_idx_4_ = zmm28;

    // zmm_idx_5_ is used in vpermb implementation only
    // zmm_conversion_tmp_ is used in vpermw implementation only
    Zmm zmm_idx_5_ = zmm27;
    Zmm zmm_conversion_tmp_ = zmm27;

    // Required for zero-point
    Zmm zmm_comp_temp_ = zmm26;
    Zmm zmm_comp_acc_ = zmm25;

    Zmm get_zmm_src(int i) {
        assert(i >= 0 && i < columns_step_);
        return Zmm(i);
    }
    void kmovd(const bool dynamic_column_size, Opmask k, unsigned w,
            bool load_mask_stage = false) {
        if (dynamic_column_size && load_mask_stage) {
            // reg_opmask_shift_compute_ is rcx, and we need cl for the shift
            mov(reg_opmask_shift_compute_, reg_loop_m_);
            mov(reg_tmp_, 1);
            shl(reg_tmp_, cl);
            sub(reg_tmp_, 1);
        } else
            mov(reg_tmp_, w);
        jit_generator_t::kmovd(k, reg_tmp_.cvt32());
    }

    void transpose_int8_vpermb(Reg64 dst, Reg64 src, int nrows, int ncolumns);
    void transpose_int8_vpermw(Reg64 dst, Reg64 src, int nrows, int ncolumns);

    inline void deploy_transpose(
            Reg64 dst, Reg64 src, int nrows, int ncolumns) {
        if (has_vpermb_)
            transpose_int8_vpermb(dst, src, nrows, ncolumns);
        else
            transpose_int8_vpermw(dst, src, nrows, ncolumns);
    }

    void reset_compensation_accumulator() {
        if (do_compute_compensation_)
            uni_vpxor(zmm_comp_acc_, zmm_comp_acc_, zmm_comp_acc_);
    }
    void accumulate_compensation(Zmm zmm_copy) {
        if (do_compute_compensation_) {
            if (conf_->src_dt == data_type::s8)
                vpmovsxbd(zmm_comp_temp_, zmm_copy);
            else
                vpmovzxbd(zmm_comp_temp_, zmm_copy);
            vpaddd(zmm_comp_acc_, zmm_comp_acc_, zmm_comp_temp_);
        }
    }
    void save_partial_compensation() {
        if (do_compute_compensation_) {
            Label no_previous_data;
            test(reg_zp_comp_prev_data_, reg_zp_comp_prev_data_);
            je(no_previous_data);
            vpaddd(zmm_comp_acc_, zmm_comp_acc_, ptr[reg_zp_comp_res_ptr_]);
            L(no_previous_data);

            vmovups(ptr[reg_zp_comp_res_ptr_], zmm_comp_acc_);
            add(reg_zp_comp_res_ptr_, sizeof(int32_t) * 16);
        }
    }

    void compute_m_loop(int nrows);
    void compute_k_loop(bool is_first_K_iter, bool is_last_K_iter);
    void generate() override;
};

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::transpose_int8_vpermb(
        Reg64 dst, Reg64 src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step_ && ncolumns >= 0
            && ncolumns <= columns_step_);
    if (!nrows) return;

    auto load = [this, src](Zmm r, int i, Reg64 reg_aux) {
        auto addr = is_dynamic_src_ld_
                ? ptr[reg_aux]
                : EVEX_compress_addr(src, i * src_stride_);
        vmovdqu8(r | kFFFF_ | T_z, addr);
        accumulate_compensation(r);
    };

    auto store = [this, dst](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride_);
        vmovdqu8(addr, r | kTail_);
    };

    Label transpose_int8_done;

    reset_compensation_accumulator();

    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld_;
    const int load_mask
            = ncolumns < columns_step_ ? (1 << ncolumns) - 1 : 0xffff;
    kmovd(dynamic_column_size, kFFFF_, load_mask, true);

    // load rows and swap bytes
    for (int i = 0; i < nrows; i += 4) {
        auto idx0 = i;
        auto zmm_src0 = get_zmm_src(idx0);
        if (is_dynamic_src_ld_) {
            if (i == 0)
                mov(reg_aux_src0_, src);
            else
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        }
        load(zmm_src0, idx0, reg_aux_src0_);

        auto idx1 = i + 1;
        auto zmm_src1 = get_zmm_src(idx1);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx1 < nrows)
            load(zmm_src1, idx1, reg_aux_src0_);
        else
            vpxord(zmm_src1, zmm_src1, zmm_src1);

        auto idx2 = i + 2;
        auto zmm_src2 = get_zmm_src(idx2);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx2 < nrows)
            load(zmm_src2, idx2, reg_aux_src0_);
        else
            vpxord(zmm_src2, zmm_src2, zmm_src2);

        auto idx3 = i + 3;
        auto zmm_src3 = get_zmm_src(idx3);
        if (is_dynamic_src_ld_)
            add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_offt_]);
        if (idx3 < nrows)
            load(zmm_src3, idx3, reg_aux_src0_);
        else
            vpxord(zmm_src3, zmm_src3, zmm_src3);

        // concatenate 4 rows
        vinserti64x2(Ymm(zmm_src0.getIdx()), Ymm(zmm_src0.getIdx()),
                Xmm(zmm_src1.getIdx()), 1);
        vinserti64x2(Ymm(zmm_src2.getIdx()), Ymm(zmm_src2.getIdx()),
                Xmm(zmm_src3.getIdx()), 1);
        vinserti64x4(zmm_src0, zmm_src0, Ymm(zmm_src2.getIdx()), 1);

        // swap bytes
        vpermb(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    // swap doubles
    for (int i = 0; i < 2; i++) {
        auto idx0 = 8 * i;
        auto idx1 = idx0 + 4;

        auto zmm_src0 = get_zmm_src(idx0);
        auto zmm_src1 = get_zmm_src(idx1);

        auto zmm_tmp0 = get_zmm_src(idx0 + 1);
        auto zmm_tmp1 = get_zmm_src(idx1 + 1);

        vmovups(zmm_tmp0, zmm_idx_2_);
        vmovups(zmm_tmp1, zmm_idx_3_);

        vpermi2d(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2d(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // swap quads
    for (int i = 0; i < 2; i++) {
        auto idx0 = 4 * i;
        auto idx1 = idx0 + 8;

        auto zmm_src0 = get_zmm_src(idx0 + 1);
        auto zmm_src1 = get_zmm_src(idx1 + 1);

        auto zmm_tmp0 = get_zmm_src(idx0);
        auto zmm_tmp1 = get_zmm_src(idx1);

        vmovups(zmm_tmp0, zmm_idx_4_);
        vmovups(zmm_tmp1, zmm_idx_5_);

        vpermi2q(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2q(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // extract columns
    for (int i = 0; i < 16; i += 4) {
        vextracti64x4(
                Ymm(get_zmm_src(i + 2).getIdx()) | T_z, get_zmm_src(i), 1);
        vextracti32x4(
                Xmm(get_zmm_src(i + 1).getIdx()) | T_z, get_zmm_src(i), 1);
        vextracti32x4(Xmm(get_zmm_src(i + 3).getIdx()) | T_z,
                Ymm(get_zmm_src(i + 2).getIdx()), 1);
    }

    // store columns
    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step_ ? (1 << rows_to_store) - 1 : 0xffff;
    kmovd(dynamic_column_size, kTail_, store_mask);

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < columns_step_ && col_idx >= 0);

        const auto div = col_idx / 4;
        const auto mod = col_idx % 4;

        return mod * 4 + div;
    };

    const int columns_to_store = dynamic_column_size ? columns_step_ : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(get_zmm_src(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute_);
            jz(transpose_int8_done, T_NEAR);
        }
    }

    L(transpose_int8_done);

    save_partial_compensation();
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::transpose_int8_vpermw(
        Reg64 dst, Reg64 src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step_ && ncolumns >= 0
            && ncolumns <= columns_step_);
    if (!nrows) return;

    auto load = [this, src](Zmm r, int i, Reg64 reg_aux) {
        auto addr = is_dynamic_src_ld_
                ? ptr[reg_aux]
                : EVEX_compress_addr(src, i * src_stride_);
        vmovdqu8(zmm_conversion_tmp_ | kFFFF_ | T_z, addr);
        accumulate_compensation(zmm_conversion_tmp_);
        if (conf_->src_dt == data_type::s8)
            vpmovsxbw(r, zmm_conversion_tmp_);
        else
            vpmovzxbw(r, zmm_conversion_tmp_);
    };

    auto store = [this, dst](Zmm r, int i) {
        if (conf_->src_dt == data_type::s8)
            vpmovswb(Ymm(zmm_conversion_tmp_.getIdx()), r);
        else
            vpmovuswb(Ymm(zmm_conversion_tmp_.getIdx()), r);
        auto addr = EVEX_compress_addr(dst, i * dst_stride_);
        vmovdqu8(addr, zmm_conversion_tmp_ | kTail_);
    };

    Label transpose_int8_done;

    reset_compensation_accumulator();

    const bool dynamic_column_size = ncolumns == 0 && is_dynamic_src_ld_;
    const int load_mask
            = ncolumns < columns_step_ ? (1 << ncolumns) - 1 : 0xffff;
    kmovd(dynamic_column_size, kFFFF_, load_mask, true);

    for (int i = 0; i < nrows / 2; i++) {
        auto idx0 = 2 * i;
        auto idx1 = 2 * i + 1;
        auto zmm_src0 = get_zmm_src(idx0);
        auto zmm_src1 = get_zmm_src(idx1);
        if (is_dynamic_src_ld_) {
            if (i == 0) {
                mov(reg_aux_src0_, src);
                mov(reg_aux_src1_, src);
                add(reg_aux_src1_, ptr[rsp + dynamic_src_ld_offt_]);
            } else {
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
                add(reg_aux_src1_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
            }
        }

        load(zmm_src0, idx0, reg_aux_src0_);
        load(zmm_src1, idx1, reg_aux_src1_);

        vinserti64x4(zmm_src0, zmm_src0, Ymm(zmm_src1.getIdx()), 1);
        vpermw(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = get_zmm_src(2 * i);
        if (is_dynamic_src_ld_) {
            if (i == 0)
                mov(reg_aux_src0_, src);
            else
                add(reg_aux_src0_, ptr[rsp + dynamic_src_ld_x_2_offt_]);
        }

        load(zmm_src0, 2 * i, reg_aux_src0_);

        vpermw(zmm_src0, zmm_idx_1_, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < rows_step_; i += 2)
        vpxord(get_zmm_src(i), get_zmm_src(i), get_zmm_src(i));

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = get_zmm_src(4 * i);
        auto zmm1 = get_zmm_src(4 * i + 2);
        auto tmp0 = get_zmm_src(4 * i + 1);
        auto tmp1 = get_zmm_src(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA_, zmm_idx_2_, zmm1);
        vpermps(tmp1 | k5555_, zmm_idx_2_, zmm0);
    }

    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = get_zmm_src(base_idx + 2 * i + 1);
        auto zmm1 = get_zmm_src(base_idx + 2 * i + 5);

        auto tmp0 = get_zmm_src(base_idx + 2 * i);
        auto tmp1 = get_zmm_src(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA_, zmm_idx_3_, zmm1);
        vpermpd(tmp1 | k55_, zmm_idx_3_, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = get_zmm_src(base_idx + 2 * i + 1);
        auto zmm1 = get_zmm_src(base_idx + 2 * i + 5);

        auto tmp0 = get_zmm_src(base_idx + 2 * i);
        auto tmp1 = get_zmm_src(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA_, zmm_idx_3_, zmm1);
        vpermpd(tmp1 | k55_, zmm_idx_3_, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = get_zmm_src(2 * i);
        auto zmm1 = get_zmm_src(2 * i + 8);

        auto tmp0 = get_zmm_src(2 * i + 1);
        auto tmp1 = get_zmm_src(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC_, zmm_idx_4_, zmm1);
        vpermpd(tmp1 | k33_, zmm_idx_4_, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(Ymm(get_zmm_src(2 * i).getIdx()) | T_z,
                get_zmm_src(2 * i + 1), 1);

    const int rows_to_store = rnd_up(nrows, 2);
    const int store_mask
            = rows_to_store < rows_step_ ? (1 << rows_to_store) - 1 : 0xffff;
    kmovd(dynamic_column_size, kTail_, store_mask);

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < columns_step_ && col_idx >= 0);
        const int blk_sz = 4;
        const int blk_idx = col_idx / blk_sz;
        const int idx_within_blk = col_idx % blk_sz;

        // 0 1 2 3 -> 0 2 1 3
        const int mapped_blk_idx = 2 * blk_idx - (blk_idx / 2) * 3;
        // 0 1 2 3 -> 1 0 3 2
        const int mapped_idx_within_blk
                = idx_within_blk + 1 - 2 * (idx_within_blk % 2);
        return blk_sz * mapped_blk_idx + mapped_idx_within_blk;
    };

    const int columns_to_store = dynamic_column_size ? columns_step_ : ncolumns;
    for (int col_idx = 0; col_idx < columns_to_store; col_idx++) {
        store(get_zmm_src(get_vec_idx(col_idx)), col_idx);
        if (dynamic_column_size) {
            dec(reg_opmask_shift_compute_);
            jz(transpose_int8_done, T_NEAR);
        }
    }

    L(transpose_int8_done);

    save_partial_compensation();
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::compute_m_loop(
        int nrows) {
    mov(reg_loop_m_, ptr[rsp + current_M_blk_offt_]);
    mov(reg_m_src_, reg_k_src_);
    mov(reg_m_dst_, reg_k_dst_);

    if (do_compute_compensation_)
        mov(reg_zp_comp_res_ptr_,
                ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);

    Label m_loop_tail_or_done, m_loop, compute_m_loop_done;
    cmp(reg_loop_m_, columns_step_);
    jl(m_loop_tail_or_done, T_NEAR);

    L(m_loop);
    {
        deploy_transpose(reg_m_dst_, reg_m_src_, nrows, columns_step_);
        add(reg_m_src_, m_loop_src_shift_);
        add(reg_m_dst_, m_loop_dst_shift_);
    }
    sub(reg_loop_m_, columns_step_);
    cmp(reg_loop_m_, columns_step_);
    jge(m_loop, T_NEAR);

    if (m_block_tail_ > 0 || last_m_block_tail_ > 0 || is_dynamic_src_ld_)
        jz(compute_m_loop_done, T_NEAR);

    L(m_loop_tail_or_done);

    if (m_block_tail_ > 0) {
        Label m_block_tail_done;
        cmp(reg_loop_m_, m_block_tail_);
        jne(m_block_tail_done, T_NEAR);

        deploy_transpose(reg_m_dst_, reg_m_src_, nrows, m_block_tail_);
        jmp(compute_m_loop_done, T_NEAR);

        L(m_block_tail_done);
    }
    if (IMPLICATION(
                last_m_block_tail_ <= 0 || last_m_block_tail_ == m_block_tail_,
                is_dynamic_src_ld_)) {
        Label last_m_block_tail_done;
        cmp(reg_loop_m_, 0);
        jle(last_m_block_tail_done, T_NEAR);

        deploy_transpose(reg_m_dst_, reg_m_src_, nrows,
                is_dynamic_src_ld_ ? 0 : last_m_block_tail_);
        L(last_m_block_tail_done);
    }

    L(compute_m_loop_done);

    if (do_compute_compensation_) mov(reg_zp_comp_prev_data_, 1);
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::compute_k_loop(
        bool is_first_K_iter, bool is_last_K_iter) {
    mov(reg_k_src_, ptr[rsp + src_offt_]);
    mov(reg_k_dst_, ptr[rsp + tr_src_offt_]);
    mov(reg_loop_k_, ptr[rsp + current_K_blk_offt_]);

    if (do_compute_compensation_) {
        if (is_first_K_iter)
            mov(reg_zp_comp_prev_data_, 0);
        else
            mov(reg_zp_comp_prev_data_, 1);
    }

    Label k_tail_or_done, k_loop, compute_k_loop_done;
    cmp(reg_loop_k_, rows_step_);
    jl(k_tail_or_done, T_NEAR);

    L(k_loop);
    {
        compute_m_loop(rows_step_);
        if (is_dynamic_src_ld_)
            add(reg_k_src_, ptr[rsp + dynamic_src_ld_x_kstep_offt_]);
        else
            add(reg_k_src_, k_loop_src_shift_);
        add(reg_k_dst_, k_loop_dst_shift_);
    }
    sub(reg_loop_k_, rows_step_);
    cmp(reg_loop_k_, rows_step_);

    jge(k_loop, T_NEAR);

    if (k_block_tail_ > 0 || last_k_block_tail_ > 0)
        jz(compute_k_loop_done, T_NEAR);

    L(k_tail_or_done);

    if (k_block_tail_ > 0) {
        Label k_block_tail_done;
        cmp(reg_loop_k_, k_block_tail_);
        jne(k_block_tail_done, T_NEAR);

        compute_m_loop(k_block_tail_);
        jmp(compute_k_loop_done, T_NEAR);

        L(k_block_tail_done);
    }
    if (last_k_block_tail_ > 0 && last_k_block_tail_ != k_block_tail_) {
        Label last_k_block_tail_done;
        cmp(reg_loop_k_, last_k_block_tail_);
        jne(last_k_block_tail_done, T_NEAR);

        compute_m_loop(last_k_block_tail_);
        jmp(compute_k_loop_done, T_NEAR);

        L(last_k_block_tail_done);
    }

    L(compute_k_loop_done);

    if (do_compute_compensation_ && is_last_K_iter) {
        mov(reg_zp_comp_res_ptr_,
                ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);

        auto calculate_final_compensation = [this]() {
            // load accumulated compensation
            vmovups(zmm_comp_acc_, ptr[reg_zp_comp_res_ptr_]);

            // add -K * zp_a_val as mixed ab_compensation component
            if (conf_->src_zp_type != brgemm_broadcast_t::none) {
                mov(reg_tmp_, ptr[param1 + GET_OFF(zp_ab_comp_ptr)]);
                vbroadcastss(get_zmm_src(0), ptr[reg_tmp_]);
                vpaddd(zmm_comp_acc_, zmm_comp_acc_, get_zmm_src(0));
            }

            // multiply by zp_b_val
            mov(reg_tmp_, ptr[param1 + GET_OFF(zp_b_neg_val_ptr)]);
            vbroadcastss(get_zmm_src(0), ptr[reg_tmp_]);
            vpmulld(zmm_comp_acc_, zmm_comp_acc_, get_zmm_src(0));

            // store the final result value
            vmovups(ptr[reg_zp_comp_res_ptr_], zmm_comp_acc_);
            add(reg_zp_comp_res_ptr_, sizeof(int32_t) * 16);
        };

        Label m_loop, m_loop_tail_or_done, compute_m_loop_done;

        mov(reg_loop_m_, ptr[rsp + current_M_blk_offt_]);
        cmp(reg_loop_m_, columns_step_);
        jl(m_loop_tail_or_done, T_NEAR);

        L(m_loop);
        calculate_final_compensation();
        sub(reg_loop_m_, columns_step_);
        cmp(reg_loop_m_, columns_step_);
        jge(m_loop, T_NEAR);

        if (m_block_tail_ > 0 || last_m_block_tail_ > 0 || is_dynamic_src_ld_)
            jz(compute_m_loop_done, T_NEAR);

        L(m_loop_tail_or_done);

        if (m_block_tail_ > 0) {
            Label m_block_tail_done;
            cmp(reg_loop_m_, m_block_tail_);
            jne(m_block_tail_done, T_NEAR);

            calculate_final_compensation();
            jmp(compute_m_loop_done, T_NEAR);

            L(m_block_tail_done);
        }
        if (IMPLICATION(last_m_block_tail_ <= 0
                            || last_m_block_tail_ == m_block_tail_,
                    is_dynamic_src_ld_)) {
            Label last_m_block_tail_done;
            cmp(reg_loop_m_, 0);
            jle(last_m_block_tail_done, T_NEAR);

            calculate_final_compensation();
            L(last_m_block_tail_done);
        }

        L(compute_m_loop_done);
    }
}

void jit_brgemm_matmul_copy_a_transposed_int8_impl_t::generate() {
    preamble();
    sub(rsp, stack_space_needed_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(current_M_blk)]);
    mov(ptr[rsp + current_M_blk_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(src)]);
    mov(ptr[rsp + src_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(tr_src)]);
    mov(ptr[rsp + tr_src_offt_], reg_tmp_);
    mov(reg_tmp_, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(ptr[rsp + current_K_blk_offt_], reg_tmp_);
    if (is_dynamic_src_ld_) {
        // dynamic src_stride
        mov(reg_tmp_, ptr[param1 + GET_OFF(dynamic_src_ld)]);
        mov(ptr[rsp + dynamic_src_ld_offt_], reg_tmp_);

        // src_stride * 2
        shl(reg_tmp_, 1);
        mov(ptr[rsp + dynamic_src_ld_x_2_offt_], reg_tmp_);

        // src_stride * rows_step_
        assert(rows_step_ == 16);
        shl(reg_tmp_, 3);
        mov(ptr[rsp + dynamic_src_ld_x_kstep_offt_], reg_tmp_);
    }

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(reg_tmp_, w);
        jit_generator_t::kmovw(k, reg_tmp_.cvt32());
    };

    kmovw(kFFFF_, 0xffff);
    kmovw(k5555_, 0x5555);
    kmovw(kAAAA_, 0xaaaa);
    kmovw(kAA_, 0xaa);
    kmovw(k55_, 0x55);
    kmovw(kCC_, 0xcc);
    kmovw(k33_, 0x33);

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(reg_tmp_, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(z, ptr[reg_tmp_]);
    };

    if (has_vpermb_) {
        alignas(64) static constexpr const uint8_t idx1[64] = {0, 16, 32, 48, 1,
                17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21,
                37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41,
                57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45,
                61, 14, 30, 46, 62, 15, 31, 47, 63};
        alignas(64) static constexpr const uint32_t idx2[16]
                = {0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
        alignas(64) static constexpr const uint32_t idx3[16]
                = {1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
        alignas(64) static constexpr const uint64_t idx4[8]
                = {0, 8, 2, 10, 4, 12, 6, 14};
        alignas(64) static constexpr const uint64_t idx5[8]
                = {1, 9, 3, 11, 5, 13, 7, 15};

        vmovdqa64(zmm_idx_1_, (const int64_t *)idx1);
        vmovdqa64(zmm_idx_2_, (const int64_t *)idx2);
        vmovdqa64(zmm_idx_3_, (const int64_t *)idx3);
        vmovdqa64(zmm_idx_4_, (const int64_t *)idx4);
        vmovdqa64(zmm_idx_5_, (const int64_t *)idx5);
    } else {
        alignas(64) static constexpr const uint16_t idx1[32]
                = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1,
                        17, 3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};
        alignas(64) static constexpr const uint32_t idx2[16]
                = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
        alignas(64) static constexpr const uint64_t idx3[8]
                = {1, 0, 3, 2, 5, 4, 7, 6};
        alignas(64) static constexpr const uint64_t idx4[8]
                = {2, 3, 0, 1, 6, 7, 4, 5};

        vmovdqa64(zmm_idx_1_, (const int64_t *)idx1);
        vmovdqa64(zmm_idx_2_, (const int64_t *)idx2);
        vmovdqa64(zmm_idx_3_, (const int64_t *)idx3);
        vmovdqa64(zmm_idx_4_, (const int64_t *)idx4);
    }

    Label done;
    if (do_compute_compensation_) {
        assert(conf_->wei_zp_type == brgemm_broadcast_t::per_tensor);

        mov(reg_tmp_, ptr[param1 + GET_OFF(current_K_start)]);
        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_tmp_, 0);
        jne(not_first, T_NEAR);
        {
            Label first_not_last;
            cmp(reg_tmp_, last_K_threshold);
            jl(first_not_last, T_NEAR);
            compute_k_loop(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            compute_k_loop(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_tmp_, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        compute_k_loop(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }
    compute_k_loop(false, false);
    L(done);

    add(rsp, stack_space_needed_);
    postamble();
}
template struct jit_brgemm_matmul_copy_a_transposed_impl_t<Zmm>;
template struct jit_brgemm_matmul_copy_a_transposed_impl_t<Ymm>;

/** 
 * @brief Common class for BRGEMM B matrix copy operations
 * 
 * This class contains common methods and properties for all copy B kernels.
 * Now it consists of `load_value` and `decompress_reg` and it's considered
 * to contain all common methods for copy B kernels.
 * Default scenario for weight decompression is:
 * 1. load_value()
 * 2. apply zero point shift (if needed)
 * 3. convert to f32
 * 4. apply scaling (if needed)
 * 5. down convert if destination datatype is not f32.
 */
struct jit_brgemm_matmul_copy_b_common_t : public jit_brgemm_matmul_copy_b_t,
                                           public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_common_t)

    jit_brgemm_matmul_copy_b_common_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf), jit_generator_t(jit_name()) {}

protected:
    template <typename Vmm>
    Vmm maybe_mask(Vmm vmm, bool is_tail, bool is_int4 = false,
            bool skip_unmask = false) {
        // Transposed kernel uses the same kTail mask for both cases
        const auto tail_mask
                = is_int4 && !conf_->transposed_B ? kTail_4bit : kTail;
        const auto unmask_tail
                = one_of(conf_->wei_dt, data_type::bf16, data_type::f16)
                && !conf_->transposed_B && !skip_unmask;
        if (isa_has_masks(conf_->isa))
            return is_tail ? vmm | tail_mask | T_z
                    // bf16 and f16 requires masking for tail
                    // to avoid AVX512F issues with zeroing upper bits
                    // of ZMM registers when using vpmovzxwd/vpmovzxbd
                    // instructions
                    : unmask_tail ? vmm | kFFFF | T_z
                                  : vmm;
        else
            return vmm;
    }

    template <typename Vmm>
    Vmm maybe_mask_4bit(Vmm vmm, bool is_tail) {
        return maybe_mask(vmm, is_tail, true);
    }

    /**
    * @brief Inserts a YMM register into the upper half of a ZMM register
    * @param zmm Destination ZMM register where data will be inserted
    * @param ymm_half Source YMM register containing data for the upper half
    */
    void copy_half_reg(const Zmm &zmm, const Ymm &ymm_half) {
        vinserti64x4(zmm, zmm, ymm_half, 1);
    }

    /**
    * @brief Inserts an XMM register into the upper half of a YMM register
    * @param ymm Destination YMM register where data will be inserted
    * @param xmm_half Source XMM register containing data for the upper half
    */
    void copy_half_reg(const Ymm &ymm, const Xmm &xmm_half) {
        vinserti128(ymm, ymm, xmm_half, 1);
    }

    /** Restores vmm register containing permute indices for int4 processing.
    * Due to lack of registers this register is used as temporary in `prepare_loaded_int4`.
     */
    void restore_vmm_permd(const Ymm &vmm_permd) {
        alignas(64) static constexpr const uint32_t int4_permute_avx2[8]
                = {0, 4, 1, 5, 2, 6, 3, 7};
        const auto reg_tmp = r15;
        mov(reg_tmp, reinterpret_cast<size_t>(int4_permute_avx2));
        vmovdqa(vmm_permd, ptr[reg_tmp]);
    }

    /** Loaded in register(half of it) containing bytes (2 int4 values)
    * The idea is duplicate each byte in two dwords
    * using `copy_half_reg` and `vpermd`. Then shift bytes left or right
    * depending on the position (odd/even) and signed/unsigned type.
     */
    template <typename Vmm>
    void prepare_loaded_int4(
            const Vmm &reg, const Vmm &vmm_permd, const bool is_signed) {
        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
        const auto vmm_lower = Vmm_lower_t(reg.getIdx());

        copy_half_reg(reg, vmm_lower);
        vpermd(reg, vmm_permd, reg);
        // Without masks int4 is going to use 2 tmp registers
        // To perform masks using VPAND operator
        if (!isa_has_masks(conf_->isa)) {
            // TODO: Unify register usage over kernels
            const auto mask_vmm = Vmm(conf_->transposed_B ? 13 : 0);
            const auto tmp_vmm = vmm_permd;
            // f32 and transposed used the same register for regq_tmp
            const auto reg_tmp = r15;
            alignas(64) static constexpr const uint32_t odd_indices[8] = {
                    0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff};
            alignas(64) static constexpr const uint32_t even_indices[8] = {
                    0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0};
            // Process odd indices
            mov(reg_tmp, reinterpret_cast<size_t>(even_indices));
            vmovdqa(mask_vmm, ptr[reg_tmp]);
            uni_vpand(tmp_vmm, reg, mask_vmm);
            uni_vpslld(tmp_vmm, tmp_vmm, 28);
            if (is_signed) {
                vpsrad(tmp_vmm, tmp_vmm, 28);
            } else {
                vpsrld(tmp_vmm, tmp_vmm, 28);
            }

            // Process even indices
            mov(reg_tmp, reinterpret_cast<size_t>(odd_indices));
            vmovdqa(mask_vmm, ptr[reg_tmp]);
            uni_vpand(reg, reg, mask_vmm);
            if (is_signed) {
                vpsrad(reg, reg, 4);
            } else {
                vpsrld(reg, reg, 4);
            }
            // Store result to desired reg
            vpaddd(reg, reg, tmp_vmm);
            // Clean tmp regs
            vpxor(mask_vmm, mask_vmm, mask_vmm);
            restore_vmm_permd(vmm_permd);
        } else {
            uni_vpslld(reg | k5555, reg, 28);
            if (is_signed) {
                vpsrad(reg | k5555, reg, 28);
                vpsrad(reg | kAAAA, reg, 4);
            } else {
                vpsrld(reg | k5555, reg, 28);
                vpsrld(reg | kAAAA, reg, 4);
            }
        }
    }

    /**
    * @brief Loads and converts data of various types into vector registers with appropriate handling
    * Integer types s4, u4, s8, u8, s32 will be loaded and converted to s32 during the loading.
    * Floating point types f16, bf16, f32 will be loaded and converted to f32 during the loading.
    * @tparam Vmm Vector register type for computation
    * @param reg Destination vector register
    * @param op Source memory operand to load from
    * @param vmm_permd Vector register containing permutation indices for INT4 processing
    * @param dt Data type being loaded
    * @param is_tail Flag indicating if tail processing is needed
    * @param vmm_f4_lookup_table Vector register containing lookup table for FP4 conversion
    * (default is Vmm(4) for kernel jit_brgemm_matmul_copy_b_f32_t)
    */
    template <typename Vmm>
    void load_value(const Vmm &reg, const Xbyak::Operand &op,
            const Vmm &vmm_permd, data_type_t dt, bool is_tail = false,
            const Vmm &vmm_f4_lookup_table = Vmm(4)) {
        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
        const auto vmm_in = maybe_mask(reg, is_tail);
        const auto vmm_lower = Vmm_lower_t(vmm_in.getIdx());
        MAYBE_UNUSED(vmm_lower);

        const bool is_xf16
                = one_of(conf_->wei_dt, data_type::bf16, data_type::f16);

        switch (dt) {
            case data_type::s32: vmovdqu32(vmm_in, op); break;
            case data_type::f32: {
                if (conf_->transposed_B)
                    vmovdqu8(vmm_in, op);
                else
                    uni_vmovups(vmm_in, op);
                break;
            }
            case data_type::f16:
                if (!is_xf16) {
                    uni_vcvtph2psx(vmm_in, op);
                    break;
                }
            case data_type::bf16:
                if (is_xf16) {
                    uni_vmovdqu16(vmm_in, op);
                } else {
                    uni_vpmovzxwd(vmm_in, op);
                    uni_vpslld(vmm_in, vmm_in, 16);
                }
                break;
            case data_type::s8: uni_vpmovsxbd(vmm_in, op); break;
            case data_type::u8: uni_vpmovzxbd(vmm_in, op); break;
            // For int4, we see two int4 as one int8 and extend them int32
            // low half stores in lower bytes of vmm and high half in higher
            // bytes of vmm, then permute them into correct order
            // Finally, we process the extend bytes for s4/u4 accordingly
            case data_type::s4:
                uni_vpmovsxbd(maybe_mask_4bit(vmm_lower, is_tail), op);
                prepare_loaded_int4(reg, vmm_permd, /* is_signed = */ true);
                break;
            case data_type::u4:
                uni_vpmovzxbd(maybe_mask_4bit(vmm_lower, is_tail), op);
                prepare_loaded_int4(reg, vmm_permd, /* is_signed = */ false);
                break;
            case data_type::f4_e2m1:
                uni_vpmovzxbd(maybe_mask_4bit(vmm_lower, is_tail), op);
                copy_half_reg(reg, vmm_lower);
                vpermd(reg, vmm_permd, reg);
                uni_vpslld(reg | k5555, reg, 28);
                vpsrld(reg | k5555, reg, 28);
                vpsrld(reg | kAAAA, reg, 4);
                vpermps(reg, reg, vmm_f4_lookup_table);
                break;
            default: assert(!"unsupported data type");
        }
    }

    /** @brief Loads common zero point value and broadcasts over `zp_vmm` register.
    *   Handles only per_k and common values.
    *   @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    *   @param zp_vmm Vector register to load and broadcast zero point value into
    *   @param ptr_reg Register containing pointer to zero point value in memory
    **/
    template <typename Vmm>
    void load_common_zp_value(const Vmm &zp_vmm, const Xbyak::Reg64 &ptr_reg) {
        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
        // Handle only per_k and common values
        const bool only_per_k
                = conf_->is_wei_zp_per_k && !conf_->is_wei_zp_per_n;
        const bool require_load = conf_->has_zero_point_b
                && (conf_->is_wei_zp_common || only_per_k);
        if (!require_load) return;

        const auto zp_dt = conf_->wei_zp_dt;
        const auto tmp_xmm = Xmm(zp_vmm.getIdx());
        const auto vmm_lower = Vmm_lower_t(zp_vmm.getIdx());
        MAYBE_UNUSED(tmp_xmm);
        MAYBE_UNUSED(vmm_lower);
        const auto &addr = ptr[ptr_reg];

        const bool need_upconvert = one_of(zp_dt, data_type::s8, data_type::u8,
                data_type::u4, data_type::s4);

        if (need_upconvert) {
            uni_vpinsrb(tmp_xmm, tmp_xmm, addr, 0);
            if (one_of(zp_dt, data_type::s4, data_type::s8))
                uni_vpmovsxbd(tmp_xmm, tmp_xmm);
            else
                uni_vpmovzxbd(tmp_xmm, tmp_xmm);

            // For 4-bit int need to shift left on 28 bits
            if (one_of(zp_dt, data_type::s4, data_type::u4))
                uni_vpslld(tmp_xmm, tmp_xmm, 28);
            // Then shift back to the right on 28 bits
            if (zp_dt == data_type::u4) vpsrld(tmp_xmm, tmp_xmm, 28);
            if (zp_dt == data_type::s4) vpsrad(tmp_xmm, tmp_xmm, 28);
        }

        const auto &op = need_upconvert
                ? static_cast<const Xbyak::Operand &>(tmp_xmm)
                : static_cast<const Xbyak::Operand &>(addr);
        uni_vpbroadcastd(zp_vmm, op);
    }

    /** @brief Loads common scale value and broadcasts over `scale_vmm` register.
    *   Handles only per_k and common values.
    *   @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    *   @param scale_vmm Vector register to load and broadcast scale value into
    *   @param ptr_reg Register containing pointer to scale value in memory
    **/
    template <typename Vmm>
    void load_common_scale_value(
            const Vmm &scale_vmm, const Xbyak::Reg64 &ptr_reg) {
        const bool only_per_k
                = conf_->is_wei_scale_per_k && !conf_->is_wei_scale_per_n;
        const bool require_scales = conf_->apply_scales_in_buffer_b
                && (conf_->is_single_wei_scale() || only_per_k);
        if (!require_scales) return;

        const auto &scales_dt = conf_->wei_scales_dt;
        const auto &addr = ptr[ptr_reg];
        switch (scales_dt) {
            case data_type::f32: uni_vbroadcastss(scale_vmm, addr); break;
            case data_type::bf16:
                vpbroadcastw(scale_vmm, addr);
                uni_vpslld(scale_vmm, scale_vmm, 16);
                break;
            case data_type::f16: vcvtph2psx(scale_vmm, addr); break;
            case data_type::e8m0:
                vpbroadcastb(scale_vmm, addr);
                uni_vpmovzxbd(scale_vmm, Xbyak::Xmm(scale_vmm.getIdx()));
                uni_vpslld(scale_vmm, scale_vmm, 23);
                break;
            default: assert(!"unsupported wei_scales data type");
        }
    }

    /** @brief Helper method to load scales into vector register.
    *   Supports f32, bf16 and f16 data types.
    *   @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    *   @param vmm Vector register to load scale value into
    *   @param op Operand to load scale value from
    **/
    template <typename Vmm>
    void load_scale_value(const Vmm &vmm, const Xbyak::Operand &op,
            data_type_t dt, bool is_tail = false) {
        const auto masked_vmm = maybe_mask(vmm, is_tail);
        switch (dt) {
            case data_type::f32: uni_vmovups(masked_vmm, op); break;
            case data_type::bf16:
                uni_vpmovzxwd(masked_vmm, op);
                uni_vpslld(vmm, vmm, 16);
                break;
            case data_type::f16: vcvtph2ps(masked_vmm, op); break;
            case data_type::e8m0:
                uni_vpmovzxbd(masked_vmm, op);
                uni_vpslld(vmm, vmm, 23);
                break;
            default: assert(!"unsupported wei_scales data type");
        }
    }

    /**
    * @brief Applies zero point shift to vector register
    * Shifts input values by subtracting zero point values.
    * 
    * @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    * @param input Vector register containing input values to be shifted
    * @param zp Vector register containing zero point values to be subtracted
    * @param src_dt Source data type that determines which instruction to use
    */
    template <typename Vmm>
    void apply_shift(const Vmm &input, const Vmm &zp, data_type_t src_dt) {
        if (!conf_->has_zero_point_b) return;

        const bool is_int_shift = one_of(src_dt, data_type::s8, data_type::u8,
                data_type::s4, data_type::u4);
        const bool is_fp_shift = one_of(
                src_dt, data_type::bf16, data_type::f16, data_type::f32);

        if (is_int_shift)
            uni_vpsubd(input, input, zp);
        else if (is_fp_shift)
            uni_vsubps(input, input, zp);
        // should be unreachable
        else
            assert(!"Unable to shift input for zero-point");
    }

    /**
    * @brief Converts float8 data types to float16 format
    * 
    * @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    * @param dt Input data type of the values to be converted
    * @param vmm_in Vector register containing data to be converted
    * @param op Source memory operand to load from
    * @param is_tail Flag indicating if tail processing is needed
    * @param skip_unmask Flag indicating if unmask tail is needed
    */
    template <typename Vmm>
    void f8_to_f16_upconvert(data_type_t dt, const Vmm &vmm_in,
            const Xbyak::Operand &op, const bool is_tail = false,
            const bool skip_unmask = false) {
        if (!one_of(dt, data_type::f8_e4m3, data_type::f8_e5m2)) {
            assert(!"unsupported data type");
            return;
        }

        const auto vmm_masked = maybe_mask(vmm_in, is_tail, false, skip_unmask);
        if (dt == data_type::f8_e4m3)
            vcvthf82ph(vmm_masked, op);
        else {
            vpmovzxbw(vmm_masked, op);
            vpsllw(vmm_masked, vmm_masked, 8);
        }
    }

    /**
    * @brief Converts integer data types to floating-point format
    *
    * @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    * @param input Vector register containing data to be converted
    * @param src_dt Source data type of the values to be converted
    */
    template <typename Vmm>
    void upconvert_to_f32(const Vmm &input, data_type_t src_dt) {
        switch (src_dt) {
            case data_type::s8:
            case data_type::u8:
            case data_type::s4:
            case data_type::u4: uni_vcvtdq2ps(input, input); break;
            case data_type::bf16:
            case data_type::f16:
            case data_type::f32:
            case data_type::f4_e2m1:
                // bf16, f16, and f4 already converted into f32 while loading
                break;
            default: assert(!"Unsupported source data type for decompression");
        }
    }

    /**
    * @brief Applies scale factors to input values when configured
    * The operation is only performed if apply_scales_in_buffer_b flag is set 
    * in the configuration.
    * 
    * @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    * @param input Vector register containing values to be scaled
    * @param scale_op Operand containing scale factors to apply
    */
    template <typename Vmm>
    void apply_scales(const Vmm &input, const Xbyak::Operand &scale_op) {
        if (conf_->apply_scales_in_buffer_b) uni_vmulps(input, input, scale_op);
    }

    /**
    * @brief Converts floating-point data from F32 to specified destination data type
    *
    * @tparam Vmm Vector register type (Zmm, Ymm, etc.)
    * @param input Vector register containing F32 values to be converted
    * @param dst_dt Destination data type for conversion
    */
    template <typename Vmm>
    void downconvert_to_dst_dt(const Vmm &input, data_type_t dst_dt) {
        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
        switch (dst_dt) {
            case data_type::bf16:
                vcvtneps2bf16(Ymm(input.getIdx()), input);
                break;
            case data_type::f16:
                vcvtps2phx(Vmm_lower_t(input.getIdx()), input);
                break;
            case data_type::f32:
                // f32 is already in the correct format
                break;
            default:
                assert(!"Unsupported destination data type for decompression");
        }
    }

    template <typename Vmm>
    void downconvert_to_dst_dt(
            const Vmm &reg1, const Vmm &reg2, data_type_t dst_dt) {
        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
        switch (dst_dt) {
            case data_type::bf16: vcvtne2ps2bf16(reg1, reg2, reg1); break;
            case data_type::f16: {
                const auto src_vmm_lower0 = Vmm_lower_t(reg1.getIdx());
                const auto src_vmm_lower1 = Vmm_lower_t(reg2.getIdx());
                vcvtps2phx(src_vmm_lower0, reg1);
                vcvtps2phx(src_vmm_lower1, reg2);
                copy_half_reg(reg1, src_vmm_lower1);
                break;
            }
            case data_type::f32:
                // f32 is already in the correct format
                break;
            default:
                assert(!"Unsupported destination data type for decompression");
        }
    }

    /**
    * @brief Decompresses values by applying zero point shift, data type conversion, and scaling.
    * @tparam Vmm Vector register type used for computation
    * @param input Vector register containing input values to be decompressed
    * @param zp Vector register containing zero point values
    * @param scale_op Operand containing scales to be applied
    * @param src_dt Source data type of the values being decompressed
    * @param dst_dt Destination data type for the decompressed values
    */
    template <typename Vmm>
    void decompress_reg(const Vmm &input, const Vmm &zp,
            const Xbyak::Operand &scale_op, data_type_t src_dt) {
        if (src_dt == data_type::f32)
            return; // Decompression doesn't support f32
        apply_shift(input, zp, src_dt);
        upconvert_to_f32(input, src_dt);
        apply_scales(input, scale_op);
    }

    template <typename Vmm>
    void decompress_and_downcvt_reg(const Vmm &input, const Vmm &zp,
            const Xbyak::Operand &scale_op, data_type_t src_dt,
            data_type_t dst_dt) {
        if (src_dt == dst_dt) return;
        decompress_reg(input, zp, scale_op, src_dt);
        downconvert_to_dst_dt(input, dst_dt);
    }

    template <typename Vmm>
    void decompress_and_downcvt_2reg(const Vmm &input1, const Vmm &input2,
            const Vmm &zp1, const Vmm &zp2, const Xbyak::Operand &scale_op1,
            const Xbyak::Operand &scale_op2, data_type_t src_dt,
            data_type_t dst_dt) {
        if (src_dt == dst_dt) return;
        decompress_reg(input1, zp1, scale_op1, src_dt);
        decompress_reg(input2, zp2, scale_op2, src_dt);
        downconvert_to_dst_dt(input1, input2, dst_dt);
    }

    // Common used masks to permute data
    using opmask_t = const Xbyak::Opmask;
    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kFFFF = k6;
    opmask_t kTail = k7;
    opmask_t kTail_4bit = k5;

private:
    template <typename T>
    T maybe_mask_impl(T vmm, bool is_tail, Xbyak::Opmask tail_mask) {
        const auto unmask_tail
                = one_of(conf_->wei_dt, data_type::bf16, data_type::f16)
                && !conf_->transposed_B;
        if (isa_has_masks(conf_->isa))
            return is_tail        ? vmm | tail_mask | T_z
                    : unmask_tail ? vmm | kFFFF | T_z
                                  : vmm;
        else
            return vmm;
    }
};

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_common_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_int8_t)

    jit_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_common_t(conf)
        , src_stride_(conf->copy_B_wei_stride)
        , tr_src_stride_(conf->LDB * k_blk_step_ * sizeof(int8_t))
        , is_amx_(mayiuse(avx512_core_amx))
        , do_compute_compensation_(
                  conf->s8s8_compensation_required || conf->has_zero_point_a)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        , is_dynamic_stride_(is_runtime_value(src_stride_))
        , is_dynamic_N_(conf->is_runtime_N)
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , has_vpermb_(cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
        , comp_acc_idx_(is_ymm_ && is_src_int4_      ? 11
                          : is_src_int4_             ? 23
                          : is_ymm_                  ? 13
                          : avx512_core_dot_product_ ? 23
                                                     : 25)
        , src_elems_per_byte_(is_src_int4_ ? 2 : 1) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

protected:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr int k_blk_step_ = 4;
    static constexpr int n_blk_step_ = 64;
    static constexpr int blk_sz_ = 6;
    static constexpr int simd_w_ = vreg_traits_t<Vmm>::vlen;

    const dim_t src_stride_;
    const dim_t tr_src_stride_;
    const bool is_amx_;
    const bool do_compute_compensation_;
    const bool avx512_core_dot_product_;
    const bool is_dynamic_stride_;
    const bool is_dynamic_N_;
    const bool is_src_int4_;
    const bool has_vpermb_;
    const int comp_acc_idx_;
    const int src_elems_per_byte_;

    constexpr static int reg_src_offs_ = 0;
    constexpr static int reg_tr_src_offs_ = 8;
    constexpr static int reg_current_K_pad_offs_ = 16;
    constexpr static int reg_n_offs_ = 24;
    constexpr static int stack_space_needed_ = 32;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_comp_ptr = rdx;
    reg64_t reg_zp_comp_ptr = r11;
    reg64_t reg_zp_a_neg_val_ptr = r12;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_src_stride = r13;
    reg64_t reg_src_backup = r14;
    reg64_t reg_tmp = r15;

    reg64_t reg_copy_block_n_shift = rsi;

    reg64_t reg_dynamic_tail = rcx;
    Xbyak::Reg8 reg8_mask_shift = reg_dynamic_tail.cvt8();

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(24);
    Vmm vmm_dot_product_temp = Vmm(25);

    // ZMM stuff
    Vmm int4_permute_table = Vmm(24);
    Vmm vreg_idx_lo_256 = Vmm(26);
    Vmm vreg_idx_hi_256 = Vmm(27);
    Vmm vreg_idx_lo_128 = Vmm(28);
    Vmm vreg_idx_hi_128 = Vmm(29);

    // Shared
    Vmm vmm_comp_mul = Vmm(is_ymm_ ? 14 : 30);
    Vmm vmm_zero = Vmm(is_ymm_ ? 15 : 31);
    // Only used for decompressing int4 values
    Vmm vmm_tmp = Vmm(is_ymm_ ? 13 : 25);
    Vmm vmm_tmp1 = Vmm(12); // Only for Ymm

    Vmm get_comp_acc(int i) { return Vmm(comp_acc_idx_ - i); }
    Vmm get_vmm_zp_comp_res(int i) { return get_comp_acc(i); }
    Vmm get_vmm_wei_scale_comp_res(int i) { return Vmm(i); }

    inline void vmovdqa64(Vmm vmm, const void *addr) {
        mov(reg_tmp, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(vmm, ptr[reg_tmp]);
    }

    inline Vmm get_vmm(int blk, int idx) {
        if (idx < 0 || idx >= isa_num_vregs(is_ymm_ ? avx2 : avx512_core))
            assert(!"idx > vregs");
        assert(IMPLICATION(!is_ymm_, idx < blk_sz_ && blk >= 0));
        auto reg_idx = blk_sz_ * blk + idx;
        return Vmm(reg_idx);
    }
    inline void load(int blk, int i, bool is_tail) {}
    inline void kmovq(Opmask k, size_t q) {}
    virtual void init_permute() {}
    virtual void copy_block(
            int nrows, int ncolumns, bool n_tail, bool zeropad) {
        UNUSED(n_tail);
        copy_4x64(nrows, ncolumns, zeropad);
    }
    virtual void copy_4x64(int nrows, int ncolumns, bool zeropad) {}
    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3, get_encoding());
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }

    /**
    * @brief Applies sign extension for 4-bit signed integers stored in 8-bit lanes using AVX-512.
    * Due to lack of byte-wise right arithmetic shift(vpsrad) this method is used.
    * This method checks the sign bit (bit 3) of each byte and fills the upper nibble with 0xF0
    * if the sign bit is set, preserving the signed semantics after conversion from int4 to int8.
    *
    * Steps:
    * 1. Broadcast LUT (0xF0) across Zmm register.
    * 2. Isolate sign bit using left shift vpslld.
    * 3. XOR and shuffle with LUT to generate the correct fill pattern.
    * 4. OR the result back into the original vector.
    *
    * @param vmm_src Zmm register containing int8 values derived from int4.
    */
    inline void signed_mask_int4(const Zmm &vmm_src) {
        if (conf_->orig_wei_dt != data_type::s4) return;

        const auto &vmm_sign_extend = vmm_tmp;
        const auto &vmm_lut = vmm_zero;

        mov(reg_tmp, 0xF0);
        uni_vpbroadcastb(vmm_lut, reg_tmp.cvt8());

        uni_vmovups(vmm_sign_extend, vmm_src);
        vpslld(vmm_sign_extend, vmm_sign_extend, 4);

        uni_vpxor(vmm_sign_extend, vmm_sign_extend, vmm_lut);
        vpshufb(vmm_sign_extend, vmm_lut, vmm_sign_extend);
        vpord(vmm_src, vmm_src, vmm_sign_extend);
    }

    /**
    * @brief Applies sign extension for 4-bit signed integers stored in 8-bit lanes using AVX2.
    * Due to lack of byte-wise right arithmetic shift(vpsrad) this method is used.
    * This method uses a lookup table (LUT) and vpshufb to map sign bit presence to the correct
    * upper nibble fill (0xF0 for negative, 0x00 for positive).
    *
    * Steps:
    * 1. Load LUT into Ymm register.
    * 2. Isolate sign bit using left shift vpslld.
    * 3. Shuffle using LUT to replicate sign into upper nibble.
    * 4. OR the result back into the original vector.
    *
    * @param vmm_src Ymm register containing int8 values derived from int4.
    */
    inline void signed_mask_int4(const Ymm &vmm_src) {
        if (conf_->orig_wei_dt != data_type::s4) return;

        const auto &vmm_sign_extend = Ymm(vmm_tmp.getIdx());
        const auto &vmm_lut = Ymm(vmm_tmp1.getIdx());
        // No matter which index at lut, for signed (7bit==1)
        // vpshufb will select the same mask
        alignas(64) static const constexpr uint8_t lut[32] = {0xF0, 0xF0, 0xF0,
                0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

        mov(reg_tmp, (size_t)lut);
        vmovdqu(vmm_lut, ptr[reg_tmp]);

        uni_vmovups(vmm_sign_extend, vmm_src);
        vpslld(vmm_sign_extend, vmm_sign_extend, 4);

        uni_vpxor(vmm_sign_extend, vmm_sign_extend, vmm_lut);
        vpshufb(vmm_sign_extend, vmm_lut, vmm_sign_extend);
        vpor(vmm_src, vmm_src, vmm_sign_extend);
    }

    /**
    * @brief Converts packed u4/s4 integers to u8/s8 using AVX-512.
    *
    * This method splits the original packed int4 values into low and high nibbles,
    * expands them into separate bytes, and then applies(if needed) sign extension.
    *
    * Steps:
    * 1. Duplicate source into low and high vectors.
    * 2. Mask and shift high nibble, mask low nibble.
    * 3. Merge low and high into full bytes:
    *    - With AVX512_VBMI: vpermb byte permute.
    *    - Without VBMI: vpunpcklbw/vpunpckhbw + vpermt2d.
    * 4. Apply sign extension via signed_mask_int4().
    *
    * @param vmm_src Zmm register containing packed int4 values.
    */
    inline void cvt_int4_to_int8(const Zmm &vmm_src) {
        if (!is_src_int4_) return;

        using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;

        const auto &vmm_low = vmm_src; // Aliases for readability
        const auto &vmm_high = vmm_zero;
        const auto &vmm_mask = vmm_tmp;

        uni_vmovups(vmm_high, vmm_low);

        mov(reg_tmp, 0xF0);
        uni_vpbroadcastb(vmm_mask, reg_tmp.cvt8());
        uni_vpand(vmm_high, vmm_high, vmm_mask);
        vpsrld(vmm_high, vmm_high, 4);

        mov(reg_tmp, 0x0F);
        uni_vpbroadcastb(vmm_mask, reg_tmp.cvt8());
        uni_vpand(vmm_low, vmm_low, vmm_mask);

        if (has_vpermb_) {
            copy_half_reg(vmm_low, Vmm_lower_t(vmm_high.getIdx()));
            vpermb(vmm_src, int4_permute_table, vmm_src);
        } else {
            vpunpcklbw(vmm_mask, vmm_low, vmm_high);
            vpunpckhbw(vmm_low, vmm_low, vmm_high);
            vpermt2d(vmm_mask, int4_permute_table, vmm_low);
            uni_vmovups(vmm_src, vmm_mask);
        }
        signed_mask_int4(vmm_src);
        // Clean vmm_zero
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    /**
    * @brief Converts packed u4/s4 to u8/s8 integers using AVX2.
    *
    * This method extracts low and high nibbles from each byte, unpacks them into separate
    * bytes, and then applies(if needed) sign extension.
    *
    * Steps:
    * 1. Copy source into low and high vectors.
    * 2. Mask and shift high nibble, mask low nibble.
    * 3. Interleave low and high bytes using vpunpcklbw/vpunpckhbw.
    * 4. Apply sign extension via signed_mask_int4().
    *
    * @param vmm_src Ymm register containing packed int4 values.
    */
    inline void cvt_int4_to_int8(const Ymm &vmm_src) {
        if (!is_src_int4_) return;

        const auto vmm_out = Ymm(vmm_src.getIdx());
        const auto vmm_low = Ymm(vmm_tmp1.getIdx());
        const auto vmm_high = Ymm(vmm_zero.getIdx());
        const auto vmm_mask = Ymm(vmm_tmp.getIdx());

        uni_vmovups(vmm_low, vmm_out);
        uni_vmovups(vmm_high, vmm_out);

        alignas(64) static const constexpr uint8_t even_vector[32]
                = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                        0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                        0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                        0xF0, 0xF0, 0xF0, 0xF0};
        mov(reg_tmp, (size_t)even_vector);
        vmovdqu(vmm_mask, ptr[reg_tmp]);
        vpand(vmm_high, vmm_high, vmm_mask);
        vpsrld(vmm_high, vmm_high, 4);

        alignas(64) static const constexpr uint8_t odd_vector[32] = {0x0F, 0x0F,
                0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
                0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
                0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};
        mov(reg_tmp, (size_t)odd_vector);
        vmovdqu(vmm_mask, ptr[reg_tmp]);
        vpand(vmm_low, vmm_low, vmm_mask);

        vpunpcklbw(vmm_mask, vmm_low, vmm_high);
        vinserti128(vmm_out, vmm_out, Xmm(vmm_mask.getIdx()), 0);
        vpunpckhbw(vmm_mask, vmm_low, vmm_high);
        vinserti128(vmm_out, vmm_out, Xmm(vmm_mask.getIdx()), 1);
        signed_mask_int4(vmm_out);
        // Clean tmp regs
        uni_vpxor(vmm_low, vmm_low, vmm_low);
        uni_vpxor(vmm_high, vmm_high, vmm_high);
        uni_vpxor(vmm_mask, vmm_mask, vmm_mask);
    }

    void generate() override;
};

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<Zmm>::load(
        int blk, int i, bool is_tail) {
    auto vmm_src = get_vmm(blk, i % k_blk_step_);
    auto src_load = is_tail ? vmm_src | kTail | T_z : vmm_src;
    const auto offset
            = is_dynamic_stride_ ? 0 : i * src_stride_ / src_elems_per_byte_;
    const auto addr = maybe_EVEX_compress_addr(reg_src, offset);
    if (is_src_int4_) {
        const auto ymm_src_load = maybe_mask(Ymm(src_load.getIdx()), is_tail);
        vmovdqu8(ymm_src_load, addr);
        cvt_int4_to_int8(vmm_src);
    } else {
        vmovdqu8(src_load, addr);
    }
    if (is_dynamic_stride_) add(reg_src, reg_src_stride);
}

template <>
inline void jit_brgemm_matmul_copy_b_int8_t<Zmm>::kmovq(Opmask k, size_t q) {
    if (is_dynamic_N_) {
        mov(reg_tmp, 1);
        shl(reg_tmp, reg8_mask_shift /* reg_dynamic_tail.cvt8() == cl */);
        sub(reg_tmp, 1);
    } else
        mov(reg_tmp, q);
    jit_generator_t::kmovq(k, reg_tmp);
}

template struct jit_brgemm_matmul_copy_b_int8_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_int8_t<Ymm>;

struct jit_amx_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm> {

    jit_amx_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm>(conf) {}

private:
    void init_permute() override {
        alignas(64) static constexpr const uint8_t idx_lo_16[64] = {0, 1, 64,
                65, 4, 5, 68, 69, 2, 3, 66, 67, 6, 7, 70, 71, 8, 9, 72, 73, 12,
                13, 76, 77, 10, 11, 74, 75, 14, 15, 78, 79, 16, 17, 80, 81, 20,
                21, 84, 85, 18, 19, 82, 83, 22, 23, 86, 87, 24, 25, 88, 89, 28,
                29, 92, 93, 26, 27, 90, 91, 30, 31, 94, 95};

        alignas(64) static constexpr const uint8_t idx_hi_16[64] = {32, 33, 96,
                97, 36, 37, 100, 101, 34, 35, 98, 99, 38, 39, 102, 103, 40, 41,
                104, 105, 44, 45, 108, 109, 42, 43, 106, 107, 46, 47, 110, 111,
                48, 49, 112, 113, 52, 53, 116, 117, 50, 51, 114, 115, 54, 55,
                118, 119, 56, 57, 120, 121, 60, 61, 124, 125, 58, 59, 122, 123,
                62, 63, 126, 127};

        alignas(64) static constexpr const uint8_t idx_lo_8[64] = {0, 64, 2, 66,
                1, 65, 3, 67, 8, 72, 10, 74, 9, 73, 11, 75, 4, 68, 6, 70, 5, 69,
                7, 71, 12, 76, 14, 78, 13, 77, 15, 79, 16, 80, 18, 82, 17, 81,
                19, 83, 24, 88, 26, 90, 25, 89, 27, 91, 20, 84, 22, 86, 21, 85,
                23, 87, 28, 92, 30, 94, 29, 93, 31, 95};

        alignas(64) static constexpr const uint8_t idx_hi_8[64] = {32, 96, 34,
                98, 33, 97, 35, 99, 40, 104, 42, 106, 41, 105, 43, 107, 36, 100,
                38, 102, 37, 101, 39, 103, 44, 108, 46, 110, 45, 109, 47, 111,
                48, 112, 50, 114, 49, 113, 51, 115, 56, 120, 58, 122, 57, 121,
                59, 123, 52, 116, 54, 118, 53, 117, 55, 119, 60, 124, 62, 126,
                61, 125, 63, 127};

        vmovdqa64(vreg_idx_lo_256, (const void *)idx_lo_16);
        vmovdqa64(vreg_idx_hi_256, (const void *)idx_hi_16);
        vmovdqa64(vreg_idx_lo_128, (const void *)idx_lo_8);
        vmovdqa64(vreg_idx_hi_128, (const void *)idx_hi_8);

        if (is_src_int4_) {
            if (has_vpermb_) {
                alignas(64) static constexpr const uint8_t int4_permute[64] = {
                        0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                        8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46,
                        15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53,
                        22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60,
                        29, 61, 30, 62, 31, 63};
                vmovdqa64(int4_permute_table, (const void *)int4_permute);
            } else {
                alignas(64) static constexpr const uint32_t int4_permute_dw[16]
                        = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22,
                                23};
                vmovdqa64(int4_permute_table, (const void *)int4_permute_dw);
            }
        }
    }

    void copy_block(
            int nrows, int ncolumns, bool n_rem, bool zeropad) override {

        if (!is_dynamic_N_ || !n_rem) {
            copy_4x64(nrows, ncolumns, zeropad);
            return;
        }

        mov(reg_dynamic_tail, reg_N_blk);
        // dynamic tail processing: main loop with ncolumns = n_blk_step and
        // finally process tail < n_blk_step with dynamically computed mask
        // NOTE: for dynamic_stride case copy_4x64() shifts reg_src pointer
        // so we need to backup/restore its value for every iteration wrt n
        // except the last one

        mov(ptr[rsp + reg_tr_src_offs_], reg_tr_src);
        xor_(reg_copy_block_n_shift, reg_copy_block_n_shift);
        const dim_t typesize = sizeof(int8_t);

        Label loop_row_start, loop_row_tail, loop_row_done;
        cmp(reg_dynamic_tail, n_blk_step_);
        jl(loop_row_tail, T_NEAR);
        L(loop_row_start);
        {
            mov(ptr[rsp + reg_src_offs_], reg_src);
            add(reg_src, reg_copy_block_n_shift);
            copy_4x64(nrows, n_blk_step_, zeropad);
            add(reg_copy_block_n_shift,
                    n_blk_step_ * typesize / src_elems_per_byte_);
            add(reg_src, n_blk_step_ * typesize / src_elems_per_byte_);

            add(reg_tr_src, n_blk_step_ * k_blk_step_ * typesize);

            sub(reg_dynamic_tail, n_blk_step_);

            cmp(reg_dynamic_tail, 0);
            jle(loop_row_done, T_NEAR);

            mov(reg_src, ptr[rsp + reg_src_offs_]);

            cmp(reg_dynamic_tail, n_blk_step_);
            jl(loop_row_tail, T_NEAR);

            jmp(loop_row_start, T_NEAR);
        }

        L(loop_row_tail);
        {
            cmp(reg_dynamic_tail, 0);
            jle(loop_row_done, T_NEAR);

            add(reg_src, reg_copy_block_n_shift);
            copy_4x64(nrows, 1 /* to force tail case */, zeropad);
        }
        L(loop_row_done);

        // restore pointers
        sub(reg_src, reg_copy_block_n_shift);
        mov(reg_tr_src, ptr[rsp + reg_tr_src_offs_]);
    }

    void copy_4x64(int nrows, int ncolumns, bool zeropad) override {

        auto tr_src_off_n = [&](int n_elem) {
            return ((n_elem / conf_->LDB) * conf_->LDB2
                    + (n_elem % conf_->LDB) * k_blk_step_);
        };

        const bool is_tail = ncolumns < n_blk_step_;
        const auto tail_size = div_up(ncolumns, src_elems_per_byte_);
        const auto tail_mask = size_t(((size_t)1 << tail_size) - 1);

        if (is_tail) kmovq(kTail, tail_mask);

        const int max_unroll = (do_compute_compensation_ ? 21 : 25) / blk_sz_;

        for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step_); kb++)
        for (int k = 0; k < nstl::min(max_unroll,
                                div_up(nrows - kb * max_unroll * k_blk_step_,
                                        k_blk_step_));
                k++) {
            const int row_start = (kb * max_unroll + k) * k_blk_step_;
            const int row_end = nstl::min(row_start + k_blk_step_, nrows);
            dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride_;

            if (!zeropad) {
                for (int i = row_start; i < row_end; i++)
                    load(k, i, is_tail);
                if (row_end == nrows && nrows % k_blk_step_ > 0) {
                    for (int i = nrows; i < rnd_up(nrows, k_blk_step_); i++) {
                        auto src_reg = get_vmm(k, i % k_blk_step_);
                        vpxord(src_reg, src_reg, src_reg);
                    }
                }
                vmovups(get_vmm(k, 4), vreg_idx_lo_256);
                vpermi2b(get_vmm(k, 4), get_vmm(k, 0), get_vmm(k, 2));
                vmovups(get_vmm(k, 5), vreg_idx_hi_256);
                vpermi2b(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 2));
                vmovups(get_vmm(k, 0), vreg_idx_lo_256);
                vpermi2b(get_vmm(k, 0), get_vmm(k, 1), get_vmm(k, 3));
                vmovups(get_vmm(k, 2), vreg_idx_hi_256);
                vpermi2b(get_vmm(k, 2), get_vmm(k, 1), get_vmm(k, 3));

                vmovups(get_vmm(k, 1), vreg_idx_lo_128);
                vpermi2b(get_vmm(k, 1), get_vmm(k, 4), get_vmm(k, 0));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                        get_vmm(k, 1));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(0), vmm_comp_mul, get_vmm(k, 1));
            } else {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                        vmm_zero);
            }
            const bool dynamic_tail = is_dynamic_N_ && ncolumns < n_blk_step_;

            Label k_loop_done;
            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 16);
                jle(k_loop_done, T_NEAR);
            }
            if (!zeropad && (ncolumns > 16 || dynamic_tail)) {
                vmovups(get_vmm(k, 3), vreg_idx_hi_128);
                vpermi2b(get_vmm(k, 3), get_vmm(k, 4), get_vmm(k, 0));
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(16)),
                        get_vmm(k, 3));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(1), vmm_comp_mul, get_vmm(k, 3));
            } else if (conf_->wei_n_blk > 16) {
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(16)),
                        vmm_zero);
            }

            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 32);
                jle(k_loop_done, T_NEAR);
            }
            if (!zeropad && (ncolumns > 32 || dynamic_tail)) {
                vmovups(get_vmm(k, 4), vreg_idx_lo_128);
                vpermi2b(get_vmm(k, 4), get_vmm(k, 5), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(32)),
                        get_vmm(k, 4));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(2), vmm_comp_mul, get_vmm(k, 4));
            } else if (conf_->wei_n_blk > 32) {
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(32)),
                        vmm_zero);
            }

            if (dynamic_tail) {
                cmp(reg_dynamic_tail, 48);
                jle(k_loop_done, T_NEAR);
            }
            if (!zeropad && (ncolumns > 48 || dynamic_tail)) {
                vmovups(get_vmm(k, 0), vreg_idx_hi_128);
                vpermi2b(get_vmm(k, 0), get_vmm(k, 5), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(48)),
                        get_vmm(k, 0));
                if (do_compute_compensation_)
                    vpdpbusd(get_comp_acc(3), vmm_comp_mul, get_vmm(k, 0));
            } else if (conf_->wei_n_blk > 48) {
                vmovups(EVEX_compress_addr(
                                reg_tr_src, tr_src_off_base + tr_src_off_n(48)),
                        vmm_zero);
            }
            L(k_loop_done);
        }
    }
};

struct jit_avx512_core_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm> {

    jit_avx512_core_brgemm_matmul_copy_b_int8_t(
            const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Zmm>(conf) {}

private:
    void init_permute() override {
        alignas(64) static constexpr const int64_t idx_lo_256[8]
                = {0, 1, 2, 3, 8, 9, 10, 11};
        alignas(64) static constexpr const int64_t idx_hi_256[8]
                = {4, 5, 6, 7, 12, 13, 14, 15};

        alignas(64) static constexpr const int64_t idx_lo_128[8]
                = {0, 1, 8, 9, 4, 5, 12, 13};
        alignas(64) static constexpr const int64_t idx_hi_128[8]
                = {2, 3, 10, 11, 6, 7, 14, 15};

        vmovdqa64(vreg_idx_lo_256, (const void *)idx_lo_256);
        vmovdqa64(vreg_idx_hi_256, (const void *)idx_hi_256);
        vmovdqa64(vreg_idx_lo_128, (const void *)idx_lo_128);
        vmovdqa64(vreg_idx_hi_128, (const void *)idx_hi_128);

        if (is_src_int4_) {
            if (has_vpermb_) {
                alignas(64) static constexpr const uint8_t int4_permute[64] = {
                        0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
                        8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46,
                        15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53,
                        22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60,
                        29, 61, 30, 62, 31, 63};
                vmovdqa64(int4_permute_table, (const void *)int4_permute);
            } else {
                alignas(64) static constexpr const uint32_t int4_permute_dw[16]
                        = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22,
                                23};
                vmovdqa64(int4_permute_table, (const void *)int4_permute_dw);
            }
        }
    }

    void copy_4x64(int nrows, int ncolumns, bool zeropad) override {
        const bool is_tail = ncolumns < n_blk_step_;
        if (is_tail) {
            const auto tail_size = div_up(ncolumns, src_elems_per_byte_);
            const auto tail_mask = size_t(((size_t)1 << tail_size) - 1);
            kmovq(kTail, tail_mask);
        }

        const int max_unroll = (do_compute_compensation_ ? 21 : 25) / blk_sz_;

        for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step_); kb++)
        for (int k = 0; k < nstl::min(max_unroll,
                                div_up(nrows - kb * max_unroll * k_blk_step_,
                                        k_blk_step_));
                k++) {
            const int row_start = (kb * max_unroll + k) * k_blk_step_;
            const int row_end = nstl::min(row_start + k_blk_step_, nrows);
            dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride_;

            if (!zeropad) {
                for (int i = row_start; i < row_end; i++)
                    load(k, i, is_tail);
                if (row_end == nrows && nrows % k_blk_step_ > 0) {
                    for (int i = nrows; i < rnd_up(nrows, k_blk_step_); i++) {
                        auto src_reg = get_vmm(k, i % k_blk_step_);
                        vpxord(src_reg, src_reg, src_reg);
                    }
                }

                vpunpcklbw(get_vmm(k, 4), get_vmm(k, 0), get_vmm(k, 1));
                vpunpckhbw(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 1));
                vpunpcklbw(get_vmm(k, 0), get_vmm(k, 2), get_vmm(k, 3));
                vpunpckhbw(get_vmm(k, 1), get_vmm(k, 2), get_vmm(k, 3));

                vpunpcklwd(get_vmm(k, 2), get_vmm(k, 4), get_vmm(k, 0));
                vpunpckhwd(get_vmm(k, 3), get_vmm(k, 4), get_vmm(k, 0));
                vpunpcklwd(get_vmm(k, 4), get_vmm(k, 5), get_vmm(k, 1));
                vpunpckhwd(get_vmm(k, 5), get_vmm(k, 5), get_vmm(k, 1));

                vmovups(get_vmm(k, 0), vreg_idx_lo_256);
                vpermi2q(get_vmm(k, 0), get_vmm(k, 2), get_vmm(k, 4));
                vmovups(get_vmm(k, 1), vreg_idx_hi_256);
                vpermi2q(get_vmm(k, 1), get_vmm(k, 2), get_vmm(k, 4));
                vmovups(get_vmm(k, 2), vreg_idx_lo_256);
                vpermi2q(get_vmm(k, 2), get_vmm(k, 3), get_vmm(k, 5));
                vmovups(get_vmm(k, 4), vreg_idx_hi_256);
                vpermi2q(get_vmm(k, 4), get_vmm(k, 3), get_vmm(k, 5));

                vmovups(get_vmm(k, 3), vreg_idx_lo_128);
                vpermi2q(get_vmm(k, 3), get_vmm(k, 0), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                        get_vmm(k, 3));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(0), vmm_comp_mul, get_vmm(k, 3));
            } else {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base),
                        vmm_zero);
            }

            if (!zeropad && ncolumns > 16) {
                vmovups(get_vmm(k, 5), vreg_idx_hi_128);
                vpermi2q(get_vmm(k, 5), get_vmm(k, 0), get_vmm(k, 2));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        get_vmm(k, 5));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(1), vmm_comp_mul, get_vmm(k, 5));
            } else if (conf_->wei_n_blk > 16) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                        vmm_zero);
            }

            if (!zeropad && ncolumns > 32) {
                vmovups(get_vmm(k, 0), vreg_idx_lo_128);
                vpermi2q(get_vmm(k, 0), get_vmm(k, 1), get_vmm(k, 4));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        get_vmm(k, 0));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(2), vmm_comp_mul, get_vmm(k, 0));
            } else if (conf_->wei_n_blk > 32) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                        vmm_zero);
            }

            if (!zeropad && ncolumns > 48) {
                vmovups(get_vmm(k, 2), vreg_idx_hi_128);
                vpermi2q(get_vmm(k, 2), get_vmm(k, 1), get_vmm(k, 4));
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        get_vmm(k, 2));
                if (do_compute_compensation_)
                    dot_product(get_comp_acc(3), vmm_comp_mul, get_vmm(k, 2));
            } else if (conf_->wei_n_blk > 48) {
                vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                        vmm_zero);
            }
        }
    }
};

struct jit_avx2_vnni_brgemm_matmul_copy_b_int8_t
    : public jit_brgemm_matmul_copy_b_int8_t<Xbyak::Ymm> {

    jit_avx2_vnni_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_int8_t<Xbyak::Ymm>(conf) {}

private:
    static constexpr int perm2i128_l
            = 0x20; // dst[127:0]=src1_low_128; dst[128:255]=src2_low_128
    static constexpr int perm2i128_h
            = 0x31; // dst[127:0]=src1_hi_128; dst[128:255]=src2_hi_128

    Xbyak::Ymm get_ymm(int idx) { return get_vmm(0, idx); }

    void load_ymm(int ymm_idx, size_t offset, bool is_tail, int tail_sz) {
        Xbyak::Ymm vmm_src = Xbyak::Ymm(ymm_idx);
        if (is_tail) {
            load_bytes(vmm_src, reg_src, offset, tail_sz);
        } else if (is_src_int4_) {
            load_bytes(vmm_src, reg_src, offset, simd_w_ / src_elems_per_byte_);
        } else
            uni_vmovups(vmm_src, ptr[reg_src + offset]);
        cvt_int4_to_int8(vmm_src);
    }

    void copy_4x64(int nrows, int ncolumns, bool zeropad) override {
        const bool is_tail = ncolumns < n_blk_step_;
        const int k_end = div_up(nrows, k_blk_step_);
        for_(int k = 0; k < k_end; k++)
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 0 && ncolumns >= simd_w_) mov(reg_src_backup, reg_src);
            assert(one_of(pass, 0, 1));
            const dim_t tr_src_off_base = k * tr_src_stride_;
            const int set_1_tr_src_offset = static_cast<int>(
                    tr_src_off_base + pass * 2 * n_blk_step_);
            const int row_start = k * k_blk_step_;
            const int row_end = nstl::min(row_start + k_blk_step_, nrows);
            if (!zeropad) {
                for (int i = row_start; i < rnd_up(row_end, k_blk_step_); i++) {
                    const bool do_load = i < row_end
                            && IMPLICATION(pass == 1, ncolumns >= simd_w_);
                    if (do_load) {
                        const bool do_tail = is_tail
                                && IMPLICATION(pass == 0, ncolumns < simd_w_);
                        auto offset = (is_dynamic_stride_ ? 0 : i * src_stride_)
                                + pass * simd_w_;
                        offset /= src_elems_per_byte_;
                        const auto tail_size
                                = div_up((ncolumns - pass * simd_w_),
                                        src_elems_per_byte_);
                        load_ymm(i % 4, offset, do_tail, tail_size);
                        if (is_dynamic_stride_) add(reg_src, reg_src_stride);
                    } else {
                        const auto src_ymm_1 = get_ymm(i % 4);
                        uni_vpxor(src_ymm_1, src_ymm_1, src_ymm_1);
                    }
                }
                if (pass == 0 && ncolumns >= simd_w_)
                    mov(reg_src, reg_src_backup);

                vpunpcklbw(get_ymm(4), get_ymm(0), get_ymm(1));
                vpunpckhbw(get_ymm(5), get_ymm(0), get_ymm(1));
                vpunpcklbw(get_ymm(0), get_ymm(2), get_ymm(3));
                vpunpckhbw(get_ymm(1), get_ymm(2), get_ymm(3));

                vpunpcklwd(get_ymm(2), get_ymm(4), get_ymm(0));
                vpunpckhwd(get_ymm(3), get_ymm(4), get_ymm(0));
                vpunpcklwd(get_ymm(4), get_ymm(5), get_ymm(1));
                vpunpckhwd(get_ymm(5), get_ymm(5), get_ymm(1));
            }

            auto get_accum
                    = [&](int idx) { return get_comp_acc(idx + pass * 4); };

            if (!zeropad
                    && IMPLICATION(pass == 1,
                            ncolumns > 32)) { // check against {0, 32}
                vperm2i128(get_ymm(0), get_ymm(2), get_ymm(3), perm2i128_l);
                vperm2i128(get_ymm(1), get_ymm(4), get_ymm(5), perm2i128_l);
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset], get_ymm(0));
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset + simd_w_],
                        get_ymm(1));
                if (do_compute_compensation_) {
                    vpdpbusd(get_accum(0), vmm_comp_mul, get_ymm(0),
                            VexEncoding);
                    vpdpbusd(get_accum(1), vmm_comp_mul, get_ymm(1),
                            VexEncoding);
                }
            } else if (conf_->wei_n_blk > 32) {
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset], vmm_zero);
                uni_vmovups(ptr[reg_tr_src + set_1_tr_src_offset + simd_w_],
                        vmm_zero);
            }

            const int set_2_tr_src_offset = set_1_tr_src_offset + n_blk_step_;
            const int upper_check = 16 + pass * 32; // check against {16, 48}
            if (!zeropad && ncolumns > upper_check) {
                vperm2i128(get_ymm(2), get_ymm(2), get_ymm(3), perm2i128_h);
                vperm2i128(get_ymm(3), get_ymm(4), get_ymm(5), perm2i128_h);
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset], get_ymm(2));
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset + simd_w_],
                        get_ymm(3));
                if (do_compute_compensation_) {
                    vpdpbusd(get_accum(2), vmm_comp_mul, get_ymm(2),
                            VexEncoding);
                    vpdpbusd(get_accum(3), vmm_comp_mul, get_ymm(3),
                            VexEncoding);
                }
            } else if (conf_->wei_n_blk > upper_check) {
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset], vmm_zero);
                uni_vmovups(ptr[reg_tr_src + set_2_tr_src_offset + simd_w_],
                        vmm_zero);
            }
        }
    }
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_int8_t<Vmm>::generate() {
    preamble();
    sub(rsp, stack_space_needed_);

    if (avx512_core_dot_product_) {
        mov(reg_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, reg_tmp.cvt16());
    }

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    if (is_dynamic_stride_) {
        mov(reg_src_stride, ptr[param1 + GET_OFF(dynamic_src_stride)]);
    }

    init_permute();

    auto compute_K_loop_body = [&](const reg64_t &reg_K, int ncolumns,
                                       bool is_N_rem, bool zeropad) {
        const int k_unroll = 4;
        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K, k_unroll * k_blk_step_);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step_, ncolumns, is_N_rem, zeropad);
        if (!zeropad && !is_dynamic_stride_)
            add(reg_src,
                    k_unroll * k_blk_step_ * src_stride_ / src_elems_per_byte_);
        add(reg_tr_src, k_unroll * tr_src_stride_);

        sub(reg_K, k_unroll * k_blk_step_);
        cmp(reg_K, k_unroll * k_blk_step_);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step_, ncolumns, is_N_rem, zeropad);
        if (!zeropad && !is_dynamic_stride_)
            add(reg_src, k_blk_step_ * src_stride_ / src_elems_per_byte_);
        add(reg_tr_src, tr_src_stride_);

        sub(reg_K, k_blk_step_);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step_;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K, 0);
            jle(K_loop_done, T_NEAR);

            copy_block(k_blk_tail, ncolumns, is_N_rem, zeropad);
            add(reg_tr_src, tr_src_stride_);
            sub(reg_K, k_blk_tail);
            L(K_loop_done);
        }
    };

    auto compute_K_loop = [&](bool is_N_rem, int ncolumns) {
        // 'param1' register (rcx on Windows) re-written in compute_K_loop_body
        // so we need to read and keep 'current_K_pad' parameter in stack before
        // the call
        mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_pad)]);
        mov(ptr[rsp + reg_current_K_pad_offs_], reg_K_iters);
        mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
        compute_K_loop_body(reg_K_iters, ncolumns, is_N_rem, false);
        mov(reg_K_iters, ptr[rsp + reg_current_K_pad_offs_]);
        compute_K_loop_body(reg_K_iters, ncolumns, is_N_rem, true);
    };

    const int wei_n_blk = static_cast<int>(conf_->wei_n_blk);

    auto compute_N_loop_body = [&](int ncolumns) {
        mov(reg_tmp, ptr[rsp + reg_n_offs_]);

        mov(reg_src, ptr[param1 + GET_OFF(src)]);
        add(reg_src, reg_tmp);
        mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);

        const int tr_src_scale = static_cast<int>(conf_->LDB2) / wei_n_blk;
        imul(reg_tmp, reg_tmp, tr_src_scale);
        add(reg_tr_src, reg_tmp);

        if (!is_dynamic_N_) mov(reg_N_blk, ncolumns);

        if (do_compute_compensation_) {
            int n_iters = div_up(conf_->wei_n_blk, 16) * (is_ymm_ ? 2 : 1);
            for (int i = 0; i < n_iters; i++)
                uni_vpxor(get_comp_acc(i), get_comp_acc(i), get_comp_acc(i));
            mov(reg_tmp, 1);
            uni_vpbroadcastb(vmm_comp_mul, reg_tmp.cvt8());
        }

        Label done;
        mov(reg_tmp, ptr[param1 + GET_OFF(current_N_blk)]);
        cmp(reg_tmp, 0);
        jle(done, T_NEAR);

        if (is_dynamic_N_) {
            Label main_N_blk;
            cmp(reg_N_blk, conf_->N_blk);
            je(main_N_blk, T_NEAR);
            compute_K_loop(true, ncolumns);
            jmp(done, T_NEAR);

            L(main_N_blk);
        }

        compute_K_loop(false, ncolumns);
        L(done);

        if (do_compute_compensation_) {
            const bool req_s8s8_comp = conf_->s8s8_compensation_required;
            const bool req_zp_comp = conf_->has_zero_point_a;
            int n_iters = div_up(conf_->wei_n_blk, 16);
            assert(IMPLICATION(req_zp_comp,
                    conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

            if (req_s8s8_comp) {
                mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
                // Runtime offset: reg_comp_ptr += n * sizeof(int32_t)
                // (n is the outer-N counter spilled to [rsp + reg_n_offs_]).
                mov(reg_tmp, ptr[rsp + reg_n_offs_]);
                shl(reg_tmp, 2); // * sizeof(int32_t) == * 4
                add(reg_comp_ptr, reg_tmp);
            }
            if (req_zp_comp) {
                mov(reg_zp_comp_ptr,
                        ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);
                // Runtime offset: reg_zp_comp_ptr += n * sizeof(int32_t).
                mov(reg_tmp, ptr[rsp + reg_n_offs_]);
                shl(reg_tmp, 2);
                add(reg_zp_comp_ptr, reg_tmp);
            }
            mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);

            // YMM Note: 16 vmm registers would be needed, so only compute by halves
            const bool do_outer_unroll = req_s8s8_comp;
            const int outer_unroll = is_ymm_ && do_outer_unroll ? 2 : 1;
            const int inner_unroll = is_ymm_ && (!do_outer_unroll) ? 2 : 1;
            for (int out_ur = 0; out_ur < outer_unroll; ++out_ur) {

                // copy 'comp_acc' into s8s8_comp accumulator
                if (req_s8s8_comp) {
                    for (int i = 0; i < n_iters; i++) {
                        const int accum_idx = i + out_ur * n_iters;
                        uni_vmovups(get_vmm_wei_scale_comp_res(i),
                                get_comp_acc(accum_idx));
                    }
                }

                Label skip_acc, store;
                cmp(reg_K_start, 0);
                je(skip_acc, T_NEAR);
                if (req_s8s8_comp) {
                    for (int i = 0; i < n_iters; i++) {
                        const int idx = i + out_ur * n_iters;
                        const auto vmm_acc = get_comp_acc(idx);
                        const auto vmm_res = get_vmm_wei_scale_comp_res(i);
                        const auto addr = !is_ymm_
                                ? EVEX_compress_addr(
                                          reg_comp_ptr, idx * simd_w_)
                                : ptr[reg_comp_ptr + idx * simd_w_];
                        uni_vpaddd(vmm_res, vmm_acc, addr);
                    }
                }

                if (req_zp_comp) {
                    for_(int i = 0; i < n_iters; i++)
                    for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                        const int idx
                                = i * inner_unroll + in_ur + out_ur * n_iters;
                        const auto vmm_acc = get_comp_acc(idx);
                        const auto vmm_res = get_vmm_zp_comp_res(idx);
                        const auto addr = !is_ymm_
                                ? EVEX_compress_addr(
                                          reg_zp_comp_ptr, idx * simd_w_)
                                : ptr[reg_zp_comp_ptr + idx * simd_w_];
                        uni_vpaddd(vmm_res, vmm_acc, addr);
                    }
                }

                L(skip_acc);
                cmp(reg_K_start, rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk);
                jl(store, T_NEAR);

                if (req_s8s8_comp) {
                    mov(reg_tmp, 0xffffffff);
                    const auto vmm_all_bits_1 = vmm_comp_mul;
                    uni_vpbroadcastd(vmm_all_bits_1, reg_tmp.cvt32());
                    mov(reg_tmp, 0x1);
                    const auto vmm_one_s32 = vmm_zero;
                    uni_vpbroadcastd(vmm_one_s32, reg_tmp.cvt32());

                    for (int i = 0; i < n_iters; i++) {
                        const auto vmm_res = get_vmm_wei_scale_comp_res(i);
                        // multiply by 128
                        uni_vpslld(vmm_res, vmm_res, 7);
                        // change sign
                        uni_vpandnd(vmm_res, vmm_res, vmm_all_bits_1);
                        uni_vpaddd(vmm_res, vmm_res, vmm_one_s32);
                    }
                }

                if (req_zp_comp) {
                    mov(reg_zp_a_neg_val_ptr,
                            ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
                    const auto vmm_zp_a_neg_val = vmm_zero;
                    uni_vbroadcastss(
                            vmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);

                    for_(int i = 0; i < n_iters; i++)
                    for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                        const int idx
                                = i * inner_unroll + in_ur + out_ur * n_iters;
                        const auto vmm_res = get_vmm_zp_comp_res(idx);
                        uni_vpmulld(vmm_res, vmm_res, vmm_zp_a_neg_val);
                    }
                }

                L(store);
                if (req_s8s8_comp) {
                    for (int i = 0; i < n_iters; i++) {
                        const auto vmm_res = get_vmm_wei_scale_comp_res(i);
                        const int idx_offset = i + out_ur * n_iters;
                        const auto addr = !is_ymm_
                                ? EVEX_compress_addr(
                                          reg_comp_ptr, idx_offset * simd_w_)
                                : ptr[reg_comp_ptr + idx_offset * simd_w_];
                        uni_vmovups(addr, vmm_res);
                    }
                }
                if (req_zp_comp) {
                    for_(int i = 0; i < n_iters; i++)
                    for (int in_ur = 0; in_ur < inner_unroll; ++in_ur) {
                        const int idx
                                = i * inner_unroll + in_ur + out_ur * n_iters;
                        const auto vmm_res = get_vmm_zp_comp_res(idx);
                        const auto addr = !is_ymm_
                                ? EVEX_compress_addr(
                                          reg_zp_comp_ptr, idx * simd_w_)
                                : ptr[reg_zp_comp_ptr + idx * simd_w_];
                        uni_vmovups(addr, vmm_res);
                    }
                }
            }
        }
    };

    mov(reg_tmp, 0);
    mov(ptr[rsp + reg_n_offs_], reg_tmp);

    if (!is_dynamic_N_) {
        // It is a hard invariant that at most one of {N_blk, N_tail} is a
        // non-multiple of wei_n_blk; otherwise the driver below cannot
        // express the trailing partial block with a single emission.
        const int n_tail_rem = conf_->N_tail % wei_n_blk;
        const int n_blk_rem = conf_->N_blk % wei_n_blk;
        assert(!(n_tail_rem != 0 && n_blk_rem != 0)
                && "N_blk and N_tail can't both be non-multiples of "
                   "wei_n_blk");
        const int n_rem_block = n_tail_rem != 0 ? n_tail_rem : n_blk_rem;

        // Main outer-N loop: emit one wei_n_blk-wide block per iteration
        // while (n + wei_n_blk) <= current_N_blk.
        Label skip_N_loop;
        mov(reg_tmp, wei_n_blk);
        cmp(reg_tmp, ptr[param1 + GET_OFF(current_N_blk)]);
        jg(skip_N_loop, T_NEAR);

        Label n_loop;
        L(n_loop);
        compute_N_loop_body(wei_n_blk);

        mov(reg_tmp, ptr[rsp + reg_n_offs_]);
        add(reg_tmp, wei_n_blk);
        mov(ptr[rsp + reg_n_offs_], reg_tmp);

        add(reg_tmp, wei_n_blk);
        cmp(reg_tmp, ptr[param1 + GET_OFF(current_N_blk)]);
        jle(n_loop, T_NEAR);

        L(skip_N_loop);

        // Trailing partial block (size = n_tail_block < wei_n_blk).
        if (n_rem_block > 0) {
            mov(reg_tmp, ptr[rsp + reg_n_offs_]);
            cmp(reg_tmp, ptr[param1 + GET_OFF(current_N_blk)]);
            Label tail_not_needed;
            jge(tail_not_needed, T_NEAR);
            compute_N_loop_body(n_rem_block);
            L(tail_not_needed);
        }
    } else {
        compute_N_loop_body(wei_n_blk);
    }

    add(rsp, stack_space_needed_);
    postamble();
}

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_bf16_t
    : public jit_brgemm_matmul_copy_b_common_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_bf16_t)

    jit_brgemm_matmul_copy_b_bf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_common_t(conf)
        , typesize(static_cast<int>(conf->b_dt_sz))
        , tr_typesize(static_cast<int>(conf->tr_b_dt_sz))
        , wei_scales_typesize(static_cast<int>(conf->wei_scales_dt_sz))
        , src_stride(conf->copy_B_wei_stride)
        , tr_src_stride(conf_->LDB * k_blk_step * tr_typesize)
        , is_src_4bit(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4,
                  data_type::f4_e2m1))
        , is_dynamic_stride(is_runtime_value(src_stride))
        , is_dynamic_N(conf->is_runtime_N)
        , do_N_loop(conf->LDB < conf->N_blk)
        , req_cvtf82f16(one_of(conf->orig_wei_dt, data_type::f8_e4m3,
                                data_type::f8_e5m2)
                  && conf->wei_dt == data_type::f16 && conf->isa == avx10_2)
        , req_cvtps2bf16(conf->is_bf32 || conf->is_bf16_with_int_wei)
        , req_zp_b_shift(conf->has_zero_point_b && conf->with_wei_decompression)
        , req_apply_wei_scales(conf->apply_scales_in_buffer_b)
        , is_wei_grouped_over_k(
                  conf_->is_wei_zp_per_k || conf_->is_wei_scale_per_k)
        , elems_per_byte(is_src_4bit ? 2 : 1) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;

    enum { k_blk_step = 2, n_blk_step = 16 };
    const int typesize, tr_typesize, wei_scales_typesize;
    const dim_t src_stride, tr_src_stride;
    const bool is_src_4bit;
    const bool is_dynamic_stride;
    const bool is_dynamic_N;
    const bool do_N_loop;
    const bool req_cvtf82f16;
    const bool req_cvtps2bf16;
    const bool req_zp_b_shift;
    const bool req_apply_wei_scales;
    const bool is_wei_grouped_over_k;
    const int elems_per_byte;

    constexpr static int reg_src_offs = 0;
    constexpr static int reg_tr_src_offs = 8;

    constexpr static int reg_k_iters_offs_ = 16;
    constexpr static int reg_current_K_pad_offs_ = 24;

    constexpr static int reg_K_start_offs_ = 32;

    constexpr static int stack_space_needed = 40;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_src_stride = r11;
    reg64_t reg_src_stride_x2 = r12;
    reg64_t reg_src_load_0 = r13;
    reg64_t reg_src_load_1 = r14;
    reg64_t reg_tmp = r15;

    reg64_t reg_copy_block_n_shift = rsi;
    reg64_t reg_wei_scales = rdx;
    reg64_t reg_zp_ptr = r13;

    reg64_t reg_dynamic_tail = rcx;
    Xbyak::Reg8 reg8_mask_shift = reg_dynamic_tail.cvt8();

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_permw = Vmm(1);
    Vmm vmm_tmp = Vmm(1); // used only for avx2_vnni_2
    Vmm vmm_zp_b_shift = Vmm(2);
    Vmm vmm_permd = Vmm(3);
    Vmm vmm_wei_scales = Vmm(4);
    Vmm vmm_f4_lookup_table = Vmm(5);

    void kmovx(Opmask k, unsigned w) {
        if (!isa_has_masks(conf_->isa)) return;
        const auto regw_tmp = reg_tmp.cvt32();
        if (is_dynamic_N) {
            mov(reg_tmp, 1);
            shl(reg_tmp, reg8_mask_shift /* reg_dynamic_tail.cvt8() == cl */);
            sub(reg_tmp, 1);
        } else
            mov(regw_tmp, w);
        if (req_cvtps2bf16)
            jit_generator_t::kmovw(k, regw_tmp);
        else
            jit_generator_t::kmovd(k, regw_tmp);
    }

    void copy_block(int nrows, int ncolumns, bool n_tail, bool zeropad);
    void copy_2x32(int nrows, int ncolumns, bool zeropad);
    void init_masks();
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::copy_2x32(
        int nrows, int ncolumns, bool zeropad) {

    const int columns_tail = ncolumns % n_blk_step;
    if (columns_tail > 0 && columns_tail < n_blk_step) {
        const auto tail_mask = (1 << columns_tail) - 1;
        kmovx(kTail, tail_mask);
        if (is_src_4bit) {
            const auto tail_mask_4bit = (1 << (columns_tail / 2)) - 1;
            kmovx(kTail_4bit, tail_mask_4bit);
        }
    }

    static constexpr int blk_sz = k_blk_step;
    const bool is_src_f4_ = conf_->orig_wei_dt == data_type::f4_e2m1;
    const bool is_src_int4_
            = one_of(conf_->orig_wei_dt, data_type::s4, data_type::u4);
    const int reserved_regs = is_src_f4_ ? (req_apply_wei_scales ? 6 : 5)
            : req_apply_wei_scales       ? 5
            : is_src_int4_               ? 4
            : req_zp_b_shift             ? 3
                                         : 2;
    const int max_isa_regs = isa_num_vregs(conf_->isa);
    const int max_regs_available = max_isa_regs - reserved_regs;
    const int max_unroll = max_regs_available / blk_sz;

    auto get_vmm = [max_unroll, max_isa_regs, reserved_regs](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = reserved_regs + max_unroll * ((idx + 1) % blk_sz) + blk;
        UNUSED(max_isa_regs);
        assert(reg_idx >= reserved_regs && reg_idx < max_isa_regs);
        return Vmm(reg_idx);
    };

    /** Loads zero points, when is_wei_zp_per_n is set.
    *   Zeropoints size over N dimension always equals to N.
    */
    auto load_zero_point = [this, ncolumns, columns_tail](int n) {
        if (!conf_->is_wei_zp_per_n) return;
        const bool is_tail = (ncolumns - n) < n_blk_step;
        const auto zp_dt = conf_->wei_zp_dt;
        const auto zp_dt_sz = types::data_type_size(zp_dt);
        const auto elems_per_byte
                = one_of(zp_dt, data_type::s4, data_type::u4) ? 2 : 1;
        const auto offset = n * zp_dt_sz / elems_per_byte;
        const auto addr = maybe_EVEX_compress_addr(reg_zp_ptr, offset);
        if (is_tail && !isa_has_masks(conf_->isa)) {
            load_bytes(vmm_zp_b_shift, addr, columns_tail / elems_per_byte);
            load_value(vmm_zp_b_shift, vmm_zp_b_shift, vmm_permd, zp_dt);
        }
        load_value(vmm_zp_b_shift, addr, vmm_permd, zp_dt, is_tail);
    };

    /** Loads scales, when is_wei_scale_per_n is set.
    *   Scales size over N dimension always equals to N.
    */
    auto load_scales = [this, ncolumns, columns_tail](int n) {
        if (!conf_->is_wei_scale_per_n || !conf_->apply_scales_in_buffer_b)
            return;

        const bool is_tail = (ncolumns - n) < n_blk_step;
        const auto &scales_dt = conf_->wei_scales_dt;
        const auto scales_dt_sz = types::data_type_size(scales_dt);
        const auto offset = n * scales_dt_sz;
        const auto addr = maybe_EVEX_compress_addr(reg_wei_scales, offset);
        if (is_tail && !isa_has_masks(conf_->isa)) {
            load_bytes(
                    vmm_wei_scales, addr, columns_tail * wei_scales_typesize);
            load_scale_value(vmm_wei_scales, vmm_wei_scales, scales_dt,
                    /*is_tail=*/false);
        }
        load_scale_value(vmm_wei_scales, addr, scales_dt, is_tail);
    };

    auto load = [this, get_vmm, ncolumns, columns_tail, load_scales,
                        load_zero_point](int blk, int k, int n) {
        auto src_reg = get_vmm(blk, k % k_blk_step);
        const bool is_tail = ncolumns - n < n_blk_step;
        auto src_load = maybe_mask(src_reg, is_tail);
        const auto offset
                = ((is_dynamic_stride ? 0 : k * src_stride) + (n * typesize))
                / elems_per_byte;
        const auto reg_src_load
                = is_dynamic_stride && k % 2 != 0 ? reg_src_load_1 : reg_src;
        auto load_addr = maybe_EVEX_compress_addr(reg_src_load, offset);
        if (!isa_has_masks(conf_->isa)) {
            if (is_tail)
                load_bytes(src_load, load_addr,
                        columns_tail * tr_typesize / elems_per_byte);
            else
                uni_vmovups(src_load, load_addr);
            load_value(src_reg, src_reg, vmm_permd, conf_->orig_wei_dt, false,
                    vmm_f4_lookup_table);
        } else if (req_cvtf82f16) {
            f8_to_f16_upconvert(
                    conf_->orig_wei_dt, src_reg, load_addr, is_tail);
        } else {
            load_value(src_reg, load_addr, vmm_permd, conf_->orig_wei_dt,
                    is_tail, vmm_f4_lookup_table);
        }
        load_zero_point(n);
        load_scales(n);
        if (!req_cvtf82f16)
            decompress_and_downcvt_reg(src_reg, vmm_zp_b_shift, vmm_wei_scales,
                    conf_->orig_wei_dt, conf_->wei_dt);
    };

    /** Stores half of the block using mask for the case when vnni_granularity == 2 */
    auto store_half_block = [&](const Vmm &src_vmm0, const Vmm &src_vmm1,
                                    const Xbyak::Address &store_addr) {
        const auto zmm1 = zmm(src_vmm1.getIdx());
        const auto zmm0 = zmm(src_vmm0.getIdx());
        uni_vxorps(zmm1, zmm1, zmm1);
        //if k % 2 == 1 then save only odd indices
        // otherwise: using only even indices
        Label even_k, end_permute;
        mov(reg_tmp, ptr[rsp + reg_K_start_offs_]);
        test(reg_tmp, 1);
        jz(even_k, T_NEAR);
        vinsertf64x4(zmm0, zmm1, ymm(src_vmm0.getIdx()), 1);
        vpermw(zmm0, vmm_permw, zmm0);
        uni_vmovdqu16(store_addr | kAAAA, zmm0);
        jmp(end_permute);
        L(even_k);
        vinsertf64x4(zmm0, zmm1, ymm(src_vmm0.getIdx()), 0);
        vpermw(zmm0, vmm_permw, zmm0);
        uni_vmovdqu16(store_addr, zmm0);
        L(end_permute);
    };

    // The case when it's required to store half block
    // When grouped over K weights and K == 1
    const auto kernel_early_stop = is_wei_grouped_over_k && nrows == 1;

    int iter = 0;
    int n_iters;
    if (is_dynamic_N || do_N_loop) {
        n_iters = ncolumns;
    } else {
        n_iters = conf_->wei_n_blk;
    }

    for_(int k = 0; k < nrows; k += k_blk_step)
    for (int n = 0; n < n_iters; n += n_blk_step) {
        const int k_blk = k / k_blk_step;
        const dim_t tr_src_off
                = k_blk * tr_src_stride + n * k_blk_step * tr_typesize;

        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);
        const auto store_addr_ymm1
                = ptr[reg_tr_src + tr_src_off + vreg_traits_t<Vmm>::vlen];
        const int blk_idx = iter % max_unroll;
        const auto src_vmm0 = get_vmm(blk_idx, 0);
        const auto src_zmm0 = zmm(src_vmm0.getIdx());
        const auto src_vmm1 = get_vmm(blk_idx, 1);
        if (is_dynamic_stride && n == 0) {
            if (k == 0) {
                mov(reg_src_load_1, reg_src);
                add(reg_src_load_1, reg_src_stride);
            } else {
                add(reg_src, reg_src_stride_x2);
                add(reg_src_load_1, reg_src_stride_x2);
            }
        }

        if (ncolumns - n <= 0 || zeropad) {
            uni_vmovups(store_addr, vmm_zero);
            if (!is_superset(conf_->isa, avx512_core))
                uni_vmovups(store_addr_ymm1, vmm_zero);
            continue;
        }

        load(blk_idx, k, n);

        // Store only half block
        if (kernel_early_stop) {
            store_half_block(src_vmm0, src_vmm1, store_addr);
            iter++;
            continue;
        }

        // Load second K half blk and downconvert if required.
        if (nrows - k >= k_blk_step)
            load(blk_idx, k + 1, n);
        else
            uni_vxorps(src_vmm1, src_vmm1, src_vmm1);

        if (is_superset(conf_->isa, avx512_core)) {
            const auto src_ymm1 = ymm(src_vmm1.getIdx());
            vinsertf64x4(src_zmm0, src_zmm0, src_ymm1, 1);
            vpermw(src_zmm0, vmm_permw, src_zmm0);
            uni_vmovups(store_addr, src_zmm0);
        } else {
            assert(is_superset(conf_->isa, avx2));
            vpunpcklwd(vmm_tmp, src_vmm0, src_vmm1);
            vpunpckhwd(src_vmm1, src_vmm0, src_vmm1);
            vperm2i128(src_vmm0, vmm_tmp, src_vmm1, 0x20);
            vperm2i128(src_vmm1, vmm_tmp, src_vmm1, 0x31);
            uni_vmovups(store_addr, src_vmm0);
            uni_vmovups(store_addr_ymm1, src_vmm1);
        }

        iter++;
    }
    if (is_dynamic_stride && nrows > 0) {
        add(reg_src, nrows % 2 == 0 ? reg_src_stride_x2 : reg_src_stride);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::init_masks() {
    alignas(64) static constexpr const int16_t bf16_vnni_permute[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    if (is_superset(conf_->isa, avx512_core)) {
        kxnorw(kFFFF, kFFFF, kFFFF); // 1111 1111 1111 1111

        mov(reg_tmp, reinterpret_cast<size_t>(bf16_vnni_permute));
        vmovdqa64(vmm_permw, ptr[reg_tmp]);

        if (isa_has_masks(conf_->isa)) {
            // 64-bit mask is also used when is_wei_[zp\scales]_per_k
            mov(reg_tmp, 0xAAAAAAAAAAAAAAAA);
            kmovq(kAAAA, reg_tmp);
            mov(reg_tmp, 0x5555555555555555);
            kmovq(k5555, reg_tmp);
        }

        if (is_src_4bit) {
            alignas(64) static constexpr const uint32_t int4_permute[16]
                    = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
            mov(reg_tmp, reinterpret_cast<size_t>(int4_permute));
            vmovdqa32(vmm_permd, ptr[reg_tmp]);
        }

        if (conf_->orig_wei_dt == data_type::f4_e2m1) {
            alignas(64) static constexpr const float f4_e2m1_table[16]
                    = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f,
                            -.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            mov(reg_tmp, reinterpret_cast<size_t>(f4_e2m1_table));
            vmovdqa32(vmm_f4_lookup_table, ptr[reg_tmp]);
        }
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::copy_block(
        int nrows, int ncolumns, bool n_tail, bool zeropad) {
    if (!do_N_loop && (!is_dynamic_N || !n_tail)) {
        copy_2x32(nrows, ncolumns, zeropad);
        return;
    }

    mov(reg_dynamic_tail, reg_N_blk);
    // dynamic tail processing: main loop with ncolumns = n_blk_step and
    // finally process tail < n_blk_step with dynamically computed mask
    // NOTE: for dynamic_stride case copy_2x32() shifts reg_src pointer
    // so we need to backup/restore its value for every iteration wrt n
    // except the last one

    mov(ptr[rsp + reg_tr_src_offs], reg_tr_src);
    xor_(reg_copy_block_n_shift, reg_copy_block_n_shift);

    int current_n_blk_step
            = do_N_loop ? static_cast<int>(conf_->LDB) : n_blk_step;

    Label loop_row_start, loop_row_tail, loop_row_done;
    cmp(reg_dynamic_tail, current_n_blk_step);
    jl(loop_row_tail, T_NEAR);
    L(loop_row_start);
    {
        mov(ptr[rsp + reg_src_offs], reg_src);
        add(reg_src, reg_copy_block_n_shift);
        copy_2x32(nrows, current_n_blk_step, zeropad);

        if (do_N_loop) {
            add(reg_tr_src,
                    (current_n_blk_step / conf_->LDB) * conf_->LDB2
                            * tr_typesize);
            add(reg_src,
                    conf_->B_strides[0] == typesize
                            ? current_n_blk_step * typesize
                            : conf_->B_strides[0]);
            add(reg_copy_block_n_shift,
                    conf_->B_strides[0] == typesize
                            ? current_n_blk_step * typesize
                            : conf_->B_strides[0]);

        } else {
            add(reg_src, current_n_blk_step * typesize);
            add(reg_tr_src, current_n_blk_step * k_blk_step * tr_typesize);
            add(reg_copy_block_n_shift, current_n_blk_step * typesize);
        }

        sub(reg_dynamic_tail, current_n_blk_step);

        cmp(reg_dynamic_tail, 0);
        jle(loop_row_done, T_NEAR);

        mov(reg_src, ptr[rsp + reg_src_offs]);

        cmp(reg_dynamic_tail, current_n_blk_step);
        jl(loop_row_tail, T_NEAR);

        jmp(loop_row_start, T_NEAR);
    }

    L(loop_row_tail);
    {
        cmp(reg_dynamic_tail, 0);
        jle(loop_row_done, T_NEAR);

        add(reg_src, reg_copy_block_n_shift);
        if (do_N_loop) {
            copy_2x32(nrows, ncolumns % current_n_blk_step, zeropad);
        } else {
            copy_2x32(nrows, 1 /* to force tail case */, zeropad);
        }
    }
    L(loop_row_done);

    // restore pointers
    sub(reg_src, reg_copy_block_n_shift);
    mov(reg_tr_src, ptr[rsp + reg_tr_src_offs]);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_bf16_t<Vmm>::generate() {
    assert(tr_typesize == sizeof(bfloat16_t));
    preamble();
    sub(rsp, stack_space_needed);
    uni_vxorps(vmm_zero, vmm_zero, vmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(ptr[rsp + reg_tr_src_offs], reg_tr_src);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_wei_scales, ptr[param1 + GET_OFF(wei_scales_ptr)]);
    // Due to lack of registers save k_iters and k_pad into stack space
    mov(reg_tmp, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(ptr[rsp + reg_k_iters_offs_], reg_tmp);
    mov(reg_tmp, ptr[param1 + GET_OFF(current_K_pad)]);
    mov(ptr[rsp + reg_current_K_pad_offs_], reg_tmp);
    mov(reg_tmp, ptr[param1 + GET_OFF(current_K_start)]);
    mov(ptr[rsp + reg_K_start_offs_], reg_tmp);
    mov(reg_tmp, 0);

    if (is_dynamic_stride) {
        mov(reg_src_stride, ptr[param1 + GET_OFF(dynamic_src_stride)]);
        mov(reg_src_stride_x2, ptr[param1 + GET_OFF(dynamic_src_stride)]);
        shl(reg_src_stride_x2, 1);
    }

    mov(reg_zp_ptr, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
    load_common_zp_value(vmm_zp_b_shift, reg_zp_ptr);
    load_common_scale_value(vmm_wei_scales, reg_wei_scales);

    init_masks();

    auto compute_K_loop_body = [&](const reg64_t &reg_K, int ncolumns,
                                       bool is_N_tail, bool zeropad) {
        // Compute special K-loop for per-k attributes
        // Only when k_group_size < k_blk_step
        // Otherwise default K-loop is used
        if (is_wei_grouped_over_k) {
            const int k_group_size = conf_->is_wei_zp_per_k
                    ? static_cast<int>(conf_->wei_zp_k_gsize)
                    : static_cast<int>(conf_->wei_scales_k_gsize);
            if (k_group_size < k_blk_step) {
                if (zeropad) return;
                copy_block(
                        k_group_size, ncolumns, is_N_tail, /*zeropad= */ false);
                return;
            }
        }

        const int k_unroll = 8;
        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;

        cmp(reg_K, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step, ncolumns, is_N_tail, zeropad);

        if (!zeropad && !is_dynamic_stride)
            add(reg_src, (k_unroll * k_blk_step * src_stride) / elems_per_byte);
        add(reg_tr_src, k_unroll * tr_src_stride);

        sub(reg_K, k_unroll * k_blk_step);
        cmp(reg_K, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step, ncolumns, is_N_tail, zeropad);
        if (!zeropad && !is_dynamic_stride)
            add(reg_src, (k_blk_step * src_stride) / elems_per_byte);
        add(reg_tr_src, tr_src_stride);

        sub(reg_K, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K, 0);
            jle(K_loop_done, T_NEAR);

            copy_block(k_blk_tail, ncolumns, is_N_tail, zeropad);
            add(reg_tr_src, tr_src_stride);
            sub(reg_K, k_blk_tail);
            L(K_loop_done);
        }
    };

    auto compute_K_loop = [&](bool is_N_tail) {
        int ncolumns = is_N_tail ? static_cast<int>(conf_->N_tail)
                                 : static_cast<int>(conf_->N_blk);
        // 'param1' register (rcx on Windows) re-written in compute_K_loop_body
        // so we need to read and keep 'current_K_pad' parameter in stack before
        // the call
        mov(reg_K_iters, ptr[rsp + reg_k_iters_offs_]);
        compute_K_loop_body(reg_K_iters, ncolumns, is_N_tail, false);
        mov(reg_K_iters, ptr[rsp + reg_current_K_pad_offs_]);
        compute_K_loop_body(reg_K_iters, ncolumns, is_N_tail, true);
    };

    Label done;
    cmp(reg_N_blk, 0);
    jle(done, T_NEAR);

    if (conf_->N_tail > 0 || is_dynamic_N) {
        Label main_N_blk;
        cmp(reg_N_blk, conf_->N_blk);
        je(main_N_blk, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(main_N_blk);
    }

    compute_K_loop(false);
    L(done);

    add(rsp, stack_space_needed);
    postamble();
}

template struct jit_brgemm_matmul_copy_b_bf16_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_bf16_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_f32_t
    : public jit_brgemm_matmul_copy_b_common_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_f32_t)

    jit_brgemm_matmul_copy_b_f32_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_common_t(conf)
        , dt_in_(conf->orig_wei_dt)
        , simd_w_(vreg_traits_t<Vmm>::vlen / sizeof(float))
        , is_src_f4_(conf->orig_wei_dt == data_type::f4_e2m1)
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , is_src_4bit_(is_src_int4_ || is_src_f4_)
        , req_zp_b_shift_(
                  conf->has_zero_point_b && conf->with_wei_decompression)
        , req_apply_wei_scales_(conf->apply_scales_in_buffer_b)
        , cached_wei_scales_regs_(is_src_f4_ && !is_ymm_
                                  && req_apply_wei_scales_
                                  && conf->is_wei_scale_per_n
                          ? div_up(conf->wei_n_blk, simd_w_)
                          : 0)
        , typesize_in_(types::data_type_size(dt_in_))
        , src_elems_per_byte_(is_src_4bit_ ? 2 : 1)
        , src_stride_(conf_->copy_B_wei_stride)
        , tr_src_stride_(conf_->LDB * typesize_out_) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    const data_type_t dt_in_;
    const int simd_w_;
    const bool is_src_f4_, is_src_int4_, is_src_4bit_;
    const bool req_zp_b_shift_, req_apply_wei_scales_;
    const int cached_wei_scales_regs_;
    const size_t typesize_in_;
    const int src_elems_per_byte_;
    const size_t typesize_out_ = sizeof(float);
    dim_t src_stride_, tr_src_stride_;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t reg_tmp = r15;
    reg32_t regw_tmp = r15d;
    reg64_t reg_wei_scales = rdx;
    reg64_t reg_zp_ptr = r11;

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_wei_scales = Vmm(1);
    Vmm vmm_permd = Vmm(2);
    Vmm vmm_zp_b_shift = Vmm(3);
    Vmm vmm_f4_lookup_table = Vmm(4);
    Ymm ymm_tail_mask = ymm1;

    Vmm wei_scales(int n) const {
        return cached_wei_scales_regs_ > 0
                ? Vmm(isa_num_vregs(conf_->isa) - 1 - n / simd_w_)
                : vmm_wei_scales;
    }

    inline void kmovw(Opmask k, unsigned w) {
        if (!isa_has_masks(conf_->isa)) return;
        mov(regw_tmp, w);
        jit_generator_t::kmovd(k, regw_tmp);
    }

    void copy_16_x_n_block(int nrows, int ncolumns);
    void load_scales(int n, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::copy_16_x_n_block(
        int nrows, int ncolumns) {
    const int max_isa_regs = isa_num_vregs(conf_->isa);
    const int reserved_regs = is_src_f4_ ? 5
            : req_zp_b_shift_            ? 4
            : is_src_int4_               ? 3
                                         : 2;
    const int max_regs_available
            = max_isa_regs - reserved_regs - cached_wei_scales_regs_;
    assert(max_regs_available > 0);

    auto get_vmm = [max_regs_available, reserved_regs](int reg_idx) {
        MAYBE_UNUSED(max_regs_available);
        MAYBE_UNUSED(reserved_regs); // some compilers detect it as unused
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return Vmm(reg_idx + reserved_regs);
    };

    const int columns_tail = ncolumns % simd_w_;
    if (columns_tail < simd_w_) {
        if (isa_has_masks(conf_->isa)) {
            const auto tail_mask = (1 << columns_tail) - 1;
            kmovw(kTail, tail_mask);
            if (is_src_4bit_) {
                const auto bytes_needed
                        = (columns_tail + src_elems_per_byte_ - 1)
                        / src_elems_per_byte_;
                const auto tail_mask_4bit = (1 << bytes_needed) - 1;
                kmovw(kTail_4bit, tail_mask_4bit);
            }
        }
    }

    auto load = [this, get_vmm, ncolumns](int blk, int k, int n) {
        auto src_vmm = get_vmm(blk);
        const bool is_tail = ncolumns - n < simd_w_;
        auto addr = maybe_EVEX_compress_addr(reg_src,
                (k * src_stride_ + n * typesize_in_) / src_elems_per_byte_);
        if (is_tail && !isa_has_masks(conf_->isa)) {
            load_bytes(src_vmm, addr,
                    static_cast<int>((ncolumns % simd_w_) * typesize_in_
                            / src_elems_per_byte_));
            load_value(src_vmm, src_vmm, vmm_permd, conf_->orig_wei_dt, false);
        } else
            load_value(src_vmm, addr, vmm_permd, conf_->orig_wei_dt, is_tail);

        decompress_reg(maybe_mask(src_vmm, is_tail), vmm_zp_b_shift,
                wei_scales(n), conf_->orig_wei_dt);
        if (is_src_f4_ && is_tail)
            uni_vmovups(maybe_mask(src_vmm, true), src_vmm);
    };

    /** Loads zero points, when is_wei_zp_per_n is set.
    *   Zeropoints size over N dimension always equals to N.
    */
    auto load_zero_point = [this, ncolumns](int n) {
        if (!conf_->is_wei_zp_per_n) return;

        const bool is_tail = (ncolumns - n) < simd_w_;
        const auto zp_dt = conf_->wei_zp_dt;
        const auto zp_dt_sz = types::data_type_size(zp_dt);
        const auto elems_per_byte
                = one_of(zp_dt, data_type::s4, data_type::u4) ? 2 : 1;
        const auto offset = n * zp_dt_sz / elems_per_byte;
        const auto addr = maybe_EVEX_compress_addr(reg_zp_ptr, offset);
        if (is_tail && !isa_has_masks(conf_->isa)) {
            load_bytes(vmm_zp_b_shift, addr,
                    static_cast<int>(
                            (ncolumns % simd_w_) * zp_dt_sz / elems_per_byte));
            load_value(vmm_zp_b_shift, vmm_zp_b_shift, vmm_permd, zp_dt, false);
        } else
            load_value(vmm_zp_b_shift, addr, vmm_permd, zp_dt, is_tail);
    };

    int iter = 0;
    for_(int k = 0; k < nrows; k++)
    for (int n = 0; n < conf_->wei_n_blk; n += simd_w_) {
        const dim_t tr_src_off = k * tr_src_stride_ + n * typesize_out_;
        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);

        const int zero_padding = ncolumns - n;
        if (zero_padding <= 0) {
            uni_vmovups(store_addr, vmm_zero);
            continue;
        }

        load_zero_point(n);
        if (cached_wei_scales_regs_ == 0) load_scales(n, ncolumns);
        const int blk_idx = iter % max_regs_available;
        load(blk_idx, k, n);

        const auto src_vmm0 = get_vmm(blk_idx);
        uni_vmovups(store_addr, src_vmm0);
        iter++;
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::load_scales(int n, int ncolumns) {
    if (!conf_->is_wei_scale_per_n || !req_apply_wei_scales_) return;

    const bool is_tail = (ncolumns - n) < simd_w_;
    const auto scales_dt = conf_->wei_scales_dt;
    const auto scales_dt_sz = types::data_type_size(scales_dt);
    const auto addr
            = maybe_EVEX_compress_addr(reg_wei_scales, n * scales_dt_sz);
    const auto vmm_scales = wei_scales(n);
    if (is_tail && !isa_has_masks(conf_->isa)) {
        load_bytes(vmm_scales, addr,
                static_cast<int>((ncolumns % simd_w_) * scales_dt_sz));
        load_scale_value(vmm_scales, vmm_scales, scales_dt, false);
    } else
        load_scale_value(vmm_scales, addr, scales_dt, is_tail);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::compute_k_loop(int ncolumns) {
    if (cached_wei_scales_regs_ > 0) {
        // Each copy call stays within one K-group, so its scales are invariant.
        kmovw(kTail, (1 << (ncolumns % simd_w_)) - 1);
        for (int n = 0; n < ncolumns; n += simd_w_)
            load_scales(n, ncolumns);
    }

    auto compute_uni_k_loop = [&](int unroll) {
        Label K_start_label, K_end_label;

        L(K_start_label);
        cmp(reg_K_iters, unroll);
        jl(K_end_label, T_NEAR);

        copy_16_x_n_block(unroll, ncolumns);
        add(reg_src, (unroll * src_stride_) / src_elems_per_byte_);
        add(reg_tr_src, unroll * tr_src_stride_);

        sub(reg_K_iters, unroll);
        jmp(K_start_label, T_NEAR);

        L(K_end_label);
    };

    constexpr int k_unroll = 16;
    compute_uni_k_loop(k_unroll);
    compute_uni_k_loop(1);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_f32_t<Vmm>::generate() {
    preamble();
    uni_vxorps(vmm_zero, vmm_zero, vmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_wei_scales, ptr[param1 + GET_OFF(wei_scales_ptr)]);
    mov(reg_zp_ptr, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
    kmovw(kFFFF, 0xffff); // 1111111111111111

    if (is_src_4bit_) {
        kmovw(kAAAA, 0xaaaa);
        kmovw(k5555, 0x5555);
        if (is_superset(conf_->isa, avx512_core)) {
            alignas(64) static constexpr const uint32_t int4_permute[16]
                    = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
            mov(reg_tmp, reinterpret_cast<size_t>(int4_permute));
            vmovdqa32(vmm_permd, ptr[reg_tmp]);
        } else if (is_superset(conf_->isa, avx2)) {
            alignas(64) static constexpr const uint32_t int4_permute_avx2[8]
                    = {0, 4, 1, 5, 2, 6, 3, 7};
            mov(reg_tmp, reinterpret_cast<size_t>(int4_permute_avx2));
            vmovdqa(vmm_permd, ptr[reg_tmp]);
        }
    }

    if (is_src_f4_ && is_superset(conf_->isa, avx512_core)) {
        alignas(64) static constexpr const float f4_e2m1_table[16]
                = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -.5f,
                        -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        mov(reg_tmp, reinterpret_cast<size_t>(f4_e2m1_table));
        vmovdqa32(vmm_f4_lookup_table, ptr[reg_tmp]);
    }

    load_common_zp_value(vmm_zp_b_shift, reg_zp_ptr);
    load_common_scale_value(vmm_wei_scales, reg_wei_scales);

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_k_loop(static_cast<int>(conf_->N_tail));
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_k_loop(static_cast<int>(conf_->N_blk));
    L(done);

    postamble();
}

template struct jit_brgemm_matmul_copy_b_f32_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_f32_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_transposed_t
    : public jit_brgemm_matmul_copy_b_common_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_transposed_t)

    jit_brgemm_matmul_copy_b_transposed_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_common_t(conf)
        , typesize_(static_cast<int>(conf_->b_dt_sz))
        , tr_typesize_(static_cast<int>(conf_->tr_b_dt_sz))
        , wei_scales_typesize_(static_cast<int>(conf_->wei_scales_dt_sz))
        , vnni_granularity_(
                  static_cast<int>(data_type_vnni_granularity(conf_->wei_dt)))
        , k_blk_step_(vlen_ / tr_typesize_)
        , is_wei_grouped_over_k_(
                  conf_->is_wei_zp_per_k || conf_->is_wei_scale_per_k)
        , do_compute_compensation_(
                  conf_->has_zero_point_a || conf_->s8s8_compensation_required)
        , is_bf32_(conf->is_bf32)
        , is_bf16_with_int_wei_(conf->is_bf16_with_int_wei)
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , is_src_f4_(conf->orig_wei_dt == data_type::f4_e2m1)
        , is_src_4bit_(is_src_int4_ || is_src_f4_)
        , req_cvtps2xf16_(conf->is_bf32 || conf->is_bf16_with_int_wei
                  || (conf->is_f16_with_int_wei
                          && conf->wei_dt == data_type::f16))
        , req_zp_comp_(conf_->has_zero_point_a)
        , req_s8s8_comp_(conf_->s8s8_compensation_required)
        , req_zp_b_shift_(
                  conf_->has_zero_point_b && conf_->with_wei_decompression)
        , req_int8_zp_b_shift_(conf_->has_zero_point_b
                  && conf_->with_int8_grouped_quantization
                  && !conf_->with_per_mn_compensation)
        , is_zp_int4_(one_of(conf->wei_zp_dt, data_type::s4, data_type::u4))
        , has_vpermb_(cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
        , req_apply_wei_scales_(conf_->apply_scales_in_buffer_b)
        , avx512_core_dot_product_(
                  do_compute_compensation_ && !isa_has_int8_vnni(conf->isa))
        // See the note in `create_brgemm_matmul_copy_b` why `orig_wei_dt` used.
        , use_fp16_instructions_(is_subset(conf_->isa, avx512_core_fp16)
                  && conf_->orig_wei_dt == data_type::f16
                  && conf_->wei_dt == data_type::f32)
        // This variable is responsible for enabling to upconversion from bf16
        // to f32 similarly to f16, mostly for proper tail handling.
        , use_bf16_instructions_(is_subset(conf_->isa, avx512_core_bf16)
                  && conf_->orig_wei_dt == data_type::bf16
                  && conf_->wei_dt == data_type::f32)
        , req_cvtf82f16_(conf_->isa == avx10_2
                  && conf_->wei_dt == data_type::f16
                  && utils::one_of(conf_->orig_wei_dt, data_type::f8_e4m3,
                          data_type::f8_e5m2))
        , max_tmp_idx(16
                  - (avx512_core_dot_product_
                                  ? 8
                                  : (do_compute_compensation_      ? 6
                                                    : is_src_f4_   ? 4
                                                    : is_src_int4_ ? 2
                                                    : req_zp_b_shift_
                                                            || req_int8_zp_b_shift_
                                                    ? 1
                                                    : 0))
                  - ((req_int8_zp_b_shift_ && do_compute_compensation_)
                                  ? (is_src_int4_ ? 2 : 1)
                                  : 0))
        , src_stride_(conf_->copy_B_wei_stride)
        , tr_src_stride_(conf_->LDB * vnni_granularity_ * tr_typesize_)
        , src_elems_per_byte_(is_src_4bit_ ? 2 : 1)
        , is_dynamic_N_(conf->is_runtime_N) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;

    static constexpr bool is_ymm_ = std::is_same<Vmm, Xbyak::Ymm>::value;
    static constexpr cpu_isa_t isa_ = is_ymm_ ? avx2 : avx512_core;
    static constexpr int max_vmm_regs_ = cpu_isa_traits_t<isa_>::n_vregs;
    static constexpr int vlen_ = vreg_traits_t<Vmm>::vlen;
    static constexpr int n_blk_step_ = is_ymm_ ? 8 : 16;
    static constexpr int req_cvt_bf16_k_blk_step_ = 16;
    static constexpr size_t comp_shift_ = vlen_;

    const int typesize_;
    const int tr_typesize_;
    const int wei_scales_typesize_;
    const int vnni_granularity_;
    const int k_blk_step_;
    const bool is_wei_grouped_over_k_;
    const bool do_compute_compensation_;
    const bool is_bf32_;
    const bool is_bf16_with_int_wei_;
    const bool is_src_int4_;
    const bool is_src_f4_;
    const bool is_src_4bit_;
    const bool req_cvtps2xf16_;
    const bool req_zp_comp_;
    const bool req_s8s8_comp_;
    const bool req_zp_b_shift_;
    const bool req_int8_zp_b_shift_;
    const bool is_zp_int4_;
    const bool has_vpermb_;
    const bool req_apply_wei_scales_;
    const bool avx512_core_dot_product_;
    const bool use_fp16_instructions_;
    const bool use_bf16_instructions_;
    const bool req_cvtf82f16_;
    const int max_tmp_idx;

    const dim_t src_stride_, tr_src_stride_;
    const int src_elems_per_byte_;
    const bool is_dynamic_N_;

    constexpr static int ldb_step_idx_offs = 0;
    constexpr static int reg_zp_b_ptr_offs_ = 8;
    constexpr static int reg_cur_k_offs_ = 16;
    constexpr static int reg_src_save_offs_ = 24;
    constexpr static int reg_init_k_rem_offs_ = 32;
    constexpr static int stack_space_needed = 40;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;
    reg64_t reg_comp_ptr = rdx;
    reg64_t reg_zp_ptr = rdx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_iters = r9;
    reg64_t reg_src = r10;
    reg64_t reg_tr_src = r11;
    reg64_t reg_zp_comp_ptr = r12;
    reg64_t reg_zp_a_neg_val_ptr = r13;
    reg64_t reg_K_start = r14;
    reg64_t reg_wei_scales = rsi;

    reg64_t regq_tmp = r15;
    reg32_t regw_tmp = r15d;
    reg64_t imm_addr64 = abi_not_param1;

    // Note: for the AVX2 implementation, reserve Ymm(8) and Ymm(9) as
    // temporary compute registers.
    Vmm vmm_comp_mul = Vmm(max_vmm_regs_ - 1);
    Vmm vmm_comp_acc = Vmm(max_vmm_regs_ - 2);
    Vmm vmm_zp_a_neg_val = Vmm(max_vmm_regs_ - 3);
    Vmm vmm_s8s8_comp_acc = Vmm(max_vmm_regs_ - 4);
    Vmm vmm_all_bits_1 = Vmm(max_vmm_regs_ - 5);
    Vmm vmm_one_s32 = Vmm(max_vmm_regs_ - 6);

    // Required in every dot product for INT8 non-VNNI computation.
    Vmm vmm_ones_words = Vmm(max_vmm_regs_ - 7);
    Vmm vmm_dot_product_temp = Vmm(max_vmm_regs_ - 8);

    // When int8-grouped-quant ZP runs alongside compensation, place
    // these registers right above the (already-shrunk) tmp_vmm range
    // so they never alias the comp registers at the top of the file.
    Vmm vmm_zp_b_val = (req_int8_zp_b_shift_ && do_compute_compensation_)
            ? Vmm(n_blk_step_ + max_tmp_idx)
            : Vmm(max_vmm_regs_ - 1);
    Vmm vmm_permd
            = (req_int8_zp_b_shift_ && do_compute_compensation_ && is_src_int4_)
            ? Vmm(n_blk_step_ + max_tmp_idx + 1)
            : Vmm(max_vmm_regs_ - 2);
    // Collide with `vmm_zp_a_neg_val` as they shouldn't intersect in
    // functionality.
    Vmm vmm_wei_scales = Vmm(max_vmm_regs_ - 3);
    Vmm vmm_f4_lookup_table = Vmm(max_vmm_regs_ - 4);

    void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    }

    void kmovq(Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator_t::kmovq(k, regq_tmp);
    }

    Vmm src_vmm(int i) {
        assert(i >= 0 && i < n_blk_step_);
        return Vmm(i);
    }

    Vmm tmp_vmm(int i) {
        // If compensation compute is required - last 6 zmms are reserved for it
        assert(i >= 0 && IMPLICATION(!is_ymm_, i < max_tmp_idx)
                && IMPLICATION(is_ymm_, i < 2));
        return Vmm(n_blk_step_ + i);
    }

    void init_tail_mask(const int columns_tail, const bool use_4bit_mask);
    bool preload_4bit(const Xmm &xmm_in, const int i, const int columns_tail,
            const bool is_tail, const dim_t offset);
    void copy_row_x_col(int nrows, int ncolumns);
    void compute_K_loop(bool is_N_tail, int curr_K_tail, bool is_first_K_iter,
            bool is_last_K_iter);
    void compute_N_loop(
            int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter);

    inline void dot_product(Vmm v1, Vmm v2, Vmm v3) {
        if (!avx512_core_dot_product_)
            vpdpbusd(v1, v2, v3, get_encoding());
        else {
            vpmaddubsw(vmm_dot_product_temp, v2, v3);
            vpmaddwd(
                    vmm_dot_product_temp, vmm_dot_product_temp, vmm_ones_words);
            vpaddd(v1, v1, vmm_dot_product_temp);
        }
    }
    inline bool valid_to_load_next(int next_row_idx, int num_rows) {
        const bool dynamic_tail = is_dynamic_N_ && num_rows < n_blk_step_;
        return next_row_idx < num_rows || dynamic_tail;
    }

    /**
     * Loads zero point value and broadcasts it over Vmm register.
     * Supported data types: s4/u4/s8/u8/s32.
     *
     * @param n        N-dimension local index.
     * @param is_tail  Bool flag indicating if tail is processing.
     */
    void load_zero_point(int n, bool is_tail) {
        if (!conf_->is_wei_zp_per_n) return;
        const auto zp_dt = conf_->wei_zp_dt;
        const auto zp_dt_sz = types::data_type_size(zp_dt);
        const auto elems_per_byte
                = one_of(zp_dt, data_type::s4, data_type::u4) ? 2 : 1;
        const auto offset = n * zp_dt_sz / elems_per_byte;
        const auto addr = maybe_EVEX_compress_addr(reg_zp_ptr, offset);

        const bool is_odd_index = n % elems_per_byte == 1;
        const auto tmp_xmm = Xmm(vmm_zp_b_val.getIdx());
        MAYBE_UNUSED(tmp_xmm);
        MAYBE_UNUSED(is_odd_index);

        const bool need_upconvert = one_of(zp_dt, data_type::s8, data_type::u8,
                data_type::s4, data_type::u4);
        if (need_upconvert) {
            uni_vpinsrb(tmp_xmm, tmp_xmm, addr, 0);
            if (one_of(zp_dt, data_type::s8, data_type::s4))
                uni_vpmovsxbd(tmp_xmm, tmp_xmm);
            else
                uni_vpmovzxbd(tmp_xmm, tmp_xmm);

            // 4-bit integer must be shifted left depending
            // which element of 2 is required
            if (one_of(zp_dt, data_type::s4, data_type::u4))
                uni_vpslld(tmp_xmm, tmp_xmm, 28 - is_odd_index * 4);
            // Then shift back to the right on 28 bits
            if (zp_dt == data_type::u4) vpsrld(tmp_xmm, tmp_xmm, 28);
            if (zp_dt == data_type::s4) vpsrad(tmp_xmm, tmp_xmm, 28);
        }
        const auto &op = need_upconvert
                ? static_cast<const Xbyak::Operand &>(tmp_xmm)
                : static_cast<const Xbyak::Operand &>(addr);
        const auto masked_vmm = maybe_mask(vmm_zp_b_val, is_tail);
        uni_vpbroadcastd(masked_vmm, op);
    }

    /**
     * Loads scales and broadcasts it over Vmm register.
     * Supported data types: f32, bf16, f16.
     * 
     * @param n        N-dimension local index.
     * @param is_tail  Bool flag indicating if tail is processing.
     */
    void load_scales(int n, bool is_tail) {
        if (!conf_->is_wei_scale_per_n || !conf_->apply_scales_in_buffer_b)
            return;

        const auto &scales_dt = conf_->wei_scales_dt;
        const auto &scales_dt_sz = conf_->wei_scales_dt_sz;
        const auto offset = n * scales_dt_sz;
        const auto masked_vmm = maybe_mask(vmm_wei_scales, is_tail);
        const auto addr = maybe_EVEX_compress_addr(
                reg_wei_scales, offset, scales_dt == data_type::f16);
        uni_vpxor(vmm_wei_scales, vmm_wei_scales, vmm_wei_scales);
        switch (scales_dt) {
            case data_type::f32: uni_vbroadcastss(vmm_wei_scales, addr); break;
            case data_type::bf16:
                vpbroadcastw(masked_vmm, addr);
                uni_vpslld(vmm_wei_scales, vmm_wei_scales, 16);
                break;
            case data_type::f16: vcvtph2psx(vmm_wei_scales, addr); break;
            case data_type::e8m0:
                vpbroadcastb(masked_vmm, addr);
                uni_vpmovzxbd(
                        vmm_wei_scales, Xbyak::Xmm(vmm_wei_scales.getIdx()));
                uni_vpslld(vmm_wei_scales, vmm_wei_scales, 23);
                break;
            default: assert(!"unsupported wei_scales data type");
        }
    }

    /**
     * Converts packed int4 data in a Ymm (half-ZMM) to int8 in a full ZMM.
     * Used for int8 grouped quantization with int4 weights in the transposed
     * copy_b kernel. The conversion is done in-place, expanding the lower half
     * (Ymm) into the full ZMM register.
     *
     * @param vmm_data The ZMM register containing packed int4 data in its
     *                 lower half (Ymm portion).
     */
    void cvt_int4_to_int8_for_transposed(const Vmm &vmm_data) {
        if (!is_src_int4_) return;

        if (is_ymm_) {
            const auto ymm_data = Ymm(vmm_data.getIdx());
            const auto ymm_high = Ymm(vmm_permd.getIdx());
            const auto ymm_tmp = Ymm(vmm_wei_scales.getIdx());

            uni_vmovups(ymm_high, ymm_data);

            mov(regq_tmp, 0xF0);
            uni_vpbroadcastb(ymm_tmp, regq_tmp.cvt8());
            uni_vpand(ymm_high, ymm_high, ymm_tmp);
            vpsrld(ymm_high, ymm_high, 4);

            mov(regq_tmp, 0x0F);
            uni_vpbroadcastb(ymm_tmp, regq_tmp.cvt8());
            uni_vpand(ymm_data, ymm_data, ymm_tmp);
            uni_vpand(ymm_high, ymm_high, ymm_tmp);

            vpunpcklbw(ymm_tmp, ymm_data, ymm_high);
            vpunpckhbw(ymm_high, ymm_data, ymm_high);
            vinserti128(ymm_data, ymm_tmp, Xmm(ymm_high.getIdx()), 0x1);

            if (conf_->orig_wei_dt == data_type::s4) {
                mov(regq_tmp, 0xF0);
                uni_vpbroadcastb(ymm_tmp, regq_tmp.cvt8());
                uni_vmovups(ymm_high, ymm_data);
                vpslld(ymm_high, ymm_high, 4);
                uni_vpxor(ymm_high, ymm_high, ymm_tmp);
                vpshufb(ymm_high, ymm_tmp, ymm_high);
                uni_vpxor(ymm_data, ymm_data, ymm_high);
            }
            return;
        }

        // Use tmp_vmm(2) and tmp_vmm(3) as scratch (0 and 1 are used by
        // the transpose algorithm which may be active when loads are
        // interleaved).
        const auto vmm_high = tmp_vmm(2);
        const auto vmm_mask_reg = tmp_vmm(3);

        uni_vmovups(vmm_high, vmm_data);

        mov(regq_tmp, 0xF0);
        uni_vpbroadcastb(vmm_mask_reg, regq_tmp.cvt8());
        uni_vpand(vmm_high, vmm_high, vmm_mask_reg);
        vpsrld(vmm_high, vmm_high, 4);

        mov(regq_tmp, 0x0F);
        uni_vpbroadcastb(vmm_mask_reg, regq_tmp.cvt8());
        uni_vpand(vmm_data, vmm_data, vmm_mask_reg);

        if (has_vpermb_) {
            copy_half_reg(
                    Zmm(vmm_data.getIdx()), Vmm_lower_t(vmm_high.getIdx()));
            // vmm_permd holds the int4 byte-interleave permute table
            vpermb(Zmm(vmm_data.getIdx()), vmm_permd, Zmm(vmm_data.getIdx()));
        } else {
            vpunpcklbw(Zmm(vmm_mask_reg.getIdx()), Zmm(vmm_data.getIdx()),
                    Zmm(vmm_high.getIdx()));
            vpunpckhbw(Zmm(vmm_data.getIdx()), Zmm(vmm_data.getIdx()),
                    Zmm(vmm_high.getIdx()));
            vpermt2d(Zmm(vmm_mask_reg.getIdx()), vmm_permd,
                    Zmm(vmm_data.getIdx()));
            uni_vmovups(vmm_data, vmm_mask_reg);
        }

        // Sign-extend for s4 using LUT-based approach (same as
        // signed_mask_int4 in jit_brgemm_matmul_copy_b_int8_t):
        // For signed int4, values 8..15 should become -8..-1.
        // After unpacking, each byte has value 0..15. Need to OR
        // with 0xF0 for values >= 8 (bit 3 set).
        if (conf_->orig_wei_dt == data_type::s4) {
            // Broadcast 0xF0 into vmm_mask_reg (LUT)
            mov(regq_tmp, 0xF0);
            uni_vpbroadcastb(vmm_mask_reg, regq_tmp.cvt8());
            // vmm_high = vmm_data << 4 (shift to make bit 3 into bit 7)
            uni_vmovups(vmm_high, vmm_data);
            vpslld(vmm_high, vmm_high, 4);
            // XOR with 0xF0 to create lookup index
            uni_vpxor(vmm_high, vmm_high, vmm_mask_reg);
            // vpshufb: for each byte, if high bit set -> 0, else -> 0xF0
            // This gives 0xF0 for values >= 8 and 0 for values < 8
            vpshufb(vmm_high, vmm_mask_reg, vmm_high);
            // OR into data to sign-extend
            vpord(Zmm(vmm_data.getIdx()), Zmm(vmm_data.getIdx()),
                    Zmm(vmm_high.getIdx()));
        }
    }

    /**
     * Applies common (broadcast) zero point shift for int8 grouped
     * quantization. Subtracts a single ZP value from all bytes.
     *
     * @param vmm_src Source data register to apply ZP shift to.
     */
    void maybe_apply_int8_common_zp(const Vmm &vmm_src) {
        if (!req_int8_zp_b_shift_ || !conf_->is_wei_zp_common) return;

        const auto vmm_zp = vmm_zp_b_val;
        // vmm_zp_b_val already loaded with broadcast common ZP value
        // in generate() via load_int8_grouped_common_zp()
        uni_vpsubb(vmm_src, vmm_src, vmm_zp);
    }

    /**
     * Applies per-N zero point shift for int8 grouped quantization.
     * For each row i (N dimension), loads the ZP byte and broadcasts+subtracts.
     *
     * @param vmm_src Source data register to apply ZP shift to.
     * @param n The N-dimension local index (row within N block).
     */
    void maybe_apply_int8_per_n_zp(const Vmm &vmm_src, int n) {
        if (!req_int8_zp_b_shift_ || !conf_->is_wei_zp_per_n
                || conf_->is_wei_zp_per_k)
            return;

        const auto vmm_zp = vmm_zp_b_val;
        // Save reg_src_base since we need rax for address computation
        mov(ptr[rsp + reg_src_save_offs_], reg_src_base);
        mov(reg_src_base, ptr[rsp + reg_zp_b_ptr_offs_]);

        const auto zp_dt_sz = types::data_type_size(conf_->wei_zp_dt);
        const auto elems_per_byte = is_zp_int4_ ? 2 : 1;
        const auto offset = n * zp_dt_sz / elems_per_byte;

        if (is_zp_int4_) {
            const bool is_odd = n % 2 == 1;
            movzx(regq_tmp.cvt32(), byte[reg_src_base + offset]);
            if (is_odd)
                shr(regq_tmp.cvt32(), 4);
            else
                and_(regq_tmp.cvt32(), 0x0F);
            if (conf_->wei_zp_dt == data_type::s4) {
                // Sign extend from 4-bit
                shl(regq_tmp.cvt32(), 28);
                sar(regq_tmp.cvt32(), 28);
            }
        } else {
            movzx(regq_tmp.cvt32(), byte[reg_src_base + offset]);
        }
        uni_vpbroadcastb(vmm_zp, regq_tmp.cvt8());
        uni_vpsubb(vmm_src, vmm_src, vmm_zp);
        mov(reg_src_base, ptr[rsp + reg_src_save_offs_]);
    }

    /**
     * Applies per-K (grouped) zero point shift for int8 grouped quantization.
     * Computes the K-group offset, loads the ZP for the current (K-group, N),
     * and applies vpsubb.
     *
     * @param vmm_src Source data register to apply ZP shift to.
     * @param n The N-dimension local index (row within N block).
     */
    void maybe_apply_int8_per_k_zp(const Vmm &vmm_src, int n) {
        if (!req_int8_zp_b_shift_ || !conf_->is_wei_zp_per_k) return;

        const auto vmm_zp = vmm_zp_b_val;
        // Save rax and rdx since division clobbers them.
        // reg_src_base = rax, reg_comp_ptr/reg_zp_ptr = rdx
        mov(ptr[rsp + reg_src_save_offs_], reg_src_base);
        push(rdx);

        // Locate the current K group:
        // k_group_idx = current_K_position / wei_zp_k_gsize
        // ZP offset = k_group_idx * N + n
        xor_(rdx, rdx); // zero rdx for division
        mov(rax, ptr[rsp + 8 /* push offset */ + reg_cur_k_offs_]);
        mov(regq_tmp, conf_->wei_zp_k_gsize);
        div(regq_tmp);
        // rax = k_group_idx
        mov(regq_tmp, conf_->N);
        mul(regq_tmp);
        // rax = k_group_idx * N

        if (is_zp_int4_) shr(rax, 1);

        // Load ZP pointer and add group offset + N offset
        mov(regq_tmp, ptr[rsp + 8 /* push offset */ + reg_zp_b_ptr_offs_]);
        add(regq_tmp, rax);
        if (is_zp_int4_)
            add(regq_tmp, n / 2);
        else
            add(regq_tmp, n);

        if (is_zp_int4_) {
            const bool is_odd = n % 2 == 1;
            movzx(eax, byte[regq_tmp]);
            if (is_odd)
                shr(eax, 4);
            else
                and_(eax, 0x0F);
            if (conf_->wei_zp_dt == data_type::s4) {
                shl(eax, 28);
                sar(eax, 28);
            }
            mov(regq_tmp, rax);
            uni_vpbroadcastb(vmm_zp, regq_tmp.cvt8());
        } else {
            movzx(eax, byte[regq_tmp]);
            mov(regq_tmp, rax);
            uni_vpbroadcastb(vmm_zp, regq_tmp.cvt8());
        }
        uni_vpsubb(vmm_src, vmm_src, vmm_zp);
        // Restore regs
        pop(rdx);
        mov(reg_src_base, ptr[rsp + reg_src_save_offs_]);
    }

    /**
     * Top-level ZP shift dispatcher for int8 grouped quantization.
     * Called from the load lambda after loading int8 data.
     *
     * @param vmm_src Source data register to apply ZP shift to.
     * @param n The N-dimension local index (row within N block).
     */
    void maybe_apply_int8_grouped_zp(const Vmm &vmm_src, int n) {
        if (!req_int8_zp_b_shift_) return;

        maybe_apply_int8_common_zp(vmm_src);
        maybe_apply_int8_per_n_zp(vmm_src, n);
        maybe_apply_int8_per_k_zp(vmm_src, n);
    }

    /**
     * Loads and broadcasts the common ZP value as a byte for int8 grouped
     * quantization. Called from generate().
     */
    void load_int8_common_zp() {
        if (!req_int8_zp_b_shift_) return;

        const bool only_per_k
                = conf_->is_wei_zp_per_k && !conf_->is_wei_zp_per_n;
        const bool is_common_or_per_k = conf_->is_wei_zp_common || only_per_k;
        if (!is_common_or_per_k) return;

        mov(regq_tmp, ptr[rsp + reg_zp_b_ptr_offs_]);
        mov(regq_tmp.cvt8(), byte[regq_tmp]);
        if (is_zp_int4_) {
            // Extract int4 value from the low nibble
            and_(regq_tmp.cvt32(), 0x0F);
            if (conf_->wei_zp_dt == data_type::s4) {
                shl(regq_tmp.cvt32(), 28);
                sar(regq_tmp.cvt32(), 28);
            }
        }
        uni_vpbroadcastb(vmm_zp_b_val, regq_tmp.cvt8());
    }

    /** Stores half of the block using mask for the case when vnni_granularity == 2 */
    void store_half_block(const Zmm &r, const Xbyak::Address &store_addr) {
        Label even_k, end_permute;
        test(reg_K_start, 1);
        jz(even_k, T_NEAR);
        // Shift left by 16 bytes to store odd indices
        uni_vpslld(r, r, 16);
        vmovdqu16(store_addr | kAAAA, r);
        jmp(end_permute);
        L(even_k);
        // Store even indices, odd set to zero.
        vmovdqu16(store_addr, r);
        L(end_permute);
    }

    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::init_tail_mask(
        const int columns_tail, const bool use_4bit_mask) {
    assert(IMPLICATION(use_4bit_mask, is_src_4bit_));
    assert(isa_has_masks(conf_->isa));
    if (columns_tail > 0) {
        const int dt_step = req_cvtps2xf16_ || use_fp16_instructions_
                        || use_bf16_instructions_
                ? 1
                : typesize_;
        const auto tail_mask = use_4bit_mask
                ? size_t(((size_t)1 << div_up(dt_step * columns_tail, 2)) - 1)
                : size_t(((size_t)1 << dt_step * columns_tail) - 1);
        if (req_cvtps2xf16_)
            kmovw(kTail, static_cast<uint32_t>(tail_mask));
        else
            kmovq(kTail, tail_mask);
    }
}

/**
 * @brief Handles special case when loading 4-bit data with unaligned src_stride
 *
 * When processing 4-bit data and (i * src_stride_) % 2 != 0, we need to perform additional operations
 * to handle the unaligned half-byte boundaries:
 *
 * - For small loads (< 8 bytes or tail cases): We perform a simple right shift by 4 bits
 *   to eliminate the unnecessary half-byte at the front
 *
 * - For large loads (8 bytes): We need two registers to correctly reconstruct the data:
 *   1. Shift the first register right by 4 bits to remove leading half-byte
 *   2. Shift the second register left by 4 bits to preserve trailing half-byte
 *   3. Combine both registers with logical OR to get the correctly aligned result
 *
 * @param xmm_in The target XMM register to store the aligned data
 * @param i Current row index
 * @param columns_tail Number of remaining columns in the tail case
 * @param is_tail Flag indicating if processing a tail section
 * @param offset Memory offset for loading from source
 * @return True if 4-bit preloading was performed, false otherwise
 */
template <typename Vmm>
bool jit_brgemm_matmul_copy_b_transposed_t<Vmm>::preload_4bit(const Xmm &xmm_in,
        const int i, const int columns_tail, const bool is_tail,
        const dim_t offset) {
    const auto addr = maybe_EVEX_compress_addr(reg_src, offset);
    const bool need_preload_4bit = is_src_4bit_ && (i * src_stride_) % 2 != 0;
    const auto max_shift_sz = 8;
    if (need_preload_4bit) {
        const auto load_sz = is_tail ? div_up(columns_tail, 2)
                : req_cvtps2xf16_    ? req_cvt_bf16_k_blk_step_ / 2
                                     : k_blk_step_ / 2;
        assert(load_sz <= max_shift_sz);
        if (load_sz < max_shift_sz || is_tail) {
            load_bytes(xmm_in, addr, load_sz);
            vpsrlq(xmm_in, xmm_in, 4);
        } else {
            const auto xmm_tmp = Xmm(tmp_vmm(3).getIdx());
            load_bytes(xmm_in, addr, load_sz);
            load_bytes(xmm_tmp, maybe_EVEX_compress_addr(reg_src, offset + 1),
                    load_sz);
            vpsrlq(xmm_in, xmm_in, 4);
            vpsllq(xmm_tmp, xmm_tmp, 4);
            vpord(xmm_in, xmm_in, xmm_tmp);
        }
        return true;
    }

    // The case when the kernel is grouped over K and need to load odd or even columns
    const auto preload_for_k_1_blk
            = is_src_int4_ && is_wei_grouped_over_k_ && columns_tail == 1;
    if (preload_for_k_1_blk) {
        // Unconditionally load 1 byte, then shift if odd index
        load_bytes(xmm_in, addr, 1);
        Label load_done, even_k;
        test(reg_K_start, 1);
        jz(even_k, T_NEAR);
        vpsrlq(xmm_in, xmm_in, 4);
        jmp(load_done);
        L(even_k);
        vpsllq(xmm_in, xmm_in, 4);
        vpsrlq(xmm_in, xmm_in, 4);
        L(load_done);
        return true;
    }
    return false;
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const auto cur_k_blk_step
            = req_cvtps2xf16_ ? req_cvt_bf16_k_blk_step_ : k_blk_step_;

    const int columns_tail = ncolumns % cur_k_blk_step;

    init_tail_mask(columns_tail, false);

    auto load2bf16 = [this, nrows, columns_tail, ncolumns](int i) {
        auto src_reg = src_vmm(i);
        auto src_reg_next = tmp_vmm(2);

        Label load_done;
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            uni_vpxor(src_reg, src_reg, src_reg);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            uni_vpxor(src_reg, src_reg, src_reg);
            return;
        }

        // check if k_tail exists and it's in the first zmm
        const auto is_tail
                = columns_tail > 0 && ncolumns < req_cvt_bf16_k_blk_step_;
        auto src_reg_masked = maybe_mask(src_reg, is_tail);
        const auto src_offset = (i * src_stride_) / src_elems_per_byte_;
        const auto addr = EVEX_compress_addr(reg_src, src_offset);
        if (is_bf32_)
            vmovups(src_reg_masked, addr);
        else if (is_bf16_with_int_wei_ || conf_->is_f16_with_int_wei) {
            const auto xmm_preload = Xmm(src_reg.getIdx());
            MAYBE_UNUSED(xmm_preload);
            const bool preloaded_4bit = preload_4bit(
                    xmm_preload, i, columns_tail, is_tail, src_offset);
            const auto &src_op = preloaded_4bit
                    ? static_cast<const Xbyak::Operand &>(xmm_preload)
                    : static_cast<const Xbyak::Operand &>(addr);
            if (is_src_4bit_) init_tail_mask(columns_tail, true);
            load_value(src_reg, src_op, vmm_permd, conf_->orig_wei_dt, is_tail,
                    vmm_f4_lookup_table);
            if (is_src_4bit_) init_tail_mask(columns_tail, false);
            load_zero_point(i, is_tail);
            load_scales(i, is_tail);
            decompress_reg(src_reg_masked, vmm_zp_b_val, vmm_wei_scales,
                    conf_->orig_wei_dt);
        } else
            assert(!"Unsupported data type in loading");

        if (ncolumns <= req_cvt_bf16_k_blk_step_) {
            uni_vpxor(src_reg_next, src_reg_next, src_reg_next);
        } else {
            const auto is_tail = columns_tail > 0;
            auto src_next_masked = maybe_mask(src_reg_next, is_tail);
            const auto next_src_offset
                    = (i * src_stride_ + req_cvt_bf16_k_blk_step_ * typesize_)
                    / src_elems_per_byte_;
            const auto next_addr = EVEX_compress_addr(reg_src, next_src_offset);
            if (is_bf32_)
                vmovups(src_next_masked, next_addr);
            else if (is_bf16_with_int_wei_ || conf_->is_f16_with_int_wei) {
                const auto xmm_preload = Xmm(src_reg_next.getIdx());
                MAYBE_UNUSED(xmm_preload);
                const bool preloaded_4bit = preload_4bit(
                        xmm_preload, i, columns_tail, is_tail, src_offset);
                const auto &src_op = preloaded_4bit
                        ? static_cast<const Xbyak::Operand &>(xmm_preload)
                        : static_cast<const Xbyak::Operand &>(next_addr);
                if (is_src_4bit_) init_tail_mask(columns_tail, true);
                load_value(src_reg_next, src_op, vmm_permd, conf_->orig_wei_dt,
                        is_tail, vmm_f4_lookup_table);
                if (is_src_4bit_) init_tail_mask(columns_tail, false);
                load_zero_point(i, is_tail);
                load_scales(i, is_tail);
                decompress_reg(src_next_masked, vmm_zp_b_val, vmm_wei_scales,
                        conf_->orig_wei_dt);
            } else
                assert(!"Unsupported data type in loading");
        }
        downconvert_to_dst_dt(src_reg, src_reg_next, conf_->wei_dt);
        L(load_done);
    };

    auto load = [this, nrows, columns_tail](int i, int base_idx) {
        Label load_done;

        auto src_reg = src_vmm(i);
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            uni_vpxor(src_reg, src_reg, src_reg);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            uni_vpxor(src_reg, src_reg, src_reg);
            return;
        }

        const auto is_tail = columns_tail > 0;
        const auto src_offset = (i * src_stride_) / src_elems_per_byte_;
        const auto addr = EVEX_compress_addr(reg_src, src_offset);
        auto src_masked_reg = maybe_mask(src_reg, is_tail);
        if ((conf_->is_f16_with_int_wei || conf_->is_f32_with_int_wei
                    || conf_->is_f32_with_f4_wei)
                && conf_->wei_dt == data_type::f32) {
            const auto xmm_preload = Xmm(src_reg.getIdx());
            MAYBE_UNUSED(xmm_preload);
            const bool preloaded_4bit = preload_4bit(
                    xmm_preload, i, columns_tail, is_tail, src_offset);
            const auto &src_op = preloaded_4bit
                    ? static_cast<const Xbyak::Operand &>(xmm_preload)
                    : static_cast<const Xbyak::Operand &>(addr);
            if (is_src_4bit_) init_tail_mask(columns_tail, true);
            load_value(src_reg, src_op, vmm_permd, conf_->orig_wei_dt, is_tail,
                    vmm_f4_lookup_table);
            if (is_src_4bit_) init_tail_mask(columns_tail, false);
            load_zero_point(i, is_tail);
            load_scales(i, is_tail);
            decompress_reg(src_masked_reg, vmm_zp_b_val, vmm_wei_scales,
                    conf_->orig_wei_dt);
        } else if (use_fp16_instructions_) {
            if (conf_->isa == avx512_core_fp16) {
                vcvtph2psx(src_masked_reg, addr);
            } else {
                vcvtph2ps(src_masked_reg, addr);
            }
        } else if (use_bf16_instructions_) {
            // Upconvert: load 16 bits and move them 16 bits left.
            uni_vpmovzxwd(src_masked_reg, addr);
            uni_vpslld(src_masked_reg, src_masked_reg, 16);
        } else if (req_cvtf82f16_) {
            f8_to_f16_upconvert(
                    conf_->orig_wei_dt, src_masked_reg, addr, is_tail);
        } else if (conf_->with_int8_grouped_quantization && is_src_int4_) {
            // Int8 grouped quantization with int4 weights:
            // Load packed int4 data into Ymm (half of Zmm), then
            // unpack to full int8 in Zmm and apply ZP shift.
            auto ymm_src = Ymm(src_reg.getIdx());
            if (is_tail) init_tail_mask(columns_tail, /*use_int4_mask=*/true);
            auto ymm_src_load = maybe_mask(ymm_src, is_tail);
            vmovdqu8(ymm_src_load, addr);
            if (is_tail) init_tail_mask(columns_tail, /*use_int4_mask=*/false);
            cvt_int4_to_int8_for_transposed(src_reg);
            maybe_apply_int8_grouped_zp(src_reg, i);
        } else {
            vmovdqu8(src_masked_reg, addr);
            if (req_int8_zp_b_shift_) maybe_apply_int8_grouped_zp(src_reg, i);
        }
        L(load_done);
    };

    auto store = [this, ncolumns](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride_);
        if (is_wei_grouped_over_k_) {
            const int valid_stores = utils::div_up(ncolumns, vnni_granularity_);
            if (i >= valid_stores) return;
            if (vnni_granularity_ == 2 && ncolumns == 1)
                store_half_block(r, addr);
            else
                vmovups(addr, r);
        } else {
            vmovups(addr, r);
        }
    };

    auto transpose16x8 = [&](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        if (req_cvtps2xf16_) {
            for (int i = 0; i < 4; i++) {
                const int src_idx0 = base_idx + i * 2;
                const int src_idx1 = src_idx0 + 1;

                if (base_idx == 0 && i == 0) {
                    load2bf16(src_idx0);
                    load2bf16(src_idx1);
                }

                const int next_src_idx0 = src_idx0 + 2;
                const int next_src_idx1 = src_idx1 + 2;

                const bool load_next = base_idx == 0 || i < 3;

                const auto tmp0 = tmp_vmm(0);
                const auto tmp1 = tmp_vmm(1);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (valid_to_load_next(next_src_idx0, nrows) && load_next)
                    load2bf16(next_src_idx0);
                valignd(tmp0, src0, src0, 0x1);

                if (valid_to_load_next(next_src_idx1, nrows) && load_next)
                    load2bf16(next_src_idx1);
                valignd(tmp1, src1, src1, 0xf);

                vmovaps(src0 | kAAAA, tmp1);
                vmovaps(src1 | k5555, tmp0);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                const int src_idx0 = base_idx + i * 2;
                const int src_idx1 = src_idx0 + 1;

                const int next_src_idx0 = src_idx0 + 2;
                const int next_src_idx1 = src_idx1 + 2;
                const bool load_next = base_idx == 0 || i < 3;

                if (base_idx == 0 && i == 0) {
                    load(src_idx0, base_idx);
                    load(src_idx1, base_idx);
                }

                const auto tmp0 = tmp_vmm(0);
                const auto tmp1 = tmp_vmm(1);
                const auto src0 = src_vmm(src_idx0);
                const auto src1 = src_vmm(src_idx1);

                if (valid_to_load_next(next_src_idx0, nrows) && load_next)
                    load(next_src_idx0, base_idx);
                valignd(tmp0, src0, src0, 0x1);

                if (valid_to_load_next(next_src_idx1, nrows) && load_next)
                    load(next_src_idx1, base_idx);
                valignd(tmp1, src1, src1, 0xf);

                vmovaps(src0 | kAAAA, tmp1);
                vmovaps(src1 | k5555, tmp0);
            }
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            const int select_half = (i < 2) ? 0 : 2;
            const int src_idx0 = base_idx + i + select_half + 0;
            const int src_idx2 = src_idx0 + 2;

            const auto tmp0 = tmp_vmm(0);
            const auto tmp1 = tmp_vmm(1);
            const auto src0 = src_vmm(src_idx0);
            const auto src2 = src_vmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }
        // swap 4
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i;
            const int src_idx4 = src_idx0 + 4;

            const auto tmp0 = tmp_vmm(0);
            const auto src0 = src_vmm(src_idx0);
            const auto src4 = src_vmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [&]() {
        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(0);
            const auto src0 = src_vmm(i);
            const auto src8 = src_vmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            if (do_compute_compensation_)
                dot_product(vmm_comp_acc, vmm_comp_mul, tmp);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_vmm(0);
            const auto src0 = src_vmm(i);
            const auto src8 = src_vmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            if (do_compute_compensation_)
                dot_product(vmm_comp_acc, vmm_comp_mul, tmp);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

template <>
void jit_brgemm_matmul_copy_b_transposed_t<Ymm>::copy_row_x_col(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= n_blk_step_ && ncolumns >= 0
            && ncolumns <= k_blk_step_);
    if (!nrows) return;

    const int columns_tail = ncolumns % k_blk_step_;

    auto load = [this, nrows, columns_tail](int i) {
        auto vmm_src = src_vmm(i);

        Label load_done;
        if (is_dynamic_N_ && nrows < n_blk_step_) {
            Label general_load;
            cmp(reg_N_iters, i);
            jg(general_load); // i < dynamic nrows -> general load

            // i >= dynamic nrows -> zero out values in src_reg
            uni_vpxor(vmm_src, vmm_src, vmm_src);
            jmp(load_done);

            L(general_load);
        } else if (i >= nrows) {
            uni_vpxor(vmm_src, vmm_src, vmm_src);
            return;
        }

        const bool is_tail = columns_tail > 0;

        if (conf_->is_f32_with_int_wei) {
            const auto src_offset = (i * src_stride_) / src_elems_per_byte_;
            const auto addr = maybe_EVEX_compress_addr(reg_src, src_offset);
            const auto xmm_preload = Xmm(vmm_src.getIdx());
            MAYBE_UNUSED(xmm_preload);
            const bool preloaded_4bit = preload_4bit(
                    xmm_preload, i, columns_tail, is_tail, src_offset);
            const auto &src_op = preloaded_4bit
                    ? static_cast<const Xbyak::Operand &>(xmm_preload)
                    : static_cast<const Xbyak::Operand &>(addr);

            if (is_tail && !preloaded_4bit) {
                load_bytes(vmm_src, addr,
                        columns_tail * typesize_ / src_elems_per_byte_);
                load_value(vmm_src, vmm_src, vmm_permd, conf_->orig_wei_dt,
                        false, vmm_f4_lookup_table);
            } else {
                load_value(vmm_src, src_op, vmm_permd, conf_->orig_wei_dt,
                        is_tail, vmm_f4_lookup_table);
            }

            load_zero_point(i, is_tail);
            load_scales(i, is_tail);
            decompress_reg(
                    vmm_src, vmm_zp_b_val, vmm_wei_scales, conf_->orig_wei_dt);
        } else if (conf_->with_int8_grouped_quantization && is_src_int4_) {
            const auto src_offset = (i * src_stride_) / src_elems_per_byte_;
            const auto addr = maybe_EVEX_compress_addr(reg_src, src_offset);
            const int packed_load_sz = is_tail
                    ? div_up(columns_tail * typesize_, src_elems_per_byte_)
                    : (k_blk_step_ * typesize_) / src_elems_per_byte_;

            uni_vpxor(vmm_src, vmm_src, vmm_src);

            // Int4 rows may start from an odd nibble, so align packed bytes
            // before expanding nibbles to signed/unsigned int8 values.
            const bool need_align_half_byte = (i * src_stride_) % 2 != 0;
            const auto xmm_src = Xmm(vmm_src.getIdx());
            if (need_align_half_byte) {
                if (packed_load_sz < 16 || is_tail) {
                    load_bytes(xmm_src, addr, packed_load_sz);
                    vpsrlq(xmm_src, xmm_src, 4);
                } else {
                    const auto xmm_tmp = Xmm(vmm_permd.getIdx());
                    load_bytes(xmm_src, addr, packed_load_sz);
                    load_bytes(xmm_tmp,
                            maybe_EVEX_compress_addr(reg_src, src_offset + 1),
                            packed_load_sz);
                    vpsrlq(xmm_src, xmm_src, 4);
                    vpsllq(xmm_tmp, xmm_tmp, 4);
                    vpor(xmm_src, xmm_src, xmm_tmp);
                }
            } else {
                load_bytes(xmm_src, addr, packed_load_sz);
            }

            cvt_int4_to_int8_for_transposed(vmm_src);

            maybe_apply_int8_grouped_zp(vmm_src, i);
        } else if (is_tail) {
            load_bytes(vmm_src, reg_src, i * src_stride_,
                    columns_tail * typesize_);
            if (use_fp16_instructions_) {
                // For f32:f16 case need to convert raw bytes after `load_bytes`
                // into f32 values.
                vcvtph2ps(vmm_src, Xmm(vmm_src.getIdx()));
            } else if (use_bf16_instructions_) {
                // Upconvert: move loaded 16 bits left.
                uni_vpmovzxwd(vmm_src, vmm_src);
                uni_vpslld(vmm_src, vmm_src, 16);
            }
        } else {
            if (use_fp16_instructions_) {
                // For non-tailed case can use the convert instruction directly.
                vcvtph2ps(vmm_src, ptr[reg_src + i * src_stride_]);
            } else if (use_bf16_instructions_) {
                // Upconvert: load 16 bits and move them 16 bits left.
                uni_vpmovzxwd(vmm_src, ptr[reg_src + i * src_stride_]);
                uni_vpslld(vmm_src, vmm_src, 16);
            } else {
                uni_vmovups(vmm_src, ptr[reg_src + i * src_stride_]);
            }
        }

        L(load_done);
    };

    // swap 1
    for (int i = 0; i < 4; ++i) {
        const int src_idx0 = i * 2;
        const int src_idx1 = src_idx0 + 1;

        const int next_src_idx0 = src_idx0 + 2;
        const int next_src_idx1 = src_idx1 + 2;
        const bool load_next = i < 3;

        if (i == 0) {
            load(src_idx0);
            load(src_idx1);
        }

        const auto tmp0 = tmp_vmm(0);
        const auto tmp1 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src1 = src_vmm(src_idx1);

        if (valid_to_load_next(next_src_idx0, nrows) && load_next) {
            load(next_src_idx0);
        }
        vperm2i128(tmp0, src0, src0, 0x1);
        vpalignr(tmp0, tmp0, src0, 0x4);

        if (valid_to_load_next(next_src_idx1, nrows) && load_next) {
            load(next_src_idx1);
        }
        vperm2i128(tmp1, src1, src1, 0x1);
        vpalignr(tmp1, src1, tmp1, 0xC);

        vpblendd(src0, src0, tmp1, 0xAA);
        vpblendd(src1, src1, tmp0, 0x55);
    }
    // swap 2
    for (int i = 0; i < 4; ++i) {
        const int select_half = (i < 2) ? 0 : 2;
        const int src_idx0 = i + select_half;
        const int src_idx2 = src_idx0 + 2;

        const auto tmp0 = tmp_vmm(0);
        const auto tmp1 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src2 = src_vmm(src_idx2);

        vperm2i128(tmp0, src0, src0, 0x1);
        vpalignr(tmp0, tmp0, src0, 0x8);

        vperm2i128(tmp1, src2, src2, 0x1);
        vpalignr(tmp1, src2, tmp1, 0x8);

        vpblendd(src2, src2, tmp0, 0x33);
        vpblendd(src0, src0, tmp1, 0xCC);
    }
    // swap 4
    for (int i = 0; i < 4; ++i) {
        const int src_idx0 = i;
        const int src_idx4 = src_idx0 + 4;

        const auto tmp0 = tmp_vmm(0);
        const auto tmp4 = tmp_vmm(1);
        const auto src0 = src_vmm(src_idx0);
        const auto src4 = src_vmm(src_idx4);

        vperm2i128(tmp4, src4, src4, 0x01);
        vperm2i128(tmp0, src0, src0, 0x01);
        vpblendd(src0, src0, tmp4, 0xF0);
        vpblendd(src4, src4, tmp0, 0x0F);
    }
    // swap 8
    for (int i = 0; i < 8; i++) {
        const auto src0 = src_vmm(i);
        if (do_compute_compensation_)
            dot_product(vmm_comp_acc, vmm_comp_mul, src0);
        const auto addr
                = maybe_EVEX_compress_addr(reg_tr_src, i * tr_src_stride_);
        if (columns_tail > 0 && i >= columns_tail) break;
        uni_vmovups(addr, src0);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::compute_K_loop(bool is_N_tail,
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    MAYBE_UNUSED(is_first_K_iter);
    MAYBE_UNUSED(is_last_K_iter);
    const int N_chunk_tail = is_dynamic_N_
            ? 1 /* just to force tail processing */
            : conf_->N % n_blk_step_;
    const int nrows = is_N_tail ? N_chunk_tail : n_blk_step_;
    if (do_compute_compensation_)
        uni_vpxor(vmm_comp_acc, vmm_comp_acc, vmm_comp_acc);

    Label K_loop, K_loop_tail_or_done;
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);

    // Reset the K-within-group counter to its initial value before each
    // N iteration's K loop, since the K loop increments it.
    if (req_int8_zp_b_shift_ && conf_->is_wei_zp_per_k) {
        mov(regq_tmp, ptr[rsp + reg_init_k_rem_offs_]);
        mov(ptr[rsp + reg_cur_k_offs_], regq_tmp);
    }

    mov(reg_src, reg_src_base);
    mov(reg_tr_src, reg_tr_src_base);

    if (curr_K_tail > 0) {
        cmp(reg_K_iters, k_blk_step_);
        jl(K_loop_tail_or_done, T_NEAR);
    }

    L(K_loop);
    copy_row_x_col(nrows, k_blk_step_);
    add(reg_src, (k_blk_step_ * typesize_) / src_elems_per_byte_);
    add(reg_tr_src, k_blk_step_ / vnni_granularity_ * tr_src_stride_);
    if (req_int8_zp_b_shift_ && conf_->is_wei_zp_per_k)
        add(qword[rsp + reg_cur_k_offs_], k_blk_step_);

    sub(reg_K_iters, k_blk_step_);
    cmp(reg_K_iters, k_blk_step_);
    jge(K_loop, T_NEAR);

    L(K_loop_tail_or_done);

    if (curr_K_tail > 0) copy_row_x_col(nrows, curr_K_tail);

    // For int8 grouped quantization the compensation buffer has a separate
    // slot per K-chunk (driver passes a different `compensation_ptr` for
    // each chunk via `s8s8_comp_k_str`). Each call must therefore produce
    // a complete, finalized comp value (raw sum * 128, sign-flipped, +1)
    // independent of other chunks.
    const bool comp_per_chunk = conf_->with_int8_grouped_quantization;
    const bool do_first = comp_per_chunk || is_first_K_iter;
    const bool do_last = comp_per_chunk || is_last_K_iter;

    if (req_s8s8_comp_) {
        const auto addr = ptr[reg_comp_ptr];
        if (!do_first)
            uni_vpaddd(vmm_s8s8_comp_acc, vmm_comp_acc, addr);
        else
            uni_vmovups(vmm_s8s8_comp_acc, vmm_comp_acc);

        if (do_last) {
            // multiply by 128
            uni_vpslld(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, 7);
            // change sign
            uni_vpandnd(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, vmm_all_bits_1);
            uni_vpaddd(vmm_s8s8_comp_acc, vmm_s8s8_comp_acc, vmm_one_s32);
        }
        uni_vmovups(addr, vmm_s8s8_comp_acc);
    }
    if (req_zp_comp_) {
        const auto addr = ptr[reg_zp_comp_ptr];
        if (!do_first) vpaddd(vmm_comp_acc, vmm_comp_acc, addr);
        if (do_last) uni_vpmulld(vmm_comp_acc, vmm_comp_acc, vmm_zp_a_neg_val);
        uni_vmovups(addr, vmm_comp_acc);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::compute_N_loop(
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    const bool generate_N_tail = is_dynamic_N_ || (conf_->N % n_blk_step_ > 0);

    Label N_loop, N_loop_tail_or_done;
    if (generate_N_tail) {
        cmp(reg_N_iters, n_blk_step_);
        jl(N_loop_tail_or_done, T_NEAR);
    }
    if (conf_->LDB2 > 0) {
        mov(regq_tmp, 0);
        mov(ptr[rsp + ldb_step_idx_offs], regq_tmp);
    }

    L(N_loop);
    compute_K_loop(false, curr_K_tail, is_first_K_iter, is_last_K_iter);

    add(reg_src_base, (n_blk_step_ * src_stride_) / src_elems_per_byte_);
    if (conf_->LDB2 > 0) {
        Label small_increment, ldb_off_done;
        mov(regq_tmp, ptr[rsp + ldb_step_idx_offs]);
        add(regq_tmp, n_blk_step_);

        cmp(regq_tmp, conf_->LDB);
        jne(small_increment, T_NEAR);

        add(reg_tr_src_base,
                -conf_->LDB * vnni_granularity_ * tr_typesize_
                        + n_blk_step_ * vnni_granularity_ * tr_typesize_
                        + conf_->LDB2 * tr_typesize_);
        mov(regq_tmp, 0);
        mov(ptr[rsp + ldb_step_idx_offs], regq_tmp);
        jmp(ldb_off_done, T_NEAR);
        L(small_increment);
        add(reg_tr_src_base, n_blk_step_ * vnni_granularity_ * tr_typesize_);
        mov(ptr[rsp + ldb_step_idx_offs], regq_tmp);
        L(ldb_off_done);
    } else {
        add(reg_tr_src_base, n_blk_step_ * vnni_granularity_ * tr_typesize_);
    }

    if (conf_->is_wei_scale_per_n) {
        const auto &scales_dt_sz = conf_->wei_scales_dt_sz;
        const dim_t offset = n_blk_step_ * scales_dt_sz;
        add(reg_wei_scales, offset);
    }

    if (conf_->is_wei_zp_per_n) {
        const auto &zp_dt = conf_->wei_zp_dt;
        const auto zp_dt_sz = types::data_type_size(zp_dt);
        const auto elems_per_byte
                = one_of(zp_dt, data_type::s4, data_type::u4) ? 2 : 1;
        const dim_t offset = n_blk_step_ * zp_dt_sz / elems_per_byte;
        // For int8 grouped quantization the zp pointer is consumed via the
        // stack-saved copy (because `reg_zp_ptr` aliases `reg_comp_ptr` and
        // gets repurposed by the s8s8/src-zp compensation code path inside
        // `compute_body`). Updating `reg_zp_ptr` here would corrupt the
        // compensation pointer. Only the stack-saved ZP pointer needs to
        // advance per N-block in that case.
        if (req_int8_zp_b_shift_)
            add(qword[rsp + reg_zp_b_ptr_offs_], offset);
        else
            add(reg_zp_ptr, offset);
    }

    if (req_zp_comp_) add(reg_zp_comp_ptr, comp_shift_);
    if (req_s8s8_comp_) add(reg_comp_ptr, comp_shift_);

    sub(reg_N_iters, n_blk_step_);
    cmp(reg_N_iters, n_blk_step_);
    jge(N_loop, T_NEAR);

    L(N_loop_tail_or_done);
    if (generate_N_tail) {
        Label N_loop_done;
        cmp(reg_N_iters, 0);
        jle(N_loop_done, T_NEAR);

        compute_K_loop(true, curr_K_tail, is_first_K_iter, is_last_K_iter);
        L(N_loop_done);
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_transposed_t<Vmm>::generate() {
    preamble();
    sub(rsp, stack_space_needed);

    if (avx512_core_dot_product_) {
        mov(regq_tmp.cvt16(), 1);
        vpbroadcastw(vmm_ones_words, regq_tmp.cvt16());
    }

    // Compute the K-within-group remainder BEFORE loading reg_src_base (rax)
    // and reg_zp_ptr (rdx), since division clobbers both rax and rdx.
    if (req_int8_zp_b_shift_ && conf_->is_wei_zp_per_k) {
        // Store the K position within the current group (remainder).
        // Since zp_b_value_ptr already points to the current K group,
        // we need the intra-group offset to compute relative group
        // crossings during the K loop.
        mov(regq_tmp, conf_->wei_zp_k_gsize);
        xor_(rdx, rdx);
        mov(rax, ptr[param1 + GET_OFF(current_K_start)]);
        div(regq_tmp);
        // rdx = current_K_start % k_gsize (position within group)
        mov(ptr[rsp + reg_cur_k_offs_], rdx);
        mov(ptr[rsp + reg_init_k_rem_offs_], rdx);
    }

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_iters, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_wei_scales, ptr[param1 + GET_OFF(wei_scales_ptr)]);
    mov(reg_zp_ptr, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
    mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);

    // Save ZP pointer for int8 grouped quantization.
    if (req_int8_zp_b_shift_) {
        mov(ptr[rsp + reg_zp_b_ptr_offs_], reg_zp_ptr);
        if (!conf_->is_wei_zp_per_k) {
            mov(ptr[rsp + reg_cur_k_offs_], reg_K_start);
        }
    }

    if (!is_ymm_) {
        // 64-bit mask is also used when is_wei_[zp\scales]_per_k
        kmovq(kAAAA, 0xAAAAAAAAAAAAAAAA);
        kmovq(k5555, 0x5555555555555555);

        kmovw(k3333, 0x3333);
        kmovw(kCCCC, 0xcccc);
        kmovw(k0F0F, 0x0f0f);
        kmovw(kF0F0, 0xf0f0);
    }
    if (is_src_4bit_) {
        if (is_src_int4_ && conf_->with_int8_grouped_quantization
                && is_superset(conf_->isa, avx512_core)) {
            // For int8 grouped quantization with int4 weights, we need a
            // byte-level interleave permute table for cvt_int4_to_int8.
            if (has_vpermb_) {
                alignas(64) static constexpr const uint8_t
                        int4_permute_bytes[64]
                        = {0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7,
                                39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13,
                                45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19,
                                51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25,
                                57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31,
                                63};
                mov(regq_tmp, reinterpret_cast<size_t>(int4_permute_bytes));
                vmovdqa64(vmm_permd, ptr[regq_tmp]);
            } else {
                alignas(64) static constexpr const uint32_t int4_permute_dw[16]
                        = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22,
                                23};
                mov(regq_tmp, reinterpret_cast<size_t>(int4_permute_dw));
                vmovdqa32(vmm_permd, ptr[regq_tmp]);
            }
        } else if (is_superset(conf_->isa, avx512_core)) {
            alignas(64) static constexpr const uint32_t int4_permute[16]
                    = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
            mov(regq_tmp, reinterpret_cast<size_t>(int4_permute));
            vmovdqa32(vmm_permd, ptr[regq_tmp]);
        } else if (is_superset(conf_->isa, avx2)) {
            alignas(64) static constexpr const uint32_t int4_permute_avx2[8]
                    = {0, 4, 1, 5, 2, 6, 3, 7};
            mov(regq_tmp, reinterpret_cast<size_t>(int4_permute_avx2));
            vmovdqa(vmm_permd, ptr[regq_tmp]);
        }
    }
    if (is_src_f4_ && is_superset(conf_->isa, avx512_core)) {
        alignas(64) static constexpr const float f4_e2m1_table[16]
                = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -.5f,
                        -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        mov(regq_tmp, reinterpret_cast<size_t>(f4_e2m1_table));
        vmovdqa32(vmm_f4_lookup_table, ptr[regq_tmp]);
    }

    load_common_zp_value(vmm_zp_b_val, reg_zp_ptr);
    load_common_scale_value(vmm_wei_scales, reg_wei_scales);
    // For int8 grouped quantization, load common ZP as byte broadcast
    load_int8_common_zp();

    const dim_t N_chunk_elems = conf_->N_chunk_elems;
    assert(N_chunk_elems % n_blk_step_ == 0 || N_chunk_elems == conf_->N);
    UNUSED(N_chunk_elems);

    const auto &k_blk = conf_->K_blk;
    const auto K_blk_tail = nstl::min(conf_->K, k_blk) % k_blk_step_;
    const auto K_tail_tail = (conf_->K % k_blk) % k_blk_step_;

    const auto grouped_k = is_wei_grouped_over_k_
            ? (conf_->is_wei_zp_per_k ? conf_->wei_zp_k_gsize
                                      : conf_->wei_scales_k_gsize)
            : 0;

    auto compute_body = [&](bool is_first_K_iter, bool is_last_K_iter) {
        // For int8 grouped quantization, every call finalizes its own
        // per-chunk comp (see comment in compute_K_loop), so the comp
        // constants must be available regardless of is_last_K_iter.
        const bool need_comp_constants
                = is_last_K_iter || conf_->with_int8_grouped_quantization;
        if (need_comp_constants) {
            if (req_s8s8_comp_) {
                mov(imm_addr64, 0xffffffff);
                uni_vpbroadcastd(vmm_all_bits_1, imm_addr64.cvt32());
                mov(imm_addr64, 0x1);
                uni_vpbroadcastd(vmm_one_s32, imm_addr64.cvt32());
            }
            if (req_zp_comp_) {
                mov(reg_zp_a_neg_val_ptr,
                        ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
                uni_vbroadcastss(vmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);
            }
        }

        if (is_wei_grouped_over_k_ && grouped_k < k_blk_step_) {
            compute_N_loop(static_cast<int>(grouped_k), is_first_K_iter,
                    is_last_K_iter);
            return;
        }

        Label compute_body_done;
        if (conf_->K_tail > 0 && K_blk_tail != K_tail_tail) {
            Label not_K_tail;
            cmp(reg_K_iters, k_blk);
            je(not_K_tail, T_NEAR);
            compute_N_loop(static_cast<int>(K_tail_tail), is_first_K_iter,
                    is_last_K_iter);
            jmp(compute_body_done, T_NEAR);

            L(not_K_tail);
        }

        compute_N_loop(
                static_cast<int>(K_blk_tail), is_first_K_iter, is_last_K_iter);
        L(compute_body_done);
    };

    Label done;
    if (do_compute_compensation_) {
        assert(IMPLICATION(req_zp_comp_,
                conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        if (req_s8s8_comp_)
            mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        if (req_zp_comp_)
            mov(reg_zp_comp_ptr, ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);

        mov(regq_tmp, 1);
        uni_vpbroadcastb(vmm_comp_mul, regq_tmp.cvt8());

        const auto last_K_threshold = rnd_up(conf_->K, k_blk) - k_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            compute_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            compute_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        compute_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }

    compute_body(false, false);
    L(done);

    add(rsp, stack_space_needed);
    postamble();
}

template struct jit_brgemm_matmul_copy_b_transposed_t<Zmm>;
template struct jit_brgemm_matmul_copy_b_transposed_t<Ymm>;

template <typename Vmm>
struct jit_brgemm_matmul_copy_b_cvt_bf16_t
    : public jit_brgemm_matmul_copy_b_common_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_cvt_bf16_t)

    jit_brgemm_matmul_copy_b_cvt_bf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_common_t(conf)
        , typesize_(static_cast<int>(conf->b_dt_sz))
        , tr_typesize_(static_cast<int>(conf->tr_b_dt_sz))
        , wei_scales_typesize_(static_cast<int>(conf_->wei_scales_dt_sz))
        , is_src_int4_(one_of(conf->orig_wei_dt, data_type::s4, data_type::u4))
        , src_elems_per_byte_(is_src_int4_ ? 2 : 1)
        , src_stride_(
                  (conf->LDB * k_blk_step * typesize_) / src_elems_per_byte_)
        , tr_src_stride_(conf_->LDB * k_blk_step * tr_typesize_)
        , req_cvtf82f16_(one_of(conf->orig_wei_dt, data_type::f8_e5m2,
                                 data_type::f8_e4m3)
                  && conf_->wei_dt == data_type::f16 && conf_->isa == avx10_2)
        , req_zp_b_shift_(
                  conf_->has_zero_point_b && conf_->with_wei_decompression)
        , req_apply_wei_scales_(conf_->apply_scales_in_buffer_b)
        , is_wei_grouped_over_k_(
                  conf_->is_wei_zp_per_k || conf_->is_wei_scale_per_k)
        , vmm_indices_([&]() {
            vmm_idx_t idx;
            int n = 0;
            if (is_src_int4_ || req_zp_b_shift_ || req_apply_wei_scales_)
                idx.permd = n++;
            if (req_zp_b_shift_) {
                idx.zp_b_val0 = n++;
                idx.zp_b_val1 = n++;
            }
            if (req_zp_b_shift_ || req_apply_wei_scales_
                    || is_wei_grouped_over_k_)
                idx.tmp = n++;
            if (req_apply_wei_scales_) {
                idx.wei_scales0 = n++;
                idx.wei_scales1 = n++;
            }
            idx.count = n;
            return idx;
        }())
        , reserved_regs_(vmm_indices_.count) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;

    enum { k_blk_step = 2, n_blk_step = 16 };
    const int typesize_, tr_typesize_, wei_scales_typesize_;
    const bool is_src_int4_;
    const int src_elems_per_byte_;
    const dim_t src_stride_, tr_src_stride_;
    const bool req_cvtf82f16_;
    const bool req_zp_b_shift_;
    const bool req_apply_wei_scales_;
    const bool is_wei_grouped_over_k_;

    struct vmm_idx_t {
        int permd = -1;
        int zp_b_val0 = -1;
        int zp_b_val1 = -1;
        int tmp = -1;
        int wei_scales0 = -1;
        int wei_scales1 = -1;
        int count = 0;
    };

    const vmm_idx_t vmm_indices_;
    const int reserved_regs_;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_wei_scales = r10;
    reg64_t reg_tmp = r11;
    reg32_t regw_tmp = r11d;

    reg64_t reg_src_back = r12;
    reg64_t reg_tr_src_back = r13;

    reg64_t reg_wei_zp = r14;
    reg64_t reg_k_start = r15;

    Vmm vmm_permd = Vmm(vmm_indices_.permd);
    Vmm vmm_zp_b_val0 = Vmm(vmm_indices_.zp_b_val0);
    Vmm vmm_zp_b_val1 = Vmm(vmm_indices_.zp_b_val1);
    Vmm vmm_tmp = Vmm(vmm_indices_.tmp);
    Vmm vmm_wei_scales0 = Vmm(vmm_indices_.wei_scales0);
    Vmm vmm_wei_scales1 = Vmm(vmm_indices_.wei_scales1);

    Vmm get_vmm(const int blk, const int idx) {
        // fp8->f16 path needs only 1 VMM per block
        // so that the full register pool is available for loop unrolling.
        const int max_isa_regs = isa_num_vregs(conf_->isa);
        const int blk_sz = req_cvtf82f16_ ? 1 : k_blk_step;
        const int max_unroll = (max_isa_regs - reserved_regs_) / blk_sz;
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        const auto reg_idx
                = max_unroll * ((idx + 1) % blk_sz) + blk + reserved_regs_;
        assert(reg_idx >= reserved_regs_ && reg_idx < max_isa_regs);
        return Vmm(reg_idx);
    }

    void init_masks();
    void get_wei_scales(
            const int n, const bool is_n_tail, const bool is_k_tail);
    void get_zero_points(const int n, const bool is_tail, const bool is_k_tail);
    void copy_block(const int nrows, const int ncolumns, bool zeropad);

    /** Adjust strides for grouped over k weights
    * k_blk_step is const 2. This case handles
    * nrows = 1
    * Move tr_src pointer to the beginning of the block 2x32
    * if the k_start % 2 = 1 is odd.
    **/
    void maybe_update_strides(int nrows) {
        if (is_wei_grouped_over_k_ && nrows < k_blk_step) {
            Label even_k;
            test(reg_k_start, 1);
            jz(even_k, T_NEAR);
            // Shift back to start of the vnni block
            sub(reg_src, typesize_ / src_elems_per_byte_);
            L(even_k);
        }
    }

    void save_half_block(const int blk_idx, const Xbyak::Address &store_addr) {
        const auto src0 = get_vmm(blk_idx, 0);
        const auto zmm0 = zmm(src0.getIdx());
        //if k % 2 == 1 then save only odd indices
        // otherwise: only even using masks
        Label even_k, end_permute;
        test(reg_k_start, 1);
        jz(even_k, T_NEAR);
        // Odd indices case
        vmovdqu16(store_addr | kAAAA, zmm0);
        jmp(end_permute);
        L(even_k);
        // Clean the whole block before storing
        uni_vpxor(vmm_tmp, vmm_tmp, vmm_tmp);
        vmovdqu16(store_addr, vmm_tmp);
        // Store only even indices
        vmovdqu16(store_addr | k5555 | T_z, zmm0);
        L(end_permute);
    }

    void generate() override;
};

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::init_masks() {
    alignas(64) static constexpr const uint32_t bf16_vnni_permute[16]
            = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

    if (is_superset(conf_->isa, avx512_core)) {
        kxnorw(kFFFF, kFFFF, kFFFF); // 1111 1111 1111 1111

        mov(reg_tmp, reinterpret_cast<size_t>(bf16_vnni_permute));
        vmovdqa32(vmm_permd, ptr[reg_tmp]);

        // 64-bit mask is also used when is_wei_[zp\scales]_per_k
        mov(reg_tmp, 0xAAAAAAAAAAAAAAAA);
        kmovq(kAAAA, reg_tmp);
        mov(reg_tmp, 0x5555555555555555);
        kmovq(k5555, reg_tmp);
    }
}

/** Loads scales into 2 registers and permutes it.
*   Since groups over K-dimension are handled outside the kernel
*   loading is performed for the same address for both registers.
*/
template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::get_wei_scales(
        const int n, const bool is_n_tail, const bool is_k_tail) {

    if (!req_apply_wei_scales_ || !conf_->is_wei_scale_per_n) return;

    const auto zmm_wei_scales1 = maybe_mask(vmm_wei_scales1, is_n_tail);
    const auto zmm_tmp = maybe_mask(vmm_tmp, is_n_tail);
    const auto base_offset
            = [&](int n_idx) { return n_idx * wei_scales_typesize_; };

    auto wei_scales_addr
            = maybe_EVEX_compress_addr(reg_wei_scales, base_offset(n));

    load_scale_value(zmm_tmp, wei_scales_addr, conf_->wei_scales_dt, is_n_tail);

    uni_vmovups(vmm_wei_scales1, vmm_tmp);

    vinsertf64x4(vmm_wei_scales0, vmm_tmp, Ymm(vmm_wei_scales1.getIdx()), 1);
    vextractf64x4(Ymm(vmm_tmp.getIdx()), vmm_tmp, 1);
    vinsertf64x4(vmm_wei_scales1, zmm_wei_scales1, Ymm(vmm_tmp.getIdx()), 0);
    vpermd(vmm_wei_scales0, vmm_permd, vmm_wei_scales0);
    vpermd(vmm_wei_scales1, vmm_permd, vmm_wei_scales1);
}

/** Loads zero points into 2 registers and permute it.
*   Since groups over K-dimension are handled outside the kernel
*   loading is performed for the same address for both registers.
*/
template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::get_zero_points(
        const int n, const bool is_n_tail, const bool is_k_tail) {
    if (!conf_->is_wei_zp_per_n) return;

    const auto zp_dt = conf_->wei_zp_dt;

    const auto base_offset = [&](int n_idx) {
        const auto zp_dt_sz = types::data_type_size(zp_dt);
        const auto elems_per_byte
                = one_of(zp_dt, data_type::s4, data_type::u4) ? 2 : 1;
        return static_cast<size_t>(n_idx) * zp_dt_sz / elems_per_byte;
    };

    const auto addr = maybe_EVEX_compress_addr(reg_wei_zp, base_offset(n));
    load_value(vmm_tmp, addr, vmm_permd, zp_dt, is_n_tail);

    uni_vmovups(vmm_zp_b_val1, vmm_tmp);

    const auto zmm_zp_b_val1 = maybe_mask(vmm_zp_b_val1, is_n_tail);
    vinserti64x4(vmm_zp_b_val0, vmm_tmp, Ymm(vmm_zp_b_val1.getIdx()), 1);
    vextracti64x4(Ymm(vmm_tmp.getIdx()), vmm_tmp, 1);
    vinserti64x4(vmm_zp_b_val1, zmm_zp_b_val1, Ymm(vmm_tmp.getIdx()), 0);
    vpermd(vmm_zp_b_val0, vmm_permd, vmm_zp_b_val0);
    vpermd(vmm_zp_b_val1, vmm_permd, vmm_zp_b_val1);
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::copy_block(
        const int nrows, int ncolumns, bool zeropad) {
    const int columns_tail = ncolumns % n_blk_step;
    if (columns_tail > 0 && columns_tail < n_blk_step) {
        const auto regw_tmp = reg_tmp.cvt32();
        const auto tail_mask = (1 << columns_tail) - 1;
        mov(regw_tmp, tail_mask);
        kmovw(kTail, regw_tmp);

        const bool is_wei_zp_int4 = conf_->is_wei_zp_per_n
                && one_of(conf_->wei_zp_dt, data_type::s4, data_type::u4);
        if (is_src_int4_ || is_wei_zp_int4) {
            const auto tail_mask_4bit = (1 << (columns_tail / 2)) - 1;
            mov(regw_tmp, tail_mask_4bit);
            kmovw(kTail_4bit, regw_tmp);
        }
    }

    // fp8->f16: only 1 VMM per block needed(not 2-register packing)
    // and the full register pool is used
    const int blk_sz = req_cvtf82f16_ ? 1 : k_blk_step;
    const int max_regs_available = isa_num_vregs(conf_->isa) - reserved_regs_;
    const int max_unroll = max_regs_available / blk_sz;

    // Every load converts unroll * k_blk_step * n_blk_step
    auto load = [this, nrows, ncolumns](int blk, int k, int n) {
        const int k_blk = k / k_blk_step;
        const auto src_vmm0 = get_vmm(blk, 0);
        const dim_t offset = k_blk * src_stride_
                + (n * k_blk_step * typesize_) / src_elems_per_byte_;
        auto load_addr0 = maybe_EVEX_compress_addr(reg_src, offset);

        const bool is_n_tail = ncolumns - n < n_blk_step;
        const bool is_k_tail = nrows - k < k_blk_step;

        if (req_cvtf82f16_) {
            f8_to_f16_upconvert(
                    conf_->orig_wei_dt, src_vmm0, load_addr0, false, true);
        } else {
            const auto src_vmm1 = get_vmm(blk, 1);
            const dim_t stride = (n_blk_step * typesize_) / src_elems_per_byte_;
            auto load_addr1
                    = maybe_EVEX_compress_addr(reg_src, offset + stride);
            load_value(src_vmm0, load_addr0, vmm_permd, conf_->orig_wei_dt);
            load_value(src_vmm1, load_addr1, vmm_permd, conf_->orig_wei_dt);

            get_wei_scales(n, is_n_tail, is_k_tail);
            get_zero_points(n, is_n_tail, is_k_tail);
            decompress_and_downcvt_2reg(src_vmm0, src_vmm1, vmm_zp_b_val0,
                    vmm_zp_b_val1, vmm_wei_scales0, vmm_wei_scales1,
                    conf_->orig_wei_dt, conf_->wei_dt);
        }
    };

    maybe_update_strides(nrows);

    int iter = 0;
    for_(int k = 0; k < nrows; k += k_blk_step)
    for (int n = 0; n < ncolumns; n += n_blk_step) {
        const int k_blk = k / k_blk_step;
        const dim_t tr_src_off
                = k_blk * tr_src_stride_ + n * k_blk_step * tr_typesize_;
        const auto store_addr
                = maybe_EVEX_compress_addr(reg_tr_src, tr_src_off);
        const int blk_idx = iter % max_unroll;

        const auto store_vmm = get_vmm(blk_idx, 0);
        if (zeropad)
            uni_vpxor(store_vmm, store_vmm, store_vmm);
        else
            load(blk_idx, k, n);

        // Special case for goruped zp/scales when nrows == 1
        if (is_wei_grouped_over_k_ && nrows == 1) {
            save_half_block(blk_idx, store_addr);
            iter++;
            continue;
        }

        uni_vmovups(store_addr, store_vmm);

        iter++;
    }
}

template <typename Vmm>
void jit_brgemm_matmul_copy_b_cvt_bf16_t<Vmm>::generate() {
    assert(tr_typesize_ == sizeof(bfloat16_t));
    preamble();

    init_masks();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    mov(reg_wei_scales, ptr[param1 + GET_OFF(wei_scales_ptr)]);
    mov(reg_wei_zp, ptr[param1 + GET_OFF(zp_b_value_ptr)]);
    mov(reg_k_start, ptr[param1 + GET_OFF(current_K_start)]);

    load_common_zp_value(vmm_zp_b_val0, reg_wei_zp);
    load_common_zp_value(vmm_zp_b_val1, reg_wei_zp);
    load_common_scale_value(vmm_wei_scales0, reg_wei_scales);
    load_common_scale_value(vmm_wei_scales1, reg_wei_scales);

    auto compute_K_loop_body
            = [&](const reg64_t &reg_K, int ncolumns, bool zeropad) {
        // Compute special K-loop for per-k attributes
        // Only when k_group_size < k_blk_step
        // Otherwise default K-loop is used
        if (is_wei_grouped_over_k_) {
            const int k_group_size = conf_->is_wei_zp_per_k
                    ? static_cast<int>(conf_->wei_zp_k_gsize)
                    : static_cast<int>(conf_->wei_scales_k_gsize);
            if (k_group_size == 1) {
                if (zeropad) return;
                copy_block(k_group_size, ncolumns, /*zeropad= */ false);
                return;
            }
        }
        const int k_unroll = 8;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_block(k_unroll * k_blk_step, ncolumns, zeropad);
        add(reg_src, k_unroll * src_stride_);
        add(reg_tr_src, k_unroll * tr_src_stride_);

        sub(reg_K, k_unroll * k_blk_step);
        cmp(reg_K, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_block(k_blk_step, ncolumns, zeropad);
        add(reg_src, src_stride_);
        add(reg_tr_src, tr_src_stride_);

        sub(reg_K, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        const int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K, 0);
            jle(K_loop_done, T_NEAR);
            copy_block(k_blk_tail, ncolumns, zeropad);
            add(reg_tr_src, tr_src_stride_);
            sub(reg_K, k_blk_tail);
            L(K_loop_done);
        }
    };

    auto compute_K_loop = [&](const int ncolumns) {
        mov(reg_src_back, reg_src);
        mov(reg_tr_src_back, reg_tr_src);

        mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
        compute_K_loop_body(reg_K_iters, ncolumns, false);
        mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_pad)]);
        compute_K_loop_body(reg_K_iters, ncolumns, true);

        mov(reg_src, reg_src_back);
        mov(reg_tr_src, reg_tr_src_back);
    };

    Label done;
    cmp(reg_N_blk, 0);
    jle(done, T_NEAR);

    if (conf_->LDB2 != 0) {
        Label main_N_loop, main_N_loop_tail;
        int tail = static_cast<int>(conf_->N % conf_->LDB);

        if (tail != 0) {
            cmp(reg_N_blk, conf_->LDB);
            jl(main_N_loop_tail, T_NEAR);
        }

        L(main_N_loop);
        compute_K_loop(static_cast<int>(conf_->LDB));
        add(reg_src, conf_->LDB2 * typesize_);
        add(reg_tr_src, conf_->LDB2 * tr_typesize_);

        sub(reg_N_blk, conf_->LDB);
        cmp(reg_N_blk, conf_->LDB);
        jge(main_N_loop, T_NEAR);

        if (tail != 0) {
            L(main_N_loop_tail);
            cmp(reg_N_blk, 0);
            jle(done, T_NEAR);
            compute_K_loop(tail);
        }

    } else {
        if (conf_->N_tail > 0) {
            Label main_N_blk;
            cmp(reg_N_blk, conf_->N_blk);
            je(main_N_blk, T_NEAR);
            compute_K_loop(static_cast<int>(conf_->N_tail));
            jmp(done, T_NEAR);

            L(main_N_blk);
        }

        compute_K_loop(static_cast<int>(conf_->N_blk));
    }
    L(done);
    postamble();
}

template struct jit_brgemm_matmul_copy_b_cvt_bf16_t<Zmm>;

struct fp8_to_xf16_cvt_helper_t {
    // Reserved by fp8 conversion routines
    static constexpr int cvt_xmm0_idx = 0;
    static constexpr int cvt_xmm1_idx = 1;
    static constexpr int cvt_xmm2_idx = 2;
    static constexpr int cvt_xmm3_idx = 3;
    static constexpr int cvt_xmm4_idx = 4;
    static constexpr int cvt_opmask_idx = 1;
    static Xbyak::Reg64 cvt_scratch_gpr() { return Xbyak::util::r8; }

    fp8_to_xf16_cvt_helper_t(
            jit_generator_t *host, const brgemm_matmul_conf_t *conf)
        : conf_(conf) {
        if (conf_->orig_wei_dt == data_type::f8_e5m2) {
            f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(host,
                    Xmm(cvt_xmm0_idx), Xmm(cvt_xmm1_idx), Xmm(cvt_xmm2_idx),
                    Xbyak::Opmask(cvt_opmask_idx), cvt_scratch_gpr());
        } else {
            f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(host,
                    Xmm(cvt_xmm0_idx), Xmm(cvt_xmm1_idx), Xmm(cvt_xmm2_idx),
                    Xmm(cvt_xmm3_idx), Xmm(cvt_xmm4_idx), cvt_scratch_gpr());
        }
    }

    void prepare_tables() {
        if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
        if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
    }

    void convert_vnni(const Xbyak::Zmm &dst0, const Xbyak::Zmm &dst1,
            const Xbyak::Operand &src_op) {
        if (conf_->wei_dt == data_type::bf16) {
            if (conf_->orig_wei_dt == data_type::f8_e5m2) {
                f8_e5m2_cvt_->vcvt_f8_to_bf16_vnni(dst0, dst1, src_op);
            } else {
                f8_e4m3_cvt_->vcvt_f8_to_bf16_vnni(dst0, dst1, src_op);
            }
        } else {
            if (conf_->orig_wei_dt == data_type::f8_e5m2) {
                f8_e5m2_cvt_->vcvt_f8_to_f16_vnni(dst0, dst1, src_op);
            } else {
                f8_e4m3_cvt_->vcvt_f8_to_f16_vnni(dst0, dst1, src_op);
            }
        }
    }

private:
    const brgemm_matmul_conf_t *conf_;
    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;
};

struct jit_brgemm_matmul_copy_cvt_fp8_to_xf16_t
    : public jit_brgemm_matmul_copy_b_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_cvt_fp8_to_xf16_t)

    jit_brgemm_matmul_copy_cvt_fp8_to_xf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , jit_generator_t(jit_name())
        , typesize_(static_cast<int>(conf->b_dt_sz))
        , tr_typesize_(static_cast<int>(conf->tr_b_dt_sz))
        , src_stride_(conf->LDB * k_blk_step * typesize_)
        , tr_src_stride_(conf->LDB * tr_k_blk_step * tr_typesize_)
        , cvt_helper_(this, conf) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak::Reg64;

    enum {
        n_blk_step = 16,
        // K-pack size in source (fp8) blocked layout
        k_blk_step = 4,
        // K-pack size in destination (xf16) blocked layout
        tr_k_blk_step = 2,
    };

    const int typesize_, tr_typesize_;
    // Bytes between consecutive fp8 packs and xf16 packs, respectively
    const dim_t src_stride_, tr_src_stride_;

    fp8_to_xf16_cvt_helper_t cvt_helper_;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_K_iters = r9;
    reg64_t reg_N_blk = r10;
    reg64_t reg_tmp = r11;
    reg64_t reg_src_back = r12;
    reg64_t reg_tr_src_back = r13;
    reg64_t reg_blk_src = r14;
    reg64_t reg_blk_dst = r15;

    Zmm zmm_dst0 = Zmm(5);
    Zmm zmm_dst1 = Zmm(6);
    Zmm zmm_src_pack = Zmm(7);

    void copy_block(const int nrows, const int ncolumns, bool zeropad) {

        const int simd_nrows_rounded = utils::rnd_up(nrows, k_blk_step);
        const int simd_ncols_rounded = utils::rnd_up(ncolumns, n_blk_step);

        for_(int k = 0; k < simd_nrows_rounded; k += k_blk_step)
        for (int n = 0; n < simd_ncols_rounded; n += n_blk_step) {
            // One loop step consumes one source K-pack (k_blk_step rows) and
            // converts it into two destination K-packs (tr_k_blk_step rows
            // each). The second destination pack is stored only when the
            // source pack actually contributes more than tr_k_blk_step rows,
            // so the write granularity equals the destination VNNI
            // granularity tr_k_blk_step rather than the source pack size.
            const int fp8_pack = k / k_blk_step;
            const int xf16_pack_0 = fp8_pack * (k_blk_step / tr_k_blk_step);
            const bool store_pack_1 = (nrows - k) > tr_k_blk_step;
            const dim_t src_off
                    = fp8_pack * src_stride_ + n * k_blk_step * typesize_;
            const dim_t tr_src_off_0 = xf16_pack_0 * tr_src_stride_
                    + n * tr_k_blk_step * tr_typesize_;
            const dim_t tr_src_off_1 = tr_src_off_0 + tr_src_stride_;

            if (zeropad) {
                uni_vpxor(zmm_dst0, zmm_dst0, zmm_dst0);
                if (store_pack_1) uni_vpxor(zmm_dst1, zmm_dst1, zmm_dst1);
            } else {
                const auto src_addr = EVEX_compress_addr(reg_src, src_off);
                vmovdqu8(zmm_src_pack, src_addr);

                cvt_helper_.convert_vnni(zmm_dst0, zmm_dst1, zmm_src_pack);
            }

            const auto store_addr_0
                    = EVEX_compress_addr(reg_tr_src, tr_src_off_0);
            uni_vmovups(store_addr_0, zmm_dst0);
            if (store_pack_1) {
                const auto store_addr_1
                        = EVEX_compress_addr(reg_tr_src, tr_src_off_1);
                uni_vmovups(store_addr_1, zmm_dst1);
            }
        }
    }

    void generate() override {
        preamble();

        mov(reg_src, ptr[param1 + GET_OFF(src)]);
        mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
        mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);

        auto compute_K_loop_body
                = [&](const reg64_t &reg_K, int ncolumns, bool zeropad) {
            constexpr int k_step = k_blk_step;
            constexpr int k_unroll = 8;
            constexpr int k_unroll_rows = k_unroll * k_step;

            Label K_loop_single, K_loop_tail_or_done;
            // In the zeropad pass reg_K == current_K_pad, which is always
            // < wei_k_blk (it pads a partial K-block up to wei_k_blk). Since
            // wei_k_blk <= k_unroll_rows here (wei_dt is bf16/f16 -> wei_k_blk == 32,
            // and k_unroll_rows == 32), it follows that reg_K < k_unroll_rows.
            // The unrolled loop below only runs when reg_K >= k_unroll_rows, so
            // it never runs in the zeropad pass. Assert the invariant
            // and skip emitting the unrolled loop there to save a dead
            // cmp/branch.
            assert(conf_->wei_k_blk <= k_unroll_rows);
            if (!zeropad) {
                Label K_loop_unrolled;
                cmp(reg_K, k_unroll_rows);
                jl(K_loop_single, T_NEAR);

                L(K_loop_unrolled);
                copy_block(k_unroll_rows, ncolumns, zeropad);
                add(reg_src, (k_unroll_rows / k_blk_step) * src_stride_);
                add(reg_tr_src,
                        (k_unroll_rows / tr_k_blk_step) * tr_src_stride_);

                sub(reg_K, k_unroll_rows);
                cmp(reg_K, k_unroll_rows);
                jge(K_loop_unrolled, T_NEAR);
            }

            L(K_loop_single);
            cmp(reg_K, k_step);
            jl(K_loop_tail_or_done, T_NEAR);

            copy_block(k_step, ncolumns, zeropad);
            add(reg_src, (k_step / k_blk_step) * src_stride_);
            add(reg_tr_src, (k_step / tr_k_blk_step) * tr_src_stride_);

            sub(reg_K, k_step);
            jmp(K_loop_single, T_NEAR);

            L(K_loop_tail_or_done);
            Label K_loop_done;
            cmp(reg_K, 0);
            jle(K_loop_done, T_NEAR);
            // Tail of 1..k_blk_step-1 K-rows. The source is read as whole
            // zero-padded fp8 packs, so the tail count only affects the
            // destination: a tail wider than tr_k_blk_step writes two xf16
            // packs, otherwise one (tails of 1 and 2 rows are identical).
            // reg_src is not advanced: this is the last source read of the
            // pass and reg_src is restored from reg_src_back afterwards.
            Label tail_one_pack;
            cmp(reg_K, tr_k_blk_step);
            jle(tail_one_pack, T_NEAR);
            copy_block(k_blk_step, ncolumns, zeropad);
            add(reg_tr_src, (k_blk_step / tr_k_blk_step) * tr_src_stride_);
            jmp(K_loop_done, T_NEAR);

            L(tail_one_pack);
            copy_block(tr_k_blk_step, ncolumns, zeropad);
            add(reg_tr_src, tr_src_stride_);

            L(K_loop_done);
        };

        auto compute_K_loop = [&](const int ncolumns) {
            mov(reg_src_back, reg_src);
            mov(reg_tr_src_back, reg_tr_src);

            mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
            compute_K_loop_body(reg_K_iters, ncolumns, false);

            // The zero-padding pass fills the K-tail block up to wei_k_blk so brgemm
            // can safely read the full block (extendable_k). It is only needed in case
            // of a non-zero current_K_pad; otherwise it can be skipped.
            // Even when emitted, a single runtime check skips it for the most common
            // case (current_K_pad == 0), saving the block-copy work.
            if (conf_->extendable_k || conf_->use_fused_copy_a) {
                Label skip_zeropad;
                mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_pad)]);
                cmp(reg_K_iters, 0);
                jle(skip_zeropad, T_NEAR);
                compute_K_loop_body(reg_K_iters, ncolumns, true);
                L(skip_zeropad);
            }

            mov(reg_src, reg_src_back);
            mov(reg_tr_src, reg_tr_src_back);
        };

        Label done;
        cmp(reg_N_blk, 0);
        jle(done, T_NEAR);

        // LDB2 is always non-zero for is_xf16_fp8: these kernels are only created
        // from the matmul path (init_brgemm_matmul_conf), which always sets LDB2.
        assert(conf_->LDB2 != 0);

        Label main_N_loop, main_N_loop_tail;
        int tail = static_cast<int>(conf_->N % conf_->LDB);

        if (tail != 0) {
            cmp(reg_N_blk, conf_->LDB);
            jl(main_N_loop_tail, T_NEAR);
        }

        L(main_N_loop);
        compute_K_loop(static_cast<int>(conf_->LDB));
        add(reg_src, conf_->B_strides[0]);
        add(reg_tr_src, conf_->LDB2 * tr_typesize_);

        sub(reg_N_blk, conf_->LDB);
        cmp(reg_N_blk, conf_->LDB);
        jge(main_N_loop, T_NEAR);

        if (tail != 0) {
            L(main_N_loop_tail);
            cmp(reg_N_blk, 0);
            jle(done, T_NEAR);
            compute_K_loop(tail);
        }

        L(done);
        postamble();
        cvt_helper_.prepare_tables();
    }
};

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    const bool is_bf16 = !conf->is_xf16_fp8
            && everyone_is(data_type::bf16, conf->src_dt, conf->wei_dt);
    const bool is_f32 = everyone_is(data_type::f32, conf->src_dt, conf->wei_dt);
    // Note: f16 support through avx512_core_fp16 sets src_dt and wei_dt as f32
    // to imply upconverting. So, the assumption is `is_f16` below evaluates to
    // `false` on avx512_core_fp16.
    const bool is_f16 = !conf->is_xf16_fp8
            && everyone_is(data_type::f16, conf->src_dt, conf->wei_dt);
    if (conf->transposed_B) {
        if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<Zmm>(conf)));
        else {
            assert(is_superset(conf->isa, avx2));
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_b_transposed_t<Ymm>(conf)));
        }
    } else {
        if ((conf->is_bf16_with_int_wei
                    || (conf->is_f16_with_int_wei
                            && conf->isa != avx512_core_fp16)
                    || (conf->is_f8 && conf->isa == avx10_2))
                && conf->blocked_B) {
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_cvt_bf16_t<Zmm>(conf)));
            else {
                assert(!"Unsupported isa for bf16_with_int_wei");
                return status::unimplemented;
            }
        } else if (is_bf16 || is_f16 || conf->is_bf32
                || (conf->is_f16_with_int_wei
                        && conf->isa != avx512_core_fp16)) {
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_bf16_t<Zmm>(conf)));
            else
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_bf16_t<Ymm>(conf)));
        } else if (is_f32
                || (conf->isa == avx512_core_fp16
                        && conf->orig_wei_dt == data_type::f16)) {
            // See the note above why `orig_wei_dt` is used.
            if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_f32_t<Zmm>(conf)));
            else
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_b_f32_t<Ymm>(conf)));
        } else if (conf->is_xf16_fp8) {
            assert((is_superset(conf->isa, avx10_2)
                           || (conf->wei_dt == data_type::f16
                                           ? is_superset(conf->isa,
                                                     avx10_1_512_amx_fp16)
                                           : is_superset(conf->isa,
                                                     avx512_core_amx)))
                    && "Unsupported isa for xf16_fp8");
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_cvt_fp8_to_xf16_t(conf)));
        } else {
            // No AMX here: this path uses vpermi2b. AVX10.2 ACE gets VBMI explicitly.
            const bool has_vbmi_copy_b = mayiuse(avx512_core_amx)
                    || is_superset(conf->isa, avx10_2_ace);
            if (has_vbmi_copy_b)
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_amx_brgemm_matmul_copy_b_int8_t(conf)));
            else if (is_superset(conf->isa, avx512_core))
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_avx512_core_brgemm_matmul_copy_b_int8_t(conf)));
            else {
                // TODO: jit_avx2_vnni_brgemm_matmul_copy_b_int8_t can handle
                // avx2 if no compensation is required. Consider enabling it
                // for avx2 and renaming the kernel (drop "vnni" part).
                const bool is_comp_required = conf->s8s8_compensation_required
                        || conf->has_zero_point_a;
                MAYBE_UNUSED(is_comp_required);
                assert(one_of(conf->isa, avx2_vnni, avx2_vnni_2, avx2)
                        && IMPLICATION(conf->isa == avx2, !is_comp_required));

                CHECK(safe_ptr_assign(copy_ker,
                        new jit_avx2_vnni_brgemm_matmul_copy_b_int8_t(conf)));
            }
        }
    }

    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    if (conf->transposed_A) {
        if (utils::one_of(conf->src_dt, data_type::s8, data_type::u8))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_int8_impl_t(conf)));
        else if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_impl_t<Zmm>(conf)));
        else
            CHECK(safe_ptr_assign(copy_ker,
                    new jit_brgemm_matmul_copy_a_transposed_impl_t<Ymm>(conf)));
    } else {
        if (is_superset(conf->isa, avx512_core))
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_a_impl_t<Zmm>(conf)));
        else {
            if (is_superset(conf->isa, avx2)) {
                CHECK(safe_ptr_assign(copy_ker,
                        new jit_brgemm_matmul_copy_a_impl_t<Ymm>(conf)));
            } else {
                assert(!"Unsupported isa for jit_brgemm_matmul_copy_a_impl_t");
                return status::unimplemented;
            }
        }
    }

    return copy_ker->create_kernel();
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
