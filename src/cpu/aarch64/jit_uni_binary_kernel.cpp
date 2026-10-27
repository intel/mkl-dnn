/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2022-2023 FUJITSU LIMITED
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

#include "cpu/aarch64/jit_uni_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#define PARAM_OFF(x) ((int32_t)offsetof(jit_uni_binary_args_t, x))

static bcast_set_t get_supported_postops_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}

binary_kernel_t::binary_kernel_t(const size_t vlen, const binary_pd_t *pd,
        const jit_binary_conf_t conf, bool tail_kernel)
    : vlen_(vlen)
    , simd_w_(vlen / sizeof(float))
    , pd_(pd)
    , conf_(conf)
    , is_tail_kernel_(tail_kernel)
    , is_src1_outer_dims_tail_(
              conf_.is_src_different_layouts && conf_.outer_dims % simd_w_)
    , tail_size_(get_tail_size())
    , padding_tail_size_(
              pd->src_md(0)->padded_dims[1] - pd->src_md(0)->dims[1]) {}

size_t binary_kernel_t::get_tail_size() const {
    memory_desc_wrapper src0_d(pd_->src_md(0));
    const auto &dims = src0_d.dims();
    const auto &ndims = src0_d.ndims();

    dim_t nelems = 0;

    if (ndims == 1)
        nelems = dims[0];
    else if (is_src1_outer_dims_tail_)
        nelems = conf_.outer_dims;
    else if (!conf_.is_i8 && conf_.op_type == op_t::c_blocked
            && (is_tail_kernel_ || conf_.bcast_type == bcast_t::per_w))
        nelems = dims[1];
    else if (conf_.bcast_type == bcast_t::none
            && !conf_.postops_per_oc_broadcast_exists)
        nelems = src0_d.nelems(true);
    else if (conf_.bcast_type == bcast_t::per_batch
            && !conf_.postops_per_oc_broadcast_exists)
        nelems = src0_d.nelems(true) / dims[0];
    else {
        if (conf_.op_type == op_t::n_spatial_c)
            nelems = dims[1];
        else if (conf_.op_type == op_t::n_c_spatial && ndims >= 3)
            nelems = conf_.bcast_type == bcast_t::per_w
                    ? utils::array_product(
                              dims + (ndims - conf_.not_bcasted_sp_dims),
                              conf_.not_bcasted_sp_dims)
                    : utils::array_product(dims + 2, ndims - 2);
    }
    // it's float due to for bfloat16 we still load 16 elements, not 32.
    return nelems % simd_w_;
}

template <cpu_isa_t isa>
jit_uni_binary_kernel_t<isa>::jit_uni_binary_kernel_t(
        const binary_pd_t *pd, const jit_binary_conf_t conf, bool tail_kernel)
    : binary_kernel_t(cpu_isa_traits<isa>::vlen, pd, conf, tail_kernel)
    , offt_src0_(vlen_)
    , offt_src1_(conf_.use_stride_src1 ? offt_src0_ : 0)
    , io_sat_conf_(vreg_saturation_lbound_, vreg_saturation_ubound_)
    , io_tail_conf_(tail_size_, tail_opmask_) {

    safe_ptr_assign(io_,
            new io::jit_io_helper_t<to_vla_sve(isa)>(
                    this, io_sat_conf_, io_tail_conf_, reg_tmp_));
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::init_post_ops_injector(
        const XReg &eltwise_table, const PReg &eltwise_mask) {
    const memory_desc_wrapper dst_d(pd_->dst_md(0));
    const auto &po = pd_->attr()->post_ops_;

    const binary_injector::rhs_arg_static_params_t rhs_arg_bsp {
            PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst_orig), dst_d,
            tail_size_, tail_opmask_, false /*use_exact_tail_scalar_bcast*/};
    const binary_injector::static_params_t bsp(this->param1,
            get_supported_postops_bcast_strategies(), rhs_arg_bsp);

    postops_injector_ = utils::make_unique<
            injector::jit_uni_postops_injector_t<to_vla_sve(inject_isa)>>(this,
            reg_pool(), po, bsp, eltwise_table, eltwise_mask, true /*is_fwd*/,
            false /*use_dst*/, data_type::f32);
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::apply_postops(int unroll, bool tail) {
    const auto sum_injector
            = [&](const injector_utils::vmm_index_set_t &vmm_idxs) {
        for (int i = 0; i < unroll; i++) {
            const int offt = simd_w_ * i;
            const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
            if (vmm_idxs.count(vreg_tmp_src0.getIdx()) == 0) continue;
            const Vmm vreg_tmp = conf_.is_src_different_layouts
                    ? vmm_gathered_src_
                    : Vmm(unroll + i + vmm_start_idx_);

            const auto &addr = prepare_xaddr_reg_with_dst_ptr(
                    offt * types::data_type_size(conf_.dst_type));
            io_->load(conf_.dst_type, vreg_tmp, addr, tail);
            float_point_fused_multiply_add(
                    vreg_tmp_src0, vreg_tmp, vreg_sum_scale_);
        }
    };

    if (conf_.do_sum)
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);

    if (conf_.with_binary) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        const XReg &reg_offt_dst = conf_.is_i8 ? reg_offt_dst_ : reg_offt_src0_;

        mov(reg_tmp1_, reg_dst_);
        add(reg_tmp1_, reg_tmp1_, reg_offt_dst);

        for (int vmm_idx = 1; vmm_idx < unroll + vmm_start_idx_; vmm_idx++) {
            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_tmp1_);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(vmm_idx,
                    (vmm_idx - vmm_start_idx_) * simd_w_
                            * types::data_type_size(conf_.dst_type));
            if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        }
        postops_injector_->compute_vector_range(
                1, unroll + vmm_start_idx_, rhs_arg_params);
    } else
        postops_injector_->compute_vector_range(1, unroll + vmm_start_idx_);
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::load_kernel_params() {
    mov(W_TMP_0, float2int(conf_.sum_scale));
    dup(vreg_sum_scale_.s, W_TMP_0);

    assert(sizeof(jit_uni_binary_args_t) <= 255);

    if (is_src1_outer_dims_tail_)
        ldr(reg_outer_dims_range_, ptr(reg_param_, PARAM_OFF(spat_offt_count)));
    else
        ldr(reg_reverse_spat_offt_,
                ptr(reg_param_, PARAM_OFF(spat_offt_count)));

    ldr(reg_src0_, ptr(reg_param_, PARAM_OFF(src0)));
    ldr(reg_src1_, ptr(reg_param_, PARAM_OFF(src1)));
    ldr(reg_dst_, ptr(reg_param_, PARAM_OFF(dst)));
    mov(reg_offt_dst_, reg_dst_);
    if (conf_.is_src_different_layouts) {
        ldr(X_DEFAULT_ADDR, Xbyak_aarch64::ptr(reg_param_, PARAM_OFF(indices)));
        this->contiguous_load_unsigned_words(vmm_indices_, X_DEFAULT_ADDR);
        ldr(reg_src1_stride_range_,
                ptr(reg_param_, PARAM_OFF(src1_stride_range)));
        mov(reg_reverse_src1_stride_range_, reg_src1_stride_range_);
    }
    if (conf_.do_scale_src0)
        ldr(reg_scales_src0_, ptr(reg_param_, PARAM_OFF(scales_src0)));
    if (conf_.do_scale_src1)
        ldr(reg_scales_src1_, ptr(reg_param_, PARAM_OFF(scales_src1)));
}

template <cpu_isa_t isa>
XReg jit_uni_binary_kernel_t<isa>::prepare_xaddr_reg_with_src0_ptr(
        size_t offt) {
    add(X_DEFAULT_ADDR, reg_src0_, reg_offt_src0_);
    if (offt) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
    return X_DEFAULT_ADDR;
}

template <cpu_isa_t isa>
XReg jit_uni_binary_kernel_t<isa>::prepare_xaddr_reg_with_src1_ptr(
        size_t offt) {
    add(X_DEFAULT_ADDR, reg_src1_, reg_offt_src1_);
    if (offt) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
    return X_DEFAULT_ADDR;
}

template <cpu_isa_t isa>
XReg jit_uni_binary_kernel_t<isa>::prepare_xaddr_reg_with_dst_ptr(size_t offt) {
    const XReg &reg_offt_dst = conf_.is_i8 ? reg_offt_dst_ : reg_offt_src0_;
    add(X_DEFAULT_ADDR, reg_dst_, reg_offt_dst);
    if (offt) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, offt, X_TMP_0);
    return X_DEFAULT_ADDR;
}

template <>
template <>
void jit_uni_binary_kernel_t<asimd>::compute_cmp_alg(
        const VReg &dst, const VReg &src, const VReg &src2, alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        case binary_ge: fcmge(dst.s, src.s, src2.s); break;
        case binary_gt: fcmgt(dst.s, src.s, src2.s); break;
        case binary_le:
            // Express a <= b as b >= a.
            fcmge(dst.s, src2.s, src.s);
            break;
        case binary_lt:
            // Express a < b as b > a.
            fcmgt(dst.s, src2.s, src.s);
            break;
        case binary_eq: fcmeq(dst.s, src.s, src2.s); break;
        case binary_ne:
            // use fcmeq to compare for equality and then "not" the mask to get !=
            fcmeq(dst.s, src.s, src2.s);
            not_(dst.b, dst.b);
            break;
        default: assert(!"unsupported compare mode"); break;
    }
}

template <cpu_isa_t isa>
template <typename T>
void jit_uni_binary_kernel_t<isa>::compute_cmp_alg(
        const T &dst, const Vmm &src, const Vmm &src2, alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        case binary_ge:
            fcmge(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        case binary_gt:
            fcmgt(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        case binary_le:
            fcmle(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        case binary_lt:
            fcmlt(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        case binary_eq:
            fcmeq(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        case binary_ne:
            fcmne(dst.s, P_ALL_ONE / Xbyak_aarch64::T_z, src.s, src2.s);
            break;
        default: assert(!"unsupported compare mode"); break;
    }
}

template <>
void jit_uni_binary_kernel_t<asimd>::perform_op(const VReg &v0, const VReg &v1,
        const VReg &s_src0, const VReg &s_src1) {
    using namespace alg_kind;
    const auto alg = pd_->desc()->alg_kind;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    if (conf_.do_scale_src0) uni_fmul(v0.s, v0.s, s_src0.s);
    if (conf_.do_scale_src1 && offt_src1_ != 0 && !conf_.broadcast_src1_value)
        uni_fmul(v1.s, v1.s, s_src1.s);

    if (alg == binary_add)
        uni_fadd(v0.s, v0.s, v1.s);
    else if (alg == binary_mul)
        uni_fmul(v0.s, v0.s, v1.s);
    else if (alg == binary_max)
        uni_fmax(v0.s, v0.s, v1.s);
    else if (alg == binary_min)
        uni_fmin(v0.s, v0.s, v1.s);
    else if (alg == binary_div)
        uni_fdiv(v0.s, v0.s, v1.s);
    else if (alg == binary_sub)
        uni_fsub(v0.s, v0.s, v1.s);
    else if (cmp_op) {
        compute_cmp_alg(v0, v0, v1, alg);
        // ASIMD compare instructions produce all-ones/all-zeros integer masks.
        // Convert them to the expected 1.0f / 0.0f compare result in place.
        and_(v0.b16, v0.b16, vreg_one_.b16);
    } else {
        assert(!"not supported operation!");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::perform_op(
        const Vmm &v0, const Vmm &v1, const Vmm &s_src0, const Vmm &s_src1) {
    using namespace alg_kind;
    const auto alg = pd_->desc()->alg_kind;
    const bool cmp_op = utils::one_of(alg, alg_kind::binary_ge,
            alg_kind::binary_gt, alg_kind::binary_le, alg_kind::binary_lt,
            alg_kind::binary_eq, alg_kind::binary_ne);
    if (conf_.do_scale_src0) uni_fmul(v0.s, v0.s, s_src0.s);
    if (conf_.do_scale_src1 && offt_src1_ != 0 && !conf_.broadcast_src1_value)
        uni_fmul(v1.s, v1.s, s_src1.s);

    if (alg == binary_add)
        uni_fadd(v0.s, v0.s, v1.s);
    else if (alg == binary_mul)
        uni_fmul(v0.s, v0.s, v1.s);
    else if (alg == binary_max)
        uni_fmax(v0.s, v0.s, v1.s);
    else if (alg == binary_min)
        uni_fmin(v0.s, v0.s, v1.s);
    else if (alg == binary_div)
        uni_fdiv(v0.s, v0.s, v1.s, ZRegS(DUMMY_IDX), P_ALL_ONE);
    else if (alg == binary_sub)
        uni_fsub(v0.s, v0.s, v1.s);
    else if (cmp_op) {
        if (is_superset(isa, sve_128)) {
            compute_cmp_alg(cmp_mask, v0, v1, alg);
            eor(v0.d, v0.d, v0.d);
            fmov(v0.s, cmp_mask / Xbyak_aarch64::T_m, 1.0);
        } else {
            assert(!"not supported isa!");
        }
    } else
        assert(!"not supported operation!");
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::prepare_isa_kernel() {
    fmov(vreg_one_.s, 1.0);
    io_->init_tail();
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::compute_bcast(bool tail) {
    if (conf_.broadcast_src1_value) {
        if (conf_.is_i8) uni_clear(xreg_bcast_src1_);
        const auto &addr = prepare_xaddr_reg_with_src1_ptr();
        io_->broadcast_load(conf_.src1_type, vreg_bcast_src1_, addr);
    } else if (!conf_.is_i8 && offt_src1_ == 0) {
        const auto &addr = prepare_xaddr_reg_with_src1_ptr();
        io_->load(conf_.src1_type, vreg_bcast_src1_, addr, tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::push(const Xbyak_aarch64::XReg &reg) {
    str(reg, pre_ptr(sp, -16));
}
template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::pop(const Xbyak_aarch64::XReg &reg) {
    ldr(reg, post_ptr(sp, 16));
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::load_src1(
        const Vmm &vreg_src1, const int offt, bool tail) {
    if (conf_.is_src_different_layouts) {
        // if different layouts, gather_load data with strides
        // after getting to stride range, offset is restored and
        // increased
        io_->gather_load(
                conf_.src1_type, vreg_src1, reg_src1_, vmm_indices_, tail);
        // gather_load is using register instead of operand to read address
        // use reg_src1_ directly, without offset stored in second
        // register
        add_imm(reg_src1_, reg_src1_,
                types::data_type_size(conf_.src1_type) * conf_.src1_stride
                        * simd_w_,
                X_TMP_0);
        sub_imm(reg_reverse_src1_stride_range_, reg_reverse_src1_stride_range_,
                types::data_type_size(conf_.src1_type) * conf_.src1_stride
                        * simd_w_,
                X_TMP_1);

        Label src1_stride_range_not_exceed, src1_C_tail_end;

        cmp(reg_reverse_src1_stride_range_, 0);
        b(GT, src1_stride_range_not_exceed);
        {
            pop(reg_src1_);
            add_imm(reg_src1_, reg_src1_,
                    types::data_type_size(conf_.src1_type), X_TMP_0);
            push(reg_src1_);
            mov(reg_reverse_src1_stride_range_, reg_src1_stride_range_);
        }
        L(src1_stride_range_not_exceed);
    } else {
        const auto addr = prepare_xaddr_reg_with_src1_ptr(
                offt * types::data_type_size(conf_.src1_type));
        io_->load(conf_.src1_type, vreg_src1, addr, tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::store(int unroll, bool tail) {
    for (int i = 0; i < unroll; i++) {
        const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
        const int offt = simd_w_ * i;
        const auto dt_size = types::data_type_size(conf_.dst_type);

        if (is_tail_kernel_ && padding_tail_size_) {
            // apply zero-padding
            auto off_base = 0;
            auto zero_pad_left = padding_tail_size_;

            if (zero_pad_left >= simd_w_ - tail_size_ && isa != asimd) {
                uni_clear(vreg_zero_);
                movprfx(ZReg(vreg_zero_.getIdx()).s, tail_opmask_ / T_m,
                        ZReg(vreg_tmp_src0.getIdx()).s);
                const auto &addr
                        = prepare_xaddr_reg_with_dst_ptr(offt * dt_size);
                io_->store(conf_.dst_type, vreg_zero_, addr, false);
                off_base = simd_w_ * dt_size;
                zero_pad_left -= simd_w_ - tail_size_;
            } else {
                const auto &addr
                        = prepare_xaddr_reg_with_dst_ptr(offt * dt_size);
                io_->store(conf_.dst_type, vreg_tmp_src0, addr, true);
                off_base = tail_size_ * dt_size;
            }

            if (zero_pad_left) {
                const auto off_start = off_base;
                const auto off_num
                        = off_start + zero_pad_left * dt_size - off_start;
                eor(X_TMP_4, X_TMP_4, X_TMP_4);
                const auto &reg_ptr = prepare_xaddr_reg_with_dst_ptr(
                        offt * dt_size + off_start);
                int done = 0;
                int residual = off_num;
                while (residual > 0) {
                    assert(done < 256);
                    if (residual >= 8) {
                        str(X_TMP_4, ptr(reg_ptr, done));
                        done += 8;
                    } else if (residual >= 4) {
                        str(W_TMP_4, ptr(reg_ptr, done));
                        done += 4;
                    } else if (residual >= 2) {
                        strh(W_TMP_4, ptr(reg_ptr, done));
                        done += 2;
                    } else if (off_num > 0) {
                        strb(W_TMP_4, ptr(reg_ptr, done));
                        done += 1;
                    }
                    residual = off_num - done;
                }
            }
        } else {
            const auto &addr = prepare_xaddr_reg_with_dst_ptr(offt * dt_size);
            io_->store(conf_.dst_type, vreg_tmp_src0, addr, tail);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::compute_dst_body(int unroll, bool tail) {
    for (int i = 0; i < unroll; i++) {
        const Vmm vreg_tmp_src0 = Vmm(i + vmm_start_idx_);
        const Vmm vreg_tmp = conf_.is_src_different_layouts
                ? vmm_gathered_src_
                : Vmm(unroll + i + vmm_start_idx_);
        const Vmm vreg_tmp_src1 = offt_src1_ ? vreg_tmp : vreg_bcast_src1_;
        const int offt = simd_w_ * i;
        const auto &addr = prepare_xaddr_reg_with_src0_ptr(
                offt * types::data_type_size(conf_.src0_type));
        io_->load(conf_.src0_type, vreg_tmp_src0, addr, tail);
        if (offt_src1_) load_src1(vreg_tmp_src1, offt, tail);

        // avoid multiple multiplication on input scale for broadcasted vreg
        // not needed for different layouts
        if (!conf_.is_src_different_layouts) {
            uni_mov(vreg_tmp, vreg_tmp_src1);
        }
        perform_op(
                vreg_tmp_src0, vreg_tmp, vreg_scales_src0_, vreg_scales_src1_);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::compute_dst(int unroll, bool tail) {
    compute_dst_body(unroll, tail);
    if (postops_injector_) apply_postops(unroll, tail);
    store(unroll, tail);
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::forward() {
    Label unroll_loop, unroll_loop_tail, nelems_tail, end;

    const auto src0_type_size = types::data_type_size(conf_.src0_type);
    const auto src1_type_size = types::data_type_size(conf_.src1_type);
    const auto dst_type_size = types::data_type_size(conf_.dst_type);

    if (conf_.is_src_different_layouts) push(reg_src1_);

    // if outer dims tail, do it outside outer dims loop
    if (!is_src1_outer_dims_tail_) {
        if (conf_.is_i8) {
            if (isa == asimd) {
                uni_clear(VReg(vreg_zero_.getIdx()));
            } else {
                uni_clear(ZReg(vreg_zero_.getIdx()));
            }
            eor(reg_offt_dst_, reg_offt_dst_,
                    reg_offt_dst_); // offt_dst to get addr of dst
        }

        eor(reg_offt_src0_, reg_offt_src0_,
                reg_offt_src0_); // offt_src0 to get addr of src0/dst
        if (!conf_.is_src_different_layouts)
            eor(reg_offt_src1_, reg_offt_src1_,
                    reg_offt_src1_); // offt_src1 to get addr of src1
        if (conf_.use_stride_rhs_postops && !conf_.is_i8)
            eor(reg_off_rhs_postops_, reg_off_rhs_postops_,
                    reg_off_rhs_postops_);
    }

    compute_bcast(false); // bcast/load vreg just one time per a kernel call

    // used in c_blocked strategy for last blocked if tail exists
    const bool treat_each_compute_step_as_tail
            = !conf_.is_i8 && is_tail_kernel_ && tail_size_;

    if (conf_.do_scale_src0) {
        uni_ld1rw(vreg_scales_src0_.s, reg_scales_src0_, 0);
    }
    if (conf_.do_scale_src1) {
        uni_ld1rw(vreg_scales_src1_.s, reg_scales_src1_, 0);
        if (conf_.broadcast_src1_value || offt_src1_ == 0)
            uni_fmul(vreg_bcast_src1_.s, vreg_bcast_src1_.s,
                    vreg_scales_src1_.s);
    }

    L(unroll_loop);
    {
        const size_t offt = unroll_regs_ * simd_w_;
        mov_imm(X_TMP_0, offt * dst_type_size);
        cmp(reg_reverse_spat_offt_, X_TMP_0);
        b(LT, unroll_loop_tail);

        compute_dst(unroll_regs_, treat_each_compute_step_as_tail);
        sub_imm(reg_reverse_spat_offt_, reg_reverse_spat_offt_,
                offt * dst_type_size, X_TMP_0);
        add_imm(reg_offt_src0_, reg_offt_src0_, offt * src0_type_size, X_TMP_1);
        if (conf_.is_i8) {
            if (!conf_.broadcast_src1_value && !conf_.is_src_different_layouts)
                add_imm(reg_offt_src1_, reg_offt_src1_, offt * src1_type_size,
                        X_TMP_0);
            add_imm(reg_offt_dst_, reg_offt_dst_, offt, X_TMP_0);
        } else {
            if (conf_.use_stride_src1 && !conf_.is_src_different_layouts)
                add_imm(reg_offt_src1_, reg_offt_src1_, offt * src1_type_size,
                        X_TMP_0);
            if (conf_.use_stride_rhs_postops)
                add_imm(reg_off_rhs_postops_, reg_off_rhs_postops_, offt,
                        X_TMP_0);
        }
        b(unroll_loop);
    }

    L(unroll_loop_tail);
    {
        mov_imm(X_TMP_0, simd_w_ * dst_type_size);
        cmp(reg_reverse_spat_offt_, X_TMP_0);
        b(LT, nelems_tail);

        compute_dst(1, treat_each_compute_step_as_tail);
        sub_imm(reg_reverse_spat_offt_, reg_reverse_spat_offt_,
                simd_w_ * dst_type_size, X_TMP_0);
        add_imm(reg_offt_src0_, reg_offt_src0_, simd_w_ * src0_type_size,
                X_TMP_1);
        if (conf_.is_i8) {
            if (!conf_.broadcast_src1_value && !conf_.is_src_different_layouts)
                add_imm(reg_offt_src1_, reg_offt_src1_,
                        simd_w_ * src1_type_size, X_TMP_0);
            add_imm(reg_offt_dst_, reg_offt_dst_, simd_w_, X_TMP_0);
        } else {
            if (conf_.use_stride_src1 && !conf_.is_src_different_layouts)
                add_imm(reg_offt_src1_, reg_offt_src1_,
                        simd_w_ * src1_type_size, X_TMP_0);
            if (conf_.use_stride_rhs_postops)
                add_imm(reg_off_rhs_postops_, reg_off_rhs_postops_, simd_w_,
                        X_TMP_0);
        }

        b(unroll_loop_tail);
    }

    L(nelems_tail);
    {
        cmp(reg_reverse_spat_offt_, 1);
        b(LT, end);

        compute_dst(1, true);
        // need to increase if forward over outer dims
        if (is_src1_outer_dims_tail_) {
            add_imm(reg_offt_src0_, reg_offt_src0_, tail_size_ * src0_type_size,
                    X_TMP_0);
            if (conf_.is_i8)
                add_imm(reg_offt_dst_, reg_offt_dst_, tail_size_, X_TMP_0);
            else {
                if (conf_.use_stride_rhs_postops)
                    add_imm(reg_off_rhs_postops_, reg_off_rhs_postops_,
                            tail_size_, X_TMP_0);
            }
        }
    }

    L(end);
    if (conf_.is_src_different_layouts) pop(reg_src1_);
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::forward_over_outer_dims() {
    const auto outer_dims_size
            = conf_.outer_dims * types::data_type_size(conf_.dst_type);

    if (conf_.is_i8) {
        if (isa == asimd) {
            uni_clear(VReg(vreg_zero_.getIdx()));
        } else {
            uni_clear(ZReg(vreg_zero_.getIdx()));
        }
        eor(reg_offt_dst_, reg_offt_dst_,
                reg_offt_dst_); // offt_dst to get addr of dst
    }

    eor(reg_offt_src0_, reg_offt_src0_,
            reg_offt_src0_); // offt_src0 to get addr of src0/dst
    if (conf_.use_stride_rhs_postops && !conf_.is_i8)
        eor(reg_off_rhs_postops_, reg_off_rhs_postops_, reg_off_rhs_postops_);

    Label c_loop;
    L(c_loop);
    {
        mov_imm(reg_reverse_spat_offt_, outer_dims_size);
        forward();
        sub_imm(reg_outer_dims_range_, reg_outer_dims_range_, outer_dims_size,
                X_TMP_0);
        cmp(reg_outer_dims_range_, 0);
        b(GT, c_loop);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_kernel_t<isa>::generate() {
    auto live_gprs = claim_regs<XReg>({reg_offt_src1_, reg_off_rhs_postops_,
            reg_scales_src0_, reg_scales_src1_, reg_tmp1_, reg_src0_, reg_src1_,
            reg_dst_, reg_offt_src0_, reg_outer_dims_range_,
            reg_reverse_spat_offt_, reg_tmp_, reg_src1_stride_range_});
    auto live_vmms = claim_regs<Vmm>({vmm_tail_vmask_, vreg_sum_scale_,
            vreg_zero_, vreg_one_, vreg_saturation_ubound_, vreg_bcast_src1_,
            vreg_scales_src0_, vreg_scales_src1_, vreg_saturation_lbound_,
            vmm_indices_});
    auto live_tail_opmask = claim_reg(tail_opmask_);

    // All result vectors are live across every post-op in the chain.
    auto live_results = claim_reg_range<Vmm>(vmm_start_idx_, unroll_regs_);

    auto eltwise_table = alloc_reg<XReg>();
    auto eltwise_mask = alloc_reg<PReg>();
    if (conf_.with_postops)
        init_post_ops_injector(eltwise_table.get(), eltwise_mask.get());

    preamble();

    if (isa == sve_256)
        ptrue(P_ALL_ONE.b, VL32);
    else if (isa == sve_128)
        ptrue(P_ALL_ONE.b, VL16);

    io_->init_tail();
    io_->init_saturation(conf_.dst_type);

    load_kernel_params();
    prepare_isa_kernel();
    // if outer dims is not aligned to simd_w, iterate over it to avoid
    // modifying the gather_load indices
    if (is_src1_outer_dims_tail_)
        forward_over_outer_dims();
    else
        forward();
    postamble();

    if ((conf_.with_eltwise || conf_.is_i8) && postops_injector_)
        postops_injector_->prepare_table();
    postops_injector_.reset();
}

#undef PARAM_OFF

template struct jit_uni_binary_kernel_t<asimd>;
template struct jit_uni_binary_kernel_t<sve_512>;
template struct jit_uni_binary_kernel_t<sve_256>;
template struct jit_uni_binary_kernel_t<sve_128>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
