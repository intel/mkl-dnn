/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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
#include <cmath>
#include <utility>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_uni_layer_normalization.hpp"
#include "cpu/aarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace memory_tracking::names;
using namespace Xbyak_aarch64;

namespace {

bcast_set_t get_supported_bcast_strategies(int ndims) {
    assert(ndims > 1 && ndims <= 5);
    bcast_set_t set {broadcasting_strategy_t::scalar};
    switch (ndims) {
        case 2: set.insert(broadcasting_strategy_t::per_oc); break;
        case 3:
        case 4:
        case 5: set.insert(broadcasting_strategy_t::per_w); break;
        default: assert(!"Unsupported ndims");
    }
    return set;
}

template <cpu_isa_t isa>
class jit_lnorm_stat_and_data_kernel_t : public stat_and_data_kernel_iface_t,
                                         public jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_lnorm_stat_and_data_kernel_t)

    using TReg = typename cpu_isa_traits<isa>::TReg;

    jit_lnorm_stat_and_data_kernel_t(const layer_normalization_pd_t *pd)
        : jit_generator_t(nullptr, MAX_CODE_SIZE, true)
        , pd_(pd)
        , src_d_(pd_->src_md())
        , dst_d_(pd_->dst_md())
        , simd_w_(isa_max_vlen(isa) / sizeof(float))
        , C_(pd_->norm_axis())
        , axis_simd_full_(C_ / simd_w_)
        , axis_simd_tail_(C_ % simd_w_)
        , use_scale_(pd_->use_scale())
        , use_shift_(pd_->use_shift())
        , save_stats_(pd_->is_training())
        , calculate_stats_(!pd_->stats_are_src())
        , eps_(pd_->desc()->layer_norm_epsilon)
        , skip_mean_(pd_->skip_mean()) {

        const io::saturation_conf_t<TReg> sat_conf(
                vec_zero_, vec_saturation_ubound_);

        io::tail_conf_t tail_conf(axis_simd_tail_, tail_opmask_);

        io_ = utils::make_unique<io::jit_io_helper_t<isa>>(
                this, sat_conf, tail_conf, reg_io_tmp1_);

        const auto &post_ops = pd_->attr()->post_ops_;
        with_postops_ = post_ops.len() != 0;
        with_binary_ = post_ops.find(primitive_kind::binary) != -1;
        with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;
        const auto &attr_scales = pd_->attr()->scales_;
        with_src_scales_ = !attr_scales.has_default_values(DNNL_ARG_SRC);
        with_dst_scales_ = !attr_scales.has_default_values(DNNL_ARG_DST);

        if (with_postops_) {
            static constexpr bool use_exact_tail_scalar_bcast = true;

            const binary_injector::rhs_arg_static_params_t binary_rhs_params {
                    static_cast<size_t>(
                            offsetof(ker_args_t, post_ops_binary_rhs_arg_vec)),
                    static_cast<size_t>(offsetof(ker_args_t, dst)), dst_d_,
                    static_cast<size_t>(axis_simd_tail_), tail_opmask_,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t binary_params {reg_param_,
                    get_supported_bcast_strategies(dst_d_.ndims()),
                    binary_rhs_params};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(this, reg_pool(),
                    pd_->attr()->post_ops_, binary_params, reg_elt_inj_table_,
                    elt_inj_opmask_);
        }
    }

    void operator()(const ker_args_t &args) const override {
        jit_generator_t::operator()(args);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    const layer_normalization_pd_t *pd_;
    const memory_desc_wrapper src_d_;
    const memory_desc_wrapper dst_d_;
    const size_t simd_w_;
    const dim_t C_;
    const dim_t axis_simd_full_;
    const dim_t axis_simd_tail_;
    const bool use_scale_;
    const bool use_shift_;
    const bool save_stats_;
    const bool calculate_stats_;
    const float eps_;
    const bool skip_mean_;
    bool with_postops_ = false;
    bool with_binary_ = false;
    bool with_eltwise_ = false;
    bool with_src_scales_ = false;
    bool with_dst_scales_ = false;

    const XReg reg_param_ = abi_param1;
    const XReg reg_po_binary_rhs_addr_ = x1;
    const XReg reg_po_helper_ = x2;
    const XReg reg_po_cache_ = x3;
    const XReg reg_elt_inj_table_ = x4;
    const XReg reg_po_binary_dst_addr_ = x5;
    const XReg reg_src_ = x8;
    const XReg reg_dst_ = x9;
    const XReg reg_scale_ = x10;
    const XReg reg_shift_ = x11;
    const XReg reg_mean_ = x12;
    const XReg reg_var_ = x13;
    const XReg reg_block_end_ = x14;
    const XReg reg_io_tmp1_ = x15;
    const XReg reg_src_scales_ = x17;
    const XReg reg_dst_scales_ = x19;
    const XReg reg_stat_loop_idx_ = X_TMP_1;

    const PReg tail_opmask_ = p2;
    const PReg elt_inj_opmask_ = p1;
    const PReg elt_inj_p_tmp0_ = p3;

    const TReg vec_tail_mask_ {0};
    const TReg vec_zero_ {1};
    const TReg vec_saturation_ubound_ {2};
    const TReg vec_qscale_ {3};
    const TReg vec_scale_ {4};
    const TReg vec_shift_ {5};
    const TReg vec_tmp_acc_ {6};
    const TReg vec_mean_ {7};
    const TReg vec_inv_sqrtvar_ {8};
    const TReg vec_tmp_data_ {9};
    const TReg vec_dst_ {10};
    const TReg vec_po_helper_ {11};
    const TReg vec_eps_ {12};

    const SReg reg_c_ {13};
    const SReg float_one_ {14};
    const SReg scalar_acc_ {vec_tmp_acc_.getIdx()};
    const SReg scalar_eps_ {vec_eps_.getIdx()};
    const SReg scalar_inv_sqrtvar_ {vec_inv_sqrtvar_.getIdx()};
    const SReg scalar_mean_ {vec_mean_.getIdx()};

    std::unique_ptr<io::jit_io_helper_t<isa>> io_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    XReg addr_with_offt(const XReg &base, size_t offt_bytes) {
        return addr_off(base, static_cast<int64_t>(offt_bytes), X_DEFAULT_ADDR,
                X_TMP_0);
    }

    void compute_postops(size_t offt_elems, bool is_tail) {
        auto live_gprs = claim_regs<XReg>({reg_po_binary_rhs_addr_,
                reg_po_helper_, reg_po_cache_, reg_elt_inj_table_,
                reg_po_binary_dst_addr_, reg_src_, reg_dst_, reg_scale_,
                reg_shift_, reg_mean_, reg_var_, reg_block_end_, reg_io_tmp1_,
                reg_src_scales_, reg_dst_scales_, reg_stat_loop_idx_});
        auto live_vmms = claim_reg_range<TReg>(0, float_one_.getIdx() + 1);
        auto live_preds = claim_regs<PReg>(
                {tail_opmask_, elt_inj_opmask_, elt_inj_p_tmp0_});

        add_imm(reg_po_binary_dst_addr_, reg_dst_,
                static_cast<int>(offt_elems * dst_d_.data_type_size()),
                X_TMP_0);

        if (with_binary_) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            rhs_arg_params.vmm_idx_to_out_reg.emplace(
                    vec_dst_.getIdx(), reg_po_binary_dst_addr_);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vec_dst_.getIdx(), 0);
            if (is_tail)
                rhs_arg_params.vmm_tail_idx_.emplace(vec_dst_.getIdx());
            postops_injector_->compute_vector(
                    vec_dst_.getIdx(), rhs_arg_params);
        } else {
            postops_injector_->compute_vector(vec_dst_.getIdx());
        }
    }

    void calculate_dst_body(size_t offt_elems, bool is_tail) {
        if (use_scale_) {
            io_->load(f32, vec_scale_,
                    addr_with_offt(reg_scale_, offt_elems * sizeof(float)),
                    is_tail);
        }
        if (use_shift_) {
            io_->load(f32, vec_shift_,
                    addr_with_offt(reg_shift_, offt_elems * sizeof(float)),
                    is_tail);
        }

        io_->load(src_d_.data_type(), vec_dst_,
                addr_with_offt(reg_src_, offt_elems * src_d_.data_type_size()),
                is_tail);

        if (!skip_mean_) fsub(vec_dst_.s, vec_dst_.s, vec_mean_.s);
        fmul(vec_dst_.s, vec_dst_.s, vec_inv_sqrtvar_.s);

        if (use_scale_ && use_shift_) {
            uni_fmad(vec_dst_.s, vec_scale_.s, vec_shift_.s);
        } else {
            if (use_scale_) fmul(vec_dst_.s, vec_dst_.s, vec_scale_.s);
            if (use_shift_) fadd(vec_dst_.s, vec_dst_.s, vec_shift_.s);
        }

        if (with_src_scales_) {
            uni_ld1rw(vec_qscale_.s, reg_src_scales_, 0);
            fmul(vec_dst_.s, vec_dst_.s, vec_qscale_.s);
        }

        if (with_postops_) { compute_postops(offt_elems, is_tail); }

        if (with_dst_scales_) {
            uni_ld1rw(vec_qscale_.s, reg_dst_scales_, 0);
            fmul(vec_dst_.s, vec_dst_.s, vec_qscale_.s);
        }

        io_->store(dst_d_.data_type(), vec_dst_,
                addr_with_offt(reg_dst_, offt_elems * dst_d_.data_type_size()),
                is_tail);
    }

    void compute_mean() {
        uni_clear(vec_tmp_acc_);

        mov(X_DEFAULT_ADDR, reg_src_);
        asm_for(reg_stat_loop_idx_, axis_simd_full_, [&]() {
            io_->load(src_d_.data_type(), vec_tmp_data_, X_DEFAULT_ADDR);

            fadd(vec_tmp_acc_.s, vec_tmp_acc_.s, vec_tmp_data_.s);

            add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR,
                    simd_w_ * src_d_.data_type_size(), X_TMP_0);
        });

        if (axis_simd_tail_) {
            io_->load(src_d_.data_type(), vec_tmp_data_,
                    addr_with_offt(reg_src_,
                            axis_simd_full_ * simd_w_
                                    * src_d_.data_type_size()),
                    true);

            fadd(vec_tmp_acc_.s, vec_tmp_acc_.s, vec_tmp_data_.s);
        }

        uni_fadd_reduce(scalar_acc_, vec_tmp_acc_.s);
        fdiv(scalar_mean_, scalar_acc_, reg_c_);

        if (save_stats_ && !skip_mean_) { str(scalar_mean_, ptr(reg_mean_)); }
    }

    void compute_var() {
        uni_clear(vec_tmp_acc_);

        dup(vec_mean_.s, vec_mean_.s[0]);

        mov(X_DEFAULT_ADDR, reg_src_);
        asm_for(reg_stat_loop_idx_, axis_simd_full_, [&]() {
            io_->load(src_d_.data_type(), vec_tmp_data_, X_DEFAULT_ADDR);

            if (!skip_mean_)
                fsub(vec_tmp_data_.s, vec_tmp_data_.s, vec_mean_.s);

            float_point_fused_multiply_add(
                    vec_tmp_acc_, vec_tmp_data_, vec_tmp_data_);

            add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR,
                    simd_w_ * src_d_.data_type_size(), X_TMP_0);
        });

        if (axis_simd_tail_) {
            io_->load(src_d_.data_type(), vec_tmp_data_,
                    addr_with_offt(reg_src_,
                            axis_simd_full_ * simd_w_
                                    * src_d_.data_type_size()),
                    true);

            if (!skip_mean_) {
                if (isa == asimd) {
                    const auto &vreg_data = VReg16B(vec_tmp_data_.getIdx());
                    const auto &vreg_mean = VReg16B(vec_mean_.getIdx());

                    // asimd does not have predicated subtraction resulting in
                    // unwanted values in the dst. E.g, if we have a tail size
                    // of two we get:
                    // [s1, s2, 0, 0] - [m, m, m, m] = [s1 - m, s2 - m, -m, -m]
                    //
                    // To deal with the extra -m elements in the tail mask we
                    // use ext to replace the zeros with +m: [s1, s2, +m, +m] -
                    // [m, m, m, m] = [s1 - m, s2 - m, 0, 0]
                    //
                    // In SVE this can be achieved directly with predicated
                    // subtraction.
                    ext(vreg_data, vreg_mean, vreg_data,
                            axis_simd_tail_ * sizeof(float));
                    fsub(vec_tmp_data_.s, vec_tmp_data_.s, vec_mean_.s);
                } else {
                    fsub(ZRegS(vec_tmp_data_.getIdx()), tail_opmask_ / T_z,
                            ZRegS(vec_mean_.getIdx()));
                }
            }

            float_point_fused_multiply_add(
                    vec_tmp_acc_, vec_tmp_data_, vec_tmp_data_);
        }

        uni_fadd_reduce(scalar_acc_, vec_tmp_acc_.s);
        fdiv(scalar_inv_sqrtvar_, scalar_acc_, reg_c_);

        if (save_stats_) str(scalar_inv_sqrtvar_, ptr(reg_var_));
    }

    void generate() override {
        const size_t c_src_size
                = C_ * types::data_type_size(src_d_.data_type());
        const size_t c_dst_size
                = C_ * types::data_type_size(dst_d_.data_type());
        static const size_t float_size = types::data_type_size(f32);

        preamble();

        io_->init_tail();
        io_->init_saturation(dst_d_.data_type());

#define PARAM_OFF(x) static_cast<int32_t>(offsetof(ker_args_t, x))
        ldr(reg_src_, ptr(reg_param_, PARAM_OFF(src)));
        ldr(reg_dst_, ptr(reg_param_, PARAM_OFF(dst)));
        ldr(reg_scale_, ptr(reg_param_, PARAM_OFF(scale)));
        ldr(reg_shift_, ptr(reg_param_, PARAM_OFF(shift)));
        ldr(reg_mean_, ptr(reg_param_, PARAM_OFF(mean)));
        ldr(reg_var_, ptr(reg_param_, PARAM_OFF(var)));
        ldr(reg_src_scales_, ptr(reg_param_, PARAM_OFF(src_scales)));
        ldr(reg_dst_scales_, ptr(reg_param_, PARAM_OFF(dst_scales)));
        ldr(reg_block_end_, ptr(reg_param_, PARAM_OFF(block_size_bytes)));
#undef PARAM_OFF

        mov_imm(W_TMP_0, float2int(eps_));
        fmov(scalar_eps_, W_TMP_0);

        mov_imm(W_TMP_0, float2int(static_cast<float>(C_)));
        fmov(reg_c_, W_TMP_0);

        fmov(float_one_, 1.0f);
        add(reg_block_end_, reg_block_end_, reg_src_);

        Label loop, done;
        L(loop);
        cmp(reg_src_, reg_block_end_);
        b(GE, done);

        if (calculate_stats_) {
            // We can't skip compute_mean() here even if skip_mean is set
            // because it will be needed for the variance calculation.
            compute_mean();
            compute_var();
        } else {
            if (!skip_mean_) { ldr(scalar_mean_, ptr(reg_mean_)); }
            ldr(scalar_inv_sqrtvar_, ptr(reg_var_));
            dup(vec_mean_.s, vec_mean_.s[0]);
        }

        fadd(scalar_inv_sqrtvar_, scalar_inv_sqrtvar_, scalar_eps_);
        fsqrt(scalar_inv_sqrtvar_, scalar_inv_sqrtvar_);
        fdiv(scalar_inv_sqrtvar_, float_one_, scalar_inv_sqrtvar_);

        dup(vec_inv_sqrtvar_.s, vec_inv_sqrtvar_.s[0]);

        for (dim_t i = 0; i < axis_simd_full_; ++i)
            calculate_dst_body(i * simd_w_, false);
        if (axis_simd_tail_)
            calculate_dst_body(axis_simd_full_ * simd_w_, true);

        add_imm(reg_src_, reg_src_, c_src_size, X_TMP_0);
        add_imm(reg_dst_, reg_dst_, c_dst_size, X_TMP_0);
        add_imm(reg_mean_, reg_mean_, float_size, X_TMP_0);
        add_imm(reg_var_, reg_var_, float_size, X_TMP_0);
        b(loop);

        L(done);
        postamble();

        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table(true);
    }
};

} // namespace

template <cpu_isa_t isa>
status_t jit_uni_layer_normalization_fwd_t<isa>::pd_t::init(
        const engine_t *engine) {
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    const memory_desc_wrapper src_d(src_md());

    VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_LNORM(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_LNORM(utils::one_of(src_md()->data_type, f32, f16, bf16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(utils::one_of(dst_md()->data_type, f32, f16, bf16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(platform::has_data_type_support(src_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(platform::has_data_type_support(dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    const auto supported_skip_mask
            = skip_mask_t::scales | skip_mask_t::post_ops;
    VDISPATCH_LNORM(attr()->has_default_values(supported_skip_mask),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_LNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LNORM(src_d.is_blocking_desc(), VERBOSE_BLOCKING_FAIL,
            "blocking descriptor fail");
    VDISPATCH_LNORM(src_d.blocking_desc().strides[ndims() - 1] == 1,
            VERBOSE_BLOCKING_FAIL, "bad stride value");
    VDISPATCH_LNORM(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    auto post_ops_ok = [&]() -> bool {
        const memory_desc_wrapper dst_d(dst_md());
        injector::post_ops_ok_args_t post_ops_args(isa,
                {injector::eltwise, injector::binary}, attr()->post_ops_,
                &dst_d, true, true, true, true,
                get_supported_bcast_strategies(dst_d.ndims()));
        return injector::post_ops_ok(post_ops_args);
    };

    VDISPATCH_LNORM(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_LNORM(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_LNORM(fill_compatible_stats_md(*src_md(), reordered_stat_md_)
                    == status::success,
            VERBOSE_INCONSISTENT_MDS, "src", "stat");

    if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
        CHECK(reorder_primitive_desc_create(reorder_pd_, engine,
                stats_are_src() ? stat_md() : &reordered_stat_md_,
                stats_are_src() ? &reordered_stat_md_ : stat_md()));
    }

    nthr_ = dnnl_get_max_threads();
    init_scratchpad();
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_layer_normalization_fwd_t<isa>::init(engine_t *engine) {
    if (pd()->reorder_pd_)
        pd()->reorder_pd_->create_primitive(reorder_, engine);

    CHECK(safe_ptr_assign(stat_and_data_kernel_,
            new jit_lnorm_stat_and_data_kernel_t<isa>(pd())));
    CHECK(stat_and_data_kernel_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa>
void jit_uni_layer_normalization_fwd_t<isa>::reorder_stat(const exec_ctx_t &ctx,
        engine_t *engine, const memory_arg_t &in,
        const memory_arg_t &out) const {
    exec_args_t r_args;
    r_args[DNNL_ARG_SRC] = in;
    r_args[DNNL_ARG_DST] = out;
    exec_ctx_t r_ctx(ctx, std::move(r_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, reorder_->pd()->scratchpad_registry());
    r_ctx.set_scratchpad_grantor(nested_grantor);
    reorder_->execute(r_ctx);
}

template <cpu_isa_t isa>
status_t jit_uni_layer_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    engine_t *engine = ctx.stream()->engine();
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const bool skip_mean = pd()->skip_mean();

    std::unique_ptr<memory_t, memory_deleter_t> mean;
    if (!skip_mean) {
        auto mean_mem = scratchpad.get_memory_storage(key_lnorm_tmp_mean);
        CHECK(safe_ptr_assign(mean,
                new memory_t(engine, &(pd()->reordered_stat_md_),
                        std::move(mean_mem))));
    }

    auto variance_mem = scratchpad.get_memory_storage(key_lnorm_tmp_var);
    std::unique_ptr<memory_t, memory_deleter_t> variance;
    CHECK(safe_ptr_assign(variance,
            new memory_t(engine, &(pd()->reordered_stat_md_),
                    std::move(variance_mem))));

    if (pd()->stats_are_src() && reorder_) {
        if (!skip_mean) {
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_MEAN),
                    {mean.get(), false});
        }
        reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                {variance.get(), false});
    }

    CHECK(execute_forward(ctx));

    if (!pd()->stats_are_src() && reorder_) {
        if (!skip_mean) {
            reorder_stat(ctx, engine, {mean.get(), true},
                    ctx.args().at(DNNL_ARG_MEAN));
        }
        reorder_stat(ctx, engine, {variance.get(), true},
                ctx.args().at(DNNL_ARG_VARIANCE));
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_layer_normalization_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    const bool skip_mean = pd()->skip_mean();

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = skip_mean ? nullptr
                         : scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                          CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const dim_t N = pd()->across_axis();
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];
    const dim_t C = src_d.dims()[pd()->ndims() - 1];

    parallel(pd()->nthr_, [= COMPAT_THIS_CAPTURE](int ithr, int nthr) {
        dim_t N_start_idx = 0, N_end_idx = 0;
        balance211(N, nthr, ithr, N_start_idx, N_end_idx);
        const char *const src_ptr = reinterpret_cast<const char *>(src)
                + N_start_idx * C_padded * src_d.data_type_size();
        char *const dst_ptr = reinterpret_cast<char *>(dst)
                + N_start_idx * C_padded * dst_d.data_type_size();
        const int N_block_count = N_end_idx - N_start_idx;
        const size_t block_size_bytes
                = N_block_count * C * types::data_type_size(src_d.data_type());
        float *mean_ptr = skip_mean ? nullptr : mean + N_start_idx;
        float *dst_scales_inv_ptr = nullptr;
        if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_DST)) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(key_lnorm_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        const stat_and_data_kernel_iface_t::ker_args_t args {src_ptr, dst_ptr,
                scale, shift, mean_ptr, &variance[N_start_idx], src_scales,
                dst_scales_inv_ptr, post_ops_binary_rhs_arg_vec.data(),
                block_size_bytes};

        (*stat_and_data_kernel_)(args);
    });

    return status::success;
}

template class jit_uni_layer_normalization_fwd_t<asimd>;
template class jit_uni_layer_normalization_fwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
