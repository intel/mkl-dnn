/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
* Copyright 2019-2026 Arm Ltd. and affiliates
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

#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"

#define IDX(a) static_cast<uint32_t>((a).getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace eltwise_injector {

bool is_isa_supported(cpu_isa_t isa) {
    return isa == asimd || isa == sve;
}

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_soft_relu, eltwise_logistic, eltwise_mish, eltwise_exp,
            eltwise_gelu_tanh, eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_swish, eltwise_log, eltwise_clip, eltwise_clip_v2,
            /*eltwise_pow, */ eltwise_gelu_erf, eltwise_round,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
            eltwise_clip_v2_use_dst_for_bwd);
}

bool is_supported(cpu_isa_t isa, alg_kind_t alg) {
    using namespace alg_kind;
    return is_isa_supported(isa) && is_alg_supported(alg);
}

} // namespace eltwise_injector

using namespace Xbyak_aarch64;

namespace {
class eltwise_probe_generator_t final : public jit_generator_t {
public:
    RegPoolManager &probe_reg_pool() { return reg_pool(); }
    const char *name() const override { return "eltwise_scratch_probe"; }
    const char *source_file() const override { return __FILE__; }

private:
    void generate() override {}
};
} // namespace

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::set_input_range(
        float min_value, float max_value) {
    min_input_ = min_value;
    max_input_ = max_value;
    scratch_vmm_count_ = static_cast<size_t>(-1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_cmp_mask(const TRegS &vmm_src,
        const TRegS &compare_operand, alg_kind_t cmp_predicate) {
    auto p_tmp0 = reg_pool_.allocScoped<PReg>();
    h->mov(PRegB(IDX(p_tmp0)), p_all / T_z, p_all.b);
    using namespace alg_kind;
    switch (cmp_predicate) {
        case binary_gt:
            h->fcmgt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case binary_ge:
            h->fcmge(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case binary_eq:
            h->fcmeq(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case binary_ne:
            h->fcmne(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case binary_lt:
            h->fcmlt(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        case binary_le:
            h->fcmle(
                    PRegS(IDX(p_mask)), p_tmp0 / T_z, vmm_src, compare_operand);
            break;
        default: assert(!"Unsupported compare mode"); break;
    }
}

// Uses injector masks objects: p_mask
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::blend_with_mask(
        const TRegS &vmm_dst, const TRegS &src) {
    h->sel(vmm_dst, p_mask / T_m, src, vmm_dst);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector_fwd(
        const TRegS &vmm_src) {
    exp_compute_vector_fwd(vmm_src, min_input_, max_input_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector_fwd(
        const TRegS &vmm_src, float min_input, float max_input) {

    const auto &t0 = vmm_src;
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    const float ln_flt_max = logf(FLT_MAX);
    const float ln_flt_min = logf(FLT_MIN);
    if (max_input > ln_flt_max)
        h->fmin(t0, p_all, table_val(exp_ln_flt_max_f, table_value));
    if (min_input < ln_flt_min)
        h->fmax(t0, p_all, table_val(exp_ln_flt_min_f, table_value));
    h->fmul(t0, t0, table_val(exp_log2ef, t1));
    h->frintm(t1, p_all, t0);
    h->fcvtzs(t2, p_all, t1);
    h->fsub(t1, t0, t1);
    h->fadd(t0, t1, table_val(one, table_value));
    h->lsr(t1, t0, 17);
    h->fexpa(t1, t1);
    h->fscale(t1, p_all, t2);
    h->and_(ZRegD(t2.getIdx()), ZRegD(t0.getIdx()),
            ZRegD(IDX(table_val(exp_not_mask17, table_value))));
    h->fsub(t2, t0, t2);
    h->fmad(table_val(exp_coeff2, t0), p_all, t2,
            table_val(exp_coeff1, table_value));
    h->fmad(t0, p_all, t2, table_val(one, table_value));
    h->fmul(t0, t1, t0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(src.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fcmgt(p_mask.s, p_all, vmm_src, 0.0);
    h->fmul(vmm_src, vmm_src, alpha);
    h->sel(vmm_src, p_mask, src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_zero_ns_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fmaxnm(vmm_src.h, p_all, 0.);
    else
        h->fmaxnm(vmm_src.s, p_all, 0.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::elu_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(src.getIdx()), ZRegD(vmm_src.getIdx()));

    // compute exponent
    exp_compute_vector_fwd(vmm_src, min_input_, 0.f);

    // alpha * (exp(x) - 1) = alpha * exp(x) - alpha
    if (alpha_ != 1.f) {
        h->fnmsb(vmm_src, p_all / T_m, alpha, alpha);
    } else {
        h->fsub(vmm_src, p_all / T_m, 1.);
    }

    // combine with mask
    h->fcmgt(p_mask.s, p_all / T_z, src, 0.);
    h->mov(vmm_src, p_mask / T_m, src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_polynomial_approx_compute_vector_fwd(
        const TRegS &vmm_src) {

    if (vlen != 64) return;

    using namespace Xbyak_aarch64::util;

    const int tanh_n_polynomials = 32;

    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto work = reg_pool_.allocScoped<TRegS>();
    auto polynomial = reg_pool_.allocScoped<TRegS>();
    auto indices = reg_pool_.allocScoped<TRegS>();
    auto src_pos = reg_pool_.allocScoped<TRegS>();
    const auto &bounds = table_value;

    auto p_tmp0 = reg_pool_.allocScoped<PReg>();
    const auto &mask = p_tmp0;

    // Helper function to gather polynomial coefficients
    auto gather_coefficient
            = [&](TRegS vmm_coeff, int coeff_idx, TRegS vmm_pol_idx) {
        auto table_addr = reg_pool_.allocScoped<XReg>();
        h->add_imm(table_addr, x_table,
                table_off(tanh_pol_table, coeff_idx * tanh_n_polynomials),
                table_addr);
        h->ld1w(vmm_coeff, p_all, ptr(table_addr, vmm_pol_idx, SXTW));
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    h->fabs(src_pos, p_all / T_z, vmm_src);

    // Compute indices for the table lookup
    h->sub(indices, src_pos, table_val(tanh_idx_bias, table_value));
    h->and_(ZRegD(IDX(indices)), ZRegD(IDX(indices)),
            ZRegD(IDX(table_val(tanh_idx_mask, table_value))));
    h->lsr(ZRegD(IDX(indices)), ZRegD(IDX(indices)), 20);

    // Argument reduction
    h->and_(ZRegD(IDX(work)), ZRegD(IDX(src_pos)),
            ZRegD(IDX(table_val(tanh_idx_mask, table_value))));
    h->fsub(src_pos, src_pos, work);

    gather_coefficient(polynomial, 6, indices);
    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(work, deg, indices);
        h->fmad(polynomial, p_all / T_m, src_pos, work);
    }

    // Restore src_pos
    h->fabs(src_pos, p_all / T_z, vmm_src);

    // Now Blend the results
    // [saturation_ubound; +inf] : return +/- 1
    table_val(one, work);

    // [linear_ubound; saturation_lbound] :  return +/- P(x)
    table_val(tanh_saturation_lbound, bounds);
    h->fcmgt(PRegS(IDX(mask)), p_all / T_z, bounds, src_pos);
    h->sel(work, mask / T_m, polynomial, work);

    // [0; linear_ubound]  :  return x
    table_val(tanh_linear_ubound, bounds);
    h->fcmgt(PRegS(IDX(mask)), p_all / T_z, bounds, src_pos);
    h->sel(work, mask / T_m, src_pos, work);

    // Reapply sign and return
    h->and_(ZRegD(IDX(src_pos)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, table_value))));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(work)), ZRegD(IDX(src_pos)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_compute_vector_fwd(
        const TRegS &vmm_src) {

    if (vlen == 64) {
        tanh_polynomial_approx_compute_vector_fwd(vmm_src);
        return;
    }

    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    const auto &t0 = vmm_src;
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    auto one = reg_pool_.allocScoped<TRegS>();
    auto mask = reg_pool_.allocScoped<PReg>();

    h->fcpy(one, p_all, 1);
    // make mask for small x
    h->mov(t2, p_all, t0);
    h->fabs(t1, p_all, t0);
    h->cmplt(mask.get().s, p_all, t1, table_val(tanh_range, table_value));

    // 2x
    h->fadd(t0, t0, t0);
    // exp(2x)
    exp_compute_vector_fwd(t0);
    // 1+exp(2x)
    h->fadd(t0, t0, one);
    // 2/(1+exp(2x))
    h->fdiv(table_val(two, t1), p_all, t0);
    // 1-2/(1+exp(2x))
    h->fsub(t0, one, t1);

    // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    h->fmul(t1, t2, t2);
    h->fmad(t1, p_all, table_val(tanh_m1d3, table_value), one);
    h->fmul(t1, p_all, t2);
    // select the correct value according to mask
    h->mov(t0, mask, t1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    auto src = reg_pool_.allocScoped<TRegS>();
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto factor = reg_pool_.allocScoped<TRegS>();
    const auto &fitting = factor;
    h->mov(ZRegD(IDX(src)), ZRegD(IDX(vmm_src)));

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->fmul(vmm_src, vmm_src, vmm_src);
    h->fmad(vmm_src, p_all / T_m, table_val(gelu_tanh_fitting_const, fitting),
            table_val(one, table_value));
    h->fmul(factor, src, table_val(gelu_tanh_sqrt_two_over_pi, table_value));
    h->fmul(vmm_src, vmm_src, factor);

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->fmad(vmm_src, p_all / T_m, src, src);
    h->fmul(vmm_src, p_all / T_m, 0.5f);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::square_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fmul(vmm_src.h, vmm_src.h, vmm_src.h);
    else
        h->fmul(vmm_src.s, vmm_src.s, vmm_src.s);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::abs_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fabs(vmm_src.h, p_all / T_m, vmm_src.h);
    else
        h->fabs(vmm_src.s, p_all / T_m, vmm_src.s);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::sqrt_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fsqrt(vmm_src.h, p_all / T_m, vmm_src.h);
    else
        h->fsqrt(vmm_src.s, p_all / T_m, vmm_src.s);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::linear_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // compute x = alpha * x + beta;
    h->fmad(vmm_src, p_all / T_m, alpha, beta);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_compute_vector_fwd(
        const TReg &vmm_src, const TRegS &lower, const TRegS &upper) {
    if (d_type_ == data_type::f16) {
        h->fmaxnm(vmm_src.h, p_all, TRegH(IDX(lower)));
        h->fminnm(vmm_src.h, p_all, TRegH(IDX(upper)));
    } else {
        h->fmaxnm(vmm_src.s, p_all, lower);
        h->fminnm(vmm_src.s, p_all, upper);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::mish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // An equation other than mish(x) = x*tanh(srelu(x)) was used
    // to calculate mish, but it should be remembered that it is equivalent
    // equation, it uses the following rule:
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x),
    // hence the equation for mish can take the form:
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
    // This option was chosen because computing tanh requires more registers
    // than exp, and also requires more constants to be stored in memory,
    // making the algorithm slower.

    auto src = reg_pool_.allocScoped<TRegS>();
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto denominator = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(src.getIdx()), ZRegD(vmm_src.getIdx()));

    // mish uses (exp(x)+1)^2;
    // avoid overflow by x <= log(sqrt(FLT_MAX))
    h->fminnm(vmm_src, p_all / T_m,
            table_val(fwd_mish_max_x_for_equation_f, table_value));
    exp_compute_vector_fwd(vmm_src, min_input_, 0.5f * logf(FLT_MAX));

    // (e^x+1)^2
    h->fadd(vmm_src, p_all / T_m, 1.);
    h->fmul(vmm_src, vmm_src, vmm_src);

    // x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1)
    h->fadd(denominator, vmm_src, table_val(one, table_value));
    h->fsub(vmm_src, p_all / T_m, 1.);
    h->fdiv(vmm_src, p_all / T_m, denominator);
    h->fmul(vmm_src, vmm_src, src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardswish_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // result = x * hardsigmoid(x)
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(src.getIdx()), ZRegD(vmm_src.getIdx()));
    hardsigmoid_compute_vector_fwd(vmm_src, alpha, beta);
    h->fmul(vmm_src, vmm_src, src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardsigmoid_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // result = max(0, min(1, alpha * x + beta))
    h->fmul(vmm_src, vmm_src, alpha);
    h->fadd(vmm_src, vmm_src, beta);
    h->fminnm(vmm_src, p_all / T_m, 1.);
    h->fmaxnm(vmm_src, p_all / T_m, 0.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::soft_relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto tmp = reg_pool_.allocScoped<TRegS>();
    auto work = reg_pool_.allocScoped<TRegS>();
    auto saved_src = reg_pool_.allocScoped<TRegS>();
    auto polynomial = reg_pool_.allocScoped<TRegS>();
    // alpha scaling
    if (alpha_ == 1.f) {
        // Nothing to do
    } else if (alpha_ == -1) {
        h->fneg(vmm_src, p_all / T_m, vmm_src);
    } else if (alpha_ == 0.5 || alpha_ == 2.0) {
        h->fmul(vmm_src, p_all / T_m, alpha_);
    } else {
        h->fmul(vmm_src, vmm_src, table_val(alpha, table_value));
    }

    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // keep src for further computations
    h->mov(ZRegD(IDX(saved_src)), ZRegD(IDX(vmm_src)));

    h->fminnm(vmm_src, p_all / T_m, table_val(exp_ln_flt_max_f, table_value));
    h->fmaxnm(vmm_src, p_all / T_m, table_val(exp_ln_flt_min_f, table_value));
    h->mov(ZRegD(IDX(work)), ZRegD(IDX(vmm_src)));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->fmul(vmm_src, vmm_src, table_val(exp_log2ef, table_value));
    h->fadd(vmm_src, p_all / T_m, 0.5f);

    // tmp = floorf(fx)
    h->frintm(tmp, p_all / T_m, vmm_src);

    // keep vmm_src = fx for further computations
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(tmp)));

    // x = x - fx * ln2
    h->fmul(tmp, tmp, table_val(ln2f, table_value));
    h->fsub(work, work, tmp);
    // compute exponent polynomial
    table_val(exp_pol, polynomial, 4);
    h->fmad(polynomial, p_all / T_m, work, table_val(exp_pol, table_value, 3));
    h->fmad(polynomial, p_all / T_m, work, table_val(exp_pol, table_value, 2));
    h->fmad(polynomial, p_all / T_m, work, table_val(exp_pol, table_value, 1));
    h->fmad(polynomial, p_all / T_m, work, table_val(exp_pol, table_value, 0));
    h->fmad(polynomial, p_all / T_m, work, table_val(one, table_value));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    h->fsub(vmm_src, p_all / T_m, 1.);
    h->fneg(work, p_all / T_m, vmm_src);
    h->fcvtzs(work, p_all / T_m, work);
    // restore vmm_src to n
    h->fadd(vmm_src, p_all / T_m, 1.);

    const auto exponent_bias = 0x0000007f;
    h->add(work, exponent_bias);
    h->lsl(work, work, n_mantissa_bits);
    // calculate ln(1 + y)
    h->fmul(polynomial, p_all / T_m, 2.); // 2*exp(r)
    h->fadd(polynomial, polynomial,
            work); // 2^-(n-1) + 2*exp(r)
    h->fmul(polynomial, p_all / T_m,
            0.5); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    h->lsr(vmm_src, polynomial, n_mantissa_bits);
    h->scvtf(vmm_src, p_all / T_m, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->fsub(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six, table_value));

    // and with mask (to get 0.5 * mantissa)
    h->and_(ZRegD(IDX(polynomial)), ZRegD(IDX(polynomial)),
            ZRegD(IDX(table_val(soft_relu_mantissa_sign_mask, table_value))));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->orr(ZRegD(IDX(polynomial)), ZRegD(IDX(polynomial)),
            ZRegD(IDX(table_val(half, table_value))));
    // y  = y - 1
    h->fsub(polynomial, p_all / T_m, 1.);

    // compute log1p polynomial
    h->mov(ZRegD(IDX(work)),
            ZRegD(IDX(table_val(soft_relu_pol, table_value, 8))));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 7));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 6));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 5));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 4));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 3));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 2));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 1));
    h->fmad(work, p_all / T_m, polynomial,
            table_val(soft_relu_pol, table_value, 0));
    //calculate ln(2) * n
    h->fmul(vmm_src, vmm_src, table_val(ln2f, table_value));
    h->fadd(vmm_src, vmm_src, work);
    h->fadd(vmm_src, vmm_src, tmp);

    // Get the mask for src > max logf.
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(saved_src, table_val(exp_ln_flt_max_f, table_value),
            alg_kind::binary_gt);
    blend_with_mask(vmm_src, saved_src);
    if (alpha_ == 1.f) { // standard soft_relu case
        // Skip an instruction.
    } else if (alpha_ == -1) { // logsigmoid case
        h->fneg(vmm_src, p_all / T_m, vmm_src);
    } else if (alpha_ == 0.5 || alpha_ == 2.0) {
        h->fmul(vmm_src, p_all / T_m, 1.f / alpha_);
    } else { // General case.
        h->fdiv(vmm_src, p_all / T_m, table_val(alpha, table_value));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector_fwd(
        const TRegS &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic being a symmetric function.

    // Store the original sign and make x negative
    constexpr unsigned sign_mask = 0x80000000;
    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.);
    h->orr(vmm_src, sign_mask);

    // Compute exponent
    exp_compute_vector_fwd(vmm_src, min_input_, 0.f);

    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto denominator = reg_pool_.allocScoped<TRegS>();

    // (exp(x) + 1)
    h->fadd(denominator, vmm_src, table_val(one, table_value));
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_src, p_all, denominator);

    // Now we have to apply the "symmetry" based on original sign
    h->fsub(denominator, table_value, vmm_src);

    // Combine using mask
    blend_with_mask(vmm_src, denominator);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::swish_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha_log2e,
        const TRegS &exponent_control, const TRegS &polynomial_c1,
        const TRegS &polynomial_c2) {
    auto v_scale = reg_pool_.allocScoped<TRegS>();
    auto v_neg_r = reg_pool_.allocScoped<TRegS>();
    auto v_poly = reg_pool_.allocScoped<TRegS>();
    auto v_tmp = reg_pool_.allocScoped<TRegS>();

    // Convert exp(alpha*x) to base 2. For a=alpha*x*log2(e)=k+r:
    //   swish(x) = x*2^k / (2^k + 2^(-r)).
    // FEXPA supplies 2^k and a quadratic approximates the residual 2^(-r).
    h->movprfx(ZReg(IDX(v_neg_r)), ZReg(IDX(vmm_src)));
    h->fmul(v_neg_r, p_all / T_m, alpha_log2e);
    // Bound the exponential argument and keep the encoded exponent safe.
    h->fmaxnm(v_neg_r, p_all / T_m, table_val(swish_polynomial_exp_min, v_tmp));
    h->fminnm(v_neg_r, p_all / T_m, table_val(swish_polynomial_exp_max, v_tmp));

    // exponent_control rounds k to FEXPA's grid: k=(control+a)-control,
    // leaving -r=k-a for the polynomial.
    h->movprfx(ZReg(IDX(v_scale)), ZReg(IDX(exponent_control)));
    h->fadd(v_scale, p_all / T_m, v_neg_r);
    h->fsub(v_poly, v_scale, exponent_control);
    // FEXPA consumes the encoded control+a bits and produces 2^k.
    h->fexpa(v_scale, v_scale);
    h->fsub(v_neg_r, v_poly, v_neg_r);

    // Form x*2^k and approximate 2^(-r) as 1+(-r)*(c1+c2*(-r)).
    h->fmul(vmm_src, vmm_src, v_scale);
    h->movprfx(ZReg(IDX(v_poly)), ZReg(IDX(polynomial_c1)));
    h->fmla(v_poly, p_all / T_m, v_neg_r, polynomial_c2);
    h->fmov(v_tmp, 1.);
    h->fmla(v_tmp, p_all / T_m, v_neg_r, v_poly);
    h->fadd(v_neg_r, v_scale, v_tmp);
    // x*2^k / (2^k + 2^(-r)).
    h->fdiv(vmm_src, p_all / T_m, v_neg_r);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector_fwd(
        const TRegS &vmm_src) {

    const auto &t0 = vmm_src;
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    auto t3 = reg_pool_.allocScoped<TRegS>();
    auto t4 = reg_pool_.allocScoped<TRegS>();
    auto p_tmp0 = reg_pool_.allocScoped<PReg>();
    auto x_tmp0 = reg_pool_.allocScoped<XReg>();
    const auto &mask = p_tmp0.get().s;
    const WReg wt0(x_tmp0.getIdx());
    auto set_imm = [&](const ZRegS &dst, uint32_t imm) {
        h->mov_imm(wt0, imm);
        h->cpy(dst, p_all, wt0);
        return dst;
    };
    Label tbl1L, tbl2L, exitL;
    const size_t tblL = 5;
    const size_t tblN = 1 << tblL;
    union fi {
        float f;
        uint32_t i;
    };
    //h->brk(0);
    h->mov(t4, p_all, t0);
    h->fmul(t0, t0, set_imm(table_value, float2int(std::sqrt(2))));
    set_imm(t3, 127 << 23);
    h->sub(t1, t0, t3);
    h->asr(t1, t1, 23); // n
    h->scvtf(t1, p_all, t1); // int -> float
    h->and_(t0, p_all, set_imm(table_value, 0x7fffff));
    h->asr(t2, t0, 23 - tblL); // d
    h->lsl(t2, t2, 2); // d *= 4
    h->orr(t0, p_all, t3); // y
    h->fmul(t0, t0, set_imm(table_value, float2int(1 / std::sqrt(2))));
    h->adr(x_tmp0, tbl1L);
    h->ld1w(t3, p_all, ptr(x_tmp0, t2, SXTW)); // f
    h->fcpy(table_value, p_all, 1.0f);
    h->fnmsb(t0, p_all, t3, table_value); // y = y * f - 1
    h->adr(x_tmp0, tbl2L);
    h->ld1w(t2, p_all, ptr(x_tmp0, t2, SXTW)); // h
    h->fsub(t3, t4, table_value); // x-1
    set_imm(table_value, float2int(1.0 / 32));
    h->facge(mask, p_all, table_value, t3); // 1/32 >= abs(x-1)
    h->mov(t0, mask, t3);
    h->eor(t2, mask, t2);
    h->fnmsb(t1, p_all, set_imm(table_value, float2int(std::log(2))),
            t2); // x = n * log2 - h
    set_imm(table_value, float2int(0.333332205));
    h->movprfx(t2, p_all, table_value);
    set_imm(table_value, float2int(-0.499999851));
    h->fmad(t2, p_all, t0, table_value); // f
    h->fcpy(table_value, p_all, 1.0f);
    h->fmad(t2, p_all, t0, table_value); // f * y + 1
    h->fmad(t0, p_all, t2, t1); // y * f + x
    // check nan/inf
    h->fcmlt(mask, p_all, t4, 0.0f); // neg
    h->mov(wt0, 0x7fc00000); // qnan
    h->cpy(t0, mask, wt0);
    h->fcmeq(mask, p_all, t4, 0.0f); // = 0
    h->mov(wt0, 0xff800000); // -Inf
    h->cpy(t0, mask, wt0);
    h->mov(wt0, 0x7f800000); // Inf
    h->dup(t1, wt0);
    h->fcmeq(mask, p_all, t4, t1);
    h->sel(t0, mask, t1, t0);
    h->b(exitL);

    h->L(tbl1L);
    // Caution: This pointer can be invalidated if the underlying
    // Xbyak_aarch64::CodeArray is reallocated. We take note of the current size
    // so we can jump back to the appropriate spot in the reallocated CodeArray
    // if needed.
    const float *tbl1Addr = (const float *)h->getCurr();
    auto tbl_max_size = h->getMaxSize();
    const auto tbl_orig_size = h->getSize();
    for (size_t i = 0; i < tblN; ++i) {
        fi fi;
        fi.i = (127 << 23) | (i << (23 - tblL));
        fi.f = std::sqrt(2) / fi.f;
        h->dd(fi.i);
    }
    h->L(tbl2L);
    for (size_t i = 0; i < tblN; ++i) {
        fi fi;

        // If the table has been reallocated due to a previous instruction then
        // jump back to the corresponding location in the new table. This will
        // essentially be the current position minus the number of instructions
        // that have been added since we initialised the original pointer.
        if (tbl_max_size != h->getMaxSize()) {
            constexpr int instr_size = sizeof(uint32_t);
            auto tbl_curr_size = h->getSize();
            auto correction = (tbl_curr_size - tbl_orig_size) / instr_size;

            tbl1Addr = (const float *)h->getCurr() - correction;
            tbl_max_size = h->getMaxSize();
        }

        fi.f = std::log(tbl1Addr[i]);
        h->dd(fi.i);
    }

    h->L(exitL);
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<
        isa>::gelu_erf_minimax_approx_compute_vector_fwd(const TRegS &vmm_src) {
    if (vlen != 64) { // TODO: change this condition based on cpu id.
        return;
    }

    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto polynomial = reg_pool_.allocScoped<TRegS>();
    auto src_pos = reg_pool_.allocScoped<TRegS>();
    auto indices = reg_pool_.allocScoped<TRegS>();
    auto work = reg_pool_.allocScoped<TRegS>();

    auto gather_coefficient
            = [&](TRegS vmm_coeff, int coeff_idx, TRegS polynomial_idx) {
        auto table_addr = reg_pool_.allocScoped<XReg>();
        // we actually have 25 polynomials but pad to avoid unaligned accesses/
        int gelu_erf_n_polynomials = 32;
        h->add_imm(table_addr, x_table,
                table_off(gelu_erf_minimax_pol,
                        coeff_idx * gelu_erf_n_polynomials),
                table_addr);
        h->ld1w(vmm_coeff, p_all / T_z, ptr(table_addr, polynomial_idx, SXTW));
    };

    // we use the erf function symmetry erf(-x) = -erf(x)
    // So we make x positive, we will reapply the sign after erf evaluation
    h->fabs(src_pos, p_all / T_z, vmm_src);

    // Compute indices for table lookup
    h->add(indices, src_pos, table_val(gelu_erf_idx_bias, table_value, 0));

    // An arithmetic shift is needed to properly map denormals to
    // their polynomial. we shift by 21 as we use 2 bits of mantissa
    // for indexing.
    h->asr(indices, indices, 21);

    // Apply special rules
    h->smax(indices, p_all / T_z, table_val(gelu_erf_one, table_value));
    h->smin(indices, p_all / T_z, table_val(gelu_erf_twenty_four, table_value));

    // We have to check
    //     index = x_pos > rbound ? 23 : index;
    // for erf to return -1/1 when we should.
    h->fcmlt(p_mask.s, p_all / T_z, src_pos,
            table_val(gelu_erf_rbound, table_value));
    h->sel(indices, p_mask, indices,
            table_val(gelu_erf_twenty_three, table_value));

    // Adjusting indices
    h->mul(indices, sizeof(float));

    // Evaluate the polynomial
    gather_coefficient(polynomial, 5, indices);
    for (int deg = 4; deg >= 0; --deg) {
        gather_coefficient(work, deg, indices);
        h->fmad(polynomial, p_all / T_z, src_pos, work);
    }

    // Set the sign of polynomial properly
    h->mov(ZRegD(IDX(work)), ZRegD(IDX(vmm_src)));
    h->and_(ZRegD(IDX(work)), ZRegD(IDX(work)),
            ZRegD(IDX(table_val(sign_mask, table_value))));
    h->eor(ZRegD(IDX(polynomial)), p_all / T_z, ZRegD(IDX(work)));

    // Compute the final output
    h->fadd(polynomial, polynomial, table_val(one, table_value));
    h->fmul(vmm_src, p_all / T_z, polynomial);
    h->fmul(vmm_src, vmm_src, table_val(half, table_value));
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_erf_compute_vector_fwd(
        const TRegS &vmm_src) {
    if (vlen == 64) {
        gelu_erf_minimax_approx_compute_vector_fwd(vmm_src);
        return;
    }

    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto aux0 = reg_pool_.allocScoped<TRegS>();
    auto aux1 = reg_pool_.allocScoped<TRegS>();
    auto aux2 = reg_pool_.allocScoped<TRegS>();
    auto aux3 = reg_pool_.allocScoped<TRegS>();
    auto aux4 = reg_pool_.allocScoped<TRegS>();
    auto aux5 = reg_pool_.allocScoped<TRegS>();

    auto p_tmp0 = reg_pool_.allocScoped<PReg>();

    // x = s / sqrt(2)
    h->fmul(aux5, vmm_src, table_val(gelu_erf_one_over_sqrt_two, table_value));

    // abs(x)
    h->fabs(aux1, p_all / T_m, aux5);

    // LUT-based ERF based on the verfq_f32 implementation from Arm Compute Library (ACL)
    // Maximum error: 1.93 ULP
    //
    // erf(x) for x in [0, 3.9375] is approxiated as follows:
    //
    //   erf(x) = erf(r) + scale(r) * d * (1 - r * d - 1/3 * d^2)
    //
    // where:
    //   r = round(x * 128) / 128
    //   d = x - r
    //
    // erf(r) and scale(r) are stored in a 513-entry lookup table.
    // The LUT covers the range from 0 to 4 with the step of 1/128.
    //
    // Special cases:
    //   erf(x) =  1 for x >  3.9375
    //   erf(x) = -1 for x < -3.9375

    // Track saturation mask from |x| and clamp x used for LUT indexing.
    table_val(gelu_erf_lut_max, aux0);
    h->facgt(p_tmp0.get().s, p_all / T_z, aux5, aux0);
    h->fmin(aux1, p_all / T_m, aux0);

    // Find LUT indices by quantizing the LUT input to the 1/128 grid.
    // Adding 2^16 moves the sum into the [2^16, 2^17) f32 range, where the ULP is
    // 2^(16 - 23) = 1/128, so the add rounds to the nearest 1/128 step.
    // The floating-point subtraction recovers the quantized value, and the integer
    // subtraction of the same bit pattern yields the corresponding LUT index.
    h->fadd(aux2, aux1, table_val(gelu_erf_lut_bias, table_value));
    h->fsub(aux3, aux2, table_value); // r
    h->sub(aux4, aux2,
            table_value); // raw index
    h->umin(aux4, p_all / T_m,
            table_val(gelu_erf_lut_max_index, table_value)); // index

    // Lookup erf(r) and scale(r) from separate LUTs.
    {
        auto table_addr = reg_pool_.allocScoped<XReg>();
        h->add_imm(
                table_addr, x_table, table_off(gelu_erf_lut_erf), table_addr);
        h->ld1w(aux0, p_all / T_z, ptr(table_addr, aux4, UXTW, 2)); // erf_r
        h->add_imm(
                table_addr, x_table, table_off(gelu_erf_lut_scale), table_addr);
        h->ld1w(aux2, p_all / T_z, ptr(table_addr, aux4, UXTW, 2)); // scale_r
    }

    // erf(x) ~= erf(r) + scale(r) * d * (1 - r * d - 1/3 * d^2).
    h->fsub(aux5, aux1, aux3); // d
    h->fmul(aux4, aux5, aux5); // d^2
    h->fmla(aux3, p_all / T_m, table_val(gelu_erf_lut_third, table_value),
            aux5); // t0 = r + 1/3 * d.
    h->fmls(aux5, p_all / T_m, aux4,
            aux3); // t1 = d - d^2 * t0
    h->fmla(aux0, p_all / T_m, aux2,
            aux5); // erf(|x|)

    // Clamp erf(|x|) for |x| > 3.9375 to 1.
    h->fmov(aux0, p_tmp0 / T_m, 1.);

    // Restore sign using original s.
    h->mov(ZRegD(IDX(aux4)), ZRegD(IDX(vmm_src)));
    h->and_(aux4, 0x80000000);
    h->orr(ZRegD(IDX(aux0)), ZRegD(IDX(aux0)), ZRegD(IDX(aux4)));

    // GELU = 0.5 * s * (1 + erf)
    h->fmul(vmm_src, p_all / T_m, 0.5f);
    h->fmad(vmm_src, p_all / T_m, aux0, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.);
    h->mov(ZRegD(vmm_src.getIdx()), ZRegD(alpha.getIdx()));
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::elu_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        auto table_value = reg_pool_.allocScoped<TRegS>();
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(
                vmm_src, table_val(one, table_value), alg_kind::binary_gt);
        // R * alpha, then blend with 1.f
        h->fmul(vmm_src, vmm_src, table_val(alpha, table_value));
        blend_with_mask(vmm_src, table_val(one, table_value));
    } else {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        // get mask of `d` > 0
        compute_cmp_mask(
                vmm_src, table_val(zero, table_value), alg_kind::binary_gt);
        // R = `d` + alpha, then blend with 1.f
        h->fadd(vmm_src, vmm_src, table_val(alpha, table_value));
        blend_with_mask(vmm_src, table_val(one, table_value));
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto result = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(IDX(result)), ZRegD(IDX(table_val(one, table_value))));

    h->fmls(result, p_all / T_m, vmm_src, vmm_src);

    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(result)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_tanh_compute_vector_bwd(
        const TRegS &vmm_src) {
    auto g2 = reg_pool_.allocScoped<TRegS>();

    {
        auto x = reg_pool_.allocScoped<TRegS>();
        auto fitting = reg_pool_.allocScoped<TRegS>();
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->mov(ZRegD(IDX(x)), ZRegD(IDX(vmm_src)));

        // G1(x) = sqrt(2/pi) * x * (1 + fitting_const * x^2)
        // G2(x) = sqrt(2/pi) * x * (1 + 3 * fitting_const * x^2)
        h->fmul(vmm_src, vmm_src, vmm_src);
        h->mov(ZRegD(IDX(g2)),
                ZRegD(IDX(table_val(
                        gelu_tanh_fitting_const_times_three, table_value))));
        h->fmad(g2, p_all / T_m, vmm_src, table_val(one, table_value));

        h->mov(ZRegD(IDX(fitting)),
                ZRegD(IDX(table_val(gelu_tanh_fitting_const, table_value))));
        h->fmad(vmm_src, p_all / T_m, fitting, table_val(one, table_value));
        h->fmul(x, x, table_val(gelu_tanh_sqrt_two_over_pi, table_value));
        h->fmul(vmm_src, vmm_src, x);
        h->fmul(g2, g2, x);
    }

    // T = tanh(G1(x)). The pool keeps G2 disjoint from tanh scratch.
    tanh_compute_vector_fwd(vmm_src);

    auto table_value = reg_pool_.allocScoped<TRegS>();
    // 0.5 * (1 + T) * (1 + G2 * (1 - T)):
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    h->fmls(g2, p_all / T_m, g2, vmm_src);
    // 2) Q = 1 + T
    h->fadd(vmm_src, vmm_src, table_val(one, table_value));
    // 3) result = Q * (1 + R) = Q + Q * R
    h->fmla(vmm_src, p_all / T_m, vmm_src, g2);
    h->fmul(vmm_src, vmm_src, table_val(half, table_value));
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::square_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 2 * s
    h->fmul(vmm_src, p_all / T_m, 2.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::abs_compute_vector_bwd(
        const TRegS &vmm_src) {
    auto table_value = reg_pool_.allocScoped<TRegS>();
    // replace positive values with 1.f
    compute_cmp_mask(
            vmm_src, table_val(zero, table_value), alg_kind::binary_gt);
    blend_with_mask(vmm_src, table_val(one, table_value));
    // replace negative values with -1.f
    compute_cmp_mask(
            vmm_src, table_val(zero, table_value), alg_kind::binary_lt);
    blend_with_mask(vmm_src, table_val(minus_one, table_value));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::sqrt_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(ZReg(IDX(vmm_src)));
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto result = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(IDX(result)), ZRegD(IDX(table_val(half, table_value))));
    h->fdiv(result, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(result)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::linear_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(alpha)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::soft_relu_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    if (alpha_ == 1.f) {
        // Skip an instruction.
    } else if (alpha_ == 0.5f || alpha_ == 2.f)
        h->fmul(vmm_src, p_all / T_m, alpha_);
    else
        h->fmul(vmm_src, vmm_src, alpha);
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::mish_compute_vector_bwd(
        const TRegS &vmm_src) {
    auto saved_x = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(saved_x.getIdx()), ZRegD(vmm_src.getIdx())); // saved_x = x

    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->fminnm(vmm_src, p_all / T_m,
                table_val(bwd_mish_max_x_for_equation_f, table_value.get()));
    }
    exp_compute_vector_fwd(vmm_src);
    auto exp_x = reg_pool_.allocScoped<TRegS>();
    auto tmp = reg_pool_.allocScoped<TRegS>();
    auto two = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(exp_x.getIdx()), ZRegD(vmm_src.getIdx()));

    // e^3x + 4*e^2x
    h->fmul(vmm_src, vmm_src, vmm_src); // e^2x
    h->mov(ZRegD(tmp.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fmul(tmp, p_all / T_m, 2.);
    h->fmul(tmp, p_all / T_m, 2.); // 4*e^2x
    h->fmad(vmm_src, p_all / T_m, exp_x, tmp);

    // e^3x + 4*e^2x + 4*e^x*(x+1.5)
    h->fadd(saved_x, p_all / T_m, 1.); // saved_x = x + 1
    h->mov(ZRegD(tmp.getIdx()), ZRegD(saved_x.getIdx()));
    h->fadd(tmp, p_all / T_m, 0.5f);
    h->fmul(tmp, p_all / T_m, 2.);
    h->fmul(tmp, p_all / T_m, 2.);
    h->fmla(vmm_src, p_all / T_m, tmp, exp_x);

    // omega = e^3x + 4*e^2x + 4*e^x*(x+1.5) + 4*(x+1)
    h->fmul(saved_x, p_all / T_m, 2.);
    /* Do not use 2.f, which is a float constant,
       but 2., which is a double constant. */
    h->fmov(two, 2.);
    h->fmla(vmm_src, p_all / T_m, saved_x, two);

    // delta = (e^x+1)^2 + 1
    h->mov(ZRegD(tmp.getIdx()), ZRegD(exp_x.getIdx()));
    h->fadd(tmp, p_all / T_m, 1.);
    h->fmul(tmp, tmp, tmp);
    h->fadd(tmp, p_all / T_m, 1.);
    h->fmul(tmp, tmp, tmp);

    // e^x * omega / delta^2
    h->fmul(vmm_src, vmm_src, exp_x);
    h->fdiv(vmm_src, p_all / T_m, tmp);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    auto one_minus_dst = reg_pool_.allocScoped<TRegS>();
    h->fmov(one_minus_dst, 1.);
    h->fsub(one_minus_dst, one_minus_dst, vmm_src);
    h->fmul(vmm_src, vmm_src, one_minus_dst);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector_bwd(
        const TRegS &vmm_src) {
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::swish_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = alpha * s
    {
        auto alpha_reg = reg_pool_.allocScoped<TRegS>();
        h->fmul(vmm_src, vmm_src, table_val(alpha, alpha_reg.get()));
    }
    auto r = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(IDX(r)), ZRegD(IDX(vmm_src)));

    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);

    // compute Q * (1 + R * (1 - Q))
    // T = R * (1 - Q) = R - R * Q
    h->fmls(r, p_all / T_m, r, vmm_src);

    // Q * (1 + T) = Q + Q * T
    h->fmla(vmm_src, p_all / T_m, vmm_src, r);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector_bwd(
        const TRegS &vmm_src) {
    // res = 1 / s
    /* Do not use 1.f, which is a float constant,
       but 1., which is a double constant. */
    auto result = reg_pool_.allocScoped<TRegS>();
    h->fmov(result, 1.);
    h->fdiv(result, p_all, vmm_src);
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(result)));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &lower, const TRegS &upper) {
    using namespace alg_kind;
    // get mask of values > beta and blend with 0.f
    if (alg_ == eltwise_clip)
        h->fcmle(p_mask.s, p_all / T_z, vmm_src, upper);
    else
        h->fcmlt(p_mask.s, p_all / T_z, vmm_src, upper);
    // get mask of values <= alpha and blend with 0.f
    h->fcmgt(p_mask.s, p_mask / T_z, vmm_src, lower);

    h->mov(vmm_src, 0);
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_erf_compute_vector_bwd(
        const TRegS &vmm_src) {
    // R = s / sqrt(2)
    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->fmul(vmm_src, vmm_src,
                table_val(gelu_erf_one_over_sqrt_two, table_value.get()));
    }

    auto r = reg_pool_.allocScoped<TRegS>();
    h->mov(ZRegD(IDX(r)), ZRegD(IDX(vmm_src)));

    // Q = exp(-R*R)
    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->fmul(vmm_src, vmm_src, vmm_src);
        h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
                ZRegD(IDX(table_val(sign_mask, table_value))));
    }
    exp_compute_vector_fwd(vmm_src);

    auto result = reg_pool_.allocScoped<TRegS>();
    auto sign = reg_pool_.allocScoped<TRegS>();
    auto denominator = reg_pool_.allocScoped<TRegS>();
    auto w = reg_pool_.allocScoped<TRegS>();
    auto table_value = reg_pool_.allocScoped<TRegS>();

    // T = R / sqrt(pi) * Q
    h->fmul(result, r, table_val(gelu_erf_one_over_sqrt_pi, table_value));
    h->fmul(result, result, vmm_src);

    // -Q
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)),
            ZRegD(IDX(table_val(sign_mask, table_value))));

    // Preserve the sign, then reuse R for abs(R).
    h->mov(ZRegD(IDX(sign)), ZRegD(IDX(r)));
    h->and_(ZRegD(IDX(sign)), ZRegD(IDX(sign)),
            ZRegD(IDX(table_val(sign_mask, table_value))));
    abs_compute_vector_fwd(ZReg(IDX(r)));

    // W = 1 / (p * abs(R) + 1)
    h->mov(ZRegD(IDX(denominator)),
            ZRegD(IDX(table_val(gelu_erf_approx_const, table_value))));
    h->mov(ZRegD(IDX(w)), ZRegD(IDX(table_val(one, table_value))));
    h->fmad(denominator, p_all / T_m, r, w);
    h->fdiv(w, p_all, denominator);

    // Q * W
    h->fmul(vmm_src, vmm_src, w);

    // Compute polynomial r.
    h->mov(ZRegD(IDX(r)), ZRegD(IDX(table_val(gelu_erf_pol, table_value, 4))));
    h->fmad(r, p_all / T_m, w, table_val(gelu_erf_pol, table_value, 3));
    h->fmad(r, p_all / T_m, w, table_val(gelu_erf_pol, table_value, 2));
    h->fmad(r, p_all / T_m, w, table_val(gelu_erf_pol, table_value, 1));
    h->fmad(r, p_all / T_m, w, table_val(gelu_erf_pol, table_value, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->fmad(vmm_src, p_all / T_m, r, table_val(one, table_value));
    h->eor(ZRegD(IDX(vmm_src)), ZRegD(IDX(vmm_src)), ZRegD(IDX(sign)));

    // res = T + 0.5 + 0.5 * erf
    h->fadd(result, result, table_val(half, table_value));
    h->fmla(result, p_all / T_m, vmm_src, table_val(half, table_value));
    h->mov(ZRegD(IDX(vmm_src)), ZRegD(IDX(result)));
}
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardswish_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta,
        const TRegS &one) {
    auto affine = reg_pool_.allocScoped<TRegS>();
    // Get mask for 0 < alpha * x + beta < 1
    h->mov(ZRegD(affine.getIdx()), ZRegD(vmm_src.getIdx()));
    h->fmul(affine, affine, alpha);
    h->fadd(affine, affine, beta);
    // Form a derivative value
    h->fmul(vmm_src, vmm_src, alpha);
    h->fadd(vmm_src, vmm_src, affine);

    h->fcmle(p_mask.s, p_all / T_z, affine, 0.);
    h->mov(vmm_src, p_mask / T_m, 0);
    h->fcmge(p_mask.s, p_all / T_z, affine, one);
    h->fmov(vmm_src, p_mask / T_m, 1.);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardsigmoid_compute_vector_bwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta,
        const TRegS &one) {
    // Get mask for 0 < alpha * x + beta < 1
    // Zero rest values.
    h->fmul(vmm_src, vmm_src, alpha);
    h->fadd(vmm_src, vmm_src, beta);

    h->fcmgt(p_mask.s, p_all / T_z, vmm_src, 0.);
    h->fcmlt(p_mask.s, p_mask / T_z, vmm_src, one);

    h->mov(vmm_src, 0);
    h->fmov(vmm_src, p_mask / T_m, 1.);
    h->fmul(vmm_src, vmm_src, alpha);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::round_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->frintn(vmm_src, p_all / T_m, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_body(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    // Apply work either once per algorithm or once per destination VMM.
    const auto for_each_idx = [&](const auto &compute) {
        std::for_each(vmm_idxs.begin(), vmm_idxs.end(), compute);
    };
    // Returning the scoped register extends its lifetime at the call site.
    const auto load_parameter = [&](key_t key) {
        auto reg = reg_pool_.allocScoped<TRegS>();
        table_val(key, reg.get());
        return reg;
    };

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
                if (alpha_ == 0.f) {
                    for_each_idx([&](size_t idx) {
                        relu_zero_ns_compute_vector_fwd(TReg(idx));
                    });
                } else {
                    auto alpha = load_parameter(key_t::alpha);
                    for_each_idx([&](size_t idx) {
                        relu_compute_vector_fwd(TRegS(idx), alpha.get());
                    });
                }
                break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: {
                auto alpha = load_parameter(key_t::alpha);
                for_each_idx([&](size_t idx) {
                    elu_compute_vector_fwd(TRegS(idx), alpha.get());
                });
                break;
            }
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
                for_each_idx([&](size_t idx) {
                    tanh_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_square:
                for_each_idx([&](size_t idx) {
                    square_compute_vector_fwd(TReg(idx));
                });
                break;
            case eltwise_abs:
                for_each_idx(
                        [&](size_t idx) { abs_compute_vector_fwd(TReg(idx)); });
                break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
                for_each_idx([&](size_t idx) {
                    sqrt_compute_vector_fwd(TReg(idx));
                });
                break;
            case eltwise_swish: {
                auto alpha_log2e = load_parameter(key_t::swish_alpha_log2e);
                if (isa == sve) {
                    auto shift = load_parameter(key_t::swish_fexpa_shift);
                    auto polynomial_c1
                            = load_parameter(key_t::swish_polynomial_exp_c1);
                    auto polynomial_c2
                            = load_parameter(key_t::swish_polynomial_exp_c2);
                    for_each_idx([&](size_t idx) {
                        swish_compute_vector_fwd(TRegS(idx), alpha_log2e.get(),
                                shift.get(), polynomial_c1.get(),
                                polynomial_c2.get());
                    });
                } else {
                    auto exponent_bias = load_parameter(key_t::exponent_bias);
                    for_each_idx([&](size_t idx) {
                        // Unused TRegS will error on use
                        swish_compute_vector_fwd(TRegS(idx), alpha_log2e.get(),
                                exponent_bias.get(), TRegS(99), TRegS(99));
                    });
                }
                break;
            }
            case eltwise_linear: {
                auto alpha = load_parameter(key_t::alpha);
                auto beta = load_parameter(key_t::beta);
                for_each_idx([&](size_t idx) {
                    linear_compute_vector_fwd(
                            TRegS(idx), alpha.get(), beta.get());
                });
                break;
            }
            case eltwise_soft_relu:
                for_each_idx([&](size_t idx) {
                    soft_relu_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_mish:
                for_each_idx([&](size_t idx) {
                    mish_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
                for_each_idx([&](size_t idx) {
                    logistic_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
                for_each_idx([&](size_t idx) {
                    exp_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_gelu_tanh:
                for_each_idx([&](size_t idx) {
                    gelu_tanh_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_log:
                for_each_idx([&](size_t idx) {
                    log_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: {
                auto lower = load_parameter(key_t::alpha);
                auto upper = load_parameter(key_t::beta);
                for_each_idx([&](size_t idx) {
                    clip_compute_vector_fwd(
                            TReg(idx), lower.get(), upper.get());
                });
                break;
            }
            case eltwise_gelu_erf:
                for_each_idx([&](size_t idx) {
                    gelu_erf_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_round:
                for_each_idx([&](size_t idx) {
                    round_compute_vector_fwd(TRegS(idx));
                });
                break;
            case eltwise_hardswish:
            case eltwise_hardsigmoid: {
                auto alpha = load_parameter(key_t::alpha);
                auto beta = load_parameter(key_t::beta);
                for_each_idx([&](size_t idx) {
                    if (alg_ == eltwise_hardswish)
                        hardswish_compute_vector_fwd(
                                TRegS(idx), alpha.get(), beta.get());
                    else
                        hardsigmoid_compute_vector_fwd(
                                TRegS(idx), alpha.get(), beta.get());
                });
                break;
            }
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: {
                auto alpha = load_parameter(key_t::alpha);
                for_each_idx([&](size_t idx) {
                    relu_compute_vector_bwd(TRegS(idx), alpha.get());
                });
                break;
            }
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu:
                for_each_idx([&](size_t idx) {
                    elu_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh:
                for_each_idx([&](size_t idx) {
                    tanh_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_square:
                for_each_idx([&](size_t idx) {
                    square_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_abs:
                for_each_idx([&](size_t idx) {
                    abs_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt:
                for_each_idx([&](size_t idx) {
                    sqrt_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_linear: {
                auto alpha = load_parameter(key_t::alpha);
                for_each_idx([&](size_t idx) {
                    linear_compute_vector_bwd(TRegS(idx), alpha.get());
                });
                break;
            }
            case eltwise_soft_relu: {
                auto alpha = load_parameter(key_t::alpha);
                for_each_idx([&](size_t idx) {
                    soft_relu_compute_vector_bwd(TRegS(idx), alpha.get());
                });
                break;
            }
            case eltwise_mish:
                for_each_idx([&](size_t idx) {
                    mish_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic:
                for_each_idx([&](size_t idx) {
                    logistic_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp:
                for_each_idx([&](size_t idx) {
                    exp_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_gelu_tanh:
                for_each_idx([&](size_t idx) {
                    gelu_tanh_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_swish:
                for_each_idx([&](size_t idx) {
                    swish_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_log:
                for_each_idx([&](size_t idx) {
                    log_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_clip:
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: {
                auto lower = load_parameter(key_t::alpha);
                auto upper = load_parameter(key_t::beta);
                for_each_idx([&](size_t idx) {
                    clip_compute_vector_bwd(
                            TRegS(idx), lower.get(), upper.get());
                });
                break;
            }
            case eltwise_gelu_erf:
                for_each_idx([&](size_t idx) {
                    gelu_erf_compute_vector_bwd(TRegS(idx));
                });
                break;
            case eltwise_hardswish:
            case eltwise_hardsigmoid: {
                auto alpha = load_parameter(key_t::alpha);
                auto beta = load_parameter(key_t::beta);
                auto one = reg_pool_.allocScoped<TRegS>();
                h->fmov(one.get(), 1.);
                for_each_idx([&](size_t idx) {
                    if (alg_ == eltwise_hardswish)
                        hardswish_compute_vector_bwd(
                                TRegS(idx), alpha.get(), beta.get(), one.get());
                    else
                        hardsigmoid_compute_vector_bwd(
                                TRegS(idx), alpha.get(), beta.get(), one.get());
                });
                break;
            }
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    if (scale_ != 1.f) {
        auto scale = load_parameter(key_t::scale);
        for_each_idx([&](size_t idx) {
            h->fmul(TRegS(idx), TRegS(idx), scale.get());
        });
    }
}

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_t<isa>::scratch_vmm_count() const {
    constexpr size_t unmeasured = static_cast<size_t>(-1);
    if (scratch_vmm_count_ != unmeasured) return scratch_vmm_count_;

    eltwise_probe_generator_t probe_host;
    auto &probe_pool = probe_host.probe_reg_pool();
    const XReg probe_table(15);
    const PReg probe_mask(1);
    constexpr int probe_result_idx = 0;

    // These registers model the caller-owned state used by the injector. They
    // are reserved rather than allocated so the recorded vector peak contains
    // scratch only.
    probe_pool.reserve_gp(probe_table.getIdx());
    probe_pool.reserve_pred(probe_mask.getIdx());
    probe_pool.reserve_vec(probe_result_idx);

    jit_uni_eltwise_injector_t<isa> probe(&probe_host, probe_pool, alg_, alpha_,
            beta_, scale_, probe_table, probe_mask, is_fwd_, use_dst_, d_type_);
    probe.min_input_ = min_input_;
    probe.max_input_ = max_input_;
    probe.compute_vector_range_no_spill({probe_result_idx});
    scratch_vmm_count_ = probe_pool.get_all_used_vec().size();
    return scratch_vmm_count_;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    if (vmm_idxs.empty()) return;

    assert(*vmm_idxs.rbegin() < vecs_count);
    assert(reg_pool_.pred_idx_in_use(p_mask.getIdx()));
    injector_utils::compute_vector_range_with_spill<isa>(h, reg_pool_, vmm_idxs,
            scratch_vmm_count(),
            [&](const injector_utils::vmm_index_set_t &idxs) {
        compute_vector_range_no_spill(idxs);
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range_no_spill(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_body(vmm_idxs);
}

namespace {
// Downcast f32 -> f16, then pack that 16-bit pattern twice into a uint32: [f16 | f16]
inline uint32_t pack_f16_twice_from_f32(float s) {
    float16_t h = float16_t(s);
    uint16_t h_bits = h.raw;
    return (uint32_t(h_bits) << 16) | uint32_t(h_bits);
}
} // anonymous namespace

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}}, // p5 = 0.00828929059f
            {exp_pol_asimd, {0x3c072010, true}}, // c0 =  0x1.0e4020p-7
            {exp_pol_asimd, {0x3d2b9f17, true}}, // c1 =  0x1.573e2ep-5
            {exp_pol_asimd, {0x3e2aaf33, true}}, // c2 =  0x1.555e66p-3
            {exp_pol_asimd, {0x3efffedb, true}}, // c3 =  0x1.fffdb6p-2
            {exp_pol_asimd, {0x3f7ffff6, true}}, // c4 =  0x1.ffffecp-1
    };
    // exp(x) constants2
    static const table_t exp_consts2 {
            {exp_coeff1, {0x3f31721c, true}},
            {exp_coeff2, {0x3e772df2, true}},
            {exp_not_mask17, {~((1u << 17) - 1), true}},
            {exp_exponent_bias,
                    {0x3f800000, true}}, // 2^7 - 1 shifted to exponent position
            {exp_neg_ln2_hi, {0xbf317200, true}}, // -0x1.62e400p-1f (hi)
            {exp_neg_ln2_lo, {0xb5bfbe8e, true}}, // -0x1.7f7d1cp-20f (lo)
            {exp_special_bound, {0x42fc0000, true}}, // 126.0f
            {exp_scale_thresh, {0x43400000, true}}, // 192.0f
            {exp_special_offset, {0x82000000, true}},
            {exp_special_bias, {0x7f000000, true}},
    };

    static const table_t sve_swish_polynomial_consts {
            {swish_fexpa_shift, {0x48001fc0, true}}, // 131199.f
            {swish_polynomial_exp_c1, {0x3f3174c1, true}}, // 0.6931878021f
            {swish_polynomial_exp_c2, {0x3e760066, true}}, // 0.2402358998f
            {swish_polynomial_exp_min, {0xc238aa3b, true}}, // -32 * log2(e)
            {swish_polynomial_exp_max, {0x4238aa3b, true}}, // 32 * log2(e)
    };
    static const table_t asimd_swish_polynomial_consts {
            {swish_polynomial_exp_min, {0xc238aa3b, true}}, // -32 * log2(e)
            {swish_polynomial_exp_max, {0x4238aa3b, true}}, // 32 * log2(e)
            {swish_asimd_exp2_c1, {0x3f3170ca, true}}, // 0.6931272745f
            {swish_asimd_exp2_c2, {0x3e75fcc9, true}}, // 0.2402221113f
            {swish_asimd_exp2_c3, {0x3d64ddb6, true}}, // 0.05587550253f
            {swish_asimd_exp2_c4, {0x3c1e721f, true}}, // 0.009670763277f
    };

    // mish(x) constants
    static const table_t mish_consts {
            {fwd_mish_max_x_for_equation_f, {0x42317217, true}},
            {bwd_mish_max_x_for_equation_f, {0x41b17217, true}}};

    // tanh(x) constants for four interval approximation
    // and for polynomial approximation
    static const table_t tanh_consts {{tanh_range, {0x3d4ccccd, true}},
            {tanh_m1d3, {0xbeaaaaab, true}},
            {tanh_idx_bias, {0x39800000, true}},
            {tanh_idx_mask, {0xffc00000, true}},
            {tanh_linear_ubound, {0x39ddb3d7, true}},
            {tanh_saturation_lbound, {0x41102cb3, true}}};

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    static const table_t tanh_polynomial_table {
            // coefficients of degree 0
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x39bfffff, false}},
            {tanh_pol_table, {0x39ffffff, false}},
            {tanh_pol_table, {0x3a3ffffe, false}},
            {tanh_pol_table, {0x3a7ffffb, false}},
            {tanh_pol_table, {0x3abffff7, false}},
            {tanh_pol_table, {0x3affffeb, false}},
            {tanh_pol_table, {0x3b3fffdc, false}},
            {tanh_pol_table, {0x3b7fffab, false}},
            {tanh_pol_table, {0x3bbfff70, false}},
            {tanh_pol_table, {0x3bfffeab, false}},
            {tanh_pol_table, {0x3c3ffdc0, false}},
            {tanh_pol_table, {0x3c7ffaab, false}},
            {tanh_pol_table, {0x3cbff701, false}},
            {tanh_pol_table, {0x3cffeaad, false}},
            {tanh_pol_table, {0x3d3fdc08, false}},
            {tanh_pol_table, {0x3d7faacd, false}},
            {tanh_pol_table, {0x3dbf7081, false}},
            {tanh_pol_table, {0x3dfeacc9, false}},
            {tanh_pol_table, {0x3e3dc7fd, false}},
            {tanh_pol_table, {0x3e7acbf5, false}},
            {tanh_pol_table, {0x3eb77a9f, false}},
            {tanh_pol_table, {0x3eec9a9f, false}},
            {tanh_pol_table, {0x3f22991f, false}},
            {tanh_pol_table, {0x3f42f7d6, false}},
            {tanh_pol_table, {0x3f67b7cc, false}},
            {tanh_pol_table, {0x3f76ca83, false}},
            {tanh_pol_table, {0x3f7ebbe9, false}},
            {tanh_pol_table, {0x3f7fd40c, false}},
            {tanh_pol_table, {0x3f7fff32, false}},
            {tanh_pol_table, {0x3f7ffffc, false}},
            {tanh_pol_table, {0x3f800000, false}},
            // coefficients of degree 1
            {tanh_pol_table, {0x3f800000, false}},
            {tanh_pol_table, {0x3f800018, false}},
            {tanh_pol_table, {0x3f7fffe8, false}},
            {tanh_pol_table, {0x3f7fffda, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffac, false}},
            {tanh_pol_table, {0x3f7fff70, false}},
            {tanh_pol_table, {0x3f7ffeec, false}},
            {tanh_pol_table, {0x3f7ffdc0, false}},
            {tanh_pol_table, {0x3f7ffbed, false}},
            {tanh_pol_table, {0x3f7ff704, false}},
            {tanh_pol_table, {0x3f7feff5, false}},
            {tanh_pol_table, {0x3f7fdbca, false}},
            {tanh_pol_table, {0x3f7fbfff, false}},
            {tanh_pol_table, {0x3f7f7041, false}},
            {tanh_pol_table, {0x3f7f009b, false}},
            {tanh_pol_table, {0x3f7dc36c, false}},
            {tanh_pol_table, {0x3f7c0aa8, false}},
            {tanh_pol_table, {0x3f7734b8, false}},
            {tanh_pol_table, {0x3f70a4de, false}},
            {tanh_pol_table, {0x3f5f1fd8, false}},
            {tanh_pol_table, {0x3f495493, false}},
            {tanh_pol_table, {0x3f18b9ec, false}},
            {tanh_pol_table, {0x3ed706cb, false}},
            {tanh_pol_table, {0x3e390b06, false}},
            {tanh_pol_table, {0x3d90b11f, false}},
            {tanh_pol_table, {0x3c21a053, false}},
            {tanh_pol_table, {0x3aaf7fdb, false}},
            {tanh_pol_table, {0x37ccc1a3, false}},
            {tanh_pol_table, {0x355c6733, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 2
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xbe4e0ff1, false}},
            {tanh_pol_table, {0x3d25b1b1, false}},
            {tanh_pol_table, {0x3d6b6dab, false}},
            {tanh_pol_table, {0x3c9fb1d5, false}},
            {tanh_pol_table, {0xbabff06f, false}},
            {tanh_pol_table, {0x3c07b3f6, false}},
            {tanh_pol_table, {0xbb3fc1bc, false}},
            {tanh_pol_table, {0x3a9f5921, false}},
            {tanh_pol_table, {0xbbbf06f2, false}},
            {tanh_pol_table, {0xbbb0f402, false}},
            {tanh_pol_table, {0xbc47db9e, false}},
            {tanh_pol_table, {0xbc73d5e7, false}},
            {tanh_pol_table, {0xbca25bda, false}},
            {tanh_pol_table, {0xbcfca780, false}},
            {tanh_pol_table, {0xbd40e07c, false}},
            {tanh_pol_table, {0xbd7dab03, false}},
            {tanh_pol_table, {0xbdbe4a0f, false}},
            {tanh_pol_table, {0xbdfb14a5, false}},
            {tanh_pol_table, {0xbe36cc8d, false}},
            {tanh_pol_table, {0xbe6bd102, false}},
            {tanh_pol_table, {0xbe9fe7c5, false}},
            {tanh_pol_table, {0xbeba0f10, false}},
            {tanh_pol_table, {0xbec206a8, false}},
            {tanh_pol_table, {0xbea3c388, false}},
            {tanh_pol_table, {0xbe277d62, false}},
            {tanh_pol_table, {0xbd8b7960, false}},
            {tanh_pol_table, {0xbc209f49, false}},
            {tanh_pol_table, {0xbaad44ca, false}},
            {tanh_pol_table, {0xb7c6eeac, false}},
            {tanh_pol_table, {0xb663aa41, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 3
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x45b3ae96, false}},
            {tanh_pol_table, {0xc414eb20, false}},
            {tanh_pol_table, {0xc450e02e, false}},
            {tanh_pol_table, {0xc3152b4e, false}},
            {tanh_pol_table, {0xbead2f56, false}},
            {tanh_pol_table, {0xc2162e02, false}},
            {tanh_pol_table, {0xbeb4bd5a, false}},
            {tanh_pol_table, {0xc11a59a4, false}},
            {tanh_pol_table, {0xbed2f507, false}},
            {tanh_pol_table, {0xc020d32c, false}},
            {tanh_pol_table, {0x3dd0f506, false}},
            {tanh_pol_table, {0xbf2a75e2, false}},
            {tanh_pol_table, {0xbff950e3, false}},
            {tanh_pol_table, {0xbed47334, false}},
            {tanh_pol_table, {0xbe809b8c, false}},
            {tanh_pol_table, {0xbeb64532, false}},
            {tanh_pol_table, {0xbe961a5b, false}},
            {tanh_pol_table, {0xbe9b63ac, false}},
            {tanh_pol_table, {0xbea0d4b2, false}},
            {tanh_pol_table, {0xbe828a77, false}},
            {tanh_pol_table, {0xbe378612, false}},
            {tanh_pol_table, {0xbdc20908, false}},
            {tanh_pol_table, {0x3d2d3957, false}},
            {tanh_pol_table, {0x3dd46e89, false}},
            {tanh_pol_table, {0x3db3f629, false}},
            {tanh_pol_table, {0x3d2c5e7b, false}},
            {tanh_pol_table, {0x3bd20403, false}},
            {tanh_pol_table, {0x3a59dfae, false}},
            {tanh_pol_table, {0x3770af45, false}},
            {tanh_pol_table, {0x372cc014, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 4
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xcc981a1b, false}},
            {tanh_pol_table, {0x4a7edd3d, false}},
            {tanh_pol_table, {0x4ab1007c, false}},
            {tanh_pol_table, {0x48fedd9c, false}},
            {tanh_pol_table, {0x41a557b5, false}},
            {tanh_pol_table, {0x477ee32a, false}},
            {tanh_pol_table, {0x422557f5, false}},
            {tanh_pol_table, {0x45ff3ce4, false}},
            {tanh_pol_table, {0x42a55641, false}},
            {tanh_pol_table, {0x446e0867, false}},
            {tanh_pol_table, {0xc33dc19a, false}},
            {tanh_pol_table, {0x42915214, false}},
            {tanh_pol_table, {0x43af4fad, false}},
            {tanh_pol_table, {0x4110fe88, false}},
            {tanh_pol_table, {0xc1099b75, false}},
            {tanh_pol_table, {0x3fc8a8dc, false}},
            {tanh_pol_table, {0xbfbeaef5, false}},
            {tanh_pol_table, {0xbe365aad, false}},
            {tanh_pol_table, {0x3f4d9652, false}},
            {tanh_pol_table, {0x3ddfa08f, false}},
            {tanh_pol_table, {0x3e34e9b8, false}},
            {tanh_pol_table, {0x3e2d07a6, false}},
            {tanh_pol_table, {0x3dc63567, false}},
            {tanh_pol_table, {0x3cdaeb78, false}},
            {tanh_pol_table, {0xbcd17537, false}},
            {tanh_pol_table, {0xbc92829c, false}},
            {tanh_pol_table, {0xbb43ab99, false}},
            {tanh_pol_table, {0xb9b471dd, false}},
            {tanh_pol_table, {0xb6baad5a, false}},
            {tanh_pol_table, {0xb78bafc7, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 5
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x52f688d5, false}},
            {tanh_pol_table, {0xd0505c72, false}},
            {tanh_pol_table, {0xd08f98e3, false}},
            {tanh_pol_table, {0xce505cc9, false}},
            {tanh_pol_table, {0xc7162b8a, false}},
            {tanh_pol_table, {0xcc5061d6, false}},
            {tanh_pol_table, {0xc7162bdf, false}},
            {tanh_pol_table, {0xca50b37f, false}},
            {tanh_pol_table, {0xc7162a3a, false}},
            {tanh_pol_table, {0xc8422086, false}},
            {tanh_pol_table, {0x471a714e, false}},
            {tanh_pol_table, {0xc5ece1f1, false}},
            {tanh_pol_table, {0xc70e3d90, false}},
            {tanh_pol_table, {0xc3eba94a, false}},
            {tanh_pol_table, {0x43e0c424, false}},
            {tanh_pol_table, {0xc21f4552, false}},
            {tanh_pol_table, {0x42217cc8, false}},
            {tanh_pol_table, {0x405e7dc4, false}},
            {tanh_pol_table, {0xc10dd401, false}},
            {tanh_pol_table, {0x3e96b602, false}},
            {tanh_pol_table, {0xbd1a6d2f, false}},
            {tanh_pol_table, {0xbd393883, false}},
            {tanh_pol_table, {0xbd674682, false}},
            {tanh_pol_table, {0xbd310016, false}},
            {tanh_pol_table, {0xb961e269, false}},
            {tanh_pol_table, {0x3ba32495, false}},
            {tanh_pol_table, {0x3a7680d5, false}},
            {tanh_pol_table, {0x38b3173c, false}},
            {tanh_pol_table, {0x35a9deea, false}},
            {tanh_pol_table, {0x375c3f2a, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 6
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xd8995ed1, false}},
            {tanh_pol_table, {0x558285ea, false}},
            {tanh_pol_table, {0x55b2cd69, false}},
            {tanh_pol_table, {0x53028625, false}},
            {tanh_pol_table, {0x4bc9991f, false}},
            {tanh_pol_table, {0x5082898a, false}},
            {tanh_pol_table, {0x4b4999b3, false}},
            {tanh_pol_table, {0x4e02c07c, false}},
            {tanh_pol_table, {0x4ac99764, false}},
            {tanh_pol_table, {0x4b72c822, false}},
            {tanh_pol_table, {0xca40c0e1, false}},
            {tanh_pol_table, {0x489413e4, false}},
            {tanh_pol_table, {0x49b12224, false}},
            {tanh_pol_table, {0x46134c4e, false}},
            {tanh_pol_table, {0xc60c2d57, false}},
            {tanh_pol_table, {0x43c83910, false}},
            {tanh_pol_table, {0xc3c872d1, false}},
            {tanh_pol_table, {0xc186bc9e, false}},
            {tanh_pol_table, {0x42325bc3, false}},
            {tanh_pol_table, {0xbf2ffa4a, false}},
            {tanh_pol_table, {0x3d9a203c, false}},
            {tanh_pol_table, {0xbc545a43, false}},
            {tanh_pol_table, {0xbae08fee, false}},
            {tanh_pol_table, {0x3c80225d, false}},
            {tanh_pol_table, {0x3b1fd1df, false}},
            {tanh_pol_table, {0xba36b9d1, false}},
            {tanh_pol_table, {0xb91de544, false}},
            {tanh_pol_table, {0xb71f100f, false}},
            {tanh_pol_table, {0xb408e2ed, false}},
            {tanh_pol_table, {0xb685fec8, false}},
            {tanh_pol_table, {0x00000000, false}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };
    static const table_t gelu_erf_lut_consts {
            {gelu_erf_lut_max, {0x407c0000, true}}, // 3.9375f
            {gelu_erf_lut_third, {0x3eaaaaab, true}}, // 1/3
            {gelu_erf_lut_bias, {0x47800000, true}}, // 65536.f
            {gelu_erf_lut_max_index, {0x00000200, true}}, // 512
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };
    // gelu_erf(x) constants for direct erf approximation (formula defined)
    static const table_t gelu_erf_minimax_consts {
            {gelu_erf_idx_bias, {0xc21fffff, true}},
            {gelu_erf_rbound, {0x40b15cee, true}},
            {gelu_erf_one, {0x00000001, true}},
            {gelu_erf_twenty_three, {0x00000017, true}},
            {gelu_erf_twenty_four, {0x00000018, true}},
    };
    // gelu_erf(x) minimax polynomials for piecewise approximaxtion
    static const table_t gelu_erf_minimax_polynomial {
            // coefficients of degree  0
            {gelu_erf_minimax_pol, {0xa6f2cb94, false}}, // -0x1.e59728p-50
            {gelu_erf_minimax_pol, {0x32827792, false}}, // 0x1.04ef24p-26
            {gelu_erf_minimax_pol, {0x3381cc0c, false}}, // 0x1.039818p-24
            {gelu_erf_minimax_pol, {0x34523d4a, false}}, // 0x1.a47a94p-23
            {gelu_erf_minimax_pol, {0x351ac44d, false}}, // 0x1.35889ap-21
            {gelu_erf_minimax_pol, {0x35f36d88, false}}, // 0x1.e6db1p-20
            {gelu_erf_minimax_pol, {0x36ee8229, false}}, // 0x1.dd0452p-18
            {gelu_erf_minimax_pol, {0x37b8a3bb, false}}, // 0x1.714776p-16
            {gelu_erf_minimax_pol, {0x3867a213, false}}, // 0x1.cf4426p-15
            {gelu_erf_minimax_pol, {0x3940033b, false}}, // 0x1.800676p-13
            {gelu_erf_minimax_pol, {0x3a2a5a1d, false}}, // 0x1.54b43ap-11
            {gelu_erf_minimax_pol, {0x3ae35863, false}}, // 0x1.c6b0c6p-10
            {gelu_erf_minimax_pol, {0x3b7828f2, false}}, // 0x1.f051e4p-9
            {gelu_erf_minimax_pol, {0x3c08b14b, false}}, // 0x1.116296p-7
            {gelu_erf_minimax_pol, {0x3c515ed3, false}}, // 0x1.a2bda6p-7
            {gelu_erf_minimax_pol, {0xbb503236, false}}, // -0x1.a0646cp-9
            {gelu_erf_minimax_pol, {0xbd8d8e5e, false}}, // -0x1.1b1cbcp-4
            {gelu_erf_minimax_pol, {0xbe8abcd9, false}}, // -0x1.1579b2p-2
            {gelu_erf_minimax_pol, {0xbf0c19a2, false}}, // -0x1.183344p-1
            {gelu_erf_minimax_pol, {0xbeccb328, false}}, // -0x1.99665p-2
            {gelu_erf_minimax_pol, {0x3e176ced, false}}, // 0x1.2ed9dap-3
            {gelu_erf_minimax_pol, {0x3f470d99, false}}, // 0x1.8e1b32p-1
            {gelu_erf_minimax_pol, {0x3f7abb28, false}}, // 0x1.f5765p-1
            {gelu_erf_minimax_pol, {0x3f800000, false}}, // 0x1p0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 1
            {gelu_erf_minimax_pol, {0x3f4c422a, false}}, // 0x1.988454p-1
            {gelu_erf_minimax_pol, {0x3f4c421f, false}}, // 0x1.98843ep-1
            {gelu_erf_minimax_pol, {0x3f4c4207, false}}, // 0x1.98840ep-1
            {gelu_erf_minimax_pol, {0x3f4c41cb, false}}, // 0x1.988396p-1
            {gelu_erf_minimax_pol, {0x3f4c413b, false}}, // 0x1.988276p-1
            {gelu_erf_minimax_pol, {0x3f4c3fad, false}}, // 0x1.987f5ap-1
            {gelu_erf_minimax_pol, {0x3f4c3a2f, false}}, // 0x1.98745ep-1
            {gelu_erf_minimax_pol, {0x3f4c2d40, false}}, // 0x1.985a8p-1
            {gelu_erf_minimax_pol, {0x3f4c146a, false}}, // 0x1.9828d4p-1
            {gelu_erf_minimax_pol, {0x3f4bc341, false}}, // 0x1.978682p-1
            {gelu_erf_minimax_pol, {0x3f4ad08c, false}}, // 0x1.95a118p-1
            {gelu_erf_minimax_pol, {0x3f48f8cf, false}}, // 0x1.91f19ep-1
            {gelu_erf_minimax_pol, {0x3f45fac7, false}}, // 0x1.8bf58ep-1
            {gelu_erf_minimax_pol, {0x3f404e07, false}}, // 0x1.809c0ep-1
            {gelu_erf_minimax_pol, {0x3f3b980f, false}}, // 0x1.77301ep-1
            {gelu_erf_minimax_pol, {0x3f48dff3, false}}, // 0x1.91bfe6p-1
            {gelu_erf_minimax_pol, {0x3f78b21b, false}}, // 0x1.f16436p-1
            {gelu_erf_minimax_pol, {0x3fbb0704, false}}, // 0x1.760e08p0
            {gelu_erf_minimax_pol, {0x40019c32, false}}, // 0x1.033864p1
            {gelu_erf_minimax_pol, {0x3fe536d6, false}}, // 0x1.ca6dacp0
            {gelu_erf_minimax_pol, {0x3f81331e, false}}, // 0x1.02663cp0
            {gelu_erf_minimax_pol, {0x3e6c8684, false}}, // 0x1.d90d08p-3
            {gelu_erf_minimax_pol, {0x3c98f936, false}}, // 0x1.31f26cp-6
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x3f800000, false}}, // 0x1p0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 2
            {gelu_erf_minimax_pol, {0xb62173f4, false}}, // -0x1.42e7e8p-19
            {gelu_erf_minimax_pol, {0x3735e4cf, false}}, // 0x1.6bc99ep-17
            {gelu_erf_minimax_pol, {0x37f2ff89, false}}, // 0x1.e5ff12p-16
            {gelu_erf_minimax_pol, {0x388c23be, false}}, // 0x1.18477cp-14
            {gelu_erf_minimax_pol, {0x3917535c, false}}, // 0x1.2ea6b8p-13
            {gelu_erf_minimax_pol, {0x39ab2ab0, false}}, // 0x1.56556p-12
            {gelu_erf_minimax_pol, {0x3a60fadb, false}}, // 0x1.c1f5b6p-11
            {gelu_erf_minimax_pol, {0x3af9b960, false}}, // 0x1.f372cp-10
            {gelu_erf_minimax_pol, {0x3b6e5491, false}}, // 0x1.dca922p-9
            {gelu_erf_minimax_pol, {0x3c0a4ec5, false}}, // 0x1.149d8ap-7
            {gelu_erf_minimax_pol, {0x3ca5aa8c, false}}, // 0x1.4b5518p-6
            {gelu_erf_minimax_pol, {0x3d2138d9, false}}, // 0x1.4271b2p-5
            {gelu_erf_minimax_pol, {0x3d8737d4, false}}, // 0x1.0e6fa8p-4
            {gelu_erf_minimax_pol, {0x3ddfb660, false}}, // 0x1.bf6ccp-4
            {gelu_erf_minimax_pol, {0x3e0f27ab, false}}, // 0x1.1e4f56p-3
            {gelu_erf_minimax_pol, {0x3d94004b, false}}, // 0x1.280096p-4
            {gelu_erf_minimax_pol, {0xbe0efdeb, false}}, // -0x1.1dfbd6p-3
            {gelu_erf_minimax_pol, {0xbf1d96c3, false}}, // -0x1.3b2d86p-1
            {gelu_erf_minimax_pol, {0xbf89db58, false}}, // -0x1.13b6bp0
            {gelu_erf_minimax_pol, {0xbf6d9897, false}}, // -0x1.db312ep-1
            {gelu_erf_minimax_pol, {0xbef69fb8, false}}, // -0x1.ed3f7p-2
            {gelu_erf_minimax_pol, {0xbdc4f8a8, false}}, // -0x1.89f15p-4
            {gelu_erf_minimax_pol, {0xbbde6422, false}}, // -0x1.bcc844p-8
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 3
            {gelu_erf_minimax_pol, {0xbe081a19, false}}, // -0x1.103432p-3
            {gelu_erf_minimax_pol, {0xbe084570, false}}, // -0x1.108aep-3
            {gelu_erf_minimax_pol, {0xbe08639b, false}}, // -0x1.10c736p-3
            {gelu_erf_minimax_pol, {0xbe089837, false}}, // -0x1.11306ep-3
            {gelu_erf_minimax_pol, {0xbe08f409, false}}, // -0x1.11e812p-3
            {gelu_erf_minimax_pol, {0xbe09ab95, false}}, // -0x1.13572ap-3
            {gelu_erf_minimax_pol, {0xbe0b66d0, false}}, // -0x1.16cdap-3
            {gelu_erf_minimax_pol, {0xbe0e400a, false}}, // -0x1.1c8014p-3
            {gelu_erf_minimax_pol, {0xbe124df8, false}}, // -0x1.249bfp-3
            {gelu_erf_minimax_pol, {0xbe1bde02, false}}, // -0x1.37bc04p-3
            {gelu_erf_minimax_pol, {0xbe2f19c9, false}}, // -0x1.5e3392p-3
            {gelu_erf_minimax_pol, {0xbe4931bf, false}}, // -0x1.92637ep-3
            {gelu_erf_minimax_pol, {0xbe685fbc, false}}, // -0x1.d0bf78p-3
            {gelu_erf_minimax_pol, {0xbe89c95f, false}}, // -0x1.1392bep-2
            {gelu_erf_minimax_pol, {0xbe96cbca, false}}, // -0x1.2d9794p-2
            {gelu_erf_minimax_pol, {0xbe8044aa, false}}, // -0x1.008954p-2
            {gelu_erf_minimax_pol, {0xbe0550f2, false}}, // -0x1.0aa1e4p-3
            {gelu_erf_minimax_pol, {0x3dcfd6a1, false}}, // 0x1.9fad42p-4
            {gelu_erf_minimax_pol, {0x3e94c826, false}}, // 0x1.29904cp-2
            {gelu_erf_minimax_pol, {0x3e79345f, false}}, // 0x1.f268bep-3
            {gelu_erf_minimax_pol, {0x3decec91, false}}, // 0x1.d9d922p-4
            {gelu_erf_minimax_pol, {0x3ca46568, false}}, // 0x1.48cadp-6
            {gelu_erf_minimax_pol, {0x3aa1e00a, false}}, // 0x1.43c014p-10
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 4
            {gelu_erf_minimax_pol, {0xba3d61db, false}}, // -0x1.7ac3b6p-11
            {gelu_erf_minimax_pol, {0x39f097a3, false}}, // 0x1.e12f46p-12
            {gelu_erf_minimax_pol, {0x3a5845dc, false}}, // 0x1.b08bb8p-11
            {gelu_erf_minimax_pol, {0x3ab1fa35, false}}, // 0x1.63f46ap-10
            {gelu_erf_minimax_pol, {0x3b0cefb8, false}}, // 0x1.19df7p-9
            {gelu_erf_minimax_pol, {0x3b653ab6, false}}, // 0x1.ca756cp-9
            {gelu_erf_minimax_pol, {0x3bcae527, false}}, // 0x1.95ca4ep-8
            {gelu_erf_minimax_pol, {0x3c221712, false}}, // 0x1.442e24p-7
            {gelu_erf_minimax_pol, {0x3c6c5840, false}}, // 0x1.d8b08p-7
            {gelu_erf_minimax_pol, {0x3cc0a703, false}}, // 0x1.814e06p-6
            {gelu_erf_minimax_pol, {0x3d1dcc19, false}}, // 0x1.3b9832p-5
            {gelu_erf_minimax_pol, {0x3d63656d, false}}, // 0x1.c6cadap-5
            {gelu_erf_minimax_pol, {0x3d955907, false}}, // 0x1.2ab20ep-4
            {gelu_erf_minimax_pol, {0x3dbf9910, false}}, // 0x1.7f322p-4
            {gelu_erf_minimax_pol, {0x3dd53f69, false}}, // 0x1.aa7ed2p-4
            {gelu_erf_minimax_pol, {0x3db7dcef, false}}, // 0x1.6fb9dep-4
            {gelu_erf_minimax_pol, {0x3d639ebe, false}}, // 0x1.c73d7cp-5
            {gelu_erf_minimax_pol, {0xba6ede48, false}}, // -0x1.ddbc9p-11
            {gelu_erf_minimax_pol, {0xbd22be69, false}}, // -0x1.457cd2p-5
            {gelu_erf_minimax_pol, {0xbd041cf1, false}}, // -0x1.0839e2p-5
            {gelu_erf_minimax_pol, {0xbc64f5ab, false}}, // -0x1.c9eb56p-7
            {gelu_erf_minimax_pol, {0xbb097a32, false}}, // -0x1.12f464p-9
            {gelu_erf_minimax_pol, {0xb8ebf380, false}}, // -0x1.d7e7p-14
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            // coefficients of degree 5
            {gelu_erf_minimax_pol, {0x3cb7d80c, false}}, // 0x1.6fb018p-6
            {gelu_erf_minimax_pol, {0x3c9b6050, false}}, // 0x1.36c0ap-6
            {gelu_erf_minimax_pol, {0x3c978d11, false}}, // 0x1.2f1a22p-6
            {gelu_erf_minimax_pol, {0x3c92e850, false}}, // 0x1.25d0ap-6
            {gelu_erf_minimax_pol, {0x3c8d058b, false}}, // 0x1.1a0b16p-6
            {gelu_erf_minimax_pol, {0x3c848454, false}}, // 0x1.0908a8p-6
            {gelu_erf_minimax_pol, {0x3c6cd623, false}}, // 0x1.d9ac46p-7
            {gelu_erf_minimax_pol, {0x3c4c824b, false}}, // 0x1.990496p-7
            {gelu_erf_minimax_pol, {0x3c2a7935, false}}, // 0x1.54f26ap-7
            {gelu_erf_minimax_pol, {0x3be0b390, false}}, // 0x1.c1672p-8
            {gelu_erf_minimax_pol, {0x3b0651ac, false}}, // 0x1.0ca358p-9
            {gelu_erf_minimax_pol, {0xbb232f53, false}}, // -0x1.465ea6p-9
            {gelu_erf_minimax_pol, {0xbbd42fa0, false}}, // -0x1.a85f4p-8
            {gelu_erf_minimax_pol, {0xbc2c5366, false}}, // -0x1.58a6ccp-7
            {gelu_erf_minimax_pol, {0xbc492c9e, false}}, // -0x1.92593cp-7
            {gelu_erf_minimax_pol, {0xbc2a7aa6, false}}, // -0x1.54f54cp-7
            {gelu_erf_minimax_pol, {0xbbd55d04, false}}, // -0x1.aaba08p-8
            {gelu_erf_minimax_pol, {0xba823a76, false}}, // -0x1.0474ecp-10
            {gelu_erf_minimax_pol, {0x3b102aa8, false}}, // 0x1.20555p-9
            {gelu_erf_minimax_pol, {0x3ae25a7e, false}}, // 0x1.c4b4fcp-10
            {gelu_erf_minimax_pol, {0x3a31f792, false}}, // 0x1.63ef24p-11
            {gelu_erf_minimax_pol, {0x38b84375, false}}, // 0x1.7086eap-14
            {gelu_erf_minimax_pol, {0x3689bb5a, false}}, // 0x1.1376b4p-18
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
            {gelu_erf_minimax_pol, {0x00000000, false}}, // 0 padd
    };

    // log(x) constants
    static const table_t log_consts {
            {log_const_127, {0x0000007f, true}},
            {log_ln2, {0x3f317218, true}},
            {log_qnan, {0x7fc00000, true}},
            {log_inf, {0x7f800000, true}},
            {log_minus_inf, {0xff800000, true}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0xc012eb5b, true}}, // -2.29561495781
            {log_pol, {0xc01e2024, true}}, // -2.47071170807
            {log_pol, {0xc0b5fb4c, true}}, // -5.68692588806
            {log_pol, {0xbe29383a, true}}, // -0.165253549814
            {log_pol, {0x40a5a113, true}}, // 5.17591238022
            {log_pol, {0x3f5810d8, true}}, // 0.844007015228
            {log_pol, {0x4092b3e3, true}}, // 4.58445882797
            {log_pol, {0x3c677861, true}}, // 0.0141278216615
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh:
                    exp_ = true;
                    gelu_tanh_ = true;
                    break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_mish: mish_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh:
                    exp_ = true;
                    tanh_ = true;
                    break;
                default: break;
            }
        }

        bool exp_ = false;
        bool mish_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_ || mish_; }
        bool mish() const { return mish_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);
    const bool use_swish_poly = is_fwd_ && alg_ == alg_kind::eltwise_swish;
    if (use_swish_poly) need.exp_ = false;

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    // For f16 data: convert alpha/beta from f32 to f16, then pack each value twice
    // into a single 32-bit word ([f16 | f16]) to match the 4-byte memory write granularity.
    // This ensures proper alignment and broadcasting when using vector registers.
    push_arg_entry_of(alpha,
            d_type_ == data_type::f16 ? pack_f16_twice_from_f32(alpha_)
                                      : float2int(alpha_),
            true);
    push_arg_entry_of(beta,
            d_type_ == data_type::f16 ? pack_f16_twice_from_f32(beta_)
                                      : float2int(beta_),
            true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.exp()) push_entries_of(exp_consts2);
    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.mish()) push_entries_of(mish_consts);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial_table);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);
    if (need.gelu_erf()) push_entries_of(gelu_erf_minimax_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_minimax_polynomial);
    if (need.gelu_erf()) {
        push_entries_of(gelu_erf_lut_consts);
        const int erf_lut_size = 513;
        const float erf_lut_step = 1.f / 128.f;
        const float two_over_sqrt_pi = 1.12837922573089599609375f;
        for (int i = 0; i < erf_lut_size; i++) {
            const float r = i * erf_lut_step;
            const float erf_val = std::erf(r);
            const float scale_val = two_over_sqrt_pi * std::exp(-r * r);
            if (isa == asimd) {
                push_arg_entry_of(gelu_erf_lut, float2int(erf_val), false);
                push_arg_entry_of(gelu_erf_lut, float2int(scale_val), false);
            } else {
                push_arg_entry_of(gelu_erf_lut_erf, float2int(erf_val), false);
                push_arg_entry_of(
                        gelu_erf_lut_scale, float2int(scale_val), false);
            }
        }
    }
    if (use_swish_poly) {
        constexpr float log2e = 1.4426950408889634f;
        push_arg_entry_of(swish_alpha_log2e, float2int(alpha_ * log2e), true);
        push_entries_of(isa == sve ? sve_swish_polynomial_consts
                                   : asimd_swish_polynomial_consts);
    }
    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template <>
size_t jit_uni_eltwise_injector_t<asimd>::get_vec_len() {
    return 16;
}

template <>
size_t jit_uni_eltwise_injector_t<sve>::get_vec_len() {
    return get_sve_length();
}

template <>
void jit_uni_eltwise_injector_t<asimd>::exp_compute_vector_fwd(
        const TRegS &vmm_src, float min_input, float max_input) {
    /*
    * Based on the expf implementation from Arm Optimized Routines (AOR)
    * Accuracy: maxerr ≈ 1.95 ULP.
    *
    * exp(x) ≈ 2^n * (1 + poly(r)), with 1 + poly(r) ∈ [1/sqrt(2),sqrt(2)]
    *   n = round(x * 1/ln2)
    *   r = x − n*ln2 (hi/lo split for precision)
    *   poly(r): short polynomial, r ∈ [−ln2/2, ln2/2]
    *
    * Special case:
    *   For large |n| > 126, split 2^n = s1*s2 to avoid overflow/underflow.
    *   For very large |n| > 196, use exp(x) = s1*s1.
    */

    Xbyak_aarch64::Label L_done, L_special;
    const auto &t0 = vmm_src;
    auto t_tmp = reg_pool_.allocScoped<TRegS>();
    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    auto t3 = reg_pool_.allocScoped<TRegS>();
    auto t4 = reg_pool_.allocScoped<TRegS>();

    const float special_case_input_threshold = 126.5f * logf(2.0f); // ~87.6831f
    const float ln_flt_min = logf(FLT_MIN); // ~-87.3365f
    bool need_clamp = min_input < ln_flt_min;
    bool need_special_case = max_input >= special_case_input_threshold;

    if (!need_special_case && need_clamp) {
        // Clamp x to avoid overflow of f32 exponent bits
        h->fmax(t0, t0, table_val(exp_ln_flt_min_f, t4));
    }

    // z = x * inv_ln2
    const auto &v_inv_ln2 = table_val(exp_log2ef, t_tmp);
    const auto &v_src = t0;
    const auto &v_z = t_tmp;
    h->fmul(v_z, v_src, v_inv_ln2);

    // n = (float)round(z)
    const auto &v_n = t1;
    h->frinta(v_n, v_z);

    // e = (int)round(z) << n_mantissa_bits
    const auto &v_e = t2;
    h->fcvtas(v_z, v_z);
    h->shl(v_e, v_z, n_mantissa_bits);

    // r = x + n*neg_ln2_hi + n*neg_ln2_lo
    const auto &v_r = v_src;
    const auto &v_neg_ln2_hi = table_val(exp_neg_ln2_hi, t3);
    const auto &v_neg_ln2_lo = table_val(exp_neg_ln2_lo, t4);
    h->fmla(v_r, v_n, v_neg_ln2_hi);
    h->fmla(v_r, v_n, v_neg_ln2_lo);

    // Polynomial approximation of exp(r), r ∈ [−ln2/2, ln2/2]
    // p = c1 + c0 * r
    const auto &v_c0 = v_z;
    const auto &v_c1 = v_neg_ln2_hi;
    table_val(exp_pol_asimd, v_c0, 0);
    table_val(exp_pol_asimd, v_c1, 1);
    h->fmla(v_c1, v_c0, v_r);

    // q = c3 + c2 * r
    const auto &v_c2 = v_c0;
    const auto &v_c3 = v_neg_ln2_lo;
    table_val(exp_pol_asimd, v_c2, 2);
    table_val(exp_pol_asimd, v_c3, 3);
    h->fmla(v_c3, v_c2, v_r);

    // r2 = r * r
    const auto &v_r2 = v_c2;
    h->fmul(v_r2, v_r, v_r);

    // q = q + p * r2
    const auto &v_p = v_c1;
    const auto &v_q = v_c3;
    h->fmla(v_q, v_p, v_r2);

    // poly = c4 * r + q * r2
    const auto &v_c4 = v_p;
    table_val(exp_pol_asimd, v_c4, 4);
    h->fmul(v_p, v_c4, v_r);
    h->fmla(v_p, v_q, v_r2);

    if (need_special_case) {
        // Check if any lane needs special-case handling
        // i.e. if any input |x| ≥ 126.5 / ln 2 ≈ 87.6831f
        // Includes ±Inf, but not NaNs, which propagate through the fast path
        // mask_special = (|n| > 126)
        const auto &v_mask_special = v_c3;
        const auto &v_special_bound
                = table_val(exp_special_bound, t_tmp); // 126.0f
        h->facgt(v_mask_special, v_n, v_special_bound);
        h->addp(DReg(t_tmp.getIdx()), VReg2D(v_mask_special.getIdx()));
        auto x_tmp0 = reg_pool_.allocScoped<XReg>();
        h->fmov(x_tmp0, DReg(t_tmp.getIdx()));
        h->cbnz(x_tmp0, L_special);
    }

    // ===== Fast path =====
    // scale = reinterpret_f32(e + 0x3f800000) = 2^n
    const auto &v_exp_scale = v_src;
    const auto &v_exp_bias_bits
            = table_val(exp_exponent_bias, t_tmp); // 0x3f800000
    h->add(v_exp_scale, v_e, v_exp_bias_bits);

    // exp(x) = scale + poly * scale = 2^n * (1 + poly(r))
    const auto &v_dst = v_src;
    h->fmla(v_dst, v_p, v_exp_scale);

    if (need_special_case) {
        // ===== Special-case handling =====
        // b = (n <= 0) ? special_offset : 0
        h->b(L_done);
        h->L(L_special);
        const auto &v_b = v_q;
        const auto &v_special_offset
                = table_val(exp_special_offset, t_tmp); // 0x82000000
        h->fcmle(v_b, v_n, 0.0);
        h->and_(VReg16B(v_b.getIdx()), VReg16B(v_b.getIdx()),
                VReg16B(v_special_offset.getIdx()));

        // mask_thresh = (|n| > 192)
        const auto &v_thresh = table_val(exp_scale_thresh, t_tmp); // 192.0f
        const auto &v_mask_thresh = v_n;
        h->facgt(v_mask_thresh, v_n, v_thresh);

        // s2_bits = e - b
        const auto &v_s2 = v_e;
        h->sub(v_s2, v_e, v_b);

        // s1_bits = b + special_bias
        const auto &v_s1 = v_b;
        const auto &v_special_bias
                = table_val(exp_special_bias, t_tmp); // 0x7f000000
        h->add(v_s1, v_b, v_special_bias);

        // r0 = (s2 + poly*s2) * s1
        const auto &v_r0 = v_p;
        h->fmla(v_s2, v_p, v_s2);
        h->fmul(v_r0, v_s2, v_s1);

        // r1 = s1 * s1
        const auto &v_r1 = v_dst;
        h->fmul(v_r1, v_s1, v_s1);

        // out_special = (|n| > 192) ? r1 : r0
        h->bif(VReg16B(v_r1.getIdx()), VReg16B(v_r0.getIdx()),
                VReg16B(v_mask_thresh.getIdx()));

        h->L(L_done);
    }
}

template <>
void jit_uni_eltwise_injector_t<asimd>::relu_zero_ns_compute_vector_fwd(
        const TReg &vmm_src) {
    auto zero = reg_pool_.allocScoped<TRegS>();
    if (d_type_ == data_type::f16) {
        h->movi(TRegH(IDX(zero)), 0);
        h->fmaxnm(vmm_src.h, vmm_src.h, TRegH(IDX(zero)));
    } else {
        h->movi(zero, 0);
        h->fmaxnm(vmm_src.s, vmm_src.s, zero);
    }
}

template <>
void jit_uni_eltwise_injector_t<asimd>::relu_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    // Compute x > 0 ? x : alpha * x.
    // For alpha <= 1, this is equivalent to max(x, alpha * x).
    if (alpha_ <= 1.f) {
        auto scaled = reg_pool_.allocScoped<TRegS>();
        h->fmul(scaled, vmm_src, alpha);
        h->fmaxnm(vmm_src, vmm_src, scaled);
    } else {
        // For alpha > 1, keep positive lanes unchanged and scale only
        // x <= 0 lanes.
        auto src = reg_pool_.allocScoped<TRegS>();
        auto mask = reg_pool_.allocScoped<TRegS>();
        h->mov(VReg16B(src.getIdx()), VReg16B(vmm_src.getIdx()));
        h->fcmgt(mask, src, 0.);
        h->fmul(vmm_src, vmm_src, alpha);
        h->bit(VReg16B(vmm_src.getIdx()), VReg16B(src.getIdx()),
                VReg16B(mask.getIdx()));
    }
}

template <>
void jit_uni_eltwise_injector_t<asimd>::elu_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha) {
    auto exp = reg_pool_.allocScoped<TRegS>();
    h->mov(VReg16B(exp.getIdx()), VReg16B(vmm_src.getIdx()));

    // Compute exponent
    exp_compute_vector_fwd(exp, min_input_, 0.f);

    auto result = reg_pool_.allocScoped<TRegS>();
    auto mask = reg_pool_.allocScoped<TRegS>();

    // alpha * (exp(x) - 1) = -alpha - (-alpha * exp(x))
    if (alpha_ != 1.f) {
        h->fneg(result, alpha);
        h->fmls(result, result, exp);
    } else {
        h->fsub(result, exp, alpha);
    }

    // Combine with mask
    h->fcmle(mask, vmm_src, 0.);
    h->bit(VReg16B(vmm_src.getIdx()), VReg16B(result.getIdx()),
            VReg16B(mask.getIdx()));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    // tanh(x) = x(1 + (-1/3)x^2) for |x| < tanh_range
    // tanh(x) = 1 - 2/(1 + exp(2 x)) for otherwise

    const auto &t0 = vmm_src;
    auto src = reg_pool_.allocScoped<TRegS>();
    auto mask = reg_pool_.allocScoped<TRegS>();

    // Make mask for small x
    h->mov(VReg16B(src.getIdx()), VReg16B(t0.getIdx()));
    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->facge(mask, table_val(tanh_range, table_value.get()), t0);
    }

    // 2x
    h->fadd(t0, t0, t0);
    // exp(2x)
    exp_compute_vector_fwd(t0);

    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    auto table_value = reg_pool_.allocScoped<TRegS>();

    // 1+exp(2x)
    h->fadd(t0, t0, table_val(one, t2));
    // 2/(1+exp(2x))
    h->fdiv(t0, table_val(two, t1), t0);
    // 1-2/(1+exp(2x))
    h->fsub(t0, t2, t0);

    // tanh(x) = x(1 - x^2/3) for |x| < tanh_range
    h->fmul(t1, src, src);
    h->fmla(t2, table_val(tanh_m1d3, table_value), t1);
    h->fmul(t1, src, t2);
    // Select the correct value according to mask
    h->bit(VReg16B(t0.getIdx()), VReg16B(t1.getIdx()), VReg16B(mask.getIdx()));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::gelu_tanh_compute_vector_fwd(
        const TRegS &vmm_src) {
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(VReg16B(IDX(src)), VReg16B(IDX(vmm_src)));

    // Compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    {
        auto polynomial = reg_pool_.allocScoped<TRegS>();
        auto fitting = reg_pool_.allocScoped<TRegS>();
        auto factor = reg_pool_.allocScoped<TRegS>();
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->fmul(vmm_src, vmm_src, vmm_src);
        h->fmla(table_val(one, polynomial),
                table_val(gelu_tanh_fitting_const, fitting), vmm_src);
        h->fmul(factor, src,
                table_val(gelu_tanh_sqrt_two_over_pi, table_value));
        h->fmul(vmm_src, polynomial, factor);
    }

    // Compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    // Compute 0.5 * x * (1 + tanh(G(x)))
    h->fmla(src, vmm_src, src);
    auto half_reg = reg_pool_.allocScoped<TRegS>();
    h->fmul(vmm_src, src, table_val(half, half_reg.get()));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::abs_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fabs(vmm_src.h, vmm_src.h);
    else
        h->fabs(vmm_src.s, vmm_src.s);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::sqrt_compute_vector_fwd(
        const TReg &vmm_src) {
    if (d_type_ == data_type::f16)
        h->fsqrt(vmm_src.h, vmm_src.h);
    else
        h->fsqrt(vmm_src.s, vmm_src.s);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::linear_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // compute x = alpha * x + beta;
    h->fmul(vmm_src, vmm_src, alpha);
    h->fadd(vmm_src, vmm_src, beta);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::clip_compute_vector_fwd(
        const TReg &vmm_src, const TRegS &lower, const TRegS &upper) {
    if (d_type_ == data_type::f16) {
        h->fmaxnm(vmm_src.h, vmm_src.h, TRegH(IDX(lower)));
        h->fminnm(vmm_src.h, vmm_src.h, TRegH(IDX(upper)));
    } else {
        h->fmaxnm(vmm_src.s, vmm_src.s, lower);
        h->fminnm(vmm_src.s, vmm_src.s, upper);
    }
}

template <>
void jit_uni_eltwise_injector_t<asimd>::mish_compute_vector_fwd(
        const TRegS &vmm_src) {
    // This is equivalent to mish(x) = x*tanh(soft_relu(x)), using
    // tanh(y) = (e^y - e^-y) / (e^y + e^-y):
    //   mish(x) = x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1).
    // It reuses the cheaper exp approximation and needs fewer constants.
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(VReg16B(src.getIdx()), VReg16B(vmm_src.getIdx()));

    // Avoid overflow in (exp(x)+1)^2 by clamping to log(sqrt(FLT_MAX)).
    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->fminnm(vmm_src, vmm_src,
                table_val(fwd_mish_max_x_for_equation_f, table_value.get()));
    }
    exp_compute_vector_fwd(vmm_src, min_input_, 0.5f * logf(FLT_MAX));

    auto numerator = reg_pool_.allocScoped<TRegS>();
    auto denominator = reg_pool_.allocScoped<TRegS>();
    auto one = reg_pool_.allocScoped<TRegS>();

    // (exp(x)+1)^2
    h->fadd(vmm_src, vmm_src, table_val(key_t::one, one));
    h->fmul(vmm_src, vmm_src, vmm_src);
    // x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1)
    h->fsub(numerator, vmm_src, one);
    h->fadd(denominator, vmm_src, one);
    h->fdiv(vmm_src, numerator, denominator);
    h->fmul(vmm_src, vmm_src, src);
}
template <>
void jit_uni_eltwise_injector_t<asimd>::hardsigmoid_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // result = max(0, min(1, alpha * x + beta))
    auto bound = reg_pool_.allocScoped<TRegS>();
    h->fmul(vmm_src, vmm_src, alpha);
    h->fadd(vmm_src, vmm_src, beta);
    h->fminnm(vmm_src, vmm_src, table_val(one, bound));
    h->fmaxnm(vmm_src, vmm_src, table_val(zero, bound));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::hardswish_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta) {
    // result = x * hardsigmoid(x)
    auto src = reg_pool_.allocScoped<TRegS>();
    h->mov(VReg16B(src.getIdx()), VReg16B(vmm_src.getIdx()));
    hardsigmoid_compute_vector_fwd(vmm_src, alpha, beta);
    h->fmul(vmm_src, vmm_src, src);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::soft_relu_compute_vector_fwd(
        const TRegS &vmm_src) {
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto tmp = reg_pool_.allocScoped<TRegS>();
    auto work = reg_pool_.allocScoped<TRegS>();
    auto saved_src = reg_pool_.allocScoped<TRegS>();
    auto polynomial = reg_pool_.allocScoped<TRegS>();
    auto one_reg = reg_pool_.allocScoped<TRegS>();
    auto half_reg = reg_pool_.allocScoped<TRegS>();

    const auto &oneS = table_val(one, one_reg);
    const auto &halfS = table_val(half, half_reg);

    // alpha scaling
    if (alpha_ == 1.f) {
        // Nothing to do
    } else if (alpha_ == -1) {
        h->fneg(vmm_src, vmm_src);
    } else {
        h->fmul(vmm_src, vmm_src, table_val(alpha, table_value));
    }

    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

    // Keep src for further computations
    h->mov(VReg16B(IDX(saved_src)), VReg16B(IDX(vmm_src)));

    h->fminnm(vmm_src, table_val(exp_ln_flt_max_f, table_value), vmm_src);
    h->fmaxnm(vmm_src, table_val(exp_ln_flt_min_f, table_value), vmm_src);
    h->mov(VReg16B(IDX(work)), VReg16B(IDX(vmm_src)));

    // Calculate exp(x)
    // fx = x * log2ef + 0.5
    h->fmul(vmm_src, vmm_src, table_val(exp_log2ef, table_value));
    h->fadd(vmm_src, vmm_src, halfS);

    // tmp = floorf(fx)
    h->frintm(tmp, vmm_src);

    // Keep vmm_src = fx for further computations
    h->mov(VReg16B(IDX(vmm_src)), VReg16B(IDX(tmp)));

    // x = x - fx * ln2
    h->fmul(tmp, tmp, table_val(ln2f, table_value));
    h->fsub(work, work, tmp);
    // Compute exponent polynomial
    table_val(exp_pol, table_value, 4);
    h->fmla(table_val(exp_pol, polynomial, 3), table_value, work);
    h->fmla(table_val(exp_pol, table_value, 2), polynomial, work);
    h->fmla(table_val(exp_pol, polynomial, 1), table_value, work);
    h->fmla(table_val(exp_pol, table_value, 0), polynomial, work);
    h->mov(VReg16B(IDX(polynomial)), VReg16B(IDX(oneS)));
    h->fmla(polynomial, table_value, work);

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // Compute 2^-(n-1)
    h->fsub(work, vmm_src, oneS);
    h->fneg(work, work);
    h->fcvtzs(work, work);
    h->add(work, work, table_val(exponent_bias, table_value));
    h->shl(work, work, n_mantissa_bits);

    // Calculate ln(1 + y)
    h->fadd(polynomial, polynomial,
            polynomial); // 2*exp(r)
    h->fadd(polynomial, polynomial,
            work); // 2^-(n-1) + 2*exp(r)
    h->fmul(polynomial, polynomial,
            halfS); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    h->ushr(vmm_src, polynomial, n_mantissa_bits);
    h->scvtf(vmm_src, vmm_src);
    // Got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->fsub(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six, table_value));

    // And with mask (to get 0.5 * mantissa)
    h->and_(VReg16B(IDX(polynomial)), VReg16B(IDX(polynomial)),
            VReg16B(IDX(table_val(soft_relu_mantissa_sign_mask, table_value))));
    // Got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->orr(VReg16B(IDX(polynomial)), VReg16B(IDX(polynomial)),
            VReg16B(IDX(halfS)));
    // y  = y - 1
    h->fsub(polynomial, polynomial, oneS);

    // Compute log1p polynomial
    table_val(soft_relu_pol, work, 8);
    h->fmla(table_val(soft_relu_pol, table_value, 7), work, polynomial);
    h->fmla(table_val(soft_relu_pol, work, 6), table_value, polynomial);
    h->fmla(table_val(soft_relu_pol, table_value, 5), work, polynomial);
    h->fmla(table_val(soft_relu_pol, work, 4), table_value, polynomial);
    h->fmla(table_val(soft_relu_pol, table_value, 3), work, polynomial);
    h->fmla(table_val(soft_relu_pol, work, 2), table_value, polynomial);
    h->fmla(table_val(soft_relu_pol, table_value, 1), work, polynomial);
    h->fmla(table_val(soft_relu_pol, work, 0), table_value, polynomial);

    // Calculate ln(2) * n
    h->fmul(vmm_src, vmm_src, table_val(ln2f, table_value));
    h->fadd(vmm_src, vmm_src, work);
    h->fadd(vmm_src, vmm_src, tmp);

    // Get the mask for src > max logf.
    // y = (x < max log f) ? soft_relu(x) : x
    h->fcmgt(work, saved_src, table_val(exp_ln_flt_max_f, table_value));
    h->bit(VReg16B(IDX(vmm_src)), VReg16B(IDX(saved_src)), VReg16B(IDX(work)));
    if (alpha_ == 1.f) { // standard soft_relu case
        // Skip an instruction.
    } else if (alpha_ == -1) { // logsigmoid case
        h->fneg(vmm_src, vmm_src);
    } else if (alpha_ == 0.5) {
        h->fadd(vmm_src, vmm_src, vmm_src);
    } else if (alpha_ == 2.0) {
        h->fmul(vmm_src, vmm_src, halfS);
    } else { // General case.
        h->fdiv(vmm_src, vmm_src, table_val(alpha, table_value));
    }
}

template <>
void jit_uni_eltwise_injector_t<asimd>::logistic_compute_vector_fwd(
        const TRegS &vmm_src) {
    // To avoid exp(x) overflow happening at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic being a symmetric function.
    auto mask = reg_pool_.allocScoped<TRegS>();
    h->fcmgt(mask, vmm_src, 0.);

    // Force input negative
    {
        auto table_value = reg_pool_.allocScoped<TRegS>();
        h->orr(VReg16B(IDX(vmm_src)), VReg16B(IDX(vmm_src)),
                VReg16B(IDX(table_val(sign_mask, table_value.get()))));
    }

    // Compute exponent
    exp_compute_vector_fwd(vmm_src, min_input_, 0.f);

    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto denominator = reg_pool_.allocScoped<TRegS>();

    // (exp(x) + 1)
    h->fadd(denominator, vmm_src, table_val(one, table_value));
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_src, vmm_src, denominator);

    // Now we have to apply the "symmetry" based on original sign
    h->fsub(denominator, table_value, vmm_src);

    // Combine with mask
    h->bit(VReg16B(IDX(vmm_src)), VReg16B(IDX(denominator)),
            VReg16B(IDX(mask)));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::swish_compute_vector_fwd(
        const TRegS &vmm_src, const TRegS &alpha_log2e,
        const TRegS &exponent_control, const TRegS &, const TRegS &) {
    auto v_neg_r = reg_pool_.allocScoped<TRegS>();
    auto v_scale = reg_pool_.allocScoped<TRegS>();
    auto v_poly = reg_pool_.allocScoped<TRegS>();
    auto v_scratch = reg_pool_.allocScoped<TRegS>();

    // For a=alpha*x*log2(e)=k+r, ASIMD replaces FEXPA with an integer FP32
    // exponent construction and approximates 2^(-r) over r in [-0.5, 0.5].
    h->fmul(v_neg_r, vmm_src, alpha_log2e);
    // Clamp a before constructing 2^k from the IEEE-754 exponent field.
    h->fmaxnm(v_neg_r, v_neg_r, table_val(swish_polynomial_exp_min, v_poly));
    h->fminnm(v_neg_r, v_neg_r, table_val(swish_polynomial_exp_max, v_poly));
    // k = round_to_nearest_even(a), then -r = k-a.
    h->frintn(v_scale, v_neg_r);
    h->fsub(v_neg_r, v_scale, v_neg_r);

    // Construct 2^k as FP32 bits: (int32(k)+127)<<23.
    h->fcvtzs(v_scale, v_scale);
    h->add(v_scale, v_scale, exponent_control);
    h->shl(v_scale, v_scale, n_mantissa_bits);

    // Horner evaluation of the fitted quartic:
    //   2^(-r) ~= 1 + (-r)*(c1 + (-r)*(c2 + (-r)*(c3 + c4*(-r)))).
    table_val(swish_asimd_exp2_c4, v_poly);
    h->fmla(table_val(swish_asimd_exp2_c3, v_scratch), v_poly, v_neg_r);
    h->fmla(table_val(swish_asimd_exp2_c2, v_poly), v_scratch, v_neg_r);
    h->fmla(table_val(swish_asimd_exp2_c1, v_scratch), v_poly, v_neg_r);
    h->fmov(v_poly, 1.f);
    h->fmla(v_poly, v_scratch, v_neg_r);

    // x*2^k / (2^k + 2^(-r)).
    h->fmul(vmm_src, vmm_src, v_scale);
    h->fadd(v_neg_r, v_scale, v_poly);
    h->fdiv(vmm_src, vmm_src, v_neg_r);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::log_compute_vector_fwd(
        const TRegS &vmm_src) {
    // ------------------------------------------------------------------------
    // Based on ACL's vlogq_f32 implementation in core/NEON/NEMath.inl.
    //
    // For a positive normal f32 input, decompose:
    //
    //   x = 2^m * y, where y ∈ [1, 2)
    //
    // by extracting the IEEE-754 biased exponent:
    //
    //   m = exponent(x) - 127
    //
    // and constructing y from x's mantissa with exponent 127. Approximate
    // log(y) using ACL's degree-7 Taylor-style polynomial, evaluated as:
    //
    //   P(y) = (c0 + c4*y + (c2 + c6*y)*y^2)
    //        + (c1 + c5*y + (c3 + c7*y)*y^2)*y^4
    //
    // Then reconstruct:
    //
    //   log(x) ≈ P(y) + m * ln(2)
    //
    // Special cases are selected after the normal-input computation:
    //   log(±0)  = -inf
    //   log(x<0) = qNaN
    //   log(+inf) = +inf
    //   NaN inputs are propagated.
    //
    // The ACL source does not state a ULP accuracy bound.
    // ------------------------------------------------------------------------
    const auto &t0 = vmm_src;
    auto t_tmp = reg_pool_.allocScoped<TRegS>();
    auto t1 = reg_pool_.allocScoped<TRegS>();
    auto t2 = reg_pool_.allocScoped<TRegS>();
    auto t3 = reg_pool_.allocScoped<TRegS>();
    auto t4 = reg_pool_.allocScoped<TRegS>();
    auto t5 = reg_pool_.allocScoped<TRegS>();
    auto t6 = reg_pool_.allocScoped<TRegS>();

    // Preserve a copy of the original to mask special cases
    const auto &v_orig = t6;
    h->mov(VReg16B(v_orig.getIdx()), VReg16B(t0.getIdx()));

    // int32x4_t m = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
    const auto &v_const_127 = table_val(log_const_127, t_tmp); // 127
    const auto &v_m = t5;
    h->ushr(v_m, t0, n_mantissa_bits);
    h->sub(v_m, v_m, v_const_127);

    // float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));
    const auto &v_val = t1;
    h->shl(v_val, v_m, n_mantissa_bits);
    h->sub(v_val, t0, v_val);

    // vtaylor_polyq_f32(...):

    // float32x4_t A = vmlaq_f32(coeffs[0], coeffs[4], x);
    const auto &v_A = t0;
    const auto &v_c4 = t2;
    table_val(log_pol, v_A, 0);
    table_val(log_pol, v_c4, 4);
    h->fmla(v_A, v_c4, v_val);

    // float32x4_t B = vmlaq_f32(coeffs[2], coeffs[6], x);
    const auto &v_B = t2;
    const auto &v_c6 = t3;
    table_val(log_pol, v_B, 2);
    table_val(log_pol, v_c6, 6);
    h->fmla(v_B, v_c6, v_val);

    // float32x4_t x2  = vmulq_f32(x, x);
    const auto &v_x2 = t_tmp;
    h->fmul(v_x2, v_val, v_val);

    // float32x4_t tmp1 = vmlaq_f32(A, B, x2)
    const auto &v_tmp1 = v_A;
    h->fmla(v_tmp1, v_B, v_x2);

    // float32x4_t C = vmlaq_f32(coeffs[1], coeffs[5], x);
    const auto &v_C = t2;
    const auto &v_c1 = t3;
    table_val(log_pol, v_C, 1);
    table_val(log_pol, v_c1, 5);
    h->fmla(v_C, v_c1, v_val);

    // float32x4_t D = vmlaq_f32(coeffs[3], coeffs[7], x);
    const auto &v_D = t3;
    const auto &v_c7 = t4;
    table_val(log_pol, v_D, 3);
    table_val(log_pol, v_c7, 7);
    h->fmla(v_D, v_c7, v_val);

    // float32x4_t tmp2 = vmlaq_f32(C, D, x2)
    const auto &v_tmp2 = v_C;
    h->fmla(v_tmp2, v_D, v_x2);

    // float32x4_t x4  = vmulq_f32(x2, x2);
    const auto &v_x4 = v_x2;
    h->fmul(v_x4, v_x2, v_x2);

    // float32x4_t res = vmlaq_f32(tmp1, tmp2, x4);
    const auto &v_poly = v_tmp1;
    h->fmla(v_poly, v_tmp2, v_x4);

    // Back to vlogq_f32:

    // poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);
    const auto &v_ln2 = table_val(log_ln2, t_tmp); // ln(2)
    h->scvtf(v_m, v_m);
    h->fmla(v_poly, v_m, v_ln2);

    // Special case handling:

    const auto &v_mask = t1;
    const auto &v_new_val = t2;

    // x = 0 --> log(x) = -inf
    h->fcmeq(v_mask, v_orig, 0.0);
    table_val(log_minus_inf, v_new_val);
    h->bit(VReg16B(IDX(v_poly)), VReg16B(IDX(v_new_val)), VReg16B(IDX(v_mask)));

    // x < 0 --> log(x) = NaN
    h->fcmlt(v_mask, v_orig, 0.0);
    table_val(log_qnan, v_new_val);
    h->bit(VReg16B(IDX(v_poly)), VReg16B(IDX(v_new_val)), VReg16B(IDX(v_mask)));

    // x = inf --> log(x) = +inf
    table_val(log_inf, v_new_val);
    h->fcmeq(v_mask, v_orig, v_new_val);
    h->bit(VReg16B(IDX(v_poly)), VReg16B(IDX(v_new_val)), VReg16B(IDX(v_mask)));

    // x = NaN --> log(x) = NaN
    h->fcmeq(v_mask, v_orig, v_orig);
    h->not_(VReg16B(IDX(v_mask)), VReg16B(IDX(v_mask)));
    h->bit(VReg16B(IDX(v_poly)), VReg16B(IDX(v_orig)), VReg16B(IDX(v_mask)));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::gelu_erf_compute_vector_fwd(
        const TRegS &vmm_src) {
    auto table_value = reg_pool_.allocScoped<TRegS>();
    auto aux0 = reg_pool_.allocScoped<TRegS>();
    auto aux1 = reg_pool_.allocScoped<TRegS>();
    auto aux2 = reg_pool_.allocScoped<TRegS>();
    auto aux3 = reg_pool_.allocScoped<TRegS>();
    auto aux4 = reg_pool_.allocScoped<TRegS>();
    auto aux5 = reg_pool_.allocScoped<TRegS>();
    // x = s / sqrt(2)
    h->fmul(aux1, vmm_src, table_val(gelu_erf_one_over_sqrt_two, table_value));

    // abs(x)
    h->fabs(aux1, aux1);

    // LUT-based ERF based on the verfq_f32 implementation from Arm Compute Library (ACL)
    // Maximum error: 1.93 ULP
    //
    // erf(x) for x in [0, 3.9375] is approxiated as follows:
    //
    //   erf(x) = erf(r) + scale(r) * d * (1 - r * d - 1/3 * d^2)
    //
    // where:
    //   r = round(x * 128) / 128
    //   d = x - r
    //
    // erf(r) and scale(r) are stored in a 513-entry lookup table.
    // The LUT covers the range from 0 to 4 with the step of 1/128.
    //
    // Special cases:
    //   erf(x) =  1 for x >  3.9375
    //   erf(x) = -1 for x < -3.9375

    // Find LUT indices by quantizing the LUT input to the 1/128 grid.
    // Adding 2^16 moves the sum into the [2^16, 2^17) f32 range, where the ULP is
    // 2^(16 - 23) = 1/128, so the add rounds to the nearest 1/128 step.
    // The floating-point subtraction recovers the quantized value, and the integer
    // subtraction of the same bit pattern yields the corresponding LUT index.
    h->fadd(aux2, aux1, table_val(gelu_erf_lut_bias, table_value));
    h->fsub(aux3, aux2, table_value); // r
    h->sub(aux5, aux2, table_value); // raw index
    h->umin(aux5, aux5,
            table_val(gelu_erf_lut_max_index, table_value)); // index

    // LUT address = table_base + index * sizeof(float) * 2 (erf and scale)
    {
        auto x_lut_base = reg_pool_.allocScoped<XReg>();
        auto x_idx = reg_pool_.allocScoped<XReg>();
        h->add_imm(x_lut_base, x_table, table_off(gelu_erf_lut), x_idx);
        auto load_lut_entry = [&](const int lane, const DReg &dst) {
            h->umov(WReg(IDX(x_idx)), aux5[lane]);
            h->ldr(dst, ptr(x_lut_base, x_idx, LSL, 3));
        };

        // Lookup erf(r) and scale(r) pairs
        load_lut_entry(0, DReg(IDX(aux0))); // idx0
        load_lut_entry(1, DReg(IDX(aux4))); // idx1
        load_lut_entry(2, DReg(IDX(aux2))); // idx2
        load_lut_entry(3, DReg(IDX(aux5))); // idx3
    }

    // Build entry_01=[idx0, idx1] and entry_23=[idx2, idx3].
    h->ins(VReg2D(IDX(aux0))[1], VReg2D(IDX(aux4))[0]);
    h->ins(VReg2D(IDX(aux2))[1], VReg2D(IDX(aux5))[0]);

    // {erf0, s0, erf1, s1}, {erf2, s2, erf3, s3} -> {erf0, erf1, erf2, erf3}, {s0, s1, s2, s3}
    h->uzp2(aux5, aux0, aux2); // scale_r
    h->uzp1(aux0, aux0, aux2); // erf_r

    // erf(x) ~= erf(r) + scale(r) * d * (1 - r * d - 1/3 * d^2).
    h->fsub(aux4, aux1, aux3); // d
    h->fmul(aux2, aux4, aux4); // d^2
    h->fmla(aux3, table_val(gelu_erf_lut_third, table_value),
            aux4); // t0 = r + 1/3 * d.
    h->fmls(aux4, aux2, aux3); // t1 = d - d^2 * t0
    h->fmla(aux0, aux5, aux4); // erf(|x|)

    // Clamp erf(|x|) for |x| > 3.9375 to 1.
    h->fcmgt(aux4, aux1, table_val(gelu_erf_lut_max, table_value));
    h->bit(VReg16B(IDX(aux0)), VReg16B(IDX(table_val(one, table_value))),
            VReg16B(IDX(aux4)));

    // Restore sign using original s
    h->and_(VReg16B(IDX(aux4)), VReg16B(IDX(vmm_src)),
            VReg16B(IDX(table_val(sign_mask, table_value))));
    h->eor(VReg16B(IDX(aux0)), VReg16B(IDX(aux0)), VReg16B(IDX(aux4)));

    // GELU = 0.5 * s * (1 + erf)
    h->fmul(vmm_src, vmm_src, table_val(half, table_value));
    h->fmla(vmm_src, aux0, vmm_src);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::round_compute_vector_fwd(
        const TRegS &vmm_src) {
    h->frintn(vmm_src, vmm_src);
}

template <>
void jit_uni_eltwise_injector_t<asimd>::load_1word_replicate(
        const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr) {
    h->ld1r(TRegS(vmm_src.getIdx()), ptr(x_addr));
}
template <>
void jit_uni_eltwise_injector_t<sve>::load_1word_replicate(
        const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr) {
    h->ld1rw(TRegS(vmm_src.getIdx()), p_all, ptr(x_addr));
}

template <>
void jit_uni_eltwise_injector_t<asimd>::load_vector(
        const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr) {
    h->ldr(QReg(vmm_src.getIdx()), ptr(x_addr));
}
template <>
void jit_uni_eltwise_injector_t<sve>::load_vector(
        const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr) {
    h->ldr(TReg(vmm_src.getIdx()), ptr(x_addr));
}

#define DEFINE_ASIMD_EMPTY_FUNC(func_name) \
    template <> \
    void jit_uni_eltwise_injector_t<asimd>::func_name(const TRegS &) {}
DEFINE_ASIMD_EMPTY_FUNC(tanh_polynomial_approx_compute_vector_fwd);
DEFINE_ASIMD_EMPTY_FUNC(gelu_erf_minimax_approx_compute_vector_fwd);

DEFINE_ASIMD_EMPTY_FUNC(elu_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(tanh_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(gelu_tanh_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(square_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(abs_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(sqrt_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(mish_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(logistic_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(exp_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(swish_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(log_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC(gelu_erf_compute_vector_bwd);
#undef DEFINE_ASIMD_EMPTY_FUNC

#define DEFINE_ASIMD_EMPTY_FUNC_2(func_name) \
    template <> \
    void jit_uni_eltwise_injector_t<asimd>::func_name( \
            const TRegS &, const TRegS &) {}
DEFINE_ASIMD_EMPTY_FUNC_2(relu_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC_2(linear_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC_2(soft_relu_compute_vector_bwd);
#undef DEFINE_ASIMD_EMPTY_FUNC_2

#define DEFINE_ASIMD_EMPTY_FUNC_3(func_name) \
    template <> \
    void jit_uni_eltwise_injector_t<asimd>::func_name( \
            const TRegS &, const TRegS &, const TRegS &) {}
DEFINE_ASIMD_EMPTY_FUNC_3(clip_compute_vector_bwd);
#undef DEFINE_ASIMD_EMPTY_FUNC_3

#define DEFINE_ASIMD_EMPTY_FUNC_4(func_name) \
    template <> \
    void jit_uni_eltwise_injector_t<asimd>::func_name( \
            const TRegS &, const TRegS &, const TRegS &, const TRegS &) {}
DEFINE_ASIMD_EMPTY_FUNC_4(hardswish_compute_vector_bwd);
DEFINE_ASIMD_EMPTY_FUNC_4(hardsigmoid_compute_vector_bwd);
#undef DEFINE_ASIMD_EMPTY_FUNC_4

template <>
void jit_uni_eltwise_injector_t<asimd>::compute_cmp_mask(const TRegS &vmm_src,
        const TRegS &compare_operand, alg_kind_t cmp_predicate) {}

template <>
void jit_uni_eltwise_injector_t<asimd>::blend_with_mask(
        const TRegS &vmm_dst, const TRegS &src) {}

template struct jit_uni_eltwise_injector_t<sve>;
template struct jit_uni_eltwise_injector_t<asimd>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
