/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP
#define CPU_AARCH64_INJECTORS_JIT_UNI_ELTWISE_INJECTOR_HPP

#include <assert.h>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace eltwise_injector {
/*
 * Checks if isa is supported by eltwise injector.
 */
bool is_isa_supported(cpu_isa_t isa);

/*
 * Checks if eltwise algorithm is supported by eltwise injector.
 */
bool is_alg_supported(alg_kind_t alg);

/*
 * Checks if eltwise injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, alg_kind_t alg);

} // namespace eltwise_injector

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_t {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;
    using TRegH = typename cpu_isa_traits<isa>::TRegH;

    // The caller must claim every register live across an injector call in
    // reg_pool. All injector scratch is allocated from that pool.
    jit_uni_eltwise_injector_t(jit_generator_t *host,
            Xbyak_aarch64::RegPoolManager &reg_pool, alg_kind_t alg,
            float alpha, float beta, float scale, Xbyak_aarch64::XReg x_table,
            Xbyak_aarch64::PReg p_mask, bool is_fwd = true,
            bool use_dst = false, data_type_t d_type = data_type::f32)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , h(host)
        , x_table(x_table)
        , p_mask(p_mask)
        , is_fwd_(is_fwd)
        , use_dst_(use_dst)
        , d_type_(d_type)
        , reg_pool_(reg_pool) {
        assert(eltwise_injector::is_supported(isa, alg_));

        p_all = h->P_ALL_ONE;
        register_table_entries();
    }

    jit_uni_eltwise_injector_t(jit_generator_t *host,
            Xbyak_aarch64::RegPoolManager &reg_pool,
            const post_ops_t::entry_t::eltwise_t &eltwise,
            Xbyak_aarch64::XReg x_table, Xbyak_aarch64::PReg p_mask,
            bool is_fwd = true, bool use_dst = false,
            data_type_t d_type = data_type::f32)
        : jit_uni_eltwise_injector_t(host, reg_pool, eltwise.alg, eltwise.alpha,
                  eltwise.beta, eltwise.scale, x_table, p_mask, is_fwd, use_dst,
                  d_type) {}

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs);
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void prepare_table(bool gen_table = true);
    void load_table_addr() { h->adr(x_table, l_table); }
    void set_input_range(float min_value, float max_value);
    size_t scratch_vmm_count() const;

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;
    float max_input_ = INFINITY;
    float min_input_ = -INFINITY;

    jit_generator_t *const h;

    const Xbyak_aarch64::XReg x_table;
    Xbyak_aarch64::PReg p_all = Xbyak_aarch64::PReg(99);
    const Xbyak_aarch64::PReg p_mask;
    const bool is_fwd_;
    const bool use_dst_;

    Xbyak_aarch64::Label l_table;

    const data_type_t d_type_;

    const size_t vlen = get_vec_len();
    static constexpr size_t vecs_count = cpu_isa_traits<isa>::n_vregs;
    static constexpr int n_mantissa_bits = 23;
    static constexpr int k_mask_size = 8;

    Xbyak_aarch64::RegPoolManager &reg_pool_;
    mutable size_t scratch_vmm_count_ = static_cast<size_t>(-1);

    void compute_body(const injector_utils::vmm_index_set_t &vmm_idxs);
    void compute_vector_range_no_spill(
            const injector_utils::vmm_index_set_t &vmm_idxs);
    void compute_cmp_mask(const TRegS &vmm_src, const TRegS &vmm_cmpare,
            alg_kind_t cmp_predicate);
    void blend_with_mask(const TRegS &vmm_dst, const TRegS &src);

    size_t get_vec_len();
    void exp_compute_vector_fwd(const TRegS &vmm_src);
    void exp_compute_vector_fwd(
            const TRegS &vmm_src, float min_input, float max_input);
    void relu_compute_vector_fwd(const TRegS &vmm_src, const TRegS &alpha);
    void relu_zero_ns_compute_vector_fwd(const TReg &vmm_src);
    void elu_compute_vector_fwd(const TRegS &vmm_src, const TRegS &alpha);
    void tanh_compute_vector_fwd(const TRegS &vmm_src);
    void tanh_polynomial_approx_compute_vector_fwd(const TRegS &vmm_src);
    void square_compute_vector_fwd(const TReg &vmm_src);
    void abs_compute_vector_fwd(const TReg &vmm_src);
    void sqrt_compute_vector_fwd(const TReg &vmm_src);
    void linear_compute_vector_fwd(
            const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta);
    void soft_relu_compute_vector_fwd(const TRegS &vmm_src);
    void mish_compute_vector_fwd(const TRegS &vmm_src);
    void logistic_compute_vector_fwd(const TRegS &vmm_src);
    void gelu_tanh_compute_vector_fwd(const TRegS &vmm_src);
    void swish_compute_vector_fwd(const TRegS &vmm_src,
            const TRegS &alpha_log2e, const TRegS &exponent_control,
            const TRegS &polynomial_c1, const TRegS &polynomial_c2);
    void log_compute_vector_fwd(const TRegS &vmm_src);
    void clip_compute_vector_fwd(
            const TReg &vmm_src, const TRegS &lower, const TRegS &upper);
    void gelu_erf_compute_vector_fwd(const TRegS &vmm_src);
    void gelu_erf_minimax_approx_compute_vector_fwd(const TRegS &vmm_src);
    void round_compute_vector_fwd(const TRegS &vmm_src);
    void hardswish_compute_vector_fwd(
            const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta);
    void hardsigmoid_compute_vector_fwd(
            const TRegS &vmm_src, const TRegS &alpha, const TRegS &beta);

    void exp_compute_vector_bwd(const TRegS &vmm_src);
    void relu_compute_vector_bwd(const TRegS &vmm_src, const TRegS &alpha);
    void elu_compute_vector_bwd(const TRegS &vmm_src);
    void tanh_compute_vector_bwd(const TRegS &vmm_src);
    void square_compute_vector_bwd(const TRegS &vmm_src);
    void abs_compute_vector_bwd(const TRegS &vmm_src);
    void sqrt_compute_vector_bwd(const TRegS &vmm_src);
    void linear_compute_vector_bwd(const TRegS &vmm_src, const TRegS &alpha);
    void soft_relu_compute_vector_bwd(const TRegS &vmm_src, const TRegS &alpha);
    void logistic_compute_vector_bwd(const TRegS &vmm_src);
    void mish_compute_vector_bwd(const TRegS &vmm_src);
    void gelu_tanh_compute_vector_bwd(const TRegS &vmm_src);
    void swish_compute_vector_bwd(const TRegS &vmm_src);
    void log_compute_vector_bwd(const TRegS &vmm_src);
    void clip_compute_vector_bwd(
            const TRegS &vmm_src, const TRegS &lower, const TRegS &upper);
    void gelu_erf_compute_vector_bwd(const TRegS &vmm_src);
    void hardswish_compute_vector_bwd(const TRegS &vmm_src, const TRegS &alpha,
            const TRegS &beta, const TRegS &one);
    void hardsigmoid_compute_vector_bwd(const TRegS &vmm_src,
            const TRegS &alpha, const TRegS &beta, const TRegS &one);
    void load_1word_replicate(const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr);
    void load_vector(const TRegS &vmm_src, Xbyak_aarch64::XReg x_addr);

    enum key_t {
        scale = 0, // scale argument
        alpha, // alpha argument
        beta, // beta argument
        zero, // 0.f
        half, // 0.5f
        one, // 1.f  or  mask for exponent bits
        two, // 2.f
        minus_one, // -1.f  or  changes sign to opposite
        minus_two, // -2.f
        ln2f, // 0.69314718f
        positive_mask, // changes sign to positive
        sign_mask, // gets sign value
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        exp_pol_asimd, // see corresponding table for float values
        exp_coeff1, // 0.6931473921 (0x3f31721c)
        exp_coeff2, // 0.2413862043 (0x3e772df2)
        exp_exponent_bias,
        exp_neg_ln2_hi, // high part of -ln(2) used for exp calculation
        exp_neg_ln2_lo, // low part of -ln(2) used for exp calculation
        exp_special_bound, // bound for exp special case handling
        exp_scale_thresh, // threshold for additional overflow/underflow handling
        exp_special_offset,
        exp_special_bias,
        exp_not_mask17, // ~((1u << 17) - 1)
        fwd_mish_max_x_for_equation_f,
        // e^x(e^3x+4e^2x+e^x*(6+4*x)+4*(1+x)) = FLT_MAX; x =~ 22.18070976278534
        bwd_mish_max_x_for_equation_f,
        tanh_range, // tanh(x) = x - x^3/3 for |x| < tanh_range
        tanh_m1d3, // -1/3
        tanh_idx_bias, // bias applied during index computation
        tanh_idx_mask, // mask applied to extract index
        tanh_linear_ubound, // arg below which tanh(x) = x
        tanh_saturation_lbound, // arg after which tanh(x) = 1.f
        tanh_pol_table, // table of polynomial coefficients
        soft_relu_one_twenty_six, // 126.f
        soft_relu_mantissa_sign_mask, // mask for mantissa bits and sign
        soft_relu_pol, // see correspondent table for float values
        gelu_tanh_fitting_const, // 0.044715f
        gelu_tanh_fitting_const_times_three, // 0.134145f
        gelu_tanh_sqrt_two_over_pi, // sqrtf(2.f/pi) = 0.797884f
        gelu_erf_idx_bias, // bias applied to compute table index
        gelu_erf_rbound, // upper bound at which we clamp erf at 1
        gelu_erf_one, // just the integer value 1, used for index clamping
        gelu_erf_twenty_three, // just the integer value 23, used for index clamping
        gelu_erf_twenty_four, // just the integer value 24, used for index clamping
        gelu_erf_minimax_pol, // see correspondent table for float values
        gelu_erf_approx_const, // 0.3275911f - implementation based for approx
        gelu_erf_one_over_sqrt_two, // 1.f / sqrtf(2.f)
        gelu_erf_one_over_sqrt_pi, // 1.f / sqrtf(pi) = 0.564190f
        gelu_erf_pol, // see correspondent table for float values
        gelu_erf_lut_max,
        gelu_erf_lut_third,
        gelu_erf_lut_bias,
        gelu_erf_lut_max_index,
        gelu_erf_lut, // ERF LUT: [erf(r), scale(r)], 513 pairs
        gelu_erf_lut_erf, // ERF LUT: erf(r), 513 entries
        gelu_erf_lut_scale, // ERF LUT: scale(r), 513 entries
        swish_alpha_log2e,
        swish_fexpa_shift, // 131199.f
        swish_polynomial_exp_c1, // fitted coefficient for t = -r
        swish_polynomial_exp_c2, // fitted coefficient for t^2
        swish_polynomial_exp_min, // -32 * log2(e)
        swish_polynomial_exp_max, // 32 * log2(e)
        swish_asimd_exp2_c1,
        swish_asimd_exp2_c2,
        swish_asimd_exp2_c3,
        swish_asimd_exp2_c4,
        log_const_127, // 127
        log_ln2, // ln(2) = 0.6931471805599453
        log_inf, // +inf
        log_minus_inf, // -inf
        log_qnan, // qnan
        log_mantissa_mask, // gets mantissa bits
        log_full_k_reg_mask, // sets k_register with all bits of 1
        log_full_vector_reg_mask, // sets vector register will all bits of 1
        log_five_bit_offset, // 5 bits off (31 = 2^5 - 1)
        log_pol, // see correspondent table for float values
        log_predefined_vals, // see correspondent table for float values
        log_i127shl23,
        log_x7fffff,
        log_log2,
        log_log1p5,
        log_f2div3,
        log_coeffTbl,
        undef_key,
    };

    size_t table_off(key_t key, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        // A key can have several entries. std::multimap::find() may return
        // any of them, so use lower_bound() to get the first one.
        const auto it = entry_map_.lower_bound(key);
        assert(it != entry_map_.end() && it->first == key);
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }

    TRegS table_val(key_t key, TRegS zreg, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // A key can have several entries. std::multimap::find() may return
        // any of them, so use lower_bound() to get the first one.
        const auto it = entry_map_.lower_bound(key);
        assert(it != entry_map_.end() && it->first == key);
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen : sizeof(table_entry_val_t);
        const auto off = te.off + key_off_val_shift * scale;

        const auto load = [&](const Xbyak_aarch64::XReg &table_addr) {
            if (te.bcast)
                load_vector(zreg, table_addr);
            else
                load_1word_replicate(zreg, table_addr);
        };
        if (off == 0) {
            load(x_table);
        } else {
            auto table_addr = reg_pool_.allocScoped<Xbyak_aarch64::XReg>();
            h->add_imm(table_addr.get(), x_table, off, table_addr.get());
            load(table_addr.get());
        }
        return zreg;
    }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<key_t, table_entry_t>;
    using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;

    void register_table_entries();
    mapped_table_t entry_map_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
