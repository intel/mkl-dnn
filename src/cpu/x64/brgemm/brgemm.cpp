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

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_utils.hpp"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/brgemm/brgemv_ir.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"
#include "cpu/x64/brgemm/jit_brgemm_amx_uker.hpp"
#include "cpu/x64/brgemm/jit_brgemm_kernel.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;
using namespace brgemm_utils;

brgemm_desc_t::brgemm_desc_t(const brgemm_desc_t &other) {
    *this = other;
    // Since copy above will make `attr_` and `dst_md_` point to `other`,
    // nulling them in `this` to avoid cleaning `other` object members.

    set_attr_null();
    set_dst_md_null();

    set_attr(other.attr());
    set_dst_md(other.dst_md());
}

brgemm_desc_t::~brgemm_desc_t() {
    cleanup_attr();
    cleanup_dst_md();
}

void brgemm_desc_t::set_attr(const primitive_attr_t *ppdattr) {
    if (ppdattr == attr_) return;
    cleanup_attr();
    if (ppdattr) attr_ = new primitive_attr_t(*ppdattr);
}

void brgemm_desc_t::set_dst_md(const memory_desc_t *pdst_md) {
    if (pdst_md == dst_md_) return;
    cleanup_dst_md();
    if (pdst_md) dst_md_ = new memory_desc_t(*pdst_md);
}

void brgemm_desc_t::cleanup_attr() {
    if (attr_ == nullptr) return;
    delete attr_;
    attr_ = nullptr;
}

void brgemm_desc_t::cleanup_dst_md() {
    if (dst_md_ == nullptr) return;
    delete dst_md_;
    dst_md_ = nullptr;
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, dim_t bs,
        const brgemm_batch_element_t *batch, void *ptr_C, void *scratch,
        const brgemm_dynamic_values_t *dynamic_values) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.do_apply_comp = 0;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;
    if (dynamic_values) {
        brgemm_p.dynamic_LDA = dynamic_values->dynamic_LDA;
        brgemm_p.dynamic_LDB = dynamic_values->dynamic_LDB;
        brgemm_p.dynamic_LDC = dynamic_values->dynamic_LDC;
        brgemm_p.dynamic_LDD = dynamic_values->dynamic_LDD;
    }

    assert(brg_kernel);

    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, dim_t bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *scratch,
        const brgemm_dynamic_values_t *dynamic_values) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.do_apply_comp = 0;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;
    if (dynamic_values) {
        brgemm_p.dynamic_LDA = dynamic_values->dynamic_LDA;
        brgemm_p.dynamic_LDB = dynamic_values->dynamic_LDB;
        brgemm_p.dynamic_LDC = dynamic_values->dynamic_LDC;
        brgemm_p.dynamic_LDD = dynamic_values->dynamic_LDD;
    }

    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, dim_t bs,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch,
        const brgemm_dynamic_values_t *dynamic_values) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = post_ops_data.bias;
    brgemm_p.ptr_src_scales = post_ops_data.src_scales;
    brgemm_p.ptr_wei_scales = post_ops_data.wei_scales;
    brgemm_p.ptr_dst_scales = post_ops_data.dst_scales;
    brgemm_p.do_post_ops
            = post_ops_data.do_only_comp || post_ops_data.do_only_zp_a_val ? 0
                                                                           : 1;
    brgemm_p.do_apply_comp = post_ops_data.do_only_zp_a_val ? 0 : 1;
    brgemm_p.skip_accm = post_ops_data.skip_accumulation ? 1 : 0;
    brgemm_p.BS = bs;
    brgemm_p.zp_a_val = post_ops_data.zp_a_val;
    brgemm_p.post_ops_binary_rhs_arg_vec = post_ops_data.binary_post_ops_rhs;
    brgemm_p.oc_logical_off = post_ops_data.oc_logical_off;
    brgemm_p.dst_row_logical_off = post_ops_data.dst_row_logical_off;
    brgemm_p.data_C_ptr_ = post_ops_data.data_C_ptr_;
    brgemm_p.first_mb_matrix_addr_off = post_ops_data.first_mb_matrix_addr_off;
    brgemm_p.a_zp_compensations = post_ops_data.a_zp_compensations;
    brgemm_p.b_zp_compensations = post_ops_data.b_zp_compensations;
    brgemm_p.c_zp_values = post_ops_data.c_zp_values;
    brgemm_p.ptr_per_mn_compensation = post_ops_data.per_mn_compensation;
    if (dynamic_values) {
        brgemm_p.dynamic_LDA = dynamic_values->dynamic_LDA;
        brgemm_p.dynamic_LDB = dynamic_values->dynamic_LDB;
        brgemm_p.dynamic_LDC = dynamic_values->dynamic_LDC;
        brgemm_p.dynamic_LDD = dynamic_values->dynamic_LDD;
    }

    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, dim_t bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch,
        const brgemm_dynamic_values_t *dynamic_values) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = post_ops_data.bias;
    brgemm_p.ptr_src_scales = post_ops_data.src_scales;
    brgemm_p.ptr_wei_scales = post_ops_data.wei_scales;
    brgemm_p.ptr_dst_scales = post_ops_data.dst_scales;
    brgemm_p.do_post_ops
            = post_ops_data.do_only_comp || post_ops_data.do_only_zp_a_val ? 0
                                                                           : 1;
    brgemm_p.do_apply_comp = post_ops_data.do_only_zp_a_val ? 0 : 1;
    brgemm_p.skip_accm = post_ops_data.skip_accumulation ? 1 : 0;
    brgemm_p.BS = bs;
    brgemm_p.zp_a_val = post_ops_data.zp_a_val;
    brgemm_p.post_ops_binary_rhs_arg_vec = post_ops_data.binary_post_ops_rhs;
    brgemm_p.oc_logical_off = post_ops_data.oc_logical_off;
    brgemm_p.dst_row_logical_off = post_ops_data.dst_row_logical_off;
    brgemm_p.data_C_ptr_ = post_ops_data.data_C_ptr_;
    brgemm_p.first_mb_matrix_addr_off = post_ops_data.first_mb_matrix_addr_off;
    brgemm_p.a_zp_compensations = post_ops_data.a_zp_compensations;
    brgemm_p.b_zp_compensations = post_ops_data.b_zp_compensations;
    brgemm_p.a_zp_values = post_ops_data.a_zp_values;
    brgemm_p.c_zp_values = post_ops_data.c_zp_values;
    brgemm_p.ptr_per_mn_compensation = post_ops_data.per_mn_compensation;
    if (dynamic_values) {
        brgemm_p.dynamic_LDA = dynamic_values->dynamic_LDA;
        brgemm_p.dynamic_LDB = dynamic_values->dynamic_LDB;
        brgemm_p.dynamic_LDC = dynamic_values->dynamic_LDC;
        brgemm_p.dynamic_LDD = dynamic_values->dynamic_LDD;
    }

    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

status_t brgemm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, bool transB,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K, const brgemm_strides_t *strides) {
    /*
    m - number of rows of the matrix op(A) and number of rows of the matrix C
    n - number of columns of the matrix op(B) and number of columns of the matrix C
    k - number of columns of the matrix op(A) and number of rows of the matrix op(B)

    Matrices are in row-major layouts:
        A: lda * m, LDA - lda must be at least max(1, k)
        B: ldb * k, LDB - ldb must be at least max(1, n)
        C: ldc * m, LDC - ldc must be at least max(1, n)

    Matrices are in column-major layouts:
        A: lda * k, LDA - lda must be at least max(1, m)
        B: ldb * n, LDB - ldb must be at least max(1, k)
        C: ldc * n, LDC - ldc must be at least max(1, m)
    */
    if (brg == nullptr) return status::invalid_arguments;
    if (transA || transB) return status::unimplemented;
    if (type == brgemm_batch_kind_t::brgemm_batch_kind_undef)
        return status::invalid_arguments;

    CHECK(brgemm_utils::init_brgemm_conf(brg, isa, type, dt_a, dt_b, layout,
            alpha, beta, LDA, LDB, LDC, M, N, K, strides, false /* is_bf32 */));

    if (utils::one_of(true, brg->is_runtime_lda, brg->is_runtime_ldb))
        return status::unimplemented;

    if (M <= 0 || N <= 0 || K <= 0) return status::invalid_arguments;

    // Upper bound, this can likely be improved by accounting for blocking
    dim_t max_a_stride = brg->LDA * types::data_type_size(brg->dt_a)
            * (brg->layout == brgemm_col_major ? K : M);
    dim_t max_b_stride = brg->LDB * types::data_type_size(brg->dt_b)
            * (brg->layout == brgemm_col_major ? N : K);
    dim_t max_c_stride = brg->LDC * types::data_type_size(brg->dt_c)
            * (brg->layout == brgemm_col_major ? N : M);

    // Required for EVEX encoding for offsets
    const dim_t max_stride = std::numeric_limits<int32_t>::max();
    if ((max_a_stride > max_stride && !brg->is_runtime_lda)
            || (max_b_stride > max_stride && !brg->is_runtime_ldb)
            || (max_c_stride >= max_stride && !brg->is_runtime_ldc))
        return status::unimplemented;

    // u8 weights need an outer product taking an unsigned second operand,
    // which only the tile ISAs have. This is an early ISA screen; whether a
    // tile kernel is really used is re-checked in brgemm_desc_finalize().
    if (!IMPLICATION(brg->dt_b == u8,
                is_superset(brg->isa_impl, avx512_core_amx)
                        || is_superset(brg->isa_impl, avx10_2_ace)))
        return status::unimplemented;

    return status::success;
}

status_t brgemv_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_x, bool transA, float alpha, float beta, dim_t LDA,
        dim_t INCY, dim_t M, dim_t N, bool treat_y_as_row) {
    using namespace data_type;

    // GEMV uses f32 accumulators with upconvert from the input dt; both
    // operands must share the same dt.
    if (!utils::one_of(dt_a, f32, bf16, f16) || dt_a != dt_x)
        return status::unimplemented;

    // The transA kernel requires the output to be contiguous.
    if (transA && INCY > 1) return status::unimplemented;

    // Set `is_gemv` before `brgemm_desc_init` so the GEMV branch in
    // `set_isa_impl` (and other downstream code) can take effect.
    brg->is_gemv = true;

    CHECK(brgemm_desc_init(brg, isa, type, dt_a, dt_x, false, false,
            brgemm_row_major, alpha, beta, LDA, 1, INCY, M, 1, N, nullptr));

    // Initialize transA here because `brgemm_desc_init` would otherwise return
    // unimplemented since brgemm doesn't support this case and we don't pass
    // the `is_gemv` flag to `brgemm_desc_init`. This keeps the GEMV
    // initialization logic in one place.
    brg->transA = transA;
    brg->treat_y_as_row = treat_y_as_row;

    return status::success;
}

status_t brdgmm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    if (brg == nullptr) return status::invalid_arguments;
    if (transA || layout != brgemm_row_major || alpha != 1.0f || beta != 0.f)
        return status::unimplemented;

    brgemm_utils::init_brdgmm_conf(brg, isa, type, dt_a, dt_b, layout, alpha,
            beta, LDA, LDC, M, N, strides);

    const bool ldx_check = (LDA < N || LDC < N);
    if (ldx_check) return status::invalid_arguments;

    if (utils::everyone_is(
                false, brg->is_int8, brg->is_bf16, brg->is_f32, brg->is_f16))
        return status::unimplemented;

    return status::success;
}

status_t brgemm_desc_set_postops(brgemm_desc_t *brg,
        const primitive_attr_t *attr, const memory_desc_t *dst_md, dim_t LDD,
        impl::data_type_t dt_bias) {
    if (!brg || !dst_md) return status::invalid_arguments;

    brg->set_attr(attr);
    brg->set_dst_md(dst_md);

    brg->with_bias = (dt_bias == data_type::undef) ? false : true;
    brg->dt_bias = dt_bias;
    brg->typesize_bias = (dt_bias == data_type::undef)
            ? 0
            : static_cast<int>(types::data_type_size(brg->dt_bias));

    brg->LDD = LDD;
    brg->is_runtime_ldd = is_runtime_value(LDD);
    const auto dt_d = dst_md->data_type;

    if (!brg->is_gemv) {
        if (!IMPLICATION(one_of(data_type::bf16, dt_bias, dt_d),
                    is_superset(brg->isa_impl, avx512_core)
                            || is_superset(brg->isa_impl, avx2_vnni_2)))
            return status::unimplemented;
        if (!IMPLICATION(one_of(data_type::f16, dt_bias, dt_d),
                    is_superset(brg->isa_impl, avx512_core_fp16)
                            || is_superset(brg->isa_impl, avx2_vnni_2)))
            return status::unimplemented;
    } else {
        // The GEMV code path requires avx512_core_bf16 for bf16 and
        // avx512_core_fp16 for f16.
        if (!IMPLICATION(one_of(data_type::bf16, dt_bias, dt_d),
                    is_superset(brg->isa_impl, avx512_core_bf16)))
            return status::unimplemented;
        if (!IMPLICATION(one_of(data_type::f16, dt_bias, dt_d),
                    is_superset(brg->isa_impl, avx512_core_fp16)))
            return status::unimplemented;
    }

    if (!IMPLICATION(one_of(data_type::f8_e5m2, dt_bias, dt_d)
                        || one_of(data_type::f8_e4m3, dt_bias, dt_d),
                utils::one_of(
                        true, mayiuse(avx512_core_amx_fp16), mayiuse(avx10_2))))
        return status::unimplemented;
    // check that combination of data types is allowed
    if ((brg->dt_a == data_type::u8 && brg->dt_b == data_type::s8)
            && (!one_of(dt_d, data_type::u8, data_type::s8, data_type::s32,
                        data_type::f32, data_type::f16, data_type::bf16)
                    || !one_of(dt_bias, data_type::undef, data_type::u8,
                            data_type::s8, data_type::s32, data_type::f32,
                            data_type::f16, data_type::bf16)))
        return status::unimplemented;
    if ((brg->dt_a == data_type::bf16 && brg->dt_b == data_type::bf16)
            && (!one_of(dt_d, data_type::bf16, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::bf16,
                    data_type::f32)))
        return status::unimplemented;
    // Matmul may upconvert f16 inputs or xf16 weights to f32
    // while retaining the original destination and bias data types.
    if ((brg->dt_a == data_type::f32 && brg->dt_b == data_type::f32)
            && (!one_of(dt_d, data_type::f32, data_type::f16, data_type::bf16)
                    || !one_of(dt_bias, data_type::undef, data_type::f32,
                            data_type::f16, data_type::bf16)))
        return status::unimplemented;
    if (!IMPLICATION(brg->is_bf16,
                one_of(dt_d, data_type::f32, data_type::bf16, data_type::f16,
                        data_type::u8, data_type::s8)
                        && one_of(dt_bias, data_type::undef, data_type::f32,
                                data_type::bf16, data_type::f16)))
        return status::unimplemented;
    if (!IMPLICATION(brg->is_f16,
                one_of(dt_d, data_type::f32, data_type::f16, data_type::bf16,
                        data_type::u8, data_type::s8, data_type::f8_e5m2,
                        data_type::f8_e4m3)
                        && one_of(dt_bias, data_type::undef, data_type::f32,
                                data_type::bf16, data_type::f16,
                                data_type::f8_e5m2, data_type::f8_e4m3)))
        return status::unimplemented;
    const auto bias_f8_e5m2_compatible
            = one_of(dt_d, data_type::f32, data_type::f16, data_type::bf16,
                      data_type::f8_e5m2)
            && one_of(dt_bias, data_type::undef, data_type::f32, data_type::f16,
                    data_type::bf16, data_type::f8_e5m2, data_type::f8_e4m3);
    const auto bias_f8_e4m3_compatible
            = one_of(dt_d, data_type::f32, data_type::f16, data_type::bf16,
                      data_type::f8_e4m3)
            && one_of(dt_bias, data_type::undef, data_type::f32, data_type::f16,
                    data_type::bf16, data_type::f8_e4m3, data_type::f8_e5m2);
    if (!IMPLICATION(brg->is_fp8,
                bias_f8_e5m2_compatible || bias_f8_e4m3_compatible))
        return status::unimplemented;

    brg->dt_d = dt_d;
    brg->typesize_D = static_cast<int>(types::data_type_size(brg->dt_d));

    if (!IMPLICATION(brg->is_int8 && brg->dt_d == bf16,
                is_superset(brg->isa_impl, avx512_core)
                        || brg->isa_impl == avx2_vnni_2))
        return status::unimplemented;

    if (brg->is_int8 && brg->dt_d == bf16)
        brg->is_bf16_emu
                = !(mayiuse(avx512_core_bf16) || brg->isa_impl == avx2_vnni_2);

    if (!brg->attr()) return status::success;

    using namespace injector;

    const auto &post_ops = brg->attr()->post_ops_;
    const memory_desc_wrapper dst_d(dst_md);

    const auto binary_ind = post_ops.find(primitive_kind::binary);
    const auto prelu_ind = post_ops.find(primitive_kind::prelu);
    brg->with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    // NOTE: Using brg->isa_impl here is a bit dangerous as it can change before
    //       kernel creation, so there is no gaurantee that the isa checked here
    //       matches the isa used at kernel creation time. For now this can only
    //       happen for bf32, where isa during this check is avx512_core and isa
    //       at kernel creation time is avx512_core_amx_bf16. It just so happens
    //       that the behavior of `post_ops_ok` is identical for those two isas,
    //       but there is no guarantee that will always be the case.
    if ((brg->with_binary && !dst_md)
            || !injector::post_ops_ok(
                    post_ops_ok_args_t(brg->isa_impl, {sum, eltwise, binary},
                            post_ops, &dst_d, false /*sum_at_pos_0_only*/,
                            false /*sum_requires_scale_one*/,
                            false /*sum_requires_zp_zero*/,
                            true /*sum_requires_same_params*/,
                            {broadcasting_strategy_t::per_oc,
                                    broadcasting_strategy_t::per_oc_d,
                                    broadcasting_strategy_t::scalar,
                                    broadcasting_strategy_t::per_mb,
                                    broadcasting_strategy_t::per_mb_spatial,
                                    broadcasting_strategy_t::per_mb_w,
                                    broadcasting_strategy_t::per_w,
                                    broadcasting_strategy_t::per_hw,
                                    broadcasting_strategy_t::batch,
                                    broadcasting_strategy_t::spatial,
                                    broadcasting_strategy_t::no_broadcast})))
        return status::unimplemented;

    const auto sum_idx = post_ops.find(primitive_kind::sum);
    const bool with_sum = sum_idx != -1;
    brg->with_sum = with_sum;
    brg->sum_scale = with_sum ? post_ops.entry_[sum_idx].sum.scale : 0;
    brg->sum_zp = with_sum ? post_ops.entry_[sum_idx].sum.zero_point : 0;
    const auto sum_dt
            = with_sum ? post_ops.entry_[sum_idx].sum.dt : data_type::undef;
    brg->sum_dt = sum_dt != data_type::undef ? sum_dt : dt_d;

    const auto eltwise_ind = post_ops.find(primitive_kind::eltwise);
    brg->with_eltwise = eltwise_ind != -1;

    const auto &src_scales = attr->scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr->scales_.get(DNNL_ARG_WEIGHTS);
    brg->with_src_scales = !src_scales.has_default_values();
    if (brg->with_src_scales) brg->dt_src_scales = src_scales.get_data_type();
    brg->with_wei_scales
            = !brg->skip_wei_scales && !wei_scales.has_default_values();

    bool wei_scales_are_set = brg->is_single_wei_scale
            || brg->is_per_n_wei_scales || brg->is_per_k_wei_scales;
    if (brg->with_wei_scales && !wei_scales_are_set) {
        // Note. the current version supports only two different wei scales
        // types:
        //     1) common (mask = 0)
        //     2) per_n_dim_scale - broadcast across n dimension;
        //        for convolution and inner product promitives it corresponds
        //        to "per_oc" mask = 1 << 1; for matmul - to
        //        mask = (1 << (ndims - 1))), where ndims is number of
        //        dimensions for original matmul problem
        // So if wei_scales.get_mask() > 0 (not common) it's assumed here that
        // scale type is per_n_dim_scale and driver which calls brgemm kernel
        // checked that mask has correct value for this case

        const auto &wei_scales_mask = wei_scales.get_mask();
        brg->is_single_wei_scale = wei_scales_mask == 0;
        brg->is_per_n_wei_scales = wei_scales_mask > 0;
        brg->is_per_k_wei_scales = false;
        brg->dt_wei_scales = wei_scales.get_data_type();
    }

    const auto &dst_scales = attr->scales_.get(DNNL_ARG_DST);
    brg->with_dst_scales = !dst_scales.has_default_values();
    const bool scales_ok = attr->scales_.has_default_values({DNNL_ARG_SRC,
                                   DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
            && IMPLICATION(!src_scales.has_default_values(),
                    src_scales.get_mask() == 0 || brg->is_per_k_src_scales)
            && IMPLICATION(!dst_scales.has_default_values(),
                    dst_scales.get_mask() == 0);
    if (!scales_ok) return status::unimplemented;

    auto init_zp_type
            = [&](brgemm_broadcast_t &zp_type, int mem_arg) -> status_t {
        const auto &zp = attr->zero_points_;
        // Always init a default value;
        zp_type = brgemm_broadcast_t::none;

        const bool skip_zero_point
                = (mem_arg == DNNL_ARG_WEIGHTS && brg->skip_zp_b_compensation)
                || (mem_arg == DNNL_ARG_SRC && brg->skip_zp_a_compensation);
        if (skip_zero_point) return status::success;

        if (!zp.has_default_values(mem_arg)) {
            int mask = zp.get_mask(mem_arg);
            if (mask == 0) {
                zp_type = brgemm_broadcast_t::per_tensor;
            } else if (mask == (1 << 1)) {
                zp_type = brgemm_broadcast_t::per_n;
            } else {
                return status::unimplemented;
            }
        }

        return status::success;
    };

    CHECK(init_zp_type(brg->zp_type_a, DNNL_ARG_SRC));
    CHECK(init_zp_type(brg->zp_type_b, DNNL_ARG_WEIGHTS));
    CHECK(init_zp_type(brg->zp_type_c, DNNL_ARG_DST));

    return status::success;
}

status_t brgemm_desc_set_attr(
        brgemm_desc_t *brg, const brgemm_attr_t &brgattr) {
    if (brg == nullptr) return status::invalid_arguments;

    // negative padding is not supported
    if (brgattr.max_top_vpad < 0 || brgattr.max_bottom_vpad < 0)
        return status::unimplemented;

    if (!brg->is_dgmm) {
        // virtual padding size is restricted by MAX_VPAD value
        if (brgattr.max_top_vpad > brgemm_desc_t::MAX_VPAD
                || brgattr.max_bottom_vpad > brgemm_desc_t::MAX_VPAD)
            return status::unimplemented;
    }

    // virtual padding is supported for "brgemm_row_major" layout
    // TODO: remove this restriction
    if ((brgattr.max_top_vpad > 0 || brgattr.max_bottom_vpad > 0)
            && brg->layout != brgemm_row_major)
        return status::unimplemented;

    brg->brgattr = brgattr;
    brg->bs_group = brgattr.hint_bs_group;

    if (brgattr.fpmath_mode != fpmath_mode::strict) maybe_try_bf32(brg);

    // virtual padding is not supported for "amx"
    if ((brgattr.max_top_vpad > 0 || brgattr.max_bottom_vpad > 0)
            && (brg->is_tmm || brg->is_ace()))
        return status::unimplemented;

    // Sprinkled prefetch is supported for brgemm_batch_size is 1
    if ((brgattr.max_bs != 1)
            && (brgattr.hint_prfB.sprinkled || brgattr.hint_prfA.sprinkled)) {
        return status::unimplemented;
    }

    // fused copy A already copies A for the BRGEMM. No need for partial copy.
    if (brgattr.hint_fused_copy_a
            && (brgattr.extendable_k || brgattr.wary_A_k_tail_read)) {
        return status::unimplemented;
    }

    if (!IMPLICATION(brgattr.hint_fused_copy_a,
                is_superset(brg->isa_impl, amx_tile))) {
        return status::unimplemented;
    }

    brg->prfA = brgattr.hint_prfA;
    brg->prfB = brgattr.hint_prfB;
    brg->prfC = brgattr.hint_prfC;
    if (brgattr.hint_innermost_loop != brgemm_innermost_undef)
        brg->innermost_loop = brgattr.hint_innermost_loop;

    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf0
            && brg->prfC.dist0 < 0)
        brg->prfC.dist0 = 0;
    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf1
            && brg->prfC.dist1 < 0)
        brg->prfC.dist1 = 0;
    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf2
            && brg->prfC.dist2 < 0)
        brg->prfC.dist2 = 0;

    if (brgattr.hint_fused_copy_a) brg->fused_copy_a = true;

    if (brg->is_fp8
            && !utils::one_of(true,
                    is_superset(brg->isa_impl, avx512_core_amx_fp16),
                    is_superset(brg->isa_impl, avx10_2)))
        return status::unimplemented;

    if (!IMPLICATION(brgattr.use_ace, brg->is_ace()))
        return status::unimplemented;

    // The ACE kernels cover bf16 and int8 only, see set_isa_impl(). Reject the
    // remaining types so that dispatch falls through to an implementation that
    // supports them.
    if (brgattr.use_ace && (brg->is_f16 || brg->is_fp8))
        return status::unimplemented;

    return status::success;
}

status_t brgemm_desc_finalize(brgemm_desc_t *brg) {
    if (brg == nullptr) return status::invalid_arguments;

    const dim_t max_vpad = nstl::max(
            brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);

    if (brg->is_dgmm)
        CHECK(brdgmm_blocking(brg));
    else
        CHECK(brgemm_blocking(brg));

    if (!brg->is_dgmm) {
        // virtual padding is restricted by bd_block size due to
        // brgemm_kernel implementation. TODO: remove this restriction
        const dim_t min_bd_block
                = brg->bdb_tail > 0 ? brg->bdb_tail : brg->bd_block;
        if ((max_vpad > min_bd_block)) return status::unimplemented;
    }

    // Tile paths take unsigned B (TMUL under AMX-INT8, TOP4BUUD/TOP4BSUD
    // under ACE); vector kernels read B as signed. is_tmm is final here.
    if (!IMPLICATION(brg->dt_b == u8, brg->is_tmm))
        return status::unimplemented;

    // Per-K (block-wise) scales and per-(M,N) compensation are applied to the
    // partial result of every brgemm call. Only the AMX uker implements that
    // for tile accumulators; the base kernel would silently drop them.
    if (brg->is_tmm && !brg->can_dispatch_uker()
            && (brg->has_per_k_scales() || brg->with_per_mn_compensation))
        return status::unimplemented;

    // Required for EVEX encoding for offsets
    // The kernel jit_brgemm_amx_uker_t has support of large offsets in
    // post-ops
    if (!brg->can_dispatch_uker()) {
        const dim_t max_d_stride
                = brg->LDD * types::data_type_size(brg->dt_d) * brg->bcast_dim;
        if (max_d_stride > std::numeric_limits<int32_t>::max())
            return status::unimplemented;
    }

    return status::success;
}

// Picks the Xbyak kernel that handles `brg`. Every factory returns `nullptr`
// for a descriptor it does not support, so a `nullptr` here means no Xbyak
// kernel covers `brg`.
static brgemm_kernel_t *create_xbyak_kernel(const brgemm_desc_t &brg) {
    if (brg.is_dgmm) return create_brdgmm_kernel(brg);
    if (brg.can_dispatch_uker()) return create_brgemm_amx_uker_kernel(brg);
    return create_brgemm_kernel(brg);
}

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_desc_t &brg) {
    if (!brg_kernel) return status::invalid_arguments;
    *brg_kernel = nullptr;

    if (utils::one_of(data_type::f64, brg.dt_a, brg.dt_b, brg.dt_c, brg.dt_d,
                brg.dt_bias, brg.sum_dt))
        return status::unimplemented;

    // Try the IR-based kernels first. `is_gemv` goes to
    // `create_brgemv_ir_kernel()`, everything else to
    // `create_brgemm_ir_kernel()`. Either returns `nullptr` when the IR kernel
    // is unsupported, and then the Xbyak-based brgemm implementations below
    // take over.
    {
        std::unique_ptr<brgemm_kernel_t> ir_ker(brg.is_gemv
                        ? create_brgemv_ir_kernel(brg)
                        : create_brgemm_ir_kernel(brg));
        if (ir_ker && ir_ker->create_kernel() == status::success) {
            *brg_kernel = ir_ker.release();
            return status::success;
        }
    }

    std::unique_ptr<brgemm_kernel_t> ker(create_xbyak_kernel(brg));
    if (!ker) return status::unimplemented;
    // Unlike the IR kernels above, a failure here is final. There is no
    // kernel to fall back to.
    CHECK(ker->create_kernel());

    *brg_kernel = ker.release();
    return status::success;
}

status_t brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel) {
    delete brg_kernel;
    return status::success;
}

status_t brgemm_init_tiles(const brgemm_desc_t &brg, char palette[64]) {
    if (!brg.is_tmm && !brg.is_ace()) return status::unimplemented;

    palette_config_t *buff = (palette_config_t *)(palette);

    char *_tc = (char *)(buff);
    static constexpr int max_palette_size_in_bytes = 64;
    for (int i = 0; i < max_palette_size_in_bytes; i++)
        _tc[i] = 0;

    // ACE uses a fixed 8x16x64 tile geometry; bytes 1-63 are zero.
    // The descriptor was already cleared, so only the palette ID is set here.
    if (brg.is_ace()) {
        buff->palette_id = amx_palette_ace;
        return status::success;
    }

    auto rd_block = (!brg.rdb && brg.rdb_tail) ? brg.rdb_tail : brg.rd_block;
    if (brg.is_input_convert())
        rd_block = utils::rnd_up(rd_block, 2 /*vnni_granularity*/);
    else
        rd_block = utils::rnd_up(rd_block, brg.rd_step);

    const int typesize_A = brg.is_input_convert()
            ? static_cast<int>(sizeof(int16_t))
            : brg.typesize_A;
    const int typesize_B = brg.is_input_convert()
            ? static_cast<int>(sizeof(int16_t))
            : brg.typesize_B;

    const int rd_step = 4 / typesize_A;

    const auto Ac = typesize_A * rd_block;

    // AMX TMUL always writes 4-byte tile elements regardless of logical
    // dt_c; use the fixed accumulator-tile element size for the palette.
    const auto Br = Ac / brgemm_desc_t::amx_c_tile_elem_size;

    if (brg.get_num_A_tiles() + brg.get_num_B_tiles() + brg.get_num_C_tiles()
            > brgemm_desc_t::AMX_TILES_NUM) {
        assert(!"brgemm internal error: invalid blocking");
        return status::runtime_error;
    }

    // Due to interleaving tileload/tmul we don't support blocking 1x6 and 6x1
    //TODO: update gemm_microkernel_amx to support such blocking
    if (brg.get_bd_block2() >= 6 || brg.get_num_C_tiles() >= 6)
        return status::unimplemented;

    for (int m = 0; m < brg.get_num_A_tiles(); m++) {
        const bool is_bd_tail
                = (brg.bdb_tail && m == (brg.get_num_A_tiles() - 1));
        const auto A_tensor = brg.get_A_tensor(m, is_bd_tail);
        const auto Ar = is_bd_tail ? brg.bdb_tail : brg.bd_block;
        tc_configure_tile(buff, A_tensor, Ar, Ac);
    }

    for (int n = 0; n < brg.get_num_B_tiles(); n++) {
        const bool is_ld_tail
                = (brg.ldb_tail && n == (brg.get_num_B_tiles() - 1));
        const auto B_tensor = brg.get_B_tensor(n, is_ld_tail);
        const auto Bc = (is_ld_tail ? brg.ldb_tail : brg.ld_block) * typesize_B
                * rd_step;
        tc_configure_tile(buff, B_tensor, Br, Bc);
    }

    for (int m = 0; m < brg.get_bd_block2(); m++) {
        const bool is_bd_tail
                = (brg.bdb_tail && m == (brg.get_bd_block2() - 1));
        const auto Cr = is_bd_tail ? brg.bdb_tail : brg.bd_block;
        for (int n = 0; n < brg.get_ld_block2(); n++) {
            const bool is_ld_tail
                    = (brg.ldb_tail && n == (brg.get_ld_block2() - 1));
            const auto Cc = (is_ld_tail ? brg.ldb_tail : brg.ld_block)
                    * brgemm_desc_t::amx_c_tile_elem_size;
            const auto C_tensor
                    = brg.get_C_tensor(m, n, is_bd_tail, is_ld_tail);
            tc_configure_tile(buff, C_tensor, Cr, Cc);
        }
    }

    buff->palette_id = static_cast<uint8_t>(amx::get_target_palette());

    return status::success;
}

namespace {
template <typename T>
inline int sign(T v) {
    return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
}

int brgemm_cmp(const brgemm_desc_t &lhs, const brgemm_desc_t &rhs) {
    // The macro CMP_BRGEMM_FIELD is designed to compare numerical parameters.
    // Float parameters must not be NaN
#define CMP_BRGEMM_FIELD(x) \
    if ((lhs.x) != (rhs.x)) return sign((lhs.x) - (rhs.x))

    // This function compares brgemm_desc_t objects within a single brgemm primitive.
    // Comparison of objects from different primitives is not guaranteed due to
    // dependencies of brgemm descriptor on a primitive attributes.

    // Compare all non-pointer parameters of brgemm_desc_t except derived
    CMP_BRGEMM_FIELD(bcast_dim);
    CMP_BRGEMM_FIELD(load_dim);
    CMP_BRGEMM_FIELD(reduce_dim);
    CMP_BRGEMM_FIELD(LDA);
    CMP_BRGEMM_FIELD(LDB);
    CMP_BRGEMM_FIELD(LDC);
    CMP_BRGEMM_FIELD(LDD);
    CMP_BRGEMM_FIELD(isa_user);
    CMP_BRGEMM_FIELD(isa_impl);
    CMP_BRGEMM_FIELD(alpha);
    CMP_BRGEMM_FIELD(beta);
    CMP_BRGEMM_FIELD(dt_a);
    CMP_BRGEMM_FIELD(dt_b);
    CMP_BRGEMM_FIELD(dt_c);
    CMP_BRGEMM_FIELD(dt_d);
    CMP_BRGEMM_FIELD(dt_bias);
    CMP_BRGEMM_FIELD(stride_a);
    CMP_BRGEMM_FIELD(stride_b);
    CMP_BRGEMM_FIELD(layout);
    CMP_BRGEMM_FIELD(type);
    CMP_BRGEMM_FIELD(is_dgmm);
    CMP_BRGEMM_FIELD(with_sum);
    CMP_BRGEMM_FIELD(req_cal_comp_pads);

    CMP_BRGEMM_FIELD(sum_scale);
    CMP_BRGEMM_FIELD(sum_zp);
    CMP_BRGEMM_FIELD(sum_dt);
    CMP_BRGEMM_FIELD(with_eltwise);
    CMP_BRGEMM_FIELD(with_binary);

    CMP_BRGEMM_FIELD(zp_type_a);
    CMP_BRGEMM_FIELD(zp_type_b);
    CMP_BRGEMM_FIELD(zp_type_c);

    CMP_BRGEMM_FIELD(skip_wei_scales);
    CMP_BRGEMM_FIELD(is_single_wei_scale);
    CMP_BRGEMM_FIELD(is_per_n_wei_scales);
    CMP_BRGEMM_FIELD(is_per_k_wei_scales);
    CMP_BRGEMM_FIELD(with_src_scales);
    CMP_BRGEMM_FIELD(is_per_k_src_scales);
    CMP_BRGEMM_FIELD(src_scale_m_stride);
    CMP_BRGEMM_FIELD(dt_src_scales);
    CMP_BRGEMM_FIELD(with_wei_scales);
    CMP_BRGEMM_FIELD(with_dst_scales);
    CMP_BRGEMM_FIELD(dt_wei_scales);
    CMP_BRGEMM_FIELD(bs_group);

    // Compare all non-pointer parameters of brgemm_attr_t except derived
    CMP_BRGEMM_FIELD(brgattr.max_bs);
    CMP_BRGEMM_FIELD(brgattr.max_top_vpad);
    CMP_BRGEMM_FIELD(brgattr.max_bottom_vpad);
    CMP_BRGEMM_FIELD(brgattr.max_top_bpad);
    CMP_BRGEMM_FIELD(brgattr.max_bottom_bpad);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_A_size);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_B_size);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_C_size);
    CMP_BRGEMM_FIELD(brgattr.hint_innermost_loop);
    CMP_BRGEMM_FIELD(brgattr.hint_loop_order);
    CMP_BRGEMM_FIELD(brgattr.hint_prefetching);
    CMP_BRGEMM_FIELD(brgattr.hint_prfA.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfA.dist2);
    CMP_BRGEMM_FIELD(brgattr.hint_prfA.sprinkled);
    CMP_BRGEMM_FIELD(brgattr.hint_prfB.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfB.dist2);
    CMP_BRGEMM_FIELD(brgattr.hint_prfB.sprinkled);
    CMP_BRGEMM_FIELD(brgattr.hint_prfC.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfC.dist2);
    CMP_BRGEMM_FIELD(brgattr.wary_A_k_tail_read);
    CMP_BRGEMM_FIELD(brgattr.extendable_k);
    CMP_BRGEMM_FIELD(brgattr.generate_skip_accumulation);
    CMP_BRGEMM_FIELD(brgattr.bd_mask_level);
    CMP_BRGEMM_FIELD(brgattr.use_uker);
    CMP_BRGEMM_FIELD(brgattr.use_interleave_stores);
    CMP_BRGEMM_FIELD(brgattr.b_is_vnni);
    CMP_BRGEMM_FIELD(brgattr.fpmath_mode);
    CMP_BRGEMM_FIELD(brgattr.LDA2);
    CMP_BRGEMM_FIELD(brgattr.LDB2);
    CMP_BRGEMM_FIELD(brgattr.LDC2_M);
    CMP_BRGEMM_FIELD(brgattr.LDC2_N);
    CMP_BRGEMM_FIELD(brgattr.var_bs);
    CMP_BRGEMM_FIELD(brgattr.postops_only);
    CMP_BRGEMM_FIELD(brgattr.hint_bs_group);

    CMP_BRGEMM_FIELD(brgattr.hint_bd_block);
    CMP_BRGEMM_FIELD(brgattr.hint_ld_block);
    CMP_BRGEMM_FIELD(brgattr.hint_bd_block2);
    CMP_BRGEMM_FIELD(brgattr.hint_ld_block2);
    CMP_BRGEMM_FIELD(brgattr.hint_ununroll_bd_loop);

    CMP_BRGEMM_FIELD(brgattr.hint_load_nt_A);
    CMP_BRGEMM_FIELD(brgattr.hint_load_nt_B);
    CMP_BRGEMM_FIELD(brgattr.K_koef);
    CMP_BRGEMM_FIELD(brgattr.use_ace);

    if (lhs.brgattr.bd_mask_level > 0)
        for (int i = 0; i < lhs.bcast_dim; i++) {
            CMP_BRGEMM_FIELD(brgattr.bd_mask[i]);
        }

    if (lhs.type == brgemm_static_offs)
        for (int i = 0; i < lhs.brgattr.max_bs; i++) {
            CMP_BRGEMM_FIELD(brgattr.static_offsets[i].offset.A);
            CMP_BRGEMM_FIELD(brgattr.static_offsets[i].offset.B);
        }

#undef CMP_BRGEMM_FIELD
    return 0;
}
} // namespace

bool brgemm_desc_t::operator==(const brgemm_desc_t &rhs) const {
    return (brgemm_cmp(*this, rhs) == 0);
}

bool brgemm_desc_t::operator<(const brgemm_desc_t &rhs) const {
    return (brgemm_cmp(*this, rhs) < 0);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
