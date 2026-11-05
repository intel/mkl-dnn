/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t brgemm_matmul_copy_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace status;
    using namespace format_tag;

    status_t status = cpu_reorder_pd_t::init(engine, src_engine, dst_engine);
    if (status != success) return status;

    const memory_desc_wrapper id(src_md_), od(dst_md_);
    const int ndims = id.ndims();

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();
    // Currently only f32 is supported.
    // TODO: enable support for bf16, f16, s8 and type_i != type_o cases.
    const bool dt_ok = type_i == data_type::f32 && type_o == data_type::f32;
    const bool has_adj_scale
            = od.extra().flags & memory_extra_flags::scale_adjust;
    const bool args_ok = true && dt_ok && id.is_dense()
            && utils::one_of(ndims, 2, 3) && !has_adj_scale
            && attr()->has_default_values() && od.is_blocking_desc()
            && !od.has_runtime_dims_or_strides() && !od.has_zero_dim();
    if (!args_ok) return invalid_arguments;

    const auto &dims = id.dims();
    // TODO: enable for itag = {ba, acb}
    format_tag_t itag = id.matches_one_of_tag(ab, abc);
    format_tag_t otag = format_tag::undef;

    const auto vnni_granularity = data_type_vnni_granularity(type_o);
    switch (vnni_granularity) {
        case 4:
            otag = od.matches_one_of_tag(aCB16b64c4b, BA16a64b4a, aCB16b48c4b,
                    BA16a48b4a, aCB16b32c4b, BA16a32b4a, aCB16b16c4b,
                    BA16a16b4a);
            break;
        case 2:
            otag = od.matches_one_of_tag(aCB16b64c2b, BA16a64b2a, aCB16b48c2b,
                    BA16a48b2a, aCB16b32c2b, BA16a32b2a, aCB16b16c2b,
                    BA16a16b2a);
            break;
        case 1:
            otag = od.matches_one_of_tag(aCB16b64c, BA16a64b, aCB16b48c,
                    BA16a48b, aCB16b32c, BA16a32b, aCB16b16c, BA16a16b);
            break;
        default: otag = format_tag::undef;
    }

    if (utils::one_of(format_tag::undef, itag, otag)) return invalid_arguments;

    // initialize all required fields to generate copy_b kernel
    matmul_conf_for_reorder_.wei_tag = itag;
    matmul_conf_for_reorder_.batch = ndims > 2 ? dims[ndims - 3] : 1;
    matmul_conf_for_reorder_.K = dims[ndims - 2];
    matmul_conf_for_reorder_.N = dims[ndims - 1];
    matmul_conf_for_reorder_.wei_n_blk = matmul_conf_for_reorder_.N_blk
            = matmul_conf_for_reorder_.LDB
            = matmul::get_default_n_block(otag, matmul_conf_for_reorder_);
    matmul_conf_for_reorder_.N_tail
            = matmul_conf_for_reorder_.N % matmul_conf_for_reorder_.N_blk;
    matmul_conf_for_reorder_.K_blk = 16 * vnni_granularity;
    matmul_conf_for_reorder_.K_tail
            = matmul_conf_for_reorder_.K % matmul_conf_for_reorder_.K_blk;
    matmul_conf_for_reorder_.src_dt = matmul_conf_for_reorder_.wei_dt = type_o;
    matmul_conf_for_reorder_.a_dt_sz = matmul_conf_for_reorder_.tr_a_dt_sz
            = types::data_type_size(matmul_conf_for_reorder_.src_dt);
    matmul_conf_for_reorder_.b_dt_sz = matmul_conf_for_reorder_.tr_b_dt_sz
            = types::data_type_size(matmul_conf_for_reorder_.wei_dt);
    matmul_conf_for_reorder_.s8s8_comp_b_str = utils::rnd_up(
            matmul_conf_for_reorder_.N, matmul_conf_for_reorder_.wei_n_blk);
    matmul_conf_for_reorder_.s8s8_comp_n_str
            = matmul_conf_for_reorder_.wei_n_blk;
    matmul_conf_for_reorder_.s8s8_compensation_required
            = od.extra().flags & memory_extra_flags::compensation_conv_s8s8;
    const bool req_asymmetric_comp = od.extra().flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    matmul_conf_for_reorder_.src_zp_type = req_asymmetric_comp
            ? brgemm_broadcast_t::per_tensor
            : brgemm_broadcast_t::none;
    matmul_conf_for_reorder_.has_zero_point_a
            = matmul_conf_for_reorder_.src_zp_type != brgemm_broadcast_t::none;

    VDISPATCH_REORDER(mayiuse(sve), VERBOSE_UNSUPPORTED_ISA);
    matmul_conf_for_reorder_.isa = sve;

    auto mask_ok = [&](bool check, int mask) {
        return IMPLICATION(
                check, mask == (1 << ndims) - 1 - (1 << (ndims - 2)));
    };

    const bool comp_masks_ok = true
            && mask_ok(matmul_conf_for_reorder_.s8s8_compensation_required,
                    od.extra().compensation_mask)
            && mask_ok(req_asymmetric_comp, od.extra().asymm_compensation_mask);
    if (!comp_masks_ok) return invalid_arguments;

    init_scratchpad();

    return status::success;
}

status_t brgemm_matmul_copy_reorder_t::execute_body(
        const exec_ctx_t &ctx) const {
    using namespace utils;

    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const memory_desc_wrapper &src_d = pd()->src_md();
    const memory_desc_wrapper &dst_d = pd()->dst_md();
    const auto sdt_sz = types::data_type_size(src_d.data_type());
    const auto type_o = dst_d.data_type();
    const auto ddt_sz = types::data_type_size(type_o);

    const auto &kernel_conf = pd()->matmul_conf_for_reorder_;
    const size_t comp_offset_bytes
            = dst_d.size() - dst_d.additional_buffer_size();
    const size_t s8s8_comp_size_bytes = kernel_conf.s8s8_compensation_required
            ? dst_d.additional_buffer_size(
                      memory_extra_flags::compensation_conv_s8s8)
            : 0;
    const size_t zp_comp_offset_bytes
            = comp_offset_bytes + s8s8_comp_size_bytes;
    int32_t *cp = kernel_conf.s8s8_compensation_required
            ? reinterpret_cast<int32_t *>(dst + comp_offset_bytes)
            : nullptr;
    int32_t *zp = kernel_conf.has_zero_point_a
            ? reinterpret_cast<int32_t *>(dst + zp_comp_offset_bytes)
            : nullptr;

    const int ndims = src_d.ndims();
#define get_blk_off(md, dt_sz, batch, d0, d1) \
    (ndims == 3 ? (dt_sz) * (md).blk_off((batch), (d0), (d1)) \
                : (dt_sz) * (md).blk_off((d0), (d1)))

    parallel_nd(kernel_conf.batch, div_up(kernel_conf.N, kernel_conf.N_blk),
            [= COMPAT_THIS_CAPTURE](dim_t batch, dim_t n_blk_idx) {
        const auto n = n_blk_idx * kernel_conf.N_blk;
        const bool is_N_tail = (kernel_conf.N - n < kernel_conf.N_blk);
        auto ker_exec_ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
        ker_exec_ctx.current_N_blk
                = is_N_tail ? kernel_conf.N_tail : kernel_conf.N_blk;

        assert(!is_runtime_value(kernel_conf.s8s8_comp_b_str));
        const auto comp_offset = batch * kernel_conf.s8s8_comp_b_str
                + n_blk_idx * kernel_conf.s8s8_comp_n_str;

        ker_exec_ctx.zp_a_compensation_ptr = kernel_conf.has_zero_point_a
                ? (void *)&zp[comp_offset]
                : nullptr;
        ker_exec_ctx.compensation_ptr = kernel_conf.s8s8_compensation_required
                ? (void *)&cp[comp_offset]
                : nullptr;

        // required to compute zp compensation
        int tmp_neg_a_zp_val = -1;
        ker_exec_ctx.zp_a_neg_value_ptr = &tmp_neg_a_zp_val;

        int k_blk_idx = 0;
        for (; k_blk_idx < kernel_conf.K / kernel_conf.K_blk; k_blk_idx++) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            ker_exec_ctx.src
                    = (void *)&src[get_blk_off(src_d, sdt_sz, batch, k, n)];
            ker_exec_ctx.tr_src = (void *)&dst[get_blk_off(
                    dst_d, ddt_sz, batch, k_blk_idx, n_blk_idx)];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_blk;
            (*kernel_)(&ker_exec_ctx);
        }
        if (kernel_conf.K_tail > 0) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            ker_exec_ctx.src
                    = (void *)&src[get_blk_off(src_d, sdt_sz, batch, k, n)];
            const auto dst_offset
                    = get_blk_off(dst_d, ddt_sz, batch, k_blk_idx, n_blk_idx);
            ker_exec_ctx.tr_src = (void *)&dst[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_tail;
            (*kernel_)(&ker_exec_ctx);
            const auto vnni_granularity = data_type_vnni_granularity(type_o);
            const auto dst_zero_out_offset
                    = rnd_up(kernel_conf.K_tail, vnni_granularity)
                    * kernel_conf.N_blk * ddt_sz;
            const auto elems_to_zero
                    = rnd_dn(kernel_conf.K_blk - kernel_conf.K_tail,
                              vnni_granularity)
                    * kernel_conf.N_blk * ddt_sz;
            array_set(&dst[dst_offset + dst_zero_out_offset], 0, elems_to_zero);
        }
    });

#undef get_blk_off

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
