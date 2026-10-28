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

#include <tuple>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/jit_brgemm_inner_product.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::cpu::x64::brgemm_inner_product_utils;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;

#define get_blk_off(d, dt, ...) \
    (types::data_type_size((dt)) * (d).blk_off(__VA_ARGS__))

static size_t blk_off(const memory_desc_wrapper &mdw, dim_t n, dim_t c, dim_t d,
        dim_t h, dim_t w) {
    switch (mdw.ndims()) {
        case 5: return get_blk_off(mdw, mdw.data_type(), n, c, d, h, w);
        case 4: return get_blk_off(mdw, mdw.data_type(), n, c, h, w);
        case 3: return get_blk_off(mdw, mdw.data_type(), n, c, w);
        case 2: return get_blk_off(mdw, mdw.data_type(), n, c);
        default: assert(!"unsupported ndims"); return size_t(0);
    }
}

namespace {
template <typename ker_type>
void copy_data_chunk(ker_type &ker, char *tr_data, const char *data,
        int os_work, bool is_last_blk) {
    auto ctx = jit_brgemm_copy_to_coarse_t::ctx_t();
    ctx.data = (void *)data;
    ctx.tr_data = (void *)tr_data;
    ctx.os_work = os_work;
    ctx.last_row_blk = is_last_blk ? 1 : 0;
    (*ker)(&ctx);
}

} // namespace

template <cpu_isa_t isa>
status_t brgemm_inner_product_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jbgp = pd()->jbgp_;

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const size_t src_dt_size = types::data_type_size(jbgp.src_dt);
    const size_t bia_dt_size
            = jbgp.with_bias ? types::data_type_size(jbgp.bia_dt) : 0;
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);
    const size_t dst_dt_size = types::data_type_size(jbgp.dst_dt);

    auto addr_batch_global = scratchpad.template get<brgemm_batch_element_t>(
            key_brgemm_primitive_batch);
    auto a_buffer_global = (jbgp.use_buffer_a)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
            : nullptr;
    auto c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    const bool is_amx = jbgp.is_amx;
    auto wsp_tile_base = is_amx
            ? ctx.get_scratchpad_grantor().template get<char>(
                      key_conv_amx_tile_buffer)
            : nullptr;

    const int ic_chunks
            = static_cast<int>(div_up(jbgp.nb_ic, jbgp.nb_ic_blocking));

    const bool are_post_ops_applicable = one_of(true, jbgp.with_sum,
            jbgp.with_bias, jbgp.with_src_scales, jbgp.with_wei_scales,
            jbgp.with_dst_scales, jbgp.with_eltwise, jbgp.with_binary,
            jbgp.acc_dt != jbgp.dst_dt, jbgp.req_s8s8_compensation);
    const bool can_use_dst_as_acc_buffer
            = jbgp.acc_dt == jbgp.dst_dt && !jbgp.with_sum;
    const int acc_buffer_idx_shift = can_use_dst_as_acc_buffer;

    size_t offset = types::data_type_size(jbgp.wei_dt)
            * (weights_d.size() - weights_d.additional_buffer_size());
    auto compensation = jbgp.req_s8s8_compensation
            ? reinterpret_cast<const int32_t *>(&weights[offset])
            : nullptr;

    const dims_t ic_dims = {0, jbgp.ic_block, 0, 0, 0};
    const auto wei_ic_stride
            = types::data_type_size(jbgp.wei_dt) * weights_d.off_v(ic_dims);

    const auto ker
            = [=, &jbgp COMPAT_THIS_CAPTURE](int ithr_oc_mb, int nthr_oc_mb,
                      int ithr_ic, dim_t osb, dim_t osb_s, dim_t ocb,
                      dim_t ocb_s, int icc, int icc_s, int kd, int kh, int kw,
                      bool copy_buffer_a, int &prev_ker_idx,
                      const void *dst_scales_ptr) {
        const dim_t cur_ocb = ocb + ocb_s;
        const dim_t cur_osb = osb + osb_s;
        const int cur_icc = icc + icc_s;
        const int n = static_cast<int>(cur_osb * jbgp.os_block);
        const int ithr = nthr_oc_mb * ithr_ic + ithr_oc_mb;
        auto addr_batch = addr_batch_global + ithr * jbgp.adjusted_batch_size;

        const size_t a_buffer_osb_stride
                = src_dt_size * jbgp.LDA * jbgp.os_block;
        const size_t a_buffer_per_thr
                = a_buffer_osb_stride * jbgp.nb_os_blocking;
        auto a_buffer = (jbgp.use_buffer_a) ? a_buffer_global
                        + ithr * a_buffer_per_thr + osb * a_buffer_osb_stride
                                            : nullptr;

        const int oc = static_cast<int>(cur_ocb * jbgp.oc_block);
        const size_t dst_off = get_blk_off(dst_d, jbgp.dst_dt, n, oc);

        const bool force_use_dst_as_acc_buffer = can_use_dst_as_acc_buffer
                && jbgp.nthr_ic_b > 1 && ithr_ic == 0;
        const bool use_c_buffer
                = jbgp.use_buffer && !force_use_dst_as_acc_buffer;

        char *c_buffer = nullptr;
        if (use_c_buffer) {
            size_t c_buf_idx = 0;
            size_t c_buf_nrows = 0;
            size_t c_buf_walk = 0;

            if (jbgp.nthr_ic_b > 1) {
                c_buf_idx = ithr_ic - acc_buffer_idx_shift;
                c_buf_nrows = jbgp.mb;

                // NOTE: This trick only works because the leading dimension of
                // buffer and memory descriptor is same.
                c_buf_walk = dst_off / dst_dt_size;
            } else {
                using loop_order_t = jit_brgemm_ip_fwd_conf_t::loop_order_t;
                switch (jbgp.loop_order) {
                    case loop_order_t::osc_occ_osb_ocb_icc:
                        c_buf_idx = ithr;
                        c_buf_nrows = jbgp.M;

                        // Single small block buffer, no walk needed.
                        c_buf_walk = 0;
                        break;
                    case loop_order_t::osc_occ_icc_osb_ocb:
                        c_buf_idx = ithr;
                        c_buf_nrows = jbgp.os_block * jbgp.nb_os_blocking;

                        // Walk for each block in os/oc-chunk.
                        c_buf_walk = osb * jbgp.os_block * jbgp.LDC
                                + ocb * jbgp.oc_block;
                        break;
                    case loop_order_t::icc_osc_occ_osb_ocb:
                    case loop_order_t::icc_occ_osc_ocb_osb:
                        c_buf_idx = 0;
                        c_buf_nrows = jbgp.os;

                        // NOTE: This trick only works because the leading
                        // dimension of buffer and memory descriptor is same.
                        c_buf_walk = dst_off / dst_dt_size;
                        break;
                }
            }

            size_t c_buf_shift = c_buf_idx * c_buf_nrows * jbgp.LDC;
            c_buf_shift += c_buf_walk;
            size_t c_buf_off = acc_dt_size * c_buf_shift;
            c_buffer = c_buffer_global + c_buf_off;
        }

        char *wsp_tile = is_amx
                ? wsp_tile_base + ithr * jbgp.amx_buf_size_per_thread
                : nullptr;
        int icb = cur_icc * jbgp.nb_ic_blocking;
        int ic = icb * jbgp.ic_block;

        bool kernel_init = cur_icc == icc_s && everyone_is(0, kd, kh, kw);

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
        bool is_last_ic_chunk = cur_icc == ic_chunks - 1;
        bool is_ic_tail = is_last_ic_chunk && jbgp.K_tail > 0;
        bool last_spatial_slice = true;
        last_spatial_slice &= kd == jbgp.kd - 1;
        last_spatial_slice &= kh == jbgp.kh - 1;
        last_spatial_slice &= kw == jbgp.kw - 1;

        const int remaining_ic_blks
                = (jbgp.use_buffer_a ? utils::rnd_up(jbgp.ic, jbgp.ic_block)
                                     : jbgp.ic)
                - ic;
        const int gemm_batch = static_cast<int>(nstl::min<dim_t>(
                jbgp.gemm_batch_size, remaining_ic_blks / jbgp.K));

        auto is_bs_tail = (gemm_batch != jbgp.gemm_batch_size);
        int brg_ker_idx = brgemm_inner_product_utils::get_brg_kernel_index(
                is_bs_tail, kernel_init, is_os_tail, is_oc_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();
        const int ic_blocks_per_batch
                = static_cast<int>(jbgp.K / jbgp.ic_block);

        const dim_t wei_cur_ocb = blk_off(weights_d, cur_ocb, 0, kd, kh, kw);

        if (copy_buffer_a) {
            assert(!jbgp.is_bf32);
            auto src_ptr = src + blk_off(src_d, n, ic, kd, kh, kw);
            copy_data_chunk(copy_src_kernel_, a_buffer, src_ptr,
                    is_os_tail ? jbgp.mb - n : jbgp.os_block, is_last_ic_chunk);
        }
        if (gemm_batch > 0 && brg_kernel != nullptr) {
            brgemm_palettes_.maybe_tile_configure(
                    is_amx, prev_ker_idx, brg_ker_idx);
            for (int b = 0; b < gemm_batch; b++) {
                auto A_ptr = jbgp.use_buffer_a
                        ? (a_buffer + src_dt_size * b * jbgp.K)
                        : (src
                                  + blk_off(src_d, n, ic + b * jbgp.K, kd, kh,
                                          kw));
                addr_batch[b].ptr.A = A_ptr;
                const dim_t wei_offset = wei_cur_ocb
                        + wei_ic_stride * (icb + b * ic_blocks_per_batch);
                addr_batch[b].ptr.B = weights + wei_offset;
            }

            auto ptr_D = dst + dst_off;
            auto ptr_C = use_c_buffer ? c_buffer : ptr_D;

            if (jbgp.nthr_ic_b == 1 && are_post_ops_applicable
                    && is_last_ic_chunk && !is_ic_tail && last_spatial_slice) {
                void *scratch = is_amx
                        ? static_cast<void *>(wsp_tile)
                        : (jbgp.req_s8s8_compensation
                                          ? static_cast<void *>(
                                                    const_cast<int *>(
                                                            &compensation[oc]))
                                          : nullptr);
                auto ptr_bias
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                const brgemm_post_ops_data_t post_ops_data {
                        static_cast<const void *>(ptr_bias),
                        post_ops_binary_rhs_arg_vec.data(),
                        static_cast<size_t>(oc), 0, dst, 0, nullptr, nullptr,
                        nullptr, false, 1, false, false, src_scales,
                        wei_scales ? static_cast<const char *>(wei_scales)
                                        + jbgp.is_oc_scale * oc * sizeof(float)
                                   : nullptr,
                        dst_scales_ptr};

                brgemm_kernel_execute_postops(brg_kernel, gemm_batch,
                        addr_batch, (void *)ptr_C, (void *)ptr_D, post_ops_data,
                        scratch);
            } else {
                brgemm_kernel_execute(brg_kernel, gemm_batch, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }

        if (is_ic_tail) {
            assert(!jbgp.use_buffer_a);
            auto use_init_ker = (kernel_init && gemm_batch == 0);
            int brg_ker_ic_tail_idx
                    = brgemm_inner_product_utils::get_brg_kernel_index(
                            false, use_init_ker, is_os_tail, is_oc_tail, true);
            brgemm_palettes_.maybe_tile_configure(
                    is_amx, prev_ker_idx, brg_ker_ic_tail_idx);

            const dim_t ic_block
                    = static_cast<dim_t>(gemm_batch) * ic_blocks_per_batch;
            addr_batch[0].ptr.A = src
                    + blk_off(src_d, n, ic + ic_block * jbgp.ic_block, kd, kh,
                            kw);
            const dim_t wei_offset
                    = wei_cur_ocb + wei_ic_stride * (icb + ic_block);
            addr_batch[0].ptr.B = weights + wei_offset;

            auto brg_kernel_ic_tail = brg_kernels_[brg_ker_ic_tail_idx].get();
            auto ptr_D = dst + dst_off;
            auto ptr_C = use_c_buffer ? c_buffer : ptr_D;
            if (jbgp.nthr_ic_b == 1 && are_post_ops_applicable
                    && last_spatial_slice) {
                void *scratch = is_amx
                        ? static_cast<void *>(wsp_tile)
                        : (jbgp.req_s8s8_compensation
                                          ? static_cast<void *>(
                                                    const_cast<int *>(
                                                            &compensation[oc]))
                                          : nullptr);
                auto ptr_bias
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                const brgemm_post_ops_data_t post_ops_data {
                        static_cast<const void *>(ptr_bias),
                        post_ops_binary_rhs_arg_vec.data(),
                        static_cast<size_t>(oc), 0, dst, 0, nullptr, nullptr,
                        nullptr, false, 1, false, false, src_scales,
                        wei_scales ? static_cast<const char *>(wei_scales)
                                        + jbgp.is_oc_scale * oc * sizeof(float)
                                   : nullptr,
                        dst_scales_ptr};

                brgemm_kernel_execute_postops(brg_kernel_ic_tail, 1, addr_batch,
                        (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
            } else {
                brgemm_kernel_execute(brg_kernel_ic_tail, 1, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }
    };

    const int os_chunks
            = static_cast<int>(div_up(jbgp.nb_os, jbgp.nb_os_blocking));
    const int oc_chunks
            = static_cast<int>(div_up(jbgp.nb_oc, jbgp.nb_oc_blocking));
    const int work_amount = oc_chunks * os_chunks;

    const auto init_thr_groups
            = [=, &jbgp](const int ithr, const int nthr, int &nthr_ic,
                      int &nthr_oc_mb, int &ithr_ic, int &ithr_oc_mb) {
        nthr_ic = jbgp.nthr_ic_b <= nthr ? jbgp.nthr_ic_b : 1;
        nthr_oc_mb = nthr / nthr_ic;
        ithr_ic = ithr / nthr_oc_mb;
        ithr_oc_mb = ithr % nthr_oc_mb;
        if (ithr_oc_mb >= work_amount || ithr_ic >= ic_chunks
                || ithr >= rnd_dn(nthr, nthr_ic))
            return false;
        return true;
    };

    // If work_amount == 1 we limit num_threads to 1 as parallel(1, ...) does
    // not create parallel section at all. We do not limit num_threads
    // for 1 < work_amount < dnnl_get_max_threads() case to avoid potential
    // overhead on spawning different number of OMP threads from layer to layer.
    const int num_threads
            = work_amount == 1 && jbgp.nthr_ic_b <= 1 ? 1 : jbgp.nthr;
    parallel(num_threads, [=, &jbgp](const int ithr, const int nthr) {
        int nthr_ic {1}, nthr_oc_mb {1}, ithr_ic {0}, ithr_oc_mb {0};
        bool ok = init_thr_groups(
                ithr, nthr, nthr_ic, nthr_oc_mb, ithr_ic, ithr_oc_mb);
        if (!ok) return;

        int start {0}, end {0};
        balance211(work_amount, nthr_oc_mb, ithr_oc_mb, start, end);

        int icc_start {0}, icc_end {ic_chunks};
        if (nthr_ic > 1)
            balance211(ic_chunks, nthr_ic, ithr_ic, icc_start, icc_end);

        float *dst_scales_inv_ptr = nullptr;
        if (jbgp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(key_iprod_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        const int icc_work = icc_end - icc_start;

        const auto order = jbgp.loop_order;

        // For keeping track of last tile configuration used.
        int prev_ker_idx = -1;

        int icc {0}, occ {0}, osc {0};
        int work {start};
        using loop_order_t = jit_brgemm_ip_fwd_conf_t::loop_order_t;

        switch (order) {
            case loop_order_t::osc_occ_icc_osb_ocb:
            case loop_order_t::osc_occ_osb_ocb_icc:
            case loop_order_t::icc_osc_occ_osb_ocb:
                nd_iterator_init(work, osc, os_chunks, occ, oc_chunks);
                break;
            case loop_order_t::icc_occ_osc_ocb_osb:
                nd_iterator_init(work, occ, oc_chunks, osc, os_chunks);
                break;
        }
        while (work < end) {
            dim_t ocb_s = occ * jbgp.nb_oc_blocking;
            dim_t ocb_e
                    = nstl::min<dim_t>(ocb_s + jbgp.nb_oc_blocking, jbgp.nb_oc);
            dim_t ocb_work = ocb_e - ocb_s;

            dim_t osb_s = osc * jbgp.nb_os_blocking;
            dim_t osb_e
                    = nstl::min<dim_t>(osb_s + jbgp.nb_os_blocking, jbgp.nb_os);
            dim_t osb_work = osb_e - osb_s;

            // Each thread runs the below loops:
            dim_t loop_start = 0, loop_end = 0;
            switch (order) {
                case loop_order_t::osc_occ_icc_osb_ocb:
                case loop_order_t::osc_occ_osb_ocb_icc:
                    loop_end = icc_work * osb_work * ocb_work;
                    break;
                case loop_order_t::icc_occ_osc_ocb_osb:
                case loop_order_t::icc_osc_occ_osb_ocb:
                    loop_end = osb_work * ocb_work;
                    break;
            }

            dim_t osb = 0, ocb = 0;
            switch (order) {
                case loop_order_t::osc_occ_icc_osb_ocb:
                    nd_iterator_init(
                            0, icc, icc_work, osb, osb_work, ocb, ocb_work);
                    break;
                case loop_order_t::osc_occ_osb_ocb_icc:
                    nd_iterator_init(
                            0, osb, osb_work, ocb, ocb_work, icc, icc_work);
                    break;
                case loop_order_t::icc_osc_occ_osb_ocb:
                    nd_iterator_init(0, osb, osb_work, ocb, ocb_work);
                    break;
                case loop_order_t::icc_occ_osc_ocb_osb:
                    nd_iterator_init(0, ocb, ocb_work, osb, osb_work);
                    break;
            }

            while (loop_start < loop_end) {
                const bool ocb_inner_most
                        = order == loop_order_t::osc_occ_icc_osb_ocb;

                const bool copy_buffer_a = jbgp.use_buffer_a
                        && IMPLICATION(ocb_inner_most, ocb == 0);

                for_(int kd = 0; kd < jbgp.kd; kd++)
                for_(int kh = 0; kh < jbgp.kh; kh++)
                for (int kw = 0; kw < jbgp.kw; kw++) {
                    ker(ithr_oc_mb, nthr_oc_mb, ithr_ic, osb, osb_s, ocb, ocb_s,
                            icc, icc_start, kd, kh, kw, copy_buffer_a,
                            prev_ker_idx, dst_scales_inv_ptr);
                }

                ++loop_start;
                switch (order) {
                    case loop_order_t::osc_occ_icc_osb_ocb:
                        nd_iterator_step(
                                icc, icc_work, osb, osb_work, ocb, ocb_work);
                        break;
                    case loop_order_t::osc_occ_osb_ocb_icc:
                        nd_iterator_step(
                                osb, osb_work, ocb, ocb_work, icc, icc_work);
                        break;
                    case loop_order_t::icc_osc_occ_osb_ocb:
                        nd_iterator_step(osb, osb_work, ocb, ocb_work);
                        break;
                    case loop_order_t::icc_occ_osc_ocb_osb:
                        nd_iterator_step(ocb, ocb_work, osb, osb_work);
                        break;
                }
            }

            ++work;
            switch (order) {
                case loop_order_t::osc_occ_icc_osb_ocb:
                case loop_order_t::osc_occ_osb_ocb_icc:
                    nd_iterator_step(osc, os_chunks, occ, oc_chunks);
                    break;
                case loop_order_t::icc_osc_occ_osb_ocb:
                    nd_iterator_step(osc, os_chunks, occ, oc_chunks);
                    if (work == end) {
                        icc++;
                        if (icc < icc_work) {
                            work = start;
                            nd_iterator_init(
                                    work, osc, os_chunks, occ, oc_chunks);
                        }
                    }
                    break;
                case loop_order_t::icc_occ_osc_ocb_osb:
                    nd_iterator_step(occ, oc_chunks, osc, os_chunks);
                    if (work == end) {
                        icc++;
                        if (icc < icc_work) {
                            work = start;
                            nd_iterator_init(
                                    work, occ, oc_chunks, osc, os_chunks);
                        }
                    }
                    break;
            }
        }
        if (is_amx) amx_tile_release();
    });

    if (jbgp.nthr_ic_b > 1) {
        // Explicit return type: `dim_t` and `size_t` have different ranks on
        // macOS, so the two returns would deduce to different types.
        const auto get_dst_reduced_off
                = [=, &jbgp](int ithr_ic, dim_t osb, dim_t ocb) -> size_t {
            assert(jbgp.nthr_ic_b > 1);
            dim_t os = osb * jbgp.os_block;
            dim_t oc = ocb * jbgp.oc_block;
            // use accumulation data type size for buffer offset computation
            const size_t dst_off = get_blk_off(dst_d, jbgp.acc_dt, os, oc);
            if (ithr_ic == 0) return dst_off;
            assert(ithr_ic > 0);
            // shift buffer idx if dst is used as accumulation buffer
            const int ic_buf_idx = ithr_ic - acc_buffer_idx_shift;
            return dst_off + (acc_dt_size * jbgp.mb * jbgp.LDC * ic_buf_idx);
        };

        parallel(num_threads,
                [=, &jbgp COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            int nthr_ic {1}, nthr_oc_mb {1}, ithr_ic {0}, ithr_oc_mb {0};
            bool ok = init_thr_groups(
                    ithr, nthr, nthr_ic, nthr_oc_mb, ithr_ic, ithr_oc_mb);
            if (!ok) return;

            int ocmb_start {0}, ocmb_end {0};
            int start {0}, end {0};
            balance211(
                    work_amount, nthr_oc_mb, ithr_oc_mb, ocmb_start, ocmb_end);
            balance211(ocmb_end - ocmb_start, nthr_ic, ithr_ic, start, end);

            int prev_ker_idx = -1;

            int occ {0}, osc {0};
            nd_iterator_init(
                    ocmb_start + start, osc, os_chunks, occ, oc_chunks);
            while (start < end) {
                dim_t ocb_s = occ * jbgp.nb_oc_blocking;
                dim_t ocb_e = nstl::min<dim_t>(
                        ocb_s + jbgp.nb_oc_blocking, jbgp.nb_oc);

                dim_t osb_s = osc * jbgp.nb_os_blocking;
                dim_t osb_e = nstl::min<dim_t>(
                        osb_s + jbgp.nb_os_blocking, jbgp.nb_os);

                for (dim_t osb = osb_s; osb < osb_e; ++osb) {
                    dim_t cur_os_block = nstl::min<dim_t>(
                            jbgp.os - osb * jbgp.os_block, jbgp.os_block);
                    const bool is_os_tail = cur_os_block < jbgp.os_block;
                    const dim_t cur_oc_chunk_size
                            = nstl::min<dim_t>(jbgp.LDC, ocb_e * jbgp.oc_block)
                            - ocb_s * jbgp.oc_block;
                    char *dst_reduced
                            = (can_use_dst_as_acc_buffer ? dst
                                                         : c_buffer_global)
                            + get_dst_reduced_off(0, osb, ocb_s);
                    const size_t os_offset = jbgp.LDC * acc_dt_size;
                    for (int ic_buf = 0; ic_buf < nthr_ic - 1; ++ic_buf) {
                        const char *c_buffer = c_buffer_global
                                + get_dst_reduced_off(ic_buf + 1, osb, ocb_s);
                        for (int os = 0; os < cur_os_block; ++os) {
                            acc_ker_->accumulate(
                                    (float *)(dst_reduced + os * os_offset),
                                    (float *)(c_buffer + os * os_offset),
                                    cur_oc_chunk_size);
                        }
                    }
                    if (are_post_ops_applicable) {
                        for (dim_t ocb = ocb_s; ocb < ocb_e; ++ocb) {
                            const bool is_oc_tail
                                    = (jbgp.oc - ocb * jbgp.oc_block
                                            < jbgp.oc_block);
                            const int brg_ker_idx = brgemm_inner_product_utils::
                                    get_brg_kernel_index(false, false,
                                            is_os_tail, is_oc_tail, false);
                            brgemm_palettes_.maybe_tile_configure(
                                    is_amx, prev_ker_idx, brg_ker_idx);
                            const auto brg_kernel
                                    = brg_kernels_[brg_ker_idx].get();
                            const dim_t os = osb * jbgp.os_block;
                            const dim_t oc = ocb * jbgp.oc_block;
                            const auto ptr_bias = jbgp.with_bias
                                    ? bias + bia_dt_size * oc
                                    : nullptr;
                            auto ptr_D = dst
                                    + get_blk_off(dst_d, jbgp.dst_dt, os, oc);
                            auto ptr_C = can_use_dst_as_acc_buffer
                                    ? ptr_D
                                    : c_buffer_global
                                            + get_dst_reduced_off(0, osb, ocb);

                            char *wsp_tile = is_amx ? wsp_tile_base
                                            + ithr * jbgp.amx_buf_size_per_thread
                                                    : nullptr;

                            void *scratch = is_amx
                                    ? static_cast<void *>(wsp_tile)
                                    : (jbgp.req_s8s8_compensation
                                                      ? static_cast<
                                                                void *>(const_cast<
                                                                int *>(
                                                                &compensation
                                                                        [oc]))
                                                      : nullptr);

                            const brgemm_post_ops_data_t post_ops_data {
                                    static_cast<const void *>(ptr_bias),
                                    post_ops_binary_rhs_arg_vec.data(),
                                    static_cast<size_t>(oc), 0, dst, 0, nullptr,
                                    nullptr, nullptr, true /* skip_accm */, 1,
                                    false, false, src_scales,
                                    wei_scales ? static_cast<const char *>(
                                                         wei_scales)
                                                    + jbgp.is_oc_scale * oc
                                                            * sizeof(float)
                                               : nullptr,
                                    dst_scales};

                            brgemm_kernel_execute_postops(brg_kernel, 0,
                                    nullptr, (void *)ptr_C, (void *)ptr_D,
                                    post_ops_data, scratch);
                        }
                    }
                }
                ++start;
                nd_iterator_step(osc, os_chunks, occ, oc_chunks);
            }
        });
    }

    return status::success;
}

template struct brgemm_inner_product_fwd_t<avx2>;
template struct brgemm_inner_product_fwd_t<avx2_vnni>;
template struct brgemm_inner_product_fwd_t<avx2_vnni_2>;
template struct brgemm_inner_product_fwd_t<avx512_core>;
template struct brgemm_inner_product_fwd_t<avx512_core_bf16>;
template struct brgemm_inner_product_fwd_t<avx512_core_vnni>;
template struct brgemm_inner_product_fwd_t<avx512_core_amx>;
template struct brgemm_inner_product_fwd_t<avx512_core_fp16>;
template struct brgemm_inner_product_fwd_t<avx512_core_amx_fp16>;
template struct brgemm_inner_product_fwd_t<avx10_2>;
template struct brgemm_inner_product_fwd_t<avx10_2_amx_2>;

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_data_t<isa>::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto diff_dst_ = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto weights_ = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto diff_src_ = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    auto diff_src = const_cast<char *>(diff_src_);
    auto weights = const_cast<char *>(weights_);
    auto diff_dst = const_cast<char *>(diff_dst_);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jbgp = pd()->jbgp_;

    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_bf16 = everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_f16 = everyone_is(f16, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_f32_out = jbgp.src_dt == f32;
    const bool is_amx = jbgp.is_amx;
    const size_t buf_dt_size = types::data_type_size(
            isa == avx512_core_fp16 ? f32 : jbgp.wei_dt);

    const dim_t wei_dt_size = types::data_type_size(jbgp.wei_dt);

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    brgemm_batch_element_t *addr_batch_global
            = scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);
    char *c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    char *b_buffer_global = jbgp.use_buffer_b
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
            : nullptr;
    char *a_buffer_global = jbgp.use_buffer_a
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
            : nullptr;
    auto wsp_tile_base = is_amx
            ? ctx.get_scratchpad_grantor().template get<char>(
                      key_conv_amx_tile_buffer)
            : nullptr;

    const dim_t acc_dt_sz = types::data_type_size(jbgp.acc_dt);
    const dim_t src_dt_sz = types::data_type_size(jbgp.src_dt);

    const int oc_chunks
            = static_cast<int>(div_up(jbgp.nb_oc, jbgp.nb_oc_blocking));
    const int os_chunks
            = static_cast<int>(div_up(jbgp.nb_os, jbgp.nb_os_blocking));
    const int work_amount
            = static_cast<int>(jbgp.nb_ic * jbgp.ks() * os_chunks);
    const int num_threads
            = work_amount == 1 && jbgp.nthr_oc_b <= 1 ? 1 : jbgp.nthr;

    const auto get_weights_ptr
            = [=](dim_t icb, dim_t ocb, int kd = 0, int kh = 0, int kw = 0) {
        int fwd_ic_block
                = (is_amx && !jbgp.is_bf32) ? 2 * jbgp.simd_w : jbgp.simd_w;
        int fwd_oc_block = jbgp.get_weights_oc_block();
        dim_t ic = icb * jbgp.ic_block;
        dim_t oc = ocb * jbgp.oc_block;

        dim_t fwd_icb = ic / fwd_ic_block;
        dim_t fwd_ocb = oc / fwd_oc_block;
        dim_t fwd_icb_simd = ic % fwd_ic_block;
        dim_t fwd_ocb_simd = oc % fwd_oc_block;

        char *ptr_wei_local
                = weights + blk_off(weights_d, fwd_ocb, fwd_icb, kd, kh, kw);

        int blk_sz = is_bf16 || (is_f16 && isa != avx512_core_fp16) ? 2 : 1;

        return ptr_wei_local
                + wei_dt_size
                * (rnd_dn(fwd_icb_simd, blk_sz) * fwd_oc_block
                        + blk_sz * fwd_ocb_simd);
    };

    const auto transform_b_chunk
            = [= COMPAT_THIS_CAPTURE](char *tr_wei, const char *wei,
                      int trans_batch, dim_t current_N, dim_t current_K) {
        auto ctx = jit_brgemm_trans_wei_t::ctx_t();
        ctx.src = (void *)wei;
        ctx.tr_src = (void *)tr_wei;
        ctx.current_gemm_batch = trans_batch;
        ctx.current_N = current_N;
        ctx.current_K = current_K;
        (*trans_B_kernel_)(&ctx);
    };

    const auto ker = [= COMPAT_THIS_CAPTURE](int ithr_ic_mb, int nthr_ic_mb,
                             int ithr_oc, int nthr_oc, int n, int icb, int occ,
                             int kd, int kh, int kw, bool do_init,
                             bool do_b_transpose, int &prev_ker_idx) {
        const int ithr = nthr_ic_mb * ithr_oc + ithr_ic_mb;
        brgemm_batch_element_t *addr_batch
                = addr_batch_global + ithr * jbgp.adjusted_batch_size;

        const int ic = icb * jbgp.ic_block;
        const int ocb = occ * jbgp.nb_oc_blocking;
        const int oc = ocb * jbgp.oc_block;
        const size_t dsrc_off = blk_off(diff_src_d, n, ic, kd, kh, kw);
        const int adj_buffers = (jbgp.src_dt == f32) ? 1 : 0;
        const size_t c_buf_shift = jbgp.nthr_oc_b > 1
                ? (ithr_oc - adj_buffers)
                        * static_cast<size_t>(jbgp.mb * jbgp.LDC)
                : ithr * static_cast<size_t>(jbgp.LDC * jbgp.M);
        const size_t c_buf_off
                = types::data_type_size(jbgp.acc_dt) * c_buf_shift
                + (jbgp.nthr_oc_b > 1 ? acc_dt_sz * dsrc_off / src_dt_sz : 0);
        bool use_c_buf = false;
        if (is_f32_out && jbgp.use_buffer) {
            use_c_buf = (jbgp.nthr_oc_b == 1 || ithr_oc > 0);
        } else if (!is_f32_out && jbgp.use_buffer) {
            if (jbgp.nthr_oc_b > 1)
                use_c_buf = true;
            else
                use_c_buf = (jbgp.nthr_oc_b == 1 || ithr_oc > 0);
        }

        const size_t a_buffer_size_per_thr
                = jbgp.os_block * jbgp.LDA * types::data_type_size(jbgp.dst_dt);
        char *c_buffer = use_c_buf ? c_buffer_global + c_buf_off : nullptr;
        char *a_buffer = jbgp.use_buffer_a
                ? a_buffer_global + ithr * a_buffer_size_per_thr
                : diff_dst;
        char *wsp_tile = is_amx
                ? wsp_tile_base + ithr * jbgp.amx_buf_size_per_thread
                : nullptr;

        bool kernel_init = do_init;

        const bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        const bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
        const bool is_last_oc_chunk = occ == oc_chunks - 1;
        const bool is_oc_tail = is_last_oc_chunk && jbgp.K_tail > 0;

        const int rnd_oc
                = rnd_up(jbgp.oc, jbgp.use_buffer_a ? jbgp.oc_block : 1);
        const int nb_oc_b
                = nstl::min((rnd_oc - oc) / jbgp.oc_block, jbgp.nb_oc_blocking);

        auto is_bs_tail = (nb_oc_b != jbgp.nb_oc_blocking);
        const int brg_ker_idx
                = brgemm_inner_product_utils::get_brg_kernel_index(
                        is_bs_tail, kernel_init, is_os_tail, is_ic_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();

        const dim_t size_B = jbgp.LDB * rnd_up(jbgp.K, 2);

        const size_t b_buf_shift = jbgp.global_b_transpose
                ? icb * jbgp.nb_oc + ocb
                : ithr * jbgp.gemm_batch_size;
        const size_t b_buf_off = buf_dt_size * b_buf_shift * size_B;
        char *b_buffer = b_buffer_global + b_buf_off;

        char *ptr_D = diff_src + dsrc_off;
        char *ptr_C = use_c_buf ? c_buffer : ptr_D;

        if (jbgp.use_buffer_a)
            copy_data_chunk(copy_diff_dst_kernel_, a_buffer,
                    diff_dst + get_blk_off(diff_dst_d, jbgp.dst_dt, n, oc),
                    is_os_tail ? jbgp.os - n : jbgp.os_block, is_last_oc_chunk);

        if (nb_oc_b > 0 && brg_kernel != nullptr) {
            brgemm_palettes_.maybe_tile_configure(
                    is_amx, prev_ker_idx, brg_ker_idx);
            for (int b = 0; b < nb_oc_b; b++) {
                addr_batch[b].ptr.A = jbgp.use_buffer_a ? a_buffer
                                + b * jbgp.oc_block
                                        * types::data_type_size(jbgp.dst_dt)
                                                        : diff_dst
                                + get_blk_off(diff_dst_d, jbgp.dst_dt, n,
                                        oc + b * jbgp.oc_block);
                addr_batch[b].ptr.B = b_buffer + buf_dt_size * (b * size_B);
                if (!jbgp.global_b_transpose && do_b_transpose)
                    transform_b_chunk((char *)addr_batch[b].ptr.B,
                            get_weights_ptr(icb, ocb + b, kd, kh, kw), 1,
                            is_ic_tail ? jbgp.ic % jbgp.ic_block
                                       : jbgp.ic_block,
                            jbgp.oc_block);
            }

            if (jbgp.use_buffer && (jbgp.nthr_oc_b <= 1 || num_threads == 1)
                    && is_last_oc_chunk && !is_oc_tail) {
                void *scratch
                        = is_amx ? static_cast<void *>(wsp_tile) : nullptr;
                const brgemm_post_ops_data_t empty_po_data {};
                brgemm_kernel_execute_postops(brg_kernel, nb_oc_b, addr_batch,
                        (void *)c_buffer, (void *)ptr_D, empty_po_data,
                        scratch);

            } else {
                brgemm_kernel_execute(brg_kernel, nb_oc_b, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }
        if (is_oc_tail) {
            assert(!jbgp.use_buffer_a);
            auto use_init_ker = (kernel_init && nb_oc_b == 0);
            const int brg_kernel_oc_tail_idx
                    = brgemm_inner_product_utils::get_brg_kernel_index(
                            false, use_init_ker, is_os_tail, is_ic_tail, true);
            brgemm_palettes_.maybe_tile_configure(
                    is_amx, prev_ker_idx, brg_kernel_oc_tail_idx);

            const int oc_block = nb_oc_b;
            addr_batch[0].ptr.A = diff_dst
                    + get_blk_off(diff_dst_d, jbgp.dst_dt, n,
                            oc + oc_block * jbgp.oc_block);
            addr_batch[0].ptr.B = b_buffer + buf_dt_size * (oc_block * size_B);
            if (!jbgp.global_b_transpose && do_b_transpose) {
                transform_b_chunk((char *)addr_batch[0].ptr.B,
                        get_weights_ptr(icb, ocb + oc_block, kd, kh, kw), 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.K_tail);
            }

            auto brg_kernel_oc_tail
                    = brg_kernels_[brg_kernel_oc_tail_idx].get();
            if (jbgp.use_buffer && jbgp.nthr_oc_b <= 1) {
                void *scratch
                        = is_amx ? static_cast<void *>(wsp_tile) : nullptr;
                const brgemm_post_ops_data_t empty_po_data {};
                brgemm_kernel_execute_postops(brg_kernel_oc_tail, 1, addr_batch,
                        (void *)c_buffer, (void *)ptr_D, empty_po_data,
                        scratch);

            } else {
                brgemm_kernel_execute(brg_kernel_oc_tail, 1, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
        }
    };

    if (jbgp.global_b_transpose && jbgp.use_buffer_b) {
        assert(jbgp.nthr_oc_b == 1
                && "No global B transpose support for oc-reduction.");
        assert(!jbgp.has_spatial_dims()
                && "No global B transpose support for spatial dims.");

        parallel(num_threads, [=](const int ithr, const int nthr) {
            int start {0}, end {0};
            int max_ch_block = nstl::max(jbgp.ic_block, jbgp.oc_block);
            int ic_chunk_sz = max_ch_block / jbgp.ic_block;
            int oc_chunk_sz = max_ch_block / jbgp.oc_block;
            int nc_ic = static_cast<int>(div_up(jbgp.nb_ic, ic_chunk_sz));
            int nc_oc = static_cast<int>(div_up(jbgp.nb_oc, oc_chunk_sz));
            int transp_work_amount = nc_ic * nc_oc;
            balance211(transp_work_amount, nthr, ithr, start, end);
            int icc, occ;
            nd_iterator_init(start, icc, nc_ic, occ, nc_oc);
            while (start < end) {
                dim_t icb_start = icc * ic_chunk_sz;
                dim_t icb_end
                        = nstl::min<dim_t>((icc + 1) * ic_chunk_sz, jbgp.nb_ic);
                dim_t ocb_start = occ * oc_chunk_sz;
                dim_t ocb_end
                        = nstl::min<dim_t>((occ + 1) * oc_chunk_sz, jbgp.nb_oc);
                for_(dim_t icb = icb_start; icb < icb_end; icb++)
                for (dim_t ocb = ocb_start; ocb < ocb_end; ocb++) {
                    dim_t ic = icb * jbgp.ic_block;
                    dim_t oc = ocb * jbgp.oc_block;
                    bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
                    bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
                    const dim_t size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
                    char *b_buffer = b_buffer_global
                            + buf_dt_size
                                    * ((dim_t)icb * jbgp.nb_oc * size_B
                                            + (dim_t)ocb * size_B);

                    transform_b_chunk(b_buffer, get_weights_ptr(icb, ocb), 1,
                            is_ic_tail ? jbgp.ic % jbgp.ic_block
                                       : jbgp.ic_block,
                            is_oc_tail ? jbgp.oc % jbgp.oc_block
                                       : jbgp.oc_block);
                }
                ++start;
                nd_iterator_step(icc, nc_ic, occ, nc_oc);
            }
        });
    }

    parallel(num_threads, [=](const int ithr, const int nthr) {
        const int nthr_oc = jbgp.nthr_oc_b <= nthr ? jbgp.nthr_oc_b : 1;
        const int nthr_ic_mb = nthr / nthr_oc;
        const int ithr_ic_mb = ithr % nthr_ic_mb;
        const int ithr_oc = ithr / nthr_ic_mb;
        if (ithr_ic_mb >= work_amount || ithr_oc >= oc_chunks
                || ithr >= rnd_dn(nthr, nthr_oc))
            return;

        int start {0}, end {0};
        balance211(work_amount, nthr_ic_mb, ithr_ic_mb, start, end);
        int occ_start {0}, occ_end {oc_chunks};
        if (nthr_oc > 1)
            balance211(oc_chunks, nthr_oc, ithr_oc, occ_start, occ_end);

        // For keeping track of last tile configuration used.
        int prev_ker_idx = -1;

        int icb {0}, osc {0}, work {start};
        int kd {0}, kh {0}, kw {0};
        nd_iterator_init(work, osc, os_chunks, kd, jbgp.kd, kh, jbgp.kh, kw,
                jbgp.kw, icb, jbgp.nb_ic);
        while (work < end) {
            const int nb_os_blocking = static_cast<int>(
                    nstl::min<dim_t>(jbgp.nb_os - osc * jbgp.nb_os_blocking,
                            jbgp.nb_os_blocking));
            const int occ_work = occ_end - occ_start;
            const int loop_iteration = nb_os_blocking * occ_work;

            for (int iter = 0; iter < loop_iteration; ++iter) {
                int osb = 0, occ = occ_start;
                if (jbgp.use_buffer || !is_f32) {
                    osb += iter / occ_work;
                    occ += iter % occ_work;
                } else {
                    occ += iter / nb_os_blocking;
                    osb += iter % nb_os_blocking;
                }
                int n = (osc * jbgp.nb_os_blocking + osb) * jbgp.os_block;
                ker(ithr_ic_mb, nthr_ic_mb, ithr_oc, nthr_oc, n, icb, occ, kd,
                        kh, kw, occ == occ_start, osb == 0 || occ_work > 1,
                        prev_ker_idx);
            }
            ++work;
            nd_iterator_step(osc, os_chunks, kd, jbgp.kd, kh, jbgp.kh, kw,
                    jbgp.kw, icb, jbgp.nb_ic);
        }
        if (is_amx) amx_tile_release();
    });

    if (jbgp.nthr_oc_b > 1) {
        parallel(num_threads,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            const int nthr_oc = jbgp.nthr_oc_b <= nthr ? jbgp.nthr_oc_b : 1;
            const int n_oc_bufs = nstl::min(oc_chunks, nthr_oc);
            if (n_oc_bufs <= 1) return;

            const dim_t dsrc_elems = jbgp.LDC * jbgp.os;
            const int reduce_chunk_size = 64;
            int start {0}, end {0};
            balance211(static_cast<int>(div_up(dsrc_elems, reduce_chunk_size)),
                    nthr, ithr, start, end);
            const dim_t reduce_start = start * reduce_chunk_size;
            const dim_t reduce_finish
                    = nstl::min<dim_t>(end * reduce_chunk_size, dsrc_elems);
            if (reduce_finish <= reduce_start) return;
            const dim_t elems_to_reduce = reduce_finish - reduce_start;

            char *dsrc_reduced = diff_src + src_dt_sz * reduce_start;
            char *c_buffer_start = c_buffer_global + acc_dt_sz * reduce_start;

            float *out_buffer = is_f32_out
                    ? reinterpret_cast<float *>(dsrc_reduced)
                    : reinterpret_cast<float *>(c_buffer_start);
            int oc_buf_idx = !is_f32_out;
            int oc_buf_end = is_f32_out;
            int n_accs = n_oc_bufs - oc_buf_end;
            for (int oc_buf = oc_buf_idx; oc_buf < n_accs; oc_buf++) {
                const dim_t c_buf_offt = acc_dt_sz
                        * (oc_buf * jbgp.os * jbgp.LDC + reduce_start);
                char *c_buffer = c_buffer_global + c_buf_offt;

                acc_ker_->accumulate((float *)out_buffer, (float *)c_buffer,
                        elems_to_reduce);
                if (!is_f32_out && oc_buf == n_accs - 1) {
                    if (is_bf16) {
                        cvt_float_to_bfloat16((bfloat16_t *)dsrc_reduced,
                                (const float *)out_buffer, elems_to_reduce);
                    } else if (is_f16) {
                        cvt_float_to_float16((float16_t *)dsrc_reduced,
                                (const float *)out_buffer, elems_to_reduce);
                    }
                }
            }
        });
    }
}

template struct brgemm_inner_product_bwd_data_t<avx2>;
template struct brgemm_inner_product_bwd_data_t<avx512_core>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_amx>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_bf16>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_amx_fp16>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_fp16>;
template struct brgemm_inner_product_bwd_data_t<avx10_2>;

template <cpu_isa_t isa>
struct brgemm_inner_product_bwd_weights_t<isa>::thread_info_t {
    const char *src;
    const char *diff_dst;
    char *diff_weights;
    char *diff_bias;

    const memory_tracking::grantor_t &scratchpad;

    char *buffer_c = nullptr;
    char *buffer_bias = nullptr;
    char *wsp_tile_base = nullptr;

    int ithr;
    int ithr_sp_ic_c, ithr_oc_c, ithr_os_c;
    int nthr;
    int nthr_sp_ic_c, nthr_oc_c, nthr_os_c;

    int os_c_start = 0, os_c_end = 0, os_c_work;
    int oc_c_start = 0, oc_c_end = 0, oc_c_work;
    int sp_ic_c_start = 0, sp_ic_c_end = 0, sp_ic_c_work;
    simple_barrier::ctx_t *barrier_ctx;

    thread_info_t(const brgemm_inner_product_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : src(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
        , diff_dst(CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST))
        , diff_weights(CTX_OUT_MEM(char *, DNNL_ARG_DIFF_WEIGHTS))
        , diff_bias(CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS))
        , scratchpad(ctx.get_scratchpad_grantor())
        , ithr(ithr)
        , buffer_a_(scratchpad.template get<char>(
                  key_brgemm_primitive_buffer_a)) {

        const auto &jbgp = self->pd()->jbgp_;

        const bool is_amx = jbgp.is_amx;

        buffer_c = (jbgp.use_buffer)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buffer_bias = (jbgp.with_bias
                              && (jbgp.bia_dt != data_type::f32
                                      || jbgp.nthr_mb > 1))
                ? scratchpad.template get<char>(key_iprod_bias_bf16_convert_wsp)
                : nullptr;

        buffer_b_ = jbgp.use_buffer_b
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        thread_local_input_buffers_ = jbgp.local_buffers_for_input_tensors;
        int ic_chunks
                = static_cast<int>(div_up(jbgp.nb_ic, jbgp.nb_ic_blocking));
        int os_chunks
                = static_cast<int>(div_up(jbgp.nb_os, jbgp.nb_os_blocking));
        int sp_ic_chunks = ic_chunks * jbgp.ks();
        const size_t os_chunks_per_thr = div_up(os_chunks, jbgp.nthr_mb);
        const size_t num_os_chunks_per_thread
                = thread_local_input_buffers_ ? 1 : os_chunks_per_thr;

        nb_ic_blocking_ = jbgp.nb_ic_blocking;
        nb_oc_blocking_ = jbgp.nb_oc_blocking;
        ic_chunks_ = ic_chunks;

        if (jbgp.use_buffer_a) {
            const size_t dt_sz = buf_dt_size(jbgp.src_dt, jbgp.isa);
            const size_t sp_ic_chunks_per_thr
                    = div_up(sp_ic_chunks, jbgp.nthr_ic_b);
            const size_t num_sp_ic_chunks_per_thread
                    = thread_local_input_buffers_ ? 1 : sp_ic_chunks_per_thr;
            sp_ic_chunks_per_thread_ = num_sp_ic_chunks_per_thread;
            const size_t block_A_size = dt_sz * jbgp.LDA * jbgp.M;
            const size_t os_chunk_A_buffer
                    = jbgp.gemm_batch_size * block_A_size;
            const size_t ic_os_chunk_A_buffer
                    = jbgp.nb_ic_blocking * os_chunk_A_buffer;

            buffer_a_icb_shift_ = os_chunk_A_buffer;
            buffer_a_osb_shift_ = block_A_size;
            buffer_a_osc_shift_ = thread_local_input_buffers_
                    ? 0
                    : sp_ic_chunks_per_thr * ic_os_chunk_A_buffer;
            const size_t buffer_a_thread_shift = num_sp_ic_chunks_per_thread
                    * num_os_chunks_per_thread * ic_os_chunk_A_buffer;

            buffer_a_ = buffer_a_ + ithr * buffer_a_thread_shift;
        }

        if (jbgp.use_buffer_b) {
            const auto buf_dt = jbgp.dst_dt == f16 && isa == avx512_core_fp16
                    ? data_type::f32
                    : jbgp.dst_dt;
            const size_t dt_sz = buf_dt_size(jbgp.dst_dt, jbgp.isa);
            assert(types::data_type_size(buf_dt) == dt_sz);
            const size_t block_B_size = dt_sz * jbgp.LDB * jbgp.K;
            const size_t os_chunk_B_buffer
                    = jbgp.gemm_batch_size * block_B_size;
            buffer_b_ocb_shift_ = dt_sz * jbgp.oc_block
                    * data_type_vnni_granularity(buf_dt);
            buffer_b_osb_shift_ = block_B_size;
            buffer_b_osc_shift_
                    = thread_local_input_buffers_ ? 0 : os_chunk_B_buffer;

            const size_t buffer_b_thread_shift
                    = num_os_chunks_per_thread * os_chunk_B_buffer;

            buffer_b_ = buffer_b_ + ithr * buffer_b_thread_shift;
        }

        wsp_tile_base = is_amx
                ? ctx.get_scratchpad_grantor().template get<char>(
                          key_conv_amx_tile_buffer)
                : nullptr;

        nthr = jbgp.nthr;
        nthr_sp_ic_c = jbgp.nthr_ic_b;
        nthr_oc_c = jbgp.nthr_oc_b;
        nthr_os_c = jbgp.nthr_mb;

        // Set thread ID per dimension.
        nd_iterator_init(ithr, ithr_os_c, nthr_os_c, ithr_oc_c, nthr_oc_c,
                ithr_sp_ic_c, nthr_sp_ic_c);

        int oc_chunks
                = static_cast<int>(div_up(jbgp.nb_oc, jbgp.nb_oc_blocking));

        /* reduction dimension */
        balance211(os_chunks, nthr_os_c, ithr_os_c, os_c_start, os_c_end);
        os_c_work = os_c_end - os_c_start;

        balance211(oc_chunks, nthr_oc_c, ithr_oc_c, oc_c_start, oc_c_end);
        oc_c_work = oc_c_end - oc_c_start;

        balance211(sp_ic_chunks, nthr_sp_ic_c, ithr_sp_ic_c, sp_ic_c_start,
                sp_ic_c_end);
        sp_ic_c_work = sp_ic_c_end - sp_ic_c_start;

        if (dnnl_thr_syncable())
            barrier_ctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx);
    }

    char *get_buffer_a_ptr(int sp_icb, int osc) const {
        if (!buffer_a_) return (char *)nullptr;

        int icb_idx = sp_icb % (sp_ic_chunks_per_thread_ * nb_ic_blocking_);
        int osc_idx = thread_local_input_buffers_ ? 0 : osc - os_c_start;

        return buffer_a_ + osc_idx * buffer_a_osc_shift_
                + icb_idx * buffer_a_icb_shift_;
    }

    char *get_buffer_b_ptr(int ocb, int osc) const {
        if (!buffer_b_) return (char *)nullptr;

        const int ocb_idx = ocb % nb_oc_blocking_;
        const int osc_idx = thread_local_input_buffers_ ? 0 : osc - os_c_start;

        return buffer_b_ + osc_idx * buffer_b_osc_shift_
                + ocb_idx * buffer_b_ocb_shift_;
    }

    size_t get_buffer_a_osb_shift() const { return buffer_a_osb_shift_; }
    size_t get_buffer_b_osb_shift() const { return buffer_b_osb_shift_; }

    int ic_chunks() const { return ic_chunks_; }

private:
    char *buffer_a_ = nullptr;
    char *buffer_b_ = nullptr;

    bool thread_local_input_buffers_ = false;
    int nb_ic_blocking_ = 1;
    int nb_oc_blocking_ = 1;
    size_t buffer_a_icb_shift_ = 0;
    size_t buffer_a_osc_shift_ = 0;
    size_t buffer_a_osb_shift_ = 0;
    size_t buffer_b_ocb_shift_ = 0;
    size_t buffer_b_osc_shift_ = 0;
    size_t buffer_b_osb_shift_ = 0;

    int ic_chunks_ = 0;
    int sp_ic_chunks_per_thread_ = 0;
};

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transform_matrix_a_chunk(
        char *tr_src, const char *src, int trans_batch, int current_m,
        int current_k) const {
    auto ctx = jit_brgemm_trans_src_t::ctx_t();
    ctx.src = (void *)src;
    ctx.tr_src = (void *)tr_src;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_M = current_m;
    ctx.current_K = current_k;
    (*trans_A_kernel_)(&ctx);
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transform_matrix_b_chunk(
        char *tr_diff_dst, const char *diff_dst, int trans_batch,
        int current_col_size, int current_row_size) const {
    auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
    ctx.src = (void *)diff_dst;
    ctx.tr_src = (void *)tr_diff_dst;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_col_size = current_col_size;
    ctx.current_row_size = current_row_size;
    (*trans_B_kernel_)(&ctx);
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transpose_matrix_c_chunk(
        const thread_info_t *ti, const dim_t ocb, const dim_t icb, int oc_size,
        int ic_size, dim_t kd, dim_t kh, dim_t kw, bool is_reduction) const {
    const auto &jbgp = pd()->jbgp_;

    if (jbgp.is_amx) {
        auto p = jit_amx_ip_trans_diff_wei_t::ctx_t();

        // Note: This assumes AxB{inner_blocking} weights memory format.
        const dim_t ext_nb_ic = div_up(jbgp.ic, ext_ic_block_);
        const dim_t ext_block_nelems
                = static_cast<dim_t>(ext_ic_block_) * ext_oc_block_;
        const dim_t n_sp_slices = jbgp.ks();

        dim_t ext_icb = icb * (jbgp.ic_block / ext_ic_block_);
        dim_t ext_ocb = ocb * (jbgp.oc_block / ext_oc_block_);
        dim_t sp_slice = kd * jbgp.kh * jbgp.kw + kh * jbgp.kw + kw;

        dim_t icb_shift = ext_icb * ext_block_nelems;
        dim_t ocb_shift = ext_ocb * ext_nb_ic * ext_block_nelems * n_sp_slices;
        dim_t sp_slice_shift = sp_slice * ext_nb_ic * ext_block_nelems;
        dim_t out_shift = sp_slice_shift + ocb_shift + icb_shift;

        size_t out_offset = types::data_type_size(jbgp.wei_dt) * out_shift;

        p.src = get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw, 0);
        p.dst = (void *)(ti->diff_weights + out_offset);

        p.last_ic_block = (jbgp.ic <= ext_ic_block_
                || (jbgp.nb_ic > 1 && icb == jbgp.nb_ic - 1));
        p.last_oc_block = (jbgp.oc <= ext_oc_block_
                || (jbgp.nb_oc > 1 && ocb == jbgp.nb_oc - 1));
        (*diff_wei_trans_kernel_)(&p);
    } else {
        auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
        ctx.src = (void *)(get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw, 0));

        const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
        size_t wei_off = blk_off(diff_weights_d, ocb, icb, kd, kh, kw);
        ctx.tr_src = (void *)(ti->diff_weights + wei_off);

        ctx.current_gemm_batch = 1;
        ctx.current_col_size = oc_size;
        ctx.current_row_size = ic_size;
        (*trans_C_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
char *brgemm_inner_product_bwd_weights_t<isa>::get_wei_acc_ptr(
        const thread_info_t *ti, dim_t ocb, dim_t icb, dim_t kd, dim_t kh,
        dim_t kw, int reduction_buf_idx) const {
    const auto &jbgp = pd()->jbgp_;

    const int reduction_buf_start_idx = jbgp.wei_dt == f32;
    // reduction_buf_idx argument allows manually set up required reduction
    // buffer index, required for reduction and transform diff_weights parts.
    // It has value -1 by default. If reduction_buf_idx < 0 then ti->ithr_os_c
    // is used for calculation of the current reduction index.
    const int buf_idx = reduction_buf_idx >= 0
            ? reduction_buf_idx
            : (ti->ithr_os_c - reduction_buf_start_idx);
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);

    if ((jbgp.nthr_mb > 1 && buf_idx < 0)
            || (jbgp.wei_dt == jbgp.acc_dt && reduction_buf_idx < 0
                    && ti->ithr_os_c == 0)) {
        MAYBE_UNUSED(reduction_buf_idx);
        const int icb_scale = (!jbgp.is_amx || jbgp.wei_dt == jbgp.acc_dt)
                ? jbgp.ic_block / jbgp.simd_w
                : 1;
        const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
        size_t dwei_off
                = blk_off(diff_weights_d, ocb, icb * icb_scale, kd, kh, kw);
        return (char *)ti->diff_weights + dwei_off;
    }

    if (!jbgp.use_buffer) return nullptr;

    const int ocb_l = ocb % jbgp.nb_oc_blocking;
    const int icb_l = icb % jbgp.nb_ic_blocking;

    if (jbgp.nthr_mb > 1 || jbgp.harness == harness_mb_reduction) {
        const size_t icc = icb / jbgp.nb_ic_blocking;
        const size_t occ = ocb / jbgp.nb_oc_blocking;
        const size_t num_ic_chunks = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
        const size_t num_oc_chunks = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
        const size_t block_size = acc_dt_size * jbgp.ic_block * jbgp.oc_block;
        const size_t chunk_size
                = block_size * jbgp.nb_ic_blocking * jbgp.nb_oc_blocking;
        const size_t sp_slice_size = num_ic_chunks * num_oc_chunks * chunk_size;
        const size_t n_sp_slices = jbgp.ks();
        const size_t reduction_buf_size = n_sp_slices * sp_slice_size;
        const size_t reduction_buf_shift = reduction_buf_size * buf_idx;
        const size_t sp_slice_idx = kd * jbgp.kh * jbgp.kw + kh * jbgp.kw + kw;
        return ti->buffer_c + reduction_buf_shift + sp_slice_idx * sp_slice_size
                + (occ * num_ic_chunks + icc) * chunk_size
                + (ocb_l * jbgp.nb_ic_blocking + icb_l) * block_size;
    } else if (jbgp.nthr_mb == 1) {
        MAYBE_UNUSED(reduction_buf_idx);
        const size_t blk_size = acc_dt_size * jbgp.ic_block * jbgp.oc_block;
        const size_t buf_size_per_thread
                = blk_size * jbgp.nb_ic_blocking * jbgp.nb_oc_blocking;
        const size_t offset_within_thread_buf
                = blk_size * (jbgp.nb_ic_blocking * ocb_l + icb_l);
        const size_t offset
                = ti->ithr * buf_size_per_thread + offset_within_thread_buf;
        return ti->buffer_c + offset;
    }

    assert(!"unsupported case");
    return nullptr;
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::compute_diff_weights_and_bias(
        const thread_info_t *ti) const {
    auto diff_dst = const_cast<char *>(ti->diff_dst);
    auto diff_bias = ti->diff_bias;

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto &jbgp = pd()->jbgp_;

    const size_t bia_dt_size
            = jbgp.with_bias ? types::data_type_size(jbgp.bia_dt) : 0;
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);

    const int oc_chunk_sz = jbgp.oc_block * jbgp.nb_oc_blocking;

    brgemm_batch_element_t *addr_batch_global
            = ti->scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);

    const bool is_bf16 = jbgp.wei_dt == bf16 || jbgp.is_bf32;
    const bool is_amx_bf16 = is_bf16 && isa == avx512_core_amx;
    char *wsp_tile_global = (is_amx_bf16) ? ti->wsp_tile_base : nullptr;
    int os_chunks
            = static_cast<int>(utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking));

    const auto get_bia_acc_ptr = [&](int oc) {
        const int reduction_buf_start_idx = jbgp.bia_dt == f32;
        if (jbgp.bia_dt != data_type::f32
                || (jbgp.nthr_mb > 1
                        && ti->ithr_os_c >= reduction_buf_start_idx)) {
            return ti->buffer_bias
                    + acc_dt_size * (ti->ithr_os_c - reduction_buf_start_idx)
                    * jbgp.oc
                    + acc_dt_size * oc;
        } else {
            return ti->diff_bias + bia_dt_size * oc;
        }
    };

    const auto a_buf_osb_shift = ti->get_buffer_a_osb_shift();
    const auto b_buf_osb_shift = ti->get_buffer_b_osb_shift();

    const auto ker
            = [&](const int sp_icc, const int osc, const int icc, const int occ,
                      const int icb_i, const int ocb_i, const int osc_prev,
                      const int sp_icc_prev, const int occ_prev, int kd, int kh,
                      int kw, int &prev_ker_idx) {
        brgemm_batch_element_t *addr_batch
                = addr_batch_global + ti->ithr * jbgp.adjusted_batch_size;
        char *wsp_tile = is_amx_bf16
                ? wsp_tile_global + ti->ithr * jbgp.amx_buf_size_per_thread
                : nullptr;
        int icb = icb_i + icc * jbgp.nb_ic_blocking;
        int ocb = ocb_i + occ * jbgp.nb_oc_blocking;
        int ic = icb * jbgp.ic_block;
        int oc = ocb * jbgp.oc_block;
        int osb = osc * jbgp.nb_os_blocking;
        int n = osb * jbgp.os_block;

        // Base buffer pointers for the current kernel iteration,
        // x_buf_osb_shift values are used for shifting wrt osb iter variable
        int sp_icb = sp_icc * jbgp.nb_ic_blocking + icb_i;
        char *a_buffer = ti->get_buffer_a_ptr(sp_icb, osc);
        char *b_buffer = ti->get_buffer_b_ptr(ocb, osc);

        bool kernel_init = (osc == ti->os_c_start);

        auto nb_os_b
                = nstl::min((jbgp.mb - n) / jbgp.os_block, jbgp.nb_os_blocking);

        bool is_bs_tail = nb_os_b != jbgp.nb_os_blocking;
        bool is_os_tail = is_bs_tail && (jbgp.mb - n) % jbgp.os_block != 0;
        bool is_ic_tail = jbgp.ic - ic < jbgp.ic_block;
        bool is_oc_tail = jbgp.oc - oc < jbgp.oc_block;
        const int oc_chunk_tail = jbgp.oc % oc_chunk_sz;
        const bool is_last_oc_chunk = jbgp.oc - oc < oc_chunk_sz;
        const int curr_oc_chunk_sz = oc_chunk_tail > 0 && is_last_oc_chunk
                ? oc_chunk_tail
                : oc_chunk_sz;

        const bool transform_weights = jbgp.wei_dt != jbgp.acc_dt
                && (jbgp.nthr_mb == 1 || os_chunks == 1)
                && osc == (os_chunks - 1);
        const bool transform_b = jbgp.local_buffers_for_input_tensors
                ? jbgp.use_buffer_b && icb % jbgp.nb_ic_blocking == 0
                        && ocb % jbgp.nb_oc_blocking == 0
                        && IMPLICATION(osc_prev == osc,
                                occ_prev != ocb / jbgp.nb_oc_blocking)
                : jbgp.use_buffer_b && icb % jbgp.nb_ic_blocking == 0
                        && sp_icc == ti->sp_ic_c_start
                        && ocb % jbgp.nb_oc_blocking == 0;
        const bool transform_a = jbgp.local_buffers_for_input_tensors
                ? jbgp.use_buffer_a && ocb % jbgp.nb_oc_blocking == 0
                        && IMPLICATION(osc_prev == osc, sp_icc_prev != sp_icc)
                : jbgp.use_buffer_a
                        && ocb == ti->oc_c_start * jbgp.nb_oc_blocking;

        const int brg_ker_idx
                = brgemm_inner_product_utils::get_brg_kernel_index(
                        is_bs_tail, kernel_init, is_ic_tail, is_oc_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();

        if (kernel_init && (is_ic_tail || is_oc_tail)) {
            // Due to big_ic_blk_ok optimization, the ic_block can be larger
            // than simd_w. last_ic_chunk makes sure we do not overflow.
            bool is_last_spatial_slice = true;
            is_last_spatial_slice &= kd == jbgp.kd - 1;
            is_last_spatial_slice &= kh == jbgp.kh - 1;
            is_last_spatial_slice &= kw == jbgp.kw - 1;

            int last_ic_chunk = jbgp.ic_block;
            if (is_last_spatial_slice)
                last_ic_chunk = nstl::min(
                        jbgp.ic_block, rnd_up(jbgp.ic - ic, jbgp.simd_w));

            utils::array_set(get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw), 0,
                    types::data_type_size(jbgp.acc_dt) * jbgp.oc_block
                            * last_ic_chunk);
        }

        bool is_1st_sp_slice = kd == 0 && kh == 0 && kw == 0;
        bool do_bias = jbgp.with_bias && icb == 0 && is_1st_sp_slice;

        if (nb_os_b > 0 && brg_kernel != nullptr) {
            brgemm_palettes_.maybe_tile_configure(
                    jbgp.is_amx, prev_ker_idx, brg_ker_idx);
            if (transform_a) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src + blk_off(src_d, n, ic, kd, kh, kw);

                transform_matrix_a_chunk(a_buffer, src_ptr, nb_os_b,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.os_block);
            }

            if (transform_b) {
                auto diff_dst_ptr = diff_dst
                        + types::data_type_size(jbgp.dst_dt)
                                * diff_dst_d.blk_off(n, oc);
                transform_matrix_b_chunk(b_buffer, diff_dst_ptr, nb_os_b,
                        curr_oc_chunk_sz, jbgp.os_block);
            }

            for (int os_block = 0; os_block < nb_os_b; os_block++) {
                auto a_ptr = a_buffer + os_block * a_buf_osb_shift;
                addr_batch[os_block].ptr.A = a_ptr;
                auto diff_dst_ptr = diff_dst
                        + types::data_type_size(jbgp.dst_dt)
                                * diff_dst_d.blk_off(
                                        n + os_block * jbgp.os_block, oc);
                if (jbgp.use_buffer_b) {
                    auto b_ptr = b_buffer + os_block * b_buf_osb_shift;
                    addr_batch[os_block].ptr.B = b_ptr;
                } else {
                    addr_batch[os_block].ptr.B = diff_dst_ptr;
                }
                if (do_bias) {
                    brgemm_kernel_diff_bias_t p;
                    auto bias_ptr = diff_bias + bia_dt_size * oc;
                    p.ptr_diff_dst = (void *)addr_batch[os_block].ptr.B;
                    p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                    p.ptr_diff_bias = (void *)bias_ptr;
                    bool is_first = kernel_init && os_block == 0;
                    bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                            && osc == os_chunks - 1 && os_block == nb_os_b - 1
                            && !is_os_tail;
                    p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                            | (is_last ? FLAG_REDUCE_LAST : 0);

                    (*kernels_db_[false][is_oc_tail])(&p);
                }
            }
            brgemm_kernel_execute(brg_kernel, nb_os_b, addr_batch,
                    (void *)get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw),
                    wsp_tile);
        }

        if (is_os_tail) {
            auto use_init_ker = (kernel_init && nb_os_b == 0);
            const int brg_ker_idx_os_tail
                    = brgemm_inner_product_utils::get_brg_kernel_index(
                            false, use_init_ker, is_ic_tail, is_oc_tail, true);
            auto brg_kernel_os_tail = brg_kernels_[brg_ker_idx_os_tail].get();
            if (brg_kernel_os_tail != nullptr)
                brgemm_palettes_.maybe_tile_configure(
                        jbgp.is_amx, prev_ker_idx, brg_ker_idx_os_tail);
            int os_block = nb_os_b;
            auto a_ptr = a_buffer + os_block * a_buf_osb_shift;

            if (transform_a) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src
                        + blk_off(src_d, n + os_block * jbgp.os_block, ic, kd,
                                kh, kw);
                transform_matrix_a_chunk(a_ptr, src_ptr, 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.mb % jbgp.os_block);
            }

            addr_batch[0].ptr.A = a_ptr;
            auto diff_dst_ptr = diff_dst
                    + types::data_type_size(jbgp.dst_dt)
                            * diff_dst_d.blk_off(
                                    n + os_block * jbgp.os_block, oc);
            if (jbgp.use_buffer_b) {
                auto b_ptr = b_buffer + os_block * b_buf_osb_shift;

                if (transform_b)
                    transform_matrix_b_chunk(b_ptr, diff_dst_ptr, 1,
                            curr_oc_chunk_sz, jbgp.mb % jbgp.os_block);
                addr_batch[0].ptr.B = b_ptr;
            } else {
                addr_batch[0].ptr.B = diff_dst_ptr;
            }

            if (do_bias) {
                brgemm_kernel_diff_bias_t p;
                auto bias_ptr = diff_bias + bia_dt_size * oc;
                p.ptr_diff_dst = (void *)addr_batch[0].ptr.B;
                p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                p.ptr_diff_bias = (void *)bias_ptr;
                bool is_first = kernel_init && os_block == 0;
                bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                        && osc == os_chunks - 1;
                p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                        | (is_last ? FLAG_REDUCE_LAST : 0);

                (*kernels_db_[true][is_oc_tail])(&p);
            }

            if (brg_kernel_os_tail != nullptr) {
                brgemm_kernel_execute(brg_kernel_os_tail, 1, addr_batch,
                        (void *)get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw),
                        wsp_tile);
            }
        }

        if (transform_weights) {
            transpose_matrix_c_chunk(ti, ocb, icb,
                    is_oc_tail ? jbgp.oc % jbgp.oc_block : jbgp.oc_block,
                    is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block, kd,
                    kh, kw);
        }
    };

    const auto occ_work = (ti->oc_c_end - ti->oc_c_start);
    const auto sp_icc_work = (ti->sp_ic_c_end - ti->sp_ic_c_start);
    const auto osc_work = (ti->os_c_end - ti->os_c_start);
    const auto loop_end = occ_work * sp_icc_work * osc_work;
    int occ_idx = 0, sp_icc_idx = 0, osc_idx = 0;

    auto loop_idx = 0;

    using loop_order_t = jit_brgemm_ip_bwd_w_conf_t::loop_order_t;
    switch (jbgp.loop_order) {
        case loop_order_t::osc_icc_occ:
            nd_iterator_init(loop_idx, osc_idx, osc_work, sp_icc_idx,
                    sp_icc_work, occ_idx, occ_work);
            break;
        case loop_order_t::osc_occ_icc:
            nd_iterator_init(loop_idx, osc_idx, osc_work, occ_idx, occ_work,
                    sp_icc_idx, sp_icc_work);
            break;
        case loop_order_t::occ_icc_osc:
            nd_iterator_init(loop_idx, occ_idx, occ_work, sp_icc_idx,
                    sp_icc_work, osc_idx, osc_work);
    };

    int prev_ker_idx = -1;
    int osc_prev = -1, sp_icc_prev = -1, occ_prev = -1;
    while (loop_idx < loop_end) {
        const int occ = ti->oc_c_start + occ_idx;
        const int sp_icc = ti->sp_ic_c_start + sp_icc_idx;
        const int osc = ti->os_c_start + osc_idx;
        const int ic_chunks = ti->ic_chunks();

        // Unsquash spatial and input channels chunks index.
        int kd = 0, kh = 0, kw = 0, icc = 0;
        nd_iterator_init(
                sp_icc, kd, jbgp.kd, kh, jbgp.kh, kw, jbgp.kw, icc, ic_chunks);

        const dim_t ocb_work = nstl::min<dim_t>(
                jbgp.nb_oc_blocking, jbgp.nb_oc - occ * jbgp.nb_oc_blocking);
        const dim_t icb_work = nstl::min<dim_t>(
                jbgp.nb_ic_blocking, jbgp.nb_ic - icc * jbgp.nb_ic_blocking);

        for_(int ocb_i = 0; ocb_i < ocb_work; ocb_i++)
        for (int icb_i = 0; icb_i < icb_work; icb_i++) {
            ker(sp_icc, osc, icc, occ, icb_i, ocb_i, osc_prev, sp_icc_prev,
                    occ_prev, kd, kh, kw, prev_ker_idx);
        }
        osc_prev = osc;
        sp_icc_prev = icc;
        occ_prev = occ;

        ++loop_idx;

        switch (jbgp.loop_order) {
            case loop_order_t::osc_icc_occ:
                nd_iterator_step(osc_idx, osc_work, sp_icc_idx, sp_icc_work,
                        occ_idx, occ_work);
                break;
            case loop_order_t::osc_occ_icc:
                nd_iterator_step(osc_idx, osc_work, occ_idx, occ_work,
                        sp_icc_idx, sp_icc_work);
                break;
            case loop_order_t::occ_icc_osc:
                nd_iterator_step(occ_idx, occ_work, sp_icc_idx, sp_icc_work,
                        osc_idx, osc_work);
        };
    }
    if (jbgp.is_amx) amx_tile_release();
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<
        isa>::reduce_and_convert_diff_weights_and_bias(const thread_info_t *ti)
        const {
    const auto &jbgp = pd()->jbgp_;

    if (dnnl_thr_syncable() && jbgp.nthr > 1)
        simple_barrier::barrier(ti->barrier_ctx, jbgp.nthr);
    if (ti->nthr_os_c == 1) return;

    const bool is_f32_out = jbgp.wei_dt == data_type::f32;
    const int icb_scale = is_f32_out ? jbgp.ic_block / jbgp.simd_w : 1;

    const dim_t icb_work
            = nstl::min<dim_t>(ti->sp_ic_c_work * jbgp.nb_ic_blocking,
                    jbgp.nb_ic - ti->sp_ic_c_start * jbgp.nb_ic_blocking);
    const dim_t ocb_work = nstl::min<dim_t>(ti->oc_c_work * jbgp.nb_oc_blocking,
            jbgp.nb_oc - ti->oc_c_start * jbgp.nb_oc_blocking);
    const dim_t work = ocb_work * icb_work * jbgp.ks();

    int os_chunks
            = static_cast<int>(utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking));
    int reduce_buffers = nstl::min(ti->nthr_os_c, os_chunks);
    int reduce_buf_idx_start = !is_f32_out;
    int reduce_buf_idx_end = reduce_buffers - is_f32_out;

    dim_t start = 0, end = 0;
    balance211(work, ti->nthr_os_c, ti->ithr_os_c, start, end);
    if (start == end) return;

    dim_t icb_l = 0, ocb_l = 0, kd = 0, kh = 0, kw = 0;
    const int acc_size = jbgp.ic_block * jbgp.oc_block;

    for (int ir = reduce_buf_idx_start; ir < reduce_buf_idx_end; ++ir) {
        dim_t counter = start;
        nd_iterator_init(start, kd, jbgp.kd, kh, jbgp.kh, kw, jbgp.kw, ocb_l,
                ocb_work, icb_l, icb_work);
        while (counter < end) {
            const dim_t ocb = ti->oc_c_start * jbgp.nb_oc_blocking + ocb_l;
            const dim_t icb = ti->sp_ic_c_start * jbgp.nb_ic_blocking + icb_l;
            char *wei_to_reduce = get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw, ir);
            const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
            char *wei_reduced = !is_f32_out
                    ? get_wei_acc_ptr(ti, ocb, icb, kd, kh, kw, 0)
                    : ti->diff_weights
                            + blk_off(diff_weights_d, ocb, icb * icb_scale, kd,
                                    kh, kw);
            acc_ker_->accumulate(
                    (float *)(wei_reduced), (float *)(wei_to_reduce), acc_size);
            if (!is_f32_out && ir + 1 == reduce_buf_idx_end) {
                transpose_matrix_c_chunk(ti, ocb, icb * icb_scale,
                        jbgp.oc_block, jbgp.ic_block, kd, kh, kw, true);
            }
            ++counter;
            nd_iterator_step(kd, jbgp.kd, kh, jbgp.kh, kw, jbgp.kw, ocb_l,
                    ocb_work, icb_l, icb_work);
        }
    }

    if (jbgp.with_bias && ti->ithr_sp_ic_c == 0 && ti->sp_ic_c_work > 0
            && ti->ithr_os_c == 0 && ti->os_c_work > 0 && ti->oc_c_work > 0) {
        const bool is_f32_bias = jbgp.bia_dt == data_type::f32;
        float *bias_reduced = is_f32_bias ? (float *)ti->diff_bias
                                          : (float *)ti->buffer_bias;
        int reduce_buf_idx_start = !is_f32_bias;
        int reduce_buf_idx_end = reduce_buffers - 1;
        int oc_chunk_size = jbgp.nb_oc_blocking * jbgp.oc_block;
        int oc = ti->oc_c_start * oc_chunk_size;
        int acc_size = nstl::min(ti->oc_c_work * oc_chunk_size, jbgp.oc - oc);

        int ir = reduce_buf_idx_start;
        for (; ir < reduce_buf_idx_end; ++ir) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            acc_ker_->accumulate(
                    &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
        }

        if (!is_f32_bias) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            switch (jbgp.bia_dt) {
                case data_type::bf16:
                    add_floats_and_cvt_to_bfloat16(
                            (bfloat16_t *)(ti->diff_bias) + oc,
                            &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
                    break;
                case data_type::f16:
                    add_floats_and_cvt_to_float16(
                            (float16_t *)(ti->diff_bias) + oc,
                            &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
                    break;
                default: assert(!"invalid data type");
            }
        }
    }
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    const auto &jbgp = pd()->jbgp_;

    if (dnnl_thr_syncable() && jbgp.nthr > 1) {
        const auto &scratchpad = ctx.get_scratchpad_grantor();
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    parallel(
            jbgp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        compute_diff_weights_and_bias(&thread_info);

        if (dnnl_thr_syncable()) {
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        }
    });

    if (!dnnl_thr_syncable()) {
        parallel(jbgp.nthr,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }
}

template struct brgemm_inner_product_bwd_weights_t<avx512_core_amx_fp16>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core_fp16>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core_amx>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core_bf16>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core>;
template struct brgemm_inner_product_bwd_weights_t<avx2>;
template struct brgemm_inner_product_bwd_weights_t<avx10_2>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
