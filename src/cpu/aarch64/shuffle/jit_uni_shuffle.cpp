/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2022-2024 FUJITSU LIMITED
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
#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/shuffle/jit_uni_shuffle.hpp"
#include "cpu/aarch64/shuffle/jit_uni_shuffle_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;
    using namespace types;

    const memory_desc_wrapper src_d(is_fwd() ? src_md() : diff_src_md());
    const memory_desc_wrapper dst_d(is_fwd() ? dst_md() : diff_dst_md());

    if (!impl::is_dense_format_kind({is_fwd() ? src_md() : diff_src_md(),
                is_fwd() ? dst_md() : diff_dst_md()}))
        return status::unimplemented;

    conf_.data_type = src_d.data_type();

    const bool ok = mayiuse(isa)
            && utils::one_of(conf_.data_type, f32, s32, bf16)
            && src_d.data_type() == dst_d.data_type()
            && platform::has_data_type_support(conf_.data_type)
            && attr()->has_default_values() && axis() == 1
            && set_default_formats_common() && src_d == dst_d;

    if (!ok) return status::unimplemented;

    conf_.isa = isa;

    const format_tag_t blocked_format
            = memory_desc_matches_one_of_tag(*src_md(), nCw16c, nChw16c,
                    nCdhw16c, nCw8c, nChw8c, nCdhw8c, nCw4c, nChw4c, nCdhw4c);

    if (blocked_format == format_tag::undef) return status::unimplemented;

    conf_.blk_size = src_d.blocking_desc().strides[ndims() - 1];
    /* Because "ST1H { <Zt>.S }, <Pg>, [<Xn|SP>, <Zm>.S, UXTW #1]" is used
       to gather data for bf16, simd_w must be calculated
       with sizeof(uint32_t). */
    conf_.simd_w = simd_bytes(isa) / sizeof(uint32_t);

    const bool has_spatial = utils::one_of(ndims(), 3, 4, 5);
    const dim_t HW = H() * W();
    conf_.sp = has_spatial ? D() * HW : HW;

    if (conf_.simd_w <= conf_.blk_size) {
        conf_.tag_kind = jit_memory_tag_kind_t::blocked;
        conf_.simd_tail = C() % conf_.simd_w;
        conf_.c_split_size = conf_.blk_size;
        if (C() < std::sqrt(conf_.sp))
            conf_.sp_split_size = conf_.sp
                    / math::gcd(conf_.sp,
                            static_cast<dim_t>(dnnl_get_max_threads()));
        else
            conf_.sp_split_size = conf_.sp;
    } else
        return status::unimplemented;

    conf_.ndims = ndims();
    conf_.mb = MB();
    conf_.c = C();
    conf_.d = D();
    conf_.h = H();
    conf_.w = W();

    conf_.dt_size = types::data_type_size(conf_.data_type);
    conf_.stride_mb = src_d.blocking_desc().strides[0];
    conf_.group_size = group_size();
    conf_.axis = axis();
    conf_.axis_size = axis_size();
    conf_.el_size_of_indices = sizeof(unsigned);

    auto sp = scratchpad_registry().registrar();
    sp.template book<int>(
            memory_tracking::names::key_shuffle_precompute_transpose,
            utils::rnd_up(conf_.c, conf_.blk_size));

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::precompute_offsets(
        const exec_ctx_t &ctx) const {
    const auto conf = pd()->get_conf();
    const int axis_size = conf.axis_size;
    const int group_size = conf.group_size;
    const int transpose_row
            = pd()->is_fwd() ? group_size : axis_size / group_size;
    const int transpose_col
            = pd()->is_fwd() ? axis_size / group_size : group_size;

    const auto &sp = ctx.get_scratchpad_grantor();
    auto sp_ptr = sp.template get<int>(
            memory_tracking::names::key_shuffle_precompute_transpose);

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        const dim_t blk_size = conf.blk_size;
        const dim_t SP = conf.sp;

        const dim_t C = conf.c;
        parallel(1, [=](int, int) {
            PRAGMA_OMP_SIMD()
            for (int cc = 0; cc < C; cc++) {
                const int cc_row_idx = cc / transpose_col;
                const int cc_col_idx = cc % transpose_col;
                const int input_c = cc_col_idx * transpose_row + cc_row_idx;
                sp_ptr[cc] = (input_c / blk_size * SP * blk_size
                                     + input_c % blk_size)
                        * conf.dt_size;
            }
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_shuffle_kernel_t<isa>(pd()->get_conf())));
    CHECK(kernel_->create_kernel());
    return status::success;
}

template <cpu_isa_t isa>
inline jit_uni_shuffle_t<isa>::jit_uni_shuffle_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_shuffle_t<isa>::~jit_uni_shuffle_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_shuffle_t<isa>::execute(const exec_ctx_t &ctx) const {
    using namespace prop_kind;
    using namespace utils;

    CHECK(precompute_offsets(ctx));

    const auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;
    auto input = CTX_IN_MEM(const uint8_t *, i_arg);
    auto output = CTX_OUT_MEM(uint8_t *, o_arg);

    const auto conf = pd()->get_conf();

    const dim_t MB = conf.mb;
    const dim_t SP = conf.sp;
    const dim_t C = conf.c;
    const dim_t stride_mb = conf.stride_mb;
    const int data_type_size = conf.dt_size;

    const auto &sp = ctx.get_scratchpad_grantor();
    auto sp_ptr = sp.template get<int>(
            memory_tracking::names::key_shuffle_precompute_transpose);

    if (pd()->get_conf().tag_kind == jit_memory_tag_kind_t::blocked) {
        const dim_t CB = utils::div_up(C, conf.c_split_size);
        const dim_t SPB = SP / conf.sp_split_size;
        parallel_nd(MB, SPB, CB,
                [= COMPAT_THIS_CAPTURE](dim_t mb, dim_t spb, dim_t cb) {
            const dim_t c_work
                    = nstl::min(conf.c_split_size, C - cb * conf.c_split_size);
            const dim_t c_curr = cb * conf.c_split_size;
            const dim_t sp_work = conf.sp_split_size;
            const dim_t sp_curr = spb * sp_work;
            const dim_t off = mb * stride_mb + sp_curr * conf.blk_size;

            jit_uni_shuffle_args_t args;
            args.src = input + off * data_type_size;
            args.dst = output + (off + SP * c_curr) * data_type_size;

            args.cb_loop_size = c_work;
            args.is_padded_block = cb + 1 == CB;

            args.input_off_ptr = sp_ptr + c_curr;
            (*kernel_)(&args);
        });
    } else {
        assert(!"Invalid memory format kind.");
        return status::invalid_arguments;
    }

    return status::success;
}

template struct jit_uni_shuffle_t<sve>;
template struct jit_uni_shuffle_t<asimd>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
