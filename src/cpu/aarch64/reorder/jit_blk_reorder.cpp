/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2026 Arm Ltd. and affiliates
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

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose_msg.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/reorder/jit_blk_reorder.hpp"

// #define DNNL_DEV_MODE
#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t jit_blk_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto prb = tr::prb_t();
    // For shapes with dimension greater than thres it is found that jit:uni is better that jit:blk
    auto upper_thres = 1920 * 4096;
    auto src_d = memory_desc_wrapper(src_md());
    auto prd = 1;

    for (int d = 0; d < src_d.ndims(); ++d) {
        const auto dim = src_d.dims()[d];
        prd *= dim;
        VDISPATCH_REORDER(prd <= upper_thres, "perf condition is not met");
    }

    auto lower_thres = 128 * 128;
    VDISPATCH_REORDER(
            IMPLICATION(get_max_cpu_isa() == sve_128, prd >= lower_thres),
            VERBOSE_IMPL_HEURISTIC_FAIL,
            "very small shapes are faster on jit uni for SVE-128");

    status_t prb_init_status = prb_init(prb, *src_md(), *dst_md(), attr());
    if (prb_init_status != status::success) return prb_init_status;
    // only uni_reorder supports tail processing now
    // TODO: Add tail processing support in blk_reorder
    VDISPATCH_REORDER(!prb.is_tail_present, "tail processing is not supported");

    prb_tile_normalize(prb);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "tile : %s\n", prb_dump(prb).c_str());
    });

    VDISPATCH_REORDER(tr::jit_single_blk_kernel_t::applicable(prb),
            "applicable() call failed");

    prb_ = prb;
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    return status::success;
}

void jit_blk_reorder_t::pd_t::prb_tile_normalize(tr::prb_t &p) {
    if (!utils::one_of(p.nodes[0].n, 4ul, 8ul, 16ul, 32ul, 64ul)
            && utils::one_of(p.nodes[1].n, 4ul, 8ul, 16ul, 32ul, 64ul)) {
        nstl::swap(p.nodes[0], p.nodes[1]);
    }
}

jit_blk_reorder_t::jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}
jit_blk_reorder_t::~jit_blk_reorder_t() = default;

status_t jit_blk_reorder_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
    return kernel_->create_kernel();
}

status_t jit_blk_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    // kernel handle 2-dimension tiles, a tail is possible
    auto &prb = this->pd()->prb_;
    ptrdiff_t BH = 1;
    for (int i = 2; i < prb.ndims; ++i) {
        BH *= prb.nodes[i].n;
    }

    auto block_sz = prb.n(0);
    auto n1 = prb.n(1);
    auto i1 = prb.is(1);
    auto o1 = prb.os(1);
    auto FL = (n1 + block_sz - 1) / block_sz;
    auto bh_stride = BH == 1 ? 0 : prb.is(2);

    auto itype_sz_ = data_type_size(pd()->prb_.itype);
    auto otype_sz_ = data_type_size(pd()->prb_.otype);

    parallel_nd(BH, FL, [= COMPAT_THIS_CAPTURE](dim_t bh, dim_t fl) {
        auto fl_b = fl * block_sz;
        auto bh_b = bh_stride * bh;
        auto *i = in + (bh_b + fl_b * i1) * itype_sz_;
        auto *o = out + (bh_b + fl_b * o1) * otype_sz_;
        (*kernel_)(i, o, n1 - fl_b < block_sz);
    });

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
