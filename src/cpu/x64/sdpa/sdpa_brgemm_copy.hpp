/*
*******************************************************************************
* Copyright 2026 Intel Corporation
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
*******************************************************************************
*/

#ifndef CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP
#define CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP

#include <limits>
#include <memory>

#include "common/utils.hpp"
#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace sdpa_brgemm_copy {

// For VNNI granularity v (1 for f32, 2 for xf16), the destination element
// offset is (k / v) * dst_ld * v + n * v + k % v in global-matrix mode.
// Separate-panel mode replaces n with n % 64 and adds
// (n / 64) * packed_k(K, dt) * 64. Each panel has 64 N columns and requires
// dst_ld == 64; the caller must allocate packed_k(K, dt) rows per panel.
// execute positions panels explicitly; the copy kernels use LDB2 == 0.
class b_packer_t {
public:
    static dim_t packed_ld(dim_t N) { return utils::rnd_up(N, dim_t(64)); }
    static dim_t k_block(data_type_t dt) { return matmul::get_wei_k_blk(dt); }
    static dim_t packed_k(dim_t K, data_type_t dt) {
        return utils::rnd_up(K, k_block(dt));
    }

    status_t init(dim_t K, dim_t N, dim_t src_k_stride, dim_t src_n_stride,
            dim_t dst_ld, data_type_t dt, bool separate_panels = false) {
        if (K <= 0 || N <= 0 || src_k_stride <= 0 || src_n_stride <= 0
                || (separate_panels ? dst_ld != n_block_
                                    : dst_ld < packed_ld(N)))
            return status::invalid_arguments;
        if (src_k_stride != 1 && src_n_stride != 1)
            return status::unimplemented;

        K_ = K;
        N_ = N;
        dst_ld_ = dst_ld;
        separate_panels_ = separate_panels;
        dt_ = dt;
        if (src_n_stride == 1) {
            src_tag_ = format_tag::ab;
            src_ld_ = src_k_stride;
        } else {
            src_tag_ = format_tag::ba;
            src_ld_ = src_n_stride;
        }
        k_vnni_ = data_type_vnni_granularity(dt_);
        if (k_vnni_ <= 0) return status::invalid_arguments;

        CHECK(matmul::init_conf(
                conf_, 1, 0, K_, N_, src_ld_, n_block_, dt_, dt_, src_tag_));
        k_pad_block_ = k_block(dt_);
        if (!separate_panels_) {
            const dim_t max_dst_elements
                    = std::numeric_limits<int32_t>::max() / conf_.tr_b_dt_sz;
            if (dst_ld_ > (max_dst_elements - conf_.wei_n_blk) / conf_.K_blk)
                return status::unimplemented;
            conf_.LDB = dst_ld_;
        }
        return matmul::create_brgemm_matmul_copy_b(kernel_, &conf_);
    }

    void execute(const void *src, void *dst) const {
        const auto src_size = types::data_type_size(dt_);
        const auto dst_size = types::data_type_size(conf_.wei_dt);
        const auto *src_bytes = static_cast<const char *>(src);
        auto *dst_bytes = static_cast<char *>(dst);

        for (dim_t n0 = 0; n0 < N_; n0 += n_block_) {
            const dim_t n = N_ - n0 < n_block_ ? N_ - n0 : n_block_;
            for (dim_t k0 = 0; k0 < K_; k0 += conf_.K_blk) {
                const dim_t k_iters = nstl::min(conf_.K_blk, K_ - k0);
                const dim_t k_remainder = k_iters % k_pad_block_;
                const dim_t k_pad = k_remainder == 0
                        ? 0
                        : k_pad_block_ - utils::rnd_up(k_remainder, k_vnni_);
                const dim_t src_offset = src_tag_ == format_tag::ab
                        ? k0 * src_ld_ + n0
                        : n0 * src_ld_ + k0;
                const dim_t dst_offset
                        = (k0 / k_vnni_ * dst_ld_ * k_vnni_
                                  + (separate_panels_ ? n0 / n_block_
                                                          * packed_k(K_, dt_)
                                                          * dst_ld_
                                                      : n0 * k_vnni_))
                        * dst_size;

                auto ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
                ctx.src = src_bytes + src_offset * src_size;
                ctx.tr_src = dst_bytes + dst_offset;
                ctx.current_K_start = k0;
                ctx.current_K_iters = k_iters;
                ctx.current_K_pad = k_pad;
                ctx.current_N_blk = n;
                (*kernel_)(&ctx);
            }
        }
    }

private:
    static constexpr dim_t n_block_ = 64;

    matmul::brgemm_matmul_conf_t conf_;
    std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t> kernel_;
    dim_t K_ = 0;
    dim_t N_ = 0;
    dim_t src_ld_ = 0;
    dim_t dst_ld_ = 0;
    dim_t k_pad_block_ = 0;
    dim_t k_vnni_ = 0;
    data_type_t dt_ = data_type::undef;
    format_tag_t src_tag_ = format_tag::undef;
    bool separate_panels_ = false;
};

} // namespace sdpa_brgemm_copy
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_SDPA_SDPA_BRGEMM_COPY_HPP
