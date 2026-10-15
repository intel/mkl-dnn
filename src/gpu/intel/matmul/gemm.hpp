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

#ifndef GPU_INTEL_MATMUL_GEMM_HPP
#define GPU_INTEL_MATMUL_GEMM_HPP

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/memory_desc.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr_quant.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/intel/gemm/utils.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct gemm_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T(gemm_pd_ ? gemm_pd_->name() : "gemm_t", gemm_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;

            primitive_attr_t gemm_attr;
            gemm_attr.fpmath_ = attr()->fpmath_;
            gemm_attr.acc_mode_ = attr()->acc_mode_;
            gemm_attr.deterministic_ = attr()->deterministic_;
            gemm_attr.rounding_mode_ = attr()->rounding_mode_;
            gemm_attr.scales_ = attr()->scales_;
            gemm_attr.precomputed_reductions_ = attr()->precomputed_reductions_;
            gemm_attr.zero_points_ = attr()->zero_points_;
            gemm_attr.post_ops_ = attr()->post_ops_;
            gemm_attr.dropout_ = attr()->dropout_;
            if (attr()->gpu_attr_)
                gemm_attr.gpu_attr_ = attr()->gpu_attr_->clone();

            auto a_md = src_md(), b_md = weights_md(), c_md = dst_md(),
                 bias_md = weights_md(1);
            const auto acc_dt = desc()->accum_data_type;
            memory_desc_t a_md_reshaped, b_md_reshaped, c_md_reshaped,
                    bia_md_reshaped;
            bool with_bia = bias_md->ndims > 0;
            auto orig_dims = a_md->ndims;

            // Check for grouped encoding early before reshape attempts
            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

            auto maybe_reshape = [&]() -> status_t {
                // reduce output is not handled by the reshape logic.
                if (with_reduce()) return status::success;

                // Cannot reshape runtime dims
                for (int i = 0; i < b_md->ndims; i++)
                    if (b_md->dims[i] == runtime_value_for<dim_t>())
                        return status::success;

                bool reshape_2d = (b_md->ndims > 2);
                for (int i = 0; reshape_2d && i < b_md->ndims - 2; i++) {
                    reshape_2d &= (b_md->dims[i] == 1);
                }
                // for batch dim can map broadcast to 2d: eg. 4x1x4096:1x4096x16 -> 4x4096:4096x16
                bool reshape_3d = (a_md->ndims > 3);
                bool allow_reshape
                        = gpu_utils::dev_getenv("GEMM_ALLOW_RESHAPE", true);

                // Early exit if not reshaping
                if (!allow_reshape || !(reshape_2d || reshape_3d))
                    return status::success;

                int ndims = a_md->ndims;
                int reshape_size = reshape_2d ? 2 : 3;
                int diff_dims = orig_dims - reshape_size;

                // Converts input dims_t to output dims_t after reshaping
                auto squash_dims = [](dims_t &out_dims, const dims_t &in_dims,
                                           int in_ndims, int out_ndims) {
                    int diff_ndims = in_ndims - out_ndims;
                    gpu_assert(diff_ndims > 0) << "Unexpected squashing dims";
                    gpu_assert(&out_dims != &in_dims)
                            << "Cannot squash in-place";
                    // Shift dims over
                    for (int i = 0; i < out_ndims; i++) {
                        out_dims[i] = in_dims[i + diff_ndims];
                    }
                    // Squash dims into 1st dim
                    for (int i = 0; i < diff_ndims; i++) {
                        out_dims[0] *= in_dims[i];
                    }
                };

                // Convert raw tensors to reshaped tensors
                dims_t a_dims, b_dims, c_dims, bia_dims;
                squash_dims(a_dims, a_md->dims, ndims, reshape_size);
                squash_dims(b_dims, b_md->dims, ndims, reshape_size);
                squash_dims(c_dims, c_md->dims, ndims, reshape_size);
                squash_dims(bia_dims, bias_md->dims, ndims, reshape_size);

                // Cannot reshape if bias is broadcast across a subset of squashed dimensions
                bool bcast_ok = IMPLICATION(
                        with_bia, utils::one_of(bia_dims[0], 1, c_dims[0]));

                // 3D reshaping is only possible if A and B batch sizes allow.
                // This means no reshaping with partial broadcasting
                // or when both tensors are broadcast in different dimensions.
                bool a_broadcast = false;
                bool b_broadcast = false;
                for (int i = 0; i < ndims - 2; i++) {
                    if (a_md->dims[i] == 1 && b_md->dims[i] > 1)
                        a_broadcast = true;
                    if (b_md->dims[i] == 1 && a_md->dims[i] > 1)
                        b_broadcast = true;
                }
                bcast_ok = bcast_ok && !(a_broadcast && b_broadcast);
                bcast_ok = bcast_ok
                        && IMPLICATION(reshape_size == 3,
                                a_dims[0] == b_dims[0]
                                        || utils::one_of(
                                                1, a_dims[0], b_dims[0]));

                // memory_desc_reshape can fail. If so, fail gracefully so
                // we can use batched gemm instead of skipping entirely.
                auto safe_reshape
                        = [](memory_desc_t &out_md, const memory_desc_t &in_md,
                                  int ndims, const dims_t dims) -> bool {
                    CHECK_BOOL(memory_desc_reshape(out_md, in_md, ndims, dims));
                    return true;
                };
                bcast_ok = bcast_ok
                        && safe_reshape(
                                a_md_reshaped, *a_md, reshape_size, a_dims);
                bcast_ok = bcast_ok
                        && safe_reshape(
                                b_md_reshaped, *b_md, reshape_size, b_dims);
                bcast_ok = bcast_ok
                        && safe_reshape(
                                c_md_reshaped, *c_md, reshape_size, c_dims);
                if (with_bia) {
                    bcast_ok = bcast_ok
                            && safe_reshape(bia_md_reshaped, *bias_md,
                                    reshape_size, bia_dims);
                }
                auto reshaped_post_ops = gemm_attr.post_ops_;
                for (int i = 0; i < attr()->post_ops_.len(); i++) {
                    auto &po = reshaped_post_ops.entry_[i];
                    if (po.is_binary()) {
                        const auto &po_desc = po.binary.src1_desc;
                        auto a_dim = po_desc.dims[po_desc.ndims - reshape_size];
                        for (int i = po_desc.ndims; i > reshape_size; i--) {
                            a_dim *= po_desc.dims[po_desc.ndims - i];
                        }
                        //post ops cannot be applied if applied on only on a subset of batch dims
                        if (a_dim != c_dims[0] && a_dim > 1) {
                            return status::success;
                        }
                        auto has_dims = po_desc.ndims > 0;
                        dims_t po_dims;
                        if (reshape_2d) {
                            po_dims[0] = a_dim;
                            po_dims[1] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 1]
                                    : 1;
                        } else {
                            po_dims[0] = a_dim;
                            po_dims[1] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 2]
                                    : 1;
                            po_dims[2] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 1]
                                    : 1;
                        }
                        memory_desc_t tmp_po_desc;
                        bcast_ok = bcast_ok
                                && safe_reshape(tmp_po_desc, po_desc,
                                        reshape_size, po_dims);
                        reshaped_post_ops.entry_[i].binary.src1_desc
                                = tmp_po_desc;
                    } else if (po.is_prelu()) {
                        auto mask = po.prelu.mask;
                        int new_mask = 0;
                        int batch_idx = reshape_size - 1;
                        dim_t batch_dim = 1;
                        dim_t mask_dim = 1;
                        //get mask for batch dim
                        for (int i = 0; i < c_md->ndims - batch_idx; i++) {
                            if (mask >> i & 1) {
                                //post ops cannot be applied if applied on only on a subset of batch dims
                                if (new_mask != 0) return status::success;
                                new_mask |= c_md->dims[i] == 1 ? 0 : 1;
                                mask_dim *= c_md->dims[i];
                            }
                            batch_dim *= c_md->dims[i];
                        }
                        //post ops cannot be applied if applied on only on a subset of batch dims
                        if (batch_dim != mask_dim) return status::success;
                        //get non-batch part of mask
                        auto shift = c_md->ndims - batch_idx;
                        auto non_batch_mask = mask >> shift;
                        //due to prelu being in axb format, if a reshape is done it
                        //implies layout is different e.g 1x30x20 -> 30 is innermost dimension
                        //but 30x20 -> 20 is innermost. Hence reshape does  not work if mask
                        //is applied across more than one dimension.
                        if (non_batch_mask > 2
                                || (non_batch_mask > 0 && new_mask > 0))
                            return status::success;
                        new_mask |= non_batch_mask << 1;
                        reshaped_post_ops.entry_[i].prelu.mask = new_mask;
                    }
                }

                // Quantization has a few wrinkles...
                // Example: --attr-scales=src:per_ocic:f16:1x128 4x1x4096:1x4096x16
                // The src scales tensor has dimensions 1x32
                // We have two options since we can't change the scales tensor dimensions:
                // (1) Change mask from 6 -> 3 (both remaining dims masked) and change grouping
                //     -> src:per_ocic:4x128
                // (2) Change mask from 6 -> 2 (just K dim masked) and don't change grouping
                //     -> src:per_dim_1:1x128
                // Currently gemmstone only supports (1) so that's what we'll do here.
                // TODO: (2) has more optimization potential and is more reusable - implement
                // this option in gemmstone.

                // Same as squash_dims, but early-outs available if quantization not present
                auto squash_quant
                        = [&](dims_t &out_dims, const quant_entry_t &quant,
                                  const memory_desc_t &qmd) {
                    if (quant.has_default_values()) return;
                    squash_dims(out_dims, qmd.dims, ndims, reshape_size);
                    return;
                };

                auto squashed_mask = [&](int mask, int diff_dims) -> int {
                    return mask >> diff_dims;
                };

                auto reshape_quant = [&](const quant_entry_t &in_entry,
                                             const memory_desc_t &reshaped_md,
                                             const dims_t &qdims,
                                             int diff_dims) -> quant_entry_t {
                    if (in_entry.has_default_values()) return in_entry;
                    int new_mask
                            = squashed_mask(in_entry.get_mask(), diff_dims);
                    data_type_t dt = in_entry.get_data_type();
                    dims_t dims {};
                    int ndims = 0;
                    if (!in_entry.has_default_groups()) {
                        ndims = 2;
                        // Recalculate group sizes to obey (1) above
                        dims[0] = reshaped_md.dims[reshaped_md.ndims - 2]
                                / qdims[reshaped_md.ndims - 2];
                        dims[1] = reshaped_md.dims[reshaped_md.ndims - 1]
                                / qdims[reshaped_md.ndims - 1];
                    }
                    quant_entry_t out_entry;
                    UNUSED_STATUS(out_entry.set(new_mask, dt, ndims, dims,
                            false, in_entry.get_quantization_mode()));
                    return out_entry;
                };

                auto adjust_quant = [&](quant_entries_t &entries, int arg,
                                            const memory_desc_t &md,
                                            const memory_desc_t &reshaped_md,
                                            int diff_dims) -> status_t {
                    const quant_entry_t &entry = entries.get(arg);
                    if (entry.is_host_scalar()) return status::success;
                    memory_desc_t qmd;
                    CHECK(entry.get_md(qmd, md));
                    dims_t qdims;
                    squash_quant(qdims, entry, qmd);
                    quant_entry_t reshaped_entry = reshape_quant(
                            entry, reshaped_md, qdims, diff_dims);
                    CHECK(entries.set(arg, reshaped_entry));
                    return status::success;
                };

                // Quantization attributes are unsupported in Gemmstone with any
                // non-trivial (neither broadcast nor full dimension) modification
                // of groups during reshape.
                // TODO: Enable arbitrary reshaped grouped quant.
                auto safe_bcast_quant = [&](const quant_entry_t &entry,
                                                const dims_t orig_dims,
                                                dims_t reshape_dims) -> bool {
                    int full_tensor_mask = ((1 << c_md->ndims) - 1);
                    int per_oc_mask = (1 << 1);
                    if (!reshape_2d || entry.has_default_values()
                            || utils::one_of(entry.get_mask(), 0, per_oc_mask,
                                    full_tensor_mask))
                        return true;

                    if (utils::one_of(orig_dims[diff_dims], reshape_dims[0], 1))
                        return true;

                    return false;
                };

                bcast_ok = bcast_ok
                        && safe_bcast_quant(gemm_attr.scales_.get(DNNL_ARG_SRC),
                                a_md->dims, a_dims);
                bcast_ok = bcast_ok
                        && safe_bcast_quant(
                                gemm_attr.scales_.get(DNNL_ARG_WEIGHTS),
                                b_md->dims, b_dims);
                bcast_ok = bcast_ok
                        && safe_bcast_quant(
                                gemm_attr.zero_points_.get(DNNL_ARG_SRC),
                                a_md->dims, a_dims);
                bcast_ok = bcast_ok
                        && safe_bcast_quant(
                                gemm_attr.zero_points_.get(DNNL_ARG_WEIGHTS),
                                b_md->dims, b_dims);
                if (!bcast_ok) return status::success;

                scales_t reshaped_scales = gemm_attr.scales_;
                zero_points_t reshaped_zp = gemm_attr.zero_points_;
                precomputed_reductions_t reshaped_pr
                        = gemm_attr.precomputed_reductions_;
                CHECK(adjust_quant(reshaped_scales, DNNL_ARG_SRC, *a_md,
                        a_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_scales, DNNL_ARG_WEIGHTS, *b_md,
                        b_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_scales, DNNL_ARG_DST, *c_md,
                        c_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_zp, DNNL_ARG_SRC, *a_md,
                        a_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_zp, DNNL_ARG_WEIGHTS, *b_md,
                        b_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_zp, DNNL_ARG_DST, *c_md,
                        c_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_pr, DNNL_ARG_SRC, *a_md,
                        a_md_reshaped, diff_dims));
                CHECK(adjust_quant(reshaped_pr, DNNL_ARG_WEIGHTS, *b_md,
                        b_md_reshaped, diff_dims));

                // Reshaping successful - lock in changes
                a_md = &a_md_reshaped;
                b_md = &b_md_reshaped;
                c_md = &c_md_reshaped;
                if (with_bia) bias_md = &bia_md_reshaped;

                gemm_attr.scales_ = std::move(reshaped_scales);
                gemm_attr.zero_points_ = std::move(reshaped_zp);
                gemm_attr.precomputed_reductions_ = std::move(reshaped_pr);
                gemm_attr.post_ops_ = reshaped_post_ops;
                return status::success;
            };

            VDISPATCH_MATMUL_SC(maybe_reshape(), VERBOSE_IMPL_HEURISTIC_FAIL,
                    "2D/3D reshaping");

            if (with_reduce()) {
                VDISPATCH_MATMUL(reduce_kind() == matmul_reduce_kind::src,
                        VERBOSE_UNSUPPORTED_ATTR);
                sum_ab_ = sum_ab::sum_a_row;
                sum_ab_type_ = reduce_md()->data_type;
            }

            // We create a gemm_pd and resolve 'any' desc by querying gemm_pd
            VDISPATCH_MATMUL_SC(
                    create_gemm_pd(gemm_pd_, engine, a_md, b_md, c_md, bias_md,
                            acc_dt, &gemm_attr, false, sum_ab_, sum_ab_type_),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "gemm");
            VDISPATCH_MATMUL_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            init_scratchpad();

            return status::success;
        }

        status_t query(query_t what, int idx, void *result) const override {
            if (what == query::gpu_kernel_count && gemm_pd_)
                return gemm_pd_->query(what, idx, result);
            return matmul::pd_t::query(what, idx, result);
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;
        sum_ab_t sum_ab_ = sum_ab::sum_none;
        data_type_t sum_ab_type_ = data_type::undef;

    private:
        // We cannot change the number of dimensions in the input mds.
        // Grab the gemm_pd_ mds (which should have resolved any tags
        // by now) and unsquash the squashed dims
        status_t set_default_params() {
            auto update_md
                    = [](memory_desc_t &md,
                              const memory_desc_t &reshaped_md) -> status_t {
                if (md.ndims == reshaped_md.ndims) {
                    md = reshaped_md;
                } else {
                    CHECK(memory_desc_reshape(
                            md, reshaped_md, md.ndims, md.dims));
                }
                return status::success;
            };

            CHECK(update_md(src_md_, *gemm_pd_->arg_md(DNNL_ARG_SRC_0)));
            CHECK(update_md(weights_md_, *gemm_pd_->arg_md(DNNL_ARG_SRC_1)));
            CHECK(update_md(dst_md_, *gemm_pd_->arg_md(DNNL_ARG_DST)));
            if (with_bias())
                CHECK(update_md(bias_md_, *gemm_pd_->arg_md(DNNL_ARG_BIAS)));
            return status::success;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(gemm_, pd()->gemm_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> gemm_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
