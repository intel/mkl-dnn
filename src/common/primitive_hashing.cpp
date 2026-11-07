/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include "primitive_attr.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "dnnl_thread.hpp"
#include "engine.hpp"
#include "memory_desc.hpp"
#include "primitive_hashing.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

key_t::key_t(const engine_t *engine, const op_desc_t *op_desc,
        const primitive_attr_t *attr, int pd_iterator_offset,
        const std::vector<memory_desc_t> &hint_mds, int skip_idx)
    : primitive_kind_(op_desc->primitive_kind)
    , op_desc_(op_desc)
    , attr_(attr)
    , pd_iterator_offset_(pd_iterator_offset)
    , impl_nthr_(dnnl_get_max_threads())
    , skip_idx_(skip_idx)
    , hint_mds_(hint_mds)
    , engine_id_(engine->engine_id())
    , thread_id_(std::this_thread::get_id()) {}

key_t::key_t(const primitive_desc_t *pd, const engine_t *engine)
    : key_t(engine, pd->op_desc(), pd->attr(), pd->pd_iterator_offset(),
              pd->hint_mds(false /* is_hint */), pd->skip_idx()) {}

bool key_t::operator==(const key_t &rhs) const {
    DNNL_SHORT_CIRCUIT_SELF_COMPARISON(rhs);
    // clang-format off
    bool ret = true
        // Less expensive comparisons come first
        && primitive_kind_ == rhs.primitive_kind_
        && engine_id_ == rhs.engine_id_
        && hint_mds_.size() == rhs.hint_mds_.size()
        && pd_iterator_offset_ == rhs.pd_iterator_offset_
        && impl_nthr_ == rhs.impl_nthr_
        && skip_idx_ == rhs.skip_idx_
        && (*attr_) == (*rhs.attr_);
    if (!ret) {
        // ANCHOR: HASHING_DEBUGINFO_16.
        VDEBUGINFO(16, primitive, hashing, "operator==,ret=%d", ret);
        return ret;
    }

    for (size_t i = 0; i < rhs.hint_mds_.size(); i++) {
        // Note: fast exit on `!ret` secures the same-sized hints. Make sure
        // there will be no access out-of-bounds for hints on both sides.
        ret = ret && hint_mds_[i] == rhs.hint_mds_[i];
    }

    if (!ret) {
        // ANCHOR: HASHING_DEBUGINFO_16.
        VDEBUGINFO(16, primitive, hashing, "operator==,ret=%d", ret);
        return ret;
    }

#define CASE(pkind) \
    case primitive_kind::pkind: \
        ret = *op_desc_t::to_desc<pkind##_desc_t>(op_desc_) \
                == *op_desc_t::to_desc<pkind##_desc_t>(rhs.op_desc_); \
        break;

        switch ((int)primitive_kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            // Use a custom comparison function that ignores alg_kind.
            case primitive_kind::convolution:
                ret = compare_conv_opdesc(*op_desc_t::to_desc<convolution_desc_t>(op_desc_),
                *op_desc_t::to_desc<convolution_desc_t>(rhs.op_desc_));
            break;
            CASE(deconvolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(group_normalization)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(prelu)
            CASE(reduction)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(sdpa)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            CASE(zero_pad)
            default: assert(!"unknown primitive kind");
        }
#undef CASE
    // clang-format on

    // ANCHOR: HASHING_DEBUGINFO_16.
    VDEBUGINFO(16, primitive, hashing, "operator==,ret=%d", ret);
    return ret;
}

// Combine hash of each memory_desc_t data member
size_t get_md_hash(const memory_desc_t &md) {
    size_t seed = 0;
    seed = get_array_hash(seed, md.dims, md.ndims);
    seed = hash_combine(seed, static_cast<size_t>(md.data_type));
    seed = get_array_hash(seed, md.padded_dims, md.ndims);
    seed = get_array_hash(seed, md.padded_offsets, md.ndims);
    seed = hash_combine(seed, md.offset0);
    seed = hash_combine(seed, static_cast<size_t>(md.format_kind));
    // format desc
    switch ((int)md.format_kind) {
        case format_kind::undef:
        case format_kind::any:
        case format_kind::host_scalar: break; // no additional data to hash
        case format_kind::blocked:
            for (int i = 0; i < md.ndims; i++) {
                if (md.dims[i] == 1 && md.padded_dims[i] == 1) continue;
                seed = hash_combine(seed, md.format_desc.blocking.strides[i]);
            }
            seed = hash_combine(seed, md.format_desc.blocking.inner_nblks);
            seed = get_array_hash(seed, md.format_desc.blocking.inner_blks,
                    md.format_desc.blocking.inner_nblks);
            seed = get_array_hash(seed, md.format_desc.blocking.inner_idxs,
                    md.format_desc.blocking.inner_nblks);
            break;
        case format_kind::wino:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.wino_desc.wino_format));
            seed = hash_combine(seed, md.format_desc.wino_desc.r);
            seed = hash_combine(seed, md.format_desc.wino_desc.alpha);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic2_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc2_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.adj_scale);
            seed = hash_combine(seed, md.format_desc.wino_desc.size);
            break;
        case format_kind::cublaslt_blocked:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.cublaslt_blocked_desc
                                                .cublaslt_format));
            seed = hash_combine(
                    seed, (md.format_desc.cublaslt_blocked_desc.size));
            break;
        case format_kind::zen_packed:
            seed = hash_combine(seed, md.format_desc.zen_packed_desc.size);
            seed = hash_combine(
                    seed, md.format_desc.zen_packed_desc.per_slice_size);
            seed = hash_combine(seed,
                    static_cast<size_t>(
                            md.format_desc.zen_packed_desc.gemm_src_dt));
            seed = hash_combine(seed,
                    static_cast<size_t>(
                            md.format_desc.zen_packed_desc.weights_transposed));
            break;
        case format_kind::rnn_packed:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.rnn_packed_desc.format));
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.n_parts);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.n);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.ldb);
            {
                int n_parts = md.format_desc.rnn_packed_desc.n_parts;
                seed = get_array_hash(
                        seed, md.format_desc.rnn_packed_desc.parts, n_parts);
                seed = get_array_hash(seed,
                        md.format_desc.rnn_packed_desc.part_pack_size, n_parts);
                seed = get_array_hash(seed,
                        md.format_desc.rnn_packed_desc.pack_part, n_parts);
            }
            seed = hash_combine(
                    seed, md.format_desc.rnn_packed_desc.offset_compensation);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.size);
            break;
        case format_kind::sparse:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.sparse_desc.encoding));
            seed = hash_combine(seed, md.format_desc.sparse_desc.nnz);
            seed = get_array_hash(seed,
                    md.format_desc.sparse_desc.metadata_types,
                    sparse_desc_t::max_metadata_types);
            // User cannot initialize `packed_desc` therefore `packed_desc`
            // is always zero initialized.
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
            seed = hash_combine(
                    seed, md.format_desc.sparse_desc.grouped_desc.group_count);
            seed = hash_combine(seed,
                    md.format_desc.sparse_desc.grouped_desc.variable_dim_idx);
#endif
            break;
        default: assert(!"unknown format_kind");
    }

    if (md.extra.flags != dnnl_memory_extra_flag_none) {
        seed = hash_combine(seed, md.extra.flags);
        if (md.extra.flags
                & (dnnl_memory_extra_flag_compensation_conv_s8s8
                        | dnnl_memory_extra_flag_rnn_u8s8_compensation)) {
            seed = hash_combine(seed, md.extra.compensation_mask);
        }

        if (md.extra.flags & dnnl_memory_extra_flag_scale_adjust) {
            seed = hash_combine(seed, md.extra.scale_adjust);
        }

        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_conv_asymmetric_src) {
            seed = hash_combine(seed, md.extra.asymm_compensation_mask);
        }

        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_gpu_conv_asymmetric_src) {
            seed = get_array_hash(seed, md.extra.idhw, 3);
            seed = get_array_hash(seed, md.extra.odhw, 3);
            seed = get_array_hash(seed, md.extra.pdhw, 3);
            seed = get_array_hash(seed, md.extra.ddhw, 3);
            seed = hash_combine(seed, md.extra.dst_size);
        }
    }
    // Combined hash for a memory descriptor
    return seed;
}

// Combine hash of each primitive_attr_t data member
size_t get_attr_hash(const primitive_attr_t &attr) {
    size_t seed = 0;
    // scratchpad_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.scratchpad_mode_));
    // fpmath_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.fpmath_.mode_));
    seed = hash_combine(seed, static_cast<size_t>(attr.fpmath_.apply_to_int_));
    // deterministic
    seed = hash_combine(seed, static_cast<size_t>(attr.deterministic_));
    // acc_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.acc_mode_));
    // rounding_mode
    if (!attr.rounding_mode_.has_default_values()) {
        for (const auto &e : attr.rounding_mode_.rounding_modes_map_) {
            seed = hash_combine(seed, e.first);
            seed = hash_combine(seed, static_cast<size_t>(e.second));
        }
    }

    if (!attr.scales_.has_default_values()) {
        seed = hash_combine(seed, attr.scales_.get_hash());
    }

    if (!attr.zero_points_.has_default_values()) {
        seed = hash_combine(seed, attr.zero_points_.get_hash());
    }

    if (!attr.precomputed_reductions_.has_default_values()) {
        seed = hash_combine(seed, attr.precomputed_reductions_.get_hash());
    }

    // post_ops: entry[:]
    for (int i = 0; i < attr.post_ops_.len(); i++) {
        const auto &entry = attr.post_ops_.entry_[i];
        switch (entry.kind) {
            case primitive_kind::eltwise:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.eltwise.alg));
                seed = hash_combine(seed, entry.eltwise.scale);
                seed = hash_combine(seed, entry.eltwise.alpha);
                seed = hash_combine(seed, entry.eltwise.beta);
                break;
            case primitive_kind::sum:
                seed = hash_combine(seed, entry.sum.scale);
                seed = hash_combine(seed, entry.sum.zero_point);
                seed = hash_combine(seed, static_cast<size_t>(entry.sum.dt));
                break;
            case primitive_kind::convolution:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.kernel));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.stride));
                seed = hash_combine(seed,
                        static_cast<size_t>(entry.depthwise_conv.padding));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.wei_dt));
                seed = hash_combine(seed,
                        static_cast<size_t>(entry.depthwise_conv.bias_dt));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.dst_dt));
                break;
            case primitive_kind::binary:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.binary.alg));
                seed = hash_combine(
                        seed, get_md_hash(entry.binary.user_src1_desc));
                if (entry.is_binary_with_ternary_op())
                    seed = hash_combine(
                            seed, get_md_hash(entry.binary.user_src2_desc));
                break;
            case primitive_kind::prelu:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.prelu.mask));
                break;
            default: assert(!"unknown post_op");
        }
    }
    // rnn_data_qparams: scale, shift
    seed = hash_combine(seed, attr.rnn_data_qparams_.scale_);
    seed = hash_combine(seed, attr.rnn_data_qparams_.shift_);
    if (!attr.rnn_weights_qparams_.has_default_values()) {
        // rnn_weights_qparams: mask
        seed = hash_combine(seed, attr.rnn_weights_qparams_.mask_);
        // rnn_weights_qparams: count
        seed = hash_combine(seed, attr.rnn_weights_qparams_.count_);
        // rnn_weights_qparams: scales[:]
        seed = get_array_hash(seed, attr.rnn_weights_qparams_.scales_,
                attr.rnn_weights_qparams_.count_);
    }
    if (attr.gpu_attr_) {
        seed = hash_combine(seed, attr.gpu_attr_->get_hash());
    }
    if (!attr.dropout_.has_default_values()) {
        seed = hash_combine(seed, attr.dropout_.get_hash());
    }
    // Combined hash for attributes
    return seed;
}

// Functions that compute hash for different op_descs
size_t get_desc_hash(const concat_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(*desc.dst_md));
    // N
    seed = hash_combine(seed, desc.n);
    // Concat dimension
    seed = hash_combine(seed, desc.concat_dimension);
    // Array of mds
    seed = get_array_hash(seed, desc.src_mds);
    // Combined hash for concat desc
    return seed;
}

size_t get_desc_hash(const batch_normalization_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc.batch_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Combined hash for batch normalization desc
    return seed;
}

size_t get_desc_hash(const binary_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc[0]));
    seed = hash_combine(seed, get_md_hash(desc.src_desc[1]));
    if (desc.alg_kind == alg_kind::binary_select)
        seed = hash_combine(seed, get_md_hash(desc.src_desc[2]));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // Combined hash for binary op desc
    return seed;
}

// (De-)Convolution
size_t get_desc_hash(const convolution_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));

    // Ignore `alg_kind` to keep hash value consistent for any algorithm.
    //
    // Background: when a convolution primitive descriptor is created for
    // the algorithm `auto` we overwrite `alg_kind` field in `op_desc` when
    // store it in the primitive descriptor. Because of that, the `op_desc`
    // stored in the primitive descriptor is different from the one user
    // passed to oneDNN API. Because of the difference the requested
    // primitive descriptor cannot be found in the cache if we hash/compare
    // `alg_kind`.
    //seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));

    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc.strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.dilates, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Internal member
    seed = hash_combine(seed, static_cast<size_t>(desc.use_inversion));
    // Combined hash for (de-)convolution desc
    return seed;
}

// Eltwise
size_t get_desc_hash(const eltwise_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Alpha, beta
    seed = hash_combine(seed, desc.alpha);
    seed = hash_combine(seed, desc.beta);
    // Combined hash for eltwise desc
    return seed;
}

size_t get_desc_hash(const gemm_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, get_md_hash(desc.a_desc));
    seed = hash_combine(seed, get_md_hash(desc.b_desc));
    seed = hash_combine(seed, get_md_hash(desc.c_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.acc_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.sum_ab));
    seed = hash_combine(seed, static_cast<size_t>(desc.sum_ab_type));
    // Combined hash for gemm desc
    return seed;
}

size_t get_desc_hash(const group_normalization_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.stat_desc));
    // Groups
    seed = hash_combine(seed, desc.groups);
    // Epsilon
    seed = hash_combine(seed, desc.group_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Combined hash for group_normalization desc
    return seed;
}

size_t get_desc_hash(const inner_product_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for inner_product desc
    return seed;
}

size_t get_desc_hash(const layer_normalization_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc.layer_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Combined hash for layer_normalization desc
    return seed;
}

size_t get_desc_hash(const lrn_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Local size
    seed = hash_combine(seed, desc.local_size);
    // Alpha, beta
    seed = hash_combine(seed, desc.lrn_alpha);
    seed = hash_combine(seed, desc.lrn_beta);
    // k
    seed = hash_combine(seed, desc.lrn_k);
    // Combined hash for lrn desc
    return seed;
}

size_t get_desc_hash(const matmul_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.reduce_desc));
    // Reduce kind.
    seed = hash_combine(seed, static_cast<size_t>(desc.reduce_kind));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for matmul op desc
    return seed;
}

size_t get_desc_hash(const pooling_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc.strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.kernel, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[1], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.dilation, DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for pooling desc
    return seed;
}

size_t get_desc_hash(const prelu_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Combined hash for prelu desc
    return seed;
}

size_t get_desc_hash(const reduction_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // P, eps
    seed = hash_combine(seed, desc.p);
    seed = hash_combine(seed, desc.eps);
    // Combined hash for reduction desc
    return seed;
}

size_t get_desc_hash(const reorder_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // Kinds of source and destination engines
    seed = hash_combine(seed, static_cast<size_t>(desc.src_engine_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.dst_engine_kind));
    seed = hash_combine(seed, desc.is_cross_engine);
    // Combined hash for reorder desc
    return seed;
}

size_t get_desc_hash(const resampling_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Factors
    seed = get_array_hash(seed, desc.factors, DNNL_MAX_NDIMS);
    // Combined hash for resampling op desc
    return seed;
}

size_t get_desc_hash(const rnn_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.cell_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.direction));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_peephole_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_projection_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_peephole_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_projection_desc));
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Activation kind
    seed = hash_combine(seed, static_cast<size_t>(desc.activation_kind));
    // Alpha, beta
    seed = hash_combine(seed, desc.alpha);
    seed = hash_combine(seed, desc.beta);
    // Combined hash for rnn desc
    return seed;
}

// Shuffle
size_t get_desc_hash(const shuffle_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // Axis
    seed = hash_combine(seed, desc.axis);
    // Groupe size
    seed = hash_combine(seed, desc.group_size);
    // Combined hash for shuffle desc
    return seed;
}

size_t get_desc_hash(const softmax_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Axis
    seed = hash_combine(seed, desc.softmax_axis);
    // Combined hash for softmax desc
    return seed;
}

size_t get_desc_hash(const sum_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(*desc.dst_md));
    // N
    seed = hash_combine(seed, desc.n);
    // Scales
    if (desc.scales) { seed = get_array_hash(seed, desc.scales, desc.n); }
    // Array of mds
    seed = get_array_hash(seed, desc.src_mds);
    // Combined hash for sum desc
    return seed;
}

size_t get_desc_hash(const zero_pad_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    return seed;
}

size_t get_desc_hash(const sdpa_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.q_desc));
    seed = hash_combine(seed, get_md_hash(desc.k_desc));
    seed = hash_combine(seed, get_md_hash(desc.v_desc));
    seed = hash_combine(seed, desc.kq_scales.get_hash());
    seed = hash_combine(seed, desc.kq_zero_points.get_hash());
    seed = hash_combine(seed, desc.vs_scales.get_hash());
    seed = hash_combine(seed, desc.vs_zero_points.get_hash());
    seed = hash_combine(seed, get_md_hash(desc.dS_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.stats_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_q_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_k_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_v_desc));
    seed = hash_combine(seed, get_md_hash(desc.attn_mask_desc));
    seed = hash_combine(seed, get_md_hash(desc.scale_desc));
    // Scale type
    seed = hash_combine(seed, static_cast<size_t>(desc.kq_acc_dt));
    seed = hash_combine(seed, static_cast<size_t>(desc.vs_acc_dt));
    seed = hash_combine(seed, desc.invert_scale);
    seed = hash_combine(seed, desc.kv_head_number);
    seed = hash_combine(seed, static_cast<size_t>(desc.mask_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.softmax_alg));
    // Combined hash for sdpa desc
    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl
