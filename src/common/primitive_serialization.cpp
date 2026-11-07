/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/primitive_serialization.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

status_t serialize_desc(
        serialization_stream_t &sstream, const op_desc_t *op_desc) {
#define CASE(pkind) \
    case primitive_kind::pkind: \
        serialize(sstream, *(const pkind##_desc_t *)op_desc); \
        break;

    switch ((int)op_desc->primitive_kind) {
        CASE(batch_normalization)
        CASE(binary)
        CASE(concat)
        CASE(convolution)
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
        default: return status::invalid_arguments;
    }
#undef CASE
    return status::success;
}

void serialize_md_blocking_desc(
        serialization_stream_t &sstream, const blocking_desc_t &bd, int ndims) {
    sstream.append_array(ndims, bd.strides);
    sstream.append(bd.inner_nblks);
    sstream.append_array(bd.inner_nblks, bd.inner_blks);
    sstream.append_array(bd.inner_nblks, bd.inner_idxs);
}

void serialize(serialization_stream_t &sstream, const memory_desc_t &md) {
    sstream.append(md.ndims);
    sstream.append_array(md.ndims, md.dims);
    sstream.append(md.data_type);
    sstream.append_array(md.ndims, md.padded_dims);
    sstream.append_array(md.ndims, md.padded_offsets);
    sstream.append(md.offset0);
    sstream.append(md.format_kind);
    // format desc
    switch ((int)md.format_kind) {
        case format_kind::undef:
        case format_kind::any:
        case format_kind::host_scalar: break; // no additional data to serialize
        case format_kind::sparse:
            sstream.append(md.format_desc.sparse_desc.encoding);
            sstream.append(md.format_desc.sparse_desc.nnz);
            sstream.append_array(sparse_desc_t::max_metadata_types,
                    md.format_desc.sparse_desc.metadata_types);
            // User cannot initialize `packed_desc` therefore `packed_desc`
            // is always zero initialized.
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
            if (md.format_desc.sparse_desc.encoding
                    == sparse_encoding::grouped) {
                sstream.append(
                        md.format_desc.sparse_desc.grouped_desc.group_count);
                sstream.append(md.format_desc.sparse_desc.grouped_desc
                                       .variable_dim_idx);
            }
#endif
            break;
        case format_kind::blocked:
            serialize_md_blocking_desc(
                    sstream, md.format_desc.blocking, md.ndims);
            break;
        case format_kind::wino:
            sstream.append(md.format_desc.wino_desc.wino_format);
            sstream.append(md.format_desc.wino_desc.r);
            sstream.append(md.format_desc.wino_desc.alpha);
            sstream.append(md.format_desc.wino_desc.ic);
            sstream.append(md.format_desc.wino_desc.oc);
            sstream.append(md.format_desc.wino_desc.ic_block);
            sstream.append(md.format_desc.wino_desc.oc_block);
            sstream.append(md.format_desc.wino_desc.ic2_block);
            sstream.append(md.format_desc.wino_desc.oc2_block);
            sstream.append(md.format_desc.wino_desc.adj_scale);
            sstream.append(md.format_desc.wino_desc.size);
            break;
        case format_kind::cublaslt_blocked:
            sstream.append(
                    md.format_desc.cublaslt_blocked_desc.cublaslt_format);
            sstream.append(md.format_desc.cublaslt_blocked_desc.size);
            break;
        case format_kind::zen_packed:
            sstream.append(md.format_desc.zen_packed_desc.size);
            sstream.append(md.format_desc.zen_packed_desc.per_slice_size);
            sstream.append(md.format_desc.zen_packed_desc.gemm_src_dt);
            sstream.append(md.format_desc.zen_packed_desc.weights_transposed);
            break;
        case format_kind::rnn_packed:
            sstream.append(md.format_desc.rnn_packed_desc.format);
            sstream.append(md.format_desc.rnn_packed_desc.n_parts);
            sstream.append(md.format_desc.rnn_packed_desc.n);
            sstream.append(md.format_desc.rnn_packed_desc.ldb);
            {
                int n_parts = md.format_desc.rnn_packed_desc.n_parts;
                sstream.append_array(
                        n_parts, md.format_desc.rnn_packed_desc.parts);
                sstream.append_array(
                        n_parts, md.format_desc.rnn_packed_desc.part_pack_size);
                sstream.append_array(
                        n_parts, md.format_desc.rnn_packed_desc.pack_part);
            }
            sstream.append(md.format_desc.rnn_packed_desc.offset_compensation);
            sstream.append(md.format_desc.rnn_packed_desc.size);
            break;
        default: assert(!"unknown format_kind");
    }

    if (md.extra.flags != dnnl_memory_extra_flag_none) {
        sstream.append(md.extra.flags);
        if (md.extra.flags
                & (dnnl_memory_extra_flag_compensation_conv_s8s8
                        | dnnl_memory_extra_flag_rnn_u8s8_compensation)) {
            sstream.append(md.extra.compensation_mask);
        }
        if (md.extra.flags & dnnl_memory_extra_flag_scale_adjust) {
            sstream.append(md.extra.scale_adjust);
        }
        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_conv_asymmetric_src) {
            sstream.append(md.extra.asymm_compensation_mask);
        }
        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_gpu_conv_asymmetric_src) {
            sstream.append_array(3, md.extra.idhw);
            sstream.append_array(3, md.extra.odhw);
            sstream.append_array(3, md.extra.pdhw);
            sstream.append_array(3, md.extra.ddhw);
            sstream.append(md.extra.dst_size);
        }
    }
}

void serialize(serialization_stream_t &sstream, const post_ops_t &post_ops) {
    // post_ops: entry[:]
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &entry = post_ops.entry_[i];
        switch (entry.kind) {
            case primitive_kind::eltwise:
                sstream.append(entry.eltwise.alg);
                sstream.append(entry.eltwise.scale);
                sstream.append(entry.eltwise.alpha);
                sstream.append(entry.eltwise.beta);
                break;
            case primitive_kind::sum:
                sstream.append(entry.sum.scale);
                sstream.append(entry.sum.zero_point);
                sstream.append(entry.sum.dt);
                break;
            case primitive_kind::convolution:
                sstream.append(entry.depthwise_conv.kernel);
                sstream.append(entry.depthwise_conv.stride);
                sstream.append(entry.depthwise_conv.padding);
                sstream.append(entry.depthwise_conv.wei_dt);
                sstream.append(entry.depthwise_conv.bias_dt);
                sstream.append(entry.depthwise_conv.dst_dt);
                break;
            case primitive_kind::binary:
                sstream.append(entry.binary.alg);
                serialize(sstream, entry.binary.user_src1_desc);
                if (entry.is_binary_with_ternary_op())
                    serialize(sstream, entry.binary.user_src2_desc);
                break;
            case primitive_kind::prelu: sstream.append(entry.prelu.mask); break;
            default: assert(!"unknown post_op");
        }
    }
}

void serialize(serialization_stream_t &sstream, const primitive_attr_t &attr) {
    // scratchpad_mode
    sstream.append(attr.scratchpad_mode_);
    // fpmath_mode
    sstream.append(attr.fpmath_.mode_);
    sstream.append(attr.fpmath_.apply_to_int_);
    // deterministic
    sstream.append(attr.deterministic_);
    // acc_mode
    sstream.append(attr.acc_mode_);

    // scales
    if (!attr.scales_.has_default_values()) {
        sstream.append('s');
        attr.scales_.serialize(sstream);
    }
    // zero_points
    if (!attr.zero_points_.has_default_values()) {
        sstream.append('z');
        attr.zero_points_.serialize(sstream);
    }

    if (!attr.precomputed_reductions_.has_default_values()) {
        sstream.append('p');
        attr.precomputed_reductions_.serialize(sstream);
    }

    // Rounding modes
    if (!attr.rounding_mode_.has_default_values()) sstream.append('r');
    for (const auto &e : attr.rounding_mode_.rounding_modes_map_) {
        if (!attr.rounding_mode_.has_default_values(e.first)) {
            sstream.append(e.first);
            sstream.append(e.second);
        }
    }

    if (!attr.dropout_.has_default_values()) {
        sstream.append('d');
        attr.dropout_.serialize(sstream);
    }

    serialize(sstream, attr.post_ops_);

    // rnn_data_qparams: scale, shift
    sstream.append(attr.rnn_data_qparams_.scale_);
    sstream.append(attr.rnn_data_qparams_.shift_);
    if (!attr.rnn_weights_qparams_.has_default_values()) {
        // rnn_weights_qparams: mask
        sstream.append(attr.rnn_weights_qparams_.mask_);
        // rnn_weights_qparams: count
        sstream.append(attr.rnn_weights_qparams_.count_);
        // rnn_weights_qparams: scales[:]
        sstream.append_array(attr.rnn_weights_qparams_.count_,
                attr.rnn_weights_qparams_.scales_);
    }
    if (attr.gpu_attr_) {
        attr.gpu_attr_->serialize(sstream);
    } else {
        int zero = 0;
        sstream.append(zero);
    }
}

void serialize(serialization_stream_t &sstream, const concat_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    // Memory descriptors
    serialize(sstream, *desc.dst_md);
    // N
    sstream.append(desc.n);
    // Concat dimension
    sstream.append(desc.concat_dimension);
    // Array of mds
    for (int i = 0; i < desc.n; i++)
        serialize(sstream, *desc.src_mds[i]);
}

void serialize(serialization_stream_t &sstream,
        const batch_normalization_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.diff_dst_desc);
    serialize(sstream, desc.scaleshift_desc);
    serialize(sstream, desc.diff_scaleshift_desc);
    serialize(sstream, desc.stat_desc);
    // Epsilon
    sstream.append(desc.batch_norm_epsilon);
    // Flags
    sstream.append(desc.flags);
}

void serialize(serialization_stream_t &sstream, const binary_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc[0]);
    serialize(sstream, desc.src_desc[1]);
    serialize(sstream, desc.src_desc[2]);
    serialize(sstream, desc.dst_desc);
}

// (De-)Convolution
void serialize(
        serialization_stream_t &sstream, const convolution_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.weights_desc);
    serialize(sstream, desc.diff_weights_desc);
    serialize(sstream, desc.bias_desc);
    serialize(sstream, desc.diff_bias_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Strides, dilates, padding
    sstream.append_array(DNNL_MAX_NDIMS, desc.strides);
    sstream.append_array(DNNL_MAX_NDIMS, desc.dilates);
    sstream.append_array(DNNL_MAX_NDIMS, desc.padding[0]);
    sstream.append_array(DNNL_MAX_NDIMS, desc.padding[1]);
    // Accumulator type
    sstream.append(desc.accum_data_type);
    // Internal member
    sstream.append(desc.use_inversion);
}

// Eltwise
void serialize(serialization_stream_t &sstream, const eltwise_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Alpha, beta
    sstream.append(desc.alpha);
    sstream.append(desc.beta);
}

void serialize(serialization_stream_t &sstream, const gemm_desc_t &desc) {
    // Kind
    sstream.append(desc.primitive_kind);
    serialize(sstream, desc.a_desc);
    serialize(sstream, desc.b_desc);
    serialize(sstream, desc.c_desc);
    serialize(sstream, desc.bias_desc);
    // Accumulator type
    sstream.append(desc.acc_type);
    sstream.append(desc.sum_ab);
    sstream.append(desc.sum_ab_type);
}

void serialize(serialization_stream_t &sstream,
        const group_normalization_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.diff_dst_desc);
    serialize(sstream, desc.scaleshift_desc);
    serialize(sstream, desc.diff_scaleshift_desc);
    serialize(sstream, desc.stat_desc);
    // Groups
    sstream.append(desc.groups);
    // Epsilon
    sstream.append(desc.group_norm_epsilon);
    // Flags
    sstream.append(desc.flags);
}

void serialize(
        serialization_stream_t &sstream, const inner_product_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.weights_desc);
    serialize(sstream, desc.diff_weights_desc);
    serialize(sstream, desc.bias_desc);
    serialize(sstream, desc.diff_bias_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Accumulator type
    sstream.append(desc.accum_data_type);
}

void serialize(serialization_stream_t &sstream,
        const layer_normalization_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.data_scaleshift_desc);
    serialize(sstream, desc.diff_data_scaleshift_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    serialize(sstream, desc.stat_desc);
    // Epsilon
    sstream.append(desc.layer_norm_epsilon);
    // Flags
    sstream.append(desc.flags);
}

void serialize(serialization_stream_t &sstream, const lrn_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Local size
    sstream.append(desc.local_size);
    // Alpha, beta
    sstream.append(desc.lrn_alpha);
    sstream.append(desc.lrn_beta);
    // k
    sstream.append(desc.lrn_k);
}

void serialize(serialization_stream_t &sstream, const matmul_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.weights_desc);
    serialize(sstream, desc.bias_desc);
    serialize(sstream, desc.dst_desc);
    // Accumulator type
    sstream.append(desc.accum_data_type);
}

void serialize(serialization_stream_t &sstream, const pooling_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Strides, dilates, padding
    sstream.append_array(DNNL_MAX_NDIMS, desc.strides);
    sstream.append_array(DNNL_MAX_NDIMS, desc.kernel);
    sstream.append_array(DNNL_MAX_NDIMS, desc.padding[0]);
    sstream.append_array(DNNL_MAX_NDIMS, desc.padding[1]);
    sstream.append_array(DNNL_MAX_NDIMS, desc.dilation);
    // Accumulator type
    sstream.append(desc.accum_data_type);
}

void serialize(serialization_stream_t &sstream, const prelu_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.weights_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.diff_weights_desc);
    serialize(sstream, desc.diff_dst_desc);
}

void serialize(serialization_stream_t &sstream, const reduction_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    // P, eps
    sstream.append(desc.p);
    sstream.append(desc.eps);
}

void serialize(serialization_stream_t &sstream, const reorder_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    // Kinds of source and destination engines
    sstream.append(desc.src_engine_kind);
    sstream.append(desc.dst_engine_kind);
    sstream.append(desc.is_cross_engine);
}

void serialize(serialization_stream_t &sstream, const resampling_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Factors
    sstream.append_array(DNNL_MAX_NDIMS, desc.factors);
}

void serialize(serialization_stream_t &sstream, const rnn_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.cell_kind);
    sstream.append(desc.direction);
    // Memory descriptors
    serialize(sstream, desc.src_layer_desc);
    serialize(sstream, desc.src_iter_desc);
    serialize(sstream, desc.src_iter_c_desc);
    serialize(sstream, desc.weights_layer_desc);
    serialize(sstream, desc.weights_iter_desc);
    serialize(sstream, desc.bias_desc);
    serialize(sstream, desc.dst_layer_desc);
    serialize(sstream, desc.dst_iter_desc);
    serialize(sstream, desc.dst_iter_c_desc);
    serialize(sstream, desc.weights_peephole_desc);
    serialize(sstream, desc.weights_projection_desc);
    serialize(sstream, desc.diff_src_layer_desc);
    serialize(sstream, desc.diff_src_iter_desc);
    serialize(sstream, desc.diff_src_iter_c_desc);
    serialize(sstream, desc.diff_weights_layer_desc);
    serialize(sstream, desc.diff_weights_iter_desc);
    serialize(sstream, desc.diff_bias_desc);
    serialize(sstream, desc.diff_dst_layer_desc);
    serialize(sstream, desc.diff_dst_iter_desc);
    serialize(sstream, desc.diff_dst_iter_c_desc);
    serialize(sstream, desc.diff_weights_peephole_desc);
    serialize(sstream, desc.diff_weights_projection_desc);
    // Flags
    sstream.append(desc.flags);
    // Activation kind
    sstream.append(desc.activation_kind);
    // Alpha, beta
    sstream.append(desc.alpha);
    sstream.append(desc.beta);
}

// Shuffle
void serialize(serialization_stream_t &sstream, const shuffle_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.dst_desc);
    // Axis
    sstream.append(desc.axis);
    // Groupe size
    sstream.append(desc.group_size);
}

void serialize(serialization_stream_t &sstream, const softmax_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    sstream.append(desc.alg_kind);
    // Memory descriptors
    serialize(sstream, desc.src_desc);
    serialize(sstream, desc.diff_src_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.diff_dst_desc);
    // Axis
    sstream.append(desc.softmax_axis);
}

void serialize(serialization_stream_t &sstream, const sum_desc_t &desc) {
    // Kinds
    sstream.append(desc.primitive_kind);
    // Memory descriptors
    serialize(sstream, *desc.dst_md);
    // N
    sstream.append(desc.n);
    // Scales
    sstream.append_array(desc.n, desc.scales);
    // Array of mds
    for (int i = 0; i < desc.n; i++)
        serialize(sstream, *desc.src_mds[i]);
}

void serialize(serialization_stream_t &sstream, const sdpa_desc_t &desc) {
    // Kind
    sstream.append(desc.primitive_kind);
    sstream.append(desc.prop_kind);
    serialize(sstream, desc.q_desc);
    serialize(sstream, desc.k_desc);
    serialize(sstream, desc.v_desc);
    desc.kq_scales.serialize(sstream);
    desc.kq_zero_points.serialize(sstream);
    desc.vs_scales.serialize(sstream);
    desc.vs_zero_points.serialize(sstream);
    serialize(sstream, desc.dS_desc);
    serialize(sstream, desc.dst_desc);
    serialize(sstream, desc.stats_desc);
    serialize(sstream, desc.diff_dst_desc);
    serialize(sstream, desc.diff_q_desc);
    serialize(sstream, desc.diff_k_desc);
    serialize(sstream, desc.diff_v_desc);
    serialize(sstream, desc.attn_mask_desc);
    serialize(sstream, desc.scale_desc);
    sstream.append(desc.kq_acc_dt);
    sstream.append(desc.vs_acc_dt);
    sstream.append(desc.invert_scale);
    sstream.append(desc.kv_head_number);
    sstream.append(desc.mask_type);
    sstream.append(desc.softmax_alg);
}

} // namespace impl
} // namespace dnnl
