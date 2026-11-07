/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void get_arg_indices_for_post_ops(
        const op_t *op, arg_indices_t &indices, size_t &base_index) {
    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (size_t i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            indices.insert({DNNL_GRAPH_ARG_POST_SRC,
                    {indices_t::type_t::input, base_index++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::_binary) {
            indices.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)i) | DNNL_ARG_SRC_1,
                            {indices_t::type_t::input, base_index++}});
            if (static_cast<dnnl::algorithm>(
                        pops[i]->get_op()->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::binary_select) {
                indices.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)i)
                                | DNNL_ARG_SRC_2,
                        {indices_t::type_t::input, base_index++}});
            }
        } else if (pops[i]->get_op()->get_kind() == op_kind::_convolution) {
            indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                    {indices_t::type_t::input, base_index++}});
            if (pops[i]->get_op()->num_inputs() > 2) {
                indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS,
                        {indices_t::type_t::input, base_index++}});
            }
        } else {
        }
    }
}

// for single-input-single-output op
arg_indices_t get_arg_indices_for_siso_op(const op_t *op) {
    arg_indices_t args;

    // add input args
    size_t idx = 0;
    args.insert({DNNL_ARG_FROM, {indices_t::type_t::input, idx++}});
    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    get_arg_indices_for_post_ops(op, args, idx);
    if (fusion_info.with_runtime_scales(false, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_dropout()) {
        args.insert({DNNL_ARG_ATTR_DROPOUT_SEED,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_OFFSET,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_PROBABILITY,
                {indices_t::type_t::input, idx++}});
    }

    // add output args
    args.insert({DNNL_ARG_TO, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    const bool is_training = op->has_attr(op_attr::is_training)
            ? op->get_attr<bool>(op_attr::is_training)
            : false;
    if (is_training) {
        args.insert({DNNL_ARG_WORKSPACE, {indices_t::type_t::output, 2}});
    }

    return args;
}

arg_indices_t get_arg_indices_for_miso_op(const op_t *op) {
    arg_indices_t args;
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + (int)i,
                {indices_t::type_t::input, i}});
    }

    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    return args;
}

// Helper function for conv and deconv forward operations argument indices
arg_indices_t get_arg_indices_for_conv_and_matmul(const op_t *op) {
    arg_indices_t args;

    // add input args
    size_t idx = 0;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, idx++}});
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        args.insert({DNNL_ARG_BIAS, {indices_t::type_t::input, idx++}});
    }

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    if (fusion_info.with_runtime_scales(true, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                {indices_t::type_t::input, idx++}});
    }

    if (fusion_info.with_runtime_scales(true, 1)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                {indices_t::type_t::input, idx++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 0)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                {indices_t::type_t::input, idx++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 1)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                {indices_t::type_t::input, idx++}});
    }

    get_arg_indices_for_post_ops(op, args, idx);
    if (fusion_info.with_runtime_scales(false, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, idx++}});
    }

    if (fusion_info.with_runtime_zero_points(false, 0)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                {indices_t::type_t::input, idx++}});
    }

    if (fusion_info.with_dropout()) {
        args.insert({DNNL_ARG_ATTR_DROPOUT_SEED,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_OFFSET,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_PROBABILITY,
                {indices_t::type_t::input, idx++}});
    }

    // add output args
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return args;
}

arg_indices_t get_arg_indices_for_norm(const op_t *op) {
    arg_indices_t args;
    size_t in_idx = 0;
    const bool is_rms = op->has_attr(op_attr::is_rms)
            ? op->get_attr<bool>(op_attr::is_rms)
            : false;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, in_idx++}});
    if (!op->has_attr(op_attr::use_affine)
            || op->get_attr<bool>(op_attr::use_affine)) {
        // rms doesn't support shift
        if (!is_rms) {
            args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, in_idx++}});
            args.insert({DNNL_ARG_SHIFT, {indices_t::type_t::input, in_idx++}});
        } else {
            if (op->has_attr(op_attr::use_affine)
                    && op->get_attr<bool>(op_attr::use_affine)) {
                args.insert(
                        {DNNL_ARG_SCALE, {indices_t::type_t::input, in_idx++}});
            }
        }
    }

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    get_arg_indices_for_post_ops(op, args, in_idx);

    if (fusion_info.with_runtime_scales(false, 0)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, in_idx++}});
    }

    size_t out_idx = 0;
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, out_idx++}});
    if (!op->has_attr(op_attr::keep_stats)
            || op->get_attr<bool>(op_attr::keep_stats)) {
        // RMSNorm OP in oneDNN Graph API only have 1 output
        if (!is_rms) {
            args.insert(
                    {DNNL_ARG_MEAN, {indices_t::type_t::output, out_idx++}});
            args.insert({DNNL_ARG_VARIANCE,
                    {indices_t::type_t::output, out_idx++}});
        }
    }

    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, out_idx++}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
