/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#define VCHECK_INSERT_OPS(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, insert_ops, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;

status_t insert_permute_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &op : sg->get_ops()) {
        if (!(op->get_kind() == op_kind::_convolution
                    || op->get_kind() == op_kind::_convtranspose
                    || op->get_kind() == op_kind::_convtranspose_bwd_data))
            continue;

        const bool need_permute_src = op->has_attr(op_attr::data_format)
                && op->get_attr<std::string>(op_attr::data_format) == "NXC";
        const bool need_permute_wei = op->has_attr(op_attr::weights_format)
                && op->get_attr<std::string>(op_attr::weights_format) != "OIX";
        // conv + depthwise case
        bool need_permute_post_dw_conv_wei = false;
        const op_t *post_dw_conv = nullptr;
        fusion_info_t fusion_info;
        if (op->has_attr(op_attr::fusion_info)) {
            fusion_info = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        }
        if (fusion_info.has_post_dw_conv()) {
            post_dw_conv = fusion_info.get_post_dw_conv()->get_op();
            need_permute_post_dw_conv_wei = post_dw_conv->get_attr<std::string>(
                                                    op_attr::weights_format)
                    == "XIO";
        }

        size_t num_post_binary_ops = 0;
        size_t dw_conv_index = 2;
        bool with_runtime_dst_scales = false;
        bool with_runtime_dst_points = false;
        const auto &pops = fusion_info.get_post_ops();
        for (size_t n = 0; n < pops.size(); ++n) {
            if (pops[n]->get_op()->get_kind() == op_kind::_binary)
                num_post_binary_ops++;
        }

        if (fusion_info.with_runtime_scales(true, 0)) { dw_conv_index += 1; }
        if (fusion_info.with_runtime_scales(true, 1)) { dw_conv_index += 1; }
        if (fusion_info.with_runtime_zero_points(true, 0)) {
            dw_conv_index += 1;
        }
        if (fusion_info.with_runtime_zero_points(true, 1)) {
            dw_conv_index += 1;
        }
        if (fusion_info.with_runtime_scales(false, 0)) {
            with_runtime_dst_scales = true;
        }
        if (fusion_info.with_runtime_zero_points(false, 0)) {
            with_runtime_dst_points = true;
        }

        for (size_t i = 0; i < op->num_inputs() - with_runtime_dst_scales
                        - with_runtime_dst_points;
                ++i) {
            auto ndims = op->get_input_logical_tensor(i).ndims;

            std::vector<int64_t> perm;
            if (i == 0 && need_permute_src) {
                // optionally permute src
                perm = get_permutation(ndims, "NXC", "NCX");
            } else if (i == 1 && need_permute_wei) {
                // optionally permute weight
                std::string filter_format
                        = op->get_attr<std::string>(op_attr::weights_format);
                perm = get_permutation(ndims, filter_format, "OIX");
            } else if (i == dw_conv_index && need_permute_post_dw_conv_wei) {
                perm = get_permutation(ndims, "XIO", "OIX");
            } else if (i >= op->num_inputs() - num_post_binary_ops
                                    - with_runtime_dst_scales
                                    - with_runtime_dst_points
                    && need_permute_src) {
                // optionally permute post-binary/post-sum inputs
                perm = get_permutation(ndims, "NXC", "NCX");
            }

            if (!perm.empty()) {
                op_ptr perm_op = std::make_shared<op_t>(op_kind::_permute);
                perm_op->set_attr<std::vector<int64_t>>(
                        op_attr::permutation, perm);
                rewriter.insert_op_before(perm_op, op, i);
            }
        }

        // remove the attrs in cur_op to avoid re-permute
        op->set_attr<std::string>(op_attr::data_format, "NCX");
        op->set_attr<std::string>(op_attr::weights_format, "OIX");
        if (need_permute_post_dw_conv_wei)
            const_cast<op_t *>(post_dw_conv)
                    ->set_attr<std::string>(op_attr::weights_format, "OIX");

        // permute output back to NXC
        if (need_permute_src) {
            auto ndims = op->get_output_logical_tensor(0).ndims;
            auto perm = get_permutation(ndims, "NCX", "NXC");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_after(perm_op, op, 0);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

using io_indices_t = std::vector<size_t>;
std::unordered_map<op_kind_t, std::pair<io_indices_t, io_indices_t>>
        io_idx_to_permute = {
                {op_kind::_batchnorm, {{0}, {0}}},
                {op_kind::_prelu, {{0, 1}, {0}}},
                {op_kind::_prelu_bwd, {{0, 1, 2}, {0, 1}}},
                {op_kind::_resampling, {{0}, {0}}},
                {op_kind::_resampling_bwd, {{0, 1}, {0}}},
                {op_kind::_groupnorm, {{0}, {0}}},
};

// insert permute for those ops only requiring data_format attribute
// (e.g batchnorm, interpolate)
status_t insert_permute_for_op_only_require_data_format(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &op : sg->get_ops()) {
        if (!io_idx_to_permute.count(op->get_kind())) continue;

        const bool need_permute = op->has_attr(op_attr::data_format)
                && op->get_attr<std::string>(op_attr::data_format) == "NXC";
        if (!need_permute) continue;

        io_indices_t in_indices = io_idx_to_permute.at(op->get_kind()).first;
        io_indices_t out_indices = io_idx_to_permute.at(op->get_kind()).second;

        // permute explicitly defined inputs
        for (auto idx : in_indices) {
            auto ndims = op->get_input_logical_tensor(idx).ndims;
            auto perm = get_permutation(ndims, "NXC", "NCX");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_before(perm_op, op, idx);
        }

        fusion_info_t fusion_info;
        if (op->has_attr(op_attr::fusion_info)) {
            fusion_info = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        }

        // permute extra inputs for fused post-binary
        const auto &pops = fusion_info.get_post_ops();
        for (size_t n = 0; n < pops.size(); ++n) {
            if (!pops[n]->is_post_binary() && !pops[n]->is_post_sum()) continue;
            const size_t idx = pops[n]->get_unfused_input_indices()[0];

            auto ndims = op->get_input_logical_tensor(idx).ndims;
            auto perm = get_permutation(ndims, "NXC", "NCX");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_before(perm_op, op, idx);
        }

        // permute explicitly defined output back to NXC
        for (auto idx : out_indices) {
            auto ndims = op->get_output_logical_tensor(idx).ndims;
            auto perm = get_permutation(ndims, "NCX", "NXC");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_after(perm_op, op, idx);
        }

        // remove the attrs in cur_op to avoid re-permute
        op->set_attr<std::string>(op_attr::data_format, "NCX");
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_permute_for_shuffle(std::shared_ptr<subgraph_t> &sg) {
    // optimization for NXC case - it will help to hit an optimized kernel
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_shuffle) continue;

        logical_tensor_t src_lt = cur_op->get_input_logical_tensor(0);
        logical_tensor_t dst_lt = cur_op->get_output_logical_tensor(0);
        const logical_tensor_wrapper_t src(src_lt), dst(dst_lt);
        const auto axis = cur_op->get_attr<int64_t>(op_attr::axis);
        const auto known_strides
                = (src.is_strided()) ? !src.is_stride_unknown() : false;
        const bool need_permute = axis == src.ndims() - 1 && known_strides
                && src.vstrides() == get_ncx_strides(src.vdims());
        if (!need_permute) continue;

        const int64_t new_axis = 1;
        cur_op->set_attr(op_attr::axis, new_axis);
        op_ptr perm_src_op = std::make_shared<op_t>(op_kind::_permute);
        auto src_perm = get_permutation(src.ndims(), "NXC", "NCX");
        perm_src_op->set_attr<std::vector<int64_t>>(
                op_attr::permutation, src_perm);
        rewriter.insert_op_before(perm_src_op, cur_op, 0);

        // permute output back to NXC
        op_ptr perm_dst_op = std::make_shared<op_t>(op_kind::_permute);
        auto dst_perm = get_permutation(dst.ndims(), "NCX", "NXC");
        perm_dst_op->set_attr<std::vector<int64_t>>(
                op_attr::permutation, dst_perm);
        rewriter.insert_op_after(perm_dst_op, cur_op, 0);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_to_group_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    auto insert_to_group = [&](const op_ptr &op, int64_t groups,
                                   const size_t offset) -> bool {
        if (groups <= 1) {
            op->set_attr<bool>(op_attr::canonicalized, true);
            return false;
        }

        op_ptr to_group_op = std::make_shared<op_t>(op_kind::_to_group);
        to_group_op->set_attr<int64_t>(op_attr::groups, groups);

        op->set_attr<bool>(op_attr::canonicalized, true);
        op->set_attr<int64_t>(op_attr::groups, 1);

        rewriter.insert_op_before(to_group_op, op, offset);

        if (op->get_kind() == op_kind::_convtranspose
                || op->get_kind() == op_kind::_convtranspose_bwd_data)
            to_group_op->set_attr<bool>(op_attr::is_convtranspose, true);

        return true;
    };

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_convolution
                && cur_op->get_kind() != op_kind::_convtranspose
                && cur_op->get_kind() != op_kind::_convtranspose_bwd_data)
            continue;

        fusion_info_t fusion_info;
        if (cur_op->has_attr(op_attr::fusion_info)) {
            fusion_info = cur_op->get_attr<fusion_info_t>(op_attr::fusion_info);
        }

        if (fusion_info.has_post_dw_conv()) {
            const auto &dw_conv = fusion_info.get_post_dw_conv()->get_op();
            auto dw_conv_groups = dw_conv->get_attr<int64_t>(op_attr::groups);
            const auto inserted = insert_to_group(cur_op, dw_conv_groups,
                    fusion_info.get_post_dw_conv()
                            ->get_unfused_input_indices()[0]);
            if (!inserted) continue;
        }

        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        const auto inserted = insert_to_group(cur_op, groups, 1);
        if (!inserted) continue;
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_to_group_for_reorder(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_reorder) continue;
        auto in_md = make_dnnl_memory_desc(cur_op->get_input_logical_tensor(0));
        auto out_md
                = make_dnnl_memory_desc(cur_op->get_output_logical_tensor(0));
        if (in_md.get_ndims() == out_md.get_ndims()) {
            // no group
            return status::success;
        } else if (in_md.get_ndims() == out_md.get_ndims() + 1) {
            // reorder's input has blocked format with group
            // while output has plain format, perhaps for
            // backward path. No such case for now, disable
            VCHECK_INSERT_OPS(false, status::unimplemented,
                    "unsupported i/o dimentions to insert to_group for "
                    "reorder, input ndims: %d, output ndims: %d",
                    in_md.get_ndims(), out_md.get_ndims());
        } else if (in_md.get_ndims() + 1 == out_md.get_ndims()) {
            // reorder's input has plain format while output
            // has blocked format with group, typically for
            // weight prepacking
            auto group = out_md.get_dims()[0];
            VCHECK_INSERT_OPS(
                    group * out_md.get_dims()[1] == in_md.get_dims()[0],
                    status::invalid_shape,
                    "unmatched shape to insert to_group for reorder, group: "
                    "%ld,"
                    "output dims[1]: %ld, input dims[0], %ld",
                    static_cast<long int>(group),
                    static_cast<long int>(out_md.get_dims()[1]),
                    static_cast<long int>(in_md.get_dims()[0]));
            // insert to_group op
            op_ptr to_group_op = std::make_shared<op_t>(op_kind::_to_group);
            to_group_op->set_attr<int64_t>(op_attr::groups, group);

            rewriter.insert_op_before(to_group_op, cur_op, 0);
        } else {
            // illegal shape
            VCHECK_INSERT_OPS(false, status::invalid_shape,
                    "invalid shape to insert to_group for reorder");
        }
    }

    rewriter.run();
    return status::success;
}

status_t insert_permute_for_dynamic_mul_scale_sub_zp(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    std::vector<op_ptr> permute_op_group;

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_sub_zps
                && cur_op->get_kind() != op_kind::_mul_scales)
            continue;

        if (!cur_op->has_attr(op_attr::mask)
                || cur_op->get_attr<int64_t>(op_attr::mask) == 0)
            continue;

        // This pass only handle dynamic quantization
        if (cur_op->get_kind() == op_kind::_sub_zps) {
            if (!cur_op->has_attr(op_attr::with_runtime_zps)
                    || !cur_op->get_attr<bool>(op_attr::with_runtime_zps))
                continue;

            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::_mul_scales) continue;
            if (!consumer_op.has_attr(op_attr::with_runtime_scales)
                    || !consumer_op.get_attr<bool>(
                            op_attr::with_runtime_scales))
                continue;

            auto scale_out_val = consumer_op.get_output_values()[0];
            auto &scale_consumer_op
                    = scale_out_val->get_consumers()[0].get_op();
            if (scale_consumer_op.get_kind() != op_kind::_matmul) continue;

            if (scale_consumer_op.has_attr(op_attr::transpose_b)
                    && scale_consumer_op.get_attr<bool>(op_attr::transpose_b)) {

                permute_op_group.emplace_back(cur_op);
            }
        } else {
            if (!cur_op->has_attr(op_attr::with_runtime_scales)
                    || !cur_op->get_attr<bool>(op_attr::with_runtime_scales))
                continue;

            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::_matmul) continue;

            if (consumer_op.has_attr(op_attr::transpose_b)
                    && consumer_op.get_attr<bool>(op_attr::transpose_b)) {
                permute_op_group.emplace_back(cur_op);
            }
        }
    }

    if (permute_op_group.empty()) return status::success;

    for (auto &cur_op : permute_op_group) {
        auto ndims = cur_op->get_input_logical_tensor(0).ndims;
        if (utils::has_group_shape(cur_op.get())) {
            op_ptr permute_op = std::make_shared<op_t>(op_kind::_permute);
            auto perm = get_last_two_dims_permutation(ndims);
            permute_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, perm);
            rewriter.insert_op_before(permute_op, cur_op, 1);

            auto group_shape = cur_op->get_attr<std::vector<int64_t>>(
                    op_attr::group_shape);
            std::swap(group_shape[ndims - 1], group_shape[ndims - 2]);
            cur_op->set_attr<std::vector<int64_t>>(
                    op_attr::group_shape, group_shape);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_permute_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_matmul) continue;

        std::vector<bool> trans_flag(2);
        trans_flag[0] = cur_op->has_attr(op_attr::transpose_a)
                && cur_op->get_attr<bool>(op_attr::transpose_a);
        trans_flag[1] = cur_op->has_attr(op_attr::transpose_b)
                && cur_op->get_attr<bool>(op_attr::transpose_b);
        if (!(trans_flag[0] || trans_flag[1])) continue;

        for (size_t i = 0; i < trans_flag.size(); ++i) {
            auto ndims = cur_op->get_input_logical_tensor(i).ndims;
            // skip if transpose flag is false or the input's ndim is 1
            if (!trans_flag[i] || ndims <= 1) continue;
            op_ptr permute_op = std::make_shared<op_t>(op_kind::_permute);
            auto perm = get_last_two_dims_permutation(ndims);
            permute_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, perm);
            rewriter.insert_op_before(permute_op, cur_op, i);
            if (cur_op->has_attr(op_attr::fusion_info)) {
                fusion_info_t fusion_info
                        = cur_op->get_attr<fusion_info_t>(op_attr::fusion_info);
                op_t *scales_op = fusion_info.get_mutable_scales(true, i);
                const bool is_per_group
                        = scales_op && utils::has_group_shape(scales_op);
                if (scales_op && scales_op->has_attr(op_attr::mask)
                        && scales_op->get_attr<int64_t>(op_attr::mask) != 0
                        && !is_per_group) {
                    const int64_t mask
                            = scales_op->get_attr<int64_t>(op_attr::mask);
                    const int64_t last_bit = 1LL << (ndims - 1);
                    const int64_t second_last_bit = 1LL << (ndims - 2);
                    const int64_t transposed_mask
                            = (mask & ~(last_bit | second_last_bit))
                            | ((mask & last_bit) ? second_last_bit : 0)
                            | ((mask & second_last_bit) ? last_bit : 0);
                    scales_op->set_attr<int64_t>(
                            op_attr::mask, transposed_mask);
                }
                cur_op->set_attr<fusion_info_t>(
                        op_attr::fusion_info, fusion_info);
            }
        }
        // remove attr to avoid re-transpose during shape inference
        cur_op->set_attr<bool>(op_attr::transpose_a, false);
        cur_op->set_attr<bool>(op_attr::transpose_b, false);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_permute_for_sdpa_bwd(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_sdpa_bwd) continue;

        for (size_t i = 0; i <= 2; ++i) {
            // check if Q,K,V has permute op in front of sdpa_bwd, if yes,
            // inserting permute for corresponding dQ, dK, dV to keep the
            // consistency of data layout
            if (!cur_op->get_input_value(i)->has_producer()
                    || cur_op->get_input_value(i)->get_producer().get_kind()
                            != op_kind::_permute)
                continue;
            op_t &input_permute_op = cur_op->get_input_value(i)->get_producer();
            auto perm = input_permute_op.get_attr<std::vector<int64_t>>(
                    op_attr::permutation);
            op_ptr output_permute_op
                    = std::make_shared<op_t>(op_kind::_permute);
            output_permute_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, perm);
            rewriter.insert_op_after(output_permute_op, cur_op, i);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_reshape_for_ndx2d_matmul(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_matmul) continue;
        // skip due to dnnl cannot reshape such kind of strided memory desc
        if (cur_op->get_input_value(0)->has_producer()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::_permute) {
            continue;
        }

        int32_t src_ndims = cur_op->get_input_logical_tensor(0).ndims;
        int32_t wei_ndims = cur_op->get_input_logical_tensor(1).ndims;
        if (wei_ndims != 2 || src_ndims <= 2) continue;

        auto src_dims = ltw(cur_op->get_input_logical_tensor(0)).vdims();
        auto wei_dims = ltw(cur_op->get_input_logical_tensor(1)).vdims();
        fusion_info_t fusion_info;
        if (cur_op->has_attr(op_attr::fusion_info)) {
            fusion_info = cur_op->get_attr<fusion_info_t>(op_attr::fusion_info);
        }
        bool can_flatten = true;
        for (const auto &pop : fusion_info.get_post_ops()) {
            if (!pop->is_post_binary() && !pop->is_post_sum()) continue;
            for (const size_t offset : pop->get_unfused_input_indices()) {
                auto post_src_dims
                        = ltw(cur_op->get_input_logical_tensor(offset)).vdims();
                if (post_src_dims.size() > src_dims.size()) {
                    can_flatten = false;
                    break;
                }
                post_src_dims.insert(post_src_dims.begin(),
                        src_dims.size() - post_src_dims.size(), 1);
                // Partially broadcast dimensions cannot be collapsed into M.
                if (!std::equal(post_src_dims.begin(), post_src_dims.end() - 1,
                            src_dims.begin())
                        && !std::all_of(post_src_dims.begin(),
                                post_src_dims.end() - 1,
                                [](dim_t d) { return d == 1; })) {
                    can_flatten = false;
                    break;
                }
            }
            if (!can_flatten) break;
        }
        if (!can_flatten) continue;

        dims expected_dims {-1, src_dims.back()};
        auto reshape_op = std::make_shared<op_t>(op_kind::_reshape);
        reshape_op->set_attr<bool>(op_attr::special_zero, false);
        reshape_op->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims);

        rewriter.insert_op_before(reshape_op, cur_op, 0);

        dims expected_dims2(src_dims);
        expected_dims2[expected_dims2.size() - 1] = wei_dims.back();
        auto reshape_op2 = std::make_shared<op_t>(op_kind::_reshape);
        reshape_op2->set_attr<bool>(op_attr::special_zero, false);
        reshape_op2->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims2);
        rewriter.insert_op_after(reshape_op2, cur_op, 0);

        if (cur_op->has_attr(op_attr::fusion_info)) {
            const auto &pops = fusion_info.get_post_ops();
            for (size_t i = 0; i < pops.size(); i++) {
                if (!pops[i]->is_post_binary() && !pops[i]->is_post_sum())
                    continue;
                for (const size_t offset :
                        pops[i]->get_unfused_input_indices()) {
                    auto post_src_dims
                            = ltw(cur_op->get_input_logical_tensor(offset))
                                      .vdims();
                    dims expected_dims3 {-1, post_src_dims.back()};
                    auto reshape_op3
                            = std::make_shared<op_t>(op_kind::_reshape);
                    reshape_op3->set_attr<bool>(op_attr::special_zero, false);
                    reshape_op3->set_attr<std::vector<int64_t>>(
                            op_attr::shape, expected_dims3);
                    rewriter.insert_op_before(reshape_op3, cur_op, offset);
                }
            }
            // update weight mask
            op_t *scales_op = fusion_info.get_mutable_scales(true, 1);
            if (scales_op && scales_op->has_attr(op_attr::mask)
                    && scales_op->get_attr<int64_t>(op_attr::mask) != 0) {
                scales_op->set_attr<int64_t>(op_attr::mask, 1LL << 1);
            }
            // update fusion info
            cur_op->set_attr<fusion_info_t>(op_attr::fusion_info, fusion_info);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_reshape_for_sdpa(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_sdpa) continue;

        int32_t query_ndims = cur_op->get_input_logical_tensor(0).ndims;
        if (query_ndims != 5) continue;

        // Insert reshape for Query
        auto query_dims = ltw(cur_op->get_input_logical_tensor(0)).vdims();
        dims expected_query_dims {
                query_dims[0], -1, query_dims[3], query_dims[4]};
        op_ptr reshape_query = std::make_shared<op_t>(op_kind::_reshape);
        reshape_query->set_attr<bool>(op_attr::special_zero, false);
        reshape_query->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_query_dims);
        rewriter.insert_op_before(reshape_query, cur_op, 0);

        // Insert reshape for Key
        auto key_dims = ltw(cur_op->get_input_logical_tensor(1)).vdims();
        dims expected_key_dims {key_dims[0], -1, key_dims[3], key_dims[4]};
        op_ptr reshape_key = std::make_shared<op_t>(op_kind::_reshape);
        reshape_key->set_attr<bool>(op_attr::special_zero, false);
        reshape_key->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_key_dims);
        rewriter.insert_op_before(reshape_key, cur_op, 1);

        // Insert reshape for value
        auto value_dims = ltw(cur_op->get_input_logical_tensor(2)).vdims();
        dims expected_value_dims {
                value_dims[0], -1, value_dims[3], value_dims[4]};
        op_ptr reshape_value = std::make_shared<op_t>(op_kind::_reshape);
        reshape_value->set_attr<bool>(op_attr::special_zero, false);
        reshape_value->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_value_dims);
        rewriter.insert_op_before(reshape_value, cur_op, 2);

        size_t index = 3;
        // Insert reshape for scale
        if (cur_op->get_attr<bool>(op_attr::with_scale)) {
            int32_t scale_ndims = cur_op->get_input_logical_tensor(index).ndims;
            if (scale_ndims == 5) {
                auto scale_dims
                        = ltw(cur_op->get_input_logical_tensor(index)).vdims();
                dims expected_scale_dims {
                        scale_dims[0], -1, scale_dims[3], scale_dims[4]};
                op_ptr reshape_scale
                        = std::make_shared<op_t>(op_kind::_reshape);
                reshape_scale->set_attr<bool>(op_attr::special_zero, false);
                reshape_scale->set_attr<std::vector<int64_t>>(
                        op_attr::shape, expected_scale_dims);
                rewriter.insert_op_before(reshape_scale, cur_op, index);
            }
            index += 1;
        }
        // Insert reshape for mask
        if (cur_op->get_attr<int64_t>(op_attr::mask_type)
                == static_cast<int64_t>(attn_mask_type::buffer)) {
            int32_t mask_ndims = cur_op->get_input_logical_tensor(index).ndims;
            if (mask_ndims == 5) {
                auto mask_dims
                        = ltw(cur_op->get_input_logical_tensor(index)).vdims();
                dims expected_mask_dims {
                        mask_dims[0], -1, mask_dims[3], mask_dims[4]};
                op_ptr reshape_mask = std::make_shared<op_t>(op_kind::_reshape);
                reshape_mask->set_attr<bool>(op_attr::special_zero, false);
                reshape_mask->set_attr<std::vector<int64_t>>(
                        op_attr::shape, expected_mask_dims);
                rewriter.insert_op_before(reshape_mask, cur_op, index);
            }
        }

        // Insert reshape for output
        auto output_dims = ltw(cur_op->get_output_logical_tensor(0)).vdims();
        const dims &expected_output_dims = output_dims;
        op_ptr reshape_output = std::make_shared<op_t>(op_kind::_reshape);
        reshape_output->set_attr<bool>(op_attr::special_zero, false);
        reshape_output->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_output_dims);
        rewriter.insert_op_after(reshape_output, cur_op, 0);

        // Insert reshape for optional stats output (output 2)
        if (cur_op->get_attr<bool>(op_attr::is_training)) {
            auto stats_dims = ltw(cur_op->get_output_logical_tensor(2)).vdims();
            const dims &expected_stats_dims = stats_dims;
            op_ptr reshape_stats = std::make_shared<op_t>(op_kind::_reshape);
            reshape_stats->set_attr<bool>(op_attr::special_zero, false);
            reshape_stats->set_attr<std::vector<int64_t>>(
                    op_attr::shape, expected_stats_dims);
            rewriter.insert_op_after(reshape_stats, cur_op, 2);
        }
    }
    rewriter.run();
    return infer_shape(sg);
}

status_t insert_reshape_for_sdpa_bwd(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_sdpa_bwd) continue;

        int32_t query_ndims = cur_op->get_input_logical_tensor(0).ndims;
        if (query_ndims != 5) continue;

        // Helper lambda: create a reshape op collapsing dims[1]*dims[2] -> -1
        // for a 5D tensor [batch, g, h, seq, d] -> [batch, -1, seq, d]
        auto make_reshape_5d_to_4d = [](const dims &in_dims) {
            dims out_dims {in_dims[0], -1, in_dims[3], in_dims[4]};
            op_ptr reshape = std::make_shared<op_t>(op_kind::_reshape);
            reshape->set_attr<bool>(op_attr::special_zero, false);
            reshape->set_attr<std::vector<int64_t>>(op_attr::shape, out_dims);
            return reshape;
        };

        // Insert reshape for Query (input 0)
        auto query_dims = ltw(cur_op->get_input_logical_tensor(0)).vdims();
        rewriter.insert_op_before(make_reshape_5d_to_4d(query_dims), cur_op, 0);

        // Insert reshape for Key (input 1)
        auto key_dims = ltw(cur_op->get_input_logical_tensor(1)).vdims();
        rewriter.insert_op_before(make_reshape_5d_to_4d(key_dims), cur_op, 1);

        // Insert reshape for Value (input 2)
        auto value_dims = ltw(cur_op->get_input_logical_tensor(2)).vdims();
        rewriter.insert_op_before(make_reshape_5d_to_4d(value_dims), cur_op, 2);

        // Insert reshape for dst (input 3, forward output, 5D)
        auto dst_dims = ltw(cur_op->get_input_logical_tensor(3)).vdims();
        rewriter.insert_op_before(make_reshape_5d_to_4d(dst_dims), cur_op, 3);

        // Insert reshape for stats (input 4)
        auto stats_dims = ltw(cur_op->get_input_logical_tensor(4)).vdims();
        rewriter.insert_op_before(make_reshape_5d_to_4d(stats_dims), cur_op, 4);

        // Insert reshape for diff_dst (input 5, 5D same as dst)
        auto diff_dst_dims = ltw(cur_op->get_input_logical_tensor(5)).vdims();
        rewriter.insert_op_before(
                make_reshape_5d_to_4d(diff_dst_dims), cur_op, 5);

        size_t index = 6;
        // Insert reshape for scale (optional)
        if (cur_op->get_attr<bool>(op_attr::with_scale)) {
            int32_t scale_ndims = cur_op->get_input_logical_tensor(index).ndims;
            if (scale_ndims == 5) {
                auto scale_dims
                        = ltw(cur_op->get_input_logical_tensor(index)).vdims();
                rewriter.insert_op_before(
                        make_reshape_5d_to_4d(scale_dims), cur_op, index);
            }
            index += 1;
        }
        // Insert reshape for mask (optional)
        if (cur_op->get_attr<int64_t>(op_attr::mask_type)
                == static_cast<int64_t>(attn_mask_type::buffer)) {
            int32_t mask_ndims = cur_op->get_input_logical_tensor(index).ndims;

            if (mask_ndims == 5) {
                auto mask_dims
                        = ltw(cur_op->get_input_logical_tensor(index)).vdims();
                rewriter.insert_op_before(
                        make_reshape_5d_to_4d(mask_dims), cur_op, index);
            }
        }

        // Insert reshape for diff_query output (output 0) -> 4D to 5D
        auto diff_query_dims
                = ltw(cur_op->get_output_logical_tensor(0)).vdims();
        const dims &expected_diff_query_dims = diff_query_dims;
        op_ptr reshape_diff_query = std::make_shared<op_t>(op_kind::_reshape);
        reshape_diff_query->set_attr<bool>(op_attr::special_zero, false);
        reshape_diff_query->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_diff_query_dims);
        rewriter.insert_op_after(reshape_diff_query, cur_op, 0);

        // Insert reshape for diff_key output (output 1) -> 4D to 5D
        auto diff_key_dims = ltw(cur_op->get_output_logical_tensor(1)).vdims();
        const dims &expected_diff_key_dims = diff_key_dims;
        op_ptr reshape_diff_key = std::make_shared<op_t>(op_kind::_reshape);
        reshape_diff_key->set_attr<bool>(op_attr::special_zero, false);
        reshape_diff_key->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_diff_key_dims);
        rewriter.insert_op_after(reshape_diff_key, cur_op, 1);

        // Insert reshape for diff_value output (output 2) -> 4D to 5D
        auto diff_value_dims
                = ltw(cur_op->get_output_logical_tensor(2)).vdims();
        const dims &expected_diff_value_dims = diff_value_dims;
        op_ptr reshape_diff_value = std::make_shared<op_t>(op_kind::_reshape);
        reshape_diff_value->set_attr<bool>(op_attr::special_zero, false);
        reshape_diff_value->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_diff_value_dims);
        rewriter.insert_op_after(reshape_diff_value, cur_op, 2);

        // Insert reshape for diff_mask output (output 4) -> 4D to 5D
        if (cur_op->num_outputs() > 4) {
            auto diff_mask_dims
                    = ltw(cur_op->get_output_logical_tensor(4)).vdims();
            const dims &expected_diff_mask_dims = diff_mask_dims;
            op_ptr reshape_diff_mask
                    = std::make_shared<op_t>(op_kind::_reshape);
            reshape_diff_mask->set_attr<bool>(op_attr::special_zero, false);
            reshape_diff_mask->set_attr<std::vector<int64_t>>(
                    op_attr::shape, expected_diff_mask_dims);
            rewriter.insert_op_after(reshape_diff_mask, cur_op, 4);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_matmul(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &op : sg->get_ops()) {
        if (op->get_kind() != op_kind::_matmul) continue;

        int32_t src_ndims = op->get_input_logical_tensor(0).ndims;
        int32_t wei_ndims = op->get_input_logical_tensor(1).ndims;
        VCHECK_INSERT_OPS(src_ndims >= 1 && wei_ndims >= 1,
                status::invalid_shape,
                "src_ndims and wei_ndims should >= 1, src_ndims: %d, "
                "wei_ndims: %d",
                src_ndims, wei_ndims);

        int32_t unsqueezed_dst_ndims
                = std::max(std::max(src_ndims, wei_ndims), 2);

        std::vector<int64_t> squeeze_axes;
        for (size_t i = 0; i < op->num_inputs(); i++) {
            logical_tensor_t input_lt = op->get_input_logical_tensor(i);
            if (input_lt.property == property_type::host_scalar) continue;
            int32_t ndims = input_lt.ndims;
            std::vector<int64_t> axes;
            if (i != 1 && src_ndims == 1) {
                // 1D src: [K] -> [1, K], dst of primitive will be [1, N]
                // so bias, post binary: [N] -> [1, N] to align with primive
                // behavior
                axes.emplace_back(-2);
                squeeze_axes.emplace_back(-2);
            } else if (i >= 1 && wei_ndims == 1) {
                // 1D weight: [K] -> [K, 1], dst of primitive will be [N, 1]
                // so bias, post binary: [N] -> [N, 1] to align with primive
                // behavior
                axes.emplace_back(-1);
                squeeze_axes.emplace_back(-1);
            }
            // skip unsqueeze runtime scales
            if (op->get_input_value(i)->has_producer()
                    && impl::utils::one_of(
                            op->get_input_value(i)->get_producer().get_kind(),
                            op_kind::_constant_scales, op_kind::_constant_zps))
                continue;

            size_t batch_dim_num = unsqueezed_dst_ndims - axes.size() - ndims;
            for (size_t b = 0; b < batch_dim_num; b++) {
                axes.emplace_back(b);
            }

            if (!axes.empty()) {
                auto unsqueeze_op = std::make_shared<op_t>(op_kind::_unsqueeze);
                unsqueeze_op->set_attr<std::vector<int64_t>>(
                        op_attr::axes, axes);
                rewriter.insert_op_before(unsqueeze_op, op, i);
            }

            // update the mask for per-channel scales only
            if (i == 1 && op->has_attr(op_attr::fusion_info)) {
                fusion_info_t fusion_info
                        = op->get_attr<fusion_info_t>(op_attr::fusion_info);
                op_t *scales_op = fusion_info.get_mutable_scales(true, 1);
                const bool is_per_group
                        = scales_op && utils::has_group_shape(scales_op);
                if (scales_op && scales_op->has_attr(op_attr::mask)
                        && scales_op->get_attr<int64_t>(op_attr::mask) != 0
                        && !is_per_group && !axes.empty()) {
                    const int64_t mask
                            = scales_op->get_attr<int64_t>(op_attr::mask);
                    const int64_t remapped_mask = wei_ndims == 1
                            ? 1LL << (unsqueezed_dst_ndims - 1)
                            : mask << batch_dim_num;
                    scales_op->set_attr<int64_t>(op_attr::mask, remapped_mask);
                }
                op->set_attr<fusion_info_t>(op_attr::fusion_info, fusion_info);
            }
        }

        // squeeze dst
        if (!squeeze_axes.empty()) {
            auto squeeze_op = std::make_shared<op_t>(op_kind::_squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_axes);
            rewriter.insert_op_after(squeeze_op, op, 0);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t insert_runtime_u8_to_s8_for_matmul(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_matmul) continue;

        int32_t new_src0_dtype = cur_op->get_input_logical_tensor(0).data_type;
        int32_t new_src1_dtype = cur_op->get_input_logical_tensor(1).data_type;
        if (!impl::utils::one_of(
                    new_src0_dtype, impl::data_type::u8, impl::data_type::s8)
                || new_src1_dtype != impl::data_type::u8)
            continue;

        bool with_bias = cur_op->has_attr(op_attr::with_bias)
                && cur_op->get_attr<bool>(op_attr::with_bias);
        const bool has_runtime_src_scales
                = with_runtime_scales(cur_op, true, 0);
        const bool has_runtime_wei_scales
                = with_runtime_scales(cur_op, true, 1);
        const bool has_runtime_src_zps = with_runtime_zps(cur_op, true, 0);
        const bool has_runtime_wei_zps = with_runtime_zps(cur_op, true, 1);

        size_t index = 2;
        if (with_bias) index += 1;
        if (has_runtime_src_scales) index += 1;
        if (has_runtime_wei_scales) index += 1;
        if (has_runtime_src_zps) index += 1;
        if (has_runtime_wei_zps) {
            if (cur_op->get_input_value(index)->has_producer()
                    && cur_op->get_input_value(index)->get_producer().get_kind()
                            == op_kind::_constant_zps) {
                auto &const_ops
                        = cur_op->get_input_value(index)->get_producer();
                std::vector<int64_t> current_zp
                        = const_ops.get_attr<std::vector<int64_t>>(
                                op_attr::zps);
                if (current_zp.size() != 1) continue;
                // the equivalent transformation: mm(src_u8, wei_u8) ->
                // mm(src_u8, wei_u8 - 128 + 128) -> mm(src_u8, wei_s8 + 128),
                // which wei_s8 = wei_u8 - 128
                std::vector<int64_t> adjusted_zp {current_zp[0] - 128};
                const_ops.set_attr<std::vector<int64_t>>(
                        op_attr::zps, adjusted_zp);
            } else {
                // add a binary add here.
            }
        } else {
            VCHECK_INSERT_OPS(cur_op->num_inputs() == index,
                    status::unimplemented,
                    "only support insert input for wei at the end of inputs");
            std::vector<int64_t> zp {-128};
            auto zps_op = std::make_shared<op_t>(op_kind::_add_zps);
            zps_op->set_attr<int64_t>(op_attr::mask, int64_t(0));
            zps_op->set_attr(op_attr::with_runtime_zps, true);
            op_ptr const_data_op;
            const_data_op = std::make_shared<op_t>(op_kind::_constant_zps);
            const_data_op->set_attr(op_attr::zps, zp);
            std::vector<int64_t> dst_shape(1, zp.size());
            const_data_op->set_attr(op_attr::shape, dst_shape);
            logical_tensor_t const_data_dst_lt
                    = empty_logical_tensor_with_default_id();
            auto const_data_dst_value = std::make_shared<value_t>(
                    *const_data_op, 0, const_data_dst_lt, true);
            const_data_dst_value->set_data_type(graph::data_type::s32);
            const_data_dst_value->set_layout_type(layout_type::strided);
            const_data_dst_value->set_strides({1});
            const_data_op->add_output(const_data_dst_value);

            if (cur_op->has_attr(op_attr::fusion_info)) {
                fusion_info_t fusion_info
                        = cur_op->get_attr<fusion_info_t>(op_attr::fusion_info);
                fusion_info.set_zero_points(
                        zps_op->shared_from_this(), true, 1);
                cur_op->set_attr<fusion_info_t>(
                        op_attr::fusion_info, fusion_info);
            } else {
                fusion_info_t fusion_info;
                fusion_info.set_zero_points(
                        zps_op->shared_from_this(), true, 1);
                cur_op->set_attr<fusion_info_t>(
                        op_attr::fusion_info, fusion_info);
            }

            // connect add_zp and constant data
            cur_op->add_input(const_data_dst_value);
            const_data_dst_value->add_consumer(*cur_op, cur_op->num_inputs());
            rewriter.to_insert(const_data_op);
        }

        op_ptr u8_to_s8_op = std::make_shared<op_t>(op_kind::_reorder);

        // make zps as a constant input
        op_ptr const_zps_op;
        const_zps_op = std::make_shared<op_t>(op_kind::_constant_zps);
        const_zps_op->set_attr(op_attr::zps, std::vector<int64_t> {-128});
        const_zps_op->set_attr(op_attr::shape, std::vector<int64_t> {1});
        logical_tensor_t const_zps_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_zps_dst_value = std::make_shared<value_t>(
                *const_zps_op, 0, const_zps_dst_lt, true);
        const_zps_dst_value->set_data_type(graph::data_type::s32);
        const_zps_dst_value->set_layout_type(layout_type::strided);
        const_zps_dst_value->set_strides({1});
        const_zps_op->add_output(const_zps_dst_value);
        u8_to_s8_op->set_attr<int64_t>(op_attr::mask, int64_t(0));
        u8_to_s8_op->set_attr(op_attr::with_runtime_dst_zps, true);
        rewriter.insert_op_before(u8_to_s8_op, cur_op, 1);
        u8_to_s8_op->connect_input(1, const_zps_dst_value);
        u8_to_s8_op->get_output_value(0)->set_data_type(graph::data_type::s8);
        insert_empty_scratchpad(u8_to_s8_op);
        rewriter.to_insert(const_zps_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_u8_to_s8_for_matmul(std::shared_ptr<subgraph_t> &sg) {

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_matmul) continue;

        int32_t new_src0_dtype = cur_op->get_input_logical_tensor(0).data_type;
        int32_t new_src1_dtype = cur_op->get_input_logical_tensor(1).data_type;
        if (!graph::utils::one_of(
                    new_src0_dtype, graph::data_type::u8, graph::data_type::s8)
                || new_src1_dtype != graph::data_type::u8)
            continue;

        if (!cur_op->has_attr(op_attr::fusion_info)) {
            fusion_info_t fusion_info;
            cur_op->set_attr<fusion_info_t>(op_attr::fusion_info, fusion_info);
        }
        fusion_info_t fusion_info
                = cur_op->get_attr<fusion_info_t>(op_attr::fusion_info);

        op_t *wei_zps_op = fusion_info.get_mutable_zero_points(
                true, /*the wei index*/ 1);
        if (wei_zps_op) { // already fused zps, update the zps
            std::vector<int64_t> current_zp
                    = wei_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            if (current_zp.size() != 1) continue;
            // the equivalent transformation: mm(src_u8, wei_u8) -> mm(src_u8,
            // wei_u8 - 128 + 128) -> mm(src_u8, wei_s8 + 128), which wei_s8 =
            // wei_u8 - 128. since the weight zps is actually subtracted in
            // primitive side, so the wei_s8 + 128 should be then converted to
            // wei_s8 - (-128)
            std::vector<int64_t> adjusted_zp {current_zp[0] - 128};
            wei_zps_op->set_attr<std::vector<int64_t>>(
                    op_attr::zps, adjusted_zp);
        } else { // fuse a 128 zps
            std::vector<int64_t> zp {-128};
            auto zps_op = std::make_shared<op_t>(op_kind::_add_zps);
            zps_op->set_attr<int64_t>(op_attr::mask, int64_t(0));
            zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zp);
            fusion_info.set_zero_points(zps_op, true, /*the wei index*/ 1);
        }
        // update fusion info
        cur_op->set_attr<fusion_info_t>(op_attr::fusion_info, fusion_info);

        op_ptr u8_to_s8_op = std::make_shared<op_t>(op_kind::_reorder);
        u8_to_s8_op->set_attr<std::vector<int64_t>>(
                op_attr::dst_zps, std::vector<int64_t> {-128});
        rewriter.insert_op_before(u8_to_s8_op, cur_op, 1);
        u8_to_s8_op->get_output_value(0)->set_data_type(graph::data_type::s8);
        insert_empty_scratchpad(u8_to_s8_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_for_prelu(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_prelu) continue;

        // check doable
        auto src_lt = cur_op->get_input_logical_tensor(0);
        auto wei_lt = cur_op->get_input_logical_tensor(1);
        const std::string data_format
                = cur_op->get_attr<std::string>(op_attr::data_format);
        const bool per_channel_broadcast
                = cur_op->get_attr<bool>(op_attr::per_channel_broadcast);

        const bool prelu_doable_status = prelu_doable(ltw(src_lt).vdims(),
                ltw(wei_lt).vdims(), data_format, per_channel_broadcast);
        VCHECK_INSERT_OPS(prelu_doable_status, status::invalid_shape,
                "invalid shape to insert unsqueeze for prelu");

        // insert unsqueeze op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            std::vector<int64_t> axes(src_ndims - wei_ndims);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format per_channel broadcast PReLU, we need to
            // unsqueeze the 1D weight to [1, C, 1, 1]
            const bool channel_first
                    = data_format == "NCX" && per_channel_broadcast;
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            int wei_id = 1; // weight is the second input
            rewriter.insert_op_before(unsqueeze_op, cur_op, wei_id);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_prelu_bwd(
        std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_prelu_bwd) continue;

        // check doable
        auto src_lt = cur_op->get_input_logical_tensor(0);
        auto wei_lt = cur_op->get_input_logical_tensor(1);
        const auto wei_vdims = ltw(wei_lt).vdims();
        const std::string data_format
                = cur_op->get_attr<std::string>(op_attr::data_format);

        // In backward pass if the slope is one-dimensional
        // and its dims[0] != 1, then per channel broadcast will be performed.
        const bool per_channel_broadcast
                = wei_vdims.size() == 1 && wei_vdims[0] != 1;

        const bool prelu_doable_status = prelu_doable(ltw(src_lt).vdims(),
                wei_vdims, data_format, per_channel_broadcast);
        VCHECK_INSERT_OPS(prelu_doable_status, status::invalid_shape,
                "invalid shape to insert unsqueeze for prelu");

        // insert unsqueeze op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            std::vector<int64_t> axes(src_ndims - wei_ndims);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format per_channel broadcast PReLU, we need to
            // unsqueeze the 1D weight to [1, C, 1, 1]
            const bool channel_first
                    = data_format == "NCX" && per_channel_broadcast;
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            int wei_id = 1; // weight is the second input
            rewriter.insert_op_before(unsqueeze_op, cur_op, wei_id);

            // the squeeze is exactly the inverse of unsqueeze, so they use same
            // axes
            std::vector<int64_t> squeeze_axes = axes;
            op_ptr squeeze_op = std::make_shared<op_t>(op_kind::_squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_axes);
            // Insert squeeze after diff weights, so that its dimensions
            // have their original shape.
            rewriter.insert_op_after(squeeze_op, cur_op, 1);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_reduction(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::_reduction) continue;

        const auto keep_dims = cur_op->get_attr<bool>(op_attr::keep_dims);
        if (keep_dims) continue;

        const auto axes = cur_op->get_attr<std::vector<int64_t>>(op_attr::axes);

        // go through successor OPs until reach the end of subgraph or OP
        // which is not supported as a reductions post-op
        op_t *cur_op_ptr = cur_op.get();
        while (!cur_op_ptr->get_output_value(0)->get_consumers().empty()) {
            value_ptr connector = cur_op_ptr->get_output_value(0);
            op_t &post_op = connector->get_consumers()[0].get_op();
            if (post_op.get_kind() != op_kind::_binary
                    && post_op.get_kind() != op_kind::_eltwise)
                break;

            size_t src1_offset
                    = (post_op.get_input_value(0).get() == connector.get()) ? 1
                                                                            : 0;
            // insert unsqueeze op before binary's src1 input
            if (post_op.get_kind() == op_kind::_binary) {
                if (!post_binary_fusible(cur_op.get(), &post_op)) break;
                op_ptr unsqueeze_op
                        = std::make_shared<op_t>(op_kind::_unsqueeze);
                unsqueeze_op->set_attr<std::vector<int64_t>>(
                        op_attr::axes, axes);
                rewriter.insert_op_before(
                        unsqueeze_op, post_op.shared_from_this(), src1_offset);
            }

            // clear the existing shape for cur_op_ptr output (which is post-op
            // input value), so later correct shape will be inferred
            cur_op_ptr->get_output_value(0)->set_ndims(-1);
            cur_op_ptr = &post_op;
        }

        op_ptr squeeze_op = std::make_shared<op_t>(op_kind::_squeeze);
        squeeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
        // insert squeeze op after reduction or after its last post-op
        rewriter.insert_op_after(squeeze_op, cur_op_ptr->shared_from_this(), 0);

        // set to true, as squeeze will be handled by separate op
        cur_op->set_attr(op_attr::keep_dims, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_host_scalar(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    std::set<value_t *> visited;
    for (const auto &val : sg->get_input_values()) {
        if (visited.count(val)) continue;
        visited.insert(val);

        logical_tensor_t lt = val->get_logical_tensor();
        if (lt.property == property_type::host_scalar) {
            // Special case for dropout op, since dropout op can accept host
            // scalar input directly
            if (val->get_consumers()[0].get_op().get_kind()
                    == op_kind::_dropout)
                continue;

            // Create a new dnnl_host_scalar op
            op_ptr host_scalar_op
                    = std::make_shared<op_t>(op_kind::_host_scalar);
            logical_tensor_t host_scalar_op_out_lt
                    = empty_logical_tensor_with_default_id();
            auto host_scalar_op_out_val = std::make_shared<value_t>(
                    *host_scalar_op, 0, host_scalar_op_out_lt, true);
            host_scalar_op_out_val->set_data_type(lt.data_type);

            std::shared_ptr<value_t> shared_val;
            auto consumers = val->get_consumers();
            for (const auto &consumer : consumers) {
                if (!shared_val) {
                    shared_val = consumer.get_op().get_input_value(
                            consumer.get_offset());
                }
                val->remove_consumer(consumer.get_op(), consumer.get_offset());
                consumer.get_op().connect_input(
                        consumer.get_offset(), host_scalar_op_out_val);
                // Outputs may have been inferred from the 0-D scalar input,
                // reset them so they are re-inferred from the _host_scalar
                // output.
                for (auto &out_val : consumer.get_op().get_output_values()) {
                    if (!out_val->get_consumers().empty())
                        out_val->set_ndims(-1);
                }
            }

            val->add_consumer(*host_scalar_op, 0);
            host_scalar_op->add_input(shared_val);
            host_scalar_op->add_output(host_scalar_op_out_val);
            rewriter.to_insert(host_scalar_op);
        }
    }
    rewriter.run();
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
