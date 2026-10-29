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
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/shape_infer.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/debug.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using ltw = logical_tensor_wrapper_t;

status_t set_given_inputs_outputs(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    sg->ins_ = inputs;
    sg->outs_ = outputs;

    // set the inputs's layout to subgraph's inputs value
    auto graph_in_vals = sg->get_input_values();
    auto graph_out_vals = sg->get_output_values();

    auto func = [](std::vector<value_t *> &edges,
                        const std::vector<logical_tensor_t> &givens,
                        bool check_given, bool must_have_shape) {
        for (auto &edge : edges) {
            size_t edge_id = edge->get_logical_tensor().id;

            // partition in/outs should not have default id. There must be some
            // errors in previous graph transformation stage
            VCHECK_UTILS(edge_id != std::numeric_limits<size_t>::max(),
                    status::invalid_graph, "Invalid edge_id %zu", edge_id);
            bool found = false;
            for (const auto &given : givens) {
                if (edge_id == given.id) {
                    if (check_given) {
                        logical_tensor_wrapper_t given_ltw(given);
                        // check given lts
                        bool valid = !given_ltw.is_data_type_undef()
                                && !given_ltw.is_layout_type_undef();
                        if (must_have_shape) {
                            valid = valid && !given_ltw.is_empty();
                            // ndims=0 means the tensor is a scalar, we don't
                            // need to check its shape
                            if (given_ltw.ndims() > 0) {
                                for (auto dim : given_ltw.vdims()) {
                                    valid = valid
                                            && dim != DNNL_GRAPH_UNKNOWN_DIM;
                                }
                            }
                        }
                        VCHECK_UTILS(valid, status::invalid_arguments,
                                "Invalid given logical tensor for given.id %zu",
                                given.id);
                    }

                    edge->set_logical_tensor(given);
                    found = true;
                    break;
                }
            }

            VCHECK_UTILS(found, status::invalid_arguments,
                    "Can't find given logical tensor for edge_id %zu", edge_id);
        }
        return status::success;
    };

    CHECK(func(graph_in_vals, inputs, true, true));
    CHECK(func(graph_out_vals, outputs, true, false));
    return status::success;
}

status_t set_given_inputs_outputs(std::vector<op_ptr> &subgraph,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    auto sg = std::make_shared<subgraph_t>(subgraph);
    return set_given_inputs_outputs(sg, inputs, outputs);
}

// Constant property should be set by users from API level, this function is
// just a workaround at this moment.
void set_weight_bias_constant(std::shared_ptr<subgraph_t> &sg) {
    for (auto &op : sg->get_ops()) {
        if (!(op->get_kind() == op_kind::_matmul
                    || op->get_kind() == op_kind::_convolution))
            continue;

        // set weight to be constant
        op->get_input_value(1)->set_property(property_type::constant);

        // set bias to be constant
        if (op->has_attr(op_attr::with_bias)
                && op->get_attr<bool>(op_attr::with_bias)) {
            op->get_input_value(2)->set_property(property_type::constant);
        }
    }
}

void merge_common_eltwise_attrs(
        const std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op) {
    if (org_op->has_attr(op_attr::alpha)) {
        new_op->set_attr<float>(
                op_attr::alpha, org_op->get_attr<float>(op_attr::alpha));
    } else if (org_op->has_attr(op_attr::min)) {
        new_op->set_attr<float>(
                op_attr::alpha, org_op->get_attr<float>(op_attr::min));
    } else if (org_op->get_kind() == graph::op_kind::HardSwish
            || org_op->get_kind() == graph::op_kind::HardSwishBackward) {
        // in v3.0, users need to explicitly specify the alpha
        new_op->set_attr<float>(op_attr::alpha, 1.f / 6.f);
    } else {
        new_op->set_attr<float>(op_attr::alpha, 0);
    }

    if (org_op->has_attr(op_attr::beta)) {
        new_op->set_attr<float>(
                op_attr::beta, org_op->get_attr<float>(op_attr::beta));
    } else if (org_op->has_attr(op_attr::max)) {
        new_op->set_attr<float>(
                op_attr::beta, org_op->get_attr<float>(op_attr::max));
    } else if (org_op->get_kind() == graph::op_kind::HardSwish
            || org_op->get_kind() == graph::op_kind::HardSwishBackward) {
        // in v3.0, users need to explicitly specify the beta
        new_op->set_attr<float>(op_attr::beta, 1.f / 2.f);
    } else {
        new_op->set_attr<float>(op_attr::beta, 0);
    }
}

std::vector<value_t *> get_constant_block_output_values(
        const std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;
    std::vector<value_t *> ret;
    auto func = [&](op_t *op) {
        auto out_vals = op->get_output_values();
        for (auto &val : out_vals) {
            if (!ltw(val->get_logical_tensor()).is_constant()) continue;
            // if a constant value feed into a consumer whose output is not
            // constant, then the value is the final output of a constant block
            auto consumers = val->get_consumers();
            for (auto &csm : consumers) {
                // A consumer is not constant
                if (!csm.get_op().get_attr<bool>(op_attr::is_constant)) {
                    ret.emplace_back(val.get());
                    break;
                }
            }
        }
        return status::success;
    };
    status_t status = topo_order_visit(sg->get_output_ops(), func);
    VCHECK_UTILS(status == status::success, {},
            "Failed to get constant block output values");
    return ret;
}

status_t infer_shape(std::shared_ptr<subgraph_t> &sg) {
    // workaround: the conv output shape will be impacted if the post-op is a
    // k3s2p1 dw conv. but with current shape infer functions' implementation,
    // we can't access the fusion info inside them. so, still remain the
    // internal dw_type attr to record the post-dw-conv info used to infer
    // correct shape. Here the dw_type attr is a temporary attr only used during
    // shape infer, and will be removed from the op before existing shape infer.
    std::vector<op_ptr> conv_fused_post_s2_dw_conv;
    for (auto &op : sg->get_ops()) {
        fusion_info_t fusion_info;
        if (op->has_attr(op_attr::fusion_info)) {
            fusion_info = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        }

        if (fusion_info.has_post_dw_conv()) {
            const auto &dw_conv = fusion_info.get_post_dw_conv()->get_op();
            const auto &dw_conv_strides
                    = dw_conv->get_attr<std::vector<int64_t>>(op_attr::strides);
            const bool is_k3s2p1 = dw_conv_strides[0] == 2;
            if (is_k3s2p1) {
                conv_fused_post_s2_dw_conv.emplace_back(op);
                op->set_attr<std::string>(op_attr::dw_type, "k3s2p1");
            }
        }
    }

    CHECK(sg->infer_shape());

    // Fill the inferred shape and strides to subgraph's outputs
    for (size_t i = 0; i < sg->outs_.size(); i++) {
        for (auto val : sg->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == sg->outs_[i].id) {
                auto inferred_shape = ltw(lt).vdims();
                set_shape_and_strides(sg->outs_[i], inferred_shape);
            }
        }
    }

    for (auto &op : conv_fused_post_s2_dw_conv) {
        op->remove_attr(op_attr::dw_type);
    }

    return status::success;
}

const std::map<op_kind_t, dnnl::algorithm> &get_binary_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &binary_alg_map
            = {{graph::op_kind::Add, dnnl::algorithm::binary_add},
                    {graph::op_kind::Multiply, dnnl::algorithm::binary_mul},
                    {graph::op_kind::Divide, dnnl::algorithm::binary_div},
                    {graph::op_kind::Minimum, dnnl::algorithm::binary_min},
                    {graph::op_kind::Maximum, dnnl::algorithm::binary_max},
                    {graph::op_kind::Subtract, dnnl::algorithm::binary_sub},
                    {graph::op_kind::BiasAdd, dnnl::algorithm::binary_add},
                    {graph::op_kind::GreaterEqual, dnnl::algorithm::binary_ge},
                    {graph::op_kind::Select, dnnl::algorithm::binary_select}};
    return binary_alg_map;
}

bool binary_doable(
        const std::vector<dim_t> &shape_0, const std::vector<dim_t> &shape_1) {
    const int ndims_0 = static_cast<int>(shape_0.size());
    const int ndims_1 = static_cast<int>(shape_1.size());
    const int small = ndims_0 < ndims_1 ? ndims_0 : ndims_1;
    for (int i = 1; i <= small; ++i) {
        bool match = shape_0[ndims_0 - i] == shape_1[ndims_1 - i]
                || shape_0[ndims_0 - i] == 1 || shape_1[ndims_1 - i] == 1;
        if (!match) return false;
    }
    return true;
}

static bool is_supported_cpu_matmul_bcast(int ndims, int mask) {
    if (ndims == 4) {
        // Supported 4D CPU masks:
        // 0 (scalar), 1 (per_mb), 2 (per_oc), 3 (batch), 8 (per_w),
        // 9 (per_mb_w), 10 (per_hw), 12 (spatial), 13 (per_mb_spatial), 15 (full)
        return dnnl::impl::utils::one_of(mask, 0, 1, 2, 3, 8, 9, 10, 12, 13, 15);
    } else if (ndims == 5) {
        // Supported 5D CPU masks:
        // 1 (per_mb), 7 (batch), 16 (per_w), 17 (per_mb_w), 29 (per_mb_spatial), 31 (full)
        return dnnl::impl::utils::one_of(mask, 0, 1, 2, 7, 16, 17, 29, 31);
    }
    return false;
}

// TODO: ekind can be removed once CPU optimized 5d tensor MatMul with
// broadcasted post op
static bool post_binary_fusible_impl(const op_t *base_op,
        const std::vector<dim_t> &fused_shape,
        const std::vector<dim_t> &other_shape, engine_kind_t ekind) {
    VCHECK_UTILS(fused_shape.size() == other_shape.size(), false,
            "binary fusible ops must have same ndims, "
            "fused_shape size is %zu, other_shape size is %zu."
            "pls run binary_canonicalization pass first",
            fused_shape.size(), other_shape.size());
    // full tensor and per tensor broadcasted
    if (fused_shape == other_shape
            || std::all_of(other_shape.begin(), other_shape.end(),
                    [](dim_t i) { return i == 1; }))
        return true;

    int32_t output_ndims = static_cast<int32_t>(fused_shape.size());

    // any broadcasted for 4d or 5d tensor MatMul
    if (base_op->get_kind() == op_kind::_matmul
            && (output_ndims == 4 || output_ndims == 5)) {
        int mask = 0;
        for (int32_t i = output_ndims - 1; i >= 0; i--) {
            if (other_shape[i] == 1) continue;
            if (fused_shape[i] != other_shape[i]) { return false; }
            mask |= (1 << i);
        }
        if(ekind == dnnl_cpu){
            return is_supported_cpu_matmul_bcast(output_ndims, mask);
        }
        return true;
    }

    // allow fusion for conv + [N,C,1,1] shape post-binary src
    if (base_op->get_kind() == op_kind::_convolution && output_ndims == 4) {
        if (base_op->get_attr<std::string>(op_attr::data_format) == "NCX"
                && other_shape[2] == 1 && other_shape[3] == 1) {
            return true;
        }
        if (base_op->get_attr<std::string>(op_attr::data_format) == "NXC"
                && other_shape[1] == 1 && other_shape[2] == 1) {
            return true;
        }
    }

    // per channel broadcasted
    const auto is_not_one = [](dim_t d) { return d != 1; };
    const auto n_not_broadcastable
            = std::count_if(other_shape.begin(), other_shape.end(), is_not_one);
    if (n_not_broadcastable != 1) return false;
    const auto c_axis_it
            = std::find_if(other_shape.begin(), other_shape.end(), is_not_one);
    const auto c_axis = static_cast<size_t>(
            std::distance(other_shape.begin(), c_axis_it));
    if (other_shape[c_axis] != fused_shape[c_axis]) return false;
    if (base_op->has_attr(op_attr::data_format)) {
        const auto data_fmt
                = base_op->get_attr<std::string>(op_attr::data_format);
        int32_t orig_c_axis = data_fmt == "NCX" ? 1 : output_ndims - 1;
        return c_axis == static_cast<size_t>(orig_c_axis);
    }

    return true;
}

std::pair<bool, std::pair<size_t, int64_t>> shuffle_fusible(
        const op_t *reshape0, op_t *reshape1, op_t *transpose) {
    using result_t = std::pair<bool, std::pair<size_t, int64_t>>;
    const result_t dflt_res {false, {0, 0}};

    const logical_tensor_t src_port = reshape0->get_input_logical_tensor(0);
    const logical_tensor_t dst_port = reshape1->get_output_logical_tensor(0);
    const auto src_lt_shape = ltw(src_port).vdims();
    const auto dst_lt_shape = ltw(dst_port).vdims();
    const auto attr_shape = reshape0->get_attr<dims>(op_attr::shape);
    const auto tp_order = transpose->get_attr<dims>(op_attr::order);

    if (src_lt_shape != dst_lt_shape) return dflt_res;
    if (src_lt_shape.size() + 1 != attr_shape.size()) return dflt_res;

    size_t last_unmatched_pos = tp_order.size();
    size_t matched_pos = 0;
    for (size_t i = 0; i < tp_order.size(); ++i) {
        if (tp_order[i] == static_cast<dim>(i))
            ++matched_pos;
        else
            last_unmatched_pos = i;
    }

    // more or less than two positions were swapped
    if (matched_pos != tp_order.size() - 2) return dflt_res;
    // all positions were matched
    if (last_unmatched_pos == tp_order.size()) return dflt_res;
    // transposition not on consecutive positions
    if (last_unmatched_pos
            != static_cast<size_t>(tp_order[last_unmatched_pos - 1]))
        return dflt_res;

    const size_t g_pos = last_unmatched_pos;
    const size_t c_over_g_pos = g_pos - 1;
    const int64_t groups = attr_shape[g_pos];
    auto mod_attr_shape = attr_shape;
    mod_attr_shape[c_over_g_pos] *= groups;
    mod_attr_shape.erase(mod_attr_shape.begin() + g_pos);

    if (src_lt_shape != mod_attr_shape) return dflt_res;

    return {true, {c_over_g_pos, groups}};
}

bool post_binary_fusible(
        const op_t *base_op, const op_t *bin_op, graph::engine_kind_t ekind) {

// conv + binary post-op fusion is unsupported on NVIDIA GPU
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if ((base_op->get_kind() == op_kind::_convolution) && ekind == dnnl_gpu) {
        return false;
    }
#endif

    auto fused_out = base_op->get_output_values()[0];
    auto consumers = fused_out->get_consumers();
    if (consumers.size() != 1) return false;
    if (consumers[0].get_op().num_inputs() != 2) return false;

    size_t fused_in_off = consumers[0].get_offset();
    auto fused_in = bin_op->get_input_logical_tensor(fused_in_off);
    auto other_in = bin_op->get_input_logical_tensor(1 - fused_in_off);

    // Special check: dnnl_reorder only support fuse non-broadcast binary_add as
    // post-sum
    if (base_op->get_kind() == op_kind::_reorder) {
        if (ltw(fused_in).vdims() != ltw(other_in).vdims()
                || static_cast<dnnl::algorithm>(
                           bin_op->get_attr<int64_t>(op_attr::alg_kind))
                        != dnnl::algorithm::binary_add)
            return false;
    }

    // Special check: dnnl_eltwise only support src and dst datatype are same
    if (base_op->get_kind() == op_kind::_eltwise) {
        auto bin_out = bin_op->get_output_values()[0]->get_logical_tensor();
        if (ltw(fused_in).data_type() != ltw(bin_out).data_type()) return false;
    }

    // Special check: subtract and divide binary post-op can only be fused
    // when base_op is their first input's producer
    if (static_cast<dnnl::algorithm>(
                bin_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::binary_sub
            || static_cast<dnnl::algorithm>(
                       bin_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::binary_div) {
        if (fused_in_off != 0) return false;
    }

    // Disable matmul + binary_sub fusion
    if (base_op->get_kind() == op_kind::_matmul
            && static_cast<dnnl::algorithm>(
                       bin_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::binary_sub) {
        return false;
    }

    return post_binary_fusible_impl(
            base_op, ltw(fused_in).vdims(), ltw(other_in).vdims(), ekind);
}

bool post_eltwise_fusible(
        const op_t *base_op, const op_t *elt_op, graph::engine_kind_t ekind) {
    if (elt_op->has_attr(op_attr::fusion_info)) {
        fusion_info_t fusion_info
                = elt_op->get_attr<fusion_info_t>(op_attr::fusion_info);
        return !fusion_info.with_dropout();
    }

// binary + sqrt post-op fusion is unsupported on NVIDIA GPU
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (base_op->get_kind() == op_kind::_binary
            && static_cast<dnnl::algorithm>(
                       elt_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::eltwise_sqrt
            && ekind == dnnl_gpu) {
        return false;
    }
    return true;
#else
    UNUSED(base_op);
    UNUSED(ekind);
    return true;
#endif
}

bool post_depthwise_conv_fusible(
        const op_t *base_conv_op, const op_t *post_conv_op) {
    using spatial_dims_t = std::vector<int64_t>;
    using oix_dims_t = std::tuple<int64_t, int64_t, spatial_dims_t>;
    const auto extract_dims_as_oix = [](const op_t *op) -> oix_dims_t {
        const size_t wei_offset = 1;
        const auto wei_dims
                = ltw(op->get_input_logical_tensor(wei_offset)).vdims();
        const auto wei_format = (op->has_attr(op_attr::weights_format))
                ? op->get_attr<std::string>(op_attr::weights_format)
                : "XIO";
        const size_t ndims = wei_dims.size();
        const int64_t o
                = (wei_format == "OIX") ? wei_dims[0] : wei_dims[ndims - 1];
        const int64_t i
                = (wei_format == "OIX") ? wei_dims[1] : wei_dims[ndims - 2];
        const auto spatial_dims = (wei_format == "OIX")
                ? spatial_dims_t(wei_dims.begin() + 2, wei_dims.end())
                : spatial_dims_t(wei_dims.begin(), wei_dims.end() - 2);

        return std::make_tuple(o, i, spatial_dims);
    };
    const auto all_equal_to = [](const dims &ds, const int64_t val) -> bool {
        return std::all_of(ds.begin(), ds.end(),
                [val](const int64_t d) { return d == val; });
    };

    spatial_dims_t conv_spatial;
    std::tie(std::ignore, std::ignore, conv_spatial)
            = extract_dims_as_oix(base_conv_op);

    int64_t dw_o = 0;
    int64_t dw_i = 0;
    spatial_dims_t dw_spatial;
    std::tie(dw_o, dw_i, dw_spatial) = extract_dims_as_oix(post_conv_op);

    // only 2D conv is supported
    const size_t expected_spatial_ndims = 2;
    if (conv_spatial.size() != expected_spatial_ndims
            || dw_spatial.size() != expected_spatial_ndims)
        return false;

    // base conv has to be 1x1 conv
    if (!all_equal_to(conv_spatial, 1)) return false;

    // post conv has to be 3x3 conv
    if (!all_equal_to(dw_spatial, 3)) return false;

    // other post conv requirements
    if (post_conv_op->has_attr(op_attr::auto_pad)
            && post_conv_op->get_attr<std::string>(op_attr::auto_pad) != "None")
        return false;
    if (!post_conv_op->has_attr(op_attr::groups)) return false;

    const auto groups = post_conv_op->get_attr<int64_t>(op_attr::groups);
    if (!(groups == dw_o && dw_o == groups * dw_i)) return false;

    const auto strides = post_conv_op->get_attr<dims>(op_attr::strides);
    if (!(all_equal_to(strides, 1) || all_equal_to(strides, 2))) return false;

    const auto pads_begin = post_conv_op->get_attr<dims>(op_attr::pads_begin);
    if (!all_equal_to(pads_begin, 1)) return false;

    const auto pads_end = post_conv_op->get_attr<dims>(op_attr::pads_end);
    if (!(all_equal_to(pads_end, 0) || all_equal_to(pads_end, 1))) return false;

    return true;
}

const std::unordered_map<op_kind_t, std::unordered_set<op_kind_t>> &
get_post_ops_fusible_map() {
    using namespace graph::op_kind;
    static const std::unordered_map<op_kind_t, std::unordered_set<op_kind_t>>
            fusible_map = {
                    {_convolution, {_eltwise, _binary, _convolution}},
                    {_convtranspose, {_eltwise, _binary}},
                    {_matmul, {_eltwise, _binary}},
                    {_pool, {_binary}},
                    {_eltwise, {_binary}},
                    {_binary, {_eltwise, _binary}},
                    // bn
                    {_batchnorm, {_eltwise}},
                    // reduction
                    {_reduction, {_eltwise, _binary}},
                    // resample
                    {_resampling, {_eltwise, _binary}},
                    {_reorder, {_binary}},
                    {_softmax, {_eltwise, _binary}},
                    {_layernorm, {_eltwise, _binary}},
                    {_groupnorm, {_eltwise, _binary}},
            };
    return fusible_map;
}

// data_format = NXC:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = false:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = true:
// (1, 2, 3, 4); (2) is doable

// src      wei
// (3, 4); (3, 4) is doable
// (1, 4); (3, 4) is not doable
// (3, 4); (1, 4) is doable
// (3, 4, 5); (4, 5) is doable
// (3, 4, 5); (1, 5) is doable
// (3, 4, 5); (2, 4, 5) is NOT doable
bool prelu_doable(const std::vector<dim_t> &src_dims,
        const std::vector<dim_t> &wei_dims, const std::string &data_format,
        const bool per_channel_broadcast) {
    const int src_ndims = static_cast<int>(src_dims.size());
    const int wei_ndims = static_cast<int>(wei_dims.size());
    // src ndims should be equal or greater than wei ndims
    if (src_ndims < wei_ndims) return false;

    bool doable = false;
    if (wei_ndims == 1) {
        if (!per_channel_broadcast || src_ndims == wei_ndims) {
            // if no broadcast to channel or src_ndims == 1
            // then wei dim should be equal to last src dim,
            // or equal to 1.
            doable = src_dims[src_ndims - 1] == wei_dims[0] || wei_dims[0] == 1;
        } else {
            // if broadcast to channel,
            // then src channel dim should be equal to wei dim
            const int channel_dim_num
                    = data_format == "NCX" ? 1 : src_ndims - 1;
            doable = src_dims[channel_dim_num] == wei_dims[0];
        }
    } else {
        for (int i = 1; i <= wei_ndims; ++i) {
            // Weights are broadcastable to src when:
            // 1) they are equal on the same ndims,
            // 2) one of them is 1,
            // 3) In the case when weights have fewer dimensions,
            //    1s are added to the front and then 1) and 2) must be met.
            doable = src_dims[src_ndims - i] == wei_dims[wei_ndims - i]
                    || wei_dims[wei_ndims - i] == 1;
            if (!doable) break;
        }
    }
    return doable;
}

bool is_typecast(const op_t *op) {
    bool is_typecast = op->get_kind() == op_kind::_reorder
            && !op->get_attr<bool>(op_attr::change_layout)
            && (!op->has_attr(op_attr::mask)
                    || op->get_attr<int64_t>(op_attr::mask) == 0)
            && !op->has_attr(op_attr::scales) && !op->has_attr(op_attr::src_zps)
            && !op->has_attr(op_attr::dst_zps)
            && (!op->has_attr(op_attr::with_runtime_scales)
                    || !op->get_attr<bool>(op_attr::with_runtime_scales))
            && (!op->has_attr(op_attr::with_runtime_src_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_src_zps))
            && (!op->has_attr(op_attr::with_runtime_dst_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_dst_zps))
            && op->get_input_logical_tensor(0).data_type
                    != op->get_output_logical_tensor(0).data_type;
    return is_typecast;
}

bool with_runtime_zps(const op_ptr &op, bool is_input, size_t index) {
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        return fusion_info.with_runtime_zero_points(is_input, index);
    } else {
        return false;
    }
}

bool with_runtime_scales(const op_ptr &op, bool is_input, size_t index) {
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        return fusion_info.with_runtime_scales(is_input, index);
    } else {
        return false;
    }
}

bool is_layout_reorder(const op_t *op) {
    bool is_layout_reorder = op->get_kind() == op_kind::_reorder
            && op->get_attr<bool>(op_attr::change_layout)
            && (!op->has_attr(op_attr::mask)
                    || op->get_attr<int64_t>(op_attr::mask) == 0)
            && !op->has_attr(op_attr::scales) && !op->has_attr(op_attr::src_zps)
            && !op->has_attr(op_attr::dst_zps)
            && (!op->has_attr(op_attr::with_runtime_scales)
                    || !op->get_attr<bool>(op_attr::with_runtime_scales))
            && (!op->has_attr(op_attr::with_runtime_src_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_src_zps))
            && (!op->has_attr(op_attr::with_runtime_dst_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_dst_zps))
            && op->get_input_logical_tensor(0).data_type
                    == op->get_output_logical_tensor(0).data_type;
    return is_layout_reorder;
}

std::shared_ptr<op_t> clone_mul_scales(const std::shared_ptr<op_t> &scale_op) {
    VCHECK_UTILS(scale_op->num_inputs() <= 1
                    && !scale_op->has_attr(op_attr::with_runtime_scales),
            nullptr,
            "scale_op should be static and have only one input value.");
    auto new_op = std::make_shared<op_t>(op_kind::_mul_scales);
    new_op->set_attr<std::vector<float>>(op_attr::scales,
            scale_op->get_attr<std::vector<float>>(op_attr::scales));
    new_op->set_attr<int64_t>(
            op_attr::mask, scale_op->get_attr<int64_t>(op_attr::mask));
    return new_op;
}

bool inverse_mul_scales(std::shared_ptr<op_t> &scale_op) {
    VCHECK_UTILS(scale_op->num_inputs() <= 1
                    && !scale_op->has_attr(op_attr::with_runtime_scales),
            false, "scale_op should be static and have only one input value.");
    const auto scales = scale_op->get_attr<std::vector<float>>(op_attr::scales);
    const auto scales_inv
            = dnnl_impl::utils::fmap(scales, [](float s) { return 1.f / s; });
    scale_op->set_attr(op_attr::scales, scales_inv);
    return true;
}

bool need_broadcast_for_inputs(
        const std::shared_ptr<op_t> &op, size_t index1, size_t index2) {
    const dims input1_dims = ltw(op->get_input_logical_tensor(index1)).vdims();
    const dims input2_dims = ltw(op->get_input_logical_tensor(index2)).vdims();

    return input1_dims != input2_dims;
}
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
