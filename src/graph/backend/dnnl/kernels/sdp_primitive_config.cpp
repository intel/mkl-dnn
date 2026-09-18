/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/sdp_primitive_config.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#include "common/compiler_workarounds.hpp"

#define VCHECK_SDP_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_primitive_kernel_t, (cond), status, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

op_ptr sdp_primitive_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

status_t sdp_primitive_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    // At least 3 inputs: Q, K, V
    VCHECK_SDP_PRIMITIVE(inputs.size() >= 3, status::invalid_arguments,
            "At least 3 inputs are required");

    for (auto &cur_op : sg->get_ops()) {
        const auto opk = cur_op->get_kind();
        // SDPA with static quantization and dequantization is currently
        // unsupported in the ukernel. Only compressed KV SDPA pattern with
        // dynamic dequantization before K and V is supported.
        VCHECK_SDP_PRIMITIVE(opk != graph::op_kind::Dequantize
                        && opk != graph::op_kind::Quantize,
                status::unimplemented, "Not support quantized SDPA");
    }

    // step1(pattern check): Not support sdpa variants with select as mask
    // We already have a pattern matcher to ensure that the sdpa patterns
    // dispatch to here are knows ones, and we have quant check in sdpa base
    // kernel, so here we only check specific variants based on support matrix.
    const std::unordered_set<graph::op_kind_t> mm1_post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr;
    bool f32_inter = true;

    const bool is_cpu = sg->p_engine_
            && sg->p_engine_->get_kind() == dnnl::engine::kind::cpu;

    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        if (op_kind == graph::op_kind::DynamicDequantize
                && cur_op->get_attr<std::string>(op_attr::qtype)
                        == "per_group") {
            if (!cur_op->has_attr(op_attr::group_shape))
                return status::invalid_arguments;
            is_compressed_ = true;
            const auto &group_shape = cur_op->get_attr<std::vector<int64_t>>(
                    op_attr::group_shape);
            const auto &input_lt = cur_op->get_input_logical_tensor(0);
            if (static_cast<int>(group_shape.size()) != ltw(input_lt).ndims())
                return status::invalid_arguments;
            // TODO(zhitao): execute the reorder for scale and zps mannually if the
            // transpose attribute is specified as true.
            auto post_op = get_post_op(cur_op);
            if (post_op && post_op->get_kind() == graph::op_kind::MatMul
                    && post_op->has_attr(op_attr::transpose_b)
                    && post_op->get_attr<bool>(op_attr::transpose_b))
                return status::unimplemented;
        }
        if (op_kind != graph::op_kind::MatMul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            mm1 = cur_op;
            const auto &lt_score = mm1->get_output_logical_tensor(0);
            f32_inter = f32_inter
                    && (ltw(lt_score).data_type() == data_type::f32);
            // Not support select between mm1 and scale(optional)
            // GPT-J:[mm1] --> [select] --> [scale]* --> [mask]* --> ...
            VCHECK_SDP_PRIMITIVE(post_op->get_kind() != graph::op_kind::Select,
                    status::unimplemented,
                    "Not support select between mm1 and scale(optional)");

            // Track the value produced by the score-processing chain so the
            // Select inputs can be identified as score versus scalar fill.
            auto score_val = mm1->get_output_value(0);

            // scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                // Scale exists, update post_op and traverse to next op
                scale = post_op;
                post_op = get_post_op(post_op);
                const auto &lt_ss = scale->get_output_logical_tensor(0);
                f32_inter = f32_inter
                        && (ltw(lt_ss).data_type() == data_type::f32);
                score_val = scale->get_output_value(0);
            }
            // mask
            if (post_op) {
                if (post_op->get_kind() == graph::op_kind::Add) {
                    // Mask exists, update post_op and traverse to next op
                    const auto mask = post_op;
                    const auto &lt_ms = mask->get_output_logical_tensor(0);
                    f32_inter = f32_inter
                            && (ltw(lt_ms).data_type() == data_type::f32);
                    score_val = mask->get_output_value(0);
                    post_op = get_post_op(post_op);

                    VCHECK_SDP_PRIMITIVE(
                            !(post_op
                                    && post_op->get_kind()
                                            == graph::op_kind::Select),
                            status::unimplemented,
                            "Select after additive mask is not supported");
                }
                // A select mask between the scale/mask and softmax
                // ([mm1] -> [scale]* -> [mask]* -> [select] -> [softmax])
                // is supported on CPU only. The GPU ukernel has no such path;
                // select between mm1 and scale is rejected above.
                // Distill-Bert:[mm1] --> [scale]* --> [mask]* --> [select] --> ...
                if (post_op && post_op->get_kind() == graph::op_kind::Select) {
                    VCHECK_SDP_PRIMITIVE(is_cpu, status::unimplemented,
                            "Not support select after scale(optional) and "
                            "mask(optional)");

                    // The score enters the select as one branch (then/else);
                    // the other branch is the fill and must be a scalar.
                    const bool then_is_score
                            = post_op->get_input_value(1) == score_val;

                    const auto &fill_lt = then_is_score
                            ? post_op->get_input_logical_tensor(2)
                            : post_op->get_input_logical_tensor(1);

                    VCHECK_SDP_PRIMITIVE(ltw(fill_lt).nelems() == 1,
                            status::unimplemented,
                            "Select mask requires a scalar fill value");

                    post_op = get_post_op(post_op);
                }
            }

            if (post_op) {
                if (post_op->get_kind() == graph::op_kind::SoftMax) {
                    const auto &softmax = post_op;
                    softmax_mode_
                            = softmax->get_attr<std::string>(op_attr::mode);
                }
            }
        } else {
            mm2 = cur_op;
        }
    }

    VCHECK_SDP_PRIMITIVE(f32_inter, status::invalid_graph,
            "only supports f32 intermediates.");

    auto find_graph_inport = [&inputs](const std::shared_ptr<value_t> &val) {
        auto tmp_val = val;
        while (tmp_val->has_producer()) {
            const op_t &prod_op = tmp_val->get_producer();
            tmp_val = prod_op.get_input_value(0);
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (tmp_val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };

    VCHECK_SDP_PRIMITIVE(
            mm1 && mm2, status::invalid_graph, "mm1 or mm2 is not found");

    int q_id = find_graph_inport(mm1->get_input_value(0));
    int k_id = find_graph_inport(mm1->get_input_value(1));
    int v_id = find_graph_inport(mm2->get_input_value(1));

    VCHECK_SDP_PRIMITIVE(q_id != -1 && k_id != -1 && v_id != -1,
            status::unimplemented, "Q, K, V are not found");

    // sdp_primitive only supports single scale value.
    if (scale) {
        const auto &s = scale->get_input_logical_tensor(1);
        VCHECK_SDP_PRIMITIVE(ltw(s).nelems() == 1, status::unimplemented,
                "Scale should be single value");
    }

    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
