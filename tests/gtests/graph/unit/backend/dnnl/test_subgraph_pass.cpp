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
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "backend/dnnl/kernels/large_partition.hpp"
#include "backend/dnnl/op_executable.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/transform.hpp"
#include "backend/dnnl/subgraph.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

using namespace dnnl::impl::graph;
using namespace dnnl::impl::graph::op_kind;
using namespace dnnl::graph::tests::unit::utils;

namespace graph = dnnl::impl::graph;

using op_ptr = std::shared_ptr<dnnl::impl::graph::op_t>;

namespace {
dnnl::impl::graph::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::impl::graph::dnnl_impl::dnnl_backend_t::get_singleton();
    auto pm = dnnl::impl::graph::pass::pass_manager_t(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::impl::graph::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

    return *find;
}
} // namespace

TEST(test_subgraph_pass, Kind2Str) {
    ASSERT_EQ(op_t::kind2str(graph::op_kind::Abs), "Abs");
    ASSERT_EQ(op_t::kind2str(graph::op_kind::_add_zps), "_add_zps");
}

TEST(test_subgraph_pass, MatmulCanonicalizationPreservesScaleMask) {
    auto matmul = std::make_shared<op_t>(0, op_kind::_matmul, "matmul");
    matmul->set_attr<bool>(op_attr::transpose_a, false);
    matmul->set_attr<bool>(op_attr::transpose_b, true);

    logical_tensor_t src
            = logical_tensor_init(0, {2, 4, 8, 16}, graph::data_type::f32);
    logical_tensor_t wei
            = logical_tensor_init(1, {2, 4, 32, 16}, graph::data_type::f32);
    logical_tensor_t dst
            = logical_tensor_init(2, {2, 4, 8, 32}, graph::data_type::f32);
    matmul->add_input(src);
    matmul->add_input(wei);
    matmul->add_output(dst);

    auto scales_op = std::make_shared<op_t>(op_kind::_mul_scales);
    scales_op->set_attr<int64_t>(op_attr::mask, 3);
    dnnl_impl::fusion_info_t fusion_info;
    fusion_info.set_runtime_scales(scales_op, true, 1);
    matmul->set_attr<dnnl_impl::fusion_info_t>(
            op_attr::fusion_info, fusion_info);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<op_ptr> {matmul});

    ASSERT_EQ(dnnl_impl::insert_permute_for_matmul(subgraph), status::success);
    auto matmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const op_ptr &op) {
        return op->get_kind() == op_kind::_matmul;
    });
    ASSERT_NE(matmul_op, subgraph->get_ops().end());
    auto get_weight_scale_mask = [&]() {
        auto info = (*matmul_op)
                            ->get_attr<dnnl_impl::fusion_info_t>(
                                    op_attr::fusion_info);
        return info.get_mutable_scales(true, 1)->get_attr<int64_t>(
                op_attr::mask);
    };
    EXPECT_EQ(get_weight_scale_mask(), 3);

    ASSERT_EQ(dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph),
            status::success);
    EXPECT_EQ(get_weight_scale_mask(), 3);
}

TEST(test_subgraph_pass, MatmulUnsqueezeShiftsScaleMask) {
    auto matmul = std::make_shared<op_t>(0, op_kind::_matmul, "matmul");
    matmul->add_input(
            logical_tensor_init(0, {2, 4, 8, 16}, graph::data_type::f32));
    matmul->add_input(
            logical_tensor_init(1, {4, 16, 32}, graph::data_type::f32));
    matmul->add_output(
            logical_tensor_init(2, {2, 4, 8, 32}, graph::data_type::f32));

    auto scales_op = std::make_shared<op_t>(op_kind::_mul_scales);
    scales_op->set_attr<int64_t>(op_attr::mask, 3);
    dnnl_impl::fusion_info_t fusion_info;
    fusion_info.set_runtime_scales(scales_op, true, 1);
    matmul->set_attr<dnnl_impl::fusion_info_t>(
            op_attr::fusion_info, fusion_info);
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<op_ptr> {matmul});

    ASSERT_EQ(dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph),
            status::success);
    auto info
            = matmul->get_attr<dnnl_impl::fusion_info_t>(op_attr::fusion_info);
    EXPECT_EQ(
            info.get_mutable_scales(true, 1)->get_attr<int64_t>(op_attr::mask),
            6);
}

TEST(test_subgraph_pass, LargerPartitionKernelCreator) {
    ASSERT_NO_THROW(graph::dnnl_impl::large_partition_kernel_creator());
}

TEST(test_subgraph_pass, LowerDownToInt8Conv) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            sum
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    const auto engine_kind = get_test_engine_kind();
    graph_t agraph(engine_kind);
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.1f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t conv {2, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t dequant_other {3, Dequantize, "dequant"};
    dequant_other.set_attr(op_attr::scales, scales);
    dequant_other.set_attr(op_attr::zps, zps);
    op_t sum {4, Add, "sum"};
    op_t relu {5, ReLU, "relu"};
    op_t quant {6, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(5, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t s8_other = logical_tensor_init(6, data_type::u8);
    logical_tensor_t fp32_other = logical_tensor_init(7, data_type::f32);
    dequant_other.add_input(s8_other);
    dequant_other.add_output(fp32_other);

    logical_tensor_t fp32_sum_out = logical_tensor_init(8, data_type::f32);
    sum.add_input(fp32_conv_out);
    sum.add_input(fp32_other);
    sum.add_output(fp32_sum_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(9, data_type::f32);
    relu.add_input(fp32_sum_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(10, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), status::success);
    ASSERT_EQ(agraph.add_op(&sum), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    graph::pass::pass_base_ptr apass
            = get_pass(engine_kind == graph::engine_kind::gpu
                            ? "x8x8x8_conv_add_post_ops_gpu"
                            : "x8x8x8_conv_add_post_ops_cpu");
    ASSERT_NE(apass, nullptr);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::quantized_convolution_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());
    ASSERT_EQ(subgraph->get_ops().size(), 7U);

    // lower the binary ops in oneDNN Graph (Add, Mul, ...) to DNNL backend
    // internal ops and canonicalize them
    int8_data = logical_tensor_init(0, {1, 112, 112, 8}, data_type::u8);
    s8_weight = logical_tensor_init(2, {3, 3, 8, 8}, data_type::s8);
    fp32_bias = logical_tensor_init(4, {8}, data_type::f32);
    s8_other = logical_tensor_init(6, {1, 110, 110, 8}, data_type::u8);

    dnnl_impl::set_given_inputs_outputs(
            subgraph, {int8_data, s8_weight, fp32_bias, s8_other}, {int8_out});

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    ASSERT_EQ(subgraph->get_ops().size(), 11U);
    auto conv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_convolution;
    });
    ASSERT_NE(conv_op, subgraph->get_ops().end());
    auto &producer0 = (*conv_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), op_kind::_mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);
    auto &producer1 = (*conv_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), op_kind::_mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);

    // 2. fuse src, wei scales
    dnnl_impl::convert_to_runtime_src_scales(subgraph);
    dnnl_impl::fuse_src_scales(subgraph);
    dnnl_impl::infer_shape(subgraph);
    dnnl_impl::binary_canonicalization(subgraph);
    dnnl_impl::infer_shape(subgraph);

    // 3. fuse post ops to int8 conv
    ASSERT_EQ(dnnl_impl::fuse_post_ops(subgraph), status::success);

    auto qconv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_convolution;
    });
    ASSERT_NE(qconv_op, subgraph->get_ops().end());
    ASSERT_TRUE((*qconv_op)->has_attr(graph::op_attr::fusion_info));
    const auto &fusion_info = (*qconv_op)->get_attr<dnnl_impl::fusion_info_t>(
            graph::op_attr::fusion_info);
    const auto &post_ops = fusion_info.get_post_ops();
    ASSERT_EQ(post_ops.size(), 2U);
}

TEST(test_subgraph_pass, LowerDownToInt8Matmul) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            matmul
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    const auto engine_kind = get_test_engine_kind();
    graph_t agraph(engine_kind);
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.5f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, false);
    matmul.set_attr<bool>(op_attr::transpose_b, false);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(5, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(6, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
    ASSERT_NE(apass, nullptr);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());
    ASSERT_EQ(subgraph->get_ops().size(), 5U);

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    ASSERT_EQ(subgraph->get_ops().size(), 8U);
    auto matmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_matmul;
    });
    ASSERT_NE(matmul_op, subgraph->get_ops().end());
    auto &producer0 = (*matmul_op)->get_input_value(0)->get_producer();
    ASSERT_EQ(producer0.get_kind(), op_kind::_mul_scales);
    ASSERT_EQ(producer0.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);
    auto &producer1 = (*matmul_op)->get_input_value(1)->get_producer();
    ASSERT_EQ(producer1.get_kind(), op_kind::_mul_scales);
    ASSERT_EQ(producer1.get_attr<std::vector<float>>(op_attr::scales)[0],
            scales[0]);

    // 2. fuse src, wei scales
    dnnl_impl::convert_to_runtime_src_scales(subgraph);
    dnnl_impl::fuse_src_scales(subgraph);
    auto qmatmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_matmul;
    });

    // 3. fuse post ops to int8 matmul
    ASSERT_EQ(dnnl_impl::fuse_post_ops(subgraph), status::success);

    qmatmul_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_matmul;
    });
    ASSERT_NE(qmatmul_op, subgraph->get_ops().end());
    ASSERT_TRUE((*qmatmul_op)->has_attr(graph::op_attr::fusion_info));
    const auto &fusion_info = (*qmatmul_op)
                                      ->get_attr<dnnl_impl::fusion_info_t>(
                                              graph::op_attr::fusion_info);
    const auto &post_ops = fusion_info.get_post_ops();
    ASSERT_EQ(post_ops.size(), 1U);
}

TEST(test_subgraph_pass, Conv2dNxcPlainDst) {
    using dims = graph::dnnl_impl::dims;
    graph::engine_t *engine = get_engine();
    dnnl::engine p_eng
            = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*engine);

    int64_t in_channel = 512, out_channel = 2048;
    int64_t kernel_size = 1;
    std::vector<int64_t> src_shape {1024, 7, 7, in_channel};
    std::vector<int64_t> weight_shape {
            kernel_size, kernel_size, in_channel, out_channel};
    std::vector<int64_t> dst_shape {1024, 7, 7, out_channel};

    graph::op_t dqdata_node(0, graph::op_kind::Dequantize, "dqdata_node");
    dqdata_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqdata_node.set_attr<std::vector<int64_t>>(op_attr::zps, {0});
    dqdata_node.set_attr<std::vector<float>>(op_attr::scales, {1 / 255.f});
    dqdata_node.set_attr<int64_t>(op_attr::axis, 0);
    graph::op_t dqweight_node(1, graph::op_kind::Dequantize, "dqweight_node");
    dqweight_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqweight_node.set_attr<std::vector<int64_t>>(op_attr::zps, {0});
    dqweight_node.set_attr<std::vector<float>>(op_attr::scales, {1 / 255.f});
    dqweight_node.set_attr<int64_t>(op_attr::axis, 0);
    graph::op_t conv_node(2, graph::op_kind::Convolution, "conv_node");
    conv_node.set_attr<dims>(op_attr::strides, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::dilations, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::pads_begin, dims(2, 0));
    conv_node.set_attr<dims>(op_attr::pads_end, dims(2, 0));
    conv_node.set_attr<int64_t>(op_attr::groups, (int64_t)1);
    conv_node.set_attr<std::string>(op_attr::data_format, "NXC");
    conv_node.set_attr<std::string>(op_attr::weights_format, "XIO");
    graph::op_t qout_node(3, graph::op_kind::Quantize, "qout_node");
    qout_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qout_node.set_attr<std::vector<int64_t>>(op_attr::zps, {0});
    qout_node.set_attr<std::vector<float>>(op_attr::scales, {1 / 255.f});
    qout_node.set_attr<int64_t>(op_attr::axis, 0);

    // prepare logical tensor
    auto src_u8 = logical_tensor_init(0, src_shape, graph::data_type::u8);
    auto src_f32_dq = logical_tensor_init(1, src_shape, graph::data_type::f32);
    auto weight_s8 = logical_tensor_init(2, weight_shape, graph::data_type::s8);
    auto weight_f32_dq
            = logical_tensor_init(3, weight_shape, graph::data_type::f32);
    auto dst_f32 = logical_tensor_init(4, dst_shape, graph::data_type::f32);
    auto dst_u8 = logical_tensor_init(5, dst_shape, graph::data_type::u8);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);
    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);
    conv_node.add_input(src_f32_dq);
    conv_node.add_input(weight_f32_dq);
    conv_node.add_output(dst_f32);
    qout_node.add_input(dst_f32);
    qout_node.add_output(dst_u8);

    graph::graph_t agraph(engine->kind());
    agraph.add_op(&dqdata_node);
    agraph.add_op(&dqweight_node);
    agraph.add_op(&conv_node);
    agraph.add_op(&qout_node);
    agraph.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);

    std::vector<graph::logical_tensor_t> lt_ins {src_u8, weight_s8};
    std::vector<graph::logical_tensor_t> lt_outs {dst_u8};
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng, graph_attr, true,
            true);
    ASSERT_EQ(subgraph->get_ops().size(), 4U);

    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(subgraph, lt_ins, lt_outs),
            graph::status::success);
    ASSERT_EQ(dnnl_impl::lower_down(subgraph), graph::status::success);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param
    ASSERT_EQ(dnnl_impl::infer_shape(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::remove_quant_data_with_no_effect(subgraph),
            graph::status::success);
    ASSERT_EQ(dnnl_impl::convert_to_runtime_src_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(dnnl_impl::fuse_src_scales(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::convert_to_runtime_src_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(
            dnnl_impl::fuse_src_zero_points(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::convert_to_runtime_dst_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(dnnl_impl::fuse_dst_scales(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::convert_to_runtime_dst_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(
            dnnl_impl::fuse_dst_zero_points(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::insert_permute_for_conv_or_deconv(subgraph),
            graph::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::infer_shape(subgraph), graph::status::success);

    for (const auto &op : subgraph->get_ops()) {
        ASSERT_TRUE(dnnl::impl::utils::one_of(op->get_kind(), op_kind::_permute,
                op_kind::_reorder, op_kind::_convolution, op_kind::_mul_scales,
                op_kind::_constant_scales, op_kind::_constant_zps));
    }

    for (auto &cur_op : subgraph->get_ops()) {
        if (cur_op->get_kind() == op_kind::_convolution) {
            const auto &dst_lt
                    = cur_op->get_output_value(0)->get_logical_tensor();
            const auto &mdesc = dnnl_impl::make_dnnl_memory_desc(dst_lt);
            ASSERT_TRUE(dnnl_impl::is_plain(mdesc));
        }
    }
}

TEST(test_subgraph_pass, Int8ConvSumRelu) {
    /*
                   | (f32, constant)
                 quant
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            conv
             | (f32)
            sum
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    SKIP_IF_NV_GPU("not supported on NVIDIA GPU");
    using dims = graph::dnnl_impl::dims;

    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    int64_t groups = 4;

    int64_t in_channel = 8, out_channel = 8;
    int64_t kernel_size = 3;
    std::vector<int64_t> src_shape {1, 112, 112, in_channel};
    std::vector<int64_t> weight_shape {
            kernel_size, kernel_size, in_channel / groups, out_channel};
    std::vector<int64_t> bias_shape {out_channel};
    std::vector<int64_t> dst_shape {1, 110, 110, out_channel};

    float scale_src = 1 / 255.f;
    float scale_other = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_other = 0;
    int64_t zp_out = 78;
    std::vector<float> scale_wei(1, 1 / 127.f);
    std::vector<int64_t> zp_wei(1, 0);

    graph::op_t dqdata_node(1, graph::op_kind::Dequantize, "dqdata_node");
    dqdata_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqdata_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_src});
    dqdata_node.set_attr<std::vector<float>>(op_attr::scales, {scale_src});
    dqdata_node.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t qweight_node(10, graph::op_kind::Quantize, "qweight_node");
    qweight_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qweight_node.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    qweight_node.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    qweight_node.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t dqweight_node(3, graph::op_kind::Dequantize, "dqweight_node");
    dqweight_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqweight_node.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    dqweight_node.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    dqweight_node.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t conv_node(4, graph::op_kind::Convolution, "conv_node");
    conv_node.set_attr<dims>(op_attr::strides, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::dilations, dims(2, 1));
    conv_node.set_attr<dims>(op_attr::pads_begin, dims(2, 0));
    conv_node.set_attr<dims>(op_attr::pads_end, dims(2, 0));
    conv_node.set_attr<int64_t>(op_attr::groups, groups);
    conv_node.set_attr<std::string>(op_attr::data_format, "NXC");
    conv_node.set_attr<std::string>(op_attr::weights_format, "XIO");

    graph::op_t relu_node(5, graph::op_kind::ReLU, "relu_node");

    graph::op_t qout_node(6, graph::op_kind::Quantize, "qout_node");
    qout_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qout_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_out});
    qout_node.set_attr<std::vector<float>>(op_attr::scales, {scale_out});
    qout_node.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t dqother_node(8, graph::op_kind::Dequantize, "dqother_node");
    dqother_node.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqother_node.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_other});
    dqother_node.set_attr<std::vector<float>>(op_attr::scales, {scale_other});
    dqother_node.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t add_node(9, graph::op_kind::Add, "add_node");

    logical_tensor_t src_u8 = logical_tensor_init(1, graph::data_type::u8);
    logical_tensor_t src_f32_dq = logical_tensor_init(2, graph::data_type::f32);
    logical_tensor_t weight_f32
            = logical_tensor_init(20, graph::data_type::f32);
    logical_tensor_t weight_s8 = logical_tensor_init(4, graph::data_type::s8);
    logical_tensor_t weight_f32_dq
            = logical_tensor_init(5, graph::data_type::f32);
    logical_tensor_t dst_f32 = logical_tensor_init(7, graph::data_type::f32);
    logical_tensor_t dst_relu_f32
            = logical_tensor_init(8, graph::data_type::f32);
    logical_tensor_t dst_s8 = logical_tensor_init(9, graph::data_type::s8);
    logical_tensor_t other_s8 = logical_tensor_init(11, graph::data_type::s8);
    logical_tensor_t other_f32_dq
            = logical_tensor_init(12, graph::data_type::f32);
    logical_tensor_t dst_add_f32
            = logical_tensor_init(13, graph::data_type::f32);
    logical_tensor_t bias_f32 = logical_tensor_init(6, graph::data_type::f32);

    dqdata_node.add_input(src_u8);
    dqdata_node.add_output(src_f32_dq);

    qweight_node.add_input(weight_f32);
    qweight_node.add_output(weight_s8);

    dqweight_node.add_input(weight_s8);
    dqweight_node.add_output(weight_f32_dq);

    conv_node.add_input(src_f32_dq);
    conv_node.add_input(weight_f32_dq);
    conv_node.add_input(bias_f32);
    conv_node.add_output(dst_f32);

    dqother_node.add_input(other_s8);
    dqother_node.add_output(other_f32_dq);

    add_node.add_input(dst_f32);
    add_node.add_input(other_f32_dq);
    add_node.add_output(dst_add_f32);

    relu_node.add_input(dst_add_f32);
    relu_node.add_output(dst_relu_f32);

    qout_node.add_input(dst_relu_f32);
    qout_node.add_output(dst_s8);

    const auto engine_kind = p_eng.get()->kind();
    graph::graph_t g(engine_kind);
    g.add_op(&dqdata_node);
    g.add_op(&qweight_node);
    g.add_op(&dqweight_node);
    g.add_op(&conv_node);
    g.add_op(&dqother_node);
    g.add_op(&add_node);
    g.add_op(&relu_node);
    g.add_op(&qout_node);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = get_pass(engine_kind == graph::engine_kind::gpu
                            ? "x8x8x8_conv_add_post_ops_gpu"
                            : "x8x8x8_conv_add_post_ops_cpu");
    ASSERT_NE(apass, nullptr);

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    src_u8 = logical_tensor_init(1, src_shape, graph::data_type::u8);
    weight_f32 = logical_tensor_init(20, weight_shape, graph::data_type::f32);
    bias_f32 = logical_tensor_init(6, bias_shape, graph::data_type::f32);
    other_s8 = logical_tensor_init(11, dst_shape, graph::data_type::s8);
    dst_s8 = logical_tensor_init(9, dst_shape, graph::data_type::s8);

    weight_f32.property = graph::property_type::constant;
    bias_f32.property = graph::property_type::constant;
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, graph_attr, false, true);

    std::vector<logical_tensor_t> inputs
            = {src_u8, weight_f32, bias_f32, other_s8};
    std::vector<logical_tensor_t> outputs = {dst_s8};

    std::vector<logical_tensor_t> wrong_inputs = {src_u8, weight_s8, bias_f32};
    std::vector<logical_tensor_t> wrong_outputs = {};

    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, wrong_inputs, outputs),
            status::invalid_arguments);
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(
                      subgraph, inputs, wrong_outputs),
            status::invalid_arguments);

    // output shape is not must
    ASSERT_EQ(dnnl_impl::set_given_inputs_outputs(subgraph, inputs,
                      {logical_tensor_init(9, graph::data_type::s8,
                              graph::layout_type::any)}),
            status::success);

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::infer_shape(subgraph);
    dnnl_impl::binary_canonicalization(subgraph);
    dnnl_impl::infer_shape(subgraph);

    // run lower down passes
    dnnl_impl::check_with_bias(subgraph);
    dnnl_impl::convert_to_runtime_src_scales(subgraph);
    dnnl_impl::fuse_src_scales(subgraph);
    dnnl_impl::convert_to_runtime_src_zero_points(subgraph);
    dnnl_impl::fuse_src_zero_points(subgraph);
    dnnl_impl::fuse_post_ops(subgraph);
    dnnl_impl::convert_to_runtime_dst_scales(subgraph);
    dnnl_impl::fuse_dst_scales(subgraph);
    dnnl_impl::convert_to_runtime_dst_zero_points(subgraph);
    dnnl_impl::fuse_dst_zero_points(subgraph);
    dnnl_impl::convert_runtime_mul_scales(subgraph);
    dnnl_impl::convert_runtime_zero_points(subgraph);
    dnnl_impl::fuse_dynamic_sub_zps_mul_scales(subgraph);
    dnnl_impl::fuse_dynamic_mul_scales_add_zps(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 8U);
    for (const auto &op : subgraph->get_ops()) {
        ASSERT_TRUE(dnnl::impl::utils::one_of(op->get_kind(), op_kind::_reorder,
                op_kind::_convolution, op_kind::_mul_scales,
                op_kind::_constant_scales, op_kind::_constant_zps));
    }

    // insert preprocess and reorder ops
    dnnl_impl::insert_permute_for_conv_or_deconv(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 12U);

    dnnl_impl::insert_to_group_for_conv_or_deconv(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 13U);

    // infer shape/type, layout propagation and memory binding
    ASSERT_EQ(subgraph->infer_shape(), graph::status::success);

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);

    // since we insert Reorder ops during layout propagation, here need
    // do shape inference and type inference again
    ASSERT_EQ(subgraph->infer_shape(), graph::status::success);

    for (auto &cur_op : subgraph->get_ops()) {
        for (auto &val : cur_op->get_input_values()) {
            auto lt = val->get_logical_tensor();
            graph::logical_tensor_wrapper_t ltw(lt);
            ASSERT_FALSE(ltw.is_shape_unknown());
            ASSERT_NE(ltw.layout_type(), layout_type::undef);
            ASSERT_NE(ltw.layout_type(), layout_type::any);
            //     ASSERT_NE(ltw.data_type(), data_type::undef);
        }

        auto lt = cur_op->get_output_value(0)->get_logical_tensor();
        graph::logical_tensor_wrapper_t ltw(lt);

        ASSERT_FALSE(ltw.is_shape_unknown());
        // inserted reorder's logical tensor is not set to new data type
        // since didn't do type inference after layout propagation
        // ASSERT_NE(ltw.data_type(), data_type::undef);
        ASSERT_NE(ltw.layout_type(), layout_type::undef);
        ASSERT_NE(ltw.layout_type(), layout_type::any);
    }

    dnnl_impl::constant_propagation(subgraph);

    dnnl_impl::memory_planner_t memory_planner;
    ASSERT_EQ(memory_planner.run(subgraph), graph::status::success);

    ASSERT_GE(memory_planner.total_internal_persistent_size(), 0U);
    ASSERT_GE(memory_planner.total_internal_temporary_size(), 0U);

    // only the final weight, bias and constant scales/zps used by conv are
    // cached
    auto cached_mem_offkeys = memory_planner.get_exec_args_set()
                                      .get_mems_use_internal_persistent();
    std::set<size_t> unique_offkeys;
    for (auto &mem_offkey : cached_mem_offkeys) {
        unique_offkeys.insert(mem_offkey.second);
    }
    ASSERT_EQ(unique_offkeys.size(), 5U);

    std::vector<graph::op_t *> topo_ordered_ops;
    dnnl::impl::graph::topo_order_visit(
            subgraph->get_output_ops(), [&](graph::op_t *op) {
        topo_ordered_ops.emplace_back(op);
        return status::success;
    });

    auto topo_ordered_args = memory_planner.get_exec_args_set().get_exec_args();

    ASSERT_EQ(topo_ordered_ops.size(), topo_ordered_args.size());

    for (size_t i = 0; i < topo_ordered_args.size(); i++) {
        std::unordered_map<int, dnnl::memory> exec_arg = topo_ordered_args[i];
        ASSERT_FALSE(exec_arg.empty());

        auto cur_op = topo_ordered_ops[i];
        if (cur_op->get_kind() == op_kind::_convolution) {
            ASSERT_NE(exec_arg.find(DNNL_ARG_SRC), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_WEIGHTS), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_BIAS), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC),
                    exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS),
                    exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST),
                    exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST),
                    exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_GRAPH_ARG_POST_SRC), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_DST), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_SCRATCHPAD), exec_arg.end());
        } else if (cur_op->get_kind() == op_kind::_reorder) {
            ASSERT_NE(exec_arg.find(DNNL_ARG_FROM), exec_arg.end());
            ASSERT_NE(exec_arg.find(DNNL_ARG_TO), exec_arg.end());
        } else {
        }
    }
}

struct matmul_params_t {
    std::vector<graph::dim_t> src_shape;
    std::vector<graph::dim_t> weight_shape;
    std::vector<graph::dim_t> bias_shape;
    std::vector<graph::dim_t> dst_shape;
    bool transpose_a;
    bool transpose_b;
    bool constant_weight;
    size_t subgraph_size_after_insertion;
    size_t final_subgraph_size;
};

class int8_matmul_with_diff_inputs_t
    : public ::testing::TestWithParam<matmul_params_t> {};

TEST_P(int8_matmul_with_diff_inputs_t, Int8MatmulPasses) {
    /*
        | (u8/s8)  | (s8)
     dequant    dequant
    (f32) \     / (f32)
            matmul
             | (f32)
            relu
             | (f32)
           quant
             | (u8/s8)
    */
    const auto &params = GetParam();

    const auto engine_kind = get_test_engine_kind();
    graph_t agraph(engine_kind);
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    std::vector<int64_t> zps {0};
    std::vector<float> scales {0.5f};
    op_t dequant1 {0, Dequantize, "dequant"};
    dequant1.set_attr(op_attr::scales, scales);
    dequant1.set_attr(op_attr::zps, zps);
    op_t dequant2 {1, Dequantize, "dequant"};
    dequant2.set_attr(op_attr::scales, scales);
    dequant2.set_attr(op_attr::zps, zps);
    op_t matmul {2, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, params.transpose_a);
    matmul.set_attr<bool>(op_attr::transpose_b, params.transpose_b);
    op_t relu {3, ReLU, "relu"};
    op_t quant {4, Quantize, "quant"};
    quant.set_attr(op_attr::scales, scales);
    quant.set_attr(op_attr::zps, zps);
    logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
    logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
    dequant1.add_input(int8_data);
    dequant1.add_output(fp32_data);

    logical_tensor_t s8_weight = logical_tensor_init(2, data_type::s8);
    logical_tensor_t fp32_weight = logical_tensor_init(3, data_type::f32);
    dequant2.add_input(s8_weight);
    dequant2.add_output(fp32_weight);

    logical_tensor_t fp32_bias = logical_tensor_init(4, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(5, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(6, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    logical_tensor_t int8_out = logical_tensor_init(7, data_type::u8);
    quant.add_input(fp32_relu_out);
    quant.add_output(int8_out);

    ASSERT_EQ(agraph.add_op(&dequant1), status::success);
    ASSERT_EQ(agraph.add_op(&dequant2), status::success);
    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&quant), status::success);

    agraph.finalize();

    graph::pass::pass_base_ptr apass = get_pass("x8x8x_matmul_post_ops");
    ASSERT_NE(apass, nullptr);
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::quantized_matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng, graph_attr, true,
            true);
    ASSERT_EQ(subgraph->get_ops().size(), 5U);

    dnnl_impl::check_with_bias(subgraph);

    int8_data = logical_tensor_init(0, params.src_shape, graph::data_type::u8);
    s8_weight
            = logical_tensor_init(2, params.weight_shape, graph::data_type::s8);
    fp32_bias
            = logical_tensor_init(4, params.bias_shape, graph::data_type::f32);
    int8_out = logical_tensor_init(7, params.dst_shape, graph::data_type::u8);

    std::vector<logical_tensor_t> inputs = {int8_data, s8_weight, fp32_bias};
    std::vector<logical_tensor_t> outputs = {int8_out};

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::convert_to_runtime_src_scales(subgraph);
    dnnl_impl::fuse_src_scales(subgraph);
    dnnl_impl::convert_to_runtime_src_zero_points(subgraph);
    dnnl_impl::fuse_src_zero_points(subgraph);
    dnnl_impl::fuse_post_ops(subgraph);
    dnnl_impl::convert_to_runtime_dst_scales(subgraph);
    dnnl_impl::fuse_dst_scales(subgraph);
    dnnl_impl::convert_to_runtime_dst_zero_points(subgraph);
    dnnl_impl::fuse_dst_zero_points(subgraph);
    dnnl_impl::convert_runtime_mul_scales(subgraph);
    dnnl_impl::convert_runtime_zero_points(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), 5U);

    subgraph->infer_shape();
    dnnl_impl::insert_permute_for_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), params.subgraph_size_after_insertion);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        if (lt.id == std::numeric_limits<size_t>::max()) continue;
        ASSERT_FALSE(graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    ASSERT_EQ(subgraph->infer_shape(), graph::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);
    ASSERT_EQ(subgraph->get_ops().size(), params.final_subgraph_size);
}

INSTANTIATE_TEST_SUITE_P(test_subgraph_pass, int8_matmul_with_diff_inputs_t,
        testing::Values(matmul_params_t {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 7, 7},
                matmul_params_t {{1, 1024}, {1000, 1024}, {1000}, {1, 1000},
                        false, true, true, 7, 7},
                matmul_params_t {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, false, 8, 8},
                matmul_params_t {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, true, 8, 8}));

class matmul_with_diff_inputs_t
    : public ::testing::TestWithParam<matmul_params_t> {};

TEST_P(matmul_with_diff_inputs_t, MatmulPasses) {
    /*
    (f32) \     / (f32)
            matmul
             | (f32)
            relu
             | (f32)
    */
    const auto &params = GetParam();

    graph_t agraph;
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    op_t matmul {0, MatMul, "matmul"};
    matmul.set_attr<bool>(op_attr::transpose_a, params.transpose_a);
    matmul.set_attr<bool>(op_attr::transpose_b, params.transpose_b);
    op_t relu {1, ReLU, "relu"};

    logical_tensor_t fp32_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t fp32_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t fp32_bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t fp32_matmul_out = logical_tensor_init(3, data_type::f32);
    matmul.add_input(fp32_data);
    matmul.add_input(fp32_weight);
    matmul.add_input(fp32_bias);
    matmul.add_output(fp32_matmul_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(4, data_type::f32);
    relu.add_input(fp32_matmul_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&matmul), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("fp_matmul_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::matmul_post_ops);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng, graph_attr, true,
            true);
    ASSERT_EQ(subgraph->get_ops().size(), 2U);

    dnnl_impl::check_with_bias(subgraph);

    fp32_data = logical_tensor_init(0, params.src_shape, graph::data_type::f32);
    fp32_weight = logical_tensor_init(
            1, params.weight_shape, graph::data_type::f32);
    fp32_bias
            = logical_tensor_init(2, params.bias_shape, graph::data_type::f32);
    fp32_relu_out
            = logical_tensor_init(4, params.dst_shape, graph::data_type::f32);

    std::vector<logical_tensor_t> inputs = {fp32_data, fp32_weight, fp32_bias};
    std::vector<logical_tensor_t> outputs = {fp32_relu_out};

    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    validator.run(subgraph); // validate and set default param

    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);
    subgraph->infer_shape();
    dnnl_impl::insert_permute_for_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_reshape_for_ndx2d_matmul(subgraph);
    subgraph->infer_shape();
    dnnl_impl::insert_unsqueeze_and_squeeze_for_matmul(subgraph);
    ASSERT_EQ(subgraph->get_ops().size(), params.subgraph_size_after_insertion);

    for (auto &val : subgraph->get_input_values()) {
        auto lt = val->get_logical_tensor();
        ASSERT_FALSE(logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    for (auto &val : subgraph->get_output_values()) {
        auto lt = val->get_logical_tensor();
        if (lt.id == std::numeric_limits<size_t>::max()) continue;
        ASSERT_FALSE(logical_tensor_wrapper_t(lt).is_shape_unknown());
    }

    ASSERT_EQ(subgraph->infer_shape(), graph::status::success);

    if (params.constant_weight) {
        dnnl_impl::set_weight_bias_constant(subgraph);
        dnnl_impl::constant_propagation(subgraph);
    }

    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);
    auto final_subgraph_size = params.final_subgraph_size;
    ASSERT_EQ(subgraph->get_ops().size(), final_subgraph_size);
    // See: DUMMY_PARALLEL.
    dnnl::impl::parallel(1, [](int, int) {});
}

INSTANTIATE_TEST_SUITE_P(test_subgraph_pass, matmul_with_diff_inputs_t,
        testing::Values(matmul_params_t {{1, 1024}, {1000, 1024}, {1000},
                                {1, 1000}, false, true, false, 4, 4},
                matmul_params_t {{4, 3, 64}, {3, 64}, {3}, {4, 3, 3}, false,
                        true, false, 6, 6},
                matmul_params_t {{4, 64, 3}, {3, 64}, {3}, {4, 3, 3}, true,
                        true, false, 6, 6}));

TEST(test_subgraph_pass, ExecutionArgsSet) {
    ///////////////////////////
    // val1    val2
    //   \     /
    //    \   /
    //     op1
    //      |
    //     val3   val4
    //       \    /
    //        \  /
    //         op2
    //          |
    //         val5
    ///////////////////////////
    using value_t = graph::value_t;
    using dtype = dnnl::memory::data_type;
    using ftag = dnnl::memory::format_tag;
    using engine = dnnl::engine;
    using exec_args = graph::dnnl_impl::exec_args;
    using execution_args_set = graph::dnnl_impl::execution_args_set_t;

    value_t *val1 = (value_t *)1;
    value_t *val2 = (value_t *)2;
    value_t *val3 = (value_t *)3;
    value_t *val4 = (value_t *)4;
    value_t *val5 = (value_t *)5;

    graph::engine_t *g_eng = get_engine();
    engine eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    dnnl::memory mem1({{1, 2, 3, 4}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem2({{2, 3, 4, 5}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem3({{3, 4, 5, 6}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem4({{4, 5, 6, 7}, dtype::f32, ftag::abcd}, eng, nullptr);
    dnnl::memory mem5({{5, 6, 7, 8}, dtype::f32, ftag::abcd}, eng, nullptr);

    // construct the execution_args_set
    execution_args_set exec_args_set;
    exec_args_set.add_value_mem_map({val1, mem1});
    exec_args_set.add_value_mem_map({val2, mem2});
    exec_args_set.add_value_mem_map({val3, mem3});
    exec_args_set.add_value_mem_map({val4, mem4});
    exec_args_set.add_value_mem_map({val5, mem5});

    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem1, 0));
    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem2, 1));
    exec_args_set.add_mem_use_external_inputs(std::make_pair(mem4, 2));

    exec_args_set.add_mem_use_external_outputs(std::make_pair(mem5, 0));

    exec_args_set.add_mem_use_internal_temporary(std::make_pair(mem3, 0));

    exec_args op1_args;
    op1_args.insert({DNNL_ARG_SRC_0, mem1});
    op1_args.insert({DNNL_ARG_SRC_1, mem2});
    op1_args.insert({DNNL_ARG_DST, mem3});
    exec_args_set.add_exec_args(op1_args);

    exec_args op2_args;
    op2_args.insert({DNNL_ARG_SRC_0, mem3});
    op2_args.insert({DNNL_ARG_SRC_1, mem4});
    op2_args.insert({DNNL_ARG_DST, mem5});
    exec_args_set.add_exec_args(op2_args);

    // create the subgraph (will deep copy the exec_args_mgr implicitly)
    auto cloned_exec_args_set_ptr = exec_args_set.clone();
    const auto &cloned_exec_args_set = *cloned_exec_args_set_ptr;

    dnnl::memory cloned_mem1, cloned_mem2, cloned_mem3, cloned_mem4,
            cloned_mem5;
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val1, cloned_mem1));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val2, cloned_mem2));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val3, cloned_mem3));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val4, cloned_mem4));
    ASSERT_TRUE(cloned_exec_args_set.find_value_mem_map(val5, cloned_mem5));

    // because of deep copy, the desc should be same but the address should be
    // different
    ASSERT_TRUE(
            cloned_mem1.get_desc() == mem1.get_desc() && cloned_mem1 != mem1);
    ASSERT_TRUE(
            cloned_mem2.get_desc() == mem2.get_desc() && cloned_mem2 != mem2);
    ASSERT_TRUE(
            cloned_mem3.get_desc() == mem3.get_desc() && cloned_mem3 != mem3);
    ASSERT_TRUE(
            cloned_mem4.get_desc() == mem4.get_desc() && cloned_mem4 != mem4);
    ASSERT_TRUE(
            cloned_mem5.get_desc() == mem5.get_desc() && cloned_mem5 != mem5);

    // the external mems and internal mems are just alias to the mem object in
    // val-mem map, so both of their desc and address should be same
    auto mems_use_external_inputs
            = cloned_exec_args_set.get_mems_use_external_inputs();
    ASSERT_TRUE(cloned_mem1.get_desc()
                    == mems_use_external_inputs[0].first.get_desc()
            && cloned_mem1 == mems_use_external_inputs[0].first);
    ASSERT_TRUE(cloned_mem2.get_desc()
                    == mems_use_external_inputs[1].first.get_desc()
            && cloned_mem2 == mems_use_external_inputs[1].first);
    ASSERT_TRUE(cloned_mem4.get_desc()
                    == mems_use_external_inputs[2].first.get_desc()
            && cloned_mem4 == mems_use_external_inputs[2].first);

    auto mems_use_external_outputs
            = cloned_exec_args_set.get_mems_use_external_outputs();
    ASSERT_TRUE(cloned_mem5.get_desc()
                    == mems_use_external_outputs[0].first.get_desc()
            && cloned_mem5 == mems_use_external_outputs[0].first);

    auto mems_use_internal_variables
            = cloned_exec_args_set.get_mems_use_internal_temporary();
    ASSERT_TRUE(cloned_mem3.get_desc()
                    == mems_use_internal_variables[0].first.get_desc()
            && cloned_mem3 == mems_use_internal_variables[0].first);

    auto args = cloned_exec_args_set.get_exec_args();

    // the mems in args should also be alias
    auto cloned_op1_args = args[0];
    ASSERT_TRUE(
            cloned_mem1.get_desc() == cloned_op1_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem1 == cloned_op1_args[DNNL_ARG_SRC_0]);
    ASSERT_TRUE(
            cloned_mem2.get_desc() == cloned_op1_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem2 == cloned_op1_args[DNNL_ARG_SRC_1]);
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op1_args[DNNL_ARG_DST].get_desc()
            && cloned_mem3 == cloned_op1_args[DNNL_ARG_DST]);

    auto cloned_op2_args = args[1];
    ASSERT_TRUE(
            cloned_mem3.get_desc() == cloned_op2_args[DNNL_ARG_SRC_0].get_desc()
            && cloned_mem3 == cloned_op2_args[DNNL_ARG_SRC_0]);
    ASSERT_TRUE(
            cloned_mem4.get_desc() == cloned_op2_args[DNNL_ARG_SRC_1].get_desc()
            && cloned_mem4 == cloned_op2_args[DNNL_ARG_SRC_1]);
    ASSERT_TRUE(
            cloned_mem5.get_desc() == cloned_op2_args[DNNL_ARG_DST].get_desc()
            && cloned_mem5 == cloned_op2_args[DNNL_ARG_DST]);
}

TEST(test_subgraph_pass, MemoryPlanning) {
    /*
                / -> dnnl_reorder -> dnnl_reorder
               /
    mul_scales -> mul_scales -> permute -> mul_scales -> permute -> mul_scales
    -> mul_scales
    */
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> shape_NCX {64, 32, 256, 256};
    std::vector<int64_t> shape_NXC {64, 256, 256, 32};

    graph::op_t op1(1, op_kind::_mul_scales, "op1");
    graph::op_t op2(2, op_kind::_mul_scales, "op2");
    graph::op_t op3(3, op_kind::_permute, "op3");
    graph::op_t op4(4, op_kind::_mul_scales, "op4");
    graph::op_t op5(5, op_kind::_permute, "op5");
    graph::op_t op6(6, op_kind::_mul_scales, "op6");
    graph::op_t op7(7, op_kind::_mul_scales, "op7");

    op1.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op2.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op4.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op6.set_attr<std::vector<float>>(op_attr::scales, {0.5});
    op7.set_attr<std::vector<float>>(op_attr::scales, {0.5});

    logical_tensor_t val0
            = logical_tensor_init(0, shape_NCX, graph::data_type::f32);
    logical_tensor_t val1
            = logical_tensor_init(1, shape_NCX, graph::data_type::f32);
    logical_tensor_t val2
            = logical_tensor_init(2, shape_NCX, graph::data_type::f32);
    logical_tensor_t val3
            = logical_tensor_init(3, shape_NXC, graph::data_type::f32);
    logical_tensor_t val4
            = logical_tensor_init(4, shape_NXC, graph::data_type::f32);
    logical_tensor_t val5
            = logical_tensor_init(5, shape_NCX, graph::data_type::f32);
    logical_tensor_t val6
            = logical_tensor_init(6, shape_NCX, graph::data_type::f32);
    logical_tensor_t val7
            = logical_tensor_init(7, shape_NCX, graph::data_type::f32);

    op1.add_input(val0);
    op1.add_output(val1);
    op2.add_input(val1);
    op2.add_output(val2);
    op3.add_input(val2);
    op3.add_output(val3);
    op4.add_input(val3);
    op4.add_output(val4);
    op5.add_input(val4);
    op5.add_output(val5);
    op6.add_input(val5);
    op6.add_output(val6);
    op7.add_input(val6);
    op7.add_output(val7);

    graph::graph_t g;
    g.add_op(&op1);
    g.add_op(&op2);
    g.add_op(&op3);
    g.add_op(&op4);
    g.add_op(&op5);
    g.add_op(&op6);
    g.add_op(&op7);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 7U);

    std::vector<logical_tensor_t> inputs = {val0};
    std::vector<logical_tensor_t> outputs = {val7};
    dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

    // the fusion_info_mgr is dummy here
    dnnl_impl::memory_planner_t memory_planner;

    ASSERT_EQ(memory_planner.run(subgraph), graph::status::success);

    auto mem_offkeys = memory_planner.get_exec_args_set()
                               .get_mems_use_internal_temporary();
    ASSERT_TRUE(mem_offkeys.empty());
}

TEST(test_subgraph_pass, FusePostOpsForConvDepthwise_CPU) {
    /*   conv
          |
         conv (depthwise)
    */
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    SKIP_IF(g_eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    // N, IC, IH, IW
    std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    std::vector<int64_t> dw_dst_shape {4, 4, 2, 2};

    graph::op_t conv {0, graph::op_kind::Convolution, "conv"};
    set_conv_dw_base_op_attr(conv);

    graph::op_t depthwise {1, graph::op_kind::Convolution, "depthwise"};
    set_conv_dw_post_op_attr(depthwise, "k3s2p1");

    graph::logical_tensor_t conv_src
            = logical_tensor_init(0, conv_src_shape, graph::data_type::f32);
    graph::logical_tensor_t conv_wei
            = logical_tensor_init(1, conv_wei_shape, graph::data_type::f32);
    graph::logical_tensor_t conv_dst
            = logical_tensor_init(2, conv_dst_shape, graph::data_type::f32);

    graph::logical_tensor_t dw_wei
            = logical_tensor_init(3, dw_wei_shape, graph::data_type::f32);
    graph::logical_tensor_t dw_dst
            = logical_tensor_init(4, dw_dst_shape, graph::data_type::f32);

    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);

    depthwise.add_input(conv_dst);
    depthwise.add_input(dw_wei);
    depthwise.add_output(dw_dst);

    graph::graph_t g;
    g.add_op(&conv);
    g.add_op(&depthwise);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = get_pass("fp_conv_postops_depthwise_postops_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, graph_attr, false, true);
    dnnl_impl::subgraph_visualizer_t vis(part->id(), [](const value_t *val) {
        (void)val;
        return std::string();
    });
    dnnl_impl::pass_pipeline_t pipeline(vis, true, true);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), graph::status::success);
    // fused conv and to_groupped ops
    ASSERT_EQ(subgraph->num_ops(), 2U);
}

TEST(test_subgraph_pass, FailToFusePostOpsForConvDepthwise_CPU) {
    /*   conv
          |
         conv (depthwise)
    */
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    SKIP_IF(g_eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    // N, IC, IH, IW
    std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    std::vector<int64_t> dw_dst_shape {4, 4, 2, 2};

    graph::op_t conv {0, graph::op_kind::Convolution, "conv"};
    set_conv_dw_base_op_attr(conv);

    graph::op_t depthwise {1, graph::op_kind::Convolution, "depthwise"};
    set_conv_dw_post_op_attr(depthwise, "k3s2p1");

    graph::logical_tensor_t conv_src
            = logical_tensor_init(0, graph::data_type::f32);
    graph::logical_tensor_t conv_wei
            = logical_tensor_init(1, graph::data_type::f32);
    graph::logical_tensor_t conv_dst
            = logical_tensor_init(2, graph::data_type::f32);

    graph::logical_tensor_t dw_wei
            = logical_tensor_init(3, graph::data_type::f32);
    graph::logical_tensor_t dw_dst
            = logical_tensor_init(4, graph::data_type::f32);

    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);

    depthwise.add_input(conv_dst);
    depthwise.add_input(dw_wei);
    depthwise.add_output(dw_dst);

    graph::graph_t g;
    g.add_op(&conv);
    g.add_op(&depthwise);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = get_pass("fp_conv_postops_depthwise_postops_cpu");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 0U);
}

TEST(test_subgraph_pass, FuseSigmoidMultiplyToSwish) {
    /*   
              /\
        sigmoid \
              \ /
             multiply
                |
    */
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> src_shape {1, 16, 4, 4};

    graph::op_t sigmoid {0, graph::op_kind::Sigmoid, "sigmoid"};
    graph::op_t multiply {1, graph::op_kind::Multiply, "multiply"};

    graph::logical_tensor_t sigmoid_src
            = logical_tensor_init(0, src_shape, graph::data_type::f32);
    graph::logical_tensor_t sigmoid_dst
            = logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t multiply_dst
            = logical_tensor_init(2, src_shape, graph::data_type::f32);

    sigmoid.add_input(sigmoid_src);
    sigmoid.add_output(sigmoid_dst);

    multiply.add_input(sigmoid_src);
    multiply.add_input(sigmoid_dst);
    multiply.add_output(multiply_dst);

    graph::graph_t g;
    g.add_op(&sigmoid);
    g.add_op(&multiply);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("fp_eltwise_binary");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            part->get_ops(), p_eng, graph_attr, false, true);
    dnnl_impl::pass_pipeline_t pipeline(
            dnnl_impl::subgraph_visualizer_t(), true, false);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), graph::status::success);
    ASSERT_EQ(subgraph->num_ops(), 1U);
    ASSERT_EQ(subgraph->get_ops()[0]->get_kind(), op_kind::_eltwise);
    ASSERT_EQ(static_cast<dnnl::algorithm>(
                      subgraph->get_ops()[0]->get_attr<int64_t>(
                              graph::op_attr::alg_kind)),
            dnnl::algorithm::eltwise_swish);
}

TEST(test_subgraph_pass_int8_matmul_passes_with_diff_inputs,
        X8X8BF16MatmulScaleAddPasses_CPU) {
    /*
        | (u8/s8)  | (u8/s8)
     dequant    dequant
        | (f32)    | (f32)
     typecast  typecast
    (bf16) \     / (bf16)
           matmul
             | (bf16)
          div/mul
             | (bf16)
            add
             | (bf16)
    */

    std::vector<op_kind_t> scale_kinds {Multiply, Divide};
    for (auto scale_kind : scale_kinds) {
        graph::engine_t *g_eng = get_engine();

        SKIP_IF(unsupported_data_type(static_cast<dnnl::memory::data_type>(
                                              graph::data_type::bf16),
                        graph::dnnl_impl::make_dnnl_engine(*g_eng)),
                "Skip bf16 examples for systems that do not support "
                "avx512_core.");

        dnnl::engine p_eng
                = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
        graph_t agraph;
        std::vector<int64_t> zps = {0};
        std::vector<float> scales = {3.1f};
        op_t dequant1 {0, Dequantize, "dequant"};
        dequant1.set_attr(op_attr::scales, scales);
        dequant1.set_attr(op_attr::zps, zps);
        op_t dequant2 {1, Dequantize, "dequant"};
        dequant2.set_attr(op_attr::scales, scales);
        dequant2.set_attr(op_attr::zps, zps);
        op_t typecast1 {2, TypeCast, "typecast"};
        op_t typecast2 {3, TypeCast, "typecast"};
        op_t matmul {4, MatMul, "matmul"};
        op_t scale {5, scale_kind, "scale"};
        op_t add {6, Add, "add"};

        logical_tensor_t int8_data = logical_tensor_init(0, data_type::u8);
        logical_tensor_t fp32_data = logical_tensor_init(1, data_type::f32);
        dequant1.add_input(int8_data);
        dequant1.add_output(fp32_data);

        logical_tensor_t bf16_data = logical_tensor_init(2, data_type::bf16);
        typecast1.add_input(fp32_data);
        typecast1.add_output(bf16_data);

        logical_tensor_t int8_weight = logical_tensor_init(3, data_type::u8);
        logical_tensor_t fp32_weight = logical_tensor_init(4, data_type::f32);
        dequant2.add_input(int8_weight);
        dequant2.add_output(fp32_weight);

        logical_tensor_t bf16_weight = logical_tensor_init(5, data_type::bf16);
        typecast2.add_input(fp32_weight);
        typecast2.add_output(bf16_weight);

        logical_tensor_t bf16_matmul_out
                = logical_tensor_init(6, data_type::bf16);
        matmul.add_input(bf16_data);
        matmul.add_input(bf16_weight);
        matmul.add_output(bf16_matmul_out);

        logical_tensor_t bf16_scale_in
                = logical_tensor_init(7, data_type::bf16);
        logical_tensor_t bf16_scale_out
                = logical_tensor_init(8, data_type::bf16);
        scale.add_input(bf16_matmul_out);
        scale.add_input(bf16_scale_in);
        scale.add_output(bf16_scale_out);

        logical_tensor_t bf16_add_in = logical_tensor_init(9, data_type::bf16);
        logical_tensor_t bf16_add_out
                = logical_tensor_init(10, data_type::bf16);
        add.add_input(bf16_scale_out);
        add.add_input(bf16_add_in);
        add.add_output(bf16_add_out);

        ASSERT_EQ(agraph.add_op(&dequant1), status::success);
        ASSERT_EQ(agraph.add_op(&dequant2), status::success);
        ASSERT_EQ(agraph.add_op(&matmul), status::success);
        ASSERT_EQ(agraph.add_op(&typecast1), status::success);
        ASSERT_EQ(agraph.add_op(&typecast2), status::success);
        ASSERT_EQ(agraph.add_op(&scale), status::success);
        ASSERT_EQ(agraph.add_op(&add), status::success);

        agraph.finalize();

        pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        ASSERT_EQ((agraph.get_partitions()[0])->get_kind(),
                graph::partition_kind_t::quantized_matmul_post_ops);
        ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
        ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 4U);
        const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
        auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
                agraph.get_partitions()[0]->get_ops(), p_eng, graph_attr, false,
                true);
        // dequant, dequant, tc, tc, matmul, scale, add
        ASSERT_EQ(subgraph->get_ops().size(), 7U);

        dnnl_impl::check_with_bias(subgraph);

        int8_data = logical_tensor_init(0, {16, 8, 8, 8}, graph::data_type::u8);
        int8_weight
                = logical_tensor_init(3, {16, 8, 8, 8}, graph::data_type::u8);
        bf16_scale_in
                = logical_tensor_init(7, {1, 1, 1, 1}, graph::data_type::bf16);
        bf16_add_in
                = logical_tensor_init(9, {16, 1, 1, 8}, graph::data_type::bf16);
        bf16_add_out = logical_tensor_init(
                10, {16, 8, 8, 8}, graph::data_type::bf16);

        std::vector<logical_tensor_t> inputs
                = {int8_data, int8_weight, bf16_scale_in, bf16_add_in};
        std::vector<logical_tensor_t> outputs = {bf16_add_out};

        dnnl_impl::set_given_inputs_outputs(subgraph, inputs, outputs);

        dnnl_impl::pass_pipeline_t pipeline(
                dnnl_impl::subgraph_visualizer_t(), true, false);
        dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
        ASSERT_EQ(pipeline.run(subgraph), graph::status::success);

        // reorder, matmul, 4 const
        ASSERT_EQ(subgraph->get_ops().size(), 6U);

        for (auto &val : subgraph->get_input_values()) {
            auto lt = val->get_logical_tensor();
            ASSERT_FALSE(
                    graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
        }

        for (auto &val : subgraph->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == std::numeric_limits<size_t>::max()) continue;
            ASSERT_FALSE(
                    graph::logical_tensor_wrapper_t(lt).is_shape_unknown());
        }

        ASSERT_EQ(subgraph->infer_shape(), graph::status::success);

        ASSERT_EQ(dnnl_impl::layout_propagation(subgraph),
                graph::status::success);
        // reorder, matmul, 4 const
        ASSERT_EQ(subgraph->get_ops().size(), 6U);
    }
}

TEST(test_subgraph_pass, FuseTypecastToQuantize) {
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    graph_t agraph;

    std::vector<int64_t> src_shape = {1, 8, 16};
    graph::op_t typecast(0, graph::op_kind::TypeCast, "typecast");
    graph::op_t quantize(1, graph::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>(op_attr::scales, {0.1f});
    quantize.set_attr<std::vector<int64_t>>(op_attr::zps, {10});
    quantize.set_attr<std::string>(op_attr::qtype, "per_tensor");
    quantize.set_attr<int64_t>(op_attr::axis, 0);

    graph::logical_tensor_t src_bf16
            = logical_tensor_init(0, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t src_f32
            = logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_int8
            = logical_tensor_init(2, src_shape, graph::data_type::u8);

    typecast.add_input(src_bf16);
    typecast.add_output(src_f32);

    quantize.add_input(src_f32);
    quantize.add_output(dst_int8);

    ASSERT_EQ(agraph.add_op(&typecast), graph::status::success);
    ASSERT_EQ(agraph.add_op(&quantize), graph::status::success);
    agraph.finalize();
    pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops(), p_eng, graph_attr, false,
            true);
    // tc, quant
    ASSERT_EQ(subgraph->get_ops().size(), 2U);

    dnnl_impl::pass_pipeline_t pipeline(
            dnnl_impl::subgraph_visualizer_t(), true, false);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), graph::status::success);

    ASSERT_EQ(subgraph->get_ops().size(), 3U);
}

TEST(test_subgraph_pass_layout_propagation, ReshapeWithSpecifiedOutputLayout) {
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {1, 384, 1024};

    graph::op_t op1(1, graph::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    auto in = logical_tensor_init(0, in_shape, graph::data_type::f32);
    // the output layout is specified to be channel last
    auto out = logical_tensor_init(1, out_shape,
            std::vector<int64_t> {384 * 1024, 1, 384}, graph::data_type::f32);
    op1.add_input(in);
    op1.add_output(out);

    graph::graph_t g;
    g.add_op(&op1);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);

    // A reorder should be inserted before reshape op
    ASSERT_EQ(subgraph->get_ops().size(), 2U);
    std::vector<graph::op_t *> sorted_ops;
    graph::topo_order_visit(subgraph->get_output_ops(), [&](graph::op_t *op) {
        sorted_ops.emplace_back(op);
        return graph::status::success;
    });
    ASSERT_EQ(sorted_ops[0]->get_kind(), op_kind::_reorder);
}

TEST(test_subgraph_pass_layout_propagation,
        ReshapeWithUnreshapableInputLayout) {
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {384 * 16, 64};

    graph::op_t op1(1, graph::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    // the input layout is nhwc, which can't be directly reshaped to out_shape
    auto in = logical_tensor_init(0, in_shape,
            std::vector<int64_t> {384 * 16 * 64, 1, 384 * 64, 384},
            graph::data_type::f32);
    auto out = logical_tensor_init(
            1, out_shape, graph::data_type::f32, graph::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    graph::graph_t g;
    g.add_op(&op1);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);

    // A reorder should be inserted before reshape op
    ASSERT_EQ(subgraph->get_ops().size(), 2U);
    std::vector<graph::op_t *> sorted_ops;
    graph::topo_order_visit(subgraph->get_output_ops(), [&](graph::op_t *op) {
        sorted_ops.emplace_back(op);
        return graph::status::success;
    });
    ASSERT_EQ(sorted_ops[0]->get_kind(), op_kind::_reorder);
}

TEST(test_subgraph_pass_layout_propagation, ReshapeWithReshapableInputLayout) {
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {384 * 16, 64};

    graph::op_t op1(1, graph::op_kind::StaticReshape, "op1");
    op1.set_attr<std::vector<int64_t>>(op_attr::shape, out_shape);
    op1.set_attr<bool>(op_attr::special_zero, true);

    auto in = logical_tensor_init(0, in_shape, graph::data_type::f32);
    auto out = logical_tensor_init(
            1, out_shape, graph::data_type::f32, graph::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    graph::graph_t g;
    g.add_op(&op1);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);

    // No reorder
    ASSERT_EQ(subgraph->get_ops().size(), 1U);
}

TEST(test_subgraph_pass_layout_propagation, Transpose) {
    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);

    std::vector<int64_t> in_shape {1, 384, 16, 64};
    std::vector<int64_t> out_shape {1, 16, 64, 384};

    graph::op_t op1(1, graph::op_kind::StaticTranspose, "op1");
    op1.set_attr<std::vector<int64_t>>(
            op_attr::order, std::vector<int64_t> {0, 2, 3, 1});

    auto in = logical_tensor_init(0, in_shape, graph::data_type::f32);
    // the output layout is specified to be channel last
    auto out = logical_tensor_init(
            1, out_shape, graph::data_type::f32, graph::layout_type::any);
    op1.add_input(in);
    op1.add_output(out);

    graph::graph_t g;
    g.add_op(&op1);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(subgraph->get_ops().size(), 1U);

    ASSERT_EQ(dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(dnnl_impl::layout_propagation(subgraph), graph::status::success);

    // the output value's layout type should be strided, and the corresponding
    // md shape should be equal to the out_shape, the stride should be
    // transposed one
    auto out_lt
            = subgraph->get_ops()[0]->get_output_value(0)->get_logical_tensor();
    ASSERT_EQ(out_lt.layout_type, graph::layout_type::strided);
    auto out_md = dnnl_impl::make_dnnl_memory_desc(out_lt);
    ASSERT_EQ(out_md.get_dims(), out_shape);
    std::vector<int64_t> out_stride {393216, 64, 1, 1024};
    const auto md_stride = out_md.get_strides();
    ASSERT_EQ(md_stride, out_stride);
}

TEST(test_subgraph_pass, FuseTypecastBeforeFusePostops) {
    graph::engine_t *engine = get_engine();

    // prepare fp32 data
    std::vector<int64_t> src_shape = {3, 8, 4};
    std::vector<int64_t> weight_shape = {4, 2};
    std::vector<int64_t> bias_shape {2};
    std::vector<int64_t> dst_shape = {3, 8, 2};

    float scale_src = 1 / 255.f; // map to 0~255
    float scale_dst = 1 / 255.f; // map to 0~255
    int64_t zp_src = 0;
    int64_t zp_dst = 6;

    size_t id = 0;

    size_t scales_wei_sizes = dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(id++, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(op_attr::axis, 0);

    graph::op_t dqweight_op(id++, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(op_attr::qtype, "per_channel");
    dqweight_op.set_attr<std::vector<int64_t>>(op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(op_attr::axis, 1);

    graph::op_t matmul_op(id++, graph::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>(op_attr::transpose_a, false);
    matmul_op.set_attr<bool>(op_attr::transpose_b, false);

    graph::op_t gelu_op(id++, graph::op_kind::GELU, "gelu_op");

    graph::op_t tcdata_op {id++, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {id++, graph::op_kind::TypeCast, "typecast_weight"};
    graph::op_t tcdst_op {id++, graph::op_kind::TypeCast, "typecast_dst"};

    graph::op_t qdst_op(id++, graph::op_kind::Quantize, "qdst_op");
    qdst_op.set_attr<std::string>(op_attr::qtype, "per_tensor");
    qdst_op.set_attr<std::vector<int64_t>>(op_attr::zps, {zp_dst});
    qdst_op.set_attr<std::vector<float>>(op_attr::scales, {scale_dst});
    qdst_op.set_attr<int64_t>(op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = logical_tensor_init(id++, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = logical_tensor_init(id++, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = logical_tensor_init(id++, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = logical_tensor_init(id++, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_bf16
            = logical_tensor_init(5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_f32_dq
            = logical_tensor_init(id++, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t bias_bf16
            = logical_tensor_init(id++, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t dst_bf16
            = logical_tensor_init(id++, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_bf16
            = logical_tensor_init(id++, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t gelu_f32
            = logical_tensor_init(id++, dst_shape, graph::data_type::f32);
    graph::logical_tensor_t dst_u8
            = logical_tensor_init(id++, dst_shape, graph::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(dst_bf16);

    gelu_op.add_input(dst_bf16);
    gelu_op.add_output(gelu_bf16);

    tcdst_op.add_input(gelu_bf16);
    tcdst_op.add_output(gelu_f32);

    qdst_op.add_input(gelu_f32);
    qdst_op.add_output(dst_u8);

    graph::graph_t g(engine->kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcdst_op);
    g.add_op(&gelu_op);
    g.add_op(&qdst_op);
    g.finalize();

    pass::pass_base_ptr apass = get_pass("x8x8x_tc_matmul_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);

    graph::engine_t *g_eng = get_engine();
    dnnl::engine p_eng = dnnl::impl::graph::dnnl_impl::make_dnnl_engine(*g_eng);
    const graph::graph_attr_t graph_attr {{fpmath_mode::strict, false}};
    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            g.get_partitions()[0]->get_ops(), p_eng, graph_attr, false, true);
    ASSERT_EQ(subgraph->get_ops().size(), 8U);

    dnnl_impl::subgraph_visualizer_t vis(0, [](const value_t *val) {
        (void)val;
        return std::string();
    });
    dnnl_impl::pass_pipeline_t pipeline(vis, true, true);
    dnnl_impl::larger_partition_kernel_t::setup_pipeline_stage1(pipeline);
    ASSERT_EQ(pipeline.run(subgraph), graph::status::success);
    // 1 bias unsqueezing, 1 bias reorder, 1 fused matmul, 2 reshape, 4 const
    ASSERT_EQ(subgraph->num_ops(), 9U);
}

TEST(test_subgraph_pass, CheckUndefinedOpAttribute) {
    /*
    (f32) \     / (f32)
            conv
             | (f32)
            relu
             | (f32)
    */
    graph_t agraph;
    op_t conv {0, Convolution, "conv"};
    set_conv_common_attr(conv);
    op_t relu {1, ReLU, "relu"};

    logical_tensor_t fp32_data = logical_tensor_init(0, data_type::f32);
    logical_tensor_t fp32_weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t fp32_bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t fp32_conv_out = logical_tensor_init(3, data_type::f32);
    conv.add_input(fp32_data);
    conv.add_input(fp32_weight);
    conv.add_input(fp32_bias);
    conv.add_output(fp32_conv_out);

    logical_tensor_t fp32_relu_out = logical_tensor_init(4, data_type::f32);
    relu.add_input(fp32_conv_out);
    relu.add_output(fp32_relu_out);

    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);

    agraph.finalize();

    pass::pass_base_ptr apass = get_pass("fp_conv_post_ops");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_outputs().size(), 1U);
    ASSERT_EQ(agraph.get_partitions()[0]->get_inputs().size(), 3U);

    auto subgraph = std::make_shared<dnnl_impl::subgraph_t>(
            agraph.get_partitions()[0]->get_ops());

    fp32_data = logical_tensor_init(0, {1, 112, 112, 8}, data_type::f32);
    fp32_weight = logical_tensor_init(1, {3, 3, 8, 8}, data_type::f32);
    fp32_bias = logical_tensor_init(2, {8}, data_type::f32);

    // set incorrect op attribute to trigger the alarm.
    std::vector<int64_t> zps = {0};
    subgraph->get_ops()[0]->set_attr(op_attr::zps, zps);
    ASSERT_TRUE(subgraph->get_ops()[0]->has_attr(op_attr::zps));

    dnnl_impl::set_given_inputs_outputs(
            subgraph, {fp32_data, fp32_weight, fp32_bias}, {fp32_relu_out});
    dnnl_impl::lower_down(subgraph);
    dnnl_impl::subgraph_validator_t validator;
    ASSERT_EQ(validator.run(subgraph), status::invalid_graph_op);
}

TEST(test_subgraph_pass, CommonReorderElimination) {
    graph::engine_t &g_eng = *get_engine();
    dnnl::engine p_eng = graph::dnnl_impl::make_dnnl_engine(g_eng);
    size_t id = 0;
    auto lt1 = logical_tensor_init(id++, {1, 3, 5, 5}, graph::data_type::f32);
    auto lt2 = logical_tensor_init(id++, {1, 3, 5, 5}, graph::data_type::f32);

    auto lt3 = logical_tensor_init(id++, {1, 3, 5, 5}, graph::data_type::f32);
    auto lt4 = logical_tensor_init(id++, {1, 3, 5, 5}, graph::data_type::f32);
    auto lt5 = logical_tensor_init(id++, {1, 3, 5, 5}, graph::data_type::f32);

    graph::op_t op0 {0, graph::op_kind::Wildcard, "wild_card"};
    op0.add_input(lt1);
    op0.add_output(lt2);

    graph::op_t reorder_op1 {1, graph::op_kind::Reorder, "reorder_op1"};
    reorder_op1.add_input(lt2);
    reorder_op1.add_output(lt3);

    graph::op_t reorder_op2 {2, graph::op_kind::Reorder, "reorder2"};
    reorder_op2.add_input(lt2);
    reorder_op2.add_output(lt4);

    graph::op_t op1 {id++, graph::op_kind::Wildcard, "op2"};
    op1.add_input(lt3);
    op1.add_input(lt4);
    op1.add_output(lt5);

    graph::graph_t g;
    g.add_op(&op0);
    g.add_op(&reorder_op1);
    g.add_op(&reorder_op2);
    g.add_op(&op1);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::any, false}};
    auto subgraph = std::make_shared<graph::dnnl_impl::subgraph_t>(
            g.get_ops(), p_eng, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(graph::dnnl_impl::common_reorder_elimination(subgraph),
            graph::status::success);
    ASSERT_EQ(subgraph->get_ops().size(), 3U);
}

TEST(test_subgraph_pass, CombineBinaryPostOpScales) {
    namespace utils = dnnl::graph::tests::unit::utils;
    dnnl_impl::dnnl_backend_t::get_singleton();
    using dims = graph::dnnl_impl::dims;
    using config_t = std::tuple<graph::op_kind_t, bool>;

    graph::engine_t &engine = *get_engine();
    dnnl::engine p_engine = graph::dnnl_impl::make_dnnl_engine(engine);

    auto conf = config_t {graph::op_kind::AvgPool, true};
    std::string qtype = "symmetric";

    graph::op_kind_t base_op = graph::op_kind::Wildcard;
    bool per_channel_broadcast = false;
    std::tie(base_op, per_channel_broadcast) = conf;

    const std::string data_format {"NCX"};
    const int64_t channels = 2;
    std::vector<int64_t> src_shape {2, channels, 4, 4};
    std::vector<int64_t> dst_shape {2, channels, 2, 2};
    std::vector<int64_t> other_shape {1, 1, 1, 1};
    if (per_channel_broadcast) other_shape[1] = channels;

    const float scale_src = 5 / 127.f;
    const float scale_out = 10 / 127.f;
    const float scale_other = 2 / 127.f;
    const int64_t zp_src = (qtype == "symmetric") ? 0 : -2;
    const int64_t zp_out = (qtype == "symmetric") ? 0 : -2;
    const int64_t zp_other = (qtype == "symmetric") ? 0 : 4;

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t pool_op(1, base_op, "pool_op");
    size_t spatial_size = src_shape.size() - 2;
    pool_op.set_attr<dims>(graph::op_attr::strides, dims(spatial_size, 2));
    pool_op.set_attr<dims>(graph::op_attr::kernel, dims(spatial_size, 2));
    pool_op.set_attr<dims>(graph::op_attr::pads_begin, dims(spatial_size, 0));
    pool_op.set_attr<dims>(graph::op_attr::pads_end, dims(spatial_size, 0));
    pool_op.set_attr<std::string>(graph::op_attr::data_format, data_format);
    pool_op.set_attr<bool>(graph::op_attr::exclude_pad, false);

    graph::op_t qout_op(2, graph::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_out});
    qout_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_out});
    qout_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqother_op(3, graph::op_kind::Dequantize, "dqother_op");
    dqother_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqother_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_other});
    dqother_op.set_attr<std::vector<float>>(
            graph::op_attr::scales, {scale_other});
    dqother_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t mul_op(4, graph::op_kind::Multiply, "mul_op");

    auto src_s8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::s8);
    auto src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    auto dst_f32
            = utils::logical_tensor_init(2, dst_shape, graph::data_type::f32);
    auto dst_s8
            = utils::logical_tensor_init(3, dst_shape, graph::data_type::s8);
    auto other_s8
            = utils::logical_tensor_init(4, other_shape, graph::data_type::s8);
    auto other_f32_dq
            = utils::logical_tensor_init(5, other_shape, graph::data_type::f32);
    auto dst_mul_f32
            = utils::logical_tensor_init(6, dst_shape, graph::data_type::f32);

    dqdata_op.add_input(src_s8);
    dqdata_op.add_output(src_f32_dq);

    pool_op.add_input(src_f32_dq);
    pool_op.add_output(dst_f32);

    dqother_op.add_input(other_s8);
    dqother_op.add_output(other_f32_dq);

    mul_op.add_input(other_f32_dq);
    mul_op.add_input(dst_f32);

    mul_op.add_output(dst_mul_f32);

    qout_op.add_input(dst_mul_f32);
    qout_op.add_output(dst_s8);

    graph::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&pool_op);
    g.add_op(&dqother_op);
    g.add_op(&mul_op);
    g.add_op(&qout_op);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::any, false}};
    auto subgraph = std::make_shared<graph::dnnl_impl::subgraph_t>(
            g.get_ops(), p_engine, graph_attr, false, /* reset_layout */ false);
    ASSERT_EQ(graph::dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_to_int8_pool(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::combine_binary_post_op_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fold_mul_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::remove_quant_data_with_no_effect(subgraph),
            graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::replace_quant_data_with_binary_post_op(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_runtime_mul_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_runtime_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_dynamic_mul_scales_add_zps(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_dynamic_sub_zps_mul_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::fuse_post_ops(subgraph), graph::status::success);
    ASSERT_EQ(subgraph->num_ops(), 2U);
}

TEST(test_subgraph_pass, FuseNCXConvolutionBinaryAddNC11PostSrc) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    namespace utils = dnnl::graph::tests::unit::utils;
    dnnl_impl::dnnl_backend_t::get_singleton();
    graph::engine_t &engine = *get_engine();
    dnnl::engine p_engine = graph::dnnl_impl::make_dnnl_engine(engine);

    int64_t in_channel = 320, out_channel = 320;
    std::vector<int64_t> src_shape = {2, in_channel, 96, 96};
    std::vector<int64_t> weight_shape = {out_channel, in_channel / 1, 3, 3};
    std::vector<int64_t> bias_shape = {2, out_channel, 1, 1};
    std::vector<int64_t> dst_shape = {2, out_channel, 96, 96};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(distribution(generator)); });
    std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 1));
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t add_op {5, graph::op_kind::Add, "add_op"};

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t conv_bf16
            = utils::logical_tensor_init(6, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t add_bf16
            = utils::logical_tensor_init(8, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t add_out_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    conv_op.add_input(src_bf16);
    conv_op.add_input(weight_bf16);
    conv_op.add_output(conv_bf16);

    add_op.add_input(conv_bf16);
    add_op.add_input(add_bf16);
    add_op.add_output(add_out_bf16);

    graph::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::any, false}};
    auto subgraph = std::make_shared<graph::dnnl_impl::subgraph_t>(g.get_ops(),
            p_engine, graph_attr, false,
            /* reset_layout */ false);
    ASSERT_EQ(graph::dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::fuse_bias_add(subgraph), graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_matmul_or_conv(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_add(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_mul_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_bias_to_f32(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::remove_quant_data_with_no_effect(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::check_with_bias(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::binary_canonicalization(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::binary_broadcast_swap(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_to_runtime_src_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_src_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_to_runtime_src_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_src_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::fuse_post_ops(subgraph), graph::status::success);
    auto qconv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_convolution;
    });
    ASSERT_NE(qconv_op, subgraph->get_ops().end());
    ASSERT_TRUE((*qconv_op)->has_attr(graph::op_attr::fusion_info));
    const auto &fusion_info = (*qconv_op)->get_attr<dnnl_impl::fusion_info_t>(
            graph::op_attr::fusion_info);
    const auto &post_ops = fusion_info.get_post_ops();
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (engine.kind() == graph::engine_kind::gpu) {
        ASSERT_EQ(post_ops.size(), 0U);
    } else {
        ASSERT_EQ(post_ops.size(), 1U);
    }

#else
    ASSERT_EQ(post_ops.size(), 1U);
#endif
}

TEST(test_subgraph_pass, FuseNXCConvolutionBinaryAddNC11PostSrc) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;
    namespace utils = dnnl::graph::tests::unit::utils;
    dnnl_impl::dnnl_backend_t::get_singleton();
    graph::engine_t &engine = *get_engine();
    dnnl::engine p_engine = graph::dnnl_impl::make_dnnl_engine(engine);

    int64_t in_channel = 320, out_channel = 320;
    std::vector<int64_t> src_shape = {2, 96, 96, in_channel};
    std::vector<int64_t> weight_shape = {out_channel, in_channel / 1, 3, 3};
    std::vector<int64_t> bias_shape = {2, 1, 1, out_channel};
    std::vector<int64_t> dst_shape = {2, 96, 96, out_channel};

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<float> weight_data(product(weight_shape));
    std::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(distribution(generator)); });
    std::uniform_real_distribution<float> distribution2(-1.f, 1.f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    graph::op_t dqdata_op(0, graph::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, {zp_src});
    dqdata_op.set_attr<std::vector<float>>(graph::op_attr::scales, {scale_src});
    dqdata_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::op_t dqweight_op(1, graph::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>(graph::op_attr::qtype, "per_channel");
    dqweight_op.set_attr<std::vector<int64_t>>(graph::op_attr::zps, zp_wei);
    dqweight_op.set_attr<std::vector<float>>(graph::op_attr::scales, scale_wei);
    dqweight_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    graph::op_t tcdata_op {2, graph::op_kind::TypeCast, "typecast_data"};
    graph::op_t tcweight_op {3, graph::op_kind::TypeCast, "typecast_weight"};

    graph::op_t conv_op(4, graph::op_kind::Convolution, "conv_op");
    conv_op.set_attr<dims>(graph::op_attr::strides, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims(2, 1));
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims(2, 1));
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::op_t add_op {5, graph::op_kind::Add, "add_op"};

    // prepare logical tensor
    graph::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, graph::data_type::u8);
    graph::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, graph::data_type::f32);
    graph::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, graph::data_type::bf16);
    graph::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, graph::data_type::s8);
    graph::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
            4, weight_shape, graph::data_type::f32);
    graph::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, graph::data_type::bf16);
    graph::logical_tensor_t conv_bf16
            = utils::logical_tensor_init(6, dst_shape, graph::data_type::bf16);
    graph::logical_tensor_t add_bf16
            = utils::logical_tensor_init(8, bias_shape, graph::data_type::bf16);
    graph::logical_tensor_t add_out_bf16
            = utils::logical_tensor_init(9, dst_shape, graph::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    conv_op.add_input(src_bf16);
    conv_op.add_input(weight_bf16);
    conv_op.add_output(conv_bf16);

    add_op.add_input(conv_bf16);
    add_op.add_input(add_bf16);
    add_op.add_output(add_out_bf16);

    graph::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dqdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcdata_op), graph::status::success);
    ASSERT_EQ(g.add_op(&dqweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&tcweight_op), graph::status::success);
    ASSERT_EQ(g.add_op(&conv_op), graph::status::success);
    ASSERT_EQ(g.add_op(&add_op), graph::status::success);
    g.finalize();
    const graph::graph_attr_t graph_attr {{fpmath_mode::any, false}};
    auto subgraph = std::make_shared<graph::dnnl_impl::subgraph_t>(g.get_ops(),
            p_engine, graph_attr, false,
            /* reset_layout */ false);
    ASSERT_EQ(graph::dnnl_impl::lower_down(subgraph), graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::fuse_bias_add(subgraph), graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_matmul_or_conv(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_add(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_typecast_to_mul_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_bias_to_f32(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::remove_quant_data_with_no_effect(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::check_with_bias(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::binary_canonicalization(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::binary_broadcast_swap(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_to_runtime_src_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_src_scales(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::convert_to_runtime_src_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(graph::dnnl_impl::fuse_src_zero_points(subgraph),
            graph::status::success);
    ASSERT_EQ(
            graph::dnnl_impl::fuse_post_ops(subgraph), graph::status::success);
    auto qconv_op = std::find_if(subgraph->get_ops().begin(),
            subgraph->get_ops().end(), [](const std::shared_ptr<op_t> &op) {
        return op->get_kind() == op_kind::_convolution;
    });
    ASSERT_NE(qconv_op, subgraph->get_ops().end());
    ASSERT_TRUE((*qconv_op)->has_attr(graph::op_attr::fusion_info));
    const auto &fusion_info = (*qconv_op)->get_attr<dnnl_impl::fusion_info_t>(
            graph::op_attr::fusion_info);
    const auto &post_ops = fusion_info.get_post_ops();
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (engine.kind() == graph::engine_kind::gpu) {
        ASSERT_EQ(post_ops.size(), 0U);
    } else {
        ASSERT_EQ(post_ops.size(), 1U);
    }

#else
    ASSERT_EQ(post_ops.size(), 1U);
#endif
}
