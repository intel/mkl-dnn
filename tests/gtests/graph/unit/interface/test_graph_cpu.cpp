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

#include <string>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/value.hpp"

#include "graph/unit/utils.hpp"

TEST(test_interface_graph, Create) {
    using namespace dnnl::impl::graph;

    graph_t g_default_engine;
    ASSERT_EQ(g_default_engine.get_engine_kind(), engine_kind::cpu);

    graph_t g_gpu {engine_kind::gpu};
    ASSERT_EQ(g_gpu.get_engine_kind(), engine_kind::gpu);
}

TEST(test_interface_graph, AddOp) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_graph_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);

    auto ret = agraph.get_ops()[0];
    ASSERT_EQ(*ret, op1);
}

TEST(test_interface_graph, FailAddOpWithInvalidAttrValue) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::weights_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, data_type::f32);

    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    ASSERT_EQ(agraph.add_op(&op0), status::success);

    op0.set_attr<std::string>(op_attr::weights_format, "IOX");
    graph_t agraph1;
    ASSERT_EQ(agraph1.add_op(&op0), status::invalid_graph_op);
}

TEST(test_interface_graph, AddNullOp) {
    using namespace dnnl::impl::graph;

    graph_t agraph;
    ASSERT_EQ(agraph.add_op(nullptr), status::invalid_graph_op);
    ASSERT_EQ(agraph.num_ops(), 0U);
    ASSERT_EQ(agraph.get_ops().size(), 0U);
}

TEST(test_interface_graph, DeleteOp) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_graph_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
    ASSERT_EQ(agraph.get_ops().size(), 1U);

    agraph.delete_op(&op1);
    ASSERT_EQ(agraph.num_ops(), 0U);
    ASSERT_EQ(agraph.get_ops().size(), 0U);
}

TEST(test_interface_graph, GetOutputOps) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, ReLU, std::string("relu1")};
    op_t op1 {1, ReLU, std::string("relu2")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 2U);
    agraph.finalize();
    ASSERT_EQ(agraph.get_output_ops().size(), 1U);
    ASSERT_EQ(*(agraph.get_output_ops()[0]), op1);
}

TEST(test_interface_graph, GetOutputOps2) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, ReLU, std::string("relu1")};
    op_t op1 {1, ReLU, std::string("relu2")};
    op_t op2 {2, ReLU, std::string("relu3")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    logical_tensor_t t3 = logical_tensor_init(3, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    op2.add_input(t2);
    op2.add_output(t3);
    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.num_ops(), 3U);
    agraph.finalize();
    ASSERT_EQ(agraph.get_ops()[1]
                      ->get_output_value(0)
                      ->get_consumers()[0]
                      .get_op(),
            *agraph.get_ops()[2]);
    ASSERT_EQ(agraph.get_ops()[0]
                      ->get_output_value(0)
                      ->get_consumers()[0]
                      .get_op(),
            *agraph.get_ops()[1]);
    ASSERT_EQ(agraph.get_ops()[2]->get_input_value(0)->has_producer(), true);
    ASSERT_EQ(agraph.get_ops()[1]->get_input_value(0)->has_producer(), true);
}

TEST(test_interface_graph, BuildGraph) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_graph_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);

    ASSERT_EQ(agraph.finalize(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
    ASSERT_EQ(agraph.finalize(), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
}

TEST(test_interface_graph, InvalidOp) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv2d")};
    op_t op1 {1, ReLU, std::string("relu")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t t2 = logical_tensor_init(2, data_type::f32);
    op0.add_input(t0);
    op0.add_output(t1);
    op1.add_input(t1);
    op1.add_output(t2);
    ASSERT_EQ(agraph.add_op(&op0), status::invalid_graph_op);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);

    /*
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::invalid_graph);
    ASSERT_EQ(agraph.finalize(), status::success);
    ASSERT_EQ(agraph.run_pass(partition_policy::fusion), status::success);
    */
}

TEST(test_interface_graph, Wildcard) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t op {0, Wildcard, std::string("wildcard")};
    logical_tensor_t t0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t t1 = logical_tensor_init(1, data_type::f32);
    op.add_input(t0);
    op.add_output(t1);
    ASSERT_EQ(agraph.add_op(&op), status::success);
    ASSERT_EQ(agraph.num_ops(), 1U);
}

TEST(test_interface_graph, GetInputOutputEdges) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::impl::graph::logical_tensor_wrapper_t;

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};
    op_t op3 {3, Wildcard, std::string("wildcard0")};
    op_t op4 {4, Wildcard, std::string("wildcard1")};
    op_t op5 {5, Wildcard, std::string("wildcard2")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::weights_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t weight = logical_tensor_init(1, data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, data_type::f32);
    logical_tensor_t other = logical_tensor_init(3, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, data_type::f32);
    logical_tensor_t dst = logical_tensor_init(6, data_type::f32);
    logical_tensor_t wild_val = logical_tensor_init(7, data_type::f32);

    op4.add_output(src);
    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    op1.add_input(conv_dst);
    op1.add_input(other);
    op1.add_output(add_dst);

    op2.add_input(add_dst);
    op2.add_output(dst);

    op3.add_input(conv_dst);
    op3.add_output(wild_val);

    op5.add_input(wild_val);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.add_op(&op3), status::success);
    ASSERT_EQ(agraph.num_ops(), 4U);
    agraph.finalize();

    auto ops = agraph.get_ops();
    ops.pop_back();
    graph_t subgraph(ops);
    ASSERT_EQ(subgraph.num_ops(), 3U);

    auto in_vals = subgraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4U);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    auto out_vals = subgraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 2U);
    logical_tensor_t out_lt1 = out_vals[0]->get_logical_tensor();
    logical_tensor_t out_lt2 = out_vals[1]->get_logical_tensor();
    if (out_lt1.id == 6)
        ASSERT_EQ(out_lt2.id, 4U);
    else if (out_lt1.id == 4)
        ASSERT_EQ(out_lt2.id, 6U);
    else
        ASSERT_TRUE(false);
}

TEST(test_interface_graph, InferShape) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;
    using ltw = dnnl::impl::graph::logical_tensor_wrapper_t;

    std::vector<int64_t> src_shape {8, 3, 227, 227};
    std::vector<int64_t> weight_shape {96, 3, 11, 11};
    std::vector<int64_t> bias_shape {96};
    std::vector<int64_t> dst_shape {8, 96, 55, 55};

    graph_t agraph;
    op_t op0 {0, Convolution, std::string("conv0")};
    op_t op1 {1, Add, std::string("add0")};
    op_t op2 {2, ReLU, std::string("relu0")};

    op0.set_attr<std::vector<int64_t>>(op_attr::strides, {4, 4});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {111, 111});
    op0.set_attr<std::vector<int64_t>>(op_attr::pads_end, {111, 111});
    op0.set_attr<std::string>(op_attr::auto_pad, "VALID");
    op0.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    op0.set_attr<std::string>(op_attr::data_format, "NCX");
    op0.set_attr<std::string>(op_attr::weights_format, "OIX");
    op0.set_attr<int64_t>(op_attr::groups, 1);

    // prepare logical tensor
    logical_tensor_t src = logical_tensor_init(0, src_shape, data_type::f32);
    logical_tensor_t weight
            = logical_tensor_init(1, weight_shape, data_type::f32);
    logical_tensor_t bias = logical_tensor_init(2, bias_shape, data_type::f32);
    logical_tensor_t other = logical_tensor_init(3, dst_shape, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(4, data_type::f32);
    logical_tensor_t add_dst = logical_tensor_init(5, data_type::f32);
    logical_tensor_t dst = logical_tensor_init(6, data_type::f32);

    op0.add_input(src);
    op0.add_input(weight);
    op0.add_input(bias);
    op0.add_output(conv_dst);

    op1.add_input(conv_dst);
    op1.add_input(other);
    op1.add_output(add_dst);

    op2.add_input(add_dst);
    op2.add_output(dst);

    ASSERT_EQ(agraph.add_op(&op0), status::success);
    ASSERT_EQ(agraph.add_op(&op1), status::success);
    ASSERT_EQ(agraph.add_op(&op2), status::success);
    ASSERT_EQ(agraph.num_ops(), 3U);
    agraph.finalize();

    auto in_vals = agraph.get_input_values();
    ASSERT_EQ(in_vals.size(), 4U);
    ASSERT_EQ(ltw(in_vals[0]->get_logical_tensor()), ltw(src));
    ASSERT_EQ(ltw(in_vals[1]->get_logical_tensor()), ltw(weight));
    ASSERT_EQ(ltw(in_vals[2]->get_logical_tensor()), ltw(bias));
    ASSERT_EQ(ltw(in_vals[3]->get_logical_tensor()), ltw(other));

    ASSERT_EQ(agraph.infer_shape(), status::success);

    auto out_vals = agraph.get_output_values();
    ASSERT_EQ(out_vals.size(), 1U);
    logical_tensor_t out_lt = out_vals[0]->get_logical_tensor();
    ASSERT_EQ(out_lt.id, 6U);
    ASSERT_EQ(out_lt.ndims, 4);
    ASSERT_EQ(out_lt.dims[0], 8);
    ASSERT_EQ(out_lt.dims[1], 96);
    ASSERT_EQ(out_lt.dims[2], 55);
    ASSERT_EQ(out_lt.dims[3], 55);
}

TEST(test_interface_graph, SetFpmathMode) {
    using namespace dnnl::impl::graph;
    ASSERT_EQ(dnnl::impl::get_fpmath_mode(), fpmath_mode::strict);

    graph_t graph;
    ASSERT_EQ(graph.get_attr().get_fpmath().mode_, fpmath_mode::strict);

    for (auto m : {fpmath_mode::strict, fpmath_mode::bf16, fpmath_mode::f16,
                 fpmath_mode::any}) {
        graph_t graph2 {engine_kind::cpu, m};
        ASSERT_EQ(graph2.get_attr().get_fpmath().mode_, m);
    }
}

TEST(test_interface_graph, SetUserInputsOutputs) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    op_t div(0, op_kind::Divide, "add_op");

    // prepare logical tensor
    logical_tensor_t src0 = logical_tensor_init(0, data_type::f32);
    logical_tensor_t src1 = logical_tensor_init(1, data_type::f32);
    logical_tensor_t dst = logical_tensor_init(2, data_type::f32);

    div.add_input(src0);
    div.add_input(src1);
    div.add_output(dst);

    graph_t g(engine_kind::cpu);
    ASSERT_EQ(g.add_op(&div), status::success);
    g.finalize();

    logical_tensor_t src0_compile
            = logical_tensor_init(0, {2, 2}, data_type::f32);
    logical_tensor_t src1_compile = logical_tensor_init(1, {}, data_type::f32);

    ASSERT_EQ(g.set_user_inputs_outputs({src0_compile, src1_compile}, {dst}),
            status::success);
    ASSERT_EQ(g.infer_shape(), status::success);
    auto out_vals = g.get_output_values();
    ASSERT_EQ(out_vals.size(), 1U);
    logical_tensor_t out_lt = out_vals[0]->get_logical_tensor();
    ASSERT_EQ(out_lt.id, 2U);
    ASSERT_EQ(out_lt.ndims, 2);
    ASSERT_EQ(out_lt.dims[0], 2);
    ASSERT_EQ(out_lt.dims[1], 2);
}

TEST(test_interface_graph, NonDAGGraph) {
    /*
          mm0 <--
          |     |
         relu1  |
          |     |
          mm2  /
          |   /
         relu3
    */
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    op_t mm0(0, op_kind::MatMul, "matmul_op");
    op_t relu1(1, op_kind::ReLU, "relu_op");
    op_t mm2(2, op_kind::MatMul, "matmul_op");
    op_t relu3(3, op_kind::ReLU, "relu_op");

    // prepare logical tensor
    logical_tensor_t mm0_src = logical_tensor_init(0, {1, 2}, data_type::f32);
    logical_tensor_t mm0_weight
            = logical_tensor_init(1, {2, 1}, data_type::f32);
    logical_tensor_t mm0_dst = logical_tensor_init(2, {1, 1}, data_type::f32);
    logical_tensor_t relu1_dst = logical_tensor_init(3, {1, 1}, data_type::f32);

    logical_tensor_t mm2_weight
            = logical_tensor_init(4, {2, 1}, data_type::f32);
    logical_tensor_t mm2_dst = logical_tensor_init(5, {1, 1}, data_type::f32);

    mm0.add_input(mm0_src);
    mm0.add_input(mm0_weight);
    mm0.add_output(mm0_dst);
    relu1.add_input(mm0_dst);
    relu1.add_output(relu1_dst);
    mm2.add_input(relu1_dst);
    mm2.add_input(mm2_weight);
    mm2.add_output(mm2_dst);
    relu3.add_input(mm2_dst);
    relu3.add_output(mm0_src);

    graph_t g(engine_kind::cpu);
    ASSERT_EQ(g.add_op(&mm0), status::success);
    ASSERT_EQ(g.add_op(&relu1), status::success);
    ASSERT_EQ(g.add_op(&mm2), status::success);
    ASSERT_EQ(g.add_op(&relu3), status::success);

    status_t status = g.finalize();
    ASSERT_EQ(status, status::invalid_graph);
}

TEST(test_interface_graph, SingleOpGraph) {
    /*                __
        \ /          |   \  /
         matmul  or  |  matmul
           |         |____|
    */
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    std::vector<status_t> statuses = {status::success, status::invalid_graph};
    for (const auto &expected_status : statuses) {
        op_t mm(0, op_kind::MatMul, "matmul_op");
        // prepare logical tensor
        logical_tensor_t mm_src
                = logical_tensor_init(0, {1, 2}, data_type::f32);
        logical_tensor_t mm_weight
                = logical_tensor_init(1, {2, 2}, data_type::f32);
        logical_tensor_t mm_dst
                = logical_tensor_init(2, {1, 2}, data_type::f32);

        mm.add_input(mm_src);
        mm.add_input(mm_weight);
        if (expected_status == status::success) {
            mm.add_output(mm_dst);
        } else {
            mm.add_output(mm_src);
        }

        graph_t g(engine_kind::cpu);
        ASSERT_EQ(g.add_op(&mm), status::success);

        status_t status = g.finalize();
        ASSERT_EQ(status, expected_status);
    }
}

TEST(test_interface_graph, DAGGraphWithNonDAGGraph) {
    /*
        mm0             mm2 <--
         |     &&        |     |
        relu1          relu3 __|
    */
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;
    using namespace dnnl::graph::tests::unit::utils;

    op_t mm0(0, op_kind::MatMul, "matmul_op");
    op_t relu1(1, op_kind::ReLU, "relu_op");
    op_t mm2(2, op_kind::MatMul, "matmul_op");
    op_t relu3(3, op_kind::ReLU, "relu_op");

    // prepare logical tensor
    logical_tensor_t mm0_src = logical_tensor_init(0, {1, 2}, data_type::f32);
    logical_tensor_t mm0_weight
            = logical_tensor_init(1, {2, 1}, data_type::f32);
    logical_tensor_t mm0_dst = logical_tensor_init(2, {1, 1}, data_type::f32);
    logical_tensor_t relu1_dst = logical_tensor_init(3, {1, 1}, data_type::f32);

    logical_tensor_t mm2_src = logical_tensor_init(4, {1, 5}, data_type::f32);
    logical_tensor_t mm2_weight
            = logical_tensor_init(5, {5, 5}, data_type::f32);
    logical_tensor_t mm2_dst = logical_tensor_init(6, {1, 5}, data_type::f32);

    mm0.add_input(mm0_src);
    mm0.add_input(mm0_weight);
    mm0.add_output(mm0_dst);
    relu1.add_input(mm0_dst);
    relu1.add_output(relu1_dst);

    mm2.add_input(mm2_src);
    mm2.add_input(mm2_weight);
    mm2.add_output(mm2_dst);
    relu3.add_input(mm2_dst);
    relu3.add_output(mm2_src);

    graph_t g(engine_kind::cpu);
    ASSERT_EQ(g.add_op(&mm0), status::success);
    ASSERT_EQ(g.add_op(&relu1), status::success);
    ASSERT_EQ(g.add_op(&mm2), status::success);
    ASSERT_EQ(g.add_op(&relu3), status::success);

    status_t status = g.finalize();
    ASSERT_EQ(status, status::invalid_graph);
}
