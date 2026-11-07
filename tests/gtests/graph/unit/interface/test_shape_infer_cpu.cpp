/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/shape_infer.hpp"

#include "backend/dnnl/common.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using graph::dims;

TEST(test_interface_shape_infer, OneWayBroadcast) {
    using dims = graph::dims;
    dims src_shape {2, 3};
    dims dst1_shape {2};
    dims dst2_shape {2, 3};
    dims dst3_shape {4, 3};
    dims dst4_shape {1, 2, 3};

    ASSERT_EQ(graph::one_way_broadcast(dst1_shape, src_shape),
            graph::status::invalid_shape);

    ASSERT_EQ(graph::one_way_broadcast(dst2_shape, src_shape),
            graph::status::success);

    ASSERT_EQ(graph::one_way_broadcast(dst3_shape, src_shape),
            graph::status::invalid_shape);

    ASSERT_EQ(graph::one_way_broadcast(dst4_shape, src_shape),
            graph::status::success);
}

TEST(test_interface_shape_infer, StaticReshapeExceedsMaxNdims) {
    // The output rank of StaticReshape comes straight from the user-provided
    // shape attribute. A rank above DNNL_MAX_NDIMS must be rejected before it
    // is written into the fixed-size dims of the output logical tensor.
    graph::op_t reshape_op {
            0, graph::op_kind::StaticReshape, std::string("reshape")};
    reshape_op.set_attr<bool>(graph::op_attr::special_zero, false);
    // all ones so the total volume still matches the input and the size check
    // does not reject it first; the rank alone is out of range.
    dims target(DNNL_MAX_NDIMS + 20, 1);
    reshape_op.set_attr<dims>(graph::op_attr::shape, target);

    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {1}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            1, graph::data_type::f32, graph::layout_type::strided);

    std::vector<graph::logical_tensor_t *> inputs {&src};
    std::vector<graph::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(graph::infer_static_reshape_output_shape(
                      &reshape_op, inputs, outputs),
            graph::status::invalid_shape);
}

TEST(test_interface_shape_infer, StaticReshapeValid) {
    graph::op_t reshape_op {
            0, graph::op_kind::StaticReshape, std::string("reshape")};
    reshape_op.set_attr<bool>(graph::op_attr::special_zero, false);
    reshape_op.set_attr<dims>(graph::op_attr::shape, dims {6});

    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            1, graph::data_type::f32, graph::layout_type::strided);

    std::vector<graph::logical_tensor_t *> inputs {&src};
    std::vector<graph::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(graph::infer_static_reshape_output_shape(
                      &reshape_op, inputs, outputs),
            graph::status::success);
    ASSERT_EQ(graph::logical_tensor_wrapper_t(dst).vdims(), (dims {6}));
}

TEST(test_interface_shape_infer, SoftMaxStatsAxisOutOfRange) {
    graph::op_t softmax_op {0, graph::op_kind::SoftMax, std::string("softmax")};

    for (int64_t axis : {-3, 2, 1000}) {
        softmax_op.set_attr<int64_t>(graph::op_attr::axis, axis);

        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t scratchpad = utils::logical_tensor_init(
                2, graph::data_type::u8, graph::layout_type::strided);
        graph::logical_tensor_t stats = utils::logical_tensor_init(
                3, graph::data_type::f32, graph::layout_type::strided);

        std::vector<graph::logical_tensor_t *> inputs {&src};
        std::vector<graph::logical_tensor_t *> outputs {&dst, &stats};
        std::vector<graph::logical_tensor_t *> dnnl_outputs {
                &dst, &scratchpad, &stats};
        ASSERT_EQ(
                graph::infer_softmax_output_shape(&softmax_op, inputs, outputs),
                graph::status::invalid_shape);
        ASSERT_EQ(graph::infer_dnnl_softmax_output_shape(
                          &softmax_op, inputs, dnnl_outputs),
                graph::status::invalid_shape);
    }
}

TEST(test_interface_shape_infer, SoftMaxStatsAxisValid) {
    graph::op_t softmax_op {0, graph::op_kind::SoftMax, std::string("softmax")};

    for (int64_t axis : {-2, -1, 0, 1}) {
        softmax_op.set_attr<int64_t>(graph::op_attr::axis, axis);

        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t stats = utils::logical_tensor_init(
                2, graph::data_type::f32, graph::layout_type::strided);

        std::vector<graph::logical_tensor_t *> inputs {&src};
        std::vector<graph::logical_tensor_t *> outputs {&dst, &stats};
        ASSERT_EQ(
                graph::infer_softmax_output_shape(&softmax_op, inputs, outputs),
                graph::status::success);

        const dims expected
                = (axis == 0 || axis == -2) ? dims {1, 3} : dims {2, 1};
        ASSERT_EQ(graph::logical_tensor_wrapper_t(stats).vdims(), expected);
    }
}

TEST(test_interface_shape_infer, LayerNormBeginNormAxisOutOfRange) {
    graph::op_t ln_op {0, graph::op_kind::LayerNorm, std::string("ln")};

    const std::vector<std::pair<dims, int64_t>> cases {
            {{2, 3}, -3}, {{2, 3}, 2}, {{2, 3}, 1000}, {{}, -1}};
    for (const auto &c : cases) {
        ln_op.set_attr<int64_t>(graph::op_attr::begin_norm_axis, c.second);

        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, c.first, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t mean = utils::logical_tensor_init(
                2, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t var = utils::logical_tensor_init(
                3, graph::data_type::f32, graph::layout_type::strided);

        std::vector<graph::logical_tensor_t *> inputs {&src};
        std::vector<graph::logical_tensor_t *> outputs {&dst, &mean, &var};
        ASSERT_EQ(graph::infer_norm_output_shape(&ln_op, inputs, outputs),
                graph::status::invalid_shape);
    }
}

TEST(test_interface_shape_infer, LayerNormBeginNormAxisValid) {
    graph::op_t ln_op {0, graph::op_kind::LayerNorm, std::string("ln")};

    const std::vector<std::pair<int64_t, dims>> cases {
            {-3, {}}, {-2, {2}}, {-1, {2, 3}}, {0, {}}, {1, {2}}, {2, {2, 3}}};
    for (const auto &c : cases) {
        ln_op.set_attr<int64_t>(graph::op_attr::begin_norm_axis, c.first);

        graph::logical_tensor_t src = utils::logical_tensor_init(
                0, {2, 3, 4}, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t mean = utils::logical_tensor_init(
                2, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t var = utils::logical_tensor_init(
                3, graph::data_type::f32, graph::layout_type::strided);

        std::vector<graph::logical_tensor_t *> inputs {&src};
        std::vector<graph::logical_tensor_t *> outputs {&dst, &mean, &var};
        ASSERT_EQ(graph::infer_norm_output_shape(&ln_op, inputs, outputs),
                graph::status::success);
        ASSERT_EQ(graph::logical_tensor_wrapper_t(mean).vdims(), c.second);
        ASSERT_EQ(graph::logical_tensor_wrapper_t(var).vdims(), c.second);
    }
}

TEST(test_interface_shape_infer, InvalidShapeForMatmul) {
    graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t weight0
            = utils::logical_tensor_init(1, {8}, graph::data_type::f32);
    graph::logical_tensor_t dst0
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs0 {&src0, &weight0};
    std::vector<graph::logical_tensor_t *> outputs0 {&dst0};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs0, outputs0),
            graph::status::invalid_shape);

    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t weight1
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t dst1
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs1 {&src1, &weight1};
    std::vector<graph::logical_tensor_t *> outputs1 {&dst1};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs1, outputs1),
            graph::status::invalid_shape);

    graph::logical_tensor_t src2
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight2
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);
    graph::logical_tensor_t dst2
            = utils::logical_tensor_init(1, {4}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs2 {&src2, &weight2};
    std::vector<graph::logical_tensor_t *> outputs2 {&dst2};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs2, outputs2),
            graph::status::invalid_shape);

    graph::logical_tensor_t src3
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight3
            = utils::logical_tensor_init(1, {4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst3
            = utils::logical_tensor_init(1, {4, 1}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs3 {&src3, &weight3};
    std::vector<graph::logical_tensor_t *> outputs3 {&dst3};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs3, outputs3),
            graph::status::invalid_shape);

    graph::logical_tensor_t src4
            = utils::logical_tensor_init(1, {8, 6, 3}, graph::data_type::f32);
    graph::logical_tensor_t weight4
            = utils::logical_tensor_init(1, {4, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst4
            = utils::logical_tensor_init(1, {4, 1, 2}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs4 {&src4, &weight4};
    std::vector<graph::logical_tensor_t *> outputs4 {&dst4};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs4, outputs4),
            graph::status::invalid_shape);

    graph::logical_tensor_t src5
            = utils::logical_tensor_init(1, {8, 6}, graph::data_type::f32);
    graph::logical_tensor_t weight5
            = utils::logical_tensor_init(1, {6, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst5
            = utils::logical_tensor_init(1, {8, 2}, graph::data_type::f32);

    std::vector<graph::logical_tensor_t *> inputs5 {&src5, &weight5};
    std::vector<graph::logical_tensor_t *> outputs5 {&dst5};
    ASSERT_EQ(graph::infer_matmul_output_shape(&matmul, inputs5, outputs5),
            graph::status::invalid_shape);
}

TEST(test_interface_shape_infer, InvalidShapeForConv) {
    using dims = graph::dnnl_impl::dims;

    graph::op_t conv_op {0, graph::op_kind::Convolution, std::string("conv")};
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 2);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");

    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(2,
            {8, 16, 222, 222}, graph::data_type::f32, graph::layout_type::any);

    std::vector<graph::logical_tensor_t *> inputs {&src, &weight};
    std::vector<graph::logical_tensor_t *> outputs {&dst};
    ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            graph::status::invalid_shape);

    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1, 1});
    ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                      &conv_op, inputs, outputs),
            graph::status::invalid_shape);
}

TEST(test_interface_shape_infer, InvalidShapeForPoolBackward) {
    graph::op_t max_pool_bk_op {
            0, graph::op_kind::MaxPoolBackward, std::string("max_pool_bk_op")};
    graph::op_t avg_pool_bk_op {
            1, graph::op_kind::AvgPoolBackward, std::string("avg_pool_bk_op")};

    graph::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t dst_lt2 = utils::logical_tensor_init(
            2, graph::data_type::f32, graph::layout_type::any);

    std::vector<graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<graph::logical_tensor_t *> outputs {&dst_lt};
    std::vector<graph::logical_tensor_t *> outputs2 {&dst_lt2};

    ASSERT_EQ(graph::infer_pool_bwd_output_shape(
                      &max_pool_bk_op, inputs, outputs),
            graph::status::invalid_shape);
    ASSERT_EQ(graph::infer_pool_bwd_output_shape(
                      &avg_pool_bk_op, inputs, outputs2),
            graph::status::unimplemented);
}

TEST(test_interface_shape_infer, CanonicalizeErrorDeathTest) {
#ifndef NDEBUG
    ASSERT_DEATH(graph::canonicalize({1, 2, 3, 4}, "XXXX"), "invalid format");
#endif
}

TEST(test_interface_shape_infer, InferConvOuputShapeError) {
    graph::op_t conv_op(graph::op_kind::Convolution);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    {
        std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<graph::dim_t> wei_shape = {8, 4, 1, 1};
        graph::data_type_t dtype = graph::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<graph::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<graph::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }

    {
        conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1, 1, 1});
        std::vector<graph::dim_t> src_shape = {8, 8, 32, 32};
        std::vector<graph::dim_t> wei_shape = {8, 8, 1, 1};
        graph::data_type_t dtype = graph::data_type::f32;
        auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
        auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
        auto conv0_dst = utils::logical_tensor_init(2, src_shape, dtype);
        std::vector<graph::logical_tensor_t *> inputs {&conv0_src, &conv0_wei};
        std::vector<graph::logical_tensor_t *> outputs {&conv0_dst};
        ASSERT_EQ(graph::infer_conv_output_shape(&conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }
}

TEST(test_interface_shape_infer, InferConvBpropDataOuputShape) {
    graph::op_t conv_op(graph::op_kind::ConvolutionBackwardData);
    conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::dilations, dims {1, 1});
    conv_op.set_attr<dims>(graph::op_attr::pads_begin, dims {0, 0});
    conv_op.set_attr<dims>(graph::op_attr::pads_end, dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op.set_attr<int64_t>(graph::op_attr::groups, 1);
    conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
    conv_op.set_attr<std::string>(graph::op_attr::weights_format, "OIX");
    conv_op.set_attr<dims>(graph::op_attr::dst_shape, dims {8, 3, 224, 224});

    // prepare logical tensor
    graph::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, graph::data_type::f32);
    graph::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, graph::data_type::f32);

    conv_op.add_input(diff_dst);
    conv_op.add_input(weights);
    conv_op.add_output(diff_src);
    std::vector<graph::logical_tensor_t *> inputs {&diff_src, &weights};
    std::vector<graph::logical_tensor_t *> outputs {&diff_dst};
    {
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "XXXXX");
        ASSERT_EQ(graph::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                graph::status::unimplemented);
    }
    {
        conv_op.set_attr<dims>(graph::op_attr::strides, dims {1, 1, 1, 1, 1});
        conv_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        ASSERT_EQ(graph::infer_conv_bprop_data_output_shape(
                          &conv_op, inputs, outputs),
                graph::status::invalid_shape);
    }
}

TEST(test_interface_shape_infer, InferConvtransposeNcxOixError) {
    graph::op_t conv {graph::op_kind::ConvTransposeBackwardData,
            graph::op_t::kind2str(graph::op_kind::ConvTransposeBackwardData)};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads_begin = {0, 0}; // empty pads_begin
    std::vector<int64_t> pads_end = {0, 0}; // empty pads_end
    std::vector<int64_t> dilations = {2, 2};
    std::string data_format = "NCX";
    std::string filter_format = "OIX";
    std::string auto_pad = "VALID";
    int64_t groups = 1;
    int64_t id = 0;
    conv.set_attr(graph::op_attr::strides, strides);
    conv.set_attr(graph::op_attr::pads_begin, pads_begin);
    conv.set_attr(graph::op_attr::pads_end, pads_end);
    conv.set_attr(graph::op_attr::dilations, dilations);
    conv.set_attr(graph::op_attr::auto_pad, auto_pad);
    conv.set_attr(graph::op_attr::data_format, data_format);
    conv.set_attr(graph::op_attr::weights_format, filter_format);
    conv.set_attr(graph::op_attr::groups, groups);

    auto lt_data = utils::logical_tensor_init(
            id++, {1, 1, 16, 16}, graph::data_type::f32);
    auto lt_weight = utils::logical_tensor_init(
            id++, {1, 1, 4, 4}, graph::data_type::f32);
    std::vector<int64_t> expected_out_shape {1, 1, 10, 10};

    auto lt_o = utils::logical_tensor_init(
            id++, graph::data_type::f32, graph::layout_type::strided);
    std::vector<graph::logical_tensor_t *> lt_in {&lt_data, &lt_weight};
    std::vector<graph::logical_tensor_t *> lt_out {&lt_o};
    {
        conv.set_attr<int64_t>(graph::op_attr::groups, 2);
        ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                graph::status::invalid_shape);
    }

    {
        conv.set_attr(graph::op_attr::groups, groups);
        conv.set_attr<graph::dims>(graph::op_attr::strides, {1, 1, 1, 1, 1});
        ASSERT_EQ(graph::infer_convtranspose_bprop_data_output_shape(
                          &conv, lt_in, lt_out),
                graph::status::invalid_shape);
    }
}

// Parameterized test for MatMul shape inference with optional post binary
struct matmul_shape_infer_params_t {
    dims src_shape;
    dims wei_shape;
    dims post_binary_shape; // empty if no post binary
    dims expected_dst_shape;
    graph::status_t expected_status;
    std::string test_name;
};

class matmul_shape_infer_t
    : public ::testing::TestWithParam<matmul_shape_infer_params_t> {
public:
    void TestMatMulShapeInfer() {
        const auto &params = ::testing::TestWithParam<
                matmul_shape_infer_params_t>::GetParam();

        graph::op_t matmul {0, graph::op_kind::MatMul, std::string("matmul")};

        graph::logical_tensor_t src = utils::logical_tensor_init(
                0, params.src_shape, graph::data_type::f32);
        graph::logical_tensor_t wei = utils::logical_tensor_init(
                1, params.wei_shape, graph::data_type::f32);
        graph::logical_tensor_t dst
                = utils::logical_tensor_init(2, graph::data_type::f32);

        std::vector<graph::logical_tensor_t *> inputs {&src, &wei};

        // Add post binary input if specified
        graph::logical_tensor_t post_binary;
        if (!params.post_binary_shape.empty()) {
            post_binary = utils::logical_tensor_init(
                    3, params.post_binary_shape, graph::data_type::f32);
            inputs.push_back(&post_binary);
        }

        std::vector<graph::logical_tensor_t *> outputs {&dst};

        auto status
                = graph::infer_matmul_output_shape(&matmul, inputs, outputs);
        ASSERT_EQ(status, params.expected_status);

        // If successful, verify the inferred shape
        if (status == graph::status::success) {
            ASSERT_EQ(dst.ndims,
                    static_cast<int32_t>(params.expected_dst_shape.size()));
            for (size_t i = 0; i < params.expected_dst_shape.size(); ++i) {
                ASSERT_EQ(dst.dims[i], params.expected_dst_shape[i]);
            }
        }
    }
};

TEST_P(matmul_shape_infer_t, TestMatMulShapeInfer) {
    TestMatMulShapeInfer();
}

INSTANTIATE_TEST_SUITE_P(test_interface_shape_infer, matmul_shape_infer_t,
        ::testing::Values(
                // Basic 2D matmul: [M, K] x [K, N] = [M, N]
                matmul_shape_infer_params_t {{8, 6}, {6, 4}, {}, {8, 4},
                        graph::status::success, "2D_matmul"},

                // 3D matmul with batch: [B, M, K] x [B, K, N] = [B, M, N]
                matmul_shape_infer_params_t {{2, 8, 6}, {2, 6, 4}, {},
                        {2, 8, 4}, graph::status::success, "3D_batch_matmul"},

                // 1D x 2D: [K] x [K, N] = [N]
                matmul_shape_infer_params_t {{6}, {6, 4}, {}, {4},
                        graph::status::success, "1D_x_2D"},

                // 1D x 3D: [K] x [B, K, N] = [B, N]
                matmul_shape_infer_params_t {{6}, {2, 6, 4}, {}, {2, 4},
                        graph::status::success, "1D_x_3D"},

                // 2D x 1D: [M, K] x [K] = [M]
                matmul_shape_infer_params_t {{8, 6}, {6}, {}, {8},
                        graph::status::success, "2D_x_1D"},

                // 3D x 1D: [B, M, K] x [K] = [B, M]
                matmul_shape_infer_params_t {{2, 8, 6}, {6}, {}, {2, 8},
                        graph::status::success, "3D_x_1D"},

                // With post binary add (same shape as output)
                matmul_shape_infer_params_t {{8, 6}, {6, 4}, {8, 4}, {8, 4},
                        graph::status::success,
                        "matmul_with_binary_same_shape"},

                // With post binary add (broadcast)
                matmul_shape_infer_params_t {{8, 6}, {6, 4}, {1, 4}, {8, 4},
                        graph::status::success, "matmul_with_binary_broadcast"},

                // With post binary add (broadcast batch)
                matmul_shape_infer_params_t {{2, 8, 6}, {2, 6, 4}, {4},
                        {2, 8, 4}, graph::status::success,
                        "matmul_with_binary_broadcast_batch"},

                // Invalid: incompatible K dimension
                matmul_shape_infer_params_t {{8, 6}, {4, 4}, {}, {},
                        graph::status::invalid_shape, "invalid_K_mismatch"},

                // Invalid: 1D x 1D (not supported)
                matmul_shape_infer_params_t {{6}, {4}, {}, {},
                        graph::status::invalid_shape, "invalid_1D_x_1D"},

                // Invalid: incompatible batch dimensions
                matmul_shape_infer_params_t {{2, 8, 6}, {3, 6, 4}, {}, {},
                        graph::status::invalid_shape,
                        "invalid_batch_mismatch"}));
