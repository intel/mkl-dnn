/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/// @example cnn_inference_int8.cpp
/// > Annotated version: @ref cnn_inference_int8_cpp

/// @page cnn_inference_int8_cpp_brief
/// @brief This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with
/// int8 data type.

/// @page cnn_inference_int8_cpp CNN int8 inference example
/// \copybrief cnn_inference_int8_cpp_brief
///
/// > Example code: @ref cnn_inference_int8.cpp

#include <stdexcept>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void simple_net_int8(engine::kind engine_kind) {
    auto eng = engine(engine_kind, 0);
    stream s(eng);

    const int batch = 8;

    /// Configure tensor shapes
    /// @snippet cnn_inference_int8.cpp Configure tensor shapes
    //[Configure tensor shapes]
    // AlexNet: conv3
    // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
    // strides: {1, 1}
    memory::dims conv_src_tz = {batch, 256, 13, 13};
    memory::dims conv_weights_tz = {384, 256, 3, 3};
    memory::dims conv_bias_tz = {384};
    memory::dims conv_dst_tz = {batch, 384, 13, 13};
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};
    //[Configure tensor shapes]

    /// Next, the example configures the scales used to quantize f32 data
    /// into int8. For this example, the scaling value is chosen as an
    /// arbitrary number, although in a realistic scenario, it should be
    /// calculated from a set of precomputed values as previously mentioned.
    /// @snippet cnn_inference_int8.cpp Choose scaling factors
    //[Choose scaling factors]
    // Choose scaling factors for input, weight and output
    std::vector<float> src_scales = {1.8f};
    std::vector<float> weight_scales = {2.0f};
    std::vector<float> dst_scales = {0.55f};

    //[Choose scaling factors]

    /// The *source, weights* and *destination* datasets use the single-scale
    /// format with mask set to '0'.
    /// @snippet cnn_inference_int8.cpp Set scaling mask
    //[Set scaling mask]
    const int src_mask = 0;
    const int weight_mask = 0;
    const int dst_mask = 0;
    //[Set scaling mask]

    // Allocate input and output buffers for user data
    std::vector<float> user_src(batch * 256 * 13 * 13);
    std::vector<float> user_dst(batch * 384 * 13 * 13);

    // Allocate and fill buffers for weights and bias
    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    /// Create the memory primitives for user data (source, weights, and bias).
    /// The user data will be in its original 32-bit floating point format.
    /// @snippet cnn_inference_int8.cpp Allocate buffers
    //[Allocate buffers]
    auto user_src_memory = memory(
            {{conv_src_tz}, memory::data_type::f32, memory::format_tag::nhwc},
            eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    auto user_weights_memory
            = memory({{conv_weights_tz}, memory::data_type::f32,
                             memory::format_tag::oihw},
                    eng);
    write_to_dnnl_memory(conv_weights.data(), user_weights_memory);
    auto user_bias_memory = memory(
            {{conv_bias_tz}, memory::data_type::f32, memory::format_tag::x},
            eng);
    write_to_dnnl_memory(conv_bias.data(), user_bias_memory);
    //[Allocate buffers]

    /// Create a memory descriptor for each convolution parameter.
    /// The convolution data uses 8-bit integer values, so the memory
    /// descriptors are configured as:
    ///
    /// * 8-bit unsigned (u8) for source and destination.
    /// * 8-bit signed (s8) for weights.
    ///
    ///  > **Note**
    ///  > The destination type is chosen as *unsigned* because the
    ///  > convolution applies a ReLU operation where data results \f$\geq 0\f$.
    ///  > **Note**
    ///  > Bias does not support quantization.
    /// @snippet cnn_inference_int8.cpp Create convolution memory descriptors
    //[Create convolution memory descriptors]
    auto conv_src_md = memory::desc(
            {conv_src_tz}, memory::data_type::u8, memory::format_tag::any);
    auto conv_bias_md = memory::desc(
            {conv_bias_tz}, memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc(
            {conv_weights_tz}, memory::data_type::s8, memory::format_tag::any);
    auto conv_dst_md = memory::desc(
            {conv_dst_tz}, memory::data_type::u8, memory::format_tag::any);
    //[Create convolution memory descriptors]

    /// Configuring int8-specific parameters in an int8 primitive is done
    /// via the Attributes Primitive. Create an attributes object for the
    /// convolution and configure it accordingly.
    /// @snippet cnn_inference_int8.cpp Configure scaling
    //[Configure scaling]
    primitive_attr conv_attr;
    conv_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    conv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
    conv_attr.set_scales_mask(DNNL_ARG_DST, dst_mask);

    // Prepare dst scales
    auto dst_scale_md
            = memory::desc({1}, memory::data_type::f32, memory::format_tag::x);
    auto dst_scale_memory = memory(dst_scale_md, eng);
    write_to_dnnl_memory(dst_scales.data(), dst_scale_memory);
    //[Configure scaling]

    /// The ReLU layer from Alexnet is executed through the PostOps feature. Create
    /// a PostOps object and configure it to execute an _eltwise relu_ operation.
    /// @snippet cnn_inference_int8.cpp Configure post-ops
    //[Configure post-ops]
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);
    //[Configure post-ops]

    // check if int8 convolution is supported
    try {
        convolution_forward::primitive_desc(eng, prop_kind::forward,
                algorithm::convolution_direct, conv_src_md, conv_weights_md,
                conv_bias_md, conv_dst_md, conv_strides, conv_padding,
                conv_padding, conv_attr);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type)
            throw example_allows_unimplemented {
                    "No int8 convolution implementation is available for this "
                    "platform.\n"
                    "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    /// Create a primitive descriptor passing the int8 memory descriptors
    /// and int8 attributes to the constructor. The primitive
    /// descriptor for the convolution will contain the specific memory
    /// formats for the computation.
    /// @snippet cnn_inference_int8.cpp Create convolution primitive descriptor
    //[Create convolution primitive descriptor]
    auto conv_prim_desc = convolution_forward::primitive_desc(eng,
            prop_kind::forward, algorithm::convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            conv_padding, conv_padding, conv_attr);
    //[Create convolution primitive descriptor]

    /// Create a memory for each of the convolution's data input
    /// parameters (source, bias, weights, and destination). Using the convolution
    /// primitive descriptor as the creation parameter enables oneDNN
    /// to configure the memory formats for the convolution.
    ///
    /// Scaling parameters are passed to the reorder primitive via the attributes
    /// primitive.
    ///
    /// User memory must be transformed into convolution-friendly memory
    /// (for int8 and memory format). A reorder layer performs the data
    /// transformation from f32 (the original user data) into int8 format
    /// (the data used for the convolution). In addition, the reorder
    /// transforms the user data into the required memory format (as explained
    /// in the simple_net example).
    ///
    /// @snippet cnn_inference_int8.cpp Quantize data and weights
    //[Quantize data and weights]
    auto conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
    primitive_attr src_attr;
    src_attr.set_scales_mask(DNNL_ARG_DST, src_mask);
    auto src_scale_md
            = memory::desc({1}, memory::data_type::f32, memory::format_tag::x);
    auto src_scale_memory = memory(src_scale_md, eng);
    write_to_dnnl_memory(src_scales.data(), src_scale_memory);
    auto src_reorder_pd
            = reorder::primitive_desc(eng, user_src_memory.get_desc(), eng,
                    conv_src_memory.get_desc(), src_attr);
    auto src_reorder = reorder(src_reorder_pd);
    src_reorder.execute(s,
            {{DNNL_ARG_FROM, user_src_memory}, {DNNL_ARG_TO, conv_src_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, src_scale_memory}});

    auto conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
    primitive_attr weight_attr;
    weight_attr.set_scales_mask(DNNL_ARG_DST, weight_mask);
    auto wei_scale_md
            = memory::desc({1}, memory::data_type::f32, memory::format_tag::x);
    auto wei_scale_memory = memory(wei_scale_md, eng);
    write_to_dnnl_memory(weight_scales.data(), wei_scale_memory);
    auto weight_reorder_pd
            = reorder::primitive_desc(eng, user_weights_memory.get_desc(), eng,
                    conv_weights_memory.get_desc(), weight_attr);
    auto weight_reorder = reorder(weight_reorder_pd);
    weight_reorder.execute(s,
            {{DNNL_ARG_FROM, user_weights_memory},
                    {DNNL_ARG_TO, conv_weights_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, wei_scale_memory}});

    auto conv_bias_memory = memory(conv_prim_desc.bias_desc(), eng);
    write_to_dnnl_memory(conv_bias.data(), conv_bias_memory);
    //[Quantize data and weights]

    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    /// Create the convolution primitive and add it to the net. The int8 example
    /// computes the same Convolution +ReLU layers from AlexNet simple-net.cpp
    /// using the int8 and PostOps approach. Although performance is not
    /// measured here, in practice it would require less computation time to achieve
    /// similar results.
    /// @snippet cnn_inference_int8.cpp Create convolution primitive
    //[Create convolution primitive]
    auto conv = convolution_forward(conv_prim_desc);
    conv.execute(s,
            {{DNNL_ARG_SRC, conv_src_memory},
                    {DNNL_ARG_WEIGHTS, conv_weights_memory},
                    {DNNL_ARG_BIAS, conv_bias_memory},
                    {DNNL_ARG_DST, conv_dst_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_memory}});
    //[Create convolution primitive]

    /// @page cnn_inference_int8_cpp
    /// Finally, *dst memory* may be dequantized from int8 into the original
    /// f32 format. Create a memory primitive for the user data in the original
    /// 32-bit floating point format and then apply a reorder to transform the
    /// computation output data.
    /// @snippet cnn_inference_int8.cpp Dequantize the result
    ///[Dequantize the result]
    auto user_dst_memory = memory(
            {{conv_dst_tz}, memory::data_type::f32, memory::format_tag::nhwc},
            eng);
    write_to_dnnl_memory(user_dst.data(), user_dst_memory);
    primitive_attr dst_attr;
    dst_attr.set_scales_mask(DNNL_ARG_SRC, dst_mask);
    auto dst_reorder_pd
            = reorder::primitive_desc(eng, conv_dst_memory.get_desc(), eng,
                    user_dst_memory.get_desc(), dst_attr);
    auto dst_reorder = reorder(dst_reorder_pd);
    dst_reorder.execute(s,
            {{DNNL_ARG_FROM, conv_dst_memory}, {DNNL_ARG_TO, user_dst_memory},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dst_scale_memory}});
    //[Dequantize the result]

    s.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(
            simple_net_int8, parse_engine_kind(argc, argv));
}
