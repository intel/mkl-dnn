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
#ifndef GRAPH_INTERFACE_OP_DEF_HPP
#define GRAPH_INTERFACE_OP_DEF_HPP

#include <limits>
#include <set>
#include <vector>

#include "graph/interface/op_def_constraint.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/shape_infer.hpp"

namespace dnnl {
namespace impl {
namespace graph {

DNNL_GRAPH_OP_SCHEMA(Abs, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(AbsBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs({0, 1})
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints("T1",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_type_constraints("T2",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_type_constraints("T3",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(AvgPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, true, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape)
                .set_op_def_constraint_function(check_pads))

DNNL_GRAPH_OP_SCHEMA(AvgPoolBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "src_shape", "T1")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, true, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::src_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T1", {data_type::s32})
                .set_shape_inference_function(infer_pool_bwd_output_shape)
                .set_op_def_constraint_function(check_avgpool_bwd_input_shape)
                .set_op_def_constraint_function(check_pads))

DNNL_GRAPH_OP_SCHEMA(BatchNormInference, 1,
        op_schema_t()
                .set_num_inputs(5)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_input(2, "beta", "T2")
                .set_input(3, "mean", "T2")
                .set_input(4, "variance", "T2")
                .set_output(0, "dst", "T1")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormForwardTraining, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_num_outputs(5)
                .set_input(0, "src", "T1")
                .set_input(1, "mean", "T2")
                .set_input(2, "variance", "T2")
                .set_input(3, "gamma", "T2")
                .set_input(4, "beta", "T2")
                .set_output(0, "dst", "T1")
                .set_output(1, "running_mean", "T2")
                .set_output(2, "running_variance", "T2")
                .set_output(3, "batch_mean", "T2")
                .set_output(4, "batch_variance", "T2")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::momentum, false, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_fwd_train_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormTrainingBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "mean", "T2")
                .set_input(3, "variance", "T2")
                .set_input(4, "gamma", "T2")
                .set_output(0, "diff_src", "T1")
                .set_output(1, "diff_gamma", "T2")
                .set_output(2, "diff_beta", "T2")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_bwd_output_shape)
                .set_op_def_constraint_function(check_bn_data_type))

DNNL_GRAPH_OP_SCHEMA(BiasAdd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "bias", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_add_output_shape))

DNNL_GRAPH_OP_SCHEMA(BiasAddBackward, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_output(0, "diff_bias", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_backprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(Clamp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::min, true, attribute_kind::f)
                .set_attr(op_attr::max, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ClampBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::min, true, attribute_kind::f)
                .set_attr(op_attr::max, true, attribute_kind::f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(1)
                .set_input(0, "src_i", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(Convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackwardData, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T1")
                .set_input(1, "weights", "T1")
                .set_input(2, "dst_shape", "T2")
                .set_output(0, "diff_src", "T1")
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_attr(op_attr::dst_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_data_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_data_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackwardWeights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "weights_shape", "T2")
                .set_output(0, "diff_weights", "T1")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_weights_weights_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_convtranspose_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackwardData, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "weights", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_convtranspose_bprop_data_output_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackwardWeights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "weights_shape", "T2")
                .set_output(0, "diff_weights", "T1")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_convtranspose_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_op_def_constraint_function(
                        check_conv_bwd_weights_weights_shape)
                .set_op_def_constraint_function(check_pads)
                .SET_CONVTRANSPOSE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Divide, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Dropout, 1,
        op_schema_t()
                .set_num_inputs(4)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "src", "T")
                .set_input(1, "seed", "T_seed")
                .set_input(2, "offset", "T_offset")
                .set_input(3, "probability", "T_p")
                .set_output(0, "dst", "T")
                .set_output(1, "mask", "T_mask")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T_seed", {data_type::s64})
                .set_type_constraints("T_offset", {data_type::s64})
                .set_type_constraints("T_p", {data_type::f32})
                .set_type_constraints("T_mask", {data_type::u8})
                .set_shape_inference_function(infer_dropout_output_shape))

DNNL_GRAPH_OP_SCHEMA(Elu, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(EluBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(End, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(0)
                .set_input(0, "src", "T")
                .set_type_constraints("T",
                        {data_type::f32, data_type::f16, data_type::bf16,
                                data_type::s8, data_type::u8, data_type::s32,
                                data_type::undef})
                .set_shape_inference_function(infer_dummy_output_shape))

DNNL_GRAPH_OP_SCHEMA(Exp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::mode, false, attribute_kind::s, "gelu_erf",
                        {"gelu_erf", "gelu_tanh"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELUBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::mode, false, attribute_kind::s, "gelu_erf",
                        {"gelu_erf", "gelu_tanh"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GenIndex, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GreaterEqual, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints("T1",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(GroupNorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_input(2, "beta", "T2")
                .set_output(0, "dst", "T1")
                .set_output(1, "mean", "T2")
                .set_output(2, "variance", "T2")
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::groups, true, attribute_kind::i)
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_groupnorm_output_shape)
                .set_op_def_constraint_function(check_norm_data_type)
                .set_op_def_constraint_function(check_ln_gn_fwd_outputs_num))

DNNL_GRAPH_OP_SCHEMA(HardSigmoid, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSigmoidBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwishBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Interpolate, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "sizes", "T2")
                .set_output(0, "dst", "T1")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_interpolate_output_shape)
                .set_op_def_constraint_function(check_interpolate_sizes_scales))

DNNL_GRAPH_OP_SCHEMA(InterpolateBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "sizes", "T2")
                .set_output(0, "diff_src", "T1")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_interpolate_sizes_scales))

DNNL_GRAPH_OP_SCHEMA(LayerNorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_input(2, "beta", "T2")
                .set_output(0, "dst", "T1")
                .set_output(1, "mean", "T2")
                .set_output(2, "variance", "T2")
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_output_shape)
                .set_op_def_constraint_function(check_norm_data_type)
                .set_op_def_constraint_function(check_ln_gn_fwd_outputs_num))

DNNL_GRAPH_OP_SCHEMA(LayerNormBackward, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "src", "T1")
                .set_input(1, "diff_dst", "T1")
                .set_input(2, "mean", "T2")
                .set_input(3, "variance", "T2")
                .set_input(4, "gamma", "T2")
                .set_input(5, "beta", "T2")
                .set_output(0, "diff_src", "T1")
                .set_output(1, "diff_gamma", "T2")
                .set_output(2, "diff_beta", "T2")
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_bprop_output_shape)
                .set_op_def_constraint_function(check_norm_data_type)
                .set_op_def_constraint_function(check_ln_bwd_use_affine))

DNNL_GRAPH_OP_SCHEMA(LeakyReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::alpha, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Log, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmaxBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T")
                .set_input(1, "dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MatMul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "weights", "T")
                .set_input(2, "bias", "T")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_matmul_output_shape)
                .set_op_def_constraint_function(check_matmul_dtype)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Maximum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs({0, 1})
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(MaxPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape)
                .set_op_def_constraint_function(check_pads)
                .set_op_def_constraint_function(check_maxpool_dilations))

DNNL_GRAPH_OP_SCHEMA(MaxPoolBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_bwd_output_shape)
                .set_op_def_constraint_function(check_pads)
                .set_op_def_constraint_function(check_maxpool_dilations))

DNNL_GRAPH_OP_SCHEMA(Minimum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs({0, 1})
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Mish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MishBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Multiply, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs({0, 1})
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Pow, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::beta, true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLU, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "slope", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::per_channel_broadcast, false,
                        attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLUBackward, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(2)
                .set_input(0, "src", "T")
                .set_input(1, "slope", "T")
                .set_input(2, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_output(1, "diff_slope", "T")
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_prelu_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReduceL1, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceL2, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMax, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMean, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMin, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceProd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceSum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "axes", "T2")
                .set_output(0, "dst", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .set_op_def_constraint_function(check_reduce_axes)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReLUBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Round, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Select, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_commutative_inputs({1, 2})
                .set_input(0, "cond", "T1")
                .set_input(1, "src_0", "T2")
                .set_input(2, "src_1", "T2")
                .set_output(0, "dst", "T2")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints("T1", {data_type::boolean})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_select_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sigmoid, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SigmoidBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMax, 1,
        op_schema_t()
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(1)
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_output(1, "stats", "T3")
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::mode, false, attribute_kind::s, "none",
                        {"none", "inf_as_zero"})
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T3", {data_type::f32})
                .set_shape_inference_function(infer_softmax_output_shape)
                .set_op_def_constraint_function(check_softmax_dtype))

DNNL_GRAPH_OP_SCHEMA(SoftMaxBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "diff_dst", "T1")
                .set_input(1, "dst", "T1")
                .set_output(0, "diff_src", "T2")
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_softmax_bwd_output_dtype))

DNNL_GRAPH_OP_SCHEMA(SoftPlus, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::beta, false, attribute_kind::f, 1.f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlusBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::beta, false, attribute_kind::f, 1.f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sqrt, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SqrtBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Square, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SquaredDifference, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T")
                .set_input(1, "src_1", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Subtract, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src_0", "T1")
                .set_input(1, "src_1", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                .set_type_constraints("T1",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_type_constraints("T2",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_type_constraints("T3",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Tanh, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TanhBackward, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src/dst", "T")
                .set_input(1, "diff_dst", "T")
                .set_output(0, "diff_src", "T")
                .set_attr(op_attr::use_dst, false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Wildcard, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_outputs_option(op_schema_t::param_num_option::variadic)
                .set_num_outputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_input(0, "src", "any")
                .set_output(0, "dst", "any")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Quantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                // for symmetric quantization or fp8 quantization, zps is not required.
                .set_attr(op_attr::zps, false, attribute_kind::is)
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints("T2",
                        {data_type::u8, data_type::s8, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Dequantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                // for symmetric quantization or fp8 quantization, zps is not required.
                .set_attr(op_attr::zps, false, attribute_kind::is)
                .set_type_constraints("T1",
                        {data_type::u8, data_type::s8, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_type_constraints("T2", {data_type::f32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Reorder, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TypeCast, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_output(0, "dst", "T2")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(check_typecast_data_type))

DNNL_GRAPH_OP_SCHEMA(StaticReshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::shape, true, attribute_kind::is)
                .set_attr(op_attr::special_zero, true, attribute_kind::b)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_reshape_output_shape))

DNNL_GRAPH_OP_SCHEMA(StaticTranspose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_attr(op_attr::order, true, attribute_kind::is)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_transpose_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicQuantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "scales", "T1")
                .set_input(2, "zps", "T2")
                .set_output(0, "dst", "T3")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::mask, false, attribute_kind::i)
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints(
                        "T2", {data_type::u8, data_type::s8, data_type::s32})
                .set_type_constraints("T3",
                        {data_type::u8, data_type::s8, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(
                        check_dyn_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(DynamicDequantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "scales", "T2")
                .set_input(2, "zps", "T3")
                .set_output(0, "dst", "T2")
                .set_attr(
                        op_attr::qtype, false, attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::mask, false, attribute_kind::i)
                .set_attr(op_attr::group_shape, false, attribute_kind::is)
                .set_type_constraints("T1",
                        {data_type::u8, data_type::s8, data_type::s4,
                                data_type::u4, data_type::f8_e5m2,
                                data_type::f8_e4m3})
                .set_type_constraints(
                        "T2", {data_type::bf16, data_type::f16, data_type::f32})
                .set_type_constraints("T3",
                        {data_type::u4, data_type::s4, data_type::u8,
                                data_type::s8, data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_op_def_constraint_function(
                        check_dyn_quant_dequant_scales_zps))

DNNL_GRAPH_OP_SCHEMA(Reciprocal, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "T")
                .set_output(0, "dst", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(RMSNorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "src", "T1")
                .set_input(1, "gamma", "T2")
                .set_output(0, "dst", "T1")
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_norm_output_shape)
                .set_op_def_constraint_function(check_norm_data_type))

// Definitions of internal ops
#define SET_ATTR_IS_CONSTANT \
    set_attr(op_attr::is_constant, false, attribute_kind::b, false)

#define SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS \
    set_attr(op_attr::strides, true, attribute_kind::is) \
            .set_attr(op_attr::pads_begin, true, attribute_kind::is) \
            .set_attr(op_attr::pads_end, true, attribute_kind::is) \
            .set_attr(op_attr::dilations, true, attribute_kind::is) \
            .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None", \
                    {"None", "SAME_UPPER", "SAME_LOWER", "VALID"}) \
            .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1) \
            .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC", \
                    {"NXC", "NCX"}) \
            .set_attr(op_attr::weights_format, false, attribute_kind::s, \
                    "XOI", {"XOI", "IOX", "OIX"})

template <typename T>
op_schema_t get_op_schema();

DNNL_GRAPH_OP_SCHEMA(_mul_scales, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "x")
                .set_input(1, "scales")
                .set_output(0, "y")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::mask, false, attribute_kind::i, int64_t(0))
                .set_attr(op_attr::group_shape, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::data_type, false, attribute_kind::i,
                        int64_t(0))
                .set_attr(op_attr::scales, false, attribute_kind::fs,
                        std::vector<float>())
                .set_attr(op_attr::with_runtime_scales, false,
                        attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape))

// We define this op to convert a host scalar tensor to a device tensor, as some
// kernels/primitives may not accept host scalar tensor as input.
DNNL_GRAPH_OP_SCHEMA(_host_scalar, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "scalar")
                .set_output(0, "output")
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_dnnl_host_scalar_output_shape))

DNNL_GRAPH_OP_SCHEMA(_constant_scales, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output")
                .set_attr(op_attr::scales, true, attribute_kind::fs)
                .set_attr(op_attr::shape, true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape))

DNNL_GRAPH_OP_SCHEMA(_add_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_input(1, "zps")
                .set_output(0, "y")
                .set_attr(op_attr::mask, false, attribute_kind::i, int64_t(0))
                .set_attr(op_attr::zps, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps, false, attribute_kind::b,
                        false)
                .set_attr(op_attr::group_shape, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::data_type, false, attribute_kind::i,
                        int64_t(0))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_sub_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_input(1, "zps")
                .set_output(0, "y")
                .set_attr(op_attr::mask, false, attribute_kind::i, int64_t(0))
                .set_attr(op_attr::zps, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps, false, attribute_kind::b,
                        false)
                .set_attr(op_attr::group_shape, false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr(op_attr::data_type, false, attribute_kind::i,
                        int64_t(0))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_constant_zps, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output")
                .set_attr(op_attr::zps, true, attribute_kind::is)
                .set_attr(op_attr::shape, true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape))

DNNL_GRAPH_OP_SCHEMA(_dropout, 1,
        op_schema_t()
                .set_num_inputs(4)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "src")
                .set_input(1, "seed")
                .set_input(2, "offset")
                .set_input(3, "probability")
                .set_output(0, "dst")
                .set_output(1, "mask")
                .set_shape_inference_function(infer_dropout_output_shape))

// The logical axes will be permuted in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[permutation[i]] = dims()[i];
//
// Note: the permutation attr in dnnl_permute is quite different from the order
// attr in dnnl_transpose. The later one is inherited from StaticTranspose op
// and are used in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[i] = dims()[order[i]];
DNNL_GRAPH_OP_SCHEMA(_permute, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::permutation, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_permute_output_shape))

DNNL_GRAPH_OP_SCHEMA(_to_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose, false, attribute_kind::b,
                        false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_to_group_output_shape))

// This op is used for grouped conv/deconv backward weight to convert a [g,
// oc/g, ic, kh, kw] shaped weight tensor to a [oc, ic, kh, kw] weight tensor.
// The former shaped weight tensor is required by oneDNN primitive, but the
// later one is required by oneDNN Graph users
DNNL_GRAPH_OP_SCHEMA(_from_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::groups, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose, false, attribute_kind::b,
                        false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_from_group_output_shape))

DNNL_GRAPH_OP_SCHEMA(_unsqueeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::axes, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_unsqueeze_output_shape))

DNNL_GRAPH_OP_SCHEMA(_squeeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x")
                .set_output(0, "y")
                .set_attr(op_attr::axes, false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_squeeze_output_shape))

DNNL_GRAPH_OP_SCHEMA(_reshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data")
                .set_output(0, "output")
                .set_attr(op_attr::shape, true, attribute_kind::is)
                .set_attr(op_attr::special_zero, true, attribute_kind::b)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_static_reshape_output_shape))

DNNL_GRAPH_OP_SCHEMA(_transpose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data")
                .set_output(0, "output")
                .set_attr(op_attr::order, true, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_static_transpose_output_shape))

DNNL_GRAPH_OP_SCHEMA(_identity, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data")
                .set_output(0, "output")
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "filter")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Convolution.
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_conv_output_shape))

DNNL_GRAPH_OP_SCHEMA(_convtranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "weight")
                .set_input(2, "bias")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTranspose.
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_output_shape))

DNNL_GRAPH_OP_SCHEMA(_convtranspose_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "filter")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTransposeBackwardData.
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_data_output_shape))

DNNL_GRAPH_OP_SCHEMA(_convtranspose_bwd_weights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_input(2, "filter_shape")
                .set_output(0, "filter_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvTransposeBackwardWeights.
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_weight_output_shape))

DNNL_GRAPH_OP_SCHEMA(_pool, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .set_output(2, "workspace")
                // Attributes inherited from MaxPool and AvgPool.
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, false, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::rounding_type, false, attribute_kind::s,
                        "floor")
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::kind, true, attribute_kind::s)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::is_training, false, attribute_kind::b)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(_pool_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "output_forward_indices")
                .set_input(2, "forward_src")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::strides, true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, true, attribute_kind::is)
                .set_attr(op_attr::pads_end, true, attribute_kind::is)
                .set_attr(op_attr::exclude_pad, false, attribute_kind::b)
                .set_attr(op_attr::kernel, true, attribute_kind::is)
                .set_attr(op_attr::auto_pad, false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::dilations, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::src_shape, true, attribute_kind::is)
                // New added attributes
                .set_attr(op_attr::kind, true, attribute_kind::s)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_pool_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(_prelu, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "slope")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from PReLU
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::per_channel_broadcast, false,
                        attribute_kind::b, true)
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_prelu_bwd, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(3)
                .set_input(0, "input_forward")
                .set_input(1, "slope")
                .set_input(2, "output_delta")
                .set_output(0, "input_delta")
                .set_output(1, "slope_delta")
                .set_output(2, "scratchpad")
                // Attributes inherited from PReLUBackward
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_prelu_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(_bn_folding, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({5, 6}))
                .set_num_outputs(3)
                .set_input(0, "weight")
                .set_input(1, "bias")
                .set_input(2, "gamma")
                .set_input(3, "beta")
                .set_input(4, "mean")
                .set_input(5, "variance")
                .set_output(0, "updated_weight")
                .set_output(1, "updated_bias")
                .set_output(2, "scratchpad")
                // No corresponding frontend op
                // Attributes
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::weights_format, false, attribute_kind::s,
                        "XIO", {"XIO", "OIX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_bn_folding_output_shape))

DNNL_GRAPH_OP_SCHEMA(_conv_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "weight")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from ConvolutionBackwardData.
                .set_attr(op_attr::output_padding, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .set_attr(op_attr::dst_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_data_output_shape))

DNNL_GRAPH_OP_SCHEMA(_conv_bwd_weights, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_output(0, "weight_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::weights_shape, false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_weight_output_shape))

// Note: if `is_training` is False, the `gamma` and `beta` are the second and
// third input (required), while `is_training` is True, the `gamma` and `beta`
// are the last two inputs (optional).
DNNL_GRAPH_OP_SCHEMA(_batchnorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 6, 7}))
                .set_input(0, "input")
                .set_input(1, "gamma")
                .set_input(2, "beta")
                .set_input(3, "mean")
                .set_input(4, "variance")
                .set_output(0, "output")
                .set_output(1, "running mean")
                .set_output(2, "running variance")
                .set_output(3, "batch mean")
                .set_output(4, "batch variance")
                .set_output(5, "scratchpad")
                .set_output(6, "workspace")
                // Attributes inherited from BatchNormInference and
                // BatchNormForwardTraining op
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::momentum, false, attribute_kind::f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::is_training, false, attribute_kind::b)
                .set_attr(op_attr::fuse_relu, false, attribute_kind::b)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_batchnorm_output_shape))

DNNL_GRAPH_OP_SCHEMA(_batchnorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 4}))
                .set_input(0, "input")
                .set_input(1, "output_delta")
                .set_input(2, "mean")
                .set_input(3, "variance")
                .set_input(4, "gamma")
                .set_output(0, "input_delta")
                .set_output(1, "gamma_delta")
                .set_output(2, "beta_delta")
                .set_output(3, "scratchpad")
                .set_attr(op_attr::epsilon, true, attribute_kind::f)
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_dnnl_batchnorm_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(_resampling_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "output_delta")
                .set_input(2, "sizes")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_sum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_binary, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "a")
                .set_input(1, "b")
                .set_input(2, "cond")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front binary ops (Add, Multiply,
                // ...).
                .set_attr(op_attr::auto_broadcast, false, attribute_kind::s,
                        "numpy", {"none", "numpy"})
                // Attributes inherited from front BiasAdd ops, will only take
                // effect when is_bias_add attr is true
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::is_bias_add, false, attribute_kind::b, false)
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_binary_output_shape))

DNNL_GRAPH_OP_SCHEMA(_eltwise, 1,
        op_schema_t()
                // dnnl_eltwise can fuse dnnl_binary, so its input number is
                // variadic
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front eltwise ops
                .set_attr(op_attr::alpha, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta, false, attribute_kind::f, 0.f)
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_eltwise_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "forward_data")
                .set_input(1, "output_delta")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::alpha, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta, false, attribute_kind::f, 0.f)
                .set_attr(op_attr::use_dst, false, attribute_kind::b, false)
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(op_attr::fwd_alg_kind, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_gen_index, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input")
                .set_output(0, "output")
                // Attributes inherited from front GenIndex ops
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_shuffle, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // No corresponding frontend op
                // Attributes
                .set_attr(op_attr::axis, true, attribute_kind::i)
                .set_attr(op_attr::groups, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_reduction, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_input(1, "axes")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from front reduction ops
                .SET_REDUCE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::alg_kind, true, attribute_kind::i)
                .set_attr(op_attr::p, false, attribute_kind::f, 0.0f)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_reduce_output_shape))

DNNL_GRAPH_OP_SCHEMA(_softmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "forward_result")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from SoftMaxBackward
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_logsoftmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta")
                .set_input(1, "forward_result")
                .set_output(0, "input_delta")
                .set_output(1, "scratchpad")
                // Attributes inherited from LogSoftmaxBackward
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)-1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_resampling, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "data")
                .set_input(1, "sizes")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Interpolate.
                .set_attr(op_attr::mode, true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::sizes, false, attribute_kind::is)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode, false,
                        attribute_kind::s, "half_pixel",
                        {"half_pixel", "align_corners"})
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_interpolate_output_shape))

DNNL_GRAPH_OP_SCHEMA(_concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(2)
                .set_input(0, "a")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from Concat
                .set_attr(op_attr::axis, true, attribute_kind::i)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(_layernorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input_forward")
                .set_input(1, "output_delta")
                .set_input(2, "mean")
                .set_input(3, "variance")
                .set_input(4, "gamma")
                .set_input(5, "beta")
                .set_output(0, "input_delta")
                .set_output(1, "gamma_delta")
                .set_output(2, "beta_delta")
                .set_output(3, "scratchpad")
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_norm_bprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(_matmul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "src0")
                .set_input(1, "src1")
                .set_input(2, "bias") // optional
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from MatMul.
                .SET_MATMUL_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::with_bias, false, attribute_kind::b, false)
                .set_attr(
                        op_attr::canonicalized, false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::keep_dst_layout, false, attribute_kind::b,
                        false)
                // Analysis rules
                .set_shape_inference_function(infer_matmul_output_shape))

DNNL_GRAPH_OP_SCHEMA(_softmax, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .set_output(2, "stats") // optional
                // Attributes inherited from SoftMax
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::mode, false, attribute_kind::s, "none",
                        {"none", "inf_as_zero"})
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_softmax_output_shape))

DNNL_GRAPH_OP_SCHEMA(_logsoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                // Attributes inherited from LogSoftmax
                .set_attr(op_attr::axis, false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_layernorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input")
                .set_input(1, "gamma")
                .set_input(2, "beta")
                .set_output(0, "output")
                .set_output(1, "mean")
                .set_output(2, "variance")
                .set_output(3, "scratchpad")
                // Attributes inherited from LayerNorm
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis, false, attribute_kind::i,
                        int64_t(-1))
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::is_rms, false, attribute_kind::b, false)
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_layernorm_output_shape))

DNNL_GRAPH_OP_SCHEMA(_reorder, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "input")
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .set_attr(op_attr::mask, false, attribute_kind::i, int64_t(0))
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(
                        op_attr::change_layout, false, attribute_kind::b, false)
                .set_attr(op_attr::scales, false, attribute_kind::fs)
                .set_attr(op_attr::src_zps, false, attribute_kind::is)
                .set_attr(op_attr::dst_zps, false, attribute_kind::is)
                .set_attr(op_attr::group_shape, false, attribute_kind::is)
                .set_attr(op_attr::with_runtime_scales, false,
                        attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_src_zps, false,
                        attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_dst_zps, false,
                        attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(_groupnorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input")
                .set_input(1, "gamma")
                .set_input(2, "beta")
                .set_output(0, "output")
                .set_output(1, "mean")
                .set_output(2, "variance")
                .set_output(3, "scratchpad")
                // Attributes inherited from GroupNorm
                .set_attr(op_attr::keep_stats, false, attribute_kind::b, true)
                .set_attr(op_attr::groups, true, attribute_kind::i)
                .set_attr(op_attr::use_affine, false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon, false, attribute_kind::f, 1e-5f)
                .set_attr(op_attr::data_format, false, attribute_kind::s, "NXC",
                        {"NCX", "NXC"})
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_groupnorm_output_shape))

DNNL_GRAPH_OP_SCHEMA(_mask, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 4}))
                .set_num_outputs(1)
                .set_input(0, "input")
                .set_input(1, "-inf")
                .set_input(2, "s_kv")
                .set_input(3, "s_q")
                .set_output(0, "output")
                // Attributes inherited from front gen_index ops
                .set_attr(op_attr::axis_row, true, attribute_kind::i)
                .set_attr(op_attr::axis_col, true, attribute_kind::i)
                // mask_type attribute indicates existence of explicit mask,
                // top-left implicit causal mask or bottm-right implicit causal mask
                .set_attr(op_attr::mask_type, true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

// The data types of query/key/value/mask/output must be consistent, and only
// f16/bf16 are supported. The data type of scale must be consistent with other
// input and output data types or fp32.
DNNL_GRAPH_OP_SCHEMA(_sdpa, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 32}))
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "query")
                .set_input(1, "key")
                .set_input(2, "value")
                .set_input(3, "scale") // optional
                .set_input(4, "mask") // optional
                .set_output(0, "output")
                .set_output(1, "scratchpad")
                .set_output(2,
                        "softmax_stats") // optional, only used for sdpa training
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::with_scale, true, attribute_kind::b)
                .set_attr(op_attr::is_invert_scale, false, attribute_kind::b,
                        false)
                .set_attr(op_attr::is_invert_select, false, attribute_kind::b,
                        false)
                .set_attr(op_attr::is_training, false, attribute_kind::b)
                .set_attr(op_attr::with_dropout, false, attribute_kind::b)
                // mask_type attribute indicates existence of explicit mask,
                // top-left implicit causal mask or bottm-right implicit causal mask
                .set_attr(op_attr::mask_type, true, attribute_kind::i)
                .set_attr(op_attr::mode, true, attribute_kind::s)
                .set_attr(op_attr::qk_acc_mode, true, attribute_kind::s)
                .set_attr(op_attr::vs_acc_mode, true, attribute_kind::s)
                .set_shape_inference_function(infer_dnnl_sdpa_output_shape))

// Backward op for SDPA
DNNL_GRAPH_OP_SCHEMA(_sdpa_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                // Inputs: query, key, value, dst, diff_dst, [dS], [scale], [mask]
                .set_num_inputs(std::set<size_t>({5, 32}))
                .set_num_outputs(std::set<size_t>({4, 5}))
                .set_input(0, "query")
                .set_input(1, "key")
                .set_input(2, "value")
                .set_input(3, "dst")
                .set_input(4, "stats")
                .set_input(5, "diff_dst")
                .set_input(6, "scale") // optional
                .set_input(7, "mask") // optional
                // Outputs: diff_query, diff_key, diff_value, scratchpad, diff_mask
                .set_output(0, "diff_query")
                .set_output(1, "diff_key")
                .set_output(2, "diff_value")
                .set_output(3, "scratchpad")
                .set_output(4, "diff_mask") // optional
                .set_attr(op_attr::fusion_info, false,
                        attribute_kind::fusion_info)
                .set_attr(op_attr::with_scale, true, attribute_kind::b)
                .set_attr(op_attr::is_invert_scale, false, attribute_kind::b,
                        false)
                .set_attr(op_attr::with_dropout, false, attribute_kind::b)
                .set_attr(op_attr::mask_type, true, attribute_kind::i)
                .set_attr(op_attr::qk_acc_mode, true, attribute_kind::s)
                .set_attr(op_attr::vs_acc_mode, true, attribute_kind::s)
                .set_shape_inference_function(infer_dnnl_sdpa_bwd_output_shape))

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
