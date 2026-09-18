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

#ifndef GRAPH_INTERFACE_C_TYPES_MAP_HPP
#define GRAPH_INTERFACE_C_TYPES_MAP_HPP

#include <string>
#include <vector>
#include <type_traits>

#include "oneapi/dnnl/dnnl_graph_sycl.h"
#include "oneapi/dnnl/dnnl_graph_types.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#endif

namespace dnnl {
namespace impl {
namespace graph {

using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using dims = std::vector<dim_t>;

using status_t = dnnl_status_t;
namespace status {
const status_t success = dnnl_success;
const status_t out_of_memory = dnnl_out_of_memory;
const status_t invalid_arguments = dnnl_invalid_arguments;
const status_t unimplemented = dnnl_unimplemented;
const status_t last_impl_reached = dnnl_last_impl_reached;
const status_t runtime_error = dnnl_runtime_error;
const status_t not_required = dnnl_not_required;
const status_t invalid_graph = dnnl_invalid_graph;
const status_t invalid_graph_op = dnnl_invalid_graph_op;
const status_t invalid_shape = dnnl_invalid_shape;
const status_t invalid_data_type = dnnl_invalid_data_type;
} // namespace status

using data_type_t = dnnl_data_type_t;
namespace data_type {
const data_type_t undef = dnnl_data_type_undef;
const data_type_t f16 = dnnl_f16;
const data_type_t bf16 = dnnl_bf16;
const data_type_t f32 = dnnl_f32;
const data_type_t s32 = dnnl_s32;
const data_type_t s8 = dnnl_s8;
const data_type_t u8 = dnnl_u8;
const data_type_t boolean = dnnl_boolean;
const data_type_t f8_e5m2 = dnnl_f8_e5m2;
const data_type_t f8_e4m3 = dnnl_f8_e4m3;
const data_type_t s4 = dnnl_s4;
const data_type_t u4 = dnnl_u4;
const data_type_t s64 = dnnl_s64;
} // namespace data_type

using partition_policy_t = dnnl_graph_partition_policy_t;
namespace partition_policy {
const partition_policy_t fusion = dnnl_graph_partition_policy_fusion;
const partition_policy_t debug = dnnl_graph_partition_policy_debug;
} // namespace partition_policy

// partition kind is moved from API to internal.
enum class partition_kind_t {
    undef = 0,
    convolution_post_ops = 1,
    convtranspose_post_ops = 2,
    interpolate_post_ops = 3,
    matmul_post_ops = 4,
    reduction_post_ops = 5,
    unary_post_ops = 6,
    binary_post_ops = 7,
    pooling_post_ops = 8,
    batch_norm_post_ops = 9,
    misc_post_ops = 10,
    quantized_convolution_post_ops = 11,
    quantized_convtranspose_post_ops = 12,
    quantized_matmul_post_ops = 13,
    quantized_unary_post_ops = 14,
    quantized_pooling_post_ops = 15,
    misc_quantized_post_ops = 16,
    convolution_backward_post_ops = 17,
    mha = 18,
    mlp = 19,
    quantized_mha = 20,
    quantized_mlp = 21,
    residual_conv_blocks = 22,
    quantized_residual_conv_blocks = 23,
    concat_fusion_memory_optim = 24,
    sdp = 25,
    quantized_sdp = 26
};

using engine_kind_t = dnnl_engine_kind_t;
namespace engine_kind {
const engine_kind_t any_engine = dnnl_any_engine;
const engine_kind_t cpu = dnnl_cpu;
const engine_kind_t gpu = dnnl_gpu;
} // namespace engine_kind

using fpmath_mode_t = dnnl_fpmath_mode_t;
namespace fpmath_mode {
const fpmath_mode_t strict = dnnl_fpmath_mode_strict;
const fpmath_mode_t bf16 = dnnl_fpmath_mode_bf16;
const fpmath_mode_t f16 = dnnl_fpmath_mode_f16;
const fpmath_mode_t any = dnnl_fpmath_mode_any;
const fpmath_mode_t tf32 = dnnl_fpmath_mode_tf32;
}; // namespace fpmath_mode

using op_kind_t = typename std::underlying_type<dnnl_graph_op_kind_t>::type;
namespace op_kind {
const op_kind_t Abs = dnnl_graph_op_abs;
const op_kind_t AbsBackward = dnnl_graph_op_abs_backward;
const op_kind_t Add = dnnl_graph_op_add;
const op_kind_t AvgPool = dnnl_graph_op_avg_pool;
const op_kind_t AvgPoolBackward = dnnl_graph_op_avg_pool_backward;
const op_kind_t BatchNormForwardTraining
        = dnnl_graph_op_batch_norm_forward_training;
const op_kind_t BatchNormInference = dnnl_graph_op_batch_norm_inference;
const op_kind_t BatchNormTrainingBackward = dnnl_graph_op_batch_norm_backward;
const op_kind_t BiasAdd = dnnl_graph_op_bias_add;
const op_kind_t BiasAddBackward = dnnl_graph_op_bias_add_backward;
const op_kind_t Clamp = dnnl_graph_op_clamp;
const op_kind_t ClampBackward = dnnl_graph_op_clamp_backward;
const op_kind_t Concat = dnnl_graph_op_concat;
const op_kind_t Convolution = dnnl_graph_op_convolution;
const op_kind_t ConvolutionBackwardData
        = dnnl_graph_op_convolution_backward_data;
const op_kind_t ConvolutionBackwardWeights
        = dnnl_graph_op_convolution_backward_weights;
const op_kind_t ConvTranspose = dnnl_graph_op_conv_transpose;
const op_kind_t ConvTransposeBackwardData
        = dnnl_graph_op_conv_transpose_backward_data;
const op_kind_t ConvTransposeBackwardWeights
        = dnnl_graph_op_conv_transpose_backward_weights;
const op_kind_t Dequantize = dnnl_graph_op_dequantize;
const op_kind_t Divide = dnnl_graph_op_divide;
const op_kind_t DynamicDequantize = dnnl_graph_op_dynamic_dequantize;
const op_kind_t DynamicQuantize = dnnl_graph_op_dynamic_quantize;
const op_kind_t Elu = dnnl_graph_op_elu;
const op_kind_t EluBackward = dnnl_graph_op_elu_backward;
const op_kind_t End = dnnl_graph_op_end;
const op_kind_t Exp = dnnl_graph_op_exp;
const op_kind_t GELU = dnnl_graph_op_gelu;
const op_kind_t GELUBackward = dnnl_graph_op_gelu_backward;
const op_kind_t GenIndex = dnnl_graph_op_gen_index;
const op_kind_t GreaterEqual = dnnl_graph_op_greater_equal;
const op_kind_t GroupNorm = dnnl_graph_op_group_norm;
const op_kind_t HardSigmoid = dnnl_graph_op_hard_sigmoid;
const op_kind_t HardSigmoidBackward = dnnl_graph_op_hard_sigmoid_backward;
const op_kind_t HardSwish = dnnl_graph_op_hard_swish;
const op_kind_t HardSwishBackward = dnnl_graph_op_hard_swish_backward;
const op_kind_t Interpolate = dnnl_graph_op_interpolate;
const op_kind_t InterpolateBackward = dnnl_graph_op_interpolate_backward;
const op_kind_t LayerNorm = dnnl_graph_op_layer_norm;
const op_kind_t LayerNormBackward = dnnl_graph_op_layer_norm_backward;
const op_kind_t LeakyReLU = dnnl_graph_op_leaky_relu;
const op_kind_t Log = dnnl_graph_op_log;
const op_kind_t LogSoftmax = dnnl_graph_op_log_softmax;
const op_kind_t LogSoftmaxBackward = dnnl_graph_op_log_softmax_backward;
const op_kind_t MatMul = dnnl_graph_op_matmul;
const op_kind_t Maximum = dnnl_graph_op_maximum;
const op_kind_t MaxPool = dnnl_graph_op_max_pool;
const op_kind_t MaxPoolBackward = dnnl_graph_op_max_pool_backward;
const op_kind_t Minimum = dnnl_graph_op_minimum;
const op_kind_t Mish = dnnl_graph_op_mish;
const op_kind_t MishBackward = dnnl_graph_op_mish_backward;
const op_kind_t Multiply = dnnl_graph_op_multiply;
const op_kind_t Pow = dnnl_graph_op_pow;
const op_kind_t PReLU = dnnl_graph_op_prelu;
const op_kind_t PReLUBackward = dnnl_graph_op_prelu_backward;
const op_kind_t Quantize = dnnl_graph_op_quantize;
const op_kind_t Reciprocal = dnnl_graph_op_reciprocal;
const op_kind_t ReduceL1 = dnnl_graph_op_reduce_l1;
const op_kind_t ReduceL2 = dnnl_graph_op_reduce_l2;
const op_kind_t ReduceMax = dnnl_graph_op_reduce_max;
const op_kind_t ReduceMean = dnnl_graph_op_reduce_mean;
const op_kind_t ReduceMin = dnnl_graph_op_reduce_min;
const op_kind_t ReduceProd = dnnl_graph_op_reduce_prod;
const op_kind_t ReduceSum = dnnl_graph_op_reduce_sum;
const op_kind_t ReLU = dnnl_graph_op_relu;
const op_kind_t ReLUBackward = dnnl_graph_op_relu_backward;
const op_kind_t RMSNorm = dnnl_graph_op_rms_norm;
const op_kind_t Reorder = dnnl_graph_op_reorder;
const op_kind_t Round = dnnl_graph_op_round;
const op_kind_t Select = dnnl_graph_op_select;
const op_kind_t Sigmoid = dnnl_graph_op_sigmoid;
const op_kind_t SigmoidBackward = dnnl_graph_op_sigmoid_backward;
const op_kind_t SoftMax = dnnl_graph_op_softmax;
const op_kind_t SoftMaxBackward = dnnl_graph_op_softmax_backward;
const op_kind_t SoftPlus = dnnl_graph_op_softplus;
const op_kind_t SoftPlusBackward = dnnl_graph_op_softplus_backward;
const op_kind_t Sqrt = dnnl_graph_op_sqrt;
const op_kind_t SqrtBackward = dnnl_graph_op_sqrt_backward;
const op_kind_t Square = dnnl_graph_op_square;
const op_kind_t SquaredDifference = dnnl_graph_op_squared_difference;
const op_kind_t StaticReshape = dnnl_graph_op_static_reshape;
const op_kind_t StaticTranspose = dnnl_graph_op_static_transpose;
const op_kind_t Subtract = dnnl_graph_op_subtract;
const op_kind_t Tanh = dnnl_graph_op_tanh;
const op_kind_t TanhBackward = dnnl_graph_op_tanh_backward;
const op_kind_t TypeCast = dnnl_graph_op_type_cast;
const op_kind_t Wildcard = dnnl_graph_op_wildcard;
const op_kind_t Dropout = dnnl_graph_op_dropout;
// end of public ops
const op_kind_t LastSymbol = dnnl_graph_op_last_symbol;

// internal ops used by the backend.
const op_kind_t _mul_scales = 1024;
const op_kind_t _constant_scales = 1025;
const op_kind_t _add_zps = 1026;
const op_kind_t _sub_zps = 1027;
const op_kind_t _constant_zps = 1028;
const op_kind_t _permute = 1029;
const op_kind_t _to_group = 1030;
const op_kind_t _from_group = 1031;
const op_kind_t _unsqueeze = 1032;
const op_kind_t _squeeze = 1033;
const op_kind_t _reshape = 1034;
const op_kind_t _transpose = 1035;
const op_kind_t _convolution = 1036;
const op_kind_t _convtranspose = 1037;
const op_kind_t _pool = 1038;
const op_kind_t _bn_folding = 1039;
const op_kind_t _conv_bwd_data = 1040;
const op_kind_t _batchnorm = 1041;
const op_kind_t _binary = 1042;
const op_kind_t _eltwise = 1043;
const op_kind_t _eltwise_bwd = 1044;
const op_kind_t _shuffle = 1045;
const op_kind_t _sum = 1046;
const op_kind_t _reduction = 1047;
const op_kind_t _prelu = 1048;
const op_kind_t _prelu_bwd = 1049;
const op_kind_t _batchnorm_bwd = 1050;
const op_kind_t _softmax_bwd = 1051;
const op_kind_t _logsoftmax_bwd = 1052;
const op_kind_t _resampling = 1053;
const op_kind_t _resampling_bwd = 1054;
const op_kind_t _concat = 1055;
const op_kind_t _layernorm_bwd = 1056;
const op_kind_t _conv_bwd_weights = 1057;
const op_kind_t _pool_bwd = 1058;
const op_kind_t _matmul = 1059;
const op_kind_t _softmax = 1060;
const op_kind_t _logsoftmax = 1061;
const op_kind_t _layernorm = 1062;
const op_kind_t _reorder = 1063;
const op_kind_t _convtranspose_bwd_data = 1064;
const op_kind_t _convtranspose_bwd_weights = 1065;
const op_kind_t _groupnorm = 1066;
const op_kind_t _gen_index = 1067;
const op_kind_t _mask = 1068;
const op_kind_t _sdpa = 1069;
const op_kind_t _host_scalar = 1070;
const op_kind_t _identity = 1071;
const op_kind_t _dropout = 1072;
const op_kind_t _sdpa_bwd = 1074;
} // namespace op_kind

using op_attr_t = typename std::underlying_type<dnnl_graph_op_attr_t>::type;
namespace op_attr {
const op_attr_t undef = dnnl_graph_op_attr_undef;

const op_attr_t alpha = dnnl_graph_op_attr_alpha;
const op_attr_t beta = dnnl_graph_op_attr_beta;
const op_attr_t epsilon = dnnl_graph_op_attr_epsilon;
const op_attr_t max = dnnl_graph_op_attr_max;
const op_attr_t min = dnnl_graph_op_attr_min;
const op_attr_t momentum = dnnl_graph_op_attr_momentum;

const op_attr_t scales = dnnl_graph_op_attr_scales;

const op_attr_t axis = dnnl_graph_op_attr_axis;
const op_attr_t begin_norm_axis = dnnl_graph_op_attr_begin_norm_axis;
const op_attr_t groups = dnnl_graph_op_attr_groups;
const op_attr_t mask = dnnl_graph_op_attr_mask;

const op_attr_t axes = dnnl_graph_op_attr_axes;
const op_attr_t dilations = dnnl_graph_op_attr_dilations;
const op_attr_t weights_shape = dnnl_graph_op_attr_weights_shape;
const op_attr_t src_shape = dnnl_graph_op_attr_src_shape;
const op_attr_t kernel = dnnl_graph_op_attr_kernel;
const op_attr_t order = dnnl_graph_op_attr_order;
const op_attr_t output_padding = dnnl_graph_op_attr_output_padding;
const op_attr_t dst_shape = dnnl_graph_op_attr_dst_shape;
const op_attr_t pads_begin = dnnl_graph_op_attr_pads_begin;
const op_attr_t pads_end = dnnl_graph_op_attr_pads_end;
const op_attr_t shape = dnnl_graph_op_attr_shape;
const op_attr_t sizes = dnnl_graph_op_attr_sizes;
const op_attr_t strides = dnnl_graph_op_attr_strides;
const op_attr_t zps = dnnl_graph_op_attr_zps;
const op_attr_t group_shape = dnnl_graph_op_attr_group_shape;

const op_attr_t exclude_pad = dnnl_graph_op_attr_exclude_pad;
const op_attr_t keep_dims = dnnl_graph_op_attr_keep_dims;
const op_attr_t keep_stats = dnnl_graph_op_attr_keep_stats;
const op_attr_t per_channel_broadcast
        = dnnl_graph_op_attr_per_channel_broadcast;
const op_attr_t special_zero = dnnl_graph_op_attr_special_zero;
const op_attr_t transpose_a = dnnl_graph_op_attr_transpose_a;
const op_attr_t transpose_b = dnnl_graph_op_attr_transpose_b;
const op_attr_t use_affine = dnnl_graph_op_attr_use_affine;
const op_attr_t use_dst = dnnl_graph_op_attr_use_dst;

const op_attr_t auto_broadcast = dnnl_graph_op_attr_auto_broadcast;
const op_attr_t auto_pad = dnnl_graph_op_attr_auto_pad;
const op_attr_t coordinate_transformation_mode
        = dnnl_graph_op_attr_coordinate_transformation_mode;
const op_attr_t data_format = dnnl_graph_op_attr_data_format;
const op_attr_t weights_format = dnnl_graph_op_attr_weights_format;
const op_attr_t mode = dnnl_graph_op_attr_mode;
const op_attr_t qtype = dnnl_graph_op_attr_qtype;
const op_attr_t rounding_type = dnnl_graph_op_attr_rounding_type;
const op_attr_t accumulation_mode = dnnl_graph_op_attr_accumulation_mode;

// end of public attributes
const op_attr_t end = dnnl_graph_op_attr_end;

// internal attributes: bool
const op_attr_t matched = 0x10000;
const op_attr_t canonicalized = 0x10001;
const op_attr_t change_layout = 0x10002;
const op_attr_t is_constant = 0x10003;
const op_attr_t is_convtranspose = 0x10004;
const op_attr_t is_training = 0x10005;
const op_attr_t with_dropout = 0x10006;
const op_attr_t fuse_relu = 0x10007;
const op_attr_t with_bias = 0x10008;
const op_attr_t with_runtime_scales = 0x10009;
const op_attr_t with_runtime_zps = 0x1000a;
const op_attr_t with_runtime_src_zps = 0x1000b;
const op_attr_t with_runtime_dst_zps = 0x1000c;
const op_attr_t is_bias_add = 0x1000d;
const op_attr_t with_sum = 0x1000e;
const op_attr_t keep_dst_layout = 0x1000f;
const op_attr_t with_scale = 0x10010;
const op_attr_t is_invert_scale = 0x10011;
const op_attr_t mask_type = 0x10012;
const op_attr_t is_rms = 0x10013;
const op_attr_t is_invert_select = 0x10014;

// int64_t
const op_attr_t partition_id = 0x10100;
const op_attr_t op_depth = 0x10101;
const op_attr_t alg_kind = 0x10102;
const op_attr_t group_mask = 0x10103;
const op_attr_t data_type = 0x10104;
const op_attr_t axis_row = 0x10105;
const op_attr_t axis_col = 0x10106;
const op_attr_t fwd_alg_kind = 0x10107;

// string
const op_attr_t backend = 0x10200;
const op_attr_t dw_type = 0x10201;
const op_attr_t kind = 0x10204;
const op_attr_t qk_acc_mode = 0x10205;
const op_attr_t vs_acc_mode = 0x10206;

// float
const op_attr_t p = 0x10300;

// vector of int64_t
const op_attr_t dst_zps = 0x10400;
const op_attr_t src_zps = 0x10401;
const op_attr_t permutation = 0x10402;

// store op's fusion info
const op_attr_t fusion_info = 0x10500;
} // namespace op_attr

using logical_tensor_t = dnnl_graph_logical_tensor_t;

using layout_type_t = dnnl_graph_layout_type_t;
namespace layout_type {
const layout_type_t undef = dnnl_graph_layout_type_undef;
const layout_type_t any = dnnl_graph_layout_type_any;
const layout_type_t strided = dnnl_graph_layout_type_strided;
const layout_type_t opaque = dnnl_graph_layout_type_opaque;
} // namespace layout_type

using property_type_t = dnnl_graph_tensor_property_t;
namespace property_type {
const property_type_t undef = dnnl_graph_tensor_property_undef;
const property_type_t variable = dnnl_graph_tensor_property_variable;
const property_type_t constant = dnnl_graph_tensor_property_constant;
const property_type_t host_scalar = dnnl_graph_tensor_property_host_scalar;
} // namespace property_type

using attribute_kind_t = size_t;
namespace attribute_kind {
const attribute_kind_t f = 0;
const attribute_kind_t fs = 1;
const attribute_kind_t i = 2;
const attribute_kind_t is = 3;
const attribute_kind_t s = 4;
const attribute_kind_t b = 5;
const attribute_kind_t fusion_info = 6; //special attribute kind for fusion info
} // namespace attribute_kind

enum class graph_dump_mode_t : uint8_t {
    none = dnnl_graph_dump_mode_none,
    subgraph = dnnl_graph_dump_mode_subgraph,
    graph = dnnl_graph_dump_mode_graph,
    // Internal-only flag for dumping partition patterns.
    pattern = 0x4U,
};

using allocator_t = dnnl_graph_allocator;
using host_allocate_f = dnnl_graph_host_allocate_f;
using host_deallocate_f = dnnl_graph_host_deallocate_f;
using sycl_allocate_f = dnnl_graph_sycl_allocate_f;
using sycl_deallocate_f = dnnl_graph_sycl_deallocate_f;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
using ocl_allocate_f = dnnl_graph_ocl_allocate_f;
using ocl_deallocate_f = dnnl_graph_ocl_deallocate_f;
#endif
using inplace_pair_t = dnnl_graph_inplace_pair_t;

using graph_t = dnnl_graph_graph;
using op_t = dnnl_graph_op;
using partition_t = dnnl_graph_partition;
using compiled_partition_t = dnnl_graph_compiled_partition;
using tensor_t = dnnl_graph_tensor;

// oneDNN common objects
using engine_t = dnnl_engine;
using stream_t = dnnl_stream;

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
