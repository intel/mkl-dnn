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

#include <algorithm>
#include <set>
#include <vector>

#include "graph/allocator.hpp"
#include "graph/utils.hpp"

#include "dnnl_common.hpp"

#include "utils/stream_kind.hpp"
#include "utils/stringstream.hpp"
#include "utils/timer.hpp"

namespace graph {

bdnn_state_t convert_state(const dnnl_status_t &s) {
    switch (s) {
        case dnnl_status_t::dnnl_success:
            return bdnn_state_t {res_state_t::PASSED};
        case dnnl_status_t::dnnl_out_of_memory:
            return bdnn_state_t {
                    res_state_t::SKIPPED, reason_t::skip_not_enough_ram};
        case dnnl_status_t::dnnl_invalid_arguments:
            return bdnn_state_t {res_state_t::INVALID_ARGUMENTS};
        case dnnl_status_t::dnnl_unimplemented:
            return bdnn_state_t {res_state_t::UNIMPLEMENTED};
        case dnnl_status_t::dnnl_last_impl_reached:
            return bdnn_state_t {res_state_t::SKIPPED, reason_t::skip_impl_hit};
        case dnnl_status_t::dnnl_runtime_error:
            return bdnn_state_t {res_state_t::FAILED};
        case dnnl_status_t::dnnl_not_required:
            return bdnn_state_t {res_state_t::INVALID_ARGUMENTS};
        case dnnl_status_t::dnnl_invalid_graph:
        case dnnl_status_t::dnnl_invalid_graph_op:
        case dnnl_status_t::dnnl_invalid_shape:
            return bdnn_state_t {res_state_t::SKIPPED, reason_t::invalid};
        case dnnl_status_t::dnnl_invalid_data_type:
        case dnnl_status_t::dnnl_unsupported_data_type:
            return bdnn_state_t {
                    res_state_t::SKIPPED, reason_t::skip_data_type};
        default: assert(!"dnnl state is not found!"); return bdnn_state_t {};
    }
}

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::stream &stream, const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs,
        const dnnl::graph::tensor &scratchpad) {
    cp.execute(stream, inputs, outputs, scratchpad);
}

inline int measure_perf_aggregate(timer::timer_t &t,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        const std::vector<dnnl::graph::tensor> &scratchpads, res_t *res) {
    const int max_batch_times = 4096;
    // Nvidia/AMD don't support profiling.
    const bool use_profiling = is_gpu() && !is_nvidia_gpu() && !is_amd_gpu();
    stream_t stream {get_graph_engine()};

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    const auto sz = perf_func_v.size();
    if (!has_bench_mode_bit(mode_bit_t::sim)) {
        for (size_t i = 0; i < sz; i++) {
            DNN_GRAPH_SAFE(perf_func_v[i](stream, inputs_v[i], outputs_v[i],
                                   scratchpads[i]),
                    WARN, res);
            DNN_GRAPH_SAFE(stream.wait(), WARN, res);
        }
    }

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();
    reset_gpu_profiling(stream);

    bool is_first_loop = true;
    size_t prim_num = 1;
    while (true) {
        for_(int i = 0; i < cur_batch_times; i++)
        for (size_t j = 0; j < sz; j++) {
            DNN_GRAPH_SAFE(perf_func_v[j](stream, inputs_v[j], outputs_v[j],
                                   scratchpads[j]),
                    WARN, res);
        }
        DNN_GRAPH_SAFE(stream.wait(), WARN, res);

        if (use_profiling) {
            std::vector<uint64_t> nsecs;
            std::vector<uint64_t> cycles;
            // Cannot determine the number of expected profiling entries
            // beforehand so pass -1.
            SAFE(get_gpu_profiling_info(
                         stream, nsecs, cycles, /*expected_num_entries=*/-1),
                    CRIT);
            reset_gpu_profiling(stream);

            // Profiling should have information to report, otherwise, stop.
            if (nsecs.empty()) {
                BENCHDNN_PRINT(0, "%s\n",
                        "WARNING: no counters were found during profiling.");
                break;
            }
            // Calculate the number of primitives in a batch
            if (is_first_loop) { prim_num = nsecs.size() / cur_batch_times; }

            for (int i = 0; i < cur_batch_times; i++) {
                int64_t cycles_res = 0;
                double nsecs_res = 0;
                for (size_t j = 0; j < prim_num; j++) {
                    cycles_res += cycles[i * prim_num + j];
                    nsecs_res += nsecs[i * prim_num + j];
                }
                t.stop(1, cycles_res, nsecs_res / 1e6);
            }
        } else {
            t.stamp(cur_batch_times);
        }

        if (should_stop(t)) break;

        // Adjust cur_batch_times after the first batch run
        if (is_first_loop) {
            double ms_min = t.ms(timer::timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                              (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                      / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
            is_first_loop = false;
        }
    }
    return OK;
}

inline int measure_perf_individual(timer::timer_t &t,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        const std::vector<dnnl::graph::tensor> &scratchpads, res_t *res) {
    stream_t stream {get_graph_engine()};

    t.reset();
    while (true) {
        auto sz = perf_func_v.size();
        for (size_t i = 0; i < sz; i++) {
            DNN_GRAPH_SAFE(perf_func_v[i](stream, inputs_v[i], outputs_v[i],
                                   scratchpads[i]),
                    WARN, res);
        }
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

int measure_perf(timer::timer_t &t, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        const std::vector<dnnl::graph::tensor> &scratchpads, res_t *res) {
    const auto &engine = get_test_engine();
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        // enable GPU profiling, Nvidia/AMD dose not support profiling.
        int ret = OK;
        if (is_async(engine)) {
            ret = measure_perf_aggregate(
                    t, perf_func_v, inputs_v, outputs_v, scratchpads, res);
        } else {
            ret = measure_perf_individual(
                    t, perf_func_v, inputs_v, outputs_v, scratchpads, res);
        }
        return ret;
    } else {
        return OK;
    }
}

int measure_perf(timer::timer_t &t,
        const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        const std::vector<dnnl::graph::tensor> &scratchpads, res_t *res) {
    std::vector<perf_function_t> perf_func_v;
    perf_func_v.reserve(cp_v.size());
    for (size_t i = 0; i < cp_v.size(); i++) {
        perf_func_v.emplace_back(std::bind(&compiled_partition_executor,
                cp_v[i], std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
    }

    int status = measure_perf(
            t, perf_func_v, inputs_v, outputs_v, scratchpads, res);
    if (res) res->state = EXECUTED;

    return status;
}

dnnl::graph::op::kind opstr2kind(const std::string &kind) {
    const std::unordered_map<std::string, dnnl::graph::op::kind> op_map = {
            {"Abs", dnnl::graph::op::kind::Abs},
            {"AbsBackward", dnnl::graph::op::kind::AbsBackward},
            {"Add", dnnl::graph::op::kind::Add},
            {"AvgPool", dnnl::graph::op::kind::AvgPool},
            {"AvgPoolBackward", dnnl::graph::op::kind::AvgPoolBackward},
            {"BatchNormInference", dnnl::graph::op::kind::BatchNormInference},
            {"BatchNormForwardTraining",
                    dnnl::graph::op::kind::BatchNormForwardTraining},
            {"BatchNormTrainingBackward",
                    dnnl::graph::op::kind::BatchNormTrainingBackward},
            {"BiasAdd", dnnl::graph::op::kind::BiasAdd},
            {"BiasAddBackward", dnnl::graph::op::kind::BiasAddBackward},
            {"Clamp", dnnl::graph::op::kind::Clamp},
            {"ClampBackward", dnnl::graph::op::kind::ClampBackward},
            {"Concat", dnnl::graph::op::kind::Concat},
            {"Convolution", dnnl::graph::op::kind::Convolution},
            {"ConvolutionBackwardData",
                    dnnl::graph::op::kind::ConvolutionBackwardData},
            {"ConvolutionBackwardWeights",
                    dnnl::graph::op::kind::ConvolutionBackwardWeights},
            {"ConvTranspose", dnnl::graph::op::kind::ConvTranspose},
            {"ConvTransposeBackwardData",
                    dnnl::graph::op::kind::ConvTransposeBackwardData},
            {"ConvTransposeBackwardWeights",
                    dnnl::graph::op::kind::ConvTransposeBackwardWeights},
            {"Dequantize", dnnl::graph::op::kind::Dequantize},
            {"Divide", dnnl::graph::op::kind::Divide},
            {"Dropout", dnnl::graph::op::kind::Dropout},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackward", dnnl::graph::op::kind::EluBackward},
            {"End", dnnl::graph::op::kind::End},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackward", dnnl::graph::op::kind::GELUBackward},
            {"GenIndex", dnnl::graph::op::kind::GenIndex},
            {"GreaterEqual", dnnl::graph::op::kind::GreaterEqual},
            {"GroupNorm", dnnl::graph::op::kind::GroupNorm},
            {"HardSigmoid", dnnl::graph::op::kind::HardSigmoid},
            {"HardSigmoidBackward", dnnl::graph::op::kind::HardSigmoidBackward},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackward", dnnl::graph::op::kind::HardSwishBackward},
            {"Interpolate", dnnl::graph::op::kind::Interpolate},
            {"InterpolateBackward", dnnl::graph::op::kind::InterpolateBackward},
            {"LayerNorm", dnnl::graph::op::kind::LayerNorm},
            {"LayerNormBackward", dnnl::graph::op::kind::LayerNormBackward},
            {"LeakyReLU", dnnl::graph::op::kind::LeakyReLU},
            {"Log", dnnl::graph::op::kind::Log},
            {"LogSoftmax", dnnl::graph::op::kind::LogSoftmax},
            {"LogSoftmaxBackward", dnnl::graph::op::kind::LogSoftmaxBackward},
            {"MatMul", dnnl::graph::op::kind::MatMul},
            {"Maximum", dnnl::graph::op::kind::Maximum},
            {"MaxPool", dnnl::graph::op::kind::MaxPool},
            {"MaxPoolBackward", dnnl::graph::op::kind::MaxPoolBackward},
            {"Minimum", dnnl::graph::op::kind::Minimum},
            {"Mish", dnnl::graph::op::kind::Mish},
            {"MishBackward", dnnl::graph::op::kind::MishBackward},
            {"Multiply", dnnl::graph::op::kind::Multiply},
            {"Pow", dnnl::graph::op::kind::Pow},
            {"PReLU", dnnl::graph::op::kind::PReLU},
            {"PReLUBackward", dnnl::graph::op::kind::PReLUBackward},
            {"Quantize", dnnl::graph::op::kind::Quantize},
            {"Reciprocal", dnnl::graph::op::kind::Reciprocal},
            {"ReduceL1", dnnl::graph::op::kind::ReduceL1},
            {"ReduceL2", dnnl::graph::op::kind::ReduceL2},
            {"ReduceMax", dnnl::graph::op::kind::ReduceMax},
            {"ReduceMean", dnnl::graph::op::kind::ReduceMean},
            {"ReduceMin", dnnl::graph::op::kind::ReduceMin},
            {"ReduceProd", dnnl::graph::op::kind::ReduceProd},
            {"ReduceSum", dnnl::graph::op::kind::ReduceSum},
            {"ReLU", dnnl::graph::op::kind::ReLU},
            {"ReLUBackward", dnnl::graph::op::kind::ReLUBackward},
            {"Reorder", dnnl::graph::op::kind::Reorder},
            {"RMSNorm", dnnl::graph::op::kind::RMSNorm},
            {"Round", dnnl::graph::op::kind::Round},
            {"Select", dnnl::graph::op::kind::Select},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackward", dnnl::graph::op::kind::SigmoidBackward},
            {"SoftMax", dnnl::graph::op::kind::SoftMax},
            {"SoftMaxBackward", dnnl::graph::op::kind::SoftMaxBackward},
            {"SoftPlus", dnnl::graph::op::kind::SoftPlus},
            {"SoftPlusBackward", dnnl::graph::op::kind::SoftPlusBackward},
            {"Sqrt", dnnl::graph::op::kind::Sqrt},
            {"SqrtBackward", dnnl::graph::op::kind::SqrtBackward},
            {"Square", dnnl::graph::op::kind::Square},
            {"SquaredDifference", dnnl::graph::op::kind::SquaredDifference},
            {"StaticReshape", dnnl::graph::op::kind::StaticReshape},
            {"StaticTranspose", dnnl::graph::op::kind::StaticTranspose},
            {"Subtract", dnnl::graph::op::kind::Subtract},
            {"Tanh", dnnl::graph::op::kind::Tanh},
            {"TanhBackward", dnnl::graph::op::kind::TanhBackward},
            {"TypeCast", dnnl::graph::op::kind::TypeCast},
            {"Wildcard", dnnl::graph::op::kind::Wildcard}};
    const auto it = op_map.find(kind);
    if (it != op_map.end()) {
        return it->second;
    } else {
        fprintf(stderr, "graph: ERROR: Unsupported opkind: `%s`, exiting...\n",
                kind.c_str());
        SAFE_V(FAIL);
    }
    return dnnl::graph::op::kind::LastSymbol;
}

// The list of operations considered Unary from documentation point of view.
//
// Note: this list is disconnected from the rest of driver internals.
//
// TODO: come up with a single table for each supported op with different
// properties (unary/binary/forward/backward/etc.) and provide different
// converters and getters for it.
bool is_unary(const std::string &kind) {
    return is_unary(opstr2kind(kind));
}
bool is_unary(dnnl::graph::op::kind akind) {
    using namespace dnnl::graph;
    static const std::vector<op::kind> unary_ops {
            op::kind::Abs,
            op::kind::Clamp,
            op::kind::Elu,
            op::kind::Exp,
            op::kind::GELU,
            op::kind::HardSigmoid,
            op::kind::HardSwish,
            op::kind::LeakyReLU,
            op::kind::Log,
            op::kind::Mish,
            op::kind::ReLU,
            op::kind::Round,
            op::kind::Sigmoid,
            op::kind::SoftPlus,
            op::kind::Sqrt,
            op::kind::Square,
            op::kind::Tanh,
    };
    return std::any_of(unary_ops.begin(), unary_ops.end(),
            [akind](dnnl::graph::op::kind _kind) { return _kind == akind; });
}

bool is_binary(const std::string &kind) {
    return is_binary(opstr2kind(kind));
}
bool is_binary(dnnl::graph::op::kind akind) {
    using namespace dnnl::graph;
    static const std::vector<op::kind> binary_ops {
            op::kind::Add,
            op::kind::BiasAdd,
            op::kind::Divide,
            op::kind::Maximum,
            op::kind::Minimum,
            op::kind::Multiply,
            op::kind::Subtract,
    };
    return std::any_of(binary_ops.begin(), binary_ops.end(),
            [akind](dnnl::graph::op::kind _kind) { return _kind == akind; });
}

bool is_backward(const std::string &kind) {
    return is_backward(opstr2kind(kind));
}
bool is_backward(dnnl::graph::op::kind akind) {
    using namespace dnnl::graph;
    static const std::vector<op::kind> backward_ops {
            op::kind::AbsBackward,
            op::kind::AvgPoolBackward,
            op::kind::BatchNormTrainingBackward,
            op::kind::BiasAddBackward,
            op::kind::ClampBackward,
            op::kind::ConvolutionBackwardData,
            op::kind::ConvolutionBackwardWeights,
            op::kind::ConvTransposeBackwardData,
            op::kind::ConvTransposeBackwardWeights,
            op::kind::EluBackward,
            op::kind::GELUBackward,
            op::kind::HardSwishBackward,
            op::kind::InterpolateBackward,
            op::kind::LayerNormBackward,
            op::kind::LogSoftmaxBackward,
            op::kind::MaxPoolBackward,
            op::kind::MishBackward,
            op::kind::ReLUBackward,
            op::kind::SigmoidBackward,
            op::kind::SoftMaxBackward,
            op::kind::SoftPlusBackward,
            op::kind::SqrtBackward,
            op::kind::TanhBackward,
    };
    return std::any_of(backward_ops.begin(), backward_ops.end(),
            [akind](dnnl::graph::op::kind _kind) { return _kind == akind; });
}

dnnl::graph::op::attr attrstr2kind(const std::string &attr_name) {
    const std::unordered_map<std::string, dnnl::graph::op::attr> attr_map = {
            // float32 attributes. The value of these attributes can be any single
            // float32 number.
            {"alpha", dnnl::graph::op::attr::alpha},
            {"beta", dnnl::graph::op::attr::beta},
            {"epsilon", dnnl::graph::op::attr::epsilon},
            {"max", dnnl::graph::op::attr::max},
            {"min", dnnl::graph::op::attr::min},
            {"momentum", dnnl::graph::op::attr::momentum},
            // float32 vector attributes. The value of these attributes can be a
            // vector of float32 numbers.
            {"scales", dnnl::graph::op::attr::scales},
            // int64_t attributes. The value of these attributes can be any single
            // int64 number.
            {"axis", dnnl::graph::op::attr::axis},
            {"begin_norm_axis", dnnl::graph::op::attr::begin_norm_axis},
            {"groups", dnnl::graph::op::attr::groups},
            {"group_shape", dnnl::graph::op::attr::group_shape},
            {"mask", dnnl::graph::op::attr::mask},
            // int64_t vector attributes. The value of these attributes can be a
            // vector of int64 numbers.
            {"axes", dnnl::graph::op::attr::axes},
            {"dilations", dnnl::graph::op::attr::dilations},
            {"weights_shape", dnnl::graph::op::attr::weights_shape},
            {"src_shape", dnnl::graph::op::attr::src_shape},
            {"kernel", dnnl::graph::op::attr::kernel},
            {"order", dnnl::graph::op::attr::order},
            {"output_padding", dnnl::graph::op::attr::output_padding},
            {"dst_shape", dnnl::graph::op::attr::dst_shape},
            {"pads_begin", dnnl::graph::op::attr::pads_begin},
            {"pads_end", dnnl::graph::op::attr::pads_end},
            {"shape", dnnl::graph::op::attr::shape},
            {"sizes", dnnl::graph::op::attr::sizes},
            {"strides", dnnl::graph::op::attr::strides},
            {"zps", dnnl::graph::op::attr::zps},
            // bool attributes. The value of these attributes can be any single bool
            // value.
            {"exclude_pad", dnnl::graph::op::attr::exclude_pad},
            {"keep_dims", dnnl::graph::op::attr::keep_dims},
            {"keep_stats", dnnl::graph::op::attr::keep_stats},
            {"per_channel_broadcast",
                    dnnl::graph::op::attr::per_channel_broadcast},
            {"special_zero", dnnl::graph::op::attr::special_zero},
            {"transpose_a", dnnl::graph::op::attr::transpose_a},
            {"transpose_b", dnnl::graph::op::attr::transpose_b},
            {"use_affine", dnnl::graph::op::attr::use_affine},
            {"use_dst", dnnl::graph::op::attr::use_dst},
            // string attributes. The value of these attributes can be a string.
            {"auto_broadcast", dnnl::graph::op::attr::auto_broadcast},
            {"auto_pad", dnnl::graph::op::attr::auto_pad},
            {"coordinate_transformation_mode",
                    dnnl::graph::op::attr::coordinate_transformation_mode},
            {"data_format", dnnl::graph::op::attr::data_format},
            {"weights_format", dnnl::graph::op::attr::weights_format},
            {"mode", dnnl::graph::op::attr::mode},
            {"qtype", dnnl::graph::op::attr::qtype},
            {"rounding_type", dnnl::graph::op::attr::rounding_type},
            {"accumulation_mode", dnnl::graph::op::attr::accumulation_mode},
    };
    const auto it = attr_map.find(attr_name);
    if (it != attr_map.end()) {
        return it->second;
    } else {
        fprintf(stderr,
                "graph: ERROR: Unsupported attribute: `%s`, exiting...\n",
                attr_name.c_str());
        SAFE_V(FAIL);
    }
    return dnnl::graph::op::attr::undef;
}

const std::string &attrstr2type(const std::string &attr_name) {
    const static std::unordered_map<std::string, std::string> attr_map = {
            {"undef", "undef"},
            // float32 attributes. The value of these attributes can be any single
            // float32 number.
            {"alpha", "f32"},
            {"beta", "f32"},
            {"epsilon", "f32"},
            {"max", "f32"},
            {"min", "f32"},
            {"momentum", "f32"},
            // float32 vector attributes. The value of these attributes can be a
            // vector of float32 numbers.
            {"scales", "f32[]"},
            // int64_t attributes. The value of these attributes can be any single
            // int64 number.
            {"axis", "s64"},
            {"begin_norm_axis", "s64"},
            {"groups", "s64"},
            {"group_shape", "s64"},
            {"mask", "s64"},
            // int64_t vector attributes. The value of these attributes can be a
            // vector of int64 numbers.
            {"axes", "s64[]"},
            {"dilations", "s64[]"},
            {"weights_shape", "s64[]"},
            {"src_shape", "s64[]"},
            {"kernel", "s64[]"},
            {"order", "s64[]"},
            {"output_padding", "s64[]"},
            {"dst_shape", "s64[]"},
            {"pads_begin", "s64[]"},
            {"pads_end", "s64[]"},
            {"shape", "s64[]"},
            {"sizes", "s64[]"},
            {"strides", "s64[]"},
            {"zps", "s64[]"},
            // bool attributes. The value of these attributes can be any single bool
            // value.
            {"exclude_pad", "bool"},
            {"keep_dims", "bool"},
            {"keep_stats", "bool"},
            {"per_channel_broadcast", "bool"},
            {"special_zero", "bool"},
            {"transpose_a", "bool"},
            {"transpose_b", "bool"},
            {"use_affine", "bool"},
            {"use_dst", "bool"},
            // string attributes. The value of these attributes can be a string.
            {"auto_broadcast", "string"},
            {"auto_pad", "string"},
            {"coordinate_transformation_mode", "string"},
            {"data_format", "string"},
            {"weights_format", "string"},
            {"mode", "string"},
            {"qtype", "string"},
            {"rounding_type", "string"},
            {"accumulation_mode", "string"},
    };
    const auto it = attr_map.find(attr_name);
    if (it != attr_map.end()) {
        return it->second;
    } else {
        fprintf(stderr,
                "graph: ERROR: Unsupported attribute: `%s`, exiting...\n",
                attr_name.c_str());
        SAFE_V(FAIL);
        return attr_map.at("undef");
    }
}

// when length is 3, return "abc", when length is 5, return "abcde"
std::string get_default_tag(size_t length) {
    std::string mtag;
    for (size_t i = 0; i < length; ++i) {
        mtag += char('a' + i);
    }
    return mtag;
}

// refer to the implementation of md2fmt_tag_str() in verbose.cpp.
std::string strides2memory_tag(const dnnl::graph::logical_tensor::dims &dims,
        const dnnl::graph::logical_tensor::dims &strides, bool use_x_tag) {
    const size_t ndims = dims.size();
    if (ndims == 0) return "";

    const std::string template_tag = "abcdefghijk";
    std::string memory_tag;

    // validate strides
    bool valid_strides = ndims == strides.size();
    for (size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] <= 0) valid_strides = false;
    }

    if (valid_strides) {
        struct sort_key_t {
            uint64_t stride_order;
            int64_t outer_block;
            uint64_t idx;
            char dim_char;
        };

        std::vector<sort_key_t> sort_keys(ndims);
        for (size_t i = 0; i < ndims; ++i) {
            sort_key_t key;
            key.stride_order = strides[i];
            key.outer_block = dims[i];
            key.idx = i;
            key.dim_char = template_tag.at(i);
            sort_keys[i] = key;
        }

        // Sort the strides to descending order
        std::sort(sort_keys.begin(), sort_keys.end(),
                [](const sort_key_t &left, const sort_key_t &right) {
            if (left.stride_order < right.stride_order) return false;
            if (left.stride_order == right.stride_order) {
                if (left.outer_block < right.outer_block) return false;
                if (left.outer_block == right.outer_block)
                    return left.idx < right.idx;
            }
            return true;
        });

        char dim_chars[DNNL_MAX_NDIMS + 1];
        for (size_t i = 0; i < ndims; ++i)
            dim_chars[i] = sort_keys[i].dim_char;
        dim_chars[ndims] = '\0';

        memory_tag = std::string(dim_chars);
    } else {
        memory_tag = template_tag.substr(0, ndims);
    }

    // translate a, ab, abc, abcd, etc. to abx
    // translate acb, acdb, etc. to axb
    // This is to handle to different memory tag required for partition and
    // primitive. E.g., for group conv, partition tensor has abcd memory tag,
    // primitive weight has abcde memory tag, in order to copy primitive
    // memory to partition, using abx is a easy way.
    if (!use_x_tag) return memory_tag;

    if (memory_tag == "a" || memory_tag == "ab" || memory_tag == "abc"
            || memory_tag == "abcd" || memory_tag == "abcde")
        return "abx";
    if (memory_tag == "acb" || memory_tag == "acdb" || memory_tag == "acdeb")
        return "axb";

    return memory_tag;
}

bool is_contiguous_memory(const dnnl::graph::logical_tensor::dims &strides,
        const dnnl::graph::logical_tensor::dims &shape,
        const std::string &tag) {

    // for invalid strides, a tensor is always treated as a dense memory.
    bool valid_stride = strides.size() == shape.size()
            && std::all_of(strides.begin(), strides.end(),
                    [](const int64_t &x) { return x > 0; });
    if (!valid_stride) return true;

    std::string template_tag = "abcdefghijk";
    const size_t ndims = shape.size();
    // use plain tag as default if the tensor shape rank changed
    std::string real_tag
            = ndims == tag.length() ? tag : template_tag.substr(0, ndims);

    dnnl_dim_t s = 1;
    for (size_t i = 0; i < ndims; ++i) {
        int adim = real_tag[ndims - 1 - i] - static_cast<int>('a');
        if (adim < 0 || adim >= static_cast<int>(strides.size())) return false;
        if (strides[adim] != s) return false;
        s *= (shape[adim] == 0 ? 1 : shape[adim]);
    }
    return true;
}

dnnl::graph::logical_tensor::dims memory_tag2strides(
        const dnnl::graph::logical_tensor::dims &shape,
        const std::string &tag) {
    std::string template_tag = "abcdefghijk";
    const size_t ndims = shape.size();
    // use plain tag as default if the tensor shape rank changed
    std::string real_tag
            = ndims == tag.length() ? tag : template_tag.substr(0, ndims);

    // map of {a:0, b:1, c:2, d:3, etc}
    std::unordered_map<char, size_t> char2dim;
    for (size_t i = 0; i < real_tag.length(); ++i) {
        char2dim[template_tag[i]] = i;
    }

    dnnl::graph::logical_tensor::dims strides(ndims);
    // start from tag's last char, find corresponding dim
    // and set stride
    // example:
    // shape = 1x3x4x4, tag = acdb
    // char2dim[b] = 1, stride[1] = 1
    // char2dim[d] = 3, stride[3] = stride[1] * shape[1] = 3
    // char2dim[c] = 2, stride[2] = stride[3] * shape[3] = 12
    // char2dim[a] = 0, stride[0] = stride[2] * shape[2] = 48
    dnnl_dim_t s = 1;
    for (size_t i = 0; i < ndims; ++i) {
        size_t adim = char2dim[real_tag[ndims - 1 - i]];
        strides[adim] = s;
        // handle the 0-D tensor case
        if (shape[adim] == 0)
            s = s * 1;
        else
            s = s * shape[adim];
    }
    return strides;
}

void change_format_to_ncx(dims_t &dims) {
    // change format from nxc to ncx
    const auto ndims = static_cast<int>(dims.size());
    dims.insert(dims.begin() + 1, dims[ndims - 1]);
    dims.erase(dims.end() - 1);
}

std::string verbose_partitions_n_ops(
        const std::vector<dnnl::graph::partition> &partitions) {
    std::string s;
    for (const auto &partition : partitions) {
        s += " {" + std::to_string(partition.get_ops_num()) + "}";
    }
    return s;
}

std::string lt_dims2str(const dnnl::graph::logical_tensor::dims &dims) {
    if (dims.empty()) return std::string();

    stringstream_t ss;
    std::copy(
            dims.begin(), dims.end(), std::ostream_iterator<int64_t>(ss, "x"));
    auto res = ss.str();
    return res.substr(0, res.length() - 1);
}

void permute_md(dnn_mem_t &mem, std::vector<int64_t> permutation) {
    std::vector<int> axes(permutation.size(), -1);
    for (int i = 0; i < static_cast<int>(permutation.size()); i++) {
        axes[permutation[i]] = i;
    }
    dnnl::memory::desc md(clone_md(mem.md_));
    dnnl::memory::desc permuted_md = md.permute_axes(axes);
    // Dirty hack to replace md with another one.
    // TODO: replace it with a better solution.
    auto st = dnnl_memory_desc_destroy(mem.md_);
    (void)st;
    assert(st == dnnl_success);
    mem.md_ = clone_md(permuted_md.get());
}

int get_prim_arg_name_from_graph_op_output_offset(
        dnnl::graph::op::kind op_kind, size_t output_offset) {
    switch (op_kind) {
        case dnnl::graph::op::kind::AbsBackward:
        case dnnl::graph::op::kind::AvgPoolBackward:
        case dnnl::graph::op::kind::ClampBackward:
        case dnnl::graph::op::kind::ConvolutionBackwardData:
        case dnnl::graph::op::kind::ConvTransposeBackwardData:
        case dnnl::graph::op::kind::EluBackward:
        case dnnl::graph::op::kind::GELUBackward:
        case dnnl::graph::op::kind::HardSigmoidBackward:
        case dnnl::graph::op::kind::HardSwishBackward:
        case dnnl::graph::op::kind::InterpolateBackward:
        case dnnl::graph::op::kind::LogSoftmaxBackward:
        case dnnl::graph::op::kind::MaxPoolBackward:
        case dnnl::graph::op::kind::MishBackward:
        case dnnl::graph::op::kind::ReLUBackward:
        case dnnl::graph::op::kind::SigmoidBackward:
        case dnnl::graph::op::kind::SoftMaxBackward:
        case dnnl::graph::op::kind::SoftPlusBackward:
        case dnnl::graph::op::kind::SqrtBackward:
        case dnnl::graph::op::kind::TanhBackward: {
            return DNNL_ARG_DIFF_SRC;
        } break;
        case dnnl::graph::op::kind::BatchNormForwardTraining: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return 0; // running mean is not supported for now
            else if (output_offset == 2)
                return 0; // running var is not supported for now
            else if (output_offset == 3)
                return DNNL_ARG_MEAN;
            else if (output_offset == 4)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNorm: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return DNNL_ARG_MEAN;
            else if (output_offset == 2)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }

        } break;
        case dnnl::graph::op::kind::BatchNormTrainingBackward:
        case dnnl::graph::op::kind::LayerNormBackward: {
            if (output_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (output_offset == 1)
                return DNNL_ARG_DIFF_SCALE;
            else if (output_offset == 2)
                return DNNL_ARG_DIFF_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BiasAddBackward: {
            return DNNL_ARG_DIFF_BIAS;
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardWeights:
        case dnnl::graph::op::kind::ConvTransposeBackwardWeights: {
            return DNNL_ARG_DIFF_WEIGHTS;
        } break;
        case dnnl::graph::op::kind::Quantize:
        case dnnl::graph::op::kind::Dequantize:
        case dnnl::graph::op::kind::DynamicDequantize:
        case dnnl::graph::op::kind::DynamicQuantize:
        case dnnl::graph::op::kind::TypeCast:
        case dnnl::graph::op::kind::Reorder: {
            return DNNL_ARG_TO;
        } break;
        case dnnl::graph::op::kind::PReLUBackward: {
            if (output_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (output_offset == 1)
                return DNNL_ARG_DIFF_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::GroupNorm: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return DNNL_ARG_MEAN;
            else if (output_offset == 2)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }

        } break;
        case dnnl::graph::op::kind::RMSNorm: {
            // RMSNorm OP in oneDNN Graph API only have 1 output
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::SoftMax: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return DNNL_ARG_DST_1;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        default: {
            return DNNL_ARG_DST;
        } break;
    }
}

int get_prim_arg_name_from_graph_op_input_offset(
        dnnl::graph::op::kind op_kind, size_t input_offset, bool use_dst) {
    switch (op_kind) {
        case dnnl::graph::op::kind::Add:
        case dnnl::graph::op::kind::BiasAdd:
        case dnnl::graph::op::kind::Divide:
        case dnnl::graph::op::kind::Maximum:
        case dnnl::graph::op::kind::Minimum:
        case dnnl::graph::op::kind::Multiply:
        case dnnl::graph::op::kind::Subtract:
        case dnnl::graph::op::kind::GreaterEqual: {
            if (input_offset == 0)
                return DNNL_ARG_SRC_0;
            else if (input_offset == 1)
                return DNNL_ARG_SRC_1;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        }
        case dnnl::graph::op::kind::Concat: {
            return DNNL_ARG_MULTIPLE_SRC + static_cast<int>(input_offset);
        } break;
        case dnnl::graph::op::kind::Convolution:
        case dnnl::graph::op::kind::ConvTranspose:
        case dnnl::graph::op::kind::MatMul: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else if (input_offset == 2)
                return DNNL_ARG_BIAS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardData:
        case dnnl::graph::op::kind::ConvTransposeBackwardData: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardWeights:
        case dnnl::graph::op::kind::ConvTransposeBackwardWeights: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::PReLU: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::PReLUBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else if (input_offset == 2)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::AvgPoolBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %zu of Avg Pool Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::MaxPoolBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %zu of Max Pool Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::SoftMaxBackward:
        case dnnl::graph::op::kind::LogSoftmaxBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1)
                return DNNL_ARG_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNorm: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE;
            else if (input_offset == 2)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNormBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2)
                return DNNL_ARG_MEAN;
            else if (input_offset == 3)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 4)
                return DNNL_ARG_SCALE;
            else if (input_offset == 5)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormForwardTraining: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_MEAN;
            else if (input_offset == 2)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 3)
                return DNNL_ARG_SCALE;
            else if (input_offset == 4)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormInference: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE;
            else if (input_offset == 2)
                return DNNL_ARG_SHIFT;
            else if (input_offset == 3)
                return DNNL_ARG_MEAN;
            else if (input_offset == 4)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormTrainingBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2)
                return DNNL_ARG_MEAN;
            else if (input_offset == 3)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 4)
                return DNNL_ARG_SCALE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::InterpolateBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %zu of Resampling "
                        "Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ReduceL1:
        case dnnl::graph::op::kind::ReduceL2:
        case dnnl::graph::op::kind::ReduceMax:
        case dnnl::graph::op::kind::ReduceMin:
        case dnnl::graph::op::kind::ReduceMean:
        case dnnl::graph::op::kind::ReduceProd:
        case dnnl::graph::op::kind::ReduceSum: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %zu of Reduction",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::AbsBackward:
        case dnnl::graph::op::kind::ClampBackward:
        case dnnl::graph::op::kind::EluBackward:
        case dnnl::graph::op::kind::GELUBackward:
        case dnnl::graph::op::kind::HardSigmoidBackward:
        case dnnl::graph::op::kind::HardSwishBackward:
        case dnnl::graph::op::kind::MishBackward:
        case dnnl::graph::op::kind::ReLUBackward:
        case dnnl::graph::op::kind::SigmoidBackward:
        case dnnl::graph::op::kind::SoftPlusBackward:
        case dnnl::graph::op::kind::SqrtBackward:
        case dnnl::graph::op::kind::TanhBackward: {
            if (input_offset == 0)
                return use_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::Quantize:
        case dnnl::graph::op::kind::Dequantize:
        case dnnl::graph::op::kind::TypeCast:
        case dnnl::graph::op::kind::Reorder: {
            return DNNL_ARG_FROM;
        } break;
        case dnnl::graph::op::kind::DynamicDequantize: {
            if (input_offset == 0)
                return DNNL_ARG_FROM;
            else if (input_offset == 1)
                return DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM;
            else if (input_offset == 2)
                return DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::DynamicQuantize: {
            if (input_offset == 0)
                return DNNL_ARG_FROM;
            else if (input_offset == 1)
                return DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO;
            else if (input_offset == 2)
                return DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::Select: {
            if (input_offset == 0)
                return DNNL_ARG_SRC_2;
            else if (input_offset == 1)
                return DNNL_ARG_SRC_0;
            else if (input_offset == 2)
                return DNNL_ARG_SRC_1;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::GroupNorm: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE;
            else if (input_offset == 2)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::RMSNorm: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE; // gamma
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::Dropout: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_ATTR_DROPOUT_SEED;
            else if (input_offset == 2)
                return DNNL_ARG_ATTR_DROPOUT_OFFSET;
            else if (input_offset == 3)
                return DNNL_ARG_ATTR_DROPOUT_PROBABILITY;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %zu",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        default: {
            return DNNL_ARG_SRC;
        } break;
    }
}

dnnl::graph::logical_tensor::layout_type str2layout(
        const std::string &layout_type) {
    if (layout_type == "strided")
        return dnnl::graph::logical_tensor::layout_type::strided;
    else if (layout_type == "any")
        return dnnl::graph::logical_tensor::layout_type::any;
    else if (layout_type == "opaque")
        return dnnl::graph::logical_tensor::layout_type::opaque;
    else
        return dnnl::graph::logical_tensor::layout_type::undef;
}

engine_t make_graph_engine(bool use_host) {

    dnnl::graph::allocator &alloc = get_graph_allocator(use_host);

    if (use_host || is_cpu()) {
        return engine_t(make_engine_with_allocator(dnnl::engine::kind::cpu,
                static_cast<size_t>(engine_index), alloc));
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        return engine_t(make_engine_with_allocator(dnnl::engine::kind::gpu,
                static_cast<size_t>(engine_index), alloc));
#else
        assert(!"unsupported gpu runtime");
        return engine_t(dnnl::engine {});
#endif
    }
}

dnnl_data_type_t convert_dt(const dnnl::graph::logical_tensor::data_type dt) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case graph_dt::f16: return dnnl_f16;
        case graph_dt::bf16: return dnnl_bf16;
        case graph_dt::f32: return dnnl_f32;
        case graph_dt::s32: return dnnl_s32;
        case graph_dt::s8: return dnnl_s8;
        case graph_dt::u8: return dnnl_u8;
        // Use `u8` instead of `boolean` in the reference path.
        // `dnn_graph_mem_t` will use the data type from the logical tensor and
        // the `u8` data handle.
        case graph_dt::boolean: return dnnl_u8;
        case graph_dt::f8_e5m2: return dnnl_f8_e5m2;
        case graph_dt::f8_e4m3: return dnnl_f8_e4m3;
        case graph_dt::s4: return dnnl_s4;
        case graph_dt::u4: return dnnl_u4;
        case graph_dt::s64: return dnnl_s64;
        case graph_dt::undef:
        default: return dnnl_data_type_undef;
    }
}

} // namespace graph
