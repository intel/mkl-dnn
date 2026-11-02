/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using data_type = logical_tensor::data_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
};

static const int min_runs = 4;

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (i % seq_len < pos)
            mask[i] = 0.f;
        else
            mask[i] = -1 * std::numeric_limits<float>::infinity();
    }
}

const char *get_type_string(logical_tensor::data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == logical_tensor::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
    TYPE_CASE(u8);
    TYPE_CASE(s8);
    TYPE_CASE(f8_e4m3);
    TYPE_CASE(f8_e5m2);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size;
    std::cout << "] " << std::flush;
}

void bench_quantized_sdpa(engine::kind ekind, const data_type dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const dims qkv_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.seq_len, p.seq_len};
    const dims scale_sz = {1};
    const dims mask_sz = {p.mb, 1, 1, p.seq_len};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // insert the dequant for quantized query to f32 query
    auto q_x8 = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto q_f32 = logical_tensor(
            id++, data_type::f32, qkv_sz, layout_type::strided);
    auto q_deq = op(id++, op::kind::Dequantize, "q_deq");
    q_deq.set_attr<std::string>(op::attr::qtype, "per_tensor");
    q_deq.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    q_deq.add_input(q_x8);
    q_deq.add_output(q_f32);

    // insert the dequant for quantized key to f32 key
    auto k_x8 = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto k_f32 = logical_tensor(
            id++, data_type::f32, qkv_sz, layout_type::strided);
    auto k_deq = op(id++, op::kind::Dequantize, "k_deq");
    k_deq.set_attr<std::string>(op::attr::qtype, "per_tensor");
    k_deq.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    k_deq.add_input(k_x8);
    k_deq.add_output(k_f32);

    // score = query x key.T.
    auto score = logical_tensor(
            id++, data_type::f32, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({q_f32, k_f32});
    bmm1.add_output(score);

    // scaled_score = score / scale
    auto scale = logical_tensor(
            id++, data_type::f32, scale_sz, layout_type::strided);
    auto scaled_score = logical_tensor(
            id++, data_type::f32, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto mask = logical_tensor(
            id++, data_type::f32, mask_sz, layout_type::strided);
    auto masked_score = logical_tensor(
            id++, data_type::f32, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(
            id++, data_type::f32, score_sz, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});

    // quantize the probs from f32 to quantized type
    auto probs_x8 = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto p_quant = op(id++, op::kind::Quantize, "p_quant");
    p_quant.set_attr<std::string>(op::attr::qtype, "per_tensor");
    p_quant.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    p_quant.add_input(probs);
    p_quant.add_output(probs_x8);

    // dequant the probs from quantized type to f32
    auto probs_f32 = logical_tensor(
            id++, data_type::f32, score_sz, layout_type::strided);
    auto p_deq = op(id++, op::kind::Dequantize, "p_deq");
    p_deq.set_attr<std::string>(op::attr::qtype, "per_tensor");
    p_deq.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    p_deq.add_input(probs_x8);
    p_deq.add_output(probs_f32);

    // dequant the value from quantized type to f32
    auto v_x8 = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto v_f32 = logical_tensor(
            id++, data_type::f32, qkv_sz, layout_type::strided);
    auto v_deq = op(id++, op::kind::Dequantize, "v_deq");
    v_deq.set_attr<std::string>(op::attr::qtype, "per_tensor");
    v_deq.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    v_deq.add_input(v_x8);
    v_deq.add_output(v_f32);

    // attention_output = attention_probs x value.
    auto output = logical_tensor(
            id++, data_type::f32, qkv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs_f32, v_f32});
    bmm2.add_outputs({output});

    // quantize the output from f32 to quantized type.
    auto output_x8 = logical_tensor(id++, dt, qkv_sz, layout_type::strided);
    auto o_quant = op(id++, op::kind::Quantize, "o_quant");
    o_quant.set_attr<std::string>(op::attr::qtype, "per_tensor");
    o_quant.set_attr<std::vector<float>>(op::attr::scales, {0.25f});
    o_quant.add_input(output);
    o_quant.add_output(output_x8);

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa(ekind);
    sdpa.add_op(q_deq);
    sdpa.add_op(k_deq);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_div);
    sdpa.add_op(mask_add);
    sdpa.add_op(softmax);
    sdpa.add_op(p_quant);
    sdpa.add_op(p_deq);
    sdpa.add_op(v_deq);
    sdpa.add_op(bmm2);
    sdpa.add_op(o_quant);
    sdpa.finalize();

    // Get partitions from the sdpa graph.
    std::vector<partition> partitions = sdpa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported sdpa partition" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {q_x8, k_x8, scale, mask, v_x8}, {output_x8}, eng);

    // Create tensor objects
    auto ts_query = tensor(q_x8, eng);
    auto ts_key = tensor(k_x8, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_mask = tensor(mask, eng);
    auto ts_value = tensor(v_x8, eng);
    auto ts_output = tensor(output_x8, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qkv_sz));
    std::vector<float> key_data(product(qkv_sz));
    std::vector<float> scale_data(
            product(scale_sz), (float)std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(qkv_sz));
    std::vector<float> output_data(product(qkv_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(scale_data.data(), ts_scale);
    write_to_dnnl_tensor(mask_data.data(), ts_mask);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp.execute(
            strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value}, {ts_output});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(
            strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value}, {ts_output});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
                {ts_output});
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
}

void bad_args() {
    std::cerr << "Usage: graph-sdpa-quantized-cpp [cpu|gpu]\n"
                 "       graph-sdpa-quantized-cpp [cpu|gpu] <mb> <seq_len> "
                 "<head_num> <head_size>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, const data_type dt, const sdpa_dims_t &p,
        double time_limit = 0.) {
    try {
        bench_quantized_sdpa(ekind, dt, p, time_limit);
        get_mem_pool().clear();
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type) {
            std::cout << "unsupported sdpa" << std::endl;
        } else
            throw;
    }
}

void sdpa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    sdpa_dims_t params = {2, 128, 16, 64};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(ekind, data_type::f8_e4m3, params, 2000.0 /*ms*/);
    bench(ekind, data_type::f8_e5m2, params, 2000.0 /*ms*/);
    bench(ekind, data_type::s8, params, 2000.0 /*ms*/);
    bench(ekind, data_type::u8, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 4), argc, argv);
}
