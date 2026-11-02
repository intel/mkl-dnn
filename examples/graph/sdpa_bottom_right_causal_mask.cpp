/*******************************************************************************
* Copyright 2025 Intel Corporation
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
using property_type = logical_tensor::property_type;
using data_type = logical_tensor::data_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
    dim query_num;
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

void print_test_case(memory::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

const char *get_type_string(logical_tensor::data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == logical_tensor::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

void bench_sdpa(engine::kind ekind, logical_tensor::data_type dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const dims qv_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const dims k_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const dims scale_sz = {1};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // score = query x key.T
    auto query = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto score = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scale_mul_out_lt = score * scale
    const logical_tensor::data_type dt_scale = logical_tensor::data_type::f32;
    auto scale = logical_tensor(id++, dt_scale, 0, layout_type::strided,
            property_type::host_scalar);
    auto scaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_mul = op(id++, op::kind::Multiply, "scale_mul");
    scale_mul.add_inputs({score, scale});
    scale_mul.add_outputs({scaled_score});

    auto index_row = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    auto gen_index_row = op(id++, op::kind::GenIndex, "gen_index_row");
    gen_index_row.set_attr<int64_t>(op::attr::axis, -2);
    gen_index_row.add_inputs({scaled_score});
    gen_index_row.add_outputs({index_row});

    auto seq_len_kv = logical_tensor(id++, data_type::s32, 0,
            layout_type::strided, property_type::host_scalar);
    auto mask_add_out = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({index_row, seq_len_kv});
    mask_add.add_outputs({mask_add_out});

    auto seq_len_q = logical_tensor(id++, data_type::s32, 0,
            layout_type::strided, property_type::host_scalar);
    auto mask_sub_out = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    auto mask_sub = op(id++, op::kind::Subtract, "mask_sub");
    mask_sub.add_inputs({mask_add_out, seq_len_q});
    mask_sub.add_outputs({mask_sub_out});

    auto index_col = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    auto gen_index_col = op(id++, op::kind::GenIndex, "gen_index_col");
    gen_index_col.set_attr<int64_t>(op::attr::axis, -1);
    gen_index_col.add_inputs({scaled_score});
    gen_index_col.add_outputs({index_col});

    auto mask_ge_out = logical_tensor(
            id++, data_type::boolean, score_sz, layout_type::strided);
    auto mask_ge = op(id++, op::kind::GreaterEqual, "mask_ge");
    mask_ge.add_inputs({mask_sub_out, index_col});
    mask_ge.add_outputs({mask_ge_out});

    auto neg_inf = logical_tensor(id++, dt_scale, 0, layout_type::strided,
            property_type::host_scalar);
    auto masked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_select = op(id++, op::kind::Select, "mask_select");
    mask_select.add_inputs({mask_ge_out, scaled_score, neg_inf});
    mask_select.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});

    // attention_output = attention_probs x value
    auto value = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa(ekind);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_mul);
    sdpa.add_op(gen_index_row);
    sdpa.add_op(mask_add);
    sdpa.add_op(mask_sub);
    sdpa.add_op(gen_index_col);
    sdpa.add_op(mask_ge);
    sdpa.add_op(mask_select);
    sdpa.add_op(softmax);
    sdpa.add_op(bmm2);
    sdpa.finalize();

    // Get partitions from the sdpa graph.
    std::vector<partition> partitions = sdpa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported sdpa" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, seq_len_kv, seq_len_q, neg_inf, value},
            {output}, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qv_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> value_data(product(k_sz));
    std::vector<float> output_data(product(qv_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);

    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);

    // Create host scalar tensors
    int32_t q_len = static_cast<int32_t>(p.query_num);
    int32_t kv_len = static_cast<int32_t>(p.seq_len);
    float scale_val = (float)std::sqrt(p.head_size);
    float neg_inf_val = -std::numeric_limits<float>::infinity();
    auto ts_mask_add = tensor::make_scalar_tensor(seq_len_kv, &kv_len);
    auto ts_mask_sub = tensor::make_scalar_tensor(seq_len_q, &q_len);
    auto ts_scale = tensor::make_scalar_tensor(scale, &scale_val);
    auto ts_neg_inf = tensor::make_scalar_tensor(neg_inf, &neg_inf_val);

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask_add, ts_mask_sub, ts_neg_inf,
                    ts_value},
            {ts_output});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask_add, ts_mask_sub, ts_neg_inf,
                    ts_value},
            {ts_output});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        cp.execute(strm,
                {ts_query, ts_key, ts_scale, ts_mask_add, ts_mask_sub,
                        ts_neg_inf, ts_value},
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
    std::cerr << "Usage: graph-sdpa-bottom-right-causal-mask-cpp [cpu|gpu]\n"
                 "       graph-sdpa-bottom-right-causal-mask-cpp [cpu|gpu] "
                 "<mb> <seq_len> <head_num> <head_size> [<query_num>]\n\n"
                 "On CPU, it's recommended to test with numactl and memory "
                 "allocation tools like jemalloc or tcmalloc.\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, dnnl_data_type_t dt, const sdpa_dims_t &p,
        double time_limit = 0.) {
    try {
        bench_sdpa(ekind, static_cast<data_type>(dt), p, time_limit);
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
    sdpa_dims_t params = {2, 128, 16, 64, 128};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.query_num = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else if (argc == 7) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
            params.query_num = std::atoi(argv[6]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    return 0;
#else
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 5), argc, argv);
#endif
}
