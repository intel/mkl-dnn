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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
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

void print_test_case(memory::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

void bench_sdpa_primitives(engine::kind ekind, memory::data_type dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Intermediate data type
    const memory::data_type dt_inter = memory::data_type::f32;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, dt, memory::format_tag::abdc);
    auto score_md = memory::desc(score_sz, dt_inter, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);
    auto probs_md = memory::desc(score_sz, dt, memory::format_tag::abcd);

    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(scratchpad_mode::user);
    post_ops bmm1_po;
    bmm1_po.append_binary(algorithm::binary_div, scale_md);
    bmm1_po.append_binary(algorithm::binary_add, mask_md);
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, query_md, key_md, score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    // attention_probs = softmax(masked_score)
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto softmax_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::softmax_accurate, score_md,
            probs_md, /* axis = */ score_md.get_ndims() - 1, softmax_attr);
    auto softmax_prim = softmax_forward(softmax_pd);

    // attention_output = attention_probs x value
    auto value_md = memory::desc(v_sz, dt, memory::format_tag::abcd);
    auto output_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    primitive_attr bmm2_attr;
    bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, probs_md, value_md, output_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // Create memory objects
    auto m_query = memory(query_md, eng);
    auto m_key = memory(key_md, eng);
    auto m_scale = memory(scale_md, eng);
    auto m_mask = memory(mask_md, eng);
    auto m_value = memory(value_md, eng);
    auto m_output = memory(output_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(
            product(scale_sz), (float)std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(v_sz));
    std::vector<float> output_data(product(q_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(scale_data.data(), m_scale);
    write_to_dnnl_memory(mask_data.data(), m_mask);
    write_to_dnnl_memory(value_data.data(), m_value);

    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, memory::format_tag::a);

    // allocate intermediate memory
    auto m_score = memory(score_md, eng);
    auto m_probs = memory(probs_md, eng);
    auto m_scratchpad = memory(scratchpad_md, eng);

    const auto loop = [&]() {
        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_query}, {DNNL_ARG_WEIGHTS, m_key},
                        {DNNL_ARG_DST, m_score},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                m_scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_mask},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        softmax_prim.execute(strm,
                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_DST, m_probs},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_probs}, {DNNL_ARG_WEIGHTS, m_value},
                        {DNNL_ARG_DST, m_output},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        loop();
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
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
    const dims mask_sz = {p.mb, 1, p.query_num, p.seq_len};

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

    // scaled_score = score / scale
    auto scale = logical_tensor(id++, dt, scale_sz, layout_type::strided);
    auto scaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto mask = logical_tensor(id++, dt, mask_sz, layout_type::strided);
    auto masked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

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
    sdpa.add_op(scale_div);
    sdpa.add_op(mask_add);
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
            {query, key, scale, mask, value}, {output}, eng);

    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_mask = tensor(mask, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qv_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(
            product(scale_sz), (float)std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(k_sz));
    std::vector<float> output_data(product(qv_sz));

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
    std::cerr << "Usage: graph-sdpa-cpp [cpu|gpu]\n"
                 "       graph-sdpa-cpp [cpu|gpu] <mb> <seq_len> "
                 "<head_num> <head_size> [<query_num>]\n\n"
                 "On CPU, it's recommended to test with numactl and memory "
                 "allocation tools like jemalloc or tcmalloc.\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

enum class api_kind {
    primitive,
    graph,
};

void bench(api_kind api, engine::kind ekind, dnnl_data_type_t dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    try {
        if (api == api_kind::primitive) {
            bench_sdpa_primitives(
                    ekind, static_cast<memory::data_type>(dt), p, time_limit);
        } else {
            // api == api_kind::graph
            bench_sdpa(ekind, static_cast<logical_tensor::data_type>(dt), p,
                    time_limit);
            get_mem_pool().clear();
        }
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
    sdpa_dims_t params = {32, 384, 16, 64, 384};

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

    bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(api_kind::graph, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(api_kind::graph, ekind, dnnl_f16, params, 2000.0 /*ms*/);

    bench(api_kind::primitive, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(api_kind::primitive, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(api_kind::primitive, ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 5), argc, argv);
}
