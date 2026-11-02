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

struct gqa_dims_t {
    dim mb;
    dim seq_len;
    dim q_head_num;
    dim kv_head_num;
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
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const gqa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", q_head_num = " << p.q_head_num
              << ", kv_head_num = " << p.kv_head_num
              << ", head_size = " << p.head_size;
    std::cout << "] " << std::flush;
}

bool bench_gqa_forward(engine::kind ekind, logical_tensor::data_type dt,
        dnnl::stream &strm, dnnl::engine &eng, const tensor &ts_query,
        const tensor &ts_key, const tensor &ts_scale, const tensor &ts_mask,
        const tensor &ts_value, tensor &ts_output, tensor &ts_stats,
        double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // Extract logical_tensor and dimensions from tensors
    auto query = ts_query.get_logical_tensor();
    auto key = ts_key.get_logical_tensor();
    auto value = ts_value.get_logical_tensor();
    auto scale = ts_scale.get_logical_tensor();
    auto mask = ts_mask.get_logical_tensor();

    const dims q_sz = query.get_dims();
    const dims kv_sz = key.get_dims();
    const dims scale_sz = scale.get_dims();
    const dims score_sz = mask.get_dims();

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 1;

    // score = query x key.T
    auto score = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scaled_score = score / scale
    auto scaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto masked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto stats = logical_tensor(id++, dt_inter, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs, stats});

    // attention_output = attention_probs x value
    auto output = logical_tensor(id++, dt, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a gqa graph with engine kind and operations.
    dnnl::graph::graph gqa(ekind);
    gqa.add_op(bmm1);
    gqa.add_op(scale_div);
    gqa.add_op(mask_add);
    gqa.add_op(softmax);
    gqa.add_op(bmm2);
    gqa.finalize();

    // Get partitions from the gqa graph.
    std::vector<partition> partitions = gqa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported gqa" << std::endl;
        return false;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, mask, value}, {output, stats}, eng);

    // Update output tensor objects with correct logical tensors
    auto output_w_shape = cp.query_logical_tensor(output.get_id());
    auto stats_w_shape = cp.query_logical_tensor(stats.get_id());
    ts_output = tensor(output_w_shape, eng);
    ts_stats = tensor(stats_w_shape, eng);

    // Execute the compiled partition of gqa.
    cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
            {ts_output, ts_stats});

    // Wait for the computation to finish.
    strm.wait();

    if (quick_test) return true;

    // First run (forward).
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
            {ts_output, ts_stats});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    // Timing runs (forward).
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
                {ts_output, ts_stats});
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results (forward).
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "forward graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    return true;
}

bool bench_gqa_backward(engine::kind ekind, logical_tensor::data_type dt,
        dnnl::stream &strm, dnnl::engine &eng, const tensor &ts_query,
        const tensor &ts_key, const tensor &ts_scale, const tensor &ts_mask,
        const tensor &ts_value, const tensor &ts_output, const tensor &ts_stats,
        const tensor &ts_doutput, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // Extract logical_tensor and dimensions from tensors
    auto query = ts_query.get_logical_tensor();
    auto key = ts_key.get_logical_tensor();
    auto value = ts_value.get_logical_tensor();
    auto scale = ts_scale.get_logical_tensor();
    auto mask = ts_mask.get_logical_tensor();
    auto output = ts_output.get_logical_tensor();
    auto stats = ts_stats.get_logical_tensor();
    auto doutput = ts_doutput.get_logical_tensor();

    const dims q_sz = query.get_dims();
    const dims kv_sz = key.get_dims();
    const dims scale_sz = scale.get_dims();
    const dims score_sz = mask.get_dims();
    const dims stats_sz = stats.get_dims();
    const dims output_sz = output.get_dims();

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 1;

    // score = query x key.T
    auto score = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scaled_score = score / scale
    auto scaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto masked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // attention_probs = softmax(masked_score) = exp(masked_score - stats)
    auto sub_out
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto subtract = op(id++, op::kind::Subtract, "subtract");
    subtract.add_inputs({masked_score, stats});
    subtract.add_outputs({sub_out});

    auto probs = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto exp = op(id++, op::kind::Exp, "exp");
    exp.add_inputs({sub_out});
    exp.add_outputs({probs});

    // the following bmm doesn't support different input dtypes, insert a typecast
    auto probs_cast = probs;
    auto typecast = op(id++, op::kind::TypeCast, "typecast");
    if (dt != dt_inter) {
        probs_cast = logical_tensor(id++, dt, score_sz, layout_type::strided);
        typecast.add_inputs({probs});
        typecast.add_outputs({probs_cast});
    }

    // compute dvalue = P^T * doutput
    auto dvalue = logical_tensor(id++, dt, q_sz, layout_type::strided);
    auto bmm_p_do = op(id++, op::kind::MatMul, "bmm_dv");
    bmm_p_do.set_attr<bool>(op::attr::transpose_a, true);
    bmm_p_do.add_inputs({probs_cast, doutput});
    bmm_p_do.add_outputs({dvalue});

    // for gqa, dv needs an additional reduce
    auto dvalue_reduced = logical_tensor(id++, dt, kv_sz, layout_type::strided);
    auto reduce_dv = op(id++, op::kind::ReduceSum, "reduce_dv");
    reduce_dv.set_attr<std::vector<int64_t>>(op::attr::axes, {2});
    reduce_dv.set_attr<bool>(op::attr::keep_dims, true);
    reduce_dv.add_inputs({dvalue});
    reduce_dv.add_outputs({dvalue_reduced});

    // compute dprobs = doutput * value^T
    auto dprobs
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm_do_v = op(id++, op::kind::MatMul, "bmm_dprobs");
    bmm_do_v.set_attr<bool>(op::attr::transpose_b, true);
    bmm_do_v.add_inputs({doutput, value});
    bmm_do_v.add_outputs({dprobs});

    // compute dmasked_score = P * (dprobs - ReduceSum(O * dO))
    // decomposed softmax backward: dS = P * (dP - rowsum(O * dO))
    auto o_do_out
            = logical_tensor(id++, dt_inter, output_sz, layout_type::strided);
    auto o_do_mul = op(id++, op::kind::Multiply, "mul_o_do");
    o_do_mul.add_inputs({output, doutput});
    o_do_mul.add_outputs({o_do_out});

    auto correction_out
            = logical_tensor(id++, dt_inter, stats_sz, layout_type::strided);
    auto correction = op(id++, op::kind::ReduceSum, "reducesum_correction");
    correction.set_attr<std::vector<int64_t>>(op::attr::axes, {4});
    correction.set_attr<bool>(op::attr::keep_dims, true);
    correction.add_inputs({o_do_out});
    correction.add_outputs({correction_out});

    auto dp_corrected_out
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto dp_corrected_op = op(id++, op::kind::Subtract, "sub_dp_corrected");
    dp_corrected_op.add_inputs({dprobs, correction_out});
    dp_corrected_op.add_outputs({dp_corrected_out});

    auto dmasked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto softmax_bwd_mul = op(id++, op::kind::Multiply, "mul_softmax_bwd");
    softmax_bwd_mul.add_inputs({probs, dp_corrected_out});
    softmax_bwd_mul.add_outputs({dmasked_score});

    // compute dscored_score = dmasked_score / scale
    auto dscaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div2 = op(id++, op::kind::Divide, "scale_div");
    scale_div2.add_inputs({dmasked_score, scale});
    scale_div2.add_outputs({dscaled_score});

    // the following bmm doesn't support different input dtypes, insert a typecast
    auto dscaled_score_cast = dscaled_score;
    auto typecast2 = op(id++, op::kind::TypeCast, "typecast");
    if (dt != dt_inter) {
        dscaled_score_cast
                = logical_tensor(id++, dt, score_sz, layout_type::strided);
        typecast2.add_inputs({dscaled_score});
        typecast2.add_outputs({dscaled_score_cast});
    }

    // compute dquery = dscaled_score * key
    auto dquery = logical_tensor(id++, dt, q_sz, layout_type::strided);
    auto bmm_dscaled_score_k = op(id++, op::kind::MatMul, "bmm_dq");
    bmm_dscaled_score_k.add_inputs({dscaled_score_cast, key});
    bmm_dscaled_score_k.add_outputs({dquery});

    // compute dkey = dscaled_score^T * query
    auto dkey = logical_tensor(id++, dt, q_sz, layout_type::strided);
    auto bmm_dscaled_score_q = op(id++, op::kind::MatMul, "bmm_dk");
    bmm_dscaled_score_q.set_attr<bool>(op::attr::transpose_a, true);
    bmm_dscaled_score_q.add_inputs({dscaled_score_cast, query});
    bmm_dscaled_score_q.add_outputs({dkey});

    // for gqa, dk needs an additional reduce
    auto dkey_reduced = logical_tensor(id++, dt, kv_sz, layout_type::strided);
    auto reduce_dk = op(id++, op::kind::ReduceSum, "reduce_dk");
    reduce_dk.set_attr<std::vector<int64_t>>(op::attr::axes, {2});
    reduce_dk.set_attr<bool>(op::attr::keep_dims, true);
    reduce_dk.add_inputs({dkey});
    reduce_dk.add_outputs({dkey_reduced});

    // Construct a gqa graph with engine kind and operations.
    dnnl::graph::graph gqa_bwd(ekind);
    gqa_bwd.add_op(bmm1);
    gqa_bwd.add_op(scale_div);
    gqa_bwd.add_op(mask_add);
    gqa_bwd.add_op(subtract);
    gqa_bwd.add_op(exp);
    gqa_bwd.add_op(bmm_p_do);
    gqa_bwd.add_op(bmm_do_v);
    gqa_bwd.add_op(o_do_mul);
    gqa_bwd.add_op(correction);
    gqa_bwd.add_op(dp_corrected_op);
    gqa_bwd.add_op(softmax_bwd_mul);
    gqa_bwd.add_op(scale_div2);
    gqa_bwd.add_op(bmm_dscaled_score_q);
    gqa_bwd.add_op(bmm_dscaled_score_k);
    gqa_bwd.add_op(reduce_dv);
    gqa_bwd.add_op(reduce_dk);
    if (dt != dt_inter) {
        // Add typecast op to the gqa graph.
        gqa_bwd.add_op(typecast);
        gqa_bwd.add_op(typecast2);
    }

    gqa_bwd.finalize();

    // Get partitions from the gqa graph.
    std::vector<partition> partitions = gqa_bwd.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported gqa" << std::endl;
        return false;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, mask, value, output, stats, doutput},
            {dquery, dkey_reduced, dvalue_reduced}, eng);

    // Create tensor objects
    auto ts_dquery = tensor(dquery, eng);
    auto ts_dkey = tensor(dkey_reduced, eng);
    auto ts_dvalue = tensor(dvalue_reduced, eng);

    // Execute the compiled partition of sdpa.
    cp.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask, ts_value, ts_output, ts_stats,
                    ts_doutput},
            {ts_dquery, ts_dkey, ts_dvalue});

    // Wait for the computation to finish.
    strm.wait();

    if (quick_test) return true;

    // First run (backward).
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask, ts_value, ts_output, ts_stats,
                    ts_doutput},
            {ts_dquery, ts_dkey, ts_dvalue});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    // Timing runs (backward).
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++)
        cp.execute(strm,
                {ts_query, ts_key, ts_scale, ts_mask, ts_value, ts_output,
                        ts_stats, ts_doutput},
                {ts_dquery, ts_dkey, ts_dvalue});
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results (backward).
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "backward graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    return true;
}

void bench_gqa(engine::kind ekind, logical_tensor::data_type dt,
        const gqa_dims_t &p, double time_limit = 0.) {
    print_test_case(dt, p);

    // Create execution dnnl::engine.
    allocator alloc = create_allocator(ekind);
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes
    dnnl_dim_t head_rep = p.q_head_num / p.kv_head_num;
    const dims q_sz = {p.mb, p.kv_head_num, head_rep, p.seq_len, p.head_size};
    const dims kv_sz = {p.mb, p.kv_head_num, 1, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.kv_head_num, head_rep, p.seq_len, p.seq_len};
    const dims stats_sz = {p.mb, p.kv_head_num, head_rep, p.seq_len, 1};
    const dims scale_sz = {1};

    // Create logical tensors for input tensors
    auto query_lt = logical_tensor(100, dt, q_sz, layout_type::strided);
    auto key_lt = logical_tensor(101, dt, kv_sz, layout_type::strided);
    auto scale_lt = logical_tensor(102, dt, scale_sz, layout_type::strided);
    auto mask_lt = logical_tensor(103, dt, score_sz, layout_type::strided);
    auto value_lt = logical_tensor(104, dt, kv_sz, layout_type::strided);

    // Create tensor objects
    tensor ts_query(query_lt, eng);
    tensor ts_key(key_lt, eng);
    tensor ts_scale(scale_lt, eng);
    tensor ts_mask(mask_lt, eng);
    tensor ts_value(value_lt, eng);
    tensor ts_output, ts_stats;

    // Allocate and initialize data
    std::vector<float> query_data(product(q_sz));
    std::vector<float> key_data(product(kv_sz));
    std::vector<float> scale_data(
            product(scale_sz), (float)std::sqrt(p.head_size));
    std::vector<float> mask_data(product(score_sz));
    std::vector<float> value_data(product(kv_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor objects
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(scale_data.data(), ts_scale);
    write_to_dnnl_tensor(mask_data.data(), ts_mask);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Run forward pass
    bool success = bench_gqa_forward(ekind, dt, strm, eng, ts_query, ts_key,
            ts_scale, ts_mask, ts_value, ts_output, ts_stats, time_limit);
    if (!success) return;

    // Prepare output gradients
    const dims doutput_sz = ts_output.get_logical_tensor().get_dims();
    auto doutput_lt = logical_tensor(105, dt, doutput_sz, layout_type::strided);
    tensor ts_doutput(doutput_lt, eng);

    // Allocate and initialize gradients
    std::vector<float> doutput_data(product(doutput_sz));
    fill_random(doutput_data);
    write_to_dnnl_tensor(doutput_data.data(), ts_doutput);

    // Run backward pass
    bench_gqa_backward(ekind, dt, strm, eng, ts_query, ts_key, ts_scale,
            ts_mask, ts_value, ts_output, ts_stats, ts_doutput, time_limit);
}

void bad_args() {
    std::cerr << "Usage: graph-gqa-training-cpp [cpu|gpu]\n"
                 "       graph-gqa-training-cpp [cpu|gpu] <mb> <seq_len> "
                 "<q_head_num> <kv_head_num> <head_size>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, dnnl_data_type_t dt, const gqa_dims_t &p,
        double time_limit = 0.) {
    try {
        bench_gqa(ekind, static_cast<logical_tensor::data_type>(dt), p,
                time_limit);
        get_mem_pool().clear();
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type) {
            std::cout << "unsupported gqa" << std::endl;
        } else
            throw;
    }
}

void gqa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    gqa_dims_t params = {2, 128, 16, 2, 64};

    if (argc > 2) {
        if (argc == 7) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.q_head_num = std::atoi(argv[4]);
            params.kv_head_num = std::atoi(argv[5]);
            params.head_size = std::atoi(argv[6]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.kv_head_num <= 0
                || params.q_head_num <= 0 || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            gqa_perf, parse_engine_kind(argc, argv, 5), argc, argv);
}
