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
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct mlp_dims_t {
    dim mb;
    dim ic;
    dim oc;
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

size_t size_of(logical_tensor::data_type dt) {
    // This example only supports f32, bf16, and f16.
    switch (dt) {
        case logical_tensor::data_type::f32: return 4;
        case logical_tensor::data_type::bf16:
        case logical_tensor::data_type::f16: return 2;
        default: assert(!"unknown data_type");
    }

    return (size_t)-1; /* not supposed to be reachable */
}

void print_test_case(logical_tensor::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

void bench_gated_mlp(engine::kind ekind, logical_tensor::data_type dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // input shape
    const dims src_sz = {p.mb, p.ic};
    // weight0/weight1 shape: fc_gate and fc_up
    const dims wei0_sz = {p.ic, p.oc};
    // hidden shape
    const dims hd_sz = {p.mb, p.oc};
    // weight2 shape: fc_down
    const dims wei2_sz = {p.oc, p.ic};
    // output shape
    const dims out_sz = {p.mb, p.ic};

    // Combined wei0 and wei1 together into shape (ic, 2 * oc), assuming the
    // first part is wei0 for fc_gate and the second part is wei1 for fc_up.
    const dims combined_wei0_sz = {p.ic, 2 * p.oc};
    const dims combined_wei0_st = {2 * p.oc, 1};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // This logical tensor is not part of the graph but is used to generate the
    // big chunk of device memory which should be already there in real user
    // application or framework.
    auto combined_wei0
            = logical_tensor(id++, dt, combined_wei0_sz, layout_type::strided);

    // fc_gate: wei0 is non-contiguous now.
    auto src = logical_tensor(id++, dt, src_sz, layout_type::strided);
    auto wei0 = logical_tensor(id++, dt, wei0_sz, combined_wei0_st);
    auto out0 = logical_tensor(id++, dt_inter, hd_sz, layout_type::strided);
    auto fc_gate = op(id++, op::kind::MatMul, "fc_gate");
    fc_gate.add_inputs({src, wei0});
    fc_gate.add_outputs({out0});

    // fc_up: wei1 is non-contiguous now.
    auto wei1 = logical_tensor(id++, dt, wei0_sz, combined_wei0_st);
    auto out1 = logical_tensor(id++, dt_inter, hd_sz, layout_type::strided);
    auto fc_up = op(id++, op::kind::MatMul, "fc_up");
    fc_up.add_inputs({src, wei1});
    fc_up.add_outputs({out1});

    // activation swish: sigmoid
    auto out2 = logical_tensor(id++, dt_inter, hd_sz, layout_type::strided);
    auto swi_sig = op(id++, op::kind::Sigmoid, "swish/sigmoid");
    swi_sig.add_inputs({out0});
    swi_sig.add_outputs({out2});

    // activation swish: multiply
    auto out3 = logical_tensor(id++, dt_inter, hd_sz, layout_type::strided);
    auto swi_mul = op(id++, op::kind::Multiply, "swish/multiply");
    swi_mul.add_inputs({out0, out2});
    swi_mul.add_outputs({out3});

    // multiplication
    auto out4 = logical_tensor(id++, dt_inter, hd_sz, layout_type::strided);
    auto mul = op(id++, op::kind::Multiply, "mul");
    mul.add_inputs({out3, out1});
    mul.add_outputs({out4});

    // downconversion when needed
    auto out4_dt = out4;
    auto typecast = op(id++, op::kind::TypeCast, "typecast");
    if (dt != dt_inter) {
        out4_dt = logical_tensor(id++, dt, hd_sz, layout_type::strided);
        typecast.add_inputs({out4});
        typecast.add_outputs({out4_dt});
    }

    // fc_down
    auto wei2 = logical_tensor(id++, dt, wei2_sz, layout_type::strided);
    auto dst = logical_tensor(id++, dt, out_sz, layout_type::strided);
    auto fc_down = op(id++, op::kind::MatMul, "fc_down");
    fc_down.add_inputs({out4_dt, wei2});
    fc_down.add_outputs({dst});

    // Construct a gated mlp graph with engine kind and operations.
    dnnl::graph::graph mlp(ekind);
    mlp.add_op(fc_gate);
    mlp.add_op(fc_up);
    mlp.add_op(swi_sig);
    mlp.add_op(swi_mul);
    mlp.add_op(mul);
    if (dt != dt_inter) mlp.add_op(typecast);
    mlp.add_op(fc_down);
    mlp.finalize();

    // Get partitions from the mlp graph.
    std::vector<partition> partitions = mlp.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported mlp" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp
            = partitions[0].compile({src, wei0, wei1, wei2}, {dst}, eng);

    // Create tensor objects
    auto ts_src = tensor(src, eng);
    auto ts_combined_wei0 = tensor(combined_wei0, eng);
    auto ts_wei2 = tensor(wei2, eng);
    auto ts_dst = tensor(dst, eng);

    // Allocate user data.
    std::vector<float> src_data(product(src_sz));
    std::vector<float> combined_wei0_data(product(combined_wei0_sz));
    std::vector<float> wei2_data(product(wei2_sz));

    fill_random(src_data);
    fill_random(combined_wei0_data);
    fill_random(wei2_data);

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(src_data.data(), ts_src);
    write_to_dnnl_tensor(combined_wei0_data.data(), ts_combined_wei0);
    write_to_dnnl_tensor(wei2_data.data(), ts_wei2);

    // create ts_wei0, ts_wei1 from the data handle of combined_wei0 and offsets.
    char *handle = reinterpret_cast<char *>(ts_combined_wei0.get_data_handle());
    auto ts_wei0 = tensor(wei0, eng, handle);
    auto ts_wei1 = tensor(wei1, eng, handle + p.oc * size_of(dt));

    // Warmup run.
    // Execute the compiled partition of mqa.
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    const int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;
}

void bad_args() {
    std::cerr << "Usage: graph-gated-mlp-wei-combined-cpp [cpu|gpu]\n"
                 "       graph-gated-mlp-wei-combined-cpp [cpu|gpu] <mb> <ic> "
                 "<oc>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, dnnl_data_type_t dt, const mlp_dims_t &p,
        double time_limit = 0.) {
    try {
        bench_gated_mlp(ekind, static_cast<logical_tensor::data_type>(dt), p,
                time_limit);
        get_mem_pool().clear();
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type) {
            std::cout << "unsupported mlp" << std::endl;
        } else
            throw;
    }
}

void mlp_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    mlp_dims_t params = {1, 4096, 14336};

    if (argc > 2) {
        if (argc == 5) {
            params.mb = std::atoi(argv[2]);
            params.ic = std::atoi(argv[3]);
            params.oc = std::atoi(argv[4]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.ic <= 0 || params.oc <= 0) { bad_args(); }
    }

    bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            mlp_perf, parse_engine_kind(argc, argv, 3), argc, argv);
}
