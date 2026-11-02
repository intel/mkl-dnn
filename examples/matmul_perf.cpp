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

/// @example matmul_perf.cpp
/// > Annotated version: @ref matmul_perf_cpp

/// @page matmul_perf_cpp_brief
/// @brief This C++ example runs a simple matrix multiplication (matmul) performance
/// test using oneDNN.

/// @page matmul_perf_cpp Matrix Multiplication Performance Example
/// \copybrief matmul_perf_cpp_brief
///
/// The workflow includes following steps:
///   - Set up and execute a matmul operation with the specified engine kind
///     and matrix dimensions, using f32, f16, bf16 and s8 data types.
///   - Measure the execution time and prints the achieved performance
///     in GFlop/s or GOp/s, depending on the data type.
///
/// To execute the example, compile it with oneDNN and run the following way:
/// ~~~sh
/// ./matmul_perf <engine_kind> <m> [<n> <k>]
/// ~~~
/// Input parameters:
///   - `<engine_kind>`: The kind of oneDNN engine to use (e.g., CPU, GPU).
///   - `<m>`: (Required) The number of rows in the first matrix.
///   - `<n>`: (Optional) The number of columns in the second matrix.
///            If not specified, `n = m`.
///   - `<k>`: (Optional) The number of columns in the first matrix.
///            If not specified, `k = m`.
///
/// @include matmul_perf.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

struct gemm_dims_t {
    memory::dim m, n, k;
};

static const int min_runs = 4;

const char *get_type_string(memory::data_type type) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (type == memory::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(f64);
    TYPE_CASE(bf16);
    TYPE_CASE(s8);
    TYPE_CASE(u8);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(memory::data_type type, gemm_dims_t dims) {
    std::cout << '[' << std::setw(4) << get_type_string(type);
    if (dims.m == dims.n && dims.m == dims.k)
        std::cout << " m = n = k = " << dims.m;
    else
        std::cout << " m = " << dims.m << ", n = " << dims.n
                  << ", k = " << dims.k;
    std::cout << "] " << std::flush;
}

void fill_random(std::vector<float> &out, bool is_integer) {
    static std::vector<float> random_data_i, random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_i.empty() || random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_int_distribution<int> dist_i(-16, 15);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_i.resize(nrand);
        for (auto &d : random_data_i)
            d = static_cast<float>(dist_i(generator));

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    auto &rd = is_integer ? random_data_i : random_data_f;

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], rd.data(), chunk * sizeof(float));
    }
}

double run_case(engine::kind engine_kind, memory::data_type type,
        gemm_dims_t dims, double time_limit = 0.) {
    bool is_integer
            = (type == memory::data_type::s8 || type == memory::data_type::u8);
    bool quick_test = (time_limit == 0.);

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Source (A), weights (B), and destination (C) matrix dimensions.
    memory::dims a_dims = {dims.m, dims.k};
    memory::dims b_dims = {dims.k, dims.n};
    memory::dims c_dims = {dims.m, dims.n};

    // Allocate buffers and random-initialize A/B
    std::vector<float> a_data(product(a_dims));
    std::vector<float> b_data(product(b_dims));
    std::vector<float> c_data(product(c_dims));

    fill_random(a_data, is_integer);
    fill_random(b_data, is_integer);

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto a_md = memory::desc(a_dims, type, memory::format_tag::any);
    auto b_md = memory::desc(b_dims, type, memory::format_tag::any);
    auto c_md = memory::desc(c_dims, type, memory::format_tag::any);

    auto a_in_md = memory::desc(
            a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto b_in_md = memory::desc(
            b_dims, memory::data_type::f32, memory::format_tag::ab);

    auto a_in_mem = memory(a_in_md, engine);
    auto b_in_mem = memory(b_in_md, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(a_data.data(), a_in_mem);
    write_to_dnnl_memory(b_data.data(), b_in_mem);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);

    // Repack and convert input data.
    auto a_mem = memory(matmul_pd.src_desc(), engine);
    reorder(a_in_mem, a_mem).execute(engine_stream, a_in_mem, a_mem);

    auto b_mem = memory(matmul_pd.weights_desc(), engine);
    reorder(b_in_mem, b_mem).execute(engine_stream, b_in_mem, b_mem);

    auto c_mem = memory(matmul_pd.dst_desc(), engine);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Start output.
    if (!quick_test) print_test_case(type, dims);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});

    // Warmup executions.
    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    auto start_first = std::chrono::steady_clock::now();
    matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();
    auto end_first = std::chrono::steady_clock::now();

    std::chrono::duration<double> dur_first = end_first - start_first;

    if (quick_test) return dur_first.count();

    int runs = std::max(min_runs, int(time_limit / dur_first.count()));

    // Timing runs.
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i <= runs; i++)
        matmul_prim.execute(engine_stream, matmul_args);
    engine_stream.wait();

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = end - start;

    // Display the result.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    double total_ops = double(dims.m) * double(dims.n) * double(dims.k) * 2;
    double perf = (total_ops / avg_time) * 1e-9;

    auto scale_string = "G";
    auto unit_string = is_integer ? "Op/s" : "Flop/s";

    if (perf >= 1000) {
        perf /= 1000;
        scale_string = "T";
    }

    std::cout << perf << ' ' << scale_string << unit_string << std::endl;

    return avg_time;
}

void run(engine::kind engine_kind, memory::data_type type, gemm_dims_t dims,
        double time_limit) {
    try {
        if (dims.m * dims.n != 0) {
            // Dimensions manually specified by user.
            run_case(engine_kind, type, dims, time_limit);
        } else {
            // Automatically choose dimensions to fit time limit.
            int mnk = 128;
            const int max_mnk = 8192;

            while (mnk < max_mnk) {
                dims.m = dims.n = dims.k = mnk;
                double time1 = run_case(engine_kind, type, dims);
                double nruns_est = std::max(1., time_limit / time1);
                double mnk_expand = std::exp2(
                        std::round(std::log2(nruns_est / min_runs) / 3.));
                if (mnk_expand <= 1) break;
                mnk = static_cast<int>(
                        std::min<double>(max_mnk, mnk * mnk_expand));
            }

            dims.m = dims.n = dims.k = mnk;
            run_case(engine_kind, type, dims, time_limit);
        }
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type) {
            print_test_case(type, dims);
            std::cout << "unsupported" << std::endl;
        } else
            throw;
    }
}

void bad_args() {
    std::cerr << "Usage: matmul-perf-cpp [cpu|gpu]\n"
                 "       matmul-perf-cpp [cpu|gpu] <size>\n"
                 "       matmul-perf-cpp [cpu|gpu] <m> <n> <k>\n"
                 "If a single <size> is specified, it is used for all three "
                 "dimensions (m/n/k).\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void matmul_perf(engine::kind engine_kind, int argc, char **argv) {
    gemm_dims_t dims = {0, 0, 0};

    if (argc > 2) {
        if (argc == 3)
            dims.m = dims.n = dims.k = std::atoi(argv[2]);
        else if (argc == 5) {
            dims.m = std::atoi(argv[2]);
            dims.n = std::atoi(argv[3]);
            dims.k = std::atoi(argv[4]);
        } else
            bad_args();

        if (dims.m <= 0 || dims.n <= 0 || dims.k <= 0) bad_args();
    }

    run(engine_kind, memory::data_type::f32, dims, 2.0);
    run(engine_kind, memory::data_type::f16, dims, 2.0);
    run(engine_kind, memory::data_type::bf16, dims, 2.0);
    run(engine_kind, memory::data_type::s8, dims, 2.0);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            matmul_perf, parse_engine_kind(argc, argv, 3), argc, argv);
}
