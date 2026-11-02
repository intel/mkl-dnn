.. index:: pair: example; matmul_perf.cpp
.. _doxid-matmul_perf_8cpp-example:

matmul_perf.cpp
===============

This C++ example runs a simple matrix multiplication (matmul) performance test using oneDNN. Annotated version: :ref:`Matrix Multiplication Performance Example <doxid-matmul_perf_cpp>`

This C++ example runs a simple matrix multiplication (matmul) performance test using oneDNN. Annotated version: :ref:`Matrix Multiplication Performance Example <doxid-matmul_perf_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2022-2025 Intel Corporation
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
	#include <chrono>
	#include <cmath>
	#include <iomanip>
	#include <iostream>
	#include <random>
	#include <string>
	#include <vector>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	struct gemm_dims_t {
	    :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` m, n, k;
	};
	
	static const int min_runs = 4;
	
	const char *get_type_string(:ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type) {
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
	
	void print_test_case(:ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type, gemm_dims_t dims) {
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
	
	double run_case(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type,
	        gemm_dims_t dims, double time_limit = 0.) {
	    bool is_integer
	            = (type == :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>` || type == :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`);
	    bool quick_test = (time_limit == 0.);
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Source (A), weights (B), and destination (C) matrix dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` a_dims = {dims.m, dims.k};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` b_dims = {dims.k, dims.n};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` c_dims = {dims.m, dims.n};
	
	    // Allocate buffers and random-initialize A/B
	    std::vector<float> a_data(product(a_dims));
	    std::vector<float> b_data(product(b_dims));
	    std::vector<float> c_data(product(c_dims));
	
	    fill_random(a_data, is_integer);
	    fill_random(b_data, is_integer);
	
	    // Create memory descriptors and memory objects for src, weights, bias, and
	    // dst.
	    auto a_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(a_dims, type, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto b_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(b_dims, type, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto c_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(c_dims, type, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    auto a_in_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            a_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto b_in_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            b_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	
	    auto a_in_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(a_in_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto b_in_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(b_in_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handles.
	    write_to_dnnl_memory(a_data.data(), a_in_mem);
	    write_to_dnnl_memory(b_data.data(), b_in_mem);
	
	    // Create primitive descriptor.
	    auto matmul_pd = :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, a_md, b_md, c_md);
	
	    // Repack and convert input data.
	    auto a_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(matmul_pd.src_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(a_in_mem, a_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, a_in_mem, a_mem);
	
	    auto b_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(matmul_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(b_in_mem, b_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, b_in_mem, b_mem);
	
	    auto c_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(matmul_pd.dst_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto matmul_prim = :ref:`matmul <doxid-structdnnl_1_1matmul>`(matmul_pd);
	
	    // Start output.
	    if (!quick_test) print_test_case(type, dims);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> matmul_args;
	    matmul_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, a_mem});
	    matmul_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, b_mem});
	    matmul_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, c_mem});
	
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
	
	void run(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type, gemm_dims_t dims,
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
	    } catch (:ref:`dnnl::error <doxid-structdnnl_1_1error>` &e) {
	        // Catch and report unimplemented cases.
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`) {
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
	
	void matmul_perf(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, int argc, char **argv) {
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
	
	    run(engine_kind, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, dims, 2.0);
	    run(engine_kind, :ref:`memory::data_type::f16 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa2449b6477c1fef79be4202906486876>`, dims, 2.0);
	    run(engine_kind, :ref:`memory::data_type::bf16 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceafe2904d9fb3b0f4a81c92b03dec11424>`, dims, 2.0);
	    run(engine_kind, :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, dims, 2.0);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            matmul_perf, parse_engine_kind(argc, argv, 3), argc, argv);
	}
