.. index:: pair: example; cpu_sgemm_and_matmul.cpp
.. _doxid-cpu_sgemm_and_matmul_8cpp-example:

cpu_sgemm_and_matmul.cpp
========================

Annotated version: :ref:`MatMul Tutorial: Comparison with SGEMM <doxid-cpu_sgemm_and_matmul_cpp>`

Annotated version: :ref:`MatMul Tutorial: Comparison with SGEMM <doxid-cpu_sgemm_and_matmul_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2025 Intel Corporation
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
	#include <cctype>
	#include <cmath>
	#include <cstdio>
	#include <iostream>
	#include <random>
	#include <stdexcept>
	#include <vector>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	namespace {
	
	void init_vector(std::vector<float> &v) {
	    std::mt19937 gen;
	    std::uniform_real_distribution<float> u(-1, 1);
	
	    for (auto &e : v)
	        e = u(gen);
	}
	
	int compare_vectors(const std::vector<float> &v1, const std::vector<float> &v2,
	        int64_t K, const char *message) {
	    double v1_l2 = 0, diff_l2 = 0;
	    for (size_t n = 0; n < v1.size(); ++n) {
	        float diff = v1[n] - v2[n];
	        v1_l2 += v1[n] * v1[n];
	        diff_l2 += diff * diff;
	    }
	
	    v1_l2 = std::sqrt(v1_l2);
	    diff_l2 = std::sqrt(diff_l2);
	
	    // Finding the reasonable (tight and accurate) threshold is quite difficult
	    // problem.
	    // The implementation testing might also use special data filling to
	    // alleviate issues related to the finite precision arithmetic.
	    // However, in simple cases the machine epsilon multiplied by log(K) should
	    // work reasonably well.
	    const double threshold = std::numeric_limits<float>::epsilon()
	            * std::log(std::max(2., (double)K));
	    bool ok = diff_l2 <= threshold * v1_l2;
	
	    printf("%s\n\tL2 Norms"
	           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n"
	           "\tAccuracy check: %s\n",
	            message, v1_l2, diff_l2, diff_l2 / v1_l2, ok ? "OK" : "FAILED");
	
	    return ok ? 0 : 1;
	}
	
	} // namespace
	
	int number_of_runs = 1;
	float fixed_beta = 0.f;
	
	const :ref:`engine <doxid-structdnnl_1_1engine>` &eng() {
	    static const :ref:`engine <doxid-structdnnl_1_1engine>` eng(:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0);
	    return eng;
	}
	
	// Create a _dynamic_ MatMul primitive that can work with arbitrary shapes
	// and alpha parameters.
	// Warning: current limitation is that beta parameter should be known in
	// advance (use fixed_beta).
	:ref:`matmul <doxid-structdnnl_1_1matmul>` dynamic_matmul_create() {
	    // We assume that beta is known at the primitive creation time
	    float beta = fixed_beta;
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` a_shape = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` b_shape = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` c_shape = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`};
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` a_strides = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` b_strides = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` c_strides = {:ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`, 1};
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` a_md(a_shape, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, a_strides);
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` b_md(b_shape, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, b_strides);
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` c_md(c_shape, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, c_strides);
	
	    // Create attributes (to handle alpha dynamically and beta if necessary)
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, /* mask */ 0);
	    if (beta != 0.f) {
	        :ref:`post_ops <doxid-structdnnl_1_1post__ops>` po;
	        po.:ref:`append_sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`(beta);
	        attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(po);
	    }
	
	    // Create a MatMul primitive
	    :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd(eng(), a_md, b_md, c_md, attr);
	    return :ref:`matmul <doxid-structdnnl_1_1matmul>`(matmul_pd);
	}
	
	// Execute a _dynamic_ MatMul primitive created earlier. All the parameters are
	// passed at a run-time (except for beta which has to be specified at the
	// primitive creation time due to the current limitation).
	void dynamic_matmul_execute(:ref:`matmul <doxid-structdnnl_1_1matmul>` &matmul_p, char transA, char transB,
	        int64_t M, int64_t N, int64_t K, float alpha, const float *A,
	        int64_t lda, const float *B, int64_t ldb, float beta, float *C,
	        int64_t ldc) {
	    using dims = :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`;
	
	    if (beta != fixed_beta)
	        throw std::logic_error("Run-time beta is not yet supported.");
	
	    // Translate transA and transB
	    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} : dims {1, lda};
	    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} : dims {1, ldb};
	
	    // Wrap raw pointers into oneDNN memories (with proper shapes)
	    :ref:`memory <doxid-structdnnl_1_1memory>` A_m({{M, K}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, a_strides}, eng(), (void *)A);
	    :ref:`memory <doxid-structdnnl_1_1memory>` B_m({{K, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, b_strides}, eng(), (void *)B);
	    :ref:`memory <doxid-structdnnl_1_1memory>` C_m({{M, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {ldc, 1}}, eng(), (void *)C);
	
	    // Prepare oneDNN memory for alpha
	    :ref:`memory <doxid-structdnnl_1_1memory>` alpha_m({{1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {1}}, eng(), &alpha);
	
	    // Execute the MatMul primitive
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng());
	    matmul_p.:ref:`execute <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, A_m}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, B_m}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, C_m},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, alpha_m}});
	    s.wait();
	}
	
	void sgemm_and_matmul_with_params(char transA, char transB, int64_t M,
	        int64_t N, int64_t K, float alpha, float beta) {
	    if (beta != fixed_beta)
	        throw std::logic_error("Run-time beta is not yet supported.");
	
	    // Allocate and initialize matrices
	    std::vector<float> A(M * K);
	    init_vector(A);
	
	    std::vector<float> B(K * N);
	    init_vector(B);
	
	    std::vector<float> C_sgemm(M * N);
	    init_vector(C_sgemm);
	
	    std::vector<float> C_dynamic_matmul = C_sgemm;
	
	    // Prepare leading dimensions
	    int64_t lda = tolower(transA) == 'n' ? K : M;
	    int64_t ldb = tolower(transB) == 'n' ? N : K;
	    int64_t ldc = N;
	
	    // 1. Execute sgemm
	    for (int run = 0; run < number_of_runs; ++run)
	        :ref:`dnnl_sgemm <doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8>`(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
	                beta, C_sgemm.data(), ldc);
	
	    // 2.a Create dynamic MatMul
	    auto dynamic_matmul = dynamic_matmul_create();
	    // 2.b Execute
	    for (int run = 0; run < number_of_runs; ++run)
	        dynamic_matmul_execute(dynamic_matmul, transA, transB, M, N, K, alpha,
	                A.data(), lda, B.data(), ldb, beta, C_dynamic_matmul.data(),
	                ldc);
	
	    int rc = 0;
	    rc |= compare_vectors(
	            C_sgemm, C_dynamic_matmul, K, "Compare SGEMM vs dynamic MatMul");
	    if (rc) throw std::logic_error("The resulting matrices diverged too much.");
	}
	
	void sgemm_and_matmul() {
	    sgemm_and_matmul_with_params('N', 'T', 10, 20, 30, 1.1f, fixed_beta);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`}, sgemm_and_matmul);
	}
