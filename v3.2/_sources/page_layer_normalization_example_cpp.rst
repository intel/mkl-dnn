.. index:: pair: page; Layer Normalization Primitive Example
.. _doxid-layer_normalization_example_cpp:

Layer Normalization Primitive Example
=====================================

This C++ API example demonstrates how to create and execute a :ref:`Layer normalization <doxid-dev_guide_layer_normalization>` primitive in forward propagation mode.

Key optimizations included in this example:

* In-place primitive execution;

* Creation of memory objects using the primitive descriptor.

.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2020-2022 Intel Corporation
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
	#include <cmath>
	#include <iostream>
	#include <string>
	#include <vector>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void layer_normalization_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` T = 12, // time steps
	            N = 3, // batch
	            C = 227; // channels
	
	    // Source (src) and destination (dst) tensors dimensions.
	    const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {T, N, C};
	
	    // Scale/shift tensor dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` scaleshift_dims = {C};
	
	    // Allocate buffer.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> scale_data(product(scaleshift_dims));
	    std::vector<float> shift_data(product(scaleshift_dims));
	
	    // Initialize src tensor.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Initialize scale.
	    std::generate(scale_data.begin(), scale_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	
	    // Initialize shift.
	    std::generate(shift_data.begin(), shift_data.end(), []() {
	        static int i = 0;
	        return std::tan(float(i++));
	    });
	
	    // Create src memory descriptor and memory objects.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::f32, tag::tnc);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::f32, tag::tnc);
	    auto scaleshift_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(scaleshift_dims, dt::f32, tag::x);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto scale_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(scaleshift_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto shift_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(scaleshift_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	    write_to_dnnl_memory(scale_data.data(), scale_mem);
	    write_to_dnnl_memory(shift_data.data(), shift_mem);
	
	    // Create primitive descriptor.
	    const float epsilon = 1.e-10f;
	    auto lnorm_pd = :ref:`layer_normalization_forward::primitive_desc <doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, src_md, dst_md, epsilon,
	            :ref:`normalization_flags::use_scale <doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beab989b02160429ba2696a658ec7a0f8e1>` | :ref:`normalization_flags::use_shift <doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beac5d8386f67a826c8ea1c1ae59a39586f>`);
	
	    // Use the memory descriptors from the primitive to create memory objects
	    // required for the primitive: mean, variance, scale/shift.
	    auto mean_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lnorm_pd.mean_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto variance_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lnorm_pd.variance_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto lnorm_prim = :ref:`layer_normalization_forward <doxid-structdnnl_1_1layer__normalization__forward>`(lnorm_pd);
	
	    // Primitive arguments. Set up in-place execution by assigning src as DST.
	    std::unordered_map<int, memory> lnorm_args;
	    lnorm_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem});
	    lnorm_args.insert({:ref:`DNNL_ARG_MEAN <doxid-group__dnnl__api__primitives__common_1ga9bcff7f442a5d6a0ac1183533e721066>`, mean_mem});
	    lnorm_args.insert({:ref:`DNNL_ARG_VARIANCE <doxid-group__dnnl__api__primitives__common_1gaa0e60e8d129936ba29555e17efb82581>`, variance_mem});
	    lnorm_args.insert({:ref:`DNNL_ARG_SCALE <doxid-group__dnnl__api__primitives__common_1ga3c5cac668bc82c90c8da051c7d430370>`, scale_mem});
	    lnorm_args.insert({:ref:`DNNL_ARG_SHIFT <doxid-group__dnnl__api__primitives__common_1gac250777ced72098caf39deae1d9039c8>`, shift_mem});
	    lnorm_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_mem});
	
	    // Primitive execution: layer normalization.
	    lnorm_prim.execute(engine_stream, lnorm_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.s
	    read_from_dnnl_memory(src_data.data(), src_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            layer_normalization_example, parse_engine_kind(argc, argv));
	}

