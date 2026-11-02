.. index:: pair: example; cross_engine_reorder.cpp
.. _doxid-cross_engine_reorder_8cpp-example:

cross_engine_reorder.cpp
========================

This C++ API example demonstrates programming flow when reordering memory between CPU and GPU engines. Annotated version: :ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_cpp>`

This C++ API example demonstrates programming flow when reordering memory between CPU and GPU engines. Annotated version: :ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2022 Intel Corporation
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
	
	
	
	#include <iostream>
	#include <stdexcept>
	#include <vector>
	
	// [Prologue]
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using namespace :ref:`std <doxid-namespacestd>`;
	// [Prologue]
	
	void fill(:ref:`memory <doxid-structdnnl_1_1memory>` &mem, const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` &adims) {
	    std::vector<float> array(product(adims));
	    for (size_t e = 0; e < array.size(); ++e) {
	        array[e] = e % 7 ? 1.0f : -1.0f;
	    }
	    write_to_dnnl_memory(array.data(), mem);
	}
	
	int find_negative(:ref:`memory <doxid-structdnnl_1_1memory>` &mem, const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` &adims) {
	    int negs = 0;
	    size_t nelems = product(adims);
	    std::vector<float> array(nelems);
	    read_from_dnnl_memory(array.data(), mem);
	
	    for (size_t e = 0; e < nelems; ++e)
	        negs += array[e] < 0.0f;
	    return negs;
	}
	
	void cross_engine_reorder_tutorial() {
	    // [Initialize engine]
	    auto cpu_engine = :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(validate_engine_kind(:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`), 0);
	    auto gpu_engine = :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(validate_engine_kind(:ref:`engine::kind::gpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947>`), 0);
	    // [Initialize engine]
	
	    // [Initialize stream]
	    auto stream_gpu = :ref:`stream <doxid-structdnnl_1_1stream>`(gpu_engine, :ref:`stream::flags::in_order <doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59af51b25ca6f591d130cd0b575bf7821b3>`);
	    // [Initialize stream]
	
	    //  [reorder cpu2gpu]
	    const auto tz = :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` {2, 16, 1, 1};
	    auto m_cpu
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{tz}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	                    cpu_engine);
	    auto m_gpu
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{tz}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	                    gpu_engine);
	    fill(m_cpu, tz);
	    auto r1 = :ref:`reorder <doxid-structdnnl_1_1reorder>`(m_cpu, m_gpu);
	    //  [reorder cpu2gpu]
	
	    // [Create a ReLU primitive]
	    // ReLU primitive descriptor, which corresponds to a particular
	    // implementation in the library. Specify engine type for the ReLU
	    // primitive. Use a GPU engine here.
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(gpu_engine,
	            :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`, :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, m_gpu.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`(),
	            m_gpu.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`(), 0.0f);
	    // ReLU primitive
	    auto relu = :ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd);
	    // [Create a ReLU primitive]
	
	    //  [reorder gpu2cpu]
	    auto r2 = :ref:`reorder <doxid-structdnnl_1_1reorder>`(m_gpu, m_cpu);
	    //  [reorder gpu2cpu]
	
	    // [Execute primitives]
	    // wrap source data from CPU to GPU
	    r1.execute(stream_gpu, m_cpu, m_gpu);
	    // Execute ReLU on a GPU stream
	    relu.execute(stream_gpu, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, m_gpu}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, m_gpu}});
	    // Get result data from GPU to CPU
	    r2.execute(stream_gpu, m_gpu, m_cpu);
	
	    stream_gpu.wait();
	    // [Execute primitives]
	
	    // [Check the results]
	    if (find_negative(m_cpu, tz) != 0)
	        throw std::logic_error(
	                "Unexpected output, find a negative value after the ReLU "
	                "execution.");
	    // [Check the results]
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, :ref:`engine::kind::gpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947>`},
	            cross_engine_reorder_tutorial);
	}
