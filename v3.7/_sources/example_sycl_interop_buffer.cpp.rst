.. index:: pair: example; sycl_interop_buffer.cpp
.. _doxid-sycl_interop_buffer_8cpp-example:

sycl_interop_buffer.cpp
=======================

Annotated version: :ref:`Getting started on both CPU and GPU with SYCL extensions API <doxid-sycl_interop_buffer_cpp>`

Annotated version: :ref:`Getting started on both CPU and GPU with SYCL extensions API <doxid-sycl_interop_buffer_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2024 Intel Corporation
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
	
	
	// [Prologue]
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	#include "oneapi/dnnl/dnnl_debug.h"
	#include "oneapi/dnnl/dnnl_sycl.hpp"
	
	#if __has_include(<sycl/sycl.hpp>)
	#include <sycl/sycl.hpp>
	#else
	#error "Unsupported compiler"
	#endif
	
	#include <cassert>
	#include <iostream>
	#include <numeric>
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	using namespace :ref:`sycl <doxid-namespacesycl>`;
	// [Prologue]
	
	class kernel_tag;
	
	void sycl_interop_buffer_tutorial(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // [Initialize engine]
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    // [Initialize engine]
	
	    // [Initialize stream]
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm(eng);
	    // [Initialize stream]
	
	    //  [memory alloc]
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` tz_dims = {2, 3, 4, 5};
	    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
	            std::multiplies<size_t>());
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` mem_d(
	            tz_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`);
	
	    :ref:`memory <doxid-structdnnl_1_1memory>` mem = :ref:`sycl_interop::make_memory <doxid-namespacednnl_1_1sycl__interop_1a5f3bf8334f86018201e14fec6a666be4>`(
	            mem_d, eng, :ref:`sycl_interop::memory_kind::buffer <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`);
	    //  [memory alloc]
	
	    // [get sycl buf]
	    auto sycl_buf = sycl_interop::get_buffer<float>(mem);
	    // [get sycl buf]
	
	    // [sycl kernel exec]
	    queue q = :ref:`sycl_interop::get_queue <doxid-namespacednnl_1_1sycl__interop_1a59a9e92e8ff59c1282270fc6edad4274>`(strm);
	    q.submit([&](handler &cgh) {
	        auto a = sycl_buf.get_access<access::mode::write>(cgh);
	        cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
	            int idx = (int)i[0];
	            a[idx] = (idx % 2) ? -idx : idx;
	        });
	    });
	    // [sycl kernel exec]
	
	    //  [relu creation]
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, mem_d, mem_d, 0.0f);
	    auto relu = :ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd);
	    //  [relu creation]
	
	    // [relu exec]
	    relu.execute(strm, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, mem}});
	    strm.wait();
	    // [relu exec]
	
	    // [Check the results]
	    auto host_acc = sycl_buf.get_host_access();
	    for (size_t i = 0; i < N; i++) {
	        float exp_value = (i % 2) ? 0.0f : i;
	        if (host_acc[i] != (float)exp_value)
	            throw std::string(
	                    "Unexpected output, find a negative value after the ReLU "
	                    "execution.");
	    }
	    // [Check the results]
	}
	
	// [Main]
	int main(int argc, char **argv) {
	    int exit_code = 0;
	
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    try {
	        sycl_interop_buffer_tutorial(engine_kind);
	    } catch (:ref:`dnnl::error <doxid-structdnnl_1_1error>` &e) {
	        std::cout << "oneDNN error caught: " << std::endl
	                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
	                  << "\tMessage: " << e.:ref:`what <doxid-structdnnl_1_1error_1afcf188632b6264fba24f3300dabd9b65>`() << std::endl;
	        exit_code = 1;
	    } catch (std::string &e) {
	        std::cout << "Error in the example: " << e << "." << std::endl;
	        exit_code = 2;
	    } catch (exception &e) {
	        std::cout << "Error in the example: " << e.what() << "." << std::endl;
	        exit_code = 3;
	    }
	
	    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
	              << engine_kind2str_upper(engine_kind) << "." << std::endl;
	    finalize();
	    return exit_code;
	}
	// [Main]
