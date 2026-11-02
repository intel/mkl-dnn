.. index:: pair: page; SYCL USM example
.. _doxid-sycl_interop_usm_cpp:

SYCL USM example
================

This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN.

This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN.

The workflow includes following steps:

* Create a GPU or CPU engine. It uses DPC++ as the runtime in this sample.

* Create a memory descriptor/object.

* Create a SYCL kernel for data initialization.

* Access a SYCL USM pointer via SYCL interoperability interface.

* Access a SYCL queue via SYCL interoperability interface.

* Execute a SYCL kernel with related SYCL queue and SYCL USM pointer

* Create operation descriptor/operation primitives descriptor/primitive.

* Execute the primitive with the initialized memory.

* Validate the result.

For a detailed walkthrough refer to the :ref:`Getting started on both CPU and GPU with SYCL extensions API <doxid-sycl_interop_buffer_cpp>` example that utilizes SYCL buffers.

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
	
	class kernel_tag;
	
	void sycl_usm_tutorial(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm(eng);
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` tz_dims = {2, 3, 4, 5};
	    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
	            std::multiplies<size_t>());
	    auto usm_buffer = (float *)malloc_shared(N * sizeof(float),
	            :ref:`sycl_interop::get_device <doxid-namespacednnl_1_1sycl__interop_1adddf805d923929f373fb6233f1fd4a27>`(eng), :ref:`sycl_interop::get_context <doxid-namespacednnl_1_1sycl__interop_1a5227caa35295b41dcdd57f8abaa7551b>`(eng));
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` mem_d(
	            tz_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`);
	
	    :ref:`memory <doxid-structdnnl_1_1memory>` mem = :ref:`sycl_interop::make_memory <doxid-namespacednnl_1_1sycl__interop_1a5f3bf8334f86018201e14fec6a666be4>`(
	            mem_d, eng, :ref:`sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`, usm_buffer);
	
	    queue q = :ref:`sycl_interop::get_queue <doxid-namespacednnl_1_1sycl__interop_1a59a9e92e8ff59c1282270fc6edad4274>`(strm);
	    auto fill_e = q.submit([&](handler &cgh) {
	        cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
	            int idx = (int)i[0];
	            usm_buffer[idx] = (idx % 2) ? -idx : idx;
	        });
	    });
	
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, mem_d, mem_d, 0.0f);
	    auto relu = :ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd);
	
	    auto relu_e = :ref:`sycl_interop::execute <doxid-namespacednnl_1_1sycl__interop_1a30c5c906dfba71774528710613165c14>`(
	            relu, strm, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, mem}}, {fill_e});
	    relu_e.wait();
	
	    for (size_t i = 0; i < N; i++) {
	        float exp_value = (i % 2) ? 0.0f : i;
	        if (usm_buffer[i] != (float)exp_value)
	            throw std::string(
	                    "Unexpected output, found a negative value after the ReLU "
	                    "execution.");
	    }
	
	    free((void *)usm_buffer, :ref:`sycl_interop::get_context <doxid-namespacednnl_1_1sycl__interop_1a5227caa35295b41dcdd57f8abaa7551b>`(eng));
	}
	
	int main(int argc, char **argv) {
	    int exit_code = 0;
	
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    try {
	        sycl_usm_tutorial(engine_kind);
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

