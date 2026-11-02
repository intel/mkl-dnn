.. index:: pair: example; getting_started.cpp
.. _doxid-getting_started_8cpp-example:

getting_started.cpp
===================

This C++ API example demonstrates the basics of the oneDNN programming model. Annotated version: :ref:`Getting started <doxid-getting_started_cpp>`

This C++ API example demonstrates the basics of the oneDNN programming model. Annotated version: :ref:`Getting started <doxid-getting_started_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2023 Intel Corporation
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
	
	
	#include <cmath>
	#include <numeric>
	#include <stdexcept>
	#include <vector>
	
	#include "oneapi/dnnl/dnnl.hpp"
	#include "oneapi/dnnl/dnnl_debug.h"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	// [Prologue]
	
	
	// [Prologue]
	
	void getting_started_tutorial(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // [Initialize engine]
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    // [Initialize engine]
	
	    // [Initialize stream]
	    :ref:`stream <doxid-structdnnl_1_1stream>` engine_stream(eng);
	    // [Initialize stream]
	
	
	    // [Create user's data]
	    const int N = 1, H = 13, W = 13, C = 3;
	
	    // Compute physical strides for each dimension
	    const int stride_N = H * W * C;
	    const int stride_H = W * C;
	    const int stride_W = C;
	    const int stride_C = 1;
	
	    // An auxiliary function that maps logical index to the physical offset
	    auto offset = [=](int n, int h, int w, int c) {
	        return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
	    };
	
	    // The image size
	    const int image_size = N * H * W * C;
	
	    // Allocate a buffer for the image
	    std::vector<float> image(image_size);
	
	    // Initialize the image with some values
	    for (int n = 0; n < N; ++n)
	        for (int h = 0; h < H; ++h)
	            for (int w = 0; w < W; ++w)
	                for (int c = 0; c < C; ++c) {
	                    int off = offset(
	                            n, h, w, c); // Get the physical offset of a pixel
	                    image[off] = -std::cos(off / 10.f);
	                }
	    // [Create user's data]
	
	    // [Init src_md]
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {N, C, H, W}, // logical dims, the order is defined by a primitive
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, // tensor's data type
	            :ref:`memory::format_tag::nhwc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa763cbf7ba1b7b8793dcdc6e2157b5c42>` // memory format, NHWC in this case
	    );
	    // [Init src_md]
	
	
	    // [Init alt_src_md]
	    auto alt_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {N, C, H, W}, // logical dims, the order is defined by a primitive
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, // tensor's data type
	            {stride_N, stride_C, stride_H, stride_W} // the strides
	    );
	
	    // Sanity check: the memory descriptors should be the same
	    if (src_md != alt_src_md)
	        throw std::logic_error("Memory descriptor initialization mismatch.");
	    // [Init alt_src_md]
	
	
	    // [Create memory objects]
	    // src_mem contains a copy of image after write_to_dnnl_memory function
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, eng);
	    write_to_dnnl_memory(image.data(), src_mem);
	
	    // For dst_mem the library allocates buffer
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, eng);
	    // [Create memory objects]
	
	    // [Create a ReLU primitive]
	    // ReLU primitive descriptor, which corresponds to a particular
	    // implementation in the library
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(
	            eng, // an engine the primitive will be created for
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	            src_md, // source memory descriptor for an operation to work on
	            src_md, // destination memory descriptor for an operation to work on
	            0.f, // alpha parameter means negative slope in case of ReLU
	            0.f // beta parameter is ignored in case of ReLU
	    );
	
	    // ReLU primitive
	    auto relu = :ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd); // !!! this can take quite some time
	    // [Create a ReLU primitive]
	
	
	    // [Execute ReLU primitive]
	    // Execute ReLU (out-of-place)
	    relu.execute(engine_stream, // The execution stream
	            {
	                    // A map with all inputs and outputs
	                    {:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem}, // Source tag and memory obj
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem}, // Destination tag and memory obj
	            });
	
	    // Wait the stream to complete the execution
	    engine_stream.wait();
	    // [Execute ReLU primitive]
	
	    // [Execute ReLU primitive in-place]
	    // Execute ReLU (in-place)
	    // relu.execute(engine_stream,  {
	    //          {DNNL_ARG_SRC, src_mem},
	    //          {DNNL_ARG_DST, src_mem},
	    //         });
	    // [Execute ReLU primitive in-place]
	
	    // [Check the results]
	    // Obtain a buffer for the `dst_mem` and cast it to `float *`.
	    // This is safe since we created `dst_mem` as f32 tensor with known
	    // memory format.
	    std::vector<float> relu_image(image_size);
	    read_from_dnnl_memory(relu_image.data(), dst_mem);
	    /*
	    // Check the results
	    for (int n = 0; n < N; ++n)
	        for (int h = 0; h < H; ++h)
	            for (int w = 0; w < W; ++w)
	                for (int c = 0; c < C; ++c) {
	                    int off = offset(
	                            n, h, w, c); // get the physical offset of a pixel
	                    float expected = image[off] < 0
	                            ? 0.f
	                            : image[off]; // expected value
	                    if (relu_image[off] != expected) {
	                        std::cout << "At index(" << n << ", " << c << ", " << h
	                                  << ", " << w << ") expect " << expected
	                                  << " but got " << relu_image[off]
	                                  << std::endl;
	                        throw std::logic_error("Accuracy check failed.");
	                    }
	                }
	    // [Check the results]
	    */
	}
	
	// [Main]
	int main(int argc, char **argv) {
	    int exit_code = 0;
	
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    try {
	        getting_started_tutorial(engine_kind);
	    } catch (:ref:`dnnl::error <doxid-structdnnl_1_1error>` &e) {
	        std::cout << "oneDNN error caught: " << std::endl
	                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
	                  << "\tMessage: " << e.:ref:`what <doxid-structdnnl_1_1error_1afcf188632b6264fba24f3300dabd9b65>`() << std::endl;
	        exit_code = 1;
	    } catch (std::string &e) {
	        std::cout << "Error in the example: " << e << "." << std::endl;
	        exit_code = 2;
	    } catch (std::exception &e) {
	        std::cout << "Error in the example: " << e.what() << "." << std::endl;
	        exit_code = 3;
	    }
	
	    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
	              << engine_kind2str_upper(engine_kind) << "." << std::endl;
	    finalize();
	    return exit_code;
	}
	// [Main]
