.. index:: pair: page; Getting started on GPU with OpenCL extensions API
.. _doxid-gpu_opencl_interop_cpp:

Getting started on GPU with OpenCL extensions API
=================================================

This C++ API example demonstrates programming for Intel(R) Processor Graphics with OpenCL\* extensions API in oneDNN.

This C++ API example demonstrates programming for Intel(R) Processor Graphics with OpenCL\* extensions API in oneDNN.

Example code: :ref:`gpu_opencl_interop.cpp <doxid-gpu_opencl_interop_8cpp-example>`

The workflow includes following steps:

* Create a GPU engine. It uses OpenCL as the runtime in this sample.

* Create a GPU memory descriptor/object.

* Create an OpenCL kernel for GPU data initialization

* Access a GPU memory via OpenCL interoperability interface

* Access a GPU command queue via OpenCL interoperability interface

* Execute a OpenCL kernel with related GPU command queue and GPU memory

* Create operation descriptor/operation primitives descriptor/primitive .

* Execute the primitive with the initialized GPU memory

* Validate the result by mapping the OpenCL memory via OpenCL interoperability interface



.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_headers:

Public headers
~~~~~~~~~~~~~~

To start using oneDNN, we must first include the ``dnnl.hpp`` header file in the application. We also include CL/cl.h for using OpenCL APIs and ``dnnl_debug.h``, which contains some debugging facilities such as returning a string representation for common oneDNN C types. All C++ API types and functions reside in the ``dnnl`` namespace. For simplicity of the example we import this namespace.

.. ref-code-block:: cpp

	#include <iostream>
	#include <numeric>
	#include <stdexcept>
	
	#include <CL/cl.h>
	
	#include "oneapi/dnnl/dnnl.hpp"
	#include "oneapi/dnnl/dnnl_ocl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	using namespace :ref:`std <doxid-namespacestd>`;





.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_tutorial:

gpu_opencl_interop_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_sub1:

Engine and stream
-----------------

All oneDNN primitives and memory objects are attached to a particular :ref:`dnnl::engine <doxid-structdnnl_1_1engine>`, which is an abstraction of a computational device (see also :ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`). The primitives are created and optimized for the device to which they are attached, and the memory objects refer to memory residing on the corresponding device. In particular, that means neither memory objects nor primitives that were created for one engine can be used on another.

To create engines, we must specify the :ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` and the index of the device of the given kind. In this example we use the first available GPU engine, so the index for the engine is 0. This example assumes OpenCL being a runtime for GPU. In such case, during engine creation, an OpenCL context is also created and attaches to the created engine.

.. ref-code-block:: cpp

	:ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>` eng(engine::kind::gpu, 0);

In addition to an engine, all primitives require a :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` for the execution. The stream encapsulates an execution context and is tied to a particular engine.

In this example, a GPU stream is created. This example assumes OpenCL being a runtime for GPU. During stream creation, an OpenCL command queue is also created and attaches to this stream.

.. ref-code-block:: cpp

	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm(eng);





.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_sub2:

Wrapping data into oneDNN memory object
---------------------------------------

Next, we create a memory object. We need to specify dimensions of our memory by passing a memory::dims object. Then we create a memory descriptor with these dimensions, with the :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>` data type, and with the :ref:`dnnl::memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>` memory format. Finally, we construct a memory object and pass the memory descriptor. The library allocates memory internally.

.. ref-code-block:: cpp

	memory::dims tz_dims = {2, 3, 4, 5};
	const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
	        std::multiplies<size_t>());

	memory::desc mem_d(
	        tz_dims, memory::data_type::f32, memory::format_tag::nchw);

	memory mem(mem_d, eng);





.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_sub3:

Initialize the data by executing a custom OpenCL kernel
-------------------------------------------------------

We are going to create an OpenCL kernel that will initialize our data. It requires writing a bit of C code to create an OpenCL program from a string literal source. The kernel initializes the data by the 0, -1, 2, -3, ... sequence: ``data[i] = (-1)^i * i``.

.. ref-code-block:: cpp

	const char *ocl_code
	        = "__kernel void init(__global float *data) {"
	          "    int id = get_global_id(0);"
	          "    data[id] = (id % 2) ? -id : id;"
	          "}";













Create/Build Opencl kernel by ``create_init_opencl_kernel()`` function. Refer to the full code example for the ``create_init_opencl_kernel()`` function.

.. ref-code-block:: cpp

	const char *kernel_name = "init";
	cl_kernel ocl_init_kernel = create_init_opencl_kernel(
	        ocl_interop::get_context(eng), kernel_name, ocl_code);











The next step is to execute our OpenCL kernel by setting its arguments and enqueueing to an OpenCL queue. You can extract the underlying OpenCL buffer from the memory object using the interoperability interface: dnnl::memory::get_ocl_mem_object() . For simplicity we can just construct a stream, extract the underlying OpenCL queue, and enqueue the kernel to this queue.

.. ref-code-block:: cpp

	cl_mem ocl_buf = ocl_interop::get_mem_object(mem);
	OCL_CHECK(clSetKernelArg(ocl_init_kernel, 0, sizeof(ocl_buf), &ocl_buf));

	cl_command_queue ocl_queue = ocl_interop::get_command_queue(strm);
	OCL_CHECK(clEnqueueNDRangeKernel(ocl_queue, ocl_init_kernel, 1, nullptr, &N,
	        nullptr, 0, nullptr, nullptr));





.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_sub4:

Create and execute a primitive
------------------------------

There are three steps to create an operation primitive in oneDNN:

#. Create an operation descriptor.

#. Create a primitive descriptor.

#. Create a primitive.

Let's create the primitive to perform the ReLU (rectified linear unit) operation: x = max(0, x). An operation descriptor has no dependency on a specific engine - it just describes some operation. On the contrary, primitive descriptors are attached to a specific engine and represent some implementation for this engine. A primitive object is a realization of a primitive descriptor, and its construction is usually much "heavier".

.. ref-code-block:: cpp

	auto relu_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward,
	        algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
	auto relu = eltwise_forward(relu_pd);







Next, execute the primitive.

.. ref-code-block:: cpp

	relu.execute(strm, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, mem}});
	strm.wait();

.. note:: 

   Our primitive mem serves as both input and output parameter.
   
   

.. note:: 

   Primitive submission on GPU is asynchronous; However, the user can call dnnl:stream::wait() to synchronize the stream and ensure that all previously submitted primitives are completed.





.. _doxid-gpu_opencl_interop_cpp_1gpu_opencl_interop_cpp_sub5:

Validate the results
--------------------

Before running validation codes, we need to copy the OpenCL memory to the host. This can be done using OpenCL API. For convenience, we use a utility function read_from_dnnl_memory() implementing required OpenCL API calls. After we read the data to the host, we can run validation codes on the host accordingly.

.. ref-code-block:: cpp

	std::vector<float> mem_data(N);
	read_from_dnnl_memory(mem_data.data(), mem);
	for (size_t i = 0; i < N; i++) {
	    float expected = (i % 2) ? 0.0f : (float)i;
	    if (mem_data[i] != expected) {
	        std::cout << "Expect " << expected << " but got " << mem_data[i]
	                  << "." << std::endl;
	        throw std::logic_error("Accuracy check failed.");
	    }
	}

Upon compiling and running the example, the output should be just:

.. ref-code-block:: cpp

	Example passed.

