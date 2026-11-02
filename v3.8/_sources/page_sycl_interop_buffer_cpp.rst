.. index:: pair: page; Getting started on both CPU and GPU with SYCL extensions API
.. _doxid-sycl_interop_buffer_cpp:

Getting started on both CPU and GPU with SYCL extensions API
============================================================

Full example text: :ref:`sycl_interop_buffer.cpp <doxid-sycl_interop_buffer_8cpp-example>`.

Full example text: :ref:`sycl_interop_buffer.cpp <doxid-sycl_interop_buffer_8cpp-example>`

This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN. The workflow includes following steps:

* Create a GPU or CPU engine. It uses DPC++ as the runtime in this sample.

* Create a memory descriptor/object.

* Create a SYCL kernel for data initialization.

* Access a SYCL buffer via SYCL interoperability interface.

* Access a SYCL queue via SYCL interoperability interface.

* Execute a SYCL kernel with related SYCL queue and SYCL buffer

* Create operation descriptor/operation primitives descriptor/primitive.

* Execute the primitive with the initialized memory.

* Validate the result through a host accessor.



.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_headers:

Public headers
~~~~~~~~~~~~~~

To start using oneDNN, we must first include the ``dnnl.hpp`` header file in the application. We also include sycl/sycl.hpp from DPC++ for using SYCL APIs and ``dnnl_debug.h``, which contains some debugging facilities such as returning a string representation for common oneDNN C types. All C++ API types and functions reside in the ``dnnl`` namespace, and SYCL API types and functions reside in the ``cl::sycl`` namespace. For simplicity of the example we import both namespaces.

.. ref-code-block:: cpp

	
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





.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_tutorial:

sycl_interop_buffer_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_sub1:

Engine and stream
-----------------

All oneDNN primitives and memory objects are attached to a particular :ref:`dnnl::engine <doxid-structdnnl_1_1engine>`, which is an abstraction of a computational device (see also :ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`). The primitives are created and optimized for the device to which they are attached, and the memory objects refer to memory residing on the corresponding device. In particular, that means neither memory objects nor primitives that were created for one engine can be used on another.

To create engines, we must specify the :ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` and the index of the device of the given kind. In this example we use the first available GPU or CPU engine, so the index for the engine is 0. This example assumes DPC++ being a runtime. In such case, during engine creation, an SYCL context is also created and attaches to the created engine.

.. ref-code-block:: cpp

	:ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>` eng(engine_kind, 0);

In addition to an engine, all primitives require a :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` for the execution. The stream encapsulates an execution context and is tied to a particular engine.

In this example, a stream is created. This example assumes DPC++ being a runtime. During stream creation, a SYCL queue is also created and attaches to this stream.

.. ref-code-block:: cpp

	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm(eng);





.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_sub2:

Wrapping data into oneDNN memory object
---------------------------------------

Next, we create a memory object. We need to specify dimensions of our memory by passing a memory::dims object. Then we create a memory descriptor with these dimensions, with the :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>` data type, and with the :ref:`dnnl::memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>` memory format. Finally, we construct a memory object and pass the memory descriptor. The library allocates memory internally.

.. ref-code-block:: cpp

	memory::dims tz_dims = {2, 3, 4, 5};
	const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
	        std::multiplies<size_t>());

	memory::desc mem_d(
	        tz_dims, memory::data_type::f32, memory::format_tag::nchw);

	memory mem = sycl_interop::make_memory(
	        mem_d, eng, sycl_interop::memory_kind::buffer);





.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_sub3:

Initialize the data executing a custom SYCL kernel
--------------------------------------------------

The underlying SYCL buffer can be extracted from the memory object using the interoperability interface: ``dnnl::sycl_interop_buffer::get_buffer<T>(const :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`)``.

.. ref-code-block:: cpp

	auto sycl_buf = sycl_interop::get_buffer<float>(mem);











We are going to create an SYCL kernel that should initialize our data. To execute SYCL kernel we need a SYCL queue. For simplicity we can construct a stream and extract the SYCL queue from it. The kernel initializes the data by the ``0, -1, 2, -3, ...`` sequence: ``data[i] = (-1)^i * i``.

.. ref-code-block:: cpp

	queue q = sycl_interop::get_queue(strm);
	q.submit([&](handler &cgh) {
	    auto a = sycl_buf.get_access<access::mode::write>(cgh);
	    cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
	        int idx = (int)i[0];
	        a[idx] = (idx % 2) ? -idx : idx;
	    });
	});





.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_sub4:

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
	strm.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();

.. note:: 

   With DPC++ runtime, both CPU and GPU have asynchronous execution; However, the user can call :ref:`dnnl::stream::wait() <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>` to synchronize the stream and ensure that all previously submitted primitives are completed.





.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_sub5:

Validate the results
--------------------

Before running validation codes, we need to access the SYCL memory on the host. The simplest way to access the SYCL-backed memory on the host is to construct a host accessor. Then we can directly read and write this data on the host. However no any conflicting operations are allowed until the host accessor is destroyed. We can run validation codes on the host accordingly.

.. ref-code-block:: cpp

	auto host_acc = sycl_buf.get_host_access();
	for (size_t i = 0; i < N; i++) {
	    float exp_value = (i % 2) ? 0.0f : i;
	    if (host_acc[i] != (float)exp_value)
	        throw std::string(
	                "Unexpected output, find a negative value after the ReLU "
	                "execution.");
	}







.. _doxid-sycl_interop_buffer_cpp_1sycl_interop_buffer_cpp_main:

main() function
~~~~~~~~~~~~~~~

We now just call everything we prepared earlier.

Because we are using the oneDNN C++ API, we use exceptions to handle errors (see :ref:`API <doxid-dev_guide_c_and_cpp_apis>`). The oneDNN C++ API throws exceptions of type :ref:`dnnl::error <doxid-structdnnl_1_1error>`, which contains the error status (of type :ref:`dnnl_status_t <doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>`) and a human-readable error message accessible through the regular ``what()`` method.

.. ref-code-block:: cpp

	int main(int argc, char **argv) {
	    int exit_code = 0;
	
	    engine::kind engine_kind = parse_engine_kind(argc, argv);
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

Upon compiling and running the example, the output should be just:

.. ref-code-block:: cpp

	Example passed.

