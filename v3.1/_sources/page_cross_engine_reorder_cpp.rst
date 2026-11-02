.. index:: pair: page; Reorder between CPU and GPU engines
.. _doxid-cross_engine_reorder_cpp:

Reorder between CPU and GPU engines
===================================

This C++ API example demonstrates programming flow when reordering memory between CPU and GPU engines.

This C++ API example demonstrates programming flow when reordering memory between CPU and GPU engines.

Example code: :ref:`cross_engine_reorder.cpp <doxid-cross_engine_reorder_8cpp-example>`



.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_headers:

Public headers
~~~~~~~~~~~~~~

To start using oneDNN, we must first include the ``dnnl.hpp`` header file in the application. We also include ``dnnl_debug.h``, which contains some debugging facilities such as returning a string representation for common oneDNN C types.

All C++ API types and functions reside in the ``dnnl`` namespace. For simplicity of the example we import this namespace.





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_tutorial:

cross_engine_reorder_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub1:

Engine and stream
-----------------

All oneDNN primitives and memory objects are attached to a particular :ref:`dnnl::engine <doxid-structdnnl_1_1engine>`, which is an abstraction of a computational device (see also :ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`). The primitives are created and optimized for the device they are attached to, and the memory objects refer to memory residing on the corresponding device. In particular, that means neither memory objects nor primitives that were created for one engine can be used on another.

To create engines, we must specify the :ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` and the index of the device of the given kind. There is only one CPU engine and one GPU engine, so the index for both engines must be 0.

.. ref-code-block:: cpp

	auto cpu_engine = :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(validate_engine_kind(engine::kind::cpu), 0);
	auto gpu_engine = :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(validate_engine_kind(engine::kind::gpu), 0);

In addition to an engine, all primitives require a :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` for the execution. The stream encapsulates an execution context and is tied to a particular engine.

In this example, a GPU stream is created.

.. ref-code-block:: cpp

	auto stream_gpu = stream(gpu_engine, stream::flags::in_order);





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub2:

Wrapping data into oneDNN GPU memory object
-------------------------------------------

Fill the data in CPU memory first, and then move data from CPU to GPU memory by reorder.

.. ref-code-block:: cpp

	const auto tz = memory::dims {2, 16, 1, 1};
	auto m_cpu
	        = memory({{tz}, memory::data_type::f32, memory::format_tag::nchw},
	                cpu_engine);
	auto m_gpu
	        = memory({{tz}, memory::data_type::f32, memory::format_tag::nchw},
	                gpu_engine);
	fill(m_cpu, tz);
	auto r1 = reorder(m_cpu, m_gpu);





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub3:

Creating a ReLU primitive
-------------------------

Let's now create a ReLU primitive for GPU.

The library implements the ReLU primitive as a particular algorithm of a more general :ref:`Eltwise <doxid-dev_guide_eltwise>` primitive, which applies a specified function to each element of the source tensor.

Just as in the case of :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`, a user should always go through (at least) three creation steps (which, however, can sometimes be combined thanks to C++11):

#. Create an operation primitive descriptor (here :ref:`dnnl::eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`) that defines the operation parameters including a GPU memory descriptor, and GPU engine. Primitive descriptor is a lightweight descriptor of the actual algorithm that implements the given operation.

#. Create a primitive (here :ref:`dnnl::eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`) that can be executed on GPU memory objects to compute the operation by a GPU engine.

.. note:: 

   Primitive creation might be a very expensive operation, so consider creating primitive objects once and executing them multiple times.
   
   
The code:

.. ref-code-block:: cpp

	// ReLU primitive descriptor, which corresponds to a particular
	// implementation in the library. Specify engine type for the ReLU
	// primitive. Use a GPU engine here.
	auto relu_pd = eltwise_forward::primitive_desc(gpu_engine,
	        prop_kind::forward, algorithm::eltwise_relu, m_gpu.get_desc(),
	        m_gpu.get_desc(), 0.0f);
	// ReLU primitive
	auto relu = eltwise_forward(relu_pd);





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub4:

Getting results from a oneDNN GPU memory object
-----------------------------------------------

After the ReLU operation, users need to get data from GPU to CPU memory by reorder.

.. ref-code-block:: cpp

	auto r2 = reorder(m_gpu, m_cpu);





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub5:

Executing all primitives
------------------------

Finally, let's execute all primitives and wait for their completion via the following sequence:

Reorder(CPU,GPU) -> ReLU -> Reorder(GPU,CPU).

#. After execution of the first Reorder, ReLU has source data in GPU.

#. The input and output memory objects are passed to the ReLU ``execute()`` method using a <tag, memory> map. Each tag specifies what kind of tensor each memory object represents. All :ref:`Eltwise <doxid-dev_guide_eltwise>` primitives require the map to have two elements: a source memory object (input) and a destination memory (output). For executing on GPU engine, both source and destination memory object must use GPU memory.

#. After the execution of the ReLU on GPU, the second Reorder moves the results from GPU to CPU.

.. note:: 

   All primitives are executed in the SAME GPU stream (the first parameter of the ``execute()`` method).
   
   
Execution is asynchronous on GPU. This means that we need to call :ref:`dnnl::stream::wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>` before accessing the results.

.. ref-code-block:: cpp

	// wrap source data from CPU to GPU
	r1.execute(stream_gpu, m_cpu, m_gpu);
	// Execute ReLU on a GPU stream
	relu.execute(stream_gpu, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, m_gpu}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, m_gpu}});
	// Get result data from GPU to CPU
	r2.execute(stream_gpu, m_gpu, m_cpu);

	stream_gpu.wait();





.. _doxid-cross_engine_reorder_cpp_1cross_engine_reorder_cpp_sub6:

Validate the result
-------------------

Now that we have the computed the result on CPU memory, let's validate that it is actually correct.

.. ref-code-block:: cpp

	if (find_negative(m_cpu, tz) != 0)
	    throw std::logic_error(
	            "Unexpected output, find a negative value after the ReLU "
	            "execution.");

Upon compiling and running the example, the output should be just:

.. ref-code-block:: cpp

	Example passed.

