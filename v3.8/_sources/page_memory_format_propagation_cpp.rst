.. index:: pair: page; Memory Format Propagation
.. _doxid-memory_format_propagation_cpp:

Memory Format Propagation
=========================

This example demonstrates memory format propagation, which is critical for deep learning applications performance.

This example demonstrates memory format propagation, which is critical for deep learning applications performance.

Example code: :ref:`memory_format_propagation.cpp <doxid-memory_format_propagation_8cpp-example>`

Memory format propagation is one of the central notions that needs to be well-understood to use oneDNN correctly.

Convolution and inner product primitives choose the memory format when you create them with the placeholder memory format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for input or output. The memory format chosen is based on different circumstances such as hardware and convolutional parameters. Using the placeholder memory format is the recommended practice for convolutions, since they are the most compute-intensive operations in most topologies where they are present.

Other primitives, such as Elementwise, LRN, batch normalization and other, on forward propagation should use the same memory format as the preceding layer thus propagating the memory format through multiple oneDNN primitives. This avoids unnecessary reorders which may be expensive and should be avoided unless a compute-intensive primitive requires a different format. For performance reasons, backward computations of such primitives requires consistent memory format with the corresponding forward computations. Hence, when initializing there primitives for backward computations you should use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` memory format tag as well.

Below is the short summary when to use and not to use memory format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` during operation description initialization:

==============================================================================================================================  =========================================================================================================================================================================================================  ===========================================================================================================================================================================================================  =========================================================================================================================================================================================================  
Primitive Kinds                                                                                                                 Forward Propagation                                                                                                                                                                                        Backward Propagation                                                                                                                                                                                         No Propagation                                                                                                                                                                                             
==============================================================================================================================  =========================================================================================================================================================================================================  ===========================================================================================================================================================================================================  =========================================================================================================================================================================================================  
Compute intensive: (De-)convolution, Inner product, RNN                                                                         Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`                                                                  Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`                                                                    N/A                                                                                                                                                                                                        
Compute intensive (no propagation): Matrix Multiplication                                                                       N/A                                                                                                                                                                                                        N/A                                                                                                                                                                                                          Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`                                                                  
Memory-bandwidth limited: Pooling, Layer and Batch Normalization, Local Response Normalization, Elementwise, Shuffle, Softmax   Use memory format from preceding layer for inputs, and :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for outputs   Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for gradient tensors, and actual memory formats for data tensors   N/A                                                                                                                                                                                                        
Memory-bandwidth limited: Reorder, Concat, Sum, Binary                                                                          N/A                                                                                                                                                                                                        N/A                                                                                                                                                                                                          Use memory format from preceding layer for inputs, and :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for outputs   
==============================================================================================================================  =========================================================================================================================================================================================================  ===========================================================================================================================================================================================================  =========================================================================================================================================================================================================

Additional format synchronization is required between forward and backward computations when running training workloads. This topic is covered in :ref:`Training-Specific Aspects <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_training>`.

For better understanding of the architecture and design of oneDNN as well as the concepts used in the library, please refer to :ref:`Understanding Memory Formats <doxid-dev_guide_understanding_memory_formats>`.



.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_intro:

Introduction to the tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This C++ API example demonstrates how to use optimized memory formats supported by oneDNN:

* How to configure primitives to use optimized memory formats.

* How to determine whether data needs to be reordered from/to optimized memory formats.

This tutorial assumes that the reader has already reviewed the :ref:`Getting started <doxid-getting_started_cpp>` tutorial.

The example is built around a CNN consisting of a convolution followed by a pooling and consists of the following steps:

#. Create a pooling primitive descriptor based on the memory format chosen by the convolution primitive.

#. Create memory descriptors for input and output data in the NCHW memory format.

#. Determine if input and output data needs to be reordered from/to the optimized memory format.

#. Create memory objects; and necessary primitives and execute them.

These steps are implemented in the :ref:`memory_format_propagation() function <doxid-memory_format_propagation_cpp_1memory_format_propagation_tutorial>` which in turn is called from ``main()`` which is also responsible for error handling.





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_tutorial:

memory_format_propagation() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub1:

Initialization
--------------

We start by creating an engine and a stream that we will use when creating primitive descriptors and executing primitives.

.. ref-code-block:: cpp

	:ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>` eng(engine_kind, 0);
	stream s(eng);





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub2:

Create convolution and pooling primitives
-----------------------------------------

To specify that a primitive should pick an optimized format for the specified computation parameters, we create memory descriptors with memory format set to :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`.

This approach works only for a limited set of primitives: convolutions and inner products. Additionally, :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` can be specified for destination memory descriptors which implies that destination will have the same memory format as the source.

.. ref-code-block:: cpp

	// Tensor and kernel dimensions. We use the same 3x3 kernel with padding=1
	// for both convolution and pooling primitives, which means that the
	// activation tensor shapes do not change.
	const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;
	auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32,
	        memory::format_tag::any // let convolution choose memory format
	);
	auto conv_weights_md = memory::desc(
	        {OC, IC, KH, KW}, memory::data_type::f32,
	        memory::format_tag::any // let convolution choose memory format
	);
	auto conv_dst_md = memory::desc({N, OC, H, W}, memory::data_type::f32,
	        memory::format_tag::any // let convolution choose memory format
	);
	const auto &pool_dst_md = conv_dst_md; // shape does not change

Next, we pass the memory descriptors to primitive descriptors constructors.

.. ref-code-block:: cpp

	auto conv_pd = convolution_forward::primitive_desc(
	        eng, prop_kind::forward_inference, algorithm::convolution_auto,
	        conv_src_md, conv_weights_md,
	        conv_dst_md, // shape information
	        {1, 1}, // strides
	        {1, 1}, {1, 1} // left and right padding
	);

	auto pool_pd
	        = pooling_forward::primitive_desc(eng, prop_kind::forward_inference,
	                algorithm::pooling_max, conv_pd.dst_desc(),
	                pool_dst_md, // shape information
	                {1, 1}, {KH, KW}, // strides and kernel
	                {0, 0}, // dilation
	                {1, 1}, {1, 1} // left and right padding
	        );





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub3:

Create source and destination memory objects
--------------------------------------------

We assume that the 'user' source and destination memory format is NCHW. Since there is no result validation in this tutorial, we do not bother with filling the data with some values and let oneDNN allocate the memory.

.. ref-code-block:: cpp

	auto src_mem = memory(
	        {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
	        eng);
	auto weights_mem = memory({{OC, IC, KH, KW}, memory::data_type::f32,
	                                  memory::format_tag::oihw},
	        eng);
	auto dst_mem = memory(
	        {{N, OC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
	        eng);





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub4:

Determine if source and destination need to be reordered
--------------------------------------------------------

The idiomatic way to check if a reorder is necessary between the memory format expected a primitive (the convolution in our case) and the available memory format is to compare the corresponding memory descriptors.

.. ref-code-block:: cpp

	bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();


.. warning:: 

   It is by design that it is not possible to just compare memory tags. The reason behind this is that a memory format tags only provide a partial description of how data is laid out in memory and do not, for example, describe memory objects obtained via sub-memory constructor.
   
   
We repeat the process for the weights and destination memory format descriptors as well.

.. ref-code-block:: cpp

	bool need_reorder_weights
	        = conv_pd.weights_desc() != weights_mem.get_desc();
	bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub45:

Allocate intermediate buffers if necessary
------------------------------------------

Based on the flags computed before, we can now decide if we need extra intermediate buffers to hold the source and weights data for the convolution and the output of the pooling.

Memory objects for the intermediate buffers are created based on the memory descriptors obtained from the primitive descriptors to ensure consistency.

.. ref-code-block:: cpp

	auto conv_src_mem
	        = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
	auto conv_weights_mem = need_reorder_weights
	        ? memory(conv_pd.weights_desc(), eng)
	        : weights_mem;
	auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
	auto pool_dst_mem
	        = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub5:

Perform reorders for source data if necessary
---------------------------------------------

Now we get to the part where we actually start executing things. We check if reorders are necessary based on the flags computed before and create and execute them immediately.

.. note:: 

   We call :ref:`dnnl::stream::wait() <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>` before reorder primitives get out of scope and destroyed to accommodate for potentially asynchronous execution.
   
   


.. ref-code-block:: cpp

	if (need_reorder_src) {
	    auto reorder_src = reorder(src_mem, conv_src_mem);
	    reorder_src.execute(
	            s, {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, src_mem}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_src_mem}});
	    s.wait(); // wait for the reorder to complete
	}

	if (need_reorder_weights) {
	    auto reorder_weights = reorder(weights_mem, conv_weights_mem);
	    reorder_weights.execute(s,
	            {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, weights_mem},
	                    {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_weights_mem}});
	    s.wait(); // wait for the reorder to complete
	}





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub6:

Create and execute convolution and pooling primitives
-----------------------------------------------------

After the reorders, we are now ready to compute convolution and pooling.

.. ref-code-block:: cpp

	auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), eng);
	auto conv = convolution_forward(conv_pd);
	conv.execute(s,
	        {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_mem}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_mem},
	                {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst_mem}});
	auto pool_scratchpad_mem = memory(pool_pd.scratchpad_desc(), eng);
	auto pool = pooling_forward(pool_pd);
	pool.execute(
	        s, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_dst_mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_mem}});
	s.wait();





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_sub7:

Reorder destination data if necessary
-------------------------------------

The only potentially remaining operation is a reorder from the pooling destination memory object to the user's one. Similarly to the reorders for the source and weights memory objects, it is performed depending on the value of the previously computed flag.

.. ref-code-block:: cpp

	if (need_reorder_dst) {
	    auto reorder_dst = reorder(pool_dst_mem, dst_mem);
	    reorder_dst.execute(
	            s, {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool_dst_mem}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, dst_mem}});
	    s.wait();
	}





.. _doxid-memory_format_propagation_cpp_1memory_format_propagation_results:

Results
-------

Upon compiling and run the example the output should be just:

.. ref-code-block:: cpp

	Example passed.

It may be interesting to check what really happens during the run. We can use ``ONEDNN_VERBOSE`` environment variable for that (see also :ref:`Verbose Mode <doxid-dev_guide_verbose>`). Here's an example output:

.. ref-code-block:: cpp

	$ ONEDNN_VERBOSE=1 ./memory-format-propagation-cpp
	onednn_verbose,v0,info,oneDNN <ver> (Git Hash <hash>)
	onednn_verbose,v0,info,cpu,runtime:OpenMP
	onednn_verbose,v0,info,cpu,isa:Intel AVX2
	onednn_verbose,v0,info,gpu,runtime:none
	onednn_verbose,v0,exec,cpu,reorder,jit:uni,undef,
	    src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,1x128x14x14,0.326904
	onednn_verbose,v0,exec,cpu,reorder,jit:uni,undef,
	    src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,256x128x3x3,0.244141
	onednn_verbose,v0,exec,cpu,convolution,jit:avx2,forward_inference,
	    src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd8b:f0,,
	    alg:convolution_direct,mb1_ic128oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,1.20312
	onednn_verbose,v0,exec,cpu,pooling,jit:avx,forward_inference,
	    src_f32::blocked:aBcd8b:f0 dst_f32::blocked:aBcd8b:f0 ws_undef::undef::f0,,
	    alg:pooling_max,mb1ic256_ih14oh14kh3sh1ph1_iw14ow14kw3sw1pw1,0.187012
	onednn_verbose,v0,exec,cpu,reorder,jit:uni,undef,
	    src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x14x14,0.0419922
	Example passed on CPU.

From this output we can deduce that:

* The convolution primitive picked up :ref:`dnnl::memory::format_tag::aBcd8b <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa448a7fc9219294ce172b0edf9498b5c4>` optimized memory format for activations. In this format the channels dimension (denoted by letter B since it is the second dimension; see also :ref:`Naming Conventions <doxid-dev_guide_conventions>`) is blocked by a factor of 8. Because of this memory format is different from the NCHW format the tutorial uses, the source and destination had to be reordered to and from this optimized memory layout.

* The convolution primitive picked up :ref:`dnnl::memory::format_tag::ABcd8b8a <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fabcbce50e9c241458767871fa053e1ba0>` optimized memory format (output (A) and input (B) channel dimensions blocked by 8) which we also had to reorder the initial weights to since they are in the OIHW memory format.

