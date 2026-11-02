.. index:: pair: page; Performance Profiling Example
.. _doxid-performance_profiling_cpp:

Performance Profiling Example
=============================

This example demonstrates the best practices for application performance optimizations with oneDNN.

This example demonstrates the best practices for application performance optimizations with oneDNN.

Example code: :ref:`performance_profiling.cpp <doxid-performance_profiling_8cpp-example>`

This example uses :ref:`ONEDNN_VERBOSE <doxid-dev_guide_verbose>` trace output to tune oneDNN code to align with the :ref:`best practices <doxid-dev_guide_inference>`.

It assumes knowledge of memory formats and their usage in oneDNN. You can read more about this topic :ref:`here <doxid-memory_format_propagation_cpp>`.

Additionally, see the :ref:`article for recommended environment for <doxid-dev_guide_performance_settings>` running benchmarks".

The example has three different implementations of the mathematical operation:

#. Naive implementation executes 2D convolution followed by ReLU on the data in NCHW format. This implementation does not align with oneDNN best practices and results in suboptimal performance.

#. Blocked format implementation executes the same operations sequence on the blocked format optimized for convolution performance. This implementation uses ``format_tag=ANY`` to create a convolution memory descriptor to determine the data format optimal for the convolution implementation. It then propagates the blocked format to the non-intensive ReLU. This implementation results in better overall performance than the naive implementation.

#. Fused implementation executes convolution fused with ReLU on blocked data format. This implementation uses ``format_tag=ANY`` to create a convolution memory descriptor, and then adds ReLU as a post-op to the convolution primitive. This version implements all of the best practices for inference resulting in the best overall performance.



.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_walkthrough:

Walkthrough
~~~~~~~~~~~

The program in :ref:`performance_profiling.cpp <doxid-performance_profiling_8cpp-example>` includes all three implementations introduced above. You can select the specific implementation using command line options.

After compilation, you can execute each implementation with:

.. ref-code-block:: cpp

	./program.exe [cpu|gpu] [implementation]

Before you run the program, set your ``ONEDNN_VERBOSE`` environment variable to 1:

.. ref-code-block:: cpp

	export ONEDNN_VERBOSE=1

The program starts by creating oneDNN memory objects in NCHW format. These are called ``user_`` because they are meant to represent the user's source data entering oneDNN with the NCHW format.

.. ref-code-block:: cpp

	// set dimensions for synthetic data and weights
	const memory::dim BATCH = 128;
	const memory::dim IC = 3, OC = 96;
	const memory::dim IH = 227, KH = 11, OH = 55;
	const memory::dim IW = 227, KW = 11, OW = 55;


.. note:: 

   Here the library allocates memory.
   
   


.. ref-code-block:: cpp

	// create oneDNN memory objects for user's tensors (in nchw and oihw formats)
	auto user_src = memory({{BATCH, IC, IH, IW}, memory::data_type::f32,
	                               memory::format_tag::nchw},
	        eng);
	auto user_wei = memory({{OC, IC, KH, KW}, memory::data_type::f32,
	                               memory::format_tag::oihw},
	        eng);
	auto user_dst = memory({{BATCH, OC, OH, OW}, memory::data_type::f32,
	                               memory::format_tag::nchw},
	        eng);


.. note:: 

   You can change the batch size to easily increase/decrease the workload.
   
   
The following descriptions of each implementation will reference each other, and are meant to be read in order.





.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_implementation1:

Naive Implementation
~~~~~~~~~~~~~~~~~~~~

This implementation is launched with the following shell code:

.. ref-code-block:: cpp

	./program.exe cpu naive

The program will call the implementation defined in the function ``conv_relu_naive()``.

First it sets the dimensions and format for convolution memory descriptors (``_md``) to match ``user_`` values one ``md`` each for source, destination, and weight data. Then it uses those ``md`` to create the convolution primitive descriptor ``conv_pd``, which tells oneDNN to use plain format (NCHW) for the convolution.

.. ref-code-block:: cpp

	// copy the dimensions and format from user's memory
	auto conv_src_md = memory::desc(user_src.get_desc());
	auto conv_wei_md = memory::desc(user_wei.get_desc());
	auto conv_dst_md = memory::desc(user_dst.get_desc());

Next the program creates a convolution primitive descriptor ``conv_pd`` and convolution primitive ``conv``. These structs will inherit NCHW format from ``md`` by way of the ``conv_d``. Finally it creates the convolution primitive ``conv`` and adds it to the stream ``s``, and then executes the ``create_and_execute_relu(user_dst)`` function.

.. ref-code-block:: cpp

	// create a convolution primitive descriptor
	auto conv_pd = convolution_forward::primitive_desc(eng,
	        prop_kind::forward_inference, algorithm::convolution_direct,
	        conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding);



.. ref-code-block:: cpp

	// create convolution primitive
	auto conv = convolution_forward(conv_pd);



.. ref-code-block:: cpp

	// execute convolution by adding it to the stream s
	conv.execute(s,
	        {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, user_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, user_wei},
	                {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, user_dst}});



.. ref-code-block:: cpp

	// execute relu (on convolution's destination format, whatever it is)
	create_and_execute_relu(user_dst, eng, s);
	s.wait();


.. note:: 

   The function for creation and execution of ReLU primitive is defined elsewhere to keep this example clean. It is an non-intensive operation, so the ``create_and_execute_relu()`` function uses whatever the input data format is at the time it is called.
   
   
Using NCHW data format may result in suboptimal performance for compute intensive primitives, as shown in the following ONEDNN_VERBOSE output by the convolution and relu execution times of 38.3 and 2.9 milliseconds, respectively.

ONEDNN_VERBOSE output (see configuration notice\*):

.. ref-code-block:: cpp

	onednn_verbose,exec,cpu,convolution,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_undef::undef::f0 dst_f32::blocked:abcd:f0,,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,38.314
	onednn_verbose,exec,cpu,eltwise,jit:avx512_common,forward_inference,data_f32::blocked:abcd:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,128x96x55x55,2.87695

In Blocked format implementation, we will incorporate the best practice of letting oneDNN determine the optimal format for convolution primitive.





.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_implementation2:

Blocked format implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This implementation is launched with the following shell code:

.. ref-code-block:: cpp

	./program.exe cpu blocked

The program will call the implementation defined in the function ``conv_relu_blocked()``.

First it creates the md as in naive implementation. Next it changes the :ref:`dnnl::memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` for each md to ``ANY``. Then it uses those md to create the convolution primitive descriptor conv_pd, which tells oneDNN to use whatever format it recommends for the convolution. oneDNN will choose a friendly blocked format.

.. ref-code-block:: cpp

	// copy the dimensions and data type from user's memory and set format tag
	// to "any" to allow convolution to pick the best implementation
	auto conv_src_md = memory::desc(user_src.get_desc().get_dims(),
	        user_src.get_desc().get_data_type(), memory::format_tag::any);
	auto conv_wei_md = memory::desc(user_wei.get_desc().get_dims(),
	        user_wei.get_desc().get_data_type(), memory::format_tag::any);
	auto conv_dst_md = memory::desc(user_dst.get_desc().get_dims(),
	        user_dst.get_desc().get_data_type(), memory::format_tag::any);

Next the program creates a convolution primitive descriptor conv_pd and convolution primitive conv as in naive implementation. However, in this implementation the structs will inherit blocked format from md by way of the conv_d.

.. ref-code-block:: cpp

	// create a convolution primitive descriptor and primitive
	auto conv_pd = convolution_forward::primitive_desc(eng,
	        prop_kind::forward_inference, algorithm::convolution_direct,
	        conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding);

Since the resulting convolution primitive will expect blocked source data, conditional reorders are inserted to convert input data to blocked format if required. The input data user_src is NCHW, so this conditional will be triggered:

.. note:: 

   The reoders are applied using oneDNN ``reorder`` primitive.
   
   


.. ref-code-block:: cpp

	// prepare convolution source
	memory conv_src = user_src;
	if (conv_pd.src_desc() != user_src.get_desc()) {
	    conv_src = memory(conv_pd.src_desc(), eng);
	    auto r_pd = reorder::primitive_desc(user_src, conv_src);
	    reorder(r_pd).execute(s, user_src, conv_src);
	}

	// prepare convolution weights
	memory conv_wei = user_wei;
	if (conv_pd.weights_desc() != user_wei.get_desc()) {
	    conv_wei = memory(conv_pd.weights_desc(), eng);
	    auto r_pd = reorder::primitive_desc(user_wei, conv_wei);
	    reorder(r_pd).execute(s, user_wei, conv_wei);
	}

	// prepare convolution destination
	memory conv_dst = user_dst;
	if (conv_pd.dst_desc() != user_dst.get_desc())
	    conv_dst = memory(conv_pd.dst_desc(), eng);

Finally it creates the convolution primitive ``conv`` and adds it to the stream ``s`` with the reordered data (``conv_src``, ``conv_wei``, ``conv_dst1``) as inputs and then executes the ``create_and_execute_relu(conv_dst)`` function.

.. ref-code-block:: cpp

	// create convolution primitive
	auto conv = convolution_forward(conv_pd);



.. ref-code-block:: cpp

	// execute convolution by adding it to the stream s
	conv.execute(s,
	        {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_wei},
	                {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst}});



.. ref-code-block:: cpp

	// execute relu (on convolution's destination format, whatever it is)
	create_and_execute_relu(conv_dst, eng, s);

Blocked memory format is recommended for oneDNN primitive execution and provides better performance, as shown in the ONEDNN_VERBOSE output by the convolution and relu execution times of 18.3 and 2.7 milliseconds (down from 38.3 and 2.9 in naive implementation), respectively. In this implementation, there is an additional reorder operation that executes before and after the the conv + relu. This small cost is worth the gain from executing in blocked format. If fact, it becomes negligible when chaining together multiple oneDNN operations in succession. In these situations, you can do one reorder at the beginning and one at the end of the chain, and only pay the reorder penalty at those points in the execution.

ONEDNN_VERBOSE output (see configuration notice\*):

.. ref-code-block:: cpp

	onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,96x3x11x11,0.0310059
	onednn_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,18.3101
	onednn_verbose,exec,cpu,eltwise,jit:avx512_common,forward_inference,data_f32::blocked:aBcd16b:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,128x96x55x55,2.66895
	onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,128x96x55x55,4.80396

This inference implementation is closer to best practices than naive implementation because it uses oneDNN recommended memory format. fused implementation will futher optimize the performance by fusing convolution with ReLU using oneDNN :ref:`post-ops <doxid-dev_guide_attributes_post_ops>`.





.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_implementation3:

Fused Implementation
~~~~~~~~~~~~~~~~~~~~

This implementation is launched with the following shell code:

.. ref-code-block:: cpp

	./program.exe cpu fused

The program will call the implementation defined in the function ``conv_relu_fused()``.

First the memory descriptors and convolution primitive descriptor are created as in naive implementation.

Then in preparation for the convolution prim desctiptor, a ReLU post-op is built and added to the primitive attribute ``attr`` :

.. ref-code-block:: cpp

	// function to create post-op attribute for fused relu
	primitive_attr create_attr_with_relu_post_op() {
	    // create a post-op with relu
	    post_ops ops;
	    ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
	
	    // create an attribute and set the corresponding post op
	    primitive_attr attr;
	    attr.set_post_ops(ops);
	
	    return attr;
	}

post-op by way of the attributes ``attr`` :

.. ref-code-block:: cpp

	// create an attribute for fused relu
	auto attr = create_attr_with_relu_post_op();

	// create a convolution primitive descriptor
	auto conv_pd = convolution_forward::primitive_desc(eng,
	        prop_kind::forward_inference, algorithm::convolution_direct,
	        conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding,
	        attr);

Then conditional reorders are applied as in blocked format implementation to convert ``user_`` format NCHW to blocked. Finally, it creates the convolution primitive ``conv`` and adds it to the stream ``s`` with the reordered data (``conv_src``, ``conv_wei``, ``conv_dst1``).

.. note:: 

   There is no separate addition to the stream for the ReLU operation because it has been added as a post-op to the ``conv`` primitive.
   
   


.. ref-code-block:: cpp

	// create convolution primitive
	auto conv = convolution_forward(conv_pd);



.. ref-code-block:: cpp

	// execute convolution by adding it to the stream s
	conv.execute(s,
	        {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_wei},
	                {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst}});

This implementation complies with best practices for f32 inference by using the oneDNN recommended blocked format for convolution and adding ReLU as a post-op to execute a fused version of conv + ReLU. The consequence to following best practices can be seen in the execution time of the fused primitive of 18.0 milliseconds.

ONEDNN_VERBOSE output (see configuration notice\*):

.. ref-code-block:: cpp

	onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,96x3x11x11,0.0148926
	onednn_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,post_ops:'eltwise_relu;';,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,17.968
	onednn_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,128x96x55x55,4.66797





.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_roundup:

Performance summary
~~~~~~~~~~~~~~~~~~~

===============  =========  ===================  
Implementation   Time, ms   Cumulative speedup   
===============  =========  ===================  
Naive            41.2       1.0                  
Blocked format   21.0       2.0                  
Fused            18.0       2.3                  
===============  =========  ===================





.. _doxid-performance_profiling_cpp_1performance_profiling_cpp_config:

Configuration Notice
~~~~~~~~~~~~~~~~~~~~

.. note:: 

   This example is meant to demonstrate oneDNN best practices.
   
   

.. note:: 

   It is not meant for benchmarking purposes. The platform is not fully
   
   

.. note:: 

   optimized, so the primitive execution times are only relevant in
   
   

.. note:: 

   relation to the other times in this example.
   
   
Runtime Settings:

* OMP_NUM_THREADS=14

* KMP_AFFINITY=granularity=fine,compact

Platform:

* CPU: Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz

* Thread(s) per core: 1

* Core(s) per socket: 28

* Socket(s): 2

* NUMA node(s): 2

* RAM (DDR4): 192 GB

