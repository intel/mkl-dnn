.. index:: pair: page; Inference
.. _doxid-dev_guide_inference:

Inference
=========

oneDNN includes primitives for operations throughout a deep learning network topology. However, it is important to note the scope of oneDNN is limited to performance critical functionality and the library does not provide all the functions necessary to implement deep learning workloads, for instance data preprocessing or computing loss function. The soft-max classifier is the sole classifier included, but the application of other classifier types will require user's own implementations. The scope of the library is depicted in the following image:

.. image:: img_inference_scope.jpg



Best Practices for Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fp32 Inference
~~~~~~~~~~~~~~

Use Forward Inference Primitives

oneDNN provides a forward pass version of each primitive, that avoids storing information required for a backward pass (as in training).

Use the :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` argument at creation of the primitive descriptor, as in this convolution example:

.. ref-code-block:: cpp

	auto conv_prim_descr = convolution_forward::primitive_desc(engine, prop_kind::forward_inference, ...);

Layout Propagation

Compute-intensive oneDNN primitives execute with highest performance on CPU-friendly data formats. Please see description of data formats :ref:`here <doxid-memory_format_propagation_cpp>`.

Performance gains are maximized by reordering once, and then propagating the CPU-friendly format through as many layers as possible in your topology. oneDNN provides the ``format_tag=any`` for memory descriptors that will be passed to compute-intensive primitives. The compute-intensive primitive types in oneDNN are :ref:`Convolution <doxid-dev_guide_convolution>`, :ref:`Inner Product <doxid-dev_guide_inner_product>`, and :ref:`RNN <doxid-dev_guide_rnn>`.

To accomplish this propagation in a robust manner, its is recommended to follow these steps:

A. On compute-intensive operations:

* Pass the ``format_tag=any`` when creating oneDNN memory descriptor for source, destination, and weights memory

* Use these three memory descriptors with ``format_tag=any`` to create primitive descriptor

* Query the primitive descriptor with ``.src_desc()`` method to get recommended format

* Write conditional reorder to execute only if user source data or weights don't match the recommended format

* Create primitive and add it to stream with ``primitive.execute(stream, args)``

B. On non-intensive operations:

* Query output primitive descriptor with ``.dst_desc()`` from previous operation to find current layout

* Pass current layout with ``format_tag=.dst_desc()`` when creating non-intensive primitive descriptor

* Create primitive and add it to stream with ``operation.execute(stream, args)``

Now let's take a look at the code syntax to accomplish the compute-intensive steps:

Pass the ``format_tag=any`` when creating oneDNN memory descriptor for source, destination, and weights memory

.. ref-code-block:: cpp

	source_mem_descr = memory::desc(args*, memory::format_tag::any);
	dest_mem_descr = memory::desc(args*, memory::format_tag::any);
	weights_mem_descr = memory::desc(args*, memory::format_tag::any);

Use these three memory descriptors with 'format_tag=any` to create primitive descriptor

.. ref-code-block:: cpp

	auto conv_prim_descr = convolution_forward::primitive_desc(...,
	            source_mem_descr, weights_mem_descr, dest_mem_descr);

Query the primitive descriptor with ``.src_desc()`` method to get recommended format Write conditional reorder to execute only if user source data or weights don't match the recommended format (Note: Do this for weight_memory as well)

.. ref-code-block:: cpp

	memory conv_source_memory = user_source_memory;
	if (conv_prim_descr.src_desc() != user_source_memory.get_desc()) {
	    conv_source_memory = memory(conv_prim_descr.src_desc(), :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`);
	    auto reorder_prim_descr = reorder::primitive_desc(user_source_memory, conv_source_memory);
	    reorder(reorder_prim_descr).execute(s, user_source_memory, conv_source_memory);
	}

Create primitive and add it to stream with ``primitive.execute(stream, args)``

.. ref-code-block:: cpp

	auto conv = convolution_forward(conv_prim_descr);
	conv.execute(s, {
	            {:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_source_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dest_memory}});

Cache Weights \ Weights are accessed many times during batched inference. At inference time these weights are essentially constants in the mapping function that the network is applying to the input data. As such, the weights should be reordered (if necessary) once and then used in the reorder form for the duration of the execution. This caching causes the computer to use them in a way similar to how a mathematical function applies a constant, i..e, "Grab-and-go" with no overhead for creation or reorder.

Primitive Reuse \ There is JIT compilation overhead associated with primitive creation. It is recommended to reuse any primitive that you can, and only create them once.

Fused Primitives \ oneDNN provides fused versions of primitives that attach a non-intensive operation to the end of a compute-intensive operation and then executes both in a single pass, reducing the number of memory accesses needed for the combined operations. The non-intensive operation is added as a post-op attribute to the compute intensive primitive descriptor. Please note that post-ops do not change the number of inputs or outputs of the primitives. Please see the "Post-ops and Attributes" section of the doc for each primitive type in /docs/primitive/ for a list of available fused primitives.

A good example is adding ReLU as a post-op to convolution, which we will use as a demonstration below. The steps are

* Create a ``post_op`` for fused ReLU

* Create primitive attribute and add the ``post_op``

* Create a convolution primitive descriptor, passing ``post_op as`` an arg

Create a ``post_op`` for fused ReLU

.. ref-code-block:: cpp

	post_ops ops;
	ops.append_eltwise(..., algorithm::eltwise_relu);

Create primitive attribute and add the ``post_op``

.. ref-code-block:: cpp

	primitive_attr attr;
	attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(ops);

Create a convolution primitive descriptor, passing the post-op infused ``attrs`` as an arg

.. ref-code-block:: cpp

	auto conv_prim_descr = convolution_forward::primitive_desc(..., attrs, engine);

int8 Inference
~~~~~~~~~~~~~~

oneDNN supports low precision int8 for inference execution. Note that not all primitives have int8 versions. Sometimes the speed benefits would be minimal, or the loss in accuracy is not acceptable. Also the soft-max classifier only supports fp32, so int8 inference will require a reorder before executing this primitive.

By default, the oneDNN reorder primitive does not scale upon casting to int8. In order to compress fp32 data to int8 precision while still preserving the entire shape of the distribution, a process called quantization must applied. Quantization will scale the data based on its range to efficiently fill the bits available for int8 type.

To achieve quantization upon casting, the user must provide a few inputs to oneDNN in order to use int8 inference:

* Specify data type at creation of primitive descriptor (int8 in this case)

* Provide a scaling factor for oneDNN reorder primitive

* Provide an output scaling factor the operation primitive

Please see the dedicated :ref:`section <doxid-dev_guide_inference_int8>` on low precision computations in oneDNN for a detailed discussion, including how to calculate the scaling factors.

