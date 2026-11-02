.. index:: pair: page; Primitive Attributes: Scratchpad
.. _doxid-dev_guide_attributes_scratchpad:

Primitive Attributes: Scratchpad
================================

Some primitives might require a temporary buffer while performing their computations. For instance, the operations that do not have enough independent work to utilize all cores on a system might use parallelization over the reduction dimension (the K dimension in the GEMM notation). In this case different threads compute partial results in private temporary buffers, and then the private results are added to produce the final result. Another example is using matrix multiplication (GEMM) to implement convolution. Before calling GEMM, the source activations must be transformed using the im2col operation. The transformation result is written to a temporary buffer that is then used as an input for the GEMM.

In both of these examples, the temporary buffer is no longer required once the primitive computation is completed. oneDNN refers to such a memory buffer as a scratchpad.

.. warning:: 

   Do not confuse scratchpad with :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>`. The workspace is a buffer that is shared between forward and backward propagation of a primitive (hence must be preserved between the calls) and is used only in training.
   
   
The amount of space required for the scratchpad depends on the primitive and its actual implementation. For example, the GEMM-based convolution requires a scratchpad for the ``im2col`` data, while the direct convolution does not.

Both types of implementation might need extra space for the reduction in case there are too few independent tasks. The amount of memory required by the ``im2col`` transformation is proportional to the size of the source image multiplied by the weights spatial size. The size of a buffer for reduction is proportional to the tensor size to be reduced (e.g., ``diff_weights`` in the case of backward by weights) multiplied by the number of threads in the reduction groups (the upper bound is the total number of threads).

By contrast, some other primitives might require very little extra space. For instance, one of the implementation of the :ref:`dnnl::sum <doxid-structdnnl_1_1sum>` primitive requires temporary space only to store the pointers to data for each and every input array (that is, the size of the scratchpad is ``n * sizeof(void *)``, where ``n`` is the number of summands).

oneDNN supports two modes for handling scratchpads:

#. :ref:`dnnl::scratchpad_mode::library <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07fad521f765a49c72507257a2620612ee96>`. The library allocates memory for each primitive during its creation. This is the default behavior which enables user to not worry about the scratchpad at all. The scratchpad management policy can be configured at compile-time using the ONEDNN_ENABLE_CONCURRENT_EXEC (:ref:`Build Options <doxid-dev_guide_build_options>`) cmake option.
   
   * When ONEDNN_ENABLE_CONCURRENT_EXEC=OFF (default), a global scratchpad memory is shared across primitives. This mode minimizes the amount of memory needed for scratchpads at the application level. The global scratchpad is freed when all the primitives referencing it are destroyed.
     
     .. warning:: 
     
        In this mode, primitives can be created and executed in parallel but must be executed in the same thread they were created in. Executing primitives in a different thread than the one they were created in will result in segmentation fault. If you might execute a primitive in a thread different than the one it was created in, consider using :ref:`dnnl::scratchpad_mode::user <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>` or ONEDNN_ENABLE_CONCURRENT_EXEC=ON.
   
   * When ONEDNN_ENABLE_CONCURRENT_EXEC=ON, each primitive allocates its own private scratchpad memory. The scratchpad memory is freed when its primitive is destroyed. This mode can lead to larger memory footprint when compared to ONEDNN_ENABLE_CONCURRENT_EXEC=OFF.
     
     .. warning:: 
     
        In this mode, primitives can be created in one thread and executed in another. Also, different primitives can be run concurrently. If the same primitive is run from two different threads concurrently, the library will return incorrect results. If you might run the same primitive in two threads concurrently, consider using :ref:`dnnl::scratchpad_mode::user <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>` or ONEDNN_ENABLE_CONCURRENT_EXEC=OFF.

#. :ref:`dnnl::scratchpad_mode::user <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>`. A user provides scratchpad memory that has sufficient space at primitive execution (using the ``DNNL_ARG_SCRATCHPAD`` tag). This enables the user to reuse the memory as well as to make the primitives thread-safe. However, this requires a good memory manager (in terms of speed and locality) on the user's side.

.. warning:: 

   Primitives are not thread-safe by default. The only way to make the primitive execution fully thread-safe is to use the :ref:`dnnl::scratchpad_mode::user <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>` mode and not pass the same scratchpad memory to two primitives that are executed concurrently.
   
   
The scratchpad mode is controlled though the :ref:`dnnl_primitive_attr_set_scratchpad_mode <doxid-group__dnnl__api__attributes_1ga4adeb17e538392ec3a16d2f6ef3f7cca>` (C API) and :ref:`dnnl::primitive_attr::set_scratchpad_mode <doxid-structdnnl_1_1primitive__attr_1a91a597649afa13b7d2416b708d0620d2>` (C++ API) primitive attributes.

All primitives support both scratchpad modes.

Scratchpad Memory Engine
~~~~~~~~~~~~~~~~~~~~~~~~

If the user provides scratchpad memory to a primitive, this memory must be created using the same engine that the primitive uses.

Examples
~~~~~~~~

Library Manages Scratchpad
++++++++++++++++++++++++++

As mentioned above, this is a default behavior. We only want to highlight how a user can query the amount of memory consumed by a primitive due to a scratchpad.

.. ref-code-block:: cpp

	// Use default attr, hence the library allocates scratchpad
	dnnl::primitive::primitive_desc op_pd(engine, params, ...);
	
	// Print how much memory would be hold by a primitive due to scratchpad
	std::cout << "primitive will use "
	          << op_pd.query_s64(:ref:`dnnl::query::memory_consumption_s64 <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0ed44d67e94c1c7ac5f219491e422506>`)
	          << " bytes" << std::endl;
	
	// In this case scratchpad is internal, hence user visible scratchpad memory
	// descriptor should be empty:
	auto zero_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	assert(op_pd.scratchpad_desc() == zero_md);

User Manages Scratchpad
+++++++++++++++++++++++

.. ref-code-block:: cpp

	// Create an empty (default) attributes
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	
	// Default scratchpad mode is `library`:
	assert(attr.:ref:`get_scratchpad_mode <doxid-structdnnl_1_1primitive__attr_1af4131b946ec3af3bc2974b603d30029b>`() == :ref:`dnnl::scratchpad_mode::library <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07fad521f765a49c72507257a2620612ee96>`);
	
	// Set scratchpad mode to `user`
	attr.:ref:`set_scratchpad_mode <doxid-structdnnl_1_1primitive__attr_1a91a597649afa13b7d2416b708d0620d2>`(:ref:`dnnl::scratchpad_mode::user <doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>`);
	
	// Create a primitive descriptor with custom attributes
	dnnl::primitive::primitive_desc op_pd(engine, ..., attr);
	
	// Query the scratchpad memory descriptor
	:ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` :ref:`scratchpad_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a9cbdd03b65c030ef560b5555be1a86c2>` = op_pd.scratchpad_desc();
	
	// Note, that a primitive does not consume memory in this configuration:
	assert(op_pd.query_s64(:ref:`dnnl::query::memory_consumption_s64 <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0ed44d67e94c1c7ac5f219491e422506>`) == 0);
	
	// Create a primitive
	:ref:`dnnl::primitive <doxid-structdnnl_1_1primitive>` prim(op_pd);
	
	// ...
	
	// Create a scratchpad memory
	// NOTE: if scratchpad is not required for a particular primitive the
	//       scratchpad_md.get_size() will return 0. It is fine to have
	//       scratchpad_ptr == nullptr in this case.
	void *scratchpad_ptr = user_memory_manager::allocate(:ref:`scratchpad_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a9cbdd03b65c030ef560b5555be1a86c2>`.get_size());
	// NOTE: engine here must much the engine of the primitive
	:ref:`dnnl::memory <doxid-structdnnl_1_1memory>` scratchpad(scratchpad_md, engine, scratchpad_ptr);
	
	// Pass a scratchpad memory to a primitive
	prim.execute(stream, {
	        ...,
	        {:ref:`DNNL_ARG_SCRATCHPAD <doxid-group__dnnl__api__primitives__common_1ga81836a4db2cb1c4a14d959e304d3f63d>`, scratchpad}});

