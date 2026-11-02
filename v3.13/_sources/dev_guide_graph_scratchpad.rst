.. index:: pair: page; Scratchpad Management
.. _doxid-dev_guide_graph_scratchpad:

Scratchpad Management
=====================

Introduction
~~~~~~~~~~~~

Some compiled partitions require temporary memory (scratchpad) during execution to store the intermediate results. The amount of space required for the scratchpad depends on the compiled partition and its actual implementation. oneDNN Graph API supports two different scratchpad management modes:

#. User-managed scratchpad : The user queries the required scratchpad size, allocates the buffer, and passes it to the execution call. This mode gives the user full control over scratchpad memory lifecycle, enabling buffer reuse across multiple executions and reducing allocation overhead.

#. Library-managed scratchpad : The library internally allocates and frees the scratchpad buffer during each execution call. This mode is simple to use but with its own limitations.

User-Managed Scratchpad
~~~~~~~~~~~~~~~~~~~~~~~

To use user-managed scratchpad, follow these steps:

#. Query the scratchpad logical tensor from the compiled partition.

#. Check if a scratchpad is required (memory size > 0).

#. Allocate a tensor with the queried logical tensor descriptor.

#. Pass the scratchpad tensor to the execution call.

.. ref-code-block:: cpp

	// Compile the partition
	:ref:`dnnl::graph::compiled_partition <doxid-classdnnl_1_1graph_1_1compiled__partition>` cp = partition.compile(inputs, outputs, engine);
	
	// Step 1: Query scratchpad requirements
	:ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` scratchpad_lt = cp.:ref:`get_scratchpad_logical_tensor <doxid-classdnnl_1_1graph_1_1compiled__partition_1a21ec243a10b40b495f0d9a84c9f78fd1>`();
	
	// Step 2: Check if scratchpad is needed
	:ref:`dnnl::graph::tensor <doxid-classdnnl_1_1graph_1_1tensor>` scratchpad_ts;
	if (scratchpad_lt.:ref:`get_mem_size <doxid-classdnnl_1_1graph_1_1logical__tensor_1a12b73d1201259d4260de5603f62c7f15>`() > 0) {
	    // Step 3: Allocate scratchpad tensor
	    void * user_scratchpad = user_allocate_method(scratchpad_lt.:ref:`get_mem_size <doxid-classdnnl_1_1graph_1_1logical__tensor_1a12b73d1201259d4260de5603f62c7f15>`());
	    scratchpad_ts = :ref:`dnnl::graph::tensor <doxid-classdnnl_1_1graph_1_1tensor>`(scratchpad_lt, engine, user_scratchpad);
	}
	
	// Step 4: Execute with user-managed scratchpad
	cp.:ref:`execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1aa93ee468a74520638dc87212415b1ea6>`(stream, input_tensors, output_tensors, scratchpad_ts);

The user-managed scratchpad tensor follows the same rules and limitations as input and output tensors. It must be created using the same engine as partition compilation unless specified otherwise. If the same compiled partition is executed in multiple threads concurrently, a separate scratchpad buffer must be used per thread to ensure the thread safety. A user-managed scratchpad buffer can be reused across multiple executions of the same or different compiled partitions where the size fits.

Library-Managed Scratchpad
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, when no scratchpad tensor is provided to the execution call, the library allocates the required scratchpad memory internally and frees it after execution completes, through the allocator interfaces associated with engine object. This mode requires no additional user action.

.. ref-code-block:: cpp

	// Compile the partition
	:ref:`dnnl::graph::compiled_partition <doxid-classdnnl_1_1graph_1_1compiled__partition>` cp = partition.compile(inputs, outputs, engine);
	
	// Execute without scratchpad - library manages it internally
	cp.:ref:`execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1aa93ee468a74520638dc87212415b1ea6>`(stream, input_tensors, output_tensors);

Work with SYCL Graph Recording Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User-managed scratchpad is required when working with SYCL graph recording mode.

SYCL graph recording captures a sequence of kernel submissions into a graph object that can be replayed multiple times. During recording, all memory buffers accessed by the recorded kernels are bound to the graph. These buffers must remain valid across all subsequent replays.

When library-managed scratchpad is used, the library allocates a temporary buffer at execution time and frees it immediately after execution completes. This is incompatible with SYCL graph recording.

To work correctly with SYCL graph recording mode, users must pass the pre-allocated scratchpad tensor to the execute call during recording and keep the scratchpad buffer alive across all replay iterations.

API Reference
~~~~~~~~~~~~~

===========================================================================================================================================================  ==============================================================  
API                                                                                                                                                          Description                                                     
===========================================================================================================================================================  ==============================================================  
:ref:`dnnl::graph::compiled_partition::get_scratchpad_logical_tensor <doxid-classdnnl_1_1graph_1_1compiled__partition_1a21ec243a10b40b495f0d9a84c9f78fd1>`   Returns the logical tensor describing the required scratchpad   
:ref:`dnnl::graph::compiled_partition::execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1aa93ee468a74520638dc87212415b1ea6>`                         Executes with optional user-managed scratchpad                  
:ref:`dnnl::graph::sycl_interop::execute <doxid-namespacednnl_1_1graph_1_1sycl__interop_1acc5ff56ff0f276367b047c3c73093a67>`                                 SYCL interop execute with user-managed scratchpad               
:ref:`dnnl::graph::ocl_interop::execute <doxid-namespacednnl_1_1graph_1_1ocl__interop_1a8b1d57febf09dc0621d7aa2a8dc13035>`                                   OpenCL interop execute with user-managed scratchpad             
===========================================================================================================================================================  ==============================================================

