.. index:: pair: page; Experimental features
.. _doxid-dev_guide_experimental:

Experimental features
=====================

To test aggressive performance optimizations that might affect accuracy or new API and functionality without an impact to regular users, oneDNN provides experimental features.

Build-time Controls
~~~~~~~~~~~~~~~~~~~

There are two kinds of experimental features:

#. Features that can be enabled at runtime with an environment variable. To enable such experimental features, the library should be built with a CMake option ``ONEDNN_EXPERIMENTAL=ON``. Each experimental feature has to be individually selected using environment variables.

#. Features that can be enabled only with a build time option. To enable such experimental features, the library should be built with a CMake option that corresponds to a particular feature.

Both kinds of experimental features can be enabled simultaneously.

Experimental features
~~~~~~~~~~~~~~~~~~~~~

=========================================  =======================================================================================================================================================================  
Environment variable                       Description                                                                                                                                                              
=========================================  =======================================================================================================================================================================  
ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS   Calculate mean and variance in batch normalization(BN) in single pass ( `RFC <https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm>`__ ).   
ONEDNN_EXPERIMENTAL_GPU_CONV_V2            Enable shapeless GPU convolution implementation (the feature is under development).                                                                                      
=========================================  =======================================================================================================================================================================

=========================================  =============================================================  
Build time option                          Description                                                    
=========================================  =============================================================  
ONEDNN_EXPERIMENTAL_UKERNEL                Enable experimental microkernel APIs and functionalities.      
ONEDNN_EXPERIMENTAL_PROFILING              Enable experimental profiling API.                             
ONEDNN_EXPERIMENTAL_LOGGING                Enable experimental logging support for oneDNN verbose mode.   
ONEDNN_EXPERIMENTAL_SYCL_KERNEL_COMPILER   Enable SYCL OpenCL online kernel compiler extension.           
=========================================  =============================================================

Features details
~~~~~~~~~~~~~~~~

ONEDNN_EXPERIMENTAL_UKERNEL
---------------------------

This option enables a new set of CPU-only APIs to support block-level functionalities. By composing these low-level, sequential operations, users can implement their own custom operations/fusions, and tailor blocking/threading logic to their applications.

More details on this API are available in the :ref:`Microkernel APIs <doxid-dev_guide_ukernel_basic_concepts>` section".

ONEDNN_EXPERIMENTAL_PROFILING
-----------------------------

This option enables profiling API that can be used to query different profiling data.

There are two ways to use the profiling capabilities:

* Create a queue with enabled profiling capabilities and use the interoperability API to create a oneDNN stream with the queue. The library will identify that the queue supports profiling and will collect profiling data

* Create a oneDNN stream using runtime agnostic API and enable profiling capabilities using the stream flag ``stream::flags::profiling``

Below is a pseudo-code that demonstrates the profiling API usage with a user-provided queue.

.. ref-code-block:: cpp

	:ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(engine::kind::gpu, 0);
	// Create a queue with enabled profiling mode.
	cl_command_queue ocl_queue {};
	cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	ocl_queue = clCreateCommandQueueWithProperties(ocl_interop::get_context(engine),
	    ocl_interop::get_device(engine), props, ...);
	// Create dnnl::stream with the queue.
	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` stream = ocl_interop::make_stream(engine, ocl_queue);
	// Create a convolution primitive ... //
	// Reset profiler's state.
	:ref:`dnnl::reset_profiling <doxid-group__dnnl__api__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(stream);
	// Enqueue same primitive twice and wait for both executions to complete.
	conv_prim.execute(stream, ...)
	conv_prim.execute(stream, ...)
	stream.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	// Query profiling data. The vector size will be equal to the number of
	// executions happened on the stream since the last `dnnl::reset_profiling`
	// call.
	std::vector<uint64_t> nsecs = :ref:`dnnl::get_profiling_data <doxid-group__dnnl__api__profiling_1ga0dc451b94cbeacb7a5e0c73c3071ee4e>`(stream, profiling_data_kind::time);
	assert(nsecs.size() == 2);
	// Reset profiler's state.
	:ref:`dnnl::reset_profiling <doxid-group__dnnl__api__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(stream);

.. warning:: 

   * When the stream is created with enabled profiling capabilities it will collect profiling data for each primitive execution. It is the user's responsibility to reset the profiler's state to avoid consuming all memory resources in the system.
   
   


Limitations
+++++++++++

* Only GPU engines with OpenCL and SYCL runtimes are supported

* Only Intel vendor is supported for SYCL runtime

* Out-of-order queue is not supported

.. warning:: 

   * Enabling some experimental features does not guarantee that the library will utilize them
   
   * Enabling some experimental features might change the accuracy of oneDNN primitives
   
   


ONEDNN_EXPERIMENTAL_LOGGING
---------------------------

This option introduces logging support in oneDNN which allows one to save the verbose outputs generated by oneDNN applications to user-specified logfiles. By setting ``ONEDNN_EXPERIMENTAL_LOGGING=ON``, a logging mechanism is built into oneDNN using the third-party `spdlog <https://github.com/gabime/spdlog>`__ library. Logging can then be enabled while running different applications by specifying the logfile path using ``ONEDNN_VERBOSE_LOGFILE`` :

.. ref-code-block:: cpp

	$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=./logs/cnn_test_logger.log ./examples/cnn-inference-f32-cpp

When logging is enabled while running an application, it also requires that the verbose mode be enabled for the run using ``ONEDNN_VERBOSE``. When no logfile is specified, logging is automatically disabled and the verbose output is printed only to the console. For the specified logfile path, the logger creates the base directory and the logfile if they do not already exist. When the specified logfile already exists, the output is appended to the existing file until it reaches the maximum file size. Note: Multiple instances using the same filepath for ``DNNL_VERBOSE_LOGFILE`` will write to the same file during the API run. The spdlog mechanism supports handling multiple instances concurrently if they write to the same logfile but the expectation is to specify different logfiles for different instances via the runtime variables.

By default, logging is disabled in oneDNN and any verbose output generated by oneDNN is printed only to ``stdout``. The API is executed as a rotating lazy logger with a file size specified by ``ONEDNN_VERBOSE_LOGFILE_SIZE(=1024*1024*50)``. When logging is enabled, the user has the option to print verbose output to both ``stdout`` and the logfile by setting ``ONEDNN_VERBOSE_LOG_WITH_CONSOLE=1``. The runtime controls for oneDNN logging are listed as follows:

================================  ====================================================  
Runtime variable                  Description                                           
================================  ====================================================  
ONEDNN_VERBOSE_LOGFILE            Enables verbose logging and specifies logfile path.   
ONEDNN_VERBOSE_LOGFILE_SIZE       Specifies maximum size for the logfile.               
ONEDNN_VERBOSE_NUM_LOGFILES       Number of rotating logfiles for the logger.           
ONEDNN_VERBOSE_LOG_WITH_CONSOLE   Enables printing to both stdout and the logfile.      
================================  ====================================================

ONEDNN_EXPERIMENTAL_SYCL_KERNEL_COMPILER
----------------------------------------

This option enables the experimental SYCL OpenCL online kernel compiler, allowing OpenCL kernels to be compiled without directly invoking the OpenCL runtime.

