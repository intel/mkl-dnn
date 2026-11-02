.. index:: pair: page; Verbose Mode
.. _doxid-dev_guide_verbose:

Verbose Mode
============

oneDNN verbose mode enables tracing the execution of oneDNN API calls. This is a useful feature for collecting statistics to profile an application or for troubleshooting API usage errors. When verbose mode is enabled oneDNN will print out information to ``stdout``.

Build-Time Controls
~~~~~~~~~~~~~~~~~~~

At build-time, support for this feature is controlled via cmake option ``ONEDNN_VERBOSE``.

===============  ====================================  ======================================================  
CMake Option     Supported values (defaults in bold)   Description                                             
===============  ====================================  ======================================================  
ONEDNN_VERBOSE   **ON** , OFF                          Enables :ref:`verbose mode <doxid-dev_guide_verbose>`   
===============  ====================================  ======================================================

Run-Time Controls
~~~~~~~~~~~~~~~~~

When the feature is enabled at build-time, the ``ONEDNN_VERBOSE`` environment variable can be used to turn verbose mode on and control the type of tracing information to display.

=============================  ======================  ==================================================  
Environment variable           Value                   Description                                         
=============================  ======================  ==================================================  
``ONEDNN_VERBOSE``             ``none``                no messages printed                                 
\                              ** ``error`` **         **error messages** (default)                        
\                              ``warn``                warning messages                                    
\                              ``check``               primitive creation parameter checking information   
\                              ``profile_create``      primitive creation timings                          
\                              ``profile_exec``        primitive execution timings                         
\                              ``profile``             primitive creation and execution timings            
\                              ``dispatch``            primitive dispatching information                   
\                              ``all``                 enables all above flags but ``none``                
\                              ``debuginfo=<level>``   enables internal debug printing (for developers)    
``ONEDNN_VERBOSE_TIMESTAMP``   **0**                   **display timestamps disabled (default)**           
\                              1                       display timestamps enabled                          
=============================  ======================  ==================================================

The verbose flags can be combined, e.g. ``ONEDNN_VERBOSE=profile,dispatch`` will enable printing both performance profiling information, and information relative to why a given oneDNN primitive implementation was dispatched. In general, we recommend using ``ONEDNN_VERBOSE=all``, unless message printing overhead becomes noticeable.

``debuginfo`` information is available only if the library is built with ``ONEDNN_DEV_MODE=ON``.

oneDNN verbose also provides a ``filter`` option, which takes a regular expression and applies the verbose output to matching components. Currently, the supported components are ``primitive``, ``graph``, ``gemm_api`` and primitive kind names. Here are some examples of usage:

* ``ONEDNN_VERBOSE=profile_exec,filter=graph`` will print verbose of compiled_partition execution profiling from graph API

* ``ONEDNN_VERBOSE=profile_exec,filter=prim`` will print verbose of primitive execution profiling from primitive API

* ``ONEDNN_VERBOSE=profile_exec,filter=conv\|matmul`` will print execution profiling verbose of (de)convolution and matmul primitive

* Filter won't work if the regular expression is invalid

* Only the last one will take effect if multiple filters are specified

oneDNN supports the following legacy settings:

=====================  ======  ====================================================================  
Environment variable   Value   Description                                                           
=====================  ======  ====================================================================  
ONEDNN_VERBOSE         0       no verbose output, replaced by ``none``                               
\                      1       primitive execution profiling timings, replaced by ``profile_exec``   
\                      2       primitive creation and execution timings, replaced by ``profile``     
=====================  ======  ====================================================================

The oneDNN verbose can also be managed at run-time with the following functions:

* :ref:`dnnl_set_verbose <doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856>`

The function setting takes precedence over the environment variable.

Example
~~~~~~~

Troubleshooting primitive creation issues
-----------------------------------------

When facing functional issues, we recommend using ``ONEDNN_VERBOSE=all`` as it will provide insights on why a given primitive cannot be created. Here is an example of output one can get when providing incorrect dimensions to a matmul primitive.

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=all ./benchdnn --matmul 256x256:25x256

This produces the following output:

.. ref-code-block:: cpp

	onednn_verbose,v0,info,oneDNN v3.2.0 (commit 6afab8e57f65a8995685d97ba6f80fa6c24b87a0)
	onednn_verbose,v0,info,cpu,runtime:OpenMP,nthr:128
	onednn_verbose,v0,info,cpu,isa:Intel AVX-512 with Intel DL Boost
	onednn_verbose,v0,info,gpu,runtime:none
	onednn_verbose,v0,info,graph,backend,0:dnnl_backend
	onednn_verbose,v0,primitive,info,template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,v0,graph,info,template:operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
	onednn_verbose,v0,primitive,create:check,matmul,dimension src:1 is inconsistent with weights:0,src/common/matmul.cpp:144

The last line here shows that the matmul primitive failed to be created because of a dimension mismatch between its two operands.

Profiling a workload
--------------------

To understand a full application performance, it is useful to break down performance bottlenecks. ``ONEDNN_VERBOSE=profile`` does just that and shows

* how much time is spent in primitive creation

* how much time is spent in each primitive execution

* how often a given primitive is called.

Please see the profiling example :ref:`here <doxid-performance_profiling_cpp>`, as it uses ONEDNN_VERBOSE output to tune oneDNN code to align with :ref:`best practices <doxid-dev_guide_inference>`.

Understanding why a given implementation is dispatched
------------------------------------------------------

When performance is lower than expected, it is usually likely due to the dispatching of a lower performing implementation. Hence it can be useful to understand what circumstance led oneDNN to dispatch a lower performance implementation. This can be observed by using ``ONEDNN_VERBOSE=dispatch``.

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=dispatch ./benchdnn --matmul --dt=u8:s8:f32 256x256:256x256

This produces the following log (shortened for brevity).

.. ref-code-block:: cpp

	onednn_verbose,v0,info,oneDNN v3.2.0 (commit 6afab8e57f65a8995685d97ba6f80fa6c24b87a0)
	onednn_verbose,v0,info,cpu,runtime:OpenMP,nthr:128
	onednn_verbose,v0,info,cpu,isa:Intel AVX-512 with Intel DL Boost
	onednn_verbose,v0,info,gpu,runtime:none
	onednn_verbose,v0,info,graph,backend,0:dnnl_backend
	onednn_verbose,v0,primitive,info,template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,v0,graph,info,template:operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
	onednn_verbose,v0,primitive,create:dispatch,matmul,cpu,matmul,brg:avx512_core_amx_fp16,undef,src_u8:a:any:any::f0 wei_s8:a:any:any::f0 dst_f32:a:any:any::f0,,,256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:97
	onednn_verbose,v0,primitive,create:dispatch,matmul,cpu,matmul,brg:avx512_core_amx,undef,src_u8:a:any:any::f0 wei_s8:a:any:any::f0 dst_f32:a:any:any::f0,,,256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:97
	onednn_verbose,v0,primitive,create:dispatch,matmul,cpu,matmul,brg:avx512_core_fp16,undef,src_u8:a:any:any::f0 wei_s8:a:any:any::f0 dst_f32:a:any:any::f0,,,256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:97
	onednn_verbose,v0,primitive,create:dispatch,matmul,cpu,matmul,brg:avx512_core_bf16,undef,src_u8:a:any:any::f0 wei_s8:a:any:any::f0 dst_f32:a:any:any::f0,,,256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:97

Above, we can see that the highest performance implementations were not dispatched either because they required a higher ISA, or because they did not support that datatype configuration. A complete list of verbose messages encountered in the dispatch mode can be found `here <https://oneapi-src.github.io/oneDNN/dev_guide_verbose_table.html>`__ along with their explanation.

Enable ONEDNN_VERBOSE with timestamps
-------------------------------------

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=profile ONEDNN_VERBOSE_TIMESTAMP=1 ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"

This produces the following output:

.. ref-code-block:: cpp

	onednn_verbose,v0,info,oneDNN v3.2.0 (commit 6afab8e57f65a8995685d97ba6f80fa6c24b87a0)
	onednn_verbose,v0,info,cpu,runtime:OpenMP,nthr:128
	onednn_verbose,v0,info,cpu,isa:Intel AVX-512 with Intel DL Boost
	onednn_verbose,v0,info,gpu,runtime:none
	onednn_verbose,v0,info,graph,backend,0:dnnl_backend
	onednn_verbose,v0,primitive,info,template:timestamp,operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,v0,graph,info,template:timestamp,operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
	onednn_verbose,v0,1693533460193.346924,primitive,create:cache_miss,cpu,convolution,jit:avx512_core,forward_training,src_f32:a:blocked:aBcd16b::f0 wei_f32:a:blocked:ABcd16b16a::f0 bia_f32:a:blocked:a::f0 dst_f32:a:blocked:aBcd16b::f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.709961
	onednn_verbose,v0,1693533460194.199951,primitive,create:cache_hit,cpu,convolution,jit:avx512_core,forward_training,src_f32:a:blocked:aBcd16b::f0 wei_f32:a:blocked:ABcd16b16a::f0 bia_f32:a:blocked:a::f0 dst_f32:a:blocked:aBcd16b::f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0161133
	onednn_verbose,v0,1693533460228.559082,primitive,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd::f0 dst_f32::blocked:ABcd16b16a::f0,,,16x16x5x5,0.724854
	onednn_verbose,v0,1693533460229.437012,primitive,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd::f0 dst_f32::blocked:ABcd16b16a::f0,,,16x16x5x5,16.481
	onednn_verbose,v0,1693533460259.165039,primitive,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd::f0 dst_f32::blocked:aBcd16b::f0,,,2x16x7x7,0.349854
	onednn_verbose,v0,1693533460259.586914,primitive,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd::f0 dst_f32::blocked:aBcd16b::f0,,,2x16x7x7,12.604
	onednn_verbose,v0,1693533460272.332031,primitive,create:cache_miss,cpu,reorder,simple:any,undef,src_f32::blocked:a::f0 dst_f32::blocked:a::f0,,,16,0.0358887
	onednn_verbose,v0,1693533460272.416992,primitive,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a::f0 dst_f32::blocked:a::f0,,,16,0.052002
	onednn_verbose,v0,1693533460272.561035,primitive,exec,cpu,convolution,jit:avx512_core,forward_training,src_f32:a:blocked:aBcd16b::f0 wei_f32:a:blocked:ABcd16b16a::f0 bia_f32:a:blocked:a::f0 dst_f32:a:blocked:aBcd16b::f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0878906
	onednn_verbose,v0,1693533460313.719971,primitive,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b::f0 dst_f32::blocked:abcd::f0,,,2x16x7x7,0.275146
	onednn_verbose,v0,1693533460314.072021,primitive,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b::f0 dst_f32::blocked:abcd::f0,,,2x16x7x7,18.8389
	0:PASSED __REPRO: --conv ic16ih7oc16oh7kh5ph2nwip

Decrypting the Output
~~~~~~~~~~~~~~~~~~~~~

The first lines of verbose information, which are denoted with ``info``, contain the build version and git hash, if available, as well as CPU and GPU runtimes. It also includes graph API backends, the supported instruction set architecture, and the verbose output format template since the amount of fields may vary depending on the set of enabled environment variables. This verbose header is printed when information is first logged.

Each subsequent line of primitive verbose information is formatted as a comma-separated list and contains the following, in order of appearance in the line from left to right:

* ``onednn_verbose`` marker string

* verbose mode version: ``v0`` or ``v1``

* if ``ONEDNN_VERBOSE_TIMESTAMP=1`` is specified, start time of the call. On Linux this number represents amount of milliseconds since Unix epoch. On Windows this number represents amount of milliseconds since the last system start.

* API kind: ``primitive|graph|common`` for API information

* operation: ``exec|create:<cache_hit|cache_miss|kernel_cache_hit|persistent_cache_hit|nested_cache_hit>`` for profiling information, ``error|check|dispatch`` for other information.

* engine kind: ``cpu`` or ``gpu`` (``cpu2gpu`` or ``gpu2cpu`` for cross-engine reorder)

* primitive name: ``convolution``, ``reorder``, ``sum``, etc

* primitive implementation

* propagation kind: ``forward_training``, ``forward_inference``, ``backward``, etc

* information about all operation tensors (separated by space)

* primitive attributes

* auxiliary information like algorithm name or number of inputs

* a problem description in :ref:`benchdnn format <doxid-dev_guide_benchdnn>`

* execution time in milliseconds

The information about a particular operation tensors has the following format: ``tensor_name`` \_ ``data_type`` : ``properties`` : ``format_kind`` : ``format_tag`` : ``strides`` : ``extra_flags``, where:

#. ``tensor_name`` is one of the tensors names listed in the :ref:`Naming Conventions <doxid-dev_guide_conventions>`, and denotes a tensor supported by the corresponding primitive.

#. ``properties`` denotes if a tensor was created with ``format_kind::any`` and has padded area or an offset from original memory.

#. ``data_type``, ``format_kind`` and ``format_tag`` denote values from :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`, :ref:`dnnl::memory::format_kind <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>` and :ref:`dnnl::memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` respectively. Note that certain markers may be missing in some cases, such as ``format_tag`` for the :math:`\weights` tensor for the Winograd convolution.

#. ``strides`` denotes stride values in case the memory is not dense. If the memory is dense, the field will be empty.

#. ``extra_flags`` is unspecified information that is intended for development purposes.

.. note:: 

   When oneDNN verbose mode is enabled with GPU engines, oneDNN adds extra stream synchronization on entry and on exit in the :ref:`dnnl::primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call. The execution time is calculated based on wall time measured before and after primitive execution.
   
   

.. note:: 

   When oneDNN verbose mode is enabled for builds with `Compute Library for the Arm architecture <https://oneapi-src.github.io/oneDNN/dev_guide_build.html#gcc-with-arm-compute-library-acl-on-aarch64-host>`__, any failures in the validation of Compute Library primitives will be detailed in the verbose output.
   
   

.. warning:: 

   Verbose mode has non-negligible performance impact especially on GPU or if the output rate is high.

