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

	onednn_verbose,info,oneDNN v3.1.0 (commit 0091797c30f16250dec7a40f9ee1a8a33bcfd65e)
	onednn_verbose,info,cpu,runtime:OpenMP,nthr:56
	onednn_verbose,info,cpu,isa:Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
	onednn_verbose,info,gpu,runtime:none
	onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,create:check,matmul,dimension src:1 is inconsistent with weights:0,src/common/matmul.cpp:69

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

	onednn_verbose,info,oneDNN v3.1.0 (commit 0091797c30f16250dec7a40f9ee1a8a33bcfd65e)
	onednn_verbose,info,cpu,runtime:OpenMP,nthr:56
	onednn_verbose,info,cpu,isa:Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
	onednn_verbose,info,gpu,runtime:none
	onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx512_core_amx_fp16,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx512_core_amx,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
	onednn_verbose,create:dispatch,brgemm_matmul,datatype configuration not supported on this isa,src/cpu/x64/matmul/brgemm_matmul_utils.cpp:931
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx512_core_bf16,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx512_core_vnni,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx2_vnni_2,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
	onednn_verbose,create:dispatch,matmul,cpu,matmul,brg:avx2_vnni,undef,src_u8::any::f0 wei_s8::any::f0 dst_f32::any::f0,,,256x256:256x256:256x256,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101

Above, we can see that the highest performance implementations were not dispatched either because they required a higher ISA, or because they did not support that datatype configuration.

Enable ONEDNN_VERBOSE with timestamps
-------------------------------------

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=profile ONEDNN_VERBOSE_TIMESTAMP=1 ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"

This produces the following output:

.. ref-code-block:: cpp

	onednn_verbose,info,oneDNN v3.1.0 (commit 2cdd9ee1364b6c5b107aff8738af352a746d0434)
	onednn_verbose,info,cpu,runtime:OpenMP,nthr:56
	onednn_verbose,info,cpu,isa:Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
	onednn_verbose,info,gpu,runtime:none
	onednn_verbose,info,prim_template:timestamp,operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	onednn_verbose,1681823859527.679932,create:cache_miss,cpu,convolution,jit:avx512_core,forward_training,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,13.313
	onednn_verbose,1681823859541.047119,create:cache_hit,cpu,convolution,jit:avx512_core,forward_training,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.00292969
	onednn_verbose,1681823859567.496094,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,,,16,0.0759277
	onednn_verbose,1681823859567.612061,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,,,16,0.00195312
	onednn_verbose,1681823859567.902100,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,16x16x5x5,0.0720215
	onednn_verbose,1681823859567.996094,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,16x16x5x5,0.201904
	onednn_verbose,1681823859568.535889,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd16b:f0,,,2x16x7x7,0.0432129
	onednn_verbose,1681823859568.597900,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd16b:f0,,,2x16x7x7,0.258057
	onednn_verbose,1681823859568.868896,exec,cpu,convolution,jit:avx512_core,forward_training,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,40.6201
	onednn_verbose,1681823859610.262939,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,2x16x7x7,0.052002
	onednn_verbose,1681823859610.383057,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,2x16x7x7,0.189941

Decrypting the Output
~~~~~~~~~~~~~~~~~~~~~

The first lines of verbose information, which are denoted with ``info``, contain the build version and git hash, if available, as well as CPU and GPU runtimes, the supported instruction set architecture and the verbose output format template since amount of fields may vary depeding on set of enviroment variables enabled. This verbose header is printed upon first logged information.

Each subsequent line of verbose information is formatted as a comma-separated list contains, in order of appearance in the line from left to right:

* ``onednn_verbose`` marker string

* if ``ONEDNN_VERBOSE_TIMESTAMP=1`` is specified, start time of the call. On Linux this number represents amount of milliseconds since Unix epoch. On Windows this number represents amount of milliseconds since the last system start.

* operation: ``exec|create:<cache_hit|cache_miss|from_cache_blob>`` for profiling information, ``error|check|dispatch`` for other information.

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

