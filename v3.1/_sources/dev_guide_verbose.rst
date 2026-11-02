.. index:: pair: page; Verbose Mode
.. _doxid-dev_guide_verbose:

Verbose Mode
============

It is often useful to collect information about how much of an application runtime is spent executing oneDNN primitives and which of those take the most time. oneDNN verbose mode enables tracing execution of oneDNN primitives and collection of basic statistics like execution time and primitive parameters. When verbose mode is enabled oneDNN will print out information to ``stdout``.

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

When the feature is enabled at build-time, the ``ONEDNN_VERBOSE`` environment variable can be used to turn verbose mode on and control the level of verbosity.

=========================  ======  ================================================  
Environment variable       Value   Description                                       
=========================  ======  ================================================  
ONEDNN_VERBOSE             **0**   **no verbose output (default)**                   
                           1       primitive information at execution                
                           2       primitive information at creation and execution   
ONEDNN_VERBOSE_TIMESTAMP   **0**   **display timestamps disabled (default)**         
                           1       display timestamps enabled                        
=========================  ======  ================================================

This feature can also be managed at run-time with the following functions:

* :ref:`dnnl_set_verbose <doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856>`

The function setting takes precedence over the environment variable.

Example
~~~~~~~

Enable ONEDNN_VERBOSE
---------------------

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=1 ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"

This produces the following output (the line breaks were added to fit the page width):

.. ref-code-block:: cpp

	dnnl_verbose,info,DNNL v1.3.0 (commit d0fc158e98590dfad0165e568ca466876a794597)
	dnnl_verbose,info,cpu,runtime:OpenMP
	dnnl_verbose,info,cpu,isa:Intel AVX2
	dnnl_verbose,info,gpu,runtime:none
	dnnl_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,0.0200195
	dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,16x16x5x5,0.0251465
	dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,0.0180664
	dnnl_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,,,16,0.0229492
	dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_training,src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0390625
	dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,2x16x7x7,0.173096

Enable ONEDNN_VERBOSE with timestamps
-------------------------------------

.. ref-code-block:: cpp

	ONEDNN_VERBOSE=1 ONEDNN_VERBOSE_TIMESTAMP=1 ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"

This produces the following output (the line breaks were added to fit the page width):

.. ref-code-block:: cpp

	dnnl_verbose,info,oneDNN v2.0.0 (commit N/A)
	dnnl_verbose,info,cpu,runtime:OpenMP
	dnnl_verbose,info,cpu,isa:Intel AVX2
	dnnl_verbose,info,gpu,runtime:none
	dnnl_verbose,info,prim_template:timestamp,operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
	dnnl_verbose,1607393146348.667969,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,3.58594
	dnnl_verbose,1607393146356.743896,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,16x16x5x5,3.63916
	dnnl_verbose,1607393146364.541992,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,2x16x7x7,2.35693
	dnnl_verbose,1607393146367.198975,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,,,16,3.71191
	dnnl_verbose,1607393146371.002930,exec,cpu,convolution,jit:avx2,forward_training,src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,3.93018
	dnnl_verbose,1607393146380.145020,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,2x16x7x7,1.75708

Decrypting the Output
~~~~~~~~~~~~~~~~~~~~~

The first lines of verbose information, which are denoted with ``info``, contain the build version and git hash, if available, as well as CPU and GPU runtimes, the supported instruction set architecture and the verbose output format template since amount of fields may vary depeding on set of enviroment variables enabled.

Each subsequent line of verbose information is formatted as a comma-separated list contains, in order of appearance in the line from left to right:

* ``dnnl_verbose`` marker string

* if ``ONEDNN_VERBOSE_TIMESTAMP=1`` is specified, start time of the call. On Linux this number represents amount of milliseconds since Unix epoch. On Windows this number represents amount of milliseconds since the last system start.

* operation: ``create:<cache_hit|cache_miss|from_cache_blob>`` or ``exec``

* engine kind: ``cpu`` or ``gpu`` (``cpu2gpu`` or ``gpu2cpu`` for cross-engine reorder)

* primitive name: ``convolution``, ``reorder``, ``sum``, etc

* primitive implementation

* propagation kind: ``forward_training``, ``forward_inference``, ``backward``, etc

* information about all operation tensors (separated by space)

* primitive attributes

* auxiliary information like algorithm name or number of inputs

* a problem description in :ref:`benchdnn format <doxid-dev_guide_benchdnn>`

* execution time in milliseconds

The information about a particular operation tensors has the following format: ``tensor_name`` \_ ``data_type`` :: ``format_kind`` : ``format_tag`` : ``extra_flags``, where:

#. ``tensor_name`` is one of the tensors names listed in the :ref:`Naming Conventions <doxid-dev_guide_conventions>`, and denotes a tensor supported by the corresponding primitive

#. ``data_type``, ``format_kind`` and ``format_tag`` denote values from :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`, :ref:`dnnl::memory::format_kind <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>` and :ref:`dnnl::memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` respectively. Note, that certain markers may be missing in some cases, such as ``format_tag`` for the :math:`\weights` tensor for the Winograd convolution.

#. ``extra_flags`` is unspecified information that is intended for development purposes

Please see the profiling example :ref:`here <doxid-performance_profiling_cpp>`, as it uses ONEDNN_VERBOSE output to tune oneDNN code to align with :ref:`best practices <doxid-dev_guide_inference>`.

.. note:: 

   When oneDNN verbose mode is enabled with GPU engines, oneDNN adds extra stream synchronization on entry and on exit in the :ref:`dnnl::primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call. The execution time is calculated based on wall time measured before and after primitive execution.
   
   

.. note:: 

   When oneDNN verbose mode is enabled for builds with `Compute Library for the Arm architecture <https://oneapi-src.github.io/oneDNN/dev_guide_build.html#gcc-with-arm-compute-library-acl-on-aarch64-host>`__, any failures in the validation of Compute Library primitives will be detailed in the verbose output.
   
   

.. warning:: 

   Verbose mode has non-negligible performance impact especially on GPU or if the output rate is high.

