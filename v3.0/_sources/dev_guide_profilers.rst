.. index:: pair: page; Profiling oneDNN Performance
.. _doxid-dev_guide_profilers:

Profiling oneDNN Performance
============================

oneDNN uses JIT (just-in-time) code generation based on the primitive parameters and instruction set supported by the system. In order to correctly attribute performance event information, profilers must be notified about address ranges containing JIT-ed code. oneDNN supports two profilers: VTune(TM) Amplifier and Linux perf.

Build-Time Controls
~~~~~~~~~~~~~~~~~~~

At build-time, support for this feature is controlled by the CMake option ``ONEDNN_ENABLE_JIT_PROFILING``.

============================  ======================  ==========================================  
CMake Option                  Supported Values        Description                                 
============================  ======================  ==========================================  
ONEDNN_ENABLE_JIT_PROFILING   **ON** (default), OFF   Enables performance profilers integration   
============================  ======================  ==========================================

Run-Time Controls
~~~~~~~~~~~~~~~~~

When the feature is enabled at build-time, the ``ONEDNN_JIT_PROFILE`` environment variable can be used to manage integration with performance profilers.

=====================  =======================================================================  ====================================  ================  =====  
Environment Variable   Value                                                                    Description                           x64               AArc   
=====================  =======================================================================  ====================================  ================  =====  
ONEDNN_JIT_PROFILE     1                                                                        Enables VTune Amplifier integration   **x** (default)   N/A    
2                      Enables basic Linux perf integration                                     x                                     **x** (default)   
6                      Enables Linux perf integration with JIT dump output                      x                                     x                 
14                     Enables Linux perf integration with JIT dump output and TSC timestamps   x                                     N/A               
=====================  =======================================================================  ====================================  ================  =====

Other valid values for ``ONEDNN_JIT_PROFILE`` include integer values representing a combination of flags accepted by the :ref:`dnnl_set_jit_profiling_flags <doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48>` function.

The default setting of the profiling flags is to enable integration with VTune Amplifier; therefore it does not require any additional setup and works out of the box. Code integrating oneDNN may override this behavior.

This feature can also be managed at run-time with the following functions:

* :ref:`dnnl_set_jit_profiling_flags <doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48>`

* :ref:`dnnl_set_jit_profiling_jitdumpdir <doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7>`

Function settings take precedence over environment variables.

Features for VTune Amplifier
----------------------------

ITT Tagging for Primitive Execution
+++++++++++++++++++++++++++++++++++

oneDNN supports ITT tagging at primitive execution in order to provide performance information on the level of a oneDNN primitive. This feature is supported on both CPU and GPU.

ITT tagging in oneDNN during primitive execution provides more information from VTune Amplifier for the items below.

#. Get the primitives timeline chart from VTune Amplifier, and identify potential performance issues.

#. Get platform information such as an L1/L2 cache miss or level of FP vectorization on the primitive level.

#. Map primitive with related computation kernels.

Build-Time Controls
*******************

At build-time, support for this feature is controlled by the CMake option ``ONEDNN_ENABLE_ITT_TASKS``.

========================  ======================  ============================================  
CMake Option              Supported Values        Desc                                          
========================  ======================  ============================================  
ONEDNN_ENABLE_ITT_TASKS   **ON** (default), OFF   Enables ITT tagging for primitive execution   
========================  ======================  ============================================

Run-Time Controls
*****************

When the feature is enabled at build-time, the ``ONEDNN_ITT_TASK_LEVEL`` environment variable can be used to enable different level of ITT tagging.

======================  ====================================================  ===============================  
Environment Variable    Value                                                 Desc                             
======================  ====================================================  ===============================  
ONEDNN_ITT_TASK_LEVEL   0                                                     no ITT event will be triggered   
1                       ITT events are only triggered in master thread        
**2** (default)         **ITT events are triggered in all OMP/TBB threads**   
======================  ====================================================  ===============================

Example: Profiling with VTune Amplifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this section, it is assumed that the performance profiling environment is already set up.

Profiling for Hotspots
----------------------

Collect profiling data:

.. ref-code-block:: cpp

	$ amplxe-cl -collect hotspots -q -no-summary -knob sampling-mode=hw -r dnnl-vtune ./benchdnn --mode=P --conv --batch=inputs/conv/shapes_alexnet
	amplxe: Warning: To enable hardware event-base sampling, VTune Amplifier has disabled the NMI watchdog timer.  
	The watchdog timer will be re-enabled after collection completes.
	Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
	perf,cpu,jit:avx512_common,"alexnet:conv1",--conv g1mb256ic3ih227oc96oh55kh11sh4ph0n"alexnet:conv1",53.9726,0,17.4285,3096.81,22.5851,2389.74
	perf,cpu,jit:avx512_common,"alexnet:conv2",--conv g2mb256ic96ih27oc256oh27kh5ph2n"alexnet:conv2",104.696,0,20.2195,5177.98,21.9233,4775.56
	perf,cpu,jit:avx512_common,"alexnet:conv3",--conv mb256ic256ih13oc384oh13kh3ph1n"alexnet:conv3",68.904,0,15.5134,4441.57,18.1391,3798.64
	perf,cpu,jit:avx512_common,"alexnet:conv4",--conv g2mb256ic384ih13oc384oh13kh3ph1n"alexnet:conv4",51.678,0,11.7397,4401.97,12.4623,4146.76
	perf,cpu,jit:avx512_common,"alexnet:conv5",--conv g2mb256ic384ih13oc256oh13kh3ph1n"alexnet:conv5",34.452,0,7.77148,4433.13,8.50435,4051.11
	tests:5 passed:5 skipped:0 mistrusted:0 unimplemented:0 failed:0 listed:0
	total perf: min(ms):72.6726 avg(ms):83.6142

.. note:: 

   Here, it is not necessary to set the ``ONEDNN_JIT_PROFILE`` environment variable.
   
   
Below are the top 10 function hotspots using the command-line interface:

.. ref-code-block:: cpp

	$ amplxe-cl -report hotspots -q -r dnnl-vtune -format csv -csv-delimiter ';' -group-by process,module,function -column 'CPU Time:Self' | head -n 10 | column -t -s';'
	Column filter is ON.
	Process   Module            Function                               CPU Time
	benchdnn  [Dynamic code]    _jit_avx512_common_conv_fwd_kernel     300.128503
	benchdnn  [Dynamic code]    _jit_avx512_common_conv_fwd_kernel     293.946143
	benchdnn  [Dynamic code]    _jit_avx512_common_conv_fwd_kernel     285.549830
	benchdnn  [Dynamic code]    _jit_avx512_common_conv_fwd_kernel     268.868599
	benchdnn  [Dynamic code]    _jit_avx512_common_conv_fwd_kernel     256.715527
	benchdnn  libgomp.so.1.0.0  func@0x194f0                           186.604226
	benchdnn  libgomp.so.1.0.0  func@0x19370                           82.609694
	benchdnn  libdnnl.so.1.8    dnnl::impl::cpu::x64::jit_avx512_co..  35.682241
	benchdnn  vmlinux           [vmlinux]
	       10.763433

The JIT-ed function ``_jit_avx512_common_conv_fwd_kernel`` is shown as belonging to the ``[Dynamic code]`` module.

Below are the top 10 primitive type hotspots using the command-line interface:

.. ref-code-block:: cpp

	$ amplxe-cl -report hotspots -q -r dnnl-vtune -format csv -csv-delimiter ';' -group-by task -column 'CPU Time:Self' | head -n 10 | column -t -s';'
	Column filter is ON.
	Task Type           CPU Time
	convolution         1451.459338
	[Outside any task]  280.489764
	reorder             10.434821 
	       10.763433

Profiling for Microarchitecture Information
-------------------------------------------

Collect profiling data:

.. ref-code-block:: cpp

	$ amplxe-cl -collect uarch-exploration -knob sampling-interval=1 -data-limit=2000  -q -no-summary -r dnnl-vtune-ue ./benchdnn --mode=P --conv --batch=inputs/conv/shapes_alexnet 
	amplxe: Warning: To enable hardware event-base sampling, VTune Amplifier has disabled the NMI watchdog timer. The watchdog timer will be re-enabled after collection completes.
	Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
	perf,cpu,jit:avx512_common,"alexnet:conv1",--conv g1mb256ic3ih227oc96oh55kh11sh4ph0n"alexnet:conv1",53.9726,0,17.2344,3131.68,24.1246,2237.24
	perf,cpu,jit:avx512_common,"alexnet:conv2",--conv g2mb256ic96ih27oc256oh27kh5ph2n"alexnet:conv2",104.696,0,20.2988,5157.74,22.6731,4617.63
	perf,cpu,jit:avx512_common,"alexnet:conv3",--conv mb256ic256ih13oc384oh13kh3ph1n"alexnet:conv3",68.904,0,15.5369,4434.87,17.1371,4020.75
	perf,cpu,jit:avx512_common,"alexnet:conv4",--conv g2mb256ic384ih13oc384oh13kh3ph1n"alexnet:conv4",51.678,0,11.428,4522.06,12.7986,4037.79
	perf,cpu,jit:avx512_common,"alexnet:conv5",--conv g2mb256ic384ih13oc256oh13kh3ph1n"alexnet:conv5",34.452,0,7.64233,4508.05,8.99841,3828.68
	tests:5 passed:5 skipped:0 mistrusted:0 unimplemented:0 failed:0 listed:0
	total perf: min(ms):72.1404 avg(ms):85.7318

Below are L1 Data Cache issues among primitive types using the command-line interface:

.. ref-code-block:: cpp

	$ amplxe-cl -report hotspots -q -r dnnl-vtune-ue -format csv -csv-delimiter ';' -group-by task -column 'L1 Bound' | head -n 10 | column -t -s';'
	Column filter is ON.
	Task Type           Back-End Bound:Memory Bound:L1 Bound(%)  Back-End Bound:Memory Bound:L1 Bound:DTLB Overhead(%)  Back-End Bound:Memory Bound:L1 Bound:Loads Blocked by Store Forwarding(%)  Back-End Bound:Memory Bound:L1 Bound:Lock Latency(%)  Back-End Bound:Memory Bound:L1 Bound:Split Loads(%)  Back-End Bound:Memory Bound:L1 Bound:4K Aliasing(%)  Back-End Bound:Memory Bound:L1 Bound:FB Full(%)
	convolution         8.2                                      0.0                                                    0.0                                                                        0.0                                                   0.0                                                  0.2                                                  26.6
	[Outside any task]  4.4                                      0.0                                                    0.0                                                                        0.0                                                   0.0                                                  0.0                                                  2.6
	reorder             16.0                                     0.0                                                    0.0                                                                        0.0                                                   0.0                                                  0.1

Below are issues with Instruction Cache misses among primitive types using the command-line interface:

.. ref-code-block:: cpp

	$ amplxe-cl -report hotspots -q -r dnnl-vtune-ue -format csv -csv-delimiter ';' -group-by task -column 'ICache Misses' | head -n 10 | column -t -s';'
	Column filter is ON.
	Task Type           Front-End Bound:Front-End Latency:ICache Misses(%)
	convolution         0.2
	[Outside any task]  0.1
	reorder             0.3

Here are some column names within micro-architecture profiling results. You could replace 'ICache Misses' with another column name.

\*"ICache Misses","ITLB Overhead","Bad Speculation","L1 Bound","L2 Bound","L3 Bound","DRAM Bound","Average CPU Frequency","Task Time","Task Count", etc\*

For getting all column names within your profiling results, you could use the command below to get more detailed information.

.. ref-code-block:: cpp

	$ amplxe-cl -report hotspots -q -r dnnl-vtune-ue -format csv -csv-delimiter ';' -group-by task -column=?
	Available values for '-column' option are:
	CPU Time:Self
	Clockticks:Self
	Instructions Retired:Self
	CPI Rate:Self
	Retiring:Self
	Retiring:General Retirement:Self
	Retiring:General Retirement:FP Arithmetic:Self
	Retiring:General Retirement:FP Arithmetic:FP x87:Self
	Retiring:General Retirement:FP Arithmetic:FP Scalar:Self
	Retiring:General Retirement:FP Arithmetic:FP Vector:Self
	Retiring:General Retirement:Other:Self
	Retiring:Microcode Sequencer:Self
	Retiring:Microcode Sequencer:Assists:Self
	Front-End Bound:Self
	Front-End Bound:Front-End Latency:Self
	Front-End Bound:Front-End Latency:ICache Misses:Self

See more examples in the `VTune Amplifier User Guide <https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/introduction/tutorials-and-samples.html>`__

Example: Profiling with Linux Perf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command instructs oneDNN to enable both jitdump and perfmap profiling modes and write jitdump files into the ``.debug`` directory in the current directory by setting environment variable ``JITDUMPDIR`` to point to the current directory.

.. ref-code-block:: cpp

	$ JITDUMPDIR=. ONEDNN_JIT_PROFILE=6 perf record -k1 ./tests/benchdnn/benchdnn --conv --mode=P mb1ic32ih14oc32oh14kh3ph1n"resnet_50:res4a_branch2b*6"
	Output template: perf,%engine%,%name%,%desc%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
	perf,cpu,resnet_50:res4a_branch2b*6,--conv mb1ic32ih14oc32oh14kh3ph1nresnet_50:res4a_branch2b*6,0.0032768,0,0.0131836,248.551,0.0262988,124.599
	tests:1 passed:0 skipped:0 mistrusted:0 unimplemented:0 failed:0 listed:0
	total perf: min(ms):0.0131836 avg(ms):0.0262988
	[ perf record: Woken up 1 times to write data ]
	[ perf record: Captured and wrote 0.884 MB perf.data (23102 samples) ]

The following command injects the information from the jitdump files into the performance data:

.. ref-code-block:: cpp

	$ perf inject -j -i perf.data -o perf.data.j

The following command displays the top hotspots:

.. ref-code-block:: cpp

	$ perf report -i perf.data.j --stdio | head -n20
	# To display the perf.data header info, please use --header/--header-only options.
	#
	#
	# Total Lost Samples: 0
	#
	# Samples: 23K of event 'cpu-clock:uhH'
	# Event count (approx.): 5775500000
	#
	# Overhead  Command   Shared Object        Symbol
	#
	    39.33%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8ba
	    29.41%  benchdnn  jitted-31475-0.so    [.] jit_avx2_conv_fwd_kernel_f32
	    20.49%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d712
	     3.47%  benchdnn  libdnnl.so.1.1       [.] dnnl::impl::cpu::jit_avx2_convolution_fwd_t::execute_forward(dnnl::impl::exec_ctx_t const&) const::{lambda(int, int)#1}::operator()
	     1.52%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8be
	     0.93%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d716
	     0.75%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8c5
	     0.55%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d8c3
	     0.46%  benchdnn  libgomp.so.1.0.0     [.] 0x000000000001d71d

.. note:: 

   Not every kernel/distribution supports displaying detailed profiling information. Symbol resolution (usually) works as long as the perfmap mode is enabled, but annotating a JIT-ed functions disassembly, which requires jitdump, seems to often fail on kernels before 5.x.
   
   
See more on `Brendan Gregg's excellent perf examples page <http://www.brendangregg.com/perf.html>`__

