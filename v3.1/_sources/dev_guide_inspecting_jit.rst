.. index:: pair: page; Inspecting JIT Code
.. _doxid-dev_guide_inspecting_jit:

Inspecting JIT Code
===================

oneDNN uses just-in-time compilation (JIT) to generate optimal code for some functions based on input parameters and the instruction set supported by the system. The library provides a mechanism to save the generated code into a file for inspection.

This behavior can be enabled with the ``ONEDNN_JIT_DUMP`` environment variable or :ref:`dnnl_set_jit_dump <doxid-group__dnnl__api__service_1ga03c8f4af3d01f76060f98e78039837fc>` function.

================  ===============================  
Value             Behavior                         
================  ===============================  
**0**             JIT dump is disabled (default)   
any other value   JIT dump is enabled              
================  ===============================

The function setting takes precedence over the environment variable.

Example (CPU)
~~~~~~~~~~~~~

.. ref-code-block:: cpp

	$ ONEDNN_JIT_DUMP=1 ./cnn-inference-f32-cpp

This will produce the following output files if running on a CPU supporting Intel(R) Advanced Vector Extensions 2 (Intel AVX2):

.. ref-code-block:: cpp

	dnnl_dump_cpu_jit_avx2_conv_fwd_kernel_f32.1.bin
	...
	dnnl_dump_cpu_jit_avx_gemv_t_f32_kern.30.bin

Use any disassembler to view the code. For example:

* ``objdump -D -b binary -mi386:x86-64 file.bin``;

* ``xed -64 -ir file.bin``

`XED <https://github.com/intelxed/xed>`__ is a decoder tool available as part as `Intel Software Development Emulator (Intel SDE) <https://software.intel.com/content/www/us/en/develop/articles/intel-software-development-emulator>`__.

Example (GPU)
~~~~~~~~~~~~~

.. ref-code-block:: cpp

	$ ONEDNN_JIT_DUMP=1 ./simple-net-cpp gpu

This will produce the following output files if running on Intel Processor Graphics Gen9:

.. ref-code-block:: cpp

	dnnl_dump_gpu_simple_reorder.0.bin
	dnnl_dump_gpu_gen9_conv_fwd.1.bin
	...

Use Intel GPU ISA disassembler to disassemble a kernel:

* ``iga64 -d -p=9 file.bin`` (usage: ``-p=<PLATFORM>``)

Links:

* `Building an Intel GPU ISA Disassembler <https://github.com/intel/opencl-intercept-layer/blob/master/docs/kernel_isa_gpu.md#building-an-intel-gpu-isa-disassembler>`__

* `Introduction to GEN Assembly <https://software.intel.com/content/www/us/en/develop/articles/introduction-to-gen-assembly.html>`__

