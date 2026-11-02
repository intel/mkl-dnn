.. index:: pair: page; Graph Compiler
.. _doxid-dev_guide_graph_compiler:

Graph Compiler
==============

oneDNN Graph Compiler is an experimental backend for oneDNN Graph API. It can generate optimized implementations for complex computational graphs including multi-head attention (MHA), multi-layer perceptron (MLP), and convolution residual blocks over typical data types for both inference and training. It also brings improved performance by providing more flexible operator fusion.

Use of oneDNN Graph Compiler is transparent for applications, as it does not involve API or programming model changes.

Build-Time Controls
~~~~~~~~~~~~~~~~~~~

The following build time options only work when ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND`` is ON.

===================================================  ========================================  ==========================================================================================================================================================================================================================================  
CMake Option                                         Supported values (defaults in bold)       Description                                                                                                                                                                                                                                 
===================================================  ========================================  ==========================================================================================================================================================================================================================================  
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT           llvm, c, **builtin**                      Selects the CPU codegen and JIT to be built by graph compiler backend. Multiple codegen approaches can be used simultaneously. See the :ref:`example <doxid-dev_guide_graph_compiler_1jit_options>` for setting multiple codegen methods.   
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_LLVM_CONFIG   **AUTO** , *path to llvm-config binary*   Defines the method for detecting and configuring LLVM.                                                                                                                                                                                      
===================================================  ========================================  ==========================================================================================================================================================================================================================================

:target:`doxid-dev_guide_graph_compiler_1jit_options`

Codegen and JIT Options
-----------------------

Graph compiler backend supports several different codegen and JIT options including C, LLVM, and builtin (xbyak). Users can choose to build a subset of available options by setting the ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT`` option.

.. ref-code-block:: cpp

	cmake .. -DONEDNN_BUILD_GRAPH=ON -DONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND=ON -DONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT="c;builtin"

This will only build ``c`` and ``builtin`` codegen options.

.. ref-code-block:: cpp

	cmake .. -DONEDNN_BUILD_GRAPH=ON -DONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND=ON -DONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT="llvm;c;builtin"

This will build all three codegen options.

C
+

C codegen generates temporary cpp files and adopts ``g++`` to compile them into the executable. It can be used for debugging purposes as the generated code is more friendly and readable to developers.

LLVM
++++

LLVM codegen generates LLVM-IR in memory. It provides the best performance among all supported codegen methods. When LLVM codegen is chosen, extra LLVM dependency is required. If LLVM does not exist in this case, a CMake error will occur.

Users can set ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_LLVM_CONFIG`` to specify the LLVM to be integrated. By default, ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_LLVM_CONFIG`` is set to ``AUTO``, which auto-detects existing LLVM in the environment. If auto-detection fails or user wants to explicitly specify the version of LLVM, a specific path to llvm-config binary shall be set.

Users can follow the `guidelines <https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm>`__ to build and install LLVM from source, or download and install the pre-built binary from `here <https://apt.llvm.org/>`__.

.. note:: 

   LLVM 10.0 or above is required to enable LLVM codegen.
   
   


Builtin
+++++++

Builtin codegen and JIT method is implemented with xbyak technology inside. Compared with C or LLVM codegen, it has no extra dependency.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The following environment variables are introduced by the graph compiler backend.

=====================================================  =============================  ========================================================================================================  
Environment Variable                                   Value                          Description                                                                                               
=====================================================  =============================  ========================================================================================================  
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT             **llvm**                       Uses LLVM as codegen and JIT method                                                                       
                                                       builtin                        Uses builtin as codegen and JIT method                                                                    
                                                       c                              Uses C as codegen and JIT method                                                                          
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_OPT_LEVEL           0                              Turns off optimization passes and sets the compilation optimization level to be 0 in C and LLVM JIT       
                                                       1,2, **3**                     Sets the compilation optimization level of C and LLVM JIT                                                 
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE        **0**                          No kernel execution trace output                                                                          
                                                       1, *stderr or filename.json*   Generates kernel execution trace to the file specified by the given filename with chrome tracing format   
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_PRINT_PASS_RESULT   **0**                          No IR output after each graph or tensor IR pass                                                           
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_PRINT_PASS_RESULT   1                              Prints the output IR of each graph and tensor IR passes                                                   
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_VERBOSE             **0**                          No verbose output                                                                                         
                                                       1                              Prints warning messages during compilation                                                                
                                                       2                              Prints warning messages and info logs (e.g. fusion-related information) during compilation                
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GENCODE        *path_to_dump*                 Dumps the generated kernel in C                                                                           
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_C_INCLUDE           *path_to_c_codegen_header*     Specifies the C codegen header for JIT compilation                                                        
=====================================================  =============================  ========================================================================================================

Enable Tracing
--------------

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE=1 ./application

This will produce a kernel execution trace in JSON format that will be stored to the default destination: ``./sctrace.json``.

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE=1,stderr ./application

This will dump a kernel execution trace to the stderr stream.

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE=1,/tmp/filename.json ./application

This will produce a kernel execution trace in JSON format that will be stored to the user specified path ``/tmp/filename.json``.

Switch Between Different Codegen Methods
----------------------------------------

By default, codegen methods have priorities ranked from higher to lower as ``llvm``, ``c``, ``builtin``. When multiple codegen and JIT methods are enabled at build stage, the method with the highest priority is adopted at runtime by default.

Users can switch to a different codegen method at runtime by setting ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT``.

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT=builtin ./application

This will switch the CPU codegen and JIT method to ``builtin`` (xbyak).

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_CPU_JIT=c ./application

This will switch the CPU codegen and JIT method to ``c``.

When using C codegen option, the generated C code will rely on existing runtime function declarations in ``cpu_include.hpp``. ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_C_INCLUDE`` environment variable is used to specify the corresponding include path. Normally, the include path is automatically set at CMake build stage. But if the following error message occurs ``environment variable ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_C_INCLUDE is not set``, users shall manually set ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_C_INCLUDE`` to ``/path_to_onednn_repo/src/graph/backend/graph_compiler/core/src``.

.. warning:: 

   The specified codegen method must be built. Otherwise, the default codegen method would be used.
   
   


Enable Code Dumping
-------------------

Users can use ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GENCODE`` variable to generate offline C kernels.

.. ref-code-block:: cpp

	ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GENCODE="./dump_code" ./application

This will dump the generated C kernels to ``dump_code`` folder.

.. warning:: 

   ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GENCODE`` works under both LLVM and C codegen.
   
   

.. warning:: 

   The user specified ``ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_DUMP_GENCODE`` path shall be an existing folder. Otherwise the code dumping will not be in effect.

