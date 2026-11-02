.. index:: pair: page; Examples and Tutorials
.. _doxid-dev_guide_examples:

Examples and Tutorials
======================

This page provides an overview of oneDNN examples organized by functionality and use case.

Functional API Examples
~~~~~~~~~~~~~~~~~~~~~~~

The Functional API provides access to individual oneDNN primitives.

Fundamental Concepts and API Basics
-----------------------------------

============================================================================  ===================================================================================================================  
Example                                                                       Description                                                                                                          
============================================================================  ===================================================================================================================  
:ref:`oneDNN API Basic Workflow Tutorial <doxid-getting_started_cpp>`         This C++ API example demonstrates the basics of the oneDNN programming model.                                        
:ref:`Memory Format Propagation <doxid-memory_format_propagation_cpp>`        This example demonstrates memory format propagation, which is critical for deep learning applications performance.   
:ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_cpp>`   This C++ API example demonstrates programming flow when reordering memory between CPU and GPU engines.               
============================================================================  ===================================================================================================================

Interoperability with External Runtimes
---------------------------------------

========================================================================================  =====================================================================================================================  
Example                                                                                   Description                                                                                                            
========================================================================================  =====================================================================================================================  
:ref:`Getting Started with SYCL Extensions API <doxid-sycl_interop_buffer_cpp>`           This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN.      
:ref:`SYCL USM Example <doxid-sycl_interop_usm_cpp>`                                      This C++ API example demonstrates programming for Intel(R) Processor Graphics with SYCL extensions API in oneDNN.      
:ref:`Getting started on GPU with OpenCL extensions API <doxid-gpu_opencl_interop_cpp>`   This C++ API example demonstrates programming for Intel(R) Processor Graphics with OpenCL* extensions API in oneDNN.   
========================================================================================  =====================================================================================================================

:target:`doxid-dev_guide_examples_1examples_matmul`

Matrix Multiplication with Different oneDNN Features
----------------------------------------------------

Basic Operations:

===============================================================================  ================================================================================================================  
Example                                                                          Description                                                                                                       
===============================================================================  ================================================================================================================  
:ref:`Matmul Primitive Example <doxid-matmul_example_cpp>`                       This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive.   
:ref:`MatMul Tutorial: Comparison with SGEMM <doxid-cpu_sgemm_and_matmul_cpp>`   C++ API example demonstrating :ref:`MatMul <doxid-dev_guide_matmul>` as a replacement for SGEMM functions.        
===============================================================================  ================================================================================================================

Quantization flavors:

=====================================================================================  ==========================================================================================================================================================================================================================  
Example                                                                                Description                                                                                                                                                                                                                 
=====================================================================================  ==========================================================================================================================================================================================================================  
:ref:`Matrix Multiplication with f8 Quantization <doxid-matmul_f8_quantization_cpp>`   C++ API example demonstrating how to use f8_e5m2 and f8_e4m3 data types for :ref:`MatMul <doxid-dev_guide_matmul>` with scaling for quantization.                                                                           
:ref:`MatMul Tutorial: Quantization <doxid-cpu_matmul_quantization_cpp>`               C++ API example demonstrating how one can perform reduced precision matrix-matrix multiplication using :ref:`MatMul <doxid-dev_guide_matmul>` and the accuracy of the result compared to the floating point computations.   
:ref:`MatMul Tutorial: INT8 Inference <doxid-inference_int8_matmul_cpp>`               C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` fused with ReLU in INT8 inference.                                                                                                     
:ref:`MatMul Tutorial: MXFP8 Inference <doxid-mxfp_matmul_cpp>`                        C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with MXFP8 datatype in inference.                                                                                                      
=====================================================================================  ==========================================================================================================================================================================================================================

Advanced Usages:

==================================================================================================  =======================================================================================================================================================================================================================================  
Example                                                                                             Description                                                                                                                                                                                                                              
==================================================================================================  =======================================================================================================================================================================================================================================  
:ref:`MatMul with Host Scalar Scale example <doxid-matmul_with_host_scalar_scale_cpp>`              This C++ API example demonstrates matrix multiplication (C = alpha * A * B) with a scalar scaling factor residing on the host.                                                                                                           
:ref:`MatMul Primitive with Sparse Memory in COO Format <doxid-cpu_matmul_coo_cpp>`                 This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive that uses a source tensor encoded with the COO sparse encoding.                                                           
:ref:`MatMul Primitive with Sparse Memory in CSR Format <doxid-cpu_matmul_csr_cpp>`                 This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive that uses a source tensor encoded with the CSR sparse encoding.                                                           
:ref:`MatMul with Packed Sparse Weights <doxid-cpu_matmul_weights_compression_cpp>`                 This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive that uses a weights tensor encoded with the packed sparse encoding.                                                       
:ref:`MatMul with Grouped Encoding <doxid-matmul_grouped_cpp>`                                      This C++ API example demonstrates how to execute grouped matrix-matrix multiplication, using grouped encoding and :ref:`MatMul <doxid-dev_guide_matmul>` primitive, that is a commonly used functionality in Mixture-of-Experts (MoE).   
:ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`   C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with quantized weights.                                                                                                                             
==================================================================================================  =======================================================================================================================================================================================================================================

Inference and Training
----------------------

Neural network implementations demonstrating inference and training workflows:

=====  ==========  ==========  =====================================================================  ==============================================================================================================  
Type   Precision   Mode        Example                                                                Description                                                                                                     
=====  ==========  ==========  =====================================================================  ==============================================================================================================  
CNN    f32         Inference   :ref:`CNN f32 inference example <doxid-cnn_inference_f32_cpp>`         This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.   
CNN    int8        Inference   :ref:`CNN int8 inference example <doxid-cnn_inference_int8_cpp>`       This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with int8 data type.                     
CNN    f32         Training    :ref:`CNN f32 training example <doxid-cnn_training_f32_cpp>`           This C++ API example demonstrates how to build an AlexNet model training.                                       
CNN    bf16        Training    :ref:`CNN bf16 training example <doxid-cnn_training_bf16_cpp>`         This C++ API example demonstrates how to build an AlexNet model training using the bfloat16 data type.          
RNN    f32         Inference   :ref:`RNN f32 Inference Example <doxid-cpu_rnn_inference_f32_cpp>`     This C++ API example demonstrates how to build GNMT model inference.                                            
RNN    int8        Inference   :ref:`RNN int8 inference example <doxid-cpu_rnn_inference_int8_cpp>`   This C++ API example demonstrates how to build GNMT model inference.                                            
RNN    f32         Training    :ref:`RNN f32 training example <doxid-rnn_training_f32_cpp>`           This C++ API example demonstrates how to build GNMT model training.                                             
=====  ==========  ==========  =====================================================================  ==============================================================================================================

:target:`doxid-dev_guide_examples_1examples_rnn`

Recurrent Neural Networks
-------------------------

=================================================================================  =======================================================================================================================================================================  
Example                                                                            Description                                                                                                                                                              
=================================================================================  =======================================================================================================================================================================  
:ref:`Vanilla RNN Primitive Example <doxid-vanilla_rnn_example_cpp>`               This C++ API example demonstrates how to create and execute a :ref:`Vanilla RNN <doxid-dev_guide_rnn>` primitive in forward training propagation mode.                   
:ref:`LSTM RNN Primitive Example <doxid-lstm_example_cpp>`                         This C++ API example demonstrates how to create and execute an :ref:`LSTM RNN <doxid-dev_guide_rnn>` primitive in forward training propagation mode.                     
:ref:`Linear-Before-Reset GRU RNN Primitive Example <doxid-lbr_gru_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Linear-Before-Reset GRU RNN <doxid-dev_guide_rnn>` primitive in forward training propagation mode.   
:ref:`AUGRU RNN Primitive Example <doxid-augru_example_cpp>`                       This C++ API example demonstrates how to create and execute an :ref:`AUGRU RNN <doxid-dev_guide_rnn>` primitive in forward training propagation mode.                    
=================================================================================  =======================================================================================================================================================================

Performance Analysis
--------------------

A few techniques for performance measurements:

=========================================================================  ====================================================================================================  
Example                                                                    Description                                                                                           
=========================================================================  ====================================================================================================  
:ref:`Matrix Multiplication Performance Example <doxid-matmul_perf_cpp>`   This C++ example runs a simple matrix multiplication (matmul) performance test using oneDNN.          
:ref:`Performance Profiling Example <doxid-performance_profiling_cpp>`     This example demonstrates the best practices for application performance optimizations with oneDNN.   
=========================================================================  ====================================================================================================

Individual Primitives
---------------------

:target:`doxid-dev_guide_examples_1examples_convolution` Convolution Operations:

=========================================================================  ======================================================================================================================================================================================================  
Example                                                                    Description                                                                                                                                                                                             
=========================================================================  ======================================================================================================================================================================================================  
:ref:`Convolution Primitive Example <doxid-convolution_example_cpp>`       This C++ API example demonstrates how to create and execute a :ref:`Convolution <doxid-dev_guide_convolution>` primitive in forward propagation mode in two configurations - with and without groups.   
:ref:`Deconvolution Primitive Example <doxid-deconvolution_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Deconvolution <doxid-dev_guide_convolution>` primitive in forward propagation mode.                                                 
=========================================================================  ======================================================================================================================================================================================================

:target:`doxid-dev_guide_examples_1examples_inner_product` Linear Operations:

=========================================================================  ===============================================================================================================================  
Example                                                                    Description                                                                                                                      
=========================================================================  ===============================================================================================================================  
:ref:`Inner Product Primitive Example <doxid-inner_product_example_cpp>`   This C++ API example demonstrates how to create and execute an :ref:`Inner Product <doxid-dev_guide_inner_product>` primitive.   
=========================================================================  ===============================================================================================================================

:target:`doxid-dev_guide_examples_1examples_pooling` Pooling and Sampling:

===================================================================  =============================================================================================================================================================  
Example                                                              Description                                                                                                                                                    
===================================================================  =============================================================================================================================================================  
:ref:`Pooling Primitive Example <doxid-pooling_example_cpp>`         This C++ API example demonstrates how to create and execute a :ref:`Pooling <doxid-dev_guide_pooling>` primitive in forward training propagation mode.         
:ref:`Resampling Primitive Example <doxid-resampling_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Resampling <doxid-dev_guide_resampling>` primitive in forward training propagation mode.   
===================================================================  =============================================================================================================================================================

:target:`doxid-dev_guide_examples_1examples_normalization` Normalization Primitives:

=====================================================================================  ===============================================================================================================================================================================  
Example                                                                                Description                                                                                                                                                                      
=====================================================================================  ===============================================================================================================================================================================  
:ref:`Batch Normalization Primitive Example <doxid-batch_normalization_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Batch Normalization <doxid-dev_guide_batch_normalization>` primitive in forward training propagation mode.   
:ref:`Group Normalization Primitive Example <doxid-group_normalization_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Group Normalization <doxid-dev_guide_group_normalization>` primitive in forward training propagation mode.   
:ref:`Layer Normalization Primitive Example <doxid-layer_normalization_example_cpp>`   This C++ API example demonstrates how to create and execute a :ref:`Layer normalization <doxid-dev_guide_layer_normalization>` primitive in forward propagation mode.            
:ref:`Local Response Normalization Primitive Example <doxid-lrn_example_cpp>`          This C++ API demonstrates how to create and execute a :ref:`Local response normalization <doxid-dev_guide_lrn>` primitive in forward training propagation mode.                  
=====================================================================================  ===============================================================================================================================================================================

:target:`doxid-dev_guide_examples_1examples_activation` Activation Functions:

==================================================================  =============================================================================================================================================================  
Example                                                             Description                                                                                                                                                    
==================================================================  =============================================================================================================================================================  
:ref:`Element-Wise Primitive Example <doxid-eltwise_example_cpp>`   This C++ API example demonstrates how to create and execute an :ref:`Element-wise <doxid-dev_guide_eltwise>` primitive in forward training propagation mode.   
:ref:`Primitive Example <doxid-prelu_example_cpp>`                  This C++ API example demonstrates how to create and execute an :ref:`PReLU <doxid-dev_guide_prelu>` primitive in forward training propagation mode.            
:ref:`Softmax Primitive Example <doxid-softmax_example_cpp>`        This C++ API example demonstrates how to create and execute a :ref:`Softmax <doxid-dev_guide_softmax>` primitive in forward training propagation mode.         
==================================================================  =============================================================================================================================================================

:target:`doxid-dev_guide_examples_1examples_tensor_operations` Tensor Operations:

===================================================================================  ==============================================================================================================================================================================================  
Example                                                                              Description                                                                                                                                                                                     
===================================================================================  ==============================================================================================================================================================================================  
:ref:`Binary Primitive Example <doxid-binary_example_cpp>`                           This C++ API example demonstrates how to create and execute a :ref:`Binary <doxid-dev_guide_binary>` primitive.                                                                                 
:ref:`Bnorm u8 by binary post-ops example <doxid-bnorm_u8_via_binary_postops_cpp>`   The example implements the Batch normalization u8 via the following operations: binary_sub(src, mean), binary_div(tmp_dst, variance), binary_mul(tmp_dst, scale), binary_add(tmp_dst, shift).   
:ref:`Concat Primitive Example <doxid-concat_example_cpp>`                           This C++ API example demonstrates how to create and execute a :ref:`Concat <doxid-dev_guide_concat>` primitive.                                                                                 
:ref:`Reduction Primitive Example <doxid-reduction_example_cpp>`                     This C++ API example demonstrates how to create and execute a :ref:`Reduction <doxid-dev_guide_reduction>` primitive.                                                                           
:ref:`Sum Primitive Example <doxid-sum_example_cpp>`                                 This C++ API example demonstrates how to create and execute a :ref:`Sum <doxid-dev_guide_sum>` primitive.                                                                                       
:ref:`Shuffle Primitive Example <doxid-shuffle_example_cpp>`                         This C++ API example demonstrates how to create and execute a :ref:`Shuffle <doxid-dev_guide_shuffle>` primitive.                                                                               
===================================================================================  ==============================================================================================================================================================================================

:target:`doxid-dev_guide_examples_1examples_reorder` Memory Transformations:

=============================================================  ==========================================================================================================  
Example                                                        Description                                                                                                 
=============================================================  ==========================================================================================================  
:ref:`Reorder Primitive Example <doxid-reorder_example_cpp>`   This C++ API demonstrates how to create and execute a :ref:`Reorder <doxid-dev_guide_reorder>` primitive.   
=============================================================  ==========================================================================================================

C API Examples
--------------

==========================================================================  ================================================================================================================================  
Example                                                                     Description                                                                                                                       
==========================================================================  ================================================================================================================================  
:ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_c>`   This C API example demonstrates programming flow when reordering memory between CPU and GPU engines.                              
:ref:`CNN f32 inference example <doxid-cnn_inference_f32_c>`                This C API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.                       
:ref:`CNN f32 training example <doxid-cpu_cnn_training_f32_c>`              This C API example demonstrates how to build an AlexNet model training. The example implements a few layers from AlexNet model.   
==========================================================================  ================================================================================================================================

Graph API Examples
~~~~~~~~~~~~~~~~~~

The Graph API provides an interface for defining computational graphs with optimization and fusion capabilities.

Getting Started with Graph API
------------------------------

=========================================================================================================  =============================================================================================  
Example                                                                                                    Description                                                                                    
=========================================================================================================  =============================================================================================  
:ref:`Getting started on CPU with Graph API <doxid-graph_cpu_getting_started_cpp>`                         This is an example to demonstrate how to build a simple graph and run it on CPU.               
:ref:`Getting started with SYCL extensions API and Graph API <doxid-graph_sycl_getting_started_cpp>`       This is an example to demonstrate how to build a simple graph and run on SYCL device.          
:ref:`Getting started with OpenCL extensions and Graph API <doxid-graph_gpu_opencl_getting_started_cpp>`   This is an example to demonstrate how to build a simple graph and run on OpenCL GPU runtime.   
=========================================================================================================  =============================================================================================

Advanced Graph API Usage
------------------------

==============================================================================================  ===============================================================================================  
Example                                                                                         Description                                                                                      
==============================================================================================  ===============================================================================================  
:ref:`Convolution int8 inference example with Graph API <doxid-graph_cpu_inference_int8_cpp>`   This is an example to demonstrate how to build an int8 graph with Graph API and run it on CPU.   
:ref:`Single op partition on CPU <doxid-graph_cpu_single_op_partition_cpp>`                     This is an example to demonstrate how to build a simple op graph and run it on CPU.              
:ref:`Single op partition on GPU <doxid-graph_sycl_single_op_partition_cpp>`                    This is an example to demonstrate how to build a simple op graph and run it on GPU.              
==============================================================================================  ===============================================================================================

Microkernel (uKernel) API Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The oneDNN microkernel API is a low-level abstraction for CPU that provides maximum flexibility by allowing users to maintain full control over threading logic, blocking logic, and code customization with minimal overhead.

=============================================================  ==============================================================================  
Example                                                        Description                                                                     
=============================================================  ==============================================================================  
:ref:`BRGeMM ukernel example <doxid-cpu_brgemm_example_cpp>`   This C++ API example demonstrates how to create and execute a BRGeMM ukernel.   
=============================================================  ==============================================================================

Running Examples
~~~~~~~~~~~~~~~~

Prerequisites and Building Examples
-----------------------------------

Before running examples, ensure:

#. oneDNN is built from source. Note that examples are built automatically when building oneDNN with ``-DONEDNN_BUILD_EXAMPLES=ON`` (enabled by default).

#. Environment is set up and oneDNN libraries are in the path.

Refer to :ref:`Build from Source <doxid-dev_guide_build>` for detailed build instructions.

Running Examples
----------------

Most examples accept an optional engine argument (``cpu`` or ``gpu``), and if no argument is provided, example will most likely default to CPU:

Linux/macOS:

.. ref-code-block:: cpp

	# Run on CPU (default)
	./examples/getting_started
	
	# Run on CPU explicitly
	./examples/getting_started cpu
	
	# Run on GPU (if available)
	./examples/getting_started gpu

Windows:

.. ref-code-block:: cpp

	# Run on CPU (default)
	examples\getting_started.exe
	
	# Run on CPU explicitly
	examples\getting_started.exe cpu
	
	# Run on GPU (if available)
	examples\getting_started.exe gpu

Examples will output "Example passed on CPU/GPU." upon successful completion and display an error status with message otherwise.

