.. index:: pair: page; Convolution Fusion Patterns
.. _doxid-dev_guide_graph_convolution_fusion_patterns:

Convolution Fusion Patterns
===========================

Overview
~~~~~~~~

oneDNN supports both floating-point and quantized Convolution fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported floating-point fusion patterns for Convolution. For quantized Convolution fusion patterns, refer to :ref:`Quantized Convolution Fusion Patterns <doxid-dev_guide_graph_quantized_convolution_fusion_patterns>` for more details.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point Convolution fusion patterns as follows. The blue nodes are required when defining a Convolution fusion pattern while the brown nodes are optional.

.. image:: conv_pattern.png
	:alt: Convolution pattern



#. Convolution Operation : Performs convolution between the ``src`` and ``weights`` tensors. The ``bias`` tensor is optional. See the :ref:`Convolution <doxid-dev_guide_op_convolution>` operation in the Graph API for more details.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * :ref:`BiasAdd <doxid-dev_guide_op_biasadd>` operation.
   
   * :ref:`BatchNormInference <doxid-dev_guide_op_batchnorminference>` operation.
   
   * :ref:`Convolution <doxid-dev_guide_op_convolution>` operation.
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_conv.png
   	:alt: epilogue subgraph
   
   
   
   * BiasAdd : If present, must be the first op in the epilogue subgraph and can only appear once.
   
   * BatchNormInference : If present, must precede Binary or Unary operations and can only appear once.
   
   * Convolution : If present, is a Depthwise Convolution which can only be fused with 1x1 Convolution and can only appear once.
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

#. F2F Conversion Subgraph : Converts the output tensor from floating-point to another floating-point. It is constructed by a :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation.
   
   .. image:: f2f_conversion.png
   	:alt: f2f_conversion_subgraph

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src, weights, bias and dst:

=============  =============  =============  =============  
src            weights        bias           dst            
=============  =============  =============  =============  
f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   
=============  =============  =============  =============

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Convolution as a post op (Depthwise Convolution) is not supported on GPU.

#. Convolution and BatchNormInference cannot co-exist in the epilogue subgraph.

#. F2F Conversion Subgraph used for ``dst`` tensor only supports bf16 to f32 data type conversion.

Example
~~~~~~~

oneDNN provides a `CPU Convolution example <https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/cpu_getting_started.cpp>`__ and a `GPU Convolution example <https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/sycl_getting_started.cpp>`__ demonstrating how to construct a typical floating-point Convolution pattern with oneDNN Graph API on CPU and GPU.

