.. index:: pair: page; Quantized Convolution Fusion Patterns
.. _doxid-dev_guide_graph_quantized_convolution_fusion_patterns:

Quantized Convolution Fusion Patterns
=====================================

Overview
~~~~~~~~

oneDNN supports both floating-point and quantized Convolution fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported quantized fusion patterns for Convolution. For floating-point Convolution fusion patterns, refer to :ref:`Convolution Fusion Patterns <doxid-dev_guide_graph_convolution_fusion_patterns>` for more details.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines quantized Convolution fusion patterns as follows. The blue nodes are required when defining a quantized Convolution fusion pattern while the brown nodes are optional.

.. image:: quantized_conv_pattern.png
	:alt: quantized Convolution pattern



#. Q2F Conversion Subgraph : Converts ``src`` and ``weights`` tensors from quantized to floating-point. It can be one of the following subgraphs, while the last two subgraphs apply only to ``weights``. See :ref:`Dequantize <doxid-dev_guide_op_dequantize>`, :ref:`TypeCast <doxid-dev_guide_op_typecast>` and :ref:`Quantize <doxid-dev_guide_op_quantize>` operations in Graph API.
   
   .. image:: q2f_conversion_quantized_conv_matmul.png
   	:alt: q2f_conversion_subgraph

#. F2F Conversion Subgraph : Converts ``bias`` tensor from floating-point to another floating-point. It is constructed by a :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation.
   
   .. image:: f2f_conversion.png
   	:alt: f2f_conversion_subgraph

#. Convolution Operation : Performs convolution between the ``src`` and ``weights`` tensors. The ``bias`` tensor is optional. See the :ref:`Convolution <doxid-dev_guide_op_convolution>` operation in the Graph API for more details.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * :ref:`BiasAdd <doxid-dev_guide_op_biasadd>` operation.
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_2.png
   	:alt: epilogue subgraph
   
   
   
   * BiasAdd : If present, must be the first op in the epilogue subgraph and can only appear once.
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

#. F2F/F2Q Conversion Subgraph : Converts the output tensor from floating-point to floating-point or quantized data type. It can be one of the following subgraphs. See :ref:`TypeCast <doxid-dev_guide_op_typecast>` and :ref:`Quantize <doxid-dev_guide_op_quantize>` operations in Graph API.
   
   .. image:: f2q_conversion_quantized_conv.png
   	:alt: f2q_conversion_subgraph

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src, weights, bias and dst:

======  ========  =============  ===================  
src     weights   bias           dst                  
======  ========  =============  ===================  
u8,s8   s8,f32    f32,bf16,f16   u8,s8,bf16,f16,f32   
======  ========  =============  ===================

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. F2Q Conversion Subgraph used for ``dst`` tensor only supports bf16 to f32 data type conversion.

Example
~~~~~~~

oneDNN provides a `quantized Convolution example <https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/cpu_inference_int8.cpp>`__ demonstrating how to construct a typical quantized Convolution pattern with oneDNN Graph API on CPU.

