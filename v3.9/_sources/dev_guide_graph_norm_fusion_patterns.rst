.. index:: pair: page; Norm Fusion Patterns
.. _doxid-dev_guide_graph_norm_fusion_patterns:

Norm Fusion Patterns
====================

Overview
~~~~~~~~

The Norm category for inference includes operations such as: GroupNorm, LayerNorm and BatchNormInference.

oneDNN supports various Norm fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported fusion patterns for Norm.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point Norm fusion patterns as follows. The blue nodes are required when defining a Norm fusion pattern while the brown nodes are optional.

.. image:: norm_pattern.png
	:alt: Norm pattern



#. Norm Operation : Performs the corresponding norm operation for the ``src`` tensor. See the :ref:`GroupNorm <doxid-dev_guide_op_groupnorm>`, :ref:`LayerNorm <doxid-dev_guide_op_layernorm>`, :ref:`BatchNormInference <doxid-dev_guide_op_batchnorminference>` operations in the Graph API for more details.

#. F2F Conversion Subgraph : Converts the output tensor from floating-point to another floating-point. It is constructed by a :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation.
   
   .. image:: f2f_conversion.png
   	:alt: f2f_conversion_subgraph

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_1.png
   	:alt: epilogue subgraph
   
   
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

#. F2Q Conversion Subgraph : Converts the output tensor from floating-point to quantized data type. It can be one of the following subgraphs. It is constructed by a :ref:`Quantize <doxid-dev_guide_op_quantize>` operation.
   
   .. image:: f2q_conversion_general.png
   	:alt: f2q_conversion_subgraph

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src and dst:

=============  ===================  
src            dst                  
=============  ===================  
bf16,f16,f32   u8,s8,bf16,f16,f32   
=============  ===================

The definition of data types and their support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. BatchNormInference:
   
   #. The Epilogue Subgraph only supports ReLU, and if present, can only appear once.
   
   #. F2F and F2Q Conversion Subgraphs are not supported.

