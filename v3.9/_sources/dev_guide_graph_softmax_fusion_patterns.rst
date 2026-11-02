.. index:: pair: page; Softmax Fusion Patterns
.. _doxid-dev_guide_graph_softmax_fusion_patterns:

Softmax Fusion Patterns
=======================

Overview
~~~~~~~~

oneDNN supports various SoftMax fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported fusion patterns for SoftMax.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point SoftMax fusion patterns as follows. The blue nodes are required when defining a SoftMax fusion pattern while the brown nodes are optional.

.. image:: softmax_pattern.png
	:alt: Softmax pattern



#. SoftMax Operation : Performs the softmax function for the ``src`` tensor. See the :ref:`SoftMax <doxid-dev_guide_op_softmax>` operation in the Graph API for more details.

#. F2F Conversion Subgraph : Converts the output tensor from floating-point to another floating-point. It is constructed by a :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation.
   
   .. image:: f2f_conversion.png
   	:alt: f2f_conversion_subgraph

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_1.png
   	:alt: epilogue subgraph
   
   
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

#. F2Q Conversion Subgraph : Converts the output tensor from floating-point to quantized data type. It can be one of the following subgraphs. See :ref:`TypeCast <doxid-dev_guide_op_typecast>` and :ref:`Quantize <doxid-dev_guide_op_quantize>` operations in Graph API.
   
   .. image:: f2q_conversion_softmax.png
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

