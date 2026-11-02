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

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   .. image:: epilogue_subgraph_general_1.png
   	:alt: epilogue subgraph
   
   
   
   * N=20, 0 to 20 Binary or Unary operations are supported in the epilogue subgraph.

#. F2F Conversion Subgraph : Converts the output tensor from floating-point to another floating-point. It is constructed by a :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation.
   
   .. image:: f2f_conversion.png
   	:alt: f2f_conversion_subgraph

#. F2Q Conversion Subgraph : Converts the output tensor from floating-point to quantized data type. It is constructed by a :ref:`Quantize <doxid-dev_guide_op_quantize>` operations in Graph API.
   
   .. image:: f2q_conversion_general.png
   	:alt: f2q_conversion_subgraph

#. If multiple optional subgraphs are present, they must follow the order defined in the pattern structure.

Data Types
~~~~~~~~~~

Refer to the document of each operation for the supported data types.

If any optional subgraph is present, the output data type of ``SoftMax`` must be ``f32``. With that, the F2F conversion subgraph converts the output from ``f32`` to ``f16`` or ``bf16`` and the F2Q conversion sugraph quantizes the output from ``f32`` to ``int8``.

The definition of data types and their support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

Post-binary Add operations in the epilogue subgraph support in-place operations when the post-binary Add is the last operation in the epilogue subgraph and the ``dst`` output shape is identical and data type size is the same as the binary Add input. In case of an in-place operation, the original input data will be overwritten. Use in-place operations whenever possible for performance.

