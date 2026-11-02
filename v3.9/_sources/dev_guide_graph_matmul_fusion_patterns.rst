.. index:: pair: page; MatMul Fusion Patterns
.. _doxid-dev_guide_graph_matmul_fusion_patterns:

MatMul Fusion Patterns
======================

Overview
~~~~~~~~

oneDNN supports both floating-point and quantized MatMul fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported floating-point fusion patterns for MatMul. For quantized MatMul fusion patterns, refer to :ref:`Quantized MatMul Fusion Patterns <doxid-dev_guide_graph_quantized_matmul_fusion_patterns>` for more details.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point MatMul fusion patterns as follows. The blue nodes are required when defining a MatMul fusion pattern while the brown nodes are optional.

.. image:: matmul_pattern.png
	:alt: MatMul pattern



#. MatMul Operation : Performs matrix multiplication between the ``src`` and ``weights`` tensors. The ``bias`` tensor is optional. See the :ref:`MatMul <doxid-dev_guide_op_matmul>` operation in the Graph API for more details.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * :ref:`BiasAdd <doxid-dev_guide_op_biasadd>` operation.
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   * :ref:`Select <doxid-dev_guide_op_select>` operation.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_matmul.png
   	:alt: epilogue subgraph
   
   
   
   * BiasAdd : If present, must be the first op in the epilogue subgraph and can only appear once.
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.
   
   * Select : If present, must follow binary/unary operations (if present) and can only appear once.

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src, weights, bias and dst:

=============  =============  =============  =============  
src            weights        bias           dst            
=============  =============  =============  =============  
f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   
=============  =============  =============  =============

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Example
~~~~~~~

oneDNN provides a `CPU MatMul example <https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/cpu_simple_op_partition.cpp>`__ and a `GPU MatMul example <https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/sycl_simple_op_partition.cpp>`__ demonstrating how to construct a typical floating-point MatMul pattern with oneDNN Graph API on CPU and GPU.

