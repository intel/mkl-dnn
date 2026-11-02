.. index:: pair: page; ConvTranspose Fusion Patterns
.. _doxid-dev_guide_graph_convtranspose_fusion_patterns:

ConvTranspose Fusion Patterns
=============================

Overview
~~~~~~~~

oneDNN supports both floating-point and quantized ConvTranspose fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported floating-point fusion patterns for ConvTranspose. For quantized ConvTranspose fusion patterns, refer to :ref:`Quantized ConvTranspose Fusion Patterns <doxid-dev_guide_graph_quantized_convtranspose_fusion_patterns>` for more details.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point ConvTranspose fusion patterns as follows. The blue nodes are required when defining a ConvTranspose fusion pattern while the brown nodes are optional.

.. image:: convtranspose_pattern.png
	:alt: ConvTranspose pattern



#. ConvTranspose Operation : Performs transposed convolution between the ``src`` and ``weights`` tensors. The ``bias`` tensor is optional. See the :ref:`ConvTranspose <doxid-dev_guide_op_convtranspose>` operation in the Graph API for more details.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * :ref:`BiasAdd <doxid-dev_guide_op_biasadd>` operation.
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_2.png
   	:alt: epilogue subgraph
   
   
   
   * BiasAdd : If present, must be the first op in the epilogue subgraph and can only appear once.
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src, weights, bias and dst:

=============  =============  =============  =============  
src            weights        bias           dst            
=============  =============  =============  =============  
f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   f32,bf16,f16   
=============  =============  =============  =============

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

