.. index:: pair: page; Unary Fusion Patterns
.. _doxid-dev_guide_graph_unary_fusion_patterns:

Unary Fusion Patterns
=====================

Overview
~~~~~~~~

oneDNN supports various unary fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported fusion patterns for Unary operations.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point Unary fusion patterns as follows. The blue nodes are required when defining a Unary fusion pattern while the brown nodes are optional.

.. image:: unary_pattern.png
	:alt: Unary pattern



#. Unary Operation : Performs the corresponding unary operation for the ``src`` tensor. Refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * Unary operations.
   
   * Binary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_1.png
   	:alt: epilogue subgraph
   
   
   
   * N=20, 0 to 20 Binary or Unary operations are supported in the epilogue subgraph.

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src and dst:

=============  =============  
src            dst            
=============  =============  
f32,bf16,f16   f32,bf16,f16   
=============  =============

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

Post-binary Add operations in the epilogue subgraph support in-place operations when the post-binary Add is the last operation in the epilogue subgraph and the ``dst`` output shape is identical and data type size is the same as the binary Add input. In case of an in-place operation, the original input data will be overwritten. Use in-place operations whenever possible for performance.

