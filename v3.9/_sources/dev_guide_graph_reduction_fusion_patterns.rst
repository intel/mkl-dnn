.. index:: pair: page; Reduction Fusion Patterns
.. _doxid-dev_guide_graph_reduction_fusion_patterns:

Reduction Fusion Patterns
=========================

Overview
~~~~~~~~

The Reduction category includes operations such as: ReduceL1, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum.

oneDNN supports various reduction fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported fusion patterns for Reduction.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point Reduction fusion patterns as follows. The blue nodes are required when defining a Reduction fusion pattern while the brown nodes are optional.

.. image:: reduction_pattern.png
	:alt: Reduction pattern



#. Reduction Operation : Performs the corresponding reduction operation for the ``src`` tensor. See the :ref:`ReduceL1 <doxid-dev_guide_op_reducel1>`, :ref:`ReduceL2 <doxid-dev_guide_op_reducel2>`, :ref:`ReduceMax <doxid-dev_guide_op_reducemax>`, :ref:`ReduceMean <doxid-dev_guide_op_reducemean>`, :ref:`ReduceMin <doxid-dev_guide_op_reducemin>`, :ref:`ReduceProd <doxid-dev_guide_op_reduceprod>` and :ref:`ReduceSum <doxid-dev_guide_op_reducesum>` operations in the Graph API for more details.

#. Epilogue Subgraph : Optional and can include the following operations:
   
   * Binary and Unary operations: refer to the Note in `Fusion Patterns <graph_fusion_patterns.html>`__.
   
   Combination Rules:
   
   .. image:: epilogue_subgraph_general_1.png
   	:alt: epilogue subgraph
   
   
   
   * 0 to 4 Binary or Unary operations are supported in the epilogue subgraph.

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for src and dst:

=============  =============  
src            dst            
=============  =============  
f32,bf16,f16   f32,bf16,f16   
=============  =============

The definition of the data types and support status on different CPU and GPU platforms follow the general description in the :ref:`Data Types Guide <doxid-dev_guide_data_types>`.

