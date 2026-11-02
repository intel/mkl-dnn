.. index:: pair: page; Interpolate Fusion Patterns
.. _doxid-dev_guide_graph_interpolate_fusion_patterns:

Interpolate Fusion Patterns
===========================

Overview
~~~~~~~~

oneDNN supports various Interpolate fusion patterns to optimize performance and reduce memory bandwidth requirements. This document describes the supported fusion patterns for Interpolate.

Pattern Structure
~~~~~~~~~~~~~~~~~

oneDNN defines floating-point Interpolate fusion patterns as follows. The blue nodes are required when defining an Interpolate fusion pattern while the brown nodes are optional.

.. image:: interpolate_pattern.png
	:alt: Interpolate pattern



#. Interpolate Operation : Performs interpolation for the ``src`` tensor at spatial dimensions. See the :ref:`Interpolate <doxid-dev_guide_op_interpolate>` operation in the Graph API for more details.

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

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The Interpolate operation only supports half_pixel coordinate_transformation_mode.

