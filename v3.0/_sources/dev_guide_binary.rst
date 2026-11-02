.. index:: pair: page; Binary
.. _doxid-dev_guide_binary:

Binary
======

:ref:`API Reference <doxid-group__dnnl__api__binary>`

General
~~~~~~~

The binary primitive computes the result of a binary elementwise operation between tensors source 0 and source 1 (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(\overline{x}) = \src_0(\overline{x}) \mathbin{op} \src_1(\overline{x}),

where :math:`op` is one of addition, subtraction, multiplication, division, greater than or equal to, greater than, less than or equal to, less than, equal to, not equal to, get maximum value, and get minimum value.

The binary primitive does not have a notion of forward or backward propagations.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src_0`                  DNNL_ARG_SRC_0                                                                                                                                                     
:math:`\src_1`                  DNNL_ARG_SRC_1                                                                                                                                                     
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
:math:`binary scale0`           DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0                                                                                                                              
:math:`binary scale1`           DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1                                                                                                                              
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

* The binary primitive requires all source and destination tensors to have the same number of dimensions.

* The binary primitive supports implicit broadcast semantics for source 0 and source 1. This means that if a dimension size is one, that single value will be broadcast (used to compute an operation with each point of the other source) for that dimension. It is recommended to use broadcast for source 1 to get better performance. Generally it should match the syntax below: ``{N,1}x{C,1}x{D,1}x{H,1}x{W,1}:{N,1}x{C,1}x{D,1}x{H,1}x{W,1} -> NxCxDxHxW``. It is consistent with `PyTorch broadcast semantic <https://pytorch.org/docs/stable/notes/broadcasting.html>`__.

* The dimensions of both sources must match unless either is equal to one.

* :math:`\src_1` and :math:`\dst` memory formats can be either specified explicitly or by :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` (recommended), in which case the primitive will derive the most appropriate memory format based on the format of the source 0 tensor. The :math:`\dst` tensor dimensions must match the ones of the source 0 and source 1 tensors (except for broadcast dimensions).

* The binary primitive supports in-place operations, meaning that source 0 tensor may be used as the destination, in which case its data will be overwritten. In-place mode requires the :math:`\dst` and source 0 data types to be the same. Different data types will unavoidably lead to correctness issues.

Post-Ops and Attributes
-----------------------

The following attributes are supported:

==========  =======================================================================================  =====================================================================================  ============================================================  
Type        Operation                                                                                Description                                                                            Res                                                           
==========  =======================================================================================  =====================================================================================  ============================================================  
Attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`   Scales the corresponding input tensor by the given scale factor(s).                    Only one scale per tensor is supported. Input tensors only.   
Post-op     :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`            Adds the operation result to the destination tensor instead of overwriting it.                                                                       
Post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`        Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result.                                                                 
Post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`         Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result       General binary post-op restrictions                           
==========  =======================================================================================  =====================================================================================  ============================================================

Data Types Support
------------------

The source and destination tensors may have ``f32``, ``bf16``, ``f16`` or ``s8/u8`` data types. The binary primitive supports the following combinations of data types:

=======================  =======================  
Source 0 / 1             Des                      
=======================  =======================  
f32, bf16, f16, u8, s8   f32, bf16, f16, u8, s8   
=======================  =======================

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check :ref:`Implementation Limitations <doxid-dev_guide_binary_1dg_binary_impl_limits>` section below.
   
   


Data Representation
-------------------

Sources, Destination
++++++++++++++++++++

The binary primitive works with arbitrary data tensors. There is no special meaning associated with any of tensors dimensions.

:target:`doxid-dev_guide_binary_1dg_binary_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

#. Whenever possible, avoid specifying different memory formats for source tensors.

Examples
~~~~~~~~

:ref:`Binary Primitive Example <doxid-binary_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Binary <doxid-dev_guide_binary>` primitive.

Key optimizations included in this example:

* In-place primitive execution;

* Primitive attributes with fused post-ops.

:ref:`Bnorm u8 by Binary Post-Ops Example <doxid-bnorm_u8_via_binary_postops_cpp>`

Bnorm u8 via binary postops example.

