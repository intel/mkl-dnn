.. index:: pair: page; Reduction
.. _doxid-dev_guide_reduction:

Reduction
=========

:ref:`API Reference <doxid-group__dnnl__api__reduction>`

General
~~~~~~~

The reduction primitive performs reduction operation on arbitrary data. Each element in the destination is the result of reduction operation with specified algorithm along one or multiple source tensor dimensions:

.. math::

	\dst(f) = \mathop{reduce\_op}\limits_{r}\src(r),

where :math:`reduce\_op` can be max, min, sum, mul, mean, Lp-norm and Lp-norm-power-p, :math:`f` is an index in an idle dimension and :math:`r` is an index in a reduction dimension.

Mean:

.. math::

	\dst(f) = \frac{\sum\limits_{r}\src(r)} {R},

where :math:`R` is the size of a reduction dimension.

Lp-norm:

.. math::

	\dst(f) = \root p \of {\mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps)},

where :math:`eps\_op` can be max and sum.

Lp-norm-power-p:

.. math::

	\dst(f) = \mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps),

where :math:`eps\_op` can be max and sum.

Notes
-----

* The reduction primitive requires the source and destination tensors to have the same number of dimensions.

* Reduction dimensions are of size 1 in a destination tensor.

* The reduction primitive does not have a notion of forward or backward propagations.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

* The :math:`\dst` memory format can be either specified explicitly or by :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` (recommended), in which case the primitive will derive the most appropriate memory format based on the format of the source tensor.

Post-Ops and Attributes
-----------------------

The following attributes are supported:

========  ==================================================================================  =====================================================================================  ====================================  
Type      Operation                                                                           Description                                                                            Restrictions                          
========  ==================================================================================  =====================================================================================  ====================================  
Post-op   :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`       Adds the operation result to the destination tensor instead of overwriting it.                                               
Post-op   :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`   Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result.                                         
Post-op   :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`    Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result       General binary post-op restrictions   
========  ==================================================================================  =====================================================================================  ====================================

Data Types Support
------------------

The source and destination tensors may have ``f32``, ``bf16``, ``f16`` or ``int8`` data types. See :ref:`Data Types <doxid-dev_guide_data_types>` page for more details.

Data Representation
-------------------

Sources, Destination
++++++++++++++++++++

The reduction primitive works with arbitrary data tensors. There is no special meaning associated with any of the dimensions of a tensor.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

#. Whenever possible, avoid specifying different memory formats for source and destination tensors.

Example
~~~~~~~

:ref:`Reduction Primitive Example <doxid-reduction_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Reduction <doxid-dev_guide_reduction>` primitive.

