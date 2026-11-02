.. index:: pair: page; Sum
.. _doxid-dev_guide_sum:

Sum
===

:ref:`API Reference <doxid-group__dnnl__api__sum>`

General
~~~~~~~

The sum primitive sums :math:`N` tensors (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(\overline{x}) = \sum\limits_{i = 1}^{N} scales(i) \cdot \src_i(\overline{x})

The sum primitive does not have a notion of forward or backward propagations. The backward propagation for the sum operation is simply an identity operation.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =========================  
primitive input/output   execution argument index   
=======================  =========================  
:math:`\src`             DNNL_ARG_MULTIPLE_SRC      
:math:`\dst`             DNNL_ARG_DST               
=======================  =========================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

* The :math:`\dst` memory format can be either specified by a user or derived the most appropriate one by the primitive. The recommended way is to allow the primitive to choose the appropriate format.

* The sum primitive requires all source and destination tensors to have the same shape. Implicit broadcasting is not supported.

* The sum primitive supports in-place operation, meaning that the :math:`\src_0` tensor can be used as both input and output. In-place operation overwrites the original data. Using in-place operation requires the memory footprint of the output tensor to be either bigger than or equal to the size of the :math:`\dst` memory descriptor used for primitive creation.

Post-Ops and Attributes
-----------------------

The sum primitive does not support any post-ops or attributes.

Data Types Support
------------------

The sum primitive supports arbitrary data types for source and destination tensors according to the :ref:`Data Types <doxid-dev_guide_data_types>` page.

Data Representation
-------------------

Sources, Destination
++++++++++++++++++++

The sum primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

* Whenever possible do not specify the destination memory format so that the primitive is able to choose the most appropriate one.

* The sum primitive is highly optimized for the cases when all source tensors have same memory format and data type matches the destination tensor data type. For other cases more general but slower code is working. Consider reordering sources to the same data format before the sum primitive.

* Use in-place operations whenever possible (see caveats in General Notes).

Example
~~~~~~~

:ref:`\*\*Sum Primitive Example\*\* <doxid-sum_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Sum <doxid-dev_guide_sum>` primitive.

Key optimizations included in this example:

* Identical memory formats for source (src) and destination (dst) tensors.

