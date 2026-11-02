.. index:: pair: page; Concat
.. _doxid-dev_guide_concat:

Concat
======

:ref:`API Reference <doxid-group__dnnl__api__concat>`

General
~~~~~~~

The concat primitive concatenates :math:`N` tensors over ``concat_dimension`` (here designated :math:`C`) and is defined as (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(\overline{ou}, c, \overline{in}) = \src_i(\overline{ou}, c', \overline{in}),

where :math:`c = C_1 + .. + C_{i-1} {}_{} + c'`.

The concat primitive does not have a notion of forward or backward propagation. The backward propagation for the concatenation operation is simply an identity operation.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =========================  
Primitive input/output   Execution argument index   
=======================  =========================  
:math:`\src`             DNNL_ARG_MULTIPLE_SRC      
:math:`\dst`             DNNL_ARG_DST               
=======================  =========================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The :math:`\dst` memory format can be either specified by a user or derived by the primitive. The recommended way is to allow the primitive to choose the most appropriate format.

#. The concat primitive requires all source and destination tensors to have the same shape except for the ``concat_dimension``. The destination dimension for the ``concat_dimension`` must be equal to the sum of the ``concat_dimension`` dimensions of the sources (i.e. :math:`C = \sum_i C_i`). Implicit broadcasting is not supported.

Data Types Support
------------------

The concat primitive supports arbitrary data types for source and destination tensors according to the :ref:`Data Types <doxid-dev_guide_data_types>` page. However, it is required that all source tensors are of the same data type (but not necessarily matching the data type of the destination tensor).

Data Representation
-------------------

The concat primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.

Post-Ops and Attributes
-----------------------

==========  =======================================================================================  ====================================================================  ============================================================  
Type        Operation                                                                                Description                                                           Res                                                           
==========  =======================================================================================  ====================================================================  ============================================================  
Attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`   Scales the corresponding input tensor by the given scale factor(s).   Only one scale per tensor is supported. Input tensors only.   
==========  =======================================================================================  ====================================================================  ============================================================

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The primitive works with several memory formats, such as plain formats :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`, :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>`, and blocked formats :ref:`dnnl_nChw16c <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa7847819b4fb840d2db20796bc607a5c>`, :ref:`dnnl_nCdhw8c <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dabacffa20b5188cda4d5f86e2e10d2572>` that appear in convolutions. The primitive does not support non-blocked formats that are typically used in prepacked weights, such as:
   
   * :ref:`Winograd <doxid-dev_guide_convolution>` format :ref:`dnnl_format_kind_opaque <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da44f131bbbd690fd1f4f94b47279657fe>`,
   
   * :ref:`RNN <doxid-dev_guide_rnn>` format :ref:`dnnl_format_kind_opaque <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da44f131bbbd690fd1f4f94b47279657fe>`, or
   
   * In some cases for blocked format with attached :ref:`compensation <doxid-dev_guide_int8_computations_1dg_i8_comp_s12>` that is used in ``s8s8`` convolutions (see :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>`).

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.



#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

#. Whenever possible, avoid specifying the destination memory format so that the primitive is able to choose the most appropriate one.

#. The concat primitive is highly optimized for the cases in which all source tensors have same memory format and data type matches the destination tensor data type. For other cases, more general but slower code is working. Consider reordering sources to the same data format before using the concat primitive.

Example
~~~~~~~

:ref:`Concat Primitive Example <doxid-concat_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Concat <doxid-dev_guide_concat>` primitive.

Key optimizations included in this example:

* Identical source (src) memory formats.

* Creation of optimized memory format for destination (dst) from the primitive descriptor

