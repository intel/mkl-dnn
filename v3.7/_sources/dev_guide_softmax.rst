.. index:: pair: page; Softmax
.. _doxid-dev_guide_softmax:

Softmax
=======

:ref:`API Reference <doxid-group__dnnl__api__softmax>`

General
~~~~~~~

The softmax primitive performs forward or backward softmax or logsoftmax operation along a particular axis on data with arbitrary dimensions. All other axes are treated as independent (batch).

Forward
-------

In general form, the operation is defined by the following formulas (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`).

Softmax:

.. math::

	\dst(\overline{ou}, c, \overline{in}) = \frac {e^{\src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})}} { \sum\limits_{ic} e^{\src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})} }

Logsoftmax:

.. math::

	\dst(\overline{ou}, c, \overline{in}) = \ln\left({\frac { e^{\src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})} } { \sum\limits_{ic} e^{\src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})} }}\right) = \left(\src(\overline{ou}, c, \overline{in}) - \nu(\overline{ou}, \overline{in})\right) - \ln\left( \sum\limits_{ic} e^{\src(\overline{ou}, ic, \overline{in}) - \nu(\overline{ou}, \overline{in})} \right)

Above

* :math:`c` is the axis over which the operation is computed on,

* :math:`\overline{ou}` is the outermost index (to the left of the axis),

* :math:`\overline{in}` is the innermost index (to the right of the axis), and

* :math:`\nu` is used to produce numerically stable results and defined as:
  
  .. math::
  
  	\nu(\overline{ou}, \overline{in}) = \max\limits_{ic} \src(\overline{ou}, ic, \overline{in})

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc(ou, c, in)`, based on :math:`\diffdst(ou, c, in)` and :math:`\dst(ou, c, in)`.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC                                                                                                                                                  
:math:`\diffdst`                DNNL_ARG_DIFF_DST                                                                                                                                                  
:math:`src scale`               DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC                                                                                                                                
:math:`dst scale`               DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST                                                                                                                                
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. Both forward and backward propagation support in-place operations, meaning that :math:`\src` can be used as input and output for forward propagation, and :math:`\diffdst` can be used as input and output for backward propagation. In case of in-place operation, the original data will be overwritten. This support is limited to cases when data types of :math:`\src` and :math:`\dst` or :math:`\diffsrc` and :math:`\diffdst` are identical.

Post-ops and Attributes
-----------------------

Attributes enable you to modify the behavior of the softmax primitive. The following attributes are supported by the softmax primitive:

============  ==========  ==================================================================================================  =====================================================================================  =======================================================================  
Propagation   Type        Operation                                                                                           Description                                                                            Restrictions                                                             
============  ==========  ==================================================================================================  =====================================================================================  =======================================================================  
forward       attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`              Scales the corresponding tensor by the given scale factor(s).                          Supported only for int8 softmax and one scale per tensor is supported.   
forward       post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`                    Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result       General binary post-op restrictions                                      
forward       Post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`                   Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result.                                                                            
forward       attribute   :ref:`Accumulation mode <doxid-structdnnl_1_1primitive__attr_1a8348fcd2259553c3537194430b7de4f4>`   Defines the implementation's accumulation arithmetic.                                  Only the values ``strict`` , ``relaxed`` , and ``any`` are supported.    
============  ==========  ==================================================================================================  =====================================================================================  =======================================================================

Accumulation Mode
+++++++++++++++++

You can optimize performance of the forward operation when the source and destination floating-point data types of the operation are equal and different from ``f32``. When the destination data type is different from ``f32``, additional memory will be used to accumulate data and store it in the destination memory buffer for a requested data type. Using the additional memory can be opted-out with an accumulation mode setting set to :ref:`relaxed <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a81f32be24a2a62fc472cc43edc97e65b>` or :ref:`any <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a100b8cad7cf2a56f6df78f171f97a1ec>`, which will use the precision of destination data type to accumulate intermediate results directly into the destination memory buffer. This performance optimization, however, results in in a minor decrease in accuracy. Depending on the actual data, the difference between ``strict`` and ``relaxed`` accumulation can reach several units in the last piece (ulps).

Data Type Support
-----------------

The softmax primitive supports the following combinations of data types:

============  ============================  ============================  
Propagation   Source                        Destination                   
============  ============================  ============================  
forward       f32, f64, bf16, f16, u8, s8   f32, f64, bf16, f16, u8, s8   
backward      f32, f64, bf16, f16           f32, f64, bf16, f16           
============  ============================  ============================

Data Representation
-------------------

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

The softmax primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions. However, the softmax axis is typically referred to as channels (hence in formulas :math:`c` is used).

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

#. Use in-place operations whenever possible.

Example
~~~~~~~

:ref:`Softmax Primitive Example <doxid-softmax_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Softmax <doxid-dev_guide_softmax>` primitive in forward training propagation mode.

Key optimizations included in this example:

* In-place primitive execution;

* Softmax along axis 1 (C) for 2D tensors.

