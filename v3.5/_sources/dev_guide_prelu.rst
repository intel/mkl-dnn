.. index:: pair: page; PReLU
.. _doxid-dev_guide_prelu:

PReLU
=====

:ref:`API Reference <doxid-group__dnnl__api__prelu>`

General
~~~~~~~

The PReLU primitive (Leaky ReLU with trainable alpha parameter) performs forward or backward operation on data tensor. Weights (alpha) tensor supports broadcast-semantics. Broadcast configuration is assumed based on src and weights dimensions.

Example broadcasts:

===============  =======================  =======================  
broadcast type   src dimensions           weights dimensions       
===============  =======================  =======================  
Channel-shared   :math:`\{n, c, h ,w\}`   :math:`\{1, 1, 1 ,1\}`   
Channel-wise     :math:`\{n, c, h ,w\}`   :math:`\{1, c, 1 ,1\}`   
Whole-tensor     :math:`\{n, c, h ,w\}`   :math:`\{n, c, h ,w\}`   
Shared-axes      :math:`\{n, c, h ,w\}`   :math:`\{n, 1, h ,1\}`   
===============  =======================  =======================

.. note:: 

   Shared-axes indicates broadcast with any combination of shared dimensions.
   
   


Forward
-------

The PReLU operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`. For no broadcast case, results are calculated using formula:

.. math::

	\dst(n, c, h, w) = \begin{cases} \src(n, c, h, w) & \mbox{if } \src(n, c, h, w) > 0 \\ \src(n, c, h, w) \cdot \weights(n, c, h, w) & \mbox{if } \src(n, c, h, w) \leq 0 \end{cases}

Depending on broadcast configuration, result is calculated taking into account shared dimensions of weights tensor.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc` and :math:`\diffweights`. For no broadcast case, results are calculated using formula:

.. math::

	\diffdst(n, c, h, w) &= \begin{cases} \diffdst(n, c, h, w) & \mbox{if } \src(n, c, h, w) > 0 \\ \diffdst(n, c, h, w) \cdot \weights(n, c, h, w) & \mbox{if } \src(n, c, h, w) \leq 0 \end{cases}\\\\ \diff_weights(n, c, h, w) &= \min(\src(n, c, h, w), 0) \cdot \diffdst(n, c, h, w)

Similar to forward propagation, result is calculated taking into account shared dimensions of weights tensor. :math:`\diffweights` results are accumulated according to weights tensor shared dimensions, since :math:`\diffweights` tensor must match :math:`\weights` tensor.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =========================  
Primitive input/output   Execution argument index   
=======================  =========================  
:math:`\src`             DNNL_ARG_SRC               
:math:`\dst`             DNNL_ARG_DST               
:math:`\weights`         DNNL_ARG_WEIGHTS           
:math:`\diffsrc`         DNNL_ARG_DIFF_SRC          
:math:`\diffdst`         DNNL_ARG_DIFF_DST          
:math:`\diffweights`     DNNL_ARG_DIFF_WEIGHTS      
=======================  =========================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

* Prelu primitive requires all input/output tensors to have the same number of dimensions. Dimension sizes can differ however.

* :math:`\weights` tensor dimensions sizes must follow broadcast semantics. Each dimension can either equal corresponding data dimension or equal 1 - to indicate that dimension is shared.

* Prelu primitive requires that :math:`\diffweights` tensor has exact same dimensions sizes as :math:`\weights` tensor, :math:`\diffsrc` as src and :math:`\diffdst` as dst.

* :math:`\weights` tensor can be initialized with format_tag::any primitive will match it to data tensor format.

Data Type Support
-----------------

The PReLU primitive supports the following combinations of data types:

===================  ============================  
Propagation          Source / Destination          
===================  ============================  
forward / backward   f32, s32, bf16, f16, s8, u8   
===================  ============================

Data Representation
-------------------

The PReLU primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Current implementation supports all tensors up to 3D spatial (n, c, d, h, w).

Performance Tips
~~~~~~~~~~~~~~~~

Its recommended to allow PReLU primitive to choose the appropriate weights memory format by passing weights_md with format_tag::any. For best performance, the weights memory format should match data memory format.

Example
~~~~~~~

:ref:`PReLU Primitive Example <doxid-prelu_example_cpp>`

This C++ API example demonstrates how to create and execute an :ref:`PReLU <doxid-dev_guide_prelu>` primitive in forward training propagation mode.

