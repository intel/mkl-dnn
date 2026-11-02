.. index:: pair: page; Resampling
.. _doxid-dev_guide_resampling:

Resampling
==========

:ref:`API reference <doxid-group__dnnl__api__resampling>`

General
~~~~~~~

The resampling primitive computes forward or backward resampling operation on 1D, 2D, or 3D spatial data. Resampling performs spatial scaling of original tensor using one of the supported interpolation algorithms:

* Nearest Neighbor

* Linear (or Bilinear for 2D spatial tensor, Trilinear for 3D spatial tensor).

Resampling operation is defined by the source tensor and scaling factors in each spatial dimension. Upsampling and downsampling are the alternative terms for resampling that are used when all scaling factors are greater (upsampling) or less (downsampling) than one.

The resampling operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

Let :math:`\src` and :math:`\dst` be :math:`N \times C \times IH \times IW` and :math:`N \times C \times OH \times OW` tensors respectively. Let :math:`F_h = \frac{OH}{IH}` and :math:`F_w = \frac{OW}{IW}` define scaling factors in each spatial dimension.

The following formulas show how oneDNN computes resampling for nearest neighbor and bilinear interpolation methods. To further simplify the formulas, we assume the following: :math:`\src(n, ic, ih, iw) = \begin{cases} \src(n, ic, ih, 0), & \text{if}\ iw < 0 \\ \src(n, ic, ih, iw), & \text{if}\ IW - 1 \geq iw \geq 0 \\ \src(n, ic, ih, IW - 1), & \text{if}\ iw > IW - 1 \end{cases}`

Same assumptions apply for :math:`ih`. Definitions of :math:`ih` and :math:`iw` are provided below with a correspondent algorithm.

Forward
-------

Nearest Neighbor Resampling
+++++++++++++++++++++++++++

.. math::

	\dst(n, c, oh, ow) = \src(n, c, ih, iw)

where

* :math:`ih = [\frac{oh + 0.5} {F_h} - 0.5]`,

* :math:`iw = [\frac{ow + 0.5} {F_w} - 0.5]`.

Bilinear Resampling
+++++++++++++++++++

.. math::

	\dst(n, c, oh, ow) = \src(n, c, ih_0, iw_0) \cdot (1 - W_{ih}) \cdot (1 - W_{iw}) + \\ \src(n, c, ih_1, iw_0) \cdot W_{ih} \cdot (1 - W_{iw}) + \\ \src(n, c, ih_0, iw_1) \cdot (1 - W_{ih}) \cdot W_{iw} + \\ \src(n, c, ih_1, iw_1) \cdot W_{ih} \cdot W_{iw} \\

where

* :math:`ih_0 = \left\lfloor{\frac {oh + 0.5} {F_h} - 0.5}\right\rfloor`,

* :math:`ih_1 = \left\lceil {\frac {oh + 0.5} {F_h} - 0.5}\right\rceil`,

* :math:`iw_0 = \left\lfloor{\frac {ow + 0.5} {F_w} - 0.5}\right\rfloor`,

* :math:`iw_1 = \left\lceil {\frac {ow + 0.5} {F_w} - 0.5}\right\rceil`,

* :math:`W_{ih} = \frac{oh + 0.5}{F_h} - 0.5 - ih_0`,

* :math:`W_{iw} = \frac{ow + 0.5}{F_w} - 0.5 - iw_0`.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc` based on :math:`\diffdst`.

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
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. Resampling implementation supports data with arbitrary data tag (nchw, nhwc, nChw16c, etc.) but memory tags for ``src`` and ``dst`` are expected to be the same. Resampling primitive supports ``dst`` and ``diff_src`` memory tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` and can define destination format based on source format.

#. Resampling primitive descriptor can be created by specifying the source and destination memory descriptors, only the source descriptor and floating point factors, or the source and destination memory descriptors and factors. In case when user does not provide the destination descriptor, the destination dimensions are deduced using the factors: :math:`output\_spatial\_size = \left\lfloor{ \frac{input\_spatial\_size} {F} }\right\rfloor`.

.. note:: 

   Implementation of resampling algorithm uses factors as defined by the relation :math:`F = \frac{output\_spatial\_ size} { input\_spatial\_size}` that do not necessarily equal to the ones passed by the user.
   
   


Data Types
----------

Resampling primitive supports the following combination of data types for source and destination memory objects:

===================  ============================  ============================  
Propagation          Source                        Destination                   
===================  ============================  ============================  
forward / backward   f32, bf16, f16, s32, s8, u8   f32, s32, bf16, s8, u8, f16   
===================  ============================  ============================

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

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. No primitive specific limitations. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

Performance Tips
~~~~~~~~~~~~~~~~

N/A

Example
~~~~~~~

:ref:`Resampling Primitive Example <doxid-resampling_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Resampling <doxid-dev_guide_resampling>` primitive in forward training propagation mode.

