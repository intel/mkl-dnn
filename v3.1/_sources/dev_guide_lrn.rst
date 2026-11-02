.. index:: pair: page; Local Response Normalization (LRN)
.. _doxid-dev_guide_lrn:

Local Response Normalization (LRN)
==================================

:ref:`API Reference <doxid-group__dnnl__api__lrn>`

General
~~~~~~~

The LRN primitive performs a forward or backward local response normalization.

Forward
-------

The LRN operation is defined by the following formulas (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

LRN `across channels <#dnnl_lrn_across_channels>`__ :

.. math::

	\dst(n, c, h, w) = \left\{k + \frac{\alpha}{n_{l}} \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1} (\src(n, c+i, h, w))^2 \right\}^{-\beta} \cdot \src(n, c, h, w),

LRN `within channel <#dnnl_lrn_within_channel>`__ :

.. math::

	\dst(n, c, h, w) = \left\{k + \frac{\alpha}{n_{l}} \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1} \sum\limits_{j=-(n_{l}-1)/2}^{(n_{l}+1)/2-1} (\src(n, c, h+i, w+j))^2 \right\}^{-\beta} \cdot \src(n, c, h, w),

where :math:`n_{l}` is the ``local_size``. Formulas are provided for 2D spatial data case.

Backward
--------

The backward propagation computes :math:`\diffsrc(n, c, h, w)`, based on :math:`\diffdst(n, c, h, w)` and :math:`\src(n, c, h, w)`.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =========================  
Primitive input/output   Execution argument index   
=======================  =========================  
:math:`\src`             DNNL_ARG_SRC               
:math:`\dst`             DNNL_ARG_DST               
workspace                DNNL_ARG_WORKSPACE         
:math:`\diffsrc`         DNNL_ARG_DIFF_SRC          
:math:`\diffdst`         DNNL_ARG_DIFF_DST          
=======================  =========================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. During training, LRN might or might not require a workspace on forward and backward passes. The behavior is implementation specific. Optimized implementations typically require a workspace and use it to save some intermediate results from the forward pass that accelerate computations on the backward pass. To check whether a workspace is required, query the LRN primitive descriptor for the workspace. Success indicates that the workspace is required and its description will be returned.

Data Type Support
-----------------

The LRN primitive supports the following combinations of data types:

===================  =====================  
Propagation          Source / Destination   
===================  =====================  
forward / backward   f32, bf16, f16         
===================  =====================

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check the :ref:`Implementation Limitations <doxid-dev_guide_lrn_1dg_lrn_impl_limits>` section below.
   
   


Data Representation
-------------------

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

Like most other primitives, the LRN primitive expects the following tensors:

========  ==============================================  
Spatial   Source / Destination                            
========  ==============================================  
0D        :math:`N \times C`                              
1D        :math:`N \times C \times W`                     
2D        :math:`N \times C \times H \times W`            
3D        :math:`N \times C \times D \times H \times W`   
========  ==============================================

The LRN primitive is optimized for the following memory formats:

========  ===============  =======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Spatial   Logical tensor   Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
========  ===============  =======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
2D        NCHW             :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` ), :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` ), *optimized^*   
========  ===============  =======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Here optimized^ means the format that :ref:`comes out <doxid-memory_format_propagation_cpp>` of any preceding compute-intensive primitive.

Post-ops and Attributes
-----------------------

The LRN primitive does not support any post-ops or attributes.

:target:`doxid-dev_guide_lrn_1dg_lrn_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Supports only 2D spatial case.

Performance Tips
~~~~~~~~~~~~~~~~

#. For backward propagation, use the same memory format for ``src``, ``diff_dst``, and ``diff_src`` (the format of the ``diff_dst`` and ``diff_src`` are always the same because of the API). Different formats are functionally supported but lead to highly suboptimal performance.

Example
~~~~~~~

:ref:`LRN Primitive Example <doxid-lrn_example_cpp>`

This C++ API demonstrates how to create and execute a :ref:`Local response normalization <doxid-dev_guide_lrn>` primitive in forward training propagation mode.

