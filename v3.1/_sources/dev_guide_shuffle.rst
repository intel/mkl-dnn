.. index:: pair: page; Shuffle
.. _doxid-dev_guide_shuffle:

Shuffle
=======

:ref:`API Reference <doxid-group__dnnl__api__shuffle>`

General
~~~~~~~

The shuffle primitive shuffles data along the shuffle axis (here is designated as :math:`C`) with the group parameter :math:`G`. Namely, the shuffle axis is thought to be a 2D tensor of size :math:`(\frac{C}{G} \times G)` and it is being transposed to :math:`(G \times \frac{C}{G})`. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

The formal definition is shown below:

Forward
-------

.. math::

	\dst(\overline{ou}, c, \overline{in}) = \src(\overline{ou}, c', \overline{in})

where

* :math:`c` dimension is called a shuffle axis,

* :math:`G` is a ``group_size``,

* :math:`\overline{ou}` is the outermost indices (to the left from shuffle axis),

* :math:`\overline{in}` is the innermost indices (to the right from shuffle axis), and

* :math:`c'` and :math:`c` relate to each other as define by the system:
  
  .. math::
  
  	\begin{cases} c &= u + v\frac{C}{G}, \\ c' &= uG + v, \\ \end{cases}

Here, :math:`0 \leq u < \frac{C}{G}` and :math:`0 \leq v < G`.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc(ou, c, in)`, based on :math:`\diffdst(ou, c, in)`.

Essentially, backward propagation is the same as forward propagation with :math:`g` replaced by :math:`C / g`.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =========================  
Primitive input/output   Execution argument index   
=======================  =========================  
:math:`\src`             DNNL_ARG_SRC               
:math:`\dst`             DNNL_ARG_DST               
:math:`\diffsrc`         DNNL_ARG_DIFF_SRC          
:math:`\diffdst`         DNNL_ARG_DIFF_DST          
=======================  =========================

Data Types
~~~~~~~~~~

The shuffle primitive supports the following combinations of data types:

===================  =====================  
Propagation          Source / Destination   
===================  =====================  
forward / backward   f32, bf16, f16         
forward              s32, s8, u8            
===================  =====================

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check the :ref:`Implementation Limitations <doxid-dev_guide_shuffle_1dg_shuffle_impl_limits>` section below.
   
   


Data Layouts
~~~~~~~~~~~~

The shuffle primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions. However, the shuffle axis is typically referred to as channels (hence in formulas we use :math:`c`).

Shuffle operation typically appear in CNN topologies. Hence, in the library the shuffle primitive is optimized for the corresponding memory formats:

========  ===============  =============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Spatial   Logical tensor   Shuffle Axis   Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
========  ===============  =============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
2D        NCHW             1 (C)          :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` ), :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` ), *optimized^*       
3D        NCDHW            1 (C)          :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>` ( :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>` ), :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` ( :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>` ), *optimized^*   
========  ===============  =============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Here optimized^ means the format that :ref:`comes out <doxid-memory_format_propagation_cpp>` of any preceding compute-intensive primitive.

Post-Ops and Attributes
-----------------------

The shuffle primitive does not support any post-ops or attributes.

:target:`doxid-dev_guide_shuffle_1dg_shuffle_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

N/A

Example
~~~~~~~

:ref:`Shuffle Primitive Example <doxid-shuffle_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Shuffle <doxid-dev_guide_shuffle>` primitive.

Key optimizations included in this example:

* Shuffle along axis 1 (channels).

