.. index:: pair: page; Pooling
.. _doxid-dev_guide_pooling:

Pooling
=======

:ref:`API Reference <doxid-group__dnnl__api__pooling>`

General
~~~~~~~

The pooling primitive performs forward or backward max or average pooling operation on 1D, 2D, or 3D spatial data.

Forward
-------

The pooling operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

Max pooling:

.. math::

	\dst(n, c, oh, ow) = \max\limits_{kh, kw} \left( \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L) \right)

Average pooling:

.. math::

	\dst(n, c, oh, ow) = \frac{1}{DENOM} \sum\limits_{kh, kw} \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L)

Here output spatial dimensions are calculated similarly to how they are done in :ref:`Convolution <doxid-dev_guide_convolution>`.

Average pooling supports two algorithms:

* :ref:`dnnl_pooling_avg_include_padding <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac13a4cc7c0dc1edfcbf1bac23391d5cb>`, in which case :math:`DENOM = KH \cdot KW`,

* :ref:`dnnl_pooling_avg_exclude_padding <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a00156580493fd7c2f4cdbaaf9fcbde79>`, in which case :math:`DENOM` equals to the size of overlap between an averaging window and images.

TODO: a picture would be nice here.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* Max pooling requires a ``workspace`` for the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` propagation kind, and does not require it for :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` (see details below).

Backward
--------

The backward propagation computes :math:`\diffsrc(n, c, h, w)`, based on :math:`\diffdst(n, c, h, w)` and (in case of max pooling) ``workspace``.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
workspace                       DNNL_ARG_WORKSPACE                                                                                                                                                 
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC                                                                                                                                                  
:math:`\diffdst`                DNNL_ARG_DIFF_DST                                                                                                                                                  
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. During training, max pooling requires a workspace on forward (:ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>`) and backward passes to save indices where a maximum was found. The workspace format is opaque, and the indices cannot be restored from it. However, one can use backward pooling to perform up-sampling (used in some detection topologies). The workspace can be created via ``workspace_desc()`` from the pooling primitive descriptor.

#. A user can use memory format tag :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` for ``dst`` memory descriptor when creating pooling forward propagation. The library would derive the appropriate format from the ``src`` memory descriptor. However, the ``src`` itself must be defined. Similarly, a user can use memory format tag :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` for the ``diff_src`` memory descriptor when creating pooling backward propagation.

Data Type Support
-----------------

The pooling primitive supports the following combinations of data types:

===================  =======  ============  =======================================================  
Propagation          Source   Destination   Accumulation data type (used for average pooling only)   
===================  =======  ============  =======================================================  
forward / backward   f32      f32           f32                                                      
forward / backward   f64      f64           f64                                                      
forward / backward   bf16     bf16          bf16                                                     
forward / backward   f16      f16           f32                                                      
forward              s8       s8            s32                                                      
forward              u8       u8            s32                                                      
forward              s32      s32           s32                                                      
forward inference    s8       u8            s32                                                      
forward inference    u8       s8            s32                                                      
forward inference    s8       f16           f32                                                      
forward inference    u8       f16           f32                                                      
forward inference    f16      s8            f32                                                      
forward inference    f16      u8            f32                                                      
forward inference    s8       f32           f32                                                      
forward inference    u8       f32           f32                                                      
forward inference    f32      s8            f32                                                      
forward inference    f32      u8            f32                                                      
===================  =======  ============  =======================================================

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check :ref:`Implementation Limitations <doxid-dev_guide_pooling_1dg_pool_impl_limits>` section below.
   
   


Data Representation
-------------------

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

Like other CNN primitives, the pooling primitive expects data to be an :math:`N \times C \times W` tensor for the 1D spatial case, an :math:`N \times C \times H \times W` tensor for the 2D spatial case, and an :math:`N \times C \times D \times H \times W` tensor for the 3D spatial case.

The pooling primitive is optimized for the following memory formats:

========  ===============  ============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Spatial   Logical tensor   Data type     Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
========  ===============  ============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
1D        NCW              f32           :ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` ( :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>` ), :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` ( :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>` ), *optimized^*           
1D        NCW              s32, s8, u8   :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` ( :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>` ), *optimized^*                                                                                                                                                                                                                                                              
2D        NCHW             f32           :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` ), :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` ), *optimized^*       
2D        NCHW             s32, s8, u8   :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` ), *optimized^*                                                                                                                                                                                                                                                            
3D        NCDHW            f32           :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>` ( :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>` ), :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` ( :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>` ), *optimized^*   
3D        NCDHW            s32, s8, u8   :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` ( :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>` ), *optimized^*                                                                                                                                                                                                                                                          
========  ===============  ============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Here optimized^ means the format that :ref:`comes out <doxid-memory_format_propagation_cpp>` of any preceding compute-intensive primitive.

Post-Ops and Attributes
-----------------------

============  ========  ==================================================================================  =====================================================================================  ====================================  
Propagation   Type      Operation                                                                           Description                                                                            Restrictions                          
============  ========  ==================================================================================  =====================================================================================  ====================================  
Forward       Post-op   :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`    Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result       General binary post-op restrictions   
Forward       Post-op   :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`   Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result.                                         
============  ========  ==================================================================================  =====================================================================================  ====================================

:target:`doxid-dev_guide_pooling_1dg_pool_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. CPU
   
   * Different data types of source and destination in forward inference are not supported.

#. GPU
   
   * :ref:`dnnl_pooling_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7>` for f64 data type will return ``-FLT_MAX`` as an output value instead of ``-DBL_MAX`` in scenarios when pooling kernel is applied to a completely padded area.

Performance Tips
~~~~~~~~~~~~~~~~

N/A

Example
~~~~~~~

:ref:`Pooling Primitive Example <doxid-pooling_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Pooling <doxid-dev_guide_pooling>` primitive in forward training propagation mode.

