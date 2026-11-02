.. index:: pair: page; Inner Product
.. _doxid-dev_guide_inner_product:

Inner Product
=============

:ref:`API Reference <doxid-group__dnnl__api__inner__product>`

General
~~~~~~~

The inner product primitive (sometimes called fully connected) treats each activation in the minibatch as a vector and computes its product with a weights 2D tensor producing a 2D tensor as an output.

Forward
-------

More precisely, let :math:`\src`, :math:`\weights`, :math:`\bias` and :math:`\dst` be :math:`N \times IC`, :math:`OC \times IC`, :math:`OC`, and :math:`N \times OC` tensors, respectively (variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`). Then:

.. math::

	\dst(n, oc) = \bias(oc) + \sum_{ic=0}^{IC-1} \src(n, ic) \cdot \weights(oc, ic)

In cases where the :math:`\src` and :math:`\weights` tensors have spatial dimensions, they are flattened to 2D. For example, if they are 4D :math:`N \times IC' \times IH \times IW` and :math:`OC \times IC' \times KH \times KW` tensors, then the formula above is applied with :math:`IC = IC' \cdot IH \cdot IW`. In such cases, the :math:`\src` and :math:`\weights` tensors must have equal spatial dimensions (e.g. :math:`KH = IH` and :math:`KW = IW` for 4D tensors).

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>` and :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc` based on :math:`\diffdst` and :math:`\weights`.

The weights update computes :math:`\diffweights` and :math:`\diffbias` based on :math:`\diffdst` and :math:`\src`.

.. note:: 

   The optimized memory formats :math:`\src` and :math:`\weights` might be different on forward propagation, backward propagation, and weights update.
   
   


Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\weights`                DNNL_ARG_WEIGHTS                                                                                                                                                   
:math:`\bias`                   DNNL_ARG_BIAS                                                                                                                                                      
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC                                                                                                                                                  
:math:`\diffweights`            DNNL_ARG_DIFF_WEIGHTS                                                                                                                                              
:math:`\diffbias`               DNNL_ARG_DIFF_BIAS                                                                                                                                                 
:math:`\diffdst`                DNNL_ARG_DIFF_DST                                                                                                                                                  
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

N/A.

Data Types
----------

Inner product primitive supports the following combination of data types for source, destination, weights, and bias:

===================  ==========  ==========  =======================  =======================  
Propagation          Source      Weights     Destination              Bias                     
===================  ==========  ==========  =======================  =======================  
forward / backward   f32         f32         f32                      f32                      
forward              f16         f16         f32, f16, u8, s8         f16, f32                 
forward              u8, s8      s8          u8, s8, s32, bf16, f32   u8, s8, s32, bf16, f32   
forward              bf16        bf16        f32, bf16                f32, bf16                
backward             f32, bf16   bf16        bf16                                              
backward             f32, f16    f16         f16                                               
weights update       bf16        f32, bf16   bf16                     f32, bf16                
weights update       f16         f32, f16    f16                      f32, f16                 
===================  ==========  ==========  =======================  =======================

Data Representation
-------------------

Like other CNN primitives, the inner product primitive expects the following tensors:

========  ==============================================  ===================  ===================================================  
Spatial   Source                                          Destination          Weights                                              
========  ==============================================  ===================  ===================================================  
1D        :math:`N \times C \times W`                     :math:`N \times C`   :math:`OC \times IC \times KW`                       
2D        :math:`N \times C \times H \times W`            :math:`N \times C`   :math:`OC \times IC \times KH \times KW`             
3D        :math:`N \times C \times D \times H \times W`   :math:`N \times C`   :math:`OC \times IC \times KD \times KH \times KW`   
========  ==============================================  ===================  ===================================================

Memory format of data and weights memory objects is critical for inner product primitive performance. In the oneDNN programming model, inner product primitive is one of the few primitives that support the placeholder format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` (shortened to ``any`` from now on) and can define data and weight memory objects formats based on the primitive parameters. When using ``any`` it is necessary to first create an inner product primitive descriptor and then query it for the actual data and weight memory objects formats.

The table below shows the combinations of memory formats the inner product primitive is optimized for. For the destination tensor (which is always :math:`N \times C`) the memory format is always :ref:`dnnl::memory::format_tag::nc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa1e7342845e24eb3b5b3554490da1c128>` (:ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`).

====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ============  
Source / Destination                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Weights                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               Limitations   
====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ============  
``any``                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ``any``                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               N/A           
:ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` , :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` , :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` , :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`   ``any``                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               N/A           
:ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` , :ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` , :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` , :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>`   ``any``                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               N/A           
:ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` , :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` , :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` , :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`   :ref:`dnnl_io <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da46de139a50746a1a22d8d33c60d9081b>` , :ref:`dnnl_wio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93eecc25f8ab1b07604b632401aa28e5>` , :ref:`dnnl_hwio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f4c7bd98c6d53fb3b69e1c8df0a80f6>` , :ref:`dnnl_dhwio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae4885779f955beeddc25443a3f8c2a63>`   N/A           
:ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` , :ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` , :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` , :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>`   :ref:`dnnl_oi <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daee91ff6fadfe8c5494be1595fd253c4e>` , :ref:`dnnl_oiw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae6bd836b8515857585336a0921e35663>` , :ref:`dnnl_oihw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11176ff202375dcd0d06e2fba5f8a8e0>` , :ref:`dnnl_oidhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da18e605a5f6afe9995961fc21fddf314e>`   N/A           
====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ============

Post-Ops and Attributes
-----------------------

Post-ops and attributes enable you to modify the behavior of the inner product primitive by chaining certain operations after the inner product operation. The following post-ops are supported by inner product primitives:

============  ==========  =======================================================================================  ====================================================================================  ====================================  
Propagation   Type        Operation                                                                                Description                                                                           Restrictions                          
============  ==========  =======================================================================================  ====================================================================================  ====================================  
forward       attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`   Scales the result of inner product by given scale factor(s)                           int8 inner products only              
forward       post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`        Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result                                         
forward       post-op     :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`            Adds the operation result to the destination tensor instead of overwriting it                                               
forward       post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`         Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result      General binary post-op restrictions   
============  ==========  =======================================================================================  ====================================================================================  ====================================

The following masks are supported by the primitive:

* 0, which applies one scale value to an entire tensor, and

* 1, which applies a scale value per output channel for ``DNNL_ARG_WEIGHTS`` argument.

When scales masks are specified, the user must provide the corresponding scales as additional input memory objects with argument ``DNNL_ARG_ATTR_SCALES | DNNL_ARG_${MEMORY_INDEX}`` during the execution stage.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Check :ref:`Data Types <doxid-dev_guide_data_types>`.

#. The CPU engine does not support ``u8`` or ``s8`` data type for ``dst`` with ``f16`` ``src`` and ``weights``.

Performance Tips
~~~~~~~~~~~~~~~~

* Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for source, weights, and destinations memory format tags when create an inner product primitive to allow the library to choose the most appropriate memory format.

Example
~~~~~~~

:ref:`Inner Product Primitive Example <doxid-inner_product_example_cpp>`

This C++ API example demonstrates how to create and execute an :ref:`Inner Product <doxid-dev_guide_inner_product>` primitive.

Key optimizations included in this example:

* Primitive attributes with fused post-ops;

* Creation of optimized memory format from the primitive descriptor.

