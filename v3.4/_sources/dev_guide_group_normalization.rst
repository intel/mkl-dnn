.. index:: pair: page; Group Normalization
.. _doxid-dev_guide_group_normalization:

Group Normalization
===================

:ref:`API Reference <doxid-group__dnnl__api__group__normalization>`

General
~~~~~~~

The group normalization primitive performs a forward or backward group normalization operation on tensors with numbers of dimensions equal to 3 or more.

Forward
-------

The group normalization operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

.. math::

	\dst(n, g \cdot C_G + c_g, h, w) = \gamma(g \cdot C_G + c_g) \cdot \frac{\src(n, g \cdot C_G + c_g, h, w) - \mu(n, g)} {\sqrt{\sigma^2(n, g) + \varepsilon}} + \beta(g \cdot C_G + c_g),

where

* :math:`C_G = \frac{C}{G}`,

* :math:`c_g \in [0, C_G).`,

* :math:`\gamma(c), \beta(c)` are optional scale and shift for a channel (see :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` and :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` flags),

* :math:`\mu(n, g), \sigma^2(n, g)` are mean and variance for a group of channels in a batch (see :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag), and

* :math:`\varepsilon` is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and variance are computed at runtime, the following formulas are used:

* :math:`\mu(n, g) = \frac{1}{(C/G)HW} \sum\limits_{c_ghw} \src(n, g \cdot C_G + c_g, h, w)_{}`,

* :math:`\sigma^2(n, g) = \frac{1}{(C/G)HW} \sum\limits_{c_ghw} {}_{} (\src(n, g \cdot C_G + c_g, h, w) - \mu(n, g))^2`.

The :math:`\gamma(c)` and :math:`\beta(c)` tensors are considered learnable.

.. note:: 

   * The group normalization primitive computes population mean and variance and not the sample or unbiased versions that are typically used to compute running mean and variance.
   
   * Using the mean and variance computed by the group normalization primitive, running mean and variance :math:`\hat\mu` and :math:`\hat\sigma^2` can be computed as
     
     .. math::
     
     	\hat\mu := \alpha \cdot \hat\mu + (1 - \alpha) \cdot \mu, \\ \hat\sigma^2 := \alpha \cdot \hat\sigma^2 + (1 - \alpha) \cdot \sigma^2.
   
   


Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* If mean and variance are computed at runtime (i.e., :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` is not set), they become outputs for the propagation kind :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` (because they would be required during the backward propagation) and are not exposed for the propagation kind :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

Backward
--------

The backward propagation computes :math:`\diffsrc(n, c, h, w)`, :math:`\diffgamma(c)^*`, and :math:`\diffbeta(c)^*` based on :math:`\diffdst(n, c, h, w)`, :math:`\src(n, c, h, w)`, :math:`\mu(n, g)`, :math:`\sigma^2(n, g)`, :math:`\gamma(c) ^*`, and :math:`\beta(c) ^*`.

The tensors marked with an asterisk are used only when the primitive is configured to use :math:`\gamma(c)` and :math:`\beta(c)` (i.e., :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are set).

Execution Arguments
~~~~~~~~~~~~~~~~~~~

Depending on the :ref:`flags <doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` and :ref:`propagation kind <doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`, the group normalization primitive requires different inputs and outputs. For clarity, a summary is shown below.

======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ================================================================================================================================================  ===================================================================================================================================================================================  ====================================================================================================================================================  
Flags                                                                                                                                                                                                                                                                                                                                                                                                                                   :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`   :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>`   :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`                                              :ref:`dnnl_backward_data <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>`          
======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ================================================================================================================================================  ===================================================================================================================================================================================  ====================================================================================================================================================  
:ref:`dnnl_normalization_flags_none <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3ab71f2077a94fd4bbc107a09b115a24a4>`                                                                                                                                                                                                                                                                                 *Inputs* : :math:`\src` *Outputs* : :math:`\dst`                                                                                                   *Inputs* : :math:`\src` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                                 *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\diffsrc`                                                                             Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`   
:ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>`                                                                                                                                                                                                                                                                                         *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\dst`                                                                  *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\dst`                                                                 *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\diffsrc`                                                                             Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`   
:ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>`                                                                                                                                                                                                                                                                                                *Inputs* : :math:`\src` , :math:`\gamma` *Outputs* : :math:`\dst`                                                                                  *Inputs* : :math:`\src` , :math:`\gamma` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` *Outputs* : :math:`\diffsrc` , :math:`\diffgamma`                                       Not supported                                                                                                                                         
:ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>`                                                                                                                                                                                                                                                                                                *Inputs* : :math:`\src` , :math:`\beta` *Outputs* : :math:`\dst`                                                                                   *Inputs* : :math:`\src` , :math:`\beta` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                 *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\beta` *Outputs* : :math:`\diffsrc` , :math:`\diffbeta`                                         Not supported                                                                                                                                         
:ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` | :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` | :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>`   *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\dst`                                 *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\dst`                                *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\diffsrc` , :math:`\diffgamma` , :math:`\diffbeta`   Not supported                                                                                                                                         
======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ================================================================================================================================================  ===================================================================================================================================================================================  ====================================================================================================================================================

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive Input/Output          Execution Argument Index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\gamma`                  DNNL_ARG_SCALE                                                                                                                                                     
:math:`\beta`                   DNNL_ARG_SHIFT                                                                                                                                                     
mean ( :math:`\mu` )            DNNL_ARG_MEAN                                                                                                                                                      
variance ( :math:`\sigma^2` )   DNNL_ARG_VARIANCE                                                                                                                                                  
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\diffdst`                DNNL_ARG_DIFF_DST                                                                                                                                                  
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC                                                                                                                                                  
:math:`\diffgamma`              DNNL_ARG_DIFF_SCALE                                                                                                                                                
:math:`\diffbeta`               DNNL_ARG_DIFF_SHIFT                                                                                                                                                
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The different flavors of the primitive are partially controlled by the ``flags`` parameter that is passed to the primitive descriptor creation function (e.g., :ref:`dnnl::group_normalization_forward::primitive_desc() <doxid-structdnnl_1_1group__normalization__forward_1_1primitive__desc>`). Multiple flags can be set using the bitwise OR operator (``|``).

#. For forward propagation, the mean and variance might be either computed at runtime (in which case they are outputs of the primitive) or provided by a user (in which case they are inputs). In the latter case, a user must set the :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag. For the backward propagation, the mean and variance are always input parameters.

#. Both forward and backward propagation support in-place operations, meaning that :math:`\src` can be used as input and output for forward propagation, and :math:`\diffdst` can be used as input and output for backward propagation. In case of an in-place operation, the original data will be overwritten. Note, however, that backward propagation requires the original :math:`\src`, hence the corresponding forward propagation should not be performed in-place.

Data Type Support
-----------------

The operation supports the following combinations of data types:

===================  =====================  =============================  
Propagation          Source / Destination   Mean / Variance / ScaleShift   
===================  =====================  =============================  
forward / backward   f32, bf16, f16         f32                            
forward              s8                     f32                            
===================  =====================  =============================

.. warning:: 

   There might be hardware- or implementation-specific restrictions. Check the :ref:`Implementation Limitations <doxid-dev_guide_group_normalization_1dg_gnorm_impl_limits>` section below.
   
   


Data Representation
-------------------

Mean and Variance
+++++++++++++++++

The mean (:math:`\mu`) and variance (:math:`\sigma^2`) are separate 2D tensors of size :math:`N \times G`.

The format of the corresponding memory object must be :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` (:ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`).

Scale and Shift
+++++++++++++++

If :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are used, the scale (:math:`\gamma`) and shift (:math:`\beta`) are separate 1D tensors of shape :math:`C`.

The format of the corresponding memory object must be :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>` (:ref:`dnnl_a <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a72c401669bf1737439d6c4af17d0be>`).

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

The group normalization primitive expects data to be :math:`N \times C \times SP_n \times \cdots \times SP_0` tensor.

The group normalization primitive is optimized for the following memory formats:

========  ===============  =============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Spatial   Logical tensor   Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
========  ===============  =============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
1D        NCW              :ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` ( :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>` ), :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` ( :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>` )           
2D        NCHW             :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` ), :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` )       
3D        NCDHW            :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>` ( :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>` ), :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` ( :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>` )   
========  ===============  =============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Post-Ops and Attributes
-----------------------

Attributes enable you to modify the behavior of the group normalization primitive. The following attributes are supported by the group normalization primitive:

============  ==========  =======================================================================================  =====================================================================================  ===================================================================================  
Propagation   Type        Operation                                                                                Description                                                                            Restrictions                                                                         
============  ==========  =======================================================================================  =====================================================================================  ===================================================================================  
forward       attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`   Scales the corresponding tensor by the given scale factor(s).                          Supported only for int8 group normalization and one scale per tensor is supported.   
forward       Post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`         Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result       General binary post-op restrictions                                                  
forward       Post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`        Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result.                                                                                        
============  ==========  =======================================================================================  =====================================================================================  ===================================================================================

:target:`doxid-dev_guide_group_normalization_1dg_gnorm_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

Performance Tips
~~~~~~~~~~~~~~~~

#. Mixing different formats for inputs and outputs is functionally supported but leads to highly suboptimal performance.

#. Use in-place operations whenever possible (see caveats in General Notes).

Examples
~~~~~~~~

:ref:`Group Normalization Primitive Example <doxid-group_normalization_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Group Normalization <doxid-dev_guide_group_normalization>` primitive in forward training propagation mode.

Key optimizations included in this example:

* In-place primitive execution;

* Source memory format for an optimized primitive implementation;

