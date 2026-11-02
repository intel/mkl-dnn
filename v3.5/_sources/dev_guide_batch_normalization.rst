.. index:: pair: page; Batch Normalization
.. _doxid-dev_guide_batch_normalization:

Batch Normalization
===================

:ref:`API Reference <doxid-group__dnnl__api__batch__normalization>`

General
~~~~~~~

The batch normalization primitive performs a forward or backward batch normalization operation on tensors with number of dimensions equal to 2 or more.

Forward
-------

The batch normalization operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

.. math::

	\dst(n, c, h, w) = \gamma(c) \cdot \frac{\src(n, c, h, w) - \mu(c)} {\sqrt{\sigma^2(c) + \varepsilon}} + \beta(c),

where

* :math:`\gamma(c), \beta(c)` are optional scale and shift for a channel (see :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` and :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` flags),

* :math:`\mu(c), \sigma^2(c)` are mean and variance for a channel (see :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag), and

* :math:`\varepsilon` is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and variance are computed at runtime, the following formulas are used:

* :math:`\mu(c) = \frac{1}{NHW} \sum\limits_{nhw} \src(n, c, h, w)_{}`,

* :math:`\sigma^2(c) = \frac{1}{NHW} \sum\limits_{nhw} {}_{} (\src(n, c, h, w) - \mu(c))^2`.

The :math:`\gamma(c)` and :math:`\beta(c)` tensors are considered learnable.

In training mode, the primitive also optionally supports:

* fusion with ReLU activation with zero negative slope applied to the result (see :ref:`dnnl_fuse_norm_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>` flag).

* fusion with binary addition and ReLU activation with zero negative slope applied to the result (see :ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>` flag).

.. note:: 

   * The batch normalization primitive computes population mean and variance and not the sample or unbiased versions that are typically used to compute running mean and variance.
   
   * Using the mean and variance computed by the batch normalization primitive, running mean and variance :math:`\hat\mu` and :math:`\hat\sigma^2` can be computed as
     
     .. math::
     
     	\hat\mu := \alpha \cdot \hat\mu + (1 - \alpha) \cdot \mu, \\ \hat\sigma^2 := \alpha \cdot \hat\sigma^2 + (1 - \alpha) \cdot \sigma^2.
   
   


Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* If mean and variance are computed at runtime (i.e., :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` is not set), they become outputs for the propagation kind :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` (because they would be required during the backward propagation) and are not exposed for the propagation kind :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

* If batch normalization is created with ReLU fusion (i.e., :ref:`dnnl_fuse_norm_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>` or :ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>` are set), for the propagation kind :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` the primitive would produce a ``workspace`` memory as one extra output. This memory is required to compute the backward propagation. When the primitive is executed with propagation kind :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`, the workspace is not produced. Behavior would be the same as creating a batch normalization primitive with ReLU as a post-op (see section below).

Backward
--------

The backward propagation computes :math:`\diffsrc(n, c, h, w)`, :math:`\diffgamma(c)^*`, and :math:`\diffbeta(c)^*` based on :math:`\diffdst(n, c, h, w)`, :math:`\src(n, c, h, w)`, :math:`\mu(c)`, :math:`\sigma^2(c)`, and :math:`\gamma(c) ^*`.

The tensors marked with an asterisk are used only when the primitive is configured to use :math:`\gamma(c)` and :math:`\beta(c)` (i.e., :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are set).

Execution Arguments
~~~~~~~~~~~~~~~~~~~

Depending on the :ref:`flags <doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` and :ref:`propagation kind <doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`, the batch normalization primitive requires different inputs and outputs. For clarity, a summary is shown below.

======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ==========================================================================================================================================================================================================================================  ==============================================================================================================================================================================================================================================  ===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Flags                                                                                                                                                                                                                                                                                                                                                                                                                                   :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`   :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>`                                                                                             :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`                                                                                                         :ref:`dnnl_backward_data <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>`                                                                                                                                                                                                                                                                                                                                                     
======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ==========================================================================================================================================================================================================================================  ==============================================================================================================================================================================================================================================  ===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
:ref:`dnnl_normalization_flags_none <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3ab71f2077a94fd4bbc107a09b115a24a4>`                                                                                                                                                                                                                                                                                 *Inputs* : :math:`\src` *Outputs* : :math:`\dst`                                                                                                   *Inputs* : :math:`\src` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                                                                                                                           *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\diffsrc`                                                                                                                                        Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`                                                                                                                                                                                                                                                                                                                                              
:ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>`                                                                                                                                                                                                                                                                                         *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\dst`                                                                  *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\dst`                                                                                                                                                           *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` *Outputs* : :math:`\diffsrc`                                                                                                                                        Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`                                                                                                                                                                                                                                                                                                                                              
:ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>`                                                                                                                                                                                                                                                                                                *Inputs* : :math:`\src` , :math:`\gamma` *Outputs* : :math:`\dst`                                                                                  *Inputs* : :math:`\src` , :math:`\gamma` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                                                                                                          *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` *Outputs* : :math:`\diffsrc` , :math:`\diffgamma`                                                                                                  Not supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
:ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>`                                                                                                                                                                                                                                                                                                *Inputs* : :math:`\src` , :math:`\beta` *Outputs* : :math:`\dst`                                                                                   *Inputs* : :math:`\src` , :math:`\beta` *Outputs* : :math:`\dst` , :math:`\mu` , :math:`\sigma^2`                                                                                                                                           *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\beta` *Outputs* : :math:`\diffsrc` , :math:`\diffbeta`                                                                                                    Not supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
:ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` | :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` | :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>`   *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\dst`                                 *Inputs* : :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\dst`                                                                                                                          *Inputs* : :math:`\diffdst` , :math:`\src` , :math:`\mu` , :math:`\sigma^2` , :math:`\gamma` , :math:`\beta` *Outputs* : :math:`\diffsrc` , :math:`\diffgamma` , :math:`\diffbeta`                                                              Not supported                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
``flags`` | :ref:`dnnl_fuse_norm_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>`                                                                                                                                                                                                                                                                               *Inputs* : same as with ``flags`` *Outputs* : same as with ``flags``                                                                               *Inputs* : same as with ``flags`` *Outputs* : same as with ``flags`` , :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>`                                                *Inputs* : same as with ``flags`` , :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>` *Outputs* : same as with ``flags``                                                    Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>` if ``flags`` do not contain :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` ; not supported otherwise   
``flags`` | :ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>`                                                                                                                                                                                                                                                                           *Inputs* : same as with ``flags`` and :math:`\src_1` for fused binary addition *Outputs* : same as with ``flags``                                  *Inputs* : same as with ``flags`` and :math:`\src_1` for fused binary addition *Outputs* : same as with ``flags`` , :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>`   *Inputs* : same as with ``flags`` , :ref:`Workspace <doxid-dev_guide_inference_and_training_aspects_1dev_guide_inference_and_training_aspects_workspace>` *Outputs* : same as with ``flags`` and :math:`\diffsrc_1` for fused binary addition   Same as for :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>` if ``flags`` do not contain :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` ; not supported otherwise   
======================================================================================================================================================================================================================================================================================================================================================================================================================================  =================================================================================================================================================  ==========================================================================================================================================================================================================================================  ==============================================================================================================================================================================================================================================  ===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =========================  
Primitive Input/Output          Execution Argument Index   
==============================  =========================  
:math:`\src`                    DNNL_ARG_SRC               
:math:`\src_1`                  DNNL_ARG_SRC_1             
:math:`\gamma`                  DNNL_ARG_SCALE             
:math:`\beta`                   DNNL_ARG_SHIFT             
mean ( :math:`\mu` )            DNNL_ARG_MEAN              
variance ( :math:`\sigma^2` )   DNNL_ARG_VARIANCE          
:math:`\dst`                    DNNL_ARG_DST               
workspace                       DNNL_ARG_WORKSPACE         
:math:`\diffdst`                DNNL_ARG_DIFF_DST          
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC          
:math:`\diffsrc_1`              DNNL_ARG_DIFF_SRC_1        
:math:`\diffgamma`              DNNL_ARG_DIFF_SCALE        
:math:`\diffbeta`               DNNL_ARG_DIFF_SHIFT        
==============================  =========================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The different flavors of the primitive are partially controlled by the ``flags`` parameter that is passed to the primitive descriptor creation function (e.g., :ref:`dnnl::batch_normalization_forward::primitive_desc() <doxid-structdnnl_1_1batch__normalization__forward_1_1primitive__desc>`). Multiple flags can be set using the bitwise OR operator (``|``).

#. For forward propagation, the mean and variance might be either computed at runtime (in which case they are outputs of the primitive) or provided by a user (in which case they are inputs). In the latter case, a user must set the :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag. For the backward propagation, the mean and variance are always input parameters.

#. Both forward and backward propagation support in-place operations, meaning that :math:`\src` can be used as input and output for forward propagation, and :math:`\diffdst` can be used as input and output for backward propagation. In case of an in-place operation, the original data will be overwritten. Note, however, that backward propagation requires original :math:`\src`, hence the corresponding forward propagation should not be performed in-place.

#. As mentioned above, the batch normalization primitive can be fused with binary addition and ReLU activation (:ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>`). In this case:
   
   * on the forward propagation the primitive has one additional input, :math:`\src_1`, that should have memory descriptor equal to primitive ``dst_desc`` memory descriptor.
   
   * on the backward propagation the primitive has one additional output, :math:`\diffsrc_1`, that should have memory descriptor equal to primitive ``diff_dst_desc`` memory descriptor.

#. As mentioned above, the batch normalization primitive can be fused with ReLU activation (:ref:`dnnl_fuse_norm_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>`) or binary addition and ReLU activation (:ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>`) even in the training mode. In this case, on the forward propagation the primitive has one additional output, ``workspace``, that should be passed during the backward propagation.

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

   There might be hardware- or implementation-specific restrictions. Check the :ref:`Implementation Limitations <doxid-dev_guide_batch_normalization_1dg_bnorm_impl_limits>` section below.
   
   


Data Representation
-------------------

Mean and Variance
+++++++++++++++++

The mean (:math:`\mu`) and variance (:math:`\sigma^2`) are separate 1D tensors of size :math:`C`.

The format of the corresponding memory object must be :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>` (:ref:`dnnl_a <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7a72c401669bf1737439d6c4af17d0be>`).

Scale and Shift
+++++++++++++++

If :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are used, the scale (:math:`\gamma`) and shift (:math:`\beta`) are separate 1D tensors of shape :math:`C`.

The format of the corresponding memory object must be :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` (:ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>`).

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

Like other CNN primitives, the batch normalization primitive expects data to be :math:`N \times C \times SP_n \times \cdots \times SP_0` tensor.

The batch normalization primitive is optimized for the following memory formats:

========  ===============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Spatial   Logical tensor   Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
========  ===============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
0D        NC               :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` ( :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>` )                                                                                                                                                                                                                                                                              
1D        NCW              :ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` ( :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>` ), :ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` ( :ref:`dnnl_acb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daf8537ed269eb5d0586456db114039c00>` ), *optimized^*           
2D        NCHW             :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` ), :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` ( :ref:`dnnl_acdb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da8fcce5dd7260b5b0740e3b37b1e9ad41>` ), *optimized^*       
3D        NCDHW            :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>` ( :ref:`dnnl_abcde <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da30d5d3c9de2931f06d265af81787ada3>` ), :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>` ( :ref:`dnnl_acdeb <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da0cfe86402763786b9b4d73062cfd2f05>` ), *optimized^*   
========  ===============  ===========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Here optimized^ means the format that :ref:`comes out <doxid-memory_format_propagation_cpp>` of any preceding compute-intensive primitive.

Post-Ops and Attributes
-----------------------

Post-ops and attributes enable you to modify the behavior of the batch normalization primitive by chaining certain operations after the batch normalization operation. The following post-ops are supported by batch normalization primitives:

============  ========  ==========  ========================================================================================================================================================================================================================================================================  
Propagation   Type      Operation   Description                                                                                                                                                                                                                                                               
============  ========  ==========  ========================================================================================================================================================================================================================================================================  
forward       post-op   eltwise     Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result (currently only :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>` algorithm is supported)   
============  ========  ==========  ========================================================================================================================================================================================================================================================================

.. note:: 

   As mentioned in :ref:`Primitive Attributes <doxid-dev_guide_attributes>`, the post-ops should be used for inference only. For instance, using ReLU as a post-op would not produce the additional output ``workspace`` that is required to compute backward propagation correctly. Hence, in case of training one should use the :ref:`dnnl_fuse_norm_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>` or :ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>` directly.
   
   
:target:`doxid-dev_guide_batch_normalization_1dg_bnorm_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. For the data types that have forward propagation support only, mean and variance must be provided by a user (i.e., :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` is set).

#. CPU implementations do not support the fusion with binary addition and ReLU activation (:ref:`dnnl_fuse_norm_add_relu <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>`).

Performance Tips
~~~~~~~~~~~~~~~~

#. For backward propagation, use the same memory format for :math:`\src`, :math:`\diffdst`, and :math:`\diffsrc` (the format of the :math:`\diffdst` and :math:`\diffsrc` are always the same because of the API). Different formats are functionally supported but lead to highly suboptimal performance.

#. Use in-place operations whenever possible (see caveats in General Notes).

#. GPU implementations support an experimental algorithm with single pass statistics calculations. Please review :ref:`experimental features <doxid-dev_guide_experimental>` for more details.

Examples
~~~~~~~~

:ref:`Batch Normalization Primitive Example <doxid-batch_normalization_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Batch Normalization <doxid-dev_guide_batch_normalization>` primitive in forward training propagation mode.

Key optimizations included in this example:

* In-place primitive execution;

* Source memory format for an optimized primitive implementation;

* Fused post-ops via operation descriptor flags;

