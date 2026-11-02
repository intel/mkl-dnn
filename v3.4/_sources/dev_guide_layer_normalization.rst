.. index:: pair: page; Layer Normalization
.. _doxid-dev_guide_layer_normalization:

Layer Normalization
===================

:ref:`API Reference <doxid-group__dnnl__api__layer__normalization>`

General
~~~~~~~

The layer normalization primitive performs a forward or backward layer normalization operation on a 2-5D data tensor.

Forward
-------

The layer normalization operation performs normalization over the last logical axis of the data tensor and is defined by the following formulas. We show formulas only for 3D data, which are straightforward to generalize to cases of higher dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

.. math::

	\dst(t, n, c) = \gamma(c) \cdot \frac{\src(t, n, c) - \mu(t, n)} {\sqrt{\sigma^2(t, n) + \varepsilon}} + \beta(c),

where

* :math:`\gamma(c), \beta(c)` are optional scale and shift for a channel (see :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>`, :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` flags),

* :math:`\mu(t, n), \sigma^2(t, n)` are mean and variance (see :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag), and

* :math:`\varepsilon` is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and variance are computed at runtime, the following formulas are used:

* :math:`\mu(t, n) = \frac{1}{C} \sum\limits_{c} \src(t, n, c)_{}`,

* :math:`\sigma^2(t, n) = \frac{1}{C} \sum\limits_{c} {}_{} (\src(t, n, c) - \mu(t, n))^2`.

The :math:`\gamma(c)` and :math:`\beta(c)` tensors are considered learnable.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

* If mean and variance are computed at runtime (i.e., :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` is not set), they become outputs for the propagation kind :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` (because they would be required during the backward propagation). Data layout for mean and variance must be specified during creation of the layer normalization primitive descriptor by passing the memory descriptor for statistics (e.g., by passing stat_desc in :ref:`dnnl::layer_normalization_forward::primitive_desc() <doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc>`). Mean and variance are not exposed for the propagation kind :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

Backward
--------

The backward propagation computes :math:`\diffsrc(t, n, c)`, :math:`\diffgamma(c)^*`, and :math:`\diffbeta(c)^*` based on :math:`\diffdst(t, n, c)`, :math:`src(t, n, c)`, :math:`\mu(t, n)`, :math:`\sigma^2(t, n)`, :math:`\gamma(c) ^*`, and :math:`\beta(c) ^*`.

The tensors marked with an asterisk are used only when the primitive is configured to use :math:`\gamma(c)`, and :math:`\beta(c)` (i.e., :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are set).

Execution Arguments
~~~~~~~~~~~~~~~~~~~

Depending on the :ref:`flags <doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` and :ref:`propagation kind <doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`, the layer normalization primitive requires different inputs and outputs. For clarity, a summary is shown below.

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

============================  ====================================  
Primitive input/output        Execution argument index              
============================  ====================================  
:math:`\src`                  DNNL_ARG_SRC                          
:math:`\gamma`                DNNL_ARG_SCALE                        
:math:`\beta`                 DNNL_ARG_SHIFT                        
mean ( :math:`\mu` )          DNNL_ARG_MEAN                         
variance ( :math:`\sigma` )   DNNL_ARG_VARIANCE                     
:math:`\dst`                  DNNL_ARG_DST                          
:math:`\diffdst`              DNNL_ARG_DIFF_DST                     
:math:`\diffsrc`              DNNL_ARG_DIFF_SRC                     
:math:`\diffgamma`            DNNL_ARG_DIFF_SCALE                   
:math:`\diffbeta`             DNNL_ARG_DIFF_SHIFT                   
:math:`src scale`             DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC   
:math:`dst scale`             DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST   
============================  ====================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The different flavors of the primitive are partially controlled by the ``flags`` parameter that is passed to the primitive descriptor creation function (e.g., :ref:`dnnl::layer_normalization_forward::primitive_desc() <doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc>`). Multiple flags can be set using the bitwise OR operator (``|``).

#. For forward propagation, the mean and variance might be either computed at runtime (in which case they are outputs of the primitive) or provided by a user (in which case they are inputs). In the latter case, a user must set the :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` flag. For the backward propagation, the mean and variance are always input parameters.

#. Both forward and backward propagation support in-place operations, meaning that :math:`\src` can be used as input and output for forward propagation, and :math:`\diffdst` can be used as input and output for backward propagation. In case of an in-place operation, the original data will be overwritten. This support is limited to cases when data types of :math:`\src` and :math:`\dst` or :math:`\diffsrc` and :math:`\diffdst` are identical. Note, however, that backward propagation requires original :math:`\src`, hence the corresponding forward propagation should not be performed in-place.

Post-ops and Attributes
-----------------------

Attributes enable you to modify the behavior of the layer normalization primitive. The following attributes are supported by the layer normalization primitive:

============  ==========  =======================================================================================  ==============================================================  ===================================================================================  
Propagation   Type        Operation                                                                                Description                                                     Restrictions                                                                         
============  ==========  =======================================================================================  ==============================================================  ===================================================================================  
forward       attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`   Scales the corresponding tensor by the given scale factor(s).   Supported only for int8 layer normalization and one scale per tensor is supported.   
============  ==========  =======================================================================================  ==============================================================  ===================================================================================

Data Type Support
-----------------

The operation supports the following combinations of data types:

============  ============================  ============================  ================  
Propagation   Source                        Destination                   Scale and Shift   
============  ============================  ============================  ================  
forward       f32, bf16, f16, u8, s8, f64   f32, bf16, f16, u8, s8, f64   f32, bf16, f16    
backward      f32, bf16, f16, f64           f32, bf16, f16, f64           f32, bf16, f16    
============  ============================  ============================  ================

Mean and Variance data types are always f32 and independent of Source and Destination data types.

Data Representation
-------------------

Mean and Variance
+++++++++++++++++

The mean (:math:`\mu`) and variance (:math:`\sigma^2`) are separate tensors with number of dimensions equal to (:math:`data\_ndims - 1`) and size :math:`(data\_dim[0], data\_dim[1], ..., data\_dim[ndims - 2])`.

The corresponding memory object can have an arbitrary memory format. Unless mean and variance are computed at runtime and not exposed (i.e., propagation kind is :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` and :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>` is not set), the user should provide a memory descriptor for statistics when creating the layer normalization primitive descriptor. For best performance, it is advised to use the memory format that follows the data memory format; i.e., if the data format is :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`, the best performance can be expected for statistics with the :ref:`dnnl_tn <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1ee747f881fc8cae805ae4051e671cf>` format and suboptimal for statistics with the :ref:`dnnl_nt <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ede27a46172c1ab2775593ba4b118dc>` format.

Scale and Shift
+++++++++++++++

If :ref:`dnnl_use_scale <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>` or :ref:`dnnl_use_shift <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>` are used, the scale (:math:`\gamma`) and shift (:math:`\beta`) are separate 1D tensors of shape :math:`C`.

Source, Destination, and Their Gradients
++++++++++++++++++++++++++++++++++++++++

The layer normalization primitive works with an arbitrary data tensor; however, it was designed for RNN data tensors (i.e., :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>`, :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`, :ref:`dnnl_ldnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c>`). Unlike CNN data tensors, RNN data tensors have a single feature dimension. Layer normalization performs normalization over the last logical dimension (feature dimension for RNN tensors) across non-feature dimensions.

The layer normalization primitive is optimized for the following memory formats:

===============  =====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
Logical tensor   Implementations optimized for memory formats                                                                                                                                                                                                                                                                                                                                                                                                                                                           
===============  =====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  
NC               :ref:`dnnl_nc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac08a541001fe70289305a5fbde48906d>` ( :ref:`dnnl_ab <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da1bd907fc29344dfe7ba88336960dcf53>` )                                                                                                                                                                                                                                                        
TNC              :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>` ( :ref:`dnnl_abc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dadff5ea69392d7e4da23179dc0ba7cbc2>` ), :ref:`dnnl_ntc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da5d73ca7a68559ef44241be5a096e6bff>` ( :ref:`dnnl_bac <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae31cc3d3a0bfbd4936f7b503cf6dc9de>` )   
LDNC             :ref:`dnnl_ldnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4fd1cf9fdb67c554bcd8281695b65b3c>` ( :ref:`dnnl_abcd <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da6e669cc61278663a5ddbd3d0b25c6c5c>` )                                                                                                                                                                                                                                                    
===============  =====================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.
   
   * Different data types for source and destination are not supported.
   
   * Only f32 data type is supported for scale and shift

Performance Tips
~~~~~~~~~~~~~~~~

#. For data tensors :math:`\src`, :math:`\dst`, :math:`\diffsrc`, and :math:`\diffdst`, use memory formats for which the last logical axis is the last in the physical memory layout.

#. For ``mean`` and ``variance``, use the memory format that follows the data memory format; i.e., if the data format is :ref:`dnnl_tnc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da2a9735ec024c9362b717304edbfe2237>`, the best performance can be expected for statistics with :ref:`dnnl_tn <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dac1ee747f881fc8cae805ae4051e671cf>` and suboptimal for statistics with the :ref:`dnnl_nt <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da7ede27a46172c1ab2775593ba4b118dc>` format.

#. For backward propagation, use the same memory format for :math:`\src`, :math:`\diffdst`, and :math:`\diffsrc`. Different formats are functionally supported but lead to highly suboptimal performance.

#. Use in-place operations whenever possible (see caveats in General Notes).

Example
~~~~~~~

:ref:`Layer Normalization Primitive Example <doxid-layer_normalization_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Layer normalization <doxid-dev_guide_layer_normalization>` primitive in forward propagation mode.

Key optimizations included in this example:

* In-place primitive execution;

* Creation of memory objects using the primitive descriptor.

