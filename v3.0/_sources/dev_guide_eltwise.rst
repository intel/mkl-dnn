.. index:: pair: page; Eltwise
.. _doxid-dev_guide_eltwise:

Eltwise
=======

:ref:`API Reference <doxid-group__dnnl__api__eltwise>`

General
~~~~~~~

Forward
-------

The eltwise primitive applies an operation to every element of the tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst_{i_1, \ldots, i_k} = Operation\left(\src_{i_1, \ldots, i_k}\right).

For notational convenience, in the formulas below we will denote individual element of :math:`\src`, :math:`\dst`, :math:`\diffsrc`, and :math:`\diffdst` tensors via s, d, ds, and dd respectively.

The following operations are supported:

============  ================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================  ===========================================================================================================================================================================  =======================================================================================================================  
Operation     oneDNN algorithm kind                                                                                                                                                                                                                                                                                             Forward formula                                                                                                                                               Backward formula (from src)                                                                                                                                                  Backward formula (from dst)                                                                                              
============  ================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================  ===========================================================================================================================================================================  =======================================================================================================================  
abs           :ref:`dnnl_eltwise_abs <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2ac04aed39c46f6d6356744d9d12df43>`                                                                                                                                                                        :math:`d = \begin{cases} s & \text{if}\ s > 0 \\ -s & \text{if}\ s \leq 0 \end{cases}`                                                                        :math:`ds = \begin{cases} dd & \text{if}\ s > 0 \\ -dd & \text{if}\ s < 0 \\ 0 & \text{if}\ s = 0 \end{cases}`                                                                                                                                                                                        
clip          :ref:`dnnl_eltwise_clip <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a026ef822b5cc28653e0730f8c8c2cf32>`                                                                                                                                                                       :math:`d = \begin{cases} \beta & \text{if}\ s > \beta \geq \alpha \\ s & \text{if}\ \alpha < s \leq \beta \\ \alpha & \text{if}\ s \leq \alpha \end{cases}`   :math:`ds = \begin{cases} dd & \text{if}\ \alpha < s \leq \beta \\ 0 & \text{otherwise}\ \end{cases}`                                                                                                                                                                                                 
clip_v2       :ref:`dnnl_eltwise_clip_v2 <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a911e6995e534a9f8e6af121bc2aba2d6>` :ref:`dnnl_eltwise_clip_v2_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a50d7ed64b4ab2a5c4a156291ac7cb98d>`     :math:`d = \begin{cases} \beta & \text{if}\ s \geq \beta \geq \alpha \\ s & \text{if}\ \alpha < s < \beta \\ \alpha & \text{if}\ s \leq \alpha \end{cases}`   :math:`ds = \begin{cases} dd & \text{if}\ \alpha < s < \beta \\ 0 & \text{otherwise}\ \end{cases}`                                                                           :math:`ds = \begin{cases} dd & \text{if}\ \alpha < d < \beta \\ 0 & \text{otherwise}\ \end{cases}`                       
elu           :ref:`dnnl_eltwise_elu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a7afda2aa9bac4a229909522235f461b5>` :ref:`dnnl_eltwise_elu_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a975aea11dce8571bf1d4b2552c652a27>`             :math:`d = \begin{cases} s & \text{if}\ s > 0 \\ \alpha (e^s - 1) & \text{if}\ s \leq 0 \end{cases}`                                                          :math:`ds = \begin{cases} dd & \text{if}\ s > 0 \\ dd \cdot \alpha e^s & \text{if}\ s \leq 0 \end{cases}`                                                                    :math:`ds = \begin{cases} dd & \text{if}\ d > 0 \\ dd \cdot (d + \alpha) & \text{if}\ d \leq 0 \end{cases}. See\ (2).`   
exp           :ref:`dnnl_eltwise_exp <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4859f1326783a273500ef294bb7c7d5c>` :ref:`dnnl_eltwise_exp_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae7f15ca067ce527eb66a35767d253e81>`             :math:`d = e^s`                                                                                                                                               :math:`ds = dd \cdot e^s`                                                                                                                                                    :math:`ds = dd \cdot d`                                                                                                  
gelu_erf      :ref:`dnnl_eltwise_gelu_erf <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a676e7d4e899ab2bbddc72f73a54c7779>`                                                                                                                                                                   :math:`d = 0.5 s (1 + \operatorname{erf}[\frac{s}{\sqrt{2}}])`                                                                                                :math:`ds = dd \cdot \left(0.5 + 0.5 \, \operatorname{erf}\left({\frac{s}{\sqrt{2}}}\right) + \frac{s}{\sqrt{2\pi}}e^{-0.5s^{2}}\right)`                                                                                                                                                              
gelu_tanh     :ref:`dnnl_eltwise_gelu_tanh <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a18c14d6904040bff94bce8a43c039c62>`                                                                                                                                                                  :math:`d = 0.5 s (1 + \tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)])`                                                                                        :math:`See\ (1).`                                                                                                                                                                                                                                                                                     
hardsigmoid   :ref:`dnnl_eltwise_hardsigmoid <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a19381a5adcfa6889394eab43c3fc4ee3>`                                                                                                                                                                :math:`d = \text{max}(0, \text{min}(1, \alpha s + \beta))`                                                                                                    :math:`ds = \begin{cases} dd \cdot \alpha & \text{if}\ 0 < \alpha s + \beta < 1 \\ 0 & \text{otherwise}\ \end{cases}`                                                                                                                                                                                 
hardswish     :ref:`dnnl_eltwise_hardswish <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ee6277dfff509e9fde3d5329b8eacd9>`                                                                                                                                                                  :math:`d = s \cdot \text{max}(0, \text{min}(1, \alpha s + \beta))`                                                                                            :math:`ds = \begin{cases} dd & \text{if}\ \alpha s + \beta > 1 \\ dd \cdot (2 \alpha s + \beta) & \text{if}\ 0 < \alpha s + \beta < 1 \\ 0 & \text{otherwise} \end{cases}`                                                                                                                            
linear        :ref:`dnnl_eltwise_linear <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aed5eec69000ddfe6ac96e161b0d723b4>`                                                                                                                                                                     :math:`d = \alpha s + \beta`                                                                                                                                  :math:`ds = \alpha \cdot dd`                                                                                                                                                                                                                                                                          
log           :ref:`dnnl_eltwise_log <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8ea10785816fd41353b49445852e0b74>`                                                                                                                                                                        :math:`d = \log_{e}{s}`                                                                                                                                       :math:`ds = \frac{dd}{s}`                                                                                                                                                                                                                                                                             
logistic      :ref:`dnnl_eltwise_logistic <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ab560981bee9e7711017423e29ba46071>` :ref:`dnnl_eltwise_logistic_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad224a5a4730407c8b97a10fb53d1fe0f>`   :math:`d = \frac{1}{1+e^{-s}}`                                                                                                                                :math:`ds = \frac{dd}{1+e^{-s}} \cdot (1 - \frac{1}{1+e^{-s}})`                                                                                                              :math:`ds = dd \cdot d \cdot (1 - d)`                                                                                    
mish          :ref:`dnnl_eltwise_mish <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae3b2cacb38f7aa0a115e631caa5d63d5>`                                                                                                                                                                       :math:`d = s \cdot \tanh{(\log_{e}(1+e^s))}`                                                                                                                  :math:`ds = dd \cdot \frac{e^{s} \cdot \omega}{\delta^{2}}. See\ (3).`                                                                                                                                                                                                                                
pow           :ref:`dnnl_eltwise_pow <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa1d0f7a69b7dfbfbd817623552558054>`                                                                                                                                                                        :math:`d = \alpha s^{\beta}`                                                                                                                                  :math:`ds = dd \cdot \alpha \beta s^{\beta - 1}`                                                                                                                                                                                                                                                      
relu          :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>` :ref:`dnnl_eltwise_relu_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aa2fffcdde8480cd08a0d6e4dee7dec53>`           :math:`d = \begin{cases} s & \text{if}\ s > 0 \\ \alpha s & \text{if}\ s \leq 0 \end{cases}`                                                                  :math:`ds = \begin{cases} dd & \text{if}\ s > 0 \\ \alpha \cdot dd & \text{if}\ s \leq 0 \end{cases}`                                                                        :math:`ds = \begin{cases} dd & \text{if}\ d > 0 \\ \alpha \cdot dd & \text{if}\ d \leq 0 \end{cases}. See\ (2).`         
round         :ref:`dnnl_eltwise_round <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adda28cb0389d39c0c43967352b116d9d>`                                                                                                                                                                      :math:`d = round(s)`                                                                                                                                                                                                                                                                                                                                                                                                                                                
soft_relu     :ref:`dnnl_eltwise_soft_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a82d95f7071af086d4b1652160d9a972f>`                                                                                                                                                                  :math:`d =\frac{1}{\alpha} \log_{e}(1+e^{\alpha s})`                                                                                                          :math:`ds = \frac{dd}{1 + e^{-\alpha s}}`                                                                                                                                                                                                                                                             
sqrt          :ref:`dnnl_eltwise_sqrt <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a2152d4664761b356bbceed3d9afe2189>` :ref:`dnnl_eltwise_sqrt_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a45b82064ee41f69c5463895c41ec24d0>`           :math:`d = \sqrt{s}`                                                                                                                                          :math:`ds = \frac{dd}{2\sqrt{s}}`                                                                                                                                            :math:`ds = \frac{dd}{2d}`                                                                                               
square        :ref:`dnnl_eltwise_square <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a4da34cea03ccb7cc2701b2f2023bcc2e>`                                                                                                                                                                     :math:`d = s^2`                                                                                                                                               :math:`ds = dd \cdot 2 s`                                                                                                                                                                                                                                                                             
swish         :ref:`dnnl_eltwise_swish <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a63447dedf2e45ab535f1365502ff3240>`                                                                                                                                                                      :math:`d = \frac{s}{1+e^{-\alpha s}}`                                                                                                                         :math:`ds = \frac{dd}{1 + e^{-\alpha s}}(1 + \alpha s (1 - \frac{1}{1 + e^{-\alpha s}}))`                                                                                                                                                                                                             
tanh          :ref:`dnnl_eltwise_tanh <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a81b20d8f0b54c7114024186a9fbb698e>` :ref:`dnnl_eltwise_tanh_use_dst_for_bwd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a04e559b66a5d43a74a9f1b91da78151c>`           :math:`d = \tanh{s}`                                                                                                                                          :math:`ds = dd \cdot (1 - \tanh^2{s})`                                                                                                                                       :math:`ds = dd \cdot (1 - d^2)`                                                                                          
============  ================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================  ===========================================================================================================================================================================  =======================================================================================================================

:math:`(1)\ ds = dd \cdot 0.5 (1 + \tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)]) \cdot (1 + \sqrt{\frac{2}{\pi}} (s + 0.134145 s^3) \cdot (1 - \tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)]) )`

:math:`(2)\ \text{Operation is supported only for } \alpha \geq 0.`

:math:`(3)\ \text{where, } \omega = e^{3s} + 4 \cdot e^{2s} + e^{s} \cdot (4 \cdot s + 6) + 4 \cdot (s + 1) \text{ and } \delta = e^{2s} + 2 \cdot e^{s} + 2.`

Note that following equations hold:

* :math:`bounded\_relu(s, alpha) = clip(s, 0, alpha)`

* :math:`logsigmoid(s) = soft\_relu(s, -1)`

* :math:`hardswish(s, alpha, beta) = s \cdot hardsigmoid(s, alpha, beta)`

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc` based on :math:`\diffdst` and :math:`\src` tensors. However, some operations support a computation using :math:`\dst` memory produced during the forward propagation. Refer to the table above for a list of operations supporting destination as input memory and the corresponding formulas.

Exceptions
++++++++++

The eltwise primitive with algorithm round does not support backward propagation.

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

#. All eltwise primitives have 3 primitive descriptor creation functions (e.g., :ref:`dnnl::eltwise_forward::primitive_desc() <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`) which may take both :math:`\alpha` and :math:`\beta`, just :math:`\alpha`, or none of them.

#. Both forward and backward propagation support in-place operations, meaning that :math:`\src` can be used as input and output for forward propagation, and :math:`\diffdst` can be used as input and output for backward propagation. In case of an in-place operation, the original data will be overwritten. Note, however, that some algorithms for backward propagation require original :math:`\src`, hence the corresponding forward propagation should not be performed in-place for those algorithms. Algorithms that use :math:`\dst` for backward propagation can be safely done in-place.

#. For some operations it might be beneficial to compute backward propagation based on :math:`\dst(\overline{s})`, rather than on :math:`\src(\overline{s})`, for improved performance.

#. For logsigmoid original formula :math:`d = \log_{e}(\frac{1}{1+e^{-s}})` was replaced by :math:`d = -soft\_relu(-s)` for numerical stability.

.. note:: 

   For operations supporting destination memory as input, :math:`\dst` can be used instead of :math:`\src` when backward propagation is computed. This enables several performance optimizations (see the tips below).
   
   


Data Type Support
-----------------

The eltwise primitive supports the following combinations of data types:

===================  =====================  ====  
Propagation          Source / Destination   Int   
===================  =====================  ====  
forward / backward   f32, bf16, f16         f32   
forward              s32 / s8 / u8          f32   
===================  =====================  ====

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check :ref:`Implementation Limitations <doxid-dev_guide_eltwise_1dg_eltwise_impl_limits>` section below.
   
   
Here the intermediate data type means that the values coming in are first converted to the intermediate data type, then the operation is applied, and finally the result is converted to the output data type.

Data Representation
-------------------

The eltwise primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.

Post-Ops and Attributes
-----------------------

============  ========  =================================================================================  =================================================================================  ====================================  
Propagation   Type      Operation                                                                          Description                                                                        Restrictions                          
============  ========  =================================================================================  =================================================================================  ====================================  
Forward       Post-op   :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`   Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result   General binary post-op restrictions   
============  ========  =================================================================================  =================================================================================  ====================================

:target:`doxid-dev_guide_eltwise_1dg_eltwise_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.

Performance Tips
~~~~~~~~~~~~~~~~

#. For backward propagation, use the same memory format for :math:`\src`, :math:`\diffdst`, and :math:`\diffsrc` (the format of the :math:`\diffdst` and :math:`\diffsrc` are always the same because of the API). Different formats are functionally supported but lead to highly suboptimal performance.

#. Use in-place operations whenever possible (see caveats in General Notes).

#. As mentioned above for all operations supporting destination memory as input, one can use the :math:`\dst` tensor instead of :math:`\src`. This enables the following potential optimizations for training:
   
   * Such operations can be safely done in-place.
   
   * Moreover, such operations can be fused as a :ref:`post-op <doxid-dev_guide_attributes>` with the previous operation if that operation does not require its :math:`\dst` to compute the backward propagation (e.g., if the convolution operation satisfies these conditions).

Example
~~~~~~~

:ref:`Eltwise Primitive Example <doxid-eltwise_example_cpp>`

This C++ API example demonstrates how to create and execute an :ref:`Element-wise <doxid-dev_guide_eltwise>` primitive in forward training propagation mode.

