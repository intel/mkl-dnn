.. index:: pair: page; Convolution
.. _doxid-dev_guide_op_convolution:

Convolution
===========

General
~~~~~~~

Convolution operation performs the convolution between src tensor and weight tensor, which is defined as by the following formulas. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

Let :math:`\src`, :math:`\weights` and :math:`\dst` tensors have shape :math:`N \times IC \times IH \times IW`, :math:`OC \times IC \times KH \times KW`, and :math:`N \times OC \times OH \times OW` respectively.

Furthermore, let the remaining convolution parameters be:

=================================  =============  =============  =============  =========================================================================================  
Parameter                          Depth          Height         Width          
=================================  =============  =============  =============  =========================================================================================  
Paddings: Front, top, and left     :math:`PD_L`   :math:`PH_L`   :math:`PW_L`   In the attributes we use ``pads_begin`` to indicate the corresponding vector of paddings   
Padding: Back, bottom, and right   :math:`PD_R`   :math:`PH_R`   :math:`PW_R`   In the attributes we use ``pads_end`` to indicate the corresponding vector of paddings     
Stride                             :math:`SD`     :math:`SH`     :math:`SW`     In the attributes we use ``strides`` to indicate the corresponding vector of strides       
Dilation                           :math:`DD`     :math:`DH`     :math:`DW`     In the attributes we use ``dilations`` to indicate the corresponding vector of dilations   
=================================  =============  =============  =============  =========================================================================================

To further simplify the formulas, we assume that the attribute ``data_format`` and ``weights_format`` are set to ``NCX`` and ``OIX`` respectively. ``NCX`` means the fist axis represents batch dimension, the second axis represents channel dimension and the rest represents spatial dimensions. ``OIX`` means the first axis represents output channel dimension, the second axis represents input channel dimension and the rest represents weights spatial dimensions.

Regular Convolution
-------------------

This is the same as the formula in :ref:`Convolution primitive <doxid-dev_guide_convolution>`.

.. math::

	\dst(n, oc, oh, ow) = \bias(oc) \\ + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, ic, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L) \cdot \weights(oc, ic, kh, kw).

Here:

* :math:`OH = \left\lfloor{\frac{IH - KH + PH_L + PH_R}{SH}} \right\rfloor + 1,`

* :math:`OW = \left\lfloor{\frac{IW - KW + PW_L + PW_R}{SW}} \right\rfloor + 1.`

Convolution with Groups
-----------------------

The attribute ``groups`` is set to :math:`>1`.

.. math::

	\dst(n, g \cdot OC_G + oc_g, oh, ow) = \bias(g \cdot OC_G + oc_g) \\ + \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, g \cdot IC_G + ic_g, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L) \cdot \weights(g, oc_g, ic_g, kh, kw),

where

* :math:`IC_G = \frac{IC}{G}`,

* :math:`OC_G = \frac{OC}{G}`, and

* :math:`oc_g \in [0, OC_G).`

Convolution with Dilation
-------------------------

The attribute ``dilation`` contains the element which is :math:`>1`.

.. math::

	\dst(n, oc, oh, ow) = \bias(oc) \\ + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, ic, oh \cdot SH + kh \cdot DH - PH_L, ow \cdot SW + kw \cdot DW - PW_L) \cdot \weights(oc, ic, kh, kw).

Here:

* :math:`OH = \left\lfloor{\frac{IH - DKH + PH_L + PH_R}{SH}} \right\rfloor + 1,` where :math:`DKH = 1 + (KH - 1) \cdot DH`, and

* :math:`OW = \left\lfloor{\frac{IW - DKW + PW_L + PW_R}{SW}} \right\rfloor + 1,` where :math:`DKW = 1 + (KW - 1) \cdot DW`.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===============================================================================================================================================================================================  =======  =====================================================================  =========  
Attribute Name                                                                                                               De                                                                                                                                                                                               
===========================================================================================================================  ===============================================================================================================================================================================================  =======  =====================================================================  =========  
:ref:`strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`          Controls the strides the weights tensor is moved when computing convolution                                                                                                                      s64      A s64 list containing positive values                                  Required   
:ref:`pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`       Controls number of zeros to be add to the front/top/left of spatial dimensions                                                                                                                   s64      A s64 list containing non-negative values                              Required   
:ref:`pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`         Controls number of zeros to be add to the back/bottom/right of spatial dimensions                                                                                                                s64      A s64 list containing non-negative values                              Required   
:ref:`dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`        Controls the amount of stretching the kernel before convolution ( `visualization link <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations>`__ )   s64      A s64 list containing positive values (>1 means dilated convolution)   Required   
:ref:`auto_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`         Controls how the padding is calculated                                                                                                                                                           string   ``none`` (default), ``same_upper`` , ``same_lower`` , ``valid``        Optional   
:ref:`groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`           Controls how input channels and output channels are divided into                                                                                                                                 s64      A positive s64 value, ``1`` by default                                 Optional   
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`      Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                                                                     string   ``NCX`` , ``NXC`` (default)                                            Optional   
:ref:`weights_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`   Controls how to interpret the shape of ``weights``                                                                                                                                               string   ``OIX`` , ``XIO`` (default)                                            Optional   
===========================================================================================================================  ===============================================================================================================================================================================================  =======  =====================================================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ============  =========  
Index   Argu          
======  ============  =========  
0       ``src``       Required   
1       ``weights``   Required   
2       ``bias``      Optional   
======  ============  =========

.. note:: 

   The shape of :math:`\weights` is :math:`(out\_channels, in\_channels / groups, spatial\_shape)` for ``OIX`` format or :math:`(spatial\_shape, in\_channels / groups, out\_channels)` for ``XIO`` format. Both :math:`in\_channels` and :math:`out\_channels` must be divisible by groups attribute.
   
   


Outputs
-------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``dst``   Required   
======  ========  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

Convolution operation supports the following data type combinations.

=====  ========  =====  =====  
Src    Weights   
=====  ========  =====  =====  
f32    f32       f32    f32    
bf16   bf16      bf16   bf16   
f16    f16       f16    f16    
=====  ========  =====  =====

