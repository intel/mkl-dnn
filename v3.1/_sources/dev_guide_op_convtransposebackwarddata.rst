.. index:: pair: page; ConvTransposeBackwardData
.. _doxid-dev_guide_op_convtransposebackwarddata:

ConvTransposeBackwardData
=========================

General
~~~~~~~

ConvTransposeBackwardData operation takes :math:`\diffdst` and :math:`\weights` and computes :math:`\diffsrc`.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ================================================================================================================================================================================================  =======  =====================================================================  =========  
Attribute Name                                                                                                               De                                                                                                                                                                                                
===========================================================================================================================  ================================================================================================================================================================================================  =======  =====================================================================  =========  
:ref:`strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`          Controls the strides the weights tensor is moved when computing convolution.                                                                                                                      s64      A s64 list containing positive values                                  Required   
:ref:`pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`       Controls number of zeros to be add to the front/top/left of spatial dimensions.                                                                                                                   s64      A s64 list containing non-negative values                              Required   
:ref:`pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`         Controls number of zeros to be add to the back/bottom/right of spatial dimensions.                                                                                                                s64      A s64 list containing non-negative values                              Required   
:ref:`dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`        Controls the amount of stretching the kernel before convolution ( `visualization link <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations>`__ ).   s64      A s64 list containing positive values (>1 means dilated convolution)   Required   
:ref:`auto_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`         Controls how the padding is calculated.                                                                                                                                                           string   ``none`` (default), ``same_upper`` , ``same_lower`` , ``valid``        Optional   
:ref:`output_padding <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a16e84dbe0f1d0f82b74ebd187a0fe466>`   Adds additional amount of padding per each spatial axis in ``dst`` .                                                                                                                              s64      A s64 list containing non-negative values, all zeros by default        Optional   
:ref:`groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`           Controls how input channels and output channels are divided into.                                                                                                                                 s64      A positive s64 value, ``1`` by default                                 Optional   
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`      Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                                                                      string   ``NCX`` , ``NXC`` (default)                                            Optional   
:ref:`weights_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`   Controls how to interpret the shape of ``weights`` .                                                                                                                                              string   ``IOX`` , ``XOI`` (default)                                            Optional   
===========================================================================================================================  ================================================================================================================================================================================================  =======  =====================================================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_dst``   Required   
1       ``weights``    Required   
======  =============  =========

.. note:: 

   The shape of :math:`\weights` is :math:`(in\_channels / groups, out\_channels, spatial\_shape)` for ``IOX`` format or :math:`(spatial\_shape, out\_channels, in\_channels / groups)` for ``XOI`` format. Both :math:`in\_channels` and :math:`out\_channels` must be divisible by groups attribute.
   
   


Outputs
-------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_src``   Required   
======  =============  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

ConvTransposeBackwardData operation supports the following data type combinations.

=========  =====  =====  
Diff_dst   
=========  =====  =====  
f32        f32    f32    
bf16       bf16   bf16   
=========  =====  =====  

f16 \| f16 \| f16 \|f16

