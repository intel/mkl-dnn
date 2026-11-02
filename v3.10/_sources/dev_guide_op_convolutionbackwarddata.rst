.. index:: pair: page; ConvolutionBackwardData
.. _doxid-dev_guide_op_convolutionbackwarddata:

ConvolutionBackwardData
=======================

General
~~~~~~~

ConvolutionBackwardData operation accepts :math:`\diffdst`, :math:`\weights` and optional dst shape as inputs, and compute the :math:`\diffsrc`.

If ``auto_pad`` attribute is specified to one of ``valid``, ``same_upper`` and ``same_lower``, ``pads_begin`` and ``pads_end`` attributes will be ignored. The paddings will be calculated by following the below formula:

Let the parameters be:

=================================  =============  =============  =============  =========================================================================================  
Parameter                          Depth          Height         Width          Comment                                                                                    
=================================  =============  =============  =============  =========================================================================================  
Paddings: Front, top, and left     :math:`PD_L`   :math:`PH_L`   :math:`PW_L`   In the attributes we use ``pads_begin`` to indicate the corresponding vector of paddings   
Padding: Back, bottom, and right   :math:`PD_R`   :math:`PH_R`   :math:`PW_R`   In the attributes we use ``pads_end`` to indicate the corresponding vector of paddings     
Stride                             :math:`SD`     :math:`SH`     :math:`SW`     In the attributes we use ``strides`` to indicate the corresponding vector of strides       
Dilation                           :math:`DD`     :math:`DH`     :math:`DW`     In the attributes we use ``dilations`` to indicate the corresponding vector of dilations   
=================================  =============  =============  =============  =========================================================================================

Firstly, :math:`total\_padding` is calculated according to :math:`src\_shape` and :math:`dst\_shape`. Let :math:`src\_h` be height dimension of :math:`src\_shape` and :math:`dst\_h` be height dimension of :math:`dst\_shape`.

.. math::

	total\_padding_h = SH \times (src\_h - 1) + ((KH -1 ) \times DH + 1) - dst\_h + output\_padding_h

If ``auto_pad`` attribute is specified as ``valid`` :

.. math::

	PD_L = 0 \\ PD_R = 0

If ``auto_pad`` attribute is specified as ``same_lower`` :

.. math::

	PD_L = floor(total\_padding / 2) \\ PD_R = total\_padding - PD_L

If ``auto_pad`` attribute is specified as ``same_upper`` :

.. math::

	PD_L = total\_padding - PD_R \\ PD_R = floor(total\_padding / 2)

where:

* :math:`dst\_shape` is either an attribute or an input tensor,

* :math:`output\_padding` is an optional attribute.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===============================================================================================================================================================================================  ===========  =====================================================================  =====================  
Attribute Name                                                                                                               Description                                                                                                                                                                                      Value Type   Supported Values                                                       Required or Optional   
===========================================================================================================================  ===============================================================================================================================================================================================  ===========  =====================================================================  =====================  
:ref:`strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`          Controls the strides the weights tensor is moved when computing convolution                                                                                                                      s64          A s64 list containing positive values                                  Required               
:ref:`pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`       Controls number of zeros to be add to the front/top/left of spatial dimensions                                                                                                                   s64          A s64 list containing non-negative values                              Required               
:ref:`pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`         Controls number of zeros to be add to the back/bottom/right of spatial dimensions                                                                                                                s64          A s64 list containing non-negative values                              Required               
:ref:`dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`        Controls the amount of stretching the kernel before convolution ( `visualization link <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations>`__ )   s64          A s64 list containing positive values (>1 means dilated convolution)   Required               
:ref:`auto_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`         Controls how the padding is calculated                                                                                                                                                           string       ``none`` (default), ``same_upper`` , ``same_lower`` , ``valid``        Optional               
:ref:`output_padding <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a16e84dbe0f1d0f82b74ebd187a0fe466>`   Adds additional amount of padding per each spatial axis in ``dst`` .                                                                                                                             s64          A s64 list containing non-negative values, all zeros by default        Optional               
:ref:`groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`           Controls how input channels and output channels are divided into                                                                                                                                 s64          A positive s64 value, ``1`` by default                                 Optional               
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`      Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                                                                     string       ``NCX`` , ``NXC`` (default)                                            Optional               
:ref:`weights_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`   Controls how to interpret the shape of ``weights`` .                                                                                                                                             string       ``OIX`` , ``XIO`` (default)                                            Optional               
:ref:`dst_shape <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8ab1066346d3720658f87bb7686f7a22>`        Denotes the shape of the ``dst`` tensor.                                                                                                                                                         s64          A s64 list containing positive values                                  Optional               
===========================================================================================================================  ===============================================================================================================================================================================================  ===========  =====================================================================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_dst``    Required               
1       ``weights``     Required               
2       ``dst_shape``   Optional               
======  ==============  =====================

.. note:: 

   The shape of :math:`\weights` is :math:`(out\_channels, in\_channels / groups, spatial\_shape)` for ``OIX`` format or :math:`(spatial\_shape, in\_channels / groups, out\_channels)` for ``XIO`` format. Both :math:`in\_channels` and :math:`out\_channels` must be divisible by groups attribute.
   
   

.. note:: 

   Either ``dst_shape`` input or ``dst_shape`` attribute should be provided. If both provided, ``dst_shape`` input will precede over ``dst_shape`` attribute.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_src``    Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

ConvolutionBackwardData operation supports the following data type combinations.

=========  ========  =========  ==========  
Diff_dst   Weights   Diff_src   Dst_shape   
=========  ========  =========  ==========  
f32        f32       f32        s32         
bf16       bf16      bf16       s32         
f16        f16       f16        s32         
=========  ========  =========  ==========

