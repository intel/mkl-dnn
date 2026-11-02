.. index:: pair: page; MaxPool
.. _doxid-dev_guide_op_maxpool:

MaxPool
=======

General
~~~~~~~

MaxPool operation performs the computation following the below formulas. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

.. math::

	\dst(n, c, oh, ow) = \max\limits_{kh, kw} \left( \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L) \right)

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==========================================================================================================================  ============================================================================================================================================================================================================  ===========  =====================================================================================  =====================  
Attribute Name                                                                                                              Description                                                                                                                                                                                                   Value Type   Supported Values                                                                       Required or Optional   
==========================================================================================================================  ============================================================================================================================================================================================================  ===========  =====================================================================================  =====================  
:ref:`strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`         Controls the strides the window is moved.                                                                                                                                                                     s64          A s64 list containing positive values                                                  Required               
:ref:`pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`      Controls number of zeros to be add to the front/top/left of spatial dimensions, the attribute will be ignored when ``auto_pad`` attribute is specified to ``same_upper`` , ``same_lower`` or ``valid`` .      s64          A s64 list containing non-negative values                                              Required               
:ref:`pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`        Controls number of zeros to be add to the back/bottom/right of spatial dimensions, the attribute will be ignored when ``auto_pad`` attribute is specified to ``same_upper`` , ``same_lower`` or ``valid`` .   s64          A s64 list containing non-negative values                                              Required               
:ref:`kernel <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a50484c19f1afdaf3841a0d821ed393d2>`          Size of pooling window.                                                                                                                                                                                       s64          A s64 list containing positive values                                                  Required               
:ref:`rounding_type <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae09cfc230f470609746f3021591072e3>`   Controls how to do rounding.                                                                                                                                                                                  string       ``floor`` (default), ``ceil``                                                          Optional               
:ref:`auto_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`        Controls how the paddings are calculated.                                                                                                                                                                     string       ``none`` (default), ``same_upper`` , ``same_lower`` , ``valid``                        Optional               
:ref:`dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`       Denotes the distance in width and height between elements in the window.                                                                                                                                      s64          A s64 list containing positive values, a list of ``1`` s (default) means no dilation   Optional               
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`     Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                                                                                  string       ``NCX`` , ``NXC`` (default)                                                            Optional               
==========================================================================================================================  ============================================================================================================================================================================================================  ===========  =====================================================================================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
======  ==============  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

MaxPool operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

