.. index:: pair: page; AvgPoolBackward
.. _doxid-dev_guide_op_avgpoolbackward:

AvgPoolBackward
===============

General
~~~~~~~

AvgPoolBackward operation accepts :math:`\diffdst` tensor and :math:`\srcshape` tensor (optional), and calculates :math:`\diffsrc` tensor.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  ============================================================================================================================================================================================================  =======  ================================================================  =========  
Attribute Name                                                                                                            De                                                                                                                                                                                                            
========================================================================================================================  ============================================================================================================================================================================================================  =======  ================================================================  =========  
:ref:`strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`       Controls the strides the window is moved.                                                                                                                                                                     s64      A s64 list containing positive values                             Required   
:ref:`pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`    Controls number of zeros to be add to the front/top/left of spatial dimensions, the attribute will be ignored when ``auto_pad`` attribute is specified to ``same_upper`` , ``same_lower`` or ``valid`` .      s64      A s64 list containing non-negative values                         Required   
:ref:`pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`      Controls number of zeros to be add to the back/bottom/right of spatial dimensions, the attribute will be ignored when ``auto_pad`` attribute is specified to ``same_upper`` , ``same_lower`` or ``valid`` .   s64      A s64 list containing non-negative values                         Required   
:ref:`kernel <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a50484c19f1afdaf3841a0d821ed393d2>`        Size of pooling window.                                                                                                                                                                                       s64      A s64 list containing positive values                             Required   
:ref:`exclude_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9e17a7762faf53a18315187610b2351c>`   Controls whether the padded values are counted.                                                                                                                                                               bool     True, False                                                       Required   
:ref:`auto_pad <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`      Controls how the paddings are calculated.                                                                                                                                                                     string   ``none`` (default), ``same_upper`` , ``same_lower`` , ``valid``   Optional   
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`   Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                                                                                  string   ``NCX`` , ``NXC`` (default)                                       Optional   
:ref:`src_shape <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a27bbe1bc8190497bf47ed8bbab478a8b>`     Denotes the shape of input of forward op.                                                                                                                                                                     string   ``NCX`` , ``NXC`` (default)                                       Optional   
========================================================================================================================  ============================================================================================================================================================================================================  =======  ================================================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =========  
Index   Argu            
======  ==============  =========  
0       ``diff_dst``    Required   
1       ``src_shape``   Optional   
======  ==============  =========

.. note:: 

   Either ``src_shape`` input or ``src_shape`` attribute should be provided. If both provided, ``src_shape`` input will precede over ``src_shape`` attribute.
   
   


Outputs
-------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_src``   Required   
======  =============  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

AvgPoolBackward operation supports the following data type combinations.

=========  =====  ====  
Diff_dst   
=========  =====  ====  
f32        f32    s64   
bf16       bf16   s64   
f16        f16    s64   
=========  =====  ====

