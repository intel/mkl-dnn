.. index:: pair: page; InterpolateBackward
.. _doxid-dev_guide_op_interpolatebackward:

InterpolateBackward
===================

General
~~~~~~~

InterpolateBackward computes the gradients of Interpolate operation.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================================  =========================================================================================================  ===========  ===========================================================  =====================  
Attribute Name                                                                                                                               Description                                                                                                Value Type   Supported Values                                             Required or Optional   
===========================================================================================================================================  =========================================================================================================  ===========  ===========================================================  =====================  
:ref:`mode <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a15d61712450a686a7f365adf4fef581f>`                             Specifies type of interpolation                                                                            string.      ``nearest`` , ``linear`` , ``bilinear`` , ``trilinear``      Required               
:ref:`coordinate_transformation_mode <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a171f02207298aa1f95eacc0907efe069>`   Specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor   string.      ``half_pixel`` (default), ``align_corners``                  Optional               
:ref:`sizes <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ab027168eed2f9d69319d4819454b8ab4>`                            Specifies dst shape for spatial axes.                                                                      s64          A s64 list containing positive values, ``none`` is default   Optional               
:ref:`scales <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8e7bb02b763a2e07d30b4ab24beb7fa1>`                           Specifies ``scales`` for spatial axes.                                                                     f32          A f32 list, ``none`` is default                              Optional               
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`                      Controls how to interpret the shape of ``src`` and ``dst`` .                                               string       ``NCX`` , ``NXC`` (default) -                                Optional               
===========================================================================================================================================  =========================================================================================================  ===========  ===========================================================  =====================

.. note:: 

   Either ``sizes`` or ``scales`` should be provided. When ``sizes`` is used, ``scales`` will be ignored.
   
   

.. note:: 

   The attribute ``coordinate_transformation_mode`` is the name of transformation mode in string format.
   
   Here ``scale[x]`` is ``dst_shape[x]/src_shape[x]`` and ``x_resized`` is a coordinate in axis ``x``,for any axis ``x`` from the src axis.
   
   For ``half_pixel`` : the coordinate in the original tensor axis ``x`` is calculated as ``((x_resized + 0.5) / scale[x]) - 0.5``.
   
   For ``align_corners`` : the coordinate in the original tensor axis ``x`` is calculated as 0 if ``dst_shape[x] == 1`` else ``x_resized * (src_shape[x] - 1) / (dst_shape[x] - 1)``.
   
   


Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
1       ``diff_dst``    Required               
2       ``sizes``       Optional               
======  ==============  =====================

.. note:: 

   ``src`` is original input tensor of Interpolate op.
   
   ``diff_dst`` is the gradient tensor with respect to the dst.
   
   ``sizes`` is a 1D tensor describing output shape for spatial axes.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_src``    Required               
======  ==============  =====================

.. note:: 

   ``diff_src`` is the gradient tensor with respect to the src of Interpolate.
   
   


Supported data types
~~~~~~~~~~~~~~~~~~~~

InterpolateBackward operation supports the following data type combinations.

=====  =========  =========  ======  
Src    Diff_dst   Diff_src   Sizes   
=====  =========  =========  ======  
f32    f32        f32        s32     
bf16   bf16       bf16       s32     
f16    f16        f16        s32     
=====  =========  =========  ======

