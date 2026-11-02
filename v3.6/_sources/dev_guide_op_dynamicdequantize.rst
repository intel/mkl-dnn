.. index:: pair: page; DynamicDequantize
.. _doxid-dev_guide_op_dynamicdequantize:

DynamicDequantize
=================

General
~~~~~~~

DynamicDequantize operation converts a quantized (s8 or u8) tensor to a f32 tensor. It supports both per-tensor and per-channel asymmetric linear de-quantization. Rounding mode is library-implementation defined. Unlike the :ref:`Dequantize <doxid-dev_guide_op_dequantize>`, DynamicDequantize takes scales and zero-points as operator src tensors.

For per-tensor de-quantization

.. math::

	dst = (src - zps)*scales

For per-channel de-quantization, taking channel axis = 1 as an example:

.. math::

	{dst}_{\cdots,i,\cdots,\cdots} = (src_{\cdots,i,\cdots,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1]

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  =====================================================================  ===========  =================================================================================================================================================  =====================  
Attribute Name                                                                                                      Description                                                            Value Type   Supported Values                                                                                                                                   Required or Optional   
==================================================================================================================  =====================================================================  ===========  =================================================================================================================================================  =====================  
:ref:`qtype <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e>`   Specifies which de-quantization type is used.                          string       ``per_tensor`` (default), ``per_channel``                                                                                                          Optional               
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`    Specifies dimension on which per-channel de-quantization is applied.   s64          A s64 value in the range of [-r, r-1] where r = rank(src), ``1`` by default. Negative value means counting the dimension backwards from the end.   Optional               
==================================================================================================================  =====================================================================  ===========  =================================================================================================================================================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
1       ``scales``      Required               
2       ``zps``         Optional               
======  ==============  =====================

.. note:: 

   ``scales`` is a f32 1D tensor to be applied to the de-quantization formula. For ``qtype`` = ``per-tensor``, there should be only one element in the scales tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of src tensor along the dimension axis.
   
   

.. note:: 

   ``zps`` is a 1D tensor with offset values that map to zero. For ``qtype`` = ``per-tensor``, there should be only one element in the zps tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of input tensor along the dimension axis. If omitted, zps values are assumed to be zero.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

DynamicDequantize operation supports the following data type combinations.

====  ====  =======  ============  
Src   Dst   Scales   Zps           
====  ====  =======  ============  
s8    f32   f32      s8, u8, s32   
u8    f32   f32      s8, u8, s32   
====  ====  =======  ============

