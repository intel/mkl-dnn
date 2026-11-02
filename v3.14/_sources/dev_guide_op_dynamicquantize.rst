:orphan:

.. index:: pair: page; DynamicQuantize
.. _doxid-dev_guide_op_dynamicquantize:

DynamicQuantize
===============

General
~~~~~~~

DynamicQuantize operation converts an ``f32`` tensor to a quantized (``s8`` or ``u8``) tensor. It supports both per-tensor and per-channel asymmetric linear quantization. The target quantized data type is specified via the data type of dst logical tensor. Rounding mode is library-implementation defined.

For per-tensor quantization

.. math::

	dst = round(src/scales + zps)

For per-channel quantization, taking channel axis = 1 as an example:

.. math::

	{dst}_{\cdots,i,\cdots,\cdots} = round(src_{\cdots,i,\cdots,\cdots}/scales_i + zps_i),i\in [0,channelNum-1]

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  =====================================================================================================================================================================  ===========  =================================================================================================================================================  =====================  
Attribute Name                                                                                                      Description                                                                                                                                                            Value Type   Supported Values                                                                                                                                   Required or Optional   
==================================================================================================================  =====================================================================================================================================================================  ===========  =================================================================================================================================================  =====================  
:ref:`qtype <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e>`   Specifies which quantization type is used.                                                                                                                             string       ``per_tensor`` (default), ``per_channel``                                                                                                          Optional               
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`    Specifies dimension on which per-channel quantization is applied.                                                                                                      s64          A s64 value in the range of [-r, r-1] where r = rank(src), ``1`` by default. Negative value means counting the dimension backwards from the end.   Optional               
:ref:`mask <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684af2ce11ebf110993621bedd8e747d7b1b>`    Specifies which dimensions the scales/zps vary over. Bit ``i`` set means scales vary on dimension ``i`` . When set, ``qtype`` must be not set or set to the default.   s64          A non-negative integer where set bits index source dimensions.                                                                                     Optional               
==================================================================================================================  =====================================================================================================================================================================  ===========  =================================================================================================================================================  =====================

.. note:: 

   The ``mask`` attribute is the recommended way to specify quantization granularity. The ``qtype`` and ``axis`` attributes are retained for backward compatibility and will be deprecated in the future.
   
   


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

   ``scales`` is an ``f32`` 1D tensor to be applied to the quantization formula. For ``qtype`` = ``per-tensor``, there should be only one element in the scales tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of src tensor along the dimension axis. When ``mask`` is specified, the ``scales`` tensor is a packed tensor whose dimensions correspond to the source dimensions selected by the mask bits.
   
   

.. note:: 

   ``zps`` is a 1D tensor with offset values that map to zero. For ``qtype`` = ``per-tensor``, there should be only one element in the zps tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of input tensor along the dimension axis. When ``mask`` is specified, ``zps`` must have the same shape as ``scales``. If omitted, zps values are assumed to be zero.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

DynamicQuantize operation supports the following data type combinations.

====  =======  ============  ====  
Src   Scales   Zps           Dst   
====  =======  ============  ====  
f32   f32      s8, u8, s32   s8    
f32   f32      s8, u8, s32   u8    
====  =======  ============  ====

