.. index:: pair: page; DynamicDequantize
.. _doxid-dev_guide_op_dynamicdequantize:

DynamicDequantize
=================

General
~~~~~~~

The Dynamic Dequantize operation converts a quantized (s4, u4, s8, or u8) tensor to an bf16, f16 or f32 tensor. It supports per-tensor, per-channel, and per-group asymmetric linear de-quantization. The rounding mode is defined by the library implementation. Unlike the :ref:`Dequantize <doxid-dev_guide_op_dequantize>`, Dynamic Dequantize takes scales and zero-points as operator src tensors.

For per-tensor de-quantization

.. math::

	dst = (src - zps)*scales

For per-channel de-quantization, taking channel axis = 1 as an example:

.. math::

	{dst}_{\cdots,i,\cdots,\cdots} = (src_{\cdots,i,\cdots,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1]

For per-group de-quantization, let's take group shape = Gx1 as an example. It indicates that one scaling factor will de adopted for G elements in the src tensor. On the dimensions where group quantization is adopted, make channelNum equal to the dimension of src and groupNum equal to channelNum/group size:

.. math::

	{dst}_{i,\cdots} = (src_{i,\cdots} - zps_j)*scales_j,i\in [0,channelNum-1],j\in [0,groupNum-1]

Where:

.. math::

	i = j*groupSize+k,k\in [0,groupSize-1]

On other dimensions:

.. math::

	{dst}_{i,\cdots} = (src_{i,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1]

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  =====================================================================  ===========  ==================================================================================================================================================  =====================  
Attribute Name                                                                                                            Description                                                            Value Type   Supported Values                                                                                                                                    Required or Optional   
========================================================================================================================  =====================================================================  ===========  ==================================================================================================================================================  =====================  
:ref:`qtype <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e>`         Specifies which de-quantization type is used.                          string       ``per_tensor`` (default), ``per_channel``                                                                                                           Optional               
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`          Specifies dimension on which per-channel de-quantization is applied.   s64          An s64 value in the range of [-r, r-1] where r = rank(src), ``1`` by default. Negative values mean counting the dimension backwards from the end.   Optional               
:ref:`group_shape <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a54171e5502dcd9ca000e79099c0ab45f>`   Specifies the group shape of an operation.                             s64          An s64 list indicates the group size on the dimensions where grouped quantization is adopted.                                                       Optional               
========================================================================================================================  =====================================================================  ===========  ==================================================================================================================================================  =====================

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

   ``scales`` is a bf16/f16/f32 tensor to be applied to the de-quantization formula. For ``qtype`` = ``per-tensor``, there should be only one element in the ``scales`` tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of the src tensor along the dimension axis. For ``qtype`` = ``per-gropup``, the ``scale`` tensor should have the same number of dimension as the ``src`` tensor. On the dimensions where grouped quantization is applied, the dimension should be the number of groups, which equals to ``src_dim`` / ``group_size``, while other dimensions should match the ``src`` tensor.
   
   

.. note:: 

   ``zps`` is a tensor with offset values that map to zero. For ``qtype`` = ``per-tensor``, there should be only one element in the ``zps`` tensor. For ``qtype`` = ``per-channel``, the element number should be equal to the element number of input tensor along the dimension axis. For ``qtype`` = ``per-group``, the ``zps`` tensor should have the same number of dimensions as the ``src`` tensor. On the dimensions where grouped quantization is applied, the dimension should be the number of groups, which equals to ``src_dim`` / ``group_size``, while other dimensions should match the ``src`` tensor. If omitted, the ``zps`` values are assumed to be zero.
   
   


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

====  ===============  ===============  ============  
Src   Dst              Scales           Zps           
====  ===============  ===============  ============  
s8    f16, bf16, f32   f16, bf16, f32   s8, u8, s32   
u8    f16, bf16, f32   f16, bf16, f32   s8, u8, s32   
s4    f16, bf16, f32   f16, bf16, f32   s4, u4, s32   
u4    f16, bf16, f32   f16, bf16, f32   s4, u4, s32   
====  ===============  ===============  ============

It's expected that the data types of scales and dst should be the same.

