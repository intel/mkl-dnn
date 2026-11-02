.. index:: pair: page; Dequantize
.. _doxid-dev_guide_op_dequantize:

Dequantize
==========

General
~~~~~~~

Dequantize operation converts a quantized (u8 or s8) tensor to a f32 tensor. It supports both per-tensor and per-channel asymmetric linear de-quantization. Rounding mode is library-implementation defined.

For per-tensor de-quantization:

.. math::

	\dst_{i} = round((\src_{i} - zps) \times scale)

For per-channel de-quantization, taking channel axis = 1 as an example:

.. math::

	dst_{\cdots,i,\cdots,\cdots} = (\src_{\cdots,i,\cdots,\cdots} - zps_i) \times scale_i, i \in {[0, ic-1]}

where :math:`ic` is the number of channels.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===================================================================================================================  =====================================================================  ===========  ============================================================================  =====================  
Attribute Name                                                                                                       Description                                                            Value Type   Supported Values                                                              Required or Optional   
===================================================================================================================  =====================================================================  ===========  ============================================================================  =====================  
:ref:`qtype <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e>`    Specifies which de-quantization type is used.                          string       ``per_tensor`` (default), ``per_channel``                                     Optional               
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`     Specifies dimension on which per-channel de-quantization is applied.   s64          A s64 value in the range of [-r, r-1] where r = rank(src), ``1`` by default   Optional               
:ref:`scales <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8e7bb02b763a2e07d30b4ab24beb7fa1>`   Scalings applied on the src data.                                      f32          A f32 list (only contain one element if qtype is ``per_tensor`` )             Required               
:ref:`zps <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a5c284a074767998e9708c3656d41a91c>`      Offset values that maps to float zero.                                 s64          A s64 list (only contain one element if qtype is ``per_tensor`` )             Required               
===================================================================================================================  =====================================================================  ===========  ============================================================================  =====================

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

Dequantize operation supports the following data type combinations.

=======  ====  
Src      Dst   
=======  ====  
s8, u8   f32   
=======  ====

.. note:: 

   This operation is to support :ref:`int8 quantization <doxid-dev_guide_graph_low_precision_1dev_guide_graph_int8_quantization_model>` model.

