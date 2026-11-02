.. index:: pair: page; GenIndex
.. _doxid-dev_guide_op_genindex:

GenIndex
========

General
~~~~~~~

The GenIndex operation creates an index tensor along a specified axis of an input tensor. The resulting index tensor has the same shape as the input tensor, with each element representing the index along the specified axis.

Operation Attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ================================================================  ===========  ===========================================================  =====================  
Attribute Name                                                                                                     Description                                                       Value Type   Supported Values                                             Required or Optional   
=================================================================================================================  ================================================================  ===========  ===========================================================  =====================  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Specifies the dimension along which index values are generated.   s64          An s64 value in the range of [-r, r-1] where r = rank(src)   Required               
=================================================================================================================  ================================================================  ===========  ===========================================================  =====================

Execution Arguments
~~~~~~~~~~~~~~~~~~~

Input
-----

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
======  ==============  =====================

Output
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

The GenIndex operation supports the following data type combinations.

=====  ====  
Src    Dst   
=====  ====  
f32    s32   
bf16   s32   
f16    s32   
=====  ====

