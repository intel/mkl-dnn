.. index:: pair: page; GreaterEqual
.. _doxid-dev_guide_op_greaterequal:

GreaterEqual
============

General
~~~~~~~

The GreaterEqual operation performs an element-wise greater-than-or-equal comparison between two given tensors. This operation applies the multi-directional broadcast rules to ensure compatibility between the tensors of different shapes.

.. math::

	dst = \begin{cases} true & \text{if}\ src_0 \ge src_1 \\ false & \text{if}\ src_0 < src_1 \end{cases}

Operation Attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
Attribute Name                                                                                                               Description                                                  Value Type   Supported Values                 Required or Optional   
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
:ref:`auto_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`   Specifies rules used for auto-broadcasting of src tensors.   string       ``none`` , ``numpy`` (default)   Optional               
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================

Execution Arguments
~~~~~~~~~~~~~~~~~~~

Input
-----

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src_0``       Required               
1       ``src_1``       Required               
======  ==============  =====================

.. note:: 

   Both src shapes should match and no auto-broadcasting is allowed if the ``auto_broadcast`` attribute is ``none``. ``src_0`` and ``src_1`` shapes can be different and auto-broadcasting is allowed if the ``auto_broadcast`` attribute is ``numpy``. Broadcasting is performed according to the ``auto_broadcast`` value.
   
   


Output
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

The GreaterEqual operation supports the following data type combinations.

==============  ========  
Src_0 / Src_1   Dst       
==============  ========  
f32             boolean   
bf16            boolean   
f16             boolean   
s32             boolean   
==============  ========

