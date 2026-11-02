.. index:: pair: page; SquaredDifference
.. _doxid-dev_guide_op_squareddifference:

SquaredDifference
=================

General
~~~~~~~

SquaredDifference operation performs element-wise subtraction operation with two given tensors applying multi-directional broadcast rules, after that each result of the subtraction is squared.

Before performing arithmetic operation, :math:`src_0` and :math:`src_1` are broadcasted if their shapes are different and ``auto_broadcast`` attributes is not ``none``. Broadcasting is performed according to ``auto_broadcast`` value. After broadcasting SquaredDifference does the following with the input tensors:

.. math::

	dst_i = (src\_0_i - src\_1_i)^2

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  =============================================================  ===========  ===============================  =====================  
Attribute Name                                                                                                               Description                                                    Value Type   Supported Values                 Required or Optional   
===========================================================================================================================  =============================================================  ===========  ===============================  =====================  
:ref:`auto_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`   Specifies rules used for auto-broadcasting of input tensors.   string       ``none`` , ``numpy`` (default)   Optional               
===========================================================================================================================  =============================================================  ===========  ===============================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src_0``       Required               
1       ``src_1``       Required               
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

SquaredDifference operation supports the following data type combinations.

==============  =====  
Src_0 / Src_1   Dst    
==============  =====  
f32             f32    
bf16            bf16   
f16             f16    
==============  =====

