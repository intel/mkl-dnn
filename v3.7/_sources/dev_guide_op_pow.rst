.. index:: pair: page; Pow
.. _doxid-dev_guide_op_pow:

Pow
===

General
~~~~~~~

Pow operation performs an element-wise power operation on a given input tensor with a single value attribute beta as its exponent. It is based on the following mathematical formula:

.. math::

	dst_{i} = {src_{i} ^ \beta}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ========================================  ===========  =====================  =====================  
Attribute Name                                                                                                     Description                               Value Type   Supported Values       Required or Optional   
=================================================================================================================  ========================================  ===========  =====================  =====================  
:ref:`beta <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`   exponent, :math:`\beta` in the formula.   f32          Arbitrary f32 value.   Required               
=================================================================================================================  ========================================  ===========  =====================  =====================

Inputs
~~~~~~

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
======  ==============  =====================

Outputs
~~~~~~~

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

Pow operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

