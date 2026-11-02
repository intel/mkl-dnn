.. index:: pair: page; Mish
.. _doxid-dev_guide_op_mish:

Mish
====

General
~~~~~~~

Mish performs element-wise activation function on a given input tensor, based on the following mathematical formula:

.. math::

	dst = src * \tanh(SoftPlus(src)) = src * \tanh(\ln(1 + e^{src}))

Operation attributes
~~~~~~~~~~~~~~~~~~~~

Mish operation does not support any attribute.

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

Mish operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

