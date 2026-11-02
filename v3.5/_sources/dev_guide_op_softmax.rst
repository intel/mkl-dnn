.. index:: pair: page; SoftMax
.. _doxid-dev_guide_op_softmax:

SoftMax
=======

General
~~~~~~~

SoftMax operation applies the following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst_i = \frac{exp(src_i)}{\sum_{j=1}^{C} exp(src_j)}

where :math:`C` is a size of tensor along axis dimension.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ==========================================================  ===========  ========================================  =====================  
Attribute Name                                                                                                     Description                                                 Value Type   Supported Values                          Required or Optional   
=================================================================================================================  ==========================================================  ===========  ========================================  =====================  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Represents the axis from which the SoftMax is calculated.   s64          Arbitrary s64 value ( ``1`` in default)   Optional               
=================================================================================================================  ==========================================================  ===========  ========================================  =====================

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

SoftMax operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

