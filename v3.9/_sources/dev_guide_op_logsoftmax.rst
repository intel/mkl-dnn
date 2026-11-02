.. index:: pair: page; LogSoftmax
.. _doxid-dev_guide_op_logsoftmax:

LogSoftmax
==========

General
~~~~~~~

LogSoftmax operation applies the :math:`\log(softmax(src))` function to an n-dimensional input Tensor. The formulation can be simplified as:

.. math::

	dst_i = \log\Big( \frac{exp(src_i)}{\sum_{j}^{ } exp(src_j)} \Big)

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ===================================================================================================================  ===========  =========================================  =====================  
Attribute Name                                                                                                     Description                                                                                                          Value Type   Supported Values                           Required or Optional   
=================================================================================================================  ===================================================================================================================  ===========  =========================================  =====================  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Represents the axis of which the LogSoftmax is calculated. Negative value means counting dimensions from the back.   s64          Arbitrary s64 value ( ``-1`` in default)   Optional               
=================================================================================================================  ===================================================================================================================  ===========  =========================================  =====================

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

LogSoftmax operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

