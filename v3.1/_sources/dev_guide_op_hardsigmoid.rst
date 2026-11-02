.. index:: pair: page; HardSigmoid
.. _doxid-dev_guide_op_hardsigmoid:

HardSigmoid
===========

General
~~~~~~~

HardSigmoid operation applies the following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = \text{max}(0, \text{min}(1, \alpha src + \beta))

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  ===============================  ====  =====================  =========  
Attribute Name                                                                                                      Descr                            
==================================================================================================================  ===============================  ====  =====================  =========  
:ref:`alpha <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9>`   :math:`\alpha` in the formula.   f32   Arbitrary f32 value.   Required   
:ref:`beta <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`    :math:`\beta` in the formula.    f32   Arbitrary f32 value.   Required   
==================================================================================================================  ===============================  ====  =====================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to the index order shown below when constructing an operation.

Inputs
------

======  ==============  =========  
Index   Argument Name   Re         
======  ==============  =========  
0       ``src``         Required   
======  ==============  =========

Outputs
-------

======  ==============  =========  
Index   Argument Name   Re         
======  ==============  =========  
0       ``dst``         Required   
======  ==============  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

HardSigmoid operation supports the following data type combinations.

=====  =====  
Src    Ds     
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

