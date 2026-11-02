.. index:: pair: page; SoftPlus
.. _doxid-dev_guide_op_softplus:

SoftPlus
========

General
~~~~~~~

SoftPlus operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = 1 / beta * \ln(e^{beta*src} + 1.0)

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ====================================  ===========  ==========================================  =====================  
Attribute Name                                                                                                     Description                           Value Type   Supported Values -------------------        Required or Optional   
=================================================================================================================  ====================================  ===========  ==========================================  =====================  
:ref:`beta <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`   Value for the SoftPlus formulation.   f32          Arbitrary f32 value ( ``1.f`` by default)   Optional               
=================================================================================================================  ====================================  ===========  ==========================================  =====================

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

SoftPlus operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

