.. index:: pair: page; Elu
.. _doxid-dev_guide_op_elu:

Elu
===

General
~~~~~~~

Elu operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = \begin{cases} \alpha(e^{src} - 1) & \text{if}\ src < 0 \\ src & \text{if}\ src \ge 0 \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  ===============================  ===========  =================================  =====================  
Attribute Name                                                                                                      Description                      Value Type   Supported Values                   Required or Optional   
==================================================================================================================  ===============================  ===========  =================================  =====================  
:ref:`alpha <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9>`   Scale for the negative factor.   f32          Arbitrary non-negative f32 value   Required               
==================================================================================================================  ===============================  ===========  =================================  =====================

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

Elu operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
f16    f16    
bf16   bf16   
=====  =====

