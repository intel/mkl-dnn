.. index:: pair: page; ReLU
.. _doxid-dev_guide_op_relu:

ReLU
====

General
~~~~~~~

ReLU applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = \begin{cases} src & \text{if}\ src > 0 \\ 0 & \text{if}\ src \leq 0 \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

ReLU operation does not support any attribute.

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

ReLU operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

