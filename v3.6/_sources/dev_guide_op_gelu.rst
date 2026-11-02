.. index:: pair: page; GELU
.. _doxid-dev_guide_op_gelu:

GELU
====

General
~~~~~~~

GELU operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = 0.5 * src * (1.0 + erf(src) / \sqrt2)

Operation attributes
~~~~~~~~~~~~~~~~~~~~

GELU operation does not support any attribute.

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

GELU operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
f16    f16    
bf16   bf16   
=====  =====

