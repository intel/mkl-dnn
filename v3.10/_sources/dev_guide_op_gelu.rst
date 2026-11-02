.. index:: pair: page; GELU
.. _doxid-dev_guide_op_gelu:

GELU
====

General
~~~~~~~

GELU operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = 0.5 * src * (1.0 + erf(src) / \sqrt2)

When the attribute ``mode`` is specified as ``gelu_tanh``, an approximation implementation is used:

.. math::

	dst = 0.5 * src * (1.0 + \tanh[\sqrt{\frac{2}{\pi}} (src + 0.044715 * s^3)])

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ========================================  ===========  ======================================  =====================  
Attribute Name                                                                                                     Description                               Value Type   Supported Values                        Required or Optional   
=================================================================================================================  ========================================  ===========  ======================================  =====================  
:ref:`mode <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a15d61712450a686a7f365adf4fef581f>`   Specifies the computation mode of GELU.   string       ``gelu_erf`` (default), ``gelu_tanh``   Optional               
=================================================================================================================  ========================================  ===========  ======================================  =====================

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

