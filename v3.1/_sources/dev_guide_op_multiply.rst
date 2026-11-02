.. index:: pair: page; Multiply
.. _doxid-dev_guide_op_multiply:

Multiply
========

General
~~~~~~~

Multiply operation performs element-wise multiply operation with two given tensors applying multi-directional broadcast rules.

.. math::

	\dst(\overline{x}) = \src_0(\overline{x}) \times \src_1(\overline{x}),

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===========================================================  =======  ===============================  =========  
Attribute Name                                                                                                               De                                                           
===========================================================================================================================  ===========================================================  =======  ===============================  =========  
:ref:`auto_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`   Specifies rules used for auto-broadcasting of src tensors.   string   ``none`` , ``numpy`` (default)   Optional   
===========================================================================================================================  ===========================================================  =======  ===============================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src_0``       Required               
1       ``src_1``       Required               
======  ==============  =====================

.. note:: 

   Both src shapes should match and no auto-broadcasting is allowed if ``auto_broadcast`` attributes is ``none``. ``src_0`` and ``src_1`` shapes can be different and auto-broadcasting is allowed if ``auto_broadcast`` attributes is ``numpy``. Broadcasting is performed according to auto_broadcast value.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

Multiply operation supports the following data type combinations.

==========  =====  
Source0/1   D      
==========  =====  
f32         f32    
bf16        bf16   
f16         f16    
==========  =====

