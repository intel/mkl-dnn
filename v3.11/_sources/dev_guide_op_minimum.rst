.. index:: pair: page; Minimum
.. _doxid-dev_guide_op_minimum:

Minimum
=======

General
~~~~~~~

Minimum operation performs element-wise minimum operation with two given tensors applying multi-directional broadcast rules.

.. math::

	\dst(\overline{x})) = min(\src\_0(\overline{x}), \src\_1(\overline{x}))

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
Attribute Name                                                                                                               Description                                                  Value Type   Supported Values                 Required or Optional   
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
:ref:`auto_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`   Specifies rules used for auto-broadcasting of src tensors.   string       ``none`` , ``numpy`` (default)   Optional               
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================

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

Minimum operation supports the following data type combinations.

==============  =====  
Src_0 / Src_1   Dst    
==============  =====  
f32             f32    
bf16            bf16   
f16             f16    
==============  =====

