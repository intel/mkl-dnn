.. index:: pair: page; Clamp
.. _doxid-dev_guide_op_clamp:

Clamp
=====

General
~~~~~~~

Clamp operation represents clipping activation function, it applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	clamp(src_i) = min(max(src_i, min\_value), max\_value)

Operation attributes
~~~~~~~~~~~~~~~~~~~~

================================================================================================================  ====================================================================================================================================  ====  ==========================  =========  
Attribute Name                                                                                                    Descr                                                                                                                                 
================================================================================================================  ====================================================================================================================================  ====  ==========================  =========  
:ref:`min <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad8bd79cc131920d5de426f914d17405a>`   The lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the ``min`` value.   f32   Arbitrary valid f32 value   Required   
:ref:`max <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2ffe4e77325d9a7152f7086ea7aa5114>`   The upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the ``max`` value.   f32   Arbitrary valid f32 value   Required   
================================================================================================================  ====================================================================================================================================  ====  ==========================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``src``   Required   
======  ========  =========

Outputs
-------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``dst``   Required   
======  ========  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

Clamp operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
f16    f16    
bf16   bf16   
=====  =====

