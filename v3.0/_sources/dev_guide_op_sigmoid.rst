.. index:: pair: page; Sigmoid
.. _doxid-dev_guide_op_sigmoid:

Sigmoid
=======

General
~~~~~~~

Sigmoid operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = \frac{1}{1 + e^{-src}}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

Sigmoid operation does not support any attribute.

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

Sigmoid operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

