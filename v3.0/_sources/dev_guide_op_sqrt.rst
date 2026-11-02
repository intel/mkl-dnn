.. index:: pair: page; Sqrt
.. _doxid-dev_guide_op_sqrt:

Sqrt
====

General
~~~~~~~

Sqrt operation performs element-wise square root operation with given tensor.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

Sqrt operation does not support any attribute.

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

Sqrt operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
f16    f16    
bf16   bf16   
=====  =====

