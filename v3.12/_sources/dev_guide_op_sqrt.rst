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

Sqrt operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
f16    f16    
bf16   bf16   
=====  =====

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

Sqrt supports in-place operations, meaning that ``src`` can be used as both input and output (``dst``). In case of in-place operation, the original ``src`` data will be overwritten. Use in-place operations whenever possible for performance.

