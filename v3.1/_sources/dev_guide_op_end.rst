.. index:: pair: page; End
.. _doxid-dev_guide_op_end:

End
===

General
~~~~~~~

End operation is used to help construct graph, for example tracking the uses of a tensor.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

End operation does not support any attribute.

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

End operation does not support output tensor.

Supported data types
~~~~~~~~~~~~~~~~~~~~

End operation supports the following data type combinations.

=====  =========  
Src    Destinat   
=====  =========  
f32    f32        
bf16   bf16       
f16    f16        
=====  =========

