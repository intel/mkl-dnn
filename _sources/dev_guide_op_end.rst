:orphan:

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

End operation supports the following data types for ``src`` : f32, f16, bf16, s8, u8, s32.

