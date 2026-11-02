.. index:: pair: page; Wildcard
.. _doxid-dev_guide_op_wildcard:

Wildcard
========

General
~~~~~~~

Wildcard operation is used to represent any compute logic and help construct graph. Typically the operation can support mapping any framework ops which are not supported by the library implementation. It's useful to make the graph completed or connected.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

Wildcard operation does not support any attribute.

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Optional               
======  ==============  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Optional               
======  ==============  =====================

.. note:: 

   WildCard operation can accept arbitrary number of inputs or outputs.
   
   


Supported data types
~~~~~~~~~~~~~~~~~~~~

Wildcard operation supports arbitrary data type combinations.

