.. index:: pair: page; LogSoftmaxBackward
.. _doxid-dev_guide_op_logsoftmaxbackward:

LogSoftmaxBackward
==================

General
~~~~~~~

LogSoftmaxBackward operation computes gradient for LogSoftmax.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ===================================================================================================================  ====  =========================================  =========  
Attribute Name                                                                                                     Descr                                                                                                                
=================================================================================================================  ===================================================================================================================  ====  =========================================  =========  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Represents the axis of which the LogSoftmax is calculated. Negative value means counting dimensions from the back.   s64   Arbitrary s64 value ( ``-1`` in default)   Optional   
=================================================================================================================  ===================================================================================================================  ====  =========================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_dst``   Required   
1       ``src``        Required   
======  =============  =========

Outputs
-------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_src``   Required   
======  =============  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

LogSoftmaxBackward operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

