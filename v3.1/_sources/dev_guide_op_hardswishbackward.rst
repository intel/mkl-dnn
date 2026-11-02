.. index:: pair: page; HardSwishBackward
.. _doxid-dev_guide_op_hardswishbackward:

HardSwishBackward
=================

General
~~~~~~~

HardSwishBackward operation computes gradient for HardSwish.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

HardSwishBackward operation does not support any attribute.

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``src``        Required   
1       ``diff_dst``   Required   
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

HardSwishBackward operation supports the following data type combinations.

=====  =======  =====  
Src    Diff_d   
=====  =======  =====  
f32    f32      f32    
f16    f16      f16    
bf16   bf16     bf16   
=====  =======  =====

