.. index:: pair: page; GELUBackward
.. _doxid-dev_guide_op_gelubackward:

GELUBackward
============

General
~~~~~~~

GELUBackward operation computes gradient for GELU.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

GELUBackward operation does not support any attribute.

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
1       ``diff_dst``    Required               
======  ==============  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_src``    Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

GELUBackward operation supports the following data type combinations.

=====  =========  =========  
Src    Diff_dst   Diff_src   
=====  =========  =========  
f32    f32        f32        
f16    f16        f16        
bf16   bf16       bf16       
=====  =========  =========

