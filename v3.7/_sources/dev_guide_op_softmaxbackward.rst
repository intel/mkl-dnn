.. index:: pair: page; SoftMaxBackward
.. _doxid-dev_guide_op_softmaxbackward:

SoftMaxBackward
===============

General
~~~~~~~

SoftMaxBackward operation computes gradient for SoftMax.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ==========================================================  ===========  ========================================  =====================  
Attribute Name                                                                                                     Description                                                 Value Type   Supported Values                          Required or Optional   
=================================================================================================================  ==========================================================  ===========  ========================================  =====================  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Represents the axis from which the SoftMax is calculated.   s64          Arbitrary s64 value ( ``1`` in default)   Optional               
=================================================================================================================  ==========================================================  ===========  ========================================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_dst``    Required               
1       ``dst``         Required               
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

SoftMaxBackward operation supports the following data type combinations.

=====  =========  =========  
Dst    Diff_dst   Diff_src   
=====  =========  =========  
f32    f32        f32        
bf16   bf16       bf16       
f16    f16        f16        
=====  =========  =========

