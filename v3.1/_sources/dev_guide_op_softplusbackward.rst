.. index:: pair: page; SoftPlusBackward
.. _doxid-dev_guide_op_softplusbackward:

SoftPlusBackward
================

General
~~~~~~~

SoftPlusBackward operation computes gradient for SoftPlus.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ====================================  ====  ==========================================  =========  
Attribute Name                                                                                                     Descr                                 
=================================================================================================================  ====================================  ====  ==========================================  =========  
:ref:`beta <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`   Value for the SoftPlus formulation.   f32   Arbitrary f32 value ( ``1.f`` by default)   Optional   
=================================================================================================================  ====================================  ====  ==========================================  =========

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

SoftPlusBackward operation supports the following data type combinations.

=====  =======  =====  
Src    Diff_d   
=====  =======  =====  
f32    f32      f32    
bf16   bf16     bf16   
f16    f16      f16    
=====  =======  =====

