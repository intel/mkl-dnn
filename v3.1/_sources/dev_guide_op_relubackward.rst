.. index:: pair: page; ReLUBackward
.. _doxid-dev_guide_op_relubackward:

ReLUBackward
============

General
~~~~~~~

ReLUBackward operation computes gradient for ReLU.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

====================================================================================================================  ===========================================================================================  =====  ==============================  =========  
Attribute Name                                                                                                        Descr                                                                                        
====================================================================================================================  ===========================================================================================  =====  ==============================  =========  
:ref:`use_dst <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a36cda38ebb5a6a6b42b9789b20bd818c>`   If true, use ``dst`` of ReLU operation to calculate the gradient. Otherwise, use ``src`` .   bool   ``true`` (default), ``false``   Optional   
====================================================================================================================  ===========================================================================================  =====  ==============================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==================  =========  
Index   Argu                
======  ==================  =========  
0       ``src`` / ``dst``   Required   
1       ``diff_dst``        Required   
======  ==================  =========

Outputs
-------

======  =============  =========  
Index   Argu           
======  =============  =========  
0       ``diff_src``   Required   
======  =============  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

ReLUBackward operation supports the following data type combinations.

=====  =======  =====  
Src    Diff_d   
=====  =======  =====  
f32    f32      f32    
f16    f16      f16    
bf16   bf16     bf16   
=====  =======  =====

