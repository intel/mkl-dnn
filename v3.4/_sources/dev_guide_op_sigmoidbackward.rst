.. index:: pair: page; SigmoidBackward
.. _doxid-dev_guide_op_sigmoidbackward:

SigmoidBackward
===============

General
~~~~~~~

SigmoidBackward operation computes gradient for Sigmoid.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

====================================================================================================================  ==============================================================================================  ===========  ==============================  =====================  
Attribute Name                                                                                                        Description                                                                                     Value Type   Supported Values                Required or Optional   
====================================================================================================================  ==============================================================================================  ===========  ==============================  =====================  
:ref:`use_dst <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a36cda38ebb5a6a6b42b9789b20bd818c>`   If true, use ``dst`` of Sigmoid operation to calculate the gradient. Otherwise, use ``src`` .   bool         ``true`` (default), ``false``   Optional               
====================================================================================================================  ==============================================================================================  ===========  ==============================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==================  =====================  
Index   Argument Name       Required or Optional   
======  ==================  =====================  
0       ``src`` / ``dst``   Required               
1       ``diff_dst``        Required               
======  ==================  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_src``    Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

SigmoidBackward operation supports the following data type combinations.

==========  =========  =========  
Src / Dst   Diff_dst   Diff_src   
==========  =========  =========  
f32         f32        f32        
f16         f16        f16        
bf16        bf16       bf16       
==========  =========  =========

