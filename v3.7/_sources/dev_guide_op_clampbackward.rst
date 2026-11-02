.. index:: pair: page; ClampBackward
.. _doxid-dev_guide_op_clampbackward:

ClampBackward
=============

General
~~~~~~~

ClampBackward operation computes gradient for Clamp.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

====================================================================================================================  ====================================================================================================================================  ===========  ==============================  =====================  
Attribute Name                                                                                                        Description                                                                                                                           Value Type   Supported Values                Required or Optional   
====================================================================================================================  ====================================================================================================================================  ===========  ==============================  =====================  
:ref:`min <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad8bd79cc131920d5de426f914d17405a>`       The lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the ``min`` value.   f32          Arbitrary valid f32 value       Required               
:ref:`max <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2ffe4e77325d9a7152f7086ea7aa5114>`       The upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the ``max`` value.   f32          Arbitrary valid f32 value       Required               
:ref:`use_dst <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a36cda38ebb5a6a6b42b9789b20bd818c>`   If true, use ``dst`` of Clamp operation to calculate the gradient. Otherwise, use ``src`` .                                           bool         ``true`` (default), ``false``   Optional               
====================================================================================================================  ====================================================================================================================================  ===========  ==============================  =====================

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

ClampBackward operation supports the following data type combinations.

==========  =========  =========  
Src / Dst   Diff_dst   Diff_src   
==========  =========  =========  
f32         f32        f32        
f16         f16        f16        
bf16        bf16       bf16       
==========  =========  =========

