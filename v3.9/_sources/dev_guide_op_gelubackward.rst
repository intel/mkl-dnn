.. index:: pair: page; GELUBackward
.. _doxid-dev_guide_op_gelubackward:

GELUBackward
============

General
~~~~~~~

GELUBackward operation computes gradient for GELU.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  ================================================  ===========  ======================================  =====================  
Attribute Name                                                                                                     Description                                       Value Type   Supported Values                        Required or Optional   
=================================================================================================================  ================================================  ===========  ======================================  =====================  
:ref:`mode <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a15d61712450a686a7f365adf4fef581f>`   Specifies the computation mode of GELUBackward.   string       ``gelu_erf`` (default), ``gelu_tanh``   Optional               
=================================================================================================================  ================================================  ===========  ======================================  =====================

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

