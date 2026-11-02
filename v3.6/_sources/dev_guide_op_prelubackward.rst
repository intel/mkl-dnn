.. index:: pair: page; PReLUBackward
.. _doxid-dev_guide_op_prelubackward:

PReLUBackward
=============

General
~~~~~~~

PReLUBackward operation computes gradient for PReLU.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  ======================================================  ===========  ============================  =====================  
Attribute Name                                                                                                            Description                                             Value Type   Supported Values              Required or Optional   
========================================================================================================================  ======================================================  ===========  ============================  =====================  
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`   Denotes the data format of the input and output data.   string       ``NCX`` , ``NXC`` (default)   Optional               
========================================================================================================================  ======================================================  ===========  ============================  =====================

Broadcasting Rules
------------------

Only slope tensor supports broadcasting semantics. Slope tensor is uni-directionally broadcasted to :math:`\src` if one of the following rules is met:

#. PyTorch case: slope is 1D tensor and broadcast per channel, length of slope is equal to the length of :math:`\src` in channel dimension.

#. PyTorch case: slope is 1D tensor and broadcast per tensor, length of slope is equal to 1.

#. Tensorflow case: slope is nD tensor and its dimensions must be equal to the :math:`\src` dimensions starting from the second element: :math:`slope\_shape = input\_forward\_shape[1:]`

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
1       ``slope``       Required               
2       ``diff_dst``    Required               
======  ==============  =====================

Outputs
-------

======  ===============  =====================  
Index   Argument Name    Required or Optional   
======  ===============  =====================  
0       ``diff_src``     Required               
1       ``diff_slope``   Required               
======  ===============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

PReLUBackward operation supports the following data type combinations.

=====  ======  =========  =========  ===========  
Src    Slope   Diff_dst   Diff_src   Diff_slope   
=====  ======  =========  =========  ===========  
f32    f32     f32        f32        f32          
bf16   bf16    bf16       bf16       bf16         
f16    f16     f16        f16        f16          
=====  ======  =========  =========  ===========

