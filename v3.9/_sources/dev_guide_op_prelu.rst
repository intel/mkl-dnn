.. index:: pair: page; PReLU
.. _doxid-dev_guide_op_prelu:

PReLU
=====

General
~~~~~~~

PReLU operation performs element-wise parametric ReLU operation on a given input tensor, based on the following mathematical formula:

.. math::

	dst = \begin{cases} src & \text{if}\ src \ge 0 \\ \alpha src & \text{if}\ src < 0 \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================================  ========================================================================  ===========  ===============================  =====================  
Attribute Name                                                                                                                      Description                                                               Value Type   Supported Values                 Required or Optional   
==================================================================================================================================  ========================================================================  ===========  ===============================  =====================  
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`             Denotes the data format of the input and output data.                     string       ``NCX`` , ``NXC`` (default)      Optional               
:ref:`per_channel_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a652a82e843431baeacb5dfdedfd49d12>`   Denotes whether to apply per_channel broadcast when slope is 1D tensor.   bool         ``false`` , ``true`` (default)   Optional               
==================================================================================================================================  ========================================================================  ===========  ===============================  =====================

Broadcasting Rules
------------------

Only slope tensor supports broadcasting semantics. Slope tensor is uni-directionally broadcasted to :math:`\src` if one of the following rules is met:

* 1: slope is 1D tensor and ``per_channel_broadcast`` is set to ``true``, the length of slope tensor is equal to the length of :math:`\src` of channel dimension.

* 2: slope is 1D tensor and ``per_channel_broadcast`` is set to ``false``, the length of slope tensor is equal to the length of :math:`\src` of the rightmost dimension.

* 3: slope is nD tensor, starting from the rightmost dimension, :math:`input.shape_i == slope.shape_i` or :math:`slope.shape_i == 1` or slope dimension i is empty.

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
======  ==============  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

PReLU operation supports the following data type combinations.

=====  =====  ======  
Src    Dst    Slope   
=====  =====  ======  
f32    f32    f32     
bf16   bf16   bf16    
f16    f16    f16     
=====  =====  ======

