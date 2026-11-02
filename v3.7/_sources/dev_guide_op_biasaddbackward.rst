.. index:: pair: page; BiasAddBackward
.. _doxid-dev_guide_op_biasaddbackward:

BiasAddBackward
===============

General
~~~~~~~

BiasAddBackward operation computes the gradients on the bias tensor for BiasAdd operator. This op accumulates all the values from :math:`\diffdst` into the channel dimension, the axis depends on the layout of :math:`\src` tensor.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  ========================================================================  ===========  ============================  =====================  
Attribute Name                                                                                                            Description                                                               Value Type   Supported Values              Required or Optional   
========================================================================================================================  ========================================================================  ===========  ============================  =====================  
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`   Controls how to interpret the shape of ``diff_dst`` and ``diff_bias`` .   string       ``NCX`` , ``NXC`` (default)   Optional               
========================================================================================================================  ========================================================================  ===========  ============================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_dst``    Required               
======  ==============  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``diff_bias``   Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

BiasAddBackward operation supports the following data type combinations.

=========  ==========  
Diff_dst   Diff_bias   
=========  ==========  
f32        f32         
bf16       bf16        
f16        f16         
=========  ==========

