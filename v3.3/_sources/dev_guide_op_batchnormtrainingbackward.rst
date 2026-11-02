.. index:: pair: page; BatchNormTrainingBackward
.. _doxid-dev_guide_op_batchnormtrainingbackward:

BatchNormTrainingBackward
=========================

General
~~~~~~~

BatchNormTrainingBackward operation calculated the gradients of input tensors.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  ================================================================  ===========  ============================  =====================  
Attribute Name                                                                                                            Description                                                       Value Type   Supported Values              Required or Optional   
========================================================================================================================  ================================================================  ===========  ============================  =====================  
:ref:`epsilon <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8>`       A number to be added to the variance to avoid division by zero.   f32          A positive float value        Required               
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`   Controls how to interpret the shape of ``src`` and ``dst`` .      string       ``NCX`` , ``NXC`` (default)   Optional               
========================================================================================================================  ================================================================  ===========  ============================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==================================  =====================  
Index   Argument Name                       Required or Optional   
======  ==================================  =====================  
0       ``src``                             Required               
1       ``diff_dst``                        Required               
2       ``mean``                            Required               
3       ``variance`` ( :math:`\sigma^2` )   Required               
4       ``gamma``                           Optional               
======  ==================================  =====================

Outputs
-------

======  ===============  =====================  
Index   Argument Name    Required or Optional   
======  ===============  =====================  
0       ``diff_src``     Required               
1       ``diff_gamma``   Optional               
2       ``diff_beta``    Optional               
======  ===============  =====================

.. note:: 

   ``diff_gamma`` and ``diff_beta`` should be either both provided or neither provided. If neither provided, the input ``gamma`` will be ignored.
   
   


Supported data types
~~~~~~~~~~~~~~~~~~~~

BatchNormTrainingBackward operation supports the following data type combinations.

==========================  =================================================  
Src / Diff_dst / Diff_src   Mean / Variance / Gamma / Diff_gamma / Diff_beta   
==========================  =================================================  
f32                         f32                                                
bf16                        f32, bf16                                          
f16                         f32                                                
==========================  =================================================

