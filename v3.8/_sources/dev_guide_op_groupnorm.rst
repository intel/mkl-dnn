.. index:: pair: page; GroupNorm
.. _doxid-dev_guide_op_groupnorm:

GroupNorm
=========

General
~~~~~~~

The GroupNorm operation performs the following transformation of the input tensor:

.. math::

	y = \gamma \cdot \frac{(x - mean)} {\sqrt{variance + \epsilon}} + \beta,

The operation is applied per batch, per group of channels. The gamma and beta coefficients are the optional inputs to the model and need to be specified separately for each channel. The ``mean`` and ``variance`` are calculated for each group.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  ===============================================================================================================================================  ===========  ===============================================================  =====================  
Attribute Name                                                                                                            Description                                                                                                                                      Value Type   Supported Values                                                 Required or Optional   
========================================================================================================================  ===============================================================================================================================================  ===========  ===============================================================  =====================  
:ref:`groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`        Specifies the number of groups ``G`` that the channel dimension will be divided into. ``groups`` should be divisible by the number of channels   s64          between 1 and the number of channels ``C`` in the input tensor   Required               
:ref:`keep_stats <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac83b685e59ae9a2f78e9996886186e99>`    Indicate whether to output mean and variance which can be later passed to backward op.                                                           bool         ``false`` , ``true`` (default is true)                           Optional               
:ref:`use_affine <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a014a6940b2c348a18720fcc350cb8e16>`    When set to True, this module has inputs ``gamma`` and ``beta``                                                                                  bool         ``false`` , ``true`` (default is true)                           Optional               
:ref:`epsilon <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8>`       The constant to improve numerical stability.                                                                                                     f32          Arbitrary positive f32 value, ``1e-5`` (default)                 Optional               
:ref:`data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`   Controls how to interpret the shape of ``src`` and ``dst`` .                                                                                     string       ``NCX`` , ``NXC`` (default is ``NXC`` )                          Optional               
========================================================================================================================  ===============================================================================================================================================  ===========  ===============================================================  =====================  



Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
1       ``gamma``       Optional               
2       ``beta``        Optional               
======  ==============  =====================

.. note:: 

   ``gamma`` is scaling for the normalized value. ``beta`` is the bias added to the scaled normalized value. They are both 1D tensor with the same span as src’s channel axis and required if the attribute ``use_affine`` is set to True.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
1       ``mean``        Optional               
2       ``variance``    Optional               
======  ==============  =====================

.. note:: 

   Both ``mean`` and ``variance`` are required if the attribute ``keep_stats`` is set to ``True``.
   
   


Supported data types
~~~~~~~~~~~~~~~~~~~~

GroupNorm operation supports the following data type combinations.

==========  ===============================  
Src / Dst   Gamma / Beta / Mean / Variance   
==========  ===============================  
f32         f32                              
bf16        f32, bf16                        
f16         f32                              
==========  ===============================

