.. index:: pair: page; BatchNormInference
.. _doxid-dev_guide_op_batchnorminference:

BatchNormInference
==================

General
~~~~~~~

The formula is the same as :ref:`Batch Normalization primitive <doxid-dev_guide_batch_normalization>` like below.

.. math::

	\dst(n, c, h, w) = \gamma(c) \cdot \frac{\src(n, c, h, w) - \mu(c)} {\sqrt{\sigma^2(c) + \varepsilon}} + \beta(c),

where

* :math:`\gamma(c), \beta(c)` are required scale and shift for a channel,

* :math:`\mu(c), \sigma^2(c)` are mean and variance for a channel, and

* :math:`\varepsilon` is a constant to improve numerical stability.

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
1       ``gamma``                           Required               
2       ``beta``                            Required               
3       ``mean``                            Required               
4       ``variance`` ( :math:`\sigma^2` )   Required               
======  ==================================  =====================

Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

BatchNormInference operation supports the following data type combinations.

==========  ===============================  
Src / Dst   Gamma / Beta / Mean / Variance   
==========  ===============================  
f32         f32                              
bf16        f32, bf16                        
f16         f32                              
==========  ===============================

