.. index:: pair: page; RMSNorm
.. _doxid-dev_guide_op_rmsnorm:

RMSNorm
=======

General
~~~~~~~

RMSNorm (Root Mean Square Layer Normalization) operation performs normalization on the input tensor using the root mean square statistic.

The RMSNorm operation performs the following transformation of the input tensor:

.. math::

	y = \gamma \cdot \frac{x}{\sqrt{\text{RMS}(x) + \epsilon}},

where

.. math::

	\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

============================================================================================================================  ================================================================================================================================================================================================================================================================================================  ===========  =================================================  =====================  
Attribute Name                                                                                                                Description                                                                                                                                                                                                                                                                                       Value Type   Supported Values                                   Required or Optional   
============================================================================================================================  ================================================================================================================================================================================================================================================================================================  ===========  =================================================  =====================  
:ref:`epsilon <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8>`           The constant to improve numerical stability.                                                                                                                                                                                                                                                      f32          Arbitrary positive f32 value, ``1e-5`` (default)   Optional               
:ref:`begin_norm_axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac4fe88742dd733999b9a5e4db0322415>`   ``begin_norm_axis`` is used to indicate which axis to start RMS normalization. The normalization is from ``begin_norm_axis`` to last dimension. Negative values means indexing from right to left. This op normalizes over the last dimension by default, e.g. C in TNC for 3D and LDNC for 4D.   s64          [-r,r-1],where r=rank(src). -1 is default          Optional               
============================================================================================================================  ================================================================================================================================================================================================================================================================================================  ===========  =================================================  =====================

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
======  ==============  =====================

.. note:: 

   ``gamma`` is scaling for the normalized value. ``gamma`` shape should be broadcastable to the ``src`` shape.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

RMSNorm operation supports the following data type combinations.

==========  ==========  
Src / Dst   Gamma       
==========  ==========  
f32         f32         
bf16        f32, bf16   
f16         f32, f16    
==========  ==========

