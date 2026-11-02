.. index:: pair: page; LayerNorm
.. _doxid-dev_guide_op_layernorm:

LayerNorm
=========

General
~~~~~~~

LayerNorm performs a layer normalization operation on :math:`\src` tensor.

The layerNorm operation performs normalization from ``begin_norm_axis`` to last dimension of the data tensor. It is defined by the following formulas which is the same as :ref:`Layer Normalization <doxid-dev_guide_layer_normalization>`.

.. math::

	\dst(t, n, c) = \gamma(c) \cdot \frac{\src(t, n, c) - \mu(t, n)} {\sqrt{\sigma^2(t, n) + \epsilon}} + \beta(c),

where

* :math:`\gamma(c), \beta(c)` are optional scale and shift for a channel

* :math:`\mu(t, n), \sigma^2(t, n)` are mean and variance (see

* :math:`\epsilon` is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and variance are computed at runtime, the following formulas are used:

* :math:`\mu(t, n) = \frac{1}{C} \sum\limits_{c} \src(t, n, c)_{}`,

* :math:`\sigma^2(t, n) = \frac{1}{C} \sum\limits_{c} {}_{} (\src(t, n, c) - \mu(t, n))^2`.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

============================================================================================================================  ==================================================================================================================================================================================================================================================================================================  =====  =================================================  =========  
Attribute Name                                                                                                                De                                                                                                                                                                                                                                                                                                  
============================================================================================================================  ==================================================================================================================================================================================================================================================================================================  =====  =================================================  =========  
:ref:`keep_stats <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac83b685e59ae9a2f78e9996886186e99>`        Indicate whether to output mean and variance which can be later passed to backward op.                                                                                                                                                                                                              bool   ``false`` , ``true`` (default)                     Optional   
:ref:`begin_norm_axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac4fe88742dd733999b9a5e4db0322415>`   ``begin_norm_axis`` is used to indicate which axis to start layer normalization. The normalization is from ``begin_norm_axis`` to last dimension. Negative values means indexing from right to left. This op normalizes over the last dimension by default, e.g. C in TNC for 3D and LDNC for 4D.   s64    [-r,r-1],where r=rank(src). -1 is default          Optional   
:ref:`use_affine <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a014a6940b2c348a18720fcc350cb8e16>`        When set to True, this module has learnable per-element affine parameters.                                                                                                                                                                                                                          bool   ``false`` , ``true`` (default)                     Optional   
:ref:`epsilon <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8>`           The constant to improve numerical stability.                                                                                                                                                                                                                                                        f32    Arbitrary positive f32 value, ``1e-5`` (default)   Optional   
============================================================================================================================  ==================================================================================================================================================================================================================================================================================================  =====  =================================================  =========

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

   ``gamma`` is scaling for normalized value. ``beta`` is the bias added to the scaled normalized value. They are both 1D tensor with the same span as src’s channel axis and required if attribute ``use_affine`` is set to True.
   
   


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

   Both ``mean`` and ``variance`` are required if attribute ``keep_stats`` is set to True.
   
   


Supported data types
~~~~~~~~~~~~~~~~~~~~

LayerNorm operation supports the following data type combinations.

==========  ==========  
Src / Dst   G           
==========  ==========  
f32         f32         
bf16        f32, bf16   
f16         f32         
==========  ==========

