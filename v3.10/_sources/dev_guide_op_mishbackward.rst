.. index:: pair: page; MishBackward
.. _doxid-dev_guide_op_mishbackward:

MishBackward
============

General
~~~~~~~

MishBackward operation computes gradient for Mish.

.. math::

	dst & = diff_{dst} * \frac{e^{src} * \omega}{\delta^{2}}, where \\ \omega & = e^{3src} + 4 * e^{2src} + e^{src} * (4 * src + 6) + 4 * (src + 1) \\ \delta & = e^{2src} + 2 * e^{src} + 2

Operation attributes
~~~~~~~~~~~~~~~~~~~~

MishBackward operation does not support any attribute.

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

MishBackward operation supports the following data type combinations.

=====  =========  =========  
Src    Diff_dst   Diff_src   
=====  =========  =========  
f32    f32        f32        
f16    f16        f16        
bf16   bf16       bf16       
=====  =========  =========

