.. index:: pair: page; StaticTranspose
.. _doxid-dev_guide_op_statictranspose:

StaticTranspose
===============

General
~~~~~~~

StaticTranspose operation rearranges the dimensions of :math:`\src`. :math:`\dst` may have a different memory layout from :math:`\src`. StaticTranspose operation is not guaranteed to return a view or a copy of :math:`\src` when :math:`\dst` is in-placed with the :math:`\src`.

.. math::

	dst[src(order[0]), src(order[1]),\cdots, src(order[N-1])]\ =src[src(0), src(1),\cdots, src(N-1)]

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  =================================================  ====  ==========================================================================================================  =========  
Attribute Name                                                                                                      De                                                 
==================================================================================================================  =================================================  ====  ==========================================================================================================  =========  
:ref:`order <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a70a17ffa722a3985b86d30b034ad06d7>`   Specifies permutation to be applied on ``src`` .   s64   A s64 list containing the element in the range of [-N, N-1], negative value means counting from last axis   Required   
==================================================================================================================  =================================================  ====  ==========================================================================================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``src``         Required               
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

StaticTranspose operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

