.. index:: pair: page; MatMul
.. _doxid-dev_guide_op_matmul:

MatMul
======

General
~~~~~~~

MatMul operation computes the product of two tensors with optional bias addition The variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`, typically taking 2D input tensors as an example, the formula is below:

.. math::

	\dst(m, n) = \sum_{k=0}^{K - 1} \left( \src(m, k) \cdot \weights(k, n) \right) + \bias(m, n)

In the shape of a tensor, two right-most axes are interpreted as row and column dimensions of a matrix while all left-most axes (if present) are interpreted as batch dimensions. The operation supports broadcasting semantics for those batch dimensions. For example :math:`\src` can be broadcasted to :math:`\weights` if the corresponding dimension in :math:`\src` is ``1`` (and vice versa). Additionally, if ranks of :math:`\src` and :math:`\weights` are different, the tensor with a smaller rank will be unsqueezed from the left side of dimensions (inserting ``1``) to make sure two ranks matched.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

========================================================================================================================  =======================================================================  =====  ======================  =========  
Attribute Name                                                                                                            De                                                                       
========================================================================================================================  =======================================================================  =====  ======================  =========  
:ref:`transpose_a <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8739d82596ce4e8592bde9475504c430>`   Controls whether to transpose the last two dimensions of ``src`` .       bool   True, False (default)   Optional   
:ref:`transpose_b <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684aa842de682cfdaec3291bbdffa551f4d7>`   Controls whether to transpose the last two dimensions of ``weights`` .   bool   True, False (default)   Optional   
========================================================================================================================  =======================================================================  =====  ======================  =========

The above transpose attribute will not in effect when rank of an input tensor is less than 2. For example, in library implementation 1D tensor is unsqueezed firstly before compilation. The rule is applied independently.

* For :math:`\src` tensor, the rule is defined like: ``[d] -> [1, d]``.

* For :math:`\weights` tensor, the rule is defined like: ``[d] -> [d, 1]``.

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ============  =========  
Index   Argu          
======  ============  =========  
0       ``src``       Required   
1       ``weights``   Required   
2       ``bias``      Optional   
======  ============  =========

Outputs
-------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``dst``   Required   
======  ========  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

MatMul operation supports the following data type combinations.

=====  ========  =====  =====  
Src    Weights   
=====  ========  =====  =====  
f32    f32       f32    f32    
bf16   bf16      bf16   bf16   
f16    f16       f16    f16    
=====  ========  =====  =====

