.. index:: pair: page; Concat
.. _doxid-dev_guide_op_concat:

Concat
======

General
~~~~~~~

Concat operation concatenates :math:`N` tensors over ``axis`` (here designated :math:`C`) and is defined as (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(\overline{ou}, c, \overline{in}) = \src_i(\overline{ou}, c', \overline{in}),

where :math:`c = C_1 + .. + C_{i-1} {}_{} + c'`.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=================================================================================================================  =======================================================  ====  ==========================================================  =========  
Attribute Name                                                                                                     De                                                       
=================================================================================================================  =======================================================  ====  ==========================================================  =========  
:ref:`axis <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`   Specifies dimension along which concatenation happens.   s64   A s64 value in the range of [-r, r-1] where r = rank(src)   Required   
=================================================================================================================  =======================================================  ====  ==========================================================  =========

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ==========  =========  
Index   Argu        
======  ==========  =========  
0       ``src_i``   Required   
======  ==========  =========

.. note:: 

   At least one input tensor is required. Data types and ranks of all input tensors should match. The dimensions of all input tensors should be the same except for the dimension specified by ``axis`` attribute.
   
   


Outputs
-------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``dst``   Required   
======  ========  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

Concat operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

