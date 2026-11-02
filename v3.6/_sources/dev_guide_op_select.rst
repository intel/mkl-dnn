.. index:: pair: page; Select
.. _doxid-dev_guide_op_select:

Select
======

General
~~~~~~~

The select operation returns a tensor filled with the elements from the second or the third input, depending on the condition (the first input) value.

.. code-block:: cpp

	dst[i] = cond[i] ? src_0[i] : src_1[i]

Broadcasting is supported.

Broadcasting rules
------------------

If the auto_broadcast attribute is not none, the select operation takes a two-step broadcast before performing the selection:

* Step 1 : Input tensors src_0 and src_1 are broadcasted to dst_shape according to the Numpy `broadcast rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__.

* Step 2 : Then, the cond tensor will be one-way broadcasted to the dst_shape of broadcasted src_0 and src_1. To be more specific, we align the two shapes to the right and compare them from right to left. Each dimension should be either equal or the dimension of cond should be 1.

* example :
  
  * cond={4, 5}, dst_shape={2, 3, 4, 5} => dst = {2, 3, 4, 5}
  
  * cond={3, 1, 5}, dst_shape={2, 3, 4, 5} => dst = {2, 3, 4, 5}
  
  * cond={3,5}, dst_shape={2, 3, 4, 5} => dst = invalid_shape

Operation attributes
~~~~~~~~~~~~~~~~~~~~

===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
Attribute Name                                                                                                               Description                                                  Value Type   Supported Values                 Required or Optional   
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================  
:ref:`auto_broadcast <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`   Specifies rules used for auto-broadcasting of src tensors.   string       ``none`` , ``numpy`` (default)   Optional               
===========================================================================================================================  ===========================================================  ===========  ===============================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to the following index order when constructing an operation.

Inputs
------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``cond``        Required               
1       ``src_0``       Required               
2       ``src_1``       Required               
======  ==============  =====================

.. note:: 

   All input shapes should match and no broadcasting is allowed if the ``auto_broadcast`` attribute is set to ``none``, or can be broadcasted according to the broadcasting rules mentioned above if ``auto_broadcast`` attribute is set to ``numpy``.
   
   


Outputs
-------

======  ==============  =====================  
Index   Argument Name   Required or Optional   
======  ==============  =====================  
0       ``dst``         Required               
======  ==============  =====================

Supported data types
~~~~~~~~~~~~~~~~~~~~

Select operation supports the following data type combinations.

========  ==============  =====  
Cond      Src_0 / Src_1   Dst    
========  ==============  =====  
boolean   f32             f32    
boolean   bf16            bf16   
boolean   f16             f16    
========  ==============  =====

