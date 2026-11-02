.. index:: pair: page; Reorder
.. _doxid-dev_guide_op_reorder:

Reorder
=======

General
~~~~~~~

Reorder operation converts :math:`\src` tensor to :math:`\dst` tensor with different layout. It supports the conversion between:

* Two different opaque layouts.

* Two different strided layouts.

* One strided layout and another opaque layout.

Reorder operation does not support layout conversion cross different backends or different engines. Unlike :ref:`reorder primitive <doxid-dev_guide_reorder>`, Reorder operation cannot be used to cast the data type from :math:`\src` to :math:`\dst`. Please check the usage of :ref:`TypeCast <doxid-dev_guide_op_typecast>` and :ref:`Dequantize <doxid-dev_guide_op_dequantize>` operation.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

Reorder operation does not support any attribute.

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to below index order when constructing an operation.

Inputs
------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``src``   Required   
======  ========  =========

Outputs
-------

======  ========  =========  
Index   Argu      
======  ========  =========  
0       ``dst``   Required   
======  ========  =========

Supported data types
~~~~~~~~~~~~~~~~~~~~

Reorder operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

