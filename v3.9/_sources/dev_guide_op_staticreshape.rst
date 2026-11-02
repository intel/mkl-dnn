.. index:: pair: page; StaticReshape
.. _doxid-dev_guide_op_staticreshape:

StaticReshape
=============

General
~~~~~~~

StaticReshape operation changes dimensions of :math:`\src` tensor according to the specified shape. The volume of :math:`\src` is equal to :math:`\dst`, where volume is the product of dimensions. :math:`\dst` may have a different memory layout from :math:`\src`. StaticReshape operation is not guaranteed to return a view or a copy of :math:`\src` when :math:`\dst` is in-placed with the :math:`\src`. StaticReshape can be used where if shape is stored in a constant node or available during graph building stage. Then shape can be passed via ``shape`` attribute.

Operation attributes
~~~~~~~~~~~~~~~~~~~~

=========================================================================================================================  ===================================================  ===========  =======================================  =====================  
Attribute Name                                                                                                             Description                                          Value Type   Supported Values                         Required or Optional   
=========================================================================================================================  ===================================================  ===========  =======================================  =====================  
:ref:`shape <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8c73a98a300905900337f535531dfca6>`          Denotes the shape of the ``dst`` tensor.             s64          A s64 list containing positive values.   Required               
:ref:`special_zero <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1ae9768d4bee269575f7464724cd97fa>`   Controls how zero values in shape are interpreted.   bool         ``true`` , ``false``                     Required               
=========================================================================================================================  ===================================================  ===========  =======================================  =====================

.. note:: 

   ``shape`` : dimension ``-1`` means that this dimension is calculated to keep the same overall elements count as the src tensor. That case that more than one ``-1`` in the shape is not supported.
   
   

.. note:: 

   ``special_zero`` : if false, ``0`` in the shape is interpreted as-is (for example a zero-dimension tensor); if true, then all ``0`` s in shape implies the copying of corresponding dimensions from src into dst.
   
   


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

StaticReshape operation supports the following data type combinations.

=====  =====  
Src    Dst    
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

