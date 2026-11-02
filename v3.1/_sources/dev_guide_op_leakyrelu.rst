.. index:: pair: page; LeakyReLU
.. _doxid-dev_guide_op_leakyrelu:

LeakyReLU
=========

General
~~~~~~~

LeakyReLU operation is a type of activation function based on ReLU. It has a small slope for negative values with which LeakyReLU can produce small, non-zero, and constant gradients with respect to the negative values. The slope is also called the coefficient of leakage.

Unlike :ref:`PReLU <doxid-dev_guide_op_prelu>`, the coefficient :math:`\alpha` is constant and defined before training.

LeakyReLU operation applies following formula on every element of :math:`\src` tensor (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	dst = \begin{cases} src & \text{if}\ src \ge 0 \\ \alpha src & \text{if}\ src < 0 \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  =====================================  ====  ========================================================  =========  
Attribute Name                                                                                                      Descr                                  
==================================================================================================================  =====================================  ====  ========================================================  =========  
:ref:`alpha <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9>`   Alpha is the coefficient of leakage.   f32   Arbitrary f32 value but usually a small positive value.   Required   
==================================================================================================================  =====================================  ====  ========================================================  =========

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

LeakyReLU operation supports the following data type combinations.

=====  =====  
Src    D      
=====  =====  
f32    f32    
bf16   bf16   
f16    f16    
=====  =====

