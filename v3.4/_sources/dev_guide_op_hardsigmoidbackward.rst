.. index:: pair: page; HardSigmoidBackward
.. _doxid-dev_guide_op_hardsigmoidbackward:

HardSigmoidBackward
===================

General
~~~~~~~

HardSigmoidBackward operation computes gradient for HardSigmoid. The formula is defined as follows:

.. math::

	diff\_src = \begin{cases} diff\_dst \cdot \alpha & \text{if}\ 0 < \alpha src + \beta < 1 \\ 0 & \text{otherwise}\ \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

==================================================================================================================  ===============================  ===========  =====================  =====================  
Attribute Name                                                                                                      Description                      Value Type   Supported Values       Required or Optional   
==================================================================================================================  ===============================  ===========  =====================  =====================  
:ref:`alpha <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9>`   :math:`\alpha` in the formula.   f32          Arbitrary f32 value.   Required               
:ref:`beta <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`    :math:`\beta` in the formula.    f32          Arbitrary f32 value.   Required               
==================================================================================================================  ===============================  ===========  =====================  =====================

Execution arguments
~~~~~~~~~~~~~~~~~~~

The inputs and outputs must be provided according to the index order shown below when constructing an operation.

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

HardSigmoidBackward operation supports the following data type combinations.

=====  =========  =========  
Src    Diff_dst   Diff_src   
=====  =========  =========  
f32    f32        f32        
f16    f16        f16        
bf16   bf16       bf16       
=====  =========  =========

