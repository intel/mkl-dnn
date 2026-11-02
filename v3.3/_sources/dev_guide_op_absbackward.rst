.. index:: pair: page; AbsBackward
.. _doxid-dev_guide_op_absbackward:

AbsBackward
===========

General
~~~~~~~

AbsBackward operation computes gradient for Abs operation.

.. math::

	dst = \begin{cases} diff\_dst & \text{if}\ src > 0 \\ -diff\_dst & \text{if}\ src < 0 \\ 0 & \text{if}\ src = 0 \\ \end{cases}

Operation attributes
~~~~~~~~~~~~~~~~~~~~

AbsBackward operation does not support any attribute.

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

AbsBackward operation supports the following data type combinations.

=====  =========  =========  
Src    Diff_dst   Diff_src   
=====  =========  =========  
f32    f32        f32        
f16    f16        f16        
bf16   bf16       bf16       
=====  =========  =========

