.. index:: pair: page; Reorder
.. _doxid-dev_guide_reorder:

Reorder
=======

:ref:`API Reference <doxid-group__dnnl__api__reorder>`

General
~~~~~~~

The reorder primitive copies data between different memory formats but doesn't change the tensor from mathematical perspective (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(\overline{x}) = \src(\overline{x})

As described in :ref:`Basic Concepts <doxid-dev_guide_basic_concepts>` in order to achieve the best performance some primitives (such as convolution) require special memory format which is typically referred to as an optimized memory format. The optimized memory format may match or may not match memory format that data is currently kept in. In this case a user can use reorder primitive to copy (reorder) the data between the memory formats.

Using the attributes and post-ops users can also use reorder primitive to quantize the data (and if necessary change the memory format simultaneously).

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

=======================  =====================================  
Primitive input/output   Execution argument index               
=======================  =====================================  
:math:`\src`             DNNL_ARG_FROM                          
:math:`\dst`             DNNL_ARG_TO                            
:math:`src scale`        DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM   
:math:`dst scale`        DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO     
=======================  =====================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The reorder primitive requires the source and destination tensors to have the same shape. Implicit broadcasting is not supported.

#. While in most of the cases the reorder should be able to handle arbitrary source and destination memory formats and data types, it might happen than some combinations are not implemented. For instance:
   
   * Reorder implementations between weights in non-plain memory formats might be limited (but if encountered in real practice should be treated as a bug and reported to oneDNN team);
   
   * Weights in one Winograd format cannot be reordered to the weights of the other Winograd format;
   
   * Quantized weights for convolution with :ref:`dnnl_s8 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a9638cfbcb7d50834a608ffae644d76b4>` source data type cannot be dequantized back to the :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>` data type;
   
   * Only reference support is available for reorders to or from f8_e4m3.
   
   * Optimized implementation of reorders to or from f8_e5m2 is available on Intel(R) Data Center GPU Max Series Only.

#. To alleviate the problem a user may rely on fact that the reorder from original plain memory format and user's data type to the optimized format with chosen data type should be always implemented.

Data Types Support
------------------

The reorder primitive supports arbitrary data types for the source and destination according to the :ref:`Data Types <doxid-dev_guide_data_types>` page.

When converting the data from one data type to a smaller one saturation is used. For instance:

.. ref-code-block:: cpp

	reorder(src={1024, data_type=f32}, dst={, data_type=s8})
	// dst == {127}
	
	reorder(src={-124, data_type=f32}, dst={, data_type=u8})
	// dst == {0}

Data Representation
-------------------

The reorder primitive works with arbitrary data tensors. There is no special meaning associated with any logical dimensions.

Post-Ops and Attributes
-----------------------

The reorder primitive support the following attributes and post-ops:

============================================================================================  =============================================================  
Attributes / Post-ops                                                                         Meaning                                                        
============================================================================================  =============================================================  
:ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`        Scales the corresponding tensor by the given scale factor(s)   
:ref:`Zero points <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`   Sets zero point(s) for the corresponding tensors               
:ref:`Sum post-op <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`         Instead of copy the data accumulate it to the previous data    
============================================================================================  =============================================================

For instance, the following pseudo-code

.. ref-code-block:: cpp

	reorder(
	        src = {dims={N, C, H, W}, data_type=dt_src, memory_format=fmt_src},
	        dst = {dims={N, C, H, W}, data_type=dt_dst, memory_format=fmt_dst},
	        attr ={
	            scales={ src={mask=0} },
	            zero_points= { src={mask=0}, dst={mask=0} },
	            post-ops = { sum={scale=beta} },
	        })

would lead to the following operation:

.. math::

	\dst(\overline{x}) = scale_{src} \cdot (\src(\overline{x}) - shift_{src}) + \beta \cdot \dst(\overline{x}) + shift_{dst}

.. note:: 

   * The intermediate operations are being done using single precision floating point data type.
   
   * :math:`scale_{src}`, :math:`shift_{src}`, :math:`scale_{dst}`, and :math:`shift_{dst}` must be passed during execution runtime as a separate memory arguments. Using :math:`scale_{src}` argument will lead to multiplication of tensor values by a scale value. Using :math:`scale_{dst}` argument will lead to division of tensor values by a scale value.
   
   


Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. CPU
   
   * Reorders between bf16, f16 and s32 data types are not supported.

#. GPU
   
   * Only tensors of 6 or fewer dimensions are supported.
   
   * Runtime dimensions are not supported.

Performance Tips
~~~~~~~~~~~~~~~~

N/A

Example
~~~~~~~

:ref:`Reorder Primitive Example <doxid-reorder_example_cpp>`

This C++ API demonstrates how to create and execute a :ref:`Reorder <doxid-dev_guide_reorder>` primitive.

Key optimizations included in this example:

* Primitive attributes for output scaling.

