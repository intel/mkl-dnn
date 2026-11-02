.. index:: pair: page; Low Precision
.. _doxid-dev_guide_graph_low_precision:

Low Precision
=============

oneDNN Graph provides low precision support with int8 (signed/unsigned 8-bit integer), bf16 and f16 data types. oneDNN Graph API expects the computation graph is converted to low precision representation, the data's precision and quantization parameters are specified explicitly. oneDNN Graph API implementation will strictly respect the numeric precision of the computation.

:target:`doxid-dev_guide_graph_low_precision_1dev_guide_graph_int8_quantization_model`

INT8
~~~~

oneDNN Graph API provides below two operations to support quantized model with static quantization:

* :ref:`Dequantize <doxid-dev_guide_op_dequantize>`

* :ref:`Quantize <doxid-dev_guide_op_quantize>`

Dequantize operation takes integer tensor with its associated scale and zero point and returns f32 tensor. Quantize operation takes f32 tensor, scale, zero point, and returns integer tensor. The scale and zero point are single dimension tensors, which could contain one value for the per-tensor quantization case or multiple values for the per-channel quantization case. The integer tensor could be represented in unsigned int8 or signed int8 data type. Zero point could be zero for symmetric quantization scheme, and a non-zero value for asymmetric quantization scheme.

Dequantize and Quantize operation should be inserted manually in the graph as part of quantization process before passing to oneDNN Graph. oneDNN Graph honors the data type passed via logical tensor and faithfully follows the numeric semantics. For example, if the graph has a Quantize operation followed by a Dequantize operation with exact same scale and zero point, oneDNN Graph implementation should not eliminate them since that implicitly changes the numeric precision.

oneDNN Graph partitioning API may return a partition containing Dequantize, Quantize, and Convolution operations in-between. It is not necessary to recognize the subgraph pattern explicitly and convert to fused operation. Depending on oneDNN Graph implementation capability, the partition may include more or fewer operations.

.. image:: int8_programming.jpg
	:alt: Figure 1: Overview of int8 programming model.

:target:`doxid-dev_guide_graph_low_precision_1dev_guide_graph_mixed_precision_model`

BF16/F16
~~~~~~~~

oneDNN Graph provides :ref:`TypeCast <doxid-dev_guide_op_typecast>` operation, which can convert a f32 tensor to bf16 or f16, and vice versa. It is used to support auto mixed precision mechanism in popular deep learning frameworks. All oneDNN Graph operations support bf16 and f16 data types.

A TypeCast operation performing down conversion should be inserted clearly to indicate the use of low numeric precision. oneDNN Graph implementation fully honors the API-specified numeric precision and only performs the computation using the API-specified or higher numeric precision.

.. image:: bf16_programming.jpg
	:alt: Figure 2: Overview of bf16 programming model.

