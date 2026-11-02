.. index:: pair: page; SDPA with Compressed Key and Value
.. _doxid-dev_guide_graph_sdpa_compressed_kv:

SDPA with Compressed Key and Value
==================================

Overview
~~~~~~~~

int4 and int8 compressions for Key and Value are exploited in fused Scaled Dot-Product Attention (SDPA)[1] to reduce the memory footprint of generative inference of LLM, especially when KV cache mechanism is adopted. Specifically, Key and Value tensors are stored using lower precision data types like int4 and int8 to reduce memory usage, and are subsequently de-quantized to wider floating point data types such as f16 and bf16 for computation.

Note that grouped quantization is required to improve the model accuracy, especially for int4 data types. In this case, group size is needed as an attribute for quantization, which indicates the number of elements that share the same scaling factor and zero-points in each quantization group.

The notations used in this topic are:

* N: The mini-batch size.

* H: The head number.

* S: The sequence length.

* D: The size of each head.

* G: The group size.

SDPA Pattern
~~~~~~~~~~~~

The SDPA pattern with compressed Key and Value is defined as a directional acyclic graph (DAG) using oneDNN Graph API. oneDNN extends :ref:`SDPA pattern <doxid-dev_guide_graph_sdpa>` to support the following three kinds of compressed SDPA patterns:

#. SDPA with compressed Key and Value.

#. SDPA with floating-point Key and compressed Value.

#. SDPA with compressed Key and floating-point Value.

The floating-point data types include f32, f16 and bf16, and the compressed data type refers to low-precision integral data types, including int4 (u4/s4) and int8 (u8/s8) data types.

In oneDNN Graph API, we support quantization through a pattern with quantization operations such as :ref:`DynamicDequantize <doxid-dev_guide_op_dynamicdequantize>` and :ref:`DynamicQuantize <doxid-dev_guide_op_dynamicquantize>`. The supported pattern is as follows. The blue nodes are required while the brown nodes are optional.

.. image:: compressed_sdpa_pattern.png
	:alt: compressed SDPA pattern

Compared to a typical SDPA pattern, there are a few differences:

#. Two additional DynamicDequantize operations are applied to the input Key and Value to convert the integral values to floating-point values.

#. Apart from the Query, Key and Value inputs, the pattern requires additional quantization information such as scale and zero-points for the dequantization of Key and Value tensors. Currently, oneDNN only supports grouped quantization on one dimension; specifically, the shapes of scale and zero-points for Key and Value de-quantization should be (N, H, S, D/G).

#. Additionally, the ``group_shape`` attribute of the quantization operations must be specified as (1, 1, 1, G) for Key and Value dequantization.

Data Types
~~~~~~~~~~

oneDNN supports the following combinations of data types for Query, Key, Value, output, scale for Key, zero-points for Key, scale for Value and zero-points for Value:

======  =======  ========  ================  =======  ========  ================  =======  
Query   Key      Scale_K   Zp_K              Value    Scale_V   Zp_V              Output   
======  =======  ========  ================  =======  ========  ================  =======  
dt_fp   dt_int   dt_fp     u4,s4,u8,s8,s32   dt_int   dt_fp     u4,s4,u8,s8,s32   dt_fp    
dt_fp   dt_int   dt_fp     u4,s4,u8,s8,s32   dt_fp    N/A       N/A               dt_fp    
dt_fp   dt_fp    N/A       N/A               dt_int   dt_fp     u4,s4,u8,s8,s32   dt_fp    
======  =======  ========  ================  =======  ========  ================  =======

Notes:

* dt_fp can be: f16, bf16 or f32.

* dt_int can be: u8, s8, u4 or s4.

* zero-point inputs are optional.

You can specify the data type via the input and output data type fields of logical tensors for each operation. The definition of the data types and support status on different CPU and GPU platforms follow the general description in :ref:`Data Types <doxid-dev_guide_data_types>`.

Floating-point Math Mode
------------------------

You should set the floating-point math mode (:ref:`Primitive Attributes: floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>`) when using SDPA with compressed Key and Value. Generally, the math mode should align with the data type of the Query, which indicates the computation data type. Additionally, the second boolean flag, ``apply_to_int``, should be set to true. You can configure these attribute values using the ``set_fpmath_mode`` API (:ref:`dnnl::graph::graph::set_fpmath_mode <doxid-classdnnl_1_1graph_1_1graph_1a19c83436928ccc4ef523e2f149a390f7>`) on the graph object.

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* oneDNN primitive-based SDPA with compressed Key and Value is implemented as a reference implementation on both Intel Architecture Processors and Intel Graphics Products. The reference implementation requires memory to store the intermediate results of the dot products between Query and Key which takes :math:`O(S^2)` memory. It may lead to Out-of-Memory error when computing long sequence length inputs on platforms with limited memory.

* The compressed SDPA patterns functionally support all input shapes meeting the shape requirements of each operation in the graph.

* CPU
  
  * oneDNN does not provide optimized implementation on CPU currently. All executions will be implemented with the primitive-based reference computation.

* GPU
  
  * Optimized implementation is available for 4D Q/K/V tensors with the shape defined as (N, H, S, D) for Query and Value, (N, H, D, S) for Key, (N, H, D/G, S) for scales and zero-points of Key (if available) and (N, H, S, D/G) for scales and zero-points of Value (if available).
  
  * Optimized implementation is available for compressed SDPA with ``f16`` computation data type on Intel Graphics Products with Intel(R) Xe Matrix Extensions (Intel(R) XMX) support.
  
  * If int4 zero-points are specified, optimized implementation will be only available when the group size equals 16.

References
~~~~~~~~~~

[1] Attention is all you need, `https://arxiv.org/abs/1706.03762v7 <https://arxiv.org/abs/1706.03762v7>`__

