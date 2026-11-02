.. index:: pair: page; 
.. _doxid-inference_int8_matmul_cpp_short:

<Untitled>
==========

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` fused with ReLU in INT8 inference.

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` fused with ReLU in INT8 inference.

Concepts:

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points_mask() <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`

* :ref:`Operation fusion <doxid-dev_guide_attributes_post_ops>`

* Create primitive once, use multiple times
  
  * Run-time tensor shapes: :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`

* Weights pre-packing: use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`

