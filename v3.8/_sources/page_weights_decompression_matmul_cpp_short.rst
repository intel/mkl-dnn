.. index:: pair: page; 
.. _doxid-weights_decompression_matmul_cpp_short:

<Untitled>
==========

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with compressed weights.

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with compressed weights.

Concepts:

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales() <doxid-structdnnl_1_1primitive__attr_1a29e8f33119d42bf7d259eafc6e6548d6>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points() <doxid-structdnnl_1_1primitive__attr_1aa7a57b0ba198c418981d41c5289fed8e>`

* :ref:`Operation fusion <doxid-dev_guide_attributes_post_ops>`

* Create primitive once, use multiple times

* Weights pre-packing: use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`

