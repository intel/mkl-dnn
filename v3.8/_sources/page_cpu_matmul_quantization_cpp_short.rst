.. index:: pair: page; 
.. _doxid-cpu_matmul_quantization_cpp_short:

<Untitled>
==========

C++ API example demonstrating how one can perform reduced precision matrix-matrix multiplication using :ref:`MatMul <doxid-dev_guide_matmul>` and the accuracy of the result compared to the floating point computations.

C++ API example demonstrating how one can perform reduced precision matrix-matrix multiplication using :ref:`MatMul <doxid-dev_guide_matmul>` and the accuracy of the result compared to the floating point computations.

Concepts:

* Static and dynamic quantization

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points_mask() <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`

