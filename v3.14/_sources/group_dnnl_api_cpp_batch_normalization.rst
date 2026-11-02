.. index:: pair: group; Batch Normalization
.. _doxid-group__dnnl__api__cpp__batch__normalization:

Batch Normalization
===================

.. toctree::
	:hidden:

	struct_dnnl_batch_normalization_backward.rst
	struct_dnnl_batch_normalization_forward.rst

A primitive to perform batch normalization.

Both forward and backward propagation primitives support in-place operation; that is, src and dst can refer to the same memory for forward propagation, and diff_dst and diff_src can refer to the same memory for backward propagation.

The batch normalization primitives computations can be controlled by specifying different :ref:`dnnl::normalization_flags <doxid-group__dnnl__api__cpp__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` values. For example, batch normalization forward propagation can be configured to either compute the mean and variance or take them as arguments. It can either perform scaling and shifting using gamma and beta parameters or not. Optionally, it can also perform a fused ReLU, which in case of training would also require a workspace.



.. rubric:: See also:

:ref:`Batch Normalization <doxid-dev_guide_batch_normalization>` in developer guide


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::batch_normalization_backward<doxid-structdnnl_1_1batch__normalization__backward>`;
	struct :ref:`dnnl::batch_normalization_forward<doxid-structdnnl_1_1batch__normalization__forward>`;

