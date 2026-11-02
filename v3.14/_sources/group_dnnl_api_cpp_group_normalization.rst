.. index:: pair: group; Group Normalization
.. _doxid-group__dnnl__api__cpp__group__normalization:

Group Normalization
===================

.. toctree::
	:hidden:

	struct_dnnl_group_normalization_backward.rst
	struct_dnnl_group_normalization_forward.rst

A primitive to perform group normalization.

Both forward and backward propagation primitives support in-place operation; that is, src and dst can refer to the same memory for forward propagation, and diff_dst and diff_src can refer to the same memory for backward propagation.

The group normalization primitives computations can be controlled by specifying different :ref:`dnnl::normalization_flags <doxid-group__dnnl__api__cpp__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` values. For example, group normalization forward propagation can be configured to either compute the mean and variance or take them as arguments. It can either perform scaling and shifting using gamma and beta parameters or not.



.. rubric:: See also:

:ref:`Group Normalization <doxid-dev_guide_group_normalization>` in developer guide


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::group_normalization_backward<doxid-structdnnl_1_1group__normalization__backward>`;
	struct :ref:`dnnl::group_normalization_forward<doxid-structdnnl_1_1group__normalization__forward>`;

