.. index:: pair: group; Eltwise
.. _doxid-group__dnnl__api__cpp__eltwise:

Eltwise
=======

.. toctree::
	:hidden:

	struct_dnnl_eltwise_backward.rst
	struct_dnnl_eltwise_forward.rst

A primitive to perform elementwise operations such as the rectifier linear unit (ReLU).

Both forward and backward propagation primitives support in-place operation; that is, src and dst can refer to the same memory for forward propagation, and diff_dst and diff_src can refer to the same memory for backward propagation.

.. warning:: 

   Because the original source data is required for backward propagation, in-place forward propagation is not generally supported in the training mode. However, for algorithms supporting destination as input memory, dst can be used for the backward propagation, which makes it possible to get performance benefit even in the training mode.



.. rubric:: See also:

:ref:`Eltwise <doxid-dev_guide_eltwise>` in developer guide


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::eltwise_backward<doxid-structdnnl_1_1eltwise__backward>`;
	struct :ref:`dnnl::eltwise_forward<doxid-structdnnl_1_1eltwise__forward>`;

