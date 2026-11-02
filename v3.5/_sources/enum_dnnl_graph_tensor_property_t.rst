.. index:: pair: enum; dnnl_graph_tensor_property_t
.. _doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f:

enum dnnl_graph_tensor_property_t
=================================

Overview
~~~~~~~~

Logical tensor property. :ref:`More...<details-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>

	enum dnnl_graph_tensor_property_t
	{
	    :ref:`dnnl_graph_tensor_property_undef<doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19faa02e5493848853a9bf77982d2fa56ab7>`    = 0,
	    :ref:`dnnl_graph_tensor_property_variable<doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19fa441941f9979f609504938d4f8b3758c4>` = 1,
	    :ref:`dnnl_graph_tensor_property_constant<doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19fa7c885f9bf2aecf29d58cef98f8073715>` = 2,
	};

.. _details-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Logical tensor property.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_graph_tensor_property_undef
.. _doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19faa02e5493848853a9bf77982d2fa56ab7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_tensor_property_undef

Undefined tensor property.

.. index:: pair: enumvalue; dnnl_graph_tensor_property_variable
.. _doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19fa441941f9979f609504938d4f8b3758c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_tensor_property_variable

Variable means the tensor may be changed during computation or between different iterations.

.. index:: pair: enumvalue; dnnl_graph_tensor_property_constant
.. _doxid-group__dnnl__graph__api__logical__tensor_1ggadf98ec2238dd9001c6fe7870ebf1b19fa7c885f9bf2aecf29d58cef98f8073715:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_tensor_property_constant

Constant means the tensor will keep unchanged during computation and between different iterations.

It's useful for the library to apply optimizations for constant tensors or cache constant tensors inside the library. For example, constant weight tensors in inference scenarios.

