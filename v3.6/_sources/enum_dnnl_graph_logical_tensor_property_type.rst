.. index:: pair: enum; property_type
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82:

enum dnnl::graph::logical_tensor::property_type
===============================================

Overview
~~~~~~~~

Tensor property. :ref:`More...<details-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum property_type
	{
	    :ref:`undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`    = dnnl_graph_tensor_property_undef,
	    :ref:`variable<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82ae04aa5104d082e4a51d241391941ba26>` = dnnl_graph_tensor_property_variable,
	    :ref:`constant<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82a617ac08757d38a5a7ed91c224f0e90a0>` = dnnl_graph_tensor_property_constant,
	};

.. _details-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Tensor property.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined tensor property.

.. index:: pair: enumvalue; variable
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82ae04aa5104d082e4a51d241391941ba26:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	variable

Variable means the tensor may be changed during computation or between different iterations.

.. index:: pair: enumvalue; constant
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82a617ac08757d38a5a7ed91c224f0e90a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	constant

Constant means the tensor will keep unchanged during computation and between different iterations.

It's useful for the library to apply optimizations for constant tensors or cache constant tensors inside the library. For example, constant weight tensors in inference scenarios.

