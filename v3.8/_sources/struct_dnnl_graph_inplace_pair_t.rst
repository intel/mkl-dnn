.. index:: pair: struct; dnnl_graph_inplace_pair_t
.. _doxid-structdnnl__graph__inplace__pair__t:

struct dnnl_graph_inplace_pair_t
================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

In-place pair definition. :ref:`More...<details-structdnnl__graph__inplace__pair__t>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>
	
	struct dnnl_graph_inplace_pair_t
	{
		// fields
	
		size_t :ref:`input_id<doxid-structdnnl__graph__inplace__pair__t_1ab097aed6af85d23116ae94680f5a08e5>`;
		size_t :ref:`output_id<doxid-structdnnl__graph__inplace__pair__t_1a75cb30d6e30955aa1a12b025611cc083>`;
	};
.. _details-structdnnl__graph__inplace__pair__t:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

In-place pair definition.

It can queried from a compiled partition indicating that an input and an output of the partition can share the same memory buffer for computation. In-place computation helps to reduce the memory footprint and improves cache locality. But since the library may not have a global view of user's application, it's possible that the tensor with ``input_id`` is used at other places in user's computation graph. In this case, the user should take the in-place pair as a hint and pass a different memory buffer for output tensor to avoid overwriting the input memory buffer which will probably cause unexpected incorrect results.

Fields
------

.. index:: pair: variable; input_id
.. _doxid-structdnnl__graph__inplace__pair__t_1ab097aed6af85d23116ae94680f5a08e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t input_id

The id of input tensor.

.. index:: pair: variable; output_id
.. _doxid-structdnnl__graph__inplace__pair__t_1a75cb30d6e30955aa1a12b025611cc083:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t output_id

The id of output tensor.

