.. index:: pair: class; dnnl::graph::tensor
.. _doxid-classdnnl_1_1graph_1_1tensor:

class dnnl::graph::tensor
=========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A tensor object. :ref:`More...<details-classdnnl_1_1graph_1_1tensor>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class tensor: public tensor_handle
	{
	public:
		// construction
	
		:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor_1a73b101ca2e1c7766b170f611041acbb0>`();
		:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor_1a14bd0ec6d331f29750cb68257983a790>`(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& lt, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, void* handle);

		// methods
	
		void* :ref:`get_data_handle<doxid-classdnnl_1_1graph_1_1tensor_1aee1d338d718fb36fa6acc87d0373663e>`() const;
		void :ref:`set_data_handle<doxid-classdnnl_1_1graph_1_1tensor_1a5162eaf95542798d834bd8899138166e>`(void* handle);
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`get_engine<doxid-classdnnl_1_1graph_1_1tensor_1a588455c7eed8230723445b9dba997fb5>`() const;
	};
.. _details-classdnnl_1_1graph_1_1tensor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A tensor object.

Construction
------------

.. index:: pair: function; tensor
.. _doxid-classdnnl_1_1graph_1_1tensor_1a73b101ca2e1c7766b170f611041acbb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	tensor()

Default constructor. Constructs an empty object.

.. index:: pair: function; tensor
.. _doxid-classdnnl_1_1graph_1_1tensor_1a14bd0ec6d331f29750cb68257983a790:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	tensor(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& lt, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, void* handle)

Constructs a tensor object according to a given logical tensor, an engine, and a memory handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- lt

		- The given logical tensor

	*
		- aengine

		- Engine to store the data on.

	*
		- handle

		- Handle of memory buffer to use as an underlying storage.

Methods
-------

.. index:: pair: function; get_data_handle
.. _doxid-classdnnl_1_1graph_1_1tensor_1aee1d338d718fb36fa6acc87d0373663e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void* get_data_handle() const

Returns the underlying memory buffer.

On the CPU engine, or when using USM, this is a pointer to the allocated memory.

.. index:: pair: function; set_data_handle
.. _doxid-classdnnl_1_1graph_1_1tensor_1a5162eaf95542798d834bd8899138166e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_data_handle(void* handle)

Sets the underlying memory handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- handle

		- Memory handle.

.. index:: pair: function; get_engine
.. _doxid-classdnnl_1_1graph_1_1tensor_1a588455c7eed8230723445b9dba997fb5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` get_engine() const

Returns the associated engine.



.. rubric:: Returns:

An engine object

