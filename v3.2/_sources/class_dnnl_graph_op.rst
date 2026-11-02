.. index:: pair: class; dnnl::graph::op
.. _doxid-classdnnl_1_1graph_1_1op:

class dnnl::graph::op
=====================

.. toctree::
	:hidden:

	enum_dnnl_graph_op_attr.rst
	enum_dnnl_graph_op_kind.rst

Overview
~~~~~~~~

An op object. :ref:`More...<details-classdnnl_1_1graph_1_1op>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class op: public op_handle
	{
	public:
		// enums
	
		enum :ref:`attr<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684>`;
		enum :ref:`kind<doxid-classdnnl_1_1graph_1_1op_1ad54d9bef2fbf3c2a5f4190fa2497568c>`;

		// construction
	
		:ref:`op<doxid-classdnnl_1_1graph_1_1op_1ac16a54f4179d73110f5f7942df37a12f>`(size_t id, :ref:`kind<doxid-classdnnl_1_1graph_1_1op_1ad54d9bef2fbf3c2a5f4190fa2497568c>` akind, const std::string& verbose_name = "");
	
		:ref:`op<doxid-classdnnl_1_1graph_1_1op_1a4a2e86def5fc0299c6b67fc02ff8c9eb>`(
			size_t id,
			:ref:`kind<doxid-classdnnl_1_1graph_1_1op_1ad54d9bef2fbf3c2a5f4190fa2497568c>` akind,
			const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& inputs,
			const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& outputs,
			const std::string& verbose_name = ""
			);

		// methods
	
		void :ref:`add_input<doxid-classdnnl_1_1graph_1_1op_1ab0e18823d70c33b0882eab0557c83c09>`(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& t);
		void :ref:`add_inputs<doxid-classdnnl_1_1graph_1_1op_1a7159cd7dfdbae7d917f48ff7a51beff8>`(const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& ts);
		void :ref:`add_output<doxid-classdnnl_1_1graph_1_1op_1a8492588050f70b9bb4a27fef16c0027a>`(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& t);
		void :ref:`add_outputs<doxid-classdnnl_1_1graph_1_1op_1a9afad8a6451e194fa6de1ffca1a8cf98>`(const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& ts);
	
		template <typename Type, req<std::is_same<Type, int64_t>::value> = true>
		op& :ref:`set_attr<doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`(
			:ref:`attr<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684>` name,
			const Type& value
			);
	};
.. _details-classdnnl_1_1graph_1_1op:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An op object.

Construction
------------

.. index:: pair: function; op
.. _doxid-classdnnl_1_1graph_1_1op_1ac16a54f4179d73110f5f7942df37a12f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	op(size_t id, :ref:`kind<doxid-classdnnl_1_1graph_1_1op_1ad54d9bef2fbf3c2a5f4190fa2497568c>` akind, const std::string& verbose_name = "")

Constructs an op object with an unique ID, an operation kind, and a name string.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- id

		- The unique ID of the op.

	*
		- akind

		- The op kind specifies which computation is represented by the op, such as Convolution or ReLU.

	*
		- verbose_name

		- The string added as the op name.

.. index:: pair: function; op
.. _doxid-classdnnl_1_1graph_1_1op_1a4a2e86def5fc0299c6b67fc02ff8c9eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	op(
		size_t id,
		:ref:`kind<doxid-classdnnl_1_1graph_1_1op_1ad54d9bef2fbf3c2a5f4190fa2497568c>` akind,
		const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& inputs,
		const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& outputs,
		const std::string& verbose_name = ""
		)

Constructs an op object with an unique ID, an operation kind, and input/output logical tensors.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- id

		- The unique ID of this op.

	*
		- akind

		- The op kind specifies which computation is represented by this op, such as Convolution or ReLU.

	*
		- inputs

		- Input logical tensor to be bound to this op.

	*
		- outputs

		- Output logical tensor to be bound to this op.

	*
		- verbose_name

		- The string added as the op name.

Methods
-------

.. index:: pair: function; add_input
.. _doxid-classdnnl_1_1graph_1_1op_1ab0e18823d70c33b0882eab0557c83c09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void add_input(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& t)

Adds an input logical tensor to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t

		- Input logical tensor.

.. index:: pair: function; add_inputs
.. _doxid-classdnnl_1_1graph_1_1op_1a7159cd7dfdbae7d917f48ff7a51beff8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void add_inputs(const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& ts)

Adds a vector of input logical tensors to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- ts

		- The list of input logical tensors.

.. index:: pair: function; add_output
.. _doxid-classdnnl_1_1graph_1_1op_1a8492588050f70b9bb4a27fef16c0027a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void add_output(const :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`& t)

Adds an output logical tensor to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t

		- Output logical tensor.

.. index:: pair: function; add_outputs
.. _doxid-classdnnl_1_1graph_1_1op_1a9afad8a6451e194fa6de1ffca1a8cf98:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void add_outputs(const std::vector<:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& ts)

Adds a vector of output logical tensors to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- ts

		- The list of output logical tensors.

.. index:: pair: function; set_attr
.. _doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename Type, req<std::is_same<Type, int64_t>::value> = true>
	op& set_attr(
		:ref:`attr<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684>` name,
		const Type& value
		)

Sets the attribute according to the name and type (int64_t).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- Type

		- Attribute's type.

	*
		- name

		- Attribute's name.

	*
		- value

		- The attribute's value.



.. rubric:: Returns:

The Op self.

