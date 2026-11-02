.. index:: pair: struct; dnnl::layer_normalization_backward
.. _doxid-structdnnl_1_1layer__normalization__backward:

struct dnnl::layer_normalization_backward
=========================================

.. toctree::
	:hidden:

	struct_dnnl_layer_normalization_backward_primitive_desc.rst

Overview
~~~~~~~~

Layer normalization backward propagation primitive. :ref:`More...<details-structdnnl_1_1layer__normalization__backward>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct layer_normalization_backward: public :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`
	{
		// structs
	
		struct :ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__backward_1_1primitive__desc>`;

		// construction
	
		:ref:`layer_normalization_backward<doxid-structdnnl_1_1layer__normalization__backward_1a942c575e4446cd508094bd2da98824ac>`();
		:ref:`layer_normalization_backward<doxid-structdnnl_1_1layer__normalization__backward_1a36cbbc365c0f37c4725d44f68bc180d3>`(const :ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__backward_1_1primitive__desc>`& pd);
	
		:ref:`layer_normalization_backward<doxid-structdnnl_1_1layer__normalization__backward_1a389cc2b4bd5778d72cdaf2031d75ecd1>`(
			const :ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__backward_1_1primitive__desc>`& pd,
			const std::vector<uint8_t>& cache_blob
			);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// enums
	
		enum :ref:`kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>`;

		// methods
	
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&);
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727>` (:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&&);
		void :ref:`reset<doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909>`(T t, bool weak = false);
		T :ref:`get<doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e>`(bool allow_empty = false) const;
		:ref:`operator T<doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3>` () const;
		:ref:`operator bool<doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5>` () const;
		bool :ref:`operator ==<doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab>` (const :ref:`handle<doxid-structdnnl_1_1handle>`& other) const;
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` :ref:`get_primitive_desc<doxid-group__dnnl__api__primitives__common_1ga6d84440f113be4c6697092a0b968aa36>`() const;
		:ref:`kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` :ref:`get_kind<doxid-group__dnnl__api__primitives__common_1gab715f58b261078428fc2a99c9ee527fc>`() const;
		std::vector<uint8_t> :ref:`get_cache_blob<doxid-group__dnnl__api__primitives__common_1ga8bf59b36c745ee8eaec9d0dd22e266e9>`() const;
		void :ref:`execute<doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream, const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args) const;
		:ref:`handle<doxid-structdnnl_1_1primitive_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1a022001b5b9c8940a1326a02b61fc4860>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1aa13f3ecf4db240717074814412c7e70c>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1a9c408c09fce1278f5cb0d1fa9818fc86>`();

.. _details-structdnnl_1_1layer__normalization__backward:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Layer normalization backward propagation primitive.

Construction
------------

.. index:: pair: function; layer_normalization_backward
.. _doxid-structdnnl_1_1layer__normalization__backward_1a942c575e4446cd508094bd2da98824ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	layer_normalization_backward()

Default constructor. Produces an empty object.

.. index:: pair: function; layer_normalization_backward
.. _doxid-structdnnl_1_1layer__normalization__backward_1a36cbbc365c0f37c4725d44f68bc180d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	layer_normalization_backward(const :ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__backward_1_1primitive__desc>`& pd)

Constructs a layer normalization backward propagation primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor for a layer normalization backward propagation primitive.

.. index:: pair: function; layer_normalization_backward
.. _doxid-structdnnl_1_1layer__normalization__backward_1a389cc2b4bd5778d72cdaf2031d75ecd1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	layer_normalization_backward(
		const :ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__backward_1_1primitive__desc>`& pd,
		const std::vector<uint8_t>& cache_blob
		)

Constructs a layer normalization backward propagation primitive from a cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor for a layer normalization backward propagation primitive.

	*
		- cache_blob

		- Cache blob.

