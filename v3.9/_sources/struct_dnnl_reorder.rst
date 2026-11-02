.. index:: pair: struct; dnnl::reorder
.. _doxid-structdnnl_1_1reorder:

struct dnnl::reorder
====================

.. toctree::
	:hidden:

	struct_dnnl_reorder_primitive_desc.rst

Overview
~~~~~~~~

Reorder primitive. :ref:`More...<details-structdnnl_1_1reorder>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct reorder: public :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`
	{
		// structs
	
		struct :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`;

		// construction
	
		:ref:`reorder<doxid-structdnnl_1_1reorder_1ae964e269b4b2fc48b3f810ba5998d6a2>`();
		:ref:`reorder<doxid-structdnnl_1_1reorder_1a8052aeb588fa278d3e56ef85490b6690>`(const :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`& pd);
		:ref:`reorder<doxid-structdnnl_1_1reorder_1a6a1fdb6e178bfd3453c2478be306047f>`(const :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`& pd, const std::vector<uint8_t>& cache_blob);
	
		:ref:`reorder<doxid-structdnnl_1_1reorder_1af04b380a824816defecb488a724bfb37>`(
			const :ref:`memory<doxid-structdnnl_1_1memory>`& src,
			const :ref:`memory<doxid-structdnnl_1_1memory>`& dst,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`()
			);

		// methods
	
		void :ref:`execute<doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream, :ref:`memory<doxid-structdnnl_1_1memory>`& src, :ref:`memory<doxid-structdnnl_1_1memory>`& dst) const;
		void :ref:`execute<doxid-structdnnl_1_1reorder_1a2c112f2449a18a87310dee2ecd8c64eb>`();
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

.. _details-structdnnl_1_1reorder:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Reorder primitive.

Construction
------------

.. index:: pair: function; reorder
.. _doxid-structdnnl_1_1reorder_1ae964e269b4b2fc48b3f810ba5998d6a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder()

Default constructor. Produces an empty object.

.. index:: pair: function; reorder
.. _doxid-structdnnl_1_1reorder_1a8052aeb588fa278d3e56ef85490b6690:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder(const :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`& pd)

Constructs a reorder primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor for reorder primitive.

.. index:: pair: function; reorder
.. _doxid-structdnnl_1_1reorder_1a6a1fdb6e178bfd3453c2478be306047f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder(const :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`& pd, const std::vector<uint8_t>& cache_blob)

Constructs a reorder primitive from a cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor for reorder primitive.

	*
		- cache_blob

		- Cache blob.

.. index:: pair: function; reorder
.. _doxid-structdnnl_1_1reorder_1af04b380a824816defecb488a724bfb37:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder(
		const :ref:`memory<doxid-structdnnl_1_1memory>`& src,
		const :ref:`memory<doxid-structdnnl_1_1memory>`& dst,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`()
		)

Constructs a reorder primitive that would reorder data between memory objects having the same memory descriptors as memory objects ``src`` and ``dst``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- src

		- Source memory object.

	*
		- dst

		- Destination memory object.

	*
		- attr

		- Primitive attributes to use (optional).

Methods
-------

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream, :ref:`memory<doxid-structdnnl_1_1memory>`& src, :ref:`memory<doxid-structdnnl_1_1memory>`& dst) const

Executes the reorder primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Stream object. The stream must belong to the same engine as the primitive.

	*
		- src

		- Source memory object.

	*
		- dst

		- Destination memory object.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1reorder_1a2c112f2449a18a87310dee2ecd8c64eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute()

Executes computations specified by the primitive in a specified stream.

Arguments are passed via an arguments map containing <index, memory object> pairs. The index must be one of the ``DNNL_ARG_*`` values such as ``DNNL_ARG_SRC``, and the memory must have a memory descriptor matching the one returned by :ref:`primitive_desc::query_md <doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16>` (:ref:`query::exec_arg_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ad531896cf1d66c4832790f428623f164>`, index) unless using dynamic shapes (see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Stream object. The stream must belong to the same engine as the primitive.

	*
		- args

		- Arguments map.

