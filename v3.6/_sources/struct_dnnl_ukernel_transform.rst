.. index:: pair: struct; dnnl::ukernel::transform
.. _doxid-structdnnl_1_1ukernel_1_1transform:

struct dnnl::ukernel::transform
===============================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	struct transform: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`transform<doxid-structdnnl_1_1ukernel_1_1transform_1a7c2caec8844bba799da0a2b19c1a6477>`();
	
		:ref:`transform<doxid-structdnnl_1_1ukernel_1_1transform_1a3d88aae628d7310ad00d53bafe5bf00b>`(
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
			:ref:`pack_type<doxid-namespacednnl_1_1ukernel_1a241c23d0afdf43a79d51ef701a9f7c54>` in_pack_type,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` in_ld,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` out_ld,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` in_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` out_dt,
			bool allow_empty = false
			);

		// methods
	
		void :ref:`generate<doxid-structdnnl_1_1ukernel_1_1transform_1a7a76cccde7eaf805d8339bfd253ff946>`();
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1transform_1ac92d44b1ed7ae0968e88809f96f8e6dc>`(const void* in, void* out) const;
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&);
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727>` (:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&&);
		void :ref:`reset<doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909>`(T t, bool weak = false);
		T :ref:`get<doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e>`(bool allow_empty = false) const;
		:ref:`operator T<doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3>` () const;
		:ref:`operator bool<doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5>` () const;
		bool :ref:`operator ==<doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab>` (const :ref:`handle<doxid-structdnnl_1_1handle>`& other) const;

.. _details-structdnnl_1_1ukernel_1_1transform:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Construction
------------

.. index:: pair: function; transform
.. _doxid-structdnnl_1_1ukernel_1_1transform_1a7c2caec8844bba799da0a2b19c1a6477:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	transform()

Default constructor. Produces an empty object.

.. index:: pair: function; transform
.. _doxid-structdnnl_1_1ukernel_1_1transform_1a3d88aae628d7310ad00d53bafe5bf00b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	transform(
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
		:ref:`pack_type<doxid-namespacednnl_1_1ukernel_1a241c23d0afdf43a79d51ef701a9f7c54>` in_pack_type,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` in_ld,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` out_ld,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` in_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` out_dt,
		bool allow_empty = false
		)

Constructs a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- K

		- Dimension K.

	*
		- N

		- Dimension N.

	*
		- in_pack_type

		- Input packing type. Must be one of ``:ref:`pack_type::no_trans <doxid-namespacednnl_1_1ukernel_1a241c23d0afdf43a79d51ef701a9f7c54a76659c0424cb9f2555bc14e7d947db13>```, or ``:ref:`pack_type::trans <doxid-namespacednnl_1_1ukernel_1a241c23d0afdf43a79d51ef701a9f7c54a4738019ef434f24099319565cd5185e5>```.

	*
		- in_ld

		- Input leading dimension.

	*
		- out_ld

		- Output leading dimension. Specifies a block by N dimension during data packing.

	*
		- in_dt

		- Input data type.

	*
		- out_dt

		- Output data type.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

Methods
-------

.. index:: pair: function; generate
.. _doxid-structdnnl_1_1ukernel_1_1transform_1a7a76cccde7eaf805d8339bfd253ff946:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void generate()

Generates an executable part of transform object.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1ukernel_1_1transform_1ac92d44b1ed7ae0968e88809f96f8e6dc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(const void* in, void* out) const

Executes a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- in

		- Pointer to an input buffer.

	*
		- out

		- Pointer to an output buffer.

