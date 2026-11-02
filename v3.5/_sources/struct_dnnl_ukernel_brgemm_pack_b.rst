.. index:: pair: struct; dnnl::ukernel::brgemm_pack_b
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b:

struct dnnl::ukernel::brgemm_pack_b
===================================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	struct brgemm_pack_b: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`brgemm_pack_b<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1af2ca6a2f31ee6e446c20ac0547aece96>`();
	
		:ref:`brgemm_pack_b<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1ae1c47cd21c0cef9877b459c585767417>`(
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` in_ld,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` out_ld,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` in_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` out_dt,
			bool allow_empty = false
			);

		// methods
	
		bool :ref:`need_pack<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1a386c2180bc980e767ed2dc58ac4c1a34>`() const;
		void :ref:`generate<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1ad8c8dd31060c34a82a63e37f58cf6a57>`();
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1a7772e03e046d345ffc7e186ac8fccc41>`(const void* in, void* out) const;
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

.. _details-structdnnl_1_1ukernel_1_1brgemm__pack___b:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Construction
------------

.. index:: pair: function; brgemm_pack_b
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1af2ca6a2f31ee6e446c20ac0547aece96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm_pack_b()

Default constructor. Produces an empty object.

.. index:: pair: function; brgemm_pack_b
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1ae1c47cd21c0cef9877b459c585767417:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm_pack_b(
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` in_ld,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` out_ld,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` in_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` out_dt,
		bool allow_empty = false
		)

Constructs a BRGeMM ukernel packing tensor B object.



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

.. index:: pair: function; need_pack
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1a386c2180bc980e767ed2dc58ac4c1a34:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool need_pack() const

Returns the flag if packing is expected by BRGeMM ukernel kernel.

.. index:: pair: function; generate
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1ad8c8dd31060c34a82a63e37f58cf6a57:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void generate()

Generates an executable part of BRGeMM ukernel packing B object.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b_1a7772e03e046d345ffc7e186ac8fccc41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(const void* in, void* out) const

Executes a BRGeMM ukernel packing tensor B object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- in

		- Pointer to an input buffer.

	*
		- out

		- Pointer to an output buffer.

