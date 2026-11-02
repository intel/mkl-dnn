.. index:: pair: struct; dnnl::ukernel::brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm:

struct dnnl::ukernel::brgemm
============================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	struct brgemm: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm_1a7a5d8ea3fb6e01894027558e2ebb868a>`();
	
		:ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm_1a582bd2d468607a5aa241835e0473de32>`(
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` batch_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` lda,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldb,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldc,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` c_dt,
			float alpha,
			float beta,
			bool allow_empty = false
			);
	
		:ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm_1a60e9db910e91e17e9aa3d5e163492136>`(
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` batch_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` lda,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldb,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldc,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldd,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` c_dt,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` d_dt,
			float alpha,
			float beta,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr,
			bool allow_empty = false
			);

		// methods
	
		size_t :ref:`get_scratchpad_size<doxid-structdnnl_1_1ukernel_1_1brgemm_1ada0b6984b8b9253cba9756c680c07d16>`() const;
		void :ref:`set_hw_context<doxid-structdnnl_1_1ukernel_1_1brgemm_1ac273853c939803d7c0f20fe1b8c41f48>`() const;
		void :ref:`generate<doxid-structdnnl_1_1ukernel_1_1brgemm_1ae7c33dba7d829ced8d6b2de161159f69>`();
	
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b>`(
			const void* A,
			const void* B,
			const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
			void* C,
			void* scratchpad
			) const;
	
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1brgemm_1ab79b84837da33d89ed7380bfd1e3deff>`(
			const void* A,
			const void* B,
			const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
			void* C,
			void* D,
			void* scratchpad,
			const void* binary_po = nullptr
			) const;
	
		static void :ref:`release_hw_context<doxid-structdnnl_1_1ukernel_1_1brgemm_1a4cdc1e8b77991a2da8a69ae5f4ce267a>`();
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

.. _details-structdnnl_1_1ukernel_1_1brgemm:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Construction
------------

.. index:: pair: function; brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a7a5d8ea3fb6e01894027558e2ebb868a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm()

Default constructor. Produces an empty object.

.. index:: pair: function; brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a582bd2d468607a5aa241835e0473de32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm(
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` batch_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` lda,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldb,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldc,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` c_dt,
		float alpha,
		float beta,
		bool allow_empty = false
		)

Constructs a BRGeMM ukernel object.

Operates by the following formula: ``C = alpha * [A x B] + beta * C``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- M

		- Dimension M of tensor A.

	*
		- N

		- Dimension N of tensor B.

	*
		- K

		- Dimension K of tensors A and B.

	*
		- batch_size

		- Number of batches to process.

	*
		- lda

		- Leading dimension of tensor A.

	*
		- ldb

		- Leading dimension of tensor B.

	*
		- ldc

		- Leading dimension of tensor C.

	*
		- a_dt

		- Data type of tensor A.

	*
		- b_dt

		- Data type of tensor B.

	*
		- c_dt

		- Data type of tensor C.

	*
		- alpha

		- Scale for an accumulation output.

	*
		- beta

		- Scale for a tensor C to append on an accumulated output.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a60e9db910e91e17e9aa3d5e163492136:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm(
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` batch_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` lda,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldb,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldc,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldd,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` c_dt,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` d_dt,
		float alpha,
		float beta,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr,
		bool allow_empty = false
		)

Constructs a BRGeMM ukernel object.

Operates by the following formula: ``C = alpha * [A x B] + beta * C``; ``D = post-operations(C)``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- M

		- Dimension M of tensor A.

	*
		- N

		- Dimension N of tensor B.

	*
		- K

		- Dimension K of tensors A and B.

	*
		- batch_size

		- Number of batches to process.

	*
		- lda

		- Leading dimension of tensor A.

	*
		- ldb

		- Leading dimension of tensor B.

	*
		- ldc

		- Leading dimension of tensor C.

	*
		- ldd

		- Leading dimension of tensor D.

	*
		- a_dt

		- Data type of tensor A.

	*
		- b_dt

		- Data type of tensor B.

	*
		- c_dt

		- Data type of tensor C. Must be :ref:`data_type::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`.

	*
		- d_dt

		- Data type of tensor D.

	*
		- alpha

		- Scale for an accumulation output.

	*
		- beta

		- Scale for a tensor C to append on an accumulated output.

	*
		- attr

		- Primitive attributes to extend the kernel operations.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

Methods
-------

.. index:: pair: function; get_scratchpad_size
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1ada0b6984b8b9253cba9756c680c07d16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_scratchpad_size() const

Returns the size of a scratchpad memory needed for the BRGeMM ukernel object.

.. index:: pair: function; set_hw_context
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1ac273853c939803d7c0f20fe1b8c41f48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_hw_context() const

Initializes the hardware-specific context.

Affects the global state for all BRGeMM ukernel objects. If no initialization required, returns.

.. index:: pair: function; generate
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1ae7c33dba7d829ced8d6b2de161159f69:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void generate()

Generates an executable part of BRGeMM ukernel object.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(
		const void* A,
		const void* B,
		const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
		void* C,
		void* scratchpad
		) const

Executes a BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- A

		- Base pointer to a tensor A.

	*
		- B

		- Base pointer to a tensor B.

	*
		- A_B_offsets

		- Vector of pairs of tensors A and B offsets for each batch. The number of batches must coincide with the ``batch_size`` value passed at object construction stage.

	*
		- C

		- Pointer to a tensor C (accumulation buffer).

	*
		- scratchpad

		- Pointer to a scratchpad buffer.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1ab79b84837da33d89ed7380bfd1e3deff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(
		const void* A,
		const void* B,
		const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
		void* C,
		void* D,
		void* scratchpad,
		const void* binary_po = nullptr
		) const

Executes a BRGeMM ukernel object with post operations.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- A

		- Base pointer to a tensor A.

	*
		- B

		- Base pointer to a tensor B.

	*
		- A_B_offsets

		- Vector of pairs of tensors A and B offsets for each batch. The number of batches must coincide with the ``batch_size`` value passed at object construction stage.

	*
		- C

		- Pointer to a tensor C (accumulation buffer).

	*
		- D

		- Pointer to a tensor D (output buffer).

	*
		- scratchpad

		- Pointer to a scratchpad buffer.

	*
		- binary_po

		- Binary post-op memory buffer. Must be passed If binary post-op was specified at construction call.

.. index:: pair: function; release_hw_context
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a4cdc1e8b77991a2da8a69ae5f4ce267a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static void release_hw_context()

Releases the hardware-specific context.

Affects the global state for all BRGeMM ukernel objects. Must be used after all the execution calls to BRGeMM ukernel objects.

