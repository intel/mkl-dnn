.. index:: pair: struct; dnnl::ukernel::brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm:

struct dnnl::ukernel::brgemm
============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

BRGeMM ukernel. :ref:`More...<details-structdnnl_1_1ukernel_1_1brgemm>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ukernel.hpp>
	
	struct brgemm: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm_1a7a5d8ea3fb6e01894027558e2ebb868a>`();
	
		:ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm_1a1e717b3e35313f8b3caf95a610b57c35>`(
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
			bool allow_empty = false
			);

		// methods
	
		void :ref:`set_add_C<doxid-structdnnl_1_1ukernel_1_1brgemm_1a4546a4aad9b1e3769ce1b5c51b7f746c>`(bool add_C);
	
		void :ref:`set_post_ops<doxid-structdnnl_1_1ukernel_1_1brgemm_1a99c44446d24cb50e8c1c20c11c4d7e4e>`(
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldd,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` d_dt,
			const :ref:`post_ops<doxid-structdnnl_1_1post__ops>`& po = :ref:`default_post_ops<doxid-structdnnl_1_1ukernel_1_1brgemm_1acea25e2313b6211468852d4c7d76f436>`()
			);
	
		void :ref:`set_A_scales<doxid-structdnnl_1_1ukernel_1_1brgemm_1a95d2fe8bc5e184eb74374c99f9111f72>`(int a_scale_mask);
		void :ref:`set_B_scales<doxid-structdnnl_1_1ukernel_1_1brgemm_1a0b17ef5afc621818865d6e41dba66ccc>`(int b_scale_mask);
		void :ref:`set_D_scales<doxid-structdnnl_1_1ukernel_1_1brgemm_1a857f6b4ed065e10519dfce02dc293e6f>`(int d_scale_mask);
		void :ref:`finalize<doxid-structdnnl_1_1ukernel_1_1brgemm_1a80543f101b056823aeed10238db70da0>`();
		:ref:`pack_type<doxid-group__dnnl__api__ukernel__utils_1ga241c23d0afdf43a79d51ef701a9f7c54>` :ref:`get_B_pack_type<doxid-structdnnl_1_1ukernel_1_1brgemm_1a41509e7835003df7976c59193b6a6f31>`() const;
		size_t :ref:`get_scratchpad_size<doxid-structdnnl_1_1ukernel_1_1brgemm_1ada0b6984b8b9253cba9756c680c07d16>`() const;
		bool :ref:`is_execute_postops_valid<doxid-structdnnl_1_1ukernel_1_1brgemm_1a2636a460ecb30c8c9535d8c18858c1ef>`() const;
		void :ref:`set_hw_context<doxid-structdnnl_1_1ukernel_1_1brgemm_1ac273853c939803d7c0f20fe1b8c41f48>`() const;
		void :ref:`generate<doxid-structdnnl_1_1ukernel_1_1brgemm_1ae7c33dba7d829ced8d6b2de161159f69>`();
	
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b>`(
			const void* A,
			const void* B,
			const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
			void* C,
			void* scratchpad
			) const;
	
		void :ref:`execute<doxid-structdnnl_1_1ukernel_1_1brgemm_1a53bcfebaf4e7ee3f099f6620e6e9ac50>`(
			const void* A,
			const void* B,
			const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
			const void* C,
			void* D,
			void* scratchpad,
			const :ref:`attr_params<doxid-structdnnl_1_1ukernel_1_1attr__params>`& params = :ref:`default_attr_params<doxid-structdnnl_1_1ukernel_1_1brgemm_1a529f502a78858748caaa10f8db3917bd>`()
			) const;
	
		static void :ref:`release_hw_context<doxid-structdnnl_1_1ukernel_1_1brgemm_1a4cdc1e8b77991a2da8a69ae5f4ce267a>`();
		static const :ref:`post_ops<doxid-structdnnl_1_1post__ops>`& :ref:`default_post_ops<doxid-structdnnl_1_1ukernel_1_1brgemm_1acea25e2313b6211468852d4c7d76f436>`();
		static const :ref:`attr_params<doxid-structdnnl_1_1ukernel_1_1attr__params>`& :ref:`default_attr_params<doxid-structdnnl_1_1ukernel_1_1brgemm_1a529f502a78858748caaa10f8db3917bd>`();
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

BRGeMM ukernel.

Construction
------------

.. index:: pair: function; brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a7a5d8ea3fb6e01894027558e2ebb868a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	brgemm()

Default constructor. Produces an empty object.

.. index:: pair: function; brgemm
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a1e717b3e35313f8b3caf95a610b57c35:

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
		bool allow_empty = false
		)

Constructs a BRGeMM ukernel object.

Operates by the following formula: ``C = [A x B]``.



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
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

Methods
-------

.. index:: pair: function; set_add_C
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a4546a4aad9b1e3769ce1b5c51b7f746c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_add_C(bool add_C)

Sets adding an intermediate result to the output tensor C instead of writing: ``C += [A x B]``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- add_C

		- Value to indicate addition. ``false`` to skip addition, and ``true`` to apply addition.

.. index:: pair: function; set_post_ops
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a99c44446d24cb50e8c1c20c11c4d7e4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_post_ops(
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` ldd,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` d_dt,
		const :ref:`post_ops<doxid-structdnnl_1_1post__ops>`& po = :ref:`default_post_ops<doxid-structdnnl_1_1ukernel_1_1brgemm_1acea25e2313b6211468852d4c7d76f436>`()
		)

Sets post-operations to a BRGeMM ukernel object: ``D = post-operations(C)``.

Post-operations applies if one of the following holds:

* Non-empty post-operations are specified.

* Output data type ``d_dt`` is different from accumulation data type ``c_dt``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- ldd

		- Leading dimension of tensor D.

	*
		- d_dt

		- Data type of tensor D.

	*
		- po

		- Primitive post-operation attributes to extend the kernel operations.

.. index:: pair: function; set_A_scales
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a95d2fe8bc5e184eb74374c99f9111f72:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_A_scales(int a_scale_mask)

Sets tensor A scales mask to a BRGeMM ukernel object.

For quantization flavor tensor A scales apply to accumulation buffer once C is ready.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- a_scale_mask

		- Tensor A scale mask. Can be ``0`` only.

.. index:: pair: function; set_B_scales
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a0b17ef5afc621818865d6e41dba66ccc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_B_scales(int b_scale_mask)

Sets tensor B scales mask to a BRGeMM ukernel object.

For quantization flavor tensor B scales apply to accumulation buffer once C is ready.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- b_scale_mask

		- Tensor B scale mask. Can be ``0`` and ``2`` only.

.. index:: pair: function; set_D_scales
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a857f6b4ed065e10519dfce02dc293e6f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_D_scales(int d_scale_mask)

Sets tensor D scales mask to a BRGeMM ukernel object.

For quantization flavor tensor D scales apply after all post-ops are applied.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- d_scale_mask

		- Tensor D scale mask. Can be ``0`` only.

.. index:: pair: function; finalize
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a80543f101b056823aeed10238db70da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void finalize()

Finalizes initialization of a BRGeMM ukernel object.

This step must be performed prior to querying information from the object.

.. index:: pair: function; get_B_pack_type
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a41509e7835003df7976c59193b6a6f31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`pack_type<doxid-group__dnnl__api__ukernel__utils_1ga241c23d0afdf43a79d51ef701a9f7c54>` get_B_pack_type() const

Returns the packing type expected by a tensor B of a BRGeMM ukernel object.

.. index:: pair: function; get_scratchpad_size
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1ada0b6984b8b9253cba9756c680c07d16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_scratchpad_size() const

Returns the size of a scratchpad memory needed for the BRGeMM ukernel object.

.. index:: pair: function; is_execute_postops_valid
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a2636a460ecb30c8c9535d8c18858c1ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool is_execute_postops_valid() const

Returns the flag indicating when the call to execute with post operations is valid.

``True`` is for a valid call, ``false``, otherwise.

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
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a53bcfebaf4e7ee3f099f6620e6e9ac50:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(
		const void* A,
		const void* B,
		const std::vector<std::pair<:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`, :ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`>>& A_B_offsets,
		const void* C,
		void* D,
		void* scratchpad,
		const :ref:`attr_params<doxid-structdnnl_1_1ukernel_1_1attr__params>`& params = :ref:`default_attr_params<doxid-structdnnl_1_1ukernel_1_1brgemm_1a529f502a78858748caaa10f8db3917bd>`()
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
		- params

		- Post-op memory arguments. Must be passed If binary post-op or scales were set.

.. index:: pair: function; release_hw_context
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a4cdc1e8b77991a2da8a69ae5f4ce267a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static void release_hw_context()

Releases the hardware-specific context.

Affects the global state for all BRGeMM ukernel objects. Must be used after all the execution calls to BRGeMM ukernel objects.

.. index:: pair: function; default_post_ops
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1acea25e2313b6211468852d4c7d76f436:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static const :ref:`post_ops<doxid-structdnnl_1_1post__ops>`& default_post_ops()

Returns a constant reference to a static instance of default constructed primitive post-operations attribute.

.. index:: pair: function; default_attr_params
.. _doxid-structdnnl_1_1ukernel_1_1brgemm_1a529f502a78858748caaa10f8db3917bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static const :ref:`attr_params<doxid-structdnnl_1_1ukernel_1_1attr__params>`& default_attr_params()

Returns a constant reference to a static instance of default constructed ukernel attributes parameters.

