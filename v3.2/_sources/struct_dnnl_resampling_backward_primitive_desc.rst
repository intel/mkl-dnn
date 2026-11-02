.. index:: pair: struct; dnnl::resampling_backward::primitive_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc:

struct dnnl::resampling_backward::primitive_desc
================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Primitive descriptor for resampling backward propagation primitive. :ref:`More...<details-structdnnl_1_1resampling__backward_1_1primitive__desc>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_desc: public :ref:`dnnl::primitive_desc<doxid-structdnnl_1_1primitive__desc>`
	{
		// construction
	
		:ref:`primitive_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a7cf7a8fd26519795751e0b045e81efb4>`();
	
		:ref:`primitive_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a464390c3b2bcd92c3f3591b29ba77d6d>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_desc,
			const :ref:`resampling_forward::primitive_desc<doxid-structdnnl_1_1resampling__forward_1_1primitive__desc>`& hint_fwd_pd,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a0e53dbcffb53bd82eae5a13bbb79dd66>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm,
			const std::vector<float>& factors,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_desc,
			const :ref:`resampling_forward::primitive_desc<doxid-structdnnl_1_1resampling__forward_1_1primitive__desc>`& hint_fwd_pd,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a2487948c1906a70bf68144d236613bfb>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd);

		// methods
	
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a34b6453d3ffa6b0b4fddd5aaf14341dd>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_desc<doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1ae6b102a2549d6e850249a3880db22c3e>`() const;
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
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`get_engine<doxid-structdnnl_1_1primitive__desc__base_1a32f7477c79e715a341bb9127df521fbc>`() const;
		const char* :ref:`impl_info_str<doxid-structdnnl_1_1primitive__desc__base_1ae680492d4e4b16a938cfc051e5c906be>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`query_s64<doxid-structdnnl_1_1primitive__desc__base_1acbedc4257eaa26f868356f9f594a856a>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what) const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_strides<doxid-structdnnl_1_1primitive__desc__base_1ab15fa4e08b75652e106ec7b7d6b13e8f>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_dilations<doxid-structdnnl_1_1primitive__desc__base_1a0f2ef952e57da908a074422822e6dbaf>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padding_l<doxid-structdnnl_1_1primitive__desc__base_1aa21112a5d3d0d38a47b9bb74024c5904>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padding_r<doxid-structdnnl_1_1primitive__desc__base_1a004d4371015dc3ba06d57752b4ec1bb6>`() const;
		float :ref:`get_epsilon<doxid-structdnnl_1_1primitive__desc__base_1ab8c5aaea19030d62c667d4b79eb0d680>`() const;
	
		template <typename T = unsigned>
		T :ref:`get_flags<doxid-structdnnl_1_1primitive__desc__base_1a8081d18d2289f5b58634a2b902bea264>`() const;
	
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_algorithm<doxid-structdnnl_1_1primitive__desc__base_1ae220e7adee4466f48e9a4781f0754461>`() const;
		float :ref:`get_alpha<doxid-structdnnl_1_1primitive__desc__base_1a73f6fcc68fd45166146a3c3d5eb49821>`() const;
		float :ref:`get_beta<doxid-structdnnl_1_1primitive__desc__base_1a29619c9f1308414bfdc8eb89fae7bd9d>`() const;
		int :ref:`get_axis<doxid-structdnnl_1_1primitive__desc__base_1af3b18a5a286f3b46e98b6fe8f45d60ac>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_local_size<doxid-structdnnl_1_1primitive__desc__base_1a00e6f182172077df46fc6f614e309592>`() const;
		float :ref:`get_k<doxid-structdnnl_1_1primitive__desc__base_1ad28f7f1c59d2fde215d71483daaaa632>`() const;
		float :ref:`get_p<doxid-structdnnl_1_1primitive__desc__base_1acfde33696840f499f6e790e415bd65ba>`() const;
		std::vector<float> :ref:`get_factors<doxid-structdnnl_1_1primitive__desc__base_1a18ce555604ee8680353bb95aeca08665>`() const;
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_cell_kind<doxid-structdnnl_1_1primitive__desc__base_1a940546a82d1381597863ecc7036b9c22>`() const;
		:ref:`dnnl::rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` :ref:`get_direction<doxid-structdnnl_1_1primitive__desc__base_1a7b9b95f2e16e3d2a9644df7ad5436f63>`() const;
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_activation_kind<doxid-structdnnl_1_1primitive__desc__base_1acc309d5adeb8dbe8a88a18105082d566>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_kernel<doxid-structdnnl_1_1primitive__desc__base_1a3c9a39d1d0375518b18cf762fdd04b7e>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_group_size<doxid-structdnnl_1_1primitive__desc__base_1a03ad4fc755ae405a8a13d2c238266a15>`() const;
		:ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` :ref:`get_prop_kind<doxid-structdnnl_1_1primitive__desc__base_1a9adba18a967c92b205e939ba34de3542>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`query_md<doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what, int idx = 0) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_desc<doxid-structdnnl_1_1primitive__desc__base_1af42e791f493e636c086e13c6d4c06b43>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a495ee7c8e1ec3eab35f6329fdcd352bb>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_desc<doxid-structdnnl_1_1primitive__desc__base_1acb13d08987cca8d8f05ec4858fa61fb4>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_desc<doxid-structdnnl_1_1primitive__desc__base_1a733e6ff4e78a758e69ae6232e8955871>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a60f17d04c493c42e4a50ad5feff5c8ca>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_desc<doxid-structdnnl_1_1primitive__desc__base_1a192f7b334efac9a5ac20344a76d4d552>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_desc<doxid-structdnnl_1_1primitive__desc__base_1af48dcff294cadb2916fd784b8474d221>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_desc<doxid-structdnnl_1_1primitive__desc__base_1addbec977643a7900f4156f7aab3fb4db>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_desc<doxid-structdnnl_1_1primitive__desc__base_1a93f0904566b399874c47b3b1ad3d1495>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_desc<doxid-structdnnl_1_1primitive__desc__base_1ab4268c6bb70dd6c22de43141cc301b77>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a6b9fb7da987329256c04db6ecbb9dc36>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_desc<doxid-structdnnl_1_1primitive__desc__base_1af51378982968e2b4f7abb6a32acfb0af>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`workspace_desc<doxid-structdnnl_1_1primitive__desc__base_1ad26f416a149cb44cc5cfc130012c614e>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`scratchpad_desc<doxid-structdnnl_1_1primitive__desc__base_1a6238358ec03afd57fb20dffa65b48d2f>`() const;
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`scratchpad_engine<doxid-structdnnl_1_1primitive__desc__base_1ad4079e0891373bd2dc841f3f94cf47ed>`() const;
		:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>` :ref:`get_primitive_attr<doxid-structdnnl_1_1primitive__desc__base_1a0ef600ea3666f0fd93c9c5d112aaf05c>`() const;
		:ref:`dnnl::primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` :ref:`get_kind<doxid-structdnnl_1_1primitive__desc__base_1a663f3511855bfc2625476019321c909d>`() const;
		std::vector<uint8_t> :ref:`get_cache_blob_id<doxid-structdnnl_1_1primitive__desc__base_1a435862df4d543eb8296424880212b22d>`() const;
		bool :ref:`next_impl<doxid-structdnnl_1_1primitive__desc_1a841df469ca54c3de2d233e46f48322b2>`();
		:ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc_1a27780142d0880bb0ca678f7c5a1845b9>`();
		:ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc_1aae07f2f06d74537546c3056bd305dfbe>`();
		:ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc_1af4eac2eea0fd4eb37c0c90ead14ad52b>`();
		:ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc_1aa4853f3190cac45b653e510b6eeed97a>`();

.. _details-structdnnl_1_1resampling__backward_1_1primitive__desc:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor for resampling backward propagation primitive.

Construction
------------

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a7cf7a8fd26519795751e0b045e81efb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc()

Default constructor. Produces an empty object.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a464390c3b2bcd92c3f3591b29ba77d6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_desc,
		const :ref:`resampling_forward::primitive_desc<doxid-structdnnl_1_1resampling__forward_1_1primitive__desc>`& hint_fwd_pd,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a resampling backward propagation primitive using source and destination memory descriptors.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aalgorithm

		- resampling algorithm kind: either :ref:`dnnl::algorithm::resampling_nearest <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a091b262cfae5e610fcad1490bfb79431>`, or :ref:`dnnl::algorithm::resampling_linear <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a05731680a83c6dccd19a758b72a4ecbd>`

	*
		- diff_src_desc

		- Diff source memory descriptor.

	*
		- diff_dst_desc

		- Diff destination memory descriptor.

	*
		- hint_fwd_pd

		- Primitive descriptor for a resampling forward propagation primitive. It is used as a hint for deciding which memory format to use.

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a0e53dbcffb53bd82eae5a13bbb79dd66:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm,
		const std::vector<float>& factors,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_desc,
		const :ref:`resampling_forward::primitive_desc<doxid-structdnnl_1_1resampling__forward_1_1primitive__desc>`& hint_fwd_pd,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for resampling backward propagation primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aalgorithm

		- resampling algorithm kind: either :ref:`dnnl::algorithm::resampling_nearest <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a091b262cfae5e610fcad1490bfb79431>`, or :ref:`dnnl::algorithm::resampling_linear <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a05731680a83c6dccd19a758b72a4ecbd>`

	*
		- factors

		- Vector of scaling factors for spatial dimension.

	*
		- diff_src_desc

		- Diff source memory descriptor.

	*
		- diff_dst_desc

		- Diff destination memory descriptor.

	*
		- hint_fwd_pd

		- Primitive descriptor for a resampling forward propagation primitive. It is used as a hint for deciding which memory format to use.

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a2487948c1906a70bf68144d236613bfb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd)

Constructs a primitive descriptor for a resampling backward propagation primitive from a C API primitive descriptor that must have a matching kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- C API primitive descriptor for a resampling backward propagation primitive.

Methods
-------

.. index:: pair: function; diff_src_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1a34b6453d3ffa6b0b4fddd5aaf14341dd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_src_desc() const

Returns a diff source memory descriptor.



.. rubric:: Returns:

Diff source memory descriptor.

A zero memory descriptor if the primitive does not have a diff source memory with.

.. index:: pair: function; diff_dst_desc
.. _doxid-structdnnl_1_1resampling__backward_1_1primitive__desc_1ae6b102a2549d6e850249a3880db22c3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_dst_desc() const

Returns a diff destination memory descriptor.



.. rubric:: Returns:

Diff destination memory descriptor.

A zero memory descriptor if the primitive does not have a diff destination parameter.

