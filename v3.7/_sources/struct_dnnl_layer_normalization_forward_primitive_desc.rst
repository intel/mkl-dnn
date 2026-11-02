.. index:: pair: struct; dnnl::layer_normalization_forward::primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc:

struct dnnl::layer_normalization_forward::primitive_desc
========================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Primitive descriptor for a layer normalization forward propagation primitive. :ref:`More...<details-structdnnl_1_1layer__normalization__forward_1_1primitive__desc>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_desc: public :ref:`dnnl::primitive_desc<doxid-structdnnl_1_1primitive__desc>`
	{
		// construction
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a9efade0e7db1ca2797b333796d58f246>`();
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1afe4d030a6a94a1343252c1c16ce3fb41>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& stat_desc,
			float epsilon,
			:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ab44622637ad01f175872a3ed7f460a20>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
			float epsilon,
			:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ae4964016291f8ae9af2e73b9fd13048b>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& stat_desc,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` scale_shift_data_type,
			float epsilon,
			:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ace7a8a1f84b25bce77ef8d89b6b0af97>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` scale_shift_data_type,
			float epsilon,
			:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a2ef6471431f9d9007e3855e1d763f276>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd);

		// methods
	
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a15f2d8ad523655199b6b9eed1a906e16>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a0f9a0d3bfab1851f4f1ed6af9e7bc511>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a8449344876675993b5dcea83612fcc60>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`workspace_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a1f0dd45d979923e85cc2fa81de24d772>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`mean_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ae0ad9c407ce41e013def79956309881e>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`variance_desc<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a5446956542a78e8bdee3bde838c6e9b1>`() const;
		:ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` :ref:`get_prop_kind<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a2101ada1fb7c3b7d4fbc44826efefc25>`() const;
		float :ref:`get_epsilon<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a4176831819b84c018e2d0e33bc3d6ee0>`() const;
		:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` :ref:`get_flags<doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a5088469b6b468ed347e169b00377cc89>`() const;
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

.. _details-structdnnl_1_1layer__normalization__forward_1_1primitive__desc:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor for a layer normalization forward propagation primitive.

Construction
------------

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a9efade0e7db1ca2797b333796d58f246:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc()

Default constructor. Produces an empty object.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1afe4d030a6a94a1343252c1c16ce3fb41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& stat_desc,
		float epsilon,
		:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a layer normalization forward propagation primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aprop_kind

		- Propagation kind. Possible values are :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, and :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- stat_desc

		- Statistics memory descriptors.

	*
		- epsilon

		- Layer normalization epsilon parameter.

	*
		- flags

		- Layer normalization flags (:ref:`dnnl::normalization_flags <doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`).

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ab44622637ad01f175872a3ed7f460a20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
		float epsilon,
		:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a layer normalization forward propagation primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aprop_kind

		- Propagation kind. Possible values are :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, and :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- epsilon

		- Layer normalization epsilon parameter.

	*
		- flags

		- Layer normalization flags (:ref:`dnnl::normalization_flags <doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`).

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ae4964016291f8ae9af2e73b9fd13048b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& stat_desc,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` scale_shift_data_type,
		float epsilon,
		:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a layer normalization forward propagation primitive with a user-provided data type for the scale and shift memory objects.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aprop_kind

		- Propagation kind. Possible values are :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, and :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- stat_desc

		- Statistics memory descriptors.

	*
		- scale_shift_data_type

		- Data type of scale and shift memory. If neither scale nor shift flag are specified the parameter is ignored.

	*
		- epsilon

		- Layer normalization epsilon parameter.

	*
		- flags

		- Layer normalization flags (:ref:`dnnl::normalization_flags <doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`).

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ace7a8a1f84b25bce77ef8d89b6b0af97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_desc,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` scale_shift_data_type,
		float epsilon,
		:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a layer normalization forward propagation primitive with a user-provided data type for the scale and shift memory objects.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aprop_kind

		- Propagation kind. Possible values are :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, and :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- scale_shift_data_type

		- Data type of scale and shift memory. If neither scale nor shift flag are specified the parameter is ignored.

	*
		- epsilon

		- Layer normalization epsilon parameter.

	*
		- flags

		- Layer normalization flags (:ref:`dnnl::normalization_flags <doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`).

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a2ef6471431f9d9007e3855e1d763f276:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd)

Constructs a primitive descriptor for a layer normalization forward propagation primitive from a C API primitive descriptor that must have a matching kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- C API primitive descriptor for a layer normalization forward propagation primitive.

Methods
-------

.. index:: pair: function; src_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a15f2d8ad523655199b6b9eed1a906e16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_desc() const

Returns a source memory descriptor.



.. rubric:: Returns:

Source memory descriptor.

A zero memory descriptor if the primitive does not have a source parameter.

.. index:: pair: function; dst_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a0f9a0d3bfab1851f4f1ed6af9e7bc511:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_desc() const

Returns a destination memory descriptor.



.. rubric:: Returns:

Destination memory descriptor.

A zero memory descriptor if the primitive does not have a destination parameter.

.. index:: pair: function; weights_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a8449344876675993b5dcea83612fcc60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_desc() const

Returns a weights memory descriptor.



.. rubric:: Returns:

Weights memory descriptor.

A zero memory descriptor if the primitive does not have a weights parameter.

.. index:: pair: function; workspace_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a1f0dd45d979923e85cc2fa81de24d772:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` workspace_desc() const

Returns the workspace memory descriptor.



.. rubric:: Returns:

Workspace memory descriptor.

A zero memory descriptor if the primitive does not require workspace parameter.

.. index:: pair: function; mean_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1ae0ad9c407ce41e013def79956309881e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` mean_desc() const

Returns memory descriptor for mean.



.. rubric:: Returns:

Memory descriptor for mean.

.. index:: pair: function; variance_desc
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a5446956542a78e8bdee3bde838c6e9b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` variance_desc() const

Returns memory descriptor for variance.



.. rubric:: Returns:

Memory descriptor for variance.

.. index:: pair: function; get_prop_kind
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a2101ada1fb7c3b7d4fbc44826efefc25:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` get_prop_kind() const

Returns a propagation kind.



.. rubric:: Returns:

A propagation kind.

:ref:`dnnl::prop_kind::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have a propagation parameter.

.. index:: pair: function; get_epsilon
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a4176831819b84c018e2d0e33bc3d6ee0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_epsilon() const

Returns an epsilon.



.. rubric:: Returns:

An epsilon.

Zero if the primitive does not have an epsilon parameter.

.. index:: pair: function; get_flags
.. _doxid-structdnnl_1_1layer__normalization__forward_1_1primitive__desc_1a5088469b6b468ed347e169b00377cc89:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` get_flags() const

Returns normalization flags.



.. rubric:: Returns:

Normalization flags.

