.. index:: pair: struct; dnnl::gru_forward::primitive_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc:

struct dnnl::gru_forward::primitive_desc
========================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Primitive descriptor for a GRU forward propagation primitive. :ref:`More...<details-structdnnl_1_1gru__forward_1_1primitive__desc>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_desc: public :ref:`dnnl::rnn_primitive_desc_base<doxid-structdnnl_1_1rnn__primitive__desc__base>`
	{
		// construction
	
		:ref:`primitive_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a7dec21003d026bf7cec53362134be4a6>`();
	
		:ref:`primitive_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a26f3c84630a14a9a13fef55fa3e7eece>`(
			const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
			:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
			:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` direction,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_iter_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& weights_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& weights_iter_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& bias_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_iter_desc,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1ac70554927ef2e67e86ca4220969531eb>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd);

		// methods
	
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_layer_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a151aa8bd2688df79756db865fdcf2448>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_iter_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a9b77d6a1d123c22a4d0d029153cf47d6>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_layer_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a534a21c5cc64d91e09c8851c749f06f1>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_iter_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a146831b7fb433c08b3bf10448b3e48ea>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`bias_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a6dfda803c06a02bf85052121f68240b4>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_layer_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a1cd31d5990750746ad0358b718e477ba>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_iter_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a8941a399a805849a67e017ba0f6e41f5>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`workspace_desc<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a97f152c8ef732520ef7c6bb275a464e9>`() const;
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_cell_kind<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a25f64f9ca2424fe61f1e2516d5e0064d>`() const;
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` :ref:`get_prop_kind<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a26d73bd80953bf7661e7ee1cd460d872>`() const;
		:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` :ref:`get_direction<doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1ae7cf5ac751b168e7fbc40a9e5803cf05>`() const;
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
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ae49709b806f2cb108affe1ec5aebb504>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`augru_attention_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a392b10f6a2c15be9e8c37a53d3aed786>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ab73a02922a9a7d2f0d6066572d5c6a09>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_iter_c_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ab37877634785901c8d8850d1a928c9b2>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a20737480e3672640657a34afd631f11f>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a2464426ea23b9d026fd5d2c8d92c942a>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_peephole_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a1d634c656c92e69393cd52a4551c951b>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_projection_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ac42d2cf801857a02db5c23e4a4d4eba1>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`bias_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ac2045ddda996b5e3ac2c0974ca70bfa1>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1aa9a4652f8f5c8d1b32d2f26deac70f92>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1af82a946c412df434d8842292d77a496c>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_iter_c_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a1d56829ac12568aea3a7203631e7c202>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1afbf93e3cf53b5ef7d04a0b2fcb41f77b>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_augru_attention_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a6d15e90406d096ea3df0233bf164114c>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a248bcc0f5bec8d76eda24ad391693fa2>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_iter_c_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a6f3d6865c312c3231ff7dec8df561ecb>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a1df1f7cb00c68c73bc4320d7b5fda7bb>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a4243bb55a8d07fa66a3c44633996e73b>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_peephole_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1af8f947d6baeee3cded8f178c0d87532a>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_projection_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1aab56611714a0496cfe15a36e68b2c495>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_bias_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1aa81c0dbda6447fe83d1d37122fdc708c>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_layer_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1aeb8d331bd2bcb80bd06ec729175e4314>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_iter_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a4a5eceabe7b1ffd9f181376192f5d397>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_iter_c_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1a33e735717f2dffa22cc28e90e20bd344>`() const;
		:ref:`primitive_desc<doxid-structdnnl_1_1rnn__primitive__desc__base_1ae5452031cfd89bda919543dace0ff7bd>`();

.. _details-structdnnl_1_1gru__forward_1_1primitive__desc:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor for a GRU forward propagation primitive.

Construction
------------

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a7dec21003d026bf7cec53362134be4a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc()

Default constructor. Produces an empty object.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a26f3c84630a14a9a13fef55fa3e7eece:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` aprop_kind,
		:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` direction,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src_iter_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& weights_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& weights_iter_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& bias_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& dst_iter_desc,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for a GRU forward propagation primitive.

The following arguments may point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the GRU forward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors except ``src_iter_desc`` may be initialized with an :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` value of ``format_tag``.



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
		- direction

		- RNN direction. See :ref:`dnnl::rnn_direction <doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- weights_layer_desc

		- Memory descriptor for the weights applied to the layer input.

	*
		- weights_iter_desc

		- Memory descriptor for the weights applied to the recurrent input.

	*
		- bias_desc

		- Bias memory descriptor.

	*
		- dst_layer_desc

		- Memory descriptor for the output vector.

	*
		- dst_iter_desc

		- Memory descriptor for the output recurrent hidden state vector.

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1ac70554927ef2e67e86ca4220969531eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd)

Constructs a primitive descriptor for a GRU forward propagation primitive from a C API primitive descriptor that must have a matching kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- C API primitive descriptor for a GRU forward propagation primitive.

Methods
-------

.. index:: pair: function; src_layer_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a151aa8bd2688df79756db865fdcf2448:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_layer_desc() const

Returns source layer memory descriptor.



.. rubric:: Returns:

Source layer memory descriptor.

.. index:: pair: function; src_iter_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a9b77d6a1d123c22a4d0d029153cf47d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_iter_desc() const

Returns source iteration memory descriptor.



.. rubric:: Returns:

Source iteration memory descriptor.

A zero memory descriptor if the primitive does not have a source iteration parameter.

.. index:: pair: function; weights_layer_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a534a21c5cc64d91e09c8851c749f06f1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_layer_desc() const

Returns weights layer memory descriptor.



.. rubric:: Returns:

Weights layer memory descriptor.

.. index:: pair: function; weights_iter_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a146831b7fb433c08b3bf10448b3e48ea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_iter_desc() const

Returns weights iteration memory descriptor.



.. rubric:: Returns:

Weights iteration memory descriptor.

.. index:: pair: function; bias_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a6dfda803c06a02bf85052121f68240b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` bias_desc() const

Returns bias memory descriptor.



.. rubric:: Returns:

Bias memory descriptor.

A zero memory descriptor if the primitive does not have a bias parameter.

.. index:: pair: function; dst_layer_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a1cd31d5990750746ad0358b718e477ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_layer_desc() const

Returns destination layer memory descriptor.



.. rubric:: Returns:

Destination layer memory descriptor.

.. index:: pair: function; dst_iter_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a8941a399a805849a67e017ba0f6e41f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_iter_desc() const

Returns destination iteration memory descriptor.



.. rubric:: Returns:

Destination iteration memory descriptor.

A zero memory descriptor if the primitive does not have a destination iteration parameter.

.. index:: pair: function; workspace_desc
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a97f152c8ef732520ef7c6bb275a464e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` workspace_desc() const

Returns the workspace memory descriptor.



.. rubric:: Returns:

Workspace memory descriptor.

A zero memory descriptor if the primitive does not require workspace parameter.

.. index:: pair: function; get_cell_kind
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a25f64f9ca2424fe61f1e2516d5e0064d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` get_cell_kind() const

Returns an RNN cell kind parameter.



.. rubric:: Returns:

An RNN cell kind parameter.

:ref:`dnnl::algorithm::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN cell kind parameter.

.. index:: pair: function; get_prop_kind
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1a26d73bd80953bf7661e7ee1cd460d872:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` get_prop_kind() const

Returns a propagation kind.



.. rubric:: Returns:

A propagation kind.

:ref:`dnnl::prop_kind::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have a propagation parameter.

.. index:: pair: function; get_direction
.. _doxid-structdnnl_1_1gru__forward_1_1primitive__desc_1ae7cf5ac751b168e7fbc40a9e5803cf05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` get_direction() const

Returns an RNN direction parameter.



.. rubric:: Returns:

An RNN direction parameter.

:ref:`dnnl::rnn_direction::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN direction parameter.

