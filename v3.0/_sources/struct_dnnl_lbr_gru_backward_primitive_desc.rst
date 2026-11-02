.. index:: pair: struct; dnnl::lbr_gru_backward::primitive_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc:

struct dnnl::lbr_gru_backward::primitive_desc
=============================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Primitive descriptor for an LBR GRU backward propagation primitive. :ref:`More...<details-structdnnl_1_1lbr__gru__backward_1_1primitive__desc>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_desc: public :ref:`dnnl::rnn_primitive_desc_base<doxid-structdnnl_1_1rnn__primitive__desc__base>`
	{
		// construction
	
		:ref:`primitive_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a72261d946a2fe263c884aaabaaca2164>`();
	
		:ref:`primitive_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aafc821875f604af4b6c1f6cf1f1dccf6>`(
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
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_iter_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_weights_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_weights_iter_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_bias_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_layer_desc,
			const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_iter_desc,
			const :ref:`lbr_gru_forward::primitive_desc<doxid-structdnnl_1_1lbr__gru__forward_1_1primitive__desc>`& hint_fwd_pd,
			const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
			bool allow_empty = false
			);
	
		:ref:`primitive_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a8cccd71b52176075c6775aebb10fac30>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd);

		// methods
	
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ac5e76271c4b1068efdf67fa2df2ebd0e>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a89f422bb6d9aa4c2a44575ef6ea900c5>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a0d00a5ab343ec1e2eaf698d2cbe2c782>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a84aa1828c6d7b420c16f83cc382644a6>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`bias_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aaae06ada730fed7102e49ff27d23b624>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1abdc21cc1dcfa82716f9a6dfa3a022e83>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a737ba2657ae745e7726002a08e639b5b>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`workspace_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aaa2ee5edf648714875439a5b16727c1e>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1af20e858fb85afcd1de1e28f765a5b8a4>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a8b74335cfbb97f998112552711213f7a>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a0e782933010de9ecae223edc3e0072e6>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aced33d6588cec7110c1589ff3cce5f6d>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_bias_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a63ce41ac65185ba5c92aa60ad0b44017>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_layer_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ae021142c658dc528db8451e63de9db6d>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_iter_desc<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a21f8cb5bad1d721e7f01a3d618dd6b3b>`() const;
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_cell_kind<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a685d66098d54044f918cacd74a8efc80>`() const;
		:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` :ref:`get_prop_kind<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ace252d157064d7457398a535facb1c30>`() const;
		:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` :ref:`get_direction<doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a25c8f48ec968dde102771f6a1bdf43b6>`() const;
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

.. _details-structdnnl_1_1lbr__gru__backward_1_1primitive__desc:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor for an LBR GRU backward propagation primitive.

Construction
------------

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a72261d946a2fe263c884aaabaaca2164:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc()

Default constructor. Produces an empty object.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aafc821875f604af4b6c1f6cf1f1dccf6:

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
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_src_iter_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_weights_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_weights_iter_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_bias_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_layer_desc,
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& diff_dst_iter_desc,
		const :ref:`lbr_gru_forward::primitive_desc<doxid-structdnnl_1_1lbr__gru__forward_1_1primitive__desc>`& hint_fwd_pd,
		const :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`& attr = :ref:`default_attr<doxid-structdnnl_1_1primitive__desc__base_1a11b068a79808c04e4f9bcaffb0e34bb2>`(),
		bool allow_empty = false
		)

Constructs a primitive descriptor for LBR GRU backward propagation primitive.

The following arguments may point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the LBR GRU backward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors may be initialized with :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` value of ``format_tag``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to use.

	*
		- aprop_kind

		- Propagation kind. Must be :ref:`dnnl::prop_kind::backward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502>`.

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
		- diff_src_layer_desc

		- Memory descriptor for the diff of input vector.

	*
		- diff_src_iter_desc

		- Memory descriptor for the diff of input recurrent hidden state vector.

	*
		- diff_weights_layer_desc

		- Memory descriptor for the diff of weights applied to the layer input.

	*
		- diff_weights_iter_desc

		- Memory descriptor for the diff of weights applied to the recurrent input.

	*
		- diff_bias_desc

		- Diff bias memory descriptor.

	*
		- diff_dst_layer_desc

		- Memory descriptor for the diff of output vector.

	*
		- diff_dst_iter_desc

		- Memory descriptor for the diff of output recurrent hidden state vector.

	*
		- hint_fwd_pd

		- Primitive descriptor for an LBR GRU forward propagation primitive. It is used as a hint for deciding which memory format to use.

	*
		- attr

		- Primitive attributes to use. Attributes are optional and default to empty attributes.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case an empty object will be produced. This flag is optional and defaults to false.

.. index:: pair: function; primitive_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a8cccd71b52176075c6775aebb10fac30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_desc(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` pd)

Constructs a primitive descriptor for a LBR GRU backward propagation primitive from a C API primitive descriptor that must have a matching kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- C API primitive descriptor for a LBR GRU backward propagation primitive.

Methods
-------

.. index:: pair: function; src_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ac5e76271c4b1068efdf67fa2df2ebd0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_layer_desc() const

Returns source layer memory descriptor.



.. rubric:: Returns:

Source layer memory descriptor.

.. index:: pair: function; src_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a89f422bb6d9aa4c2a44575ef6ea900c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_iter_desc() const

Returns source iteration memory descriptor.



.. rubric:: Returns:

Source iteration memory descriptor.

A zero memory descriptor if the primitive does not have a source iteration parameter.

.. index:: pair: function; weights_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a0d00a5ab343ec1e2eaf698d2cbe2c782:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_layer_desc() const

Returns weights layer memory descriptor.



.. rubric:: Returns:

Weights layer memory descriptor.

.. index:: pair: function; weights_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a84aa1828c6d7b420c16f83cc382644a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_iter_desc() const

Returns weights iteration memory descriptor.



.. rubric:: Returns:

Weights iteration memory descriptor.

.. index:: pair: function; bias_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aaae06ada730fed7102e49ff27d23b624:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` bias_desc() const

Returns bias memory descriptor.



.. rubric:: Returns:

Bias memory descriptor.

A zero memory descriptor if the primitive does not have a bias parameter.

.. index:: pair: function; dst_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1abdc21cc1dcfa82716f9a6dfa3a022e83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_layer_desc() const

Returns destination layer memory descriptor.



.. rubric:: Returns:

Destination layer memory descriptor.

.. index:: pair: function; dst_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a737ba2657ae745e7726002a08e639b5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_iter_desc() const

Returns destination iteration memory descriptor.



.. rubric:: Returns:

Destination iteration memory descriptor.

A zero memory descriptor if the primitive does not have a destination iteration parameter.

.. index:: pair: function; workspace_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aaa2ee5edf648714875439a5b16727c1e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` workspace_desc() const

Returns the workspace memory descriptor.



.. rubric:: Returns:

Workspace memory descriptor.

A zero memory descriptor if the primitive does not require workspace parameter.

.. index:: pair: function; diff_src_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1af20e858fb85afcd1de1e28f765a5b8a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_src_layer_desc() const

Returns diff source layer memory descriptor.



.. rubric:: Returns:

Diff source layer memory descriptor.

.. index:: pair: function; diff_src_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a8b74335cfbb97f998112552711213f7a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_src_iter_desc() const

Returns diff source iteration memory descriptor.



.. rubric:: Returns:

Diff source iteration memory descriptor.

A zero memory descriptor if the primitive does not have a diff source iteration parameter.

.. index:: pair: function; diff_weights_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a0e782933010de9ecae223edc3e0072e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_weights_layer_desc() const

Returns diff weights layer memory descriptor.



.. rubric:: Returns:

Diff weights layer memory descriptor.

.. index:: pair: function; diff_weights_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1aced33d6588cec7110c1589ff3cce5f6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_weights_iter_desc() const

Returns diff weights iteration memory descriptor.



.. rubric:: Returns:

Diff weights iteration memory descriptor.

.. index:: pair: function; diff_bias_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a63ce41ac65185ba5c92aa60ad0b44017:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_bias_desc() const

Returns diff bias memory descriptor.



.. rubric:: Returns:

Diff bias memory descriptor.

A zero memory descriptor if the primitive does not have a diff bias parameter.

.. index:: pair: function; diff_dst_layer_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ae021142c658dc528db8451e63de9db6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_dst_layer_desc() const

Returns diff destination layer memory descriptor.



.. rubric:: Returns:

Diff destination layer memory descriptor.

.. index:: pair: function; diff_dst_iter_desc
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a21f8cb5bad1d721e7f01a3d618dd6b3b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_dst_iter_desc() const

Returns diff destination iteration memory descriptor.



.. rubric:: Returns:

Diff destination iteration memory descriptor.

A zero memory descriptor if the primitive does not have a diff destination iteration parameter.

.. index:: pair: function; get_cell_kind
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a685d66098d54044f918cacd74a8efc80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` get_cell_kind() const

Returns an RNN cell kind parameter.



.. rubric:: Returns:

An RNN cell kind parameter.

:ref:`dnnl::algorithm::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN cell kind parameter.

.. index:: pair: function; get_prop_kind
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1ace252d157064d7457398a535facb1c30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` get_prop_kind() const

Returns a propagation kind.



.. rubric:: Returns:

A propagation kind.

:ref:`dnnl::prop_kind::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have a propagation parameter.

.. index:: pair: function; get_direction
.. _doxid-structdnnl_1_1lbr__gru__backward_1_1primitive__desc_1a25c8f48ec968dde102771f6a1bdf43b6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` get_direction() const

Returns an RNN direction parameter.



.. rubric:: Returns:

An RNN direction parameter.

:ref:`dnnl::rnn_direction::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN direction parameter.

