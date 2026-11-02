.. index:: pair: group; RNN
.. _doxid-group__dnnl__api__rnn:

RNN
===

.. toctree::
	:hidden:

	enum_dnnl_rnn_direction_t.rst
	enum_dnnl_rnn_flags_t.rst
	enum_dnnl_rnn_direction.rst
	enum_dnnl_rnn_flags.rst
	struct_dnnl_augru_backward.rst
	struct_dnnl_augru_forward.rst
	struct_dnnl_gru_backward.rst
	struct_dnnl_gru_forward.rst
	struct_dnnl_lbr_augru_backward.rst
	struct_dnnl_lbr_augru_forward.rst
	struct_dnnl_lbr_gru_backward.rst
	struct_dnnl_lbr_gru_forward.rst
	struct_dnnl_lstm_backward.rst
	struct_dnnl_lstm_forward.rst
	struct_dnnl_rnn_primitive_desc_base.rst
	struct_dnnl_vanilla_rnn_backward.rst
	struct_dnnl_vanilla_rnn_forward.rst

Overview
~~~~~~~~

A primitive to compute recurrent neural network layers. :ref:`More...<details-group__dnnl__api__rnn>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>`;
	enum :ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>`;
	enum :ref:`dnnl::rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>`;
	enum :ref:`dnnl::rnn_flags<doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>`;

	// structs

	struct :ref:`dnnl::augru_backward<doxid-structdnnl_1_1augru__backward>`;
	struct :ref:`dnnl::augru_forward<doxid-structdnnl_1_1augru__forward>`;
	struct :ref:`dnnl::gru_backward<doxid-structdnnl_1_1gru__backward>`;
	struct :ref:`dnnl::gru_forward<doxid-structdnnl_1_1gru__forward>`;
	struct :ref:`dnnl::lbr_augru_backward<doxid-structdnnl_1_1lbr__augru__backward>`;
	struct :ref:`dnnl::lbr_augru_forward<doxid-structdnnl_1_1lbr__augru__forward>`;
	struct :ref:`dnnl::lbr_gru_backward<doxid-structdnnl_1_1lbr__gru__backward>`;
	struct :ref:`dnnl::lbr_gru_forward<doxid-structdnnl_1_1lbr__gru__forward>`;
	struct :ref:`dnnl::lstm_backward<doxid-structdnnl_1_1lstm__backward>`;
	struct :ref:`dnnl::lstm_forward<doxid-structdnnl_1_1lstm__forward>`;
	struct :ref:`dnnl::rnn_primitive_desc_base<doxid-structdnnl_1_1rnn__primitive__desc__base>`;
	struct :ref:`dnnl::vanilla_rnn_backward<doxid-structdnnl_1_1vanilla__rnn__backward>`;
	struct :ref:`dnnl::vanilla_rnn_forward<doxid-structdnnl_1_1vanilla__rnn__forward>`;

	// global functions

	:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__rnn_1ga0a340195a137f906e858418d91397777>`(:ref:`rnn_flags<doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>` flags);
	:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__rnn_1ga1915ea2d2fe94077fa30734ced88a225>`(:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` dir);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_vanilla_rnn_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1gacae0d89a99432e71bf935f4b8bc8c370>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		const :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` activation,
		const :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_vanilla_rnn_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga01f9132fb153989baaa01f5d2f4f9097>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		const :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` activation,
		const :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lstm_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga7184f51a15df7f2d0105caf4bb810c5e>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_c_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lstm_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga7b80d3a81aa9772d20e6061cbf4ffc32>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_c_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gru_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1gaa2c83abbb22c101e2369161750d80e5e>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gru_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga1911de70db4048436e71c1ed90d2c214>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lbr_gru_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga399570ef09834faa8b426ca8dbe231ae>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lbr_gru_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga0a6d44b064f5c819ed335d3061dbcc7d>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_augru_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga53c0f055391601f4e305f50fe1459f8b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_augru_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1gadf5b33eaacf3396fa2f68b2cb9fff2aa>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lbr_augru_forward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga5ed039eae7580b12e460095f026c6139>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lbr_augru_backward_primitive_desc_create<doxid-group__dnnl__api__rnn_1ga8d9f06ca2b9280715cd8dc9cabb2297e>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__rnn:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to compute recurrent neural network layers.



.. rubric:: See also:

:ref:`RNN <doxid-dev_guide_rnn>` in developer guide

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__rnn_1ga0a340195a137f906e858418d91397777:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` dnnl::convert_to_c(:ref:`rnn_flags<doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>` flags)

Converts RNN cell flags enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flags

		- C++ API RNN cell flags enum value.



.. rubric:: Returns:

Corresponding C API RNN cell flags enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__rnn_1ga1915ea2d2fe94077fa30734ced88a225:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` dnnl::convert_to_c(:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` dir)

Converts RNN direction enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- dir

		- C++ API RNN direction enum value.



.. rubric:: Returns:

Corresponding C API RNN direction enum value.

.. index:: pair: function; dnnl_vanilla_rnn_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1gacae0d89a99432e71bf935f4b8bc8c370:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_vanilla_rnn_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		const :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` activation,
		const :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for vanilla RNN forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the RNN forward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- activation

		- Activation kind. Possible values are :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, :ref:`dnnl_eltwise_tanh <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a81b20d8f0b54c7114024186a9fbb698e>` or :ref:`dnnl_eltwise_logistic <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ab560981bee9e7711017423e29ba46071>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- alpha

		- Negative slope if activation is :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`.

	*
		- beta

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_vanilla_rnn_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga01f9132fb153989baaa01f5d2f4f9097:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_vanilla_rnn_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		const :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` activation,
		const :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for vanilla RNN backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the RNN backward propagation primitive should not use the respective data and should use zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- activation

		- Activation kind. Possible values are :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, :ref:`dnnl_eltwise_tanh <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a81b20d8f0b54c7114024186a9fbb698e>` or :ref:`dnnl_eltwise_logistic <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ab560981bee9e7711017423e29ba46071>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- alpha

		- Negative slope if activation is :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`.

	*
		- beta

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lstm_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga7184f51a15df7f2d0105caf4bb810c5e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lstm_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_c_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an LSTM forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``src_iter_c_desc``,

* ``weights_peephole_desc``,

* ``bias_desc``,

* ``dst_iter_desc`` together with ``dst_iter_c_desc``.

This would then indicate that the LSTM forward propagation primitive should not use them and should default to zero values instead.

The ``weights_projection_desc`` could either be ``NULL`` or point to a zero memory descriptor. This would then indicate that the LSTM doesn't have recurrent projection layer.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- src_iter_c_desc

		- Memory descriptor for the input recurrent cell state vector.

	*
		- weights_layer_desc

		- Memory descriptor for the weights applied to the layer input.

	*
		- weights_iter_desc

		- Memory descriptor for the weights applied to the recurrent input.

	*
		- weights_peephole_desc

		- Memory descriptor for the weights applied to the cell states (according to the Peephole LSTM formula).

	*
		- weights_projection_desc

		- Memory descriptor for the weights applied to the hidden states to get the recurrent projection (according to the Projection LSTM formula).

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
		- dst_iter_c_desc

		- Memory descriptor for the output recurrent cell state vector.

	*
		- flags

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lstm_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga7b80d3a81aa9772d20e6061cbf4ffc32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lstm_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_c_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_peephole_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_projection_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_c_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an LSTM backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``src_iter_c_desc``, ``diff_src_iter_desc``, and ``diff_src_iter_c_desc``,

* ``weights_peephole_desc`` together with ``diff_weights_peephole_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``dst_iter_c_desc``, ``diff_dst_iter_desc``, and ``diff_dst_iter_c_desc``.

This would then indicate that the LSTM backward propagation primitive should not use them and should default to zero values instead.

The ``weights_projection_desc`` together with ``diff_weights_projection_desc`` could either be ``NULL`` or point to a zero memory descriptor. This would then indicate that the LSTM doesn't have recurrent projection layer.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- src_iter_c_desc

		- Memory descriptor for the input recurrent cell state vector.

	*
		- weights_layer_desc

		- Memory descriptor for the weights applied to the layer input.

	*
		- weights_iter_desc

		- Memory descriptor for the weights applied to the recurrent input.

	*
		- weights_peephole_desc

		- Memory descriptor for the weights applied to the cell states (according to the Peephole LSTM formula).

	*
		- weights_projection_desc

		- Memory descriptor for the weights applied to the hidden states to get the recurrent projection (according to the Projection LSTM formula).

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
		- dst_iter_c_desc

		- Memory descriptor for the output recurrent cell state vector.

	*
		- diff_src_layer_desc

		- Memory descriptor for the diff of input vector.

	*
		- diff_src_iter_desc

		- Memory descriptor for the diff of input recurrent hidden state vector.

	*
		- diff_src_iter_c_desc

		- Memory descriptor for the diff of input recurrent cell state vector.

	*
		- diff_weights_layer_desc

		- Memory descriptor for the diff of weights applied to the layer input.

	*
		- diff_weights_iter_desc

		- Memory descriptor for the diff of weights applied to the recurrent input.

	*
		- diff_weights_peephole_desc

		- Memory descriptor for the diff of weights applied to the cell states (according to the Peephole LSTM formula).

	*
		- diff_weights_projection_desc

		- Memory descriptor for the diff of weights applied to the hidden states to get the recurrent projection (according to the Projection LSTM formula).

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
		- diff_dst_iter_c_desc

		- Memory descriptor for the diff of output recurrent cell state vector.

	*
		- flags

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_gru_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1gaa2c83abbb22c101e2369161750d80e5e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_gru_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for GRU forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the GRU forward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_gru_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga1911de70db4048436e71c1ed90d2c214:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_gru_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for GRU backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the GRU backward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lbr_gru_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga399570ef09834faa8b426ca8dbe231ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lbr_gru_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a descriptor for LBR GRU forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the LBR GRU forward propagation primitive should not use them and should default to zero values instead.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lbr_gru_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga0a6d44b064f5c819ed335d3061dbcc7d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lbr_gru_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for LBR GRU backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the LBR GRU backward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

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
		- flags

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_augru_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga53c0f055391601f4e305f50fe1459f8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_augru_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for AUGRU forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the AUGRU forward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- attention_desc

		- Memory descriptor for the attention vector.

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
		- flags

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_augru_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1gadf5b33eaacf3396fa2f68b2cb9fff2aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_augru_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for AUGRU backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the AUGRU backward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- attention_desc

		- Memory descriptor for the attention vector.

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
		- diff_attention_desc

		- Memory descriptor for the diff of attention vector.

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
		- flags

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lbr_augru_forward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga5ed039eae7580b12e460095f026c6139:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lbr_augru_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for LBR AUGRU forward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc``,

* ``bias_desc``,

* ``dst_iter_desc``.

This would then indicate that the LBR AUGRU forward propagation primitive should not use them and should default to zero values instead.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- attention_desc

		- Memory descriptor for the attention vector.

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
		- flags

		- Unused.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lbr_augru_backward_primitive_desc_create
.. _doxid-group__dnnl__api__rnn_1ga8d9f06ca2b9280715cd8dc9cabb2297e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lbr_augru_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` direction,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_attention_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_iter_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_layer_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_iter_desc,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for LBR AUGRU backward propagation primitive.

The following arguments may either be ``NULL`` or point to a zero memory descriptor:

* ``src_iter_desc`` together with ``diff_src_iter_desc``,

* ``bias_desc`` together with ``diff_bias_desc``,

* ``dst_iter_desc`` together with ``diff_dst_iter_desc``.

This would then indicate that the LBR AUGRU backward propagation primitive should not use them and should default to zero values instead.

.. note:: 

   All memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Must be :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

	*
		- direction

		- RNN direction. See :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` for more info.

	*
		- src_layer_desc

		- Memory descriptor for the input vector.

	*
		- src_iter_desc

		- Memory descriptor for the input recurrent hidden state vector.

	*
		- attention_desc

		- Memory descriptor for the attention vector.

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
		- diff_attention_desc

		- Memory descriptor for the diff of attention vector.

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
		- flags

		- Unused.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

