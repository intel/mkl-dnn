.. index:: pair: group; Attributes
.. _doxid-group__dnnl__api__attributes:

Attributes
==========

.. toctree::
	:hidden:

	enum_dnnl_algorithm.rst
	enum_dnnl_scratchpad_mode_t.rst
	enum_dnnl_prop_kind.rst
	enum_dnnl_scratchpad_mode.rst
	struct_dnnl_post_ops.rst
	struct_dnnl_primitive_attr.rst
	struct_dnnl_post_ops-2.rst
	struct_dnnl_primitive_attr-2.rst

Overview
~~~~~~~~

A container for parameters that extend primitives behavior. :ref:`More...<details-group__dnnl__api__attributes>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* :ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`;
	typedef const struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* :ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>`;
	typedef struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* :ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`;
	typedef const struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* :ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>`;

	// enums

	enum :ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`;
	enum :ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>`;
	enum :ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>`;
	enum :ref:`dnnl::scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>`;

	// structs

	struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`;
	struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`;
	struct :ref:`dnnl::post_ops<doxid-structdnnl_1_1post__ops>`;
	struct :ref:`dnnl::primitive_attr<doxid-structdnnl_1_1primitive__attr>`;

	// global functions

	:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__attributes_1gaa30f540e1ed09b2865f153fd599c967b>`(:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode);
	:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__attributes_1gae13881206fecd43ce0e0daead7f0009e>`(:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` akind);
	:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__attributes_1gad4c07d30e46391ce7ce0900d18cbfa30>`(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_create<doxid-group__dnnl__api__attributes_1gaf630fdc0d8d0fd8522ec93852a559081>`(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`* attr);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_clone<doxid-group__dnnl__api__attributes_1gab6ac5a4b13fa1ab3251c51f3c750bd63>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`* attr,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` existing_attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_destroy<doxid-group__dnnl__api__attributes_1ga96a7539382945195627f2932bff8fadb>`(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_fpmath_mode<doxid-group__dnnl__api__attributes_1gac63b70ab1d2fe88c31f03c961b2e924a>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_fpmath_mode<doxid-group__dnnl__api__attributes_1gafe55fa618bc10b65b6c0b6eca7e43840>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_fpmath_mode_v2<doxid-group__dnnl__api__attributes_1gab6dd49fedbc548aea2d6ede5a0c42a6c>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode,
		int* apply_to_int
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_fpmath_mode_v2<doxid-group__dnnl__api__attributes_1ga96edebcfaf7451fa96d698be110a18e9>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode,
		int apply_to_int
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_deterministic<doxid-group__dnnl__api__attributes_1gacb11e4d0243975ef944eb25fffe2ef0a>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		int* value
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_deterministic<doxid-group__dnnl__api__attributes_1ga69af4e29cba07fdb95672c070ac26511>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int value
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_accumulation_mode<doxid-group__dnnl__api__attributes_1ga1add29950cb3ec6595aebd572bcf7f92>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>`* mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_accumulation_mode<doxid-group__dnnl__api__attributes_1ga691c818641709d7dc94b92a2db7686b5>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>` mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_scratchpad_mode<doxid-group__dnnl__api__attributes_1gab14d8e830a52510a75a917f75764a6b8>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>`* mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_scratchpad_mode<doxid-group__dnnl__api__attributes_1ga4adeb17e538392ec3a16d2f6ef3f7cca>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_scales_mask<doxid-group__dnnl__api__attributes_1gad7eac877f75cfa282be094b1e48cb71d>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_scales<doxid-group__dnnl__api__attributes_1ga6c56ea6104c9275574370a40e2d5b273>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` group_dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_zero_points_mask<doxid-group__dnnl__api__attributes_1ga24e429b5410f5657bc5bdda0a6c5d0a7>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_zero_points<doxid-group__dnnl__api__attributes_1ga94f3e4f640fb0ca210c2413bb5cf2255>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` group_dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_post_ops<doxid-group__dnnl__api__attributes_1ga50c92661cc69e1eeb17b61f006320a05>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>`* post_ops
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_post_ops<doxid-group__dnnl__api__attributes_1ga7045d42606599f156bfca69820c21ea2>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_create<doxid-group__dnnl__api__attributes_1gaa8d8c32ad4472de464e47336ad702a48>`(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`* post_ops);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_clone<doxid-group__dnnl__api__attributes_1ga087b5f530ae5cfd1134cfad694e84de1>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`* post_ops,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` existing_post_ops
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_destroy<doxid-group__dnnl__api__attributes_1ga67487a65afa2e2066f4b4eb12d47535b>`(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops);
	int DNNL_API :ref:`dnnl_post_ops_len<doxid-group__dnnl__api__attributes_1ga98550f7eddff153ea819a6c4a68e7eec>`(:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops);

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` DNNL_API :ref:`dnnl_post_ops_get_kind<doxid-group__dnnl__api__attributes_1gabb9d82e4e8f1c83f169468d4b92f4109>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_append_sum<doxid-group__dnnl__api__attributes_1ga21a32731c8cf6e6034fd4f8704bd63db>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		float scale,
		int32_t zero_point,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_get_params_sum<doxid-group__dnnl__api__attributes_1ga029625f8a29d82a49ddb966428b6143e>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		float* scale,
		int32_t* zero_point,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* data_type
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_append_eltwise<doxid-group__dnnl__api__attributes_1gaf5927e8931bf113abb94837541cec662>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		float alpha,
		float beta
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_get_params_eltwise<doxid-group__dnnl__api__attributes_1gaedc7af352b0ae178c025e9272a428533>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`* alg_kind,
		float* alpha,
		float* beta
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_append_dw<doxid-group__dnnl__api__attributes_1ga38509493009271e2b8c6d8fadb1fcac1>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` weights_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` bias_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dst_data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` kernel_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` stride_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` padding_l_size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_get_params_dw<doxid-group__dnnl__api__attributes_1ga5e474604cf257e0dfae1ada352cf2f36>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* weights_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* bias_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* dst_data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* kernel_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* stride_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* padding_l_size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_append_binary<doxid-group__dnnl__api__attributes_1gabc40e53d80f6f1d61cc5b17807d2446c>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src1_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_get_params_binary<doxid-group__dnnl__api__attributes_1ga29acfcbce0ad42f36627469aa67b4046>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`* alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`* src1_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_append_prelu<doxid-group__dnnl__api__attributes_1ga833465b0aac349988b29245e1112656f>`(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		int mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_post_ops_get_params_prelu<doxid-group__dnnl__api__attributes_1ga5207e88213978239909da6e9f346cda7>`(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		int* mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_rnn_data_qparams<doxid-group__dnnl__api__attributes_1ga0067a4b6e5dd2fe7578cd4a25dddfe39>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		const float scale,
		const float shift
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_rnn_data_qparams<doxid-group__dnnl__api__attributes_1gae04744b95cdabcbcda1087229759be04>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		float* scale,
		float* shift
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_rnn_weights_qparams<doxid-group__dnnl__api__attributes_1ga815dbfe548cfcb70076fe091888e5466>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` count,
		int mask,
		const float* scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_rnn_weights_qparams<doxid-group__dnnl__api__attributes_1ga5bb88cfe42454f01884ddcdb906f6f7c>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* count,
		int* mask,
		const float** scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_rnn_weights_projection_qparams<doxid-group__dnnl__api__attributes_1gac7973cc7b4c62eb6766e9ac96c51d49d>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` count,
		int mask,
		const float* scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_get_rnn_weights_projection_qparams<doxid-group__dnnl__api__attributes_1gaa33206be6e7a0b7de2341041da75cc90>`(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* count,
		int* mask,
		const float** scales
		);

.. _details-group__dnnl__api__attributes:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A container for parameters that extend primitives behavior.

Attributes can also contain Post-ops, which are computations executed after the primitive.



.. rubric:: See also:

:ref:`Primitive Attributes <doxid-dev_guide_attributes>`

:ref:`Primitive Attributes: Post-ops <doxid-dev_guide_attributes_post_ops>`

Typedefs
--------

.. index:: pair: typedef; dnnl_primitive_attr_t
.. _doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* dnnl_primitive_attr_t

A primitive descriptor attributes handle that controls primitive behavior.

.. index:: pair: typedef; const_dnnl_primitive_attr_t
.. _doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* const_dnnl_primitive_attr_t

A constant primitive descriptor attributes handle.

.. index:: pair: typedef; dnnl_post_ops_t
.. _doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* dnnl_post_ops_t

A post operation chain handle.

.. index:: pair: typedef; const_dnnl_post_ops_t
.. _doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* const_dnnl_post_ops_t

A constant post operation chain handle.

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__attributes_1gaa30f540e1ed09b2865f153fd599c967b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` dnnl::convert_to_c(:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode)

Converts a scratchpad mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API scratchpad mode enum value.



.. rubric:: Returns:

Corresponding C API scratchpad mode enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__attributes_1gae13881206fecd43ce0e0daead7f0009e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` dnnl::convert_to_c(:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` akind)

Converts propagation kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API propagation kind enum value.



.. rubric:: Returns:

Corresponding C API propagation kind enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__attributes_1gad4c07d30e46391ce7ce0900d18cbfa30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` dnnl::convert_to_c(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm)

Converts algorithm kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aalgorithm

		- C++ API algorithm kind enum value.



.. rubric:: Returns:

Corresponding C API algorithm kind enum value.

.. index:: pair: function; dnnl_primitive_attr_create
.. _doxid-group__dnnl__api__attributes_1gaf630fdc0d8d0fd8522ec93852a559081:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_create(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`* attr)

Creates an empty (default) primitive attributes with all the parameters set to their default values.

Empty attributes are implied whenever the respective argument is NULL.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Output primitive attributes.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_clone
.. _doxid-group__dnnl__api__attributes_1gab6ac5a4b13fa1ab3251c51f3c750bd63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_clone(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`* attr,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` existing_attr
		)

Clones primitive attributes.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Output primitive attributes.

	*
		- existing_attr

		- Primitive attributes to clone.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_destroy
.. _doxid-group__dnnl__api__attributes_1ga96a7539382945195627f2932bff8fadb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_destroy(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr)

Destroys primitive attributes.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_fpmath_mode
.. _doxid-group__dnnl__api__attributes_1gac63b70ab1d2fe88c31f03c961b2e924a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_fpmath_mode(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode
		)

Returns the floating-point math mode primitive attribute.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Output FP math mode.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_fpmath_mode
.. _doxid-group__dnnl__api__attributes_1gafe55fa618bc10b65b6c0b6eca7e43840:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_fpmath_mode(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode
		)

Sets the floating-point math mode primitive attributes.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- FP math mode. The possible values are: :ref:`dnnl_fpmath_mode_strict <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3>` (default), :ref:`dnnl_fpmath_mode_bf16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7>`, :ref:`dnnl_fpmath_mode_f16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf>`, :ref:`dnnl_fpmath_mode_tf32 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc>`, :ref:`dnnl_fpmath_mode_any <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_fpmath_mode_v2
.. _doxid-group__dnnl__api__attributes_1gab6dd49fedbc548aea2d6ede5a0c42a6c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_fpmath_mode_v2(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode,
		int* apply_to_int
		)

Returns the floating-point math mode primitive attribute.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Output FP math mode.

	*
		- apply_to_int

		- Output use floating-point arithmetic for integer primitives.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_fpmath_mode_v2
.. _doxid-group__dnnl__api__attributes_1ga96edebcfaf7451fa96d698be110a18e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_fpmath_mode_v2(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode,
		int apply_to_int
		)

Sets the floating-point math mode primitive attributes.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- FP math mode. The possible values are: :ref:`dnnl_fpmath_mode_strict <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3>` (default), :ref:`dnnl_fpmath_mode_bf16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7>`, :ref:`dnnl_fpmath_mode_f16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf>`, :ref:`dnnl_fpmath_mode_tf32 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc>`, :ref:`dnnl_fpmath_mode_any <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c>`.

	*
		- apply_to_int

		- Boolean. Use of floating-point arithmetic for integer primitives.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_deterministic
.. _doxid-group__dnnl__api__attributes_1gacb11e4d0243975ef944eb25fffe2ef0a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_deterministic(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		int* value
		)

Returns the deterministic primitive attribute value.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- value

		- Output deterministic attribute value



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_deterministic
.. _doxid-group__dnnl__api__attributes_1ga69af4e29cba07fdb95672c070ac26511:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_deterministic(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int value
		)

Sets the deterministic primitive attribute value.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- value

		- Boolean value to set deterministic attribute.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_accumulation_mode
.. _doxid-group__dnnl__api__attributes_1ga1add29950cb3ec6595aebd572bcf7f92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_accumulation_mode(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>`* mode
		)

Returns the accumulation mode primitive attribute.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Output accumulation mode.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_accumulation_mode
.. _doxid-group__dnnl__api__attributes_1ga691c818641709d7dc94b92a2db7686b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_accumulation_mode(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>` mode
		)

Sets the accumulation mode primitive attribute.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Accumulation mode. The possible values are: :ref:`dnnl_accumulation_mode_strict <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafb83d4725f8c96479cd558a23cd60b6d>` (default), which is s32 for quantized primitives, f32/f64 otherwise :ref:`dnnl_accumulation_mode_relaxed <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fa9d1932f25fb8115758987627620d0c7d>`, which is same as strict but allows intermediate accumulators to be in src/dst datatype :ref:`dnnl_accumulation_mode_any <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fab49ca983cbe38a75a1ff0948c55f74bb>`, which allows accumulators to be src/dst datatype or any wider type. :ref:`dnnl_accumulation_mode_f32 <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7face0452131be499ecbd227f05f0c330ec>`, :ref:`dnnl_accumulation_mode_s32 <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fade88c19d4a39028a05d61501e88fe23d>`, :ref:`dnnl_accumulation_mode_f16 <doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafbe48b9827d45c4881477b102feef6a4>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_scratchpad_mode
.. _doxid-group__dnnl__api__attributes_1gab14d8e830a52510a75a917f75764a6b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_scratchpad_mode(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>`* mode
		)

Returns the primitive attributes scratchpad mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Output scratchpad mode.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_scratchpad_mode
.. _doxid-group__dnnl__api__attributes_1ga4adeb17e538392ec3a16d2f6ef3f7cca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_scratchpad_mode(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` mode
		)

Sets primitive attributes scratchpad mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- mode

		- Scratchpad mode. The possible values are: :ref:`dnnl_scratchpad_mode_library <doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4ac6aab09a2f8ef442a6a59800549b0487>` (default) and :ref:`dnnl_scratchpad_mode_user <doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4a7e9d97b9ceefc5e47512d83c097d6927>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_scales_mask
.. _doxid-group__dnnl__api__attributes_1gad7eac877f75cfa282be094b1e48cb71d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_scales_mask(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask
		)

Sets primitive attributes scaling factors for primitive operations for a given memory argument.

The scaling factors must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- arg

		- Parameter argument index as passed to the :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>` call.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the tensor dimensions and the ``scales`` array. The set i-th bit indicates that a dedicated scaling factor is used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_scales_mask <doxid-group__dnnl__api__attributes_1gad7eac877f75cfa282be094b1e48cb71d>`

.. index:: pair: function; dnnl_primitive_attr_set_scales
.. _doxid-group__dnnl__api__attributes_1ga6c56ea6104c9275574370a40e2d5b273:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_scales(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` group_dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		)

Sets primitive attributes scaling factors for primitive operations for a given memory argument.

The scaling factors must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- arg

		- Parameter argument index as passed to the :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>` call.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the tensor dimensions and the ``scales`` array. The set i-th bit indicates that a dedicated scaling factor is used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- ndims

		- Number of group dimensions.

	*
		- group_dims

		- Scaling factors correspondence groups that define the correspondence between the tensor dimensions and the scales array. The group dimensions should only be provided for each logical dimension that has correspondence mask ``mask`` set.

	*
		- data_type

		- Scaling factors data_type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_scales <doxid-group__dnnl__api__attributes_1ga6c56ea6104c9275574370a40e2d5b273>`

.. index:: pair: function; dnnl_primitive_attr_set_zero_points_mask
.. _doxid-group__dnnl__api__attributes_1ga24e429b5410f5657bc5bdda0a6c5d0a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_zero_points_mask(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask
		)

Sets primitive attributes zero points for primitive operations for a given memory argument.

The zero points must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_ZERO_POINTS <doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- arg

		- Parameter argument index as passed to the :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>` call.

	*
		- mask

		- Zero point correspondence mask that defines the correspondence between the tensor dimensions and the ``zero_points`` array. The set i-th bit indicates that a dedicated zero point is used for each index along that dimension. Set the mask to 0 to use a common zero point for the whole output tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_zero_points_mask <doxid-group__dnnl__api__attributes_1ga24e429b5410f5657bc5bdda0a6c5d0a7>`

.. index:: pair: function; dnnl_primitive_attr_set_zero_points
.. _doxid-group__dnnl__api__attributes_1ga94f3e4f640fb0ca210c2413bb5cf2255:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_zero_points(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` group_dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		)

Sets primitive attributes zero points for primitive operations for a given memory argument.

The zero points must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_ZERO_POINTS <doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- arg

		- Parameter argument index as passed to the :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>` call.

	*
		- mask

		- Zero point correspondence mask that defines the correspondence between the tensor dimensions and the ``zero_points`` array. The set i-th bit indicates that a dedicated zero point is used for each index along that dimension. Set the mask to 0 to use a common zero point for the whole output tensor.

	*
		- ndims

		- Number of group dimensions.

	*
		- group_dims

		- Zero point factors correspondence groups that define the correspondence between the tensor dimensions and the zero_points array. The group dimensions should be only provided for each logical dimension that has the bit set correspondence mask ``mask`` set.

	*
		- data_type

		- Zero points factors data_type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_zero_points <doxid-group__dnnl__api__attributes_1ga94f3e4f640fb0ca210c2413bb5cf2255>`

.. index:: pair: function; dnnl_primitive_attr_get_post_ops
.. _doxid-group__dnnl__api__attributes_1ga50c92661cc69e1eeb17b61f006320a05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_post_ops(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>`* post_ops
		)

Returns primitive attributes post-ops.

.. warning:: 

   The output ``post_ops`` points to the internal ``attr`` field, so it is an error to modify or destroy them. The lifetime of ``post_ops`` is the same as that of the ``attr`` it belongs to, so it is an error to use ``post_ops`` after ``attr`` has been destroyed.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- post_ops

		- Output post-ops.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_post_ops
.. _doxid-group__dnnl__api__attributes_1ga7045d42606599f156bfca69820c21ea2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_post_ops(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops
		)

Sets primitive attributes post-ops.

.. note:: 

   There is no way to check whether the post-ops would be supported by the target primitive. Any error will be reported by the dnnl\_<primitive name>\_[propagation kind]_primitive_desc_create() function call.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- post_ops

		- Post-ops to set.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_create
.. _doxid-group__dnnl__api__attributes_1gaa8d8c32ad4472de464e47336ad702a48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_create(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`* post_ops)

Creates empty post-ops sequence.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Output post-ops.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_clone
.. _doxid-group__dnnl__api__attributes_1ga087b5f530ae5cfd1134cfad694e84de1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_clone(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`* post_ops,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` existing_post_ops
		)

Clones post-ops primitive attribute.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Output post-ops primitive attribute.

	*
		- existing_post_ops

		- Post-ops primitive attribute to clone.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_destroy
.. _doxid-group__dnnl__api__attributes_1ga67487a65afa2e2066f4b4eb12d47535b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_destroy(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops)

Destroys post-ops.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_len
.. _doxid-group__dnnl__api__attributes_1ga98550f7eddff153ea819a6c4a68e7eec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int DNNL_API dnnl_post_ops_len(:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops)

Returns the length of post-ops.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.



.. rubric:: Returns:

The number of post-ops entries.

.. index:: pair: function; dnnl_post_ops_get_kind
.. _doxid-group__dnnl__api__attributes_1gabb9d82e4e8f1c83f169468d4b92f4109:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` DNNL_API dnnl_post_ops_get_kind(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index
		)

Returns the kind of a post-op entry.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Post-op entry index.



.. rubric:: Returns:

The kind of the post-op with the specified index.

:ref:`dnnl_undefined_primitive <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa89c48c6939d9f939b0f66d9b018a03b9>` if there is no post-op at the specified index.

.. index:: pair: function; dnnl_post_ops_append_sum
.. _doxid-group__dnnl__api__attributes_1ga21a32731c8cf6e6034fd4f8704bd63db:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_append_sum(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		float scale,
		int32_t zero_point,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type
		)

Appends an accumulation v3 (sum) to post-ops.

Prior to accumulating the result, a zero point is subtracted from the previous value and is multiplied by the scale.

The kind of this post-op is :ref:`dnnl_sum <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8ab0990125f2a743db86666c9d8b401b>`.

This feature may improve performance for cases like dequantize the asymmetrically quantized sum's src1 tensor to f32 domain before performing the sum operation by subtracting the ``zero_point`` before the scaling.

In the simplest case where accumulation is the only post-op, the computations will be:

.. code-block:: cpp

	dst[:] <- scale * (dst[:] - zero_point) + op(...)
	                                        // instead of dst[:] <- op(...)

If ``data_type`` is specified, original dst tensor will be reinterpreted as a tensor with provided data type. Since it is reinterpretation, data_type and dst data type should have the same size. As a result, computations will be:

.. code-block:: cpp

	dst[:] <- scale * (as_data_type(dst[:]) - zero_point) + op(...)
	                                   // instead of dst[:] <- op(...)


.. note:: 

   This post-op executes in-place and does not change the destination layout.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- scale

		- Accumulation scaling factor.

	*
		- zero_point

		- Single scalar int32_t value of zero point.

	*
		- data_type

		- Accumulation data_type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_get_params_sum
.. _doxid-group__dnnl__api__attributes_1ga029625f8a29d82a49ddb966428b6143e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_get_params_sum(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		float* scale,
		int32_t* zero_point,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* data_type
		)

Returns the parameters of an accumulation (sum) post-op with zero point and data type parameter.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Index of the sum post-op.

	*
		- scale

		- Output accumulation scaling factor.

	*
		- zero_point

		- Zero point.

	*
		- data_type

		- Data type for accumulation.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_append_eltwise
.. _doxid-group__dnnl__api__attributes_1gaf5927e8931bf113abb94837541cec662:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_append_eltwise(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		float alpha,
		float beta
		)

Appends an elementwise post-op.

The kind of this post operation is :ref:`dnnl_eltwise <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa9d7e709dd7e25d7ff11cf51c13fa2819>`.

In the simplest case when the elementwise is the only post operation, the computations would be:

.. code-block:: cpp

	dst[:] <- eltwise_op (op(...)) // instead of dst[:] <- op(...)

where eltwise_op is configured with the given parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- alg_kind

		- Elementwise algorithm for the post-op.

	*
		- alpha

		- Alpha parameter for the elementwise algorithm.

	*
		- beta

		- Beta parameter for the elementwise algorithm.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_get_params_eltwise
.. _doxid-group__dnnl__api__attributes_1gaedc7af352b0ae178c025e9272a428533:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_get_params_eltwise(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`* alg_kind,
		float* alpha,
		float* beta
		)

Returns the parameters of an elementwise post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Index of the elementwise post-op.

	*
		- alg_kind

		- Output elementwise algorithm kind.

	*
		- alpha

		- Output alpha parameter for the elementwise algorithm.

	*
		- beta

		- Output beta parameter for the elementwise algorithm.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if ``index`` does not refer to an elementwise post-op.

.. index:: pair: function; dnnl_post_ops_append_dw
.. _doxid-group__dnnl__api__attributes_1ga38509493009271e2b8c6d8fadb1fcac1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_append_dw(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` weights_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` bias_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dst_data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` kernel_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` stride_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` padding_l_size
		)

Appends a depthwise post-op convolution.

This post-op can only be fused with a 2D 1x1 convolution (convolution with weights spatial dimensions equal to 1 i.e., kh=kw=1).

The kind of this post-op is :ref:`dnnl_convolution <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa402cfeaa257524d301bb73e770bc87f6>`.

The number of outputs for primitive with fusion is one. The output spatial size can be derived as below:

output_height = ceil(output_height_1x1_convolution, stride) output_width = ceil(output_width_1x1_convolution, stride)

See :ref:`dev_guide_attributes_post_ops_depthwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise>` and :ref:`dev_guide_attributes_post_ops_depthwise_fusion <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise_fusion>` for more info.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- weights_data_type

		- Weights data type of depthwise post-op

	*
		- bias_data_type

		- Bias data type of depthwise post-op

	*
		- dst_data_type

		- Output data type of depthwise post-op

	*
		- kernel_size

		- Size of kernel of depthwise post-op

	*
		- stride_size

		- Size of stride of depthwise post-op

	*
		- padding_l_size

		- Size of left and top paddings of depthwise post-op



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise

.. index:: pair: function; dnnl_post_ops_get_params_dw
.. _doxid-group__dnnl__api__attributes_1ga5e474604cf257e0dfae1ada352cf2f36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_get_params_dw(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* weights_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* bias_data_type,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`* dst_data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* kernel_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* stride_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* padding_l_size
		)

Returns the parameters of an depthwise post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Index of the elementwise post-op.

	*
		- weights_data_type

		- Weights data type of depthwise post-op

	*
		- bias_data_type

		- Bias data type of depthwise post-op

	*
		- dst_data_type

		- Output data type of depthwise post-op

	*
		- kernel_size

		- Size of kernel of depthwise post-op

	*
		- stride_size

		- Size of stride of depthwise post-op

	*
		- padding_l_size

		- Size of left and top paddings of depthwise post-op



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise

.. index:: pair: function; dnnl_post_ops_append_binary
.. _doxid-group__dnnl__api__attributes_1gabc40e53d80f6f1d61cc5b17807d2446c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_append_binary(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src1_desc
		)

Appends a binary post-op.

The kind of this post operation is :ref:`dnnl_binary <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa1d51705e2642ce2ce19a3e163bb25f93>`.

In the simplest case when the binary is the only post operation, the computations would be:

.. code-block:: cpp

	dst[:] <- binary_op (dst[:], another_input[:])

where binary_op is configured with the given parameters. binary_op supports broadcast semantics for a second operand.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- alg_kind

		- Binary algorithm for the post-op.

	*
		- src1_desc

		- Memory descriptor of a second operand.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_get_params_binary
.. _doxid-group__dnnl__api__attributes_1ga29acfcbce0ad42f36627469aa67b4046:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_get_params_binary(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`* alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`* src1_desc
		)

Returns the parameters of a binary post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Index of the binary post-op.

	*
		- alg_kind

		- Output binary algorithm kind.

	*
		- src1_desc

		- Output memory descriptor of a second operand.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if ``index`` does not refer to a binary post-op.

.. index:: pair: function; dnnl_post_ops_append_prelu
.. _doxid-group__dnnl__api__attributes_1ga833465b0aac349988b29245e1112656f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_append_prelu(
		:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops,
		int mask
		)

Appends a prelu forward post-op.

The kind of this post-op is :ref:`dnnl::primitive::kind::prelu <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a837c39f77d473b24eb27c0758d5c7c1b>`.

The post-op can be defined as:

.. code-block:: cpp

	dst[:] <- prelu(dst[:], weights[:])
	prelu:
	dst[:] <- dst[:] if dst[:] > 0
	dst[:] <- dst[:] * weights[:] if dst[:] <= 0

.. note:: 

   The order of dimensions does not depend on how elements are laid out in memory. For example:
   
   * for a 2D CNN activations tensor the order is always (n, c)
   
   * for a 4D CNN activations tensor the order is always (n, c, h, w)
   
   * for a 5D CNN weights tensor the order is always (g, oc, ic, kh, kw)
   
   
Prelu weights tensor is passed in runtime execution phase. Prelu weights tensor data type is implicitly assumed as f32 using plain layout (a, ab, acb, acdb, acdeb)



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- mask

		- Defines the correspondence between the output tensor dimensions and the prelu weights tensor. The set i-th bit indicates that a dedicated weights value is used for each index along that dimension. Set the mask to 0 to use a common weights value for the whole output tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_post_ops_get_params_prelu
.. _doxid-group__dnnl__api__attributes_1ga5207e88213978239909da6e9f346cda7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_post_ops_get_params_prelu(
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops,
		int index,
		int* mask
		)

Returns the parameters of a prelu post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops

		- Post-ops.

	*
		- index

		- Index of the prelu post-op.

	*
		- mask

		- Mask of the prelu post-op.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_rnn_data_qparams
.. _doxid-group__dnnl__api__attributes_1ga0067a4b6e5dd2fe7578cd4a25dddfe39:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_rnn_data_qparams(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		const float scale,
		const float shift
		)

Set quantization scale and shift parameters for RNN data tensors.

For performance reasons, the low-precision configuration of the RNN primitives expects input activations to have the unsigned 8-bit integer data type. The scale and shift parameters are used to quantize floating-point data to unsigned integer and must be passed to the RNN primitive using attributes.

The quantization formula is ``scale * data + shift``.

.. note:: 

   Quantization scale and shift are common for src_layer, src_iter, dst_iter, and dst_layer.
   
   
Example usage:

.. ref-code-block:: cpp

	// RNN parameters
	int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
	// Activations quantization parameters
	float scale = 63.f, shift = 64.f;
	
	dnnl_primitive_attr_t rnn_attr;
	// Create default attributes
	dnnl_primitive_attr_create(&rnn_attr);
	
	// Set scale and shift for int8 quantization of activation
	dnnl_primitive_attr_set_rnn_data_qparams(rnn_attr, scale, shift);
	
	// Create an RNN primitive descriptor.
	dnnl_primitive_desc_t rnn_pd;
	dnnl_vanilla_rnn_forward_primitive_desc_create(&rnn_pd,
	        engine, /* arguments */, attr);



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- scale

		- The value to scale the data by.

	*
		- shift

		- The value to shift the data by.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_rnn_data_qparams
.. _doxid-group__dnnl__api__attributes_1gae04744b95cdabcbcda1087229759be04:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_rnn_data_qparams(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		float* scale,
		float* shift
		)

Returns the quantization scale and shift parameters for RNN data tensors.

.. note:: 

   Quantization scale and shift are common for src_layer, src_iter, dst_iter, and dst_layer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- scale

		- The value to scale the data by.

	*
		- shift

		- The value to shift the data by.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_rnn_weights_qparams
.. _doxid-group__dnnl__api__attributes_1ga815dbfe548cfcb70076fe091888e5466:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_rnn_weights_qparams(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` count,
		int mask,
		const float* scales
		)

Sets quantization scaling factors for RNN weights tensors.

The low-precision configuration of the RNN primitives expects input weights to use the signed 8-bit integer data type. The scaling factors are used to quantize floating-point data to signed integer and must be passed to RNN primitives using attributes.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.
   
   

.. note:: 

   Quantization scales are common for weights_layer and weights_iteration



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- count

		- Number of elements in the ``scales`` array.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- 
		  Array of output scaling factors that must contain ``count`` values and the following equality must hold:
		  
		  .. math::
		  
		  	count = \prod\limits_{d \in mask} weights.dims[d].
		  
		  Violations can only be detected when the attributes are used to create a primitive descriptor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_rnn_weights_qparams
.. _doxid-group__dnnl__api__attributes_1ga5bb88cfe42454f01884ddcdb906f6f7c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_rnn_weights_qparams(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* count,
		int* mask,
		const float** scales
		)

Returns the quantization scaling factors for RNN weights tensors.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- count

		- Number of elements in the ``scales`` array.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- 
		  Array of output scaling factors that contain ``count`` values and the following equality must hold:
		  
		  .. math::
		  
		  	count = \prod\limits_{d \in mask} weights.dims[d].



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_set_rnn_weights_projection_qparams
.. _doxid-group__dnnl__api__attributes_1gac7973cc7b4c62eb6766e9ac96c51d49d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_set_rnn_weights_projection_qparams(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` count,
		int mask,
		const float* scales
		)

Sets quantization scaling factors for RNN projection weights tensors.

The low-precision configuration of the RNN primitives expects input weights to use the signed 8-bit integer data type. The scaling factors are used to quantize floating-point data to signed integer and must be passed to RNN primitives using attributes.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- count

		- Number of elements in the ``scales`` array.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- 
		  Array of output scaling factors that must contain ``count`` values and the following equality must hold:
		  
		  .. math::
		  
		  	count = \prod\limits_{d \in mask} weights.dims[d].
		  
		  Violations can only be detected when the attributes are used to create a primitive descriptor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_attr_get_rnn_weights_projection_qparams
.. _doxid-group__dnnl__api__attributes_1gaa33206be6e7a0b7de2341041da75cc90:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_attr_get_rnn_weights_projection_qparams(
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* count,
		int* mask,
		const float** scales
		)

Returns the quantization scaling factors for RNN projection weights tensors.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- Primitive attributes.

	*
		- count

		- Number of elements in the ``scales`` array.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- 
		  Array of output scaling factors that contain ``count`` values and the following equality must hold:
		  
		  .. math::
		  
		  	count = \prod\limits_{d \in mask} weights.dims[d].



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

