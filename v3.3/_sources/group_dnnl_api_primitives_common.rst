.. index:: pair: group; Common
.. _doxid-group__dnnl__api__primitives__common:

Common
======

.. toctree::
	:hidden:

	enum_dnnl_alg_kind_t.rst
	enum_dnnl_normalization_flags_t.rst
	enum_dnnl_primitive_kind_t.rst
	enum_dnnl_prop_kind_t.rst
	enum_dnnl_query_t.rst
	enum_dnnl_normalization_flags.rst
	enum_dnnl_query.rst
	struct_dnnl_exec_arg_t.rst
	struct_dnnl_primitive.rst
	struct_dnnl_primitive_desc.rst
	struct_dnnl_primitive-2.rst
	struct_dnnl_primitive_desc-2.rst
	struct_dnnl_primitive_desc_base.rst

Overview
~~~~~~~~

Common operations to create, destroy and inspect primitives. :ref:`More...<details-group__dnnl__api__primitives__common>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* :ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`;
	typedef const struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* :ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>`;
	typedef struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* :ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`;
	typedef const struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* :ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>`;

	// enums

	enum :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`;
	enum :ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>`;
	enum :ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>`;
	enum :ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`;
	enum :ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>`;
	enum :ref:`dnnl::normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`;
	enum :ref:`dnnl::query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>`;

	// structs

	struct :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`;
	struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`;
	struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`;
	struct :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`;
	struct :ref:`dnnl::primitive_desc<doxid-structdnnl_1_1primitive__desc>`;
	struct :ref:`dnnl::primitive_desc_base<doxid-structdnnl_1_1primitive__desc__base>`;

	// global functions

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__primitives__common_1gaaa215c424a2a5c5f734600216dfb8873>`(:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` akind);
	:ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__primitives__common_1gae3d2ea872c5ab424c74d7549d2222926>`(:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags);
	:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__primitives__common_1ga01d8a1881875cdb94e230db4e53ccb97>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` aquery);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_desc_next_impl<doxid-group__dnnl__api__primitives__common_1ga8fc906c6f9b705d747e034097b74965c>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` primitive_desc);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_desc_clone<doxid-group__dnnl__api__primitives__common_1gae40abecf7360106805eabc049cc86e4b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` existing_primitive_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_desc_get_attr<doxid-group__dnnl__api__primitives__common_1ga47e492dff0bba4376b8e9f30522c6207>`(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>`* attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_desc_destroy<doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` primitive_desc);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_desc_query<doxid-group__dnnl__api__primitives__common_1ga041881114858228279174aff5c1f5e75>`(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index,
		void* result
		);

	:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` DNNL_API :ref:`dnnl_primitive_desc_query_md<doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index
		);

	int DNNL_API :ref:`dnnl_primitive_desc_query_s32<doxid-group__dnnl__api__primitives__common_1ga314bfec9b68ad50e76ac4c87816cc3aa>`(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_create<doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_create_from_cache_blob<doxid-group__dnnl__api__primitives__common_1gaeee0deb9aa704e3b7c58291c2a3d022b>`(
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		size_t size,
		const uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_execute<doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_get_primitive_desc<doxid-group__dnnl__api__primitives__common_1ga8324e883e41c0b1b9b95bdb7718d35f9>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>`* primitive_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_get_cache_blob<doxid-group__dnnl__api__primitives__common_1gafceed39f28cd3bec5f530317a2a88719>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		size_t* size,
		uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_destroy<doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>` primitive);

	// macros

	#define :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP<doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(idx)
	#define :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE<doxid-group__dnnl__api__primitives__common_1gadec6bcdebcd35d41c394622af3e0cb75>`
	#define :ref:`DNNL_ARG_ATTR_OUTPUT_SCALES<doxid-group__dnnl__api__primitives__common_1ga0afb48b0c2b8f3ee30609aaa47aa29db>`
	#define :ref:`DNNL_ARG_ATTR_POST_OP_DW<doxid-group__dnnl__api__primitives__common_1ga47534804c9b2f9ede6b875f6cb08cc35>`
	#define :ref:`DNNL_ARG_ATTR_SCALES<doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>`
	#define :ref:`DNNL_ARG_ATTR_ZERO_POINTS<doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6>`
	#define :ref:`DNNL_ARG_AUGRU_ATTENTION<doxid-group__dnnl__api__primitives__common_1ga635c81b2547d4291a82d53a70b6aa8d6>`
	#define :ref:`DNNL_ARG_BIAS<doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`
	#define :ref:`DNNL_ARG_DIFF_AUGRU_ATTENTION<doxid-group__dnnl__api__primitives__common_1ga04f2d46dcd9b18f2f2d240d3efbe2e74>`
	#define :ref:`DNNL_ARG_DIFF_BIAS<doxid-group__dnnl__api__primitives__common_1ga1cd79979dda6df65ec45eef32a839901>`
	#define :ref:`DNNL_ARG_DIFF_DST<doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`
	#define :ref:`DNNL_ARG_DIFF_DST_0<doxid-group__dnnl__api__primitives__common_1ga5cf662f6dc742a3500ec4f54290b86da>`
	#define :ref:`DNNL_ARG_DIFF_DST_1<doxid-group__dnnl__api__primitives__common_1ga87fad7b00d52b6bff673fca344d0f96d>`
	#define :ref:`DNNL_ARG_DIFF_DST_2<doxid-group__dnnl__api__primitives__common_1gaf6ae542064ad032e7c335f3701398034>`
	#define :ref:`DNNL_ARG_DIFF_DST_ITER<doxid-group__dnnl__api__primitives__common_1gad9c83f558d1b229b4185ccbf939590a3>`
	#define :ref:`DNNL_ARG_DIFF_DST_ITER_C<doxid-group__dnnl__api__primitives__common_1ga5524b26b690b9b4b81f0c7f3f9ac3b62>`
	#define :ref:`DNNL_ARG_DIFF_DST_LAYER<doxid-group__dnnl__api__primitives__common_1gafc6053e276352b05b3b526141586e0ac>`
	#define :ref:`DNNL_ARG_DIFF_SCALE<doxid-group__dnnl__api__primitives__common_1gaeb524b0816a1748026a60195523d3393>`
	#define :ref:`DNNL_ARG_DIFF_SHIFT<doxid-group__dnnl__api__primitives__common_1gaba70712528e1e3922b1b9e3072915df6>`
	#define :ref:`DNNL_ARG_DIFF_SRC<doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`
	#define :ref:`DNNL_ARG_DIFF_SRC_0<doxid-group__dnnl__api__primitives__common_1ga5d7ebcd603a037e0cd53a22377d1addf>`
	#define :ref:`DNNL_ARG_DIFF_SRC_1<doxid-group__dnnl__api__primitives__common_1ga79861408c787de29acec450006dace7f>`
	#define :ref:`DNNL_ARG_DIFF_SRC_2<doxid-group__dnnl__api__primitives__common_1gae5520b100e7e94e218ccda5697d720bf>`
	#define :ref:`DNNL_ARG_DIFF_SRC_3<doxid-group__dnnl__api__primitives__common_1gad0ef2743a16142c7818726b4c970ac07>`
	#define :ref:`DNNL_ARG_DIFF_SRC_ITER<doxid-group__dnnl__api__primitives__common_1ga4f7ed97882e020a1cbaa891bbe0da45b>`
	#define :ref:`DNNL_ARG_DIFF_SRC_ITER_C<doxid-group__dnnl__api__primitives__common_1ga1d8616925684111f3a1b6d8116ab0077>`
	#define :ref:`DNNL_ARG_DIFF_SRC_LAYER<doxid-group__dnnl__api__primitives__common_1ga24709fa44c67cf453facbc1c52b0d598>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS<doxid-group__dnnl__api__primitives__common_1ga3324092ef421f77aebee83b0117cac60>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_0<doxid-group__dnnl__api__primitives__common_1gaece55926acecbbf375cd4fb06c5786ab>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_1<doxid-group__dnnl__api__primitives__common_1ga789a5f68ece3af7f3c86217a4a5dab7e>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_2<doxid-group__dnnl__api__primitives__common_1ga4b4fca234fa86cd9270fb3ef7cf9446d>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_3<doxid-group__dnnl__api__primitives__common_1ga98277fe5c9232f3089b8a9b5841e821c>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_ITER<doxid-group__dnnl__api__primitives__common_1ga4a8e5f32de3856588b2976a766d0af0f>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_LAYER<doxid-group__dnnl__api__primitives__common_1gac0bd0c223011ee2fbbc3c430c047c756>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE<doxid-group__dnnl__api__primitives__common_1ga24d0c96468ef0e28a96f8a6f1b2869b3>`
	#define :ref:`DNNL_ARG_DIFF_WEIGHTS_PROJECTION<doxid-group__dnnl__api__primitives__common_1ga3e29edb164be584b8f2df0b4ed6256fb>`
	#define :ref:`DNNL_ARG_DST<doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`
	#define :ref:`DNNL_ARG_DST_0<doxid-group__dnnl__api__primitives__common_1ga7ba773575a68b592f3ef0fa885d87c20>`
	#define :ref:`DNNL_ARG_DST_1<doxid-group__dnnl__api__primitives__common_1gae4215f24ddcf9e4b20b246e10a992687>`
	#define :ref:`DNNL_ARG_DST_2<doxid-group__dnnl__api__primitives__common_1gabec2200437d5553747fe56a15548a9b1>`
	#define :ref:`DNNL_ARG_DST_ITER<doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6>`
	#define :ref:`DNNL_ARG_DST_ITER_C<doxid-group__dnnl__api__primitives__common_1ga8b77d8716fc0ab9923d6cb409dbdf900>`
	#define :ref:`DNNL_ARG_DST_LAYER<doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`
	#define :ref:`DNNL_ARG_FROM<doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`
	#define :ref:`DNNL_ARG_MEAN<doxid-group__dnnl__api__primitives__common_1ga9bcff7f442a5d6a0ac1183533e721066>`
	#define :ref:`DNNL_ARG_MULTIPLE_DST<doxid-group__dnnl__api__primitives__common_1gaa4183403de9fac3efd3f32c7a975580b>`
	#define :ref:`DNNL_ARG_MULTIPLE_SRC<doxid-group__dnnl__api__primitives__common_1ga1f0da423df3fb6853ddcbe6ffe964267>`
	#define :ref:`DNNL_ARG_SCALE<doxid-group__dnnl__api__primitives__common_1ga3c5cac668bc82c90c8da051c7d430370>`
	#define :ref:`DNNL_ARG_SCRATCHPAD<doxid-group__dnnl__api__primitives__common_1ga81836a4db2cb1c4a14d959e304d3f63d>`
	#define :ref:`DNNL_ARG_SHIFT<doxid-group__dnnl__api__primitives__common_1gac250777ced72098caf39deae1d9039c8>`
	#define :ref:`DNNL_ARG_SRC<doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`
	#define :ref:`DNNL_ARG_SRC_0<doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`
	#define :ref:`DNNL_ARG_SRC_1<doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`
	#define :ref:`DNNL_ARG_SRC_2<doxid-group__dnnl__api__primitives__common_1ga2ad44d7072cc1c13f0d2eeb3f5f59a24>`
	#define :ref:`DNNL_ARG_SRC_3<doxid-group__dnnl__api__primitives__common_1ga3cf129098380c0ee634875c4d65b2c43>`
	#define :ref:`DNNL_ARG_SRC_ITER<doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143>`
	#define :ref:`DNNL_ARG_SRC_ITER_C<doxid-group__dnnl__api__primitives__common_1ga8ef6969516e717208a33766542410410>`
	#define :ref:`DNNL_ARG_SRC_LAYER<doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`
	#define :ref:`DNNL_ARG_TO<doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`
	#define :ref:`DNNL_ARG_VARIANCE<doxid-group__dnnl__api__primitives__common_1gaa0e60e8d129936ba29555e17efb82581>`
	#define :ref:`DNNL_ARG_WEIGHTS<doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`
	#define :ref:`DNNL_ARG_WEIGHTS_0<doxid-group__dnnl__api__primitives__common_1ga12d6c5c4ea17c90ade1e324d47756c3f>`
	#define :ref:`DNNL_ARG_WEIGHTS_1<doxid-group__dnnl__api__primitives__common_1ga6daaa6a32731e2a670821f95f8e353fd>`
	#define :ref:`DNNL_ARG_WEIGHTS_2<doxid-group__dnnl__api__primitives__common_1gab5744466811fbf73e69f47100a5276e6>`
	#define :ref:`DNNL_ARG_WEIGHTS_3<doxid-group__dnnl__api__primitives__common_1gab6c802a32495f116c5918a987dc6a6ed>`
	#define :ref:`DNNL_ARG_WEIGHTS_ITER<doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`
	#define :ref:`DNNL_ARG_WEIGHTS_LAYER<doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`
	#define :ref:`DNNL_ARG_WEIGHTS_PEEPHOLE<doxid-group__dnnl__api__primitives__common_1gab1638cabf7e6c1c3e0d6a1f470707e08>`
	#define :ref:`DNNL_ARG_WEIGHTS_PROJECTION<doxid-group__dnnl__api__primitives__common_1ga28e553bafbca27e9c87020adc8863340>`
	#define :ref:`DNNL_ARG_WORKSPACE<doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`

.. _details-group__dnnl__api__primitives__common:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Common operations to create, destroy and inspect primitives.

Typedefs
--------

.. index:: pair: typedef; dnnl_primitive_desc_t
.. _doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* dnnl_primitive_desc_t

A primitive descriptor handle.

.. index:: pair: typedef; const_dnnl_primitive_desc_t
.. _doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* const_dnnl_primitive_desc_t

A constant primitive descriptor handle.

.. index:: pair: typedef; dnnl_primitive_t
.. _doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* dnnl_primitive_t

A primitive handle.

.. index:: pair: typedef; const_dnnl_primitive_t
.. _doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* const_dnnl_primitive_t

A constant primitive handle.

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__primitives__common_1gaaa215c424a2a5c5f734600216dfb8873:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` dnnl::convert_to_c(:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` akind)

Converts primitive kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API primitive kind enum value.



.. rubric:: Returns:

Corresponding C API primitive kind enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__primitives__common_1gae3d2ea872c5ab424c74d7549d2222926:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` dnnl::convert_to_c(:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags)

Converts normalization flags enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flags

		- C++ API normalization flags enum value.



.. rubric:: Returns:

Corresponding C API normalization flags enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__primitives__common_1ga01d8a1881875cdb94e230db4e53ccb97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` dnnl::convert_to_c(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` aquery)

Converts query enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aquery

		- C++ API query enum value.



.. rubric:: Returns:

Corresponding C API query enum value.

.. index:: pair: function; dnnl_primitive_desc_next_impl
.. _doxid-group__dnnl__api__primitives__common_1ga8fc906c6f9b705d747e034097b74965c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_desc_next_impl(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` primitive_desc)

Changes the primitive descriptor to point to the next available implementation.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- A primitive descriptor to change.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

:ref:`dnnl_last_impl_reached <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa2c6653ea2885f9dbafdc0bf2ee8693f8>` if no more implementations available, in which case the primitive descriptor itself is kept unchanged.

.. index:: pair: function; dnnl_primitive_desc_clone
.. _doxid-group__dnnl__api__primitives__common_1gae40abecf7360106805eabc049cc86e4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_desc_clone(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` existing_primitive_desc
		)

Clones a primitive descriptor.

The resulting primitive descriptor must be destroyed separately.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- existing_primitive_desc

		- Primitive descriptor to clone.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_desc_get_attr
.. _doxid-group__dnnl__api__primitives__common_1ga47e492dff0bba4376b8e9f30522c6207:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_desc_get_attr(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>`* attr
		)

Returns a constant reference to the attributes of a primitive descriptor.

.. warning:: 

   It is an error to destroy the resulting ``attr``.
   
   

.. warning:: 

   The lifetime of an ``attr`` is the same as that of a ``primitive_desc``, so it is an error to use the ``attr`` once the ``primitive_desc`` has been destroyed.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Primitive descriptor.

	*
		- attr

		- Output primitive attributes.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_desc_destroy
.. _doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_desc_destroy(:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>` primitive_desc)

Destroys a primitive descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Primitive descriptor to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_desc_query
.. _doxid-group__dnnl__api__primitives__common_1ga041881114858228279174aff5c1f5e75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_desc_query(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index,
		void* result
		)

Queries a primitive descriptor for various pieces of information.

The most common use case is to query a primitive descriptor, created with source, weights, and destination memory descriptors with format tags set to :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`, for the corresponding memory descriptors (in this case the ``what`` is set to :ref:`dnnl_query_src_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5>`, :ref:`dnnl_query_weights_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a12ea0b4858b84889acab34e498323355>`, and :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>` respectively) so that it is possible to create memory objects and reorder primitives if necessary.

Another typical use case is to query a primitive descriptor for workspace memory descriptor (with ``what`` set to :ref:`dnnl_query_workspace_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`). If this query returns :ref:`dnnl_not_required <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaff3988320148106126bce50dd76d6a97>` status, then workspace memory is not required.

.. note:: 

   When querying for a memory descriptor for a scratchpad, a workspace, or an optional parameter, the query will return a pointer to a zero memory descriptor if the parameter is not needed.
   
   
A few other use cases:

* query a primitive descriptor for the implementation information string (:ref:`dnnl_query_impl_info_str <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5a44980a7317e63cc7b6877d15a549aa>`)

* query a primitive descriptor for the number of inputs and outputs (:ref:`dnnl_query_num_of_inputs_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1fe7a52f5934c92b0bd0330463549c0e>` and :ref:`dnnl_query_num_of_outputs_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a06833f7d865faf3eaaad3b71976ba16a>` respectively)



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Primitive descriptor.

	*
		- what

		- Parameter to query.

	*
		- index

		- Index of the parameter to query for.

	*
		- result

		- Output result. The type depends on the query. For example, it must be a ``dnnl_memory_desc_t*`` if querying for a memory descriptor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_query_t <doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` for more options

.. index:: pair: function; dnnl_primitive_desc_query_md
.. _doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` DNNL_API dnnl_primitive_desc_query_md(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index
		)

Queries primitive descriptor for a memory descriptor.

.. note:: 

   This function is a convenience version of :ref:`dnnl_primitive_desc_query() <doxid-group__dnnl__api__primitives__common_1ga041881114858228279174aff5c1f5e75>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Primitive descriptor.

	*
		- what

		- Kind of memory descriptor parameter to query for.

	*
		- index

		- Index of the parameter to query.



.. rubric:: Returns:

A pointer to the requested memory descriptor.

A pointer to a zero memory descriptor if the parameter is not needed.

NULL in case of any error.

.. index:: pair: function; dnnl_primitive_desc_query_s32
.. _doxid-group__dnnl__api__primitives__common_1ga314bfec9b68ad50e76ac4c87816cc3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int DNNL_API dnnl_primitive_desc_query_s32(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index
		)

Queries primitive descriptor for a signed 32bit int.

.. note:: 

   This function is a convenience version of :ref:`dnnl_primitive_desc_query() <doxid-group__dnnl__api__primitives__common_1ga041881114858228279174aff5c1f5e75>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Primitive descriptor.

	*
		- what

		- Kind of the value to query for.

	*
		- index

		- Index of the parameter to query.



.. rubric:: Returns:

The requested value.

0 in case of any error (in particular if the queried entity is not of type int32_t). Note that 0 may also be the actual returned value.

.. index:: pair: function; dnnl_primitive_create
.. _doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_create(
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc
		)

Creates a primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Output primitive.

	*
		- primitive_desc

		- Primitive descriptor used to create the primitive.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_create_from_cache_blob
.. _doxid-group__dnnl__api__primitives__common_1gaeee0deb9aa704e3b7c58291c2a3d022b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_create_from_cache_blob(
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` primitive_desc,
		size_t size,
		const uint8_t* cache_blob
		)

Creates a primitive from a cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Output primitive.

	*
		- primitive_desc

		- Primitive descriptor used to create the primitive.

	*
		- size

		- Size of the cache blob in bytes.

	*
		- cache_blob

		- Cache blob of size ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_execute
.. _doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_execute(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args
		)

Executes a primitive.

.. note:: 

   If any argument in ``args`` is padded (padded_dims > dims), the primitive execution will assume properly zero-padded input arguments, and produce zero-padded output arguments.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Primitive to execute.

	*
		- stream

		- Stream to use.

	*
		- nargs

		- Number of arguments.

	*
		- args

		- Array of arguments. Each argument is an <index, :ref:`dnnl_memory_t <doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`> pair. The index is one of the ``DNNL_ARG_*`` values such as ``DNNL_ARG_SRC``. Unless runtime shapes are used (see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`), the memory object must have the same memory descriptor as that returned by :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>` (:ref:`dnnl_query_exec_arg_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ac7ecf09260d89d54ddd7f35c51a244da>`, index).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_get_primitive_desc
.. _doxid-group__dnnl__api__primitives__common_1ga8324e883e41c0b1b9b95bdb7718d35f9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_get_primitive_desc(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>`* primitive_desc
		)

Retrieves a constant reference to the primitive descriptor of a given primitive.

.. warning:: 

   It is an error to destroy the returned object. It is owned by the primitive. The ``const`` qualifier of the returned object prevents such attempts.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Primitive to query for the primitive descriptor.

	*
		- primitive_desc

		- Output primitive descriptor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_get_cache_blob
.. _doxid-group__dnnl__api__primitives__common_1gafceed39f28cd3bec5f530317a2a88719:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_get_cache_blob(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		size_t* size,
		uint8_t* cache_blob
		)

Retrieves a cache blob associated with the given primitive.

.. note:: 

   The cache blob can be empty. It's the user's responsibility to check whether it's empty prior to passing it to :ref:`dnnl_primitive_create_from_cache_blob() <doxid-group__dnnl__api__primitives__common_1gaeee0deb9aa704e3b7c58291c2a3d022b>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Primitive to query for the cache blob.

	*
		- size

		- Size of the cache blob in bytes.

	*
		- cache_blob

		- Cache blob of size ``size``. If the ``cache_blob`` is nullptr then the size of the cache blob is returned in ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_primitive_destroy
.. _doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_primitive_destroy(:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>` primitive)

Destroys a primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- The primitive to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

Macros
------

.. index:: pair: define; DNNL_ARG_ATTR_MULTIPLE_POST_OP
.. _doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)

Arguments for a binary post operation.

Up to 32 arguments are supported. See :ref:`dev_guide_attributes_post_ops_binary_fusion <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary_fusion>`

.. index:: pair: define; DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE
.. _doxid-group__dnnl__api__primitives__common_1gadec6bcdebcd35d41c394622af3e0cb75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE

Starting point for a binary post operation.

.. index:: pair: define; DNNL_ARG_ATTR_OUTPUT_SCALES
.. _doxid-group__dnnl__api__primitives__common_1ga0afb48b0c2b8f3ee30609aaa47aa29db:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_OUTPUT_SCALES

Output scaling factors provided at execution time.

.. index:: pair: define; DNNL_ARG_ATTR_POST_OP_DW
.. _doxid-group__dnnl__api__primitives__common_1ga47534804c9b2f9ede6b875f6cb08cc35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_POST_OP_DW

Arguments for fused depthwise convolution.

See :ref:`dev_guide_attributes_post_ops_depthwise_fusion <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise_fusion>`

.. index:: pair: define; DNNL_ARG_ATTR_SCALES
.. _doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_SCALES

Scaling factors provided at execution time.

.. index:: pair: define; DNNL_ARG_ATTR_ZERO_POINTS
.. _doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_ATTR_ZERO_POINTS

Zero points provided at execution time.

.. index:: pair: define; DNNL_ARG_AUGRU_ATTENTION
.. _doxid-group__dnnl__api__primitives__common_1ga635c81b2547d4291a82d53a70b6aa8d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_AUGRU_ATTENTION

A special mnemonic for RNN input recurrent cell attention vector.

An alias for :ref:`DNNL_ARG_SRC_3 <doxid-group__dnnl__api__primitives__common_1ga3cf129098380c0ee634875c4d65b2c43>`.

.. index:: pair: define; DNNL_ARG_BIAS
.. _doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_BIAS

Bias tensor argument.

.. index:: pair: define; DNNL_ARG_DIFF_AUGRU_ATTENTION
.. _doxid-group__dnnl__api__primitives__common_1ga04f2d46dcd9b18f2f2d240d3efbe2e74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_AUGRU_ATTENTION

A special mnemonic for gradient (diff) of RNN input recurrent cell attention vector.

An alias for :ref:`DNNL_ARG_DIFF_SRC_3 <doxid-group__dnnl__api__primitives__common_1gad0ef2743a16142c7818726b4c970ac07>`.

.. index:: pair: define; DNNL_ARG_DIFF_BIAS
.. _doxid-group__dnnl__api__primitives__common_1ga1cd79979dda6df65ec45eef32a839901:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_BIAS

Gradient (diff) of the bias tensor argument.

.. index:: pair: define; DNNL_ARG_DIFF_DST
.. _doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST

A special mnemonic for primitives that have a single diff destination argument.

An alias for :ref:`DNNL_ARG_DIFF_DST_0 <doxid-group__dnnl__api__primitives__common_1ga5cf662f6dc742a3500ec4f54290b86da>`.

.. index:: pair: define; DNNL_ARG_DIFF_DST_0
.. _doxid-group__dnnl__api__primitives__common_1ga5cf662f6dc742a3500ec4f54290b86da:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_0

Gradient (diff) of the destination argument #0.

.. index:: pair: define; DNNL_ARG_DIFF_DST_1
.. _doxid-group__dnnl__api__primitives__common_1ga87fad7b00d52b6bff673fca344d0f96d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_1

Gradient (diff) of the destination argument #1.

.. index:: pair: define; DNNL_ARG_DIFF_DST_2
.. _doxid-group__dnnl__api__primitives__common_1gaf6ae542064ad032e7c335f3701398034:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_2

Gradient (diff) of the destination argument #2.

.. index:: pair: define; DNNL_ARG_DIFF_DST_ITER
.. _doxid-group__dnnl__api__primitives__common_1gad9c83f558d1b229b4185ccbf939590a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_ITER

A special mnemonic for gradient (diff) of RNN input recurrent hidden state vector.

An alias for :ref:`DNNL_ARG_DIFF_DST_1 <doxid-group__dnnl__api__primitives__common_1ga87fad7b00d52b6bff673fca344d0f96d>`.

.. index:: pair: define; DNNL_ARG_DIFF_DST_ITER_C
.. _doxid-group__dnnl__api__primitives__common_1ga5524b26b690b9b4b81f0c7f3f9ac3b62:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_ITER_C

A special mnemonic for gradient (diff) of RNN input recurrent cell state vector.

An alias for :ref:`DNNL_ARG_DIFF_DST_2 <doxid-group__dnnl__api__primitives__common_1gaf6ae542064ad032e7c335f3701398034>`.

.. index:: pair: define; DNNL_ARG_DIFF_DST_LAYER
.. _doxid-group__dnnl__api__primitives__common_1gafc6053e276352b05b3b526141586e0ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_DST_LAYER

A special mnemonic for gradient (diff) of RNN output vector.

An alias for :ref:`DNNL_ARG_DIFF_DST_0 <doxid-group__dnnl__api__primitives__common_1ga5cf662f6dc742a3500ec4f54290b86da>`.

.. index:: pair: define; DNNL_ARG_DIFF_SCALE
.. _doxid-group__dnnl__api__primitives__common_1gaeb524b0816a1748026a60195523d3393:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SCALE

A special mnemonic for scale argument of normalization primitives.

.. index:: pair: define; DNNL_ARG_DIFF_SHIFT
.. _doxid-group__dnnl__api__primitives__common_1gaba70712528e1e3922b1b9e3072915df6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SHIFT

A special mnemonic for shift argument of normalization primitives.

.. index:: pair: define; DNNL_ARG_DIFF_SRC
.. _doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC

A special mnemonic for primitives that have a single diff source argument.

An alias for :ref:`DNNL_ARG_DIFF_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga5d7ebcd603a037e0cd53a22377d1addf>`.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_0
.. _doxid-group__dnnl__api__primitives__common_1ga5d7ebcd603a037e0cd53a22377d1addf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_0

Gradient (diff) of the source argument #0.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_1
.. _doxid-group__dnnl__api__primitives__common_1ga79861408c787de29acec450006dace7f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_1

Gradient (diff) of the source argument #1.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_2
.. _doxid-group__dnnl__api__primitives__common_1gae5520b100e7e94e218ccda5697d720bf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_2

Gradient (diff) of the source argument #2.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_3
.. _doxid-group__dnnl__api__primitives__common_1gad0ef2743a16142c7818726b4c970ac07:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_3

Gradient (diff) of the source argument #3.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_ITER
.. _doxid-group__dnnl__api__primitives__common_1ga4f7ed97882e020a1cbaa891bbe0da45b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_ITER

A special mnemonic for gradient (diff) of RNN input recurrent hidden state vector.

An alias for :ref:`DNNL_ARG_DIFF_SRC_1 <doxid-group__dnnl__api__primitives__common_1ga79861408c787de29acec450006dace7f>`.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_ITER_C
.. _doxid-group__dnnl__api__primitives__common_1ga1d8616925684111f3a1b6d8116ab0077:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_ITER_C

A special mnemonic for gradient (diff) of RNN input recurrent cell state vector.

An alias for :ref:`DNNL_ARG_DIFF_SRC_1 <doxid-group__dnnl__api__primitives__common_1ga79861408c787de29acec450006dace7f>`.

.. index:: pair: define; DNNL_ARG_DIFF_SRC_LAYER
.. _doxid-group__dnnl__api__primitives__common_1ga24709fa44c67cf453facbc1c52b0d598:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_SRC_LAYER

A special mnemonic for gradient (diff) of RNN input vector.

An alias for :ref:`DNNL_ARG_DIFF_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga5d7ebcd603a037e0cd53a22377d1addf>`.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS
.. _doxid-group__dnnl__api__primitives__common_1ga3324092ef421f77aebee83b0117cac60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS

A special mnemonic for primitives that have a single diff weights argument.

Alias for :ref:`DNNL_ARG_DIFF_WEIGHTS_0 <doxid-group__dnnl__api__primitives__common_1gaece55926acecbbf375cd4fb06c5786ab>`.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_0
.. _doxid-group__dnnl__api__primitives__common_1gaece55926acecbbf375cd4fb06c5786ab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_0

Gradient (diff) of the weights argument #0.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_1
.. _doxid-group__dnnl__api__primitives__common_1ga789a5f68ece3af7f3c86217a4a5dab7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_1

Gradient (diff) of the weights argument #1.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_2
.. _doxid-group__dnnl__api__primitives__common_1ga4b4fca234fa86cd9270fb3ef7cf9446d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_2

Gradient (diff) of the weights argument #2.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_3
.. _doxid-group__dnnl__api__primitives__common_1ga98277fe5c9232f3089b8a9b5841e821c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_3

Gradient (diff) of the weights argument #3.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_ITER
.. _doxid-group__dnnl__api__primitives__common_1ga4a8e5f32de3856588b2976a766d0af0f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_ITER

A special mnemonic for diff of RNN weights applied to the recurrent input.

An alias for :ref:`DNNL_ARG_DIFF_WEIGHTS_1 <doxid-group__dnnl__api__primitives__common_1ga789a5f68ece3af7f3c86217a4a5dab7e>`.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_LAYER
.. _doxid-group__dnnl__api__primitives__common_1gac0bd0c223011ee2fbbc3c430c047c756:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_LAYER

A special mnemonic for diff of RNN weights applied to the layer input.

An alias for :ref:`DNNL_ARG_DIFF_WEIGHTS_0 <doxid-group__dnnl__api__primitives__common_1gaece55926acecbbf375cd4fb06c5786ab>`.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE
.. _doxid-group__dnnl__api__primitives__common_1ga24d0c96468ef0e28a96f8a6f1b2869b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE

A special mnemonic for diff of RNN weights applied to the peephole weights.

An alias for :ref:`DNNL_ARG_DIFF_WEIGHTS_2 <doxid-group__dnnl__api__primitives__common_1ga4b4fca234fa86cd9270fb3ef7cf9446d>`.

.. index:: pair: define; DNNL_ARG_DIFF_WEIGHTS_PROJECTION
.. _doxid-group__dnnl__api__primitives__common_1ga3e29edb164be584b8f2df0b4ed6256fb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DIFF_WEIGHTS_PROJECTION

A special mnemonic for diff of RNN weights applied to the projection weights.

An alias for :ref:`DNNL_ARG_DIFF_WEIGHTS_3 <doxid-group__dnnl__api__primitives__common_1ga98277fe5c9232f3089b8a9b5841e821c>`.

.. index:: pair: define; DNNL_ARG_DST
.. _doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST

A special mnemonic for destination argument for primitives that have a single destination.

An alias for :ref:`DNNL_ARG_DST_0 <doxid-group__dnnl__api__primitives__common_1ga7ba773575a68b592f3ef0fa885d87c20>`.

.. index:: pair: define; DNNL_ARG_DST_0
.. _doxid-group__dnnl__api__primitives__common_1ga7ba773575a68b592f3ef0fa885d87c20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_0

Destination argument #0.

.. index:: pair: define; DNNL_ARG_DST_1
.. _doxid-group__dnnl__api__primitives__common_1gae4215f24ddcf9e4b20b246e10a992687:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_1

Destination argument #1.

.. index:: pair: define; DNNL_ARG_DST_2
.. _doxid-group__dnnl__api__primitives__common_1gabec2200437d5553747fe56a15548a9b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_2

Destination argument #2.

.. index:: pair: define; DNNL_ARG_DST_ITER
.. _doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_ITER

A special mnemonic for RNN input recurrent hidden state vector.

An alias for :ref:`DNNL_ARG_DST_1 <doxid-group__dnnl__api__primitives__common_1gae4215f24ddcf9e4b20b246e10a992687>`.

.. index:: pair: define; DNNL_ARG_DST_ITER_C
.. _doxid-group__dnnl__api__primitives__common_1ga8b77d8716fc0ab9923d6cb409dbdf900:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_ITER_C

A special mnemonic for LSTM output recurrent cell state vector.

An alias for :ref:`DNNL_ARG_DST_2 <doxid-group__dnnl__api__primitives__common_1gabec2200437d5553747fe56a15548a9b1>`.

.. index:: pair: define; DNNL_ARG_DST_LAYER
.. _doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_DST_LAYER

A special mnemonic for RNN output vector. An alias for :ref:`DNNL_ARG_DST_0 <doxid-group__dnnl__api__primitives__common_1ga7ba773575a68b592f3ef0fa885d87c20>`.

.. index:: pair: define; DNNL_ARG_FROM
.. _doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_FROM

A special mnemonic for reorder source argument.

An alias for :ref:`DNNL_ARG_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`.

.. index:: pair: define; DNNL_ARG_MEAN
.. _doxid-group__dnnl__api__primitives__common_1ga9bcff7f442a5d6a0ac1183533e721066:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_MEAN

Mean values tensor argument.

.. index:: pair: define; DNNL_ARG_MULTIPLE_DST
.. _doxid-group__dnnl__api__primitives__common_1gaa4183403de9fac3efd3f32c7a975580b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_MULTIPLE_DST

Starting index for destination arguments for primitives that produce a variable number of destination arguments.

.. index:: pair: define; DNNL_ARG_MULTIPLE_SRC
.. _doxid-group__dnnl__api__primitives__common_1ga1f0da423df3fb6853ddcbe6ffe964267:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_MULTIPLE_SRC

Starting index for source arguments for primitives that take a variable number of source arguments.

.. index:: pair: define; DNNL_ARG_SCALE
.. _doxid-group__dnnl__api__primitives__common_1ga3c5cac668bc82c90c8da051c7d430370:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SCALE

A special mnemonic for scale argument of normalization primitives.

.. index:: pair: define; DNNL_ARG_SCRATCHPAD
.. _doxid-group__dnnl__api__primitives__common_1ga81836a4db2cb1c4a14d959e304d3f63d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SCRATCHPAD

Scratchpad (temporary storage) tensor argument.

.. index:: pair: define; DNNL_ARG_SHIFT
.. _doxid-group__dnnl__api__primitives__common_1gac250777ced72098caf39deae1d9039c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SHIFT

A special mnemonic for shift argument of normalization primitives.

.. index:: pair: define; DNNL_ARG_SRC
.. _doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC

A special mnemonic for source argument for primitives that have a single source.

An alias for :ref:`DNNL_ARG_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`.

.. index:: pair: define; DNNL_ARG_SRC_0
.. _doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_0

Source argument #0.

.. index:: pair: define; DNNL_ARG_SRC_1
.. _doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_1

Source argument #1.

.. index:: pair: define; DNNL_ARG_SRC_2
.. _doxid-group__dnnl__api__primitives__common_1ga2ad44d7072cc1c13f0d2eeb3f5f59a24:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_2

Source argument #2.

.. index:: pair: define; DNNL_ARG_SRC_3
.. _doxid-group__dnnl__api__primitives__common_1ga3cf129098380c0ee634875c4d65b2c43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_3

Source argument #3.

.. index:: pair: define; DNNL_ARG_SRC_ITER
.. _doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_ITER

A special mnemonic for RNN input recurrent hidden state vector.

An alias for :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`.

.. index:: pair: define; DNNL_ARG_SRC_ITER_C
.. _doxid-group__dnnl__api__primitives__common_1ga8ef6969516e717208a33766542410410:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_ITER_C

A special mnemonic for RNN input recurrent cell state vector.

An alias for :ref:`DNNL_ARG_SRC_2 <doxid-group__dnnl__api__primitives__common_1ga2ad44d7072cc1c13f0d2eeb3f5f59a24>`.

.. index:: pair: define; DNNL_ARG_SRC_LAYER
.. _doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_SRC_LAYER

A special mnemonic for RNN input vector.

An alias for :ref:`DNNL_ARG_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`.

.. index:: pair: define; DNNL_ARG_TO
.. _doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_TO

A special mnemonic for reorder destination argument.

An alias for :ref:`DNNL_ARG_DST_0 <doxid-group__dnnl__api__primitives__common_1ga7ba773575a68b592f3ef0fa885d87c20>`.

.. index:: pair: define; DNNL_ARG_VARIANCE
.. _doxid-group__dnnl__api__primitives__common_1gaa0e60e8d129936ba29555e17efb82581:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_VARIANCE

Variance values tensor argument.

.. index:: pair: define; DNNL_ARG_WEIGHTS
.. _doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS

A special mnemonic for primitives that have a single weights argument.

Alias for :ref:`DNNL_ARG_WEIGHTS_0 <doxid-group__dnnl__api__primitives__common_1ga12d6c5c4ea17c90ade1e324d47756c3f>`.

.. index:: pair: define; DNNL_ARG_WEIGHTS_0
.. _doxid-group__dnnl__api__primitives__common_1ga12d6c5c4ea17c90ade1e324d47756c3f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_0

Weights argument #0.

.. index:: pair: define; DNNL_ARG_WEIGHTS_1
.. _doxid-group__dnnl__api__primitives__common_1ga6daaa6a32731e2a670821f95f8e353fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_1

Weights argument #1.

.. index:: pair: define; DNNL_ARG_WEIGHTS_2
.. _doxid-group__dnnl__api__primitives__common_1gab5744466811fbf73e69f47100a5276e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_2

Weights argument #2.

.. index:: pair: define; DNNL_ARG_WEIGHTS_3
.. _doxid-group__dnnl__api__primitives__common_1gab6c802a32495f116c5918a987dc6a6ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_3

Weights argument #3.

.. index:: pair: define; DNNL_ARG_WEIGHTS_ITER
.. _doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_ITER

A special mnemonic for RNN weights applied to the recurrent input.

An alias for :ref:`DNNL_ARG_WEIGHTS_1 <doxid-group__dnnl__api__primitives__common_1ga6daaa6a32731e2a670821f95f8e353fd>`.

.. index:: pair: define; DNNL_ARG_WEIGHTS_LAYER
.. _doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_LAYER

A special mnemonic for RNN weights applied to the layer input.

An alias for :ref:`DNNL_ARG_WEIGHTS_0 <doxid-group__dnnl__api__primitives__common_1ga12d6c5c4ea17c90ade1e324d47756c3f>`.

.. index:: pair: define; DNNL_ARG_WEIGHTS_PEEPHOLE
.. _doxid-group__dnnl__api__primitives__common_1gab1638cabf7e6c1c3e0d6a1f470707e08:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_PEEPHOLE

A special mnemonic for RNN weights applied to the peephole weights.

An alias for :ref:`DNNL_ARG_WEIGHTS_2 <doxid-group__dnnl__api__primitives__common_1gab5744466811fbf73e69f47100a5276e6>`.

.. index:: pair: define; DNNL_ARG_WEIGHTS_PROJECTION
.. _doxid-group__dnnl__api__primitives__common_1ga28e553bafbca27e9c87020adc8863340:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WEIGHTS_PROJECTION

A special mnemonic for RNN weights applied to the projection weights.

An alias for :ref:`DNNL_ARG_WEIGHTS_3 <doxid-group__dnnl__api__primitives__common_1gab6c802a32495f116c5918a987dc6a6ed>`.

.. index:: pair: define; DNNL_ARG_WORKSPACE
.. _doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_ARG_WORKSPACE

Workspace tensor argument.

Workspace is used to pass information from forward propagation to backward propagation computations.

