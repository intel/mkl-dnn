.. index:: pair: enum; query
.. _doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1:

enum dnnl::query
================

Overview
~~~~~~~~

Primitive descriptor query specification. :ref:`More...<details-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum query
	{
	    :ref:`undef<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1af31ee5e3824f1f5e5d206bdf3029f22b>`                  = dnnl_query_undef,
	    :ref:`engine<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`                 = dnnl_query_engine,
	    :ref:`primitive_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a6e115efac481dc815e442e6ff181f7e2>`         = dnnl_query_primitive_kind,
	    :ref:`num_of_inputs_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a8e76d4a603e890128e5205c75581f80b>`      = dnnl_query_num_of_inputs_s32,
	    :ref:`num_of_outputs_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a60ba192a313fabc2c91f5295d1422491>`     = dnnl_query_num_of_outputs_s32,
	    :ref:`time_estimate_f64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab307b4b4cf28742beddf2e9e2de6bce0>`      = dnnl_query_time_estimate_f64,
	    :ref:`memory_consumption_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0ed44d67e94c1c7ac5f219491e422506>` = dnnl_query_memory_consumption_s64,
	    :ref:`scratchpad_engine<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a75db5e5697414843589825652e338a9a>`      = dnnl_query_scratchpad_engine,
	    :ref:`reorder_src_engine<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a783fb4ccfa962b218a2acbd35dd7ec27>`     = dnnl_query_reorder_src_engine,
	    :ref:`reorder_dst_engine<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa08e784e657998224809d67bde0787ce>`     = dnnl_query_reorder_dst_engine,
	    :ref:`impl_info_str<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a2c6dbd26a5e4bd5689bdcbfdf00e35cf>`          = dnnl_query_impl_info_str,
	    :ref:`prop_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a2ba5356a73a761f488b6d9e5f028134f>`              = dnnl_query_prop_kind,
	    :ref:`cache_blob_id_size_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a201cac901044470f4abfe41c760dc904>` = dnnl_query_cache_blob_id_size_s64,
	    :ref:`cache_blob_id<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab709f23530e04f96d6fbd9702e5506f4>`          = dnnl_query_cache_blob_id,
	    :ref:`strides<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3372f3d8ac7d6db0997a8fe6b38d549a>`                = dnnl_query_strides,
	    :ref:`dilations<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1acbcf9c952f6e423b94fe04593665b49e>`              = dnnl_query_dilations,
	    :ref:`padding_l<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a364bea0036bd487dbd697b5401a400e5>`              = dnnl_query_padding_l,
	    :ref:`padding_r<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3c3250739d8474d7c8a49132221e8680>`              = dnnl_query_padding_r,
	    :ref:`epsilon_f32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0e4c72953ac0b43b905c26835d0d698b>`            = dnnl_query_epsilon_f32,
	    :ref:`flags<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a4e5868d676cb634aa75b125a0f741abf>`                  = dnnl_query_flags,
	    :ref:`alg_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a81b70f7363a5b96ee4720a9cc6067177>`               = dnnl_query_alg_kind,
	    :ref:`alpha_f32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3b26a81b2c3b977b66cbd40b57dc736f>`              = dnnl_query_alpha_f32,
	    :ref:`beta_f32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a246b4b4974fbd719f2d4d625466d2c8f>`               = dnnl_query_beta_f32,
	    :ref:`axis_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aaed746342e0653956288e797c3226442>`               = dnnl_query_axis_s32,
	    :ref:`local_size_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ae6b06c486734ca932d046340484b5e2f>`         = dnnl_query_local_size_s64,
	    :ref:`k_f32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa6d2008ab1a909e5a12cb9e8beb263b9>`                  = dnnl_query_k_f32,
	    :ref:`p_f32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab42859604106392e6c836373beb517cc>`                  = dnnl_query_p_f32,
	    :ref:`factors<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ac6b7ed918ad52b0b075a928a0d40dcc6>`                = dnnl_query_factors,
	    :ref:`cell_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a29996a3fc1a66b059e962413a152cf59>`              = dnnl_query_cell_kind,
	    :ref:`direction<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aef72c37be9d1b9e6e5bbd6ef09448abe>`              = dnnl_query_direction,
	    :ref:`activation_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b2a64f0f7accddec932b42dc9ab3ef8>`        = dnnl_query_activation_kind,
	    :ref:`kernel<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a50484c19f1afdaf3841a0d821ed393d2>`                 = dnnl_query_kernel,
	    :ref:`group_size_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b5391813b2603a33685094e31408405>`         = dnnl_query_group_size_s64,
	    :ref:`src_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>`                 = dnnl_query_src_md,
	    :ref:`diff_src_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a8f85b9dfff73532e30d8aab798020233>`            = dnnl_query_diff_src_md,
	    :ref:`weights_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>`             = dnnl_query_weights_md,
	    :ref:`diff_weights_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a345e134a47299ca126f8cc4aeeeb05ac>`        = dnnl_query_diff_weights_md,
	    :ref:`dst_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>`                 = dnnl_query_dst_md,
	    :ref:`diff_dst_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab731735b7280a17b7b03c964cf9b1967>`            = dnnl_query_diff_dst_md,
	    :ref:`workspace_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3874c56bb4069607213666573dff2a96>`           = dnnl_query_workspace_md,
	    :ref:`scratchpad_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a9cbdd03b65c030ef560b5555be1a86c2>`          = dnnl_query_scratchpad_md,
	    :ref:`exec_arg_md<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ad531896cf1d66c4832790f428623f164>`            = dnnl_query_exec_arg_md,
	    :ref:`ndims_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a77f62a7119e4c0f69a941e11d41760a9>`              = dnnl_query_ndims_s32,
	    :ref:`dims<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`                   = dnnl_query_dims,
	    :ref:`data_type<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a17f71d965fe9589ddbd11caf7182243e>`              = dnnl_query_data_type,
	    :ref:`submemory_offset_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ace4f612879d8a5506ae92c04f6c3a658>`   = dnnl_query_submemory_offset_s64,
	    :ref:`padded_dims<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa5c3e8ef954f0f88db5c2cc19dfc67ef>`            = dnnl_query_padded_dims,
	    :ref:`padded_offsets<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a52a01adcb08b48d239c7d48fa6233264>`         = dnnl_query_padded_offsets,
	    :ref:`format_kind<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5965993b2f9e9320be929e2306122e28>`            = dnnl_query_format_kind,
	    :ref:`inner_nblks_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ac325a338279b20891bbc5dff7df1b6ea>`        = dnnl_query_inner_nblks_s32,
	    :ref:`inner_blks<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a917b86ca9ffa3aa65ecd37c68f46aa58>`             = dnnl_query_inner_blks,
	    :ref:`inner_idxs<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3f2c7323955b5d91b14b4fbce6ee95f4>`             = dnnl_query_inner_idxs,
	    :ref:`sparse_encoding<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b4867c142c9f6c67e697fb5c78f1797>`        = dnnl_query_sparse_encoding,
	    :ref:`nnz_s64<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a69dcad5d4fb3e795bcde5b595fea50a6>`                = dnnl_query_nnz_s64,
	    :ref:`num_handles_s32<doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1af0169ef649f325d826295e7b315ea2e7>`        = dnnl_query_num_handles_s32,
	};

.. _details-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor query specification.

In general, queries are not used with the C++ API because most queries are implemented as class members.

See :ref:`dnnl_query_t <doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` for more information.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

no query

.. index:: pair: enumvalue; engine
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	engine

execution engine

.. index:: pair: enumvalue; primitive_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a6e115efac481dc815e442e6ff181f7e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_kind

primitive kind

.. index:: pair: enumvalue; num_of_inputs_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a8e76d4a603e890128e5205c75581f80b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	num_of_inputs_s32

number of inputs expected

.. index:: pair: enumvalue; num_of_outputs_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a60ba192a313fabc2c91f5295d1422491:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	num_of_outputs_s32

number of outputs expected

.. index:: pair: enumvalue; time_estimate_f64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab307b4b4cf28742beddf2e9e2de6bce0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	time_estimate_f64

runtime estimation (seconds), unimplemented

.. index:: pair: enumvalue; memory_consumption_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0ed44d67e94c1c7ac5f219491e422506:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	memory_consumption_s64

memory required for scratchpad (bytes)



.. rubric:: See also:

:ref:`Primitive Attributes: Scratchpad <doxid-dev_guide_attributes_scratchpad>`

.. index:: pair: enumvalue; scratchpad_engine
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a75db5e5697414843589825652e338a9a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	scratchpad_engine

scratchpad engine

engine to be used for creating scratchpad memory

.. index:: pair: enumvalue; reorder_src_engine
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a783fb4ccfa962b218a2acbd35dd7ec27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder_src_engine

reorder source engine

.. index:: pair: enumvalue; reorder_dst_engine
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa08e784e657998224809d67bde0787ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder_dst_engine

reorder destination engine

.. index:: pair: enumvalue; impl_info_str
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a2c6dbd26a5e4bd5689bdcbfdf00e35cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	impl_info_str

implementation name

.. index:: pair: enumvalue; prop_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a2ba5356a73a761f488b6d9e5f028134f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	prop_kind

propagation kind

.. index:: pair: enumvalue; cache_blob_id_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a201cac901044470f4abfe41c760dc904:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cache_blob_id_size_s64

size of cache blob ID in bytes

.. index:: pair: enumvalue; cache_blob_id
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab709f23530e04f96d6fbd9702e5506f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cache_blob_id

cache blob ID (pointer to array)

.. index:: pair: enumvalue; strides
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3372f3d8ac7d6db0997a8fe6b38d549a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	strides

strides

.. index:: pair: enumvalue; dilations
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1acbcf9c952f6e423b94fe04593665b49e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dilations

dilations

.. index:: pair: enumvalue; padding_l
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a364bea0036bd487dbd697b5401a400e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	padding_l

left padding

.. index:: pair: enumvalue; padding_r
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3c3250739d8474d7c8a49132221e8680:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	padding_r

right padding

.. index:: pair: enumvalue; epsilon_f32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0e4c72953ac0b43b905c26835d0d698b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	epsilon_f32

epsilon

.. index:: pair: enumvalue; flags
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a4e5868d676cb634aa75b125a0f741abf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	flags

flags

.. index:: pair: enumvalue; alg_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a81b70f7363a5b96ee4720a9cc6067177:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	alg_kind

algorithm kind

.. index:: pair: enumvalue; alpha_f32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3b26a81b2c3b977b66cbd40b57dc736f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	alpha_f32

alpha

.. index:: pair: enumvalue; beta_f32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a246b4b4974fbd719f2d4d625466d2c8f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	beta_f32

beta

.. index:: pair: enumvalue; axis_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aaed746342e0653956288e797c3226442:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	axis_s32

axis

.. index:: pair: enumvalue; local_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ae6b06c486734ca932d046340484b5e2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	local_size_s64

LRN parameter local size.

.. index:: pair: enumvalue; k_f32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa6d2008ab1a909e5a12cb9e8beb263b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	k_f32

LRN parameter K.

.. index:: pair: enumvalue; p_f32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab42859604106392e6c836373beb517cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	p_f32

Reduction parameter P.

.. index:: pair: enumvalue; factors
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ac6b7ed918ad52b0b075a928a0d40dcc6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	factors

Resampling parameter factors.

.. index:: pair: enumvalue; cell_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a29996a3fc1a66b059e962413a152cf59:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cell_kind

RNN parameter cell kind.

.. index:: pair: enumvalue; direction
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aef72c37be9d1b9e6e5bbd6ef09448abe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	direction

RNN parameter direction.

.. index:: pair: enumvalue; activation_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b2a64f0f7accddec932b42dc9ab3ef8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	activation_kind

RNN parameter activation kind.

.. index:: pair: enumvalue; kernel
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a50484c19f1afdaf3841a0d821ed393d2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	kernel

Pooling parameter kernel.

.. index:: pair: enumvalue; group_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b5391813b2603a33685094e31408405:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	group_size_s64

Shuffle parameter group size.

.. index:: pair: enumvalue; src_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	src_md

source memory desc

.. index:: pair: enumvalue; diff_src_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a8f85b9dfff73532e30d8aab798020233:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	diff_src_md

source gradient (diff) memory desc

.. index:: pair: enumvalue; weights_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	weights_md

weights memory descriptor desc

.. index:: pair: enumvalue; diff_weights_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a345e134a47299ca126f8cc4aeeeb05ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	diff_weights_md

weights gradient (diff) memory desc

.. index:: pair: enumvalue; dst_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dst_md

destination memory desc

.. index:: pair: enumvalue; diff_dst_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ab731735b7280a17b7b03c964cf9b1967:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	diff_dst_md

destination gradient (diff) memory desc

.. index:: pair: enumvalue; workspace_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3874c56bb4069607213666573dff2a96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	workspace_md

workspace memory desc

.. index:: pair: enumvalue; scratchpad_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a9cbdd03b65c030ef560b5555be1a86c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	scratchpad_md

scratchpad memory desc

.. index:: pair: enumvalue; exec_arg_md
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ad531896cf1d66c4832790f428623f164:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	exec_arg_md

memory desc of an execute argument

.. index:: pair: enumvalue; ndims_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a77f62a7119e4c0f69a941e11d41760a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ndims_s32

number of dimensions

.. index:: pair: enumvalue; dims
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dims

vector of dimensions

.. index:: pair: enumvalue; data_type
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a17f71d965fe9589ddbd11caf7182243e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	data_type

data type

.. index:: pair: enumvalue; submemory_offset_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ace4f612879d8a5506ae92c04f6c3a658:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	submemory_offset_s64

submemory offset

.. index:: pair: enumvalue; padded_dims
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aa5c3e8ef954f0f88db5c2cc19dfc67ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	padded_dims

vector of padded dimensions

.. index:: pair: enumvalue; padded_offsets
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a52a01adcb08b48d239c7d48fa6233264:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	padded_offsets

vector of padded offsets

.. index:: pair: enumvalue; format_kind
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5965993b2f9e9320be929e2306122e28:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	format_kind

format kind

.. index:: pair: enumvalue; inner_nblks_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ac325a338279b20891bbc5dff7df1b6ea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	inner_nblks_s32

number of innermost blocks

.. index:: pair: enumvalue; inner_blks
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a917b86ca9ffa3aa65ecd37c68f46aa58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	inner_blks

vector of sizes of the innermost blocks

.. index:: pair: enumvalue; inner_idxs
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a3f2c7323955b5d91b14b4fbce6ee95f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	inner_idxs

vector of logical indices of the blocks

.. index:: pair: enumvalue; sparse_encoding
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a0b4867c142c9f6c67e697fb5c78f1797:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sparse_encoding

Sparse encoding.

.. index:: pair: enumvalue; nnz_s64
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a69dcad5d4fb3e795bcde5b595fea50a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	nnz_s64

Number of non-zero entries.

.. index:: pair: enumvalue; num_handles_s32
.. _doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1af0169ef649f325d826295e7b315ea2e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	num_handles_s32

Number of buffers required for a memory descriptor.

