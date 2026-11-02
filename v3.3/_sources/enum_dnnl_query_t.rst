.. index:: pair: enum; dnnl_query_t
.. _doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059:

enum dnnl_query_t
=================

Overview
~~~~~~~~

Primitive descriptor query specification. :ref:`More...<details-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_query_t
	{
	    :ref:`dnnl_query_undef<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aaa3009651cd11cc84f7a73ef88671b61>`                  = 0,
	    :ref:`dnnl_query_engine<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad089d541f9a2e7c98ab889dc4bfaaad2>`,
	    :ref:`dnnl_query_primitive_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a43cbb3f840bd56f2fd1d5b8b20493b55>`,
	    :ref:`dnnl_query_num_of_inputs_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1fe7a52f5934c92b0bd0330463549c0e>`,
	    :ref:`dnnl_query_num_of_outputs_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a06833f7d865faf3eaaad3b71976ba16a>`,
	    :ref:`dnnl_query_time_estimate_f64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5a72c2b4080956d6834c38473d2ce88d>`,
	    :ref:`dnnl_query_memory_consumption_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a313c02f42fd648d45795fa0d4b1f93af>`,
	    :ref:`dnnl_query_scratchpad_engine<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aa62d8ff7a29ccf566c5cfbf8fa168097>`,
	    :ref:`dnnl_query_impl_info_str<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5a44980a7317e63cc7b6877d15a549aa>`,
	    :ref:`dnnl_query_reorder_src_engine<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9f81ab2ba3cb5463579f8ba438206448>`,
	    :ref:`dnnl_query_reorder_dst_engine<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9c45c82900a38af7406c3834079318ac>`,
	    :ref:`dnnl_query_prop_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab73dd19af163f8059de03d51898b3a1b>`,
	    :ref:`dnnl_query_cache_blob_id_size_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a31e8bf3af71e992a6bc44720016dced7>`,
	    :ref:`dnnl_query_cache_blob_id<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2b380a4f0c67079a4bba3a434cc83abb>`,
	    :ref:`dnnl_query_strides<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab5f542868da5bc8c3b9d3a80b6e46d25>`,
	    :ref:`dnnl_query_dilations<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1d913ec7d5fe4abdd135bcc12d466e26>`,
	    :ref:`dnnl_query_padding_l<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059af06d8e7df41ec67a72e83b34615039eb>`,
	    :ref:`dnnl_query_padding_r<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9e590958d53ec0c6a349b8209fe1b363>`,
	    :ref:`dnnl_query_epsilon_f32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a568dbf1f44bee9380c7088c98b33b076>`,
	    :ref:`dnnl_query_flags<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5e3f56de8fa19ee5bfe71acc210b9e88>`,
	    :ref:`dnnl_query_alg_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a4985e8a6012dafe63a27d949300a9950>`,
	    :ref:`dnnl_query_alpha_f32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8ab243cc209f01a6500f54e9748e6e7b>`,
	    :ref:`dnnl_query_beta_f32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059acdb00ffb63d304f2be54500a4fc45f6d>`,
	    :ref:`dnnl_query_axis_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f895b53aab59f1ee4137c10bde8bef3>`,
	    :ref:`dnnl_query_local_size_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a70386c09298d5db5265389c3141b7e9a>`,
	    :ref:`dnnl_query_k_f32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0168cb853fa5e77a6c8d6442ef6279c1>`,
	    :ref:`dnnl_query_p_f32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f762ab7a19a7510ee85d1f491f79e8e>`,
	    :ref:`dnnl_query_factors<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aa1410332aa48e1f25f05826789e99cd2>`,
	    :ref:`dnnl_query_cell_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059af754757bf0373a4e1ac7bda8e1b004bb>`,
	    :ref:`dnnl_query_direction<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a05c089679515e0b941a05706339cf304>`,
	    :ref:`dnnl_query_activation_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a3f2afb8643bf0cad548083633297f3ef>`,
	    :ref:`dnnl_query_kernel<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ade9b5c82879e77cf4a5a23c4bd258e3e>`,
	    :ref:`dnnl_query_group_size_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9cd84667c0caafbb8b797de9fe3d6d0e>`,
	    :ref:`dnnl_query_some_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0c4a1096638c39c0771db9a4cb2a3336>`                = 128,
	    :ref:`dnnl_query_src_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5>`,
	    :ref:`dnnl_query_diff_src_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2367b043da6df5d691d6038693f363d6>`,
	    :ref:`dnnl_query_weights_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a12ea0b4858b84889acab34e498323355>`,
	    :ref:`dnnl_query_diff_weights_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8551246c3e70fa1e420411507dbdfe32>`,
	    :ref:`dnnl_query_dst_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`,
	    :ref:`dnnl_query_diff_dst_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae28e33688bf6c55edcf108bd24eb90de>`,
	    :ref:`dnnl_query_workspace_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`,
	    :ref:`dnnl_query_scratchpad_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2ef0eedf44417ce40237f6f459347663>`,
	    :ref:`dnnl_query_exec_arg_md<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ac7ecf09260d89d54ddd7f35c51a244da>`            = 255,
	    :ref:`dnnl_query_ndims_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059afe40d0bef09ca1d2567c46eb413e8580>`,
	    :ref:`dnnl_query_dims<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059abe3af06a74e32063626361f1902aaa87>`,
	    :ref:`dnnl_query_data_type<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530>`,
	    :ref:`dnnl_query_submemory_offset_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a58f5f05e331cf0974fbccad0e2429e67>`,
	    :ref:`dnnl_query_padded_dims<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2bc0848a5ee584227253aa71773db112>`,
	    :ref:`dnnl_query_padded_offsets<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f91293e9b3007cc89ce919852139a36>`,
	    :ref:`dnnl_query_format_kind<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad534a84e6f4709a8f597bf8558730c3e>`,
	    :ref:`dnnl_query_inner_nblks_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a942da7995fe07b02ba1d48be13c6d951>`,
	    :ref:`dnnl_query_inner_blks<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a6c18535baa6bdb2a264c4e62e5f66b73>`,
	    :ref:`dnnl_query_inner_idxs<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae65233dcfb5128c05ed7c97319c00a35>`,
	    :ref:`dnnl_query_sparse_encoding<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a21815bb69d71340b0556f123ba6fdd69>`,
	    :ref:`dnnl_query_nnz_s64<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5ca45f20f5864e069149106f21f5ff92>`,
	    :ref:`dnnl_query_num_handles_s32<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a7d92c3824fd1811f6bc641e2fdfbc2bb>`,
	    :ref:`dnnl_query_max<doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0c1f3b9e3113ee4ba2156c3e6cee4917>`                    = 0x7fff,
	};

.. _details-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive descriptor query specification.

For generic function :ref:`dnnl_primitive_desc_query() <doxid-group__dnnl__api__primitives__common_1ga041881114858228279174aff5c1f5e75>`, the type of result must agree with the queried argument. The correspondence table:

====================================================================================================================================================  ======================================================================================================================  
Query kind                                                                                                                                            Type of query result ----                                                                                               
====================================================================================================================================================  ======================================================================================================================  
dnnl_query_*_engine                                                                                                                                   :ref:`dnnl_engine_t <doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` *                             
:ref:`dnnl_query_primitive_kind <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a43cbb3f840bd56f2fd1d5b8b20493b55>`   :ref:`dnnl_primitive_kind_t <doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` *         
dnnl_query_*_s32                                                                                                                                      int *                                                                                                                   
dnnl_query_*_s64                                                                                                                                      :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` * (same as int64_t *)       
dnnl_query_*_f32                                                                                                                                      float *                                                                                                                 
dnnl_query_*_f64                                                                                                                                      double *                                                                                                                
dnnl_query_*_str                                                                                                                                      const char **                                                                                                           
dnnl_query_*_md                                                                                                                                       :ref:`const_dnnl_memory_desc_t <doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` *                  
dnnl_query_*_pd                                                                                                                                       :ref:`const_dnnl_primitive_desc_t <doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` *   
dnnl_query_cache_blob_id                                                                                                                              const uint8_t **                                                                                                        
dnnl_query_strides                                                                                                                                    const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_dilations                                                                                                                                  const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_padding_l                                                                                                                                  const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_padding_r                                                                                                                                  const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_flags                                                                                                                                      unsigned *                                                                                                              
dnnl_query_alg_kind                                                                                                                                   :ref:`dnnl_alg_kind_t <doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` *               
dnnl_query_factors                                                                                                                                    const float **                                                                                                          
dnnl_query_cell_kind                                                                                                                                  :ref:`dnnl_alg_kind_t <doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` *               
dnnl_query_direction                                                                                                                                  :ref:`dnnl_rnn_direction_t <doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` *                         
dnnl_query_activation_kind                                                                                                                            :ref:`dnnl_alg_kind_t <doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` *               
dnnl_query_kernel                                                                                                                                     const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_dims                                                                                                                                       const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_data_type                                                                                                                                  :ref:`dnnl_data_type_t <doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` *                     
dnnl_query_padded_dims                                                                                                                                const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_padded_offsets                                                                                                                             const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_format_kind                                                                                                                                :ref:`dnnl_format_kind_t <doxid-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d>` *                        
dnnl_query_inner_blks                                                                                                                                 const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_inner_idxs                                                                                                                                 const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` **                   
dnnl_query_sparse_encoding                                                                                                                            :ref:`dnnl_sparse_encoding_t <doxid-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552>` *                    
====================================================================================================================================================  ======================================================================================================================

.. note:: 

   Rule of thumb: all opaque types and structures are returned by reference. All numbers are returned by value.
   
   

.. warning:: 

   All returned references point to constant objects and are valid only during the lifetime of the queried primitive descriptor. Returned objects must not be destroyed by the user. If you need to keep the object longer than the lifetime of the queried primitive descriptor, use :ref:`dnnl_primitive_desc_clone() <doxid-group__dnnl__api__primitives__common_1gae40abecf7360106805eabc049cc86e4b>` to make a copy.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_query_undef
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aaa3009651cd11cc84f7a73ef88671b61:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_undef

no query

.. index:: pair: enumvalue; dnnl_query_engine
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad089d541f9a2e7c98ab889dc4bfaaad2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_engine

execution engine

.. index:: pair: enumvalue; dnnl_query_primitive_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a43cbb3f840bd56f2fd1d5b8b20493b55:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_primitive_kind

primitive kind

.. index:: pair: enumvalue; dnnl_query_num_of_inputs_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1fe7a52f5934c92b0bd0330463549c0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_num_of_inputs_s32

number of inputs expected

.. index:: pair: enumvalue; dnnl_query_num_of_outputs_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a06833f7d865faf3eaaad3b71976ba16a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_num_of_outputs_s32

number of outputs expected

.. index:: pair: enumvalue; dnnl_query_time_estimate_f64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5a72c2b4080956d6834c38473d2ce88d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_time_estimate_f64

runtime estimation (seconds)

.. index:: pair: enumvalue; dnnl_query_memory_consumption_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a313c02f42fd648d45795fa0d4b1f93af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_memory_consumption_s64

memory consumption extra

.. index:: pair: enumvalue; dnnl_query_scratchpad_engine
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aa62d8ff7a29ccf566c5cfbf8fa168097:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_scratchpad_engine

(scratch) memory, additional to all inputs and outputs memory (bytes)

scratchpad engine engine to be used

.. index:: pair: enumvalue; dnnl_query_impl_info_str
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5a44980a7317e63cc7b6877d15a549aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_impl_info_str

for creating scratchpad memory

implementation name

.. index:: pair: enumvalue; dnnl_query_reorder_src_engine
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9f81ab2ba3cb5463579f8ba438206448:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_reorder_src_engine

source engine

.. index:: pair: enumvalue; dnnl_query_reorder_dst_engine
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9c45c82900a38af7406c3834079318ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_reorder_dst_engine

destination engine

.. index:: pair: enumvalue; dnnl_query_prop_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab73dd19af163f8059de03d51898b3a1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_prop_kind

propagation kind

.. index:: pair: enumvalue; dnnl_query_cache_blob_id_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a31e8bf3af71e992a6bc44720016dced7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_cache_blob_id_size_s64

size of cache blob ID in bytes

.. index:: pair: enumvalue; dnnl_query_cache_blob_id
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2b380a4f0c67079a4bba3a434cc83abb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_cache_blob_id

cache blob ID (pointer to array)

.. index:: pair: enumvalue; dnnl_query_strides
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab5f542868da5bc8c3b9d3a80b6e46d25:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_strides

strides

.. index:: pair: enumvalue; dnnl_query_dilations
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1d913ec7d5fe4abdd135bcc12d466e26:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_dilations

dilations

.. index:: pair: enumvalue; dnnl_query_padding_l
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059af06d8e7df41ec67a72e83b34615039eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_padding_l

left padding

.. index:: pair: enumvalue; dnnl_query_padding_r
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9e590958d53ec0c6a349b8209fe1b363:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_padding_r

right padding

.. index:: pair: enumvalue; dnnl_query_epsilon_f32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a568dbf1f44bee9380c7088c98b33b076:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_epsilon_f32

epsilon

.. index:: pair: enumvalue; dnnl_query_flags
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5e3f56de8fa19ee5bfe71acc210b9e88:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_flags

flags

.. index:: pair: enumvalue; dnnl_query_alg_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a4985e8a6012dafe63a27d949300a9950:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_alg_kind

algorithm kind

.. index:: pair: enumvalue; dnnl_query_alpha_f32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8ab243cc209f01a6500f54e9748e6e7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_alpha_f32

alpha

.. index:: pair: enumvalue; dnnl_query_beta_f32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059acdb00ffb63d304f2be54500a4fc45f6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_beta_f32

beta

.. index:: pair: enumvalue; dnnl_query_axis_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f895b53aab59f1ee4137c10bde8bef3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_axis_s32

axis

.. index:: pair: enumvalue; dnnl_query_local_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a70386c09298d5db5265389c3141b7e9a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_local_size_s64

LRN parameter local size.

.. index:: pair: enumvalue; dnnl_query_k_f32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0168cb853fa5e77a6c8d6442ef6279c1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_k_f32

LRN parameter K.

.. index:: pair: enumvalue; dnnl_query_p_f32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f762ab7a19a7510ee85d1f491f79e8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_p_f32

Reduction parameter P.

.. index:: pair: enumvalue; dnnl_query_factors
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aa1410332aa48e1f25f05826789e99cd2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_factors

Resampling parameter factors.

.. index:: pair: enumvalue; dnnl_query_cell_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059af754757bf0373a4e1ac7bda8e1b004bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_cell_kind

RNN parameter cell kind.

.. index:: pair: enumvalue; dnnl_query_direction
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a05c089679515e0b941a05706339cf304:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_direction

RNN parameter direction.

.. index:: pair: enumvalue; dnnl_query_activation_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a3f2afb8643bf0cad548083633297f3ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_activation_kind

RNN parameter activation kind.

.. index:: pair: enumvalue; dnnl_query_kernel
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ade9b5c82879e77cf4a5a23c4bd258e3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_kernel

Pooling parameter kernel.

.. index:: pair: enumvalue; dnnl_query_group_size_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a9cd84667c0caafbb8b797de9fe3d6d0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_group_size_s64

Shuffle parameter group size.

.. index:: pair: enumvalue; dnnl_query_some_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0c4a1096638c39c0771db9a4cb2a3336:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_some_md

stub

.. index:: pair: enumvalue; dnnl_query_src_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_src_md

source memory desc

.. index:: pair: enumvalue; dnnl_query_diff_src_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2367b043da6df5d691d6038693f363d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_diff_src_md

source gradient memory desc

.. index:: pair: enumvalue; dnnl_query_weights_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a12ea0b4858b84889acab34e498323355:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_weights_md

weights memory descriptor desc

.. index:: pair: enumvalue; dnnl_query_diff_weights_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8551246c3e70fa1e420411507dbdfe32:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_diff_weights_md

weights grad. memory desc

.. index:: pair: enumvalue; dnnl_query_dst_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_dst_md

destination memory desc

.. index:: pair: enumvalue; dnnl_query_diff_dst_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae28e33688bf6c55edcf108bd24eb90de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_diff_dst_md

destination grad. memory desc

.. index:: pair: enumvalue; dnnl_query_workspace_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_workspace_md

workspace memory desc

.. index:: pair: enumvalue; dnnl_query_scratchpad_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2ef0eedf44417ce40237f6f459347663:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_scratchpad_md

scratchpad memory desc

.. index:: pair: enumvalue; dnnl_query_exec_arg_md
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ac7ecf09260d89d54ddd7f35c51a244da:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_exec_arg_md

memory desc of an execute argument

.. index:: pair: enumvalue; dnnl_query_ndims_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059afe40d0bef09ca1d2567c46eb413e8580:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_ndims_s32

number of dimensions

.. index:: pair: enumvalue; dnnl_query_dims
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059abe3af06a74e32063626361f1902aaa87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_dims

vector of dimensions

.. index:: pair: enumvalue; dnnl_query_data_type
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_data_type

data type

.. index:: pair: enumvalue; dnnl_query_submemory_offset_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a58f5f05e331cf0974fbccad0e2429e67:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_submemory_offset_s64

submemory offset

.. index:: pair: enumvalue; dnnl_query_padded_dims
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2bc0848a5ee584227253aa71773db112:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_padded_dims

vector of padded dimensions

.. index:: pair: enumvalue; dnnl_query_padded_offsets
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f91293e9b3007cc89ce919852139a36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_padded_offsets

vector of padded offsets

.. index:: pair: enumvalue; dnnl_query_format_kind
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad534a84e6f4709a8f597bf8558730c3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_format_kind

format kind

.. index:: pair: enumvalue; dnnl_query_inner_nblks_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a942da7995fe07b02ba1d48be13c6d951:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_inner_nblks_s32

number of innermost blocks

.. index:: pair: enumvalue; dnnl_query_inner_blks
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a6c18535baa6bdb2a264c4e62e5f66b73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_inner_blks

vector of sizes of the innermost blocks

.. index:: pair: enumvalue; dnnl_query_inner_idxs
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae65233dcfb5128c05ed7c97319c00a35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_inner_idxs

vector of logical indices of the blocks

.. index:: pair: enumvalue; dnnl_query_sparse_encoding
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a21815bb69d71340b0556f123ba6fdd69:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_sparse_encoding

Sparse encoding.

.. index:: pair: enumvalue; dnnl_query_nnz_s64
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5ca45f20f5864e069149106f21f5ff92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_nnz_s64

Number of non-zero entries.

.. index:: pair: enumvalue; dnnl_query_num_handles_s32
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a7d92c3824fd1811f6bc641e2fdfbc2bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_num_handles_s32

Number of buffers required for a memory.

.. index:: pair: enumvalue; dnnl_query_max
.. _doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a0c1f3b9e3113ee4ba2156c3e6cee4917:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_query_max

descriptor

