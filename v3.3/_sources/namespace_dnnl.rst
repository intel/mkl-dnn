.. index:: pair: namespace; dnnl
.. _doxid-namespacednnl:

namespace dnnl
==============

.. toctree::
	:hidden:

	namespace_dnnl_graph.rst

oneDNN namespace


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace dnnl {

	// namespaces

	namespace :ref:`dnnl::graph<doxid-namespacednnl_1_1graph>`;
		namespace :ref:`dnnl::graph::sycl_interop<doxid-namespacednnl_1_1graph_1_1sycl__interop>`;
	namespace :ref:`dnnl::ocl_interop<doxid-namespacednnl_1_1ocl__interop>`;
	namespace :ref:`dnnl::sycl_interop<doxid-namespacednnl_1_1sycl__interop>`;
	namespace :ref:`dnnl::threadpool_interop<doxid-namespacednnl_1_1threadpool__interop>`;

	// typedefs

	typedef :ref:`dnnl_version_t<doxid-structdnnl__version__t>` :ref:`version_t<doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af>`;

	// enums

	enum :ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`;
	enum :ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>`;
	enum :ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>`;
	enum :ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`;
	enum :ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`;
	enum :ref:`profiling_data_kind<doxid-group__dnnl__api__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>`;
	enum :ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>`;
	enum :ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>`;
	enum :ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>`;
	enum :ref:`rnn_flags<doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>`;
	enum :ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>`;
	enum :ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>`;

	// structs

	struct :ref:`augru_backward<doxid-structdnnl_1_1augru__backward>`;
	struct :ref:`augru_forward<doxid-structdnnl_1_1augru__forward>`;
	struct :ref:`batch_normalization_backward<doxid-structdnnl_1_1batch__normalization__backward>`;
	struct :ref:`batch_normalization_forward<doxid-structdnnl_1_1batch__normalization__forward>`;
	struct :ref:`binary<doxid-structdnnl_1_1binary>`;
	struct :ref:`concat<doxid-structdnnl_1_1concat>`;
	struct :ref:`convolution_backward_data<doxid-structdnnl_1_1convolution__backward__data>`;
	struct :ref:`convolution_backward_weights<doxid-structdnnl_1_1convolution__backward__weights>`;
	struct :ref:`convolution_forward<doxid-structdnnl_1_1convolution__forward>`;
	struct :ref:`deconvolution_backward_data<doxid-structdnnl_1_1deconvolution__backward__data>`;
	struct :ref:`deconvolution_backward_weights<doxid-structdnnl_1_1deconvolution__backward__weights>`;
	struct :ref:`deconvolution_forward<doxid-structdnnl_1_1deconvolution__forward>`;
	struct :ref:`eltwise_backward<doxid-structdnnl_1_1eltwise__backward>`;
	struct :ref:`eltwise_forward<doxid-structdnnl_1_1eltwise__forward>`;
	struct :ref:`engine<doxid-structdnnl_1_1engine>`;
	struct :ref:`error<doxid-structdnnl_1_1error>`;
	struct :ref:`group_normalization_backward<doxid-structdnnl_1_1group__normalization__backward>`;
	struct :ref:`group_normalization_forward<doxid-structdnnl_1_1group__normalization__forward>`;
	struct :ref:`gru_backward<doxid-structdnnl_1_1gru__backward>`;
	struct :ref:`gru_forward<doxid-structdnnl_1_1gru__forward>`;

	template <typename T, typename traits = handle_traits<T>>
	struct :ref:`handle<doxid-structdnnl_1_1handle>`;

	template <typename T>
	struct :ref:`handle_traits<doxid-structdnnl_1_1handle__traits>`;

	struct :ref:`inner_product_backward_data<doxid-structdnnl_1_1inner__product__backward__data>`;
	struct :ref:`inner_product_backward_weights<doxid-structdnnl_1_1inner__product__backward__weights>`;
	struct :ref:`inner_product_forward<doxid-structdnnl_1_1inner__product__forward>`;
	struct :ref:`layer_normalization_backward<doxid-structdnnl_1_1layer__normalization__backward>`;
	struct :ref:`layer_normalization_forward<doxid-structdnnl_1_1layer__normalization__forward>`;
	struct :ref:`lbr_augru_backward<doxid-structdnnl_1_1lbr__augru__backward>`;
	struct :ref:`lbr_augru_forward<doxid-structdnnl_1_1lbr__augru__forward>`;
	struct :ref:`lbr_gru_backward<doxid-structdnnl_1_1lbr__gru__backward>`;
	struct :ref:`lbr_gru_forward<doxid-structdnnl_1_1lbr__gru__forward>`;
	struct :ref:`lrn_backward<doxid-structdnnl_1_1lrn__backward>`;
	struct :ref:`lrn_forward<doxid-structdnnl_1_1lrn__forward>`;
	struct :ref:`lstm_backward<doxid-structdnnl_1_1lstm__backward>`;
	struct :ref:`lstm_forward<doxid-structdnnl_1_1lstm__forward>`;
	struct :ref:`matmul<doxid-structdnnl_1_1matmul>`;
	struct :ref:`memory<doxid-structdnnl_1_1memory>`;
	struct :ref:`pooling_backward<doxid-structdnnl_1_1pooling__backward>`;
	struct :ref:`pooling_forward<doxid-structdnnl_1_1pooling__forward>`;
	struct :ref:`post_ops<doxid-structdnnl_1_1post__ops>`;
	struct :ref:`prelu_backward<doxid-structdnnl_1_1prelu__backward>`;
	struct :ref:`prelu_forward<doxid-structdnnl_1_1prelu__forward>`;
	struct :ref:`primitive<doxid-structdnnl_1_1primitive>`;
	struct :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`;
	struct :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`;
	struct :ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc__base>`;
	struct :ref:`reduction<doxid-structdnnl_1_1reduction>`;
	struct :ref:`reorder<doxid-structdnnl_1_1reorder>`;
	struct :ref:`resampling_backward<doxid-structdnnl_1_1resampling__backward>`;
	struct :ref:`resampling_forward<doxid-structdnnl_1_1resampling__forward>`;
	struct :ref:`rnn_primitive_desc_base<doxid-structdnnl_1_1rnn__primitive__desc__base>`;
	struct :ref:`shuffle_backward<doxid-structdnnl_1_1shuffle__backward>`;
	struct :ref:`shuffle_forward<doxid-structdnnl_1_1shuffle__forward>`;
	struct :ref:`softmax_backward<doxid-structdnnl_1_1softmax__backward>`;
	struct :ref:`softmax_forward<doxid-structdnnl_1_1softmax__forward>`;
	struct :ref:`stream<doxid-structdnnl_1_1stream>`;
	struct :ref:`sum<doxid-structdnnl_1_1sum>`;
	struct :ref:`vanilla_rnn_backward<doxid-structdnnl_1_1vanilla__rnn__backward>`;
	struct :ref:`vanilla_rnn_forward<doxid-structdnnl_1_1vanilla__rnn__forward>`;

	// global functions

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` :ref:`convert_to_c<doxid-group__dnnl__api__primitives__common_1gaaa215c424a2a5c5f734600216dfb8873>`(:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` akind);
	:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` :ref:`convert_to_c<doxid-group__dnnl__api__attributes_1gaa30f540e1ed09b2865f153fd599c967b>`(:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode);
	:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` :ref:`convert_to_c<doxid-group__dnnl__api__attributes_1gae13881206fecd43ce0e0daead7f0009e>`(:ref:`prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` akind);
	:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` :ref:`convert_to_c<doxid-group__dnnl__api__attributes_1gad4c07d30e46391ce7ce0900d18cbfa30>`(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm);
	:ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` :ref:`convert_to_c<doxid-group__dnnl__api__primitives__common_1gae3d2ea872c5ab424c74d7549d2222926>`(:ref:`normalization_flags<doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags);
	:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` :ref:`convert_to_c<doxid-group__dnnl__api__rnn_1ga0a340195a137f906e858418d91397777>`(:ref:`rnn_flags<doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>` flags);
	:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` :ref:`convert_to_c<doxid-group__dnnl__api__rnn_1ga1915ea2d2fe94077fa30734ced88a225>`(:ref:`rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` dir);
	:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` :ref:`convert_to_c<doxid-group__dnnl__api__primitives__common_1ga01d8a1881875cdb94e230db4e53ccb97>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` aquery);
	bool :target:`operator ==<doxid-group__dnnl__api__memory_1gaf97bbd7e992c0e211da42bc6eaf12758>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`operator !=<doxid-group__dnnl__api__memory_1ga03fa7afa494ab8dc8484f83a34ce20a6>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`operator ==<doxid-group__dnnl__api__memory_1gafa9de7b46bedc943161863d3eaa84100>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`operator !=<doxid-group__dnnl__api__memory_1ga7d9b4a4b2297d66c9495d3a1f2769167>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`operator ==<doxid-group__dnnl__api__memory_1ga659960c63f701a0608368e89c5c4ab04>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`operator !=<doxid-group__dnnl__api__memory_1gaca9a006590333e3764895a66f0e1a3f2>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`operator ==<doxid-group__dnnl__api__memory_1ga3ae4b6e7ef0bf507b64d875a7c24ae7e>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);
	bool :target:`operator !=<doxid-group__dnnl__api__memory_1ga6806a5794c45a09b5a9948a5628ffc34>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_verbose<doxid-group__dnnl__api__service_1ga37bcab6f832df551a9fc418e48743b15>`(int level);
	const :ref:`version_t<doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af>`* :ref:`version<doxid-group__dnnl__api__service_1gaad8292408620d0296f22bdf65afb752d>`();
	:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` :ref:`get_default_fpmath_mode<doxid-group__dnnl__api__service_1ga782a2388fc46e80deac409110886db75>`();
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_default_fpmath_mode<doxid-group__dnnl__api__service_1ga0d55da5f92d60a7324cfdc97004ad975>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_jit_dump<doxid-group__dnnl__api__service_1ga2344639528a341878d2ce46fe1c1ac83>`(int enable);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_jit_profiling_flags<doxid-group__dnnl__api__service_1ga966c54ccb1d9ff33d20c4ea47e34675d>`(unsigned flags);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_jit_profiling_jitdumpdir<doxid-group__dnnl__api__service_1ga533341aaf1402e27d1225d1a59819a62>`(const std::string& dir);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_max_cpu_isa<doxid-group__dnnl__api__service_1ga08734310b5f1ca794c64b6a5b944b698>`(:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` isa);
	:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` :ref:`get_effective_cpu_isa<doxid-group__dnnl__api__service_1ga3953f71c3f0126d9cc005a1ceff65e8b>`();
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`set_cpu_isa_hints<doxid-group__dnnl__api__service_1ga29aa5fb708d803e091ac61dc67f9e6ed>`(:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` isa_hints);
	:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` :ref:`get_cpu_isa_hints<doxid-group__dnnl__api__service_1ga8bee13aa79a9711489b401e9c4252ff2>`();
	void :ref:`reset_profiling<doxid-group__dnnl__api__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(:ref:`stream<doxid-structdnnl_1_1stream>`& stream);

	std::vector<uint64_t> :ref:`get_profiling_data<doxid-group__dnnl__api__profiling_1ga0dc451b94cbeacb7a5e0c73c3071ee4e>`(
		:ref:`stream<doxid-structdnnl_1_1stream>`& stream,
		:ref:`profiling_data_kind<doxid-group__dnnl__api__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>` data_kind
		);

	int :ref:`get_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1gacc0f23351595504f3e2c2b6fcf603770>`();
	void :ref:`set_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1ga12eefad64ac6917a161994c005abe69c>`(int capacity);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`sgemm<doxid-group__dnnl__api__blas_1gace5cc61273dc46ccd9c08eee76d4057b>`(
		char transa,
		char transb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const float* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		const float* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		float beta,
		float* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`gemm_u8s8s32<doxid-group__dnnl__api__blas_1ga454c26361de7d3a29f6e23c641380fb0>`(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const uint8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		uint8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`gemm_s8s8s32<doxid-group__dnnl__api__blas_1ga6bb7da88545097f097bbcd5778787826>`(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const int8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		int8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co
		);

	:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` :ref:`convert_to_c<doxid-group__dnnl__api__engine_1gae472e59f404ba6527988b046ef24c743>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind);
	:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` :ref:`convert_to_c<doxid-group__dnnl__api__fpmath__mode_1gad095d0686c7020ce49be483cb44e8535>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);

	} // namespace dnnl
