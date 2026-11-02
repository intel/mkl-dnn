.. _global:
.. index:: pair: namespace; global

Global Namespace
================

.. toctree::
	:hidden:

	namespace_std.rst
	namespace_sycl.rst
	struct_args_t.rst
	struct_cpu_deletor.rst
	struct_example_allows_unimplemented.rst
	struct_gemm_dims_t.rst
	struct_sycl_deletor.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl<doxid-namespacednnl>`;
		namespace :ref:`dnnl::graph<doxid-namespacednnl_1_1graph>`;
			namespace :ref:`dnnl::graph::sycl_interop<doxid-namespacednnl_1_1graph_1_1sycl__interop>`;
		namespace :ref:`dnnl::ocl_interop<doxid-namespacednnl_1_1ocl__interop>`;
		namespace :ref:`dnnl::sycl_interop<doxid-namespacednnl_1_1sycl__interop>`;
		namespace :ref:`dnnl::threadpool_interop<doxid-namespacednnl_1_1threadpool__interop>`;
	namespace :ref:`oneapi<doxid-namespaceoneapi>`;
	namespace :ref:`std<doxid-namespacestd>`;
	namespace :ref:`sycl<doxid-namespacesycl>`;

	// typedefs

	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-bnorm__u8__via__binary__postops_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-bnorm__u8__via__binary__postops_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-cpu__matmul__csr_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-cpu__matmul__csr_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`dnnl::memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :target:`dim_t<doxid-cpu__rnn__inference__f32_8cpp_1abe6d07841e63224fa50aeb0bca16bece>`;
	typedef :ref:`dnnl::memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :target:`dim_t<doxid-cpu__rnn__inference__int8_8cpp_1abe6d07841e63224fa50aeb0bca16bece>`;
	typedef :ref:`logical_tensor::data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` :target:`data_type<doxid-cpu__getting__started_8cpp_1a35af718e2cb46e555772662cef6435c2>`;
	typedef :ref:`logical_tensor::layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` :target:`layout_type<doxid-cpu__getting__started_8cpp_1ac1b25e3cc344221674ec31350ae4ad66>`;
	typedef :ref:`logical_tensor::dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>` :target:`dim<doxid-cpu__getting__started_8cpp_1ad83b61f43e3156ef3292c573495099d2>`;
	typedef :ref:`logical_tensor::dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` :target:`dims<doxid-cpu__getting__started_8cpp_1a47c77b261e3284fd1ade12bb3a418347>`;
	typedef :ref:`logical_tensor::data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` :target:`data_type<doxid-cpu__inference__int8_8cpp_1a35af718e2cb46e555772662cef6435c2>`;
	typedef :ref:`logical_tensor::layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` :target:`layout_type<doxid-cpu__inference__int8_8cpp_1ac1b25e3cc344221674ec31350ae4ad66>`;
	typedef :ref:`logical_tensor::property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` :target:`property_type<doxid-cpu__inference__int8_8cpp_1a4843da079c19b156c5d61734c9a05bff>`;
	typedef :ref:`logical_tensor::dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>` :target:`dim<doxid-cpu__inference__int8_8cpp_1ad83b61f43e3156ef3292c573495099d2>`;
	typedef :ref:`logical_tensor::dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` :target:`dims<doxid-cpu__inference__int8_8cpp_1a47c77b261e3284fd1ade12bb3a418347>`;
	typedef :ref:`logical_tensor::data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` :target:`data_type<doxid-sycl__getting__started_8cpp_1a35af718e2cb46e555772662cef6435c2>`;
	typedef :ref:`logical_tensor::layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` :target:`layout_type<doxid-sycl__getting__started_8cpp_1ac1b25e3cc344221674ec31350ae4ad66>`;
	typedef :ref:`logical_tensor::dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>` :target:`dim<doxid-sycl__getting__started_8cpp_1ad83b61f43e3156ef3292c573495099d2>`;
	typedef :ref:`logical_tensor::dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` :target:`dims<doxid-sycl__getting__started_8cpp_1a47c77b261e3284fd1ade12bb3a418347>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-matmul__perf_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-matmul__perf_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-augru_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-augru_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-batch__normalization_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-batch__normalization_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-binary_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-binary_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-concat_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-concat_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-convolution_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-convolution_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-eltwise_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-eltwise_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-group__normalization_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-group__normalization_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-inner__product_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-inner__product_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-layer__normalization_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-layer__normalization_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-lrn_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-lrn_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-lstm_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-lstm_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-matmul_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-matmul_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-pooling_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-pooling_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-prelu_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-prelu_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-reduction_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-reduction_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-reorder_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-reorder_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-resampling_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-resampling_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-shuffle_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-shuffle_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-softmax_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-softmax_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` :target:`tag<doxid-sum_8cpp_1a41706f4f1dc2c5cd5c3fe0f8a1bd65f1>`;
	typedef :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :target:`dt<doxid-sum_8cpp_1a37404f6ceace52479d5a154fdfbcc3b6>`;
	typedef struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* :ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`;
	typedef const struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* :ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`;
	typedef struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`;
	typedef const struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* :ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>`;
	typedef struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* :ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`;
	typedef const struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`* :ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>`;
	typedef struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* :ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>`;
	typedef const struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`* :ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>`;
	typedef struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* :ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>`;
	typedef const struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`* :ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>`;
	typedef struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* :ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`;
	typedef const struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`* :ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>`;
	typedef int64_t :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`;
	typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>`[DNNL_MAX_NDIMS];
	typedef struct :ref:`dnnl_engine<doxid-structdnnl__engine>`* :ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`;
	typedef struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* :ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`;
	typedef const struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* :ref:`const_dnnl_stream_t<doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36>`;

	typedef void* (*:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>`)(
		size_t size,
		size_t alignment,
		const void *dev,
		const void *context
		);

	typedef void (*:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>`)(
		void *buf,
		const void *dev,
		const void *context,
		void *event
		);

	typedef struct dnnl_graph_partition* :ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`;
	typedef const struct dnnl_graph_partition* :ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>`;
	typedef struct dnnl_graph_graph* :ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`;
	typedef const struct dnnl_graph_graph* :ref:`const_dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5>`;
	typedef struct dnnl_graph_op* :ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>`;
	typedef const struct dnnl_graph_op* :ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>`;

	typedef void* (*:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>`)(
		size_t size,
		size_t alignment
		);

	typedef void (*:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>`)(void *);
	typedef struct dnnl_graph_allocator* :ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`;
	typedef const struct dnnl_graph_allocator* :ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>`;
	typedef struct dnnl_graph_compiled_partition* :ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>`;
	typedef const struct dnnl_graph_compiled_partition* :ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>`;
	typedef struct dnnl_graph_tensor* :ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>`;
	typedef const struct dnnl_graph_tensor* :ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`;

	// enums

	enum :ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>`;
	enum :ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>`;
	enum :ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>`;
	enum :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`;
	enum :ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`;
	enum :ref:`dnnl_format_kind_t<doxid-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d>`;
	enum :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>`;
	enum :ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`;
	enum :ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>`;
	enum :ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>`;
	enum :ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>`;
	enum :ref:`dnnl_graph_partition_policy_t<doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>`;
	enum :ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>`;
	enum :ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>`;
	enum :ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`;
	enum :ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>`;
	enum :ref:`dnnl_profiling_data_kind_t<doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>`;
	enum :ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`;
	enum :ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>`;
	enum :ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>`;
	enum :ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>`;
	enum :ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>`;
	enum :ref:`dnnl_sparse_encoding_t<doxid-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552>`;
	enum :ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>`;
	enum :ref:`dnnl_stream_flags_t<doxid-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547>`;
	enum :ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`;

	// structs

	struct :ref:`args_t<doxid-structargs__t>`;
	struct :ref:`cpu_deletor<doxid-structcpu__deletor>`;
	struct :ref:`dnnl_engine<doxid-structdnnl__engine>`;
	struct :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`;
	struct :ref:`dnnl_graph_inplace_pair_t<doxid-structdnnl__graph__inplace__pair__t>`;
	struct :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`;
	struct :ref:`dnnl_memory<doxid-structdnnl__memory>`;
	struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`;
	struct :ref:`dnnl_post_ops<doxid-structdnnl__post__ops>`;
	struct :ref:`dnnl_primitive<doxid-structdnnl__primitive>`;
	struct :ref:`dnnl_primitive_attr<doxid-structdnnl__primitive__attr>`;
	struct :ref:`dnnl_primitive_desc<doxid-structdnnl__primitive__desc>`;
	struct :ref:`dnnl_stream<doxid-structdnnl__stream>`;
	struct :ref:`dnnl_version_t<doxid-structdnnl__version__t>`;
	struct :ref:`example_allows_unimplemented<doxid-structexample__allows__unimplemented>`;
	struct :ref:`gemm_dims_t<doxid-structgemm__dims__t>`;
	struct :ref:`sycl_deletor<doxid-structsycl__deletor>`;

	// global variables

	const dim_t :target:`batch<doxid-cpu__rnn__inference__f32_8cpp_1aba72b499a97ba8a3b4c60d5efa41ebce>` = 32;
	const dim_t :target:`src_seq_length_max<doxid-cpu__rnn__inference__f32_8cpp_1a6787fdac1e32b13a0ec52a250f418f63>` = 10;
	const dim_t :target:`tgt_seq_length_max<doxid-cpu__rnn__inference__f32_8cpp_1a6b0f831026cfa28f413c667134a0a9c8>` = 10;
	const dim_t :target:`feature_size<doxid-cpu__rnn__inference__f32_8cpp_1a815e1583719d9f32ae976a5bd6a80d3e>` = 256;
	const dim_t :target:`enc_bidir_n_layers<doxid-cpu__rnn__inference__f32_8cpp_1a8cb06a347b6fa940450974c4f4d171d2>` = 1;
	const dim_t :target:`enc_unidir_n_layers<doxid-cpu__rnn__inference__f32_8cpp_1a6fd477fbad7669457d93cef44f1452a8>` = 3;
	const dim_t :target:`dec_n_layers<doxid-cpu__rnn__inference__f32_8cpp_1a32930a645258d837d32b07d2c9746d62>` = 4;
	const int :target:`lstm_n_gates<doxid-cpu__rnn__inference__f32_8cpp_1acdf8c8c42a0270e53979b62a3dc786e1>` = 4;
	const dim_t :target:`batch<doxid-cpu__rnn__inference__int8_8cpp_1aba72b499a97ba8a3b4c60d5efa41ebce>` = 32;
	const dim_t :target:`src_seq_length_max<doxid-cpu__rnn__inference__int8_8cpp_1a6787fdac1e32b13a0ec52a250f418f63>` = 10;
	const dim_t :target:`tgt_seq_length_max<doxid-cpu__rnn__inference__int8_8cpp_1a6b0f831026cfa28f413c667134a0a9c8>` = 10;
	const dim_t :target:`feature_size<doxid-cpu__rnn__inference__int8_8cpp_1a815e1583719d9f32ae976a5bd6a80d3e>` = 256;
	const dim_t :target:`enc_bidir_n_layers<doxid-cpu__rnn__inference__int8_8cpp_1a8cb06a347b6fa940450974c4f4d171d2>` = 1;
	const dim_t :target:`enc_unidir_n_layers<doxid-cpu__rnn__inference__int8_8cpp_1a6fd477fbad7669457d93cef44f1452a8>` = 3;
	const dim_t :target:`dec_n_layers<doxid-cpu__rnn__inference__int8_8cpp_1a32930a645258d837d32b07d2c9746d62>` = 4;
	const int :target:`lstm_n_gates<doxid-cpu__rnn__inference__int8_8cpp_1acdf8c8c42a0270e53979b62a3dc786e1>` = 4;
	static const int :target:`min_runs<doxid-matmul__perf_8cpp_1ae659a2a0d78779a2018a1939ff3c9374>` = 4;
	const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :target:`strides<doxid-performance__profiling_8cpp_1aac67f1001820199454b28acba2aa5e9d>` = {4, 4};
	const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :target:`padding<doxid-performance__profiling_8cpp_1a5328abfd6a143e8f1fed722beff88763>` = {0, 0};
	const int :target:`N0<doxid-rnn__training__f32_8cpp_1a34916025112975a1be34a43569d389dd>` = 1 + rand() % 31;
	const int :target:`N1<doxid-rnn__training__f32_8cpp_1a12f16ce85fbf1ca612dc0c83c2191f02>` = 1 + rand() % 31;
	const int :target:`T0<doxid-rnn__training__f32_8cpp_1ac6f92638257fcff33a3dd08567a17133>` = 31 + 1 + rand() % 31;
	const int :target:`T1<doxid-rnn__training__f32_8cpp_1a7ae15770f11a8d4a70d305db8304226e>` = 1 + rand() % 31;
	const int :target:`leftmost_batch<doxid-rnn__training__f32_8cpp_1a6e491183a7fd67a7f21856cd60dd8409>` = N0 + N1;
	const int :target:`rightmost_batch<doxid-rnn__training__f32_8cpp_1aa3325e18617d0c6c7b23f620e3ee19b2>` = N0;
	const int :target:`leftmost_seq_length<doxid-rnn__training__f32_8cpp_1a0e84936696cbba7e0cbe48a318340ed5>` = T1;
	const int :target:`rightmost_seq_length<doxid-rnn__training__f32_8cpp_1a43107a206457529180e4c4b57de8267a>` = T0 - T1;
	const int :target:`common_feature_size<doxid-rnn__training__f32_8cpp_1a387fec44825dc0a163a1e24c04bbe6b8>` = 1024;
	const int :target:`common_n_layers<doxid-rnn__training__f32_8cpp_1add7b4afce9f835f7915ffb4907073439>` = 1;
	const int :target:`lstm_n_gates<doxid-rnn__training__f32_8cpp_1acdf8c8c42a0270e53979b62a3dc786e1>` = 4;
	:ref:`engine<doxid-structdnnl_1_1engine>` :target:`eng<doxid-cpu__matmul__quantization_8cpp_1aa2ac17ef2c8d2c6de96c9446517d425c>`(engine::kind::cpu, 0);
	int :target:`number_of_runs<doxid-cpu__sgemm__and__matmul_8cpp_1afeb16b6f12b370b3c0e69e1e40c00e57>` = 1;
	float :target:`fixed_beta<doxid-cpu__sgemm__and__matmul_8cpp_1ae3f21041e3deee9dec940ff4a85c8b51>` = 0.f;
	:ref:`engine<doxid-structdnnl_1_1engine>` :target:`eng<doxid-cpu__sgemm__and__matmul_8cpp_1aa2ac17ef2c8d2c6de96c9446517d425c>`(engine::kind::cpu, 0);
	int :target:`number_of_runs<doxid-inference__int8__matmul_8cpp_1afeb16b6f12b370b3c0e69e1e40c00e57>` = 1;

	// global functions

	void :target:`bnorm_u8_via_binary_postops<doxid-bnorm__u8__via__binary__postops_8cpp_1aba296ca2dfa39e8f441d4eb992d05a5e>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-bnorm__u8__via__binary__postops_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	static size_t :target:`product<doxid-cnn__inference__f32_8c_1a051c2cc71ec4b893da92e43993a136d7>`(:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* arr, size_t size);
	static void :target:`init_net_data<doxid-cnn__inference__f32_8c_1a91ca459aaabb5ab5bf474c61dda70575>`(float* data, uint32_t dim, const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* dims);
	static void :target:`prepare_arg_node<doxid-cnn__inference__f32_8c_1ace227aeb69784c8121f1dc68254381b4>`(args_t* node, int nargs);
	static void :target:`free_arg_node<doxid-cnn__inference__f32_8c_1a64a92c6f07a4924ac8231fcddfec3f98>`(args_t* node);
	static void :target:`set_arg<doxid-cnn__inference__f32_8c_1a97c83463b671846b29114fdc08ddbb80>`(:ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* arg, int arg_idx, :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory);

	static void :target:`init_data_memory<doxid-cnn__inference__f32_8c_1a18adbb616e54de7932c1bbe31d5e9a43>`(
		uint32_t dim,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* dims,
		:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` user_tag,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		float* data,
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` :target:`prepare_reorder<doxid-cnn__inference__f32_8c_1a392827838f12b70d729ff9bef12f3b3e>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* user_memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` prim_memory_md,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` prim_engine,
		int dir_is_user_to_prim,
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* prim_memory,
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* reorder,
		uint32_t* net_index,
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* net,
		args_t* net_args
		);

	void :target:`simple_net<doxid-cnn__inference__f32_8c_1ab350fd0c643f532a51123eb3c7379737>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind);
	int :target:`main<doxid-cnn__inference__f32_8c_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`cnn_inference_f32<doxid-cnn__inference__f32_8cpp_1ae5ed9f925026f00a67eb7cff5fbe6051>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-cnn__inference__f32_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	int :target:`main<doxid-cnn__inference__int8_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`simple_net<doxid-cnn__training__bf16_8cpp_1a4b58be5644e1d66a61d7dd52c34fd15d>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-cnn__training__bf16_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`simple_net<doxid-cnn__training__f32_8cpp_1a4b58be5644e1d66a61d7dd52c34fd15d>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-cnn__training__f32_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	static size_t :target:`product<doxid-cpu__cnn__training__f32_8c_1a051c2cc71ec4b893da92e43993a136d7>`(:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* arr, size_t size);
	static void :target:`init_net_data<doxid-cpu__cnn__training__f32_8c_1a91ca459aaabb5ab5bf474c61dda70575>`(float* data, uint32_t dim, const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* dims);
	static void :target:`prepare_arg_node<doxid-cpu__cnn__training__f32_8c_1ace227aeb69784c8121f1dc68254381b4>`(args_t* node, int nargs);
	static void :target:`free_arg_node<doxid-cpu__cnn__training__f32_8c_1a64a92c6f07a4924ac8231fcddfec3f98>`(args_t* node);
	static void :target:`set_arg<doxid-cpu__cnn__training__f32_8c_1a97c83463b671846b29114fdc08ddbb80>`(:ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* arg, int arg_idx, :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory);

	static void :target:`init_data_memory<doxid-cpu__cnn__training__f32_8c_1a18adbb616e54de7932c1bbe31d5e9a43>`(
		uint32_t dim,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* dims,
		:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` user_tag,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		float* data,
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` :target:`prepare_reorder<doxid-cpu__cnn__training__f32_8c_1a392827838f12b70d729ff9bef12f3b3e>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* user_memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` prim_memory_md,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` prim_engine,
		int dir_is_user_to_prim,
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* prim_memory,
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* reorder,
		uint32_t* net_index,
		:ref:`dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga226afc85f63f0d1029d4ea90a60cae47>`* net,
		args_t* net_args
		);

	void :target:`simple_net<doxid-cpu__cnn__training__f32_8c_1a6c4702530e1ed6bb9906e0a917451c5f>`();
	int :target:`main<doxid-cpu__cnn__training__f32_8c_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	bool :target:`check_result<doxid-cpu__matmul__csr_8cpp_1abc51d6890c7e52c289686153e59fa15f>`(:ref:`dnnl::memory<doxid-structdnnl_1_1memory>` dst_mem);
	void :target:`sparse_matmul<doxid-cpu__matmul__csr_8cpp_1a64d87435e0fecd631f542bab4d86174b>`();
	int :target:`main<doxid-cpu__matmul__csr_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	std::vector<float> :target:`weighted_src_layer<doxid-cpu__rnn__inference__f32_8cpp_1a36ae7629c9d8ec5beae5bcddc3549c61>`(batch* feature_size, 1. 0f);

	std::vector<float> :target:`alignment_model<doxid-cpu__rnn__inference__f32_8cpp_1aced50dc397754c6be276171447c8850e>`(
		src_seq_length_max*batch* feature_size,
		1. 0f
		);

	std::vector<float> :target:`alignments<doxid-cpu__rnn__inference__f32_8cpp_1adde713557ca69803b66c45d330ae4df3>`(src_seq_length_max* batch, 1. 0f);
	std::vector<float> :target:`exp_sums<doxid-cpu__rnn__inference__f32_8cpp_1ac3d76cab855498ec992fd87e3c848308>`(batch, 1. 0f);

	void :target:`compute_weighted_annotations<doxid-cpu__rnn__inference__f32_8cpp_1ae1ee562c5b9dff54077009d71f82a15f>`(
		float* weighted_annotations,
		dim_t src_seq_length_max,
		dim_t batch,
		dim_t feature_size,
		float* weights_annot,
		float* annotations
		);

	void :target:`compute_attention<doxid-cpu__rnn__inference__f32_8cpp_1ae84ca1fe41919a0e975af4623c183812>`(
		float* context_vectors,
		dim_t src_seq_length_max,
		dim_t batch,
		dim_t feature_size,
		float* weights_src_layer,
		float* dec_src_layer,
		float* annotations,
		float* weighted_annotations,
		float* weights_alignments
		);

	void :target:`copy_context<doxid-cpu__rnn__inference__f32_8cpp_1a566974a519a2eb76db29ca16276a8b21>`(
		float* src_iter,
		dim_t n_layers,
		dim_t batch,
		dim_t feature_size
		);

	int :target:`main<doxid-cpu__rnn__inference__f32_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	std::vector<int32_t> :target:`weighted_src_layer<doxid-cpu__rnn__inference__int8_8cpp_1acd02148d1a452f6624225670df112e88>`(batch* feature_size, 1);

	std::vector<float> :target:`alignment_model<doxid-cpu__rnn__inference__int8_8cpp_1aced50dc397754c6be276171447c8850e>`(
		src_seq_length_max*batch* feature_size,
		1. 0f
		);

	std::vector<float> :target:`alignments<doxid-cpu__rnn__inference__int8_8cpp_1adde713557ca69803b66c45d330ae4df3>`(src_seq_length_max* batch, 1. 0f);
	std::vector<float> :target:`exp_sums<doxid-cpu__rnn__inference__int8_8cpp_1ac3d76cab855498ec992fd87e3c848308>`(batch, 1. 0f);

	void :target:`compute_weighted_annotations<doxid-cpu__rnn__inference__int8_8cpp_1ae1ee562c5b9dff54077009d71f82a15f>`(
		float* weighted_annotations,
		dim_t src_seq_length_max,
		dim_t batch,
		dim_t feature_size,
		float* weights_annot,
		float* annotations
		);

	void :target:`compute_sum_of_rows<doxid-cpu__rnn__inference__int8_8cpp_1a03e7912ebc20d6caff9274a9f2a36423>`(int8_t* a, dim_t rows, dim_t cols, int32_t* a_reduced);

	void :target:`compute_attention<doxid-cpu__rnn__inference__int8_8cpp_1acb29f5a4e13156d5a3603710cffa40bd>`(
		float* context_vectors,
		dim_t src_seq_length_max,
		dim_t batch,
		dim_t feature_size,
		int8_t* weights_src_layer,
		float weights_src_layer_scale,
		int32_t* compensation,
		uint8_t* dec_src_layer,
		float dec_src_layer_scale,
		float dec_src_layer_shift,
		uint8_t* annotations,
		float* weighted_annotations,
		float* weights_alignments
		);

	void :target:`copy_context<doxid-cpu__rnn__inference__int8_8cpp_1a566974a519a2eb76db29ca16276a8b21>`(
		float* src_iter,
		dim_t n_layers,
		dim_t batch,
		dim_t feature_size
		);

	int :target:`main<doxid-cpu__rnn__inference__int8_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	size_t :target:`product<doxid-cross__engine__reorder_8c_1a19fda44ba6e5d2f8cb6c714069df905b>`(int n_dims, const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]);
	void :target:`fill<doxid-cross__engine__reorder_8c_1a8e7fa6fac3ee15f927fab82eb67e6c37>`(:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` mem, int n_dims, const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]);
	int :target:`find_negative<doxid-cross__engine__reorder_8c_1a84d3e2b6fa41330d534d14c6652a6c9e>`(:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` mem, int n_dims, const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]);
	void :target:`cross_engine_reorder<doxid-cross__engine__reorder_8c_1a053ed891c7f326b16491ccac451bfe57>`();
	int :target:`main<doxid-cross__engine__reorder_8c_1ae66f6b31b5ad750f1fe042a706a4e3d4>`();
	void :target:`fill<doxid-cross__engine__reorder_8cpp_1a8ae5f3aef172d27fe6926f28583dac12>`(:ref:`memory<doxid-structdnnl_1_1memory>`& mem, const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims);
	int :target:`find_negative<doxid-cross__engine__reorder_8cpp_1a14c7e3f85f486b8ea46e2ce83dffbd65>`(:ref:`memory<doxid-structdnnl_1_1memory>`& mem, const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims);
	int :target:`main<doxid-cross__engine__reorder_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	static :ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` :target:`validate_engine_kind<doxid-example__utils_8h_1ac15b7e9121d04f26a4c1d210915eaeec>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` akind);
	static :ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` :target:`parse_engine_kind<doxid-example__utils_8h_1a9baf173e8b4ef306da5b7d6732851e47>`(int argc, char** argv);
	static const char* :target:`engine_kind2str_upper<doxid-example__utils_8h_1a37589bd23a44429798ae161c68f2edb7>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind);
	static void :target:`read_from_dnnl_memory<doxid-example__utils_8h_1a01f71364bca959bcf076f65609304e58>`(void* handle, :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` mem);
	static void :target:`write_to_dnnl_memory<doxid-example__utils_8h_1a5ec58a4fb1f2487ab23ac168eeb5a707>`(void* handle, :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` mem);
	void :target:`finalize<doxid-example__utils_8hpp_1a32d626626eee0bc4ade146973f6abb1c>`();
	:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` :target:`validate_engine_kind<doxid-example__utils_8hpp_1a6cd81f7e71732fc39703464932c326b1>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind);
	const char* :target:`engine_kind2str_upper<doxid-example__utils_8hpp_1a093db4ddec9fa9e37ba66518067f4eb6>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind);

	int :target:`handle_example_errors<doxid-example__utils_8hpp_1a2142dfedcc5465a445dbeb02fc2ff90a>`(
		std::initializer_list<:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`> engine_kinds,
		std::function<void()> example
		);

	int :target:`handle_example_errors<doxid-example__utils_8hpp_1a1cb6abbfb923c9c8d8a8a83d21ab1e98>`(
		std::function<void(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`, int, char**)> example,
		:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind,
		int argc,
		char** argv
		);

	int :target:`handle_example_errors<doxid-example__utils_8hpp_1a84b6d79070cf27d9d95a39e54da53194>`(
		std::function<void(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`)> example,
		:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind
		);

	:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` :target:`parse_engine_kind<doxid-example__utils_8hpp_1a10c83c4b5fd8fbb3229b450b7c9e8c3b>`(int argc, char** argv, int extra_args = 0);
	:ref:`dnnl::memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :target:`product<doxid-example__utils_8hpp_1a86f7bb04bf9f6f35db5ab6a57c809d5f>`(const :ref:`dnnl::memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& dims);
	void :target:`read_from_dnnl_memory<doxid-example__utils_8hpp_1a03fc96d461ea0cb65f3944d21d2d51f0>`(void* handle, :ref:`dnnl::memory<doxid-structdnnl_1_1memory>`& mem);
	void :target:`write_to_dnnl_memory<doxid-example__utils_8hpp_1ab651cd9d43b44d859f7e6085dad2eb23>`(void* handle, :ref:`dnnl::memory<doxid-structdnnl_1_1memory>`& mem);
	int :target:`main<doxid-getting__started_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);

	cl_kernel :target:`create_init_opencl_kernel<doxid-gpu__opencl__interop_8cpp_1a6cdd3a45923111c34c6d750dd38d2533>`(
		cl_context ocl_ctx,
		const char* kernel_name,
		const char* ocl_code
		);

	int :target:`main<doxid-gpu__opencl__interop_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	int :target:`main<doxid-cpu__getting__started_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	int :target:`main<doxid-cpu__inference__int8_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);

	void :ref:`set_any_layout<doxid-graph__example__utils_8hpp_1af80343ea471266a6b9747466539e80ee>`(
		const std::vector<:ref:`dnnl::graph::partition<doxid-classdnnl_1_1graph_1_1partition>`>& partitions,
		std::unordered_set<size_t>& id_to_set_any_layout
		);

	void* :target:`sycl_malloc_wrapper<doxid-graph__example__utils_8hpp_1ad2fd2b5dfbdb7e84eaafc3bf6d678fcb>`(
		size_t size,
		size_t alignment,
		const void* dev,
		const void* ctx
		);

	void :target:`sycl_free_wrapper<doxid-graph__example__utils_8hpp_1a7fae837d1a45d33e07a1bde733ddc609>`(
		void* ptr,
		const void* device,
		const void* context,
		void* event
		);

	void :target:`allocate_graph_mem<doxid-graph__example__utils_8hpp_1a816d841f563698c8979639f00a408f79>`(
		std::vector<:ref:`dnnl::graph::tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& tensors,
		const std::vector<:ref:`dnnl::graph::logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& lts,
		std::vector<std::shared_ptr<void>>& data_buffer,
		std::unordered_map<size_t, :ref:`dnnl::graph::tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& global_outputs_ts_map,
		const :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`& eng,
		bool is_input
		);

	void :target:`allocate_sycl_graph_mem<doxid-graph__example__utils_8hpp_1abb40d8190fba222af32b8f47837118cc>`(
		std::vector<:ref:`dnnl::graph::tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& tensors,
		const std::vector<:ref:`dnnl::graph::logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`>& lts,
		std::vector<std::shared_ptr<void>>& data_buffer,
		std::unordered_map<size_t, :ref:`dnnl::graph::tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& global_outputs_ts_map,
		sycl::queue& q,
		const :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`& eng,
		bool is_input
		);

	int :target:`main<doxid-sycl__getting__started_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	const char* :target:`get_type_string<doxid-matmul__perf_8cpp_1a2d431ae442f937df567413b7db38618b>`(:ref:`dt<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type);
	void :target:`print_test_case<doxid-matmul__perf_8cpp_1af4bd60ab88381854ac156caa0441ce16>`(:ref:`dt<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type, gemm_dims_t dims);
	void :target:`fill_random<doxid-matmul__perf_8cpp_1aa6ca9381d7ea0e0662f1816a0ba28c67>`(std::vector<float>& out, bool is_integer);

	double :target:`run_case<doxid-matmul__perf_8cpp_1ac4b5b5ff7aa063c14ddefa388690bfb6>`(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind,
		:ref:`dt<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type,
		gemm_dims_t dims,
		double time_limit = 0.
		);

	void :target:`run<doxid-matmul__perf_8cpp_1ae61ba45d652614f347032b3ff64514b7>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, :ref:`dt<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` type, gemm_dims_t dims, double time_limit);
	void :target:`bad_args<doxid-matmul__perf_8cpp_1a847ab4e9bab0692595af2434ac34f2c1>`();
	void :target:`matmul_perf<doxid-matmul__perf_8cpp_1a81733f5fa707fd50cb81d8b1425d41f1>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, int argc, char** argv);
	int :target:`main<doxid-matmul__perf_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	int :target:`main<doxid-memory__format__propagation_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`init_data<doxid-performance__profiling_8cpp_1a9f1397a2d66af8bead4fefefeab6dac5>`(:ref:`memory<doxid-structdnnl_1_1memory>`& m, float v);
	void :target:`create_and_execute_relu<doxid-performance__profiling_8cpp_1ac84a0e8e8420ee809fe6ef5338ca962c>`(:ref:`memory<doxid-structdnnl_1_1memory>`& data, :ref:`engine<doxid-structdnnl_1_1engine>`& eng, :ref:`stream<doxid-structdnnl_1_1stream>`& s);
	:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>` :target:`create_attr_with_relu_post_op<doxid-performance__profiling_8cpp_1a83c6dc06ed8ef97fb5d77ddbefad845b>`();
	void :target:`performance_profiling<doxid-performance__profiling_8cpp_1a893a841365f691bfb793b06180db087a>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, int argc, char** argv);
	int :target:`main<doxid-performance__profiling_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`augru_example<doxid-augru_8cpp_1a9a6357bd76403c3b95fddcdb7490c770>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-augru_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`batch_normalization_example<doxid-batch__normalization_8cpp_1a80510a5bbbf3499d968c54e8c9c5ac82>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-batch__normalization_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`binary_example<doxid-binary_8cpp_1a74761e7007f2af6e48c8c18f5af19e77>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-binary_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`concat_example<doxid-concat_8cpp_1a3e4428073ea59c340578240d2d151a92>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-concat_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`convolution_example<doxid-convolution_8cpp_1a2b13c40c247e2f459fae8ba569a766d8>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	void :target:`depthwise_convolution_example<doxid-convolution_8cpp_1ad3c2cb627f6db75705e9de99406d4328>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-convolution_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`eltwise_example<doxid-eltwise_8cpp_1aef86b6a931d860543bc77084701e8f4a>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-eltwise_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`group_normalization_example<doxid-group__normalization_8cpp_1a4cac4286db049cf929e64bff2a7ab432>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-group__normalization_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`inner_product_example<doxid-inner__product_8cpp_1aaec91fd167f86ccab9d187186ed795cf>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-inner__product_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`layer_normalization_example<doxid-layer__normalization_8cpp_1ac5c0fb5a965353bb247442af741e69b2>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-layer__normalization_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`lrn_example<doxid-lrn_8cpp_1a1b379cf156ab4bdb4abadd9e53174882>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-lrn_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`lstm_example<doxid-lstm_8cpp_1a03ed31b8a724a58b537aef294804f821>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-lstm_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`matmul_example<doxid-matmul_8cpp_1a04692eec043aa46ca957989d031a7bdc>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-matmul_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`pooling_example<doxid-pooling_8cpp_1a1637500190d9e57e15576389ec32bde7>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-pooling_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`prelu_example<doxid-prelu_8cpp_1ad6e2f6fd9abb05825bdfa599adf9ed85>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-prelu_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`reduction_example<doxid-reduction_8cpp_1a504075381ab0a3d226f8f2e90fc8ce35>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-reduction_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`reorder_example<doxid-reorder_8cpp_1a8651dafa665b98f52a453564cbd7878a>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-reorder_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`resampling_example<doxid-resampling_8cpp_1a995c7ada76a490912dd706e3ee3ef89c>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-resampling_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`shuffle_example<doxid-shuffle_8cpp_1a6b7ffb77b50e72ff0f7b4d93c67e5f45>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-shuffle_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`softmax_example<doxid-softmax_8cpp_1a024aeb0ad2aa7917c38c7c4c14dee02a>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-softmax_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`sum_example<doxid-sum_8cpp_1a776f13223651ac585c84fb5df9c80806>`(:ref:`dnnl::engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-sum_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`simple_net<doxid-rnn__training__f32_8cpp_1a4b58be5644e1d66a61d7dd52c34fd15d>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-rnn__training__f32_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	int :target:`main<doxid-sycl__interop__buffer_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	void :target:`sycl_usm_tutorial<doxid-sycl__interop__usm_8cpp_1a689ce4fc262f484c168fd01823987fd7>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-sycl__interop__usm_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);

	void :target:`quantize<doxid-cpu__matmul__quantization_8cpp_1a9f4ac072ccb99e289622640fdd7016e1>`(
		const std::vector<float>& X_f32,
		float scale_X,
		int32_t zp_X,
		:ref:`memory<doxid-structdnnl_1_1memory>`& X_int_m
		);

	void :target:`f32_matmul_compute<doxid-cpu__matmul__quantization_8cpp_1a3634b1c57a3d841673984e91596aecf8>`(
		int64_t M,
		int64_t N,
		int64_t K,
		const std::vector<float>& A_f32,
		const std::vector<float>& B_f32,
		std::vector<float>& C_f32
		);

	void :target:`dynamic_q10n_matmul<doxid-cpu__matmul__quantization_8cpp_1a972ac8e11ebdc7ce7fdd9663c43a18b6>`(
		int64_t M,
		int64_t N,
		int64_t K,
		const std::vector<float>& A_f32,
		const std::vector<float>& B_f32,
		std::vector<uint8_t>& C_u8,
		float& scale_C,
		int32_t& zp_C
		);

	void :target:`compare_f32_and_quantized_matmuls<doxid-cpu__matmul__quantization_8cpp_1ae796a67e38004b93840e0ffb4be50925>`();
	int :target:`main<doxid-cpu__matmul__quantization_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	:ref:`matmul<doxid-structdnnl_1_1matmul>` :target:`dynamic_matmul_create<doxid-cpu__sgemm__and__matmul_8cpp_1a5d9c99bbb372aadff087584abaf121a9>`();

	void :target:`dynamic_matmul_execute<doxid-cpu__sgemm__and__matmul_8cpp_1a0a5a6ae793e6ff05d44a6e86509aa9ca>`(
		:ref:`matmul<doxid-structdnnl_1_1matmul>`& matmul_p,
		char transA,
		char transB,
		int64_t M,
		int64_t N,
		int64_t K,
		float alpha,
		const float* A,
		int64_t lda,
		const float* B,
		int64_t ldb,
		float beta,
		float* C,
		int64_t ldc
		);

	void :target:`sgemm_and_matmul_with_params<doxid-cpu__sgemm__and__matmul_8cpp_1a37f872782639bb87fe5dc8a85e89d08a>`(
		char transA,
		char transB,
		int64_t M,
		int64_t N,
		int64_t K,
		float alpha,
		float beta
		);

	void :target:`sgemm_and_matmul<doxid-cpu__sgemm__and__matmul_8cpp_1a0532a344f88b2d31afc1dbb2bbc2c26f>`();
	int :target:`main<doxid-cpu__sgemm__and__matmul_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
	:ref:`matmul::primitive_desc<doxid-structdnnl_1_1matmul_1_1primitive__desc>` :target:`matmul_pd_create<doxid-inference__int8__matmul_8cpp_1a4337d03a3bc12a715b7d546a0bc4a4c8>`(int64_t K, int64_t N, const :ref:`engine<doxid-structdnnl_1_1engine>`& eng);

	void :target:`prepare_input<doxid-inference__int8__matmul_8cpp_1a680d84f6bb1babaa3b41830a6748e5e6>`(
		:ref:`memory<doxid-structdnnl_1_1memory>`& A_u8_mem,
		:ref:`memory<doxid-structdnnl_1_1memory>`& sc_A_mem,
		:ref:`memory<doxid-structdnnl_1_1memory>`& sc_B_mem,
		:ref:`memory<doxid-structdnnl_1_1memory>`& sc_C_mem,
		:ref:`memory<doxid-structdnnl_1_1memory>`& zp_A_mem,
		:ref:`memory<doxid-structdnnl_1_1memory>`& zp_C_mem
		);

	void :target:`sanity_check<doxid-inference__int8__matmul_8cpp_1a9576c673085bab126bc93490c5b363de>`(:ref:`memory<doxid-structdnnl_1_1memory>`& C_u8_mem, :ref:`memory<doxid-structdnnl_1_1memory>`& zp_C_mem);

	void :target:`infer<doxid-inference__int8__matmul_8cpp_1a5581b1369b40f1fd6ad25b255caa50d8>`(
		const :ref:`matmul<doxid-structdnnl_1_1matmul>`& matmul_p,
		int64_t M,
		int64_t N,
		int64_t K,
		const :ref:`memory<doxid-structdnnl_1_1memory>`& B_s8_mem,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& eng
		);

	void :target:`inference_int8_matmul<doxid-inference__int8__matmul_8cpp_1ae6a32023a986a9a56b6abed4b2166cf2>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
	int :target:`main<doxid-inference__int8__matmul_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627>`(int argc, char** argv);
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_primitive_attr_set_zero_points_mask<doxid-group__dnnl__api__attributes_1ga24e429b5410f5657bc5bdda0a6c5d0a7>`(
		:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr,
		int arg,
		int mask
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_destroy<doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` memory_desc);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_clone<doxid-group__dnnl__api__memory_1ga46bc058f1cabc17a49bedfd2633151f7>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` existing_memory_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_strides<doxid-group__dnnl__api__memory_1ga97217bb7179b751aa52bc867ac0092fd>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_tag<doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` tag
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_csr_encoding<doxid-group__dnnl__api__memory_1gad072492c74c31bbc576b96bea15cb09c>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` indices_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` pointers_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_submemory<doxid-group__dnnl__api__memory_1ga44a99d9ec1dfb2bd80a59e656aebd3b4>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` parent_memory_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` offsets
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_reshape<doxid-group__dnnl__api__memory_1gac6985dc70a545b3aa8415d97b990167b>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_permute_axes<doxid-group__dnnl__api__memory_1ga2d1ffe0e07d0be1ab066ac912edbbade>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		const int* permutation
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_query<doxid-group__dnnl__api__memory_1gacc0b7e295e3e970ba738ad5515d8f837>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		void* result
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_query_v2<doxid-group__dnnl__api__memory_1gad083b8bb9d7bbae44e7e33adbd8234d1>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index,
		void* result
		);

	int DNNL_API :ref:`dnnl_memory_desc_equal<doxid-group__dnnl__api__memory_1gad722c21c9af227ac7dc25c3ab649aae5>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` lhs,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` rhs
		);

	size_t DNNL_API :ref:`dnnl_memory_desc_get_size<doxid-group__dnnl__api__memory_1gae7569a047fdd954866df70f01b63e647>`(:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc);

	size_t DNNL_API :ref:`dnnl_memory_desc_get_size_v2<doxid-group__dnnl__api__memory_1gad8ada49d1107442436109ec1de73f370>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		int index
		);

	size_t DNNL_API :ref:`dnnl_data_type_size<doxid-group__dnnl__api__memory_1ga2016d117865455e5d117173dae1b52cb>`(:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_create<doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_create_v2<doxid-group__dnnl__api__memory_1ga90300ec211ec108950e2c4916d56a78a>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		int nhandles,
		void** handles
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_memory_desc<doxid-group__dnnl__api__memory_1ga82045853279cc76f52672b8172afdaee>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`* memory_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_engine<doxid-group__dnnl__api__memory_1ga583a4a06428de7d6c4251313e57ad814>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_map_data<doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_map_data_v2<doxid-group__dnnl__api__memory_1ga6946e1edc50752e4ff1ae1b67bace1d9>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_unmap_data<doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_unmap_data_v2<doxid-group__dnnl__api__memory_1gadc4c9d7f47a209373bb5db0186136318>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_data_handle<doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_set_data_handle<doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_data_handle_v2<doxid-group__dnnl__api__memory_1ga662d48fd7e6f5c4df0e543eea11d94a7>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_set_data_handle_v2<doxid-group__dnnl__api__memory_1ga5b815baf872121e73204d844b5a0e9fa>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_destroy<doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reorder_primitive_desc_create<doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* reorder_primitive_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` src_engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` dst_engine,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_concat_primitive_desc_create<doxid-group__dnnl__api__concat_1ga1bf9669d55a86d8ac8ff10d3e28f52b8>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* concat_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		int concat_dimension,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sum_primitive_desc_create<doxid-group__dnnl__api__sum_1ga10b304125badf7e33eea8ddead1f2e3e>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* sum_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		const float* scales,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_binary_primitive_desc_create<doxid-group__dnnl__api__binary_1ga50078dffd48c6ebd6f6671b7656f5cdb>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src0_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src1_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_convolution_forward_primitive_desc_create<doxid-group__dnnl__api__convolution_1gab5d114c896caa5c32e0035eaafbd5f40>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_convolution_backward_data_primitive_desc_create<doxid-group__dnnl__api__convolution_1ga182e20bc7eae9df73d186acf869471da>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_convolution_backward_weights_primitive_desc_create<doxid-group__dnnl__api__convolution_1gadfb6988120ff24a0b62d9e8a7443ba09>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_forward_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1gaf0d6b55570014911d30a867e3de12258>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_backward_data_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1ga531dbfb58d4fe4526c96c982dd13780c>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_backward_weights_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1ga45ad0e8c95597f9dc8fc36676cf38f55>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_shuffle_forward_primitive_desc_create<doxid-group__dnnl__api__shuffle_1gaab9289838e10ee76966173e20dd24562>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int axis,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` group_size,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_shuffle_backward_primitive_desc_create<doxid-group__dnnl__api__shuffle_1ga2edc80cc4334ec0aba0c6ad5d9d5bcd6>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		int axis,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` group_size,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_eltwise_forward_primitive_desc_create<doxid-group__dnnl__api__eltwise_1gaf5ae8472e1a364502103dea646ccb5bf>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_eltwise_backward_primitive_desc_create<doxid-group__dnnl__api__eltwise_1gaba11ca62016a1c23d997db47bcd6c27d>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` data_desc,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_softmax_forward_primitive_desc_create<doxid-group__dnnl__api__softmax_1ga4ca3adbc99470d1f4111466dac4d1c76>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int softmax_axis,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_softmax_backward_primitive_desc_create<doxid-group__dnnl__api__softmax_1ga99a72d524ea876adee57dbd91408a83c>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int softmax_axis,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_pooling_forward_primitive_desc_create<doxid-group__dnnl__api__pooling_1ga4921dcd2653e2046ef8de99c354957fe>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` kernel,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilation,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_pooling_backward_primitive_desc_create<doxid-group__dnnl__api__pooling_1ga0f1637d5ab52a8b784e642d6afac9fec>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` kernel,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilation,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_prelu_forward_primitive_desc_create<doxid-group__dnnl__api__prelu_1gaf74409d6e35b9935ee44d355e10e200c>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_prelu_backward_primitive_desc_create<doxid-group__dnnl__api__prelu_1ga525d58d10e3e8c17081f750fdf464a52>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lrn_forward_primitive_desc_create<doxid-group__dnnl__api__lrn_1ga7d2550452cd5858747686b338cfde252>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lrn_backward_primitive_desc_create<doxid-group__dnnl__api__lrn_1gafc38999581f962346f08517ef3383617>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_batch_normalization_forward_primitive_desc_create<doxid-group__dnnl__api__batch__normalization_1ga65dc3c0a16325b360a7e6fb676900b2b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_batch_normalization_backward_primitive_desc_create<doxid-group__dnnl__api__batch__normalization_1ga851cc56a63560c3de6217d6d812d169f>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_group_normalization_forward_primitive_desc_create<doxid-group__dnnl__api__group__normalization_1ga890cb9918ad36f8a939921deaf8bd918>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` groups,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_group_normalization_backward_primitive_desc_create<doxid-group__dnnl__api__group__normalization_1gab815e05899bd4df78aafe5d3c5effa8d>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` groups,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_layer_normalization_forward_primitive_desc_create<doxid-group__dnnl__api__layer__normalization_1ga3f99050f79a43b697bf35cc0f39e21fe>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` stat_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_layer_normalization_backward_primitive_desc_create<doxid-group__dnnl__api__layer__normalization_1gaea99a932365f2e330420fdefabcfcd05>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` stat_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_inner_product_forward_primitive_desc_create<doxid-group__dnnl__api__inner__product_1gad639955af0f0daefd3ea9beda50f7fa8>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_inner_product_backward_data_primitive_desc_create<doxid-group__dnnl__api__inner__product_1gadbb37ee1140b71d8d40aa23054b1d2db>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_inner_product_backward_weights_primitive_desc_create<doxid-group__dnnl__api__inner__product_1ga2924b2a46b5d6e55854b0d785c4f11ae>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_matmul_primitive_desc_create<doxid-group__dnnl__api__matmul_1gaac0ca3eed6070331c7d4020028b00fe6>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_resampling_forward_primitive_desc_create<doxid-group__dnnl__api__resampling_1gad374ec6e08ef55ed8fa3682bfae490d8>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		const float* factors,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_resampling_backward_primitive_desc_create<doxid-group__dnnl__api__resampling_1gab4380bb763c173e0b3b3122132b14868>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		const float* factors,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reduction_primitive_desc_create<doxid-group__dnnl__api__reduction_1gab26d42a8553d69b5fc3fbdb3b44d3f98>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float p,
		float eps,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_get_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1gaaffb070446181187b04ee1a321cc24f0>`(int* capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1ga53456304297195ae9f053cc60ffe70a2>`(int capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_dump<doxid-group__dnnl__api__service_1ga03c8f4af3d01f76060f98e78039837fc>`(int enable);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_flags<doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48>`(unsigned flags);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_jitdumpdir<doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7>`(const char* dir);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_max_cpu_isa<doxid-group__dnnl__api__service_1ga4b7f3b3299482f88f1a0aa61a4707156>`(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` isa);
	:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` DNNL_API :ref:`dnnl_get_effective_cpu_isa<doxid-group__dnnl__api__service_1gac55836cf36bc25f8635e459678303570>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_cpu_isa_hints<doxid-group__dnnl__api__service_1gad078a384ab0e078d81595686efd26ed2>`(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` isa_hints);
	:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` DNNL_API :ref:`dnnl_get_cpu_isa_hints<doxid-group__dnnl__api__service_1gad93f9f4bf3c9e12a2be7337b1e41d145>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reset_profiling<doxid-group__dnnl__api__profiling_1gaaf7e8e00d675e7362ccf75b30a9c47bd>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_query_profiling_data<doxid-group__dnnl__api__profiling_1gae92506d856399892636be1c86a3a94a7>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		:ref:`dnnl_profiling_data_kind_t<doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>` data_kind,
		int* num_entries,
		uint64_t* data
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sgemm<doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8>`(
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gemm_u8s8s32<doxid-group__dnnl__api__blas_1gaef24848fd198d8a178d3ad95a78c1767>`(
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gemm_s8s8s32<doxid-group__dnnl__api__blas_1ga2b763b7629846913507d88fba875cc26>`(
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

	const char DNNL_API* :target:`dnnl_status2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a9dab65af5ac10c52076e972c50e9da87>`(:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` v);
	const char DNNL_API* :target:`dnnl_dt2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a7ea29f2aa1891da026ebdcc60bf605e5>`(:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` v);
	const char DNNL_API* :target:`dnnl_fpmath_mode2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1aa0ed5cee05b0ee974bd7d979a8edd3de>`(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` v);
	const char DNNL_API* :target:`dnnl_engine_kind2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a769355ce0a1d2d0550752595ef077022>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` v);
	const char DNNL_API* :target:`dnnl_sparse_encoding2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a99721b2bf7eb9bb5550c2610dcbdf0e0>`(:ref:`dnnl_sparse_encoding_t<doxid-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552>` v);
	const char DNNL_API* :target:`dnnl_fmt_tag2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a23bb6ca72b5766e4f2c51d4239c586db>`(:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` v);
	const char DNNL_API* :target:`dnnl_prop_kind2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a7da9538b6c1f36d0509ac60957529cac>`(:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` v);
	const char DNNL_API* :target:`dnnl_prim_kind2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1ae25f385a44021f859c0064096256a484>`(:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` v);
	const char DNNL_API* :target:`dnnl_alg_kind2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a5dceb63335ae0b4cea2415c9f2c02380>`(:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` v);
	const char DNNL_API* :target:`dnnl_rnn_flags2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1abeb7c515c9ee0d6958c47aff56ee7d8f>`(:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` v);
	const char DNNL_API* :target:`dnnl_rnn_direction2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a7a68d959687b2f13fdd643e19c1c7f26>`(:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` v);
	const char DNNL_API* :target:`dnnl_scratchpad_mode2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1ada7e755062dc5e629a1f280ca89e6f7a>`(:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` v);
	const char DNNL_API* :target:`dnnl_cpu_isa2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1a5f0919e18cdb4854ae8f0bd5f445a43c>`(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` v);
	const char DNNL_API* :target:`dnnl_cpu_isa_hints2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1aa38d5a0385ce11a2c488b860a2885567>`(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` v);
	const char DNNL_API* :target:`dnnl_runtime2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1acb1cd6b77cd3c98bde161bd12f9333bf>`(unsigned v);
	const char DNNL_API* :target:`dnnl_fmt_kind2str<doxid-oneapi_2dnnl_2dnnl__debug_8h_1ab60d46b19aa422ad0dddf171b1aa75c9>`(:ref:`dnnl_format_kind_t<doxid-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d>` v);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_create<doxid-group__dnnl__api__ocl__interop_1gad5b8aba7d6108ba727505d0db5062342>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` memory_kind,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_get_memory_kind<doxid-group__dnnl__api__ocl__interop_1gaa6c00a54ba3ca30d00d9e1c43e9b4bc2>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`* memory_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_get_mem_object<doxid-group__dnnl__api__ocl__interop_1ga383b09734d764bb45872b2c65f7dad70>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		cl_mem* mem_object
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_set_mem_object<doxid-group__dnnl__api__ocl__interop_1ga493946609c6bae83329241c4950edbce>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		cl_mem mem_object
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_cache_blob_id<doxid-group__dnnl__api__ocl__interop_1gad1e18db981b46c04640dde395f75845c>`(
		cl_device_id device,
		size_t* size,
		uint8_t* cache_blob_id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_cache_blob<doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		size_t* size,
		uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_create_from_cache_blob<doxid-group__dnnl__api__ocl__interop_1gaf4d8ed8673cf2d90a326cb0e66a41ccd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		size_t size,
		const uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_create<doxid-group__dnnl__api__ocl__interop_1ga52edd1810d72a2a08a881b122c7ada70>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_context<doxid-group__dnnl__api__ocl__interop_1ga6be452e1d11ad63d1f20a072258547c8>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_context* context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_get_device<doxid-group__dnnl__api__ocl__interop_1gafd0a653afb5a16d4d1fc71cd0615e44a>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_device_id* device
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_stream_create<doxid-group__dnnl__api__ocl__interop_1ga9a9007c6661472d701b2bbfb43ddf07c>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_command_queue queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_stream_get_command_queue<doxid-group__dnnl__api__ocl__interop_1ga0290e83f9217e83eba910454348c0819>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		cl_command_queue* queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_primitive_execute<doxid-group__dnnl__api__ocl__interop_1ga95b6ba55db1e53a8b530ef38b09d1953>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const cl_event* deps,
		int ndeps,
		cl_event* return_event
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_create<doxid-group__dnnl__api__sycl__interop_1ga7d768ee527493380e13fdf2983b32b70>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_get_context<doxid-group__dnnl__api__sycl__interop_1ga23b777c4d60c3dd9b542126973cb69a5>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_get_device<doxid-group__dnnl__api__sycl__interop_1gaf6a88b22743cc5ca54ad46ee88fbb71e>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** device
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_create<doxid-group__dnnl__api__sycl__interop_1gafe0b9a934268c1954b87475d376d600e>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>` memory_kind,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_get_memory_kind<doxid-group__dnnl__api__sycl__interop_1ga3c24bba041823efb72fd4ce003a4436c>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`* memory_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_set_buffer<doxid-group__dnnl__api__sycl__interop_1ga62c1cb33d766f2035d83b7010db7adf9>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* buffer
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_stream_create<doxid-group__dnnl__api__sycl__interop_1ga64dc62b1586d688afcd110840e570cd5>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_stream_get_queue<doxid-group__dnnl__api__sycl__interop_1gab279fc3922a8b4ab59d4b328f1610172>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		void** queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_primitive_execute<doxid-group__dnnl__api__sycl__interop_1ga49aea2229b2e3afcd66e31ef76fcbe64>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const void* deps,
		void* return_event
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_stream_create<doxid-group__dnnl__api__threadpool__interop_1ga45a92b2adda6ff7a31784d73cbd61c26>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_stream_get_threadpool<doxid-group__dnnl__api__threadpool__interop_1ga1dac9e0e17855a5196cb96295277a86d>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` astream,
		void** threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_set_max_concurrency<doxid-group__dnnl__api__threadpool__interop_1ga39b66c510d4f46fde238fedb4343fa2d>`(int max_concurrency);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_get_max_concurrency<doxid-group__dnnl__api__threadpool__interop_1ga4017c3cc0fe9b66a8d50352015d0dbcf>`(int* max_concurrency);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_sgemm<doxid-group__dnnl__api__threadpool__interop_1ga2726272c8ce83f4231cc81e326336193>`(
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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_gemm_u8s8s32<doxid-group__dnnl__api__threadpool__interop_1gaeb14ed904eaed73cfd01a29bc2a4ac1e>`(
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
		const int32_t* co,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_gemm_s8s8s32<doxid-group__dnnl__api__threadpool__interop_1ga39dd6bf602ca1ebb2039eb4c27f07fdf>`(
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
		const int32_t* co,
		void* threadpool
		);

	size_t DNNL_API :ref:`dnnl_engine_get_count<doxid-group__dnnl__api__engine_1gadff5935622df99a2f89acb5cbea09ab5>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_create<doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_get_kind<doxid-group__dnnl__api__engine_1ga8a38bdce17f51616d03310a8e8764c8c>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_destroy<doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_create<doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		unsigned flags
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_get_engine<doxid-group__dnnl__api__stream_1ga817016eb87a4d87a889f32b52b71a93b>`(
		:ref:`const_dnnl_stream_t<doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36>` stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_wait<doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_destroy<doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_get_default_fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1gada52f7858332a7cda0e0c5e7907056d7>`(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_default_fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga97dd535e43073cee2ebc4b709e42c3ca>`(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_verbose<doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856>`(int level);
	const :ref:`dnnl_version_t<doxid-structdnnl__version__t>` DNNL_API* :ref:`dnnl_version<doxid-group__dnnl__api__service_1ga73e40d184386e9d9ca917756e76fb232>`(void);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_allocator_create<doxid-group__dnnl__graph__api__allocator_1gaac19f3f00e51bdd323be1a9073282fcd>`(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>` host_malloc,
		:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>` host_free
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_allocator_destroy<doxid-group__dnnl__graph__api__allocator_1gad2c3000cd39878198f6e461a30dd42c8>`(:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>` allocator);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_make_engine_with_allocator<doxid-group__dnnl__graph__api__engine_1gacd72ae9dc87f2fab2d155faa2bcf0258>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init<doxid-group__dnnl__graph__api__logical__tensor_1gab18ae5c4f5bfe5bd966305d9c2690a7e>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init_with_dims<doxid-group__dnnl__graph__api__logical__tensor_1ga13f140ecc327c9d8acb5a5832b2d0710>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init_with_strides<doxid-group__dnnl__graph__api__logical__tensor_1ga719f24a5aec5fc929a3ab620d6d5dc97>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_get_mem_size<doxid-group__dnnl__graph__api__logical__tensor_1ga56f57a976b591e6d428daea2f115207c>`(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t* size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_is_equal<doxid-group__dnnl__graph__api__logical__tensor_1gacc21c4aa2240c9a56616259e7ed71df0>`(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt1,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt2,
		uint8_t* is_equal
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_create<doxid-group__dnnl__graph__api__tensor_1gaf1ce44e7c73d38f7dfd2e4f374d341e5>`(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>`* tensor,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_destroy<doxid-group__dnnl__graph__api__tensor_1ga42e42c1059fbb4f86919754d31c5888d>`(:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_get_data_handle<doxid-group__dnnl__graph__api__tensor_1ga39f0b7ce6ba2067dc0a166075abebb16>`(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		void** handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_set_data_handle<doxid-group__dnnl__graph__api__tensor_1ga2b81562df6173e0f2ff1b4360c4cf3ec>`(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_get_engine<doxid-group__dnnl__graph__api__tensor_1ga09d0a460550d1b399a0614c20663f73b>`(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_create<doxid-group__dnnl__graph__api__op_1ga89f9449ddd533e166e3deaf253520ba1>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>`* op,
		size_t id,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>` kind,
		const char* verbose_name
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_destroy<doxid-group__dnnl__graph__api__op_1ga9078b97ce5f2e44cb318d08ff96fe391>`(:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_add_input<doxid-group__dnnl__graph__api__op_1gac1cc01522c2328069e8bd045f563554f>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* input
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_add_output<doxid-group__dnnl__graph__api__op_1gad2ada5d285eb5cc8aa38785585525b3d>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* output
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_f32<doxid-group__dnnl__graph__api__op_1gaa4605432c3cd40570607a40a1448e777>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const float* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_bool<doxid-group__dnnl__graph__api__op_1ga122b16165d16f9e1b36fa04c4df783de>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const uint8_t* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_s64<doxid-group__dnnl__graph__api__op_1gaca7be5242f3fd61421bcc49365129965>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const int64_t* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_str<doxid-group__dnnl__graph__api__op_1gae832731052f5072256527a73326a7d43>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const char* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_get_id<doxid-group__dnnl__graph__api__op_1ga9258f54424d3e9f3e88356982864d1e0>`(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		size_t* id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_get_kind<doxid-group__dnnl__graph__api__op_1ga11559f93efe532d71c0c6284896d8444>`(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>`* kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_create_with_op<doxid-group__dnnl__graph__api__partition_1gade597975f67997d0242315b847e288aa>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partition,
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` ekind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_destroy<doxid-group__dnnl__graph__api__partition_1ga44f173aef7d5c593d305d6abd0927507>`(:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_op_num<doxid-group__dnnl__graph__api__partition_1gaf54ad50ee43f413a0e9bcd2ed3866d30>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_ops<doxid-group__dnnl__graph__api__partition_1ga194ebb49cbf9bcb26f6bd94c202fd76c>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		size_t num,
		size_t* ids
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_id<doxid-group__dnnl__graph__api__partition_1ga4f193dd55464dfb5d74a44e0d06ecba3>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_compile<doxid-group__dnnl__graph__api__partition_1ga0de016808a18bea4d23694dacd438035>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition,
		size_t in_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** inputs,
		size_t out_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** outputs,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_input_ports_num<doxid-group__dnnl__graph__api__partition_1gae68562298df62a9d3fc12042ac2b9ab2>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_input_ports<doxid-group__dnnl__graph__api__partition_1ga32868fe8a784b661c0d2865fa530bbe5>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* inputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_output_ports_num<doxid-group__dnnl__graph__api__partition_1ga574edb86ed4abb2fc129cba4bb66e7c9>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_output_ports<doxid-group__dnnl__graph__api__partition_1gaea3a1581038bc059c14bef77f5034ebd>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* outputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_is_supported<doxid-group__dnnl__graph__api__partition_1ga9899bfedf8d4e3530f76824c318cb0d5>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		uint8_t* is_supported
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_engine_kind<doxid-group__dnnl__graph__api__partition_1ga54a4685d7b3deee2728ea4a6268bc822>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_create<doxid-group__dnnl__graph__api__compiled__partition_1ga86b0ee196722ef06f4525416a7a41e92>`(
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>`* compiled_partition,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_execute<doxid-group__dnnl__graph__api__compiled__partition_1ga34203136d999a80d09dcc24d0a3d2268>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_destroy<doxid-group__dnnl__graph__api__compiled__partition_1gae89b0fccf8e91d7796f304a9f14b8dec>`(:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_query_logical_tensor<doxid-group__dnnl__graph__api__compiled__partition_1ga03a2fbb5505cc60962e05f1cd0e60f6a>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t tid,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_get_inplace_ports<doxid-group__dnnl__graph__api__compiled__partition_1ga5fc8b08404ce7e7063eac908e59c0158>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t* num_inplace_pairs,
		const :ref:`dnnl_graph_inplace_pair_t<doxid-structdnnl__graph__inplace__pair__t>`** inplace_pairs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_create<doxid-group__dnnl__graph__api__graph_1gae0ebc9c6eada0fe52f136b63850e1b4c>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_create_with_fpmath_mode<doxid-group__dnnl__graph__api__graph_1gae13c8c4edc6cd30c96d81329e3973c83>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_destroy<doxid-group__dnnl__graph__api__graph_1gaac9d64ff0a5a010ff8800f70f472e207>`(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_add_op<doxid-group__dnnl__graph__api__graph_1ga69eff2efb6ccf06f27827278a974d5be>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_finalize<doxid-group__dnnl__graph__api__graph_1ga3ef94c50f6451091d549dd7f6e085ff6>`(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_is_finalized<doxid-group__dnnl__graph__api__graph_1gaf850e0710060ffebe1418c3addf1955e>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		uint8_t* finalized
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_filter<doxid-group__dnnl__graph__api__graph_1ga28f03fcd7dcfaac5eb2a3ba08fba3ff0>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_partition_policy_t<doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>` policy
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_get_partition_num<doxid-group__dnnl__graph__api__graph_1ga603d39f0b799244de8b157c2967646d1>`(
		:ref:`const_dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5>` graph,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_get_partitions<doxid-group__dnnl__graph__api__graph_1ga68fdf628f24078ccdd89755cdd090881>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		size_t num,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partitions
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga341079185a3e263dc490a8d24d0fdc94>`(int* capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1gabed28f32f3f39e2b4053c5b53620a292>`(int capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga9e37974d35ff5aafe1cbae2f69a2ab00>`(int flag);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga79be61eb82b59a52145bb730197283c1>`(int* flag);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_allocator_create<doxid-group__dnnl__graph__api__sycl__interop_1ga06e949434a4fc257e1c89185e97593dc>`(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` sycl_malloc,
		:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` sycl_free
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_make_engine_with_allocator<doxid-group__dnnl__graph__api__sycl__interop_1ga84bf2a778aeb99c8134c541ee2b603bd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_compiled_partition_execute<doxid-group__dnnl__graph__api__sycl__interop_1ga7e51f65c06cd550a282db11ee86b8e47>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs,
		const void* deps,
		void* sycl_event
		);

	// macros

	#define :target:`BATCH<doxid-cnn__inference__f32_8c_1ae2daa49a9ae9f39ffbe4177490d37eb3>`
	#define :target:`BATCH<doxid-cpu__cnn__training__f32_8c_1ae2daa49a9ae9f39ffbe4177490d37eb3>`
	#define :target:`CHECK<doxid-example__utils_8h_1aa59a70e2ff40af3af6c08aafdca8c713>`(f)
	#define :target:`COMPLAIN_DNNL_ERROR_AND_EXIT<doxid-example__utils_8h_1a03c1c14b7705587c0e1517b792bcacae>`(what, status)
	#define :target:`COMPLAIN_EXAMPLE_ERROR_AND_EXIT<doxid-example__utils_8h_1ad0303945972130d90fec10b3378ac7b5>`(complain_fmt, ...)
	#define :target:`CONV_IH<doxid-cpu__cnn__training__f32_8c_1aaf549f2459528335302d53f3bf6d27a5>`
	#define :target:`CONV_IH<doxid-cnn__inference__f32_8c_1aaf549f2459528335302d53f3bf6d27a5>`
	#define :target:`CONV_IW<doxid-cpu__cnn__training__f32_8c_1a393d13737a5d2159354679d19cf39167>`
	#define :target:`CONV_IW<doxid-cnn__inference__f32_8c_1a393d13737a5d2159354679d19cf39167>`
	#define :target:`CONV_OH<doxid-cpu__cnn__training__f32_8c_1a72c3f58d67be975a41a8fa513f2b03c3>`
	#define :target:`CONV_OH<doxid-cnn__inference__f32_8c_1a72c3f58d67be975a41a8fa513f2b03c3>`
	#define :target:`CONV_OW<doxid-cnn__inference__f32_8c_1af1cf557b4506fef2670e953c39412520>`
	#define :target:`CONV_OW<doxid-cpu__cnn__training__f32_8c_1af1cf557b4506fef2670e953c39412520>`
	#define :target:`CONV_PAD<doxid-cnn__inference__f32_8c_1a0c0c5268aaef0b1413eb1e8d77768c29>`
	#define :target:`CONV_PAD<doxid-cpu__cnn__training__f32_8c_1a0c0c5268aaef0b1413eb1e8d77768c29>`
	#define :target:`CONV_STRIDE<doxid-cnn__inference__f32_8c_1ae602d4d76a4e0e06d92e85ff0b55e4d8>`
	#define :target:`CONV_STRIDE<doxid-cpu__cnn__training__f32_8c_1ae602d4d76a4e0e06d92e85ff0b55e4d8>`
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
	#define :target:`DNNL_ENABLE_EXCEPTIONS<doxid-dnnl__common_8hpp_1a419f8d5758faa3d80c1ff3085fbfa0f6>`
	#define :ref:`DNNL_GRAPH_UNKNOWN_DIM<doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>`
	#define :ref:`DNNL_GRAPH_UNKNOWN_NDIMS<doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP<doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC<doxid-group__dnnl__api__service_1ga66a48a940ab2916d360b0bb677a70e5f>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_PERF<doxid-group__dnnl__api__service_1ga5a1d61af9d5b15dbc6d7d33f0f3e22bc>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_PERFMAP<doxid-group__dnnl__api__service_1gacb5b174589525cce34589ef4ef56462f>`
	#define :ref:`DNNL_JIT_PROFILE_NONE<doxid-group__dnnl__api__service_1ga7ceacd6430988ed4bf58f5b01cd9c5a4>`
	#define :ref:`DNNL_JIT_PROFILE_VTUNE<doxid-group__dnnl__api__service_1ga137013d98ef736973ebbe1ecd4a4b2c9>`
	#define :ref:`DNNL_MAX_NDIMS<doxid-group__dnnl__api__data__types_1gaa9e648b617df0f0186143abdf78ca5f2>`
	#define :ref:`DNNL_MEMORY_ALLOCATE<doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`
	#define :ref:`DNNL_MEMORY_NONE<doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`
	#define :ref:`DNNL_RUNTIME_DIM_VAL<doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`
	#define :ref:`DNNL_RUNTIME_F32_VAL<doxid-group__dnnl__api__memory_1gab16365c11b4dc88fbb453edb51f1979f>`
	#define :ref:`DNNL_RUNTIME_S32_VAL<doxid-group__dnnl__api__memory_1ga30139d5110e9e895ccd93fe503ca4c35>`
	#define :ref:`DNNL_RUNTIME_SIZE_VAL<doxid-group__dnnl__api__memory_1ga61466fbd352b6c94b6541977fbe199b8>`
	#define :target:`DNNL_THROW_ERROR<doxid-dnnl__common_8hpp_1af2801b56d6f53583745aee91430b411d>`(status, msg)
	#define :target:`IC<doxid-cnn__inference__f32_8c_1a9a554ab1c3840c9c77f6476f776232c2>`
	#define :target:`IC<doxid-cpu__cnn__training__f32_8c_1a9a554ab1c3840c9c77f6476f776232c2>`
	#define :target:`OC<doxid-cpu__cnn__training__f32_8c_1a9742ab6c30cea70a694a7eace4d161b3>`
	#define :target:`OC<doxid-cnn__inference__f32_8c_1a9742ab6c30cea70a694a7eace4d161b3>`
	#define :target:`OCL_CHECK<doxid-gpu__opencl__interop_8cpp_1a332d012f5b4383f379415c79953479da>`(x)
	#define :target:`POOL_OH<doxid-cnn__inference__f32_8c_1af1e16083d3b286d1428230524b4195c5>`
	#define :target:`POOL_OH<doxid-cpu__cnn__training__f32_8c_1af1e16083d3b286d1428230524b4195c5>`
	#define :target:`POOL_OW<doxid-cnn__inference__f32_8c_1af83c4d4357133c09cf0f2ff2907c82bd>`
	#define :target:`POOL_OW<doxid-cpu__cnn__training__f32_8c_1af83c4d4357133c09cf0f2ff2907c82bd>`
	#define :target:`POOL_PAD<doxid-cnn__inference__f32_8c_1ad5ff69a3a1ede1c265ee82b4689eb6d5>`
	#define :target:`POOL_PAD<doxid-cpu__cnn__training__f32_8c_1ad5ff69a3a1ede1c265ee82b4689eb6d5>`
	#define :target:`POOL_STRIDE<doxid-cpu__cnn__training__f32_8c_1a17860df9568182c2ee4b2eacfb425bd1>`
	#define :target:`POOL_STRIDE<doxid-cnn__inference__f32_8c_1a17860df9568182c2ee4b2eacfb425bd1>`
	#define :target:`PRAGMA_MACRO<doxid-example__utils_8hpp_1a819667b23022cf8a198a0c226302fe75>`(x)
	#define :target:`PRAGMA_MACRo<doxid-example__utils_8hpp_1a89583a0afaeaffc570d06411e1bbd99c>`(x)
	#define :target:`PRAGMA_OMP_PARALLEL_FOR_COLLAPSE<doxid-example__utils_8hpp_1a7dc66745f119a23530ade477e5778e84>`(n)
	#define :target:`TYPE_CASE<doxid-matmul__perf_8cpp_1a938ca528442b632c4ac86e2fb08badb9>`(T)
	#define :target:`UNUSED<doxid-graph__example__utils_8hpp_1a86d500a34c624c2cae56bc25a31b12f3>`(x)
	#define :target:`_POSIX_C_SOURCE<doxid-cpu__cnn__training__f32_8c_1a3024ccd4a9af5109d24e6c57565d74a1>`
	#define :target:`_POSIX_C_SOURCE<doxid-cnn__inference__f32_8c_1a3024ccd4a9af5109d24e6c57565d74a1>`

.. _details-global:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; set_any_layout
.. _doxid-graph__example__utils_8hpp_1af80343ea471266a6b9747466539e80ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_any_layout(
		const std::vector<:ref:`dnnl::graph::partition<doxid-classdnnl_1_1graph_1_1partition>`>& partitions,
		std::unordered_set<size_t>& id_to_set_any_layout
		)

Set any layout according to the connection relationship of partitions.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partitions

		- a list of partitions

	*
		- id_to_set_any_layout

		- a set of ids of logical tensors with any layout type

