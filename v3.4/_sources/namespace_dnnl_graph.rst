.. index:: pair: namespace; dnnl::graph
.. _doxid-namespacednnl_1_1graph:

namespace dnnl::graph
=====================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace graph {

	// namespaces

	namespace :ref:`dnnl::graph::sycl_interop<doxid-namespacednnl_1_1graph_1_1sycl__interop>`;

	// enums

	enum :ref:`status<doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>`;

	// classes

	class :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`;
	class :ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`;
	class :ref:`graph<doxid-classdnnl_1_1graph_1_1graph>`;
	class :ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`;
	class :ref:`op<doxid-classdnnl_1_1graph_1_1op>`;
	class :ref:`partition<doxid-classdnnl_1_1graph_1_1partition>`;
	class :ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`;

	// global functions

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine_with_allocator<doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind,
		size_t index,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		);

	int :ref:`get_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga6a1bba962b6499e55ba1fb81692d04b7>`();
	void :ref:`set_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga2260760847caf233f319967e73811319>`(int capacity);
	void :ref:`set_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1gad0a7165aed94d587fb232b0e7e4783ce>`(int flag);
	int :ref:`get_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1gaad7e9879171c5825df486370f73b8cf2>`();
	void :ref:`set_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1gafb2292cee6b5833286a3c19bad4ee93e>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind, size_t size);
	size_t :ref:`get_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga737fd9a92da083bc778f43752f8b3ffb>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind);

	} // namespace graph
