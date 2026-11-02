.. index:: pair: group; Engine
.. _doxid-group__dnnl__graph__api__engine:

Engine
======

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`dnnl::graph::make_engine_with_allocator<doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind,
		size_t index,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_make_engine_with_allocator<doxid-group__dnnl__graph__api__engine_1gacd72ae9dc87f2fab2d155faa2bcf0258>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		);

.. _details-group__dnnl__graph__api__engine:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; make_engine_with_allocator
.. _doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` dnnl::graph::make_engine_with_allocator(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind,
		size_t index,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		)

This API is a supplement for existing onednn engine API.

.. index:: pair: function; dnnl_graph_make_engine_with_allocator
.. _doxid-group__dnnl__graph__api__engine_1gacd72ae9dc87f2fab2d155faa2bcf0258:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_make_engine_with_allocator(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		)

This API is a supplement for existing onednn engine API.

