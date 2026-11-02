.. index:: pair: group; Data types
.. _doxid-group__dnnl__api__data__types:

Data types
==========

.. toctree::
	:hidden:

	enum_dnnl_data_type_t.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef int64_t :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`;
	typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>`[DNNL_MAX_NDIMS];

	// enums

	enum :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`;

	// macros

	#define :ref:`DNNL_MAX_NDIMS<doxid-group__dnnl__api__data__types_1gaa9e648b617df0f0186143abdf78ca5f2>`

.. _details-group__dnnl__api__data__types:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Typedefs
--------

.. index:: pair: typedef; dnnl_dim_t
.. _doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int64_t dnnl_dim_t

A type to describe tensor dimension.

.. index:: pair: typedef; dnnl_dims_t
.. _doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dnnl_dims_t[DNNL_MAX_NDIMS]

A type to describe tensor dimensions.

Macros
------

.. index:: pair: define; DNNL_MAX_NDIMS
.. _doxid-group__dnnl__api__data__types_1gaa9e648b617df0f0186143abdf78ca5f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_MAX_NDIMS

Maximum number of dimensions a tensor can have.

Only restricts the amount of space used for the tensor description. Individual computational primitives may support only tensors of certain dimensions.

