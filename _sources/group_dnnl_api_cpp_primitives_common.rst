.. index:: pair: group; Common
.. _doxid-group__dnnl__api__cpp__primitives__common:

Common
======

.. toctree::
	:hidden:

	enum_dnnl_normalization_flags.rst
	enum_dnnl_query.rst
	struct_dnnl_primitive-2.rst
	struct_dnnl_primitive_desc-2.rst
	struct_dnnl_primitive_desc_base.rst

Overview
~~~~~~~~

Common operations to create, destroy and inspect primitives :ref:`More...<details-group__dnnl__api__cpp__primitives__common>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::normalization_flags<doxid-group__dnnl__api__cpp__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`;
	enum :ref:`dnnl::query<doxid-group__dnnl__api__cpp__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>`;

	// structs

	struct :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`;
	struct :ref:`dnnl::primitive_desc<doxid-structdnnl_1_1primitive__desc>`;
	struct :ref:`dnnl::primitive_desc_base<doxid-structdnnl_1_1primitive__desc__base>`;

	// global functions

	:ref:`dnnl_primitive_kind_t<doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__primitives__common_1gaaa215c424a2a5c5f734600216dfb8873>`(:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` akind);
	:ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__primitives__common_1gae3d2ea872c5ab424c74d7549d2222926>`(:ref:`normalization_flags<doxid-group__dnnl__api__cpp__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags);
	:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__primitives__common_1ga01d8a1881875cdb94e230db4e53ccb97>`(:ref:`query<doxid-group__dnnl__api__cpp__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` aquery);

.. _details-group__dnnl__api__cpp__primitives__common:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Common operations to create, destroy and inspect primitives

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__primitives__common_1gaaa215c424a2a5c5f734600216dfb8873:

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
.. _doxid-group__dnnl__api__cpp__primitives__common_1gae3d2ea872c5ab424c74d7549d2222926:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_normalization_flags_t<doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>` dnnl::convert_to_c(:ref:`normalization_flags<doxid-group__dnnl__api__cpp__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` flags)

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
.. _doxid-group__dnnl__api__cpp__primitives__common_1ga01d8a1881875cdb94e230db4e53ccb97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` dnnl::convert_to_c(:ref:`query<doxid-group__dnnl__api__cpp__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` aquery)

Converts query enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aquery

		- C++ API query enum value.



.. rubric:: Returns:

Corresponding C API query enum value.

