.. index:: pair: group; Floating-point Math Mode
.. _doxid-group__dnnl__api__fpmath__mode:

Floating-point Math Mode
========================

.. toctree::
	:hidden:

	enum_dnnl_fpmath_mode_t.rst
	enum_dnnl_fpmath_mode.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`;
	enum :ref:`dnnl::fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`;

	// global functions

	:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__fpmath__mode_1gad095d0686c7020ce49be483cb44e8535>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_get_default_fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1gada52f7858332a7cda0e0c5e7907056d7>`(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_default_fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga97dd535e43073cee2ebc4b709e42c3ca>`(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode);

.. _details-group__dnnl__api__fpmath__mode:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__fpmath__mode_1gad095d0686c7020ce49be483cb44e8535:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` dnnl::convert_to_c(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode)

Converts an fpmath mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API fpmath mode enum value.



.. rubric:: Returns:

Corresponding C API fpmath mode enum value.

.. index:: pair: function; dnnl_get_default_fpmath_mode
.. _doxid-group__dnnl__api__fpmath__mode_1gada52f7858332a7cda0e0c5e7907056d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_get_default_fpmath_mode(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode)

Returns the floating-point math mode that will be used by default for all subsequently created primitives.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- Output FP math mode.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_set_default_fpmath_mode
.. _doxid-group__dnnl__api__fpmath__mode_1ga97dd535e43073cee2ebc4b709e42c3ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_default_fpmath_mode(:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode)

Sets the floating-point math mode that will be used by default for all subsequently created primitives.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- FP math mode. The possible values are: :ref:`dnnl_fpmath_mode_strict <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3>`, :ref:`dnnl_fpmath_mode_bf16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7>`, :ref:`dnnl_fpmath_mode_f16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf>`, :ref:`dnnl_fpmath_mode_tf32 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc>`, :ref:`dnnl_fpmath_mode_any <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

