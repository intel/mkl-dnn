.. index:: pair: group; Floating-point Math Mode
.. _doxid-group__dnnl__api__cpp__fpmath__mode:

Floating-point Math Mode
========================

.. toctree::
	:hidden:

	enum_dnnl_fpmath_mode.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::fpmath_mode<doxid-group__dnnl__api__cpp__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`;

	// global functions

	:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__fpmath__mode_1gad095d0686c7020ce49be483cb44e8535>`(:ref:`fpmath_mode<doxid-group__dnnl__api__cpp__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);

.. _details-group__dnnl__api__cpp__fpmath__mode:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__fpmath__mode_1gad095d0686c7020ce49be483cb44e8535:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` dnnl::convert_to_c(:ref:`fpmath_mode<doxid-group__dnnl__api__cpp__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode)

Converts an fpmath mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API fpmath mode enum value.



.. rubric:: Returns:

Corresponding C API fpmath mode enum value.

