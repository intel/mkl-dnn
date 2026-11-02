.. index:: pair: group; Accumulation Mode
.. _doxid-group__dnnl__api__accumulation__mode:

Accumulation Mode
=================

.. toctree::
	:hidden:

	enum_dnnl_accumulation_mode.rst
	enum_dnnl_accumulation_mode_t.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>`;
	enum :ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>`;

	// global functions

	:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__accumulation__mode_1ga574eebb02aca166abace712d976ed55a>`(:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` mode);

.. _details-group__dnnl__api__accumulation__mode:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__accumulation__mode_1ga574eebb02aca166abace712d976ed55a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_accumulation_mode_t<doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>` dnnl::convert_to_c(:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` mode)

Converts an accumulation mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API accumulation mode enum value.



.. rubric:: Returns:

Corresponding C API accumulation mode enum value.

