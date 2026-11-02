.. index:: pair: enum; dnnl_fpmath_mode_t
.. _doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912:

enum dnnl_fpmath_mode_t
=======================

Overview
~~~~~~~~

Floating-point math mode. :ref:`More...<details-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_fpmath_mode_t
	{
	    :ref:`dnnl_fpmath_mode_strict<doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3>`,
	    :ref:`dnnl_fpmath_mode_bf16<doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7>`,
	    :ref:`dnnl_fpmath_mode_f16<doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf>`,
	    :ref:`dnnl_fpmath_mode_any<doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c>`,
	    :ref:`dnnl_fpmath_mode_tf32<doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc>`,
	};

.. _details-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Floating-point math mode.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_fpmath_mode_strict
.. _doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fpmath_mode_strict

Default behavior, no downconversions allowed.

.. index:: pair: enumvalue; dnnl_fpmath_mode_bf16
.. _doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fpmath_mode_bf16

Implicit f32->bf16 conversions allowed.

.. index:: pair: enumvalue; dnnl_fpmath_mode_f16
.. _doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fpmath_mode_f16

Implicit f32->f16 conversions allowed.

.. index:: pair: enumvalue; dnnl_fpmath_mode_any
.. _doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fpmath_mode_any

Implicit f32->f16 or f32->bf16 conversions allowed.

.. index:: pair: enumvalue; dnnl_fpmath_mode_tf32
.. _doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fpmath_mode_tf32

Implicit f32->tf32 conversions allowed.

