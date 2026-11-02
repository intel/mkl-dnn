.. index:: pair: enum; fpmath_mode
.. _doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725:

enum dnnl::fpmath_mode
======================

Overview
~~~~~~~~

Floating-point math mode. :ref:`More...<details-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>

	enum fpmath_mode
	{
	    :ref:`strict<doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a2133fd717402a7966ee88d06f9e0b792>` = dnnl_fpmath_mode_strict,
	    :ref:`bf16<doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`   = dnnl_fpmath_mode_bf16,
	    :ref:`f16<doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725aa2449b6477c1fef79be4202906486876>`    = dnnl_fpmath_mode_f16,
	    :ref:`tf32<doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725ae00ad49f2e4f0fda1539ae29b2aaef57>`   = dnnl_fpmath_mode_tf32,
	    :ref:`any<doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`    = dnnl_fpmath_mode_any,
	};

.. _details-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Floating-point math mode.

Enum Values
-----------

.. index:: pair: enumvalue; strict
.. _doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a2133fd717402a7966ee88d06f9e0b792:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	strict

Default behavior, no downconversions allowed.

.. index:: pair: enumvalue; bf16
.. _doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bf16

Implicit f32->bf16 conversions allowed.

.. index:: pair: enumvalue; f16
.. _doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725aa2449b6477c1fef79be4202906486876:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f16

Implicit f32->f16 conversions allowed.

.. index:: pair: enumvalue; tf32
.. _doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725ae00ad49f2e4f0fda1539ae29b2aaef57:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	tf32

Implicit f32->tf32 conversions allowed.

.. index:: pair: enumvalue; any
.. _doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	any

Implicit f32->f16, f32->tf32 or f32->bf16 conversions allowed.

