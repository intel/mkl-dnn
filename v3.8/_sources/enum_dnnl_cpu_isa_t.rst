.. index:: pair: enum; dnnl_cpu_isa_t
.. _doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2:

enum dnnl_cpu_isa_t
===================

Overview
~~~~~~~~

CPU instruction set flags. :ref:`More...<details-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_cpu_isa_t
	{
	    :ref:`dnnl_cpu_isa_default<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a334f526a8651da897123990b8c919928>`              = 0x0,
	    :ref:`dnnl_cpu_isa_sse41<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a5e2f2cccadb94b34700a90bba91e0fe3>`                = 0x1,
	    :ref:`dnnl_cpu_isa_avx<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a270db093c67689e8e926afffc16706a2>`                  = 0x3,
	    :ref:`dnnl_cpu_isa_avx2<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a45f38960497cf614c1adfffddaa57032>`                 = 0x7,
	    :ref:`dnnl_cpu_isa_avx2_vnni<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a8f2cbdae2834cebd2e5bf86b8c65e9d4>`            = 0xf,
	    :ref:`dnnl_cpu_isa_avx2_vnni_2<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a880cf4e5c7e0d661478aa081c9b188ff>`          = 0x1f,
	    :ref:`dnnl_cpu_isa_avx512_core<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a574f09a9b057ba134d48dadf6d8aa201>`          = 0x27,
	    :ref:`dnnl_cpu_isa_avx512_core_vnni<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a3aced59a3047f7e407b1fe3310430554>`     = 0x67,
	    :ref:`dnnl_cpu_isa_avx512_core_bf16<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a9ced36845ccb9a8dd63cd49ec103412b>`     = 0xe7,
	    :ref:`dnnl_cpu_isa_avx10_1_512<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a15ac88bbd92355013ce4c5e715821ccb>`          = 0x1ef,
	    :ref:`dnnl_cpu_isa_avx512_core_fp16<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2aba37df95641b63e78c7e4a52c2acdd84>`     = dnnl_cpu_isa_avx10_1_512,
	    :ref:`dnnl_cpu_isa_avx10_1_512_amx<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2afe9a8152f35e294a2eb24d2c20f0ae37>`      = 0xfef,
	    :ref:`dnnl_cpu_isa_avx512_core_amx<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a3a4b0c594f109982fde90e221087ded9>`      = dnnl_cpu_isa_avx10_1_512_amx,
	    :ref:`dnnl_cpu_isa_avx10_1_512_amx_fp16<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a9182399556e9a9da18e6f0a9706d3bb0>` = 0x1fef,
	    :ref:`dnnl_cpu_isa_avx512_core_amx_fp16<doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a8df60d65867e94bccd8c03811e4b6343>` = dnnl_cpu_isa_avx10_1_512_amx_fp16,
	};

.. _details-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

CPU instruction set flags.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_cpu_isa_default
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a334f526a8651da897123990b8c919928:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_default

Library choice of ISA (excepting those listed as initial support)

.. index:: pair: enumvalue; dnnl_cpu_isa_sse41
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a5e2f2cccadb94b34700a90bba91e0fe3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_sse41

Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)

.. index:: pair: enumvalue; dnnl_cpu_isa_avx
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a270db093c67689e8e926afffc16706a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx

Intel Advanced Vector Extensions (Intel AVX)

.. index:: pair: enumvalue; dnnl_cpu_isa_avx2
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a45f38960497cf614c1adfffddaa57032:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx2

Intel Advanced Vector Extensions 2 (Intel AVX2)

.. index:: pair: enumvalue; dnnl_cpu_isa_avx2_vnni
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a8f2cbdae2834cebd2e5bf86b8c65e9d4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx2_vnni

Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) support.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx2_vnni_2
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a880cf4e5c7e0d661478aa081c9b188ff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx2_vnni_2

Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) with 8-bit integer, float16 and bfloat16 support.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a574f09a9b057ba134d48dadf6d8aa201:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core

Intel AVX-512 subset for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core_vnni
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a3aced59a3047f7e407b1fe3310430554:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core_vnni

Intel AVX-512 and Intel Deep Learning Boost (Intel DL Boost) support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core_bf16
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a9ced36845ccb9a8dd63cd49ec103412b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core_bf16

Intel AVX-512, Intel DL Boost and bfloat16 support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx10_1_512
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a15ac88bbd92355013ce4c5e715821ccb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx10_1_512

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core_fp16
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2aba37df95641b63e78c7e4a52c2acdd84:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core_fp16

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx10_1_512_amx
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2afe9a8152f35e294a2eb24d2c20f0ae37:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx10_1_512_amx

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer and bfloat16 support.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core_amx
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a3a4b0c594f109982fde90e221087ded9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core_amx

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer and bfloat16 support.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx10_1_512_amx_fp16
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a9182399556e9a9da18e6f0a9706d3bb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx10_1_512_amx_fp16

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer, bfloat16 and float16 support.

.. index:: pair: enumvalue; dnnl_cpu_isa_avx512_core_amx_fp16
.. _doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a8df60d65867e94bccd8c03811e4b6343:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_avx512_core_amx_fp16

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer, bfloat16 and float16 support.

