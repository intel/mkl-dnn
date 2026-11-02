.. index:: pair: enum; cpu_isa
.. _doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83:

enum dnnl::cpu_isa
==================

Overview
~~~~~~~~

CPU instruction set flags. :ref:`More...<details-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum cpu_isa
	{
	    :ref:`isa_default<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a56a0edddbeaaf449d233434fb1860724>`          = dnnl_cpu_isa_default,
	    :ref:`sse41<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ab14e8c38c0a37f70c070f2a862d30d65>`                = dnnl_cpu_isa_sse41,
	    :ref:`avx<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a73758c37e4499f20ac5f995a144abba6>`                  = dnnl_cpu_isa_avx,
	    :ref:`avx2<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a220c4ad92e33497ef256a48712352b84>`                 = dnnl_cpu_isa_avx2,
	    :ref:`avx2_vnni<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a59548f83cb32f5f6272186734a9a711d>`            = dnnl_cpu_isa_avx2_vnni,
	    :ref:`avx2_vnni_2<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83aa62f186f7e3a3fc401aa9f91bb87b115>`          = dnnl_cpu_isa_avx2_vnni_2,
	    :ref:`avx512_core<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83aa427cc9f00ac692056a83a8cb5e37fa4>`          = dnnl_cpu_isa_avx512_core,
	    :ref:`avx512_core_vnni<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83acac6fe12844735aafd8fd1fd81738f8e>`     = dnnl_cpu_isa_avx512_core_vnni,
	    :ref:`avx512_core_bf16<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83af9b353b49d5aa4dfe76e22337b5a02cf>`     = dnnl_cpu_isa_avx512_core_bf16,
	    :ref:`avx512_core_fp16<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ae715ea1a784388f3588ed5434a333e93>`     = dnnl_cpu_isa_avx512_core_fp16,
	    :ref:`avx512_core_amx<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ad49189665b06a3259b3bf3603319fd0d>`      = dnnl_cpu_isa_avx512_core_amx,
	    :ref:`avx512_core_amx_fp16<doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a0329053700e07b75bc2a41e5be282d83>` = dnnl_cpu_isa_avx512_core_amx_fp16,
	};

.. _details-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

CPU instruction set flags.

Enum Values
-----------

.. index:: pair: enumvalue; isa_default
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a56a0edddbeaaf449d233434fb1860724:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	isa_default

Library choice of ISA (excepting those listed as initial support)

.. index:: pair: enumvalue; sse41
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ab14e8c38c0a37f70c070f2a862d30d65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sse41

Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)

.. index:: pair: enumvalue; avx
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a73758c37e4499f20ac5f995a144abba6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx

Intel Advanced Vector Extensions (Intel AVX)

.. index:: pair: enumvalue; avx2
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a220c4ad92e33497ef256a48712352b84:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx2

Intel Advanced Vector Extensions 2 (Intel AVX2)

.. index:: pair: enumvalue; avx2_vnni
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a59548f83cb32f5f6272186734a9a711d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx2_vnni

Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) support.

.. index:: pair: enumvalue; avx2_vnni_2
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83aa62f186f7e3a3fc401aa9f91bb87b115:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx2_vnni_2

Intel AVX2 and Intel Deep Learning Boost (Intel DL Boost) with 8-bit integer, float16 and bfloat16 support (preview support)

.. index:: pair: enumvalue; avx512_core
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83aa427cc9f00ac692056a83a8cb5e37fa4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core

Intel AVX-512 subset for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; avx512_core_vnni
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83acac6fe12844735aafd8fd1fd81738f8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core_vnni

Intel AVX-512 and Intel Deep Learning Boost (Intel DL Boost) support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; avx512_core_bf16
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83af9b353b49d5aa4dfe76e22337b5a02cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core_bf16

Intel AVX-512, Intel DL Boost and bfloat16 support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; avx512_core_fp16
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ae715ea1a784388f3588ed5434a333e93:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core_fp16

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support for Intel Xeon Scalable processor family and Intel Core processor family.

.. index:: pair: enumvalue; avx512_core_amx
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83ad49189665b06a3259b3bf3603319fd0d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core_amx

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer and bfloat16 support.

.. index:: pair: enumvalue; avx512_core_amx_fp16
.. _doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a0329053700e07b75bc2a41e5be282d83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	avx512_core_amx_fp16

Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer, bfloat16 and float16 support (preview support)

