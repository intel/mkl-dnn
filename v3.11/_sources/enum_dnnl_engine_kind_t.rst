.. index:: pair: enum; dnnl_engine_kind_t
.. _doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2:

enum dnnl_engine_kind_t
=======================

Overview
~~~~~~~~

Kinds of engines. :ref:`More...<details-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_engine_kind_t
	{
	    :ref:`dnnl_any_engine<doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2af14bd66bd4a7130da204c539f7cf08b6>`,
	    :ref:`dnnl_cpu<doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2abde7b942413dd36f8285dd360fc0c797>`,
	    :ref:`dnnl_gpu<doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2a6ab900ae0fc26be67742c0e59a015438>`,
	};

.. _details-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of engines.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_any_engine
.. _doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2af14bd66bd4a7130da204c539f7cf08b6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_any_engine

An unspecified engine.

.. index:: pair: enumvalue; dnnl_cpu
.. _doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2abde7b942413dd36f8285dd360fc0c797:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu

CPU engine.

.. index:: pair: enumvalue; dnnl_gpu
.. _doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2a6ab900ae0fc26be67742c0e59a015438:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_gpu

GPU engine.

