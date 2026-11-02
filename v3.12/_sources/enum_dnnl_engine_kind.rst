.. index:: pair: enum; kind
.. _doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a:

enum dnnl::engine::kind
=======================

Overview
~~~~~~~~

Kinds of engines. :ref:`More...<details-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>

	enum kind
	{
	    :ref:`any<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa100b8cad7cf2a56f6df78f171f97a1ec>` = dnnl_any_engine,
	    :ref:`cpu<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>` = dnnl_cpu,
	    :ref:`gpu<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947>` = dnnl_gpu,
	};

.. _details-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of engines.

Enum Values
-----------

.. index:: pair: enumvalue; any
.. _doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa100b8cad7cf2a56f6df78f171f97a1ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	any

An unspecified engine.

.. index:: pair: enumvalue; cpu
.. _doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cpu

CPU engine.

.. index:: pair: enumvalue; gpu
.. _doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	gpu

GPU engine.

