.. index:: pair: enum; accumulation_mode
.. _doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215:

enum dnnl::accumulation_mode
============================

Overview
~~~~~~~~

Accumulation mode. :ref:`More...<details-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>

	enum accumulation_mode
	{
	    :ref:`strict<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a2133fd717402a7966ee88d06f9e0b792>`  = dnnl_accumulation_mode_strict,
	    :ref:`relaxed<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a81f32be24a2a62fc472cc43edc97e65b>` = dnnl_accumulation_mode_relaxed,
	    :ref:`any<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a100b8cad7cf2a56f6df78f171f97a1ec>`     = dnnl_accumulation_mode_any,
	    :ref:`s32<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa860868d23f3a68323a2e3f6563d7f31>`     = dnnl_accumulation_mode_s32,
	    :ref:`f32<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`     = dnnl_accumulation_mode_f32,
	    :ref:`f16<doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa2449b6477c1fef79be4202906486876>`     = dnnl_accumulation_mode_f16,
	};

.. _details-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Accumulation mode.

Enum Values
-----------

.. index:: pair: enumvalue; strict
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a2133fd717402a7966ee88d06f9e0b792:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	strict

Default behavior, f32 for floating point computation, s32 for integer.

.. index:: pair: enumvalue; relaxed
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a81f32be24a2a62fc472cc43edc97e65b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	relaxed

same as strict except some partial accumulators can be rounded to src/dst datatype in memory.

.. index:: pair: enumvalue; any
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a100b8cad7cf2a56f6df78f171f97a1ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	any

uses fastest implementation, could use src/dst datatype or wider datatype for accumulators

.. index:: pair: enumvalue; s32
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa860868d23f3a68323a2e3f6563d7f31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s32

use s32 accumulators during computation

.. index:: pair: enumvalue; f32
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f32

use f32 accumulators during computation

.. index:: pair: enumvalue; f16
.. _doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa2449b6477c1fef79be4202906486876:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f16

use f16 accumulators during computation

