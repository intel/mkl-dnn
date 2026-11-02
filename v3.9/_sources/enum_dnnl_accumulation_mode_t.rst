.. index:: pair: enum; dnnl_accumulation_mode_t
.. _doxid-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f:

enum dnnl_accumulation_mode_t
=============================

Overview
~~~~~~~~

Accumulation mode. :ref:`More...<details-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_accumulation_mode_t
	{
	    :ref:`dnnl_accumulation_mode_strict<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafb83d4725f8c96479cd558a23cd60b6d>`,
	    :ref:`dnnl_accumulation_mode_relaxed<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fa9d1932f25fb8115758987627620d0c7d>`,
	    :ref:`dnnl_accumulation_mode_any<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fab49ca983cbe38a75a1ff0948c55f74bb>`,
	    :ref:`dnnl_accumulation_mode_s32<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fade88c19d4a39028a05d61501e88fe23d>`,
	    :ref:`dnnl_accumulation_mode_f32<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7face0452131be499ecbd227f05f0c330ec>`,
	    :ref:`dnnl_accumulation_mode_f16<doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafbe48b9827d45c4881477b102feef6a4>`,
	};

.. _details-group__dnnl__api__accumulation__mode_1gaaafa6b3dae454d4bacc298046a748f7f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Accumulation mode.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_accumulation_mode_strict
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafb83d4725f8c96479cd558a23cd60b6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_strict

Default behavior, f32/f64 for floating point computation, s32 for integer.

.. index:: pair: enumvalue; dnnl_accumulation_mode_relaxed
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fa9d1932f25fb8115758987627620d0c7d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_relaxed

Same as strict but allows some partial accumulators to be rounded to src/dst datatype in memory.

.. index:: pair: enumvalue; dnnl_accumulation_mode_any
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fab49ca983cbe38a75a1ff0948c55f74bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_any

uses fastest implementation, could use src/dst datatype or wider datatype for accumulators

.. index:: pair: enumvalue; dnnl_accumulation_mode_s32
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fade88c19d4a39028a05d61501e88fe23d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_s32

use s32 accumulators during computation

.. index:: pair: enumvalue; dnnl_accumulation_mode_f32
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7face0452131be499ecbd227f05f0c330ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_f32

use f32 accumulators during computation

.. index:: pair: enumvalue; dnnl_accumulation_mode_f16
.. _doxid-group__dnnl__api__accumulation__mode_1ggaaafa6b3dae454d4bacc298046a748f7fafbe48b9827d45c4881477b102feef6a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_accumulation_mode_f16

use f16 accumulators during computation

