.. index:: pair: enum; dnnl_data_type_t
.. _doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130:

enum dnnl_data_type_t
=====================

Overview
~~~~~~~~

Data type specification. :ref:`More...<details-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_data_type_t
	{
	    :ref:`dnnl_data_type_undef<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a7b0351f23ccd840c87a0a9d869339888>` = 0,
	    :ref:`dnnl_f16<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a1c7bb1ce333c6ed8226508017a7f47b8>`             = 1,
	    :ref:`dnnl_bf16<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a35111b4783ae26a46ecb816a32878e82>`            = 2,
	    :ref:`dnnl_f32<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`             = 3,
	    :ref:`dnnl_s32<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a9ce2117fd91c023d8da430800ff53d82>`             = 4,
	    :ref:`dnnl_s8<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a9638cfbcb7d50834a608ffae644d76b4>`              = 5,
	    :ref:`dnnl_u8<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ac5608ac5efc4d052b251c72761ecc1fd>`              = 6,
	    :ref:`dnnl_f64<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a5c78081e72def01de5160992675dc784>`             = 7,
	    :ref:`dnnl_boolean<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ab099b806f4a560d865597db94913c71d>`         = 8,
	    :ref:`dnnl_data_type_max<doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ac484ce1feffefdc9be913fee93db2590>`   = 0x7fff,
	};

.. _details-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Data type specification.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_data_type_undef
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a7b0351f23ccd840c87a0a9d869339888:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_data_type_undef

Undefined data type, used for empty memory descriptors.

.. index:: pair: enumvalue; dnnl_f16
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a1c7bb1ce333c6ed8226508017a7f47b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_f16

16-bit/half-precision floating point.

.. index:: pair: enumvalue; dnnl_bf16
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a35111b4783ae26a46ecb816a32878e82:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bf16

non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.

.. index:: pair: enumvalue; dnnl_f32
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_f32

32-bit/single-precision floating point.

.. index:: pair: enumvalue; dnnl_s32
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a9ce2117fd91c023d8da430800ff53d82:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_s32

32-bit signed integer.

.. index:: pair: enumvalue; dnnl_s8
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a9638cfbcb7d50834a608ffae644d76b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_s8

8-bit signed integer.

.. index:: pair: enumvalue; dnnl_u8
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ac5608ac5efc4d052b251c72761ecc1fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_u8

8-bit unsigned integer.

.. index:: pair: enumvalue; dnnl_f64
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a5c78081e72def01de5160992675dc784:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_f64

64-bit/double-precision floating point.

.. index:: pair: enumvalue; dnnl_boolean
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ab099b806f4a560d865597db94913c71d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_boolean

Boolean data type. Size is C++ implementation defined.

.. index:: pair: enumvalue; dnnl_data_type_max
.. _doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130ac484ce1feffefdc9be913fee93db2590:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_data_type_max

Parameter to allow internal only data_types without undefined behavior.

This parameter is chosen to be valid for so long as sizeof(int) >= 2.

