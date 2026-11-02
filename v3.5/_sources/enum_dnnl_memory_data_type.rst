.. index:: pair: enum; data_type
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce:

enum dnnl::memory::data_type
============================

Overview
~~~~~~~~

Data type specification. :ref:`More...<details-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum data_type
	{
	    :ref:`undef<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b>`   = dnnl_data_type_undef,
	    :ref:`f8_e5m2<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>` = dnnl_f8_e5m2,
	    :ref:`f8_e4m3<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>` = dnnl_f8_e4m3,
	    :ref:`f16<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa2449b6477c1fef79be4202906486876>`     = dnnl_f16,
	    :ref:`bf16<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceafe2904d9fb3b0f4a81c92b03dec11424>`    = dnnl_bf16,
	    :ref:`f32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`     = dnnl_f32,
	    :target:`f64<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea714b98e0a797e8f119f257a4ab802f86>`     = dnnl_f64,
	    :ref:`s32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`     = dnnl_s32,
	    :ref:`s8<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`      = dnnl_s8,
	    :ref:`u8<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`      = dnnl_u8,
	    :ref:`s4<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea7e7cb6814b74d6596098fc80127569a5>`      = dnnl_s4,
	    :ref:`u4<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea7b8d62fd2f0f5b2e3ba5437e5b983128>`      = dnnl_u4,
	};

.. _details-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Data type specification.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined data type (used for empty memory descriptors).

.. index:: pair: enumvalue; f8_e5m2
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f8_e5m2

`OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with a 5-bit exponent and a 2-bit mantissa.

.. index:: pair: enumvalue; f8_e4m3
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f8_e4m3

`OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with a 4-bit exponent and a 3-bit mantissa.

.. index:: pair: enumvalue; f16
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa2449b6477c1fef79be4202906486876:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f16

`16-bit/half-precision floating point <https://en.wikipedia.org/wiki/Half-precision_floating-point_format>`__.

.. index:: pair: enumvalue; bf16
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceafe2904d9fb3b0f4a81c92b03dec11424:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bf16

non-standard `16-bit floating point with 7-bit mantissa <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`__.

.. index:: pair: enumvalue; f32
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f32

`32-bit/single-precision floating point <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>`__.

.. index:: pair: enumvalue; s32
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s32

32-bit signed integer.

.. index:: pair: enumvalue; s8
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s8

8-bit signed integer.

.. index:: pair: enumvalue; u8
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	u8

8-bit unsigned integer.

.. index:: pair: enumvalue; s4
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea7e7cb6814b74d6596098fc80127569a5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s4

4-bit signed integer.

.. index:: pair: enumvalue; u4
.. _doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea7b8d62fd2f0f5b2e3ba5437e5b983128:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	u4

4-bit unsigned integer.

