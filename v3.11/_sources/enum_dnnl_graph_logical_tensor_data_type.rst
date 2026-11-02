.. index:: pair: enum; data_type
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227:

enum dnnl::graph::logical_tensor::data_type
===========================================

Overview
~~~~~~~~

Data Type. :ref:`More...<details-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum data_type
	{
	    :target:`undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227af31ee5e3824f1f5e5d206bdf3029f22b>`   = dnnl_data_type_undef,
	    :ref:`f16<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227aa2449b6477c1fef79be4202906486876>`     = dnnl_f16,
	    :ref:`bf16<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227afe2904d9fb3b0f4a81c92b03dec11424>`    = dnnl_bf16,
	    :ref:`f32<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a512dc597be7ae761876315165dc8bd2e>`     = dnnl_f32,
	    :ref:`s32<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227aa860868d23f3a68323a2e3f6563d7f31>`     = dnnl_s32,
	    :ref:`s8<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a3e8d88fdd85d7153525e0647cdd97686>`      = dnnl_s8,
	    :ref:`u8<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a077393852be20e37026d6281827662f2>`      = dnnl_u8,
	    :ref:`boolean<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a84e2c64f38f78ba3ea5c905ab5a2da27>` = dnnl_boolean,
	    :ref:`f8_e5m2<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a12ad5ee1ad075296bc5566a2d366678c>` = dnnl_f8_e5m2,
	    :ref:`f8_e4m3<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227af5ede3d43b879551314bbb05684fa758>` = dnnl_f8_e4m3,
	    :ref:`s4<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a7e7cb6814b74d6596098fc80127569a5>`      = dnnl_s4,
	    :ref:`u4<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a7b8d62fd2f0f5b2e3ba5437e5b983128>`      = dnnl_u4,
	};

.. _details-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Data Type.

Enum Values
-----------

.. index:: pair: enumvalue; f16
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227aa2449b6477c1fef79be4202906486876:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f16

16-bit/half-precision floating point.

.. index:: pair: enumvalue; bf16
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227afe2904d9fb3b0f4a81c92b03dec11424:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bf16

non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.

.. index:: pair: enumvalue; f32
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a512dc597be7ae761876315165dc8bd2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f32

32-bit/single-precision floating point.

.. index:: pair: enumvalue; s32
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227aa860868d23f3a68323a2e3f6563d7f31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s32

32-bit signed integer.

.. index:: pair: enumvalue; s8
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a3e8d88fdd85d7153525e0647cdd97686:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s8

8-bit signed integer.

.. index:: pair: enumvalue; u8
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a077393852be20e37026d6281827662f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	u8

8-bit unsigned integer.

.. index:: pair: enumvalue; boolean
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a84e2c64f38f78ba3ea5c905ab5a2da27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	boolean

Boolean data type. Size is C++ implementation defined.

.. index:: pair: enumvalue; f8_e5m2
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a12ad5ee1ad075296bc5566a2d366678c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f8_e5m2

`OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with a 5-bit exponent and a 2-bit mantissa.

.. index:: pair: enumvalue; f8_e4m3
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227af5ede3d43b879551314bbb05684fa758:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	f8_e4m3

`OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with a 4-bit exponent and a 3-bit mantissa.

.. index:: pair: enumvalue; s4
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a7e7cb6814b74d6596098fc80127569a5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	s4

4-bit signed integer.

.. index:: pair: enumvalue; u4
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227a7b8d62fd2f0f5b2e3ba5437e5b983128:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	u4

4-bit unsigned integer.

