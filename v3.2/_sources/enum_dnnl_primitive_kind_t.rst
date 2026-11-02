.. index:: pair: enum; dnnl_primitive_kind_t
.. _doxid-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a:

enum dnnl_primitive_kind_t
==========================

Overview
~~~~~~~~

Kinds of primitives. :ref:`More...<details-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_primitive_kind_t
	{
	    :ref:`dnnl_undefined_primitive<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa89c48c6939d9f939b0f66d9b018a03b9>`,
	    :ref:`dnnl_reorder<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa2a9143625717f90ce928087c31afe1a1>`,
	    :ref:`dnnl_shuffle<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaaa09b9e85c78e425237c1d8c67a4cfac>`,
	    :ref:`dnnl_concat<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa25812ad02a476903064350bedbeaa763>`,
	    :ref:`dnnl_sum<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8ab0990125f2a743db86666c9d8b401b>`,
	    :ref:`dnnl_convolution<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa402cfeaa257524d301bb73e770bc87f6>`,
	    :ref:`dnnl_deconvolution<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aacd3571ebae73142a0f690a87cca1fbb2>`,
	    :ref:`dnnl_eltwise<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa9d7e709dd7e25d7ff11cf51c13fa2819>`,
	    :ref:`dnnl_lrn<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aafa8035256ece12602e5f25330699b548>`,
	    :ref:`dnnl_batch_normalization<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8483b015fd10c9832c6326a1c5b59ed5>`,
	    :ref:`dnnl_inner_product<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa443fbc5518e5e47cd97d43339a2c4987>`,
	    :ref:`dnnl_rnn<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaf383485481e1d188df5b59c5424ee3ea>`,
	    :ref:`dnnl_gemm<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa3b0a4f28c65ae19ceb17f8cc9c4a1ad8>`,
	    :ref:`dnnl_binary<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa1d51705e2642ce2ce19a3e163bb25f93>`,
	    :ref:`dnnl_matmul<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa9e943e8007f69f82a00e30a3079cb407>`,
	    :ref:`dnnl_resampling<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa7ab1ead09194e7acb0813c953e66988f>`,
	    :ref:`dnnl_pooling<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aac16fd61735aacd6937755e41274dca0c>`,
	    :ref:`dnnl_reduction<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaf3bb69107badd983d16909afe507ed04>`,
	    :ref:`dnnl_prelu<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa96f5fd519486bbb41cdf6e297b96b28f>`,
	    :ref:`dnnl_softmax<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aad6c6b37d892628e2b869669557d0098e>`,
	    :ref:`dnnl_layer_normalization<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8fe4eac2a079b1278abbc80cf0d90aff>`,
	    :ref:`dnnl_primitive_kind_max<doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa3246cd9a7193bd0b94facea9e0a2cca9>`  = 0x7fff,
	};

.. _details-group__dnnl__api__primitives__common_1ga9878f4795e53ad8443e5c0a29e53286a:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of primitives.

Used to implement a way to extend the library with new primitives without changing the ABI.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_undefined_primitive
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa89c48c6939d9f939b0f66d9b018a03b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_undefined_primitive

Undefined primitive.

.. index:: pair: enumvalue; dnnl_reorder
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa2a9143625717f90ce928087c31afe1a1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reorder

A reorder primitive.

.. index:: pair: enumvalue; dnnl_shuffle
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaaa09b9e85c78e425237c1d8c67a4cfac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_shuffle

A shuffle primitive.

.. index:: pair: enumvalue; dnnl_concat
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa25812ad02a476903064350bedbeaa763:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_concat

A (out-of-place) concat primitive.

.. index:: pair: enumvalue; dnnl_sum
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8ab0990125f2a743db86666c9d8b401b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_sum

A sum primitive.

.. index:: pair: enumvalue; dnnl_convolution
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa402cfeaa257524d301bb73e770bc87f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_convolution

A convolution primitive.

.. index:: pair: enumvalue; dnnl_deconvolution
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aacd3571ebae73142a0f690a87cca1fbb2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_deconvolution

A deconvolution primitive.

.. index:: pair: enumvalue; dnnl_eltwise
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa9d7e709dd7e25d7ff11cf51c13fa2819:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_eltwise

An element-wise primitive.

.. index:: pair: enumvalue; dnnl_lrn
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aafa8035256ece12602e5f25330699b548:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_lrn

An LRN primitive.

.. index:: pair: enumvalue; dnnl_batch_normalization
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8483b015fd10c9832c6326a1c5b59ed5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_batch_normalization

A batch normalization primitive.

.. index:: pair: enumvalue; dnnl_inner_product
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa443fbc5518e5e47cd97d43339a2c4987:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_inner_product

An inner product primitive.

.. index:: pair: enumvalue; dnnl_rnn
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaf383485481e1d188df5b59c5424ee3ea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_rnn

A rnn primitive.

.. index:: pair: enumvalue; dnnl_gemm
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa3b0a4f28c65ae19ceb17f8cc9c4a1ad8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_gemm

A matrix multiplication primitive (internal).

.. index:: pair: enumvalue; dnnl_binary
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa1d51705e2642ce2ce19a3e163bb25f93:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_binary

A binary primitive.

.. index:: pair: enumvalue; dnnl_matmul
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa9e943e8007f69f82a00e30a3079cb407:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_matmul

A matrix multiplication primitive.

.. index:: pair: enumvalue; dnnl_resampling
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa7ab1ead09194e7acb0813c953e66988f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_resampling

A resampling primitive.

.. index:: pair: enumvalue; dnnl_pooling
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aac16fd61735aacd6937755e41274dca0c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pooling

A pooling primitive.

.. index:: pair: enumvalue; dnnl_reduction
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aaf3bb69107badd983d16909afe507ed04:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_reduction

A reduction primitive.

.. index:: pair: enumvalue; dnnl_prelu
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa96f5fd519486bbb41cdf6e297b96b28f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_prelu

A PReLU primitive.

.. index:: pair: enumvalue; dnnl_softmax
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aad6c6b37d892628e2b869669557d0098e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_softmax

A softmax primitive.

.. index:: pair: enumvalue; dnnl_layer_normalization
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa8fe4eac2a079b1278abbc80cf0d90aff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_layer_normalization

A layer normalization primitive.

.. index:: pair: enumvalue; dnnl_primitive_kind_max
.. _doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa3246cd9a7193bd0b94facea9e0a2cca9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_primitive_kind_max

Parameter to allow internal only primitives without undefined behavior.

This parameter is chosen to be valid for so long as sizeof(int) >= 2.

