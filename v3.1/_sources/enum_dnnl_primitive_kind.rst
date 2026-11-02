.. index:: pair: enum; kind
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169:

enum dnnl::primitive::kind
==========================

Overview
~~~~~~~~

Kinds of primitives supported by the library. :ref:`More...<details-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum kind
	{
	    :ref:`undef<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169af31ee5e3824f1f5e5d206bdf3029f22b>`               = dnnl_undefined_primitive,
	    :ref:`reorder<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a8dbc5523c131d7900138d32d1dcbf093>`             = dnnl_reorder,
	    :ref:`shuffle<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a644c45fc2b003bf415a3fb655412508d>`             = dnnl_shuffle,
	    :ref:`concat<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a414e7c8ede73a0c2c3d17699134f4080>`              = dnnl_concat,
	    :ref:`sum<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a1d623b89683f9ce4e074de1676d12416>`                 = dnnl_sum,
	    :ref:`convolution<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169aa9595c1c24c33b16056d2ad07e71682d>`         = dnnl_convolution,
	    :ref:`deconvolution<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a2294014fceafa2f8ca8d16ad35f25875>`       = dnnl_deconvolution,
	    :ref:`eltwise<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a98b908c7d0339bb6a4832db44fc2c8da>`             = dnnl_eltwise,
	    :ref:`lrn<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a7d9c7d1747abfac2c3c3ff337712f575>`                 = dnnl_lrn,
	    :ref:`batch_normalization<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a780789647a31acedb6477412a99154bb>` = dnnl_batch_normalization,
	    :ref:`inner_product<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a30813812e3beef3eefcc2b97dc2e0783>`       = dnnl_inner_product,
	    :ref:`rnn<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a6eed45b9327120cabad6916919a9a600>`                 = dnnl_rnn,
	    :ref:`binary<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a9d7183f16acce70658f686ae7f1a4d20>`              = dnnl_binary,
	    :ref:`matmul<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a0f2e1c85f0064ca22d3ac468c44b0327>`              = dnnl_matmul,
	    :ref:`resampling<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a4f9776b23053c32400f732de532df397>`          = dnnl_resampling,
	    :ref:`pooling<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a4abcef116566139e94c82bb1bb37583f>`             = dnnl_pooling,
	    :ref:`reduction<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a9477e24c78f1ca8ee9b03111f6227fc2>`           = dnnl_reduction,
	    :ref:`prelu<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a837c39f77d473b24eb27c0758d5c7c1b>`               = dnnl_prelu,
	    :ref:`softmax<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169ace2f1fbd249d24aabc07ac4488ab5b8c>`             = dnnl_softmax,
	    :ref:`layer_normalization<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a40f3d68401fd64071877b5263c3d8706>` = dnnl_layer_normalization,
	};

.. _details-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of primitives supported by the library.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined primitive.

.. index:: pair: enumvalue; reorder
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a8dbc5523c131d7900138d32d1dcbf093:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reorder

A reorder primitive.

.. index:: pair: enumvalue; shuffle
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a644c45fc2b003bf415a3fb655412508d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	shuffle

A shuffle primitive.

.. index:: pair: enumvalue; concat
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a414e7c8ede73a0c2c3d17699134f4080:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	concat

A (out-of-place) tensor concatenation primitive.

.. index:: pair: enumvalue; sum
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a1d623b89683f9ce4e074de1676d12416:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sum

A summation primitive.

.. index:: pair: enumvalue; convolution
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169aa9595c1c24c33b16056d2ad07e71682d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	convolution

A convolution primitive.

.. index:: pair: enumvalue; deconvolution
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a2294014fceafa2f8ca8d16ad35f25875:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	deconvolution

A deconvolution primitive.

.. index:: pair: enumvalue; eltwise
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a98b908c7d0339bb6a4832db44fc2c8da:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	eltwise

An element-wise primitive.

.. index:: pair: enumvalue; lrn
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a7d9c7d1747abfac2c3c3ff337712f575:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	lrn

An LRN primitive.

.. index:: pair: enumvalue; batch_normalization
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a780789647a31acedb6477412a99154bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	batch_normalization

A batch normalization primitive.

.. index:: pair: enumvalue; inner_product
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a30813812e3beef3eefcc2b97dc2e0783:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	inner_product

An inner product primitive.

.. index:: pair: enumvalue; rnn
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a6eed45b9327120cabad6916919a9a600:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	rnn

An RNN primitive.

.. index:: pair: enumvalue; binary
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a9d7183f16acce70658f686ae7f1a4d20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	binary

A binary primitive.

.. index:: pair: enumvalue; matmul
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a0f2e1c85f0064ca22d3ac468c44b0327:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	matmul

A matmul (matrix multiplication) primitive.

.. index:: pair: enumvalue; resampling
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a4f9776b23053c32400f732de532df397:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	resampling

A resampling primitive.

.. index:: pair: enumvalue; pooling
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a4abcef116566139e94c82bb1bb37583f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pooling

A pooling primitive.

.. index:: pair: enumvalue; reduction
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a9477e24c78f1ca8ee9b03111f6227fc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	reduction

A reduction primitive.

.. index:: pair: enumvalue; prelu
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a837c39f77d473b24eb27c0758d5c7c1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	prelu

A PReLU primitive.

.. index:: pair: enumvalue; softmax
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169ace2f1fbd249d24aabc07ac4488ab5b8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	softmax

A softmax primitive.

.. index:: pair: enumvalue; layer_normalization
.. _doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a40f3d68401fd64071877b5263c3d8706:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	layer_normalization

A layer normalization primitive.

