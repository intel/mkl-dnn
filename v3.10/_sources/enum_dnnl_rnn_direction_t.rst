.. index:: pair: enum; dnnl_rnn_direction_t
.. _doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0:

enum dnnl_rnn_direction_t
=========================

Overview
~~~~~~~~

A direction of RNN primitive execution. :ref:`More...<details-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_rnn_direction_t
	{
	    :ref:`dnnl_rnn_direction_undef<doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0af99114c11e7a82bdd1969ac59a10750f>`       = 0,
	    :ref:`dnnl_unidirectional_left2right<doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a97cbdb39a208127cc83a9249517c5180>`,
	    :ref:`dnnl_unidirectional_right2left<doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0ad95dd744e0485467b82aacdbbe4590a1>`,
	    :ref:`dnnl_bidirectional_concat<doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a69299415608e015c7334dc342d52743d>`,
	    :ref:`dnnl_bidirectional_sum<doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a6b39b772b540368ef1c80eee3ef1ff27>`,
	};

.. _details-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A direction of RNN primitive execution.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_rnn_direction_undef
.. _doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0af99114c11e7a82bdd1969ac59a10750f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_rnn_direction_undef

Undefined RNN direction.

.. index:: pair: enumvalue; dnnl_unidirectional_left2right
.. _doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a97cbdb39a208127cc83a9249517c5180:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_unidirectional_left2right

Unidirectional execution of RNN primitive from left to right.

.. index:: pair: enumvalue; dnnl_unidirectional_right2left
.. _doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0ad95dd744e0485467b82aacdbbe4590a1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_unidirectional_right2left

Unidirectional execution of RNN primitive from right to left.

.. index:: pair: enumvalue; dnnl_bidirectional_concat
.. _doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a69299415608e015c7334dc342d52743d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bidirectional_concat

Bidirectional execution of RNN primitive with concatenation of the results.

.. index:: pair: enumvalue; dnnl_bidirectional_sum
.. _doxid-group__dnnl__api__rnn_1gga629de1827647bf1824361a276c5169f0a6b39b772b540368ef1c80eee3ef1ff27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_bidirectional_sum

Bidirectional execution of RNN primitive with summation of the results.

