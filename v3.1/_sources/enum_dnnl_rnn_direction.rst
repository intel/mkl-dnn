.. index:: pair: enum; rnn_direction
.. _doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5:

enum dnnl::rnn_direction
========================

Overview
~~~~~~~~

A direction of RNN primitive execution. :ref:`More...<details-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum rnn_direction
	{
	    :ref:`undef<doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5af31ee5e3824f1f5e5d206bdf3029f22b>`                     = dnnl_rnn_direction_undef,
	    :ref:`unidirectional_left2right<doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>` = dnnl_unidirectional_left2right,
	    :ref:`unidirectional_right2left<doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a0b694765eed7cf5a48e18c1d05b74118>` = dnnl_unidirectional_right2left,
	    :ref:`bidirectional_concat<doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a7a1bb9f8699e8c03cbe4bd681fb50830>`      = dnnl_bidirectional_concat,
	    :ref:`bidirectional_sum<doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5aa6199c5b651803844c8c054b11e88d8c>`         = dnnl_bidirectional_sum,
	};

.. _details-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A direction of RNN primitive execution.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined RNN direction.

.. index:: pair: enumvalue; unidirectional_left2right
.. _doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unidirectional_left2right

Unidirectional execution of RNN primitive from left to right.

.. index:: pair: enumvalue; unidirectional_right2left
.. _doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a0b694765eed7cf5a48e18c1d05b74118:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unidirectional_right2left

Unidirectional execution of RNN primitive from right to left.

.. index:: pair: enumvalue; bidirectional_concat
.. _doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a7a1bb9f8699e8c03cbe4bd681fb50830:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bidirectional_concat

Bidirectional execution of RNN primitive with concatenation of the results.

.. index:: pair: enumvalue; bidirectional_sum
.. _doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5aa6199c5b651803844c8c054b11e88d8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bidirectional_sum

Bidirectional execution of RNN primitive with summation of the results.

