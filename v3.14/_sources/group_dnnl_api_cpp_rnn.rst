.. index:: pair: group; RNN
.. _doxid-group__dnnl__api__cpp__rnn:

RNN
===

.. toctree::
	:hidden:

	enum_dnnl_rnn_direction.rst
	enum_dnnl_rnn_flags.rst
	struct_dnnl_augru_backward.rst
	struct_dnnl_augru_forward.rst
	struct_dnnl_gru_backward.rst
	struct_dnnl_gru_forward.rst
	struct_dnnl_lbr_augru_backward.rst
	struct_dnnl_lbr_augru_forward.rst
	struct_dnnl_lbr_gru_backward.rst
	struct_dnnl_lbr_gru_forward.rst
	struct_dnnl_lstm_backward.rst
	struct_dnnl_lstm_forward.rst
	struct_dnnl_rnn_primitive_desc_base.rst
	struct_dnnl_vanilla_rnn_backward.rst
	struct_dnnl_vanilla_rnn_forward.rst

Overview
~~~~~~~~

A primitive to compute recurrent neural network layers. :ref:`More...<details-group__dnnl__api__cpp__rnn>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::rnn_direction<doxid-group__dnnl__api__cpp__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>`;
	enum :ref:`dnnl::rnn_flags<doxid-group__dnnl__api__cpp__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>`;

	// structs

	struct :ref:`dnnl::augru_backward<doxid-structdnnl_1_1augru__backward>`;
	struct :ref:`dnnl::augru_forward<doxid-structdnnl_1_1augru__forward>`;
	struct :ref:`dnnl::gru_backward<doxid-structdnnl_1_1gru__backward>`;
	struct :ref:`dnnl::gru_forward<doxid-structdnnl_1_1gru__forward>`;
	struct :ref:`dnnl::lbr_augru_backward<doxid-structdnnl_1_1lbr__augru__backward>`;
	struct :ref:`dnnl::lbr_augru_forward<doxid-structdnnl_1_1lbr__augru__forward>`;
	struct :ref:`dnnl::lbr_gru_backward<doxid-structdnnl_1_1lbr__gru__backward>`;
	struct :ref:`dnnl::lbr_gru_forward<doxid-structdnnl_1_1lbr__gru__forward>`;
	struct :ref:`dnnl::lstm_backward<doxid-structdnnl_1_1lstm__backward>`;
	struct :ref:`dnnl::lstm_forward<doxid-structdnnl_1_1lstm__forward>`;
	struct :ref:`dnnl::rnn_primitive_desc_base<doxid-structdnnl_1_1rnn__primitive__desc__base>`;
	struct :ref:`dnnl::vanilla_rnn_backward<doxid-structdnnl_1_1vanilla__rnn__backward>`;
	struct :ref:`dnnl::vanilla_rnn_forward<doxid-structdnnl_1_1vanilla__rnn__forward>`;

	// global functions

	:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__rnn_1ga0a340195a137f906e858418d91397777>`(:ref:`rnn_flags<doxid-group__dnnl__api__cpp__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>` flags);
	:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__rnn_1ga1915ea2d2fe94077fa30734ced88a225>`(:ref:`rnn_direction<doxid-group__dnnl__api__cpp__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` dir);

.. _details-group__dnnl__api__cpp__rnn:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to compute recurrent neural network layers.



.. rubric:: See also:

:ref:`RNN <doxid-dev_guide_rnn>` in developer guide

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__rnn_1ga0a340195a137f906e858418d91397777:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_rnn_flags_t<doxid-group__dnnl__api__rnn_1ga3e71b827ee442f0302111d214a6d35b5>` dnnl::convert_to_c(:ref:`rnn_flags<doxid-group__dnnl__api__cpp__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>` flags)

Converts RNN cell flags enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flags

		- C++ API RNN cell flags enum value.



.. rubric:: Returns:

Corresponding C API RNN cell flags enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__rnn_1ga1915ea2d2fe94077fa30734ced88a225:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_rnn_direction_t<doxid-group__dnnl__api__rnn_1ga629de1827647bf1824361a276c5169f0>` dnnl::convert_to_c(:ref:`rnn_direction<doxid-group__dnnl__api__cpp__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` dir)

Converts RNN direction enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- dir

		- C++ API RNN direction enum value.



.. rubric:: Returns:

Corresponding C API RNN direction enum value.

