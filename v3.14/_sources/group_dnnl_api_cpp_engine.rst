.. index:: pair: group; Engine
.. _doxid-group__dnnl__api__cpp__engine:

Engine
======

.. toctree::
	:hidden:

	struct_dnnl_engine-2.rst

Overview
~~~~~~~~

An abstraction of a computational device: a CPU, a specific GPU card in the system, etc. :ref:`More...<details-group__dnnl__api__cpp__engine>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`;

	// global functions

	:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__engine_1gae472e59f404ba6527988b046ef24c743>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind);

.. _details-group__dnnl__api__cpp__engine:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An abstraction of a computational device: a CPU, a specific GPU card in the system, etc. Most primitives are created to execute computations on one specific engine. The only exceptions are reorder primitives that transfer data between two different engines.



.. rubric:: See also:

:ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__engine_1gae472e59f404ba6527988b046ef24c743:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` dnnl::convert_to_c(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind)

Converts engine kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API engine kind enum value.



.. rubric:: Returns:

Corresponding C API engine kind enum value.

