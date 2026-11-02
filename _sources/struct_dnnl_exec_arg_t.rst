.. index:: pair: struct; dnnl_exec_arg_t
.. _doxid-structdnnl__exec__arg__t:

struct dnnl_exec_arg_t
======================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A structure that contains an index and a memory object, and is used to pass arguments to :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`. :ref:`More...<details-structdnnl__exec__arg__t>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>
	
	struct dnnl_exec_arg_t
	{
		// fields
	
		int :ref:`arg<doxid-structdnnl__exec__arg__t_1a46c7f77870713b8af3fd37dc66e9690b>`;
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` :ref:`memory<doxid-structdnnl__exec__arg__t_1a048f23e80b923636267c4dece912cd0d>`;
	};
.. _details-structdnnl__exec__arg__t:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A structure that contains an index and a memory object, and is used to pass arguments to :ref:`dnnl_primitive_execute() <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`.

Fields
------

.. index:: pair: variable; arg
.. _doxid-structdnnl__exec__arg__t_1a46c7f77870713b8af3fd37dc66e9690b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int arg

An argument index, e.g. DNNL_ARG_SRC.

.. index:: pair: variable; memory
.. _doxid-structdnnl__exec__arg__t_1a048f23e80b923636267c4dece912cd0d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory

Input/output memory.

