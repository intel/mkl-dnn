.. index:: pair: struct; dnnl::error
.. _doxid-structdnnl_1_1error:

struct dnnl::error
==================

.. toctree::
	:hidden:

Overview
~~~~~~~~

oneDNN exception class. :ref:`More...<details-structdnnl_1_1error>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>
	
	struct error: public exception
	{
		// fields
	
		:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` :target:`status<doxid-structdnnl_1_1error_1a9a75be95ed102125ca4e77805f87d28d>`;
		const char* :target:`message<doxid-structdnnl_1_1error_1a0812e8eccd21c861f150839422eacc38>`;

		// construction
	
		:ref:`error<doxid-structdnnl_1_1error_1a67984998eff9083077daf1cd1a1acb01>`(:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` status, const char* message);

		// methods
	
		const char* :ref:`what<doxid-structdnnl_1_1error_1afcf188632b6264fba24f3300dabd9b65>`() const;
		static void :ref:`wrap_c_api<doxid-structdnnl_1_1error_1a9d91127d0524c0b7ac1ae4ba4c79d0af>`(:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` status, const char* message);
	};
.. _details-structdnnl_1_1error:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

oneDNN exception class.

This class captures the status returned by a failed C API function and the error message from the call site.

Construction
------------

.. index:: pair: function; error
.. _doxid-structdnnl_1_1error_1a67984998eff9083077daf1cd1a1acb01:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	error(:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` status, const char* message)

Constructs an instance of an exception class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- status

		- The error status returned by a C API function.

	*
		- message

		- The error message.

Methods
-------

.. index:: pair: function; what
.. _doxid-structdnnl_1_1error_1afcf188632b6264fba24f3300dabd9b65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const char* what() const

Returns the explanatory string.

.. index:: pair: function; wrap_c_api
.. _doxid-structdnnl_1_1error_1a9d91127d0524c0b7ac1ae4ba4c79d0af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static void wrap_c_api(:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` status, const char* message)

A convenience function for wrapping calls to C API functions.

Checks the return status and throws an :ref:`dnnl::error <doxid-structdnnl_1_1error>` in case of failure.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- status

		- The error status returned by a C API function.

	*
		- message

		- The error message.

