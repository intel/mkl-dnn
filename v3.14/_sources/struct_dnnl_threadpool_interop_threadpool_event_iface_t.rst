.. index:: pair: struct; dnnl::threadpool_interop::threadpool_event_iface_t
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t:

struct dnnl::threadpool_interop::threadpool_event_iface_t
=========================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Completion event interface for async threadpool work. :ref:`More...<details-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_threadpool_iface.hpp>
	
	struct threadpool_event_iface_t
	{
		// methods
	
		virtual bool :ref:`is_complete<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1acd8ff64ed52794a432a7eb7b3ae7acb8>`() const = 0;
		virtual void :ref:`wait<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a2edbf1e530f1747bf4be58728f591635>`() const = 0;
		virtual double :ref:`exec_time_ms<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a1e411516ce3e1fb339c4507ce7d635bd>`() const = 0;
	};
.. _details-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Completion event interface for async threadpool work. oneDNN's verbose profiler polls these to determine when deferred execution has finished and to extract timing information.

Methods
-------

.. index:: pair: function; is_complete
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1acd8ff64ed52794a432a7eb7b3ae7acb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool is_complete() const = 0

Returns true if the associated work has completed.

.. index:: pair: function; wait
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a2edbf1e530f1747bf4be58728f591635:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void wait() const = 0

Blocks until completion.

.. index:: pair: function; exec_time_ms
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a1e411516ce3e1fb339c4507ce7d635bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual double exec_time_ms() const = 0

Measured execution time in milliseconds. Valid only after completion. Should reflect wall-clock duration from when the first worker begins to when the last worker finishes. Use a compare-and-swap atomic operation to stamp the start time so only the first worker records the asynchronous completion callback once all workers have finished.

