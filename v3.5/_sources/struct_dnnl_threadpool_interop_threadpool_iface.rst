.. index:: pair: struct; dnnl::threadpool_interop::threadpool_iface
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface:

struct dnnl::threadpool_interop::threadpool_iface
=================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Abstract threadpool interface. :ref:`More...<details-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_threadpool_iface.hpp>
	
	struct threadpool_iface
	{
		// fields
	
		static constexpr uint64_t :ref:`ASYNCHRONOUS<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a9e6d861d659445fe5abcf302e464d9e5>` = 1;

		// methods
	
		virtual int :ref:`get_num_threads<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114>`() const = 0;
		virtual bool :ref:`get_in_parallel<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a8279221c6e2f903a4c811688f7a033be>`() const = 0;
		virtual void :ref:`parallel_for<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1ac3d85ff935c11ec038ecabeeabd03ffb>`(int n, const std::function<void(int, int)>& fn) = 0;
		virtual uint64_t :ref:`get_flags<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a868267178f259cee5f1d5b33a8781a3e>`() const = 0;
	};
.. _details-structdnnl_1_1threadpool__interop_1_1threadpool__iface:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Abstract threadpool interface.

The users are expected to subclass this interface and pass an object to the library during CPU stream creation or directly in case of BLAS functions.

Fields
------

.. index:: pair: variable; ASYNCHRONOUS
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a9e6d861d659445fe5abcf302e464d9e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static constexpr uint64_t ASYNCHRONOUS = 1

If set, :ref:`parallel_for() <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1ac3d85ff935c11ec038ecabeeabd03ffb>` returns immediately and oneDNN needs implement waiting for the submitted closures to finish execution on its own.

Methods
-------

.. index:: pair: function; get_num_threads
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual int get_num_threads() const = 0

Returns the number of worker threads.

.. index:: pair: function; get_in_parallel
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a8279221c6e2f903a4c811688f7a033be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool get_in_parallel() const = 0

Returns true if the calling thread belongs to this threadpool.

.. index:: pair: function; parallel_for
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1ac3d85ff935c11ec038ecabeeabd03ffb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void parallel_for(int n, const std::function<void(int, int)>& fn) = 0

Submits n instances of a closure for execution in parallel:

for (int i = 0; i < n; i++) fn(i, n);

.. index:: pair: function; get_flags
.. _doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a868267178f259cee5f1d5b33a8781a3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual uint64_t get_flags() const = 0

Returns threadpool behavior flags bit mask (see below).

