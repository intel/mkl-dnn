.. index:: pair: struct; dnnl_version_t
.. _doxid-structdnnl__version__t:

struct dnnl_version_t
=====================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Structure containing version information as per `Semantic Versioning <https://semver.org>`__ :ref:`More...<details-structdnnl__version__t>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>
	
	struct dnnl_version_t
	{
		// fields
	
		int :ref:`major<doxid-structdnnl__version__t_1ab0051e104a0b84b25c22f2ed98a9a16c>`;
		int :ref:`minor<doxid-structdnnl__version__t_1a30f17c81ffa45f1929bbe4f2c14fe42f>`;
		int :ref:`patch<doxid-structdnnl__version__t_1aa29d1e7545c879cafc88426086443d1b>`;
		const char* :ref:`hash<doxid-structdnnl__version__t_1a1c2a7e8b3f26ed376b537dd511feccad>`;
		unsigned :ref:`cpu_runtime<doxid-structdnnl__version__t_1a2d0bade7e7ab7ff25d68f9011a3978e2>`;
		unsigned :ref:`gpu_runtime<doxid-structdnnl__version__t_1ac15e5566f96a65b2c97903c787f129ac>`;
	};
.. _details-structdnnl__version__t:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Structure containing version information as per `Semantic Versioning <https://semver.org>`__

Fields
------

.. index:: pair: variable; major
.. _doxid-structdnnl__version__t_1ab0051e104a0b84b25c22f2ed98a9a16c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int major

Major version.

.. index:: pair: variable; minor
.. _doxid-structdnnl__version__t_1a30f17c81ffa45f1929bbe4f2c14fe42f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int minor

Minor version.

.. index:: pair: variable; patch
.. _doxid-structdnnl__version__t_1aa29d1e7545c879cafc88426086443d1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int patch

Patch version.

.. index:: pair: variable; hash
.. _doxid-structdnnl__version__t_1a1c2a7e8b3f26ed376b537dd511feccad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const char* hash

Git hash of the sources (may be absent)

.. index:: pair: variable; cpu_runtime
.. _doxid-structdnnl__version__t_1a2d0bade7e7ab7ff25d68f9011a3978e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned cpu_runtime

CPU runtime.

.. index:: pair: variable; gpu_runtime
.. _doxid-structdnnl__version__t_1ac15e5566f96a65b2c97903c787f129ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned gpu_runtime

GPU runtime.

