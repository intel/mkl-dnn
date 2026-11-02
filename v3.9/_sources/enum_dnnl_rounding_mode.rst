.. index:: pair: enum; rounding_mode
.. _doxid-group__dnnl__api__attributes_1ga0ddf600480e2014503ce39429a7d0d7f:

enum dnnl::rounding_mode
========================

Overview
~~~~~~~~

Rounding mode. :ref:`More...<details-group__dnnl__api__attributes_1ga0ddf600480e2014503ce39429a7d0d7f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum rounding_mode
	{
	    :ref:`environment<doxid-group__dnnl__api__attributes_1gga0ddf600480e2014503ce39429a7d0d7fae900e40bc91d3f9f7f0a99fed68a2e96>` = dnnl_rounding_mode_environment,
	    :ref:`stochastic<doxid-group__dnnl__api__attributes_1gga0ddf600480e2014503ce39429a7d0d7fa4f3690e0a031da4a20e46c2f0bee0c15>`  = dnnl_rounding_mode_stochastic,
	};

.. _details-group__dnnl__api__attributes_1ga0ddf600480e2014503ce39429a7d0d7f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Rounding mode.

Enum Values
-----------

.. index:: pair: enumvalue; environment
.. _doxid-group__dnnl__api__attributes_1gga0ddf600480e2014503ce39429a7d0d7fae900e40bc91d3f9f7f0a99fed68a2e96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	environment

rounding mode dictated by the floating-point environment

.. index:: pair: enumvalue; stochastic
.. _doxid-group__dnnl__api__attributes_1gga0ddf600480e2014503ce39429a7d0d7fa4f3690e0a031da4a20e46c2f0bee0c15:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	stochastic

stochastic rounding mode where a random bias is added to the trailing mantissa bits before conversion.

