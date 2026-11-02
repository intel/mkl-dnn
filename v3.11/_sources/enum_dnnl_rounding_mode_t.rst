.. index:: pair: enum; dnnl_rounding_mode_t
.. _doxid-group__dnnl__api__attributes_1gaf7a86e2c4b885ba1512aff12da7dadbb:

enum dnnl_rounding_mode_t
=========================

Overview
~~~~~~~~

Rounding mode. :ref:`More...<details-group__dnnl__api__attributes_1gaf7a86e2c4b885ba1512aff12da7dadbb>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_rounding_mode_t
	{
	    :ref:`dnnl_rounding_mode_environment<doxid-group__dnnl__api__attributes_1ggaf7a86e2c4b885ba1512aff12da7dadbba2210334bac23a60d81e5fffe48bfe61e>`,
	    :ref:`dnnl_rounding_mode_stochastic<doxid-group__dnnl__api__attributes_1ggaf7a86e2c4b885ba1512aff12da7dadbbaeb079ac10512faf9c37d062f387558bb>`,
	};

.. _details-group__dnnl__api__attributes_1gaf7a86e2c4b885ba1512aff12da7dadbb:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Rounding mode.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_rounding_mode_environment
.. _doxid-group__dnnl__api__attributes_1ggaf7a86e2c4b885ba1512aff12da7dadbba2210334bac23a60d81e5fffe48bfe61e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_rounding_mode_environment

rounding mode dictated by the floating-point environment

.. index:: pair: enumvalue; dnnl_rounding_mode_stochastic
.. _doxid-group__dnnl__api__attributes_1ggaf7a86e2c4b885ba1512aff12da7dadbbaeb079ac10512faf9c37d062f387558bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_rounding_mode_stochastic

stochastic rounding mode where a random bias is added to the trailing mantissa bits before conversion.

