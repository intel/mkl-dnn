.. index:: pair: enum; dnnl_quantization_mode_t
.. _doxid-group__dnnl__api__attributes_1ga5342e1d6b2a09ea01660b3a3c2400826:

enum dnnl_quantization_mode_t
=============================

Overview
~~~~~~~~

Quantization kind. :ref:`More...<details-group__dnnl__api__attributes_1ga5342e1d6b2a09ea01660b3a3c2400826>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_quantization_mode_t
	{
	    :ref:`dnnl_quantization_mode_undef<doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a3d9f31450fce98be57ec1d10839a2ab8>`,
	    :ref:`dnnl_quantization_mode_static_sazp<doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a69f0c0079f39bec2332481404e199315>`,
	    :ref:`dnnl_quantization_mode_dynamic_mx<doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a9d29b3c3bf3c43cab388533e093cd8a6>`,
	    :ref:`dnnl_quantization_mode_dynamic_fp<doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a3978acf7ac10a5dbf6376b1472b002be>`,
	};

.. _details-group__dnnl__api__attributes_1ga5342e1d6b2a09ea01660b3a3c2400826:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Quantization kind.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_quantization_mode_undef
.. _doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a3d9f31450fce98be57ec1d10839a2ab8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_quantization_mode_undef

used for unspecified quantization kind

.. index:: pair: enumvalue; dnnl_quantization_mode_static_sazp
.. _doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a69f0c0079f39bec2332481404e199315:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_quantization_mode_static_sazp

static quantization mode: quantization parameter is computed ahead of time with scale applied after zero-point (:math:`x_{f32} = scale * (x_{quant} - zp)`) and passed to oneDNN as an input.

.. index:: pair: enumvalue; dnnl_quantization_mode_dynamic_mx
.. _doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a9d29b3c3bf3c43cab388533e093cd8a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_quantization_mode_dynamic_mx

dynamic quantization mode following OCP MX spec: quantization parameter is computed by oneDNN following the OCP MX spec formula and written as an output.

.. index:: pair: enumvalue; dnnl_quantization_mode_dynamic_fp
.. _doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a3978acf7ac10a5dbf6376b1472b002be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_quantization_mode_dynamic_fp

dynamic quantization mode where quantization parameter is computed by oneDNN as :math:`scale\_dt(amax(X) / max(dst\_dt))` in ``f32`` then converted to a scale type and written as an output.

