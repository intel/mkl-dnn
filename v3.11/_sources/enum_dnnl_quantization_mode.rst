.. index:: pair: enum; quantization_mode
.. _doxid-group__dnnl__api__attributes_1ga43df4b809a4544d34bbc106d3e409b2c:

enum dnnl::quantization_mode
============================

Overview
~~~~~~~~

Quantization kind. :ref:`More...<details-group__dnnl__api__attributes_1ga43df4b809a4544d34bbc106d3e409b2c>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum quantization_mode
	{
	    :ref:`undef<doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2caf31ee5e3824f1f5e5d206bdf3029f22b>`       = dnnl_quantization_mode_undef,
	    :ref:`static_sazp<doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca000df1fbb9ecf80708c726fb84d674c3>` = dnnl_quantization_mode_static_sazp,
	    :ref:`dynamic_mx<doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca10dabb84b08ade6e41ee83eba1e96f9d>`  = dnnl_quantization_mode_dynamic_mx,
	    :ref:`dynamic_fp<doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2caf962028f6156d4785b104f7f0f362685>`  = dnnl_quantization_mode_dynamic_fp,
	};

.. _details-group__dnnl__api__attributes_1ga43df4b809a4544d34bbc106d3e409b2c:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Quantization kind.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2caf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

used for unspecified quantization kind

.. index:: pair: enumvalue; static_sazp
.. _doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca000df1fbb9ecf80708c726fb84d674c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static_sazp

static quantization mode: quantization parameter is computed ahead of time and passed to oneDNN as an input.

.. index:: pair: enumvalue; dynamic_mx
.. _doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca10dabb84b08ade6e41ee83eba1e96f9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dynamic_mx

dynamic quantization mode following OCP MX spec: quantization parameter is computed by oneDNN following the OCP MX spec formula and written as an output.

.. index:: pair: enumvalue; dynamic_fp
.. _doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2caf962028f6156d4785b104f7f0f362685:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dynamic_fp

dynamic quantization mode where quantization parameter is computed by oneDNN as :math:`scale\_dt(amax(X) / max(dst\_dt))` in ``f32`` then converted to a scale type and written as an output.

