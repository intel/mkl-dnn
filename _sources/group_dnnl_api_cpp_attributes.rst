.. index:: pair: group; Attributes
.. _doxid-group__dnnl__api__cpp__attributes:

Attributes
==========

.. toctree::
	:hidden:

	enum_dnnl_algorithm.rst
	enum_dnnl_prop_kind.rst
	enum_dnnl_quantization_mode.rst
	enum_dnnl_rounding_mode.rst
	enum_dnnl_scratchpad_mode.rst
	struct_dnnl_post_ops-2.rst
	struct_dnnl_primitive_attr-2.rst

Overview
~~~~~~~~

A container for parameters that extend primitives behavior. :ref:`More...<details-group__dnnl__api__cpp__attributes>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::algorithm<doxid-group__dnnl__api__cpp__attributes_1ga00377dd4982333e42e8ae1d09a309640>`;
	enum :ref:`dnnl::prop_kind<doxid-group__dnnl__api__cpp__attributes_1gac7db48f6583aa9903e54c2a39d65438f>`;
	enum :ref:`dnnl::quantization_mode<doxid-group__dnnl__api__cpp__attributes_1ga43df4b809a4544d34bbc106d3e409b2c>`;
	enum :ref:`dnnl::rounding_mode<doxid-group__dnnl__api__cpp__attributes_1ga0ddf600480e2014503ce39429a7d0d7f>`;
	enum :ref:`dnnl::scratchpad_mode<doxid-group__dnnl__api__cpp__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>`;

	// structs

	struct :ref:`dnnl::post_ops<doxid-structdnnl_1_1post__ops>`;
	struct :ref:`dnnl::primitive_attr<doxid-structdnnl_1_1primitive__attr>`;

	// global functions

	:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__attributes_1gaa30f540e1ed09b2865f153fd599c967b>`(:ref:`scratchpad_mode<doxid-group__dnnl__api__cpp__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode);
	:ref:`dnnl_rounding_mode_t<doxid-group__dnnl__api__attributes_1gaf7a86e2c4b885ba1512aff12da7dadbb>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__attributes_1ga1444e4c2f7710b3f069fa76c355c4a1b>`(:ref:`rounding_mode<doxid-group__dnnl__api__cpp__attributes_1ga0ddf600480e2014503ce39429a7d0d7f>` mode);
	:ref:`dnnl_quantization_mode_t<doxid-group__dnnl__api__attributes_1ga5342e1d6b2a09ea01660b3a3c2400826>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__attributes_1ga6678e7dafb8afa412be59d2e9c615d0a>`(:ref:`quantization_mode<doxid-group__dnnl__api__cpp__attributes_1ga43df4b809a4544d34bbc106d3e409b2c>` qmode);
	:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__attributes_1gae13881206fecd43ce0e0daead7f0009e>`(:ref:`prop_kind<doxid-group__dnnl__api__cpp__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` akind);
	:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__cpp__attributes_1gad4c07d30e46391ce7ce0900d18cbfa30>`(:ref:`algorithm<doxid-group__dnnl__api__cpp__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm);

.. _details-group__dnnl__api__cpp__attributes:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A container for parameters that extend primitives behavior.

Attributes can also contain Post-ops, which are computations executed after the primitive.

A container for parameters that extend primitives behavior.



.. rubric:: See also:

:ref:`Primitive Attributes <doxid-dev_guide_attributes>`

:ref:`Post-ops <doxid-dev_guide_attributes_post_ops>`

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__attributes_1gaa30f540e1ed09b2865f153fd599c967b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_scratchpad_mode_t<doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>` dnnl::convert_to_c(:ref:`scratchpad_mode<doxid-group__dnnl__api__cpp__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode)

Converts a scratchpad mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API scratchpad mode enum value.



.. rubric:: Returns:

Corresponding C API scratchpad mode enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__attributes_1ga1444e4c2f7710b3f069fa76c355c4a1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_rounding_mode_t<doxid-group__dnnl__api__attributes_1gaf7a86e2c4b885ba1512aff12da7dadbb>` dnnl::convert_to_c(:ref:`rounding_mode<doxid-group__dnnl__api__cpp__attributes_1ga0ddf600480e2014503ce39429a7d0d7f>` mode)

Converts a rounding mode enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- C++ API rounding mode enum value.



.. rubric:: Returns:

Corresponding C API rounding mode enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__attributes_1ga6678e7dafb8afa412be59d2e9c615d0a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_quantization_mode_t<doxid-group__dnnl__api__attributes_1ga5342e1d6b2a09ea01660b3a3c2400826>` dnnl::convert_to_c(:ref:`quantization_mode<doxid-group__dnnl__api__cpp__attributes_1ga43df4b809a4544d34bbc106d3e409b2c>` qmode)

Converts a quantization kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- qmode

		- C++ API quantization kind enum value.



.. rubric:: Returns:

Corresponding C API quantization kind enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__attributes_1gae13881206fecd43ce0e0daead7f0009e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` dnnl::convert_to_c(:ref:`prop_kind<doxid-group__dnnl__api__cpp__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` akind)

Converts propagation kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API propagation kind enum value.



.. rubric:: Returns:

Corresponding C API propagation kind enum value.

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__cpp__attributes_1gad4c07d30e46391ce7ce0900d18cbfa30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` dnnl::convert_to_c(:ref:`algorithm<doxid-group__dnnl__api__cpp__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm)

Converts algorithm kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aalgorithm

		- C++ API algorithm kind enum value.



.. rubric:: Returns:

Corresponding C API algorithm kind enum value.

