.. index:: pair: struct; dnnl::engine
.. _doxid-structdnnl_1_1engine:

struct dnnl::engine
===================

.. toctree::
	:hidden:

	enum_dnnl_engine_kind.rst

Overview
~~~~~~~~

An execution engine. :ref:`More...<details-structdnnl_1_1engine>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>
	
	struct engine: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// enums
	
		enum :ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>`;

		// construction
	
		:ref:`engine<doxid-structdnnl_1_1engine_1a400b0f6aacb6ce747d2aa747be23fa19>`();
		:ref:`engine<doxid-structdnnl_1_1engine_1a2e23ffff24322cc9d3e108a098a88d31>`(:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind, size_t index);

		// methods
	
		:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` :ref:`get_kind<doxid-structdnnl_1_1engine_1afe81649a67fff16348ca2b7e15f78c3b>`() const;
		:ref:`handle<doxid-structdnnl_1_1engine_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1engine_1a022001b5b9c8940a1326a02b61fc4860>`();
		:ref:`handle<doxid-structdnnl_1_1engine_1aa13f3ecf4db240717074814412c7e70c>`();
		:ref:`handle<doxid-structdnnl_1_1engine_1a9c408c09fce1278f5cb0d1fa9818fc86>`();
		static size_t :ref:`get_count<doxid-structdnnl_1_1engine_1a4377c19abca7fc5d761bb5e512c732d1>`(:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&);
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727>` (:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&&);
		void :ref:`reset<doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909>`(T t, bool weak = false);
		T :ref:`get<doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e>`(bool allow_empty = false) const;
		:ref:`operator T<doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3>` () const;
		:ref:`operator bool<doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5>` () const;
		bool :ref:`operator ==<doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab>` (const :ref:`handle<doxid-structdnnl_1_1handle>`& other) const;

.. _details-structdnnl_1_1engine:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An execution engine.

Construction
------------

.. index:: pair: function; engine
.. _doxid-structdnnl_1_1engine_1a400b0f6aacb6ce747d2aa747be23fa19:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	engine()

Constructs an empty engine.

An empty engine cannot be used in any operations.

.. index:: pair: function; engine
.. _doxid-structdnnl_1_1engine_1a2e23ffff24322cc9d3e108a098a88d31:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	engine(:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind, size_t index)

Constructs an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- The kind of engine to construct.

	*
		- index

		- The index of the engine. Must be less than the value returned by :ref:`get_count() <doxid-structdnnl_1_1engine_1a4377c19abca7fc5d761bb5e512c732d1>` for this particular kind of engine.

Methods
-------

.. index:: pair: function; get_kind
.. _doxid-structdnnl_1_1engine_1afe81649a67fff16348ca2b7e15f78c3b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` get_kind() const

Returns the kind of the engine.



.. rubric:: Returns:

The kind of the engine.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1engine_1a5c631f7e5e4c92a13edb8e3422d3a973:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs an empty handle object.

.. warning:: 

   Uninitialized object cannot be used in most library calls and is equivalent to a null pointer. Any attempt to use its methods, or passing it to the other library function, will cause an exception to be thrown.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1engine_1a022001b5b9c8940a1326a02b61fc4860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Copy constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1engine_1aa13f3ecf4db240717074814412c7e70c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Move constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1engine_1a9c408c09fce1278f5cb0d1fa9818fc86:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs a handle wrapper object from a C API handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t

		- The C API handle to wrap.

	*
		- weak

		- A flag specifying whether to construct a weak wrapper; defaults to ``false``.

.. index:: pair: function; get_count
.. _doxid-structdnnl_1_1engine_1a4377c19abca7fc5d761bb5e512c732d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static size_t get_count(:ref:`kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind)

Returns the number of engines of a certain kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- The kind of engines to count.



.. rubric:: Returns:

The number of engines of the specified kind.

