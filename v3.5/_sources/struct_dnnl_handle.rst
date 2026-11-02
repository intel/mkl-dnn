.. index:: pair: struct; dnnl::handle
.. _doxid-structdnnl_1_1handle:

template struct dnnl::handle
============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

oneDNN C API handle wrapper class. :ref:`More...<details-structdnnl_1_1handle>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>
	
	template <typename T, typename traits = handle_traits<T>>
	struct handle
	{
		// construction
	
		:ref:`handle<doxid-structdnnl_1_1handle_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1handle_1a022001b5b9c8940a1326a02b61fc4860>`(const handle<T, traits>&);
		:ref:`handle<doxid-structdnnl_1_1handle_1aa13f3ecf4db240717074814412c7e70c>`(handle<T, traits>&&);
		:ref:`handle<doxid-structdnnl_1_1handle_1a9c408c09fce1278f5cb0d1fa9818fc86>`(T t, bool weak = false);

		// methods
	
		handle<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52>` (const handle<T, traits>&);
		handle<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727>` (handle<T, traits>&&);
		void :ref:`reset<doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909>`(T t, bool weak = false);
		T :ref:`get<doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e>`(bool allow_empty = false) const;
		:ref:`operator T<doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3>` () const;
		:ref:`operator bool<doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5>` () const;
		bool :ref:`operator ==<doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e>` (const handle<T, traits>& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab>` (const handle& other) const;
	};

	// direct descendants

	struct :ref:`brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm>`;
	struct :ref:`brgemm_pack_b<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b>`;
	struct :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`;
	struct :ref:`engine<doxid-structdnnl_1_1engine>`;
	struct :ref:`memory<doxid-structdnnl_1_1memory>`;
	struct :ref:`post_ops<doxid-structdnnl_1_1post__ops>`;
	struct :ref:`primitive<doxid-structdnnl_1_1primitive>`;
	struct :ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>`;
	struct :ref:`primitive_desc_base<doxid-structdnnl_1_1primitive__desc__base>`;
	struct :ref:`stream<doxid-structdnnl_1_1stream>`;
.. _details-structdnnl_1_1handle:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

oneDNN C API handle wrapper class.

This class is used as the base class for primitive (:ref:`dnnl::primitive <doxid-structdnnl_1_1primitive>`), engine (:ref:`dnnl::engine <doxid-structdnnl_1_1engine>`), and stream (:ref:`dnnl::stream <doxid-structdnnl_1_1stream>`) classes, as well as others. An object of the :ref:`dnnl::handle <doxid-structdnnl_1_1handle>` class can be passed by value.

A handle can be weak, in which case it follows std::weak_ptr semantics. Otherwise, it follows ``std::shared_ptr`` semantics.

.. note:: 

   The implementation stores oneDNN C API handles in a ``std::shared_ptr`` with deleter set to a dummy function in the weak mode.

Construction
------------

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1handle_1a5c631f7e5e4c92a13edb8e3422d3a973:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs an empty handle object.

.. warning:: 

   Uninitialized object cannot be used in most library calls and is equivalent to a null pointer. Any attempt to use its methods, or passing it to the other library function, will cause an exception to be thrown.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1handle_1a022001b5b9c8940a1326a02b61fc4860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle(const handle<T, traits>&)

Copy constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1handle_1aa13f3ecf4db240717074814412c7e70c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle(handle<T, traits>&&)

Move constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1handle_1a9c408c09fce1278f5cb0d1fa9818fc86:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle(T t, bool weak = false)

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

Methods
-------

.. index:: pair: function; operator=
.. _doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle<T, traits>& operator = (const handle<T, traits>&)

Assignment operator.

.. index:: pair: function; operator=
.. _doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle<T, traits>& operator = (handle<T, traits>&&)

Move assignment operator.

.. index:: pair: function; reset
.. _doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void reset(T t, bool weak = false)

Resets the handle wrapper objects to wrap a new C API handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t

		- The new value of the C API handle.

	*
		- weak

		- A flag specifying whether the wrapper should be weak; defaults to ``false``.

.. index:: pair: function; get
.. _doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	T get(bool allow_empty = false) const

Returns the underlying C API handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- allow_empty

		- A flag signifying whether the method is allowed to return an empty (null) object without throwing an exception.



.. rubric:: Returns:

The underlying C API handle.

.. index:: pair: function; operator T
.. _doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	operator T () const

Converts a handle to the underlying C API handle type.

Does not throw and returns ``nullptr`` if the object is empty.



.. rubric:: Returns:

The underlying C API handle.

.. index:: pair: function; operator bool
.. _doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	operator bool () const

Checks whether the object is not empty.



.. rubric:: Returns:

Whether the object is not empty.

.. index:: pair: function; operator==
.. _doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool operator == (const handle<T, traits>& other) const

Equality operator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- other

		- Another handle wrapper.



.. rubric:: Returns:

``true`` if this and the other handle wrapper manage the same underlying C API handle, and ``false`` otherwise. Empty handle objects are considered to be equal.

.. index:: pair: function; operator!=
.. _doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool operator != (const handle& other) const

Inequality operator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- other

		- Another handle wrapper.



.. rubric:: Returns:

``true`` if this and the other handle wrapper manage different underlying C API handles, and ``false`` otherwise. Empty handle objects are considered to be equal.

