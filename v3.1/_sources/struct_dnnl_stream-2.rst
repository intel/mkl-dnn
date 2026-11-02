.. index:: pair: struct; dnnl::stream
.. _doxid-structdnnl_1_1stream:

struct dnnl::stream
===================

.. toctree::
	:hidden:

	enum_dnnl_stream_flags.rst

Overview
~~~~~~~~

An execution stream. :ref:`More...<details-structdnnl_1_1stream>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>
	
	struct stream: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// enums
	
		enum :ref:`flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59>`;

		// construction
	
		:ref:`stream<doxid-structdnnl_1_1stream_1ad264b142d2f7c728fe8c84aa65d62b13>`();
		:ref:`stream<doxid-structdnnl_1_1stream_1a91cddfb27bc0d7577f8f9ff0327ca598>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, :ref:`flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59>` aflags = :ref:`flags::default_flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59aeffb2d149f637ca450767d77cd927108>`);

		// methods
	
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`get_engine<doxid-structdnnl_1_1stream_1a1f00b7d78e3de80b5e1d62dd26b986bc>`() const;
		stream& :ref:`wait<doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
		:ref:`handle<doxid-structdnnl_1_1stream_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1stream_1a022001b5b9c8940a1326a02b61fc4860>`();
		:ref:`handle<doxid-structdnnl_1_1stream_1aa13f3ecf4db240717074814412c7e70c>`();
		:ref:`handle<doxid-structdnnl_1_1stream_1a9c408c09fce1278f5cb0d1fa9818fc86>`();
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

.. _details-structdnnl_1_1stream:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An execution stream.

Construction
------------

.. index:: pair: function; stream
.. _doxid-structdnnl_1_1stream_1ad264b142d2f7c728fe8c84aa65d62b13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	stream()

Constructs an empty stream.

An empty stream cannot be used in any operations.

.. index:: pair: function; stream
.. _doxid-structdnnl_1_1stream_1a91cddfb27bc0d7577f8f9ff0327ca598:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	stream(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, :ref:`flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59>` aflags = :ref:`flags::default_flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59aeffb2d149f637ca450767d77cd927108>`)

Constructs a stream for the specified engine and with behavior controlled by the specified flags.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to create the stream on.

	*
		- aflags

		- Flags controlling stream behavior.

Methods
-------

.. index:: pair: function; get_engine
.. _doxid-structdnnl_1_1stream_1a1f00b7d78e3de80b5e1d62dd26b986bc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` get_engine() const

Returns the associated engine.

.. index:: pair: function; wait
.. _doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	stream& wait()

Waits for all primitives executing in the stream to finish.



.. rubric:: Returns:

The stream itself.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1stream_1a5c631f7e5e4c92a13edb8e3422d3a973:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs an empty handle object.

.. warning:: 

   Uninitialized object cannot be used in most library calls and is equivalent to a null pointer. Any attempt to use its methods, or passing it to the other library function, will cause an exception to be thrown.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1stream_1a022001b5b9c8940a1326a02b61fc4860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Copy constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1stream_1aa13f3ecf4db240717074814412c7e70c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Move constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1stream_1a9c408c09fce1278f5cb0d1fa9818fc86:

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

