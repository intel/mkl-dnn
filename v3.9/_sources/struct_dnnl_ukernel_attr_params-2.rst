.. index:: pair: struct; dnnl::ukernel::attr_params
.. _doxid-structdnnl_1_1ukernel_1_1attr__params:

struct dnnl::ukernel::attr_params
=================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Ukernel attributes memory storage. :ref:`More...<details-structdnnl_1_1ukernel_1_1attr__params>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ukernel.hpp>
	
	struct attr_params: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// methods
	
		void :ref:`set_post_ops_args<doxid-structdnnl_1_1ukernel_1_1attr__params_1af991f15932b7c0fef737cdc61dd56de0>`(const void** post_ops_args);
		void :ref:`set_A_scales<doxid-structdnnl_1_1ukernel_1_1attr__params_1a30e65d3632b42c55daceeb6c1c6fa5ae>`(const void* a_scales);
		void :ref:`set_B_scales<doxid-structdnnl_1_1ukernel_1_1attr__params_1a9e2c17ea304a349479bc36124b08e200>`(const void* b_scales);
		void :ref:`set_D_scales<doxid-structdnnl_1_1ukernel_1_1attr__params_1aad9de5f26f633ab67e22953942b8ad10>`(const void* d_scales);
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

.. _details-structdnnl_1_1ukernel_1_1attr__params:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Ukernel attributes memory storage.

Methods
-------

.. index:: pair: function; set_post_ops_args
.. _doxid-structdnnl_1_1ukernel_1_1attr__params_1af991f15932b7c0fef737cdc61dd56de0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_post_ops_args(const void** post_ops_args)

Sets post-operations arguments to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- post_ops_args

		- Pointer to pointers of :ref:`post_ops <doxid-structdnnl_1_1post__ops>` storages. Expected to be packed together.

.. index:: pair: function; set_A_scales
.. _doxid-structdnnl_1_1ukernel_1_1attr__params_1a30e65d3632b42c55daceeb6c1c6fa5ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_A_scales(const void* a_scales)

Sets tensor A scales arguments to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- a_scales

		- Pointer to scales storage.

.. index:: pair: function; set_B_scales
.. _doxid-structdnnl_1_1ukernel_1_1attr__params_1a9e2c17ea304a349479bc36124b08e200:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_B_scales(const void* b_scales)

Sets tensor B scales arguments to a storage.

If :ref:`attr_params::set_B_scales <doxid-structdnnl_1_1ukernel_1_1attr__params_1a9e2c17ea304a349479bc36124b08e200>` used mask of 2, then at least N values of selected data type are expected.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- b_scales

		- Pointer to scales storage.

.. index:: pair: function; set_D_scales
.. _doxid-structdnnl_1_1ukernel_1_1attr__params_1aad9de5f26f633ab67e22953942b8ad10:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_D_scales(const void* d_scales)

Sets tensor D scales arguments to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- d_scales

		- Pointer to scales storage.

