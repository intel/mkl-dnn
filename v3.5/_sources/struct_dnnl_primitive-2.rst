.. index:: pair: struct; dnnl::primitive
.. _doxid-structdnnl_1_1primitive:

struct dnnl::primitive
======================

.. toctree::
	:hidden:

	enum_dnnl_primitive_kind.rst

Overview
~~~~~~~~

Base class for all computational primitives. :ref:`More...<details-structdnnl_1_1primitive>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// enums
	
		enum :ref:`kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>`;

		// construction
	
		:ref:`primitive<doxid-structdnnl_1_1primitive_1a4ba1e667dfc4d3abda8fa47e8bc8bbdb>`();
		:ref:`primitive<doxid-structdnnl_1_1primitive_1add43d05c97c3cd96a5d7cab07d6935f4>`(:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` c_pd);
	
		:ref:`primitive<doxid-structdnnl_1_1primitive_1abd151bdbbd2ccc7074e59cda584d38f4>`(
			:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` c_pd,
			const std::vector<uint8_t>& cache_blob
			);
	
		:ref:`primitive<doxid-structdnnl_1_1primitive_1ac356cacc1b41b95397d2c6a8c091e700>`(const :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`& pd);
		:ref:`primitive<doxid-structdnnl_1_1primitive_1a6aabbb4e883e6cdd796a67e5e9fb2228>`(const :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`& pd, const std::vector<uint8_t>& cache_blob);

		// methods
	
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` :ref:`get_primitive_desc<doxid-group__dnnl__api__primitives__common_1ga6d84440f113be4c6697092a0b968aa36>`() const;
		:ref:`kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` :ref:`get_kind<doxid-group__dnnl__api__primitives__common_1gab715f58b261078428fc2a99c9ee527fc>`() const;
		std::vector<uint8_t> :ref:`get_cache_blob<doxid-group__dnnl__api__primitives__common_1ga8bf59b36c745ee8eaec9d0dd22e266e9>`() const;
		void :ref:`execute<doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream, const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args) const;
		:ref:`handle<doxid-structdnnl_1_1primitive_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1a022001b5b9c8940a1326a02b61fc4860>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1aa13f3ecf4db240717074814412c7e70c>`();
		:ref:`handle<doxid-structdnnl_1_1primitive_1a9c408c09fce1278f5cb0d1fa9818fc86>`();
	};

	// direct descendants

	struct :ref:`augru_backward<doxid-structdnnl_1_1augru__backward>`;
	struct :ref:`augru_forward<doxid-structdnnl_1_1augru__forward>`;
	struct :ref:`batch_normalization_backward<doxid-structdnnl_1_1batch__normalization__backward>`;
	struct :ref:`batch_normalization_forward<doxid-structdnnl_1_1batch__normalization__forward>`;
	struct :ref:`binary<doxid-structdnnl_1_1binary>`;
	struct :ref:`concat<doxid-structdnnl_1_1concat>`;
	struct :ref:`convolution_backward_data<doxid-structdnnl_1_1convolution__backward__data>`;
	struct :ref:`convolution_backward_weights<doxid-structdnnl_1_1convolution__backward__weights>`;
	struct :ref:`convolution_forward<doxid-structdnnl_1_1convolution__forward>`;
	struct :ref:`deconvolution_backward_data<doxid-structdnnl_1_1deconvolution__backward__data>`;
	struct :ref:`deconvolution_backward_weights<doxid-structdnnl_1_1deconvolution__backward__weights>`;
	struct :ref:`deconvolution_forward<doxid-structdnnl_1_1deconvolution__forward>`;
	struct :ref:`eltwise_backward<doxid-structdnnl_1_1eltwise__backward>`;
	struct :ref:`eltwise_forward<doxid-structdnnl_1_1eltwise__forward>`;
	struct :ref:`group_normalization_backward<doxid-structdnnl_1_1group__normalization__backward>`;
	struct :ref:`group_normalization_forward<doxid-structdnnl_1_1group__normalization__forward>`;
	struct :ref:`gru_backward<doxid-structdnnl_1_1gru__backward>`;
	struct :ref:`gru_forward<doxid-structdnnl_1_1gru__forward>`;
	struct :ref:`inner_product_backward_data<doxid-structdnnl_1_1inner__product__backward__data>`;
	struct :ref:`inner_product_backward_weights<doxid-structdnnl_1_1inner__product__backward__weights>`;
	struct :ref:`inner_product_forward<doxid-structdnnl_1_1inner__product__forward>`;
	struct :ref:`layer_normalization_backward<doxid-structdnnl_1_1layer__normalization__backward>`;
	struct :ref:`layer_normalization_forward<doxid-structdnnl_1_1layer__normalization__forward>`;
	struct :ref:`lbr_augru_backward<doxid-structdnnl_1_1lbr__augru__backward>`;
	struct :ref:`lbr_augru_forward<doxid-structdnnl_1_1lbr__augru__forward>`;
	struct :ref:`lbr_gru_backward<doxid-structdnnl_1_1lbr__gru__backward>`;
	struct :ref:`lbr_gru_forward<doxid-structdnnl_1_1lbr__gru__forward>`;
	struct :ref:`lrn_backward<doxid-structdnnl_1_1lrn__backward>`;
	struct :ref:`lrn_forward<doxid-structdnnl_1_1lrn__forward>`;
	struct :ref:`lstm_backward<doxid-structdnnl_1_1lstm__backward>`;
	struct :ref:`lstm_forward<doxid-structdnnl_1_1lstm__forward>`;
	struct :ref:`matmul<doxid-structdnnl_1_1matmul>`;
	struct :ref:`pooling_backward<doxid-structdnnl_1_1pooling__backward>`;
	struct :ref:`pooling_forward<doxid-structdnnl_1_1pooling__forward>`;
	struct :ref:`prelu_backward<doxid-structdnnl_1_1prelu__backward>`;
	struct :ref:`prelu_forward<doxid-structdnnl_1_1prelu__forward>`;
	struct :ref:`reduction<doxid-structdnnl_1_1reduction>`;
	struct :ref:`reorder<doxid-structdnnl_1_1reorder>`;
	struct :ref:`resampling_backward<doxid-structdnnl_1_1resampling__backward>`;
	struct :ref:`resampling_forward<doxid-structdnnl_1_1resampling__forward>`;
	struct :ref:`shuffle_backward<doxid-structdnnl_1_1shuffle__backward>`;
	struct :ref:`shuffle_forward<doxid-structdnnl_1_1shuffle__forward>`;
	struct :ref:`softmax_backward<doxid-structdnnl_1_1softmax__backward>`;
	struct :ref:`softmax_forward<doxid-structdnnl_1_1softmax__forward>`;
	struct :ref:`sum<doxid-structdnnl_1_1sum>`;
	struct :ref:`vanilla_rnn_backward<doxid-structdnnl_1_1vanilla__rnn__backward>`;
	struct :ref:`vanilla_rnn_forward<doxid-structdnnl_1_1vanilla__rnn__forward>`;

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

.. _details-structdnnl_1_1primitive:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Base class for all computational primitives.

Construction
------------

.. index:: pair: function; primitive
.. _doxid-structdnnl_1_1primitive_1a4ba1e667dfc4d3abda8fa47e8bc8bbdb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive()

Default constructor. Constructs an empty object.

.. index:: pair: function; primitive
.. _doxid-structdnnl_1_1primitive_1add43d05c97c3cd96a5d7cab07d6935f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive(:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` c_pd)

Constructs a primitive from a C API primitive descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- c_pd

		- C API primitive descriptor.

.. index:: pair: function; primitive
.. _doxid-structdnnl_1_1primitive_1abd151bdbbd2ccc7074e59cda584d38f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive(
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` c_pd,
		const std::vector<uint8_t>& cache_blob
		)

Constructs a primitive from a C API primitive descriptor and a cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- c_pd

		- C API primitive descriptor.

	*
		- cache_blob

		- Cache blob.

.. index:: pair: function; primitive
.. _doxid-structdnnl_1_1primitive_1ac356cacc1b41b95397d2c6a8c091e700:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive(const :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`& pd)

Constructs a primitive from a primitive descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor.

.. index:: pair: function; primitive
.. _doxid-structdnnl_1_1primitive_1a6aabbb4e883e6cdd796a67e5e9fb2228:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive(const :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`& pd, const std::vector<uint8_t>& cache_blob)

Constructs a primitive from a primitive descriptor and a cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pd

		- Primitive descriptor.

	*
		- cache_blob

		- Cache blob.

Methods
-------

.. index:: pair: function; get_primitive_desc
.. _doxid-group__dnnl__api__primitives__common_1ga6d84440f113be4c6697092a0b968aa36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` get_primitive_desc() const

Returns the C API primitive descriptor of the underlying C API primitive.



.. rubric:: Returns:

The underlying C API primitive descriptor.

.. index:: pair: function; get_kind
.. _doxid-group__dnnl__api__primitives__common_1gab715f58b261078428fc2a99c9ee527fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` get_kind() const

Returns the kind of the primitive.



.. rubric:: Returns:

The primitive kind.

.. index:: pair: function; get_cache_blob
.. _doxid-group__dnnl__api__primitives__common_1ga8bf59b36c745ee8eaec9d0dd22e266e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_cache_blob() const

Returns a cache blob for the primitive.

.. note:: 

   The cache blob can be empty. It's the user's responsibility to check whether it's empty prior to passing it to the primitive constructor.



.. rubric:: Returns:

Vector containing the cache blob.

.. index:: pair: function; execute
.. _doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream, const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args) const

Executes computations specified by the primitive in a specified stream.

Arguments are passed via an arguments map containing <index, memory object> pairs. The index must be one of the ``DNNL_ARG_*`` values such as ``DNNL_ARG_SRC``, and the memory must have a memory descriptor matching the one returned by :ref:`primitive_desc::query_md <doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16>` (:ref:`query::exec_arg_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ad531896cf1d66c4832790f428623f164>`, index) unless using dynamic shapes (see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Stream object. The stream must belong to the same engine as the primitive.

	*
		- args

		- Arguments map.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1primitive_1a5c631f7e5e4c92a13edb8e3422d3a973:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs an empty handle object.

.. warning:: 

   Uninitialized object cannot be used in most library calls and is equivalent to a null pointer. Any attempt to use its methods, or passing it to the other library function, will cause an exception to be thrown.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1primitive_1a022001b5b9c8940a1326a02b61fc4860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Copy constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1primitive_1aa13f3ecf4db240717074814412c7e70c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Move constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1primitive_1a9c408c09fce1278f5cb0d1fa9818fc86:

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

