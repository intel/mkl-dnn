.. index:: pair: struct; dnnl::primitive_desc_base
.. _doxid-structdnnl_1_1primitive__desc__base:

struct dnnl::primitive_desc_base
================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Base class for all primitive descriptors. :ref:`More...<details-structdnnl_1_1primitive__desc__base>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_desc_base: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// methods
	
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`get_engine<doxid-structdnnl_1_1primitive__desc__base_1a32f7477c79e715a341bb9127df521fbc>`() const;
		const char* :ref:`impl_info_str<doxid-structdnnl_1_1primitive__desc__base_1ae680492d4e4b16a938cfc051e5c906be>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`query_s64<doxid-structdnnl_1_1primitive__desc__base_1acbedc4257eaa26f868356f9f594a856a>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what) const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_strides<doxid-structdnnl_1_1primitive__desc__base_1ab15fa4e08b75652e106ec7b7d6b13e8f>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_dilations<doxid-structdnnl_1_1primitive__desc__base_1a0f2ef952e57da908a074422822e6dbaf>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padding_l<doxid-structdnnl_1_1primitive__desc__base_1aa21112a5d3d0d38a47b9bb74024c5904>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padding_r<doxid-structdnnl_1_1primitive__desc__base_1a004d4371015dc3ba06d57752b4ec1bb6>`() const;
		float :ref:`get_epsilon<doxid-structdnnl_1_1primitive__desc__base_1ab8c5aaea19030d62c667d4b79eb0d680>`() const;
	
		template <typename T = unsigned>
		T :ref:`get_flags<doxid-structdnnl_1_1primitive__desc__base_1a8081d18d2289f5b58634a2b902bea264>`() const;
	
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_algorithm<doxid-structdnnl_1_1primitive__desc__base_1ae220e7adee4466f48e9a4781f0754461>`() const;
		float :ref:`get_alpha<doxid-structdnnl_1_1primitive__desc__base_1a73f6fcc68fd45166146a3c3d5eb49821>`() const;
		float :ref:`get_beta<doxid-structdnnl_1_1primitive__desc__base_1a29619c9f1308414bfdc8eb89fae7bd9d>`() const;
		int :ref:`get_axis<doxid-structdnnl_1_1primitive__desc__base_1af3b18a5a286f3b46e98b6fe8f45d60ac>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_local_size<doxid-structdnnl_1_1primitive__desc__base_1a00e6f182172077df46fc6f614e309592>`() const;
		float :ref:`get_k<doxid-structdnnl_1_1primitive__desc__base_1ad28f7f1c59d2fde215d71483daaaa632>`() const;
		float :ref:`get_p<doxid-structdnnl_1_1primitive__desc__base_1acfde33696840f499f6e790e415bd65ba>`() const;
		std::vector<float> :ref:`get_factors<doxid-structdnnl_1_1primitive__desc__base_1a18ce555604ee8680353bb95aeca08665>`() const;
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_cell_kind<doxid-structdnnl_1_1primitive__desc__base_1a940546a82d1381597863ecc7036b9c22>`() const;
		:ref:`dnnl::rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` :ref:`get_direction<doxid-structdnnl_1_1primitive__desc__base_1a7b9b95f2e16e3d2a9644df7ad5436f63>`() const;
		:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` :ref:`get_activation_kind<doxid-structdnnl_1_1primitive__desc__base_1acc309d5adeb8dbe8a88a18105082d566>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_kernel<doxid-structdnnl_1_1primitive__desc__base_1a3c9a39d1d0375518b18cf762fdd04b7e>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_group_size<doxid-structdnnl_1_1primitive__desc__base_1a03ad4fc755ae405a8a13d2c238266a15>`() const;
		:ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` :ref:`get_prop_kind<doxid-structdnnl_1_1primitive__desc__base_1a9adba18a967c92b205e939ba34de3542>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`query_md<doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16>`(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what, int idx = 0) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_desc<doxid-structdnnl_1_1primitive__desc__base_1af42e791f493e636c086e13c6d4c06b43>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a495ee7c8e1ec3eab35f6329fdcd352bb>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_desc<doxid-structdnnl_1_1primitive__desc__base_1acb13d08987cca8d8f05ec4858fa61fb4>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_desc<doxid-structdnnl_1_1primitive__desc__base_1a733e6ff4e78a758e69ae6232e8955871>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a60f17d04c493c42e4a50ad5feff5c8ca>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_desc<doxid-structdnnl_1_1primitive__desc__base_1a192f7b334efac9a5ac20344a76d4d552>`(int idx) const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`src_desc<doxid-structdnnl_1_1primitive__desc__base_1af48dcff294cadb2916fd784b8474d221>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`dst_desc<doxid-structdnnl_1_1primitive__desc__base_1addbec977643a7900f4156f7aab3fb4db>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`weights_desc<doxid-structdnnl_1_1primitive__desc__base_1a93f0904566b399874c47b3b1ad3d1495>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_src_desc<doxid-structdnnl_1_1primitive__desc__base_1ab4268c6bb70dd6c22de43141cc301b77>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_dst_desc<doxid-structdnnl_1_1primitive__desc__base_1a6b9fb7da987329256c04db6ecbb9dc36>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`diff_weights_desc<doxid-structdnnl_1_1primitive__desc__base_1af51378982968e2b4f7abb6a32acfb0af>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`workspace_desc<doxid-structdnnl_1_1primitive__desc__base_1ad26f416a149cb44cc5cfc130012c614e>`() const;
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`scratchpad_desc<doxid-structdnnl_1_1primitive__desc__base_1a6238358ec03afd57fb20dffa65b48d2f>`() const;
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`scratchpad_engine<doxid-structdnnl_1_1primitive__desc__base_1ad4079e0891373bd2dc841f3f94cf47ed>`() const;
		:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>` :ref:`get_primitive_attr<doxid-structdnnl_1_1primitive__desc__base_1a0ef600ea3666f0fd93c9c5d112aaf05c>`() const;
		:ref:`dnnl::primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` :ref:`get_kind<doxid-structdnnl_1_1primitive__desc__base_1a663f3511855bfc2625476019321c909d>`() const;
		std::vector<uint8_t> :ref:`get_cache_blob_id<doxid-structdnnl_1_1primitive__desc__base_1a435862df4d543eb8296424880212b22d>`() const;
	};

	// direct descendants

	struct :ref:`primitive_desc<doxid-structdnnl_1_1concat_1_1primitive__desc>`;
	struct :ref:`primitive_desc<doxid-structdnnl_1_1primitive__desc>`;
	struct :ref:`primitive_desc<doxid-structdnnl_1_1reorder_1_1primitive__desc>`;
	struct :ref:`primitive_desc<doxid-structdnnl_1_1sum_1_1primitive__desc>`;

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

.. _details-structdnnl_1_1primitive__desc__base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Base class for all primitive descriptors.

Methods
-------

.. index:: pair: function; get_engine
.. _doxid-structdnnl_1_1primitive__desc__base_1a32f7477c79e715a341bb9127df521fbc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` get_engine() const

Returns the engine of the primitive descriptor.



.. rubric:: Returns:

The engine of the primitive descriptor.

.. index:: pair: function; impl_info_str
.. _doxid-structdnnl_1_1primitive__desc__base_1ae680492d4e4b16a938cfc051e5c906be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const char* impl_info_str() const

Returns implementation name.



.. rubric:: Returns:

The implementation name.

.. index:: pair: function; query_s64
.. _doxid-structdnnl_1_1primitive__desc__base_1acbedc4257eaa26f868356f9f594a856a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` query_s64(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what) const

Returns a :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` value (same as int64_t).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- what

		- The value to query.



.. rubric:: Returns:

The result of the query.

.. index:: pair: function; get_strides
.. _doxid-structdnnl_1_1primitive__desc__base_1ab15fa4e08b75652e106ec7b7d6b13e8f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_strides() const

Returns strides.



.. rubric:: Returns:

Strides.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the primitive does not have a strides parameter.

.. index:: pair: function; get_dilations
.. _doxid-structdnnl_1_1primitive__desc__base_1a0f2ef952e57da908a074422822e6dbaf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_dilations() const

Returns dilations.



.. rubric:: Returns:

Dilations.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the primitive does not have a dilations parameter.

.. index:: pair: function; get_padding_l
.. _doxid-structdnnl_1_1primitive__desc__base_1aa21112a5d3d0d38a47b9bb74024c5904:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_padding_l() const

Returns a left padding.



.. rubric:: Returns:

A left padding.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the primitive does not have a left padding parameter.

.. index:: pair: function; get_padding_r
.. _doxid-structdnnl_1_1primitive__desc__base_1a004d4371015dc3ba06d57752b4ec1bb6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_padding_r() const

Returns a right padding.



.. rubric:: Returns:

A right padding.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the primitive does not have a right padding parameter.

.. index:: pair: function; get_epsilon
.. _doxid-structdnnl_1_1primitive__desc__base_1ab8c5aaea19030d62c667d4b79eb0d680:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_epsilon() const

Returns an epsilon.



.. rubric:: Returns:

An epsilon.

Zero if the primitive does not have an epsilon parameter.

.. index:: pair: function; get_flags
.. _doxid-structdnnl_1_1primitive__desc__base_1a8081d18d2289f5b58634a2b902bea264:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T = unsigned>
	T get_flags() const

Returns flags.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- T

		- Flags enumeration type.



.. rubric:: Returns:

Flags.

Zero if the primitive does not have a flags parameter.

.. index:: pair: function; get_algorithm
.. _doxid-structdnnl_1_1primitive__desc__base_1ae220e7adee4466f48e9a4781f0754461:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` get_algorithm() const

Returns an algorithm kind.



.. rubric:: Returns:

An algorithm kind.

:ref:`dnnl::algorithm::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an algorithm parameter.

.. index:: pair: function; get_alpha
.. _doxid-structdnnl_1_1primitive__desc__base_1a73f6fcc68fd45166146a3c3d5eb49821:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_alpha() const

Returns an alpha.



.. rubric:: Returns:

An alpha.

Zero if the primitive does not have an alpha parameter.

.. index:: pair: function; get_beta
.. _doxid-structdnnl_1_1primitive__desc__base_1a29619c9f1308414bfdc8eb89fae7bd9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_beta() const

Returns a beta.



.. rubric:: Returns:

A beta.

Zero if the primitive does not have a beta parameter.

.. index:: pair: function; get_axis
.. _doxid-structdnnl_1_1primitive__desc__base_1af3b18a5a286f3b46e98b6fe8f45d60ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int get_axis() const

Returns an axis.



.. rubric:: Returns:

An axis.

A negative number if the primitive does not have an axis parameter.

.. index:: pair: function; get_local_size
.. _doxid-structdnnl_1_1primitive__desc__base_1a00e6f182172077df46fc6f614e309592:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` get_local_size() const

Returns an LRN local size parameter.



.. rubric:: Returns:

An LRN local size parameter.

Zero if the primitive does not have an LRN local size parameter.

.. index:: pair: function; get_k
.. _doxid-structdnnl_1_1primitive__desc__base_1ad28f7f1c59d2fde215d71483daaaa632:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_k() const

Returns an LRN K parameter.



.. rubric:: Returns:

An LRN K parameter.

Zero if the primitive does not have an LRN K parameter.

.. index:: pair: function; get_p
.. _doxid-structdnnl_1_1primitive__desc__base_1acfde33696840f499f6e790e415bd65ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	float get_p() const

Returns a reduction P parameter.



.. rubric:: Returns:

A reduction P parameter.

Zero if the primitive does not have a reduction P parameter.

.. index:: pair: function; get_factors
.. _doxid-structdnnl_1_1primitive__desc__base_1a18ce555604ee8680353bb95aeca08665:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<float> get_factors() const

Returns a resampling factors parameters.



.. rubric:: Returns:

A vector of factors.

An empty vector if the primitive does not have a resampling factors parameter.

.. index:: pair: function; get_cell_kind
.. _doxid-structdnnl_1_1primitive__desc__base_1a940546a82d1381597863ecc7036b9c22:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` get_cell_kind() const

Returns an RNN cell kind parameter.



.. rubric:: Returns:

An RNN cell kind parameter.

:ref:`dnnl::algorithm::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN cell kind parameter.

.. index:: pair: function; get_direction
.. _doxid-structdnnl_1_1primitive__desc__base_1a7b9b95f2e16e3d2a9644df7ad5436f63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::rnn_direction<doxid-group__dnnl__api__rnn_1ga33315cf335d1cbe26fd6b70d956e23d5>` get_direction() const

Returns an RNN direction parameter.



.. rubric:: Returns:

An RNN direction parameter.

:ref:`dnnl::rnn_direction::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN direction parameter.

.. index:: pair: function; get_activation_kind
.. _doxid-structdnnl_1_1primitive__desc__base_1acc309d5adeb8dbe8a88a18105082d566:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` get_activation_kind() const

Returns an RNN activation kind parameter.



.. rubric:: Returns:

An RNN activation kind parameter.

:ref:`dnnl::algorithm::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have an RNN activation kind parameter.

.. index:: pair: function; get_kernel
.. _doxid-structdnnl_1_1primitive__desc__base_1a3c9a39d1d0375518b18cf762fdd04b7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_kernel() const

Returns a pooling kernel parameter.



.. rubric:: Returns:

A pooling kernel parameter.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the primitive does not have a pooling kernel parameter.

.. index:: pair: function; get_group_size
.. _doxid-structdnnl_1_1primitive__desc__base_1a03ad4fc755ae405a8a13d2c238266a15:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` get_group_size() const

Returns a shuffle group size parameter.



.. rubric:: Returns:

A shuffle group size parameter.

Zero if the primitive does not have a shuffle group size parameter.

.. index:: pair: function; get_prop_kind
.. _doxid-structdnnl_1_1primitive__desc__base_1a9adba18a967c92b205e939ba34de3542:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::prop_kind<doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>` get_prop_kind() const

Returns a propagation kind.



.. rubric:: Returns:

A propagation kind.

:ref:`dnnl::prop_kind::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>` if the primitive does not have a propagation parameter.

.. index:: pair: function; query_md
.. _doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` query_md(:ref:`query<doxid-group__dnnl__api__primitives__common_1ga94efdd650364f4d9776cfb9b711cbdc1>` what, int idx = 0) const

Returns a memory descriptor.

.. note:: 

   There are also convenience methods :ref:`dnnl::primitive_desc_base::src_desc() <doxid-structdnnl_1_1primitive__desc__base_1af42e791f493e636c086e13c6d4c06b43>`, :ref:`dnnl::primitive_desc_base::dst_desc() <doxid-structdnnl_1_1primitive__desc__base_1a495ee7c8e1ec3eab35f6329fdcd352bb>`, and others.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- what

		- The kind of parameter to query; can be :ref:`dnnl::query::src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>`, :ref:`dnnl::query::dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>`, etc.

	*
		- idx

		- Index of the parameter. For example, convolution bias can be queried with what = :ref:`dnnl::query::weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>` and idx = 1.



.. rubric:: Returns:

The requested memory descriptor.

A zero memory descriptor if the primitive does not have a parameter of the specified kind or index.

.. index:: pair: function; src_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1af42e791f493e636c086e13c6d4c06b43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_desc(int idx) const

Returns a source memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Source index.



.. rubric:: Returns:

Source memory descriptor.

A zero memory descriptor if the primitive does not have a source parameter with index ``idx``.

.. index:: pair: function; dst_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a495ee7c8e1ec3eab35f6329fdcd352bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_desc(int idx) const

Returns a destination memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Destination index.



.. rubric:: Returns:

Destination memory descriptor.

A zero memory descriptor if the primitive does not have a destination parameter with index ``idx``.

.. index:: pair: function; weights_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1acb13d08987cca8d8f05ec4858fa61fb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_desc(int idx) const

Returns a weights memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Weights index.



.. rubric:: Returns:

Weights memory descriptor.

A zero memory descriptor if the primitive does not have a weights parameter with index ``idx``.

.. index:: pair: function; diff_src_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a733e6ff4e78a758e69ae6232e8955871:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_src_desc(int idx) const

Returns a diff source memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Diff source index.



.. rubric:: Returns:

Diff source memory descriptor.

A zero memory descriptor if the primitive does not have a diff source parameter with index ``idx``.

.. index:: pair: function; diff_dst_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a60f17d04c493c42e4a50ad5feff5c8ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_dst_desc(int idx) const

Returns a diff destination memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Diff destination index.



.. rubric:: Returns:

Diff destination memory descriptor.

A zero memory descriptor if the primitive does not have a diff destination parameter with index ``idx``.

.. index:: pair: function; diff_weights_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a192f7b334efac9a5ac20344a76d4d552:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_weights_desc(int idx) const

Returns a diff weights memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- idx

		- Diff weights index.



.. rubric:: Returns:

Diff weights memory descriptor.

A zero memory descriptor if the primitive does not have a diff weights parameter with index ``idx``.

.. index:: pair: function; src_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1af48dcff294cadb2916fd784b8474d221:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` src_desc() const

Returns a source memory descriptor.



.. rubric:: Returns:

Source memory descriptor.

A zero memory descriptor if the primitive does not have a source parameter.

.. index:: pair: function; dst_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1addbec977643a7900f4156f7aab3fb4db:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` dst_desc() const

Returns a destination memory descriptor.



.. rubric:: Returns:

Destination memory descriptor.

A zero memory descriptor if the primitive does not have a destination parameter.

.. index:: pair: function; weights_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a93f0904566b399874c47b3b1ad3d1495:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` weights_desc() const

Returns a weights memory descriptor.



.. rubric:: Returns:

Weights memory descriptor.

A zero memory descriptor if the primitive does not have a weights parameter.

.. index:: pair: function; diff_src_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1ab4268c6bb70dd6c22de43141cc301b77:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_src_desc() const

Returns a diff source memory descriptor.



.. rubric:: Returns:

Diff source memory descriptor.

A zero memory descriptor if the primitive does not have a diff source memory with.

.. index:: pair: function; diff_dst_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a6b9fb7da987329256c04db6ecbb9dc36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_dst_desc() const

Returns a diff destination memory descriptor.



.. rubric:: Returns:

Diff destination memory descriptor.

A zero memory descriptor if the primitive does not have a diff destination parameter.

.. index:: pair: function; diff_weights_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1af51378982968e2b4f7abb6a32acfb0af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` diff_weights_desc() const

Returns a diff weights memory descriptor.



.. rubric:: Returns:

Diff weights memory descriptor.

A zero memory descriptor if the primitive does not have a diff weights parameter.

.. index:: pair: function; workspace_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1ad26f416a149cb44cc5cfc130012c614e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` workspace_desc() const

Returns the workspace memory descriptor.



.. rubric:: Returns:

Workspace memory descriptor.

A zero memory descriptor if the primitive does not require workspace parameter.

.. index:: pair: function; scratchpad_desc
.. _doxid-structdnnl_1_1primitive__desc__base_1a6238358ec03afd57fb20dffa65b48d2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>` scratchpad_desc() const

Returns the scratchpad memory descriptor.



.. rubric:: Returns:

scratchpad memory descriptor.

A zero memory descriptor if the primitive does not require scratchpad parameter.



.. rubric:: See also:

:ref:`Primitive Attributes: Scratchpad <doxid-dev_guide_attributes_scratchpad>`

.. index:: pair: function; scratchpad_engine
.. _doxid-structdnnl_1_1primitive__desc__base_1ad4079e0891373bd2dc841f3f94cf47ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` scratchpad_engine() const

Returns the engine on which the scratchpad memory is located.



.. rubric:: Returns:

The engine on which the scratchpad memory is located.

.. index:: pair: function; get_primitive_attr
.. _doxid-structdnnl_1_1primitive__desc__base_1a0ef600ea3666f0fd93c9c5d112aaf05c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr>` get_primitive_attr() const

Returns the primitive attributes.



.. rubric:: Returns:

The primitive attributes.

.. index:: pair: function; get_kind
.. _doxid-structdnnl_1_1primitive__desc__base_1a663f3511855bfc2625476019321c909d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` get_kind() const

Returns the kind of the primitive descriptor.



.. rubric:: Returns:

The kind of the primitive descriptor.

.. index:: pair: function; get_cache_blob_id
.. _doxid-structdnnl_1_1primitive__desc__base_1a435862df4d543eb8296424880212b22d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_cache_blob_id() const

Returns the cache blob ID of the primitive descriptor.



.. rubric:: Returns:

The cache blob ID of the primitive descriptor.

