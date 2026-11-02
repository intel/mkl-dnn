.. index:: pair: struct; dnnl::primitive_attr
.. _doxid-structdnnl_1_1primitive__attr:

struct dnnl::primitive_attr
===========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Primitive attributes. :ref:`More...<details-structdnnl_1_1primitive__attr>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct primitive_attr: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr_1acfbfd85b7ca82bf97e2b07c2427427de>`();
		:ref:`primitive_attr<doxid-structdnnl_1_1primitive__attr_1aafb54e73f3abe59555f1cfe62407280e>`(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr);

		// methods
	
		:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` :ref:`get_fpmath_mode<doxid-structdnnl_1_1primitive__attr_1af335f8e1e74b69c5b2d4da0ecf8a23d5>`() const;
		void :ref:`get_fpmath_mode<doxid-structdnnl_1_1primitive__attr_1a4c5566628d4c893a321a8f053a5cc5fd>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`& mode, bool& apply_to_int) const;
		void :ref:`set_fpmath_mode<doxid-structdnnl_1_1primitive__attr_1ab00639157a283596834ee5b0e8478a2d>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode, bool apply_to_int = false);
		:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` :ref:`get_accumulation_mode<doxid-structdnnl_1_1primitive__attr_1a054e1d0097e9e85411cfb539bd45e247>`() const;
		void :ref:`set_accumulation_mode<doxid-structdnnl_1_1primitive__attr_1a8348fcd2259553c3537194430b7de4f4>`(:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` mode);
		bool :ref:`get_deterministic<doxid-structdnnl_1_1primitive__attr_1a29e003998b46762e5fa0532844ca12c4>`() const;
		void :ref:`set_deterministic<doxid-structdnnl_1_1primitive__attr_1ad4c1ea7156ce31cd231a00d4e3399add>`(bool value);
		:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` :ref:`get_scratchpad_mode<doxid-structdnnl_1_1primitive__attr_1af4131b946ec3af3bc2974b603d30029b>`() const;
		void :ref:`set_scratchpad_mode<doxid-structdnnl_1_1primitive__attr_1a91a597649afa13b7d2416b708d0620d2>`(:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode);
		void :ref:`set_scales_mask<doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(int arg, int mask);
	
		void :ref:`set_scales<doxid-structdnnl_1_1primitive__attr_1a29e8f33119d42bf7d259eafc6e6548d6>`(
			int arg,
			int mask,
			const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& groups,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::f32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`
			);
	
		void :ref:`set_zero_points_mask<doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`(int arg, int mask);
	
		void :ref:`set_zero_points<doxid-structdnnl_1_1primitive__attr_1aa7a57b0ba198c418981d41c5289fed8e>`(
			int arg,
			int mask,
			const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& groups,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::s32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`
			);
	
		const :ref:`post_ops<doxid-structdnnl_1_1post__ops>` :ref:`get_post_ops<doxid-structdnnl_1_1primitive__attr_1a05664ef63c94acbcc59e921c4a4da6b8>`() const;
		void :ref:`set_post_ops<doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(const :ref:`post_ops<doxid-structdnnl_1_1post__ops>` ops);
		void :ref:`set_rnn_data_qparams<doxid-structdnnl_1_1primitive__attr_1a39ce5aa8b06ed331d8e2158108cc8324>`(float scale, float shift);
		void :ref:`get_rnn_data_qparams<doxid-structdnnl_1_1primitive__attr_1a47d567defa762761daa2af604798d799>`(float& scale, float& shift);
		void :ref:`set_rnn_weights_qparams<doxid-structdnnl_1_1primitive__attr_1a61bd70f97baa628fd49b2c8b334b913e>`(int mask, const std::vector<float>& scales);
		void :ref:`get_rnn_weights_qparams<doxid-structdnnl_1_1primitive__attr_1a3bbe9ac516e3aabe7dfea214210a3335>`(int& mask, std::vector<float>& scales);
	
		void :ref:`set_rnn_weights_projection_qparams<doxid-structdnnl_1_1primitive__attr_1a6e5a8c12f28421c249633bf2092fbe3f>`(
			int mask,
			const std::vector<float>& scales
			);
	
		void :ref:`get_rnn_weights_projection_qparams<doxid-structdnnl_1_1primitive__attr_1a53c41b29c5cd74d9485b5e753ba57f1d>`(int& mask, std::vector<float>& scales);
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

.. _details-structdnnl_1_1primitive__attr:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Primitive attributes.



.. rubric:: See also:

:ref:`Primitive Attributes <doxid-dev_guide_attributes>`

Construction
------------

.. index:: pair: function; primitive_attr
.. _doxid-structdnnl_1_1primitive__attr_1acfbfd85b7ca82bf97e2b07c2427427de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_attr()

Constructs default (empty) primitive attributes.

.. index:: pair: function; primitive_attr
.. _doxid-structdnnl_1_1primitive__attr_1aafb54e73f3abe59555f1cfe62407280e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	primitive_attr(:ref:`dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` attr)

Creates primitive attributes from a C API :ref:`dnnl_primitive_attr_t <doxid-group__dnnl__api__attributes_1ga06d701a25b82d4c8a93aaabb93e03dc3>` handle.

The resulting handle is not weak and the C handle will be destroyed during the destruction of the C++ object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr

		- The C API primitive attributes.

Methods
-------

.. index:: pair: function; get_fpmath_mode
.. _doxid-structdnnl_1_1primitive__attr_1af335f8e1e74b69c5b2d4da0ecf8a23d5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` get_fpmath_mode() const

Returns the fpmath mode.

.. index:: pair: function; get_fpmath_mode
.. _doxid-structdnnl_1_1primitive__attr_1a4c5566628d4c893a321a8f053a5cc5fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_fpmath_mode(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>`& mode, bool& apply_to_int) const

Returns the fpmath mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- Specified fpmath mode.

	*
		- apply_to_int

		- Use floating-point arithmetic for integer primitives.

.. index:: pair: function; set_fpmath_mode
.. _doxid-structdnnl_1_1primitive__attr_1ab00639157a283596834ee5b0e8478a2d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_fpmath_mode(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode, bool apply_to_int = false)

Sets fpmath mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- Specified fpmath mode.

	*
		- apply_to_int

		- Boolean. Use of floating-point arithmetic for integer primitives.

.. index:: pair: function; get_accumulation_mode
.. _doxid-structdnnl_1_1primitive__attr_1a054e1d0097e9e85411cfb539bd45e247:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` get_accumulation_mode() const

Returns the accumulation mode.

.. index:: pair: function; set_accumulation_mode
.. _doxid-structdnnl_1_1primitive__attr_1a8348fcd2259553c3537194430b7de4f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_accumulation_mode(:ref:`accumulation_mode<doxid-group__dnnl__api__accumulation__mode_1gad6b8b3ca2e61b8a9703227f4d58ac215>` mode)

Sets accumulation mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- Specified accumulation mode.

.. index:: pair: function; get_deterministic
.. _doxid-structdnnl_1_1primitive__attr_1a29e003998b46762e5fa0532844ca12c4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_deterministic() const

Returns the deterministic attribute value.

.. index:: pair: function; set_deterministic
.. _doxid-structdnnl_1_1primitive__attr_1ad4c1ea7156ce31cd231a00d4e3399add:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_deterministic(bool value)

Sets deterministic attribute value.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- value

		- Specified deterministic mode.

.. index:: pair: function; get_scratchpad_mode
.. _doxid-structdnnl_1_1primitive__attr_1af4131b946ec3af3bc2974b603d30029b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` get_scratchpad_mode() const

Returns the scratchpad mode.

.. index:: pair: function; set_scratchpad_mode
.. _doxid-structdnnl_1_1primitive__attr_1a91a597649afa13b7d2416b708d0620d2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_scratchpad_mode(:ref:`scratchpad_mode<doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>` mode)

Sets scratchpad mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- Specified scratchpad mode.

.. index:: pair: function; set_scales_mask
.. _doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_scales_mask(int arg, int mask)

Sets scaling factors for primitive operations for a given memory argument.

The scaling factors must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- arg

		- Parameter argument index as passed to the :ref:`primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call.

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor is used for each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_scales_mask <doxid-group__dnnl__api__attributes_1gad7eac877f75cfa282be094b1e48cb71d>`

.. index:: pair: function; set_scales
.. _doxid-structdnnl_1_1primitive__attr_1a29e8f33119d42bf7d259eafc6e6548d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_scales(
		int arg,
		int mask,
		const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& groups,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::f32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`
		)

Sets scaling factors for primitive operations for a given memory argument.

The scaling factors must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- arg

		- Parameter argument index as passed to the :ref:`primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call.

	*
		- mask

		- Scales correspondence mask that defines the correspondence between the tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scale is used for each index along that dimension. Set the mask to 0 to use a common scale for the whole output tensor.

	*
		- groups

		- Scaling factors correspondence groups that define the correspondence between the tensor dimensions and the scales array. The set i-th dimension indicates a number of groups of scaling factors used for that logical dimension in a memory indicated by ``arg``.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_scales <doxid-group__dnnl__api__attributes_1ga6c56ea6104c9275574370a40e2d5b273>`

.. index:: pair: function; set_zero_points_mask
.. _doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_zero_points_mask(int arg, int mask)

Sets zero points for primitive operations for a given memory argument.

The zero points must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_ZERO_POINTS <doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- arg

		- Parameter argument index as passed to the :ref:`primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call.

	*
		- mask

		- Zero point correspondence mask that defines the correspondence between the tensor dimensions and the ``zero_points`` vector. The set i-th bit indicates that a dedicated zero point is used for each index along that dimension. Set the mask to 0 to use a common zero point for the whole output tensor.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_zero_points_mask <doxid-group__dnnl__api__attributes_1ga24e429b5410f5657bc5bdda0a6c5d0a7>`

.. index:: pair: function; set_zero_points
.. _doxid-structdnnl_1_1primitive__attr_1aa7a57b0ba198c418981d41c5289fed8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_zero_points(
		int arg,
		int mask,
		const :ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& groups,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::s32<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`
		)

Sets zero points for primitive operations for a given memory argument.

The zero points must be passed at execution time as an argument with index :ref:`DNNL_ARG_ATTR_ZERO_POINTS <doxid-group__dnnl__api__primitives__common_1gaf8d879adfe2baa2f9f2a5143a0f274b6>` \| arg.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- arg

		- Parameter argument index as passed to the :ref:`primitive::execute() <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>` call.

	*
		- mask

		- Zero point correspondence mask that defines the correspondence between the tensor dimensions and the ``zero_points`` vector. The set i-th bit indicates that a dedicated zero point is used for each index along that dimension. Set the mask to 0 to use a common zero point for the whole output tensor.

	*
		- groups

		- Zero point factors correspondence groups that define the correspondence between the tensor dimensions and the zero_points array. The set i-th dimension indicates a number of groups of zero point factors used for that logical dimension in a memory indicated by ``arg``.



.. rubric:: See also:

:ref:`dnnl_primitive_attr_set_zero_points <doxid-group__dnnl__api__attributes_1ga94f3e4f640fb0ca210c2413bb5cf2255>`

.. index:: pair: function; get_post_ops
.. _doxid-structdnnl_1_1primitive__attr_1a05664ef63c94acbcc59e921c4a4da6b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const :ref:`post_ops<doxid-structdnnl_1_1post__ops>` get_post_ops() const

Returns post-ops previously set via :ref:`set_post_ops() <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`.



.. rubric:: Returns:

Post-ops.

.. index:: pair: function; set_post_ops
.. _doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_post_ops(const :ref:`post_ops<doxid-structdnnl_1_1post__ops>` ops)

Sets post-ops.

.. note:: 

   There is no way to check whether the post-ops would be supported by the target primitive. Any error will be reported by the respective primitive descriptor constructor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- ops

		- Post-ops object to copy post-ops from.

.. index:: pair: function; set_rnn_data_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a39ce5aa8b06ed331d8e2158108cc8324:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_rnn_data_qparams(float scale, float shift)

Sets quantization scale and shift parameters for RNN data tensors.

For performance reasons, the low-precision configuration of the RNN primitives expect input activations to have the unsigned 8-bit integer data type. The scale and shift parameters are used to quantize floating-point data to unsigned integer and must be passed to the RNN primitive using attributes.

The quantization formula is ``scale * data + shift``.

Example usage:

.. ref-code-block:: cpp

	// RNN parameters
	int l = 2, t = 2, mb = 32, sic = 32, slc = 32, dic = 32, dlc = 32;
	// Activations quantization parameters
	float scale = 63.f, shift = 64.f;
	
	primitive_attr attr;
	
	// Set scale and shift for int8 quantization of activation
	attr.set_rnn_data_qparams(scale, shift);
	
	// Create an RNN primitive descriptor.
	vanilla_rnn_forward::primitive_desc rnn_d(
	        engine, /* arguments */, attr);

.. note:: 

   Quantization scale and shift are common for src_layer, src_iter, dst_iter, and dst_layer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- scale

		- The value to scale the data by.

	*
		- shift

		- The value to shift the data by.

.. index:: pair: function; get_rnn_data_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a47d567defa762761daa2af604798d799:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_rnn_data_qparams(float& scale, float& shift)

Returns the quantization scale and shift parameters for RNN data tensors.

.. note:: 

   Quantization scale and shift are common for src_layer, src_iter, dst_iter, and dst_layer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- scale

		- The value to scale the data by.

	*
		- shift

		- The value to shift the data by.

.. index:: pair: function; set_rnn_weights_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a61bd70f97baa628fd49b2c8b334b913e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_rnn_weights_qparams(int mask, const std::vector<float>& scales)

Sets quantization scaling factors for RNN weights tensors.

The low-precision configuration of the RNN primitives expect input weights to use the signed 8-bit integer data type. The scaling factors are used to quantize floating-point data to signed integer and must be passed to RNN primitives using attributes.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.
   
   

.. note:: 

   Quantization scales are common for weights_layer and weights_iteration



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- Constant vector of output scaling factors. The following equality must hold: :math:`scales.size() = \prod\limits_{d \in mask} weights.dims[d].` Violations can only be detected when the attributes are used to create a primitive descriptor.

.. index:: pair: function; get_rnn_weights_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a3bbe9ac516e3aabe7dfea214210a3335:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_rnn_weights_qparams(int& mask, std::vector<float>& scales)

Returns the quantization scaling factors for RNN projection weights tensors.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- Constant vector of output scaling factors. The following equality must hold: :math:`scales.size() = \prod\limits_{d \in mask} weights.dims[d].` Violations can only be detected when the attributes are used to create a primitive descriptor.

.. index:: pair: function; set_rnn_weights_projection_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a6e5a8c12f28421c249633bf2092fbe3f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_rnn_weights_projection_qparams(
		int mask,
		const std::vector<float>& scales
		)

Sets quantization scaling factors for RNN projection weights tensors.

passed to RNN primitives using attributes.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.
   
   

.. note:: 

   Quantization scales are common for weights_layer and weights_iteration



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- Constant vector of output scaling factors. The following equality must hold: :math:`scales.size() = \prod\limits_{d \in mask} weights.dims[d].` Violations can only be detected when the attributes are used to create a primitive descriptor.

.. index:: pair: function; get_rnn_weights_projection_qparams
.. _doxid-structdnnl_1_1primitive__attr_1a53c41b29c5cd74d9485b5e753ba57f1d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_rnn_weights_projection_qparams(int& mask, std::vector<float>& scales)

Returns the quantization scaling factors for RNN projection weights tensors.

.. note:: 

   The dimension order is always native and does not depend on the actual layout used. For example, five-dimensional weights always have (l, d, i, g, o) logical dimension ordering.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mask

		- Scaling factors correspondence mask that defines the correspondence between the output tensor dimensions and the ``scales`` vector. The set i-th bit indicates that a dedicated scaling factor should be used each index along that dimension. Set the mask to 0 to use a common scaling factor for the whole output tensor.

	*
		- scales

		- Constant vector of output scaling factors. The following equality must hold: :math:`scales.size() = \prod\limits_{d \in mask} weights.dims[d].` Violations can only be detected when the attributes are used to create a primitive descriptor.

