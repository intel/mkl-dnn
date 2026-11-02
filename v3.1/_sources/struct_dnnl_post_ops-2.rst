.. index:: pair: struct; dnnl::post_ops
.. _doxid-structdnnl_1_1post__ops:

struct dnnl::post_ops
=====================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Post-ops. :ref:`More...<details-structdnnl_1_1post__ops>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct post_ops: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`post_ops<doxid-structdnnl_1_1post__ops_1a8e1d47722db8f53b3689468788ec2c01>`();
		:ref:`post_ops<doxid-structdnnl_1_1post__ops_1a22c94a4cefcb6bab85e60da4e6503341>`(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops);

		// methods
	
		int :ref:`len<doxid-structdnnl_1_1post__ops_1a84653b68d83c2d84d3ac432a8dc1f5fd>`() const;
		:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` :ref:`kind<doxid-structdnnl_1_1post__ops_1a454acad1a18f2763f07b42912778c0f8>`(int index) const;
	
		void :ref:`append_sum<doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`(
			float scale = 1.f,
			int32_t zero_point = 0,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::undef<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b>`
			);
	
		void :ref:`get_params_sum<doxid-structdnnl_1_1post__ops_1aaeb404084e7f9c65e8e266acca2ea6ac>`(int index, float& scale) const;
		void :ref:`get_params_sum<doxid-structdnnl_1_1post__ops_1a19a49b0e1c5b2ce4f38d7774a50b0824>`(int index, float& scale, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& data_type) const;
	
		void :ref:`get_params_sum<doxid-structdnnl_1_1post__ops_1a11c17d7a4ebbd50251eb20f535eb08e8>`(
			int index,
			float& scale,
			int32_t& zero_point,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& data_type
			) const;
	
		void :ref:`append_eltwise<doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm, float alpha, float beta);
	
		void :ref:`get_params_eltwise<doxid-structdnnl_1_1post__ops_1ade89dd346e9c8682e617b85bc8623185>`(
			int index,
			:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`& aalgorithm,
			float& alpha,
			float& beta
			) const;
	
		void :ref:`append_dw<doxid-structdnnl_1_1post__ops_1a55aad3b45a25087e0045a005384bde3a>`(
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` weights_data_type,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` bias_data_type,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` dst_data_type,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` kernel_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` stride_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` padding_l_size
			);
	
		void :ref:`get_params_dw<doxid-structdnnl_1_1post__ops_1a2762641ef41b1c1af25e882810ac9224>`(
			int index,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& weights_data_type,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& bias_data_type,
			:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& dst_data_type,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& kernel_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& stride_size,
			:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& padding_l_size
			) const;
	
		void :ref:`append_binary<doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm, const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src1_desc);
	
		void :ref:`get_params_binary<doxid-structdnnl_1_1post__ops_1a0a367859cac33d597743de491d26dcb9>`(
			int index,
			:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`& aalgorithm,
			:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src1_desc
			) const;
	
		void :ref:`append_prelu<doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70>`(int mask);
		void :ref:`get_params_prelu<doxid-structdnnl_1_1post__ops_1a0cacfa74daa86b05bedb351d905e1516>`(int index, int& mask) const;
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

.. _details-structdnnl_1_1post__ops:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Post-ops.

Post-ops are computations executed after the main primitive computations and are attached to the primitive via primitive attributes.



.. rubric:: See also:

:ref:`Primitive Attributes: Post-ops <doxid-dev_guide_attributes_post_ops>`

Construction
------------

.. index:: pair: function; post_ops
.. _doxid-structdnnl_1_1post__ops_1a8e1d47722db8f53b3689468788ec2c01:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	post_ops()

Constructs an empty sequence of post-ops.

.. index:: pair: function; post_ops
.. _doxid-structdnnl_1_1post__ops_1a22c94a4cefcb6bab85e60da4e6503341:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	post_ops(:ref:`dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` post_ops)

Creates post-ops primitive attribute from a C API :ref:`dnnl_post_ops_t <doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` handle.

The resulting handle is not weak and the C handle will be destroyed during the destruction of the C++ object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- :ref:`post_ops <doxid-structdnnl_1_1post__ops>`

		- The C API post-ops primitive attribute.

Methods
-------

.. index:: pair: function; len
.. _doxid-structdnnl_1_1post__ops_1a84653b68d83c2d84d3ac432a8dc1f5fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int len() const

Returns the number of post-ops entries.

.. index:: pair: function; kind
.. _doxid-structdnnl_1_1post__ops_1a454acad1a18f2763f07b42912778c0f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`primitive::kind<doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` kind(int index) const

Returns the primitive kind of post-op at entry with a certain index.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the post-op to return the kind for.



.. rubric:: Returns:

Primitive kind of the post-op at the specified index.

.. index:: pair: function; append_sum
.. _doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void append_sum(
		float scale = 1.f,
		int32_t zero_point = 0,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`memory::data_type::undef<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Appends an accumulation (sum) post-op.

Prior to accumulating the result, the previous value will be will be reduced by zero point ``zero_point`` and multiplied by a scaling factor ``scale``.

The kind of this post-op is :ref:`dnnl::primitive::kind::sum <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a1d623b89683f9ce4e074de1676d12416>`.

This feature may improve performance for cases like dequantize the asymmetrically quantized sum's src1 tensor to f32 domain before performing the sum operation by subtracting ``zero_point`` before the scaling.

In the simplest case when the accumulation is the only post-op, the computations will be ``dst[:] := scale * (dst[:] - zero_point) + op(...)`` instead of ``dst[:] := op(...)``.

If ``data_type`` is specified, the original dst tensor will be reinterpreted as a tensor with the provided data type. Because it is a reinterpretation, data_type and dst data type should have the same size. As a result, computations will be ``dst[:] <- scale * (as_data_type(dst[:]) - zero_point) + op(...)`` instead of ``dst[:] <- op(...)``.

.. note:: 

   This post-op executes in-place and does not change the destination layout.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- scale

		- Scaling factor.

	*
		- zero_point

		- Zero point.

	*
		- data_type

		- Data type.

.. index:: pair: function; get_params_sum
.. _doxid-structdnnl_1_1post__ops_1aaeb404084e7f9c65e8e266acca2ea6ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_sum(int index, float& scale) const

Returns the parameters of an accumulation (sum) post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the sum post-op.

	*
		- scale

		- Scaling factor of the sum post-op.

.. index:: pair: function; get_params_sum
.. _doxid-structdnnl_1_1post__ops_1a19a49b0e1c5b2ce4f38d7774a50b0824:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_sum(int index, float& scale, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& data_type) const

Returns the parameters of an accumulation (sum) post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the sum post-op.

	*
		- scale

		- Scaling factor of the sum post-op.

	*
		- data_type

		- Data type of the sum post-op.

.. index:: pair: function; get_params_sum
.. _doxid-structdnnl_1_1post__ops_1a11c17d7a4ebbd50251eb20f535eb08e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_sum(
		int index,
		float& scale,
		int32_t& zero_point,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& data_type
		) const

Returns the parameters of an accumulation (sum) post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the sum post-op.

	*
		- scale

		- Scaling factor of the sum post-op.

	*
		- zero_point

		- Single scalar int32_t value of zeropoint.

	*
		- data_type

		- Data type of the sum post-op.

.. index:: pair: function; append_eltwise
.. _doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void append_eltwise(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm, float alpha, float beta)

Appends an elementwise post-op.

The kind of this post-op is :ref:`dnnl::primitive::kind::eltwise <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a98b908c7d0339bb6a4832db44fc2c8da>`.

In the simplest case when the elementwise is the only post-op, the computations would be ``dst[:] := eltwise_op (op(...))`` instead of ``dst[:] <- op(...)``, where eltwise_op is configured with the given parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aalgorithm

		- Elementwise algorithm.

	*
		- alpha

		- Alpha parameter for the elementwise algorithm.

	*
		- beta

		- Beta parameter for the elementwise algorithm.

.. index:: pair: function; get_params_eltwise
.. _doxid-structdnnl_1_1post__ops_1ade89dd346e9c8682e617b85bc8623185:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_eltwise(
		int index,
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`& aalgorithm,
		float& alpha,
		float& beta
		) const

Returns parameters of an elementwise post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the post-op.

	*
		- aalgorithm

		- Output elementwise algorithm kind.

	*
		- alpha

		- Output alpha parameter for the elementwise algorithm.

	*
		- beta

		- Output beta parameter for the elementwise algorithm.

.. index:: pair: function; append_dw
.. _doxid-structdnnl_1_1post__ops_1a55aad3b45a25087e0045a005384bde3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void append_dw(
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` weights_data_type,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` bias_data_type,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` dst_data_type,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` kernel_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` stride_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` padding_l_size
		)

Appends a depthwise post-op convolution.

This post-op can only be fused with a 2D 1x1 convolution (convolution with weights spatial dimension equal to 1 i.e., kh=kw=1).

The kind of this post-op is :ref:`dnnl_convolution <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa402cfeaa257524d301bb73e770bc87f6>`.

The number of outputs for primitive remain same as before. The output spatial size can be derived as below:

output_height = ceil(output_height_1x1_convolution, stride) output_width = ceil(output_width_1x1_convolution, stride)

See :ref:`dev_guide_attributes_post_ops_depthwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise>` and :ref:`dev_guide_attributes_post_ops_depthwise_fusion <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise_fusion>` for more info.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- weights_data_type

		- Weights data type of depthwise post-op

	*
		- bias_data_type

		- Bias data type of depthwise post-op

	*
		- dst_data_type

		- Output data type of depthwise post-op

	*
		- kernel_size

		- Size of kernel of depthwise post-op

	*
		- stride_size

		- Size of stride of depthwise post-op

	*
		- padding_l_size

		- Size of left and top paddings of depthwise post-op

.. index:: pair: function; get_params_dw
.. _doxid-structdnnl_1_1post__ops_1a2762641ef41b1c1af25e882810ac9224:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_dw(
		int index,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& weights_data_type,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& bias_data_type,
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`& dst_data_type,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& kernel_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& stride_size,
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`& padding_l_size
		) const

Returns the parameters of an depthwise post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the elementwise post-op.

	*
		- weights_data_type

		- Weights data type of depthwise post-op

	*
		- bias_data_type

		- Bias data type of depthwise post-op

	*
		- dst_data_type

		- Output data type of depthwise post-op

	*
		- kernel_size

		- Size of kernel of depthwise post-op

	*
		- stride_size

		- Size of stride of depthwise post-op

	*
		- padding_l_size

		- Size of left and top paddings of depthwise post-op

.. index:: pair: function; append_binary
.. _doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void append_binary(:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` aalgorithm, const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src1_desc)

Appends a binary post-op.

The kind of this post operation is :ref:`dnnl_binary <doxid-group__dnnl__api__primitives__common_1gga9878f4795e53ad8443e5c0a29e53286aa1d51705e2642ce2ce19a3e163bb25f93>`.

In the simplest case when the binary is the only post operation, the computations would be:

.. code-block:: cpp

	dst[:] <- binary_op (dst[:], another_input[:])

where binary_op is configured with the given parameters. binary_op supports broadcast semantics for a second operand.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aalgorithm

		- Binary algorithm for the post-op.

	*
		- src1_desc

		- Memory descriptor of a second operand.

.. index:: pair: function; get_params_binary
.. _doxid-structdnnl_1_1post__ops_1a0a367859cac33d597743de491d26dcb9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_binary(
		int index,
		:ref:`algorithm<doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>`& aalgorithm,
		:ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& src1_desc
		) const

Returns the parameters of a binary post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the binary post-op.

	*
		- aalgorithm

		- Output binary algorithm kind.

	*
		- src1_desc

		- Output memory descriptor of a second operand.

.. index:: pair: function; append_prelu
.. _doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void append_prelu(int mask)

Appends a prelu forward post-op.

The kind of this post-op is :ref:`dnnl::primitive::kind::prelu <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a837c39f77d473b24eb27c0758d5c7c1b>`.

The post-op can be defined as:

.. code-block:: cpp

	dst[:] <- prelu(dst[:], weights[:])
	prelu:
	dst[:] <- dst[:] if dst[:] > 0
	dst[:] <- dst[:] * weights[:] if dst[:] <= 0

Example usage:

.. ref-code-block:: cpp

	int mb = 32, oc = 32,
	    oh = 14, ow = 14; // convolution output params
	// unique weights per output channel
	vector<float> weights = { ... };
	int oc_dim = 1; // mb_dim = 0, channel_dim = 1, height_dim = 2, ...
	
	// construct a convolution descriptor
	dnnl::convolution::desc conv_d;
	
	dnnl::primitive_attr attr;
	attr.append_prelu(1 << oc_dim);
	
	dnnl::primitive_desc conv_pd(conv_d, attr, engine);
	memory prelu_weights({{1}, dt::f32, {1}}, eng, weights.data());
	
	std::unordered_map<int, memory> conv_args;
	
	conv_args.insert(
	 {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_WEIGHTS, prelu_weights})

.. note:: 

   The order of dimensions does not depend on how elements are laid out in memory. For example:
   
   * for a 2D CNN activations tensor the order is always (n, c)
   
   * for a 4D CNN activations tensor the order is always (n, c, h, w)
   
   * for a 5D CNN weights tensor the order is always (g, oc, ic, kh, kw)
   
   
Prelu weights tensor is passed in runtime execution phase. Prelu weights tensor data type is implicitly assumed as f32 using plain layout (a, ab, acb, acdb, acdeb).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mask

		- Defines the correspondence between the output tensor dimensions and the prelu weights tensor. The set i-th bit indicates that a dedicated weights value is used for each index along that dimension. Set the mask to 0 to use a common weights value for the whole output tensor.

.. index:: pair: function; get_params_prelu
.. _doxid-structdnnl_1_1post__ops_1a0cacfa74daa86b05bedb351d905e1516:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void get_params_prelu(int index, int& mask) const

Returns the parameters of a prelu post-op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Index of the prelu post-op.

	*
		- mask

		- Weights mask of prelu post-op.

