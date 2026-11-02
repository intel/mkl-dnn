.. index:: pair: page; Matrix Multiplication
.. _doxid-dev_guide_matmul:

Matrix Multiplication
=====================

:ref:`API Reference <doxid-group__dnnl__api__matmul>`

General
~~~~~~~

The matrix multiplication (MatMul) primitive computes the product of two 2D tensors with optional bias addition (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(m, n) = \sum_{k=0}^{K - 1} \left( \src(m, k) \cdot \weights(k, n) \right) + \bias(m, n)

The MatMul primitive also supports batching multiple independent matrix multiplication operations, in which case the tensors can be up to 12D:

.. math::

	\dst(bs_0, bs_1, \ldots, m, n) = \sum_{k=0}^{K - 1} \left( \src(bs_0, bs_1, \ldots, m, k) \cdot \weights(bs_0, bs_1, \ldots, k, n) \right) + \bias(bs_0, bs_1, \ldots, m, n)

MatMul also supports implicit broadcast semantics, i.e., :math:`\src` can be broadcasted into :math:`\weights` if the corresponding dimension in :math:`\src` is 1 (and vice versa). However, all tensors (including :math:`\bias`, if it exists) must have the same number of dimensions.

The shape of :math:`\dst` only depends on :math:`\src` and :math:`\weights` tensors. The :math:`\bias` cannot change the dimensions of :math:`\dst` by broadcasting. In other words, for every dimension, the following constraint must hold true: ``dimension(bias) == dimension(dst) || dimension(bias) == 1``.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==========================================================  ==================================================================================================================================================================  =======  
Argument                                                    Index                                                                                                                                                               Type     
==========================================================  ==================================================================================================================================================================  =======  
:math:`\src`                                                DNNL_ARG_SRC                                                                                                                                                        Input    
:math:`\weights`                                            DNNL_ARG_WEIGHTS                                                                                                                                                    Input    
:math:`\bias`                                               DNNL_ARG_BIAS                                                                                                                                                       Input    
:math:`\dst`                                                DNNL_ARG_DST                                                                                                                                                        Output   
:math:`\text{dropout output mask}`                          DNNL_ARG_ATTR_DROPOUT_MASK                                                                                                                                          Output   
:math:`\text{dropout probability}`                          DNNL_ARG_ATTR_DROPOUT_PROBABILITY                                                                                                                                   Input    
:math:`\text{dropout rng seed}`                             DNNL_ARG_ATTR_DROPOUT_SEED                                                                                                                                          Input    
:math:`\text{binary post-op}`                               :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1    Input    
                                                            :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_2    Input    
:math:`\text{prelu post-op}`                                :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(prelu_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_WEIGHTS   Input    
:ref:`scratchpad <doxid-dev_guide_attributes_scratchpad>`   DNNL_ARG_SCRATCHPAD                                                                                                                                                 Output   
==========================================================  ==================================================================================================================================================================  =======

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The MatMul primitive supports input and output tensors with run-time specified shapes and memory formats. The run-time specified dimensions or strides are specified using the :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>` wildcard value during the primitive initialization and creation stage. At the execution stage, the the user must pass fully specified memory objects so that the primitive is able to perform the computations. Note that the less information about shapes or format is available at the creation stage, the less performant the execution will be. In particular, if the shape is not known at the creation stage, you cannot use the special format tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` to enable an implementation to choose the most appropriate memory format for the corresponding input or output shapes. On the other hand, run-time specified shapes enable users to create a primitive once and use it in different situations.

#. Inconsistency with dimensions being "primitive-creation-time-defined" vs. "runtime-defined" is invalid. For example, :math:`\src` and :math:`\weights` with dimensions set to ``{3, 4, 4}`` and ``{DNNL_RUNTIME_DIM_VAL, 4, 4}`` respectively is invalid.

#. The broadcasting shape consistency check is not done for the dimensions with :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`. Make sure the dimensions for the tensors are valid.

#. Multiple batch dimensions and broadcasting of batch dimensions of :math:`\src` and :math:`\weights` are supported for both CPU and GPU engines.

.. note:: 

   Check the :ref:`MatMul Tutorial: INT8 Inference <doxid-inference_int8_matmul_cpp>` and :ref:`MatMul Tutorial: Comparison with SGEMM <doxid-cpu_sgemm_and_matmul_cpp>` to see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>` support in use.
   
   


Data Types
----------

The MatMul primitive supports the following combinations of data types for source, destination, weights, and bias tensors:

====================  ======================================  ====================================  ============================  
Source                Weights                                 Destination                           Bias                          
====================  ======================================  ====================================  ============================  
f64                   f64                                     f64                                   f64, f32, f16, bf16, s8, u8   
f32                   f32, u8, s8, u4, s4                     f32                                   f32, bf16, f16, u8, s8        
f16                   f16, u8, s8, u4, s4                     f16, u8, s8                           f32                           
f16                   f16, u8, s8, u4, s4                     f32, f16                              f32, f16                      
bf16                  bf16, u8, s8, u4, s4                    f32, bf16                             f32, bf16                     
f32, bf16, f16        u8, s8, u4, s4                          f32, bf16, f16                        f32, bf16, f16                
bf16, f16             f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0(1)   f32, f16, bf16                        f32, bf16, f16                
f8_e5m2, f8_e4m3      f8_e5m2, f8_e4m3                        f32, f16, bf16, f8_e5m2, f8_e4m3      f32, bf16, f16                
f4_e2m1, f4_e3m0(1)   f4_e2m1, f4_e3m0(1)                     f32, f16, bf16, f4_e2m1, f4_e3m0(1)   f32, bf16, f16                
u8, s8                u8, s8, u4, s4                          u8, s8, s32, f32, f16, bf16           u8, s8, s32, f32, f16, bf16   
====================  ======================================  ====================================  ============================

Footnotes:

#. f4_e3m0 is deprecated, and will be removed in a future release.

Data Representation
-------------------

The MatMul primitive expects the following tensors:

=====  ====================================  ====================================  ====================================  ===========================================================  
Dims   Source                                Weights                               Destination                           Bias                                                         
=====  ====================================  ====================================  ====================================  ===========================================================  
2D     M :math:`\times` K                    K :math:`\times` N                    M :math:`\times` N                    None or :math:`(M \text{ or } 1) \times (N \text{ or } 1)`   
ND     S :math:`\times` M :math:`\times` K   W :math:`\times` K :math:`\times` N   D :math:`\times` M :math:`\times` N   None or B                                                    
=====  ====================================  ====================================  ====================================  ===========================================================

where for the sake of notational convenience, we have

.. math::

	S = \prod_{i = 0}^{ND - 3} \mathrm{src\_dims}[i], \; W = \prod_{i = 0}^{ND - 3} \mathrm{weights\_dims}[i] \\ D = \prod_{i = 0}^{ND - 3} \mathrm{\dst\_dims}[i], \; B = \prod_{i = 0}^{ND - 1} \left( \mathrm{\dst\_dims}[i] \mbox{ or } 1 \right)

The MatMul primitive is generally optimized for the case in which memory objects use plain memory formats. Additionally, the :math:`\src` and :math:`\weights` must have at least one of the axes ``m`` or ``k`` and ``n`` or ``k`` contiguous (i.e., ``stride=1``) respectively. However, it is recommended to use the placeholder memory format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` if an input tensor is reused across multiple executions. In this case, the primitive will set the most appropriate memory format for the corresponding input tensor.

The memory format of the destination tensor should always be plain with ``n`` axis contiguous. For example, :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>` for the 2D case and :ref:`dnnl::memory::format_tag::abc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>` or :ref:`dnnl::memory::format_tag::bac <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ec16df80b57696a03bb364410061f3>` for the 3D one.

Attributes and Post-ops
-----------------------

Attributes and post-ops enable modifying the behavior of the MatMul primitive. The following attributes and post-ops are supported:

==========  =======================================================================================================  ========================================================================================================================================  =================================================  
Type        Operation                                                                                                Description                                                                                                                               Restrictions                                       
==========  =======================================================================================================  ========================================================================================================================================  =================================================  
Attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`                   :ref:`Scales <doxid-dev_guide_attributes_quantization_1dgaq_scaling>` the result by given scaling factor(s)                                                                                  
Attribute   :ref:`Zero-points <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`              Sets :ref:`zero-point(s) <doxid-dev_guide_attributes_quantization_1dgaq_zps>` for the corresponding tensors                                                                                  
Attribute   :ref:`Dropout <doxid-structdnnl_1_1primitive__attr_1abe989b6c932434a755bade257d299755>`                  Applies pseudo-random :ref:`dropout <doxid-dev_guide_attributes_dropout>` to destination buffer, also fills mask buffer                                                                      
Attribute   :ref:`Precomputed reductions <doxid-structdnnl_1_1primitive__attr_1a24a349d345ac97756a54b01b634b1b3c>`   Sets :ref:`precomputed reductions <doxid-dev_guide_attributes_quantization_1dgaq_precomputed_reductions>` for the corresponding tensors   Requires weight zero-points and full matrix mask   
Post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`                        Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result                                                                                                          
Post-op     :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`                            :ref:`Adds <doxid-group__dnnl__api__sum>` the operation result to the destination tensor instead of overwriting it                                                                           
Post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`                         Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result                                                          General binary post-op restrictions                
Post-op     :ref:`Prelu <doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70>`                          Applies an :ref:`PReLU <doxid-group__dnnl__api__prelu>` operation to the result                                                                                                              
==========  =======================================================================================================  ========================================================================================================================================  =================================================

The following masks are supported by the primitive:

* 0, which applies one scale / zero point value to an entire tensor

* 1, which applies a scale / zero point values along ``k`` -dimension for :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`. Values could be grouped along this dimension via specifying scales / zero points shapes for the attribute (see the code example :ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`).

* 2, which applies a scale / zero point values per column along the ``n`` -dimension for :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`.

When scales and/or zero-points masks are specified, the user must provide the corresponding scales and/or zero-points as additional input memory objects with argument ``DNNL_ARG_ATTR_SCALES | DNNL_ARG_${MEMORY_INDEX}`` or ``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_${MEMORY_INDEX}`` during the execution stage. For instance, a source tensor zero points memory argument would be passed with index (``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC``).

When Dropout is specified, at the execution stage the user must provide 2 input memory objects with :ref:`DNNL_ARG_ATTR_DROPOUT_PROBABILITY <doxid-group__dnnl__api__primitives__common_1ga97211384b4573d1f4160f88b1fc764fb>` (1x1x...x1 f32 value from 0.f to 1.f) and :ref:`DNNL_ARG_ATTR_DROPOUT_SEED <doxid-group__dnnl__api__primitives__common_1ga00f50d06255b72868fdba100fbb53b1a>` (1x1x...x1 s32 value from INT_MIN to INT_MAX), and 1 output memory object with :ref:`DNNL_ARG_ATTR_DROPOUT_MASK <doxid-group__dnnl__api__primitives__common_1ga75f71c3764ec4754333bd400a290c217>` (u8 memory buffer that shares its shape with the destination buffer).

.. note:: 

   Check the `list of examples and tutorials <#examples>`__ below to see run-time attributes in use.
   
   


Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Check :ref:`Data Types <doxid-dev_guide_data_types>`.

#. GPU
   
   * Supports up to 6 dimensions.
   
   * Source zero point mask of ``0`` is only supported.
   
   * Sum post-op doesn't support data types other than destination data type.
   
   * Bias of bf16 data type is supported for configurations with bf16 source data type and weights bf16 data type, and up to three-dimensional matrices.
   
   * Optimized implementations for fp8 data type are available only on Intel(R) Data Center GPU Max Series and Intel(R) Xe2 Graphics.
   
   * Configuration with int8 source data type, s8 weight data type and bf16 destination data type doesn't support:
     
     * Destination zero point.
     
     * Runtime dimensions.
     
     * Three and higher-dimensional matrices.
   
   * The layout of dropout mask has to be exactly the same as that of dst.

#. CPU
   
   * Configurations with int8 source data type, s8 weight data type and f16 destination data type aren't supported.
   
   * Configurations with floating point source data type, integer weights data type and floating point destination data type are not optimized.
   
   * The layout of dropout mask has to be exactly the same as that of dst.

Performance Tips
~~~~~~~~~~~~~~~~

* Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for either of the input tensors if and only if the shape of the corresponding tensor is fully known at creation time and it is possible to cache reordered tensors across multiple primitive executions. For instance, a good candidate for reuse are the weights tensors during inference: their shapes and data types are known in advance; thus they can be reordered during the first inference pass and can be reused during the subsequent passes. However, if any of the input tensors cannot be reused, it is best to force the primitive to use the same format as that used by the tensors.

:target:`doxid-dev_guide_matmul_1dev_guide_matmul_grouped_gemm`

Sparse Matrix Multiplication Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSR encoding
------------

Supported only for the CPU engine. Only one of the input tensors can be sparse. The output tensor is always dense.

The following data type combinations are supported:

==========================  ========  
Values (src, weight, dst)   Indices   
==========================  ========  
f16, f16, f16               s32       
f32, f32, f32               s32       
==========================  ========

The following format tags are supported for dense input/output tensors:

* ab

.. note:: 

   Check the example :ref:`MatMul Primitive with Sparse Memory in CSR Format <doxid-cpu_matmul_csr_cpp>`.
   
   
Benchdnn can be used to test matmul with a CSR input tensor as follows: ``./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128``

For the case above, the number of non-zero elements for the source tensor is calculated as ``max(4 * 1000000 * (1 - 0.99), 1)``.

COO encoding
------------

Supported only for the CPU and GPU engines. Only one of the input tensors can be sparse. The output tensor is always dense.

The following data type combinations are supported:

==========================  ========  
Values (src, weight, dst)   Indices   
==========================  ========  
f16, f16, f16               s32       
f32, f32, f32               s32       
==========================  ========

The following format tags are supported for dense weights tensor:

* ab

* ba

The following format tags are supported for dense destination tensor:

* ab

.. note:: 

   Check the example :ref:`MatMul Primitive with Sparse Memory in COO Format <doxid-cpu_matmul_coo_cpp>`.
   
   
Benchdnn can be used to test matmul with a COO input tensor as follows: ``./benchdnn --matmul --encoding=coo+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128``

For the case above, the number of non-zero elements for the source tensor is calculated as ``max(4 * 1000000 * (1 - 0.99), 1)``.

PACKED encoding
---------------

Only the weights tensor is allowed to be sparse. The other tensors are always dense.

In general, it is expected that all matmul related functionality (e.g. post-ops, scales, zero-points, etc) that is supported for the dense weights should also work for the sparse weights.

Currently, matmul has the following limitations for the PACKED encoding:

* Supported only for the CPU engine

* Only Intel Advanced Matrix Extensions (Intel AMX) instruction set architecture (ISA) is supported

* Only ``s8`` data type for the weights is supported

* Only 1 batch dimension is supported

.. note:: 

   Check the example :ref:`MatMul with Packed Sparse Weights <doxid-cpu_matmul_weights_compression_cpp>`.
   
   
Benchdnn can be used to test matmul with the PACKED weights tensor as follows: ``./benchdnn --matmul --dt=s8:s8:s32 --encoding=:packed+0.99: 3x512x1024:1x1024x512``

For the case above, the number of non-zero elements for the weights tensor is calculated as ``max(1024 * 512 * (1 - 0.99), 1)``.

Refer to :ref:`Sparsity Advanced Topic <doxid-dev_guide_sparsity>` page for more information on sparse encoding.

Grouped GEMM Support
~~~~~~~~~~~~~~~~~~~~

.. note:: 

   This is an :ref:`experimental feature <doxid-dev_guide_experimental>`. Build oneDNN with ``ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`` to enable grouped GEMM support.
   
   
Grouped GEMM enables matrix multiplication when one dimension varies across groups, as occurs in Mixture-of-Experts (MoE) models where tokens are dynamically routed to different experts.

The computation for grouped GEMM with :math:`G` groups is defined as:

.. math::

	\dst_g(m, n) = \sum_{k=0}^{K - 1} \src_g(m, k) \cdot \weights_g(k, n) , \quad g = 0, \ldots, G-1

where :math:`m \in [0, M_g)` and :math:`M_g` is the number of rows in group :math:`g`.

The source and destination tensors use :ref:`grouped memory format <doxid-dev_guide_grouped_mem>` because the number of tokens per expert varies dynamically in MoE workloads. The grouped encoding stores values as concatenated buffers with an offsets array specifying group boundaries. Weights are represented as a regular dense 3D tensor ``[num_groups, K, N]`` because all experts have uniform dimensions, making grouped encoding unnecessary.

Code Snippet
------------

.. ref-code-block:: cpp

	const memory::dim num_groups = 4;
	const memory::dim K = 512, N = 256;
	
	// MoE routing result:
	// Expert 0: 800 tokens
	// Expert 1: 600 tokens
	// Expert 2: 0 tokens
	// Expert 3: 950 tokens
	const memory::dim total_tokens = 2350;  // Sum of all token counts
	
	// Source: grouped encoding for variable M dimension
	// Descriptor: [total_tokens, K] with grouped encoding
	// Memory layout: [expert0_tokens | expert1_tokens | expert2_tokens | expert3_tokens]
	auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = memory::desc::grouped(
	    {total_tokens, K}, memory::data_type::f32,
	    0, num_groups);  // dimension 0 (M) varies per group
	
	// Weights: standard 3D dense tensor [num_groups, K, N]
	// Each expert has its own K by N weight matrix
	auto :ref:`weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>` = memory::desc({num_groups, K, N},
	    memory::data_type::f32, memory::format_tag::abc);
	
	// Destination: grouped encoding matching source structure
	auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = memory::desc::grouped(
	    {total_tokens, N}, memory::data_type::f32,
	    0, num_groups);
	
	auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);
	
	// Offsets mark the boundary of each expert's tokens
	// Format: [end_expert0, end_expert1, end_expert2, end_expert3]
	std::vector<int32_t> offsets = {800, 1400, 1400, 2350};
	
	// Set offsets for both input and output memory objects
	auto src_mem = memory(src_md, engine, {src_data, offsets.data()});
	auto dst_mem = memory(dst_md, engine, {dst_data, offsets.data()});

Attributes Support
------------------

Setting attributes for grouped GEMM follows the regular matmul attribute API. Below are some examples of common use cases for MoE workloads. For more details on how to set attributes, refer to the :ref:`Primitive Attributes <doxid-dev_guide_attributes>` page.

Per-token source scales:

.. ref-code-block:: cpp

	attr.set_scales_mask(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, (1 << 0));  // Varies along M dimension
	// Scale tensor: [total_tokens] - one scale per token
	// Layout: concatenated like source data, uses same offsets

K-grouped source scales with group size of 128:

.. ref-code-block:: cpp

	attr.set_scales(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, (1 << 0) | (1 << 1), {1, 128}, memory::data_type::f16);
	// Scale tensor: [total_tokens, K/128] - one scale per (token, K-block)
	// Layout: concatenated like source data, uses same offsets

Per-expert-column weight scales:

.. ref-code-block:: cpp

	attr.set_scales_mask(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, (1 << 0) | (1 << 2));
	// Scale tensor: [num_groups, N] - dense 2D tensor
	// Layout: standard ab layout

Bias per expert:

.. ref-code-block:: cpp

	// Bias: [num_groups, N] - dense 2D tensor
	auto bias_md = memory::desc({num_groups, N},
	    memory::data_type::f32, memory::format_tag::ab);
	// Layout: standard ab layout

Execution Hints
---------------

An optional execution-time hint ``DNNL_ARG_HINT_MAX_GROUP_SIZE`` can be provided to communicate the maximum size of the group across the variable dimension for the current execution call. Implementations may choose to use this input to tune dispatch, and therefore using this hint may provide performance benefits.

If chosen, the hint is passed as a host scalar ``s32`` memory at execution time:

.. ref-code-block:: cpp

	int32_t max_size = 950; // upper bound on variable dimension across all groups for this call
	auto hint_md = memory::desc::host_scalar(memory::data_type::s32);
	auto hint_mem = memory(hint_md, engine, &max_size);
	
	matmul_prim.execute(stream, {
	    {:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem},
	    {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, weights_mem},
	    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem},
	    {:ref:`DNNL_ARG_HINT_MAX_GROUP_SIZE <doxid-group__dnnl__api__primitives__common_1gad7100f22db424dfeae40977e8a405f1a>`, hint_mem}  // optional
	});

.. warning:: 

   Providing a value smaller than the actual maximum variable dimension across groups for the current call will produce incorrect results. It is the caller's responsibility to ensure the hint is a valid upper bound.
   
   


Implementation Notes
--------------------

The following are supported:

* Currently, only single dimension ``0`` can vary.

* Source and destination must use identical grouping.

* Scales attribute for source and weights tensors:
  
  * Source Scales: row-wise (``mask = (1 << 0)``) and K-grouped (``mask = (1 << 0) | (1 << 1)``) with group specification are supported. The scale tensor follows the same concatenated layout as src, with total size ``[total_tokens, K/gK]``.
  
  * Weight Scales: column-wise (``mask = (1 << 0) | (1 << 2)``) and K-grouped (``mask = (1 << 0) | (1 << 1) | (1 << 2)``) with group specification are supported.
  
  * Scale data type includes: ``f32``, ``bf16``, ``f16``, ``e8m0`` for MXFP8 and MXFP4, ``f8_e4m3`` for NVFP4 block scales.

* Zero points attribute for weights tensors with the same masks as scales.

* Post-ops: binary post-ops are supported (e.g., binary ``mul`` with a scalar ``f32`` tensor can be used to apply a global scale factor, as needed for NVFP4 two-level scaling).

* Bias supports per-expert shape.

* Supported on CPU and GPU engines.

Supported Data Types
++++++++++++++++++++

The following combinations of data types for source, destination, weights, and bias tensors are supported.

=================  ===================  ===============  ===============  
Source             Weights              Destination      Bias             
=================  ===================  ===============  ===============  
f32, bf16, f16     f32, bf16, f16       f32, bf16, f16   f32, bf16, f16   
f8_e5m2, f8_e4m3   f8_e5m2, f8_e4m3     f32, bf16, f16                    
f4_e2m1            f4_e2m1              f32, bf16, f16                    
f32, bf16, f16     u8, s8, s4, u4 (1)   f32, bf16, f16   f32, bf16, f16   
u8, s8             u8, s8, s4, u4       f32, bf16, f16   f32, bf16, f16   
=================  ===================  ===============  ===============

Footnotes:

#. Weight-Only Quantization (WOQ): floating-point source with integer weights requires weight scales attribute and ``fpmath`` mode with ``apply_to_int`` enabled.

Examples
~~~~~~~~

See :ref:`Examples and Tutorials <doxid-dev_guide_examples>` page for a complete list. MatMul examples are listed in the :ref:`Matrix Multiplication <doxid-dev_guide_examples_1examples_matmul>` section.

