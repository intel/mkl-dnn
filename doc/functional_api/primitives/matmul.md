Matrix Multiplication {#dev_guide_matmul}
=========================================

>
> [API Reference](@ref dnnl_api_matmul)
>

## General

The matrix multiplication (MatMul) primitive computes the product of two 2D
tensors with optional bias addition (the variable names follow the standard @ref
dev_guide_conventions):

\f[
    \dst(m, n) =
        \sum_{k=0}^{K - 1} \left(
            \src(m, k) \cdot \weights(k, n)
        \right) +
        \bias(m, n)
\f]

The MatMul primitive also supports batching multiple independent matrix
multiplication operations, in which case the tensors can be up to 12D:

\f[
    \dst(bs_0, bs_1, \ldots, m, n) =
        \sum_{k=0}^{K - 1} \left(
            \src(bs_0, bs_1, \ldots, m, k) \cdot
            \weights(bs_0, bs_1, \ldots, k, n) \right) + \bias(bs_0, bs_1, \ldots, m, n)
\f]

MatMul also supports implicit broadcast semantics, i.e., \src can be broadcasted
into \weights if the corresponding dimension in \src is 1 (and vice versa).
However, all tensors (including \bias, if it exists) must have the same number
of dimensions.

The shape of \dst only depends on \src and \weights tensors. The \bias cannot
change the dimensions of \dst by broadcasting. In other words, for every
dimension, the following constraint must hold true:
`dimension(bias) == dimension(dst) || dimension(bias) == 1`.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Argument                         | Index                                                                      | Type   |
|----------------------------------|----------------------------------------------------------------------------|--------|
| \src                             | DNNL_ARG_SRC                                                               | Input  |
| \weights                         | DNNL_ARG_WEIGHTS                                                           | Input  |
| \bias                            | DNNL_ARG_BIAS                                                              | Input  |
| \dst                             | DNNL_ARG_DST                                                               | Output |
| \f$\text{dropout output mask}\f$ | DNNL_ARG_ATTR_DROPOUT_MASK                                                 | Output |
| \f$\text{dropout probability}\f$ | DNNL_ARG_ATTR_DROPOUT_PROBABILITY                                          | Input  |
| \f$\text{dropout rng seed}\f$    | DNNL_ARG_ATTR_DROPOUT_SEED                                                 | Input  |
| \f$\text{binary post-op}\f$      | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1  | Input  |
| \                                | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_2  | Input  |
| \f$\text{prelu post-op}\f$       | DNNL_ARG_ATTR_MULTIPLE_POST_OP(prelu_post_op_position) \| DNNL_ARG_WEIGHTS | Input  |
| [scratchpad]                     | DNNL_ARG_SCRATCHPAD                                                        | Output |

[scratchpad]: @ref dev_guide_attributes_scratchpad

## Implementation Details

### General Notes

1. The MatMul primitive supports input and output tensors with run-time
   specified shapes and memory formats. The run-time specified dimensions or
   strides are specified using the #DNNL_RUNTIME_DIM_VAL wildcard value during
   the primitive initialization and creation stage. At the execution stage, the
   the user must pass fully specified memory objects so that the primitive is able
   to perform the computations. Note that the less information about shapes
   or format is available at the creation stage, the less performant the execution
   will be. In particular, if the shape is not known at the creation stage, you
   cannot use the special format tag #dnnl::memory::format_tag::any to enable an
   implementation to choose the most appropriate memory format for the
   corresponding input or output shapes. On the other hand, run-time specified
   shapes enable users to create a primitive once and use it in different
   situations.

2. Inconsistency with dimensions being "primitive-creation-time-defined" vs.
   "runtime-defined" is invalid. For example, \src and \weights with dimensions
   set to `{3, 4, 4}` and `{DNNL_RUNTIME_DIM_VAL, 4, 4}` respectively is
   invalid.

3. The broadcasting shape consistency check is not done for the dimensions with
   #DNNL_RUNTIME_DIM_VAL. Make sure the dimensions
   for the tensors are valid.

4. Multiple batch dimensions and broadcasting of batch dimensions of \src
   and \weights are supported for both CPU and GPU engines.

@note Check the @ref inference_int8_matmul_cpp to see #DNNL_RUNTIME_DIM_VAL
support in use.

### Data Types

The MatMul primitive supports the following combinations of data
types for source, destination, weights, and bias tensors:


| Source              | Weights                                | Destination                         | Bias                        |
|:--------------------|:---------------------------------------|:------------------------------------|:----------------------------|
| f64                 | f64                                    | f64                                 | f32, f16, bf16              |
| f32                 | f32, u8, s8, u4, s4, u2, u3            | f32                                 | f32, bf16, f16              |
| f16                 | f16, u8, s8, u4, s4, u2, u3            | f16, u8, s8                         | f32                         |
| f16                 | f16, u8, s8, u4, s4, u2, u3            | f32, f16                            | f32, f16                    |
| bf16                | bf16, u8, s8, u4, s4, u2, u3           | f32, bf16                           | f32, bf16                   |
| f32, bf16, f16      | u8, s8, u4, s4, u2, u3                 | f32, bf16, f16                      | f32, bf16, f16              |
| bf16, f16           | f8_e5m2, f8_e4m3, f4_e2m1              | f32, f16, bf16                      | f32, bf16, f16              |
| f8_e5m2, f8_e4m3    | f8_e5m2, f8_e4m3                       | f32, f16, bf16, f8_e5m2, f8_e4m3    | f32, bf16, f16              |
| f4_e2m1             | f4_e2m1                                | f32, f16, bf16, f4_e2m1             | f32, bf16, f16              |
| u8, s8              | u8, s8, u4, s4, u3                     | u8, s8, s32, f32, f16, bf16         | u8, s8, s32, f32, f16, bf16 |

@note Sub-byte weights types pack several elements into each byte, so every stride
greater than 1 in such memory descriptor must be a multiple of the number of
packed elements: 2 for the 4-bit types (`s4`, `u4`, `f4_e2m1`),
4 for `u2` (2-bit), and 8 for `u3` (3-bit, packs 8 elements into 3 bytes).

### Data Representation

The MatMul primitive expects the following tensors:

| Dims | Source                          | Weights                         | Destination                     | Bias                                                      |
|:-----|:--------------------------------|:--------------------------------|:--------------------------------|:----------------------------------------------------------|
| 2D   | M \f$\times\f$ K                | K \f$\times\f$ N                | M \f$\times\f$ N                | None or \f$(M \text{ or } 1) \times (N  \text{ or } 1)\f$ |
| ND   | S \f$\times\f$ M \f$\times\f$ K | W \f$\times\f$ K \f$\times\f$ N | D \f$\times\f$ M \f$\times\f$ N | None or B                                                 |

where for the sake of notational convenience, we have

\f[
S = \prod_{i = 0}^{ND - 3} \mathrm{src\_dims}[i], \; W = \prod_{i = 0}^{ND - 3} \mathrm{weights\_dims}[i] \\
D = \prod_{i = 0}^{ND - 3} \mathrm{\dst\_dims}[i], \; B = \prod_{i = 0}^{ND - 1} \left( \mathrm{\dst\_dims}[i] \mbox{ or } 1 \right)
\f]

The MatMul primitive is generally optimized for the case in which memory objects
use plain memory formats. Additionally, the \src and \weights must have at least
one of the axes `m` or `k` and `n` or `k` contiguous (i.e., `stride=1`)
respectively. However, it is recommended to use the placeholder memory format
#dnnl::memory::format_tag::any if an input tensor is reused across multiple
executions. In this case, the primitive will set the most appropriate memory
format for the corresponding input tensor.

The memory format of the destination tensor should always be plain with `n` axis
contiguous. For example, #dnnl::memory::format_tag::ab for the 2D case and
#dnnl::memory::format_tag::abc or #dnnl::memory::format_tag::bac for the 3D one.

### Attributes and Post-ops

Attributes and post-ops enable modifying the behavior of the MatMul primitive.
The following attributes and post-ops are supported:

| Type      | Operation                                                      | Description                                                                   | Restrictions                        |
|:----------|:---------------------------------------------------------------|:------------------------------------------------------------------------------|:------------------------------------|
| Attribute | [Scales](@ref dnnl::primitive_attr::set_scales_mask)           | [Scales](@ref dgaq_scaling)  the result by given scaling factor(s)                                |                                     |
| Attribute | [Zero-points](@ref dnnl::primitive_attr::set_zero_points_mask) | Sets [zero-point(s)](@ref dgaq_zps) for the corresponding tensors                             |                     |
| Attribute | [Dropout](@ref dnnl::primitive_attr::set_dropout)              | Applies pseudo-random [dropout](@ref dev_guide_attributes_dropout) to destination buffer, also fills mask buffer   |                                     |
| Attribute | [Precomputed reductions](@ref dnnl::primitive_attr::set_precomputed_reductions) | Sets [precomputed reductions](@ref dgaq_precomputed_reductions) for the corresponding tensors  |  Requires weight zero-points, full matrix mask, and 8-bit integer weights |
| Post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)                 | Applies an @ref dnnl_api_eltwise operation to the result                      |                                     |
| Post-op   | [Sum](@ref dnnl::post_ops::append_sum)                         | [Adds](@ref dnnl_api_sum) the operation result to the destination tensor instead of overwriting it |                                     |
| Post-op   | [Binary](@ref dnnl::post_ops::append_binary)                   | Applies a @ref dnnl_api_binary operation to the result                        | General binary post-op restrictions |
| Post-op   | [Prelu](@ref dnnl::post_ops::append_prelu)                     | Applies an @ref dnnl_api_prelu operation to the result                        |                                     |

The `mask` and `groups` parameters for scales and zero-points follow the
conventions described in the
[quantization guide](@ref dgaq_constructing_mask_and_groups).

Scales, zero-points, and dropout require additional memory arguments at
execution time. See the
[quantization guide](@ref dgaq_execution) and the
[dropout guide](@ref dev_guide_attributes_dropout) for details.

@note Check the [list of examples and tutorials](#examples) below to see
run-time attributes in use.

## Implementation Limitations

1. Check @ref dev_guide_data_types.

2. **GPU**
   - Supports up to 6 dimensions.
   - Source zero point mask of `0` is only supported.
   - Sum post-op doesn't support data types other than destination data type.
   - Bias of `bf16` data type is supported for configurations with `bf16` source data
     type and weights `bf16` data type, and up to three-dimensional matrices.
   - Optimized implementations for `f8` data type are available only on Intel(R)
     Data Center GPU Max Series and Intel(R) Xe2 Graphics.
   - Configuration with `s8`/`u8` source data type, `s8` weight data type and `bf16`
     destination data type doesn't support:
     * Destination zero point.
     * Runtime dimensions.
     * Three and higher-dimensional matrices.
   - The layout of dropout mask has to be exactly the same as that of dst.


3. **CPU**
   - Configurations with `s8`/`u8` source data type, `s8` weight data type and `f16`
     destination data type aren't supported.
   - Configurations with floating point source data type, integer weights data
     type and floating point destination data type are not optimized.
   - Block-wise (K-grouped, e.g. `mask = (1 << 0) | (1 << 1)`) weights scales
     require K group size to be a multiple of 16 elements.
   - The layout of dropout mask has to be exactly the same as that of dst.

## Performance Tips

- Use #dnnl::memory::format_tag::any for either of the input tensors if and
  only if the shape of the corresponding tensor is fully known at creation
  time and it is possible to cache reordered tensors across multiple primitive
  executions. For instance, a good candidate for reuse are the weights tensors
  during inference: their shapes and data types are known in advance; thus
  they can be reordered during the first inference pass and can be reused
  during the subsequent passes. However, if any of the input tensors cannot be
  reused, it is best to force the primitive to use the same format as that used
  by the tensors.

@anchor dev_guide_matmul_grouped_gemm

## Sparse Matrix Multiplication Support

### CSR encoding

Supported only for the CPU engine. Only one of the input tensors can be sparse.
The output tensor is always dense.

The following data type combinations are supported:

| Values (src, weight, dst)   | Indices  |
|:----------------------------|:---------|
| f16, f16, f16               | s32      |
| f32, f32, f32               | s32      |

The following format tags are supported for dense input tensors:

* `ab`
* `ba`

Dense output tensor supports only format tag `ab`.

@note Check the example @ref cpu_matmul_csr_cpp.

Benchdnn can be used to test matmul with a CSR input tensor as follows:
`./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128`

For the case above, the number of non-zero elements for the source tensor is
calculated as `max(4 * 1000000 * (1 - 0.99), 1)`.

### COO encoding

Supported only for the CPU and GPU engines. Only one of the input tensors can
be sparse. The output tensor is always dense.

The following data type combinations are supported:

| Values (src, weight, dst)   | Indices  |
|:----------------------------|:---------|
| f16, f16, f16               | s32      |
| f32, f32, f32               | s32      |

The following format tags are supported for dense weights tensor:

* ab
* ba

The following format tags are supported for dense destination tensor:

* ab

@note Check the example @ref cpu_matmul_coo_cpp.

Benchdnn can be used to test matmul with a COO input tensor as follows:
`./benchdnn --matmul --encoding=coo+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128`

For the case above, the number of non-zero elements for the source tensor is
calculated as `max(4 * 1000000 * (1 - 0.99), 1)`.

### PACKED encoding

Only the weights tensor is allowed to be sparse. The other tensors
are always dense.

In general, it is expected that all matmul related functionality (e.g. post-ops,
scales, zero-points, etc) that is supported for the dense weights should
also work for the sparse weights.

Currently, matmul has the following limitations for the PACKED encoding:
* Supported only for the CPU engine
* Only Intel Advanced Matrix Extensions (Intel AMX) instruction set
architecture (ISA) is supported
* Only `s8` data type for the weights is supported
* Only 1 batch dimension is supported

@note Check the example @ref cpu_matmul_weights_compression_cpp.

Benchdnn can be used to test matmul with the PACKED weights tensor as follows:
`./benchdnn --matmul --dt=s8:s8:s32 --encoding=:packed+0.99: 3x512x1024:1x1024x512`

For the case above, the number of non-zero elements for the weights tensor is
calculated as `max(1024 * 512 * (1 - 0.99), 1)`.

Refer to [Sparsity Advanced Topic](@ref dev_guide_sparsity) page for more
information on sparse encoding.

## Grouped GEMM Support

@note This is an [experimental feature](@ref dev_guide_experimental). Build oneDNN
with `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON` to enable grouped GEMM support.

Grouped GEMM enables matrix multiplication when one dimension varies across
groups, as occurs in Mixture-of-Experts (MoE) models where tokens are dynamically
routed to different experts.

The computation for grouped GEMM with \f$G\f$ groups is defined as:

\f[
    \dst_g(m, n) =
        \sum_{k=0}^{K - 1}
            \src_g(m, k) \cdot \weights_g(k, n)
        , \quad g = 0, \ldots, G-1
\f]

where \f$m \in [0, M_g)\f$ and \f$M_g\f$ is the number of rows in group \f$g\f$.

Two variants are supported defined to which dimension varies across groups.
Grouped tensors use the [grouped memory format](@ref dev_guide_grouped_mem):
values are stored as concatenated buffers with an offsets array marking group
boundaries.

- **Variable token dimension (M)** (e.g., MoE forward pass).
  The token count varies per expert, so source `[total_M, K]` and destination
  `[total_M, N]` are grouped (row-major),
  while weights are represented as a regular dense 3D tensor `[num_groups, K, N]`.
- **Variable contraction dimension (K)** (e.g., MoE backward pass).
  The contraction dimension varies per group, so source `[M, total_K]` (col-major)
  and weights `[total_K, N]` are grouped and with the same partition,
  while the destination is represented as a regular dense 3D tensor
  `[num_groups, M, N]`.


### Code Snippet

~~~cpp
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
auto src_md = memory::desc::grouped(
    {total_tokens, K}, memory::data_type::f32,
    0, num_groups);  // dimension 0 (M) varies per group

// Weights: standard 3D dense tensor [num_groups, K, N]
// Each expert has its own K by N weight matrix
auto weights_md = memory::desc({num_groups, K, N},
    memory::data_type::f32, memory::format_tag::abc);

// Destination: grouped encoding matching source structure
auto dst_md = memory::desc::grouped(
    {total_tokens, N}, memory::data_type::f32,
    0, num_groups);

auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);

// Offsets mark the boundary of each expert's tokens
// Format: [end_expert0, end_expert1, end_expert2, end_expert3]
std::vector<int32_t> offsets = {800, 1400, 1400, 2350};

// Set offsets for both input and output memory objects
auto src_mem = memory(src_md, engine, {src_data, offsets.data()});
auto dst_mem = memory(dst_md, engine, {dst_data, offsets.data()});
~~~

### Attributes Support

Setting attributes for grouped GEMM follows the regular matmul attribute API.
Below are some examples of common use cases for MoE workloads.
For more details on how to set attributes, refer to the @ref dev_guide_attributes page.

#### Scales and Zero Points

Scales and zero points for grouped GEMM follow the same API as regular matmul.
Tensors must use dense memory descriptors.
The data in the scales/zero points tensor must follow the same flat concatenated
order as the grouped source/weights.

Per-token source scales:
~~~cpp
attr.set_scales_mask(DNNL_ARG_SRC, (1 << 0));  // Varies along M dimension
// Scale tensor: [total_tokens] - one scale per token
// Layout: concatenated like source data, uses same offsets
~~~

K-grouped source scales with group size of 128:
~~~cpp
attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, 128}, memory::data_type::f16);
// Scale tensor: [total_tokens, K/128] - one scale per (token, K-block)
// Layout: concatenated like source data, uses same offsets
~~~

Per-expert-column weight scales:
~~~cpp
attr.set_scales_mask(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2));
// Scale tensor: [num_groups, N] - dense 2D tensor
// Layout: standard ab layout
~~~

K-grouped weight scales (e.g., MXFP8 with block size 32):
~~~cpp
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2),
    {32, 1}, memory::data_type::e8m0);
// Groups are 2D {gK, gN} and are applied to the last two dimensions of the tensor
// Scale tensor: [num_groups, K/32, N] - dense 3D tensor
// Layout: standard abc layout
~~~

Bias per expert:
~~~cpp
// Bias: [num_groups, N] - dense 2D tensor
auto bias_md = memory::desc({num_groups, N},
    memory::data_type::f32, memory::format_tag::ab);
// Layout: standard ab layout
~~~

#### Post-ops Support

Post-ops for grouped GEMM follow the same API as regular matmul.
Any eltwise algorithm and binary multiplication post-ops are supported.

Binary post-op tensors accept both dense and grouped memory descriptors.
Note that when a grouped descriptor is provided, we use the binary tensor's
own per-group offsets for addressing, but they must describe the same grouped
partition as the grouped destination tensor.

SiLU and element-wise matrix multiplication as post-op:
~~~cpp
// [total_tokens, N] tensor, that could be either grouped or dense with
// memory::format_tag::ab
auto binary_md = /*...*/;

post_ops po;
po.append_eltwise(algorithm::eltwise_swish, 1.0f, 0.f);  // SiLU first
po.append_binary(algorithm::binary_mul, binary_md);      // then mul
attr.set_post_ops(po);
~~~

Note that the dense tensor must follow the same flat concatenated order as the grouped dst.

Binary multiply with per-row broadcast:
~~~cpp
// [total_tokens, 1] tensor, that is one scalar per row
// in the same flat concatenated order as the grouped destination
auto routing_md = memory::desc({total_tokens, 1},
    memory::data_type::f32, memory::format_tag::ab);

post_ops po;
po.append_binary(algorithm::binary_mul, routing_md);
attr.set_post_ops(po);
~~~

Binary multiply with per-group broadcast (one value per expert):
~~~cpp
// [num_groups, 1] tensor, one scalar per expert
auto scale_md = memory::desc({num_groups, 1},
    memory::data_type::f32, memory::format_tag::ab);

post_ops po;
po.append_binary(algorithm::binary_mul, scale_md);
attr.set_post_ops(po);
~~~

### Execution Hints

An optional execution-time hint `DNNL_ARG_HINT_MAX_GROUP_SIZE` can be provided to
communicate the maximum size of the group across the variable dimension for the
current execution call. Implementations may choose to use this input to tune
dispatch, and therefore using this hint may provide performance benefits.

If chosen, the hint is passed as a host scalar `s32` memory at execution time:
~~~cpp
int32_t max_size = 950; // upper bound on variable dimension across all groups for this call
auto hint_md = memory::desc::host_scalar(memory::data_type::s32);
auto hint_mem = memory(hint_md, engine, &max_size);

matmul_prim.execute(stream, {
    {DNNL_ARG_SRC, src_mem},
    {DNNL_ARG_WEIGHTS, weights_mem},
    {DNNL_ARG_DST, dst_mem},
    {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}  // optional
});
~~~

@warning Providing a value smaller than the actual maximum variable dimension across
groups for the current call will produce incorrect results. It is the caller's
responsibility to ensure the hint is a valid upper bound.

### Implementation Notes

The following are supported:
- Source and destination must use identical grouping.
- Scales attribute for source and weights tensors:
  - Source Scales: row-wise (`mask = (1 << 0)`) and K-grouped
    (`mask = (1 << 0) | (1 << 1)`) with group specification are supported.
    The scale tensor follows the same concatenated layout as src, with total
    size `[total_tokens, K/gK]`.
  - Weight Scales: column-wise (`mask = (1 << 0) | (1 << 2)`) and
    K-grouped or K- and N-grouped (`mask = (1 << 0) | (1 << 1) | (1 << 2)`)
    with group specification are supported. The scale tensor has size
    `[num_groups, K/gK, N/gN]`.
  - Scales are not supported when the corresponding tensor data type is
    `f32`, `bf16`, or `f16`. Scale data type depends on tensor data type:

    | Tensor data type           | Scale data type           |
    |:---------------------------|:--------------------------|
    | f8_e5m2, f8_e4m3, f4_e2m1  | f32, e8m0, f8_e4m3        |
    | u8, s8, s4, u4, u3         | f32, bf16, f16            |
- Zero points attribute for source and weights tensors:
  - The masks must match the scales mask. The exception is a common weights
    zero point (`mask = 0`), which is supported for `u3` weights only.
  - Source zero points data types include `u8`, `s8`, `s32`
  - Weights zero points data types include `u8`, `s8`, `u4`, `s4`, `s32`
- Post-ops: eltwise and binary multiplication post-ops are supported. Binary post-op
  tensors accept both dense and grouped memory descriptors.
  Common patterns include `eltwise_swish` (SiLU) or
  `eltwise_gelu_tanh` / `eltwise_gelu_erf` (GELU) for the gate activation,
  `binary_mul` with a `[total_tokens, N]` grouped or dense tensor,
  `binary_mul` with a `[total_tokens, 1]` dense tensor,
  and `[num_groups, 1]` for a per-group global scale (e.g. NVFP4).
  Scalar `[1, 1]` binary post-ops are not supported for grouped matmul.
  For grouped binary tensors, the per-group offsets must match the grouped dst
  partition.

 @attention The GPU implementation supports at most one dense binary multiplication
  post-op and at most one grouped binary multiplication post-op. Providing more
  than one of either kind is not supported on GPU.

- Bias supports per-expert shape.
- Only default attributes are supported for the variable-K variant.
- Supported on CPU and GPU engines.

#### Supported Data Types

The following combinations of data types for source, destination, weights, and bias tensors are supported.

| Source           | Weights                | Destination    | Bias           |
|:-----------------|:-----------------------|:---------------|:---------------|
| f32, bf16, f16   | f32, bf16, f16         | f32, bf16, f16 | f32, bf16, f16 |
| f8_e5m2, f8_e4m3 | f8_e5m2, f8_e4m3       | f32, bf16, f16 |                |
| f4_e2m1          | f4_e2m1                | f32, bf16, f16 |                |
| f32, bf16, f16   | u8, s8, s4, u4, u3 (1) | f32, bf16, f16 | f32, bf16, f16 |
| u8, s8           | u8, s8, s4, u4         | f32, bf16, f16 | f32, bf16, f16 |

Footnotes:
1. Weight-Only Quantization (WOQ): floating-point source with integer weights
   requires weight scales attribute and `fpmath` mode with `apply_to_int` enabled.


## Examples

See @ref dev_guide_examples page for a complete list. MatMul examples are listed in the
[Matrix Multiplication](@ref examples_matmul) section.
