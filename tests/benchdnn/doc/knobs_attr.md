# Attributes

## Usage
```
    --attr-scratchpad=MODE
    --attr-fpmath=MATHMODE[:APPLY_TO_INT]
    --attr-acc-mode=ACCMODE
    --attr-rounding-mode=ARG:MODE[+...]
    --attr-deterministic=BOOL
    --attr-dropout=PROBABILITY[:SEED[:TAG[:OFFSET[:HOST_SCALARS]]]]
    --attr-scales=ARG:MASK_INPUT[:SCALE[:DATA_TYPE]][+...]
                  ARG:MASK_INPUT[:DATA_TYPE[:GROUPS][:QUANT_MODE]][+...]
    --attr-zero-points=ARG:MASK_INPUT[:ZEROPOINT[:DATA_TYPE]][+...]
                       ARG:MASK_INPUT[:DATA_TYPE[:GROUPS][:QUANT_MODE]][+...]
    --attr-precomputed-reductions=ARG:MASK_INPUT:DATA_TYPE:GROUPS[+...]
    --attr-post-ops=SUM[:SCALE[:ZERO_POINT[:DATA_TYPE]]]
                    ELTWISE[:ALPHA[:BETA[:SCALE]]]
                    DW:KkSsPp[:DST_DT]
                    BINARY:DT[:MASK_INPUT[:TAG[:STRIDES]]]
                    PRELU[:MASK_INPUT]
```

## --attr-scratchpad
`--attr-scratchpad` specifies the scratchpad mode to be used for benchmarking.
`MODE` values can be `library` (the default) or `user`. Refer to
[scratchpad primitive attribute](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_scratchpad.html)
for details.

## --attr-fpmath
`--attr-fpmath` specifies the fpmath mode to be used for benchmarking.
`MATHMODE` values can be any of `strict` (the default), `bf16`, `f16`, `tf32`,
or `any`.
`APPLY_TO_INT` values can be either `true` or `false` (the default).
Refer to
[fpmath primitive attribute](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html)
for details.


## --attr-acc-mode
`--attr-acc-mode` specifies the accumulation mode to be used for benchmarking.
`ACCMODE` values can be any of `strict` (the default), `relaxed`, `any`, `f32`,
`s32` or `f16`. Refer to
[accumulation mode primitive attribute](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html)
for details.

## --attr-rounding-mode
`--attr-rounding-mode` specifies the rounding mode to be used for benchmarking.
`ARG` specifies which memory argument will be modified. Supported values are:
  - `dst` corresponds to `DNNL_ARG_DST`.
`MODE` specifies which mode to apply to the corresponding memory
argument. Supported values are: `environment` (default) and `stochastic`.  Refer
to [rounding mode primitive attribute](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_rounding_mode.html)
for details.

## --attr-deterministic
`--attr-deterministic` specifies the deterministic mode to be used for
benchmarking. `BOOL` values can be `true`, which enables the deterministic
mode and `false` (the default), which disables it. Refer to
[deterministic primitive attribute](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html)
for details.

## --attr-dropout
`--attr-dropout` defines the dropout attribute; right before the post-ops get
applied, dropout fills a part of the output buffer with zeroes at random offsets
(using a version of Philox as PRNG) and divides what's left by (1 - p), where p
means probability, see below.
`PROBABILITY` is a floating-point value between `0` and `1` that specifies how
likely it is for any given output value to be zeroed (i.e. 'dropped out'): when
`0` is specified the output buffer is to remain intact, when `1` is specified
all output values are to be dropped out.
`SEED` is the 64-bit integer seed of the Philox RNG, 0 by default.
`TAG` specifies the memory format of the output buffer where the
dropout mask will be stored. When set to `undef`, no dropout mask
memory is passed. `TAG` values use the same notation as in
drivers. The default value of `TAG` is `any`. Refer to
[tags](knobs_tag.md) for details.
`OFFSET` is a 64-bit integer used as offset added to the index upon
calls to the Philox RNG, 0 (unset) by default.
`HOST_SCALARS` is a boolean specifying if scalar parameters are passed
as host_scalars (`true`) or as device memory objects (`false`, default).

## --attr-scales
`--attr-scales` defines per memory argument primitive scales attribute.
`ARG` specifies which memory argument will be modified. Supported values are:
  - `src` or `src0` corresponds to `DNNL_ARG_SRC`.
  - `src1` corresponds to `DNNL_ARG_SRC_1`.
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`.
  - `dst` corresponds to `DNNL_ARG_DST`.
  - `msrci` corresponds to `DNNL_ARG_MULTIPLE_SRC + i`.

`MASK_INPUT` specifies the way scale values will be applied to an `ARG` tensor.
`MASK_INPUT` may be provided in two ways:
* An integer value encoding the mask. This is the way masks are reported in
  oneDNN verbose logs.
* A string literal value encoding the mask (see below). Possible options are:
  - `common`         corresponds to `mask = 0` and means a whole tensor will be
                     multiplied by a single `SCALE` value.
  - `host_scalar`    corresponds to `mask = 0` with an additional property of
                     residing in the host memory.
  - `per_dim_0`      corresponds to `mask = 1 << 0` and means elements of dim0
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[0].
  - `per_dim_1`      corresponds to `mask = 1 << 1` and means elements of dim1
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[1].
  - `per_dim_01`     corresponds to `mask = (1 << 0) + (1 << 1)` and means
                     elements of dim0 and dim1 will be multiplied by scale
                     factors different for a pair of {dim0, dim1} points.
                     Number of scale factors is equal to dims[0] * dims[1].
  - `per_oc`         same as `per_dim_0` for non-grouped case of `WEI` argument,
                     same as `per_dim_01` for grouped case of `WEI` argument,
                     same as `per_dim_1` for arguments other than `WEI`.
  - `per_ocic`       For non-Matmul primitives:
                     same as `per_dim_01` for non-grouped case of `WEI` argument,
                     corresponds to `mask = (1 << 1) + (1 << 2)` for grouped case
                     of `WEI` argument.
                     For Matmul primitive:
                     same as `per_dim_01` for non-batched case,
                     corresponds to `mask = (1 << batch_ndims) + (1 << batch_ndims + 1)`
                     for batched case.
  - `per_dim_2`      corresponds to `mask = 1 << 2` and means elements of dim2
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[2].
                     Currently supported only in matmul primitive for 3D tensors.
  - `per_dim_3`      corresponds to `mask = 1 << 3` and means elements of dim3
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[3].
                     Currently supported only in matmul primitive for 4D tensors.
  - `per_tensor`     means each element, or a group elements expressed by
                     `GROUP`, of original tensor will be multiplied by a unique
                     scale value. The number of scale factors equals to
                     `nelems`, or `nelems / GROUP` for a grouped case.
  - `mx`             corresponds to `per_tensor` value with an additional scales
                     property to be computed by the primitive and stored,
                     following the OCP MX specification. Applicable only for
                     destination scales.
  - `dynamic_fp`     same as `mx` value, but uses different formula to compute
                     scales: `to_scale_dt(amax(x) / max(dst_dt))`

Note: some values can't be expressed with an integer as they encode not only the
mask value but also a special mechanics behind. Such values are `host_scalar`,
`mx`, and `dynamic_fp`.

`SCALE` is required for the `common`, or `mask = 0`, value only, and specifies a
floating-point value which is passed for execution at runtime. Specifying a
value for any other policies will trigger an error.

`DATA_TYPE` specifies data type for scale factors. Supported values are
`f32` (the default), `f16`, `bf16`, `e8m0`, `f8_e5m2`, `f8_e4m3`.

`GROUPS` specifies how scales are grouped along dimensions where multiple scale
factors are used.

`QUANT_MODE` specifies the quantization mode and maps directly to a library
`dnnl_quantization_mode_t` value. Supported values are:
  - `sazp`       (the default) static mode with the scale applied after the
                 zero-point: `x = scale * (q - zp)`. Zero-points are integers.
  - `sbzp`       static mode with the scale applied before the zero-point:
                 `x = scale * q - zp`. Zero-points are floating-point and the
                 mode must be set on both the scales and the zero-points of the
                 weights.
  - `mx`         dynamic mode computed by the primitive following the OCP MX
                 specification. Applicable only to destination scales.
  - `dynamic_fp` dynamic mode computed by the primitive as
                 `to_scale_dt(amax(x) / max(dst_dt))`. Applicable only to
                 destination scales.

To specify more than one memory argument for this attribute, `+` delimiter is
used.

## --attr-zero-points
`--attr-zero-points` defines zero points per memory argument primitive
attribute. This attribute is supported only for integer data types as of now.

`ARG` specifies which memory argument will be modified. Supported values are:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`
  - `dst` corresponds to `DNNL_ARG_DST`

`MASK_INPUT` has the same semantics and meaning as for `--attr-scales`.

Supported string literal values (with same semantics) are the same as for
`--attr-scales`, except `mx` and `dynamic_fp` which apply to scales only:
  - `common`
  - `host_scalar`
  - `per_dim_0`
  - `per_dim_1`
  - `per_dim_01`
  - `per_oc`
  - `per_ocic`
  - `per_dim_2`
  - `per_dim_3`
  - `per_tensor`

`ZEROPOINT` is required for the `common`, or `mask = 0` value only, and
specifies an integer value which is passed for execution at runtime. Specifying
a value for any other policies will trigger an error.

`DATA_TYPE` specifies data type for zero points. Supported values are
`s32` (the default), `s8`, `u8`, `s4`, `u4`, and `f16`, `bf16` with the `sbzp`
quantization mode (see `QUANT_MODE` below).

`GROUPS` specifies how zero points are grouped along dimensions where multiple
zero points are used.

`QUANT_MODE` has the same semantics as for `--attr-scales`. Only `sazp` (the
default) and `sbzp` values are supported. The `sbzp` mode must be set with a
matching mode on the scales for the same argument.

To specify more than one memory argument for this attribute, `+` delimiter is
used.

## --attr-precomputed-reductions
`--attr-precomputed-reductions` defines precomputed reductions per memory
argument primitive attribute. This attribute is supported only for integer data
types as of now.

`ARG` specifies which memory argument will be modified. Supported values are:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`

`MASK_INPUT` has the same semantics and meaning as for `--attr-scales`.

Supported string literal values (with same semantics) are:
  - `per_tensor`

`DATA_TYPE` specifies data type for precomputed reductions. Supported values are
`s32` (the default), `s8`, `u8`.

`GROUPS` specifies how precomputed reductions are grouped along dimensions where
multiple precomputed reduction factors are used.

To specify more than one memory argument for this attribute, `+` delimiter is
used.

## --attr-post-ops
`--attr-post-ops` defines post operations primitive attribute. Depending on
post operations kind, the syntax differs. To specify more than one post
operation, plus delimiter `+` is used.

Operations may be called in any order, e.g. apply `SUM` at first and then apply
`ELTWISE`, or vice versa - apply `ELTWISE` and then `SUM` it with destination.

### Sum
`SUM` post operation kind appends operation result to the output. It supports
optional arguments `SCALE` parsed as a real number, which scales the output
before appending the result, `ZERO_POINT` parsed as a integer, which shifts the
output before using the scale and `DATA_TYPE` argument which defines sum data
type parameter. If invalid or `undef` value of `DATA_TYPE` is specified, an
error will be returned.

### Eltwise
`ELTWISE` post operation kind applies one of supported element-wise algorithms
to the operation result and then stores it. It supports optional arguments
`ALPHA` and `BETA` parsed as real numbers. To specify `BETA`, `ALPHA` must be
specified. `SCALE` has same notation and semantics as for `SUM` kind, but
requires both `ALPHA` and `BETA` to be specified. `SCALE` is applicable only
when output tensor has integer data type.

`ELTWISE` supported values are:
- Eltwise operations that support no alpha or beta:
  - `abs`
  - `exp`
  - `exp_dst`
  - `gelu_erf`
  - `gelu_tanh`
  - `log`
  - `logistic`
  - `logistic_dst`
  - `mish`
  - `round`
  - `sqrt`
  - `sqrt_dst`
  - `square`
  - `tanh`
  - `tanh_dst`
- Eltwise operations that support only alpha:
  - `elu`
  - `elu_dst`
  - `relu`
  - `relu_dst`
  - `soft_relu`
  - `swish`
- Eltwise operations that support both alpha and beta:
  - `clip`
  - `clip_v2`
  - `clip_v2_dst`
  - `hardsigmoid`
  - `hardswish`
  - `linear`
  - `pow`

### Depthwise convolution
`DW:KkSsPp` post operation kind appends depthwise convolution with kernel size
of `k`, stride size of `s`, and left padding size of `p`.
These kinds are only applicable for 1x1 2D-spatial main convolution operation.
They support optional argument `DST_DT`, which defines destination
tensor data type. Refer to [data types](knobs_dt.md) for details.

### Binary
`BINARY` post operation kind applies one of the supported binary algorithms to
the operation result and then stores it. It requires a mandatory argument of
`DT` specifying a data type of the second memory operand. It supports an
optional argument of `MASK_INPUT` providing the information what the dimensions
for the second memory operand are. `MASK_INPUT` may be provided in two ways:
* An integer value encoding the mask. This is the way masks are reported in
  oneDNN verbose logs.
* A string literal value encoding the mask (see below). Possible options are:
  - `common`         corresponds to `mask = 0` and means a whole tensor will be
                     multiplied by a single `SCALE` value.
  - `per_oc`         corresponds to `mask = 1 << 1` and means elements of dim1
                     will be multiplied by scale factors different for each
                     point. Number of scale factors is equal to dims[1].
  - `per_tensor`     means each element, or a group elements expressed by
                     `GROUP`, of original tensor will be multiplied by a unique
                     scale value. The number of scale factors equals to
                     `nelems`, or `nelems / GROUP` for a grouped case.

In case the `MASK_INPUT` value affects more than one dimension (e.g.,
`per_tensor` input value or integer `15` input value), additional optional
argument `TAG` is supported, positioned after `MASK_INPUT`, to specify a
physical memory format. `TAG` values use same notation as in drivers. The
default value of the `TAG` option is `any`. Refer to [tags](knobs_tag.md) for
details.

`STRIDES` values use same notion as in drivers and intended to specify
non-trivial strides which is a rare case. It's a single dims-like tensor applied
to a specific binary post-op input. It's recommended to use this one only when
the verbose_converter generates it. Refer to [strides](knob_strides.md) for
details.

`BINARY` supported values are:
- `add`
- `div`
- `eq`
- `ge`
- `gt`
- `le`
- `lt`
- `max`
- `min`
- `mul`
- `ne`
- `sub`
- `select`

For the `select` algorithm, the operation also requires a third conditional
tensor to be specified. A different format is thus used for providing the
arguments for both the tensors:

```
--attr-post-ops=select:DT[.S1_MASK_INPUT[.S1_TAG]][:S2_MASK_INPUT[.S2_TAG]]
```

The arguments for each tensor are are delimited using `.`, with the arguments
for the third tensor positioned after that of the second and are separated by `:`.
The arguments are optional for the third tensor and the data type value for
the third tensor is fixed at `s8`.

`S2_MASK_INPUT` follows the same `MASK_INPUT` semantics as above and defines the
dimensions of the conditional (third) tensor, so it may be broadcast against the
destination: `common` (`mask = 0`) is a single-value condition applied to the
whole tensor, while unset (no `S2_MASK_INPUT`) spans the full destination shape.

### Prelu
`PRELU` post operation kind applies forward algorithm to the operations result
and then stores it. Weights `DT` is always implicitly f32.

`MASK_INPUT` has the same semantics and meaning as for binary `--attr-post-ops`.
Supported values are same.


## Examples:

Run a set of f32 forward convolutions without bias appending accumulation into
destination and perform relu on the output with scale set to 0.5:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_D \
               --attr-post-ops=sum+relu:0.5 --batch=shapes_tails
```

Run a 1D-spatial reorder problem with s8 input data and u8 output data in four
different physical memory layout combinations {ncw, ncw}, {ncw, nwc},
{nwc, ncw} and {nwc, nwc} applying source scale 2.5 for each source point:
``` sh
    ./benchdnn --reorder --sdt=s8 --ddt=u8 \
               --stag=ncw,nwc --dtag=ncw,nwc \
               --attr-scales=src:common:2.5 2x8x8
```

Run a binary problem with s8 input data and u8 output data in nc layout
applying scales to both inputs without any post operations:
``` sh
    ./benchdnn --binary --sdt=u8:s8 --ddt=u8 --stag=nc:nc \
               --attr-scales=src:common:1.5+src1:common:2.5 \
               100x100:100x100
```

Run a 1x1 convolution fused with depthwise convolution where destination scales
are set to 0.5 for 1x1 convolution and 1.5 for depthwise post-op followed by a
relu post-op. The final dst datatype after the fusion in the example below is
`s8`. The weights datatype is inferred as `s8`, `f32` and `bf16` for int8, f32
and bf16 convolutions respectively.
``` sh
  ./benchdnn --conv --cfg=u8s8u8 --attr-scales=dst:per_oc \
             --attr-post-ops=relu+dw:k3s1p1:s8+relu \
             ic16oc16ih4oh4kh1ph0
```

Run a convolution problem with binary post operation, first one adds single int
value to the destination memory, second one adds a tensor of same size with
`nhwc`-like physical memory layout with float values to the destination memory:
``` sh
  ./benchdnn --conv --attr-post-ops=add:s32:common,add:f32:per_tensor:axb \
             ic16oc16ih4oh4kh1ph0
```

Run a matmul problem with weight-only quantization where weights `bf16` scales and
`s8` zero points have scales along `ic` and `oc` dimensions, have groups along
`ic` dimension:
``` sh
  ./benchdnn --matmul --dt=f32:s8:f32 --attr-fpmath=bf16:true \
             --attr-scales=wei:per_ocic:bf16:2x1 \
             --attr-zero-points=wei:per_ocic:s8:2x1 6x4:4x5
```

Run a matmul problem with MXFP4 semantics:
``` sh
  ./benchdnn --matmul --dt=f4_e2m1 \
             --attr-scales=src:per_tensor:e8m0:1x32+wei:per_tensor:e8m0:32x1+dst:mx:e8m0:1x32 \
             32x32:32x64
```

Run a matmul problem with NVFP4 semantics:
``` sh
  ./benchdnn --matmul --dt=f4_e2m1 \
             --attr-scales=src:per_tensor:f8_e4m3:1x16+wei:per_tensor:f8_e4m3:16x1+dst:dynamic_fp:f8_e4m3:1x16 \
             32x32:32x64
```

Run a matmul problem with weights decompression using the `sbzp` quantization
mode, where `f16` weights scales and floating-point zero-points are applied as
`x = scale * q - zp` and grouped along the `K` dimension:
``` sh
  ./benchdnn --matmul --dt=f16:u4:f16 --attr-fpmath=f16:true \
             --attr-scales=wei:per_ocic:f16:32x1:sbzp \
             --attr-zero-points=wei:per_ocic:f16:32x1:sbzp \
             128x1024:1024x512
```
