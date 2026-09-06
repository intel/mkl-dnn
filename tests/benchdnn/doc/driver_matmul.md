# MatMul Driver

## Usage
``` sh
    ./benchdnn --matmul [benchdnn-knobs] [matmul-knobs] [matmul-desc] ...
```

where *matmul-knobs* are:

 - `--dt={f32:f32:f32 [default], ...}` -- source, weights and destination data
            types. Interface supports broadcasting, when a single input is
            provided, e.g., `--dt=f32`, and the value will be applied for all
            tensors. Refer to [data types](knobs_dt.md) for details.
 - `--stag={any [default], ...}` -- memory format of the source memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={any [default], ...}` -- memory format of the weights memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={any [default], ...}` -- memory format of the destination memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--strides=SRC_STRIDES:WEI_STRIDES:DST_STRIDES` -- physical memory layout
            specification for `src`, `weights`, and `dst` tensors through
            strides values. Refer to [option documentation](knob_strides.md)
            for details.
 - `--bia-dt={undef [default], f32, s32, s8, u8}` -- bias data type.
            To run MatMul without bias, use `undef` data type (default).
            Refer to [data types](knobs_dt.md) for details.
 - `--bia_mask=INT` -- a bit-mask that indicates which bias dimensions are
            broadcasted. 0-bit means broadcast, 1-bit means full dimension.
            Note: for grouped matmul (`--grouped`) only `bia_mask=2`
            (N-axis only) is supported. The resulting bias shape is `[G, N]`,
            i.e. one value per output column per expert.
 - `--runtime_dims_masks=[INT][:INT]` -- a bit-mask with values for `src` and
            `weights` that indicates whether a dimension is
            `DNNL_RUNTIME_DIM_VAL` (indicated as 1-bit in the corresponding
            dimension position). The default is `0` for all dimensions, meaning
            all tensor dimensions are fully defined at primitive creation. For
            tensors with option values other than `0`, a correspondent memory
            format tag must be specified.
 - `--encoding=STRING` - sparse encodings and sparsity. No encodings are set by
            default. Refer to [encodings](knobs_encoding.md) for details.
 - `--grouped=DIM_IDX:NUM_GROUPS:size0+size1+...+sizeN[:max_size]` -- specifies
            grouped encoding for a given dimension to execute grouped GEMM.
            `DIM_IDX` is the dimension index (`0` is `M`, `1` is `K`),
            `NUM_GROUPS` is the number of tensors in a group or number of experts,
            `size0+size1+...+sizeN` are `+`-separated sizes or resolved variable
            dimensions, and optional `max_size` is the max variable dim hint for
            optimal dispatch.
            When grouped encoding is specified, weights must be provided as 3D
            `[G,K,N]` where `G` matches `NUM_GROUPS`.
            Multiple configs can be comma-separated to iterate over them.
            Example: `--grouped=0:4:32+0+32+96 160x512:4x512x256`
            means `M` dimension is variable, `4` experts are used, `32, 0, ...`
            are sizes of each group with total `32+0+32+96 = 160`.
            Example: `--grouped=0:4:8+8+8+8:8` specifies max group size of `8`.
            Example: `--grouped=0:4:8+8+8+8,0:3:2+6+8` iterates over two configs.
 - `--grouped=DIM_IDX:NUM_GROUPS:PROFILE[.PARAM][:HINT]` -- alternative form
            that generates per-group sizes from a `PROFILE`. The total variable
            dimension is taken from the problem shape (e.g. `total_M`).
            Generation is deterministic. Supported profiles:
            - `balanced[.seed]`, tokens are distributed across all `NUM_GROUPS`
              like a balanced router. `seed` (default `0`) selects the draw.
            * `hot[.pct]`, one hot expert absorbs `pct`% of the tokens (default
              `50`; `pct=100` is absorb-all), the rest are spread evenly across.
              `pct` must be in `[1, 100]`.
            * `decode`, one token per active expert (`total_M` groups of size
              `1`, the rest of the groups are empty).
              Requires `total_M <= NUM_GROUPS`.
            Optional `PARAM` is attached to `PROFILE` with a `.`.
            Optional `HINT` is the max_variable_dim dispatch hint, if not
            provided, it defaults to the largest generated group.
            Example: `--grouped=0:64:balanced 512x512:64x512x256`
            Example: `--grouped=0:64:balanced:512 512x512:64x512x256`
            Example: `--grouped=0:32:hot.80 256x512:32x512x256`
            Example: `--grouped=0:16:decode 8x512:16x512x256`
            Note: a binary post-op with `mask = 0` (e.g.
            `--attr-post-ops=mul:f32:0`) is applied per group (`[G, 1]`,
            e.g. NVFP4 global scale), not a whole-tensor.
 - `--gpu-kernel=STR` -- GPU kernel override, passed via a primitive
            attribute. `N` runs the Nth candidate after sort (rank); `A-B` and
            `all` run ranks `A` through `B` or every rank, clamped to the
            problem's candidate count (a problem with an empty range is
            skipped); problems with no jit:gemm candidate count run without
            an override. Any other string is a strategy override; quote it (e.g.
            `--gpu-kernel="gemm ..."`) on the command line and in batch files;
            in batch files, a `\` line break inside the quotes joins the
            lines without a space, so put a space before `\` between words.
            Empty by default (no override). Requires `ONEDNN_DEV_MODE=ON`;
            ignored on non-GPU engines. `ONEDNN_VERBOSE=xe=info` prints the
            deployed kernel.
 - `--match=REGEX` -- skip problems not matching the regular expression in
            `REGEX`. By default no pattern is applied (run everything).
            Note: Windows may interpret only string arguments surrounded by
            double quotation marks.
 - Any attributes options. Refer to [attributes](knobs_attr.md) for details.

and *matmul-desc* is a problem descriptor. The canonical form is:
```
    d0xd1xd2x..xMxK:d0xd1xd2x..xKxN[:d0xd1xd2x..xMxN][nS]
```
Here `x` is the delimiter for dimensions within a tensor and `:` is the
delimiter for tensors in the order `src`, `weights`, and `dst`. The `dst` is
optional, and each of its individual dimensions are computed as
`max(src_dimension, weights_dimension)` by the driver if not provided by user.
`d0`, `d1`, `d2` and so on are dimension values of the corresponding tensor,
where `m`, `n`, and `k` are inner dimensions for matrix multiplication.

## Examples

Run the default validation set of MatMul using `inputs/matmul/shapes_2d`
file:
``` sh
    ./benchdnn --matmul --batch=inputs/matmul/shapes_2d
```

Run single precision matrix multiplication with all sizes provided at run-time
using plain `ab` format for all inputs:
``` sh
    ./benchdnn --matmul --stag=ab --wtag=ab --dtag=ab \
                        --runtime_dims_masks=3:3 10x30:30x20
```

Run reduced precision (int8) matrix multiplication with asymmetric quantization
for the source and destination memory (both use `uint8_t` data type) and
symmetric quantization for weights memory (with `int8_t` data type and allowing
the library to choose the proper memory format), with zero points provided at
runtime, but sizes specified at creation time:
``` sh
    ./benchdnn --matmul \
               --dt=u8:s8:u8 \
               --wtag=any \
               --attr-zero-points=src:common:1+dst:common:-2 \
               10x30:30x20
```

Run single precision batched matrix multiplication with bias, of which only the
full dimension is along the `n`-axis:
``` sh
    ./benchdnn --matmul --bia-dt=f32 --bia_mask=4 2x10x30:2x30x20
```

Run single precision batched matrix multiplication with strides so that `dst` tensor
has non-dense memory layout:
``` sh
    ./benchdnn --matmul --strides=8x4x1:24x6x1:21x7x1 3x2x4:3x4x6
```

or
``` sh
    # --dtag cannot be specified here
    ./benchdnn --matmul --stag=bax --wtag=abx --strides=::8x4x1 2x2x3:2x3x2
```

More examples with different driver options can be found at
inputs/matmul/test_\*.

## Experimental: paneled ("periodic") fast reference

The `--mode-modifier=A` option speeds up the native CPU reference used for GPU
correctness validation. When set, the driver fills the inputs so the result is
periodic along `M` and `N` (the `K` dimension is kept full), computes the
reference for a single representative `panel_m x panel_n` panel, and broadcasts it
across the whole output. This reduces the reference cost from `O(MB*M*N*K)` to
roughly `O(panel_m*panel_n*K)` while the GPU primitive still runs the full shape.

``` sh
    ./benchdnn --matmul --mode=C --mode-modifier=A --engine=gpu \
        --dt=f8_e4m3:f8_e4m3:f16 4096x4096:4096x4096
```

The environment variable `DNNL_BENCHDNN_MATMUL_PANEL_FILL=1` is kept as a
deprecated alias for the modifier.

Notes:
- It only accelerates the *native* reference path, and only when that path is
  actually used, i.e. when no fast CPU primitive reference is available for the
  problem (`--fast-ref` cannot build a CPU primitive for the config, e.g. `f8`,
  `f4`, `mx`, some weight-decompression cases) or when `--fast-ref=false` is
  passed. When a fast CPU primitive reference *is* available it runs the full
  shape with vectorized kernels and is generally faster than the paneled scalar
  native reference, so the modifier is a no-op in that case (it does not disable
  `--fast-ref`). This avoids regressing configs that already have a fast CPU
  reference.
- The panel width is at least 128 and is aligned to the destination scale/zero-point
  groups so no block/group quantization parameter straddles a panel boundary.
- The driver prints whether the periodic fill was selected (and the chosen
  `panel_m`/`panel_n`), or the reason it fell back to the normal reference, at
  verbose level >= 0.
- Periodic inputs can mask bugs in `M`/`N` address arithmetic, so this mode is
  intended for speeding up local validation of large shapes, not as a replacement
  for the default reference in CI. It is disabled by default.
- Unsupported configurations (dropout/stochastic rounding, sparse/grouped
  dimensions, and shapes too small to form a 128-wide panel) automatically fall
  back to the normal reference. Runtime dimensions, binary/prelu post-ops, and
  dynamic/derived destination scales (e.g. `dst:dynamic_fp:...`, `dst:mx:e8m0:...`)
  are supported: because the panel is a multiple of the destination-scale group,
  the per-block dynamic scales are themselves periodic and are broadcast across the
  full destination-scale tensor.
