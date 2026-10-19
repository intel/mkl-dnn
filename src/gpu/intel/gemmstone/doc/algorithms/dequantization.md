Dequantization-Related Algorithms
=================================

In deep learning, the weights matrix (A matrix in column-major GEMM terminology) is often reduced from its original precision to a smaller data type (_quantization_) to reduce memory usage ("compressed weights"). Sometimes a similar operation is performed on the activations matrix (B matrix), sometimes call "dynamic quantization."

_Dequantization_ is the reverse process of approximately recovering the original data. In addition to upconversion (converting back to a higher-precision type), dequantization usually involves applying user-provided _scales_ and possibly _zero points_, or _offsets_, as they are termed in MKL and gemmstone. Mathematically, the kernel must compute

```math
C = \big[(A \pm A_{\text{offset}}) \odot A_{\text{scale}}\big]\cdot\big[(B \pm B_{\text{offset}}) \odot B_{\text{scale}}\big].
```

Here $\odot$ refers to elementwise multiplication of matrices, while $\cdot$ refers to regular matrix multiplication.

> [!NOTE]
> In oneDNN, and deep learning frameworks in general, zero points are subtracted from the data. That is, the $\pm$ in the formula above is a "$`-`$". MKL's int8 GEMM APIs, however, use the offset convention, with "$`+`$".
>
> Also, there are some quantization schemes where scales are applied before offsets &mdash; this is not currently supported in gemmstone or oneDNN.

The offsets and scales may each come in several formats, depending on the quantization procedure:
* *common*: single value applied to all elements. That is, the offset/scale matrix is constant.
* *per-channel*: one value for each row of A or column of B.
* *grouped*: A/B is tiled into blocks ("groups") of a fixed size; one value for each block.
  - Note: gemmstone primarily supports k-grouping (that is, `1xk` (A) or `kx1` (B) group sizes), with $k\geq 16$. There is some limited support for other kinds of grouping &mdash; as used by e.g. quantized SDPA &mdash; but likely bugs/unimplemented functionality on this path.

The formats for offsets/scales, together with the data types involved, determines the choice of dequantization algorithms and strongly influences GEMM performance. Let's look into the algorithms available.

For simplicity and brevity, in the following sections, we will show A and B having the same quantization scheme, but both in practice and in gemmstone they may have independent quantization schemes.


Upconversion
------------

Upconversion is, in some sense, the simplest aspect of dequantization. It is performed during the repack step (see more on pipelines [here](pipelines.md)), and is generally performed by [`copyRegisters`](../../generator/pieces/copy.cxx) at the same time as any necessary layout changes/reorders. Copies are executed by a standalone component (shared with the oneDNN IR system), the [copy planner](../../generator/pieces/copy_plan.hpp), which contains many optimized data conversion sequences.

gemmstone recognizes the need for upconversion by means of "internal" and "external" A and B types within the `GEMMProblem` structure.
`GEMMProblem::{Ta,Tb}` hold the internal A/B types &mdash; that is, the data types used by the computational operation (FMA/dp4a/DPAS), while `GEMMProblem::{Ta,Tb}_ext` hold the external A/B types &mdash; the type of A/B in memory. gemmstone users (oneMKL/oneDNN) can choose internal types themselves to specify the need for upconversion, or [`GEMMProblem::autoTypeConversions`](../../include/gemmstone/problem.hpp) will automatically choose upconversions as needed based on hardware capabilities and the problem setup.

> [!WARNING]
> gemmstone users should be careful to apply automatic upconversions as early as possible, as many GEMMProblem/GEMMStrategy methods (e.g. [`GEMMProblem::needsASums`](../../include/gemmstone/problem.hpp)) depend on having the upconversions properly configured beforehand. This part of the gemmstone API is unfortunately a bit fragile.


Offsets/Zero Points
-------------------

gemmstone has three implementation paths for A/B offsets.

### Early Offsets

If the internal A data type has enough room to store $A\pm A_{\text{offset}}$, it's generally fastest the perform this calculation as a preprocessing step during repack, prior to computation ("early offsets"). 
This preprocessing is handled by [`gemmDequantizeAB`](../../generator/pieces/quantization.cxx).

> [!NOTE]
> There is some fudging here in the case of int4 quantized data with int8 offsets and int8 internal type. It's certainly possible that int4 $\pm$ int8 does not fit in 8 bits, but in actual usage the int8 offset always falls within a small range (in fact, it is usually the constant 8), so gemmstone cheats in order to get good performance. A priori, a large offset would indicate an extremely asymmetric data distribution, which is quite unlikely.

### Sum-Based Offsets

When offsets are not grouped, and scales (if present) are also not grouped, we can mathematically rearrange the GEMM operation so that the offset calculation occurs outside the main k loop. This is simplest to show when there is a common offset, and no scales. Letting $O$ denote an all-1s matrix of appropriate size,

```math
\begin{align}
C &= (A - A_{\text{offset}}) \cdot (B - B_{\text{offset}}) \\
  &= (A - a_{\text{offset}}O) \cdot (B - b_{\text{offset}}O) \\
  &= A\cdot B - a_{\text{offset}}(O\cdot B) - b_{\text{offset}}(A\cdot O) + a_{\text{offset}}b_{\text{offset}}k.
\end{align}
```

At first sight, we've exchanged one matrix multiplication for three ($A\cdot B$, $A\cdot O$, $O\cdot B$). However, $A\cdot O$ is constant within each row, and its $i^{\text{th}}$ row just contains the sum of the $i^{\text{th}}$ row of $A$. Similarly, $O\cdot B$ consists of column sums of $B$. If we can efficiently compute these row and column sums in the GEMM inner loop (or pre-compute them) this transformation will be worthwhile.

Precomputing row/column sums, where possible, is the best route. gemmstone supports the use of precomputed sums when A/B are pre-packed &mdash; the reasoning being that the packing routine (copy kernel) can easily compute sums while packing. In this case, precomputed sums should be placed in memory at the end of each packed panel, immediately prior to the following panel.

Row/column sums may also be computed on-the-fly inside the k loop; this is performed either by [`accumulateSum`](../../generator/pieces/row_column_sums.cxx) (the generic path), or as a faster special case, within [`outerProductSystolic`](../../generator/pieces/matrix_multiply.cxx). On this path we must reduce incoming A/B tiles from global memory and accumulate.

When the reduction axis (k) is strided in registers (A column-major, or B row-major) we have an efficient _vertical_ reduction, which is performed by FMA/dp4a/dpas as appropriate. The DPAS route is fused into `outerProductSystolic` for better register reuse.

When the reduction axis is contiguous in registers (A row-major/B column-major) we're faced with the dreaded _horizontal_ reduction case, where sums need to be performed horizontally across all elements in a register. This is extremely inefficient in current hardware, and gemmstone makes no effort to implement it inside the k loop. Instead, gemmstone merely adds the entire A/B tile to the sum accumulation registers. One potential optimization here is to reload the A/B tile _transposed_ (especially if it is coming from SLM) so as to convert the problem to a vertical reduction.

In SLM pipelines, each thread only needs to accumulate sums for the subtile it loads from global memory. After the k loop completes, computed row/column sums are shared across threads and reduced as needed, via SLM, in [`gemmFinalizeSums`](../../generator/pieces/row_column_sums.cxx).


### Late Offsets

When offsets are grouped, or grouped scales are present, and we are unable to use early offsets, what should we do?

Let's try to adapt the sum-based algorithm to the grouped case. We can't apply the trick there to the entire GEMM, because each k-group has its own offset, but we can break the GEMM up into k-groups and use the same transformation on each k-group. Let $g$ be the k group size, and let $`A_p=A[:,pg:(p+1)g-1]`$, $`B_p=B[pg:(p+1)g-1,:]`$ denote the $p^{\text{th}}$ k-group of $A$ and $B$. Also, let's temporarily assume that A offsets are constant across rows, and B offsets are constant across columns. This isn't necessary for the algorithm, but simplifies the notation.

```math
\begin{align}
C &= \sum_{p=0}^{k/g} \left[A_p\cdot B_p - a_{\text{offset},p}(O\cdot B_p) - b_{\text{offset},p}(A_p\cdot O) + a_{\text{offset},p}b_{\text{offset},p}g\right].
\end{align}
```

The first term in the sum is just the full matrix product $A\cdot B$. We can simplify the rest by further assuming that only one of A/B has grouped offsets; here we'll pick A:
```math
\begin{align}
C &= A\cdot B - \sum_{p=0}^{k/g} a_{\text{offset},p}(O\cdot B_p).
\end{align}
```

This new algorithm evidently needs $O\cdot B_p$, which are column sums from the $p^{\text{th}}$ k-group of $B$. These can be computed on-the-fly in the k loop, just as before. Unfortunately, for cases currently of interest these column sums always end up being horizontal reductions, and we can no longer amortize the cost of the reductions by pushing them outside the k loop.

We're stuck once again, so we ask the user to help us out by _precomputing_ the group sums $O\cdot B_p$ &mdash; and fortunately this turns out to be reasonable for the OpenVINO dynamic quantization scenarios where such situations arise. This precomputation approach is due to Andrey Guskov (thanks, Andrey!). Alternatively, it may be possible to reload B transposed and use a vertical reduction (as mentioned earlier) to avoid the need for precomputed group sums.

gemmstone refers to this approach as "late offsets" and the computation of the offset terms occurs in [`applyLateABOffsets`](../../generator/pieces/matrix_multiply.cxx). Late offsets can also be used for non-grouped quantization as long as the user-provided group sums are present. Currently, gemmstone does not support using late offsets for A and B at the same time, as it requires computation of the additional cross-term $`a_{\text{offset},p}b_{\text{offset},p}g`$.


### `GEMMProblem` Controls

Offsets are configured in `GEMMProblem` via these knobs:

* `aOffset`, `bOffset`: offset calculation type
  - `ABOffset::None`: no offsets
  - `ABOffset::Load`: offsets, using pre-calculated A row/B column sums
  - `ABOffset::Calc`: offsets, no pre-calculated row/column sums
* `aoPtrDims`, `boPtrDims`: dimensionality of offsets:
  - `-1`: common, passed by value as a kernel argument
  - `0`: common, loaded from memory
  - `1`: per-channel (constant along $k$)
  - `2`/`3`: grouped (3 indicates batching)
* `Tao`/`Tbo`: offset data type
* `AO`/`BO`: (grouped only) `MatrixAddressing` structures describing the offset matrix's memory layout.


Scales
------

### Common/Per-Channel

Common and per-channel scales factor through the GEMM operation, so they can be applied after the k loop, as a post-operation:

```math
C = \big[(A \pm A_{\text{offset}})\cdot(B \pm B_{\text{offset}})\big]\odot A_{\text{scale}}\odot B_{\text{scale}}.
```

Common scales, if known on the host side (always in MKL, an option in oneDNN), are merged into GEMM's alpha scaling parameter. Device-side common scales and per-channel scales are converted to equivalent binary post-ops in oneDNN.


### Grouped

There are two main algorithms for applying grouped scales in gemmstone.

#### Early Scaling
If the A/B computation data type is at least as large as the scale data type, scales are applied as a preprocessing step, prior to multiplication ("early scaling"). Early scaling happens as part of the repack step, and is performed by [`gemmDequantizeAB`](../../generator/pieces/quantization.cxx).

#### Late Scaling
If the scale range does not fit in the computation data type, then the only option is performing the scaling after multiplying the A/B tiles together. This requires a k group size divisible by the k granularity of the compute operation (`minOuterProductCount` in gemmstone code). Late scaling comes with significant overheads as the k group size decreases, especially when the scaling cannot be overlapped with compute. However, it is the only option for group-scaled dynamic quantization scenarios where computation is performed in int8 or smaller integer types.

In late scaling, the A/B tiles are accumulated into a temporary accumulator (Cr). At the end of each k group, scales are applied, and the temporary accumulator is added to the main accumulator. That is, we break the k loop into chunks starting at $h=0, g, 2g,\ldots$, where $g$ is the k group size, and for each chunk perform:
```math
\begin{align}
Cr &\leftarrow A[:,h:h+g-1]\cdot B[h:h+g-1,:]\\
C &\leftarrow C + Cr \odot A_{\text{scale}}\odot B_{\text{scale}}
\end{align}
```

gemmstone refers to the application of scales and accumulation to the main accumulator as _C repacking_, and it is performed by [`gemmOuterProductRepackC`](../../generator/pieces/matrix_multiply.cxx) as a subroutine inside the main computation (`outerProduct`).

Keeping a separate temporary accumulator `Cr` is a huge strain on register resources, and reduces the maximum C tile size that can be supported. To reduce register pressure, gemmstone supports an arrangement where `Cr` only holds a few row/column panels of the C tile at a time, and computation and repacking are interleaved. When the k group size is small, a partial Cr may be preferable as we can support larger tiles. When the k group size is large, though, partial Cr usage significantly increases computation, because C must be updated after every outer product block, instead of once per k group.

By default, gemmstone enables partial Cr, and the `crN` strategy parameter controls how many rows/columns Cr holds. Setting `cr0` disables partial Cr.


#### `GEMMProblem` Controls

Grouped scales are configured in `GEMMProblem` via knobs similar to offsets (but a little simpler):

* `asPtrDims`/`bsPtrDims`: dimensionality of scales, always 2 (or 3 if batched)
* `Ta_scale`/`Tb_scale`: scale data type
* `A_scale`/`B_scale`: `MatrixAddressing` structures describing the scale matrix's memory layout.
