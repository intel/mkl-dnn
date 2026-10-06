Generator Frontend
==================

gemmstone's frontend tool provides a command-line interface to the kernel generator. This document describes its syntax and functionality.


Frontend Variations
-------------------

Each frontend executable is built for a particular _output type_, which is encoded in the name of the executable.

Output Types:
* `frontend`: assembly text, for human reading/editing
* `ofrontend`: kernels (properly, _program binaries_) ready to be run with OpenCL/L0
* `bfrontend`: raw ISA binaries
* `efrontend`: zebin output, for newer drivers

Frontend will automatically detect your GPU and create binaries and assembly for it. To generate code/binaries for a different GPU, use the `GEN` environment variable with a value from the `type` column in gemmstone's README.

For most output types, there are associated Perl scripts (Linux only) that prepare the resulting kernels for use with gemmstone's benchmark, placing them in a temporary location where `benchstone` will find them. They take the same input arguments as the frontend does (described in the next section).

* `oprep.pl ...`: generate program binaries directly, via `ofrontend`.
* `eprep.pl ...`: generate zebin directly, via `efrontend`.
* `clprep.pl <source.cl>`: package an already-generated OpenCL `.cl` file (not a kernel description) so `benchstone` can find it.

On Windows the Perl scripts are not available, but `oprep.bat` and `eprep.bat` provide the same functionality as `oprep.pl` and `eprep.pl`.


Basic Command-Line Syntax
-------------------------

The frontend takes as arguments a compact text description of a kernel to be generated, and dumps the generated kernel (assembly text or binary) to stdout.

Here is the syntax for GEMM and other level-3 BLAS kernels:

<pre>
*frontend  <a href="#kernel-types"><i>kernel-type</i></a>
           <a href="#precisions"><i>data-types</i></a>
           <a href="#memory-layouts"><i>memory-layouts</i></a>
           <a href="#tile-sizes"><i>m-tile-size n-tile-size</i></a>
           [<a href="#alphabeta-values"><i>alpha beta</i></a>]
           [<a href="#strategy-parameters"><i>strategy...</i></a>]
</pre>

This text format is common to all gemmstone tools. Here is a basic example:

    frontend gemm SSS TNN 16 8 1 0 as1 as1 ab

which creates an SGEMM kernel with 16x8 tile size, alpha = 1, beta = 0, and the strategy parameters `as1 as1 ab`.

The kernel type, data types, memory layout, and alpha/beta parameters all determine the GEMM *problem* (`GEMMProblem` structure inside gemmstone), describing the problem the kernel should solve. The remaining parameters are the GEMM *strategy* (`GEMMStrategy` structure), describing how it should solve it.

By default, the frontend autodetects the machine's GPU architecture and generates kernels for it. Alternately, the architecture may be specified by putting the abbreviated architecture name in the GEN environment variable. See tools/device_utils.hpp for more information.

    GEN=12LP frontend gemm SSS NNN 32 12

The next sections cover the different parts of the kernel description in detail.

### Kernel Types

The basic kernel types are:

* `gemm`: GEMM kernel: $C \leftarrow \beta\cdot C + \alpha\cdot AB$.
* `gemml/gemmu`: SYRK/GEMMT kernel -- like GEMM, but only updating the lower/upper triangle of C.
* `gemmlh`/`gemmuh`: HERK/GEMMT kernel -- same as `gemml`/`gemmu` but also zeros out the imaginary parts of the diagonal of C.
* `trmm{ll,lu,rl,ru}`: copy-based TRMM kernel; packed A/B only. The last two letters select left/right side and lower/upper triangles; e.g. `trmmrl` for right side, lower triangle.
* `symm{ll,lu,rl,ru}`: SYMM kernel -- like GEMM, but A (left) or B (right) is a symmetric matrix, of which only the lower or upper triangle is stored in memory.
* `hemm{ll,lu,rl,ru}`: HEMM kernel -- similar to SYMM, but A/B are Hermitian symmetric.

### Kernel Type Modifiers

For some kernel types, *modifiers* may be added before the basic kernel type:

* `batchs`: strided batch kernel (constant stride between A/B/C matrices in each batch), e.g. `batchs gemm`.
* `batchn`: non-strided batch kernel (addresses for each A/B/C matrix stored in an array).
* `batchnv`: non-strided batch kernel, allowing alpha/beta to vary (MKL's span group batch API).

The following modifiers apply to GEMM only:

*Offsets (zero points):*

* `offseta`: add an offset to A. Computes $C \leftarrow \beta\cdot C + \alpha\cdot (A + A_{\text{offset}})B$. By default $A_{\text{offset}}$ is a scalar, but other options are possible:
  - `offseta[s]`: $A_{\text{offset}}$ is a scalar, passed by value (the explicit form of the default).
  - `offseta[p]`: $A_{\text{offset}}$ is a scalar, passed by pointer.
  - `offseta[v]`: $A_{\text{offset}}$ is a (broadcast) column vector.
  - `offseta[g<K>]`: Grouped quantization: $A_{\text{offset}}$ is an $m\times\frac kK$ matrix, with each entry applied to a 1xK chunk of $A$.
  - `offseta[g<M>x<K>]`: Grouped quantization: $A_{\text{offset}}$ is an $\frac mM\times\frac kK$ matrix, with each entry applied to an MxK chunk of $A$.
  - Uppercasing the offset kind (`offseta[P]`, `offseta[V]`, ...) marks the offset as precalculated, applied to A/B before the main loop.
  - The data type of the offset is selected automatically, but can be overridden by appending a comma and one of the [precision characters](#precisions), e.g. `offseta[p,O]`.
* `offsetb`: add an offset to B, similar to A. The same offset types are available.
* `offsetc`: add an offset to C at the end of the computation, after alpha/beta scaling. The offset may be scalar, a (broadcast) vector, or a matrix, and is runtime configurable. Computes $C \leftarrow \beta\cdot C + \alpha\cdot AB + C_{\text{offset}}$.

Multiple offset kinds can be combined: `offsetab`, `offseta[v]b[g32]c`, etc.

*Scales:*

Grouped scales use a very similar syntax:

* `scalea[g<K>]`: Grouped scales for A. Multiplies A by an $m\times\frac kK$ scaling matrix, where each scale is applied to a 1xK chunk of A.
* `scalea[g<M>x<K>]`: Grouped scales for A. Multiplies A by an $\frac mM\times\frac kK$ scaling matrix, where each scale is applied to an MxK chunk of A.
* `scaleb[g<K>]`, `scaleb[g<K>x<N>]`: Grouped scales for B.
* Data type overrides are allowed, e.g. `scaleb[g32,H]`.

Multiple scale kinds can be combined, as with offsets: e.g. `scalea[g32]b[g128]`.

Modifiers may appear in any order before the basic kernel type. Multiple offset or scale tokens may be used (for example, `scaleb[g32] offseta scalea[g32] gemm`).

Kernel catalog selectors store only the base kernel type, stripped of modifiers. If the kernel only supports a subset of modifier configurations, that information is contained in tags, which restrict the kernel to appropriate use-cases.

More information on offsets/scales in gemmstone can be found [here](algorithms/dequantization.md).

*Other modifiers:*
* `bias`: add an offset to C just after accumulation. Computes $C \leftarrow \beta\cdot C + \alpha\cdot(AB + \text{bias})$.
* `suma`: calculate and store the sum of each row of `A` in addition to the main GEMM calculation.
* `sumb`: calculate and store the sum of each column of `B` in addition to the main GEMM calculation.
* `hpl`: HPL "pack kernel." After updating C, stores a second copy of the final C matrix in another location.
* `u` prefix (e.g. `ugemm`): generate a GEMM *microkernel* -- a register-level building block taking raw pointers and tile origins, meant to be inlined into a larger fused kernel rather than launched standalone.

### Precisions

The `<precisions>` parameter contain the A, B, and C data types, concatenated in that order. For example, `HHS` means A and B have type `H` (half precision) and C has type `S` (single precision).

gemmstone supports the following precisions:

*Floating point:*

|          | Real | Complex |
| -------- | ---- | ------- |
| Double   | D    | Z       |
| Single   | S    | C       |
| tf32     | T    | U       |
| Half     | H    | J       |
| bfloat16 | B    | K       |
| fp8 E5M2 | Q    | *n/a*   |
| fp8 E4M3 | q    | *n/a*   |
| fp4 E2M1 | E    | *n/a*   |
| nf4      | N    | *n/a*   |
| E8M0     | X    | *n/a*   |


*Integer:*

|          | Signed | Unsigned |
| -------- | ------ | -------- |
| int32    | I      | i        |
| int16    | W      | w        |
| int8     | O      | o        |
| int4     | F      | f        |
| int2     | P      | p        |

By default, alpha/beta have the same type as C, but this can be overridden by adding the type for alpha/beta at the end: `OoIS` means A int8, B uint8, C int32, alpha/beta single precision.

In addition, special "multicomponent" types support accelerated SGEMM and CGEMM kernels, when doing type conversions (see below):

|          | Real | Complex |
| -------- | ---- | ------- |
| bf16x2   | Bx2  | Kx2     |
| bf16x3   | Bx3  | Kx3     |
| tf32x2   | Tx2  | Ux2     |


#### Type Conversions

In some cases, we'd like to convert A or B on-the-fly from its type in memory to some other data type prior to multiplication. Similarly, it's common to accumulate the C matrix in higher precision, then downconvert at the end of the computation. gemmstone's basic type conversion syntax is:

    [<old_type><new_type>]

For A/B, `<old_type>` is type of A/B in memory and `<new_type>` is the converted type that will be used in multiplication. For C, `<old_type>` is the type used for accumulating C, and `<new_type>` is the type in memory.

*Note:* gemmstone may transparently enable A/B upconversions when the GPU does not natively support computations with the given data types. For instance, Gen12LP doesn't support fp8, so fp8 A/B will be automatically upconverted to half or single precision.

*Examples:*

* `HH[SH]`: multiply half * half -> single and accumulate C in single precision. At the end of the computation, convert C to half precision.
* `[OW][OW]I`: convert int8 A/B to int16 before multiplying and accumulating in int32 precision.
* `[SBx2][SBx2]S`: emulated SGEMM: A/B are converted from single precision to sums of 2 bfloat16 values, then multiplied, with accumulation in single precision.

### Memory Layouts

Similar to precision, the `<memory-layouts>` parameter contains memory layouts for A/B/C, concatenated. In the most basic form, a memory layout is a single character, one of the following:

* `N`: column-major
* `T`: row-major
* `A`: packed column-major
* `B`: packed row-major

*Example"* `TBN` indicates A row-major, B packed row-major, and C column-major.

Often a kernel strategy requires a specific memory alignment for A/B. This is expressed by postfixing the layout with `@n`, where `n` is the required alignment. For example, `NN@8N` indicates B must be 8-byte-aligned (and all matrices are column-major).

Note that a matrix may be implicitly transposed by viewing column-major data as row-major, and vice versa. With this in mind, gemmstone has no separate transposition parameters; instead, A/B transposition is configured via the selection of memory layouts.

#### Packed Layout Parameters

The packed layouts (`A`/`B`) support additional parameters to create more complicated VNNI-like and tiled layouts as needed by some kernels. The general format is

    <layout>[<crosspack>][#<tile_rows>,<tile_cols>][%<pack_size>]

*Crosspack* creates VNNI-like kernels. A crosspack of *n* means that groups of *n* elements, adjacent in the k-dimension, are grouped together consecutively in memory. For `A`, these groups are ordered in column-major order; for `B` in row-major order.

*Tiling* divides the matrix into blocks of size `tile_rows` by `tile_cols`. Within the tile matrix entries are ordered as described by the layout and crosspack. Tiles are ordered in memory in column-major order (`A`) or row-major order (`B`).

*Pack size* (`%<pack_size>`) sets the number of elements in a packed row/column. If omitted, it defaults to the m/n unroll.

Any of A/B/C may be given separate external (in-memory) and internal (in-kernel) layouts by wrapping the two layout specs in brackets: `[<external><internal>]`. For example, `[N16A]` reads the matrix from memory as column-major with crosspack 16, but treats it internally as packed column-major.

The natural packed layouts for the A and B matrices depend on the instructions used to implement the matrix multiply:

|                           | A matrix   | B matrix |
| ------------------------- | ---------- | -------- |
| FMA                       | `A`        | `B`      |
| dp4a                      | `A4`       | `B4`     |
| dpas (ATS/DG2, f16/bf16)  | `A2#8,16`  | `B16`    |
| dpas (ATS/DG2, int8)      | `A4#8,32`  | `B32`    |
| dpas (ATS/DG2, int4)      | `A8#8,64`  | `B64`    |
| dpas (PVC-Xe3p, tf32)     | `A#16,8`   | `B8`     |
| dpas (PVC-Xe3p, f16/bf16) | `A2#16,16` | `B16`    |
| dpas (PVC-Xe3p, int8)     | `A4#16,32` | `B32`    |
| dpas (PVC-Xe3p, int4/fp4) | `A8#16,64` | `B64`    |

### Tile Sizes

In a GEMM-like kernel, each thread computes a fixed-size tile of C. These sizes are included in the kernel description.

### Alpha/Beta Values

Kernels may be hardcoded for specific, special values of alpha/beta, or support arbitrary values. Supported values for alpha/beta are:

* `-`, meaning arbitrary (runtime scalar passed by value)
* `0`
* `1`
* `-1` (alpha only)
* `@`: arbitrary runtime scalar passed by pointer (may be complex)
* `R`: arbitrary runtime scalar passed by pointer to a real value

If not specified, the default is alpha = 1, beta = 0 for most kernels, except that beta = 1 is used by default with global k-parallelized kernels.

Note that some kernels (copy-based TRMM) set beta automatically.

Strategy Parameters
-------------------

Following the initial kernel description, a list of _strategy parameters_ can be provided. An example strategy is

    ab4/2x2{sc} ab4+b4@8 ab ca1 wg 2x4x2 kr li nb 1x2

If these parameters are omitted, the frontend first checks if there is a _recipe_ (built-in strategy) for the parameters given. If that fails, a basic strategy is attempted.

The next sections describe the available strategy parameters.

### A/B/C Strategies

The first three "words" in the strategy are required and describe A/B/C memory access, in that order. Within each word, the characters are:

* *Memory model:* `a` (stateless), `s` (surface), `l` (SLM). The letters `c`/`m`/`r` select internal address models not normally written by hand.
* *Memory access type:* `b`/`p`/`s`/`u`/`m`/`v`/`t`/`c` (see [table below](#memory-access-types))
    - An unaligned-path access override may be given as `/x` immediately after the access letter (before the k chunk size), e.g. `m/b`: use block 2D normally, but block messages on the unaligned remainder path.
* *k chunk size:* (A/B only) number of rows (A) or columns (B) to load at once
    - Separate k chunk sizes can be chosen for the main loop and k remainder handling, separated by `/`.
* *Register tiling:* (optional) `#R,C` requests an R-by-C register tile for this matrix.
* *Multiple buffering:* (A/B only, optional) `x2`, `x3`, ...
* *Cache settings:* (optional, LSC only) e.g. `{cc}` (see [section below](#cache-settings))

An optional prefetch strategy may follow, starting with `+`, then the following:

* *Memory access type*
* *k chunk size* (A/B only)
* *k stride:* (A/B only, optional) `,N` where `N` is the stride (in the k dimension) between prefetches
* *Prefetch distance:* `@d` where d indicates the prefetch distance (in units of k)
    - Separate prefetch distances can be chosen for the main loop and k remainder handling, separated by `/` (0 for remainder handling indicates no prefetch)
* *Cache settings:* (optional, LSC only)

An optional L2/L3 prefetch strategy may follow, e.g. `am16+m16@32+m16@128`. If not performing an L1 prefetch, use two `+` signs: `am16++m16@128`. The L2/L3 prefetch accepts only an access type, k chunk size, optional `@d` distance, and cache settings -- it does not support the `,N` k stride or `/` masked distance available for the L1 prefetch.

#### Memory Access Types

| Character | Access Type       | Transposes matrix? | Description |
| --------- | ----------------- | ------------------ | ----------- |
| b         | Block             | no                 | Contiguous access, using block messages when available |
| p         | Pseudo-block      | no                 | Contiguous access, using scattered messages |
| s         | Scattered         | yes                | Transposing access, using scattered messages |
| u         | Channel Scattered | yes                | DWord transposing access, using surface scattered/cmask4 messages |
| m         | Block 2D          | no                 | Block 2D message (XeHPC+) |
| v         | Block 2D VNNI     | no                 | Block 2D message with VNNI transform (XeHPC+) |
| t         | Transposed Block 2D | yes              | Block 2D message with transposition (XeHPC+) |
| c         | Compressed Cache Line | no             | Fetch one element per cache line (prefetches only) |

**Capitalized** characters (e.g. `B` instead of `b`) mean that the new LSC messages, available starting with XeHPG, should be used. On XeHPC and later, the LSC messages are used by default even for lowercase characters.

Note that block 2D messages have alignment requirements, which vary by architecture and message type:
| Architecture | Type          | Alignment requirement |
| ------------ | ------------- | --------------------- |
| XeHPC        | nontransposed | 8 bytes               |
|              | transposed    | 4 bytes               |
| Xe2, Xe3     | all           | 16 bytes              |
| Xe3p         | all           | 4 bytes               |

On XeHPC, prefetch accesses only require 4-byte alignment (same as transposed accesses), regardless of the nontransposed requirement above.

If these requirements aren't met an exception will be thrown.

#### Cache Settings

LSC messages allow per-memory-access control on how memory is cached. Separate cacheability settings are allowed for L1 and L3. The cache control parameters are indicated with one character each for L1 and L3 (in that order).

*Available cache settings for reads:* `dd`, `uu`, `uc`, `cu`, `cc`, `su`, `sc`, `ic`.

*Available cache settings for writes:* `dd`, `uu`, `ub`, `tu`, `tb`, `su`, `sb`, `bb`.

| Character | Meaning               |
| --------- | --------------------- |
| d         | driver default        |
| u         | uncacheable           |
| c         | cacheable             |
| s         | streaming             |
| i         | invalidate-after-read |
| t         | write-through         |
| b         | write-back            |

For C access, separate cache controls can be provided for reads and writes, separated by a slash: `{cc/tb}`. Where used, atomic C accesses are always L1 uncacheable.

Xe3p adds a new L2 level to the cache hierarchy, and allows L1/L2/L3 caching to be controlled separately. Accordingly, for Xe3p kernels there are three cache characters, for L1/L2/L3, in that order.

*Note:* on pre-Xe3p hardware the write-through (`t`) and write-back (`b`) L1 modes are not separately realized -- when parsed, `t` behaves as cacheable (`c`) and `b` as invalidate-after-read (`i`). A distinct write-back (`b`) mode takes effect only on Xe3p, and only when L2 is uncacheable. The read/write character sets listed above are the canonical (round-trippable) settings; the parser is character-based and accepts additional combinations that map onto these modes.

### Strategy Parameters

Following the A/B/C strategies, the remaining strategy parameters may be in any order and all are optional. In the descriptions that follow, capital letters represent numerical arguments.

#### Workgroup Control

* `wg MxN`: set the number of threads per workgroup in the m/n dimensions.
* `wg MxNxK`: set the number of threads per workgroup in the m/n/k dimensions (when local k-parallelization is enabled).
* `wxN`: pad workgroup size, making it `N` times larger. The extra threads are "dummy threads" that perform no work, limiting GPU occupancy.

*Less commonly used:*

* `ff`: force the workgroup size to be fixed even if the kernel could support a flexible workgroup size.
* `ffk`: force a fixed workgroup size in the k dimension.

#### Kernel Dispatch and Tile Ordering

These settings control the order in which C tiles are traversed, both globally and within a workgroup.

* `mnk`: (default) workgroups are dispatched in column-major fashion, walking down one workgroup tile column at a time. The innermost dimension in kernel dispatch is the m dimension. Also affects individual threads' tiles are ordered within a workgroup, again ordering them in column-major fashion.
* `nmk`: similar to `mnk`, but with m/n swapped.
* `bo`: enable panel boustrophedon walk order for C tiles.
* `hi`: enable cache oblivious Hilbert-like walk order for C tiles, supporting both rectangular and triangular updates (`gemml` etc.).
* `nl`: enable nested linear walk order for C tiles.
* `li`: enable linear ordering without changing kernel dispatch order. Useful for strategies that require a linear ordering (e.g. `pt`/`fb`/`fp`).
* `pt`: enable persistent threading. The driver only dispatches enough workgroups to fill the GPU; kernels wrapped in an outer loop over C tiles. Used on PVC; not a good idea when oldest-first thread arbitration is active (i.e. XeHP/XeHPG).
* `cti`: enable C-interleaving of columns, with the interleave size chosen from the C data type. Only supported when the C matrix has the N layout. May require padding the number of workgroups to be dispatched.

*Less commonly used:*

* `fm`: (Gen12LP/XeHP/XeHPG; default) arrange local IDs so that EUs are fused in the m dimension. When using `dpasw` instructions, `fm` is required if C is column-major in registers.
* `fn`: (Gen12LP/XeHP/XeHPG) arrange local IDs so that EUs are fused in the n dimension. When using `dpasw` instructions, `fn` is required if C is row-major in registers.
* `ql`/`qlN`: reorder local IDs within a workgroup so the threads on an EU do not share A/B data. `qlN` (non-oneDNN builds only) applies the reorder only when the x dimension is at most `N`.
* `rm`/`rn`: flip tile traversal order in the m/n dimension (respectively), working from bottom to top or right to left. Used for better load-balancing for TRMM (longest-running threads dispatched first).
* `ym`/`yn`: "scramble" tile ordering in the m/n dimension (respectively), walking through these dimensions with a configurable stride. Can improve cold-TLB performance for some memory-bound kernels.
* `bmN`: set m block size. Indicates to the driver code that problems should be split into chunks of size roughly `N` in the `m` dimension. Each chunk is a separate kernel launch. This is a primitive form of cache blocking.
* `bnN`, `bkN`: set n/k block sizes.

#### k-Parallelization Control

* `kr`: local k-parallelization. The full k range is divided between threads into (roughly) equal contiguous chunks, and the partial results from each thread are reduced via SLM.
    - `ikr`: interleaved local k-parallelization. Like `kr` but instead of dividing the k range into contiguous chunks, it is divided into fixed-size chunks, interleaved in the k dimension.
        * `kiN`: set the chunk size for k-interleaving to `N`.
    - `akr`: adjustable local k-parallelization. Like `kr`, but dynamically reduces k-parallelization in order to fit the dispatch into a single wave.
        * `fg <fraction>`: modify the fill goal when using `akr`. Instead of attempting to fill the GPU, fill only the given fraction of the GPU, *e.g.* `fg 0.75`.
* `kb`: global k-parallelization, using atomic additions. Each thread handles a fixed-size subrange of the total k range.
    - `fb`: fuse beta scaling kernel into atomic update kernels (`kb`/`kv`) if beta != 1.
    - `afb`: alternate implementation of `fb` that is typically faster and also has some subtle requirements on HW workgroup dispatch.
        * `pkN`: set virtual k-padding amount to N.
    - `fp`: fuse post-ops (including C type conversions) into atomic update kernels.
        * `zt`: enable alternate fused post-ops implementation that relies on pre-zeroed temporary C memory.
        * `rx`: enable relaxed accumulation: the final reduction over k may be done with lower precision instead of the normal accumulation precision.
* `kv`: variable k-slicing (stream-k), using atomic addition. At the end of the computation, the remaining k ranges are divided between available workgroups.
    - *Note:* `kv` automatically enables `fb` and `fp` as needed, but it's usually a good idea to enable `afb` as well on PVC and later.

More details on global k-parallelization and stream-k are available [here](algorithms/stream_k.md).

#### FMA Controls

* `dot`: generate a dot-product-style kernel (parallelized in k dimension) instead of the default axpy-style kernel (parallelized in m or n dimensions).
    - `dotN`: set the dot product vector length to `N` (default: 1 register). In the inner loop, one vector is maintained per element of C.
* `kcN`: enable k-chaining with chain length `N`. k-chaining groups FMAs in the k dimension and use accumulator registers to store the intermediate results, for reduced GRF bandwidth and power.

*Note*: there is no special setting for enabling dp4a usage (int8 * int8 -> int32). dp4a will be used automatically for int8 kernels whenever supported by the hardware.

#### Systolic Array

* `sys`: (XeHP+) enable systolic array.
* `fs`: (XeHP/XeHPG) use special "fixed" systolic inner loops, available for 32x32 and 32x48 tile sizes.
    - `sc`: (XeHP/XeHPG) use split-copy inner loops, in which half of threads only copy data while the other half do computation.
* `dw`: (XeHP/XeHPG) use `dpasw` instructions (dpas with data shared between fused EU pairs).
* `af`: make dpas instructions into macro chains with `{Atomic}`. Each macro chain shares the same `src1` operand. Should be used by default for all systolic kernels.
* `xaf`: like `af` but forms even longer chains by also linking `dpas` instructions that have different `src1` operands. `st` is recommended when using `xaf`.
* `fx`: sequence dpas instructions in a boustrophedon ("snake") pattern for better src2 read suppression.
* `kcN`: enabling k-chaining with chain length `N` (see [above](#fma-controls)). While the systolic array cannot use accumulators, on Xe3p this option enables forwarding for use with 2xDPAS.
* `crN`: repack and type-convert C data every `N` rows in dynamic quantization kernels. If `N` = 0, then accumulate a whole tile before repacking. In this case, if `rc` is not specified, then repacks as infrequently as possible.
    - `rcN`: perform `N` outer products before repacking; implies `cr0`. Usually `cr0` should be used instead of directly modifying `rc`; this option is intended only for manual performance evaluation.
* `clN`: for bf16x{2,3}, tf32x2-emulated SGEMM, limit the number of C buffers to `N`, reducing accuracy in exchange for larger tile sizes and better performance. (Not available in oneDNN builds.)

#### SLM Copies for A/B

* `caN`, `cbN`, `cabN`, `cbaN`: enable SLM copies for A and/or B, using `N` SLM buffers (1 <= N <= 4). `cabN` sets A's buffer count and mirrors it to B; `cbaN` sets B's and mirrors it to A.
    - Multiple buffering for global A/B reads can be enabled by adding `xG` (G >= 2), e.g. `cab4x2`.
* `ksN`: set the k chunk size for SLM copies. During each SLM copy, threads will cooperatively copy `N` columns of A/rows of B.
* `ksN/R`: like `ksN`, but set a separate (smaller) k chunk size `R` during k remainder handling.

*Less commonly used:*

* `ta`: store A row-major in SLM, instead of the default column-major.
* `tb`: store B column-major in SLM, instead of the default row-major.
* `ni`: disable incremental SLM copies.

More information on SLM pipelines can be found [here](algorithms/pipelines.md).

#### Cooperative Operation Controls

These controls affect how threads work cooperatively when performing SLM copies and prefetches. By default threads split load/prefetch tiles in the k dimension.

* `sm`: divide A work among threads in the m dimension, instead of k.
* `sn`: divide B work among threads in the n dimension, instead of k.
* `ska`: merge A work for the entire workgroup, then split it in the k dimension among all workgroup threads.
* `skb`: same as `ska`, but for the B matrix.

#### k-Loop Timing and Synchronization

* `bN`: enable periodic barriers -- every k = `N` iterations, perform a full barrier to keep threads together.
* `sbN`: enable periodic split barriers every k = `N` iterations.
* `nb MxN`: (XeHPC+) enable named barrier usage for SLM copies/periodic barriers, using `M`/`N` named barriers in the m/n dimensions, respectively. `M` and or `N` may be 0 to disable named barrier usage in that dimension.
* `st`: intentionally stall the thread after loading A/B to promote thread switches.
* `ba`: load B first, then A (default is to load A first, then B).
* Thread arbitration policy overrides -- available on selected GPUs (XeHPC and later):
    - `of`: oldest-first thread arbitration.
    - `rr`: round-robin thread arbitration.
    - `rrs`: round-robin-on-stall thread arbitration.

#### A/B Alignment and Remainder Handling

* `pab`: assume A/B are padded in m/n dimensions and omit all m/n remainder handling for them. (**Note**: `pab` is only safe in production use for surface accesses. Out-of-bounds stateless accesses can cause page faults, leading to application hangs/aborts.)
* `np`: don't assume A/B are padded. This is the default.

* `dm`: enable double-masking for A/B. Double-masking allows the kernel generator to use two masks for each A/B load (one in the row dimension, one in the column dimension) to avoid out-of-bounds reads. If double-masking is disabled, the kernel generator may change access types and reduce A/B load sizes on the m/n remainder path instead, which can be slower, though it takes fewer registers.
* `lN`: add runtime check for A/B alignment and create separate paths, one for A/B aligned to `N` bytes, and one without extra alignment. Typically the A/B aligned path uses more efficient messages (e.g. block 2D, block, scattered DW).
* `l2d`: similar to `lN`, adding a runtime check that A/B has proper alignment for block 2D messages.

*Less common:*

* `kd`: (XeHPC and older) enable descriptor-based remainders for A/B k remainder handling. Indirect descriptors can vary the k chunk size for certain messages (channel scattered, LSC scattered).
* `wt`: enable TLB warmup before the main loop.

#### C Remainder Handling / C Update

* `sr`: standard C remainder handling. Reads/writes to C use a combination of jump tables, masking, conditionals, and variable send descriptors to completely unroll the C update.
* `ar`: alternate C remainder handling, using a simple double loop. Much less code than `sr` but somewhat slower. This is the only option supported currently for triangular update kernels (`gemml`/`gemmu`/`gemmlh`/`gemmuh`).
* `br`: (XeHPC+) block 2D based C remainder handling. At runtime the kernel will check if block 2D accesses can be used, and if so will dispatch a special path that uses block 2D messages to load/store C.
    - `bf`: add-on option for `br`: use block 2D loads/stores to C even for full tiles. This is the default when the C data type in memory is smaller than 4 bytes.

*Less common:*

* `pc`: omit all remainder handling for C, assuming it is padded. Intended for kernel debug.
* `au`: when beta = 1, use atomic updates for C; default for XeHPG+ when native atomic addition is available.
* `nau`: opposite of `au`.

#### Register Allocation

This group of options control how registers are allocated, to avoid performance loss from bank conflicts.

The terms "vectorized matrix" and "broadcast matrix" below refer to the A/B matrices. If C is column-major in registers, A is the vectorized matrix (because FMA vectorization happens in the m dimension) and B is the broadcast matrix; vice versa if C is row-major in registers.

* `cs`: C is stored in its own bundle(s); A/B are in other bundles, starting in opposite banks.
* `acb`: A allocated first, then C, then B.
* `bca`: B allocated first, then C, then A.
* `vnc`: Vectorized matrix first, then broadcast matrix, then C matrix (used for Gen12LP kernels).
* `int`: A/B interleaved in bundles, then C allocated (used for Gen12LP if A/B load chunks have the same number of elements).
* `nse`: Broadcast matrix is stored in its own bundle(s), while the other two matrices are stored in the remaining bundles, starting in opposite banks (used for XeHPG+ non-systolic kernels).
* `vav`: A and B are allocated, then C registers are allocated to avoid conflicts with A (used for XeHPG+ systolic kernels).

* `grfN`: use `N` GRFs. By default N is 128 (regular GRF mode). XeHP+ also support N = 256 (large GRF mode) for more register space, at the "cost" of half as many threads/EU. ("Cost" is in quotes as sometimes having fewer threads is helpful to reduce cache contention.) Xe3 introduces the VRT (Variable Registers per Thread) feature, which provides many additional options for GRF size.
* `tk`: use dedicated SWSB token allocation for k-loop loads/stores.
* `ra<arch>`: perform register allocation as if targeting a different GPU architecture, e.g. `raxehpc`. (Contrast the numeric `raN`, which sets the A repack k chunk size.)

#### Complex Data

* `xa`/`xb`/`xc`: configure load instructions so as to split A/B/C (respectively) into real and imaginary components, placing them in separate registers. Note that using `xc` halves the register space required for storing C, allowing larger tile sizes.
* `3m`: compute complex multiplication using 3 real multiplications and some extra additions instead of using 4 real multiplications.

#### Miscellaneous

* `kN`: set the k unroll to `N`. An optional `/R` (e.g. `k16/8`) sets a separate k unroll `R` for the masked remainder path.
* `kaN`: align k splits (k blocking, k-parallel/stream-k chunks, k padding) to a multiple of `N`, on top of any other required k alignment.
* `raN`, `rbN`: set k chunk size for A/B repacking.
* `ch`: check at runtime whether A/B are each contained a single 4GB page, and if so only do address arithmetic on the low 32 bits (enabled by default for Gen12LP/XeHPG/XeHPC).
* `nch`: opposite of `ch`.
* `cc`: copy C matrix before writing it out, even when not required.
* `sf`: add additional SLM fences that HW does not appear to need.
* `ws`: force WGs to be contained within a single subslice.
* `wc`: use wider SIMD for scattered C accesses, to avoid running out of flag registers.
* `njs`: generate four paths and four versions of inner loop for m/n remainder handling, checking separately whether m and n remainder handling need to be enabled. By default there are two paths, one allowing both m/n remainders and one with no remainders.
* `msN`, `nsN`: set the minimum m/n size below which split remainder handling is used (0 = always split).
* `up`: for packed A/B, use bounds-checking on panels instead of assuming panels are always in bounds. Used for oneDNN; in oneMKL, padding is always available.
* `pl`: convert all prefetches to regular loads. Useful in combination with [read checks](#read-check-kernels) to check for out-of-bounds prefetches.
* `ek`: prepare A/B layouts for k-masking even at the beginning of the loop (probably should be removed and not used any more).
* `cp`: chain prefetches together (sharing SWSB tokens) to reduce token pressure.
* `di`: delay incrementing A/B addresses after loads to avoid (short) stalls (default).
* `nq`: do not pad block 2D widths (may reduce performance).

Copy Kernels
------------

The level-3 copy kernels repack matrix data in global memory into a more efficient format so that A/B data needed by each thread is contiguous in memory and can be directly used by FMAs/dp4a/dpas.

The syntax for copy kernels is similar to compute kernels:

    *frontend <kernel type> <precisions> <memory layouts> <panel size> [<alpha> [<strategy>...]]

Examples:

    frontend copy DD NA 16 1
    frontend copy ST TB 32 - ab4 ab4 ly

### Kernel Types

| Type | Description         |
| ---- | ------------------- |
| `copy` | Copy general matrix |
| `copyls`/`copyus`| Copy symmetric matrix from lower/upper triangle |
| `copylh`/`copyuh`| Copy Hermitian matrix |
| `copy{l,u}t{n,u}` | Copy triangular matrix for TRMM, with nonunit (`n`) or unit (`u`) diagonal |

In addition, the `c` flag may be added to any of the above kernel types to apply complex conjugation, e.g. `copyltnsc`.

### Precisions

The precisions for the source and destination matrices are concatenated, e.g. `DD` for double -> double copy.

Additionally, some int8 GEMMs require copy kernels to calculate row sums (A) or column sums (B) and store them alongside the matrix data. This is done by appending `+<sum precision>`, e.g. `OO+I` indicates the kernel should copy int8->int8, and accumulate sums in int32 precision.

### Memory Layouts

Similar to compute kernels, layouts for source and destination are concatenated. The destination matrix must have a packed layout (crosspack and tiling are supported).

### Panel Size

This parameter indicates for A copies the height of the panel in rows (m tile size), and for B copies the width of the panel in columns (n tile size).

### Alpha

A scaling factor (alpha) can optionally be applied during copying.

### Strategy Parameters

As for compute kernels, strategy parameters are optional and will be chosen automatically if not provided by the user.

The first two words of the strategy are required, providing strategies for source and destination memory access, in that order, e.g. `ab4/2 ab4`. The syntax is similar to compute kernels:

* *Memory model:* `a` (stateless), `s` (surface)
* *Memory access type:* `b`/`p`/`s`/`u` (block 2D messages are not supported currently)
* *z chunk size:* chunk size in the direction of the copy loop (see `lx`/`ly` later). For x-loop copies, must be the same as the panel size.
    - Separate chunk sizes may be chosen for the main and remainder loops, separated by `/`.
* *Cache settings* (optional, LSC only)

The remaining strategy parameters are optional and can occur in any order:

* `ly`: (default) copy kernel loops in the y dimension (along the panel).
* `lx`: copy kernel loops in the x dimension (across panels).
* `yN`: set `y` unroll to `N`.
* `xN`: set `x` unroll to `N` (default: the panel size).
* `bN`: periodic barriers every `N` iterations to keep threads together.
* `grfN`: set GRF count to `N`.
* `zb`: enable 2D parallelization.
* `twgN`: set the target workgroup size to `N` (positional argument, given after `zb`).
* `ps`: disable bounds checking on source matrix S.
* `np`: enable bounds checking on source matrix to prevent out-of-bounds access.
* `dm`: enable double-masking, allowing the source matrix `S` to be masked in both dimensions.
* `aN`: add runtime check for source matrix alignment and create separate paths, one for S aligned to `N` bytes, and one without extra alignment.


Special Kernels for Debugging and Development
---------------------------------------------

The kernel generator (and frontend) has a number of options that can be helpful when debugging functionality issues, or for doing performance investigations.

### Read Check Kernels

Most kernel types support a *read check* mode that ensures that no illegal out-of-bounds reads are performed. Before executing a read check kernel, `benchstone` fills valid matrix entries with zeros, and adds nonzero padding values around the edges of the matrix. The read check kernel checks that all loaded data is zero, and writes out its result to the upper-left corner of its C tile.

To create a read check kernel, simply prefix the kernel description with `rc`, e.g.

    rc gemm SSS NNN 16 8 1 0

You can also restrict the read check to A, B, or C only with `rca`/`rcb`/`rcc`. Read check kernels can't check prefetches, but by adding the `pl` strategy flag, prefetches will be replaced with real loads, which are then checked. Note that the kernel may run out of registers, however.

### Mockup Kernels

When analyzing kernel performance, it can be very useful to omit certain parts of the kernel for testing. For instance, you can remove the C update step to get a sense for how much overhead it adds, or remove A/B loads to check the performance of the computation alone aside from memory effects.

The general syntax for making mockup kernels is:

    *frontend <kernel description> [--] mock <setting1> ... <settingN>

The `--` is needed whenever you specify strategy parameters, and marks the end of the strategy parameter list.

Examples:

    frontend gemm SSS NNN 16 8 1 0 ab1x2 as8 ab -- mock abi c
    frontend gemm SSS NNN 16 8 1 0 mock abi c

Here is a list of available mockup settings:

| Code | Description |
| ---- | ----------- |
| ab   | Remove A/B loads |
| abi  | Remove A/B address increments |
| ab0  | Keep A/B address increment instructions, but increment by 0 |
| abr  | Remove A/B loads, pre-filling A/B load buffers with random data (more power consumption) |
| abn  | Remove A/B loads and leave the load buffers uninitialized |
| c    | Remove C update |
| cr   | Remove C reads |
| cw   | Remove C writes |
| cn   | Omit C accumulator initialization |
| kkr  | Keep the local-k reduction (which `c` otherwise removes) |
| m    | Remove actual computation |
| b    | Remove barriers |
| sw   | Remove SLM writes |
| sn   | Initialize SLM to NaN |
| s    | Remove loads (copy kernels) |
| si   | Remove source address increments (copy kernels) |
| d    | Remove stores (copy kernels) |

### Kernel Progress Tracing

Kernels can be configured to output timing information on when various events happen in the kernel. Timing information is measured in ticks of the EU's timer.

The general syntax is very similar to mockup kernel syntax:

    *frontend <kernel description> [--] track <setting1> ... <settingN>

Again, the `--` is needed if there are any strategy parameters or mockup settings preceding the tracking settings.

Currently supported tracking settings:

| Code | Description |
| ---- | ----------- |
| se   | Track start/end times for each thread, and for each subthread when persistent threading is enabled |
| ad   | Track completion of C accumulation |
| all  | Track both start/end and C accumulation completion |
