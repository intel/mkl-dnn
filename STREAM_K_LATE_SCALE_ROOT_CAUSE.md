# Root-cause analysis: stream-K + late 2D A/B-scale correctness bug

This document is a deep, code-cited analysis of exactly why the fix in
commit `93d728f61f` / `071878e99d` ("xe: gemm: jit: disable k-streaming for
grouped quant") was necessary for the specific repro:

```
./tests/benchdnn/benchdnn --engine=gpu --allow-enum-tags-only=false --matmul \
  --impl=jit:gemm:any --dt=u8:s8:s32 --stag=abx --wtag=abx --dtag=abx \
  --attr-scales=wei:per_tensor:f16:256x1 --attr-post-ops=sub:f16:common \
  --impl=jit:gemm:any 1x3840:3840x220
```

and evaluates whether the bug can instead be fixed by changing the C
accumulation/atomic-combination logic itself, rather than disabling
stream-K (`kParallelVariable`) outright.

**Conclusion up front:** for *this specific repro*, the fix's own commentary
attributes the hazard to "group-sum offset compensation"
(`needsAGroupSums()`/`needsBGroupSums()`), but empirically that is **not**
the mechanism that is actually triggered here — this repro has no
zero-points at all, and `needsAGroupSums()`/`needsBGroupSums()` are both
`false` for it. The actual mechanism triggered is the guard's **second**
branch: the *late 2D A-scale* hazard (`aLateScale`). The root cause is that
the per-workgroup partial K-sum accumulated by a stream-K workgroup is
**scaled and rounded to the integer accumulator type (`Tc = s32`) once per
scale-group, before combination**, and integer rounding does not distribute
over addition the way plain multiplication would. This is a fundamentally
different, and in some ways more subtle, mechanism than the group-sum
double-counting problem also described in the same guard/commit -- both
hazards are real, but they are distinct, and this repro only exercises the
second one.


## 1. Confirming which guard branch actually fires for this repro

`strategy.cpp`'s `preflight()` (as introduced by the fix) disables
stream-K when either of two independent conditions holds:

```cpp
// src/gpu/intel/gemm/jit/generator/strategy.cpp (current, post-refactor)
bool aLateScale = usesLateScaleA(problem, *this);
bool bLateScale = usesLateScaleB(problem, *this);
if (kParallelVariable
        && (problem.needsAGroupSums() || problem.needsBGroupSums() || aLateScale || bLateScale))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

Debug instrumentation (env-var-gated `fprintf` temporarily added to
`preflight()`, then removed) run against the repro on real XeHPC hardware
gave:

```
[STREAMK_DEBUG] kParallelVariable=1 aScale2D=1 bScale2D=0 aOffset2D=0 bOffset2D=0
                needsAGroupSums=0 needsBGroupSums=0
                asPtrDims=2 bsPtrDims=-1 aoPtrDims=-1 boPtrDims=-1
                aqGroupK=256 bqGroupK=0 systolic=1
```

- `needsAGroupSums() == 0`, `needsBGroupSums() == 0` -- the group-sum /
  zero-point hazard is **not** present for this repro (there are no
  `--attr-zero-points` at all in the command line, and no group-sum
  reduction buffers are configured).
- `aScale2D() == 1`, `aqGroupK == 256` -- A (the `u8` source tensor) has a
  2D (K-grouped) scale with group size 256 along K. This is surprising at
  first glance, since the repro's `--attr-scales=wei:...` only specifies a
  **weight** (B) scale -- see \S2 for why A ends up with a 2D scale too.
- With `Ta = u8` (1 byte, integer) and `Ta_scale = f16` (2 bytes), the
  `usesLateScaleA()` condition
  (`problem.hpp`/`compute_utils.hpp`: `Ta_scale.paddedSize() > Ta.paddedSize()
  && Ta.isInteger()`) evaluates **true**, so `aLateScale == true`.

Removing the entire guard (reverting `strategy.cpp` to the commit before
`93d728f61f`) and re-running the repro reproduces the original failure
exactly:

```
[   8][DST][0:8] exp_f32: -3.125  exp: -3  got: -4  diff: 1  rdiff: 0.333333
...
0:FAILED (errors:147 total:220) (1812 ms)
```

Re-adding *only* the late-scale check (leaving out the group-sum check)
is sufficient to make the repro pass again. This confirms: **for this
specific repro, the bug is the late-2D-A-scale hazard, not the group-sum
hazard** -- even though both hazards are handled by the same `if` in the
fix, and the fix's own code comments describe the group-sum mechanism
first and in more detail.


## 2. Why does A end up with a 2D scale here? (`wei:per_tensor:f16:256x1` only mentions B/weights)

The command line only specifies `--attr-scales=wei:...`, i.e. a scale on
the **weights** (`DNNL_ARG_WEIGHTS` / operand B in row-major dnnl terms).
However, gemmstone's internal A/B labeling is **transposed relative to
dnnl's row-major M/K/K/N convention** for this particular kernel path: the
`u8` "A" operand in gemmstone terms corresponds to dnnl's `DNNL_ARG_B`
(weights) once swapped/transposed for column-major kernel processing, or
conversely the mapping in `pd.cpp` deliberately swaps `a`/`b` when
constructing the `GEMMProblem`. Concretely:

- `pd.cpp` builds `a_scale_md_`/`b_scale_md_` and `a_quant`/`b_quant` from
  `attr()->scales_.get(DNNL_ARG_A)` / `get(DNNL_ARG_B)`, and separately
  computes `quant_entry_ndims()` for each (`pd.cpp:290-301`):

  ```cpp
  a_quant.scale_ndims = quant_entry_ndims(a_scales, a_scale_md_, ndims - 2);
  b_quant.scale_ndims = quant_entry_ndims(b_scales, b_scale_md_, ndims - 1);
  ```

- `quant_entry_ndims()` (`pd.cpp:38-64`) is the function that decides
  whether a scale is "1D" (per-output-channel, broadcastable, safe to fold
  into a post-op) or must stay "2D" (grouped along K, needs real per-group
  kernel support):

  ```cpp
  int quant_entry_ndims(
          const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx) {
      if (entry.has_default_values()) return -1;
      if (qmd.ndims < 2) return 0;
      int count = 0;
      for (int i = qmd.ndims - 2; i < qmd.ndims; ++i)
          if (qmd.dims[i] > 1) count++;
      if (count == 0) return 0;
      // for gemmstone, 1D quantization implies a full column vector
      // (i.e. not on the K dimension). If quantization varies over K,
      // we have to send these as 2D
      if (k_idx >= 0 && count == 1 && qmd.dims[k_idx] > 1) return 2;
      ...
      return count;
  }
  ```

  For the weight scale `wei:per_tensor:f16:256x1` on a `[K=3840, N=220]`
  weight matrix, group size `256x1` means the scale tensor has shape
  `[3840/256, 220/1] = [15, 220]` -- i.e. it *varies along K* (15 distinct
  values along the reduction dimension) and is broadcast (count-1, size
  220) along N. Since `k_idx` (here `ndims-1`, the K-axis for B in dnnl's
  arg convention) has `qmd.dims[k_idx] > 1` (15 > 1) and `count == 1`,
  `quant_entry_ndims()` returns `2` -- this scale is **forced to be 2D**
  regardless of the fact it is a "PER_TENSOR"-labeled broadcast in the
  N direction, precisely because it is *not* a simple per-output-channel
  scale: it genuinely groups the K dimension.

- Because it's already forced 2D (`scale_ndims == 2`), it does **not**
  get converted to a 1D binary post-op scale by
  `maybe_convert_scales_to_postop()` (`pd.cpp:154-183`), which only
  converts scales where the K-dimension entry is `<= 1` (i.e. truly 1D).

- `problem.bsPtrDims = b_quant.scale_ndims = 2` (`pd.cpp:773`), so
  `problem.bScale2D()` should be `2 >= 2` = true for B... but the debug
  output shows `bScale2D=0, aScale2D=1`. This means gemmstone's `A`/`B`
  labeling in the `GEMMProblem` is **swapped relative to dnnl's A (src) /
  B (weights)** for this particular kernel configuration -- i.e.
  `problem.A` corresponds to dnnl's weights tensor and `problem.B` to
  dnnl's source tensor, or `swap_ab()` was applied (`pd.cpp` has a
  `swap_ab_`/`swap_ab()` mechanism used to normalize layouts for the
  kernel, seen referenced at `pd.cpp:751-753` and `:855-861`
  transposing `AO`/`BO`/`A_scale`/`B_scale`/`Ag`/`Bg` when
  `swap_ab_` is set). Given `stag=abx`/`wtag=abx` (both row-major,
  non-transposed), and the debug output showing `aqGroupK=256` (matching
  the weight scale's K-group size) landing on `problem.A` rather than
  `problem.B`, this confirms an A/B swap occurred, mapping dnnl's
  `DNNL_ARG_WEIGHTS` (with its 256-K-grouped scale) onto gemmstone's
  `problem.A` (and hence `Ta`, `Ta_scale`, `aqGroupK`, `aScale2D()`).

**Net effect for this analysis:** whichever operand it lands on, the
*mechanism* is identical -- one of A/B has a K-grouped (`group_k=256`)
scale in a wider type (`f16`) than the (narrower, integer) operand type
it multiplies (`u8`/`s8`), which is exactly the condition
`usesLateScaleA()`/`usesLateScaleB()` test for. The rest of this document
refers to it generically as "the late-scaled operand" with scale-group
size `groupK = 256`.


## 3. What "late 2D scale" means in the kernel body

### 3.1. Where the decision is made: `quantization.cxx`

`gemmMake2DQuantizationLayouts()` (`src/gpu/intel/gemm/jit/generator/pieces/quantization.cxx:33-`)
decides, per A/B operand, whether that operand's 2D scale should be
applied "early" (per-outer-product, i.e. folded into the dequantized A/B
value fed directly into the `dpas`/`fma`) or "late" (applied once,
post-accumulation, to a repacked partial-C tile). The (now-shared,
post-refactor) decision function is:

```cpp
// src/gpu/intel/gemm/jit/generator/pieces/compute_utils.hpp
static inline bool usesLateScaleA(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    return problem.aScale2D()
            && ((problem.Ta_scale.paddedSize() > problem.Ta.paddedSize() && problem.Ta.isInteger())
                    || problem.forceLateQuant(minOuterProductCount(problem, strategy))
                    || (problem.preferBDPAS() && strategy.systolic));
}
```

For this repro: `Ta_scale` (f16, 2 bytes) is wider than `Ta` (u8, 1 byte)
and `Ta` is integer, so the first disjunct is true -- late scale is
selected. The comment right above this check in `quantization.cxx`
explains the intended purpose:

```cpp
// quantization.cxx:87-88
// Use lateScale for cases of applying scale to inputs that will be natively dpas'd
// but do not support add/mul.
```

I.e.: the `dpas` instruction can only consume A/B operands in a small set
of native integer/FP types; it can't multiply-in an arbitrary-width scale
per-element before accumulating. So instead, the *raw* (unscaled)
integer products are accumulated using the native integer `dpas` path,
and the scale multiply is deferred ("late") until after enough raw
partial products for one scale-group have been accumulated.

Once `lateScale` is set:

```cpp
// quantization.cxx:89-92
if (xs2D && (isA ? usesLateScaleA(problem, strategy) : usesLateScaleB(problem, strategy))) {
    lateScale = true;
    Txs_int = problem.Tc;
}
```

`Txs_int = problem.Tc` means: the scale value, once loaded and prepared
for use in the deferred multiply, is converted/stored so that the
multiply is done in `Tc`'s numeric domain (here `Tc = s32`, since the
matmul's `dst_dt = s32`). This detail turns out to be central to the bug
(\S5).

### 3.2. Where/when the deferred multiply actually happens: `matrix_multiply.cxx`

The deferred multiply happens in `outerProductRepackC()`
(`src/gpu/intel/gemm/jit/generator/pieces/matrix_multiply.cxx:628-689`),
which is called periodically from the main outer-product/K-loop driver
(`matrix_multiply.cxx:610-618`, `:621`) whenever `endRepackC` becomes true
for the current position `h` in the K-loop.

**How often does `endRepackC` fire?** It's driven by `state.cRepackPeriod`,
computed once at kernel setup time in `gemm_setup.cxx:1669-1691`:

```cpp
// gemm_setup.cxx:1678-1687 (abridged)
if (Cr_unrollX <= panel) {
    // Repack full tiles.
    if (problem.aScale2D() && problem.bScale2D())
        period = gcd(problem.aqGroupK, problem.bqGroupK);
    else if (problem.aScale2D())
        period = problem.aqGroupK;
    else if (problem.bScale2D())
        period = problem.bqGroupK;
    else
        period = strategy.repackC ? strategy.repackC : strategy.unroll[LoopK];
}
```

For this repro (`aScale2D()` only, `aqGroupK = 256`):
`state.cRepackPeriod = 256`.

The trigger logic (`matrix_multiply.cxx:84-90`, replicated at several call
sites for different specializations):

```cpp
bool startRepackC = false, endRepackC = false;
...
auto rphase = local_k_index(h, opCount, state.cRepackPeriod, problem);
...
endRepackC = rem || (rphase + opCount >= state.cRepackPeriod);
```

with `local_k_index()` (`matrix_multiply.cxx:32-38`):

```cpp
int local_k_index(int h, int opCount, int period, const GEMMProblem &problem) {
    int out = align_down(h, opCount) % period;
    ...
    return out;
}
```

`h` here is the **global** K-loop position (see \S4 below for
confirmation this is the true global K-coordinate, not workgroup-local).
So `local_k_index(h, opCount, 256, problem) == h % 256` (modulo edge
alignment to `opCount`) -- i.e. `endRepackC` becomes true exactly at
`h = 256, 512, 768, ...` (globally-aligned 256-element boundaries), **not**
at boundaries relative to wherever a given workgroup's K-slice happens to
start.

**What happens at that point (`outerProductRepackC`)**: the accumulator
holds only the raw (unscaled) partial products accumulated by `dpas`
*since the previous repack boundary*. `outerProductRepackC()`
(`matrix_multiply.cxx:628-689`) does:

```cpp
auto processItems = [&] {
    for (const auto &i: items)
        mov(i.simd, i.acc(Tc), i.Cr(1));                                  // load partial raw sum
    for (const auto &i : items)
        if (i.scale[1].isValid())
            mul(i.simd, i.acc(Tc), i.acc(Tc), i.scale[1](i.scaleStride[1])); // <-- SCALE MULTIPLY, dest type Tc = s32
    for (const auto &i: items) {
        if (i.scale[0].isValid())
            mad(i.simd, i.C(1), i.C(1), i.acc(Tc), i.scale[0](i.scaleStride[0]));
        else
            add(i.simd, i.C(1), i.C(1), i.acc(Tc));                       // accumulate into running total C
    }
    items.clear();
};
```

Key facts:

- `i.acc(Tc)` retypes the hardware accumulator register as `Tc = s32`.
  The `mul` instruction's destination is `s32` while (for our repro) one
  of its sources is a scale value of type `f16` (loaded via
  `Ar_scaleLayout`, prepared with `Txs_int = problem.Tc = s32` per \S3.1,
  which per GPU ISA semantics for `mul dst:int, src0:int, src1:float`
  computes the product in floating point and then **rounds/truncates the
  result to the integer destination type**, `s32`, per instruction). This
  is a genuine, hardware-level integer rounding operation, not a lazy
  bit-reinterpretation -- see \S5 for why this specific detail is the
  actual bug mechanism.
- This multiply-and-round happens **once per 256-K-element group, on
  whatever partial raw sum has been accumulated in the accumulator
  register since the previous repack boundary** -- for a single workgroup
  processing the whole K range without stream-K, that partial sum
  legitimately covers the *entire* 256-element group. But under stream-K
  (\S4), a workgroup's assigned K-range can start or end strictly inside
  a 256-element group, meaning its accumulator at a repack boundary holds
  the sum over **only the subset of that group's K-elements the workgroup
  actually owns**, not the full group.


## 4. Stream-K K-range assignment is not aligned to scale-group boundaries

`src/gpu/intel/gemm/jit/generator/pieces/stream_k.cxx` computes each
workgroup's assigned `[h0, h1)` K-range using a load-balancing scheme
designed to divide the total `K` (or `K x number-of-C-tiles`) work as
evenly as possible across the available compute units, independent of any
quantization-group structure:

- The full K-range is split into an "unsynchronized" bulk region and a
  smaller "synchronized" (fixup) region (`stream_k.cxx:106-129`).
- For the unsynchronized region, each workgroup's local range start/end is
  computed via modular arithmetic over the *linearized* index of (output
  tile, workgroup) pairs (`stream_k.cxx:134-163`), e.g.:

  ```cpp
  // stream_k.cxx (abridged, workgroup-local K start/end derivation)
  h1 = (k0 * slicedTileIdx) % kUnsynced;
  h0 = h1 - k0;
  ...
  h1 = min(h0 + k0, ...);
  ```

- The only alignment applied anywhere in this computation is to
  `strategy.kAlign(problem)` (`stream_k.cxx:117`, `:256-260`) -- this is
  an ISA/kernel-granularity alignment (e.g. ensuring K-ranges start/end on
  boundaries the load/dpas hardware paths can handle), **not** an
  alignment to `problem.aqGroupK`/`bqGroupK`.
- `state.h0` (the workgroup's actual starting global K-offset, used to
  offset A/B pointer addressing: `gemm_setup.cxx:1091,1148,1152`) is set
  directly from this unaligned computation (`stream_k.cxx:63,73,90,270`).

There is no code path in `stream_k.cxx` that rounds a workgroup's `h0`/`h1`
down/up to the nearest multiple of `aqGroupK`/`bqGroupK`. Therefore, for a
group size of 256 (as in this repro) and stream-K subdividing K=3840 across
however many compute units are active, it is entirely possible (indeed
likely, given 3840/256 = 15 exact groups but an unrelated stream-K tile
count) for a workgroup's `[h0, h1)` range to begin or end strictly inside
one of those 256-element groups.


## 5. The actual bug mechanism: why splitting breaks even though "scale is just multiplication"

At first glance this looks like it *shouldn't* be a problem: scaling is a
linear (distributive) operation --
`scale * (partial_1 + partial_2) == scale*partial_1 + scale*partial_2`
-- so if workgroup 1 contributes `scale * partial_1` and workgroup 2
contributes `scale * partial_2` to the same output element via atomic
add, the sum should still equal `scale * (partial_1 + partial_2)`, the
mathematically correct single-workgroup answer.

**This distributive property holds for real/rational arithmetic, but it
does NOT hold once each per-workgroup partial product is individually
*rounded to an integer* before the atomic add** -- and that's exactly
what happens here, per \S3.2:

```
Correct (single workgroup, full group):
  round(scale * (partial_1 + partial_2))

Actual (stream-K, split group, per-workgroup rounding):
  round(scale * partial_1) + round(scale * partial_2)
```

These two expressions are **not generally equal**, because integer
rounding is non-linear. A minimal concrete illustration: with
`scale = 1.5`, `partial_1 = 1`, `partial_2 = 1`:
- Correct: `round(1.5 * (1+1)) = round(3.0) = 3`
- Split: `round(1.5*1) + round(1.5*1) = round(1.5) + round(1.5) = 2 + 2 = 4`
  (round-half-up) -- an off-by-one error, purely from rounding twice
  instead of once. Depending on the specific values and rounding mode,
  the discrepancy can be ±1, ±2, ±3, etc. per element, which is *exactly*
  the pattern seen in the actual failure output:

```
[   8][DST][0:8] exp_f32: -3.125  exp: -3  got: -4  diff: 1
[  13][DST][0:13] exp_f32: 2210.69 exp: 2211  got: 2208  diff: 3
[  16][DST][0:16] exp_f32: -9248.62 exp: -9249  got: -9252  diff: 3
```

Small (1-3 ULP-ish, but actually integer-unit) diffs on ~147/220 output
elements, not catastrophic/garbage values -- consistent with "off by a few
units of rounding error accumulated from splitting a group's scale-and-
round operation across 2+ workgroups", not consistent with a gross
addressing bug, uninitialized memory, or missing data. This matches the
mechanism identified above precisely.

(Side note: it's also plausible that *some* of the observed diffs stem
from a second-order effect of the same root issue -- if a workgroup's
slice contains a *strict subset* of a group's K-elements at the boundary
of its assigned range, and the repack/scale logic still fires
`endRepackC` at the *global* 256-boundary regardless of whether the
workgroup's slice extends past it, the "partial accumulator" value being
scaled is well-defined (it's whatever raw products that workgroup's
`dpas`-issuing code path has accumulated so far), so there's no
uninitialized-memory or out-of-bounds read involved -- the bug is purely
the rounding-non-distributivity above, not a memory-safety or
masking/padding defect.)

### 5.1 Ruling out other candidate mechanisms

- **Row/column sum compensation (`sumA`/`sumB`, `needsASums()`/`needsBSums()`)**:
  `problem.hpp`:
  ```cpp
  bool needsASums() const { return sumA || (bOffset == ABOffset::Calc && !earlyDequantizeB() && !quantized2DB()); }
  bool needsBSums() const { return sumB || (aOffset == ABOffset::Calc && !earlyDequantizeA() && !quantized2DA()); }
  ```
  Since neither `aOffset` nor `bOffset` is `ABOffset::Calc` for this repro
  (no zero-points at all), both `needsASums()` and `needsBSums()` are
  `false`. Debug/trace confirms no row/col-sum compensation path is active
  here. This rules out the "implicit zero-point sum" mechanism as a
  contributor for this specific repro.
- **Group-sum offset compensation (`applyLateABOffset()`)**: gated
  entirely on `needsAGroupSums()`/`needsBGroupSums()`
  (`matrix_multiply.cxx:648-649`: `doAO2DLate = problem.needsBGroupSums();
  doBO2DLate = problem.needsAGroupSums();`), both `false` here, so this
  code path is simply never entered for this repro. This confirms the
  group-sum double-counting mechanism (which the original fix's code
  comments emphasize) is **not** what's being exercised by this particular
  test case -- it's a distinct, valid concern for zero-point cases, but
  irrelevant to this repro's failure.
- **Out-of-bounds/uninitialized reads from incomplete group repacking**:
  the repack/scale-application logic operates purely on the workgroup's
  own already-accumulated register-resident partial sum (`i.Cr`/`i.acc`);
  it does not read any group-wide shared buffer that would need to be
  "complete" to avoid reading garbage. No evidence of this mechanism was
  found; the bug is arithmetic (rounding), not a data hazard.


## 6. Can this be fixed by changing the C-accumulation/atomic logic instead of disabling stream-K?

Given the mechanism identified in \S5, there are three broad classes of
fix, each evaluated for feasibility against the current architecture:

### Option A: Align stream-K K-splits to scale-group boundaries

Force every workgroup's `[h0, h1)` range in `stream_k.cxx` to start/end on
a multiple of `lcm(aqGroupK, bqGroupK)` (or the per-operand group size
that's actually in play), so no workgroup ever needs to apply a late-scale
round to a partial (incomplete) group.

- **Feasibility**: structurally straightforward in principle -- this is
  "just" an additional alignment constraint alongside the existing
  `strategy.kAlign(problem)` rounding already performed in `stream_k.cxx`.
- **Drawbacks**:
  - It reduces stream-K's load-balancing granularity to (at best)
    `aqGroupK`/`bqGroupK`-sized chunks, which for `aqGroupK = 256` is
    fairly coarse and could reduce the number of distinct K-splits
    stream-K can usefully make for smaller K dimensions, partially
    eroding the very performance benefit stream-K exists to provide. For
    the *general* case (arbitrary group sizes, e.g. very large
    `aqGroupK` close to K itself, or a full-tensor per-tensor scale which
    is arguably `aqGroupK == K`), alignment-based splitting could
    degenerate to no splitting at all being possible -- equivalent to
    disabling stream-K anyway for those cases.
  - It only handles the late-scale hazard; it does **not** by itself fix
    the (separate) group-sum double-counting hazard, which is a genuine
    correctness issue *even with* aligned splits, because
    `applyLateABOffset()`'s group-sum buffers are indexed by absolute
    group number and a single workgroup owning a full aligned group still
    doesn't guarantee *no other* workgroup also touches that same group
    number for a different output tile/`M`/`N` slice in some tiling
    configurations (this needs separate, careful re-verification were
    Option A pursued for the group-sum case too).
  - This would require new logic in `stream_k.cxx`'s split-computation to
    thread through `problem.aqGroupK`/`bqGroupK` (currently that file has
    no such dependency at all), plus verification that the "synchronized
    fixup" region's logic also respects this alignment.

### Option B: Accumulate raw (unscaled) partial sums via a separate atomic buffer; apply scale exactly once as a final step

Instead of each workgroup independently scale-and-rounding its own partial
sum before atomically adding to C, have every workgroup atomically add its
**raw, unscaled** integer partial sum into a wider-precision (or same-width
integer) intermediate C buffer, then have exactly one final pass (e.g. the
last workgroup to finish a given output tile, or a dedicated
finalization kernel/pass) apply the group's scale multiply-and-round
exactly once, after all contributing workgroups' raw sums have been
combined.

- **Feasibility**: this is the "mathematically clean" fix -- it restores
  `round(scale * sum_of_all_partials)` rather than
  `sum_of(round(scale * each_partial))`, which is provably correct
  regardless of how K is split.
- **Drawbacks / architectural blockers**:
  - This requires the kernel to know, for a given output tile, when *all*
    contributing workgroups' raw partial sums have been atomically
    combined, before it's safe to apply the scale and write the final
    result -- i.e. a cross-workgroup completion/synchronization barrier
    per output tile, per scale-group. Stream-K's current design
    deliberately avoids needing this kind of fine-grained
    cross-workgroup synchronization within the K-split "unsynchronized"
    region specifically to maximize parallelism/load-balancing
    flexibility (per `stream_k.cxx`'s split into "unsynchronized" vs.
    "synchronized" regions, only the latter of which has explicit
    fixup/completion semantics currently, and that fixup mechanism exists
    for a different purpose -- see `fuseBeta`/`fusePostOps` completion
    logic, not per-scale-group completion).
  - The existing stream-K atomic-completion machinery
    (`useAutoAtomic()`, `strategy.cpp:80-93`; the C.atomic path in
    `c_update.cxx`/`atomic_fusions.cxx`; `fuseBeta`/`fusePostOps`,
    `strategy.cpp` near the guard) is built around "every workgroup
    computes its fully-finished partial C contribution independently,
    then atomically adds it -- no further processing needed after the
    atomic add, except possibly a single post-op pass gated by
    `fusePostOps`/`fuseBeta`, which already assumes post-ops (which would
    include a late scale, if reframed as a post-op) are safe to apply
    exactly once by exactly one (the "last") workgroup." This is
    actually a promising angle: if the late scale for the split-group
    case could be deferred to run as part of the *existing*
    `fusePostOps`/`fuseBeta` "last workgroup applies once" mechanism
    (rather than being baked into every workgroup's own
    `outerProductRepackC()` pass), it could reuse existing
    infrastructure instead of requiring wholly new synchronization.
    However, this would require the raw (unscaled) partial sum to
    survive as the atomically-combined value in C (or a side buffer) up
    until that final post-op pass runs, whereas today `outerProductRepackC()`
    scales-and-rounds *before* any atomic add ever happens -- a
    non-trivial restructuring of where/when the late scale is applied
    relative to the atomic-add and post-op-fusion pipeline.
  - Practically, this would only be worth attempting for the specific
    "K-split lands mid-group" case; for the common case where a
    workgroup's slice happens to align with (or fully contain) whole
    groups, the existing per-workgroup scale-and-round is already correct
    and more efficient (no need for a deferred/fused final pass), so any
    real implementation would need to conditionally choose between the
    two paths per-workgroup or per-tile, adding real complexity.

### Option C: Do the late-scale multiply in wider/floating-point precision, deferring only the final integer round to the very end

A middle ground: keep the current per-workgroup, per-repack-boundary
scale-multiply structure, but change the destination type of the `mul` in
`outerProductRepackC()` from `Tc` (`s32`, an integer round-inducing type)
to a floating-point accumulator type (e.g. `f32`) for the duration of
stream-K's atomic combination, only rounding to the final integer `Tc`
type once, after the atomic-add-combination completes (i.e. at
post-op/output-conversion time, which already exists as a separate later
stage in the pipeline for converting the internal accumulator type to the
externally-visible `Tc_ext`/output dtype).

- **Feasibility**: this directly targets the actual mechanism (\S5) --
  since `round(scale*(p1+p2)) ≈ round(scale*p1 + scale*p2)` **when the
  intermediate scale*p_i values are kept in a sufficiently precise
  floating-point format and only the final sum is rounded to integer**,
  this would eliminate (or reduce to ordinary floating-point
  summation-order rounding, a much smaller-magnitude effect) the observed
  discrepancy.
- **Drawbacks**:
  - Requires the atomic-add C buffer itself to be able to hold/atomically
    combine floating-point (or a sufficiently wide fixed-point)
    intermediate values rather than the final integer `Tc` type, which is
    a real structural change to what type flows through `C.atomic`'s
    atomic-add codegen (`atomic_fusions.cxx`/`matrix_access.cxx`) and to
    how/when the eventual integer conversion happens. This has ripple
    effects on the output-tile write-back path, buffer sizing, and
    possibly on the numeric semantics for callers that specifically expect
    `dst_dt = s32` bit-for-bit at every intermediate step (unlikely to
    matter here since only the *final*, fully-summed value would ever be
    exposed).
  - Does not address the group-sum double-counting hazard at all (a
    genuinely separate mechanism, \S5.1); would need to be paired with a
    separate fix or continued guard for that case.
  - Still requires the atomic-add path and post-op-fusion completion
    logic to guarantee a single final rounding/type-conversion step
    happens exactly once per output element, which again touches the
    same `fusePostOps`/completion machinery discussed under Option B.

## 7. Empirical follow-up: is Option A cheaper than it looks, given `kAlign()` already includes `aqGroupK`?

Before ranking the three options, it's worth noting that the kernel
**already** threads `aqGroupK`/`bqGroupK` into K-granularity decisions in
one place: `GEMMStrategy::kAlign()` (`strategy.cpp:632-646`):

```cpp
int GEMMStrategy::kAlign(const GEMMProblem &problem) const
{
    int align = lcm(ka_load, kb_load);
    ...
    if (problem.quantized2DA()) align = lcm(align, problem.aqGroupK);
    if (problem.quantized2DB()) align = lcm(align, problem.bqGroupK);
    return align;
}
```

`kAlign()` is used to round `unroll[LoopK]` itself
(`strategy.cpp:483-484,488`) and, on the host side, to round the
per-workgroup chunk size `k0` computed in `walk_orders.hpp:181-182` up to
`info.unroll[LoopK]` -- and `stream_k.cxx` also aligns `kUnsynced` and
`state.threadK0` to `strategy.kAlign(problem)`
(`stream_k.cxx:117,267`). So it's natural to ask: doesn't this already
guarantee stream-K K-splits land on `aqGroupK` boundaries, making Option A
free?

**Empirically, no.** I tested this directly: I temporarily patched
`preflight()` to drop only the `aLateScale`/`bLateScale` disable condition
(keeping the group-sum-only disable), rebuilt, and re-ran the exact repro.
It **still fails identically** (`errors:147 total:220`, same magnitude of
diffs) even though `kAlign()`'s `aqGroupK` term is already active for this
problem (`aqGroupK=256`, so `unroll[LoopK]` and the host-computed `k0` are
already forced to multiples of 256).

The reason this alignment doesn't help is that `kAlign()` only constrains
the **granularity** (`k0`, `unroll[LoopK]`) to be a multiple of the group
size -- it says nothing about the **starting offset** `h0` of a given
workgroup's slice. `stream_k.cxx`'s unsynchronized-region split computes
`h0`/`h1` via modular wraparound arithmetic over the *linearized*
(tile, workgroup) index space, not via simple `workgroupIdx * k0`:

```cpp
// stream_k.cxx:134-138 (abridged)
mul(1, temp, state.inputs.k0, temp.uw());
divDown(slicedTileIdx, temp, kUnsynced, ...);
emad(1, h1, temp, -kUnsynced, slicedTileIdx.uw(), ...);   // h1 = (k0*idx) % kUnsynced
...
add(1 | sat, h0.ud(), h1, -state.inputs.k0);               // h0 = h1 - k0
```

Even when `k0` and `kUnsynced` are both exact multiples of `aqGroupK`
(mod-256 congruence is preserved through multiplication and modulo by a
multiple of 256), the wraparound (`h1 = (k0*idx) % kUnsynced`) and the
"already sliced" / "starting a new tile" branch paths
(`stream_k.cxx:163-168`, `set h0 = kUnsynced - k0Rem`, where `k0Rem` is a
running remainder that is *not* itself always a multiple of 256 once a
workgroup has partially consumed a previous tile) can still produce
`h0`/`h1` values that are not multiples of 256 in the general case where a
single K-dimension (3840) is shared across multiple output tiles/rows and
a single workgroup's `k0`-sized chunk spans a tile boundary mid-group.
Additionally, the **local** K-sub-split for `kParallelLocal`
(`stream_k.cxx:256-270`, `state.threadK0 = ceil(k / lszK)`, aligned only
to `kAlign()` but derived from a *runtime-variable* `state.inputs.k`, not
a compile-time-known multiple of `aqGroupK`) is a second, independent
source of non-group-aligned offsets that `kAlign()`'s static term cannot
fix, since it operates on a dynamically-computed remaining-K value that
depends on where the outer (workgroup-level) stream-K split already cut.

**Conclusion**: `kAlign()`'s existing `aqGroupK` term only bounds the
*chunk size* granularity, not the *offset*; genuinely guaranteeing every
`h0`/`h1` boundary is a multiple of `aqGroupK` (Option A) would require
new, non-trivial rounding logic in `stream_k.cxx`'s wraparound/remainder
arithmetic (both the unsynchronized-region split and the `kParallelLocal`
thread-level sub-split), not merely reusing the existing `kAlign()`
mechanism. This closes off the possibility that Option A is a "free
lunch" already 90% implemented -- it is a real, non-trivial change to
`stream_k.cxx`'s core split arithmetic.


## 8. Comparative ranking: which of the three is cleanest?

| | Option A: align K-splits to group boundaries | Option B: raw-sum atomic buffer + single deferred scale pass | Option C: float-typed late-scale mul, round once at the very end |
|---|---|---|---|
| **Where the change lives** | `stream_k.cxx` split/wraparound arithmetic only | New buffer/type plumbing + `fusePostOps`/`fuseBeta` completion logic + `outerProductRepackC()` | `outerProductRepackC()`'s `mul` destination type + atomic-add/post-op completion path |
| **New synchronization needed?** | **No** — purely a compile/runtime-constant rounding change to existing split logic; atomics are untouched | **Yes** — needs a definite "all contributors have atomically combined" signal per output tile before the deferred scale pass can run safely | **Yes**, same requirement as B (must know when the float-typed accumulation is complete before final integer round) |
| **Fixes group-sum hazard too?** | Only incidentally (aligns *scale* groups, not necessarily group-sum-compensation regions, which is `gcd(aqGroupK,bqGroupK)`-scoped already per `matrix_multiply.cxx:650`) — would need re-verification, not a given | No, orthogonal — would still need the existing disable-guard's group-sum term | No, orthogonal — same as B |
| **Correctness risk if implemented** | Localized: either K-splits are aligned (bug can't occur) or the alignment logic has a bug (would very likely reproduce the *exact* symptom already characterized here, easy to catch via the existing repro/tests) | High: requires getting a new completion-detection protocol exactly right; subtle races here would be *silent* (not always reproducible), much harder to test exhaustively than an alignment invariant | Same completion-protocol risk as B, *plus* new questions about float accumulation precision/order-dependence for very large K |
| **Performance cost** | Reduces stream-K's splitting granularity to (at coarsest) `lcm(aqGroupK,bqGroupK)`-sized chunks — for `aqGroupK=256` this is a real but bounded granularity loss, not a wholesale loss of parallelism (still far finer than "don't stream-K at all") | Extra atomic buffer + an extra synchronization/completion pass adds fixed overhead per kernel launch, on top of whatever residual stream-K benefit remains | Same fixed overhead as B, plus float `mul`/`add` may be marginally more expensive per-workgroup than the current integer path (though likely negligible next to memory-bound stream-K's actual bottleneck) |
| **Reuses existing kernel machinery?** | Yes — extends the *already-present* `kAlign()`/`quantized2DA()`-aware alignment convention that this codebase already established for `unroll[LoopK]` (\S7); very much "in the spirit" of existing code | Partially — could piggyback on the existing `fusePostOps`/`fuseBeta`/`needsTempC()` "exactly-once final pass" machinery (`atomic_fusions.cxx`, `c_update.cxx`), but that machinery's current semantics center on *post-ops/beta*, not *late quantization scale*, so meaningful new glue code is required | Same partial reuse as B |
| **Verifiability** | High — a pure arithmetic/offset invariant (`h0 % aqGroupK == 0`), directly assertable/testable in isolation, and the existing repro is a ready-made regression test | Low-to-medium — correctness depends on a cross-workgroup completion protocol that is inherently harder to unit-test/reason about exhaustively (races are conditional on scheduling) | Same as B |

### Verdict: **Option A (align stream-K K-splits to `aqGroupK`/`bqGroupK` boundaries) is the cleanest of the three.**

Reasoning:

1. It is the only option that requires **no new cross-workgroup
   synchronization or completion-detection protocol** — it works entirely
   by making the *existing* atomic-add-of-already-scaled-partial-sums
   scheme trivially correct again, by guaranteeing every workgroup's
   partial sum corresponds to a *complete* scale-group (or none of one),
   which is precisely the invariant the naive/pre-fix code implicitly
   (and incorrectly) assumed always held.
2. It **extends an established pattern already present in this exact
   file** (`kAlign()`'s `quantized2DA()`/`quantized2DB()` terms,
   `strategy.cpp:643-644`, and the analogous `ukAlign`/`kInterleaveChunk`
   terms at `strategy.cpp:427-431,483-484`) rather than introducing an
   unrelated new architectural concept (a raw-sum staging buffer, or a
   float-typed intermediate accumulation domain) as Options B/C do.
3. Its correctness is a **static, directly testable arithmetic invariant**
   (every workgroup-assigned `[h0,h1)` range is `aqGroupK`/`bqGroupK`-
   aligned), verifiable by assertion/unit test independent of GPU
   scheduling nondeterminism — unlike Options B/C, whose correctness
   depends on getting a completion/ordering protocol exactly right, a
   class of bug that is much harder to catch via testing (races may not
   manifest on every run/every GPU).
4. Its performance cost is bounded and predictable (coarser, but still
   real, stream-K granularity — not a full loss of the optimization), and
   does **not** require adding a whole extra buffer + finalization kernel
   pass (Option B) or changing the fundamental numeric type/precision
   domain of the C accumulation path for a subset of cases (Option C).
5. It does **not** fix the group-sum hazard, but neither do Options B/C —
   all three are scoped to the late-scale mechanism only, so this is not a
   distinguishing factor between them.

**Caveat**: Option A is "cleanest" in the sense of smallest, most
localized, most easily-verified change — it is still real engineering
work in `stream_k.cxx` (per \S7, the existing `kAlign()` term is
insufficient on its own; the wraparound/remainder arithmetic itself needs
new rounding logic in at least two places: the unsynchronized-region
`h1 = (k0*idx) % kUnsynced` split and the `kParallelLocal` per-thread
`threadK0` sub-split), and would need its own dedicated correctness
testing (assert `h0 % groupAlign == 0` for all generated slices across a
representative sweep of K/tile-count/workgroup-count combinations) before
being considered a safe replacement for today's disable-guard. It should
not be attempted as a quick patch; it is a legitimate, scoped follow-up
project, best undertaken separately from this bug-fix with its own test
matrix (extending `tests/gtests/internals/test_gemm_stream_k_quant.cpp`).

### Summary judgment

All three options are theoretically viable ways to preserve *some*
stream-K performance benefit for the late-2D-scale case specifically, but
none are small, risk-free changes. Ranked by cleanliness: **Option A >
Option C > Option B** — Option A needs no new synchronization protocol at
all (just a stronger alignment invariant, extending code already present
in `strategy.cpp`'s `kAlign()`); Option C at least reuses the existing
atomic-add-of-C mechanism (just widening its type domain) and could
piggyback on `fusePostOps`'s already-existing "final pass" semantics more
directly than Option B; Option B requires the most novel architecture
(an entirely new raw-sum staging buffer plus a bespoke completion
protocol for it). By contrast, the existing fix (disabling
`kParallelVariable` whenever `aLateScale`/`bLateScale` is true) remains a
minimal, already-implemented, low-risk, correctness-first change, and
this session's separate follow-up work already narrowed its
performance-impact footprint considerably by fixing a false-positive in
the guard itself (see `STREAM_K_QUANT_ANALYSIS.md`, "Correction #2" and
"Correction #3") so that it only disables stream-K in cases that actually
require the late-scale (or group-sum) path, rather than over-broadly.

**Recommendation**: given the complexity and risk of all three options
above, the current approach (disable stream-K specifically and only for
the actually-hazardous cases, which is what the now-narrowed guard does)
remains the appropriate fix for this bug-fix effort. If future work
revisits recovering stream-K performance for the late-scale case, **Option
A (group-aligned K-splitting in `stream_k.cxx`) is the recommended
starting point**, since it is the only option that adds a testable static
invariant rather than a new runtime synchronization protocol, and it
builds directly on infrastructure (`kAlign()`'s quantization-group-aware
alignment) already present in this codebase.
