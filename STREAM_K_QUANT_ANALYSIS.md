# Stream-K vs. grouped 2D quantization: analysis, regression, and fix

## Background

Commit `bdf4283987` ("xe: gemm: jit: disable k-streaming for grouped quant")
added the following guard to `GEMMStrategy::preflight()`
(`src/gpu/intel/gemm/jit/generator/strategy.cpp`):

```cpp
// Variable k-parallelization (stream-K) combines per-workgroup partial C
// contributions via an atomic add straight into memory once each workgroup's
// slice has been late-scale-dequantized and rounded to Tc. With grouped 2D
// scaling/offsets, several workgroups may independently round their own
// partial contribution to the same output element before those roundings are
// summed, whereas a single-workgroup-per-tile strategy rounds only once
// (after accumulating the full K reduction). This makes the result depend on
// how work happened to be split across workgroups and introduces incorrect,
// K-split-dependent error beyond ordinary quantization rounding. Disable
// stream-K for such problems so C is always produced by one final rounding.
if (kParallelVariable && (problem.quantized2DA() || problem.quantized2DB()))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

This was reported to have **extreme negative performance impact** in
production, because it unconditionally vetoes stream-K for *any* GEMM problem
with grouped 2D quantization (`quantized2DA()`/`quantized2DB()`), which is
also the dominant, hottest shape for weight-only-quantized LLM matmul (grouped
per-K scales on int4/int8 weights). The kernel-database selector
(`selector/kernel_evaluator.cpp:443-476,601-621`) actively prefers stream-K
(`kv` tag, see `strategy_parser.cpp:387-390`) for exactly these large-K,
narrow-M shapes, so vetoing it broadly removes a heavily-relied-upon fast
path.

## Root-cause re-analysis: what is `quantized2DA()`/`quantized2DB()` actually?

From `src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp:259-265`:

```cpp
bool aScale2D() const { return (asPtrDims >= 2); }
bool bScale2D() const { return (bsPtrDims >= 2); }
bool aOffset2D() const { return (aoPtrDims >= 2); }
bool bOffset2D() const { return (boPtrDims >= 2); }

bool quantized2DA() const { return hasGroupSumsB || aOffset2D() || aScale2D(); }
bool quantized2DB() const { return hasGroupSumsA || bOffset2D() || bScale2D(); }
```

`quantized2DA()`/`quantized2DB()` are **broad**: they fire for grouped
*scaling alone* (`aScale2D()`/`bScale2D()`), with no zero-point/offset
involved at all. This is precisely the common weight-only quantization case
(e.g. int4/int8 weights with per-group `f32`/`f8_e8m0` scales, symmetric
quantization, no zero-points) — the shape the perf regression report is about.

### Where the atomic C add and rounding actually happen

`src/gpu/intel/gemm/jit/generator/pieces/c_update.cxx:1924,1949`:

```cpp
convert(Cacc, state.Tacc, Tc_ext, strategy, state);
...
atomicAddMatrixBlock(Tc_ext, Cacc, block, problem.C, strategy.C, header[0], problem, strategy, state);
```

The conversion to `Tc_ext` before the atomic add only introduces a rounding
step if `Tc_ext` is narrower than the internal accumulator type
(`state.Tacc`, generally `f32`). For the common case `Tc_ext == f32` (full
accumulation, `Ts == f32`), there is **no extra rounding step at all** — the
atomic add simply sums exact partial `f32` products. So the guard's stated
rationale ("several workgroups may independently round their own partial
contribution... before those roundings are summed") does not describe an
actual defect for scale-only grouped quantization with `f32` accumulation:
scales are applied multiplicatively to each workgroup's own K-slice
contribution with no shared/cross-workgroup state, so this is
mathematically equivalent (up to ordinary floating-point summation-order
effects already inherent to any parallel reduction, quantized or not) to the
non-quantized case.

### The actual hazard: group-sum offset compensation

Grouped-2D quantization becomes newly hazardous under stream-K specifically
when **zero-point/offset compensation** is required, which is a strictly
narrower condition captured by:

```cpp
// problem.hpp:299-300
bool needsAGroupSums() const { return (bOffset == ABOffset::Calc && quantized2DB() && !earlyDequantizableOffset(Tb_ext, Tbo, Tb)); }
bool needsBGroupSums() const { return (aOffset == ABOffset::Calc && quantized2DA() && !earlyDequantizableOffset(Ta_ext, Tao, Ta)); }
```

When this is true, the generator precomputes whole-quantization-group offset
compensation sums (`Ag`/`Bg`) and applies them **once per quantization
group** via `applyLateABOffset()`
(`src/gpu/intel/gemm/jit/generator/pieces/matrix_multiply.cxx:789-850`,
also referenced from `k_loop.cxx:1086-1094`):

```cpp
// matrix_multiply.cxx:789
void Generator<hw>::applyLateABOffset(bool isA, int h, ...)
{
    ...
    int haq = (h / problem.aqGroupK) % state.kaqLate;
    int hbq = (h / problem.bqGroupK) % state.kbqLate;
    ...
    A = isA ? state.Ar_offsetLayout.find(i / problem.aqGroupM, haq, ...)
            : state.Agr_layout.find(i / problem.aqGroupM, haq, ...);
    ...
}
```

These `Ag`/`Bg` group-sum terms are indexed **only by quantization-group
number**, not by which workgroup's K-slice happens to touch that group. In
stream-K, a K-split boundary is not guaranteed to align with a quantization
group boundary (`aqGroupK`/`bqGroupK`), so **more than one workgroup can
independently apply the full offset-compensation term for the same shared
quantization group** to the same output tile via their separate atomic C
adds — a genuine double-counting bug, not a rounding-precision nuance. This
is the real, narrower hazard.

Confirmed via `k_loop.cxx:91-96`:
```cpp
bool ao2DLate = ao2D && problem.needsBGroupSums();
bool bo2DLate = bo2D && problem.needsAGroupSums();
bool ag2DLate = problem.needsAGroupSums();
bool bg2DLate = problem.needsBGroupSums();
```
— the whole late-group-sum machinery is gated on `needsAGroupSums()` /
`needsBGroupSums()`, never on `quantized2DA()`/`quantized2DB()` alone.

## Conclusion

The original guard's condition (`quantized2DA() || quantized2DB()`) is a
significant overshoot: it disables stream-K for the *entire* grouped
quantization space, including the scale-only case that has no correctness
issue, when the actual unsafe condition is the narrower
`needsAGroupSums() || needsBGroupSums()` (requires zero-point offsets **and**
grouped quantization **and** a non-early-dequantizable offset type).

## Fix applied

`src/gpu/intel/gemm/jit/generator/strategy.cpp`:

```cpp
if (kParallelVariable && (problem.needsAGroupSums() || problem.needsBGroupSums()))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

This restores stream-K for the dominant scale-only grouped-quantization
shapes (the LLM weight-only-quantized GEMM hot path) while still disabling it
for the narrower, genuinely unsafe case of grouped quantization combined with
zero-point/offset group-sum compensation.

## Test coverage

`tests/gtests/internals/test_gemm_stream_k_quant.cpp` is a whitebox GTest
that calls `GEMMStrategy::preflight()` directly (see "Test infrastructure"
below for how this is built/linked). It covers:

- `StreamKQuantTest.ScalarQuantizationDoesNotDisableStreamK` /
  `GroupedQuantizationCombinations/StreamKQuantTest.ScaleOnlyGroupedQuantizationKeepsStreamK/*`
  — **regression tests for the perf fix**: grouped/scalar quantization with
  no zero-point offsets must leave `kParallelVariable`/`kParallelLocal`/
  `C.atomic`/`CO.atomic` enabled (A-only, B-only, both, and no-grouping
  cases).
- `GroupSumCombinations/StreamKGroupSumTest.DisablesStreamKForGroupSumOffsetCompensation/*`
  — **correctness regression tests**: whenever `needsAGroupSums()`/
  `needsBGroupSums()` is true (A-only, B-only, both), stream-K and the C
  atomics must still be disabled.
- `StreamKQuantTest.DisablesWithoutBatching` — confirms the group-sum guard
  fires independently of the pre-existing batching guard.

All 9 test cases pass with the fix in place. To confirm the tests are a real
regression guard (not tautological), the guard was manually reverted to the
pre-fix `bdf4283987^` version and re-run: 6/9 failed as expected (the 3
`DisablesWithoutBatching`/`StreamKGroupSumTest` cases that require some
disabling logic, plus related assertions), then the fix was restored and all
9 passed again.

### Test infrastructure notes (for future maintenance)

`GEMMStrategy::preflight()` is an internal, hidden-visibility (`libdnnl.so`
is built with `-fvisibility=internal`) symbol, so a normal gtest linking only
against the public `dnnl` shared library cannot call it directly. The test
executable (`test_internals_gemm_stream_k_quant`, registered in
`tests/gtests/internals/CMakeLists.txt`) works around this by linking against
the *same* full set of internal object files used to build `libdnnl.so`
itself (`src/CMakeFiles/dnnl.dir/objects1.rsp`, absolutized and re-emitted as
`dnnl_objects_abs.rsp` at configure time), rather than hand-picking a subset
of object libraries — hand-picking led to an unbounded chase of undefined
references, since the internal implementation is one interconnected whole.
This requires the main `dnnl` target to have been built at least once before
configuring (so `objects1.rsp` exists).

## Build/verification log (for reference)

```
$ ./tests/gtests/internals/test_internals_gemm_stream_k_quant --gtest_color=yes
[==========] Running 9 tests from 3 test suites.
...
[==========] 9 tests from 3 test suites ran. (11 ms total)
[  PASSED  ] 9 tests.
```

Ran on real Intel(R) Data Center GPU Max 1100 (XeHPC/PVC) hardware present in
this environment (confirmed via `clinfo -l`), though `preflight()` itself is
pure host-side strategy logic and does not require GPU execution.

## Follow-up investigation: can the guard be narrowed further?

After landing the `needsAGroupSums()||needsBGroupSums()` fix above, a
follow-up question was raised: since compile-time code already forces
stream-K's K-split alignment to be a multiple of `aqGroupK`/`bqGroupK`
whenever *any* grouped quantization is present (`ukAlign` computation in
`strategy.cpp` ~line 453-458, and `kAlign()` ~line 601-616, both gated on the
broad `quantized2DA()`/`quantized2DB()` predicate, not the narrower
group-sum predicate), is it possible that K-split boundaries are *always*
group-aligned in practice — meaning even the `needsAGroupSums()`/
`needsBGroupSums()` cases are actually safe, and the guard could be removed
or narrowed further?

### Static analysis

Tracing the alignment machinery through to its runtime consumers
(`stream_k.cxx`'s `alignUp(kUnsynced, ...)` at line ~117 and
`alignUp(state.threadK0, ...)` at line ~267, both driven by
`strategy.kAlign(problem)`) suggests the invariant *does* hold along the
normal code path. However, several edge cases could not be conclusively
ruled out via static review alone: a ragged final-K remainder when `fullK`
is not itself a multiple of the group size, the `kPadding`/virtual-padding
path, the interaction between `alignUp()` and the subsequent
`min_(h1, h1, state.fullK)` clamp (which could in principle produce an
unaligned final boundary), and whether a single workgroup's own K-loop could
internally re-apply a group's compensation more than once. Given that this
touches logic the original author was specifically cautious about (see the
related `Tacc_min` atomic-accumulation restriction elsewhere in the
codebase), static analysis was judged insufficient on its own to justify
removing/relaxing a correctness guard.

### Empirical verification attempt

Per explicit request, a further round of *empirical*, GPU-hardware-driven
verification was carried out (in addition to this document's existing
whitebox unit tests) before deciding whether to act on the static-analysis
result. This used:

- A temporary, non-production env-var bypass (`DNNL_STREAMK_ALLOW_GROUPSUM`)
  added to `preflight()` purely for local A/B testing, allowing stream-K to
  stay enabled even when `needsAGroupSums()`/`needsBGroupSums()` is true, so
  its numerical output could be diffed against the known-safe
  (guard-enabled) baseline on real hardware.
- A temporary debug print (`DNNL_STREAMK_DEBUG`) reporting the exact
  predicate values (`needsAGroupSums`, `needsBGroupSums`, `aOffset`,
  `bOffset`, `quantized2DA/B`, the relevant `T*_ext`/`T*o`/`T*` bit-widths,
  and `earlyDequantizableOffset(...)`) chosen by `preflight()` for each
  compiled kernel, to find real problems that actually reach the
  group-sum-required branch.
- Rebuilding with `DNNL_DEV_MODE=ON` (temporarily) to enable the
  `GEMM_KERNEL` strategy-string override mechanism in `gen_kernel.cpp`, so
  a specific "kv" (stream-K) kernel-DB strategy string could be forced onto
  a chosen problem for controlled comparison independent of the kernel
  selector's heuristics.
- Both direct manual testing and a dedicated background sub-agent were used
  to explore many `benchdnn --matmul` dtype/shape/grouping combinations on
  the real Intel(R) Data Center GPU Max 1100 (XeHPC) hardware present in
  this environment, specifically targeting shapes/dtypes expected to select
  stream-K (large K, small M, matching "kv"-tagged kernel-DB entries) with
  grouped (2D) zero-points (i.e., genuinely needing group-sum
  compensation), as opposed to the much more common grouped-scale-only or
  scalar/common-zero-point weight-only-quantization case.

**Result: no real, end-to-end `benchdnn`-runnable GPU matmul case was found
where `needsAGroupSums()||needsBGroupSums()` is true *and* stream-K
(`kParallelVariable`) is actually selected.** Every stream-K-selected case
tested (plain f16/bf16, int8 with common/per-tensor zero-points and scales,
etc.) had `needsAGroupSums=0 needsBGroupSums=0`. Every attempt to construct
a genuine grouped/2D zero-point case either failed to parse in `benchdnn`
(`policy2mask`/group-count errors) or reached primitive creation but
returned `UNIMPLEMENTED` on this build/hardware. This is consistent with the
root-cause finding above: `needsAGroupSums()`/`needsBGroupSums()` require
`earlyDequantizableOffset(...)` to be *false*, but essentially all
practically-reachable weight-only-quantization dtype combinations (e.g. u4
weights with s8/u4 grouped zero-points accumulated in f16, or s8 weights
with s8 grouped zero-points upconverted via bf16 `fpmath`) have a
zero-point storage type narrower than the accumulation type, which makes
`earlyDequantizableOffset(...)` true and routes them through the
already-safe early-dequantization path instead — never onto the late
group-sum-compensation path this guard protects.

Full commands, debug output and per-case results are recorded in
`STREAMK_EMPIRICAL_FINDINGS.md` (repo root).

### Conclusion

The narrowed guard (`needsAGroupSums()||needsBGroupSums()`) already in
place is being kept as the final fix, **unchanged**. The empirical work
found the guard's triggering condition to be reachable only via exotic
kernel/dtype combinations (e.g. native low-precision `dpas` GEMM paths on
newer hardware generations not present in this environment) that could not
be reproduced end-to-end here, so:

- there is no evidence the guard causes any further practical performance
  regression (it simply doesn't fire for realistic benchdnn/LLM-style
  weight-only-quantization workloads, which all take the
  early-dequantization path and keep stream-K enabled after the first
  narrowing), and
- there is no empirical evidence — positive or negative — regarding the
  numerical safety of *removing* the guard for the true group-sum case,
  since no real GPU repro of that case could be constructed on this
  hardware/build to test it against.

Given the asymmetric risk (a confirmed, explainable double-counting hazard
vs. an unconfirmed, currently-unreachable-in-practice performance benefit),
the guard is left in place as implemented. All temporary instrumentation
(`DNNL_STREAMK_ALLOW_GROUPSUM`, `DNNL_STREAMK_DEBUG`, and the associated
`<cstdio>`/`<cstdlib>` includes) added to `strategy.cpp` for this
investigation has been reverted; the file is back to the clean
`needsAGroupSums()||needsBGroupSums()` guard described earlier in this
document, and `DNNL_DEV_MODE` has been restored to `OFF` in the local build.
The whitebox test suite (`test_internals_gemm_stream_k_quant`, 9/9 tests)
was re-verified passing against this final, clean state.

## Correction (post-narrowing regression): a second, unguarded hazard

**The conclusion above was incomplete.** After the guard was narrowed to
`needsAGroupSums()||needsBGroupSums()` and committed (`b4aef2355a`), a real
benchdnn repro was reported as failing again on real XeHPC hardware, with
**no zero-points at all**:

```
./tests/benchdnn/benchdnn --engine=gpu --allow-enum-tags-only=false --matmul \
  --impl=jit:gemm:any --dt=u8:s8:s32 --stag=abx --wtag=abx --dtag=abx \
  --attr-scales=wei:per_tensor:f16:256x1 --attr-post-ops=sub:f16:common \
  --impl=jit:gemm:any 1x3840:3840x220
```

Confirmed failing: 140-147/220 mismatches, diffs of 1-3 ULP-scale, at
K=3840 (15 full 256-element scale groups); passes at K=256/K=512 (1-2
groups). `needsAGroupSums()`/`needsBGroupSums()` are both false here since
no `--attr-zero-points` is used — the narrowed guard never disables
stream-K for this case, yet it still produces wrong numerics.

### Root cause: late-applied 2D scales (`state.lateScale2DA/B`)

`gemmMake2DQuantizationLayouts()` (`generator/pieces/quantization.cxx`,
~line 88) sets `state.lateScale2DA`/`lateScale2DB = true` (deferring the
scale multiply to post-accumulation, on the *repacked* C tile, in
`outerProductRepackC()`, `generator/pieces/matrix_multiply.cxx` ~line 637)
whenever:

```
xs2D && ((Txs.paddedSize() > Tx.paddedSize() && Tx.isInteger())
         || problem.forceLateQuant(minOuterProductCount(problem, strategy))
         || state.useBDPAS)
```

In the failing repro, B (weights) is `s8` (1 byte) and its scale is `f16`
(2 bytes): `Txs.paddedSize() > Tx.paddedSize() && Tx.isInteger()` is true,
so `lateScale2DB = true`. This condition is **entirely independent of**
`needsAGroupSums()`/`needsBGroupSums()` (which only cover zero-point
group-sum compensation) — it fires for a pure grouped-*scale* case with no
offsets whatsoever, exactly matching the failing repro and exactly the kind
of case the narrowed guard's code comment claimed was safe ("scale-only
grouped quantization ... is safe ... since scales are applied
multiplicatively to each workgroup's own partial product with no
cross-workgroup shared state"). That claim holds only for *early*-applied
scales; when the scale is applied *late*, indexed by K-group across the
repacked accumulator, a stream-K K-split boundary falling inside a scale
group causes the same class of hazard as the late group-sum offset case
(each contributing workgroup's partial atomic-summed contribution reflects
an unscaled/partially-scaled slice of the group before the deferred
group-scale multiply, which the plain additive atomic combination cannot
correct for after the fact).

### Fix

Extended the `preflight()` guard in `strategy.cpp` to also disable
stream-K when a grouped 2D scale will be applied late, mirroring
`gemmMake2DQuantizationLayouts()`'s own late-scale decision:

```cpp
bool aLateScale = problem.aScale2D()
        && ((problem.Ta_scale.paddedSize() > problem.Ta.paddedSize() && problem.Ta.isInteger())
                || problem.forceLateQuant(minOuterProductCount(problem, *this))
                || (problem.preferBDPAS() && systolic));
bool bLateScale = problem.bScale2D()
        && ((problem.Tb_scale.paddedSize() > problem.Tb.paddedSize() && problem.Tb.isInteger())
                || problem.forceLateQuant(minOuterProductCount(problem, *this))
                || (problem.preferBDPAS() && systolic));
if (kParallelVariable
        && (problem.needsAGroupSums() || problem.needsBGroupSums() || aLateScale || bLateScale))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

(`state.useBDPAS` itself isn't computed until later in `preflight()`, so
its own gating condition, `problem.preferBDPAS() && strategy.systolic`, is
inlined here instead of reading `state.useBDPAS` directly.)

### Verification

- User's originally-failing repro: **PASSED** (previously 140-147/220
  mismatches).
- Same shape family at K=256, K=512 (fewer scale groups): **PASSED**
  (unaffected either way, but re-verified after the fix).
- Original group-sum fix scenario (`f16:s8:f16`, `per_ocic` grouped scale +
  grouped zero-points, `bf16` fpmath): **PASSED**, confirming
  `needsAGroupSums()/needsBGroupSums()` still independently disables
  stream-K where required and the new `aLateScale`/`bLateScale` condition
  doesn't regress that case (bf16 scale/zero-point here triggers early
  dequantization, not late-scale, since `Tb_scale` there is not wider than
  `Tb` in the paddedSize+isInteger sense once fpmath upconversion applies).
- All whitebox `preflight()`-only tests (`needsAGroupSums()`/
  `needsBGroupSums()` group-sum cases and the scale-only-keeps-stream-K
  cases) still pass, since those use `bf16`/`bf16` A/B types where
  `Tx.isInteger()` is false and the new condition never fires for them.

### Known gap

The whitebox test suite (`test_gemm_stream_k_quant.cpp`) only exercises
`GEMMStrategy::preflight()` in isolation; it does not build/link against a
full GPU-capable `dnnl` binary in this checkout (attempting to wire it into
`tests/gtests/internals`'s `test_internals` executable hit unresolved
external symbols and duplicate-symbol conflicts from `dnnl`'s internal
object libraries, which was not resolved in this session due to time
constraints). It therefore **cannot** catch late-scale-application bugs
like this one, which only manifest in actual GPU kernel execution. A new
`StreamKLateScaleTest` was added to the whitebox suite to lock in the new
guard condition at the `preflight()` level, but the true regression test
for this specific bug class is the benchdnn repro above, which should be
added to a permanent regression list (e.g. CI benchdnn inputs) rather than
relying solely on whitebox coverage.

## Correction #2: guard fired spuriously for weight-only-quant (no zero-points), causing a ~7x perf regression

After the late-scale guard above was added, a new severe performance
regression was reported for a common weight-only-quantization (WOQ) shape
with **no zero-points at all**:

```
./tests/benchdnn/benchdnn --engine=gpu --allow-enum-tags-only=false --matmul \
  --impl=jit:gemm:any --dt=f16:u4:f16 --stag=ab --wtag=ba --dtag=ab \
  --attr-scales=wei:3:f16:128x1 --attr-fpmath=f16:true 33x11264:11264x1024
```

Measured: **0.171ms with the guard vs. 0.023ms with the guard fully
removed — a ~7.4x slowdown**, all on a correct (`PASSED`) result either way,
i.e. the guard was disabling stream-K in a case where it was never actually
unsafe.

### Root cause: `preflight()` reads pre-`autoTypeConversions()` types

`GEMMProblem::autoTypeConversions()` normalizes `Ta`/`Tb` for weight-only
quantization, e.g. converting an integer weight type (`u4`) to match the
other operand's FP type (`f16`) when there is no group-sum offset
compensation needed:

```cpp
if ((Tb.isInt8() || Tb.isInt4()) && Ta.isFP() && Tc.isFP()) Tb = Ta;
```

The real kernel body (`gemm.cxx`'s `Generator<hw>::gemm()`) calls this
**unconditionally** before running `gemmMake2DQuantizationLayouts()`
(`quantization.cxx`), which is what actually decides
`state.lateScale2DA/B`. But `pd.cpp`/`gen_kernel.cpp` only call
`problem.autoTypeConversions()` *before* `strategy.preflight()`
conditionally — only when `needsAGroupSums()||needsBGroupSums()` is true.
For the failing repro (u4 weights, f16 scale, no zero-points), that
condition is false, so `preflight()` saw the **raw, unconverted** `Ta = u4`
(or `Tb = u4`, depending on A/B swap heuristics) instead of the `f16` type
the real kernel ends up using. My added guard's condition
(`Tx.isInteger() && Txs.paddedSize() > Tx.paddedSize()`) then spuriously
matched (`u4` "isInteger" and narrower than the `f16` scale), disabling
stream-K even though the actual kernel — after its own unconditional
`autoTypeConversions()` call — sees `Tx = f16` (not integer) and never
takes the late-scale path at all.

### Fix (revised): fix at the source instead of duplicating the conversion in the guard

An initial version of this fix worked around the problem locally in
`preflight()` by evaluating the late-scale check against a scratch copy of
`problem` with `autoTypeConversions()` re-applied. That worked, but it was
ugly: it duplicated conversion logic that's supposed to run once, and left
`preflight()` needing to know about a call-site inconsistency that really
belongs to its callers.

`autoTypeConversions()` is already called *unconditionally* in half a
dozen other places (`gemm.cxx`, `driver_info.cxx`, `gemm_microkernel.cxx`,
`compute_utils.hpp`, `kernel_selector.cpp`) -- it's designed to be
idempotent and safe to call more than once. The only two call sites that
gated it behind `needsAGroupSums()||needsBGroupSums()` were the ones
feeding `problem` into `strategy.preflight()`:

- `gen_kernel.cpp` (`gen_desc_t::finalize()`), immediately before
  `strategy_.preflight(hw_, problem_)`.
- `pd.cpp` (`pd_t::init_GEMMProblem()`), which builds and stores the
  `GEMMProblem` that eventually reaches `preflight()` via `gen_kernel.cpp`.

The clean fix is to remove the conditional gate at both of those call
sites and simply call `problem.autoTypeConversions(...)` unconditionally,
matching every other call site. This makes `problem.Ta`/`Tb` already
reflect their final, kernel-body-matching form by the time `preflight()`
runs, so `preflight()` itself needs **no changes at all** beyond a comment
noting the invariant -- the late-scale check goes back to reading
`problem.Ta`/`Tb`/etc. directly, with no scratch copy required.

`needsAGroupSums()/needsBGroupSums()` are unaffected by this reordering:
they key off `Ta_ext`/`Tb_ext`/`Tao`/`Tbo` (not `Ta`/`Tb`), which
`autoTypeConversions()` doesn't touch.

This is strictly better than the scratch-copy workaround: it fixes the
actual inconsistency (conditional vs. unconditional conversion across call
sites) rather than papering over its effect in one specific consumer, and
it means any *other* future logic added to `preflight()` that depends on
`Ta`/`Tb` will automatically see correct, converted types without needing
its own local workaround.

### Verification

- User's new WOQ perf repro (`f16:u4:f16`, `per_ocic`-style scale, no
  zero-points): perf restored to **0.023ms** (matching the no-guard
  baseline), `PASSED` correctness.
- Original stream-K late-scale correctness repro (`u8:s8:s32`,
  `per_tensor:f16:256x1`, no zero-points): still `PASSED`.
- Original group-sum correctness repro (`f16:s8:f16`, `per_ocic` grouped
  scale + zero-points): still `PASSED`.

This highlights that `preflight()`-time predicates relying on `problem.Ta`/
`Tb` for late-quantization decisions must always evaluate them
post-`autoTypeConversions()` to match the kernel body's actual code path,
since those types are conditionally converted upstream depending on
whether group-sum compensation is needed.

### Follow-up: extract the late-scale condition into a shared helper

The late-scale predicate duplicated in `strategy.cpp`'s `preflight()` guard
was hand-copied from the `if` condition in
`gemmMake2DQuantizationLayouts()` (`quantization.cxx`). Duplicated boolean
logic like this is exactly the kind of thing that caused earlier bugs in
this investigation (see Correction #2) -- two copies of "the same" check
can silently diverge over time as one site is edited and the other isn't.

To eliminate that risk, the condition is now defined once as
`usesLateScaleA()`/`usesLateScaleB()` in `compute_utils.hpp` (alongside the
related `minOuterProductCount()`/`outerProductCount()` helpers, which the
late-scale condition also depends on), taking `(problem, strategy)` and
returning whether that operand's 2D scale will be applied late by the
kernel body. Both `quantization.cxx`'s `gemmMake2DQuantizationLayouts()`
and `strategy.cpp`'s `preflight()` guard now call this single function, so
there is exactly one place that encodes "is this operand's scale
late-applied" and both consumers are guaranteed to agree, by construction.

Note: `quantization.cxx`'s original condition referenced `state.useBDPAS`
directly; the shared helper instead uses `problem.preferBDPAS() &&
strategy.systolic`, which is what `state.useBDPAS` is unconditionally set
to (`gemm_setup.cxx`, `state.useBDPAS = problem.preferBDPAS() &&
strategy.systolic;`) before `gemmMake2DQuantizationLayouts()` is ever
called -- so this is not a behavior change, just expressing the same value
without requiring a `GEMMState` (which `preflight()` doesn't have).

Verified: rebuilt cleanly across all HW generator targets (XeLP/XeHP/
XeHPG/XeHPC/Xe2/Xe3/Xe3p); all previous correctness/perf repros
(user's WOQ regression, original late-scale hazard, group-sum hazard, and
the s4/s8/bf16:u4 dtype sweep) still pass with identical results.

## Correction #3: unconditional `autoTypeConversions()` in `pd.cpp` broke f8/MX-scale catalog matching

After making the `pd.cpp` call to `problem.autoTypeConversions()` unconditional (Correction #2), a regression was reported for an f8 MX-scale case that previously passed:

```
--engine=gpu --matmul --impl=jit:gemm:any --dt=f8_e5m2:f8_e5m2:f8_e5m2 \
  --stag=abx --dtag=abx \
  --attr-scales=src:per_tensor:e8m0:1x32+dst:mx:e8m0:1x32+wei:per_tensor:f32:32x1 \
  --attr-post-ops=sum:4 1208x672:672x544
```

This now returned `UNIMPLEMENTED` (primitive creation failed) instead of `PASSED`.

### Root cause

Unlike `gen_kernel.cpp` (whose catalog-matching logic keys off
`problem.Ta_ext`/`Tb_ext`, which `autoTypeConversions()` never touches),
`pd.cpp` performs its own kernel/catalog selection and problem
finalization logic (e.g. `nativeBDPAS()`, `bdpasEnabled`, s8/s4 DPAS
mixing checks) using `problem.Ta`/`Tb` directly, at various points *after*
the point where the conversion call was moved to be unconditional. Making
`autoTypeConversions()` fire even when there are no zero-points (e.g. this
f8 case, which needs no group-sum compensation at all) converted `Ta`/`Tb`
away from `f8` (via the `Ta.isF8() -> Ta = Type::f16` branch, gated only on
HW/systolic availability) before `pd.cpp`'s own downstream f8/MX-specific
logic ran, which no impl in the kernel catalog could then satisfy --
hence `UNIMPLEMENTED`.

### Fix: revert `pd.cpp`, keep `gen_kernel.cpp` unconditional

`pd.cpp`'s call is reverted back to its original conditional form
(`if (problem.needsAGroupSums() || problem.needsBGroupSums())
    problem.autoTypeConversions(has_systolic);`), since it has different,
non-idempotent-safe consumers downstream of it in the same function.
`gen_kernel.cpp`'s call remains unconditional, since that file's own
catalog-matching logic operates on `Ta_ext`/`Tb_ext` (unaffected by the
conversion) rather than `Ta`/`Tb`, and it is the last consumer before
`strategy.preflight()` runs -- so `preflight()` still always sees
converted types, without perturbing `pd.cpp`'s independent, f8-sensitive
selection logic.

### Verification

- f8 MX-scale repro: restored to `PASSED`.
- Additional f8/f4 sweep (`f8_e4m3` per-tensor, `f4_e2m1` per-tensor):
  `PASSED`.
- All previously-fixed WOQ perf/correctness repros re-verified: still
  `PASSED` with perf unchanged (~0.023ms on the u4 regression repro).

This is a reminder that "make a conditional call unconditional" fixes are
only safe when *every* consumer downstream of the call site is
insensitive to the difference (or has already been checked against it) --
here, `gen_kernel.cpp` was safe to convert but `pd.cpp` was not, because
they have different downstream logic operating on the converted fields.
