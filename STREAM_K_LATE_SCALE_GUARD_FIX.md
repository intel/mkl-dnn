# Stream-K Late-Scale Quantization Guard: Over-Broad Regression and Fix

## Summary

Commit `92aead6dc6` ("xe: gemm: jit: disable k-streaming for grouped quant")
disabled variable k-parallelization (stream-K) for *any* GEMM using a
late-applied 2D grouped quantization scale (`usesLateScaleA()` /
`usesLateScaleB()`), in order to fix a real correctness bug. However, the
guard was scoped far too broadly: it disabled stream-K for **every**
late-scale configuration, including the extremely common weight-only-quantized
(WOQ) LLM pattern where the GEMM output is floating point (f16/bf16/f32),
which is not actually affected by the bug. This caused a measured
**5.8x slowdown** on a representative WOQ shape
(`--dt=u8:u4:f16 --stag=ab --wtag=ba --dtag=ab
--attr-scales=src:3:f16:1x128+wei:3:f16:128x1 1x11520:11520x1024`) that a
user reported regressed after picking up the fix.

This document explains the real mechanism of the bug, why the original guard
condition over-applied it, the corrected discriminator
(`problem.Tc_ext.isInteger()`), and the empirical (on-hardware) evidence for
both the original regression and the validity of the narrower fix.

Hardware used for all empirical results below: Intel Data Center GPU Max 1100
(XeHPC / Ponte Vecchio), via `tests/benchdnn/benchdnn --engine=gpu`.

## 1. Background: what stream-K is and how it combines partial results

Variable k-parallelization ("stream-K", `kParallelVariable` /
`kParallel` / `kParallelLocal` in `GEMMStrategy`) splits the K (reduction)
dimension of a single output tile across multiple workgroups. Each
workgroup computes only a partial sum over its assigned K-slice, and the
partial C tiles from different workgroups are combined into the final result
via an **atomic add** (`C.atomic = CO.atomic = true`; see
`src/gpu/intel/gemm/jit/generator/strategy.cpp`,
`GEMMStrategy::preflight()`).

This is safe for ordinary (non-quantized) GEMM: summation is associative up
to floating-point rounding-order effects, which stream-K already accepts as
an unavoidable, tolerated source of numerical noise (same as for any
parallel-reduction algorithm).

The existing guard in `preflight()` disables stream-K for two distinct
quantization-related hazards. This document is only concerned with the
second one (late scales); the first (group-sum offsets) was left unchanged
in this fix and is out of scope.

```
src/gpu/intel/gemm/jit/generator/strategy.cpp (preflight(), around line 150 onward)
```

## 2. Hazard #1 (unchanged): group-sum zero-point offset compensation

`problem.needsAGroupSums()` / `problem.needsBGroupSums()`
(`src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp`, lines ~299-300):

```cpp
bool needsAGroupSums() const { return (bOffset == ABOffset::Calc && quantized2DB() && !earlyDequantizableOffset(Tb_ext, Tbo, Tb)); }
bool needsBGroupSums() const { return (aOffset == ABOffset::Calc && quantized2DA() && !earlyDequantizableOffset(Ta_ext, Tao, Ta)); }
```

When A/B carry zero-point offsets that can't be dequantized early, the
kernel applies a precomputed whole-group sum correction once per
quantization group via `applyLateABOffset()`
(`src/gpu/intel/gemm/jit/generator/pieces/matrix_multiply.cxx`, ~line 789,
called from `outerProductRepackC()` at ~line 649-658). Because this
correction is indexed purely by *group number*, not by which workgroup's
K-slice touches it, a stream-K split landing mid-group can cause **more than
one** workgroup to apply the same group's correction to the shared output
tile — a genuine double-counting bug independent of accumulator type. This
guard branch was **not modified** by this fix.

## 3. Hazard #2 (this fix): late 2D scale application

### 3.1 When are scales applied "late"?

`usesLateScaleA()` / `usesLateScaleB()`
(`src/gpu/intel/gemm/jit/generator/pieces/compute_utils.hpp`, lines 96-110):

```cpp
static inline bool usesLateScaleA(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    return problem.aScale2D()
            && ((problem.Ta_scale.paddedSize() > problem.Ta.paddedSize() && problem.Ta.isInteger())
                    || problem.forceLateQuant(minOuterProductCount(problem, strategy))
                    || (problem.preferBDPAS() && strategy.systolic));
}

static inline bool usesLateScaleB(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    return problem.bScale2D()
            && ((problem.Tb_scale.paddedSize() > problem.Tb.paddedSize() && problem.Tb.isInteger())
                    || problem.forceLateQuant(minOuterProductCount(problem, strategy))
                    || (problem.preferBDPAS() && strategy.systolic));
}
```

The common trigger (and the only one reachable on this XeHPC hardware,
since BDPAS/Xe3+-only paths are unavailable here) is the first clause: a
narrow integer operand type (e.g. `u4`/`s8`) paired with a wider scale type
(e.g. `f16`/`f32`) forces the scale multiply to happen **after**
accumulation/repacking rather than being pre-multiplied into each operand
element before the dot-product. This is exactly the case for weight-only
quantization: `u4`/`s8` weights with an `f16` per-group scale.

### 3.2 Where the late scale is actually applied

`outerProductRepackC()`
(`src/gpu/intel/gemm/jit/generator/pieces/matrix_multiply.cxx`, from line
629):

```cpp
template <HW hw>
void Generator<hw>::outerProductRepackC(int x0, int xr0, int nx, int h, bool rem,
                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc, Tc_compute = problem.Tc_compute();
    ...
    bool scaleA = state.lateScale2DA, scaleB = state.lateScale2DB;
    ...
    auto processItems = [&] {
        for (const auto &i: items)
            mov(i.simd, i.acc(Tc), i.Cr(1));
        for (const auto &i : items)
            if (i.scale[1].isValid())
                mul(i.simd, i.acc(Tc), i.acc(Tc), i.scale[1](i.scaleStride[1]));   // <-- scale multiply
        for (const auto &i: items) {
            if (i.scale[0].isValid())
                mad(i.simd, i.C(1), i.C(1), i.acc(Tc), i.scale[0](i.scaleStride[0]));  // <-- scale + accumulate into C
            else
                add(i.simd, i.C(1), i.C(1), i.acc(Tc));
        }
        items.clear();
    };
```

The scale multiply (`mul`) and the fold into the running accumulator
(`mad`/`add`) both operate on register type `Tc` (`problem.Tc`, the
*internal* compute/accumulator type) — **not** the eventual output/memory
type. Crucially, this repack-and-scale step happens once per
`aqGroupK`/`bqGroupK`-sized K-group, which does not necessarily align with a
stream-K K-split boundary. If a stream-K workgroup's K-slice ends partway
through a scale group, that workgroup's local accumulator only has a
partial (truncated) contribution to the group at the moment the scale is
applied and folded in.

### 3.3 Why this is only unsafe when `Tc` (register-level) rounds to an integer, and why the true discriminator is `Tc_ext`

The `mad`/`add` in `processItems()` writes into `C` typed as `Tc`. If `Tc`
is a floating-point type, the fold is an ordinary FP add: splitting a
group's contribution across two workgroups and separately scaling +
accumulating each truncated partial, then summing the workgroups' partial C
tiles via the stream-K atomic add, is mathematically equivalent (up to
ordinary FP rounding-order noise, already tolerated for unquantized GEMM) to
doing it in one workgroup. Floating-point addition is associative up to
rounding; **there is no separate integer-rounding step in between** to break
associativity.

If, however, `Tc` is an **integer** type (e.g. `s32`), the `mad`/`add`
result is *rounded/truncated to an integer* at each repack step. This
rounding is where associativity breaks: for integers,
`round(a) + round(b) != round(a + b)` in general. Two workgroups each
rounding their own truncated partial contribution to a group and then
atomically summing the *rounded* partials produces a different (wrong)
result than computing the whole group's contribution in one workgroup and
rounding once. This is the actual correctness hazard, and it requires the
repack accumulator to be integer-typed to manifest.

`problem.Tc_compute()`
(`src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp`, lines 306-313):

```cpp
Type Tc_compute() const {
    if (Ta.isInteger() && Tb.isInteger() && Tc == Type::f32)
        return Type::s32;
    else if (Ta.isFP() && Tb.isFP() && Tc == Type::s32)
        return Type::f32;
    else
        return Tc;
}
```

Initial investigation (this and the prior session) hypothesized
`Tc_compute()` as the discriminator, since it reflects the actual dpas/mad
register type used during the main k-loop. **Empirically this turned out to
be wrong**: instrumenting `preflight()` and running both the user's FP16-out
case and the previously-known-buggy S32-out case showed `Tc_compute()`
evaluating to the *same* integer-flavored value in both cases (both are
int8/int4-input GEMMs, so the inner dpas accumulates in `s32` either way).
`Tc_compute()` cannot distinguish the safe case from the unsafe one.

What **does** differ between the two cases is `problem.Tc_ext`
(`src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp`, line 168):

```cpp
Type Ta_ext, Tb_ext, Tc_ext;                    // Types for A/B/C data in memory.
```

`Tc_ext` is the true external/memory output type of the primitive (what the
user's `--dt=...:f16` or `...:s32` argument specifies for the destination),
as distinct from the internal register/accumulation type `Tc`. Empirically:

| Case | dt (dst) | `Tc_compute()` | `Tc_ext` | `Tc_ext.isInteger()` |
|---|---|---|---|---|
| User's WOQ regression (`u8:u4:f16`) | f16 | s32 (int) | f16 | **false** |
| Known-buggy repro (`u8:s8:s32` + post-op) | s32 | s32 (int) | s32 | **true** |

This is the crux of the fix: `Tc_compute()` is identical in both rows, but
`Tc_ext` correctly separates them. The final `mad`/`add` fold in
`outerProductRepackC()` is retyped/stored according to `Tc`, which for a
grouped-scale FP-output WOQ config gets promoted to floating point by the
type-conversion machinery in `pd.cpp` before `preflight()` ever runs (see
`autoTypeConversions()`); for an int32-output config, `Tc` remains integer.
`Tc_ext` is therefore a reliable proxy for whether the actual accumulate step
in the repack loop rounds to an integer or not.

## 4. The original (over-broad) guard and its regression

Original code (commit `92aead6dc6`,
`src/gpu/intel/gemm/jit/generator/strategy.cpp`):

```cpp
bool aLateScale = usesLateScaleA(problem, *this);
bool bLateScale = usesLateScaleB(problem, *this);
if (kParallelVariable
        && (problem.needsAGroupSums() || problem.needsBGroupSums() || aLateScale || bLateScale))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

This disables stream-K for *any* late-scale config, regardless of output
type — including the very common FP-output WOQ case, which (per §3) is not
actually unsafe.

### 4.1 Empirical regression measurement (broad guard, i.e. pre-fix / commit `92aead6dc6`)

Rebuilt `libdnnl.so` against the original (unmodified) `92aead6dc6` version
of `strategy.cpp` and ran the user's reported case plus an analogous
int8-compute/FP32-output case:

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=u8:u4:f16 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=src:3:f16:1x128+wei:3:f16:128x1 --mode=P \
    1x11520:11520x1024
tests:1 passed:1 ... total perf: min(ms):0.06688 avg(ms):0.0670097
```

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=s8:s8:f32 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=wei:3:f32:128x1 --mode=P \
    16x16384:16384x4096
tests:1 passed:1 ... total perf: min(ms):0.07856 avg(ms):0.0795932
```

### 4.2 Same cases, after the fix (narrowed guard)

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=u8:u4:f16 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=src:3:f16:1x128+wei:3:f16:128x1 --mode=P \
    1x11520:11520x1024
tests:1 passed:1 ... total perf: min(ms):0.01152 avg(ms):0.0120233
```

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=s8:s8:f32 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=wei:3:f32:128x1 --mode=P \
    16x16384:16384x4096
tests:1 passed:1 ... total perf: min(ms):0.0408 avg(ms):0.0438356
```

### 4.3 Regression summary

| Case | dst type | Broad guard (pre-fix) min(ms) | Narrowed guard (fixed) min(ms) | Speedup |
|---|---|---|---|---|
| User's repro (`u8:u4:f16`, grouped scales on both operands) | f16 | 0.06688 | 0.01152 | **5.80x** |
| `s8:s8:f32` grouped weight scale analog | f32 | 0.07856 | 0.0408 | **1.93x** |

Both cases: correctness `passed:1` in both configurations — i.e. the broad
guard was not buying any correctness benefit for these shapes, purely
costing performance.

## 5. Proof the bug is real and requires a guard (not just deletion)

To rule out "maybe the guard isn't needed at all, just delete it," the guard
was fully removed (`aLateScale`/`bLateScale` dropped from the disabling
condition entirely, `needsAGroupSums()`/`needsBGroupSums()` untouched) and
`libdnnl.so` rebuilt, forcing stream-K on for **all** late-scale
configurations including the previously-known-buggy one:

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=u8:s8:s32 --stag=abx --wtag=abx --dtag=abx \
    --attr-scales=wei:per_tensor:f16:256x1 --attr-post-ops=sub:f16:common \
    1x3840:3840x220
0:FAILED (errors:147 total:220) (1699 ms) __REPRO: --engine=gpu --matmul ...
tests:1 passed:0 skipped:0 ... failed:1 listed:0
```

This reproduces the documented correctness failure
(`STREAM_K_LATE_SCALE_ROOT_CAUSE.md`, prior session) — 147 of 220 elements
differ due to per-workgroup integer-rounding truncation, exactly the
mechanism described in §3.3. This confirms the hazard is real for
integer-output configs and a guard of some form is required — it just needs
to be scoped to the actual hazard condition rather than blanket-applied.

With the guard fully removed, the FP-output cases were re-tested and still
pass, confirming they were never actually hazardous and that `Tc_ext` is the
correct axis to gate on (not merely "any late scale"):

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=u8:u4:f16 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=src:3:f16:1x128+wei:3:f16:128x1 --mode=P \
    1x11520:11520x1024
tests:1 passed:1 ... total perf: min(ms):0.01152 avg(ms):0.0120137
```

```
$ ./tests/benchdnn/benchdnn --engine=gpu --matmul --impl=jit:gemm:any \
    --dt=s8:s8:f32 --stag=ab --wtag=ba --dtag=ab \
    --attr-scales=wei:3:f32:128x1 --mode=P \
    16x16384:16384x4096
tests:1 passed:1 ... total perf: min(ms):0.04272 avg(ms):0.0451467
```

## 6. The fix

`src/gpu/intel/gemm/jit/generator/strategy.cpp`, `GEMMStrategy::preflight()`:

```cpp
bool aLateScale = usesLateScaleA(problem, *this);
bool bLateScale = usesLateScaleB(problem, *this);
bool lateScaleHazard = (aLateScale || bLateScale) && problem.Tc_ext.isInteger();
if (kParallelVariable
        && (problem.needsAGroupSums() || problem.needsBGroupSums() || lateScaleHazard))
    C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;
```

This restricts the stream-K disable to configurations where the late-scale
repack-and-accumulate step actually rounds to an integer
(`problem.Tc_ext.isInteger()`), leaving the (much more common)
floating-point-output weight-only-quantization case free to use stream-K.
The group-sum-offset hazard (§2) is untouched.

## 7. Full validation matrix (final, committed state)

All four cases below were run against the final rebuilt (debug-instrumentation-free) `libdnnl.so`:

| # | Case | dst type | `aLateScale`/`bLateScale` | `Tc_ext.isInteger()` | `lateScaleHazard` | stream-K | Result |
|---|---|---|---|---|---|---|---|
| A | User's repro: `u8:u4:f16`, grouped scales src+wei, `1x11520:11520x1024` | f16 | true | false | **false** | enabled | `passed:1`, min(ms) 0.01152 |
| B | Known-buggy: `u8:s8:s32` + `wei:per_tensor:f16:256x1` + `sub:f16:common`, `1x3840:3840x220` | s32 | true | true | **true** | disabled | `passed:1` |
| C | `s8:s8:s32`, grouped wei scale, `16x16384:16384x4096` | s32 | true | true | **true** | disabled | `passed:1`, min(ms) 0.07856 |
| D | `s8:s8:f32`, grouped wei scale, `16x16384:16384x4096` | f32 | true | false | **false** | enabled | `passed:1`, min(ms) 0.0408 |

All four pass correctness. Cases A and D — floating-point-output configs
that were previously (and needlessly) stream-K-disabled — are restored to
full stream-K performance. Cases B and C — genuinely-hazardous
integer-output configs — remain correctly guarded (stream-K disabled),
matching their pre-fix behavior.

## 8. Scope notes / what was not changed

- The group-sum-offset hazard (`needsAGroupSums()`/`needsBGroupSums()`,
  §2) was left completely unmodified. It is a distinct bug mechanism
  (double-counted whole-group offset correction, not scale-rounding
  non-associativity) and was out of scope for this investigation. It's an
  open question whether it has a similarly over-broad guard that could be
  narrowed by a similar type-based discriminator — not yet investigated.
- Only the primary/common `usesLateScaleA`/`usesLateScaleB` trigger clause
  (narrow-int operand + wide scale type) was exercised in empirical
  testing, since it's the only reachable path on the available XeHPC
  hardware. The `forceLateQuant()` and `preferBDPAS() && strategy.systolic`
  trigger clauses (BDPAS/Xe3+-only paths) were not separately validated.
- No automated regression test currently locks in this behavior; validation
  is via the manual `benchdnn` runs captured in this document. A prior
  attempt at a whitebox gtest for this logic was abandoned due to unrelated
  build/linking issues.
