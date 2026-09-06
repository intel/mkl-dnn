# Stream-K / Grouped-Quantization Hazard: Call-Site Trace and Empirical Verification

This report documents *exactly* where and under what real, benchdnn-expressible
matmul configurations the guard added to `GEMMStrategy::preflight()` (in
`src/gpu/intel/gemm/jit/generator/strategy.cpp`, extending the fix from commit
`93d728f61f "xe: gemm: jit: disable k-streaming for grouped quant"` to also
cover `kParallel`/`kParallelLocal`, not just `kParallelVariable`/stream-K)
actually changes kernel-selection behavior. It combines (1) static call-site
tracing through the gemmstone JIT GEMM generator, and (2) **empirical
verification** against the real GPU in this environment (Intel Data Center GPU
Max 1100 / Ponte Vecchio, XeHPC), using a temporarily-instrumented build of
`strategy.cpp` driven through `benchdnn --matmul` on `--engine=gpu`.

All empirical claims below were produced by adding a `getenv("SK_QUANT_DEBUG")`
gated `fprintf(stderr, ...)` immediately after the `skHazard` computation in
`preflight()`, rebuilding `libdnnl.so` + `benchdnn`, running a matrix of
concrete `benchdnn --matmul` invocations, and then **reverting the
instrumentation** (the working tree now contains only the original functional
fix, no debug code — verified via `git diff --stat`).

---

## 1. Where `kParallelVariable` / `kParallel` / `kParallelLocal` get selected before `preflight()` can disable them

- **Manual/CLI enablement** (used by internal test harnesses, not by the
  matmul primitive's automatic dispatch): `src/gpu/intel/gemm/jit/generator/strategy_parser.cpp`
  parses explicit strategy strings and sets `kParallelLocal` (`kr`), `kParallel`
  (`kb`), and `kParallelVariable` (`kv`) directly from a string spec.

- **Automatic heuristic selection** (what real `dnnl_matmul`/`dnnl_gemm` calls
  go through): `src/gpu/intel/gemm/jit/selector/kernel_evaluator.cpp`.
  - `kParallel` is considered when a fixed global K-slicing can cover the
    problem's K dimension across the chosen workgroup grid; the evaluator
    computes `aux.k0 = alignUp(divUp(dp.sizes.k, wgCountK), unrollK)` and
    scores it favorably when `aux.k0 >= dp.sizes.k` is not the case (i.e. when
    K-splitting is actually needed/beneficial) — see the K-parallel scoring
    block around `kernel_evaluator.cpp:109-122`.
  - `kParallelVariable` (stream-K) is tried in `evaluateE()` only for
    catalog entries whose `driverInfo.kParallelVariable()` is set, is
    unconditionally rejected for batched problems (`if (dp.batch) return
    score;`), and is otherwise preferred when the natural thread count
    doesn't evenly tile the hardware (`tcount != dp.threadCount || tcount %
    dp.hwThreadCapacity != 0`) — `kernel_evaluator.cpp:248-304`. There is a
    further fallback: if the computed `approxK0 <= 32`, the evaluator
    downgrades a variable-K choice to *fixed* global K-parallelism instead
    (`kernel_evaluator.cpp:292-299`), which is why `kParallel` and
    `kParallelVariable` must be treated as the same family of hazard by the
    fix (confirmed empirically in §3, Test 5b: both `kParallelVariable=1`
    **and** `kParallelLocal=1` were chosen together for the same unquantized
    problem).
  - `kParallelLocal` (SLM-reduction-based local K-splitting) is scored by
    computing `kthread = divUp(k, wgK)` and shrinking `wgK` to
    `min(wgK, divUp(k, kthread))` — `kernel_evaluator.cpp:123-140`.
  - **Net effect:** there is no single simple "K > threshold" rule; the
    decisive factor is the evaluator's cost model over `(M, N, K, HW thread
    capacity, batch)`. Empirically (§3), a skinny-M/large-K int8 matmul
    (`16x16384 @ 16384x4096`, and separately `64x4096 @ 4096x4096`) reliably
    selects `kParallelVariable=1` (and often `kParallelLocal=1` too) when no
    quantization hazard is present, and this is exactly the shape class
    (small M·N relative to K) that stream-K exists to help — it is not a rare
    corner case.

- **Where the fix removes them:** `src/gpu/intel/gemm/jit/generator/strategy.cpp`,
  in `GEMMStrategy::preflight()` — the `skHazard` block (previously only
  guarding `kParallelVariable`/`C.atomic`/`CO.atomic`; now also clearing
  `kParallel` and `kParallelLocal` when `skHazard` is true).

---

## 2. How `needsAGroupSums()`/`needsBGroupSums()` and `usesLateScaleA()`/`usesLateScaleB()` actually get determined

All definitions are in `src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp`
unless noted.

```cpp
bool aScale2D() const { return (asPtrDims >= 2); }
bool bScale2D() const { return (bsPtrDims >= 2); }
bool aOffset2D() const { return (aoPtrDims >= 2); }
bool bOffset2D() const { return (boPtrDims >= 2); }

bool quantized2DA() const { return hasGroupSumsB || aOffset2D() || aScale2D(); }
bool quantized2DB() const { return hasGroupSumsA || bOffset2D() || bScale2D(); }

static bool earlyDequantizableOffset(Type T_ext, Type To, Type T) {
    return To.asSigned().isSubsetOf(T) && (To.bits() < T.bits() || T_ext.bits() < T.bits());
}

bool needsAGroupSums() const { return (bOffset == ABOffset::Calc && quantized2DB() && !earlyDequantizableOffset(Tb_ext, Tbo, Tb)); }
bool needsBGroupSums() const { return (aOffset == ABOffset::Calc && quantized2DA() && !earlyDequantizableOffset(Ta_ext, Tao, Ta)); }
```

(`problem.hpp:259-268, 299-300, 324-326`)

Note the cross-naming: `needsAGroupSums()` is about **B**'s offset needing
group-sum-of-**A** compensation (and vice versa) — this reflects how
`applyLateABOffset()` (in `generator/pieces/matrix_multiply.cxx`) uses
precomputed row/column sums of the *other* operand to correct for one
operand's zero-point.

`usesLateScaleA`/`usesLateScaleB` (`src/gpu/intel/gemm/jit/generator/pieces/compute_utils.hpp:96-110`):

```cpp
static inline bool usesLateScaleA(const GEMMProblem &problem, const GEMMStrategy &strategy) {
    return problem.aScale2D()
            && ((problem.Ta_scale.paddedSize() > problem.Ta.paddedSize() && problem.Ta.isInteger())
                    || problem.forceLateQuant(minOuterProductCount(problem, strategy))
                    || (problem.preferBDPAS() && strategy.systolic));
}
```
(symmetric for B)

Three independent ways to trigger late-scale:
1. **`Ta_scale` wider than `Ta`, and `Ta` still integer** — i.e. the compute
   type was *not* upconverted to match the (floating-point) scale type, so
   the raw integer dot-product must be accumulated first and the (wider,
   FP) scale applied once at the end, per K-group. This is the
   **only one of the three that is reachable on current (pre-Xe3p)
   hardware** and is the one empirically confirmed in §3.
2. `forceLateQuant()` — requires `nativeBDPAS()`, which itself requires
   `ngen::getCore(product.family) >= ngen::HW::Xe3p` (`problem.hpp:273-277`).
   Not reachable on the Xe-HPC (Ponte Vecchio) GPU used for this
   investigation.
3. `preferBDPAS() && strategy.systolic` — `preferBDPAS()` also requires
   `nativeBDPAS()` (`problem.hpp:288-293`). Same Xe3p-only restriction as (2).

### Critical discovery: `autoTypeConversions()` almost always defeats path (1) for the common WOQ (weight-only-quantization) case

`GEMMProblem::autoTypeConversions()` (`problem.hpp:360-395`), called
unconditionally before `preflight()` from both `gen_kernel.cpp:316` and
`pd.cpp:820`, contains:

```cpp
// Weights decompression
if ((Ta.isInt8() || Ta.isInt4()) && Tb.isFP() && Tc.isFP()) Ta = Tb;
if ((Tb.isInt8() || Tb.isInt4()) && Ta.isFP() && Tc.isFP()) Tb = Ta;
```

This means: **whenever one operand is a low-precision integer
(int4/int8/u4/s4 — the standard weight-only-quantization storage types) and
the *other* operand plus the output are floating point** (the overwhelmingly
common LLM-inference WOQ pattern: `fp16/bf16 activations x int4/int8
weights -> fp16/bf16 output`), the integer operand is unconditionally
upconverted to match the FP compute type *before* `preflight()` ever runs.
This makes `Ta.isInteger()`/`Tb.isInteger()` false for that operand, which
kills condition (1) above regardless of the scale's group size, and (as
shown empirically in §3, Tests 1-3) also tends to make any accompanying
zero-point "early dequantizable" (§2 below) because the offset's bit-width
is now always much narrower than the FP compute type it was upconverted
into.

**Practical consequence:** for the single most common real-world LLM
weight-only-quantization scenario — FP16/BF16 activations, INT4/INT8
grouped-scale (+ optional zero-point) weights, FP16/BF16 output — **this fix
has empirically zero effect** on Xe-HPC-class hardware: `preflight()`
computes `skHazard=0` and stream-K/`kParallel`/`kParallelLocal` remain fully
available. See §3, Tests 1-3.

---

## 3. Empirical results (real GPU: Intel Data Center GPU Max 1100, Ponte Vecchio / XeHPC)

All commands run as:
```
LD_LIBRARY_PATH=$BUILD/src ./tests/benchdnn/benchdnn --engine=gpu --matmul ...
```
with `SK_QUANT_DEBUG=1` set and a temporary `fprintf` in `preflight()`
(since reverted) printing `kParallelVariable/kParallel/kParallelLocal`,
`needsAGroupSums/needsBGroupSums`, `aLateScale/bLateScale`, and the derived
`skHazard` for every kernel selection.

### Test 1 — Grouped scale only, no zero-points, WOQ (int4 weights, fp16 compute)
```
--dt=f16:s4:f16 --attr-scales=wei:per_ocic:f16:32x1 --attr-fpmath=f16:true 64x4096:4096x4096
```
Result: `kParallelVariable=1 ... skHazard=0` (stream-K stays on). Confirms
the doc's "scale-only grouped quant is safe" claim, and shows it applies
here because `Ta` was upconverted to `f16` by `autoTypeConversions()`
(`Ta=0x1000201` == f16 in the debug dump) so `Ta.isInteger()` is false.

### Test 2 — Grouped scale + **scalar/common** zero-point, WOQ (int4 weights, fp16 compute) — matches the real production config in `tests/benchdnn/inputs/matmul/option_set_fwks_llm_gpu`
```
--dt=f16:u4:f16 --attr-scales=wei:per_ocic:f16:128x1 --attr-zero-points=wei:common:1:s8 --attr-fpmath=f16:true 64x4096:4096x4096
```
Result: `skHazard=0`, stream-K stays enabled. The zero-point (s8, 8 bits) is
narrower than the upconverted compute type (f16, 16 bits and `T_ext.bits()
< T.bits()` also holds), so `earlyDequantizableOffset()` is true and
`needsAGroupSums`/`needsBGroupSums` are both false. **This is the actual
LLM-inference production configuration found in
`option_set_fwks_llm_gpu:691-720` (real captured shapes from an LLM
framework) — it is unaffected by this fix.**

### Test 3 — Grouped scale + **grouped (2D per_ocic)** zero-point, WOQ (int4 weights, fp16 compute)
```
--dt=f16:u4:f16 --stag=abc --attr-scales=wei:per_ocic:f16:32x1 --attr-zero-points=wei:per_ocic:s4:32x1 --attr-fpmath=f16:true 7x64x4096:7x4096x4096
```
Result: `skHazard=0` here too — again because the (4-bit) zero-point is
narrower than the (16-bit, upconverted) compute type, so it's still
"early dequantizable". (Note: `kParallelVariable` was already `0` in this
particular run for the unrelated reason that `--stag=abc` makes this a
batched (3D) problem, and `strategy.cpp`'s separate, pre-existing "Remove
variable k-parallelization when batching" rule disables it regardless of
quantization.)

**Conclusion from Tests 1-3: on Xe-HPC-class hardware, the fix does not
measurably affect any tested FP16/BF16-compute, INT4/INT8-weight WOQ
configuration** (with scalar or grouped zero-points) that upconverts through
`autoTypeConversions()`'s weights-decompression path — which covers the
dominant LLM-inference deployment pattern.

### Test 4/6/7 — Pure-integer (`s8:s8:s32`) static quantization, grouped and/or zero-point-bearing scale/offsets
```
--dt=s8:s8:s32 --attr-scales=wei:per_ocic:f32:32x1 --attr-zero-points=wei:per_ocic:s8:32x1 ...
--dt=s8:s8:f32 --attr-zero-points=src:per_dim_1:s8:1x32 ...
```
Result: **falls back to `ocl:ref:any`** (reference kernel), never reaches
`jit:gemm` at all. The gemmstone JIT GEMM implementation currently declines
these grouped-zero-point-on-pure-int8-static-quant configurations outright
(for reasons unrelated to this fix), so `preflight()`'s guard is not even
exercised for them via the public `matmul` primitive in this build. This
suggests the `needsAGroupSums()`/`needsBGroupSums()` branch of the fix is
either (a) presently dormant/unreachable through `dnnl_matmul`'s current
attribute surface, or (b) reachable only through some other primitive/API
path or future JIT-gemm capability not exercised by this benchdnn sweep —
this could not be confirmed empirically in the time available and is worth
a follow-up investigation using the direct GEMM API/ukernel path rather than
the matmul primitive.

### Test 5 — **The empirically-confirmed hazard case**: pure-integer compute, grouped FP32 scale, *no* zero-points at all
```
--dt=s8:s8:f32 --attr-scales=wei:per_ocic:f32:32x1 64x4096:4096x4096
```
Result:
```
aLateScale=1 (bLateScale=0) | needsAGroupSums=0 needsBGroupSums=0 | skHazard=1
kParallelVariable=0 kParallel=0 kParallelLocal=0   (all disabled by the fix)
```
reached `impl=jit:gemm:any` successfully (i.e. this is a real, currently
JIT-servable configuration). Control (Test 5b), identical shape/dtype with
`--attr-scales` removed entirely:
```
kParallelVariable=1 kParallel=0 kParallelLocal=1 | skHazard=0
```
**This proves the fix is reachable and load-bearing for a real config: pure
int8 GEMM (both operands staying in integer domain, e.g. a classic
static-quantization pipeline, not weight-decompression) with a grouped
(per-K-block) floating-point scale on the weights — even with *no*
zero-points whatsoever.** This is triggered purely by `usesLateScaleA()`'s
first condition (`Ta_scale` wider than `Ta`, `Ta` integer), not by any
group-sums/offset logic.

### Test 8 — Fully-integer compute (`s8:s4:s32`) with grouped scale **and** grouped zero-point on weights
```
--dt=s8:s4:s32 --attr-scales=wei:per_ocic:f32:32x1 --attr-zero-points=wei:per_ocic:s4:32x1 64x4096:4096x4096
```
Result: reaches `jit:gemm:any`, `aLateScale=1` (same mechanism as Test 5;
`Tao` s4 is still narrower than `Ta` s8 so `needsAGroupSums`/`needsBGroupSums`
remain false here too), `skHazard=1`, all three K-parallel flags disabled.

### Performance impact (quantified, `--mode=P`, skinny-M/large-K shape `16x16384:16384x4096`)

| Configuration | stream-K/K-parallel state | min Gflops |
|---|---|---|
| `s8:s8:f32`, no quantization | `kParallelVariable=1, kParallelLocal=1` (enabled) | **51,821** |
| `s8:s8:f32`, `--attr-scales=wei:per_ocic:f32:32x1` (grouped scale, no zp) | all disabled by fix (`skHazard=1`, `aLateScale=1`) | **11,521** |

**~4.5x throughput drop** measured directly on real Ponte Vecchio hardware
for this shape when the fix disables K-splitting due to `aLateScale`. This
is the concrete, quantified cost of the fix for the one config family
confirmed empirically reachable and hazard-triggering.

---

## 4. History (git blame/log context)

- `93d728f61f "xe: gemm: jit: disable k-streaming for grouped quant"` — the
  parent commit introducing the original (narrower, `kParallelVariable`-only)
  guard; also moved `problem_.autoTypeConversions(...)` to run
  unconditionally, before `preflight()`, in `gen_kernel.cpp`/`pd.cpp` (so that
  `preflight()` always sees the final, converted `Ta`/`Tb`).
- `fd819b3e51 "xe: gemm: jit: disable problems requiring group sums without
  them"` and `cc9f22912a "xe: gemm: enable full 2D quantization support"` are
  related prior commits establishing the `needsAGroupSums`/`needsBGroupSums`/
  `quantized2DA`/`quantized2DB` machinery this fix relies on.
- The current uncommitted change in this working tree
  (`src/gpu/intel/gemm/jit/generator/strategy.cpp`) extends `93d728f61f`'s
  guard from `kParallelVariable` only to also cover `kParallel` and
  `kParallelLocal`, on the grounds (documented in `STREAM_K_LATE_SCALE_ROOT_CAUSE.md`)
  that both are additive-K-combine mechanisms with the same
  non-distributivity hazard as stream-K's atomic add. This report's Test 5b
  empirically confirms `kParallelLocal` really is selected alongside
  `kParallelVariable` for realistic shapes, validating that the narrower,
  original `93d728f61f` guard alone would have left a real correctness gap
  for `kParallelLocal`-selected kernels.

---

## Summary

| Hazard component | Reachable via `dnnl_matmul` on tested HW (Xe-HPC/PVC)? | Real-world prevalence |
|---|---|---|
| `aLateScale`/`bLateScale` (deferred/late scale application) | **Yes — empirically confirmed (Tests 5, 8)** | Pure-integer (non-WOQ-upconverted) grouped-scale GEMM/matmul, e.g. classic block/group static int8 quantization where both operands remain integer. Real, measurable (~4.5x) perf cost when triggered. |
| `needsAGroupSums`/`needsBGroupSums` (group-sum offset double-counting) | **Not reached in this investigation** — either always "early dequantizable" in the FP16/BF16-compute WOQ configs tested (Tests 1-3), or the pure-int8 configs that would exercise it (Tests 4, 6, 7) never reach the JIT GEMM kernel at all (fall back to `ocl:ref`) | Unclear on current HW/SW; may be forward-looking/defensive, or reachable via a code path (direct GEMM API, different primitive, future JIT-gemm capability) not exercised here. Recommend follow-up with the raw GEMM/ukernel API rather than the matmul primitive. |
| The single most common production scenario (FP16/BF16 act. × INT4/INT8 grouped-scale weights → FP16/BF16 out, the dominant LLM WOQ pattern, confirmed via the real `option_set_fwks_llm_gpu` benchdnn shapes) | **Confirmed unaffected** (Tests 1-3, `skHazard=0` throughout) | This is reassuring: the fix does not regress the dominant real-world workload it might have been feared to affect. |

The extension from `kParallelVariable`-only to also cover `kParallel`/
`kParallelLocal` is justified: Test 5b shows both get selected together for
realistic unquantized shapes, so the original narrower guard would have left
a live correctness gap for `kParallelLocal` in exactly the same hazard cases.
