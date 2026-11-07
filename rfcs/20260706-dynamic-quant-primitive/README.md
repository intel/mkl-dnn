# RFC: Public Dynamic Quantization Primitive Support in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC adds dynamic quantization as a public oneDNN primitive, `dnnl::dynamic_quantize`, backed by an optimized x86 CPU path plus a portable CPU reference. Dynamic quantization derives a
tensor's quantization parameters (a `scale`, and for asymmetric mode a `zero_point`) from the data **at runtime** and quantizes it to INT8 / UINT8 at the requested granularity, in one
fused pass. It dominates LLM inference: per-token (one scale per row of `[seq_len, hidden]`) for W8A8 / W4A8 decode, and per-group (groups of `K` columns) for Llama.cpp / AWQ / GPTQ.
oneDNN today supports only *static* quantization: the caller supplies `attr.scales` / `attr.zero_points` and the library applies them inside `reorder`, `matmul`, etc. Nothing derives
those parameters at runtime, so a framework on `dnnl::matmul` has no `dnnl::dynamic_quantize` to call — it runs the reduction and the quantize as two separate passes, or routes to a
vendor library.

The first PR promotes this operation to a public C++/C API with two CPU impls: an optimized **x86** path that ports fused per-token and per-group **AVX-512 intrinsic kernels**
(min/max scan and quantize fused per group, so the source is read once and no intermediate is materialized), and a portable reference for the remaining
cases. The port lands the intrinsic bodies in `src/cpu/x64/` with oneDNN-native threading and glue. It is proposed as an **experimental**
API (§3.8): opt-in at build time and not yet under oneDNN's API/ABI stability guarantee, so the interface can evolve from integration feedback before promotion to a stable one.

## 1. Motivation

Dynamic quantization is the dominant activation path in INT8 LLM inference; at the model level the dynamic-quant step alone can account for **~5–20% of total run time**, so it is worth
optimizing in its own right. Frameworks such as PyTorch, ONNX Runtime, and vLLM expose it as a single operation. The activation changes every step, so its scale (and zero-point) must be
recomputed *per call*, before the matmul that consumes it. A public `dnnl::dynamic_quantize` lets oneDNN own that operation directly instead of reassembling it from a reduction plus a
separate quantize on the framework side. A framework calling its native quantize cannot easily tune threading and memory access patterns per shape on x86, but a primitive can select
blocking and parallelization per shape, as oneDNN already does for `matmul`.

The op is memory-bandwidth bound, so those x86-side optimizations compound. The **operator-level** payoff — the standalone kernel in isolation (no GEMM), the ported **x86 AVX-512**
kernel vs [vLLM's hand-tuned x86 int8 quant kernel][vllm-quant] (per-token BF16→INT8 on an x86 server, 128 threads) — shows each one in a different regime:

- **Fusion.** Against an eager two-pass path (reduction then a separate quantize, as PyTorch / ATen runs it), the fused kernel reads the activation once and keeps the scale in
  registers, instead of sweeping the tensor twice through DRAM and round-tripping an intermediate.
- **Single-token decode (`M = 1`, large `N`): ~14–100× faster** (mean ≈38× over the hidden sizes measured). vLLM issues an OpenMP parallel-for over a single row and pays the full
  fork/join; ZenDNN detects `M = 1` and runs the row on one thread, then a split variant computes the row scale and parallelizes the quantize over all `M x N` elements for full core
  utilization.
- **Small / mid batch (`1 < M < num_threads`, `M = 8–32`): ~6–19% faster on average** (up to ~34% at `K = 3584`), from CCX / CCD-aware row striping across AMD Zen Core Complexes (a CCX
  is 8 cores sharing an L3), spreading scan / quantize bandwidth across Core Complexes instead of piling onto one.

vLLM also exposes **only per-token** activation quant on x86 CPU — **no per-group (AWQ / GPTQ) x86 CPU kernel** — so per-group has no framework x86 CPU path at all; the primitive
provides the only optimized x86 path.

At the **model level**, both runs use the **same ZenDNN INT8 GEMM** and differ only in the activation dynamic-quant step — ZenDNN's fused kernel (via `zentorch`) vs ATen's two-pass
quantize — so the end-to-end delta isolates the dynamic-quant kernel. On serving throughput across Llama-2-7B, Mistral-7B, GPT-J-6B, Phi-4-reasoning-plus, and Qwen3-14B, the win is in
the **decode phase**: at 128 output tokens the ZenDNN-quant path delivers **~4–12% higher requests/s** than the ATen-quant path (mean ≈9%), largest at batch size 1 (≈1.13–1.18×), easing
to ≈1.04–1.07× at batch 32 — decode is memory-bound and small-batch, exactly where the fused kernel and its threading pay off.

Keeping dynamic quantization in the library gives one fused **x86** implementation shared across per-token and per-group paths, one public API for x86 kernel selection / threading and
for new x86 ISAs / CPU generations, and a single integration path for PyTorch, vLLM, and ONNX Runtime. Its semantics map directly onto framework operators — vLLM's per-token W8A8 /
W4A8, ONNX Runtime's `DynamicQuantizeLinear` (the per-tensor asymmetric u8 case), and PyTorch per-tensor / per-channel — with granularity encoded by the scale-tensor shape.

## 2. Non-Goals

- Do not extend or change `reorder`'s static-quantization contract.
- Do not add the upstream grouped / MoE `group_dynamic_quant` variant (batched over a `std::vector` of expert matrices) in the first PR.
- Do not add `bf16` / `f16` scale output, sub-byte / FP8 destinations, or fusion with a downstream matmul in the first PR.
- Do not add backward propagation (this is an inference-side primitive).
- Do not add a GPU implementation (the GPU impl list returns `empty_list`).

## 3. Proposal - public dynamic-quantize primitive

The design adds a first-class `dynamic_quantize` primitive: a new primitive kind, op descriptor, C/C++ API, and x86 / portable CPU implementations. `scale` / `zero_point` are output memory descriptors
and `compute_only` is a flag; `reorder`'s contract is untouched, and consumers that do not call `dnnl::dynamic_quantize` are unaffected.

### 3.1 Architecture overview

```
framework (vLLM / Llama.cpp / PyTorch / app)
        │  Primitive API
        ▼
dnnl::dynamic_quantize::primitive_desc(eng, alg, src, dst, scale, zp, [flags])
        │  → dnnl_dynamic_quantize_primitive_desc_create(...)   (new C entry)
        ▼  CASE(dynamic_quantize) → get_dynamic_quantize_impl_list → walk list
   ┌──────────────────────────────────────────────────────────┐
   │ x64::dynamic_quantize_t::pd_t::init()  (x86 CPU)          │
   │   2D dense · f32/bf16/f16 → s8/u8 ·                       │
   │   per-token [M,1] / per-group-col [M,G] · avx512_core     │
   └──────────────────────────────────────────────────────────┘
        │ success                              │ unimplemented
        ▼                                      ▼
   dynamic_quantize_t (x86 AVX-512)     ref_dynamic_quantize_t
        │  execute(): scan → scale/zp    (per-tensor / per-channel /
        │  → quantize (one fused pass)    per-group-row / compute-only)
        ▼
   dst + scale (+ zero_point) written
```

### 3.2 Operation semantics

The source is a 2D matrix `[M, N]` (rank note below). Granularity is encoded by the shape of `scale` (and, asymmetric, `zero_point`); formulas apply independently per group `g` (the
elements sharing one scale). Non-finite source values (`NaN` / `Inf`) are skipped in the reduction and quantize to 0, matching the vLLM's CPU quant path. Following
vLLM's CPU int8 kernel, the scale is **not** floored and the symmetric path uses the full `[-128, 127]` range; an all-zero group yields scale `0` (the caller's to avoid). The `u8`
output equals vLLM's signed form up to the 128 offset (`u8_dst = s8_dst + 128`, `zp = azp + 128`):

```
Symmetric  (dst = s8, zero-point implicitly 0):
  absmax(g) = max(|S[i]| : i ∈ g)
  scale(g)  = absmax(g) / 127                          ; f32, no floor
  dst[i]    = clamp(round_to_even(S[i] / scale(g)), -128, 127)
Asymmetric (dst = u8, zero-point computed):
  scale(g)  = (max(g) - min(g)) / 255                  ; f32, no floor
  zp(g)     = clamp(round(-min(g) / scale(g)), 0, 255) ; stored s32
  dst[i]    = clamp(round_to_even(S[i] / scale(g)) + zp(g), 0, 255)
```

Granularity is inferred from `scale.dims` (`G | N`, `Gr | M`); asymmetric `zero_point` has the same shape:

| Granularity           | `scale.dims`        | Optimized (x86) |
| --------------------- | ------------------- | --------------- |
| per-token (per-row)   | `[M, 1]`            | yes             |
| per-group-col         | `[M, G]`            | yes             |
| per-tensor            | `[1, 1]`            | reference       |
| per-channel (per-col) | `[1, N]`            | reference       |
| per-group-row         | `[Gr, N]`           | reference       |

The first PR is **2D-only** (`src` is `[M, N]`; rank `> 2` → `unimplemented`, rank `< 2` → `invalid_arguments`). These names are the primitive's own, since granularity is read from the
`scale` shape; oneDNN's attribute-based static quant instead names it with `mask` + `groups` (dev guide "Quantization"), and the correspondence is: per-tensor → `mask 0`;
per-token `[M, 1]` → per-dimension (`mask 1<<dim`) on the `M` (row) axis; per-channel `[1, N]` → per-dimension on the `N` (col) axis; per-group-col `[M, G]` / per-group-row `[Gr, N]` → grouped
(`groups {G0, G1}` over the last two dims) with `groups {1, N/G}` / `{M/Gr, 1}`.

### 3.3 Public C++ API

Add `dnnl::dynamic_quantize` and its `primitive_desc` to `include/oneapi/dnnl/dnnl.hpp`; the constructor forwards to a new C entry `dnnl_dynamic_quantize_primitive_desc_create`. `scale`
/ `zero_point` are first-class output memory descriptors and `compute_only` is a flag.

```cpp
struct dynamic_quantize : public primitive {
    struct primitive_desc : public primitive_desc_base {
        primitive_desc() = default;
        primitive_desc(const engine &aengine, algorithm aalgorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::desc &scale_desc,
                const memory::desc &zero_point_desc,
                dynamic_quantize_flags aflags = dynamic_quantize_flags::none,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
        memory::desc src_desc()        const;
        memory::desc dst_desc()        const;
        memory::desc scale_desc()      const;  // = dst_desc(1)
        memory::desc zero_point_desc() const;  // = dst_desc(2)
    };
    dynamic_quantize() = default;
    dynamic_quantize(const primitive_desc &pd);
    dynamic_quantize(const primitive_desc &pd, const std::vector<uint8_t> &blob);
};
```

New enums: `algorithm::dynamic_quantize_{symmetric,asymmetric}` and `dynamic_quantize_flags::{none, compute_only}`. Arguments `DNNL_ARG_SRC`, `DNNL_ARG_DST`, `DNNL_ARG_DYNAMIC_SCALE` (=
`DNNL_ARG_DST_1`), and `DNNL_ARG_DYNAMIC_ZERO_POINT` (= `DNNL_ARG_DST_2`) reuse the `DST_1` / `DST_2` slots as oneDNN already does for auxiliary outputs (e.g. RNN's `DST_ITER`).
Validation (`src/common/dynamic_quantize.cpp`): `src` is 2D `[M, N]` (rank `> 2` → `unimplemented`), concrete format, dtype `{f32, bf16, f16}`; `scale` `f32`; symmetric ⇒ `dst = s8`, no
zero-point; asymmetric ⇒ `dst = u8`, `zero_point = s32` shaped like `scale`; `scale.dims` matches one granularity template (per-group divisors divide the axis); `compute_only` ⇒ empty
`dst`. Only `scratchpad_mode` is accepted. The API is additive (no ABI break) and introduced as **experimental** (§3.8).

### 3.4 x86 CPU registration and kernels

`cpu_engine.hpp` routes `primitive_kind::dynamic_quantize` to a new x86 impl list (`src/cpu/cpu_dynamic_quantize_list.cpp`) that registers the optimized x86 kernel ahead of the reference:

```cpp
constexpr impl_list_item_t impl_list[] = REG_DYNAMIC_QUANTIZE_P({
        CPU_INSTANCE_X64(dynamic_quantize_t)   // x86 AVX-512 intrinsics port
        CPU_INSTANCE(ref_dynamic_quantize_t)   // portable reference
        nullptr,
});
```

`x64::dynamic_quantize_t` (`src/cpu/x64/dynamic_quantize.{hpp,cpp}` + `dynamic_quantize_kernels_per_{token,group}.cpp`) wraps the ported **x86 AVX-512** fused kernels with the intrinsic
bodies verbatim. `pd_t::init()` accepts only 2D dense row-major (`ab`), `f32/bf16/f16 → s8/u8`, per-token or per-group-col, on `avx512_core` (x86 ISA ladder
`avx512_core_fp16 → avx512_core_bf16 → avx512_core`; `f16` uses F16C). Only threading/glue are oneDNN-native: the original OpenMP driver becomes `parallel_nd` (per-row for per-token,
over `M x G` for per-group), scalar `bf16` /
`f16` conversions use oneDNN helpers. It is intentionally **not** JIT (no shape-dependent code-gen; plain `__attribute__((target(...)))` intrinsics, no Xbyak); a JIT impl can layer
above it later behind the same API. Anything it does not accept returns `unimplemented`. `ref_dynamic_quantize_t` (`src/cpu/ref_dynamic_quantize.{hpp,cpp}`) reads algorithm and
granularity from the descriptor at execute time, so one class is correct for all five granularities, both algorithms, and `compute_only`; it parallelizes over scale elements with
`parallel_nd` and is the only path for per-tensor / per-channel / per-group-row / compute-only in this PR.

### 3.5 Compute-only mode

With `compute_only` set, the primitive performs only the scan and writes `scale` (and `zero_point`); `dst` is a zero memory descriptor and `DNNL_ARG_DST` is omitted at execute. This
lets a framework compute parameters once and reuse them across downstream consumers. In the first PR the reference serves this mode.

### 3.6 No disruption to existing paths

`reorder`'s contract and the static-quantization attribute machinery are untouched and no third-party dependencies are added. Build integration is minimal: x86 x64 sources are picked up
by the existing glob and gated by architecture via `CPU_INSTANCE_X64` / `DNNL_X64`, with the whole primitive behind the opt-in experimental flag (§3.8), so a default build is unchanged.
The one compiler-specific point is that the ported x86 kernels use GCC/Clang-style `__attribute__((target(...)))`, so on MSVC the x64 translation unit is guarded out at compile time and
only the reference is registered (see §3.7).

### 3.7 Compiler and toolchain requirements

No new language baseline is introduced: the public API and the `ref_dynamic_quantize_t` reference are portable **C++11** and build on every compiler in oneDNN's Validated Configurations
— GCC 8.5+, Clang 11.0+ (Apple LLVM 15), Intel oneAPI DPC++/C++ (ICX) 2025.1, MSVC 19.43 (Visual Studio 2022) — so the primitive is functionally available (via `ref:any`) on the full
compiler matrix without raising the floor. The optimized x86 (x64) kernels add one requirement: a compiler supporting the GNU-style per-function `__attribute__((target(...)))` attribute
together with x86 AVX-512F / BW / VL and F16C intrinsics. No `avx512_bf16` / `avx512_fp16` ISA is needed (`bf16` uses an AVX-512F bit-shift shim, `f16` uses F16C), so the requirement is
met
by the GNU, LLVM, and Intel toolchains oneDNN already validates:

| Toolchain                                  | Min. version (optimized x86)   | Notes                                                               |
| ------------------------------------------ | ------------------------------ | ------------------------------------------------------------------- |
| GCC                                        | 8.5 (oneDNN's validated floor) | `target` attribute + AVX-512F/BW/VL intrinsics need ≥ 5.0           |
| Clang / Apple LLVM                         | 11.0 / 15.0                    | Linux / macOS                                                       |
| Intel oneAPI DPC++/C++ (ICX) / classic ICC | 2025.1 / 19.x                  | oneDNN validates ICX; both accept GCC/Clang-style attributes        |
| MSVC                                       | not supported                  | no `__attribute__((target))`; x64 TU is guarded out, uses `ref:any` |

On GCC/Clang/Intel the primitive dispatches to `jit:avx512_core` for the optimized x86 granularities; on MSVC (or any compiler without the target attribute) the optimized x86 kernels are
excluded at compile time and the portable reference (`ref:any`) handles every case. A future Xbyak JIT impl (see §7) would side-step the attribute entirely and restore the optimized x86
path on MSVC.

### 3.8 Experimental status

The primitive is proposed as an **experimental** API, gated behind an opt-in build flag `ONEDNN_EXPERIMENTAL_DYNAMIC_QUANTIZE` (macro `DNNL_EXPERIMENTAL_DYNAMIC_QUANTIZE`), default
**OFF**, following oneDNN's existing `DNNL_EXPERIMENTAL_*` features (grouped memory, ukernel, profiling): the primitive kind, public headers, impl-list entry, and kernels compile only
when enabled, so a default build is byte-for-byte unchanged. While experimental the API is **not** covered by oneDNN's backward-compatibility guarantee, so the op descriptor, enums,
argument layout, and granularity encoding can still change from integration feedback before promotion to a stable primitive.

## 4. PoC - dynamic-quantize primitive on x86 CPU

The public primitive was validated end-to-end on x86 CPU: it builds through public oneDNN headers, dispatches to the ported x86 AVX-512 kernels for per-token and per-group, and falls
through to the reference for the other granularities and compute-only.

**Verbose evidence.**

```
onednn_verbose,v1,primitive,exec,cpu,dynamic_quantize,jit:avx512_core,undef,
  src:f32::blocked:ab dst:s8::blocked:ab scale:f32::blocked:ab,,
  alg:dynamic_quantize_symmetric,8x128
onednn_verbose,v1,primitive,exec,cpu,dynamic_quantize,jit:avx512_core,undef,
  src:f32::blocked:ab dst:u8::blocked:ab scale:f32::blocked:ab
  zp:s32::blocked:ab,,alg:dynamic_quantize_asymmetric,8x128
```

Per-token cases show a `[M,1]` scale; per-group uses `[M, G]`. `zp:s32` appears only for the asymmetric algorithm, and `jit:avx512_core` confirms dispatch to the ported x86 AVX-512
kernels.

**Accuracy.** A benchdnn `--dynamic-quantize` driver exercises per-token and per-group, symmetric (`s8`) and asymmetric (`u8` + `s32` zero-point), `f32` and `bf16` source, comparing the
quantized `dst` against an independent re-quantization reference. The CI input set passes with 100% of x86 x64-eligible cases on `jit:avx512_core`; per-tensor / per-channel / per-group-row
/ compute-only pass on `ref:any`. gtest additionally covers descriptor creation, queries, invalid arguments, and small-shape numerics for every granularity / algorithm / dtype
combination.

## 5. Framework-Side Changes

Backends map their existing dynamic-quant activation step to `dnnl::dynamic_quantize`: pass src, dst, scale, and optional zero-point memory descriptors plus the algorithm, instead of a
reduction followed by a separate quantize, then feed the produced `scale` / `zero_point` into the downstream `matmul` as `attr.scales` / `attr.zero_points`. Applications need no source
changes — only the backend mapping changes, and later oneDNN optimizations reuse it.

## 6. First PR - Scope and Acceptance Criteria

The first PR is complete when, on x86 CPU: public `dnnl::dynamic_quantize` construction and execution work through public headers only, for symmetric and asymmetric algorithms and full /
compute-only modes; the x86 AVX-512 port is registered ahead of the reference and appears as `cpu,dynamic_quantize,jit:avx512_core` for per-token and per-group on an x86 AVX-512 host,
while
`ref_dynamic_quantize_t` serves per-tensor / per-channel / per-group-row / compute-only as `ref:any` with correct scale / zero-point / dst; forward quantization works for `f32` / `bf16`
/ `f16` → `s8` (symmetric) / `u8` (asymmetric) with `f32` scale and `s32` zero-point, 2D-only (rank `> 2` → `unimplemented`); benchdnn `--dynamic-quantize` covers the CI input set
against an independent reference, gtests cover creation, queries, invalid arguments, and execute smoke cases, and a tutorial example and developer-guide page are included; and the
primitive kind, public API, impls, and registration compile under the opt-in `ONEDNN_EXPERIMENTAL_DYNAMIC_QUANTIZE` flag (default OFF), leaving a default build unchanged, documented as
experimental.

Follow-up: grouped / MoE `group_dynamic_quant` scheduler, x86 topology-aware threading and the split single-token decode kernel (the port currently uses plain `parallel_nd`),
`bf16` / `f16` scale output, and sub-byte / FP8 destinations.

## 7. Alternatives Considered

- Extend `reorder` with a `dynamic_quant` mode. This inverts the meaning of `attr.scales` (caller input → library output) under a flag, mixes reduction-then-cast kernels into
  `reorder`'s impl list, and makes `compute_only` a reorder with no destination. `batch_normalization` is the precedent — its `mean` / `variance` flip between input and output under a
  flag, and oneDNN made it its **own** primitive — so dynamic quant follows the same pattern with dedicated output MDs; rejected.
- Tag `attr.scales` "compute-from-input" on the consumer (e.g. `matmul`). Tightest fusion, but locks computation to one consumer and blocks reuse of the quantized activation across
  consumers; out of scope for this RFC.
- Port as an Xbyak JIT primitive. No shape-specialization win over the intrinsic kernels, which keep the diff small and reviewable; a JIT impl can follow behind the same API.

## 8. Open Questions

- Granularity is inferred from `scale.dims`. Reviewers may prefer an explicit `granularity_t` field; it can be added later with an `auto`-infer default without breaking the API.
- First-PR scope: should the first PR start with the **portable reference only** (every granularity on `ref:any`, establishing the API and correctness) and add the optimized x86 x64
  per-token / per-group kernels in a follow-up, or land the API and the optimized x86 kernels together, as currently proposed? Reference-first keeps the initial diff small and
  API-focused; landing both together delivers the performance win immediately.
- The grouped / MoE `group_dynamic_quant` scheduler is the natural next port and can layer on the single-matrix kernels proposed here.

[vllm-quant]: https://github.com/vllm-project/vllm/blob/c44e191b014db0619bd51921e94c86b901ab952e/csrc/cpu/dnnl_kernels.cpp#L468
