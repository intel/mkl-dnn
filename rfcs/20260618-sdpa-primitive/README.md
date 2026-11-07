# RFC: Public SDPA Primitive Support in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC adds Scaled Dot-Product Attention (SDPA) as a public oneDNN primitive,
`dnnl::sdpa`, backed first by a portable CPU reference implementation.

oneDNN already supports SDPA through the Graph API as an MHA fusion pattern, and
it also has internal SDPA primitive plumbing used by tests and GPU code.
However, primitive API users do not have a supported CPU entry point comparable
to `dnnl::matmul`. Frameworks that integrate with oneDNN primitives must either
build a graph pattern or rely on non-public interfaces.

The first PR promotes the existing primitive plumbing to a public C++ API, adds
the missing CPU reference implementation, and validates the behavior needed by
framework tensors: explicit additive and causal masks, plus stride-driven
handling for non-contiguous layouts such as transposed views and packed
fused-QKV buffers. Future optimized SDPA implementations can use the same API.

## 1. Motivation

Attention is a core operation in transformer inference, and frameworks such as
PyTorch, vLLM, and ONNX Runtime expose SDPA as a single fused operation. A
public `dnnl::sdpa` primitive lets oneDNN own that operation directly, instead
of leaving it to be reassembled from smaller ops on the framework side.

The main reason to own it is performance. There is no single best SDPA
implementation: Flash Attention-style kernels are better for some shapes, while
BMM-Softmax-BMM kernels are better for others (BRGEMM on Intel CPUs, Zen-tuned
BMM on AMD CPUs). The right choice depends on sequence length, head size, mask,
data type, and hardware. As a primitive, oneDNN can pick the kernel, blocking,
tiling, and threading in one place, the same way it already does for `matmul`.
This is CPU-specific tuning, since AMD and Intel differ in cache hierarchy, core
topology, and memory behavior.

Keeping SDPA in the library then gives:

- one set of heuristics for Flash Attention-style and BMM-Softmax-BMM paths;
- platform-specific tiling and threading behind the same public API;
- one integration path for PyTorch, vLLM, llama.cpp, and other oneDNN users;
- support for new ISAs, CPU generations, and customer-tuned paths without
  framework changes.

On the PyTorch path, ZenDNN already shows the payoff: `zentorch` rewrites
`aten::scaled_dot_product_attention` to ZenDNN SDPA and gets up to 15% geomean
improvement on Torch Inductor dashboard models on AMD, with only the SDPA
rewrite enabled.

Operator-level numbers on AMD Turin show why the choice should sit in the
library. For the same dashboard shapes, Flash Attention wins on some cases and
the ZenDNN BMM-Softmax-BMM parallel-primitive path wins on others. On geomean
the BMM-Softmax-BMM path is ahead of ZenDNN Flash Attention by 1.13x on 64
cores and 1.08x on 128 cores.

vLLM is one concrete consumer. Encoder and encoder-only models (BERT-style
encoders, sentence-transformer embeddings, rerankers) are heavy CPU workloads,
and their attention runs as SDPA without a KV cache. vLLM's
[CPU attention backend][vllm-cpu-attn] already calls native kernels through
[`vllm._custom_ops`][vllm-custom-ops], so a public `dnnl::sdpa` can be called
directly with Q/K/V, mask, scale, and causal arguments, without PyTorch SDPA in
the path.

llama.cpp follows the same library-owned model. It treats attention as one
fused op, `ggml_flash_attn_ext`
([PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)), with a dedicated
CPU implementation in ggml. Rather than rebuilding attention from separate
MatMul and Softmax calls, it keeps the CPU chunking, tiling, and threading
inside the library, modeled on ggml's own MatMul chunking
([PR #16829](https://github.com/ggml-org/llama.cpp/pull/16829)). That is the
ownership model this RFC proposes for oneDNN.

## 2. Non-Goals

- Do not replace the Graph API MHA fusion path.
- Do not add quantized SDPA, GQA, custom scales, or backward propagation in the
  first PR.
- Do not change the existing internal GPU SDPA implementation.

## 3. Proposal - public SDPA primitive

The design reuses the existing SDPA primitive infrastructure instead of defining
a new operation model. It adds a public header API, a CPU implementation-list
entry, and the CPU kernel; consumers that do not call `dnnl::sdpa` are
unaffected.

### 3.1 Architecture overview

```
framework (PyTorch SDPA / zentorch / app)
        │  Primitive API
        ▼
dnnl::sdpa::primitive_desc(eng, q, k, v, o[, mask, attn_mask_type])
        │  → ::sdpa_primitive_desc_create(...)   (existing internal C entry)
        ▼  CASE(sdpa) → get_sdpa_impl_list → walk impl_list
   ┌──────────────────────────────────────────────┐
   │ ref_sdpa_fwd_t::pd_t::init()  (CPU)          │
   │   4D · f32/bf16 · mask f32/bf16 2D/4D ·       │
   │   default scale · softmax_accurate            │
   └──────────────────────────────────────────────┘
        │ success                         │ unimplemented
        ▼                                 ▼
   dnnl::sdpa(pd) → ref_sdpa_fwd_t   (next impl / fall-through)
        │  execute(): scores=scale·QKᵀ → mask → softmax → ·V
        ▼
   dst written
```

### 3.2 Operation semantics

SDPA computes:

```
dst = matmul( softmax( matmul(Q, Kᵀ) * scale + mask ), V )
```

The first PR supports 4D BHSD tensors: Q and dst are `(N,H,Sq,D)`, K and V are
`(N,H,Skv,D)`. K is passed with the descriptor needed for the logical transpose
used by `Q * K^T`.

### 3.3 Public C++ API

Add `dnnl::sdpa` and its `primitive_desc` to
`include/oneapi/dnnl/dnnl.hpp`. Both constructors forward to the existing C
entry `sdpa_primitive_desc_create`. Use two constructor forms: one without a
mask and one with an explicit mask descriptor.

```cpp
struct sdpa : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        // No explicit mask (common case).
        primitive_desc(const engine &aengine,
                const memory::desc &query_desc, const memory::desc &key_desc,
                const memory::desc &value_desc, const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        // Explicit additive buffer mask and/or causal mask.
        primitive_desc(const engine &aengine,
                const memory::desc &query_desc, const memory::desc &key_desc,
                const memory::desc &value_desc, const memory::desc &dst_desc,
                const memory::desc &mask_desc, int attn_mask_type,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
        // query/key/value/dst_desc() accessors
    };
    sdpa() = default;
    sdpa(const primitive_desc &pd) : primitive(pd) {}
    sdpa(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob);
};
```

`attn_mask_type` follows `dnnl_attn_mask_type_t`. The first PR supports no mask,
explicit additive masks, and top-left causal masks.

### 3.4 CPU registration and reference kernel

`cpu_engine.hpp` routes `primitive_kind::sdpa` to a new implementation list
(`src/cpu/cpu_sdpa_list.cpp`) that registers the reference kernel:

```cpp
constexpr impl_list_item_t impl_list[] = REG_SDPA_P({
        CPU_INSTANCE(ref_sdpa_fwd_t)
        nullptr,
});
```

`ref_sdpa_fwd_t` supports forward inference for 4D Q/K/V/dst with uniform `f32`
or `bf16` data types, default scale, and `softmax_accurate`. It computes in
`f32`; `bf16` is converted at load/store boundaries. Unsupported cases return
`unimplemented` so later optimized implementations can be registered before the
reference kernel.

### 3.5 Masking

The first PR supports:

- additive masks: `f32` or `bf16`, 2D `(Sq,Skv)` or 4D `(N,H,Sq,Skv)`, with
  broadcast through strides;
- top-left causal masks;
- additive and causal masks together.

### 3.6 Stride-aware layout handling

Framework tensors are often non-contiguous because of transposes or fused-QKV
packing. The implementation uses memory descriptor strides for Q/K/V/dst rather
than assuming dense layout, so dense, transposed, and packed fused-QKV layouts
are handled correctly.

## 4. PoC - SDPA primitive end-to-end on Torch Inductor dashboard models

The public primitive was validated end-to-end through the ZenDNN `zentorch`
plugin by routing PyTorch SDPA to `sdpa_direct`, then to `dnnl::sdpa` and
`ref_sdpa_fwd_t` on CPU.

### 4.1 Verbose evidence (masked SDPA, packed-QKV layout)

```
onednn_verbose,v1,primitive,exec,cpu,sdpa,ref:any,forward_inference,
  query:f32::blocked:acbd:48000x64x384x1 key:f32::blocked:adbc:48000x64x1x384
  val:f32::blocked:acbd:48000x64x384x1 msk:f32::blocked:abcd
  dst:f32::blocked:acbd,
  ,alg:softmax_accurate msk:2d, 1x2x125x64:1x2x64x125:1x2x125x64, 2.62
```

The explicit Q/K/V strides (`...x384x1`) are the fused-QKV packing; `msk:` shows
the additive mask flowing through the primitive. Both are handled by the
stride-aware kernel.

### 4.2 Accuracy (Torch Inductor dashboard models, including BERT-QA)

- **f32:** Torch Inductor dashboard model validation passes.
- **bf16:** Torch Inductor dashboard model validation passes within the expected
  tolerance.

### 4.3 What is validated

- Public `dnnl::sdpa` builds, dispatches to `ref_sdpa_fwd_t` on CPU, executes.
- Masked / causal / no-mask, f32 / bf16, self- and cross-attention.
- Packed/transposed (fused-QKV) layouts produce correct results.

## 5. Framework-Side Changes

Framework backends can map their existing SDPA operator to `dnnl::sdpa`. They
pass Q/K/V/dst memory descriptors, an optional additive mask descriptor, and the
attention-mask type to the primitive descriptor.

Applications need no source changes: they keep calling the framework SDPA API
(for example PyTorch `aten::scaled_dot_product_attention`), and only the backend
mapping changes. Later oneDNN SDPA optimizations reuse the same mapping.

## 6. First PR - Scope and Acceptance Criteria

The first PR is complete when the following are in place and passing on CPU:

- Public `dnnl::sdpa` construction and execution work through public oneDNN
  headers only, for both no-mask and masked forms.
- `ref_sdpa_fwd_t` is registered in the CPU implementation list and appears in
  verbose output as `cpu,sdpa,ref:any`.
- Forward inference works for 4D BHSD `f32` and `bf16` tensors with default
  scale and `softmax_accurate`.
- Additive masks, top-left causal masks, and combined mask cases match the
  reference within tolerance.
- Dense, transposed, and packed fused-QKV layouts produce correct results using
  memory-descriptor strides.
- benchdnn `--sdpa` covers the CI input set with an independent `f32`
  reference.
- gtests cover descriptor creation, queries, invalid arguments, and execute
  smoke cases.

Follow-up work includes optimized CPU kernels, bottom-right causal masks, GQA,
custom scales, backward propagation, and additional bf16 numerics tuning.

## 7. Alternatives Considered

- Keep using only the Graph API. This avoids a new primitive API, but still
  requires graph construction for users that want one SDPA primitive call.
- Keep SDPA internal. This avoids a public API commitment, but leaves framework
  users without a supported CPU primitive path.
- Add a public primitive with a CPU reference implementation. This is the
  proposed option because it gives frameworks a stable primitive entry point and
  leaves optimized implementations as follow-up work behind the same API.

## 8. Open Questions

- Future optimized dispatch policy: should Intel and AMD CPUs use separate
  policies for BMM-Softmax-BMM and Flash Attention-style paths, or share one
  policy where possible?
- First PR scope: should the first PR stop at the public API and CPU reference
  path, or also include an optimized BMM-Softmax-BMM implementation?

[vllm-cpu-attn]: https://github.com/vllm-project/vllm/blob/e9f331d7/vllm/v1/attention/backends/cpu_attn.py
[vllm-custom-ops]: https://github.com/vllm-project/vllm/blob/b00e76ff/vllm/_custom_ops.py

