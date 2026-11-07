# RFC: Public Embedding Bag Primitive in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC aims to improve Embedding Bag (EMB) functionality improvement in various frameworks.

oneDNN currently has no direct primitive equivalent to Embedding and Embedding Bag (for example,
PyTorch Embedding Bag). Framework users that integrate with oneDNN either need to depend on
its framework implementation, or compose the operator using graph level API and multiple primitives.
Such composition is complex and sub-optimal.

The first PR promotes the EMB plumbing to a public C++ API, adds the CPU
reference implementation, and validates the behavior needed by recommendation
and NLP workloads: sum / mean / max reduction over variable-length bags,
optional per-sample weights, `padding_idx` skipping, and the
`include_last_offset` flag matching PyTorch semantics.

Future PRs include adding optimized implementations using OneDNN dense-sparse matrix multiplication
(DSMM) based kernels that reuse the same API without any framework changes.

## 1. Motivation

Embedding lookups and bag reductions are the dominant CPU operation in
recommendation models (DLRM, DCNv2, MMoE) and token-id-consuming NLP models.
[PyTorch nn.EmbeddingBag](https://docs.pytorch.org/docs/2.13/generated/torch.nn.EmbeddingBag.html),
and [TensorFlow embedding_lookup_sparse](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse),
each implement this on CPU today — either
through per-framework hand-rolled kernels or FBGEMM — because oneDNN offers no
equivalent primitive.

The main reason to own it in oneDNN is performance and consolidation. A
high-quality embedding kernel requires
- careful work splitting (table-threaded vs bag-threaded),
- int32/int64 index handling,
- optional per-sample weights without branchy inner loops, and
- ISA-specific load/scatter strategies (AVX-512 BF16, AVX-512-FP16).

Different frameworks repeatedly re-implement this. Centralizing it in
oneDNN keeps the optimizations in one place and gives AMD and Intel a single
tuning target.

Effect of the above threading strategies is measured on the primitive performance. The above
strategies improve the primitive latency performance by 1.2X to 1.5X on native PyTorch performance, as
shown on few representative loads in the table below.

| Precision | Tbl Sz | Embed Dim| Batch Sz | PyTorch Latency (ms) | ZenDNN Latency (ms) | Speedup |
|---|---|---|---|---|---|---|
| fp32 | 40000000 | 128 | 4096 | 0.0458 | 0.0300 | 1.53x |
| bf16 | 40000000 | 128 | 4096 | 0.0452 | 0.0287 | 1.58x |

Throughput gain in MLPERF DLRM V2, having 26 embedding bags, and running 127 instances
is given below for few representative loads.

| Precision | Bach Size | PyTorch Throughput (QPS)| ZenDNN Throughput (QPS) | Speedup |
|---|---|---|---|---|---|---|
| fp32 | 512 | 1035 | 1931 | 1.87x |
| bf16 | 512 | 3480 | 3858 | 1.11x |

In production grade DLRM code, the embedding tables are more than 100 so we expect
a performance gain of ~15% due to this operator on such networks.

First PR only aims to add only reference kernel. Subsequent PRs aim to add optimized kernels with vector
intrinsics (AVX-512 / AVX-512-FP16) from AMD's ZenDNN, which ships these highly
optimized  implementations through its internal
`lowoha::embag` API. This RFC aims to upstreams those intrinsic kernels into oneDNN as a
first-class primitive lifted into `src/cpu/x64/`, with ZenDNN removed from the
runtime path after integration. This PR specifically DOES NOT aim to add the following.

- backward / training support.
- low precision (`bf16` / `fp16`) and quantized (`s8` / `s4` / `u4`)
  embedding tables in the first PR.
- high precision (`s64` / `u64`) indexing in the first PR.
- optimized (AVX-512 / AVX512-FP16) implementations in the first PR.
- a GPU implementation in the first PR.
- replace any existing oneDNN primitive or Graph API path.

## 3. Proposal — public embedding_bag primitive

The design follows the same skeleton used by other oneDNN primitives. It adds
a public header API, a CPU implementation-list entry, and the CPU reference
kernel; consumers that do not call `dnnl::embedding_bag` are unaffected.

This primitive will be addded as an `Experimental Features`
(https://uxlfoundation.github.io/oneDNN/dev_guide_experimental.html) with appropriate
build and runtime controls.

### 3.1 Architecture overview

```
framework (PyTorch nn.EmbeddingBag / zentorch / app)
        │  Primitive API
        ▼
dnnl::embedding_bag::primitive_desc(eng, alg, src, indices, offsets,
                                    weights, dst[, padding_idx, flags])
        │  → dnnl_embedding_bag_primitive_desc_create(...)
        ▼  CASE(embedding_bag) → get_embedding_bag_impl_list → walk impl_list
   ┌──────────────────────────────────────────────────────┐
   │ ref_embedding_bag_t::pd_t::init()  (CPU)             │
   │   2D table · f32 · s32/s64 indices · sum/mean/max/   │
   │   lookup · optional weights / padding_idx            │
   └──────────────────────────────────────────────────────┘
        │ success                         │ unimplemented
        ▼                                 ▼
   dnnl::embedding_bag(pd)           (next impl / fall-through)
        │  execute(): Y[b,:] = REDUCE over bag_b of T[I[k],:] * w[k]
        ▼
   dst written
```

### 3.2 Operation semantics

Let T be a 2D embedding table `[V, D]`, I a 1D indices vector of length N, O a
1D offsets vector defining B bags, and `algo` one of {sum, mean, max, lookup}.

**Lookup mode** (no offsets, no reduction):

```
Y[n, :] = T[I[n], :]          for n in [0, N),  shape Y = [N, D]
```

**Bag modes** (sum / mean / max):

```
bag_b   = { I[k] : O[b] <= k < O[b+1] }
Y[b, :] = REDUCE_algo over bag_b of  T[I[k], :] * w[k]
shape Y = [B, D]
```

- `w[k]` = `per_sample_weights[k]` if provided, else `1.0`
- Indices equal to `padding_idx` are skipped
- `max` mode does not combine with per-sample weights (matches PyTorch)
- When `include_last_offset == true`, `O` has length `B+1` and `O[B]` is read
  from data; otherwise the implicit terminator `O[B] = N` is used

#### 3.2.1 Comparison with PyTorch nn.EmbeddingBag semantics

PyTorch nn.EmbeddingBag semantics are given in [PyTorch nn.EmbeddingBag](https://docs.pytorch.org/docs/2.13/generated/torch.nn.EmbeddingBag.html).

The operator matches PyTorch nn.EmbedddingBag semantics except the following

- The operator supports `per-sample weights` for all three reduction methods
  (sum, mean and max). By contrast PyTorch `nn.EmbeddingBag` supports per-sample
  weights only for `sum`.

#### 3.2.2 Comparison with TensorFlow nn.embedding_lookup_sparse semantics

TensorFlow nn.embeddding_lookup_sparse semantics are given in
[tf.nn.embedding_lookup_sparse](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse)

The operator can be used to implement it by mapping its inputs to the operator as follows

- params can be mapped to the embedding table.
- indices and offsets can be derived from sp_ids sparse matrix.
- per-sample weights can be derived from sp_weights.
- combiner is `sum` or `mean`.
- padding_idx is -1.
- include_last_offset is `false`.

### 3.3 Public C++ API

Add `dnnl::embedding_bag` and its `primitive_desc` to
`include/oneapi/dnnl/dnnl.hpp`. The constructor forwards to the new C entry
`dnnl_embedding_bag_primitive_desc_create`.

```cpp
struct embedding_bag : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        // No weights
        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &offsets_desc,
                const memory::desc &dst_desc,
                int64_t  padding_idx = -1,
                bool  include_last_offset = false,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        // With weights
        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &offsets_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                int64_t  padding_idx = -1,
                bool  include_last_offset = false,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
    };

    embedding_bag() = default;
    explicit embedding_bag(const primitive_desc &pd) : primitive(pd.get()) {}
};

```

New algorithm values in `dnnl_alg_kind_t` (`0x40000` band to avoid collisions):
`dnnl_embedding_bag_sum`, `dnnl_embedding_bag_mean`, `dnnl_embedding_bag_max`,
`dnnl_embedding_lookup`.

### 3.4 Argument map

| Argument | Memory | When required |
|---|---|---|
| `DNNL_ARG_WEIGHTS` | embedding table `[V, D]` | always |
| `DNNL_ARG_SRC` | indices `[N]`, `s32` | always |
| `DNNL_ARG_SRC_1` | offsets `[B]` or `[B+1]`, same dtype as indices | bag modes only |
| `DNNL_ARG_SRC_2` | per-sample weights `[N]`, `f32` | optional; ingnored if `max` |
| `DNNL_ARG_DST` | output `[B, D]` (bag) or `[N, D]` (lookup) | always |

### 3.5 CPU registration and reference kernel

`cpu_engine.hpp` routes `primitive_kind::embedding_bag` to a new implementation
list (`src/cpu/cpu_embedding_bag_list.cpp`):

```cpp
constexpr impl_list_item_t impl_list[] = REG_EMBEDDING_BAG_P({
        CPU_INSTANCE_X64(embedding_bag_t)   // AVX-512 intrinsic impl (Phase 2)
        CPU_INSTANCE(ref_embedding_bag_t)   // portable C++ reference
        nullptr,
});
```

`ref_embedding_bag_t` supports forward inference for all four algorithms with
`f32` table / output, `s32` indices, and optional per-sample weights.Unsupported
cases return `unimplemented` so that later optimized implementations can be registered above
the reference kernel in the impl-list.

### 3.6 Validation rules

Enforced in `embedding_bag_desc_init` and `embedding_bag_pd_t::init_*`:

1. `src_desc` is 2D, blocked, dtype in the supported set (Phase 1: `f32`).
2. `indices_desc` is 1D; dtype in {`s32`}.
3. Bag modes: `offsets_desc` is 1D, same dtype as indices, length B (or B+1
   with `include_last_offset`); `dst_desc` is 2D `[B, D]`.
4. Lookup mode: `offsets_desc` and `weights_desc` must be empty (`zero_md`);
   `dst_desc` is `[N, D]`.
5. If `weights_desc` is non-empty: `f32`, length N, and
   `alg_kind != embedding_bag_max`.
6. `padding_idx` in `[-1, V)`; `-1` means "no padding".
7. All MDs must be fully described.
8. `dst_desc.dims[1] == src_desc.dims[1]` (embedding dim matches).

## 4. PoC — embedding_bag primitive with a benchdnn driver

The public primitive was validated end-to-end through the benchdnn
`--embedding-bag` driver, running against the CPU reference implementation.

### 4.1 Verbose evidence (f32 sum, with weights and padding_idx)

```
onednn_verbose,v1,primitive,exec,cpu,embedding_bag,ref:any,forward_inference,
  src:f32::blocked:ab:100000x128 idx:s32::blocked:a:1000
  off:s32::blocked:a:64 wts:f32::blocked:a:1000
  dst:f32::blocked:ab:64x128,
  ,alg:embedding_bag_sum pad:0 flags:, 1x64x128:1x1000:1x64:1x1000:1x64x128, 3.14
```

### 4.2 Accuracy

- **f32 sum / mean / max:** benchdnn correctness driver (`--mode=C`) passes for
  basic shapes and DLRM-representative shapes.
- **f32 lookup:** passes for index sets with and without `padding_idx`.
- **Edge cases:** all-padding bag returns zero (matches PyTorch); single-element
  bag; `include_last_offset` with explicit `O[B]` entry.

### 4.3 What is validated

- Public `dnnl::embedding_bag` builds, dispatches to `ref_embedding_bag_t` on
  CPU, executes.
- Sum / mean / max / lookup algorithms, with and without per-sample weights.
- `padding_idx` skipping and zero output for fully-padded bags.
- `include_last_offset` flag with explicit and implicit last-offset semantics.
- `s32` index dtypes.

## 5. Framework-Side Changes

Framework backends can map their existing embedding operator to
`dnnl::embedding_bag`. They pass table, indices, offsets, and optional weights
memory descriptors plus `padding_idx` and `flags` to the primitive descriptor.

Applications need no source changes: they keep calling the framework API (for
example PyTorch `nn.EmbeddingBag`), and only the backend mapping changes. Later
oneDNN embedding optimizations reuse the same mapping.

## 6. First PR — Scope and Acceptance Criteria

The first PR is complete when the following are in place and passing on CPU:

- Public `dnnl::embedding_bag` construction and execution work through public
  oneDNN headers only.
- `ref_embedding_bag_t` is registered in the CPU implementation list and appears
  in verbose output as `cpu,embedding_bag,ref:any`.
- Forward inference works for `f32` table/output with `s32` indices
  for all four algorithms.
- Per-sample weights (sum / mean), `padding_idx`, and `include_last_offset` work
  correctly and match PyTorch reference outputs.
- benchdnn `--embedding-bag` covers the CI input set with an independent `f32`
  reference.
- gtests cover descriptor creation, accessors, validation errors, and execute
  smoke cases.

Follow-up work includes the AVX-512 intrinsic kernel (Phase 2), BF16/FP16
paths, quantized tables (Phase 3), and GPU support (future).

## 8. Alternatives Considered

- Keep using framework-native kernels (FBGEMM, hand-rolled). Avoids a new
  primitive API, but each framework carries its own embedding kernel with no
  shared ISA tuning.
- Implement via Graph API gather + reduction fusion. Does not cover
  variable-length bag semantics or per-sample weights without a bespoke graph
  pattern.
- Implement using existing oneDNN sparse matrix multiplication

  Both embedding bag (EMB), with `sum` as reduction method, algorithmically resembles
  dense-sparse matrix multiplication (DSMM), where embedding tables act as a dense
  matrix, and indices-offsets pair with per-sample-weights could be taken
  as a sparse matrix.

  This implementation is tried out and benchmarking results are compared with
  ZenDNN implementation in the following tables

  - Latency with single thread, embedding size 128

  | Table Sz | Bach Sz | ZenDNN (ms)| PyTorch (ms)| MKLDNN (ms)|
  |---|---|---|---|---|
  | 4 | 4096 | 0.0455 | 0.0454 | 0.0908 |
  | 1000 | 4096 | 0.0409 | 0.0411 | 0.1119 |
  | 10000 | 4096 | 0.0588 | 0.0583 | 0.2143 |
  | 100000 | 4096 | 0.0532 | 0.0540 | 2.8385 |

  - Latency with two threads, embedding size 128
  
  | Table Sz | Bach Sz | ZenDNN (ms)| PyTorch (ms)| MKLDNN (ms)|
  |---|---|---|---|---|
  | 4 | 4096 | 0.0371 | 0.0391 | 0.0619 |
  | 1000 | 4096 | 0.0338 | 0.0333 | 0.0806 |
  | 10000 | 4096 | 0.0386 | 0.0396 | 0.1412 |
  | 100000 | 4096 | 0.0414 | 0.0490 | 2.5519 |

  These tables show that DSMM performance is not tuned for large and very large
  table sizes.

  Also DSMM and EMB serve different purposes, and it is advisable
  not to roll their implementation into one kernel (separation of concerns).

  - In a production grade recommender system, there are hundreds of embedding
    table, with each table being of sizes of widely varying orders from ~10
    to 10^6. The tables may be spread accross different distributed memories.
    By contrast, DSMM may not be needed to
    support such a widely varying order, can can be highly optimized for the
    orders it is supposed to support.

  - Depending on the work-load, EMB and DSMM have different latency, throughput
    and scaling requirements.
    It is difficult to achieve these in only one kernel without making the
    kernel complex to read and reason.

  - Future requirements from EMB, and DSMM may make their optimization paths
    diverge. This will again be problematic with one kernel for both the
    operators.

  - EMB `max` reduction can not be implemented by DSMM, and a separate path
    will have to be created for it.

  - The top level API will still need an input for reduction method, and DSMM
    kernel can only be used inside an embedding bag wrapper API.

## 9. Open Questions

1. **Intrinsics vs JIT for the optimized impl.** Almost every optimized x64
   oneDNN primitive uses Xbyak JIT. The embedding kernel logic (gather a row,
   multiply by a scalar weight, reduce element-wise) is simple enough that there
   is no shape-dependent code-gen opportunity that benefits from runtime
   emission. ZenDNN's intrinsic implementation is well-validated today. The
   impl-list mechanism leaves room for a `jit_uni_embedding_bag_t` above the
   intrinsic impl in a follow-up if profiling justifies it — users see no API
   change.

2. **`max` + per-sample weights.** PyTorch forbids this; we mirror that. Some
   users may expect `max(T[i] * w[i])`; the RFC defaults to the PyTorch
   behavior.

3. **All-padding bag output.** PyTorch returns zeros for sum / mean and 0 for
   max when all indices in a bag equal `padding_idx`. We match this; document
   and test explicitly.

4. **Performance gate.** Every new oneDNN primitive must show material
   workload-level impact. A documented benchdnn perf comparison vs framework
   baselines on representative DLRM shapes is required before the Phase 2 PR
   can land.
