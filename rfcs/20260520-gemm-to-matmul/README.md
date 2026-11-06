# RFC: Drop the internal GEMM API

## Background

oneDNN exposes two distinct representations for matrix multiplication on GPU:

- **Matmul** - row-major by default, user-facing, with `DNNL_ARG_SRC / WEIGHTS / DST` arguments
- **GEMM** - column-major, internal, BLAS-shaped (`M, N, K, lda, ldb, ldc, transa,
  transb`), with cross-wired mapping (`DNNL_ARG_SRC` -> `B` and `DNNL_ARG_WEIGHTS` -> `A`) for `B^T x A^T = C^T` reformulation from row- to column-major layouts

These two descriptions express the same arithmetic. Today the GEMM internal API
exists primarily as the bridge between matmul and the JIT GEMM kernels
(gemmstone-based) and as an internal interface for RNN and inner product.

The GEMM API is represented by:

- Descriptor: `gemm_desc_t`
- Primitive descriptor (PD): `gemm_pd_t`
- Primitive kind: `primitive_kind::gemm`

oneDNN GPU has four GEMM implementations:

- `intel::gemm::xe_hp_systolic_t` - XeHP+ implementation with blocked layouts
- `intel::gemm::gen_t` - JIT GEMM implementation
- `intel::gemm::with_post_ops_t` - a wrapper calling JIT GEMM and applying post-ops in a separate OpenCL kernel
- `intel::gemm::ref_t` - reference GEMM implementation

## Motivation

Maintaining a parallel description for the matmul operation is hard to justify:

- JIT GEMM itself uses an internal `GEMMProblem` abstraction so we end up with
  a `matmul_desc` -> `gemm_desc` -> `GEMMProblem` translation chain, where the middle layer can be removed.
- The GEMM API handles row-to-column-major translation poorly: it
  [auto-swaps](https://github.com/uxlfoundation/oneDNN/blob/9a7e1ac215b0f571749dff8a5d8373cb3922a618/src/common/gemm_types.hpp#L144)
  some properties but never re-assigns indices, and attribute-related translation is done ad hoc inside the JIT GEMM implementations.
- One of the reasons to have an internal GEMM API was to support custom A or B reduction (used for backward inner product).
  Matmul has since gained the same [capability](https://github.com/uxlfoundation/oneDNN/blob/455ac84e8fd8f2450b13fc29f06f43f18caf842c/src/common/opdesc.hpp#L279),
  so this reason no longer holds.

## Design

The proposal is to delete the internal GEMM descriptor and primitive descriptor
(`impl::gemm_desc_t` and `impl::gemm_pd_t`) and let matmul be the single
internal representation for the operation:

1. **One internal descriptor.** All GPU consumers of GEMM - matmul
   itself, inner-product and RNN are ported to matmul API (`create_gemm_pd` -> `create_matmul_pd`).
   The internal GEMM descriptor, its PD base, its A/B/C argument keys, its construction
   helper, and its registration list are removed.

2. **All GEMM kernels become matmul implementations.** The existing JIT
   GEMM implementations (`intel::gemm::xe_hp_systolic_t` and `intel::gemm::gen_t`) register under
   matmul. They share a single base PD that owns the JIT GEMM common
   logic - including post-op canonicalization, swap-ab translation,
   quantization-related logic. `intel::gemm::with_post_ops_t` is reimplemented
   on top of the matmul descriptor and PD, and `intel::gemm::ref_t` is merged with
   `intel::matmul::ref_t`.

3. **Column-major orientation is a private JIT GEMM implementation detail.** The matmul
   descriptor is kept unchanged. The decision to swap A and B
   in order to match a column-major kernel lives entirely inside the JIT GEMM PDs.

## Out of scope

- The public GEMM API (`dnnl_sgemm`, `dnnl_gemm_*`). It dispatches directly to
  CPU and is independent of the internal PD.
- `brgemm_desc_t` and the BRGEMM ecosystem - a separate path with a separate API.
- CPU GEMM implementations. The proposal is a GPU-side cleanup. CPU matmul/GEMM paths are out of scope.

## Update: handling transformations and swap-ab inside JIT GEMM

Today, row-to-column-major mapping is split between `gemm_desc_t`, which
implicitly rewires arguments, and JIT GEMM PDs, which handle `swap_ab` through
scattered init/execute branches. Additionally, PDs apply post-op-related
transformations (e.g. bias-to-post-op), which forces a scattered, intermediate
representation that diverges from the external descriptor and PD.

The proposal is to build one canonical "view" from the descriptor and
attributes. The view is first populated directly from the descriptor and PD.
Two transformations are then applied - (1) `swap_ab` and (2) post-op - to match
the kernel's expectations. This design gives us a single source of truth that
is fully coherent at any given time.

A few comments for further discussion:

- Can we use `GEMMProblem` as the "view" representation? Partially yes, but
  with nuances:
    - `GEMMProblem` doesn't include M/N/K/LDA/LDB, host scalars, or offsets so
       we still need a complementary structure to represent these
    - Moving to `GEMMProblem` may require operating based on `GEMMProblem`
      derived parameters or adding another conversion back to the oneDNN world.
      It's hard to fully separate the oneDNN and gemmstone spaces, so the
      decision of how much to rely on `GEMMProblem` falls on a spectrum.
      Additional input is appreciated.
- The first step in implementing the RFC is to clean up the existing PD flow by
  introducing a view abstraction. This will concentrate the mapping logic in one
  place and make the transition from the GEMM API to the matmul descriptor/PD
  more mechanical.
