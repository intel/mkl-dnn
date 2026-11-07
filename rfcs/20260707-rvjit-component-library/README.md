[//]: # (Base for all permalinks below: alexandrelimassantana/oneDNN@c2ea612b037f9bbe91bc087f53abed7b7a491209, the exact head commit of PR #5500 at time of writing.)

# `rvjit`: A Component Library for RV64 JIT Kernels


Related list of PRs

- [PR #5500](https://github.com/uxlfoundation/oneDNN/pull/5500): previous proposal, deprecated.
- [PR #6383](https://github.com/uxlfoundation/oneDNN/pull/6383): current integration proposal.
- [PR #6320](https://github.com/uxlfoundation/oneDNN/pull/6320): possible first step to a technical solution to Roadmap Step 2.


## Introduction


This RFC proposes [`rvjit`](https://github.com/alexandrelimassantana/rvjit), a header-only library of code-generation components that emit reusable segments of runtime assembly through Xbyak_riscv.
Its components range from branches and constant folding to auto-tuned matmul micro-kernels.
The library grew out of five years of work on a oneDNN fork at the Barcelona Supercomputing Center (BSC), aimed at performance-portable kernels despite the micro-architectural differences across RISC-V systems.


The proposal builds on our IPDPS'26 paper, *[Just-in-Time Convolution and GEMM Code Generation for SIMD Architectures](https://doi.org/10.1109/IPDPS65963.2026.00093)*.
We found that the best choices for variables such as `LMUL` and unrolling differ across processors: the optimizations behind high-performance kernels are reusable, but how they are tuned and materialized as instructions varies.
`rvjit` represents the reusable concepts as configurable intermediate forms, so micro-kernels are built by composing and specializing them under different scenarios.


## Proposal


The library is split into three layers:


- The **interface** ([`rvjit/interface`](https://github.com/alexandrelimassantana/rvjit/tree/v0.1.0/rvjit/interface)) expresses components in ISA-agnostic terms. A matmul, for instance, is an unrolled loop nest with arbitrary tiling, where A and B memory streams feed type-agnostic multiply-adds into form-agnostic register blocks.
- The **plans** (e.g. [`matmul/plan.hpp`](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/rvjit/interface/linalg/matmul/plan.hpp), [`loop/plan.hpp`](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/rvjit/interface/loop/plan.hpp)) hold the ISA-agnostic optimization logic that links the interface to an implementation: register blocks large enough to hide arithmetic latency, double-buffered input streams, pivot pointers instead of pointer arithmetic, and others.
- The **implementation** ([`rvjit/implementation`](https://github.com/alexandrelimassantana/rvjit/tree/v0.1.0/rvjit/implementation)) supplies the ISA-specific decisions and emits the code. Each component holds one responsibility, such as choosing the instruction for a `macc` or `load`, or tracking `vtype` on RVV.


The interface and plans are reusable across systems and ISAs; supporting a new extension means specializing the implementation layer, not re-implementing the software-level optimizations.
We currently support a vector-length agnostic `matmul` RVV kernel which applies a different tuning strategy for short (128–256-bit), medium (512–1024-bit), and long (> 1024-bit) vectors. The tuning variables may be revisited at any point without touching the other components.


### Relationship to Xbyak


`rvjit` does not replace or duplicate Xbyak: its components are [composed](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/rvjit/implementation/rv64/isa_traits.hpp) around a reference to the host `Xbyak_riscv::CodeGenerator`, which remains the only instruction emitter.
The pattern already exists in oneDNN: the x64 [injectors](https://github.com/uxlfoundation/oneDNN/tree/main/src/cpu/x64/injectors) (`jit_uni_eltwise_injector`, `jit_uni_binary_injector`, `jit_uni_postops_injector`) are held by kernels as members and called to emit reusable instruction sequences.
`rvjit`'s scope is broader, covering constant folding, control flow, register blocking and allocation, and type-adaptable meta-instructions.


### Scope


Currently, `rvjit` is used exclusively to materialize RVV matrix multiplication kernels empowering convolutions, gemm, and inner product operations.
Nonetheless, most `rvjit` concepts apply beyond this scope, and the library is intended to grow to add support other ISAs such as IME, AME, and VEM matrix extensions.
We welcome the community to evolve the components and their API.
Development may take the form of new algorithms, new primitives, support for new ISA, and overall improving the oneDNN RV64 backend collaboratively rather than through scattered forks.

This updated RFC settles the questions about the library and the plan for evaluating its integration, delivered as four PRs: `rvjit` as a third-party library, then the migration of the `brgemm`, `gemm`, and 1x1 convolution kernels, which back the matmul, inner product, and convolution primitives.


### Dependency model


`rvjit` is developed, tested, and versioned in its own repository, [alexandrelimassantana/rvjit](https://github.com/alexandrelimassantana/rvjit), and oneDNN vendors a tagged release of it, following the model already used for `third_party/xbyak_riscv`.
The integration pins [`v0.1.0`](https://github.com/alexandrelimassantana/rvjit/tree/v0.1.0), copied verbatim to [`third_party/rvjit`](https://github.com/alexandrelimassantana/oneDNN/tree/07f60d9f29dbc75101f4c52f355f0a901ae2f83c/third_party/rvjit) by [`scripts/sync_rvjit.sh`](https://github.com/alexandrelimassantana/oneDNN/blob/07f60d9f29dbc75101f4c52f355f0a901ae2f83c/scripts/sync_rvjit.sh).
The vendored files are never edited in oneDNN; an update is a sync to a new tag in its own PR.
Nonetheless, `rvjit` is designed to be extended, and the oneDNN codebase may host custom `rvjit` components and kernel compositions that are not featured on the `rvjit` repository.


Primitives should not include `rvjit` headers.
Dedicated source files are created to house the `rvjit` system.
The current matmul integration centralizes the generation of all matmul micro-kernels in [`src/cpu/rv64/rvjit/rvjit_matmul.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/07f60d9f29dbc75101f4c52f355f0a901ae2f83c/src/cpu/rv64/rvjit/rvjit_matmul.hpp).
This methodology may be applied to future micro-kernels.

## Integration

The integration of `rvjit` is to be split by a stack of four PRs, each reviewable on its own:

| PR | Branch | Change |
|---|---|---|
| 1/4 third-party | [`rvjit/third-party`](https://github.com/alexandrelimassantana/oneDNN/compare/d3f8eb8416caa2ec0e546668a6336a054b9154d4...529426c7250e24a518ced15ca7332677963c7762) | vendors `rvjit` and adds the interface; no kernel changes |
| 2/4 brgemm | [`rvjit/brgemm`](https://github.com/alexandrelimassantana/oneDNN/compare/529426c7250e24a518ced15ca7332677963c7762...f20711fd873a5e47332b3f447ac915c1d827ab98) | brgemm kernels (matmul, inner product, convolution); +369 / −1,835 lines |
| 3/4 gemm | [`rvjit/gemm`](https://github.com/alexandrelimassantana/oneDNN/compare/f20711fd873a5e47332b3f447ac915c1d827ab98...4deafb62d2336d5c4fae65569c21e1b2d537445b) | GEMM micro-kernels (matmul, sgemm callers); +634 / −2,074 lines |
| 4/4 conv-1x1 | [`rvjit/conv-1x1`](https://github.com/alexandrelimassantana/oneDNN/compare/4deafb62d2336d5c4fae65569c21e1b2d537445b...07f60d9f29dbc75101f4c52f355f0a901ae2f83c) | 1x1 convolution kernel; +243 / −957 lines |

The stack is based on oneDNN `main` at [`d3f8eb8416`](https://github.com/uxlfoundation/oneDNN/commit/d3f8eb8416caa2ec0e546668a6336a054b9154d4).
Every other RV64 kernel builds, registers, and runs exactly as before.


Adoption stays gradual and justified case by case:


- New RV64 kernel contributions are **not** expected to use `rvjit`.
- Existing kernels are migrated only when it provides a measurable benefit in performance or code quality.


### Alternatives considered


1) **Extend the existing Xbyak-based approach directly** (helpers in `jit_generator_t` or per kernel). No new abstraction to review, but `jit_generator_t` would aggregate the responsibilities that `rvjit` splits between components.
2) **Keep `rvjit` in-tree** under `src/cpu/rv64/rvjit`, as in the previous version of this proposal. It was dropped in favor of the vendored model as `rvjit` requires its own set of tool for testing and correctness validation.


### API and ABI backwards compatibility


`rvjit` and its interface are compiled only when `DNNL_TARGET_ARCH=RV64` and are not exposed through any public oneDNN header or API.
No public symbol, struct layout, or documented primitive interface changes.
User-visible behavior changes are limited to the weight layouts chosen for `any` and the user weight tags each kernel accepts; each PR lists them in a "Behavior changes" table.


### Build system changes


Limited to the RV64 build, in [`src/cpu/rv64/CMakeLists.txt`](https://github.com/alexandrelimassantana/oneDNN/blob/07f60d9f29dbc75101f4c52f355f0a901ae2f83c/src/cpu/rv64/CMakeLists.txt#L35-L45): `third_party/rvjit` on the include path, and the `XBYAK_RISCV_V`, `RVJIT_VERBOSE_HEADER` and (under `DNNL_DEV_MODE`) `RVJIT_DEBUG` definitions.
`rvjit` is header-only and adds no CMake target.
The vendored tree is listed in `THIRD-PARTY-PROGRAMS` and excluded from the license check, as `xbyak_riscv` is.


### Dependencies and support matrix


`rvjit` (Apache-2.0) builds on the already-vendored `third_party/xbyak_riscv` and brings no other dependency.
x64, AArch64, and other backends are unchanged and do not compile it.


### `rvjit` components


The components, their APIs, and usage are documented in the [`rvjit` README](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#usage) and its Doxygen [API reference](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#documentation):
[constant arithmetic](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#constant-arithmetic),
[control flow](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#control-flow),
[buffer pool and stack](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#buffer-pool-and-stack),
[memory](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#memory),
[arithmetic](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#arithmetic),
[loops](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#loops),
[streams](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#streams),
and the [matrix multiplication](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#matrix-multiplication) engine composed from them.
ISA-agnostic interfaces live in [`rvjit/interface`](https://github.com/alexandrelimassantana/rvjit/tree/v0.1.0/rvjit/interface) and the RV64/RVV implementations in [`rvjit/implementation`](https://github.com/alexandrelimassantana/rvjit/tree/v0.1.0/rvjit/implementation).


### Testing


Testing is split by ownership:


- **`rvjit`** is tested in its repository ([Test](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#test)): a gtest suite that generates small kernels, runs them, and checks the values they compute, plus correctness benchmarks across VLEN 128–1024 and `zvfh`/`zvfbfwma`, run by its [CI](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/.github/workflows/ci.yml) under QEMU.
- **oneDNN** tests what it builds with `rvjit` through its existing ctest, gtests, and benchdnn, like any other kernel. No `rvjit`-specific test is added to oneDNN.

The `rvjit` test suite ensures that the library produces correct code under a wide range of scenarios that supersedes oneDNN testing framework. Nonetheless, `rvjit` relies on oneDNN for the validation of performance and integration, as these are not featured on `rvjit` CI/CD and testing pipelines.


### Reproducibility and evaluation

To facilitate the integration of `rvjit`, we have prepared a standalone repository so that users may reproduce our validation and performance analysis of `rvjit` at [reproduce-rvjit-onednn-integration](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration), which pins the PR head and its base, builds both ([Build](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#build)), and runs them natively on any RV64 machine, with scripts for native and slurm execution ([Run](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#run)).
It has five [tasks](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#the-five-tasks), run for inference and split into independent jobs that each write their own upstream-vs-`rvjit` report:


- **ctest**: oneDNN's API gtests of the affected primitives.
- **correctness-onednn**: benchdnn correctness over oneDNN's CI batches and shapes, per affected primitive.
- **correctness-extended**: smoke tests of the edges each PR changes.
- **performance-onednn**: oneDNN's perf-CI method over its perf-CI inputs and model shapes.
- **performance-extended**: the dataset used on our research featuring convolutions and matrix multiplication shapes extracted from open-source computer vision models for pytorch.


The [datasets](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#datasets) and the [data types](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#data-types) each kernel is responsible for are documented there.
The [analysis](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#analysis) reports speedup and creation time per implementation, dataset, and primitive, with oneDNN's perf-CI regression test, and every phase is [timed](https://github.com/alexandrelimassantana/reproduce-rvjit-onednn-integration#timing).
Creation time is reported next to execution time because `rvjit` tunes and generates code when a primitive is created.


### Performance


We will provide a set of performance evaluation results measured on a BananaPi-F3 node showcasing the performance before and after the introduction of all `rvjit` pull requests.
We will also verify correctness of micro-kernels generated for 16384-bit vector architectures executed on an FPGA. Evaluations on other microarchitectures are welcome to de-risk the vendor-agnostic claims; the reproducibility repository runs unchanged on them.


### Ownership and maintenance


BSC pledges to maintain `rvjit` and support new use cases in oneDNN.
Vendor-agnostic code generation for RV64 is an active research line at the institution, and we are open to assisting with tuning efforts.


- **Versioning**: Semantic Versioning over the public interface; releases are tagged on `main`, and oneDNN pins a tag ([Charter](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#charter)).
- **Contributions**: rules for new components, tests, and QEMU validation across VLENs are in [Contributing](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#contributing).
- **Documentation**: Doxygen, with pre- and post-conditions of each component call, published with every release ([Documentation](https://github.com/alexandrelimassantana/rvjit/blob/v0.1.0/README.md#documentation)).
- **Debugging**: in `DNNL_DEV_MODE`, `rvjit` failure traces go to oneDNN's debuginfo verbose log through [`rvjit_verbose.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/07f60d9f29dbc75101f4c52f355f0a901ae2f83c/src/cpu/rv64/rvjit/rvjit_verbose.hpp).


## Open Questions


All open questions from the previous RFC have been answered and included into this updated version. We welcome any comments on the integration and evaluation plans.
