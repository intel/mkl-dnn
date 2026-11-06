# RFC: RVCL Integration in oneDNN — a Unified RISC-V Computation Library

## 1. Introduction & Motivation: the dilemma of integrating RISC-V vendor operators

This RFC proposed rvcl Integration in oneDNN — a Unified RISC-V Computation 
Library (https://github.com/zte-riscv/rvcl) to address following issues:

### 1.1 More libraries, a wider integration surface

Fragmentation is the current reality of the RISC-V operator ecosystem. The
RVV 1.0 vector extension is ratified, but VLEN ranges from 128 to 1024 bits;
the matrix extensions IME/VME/AME sit at different stages between draft and
ratification; vendor-proprietary extensions (T-Head, SiFive, SpacemiT, ZTE,
and others) evolve independently. The motivation that started
[#6320](https://github.com/uxlfoundation/oneDNN/issues/6320) prefigures the
reality ahead: generic RVV kernels cannot exploit the A100 compute cores
(VLEN=1024 with the IME matrix extension) of the SpacemiT K3 heterogeneous
SoC, and RISC-V vendors each maintaining their own operator libraries is
fragmenting the ecosystem.

The list of vendor libraries keeps growing: SiFive SKL and T-Head XuanTie
CSI-NN2/SHL are already shipping, and SpacemiT is in volume production. The
vendor × extension × VLEN combination space of RISC-V is open-ended — there
is no enumerable upper bound. If every library enters oneDNN through its own
adapter, the cost structure is: one RFC review per library, one adapter
directory, one build dependency, one line in the supported-operator list,
plus each library's own threading and memory conventions to reconcile and
audit one by one. The integration surface grows linearly with the number of
vendors and nothing is reused across integrations: for oneDNN it is an
ever-growing review and maintenance burden; for vendors it is duplicated
integration cost; for users it is a fragmented support matrix. This is
exactly the situation #6320 is trying to avoid — and which per-library
integration cannot avoid.

### 1.2 The binary-size cost of the in-tree route

Binary size is the more direct constraint. In-tree paths are compiled for
all users: every new vendor path enlarges the binary of every RV64 build,
while the dominant RISC-V deployment scenarios today are edge and endpoint
devices — precisely the ones most sensitive to binary size. Working around
the size with per-vendor build switches instead explodes the build matrix
and the QA surface.

### 1.3 The problem is a mechanism problem

For oneDNN to support RISC-V hardware diversity,
[#6406](https://github.com/uxlfoundation/oneDNN/pull/6406) lays out Route 1
(native in-tree JIT) and Route 2 (external-library adapters) — both at
per-vendor granularity, which multiplies the maintenance cost. This proposal
adds a third granularity: **integrate once, absorb many vendors**.

---

## 2. Proposal: follow the ACL/KleidiAI style to lower the integration pressure and maintenance cost of vendor customization on the RV64 backend

**Core claim:** the mechanism oneDNN needs in order to support RISC-V
diversity should be no more than what aarch64 already requires. No N vendor
adapters, and no in-tree zoo of vendor kernels — one integration of an
ACL/KleidiAI-shaped, kernel-level unified operator layer is enough, with
vendor diversity digested inside the layer.

The claim unfolds into three alignments:

1. **Maintenance cost aligned with aarch64.** One RFC, one adapter, one
   opt-in dependency, one line in the support matrix. Today aarch64
   maintains ACL/KleidiAI and x64 maintains ZenDNN; rv64 would maintain one
   rvcl — inside the baseline every existing architecture already sets.
2. **Binary size aligned with aarch64.** Default OFF; when not enabled, the
   adapter disappears entirely at compile time and the OFF build is
   byte-identical to today; when enabled, what gets linked is still a single
   library. Kernels for not-yet-standardized vendor extensions never enter
   the oneDNN binary.
3. **Framework boundary aligned with aarch64.** Highly customized vendor
   operators take the third-party path, generic kernels (including
   matrix-extension compute kernels as they gradually standardize) take the
   in-tree oneDNN path, with runtime capability detection aligned (VLEN,
   matrix-extension features, the MVENDORID vendor probe).

The relation to the two routes of #6406 is complementary rather than
competitive:

- **Existing standard-RVV layer:** rvcl targets a unified abstraction layer
  for vendor-custom operators and only fills gaps that oneDNN JIT has not
  yet covered; at integration time it contributes no generic kernels. The
  in-tree oneDNN JIT remains the stable delivery vehicle for standard RVV.
- **Matrix extensions (IME/VME/AME):** on Route 1, the chain from spec
  ratification through Linux hwprobe upstreaming, Xbyak_riscv encoding
  support, and a oneDNN release that carries it is measured in years. rvcl
  incubates draft-stage extensions first, under versioned naming and
  tolerance-based differential testing; once a spec freezes, the validated
  kernel designs and calibration data are handed over to Route 1 — meeting
  users' interim customization needs.
- **The vendor long tail:** vendor-library integration is supported but is
  usually a transitional arrangement, with JIT the better long-term route —
  **rvcl is not a replacement for the in-tree JIT; it is a fast incubation
  channel for vendor libraries, absorbing the integration pressure and
  maintenance cost of vendor customization for oneDNN's RV64 backend**.

---

## 3. rvcl: an open-source RISC-V compute library providing a kernel-level unified operator layer

### 3.1 Project overview

rvcl (RISC-V Compute Library, https://github.com/zte-riscv/rvcl) is an
open-source project initiated and maintained long-term by ZTE:

- **License and governance:** Apache-2.0, DCO contribution flow,
  MAINTAINERS/SECURITY/CITATION all in place.
- **Interface shape:** the public API is a set of C ABI headers
  (`include/rvcl/*.h`) over an internal C++17 implementation — naturally
  compatible with oneDNN's C++ build, with no language boundary.
- **Quality gates:** dual CI on x86 and QEMU rv64, clang-format/clang-tidy,
  ASan/UBSan, plus machine checks for conventional commits and copyright
  headers — all already in effect on mainline.
- **Same-origin JIT layer:** the JIT is built on a vendored xbyak_riscv,
  tag 1.31 (BSD-3) — the same origin and version as oneDNN's
  `third_party/xbyak_riscv`, so kernel designs and encoding experience can
  flow in both directions.
- **Roadmap:** v0.1 (2026-10-31, framework interface freeze); v0.2
  (2026-11-15, IME demo kernel and multicore framework wrap-up); v0.3
  (2026-12-15, the oneDNN adapter lands — the first PR of this RFC); v0.4
  (2027-01-31, the first vendor extension integrated through runtime
  registration); v0.5 (2027-03-31, full int8/bf16/post-op coverage opened
  to the oneDNN adapter); v1.0 (2027-04-30, PyTorch linear layer working
  end to end through oneDNN to rvcl, and a stable-ABI commitment). Key
  dates in §4.

### 3.2 Feature 1: simple integration — a kernel-level abstraction absorbs hardware diversity

rvcl places the abstraction at the kernel level rather than the framework
level: what it exposes to a host is a narrow interface from an operator
request to a kernel, and hardware differences are digested inside the layer.
This shape mirrors the evolution from ACL to KleidiAI: ACL, designed as a
framework, carries reorder caching, thread management, and memory management
of its own, which conflicted with host mechanisms and produced a large
amount of workaround code; KleidiAI replaced it in the form of an ops
library that is single-threaded and allocates no memory inside kernels.
rvcl turns this lesson into a kernel-level contract that rules out
framework-level state at the interface.

```
Capability (what the hardware can do)
    -> Registry (which kernels exist)
    -> Dispatch (which one should run)
    -> Kernel (execute)
Cross-cutting contracts: Scheduler (injectable orchestrator)
                        + Memory (zero-copy import, workspace reporting)
```

Hardware diversity is digested inside this four-stage pipeline. First, the
Capability layer characterizes the hardware with compile-time macros plus
runtime hwprobe detection (VLEN read through the `vlenb` CSR; MVENDORID
already probed). rvcl compiles against a single rv64gc baseline and selects
paths at runtime, so one binary covers different VLENs and
microarchitectures (proven end to end on a real SpacemiT K3: a single
binary selects the JIT path through hwprobe, 12/12 differential checks
green). Second, the Registry layer indexes kernels by an operation × dtype
tuple and exposes public runtime-registration APIs (`rvcl_register_kernel`
/ `rvcl_unregister_kernel`); vendor kernels, including out-of-tree
closed-source .so files, enter as registry entries without any change to
the rvcl mainline. Third, the Dispatch layer filters and scores candidates
by operation, dtype, capability, and shape; when there is no candidate it
returns UNSUPPORTED and the host gets a clean fallback. Finally, the Kernel
contract requires every kernel to implement a validate/configure/run
three-phase lifecycle: the `run()` path must not allocate dynamically, must
not create threads, must take all memory from pack slots, and must
correctly execute any legal work-range slice. The contract is written into
CONTRIBUTING and enforced for all contributions.

The integration surface for oneDNN therefore converges to:

1. **A narrow interface.** The public API is C ABI headers; the adapter does
   nothing but memory-descriptor translation and a dispatch call — no
   language boundary, no framework-level callbacks.
2. **Parallelism belongs to the host.** Kernels never create threads;
   parallel needs are expressed as work ranges. The host can slice and
   drive the work directly, or inject an orchestrator through
   `rvcl_scheduler_set()` (a process-wide one-time setting, mirroring ACL's
   `IScheduler`). A multicore matmul example with an injectable worker pool
   has already merged into mainline, and the TSan configuration and
   concurrency stress drivers are in place.
3. **Aligned memory conventions.** Tensors are imported zero-copy; kernels
   report their workspace requirements and the host allocates — mapping
   directly onto oneDNN's scratchpad mechanism.
4. **Clean fallback.** Dispatch returns UNSUPPORTED when no candidate
   exists; the adapter lets the driver list fall through, with no risk of a
   wrong selection.
5. **No further oneDNN-side changes.** After integration, new hardware (a
   new VLEN, a matrix extension, custom instructions) appears only as an
   rvcl version bump; the oneDNN-side integration surface stays the same.

### 3.3 Feature 2: a fast lane for new vendor silicon, with self-managed quality and user opt-in

**A fast lane for new hardware.** Route 1 needs a new piece of silicon to
wait, in sequence, for Linux hwprobe upstreaming, Xbyak_riscv encoding
support, and a oneDNN release that carries it before it reaches the default
build — a timeline measured in years (#6406 itself notes the dependence on
Linux merges and vendor OS follow-through). rvcl shortens this path to a
single release: the vendor kernel is validated and calibrated inside rvcl
and delivered under versioned naming. Draft-stage matrix extensions
(IME/VME/AME) are incubated first with tolerance-based differential
testing; once a spec freezes, the validated kernel designs and calibration
data are handed to Route 1. Serving new hardware that has not yet reached
the default build through an opt-in external library is a legitimate use of
an external library as a transitional solution — the
[ZenDNN RFC](https://github.com/uxlfoundation/oneDNN/blob/rfcs/rfcs/20260616-zendnn-integration/README.md)
was accepted on the same logic.

**Self-managed quality.** Kernels entering the rvcl registry pass a unified
set of quality gates: dual CI on x86 and QEMU rv64, differential corpora
graded into bit-exact and tolerance classes, ASan/UBSan, clang-format /
clang-tidy, and machine checks for conventional commits — all in effect on
mainline. The kernel contract is enforced equally on kernels from every
source, and QEMU dual-VLEN differentials plus real-hardware (SpacemiT K3)
validation are regular release gates. oneDNN needs to audit the quality of
one library and one interface instead of evaluating each vendor's
engineering process separately; kernels that do not pass the gates cannot
enter the registry — and therefore cannot reach users.

**User opt-in.** `RV64_USE_RVCL` defaults to OFF, and the OFF build is
byte-identical to today: users who do not enable it pay nothing in review,
binary size, or QA for the vendor long tail; users who need specific vendor
hardware enable it explicitly and get kernels validated by the gates above,
not unaudited vendor binaries.

### 3.4 Feature 3: not only ZTE's high-performance matrix kernels — multi-vendor custom operators to follow

rvcl's current high-performance matrix kernels come from ZTE and are
maintained long-term: the three-path RVV f32 implementation (C fallback /
intrinsics / JIT) is already on mainline; the IME demo kernel is scheduled
for v0.2/v0.3 per the roadmap; ZTE's custom extension will be integrated
through runtime registration, with development starting in v0.2 and
completing in v0.4. The last item is the emblematic one: ZTE's own custom
extension enters through the same registration channel as any third-party
vendor. The unified layer is not a privileged path for its founding vendor
— the neutrality of the mechanism is demonstrated first by the founder
itself.

Multi-vendor custom operators are a design goal, not a distant vision.
First, the mechanism is ready: the runtime-registration APIs have landed,
MVENDORID is probed, and the vendor-probe dispatch interface is scheduled
for the next version per the roadmap; any vendor kernel, including
out-of-tree closed-source .so files, enters as a registry entry without
modifying the rvcl mainline. Second, the quality bar is identical for every
source, so multi-vendor participation does not dilute quality (§3.3).
Third, the multi-vendor unified-operator-layer direction has the support of
the RISE (RISC-V Software Ecosystem) community. Fourth, contact has been
established with SiFive regarding their SKL operator library, exploring how
SKL matrix kernels could enter rvcl through the registration mechanism.

For oneDNN, the meaning returns to the cost structure of §1: vendor
operators converge into one registry, one set of quality gates, and one C
ABI, so oneDNN's integration and review surface does not grow with the
number of vendors; more vendor libraries show up as more registry entries
inside rvcl, not more adapters inside oneDNN.

### 3.5 Scope of the first rvcl PR

Timing, tied to §4: this RFC enters discussion within 2026-11; the adapter
and the validation data are completed within the v0.3 window; the first
implementation PR is submitted once the data is ready, expected within
2026-12.

| Dimension | Scope | Rationale |
| --- | --- | --- |
| Operator | matmul (batched shapes included) | rvcl's current capability |
| dtypes | f32 or f16 | narrow starting point; int8/bf16/post-ops open up in later versions |
| Feature surface | plain layouts, no post-ops (accepted only with default attr values) | conservative starting predicates; failures fall through, no wrong-selection risk |
| Build | `RV64_USE_RVCL` default OFF, CMake FetchContent pinned to a reviewed commit, `DNNL_RV64_USE_RVCL` defined when enabled | ACL/KleidiAI precedent |
| Registration slot | ahead of `rvv_matmul_t` | ACL/KleidiAI precedent |
| Threading | oneDNN `parallel()` splitting the M dimension as the only source of parallelism; kernels single-threaded | eliminates a second parallel framework |
| Integration surface | an adapter under `src/cpu/rv64/rvcl/` plus one registration macro, compiled out when not enabled | same shape as the ACL/ZenDNN integrations |
| Validation | QEMU dual-VLEN differentials + K3 real-hardware benchdnn comparison + byte-identical OFF build | #6406 validation clauses; ZenDNN acceptance gates |

---

## 4. Roadmap and key milestones

These are the target dates of rvcl's public roadmap (internal planning
anchors; calendar risk has already been absorbed into the schedule):

| Version | Target date | Deliverables (the oneDNN-relevant part) |
| --- | --- | --- |
| Landed (2026-11) | — | Four-stage framework + two cross-cutting contracts; three-path RVV f32 (C fallback / intrinsics / JIT); QEMU dual-VLEN CI and differential corpora (bit-exact / tolerance-graded); runtime-registration API; vendored xbyak_riscv 1.31; 12/12 proven on a real K3; multicore example with injectable worker pool |
| v0.1 | 2026-10-31 | Framework interface freeze; versioned interface `version.h`; v0.1.0 tagging and release engineering |
| v0.2 | 2026-11-15 | IME demo kernel (tolerance differential against a simulated oracle, validating the draft-extension incubation path); dispatch cost model; multicore framework wrap-up |
| **v0.3** | **2026-12-15** | **The oneDNN adapter lands (the first PR of this RFC); full benchdnn `--matmul` correctness (f32/f16); real-hardware comparison data against `rvv_matmul_t`** |
| v0.4 | 2027-01-31 | The first vendor extension (ZTE custom extension) integrated through runtime registration — the open registration mechanism proven in practice; FP8/FP4 dtypes |
| v0.5 | 2027-03-31 | Full int8/bf16/post-op coverage opened to the oneDNN adapter; scheduler injection bridge (the oneDNN thread pool driving rvcl directly); real-hardware performance baselines |
| v1.0 | 2027-04-30 | PyTorch linear layer working end to end through oneDNN to rvcl (DNNL_VERBOSE attribution); **stable-ABI commitment** (major/minor compatibility rules and a deprecation process) |

Timeline linkage: this RFC is expected to enter discussion within 2026-11;
the adapter implementation and the benchdnn data complete within the v0.3
window; the first implementation PR is submitted once the data is ready
(expected 2026-12 to 2027-01). The oneDNN review cycle does not block
rvcl's own roadmap — the matrix-extension line (the IME/VME/AME demos of
v0.2/v0.3) and the oneDNN adapter line are scheduled independently.

---

## 5. Open Questions

Before listing the open questions, two anticipated ones are answered up
front:

**Q1: Why not contribute the kernels directly in tree?**
Three reasons. First, cross-vendor runtime distribution: a single rv64gc
binary serves multiple hardware through runtime probing, which the in-tree
path cannot do without recompiling. Second, the encoding churn of
draft-stage extensions does not fit the in-tree rhythm: the IME/VME specs
are not frozen, and kernels being rewritten as spec revisions land is the
norm; rvcl absorbs this churn under versioned naming, while the tree is not
a good place to carry it. Third, kernels stay conveyable: the JIT layer
shares its origin with oneDNN (xbyak_riscv at the same tag), so once a
spec freezes and hardware is ready, validated kernel designs can move into
Route 1 wholesale — rvcl is not a substitute channel for bringing kernels
in tree; it is a parallel channel.

**Q2: What is the relation to other RISC-V vendor libraries (for example
the kernel library SpacemiT plans to open-source)?**
rvcl is a vendor-neutral unified layer. A single-vendor library delivers
the best performance on its own hardware; rvcl, through a stable C ABI and
runtime registration, lets any vendor kernel (including closed-source .so
forms) become a registry entry — oneDNN integrates only once. We welcome
every vendor to participate in rvcl as a registration contributor.

The open-question list:

1. **Registration order.** The first PR registers ahead of `rvv_matmul_t`
   (oneDNN's own JIT implementation) and ahead of the generic gemm: rvcl
   first judges whether a high-priority vendor custom kernel can serve the
   request, then dispatch falls through to oneDNN's entries in order. Is
   that acceptable?
2. **Dependency form.** We plan to introduce it via CMake FetchContent
   pinned to a reviewed commit — does upstream prefer another form (for
   example a submodule)?
3. **Threading model evolution.** The first PR uses oneDNN `parallel()`
   splitting the M dimension as the only source of parallelism, with rvcl
   providing single-threaded kernels only, to keep the model consistent.
   Any objections?
4. **Acceptance gates.** We propose the first PR be accepted when: the
   benchdnn differential against `--skip-impl=rvcl` is fully green; there
   is zero regression on non-target configurations; the OFF build is
   byte-identical; and size and performance are measured with the option ON
   and OFF. Anything to add or change?
5. **Follow-on operator order.** After matmul, we solicit priorities for
   conv/brgemm/grouped (we will adjust the roadmap based on oneDNN
   community feedback).
6. **Naming.** The `src/cpu/rv64/rvcl/` directory, the
   `DNNL_RV64_USE_RVCL` macro, and the impl name `rvcl:gemm` — do these
   match upstream conventions?

We also welcome: feedback from hardware vendors on the registration
mechanism and the vendor-probe interface; suggestions on how to reuse the
validation assets (QEMU reproduction, differential corpora); and a
realistic assessment of the roadmap pace.
