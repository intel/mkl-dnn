# RFC: Vendor-specific ISA and microarchitecture support for RV64

## Introduction

We would like to give contributors a clear way to bring RISC-V optimizations
into oneDNN. The design follows the discussion in
[#6320](https://github.com/uxlfoundation/oneDNN/issues/6320).

We propose two routes:

- **Route 1: Native JIT integration.** Add Xbyak_riscv support where new
  instructions are needed, then implement optimized kernels in oneDNN. This is
  our preferred long-term route because primitives can share kernel code.
- **Route 2: Vendor-library integration.** Connect an existing vendor library
  to oneDNN through an adapter. Each new library needs its own RFC.

SpacemiT K3 is a useful example of why this matters. Its X100 cores have
256-bit vectors, while its A100 cores have 1024-bit vectors and specialized
matrix and dot-product instructions. In our tests, longer vectors alone did
not always make A100 faster. Kernels also need suitable blocking, register
usage, memory access, and scheduling.

Standardization work on VME, IME, and AME can proceed alongside vendor-specific
optimizations.

## Proposal

### Route 1: Native JIT integration

Route 1 follows these steps:

1. **Make RV64 JIT kernels selectable at build time.** Allow existing kernels
   and their dispatch registrations to be compiled out. Keep all enabled by
   default.
2. **Make the hardware detectable through upstream Linux.** Linux must have
   merged the support needed to identify the target CPUs with
   `riscv_hwprobe()`. At minimum, we need to identify the microarchitecture.
   We recommend directly reporting the supported instruction extensions.
3. **Update Xbyak_riscv.** Add CPU-set queries so ISA support and VLEN are
   detected for the CPUs that will run the kernel, including heterogeneous
   systems. Add encodings, operand checks, and tests for new instructions. For
   vendor extensions reported only through CPU identifiers, propose a lookup
   table in Xbyak_riscv.
4. **Update oneDNN's Xbyak_riscv version** after any required upstream changes
   merge.
5. **Submit the optimized oneDNN kernels.** Include correctness results,
   performance measurements and supported configurations.

Direct ISA reporting can take time to reach upstream Linux, and vendor
operating systems may take considerably longer to adopt it after the merge.
CPU identification plus a lookup table is acceptable. If Xbyak_riscv declines
the table, put it in oneDNN's RV64 detection code.

#### What does “detectable” mean?

There are two ways to obtain the information we need.

**Example A: Linux tells us which CPU it is; a table tells us what it supports.**

Our K3 measurements in #6320 returned these architecture identifiers:

| CPU group on the tested machine | `MARCHID` returned by Linux | Core type |
| --- | --- | --- |
| CPUs 0-7 | `0x8000000058000002` | X100 |
| CPUs 8-15 | `0x8000000041000002` | A100 |

Suppose we want to run an IME kernel on A100. We ask `riscv_hwprobe()` for the
vendor, architecture, and implementation identifiers of the selected CPUs:
`MVENDORID`, `MARCHID`, and `MIMPID`. We then look up those identifiers in a
table based on the vendor's documented specifications.

```text
Query the selected CPUs with riscv_hwprobe()
    -> identify an A100 implementation
Look up that implementation in the vendor table
    -> find the supported extensions and their versions
Select a oneDNN kernel that uses those extensions
```

In this example, Linux answers “which CPU?”, and the table answers “which
instructions?”. Each table entry documents the mapping, including CPU
revisions, OS requirements, and its specification references.

**Example B: Linux directly tells us whether an extension is supported.**

SiFive's `Xsfvqmaccdod` extension provides an existing vendor-specific example.
Linux defines the query key `RISCV_HWPROBE_KEY_VENDOR_EXT_SIFIVE_0` and the
capability bit `RISCV_HWPROBE_VENDOR_EXT_XSFVQMACCDOD`.

We query that key for the CPUs that will execute the kernel and check the
returned bit. When it is set, Linux reports that those CPUs support the
extension:

```text
hwprobe(selected CPU set, RISCV_HWPROBE_KEY_VENDOR_EXT_SIFIVE_0)
    -> value contains RISCV_HWPROBE_VENDOR_EXT_XSFVQMACCDOD
The extension is available on the queried CPU set
    -> a kernel using Xsfvqmaccdod is eligible for ISA-based selection
```

The key, bit, and detection implementation are already present in upstream
Linux: see the [hwprobe keys](https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/hwprobe.h),
[SiFive capability bits](https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/vendor/sifive.h),
and [SiFive hwprobe implementation](https://github.com/torvalds/linux/blob/master/arch/riscv/kernel/vendor_extensions/sifive_hwprobe.c).

This is the model we recommend for other vendors.
With direct reporting, a new CPU can advertise the same extension without
needing another model entry in our table.

**Both examples use `riscv_hwprobe()`. We accept Example A as the minimum and
recommend Example B.** When direct reporting is available, it takes priority.
An explicit unsupported result must not be overridden by the lookup table.
If neither method confirms support, oneDNN uses its existing fallback.
Extensions that add OS-managed state also need the corresponding OS support.

The [Linux hwprobe documentation](https://docs.kernel.org/arch/riscv/hwprobe.html)
explains CPU identifiers, extension bits, and CPU-set queries.

#### Heterogeneous CPU support

We separate three responsibilities: the OS reports usable hardware features,
Xbyak_riscv detects them, and the application or threading runtime controls
where workers run. oneDNN uses that information to select and tune kernels.

Other oneDNN backends provide useful examples:

- **x64:** ISA selection uses cached Xbyak CPU information. Hybrid-core topology
  is also available, and oneDNN uses it to choose conservative cache sizes
  across P/E core types. See [ISA detection](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/x64/cpu_isa_traits.hpp)
  and [hybrid cache handling](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/x64/platform.cpp).
- **AArch64:** the native JIT backend uses HWCAP information and caches the SVE
  length obtained through `PR_SVE_GET_VL`. Linux exposes system-safe feature
  information and manages SVE vector length as thread state. This differs from
  reading the physical RVV VLEN on whichever core runs the detection code.
  See [AArch64 detection](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/aarch64/cpu_isa_traits.hpp),
  [Linux feature reporting](https://docs.kernel.org/arch/arm64/cpu-feature-registers.html),
  and [SVE thread state](https://docs.kernel.org/arch/arm64/sve.html).

For RV64, we use the following execution policy.

**Default execution uses common capabilities.** On platforms where the OS
allows workers to run across different core types, detection covers every CPU on
which the workload may run. If the runtime cannot establish a narrower set,
use all online CPUs. Vendor extensions can be enabled only when supported
throughout that set. For a mixed-microarchitecture set, use generic tuning.
RVV execution requires OS support for vector state and thread migration.
Enable fixed-VLEN kernels when the execution set has a known, compatible VLEN;
otherwise select a compatible fallback.

**Specialized execution uses a fixed, homogeneous CPU group.** The application
or launcher establishes the group before oneDNN capability initialization and
before creating OpenMP or thread-pool workers. The application must keep every
worker and later primitive execution within that group. The launcher or
runtime also performs any platform-specific setup.
A process-wide cpuset restriction is one way to enforce the CPU boundary.

The initial implementation supports one fixed execution group per process.
The backend records its CPU set and capabilities once and uses them for
primitive creation, JIT generation, and execution. Keep that set fixed for the
lifetime of the generated kernels.
To use X100 and A100 concurrently in this initial design, applications use
separate processes, each configured for its core group.

#### Changes required in Xbyak_riscv

The current [Xbyak_riscv detector](https://github.com/herumi/xbyak_riscv/blob/main/xbyak_riscv/xbyak_riscv_util.hpp)
queries all online CPUs with `cpusetsize=0` and `cpus=NULL`, then reads `vlenb`
on the calling CPU. We need an explicit CPU-set query so both results describe
the same target CPUs.

1. **Accept the target CPU set.** Add an explicit-set detection API alongside
   the existing API. oneDNN supplies its established execution set.
2. **Query and group CPUs.** Read vendor, architecture, implementation, and
   extension information per CPU. Group matching results and return the
   common extension set as well. `RISCV_HWPROBE_WHICH_CPUS` can accelerate
   matching; intersect its results with the requested set. Per-CPU queries
   provide a fallback when that flag is unavailable.
3. **Resolve vendor capabilities.** Prefer direct hwprobe extension reporting,
   then use the documented identity table where needed. Keep an unknown
   identity distinct from a confirmed unsupported capability.
4. **Obtain VLEN for the selected group.** Prefer an OS interface where one is
   available. Otherwise use a dedicated probe thread bound to a target CPU
   before its first vector instruction. Check vector availability first and
   create fresh probe threads for other CPUs rather than migrating live vector
   state across unlike cores. The launcher must already have completed any
   required core enrollment. Failed probes leave VLEN unknown.
5. **Return an immutable detection result.** Include the queried CPU set,
   identifiers, extensions, VLEN information, and the source of the capability
   information. Each explicit query returns a separate object.

oneDNN uses this result during initialization and kernel selection. Target
properties that affect generated code, including VLEN, must be represented in
kernel-cache keys or isolated by the fixed execution configuration. Capability
checks stay outside the kernel's hot loops. Tests must cover restricted CPU
sets, mixed groups, unknown VLEN, and rejection of incompatible reuse.

#### Where should the work live?

| Work | Preferred location |
| --- | --- |
| Instruction encodings and operand checks | Xbyak_riscv |
| CPU-set detection and vendor lookup table | Xbyak_riscv, proposed there first |
| Lookup table if Xbyak_riscv declines it | Centralized oneDNN RV64 detection code |
| Primitive selection and workload thresholds | oneDNN |
| Blocking and scheduling | oneDNN or a shared kernel-generation library |

If oneDNN owns the table, primitives should use the shared detection result.
We can move the table to Xbyak_riscv or remove it if Linux reports the
extensions directly.

A reusable GEMM/BRGEMM kernel is a useful first target because it can benefit
matmul, convolution, and inner product. The same CPU information can help tune
standard RVV kernels, including LMUL, packing, and blocking choices.
If [rvjit](https://github.com/uxlfoundation/oneDNN/pull/6368) is merged,
evaluate its reusable matmul components before adding another GEMM generator.

#### ISA and architecture-specific optimizations

We also welcome optimizations that use ISA and architectural features, such as
specialized data-movement instructions, DMA, and on-chip memory. The integration
depends on how the vendor exposes the feature to userspace.

**User-mode instructions belong in the JIT kernel.** If a feature is available
through documented user-mode instructions, Xbyak_riscv can encode them and
oneDNN can use them in a generated kernel. For example, a kernel may issue a
supported data-movement instruction or access a prepared scratchpad address.
Runtime detection must establish the required ISA support, and the surrounding
execution code supplies the addresses and parameters.

**OS and driver operations belong in the C++ layer.** Some features require a
device to be opened, memory to be allocated or mapped, or work to be submitted
through a driver. oneDNN's C++ initialization and execution code handles those
operations through the documented OS interface. The generated kernel receives
the resulting buffers, addresses, or parameters. The contribution should name
the actual interface, such as a device mapping or an `ioctl`, and describe
which operations happen during initialization and which happen per execution.

For example, a driver-backed data-movement path could follow this sequence:

```text
C++ initialization
    -> open the device and prepare reusable driver resources
C++ execution
    -> acquire buffers for this invocation
    -> submit the input transfer and wait for input readiness
JIT kernel
    -> compute using the prepared buffers
C++ execution
    -> submit any output transfer and wait for completion
    -> release invocation resources
C++ cleanup
    -> release reusable resources and close the device
```

Transfers may overlap with computation when dependencies and completion follow
oneDNN's stream semantics. Resource ownership must support
concurrent executions of a primitive; reusable device resources and per-call
buffers should have clearly defined lifetimes.

On-chip memory follows the same division. If the OS exposes it as mapped
user-addressable memory, C++ code obtains the mapping and passes its address to
the kernel. If access uses special user-mode instructions, those instructions
are emitted by the JIT. Instructions requiring kernel privilege must be used
through the driver interface. An interface requiring a process-wide device
context or core enrollment must describe how the application or runtime
establishes it before execution.

For both instruction-based and driver-backed paths, document the required
kernel and driver support, permissions, memory alignment and accessibility,
cache-coherency rules, and completion handling. Initialization failures must
release acquired resources, and execution errors must be returned through
oneDNN's status handling. Include transfer, setup, and synchronization costs
in performance measurements.

**A required vendor runtime library needs a separate RFC.** The RFC must name
the library's license and describe any limits on use or redistribution. Adding
it to the default build normally requires independent evidence that more than
20% of RV64 devices support the feature. Cite the source, date, and market
scope. Maintainers may make an exception based on measured benefits. Opt-in
libraries have no adoption threshold.

If a vendor library is needed to manage the device, memory, or submissions,
its RFC describes that role, the interface oneDNN calls, and how the library
is packaged and updated. Reviewers weigh its size and dependency cost against
measured performance, adoption, and maintenance.

#### Build-time selection of JIT extensions

Route 1 allows build-time selection of JIT ISA extensions to control binary
size. Users can omit extension-specific kernel generators and their dispatch
registrations when those implementations are unnecessary for their deployment.
The selection should use a common RV64 mechanism rather than separate build
logic for each primitive. Compiling out an extension must also remove its
exclusive code and dependencies.

Build-time selection determines which implementations are included. Runtime
detection then determines which included implementations the hardware and OS
can execute. Keep the portable compiler ISA baseline unchanged.
Existing standard RVV and scalar fallbacks remain available according to the
normal build configuration. Tests should cover both inclusion and exclusion
of each selectable extension, and contributions should report the code-size
impact of enabling it.

### Route 2: Vendor-library integration

A vendor may already have an optimized operator library. This route connects
it to oneDNN through an adapter. Treat this route as secondary to native JIT
integration. Closed-source libraries are eligible.

**Each new library needs its own approved RFC before integration.** The RFC
describes supported operators and configurations, performance, binary size,
threading, hardware adoption, and maintenance. It must name the library's
license and describe any limits on use or redistribution. For a closed-source
library, explain how users obtain and update it. Also describe hardware
detection, OS requirements, and handling of heterogeneous CPUs.

**Open-source libraries use a Git submodule pinned to a reviewed commit.** Users
initialize it when enabling the library's optional build configuration. Library
updates change that recorded commit through code review. Closed-source
libraries may instead be supplied separately by users; they do not require a
submodule.

The adapter translates oneDNN descriptors and arguments into library calls.
It joins the existing primitive-selection process and checks that the library
supports the requested operation and execution CPU set. Other configurations
continue through oneDNN's fallback implementations.

### Compatibility and validation

Vendor ISA extensions must be verifiable for correctness by the public using
a publicly available QEMU distribution. Both upstream QEMU releases and
vendor-distributed versions are accepted. Each PR must state the QEMU version
and the versions of the implementation and test environment used.

The PR author must provide performance measurements on real hardware. At
least one set of measurements must use a benchdnn input batch with sufficiently
broad coverage of the workloads supported by the optimization.

### Deprecating and removing vendor ISA support

Consider removing vendor ISA support when few devices use it, a standard ISA
replaces it, or its maintenance cost outweighs its measured benefit.

The removal process is:

1. **Open a tracking issue.** List the affected extensions, CPU generations,
   primitives, and build selections. Explain the reason for removal and give
   performance or maintenance evidence. Notify contributors and users in the
   relevant issues and identify the fallback or replacement implementation.
2. **Announce deprecation.** After maintainer agreement, document it in the
   release notes and support documentation. Keep the implementation available
   for at least one complete release cycle after that announcement, and state
   the earliest release in which removal may occur. During this period,
   contributors can address the reported issues or provide a replacement.
3. **Check the replacement.** Confirm its correctness,
   supported data types and layouts, ISA requirements, and performance on the
   affected hardware where available. Record any loss of functionality or
   performance in the tracking issue and release notes.
4. **Remove the oneDNN implementation.** Delete the obsolete JIT kernels,
   dispatch registrations, exclusive tuning and detection-table entries,
   tests, build selections, and documentation. Preserve shared components
   still used by other implementations. If a removed build option is selected,
   report the replacement option.
5. **Remove unused dependencies.** Stop relying on obsolete vendor support
   in oneDNN's Xbyak_riscv integration. Removal of encodings or detection APIs
   from Xbyak_riscv itself follows that project's review process. For Route 2,
   remove an unused adapter and submodule through the same deprecation process.

If removal raises the minimum ISA or removes user-visible functionality,
document that change in the release notes.
Urgent correctness or security problems may require disabling a path before
the normal deprecation period; explain that decision and the fallback in the
tracking issue and release notes.

### First integration

The first target is matmul on SpacemiT A100 using IME:

1. Make existing RV64 JIT kernels selectable at build time, including their
   dispatch entries. Keep all kernels enabled by default.
2. Update Xbyak_riscv to detect ISA support and VLEN for a specified CPU group.
   Check X100 and A100 separately on K3.
3. Add IME detection, instruction encodings, and tests to Xbyak_riscv. Linux
   must first expose the required CPU identification. Provide a public QEMU
   build that runs the IME tests.
4. Update oneDNN's Xbyak_riscv version and add an IME GEMM/BRGEMM kernel with
   runtime dispatch to A100.
5. Run correctness tests in QEMU and performance tests on A100, including
   benchdnn batches that cover the supported matmul workloads.
