[![UXL Foundation Logo](https://github.com/uxlfoundation/artwork/blob/main/foundation/uxl-foundation-logo-horizontal-color.png)][UXL Foundation]

# oneAPI Deep Neural Network Library (oneDNN)

<div align="center">

[![Release](https://img.shields.io/github/v/release/uxlfoundation/oneDNN)](https://github.com/uxlfoundation/oneDNN/releases)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/onednn)](https://anaconda.org/conda-forge/onednn)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://uxlfoundation.github.io/oneDNN)
[![Slack](https://img.shields.io/badge/slack-%23onednn-4A154B?logo=slack)](https://slack-invite.uxlfoundation.org/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8762/badge)](https://www.bestpractices.dev/projects/8762)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/uxlfoundation/oneDNN/badge)](https://securityscorecards.dev/viewer/?uri=github.com/uxlfoundation/oneDNN)

[Highlights](#highlights) | [Used By](#used-by) | [Quick Start](#quick-start) |
[Installation](#installation) | [Documentation](#documentation) |
[System Requirements](#system-requirements)

[Support](#support) | [Governance](#governance) | [Contributing](#contributing) |
[License](#license) | [Security](#security)

</div>


**oneDNN is an open-source, cross-platform library of optimized deep learning
operations for CPUs and GPUs.** It provides highly optimized
implementations of matrix multiplication, convolution, attention,
normalization, and other operations, and is the acceleration layer behind
PyTorch, TensorFlow, OpenVINO, and many other AI frameworks.

The library is optimized for Intel 64/AMD64 architecture-based processors,
Arm(R) 64-bit Architecture (AArch64)-based processors,
RISC-V 64-bit (RV64)-based processors, and Intel Graphics.
oneDNN has experimental support for the following architectures: NVIDIA\* GPU,
AMD\* GPU, OpenPOWER\* Power ISA (PPC64), and IBMz\* (s390x).

The oneDNN project is part of the [UXL Foundation].

[UXL Foundation]: https://www.uxlfoundation.org

## Highlights

* **Single API, many devices:** The same code targets Intel 64/AMD64, AArch64,
  and RISC-V CPUs, and Intel GPUs.
* **Hardware specialization:** oneDNN detects hardware features at runtime
  and JIT-generates kernels specialized for the instruction set, number of
  cores, and cache sizes.
* **Low-precision inference:** Supports `f32`, `bf16`, `f16`, `f8`, `f4`,
  `u8`/`s8`, and `u4`/`s4` data types with various quantization schemes,
  including [OCP microscaling formats].
* **Operation fusion:** Post-ops fuse activations, residual connections, and
  quantization into matmul and convolution. The [Graph API] fuses larger
  patterns such as scaled dot-product attention (SDPA) and grouped-query
  attention (GQA).
* **Interoperability:** Supports OpenMP, TBB, and custom threadpools on CPUs,
  and SYCL and OpenCL on GPUs.

[OCP microscaling formats]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
[Graph API]: https://uxlfoundation.github.io/oneDNN/dev_guide_graph_basic_concepts.html

## Used By

oneDNN is integrated into the following frameworks and applications. Deep
learning practitioners should use one of these rather than oneDNN directly:
* [Apache SINGA](https://singa.apache.org)
* [DeepLearning4J\*](https://deeplearning4j.konduit.ai)
* [Flashlight\*](https://github.com/flashlight/flashlight)
* [llama.cpp](https://github.com/ggml-org/llama.cpp)
* [ONNX Runtime](https://onnxruntime.ai)
* [OpenNMT CTranslate2](https://github.com/OpenNMT/CTranslate2)
* [OpenVINO(TM) toolkit](https://github.com/openvinotoolkit/openvino)
* [PaddlePaddle\*](https://www.paddlepaddle.org)
* [PyTorch\*](https://pytorch.org)
* [TensorFlow\*](https://www.tensorflow.org)

## Quick Start

Install pre-built oneDNN from [conda-forge]:

```sh
conda install -c conda-forge onednn
```

Build and run the [matrix multiplication performance example], which
measures matmul throughput for `f32`, `f16`, `bf16`, and `s8` data types:

```sh
wget https://raw.githubusercontent.com/uxlfoundation/oneDNN/main/examples/matmul_perf.cpp \
     https://raw.githubusercontent.com/uxlfoundation/oneDNN/main/examples/example_utils.hpp
g++ matmul_perf.cpp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -ldnnl -o matmul_perf
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./matmul_perf cpu
```

Next steps:
* [Examples and Tutorials] cover quantization, fusion, Graph API, GPU
  execution, and more.
* [Build from Source] to customize the build.

If oneDNN is useful in your work, consider giving the project a star on
GitHub. It helps others discover it.

[matrix multiplication performance example]: examples/matmul_perf.cpp
[Examples and Tutorials]: https://uxlfoundation.github.io/oneDNN/dev_guide_examples.html
[Build from Source]: https://uxlfoundation.github.io/oneDNN/dev_guide_build.html

## Installation

You can download and install the oneDNN library using one of the following options:

- [conda-forge]: If the configuration you need is not available on
  the conda-forge channel, you can build the library using the
  Source Distribution.
- Intel oneAPI:
  - [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html)
  - [Intel oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn-download.html)
- Source Distribution: You can build the library from source by
  following the instructions on the [Build from Source] page.

[conda-forge]: https://anaconda.org/conda-forge/onednn

## Documentation

* [oneDNN Developer Guide and Reference] explains the programming
  model, supported functionality, implementation details, and includes
  annotated examples.
* [API Reference] provides a comprehensive reference of the library
  API.
* [Release Notes] explain the new features, performance
  optimizations, and improvements implemented in each version of
  oneDNN.

[oneDNN Developer Guide and Reference]: https://uxlfoundation.github.io/oneDNN
[API Reference]: https://uxlfoundation.github.io/oneDNN/group_dnnl_api.html
[Release Notes]: https://github.com/uxlfoundation/oneDNN/releases

## System Requirements

oneDNN supports platforms based on the following architectures:
- [Intel 64/AMD64](https://en.wikipedia.org/wiki/X86-64)
- [Arm 64-bit Architecture (AArch64)](https://developer.arm.com/architectures/cpu-architecture/a-profile)
- [RISC-V 64-bit (RV64)](https://en.wikipedia.org/wiki/RISC-V)
- **[experimental]** [OpenPOWER](https://openpowerfoundation.org/) / [IBM Power ISA](https://en.wikipedia.org/wiki/Power_ISA)
- **[experimental]** [IBMz z/Architecture (s390x)](https://en.wikipedia.org/wiki/Z/Architecture)

> **WARNING**
>
> Power ISA (PPC64) and IBMz (s390x) support is **experimental** with limited
> validation.

The library is optimized for the following CPUs:
* Intel 64/AMD64 architecture
  * Intel Xeon(R) processor E3, E5, and E7 family v3+ lineups (formerly Haswell and Broadwell)
  * Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper
    Lake, Ice Lake, Sapphire Rapids, and Emerald Rapids)
  * Intel Xeon CPU Max Series (formerly Sapphire Rapids HBM)
  * Intel Core Ultra processors (formerly Meteor Lake, Arrow Lake,
    Lunar Lake, and Panther Lake)
  * Intel Xeon 6 processors (formerly Sierra Forest and Granite Rapids)
  * future Intel Core processor with Intel AVX10.2 instruction set support
    (code name Nova Lake)
  * future Intel Xeon processor with Intel AVX10.2 instruction set support
    (code name Diamond Rapids)
* AArch64 architecture
  * Arm Neoverse(TM) N-Series and V-Series
* RISC-V architecture-based processors with the following ISA extensions:
  * RISC-V Vector (RVV) extension, version 1.0
  * Zvfh (FP16 vector) and Zvfbfwma (BF16 vector)

On a CPU based on Intel 64/AMD64, AArch64, or RISC-V architecture, oneDNN detects
the instruction set architecture (ISA) at runtime and uses just-in-time (JIT)
code generation to deploy the code optimized for the latest supported ISA.
Future ISAs may have initial support in the library disabled by default and
require the use of run-time controls to enable them. See
[CPU dispatcher control] for more details.

> **WARNING**
>
> On macOS, applications that use oneDNN may need to request special
> entitlements if they use the hardened runtime. See the
> [Linking Guide] for more details.

The library is optimized for the following GPUs:
* Intel discrete GPUs:
  * Intel Arc(TM) A-Series Graphics (formerly Alchemist)
  * Intel Data Center GPU Flex Series (formerly Arctic Sound)
  * Intel Data Center GPU Max Series (formerly Ponte Vecchio)
  * Intel Arc B-Series Graphics and Intel Arc Pro B-Series Graphics
   (formerly Battlemage)
  * Intel GPU D-Series (formerly Crescent Island)
* Intel Graphics integrated with:
  * Intel Graphics for Intel Core Ultra Series 1 processors (formerly Meteor Lake)
  * Intel Graphics for Intel Core Ultra Series 2 processors (formerly Arrow Lake and Lunar Lake)
  * Intel Graphics for Intel Core Ultra Series 3 processors (formerly Panther Lake)
  * Intel Graphics for Intel Core Series 3 processors (formerly Wildcat Lake)
  * Intel Graphics for future Intel Core Ultra processors (code name Nova Lake)

[CPU dispatcher control]: https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html
[Linking Guide]: https://uxlfoundation.github.io/oneDNN/dev_guide_link.html

### Requirements for Building from Source

oneDNN supports systems meeting the following requirements:
* Operating system with Intel 64/AMD64, AArch64, RV64, PPC64, or s390x
  architecture support
* C++ compiler with C++11 standard support
* [CMake] 3.13 or later

The following tools are required to build oneDNN documentation:
* [Doxygen] 1.8.5 or later
* [Doxyrest] 2.1.2 or later
* [Sphinx] 7.4.7 or later
* [sphinx-book-theme] 1.2.0 or later
* [sphinx-copybutton] 0.5.2 or later
* [graphviz] 2.40.1

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

[CMake]: https://cmake.org/download/
[Doxygen]: https://www.doxygen.nl/download.html#srcbin
[Doxyrest]: https://github.com/vovkos/doxyrest
[Sphinx]: https://www.sphinx-doc.org/en/master/usage/installation.html
[sphinx-book-theme]: https://sphinx-book-theme.readthedocs.io/en/latest
[sphinx-copybutton]: https://sphinx-copybutton.readthedocs.io/en/latest/
[graphviz]: https://www.linuxfromscratch.org/blfs/view/8.2/general/graphviz.html

#### CPU Engine

The oneDNN CPU engine is used to execute primitives on Intel 64/AMD64-based
processors, 64-bit Arm Architecture (AArch64) processors, RISC-V 64-bit (RV64)
processors, 64-bit Power ISA (PPC64) processors, IBMz (s390x), and compatible
devices.

The CPU engine is built by default but can be disabled at build time by setting
`ONEDNN_CPU_RUNTIME` to `NONE`. In this case, the GPU engine must be enabled.
The CPU engine can be configured to use the OpenMP, TBB, SYCL, or threadpool
runtime. The following additional requirements apply:
* OpenMP runtime requires C++ compiler with OpenMP 2.0 or later
  standard support
* TBB runtime requires [Threading Building Blocks (TBB)] 2017 or later
* SYCL runtime requires
  * [Intel oneAPI DPC++/C++ Compiler]
  * [Threading Building Blocks (TBB)]

Some implementations rely on OpenMP 4.0 SIMD extensions. For the best
performance results on Intel Architecture Processors, we recommend using the
[Intel oneAPI DPC++/C++ Compiler].

[Threading Building Blocks (TBB)]: https://www.threadingbuildingblocks.org/
[Intel oneAPI DPC++/C++ Compiler]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html

On a CPU based on Arm AArch64 architecture, oneDNN CPU engine can be built with
[Arm Compute Library (ACL)] integration. ACL is an open-source library for
machine learning applications and provides AArch64-optimized implementations
of core functions. This functionality currently requires that ACL is downloaded
and built separately. See the [Build from Source] section of the Developer Guide for
details. The minimum supported version of ACL is 53.1.0.

[Arm Compute Library (ACL)]: https://github.com/arm-software/ComputeLibrary

#### GPU Engine

The oneDNN GPU engine is used to execute primitives on various accelerators
including Intel integrated and discrete GPUs, NVIDIA GPUs, AMD GPUs, and
other devices supporting the SYCL programming language. The GPU engine is
disabled in the default build configuration and can be enabled by setting the
`ONEDNN_GPU_RUNTIME` build option to a value other than `NONE`. The target
accelerator vendor must be selected at build time using the
`ONEDNN_GPU_VENDOR` build option.

> **WARNING**
>
> Linux resets the GPU when kernel runtime exceeds several seconds. The user
> can prevent this behavior by [disabling hangcheck] for Intel GPU driver.
> Windows has a built-in [timeout detection and recovery] mechanism that results
> in similar behavior. The user can prevent this behavior by increasing the
> [TdrDelay] value.

The following additional requirements apply for Intel integrated and discrete
GPUs:
* With OpenCL(TM) runtime:
  * OpenCL SDK (with OpenCL 1.2 support)
  * Intel Graphics Driver with support for OpenCL C 2.0, Intel subgroups
  support, and USM extensions support
* With SYCL runtime:
  * [Intel oneAPI DPC++/C++ Compiler]
  * OpenCL SDK (with OpenCL 3.0 support)
  * [oneAPI Level Zero] with support for API v1.11 or later
  * Intel Graphics Driver with support for OpenCL C 2.0, Intel subgroups
    support, and USM extensions support

The following additional requirements apply for NVIDIA GPUs:
* [oneAPI DPC++ Compiler with support for CUDA] or [oneAPI for NVIDIA GPUs]
* NVIDIA CUDA\* driver
* cuBLAS 10.1 or later
* cuDNN 7.6 or later

> **WARNING**
>
> NVIDIA GPU support is experimental. General information, build instructions,
> and implementation limitations are available in the
> [NVIDIA backend readme](src/gpu/nvidia/README.md).

The following additional requirements apply for AMD GPUs:
* [oneAPI DPC++ Compiler with support for HIP AMD] or [oneAPI for AMD GPUs]
* [AMD ROCm] version 5.3 or later
* [MIOpen] version 2.18 or later (optional if AMD ROCm includes
  the required version of MIOpen)
* [rocBLAS] version 2.45.0 or later (optional if AMD ROCm includes
  the required version of rocBLAS)

> **WARNING**
>
> AMD GPU support is experimental. General information, build instructions,
> and implementation limitations are available in the
> [AMD backend readme](src/gpu/amd/README.md).

Other devices supporting the SYCL programming model require
oneAPI DPC++/C++ Compiler that supports the target GPU. Refer to
[generic GPU vendor] documentation for additional details.

[oneAPI Level Zero]: https://github.com/oneapi-src/level-zero
[oneAPI DPC++ Compiler with support for CUDA]: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda
[oneAPI for NVIDIA GPUs]: https://developer.codeplay.com/products/oneapi/nvidia/home
[oneAPI DPC++ Compiler with support for HIP AMD]: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-hip-amd
[oneAPI for AMD GPUs]: https://developer.codeplay.com/products/oneapi/amd/home/
[AMD ROCm]: https://github.com/ROCm/ROCm
[MIOpen]: https://github.com/ROCm/MIOpen
[rocBLAS]: https://github.com/ROCm/rocBLAS
[disabling hangcheck]: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/gpu-disable-hangcheck.html
[timeout detection and recovery]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
[TdrDelay]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay
[generic GPU vendor]: src/gpu/generic/sycl/README.md

### Runtime Dependencies

When oneDNN is built from source, the library runtime dependencies and specific
versions are defined by the build environment.

#### Linux

Common dependencies:
* GNU C Library (`libc.so`)
* GNU Standard C++ Library v3 (`libstdc++.so`)
* Dynamic Linking Library (`libdl.so`)
* C Math Library (`libm.so`)
* POSIX Threads Library (`libpthread.so`)

Runtime-specific dependencies:

| Runtime configuration      | Compiler                      | Dependency
| :------------------------- | :---------------------------- | :---------
| `ONEDNN_CPU_RUNTIME=OMP`   | GCC                           | GNU OpenMP runtime (`libgomp.so`)
| `ONEDNN_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.so`)
| `ONEDNN_CPU_RUNTIME=OMP`   | Clang                         | Intel OpenMP runtime (`libiomp5.so`)
| `ONEDNN_CPU_RUNTIME=TBB`   | any                           | TBB (`libtbb.so`)
| `ONEDNN_CPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), TBB (`libtbb.so`), OpenCL loader (`libOpenCL.so`)
| `ONEDNN_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`libOpenCL.so`)
| `ONEDNN_GPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), OpenCL loader (`libOpenCL.so`), oneAPI Level Zero loader (`libze_loader.so`)
| `ONEDNN_GPU_RUNTIME=ZE`    | any                           | oneAPI Level Zero loader (`libze_loader.so`)

#### Windows

Common dependencies:
* Microsoft Visual C++ Redistributable (`vcruntime140.dll`, `msvcp140.dll`)

Runtime-specific dependencies:

| Runtime configuration      | Compiler                      | Dependency
| :------------------------- | :---------------------------- | :---------
| `ONEDNN_CPU_RUNTIME=OMP`   | Microsoft Visual C++ Compiler | No additional requirements
| `ONEDNN_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`iomp5.dll`)
| `ONEDNN_CPU_RUNTIME=TBB`   | any                           | TBB (`tbb.dll`)
| `ONEDNN_CPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), TBB (`tbb.dll`), OpenCL loader (`OpenCL.dll`)
| `ONEDNN_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`OpenCL.dll`)
| `ONEDNN_GPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), OpenCL loader (`OpenCL.dll`), oneAPI Level Zero loader (`ze_loader.dll`)
| `ONEDNN_GPU_RUNTIME=ZE`    | any                           | oneAPI Level Zero loader (`ze_loader.dll`)

#### macOS

Common dependencies:
* System C/C++ runtime (`libc++.dylib`, `libSystem.dylib`)

Runtime-specific dependencies:

| Runtime configuration    | Compiler                      | Dependency
| :----------------------- | :---------------------------- | :---------
| `ONEDNN_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.dylib`)
| `ONEDNN_CPU_RUNTIME=TBB` | any                           | TBB (`libtbb.dylib`)

## Support

Ask questions and share your projects in [GitHub Discussions]. Submit
feature requests and bug reports on the [GitHub issues] page.

You can also contact oneDNN developers via [UXL Foundation Slack] in the
[#onednn] channel.

[GitHub Discussions]: https://github.com/uxlfoundation/oneDNN/discussions
[GitHub issues]: https://github.com/uxlfoundation/oneDNN/issues
[UXL Foundation Slack]: https://slack-invite.uxlfoundation.org/
[#onednn]: https://uxlfoundation.slack.com/channels/onednn

## Governance

The oneDNN project is governed by the [UXL Foundation], and you can get
involved in this project in multiple ways. You can join the [AI Special Interest
Group (SIG)] meetings where the groups discuss and demonstrate work using this
project. Members can also join the Open Source and Specification Working Group
meetings.

You can also join the [mailing lists for the UXL Foundation] to be informed
of when meetings are happening and receive the latest information and
discussions.

[AI Special Interest Group (SIG)]: https://github.com/uxlfoundation/foundation
[mailing lists for the UXL Foundation]: https://lists.uxlfoundation.org/g/main/subgroups

## Contributing

We welcome community contributions to oneDNN. You can find the oneDNN release
schedule and work already in progress towards future milestones in GitHub's
[Milestones] section. If you are looking for a specific task to start,
consider selecting from issues that are marked with the [help wanted] label.

See [contribution guidelines](CONTRIBUTING.md) to start contributing
to oneDNN. You can also contact oneDNN developers and maintainers via
[UXL Foundation Slack] in the [#onednn] channel.

This project is intended to be a safe, welcoming space for
collaboration, and contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

[Milestones]: https://github.com/uxlfoundation/oneDNN/milestones
[help wanted]: https://github.com/uxlfoundation/oneDNN/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22

## License

oneDNN is licensed under [Apache License Version 2.0](LICENSE). Refer
to the "[LICENSE](LICENSE)" file for the full license text and
copyright notice.

This distribution includes third-party software governed by separate
license terms.

3-clause BSD license:
* [Xbyak](https://github.com/herumi/xbyak)
* [Xbyak_riscv](https://github.com/herumi/xbyak_riscv)
* [gtest](https://github.com/google/googletest)
* [Instrumentation and Tracing Technology API
(ITT API)](https://github.com/intel/ittapi)
* [CMake](https://github.com/Kitware/CMake)

2-clause BSD license:
* [Sphinx](https://www.sphinx-doc.org/)

Apache License Version 2.0:
* [Xbyak_aarch64](https://github.com/fujitsu/xbyak_aarch64)
* [OpenCL(TM) API Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [Arm(R) KleidiAI(TM)](https://gitlab.arm.com/kleidi/kleidiai)

Boost Software License, Version 1.0:
* [Boost C++ Libraries](https://www.boost.org/)

MIT License:
* [Intel Graphics Compute Runtime for oneAPI Level Zero
and OpenCL Driver](https://github.com/intel/compute-runtime)
* [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
* [oneAPI Level Zero](https://github.com/oneapi-src/level-zero)
* [Doxyrest](https://github.com/vovkos/doxyrest)
* [Intel Metrics Discovery Application Programming
Interface](https://github.com/intel/metrics-discovery)
* [spdlog](https://github.com/gabime/spdlog)
* [sphinx-copybutton](https://github.com/executablebooks/sphinx-copybutton)

This third-party software, even if included with the distribution of
other software, may be governed by separate license terms,
including without limitation, third-party license terms, other
software license terms, and open source software license terms. These
separate license terms govern your use of the third-party programs as
set forth in the "[THIRD-PARTY-PROGRAMS](THIRD-PARTY-PROGRAMS)" file.

## Security

[Security Policy](SECURITY.md) outlines our guidelines and procedures
for ensuring the highest level of security and trust for our users
who consume oneDNN.

-------------------------------------------------------------------------------

[Legal Information](doc/legal_information.md)
