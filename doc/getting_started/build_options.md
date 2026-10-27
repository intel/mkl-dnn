Build Options {#dev_guide_build_options}
============================================

oneDNN provides extensive configuration capabilities via build options:
* [Common options](@ref opt_common) manage library platform-agnostic
  capabilities.
* [CPU options](@ref opt_cpu) manage behavior of CPU engine and
  platform-specific code generation for CPUs.
* [GPU options](@ref opt_gpu) manage behavior of GPU engine.

All other building options or values that can be found in CMake files are
intended for development/debug purposes and are subject to change without
notice. Please avoid using them.

@anchor opt_common
## Common options

These options apply to the whole library regardless of the target engine.

### Library configuration

| CMake Option                    | Default     | Supported values   | Description                                                                                |
|:--------------------------------|:------------|:-------------------|:-------------------------------------------------------------------------------------------|
| ONEDNN_LIBRARY_TYPE             | **SHARED**  | STATIC             | Defines the resulting library type                                                         |
| ONEDNN_LIBRARY_NAME             | **dnnl**    | \<string\>         | Specifies name of the library                                                              |
| [ONEDNN_ARCH_OPT_FLAGS]         | *varies*    | \<compiler flags\> | Specifies compiler optimization flags. Default value depends on the platform and compiler. |
| [ONEDNN_DPCPP_HOST_COMPILER]    | **DEFAULT** | g++, clang++, cl   | Specifies host compiler executable for SYCL runtime                                        |

[ONEDNN_ARCH_OPT_FLAGS]: @ref opt_arch_opt_flags
[ONEDNN_DPCPP_HOST_COMPILER]: @ref opt_dpcpp_host_compiler

@anchor opt_arch_opt_flags
#### ONEDNN_ARCH_OPT_FLAGS

oneDNN uses JIT code generation to implement most of its functionality
and will choose the best code based on detected processor features. However,
some oneDNN functionality will still benefit from targeting a specific
processor architecture at build time. You can use `ONEDNN_ARCH_OPT_FLAGS` CMake
option for this.

For Intel(R) C++ Compilers, the default option is `-xSSE4.1`, which instructs
the compiler to generate the code for the processors that support SSE4.1
instructions. This option would not allow you to run the library on
older processor architectures.

For GNU\* Compilers and Clang, the default option is `-msse4.1`.

@warning
While use of `ONEDNN_ARCH_OPT_FLAGS` option gives better performance, the
resulting library can be run only on systems that have instruction set
compatible with the target instruction set. Therefore, `ONEDNN_ARCH_OPT_FLAGS`
should be set to an empty string (`""`) if the resulting library needs to be
portable.

@anchor opt_dpcpp_host_compiler
#### ONEDNN_DPCPP_HOST_COMPILER

When building oneDNN with oneAPI DPC++/C++ Compiler user can specify a custom
host compiler. The host compiler is a compiler that will be used by the main
compiler driver to perform host compilation step.

The host compiler can be specified with `ONEDNN_DPCPP_HOST_COMPILER` CMake
option. It should be specified either by name (in this case, the standard system
environment variables will be used to discover it) or an absolute path to the
compiler executable.

The default value of `ONEDNN_DPCPP_HOST_COMPILER` is `DEFAULT`, which is the
default host compiler used by the compiler specified with `CMAKE_CXX_COMPILER`.

The library supports GNU C++ compiler or Clang as the host compiler on Linux and
Microsoft Visual C++ Compiler (`cl`) on Windows. With `ONEDNN_CPU_RUNTIME=OMP`
the library uses OpenMP implementation provided by the host compiler instead
of the Intel OpenMP runtime.

@warning
The minimum allowed host compiler versions are:
* GNU C++ 8.0.0
* Clang C++ 11.0.0
* Microsoft Visual C++ Compiler 19.29 (Visual Studio 2019 version 16.10)

See [GCC* Compatibility and Interoperability] section in
oneAPI DPC++/C++ Compiler Developer Guide.

[GCC* Compatibility and Interoperability]: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/gcc-compatibility-and-interoperability.html

### Functionality

| CMake Option                  | Default      | Supported values | Description                                                                                     |
|:------------------------------|:-------------|:-----------------|:------------------------------------------------------------------------------------------------|
| ONEDNN_BUILD_GRAPH            | **ON**       | OFF              | Controls building graph component                                                               |
| [ONEDNN_ENABLE_WORKLOAD]      | **TRAINING** | INFERENCE        | Specifies a set of functionality to be available based on workload                              |
| [ONEDNN_ENABLE_PRIMITIVE]     | **ALL**      | \<list\>         | Specifies a set of functionality to be available based on primitives                            |
| ONEDNN_ENABLE_CONCURRENT_EXEC | **OFF**      | ON               | Disables sharing a common scratchpad between primitives in #dnnl::scratchpad_mode::library mode |
| ONEDNN_ENABLE_PRIMITIVE_CACHE | **ON**       | OFF              | Enables [primitive cache](@ref dev_guide_primitive_cache)                                       |
| ONEDNN_EXPERIMENTAL           | **OFF**      | ON               | Enables [experimental features](@ref dev_guide_experimental)                                    |

[ONEDNN_ENABLE_WORKLOAD]: @ref opt_enable_workload
[ONEDNN_ENABLE_PRIMITIVE]: @ref opt_enable_primitive

Using `ONEDNN_ENABLE_WORKLOAD` and `ONEDNN_ENABLE_PRIMITIVE` it is possible to
limit functionality available in the final shared object or statically linked
application. This helps to reduce the amount of disk space occupied by an app.

@anchor opt_enable_workload
#### ONEDNN_ENABLE_WORKLOAD

This option supports only two values: `TRAINING` (the default) and `INFERENCE`.
`INFERENCE` enables only forward propagation kind part of functionality,
removing all backward-related functionality, except those which are
dependencies for forward propagation kind part.

@anchor opt_enable_primitive
#### ONEDNN_ENABLE_PRIMITIVE

This option supports several values: `ALL` (the default) which enables all
primitives implementations or any subset of the following list: `BATCH_NORMALIZATION`,
`BINARY`, `CONCAT`, `CONVOLUTION`, `DECONVOLUTION`, `ELTWISE`, `GROUP_NORMALIZATION`,
`INNER_PRODUCT`, `LAYER_NORMALIZATION`, `LRN`, `MATMUL`, `POOLING`, `PRELU`,
`REDUCTION`, `REORDER`, `RESAMPLING`, `RNN`, `SDPA`, `SHUFFLE`, `SOFTMAX`,
`SUM`. When a set is used, only those selected primitives implementations will
be available. Attempting to use other primitive implementations will end up
returning an unimplemented status when creating primitive descriptor. In order
to specify a set, a CMake-style string should be used, with semicolon
delimiters, as in this example:
~~~sh
-DONEDNN_ENABLE_PRIMITIVE=CONVOLUTION;MATMUL;REORDER
~~~

@note
Graph API (enabled via `ONEDNN_BUILD_GRAPH`) is not compatible with
`ONEDNN_ENABLE_PRIMITIVE` values other than `ALL`.

### Profiling and debug

| CMake Option                | Default | Supported values | Description                                                                                |
|:----------------------------|:--------|:-----------------|:-------------------------------------------------------------------------------------------|
| ONEDNN_ENABLE_JIT_PROFILING | **ON**  | OFF              | Enables [integration with performance profilers](@ref dev_guide_profilers)                 |
| ONEDNN_ENABLE_ITT_TASKS     | **ON**  | OFF              | Enables [integration with performance profilers](@ref dev_guide_profilers)                 |
| ONEDNN_ENABLE_GRAPH_DUMP    | **ON**  | OFF              | Controls dumping graph artifacts                                                           |
| ONEDNN_VERBOSE              | **ON**  | OFF              | Enables [verbose mode](@ref dev_guide_verbose)                                             |
| ONEDNN_DEV_MODE             | **OFF** | ON               | Enables internal tracing and `debuginfo` logging in verbose output (for oneDNN developers) |

### Documentation

| CMake Option             | Default | Supported values | Description                                                                                                        |
|:-------------------------|:--------|:-----------------|:-------------------------------------------------------------------------------------------------------------------|
| ONEDNN_BUILD_DOC         | **OFF** | ON               | Controls building the documentation                                                                                |
| ONEDNN_DOC_VERSIONS_JSON | \       | \<url\>          | Location of JSON file for [PyData Sphinx Theme version switcher]. Enables documentation version switcher when set. |

[PyData Sphinx Theme version switcher]: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html

### Validation

| CMake Option                 | Default   | Supported values                                           | Description                                                                  |
|:-----------------------------|:----------|:-----------------------------------------------------------|:-----------------------------------------------------------------------------|
| ONEDNN_BUILD_EXAMPLES        | **ON**    | OFF                                                        | Controls building the examples                                               |
| ONEDNN_BUILD_TESTS           | **ON**    | OFF                                                        | Controls building the tests                                                  |
| [ONEDNN_TEST_SET]            | **CI**    | SMOKE, NIGHTLY, WEEKLY, \<list\>                           | Specifies the testing coverage enabled through the generated testing targets |
| ONEDNN_CODE_COVERAGE         | **NONE**  | GCOV                                                       | Enables code coverage instrumentation                                        |
| [ONEDNN_USE_CLANG_SANITIZER] | \         | Address, Leak, Memory, MemoryWithOrigin, Thread, Undefined | Instructs build system to use a Clang sanitizer                              |
| [ONEDNN_USE_CLANG_TIDY]      | **NONE**  | CHECK, FIX                                                 | Instructs build system to use clang-tidy                                     |
| ONEDNN_WERROR                | **OFF**   | ON                                                         | Enables treating warnings as errors                                          |

[ONEDNN_TEST_SET]: @ref opt_test_set
[ONEDNN_USE_CLANG_SANITIZER]: @ref opt_use_clang_sanitizer
[ONEDNN_USE_CLANG_TIDY]: @ref opt_use_clang_tidy

@note `ONEDNN_BUILD_EXAMPLES` and `ONEDNN_BUILD_TESTS` are disabled
by default when oneDNN is built as a sub-project.

@anchor opt_test_set
#### ONEDNN_TEST_SET

This option specifies testing coverage enabled through testing targets generated
by the build system. The variable consists of two parts: the set value which
defines the number of test cases, and the modifiers for testing commands. The
final string must contain a single value for a set and as many compatible values
for modifiers.

The set value is defined by one of: `SMOKE`, `CI`, `NIGHTLY`, or `WEEKLY`. These
may be used with one of the following modifier values: `NO_CORR`, `ADD_BITWISE`.
The set and modifiers are passed as a semicolon separated list. For example:
~~~sh
-DONEDNN_TEST_SET=CI;NO_CORR
~~~

When `SMOKE` value is specified, it enables a short set of test cases which
verifies that basic library functionality works as expected.
When `CI` value is specified, it enables a regular set of test cases which
verifies that all library supported functionality works as expected.
When `NIGHTLY` value is specified, it enables more comprehensive set of test
cases which verifies that all library supported functionality and all kernel
optimizations work as expected.
When `WEEKLY` value is specified, it enables the most comprehensive set
including real workload problems.

When `NO_CORR` modifier value is specified, it removes correctness validation,
which is set by default, from benchdnn testing targets. It helps to save time
when correctness validation is not necessary.
When `ADD_BITWISE` modifier value is specified, the build system will add an
additional set of tests with a bitwise validation mode for benchdnn. The
correctness set remains unmodified.

@anchor opt_use_clang_sanitizer
#### ONEDNN_USE_CLANG_SANITIZER

Instructs build system to use a Clang sanitizer. Supported values:
* Address: enables AddressSanitizer
* Leak: enables LeakSanitizer
* Memory: enables MemorySanitizer
* MemoryWithOrigin: enables MemorySanitizer with origin tracking
* Thread: enables ThreadSanitizer
* Undefined: enables UndefinedBehaviourSanitizer

This feature is only available on Linux.

@anchor opt_use_clang_tidy
#### ONEDNN_USE_CLANG_TIDY

Instructs build system to use clang-tidy. Valid values:
* NONE (default): Clang-tidy is disabled.
* CHECK: Enables checks from .clang-tidy.
* FIX: Enables checks from .clang-tidy and fix found issues.

This feature is only available on Linux.

@anchor opt_cpu
## CPU Options

### Common CPU options

| CMake Option         | Default     | Supported values                 | Description                                                          |
|:---------------------|:------------|:---------------------------------|:---------------------------------------------------------------------|
| [ONEDNN_CPU_RUNTIME] | **OMP**     | NONE, TBB, SEQ, THREADPOOL, SYCL | Defines the threading runtime for CPU engines                        |
| [ONEDNN_BLAS_VENDOR] | **NONE**    | ARMPL, ACCELERATE, ANY           | Defines an external BLAS library to link to for GEMM-like operations |

[ONEDNN_CPU_RUNTIME]: @ref opt_cpu_runtime
[ONEDNN_BLAS_VENDOR]: @ref opt_blas_vendor

@anchor opt_cpu_runtime
#### ONEDNN_CPU_RUNTIME

CPU engine can use OpenMP, Threading Building Blocks (TBB) or sequential
threading runtimes. OpenMP threading is the default build mode. Choose the runtime
that matches threading runtime used in your application.

##### OpenMP
oneDNN uses OpenMP runtime library provided by the compiler.

When building oneDNN with oneAPI DPC++/C++ Compiler the library will link
to Intel OpenMP runtime. This behavior can be changed by changing the host
compiler with `ONEDNN_DPCPP_HOST_COMPILER` option.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

##### Threading Building Blocks (TBB)
To build oneDNN with TBB support, set `ONEDNN_CPU_RUNTIME` to `TBB`:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=TBB ..
~~~

Optionally, set the `TBBROOT` environmental variable to point to the TBB
installation path or pass the path directly to CMake:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=TBB -DTBBROOT=/opt/intel/path/tbb ..
~~~

oneDNN has functional limitations if built with TBB:
* Winograd convolution algorithm is not supported for `f32` backward
  by data and backward by weights propagation.

##### Threadpool
To build oneDNN with support for threadpool threading, set `ONEDNN_CPU_RUNTIME`
to `THREADPOOL`:

~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=THREADPOOL ..
~~~

Threadpool threading support has the same limitations as TBB plus more:
* As threadpools are attached to streams which are only passed during primitive
  execution, work decomposition is performed statically at primitive creation
  time. At the primitive execution time, the threadpool is responsible for
  balancing this decomposition across available worker threads.

###### Threadpool validation
The `_DNNL_TEST_THREADPOOL_IMPL` CMake variable controls which of the four
threadpool implementations would be used for testing: `STANDALONE`, `TBB`,
`EIGEN`, `EIGEN_ASYNC`.

The `TBB` requires passing `TBBROOT` for CMake to find a package.

The `EIGEN` requires Eigen 5.0 or higher and Abseil-CPP packages to be
discoverable by CMake.

The `EIGEN_ASYNC` has same requirements as `EIGEN` and additionally requires
OpenXLA threadpool package, however, additional actions might be required to
compile tests since this threadpool implementation relies on internal OpenXLA
headers. The OpenXLA dependency can be provided either as an install/package
prefix with `xlaConfig.cmake`, or by setting `ONEDNN_XLA_SOURCE_DIR` to an
OpenXLA/XLA source tree. The source-tree path builds the minimal XLA async
runtime subset required by the tests with the same compiler used for oneDNN.

For example:
~~~sh
$ cmake -DONEDNN_CPU_RUNTIME=THREADPOOL -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN -DCMAKE_PREFIX_PATH="/path/to/eigen/share/eigen3/cmake;/path/to/absl/lib64/cmake" ..
$ cmake -DONEDNN_CPU_RUNTIME=THREADPOOL -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN_ASYNC -DEigen3_DIR=/path/to/eigen/share/eigen3/cmake -Dabsl_DIR=/path/to/absl/lib64/cmake/absl -DONEDNN_XLA_SOURCE_DIR=/path/to/xla ..
~~~

@anchor opt_blas_vendor
#### ONEDNN_BLAS_VENDOR

oneDNN can use an external BLAS library to improve performance
of GEMM operations. The following options are supported:
* `NONE` (default): Use internal GEMM implementation.
* `ARMPL`: [Arm Performance Libraries] available on AArch64 CPUs
* `ACCELERATE`: [Accelerate BLAS] available on Apple Silicon
* `ANY`: CMake FindBLAS will search default library paths for one
  of supported BLAS libraries. This option is supported for performance
  analysis purposes only.

[Arm Performance Libraries]: https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries
[Accelerate BLAS]: https://developer.apple.com/documentation/accelerate/blas

### x64 CPU options

| CMake Option                      | Default | Supported values | Description                                                                                    |
|:----------------------------------|:--------|:-----------------|:-----------------------------------------------------------------------------------------------|
| ONEDNN_ENABLE_MAX_CPU_ISA         | **ON**  | OFF              | Enables [CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control)                       |
| [ONEDNN_ENABLE_PRIMITIVE_CPU_ISA] | **ALL** | \<list\>         | Specifies a set of functionality to be available for CPU backend based on CPU ISA              |
| [ONEDNN_ENABLE_GEMM_KERNELS_ISA]  | **ALL** | NONE, \<list\>   | Specifies a set of functionality to be available for GeMM kernels for CPU backend based on ISA |
| ONEDNN_ENABLE_CPU_ISA_HINTS       | **ON**  | OFF              | Enables [CPU ISA hints](@ref dev_guide_cpu_isa_hints)                                          |
| [ONEDNN_SAFE_RBP]                 | **OFF** | ON               | Enables restriction for JIT kernels to pollute RBP vector register content                     |
| [ONEDNN_X64_USE_ZEN]              | **OFF** | ON               | Enables integration with the ZenDNN library for AMD (Zen) CPUs                                  |
| [ONEDNN_ZENDNN_SOURCE]            | [ZenDNN\@v6.1.2][ZenDNN] | \<path\>, \<url\>\@\<tag\> | ZenDNN source to build when no ZenDNN package is found           |

[ONEDNN_ENABLE_PRIMITIVE_CPU_ISA]: @ref opt_enable_primitive_cpu_isa
[ONEDNN_ENABLE_GEMM_KERNELS_ISA]: @ref opt_enable_gemm_kernels_isa
[ONEDNN_SAFE_RBP]: @ref opt_safe_rbp
[ONEDNN_X64_USE_ZEN]: @ref opt_x64_use_zen
[ONEDNN_ZENDNN_SOURCE]: @ref opt_zendnn_source

@anchor opt_enable_primitive_cpu_isa
#### ONEDNN_ENABLE_PRIMITIVE_CPU_ISA

This option supports several values: `ALL` (the default) which enables all
ISA implementations or one of `AVX2`, `AVX512`, and `AMX`. Values are
linearly ordered as `AVX2` < `AVX512` < `AMX`. When specified,
selected ISA and all ISA that are "smaller" will be available. When specified,
[CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control) are also
affected in compliance with the option.

Note that `AVX2` denotes whole AVX2-based family ISAs, `AVX512` denotes whole
AVX512-based family ISAs, as well as `AMX` denotes any ISA containing AMX unit.

Example that enables the AVX2 set:
```
-DONEDNN_ENABLE_PRIMITIVE_CPU_ISA=AVX2
```

@anchor opt_enable_gemm_kernels_isa
#### ONEDNN_ENABLE_GEMM_KERNELS_ISA

This option supports several values: `ALL` (the default) which enables all
ISA kernels from x64/gemm folder, `NONE` which disables all kernels and removes
correspondent interfaces, or one of `AVX2` and `AVX512`. Values are
linearly ordered as `AVX2` < `AVX512`. When specified, selected ISA and all
ISA that are "smaller" will be available. Example that leaves AVX2 kernels
but removes AVX512 and AMX kernels:
```
-DONEDNN_ENABLE_GEMM_KERNELS_ISA=AVX2
```

@anchor opt_safe_rbp
#### ONEDNN_SAFE_RBP

Supported exclusively on x64 CPU architectures for BRGEMM-based primitives.
When enabled (`ON`), this control ensures that JIT-generated kernels preserve
the RBP register state, preventing corruption of frame pointers. This
facilitates accurate stack unwinding and profiler trace collection from
JIT-compiled code regions. Enabling this feature may introduce performance
overhead due to additional register management.

@anchor opt_x64_use_zen
#### ONEDNN_X64_USE_ZEN

This option enables integration with the [ZenDNN] library, which provides
kernels tuned for AMD (Zen) CPUs. It is an opt-in CPU backend that engages at
runtime only on AMD CPUs. The option is `OFF` by default.

~~~sh
$ cmake -DONEDNN_X64_USE_ZEN=ON ..
~~~

When enabled, oneDNN looks for ZenDNN in this order:
1. An installed ZenDNN package, found through `ZENDNNROOT` (CMake variable or
   environment variable), `zendnnl_DIR`, or the default CMake package search
   paths. If `ZENDNNROOT` or `zendnnl_DIR` is set, it must contain a package;
   configure fails otherwise.
2. The source given by [ONEDNN_ZENDNN_SOURCE](@ref opt_zendnn_source), built
   from source. By default this downloads a ZenDNN release.

If a package is found, `ONEDNN_ZENDNN_SOURCE` is ignored; configure with
`-DCMAKE_DISABLE_FIND_PACKAGE_zendnnl=ON` to build from source anyway. Both
static (archive) and shared ZenDNN packages are supported. A source build
links the resulting static archive (`zendnnl::zendnnl_archive`).

Otherwise ZenDNN is compiled as an external project. The first build in a
build directory also downloads ZenDNN's own dependencies (AOCL-DLP,
AOCL-Utils, and nlohmann-json), so it needs network access even when the
ZenDNN source is already in place. Later builds in the same build directory
reuse them.

A static oneDNN built from ZenDNN source installs that ZenDNN and its
dependencies under `${CMAKE_INSTALL_LIBDIR}/zendnn`, and their license files
under `${CMAKE_INSTALL_DATAROOTDIR}/doc/dnnl/zendnn`. The installed oneDNN
package finds ZenDNN there. A static oneDNN built against a ZenDNN
package finds that package instead. When oneDNN is built as a subproject,
ZenDNN is not installed.

When `ONEDNN_X64_USE_ZEN=ON`, the following additional requirements apply
(they do not affect the default `OFF` build):
* Targets Linux on x86_64; Windows builds are rejected at configure time.
* Requires CMake 3.26 or later.
* Requires GCC 11.2 or later, or Clang 14 or later; other compilers are
  rejected.
* Requires ZenDNN version 6.1.2 or later. For a package, the version comes
  from its package config; for a source build, from the `project()`
  declaration in the ZenDNN `CMakeLists.txt`.
* Requires `ONEDNN_CPU_RUNTIME=OMP`; ZenDNN only supports the OpenMP threading
  runtime, and other runtimes are rejected at configure time.

@anchor opt_zendnn_source
#### ONEDNN_ZENDNN_SOURCE

This option selects the ZenDNN source that oneDNN builds when
`ONEDNN_X64_USE_ZEN=ON` and no ZenDNN package is found. It takes one of:
* An absolute path to a ZenDNN source tree, which is used in place.
* `<git url>@<tag or commit>`, which is cloned into the build directory. The
  value is split at the last `@`.

~~~sh
$ cmake -DONEDNN_X64_USE_ZEN=ON -DONEDNN_ZENDNN_SOURCE=/path/to/ZenDNN ..
$ cmake -DONEDNN_X64_USE_ZEN=ON \
    -DONEDNN_ZENDNN_SOURCE=https://github.com/amd/ZenDNN.git@<tag> ..
~~~

Changing `ONEDNN_ZENDNN_SOURCE` in an existing build directory rebuilds
ZenDNN. Edits inside an existing local source tree are not picked up; use a
clean build directory after changing it.

[ZenDNN]: https://github.com/amd/ZenDNN

### AArch64 CPU options

| CMake Option             | Default | Supported values | Description                                                     |
|:-------------------------|:--------|:-----------------|:----------------------------------------------------------------|
| [ONEDNN_AARCH64_USE_ACL] | **OFF** | ON               | Enables integration with Arm Compute Library for AArch64 builds |

[ONEDNN_AARCH64_USE_ACL]: @ref opt_aarch64_use_acl

@anchor opt_aarch64_use_acl
#### ONEDNN_AARCH64_USE_ACL

This option enables [Arm Compute Library] based primitives. ACL is an
open-source library for machine learning applications.
The `ONEDNN_AARCH64_USE_ACL` CMake option is used to enable ACL integration:

~~~sh
$ cmake -DONEDNN_AARCH64_USE_ACL=ON ..
~~~

This assumes that the environment variable `ACL_ROOT_DIR` is
set to the location of Arm Compute Library (`ACL_ROOT_DIR=</path/to/ComputeLibrary>`),
which must be downloaded and built independently of oneDNN.

@warning
For a debug build of oneDNN it is advisable to specify a Compute Library build
which has also been built with debug enabled.

@warning
oneDNN only supports builds with Compute Library v53.1 or later.

[Arm Compute Library]: https://github.com/ARM-software/ComputeLibrary

@anchor opt_gpu
## GPU Options

### Common GPU options

| CMake Option         | Default   | Supported values     | Description                                                                     |
|:---------------------|:----------|:---------------------|:--------------------------------------------------------------------------------|
| [ONEDNN_GPU_RUNTIME] | **NONE**  | SYCL, OCL, ZE        | Defines the offload runtime for GPU engines                                     |
| ONEDNN_GPU_VENDOR    | **INTEL** | NVIDIA, AMD, GENERIC | Specifies target GPU vendor for GPU engines when ONEDNN_GPU_RUNTIME is not NONE |

[ONEDNN_GPU_RUNTIME]: @ref opt_gpu_runtime

@anchor opt_gpu_runtime
#### ONEDNN_GPU_RUNTIME

To enable GPU support you need to specify the GPU runtime by setting the
`ONEDNN_GPU_RUNTIME` CMake option. Choose the runtime that matches how your
application manages GPU devices, queues, and memory:
- `SYCL` links to SYCL runtime and enables [SYCL interoperability API].
  This runtime requires [oneAPI DPC++ Compiler]. SYCL is recommended runtime
  for new applications.
- `OCL` links to OpenCL runtime and enables [OpenCL interoperability API].
  Supported only for Intel GPUs.
- `ZE` links to Level Zero runtime and enables [Level Zero interoperability API].
  Supported only for Intel GPUs.

[SYCL interoperability API]: @ref dev_guide_dpcpp_interoperability
[OpenCL interoperability API]: @ref dev_guide_opencl_interoperability
[Level Zero interoperability API]: @ref dev_guide_level_zero_interoperability
[oneAPI DPC++ Compiler]: https://github.com/intel/llvm#oneapi-dpc-compiler

### Intel GPU options

| CMake Option                      | Default                    | Supported values | Description                                                                 |
|:----------------------------------|:---------------------------|:-----------------|:----------------------------------------------------------------------------|
| [ONEDNN_ENABLE_PRIMITIVE_GPU_ISA] | **ALL**                    | \<list\>         | Specifies the list Intel GPU microarchitectures supported by JIT generators |
| ONEDNN_OCL_INCLUDE_DIR            | **third_party/opencl**     | \<path\>         | Location of OpenCL headers                                                  |
| ONEDNN_ZE_INCLUDE_DIR             | **third_party/level_zero** | \<path\>         | Location of Level Zero headers and Intel GPU driver extensions              |

[ONEDNN_ENABLE_PRIMITIVE_GPU_ISA]: @ref opt_enable_primitive_gpu_isa

@anchor opt_enable_primitive_gpu_isa
#### ONEDNN_ENABLE_PRIMITIVE_GPU_ISA

This option controls support of Intel GPU microarchitectures in oneDNN JIT
generator. By default all microarchitectures supported by the library are
enabled. The list of supported microarchitectures can be restricted to any
subset of the following list: `XEHPG`, `XEHPC`, `XE2`, `XE3`,
and `XE3P`.

To enable support for JIT optimizations on Xe2 archtiecture and newer GPUs set
the value as follows:
~~~sh
-DONEDNN_ENABLE_PRIMITIVE_GPU_ISA=XE2;XE3;XE3P
~~~

OpenCL C implementations are not affected by this option.
