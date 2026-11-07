# Common Options

**Benchdnn** drivers support a set of options that are available for every
driver. Some of them represent global state settings which modify the execution
behavior. Options may be supported in any `mode` or be specific to correctness
or performance validation.

## Any mode settings

### --allow-enum-tags-only
`--allow-enum-tags-only=BOOL` instructs the driver to validate format tags
against the documented tags from `dnnl_format_tag_t` enumeration only when
`BOOL` is `true` (the default). When `BOOL` is `false`, any valid format tag
is accepted when creating a testing object.

### --batch
`--batch=FILE` instructs the driver to take options and problem descriptors
from a `FILE`. If several `--batch` options are specified, the driver reads
input files consecutively. Nested inclusion of the `--batch` option is
supported. The driver searches for a file by extracting a directory where `FILE`
is located and tries to open `dirname(FILE)/FILE`. If the file is not found, it
tries to find the file in a default path
`/path_to_benchdnn_binary/inputs/DRIVER/FILE`. If the file is still not
found, an error is reported. Note that `--batch` option doesn't change the
previous state.

### --buffer-prefix
`--buffer-prefix=PREFIX` instructs the driver not to generate the synthetic data
for certain buffers but fill them with data taken from the files whose paths and
names are specified in `PREFIX`; see below. This makes benchdnn capable of using
real workloads, which may come in handy to identify HW issues and driver faults,
or reveal hidden library bugs potentially escaping scrutiny due to a blind spot
in the benchdnn test coverage.

The feature has numerous requirements and caveats and must be used with caution.
It is designed to only be used with one test case at a time.

Here's the list of requirements:
* `PREFIX` must contain a single string that points to an arbitrary place in the
  file system, relative to the working directory or absolute, followed via slash
  by a common part of the file name for all files that contain their respective
  buffer data.
* All files in the file system that contain the buffer data should be named like
  this: `PREFIX.ID.bin`, where `PREFIX` is the string passed to benchdnn via the
  knob, `ID` is the execution argument ID in the library (e.g. `DNNL_ARG_SRC` is
  `1`, `DNNL_ARG_WEIGHTS` is `33`, `DNNL_ARG_BIAS` is `41`, etc.; for more info
  see [dnnl_types.h](../../../include/oneapi/dnnl/dnnl_types.h)), `bin` is the
  file extension that is the same for all of the files.
* The files are expected to contain binary data, with exactly the type and the
  layout that the library uses for the corresponding buffers. Any file the size
  of which doesn't precisely match the size of the memory buffer it is supposed
  to contain the data for is considered a reason to output an error message and
  halt the execution.
* If a primitive needs more buffers than there are files, those not filled from
  files are generated following the regular benchdnn process; e.g. ID `80`, the
  `DNNL_ARG_SCRATCHPAD` buffer, is never initialized by benchdnn unless filled.
* Due to certain limitations of the machine arithmetic, it is probable that the
  results the GPU primitives output on real data won't exactly match those from
  the CPU primitives used by benchdnn as reference; please keep that in mind if
  benchdnn still reports an accuracy error with buffers filled from files after
  the root-cause bug got fixed; use `-v99` to see the full output comparison.

#### Usage example
Consider the following files:
* `/path/to/faulty/conv_bufs.1.bin` (contains `DNNL_ARG_SRC`)
* `/path/to/faulty/conv_bufs.33.bin` (contains `DNNL_ARG_WEIGHTS`)
* `/path/to/faulty/conv_bufs.41.bin` (contains `DNNL_ARG_BIAS`)

This line will instruct benchdnn to pick them up:
```bash
./benchdnn --buffer-prefix=/path/to/faulty/conv_bufs -v2 --conv --dir=FWD_B ...
```

`-v2` (and higher verbosity levels: `-vN`, N≥2) can be used to see what happens
with the buffers, e.g. when a binary post-op was also specified in the benchdnn
run line but the user calculated its ID incorrectly, the following message will
be shown:
```
...
File '/path/to/faulty/conv_bufs.32770.bin' [DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] not found; buffer not imported.
...
```

The buffers that got picked up leave the following lines in the verbose log:
```
...
File '/path/to/faulty/conv_bufs.1.bin' [DNNL_ARG_SRC_0] successfully processed; buffer imported.
File '/path/to/faulty/conv_bufs.33.bin' [DNNL_ARG_WEIGHTS_0] successfully processed; buffer imported.
File '/path/to/faulty/conv_bufs.41.bin' [DNNL_ARG_BIAS] successfully processed; buffer imported.
...
```

### --canonical
`--canonical=BOOL` instructs the driver to print a canonical form of a
reproducer line. When `BOOL` is `false` (the default), the driver prints the
minimal reproducer line, omitting options and problem descriptor entries with
default values.

### --cpu-isa-hints
`--cpu-isa-hints=HINTS` specifies the ISA specific hints to the CPU engine.
`HINTS` values can be `none` (the default), `no_hints` or `prefer_ymm`.
`None` value respects the `DNNL_CPU_ISA_HINTS` environment variable setting,
while others override it with a chosen value. Settings other than `none` take
place immediately after the parsing and subsequent attempts to set the hints
result in a runtime error.

### --ctx-init
`--ctx-init=MAX_CONCURRENCY[:CORE_TYPE[:THREADS_PER_CORE]]` specifies the
threading context for a testing object creation.
* `MAX_CONCURRENCY` is a
  positive integer value or `auto` (default) and specifies the maximum number of
  threads in the context.
* `CORE_TYPE` is a non-negative integer value or `auto` (default) that specifies
  the type of cores used in the context for hybrid CPU systems, `0` being the
  largest cores available on the system (TBB runtime only).
* `THREADS_PER_CORE` is a positive integer value or `auto` that allows users to
  enable (value `2`) or disable (value `1`) hyper-threading (TBB runtime only).

### --ctx-exe
`--ctx-exe=MAX_CONCURRENCY[:CORE_TYPE[:THREADS_PER_CORE]]` specifies the
threading context for a testing object execution. The setting values follow ones
from the `ctx-init` option.

### --engine
`--engine=KIND[:INDEX]` specifies an engine kind `KIND` to be used for
benchmarking. `KIND` values can be `cpu` (the default) or `gpu`. An optional
non-negative integer value of `INDEX` may be specified followed by a colon
`:`, such as `--engine=gpu:1`, which means to use a second GPU device.
Enumeration follows the library enumeration identification. It may be
checked up with ONEDNN_VERBOSE output. By default, `INDEX` is `0`. If the
index is greater or equal to the number of devices of requested kind
discovered on a system, a runtime error occurs.

### --impl
`--impl=STRING_NAME[,STRING_NAME...]` option is used to search for an
implementation containing one of the names provided.
Refer to [implementation filtering](knob_impl_filter.md) for details.

### --mem-check
`--mem-check=BOOL` instructs the driver to perform a device RAM capability
check if the problem fits the device including all service memory allocations.
The check is enabled when `BOOL` is `true` (the default) and disabled otherwise.

### --memory-kind
`--memory-kind=KIND` specifies the memory kind to test with DPC++ and OpenCL
runtimes. `KIND` values can be `usm` (default), `buffer`, `usm_device`
(to use malloc_device) or `usm_shared` (to use malloc_shared).

### --mode
`--mode=MODE` specifies **benchdnn** mode to be used for benchmarking.
`MODE` values can be:
  - `C` or `c` for correctness testing (the default)
  - `P` or `p` for performance testing. On a GPU engine, the `M` modifier is
               automatically enabled to align data filling with `--mode=F`
  - `F` or `f` for fast performance testing, an alias for
               `--mode=P --mode-modifier=PM --max-ms-per-prb=10`
  - `S` or `s` for simulation performance testing: skips warm-up, skips input
    tensor fill-in, limits execution to a single run per problem, and disables
    cold cache. Useful when execution time is prohibitively expensive, e.g.
    when running under a hardware simulator.
  - `B` or `b` for bitwise (numerical determinism) testing
  - `R` or `r` for run mode, enables `--mode-modifier=M`
  - `I` or `i` for initialization mode
  - `L` or `l` for listing mode

Refer to [modes](benchdnn_general_info.md) for details.

### --mode-modifier
`--mode-modifier=MODIFIER` specifies a modifier to a selected benchmarking mode.
`MODIFIER` values can be:
  - empty for no modifiers (the default)
  - `P` or `p` for parallel backend object creation
  - `M` or `m` for disabling usage of reference memory

Refer to [mode modifiers](benchdnn_general_info.md) for details.

Note: The `P` modifier sets the default value of scratchpad mode to `user`.
For the **benchdnn** functionality to work properly, the recommendation is to
pass this option **before** the driver name so that the modifier is processed
before the execution flow starts and can propagate a new scratchpad value. The
flow is affected when the user passes descriptors directly. When using batch
files, no difference is observed since a batch file starts a new parsing cycle
underneath, and a scratchpad value is propagated.

### --stream-kind
`--stream-kind=KIND` specifies the stream kind to test with DPC++ and OpenCL
runtimes by providing flags to the stream. The queue object is managed inside
the library. `KIND` values can be `def` (default), `in_order`, or
`out_of_order`. Refer to `dnnl_stream_flags_t` for more information.

### --repeats-per-prb
`--repeats-per-prb=N` specifies the `N` number of times to run a given problem.
The default `N` is `1`. When several problems are provided, each of them will be
executed `N` times. The option is designed to help reproduce sporadic failures
when effects like race condition or garbage values in a memory may not be
triggered from a single run but from several runs.

### --reset
`--reset` instructs the driver to reset DRIVER-OPTIONS (not COMMON-OPTIONS!) to
their default values. The only exception is `--perf-template` option which will
not be reset. COMMON-OPTIONS describe a global state and, thus, are not affected
by this option.

### --skip-impl
`--skip-impl=STRING_NAME[,STRING_NAME...]` option is used to search for an
implementation that doesn't contain any of the names provided.
Refer to [implementation filtering](knob_impl_filter.md) for details.

### --start
`--start=N` specifies the test index `N` to start testing from. All tests
before the index `N` will be skipped.

### --summary
`--summary=VALUE` provides additional specific statistics output. Refer to
[summary documentation](knob_summary.md) for details.

### --verbose
`--verbose=N`, or a short form `-vN`, specifies the driver verbosity level.
Additional information is printed to the stdout depending on a level `N`. `N` is
a non-negative integer value. The default value is `0`. Refer to
[verbose](knobs_verbose.md) for details.

### --execution-mode
`--execution-mode=MODE` specifies the execution mode to be used. When `MODE`
is set to `direct` (the default), the driver will execute normally. When `MODE`
is set to `graph` it instructs the driver to execute on a graph backend.
Currently this feature is limited to the experimental SYCL Graph feature on
DPC++ runtime with Level Zero backend.

## Correctness mode settings

### --attr-same-pd-check
`--attr-same-pd-check=BOOL` instructs the driver to compare two primitive
descriptors - the one with user requested attributes and the one without any
attributes. When `BOOL` is `true`, the check returns an error if implementation
names mismatch for two descriptors. It indicates that appending an attribute
changes the implementation dispatching which is an undesired behavior. When
`BOOL` is `false` (the default), the check is disabled.

### --check-ref-impl
`--check-ref-impl=STR` instructs the driver to compare the implementation name
string against the `ref` string pattern. When the setting is enabled, the check
returns an error if the name matches the reference pattern. `STR` specifies
which runtime is affected by the setting. If `STR` is a boolean value `true/1`,
both CPU and GPU runtimes are affected. Other possible values include `cpu`,
`gpu` and `false/0`. By default, the check is disabled. It's useful to catch
unexpected fallbacks to slow reference implementations from a big batch of
problems. This option is always disabled on NVIDIA, AMD, and Generic vendors.

### --fast-ref
`--fast-ref=BOOL` instructs the driver to use an optimized implementation
from the library as a reference path for correctness comparison when `BOOL` is
`true` (the default). Refer to [additional documentation](knob_use_fast_ref.md)
for more information.

## Performance mode settings

### --cold-cache
`--cold-cache=MODE` instructs the driver to enable a cold cache measurement
mode. When `MODE` is set to `none` (the default), cold cache is disabled.
When `MODE` is set to `wei`, cold cache is enabled for weights argument
only. This mode targets forward and backward by data propagation kinds. When
`MODE` is set to `all`, cold cache is enabled for each execution argument except
a scratchpad. This targets any propagation kind but mostly bandwidth-limited
functionality to emulate first access to data or branching cases. When `MODE` is
set to `custom`, cold cache is enabled for specified arguments, but it requires
source code adjustments. Refer to [cold cache](knob_cold_cache.md) for more
information.

### --fix-times-per-prb
`--fix-times-per-prb=N` specifies the `N` number of rounds per problem to run,
where `N` is a non-negative integer value. When `N` is set to `0` (the default),
the number of rounds will be established by the time criterion instead. For `N`
greater than `0`, the number of runs will be overridden by this setting. The
option makes performance profiling easier when a certain number of cycles is
desired or when a specific number of runs is expected.

### --max-ms-per-prb
`--max-ms-per-prb=N` specifies the `N` time limit in milliseconds per problem to
run. `N` is a positive integer value in a `[1e1, 6e4]` range. When a provided
value is out of the range, it is saturated to the board values. The default is
`3e3`, or 3 seconds. The option is useful, for example, to stabilize the
performance numbers reported for small problems on CPU.

### --num-streams
`--num-streams=N` specifies the number `N` of streams used for performance
benchmarking. The option takes place for GPU only and uses a single stream by
default.

### --perf-template
`--perf-template=STR` specifies the format of a performance report. `STR`
values can be `def` (the default), `csv` or a custom set of supported flags.
Prefix a comma-delimited custom template with `csv:` to use CSV output.
Refer to [performance report](knobs_perf_report.md) for details.
