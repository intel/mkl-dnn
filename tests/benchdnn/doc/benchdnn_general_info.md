# General Notes and Details

**benchdnn** follows the concept of
[state machine](https://en.wikipedia.org/wiki/Finite-state_machine).
Command line options either common or driver-specific together with default
option values form a set of states. Execution iterates over all states when a
problem descriptor (or dimensions) is parsed. This means that options that are
specified _after_ descriptor won't be applied for the previous descriptor, only
for the next one if/when met. By default, each driver has the set of option
values that results in a single state for a given problem descriptor.

## Return status

Returns `1` if any submitted test returned a failure status (for example
`FAILED`, `UNIMPLEMENTED`, `UNTESTED`, or `INVALID_ARGUMENTS`), `0` otherwise.

## Running Tests

oneDNN comes with its own testing infrastructure enabled through CMake.
If the project is built with `DNNL_BUILD_TESTS` set to `TRUE`, then Cmake will
add test targets automatically.
Targets can be triggered by the following command:
``` sh
    ctest [-R ".*benchdnn.*"]
```
, where the content in brackets is optional and limits testing to benchdnn only.
The pattern can be extended further for finer granularity.

The available tests can be found in the oneDNN directory:
tests/benchdnn/inputs/

## Glossary

| Abbreviation | Description
| :---         | :---
| src          | Source/input image
| wei          | Weights (or filter)
| bia          | Bias
| dst          | Destination/output image
| acc          | Accumulation (typically in terms of data type)

## Modes

**benchdnn** supports several execution flows, or "modes". The driver takes the
following steps to execute any flow:
1. Parse user input.
2. Iterate over multiple selected options for each problem descriptor and create
   a driver problem object (benchdnn internal abstraction) for each unique set
   of all options available for the driver. Each problem object continues
   performing the next steps.
3. Call backend API to create backend objects to execute.
4. Create memory objects for backend and reference paths and fill them with
   reasonable data.
5. Execute backend path.
6. Correctness validation:
   * Check that padded area, if present, is properly zeroed for each memory.
   * For GPU: check that the backend didn't write out-of-boundary.
   * Execute reference path.
   * Setup compare object.
   * Compare outputs of backend and reference and save the status.
7. Bitwise validation:
   * Save results from the first run.
   * Refresh data in tensors if needed.
   * Execute backend path for a second time.
   * Compare outputs to be bitwise exact.
8. Performance validation:
   * Execute backend path in a loop until one of selected criterion to stop is
     triggered. Refer to [performance options](knobs_common.md) for details.
9. Report a test case status and repro line.
   * If performance validation was requested, print a performance report output
     based on selected options and collected statistics.
10. Repeat steps 2-9 until all setups are validated.
11. Report the summary and return the status.

Each mode is standalone since most of them include one another, unless specified
otherwise. The following modes (`--mode`) are supported:
* Listing (`L`). This flow executes steps 1-2. It allows to verify input
  files by parsing syntax and check, if all problem repro lines are valid.
* Initialization (`I`). This flow executes steps 1-3. It allows to verify
  successful backend objects creation (especially large problems that take
  excessive memory and/or time to execute).
* Execution (`R`). This flow executes steps 1-5. It saves time from running
  correctness when it is not needed.
* Correctness (`C`). This is the default driver flow. It executes all steps,
  skipping bitwise and performance validation steps.
* Bitwise (`B`). This flow executes all steps, skipping correctness and
  performance steps. The mode purpose is to check that two different library
  runs of the same problem return the exactly the same output. For some
  applications it may be a hard requirement. To increase the chances of
  detecting a potential issue, completely random float values are used for the
  accumulation order to impact the output results. It is important to note that
  if the case is passing, it doesn't necessarily mean the problem is free of
  issues; it may happen that it was not triggered.
* Performance (`P`): This flow executes all steps, skipping steps 6 and 7.
  When running on a GPU engine, the `M` modifier (see below) is automatically
  enabled to align data filling with the Fast Performance mode.
* Fast Performance (`F`): This flow executes Performance mode with `P` and `M`
  modifiers (see below) enabled and updated maximum measuring time per case.
* Simulation Performance (`S`): This flow is intended for scenarios where
  execution time is prohibitively expensive, such as running under a hardware
  simulator. It behaves like Performance mode with the following specifics:
  - Input tensor fill-in is skipped. Since data values are not important (no
    correctness check is performed), the `M` modifier (see below) is enabled,
    which also skips mapping/unmapping memory objects.
  - The warm-up run is skipped.
  - Execution is limited to a single run per problem (equivalent to
    `--fix-times-per-prb=1`).
  - Cold cache is not supported.

## Mode modifiers

Modes may have extensions to their default behavior. These extensions are
enabled by default for certain modes and further modifiers may be enabled by
(`--mode-modifier`). See details next to each modifier to know their limits. The
following modifiers are supported:
* Parallel test object creation (`P`). This is an extension of step 4, when
  several backend objects, up to the number of threads identified on the system,
  are created in parallel and then executed in order. This allows to overlap
  creation overhead. Applicable for both CPU and GPU and for all modes but
  listing.
  Note: this modifier changes the default scratchpad mode from `library` to
  `user` because of thread-safety issue. The library scratchpad mode can't be
  used  unless "-DDNNL_ENABLE_CONCURRENT_EXEC=ON" is enabled at the build time.
  Otherwise scratchpad pointers are invalidated due to threads used for creation
  are no longer alive at the point when execution time comes.
* Disable reference memory (`M`). This is an extension for modes which do not
  need reference memory, like run mode. It includes skipping
  mapping/unmapping memory objects and also skipping filling functions. Every
  value of a device memory object is assigned with a special value directly.

## Problem Statuses

Each problem in **benchdnn** receives a status reflecting the outcome of the
problem execution. Following statuses are supported (in order of processing the
problem):
* `LISTED`. It means that a driver problem object was created, and the
  reproducer line might be reported. The execution was stopped before creating
  any library objects.
* `SKIPPED`. Same as `LISTED` but the execution was stopped intentionally for
  the reason given in the short description, e.g. "Case not supported" or
  "Skip-impl option hit".
  Note: The library `unsupported_data_type` status errors are always treated
  as `SKIPPED "(Data type not supported)"` as the hardware lacks support for a
  requested data type.
  Note: Nvidia backend is treated specially. See a note below.
* `INVALID_ARGUMENTS`. It means that the library API returned an error due to
  incorrect argument values. It is treated as a failure.
* `UNIMPLEMENTED`. It means that the library does not have an implementation for
  a requested problem. It is treated as a failure.
  Note: All Nvidia backend `unimplemented` status errors are always treated as
  `SKIPPED "(Case not supported)"` to simplify validation.
* `INITIALIZED`. It means that a problem was initialized, and the primitive
  creation was successful, but there was no execution call or validation.
* `EXECUTED`. It means that a problem was run, and the library execution call
  was successful, but the correctness was not validated.
* `PASSED`. It means that a problem passed the correctness validation, and the
  library output matches the driver's reference output.
* `MISTRUSTED`. It means that the quality of correctness validation is under
  question. This often happens when the ratio of the number of zero values to
  the total number of elements in the output exceeds a certain threshold. One
  possible reason is improper filling of input data for a given driver,
  specific algorithm, or a specific problem. Though the validation may not
  fulfil the purpose, as long as values are same for driver reference and the
  library outputs, it is not treated as a failure.
* `FAILED`. It means that a problem did not pass the correctness validation,
  and the library output differs from the driver's reference output.
* `UNTESTED`. It means that none of above statuses were assigned, and the
  execution was aborted at unexpected place. It is treated as a failure.

## Input Files Naming Convention

Benchdnn follows certain [guidelines](benchdnn_input_files_naming_convention.md)
regarding input files naming convention.
