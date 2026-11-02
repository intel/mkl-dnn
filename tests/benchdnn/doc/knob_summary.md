# Summary information

## Introduction
Benchdnn is designed to operate with batch files, and while some of them can be
small, some of them might be really large. While benchdnn provides tools to get
information from a single test case, it might be challenging to acquire the
interested data from the whole batch file, e.g., when running the benchmark
manually, the screen buffer may not fit the whole output and the data is simply
lost if not redirected to the file.

This is where the summary options come in handy to provide the batch level
statistics.

## Usage
```
    --summary=[no-]SETTING1[+[no-]SETTING2...]
```

The `--summary` knob is a global state of benchdnn and provides the summary
statistics at the end of the run. Different options are separated with `+`
delimiter. To negate the effect of the particular summary option, use the "no-"
prefix in front of the option value.

If the same setting is specified multiple times, only the latter value is
considered.

## Failed cases summary

### Usage
```
    --summary=[no-]failures
```

This knob provides a list of failures up to ten entries starting from the
beginning of the run to help to identify problematic cases during the run
without necessity to process the whole output manually. Enabled by default.

## Implementations summary

### Usage
```
    --summary=[no-]impl
    --summary=[no-]impl-csv
```

This knob provides a list of implementation names and the number of hits used
for problems. A table view is enabled by default, and CSV-style output is
disabled by default.

## Performance hints summary

### Usage
```
    --summary=[no-]perf-hints
```

This knob inspects performance-related system settings and reports the ones
that are not set to values providing stable performance measurements. Enabled by
default. It is only effective on Linux and reports the hints at most once per
run, before any performance data is printed. The checks are engine-specific.

For the CPU engine the following settings are verified:
* Min and max CPU frequency. The effective (scaling) range is expected to
  coincide; otherwise the frequency may float during the run.
* Turbo mode. It is expected to be disabled; otherwise the frequency may
  increase during the run.
* C-states. Idle states deeper than C1 are expected to be disabled, either with
  the boot parameter or at runtime through cpuidle `disable` controls; otherwise
  it will take more time to wake cores up affecting initial results.

Frequency and C-states are verified on one online CPU per core type.

For the GPU engine the following settings are verified:
* Min and max frequency of each Intel GPU render/compute tile. The range is
  expected to coincide; otherwise the frequency may float during the run.
