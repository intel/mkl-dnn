# Performance Report

## Usage
```
    [--perf-template={def [default], csv, CUSTOM_TEMPLATE, csv:CUSTOM_TEMPLATE}]
```

where:
 - `def` -- default template. Has problem name, problem dump, ops (if applied),
          minimum and average time and GFLOPS (if applied)
 - `csv` -- comma-separated values style template. Same as default, but dumps
          problem and descriptor values with comma delimiter.
 - `CUSTOM_TEMPLATE` -- user-defined template. Should consist of special options
                      supported by specific driver. Refer to the list of
                      options supported below.
 - `csv:CUSTOM_TEMPLATE` -- user-defined CSV template, for example
                           `csv:%name%,%prb%,%-time%,%0time%`.

## Reading the default header

For primitive drivers, the default report has the following header:

```text
perf,engine,impl,name,prb,Gops,max_ctime,min_time,min_Gflops,avg_time,avg_Gflops
```

| Column | Template entry | Meaning and units |
| :-- | :-- | :-- |
| `perf` | Literal `perf` | Performance record marker. |
| `engine` | `%engine%` | Execution engine, such as `cpu` or `gpu`. |
| `impl` | `%impl%` | Implementation selected for this problem. |
| `name` | `%name%` | Optional problem name from the input descriptor. |
| `prb` | `%prb%` | Reproducer containing global settings, driver settings, and the problem descriptor. |
| `Gops` | `%Gops%` | Estimated operation count for one execution, divided by 10^9. |
| `max_ctime` | `%+ctime%` | Sum of maximum primitive-descriptor creation time and maximum primitive creation time, in milliseconds. See Create Time Notes below. |
| `min_time` | `%-time%` | Minimum execution time, in milliseconds. |
| `min_Gflops` | `%-Gflops%` | Throughput corresponding to minimum execution time: `Gops / (min_time / 1000)`. |
| `avg_time` | `%0time%` | Average execution time, in milliseconds. |
| `avg_Gflops` | `%0Gflops%` | Throughput corresponding to average execution time: `Gops / (avg_time / 1000)`. |

For example, `Gops=0.008` and `min_time=0.02` give `min_Gflops=400`.
The `Gflops` label is also used for operation rates of integer problems.

`Template entries:` shows the template used to generate the report, followed
by the column header. In the header, `%` characters are removed, and the time
modifiers `-`, `0`, and `+` become `min_`, `avg_`, and `max_`.
For example, `%0Gbw%` produces `avg_Gbw` (GB/s). Unmodified `%time%` defaults
to minimum time; unmodified creation-time options default to maximum time.
The expanded `csv` template includes driver-specific fields.


**benchdnn** supports both out-of-the-box and custom performance reports.
A custom template should be passed via the command line and consists of terminal
and nonterminal symbols. Nonterminal symbols are printed as-is. Descriptions of
terminal symbols are given below.

> **Note:** Following generalized types are used below:
>
> * 'Data md based' = {Bnorm, Eltwise, Lnorm, Lrn, Prelu, Shuffle, Softmax}
> * 'Problem desc based' = {Bnorm, Conv, IP, Lrn, Matmul, Pool, Resampling, RNN}
> * 'Ops based' = {Conv, IP, Matmul, RNN}

Data types options supported:

| Syntax | Primitives                                     | Description
| :--    | :--                                            | :--
| %cfg%  | Conv, IP, Matmul, Pool, RNN                    | Config which describes data types and filling rules
| %bia_dt% | BRGEMM, Conv, Deconv, IP, Matmul             | Bias data type (precision)
| %dt%   | Data md based, Resampling, Zeropad             | Source and Destination Data type (precision)
| %ddt%  | Binary, Concat, Reduction, Reorder, Sum        | Destination data types (precision)
| %sdt%  | Binary, Concat, Prelu, Reduction, Reorder, Sum | Source data types (precision)

Format tags (physical memory layout) options supported:

| Syntax     | Primitives                                                       | Description
| :--        | :--                                                              | :--
| %tag%      | Data md based, Pool, Resampling, Zeropad                         | Source and Destination format tag
| %dtag%     | Binary, Concat, Conv, IP, Matmul, Reduction, Reorder, Sum        | Destination format tag
| %stag%     | Binary, Concat, Conv, IP, Matmul, Prelu, Reduction, Reorder, Sum | Source format tag
| %wtag%     | Conv, IP, Matmul                                                 | Weights format tag
| %stat_tag% | Lnorm                                                            | Layer Normalization statistics (mean and variance) format tag

Other problem specific options supported:

| Syntax       | Primitives                                                            | Description
| :--          | :--                                                                   | :--
| %activation% | RNN                                                                   | RNN activation function
| %alg%        | Binary, Conv, Eltwise, Lrn, Pool, Reduction, Reorder, Resampling, RNN | Primitive algorithm
| %attr%       | All                                                                   | Primitive attributes
| %axis%       | Concat, Shuffle, Softmax                                              | Primitive axis
| %desc%       | All                                                                   | String style problem descriptor
| %dir%        | All, except Concat, RNN, Reorder, Sum                                 | Primitive prop kind
| %direction%  | RNN                                                                   | RNN direction execution
| %driver%     | All                                                                   | Name of the current driver (e.g. conv, reorder)
| %engine%     | All                                                                   | Engine kind
| %flags%      | Bnorm, Lnorm, Reorder                                                 | Primitive flags
| %group%      | Shuffle                                                               | Shuffle group
| %impl%       | All                                                                   | Library implementation name for a given problem
| %idx%        | All                                                                   | Test index
| %mb%         | Problem desc based, Eltwise, Softmax                                  | Mini-batch value from user input. Prints `0` in case of input `--mb=0`
| %name%       | Problem desc based                                                    | Problem name
| %prb%        | All                                                                   | Canonical problem (options and descriptor in REPRO style)
| %prop%       | RNN                                                                   | RNN prop kind

Performance profiling. All options are modifier extended (see below). Modifiers
change the meaning of terminal symbols. I.e., the sign '-' means minimum of
(in terms of time). Extensions should be specified after first percent,
describing the option in a specific order: first is time modifier, second is
unit modifier. I.e. `%-Gflops%`, not `%G-flops%`.

> **Caution:** Threads must be pinned in order to get consistent frequency.

Performance profiling options supported:

| Syntax     | Primitives | Description
| :--        | :--        | :--
| %@time%    | All        | Execution time in milliseconds
| %@clocks%  | All        | Execution time in clocks
| %@freq%    | All        | Effective CPU frequency computed as `clocks / time`
| %@ibytes%  | All        | Number of input memories bytes of a problem
| %@obytes%  | All        | Number of output memories bytes of a problem
| %@iobytes% | All        | Number of input and output memories bytes of a problem
| %@bw%      | All        | Bandwidth computed as `iobytes / time`
| %@ops%     | Ops based  | Number of ops required (padding is not taken into account)
| %@flops%   | Ops based  | FLOPS computed as `ops / time`
| %@cpdtime% | All        | Primitive descriptor creation time in milliseconds. See `Create Time Notes`.
| %@cptime%  | All        | Primitive creation time in milliseconds. See `Create Time Notes`.
| %@ctime%   | All        | Total creation time (primitive descriptor + primitive) in milliseconds. See `Create Time Notes`.

Modifiers supported:

| Name  | Description
| :--   | :--
| Time: |
| -     | min (time) -- default
| 0     | avg (time)
| +     | max (time)
|       |
| Unit: |      (1e0) -- default
| K     | Kilo (1e3)
| M     | Mega (1e6)
| G     | Giga (1e9)

### Create Time Notes

Benchdnn runs two create calls when primitive cache feature is enabled. A timer,
responsible for collecting create milliseconds, catches both cases. A case when
primitive cache was not hit can be obtained through the empty or `max` modifier
(the default). A case when primitive cache was hit can be obtained through the
`min` modifier. The average modifier for create times is not recommended since
this time doesn't represent any specific scenario.

## Examples

Runs a set of inner products measuring performance with 6 seconds per problem
dumping results with a standard performance template:
``` sh
    ./benchdnn --ip --mode=p --max-ms-per-prb=6000 \
               --batch=inputs/ip/test_ip_all
```
```
Template entries: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,engine,impl,name,prb,Gops,max_ctime,min_time,min_Gflops,avg_time,avg_Gflops
perf,cpu,brg_matmul:avx512_core,"""resnet:ip1""","--mode=P --max-ms-per-prb=6000 --ip ic2048oc1000n""resnet:ip1""",0.008192,0.935547,0.0234375,349.525,0.0269902,303.518
...
```

Runs a set of inner products measuring performance and dumping results in
CSV-style:
``` sh
    ./benchdnn --ip --mode=p --perf-template=csv \
               --batch=inputs/ip/test_ip_all
```
```
Template entries: perf,%engine%,%impl%,%name%,%dir%,%sdt%,%bia_dt%,%stag%,%wtag%,%dtag%,%attr%,%desc%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,engine,impl,name,dir,sdt,bia_dt,stag,wtag,dtag,attr,desc,Gops,max_ctime,min_time,min_Gflops,avg_time,avg_Gflops
perf,cpu,brg_matmul:avx512_core,"""resnet:ip1""",FWD_B,f32:f32:f32,f32,any,any,any,,"ic2048oc1000n""resnet:ip1""",0.008192,0.9375,0.0234375,349.525,0.027302,300.051
...
```

Runs a set of inner products measuring performance and dumping custom template -
reporting descriptor, minimum time, and corresponding gigaFLOPS. Note: ',' is
not a special symbol here; any other delimiter can be used:
``` sh
    ./benchdnn --ip --mode=p --perf-template=%prb%,%-time%,%-Gflops% \
               --batch=inputs/ip/test_ip_all
```
```
Template entries: %prb%,%-time%,%-Gflops%
prb,min_time,min_Gflops
--mode=P --ip ic2048oc1000n"resnet:ip1",0.0234375,349.525
...
```
