# SDPA forward config tuner

Scripts that find the best tile configuration of the fused forward SDPA
kernel for given problems and devices, and build the results into the
library (`src/gpu/intel/sdpa/select.cpp`).

The kernel's performance depends heavily on its microkernel tile
configuration: the KQ and VS tile sizes and how they are split across
subgroups. The best configuration changes with head size, sequence lengths,
mask, data types, K layout and device. The hand-tuned table in
`configs.cpp` is keyed only on architecture, head size, a sequence-length
bucket and a few flags, so many shapes run with a config well off the best
one available to them. Timing every valid config of every problem on every
device would find the best, but is too slow to repeat for each new workload
or SKU.

These scripts make that search cheap and targeted:

- **Model-guided sweeps.** A cost model in the library ranks the valid
  configs of a problem, and a sweep times only the model's top candidates
  plus a small random sample instead of the whole space. Fitting the model
  to the measurements improves the ranking for the next sweep.
- **Per-problem lookup table.** Measured winners go into a lookup table
  keyed on the full problem description, so a single workload can be tuned
  without changing the selection for any other shape. Entries can apply to
  a whole architecture or to one device.
- **Verified results.** Every table entry is built and checked for
  correctness by benchdnn, and `--compare` times the shipped selection
  against the hand-tuned one before a change lands.

The scripts need a library built with `ONEDNN_DEV_MODE=ON`: the cost model
API, the config overrides and the selection trace they rely on are
compiled only in dev mode. The table they produce is used by every build.

## How selection works

When a forward SDPA primitive is created, the library describes the
problem and the device, reduces that description to a lookup key and picks
the first config that applies, in this order:

1. A config forced through an environment variable (dev mode only;
   source `env`).
2. A table entry for this key on this exact device (`table_dev`).
3. A table entry for this key on the whole architecture (`table`).
4. The hand-tuned pick from `configs.cpp` (`legacy`).

The table is `fwd_table_data[]` in `select.cpp`, one `key config` line per
entry, written by the sweeps in the [workflow](#workflow). The key is exact
on everything that changes which kernel is built (data types, mask,
quantization, K layout, alignment, discrete or integrated device) and
bucketed on sizes, so one entry covers a range of sequence lengths and
batch sizes and leaves every other shape untouched; see
[Table keys](#table-keys) for the full format. An entry that the tiler's
constraints (`fwd_describe()`) reject is skipped in favour of the
hand-tuned pick.

The [cost model](#cost-model) is not part of this path. A config the
microkernel generator rejects fails primitive creation, and only table
entries have been built and verified, so the model is used only by the
sweeps to choose which candidates to time.

A `ONEDNN_DEV_MODE` build exposes the hooks the scripts rely on: a verbose
line that reports every selection, the environment variables that force a
config, and a C API to the cost model. They are described in
[Dev-mode interface](#dev-mode-interface).

## Files

In the order the [workflow](#workflow) uses them; `sdpa_model.py` is the
library binding the others import.

- `tune_fwd_config.py`: **runs a sweep.** For each problem it ranks the
  valid configs with the model, times the top candidates plus a random
  sample (or every config with `--all`) through benchdnn, checks the
  fastest for correctness, and appends one CSV row per timed config to
  `--out`. `--table-out` writes the verified winners as table lines. With
  `--compare` it instead times the library's current selection against
  the hand-tuned config on each problem and prints the ratio: the
  acceptance check after a rebuild.
- `report_fails.py`: **checks the tiler against a sweep.** Lists the
  configs that failed to build but the tiler still accepts (a rule is
  missing) and the configs that ran but the tiler rejects (a rule is too
  broad). Run it on every sweep before baking; both lists should be empty.
- `analyze_landscape.py`: **inspects a sweep.** Per problem, the fastest
  configs, the best and median time for each value of each config
  parameter, and how the model's top picks measured. Best on an `--all`
  sweep.
- `fit_fwd_model.py`: **scores and fits the model.** Reports how well the
  current coefficients rank the configs in sweep CSVs (regret, rank
  correlation). With `--fit` it searches new coefficients, cross-validates
  them with `--folds`, and prints them as a C++ block for `bake_fit.py`.
  `--check-recorded` warns when a CSV was recorded with a different model
  than the loaded library.
- `resolve_collisions.py`: **settles table keys shared by several
  problems.** Sizes are bucketed, so problems from different sweeps can
  land on one key, and `bake_fit.py` would keep whichever winner it merged
  last. For each shared key it times every candidate on all of the key's
  problems, verifies the best, and writes table lines for `bake_fit.py`.
  Run it before baking when combining sweeps of one architecture.
- `bake_fit.py`: **applies the results to `select.cpp`.** Replaces the
  architecture's coefficients in `seed_coefs()` with a fit (`--fit`) and
  merges table lines (`--table`, repeatable; later files win on equal
  keys) into `fwd_table_data[]`. Run `clang-format` on `select.cpp`
  afterwards.
- `problems_to_batch.py`: **writes a benchdnn batch file** for the
  `--sdpa` lines of a problems file, to stdout, so the same problems can
  be run on two different builds (before and after a change, or two
  branches). `graph=` lines are skipped.
- `compare_perf.py`: **compares two benchdnn perf logs** problem by
  problem and prints the geomean time ratio, the win/loss counts and the
  largest differences. Pairs with `problems_to_batch.py`; within a single
  build, `tune_fwd_config.py --compare` is simpler.
- `sdpa_model.py`: **the Python binding to the library's cost model,**
  imported by the scripts above. It loads `build/src/dnnl.dll` or
  `libdnnl.so` from the repo, or the library given by `--lib` or
  `$DNNL_LIB`, which must be a `ONEDNN_DEV_MODE` build. Run directly, it
  ranks the configs of a sample problem: a quick check that the library
  loads.
- `problems/`: the shipped problem sets, see
  [Problem files](#problem-files).

## Problem files

A sweep runs over a list of problems. `tune_fwd_config.py` reads them from
files (`--problems`, repeatable) or the command line (`--problem`), or
draws a random sample over head sizes, sequence lengths, batch, heads and
masks (`--design N`). A problem file has one problem per line; `#` starts
a comment.

### Line format

A line describes the problem for one of two benchdnn drivers, and a file
may mix both.

**`--sdpa` driver**, as `key=value` tokens:

| key | meaning | default |
|---|---|---|
| `b`, `h` | batch and query heads | 1, 1 |
| `hk` | K/V heads, for grouped-query attention | `h` |
| `q`, `k` | queries and keys | required |
| `d`, `dv` | head size of Q/K and of V | required, `d` |
| `dt` | data type | `f16` |
| `mask`, `mdt` | benchdnn `--mask` (`causal_top_left`, `buffer_2d`, ...) and the mask data type | `none` |
| `args` | further driver flags, verbatim, such as `--ktag=abdc` for the framework K layout | |

    b=2 h=20 q=64 k=77 d=64 dt=f16
    b=1 h=32 hk=8 q=1 k=2049 d=128 dt=f32 mask=causal_bottom_right args=--ktag=abdc

**`--graph` driver**, as `graph="..."` holding the arguments of a benchdnn
`--graph` run verbatim. Use it for what the `--sdpa` driver cannot
express: int4/int8 K/V with grouped scales and zero points, or f16
accumulation through `accumulation_mode`. `--case` takes the same paths as
on the benchdnn command line, so the in-tree cases under
`tests/benchdnn/inputs/graph` work as they are. The script reads the
problem description from the library's selection line, so nothing else
about the case needs to be spelled out.

    graph="--reset --dt=0:s8+2:s8+... --in-shapes=... --case=complex_fusion/mha/sdpa-compressed-kv-int4-gs32.json"

`problems_to_batch.py` converts only the `--sdpa` lines.

### Shipped problem sets

The sets in `problems/` cover the shapes the shipped table was built from;
sweep them on a new architecture before adding anything specific.

| file | contents |
|---|---|
| `kpi.txt` | SDPA KPI shapes (SD UNet, SDXL head sizes, LLM prefill and decode with explicit masks, Whisper), one line per table key, plus int8 K/V graph cases; replace `KPI_DIR` with the directory of their json files |
| `ci.txt` | coverage shapes from the benchdnn harness: both K layouts, grouped-query attention, head sizes 16 to 512 |
| `compute.txt` | compute-bound shapes, needed to fit the systolic terms of the model |
| `f32.txt` | the KPI and coverage regions in f32, which always take the FMA kernel |
| `quant.txt` | int4/int8 K/V with grouped scales and zero points, through the in-tree graph cases |
| `f16acc.txt` | f16 accumulation through the graph cases, reachable only on parts without systolic arrays (xe_lpg) |

Fits that include `f32.txt`, `quant.txt` or `f16acc.txt` need
`--unfix fma_cycles_per_flop --unfix unroll_overhead
--unfix dequant_cycles_per_elem`, on the fit and on `bake_fit.py` (see
[Fitting](#fitting)).

### Adding your own problems

A sweep over the shapes of a specific workload, such as one model or one
customer's deployment, produces table entries for exactly those shapes and
leaves every other shape on its current config. This is the main way to
improve a workload that the shipped sets do not cover; the steps are in
[Tuning a specific workload](#tuning-a-specific-workload).

- Match the workload's data types, mask, grouped-query heads and K layout
  exactly: the table key is exact on them (see [Table keys](#table-keys)).
  Frameworks usually pass K with the head dimension contiguous, which is
  `args=--ktag=abdc`.
- One line per table key is enough. Sizes are bucketed (keys and
  batch x heads round up to a power of two), so shapes that differ only
  within a bucket share one entry; the `key` field of the selection line
  shows where a shape lands.
- Keep odd sequence lengths as they are: they change the K alignment,
  which is part of the key and often of the best config.
- To refit the model as well, combine the workload's sweep with the
  shipped sets; a fit on a single workload says little about other
  shapes, and cross-validation will show it.

## Workflow

There are two flows. Tuning a specific workload only adds table entries
and is the common case. Bringing up an architecture, or refreshing one,
also refits the cost model so later sweeps pick better candidates.

### Setup

Build oneDNN with `-DONEDNN_DEV_MODE=ON` and a GPU runtime
(`-DONEDNN_GPU_RUNTIME=OCL` or `SYCL`), including benchdnn and the gtests.
The scripts need Python 3.8 or newer; SciPy is optional and only speeds up
fitting. The examples below run from the repository root and use
`build/tests/benchdnn/benchdnn` (`benchdnn.exe` on Windows).

Keep the GPU state constant across the sweeps that feed one table or fit:
no display or other GPU load, fixed clocks where the driver allows it. The
same config can time very differently from one session to the next, and a
sweep only detects drift within a problem: it re-times a reference config
every `--ref-every` runs and warns when that drifts, so stop if the
warning fires repeatedly (the part is throttling). Sweeps use benchdnn
perf mode P with `--max-ms` per config (default 300 ms); the 10 ms fast
mode (`--perf-mode F`) is quicker but leaves the GPU clock unsettled and
is noticeably less accurate. Because of the drift, `--compare` after the
rebuild, not the sweep itself, is the acceptance test of a table entry.

### Tuning a specific workload

1. Write a problems file with the workload's shapes (see
   [Adding your own problems](#adding-your-own-problems)).
2. Sweep it and keep the verified winners as table lines:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn \
           --problems my_workload.txt \
           --out sweep_my_workload.csv --table-out table_my_workload.txt

3. Check that the tiler agrees with what built, and add a rule for any
   config it still accepts (see [Adding a rule](#adding-a-rule)):

       python scripts/sdpa_tuner/report_fails.py sweep_my_workload.csv

4. Merge the table lines into `select.cpp`. A line replaces an existing
   entry with the same key, which may have been tuned for other shapes;
   when the sweep that produced that entry is at hand, settle the key over
   both with `resolve_collisions.py` and merge its output last.

       python scripts/sdpa_tuner/bake_fit.py --table table_my_workload.txt
       clang-format -i src/gpu/intel/sdpa/select.cpp

5. Rebuild and compare the new selection against the hand-tuned configs;
   every covered shape should be at or below 1.0:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn \
           --problems my_workload.txt --compare

Only the table lines go into the tree; keep workload-specific problem
files and CSVs out of it.

### Bringing up an architecture

1. Sweep the [shipped problem sets](#shipped-problem-sets), optionally
   with a random design on top. `--out` and `--table-out` append, so
   several runs can feed one fit:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn \
           --problems scripts/sdpa_tuner/problems/kpi.txt \
           --problems scripts/sdpa_tuner/problems/ci.txt \
           --problems scripts/sdpa_tuner/problems/compute.txt \
           --out sweep_a750.csv --table-out table_a750.txt
       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn \
           --design 60 --seed 1 --dtype f16 \
           --out sweep_a750.csv --table-out table_a750.txt

2. Check the tiler, as in the workload flow:

       python scripts/sdpa_tuner/report_fails.py sweep_a750.csv

3. Score the current model, then fit and cross-validate new coefficients
   (see [Fitting](#fitting)):

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --check-recorded -v
       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1 > fit_a750.txt

4. Settle table keys that several swept problems share, then bake the fit
   and the table lines:

       python scripts/sdpa_tuner/resolve_collisions.py \
           --benchdnn build/tests/benchdnn/benchdnn \
           --tag a750 --out fixes_a750.txt sweep_a750.csv
       python scripts/sdpa_tuner/bake_fit.py --arch xe_hpg --device A750 \
           --fit fit_a750.txt --table table_a750.txt --table fixes_a750.txt
       clang-format -i src/gpu/intel/sdpa/select.cpp

5. Rebuild, run `test_internals --gtest_filter=sdpa_select.*`, confirm
   that the `seed:` line of `fit_fwd_model.py` now matches the fit's
   `fitted/train` line, and run `--compare` on the swept problem sets as
   in the workload flow.
6. For a second SKU of the architecture (A770 after A750, B570 after
   B580), run `--compare` on the same problems. Shapes that regress get a
   device-tagged line from a sweep with `--table-tag-device`; the
   arch-wide lines stay shared. Occupancy-bound shapes with few
   work-groups are the ones that differ between SKUs. One fit can cover
   several devices of an architecture: given CSVs from more than one,
   `fit_fwd_model.py` names each problem per device (`@eu<N>`). When they
   mix discrete and integrated parts (BMG and LNL), pass
   `--unfix mem_bw_integrated` to the fit and to `bake_fit.py`.

## Cost model

The cost model is an analytical performance model with fitted
coefficients. Given a problem, a device and a config, it walks through
what the kernel does: how much work each subgroup does per tile of keys,
how many work-groups the device runs at once, and how many bytes move
through memory. From that it derives two times: a compute time, the
work-groups' execution scheduled over the device, and a memory time, the
bytes moved over the bandwidth the config can actually achieve. The
estimate blends the two with a smooth maximum,

$$t = t_{launch} + \left(t_{compute}^{\,p} + t_{memory}^{\,p}\right)^{1/p},$$

which follows the larger term but still counts the smaller one near the
crossover, so configs that differ only in the non-dominant term are still
ordered. The model's job is to rank the configs of one problem, not to
predict absolute times, so it is fitted and judged on regret (see
[Fitting metrics](#fitting-metrics)).

The model lives in `fwd_estimate_cost()` in `select.cpp`. Its coefficients
are per architecture, in cycles or bytes per cycle, in `seed_coefs()`:
first-principles values until an architecture is swept, fitted values
after. Before a config is costed, `fwd_describe()` rejects configs that
cannot run (see [Tile constraints](#tile-constraints)), and
`fwd_enumerate()` returns every remaining config ranked by cost.

### Components

The coefficient behind each term is named in parentheses.

**Per key tile, per subgroup.** One iteration of the kernel's main loop
processes `kv` keys against the work-group's `q` queries:

- Matrix products. On systolic parts, KQ and VS are chains of k-blocks
  (16 elements, 8 for f32); each block issues
  `(unroll_m/8) x (unroll_n/simd)` dpas and takes at least the pipe latency
  (`dpas_issue_cycles`, `dpas_latency_cycles`), so small tiles are latency
  bound. A throughput floor (`mma_cycles_per_flop`) keeps compute-bound
  shapes from being underestimated. FMA kernels use flops over a rate
  instead, with a penalty for small unrolls (`fma_cycles_per_flop`,
  `unroll_overhead`).
- Softmax over the `kv x q` score tile (`softmax_cycles_per_elem`) and its
  round trip through SLM (`slm_cycles_per_byte`).
- K/V loads: one message per k-block of `unroll_m` rows, bound by latency
  or bandwidth (`msg_latency_cycles`, `msg_bytes_per_cycle`), slowed for K
  or V rows below 16-byte alignment (`align4_penalty`,
  `unaligned_penalty`). Mask, scale and zero-point bytes are charged
  separately (`load_cycles_per_byte`), as is dequantization of int4/int8
  K/V (`dequant_cycles_per_elem`).
- Two work-group barriers, whose cost grows with the number of subgroups
  (`barrier_cycles`, `barrier_sg_cycles`).

**Per work-group.** A fixed prologue and epilogue (`wg_fixed_cycles`),
loading Q and storing the output (`io_cycles_per_byte`), and the mean
number of key iterations; causal masks end each query tile early.

**Over the device (compute time).** Work-groups run in waves; a wave fills
every subslice with as many work-groups as its threads and SLM allow.
When resident subgroups per EU exceed what overlaps for free
(`overlap_threads`), the wave slows down. A partial last wave is charged
according to how full it is (`partial_wave_c0`, `partial_wave_c1`). Under a
causal mask the longest work-groups finish last on a partly idle device
(`causal_tail_frac`).

**Memory time.** K and V are read once per query tile. While the K/V of
the heads resident at once fit in a share of L3 (`l3_fraction`), only the
first read goes to memory (`mem_bw_discrete` or `mem_bw_integrated`) and
the rest hit L3 (`l3_bw`). The achieved bandwidth is further capped by
the bytes in flight over the memory latency (`inflight_depth`,
`mem_latency_cycles`), which penalizes configs with few threads or small
loads.

**Total.** The smooth maximum above (`mix_power` is $p$), plus a launch
cost (`launch_cycles`), converted to microseconds (`clock_ghz`).

### Fitting metrics

Regret is `time(config the model ranks first) / time(best measured config)`
per problem. Table entries are 1.0 by construction. Targets before the
model could become a selection step of its own: geomean regret at or below
1.05, worst case at or below 1.15, and a fallback to the hand-tuned config
when the generator rejects the pick.

Fit with `--folds 4 --restarts 1` and quote the cross-validated line: it is
the out-of-sample regret over every problem, each scored by a fit that did
not see it. The in-sample `fitted/train` line is always better and is only
the number to bake.

### Fitting

`fit_fwd_model.py` evaluates the library's model on every timed config of
a sweep and searches the coefficients of one architecture to minimize a
mix of three terms: a pairwise ranking loss (configs ordered against their
measured times, weighted by the time gap), the mean log regret, and a small
log-error term that keeps estimates on scale. The search runs over
log-coefficients, so every coefficient stays positive, with SciPy's
Nelder-Mead when available. `bake_fit.py` writes the result into
`seed_coefs()`.

Some coefficients are held at their seeds unless released with `--unfix`:
`clock_ghz` only rescales the others, `grf_k_load` drives a step function
a continuous search cannot fit, and the FMA, dequantization, alignment and
integrated-bandwidth terms are only exercised by sweeps that cover those
cases (`problems/f32.txt`, `quant.txt` and `f16acc.txt`, misaligned
shapes, integrated parts). Pass the same `--unfix` flags to `bake_fit.py`
so it writes the released coefficients. Include compute-bound shapes
(`problems/compute.txt`) in every fit: without them it cannot separate the
dpas chain terms from the throughput floor.

### Limits

The model cannot be exact, because the KQ and VS products are gemmstone
microkernels generated per config when the primitive is created. Inside
each one, gemmstone picks its own strategy (k-block size, load messages,
register layout, prefetch, GRF mode), and the model sees none of it. It
assumes one load message per 16-element k-block, treats each product as a
plain dpas chain, and estimates the GRF mode by mirroring gemmstone's
heuristic with a nominal k load. The fitted coefficients absorb what these
assumptions miss, so they are effective rather than physical values and
can sit far from their first-principles seeds.

For the same reason, whether gemmstone can generate a config at all is not
something the model can derive; that is left to the rules described in
[Tile constraints](#tile-constraints). This is why the model only chooses
which configs a sweep times, and only measured, verified table entries
ship.

## Tile constraints

Not every config can run every problem. `fwd_describe()` in `select.cpp`
(the tiler) decides which configs are valid before the model costs them:
`fwd_enumerate()` only returns configs it accepts, so sweeps never spend
time on them, and a table entry it rejects falls back to the hand-tuned
pick. Each rejection carries a short reason string, which the selection
prints at `ONEDNN_VERBOSE=debuginfo=4` when the hand-tuned pick itself
fails the rules.

The rules fall into four groups, from structural to empirical. The rules
themselves, each with a comment on the sweep that found it, are in
`fwd_describe()` in [select.cpp](../../src/gpu/intel/sdpa/select.cpp);
the test cases in
[test_sdpa_select.cpp](../../tests/gtests/internals/test_sdpa_select.cpp)
show, for each rule, a config it rejects next to one it still accepts.

- **Kernel structure.** The SDPA kernel runs the KQ and VS products in one
  work-group over the same query tile and subgroup grid, so the two
  products' tiles and subgroup counts must agree. The tiler also mirrors
  the problem-level dispatch checks of `micro.cpp`.
- **Generator ranges.** The tile sizes and subgroup counts gemmstone
  generates microkernels for, per architecture.
- **Resources.** Registers, SLM, work-group size and subslice occupancy,
  computed from the device description (`hw:`); these need no sweeps.
- **Found by sweeps.** Configs that pass everything above but that
  gemmstone or the driver still rejects, or that build and produce wrong
  results.

The last group is the one that grows. Its failures usually show up as one
of these:

| symptom | usual cause |
|---|---|
| "Cooperative operation cannot be split evenly" | a cooperative load cannot divide a tile evenly over the subgroups along N |
| "Insufficient registers", "No matching kernel" | no load strategy fits the registers, often for 2-byte aligned K rows (odd key counts) or f32 |
| microkernel preflight failure | a work-group shape the systolic generator does not support |
| `CL_OUT_OF_RESOURCES` at kernel build | the microkernels generate, but the whole kernel does not fit |
| builds, fails correctness | a generator bug for that shape; only a correctness run finds it |

Rules are scoped to the architectures and problem features where a sweep
showed the failure, because the same tile often works elsewhere: a
1024-element KQ tile over 2-byte aligned K fails on PVC but builds and
verifies on BMG.

### Adding a rule

1. Run `report_fails.py` on the sweep CSVs. It lists the configs that
   failed to build but the tiler still accepts, grouped with the problem
   features that usually explain a generator limit (subgroups, tile sizes,
   head size, alignment, K layout).
2. Reproduce one of them with the config forced through `SDPA_CONFIG` and
   benchdnn `--mode=I` at `ONEDNN_VERBOSE=all` to read the generator's
   reason.
3. Add a condition to `fwd_describe()` scoped to the architectures and
   features that failed, with a one-line comment naming the sweep and the
   generator's message, and a `fail("...")` reason.
4. Add a case to `rejects_broken_configs` in
   `tests/gtests/internals/test_sdpa_select.cpp`: an `EXPECT_FALSE` on a
   failing config from the sweep, and an `EXPECT_TRUE` on the closest
   config that built (a neighbouring tile, or the same config on another
   architecture or with aligned K), so the test pins down the scope of
   the rule as well as its trigger. Run it with
   `test_internals --gtest_filter=sdpa_select.*`.
5. Rebuild and rerun `report_fails.py` until both of its lists are empty.
   The second list, configs that ran but the tiler rejects, catches a rule
   that is too broad; such a rule hides tiles from every later sweep.

## Table keys

A table line is a key and a config separated by a space:

    xe2:d128x128:k1024:q1024:bh128:m1:t16/16/16/16:qz00:g0/0:sys16:acc32/32:al64/64/64/64:i0:tr0:tk1 32,32,32,16,8,1,4,2

The config is eight integers: `unroll_m_kq,unroll_n_kq,unroll_m_vs,
unroll_n_vs,wg_m_kq,wg_n_kq,wg_m_vs,wg_n_vs`, the tile sizes and subgroup
counts of the KQ and VS products. The key is built by `fwd_make_key()` and
`fwd_key_t::str()` in `select.cpp` from the fields below, in order:

| field | meaning |
|---|---|
| `xe2` | GPU architecture |
| `d<qk>x<v>` | head sizes of K and V |
| `k<N>` | keys, rounded up to a power of two, at least 32 |
| `q<N>` | queries: 1 for decode, 16 for 2 to 16, else rounded up to a power of two; grouped-query decode counts the query heads per KV head |
| `bh<N>` | batch x heads, rounded up to a power of two |
| `m<N>` | mask: 0 none, 1 causal top-left, 2 causal bottom-right, 3 buffer |
| `t<q>/<k>/<v>/<dst>` | element sizes in bits |
| `qz<k><v>` | 1 when K or V is quantized |
| `g<k>/<v>` | quantization group sizes, 0 for none or a common scale |
| `sys16`, `sys32`, `fma16`, `fma32` | systolic or FMA microkernels; f32 compute or not |
| `acc<kq>/<vs>` | accumulation bits of the KQ and VS products |
| `al<q>/<k>/<v>/<dst>` | alignment in bytes: 64, 16, 4, or the exact value below 4 |
| `i<0/1>` | integrated device |
| `tr<0/1>` | forward training |
| `tk<0/1>` | K layout: K rows contiguous (benchdnn `--ktag=abcd`, the driver default) or the head dimension contiguous (`--ktag=abdc`, what frameworks pass) |
| `eu<N>` | optional: the line applies only to devices with N EUs |

A sweep only covers the exact variants it runs, so sweep the data types,
masks and K layouts your workload uses (`problems/ci.txt` covers both
layouts), and sweep discrete and integrated parts separately.
`tune_fwd_config.py --table-tag-device` writes `:eu<N>` lines for SKUs that
need their own entry.

## Dev-mode interface

All of the following needs a `ONEDNN_DEV_MODE=ON` build.

**Forcing a config.** `SDPA_CONFIG` sets the config of a non-quantized
problem and `QUANTIZED_SDPA_CONFIG` that of a problem with K/V scales or
zero points, both as the eight comma-separated integers of a table line.
They override every other source, which is how the sweeps time candidates.

**Selection line.** With `ONEDNN_VERBOSE=debuginfo=4` every forward
selection prints one line, which `tune_fwd_config.py` parses:

    fwd_select,key:...,src:...,cfg:...,legacy:...,kv:...,q:...,sg:...,slm:...,grf:...,wgss:...,cost_us:...,hw:...,prb:...

| field | meaning |
|---|---|
| `key` | the lookup key, as in [Table keys](#table-keys) |
| `src` | where the config came from: `env`, `table_dev`, `table` or `legacy` |
| `cfg` | the config that runs |
| `legacy` | the hand-tuned pick, printed even when it is not used |
| `kv`, `q` | key and query rows one work-group processes |
| `sg` | subgroups per work-group |
| `slm` | shared local memory per work-group, in bytes |
| `grf` | expected GRF mode (128 or 256) |
| `wgss` | work-groups resident per subslice |
| `cost_us` | the cost model's estimate |
| `hw`, `prb` | device and problem descriptions, `name=value;...` |

**Cost model API.** `select_api.cpp` exports the model so that the Python
scripts call the library's implementation rather than a copy of it.
`sdpa_model.py` wraps these functions; they take the `hw` and `prb` strings
of the selection line and, optionally, a coefficient vector that replaces
the architecture's built-in values:

| function | purpose |
|---|---|
| `dnnl_impl_sdpa_fwd_coef_count`, `dnnl_impl_sdpa_fwd_coef_name` | coefficient names, in order |
| `dnnl_impl_sdpa_fwd_coef_seeds` | built-in coefficients of an architecture |
| `dnnl_impl_sdpa_fwd_estimate` | cost and derived quantities of one config, or invalid |
| `dnnl_impl_sdpa_fwd_enumerate` | every valid config of a problem, best estimate first |

## Sample results

A snapshot of where the sweeps stand; the numbers move as problem sets
grow and architectures are refitted. Regret is the geomean over the swept
problems of each architecture (see [Fitting metrics](#fitting-metrics)),
for the hand-tuned table and for the cost model out of sample. The current
numbers for each fit are in the comments of `seed_coefs()`, which
`bake_fit.py` writes with each fit.

| architecture | problems | hand-tuned | model, out of sample | table entries |
|---|---|---|---|---|
| xe_hpg (A750) | 116 | 1.259 | 1.082 | 305 |
| xe_hpc (PVC) | 129 | 1.209 | 1.122 | 324 |
| xe2 (BMG, LNL) | 882 | 1.387 | 1.090 | 633 |
| xe3 (PTL) | 87 | 1.303 | 1.086 | 287 |
| xe3p (NVL, CRI) | 622 | 1.859 | 1.055 | 631 |

xe_lp and xe_hp are not swept yet: they use first-principles seeds and
have no table entries. Where `--compare` was run, the shipped table takes
0.74 to 0.86 of the hand-tuned time on the shapes it covers, with at most
one problem slower by more than 5%.

Lessons shaping the workflow:

- The model's top 12 plus 8 random candidates contained the best measured
  config on every A750 problem, so sweeps time only that subset rather
  than every valid config.
- The throughput floor `mma_cycles_per_flop` must be fitted, and
  compute-bound shapes must be in the set to fit it: held at its
  first-principles value, the A750 fit cross-validates at 1.105 instead of
  1.082, and the first PVC fit, without compute-bound shapes, could not
  separate it from the dpas chain terms at all.
- The K layout matters: decode runs up to 3x faster with the framework
  layout (`:tk1`) than with the benchdnn default, so sweep the layout the
  workload uses.
- Timings drift between sessions (one A750 decode config measured 0.099 ms
  in one session and 0.044 ms in another), and the 10 ms fast mode was
  1.5x off; hence perf mode P and `--compare` as the acceptance test.
- The remaining A750 misses are short-key prefill (K of 77 and 197) and
  long prefill at D 80 to 128 in the framework K layout, where the model
  prefers a 128-column Q tile over the 64-column tile that wins. More
  measurements rather than more terms are the next step there.
