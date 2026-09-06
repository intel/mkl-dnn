# Stream-K empirical findings

## Environment
- Repo: `atkassen/oneDNN` branch `dnn-habana`
- GPU: Intel Data Center GPU Max 1100 (XeHPC)
- Build: existing `./build`, rebuilt with `make -j$(nproc) dnnl benchdnn`
- Instrumentation used: `DNNL_STREAMK_DEBUG=1`, optional `DNNL_STREAMK_ALLOW_GROUPSUM=1`

## Bottom line
I could **not** find a real benchdnn matmul case on this XeHPC GPU where **both**:
1. `kParallelVariable=1` (stream-K selected), and
2. `needsAGroupSums()==1 || needsBGroupSums()==1`

In practice, the narrowed guard appears **empirically moot / unreachable** for realistic benchdnn matmul coverage on this setup. I therefore found **no benchdnn-reproducible evidence** that the guard currently fires on real workloads, and consequently no benchdnn-reproducible GPU correctness failure or proof of safety for `DNNL_STREAMK_ALLOW_GROUPSUM=1` on a real production path.

## What I confirmed empirically

### 1. Stream-K is selected for realistic non-group-sum cases
These cases all showed `kParallelVariable=1` in the `streamk-debug` line and passed correctness.

#### Plain f16
Command:
```bash
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=f16:f16:f16 --stag=ab --wtag=ab --dtag=ab \
  64x8192:8192x64
```
Observed debug:
```text
[streamk-debug] preflight: kParallelVariable=1 needsAGroupSums=0 needsBGroupSums=0 ...
```
Result: `PASSED`

Also passed with stream-K selected for:
- `64x4096:4096x64`
- `64x16384:16384x64`
- `256x8192:8192x256`

#### Int8 with common zero-points
Commands:
```bash
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-zero-points=src:common:0+wei:common:0 \
  64x8192:8192x64

DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=u8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-zero-points=src:common:0+wei:common:0 \
  64x8192:8192x64
```
Observed debug (both):
```text
[streamk-debug] preflight: kParallelVariable=1 needsAGroupSums=0 needsBGroupSums=0 aOffset=1 bOffset=1 ... earlyDeqA=0 ... earlyDeqB=0
```
Results: both `PASSED`

#### Int8 with common scales
Commands:
```bash
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-scales=src:common:1+wei:common:1 \
  64x8192:8192x64

DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=u8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-scales=src:common:1+wei:common:1 \
  64x8192:8192x64
```
Observed debug: `kParallelVariable=1`, `needsAGroupSums=0`, `needsBGroupSums=0`
Results: both `PASSED`

## Attempts to trigger real group-sum cases

### 2. Benchdnn grouped per-K quantization syntax was not enough to produce a real runnable case here
I attempted grouped scales / grouped zero-points via benchdnn attributes, e.g. variants of:
```bash
--attr-zero-points=src:2:s32:1x32+wei:2:s32:1x32
--attr-scales=src:...:1x32+wei:...:1x32
```
with matmul shapes like:
```bash
64x4096:4096x64
64x8192:8192x64
```
Outcomes:
- several forms were rejected by benchdnn parser (`Error: the number of groups is expected to be 2`, `policy2mask`, or dt parse assertions depending on exact syntax), and
- the most plausible grouped zero-point form that benchdnn accepted reached primitive creation but returned:
```text
UNIMPLEMENTED
```
Example command:
```bash
./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C \
  --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-zero-points=src:2:s32:1x32+wei:2:s32:1x32 \
  64x4096:4096x64
```
Result:
```text
0:UNIMPLEMENTED
```
So on this hardware/build, the benchdnn path that might have exercised grouped per-K zero-point compensation did **not** materialize into a runnable GPU primitive.

## Forced stream-K / kv strategy checks

### 3. Forcing a `kv` strategy via `GEMM_KERNEL` works, but only on non-group-sum cases tested
I verified the dev-mode override path works with a `kv` strategy string template from `kernel.db`:
```bash
GEMM_KERNEL='gemm f16f16f16 NNN 64 64 1 1 av16+m32@72 am32+m32@64 aB wg 4x8 xaf vav hi pt sr br sb64 bk0 sn grf256 sys kv afb' \
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn \
  --engine=gpu --matmul --mode=C \
  --dt=f16:f16:f16 --stag=ab --wtag=ab --dtag=ab \
  64x8192:8192x64
```
Observed debug:
```text
[streamk-debug] preflight: kParallelVariable=1 needsAGroupSums=0 needsBGroupSums=0 ...
```
Result: `PASSED`

Also passed with this forced-`kv` setup for:
- `64x4096:4096x64`
- `64x16384:16384x64`

This confirms the override path is usable, but I still did **not** have a real benchdnn-runnable grouped-zero-point case to combine with it.

## Whitebox/gtest evidence

### 4. Whitebox unit test covers the actually-unsafe condition structurally
The repository contains and I ran:
```bash
./build/tests/gtests/internals/test_internals_gemm_stream_k_quant
```
Result:
```text
[  PASSED  ] 9 tests.
```
That test explicitly constructs `GEMMProblem` objects where `needsAGroupSums()` / `needsBGroupSums()` are true and verifies that `preflight()` disables stream-K. It is useful structural evidence that the narrowed guard targets the intended hazardous condition.

However, this is **not** a real GPU benchdnn reproduction of the numerical failure; it is a whitebox construction, not an end-to-end executable matmul case on this setup.

## A/B matrix for `DNNL_STREAMK_ALLOW_GROUPSUM`
I could not produce a real GPU benchdnn case with `needsAGroupSums=1 || needsBGroupSums=1`, so I could not run the requested end-to-end matrix of:
- default (`DNNL_STREAMK_ALLOW_GROUPSUM` unset / guard active), versus
- force-enabled (`DNNL_STREAMK_ALLOW_GROUPSUM=1`)

for a true group-sum workload.

Therefore there are **no real PASS/FAIL A/B entries** to report for group-sum cases on this setup.

## Exact notable commands and outcomes

### Real stream-K-selected cases that passed
```bash
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=f16:f16:f16 --stag=ab --wtag=ab --dtag=ab 64x4096:4096x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=f16:f16:f16 --stag=ab --wtag=ab --dtag=ab 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=f16:f16:f16 --stag=ab --wtag=ab --dtag=ab 64x16384:16384x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=bf16:bf16:f32 --stag=ab --wtag=ab --dtag=ab 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=u8:s8:f32 --stag=ab --wtag=ab --dtag=ab 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab --attr-zero-points=src:common:0+wei:common:0 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=u8:s8:f32 --stag=ab --wtag=ab --dtag=ab --attr-zero-points=src:common:0+wei:common:0 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab --attr-scales=src:common:1+wei:common:1 64x8192:8192x64
DNNL_STREAMK_DEBUG=1 ONEDNN_VERBOSE=1 ./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C --dt=u8:s8:f32 --stag=ab --wtag=ab --dtag=ab --attr-scales=src:common:1+wei:common:1 64x8192:8192x64
```
All of the above: `kParallelVariable=1`, `needsAGroupSums=0`, `needsBGroupSums=0`, result `PASSED`.

### Grouped-zero-point attempt that stayed unreachable for end-to-end validation
```bash
./build/tests/benchdnn/benchdnn --engine=gpu --matmul --mode=C \
  --dt=s8:s8:f32 --stag=ab --wtag=ab --dtag=ab \
  --attr-zero-points=src:2:s32:1x32+wei:2:s32:1x32 \
  64x4096:4096x64
```
Result: `UNIMPLEMENTED`

## Conclusion

### Answer to (a): does the narrowed guard ever actually fire for realistic problems?
**Empirically on this setup: no real benchdnn matmul case was found where it fires.**
I found many realistic stream-K cases, but all had:
- `needsAGroupSums=0`
- `needsBGroupSums=0`

The only cases that plausibly target grouped zero-point compensation were not available as runnable GPU primitives in benchdnn here (`UNIMPLEMENTED` / parser dead ends).

### Answer to (b): if it fires, is stream-K safe when force-enabled?
**No real GPU benchdnn evidence either way was obtained**, because I could not create a runnable end-to-end group-sum case on this setup.

So the empirical decision point is:
- the current narrowed guard looks **practically dead / no-op** for realistic benchdnn workloads on XeHPC here;
- but I did **not** obtain end-to-end evidence strong enough to justify removing it outright, because I could not run a real `needsAGroupSums/needsBGroupSums == true` GPU matmul with `DNNL_STREAMK_ALLOW_GROUPSUM=1`.

If production guard changes are being considered, the strongest evidence from this run is:
1. **practical reachability appears extremely low to nonexistent**, and
2. **no real benchdnn workload hit the guard**,
3. but **no end-to-end proof of safety** was established for the true group-sum case.

My recommendation from these results alone: treat the guard as **moot in practice on current realistic benchdnn coverage**, but do **not** remove it solely on the basis of this experiment unless you can produce a runnable late-dequantized grouped-zero-point GPU case (or inspect/prove the compensation math another way).
