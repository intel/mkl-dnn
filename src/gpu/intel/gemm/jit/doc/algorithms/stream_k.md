Variable k-Slicing/Stream-k
===========================

Parallelizing GEMM in the k dimension ("k-slicing" or "k-blocking") is a standard technique for increasing parallelism when parallelization in m/n alone is insufficient.

Suppose we have a GEMM of size $m = n = 96$, $k = 480$, and suppose each workgroup computes a $32\times32$ tile of the $C$ matrix, so there are 9 tiles available:

<p align="center">
<img width="333" height="337" alt="C tiles" src="images/stream_k_tiles.png" />
</p>

Let's imagine our GPU can execute 32 workgroups concurrently. Without any k parallelization, only 9 workgroups will be launched &mdash; most of the GPU will remain idle.

To increase parallelization, we can split the $k = 480$ dimension into three chunks of size $k_0 = 160$. We now have work for $9 \cdot 3 = 27$ workgroups. Let's combine the m/n dimensions into a single "tile" dimension on the y axis, and place the k dimension on the x axis:

<p align="center">
<img width="532" height="463" alt="k-slicing" src="images/stream_k_k_slicing.png" />
</p>

This is the global k-parallelization strategy (`kb` in gemmstone terminology). Note that the increased parallelism is not free: each $32\times 32$ tile of C receives contributions from 3 workgroups, requiring a careful (and potentially costly) cross-workgroup reduction &mdash; more on that later

With a little more work, k-slicing can be turned into a load-balancing technique known as _stream-k_. Note that in the example above, even with k-slicing we only launched 27 workgroups out of 32 possible, so the load is still not perfectly balanced across the GPU. Could we divide the work so exactly 32 workgroups run in parallel? The _stream-k algorithm_, introduced by [Osama et al.](https://arxiv.org/pdf/2301.03598), makes this possible. The idea is to concatenate all the GEMM work (9 tiles, each with $k = 480$) into a single "k stream" of length $9\cdot480=4320$, then divide this stream into 32 equal pieces of length $k_0 = 4320/32 = 135$:

<p align="center">
<img width="695" height="230" alt="Stream-k split of the combined k dimension" src="images/stream_k_split.png" />
</p>

Note that some workgroups (such as WG 3) contribute to multiple C tiles: their segment of the k stream crosses a C tile boundary. As in global k-parallelization, a cross-workgroup reduction is needed to combine the partial results for each tile.

If we could remove all overheads associated with stream-k, it would provide essentially perfect parallelization for our GEMM: each workgroup does the same amount of compute ($m\cdot n \cdot k_0 = 32\cdot 32\cdot 135$), and requires the same amount of A/B data.

Real stream-k implementations come with overheads, including:
* Additional overhead associated with computing multiple C tiles in a single workgroup: k loop warmup and cooldown, writing out C data.
* Reduction across workgroups: reduction in global memory or in registers, requiring costly inter-workgroup synchronization. In addition, these reductions may need an initialization step.
* Imperfect k partitioning: for good performance, the k pieces typically need to be aligned to a boundary of 16/32/64/128 elements, to ensure proper memory alignment and fill compute instructions (e.g. dpas).
* Post-ops: Any fused post-GEMM operations, including data type conversions, are not automatically load-balanced. In addition, they must wait for cross-workgroup reduction to be complete.
* Synchronization costs: Inter-workgroup synchronization can only be done via atomic operations, which incur non-trivial latency that is hard to avoid.

In gemmstone, stream-k (or variable k-slicing as it is called in the code) is activated with the `kv` strategy modifier.


Algorithmic Refinements
-----------------------

This section discusses how the basic stream-k algorithm can be refined for better performance. Throughout, we use the following notation:
```math
\begin{align}
D &= \text{\# of workgroups that can execute concurrently on the GPU}\\
N &= \text{total \# of C tiles in matrix}
\end{align}
```
A _full wave_ of tiles is a set of $D$ tiles. If we partition the $N$ total tiles into full waves, we may have a leftover _partial wave_ of size $N\text{ mod } D$ at the end.

### Limiting Stream-k Usage 

The best way to avoid stream-k overheads is to minimize usage of stream-k.

In a large GEMM, when there are many more tiles than concurrently executing workgroups, there's no need to apply stream-k to all the tiles. Instead we apply stream-k only to the final partial wave and the full wave preceding it, if any. Other full waves are executed normally. The rationale for including the final full wave is to ensure that each C tile has at most two workgroups assigned to it, which reduces the synchronization cost for reduction.

### Virtual k-Padding

In stream-k, workgroups can contribute to one or more C tiles. As mentioned earlier, there is a penalty (cooldown/writeout/warmup) associated with the WG moving from one tile to the next. This doesn't affect load balancing if all workgroups contribute to the same number of C tiles, but in general different workgroups will contribute to different numbers of C tiles, so the penalty is not evenly balanced.

gemmstone tries to improve load balancing in these conditions by introducing _virtual k-padding_. In this scheme, before splitting the k work between workgroups, we pad the k dimension by a (usually) fixed amount (strategy knob: `pk<N>` / `GEMMStrategy::kPadding`, default `pk32`). This padding is taken into account when splitting the k work, but then removed before computation (as it doesn't represent real GEMM work). Currently, padding is placed at the beginning of the k range.

The effect of padding is to reduce work for WGs that cover more tiles.

> [!NOTE]
> Padding is automatically limited to $2k_0$ (to avoid assigning no work whatsoever to a workgroup). Host code must be aware of this when choosing $k_0$.

### Aligning k-Streams for Cache Coherence

When k is relatively large, stream-k usage can lead to poor use or thrash of the last-level cache (LLC). Let's take $D=7$, $N=3$ as a toy example, and visualize the basic stream-k algorithm:

<p align="center">
<img alt="Basic stream-k tile order" src="images/stream_k_llc_basic.png" style="width:10%; height:auto;"/>
</p>

The 3 tiles are arranged in the x dimension, and k is placed along the y dimension. Each color represents one of the 7 workgroups, while the numbers show the order in which the workgroups process GEMM work. The key thing to notice is that _no two workgroups are ever simultaneously using the same A or B data_. Hence at the beginning of the kernel (processing the blocks marked 1) the peak BW required from global memory is potentially 7x larger. And, if LLC is unable to hold all A/B data, data from the start of the "1" blocks may be evicted from cache before the "2" blocks even begin.

We can improve the situation somewhat by having each workgroup traverse its C tiles in reverse order:

<p align="center">
<img alt="Reversed tile order" src="images/stream_k_llc_reversed.png" style="width:10%; height:auto;"/>
</p>

Note that this arrangement moves the 1s closer together, as well as the 2s and 3s, increasing cache reuse. The minimum LLC capacity required to avoid reduction is also slightly reduced. By itself, though, this is a small effect. Better improvements can be obtained by aligning some of the k streams:

<p align="center">
<img alt="Aligned k streams" src="images/stream_k_llc_aligned.png" style="width:10%; height:auto;"/>
</p>

In the aligned section of k space ("synchronized section" in gemmstone parlance), we get perfect LLC coherence, and aren't affected by eviction. gemmstone refers to a group of k-aligned workgroups as a _slab_. Stream-k is used in the remaining section of k space ("unsynchronized section"), but because there are fewer WG participants, LLC capacity and bandwidth utilization are improved. Based on analytical considerations, gemmstone aims to choose the number of slabs so that the number of remaining workgroups is either 0 or between $D/2$ and $3D/2$. The closer this remainder is to $D$, the smaller the BW and LLC capacity requirements.

Even with slabs, there are stream-k cases with bad LLC coherence. gemmstone has an option to split the set of stream-k-balanced tiles into two _phases_ and process each phase independently. This can improve LLC coherence in certain cases, at the cost of additional reduction overhead. As it is difficult to judge when this trade-off is worthwhile, the two-phase option is not currently enabled in oneMKL/oneDNN.


Cross-Workgroup Reductions
--------------------------

One of the major optimization hurdles in global k-parallelized kernels (including stream-k) is handling the cross-workgroup reductions. Reductions can only occur in global memory as that is the only memory space available to all workgroups.

There are two main implementation strategies:
1. Each workgroup writes its partial C tile to per-WG temporary memory. After all WGs complete, their results are read back, reduced, and the final tile is written to the C matrix.
2. All workgroups share a single C tile in memory. Each workgroup atomically adds its partial C tile to the shared tile.

The original stream-k paper seems to use strategy 1 -- however this requires a large amount of scratch memory and extra time and memory traffic from the reduction pass.
Since all modern Intel GPUs have native int32 and f32 (since ATS-P) atomic add, gemmstone uses atomic reductions (strategy 2).

We can divide atomic reductions into three phases:
* **Initialization**: shared C tile must be initialized to zero (or scaled by beta) prior to any atomics
* **Reduction**: atomic add operations occur
* **Finalization**: any post-GEMM operations, including converting C to a lower precision or applying fused post-ops

A cross-WG synchronization is needed between these three phases. The simplest way to achieve this synchronization is via separate kernel launches; however, the separate launches incur significant overhead. 
Instead, gemmstone by default fuses initialization/finalization into the stream-k kernel, using atomic counters to synchronize across WGs.

To fuse initialization, one workgroup is elected to perform the initialization step (_beta scaling_ in gemmstone) and sets the atomic counter to inform other WGs upon completion. gemmstone's default method (`fb`) uses an atomic counter to elect this WG, which writes to C prior to entering the k loop. gemmstone also has a faster method (`afb` = alternate fused beta) that relies on HW workgroup dispatch and is used in production strategies. In this method, the initializing WG is statically assigned, and instead of separate initialization + reduction steps, it writes its results directly to the shared C tile before signalling other WGs.

To fuse finalization, a separate atomic counter keeps track of accumulations to the shared C tile. When the final WG adds its contribution to the tile, it reads back the shared C tile, applies post-ops/conversions, and writes the C tile to memory. Note that when the accumulator has higher precision than C, we need to maintain the shared C tile in accumulator precision in temporary memory. _Future optimization:_ statically assign final WG &mdash; it should load the other WG's contributions to registers and reduce there. 

Both fusions (initialization/finalization) make use of atomic counters for synchronization, and these atomic counters themselves need initialization outside the kernel! To avoid the cost of initializing the counters, oneMKL/oneDNN maintain a special memory area called the _zero pool_. The zero pool is initialized to zero upon creation. Afterward, it holds zeros whenever not in use by a kernel. A kernel may write non-zero data into the pool during execution, but must reset the memory back to zero before completion. The synchronization algorithms in gemmstone are designed to be compatible with these requirements, so that by using the zero pool for atomic counters, no separate counter initialization is required.


Stream-k Kernel Evaluation
--------------------------

When choosing a kernel for a particular GEMM problem size, gemmstone's kernel evaluator estimates the performance of each candidate kernel from the catalog for that problem size. In the case of stream-k-enabled kernels, gemmstone will estimate performance both with and without stream-k (see [`evaluateE`](../../selector/kernel_evaluator.cpp)), and indicate whether stream-k is beneficial via `EvaluateAuxOutput::kParallelVariable`. If stream-k is not beneficial, it will be disabled by `modifyStrategy`.
