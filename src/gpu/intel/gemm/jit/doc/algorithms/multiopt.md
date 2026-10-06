Multiopt: Kernel Optimizer
==========================

This page provides design details for multiopt, the optimizer core used by the multiderby autotuner.

Multiopt supports multiderby's major objectives:
* Search intelligently through the (very large) kernel strategy space.
  - Stay near known-good strategies; avoid bad strategies.
  - If the current strategy is near-roofline, try to make small steps (fine-tune the current strategy). If it is not, try to make more drastic strategy changes.
  - Focus on problem sizes where we have not yet identified good strategies.
* Automatically learn which strategy parameters have large effects on performance, and which have smaller effects (a _sensitivity metric_).
* Build up a roofline model based on observed data.
* Find a minimal set of kernel strategies that provide sufficiently good performance over the full space of problems.

Multiopt is designed as a standalone kernel optimizer, not tied to gemmstone's generator or GEMM-like kernels. Both problems and strategies are encoded as arbitrary integer vectors, which we call problem vectors (PVs) and strategy vectors (SVs).

The multiderby [_client_](../../tools/multiderby_client.hpp) encodes and decodes these vectors, and interfaces to benchstone for performance measurement. In multiderby, problem vectors are 3-dimensional (m, n, k), while strategy vectors are much larger, with entries for tile sizes, prefetch distances, compute type, etc. (definitions [here](../../tools/multiderby_common.hpp)).


Search Methodology
------------------

Multiopt implements a multi-agent local search, where each agent stochastically moves through (PV,SV) space. The [`Agent`](../../tools/multiopt/agent.hpp) class encapsulates a single agent.

The main events in an agent's lifecycle are `next()` (generate new benchmarking requests) and `notify()` (called on benchmark completion). For a standard local agent, `next` looks like:
* Randomly modify one entry of current SV
  - Not all SVs are associated with viable kernels (e.g. incompatible strategy options, SV entry out of bounds, etc.), so the client can pre-validate SVs to improve the odds that the mutated SV can be generated.
* Decide whether to accept this candidate SV (`acceptStep`)
  - This is choice is made by estimating the probability that the candidate SV will be better than the current PV's best SV, or sufficiently better than the current SV.
  - The estimate is made using the sensitivity model (described later).
* Submit a benchmarking request (`AgentRequest`) for the new SV on the current PV, as well as _neighbors_ of the current PV (nearby problem sizes).
  - Note: neighbors will only be checked if the new SV's performance on the current PV is good enough.

After receiving benchmark data, `notify()` does the following:
* Check if strategy is successfully generated (i.e. didn't run out of registers, etc.)
* If successful, compare its performance to the previous SV and decide whether to accept it
  - The new SV is always accepted if better than the old.
  - If the new SV is (slightly) worse than the old, multiopt will probabilistically accept it (somewhat akin to simulated annealing) in an effort to avoid local maxima.
* Decide whether to retarget
  - Retargeting means to abort the current local search, and jump to a new (PV,SV) pair.
  - Retargeting generally occurs probabilistically, depending on how far we are from the roofline (avoid spending too much time on bad strategies).
  - In a retarget, we choose:
    * A new PV -- chosen randomly, with more weight on PVs that are further from the roofline
    * A new SV -- either starting with the current best SV (more likely as it approaches roofline performance), or generating a new SV randomly and estimating its performance based on proximity to existing SV data
      - Note that the sensitivity model provides an estimate of the performance for any SV on a PV based on available data from other SVs on that PV. The sensitivity model does not attempt to correlate performance across PVs.

Multiopt also has a special kind of agent, the _spreading agent_ (`SpreadingAgent`), which is activated periodically during the optimization process. While regular agents perform local searches, multiopt's eventual goal is global &mdash; to find a minimal set of SVs that cover the full space of PVs. The spreading agent acts as a local-to-global step. It takes a set of top SVs and benchmarks them on neighboring PVs (in a extended local search) as well as nonadjacent PVs (a global search). Like `Agent`, `SpreadingAgent` interacts with the optimizer via a `next()`/`notify()` interface.


Sensitivity Model
-----------------

A crucial component of multiderby's search methodology is the _sensitivity model_. Multiopt defines the sensitivity $\sigma_i$ to be the variance of the absolute value of the log performance ratio arising from modifying an SV by 1 in its $i^{\text{th}}$ dimension.

That is, given a random SV $X=(x_i)$ and a neighbor $X'=(x'_i)$ such that $x_j = x'_j$ for $j\neq i$ and $\left|x_i - x'_i\right| = 1$, with performances $\text{perf}(X)$, $\text{perf}(X')$,
```math
\sigma_i = \text{Var\ } S_i, \text{\ \ where \ \ } S_i = \left| \text{log2}\left(\frac{\text{perf}(X)}{\text{perf}{X'}}\right) \right|
```
For simplicity, multiopt assumes that the $S_i$ are i.i.d. and normally distributed. This is a questionable assumption for some strategy entries, where some covariance is expected, but significantly accelerates the time needed to initialize the sensitivity model.

Multiopt periodically updates the sensitivity model based on available data, using maximum likelihood estimation (which reduces to a simple least-squares fit in this case). This logic is part of the [`DataStore` class](#data-store).


Roofline Model
--------------

As described in the [multiderby documentation](../autotuners.md), multiopt builds up an empirical roofline model from observed data, in addition to using a client-supplied absolute roofline.

For simplicity, the current empirical roofline model is a simple neighborhood-weighted maximum, where a PV's roofline is determined as the maximum of the achieved performance (relative to the absolute roofline) of neighboring PVs, with a fall-off factor as these neighboring PVs become more distant. The fall-off ensures that the roofline extends to a continuous function of PVs.

In the future, a distance-weighted convex hull model would be more appropriate.


Optimizer
---------

The main optimizer loop is implemented in the [`Optimizer` class](../../tools/multiopt/optimizer.hpp). It orchestrates the agents, relays benchmarking requests to and from the client, and performs periodic tasks (spreading, adding new PVs).


Extractor
---------

The _extractor_ ([`Extractor` class](../../tools/multiopt/extractor.hpp)) is responsible for identifying a near-minimal set of SVs that provide adequate performance for all PVs. (As usual, "adequate" is defined via a tunable setting.) The extractor should be run just after the spreading agent completes, to ensure that the best SVs have been benchmarked on relevant PVs.

The current extractor implementation uses a simple greedy algorithm for choosing SVs, so it does not guarantee minimality (which is likely an NP-hard problem).


Data Store
----------

Multiopt keeps all benchmarked data, the sensitivity model, neighborhood information, and other derived data in the [`DataStore` class](../../tools/multiopt/data.hpp). It is designed to be interrupted and restarted at any time, supporting multiderby's checkpointing feature.


Tuning Multiopt
---------------

Multiopt, like most optimizers, has a large number of heuristic settings (contained [here](../../tools/multiopt/settings.hpp)) that can/should be tweaked depending on the specific client. The current settings were tested and adjusted for multiderby, but different projects may need different settings.
