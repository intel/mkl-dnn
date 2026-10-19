ktool &mdash; Kernel Cataloging
===============================

*`ktool`* is gemmstone's kernel cataloging tool, performing both catalog maintenance (adding/removing/modifying kernel strategies) and a variety of automated testing (generation/functional/performance tests).


Kernel Catalogs
---------------

gemmstone maintains _catalogs_ of pre-tuned kernel strategies, which are used by gemmstone's kernel selector at runtime to choose an appropriate kernel for a given problem.

Each strategy is keyed to a specific GPU architecture, choice of data types, memory layouts, and memory alignments. To learn more about gemmstone's strategy options and strategy text format, see the [frontend documentation](frontend.md).

gemmstone comes with several catalogs, in the [`selector/db`](../selector/db/) folder:

* `kernel.db`: main catalog for GEMM kernel strategies, also used for SYRK/HERK and copy-based SYMM/HEMM/TRMM.
* `symm.db`: nocopy SYMM/HEMM kernel strategies.

There are also several catalogs for the experimental microkernel feature:

* `ukernel_mmr.db`: strategies with A/B coming from main memory.
* `ukernel_lmr.db`: A coming from local memory, B coming from global memory.
* `ukernel_mlr.db`: A coming from global memory, B coming from local memory.

By default, `ktool` operates on `kernel.db`; another catalog may be selected with the `-c/--catalog` option.

During development, it is often convenient to create temporary catalogs for new kernels, and later paste them into one of the official catalogs above. The `-c/--catalog` setting can be used to create and operate on such a temporary catalog.


`ktool` Operations
------------------

When run without arguments, `ktool` will print a help message with information on possible options and operations. The general command line syntax is:

    ktool <options> <operation> [filter | strategy...]

Many operations take an optional [kernel `filter`](#filtering-entries) that can be used to apply that operation to a subset of kernels in the catalog.

Others (e.g. `-a/--add`) take a complete `strategy` string, in the same format as use by the generator [frontend](frontend.md).

Note that both filters and strategies must come at the end of the command line. Other options can come in any order.


### Inspecting Catalogs

#### `-l/--list`

`ktool -l` lists the contents of a catalog. By default, catalog entries are shown in a condensed format, with one line per entry.

    > ./ktool -l
    #0  Gen9  gemm BBS NNN 32 16 ab2x2 as8x2 ab l4 acb       Model W [priority = 512] Restrictions: q
    #1  Gen9  gemm BBS NTN 32 16 ab4 ab4x2 ab l4 acb nmk     Model W [priority = 512]
    #2  Gen9  gemm BBS TNN 16 16 as8 as16 ab l4 acb          Model W [priority = 256]
    [...]

Each entry has:

* an _ID_, which may be used to identify entries in future operations.
* the associated GPU architecture
* the _strategy_ in string format
* a pre-fitted _performance model_, for estimating kernel performance on a specific problem size.
* a set of optional _restrictions_ limiting usage of this strategy.

By default, the performance model coefficients and restrictions are summarized briefly; more detail is available with the `-v/--verbose` option:

    > ./ktool -v -l
    #0      Gen9    gemm BBS NNN 32 16 ab2x2 as8x2 ab l4 acb
    Model Parameters (W)
      priority: 512
    Restrictions
      no suma

#### Filtering Entries

We usually don't need to list (or apply any other operation on) the entire catalog. `ktool` provides a number of mechanisms for restricting the scope of an operation.

*Filter Arguments*

`--list` (and many other operations) allows for additional filter arguments at the end of the command line, which may take the form:

* `<n>`, `#<n>`: the entry with ID `n`.
* `<string>`: any entry whose strategy contains `<string>`. If `<string>` begins with a digit, add `*` before it to avoid confusion with kernel IDs.
* `+onemkl`: kernels relevant to oneMKL.
* `+onednn`: kernels relevant to oneDNN.
* `-- -onemkl`: kernels not relevant to oneMKL (`--` is mandatory to avoid confusion with an option).
* `-- -onednn`: kernels not relevant to oneDNN.

*Filter Options*

* `-A <arch>, --arch=<arch>`: only show kernels matching the given architecture, e.g. `XeHPC`, `Xe3p`.

### `--list-ids`

This command is similar to `--list` but outputs only matching kernel IDs, separated by spaces, for easy consumption by scripts.


### Catalog Insertion/Deletion

#### `-i/--init`

Initialize or re-initialize a catalog.

Together with `-c/--catalog`, can be used to create a new temporary catalog. This is not usually needed as all operations will initialize a new catalog as needed.

#### `-a/--add`

Add a new strategy to the catalog. The new strategy is given as argument at the end of the command line. `ktool` will automatically fit the default performance model to the catalog; this will take several minutes.

> [!TIP]
> Prior to adding a new kernel, ensure your machine is set up properly:
>   * [MDAPI metrics](benchstone.md#performance-testing) must be available on your system. Try running `benchstone` performance testing on a sample kernel, and ensure that effective efficiency is displayed.
>   * GPU frequency should be fixed, if at all possible
>   * Use a consistent GPU configuration when adding kernels. A top-bin system (maximum EU count) should be used.
>   * It's recommended to use OpenCL (ktool's default) when fitting kernels, as Level Zero tends to show additional overheads and greater performance variation.

By default, `ktool` uses the "extended" model (model `E`) for new entries; you may override this with the `-m/--model` option. In particular, the "wave-based" model (`--model=W`) is useful when benchmarking isn't feasible and you have a specific size range in mind for the entry (see [Restrictions](#-r--restrict-limits) to set up a size range). Model `W` does not benchmark kernels; when no GPU is available, specify the target architecture with `-A/--arch`.

> [!TIP]
> After adding a new strategies, run `ktool --test` to ensure that your new kernel generates correctly and passes accuracy tests.

#### `-f/--fit`

Fit a performance model to the given strategy, but do not add it to the catalog. Results from fitting are displayed.

#### `-d/--delete`

Delete catalog entries.

> [!TIP]
> Deleting an entry will adjust the IDs after it, so if you have a list of IDs to delete, proceed from largest to smallest!

#### `-x/--extract <other_catalog>`

Extract matching entries and place them in `other_catalog`. If `other_catalog` already exists, the entries are appended to it.

#### `-p/--append <other_catalog>`

Paste all entries from another catalog into this one.

#### `-k/--clone <types>`

Clone matching entries, changing the data types to the given `<types>`. For instance, a set of half precision kernels can be cloned to bfloat16 kernels with `ktool --clone BBS HHS`.


### Catalog Modifications

#### `-r/--restrict <limits>`

Add, modify, or remove restrictions on an entry. The `<limits>` argument can be one of:

* `steppings:<s0>-<s1>`: limit entry to steppings in the range $[s0, s1]$ (inclusive). `any` may be used in place of a number to indicate no restriction.
* `allowed:(m0,n0,k0)-(m1,n1,k1)`: limit entry to problem sizes $m_0 <= m <= m_1$, $n_0 <= n <= n_1$, $k_0 <= k <= k_1$. Use `any` to indicate no restriction.
    - Note that you will likely need to quote this argument to protect it from the shell.
* `accept:(m0,n0,k0)-(m1,n1,k1)`: always choose this entry for the given range of problem sizes.
    - If two catalog entries have overlapping `accept` ranges, the behavior is unspecified.
* `-<tag>`: do not use entry in cases described by the given `<tag>`.
    - A full list of tags can be found by running `ktool --help`.
    - Tags are often used if a strategy runs out of registers when used with a particular kernel variant (e.g. when A/B offsets are enabled).
    - Some tags are added automatically when adding a kernel (e.g. 64-bit buffer support).
* `+<tag>`: only use entry in cases described by the given `<tag>`.
* `xsteppings`, `xallowed`, `xaccept`, `x<tag>`: remove an existing restriction.

Certain operations allow `--restrict` to be used as a modifier:
* `-a/--add` will add the provided restriction to the new kernel.
* `-l/--list` will filter listed results by the given restriction, e.g. `ktool -l -suma` shows all kernels tagged "no suma".

#### `-M/--modify <script>`

Modify kernel strategies by filtering the strategies through the given `<script>`.

The script is invoked once for each entry. The current strategy string is passed as input to the script. `ktool` will replace this string with the first line of the output from the script.

After using `--modify`, you will want to invoke `ktool --reinfo` on the modified entries.

#### `-I/--reinfo`

Invoke the generator frontend to update the driver information structures in the catalog. This should be done each time a strategy is modified, either with `--modify` or manually.

#### `-R/--refit`

Refit the performance model for matching catalog entries (see [above](#-a--add) for notes on fitting models).


### Functional Testing

#### `-g/--gentest`

Check that kernels can be successfully generated for the given entries. `ktool` will automatically check several variants of the given kernel (e.g. syrk, batch, offsets, scales).

If some variants fail to generate (e.g. running out of registers), it may be appropriate to [restrictions](#-r--restrict-limits) that avoid this.

Several options affect the kernel variants that `ktool --gentest` tests:
* `-N/--no-derived`: only test the first variant, and nothing fancier.
* `-D <variant>, --derived=<variant>`: test the given variant, e.g. `scalea[g32] gemm`.

#### `-t/--test`

Like `-g/--gentest`, but also run accuracy testing on the given kernels. This includes out-of-bounds (read check/`rc`) testing.

#### `-j/--compare-test`

Runs two different builds of the frontend and checks whether they produce identical kernels. This is useful to verify that generator changes don't (or do) affect kernels you didn't (or did) intend to affect.

The "base" version should be suffixed with `-stable`, e.g. `frontend12p8-stable`.


### Kernel Selection

#### `-e/--eval <size>`

Run the kernel selection process for the given problem `<size>`, which should be specified as `<m>x<n>x<k>`, or for batched kernels, `<b>x<m>x<n>x<k>`.

The command line should include the problem information (kernel type, data types, memory layouts/alignment, alpha/beta), e.g.

    ktool --eval 100x200x300 gemm BB[SB] N@8T@8N
    ktool --eval 32x32x29900 gemm ZZZ TNN 1 1

Matching kernels will be listed, in order of score (lowest to highest, best to worst). A score of -inf indicates an automatically accepted kernel, due to the problem size (see [Restrictions](#-r--restrict-limits)).

If sorting is not desired, use the `-U/--unsorted` option.

#### `-y/--evaltest <size>`

Similar to `-e/--eval` except that `ktool` invokes `benchstone` to check actual performance for each candidate strategy.

By default, `ktool`, like `benchstone`, will assume well-aligned leading dimensions. Use the `--tight-ld` argument to evaluate tight leading dimensions.

> [!NOTE]
> In certain cases `benchstone` performance, as reported by `-y/--evaltest`, may differ from actual performance in oneMKL/oneDNN due to different choices made in the host-side code.


`ktool` Options
---------------

*General options:*

* `-c, --catalog=<path>`: specify a different catalog. If `<path>` contains only a file name, `ktool` will look for the catalog in `selector/db`.
* `-v, --verbose`: enable extra output.
* `-O, --opencl`: use OpenCL runtime for benchmarking.
* `-0, --l0`: use Level Zero runtime for benchmarking.

*GPU information:*

* `-A, --arch=<arch>`: specify GPU architecture.
* `-S, --stepping=<n>`: specify stepping (default is 0).
* `-E, --eu=<count>`: specify EU count (default: autodetect).
* `-C, --llc-mb=<MB>`: specify last-level cache size, in MB (default: autodetect).
* `-F, --frequency=<MHz>`: specify nominal frequency (default: autodetect).

You can use options to emulate a different GPU architecture, e.g. for the `--eval` operation.
Model `E` and `S` fitting benchmarks kernels on the host GPU; `-A/--arch` must match that GPU. The `-E/--eu`, `-F/--frequency`, and `-C/--llc-mb` overrides are also used when fitting. Model `W` does not require benchmarking and can target a different architecture.
