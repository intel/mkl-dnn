# RFC: Symbol Visibility for Internals Unit Tests

## Problem Statement

Sometimes there's a need to validate internal oneDNN functionality that is not
exposed with public symbols.
The current state of the art is that needed symbols are simply exposed from
inside the library to let the linker discover them, since oneDNN compiles every
translation unit with `-fvisibility=internal` and `-fvisibility-inlines-hidden`
options.
Only `DNNL_API` symbols get `visibility("default")` by design.
The shipped `libdnnl.so` therefore exports only the public API; internal symbols
are absent from the dynamic symbol table,
so a test can't resolve them by linking the product library, and making holes
in the "export" lists is necessary.

This is not an active bug today, but there's always a chance to somehow exploit
exposed symbols, especially if they work with pointers.
The shipped package must keep exporting exactly today's public API.

Baseline (release, `--graph=on --gpu=ocl`, x64):

| Artifact | Size |
|----------|------|
| Product `libdnnl.so` | ~97 MB |
| Product object footprint (all `*.o`) | ~281 MB |

Additionally, code to test must be visible from header files (can't live in
dedicated translation units), which nails it to a specific placement.

There's a desire to change the current approach in two aspects:
* Resolve the internal symbols exposure problem.
* Stop manually controlling the desired symbols and their dependencies.

## Proposals

### Option 1 — `DNNL_TEST_API` conditional annotation

Add a `DNNL_TEST_API` macro. In a product build it expands to nothing (symbol
stays hidden); in a test build it expands to
`__attribute__((visibility("default")))` (what `DNNL_API` macro does),
exporting the annotated internals from the same `libdnnl.so` the tests link.

Trade-offs:
* **Test-enabled and shipped artifacts are different binaries** built from
  different preprocessor configurations. Binary packages must ship the non-test
  build, so tests don't validate exactly the binary that is shipped.
* **Still manually driving test dependencies**.
* Minimal changes and simple flow.

### Option 2 — Static archive (recommended)

Compose a static archive from all object files during the build and link each
unit test (and benchdnn) against it.
Static linking ignores visibility, so the shipped `.so` is untouched and no leak
reaches any distributed artifact.

Trade-offs:
* Two libraries - one shared object for shipping, and one archive for tests.
  Same compiled object files are tested.
* **Binary size**. Due to past architectural designs, most tests would have the
  size of the library.
  It's caused by the following chain of events: engine contains the list of
  implementations for each primitive, which is all the implementations the
  library has, which will pull all auxiliary functions to support them.
  Basically, any test that touches a function that depends internally on engine
  or primitive descriptor pulls all `.o` files transitively approaching the full
  library size.
  The library size is provided above with GPU and Graph components enabled.

### Option 3 — Dedicated all-visible shared object

Compile a second twin of every component with `-fvisibility=default` and
assemble the twin into the test shared object that exports all internal symbols.
Tests link this `.so`; the product library is built in its current way.
The all-visible `.so` is ~8% larger than the product `.so`; the delta is the
dynamic symbol table.
The advantage of the test shared object is that test binary sizes remain small,
unlike for the archive option.

Trade-offs:
* Two libraries - one shared object for shipping, and one - for tests. **Not**
  the same compiled object files are tested.
* **Double compilation** - one build run produces both flavors; product library
  unaffected; leaked symbols confined to a separate, non-shipped artifact.
* **Duplicated targets in CMake infrastructure**: every source compiled twice,
  with hidden and default visibilities, roughly doubling object compilation
  (especially on low-powered systems) and object footprint.
* Test binaries need a new shared object at runtime (RPATH/LD_LIBRARY_PATH).

### Option 4 — Visibility manipulation via export map

Since symbol visibility in the shipped shared object is a linking property
(unless compiled the way oneDNN compiles), there's another option to expose
symbols for testing.
The visibility of object files can be switched to default. In such a case,
controlling what the shipped library exports can be done through a linker
version script (or the export map) that localizes everything except the public
API.
A version script hides symbols at link time but cannot re-export symbols already
hidden at compile time; compile visibility and map must be designed together.
This addresses the double compilation trade-off of the previous option but
reintroduces manual driving of what symbols should be exported.

Trade-offs:
* Two libraries - one shared object for shipping, and one - for tests. Same
  compiled object files are tested.
* **Manual driving of an export map which is responsible for the product ABI**.
  Higher impact from making a mistake in a list.

## Comparison

| Criterion \ Options        | `DNNL_TEST_API` | Archive        | All-visible `.so` | Export map      |
|:---------------------------|:----------------|:---------------|:------------------|:----------------|
| Validating shipped lib (1) | Not quite, low  | Not quite, low | No, high          | Not quite, high |
| Compilation                | Once            | Once           | Twice             | Once            |
| Test binary size           | Small           | Big / Lib size | Small             | Small           |
| New runtime deps           | No              | No             | Yes               | Yes             |
| Manual managing            | Yes             | No             | No                | Yes             |

(1) also explains risks for testing not quite the same thing. Low risk means
that proposed effects validate same material content but the content's package
is not the same. High means the material is not the same compared to the shipped
material.

## Recommendation

It's recommended to go with the static lib option despite the binary size
concerns.
The potential solution to overcome it is to re-use binaries to test different
functionality. If the binary is proven not to occupy the library size space
(which can happen given it doesn't have dependencies on strings that pull the
library), it's fine to keep it separate, they usually occupy 1 MB.

The suggestion for the problem with managing large product packages is to
postpone it until it becomes a real burden, while keeping an eye on how binaries
are managed.
