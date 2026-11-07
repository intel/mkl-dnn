# Introduction
The purpose of this RFC is to re-establish the content of oneDNN functional validation and its procedural update. The request comes from external parties that rely on oneDNN quality.
The challenge is to provide sufficient coverage to avoid bugs escapes in a rapidly changing environment of new functionality added to the library.

## Vision
Validation exists to prove a change, or a patch set is safe to promote into the repo.
(Though validation should cover both functional and performance aspects, this RFC will focus on the functional part of it).
The more complete functional coverage tests provide the higher chances are the change is safe.
This proof must come in reasonable time. If it takes a long time (e.g., a day) to verify any kind of change, such process gives less value for the development compared to one that responds to results much quicker.
However, expanding functional coverage with the bigger number of test cases takes more testing time, thus, the resulting vision statement takes the following form:
The balance between execution time and coverage must be kept wisely; the team must achieve the most possible coverage under specific time budget.

# Goals
Here are the ultimate goals the team wants to achieve:
* Time resources spent per single validation instance, or testing job (under assumption of infinite hardware resources):
    - CI: 2 hours.
    - Nightly: 8 hours.
* Functional coverage:
    - Most useful API combinations to ensure customer satisfaction with day-to-day functionality used in their environment.
      This coverage should be expressed in a generalized way, not tied to specific shapes or a fixed set of API parameters.
    - Corner cases to ensure getting off the streamline path doesn't lead to undesired behavior.
    - Hit optimized library implementations with 80+% rate as slow implementations (reference codes) don't bring value to customers.

# Proposal
To fit more combinations with less test cases is possible when a single test case has various flags with non-default values.
Given the infinite space of possibilities of various combinations, the best shot is to create a tool that would generate distributed points in that combination space. Coarse distribution can cover more various, usually independent implementation-wise, features of the library versus the targeted specific feature validation through iteration over some of options (as it's done currently). A tool (over the manual update) is preferred as more reliable and faster way to extend the coverage points in shorter time.

A smart and scalable test case generator that would provide a single (no iterations over any arguments) benchdnn test case as its output is envisioned as such a tool.
Its smartness comes in two aspects:
    - It's adapting to the library API combinations, returning only test cases that make sense from the library perspective, not a sporadic combination (a.k.a. black box testing).
        E.g., quantization is applied to integer data types; test case masks follow dimensions used and reflect real-world applications; etc.
    - Uses probability mechanism to balance the amount of most useful combinations versus corner cases (test most useful functionality more often).
    - Control ranges of random values to reach the desired goal of 80% hit rate.
Scalability comes with an architecture that relies on regex strings to express more supported patterns with less implementation logic.
E.g., data types are represented as x4 for s4 or u4, xf16 for bf16 and f16, etc.; use cases are combined in some generic policies like MXFP8 quantization; not fixed values represented with '*', etc.

Once such generator exists and covers a benchdnn driver, here is the proposed model of its usage on the example of CI set:
- Determine the list of features for a given driver that would form the core functionality.
    * Core functionality must satisfy the following requirements:
        - It must be shared between CPU and GPU backends.
        - It should have optimized support in the library. It doesn't necessarily mean the optimized scope must be identical between backends.
    * Core functionality should occupy the most space of coverage for a driver.
    * Test cases targeting core would reside in `harness_driver_core_ci` file.
- Determine the list of exclusive features for a backend or limited support scope.
    * Such features should occupy a small portion of the total number of all test cases.
    * Test case targeting such features would reside in `harness_driver_featureName_ci`.
        E.g., ‘featureName’ can be 'f64' which is exclusive to GPUs and exists in various primitives.
- A set of harnesses form `test_driver_ci` file which will replace existing file and its scope.
- Verify the coverage fits the time limits across all supported hardware and satisfies proposed optimized implementation rates.
    - In case it doesn't, analyze rates and/or times, make decisions based on the assessment.
    - Refine the coverage.
- Rough estimation is 5,000 cases.

Same process will apply to Nightly set. The difference between Nightly and CI is the number of cases covered, which in case of Nightly would be around 20,00 - 25,000 and allowed problems might exceed 50 MBs up to (500 MB).
Larger shapes will be validated in Nightly as it has bigger time budget.

The process of appending coverage to existing files will be manual and will consist of several steps:
1. Update the generator with a new scope (extending existing feature capabilities / introducing new feature through benchdnn flags) to the desired percent-wise state of core functionality as if it becomes a part of the core. If it takes a dedicated file, there is no need to take this step.
2. Once the desired state is achieved, generate test cases containing only the new scope and append these cases to a correspondent file manually.

The renewal process of core files (or re-generation) may be triggered by several factors:
* Periodically, somewhere in the middle of development process for a coming release.
* Once the time budget is exhausted, leading to longer jobs times stop fitting the time goal.
* Once the scope of functionality changes, e.g. proportions of cases were adjusted in a major way to comply with the trend industry path.

This tool is expected to be posted in oneDNN repository for everyone to use it.
Every developer will be responsible for extending the tool to cover functionality they are adding whether it's a new feature or a specific important set of settings.
Additional value of the tool is when it's mastered, a developer will be able to tailor local testing coverage for features/bugs they are working on.

The final state of input files includes a various harness_driver_xxx_ci files and three test_driver_{smoke/ci/nightly} files.
Any kind of filtering problems (don't run these on that HW or run those on this HW) must be regulated through other approaches but not through “test” files.
Same approach shouldn't be also used for better balancing.
Multiple "test" files per driver lead to repetitive validation, higher pressure on Intel internal automation system and lack of transparency what's included and validated.
Having a single file should address these problems.
Balancing can be done by CMake, if or when needed, by splitting a list into chunks and keeping shorter test files and targets associated with them in a build folder.

## Synthdnn
Currently, there's already a script that can perform case generation, located at src/scripts/synthdnn.
So far it supports various data types, fpmath-mode, tags and shapes.
The problem is it lacks a probability architecture piece which makes it less controlled on the output it provides.
It acts more like a black box testing without adjusting to actually supported scenarios which wouldn't help with the announced goals once more features are added to it.

## Fuzzdnn
The showcase work is done on the base of another internal generator, "fuzzdnn".
Its latest output can be checked out [here](harness_matmul_core_ci). Notice the variability of knobs used per case, sizes and other settings.
This is still work-in-progress, some complete test cases may not make total sense, and they will be adjusted once it becomes easier to analyze what the limitations are.
