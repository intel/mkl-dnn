.. index:: pair: page; Basic Concepts
.. _doxid-dev_guide_ukernel_basic_concepts:

Basic Concepts
==============

Introduction
~~~~~~~~~~~~

The oneDNN micro-kernel API (also denoted ukernel API), is a low-level, sequential abstraction for CPU only. This API allows maximum flexibility and composability with user provided code. In particular, the user keeps full control of:

* threading logic, as this API is sequential and independent of any threading runtime.

* blocking logic, as the user can configure ukernel objects sizes to fit in local caches.

* customization, as user can interleave its custom code with ukernel code within a parallel region.

The API is designed to be as simple as possible, with the small number of abstractions, to have minimal potential overhead.

Memory representation
~~~~~~~~~~~~~~~~~~~~~

In oneDNN ukernel API, there is no dedicated abstraction for memory object. Users must describe memory properties for each ukernel operation through:

* a pointer containing the address of the start of a buffer.

* a set of dimensions (1 dimension for vectors, 2 dimensions for matrices).

* a set of strides, which for 2d matrices is the number of elements between two consecutive rows.

Some operations might require data in a given layout on some hardware architectures to benefit from hardware acceleration (e.g., interleaved rows/columns with a given granularity). This is exposed through a dedicated enum value.

Operation representation
~~~~~~~~~~~~~~~~~~~~~~~~

For all ukernel operations, there are 5 fundamental steps:

* create a dedicated ukernel object. This step uses only fundamental parameters that define the operation (e.g., memory input/output shapes, datatypes, ...).

* configure the ukernel object. This step is to guide code generation by setting attributes. Once configured, the user must finalize the object.

* query the ukernel object. This step freezes the configuration of the ukernel object. At this point, the user can query various information (e.g., if the ukernel code will require inputs/output in a specific format).

* generate binary code. This will effectively generate the code that will be executed. This operation is time consuming, so it is advised to hoist it out of the main computation loop as much as possible. This must happen only once for each ukernel object.

* execute the generated code.

