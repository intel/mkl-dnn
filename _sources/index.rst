oneAPI Deep Neural Network Library (oneDNN) Developer Guide and Reference
=========================================================================

oneAPI Deep Neural Network Library (oneDNN) is an open-source cross-platform
performance library of basic building blocks for deep learning applications.

The library is optimized for Intel(R) Architecture Processors, Intel Graphics,
and Arm(R) 64-bit Architecture (AArch64)-based processors. oneDNN has experimental
support for the following architectures: NVIDIA* GPU, AMD* GPU,
OpenPOWER* Power ISA (PPC64), IBMz* (s390x), and RISC-V.

oneDNN is intended for deep learning applications and framework developers
interested in improving application performance on CPUs and GPUs.

.. toctree::
   :caption: About
   :hidden:
   :maxdepth: 1

   Introduction<self>

.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 1

   dev_guide_system_requirements
   dev_guide_build
   dev_guide_build_options
   dev_guide_link
   Basic Workflow<page_getting_started_cpp>
   dev_guide_examples

.. toctree::
   :caption: Developer Guide
   :hidden:
   :maxdepth: 2

   common_concepts
   functional_api
   graph_extension
   ukernels
   performance_profiling_and_inspection
   advanced_topics

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   C++ API<group_dnnl_api_cpp>
   C API<group_dnnl_api_c>
