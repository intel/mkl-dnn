.. index:: pair: page; CPU ISA Hints
.. _doxid-dev_guide_cpu_isa_hints:

CPU ISA Hints
=============

For performance reasons, extra hints may be provided to oneDNN which enable the just-in-time (JIT) code generation to prefer or avoid certain CPU ISA features.

Build-time Controls
~~~~~~~~~~~~~~~~~~~

At build-time, support for this feature is controlled via cmake option ``ONEDNN_ENABLE_CPU_ISA_HINTS``.

============================  ====================================  ======================  
CMake Option                  Supported values (defaults in bold)   Description             
============================  ====================================  ======================  
ONEDNN_ENABLE_CPU_ISA_HINTS   **ON** , OFF                          Enables CPU ISA hints   
============================  ====================================  ======================

This results in making oneDNN aware of ONEDNN_CPU_ISA_HINTS environment variable and corresponding setter utility namely, :ref:`dnnl::set_cpu_isa_hints <doxid-group__dnnl__api__service_1ga29aa5fb708d803e091ac61dc67f9e6ed>` routine.

Run-time Controls
~~~~~~~~~~~~~~~~~

During run-time the ONEDNN_CPU_ISA_HINTS environment variable can be used to specify ISA specific hint to enable oneDNN to dispatch appropriate kernel.

=====================  ===========  ==================================================  
Environment variable   Value        Description                                         
=====================  ===========  ==================================================  
ONEDNN_CPU_ISA_HINTS   NO_HINTS     Use default configuration for ISA                   
                       PREFER_YMM   Prefer to use YMM registers for vector operations   
=====================  ===========  ==================================================

This feature can also be managed at run-time with the following functions:

* :ref:`dnnl::set_cpu_isa_hints <doxid-group__dnnl__api__service_1ga29aa5fb708d803e091ac61dc67f9e6ed>` function allows changing the CPU ISA hint at run-time. The limitation is that it is possible to set the value only once. In addition, it is advised to call this function before any other oneDNN API. This is because the first internal query of CPU ISA hints will disable the ability to change it. Once disabled, changing the CPU ISA hint will return an error.

* :ref:`dnnl::get_cpu_isa_hints <doxid-group__dnnl__api__service_1ga8bee13aa79a9711489b401e9c4252ff2>` function returns the currently used CPU ISA hints and by default poses no restrictions.

Function settings take precedence over environment variables. Moreover, if the hint is not applicable then it would be silently ignored. For instance with SSE41 there are no YMM registers and so hint to prefer YMM registers would be silently bypassed.

