.. index:: pair: page; API
.. _doxid-dev_guide_c_and_cpp_apis:

API
===

oneDNN has both C and C++ APIs available to users for convenience. There is almost a one-to-one correspondence as far as features are concerned, so users can choose based on language preference and switch back and forth in their projects if they desire. Most of the users choose C++ API though.

The differences are shown in the table below.

=========================  ===================================================================================================  ==============================================================  
Features                   **C API**                                                                                            **C++ API**                                                     
=========================  ===================================================================================================  ==============================================================  
Minimal standard version   C99                                                                                                  C++11                                                           
Functional coverage        Full                                                                                                 May require use of the **C API**                                
Error handling             Functions return :ref:`status <doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>`   Functions throw :ref:`exceptions <doxid-structdnnl_1_1error>`   
Verbosity                  High                                                                                                 Medium                                                          
Implementation             Completely inside the library                                                                        Header-based thin wrapper around the **C API**                  
Purpose                    Provide simple API and stable ABI to the library                                                     Improve usability                                               
Target audience            Experienced users, FFI                                                                               Most of the users and framework developers                      
=========================  ===================================================================================================  ==============================================================

Input validation notes
~~~~~~~~~~~~~~~~~~~~~~

oneDNN performs limited input validation to minimize the performance overheads. The user application is responsible for sanitizing inputs passed to the library. Examples of the inputs that may result in unexpected consequences:

* Not-a-number (NaN) floating point values

* Large ``u8`` or ``s8`` inputs may lead to accumulator overflow

* While the ``bf16`` 16-bit floating point data type has range close to 32-bit floating point data type, there is a significant reduction in precision.

As oneDNN API accepts raw pointers as parameters it's the calling code responsibility to

* Allocate memory and validate the buffer sizes before passing them to the library

* Ensure that the data buffers do not overlap unless the functionality explicitly permits in-place computations

Memory Alignment Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On certain architectures, proper memory alignment is required to maximize efficiency and avoid runtime issues when using oneDNN primitives.

Intel(R) Architecture Processors
--------------------------------

No memory alignment requirements.

Intel(R) Processor Graphics and Xe Architecture graphics
--------------------------------------------------------

For Intel Processor graphics and Xe Architecture graphics, oneDNN requires a minimum memory alignment of 64 bytes, with 128 bytes recommended for optimal performance.

If your use case requires element-wise alignment, as a possible workaround, you can use the reference primitive implementation which provides functional coverage, without performance optimizations. To select the reference implementation, use :ref:`dnnl::primitive_desc::next_impl <doxid-structdnnl_1_1primitive__desc_1a841df469ca54c3de2d233e46f48322b2>` API:

.. ref-code-block:: cpp

	auto pd = ...
	while (!strstr(pd.impl_info_str(), "ref")) {
	    pd.next_impl();
	}
	// Now pd points to the reference implementation.

