.. index:: pair: group; Service
.. _doxid-group__dnnl__api__service:

Service
=======

.. toctree::
	:hidden:

	enum_dnnl_cpu_isa_hints_t.rst
	enum_dnnl_cpu_isa_t.rst
	struct_dnnl_version_t.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>`;
	enum :ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>`;

	// structs

	struct :ref:`dnnl_version_t<doxid-structdnnl__version__t>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_dump<doxid-group__dnnl__api__service_1ga03c8f4af3d01f76060f98e78039837fc>`(int enable);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_flags<doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48>`(unsigned flags);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_jitdumpdir<doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7>`(const char* dir);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_max_cpu_isa<doxid-group__dnnl__api__service_1ga4b7f3b3299482f88f1a0aa61a4707156>`(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` isa);
	:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` DNNL_API :ref:`dnnl_get_effective_cpu_isa<doxid-group__dnnl__api__service_1gac55836cf36bc25f8635e459678303570>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_cpu_isa_hints<doxid-group__dnnl__api__service_1gad078a384ab0e078d81595686efd26ed2>`(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` isa_hints);
	:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` DNNL_API :ref:`dnnl_get_cpu_isa_hints<doxid-group__dnnl__api__service_1gad93f9f4bf3c9e12a2be7337b1e41d145>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_verbose<doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856>`(int level);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_verbose_profiling_enabled<doxid-group__dnnl__api__service_1ga50608f3f3888ebbb3d007a50a63e0e04>`(int* enabled);
	const :ref:`dnnl_version_t<doxid-structdnnl__version__t>` DNNL_API* :ref:`dnnl_version<doxid-group__dnnl__api__service_1ga73e40d184386e9d9ca917756e76fb232>`(void);

	// macros

	#define :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP<doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC<doxid-group__dnnl__api__service_1ga66a48a940ab2916d360b0bb677a70e5f>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_PERF<doxid-group__dnnl__api__service_1ga5a1d61af9d5b15dbc6d7d33f0f3e22bc>`
	#define :ref:`DNNL_JIT_PROFILE_LINUX_PERFMAP<doxid-group__dnnl__api__service_1gacb5b174589525cce34589ef4ef56462f>`
	#define :ref:`DNNL_JIT_PROFILE_NONE<doxid-group__dnnl__api__service_1ga7ceacd6430988ed4bf58f5b01cd9c5a4>`
	#define :ref:`DNNL_JIT_PROFILE_VTUNE<doxid-group__dnnl__api__service_1ga137013d98ef736973ebbe1ecd4a4b2c9>`

.. _details-group__dnnl__api__service:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; dnnl_set_jit_dump
.. _doxid-group__dnnl__api__service_1ga03c8f4af3d01f76060f98e78039837fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_jit_dump(int enable)

Configures dumping of JIT-generated code.

.. note:: 

   This setting overrides the DNNL_JIT_DUMP environment variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- enable

		- Flag value. Set to 0 to disable and set to 1 to enable.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flag`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_set_jit_profiling_flags
.. _doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_jit_profiling_flags(unsigned flags)

Sets library profiling flags. The flags define which profilers are supported.

.. note:: 

   This setting overrides DNNL_JIT_PROFILE environment variable.
   
   
Passing :ref:`DNNL_JIT_PROFILE_NONE <doxid-group__dnnl__api__service_1ga7ceacd6430988ed4bf58f5b01cd9c5a4>` disables profiling completely.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flags

		- 
		  Profiling flags that can contain the following bits:
		  
		  * :ref:`DNNL_JIT_PROFILE_VTUNE <doxid-group__dnnl__api__service_1ga137013d98ef736973ebbe1ecd4a4b2c9>` integration with VTune Profiler (on by default)
		  
		  * :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP <doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7>` produce Linux-specific jit-pid.dump output (off by default). The location of the output is controlled via JITDUMPDIR environment variable or via :ref:`dnnl_set_jit_profiling_jitdumpdir() <doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7>` function.
		  
		  * :ref:`DNNL_JIT_PROFILE_LINUX_PERFMAP <doxid-group__dnnl__api__service_1gacb5b174589525cce34589ef4ef56462f>` produce Linux-specific perf-pid.map output (off by default). The output is always placed into /tmp.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flags`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; dnnl_set_jit_profiling_jitdumpdir
.. _doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_jit_profiling_jitdumpdir(const char* dir)

Sets JIT dump output path. Only applicable to Linux and is only used when profiling flags have DNNL_JIT_PROFILE_LINUX_PERF bit set.

After the first JIT kernel is generated, the jitdump output will be placed into temporary directory created using the mkdtemp template 'dir/.debug/jit/dnnl.XXXXXX'.

.. note:: 

   This setting overrides JITDUMPDIR environment variable. If JITDUMPDIR is not set, and this function is never called, the path defaults to HOME. Passing NULL reverts the value to default.
   
   

.. note:: 

   The directory is accessed only when the first JIT kernel is being created. JIT profiling will be disabled in case of any errors accessing or creating this directory.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- dir

		- JIT dump output path.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` if the output directory was set correctly and an error status otherwise.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` on Windows.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; dnnl_set_max_cpu_isa
.. _doxid-group__dnnl__api__service_1ga4b7f3b3299482f88f1a0aa61a4707156:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_max_cpu_isa(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` isa)

Sets the maximal ISA the library can dispatch to on the CPU. See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__cpp__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_MAX_CPU_ISA environment variable. The environment variable can be set to the desired maximal ISA name in upper case and with dnnl_cpu_isa prefix removed. For example: ``DNNL_MAX_CPU_ISA=AVX2``.

.. note:: 

   The ISAs are only partially ordered:
   
   * AVX2 < AVX2_VNNI < AVX2_VNNI_2,
   
   * AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16 < AVX10_1_512 < AVX10_2,
   
   * AVX10_1_512 < AVX10_1_512_AMX < AVX10_1_512_AMX_FP16 < AVX10_2_AMX_2,
   
   * AVX2_VNNI < AVX10_1_512,
   
   * AVX10_2 < AVX10_2_AMX_2,
   
   * AVX10_2 < AVX10_2_ACE
   
   
Aliases:

* AVX512_CORE_FP16 = AVX10_1_512

* AVX512_CORE_AMX = AVX10_1_512_AMX

* AVX512_CORE_AMX_FP16 = AVX10_1_512_AMX_FP16

* AVX10_2_512 = AVX10_2

* AVX10_2_512_AMX_2 = AVX10_2_AMX_2



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa

		- Maximal ISA the library should dispatch to. Pass :ref:`dnnl_cpu_isa_default <doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a334f526a8651da897123990b8c919928>` / :ref:`dnnl::cpu_isa::isa_default <doxid-group__dnnl__api__cpp__service_1ggabad017feb1850634bf3babdb68234f83a56a0edddbeaaf449d233434fb1860724>` to remove ISA restrictions (except for ISAs with initial support in the library).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``isa`` parameter is invalid or the ISA cannot be changed at this time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; dnnl_get_effective_cpu_isa
.. _doxid-group__dnnl__api__service_1gac55836cf36bc25f8635e459678303570:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` DNNL_API dnnl_get_effective_cpu_isa(void)

Gets the maximal ISA the library can dispatch to on the CPU. See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__cpp__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values returned by the C and C++ API functions respectively.



.. rubric:: Returns:

:ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` value reflecting the maximal ISA the library may dispatch to.



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; dnnl_set_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1gad078a384ab0e078d81595686efd26ed2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_cpu_isa_hints(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` isa_hints)

Sets the hints flag for the CPU ISA. See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__cpp__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_CPU_ISA_HINTS environment variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa_hints

		- CPU ISA hints to be passed over to the implementation. Pass :ref:`dnnl_cpu_isa_no_hints <doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5da9e598ac27ce94827b20cab264d623da4>` / :ref:`dnnl::cpu_isa_hints::no_hints <doxid-group__dnnl__api__cpp__service_1ggaf574021058ebc6965da14fc4387dd0c4a5c2d3f6f845dca6d90d7a1c445644c99>` to use default features i.e. no hints.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_runtime_error <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa38efb4adabcae7c9e6479e8ee1242b9b>` / :ref:`dnnl::status::runtime_error <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>` if the ISA hints cannot be specified at the current time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU ISA Hints <doxid-dev_guide_cpu_isa_hints>` for more details

.. index:: pair: function; dnnl_get_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1gad93f9f4bf3c9e12a2be7337b1e41d145:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` DNNL_API dnnl_get_cpu_isa_hints(void)

Gets the ISA specific hints that library can follow. See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__cpp__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values returned by the C and C++ API functions respectively.



.. rubric:: Returns:

:ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` value reflecting the ISA specific hints the library can follow.



.. rubric:: See also:

:ref:`CPU ISA Hints <doxid-dev_guide_cpu_isa_hints>` for more details

.. index:: pair: function; dnnl_set_verbose
.. _doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_verbose(int level)

Configures verbose output to stdout.

.. note:: 

   Enabling verbose output affects performance. This setting overrides the ONEDNN_VERBOSE environment variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- level

		- 
		  Verbosity level:
		  
		  * 0: no verbose output (default),
		  
		  * 1: primitive and graph information at execution,
		  
		  * 2: primitive and graph information at creation/compilation and execution.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``level`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_verbose_profiling_enabled
.. _doxid-group__dnnl__api__service_1ga50608f3f3888ebbb3d007a50a63e0e04:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_verbose_profiling_enabled(int* enabled)

Checks if verbose profiling mode is enabled.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- enabled

		- Output flag set to 1 if any verbose profiling output is enabled and 0 otherwise.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``enabled`` pointer is NULL, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__cpp__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_version
.. _doxid-group__dnnl__api__service_1ga73e40d184386e9d9ca917756e76fb232:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const :ref:`dnnl_version_t<doxid-structdnnl__version__t>` DNNL_API* dnnl_version(void)

Returns library version information.



.. rubric:: Returns:

Pointer to a constant structure containing

* major: major version number,

* minor: minor version number,

* patch: patch release number,

* hash: git commit hash.

Macros
------

.. index:: pair: define; DNNL_JIT_PROFILE_LINUX_JITDUMP
.. _doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_LINUX_JITDUMP

Enable Linux perf integration via jitdump files.

.. index:: pair: define; DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC
.. _doxid-group__dnnl__api__service_1ga66a48a940ab2916d360b0bb677a70e5f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC

Instruct Linux perf integration via jitdump files to use TSC. :ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP <doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7>` must be set too for this to take effect.

.. index:: pair: define; DNNL_JIT_PROFILE_LINUX_PERF
.. _doxid-group__dnnl__api__service_1ga5a1d61af9d5b15dbc6d7d33f0f3e22bc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_LINUX_PERF

Enable Linux perf integration (both jitdump and perfmap)

.. index:: pair: define; DNNL_JIT_PROFILE_LINUX_PERFMAP
.. _doxid-group__dnnl__api__service_1gacb5b174589525cce34589ef4ef56462f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_LINUX_PERFMAP

Enable Linux perf integration via perfmap files.

.. index:: pair: define; DNNL_JIT_PROFILE_NONE
.. _doxid-group__dnnl__api__service_1ga7ceacd6430988ed4bf58f5b01cd9c5a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_NONE

Disable profiling completely.

.. index:: pair: define; DNNL_JIT_PROFILE_VTUNE
.. _doxid-group__dnnl__api__service_1ga137013d98ef736973ebbe1ecd4a4b2c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_JIT_PROFILE_VTUNE

Enable VTune Profiler integration.

