.. index:: pair: group; Service
.. _doxid-group__dnnl__api__service:

Service
=======

.. toctree::
	:hidden:

	enum_dnnl_cpu_isa.rst
	enum_dnnl_cpu_isa_hints.rst
	enum_dnnl_cpu_isa_hints_t.rst
	enum_dnnl_cpu_isa_t.rst
	enum_dnnl_status.rst
	struct_dnnl_version_t.rst

Overview
~~~~~~~~

A set of functions that aid in oneDNN debugging and profiling. :ref:`More...<details-group__dnnl__api__service>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef :ref:`dnnl_version_t<doxid-structdnnl__version__t>` :ref:`dnnl::version_t<doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af>`;

	// enums

	enum :ref:`dnnl::cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>`;
	enum :ref:`dnnl::cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>`;
	enum :ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>`;
	enum :ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>`;
	enum :ref:`dnnl::status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>`;

	// structs

	struct :ref:`dnnl_version_t<doxid-structdnnl__version__t>`;

	// global functions

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_verbose<doxid-group__dnnl__api__service_1ga37bcab6f832df551a9fc418e48743b15>`(int level);
	const :ref:`version_t<doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af>`* :ref:`dnnl::version<doxid-group__dnnl__api__service_1gaad8292408620d0296f22bdf65afb752d>`();
	:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` :ref:`dnnl::get_default_fpmath_mode<doxid-group__dnnl__api__service_1ga782a2388fc46e80deac409110886db75>`();
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_default_fpmath_mode<doxid-group__dnnl__api__service_1ga0d55da5f92d60a7324cfdc97004ad975>`(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_jit_dump<doxid-group__dnnl__api__service_1ga2344639528a341878d2ce46fe1c1ac83>`(int enable);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_jit_profiling_flags<doxid-group__dnnl__api__service_1ga966c54ccb1d9ff33d20c4ea47e34675d>`(unsigned flags);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_jit_profiling_jitdumpdir<doxid-group__dnnl__api__service_1ga533341aaf1402e27d1225d1a59819a62>`(const std::string& dir);
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_max_cpu_isa<doxid-group__dnnl__api__service_1ga08734310b5f1ca794c64b6a5b944b698>`(:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` isa);
	:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` :ref:`dnnl::get_effective_cpu_isa<doxid-group__dnnl__api__service_1ga3953f71c3f0126d9cc005a1ceff65e8b>`();
	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::set_cpu_isa_hints<doxid-group__dnnl__api__service_1ga29aa5fb708d803e091ac61dc67f9e6ed>`(:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` isa_hints);
	:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` :ref:`dnnl::get_cpu_isa_hints<doxid-group__dnnl__api__service_1ga8bee13aa79a9711489b401e9c4252ff2>`();
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_dump<doxid-group__dnnl__api__service_1ga03c8f4af3d01f76060f98e78039837fc>`(int enable);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_flags<doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48>`(unsigned flags);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_jit_profiling_jitdumpdir<doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7>`(const char* dir);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_max_cpu_isa<doxid-group__dnnl__api__service_1ga4b7f3b3299482f88f1a0aa61a4707156>`(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` isa);
	:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` DNNL_API :ref:`dnnl_get_effective_cpu_isa<doxid-group__dnnl__api__service_1gac55836cf36bc25f8635e459678303570>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_cpu_isa_hints<doxid-group__dnnl__api__service_1gad078a384ab0e078d81595686efd26ed2>`(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` isa_hints);
	:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` DNNL_API :ref:`dnnl_get_cpu_isa_hints<doxid-group__dnnl__api__service_1gad93f9f4bf3c9e12a2be7337b1e41d145>`(void);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_verbose<doxid-group__dnnl__api__service_1ga14cc3b56337322e1e5132c5ee0c84856>`(int level);
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

A set of functions that aid in oneDNN debugging and profiling.

Typedefs
--------

.. index:: pair: typedef; version_t
.. _doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`dnnl_version_t<doxid-structdnnl__version__t>` dnnl::version_t

Structure containing version information as per `Semantic Versioning <https://semver.org>`__

Global Functions
----------------

.. index:: pair: function; set_verbose
.. _doxid-group__dnnl__api__service_1ga37bcab6f832df551a9fc418e48743b15:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_verbose(int level)

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
		  
		  * 1: primitive information at execution,
		  
		  * 2: primitive information at creation and execution.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``level`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; version
.. _doxid-group__dnnl__api__service_1gaad8292408620d0296f22bdf65afb752d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const :ref:`version_t<doxid-group__dnnl__api__service_1ga7b6ec8722f5ad94170755b8be0cdd3af>`* dnnl::version()

Returns library version information.



.. rubric:: Returns:

Pointer to a constant structure containing

* major: major version number,

* minor: minor version number,

* patch: patch release number,

* hash: git commit hash.

.. index:: pair: function; get_default_fpmath_mode
.. _doxid-group__dnnl__api__service_1ga782a2388fc46e80deac409110886db75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` dnnl::get_default_fpmath_mode()

Returns the floating-point math mode that will be used by default for all subsequently created primitives.



.. rubric:: Returns:

Output FP math mode.

.. index:: pair: function; set_default_fpmath_mode
.. _doxid-group__dnnl__api__service_1ga0d55da5f92d60a7324cfdc97004ad975:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_default_fpmath_mode(:ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode)

Sets the floating-point math mode that will be used by default for all subsequently created primitives.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mode

		- FP math mode. The possible values are: :ref:`dnnl_fpmath_mode_strict <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ab062cd5c71803f26ab700073c8f18bd3>`, :ref:`dnnl_fpmath_mode_bf16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ac7e140804cd26325c9c5563fa421b7f7>`, :ref:`dnnl_fpmath_mode_f16 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912aa128d95a43cba562c8b90cd820d3faaf>`, :ref:`dnnl_fpmath_mode_tf32 <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912a7c89cac55a7b6a6e4692a5805ba10edc>`, :ref:`dnnl_fpmath_mode_any <doxid-group__dnnl__api__fpmath__mode_1gga62f956692c5a70353f164e09ff524912ad54e0a51f937a49dd4c2c3d50ca1b94c>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; set_jit_dump
.. _doxid-group__dnnl__api__service_1ga2344639528a341878d2ce46fe1c1ac83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_jit_dump(int enable)

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

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flag`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; set_jit_profiling_flags
.. _doxid-group__dnnl__api__service_1ga966c54ccb1d9ff33d20c4ea47e34675d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_jit_profiling_flags(unsigned flags)

Sets library profiling flags.

The flags define which profilers are supported.

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

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flags`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; set_jit_profiling_jitdumpdir
.. _doxid-group__dnnl__api__service_1ga533341aaf1402e27d1225d1a59819a62:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_jit_profiling_jitdumpdir(const std::string& dir)

Sets JIT dump output path.

Only applicable to Linux and is only used when profiling flags have DNNL_JIT_PROFILE_LINUX_PERF bit set.

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

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` if the output directory was set correctly and an error status otherwise.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` on Windows.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; set_max_cpu_isa
.. _doxid-group__dnnl__api__service_1ga08734310b5f1ca794c64b6a5b944b698:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_max_cpu_isa(:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` isa)

Sets the maximal ISA the library can dispatch to on the CPU.

See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_MAX_CPU_ISA environment variable. The environment variable can be set to the desired maximal ISA name in upper case and with dnnl_cpu_isa prefix removed. For example: ``DNNL_MAX_CPU_ISA=AVX2``.

.. note:: 

   The ISAs are only partially ordered:
   
   * SSE41 < AVX < AVX2 < AVX2_VNNI < AVX2_VNNI_2,
   
   * AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16 < AVX512_CORE_FP16 < AVX512_CORE_AMX < AVX512_CORE_AMX_FP16,
   
   * AVX2_VNNI < AVX512_CORE_FP16.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa

		- Maximal ISA the library should dispatch to. Pass :ref:`dnnl_cpu_isa_default <doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a334f526a8651da897123990b8c919928>` / :ref:`dnnl::cpu_isa::isa_default <doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a56a0edddbeaaf449d233434fb1860724>` to remove ISA restrictions (except for ISAs with initial support in the library).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``isa`` parameter is invalid or the ISA cannot be changed at this time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; get_effective_cpu_isa
.. _doxid-group__dnnl__api__service_1ga3953f71c3f0126d9cc005a1ceff65e8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`cpu_isa<doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` dnnl::get_effective_cpu_isa()

Gets the maximal ISA the library can dispatch to on the CPU.

See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values returned by the C and C++ API functions respectively.



.. rubric:: Returns:

:ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` value reflecting the maximal ISA the library may dispatch to.



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; set_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1ga29aa5fb708d803e091ac61dc67f9e6ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::set_cpu_isa_hints(:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` isa_hints)

Sets the hints flag for the CPU ISA.

See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_CPU_ISA_HINTS environment variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa_hints

		- CPU ISA hints to be passed over to the implementation. Pass :ref:`dnnl_cpu_isa_no_hints <doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5da9e598ac27ce94827b20cab264d623da4>` / :ref:`dnnl::cpu_isa_hints::no_hints <doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4a5c2d3f6f845dca6d90d7a1c445644c99>` to use default features i.e. no hints.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_runtime_error <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa38efb4adabcae7c9e6479e8ee1242b9b>` / :ref:`dnnl::status::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>` if the ISA hints cannot be specified at the current time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU ISA Hints <doxid-dev_guide_cpu_isa_hints>` for more details

.. index:: pair: function; get_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1ga8bee13aa79a9711489b401e9c4252ff2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`cpu_isa_hints<doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` dnnl::get_cpu_isa_hints()

Gets the ISA specific hints that library can follow.

See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values returned by the C and C++ API functions respectively.



.. rubric:: Returns:

:ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` value reflecting the ISA specific hints the library can follow.



.. rubric:: See also:

:ref:`CPU ISA Hints <doxid-dev_guide_cpu_isa_hints>` for more details

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

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flag`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_set_jit_profiling_flags
.. _doxid-group__dnnl__api__service_1ga51ef634e4f201a12d32e573955943f48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_jit_profiling_flags(unsigned flags)

Sets library profiling flags.

The flags define which profilers are supported.

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

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``flags`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; dnnl_set_jit_profiling_jitdumpdir
.. _doxid-group__dnnl__api__service_1gafb0fb0d37d72bc58386ba97bb858f8f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_jit_profiling_jitdumpdir(const char* dir)

Sets JIT dump output path.

Only applicable to Linux and is only used when profiling flags have DNNL_JIT_PROFILE_LINUX_PERF bit set.

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

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` if the output directory was set correctly and an error status otherwise.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` on Windows.



.. rubric:: See also:

:ref:`Profiling oneDNN Performance <doxid-dev_guide_profilers>`

.. index:: pair: function; dnnl_set_max_cpu_isa
.. _doxid-group__dnnl__api__service_1ga4b7f3b3299482f88f1a0aa61a4707156:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_max_cpu_isa(:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` isa)

Sets the maximal ISA the library can dispatch to on the CPU.

See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_MAX_CPU_ISA environment variable. The environment variable can be set to the desired maximal ISA name in upper case and with dnnl_cpu_isa prefix removed. For example: ``DNNL_MAX_CPU_ISA=AVX2``.

.. note:: 

   The ISAs are only partially ordered:
   
   * SSE41 < AVX < AVX2 < AVX2_VNNI < AVX2_VNNI_2,
   
   * AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16 < AVX512_CORE_FP16 < AVX512_CORE_AMX < AVX512_CORE_AMX_FP16,
   
   * AVX2_VNNI < AVX512_CORE_FP16.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa

		- Maximal ISA the library should dispatch to. Pass :ref:`dnnl_cpu_isa_default <doxid-group__dnnl__api__service_1gga303bab5d2e7b371bb44495864df21dd2a334f526a8651da897123990b8c919928>` / :ref:`dnnl::cpu_isa::isa_default <doxid-group__dnnl__api__service_1ggabad017feb1850634bf3babdb68234f83a56a0edddbeaaf449d233434fb1860724>` to remove ISA restrictions (except for ISAs with initial support in the library).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``isa`` parameter is invalid or the ISA cannot be changed at this time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; dnnl_get_effective_cpu_isa
.. _doxid-group__dnnl__api__service_1gac55836cf36bc25f8635e459678303570:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_cpu_isa_t<doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` DNNL_API dnnl_get_effective_cpu_isa(void)

Gets the maximal ISA the library can dispatch to on the CPU.

See :ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` and :ref:`dnnl::cpu_isa <doxid-group__dnnl__api__service_1gabad017feb1850634bf3babdb68234f83>` for the list of the values returned by the C and C++ API functions respectively.



.. rubric:: Returns:

:ref:`dnnl_cpu_isa_t <doxid-group__dnnl__api__service_1ga303bab5d2e7b371bb44495864df21dd2>` value reflecting the maximal ISA the library may dispatch to.



.. rubric:: See also:

:ref:`CPU Dispatcher Control <doxid-dev_guide_cpu_dispatcher_control>` for more details

.. index:: pair: function; dnnl_set_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1gad078a384ab0e078d81595686efd26ed2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_cpu_isa_hints(:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` isa_hints)

Sets the hints flag for the CPU ISA.

See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values accepted by the C and C++ API functions respectively.

This function has effect only once, and returns an error on subsequent calls. It should also be invoked before any other oneDNN API call, otherwise it may return an error.

This function overrides the DNNL_CPU_ISA_HINTS environment variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- isa_hints

		- CPU ISA hints to be passed over to the implementation. Pass :ref:`dnnl_cpu_isa_no_hints <doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5da9e598ac27ce94827b20cab264d623da4>` / :ref:`dnnl::cpu_isa_hints::no_hints <doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4a5c2d3f6f845dca6d90d7a1c445644c99>` to use default features i.e. no hints.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a :ref:`dnnl_runtime_error <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa38efb4adabcae7c9e6479e8ee1242b9b>` / :ref:`dnnl::status::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>` if the ISA hints cannot be specified at the current time.

:ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` / :ref:`dnnl::status::unimplemented <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>` if the feature was disabled at build time (see :ref:`Build Options <doxid-dev_guide_build_options>` for more details).



.. rubric:: See also:

:ref:`CPU ISA Hints <doxid-dev_guide_cpu_isa_hints>` for more details

.. index:: pair: function; dnnl_get_cpu_isa_hints
.. _doxid-group__dnnl__api__service_1gad93f9f4bf3c9e12a2be7337b1e41d145:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_cpu_isa_hints_t<doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` DNNL_API dnnl_get_cpu_isa_hints(void)

Gets the ISA specific hints that library can follow.

See :ref:`dnnl_cpu_isa_hints_t <doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>` and :ref:`dnnl::cpu_isa_hints <doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>` for the list of the values returned by the C and C++ API functions respectively.



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
		  
		  * 1: primitive information at execution,
		  
		  * 2: primitive information at creation and execution.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``level`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

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

Instruct Linux perf integration via jitdump files to use TSC.

:ref:`DNNL_JIT_PROFILE_LINUX_JITDUMP <doxid-group__dnnl__api__service_1ga5afb7d615d8507b8d5469553e6dde2a7>` must be set too for this to take effect.

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

