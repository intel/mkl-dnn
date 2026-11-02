.. index:: pair: group; Threadpool interoperability API
.. _doxid-group__dnnl__api__threadpool__interop:

Threadpool interoperability API
===============================

.. toctree::
	:hidden:

	namespace_dnnl_threadpool_interop.rst

Overview
~~~~~~~~

API extensions to interact with the underlying Threadpool run-time. :ref:`More...<details-group__dnnl__api__threadpool__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::threadpool_interop<doxid-namespacednnl_1_1threadpool__interop>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_stream_create<doxid-group__dnnl__api__threadpool__interop_1ga45a92b2adda6ff7a31784d73cbd61c26>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_stream_get_threadpool<doxid-group__dnnl__api__threadpool__interop_1ga1dac9e0e17855a5196cb96295277a86d>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` astream,
		void** threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_set_max_concurrency<doxid-group__dnnl__api__threadpool__interop_1ga39b66c510d4f46fde238fedb4343fa2d>`(int max_concurrency);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_get_max_concurrency<doxid-group__dnnl__api__threadpool__interop_1ga4017c3cc0fe9b66a8d50352015d0dbcf>`(int* max_concurrency);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_sgemm<doxid-group__dnnl__api__threadpool__interop_1ga2726272c8ce83f4231cc81e326336193>`(
		char transa,
		char transb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const float* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		const float* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		float beta,
		float* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_gemm_u8s8s32<doxid-group__dnnl__api__threadpool__interop_1gaeb14ed904eaed73cfd01a29bc2a4ac1e>`(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const uint8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		uint8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co,
		void* threadpool
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_threadpool_interop_gemm_s8s8s32<doxid-group__dnnl__api__threadpool__interop_1ga39dd6bf602ca1ebb2039eb4c27f07fdf>`(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const int8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		int8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co,
		void* threadpool
		);

.. _details-group__dnnl__api__threadpool__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

API extensions to interact with the underlying Threadpool run-time.

Global Functions
----------------

.. index:: pair: function; dnnl_threadpool_interop_stream_create
.. _doxid-group__dnnl__api__threadpool__interop_1ga45a92b2adda6ff7a31784d73cbd61c26:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_stream_create(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* threadpool
		)

Creates an execution stream with specified threadpool.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Output execution stream.

	*
		- engine

		- Engine to create the execution stream on.

	*
		- threadpool

		- Pointer to an instance of a C++ class that implements dnnl::threapdool_iface interface.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`Using oneDNN with Threadpool-Based Threading <doxid-dev_guide_threadpool>`

.. index:: pair: function; dnnl_threadpool_interop_stream_get_threadpool
.. _doxid-group__dnnl__api__threadpool__interop_1ga1dac9e0e17855a5196cb96295277a86d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_stream_get_threadpool(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` astream,
		void** threadpool
		)

Returns a threadpool to be used by the execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Execution stream.

	*
		- threadpool

		- Output pointer to an instance of a C++ class that implements dnnl::threapdool_iface interface. Set to NULL if the stream was created without threadpool.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`Using oneDNN with Threadpool-Based Threading <doxid-dev_guide_threadpool>`

.. index:: pair: function; dnnl_threadpool_interop_set_max_concurrency
.. _doxid-group__dnnl__api__threadpool__interop_1ga39b66c510d4f46fde238fedb4343fa2d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_set_max_concurrency(int max_concurrency)

Sets the maximum concurrency assumed by oneDNN when outside a parallel call.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- max_concurrency

		- The maximum concurrency assumed by oneDNN when outside a parallel call. This is a threadlocal setting.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_threadpool_interop_get_max_concurrency
.. _doxid-group__dnnl__api__threadpool__interop_1ga4017c3cc0fe9b66a8d50352015d0dbcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_get_max_concurrency(int* max_concurrency)

Gets the maximum concurrency assumed by oneDNN when outside a parallel call.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- max_concurrency

		- The maximum concurrency assumed by oneDNN when outside a parallel call. This is a threadlocal setting.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_threadpool_interop_sgemm
.. _doxid-group__dnnl__api__threadpool__interop_1ga2726272c8ce83f4231cc81e326336193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_sgemm(
		char transa,
		char transb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const float* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		const float* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		float beta,
		float* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		void* threadpool
		)

Performs single-precision matrix-matrix multiply.

The operation is defined as:

``C := alpha * op( A ) * op( B ) + beta * C``

where

* ``op( X ) = X`` or ``op( X ) = X**T``,

* ``alpha`` and ``beta`` are scalars, and

* ``A``, ``B``, and ``C`` are matrices:
  
  * ``op( A )`` is an ``MxK`` matrix,
  
  * ``op( B )`` is an ``KxN`` matrix,
  
  * ``C`` is an ``MxN`` matrix.

The matrices are assumed to be stored in row-major order (the elements in each of the matrix rows are contiguous in memory).

.. note:: 

   This API does not support XERBLA. Instead, unlike the standard BLAS functions, this one returns a dnnl_status_t value to allow error handling.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transa

		- Transposition flag for matrix A: 'N' or 'n' means A is not transposed, and 'T' or 't' means that A is transposed.

	*
		- transb

		- Transposition flag for matrix B: 'N' or 'n' means B is not transposed, and 'T' or 't' means that B is transposed.

	*
		- M

		- The M dimension.

	*
		- N

		- The N dimension.

	*
		- K

		- The K dimension.

	*
		- alpha

		- The alpha parameter that is used to scale the product of matrices A and B.

	*
		- A

		- A pointer to the A matrix data.

	*
		- lda

		- The leading dimension for the matrix A.

	*
		- B

		- A pointer to the B matrix data.

	*
		- ldb

		- The leading dimension for the matrix B.

	*
		- beta

		- The beta parameter that is used to scale the matrix C.

	*
		- C

		- A pointer to the C matrix data.

	*
		- ldc

		- The leading dimension for the matrix C.

	*
		- threadpool

		- A pointer to a threadpool interface (only when built with the THREADPOOL CPU runtime).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_threadpool_interop_gemm_u8s8s32
.. _doxid-group__dnnl__api__threadpool__interop_1gaeb14ed904eaed73cfd01a29bc2a4ac1e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_gemm_u8s8s32(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const uint8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		uint8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co,
		void* threadpool
		)

Performs integer matrix-matrix multiply on 8-bit unsigned matrix A, 8-bit signed matrix B, and 32-bit signed resulting matrix C.

The operation is defined as:

``C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset``

where

* ``op( X ) = X`` or ``op( X ) = X**T``,

* ``alpha`` and ``beta`` are scalars, and

* ``A``, ``B``, and ``C`` are matrices:
  
  * ``op( A )`` is an ``MxK`` matrix,
  
  * ``op( B )`` is an ``KxN`` matrix,
  
  * ``C`` is an ``MxN`` matrix.

* ``A_offset`` is an ``MxK`` matrix with every element equal the ``ao`` value,

* ``B_offset`` is an ``KxN`` matrix with every element equal the ``bo`` value,

* ``C_offset`` is an ``MxN`` matrix which is defined by the ``co`` array of size ``len`` :
  
  * if ``offsetc = F`` : the ``len`` must be at least ``1``,
  
  * if ``offsetc = C`` : the ``len`` must be at least ``max(1, m)``,
  
  * if ``offsetc = R`` : the ``len`` must be at least ``max(1, n)``,

The matrices are assumed to be stored in row-major order (the elements in each of the matrix rows are contiguous in memory).

.. note:: 

   This API does not support XERBLA. Instead, unlike the standard BLAS functions, this one returns a dnnl_status_t value to allow error handling.
   
   

.. warning:: 

   On some architectures saturation may happen during intermediate computations, which would lead to unexpected results. For more details, refer to :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transa

		- Transposition flag for matrix A: 'N' or 'n' means A is not transposed, and 'T' or 't' means that A is transposed.

	*
		- transb

		- Transposition flag for matrix B: 'N' or 'n' means B is not transposed, and 'T' or 't' means that B is transposed.

	*
		- offsetc

		- 
		  Flag specifying how offsets should be applied to matrix C:
		  
		  * 'F' means that the same offset will be applied to each element of the matrix C,
		  
		  * 'C' means that individual offset will be applied to each element within each column,
		  
		  * 'R' means that individual offset will be applied to each element within each row.

	*
		- M

		- The M dimension.

	*
		- N

		- The N dimension.

	*
		- K

		- The K dimension.

	*
		- alpha

		- The alpha parameter that is used to scale the product of matrices A and B.

	*
		- A

		- A pointer to the A matrix data.

	*
		- lda

		- The leading dimension for the matrix A.

	*
		- ao

		- The offset value for the matrix A.

	*
		- B

		- A pointer to the B matrix data.

	*
		- ldb

		- The leading dimension for the matrix B.

	*
		- bo

		- The offset value for the matrix B.

	*
		- beta

		- The beta parameter that is used to scale the matrix C.

	*
		- C

		- A pointer to the C matrix data.

	*
		- ldc

		- The leading dimension for the matrix C.

	*
		- co

		- An array of offset values for the matrix C. The number of elements in the array depends on the value of ``offsetc``.

	*
		- threadpool

		- A pointer to a threadpool interface (only when built with the THREADPOOL CPU runtime).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_threadpool_interop_gemm_s8s8s32
.. _doxid-group__dnnl__api__threadpool__interop_1ga39dd6bf602ca1ebb2039eb4c27f07fdf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_threadpool_interop_gemm_s8s8s32(
		char transa,
		char transb,
		char offsetc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		float alpha,
		const int8_t* A,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		int8_t ao,
		const int8_t* B,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		int8_t bo,
		float beta,
		int32_t* C,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		const int32_t* co,
		void* threadpool
		)

Performs integer matrix-matrix multiply on 8-bit signed matrix A, 8-bit signed matrix B, and 32-bit signed resulting matrix C.

The operation is defined as:

``C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset``

where

* ``op( X ) = X`` or ``op( X ) = X**T``,

* ``alpha`` and ``beta`` are scalars, and

* ``A``, ``B``, and ``C`` are matrices:
  
  * ``op( A )`` is an ``MxK`` matrix,
  
  * ``op( B )`` is an ``KxN`` matrix,
  
  * ``C`` is an ``MxN`` matrix.

* ``A_offset`` is an ``MxK`` matrix with every element equal the ``ao`` value,

* ``B_offset`` is an ``KxN`` matrix with every element equal the ``bo`` value,

* ``C_offset`` is an ``MxN`` matrix which is defined by the ``co`` array of size ``len`` :
  
  * if ``offsetc = F`` : the ``len`` must be at least ``1``,
  
  * if ``offsetc = C`` : the ``len`` must be at least ``max(1, m)``,
  
  * if ``offsetc = R`` : the ``len`` must be at least ``max(1, n)``,

The matrices are assumed to be stored in row-major order (the elements in each of the matrix rows are contiguous in memory).

.. note:: 

   This API does not support XERBLA. Instead, unlike the standard BLAS functions, this one returns a dnnl_status_t value to allow error handling.
   
   

.. warning:: 

   On some architectures saturation may happen during intermediate computations, which would lead to unexpected results. For more details, refer to :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transa

		- Transposition flag for matrix A: 'N' or 'n' means A is not transposed, and 'T' or 't' means that A is transposed.

	*
		- transb

		- Transposition flag for matrix B: 'N' or 'n' means B is not transposed, and 'T' or 't' means that B is transposed.

	*
		- offsetc

		- 
		  Flag specifying how offsets should be applied to matrix C:
		  
		  * 'F' means that the same offset will be applied to each element of the matrix C,
		  
		  * 'C' means that individual offset will be applied to each element within each column,
		  
		  * 'R' means that individual offset will be applied to each element within each row.

	*
		- M

		- The M dimension.

	*
		- N

		- The N dimension.

	*
		- K

		- The K dimension.

	*
		- alpha

		- The alpha parameter that is used to scale the product of matrices A and B.

	*
		- A

		- A pointer to the A matrix data.

	*
		- lda

		- The leading dimension for the matrix A.

	*
		- ao

		- The offset value for the matrix A.

	*
		- B

		- A pointer to the B matrix data.

	*
		- ldb

		- The leading dimension for the matrix B.

	*
		- bo

		- The offset value for the matrix B.

	*
		- beta

		- The beta parameter that is used to scale the matrix C.

	*
		- C

		- A pointer to the C matrix data.

	*
		- ldc

		- The leading dimension for the matrix C.

	*
		- co

		- An array of offset values for the matrix C. The number of elements in the array depends on the value of ``offsetc``.

	*
		- threadpool

		- A pointer to a threadpool interface (only when built with the THREADPOOL CPU runtime).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

