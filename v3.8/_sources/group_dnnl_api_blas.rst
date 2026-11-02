.. index:: pair: group; BLAS functions
.. _doxid-group__dnnl__api__blas:

BLAS functions
==============

.. toctree::
	:hidden:

Overview
~~~~~~~~

A subset of Basic Linear Algebra (BLAS) functions that perform matrix-matrix multiplication. :ref:`More...<details-group__dnnl__api__blas>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::sgemm<doxid-group__dnnl__api__blas_1gace5cc61273dc46ccd9c08eee76d4057b>`(
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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::gemm_u8s8s32<doxid-group__dnnl__api__blas_1ga454c26361de7d3a29f6e23c641380fb0>`(
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
		const int32_t* co
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`dnnl::gemm_s8s8s32<doxid-group__dnnl__api__blas_1ga6bb7da88545097f097bbcd5778787826>`(
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
		const int32_t* co
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sgemm<doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8>`(
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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gemm_u8s8s32<doxid-group__dnnl__api__blas_1gaef24848fd198d8a178d3ad95a78c1767>`(
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
		const int32_t* co
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_gemm_s8s8s32<doxid-group__dnnl__api__blas_1ga2b763b7629846913507d88fba875cc26>`(
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
		const int32_t* co
		);

.. _details-group__dnnl__api__blas:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A subset of Basic Linear Algebra (BLAS) functions that perform matrix-matrix multiplication.

Global Functions
----------------

.. index:: pair: function; sgemm
.. _doxid-group__dnnl__api__blas_1gace5cc61273dc46ccd9c08eee76d4057b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::sgemm(
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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; gemm_u8s8s32
.. _doxid-group__dnnl__api__blas_1ga454c26361de7d3a29f6e23c641380fb0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::gemm_u8s8s32(
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
		const int32_t* co
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; gemm_s8s8s32
.. _doxid-group__dnnl__api__blas_1ga6bb7da88545097f097bbcd5778787826:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` dnnl::gemm_s8s8s32(
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
		const int32_t* co
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sgemm
.. _doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sgemm(
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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_gemm_u8s8s32
.. _doxid-group__dnnl__api__blas_1gaef24848fd198d8a178d3ad95a78c1767:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_gemm_u8s8s32(
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
		const int32_t* co
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_gemm_s8s8s32
.. _doxid-group__dnnl__api__blas_1ga2b763b7629846913507d88fba875cc26:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_gemm_s8s8s32(
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
		const int32_t* co
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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success and a status describing the error otherwise.

