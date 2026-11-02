.. index:: pair: namespace; dnnl::threadpool_interop
.. _doxid-namespacednnl_1_1threadpool__interop:

namespace dnnl::threadpool_interop
==================================

.. toctree::
	:hidden:

	struct_dnnl_threadpool_interop_threadpool_iface.rst

Overview
~~~~~~~~

Threadpool interoperability namespace. :ref:`More...<details-namespacednnl_1_1threadpool__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace threadpool_interop {

	// structs

	struct :ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`;

	// global functions

	:ref:`dnnl::stream<doxid-structdnnl_1_1stream>` :ref:`make_stream<doxid-namespacednnl_1_1threadpool__interop_1aaa7ec7af54363d81f8d4bcd2ff4af80d>`(
		const :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
		);

	:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* :ref:`get_threadpool<doxid-namespacednnl_1_1threadpool__interop_1abf6eacb2add7d269b3c6f4d027874969>`(const :ref:`dnnl::stream<doxid-structdnnl_1_1stream>`& astream);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`sgemm<doxid-namespacednnl_1_1threadpool__interop_1a84e972ec131a40b1fd63cf4ff435a047>`(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`gemm_u8s8s32<doxid-namespacednnl_1_1threadpool__interop_1adef03be3e852be9305c9654d07aa08d2>`(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
		);

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` :ref:`gemm_s8s8s32<doxid-namespacednnl_1_1threadpool__interop_1a1350fd057c475eae49424cfc7487e683>`(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
		);

	} // namespace threadpool_interop
.. _details-namespacednnl_1_1threadpool__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Threadpool interoperability namespace.

Global Functions
----------------

.. index:: pair: function; make_stream
.. _doxid-namespacednnl_1_1threadpool__interop_1aaa7ec7af54363d81f8d4bcd2ff4af80d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl::stream<doxid-structdnnl_1_1stream>` make_stream(
		const :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
		)

Constructs an execution stream for the specified engine and threadpool.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to create the stream on.

	*
		- threadpool

		- Pointer to an instance of a C++ class that implements dnnl::threapdool_iface interface.



.. rubric:: Returns:

An execution stream.



.. rubric:: See also:

:ref:`Using oneDNN with Threadpool-Based Threading <doxid-dev_guide_threadpool>`

.. index:: pair: function; get_threadpool
.. _doxid-namespacednnl_1_1threadpool__interop_1abf6eacb2add7d269b3c6f4d027874969:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* get_threadpool(const :ref:`dnnl::stream<doxid-structdnnl_1_1stream>`& astream)

Returns the pointer to a threadpool that is used by an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- An execution stream.



.. rubric:: Returns:

Output pointer to an instance of a C++ class that implements dnnl::threapdool_iface interface or NULL if the stream was created without threadpool.



.. rubric:: See also:

:ref:`Using oneDNN with Threadpool-Based Threading <doxid-dev_guide_threadpool>`

.. index:: pair: function; sgemm
.. _doxid-namespacednnl_1_1threadpool__interop_1a84e972ec131a40b1fd63cf4ff435a047:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` sgemm(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
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

.. index:: pair: function; gemm_u8s8s32
.. _doxid-namespacednnl_1_1threadpool__interop_1adef03be3e852be9305c9654d07aa08d2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` gemm_u8s8s32(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
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

.. index:: pair: function; gemm_s8s8s32
.. _doxid-namespacednnl_1_1threadpool__interop_1a1350fd057c475eae49424cfc7487e683:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>` gemm_s8s8s32(
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
		:ref:`threadpool_iface<doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>`* threadpool
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

