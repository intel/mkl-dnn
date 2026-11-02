.. index:: pair: struct; sycl_deletor_t
.. _doxid-structsycl__deletor__t:

struct sycl_deletor_t
=====================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	struct sycl_deletor_t
	{
		// fields
	
		::sycl::context :target:`ctx_<doxid-structsycl__deletor__t_1afdb5fc85a57c5600ba5e211a39e4395b>`;

		// construction
	
		:target:`sycl_deletor_t<doxid-structsycl__deletor__t_1a9613e102040535dbb323c589217236dc>`();
		:target:`sycl_deletor_t<doxid-structsycl__deletor__t_1a5d78355399fc98109f343c8379778855>`(const ::sycl::context& ctx);

		// methods
	
		void :target:`operator ()<doxid-structsycl__deletor__t_1a457c06ece08e381fa1c0f684a6f6d71a>` (void* ptr);
	};
