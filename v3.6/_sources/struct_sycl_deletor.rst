.. index:: pair: struct; sycl_deletor
.. _doxid-structsycl__deletor:

struct sycl_deletor
===================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	struct sycl_deletor
	{
		// fields
	
		::sycl::context :target:`ctx_<doxid-structsycl__deletor_1ae8e498519b1968fb1782b5f0fae399bc>`;

		// construction
	
		:target:`sycl_deletor<doxid-structsycl__deletor_1a66ce0382c58c0759d2192f4d06b3c2e4>`();
		:target:`sycl_deletor<doxid-structsycl__deletor_1a732683a3d1464bf3196bd249d3601f3b>`(const ::sycl::context& ctx);

		// methods
	
		void :target:`operator ()<doxid-structsycl__deletor_1a4368dec0c431be517369b4372adb6216>` (void* ptr);
	};
