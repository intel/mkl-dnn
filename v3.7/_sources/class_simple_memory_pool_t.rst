.. index:: pair: class; simple_memory_pool_t
.. _doxid-classsimple__memory__pool__t:

class simple_memory_pool_t
==========================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class simple_memory_pool_t
	{
	public:
		// fields
	
		void size_t :target:`alignment<doxid-classsimple__memory__pool__t_1a7980f1fc88c4402a9efee4282c0c0932>`;
		void size_t cl_device_id :target:`dev<doxid-classsimple__memory__pool__t_1a9136e5c1abf8996ac73fd4cd888fdae0>`;
		void size_t cl_device_id cl_context :target:`ctx<doxid-classsimple__memory__pool__t_1a7fb281b1ca5e6d182209d86c7dbe3bff>` {         std::lock_guard<std::mutex> pool_guard(pool_lock);
		void* :target:`ptr<doxid-classsimple__memory__pool__t_1ae6345b88f4589083f92c5c63146d12e2>` {nullptr};
		bool :target:`need_alloc_new_mm<doxid-classsimple__memory__pool__t_1ac291807a3089bfba555cb53e043d78d9>` = true;
		const auto :target:`cnt<doxid-classsimple__memory__pool__t_1a3afdc979edee7b39719033731cd5a2e6>` = map_size_ptr_.count(size);
		return :target:`ptr<doxid-classsimple__memory__pool__t_1a25690c1133ddc02373fdac5a8fc3f2af>`;

		// methods
	
		void* :target:`allocate<doxid-classsimple__memory__pool__t_1a8741b91092c5ef7a31b05c6419bac3ae>`(size_t size, size_t alignment, const void* dev, const void* ctx);
		:target:`if<doxid-classsimple__memory__pool__t_1a95466afb7a58832df2c395cbf4723236>`(size = =0);
		:target:`if<doxid-classsimple__memory__pool__t_1afcf36ed4816d4bf2fb845df3f8788128>`(cnt, 0);
		:target:`if<doxid-classsimple__memory__pool__t_1afbb194cb703d8606037f001b73c444cb>`(need_alloc_new_mm);
		void* :target:`allocate_host<doxid-classsimple__memory__pool__t_1a164d75627b6478e17111ff28fcc3c6f6>`(size_t size, size_t alignment);
		void :target:`deallocate<doxid-classsimple__memory__pool__t_1acd9808f4c7f1c234fa9d6f9eb4d44755>`(void* ptr, const void* device, const void* context, void* event);
	
		void :target:`deallocate<doxid-classsimple__memory__pool__t_1a48bbc380fa5d938e20d7036de6fe6169>`(
			void* ptr,
			cl_device_id dev,
			const cl_context ctx,
			cl_event event
			);
	
		void :target:`deallocate_host<doxid-classsimple__memory__pool__t_1a025c1d9565fdcf591e6e1a1b793224b0>`(void* ptr);
		void :target:`clear<doxid-classsimple__memory__pool__t_1a22c3c995262ff714f3981fe722ea467b>`();
	};
