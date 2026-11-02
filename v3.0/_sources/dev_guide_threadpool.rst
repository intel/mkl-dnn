.. index:: pair: page; Using oneDNN with Threadpool-Based Threading
.. _doxid-dev_guide_threadpool:

Using oneDNN with Threadpool-Based Threading
============================================

When oneDNN is built with the threadpool CPU runtime (see :ref:`Build Options <doxid-dev_guide_build_options>`), oneDNN requires the user to implement a threadpool interface to enable the library to perform computations using multiple threads.

The threadpool interface is defined in ``:ref:`include/oneapi/dnnl/dnnl_threadpool_iface.hpp <doxid-oneapi_2dnnl_2dnnl__threadpool__iface_8hpp_source>```. Below is a sample implementation based on the Eigen threadpool that is also used for testing (see ``tests/test_thread.cpp``).

.. ref-code-block:: cpp

	#include "dnnl_threadpool_iface.hpp"
	
	class threadpool_t : public :ref:`dnnl::threadpool_interop::threadpool_iface <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>` {
	private:
	    // Change to Eigen::NonBlockingThreadPool if using Eigen <= 3.3.7
	    std::unique_ptr<Eigen::ThreadPool> tp_;
	
	public:
	    explicit threadpool_t(int num_threads = 0) {
	        if (num_threads <= 0)
	            num_threads = (int)std::thread::hardware_concurrency();
	        tp_.reset(new Eigen::ThreadPool(num_threads));
	    }
	    int :ref:`get_num_threads <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114>`() const override { return tp_->NumThreads(); }
	    bool :ref:`get_in_parallel <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a8279221c6e2f903a4c811688f7a033be>`() const override {
	        return tp_->CurrentThreadId() != -1;
	    }
	    uint64_t :ref:`get_flags <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a868267178f259cee5f1d5b33a8781a3e>`() override { return :ref:`ASYNCHRONOUS <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a9e6d861d659445fe5abcf302e464d9e5>`; }
	    void :ref:`parallel_for <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1ac3d85ff935c11ec038ecabeeabd03ffb>`(int n, const std::function<void(int, int)> &fn) override {
	        int nthr = :ref:`get_num_threads <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114>`();
	        int njobs = std::min(n, nthr);
	
	        for (int i = 0; i < njobs; i++) {
	            tp_->Schedule([i, n, njobs, fn]() {
	                int start, end;
	                impl::balance211(n, njobs, i, start, end);
	                for (int j = start; j < end; j++)
	                    fn(j, n);
	            });
	        }
	    };
	};

