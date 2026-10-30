Using Threadpool-based Threading {#dev_guide_threadpool}
========================================================

When oneDNN is built with the threadpool CPU runtime (see @ref
dev_guide_build_options), oneDNN requires the user to implement a threadpool
interface to enable the library to perform computations using multiple
threads.

The threadpool interface is defined in
``include/oneapi/dnnl/dnnl_threadpool_iface.hpp``. Below is a sample
implementation based on the Eigen threadpool that is also used for testing (see
`tests/test_thread.cpp`).

~~~cpp
#include "dnnl_threadpool_iface.hpp"

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
private:
    // Change to Eigen::NonBlockingThreadPool if using Eigen <= 3.3.7
    std::unique_ptr<Eigen::ThreadPool> tp_;

public:
    explicit threadpool_t(int num_threads = 0) {
        if (num_threads <= 0)
            num_threads = (int)std::thread::hardware_concurrency();
        tp_.reset(new Eigen::ThreadPool(num_threads));
    }
    int get_num_threads() const override { return tp_->NumThreads(); }
    bool get_in_parallel() const override {
        return tp_->CurrentThreadId() != -1;
    }
    uint64_t get_flags() const override { return ASYNCHRONOUS; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        int nthr = get_num_threads();
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
~~~

oneDNN supports non-blocked verbose profiling of primitive executions for
threadpool runtimes provided the user define a threadpool event and event interface 
for the profiling mechanism to poll for event completion. An example of this event
interfasce is also provided in ``include/oneapi/dnnl/dnnl_threadpool_iface.hpp`` 
and the following sample implementation from `test/test_thread.cpp` can be used for
testing.

~~~cpp
// Lightweight completion event exposed to oneDNN's verbose profiler.
// the start time for execution is stamped by the first parallel_for()
// worker via a compare-and-swap atomic operation to avoid races
// between workers. The end time and completion flag are set by the
// AndThen callback registered in get_event() when the chain resolves.
class threadpool_event_t
    : public dnnl::threadpool_interop::threadpool_event_iface_t {
public:
    threadpool_event_t() = default;

    bool is_complete() const override {
        return complete_.load(std::memory_order_acquire);
    }

    void wait() const override {
        while (!is_complete()) {}
    }

    double exec_time_ms() const override {
        int64_t start = start_ns_.load(std::memory_order_relaxed);
        int64_t end = end_ns_.load(std::memory_order_relaxed);
        return static_cast<double>(end - start) * 1e-6;
    }

    void stamp_start(int64_t ns) {
        int64_t expected = 0;
        start_ns_.compare_exchange_strong(
                expected, ns, std::memory_order_relaxed);
    }

    // Records the end timestamp and sets complete_.
    // end_ns_ is written before complete_ to ensure exec_time_ms() is
    // valid when is_complete() returns true.
    void mark_complete() {
        end_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now()
                                      .time_since_epoch())
                              .count(),
                std::memory_order_relaxed);
        complete_.store(true, std::memory_order_release);
    }

private:
    // Completion flag set by mark_complete() in the AndThen callback.
    std::atomic<bool> complete_ {false};

    // Start timestamp in nanoseconds. Stamped by the first worker via a
    // a compare-and-swap atomic operation.
    std::atomic<int64_t> start_ns_ {0};

    // End timestamp in nanoseconds. Written by mark_complete() before
    // complete_ is set.
    std::atomic<int64_t> end_ns_ {0};
};
~~~
