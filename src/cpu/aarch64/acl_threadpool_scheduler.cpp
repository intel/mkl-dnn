/*******************************************************************************
* Copyright 2022-2026 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/aarch64/acl_threadpool_scheduler.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include "common/counting_barrier.hpp"
#include "common/dnnl_thread.hpp"
#include "cpu/aarch64/acl_thread.hpp"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IScheduler.h"

#include <atomic>
#include <cassert>
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace arm_compute;

class thread_feeder_t {
public:
    explicit thread_feeder_t(unsigned int start = 0, unsigned int end = 0)
        : _atomic_counter(start), _end(end) {}

    /// Function to check the next element in the range if there is one.
    bool get_next(unsigned int &next) {
        next = std::atomic_fetch_add_explicit(
                &_atomic_counter, 1u, std::memory_order_relaxed);
        return next < _end;
    }

private:
    std::atomic_uint _atomic_counter;
    const unsigned int _end;
};

void process_workloads(std::vector<IScheduler::Workload> &workloads,
        thread_feeder_t &feeder, const ThreadInfo &info) {
    unsigned int workload_index = info.thread_id;
    do {
        ARM_COMPUTE_ERROR_ON(workload_index >= workloads.size());
        workloads[workload_index](info);
    } while (feeder.get_next(workload_index));
}

threadpool_scheduler_t::threadpool_scheduler_t()
    : _num_threads(dnnl_get_max_threads()) {}

threadpool_scheduler_t::~threadpool_scheduler_t() = default;

unsigned int threadpool_scheduler_t::num_threads() const {
    return _num_threads;
}

void threadpool_scheduler_t::set_num_threads(unsigned int num_threads) {
    std::lock_guard<std::mutex> lock(this->_mtx);
    _num_threads = num_threads == 0 ? dnnl_get_max_threads() : num_threads;
}

void threadpool_scheduler_t::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    // Retrieve threadpool size during primitive execution and set threadpool_scheduler_t num_threads
    acl_thread_utils::acl_set_threadpool_num_threads();
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void threadpool_scheduler_t::schedule_op(ICPPKernel *kernel, const Hints &hints,
        const Window &window, ITensorPack &tensors) {
    // Retrieve threadpool size during primitive execution and set threadpool_scheduler_t num_threads
    acl_thread_utils::acl_set_threadpool_num_threads();
    schedule_common(kernel, hints, window, tensors);
}

void threadpool_scheduler_t::run_workloads(
        std::vector<arm_compute::IScheduler::Workload> &workloads) {

    std::lock_guard<std::mutex> lock(this->_mtx);

    const unsigned int num_threads
            = std::min(static_cast<unsigned int>(_num_threads),
                    static_cast<unsigned int>(workloads.size()));
    if (num_threads < 1) { return; }
    thread_feeder_t feeder(num_threads, workloads.size());
    using namespace dnnl::impl::threadpool_utils;
    dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    // Non-initialized threadpool can cause a segmentation fault.
    // Here, the task is executed in a single thread when threadpool
    // is unavailable or is single-threaded.
    if (!tp || num_threads == 1) {
        threadpool_utils::deactivate_threadpool();
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        info.num_threads = 1;
        info.thread_id = 0;
        process_workloads(workloads, feeder, info);
        threadpool_utils::activate_threadpool(tp);
        return;
    }
    bool is_async = tp->get_flags()
            & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
    counting_barrier_t b;
    if (is_async) b.init(num_threads);
    tp->parallel_for(num_threads, [&](int ithr, int nthr) {
        bool is_main = get_active_threadpool() == tp;
        if (!is_main) activate_threadpool(tp);
        // Make ThreadInfo local to avoid race conditions
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        info.num_threads = nthr;
        info.thread_id = ithr;
        process_workloads(workloads, feeder, info);
        if (!is_main) deactivate_threadpool();
        if (is_async) b.notify();
    });
    if (is_async) b.wait();
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
