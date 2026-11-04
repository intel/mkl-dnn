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

#include "cpu/aarch64/acl_thread.hpp"
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "cpu/aarch64/acl_threadpool_scheduler.hpp"
#endif
#include "cpu/aarch64/acl_benchmark_scheduler.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_thread_utils {

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
void acl_thread_bind() {
    static std::once_flag flag_once;
    // The threads in Compute Library are bound for the cores 0..max_threads-1
    // dnnl_get_max_threads() returns OMP_NUM_THREADS
    const int max_threads = dnnl_get_max_threads();
    // arm_compute::Scheduler does not support concurrent access thus a
    // workaround here restricts it to only one call
    std::call_once(flag_once, [&]() {
        arm_compute::Scheduler::get().set_num_threads(max_threads);
    });
}
// Swap benchmark_scheduler_t for default ACL scheduler builds (i.e. CPPScheduler, OMPScheduler)
void acl_set_benchmark_scheduler_default() {
    static std::once_flag flag_once;
    arm_compute::IScheduler *_real_scheduler = &arm_compute::Scheduler::get();
    std::shared_ptr<arm_compute::IScheduler> benchmark_scheduler
            = std::make_unique<benchmark_scheduler_t>(*_real_scheduler);
    // set Benchmark scheduler in ACL
    std::call_once(flag_once, [&]() {
        arm_compute::Scheduler::set(
                std::static_pointer_cast<arm_compute::IScheduler>(
                        benchmark_scheduler));
    });
}
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

void acl_set_tp_scheduler() {
    static thread_local std::once_flag flag_once;
    // Create threadpool scheduler
    std::call_once(flag_once, [&]() {
        // Create threadpool scheduler
        std::shared_ptr<arm_compute::IScheduler> threadpool_scheduler
                = std::make_unique<threadpool_scheduler_t>();
        arm_compute::Scheduler::set(threadpool_scheduler);
    });
}

void acl_set_threadpool_num_threads() {
    using namespace dnnl::impl::threadpool_utils;
    static thread_local std::once_flag flag_once;
    threadpool_interop::threadpool_iface *tp = get_active_threadpool();
    // Check active threadpool
    bool is_main = get_active_threadpool() == tp;
    if (is_main) {
        // Set num threads based on threadpool size
        const int num_threads = (tp) ? dnnl_get_max_threads() : 1;
        std::call_once(flag_once, [&]() {
            arm_compute::Scheduler::get().set_num_threads(num_threads);
        });
    }
}
// Swap benchmark_scheduler_t for custom scheduler builds (i.e. ThreadPoolScheduler)
void acl_set_tp_benchmark_scheduler() {
    static thread_local std::once_flag flag_once;
    std::call_once(flag_once, [&]() {
        // Create threadpool scheduler
        std::unique_ptr<arm_compute::IScheduler> threadpool_scheduler
                = std::make_unique<threadpool_scheduler_t>();
        arm_compute::IScheduler *_real_scheduler = nullptr;
        _real_scheduler = threadpool_scheduler.release();

        // Create benchmark scheduler and set TP as real scheduler
        std::shared_ptr<arm_compute::IScheduler> benchmark_scheduler
                = std::make_unique<benchmark_scheduler_t>(*_real_scheduler);

        arm_compute::Scheduler::set(benchmark_scheduler);
    });
}
#endif

void set_acl_threading() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    acl_thread_bind();
    if (get_verbose(verbose_t::profile_externals)) {
        acl_set_benchmark_scheduler_default();
    }
#endif
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (get_verbose(verbose_t::profile_externals)) {
        acl_set_tp_benchmark_scheduler();
    } else {
        acl_set_tp_scheduler();
    }

#endif
}

} // namespace acl_thread_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
