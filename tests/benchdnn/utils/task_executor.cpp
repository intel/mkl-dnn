/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "utils/task_executor.hpp"
#include "utils/parallel.hpp"

#include <cassert>

task_executor_t::~task_executor_t() {
    assert(tasks_.empty());
}

void task_executor_t::submit(const std::shared_ptr<base_prb_t> &prb,
        const std::string &perf_template, const create_func_t &create_func,
        const check_func_t &check_func, const do_func_t &do_func) {
    static const int nthreads = benchdnn_get_max_threads();
    for (int r = 0; r < repeats_per_prb; r++) {
        tasks_.emplace_back(prb, perf_template, create_func, check_func,
                do_func, get_idx());
        if (has_bench_mode_modifier(mode_modifier_t::par_create)
                && static_cast<int>(tasks_.size()) < nthreads)
            continue;
        flush();
    }
}

void task_executor_t::flush() {
    // Special case is needed for THREADPOOL RUNTIME. Both `Parallel_nd` and
    // `createit` calls activate threadpool which causes undesired behavior.
    if (tasks_.size() == 1)
        tasks_[0].create(/* in_parallel = */ false);
    else
        benchdnn_parallel_nd(tasks_.size(),
                [&](int i) { tasks_[i].create(/* in_parallel = */ true); });

    // Check caches first to avoid filling cache with service reorders.
    for (auto &t : tasks_) {
        t.check();
    }

    for (auto &t : tasks_) {
        t.exec();
    }

    tasks_.clear();
}

int task_executor_t::get_idx() {
    static int idx = 0;
    return idx++;
}
