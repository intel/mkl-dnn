/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef UTILS_TASK_HPP
#define UTILS_TASK_HPP

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "utils/prb.hpp"
#include "utils/wrapper.hpp"

using create_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const base_prb_t *, res_t *)>;
using check_func_t = create_func_t;
using do_func_t = std::function<int(
        const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const base_prb_t *, res_t *)>;

template <typename perf_report_t>
struct task_t {
    task_t(const std::shared_ptr<base_prb_t> &prb,
            const std::string &perf_template, const create_func_t &create_func,
            const check_func_t &check_func, const do_func_t &do_func, int idx)
        : prb_(prb)
        , create_func_(create_func)
        , check_func_(check_func)
        , do_func_(do_func)
        , perf_template_(perf_template)
        , idx_(idx)
        , fix_times_(fix_times_per_prb) {}

    int create(bool in_parallel) {
        // Report creation status for problems only in sequential mode as in
        // parallel it's still not clear which one failed.
        if (!in_parallel) BENCHDNN_PRINT(1, "create: %s\n", prb_->str());

        if (skip_start(&res_, idx_)) return OK;
        if (bench_mode == bench_mode_t::list) return res_.state = LISTED, OK;

        v_prim_ = std::make_shared<
                std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>>();

        // A timer for each test case. Wraps each call for fine timings.
        // Otherwise, due to parallel creation, timer for a problem starts but
        // waits for its line.
        auto &tct = res_.timer_map.get_timer(timer::names::test_case_timer);
        tct.start();
        SAFE(create_func_(*v_prim_, prb_.get(), &res_), WARN);
        tct.stamp();
        return OK;
    }

    // Since `task_t` doesn't have control over primitives, it delegates the
    // primitive-based checks to the driver.
    int check() {
        // No alive testing objects - no checks.
        if (res_.state != INITIALIZED) return OK;

        auto &tct = res_.timer_map.get_timer(timer::names::test_case_timer);
        tct.start();
        SAFE(check_func_(*v_prim_, prb_.get(), &res_), WARN);
        tct.stamp();
        return OK;
    }

    int exec() {
        // Checking for `INITIALIZED` state here prevents from `SKIPPED`
        // problems being executed.
        auto &tct = res_.timer_map.get_timer(timer::names::test_case_timer);
        tct.start();
        if (res_.state == INITIALIZED && bench_mode != bench_mode_t::init) {
            // Differentiate a message when the run happens...
            BENCHDNN_PRINT(1, "run: %s\n", prb_->str());
            // Parallel create leaves the chunk's last parsed value.
            fix_times_per_prb = fix_times_;
            do_func_(*v_prim_, prb_.get(), &res_);
        } else {
            // ... versus when it didn't but still indicating the problem went
            // through this part of the flow.
            BENCHDNN_PRINT(1, "run (just report, no exec): %s\n", prb_->str());
        }

        tct.stamp();

        return report();
    }

private:
    // Held via a base-class pointer because storing `base_prb_t` by value would
    // slice the derived problem descriptor. A pointer to a heap-allocated
    // concrete object (upcast to `base_prb_t`) is required. `shared_ptr` (over
    // `unique_ptr`) keeps `task_t` copyable for `tasks_` vector storage.
    std::shared_ptr<const base_prb_t> prb_;
    create_func_t create_func_;
    check_func_t check_func_;
    do_func_t do_func_;
    std::string perf_template_;
    res_t res_ {};
    int idx_;
    int fix_times_;
    // Use vector to handle any number of primitives needed for a driver.
    // It's a driver responsibility to initialize vector at `create()` stage
    // and it utilizes the knowledge of primitive order inside the vector.
    //
    // Note: shared_ptr here is to work around old compiler issue that triggers
    // `vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>` copy constructor,
    // which is deleted for `benchdnn_dnnl_wrapper_t`. New compilers don't do
    // that since `v_prim_` is not initialized at constructor time.
    //
    // TODO: remove shared_ptr work around when migrate to newer miminal default
    // compiler.
    std::shared_ptr<std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>>>
            v_prim_;

    // Note: can't be `const` because of `parse_result`.
    int report() {
        parse_result(res_, prb_->str());
        if (has_bench_mode_bit(mode_bit_t::perf)) {
            perf_report_t pr(prb_.get(), perf_template_.c_str());
            pr.report(&res_, prb_->str());
        }
        return OK;
    }
};

#endif
