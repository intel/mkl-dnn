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
#include <string>
#include <vector>

#include "utils/prb.hpp"
#include "utils/wrapper.hpp"

using create_func_t = std::function<int(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const base_prb_t *, res_t *)>;
using check_func_t = create_func_t;
using do_func_t = std::function<int(
        const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &,
        const base_prb_t *, res_t *)>;

struct task_t {
    task_t(const std::shared_ptr<base_prb_t> &prb,
            const std::string &perf_template, const create_func_t &create_func,
            const check_func_t &check_func, const do_func_t &do_func, int idx);

    int create(bool in_parallel);
    // Since `task_t` doesn't have control over primitives, it delegates the
    // primitive-based checks to the driver.
    int check();
    int exec();

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
    int report();
};

#endif
