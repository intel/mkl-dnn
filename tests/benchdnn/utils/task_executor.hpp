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

#ifndef UTILS_TASK_EXECUTOR_HPP
#define UTILS_TASK_EXECUTOR_HPP

#include "utils/task.hpp"

extern int repeats_per_prb;

struct task_executor_t {
    virtual ~task_executor_t();

    void submit(const std::shared_ptr<base_prb_t> &prb,
            const std::string &perf_template, const create_func_t &create_func,
            const check_func_t &check_func, const do_func_t &do_func);

    void flush();

    std::vector<task_t> tasks_;

    int get_idx();
};

#endif
