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

#ifndef UTILS_FILL_HPP
#define UTILS_FILL_HPP

#include "dnn_types.hpp"
#include "dnnl_memory.hpp"

#include "utils/numeric.hpp"

// `fill_cfg_t` specifies filling parameters for a `fill_random_real` function.
// By default, the filling is assumed to be random floating-point values in a
// range of [-16.f; 16.f] represented by `float` data type.
// Depending on `dt` specified, different behavior are implied, e.g., integer
// data types imply `only_integer_` set to `true`.
// When `only_integer_` is set, generated values have their fractional part
// zeroed.
// When `alg` is passed, it may implicitly update ranges to work around
// potential floating-point issues. E.g., `sub` algorithm will inverse the range
// borders to act like `add`, which allows to keep output data positive.
//
// Note: members are exposed as public for higher flexibility on modifying
// configs.
//
struct fill_cfg_t {
    fill_cfg_t()
        : dt_(dnnl_f32)
        , range_min_val_(-16.f)
        , range_max_val_(16.f)
        , predefined_set_({})
        , density_(1.f)
        , only_integer_(false) {}

    fill_cfg_t(dnnl_data_type_t dt, float range_min_val, float range_max_val,
            bool only_integer, attr_t::post_ops_t::kind_t alg,
            const std::string &name);

    fill_cfg_t(const std::vector<float> &user_set, const std::string &name);
    fill_cfg_t(const std::vector<float> &user_set, float density,
            const std::string &name);

    std::string print_verbose() const;

    // Data type used for rounding final values to.
    dnnl_data_type_t dt_;
    // The lower bound for the filling range.
    float range_min_val_;
    // The upper bound for the filling range.
    float range_max_val_;
    // It's an alternative approach to fill the data. Instead of setting range
    // borders, pass a set of fixed values. The function will use uniform_int
    // randomizer to choose the value.
    //
    // Helpful to generate pow2 filling, by passing {2.f, 4.f, ...};
    std::vector<float> predefined_set_;
    // Specify density as a value from [0.f; 1.f] range, where 0.f means
    // completely sparse (all zeros), and 1.f means completely dense.
    //
    // Works in pair with `predefined_set_`.
    float density_;
    // A flag to generate only integer values.
    bool only_integer_;
    // Config name for verbosity.
    std::string name_;
};

const fill_cfg_t &get_default_fill_cfg();
const fill_cfg_t &get_perf_fill_cfg(dnnl_data_type_t dt);

int fill_dropout_mask(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp);

int fill_scales(const attr_t &attr, int arg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res);
int fill_scales(const attr_t::arg_scales_t::entry_t &e, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res);

int fill_zero_points(const attr_t &attr, int arg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res);

int fill_random_real(dnn_mem_t &mem, dnn_mem_t &mem_ref, res_t *res,
        const fill_cfg_t &fill_cfg = get_default_fill_cfg(),
        const_dnnl_memory_t dnnl_memory = nullptr);

int fill_random_real(dnn_mem_t &mem_ref,
        const fill_cfg_t &fill_cfg = get_default_fill_cfg(),
        const_dnnl_memory_t dnnl_memory = nullptr);

extern std::string buffer_prefix;

struct fill_input_t {
    int value = -1; // -1: random data, otherwise the constant value
    uint32_t mask = 0xFFFFFFFFu;
    bool is_const() const { return value >= 0; }
    bool is_def() const { return !is_const() && mask == 0xFFFFFFFFu; }
};
std::ostream &operator<<(std::ostream &s, const fill_input_t &fill);

extern fill_input_t fill_input;
extern bool fill_input_applied;

int fill_by_input(const dnn_mem_t &mem, const fill_input_t &fill, size_t size,
        int buffer_index);

bool fill_from_file(
        int exec_arg, dnn_mem_t &mem, dnn_mem_t &ref_mem, res_t *res);

#endif
