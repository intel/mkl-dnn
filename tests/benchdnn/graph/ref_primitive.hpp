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

#ifndef BENCHDNN_GRAPH_REF_PRIMITIVE_HPP
#define BENCHDNN_GRAPH_REF_PRIMITIVE_HPP

#include "dnnl_common.hpp"

#include "deserialize.hpp"

#include "utils/dnnl_query.hpp"

namespace graph {

// A function that executes a graph reference path for a driver that has no
// primitive (or a custom driver).
using execute_func_t
        = std::function<int(const base_prb_t *, const args_t &, res_t *)>;

// A function that initializes the reference memory arguments for a driver
// that runs through its own reference (an empty primitive or a custom driver).
using init_memory_args_native_func_t = std::function<void(dnn_mem_map_t &,
        const base_prb_t *, const deserialized_op_t &, const engine_t &)>;

// A function that initializes the reference memory arguments for a driver.
using init_ref_memory_args_func_t
        = std::function<int(dnn_mem_map_t &, dnn_mem_map_t &, dnnl_primitive_t,
                const base_prb_t *, res_t *, dnnl_primitive_t)>;

// `ref_primitive_t` is an abstraction to connect a graph op and a primitive
// driver. Its purpose is to translate a graph op into a primitive and execute
// it. Any primitive driver with template programming work should be done
// through this class.
class ref_primitive_t {
public:
    ref_primitive_t() = default;
    ref_primitive_t(const deserialized_op_t &op);

    int init_prb(res_t *res);
    // By default, the reference primitives are created with f32 data type.
    // However, there's a displacer that relies on the logic that would fill
    // memories with int8 data. `force_override` flag restricts forcing f32
    // data type primarily for this use case.
    int init_prim(const engine_t &eng, res_t *res, bool force_override = false);
    void init_memory_args(const engine_t &eng, res_t *res);
    int init_ref_memory_args(const engine_t &eng, res_t *res);
    int execute_prim(res_t *res) const;
    void check_correctness(const args_t &args, bool has_eltwise, bool has_nans,
            res_t *res, float p2p_threshold = 0.f) const;
    // some util function for ref_partition_t to link args
    void replace_arg(const int arg, const dnn_mem_t &mem) {
        // Only compatible memory objects can be replaced.
        const auto &orig_mem = args_.find(arg);
        if (orig_mem.size() != mem.size()) {
            BENCHDNN_PRINT(0,
                    "Error: can't replace mem_%s (%zu) with mem_%s (%zu) for "
                    "%s op.\n",
                    dt2str(orig_mem.dt()), orig_mem.size(), dt2str(mem.dt()),
                    mem.size(), op_.kind_.c_str());
            SAFE_V(FAIL);
        }

        args_.replace(arg, &mem);
    }
    const dnn_mem_t &get_arg(const int arg) const { return args_.find(arg); }
    ::dnnl::graph::op::kind get_kind() const { return kind_; }
    // Displaces scale values in a memory object with scale values from `op`.
    int displace_scales() const;
    dnnl_data_type_t get_lt_dt(size_t id) const;
    const_dnnl_primitive_desc_t get_pd() const { return query_pd(prim_); }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(ref_primitive_t);

    deserialized_op_t op_;
    dnnl::graph::op::kind kind_;
    dnnl_driver_t driver_;
    bool is_special_backward_op_;
    std::shared_ptr<const base_prb_t> prb_;
    // Driver-specific compare object setup function, assigned in `init_prb`.
    setup_cmp_func_t setup_cmp_func_ = nullptr;
    // Driver-specific reference execute function, assigned in `init_prb`.
    execute_func_t execute_func_ = nullptr;
    // Driver-specific reference memory args init function, assigned in
    // `init_prb`.
    init_ref_memory_args_func_t init_ref_memory_args_func_ = nullptr;
    // Driver-specific native memory args init function (used when there's no
    // primitive), assigned in `init_prb`.
    init_memory_args_native_func_t init_memory_args_native_func_ = nullptr;
    // Driver-specific primitive descriptor init function, assigned in
    // `init_prb`. Not used for the custom driver as it has no primitive.
    init_pd_func_t init_pd_func_ = nullptr;
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> fwd_prim_, prim_;
    dnn_mem_map_t mems_;
    args_t args_;
};

} // namespace graph

#endif
