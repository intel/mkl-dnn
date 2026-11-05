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

#ifndef BENCHDNN_GRAPH_CUSTOM_DRIVER_H
#define BENCHDNN_GRAPH_CUSTOM_DRIVER_H

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "utils/prb.hpp"
#include "utils/settings.hpp"

namespace custom {

enum alg_t {
    GENINDEX,
    TRANSPOSE,
    RESHAPE,
    ALG_UNKNOWN,
};

using arg_md_t = ::std::tuple<::std::string, dims_t, dnnl_data_type_t>;

struct settings_t {

    settings_t() = default;

    ::std::unordered_map<int, arg_md_t> arg_mds_;
    ::std::vector<int64_t> order;
    int64_t axis = -1;
    alg_t alg = ALG_UNKNOWN;

    // A stub to be compliant with `base_settings_t`.
    void finalize() {};
};

struct prb_t : public base_prb_t {
    prb_t(const settings_t &s) : arg_mds_(s.arg_mds_), alg(s.alg) {
        switch (alg) {
            case GENINDEX: axis = s.axis; break;
            case TRANSPOSE: order = s.order; break;
            default: break;
        }
    }

    ::std::unordered_map<int, arg_md_t> arg_mds_;
    ::std::vector<int64_t> order;
    int64_t axis = -1;
    alg_t alg = ALG_UNKNOWN;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

    // The custom driver runs only from the graph driver, which reports
    // through its own `graph::perf_report_t`, never through `task_t`.
    std::unique_ptr<base_perf_report_t> create_perf_report(
            const char *perf_template) const override {
        return nullptr;
    }

private:
    std::string set_repro_line() override { return std::string(); }
};

dnnl_status_t init_pd(init_pd_args_t &init_pd_args);

int fill_mem(
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int f_min, int f_max, res_t *res);
void setup_cmp(compare::compare_t &cmp, const base_prb_t *prb, data_kind_t kind,
        const args_t &ref_args);

void init_memory_args(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        bool override_dir_with_fwd = false, res_t *res = nullptr,
        const engine_t &test_engine = get_test_engine());

void init_memory_args_native(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        const graph::deserialized_op_t &base_op_ref, const engine_t &ref_eng);

void collect_mem_size(
        check_mem_size_args_t &mem_size_args, const base_prb_t *base_prb);

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

int execute(const base_prb_t *prb, const args_t &args, res_t *res);

} // namespace custom
#endif
