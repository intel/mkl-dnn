/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef BENCHDNN_GRAPH_GRAPH_HPP
#define BENCHDNN_GRAPH_GRAPH_HPP

#define UNMAP 0
#define MAP 1

#include <iostream>
#include <map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"
#include "utils.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

namespace graph {

using namespace dnnl::graph;

struct settings_t : public base_settings_t {
    settings_t(const char *perf_template = perf_template_def)
        : base_settings_t(perf_template) {}

    std::string json_file;
    std::vector<std::map<size_t, std::string>> in_shapes_vec {{{0, "default"}}};
    std::vector<std::map<size_t, std::string>> op_attrs_vec {{{0, "default"}}};
    // By default, we expect the graph should be fused as a single partition.
    // The user can specify `--expected-n-partitions=0` to skip the partition
    // number check.
    std::vector<size_t> expected_n_partition_vec {1};
    std::vector<graph_fpmath_mode_t> fpmath_mode_vec {graph_fpmath_mode_t {}};
    std::vector<dnnl_data_type_t> dt {dnnl_data_type_undef};
    std::vector<std::map<size_t, dnnl_data_type_t>> dt_map {
            {{SIZE_MAX, dnnl_data_type_undef}}};
    std::vector<std::map<size_t, std::string>> op_kind_map {
            {{SIZE_MAX, "default"}}};
    std::vector<std::map<size_t, std::string>> tensor_property_vec {
            {{SIZE_MAX, "default"}}};

    const char *perf_template_csv = "perf,%engine%,%desc%,%-time%,%0time%";
    static constexpr const char *perf_template_def
            = "perf,%engine%,%prb%,%-time%,%0time%";

    void reset() { *this = settings_t(perf_template); }
};

// TODO evaluate prb_t struct
struct prb_t {
    prb_t(const deserialized_graph_t &dg, const size_t &expected_n_partition);

    deserialized_graph_t dg;
    size_t expected_n_partition;
    graph_fpmath_mode_t fpmath_mode;
};

std::string case_to_str(const std::string &json_file,
        const std::map<size_t, std::string> &in_shapes,
        const std::map<size_t, std::string> &op_attrs,
        const graph_fpmath_mode_t &fpmath_mode,
        const size_t expected_n_partitions, const int64_t mb,
        const dnnl_data_type_t dt,
        const std::map<size_t, dnnl_data_type_t> &dt_map,
        const std::map<size_t, std::string> &op_kind_map,
        const std::map<size_t, std::string> &tensor_property);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const std::string &case_str, const char *perf_template)
        : base_perf_report_t(perf_template), case_str_(case_str) {}

    void dump_desc(std::ostream &s) const override { s << case_str_; }

private:
    const std::string case_str_;
};

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);
} // namespace graph

#endif
