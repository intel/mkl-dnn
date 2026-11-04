/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <memory>
#include <stdlib.h>
#include <vector>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl_graph.h"

#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/partition.hpp"

#include "graph/utils/debug.hpp"
#include "graph/utils/utils.hpp"
#include "graph/utils/verbose.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

void print_verbose_header() {
    std::vector<const backend_t *> &backends
            = backend_registry_t::get_singleton().get_registered_backends();
    for (size_t i = 0; i < backends.size() - 1; ++i) {
        backend_t *bkd = const_cast<backend_t *>(backends[i]);
        verbose_printf(
                "info,graph,backend,%zu:%s\n", i, bkd->get_name().c_str());
    }
}

namespace {

std::string logical_tensor2dim_str(const logical_tensor_t &logical_tensor) {
    std::string s;

    auto lt = logical_tensor_wrapper_t(logical_tensor);

    s += ":";
    s += std::to_string(lt.dims()[0]);
    for (int d = 1; d < lt.ndims(); ++d)
        s += ("x" + std::to_string(lt.dims()[d]));

    return s;
}

std::string logical_tensor2layout_str(const logical_tensor_t &logical_tensor) {
    std::string s;

    auto lt = logical_tensor_wrapper_t(logical_tensor);

    s += ":";
    if (lt.layout_type() == layout_type::strided) {
        const auto strides = lt.strides();
        for (int i = 0; i < lt.ndims() - 1; ++i) {
            s += std::to_string(strides[i]);
            s += "s";
        }
        s += std::to_string(strides[lt.ndims() - 1]);
    } else if (lt.layout_type() == layout_type::opaque) {
        s += std::to_string(lt.layout_id());
    } else if (lt.layout_type() == layout_type::any) {
        s += "any";
    } else {
        assert(!"layout type must be any, strided or opaque.");
    }

    return s;
}

std::string logical_tensor2str(const logical_tensor_t &logical_tensor) {
    std::string s;

    s += std::string(data_type2str(logical_tensor.data_type));
    s += ":";
    s += std::to_string(logical_tensor.id);
    s += ":";
    s += std::string(layout_type2str(logical_tensor.layout_type));
    s += ":";
    s += std::string(property_type2str(logical_tensor.property));

    return s;
}

std::string partition2fmt_str(const partition_t &partition) {
    std::string s;

    const std::vector<std::shared_ptr<graph::op_t>> &operators
            = partition.get_ops();
    const size_t num_operator = operators.size();
    if (num_operator == 0) return s;

    bool data_filled = false;
    bool filter_filled = false;
    for (size_t i = 0; i < num_operator; ++i) {
        const std::shared_ptr<op_t> &op = operators[i];
        if (op->has_attr(op_attr::data_format)) {
            // If the first i ops have no data_format, empty string with suffix
            // `;` should be printed out for each of them.
            if (!data_filled) {
                s += "data:";
                for (size_t ii = 0; ii < i; ++ii)
                    s += ";";
                // Indicates that at least one op in the list have data format
                // spec.
                data_filled = true;
            }
            const auto data_format
                    = op->get_attr<std::string>(op_attr::data_format);
            if (i == num_operator - 1) {
                s += data_format;
                s += " ";
            } else {
                s += data_format;
                s += ";";
            }
        } else if (data_filled) {
            // If at least one op have data format, op without format spec
            // should give `;` except the last one of data which should give
            // ` `.
            if (i == num_operator - 1) {
                s += " ";
            } else {
                s += ";";
            }
        }
    }
    for (size_t i = 0; i < num_operator; ++i) {
        const std::shared_ptr<op_t> &op = operators[i];
        if (op->has_attr(op_attr::weights_format)) {
            if (!filter_filled) {
                s += "filter:";
                for (size_t ii = 0; ii < i; ++ii)
                    s += ";";
                filter_filled = true;
            }
            const auto filter_format
                    = op->get_attr<std::string>(op_attr::weights_format);
            if (i == num_operator - 1) {
                s += filter_format;
                s += " ";
            } else {
                s += filter_format;
                s += ";";
            }
        } else if (filter_filled) {
            s += ";";
        }
    }

    return s;
}

std::string init_info_partition(const engine_t *engine,
        const compiled_partition_t *compiled_partition) {
    stringstream_t ss;

    const auto &partition = compiled_partition->src_partition();

    ss << std::string(engine_kind2str(engine->kind())) << "," << partition.id()
       << "," << partition_kind2str(partition.get_kind()) << ",";

    const std::vector<std::shared_ptr<graph::op_t>> &operators
            = partition.get_ops();
    const size_t num_operators = operators.size();
    for (size_t i = 0; i < num_operators; ++i) {
        ss << operators[i]->get_name()
           << ((i == num_operators - 1) ? "," : ";");
    }

    ss << partition2fmt_str(partition) << ",";
    {
        const auto &inputs = compiled_partition->get_inputs();
        const size_t inputs_size = inputs.size();
        for (size_t i = 0; i < inputs_size; ++i) {
            ss << "in" << i << "_" << logical_tensor2str(inputs[i])
               << logical_tensor2dim_str(inputs[i])
               << logical_tensor2layout_str(inputs[i]) << " ";
        }
    }

    {
        const auto &outputs = compiled_partition->get_outputs();
        const size_t outputs_size = outputs.size();
        for (size_t i = 0; i < outputs_size; ++i) {
            ss << "out" << i << "_" << logical_tensor2str(outputs[i])
               << logical_tensor2dim_str(outputs[i])
               << logical_tensor2layout_str(outputs[i]);
            if (i < outputs_size - 1) ss << " ";
        }
    }

    const auto &fpm = partition.get_pimpl()->get_attr().get_fpmath();
    ss << ",fpm:" << fpmath_mode2str(fpm.mode_);
    if (fpm.apply_to_int_) ss << ":true";

    ss << ",scratchpad:" << compiled_partition->get_scratchpad_size();

    ss << "," << compiled_partition->get_pimpl()->str();

    ss << "," << partition.get_assigned_backend()->get_name();

    return ss.str();
}

} // namespace

void partition_info_t::init(const engine_t *engine,
        const compiled_partition_t *compiled_partition) {
    // Handles VERBOSE_DISABLE since `is_initialized_` is set to `true`.
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
        str_ = init_info_partition(engine, compiled_partition);
        is_initialized_ = true;
    });
}

#ifndef DNNL_DISABLE_GRAPH_DUMP
static setting_t<uint8_t> graph_dump_modes {0};
#endif

uint8_t parse_graph_dump_mode(const std::string &modes) {
    uint8_t m = 0;
    if (modes.empty()) return m;

    std::string user_opt = modes;
    std::transform(
            user_opt.begin(), user_opt.end(), user_opt.begin(), ::tolower);
    user_opt += ','; // Add a trailing comma to process the last token
    size_t start = 0, end = 0;
    while ((end = user_opt.find(',', start)) != std::string::npos) {
        std::string token = user_opt.substr(start, end - start);
        if (token == "subgraph")
            m |= static_cast<uint8_t>(graph_dump_mode_t::subgraph);
        else if (token == "graph")
            m |= static_cast<uint8_t>(graph_dump_mode_t::graph);
        else if (token == "pattern")
            m |= static_cast<uint8_t>(graph_dump_mode_t::pattern);
        else
            m = static_cast<uint8_t>(graph_dump_mode_t::none);
        start = end + 1;
    }
    return m;
}

bool get_graph_dump_mode(graph_dump_mode_t mode) {
#ifdef DNNL_DISABLE_GRAPH_DUMP
    return false;
#else
    if (!graph_dump_modes.initialized()) {
        // Assumes that all threads see the same environment
        static std::string env = getenv_string_user("GRAPH_DUMP");
        graph_dump_modes.set(parse_graph_dump_mode(env));
    }

    uint8_t saved_mode = graph_dump_modes.get();
    uint8_t target_mode = static_cast<uint8_t>(mode);
    if (saved_mode == 0) {
        return saved_mode == target_mode;
    } else {
        return (saved_mode & target_mode) != 0;
    }
#endif
}

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

dnnl::impl::graph::status_t dnnl_graph_set_dump_mode(
        dnnl_graph_dump_mode_t modes) {
#ifdef DNNL_DISABLE_GRAPH_DUMP
    return dnnl::impl::graph::status::invalid_arguments;
#else
    const uint8_t mask = static_cast<uint8_t>(modes);
    const uint8_t allowed_mask
            = static_cast<uint8_t>(dnnl_graph_dump_mode_graph)
            | static_cast<uint8_t>(dnnl_graph_dump_mode_subgraph);

    if ((mask & ~allowed_mask) != 0)
        return dnnl::impl::graph::status::invalid_arguments;

    dnnl::impl::graph::utils::graph_dump_modes.set(mask);
    return dnnl::impl::graph::status::success;
#endif
}
