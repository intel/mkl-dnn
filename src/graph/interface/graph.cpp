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

#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/partition.hpp"

#include "graph/utils/pm/dag_check_pass.hpp"
#include "graph/utils/pm/op_depth_check_pass.hpp"
#include "graph/utils/pm/pass_manager.hpp"

#include "graph/utils/json.hpp"

using namespace dnnl::impl::graph;

namespace {
std::vector<size_t> get_ids(
        const std::vector<std::shared_ptr<value_t>> &values) {
    std::vector<size_t> ids;
    ids.reserve(values.size());
    for (const auto &t : values) {
        ids.push_back(t->get_logical_tensor().id);
    }
    return ids;
}

bool logical_tensor_sanity_check(
        std::unordered_map<size_t, logical_tensor_t> &id_to_lts,
        const std::vector<std::shared_ptr<value_t>> &values) {
    for (const auto &v : values) {
        auto lt = v->get_logical_tensor();
        auto id_search = id_to_lts.find(lt.id);
        if (id_search == id_to_lts.end()) {
            id_to_lts[lt.id] = lt;
        } else {
            // compare two tensors with the same id;
            if (logical_tensor_wrapper_t(id_search->second)
                    != logical_tensor_wrapper_t(lt)) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

// One requirement of this function is returning the vector of partitions in
// topological order of the fused graph, as the topological order of a fused
// graph and unfused graph may be different(as showed in the example subgraph,
// the topological order of unfused graph is 1234 or 1324, while the one of
// fused graph should be: 13(24) ), the whole workflow is:
// 1. construct an op set `topo_unfused_ops` in topological order of the
// unfused graph
// 2. traverse  `topo_unfused_ops`  in reverse and insert proxy op of each
// partition into `topo_fused_ops`, the definition of “proxy op” is: the last
// one in the topological order of the partition
status_t dnnl_graph_graph::get_ordered_partitions(
        std::vector<partition_t *> &partitions) const {

    std::vector<op_t *> topo_unfused_ops;
    std::vector<op_t *> topo_fused_ops;
    std::unordered_set<partition_impl_t *> visited_parts;
    size_t count = 0;
    topo_unfused_ops.reserve((*this).num_ops());
    topo_fused_ops.reserve((*this).num_ops());
    auto ret = topo_order_visit((*this).get_output_ops(), [&](op_t *n) {
        topo_unfused_ops.emplace_back(n);
        return status::success;
    });
    if (ret != status::success) return ret;

    // Get the last op in topo order as proxy op for each partition.
    std::for_each(topo_unfused_ops.rbegin(), topo_unfused_ops.rend(),
            [&](op_t *op_ptr) {
        partition_impl_t *part = op_ptr->get_partition();
        if (!part) return;
        if (!visited_parts.count(part)) {
            topo_fused_ops.emplace_back(op_ptr);
            visited_parts.emplace(part);
        }
    });

    std::for_each(
            topo_fused_ops.rbegin(), topo_fused_ops.rend(), [&](op_t *op_ptr) {
        partition_impl_t *part = op_ptr->get_partition();
        partitions[count]->init(part->shared_from_this());
        count++;
    });
    return ret;
}

status_t dnnl_graph_graph::finalize() {
    // if the graph is already built, return directly.
    // TODO(xxx): actually we may need to a verification here.
    if (finalized_) return status::success;
    num_unpartitioned_ops_ = num_ops();

    // map of {backend op: vector of input tensor_ids}
    std::unordered_map<graph_t::op_t *, std::vector<size_t>> op_to_tensor_id;
    // map of {tensor_id, (producer_op, producer_offset)}
    std::unordered_map<size_t, std::pair<graph_t::op_t *, size_t>>
            tensor_id_to_producer;
    // map of {tensor_id: tensor}
    std::unordered_map<size_t, logical_tensor_t> id_to_tensor;

    for (const auto &op : ops_) {
        const auto &in_values = op->get_input_values();
        auto out_values = op->get_output_values();
        if (!logical_tensor_sanity_check(id_to_tensor, in_values)
                || !logical_tensor_sanity_check(id_to_tensor, out_values)) {
            return status::invalid_graph;
        }

        // save the input tensor ids of current op
        op_to_tensor_id[op.get()] = get_ids(in_values);
        // save the outputs of current op
        for (size_t i = 0; i < out_values.size(); ++i) {
            const logical_tensor_t out = out_values[i]->get_logical_tensor();
            tensor_id_to_producer[out.id] = std::make_pair(op.get(), i);
        }
    }

    std::list<op_t> dummy_ops; // use list to avoid re-allocation
    std::unordered_map<size_t, std::pair<graph_t::op_t *, size_t>>
            tensor_id_to_dummy_producer;

    for (const op_ptr &op : ops_) {
        // find the input tensor ids of current op
        const std::vector<size_t> &input_tensor_ids = op_to_tensor_id[op.get()];

        for (size_t i = 0; i < input_tensor_ids.size(); ++i) {
            // find the producer of the input tensor
            auto id_search = tensor_id_to_producer.find(input_tensor_ids[i]);
            // if the producer of the input tensor does not exist in the graph,
            // the input tensor is not produced by another op,
            // it's the input of the whole graph
            if (id_search != tensor_id_to_producer.end()) {
                std::pair<op_t *, size_t> producer = id_search->second;
                // set input of current op
                op->connect_input(i, *(producer.first), producer.second);
            } else {
                auto id_search
                        = tensor_id_to_dummy_producer.find(input_tensor_ids[i]);
                if (id_search != tensor_id_to_dummy_producer.end()) {
                    std::pair<op_t *, size_t> dummy_producer
                            = id_search->second;
                    op->connect_input(
                            i, *(dummy_producer.first), dummy_producer.second);
                    op->get_input_value(i)->reset_producer();
                } else { // create a dummy input op
                    dummy_ops.emplace_back(op_kind::Wildcard);
                    dummy_ops.back().add_output(op->get_input_value(i));
                    tensor_id_to_dummy_producer[input_tensor_ids[i]]
                            = std::make_pair(&dummy_ops.back(), 0);
                    op->connect_input(i, op->get_input_value(i));
                    op->get_input_value(i)->reset_producer();
                }
            }
        }
    }

    // run analysis passes before finalizing the graph
    status_t ret = analyze();
    if (ret != status::success) return ret;

    finalized_ = true;
    return status::success;
}

status_t dnnl_graph_graph::analyze() {
    // run analysis passes before finalizing the graph
    graph::pass::pass_registry_t analysis_pass_reg;
    analysis_pass_reg.register_pass(
            "common", "dag_check_pass", &pass::dag_check_pass_t::create);
    analysis_pass_reg.register_pass("common", "graph_op_depth_check_pass",
            &graph::utils::pm::graph_op_depth_check_pass_t::create);
    graph::pass::pass_manager_t pm(analysis_pass_reg);
    status_t ret = pm.run_passes(*this, "");
    return ret;
}

status_t dnnl_graph_graph::serialize(const std::string &filename) const {
    const auto &fpmath = get_attr().get_fpmath();
    dnnl::impl::verbose_printf(
            "graph,info,serialize graph to a json file %s\n", filename.c_str());
    std::ofstream of(filename);
    utils::json::json_writer_t writer(&of);
    writer.begin_object();
    std::string version = std::to_string(dnnl_version()->major) + "."
            + std::to_string(dnnl_version()->minor) + "."
            + std::to_string(dnnl_version()->patch);
    writer.write_keyvalue("version", version);
    writer.write_keyvalue("engine_kind",
            std::string(utils::engine_kind2str(get_engine_kind())));
    writer.write_keyvalue(
            "fpmath_mode", std::string(utils::fpmath_mode2str(fpmath.mode_)));
    writer.write_keyvalue("fpmath_mode_apply_to_int",
            std::string(fpmath.apply_to_int_ ? "true" : "false"));
    std::vector<size_t> inputs_id;
    inputs_id.reserve(get_input_values().size());
    for (const auto &val : get_input_values()) {
        auto lt = val->get_logical_tensor();
        auto ltw = logical_tensor_wrapper_t(lt);
        inputs_id.push_back(ltw.id());
    }
    writer.write_keyvalue("input_ports", inputs_id);
    std::vector<size_t> outputs_id;
    outputs_id.reserve(get_output_values().size());
    for (const auto &val : get_output_values()) {
        auto lt = val->get_logical_tensor();
        auto ltw = logical_tensor_wrapper_t(lt);
        outputs_id.push_back(ltw.id());
    }
    writer.write_keyvalue("output_ports", outputs_id);
    writer.write_keyvalue("graph", get_ops());
    writer.end_object();
    writer.write_newline();

    return graph::status::success;
}

// Deep copy a graph
std::vector<dnnl_graph_graph::op_ptr> dnnl_graph_graph::deep_copy(
        const std::vector<dnnl_graph_graph::op_ptr> &ops) {
    using op_ptr = dnnl_graph_graph::op_ptr;
    using value_ptr = std::shared_ptr<value_t>;

    std::vector<op_ptr> copied_ops;

    // Create org_op to new_op map
    std::unordered_map<op_ptr, op_ptr> op_map;
    for (const op_ptr &cur_op : ops) {
        // copy the op
        op_ptr copied_op = std::make_shared<op_t>(
                cur_op->get_id(), cur_op->get_kind(), cur_op->get_name());
        copied_op->merge_attributes(cur_op->get_attributes());
        copied_op->set_partition(cur_op->get_partition());

        op_map[cur_op] = copied_op;
        copied_ops.emplace_back(copied_op);
    }

    // Connect the new ops according to org ops
    std::unordered_map<value_ptr, value_ptr> value_map;
    for (const op_ptr &cur_op : ops) {
        op_ptr copied_op = op_map[cur_op];

        for (size_t i = 0; i < cur_op->num_outputs(); i++) {
            auto value = cur_op->get_output_value(i);

            value_ptr copied_value;
            if (value_map.count(value) == 0) {
                copied_value = std::make_shared<value_t>(
                        value->get_logical_tensor(), value->is_internal());
                value_map[value] = copied_value;
            } else {
                copied_value = value_map[value];
            }

            copied_op->add_output(copied_value);
        }

        for (size_t i = 0; i < cur_op->num_inputs(); i++) {
            auto value = cur_op->get_input_value(i);

            value_ptr copied_value;
            if (value_map.count(value) == 0) {
                copied_value = std::make_shared<value_t>(
                        value->get_logical_tensor(), value->is_internal());
                value_map[value] = copied_value;
            } else {
                copied_value = value_map[value];
            }

            copied_op->add_input(copied_value);
            copied_value->add_consumer(*copied_op, i);
        }
    }

    return copied_ops;
}

status_t dnnl_graph_graph_create(graph_t **graph, engine_kind_t engine_kind) {
    *graph = new graph_t(engine_kind);
    return status::success;
}

status_t dnnl_graph_graph_create_with_fpmath_mode(
        graph_t **graph, engine_kind_t engine_kind, fpmath_mode_t fpmath_mode) {
    if (graph == nullptr) return status::invalid_arguments;
    *graph = new graph_t(engine_kind, fpmath_mode);
    return status::success;
}

status_t dnnl_graph_graph_destroy(graph_t *graph) {
    delete graph;
    return status::success;
}

status_t dnnl_graph_graph_set_fpmath_mode(
        dnnl_graph_graph_t graph, dnnl_fpmath_mode_t mode, int apply_to_int) {

    if (graph == nullptr) { return status::invalid_arguments; }

    if (graph->is_finalized()) { return status::invalid_graph; }

    return graph->set_fpmath_mode(mode, apply_to_int);
}

status_t dnnl_graph_graph_get_fpmath_mode(
        dnnl_graph_graph_t graph, dnnl_fpmath_mode_t *mode, int *apply_to_int) {

    if (graph == nullptr) { return status::invalid_arguments; }
    if (graph->is_finalized()) { return status::invalid_graph; }

    const auto &fpmath = graph->get_attr().get_fpmath();
    if (mode) *mode = fpmath.mode_;
    if (apply_to_int) *apply_to_int = fpmath.apply_to_int_;

    return status::success;
}

status_t dnnl_graph_add_op(graph_t *graph, op_t *op) {
    if (graph == nullptr || op == nullptr) { return status::invalid_arguments; }

    if (graph->is_finalized()) { return status::invalid_graph; }

    return graph->add_op(op);
}

status_t dnnl_graph_graph_finalize(graph_t *graph) {
    if (graph == nullptr) return status::invalid_arguments;

    auto ret = graph->finalize();

    return ret;
}

status_t dnnl_graph_graph_is_finalized(graph_t *graph, uint8_t *finalized) {
    if (utils::any_null(graph, finalized)) return status::invalid_arguments;

    *finalized = static_cast<uint8_t>(graph->is_finalized());

    return status::success;
}

status_t dnnl_graph_graph_filter(graph_t *graph, partition_policy_t policy) {
    if (graph == nullptr || (!graph->is_finalized())) {
        return status::invalid_graph;
    }

    // recover the ops to unmatched status and clean the partition_impl list to
    // make the function reentrant
    for (auto &op : graph->get_ops()) {
        op->remove_attr(op_attr::matched);
        op->set_partition(nullptr);
    }
    graph->clean_partitions();

    if (utils::get_graph_dump_mode(graph_dump_mode_t::graph)) {
        // deep copy for graph serialization. note that this is for
        // visualization purpose
        graph_t agraph(*graph);
        dnnl::impl::stringstream_t filename;
        filename << "graph-" << agraph.id() << ".json";
        agraph.serialize(filename.str());
    }

    // Get partition_impl by calling each backends
    std::vector<const backend_t *> &backends
            = backend_registry_t::get_singleton().get_registered_backends();
    for (auto cbkd : backends) {
        if (graph->num_unpartitioned_ops() == 0) break;
        backend_t *bkd = const_cast<backend_t *>(cbkd);
        status_t ret = bkd->get_partitions(*graph, policy);
        if (ret != status::success) return status::invalid_graph;
    }

    // Check the partition_impl
    auto &partition_vec = graph->get_partitions();
    for (auto &p : partition_vec) {
        if (p->get_assigned_backend() == nullptr) {
            return status::invalid_graph;
        }
    }

    return status::success;
}

status_t dnnl_graph_graph_get_partition_num(const graph_t *graph, size_t *num) {
    if (graph == nullptr) { return status::invalid_graph; }
    *num = graph->get_num_partitions();
    return status::success;
}

status_t dnnl_graph_graph_get_partitions(
        graph_t *graph, size_t num, partition_t **partition) {
    if (utils::any_null(graph, partition) || num == 0) {
        return status::invalid_graph;
    }

    // allocate partitions
    for (size_t i = 0; i < num; i++) {
        partition[i] = new partition_t();
    }

    // initialize partitions
    std::vector<partition_t *> partitions {partition, partition + num};
    graph->get_ordered_partitions(partitions);

    if (utils::get_graph_dump_mode(graph_dump_mode_t::graph)) {
        // graph serialization after partitioning. note that this is for
        // visualization purpose
        graph_t agraph(*graph);
        for (auto &aop : agraph.get_ops()) {
            // p_impl is shallow copy
            const auto p_impl = aop->get_partition();
            const auto p_id = p_impl->id();
            const auto *bkd = p_impl->get_assigned_backend();
            const auto &bkd_name = bkd->get_name();
            aop->set_attr<std::string>(
                    op_attr::partition_id, std::to_string(p_id));
            aop->set_attr<std::string>(op_attr::backend, bkd_name);
        }

        dnnl::impl::stringstream_t filename;
        filename << "graph-" << agraph.id() << "-partitioning.json";
        agraph.serialize(filename.str());
    }

    return status::success;
}
