/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "graph/interface/partition.hpp"
#include "graph/interface/partition_hashing.hpp"

#include "common/dnnl_thread.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace partition_hashing {

key_t::key_t(const impl::engine_t *engine,
        const std::vector<std::shared_ptr<op_t>> &ops,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs,
        const impl::graph::graph_attr_t &attr)
    : ops_(get_raw_ptrs(ops))
    , nthread_(dnnl_get_max_threads())
    // Here we use engine as a member of partition_hashing key_t, because for
    // CPU engine and nativa runtime, the engine_id is nullptr for all engine
    // instances. The compiled partition would be hit under different engine
    // instances. For example, first compile with engine 1 -> execute with
    // engine1 -> second compile with engine 2 (cache hit) -> execute with
    // engine 2(fail). So we need to use engine as a member of key_t to avoid
    // execution crash.
    , engine_(engine)
    , attr_(attr)
    , thread_id_(std::this_thread::get_id()) {
    ins_.reserve(ins.size());
    outs_.reserve(outs.size());
    for (auto &in : ins) {
        ins_.emplace_back(*in);
    }
    for (auto &out : outs) {
        outs_.emplace_back(*out);
    }
}

key_t::key_t(const partition_t *partition, const impl::engine_t *engine,
        const std::vector<const logical_tensor_t *> &ins,
        const std::vector<const logical_tensor_t *> &outs)
    : key_t(engine, partition->get_ops(), ins, outs, partition->get_attr()) {}

bool key_t::operator==(const key_t &rhs) const {
    if (this == &rhs) return true;

    const size_t lhs_num_ops = ops_.size();
    const size_t rhs_num_ops = rhs.ops_.size();
    const size_t lhs_num_ins = ins_.size();
    const size_t rhs_num_ins = rhs.ins_.size();
    const size_t lhs_num_outs = outs_.size();
    const size_t rhs_num_outs = rhs.outs_.size();

    bool ret = true && lhs_num_ops == rhs_num_ops && lhs_num_ins == rhs_num_ins
            && lhs_num_outs == rhs_num_outs && nthread_ == rhs.nthread_
            && engine_ == rhs.engine_ && attr_ == rhs.attr_;
    if (!ret) return false;

    for (size_t i = 0; i < lhs_num_ops; ++i) {
        const op_t *op = ops_[i];
        if (std::find_if(rhs.ops_.begin(), rhs.ops_.end(),
                    [op](const op_t *rhs_op) { return *op == *rhs_op; })
                == rhs.ops_.end())
            return false;
    }

    for (size_t i = 0; i < lhs_num_ins; ++i) {
        const logical_tensor_wrapper_t lhs_lt {ins_[i]};
        if (std::find_if(rhs.ins_.begin(), rhs.ins_.end(),
                    [&lhs_lt](const logical_tensor_t &rhs_lt) {
            return logical_tensor_wrapper_t(rhs_lt) == lhs_lt;
        }) == rhs.ins_.end())
            return false;
    }

    for (size_t i = 0; i < lhs_num_outs; ++i) {
        const logical_tensor_wrapper_t lhs_lt {outs_[i]};
        if (std::find_if(rhs.outs_.begin(), rhs.outs_.end(),
                    [&lhs_lt](const logical_tensor_t &rhs_lt) {
            return logical_tensor_wrapper_t(rhs_lt) == lhs_lt;
        }) == rhs.outs_.end())
            return false;
    }

    return true;
}

size_t get_logical_tensors_hash(
        const std::vector<std::shared_ptr<value_t>> &values) {
    size_t seed = 0;
    for (auto &value : values) {
        seed = hash_combine(seed,
                logical_tensor_wrapper_t(value->get_logical_tensor()).hash());
    }
    return seed;
}

using attribute_value_t = dnnl::impl::graph::utils::attribute_value_t;

size_t get_attributes_hash(
        const std::unordered_map<op_attr_t, attribute_value_t> &attributes) {
    size_t seed = 0;
    for (auto &attr : attributes) {
        seed = hash_combine(seed, static_cast<size_t>(attr.first));
        auto kind = attr.second.get_kind();
        try {
            switch (kind) {
                case attribute_kind::f:
                    seed = hash_combine(seed, attr.second.get<float>());
                    break;
                case attribute_kind::i:
                    seed = hash_combine(seed, attr.second.get<int64_t>());
                    break;
                case attribute_kind::s:
                    seed = hash_combine(seed, attr.second.get<std::string>());
                    break;
                case attribute_kind::b:
                    seed = hash_combine(seed, attr.second.get<bool>());
                    break;
                case attribute_kind::fs:
                    seed = get_array_hash(seed,
                            attr.second.get<std::vector<float>>().data(),
                            attr.second.get<std::vector<float>>().size());
                    break;
                case attribute_kind::is:
                    seed = get_array_hash(seed,
                            attr.second.get<std::vector<int64_t>>().data(),
                            attr.second.get<std::vector<int64_t>>().size());
                    break;
                default: break;
            }
        } catch (...) { assert(!"should not reach here"); }
    }
    return seed;
}

size_t get_op_hash(const op_t &op) {
    size_t seed = 0;
    seed = hash_combine(seed, op.get_id());
    seed = hash_combine(seed, static_cast<size_t>(op.get_kind()));
    seed = hash_combine(seed, get_logical_tensors_hash(op.get_input_values()));
    seed = hash_combine(seed, get_logical_tensors_hash(op.get_output_values()));
    seed = hash_combine(seed, get_attributes_hash(op.get_attributes()));
    return seed;
}

} // namespace partition_hashing
} // namespace graph
} // namespace impl
} // namespace dnnl
