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

#include "gtest/gtest.h"

#include "interface/op.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"

using namespace dnnl::impl::graph;
using namespace dnnl::graph::tests::unit::utils;

TEST(test_partition, CreateSimple) {
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
            engine_kind::cpu, attr, partition_kind_t::undef);
    ASSERT_EQ(p.get_ops().size(), 0U);
    ASSERT_EQ(p.get_attr().get_fpmath().mode_, fpmath_mode::strict);
    ASSERT_EQ(p.get_kind(), partition_kind_t::undef);
}

TEST(test_partition, AddOps) {
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    for (const auto &engine_kind : engine_kinds) {
        dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
                engine_kind, attr, partition_kind_t::undef);
        size_t id = 100;
        std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
        p.add_op(n);
        ASSERT_EQ(p.get_ops().size(), 1U);

        std::vector<size_t> ids {101, 102};
        std::vector<std::shared_ptr<op_t>> ops;
        for (auto id : ids) {
            ops.emplace_back(new op_t(id, op_kind::Wildcard, "Wildcard"));
            p.add_op(ops.back());
        }

        ASSERT_EQ(p.get_ops().size(), 3U);
    }
}

TEST(test_partition, GetOps) {
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
            engine_kind::cpu, attr, partition_kind_t::undef);
    size_t id = 100;
    std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
    p.add_op(n);
    auto ops = p.get_ops();
    ASSERT_EQ(ops.size(), 1U);
    ASSERT_EQ(ops[0]->get_id(), 100U);
}

TEST(test_partition, Init) {
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    for (const auto &engine_kind : engine_kinds) {
        dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
                engine_kind, attr, partition_kind_t::undef);
        std::shared_ptr<op_t> n(new op_t(0, op_kind::Convolution, "Conv"));
        n->set_attr<int64_t>(op_attr::groups, 0);
        p.add_op(n);
        ASSERT_FALSE(p.is_initialized());
        ASSERT_TRUE(p.get_assigned_backend()->get_name() != "fake_backend");
    }
}

TEST(test_partition, Clone) {
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    for (const auto &engine_kind : engine_kinds) {
        dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
                engine_kind, attr, partition_kind_t::convolution_post_ops);
        auto n = std::make_shared<op_t>(op_kind::Convolution);
        n->set_attr<int64_t>(op_attr::groups, 1);

        p.add_op(n); // the subgraph

        ASSERT_FALSE(p.is_initialized());
        ASSERT_TRUE(p.get_assigned_backend()->get_name() == "dnnl_backend");
        ASSERT_EQ(p.get_ops()[0]->get_kind(), op_kind::Convolution);
        ASSERT_TRUE(p.get_ops()[0]->has_attr(op_attr::groups));
        ASSERT_EQ(p.get_ops()[0]->get_attr<int64_t>(op_attr::groups), 1);

        // clone the partition
        auto p_copy = std::dynamic_pointer_cast<
                dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t>(p.clone());
        ASSERT_NE(p_copy, nullptr);
        ASSERT_FALSE(p_copy->is_initialized());
        ASSERT_TRUE(
                p_copy->get_assigned_backend()->get_name() == "dnnl_backend");
        ASSERT_EQ(p_copy->get_ops()[0]->get_kind(), op_kind::Convolution);
        ASSERT_TRUE(p_copy->get_ops()[0]->has_attr(op_attr::groups));
        ASSERT_EQ(p_copy->get_ops()[0]->get_attr<int64_t>(op_attr::groups), 1);
    }
}

TEST(test_partition_op, AssignedPartition) {
    using namespace dnnl::impl::graph;

    op_t conv {0, op_kind::Convolution, std::string("convolution")};

    ASSERT_EQ(conv.get_partition(), nullptr);
    const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
    auto part = std::make_shared<
            dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t>(
            engine_kind::cpu, attr, partition_kind_t::convolution_post_ops);
    conv.set_partition(part.get());
    ASSERT_EQ(conv.get_partition(), part.get());
}

TEST(test_partition, SetFpmathMode) {
    engine_t *eng = get_engine();
    for (auto m : {fpmath_mode::strict, fpmath_mode::bf16, fpmath_mode::f16,
                 fpmath_mode::any}) {
        const graph::graph_attr_t attr {{m, false}};
        dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t p(
                eng->kind(), attr, partition_kind_t::undef);
        ASSERT_EQ(p.get_attr().get_fpmath().mode_, m);
    }
}

TEST(test_partition, InferShape) {
    std::vector<engine_kind_t> engine_kinds
            = {engine_kind::cpu, engine_kind::gpu};
    for (const auto &engine_kind : engine_kinds) {
        size_t id = 0;

        graph::logical_tensor_t lt1
                = logical_tensor_init(id++, graph::data_type::f32);
        graph::logical_tensor_t lt2
                = logical_tensor_init(id++, graph::data_type::f32);
        graph::logical_tensor_t lt3
                = logical_tensor_init(id++, graph::data_type::f32);

        std::vector<const graph::logical_tensor_t *> inputs {&lt1, &lt2};
        std::vector<graph::logical_tensor_t *> outputs {&lt3};
        const graph::graph_attr_t attr {{fpmath_mode::strict, false}};
        auto par = std::make_shared<
                dnnl::impl::graph::dnnl_impl::dnnl_partition_impl_t>(
                engine_kind, attr, graph::partition_kind_t::undef);
        ASSERT_EQ(par->infer_shape(inputs, outputs), graph::status::success);
    }
}
