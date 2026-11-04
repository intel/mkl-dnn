/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "interface/c_types_map.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/layout_propagator.hpp"

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace dnnl_impl = graph::dnnl_impl;

TEST(test_layout_propagator, LayoutPropagatorForPermute) {
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    op->set_attr<std::vector<int64_t>>(graph::op_attr::permutation, {1, 0});
    auto lt_in = utils::logical_tensor_init(
            0, {1, 1}, graph::data_type::f32, graph::layout_type::any);
    auto lt_out = utils::logical_tensor_init(1, {1, 1}, graph::data_type::f32);

    op->add_input(lt_in);
    op->add_output(lt_out);

    graph::engine_t &eng = *get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::pd_cache_t pd_cache;
    bool use_block_layout = false;
    const graph::graph_attr_t graph_attr {{graph::fpmath_mode::any, false}};
    auto sg = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<std::shared_ptr<graph::op_t>> {op}, p_engine,
            graph_attr, false, false);
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
    ASSERT_EQ(dnnl_impl::layout_propagator_for_permute(op, p_engine, pd_cache,
                      graph_attr, use_block_layout, rewriter),
            graph::status::success);
}

TEST(test_layout_propagator, LayoutPropagatorForReorder) {
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    op->set_attr<std::vector<int64_t>>(graph::op_attr::permutation, {1, 0});
    auto lt_in = utils::logical_tensor_init(
            0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
    auto lt_out = utils::logical_tensor_init(
            1, {1, 2}, graph::data_type::f32, graph::layout_type::strided);

    op->add_input(lt_in);
    op->add_output(lt_out);

    graph::engine_t &eng = *get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::pd_cache_t pd_cache;
    bool use_block_layout = false;
    const graph::graph_attr_t graph_attr {{graph::fpmath_mode::any, false}};
    auto sg = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<std::shared_ptr<graph::op_t>> {op}, p_engine,
            graph_attr, false, false);
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
    ASSERT_EQ(layout_propagator_for_reorder(op, p_engine, pd_cache, graph_attr,
                      use_block_layout, rewriter),
            graph::status::success);
    // See: DUMMY_PARALLEL.
    dnnl::impl::parallel(1, [](int, int) {});
}

TEST(test_layout_propagator, LayoutPropagatorForSumDeathTest) {
    graph::engine_t &eng = *get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::pd_cache_t pd_cache;
    bool use_block_layout = false;
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    auto lt_in = utils::logical_tensor_init(
            1, {1, 2}, graph::data_type::f32, graph::layout_type::strided);
    auto lt_out = utils::logical_tensor_init(
            2, {1, 2}, graph::data_type::f32, graph::layout_type::any);

    op->add_input(lt_in);
    op->add_output(lt_out);
    const graph::graph_attr_t graph_attr {{graph::fpmath_mode::any, false}};
    auto sg = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<std::shared_ptr<graph::op_t>> {op}, p_engine,
            graph_attr, false, false);
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
    ASSERT_EQ(layout_propagator_for_sum(op, p_engine, pd_cache, graph_attr,
                      use_block_layout, rewriter),
            graph::status::success);
}

TEST(test_layout_propagator, LayoutPropagatorForSumFailDeathTest) {
    dnnl::engine p_engine;
    dnnl_impl::pd_cache_t pd_cache;
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    auto lt_in = utils::logical_tensor_init(
            0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
    auto lt_out = utils::logical_tensor_init(
            1, {1, 2}, graph::data_type::f32, graph::layout_type::strided);

    op->add_input(lt_in);
    op->add_output(lt_out);
    std::shared_ptr<dnnl_impl::subgraph_t> sg;
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
#ifndef NDEBUG
    graph::graph_attr_t graph_attr;
    EXPECT_DEATH(layout_propagator_for_sum(
                         op, p_engine, pd_cache, graph_attr, false, rewriter),
            "input format of sum primitive cannot be any.");
#endif
}

TEST(test_layout_propagator, LayoutPropagatorForSubZpsDeathTest) {
    graph::engine_t &eng = *get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::pd_cache_t pd_cache;
    bool use_block_layout = false;
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    const graph::graph_attr_t graph_attr {{graph::fpmath_mode::any, false}};
    auto sg = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<std::shared_ptr<graph::op_t>> {op}, p_engine,
            graph_attr, false, false);
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
#ifndef NDEBUG
    EXPECT_DEATH(dnnl_impl::layout_propagator_for_sub_zps(op, p_engine,
                         pd_cache, graph_attr, use_block_layout, rewriter),
            "");
#else
    ASSERT_EQ(dnnl_impl::layout_propagator_for_sub_zps(op, p_engine, pd_cache,
                      graph_attr, use_block_layout, rewriter),
            graph::status::invalid_graph_op);
#endif
}

TEST(test_layout_propagator, LayoutPropagatorForAddZpsDeathTest) {
    graph::engine_t &eng = *get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::pd_cache_t pd_cache;
    bool use_block_layout = false;
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    const graph::graph_attr_t graph_attr {{graph::fpmath_mode::any, false}};
    auto sg = std::make_shared<dnnl_impl::subgraph_t>(
            std::vector<std::shared_ptr<graph::op_t>> {op}, p_engine,
            graph_attr, false, false);
    dnnl_impl::subgraph_rewriter_t rewriter {sg};
#ifndef NDEBUG
    EXPECT_DEATH(dnnl_impl::layout_propagator_for_add_zps(op, p_engine,
                         pd_cache, graph_attr, use_block_layout, rewriter),
            "");
#else
    ASSERT_EQ(dnnl_impl::layout_propagator_for_add_zps(op, p_engine, pd_cache,
                      graph_attr, use_block_layout, rewriter),
            graph::status::invalid_graph_op);
#endif
}
