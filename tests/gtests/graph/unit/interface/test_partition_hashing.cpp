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
#include <thread>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/partition_hashing.hpp"
#include "interface/shape_infer.hpp"

#include "graph/unit/unit_test_common.hpp"

namespace graph = dnnl::impl::graph;

TEST(test_interface_partition_hashing, ThreadId) {
    graph::engine_t &engine = *get_engine();
    graph::partition_hashing::key_t key {&engine, {}, {}, {}, {}};
    ASSERT_EQ(std::this_thread::get_id(), key.thread_id());
    // See: DUMMY_PARALLEL.
    dnnl::impl::parallel(1, [](int, int) {});
}

TEST(test_interface_partition_hashing, GetArrayHash) {
    size_t seed = 10000;
    const size_t num = 3;
    float arr[num] {1.0f, 2.0f, 3.0f};
    EXPECT_NO_FATAL_FAILURE(
            graph::partition_hashing::get_array_hash(seed, arr, num));
}

TEST(test_interface_partition_hashing, GetOpHash) {
    graph::op_t op {0, graph::op_kind::Wildcard, "wildcard"};
    ASSERT_NO_THROW(graph::partition_hashing::get_op_hash(op));
}

TEST(test_interface_partition_hashing, GraphAttributesHash) {
    const graph::graph_attr_t default_attr;
    const graph::graph_attr_t strict_attr {{graph::fpmath_mode::strict, false}};
    const graph::graph_attr_t bf16_attr {{graph::fpmath_mode::bf16, false}};
    const graph::graph_attr_t int_attr {{graph::fpmath_mode::strict, true}};

    EXPECT_TRUE(default_attr == strict_attr);
    EXPECT_EQ(default_attr.hash(), strict_attr.hash());
    EXPECT_FALSE(strict_attr == bf16_attr);
    EXPECT_NE(strict_attr.hash(), bf16_attr.hash());
    EXPECT_FALSE(strict_attr == int_attr);
    EXPECT_NE(strict_attr.hash(), int_attr.hash());
}

TEST(test_interface_partition_hashing, GraphAttributesKey) {
    graph::engine_t &engine = *get_engine();
    const graph::graph_attr_t default_attr;
    const graph::graph_attr_t strict_attr {{graph::fpmath_mode::strict, false}};
    const graph::graph_attr_t bf16_attr {{graph::fpmath_mode::bf16, false}};
    const graph::graph_attr_t int_attr {{graph::fpmath_mode::strict, true}};
    const graph::partition_hashing::key_t default_key {
            &engine, {}, {}, {}, default_attr};
    const graph::partition_hashing::key_t strict_key {
            &engine, {}, {}, {}, strict_attr};
    const graph::partition_hashing::key_t bf16_key {
            &engine, {}, {}, {}, bf16_attr};
    const graph::partition_hashing::key_t int_key {
            &engine, {}, {}, {}, int_attr};
    const std::hash<graph::partition_hashing::key_t> hash;

    EXPECT_TRUE(default_key == strict_key);
    EXPECT_EQ(hash(default_key), hash(strict_key));
    EXPECT_FALSE(strict_key == bf16_key);
    EXPECT_NE(hash(strict_key), hash(bf16_key));
    EXPECT_FALSE(strict_key == int_key);
    EXPECT_NE(hash(strict_key), hash(int_key));
    // See: DUMMY_PARALLEL.
    dnnl::impl::parallel(1, [](int, int) {});
}
