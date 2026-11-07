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

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_concat_compile, ConcatWithMoreInputsFail) {
    size_t num_inputs = 65;
    const std::vector<graph::dim_t> src_dims = {1, 2, 2, 2};
    const std::vector<graph::dim_t> dst_dims
            = {1, 2, 2, 2 * static_cast<graph::dim_t>(num_inputs)};
    std::vector<float> dst_data(dst_dims.size(), 0.);
    const auto axis = 3;

    graph::op_t concat_op(graph::op_kind::Concat, "concat");
    concat_op.set_attr<int64_t>(graph::op_attr::axis, axis);

    std::vector<graph::logical_tensor_t> input_lts;
    for (size_t i = 0; i < num_inputs; ++i) {
        graph::logical_tensor_t src_lt = utils::logical_tensor_init(i, src_dims,
                graph::data_type::f32, graph::layout_type::strided);
        input_lts.push_back(src_lt);
        concat_op.add_input(input_lts[i]);
    }
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(num_inputs,
            dst_dims, graph::data_type::f32, graph::layout_type::strided);
    concat_op.add_output(dst_lt);

    auto *eng = get_engine();
    graph::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&concat_op), graph::status::invalid_graph_op);
}
