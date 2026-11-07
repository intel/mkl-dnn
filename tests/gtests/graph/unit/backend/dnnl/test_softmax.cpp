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

#include <random>

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/backend/dnnl/ref_func.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_softmax_execute, Softmax) {
    graph::engine_t *eng = get_engine();

    graph::op_t softmax_op(graph::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    std::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    std::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, graph::data_type::f32, graph::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&softmax_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    test_tensor_t src_ts(src, eng, src_data);
    test_tensor_t dst_ts(lt, eng, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(test_softmax_execute, SoftmaxWithLastDim) {
    graph::engine_t *eng = get_engine();

    graph::op_t softmax_op(graph::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(graph::op_attr::axis, -1);

    std::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    std::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, graph::data_type::f32, graph::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&softmax_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    test_tensor_t src_ts(src, eng, src_data);
    test_tensor_t dst_ts(lt, eng, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

static inline void test_softmax_bwd_common(const graph::op_kind_t op_kind,
        std::vector<float> &dst, std::vector<float> &diff_dst,
        std::vector<float> &ref_diff_src, const graph::dims &dims) {
    graph::op_t softmax_bwd_op(op_kind);
    graph::engine_t *eng = get_engine();

    softmax_bwd_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    std::vector<float> diff_src(ref_diff_src.size(), 0.0);

    graph::logical_tensor_t diff_dst_lt;
    // prepare logical tensor
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, dims, graph::data_type::f32);
    diff_dst_lt = utils::logical_tensor_init(1, dims, graph::data_type::f32);
    graph::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, graph::data_type::f32, graph::layout_type::any);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&softmax_bwd_op);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = op_kind == graph::op_kind::SoftMaxBackward
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    std::vector<const graph::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src_lt};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t q_lt, d_lt;
    cp.query_logical_tensor(diff_src_lt.id, &q_lt);
    cp.query_logical_tensor(diff_dst_lt.id, &d_lt);
    ASSERT_EQ(q_lt.layout_type, graph::layout_type::strided);

    for (auto i = 0; i < dst_lt.ndims; i++)
        ASSERT_EQ(q_lt.dims[i], dst_lt.dims[i]);

    test_tensor_t dst_ts(dst_lt, eng, dst);
    test_tensor_t diff_dst_ts(d_lt, eng, diff_dst);
    test_tensor_t diff_src_ts(q_lt, eng, diff_src);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {diff_dst_ts.get(), dst_ts.get()},
                      {diff_src_ts.get()}),
            graph::status::success);
    strm->wait();
    diff_src = diff_src_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(test_softmax_execute, SoftMaxBackward) {
    std::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    std::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    std::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const graph::dims dims {2, 2};

    test_softmax_bwd_common(
            graph::op_kind::SoftMaxBackward, dst, diff_dst, ref_diff_src, dims);
}

TEST(test_softmax_execute, SoftMaxBackwardPlainGrad) {
    std::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    std::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    std::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const graph::dims dims {2, 2};

    test_softmax_bwd_common(
            graph::op_kind::SoftMaxBackward, dst, diff_dst, ref_diff_src, dims);
}

TEST(test_softmax_execute, LogSoftmax) {
    graph::engine_t *eng = get_engine();

    graph::op_t logsoftmax_op(graph::op_kind::LogSoftmax);
    logsoftmax_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    std::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    std::vector<float> ref_dst_data {-0.6931472f, -0.6931472f, -0.6931472f,
            -0.6931472f, -0.6931472f, -0.6931472f};
    std::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, graph::data_type::f32, graph::layout_type::any);

    logsoftmax_op.add_input(src);
    logsoftmax_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&logsoftmax_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("logsoftmax_pass");
    ASSERT_TRUE(apass);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    test_tensor_t src_ts(src, eng, src_data);
    test_tensor_t dst_ts(lt, eng, dst_data);

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    ASSERT_TRUE(allclose(dst_data, ref_dst_data, 1e-5f, 1e-8f));
}

TEST(test_softmax_execute, LogSoftmaxBackward) {
    std::vector<float> dst {-0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    std::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    std::vector<float> ref_diff_src {-2, 2, -1, 1};

    const graph::dims dims {2, 2};

    test_softmax_bwd_common(graph::op_kind::LogSoftmaxBackward, dst, diff_dst,
            ref_diff_src, dims);
}

TEST(test_softmax_execute, LogSoftmaxBackwardPlainGrad) {
    std::vector<float> dst {-0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    std::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    std::vector<float> ref_diff_src {-2, 2, -1, 1};

    const graph::dims dims {2, 2};

    test_softmax_bwd_common(graph::op_kind::LogSoftmaxBackward, dst, diff_dst,
            ref_diff_src, dims);
}

static inline void test_softmax_bwd_get_inplace_pair_common(
        graph::op_kind_t op_kind) {
    graph::engine_t *eng = get_engine();
    graph::op_t softmax_bwd_op(op_kind);

    softmax_bwd_op.set_attr<int64_t>(graph::op_attr::axis, 1);

    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, {2, 2}, graph::data_type::f32);
    graph::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {2, 2}, graph::data_type::f32);
    graph::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(2, {2, 2}, graph::data_type::f32);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&softmax_bwd_op);
    g.finalize();

    graph::pass::pass_base_ptr apass
            = op_kind == graph::op_kind::SoftMaxBackward
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const graph::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<graph::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1U);
    ASSERT_EQ(inplace_pairs[0].input_id, diff_dst_lt.id);
    ASSERT_EQ(inplace_pairs[0].output_id, diff_src_lt.id);
}

TEST(test_softmax_compile, SoftMaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(graph::op_kind::SoftMaxBackward);
}

TEST(test_softmax_compile, LogSoftmaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(
            graph::op_kind::LogSoftmaxBackward);
}

TEST(test_softmax_compile, SoftmaxAdd) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    std::vector<int64_t> softmax_shape {2, 2, 2};
    std::vector<float> src_data(product(softmax_shape));
    std::vector<float> src1_data(product(softmax_shape), 0.5f);

    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.f, 1.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });

    graph::op_t softmax(0, graph::op_kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(graph::op_attr::axis, 2);
    graph::op_t add(1, graph::op_kind::Add, "add");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, softmax_shape, graph::data_type::f32);
    graph::logical_tensor_t softmax_dst = utils::logical_tensor_init(
            1, softmax_shape, graph::data_type::f32);
    graph::logical_tensor_t add_src1 = utils::logical_tensor_init(
            2, softmax_shape, graph::data_type::f32);
    graph::logical_tensor_t add_dst = utils::logical_tensor_init(
            3, softmax_shape, graph::data_type::f32);

    softmax.add_input(src);
    softmax.add_output(softmax_dst);
    add.add_input(softmax_dst);
    add.add_input(add_src1);
    add.add_output(add_dst);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&softmax), graph::status::success);
    ASSERT_EQ(g.add_op(&add), graph::status::success);
    ASSERT_EQ(g.finalize(), graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("fp_softmax_post_ops");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&src, &add_src1};
    std::vector<const graph::logical_tensor_t *> lt_outs {&add_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, engine), graph::status::success);

    std::vector<float> dst_data(product(softmax_shape));
    std::vector<float> ref_data(product(softmax_shape));
    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t src1_ts(add_src1, engine, src1_data);
    test_tensor_t dst_ts(add_dst, engine, dst_data);
    test_tensor_t ref_ts(add_dst, engine, ref_data);

    ASSERT_EQ(run_graph(g, {src_ts, src1_ts}, {ref_ts}, *engine, *strm),
            graph::status::success);
    ASSERT_EQ(cp.execute(strm, {src_ts.get(), src1_ts.get()}, {dst_ts.get()}),
            graph::status::success);
    strm->wait();
    dst_data = dst_ts.as_vec_type<float>();
    ref_data = ref_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_FLOAT_EQ(ref_data[i], dst_data[i]);
    }
}

TEST(test_softmax_compile, SoftmaxStatsInvalidAxis) {
    graph::engine_t *eng = get_engine();

    for (int64_t axis : {-3, 2, 1000}) {
        graph::op_t softmax_op(graph::op_kind::SoftMax);
        softmax_op.set_attr<int64_t>(graph::op_attr::axis, axis);

        graph::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                1, graph::data_type::f32, graph::layout_type::strided);
        graph::logical_tensor_t stats = utils::logical_tensor_init(
                2, graph::data_type::f32, graph::layout_type::strided);

        softmax_op.add_input(src);
        softmax_op.add_output(dst);
        softmax_op.add_output(stats);

        graph::graph_t g(eng->kind());
        g.add_op(&softmax_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("softmax_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];
        graph::partition_t p;
        p.init(part);

        std::vector<const graph::logical_tensor_t *> inputs {&src};
        std::vector<const graph::logical_tensor_t *> outputs {&dst, &stats};

        graph::compiled_partition_t cp(p);
        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng),
                graph::status::invalid_shape);
    }
}

TEST(test_softmax_compile, SoftmaxGetInplacePair) {
    graph::engine_t *eng = get_engine();

    graph::op_t softmax_op(graph::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>(graph::op_attr::axis, 0);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, graph::data_type::f32);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&softmax_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1U);
    ASSERT_EQ(pairs[0].input_id, 0U);
    ASSERT_EQ(pairs[0].output_id, 1U);
}
