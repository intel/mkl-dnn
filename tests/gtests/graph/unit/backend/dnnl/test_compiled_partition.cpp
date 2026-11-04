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

#include "interface/partition.hpp"
#include "interface/tensor.hpp"

#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/kernels/pool.hpp"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_compiled_partition, Relu) {
    graph::engine_t *eng = get_engine();

    graph::op_t relu_op(graph::op_kind::ReLU, "relu");

    const graph::logical_tensor_t lt_in = utils::logical_tensor_init(
            /* tid= */ 1, {1, 1, 3, 3}, graph::data_type::f32);
    const graph::logical_tensor_t lt_out
            = utils::logical_tensor_init(/* tid= */ 2, {1, 1, 3, 3},
                    graph::data_type::f32, graph::layout_type::any);

    relu_op.add_input(lt_in);
    relu_op.add_output(lt_out);

    graph::graph_t g(eng->kind());
    g.add_op(&relu_op);
    g.finalize();
    run_all_passes(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.id(), cp.src_partition().id());

    std::vector<const graph::logical_tensor_t *> lt_inputs {&lt_in};
    std::vector<const graph::logical_tensor_t *> lt_outputs {&lt_out};
    graph::status_t status = p.compile(&cp, lt_inputs, lt_outputs, eng);
    ASSERT_EQ(status, graph::status::success);
    graph::logical_tensor_t query_in_lt, query_out_lt;
    graph::status_t status_query
            = cp.query_logical_tensor(lt_out.id, &query_out_lt);
    ASSERT_EQ(status_query, graph::status::success);
    ASSERT_EQ(query_out_lt.layout_type, graph::layout_type::strided);

    size_t size_in = 0, size_out = 0;
    cp.query_logical_tensor(lt_in.id, &query_in_lt);
    size_in = graph::logical_tensor_wrapper_t(query_in_lt).size();
    size_out = graph::logical_tensor_wrapper_t(query_out_lt).size();
    ASSERT_EQ(size_in, 9 * sizeof(float));
    ASSERT_EQ(size_in, size_out);

    size_t ele_num_in = size_in / sizeof(float);
    std::vector<float> data_in(ele_num_in);
    std::vector<float> data_out(ele_num_in);
    for (size_t i = 0; i < ele_num_in; i++) {
        data_in[i] = static_cast<float>(i) - static_cast<float>(ele_num_in / 2);
    }

    test_tensor_t t_in(lt_in, eng, data_in), t_out(query_out_lt, eng, data_out);

    std::vector<graph::tensor_t> t_inputs, t_outputs;
    t_inputs.emplace_back(t_in.get());
    t_outputs.emplace_back(t_out.get());

    graph::stream_t *strm = get_stream();
    EXPECT_SUCCESS(cp.execute(strm, t_inputs, t_outputs));
    strm->wait();

    std::unique_ptr<float[]> ref_out(new float[ele_num_in]);
    for (size_t i = 0; i < ele_num_in; i++) {
        ref_out[i] = (i < ele_num_in / 2)
                ? 0.0f
                : static_cast<float>(i) - static_cast<float>(ele_num_in / 2);
    }
    data_out = t_out.as_vec_type<float>();
    for (size_t i = 0; i < ele_num_in; i++) {
        ASSERT_FLOAT_EQ(ref_out[i], data_out[i]);
    }
}

TEST(test_compiled_partition, SearchRequiredInputsOutputs) {
    graph::engine_t *eng = get_engine();

    graph::op_t relu_op(graph::op_kind::ReLU, "relu");

    graph::logical_tensor_t lt_in = utils::logical_tensor_init(
            /* tid= */ 1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_out = utils::logical_tensor_init(/* tid= */ 2,
            {1, 1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    relu_op.add_input(lt_in);
    relu_op.add_output(lt_out);

    graph::graph_t g(eng->kind());
    g.add_op(&relu_op);
    g.finalize();

    run_all_passes(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.id(), cp.src_partition().id());

    graph::logical_tensor_t lt_in_additional1 = utils::logical_tensor_init(
            /* tid= */ 3, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_in_additional2 = utils::logical_tensor_init(
            /* tid= */ 4, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_out_additional1 = utils::logical_tensor_init(
            /* tid= */ 5, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_out_additional2 = utils::logical_tensor_init(
            /* tid= */ 6, {1, 1, 3, 3}, graph::data_type::f32);

    // in/outputs list have to contain required logical tensor
    std::vector<const graph::logical_tensor_t *> lt_inputs_wrong {
            &lt_in_additional1, &lt_in_additional2}; // no required
    std::vector<const graph::logical_tensor_t *> lt_outputs_wrong {
            &lt_out_additional1, &lt_out_additional2}; // no required

    // compile function return a miss_ins_outs error, since it can't find
    // required inputs and outputs from the given arguments
    graph::status_t status
            = p.compile(&cp, lt_inputs_wrong, lt_outputs_wrong, eng);
    ASSERT_EQ(status, graph::status::invalid_arguments);

    // in/outputs list can contain more logical tensors than required
    std::vector<const graph::logical_tensor_t *> lt_inputs_correct {
            &lt_in_additional1, /* required */ &lt_in, &lt_in_additional2};
    std::vector<const graph::logical_tensor_t *> lt_outputs_correct {
            &lt_out_additional1, &lt_out_additional2, /* required */ &lt_out};

    // compile function will search its required inputs and outputs by itself
    status = p.compile(&cp, lt_inputs_correct, lt_outputs_correct, eng);
    ASSERT_EQ(status, graph::status::success);

    //query logical_tensor to get its layout
    graph::logical_tensor_t query_lt_in, query_lt_out;
    ASSERT_EQ(cp.query_logical_tensor(lt_out.id, &query_lt_out),
            graph::status::success);
    ASSERT_EQ(query_lt_out.layout_type, graph::layout_type::strided);

    size_t size_in = 0, size_out = 0;
    cp.query_logical_tensor(lt_in.id, &query_lt_in);
    size_in = graph::logical_tensor_wrapper_t(query_lt_in).size();
    size_out = graph::logical_tensor_wrapper_t(query_lt_out).size();
    ASSERT_EQ(size_in, 9 * sizeof(float));
    ASSERT_EQ(size_in, size_out);

    size_t ele_num_in = size_in / sizeof(float);
    size_t ele_num_out = size_out / sizeof(float);
    std::vector<float> data_in(ele_num_in);
    std::vector<float> data_out(ele_num_out);
    for (size_t i = 0; i < ele_num_in; i++) {
        data_in[i] = static_cast<float>(i) - static_cast<float>(ele_num_in / 2);
    }

    test_tensor_t t_in(lt_in, eng, data_in), t_out(query_lt_out, eng, data_out);
    test_tensor_t t_in_additional1(lt_in_additional1, eng),
            t_in_additional2(lt_in_additional2, eng);

    test_tensor_t t_out_additional1(lt_out_additional1, eng),
            t_out_additional2(lt_out_additional2, eng);

    // when submit, in/outputs tensor's order must be same as compile
    // funcstion's in/outputs logical tensor
    std::vector<test_tensor_t> t_inputs_correct {
            t_in_additional1, t_in, t_in_additional2};
    std::vector<test_tensor_t> t_outputs_correct {
            t_out_additional1, t_out_additional2, t_out};

    graph::stream_t *strm = get_stream();
    EXPECT_SUCCESS(
            cp.execute(strm, test_tensor_t::to_graph_tensor(t_inputs_correct),
                    test_tensor_t::to_graph_tensor(t_outputs_correct)));
    strm->wait();

    std::vector<float> ref_out(ele_num_in);
    for (size_t i = 0; i < ele_num_in; i++) {
        ref_out[i] = (i < ele_num_in / 2)
                ? 0.0f
                : static_cast<float>(i) - static_cast<float>(ele_num_in / 2);
    }

    data_out = t_out.as_vec_type<float>();
    for (size_t i = 0; i < ele_num_in; i++) {
        ASSERT_FLOAT_EQ(ref_out[i], data_out[i]);
    }
}

TEST(test_compiled_partition, AllowRepeatedInputs) {
    graph::engine_t *eng = get_engine();

    graph::op_t n(graph::op_kind::Multiply);

    graph::logical_tensor_t lt_in1 = utils::logical_tensor_init(
            /* tid= */ 1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_out = utils::logical_tensor_init(/* tid= */ 2,
            {1, 1, 3, 3}, graph::data_type::f32, graph::layout_type::any);

    // repeated inputs
    n.add_input(lt_in1);
    n.add_input(lt_in1);
    n.add_output(lt_out);

    graph::graph_t g(eng->kind());
    g.add_op(&n);
    g.finalize();
    run_all_passes(g);

    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    // only one input
    std::vector<const graph::logical_tensor_t *> lt_ins {&lt_in1};
    std::vector<const graph::logical_tensor_t *> lt_outs {&lt_out};

    graph::status_t status = p.compile(&cp, lt_ins, lt_outs, eng);
    ASSERT_EQ(status, graph::status::success);

    graph::logical_tensor_t query_lt_out;
    ASSERT_EQ(cp.query_logical_tensor(lt_out.id, &query_lt_out),
            graph::status::success);
    ASSERT_EQ(query_lt_out.layout_type, graph::layout_type::strided);

    size_t size_in = 0, size_out = 0;
    size_in = graph::logical_tensor_wrapper_t(lt_in1).size();
    size_out = graph::logical_tensor_wrapper_t(query_lt_out).size();
    ASSERT_EQ(size_in, 9 * sizeof(float));
    ASSERT_EQ(size_in, size_out);

    std::vector<float> data_in {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> data_out(data_in.size());
    std::vector<float> ref_out {
            1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f, 81.0f};

    test_tensor_t t_in1(lt_in1, eng, data_in);
    test_tensor_t t_out(query_lt_out, eng, data_out);

    // only one input
    std::vector<test_tensor_t> t_ins {t_in1};
    std::vector<test_tensor_t> t_outs {t_out};

    graph::stream_t *strm = get_stream();
    EXPECT_SUCCESS(cp.execute(strm, test_tensor_t::to_graph_tensor(t_ins),
            test_tensor_t::to_graph_tensor(t_outs)));
    strm->wait();
    data_out = t_out.as_vec_type<float>();
    for (size_t i = 0; i < ref_out.size(); i++) {
        ASSERT_FLOAT_EQ(ref_out[i], data_out[i]);
    }
}

TEST(test_compiled_partition, GetAndInfoMethod) {
    using ltw = graph::logical_tensor_wrapper_t;

    graph::engine_t &engine = *get_engine();
    size_t id = 0;

    graph::logical_tensor_t lt1 = utils::logical_tensor_init(
            id++, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t lt2 = utils::logical_tensor_init(
            id++, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t lt3 = utils::logical_tensor_init(
            id++, graph::data_type::f32, graph::layout_type::any);

    std::vector<graph::logical_tensor_t> inputs {lt1, lt2};
    std::vector<graph::logical_tensor_t> outputs {lt3};
    graph::dnnl_impl::kernel_ptr kernel
            = std::make_shared<graph::dnnl_impl::float_pooling_fwd>();
    auto cp_impl = std::make_shared<
            graph::dnnl_impl::dnnl_compiled_partition_impl_t>(
            &engine, inputs, outputs, kernel);
    graph::partition_t par;
    const graph::graph_attr_t attr {{graph::fpmath_mode::strict, false}};
    auto par_impl = std::make_shared<graph::dnnl_impl::dnnl_partition_impl_t>(
            engine.kind(), attr, graph::partition_kind_t::undef);
    par.init(par_impl);
    graph::compiled_partition_t cp(par);
    cp.init(cp_impl);

    ASSERT_EQ(cp.get_inputs().size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_TRUE(ltw(cp.get_inputs()[i]).is_identical(ltw(inputs[i])));
    }

    ASSERT_EQ(cp.get_outputs().size(), outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        ASSERT_TRUE(ltw(cp.get_outputs()[i]).is_identical(ltw(outputs[i])));
    }
    graph::utils::partition_info_t info;
    info.init(&engine, &cp);
    ASSERT_EQ(std::string(cp.info()), std::string(info.c_str()));
}

TEST(test_compiled_partition, GetInputsAndOutputs) {
    using ltw = graph::logical_tensor_wrapper_t;
    graph::engine_t &engine = *get_engine();
    size_t id = 0;

    graph::logical_tensor_t lt1
            = utils::logical_tensor_init(id++, graph::data_type::f32);
    graph::logical_tensor_t lt2
            = utils::logical_tensor_init(id++, graph::data_type::f32);
    graph::logical_tensor_t lt3
            = utils::logical_tensor_init(id++, graph::data_type::f32);

    std::vector<graph::logical_tensor_t> inputs {lt1, lt2};
    std::vector<graph::logical_tensor_t> outputs {lt3};
    graph::dnnl_impl::kernel_ptr kernel
            = std::make_shared<graph::dnnl_impl::float_pooling_fwd>();
    auto cp = std::make_shared<
            graph::dnnl_impl::dnnl_compiled_partition_impl_t>(
            &engine, inputs, outputs, kernel);

    ASSERT_EQ(cp->get_inputs().size(), inputs.size());
    for (size_t i = 0; i < cp->get_inputs().size(); ++i) {
        ASSERT_EQ(ltw(cp->get_inputs()[i]).is_identical(ltw(inputs[i])), true);
    }
    ASSERT_EQ(cp->get_outputs().size(), outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(
                ltw(cp->get_outputs()[i]).is_identical(ltw(outputs[i])), true);
    }
}
