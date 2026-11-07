/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <functional>
#include <random>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using dim_t = dnnl_dim_t;
using dims_t = dnnl_dims_t;
using dims = std::vector<dim_t>;

static inline void custom_setenv(
        const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    SetEnvironmentVariable(name, value);
#else
    ::setenv(name, value, overwrite);
#endif
}

// Test multiple thread execute
TEST(test_mqa_decomp_execute, MultithreadMqaDecomp_CPU) {
    graph::engine_t *eng = get_engine();

    SKIP_IF(eng->kind() == graph::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    // 4 concurrent threads will be executed below. Limiting the number of OMP
    // threads to avoid oversubscription.
    const int nthr = dnnl_get_current_num_threads() / 4;
    SKIP_IF(nthr == 0, "skip as not enough threads for test");
    omp_set_num_threads(nthr);
#endif

    int batch_size = 56;
    graph::graph_t g(eng->kind());
    utils::construct_dnnl_float_JAX_MQA(
            &g, dnnl::impl::data_type::f32, batch_size);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("float_mqa_jax_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 4U);
    ASSERT_EQ(partition_outputs.size(), 1U);

    std::vector<const graph::logical_tensor_t *> inputs, outputs;

    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, graph::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    custom_setenv("_ONEDNN_GRAPH_SDPA_FORCE_PRIMITIVE", "0", 1);
    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    std::vector<test_tensor_t> inputs_ts;
    for (auto &lt : inputs) {
        inputs_ts.emplace_back(*lt, eng);
        inputs_ts.back().fill<float>();
    }

    auto func = [&]() {
        graph::stream_t *strm;
        dnnl_stream_create(&strm, eng, dnnl_stream_in_order);
        std::vector<test_tensor_t> outputs_ts;
        for (auto &lt : outputs) {
            graph::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            outputs_ts.emplace_back(compiled_output, eng);
        }
        for (int i = 0; i < 5; i++) {
            ASSERT_EQ(
                    cp.execute(strm, test_tensor_t::to_graph_tensor(inputs_ts),
                            test_tensor_t::to_graph_tensor(outputs_ts)),
                    graph::status::success);
        }
        strm->wait();
        dnnl_stream_destroy(strm);
    };

    std::thread t1(func);
    std::thread t2(func);
    std::thread t3(func);
    std::thread t4(func);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}
