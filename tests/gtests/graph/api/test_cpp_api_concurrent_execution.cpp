/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "test_api_common.hpp"
#include "gtest/gtest.h"

using namespace dnnl::graph;

struct sdpa_dims_t {
    logical_tensor::dim mb;
    logical_tensor::dim seq_len;
    logical_tensor::dim head_num;
    logical_tensor::dim head_size;
    logical_tensor::dim query_num;
};

const int num_threads = 4;
// execution times for each thread to run the compiled partition.
const int num_executions = 500;

// Helper function to create SDPA graph
std::pair<dnnl::graph::graph, std::vector<logical_tensor>> create_sdpa_graph(
        dnnl::engine::kind engine_kind, logical_tensor::data_type dt,
        const sdpa_dims_t &p) {

    // Prepare input and output shapes
    const dims_t qv_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const dims_t k_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims_t score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const dims_t scale_sz = {1};
    const dims_t mask_sz = {p.mb, 1, p.query_num, p.seq_len};

    // Incremental IDs for logical tensors and operations
    size_t id = 0;

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // Create logical tensors
    auto query = logical_tensor(
            id++, dt, qv_sz, logical_tensor::layout_type::strided);
    auto key = logical_tensor(
            id++, dt, k_sz, logical_tensor::layout_type::strided);
    auto score = logical_tensor(
            id++, dt_inter, score_sz, logical_tensor::layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // Scale operation
    auto scale = logical_tensor(
            id++, dt, scale_sz, logical_tensor::layout_type::strided);
    auto scaled_score = logical_tensor(
            id++, dt_inter, score_sz, logical_tensor::layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // Mask operation
    auto mask = logical_tensor(
            id++, dt, mask_sz, logical_tensor::layout_type::strided);
    auto masked_score = logical_tensor(
            id++, dt_inter, score_sz, logical_tensor::layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // Softmax
    auto probs = logical_tensor(
            id++, dt, score_sz, logical_tensor::layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});

    // Final matmul
    auto value = logical_tensor(
            id++, dt, k_sz, logical_tensor::layout_type::strided);
    auto output = logical_tensor(
            id++, dt, qv_sz, logical_tensor::layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct graph
    dnnl::graph::graph sdpa_graph(engine_kind);
    sdpa_graph.add_op(bmm1);
    sdpa_graph.add_op(scale_div);
    sdpa_graph.add_op(mask_add);
    sdpa_graph.add_op(softmax);
    sdpa_graph.add_op(bmm2);
    sdpa_graph.finalize();

    // Return graph and input/output tensors
    std::vector<logical_tensor> tensors;
    tensors.push_back(query);
    tensors.push_back(key);
    tensors.push_back(scale);
    tensors.push_back(mask);
    tensors.push_back(value);
    tensors.push_back(output);
    return std::make_pair(std::move(sdpa_graph), std::move(tensors));
}

// Thread worker function for concurrent execution
void execute_partition_worker(int thread_id, const compiled_partition &cp,
        std::vector<logical_tensor> input_tensors, logical_tensor output_tensor,
        const dnnl::engine &eng, std::atomic<int> &success_count,
        std::atomic<int> &error_count) {
    std::cout << "Thread " << thread_id << " starting execution" << std::endl;
    try {
        // Create stream for this thread
        dnnl::stream strm(eng);

        // each thread creates its own tensors to avoid data races.
        auto ts_query = tensor(input_tensors[0], eng);
        auto ts_key = tensor(input_tensors[1], eng);
        auto ts_scale = tensor(input_tensors[2], eng);
        auto ts_mask = tensor(input_tensors[3], eng);
        auto ts_value = tensor(input_tensors[4], eng);
        auto ts_output = tensor(output_tensor, eng);

        std::vector<tensor> input_tensors
                = {ts_query, ts_key, ts_scale, ts_mask, ts_value};
        std::vector<tensor> output_tensors = {ts_output};
        for (int i = 0; i < num_executions; ++i) {
            cp.execute(strm, input_tensors, output_tensors);
            strm.wait();
        }

        success_count.fetch_add(num_executions);
    } catch (const std::exception &e) {
        std::cerr << "Thread " << thread_id << " error: " << e.what()
                  << std::endl;
        error_count.fetch_add(num_executions); // Mark all executions as failed
    }

    std::cout << "Thread " << thread_id << " finished execution" << std::endl;
}

TEST(APIConcurrentExecution, SDPAConcurrentTest) {
    using namespace dnnl::graph;

    dnnl::engine::kind engine_kind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    dnnl::engine eng = cpp_api_test_dnnl_engine_create(engine_kind);

    // Define SDPA dimensions for test
    sdpa_dims_t dims = {2, 128, 8, 64, 128};

    logical_tensor::data_type dt = logical_tensor::data_type::f32;

    // Create SDPA graph
    std::pair<dnnl::graph::graph, std::vector<logical_tensor>> graph_tensor_pair
            = create_sdpa_graph(engine_kind, dt, dims);
    dnnl::graph::graph sdpa_graph = graph_tensor_pair.first;
    std::vector<logical_tensor> tensors = graph_tensor_pair.second;

    // Get partitions
    std::vector<partition> partitions = sdpa_graph.get_partitions();
    ASSERT_EQ(partitions.size(), 1U) << "Should be only one partition";

    // Compile the partition
    const auto &part = partitions[0];
    std::vector<logical_tensor> inputs(tensors.begin(), tensors.end() - 1);
    std::vector<logical_tensor> outputs = {tensors.back()};
    compiled_partition cp = part.compile(inputs, outputs, eng);

    // Create atomic counters to track execution results
    std::atomic<int> success_count {0};
    std::atomic<int> error_count {0};

    // Launch the concurrent threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        std::vector<logical_tensor> thread_inputs(
                tensors.begin(), tensors.end() - 1);
        logical_tensor thread_output = tensors.back();

        threads.emplace_back(execute_partition_worker, i, cp, thread_inputs,
                thread_output, eng, std::ref(success_count),
                std::ref(error_count));
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
        thread.join();
    }

    // Verify results
    const int expected_total = num_threads * num_executions;

    EXPECT_EQ(error_count.load(), 0)
            << "Encountered " << error_count.load() << " execution errors";
    EXPECT_EQ(success_count.load(), expected_total)
            << "Expected " << expected_total << " successful executions, got "
            << success_count.load();
}
