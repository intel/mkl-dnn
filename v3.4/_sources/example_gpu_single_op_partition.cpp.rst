.. index:: pair: example; gpu_single_op_partition.cpp
.. _doxid-gpu_single_op_partition_8cpp-example:

gpu_single_op_partition.cpp
===========================

This is an example to demonstrate how to build a simple op graph and run it on gpu. Annotated version: :ref:`Single op partition on GPU <doxid-graph_gpu_single_op_partition_cpp>`

This is an example to demonstrate how to build a simple op graph and run it on gpu. Annotated version: :ref:`Single op partition on GPU <doxid-graph_gpu_single_op_partition_cpp>`



.. ref-code-block:: cpp

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
	
	
	
	//[Headers and namespace]
	#include "oneapi/dnnl/dnnl_graph.hpp"
	#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
	#include "oneapi/dnnl/dnnl_sycl.hpp"
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using namespace :ref:`sycl <doxid-namespacesycl>`;
	
	#include <assert.h>
	#include <iostream>
	#include <memory>
	#include <vector>
	#include <unordered_map>
	#include <unordered_set>
	
	#include "example_utils.hpp"
	#include "graph_example_utils.hpp"
	
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using :ref:`data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` = :ref:`logical_tensor::data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`;
	using :ref:`layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` = :ref:`logical_tensor::layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`;
	using dim = :ref:`logical_tensor::dim <doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
	using dims = :ref:`logical_tensor::dims <doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;
	//[Headers and namespace]
	
	void gpu_single_op_partition_tutorial() {
	
	    dim M = 32, K = 1024, N = 2048;
	
	    dims src0_dims {M, K};
	    dims src1_dims {K, N};
	
	
	    //[Create matmul]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` matmul_src0_desc {0, :ref:`data_type::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` matmul_src1_desc {1, :ref:`data_type::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` matmul_dst_desc {2, :ref:`data_type::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` :ref:`matmul <doxid-structdnnl_1_1matmul>`(0, op::kind::MatMul, {matmul_src0_desc, matmul_src1_desc},
	            {matmul_dst_desc}, "matmul");
	    :ref:`matmul <doxid-structdnnl_1_1matmul>`.set_attr<bool>(:ref:`op::attr::transpose_a <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8739d82596ce4e8592bde9475504c430>`, false);
	    :ref:`matmul <doxid-structdnnl_1_1matmul>`.set_attr<bool>(:ref:`op::attr::transpose_b <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684aa842de682cfdaec3291bbdffa551f4d7>`, false);
	    //[Create matmul]
	
	    //[Create allocator]
	    :ref:`allocator <doxid-classdnnl_1_1graph_1_1allocator>` alloc = :ref:`sycl_interop::make_allocator <doxid-namespacednnl_1_1graph_1_1sycl__interop_1afbfd5202a21eebb29d010f14bcbbbb13>`(
	            sycl_malloc_wrapper, sycl_free_wrapper);
	    //[Create allocator]
	
	    //[Define sycl queue]
	    sycl::queue q = sycl::queue(
	            sycl::gpu_selector_v, sycl::property::queue::in_order {});
	    //[Define sycl queue]
	
	    //[Create engine]
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` eng = sycl_interop::make_engine_with_allocator(
	            q.get_device(), q.get_context(), alloc);
	    //[Create engine]
	
	    //[Create stream]
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm = :ref:`dnnl::sycl_interop::make_stream <doxid-namespacednnl_1_1sycl__interop_1a170bddd16d53869fc18412894400ccab>`(eng, q);
	    //[Create stream]
	
	    // Memory buffers bound to the partition input/output tensors
	    // that helps manage the lifetime of these tensors
	    std::vector<std::shared_ptr<void>> data_buffer;
	
	    // Mapping from logical tensor id to the concrete shapes.
	    // In practical usage, concrete shapes and layouts are not given
	    // until compilation stage, hence need this mapping to mock the step.
	    std::unordered_map<size_t, dims> concrete_shapes {
	            {0, src0_dims}, {1, src1_dims}};
	
	    // Compile and execute the partitions, including the following steps:
	    //
	    // 1. Update the input/output logical tensors with concrete shape and layout
	    // 2. Compile the partition
	    // 3. Update the output logical tensors with queried ones after compilation
	    // 4. Allocate memory and bind the data buffer for the partition
	    // 5. Execute the partition
	    //
	    // Although they are not part of the APIs, these steps are essential for
	    // the integration of Graph API., hence users need to implement similar
	    // logic.
	
	    //[Create partition]
	    :ref:`partition <doxid-classdnnl_1_1graph_1_1partition>` part(:ref:`matmul <doxid-structdnnl_1_1matmul>`, :ref:`dnnl::engine::kind::gpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947>`);
	    //[Create partition]
	    if (!part.is_supported()) {
	        std::cout << "gpu_single_op_partition: Got unsupported partition, "
	                     "users need to handle the operators by themselves."
	                  << std::endl;
	        return;
	    }
	
	    std::vector<logical_tensor> inputs = part.get_input_ports();
	    std::vector<logical_tensor> outputs = part.get_output_ports();
	
	    // Update input logical tensors with concrete shape and layout
	    for (auto &input : inputs) {
	        const auto id = input.get_id();
	        // Create logical tensor with strided layout
	        input = :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` {id, input.:ref:`get_data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f>`(), concrete_shapes[id],
	                layout_type::strided};
	    }
	
	    // Update output logical tensors with concrete shape and layout
	    for (auto &output : outputs) {
	        const auto id = output.get_id();
	        output = :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` {id, output.:ref:`get_data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f>`(),
	                :ref:`DNNL_GRAPH_UNKNOWN_NDIMS <doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf>`,
	                // do not require concrete shape as the shape will be inferred
	                // based on input shapes during compilation
	                layout_type::strided};
	    }
	
	    //[Compile partition]
	    :ref:`compiled_partition <doxid-classdnnl_1_1graph_1_1compiled__partition>` cp = part.compile(inputs, outputs, eng);
	    //[Compile partition]
	
	    // Update output logical tensors with queried one
	    for (auto &output : outputs) {
	        const auto id = output.get_id();
	        output = cp.:ref:`query_logical_tensor <doxid-classdnnl_1_1graph_1_1compiled__partition_1a85962826e94cc3cefb3c19c0fadc4e09>`(id);
	    }
	
	    // Allocate memory for the partition, and bind the data buffers with
	    // input and output logical tensors
	    std::vector<tensor> inputs_ts, outputs_ts;
	    allocate_sycl_graph_mem(inputs_ts, inputs, data_buffer, q, eng);
	    allocate_sycl_graph_mem(outputs_ts, outputs, data_buffer, q, eng);
	
	    //[Execute compiled partition]
	    cp.:ref:`execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1a558ed47b3cbc5cc2167001da3faa0339>`(strm, inputs_ts, outputs_ts);
	    //[Execute compiled partition]
	
	    // Wait for all compiled partition's execution finished
	    strm.wait();
	
	    std::cout << "Graph:" << std::endl
	              << " [matmul_src0] [matmul_src1]" << std::endl
	              << "       \\       /" << std::endl
	              << "         matmul" << std::endl
	              << "            |" << std::endl
	              << "        [matmul_dst]" << std::endl
	              << "Note:" << std::endl
	              << " '[]' represents a logical tensor, which refers to "
	                 "inputs/outputs of the graph. "
	              << std::endl;
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({validate_engine_kind(:ref:`engine::kind::gpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aa0aa0be2a866411d9ff03515227454947>`)},
	            gpu_single_op_partition_tutorial);
	}
