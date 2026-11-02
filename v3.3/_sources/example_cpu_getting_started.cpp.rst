.. index:: pair: example; cpu_getting_started.cpp
.. _doxid-cpu_getting_started_8cpp-example:

cpu_getting_started.cpp
=======================

This is an example to demonstrate how to build a simple graph and run it on CPU. Annotated version: :ref:`Getting started on CPU with Graph API <doxid-graph_cpu_getting_started_cpp>`

This is an example to demonstrate how to build a simple graph and run it on CPU. Annotated version: :ref:`Getting started on CPU with Graph API <doxid-graph_cpu_getting_started_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2023 Intel Corporation
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
	#include <iostream>
	#include <memory>
	#include <vector>
	#include <unordered_map>
	#include <unordered_set>
	
	#include <assert.h>
	
	#include "oneapi/dnnl/dnnl_graph.hpp"
	
	#include "example_utils.hpp"
	#include "graph_example_utils.hpp"
	
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using :ref:`data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` = :ref:`logical_tensor::data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`;
	using :ref:`layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` = :ref:`logical_tensor::layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`;
	using dim = :ref:`logical_tensor::dim <doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
	using dims = :ref:`logical_tensor::dims <doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;
	//[Headers and namespace]
	
	void cpu_getting_started_tutorial() {
	
	    dim N = 8, IC = 3, OC1 = 96, OC2 = 96;
	    dim IH = 225, IW = 225, KH1 = 11, KW1 = 11, KH2 = 1, KW2 = 1;
	
	    dims conv0_input_dims {N, IC, IH, IW};
	    dims conv0_weight_dims {OC1, IC, KH1, KW1};
	    dims conv0_bias_dims {OC1};
	    dims conv1_weight_dims {OC1, OC2, KH2, KW2};
	    dims conv1_bias_dims {OC2};
	
	
	    //[Create conv's logical tensor]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv0_src_desc {0, data_type::f32};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv0_weight_desc {1, data_type::f32};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv0_dst_desc {2, data_type::f32};
	    //[Create conv's logical tensor]
	
	    //[Create first conv]
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
	            {conv0_dst_desc}, "conv0");
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`, {4, 4});
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`, {0, 0});
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`, {0, 0});
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`, {1, 1});
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<int64_t>(:ref:`op::attr::groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`, 1);
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`, "NCX");
	    conv0.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::weights_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`, "OIX");
	    //[Create first conv]
	
	    //[Create first bias_add]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv0_bias_desc {3, data_type::f32};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv0_bias_add_dst_desc {
	            4, data_type::f32, :ref:`layout_type::undef <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>`};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
	            {conv0_bias_add_dst_desc}, "conv0_bias_add");
	    conv0_bias_add.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`, "NCX");
	    //[Create first bias_add]
	
	    //[Create first relu]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` relu0_dst_desc {5, data_type::f32};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
	            "relu0");
	    //[Create first relu]
	
	    //[Create second conv]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv1_weight_desc {6, data_type::f32};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv1_dst_desc {7, data_type::f32};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` conv1(3, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
	            {conv1_dst_desc}, "conv1");
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::strides <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`, {1, 1});
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::pads_begin <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`, {0, 0});
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::pads_end <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`, {0, 0});
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<dims>(:ref:`op::attr::dilations <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`, {1, 1});
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<int64_t>(:ref:`op::attr::groups <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`, 1);
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`, "NCX");
	    conv1.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::weights_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`, "OIX");
	    //[Create second conv]
	
	    //[Create second bias_add]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv1_bias_desc {8, data_type::f32};
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` conv1_bias_add_dst_desc {9, data_type::f32};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
	            {conv1_bias_add_dst_desc}, "conv1_bias_add");
	    conv1_bias_add.:ref:`set_attr <doxid-classdnnl_1_1graph_1_1op_1ab668a4f176f86967dd6fe2a7478f008f>`<std::string>(:ref:`op::attr::data_format <doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`, "NCX");
	    //[Create second bias_add]
	
	    //[Create second relu]
	    :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` relu1_dst_desc {10, data_type::f32};
	    :ref:`op <doxid-classdnnl_1_1graph_1_1op>` relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
	            "relu1");
	    //[Create second relu]
	
	    //[Create graph and add ops]
	    :ref:`graph <doxid-classdnnl_1_1graph_1_1graph>` g(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`);
	
	    g.add_op(conv0);
	    g.add_op(conv0_bias_add);
	    g.add_op(relu0);
	
	    g.add_op(conv1);
	    g.add_op(conv1_bias_add);
	    g.add_op(relu1);
	    //[Create graph and add ops]
	
	    //[Finialize graph]
	    g.finalize();
	    //[Finialize graph]
	
	    //[Get partition]
	    auto partitions = g.get_partitions();
	    //[Get partition]
	
	    // Check partitioning results to ensure the examples works. Users do
	    // not need to follow this step.
	    assert(partitions.size() == 2);
	
	    //[Create engine]
	    :ref:`allocator <doxid-classdnnl_1_1graph_1_1allocator>` alloc {};
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` eng
	            = :ref:`make_engine_with_allocator <doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0, alloc);
	    //[Create engine]
	
	    //[Create stream]
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm {eng};
	    //[Create stream]
	
	    // Mapping from logical tensor id to output tensors
	    // used to the connection relationship between partitions (e.g partition 0's
	    // output tensor is fed into partition 1)
	    std::unordered_map<size_t, tensor> global_outputs_ts_map;
	
	    // Memory buffers binded to the partition input/output tensors
	    // that helpe manage the lifetime of these tensors
	    std::vector<std::shared_ptr<void>> data_buffer;
	
	    // Mapping from id to queried logical tensor from compiled partition
	    // used to record the logical tensors that are previously enabled with
	    // ANY layout
	    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;
	
	    // This is a helper function which helps decide which logical tensor is
	    // needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
	    // layout.
	    // This function is not a part to Graph API, but similar logic is
	    // essential for Graph API integration to achieve best performance.
	    // Typically, users need implement the similar logic in their code.
	    std::unordered_set<size_t> ids_with_any_layout;
	    set_any_layout(partitions, ids_with_any_layout);
	
	    // Mapping from logical tensor id to the concrete shapes.
	    // In practical usage, concrete shapes and layouts are not given
	    // until compilation stage, hence need this mapping to mock the step.
	    std::unordered_map<size_t, dims> concrete_shapes {{0, conv0_input_dims},
	            {1, conv0_weight_dims}, {3, conv0_bias_dims},
	            {6, conv1_weight_dims}, {8, conv1_bias_dims}};
	
	    // Compile and execute the partitions, including the following steps:
	    //
	    // 1. Update the input/output logical tensors with concrete shape and layout
	    // 2. Compile the partition
	    // 3. Update the output logical tensors with queried ones after compilation
	    // 4. Allocate memory and bind the data buffer for the partition
	    // 5. Execute the partition
	    //
	    // Although they are not part of the APIs, these steps are esstential for
	    // the integration of Graph API., hence users need to implement similar
	    // logic.
	    for (const auto &:ref:`partition <doxid-classdnnl_1_1graph_1_1partition>` : partitions) {
	        if (!:ref:`partition <doxid-classdnnl_1_1graph_1_1partition>`.:ref:`is_supported <doxid-classdnnl_1_1graph_1_1partition_1ad80536833d69e2660c496adbd9ec0aa3>`()) {
	            std::cout
	                    << "cpu_get_started: Got unsupported partition, users need "
	                       "handle the operators by themselves."
	                    << std::endl;
	            continue;
	        }
	
	        std::vector<logical_tensor> inputs = :ref:`partition <doxid-classdnnl_1_1graph_1_1partition>`.:ref:`get_input_ports <doxid-classdnnl_1_1graph_1_1partition_1a415319dcb89d9e1d77bd4b7b0058df52>`();
	        std::vector<logical_tensor> outputs = :ref:`partition <doxid-classdnnl_1_1graph_1_1partition>`.:ref:`get_output_ports <doxid-classdnnl_1_1graph_1_1partition_1aaa4abecc6e09f417742402ab207a1e6d>`();
	
	        // Update input logical tensors with concrete shape and layout
	        for (auto &input : inputs) {
	            const auto id = input.get_id();
	            // If the tensor is an output of another partition,
	            // use the cached logical tensor
	            if (id_to_queried_logical_tensors.find(id)
	                    != id_to_queried_logical_tensors.end())
	                input = id_to_queried_logical_tensors[id];
	            else
	                // Create logical tensor with strided layout
	                input = :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` {id, input.:ref:`get_data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f>`(),
	                        concrete_shapes[id], layout_type::strided};
	        }
	
	        // Update output logical tensors with concrete shape and layout
	        for (auto &output : outputs) {
	            const auto id = output.get_id();
	            output = :ref:`logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` {id, output.:ref:`get_data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f>`(),
	                    :ref:`DNNL_GRAPH_UNKNOWN_NDIMS <doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf>`, // set output dims to unknown
	                    ids_with_any_layout.count(id) ? :ref:`layout_type::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`
	                                                  : layout_type::strided};
	        }
	
	        //[Compile partition]
	        :ref:`compiled_partition <doxid-classdnnl_1_1graph_1_1compiled__partition>` cp = :ref:`partition <doxid-classdnnl_1_1graph_1_1partition>`.:ref:`compile <doxid-classdnnl_1_1graph_1_1partition_1a5c2af93c65a09c9d0a1507571ada0318>`(inputs, outputs, eng);
	        //[Compile partition]
	
	        // Update output logical tensors with queried one
	        for (auto &output : outputs) {
	            const auto id = output.get_id();
	            output = cp.:ref:`query_logical_tensor <doxid-classdnnl_1_1graph_1_1compiled__partition_1a85962826e94cc3cefb3c19c0fadc4e09>`(id);
	            id_to_queried_logical_tensors[id] = output;
	        }
	
	        // Allocate memory for the partition, and bind the data buffers with
	        // input and output logical tensors
	        std::vector<tensor> inputs_ts, outputs_ts;
	        allocate_graph_mem(inputs_ts, inputs, data_buffer,
	                global_outputs_ts_map, eng, /*is partition input=*/true);
	        allocate_graph_mem(outputs_ts, outputs, data_buffer,
	                global_outputs_ts_map, eng, /*is partition input=*/false);
	
	        //[Execute compiled partition]
	        cp.:ref:`execute <doxid-classdnnl_1_1graph_1_1compiled__partition_1a558ed47b3cbc5cc2167001da3faa0339>`(strm, inputs_ts, outputs_ts);
	        //[Execute compiled partition]
	    }
	
	    // Wait for all compiled partition's execution finished
	    strm.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	
	    std::cout << "Graph:" << std::endl
	              << " [conv0_src] [conv0_wei]" << std::endl
	              << "       \\      /" << std::endl
	              << "         conv0" << std::endl
	              << "          \\    [conv0_bias_src1]" << std::endl
	              << "           \\      /" << std::endl
	              << "         conv0_bias_add" << std::endl
	              << "                |" << std::endl
	              << "              relu0" << std::endl
	              << "                \\   [conv1_wei]" << std::endl
	              << "                 \\    /" << std::endl
	              << "                  conv1" << std::endl
	              << "                    \\  [conv1_bias_src1]" << std::endl
	              << "                     \\      /" << std::endl
	              << "                  conv1_bias_add" << std::endl
	              << "                          |" << std::endl
	              << "                        relu1" << std::endl
	              << "                          |" << std::endl
	              << "                      [relu_dst]" << std::endl
	              << "Note:" << std::endl
	              << " '[]' represents a logical tensor, which refers to "
	                 "inputs/outputs of the graph. "
	              << std::endl;
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            {:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`}, cpu_getting_started_tutorial);
	}
