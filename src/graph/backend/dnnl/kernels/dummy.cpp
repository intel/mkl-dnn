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

#include "graph/interface/shape_infer.hpp"

#include "graph/backend/dnnl/kernels/dummy.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t dummy_kernel_t::compile_impl(const dnnl_partition_impl_t *part,
        engine_t *eng, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*eng);

    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));
    subgraph_->infer_shape();

    for (size_t i = 0; i < subgraph_->outs_.size(); i++) {
        for (auto val : subgraph_->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == subgraph_->outs_[i].id) {
                subgraph_->outs_[i].layout_type = graph::layout_type::strided;
                auto inferred_shape = logical_tensor_wrapper_t(lt).vdims();
                set_shape_and_strides(subgraph_->outs_[i], inferred_shape);
            }
        }
    }

    // fill information for outputs logical tensors
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    return status::success;
}

status_t dummy_kernel_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    return status::success;
}

#ifdef DNNL_WITH_SYCL
status_t dummy_kernel_t::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    dnnl::stream p_stream = make_dnnl_stream(*strm);

    if (sycl_event) {
        // Fast path: if only one event, return it.
        if (sycl_deps.size() == 1) {
            *sycl_event = sycl_deps[0];
        } else {
            // Otherwise, we run a trivial kernel to gather all deps. The
            // dummy task is needed to not get an error related to empty
            // kernel.
            auto q = dnnl::sycl_interop::get_queue(p_stream);
            *sycl_event = q.submit([&](::sycl::handler &cgh) {
                cgh.depends_on(sycl_deps);
                cgh.single_task<class dnnl_graph_fake_kernel>([]() {});
            });
        }
    }

    return status::success;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
status_t dummy_kernel_t::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &cl_deps, ocl_event_t &ret_event) {

    dnnl::stream p_stream = make_dnnl_stream(*strm);

    ret_event = {};
    if (!cl_deps.empty()) {
        // Fast path: if only one event, return it (copy retains).
        if (cl_deps.size() == 1) {
            ret_event = cl_deps[0];
        } else {
            // Otherwise, gather all dependencies.
            auto q = dnnl::ocl_interop::get_command_queue(p_stream);
            std::vector<cl_event> raw_deps(cl_deps.begin(), cl_deps.end());
            cl_event e;
            auto err = xpu::ocl::clEnqueueMarkerWithWaitList(q,
                    static_cast<cl_uint>(raw_deps.size()), raw_deps.data(), &e);
            assert(err == CL_SUCCESS);
            if (err != CL_SUCCESS) return status::runtime_error;
            ret_event = ocl_event_t(e);
        }
    }

    return status::success;
}
#endif

kernel_ptr dummy_kernel_creator() {
    return std::make_shared<dummy_kernel_t>();
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
