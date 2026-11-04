/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct host_scalar_executable_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER;

    host_scalar_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        UNUSED(pd_cache);
        UNUSED(graph_attr);
        UNUSED(use_block_layout);
        info_ = std::string(dnnl_engine_kind2str(
                        static_cast<dnnl_engine_kind_t>(p_engine.get_kind())))
                + "," + op->str();
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override;
    void execute_impl(const stream &stream,
            const std::unordered_map<int, memory> &args) const;

#ifdef DNNL_WITH_SYCL
    std::optional<::sycl::event> execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override;
    std::optional<::sycl::event> execute_sycl_impl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    ocl_event_t execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const override;

    ocl_event_t execute_ocl_impl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const;
#endif

    bool is_initialized() const override { return true; }

private:
    // needed for verbose profiling.
    std::string info_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP
