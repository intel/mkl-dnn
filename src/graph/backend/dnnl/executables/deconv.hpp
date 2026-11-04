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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_DECONV_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_DECONV_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct deconv_fwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::deconvolution_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::deconvolution_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override;

#ifdef DNNL_WITH_SYCL
    std::optional<::sycl::event> execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    ocl_event_t execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const override;
#endif

    bool is_initialized() const override { return bool(prim_); }

private:
    dnnl::deconvolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_bwd_data_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_data::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::deconvolution_backward_data(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    std::optional<::sycl::event> execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    ocl_event_t execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const override {
        std::vector<cl_event> raw_deps(deps.begin(), deps.end());
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, raw_deps);
        return ocl_event_t(e);
    }
#endif

    bool is_initialized() const override { return bool(prim_); }

private:
    dnnl::deconvolution_backward_data prim_;
};

struct deconv_bwd_weights_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_weights::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::deconvolution_backward_weights(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    std::optional<::sycl::event> execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    ocl_event_t execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<ocl_event_t> &deps) const override {
        std::vector<cl_event> raw_deps(deps.begin(), deps.end());
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, raw_deps);
        return ocl_event_t(e);
    }
#endif

    bool is_initialized() const override { return bool(prim_); }

private:
    dnnl::deconvolution_backward_weights prim_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_DECONV_HPP
