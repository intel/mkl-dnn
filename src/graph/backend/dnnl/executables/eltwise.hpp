/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_ELTWISE_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_ELTWISE_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct eltwise_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::eltwise_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    eltwise_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::eltwise_forward(desc);
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
    dnnl::eltwise_forward prim_;
};

struct eltwise_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::eltwise_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    eltwise_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::eltwise_backward(desc);
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
    dnnl::eltwise_backward prim_;
};

struct binary_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::binary::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    binary_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
            bool use_block_layout) {
        using ltw = logical_tensor_wrapper_t;
        // if with zero dimension, the binary op will take no effect, we
        // construct a dummy kernel
        if (ltw(op->get_input_logical_tensor(0)).has_zero_dim()
                || ltw(op->get_input_logical_tensor(1)).has_zero_dim()) {
            is_dummy_ = true;
            return;
        }

        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::binary(desc);

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
    dnnl::binary prim_;
    bool with_sum_ {false};
    bool is_dummy_ {false};
    dummy_impl_t dummy_impl_;
};

struct prelu_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::prelu_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    prelu_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
            bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::prelu_forward(desc);
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
    dnnl::prelu_forward prim_;
};

struct prelu_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::prelu_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    prelu_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::prelu_backward(desc);
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
    dnnl::prelu_backward prim_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
