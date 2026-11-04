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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct bn_folding_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER

    // bn_folding_t is a aggregated executable by using multiple primitives, so
    // we need a customized desc class to describe it.
    class desc_t {
        friend struct bn_folding_t;

        float epsilon_ = 1e-5f;
        std::string data_format_;
        std::string filter_format_;

        memory::desc epsilon_desc_;
        memory::desc new_scale_desc_;
        memory::desc new_variance_desc_;
        memory::desc scratchpad_desc_;

        dnnl::binary::primitive_desc add_pd_;
        dnnl::binary::primitive_desc mul_pd_;
        dnnl::binary::primitive_desc sub_pd_;

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
        // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
        dnnl::eltwise_forward::primitive_desc sqrt_pd_;
#endif

        bool with_bias_ {false};

    public:
        const memory::desc &scratchpad_desc() const { return scratchpad_desc_; }
    };

    static desc_t create_desc(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout);

    bn_folding_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
            bool use_block_layout);

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

    bool is_initialized() const override {
        return add_prim_ && mul_prim_ && sub_prim_
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
                && sqrt_prim_
#endif
                ;
    }

private:
    desc_t desc_;
    dnnl::binary add_prim_;
    dnnl::binary mul_prim_;
    dnnl::binary sub_prim_;
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
    dnnl::eltwise_forward sqrt_prim_;
#endif
}; // struct bn_folding_t

struct batchnorm_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    batchnorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout)
        : is_training_(op->get_attr<bool>(op_attr::is_training)) {
        float momentum = 0.5;
        if (op->has_attr(op_attr::momentum))
            momentum = op->get_attr<float>(op_attr::momentum);
        scales_ = {momentum, 1 - momentum};
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::batch_normalization_forward(desc);
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
    dnnl::batch_normalization_forward prim_;
    bool is_training_ {false};
    std::vector<float> scales_;
}; // struct batchnorm_executable_t

struct batchnorm_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    batchnorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const graph_attr_t &graph_attr, bool use_block_layout) {
        auto desc = create_desc(
                op, p_engine, pd_cache, graph_attr, use_block_layout);
        prim_ = dnnl::batch_normalization_backward(desc);
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
    dnnl::batch_normalization_backward prim_;
}; // struct batchnorm_bwd_executable_t

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_BATCH_NORM_HPP
