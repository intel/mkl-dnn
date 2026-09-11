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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_SDPA_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_SDPA_HPP

#include "graph/backend/dnnl/executables/base.hpp"
#include "graph/backend/dnnl/executables/deleter_util.hpp"

#include "common/sdpa_utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// Creates the dnnl_sdpa primitive descriptor from a graph sdpa op. Shared by
// the executable (which owns the pd to build the primitive) and the layout
// propagator (which queries the scratchpad size before memory planning).
status_t create_sdpa_pd(std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> &pd,
        const std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        const fpmath_t &fpmath);

struct sdpa_executable_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER;

    sdpa_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            pd_cache_t &pd_cache, const fpmath_t &fpmath,
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

    bool is_initialized() const override { return pd_ && prim_; }

private:
    std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> pd_;
    std::unique_ptr<dnnl_primitive, prim_deleter_t> prim_;

    bool with_scale_;
    bool is_training_;
    bool with_explicit_mask_;
    attn_mask_type_t mask_type_;
    bool is_invert_scale_;
    bool with_dropout_;
};

struct sdpa_bwd_executable_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER;

    sdpa_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout);

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

    bool is_initialized() const override { return hint_pd_ && pd_ && prim_; }

private:
    std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> hint_pd_;
    std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> pd_;
    std::unique_ptr<dnnl_primitive, prim_deleter_t> prim_;
    bool with_scale_;
    attn_mask_type_t mask_type_;
    bool is_invert_scale_;
    bool with_explicit_mask_;
    bool with_dropout_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_SDPA_HPP
