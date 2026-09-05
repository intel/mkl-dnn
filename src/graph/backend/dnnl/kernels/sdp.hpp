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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"
#include "graph/backend/dnnl/kernels/sdp_decomp.hpp"
#include "graph/backend/dnnl/kernels/sdp_decomp_training.hpp"
#include "graph/backend/dnnl/kernels/sdp_fused_brgemm.hpp"
#include "graph/backend/dnnl/kernels/sdp_primitive.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

#define VDISPATCH_GRAPH_SDP(msg, ...) \
    VINFO(graph, create, dispatch, compile, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// The second template param dt is used to indicate the internal data type of
// int8 sdp pattern. It doesn't take any effect if quantized param is false.
template <bool quantized = false, memory::data_type dt = memory::data_type::f32>
struct sdp_base_t : public kernel_base_t {
private:
    std::shared_ptr<kernel_base_t> kernel;

    // Which SDPA implementation to use. `automatic` runs the default dispatch
    // cascade; the others force a single implementation for debug/testing.
    enum class sdpa_impl_kind_t {
        automatic,
        primitive,
        decomp,
        fused_brgemm,
        fused_brgemm_blocked,
        large,
    };

public:
    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override {
        const sdpa_impl_kind_t forced = forced_impl();

        status_t ret = status::unimplemented;

        switch (forced) {
            case sdpa_impl_kind_t::large:
                kernel = std::make_shared<larger_partition_kernel_t>();
                ret = kernel->compile_impl(part, eng, inputs, outputs);
                break;
            case sdpa_impl_kind_t::primitive:
                kernel = std::make_shared<sdp_primitive_kernel_t<quantized>>();
                ret = kernel->compile_impl(part, eng, inputs, outputs);
                break;
            case sdpa_impl_kind_t::decomp:
                kernel = std::make_shared<sdp_decomp_kernel_t<quantized, dt>>();
                ret = kernel->compile_impl(part, eng, inputs, outputs);
                break;
            case sdpa_impl_kind_t::fused_brgemm:
                kernel = std::make_shared<sdp_fused_brgemm_kernel_t>();
                ret = kernel->compile_impl(part, eng, inputs, outputs);
                break;
            case sdpa_impl_kind_t::fused_brgemm_blocked: {
                auto k = std::make_shared<sdp_fused_brgemm_kernel_t>();
                k->set_blocked(true);
                kernel = k;
                ret = kernel->compile_impl(part, eng, inputs, outputs);
                break;
            }
            case sdpa_impl_kind_t::automatic:
            default:
                kernel = std::make_shared<sdp_primitive_kernel_t<quantized>>();
                ret = kernel->compile_impl(part, eng, inputs, outputs);

                if (ret != status::success) {
                    kernel = std::make_shared<
                            sdp_decomp_kernel_t<quantized, dt>>();
                    ret = kernel->compile_impl(part, eng, inputs, outputs);
                }

                // TODO(xxx): merge with the decomp kernel above.
                if (ret != status::success) {
                    kernel = std::make_shared<sdp_decomp_training_kernel_t>();
                    ret = kernel->compile_impl(part, eng, inputs, outputs);
                }

                if (ret != status::success) {
                    kernel = std::make_shared<larger_partition_kernel_t>();
                    ret = kernel->compile_impl(part, eng, inputs, outputs);
                }
                break;
        }

        if (ret == status::success)
            VDISPATCH_GRAPH_SDP(
                    "sdpa is dispatched to %s", kernel->str().c_str());
        else
            VDISPATCH_GRAPH_SDP("sdpa is failed to dispatch");
        return ret;
    }

    // Internal env var to force a specific SDPA implementation, for oneDNN
    // debug and testing only:
    //   ONEDNN_GRAPH_SDPA_IMPL={auto|primitive|decomp|fused_brgemm
    //       |fused_brgemm_blocked|large}
    // The legacy knob ONEDNN_GRAPH_SDPA_FORCE_PRIMITIVE>0 is preserved and maps
    // to `large` (its historical behavior).
    sdpa_impl_kind_t forced_impl() const {
        using graph::utils::check_verbose_string_user;
        if (check_verbose_string_user("GRAPH_SDPA_IMPL", "large"))
            return sdpa_impl_kind_t::large;
        if (check_verbose_string_user(
                    "GRAPH_SDPA_IMPL", "fused_brgemm_blocked"))
            return sdpa_impl_kind_t::fused_brgemm_blocked;
        if (check_verbose_string_user("GRAPH_SDPA_IMPL", "fused_brgemm"))
            return sdpa_impl_kind_t::fused_brgemm;
        if (check_verbose_string_user("GRAPH_SDPA_IMPL", "decomp"))
            return sdpa_impl_kind_t::decomp;
        if (check_verbose_string_user("GRAPH_SDPA_IMPL", "primitive"))
            return sdpa_impl_kind_t::primitive;
        if (graph::utils::getenv_int_internal("GRAPH_SDPA_FORCE_PRIMITIVE", 0)
                > 0)
            return sdpa_impl_kind_t::large;
        return sdpa_impl_kind_t::automatic;
    }

    status_t execute_impl(stream_t *strm, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override {
        return kernel->execute_impl(strm, inputs, outputs, scratchpad_buf);
    }

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        return kernel->sycl_execute_impl(
                strm, inputs, outputs, scratchpad_buf, sycl_deps, sycl_event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<ocl_event_t> &deps, ocl_event_t &event) override {
        return kernel->ocl_execute_impl(
                strm, inputs, outputs, scratchpad_buf, deps, event);
    }
#endif

    std::string str() const override { return kernel->str(); }
    size_t get_scratchpad_size() const override {
        return kernel->get_scratchpad_size();
    }
};
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
