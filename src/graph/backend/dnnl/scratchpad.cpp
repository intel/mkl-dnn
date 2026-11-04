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

#include "graph/interface/tensor.hpp"

#include "graph/backend/dnnl/dnnl_allocator.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"

#include "common/dnnl_thread.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

scratchpad_t::scratchpad_t(
        const tensor_t *user_buf, size_t size, const dnnl::engine &eng)
    : buffer_(nullptr)
    , size_(size)
    , user_managed_(user_buf != nullptr)
    , eng_(eng.get()) {
    if (user_buf) {
        buffer_ = reinterpret_cast<char *>(user_buf->get_data_handle());
    } else if (size > 0) {
        buffer_ = reinterpret_cast<char *>(dnnl_allocator_t::malloc(
                size, *eng_, allocator_t::mem_type_t::temp));
        if (!buffer_) { size_ = 0; }
    }
}

scratchpad_t::~scratchpad_t() {
    if (user_managed_) return;
    const auto ekind = eng_->kind();
    if (ekind == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        dnnl_allocator_t::free(buffer_, *eng_, e_);
#else
        dnnl_allocator_t::free(buffer_, *eng_);
#endif
    } else if (ekind == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        dnnl_allocator_t::free(buffer_, *eng_, ocl_e_);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        dnnl_allocator_t::free(buffer_, *eng_, e_);
#else
        assert(!"unsupported gpu runtime");
#endif
    } else {
        assert(!"unsupported engine kind");
    }
}

void prolong_scratchpad_lifetime(
        stream_t *strm, const std::shared_ptr<scratchpad_t> &scratchpad) {
    if (scratchpad->is_user_managed()) return;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    strm->before_exec_hook();

    parallel(1, [=](int, int) { UNUSED(scratchpad); });

    strm->after_exec_hook();
#endif
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
