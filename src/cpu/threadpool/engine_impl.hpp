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

#ifndef CPU_THREADPOOL_ENGINE_IMPL_HPP
#define CPU_THREADPOOL_ENGINE_IMPL_HPP

#include "common/engine_impl.hpp"

#include "cpu/threadpool/stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace threadpool {

// Ensures streams created through dnnl_stream_create() are also backed by
// threadpool::stream_impl_t.
class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t(size_t index)
        : impl::engine_impl_t(
                  engine_kind::cpu, runtime_kind::threadpool, index) {}

    ~engine_impl_t() override = default;

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override {
        auto *si = new threadpool::stream_impl_t(flags);
        if (!si) return status::out_of_memory;
        *stream_impl = si;
        return status::success;
    }
};

} // namespace threadpool
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
