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

#ifndef CPU_CPU_ENGINE_IMPL_HPP
#define CPU_CPU_ENGINE_IMPL_HPP

#include "common/engine_impl.hpp"
#include "cpu/cpu_stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// CPU-specific engine implementation subclass. Overrides create_stream_impl()
// to ensure all CPU streams are backed by cpu::stream_impl_t rather than the
// base impl::stream_impl_t, making the downcast in cpu_stream_t::impl() safe.
struct engine_impl_t : public impl::engine_impl_t {
    engine_impl_t(engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : impl::engine_impl_t(kind, runtime_kind, index) {}

    ~engine_impl_t() override = default;

    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override {
        *stream_impl = new cpu::stream_impl_t(flags);
        if (!*stream_impl) return status::out_of_memory;
        return status::success;
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
