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

#ifndef CPU_THREADPOOL_ENGINE_HPP
#define CPU_THREADPOOL_ENGINE_HPP

#include "common/utils.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/threadpool/engine_impl.hpp"
#include "cpu/threadpool/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace threadpool {

class engine_t : public cpu::cpu_engine_t {
public:
    engine_t(size_t index)
        : cpu::cpu_engine_t(new threadpool::engine_impl_t(index)) {}

    status_t create_stream(impl::stream_t **stream,
            impl::stream_impl_t *stream_impl) override {
        return safe_ptr_assign(*stream,
                new threadpool::stream_t(this,
                        utils::downcast<threadpool::stream_impl_t *>(
                                stream_impl)));
    }

protected:
    ~engine_t() override = default;
};

} // namespace threadpool
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
