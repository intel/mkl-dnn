/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <assert.h>

#include "common/memory.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_memory_storage.hpp"
#include "cpu/cpu_stream.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "cpu/threadpool/engine.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

status_t cpu_engine_t::create_memory_storage(
        memory_storage_t **storage, unsigned flags, size_t size, void *handle) {
    assert(runtime_kind() != runtime_kind::sycl);
    if (runtime_kind() == runtime_kind::sycl) return status::runtime_error;

    auto _storage = new cpu_memory_storage_t(this);
    if (_storage == nullptr) return status::out_of_memory;
    status_t status = _storage->init(flags, size, handle);
    if (status != status::success) {
        delete _storage;
        return status;
    }
    *storage = _storage;
    return status::success;
}

status_t cpu_engine_t::create_stream(
        stream_t **stream, impl::stream_impl_t *stream_impl) {
    return safe_ptr_assign(*stream, new cpu_stream_t(this, stream_impl));
}

status_t cpu_engine_factory_t::engine_create(
        engine_t **engine, size_t index) const {
    assert(index == 0);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    *engine = new threadpool::engine_t(index);
#else
    *engine = new cpu_engine_t(new impl::engine_impl_t(
            engine_kind::cpu, get_cpu_native_runtime(), index));
#endif

#if DNNL_AARCH64 && defined(DNNL_AARCH64_USE_ACL)
    dnnl::impl::cpu::aarch64::acl_thread_utils::set_acl_threading();
#endif
    return status::success;
}

engine_t *get_service_engine() {
    static std::unique_ptr<engine_t, engine_deleter_t> cpu_engine;
    static std::once_flag initialized;
    std::call_once(initialized, [&]() {
        engine_t *cpu_engine_ptr;
        cpu::cpu_engine_factory_t f;
        auto status = f.engine_create(&cpu_engine_ptr, 0);
        assert(status == status::success);
        MAYBE_UNUSED(status);
        cpu_engine.reset(cpu_engine_ptr);
    });
    return cpu_engine.get();
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
