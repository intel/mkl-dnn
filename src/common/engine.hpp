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

#ifndef COMMON_ENGINE_HPP
#define COMMON_ENGINE_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "common/engine_impl.hpp"
#include "common/stream_impl.hpp"
#include "engine_id.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VERROR_ENGINE(cond, stat, msg, ...) \
    do { \
        if (!(cond)) { \
            VERROR(common, runtime, msg, ##__VA_ARGS__); \
            return stat; \
        } \
    } while (0)

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Provide engine specific memory allocation
 *   - Provide engine specific primitive_desc_t creators
 */
struct dnnl_engine {
    dnnl_engine(dnnl::impl::engine_impl_t *impl) : impl_(impl), counter_(1) {}

    /** get kind of the current engine */
    dnnl::impl::engine_kind_t kind() const { return impl()->kind(); }

    /** get the runtime kind of the current engine */
    dnnl::impl::runtime_kind_t runtime_kind() const {
        return impl()->runtime_kind();
    }

    /** get index of the current engine */
    size_t index() const { return impl()->index(); }

    virtual dnnl::impl::engine_id_t engine_id() const {
        return impl()->engine_id();
    }

    /** create memory storage */
    virtual dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, unsigned flags, size_t size,
            void *handle) {
        assert(impl());
        if (!impl()) return dnnl::impl::status::runtime_error;
        return impl()->create_memory_storage(
                storage, this, flags, size, handle);
    }

    dnnl::impl::status_t create_memory_storage(
            dnnl::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, dnnl::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual dnnl::impl::status_t create_stream(dnnl::impl::stream_t **stream,
            dnnl::impl::stream_impl_t *stream_impl)
            = 0;

    dnnl::impl::status_t create_stream(
            dnnl::impl::stream_t **stream, unsigned flags) {
        std::unique_ptr<dnnl::impl::stream_impl_t> stream_impl;
        dnnl::impl::stream_impl_t *stream_impl_ptr = nullptr;
        CHECK(impl()->create_stream_impl(&stream_impl_ptr, flags));
        stream_impl.reset(stream_impl_ptr);

        dnnl::impl::stream_t *s;
        CHECK(create_stream(&s, stream_impl.get()));
        // stream (`s`) takes ownership of `stream_impl` if `create_stream` call
        // is successful.
        stream_impl.release();
        *stream = s;
        return dnnl::impl::status::success;
    }

    virtual dnnl::impl::status_t get_service_stream(
            dnnl::impl::stream_t *&stream) {
        stream = nullptr;
        return dnnl::impl::status::success;
    }

    /* implementation section */

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() const
            = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const dnnl::impl::impl_list_item_t *
    get_sum_implementation_list() const
            = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */

    virtual const dnnl::impl::impl_list_item_t *get_implementation_list(
            const dnnl::impl::op_desc_t *desc) const
            = 0;

    virtual dnnl::impl::status_t serialize_device(
            dnnl::impl::serialization_stream_t &sstream) const {
        assert(!"unexpected");
        return dnnl::impl::status::runtime_error;
    }

    virtual dnnl::impl::status_t get_cache_blob_size(size_t *size) const {
        assert(!"unexpected");
        return dnnl::impl::status::runtime_error;
    }

    virtual dnnl::impl::status_t get_cache_blob(
            size_t size, uint8_t *cache_blob) const {
        assert(!"unexpected");
        return dnnl::impl::status::runtime_error;
    }

    bool is_cache_blob_supported() const {
        if (kind() != dnnl::impl::engine_kind::gpu) return false;
        if (!dnnl::impl::utils::one_of(runtime_kind(),
                    dnnl::impl::runtime_kind::ocl,
                    dnnl::impl::runtime_kind::ze))
            return false;
        return true;
    }

    virtual bool mayiuse_system_memory_allocators() const { return false; }

    const dnnl::impl::engine_impl_t *impl() const { return impl_.get(); }

#ifdef ONEDNN_BUILD_GRAPH
    // only used in graph implementation
    void *get_allocator() const { return impl()->get_allocator(); }
    // TODO: consider moving it to constructor.
    void set_allocator(dnnl::impl::graph::allocator_t *alloc) {
        impl_->set_allocator(alloc);
    }
#endif

    void retain() { counter_++; }

    void release() {
        if (--counter_ == 0) { delete this; }
    }

protected:
    dnnl::impl::status_t init_impl() { return impl_->init(); }
    virtual ~dnnl_engine() = default;

private:
    std::unique_ptr<dnnl::impl::engine_impl_t> impl_;
    std::atomic<int> counter_;
};

namespace dnnl {
namespace impl {

inline runtime_kind_t get_default_runtime(engine_kind_t kind) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu) return runtime_kind::ocl;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (kind == engine_kind::gpu) return runtime_kind::sycl;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    if (kind == engine_kind::gpu) return runtime_kind::ze;
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    return runtime_kind::sycl;
#else
    return runtime_kind::none;
#endif
}

inline runtime_kind_t get_cpu_native_runtime() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
    return runtime_kind::seq;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    return runtime_kind::omp;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
    return runtime_kind::tbb;
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#else
    return runtime_kind::none;
#endif
}

struct engine_factory_t {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

struct engine_deleter_t {
    void operator()(engine_t *e) const { e->release(); }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
