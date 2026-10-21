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

#ifndef COMMON_STREAM_HPP
#define COMMON_STREAM_HPP

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream_impl.hpp"
#include "common/utils.hpp"

struct dnnl_stream {
    dnnl_stream(dnnl::impl::engine_t *engine, dnnl::impl::stream_impl_t *impl)
        : engine_(engine), impl_(impl) {}
    virtual ~dnnl_stream() = default;

    /** returns stream's engine */
    dnnl::impl::engine_t *engine() const { return engine_; }
    template <typename tgt_engine_t>
    tgt_engine_t *engine() const {
        return dnnl::impl::utils::downcast<tgt_engine_t *>(engine_);
    }

    /** returns stream's kind */
    unsigned flags() const { return impl_->flags(); }

    virtual dnnl::impl::status_t enqueue_primitive(
            const primitive_iface_t *primitive_iface,
            dnnl::impl::exec_ctx_t &ctx);

    /** blocks until all submitted primitives to the stream are completed */
    virtual dnnl::impl::status_t wait() = 0;

    virtual void before_exec_hook() {}
    virtual void after_exec_hook() {}

    virtual dnnl::impl::status_t reset_profiling() {
        if (!is_profiling_enabled())
            return dnnl::impl::status::invalid_arguments;
        return dnnl::impl::status::unimplemented;
    }

    virtual dnnl::impl::status_t get_profiling_data(
            dnnl::impl::profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const {
        if (!is_profiling_enabled())
            return dnnl::impl::status::invalid_arguments;
        return dnnl::impl::status::unimplemented;
    }

    virtual dnnl::impl::status_t notify_profiling_complete() const {
        if (!is_profiling_enabled())
            return dnnl::impl::status::invalid_arguments;
        return dnnl::impl::status::unimplemented;
    }

    bool is_profiling_enabled() const { return impl_->is_profiling_enabled(); }

    bool is_verbose_profiler_enabled() const {
        return impl_->is_verbose_profiler_enabled();
    }

    virtual dnnl::impl::status_t run_verbose_profiler(
            const std::string &pd_info, double start_ms, uint64_t component) {
        return dnnl::impl::status::unimplemented;
    }

    virtual dnnl::impl::status_t zero_pad(const dnnl::impl::memory_t *memory,
            const dnnl::impl::exec_ctx_t &ctx);

    dnnl::impl::stream_impl_t *impl() const { return impl_.get(); }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::impl::status_t get_threadpool(
            dnnl::threadpool_interop::threadpool_iface **threadpool) const {
        using namespace dnnl::impl;
        if (engine_->kind() != engine_kind::cpu)
            return status::invalid_arguments;
        return impl_->get_threadpool(threadpool);
    }
#endif

protected:
    dnnl::impl::engine_t *engine_;
    std::unique_ptr<dnnl::impl::stream_impl_t> impl_;
};

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
