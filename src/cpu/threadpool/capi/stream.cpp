/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <memory>

#include "oneapi/dnnl/dnnl_threadpool.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/threadpool/stream_impl.hpp"

using namespace dnnl::impl;

dnnl_status_t dnnl_threadpool_interop_stream_create(
        stream_t **stream, engine_t *engine, void *threadpool) {
    bool args_ok = !utils::any_null(stream, engine)
            && engine->runtime_kind() == runtime_kind::threadpool;
    if (!args_ok) return status::invalid_arguments;

    auto *tp = static_cast<dnnl::threadpool_interop::threadpool_iface *>(
            threadpool);

    std::unique_ptr<dnnl::impl::stream_impl_t> stream_impl(
            new cpu::threadpool::stream_impl_t(tp));
    if (!stream_impl) return status::out_of_memory;

    CHECK(engine->create_stream(stream, stream_impl.get()));
    // stream takes ownership of stream_impl if create_stream() succeeds.
    stream_impl.release();
    return status::success;
}

dnnl_status_t dnnl_threadpool_interop_stream_get_threadpool(
        dnnl_stream_t stream, void **threadpool) {
    bool args_ok = !utils::any_null(stream, threadpool)
            && stream->engine()->runtime_kind() == runtime_kind::threadpool;
    if (!args_ok) return status::invalid_arguments;

    const auto *tp_stream_impl
            = utils::downcast<const cpu::threadpool::stream_impl_t *>(
                    stream->impl());
    *threadpool = static_cast<void *>(tp_stream_impl->get_threadpool());
    return status::success;
}
