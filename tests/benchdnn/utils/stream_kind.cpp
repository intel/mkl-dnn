/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "utils/stream_kind.hpp"
#include "common.hpp"
#include "utils/bench_mode.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"

#include "utils/dnnl_query.hpp"

#include "tests/test_thread_decl.hpp"

#include <thread>
#endif

stream_kind_t default_stream_kind {stream_kind_t::def};
stream_kind_t stream_kind {default_stream_kind};

dnnl_stream_flags_t stream_kind2stream_flags(
        stream_kind_t stream_kind, bool use_profiling) {
    dnnl_stream_flags_t flags = dnnl_stream_default_flags;
    switch (stream_kind) {
        case stream_kind_t::def: break;
        case stream_kind_t::in_order: flags = dnnl_stream_in_order; break;
        case stream_kind_t::out_of_order:
            flags = dnnl_stream_out_of_order;
            break;
        default: SAFE_V(FAIL);
    }

#ifdef DNNL_EXPERIMENTAL_PROFILING
    dnnl_stream_flags_t profiling_flag = dnnl_stream_profiling;
#else
    dnnl_stream_flags_t profiling_flag = static_cast<dnnl_stream_flags_t>(0x4);
#endif
    if (use_profiling)
        flags = static_cast<dnnl_stream_flags_t>(flags | profiling_flag);
    return flags;
}

stream_kind_t str2stream_kind(const char *str) {
#define CASE(param) \
    if (!strcasecmp(#param, str)) return stream_kind_t::param

    CASE(def);
    CASE(in_order);
    CASE(out_of_order);

#undef CASE

    BENCHDNN_PRINT(
            0, "Error: stream kind value \'%s\' is not recognized.\n", str);
    SAFE_V(FAIL);
    return stream_kind_t::def;
}

std::ostream &operator<<(std::ostream &s, stream_kind_t stream_kind) {
    if (stream_kind == stream_kind_t::def) s << "def";
    if (stream_kind == stream_kind_t::in_order) s << "in_order";
    if (stream_kind == stream_kind_t::out_of_order) s << "out_of_order";

    return s;
}

stream_t::stream_t(const engine_t &engine, void *interop_obj) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (is_cpu(engine)) {
        auto tp = static_cast<dnnl::threadpool_interop::threadpool_iface *>(
                interop_obj);
        if (tp == nullptr) tp = dnnl::testing::get_threadpool();
        stream_ = dnnl::threadpool_interop::make_stream(engine, tp);
        return;
    }
#endif

    const bool use_profiling = has_bench_mode_bit(mode_bit_t::perf)
            && is_gpu(engine) && !is_nvidia_gpu(engine) && !is_amd_gpu(engine);
    const auto flags = static_cast<dnnl::stream::flags>(
            stream_kind2stream_flags(stream_kind, use_profiling));
    stream_ = dnnl::stream(engine, flags);
}

stream_staller_t::stream_staller_t(stream_t &stream) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto eng = query_engine(stream);
    auto eng_kind = query_engine_kind(eng);
    if (eng_kind != dnnl_cpu) return;

    auto tp = dnnl::threadpool_interop::get_threadpool(stream);

    // `tp` is not expected to be empty for CPU streams with threadpol runtime.
    if (!tp) SAFE_V(FAIL);

    // Only relevant for asynchronous threadpool, synchronous will
    // deadlock.
    if (tp->get_flags()
            != dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS)
        return;

    // Each thread from the threadpool should get the task to be stalled.
    const int num_tasks = tp->get_num_threads();

    // The main thread must be let through, otherwise it deadlocks as
    // task submission won't happen.
    std::thread::id main_thr_id = std::this_thread::get_id();

    // Shared future allows to pass all waiting threads at once inside the
    // palallel call.
    std::shared_future<void> fut(prom_.get_future());

    tp->parallel_for(num_tasks, [=](int, int) {
        std::thread::id id_thr = std::this_thread::get_id();
        if (id_thr != main_thr_id) fut.wait();
    });
#endif
}

void stream_staller_t::release() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    prom_.set_value();
#endif
}
