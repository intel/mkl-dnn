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

#ifndef UTILS_STREAM_KIND_HPP
#define UTILS_STREAM_KIND_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "common.hpp"
#include "utils/engine.hpp"

#include <future>
#include <sstream>

enum class stream_kind_t {
    // The library-defined stream kind.
    def = 0x0,
    in_order = 0x1,
    out_of_order = 0x2,
};

extern stream_kind_t stream_kind; // user stream kind
extern stream_kind_t default_stream_kind; // the default stream kind

dnnl_stream_flags_t stream_kind2stream_flags(
        stream_kind_t stream_kind, bool use_profiling);

stream_kind_t str2stream_kind(const char *str);

std::ostream &operator<<(std::ostream &s, stream_kind_t stream_kind);

struct stream_t {
    stream_t() = default;
    stream_t(const engine_t &engine, void *interop_obj = nullptr);
    operator dnnl_stream_t() const {
        return stream_.get(/* allow_empty = */ true);
    }
    operator dnnl::stream &() { return stream_; }
    // Wrapper over dnnl::stream::wait() to avoid explicit casts to dnnl::stream
    // at graph driver call sites.
    void wait() { stream_.wait(); }
    stream_t &operator=(stream_t &&rhs) = default;

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
    dnnl::stream stream_;
};

struct stream_staller_t {
    // Enqueue tasks to stall a primitive execution tasks for asynchronous
    // threadpool runtime. For rest runtimes does nothing.
    stream_staller_t(stream_t &stream);

    // A signal the submission has completed and ready for execution.
    void release();

private:
    std::promise<void> prom_;
};

#endif
