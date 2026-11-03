/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_STREAM_IMPL_HPP
#define COMMON_STREAM_IMPL_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

class stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags)
        : use_verbose_profiler_(false), flags_(flags) {}

    virtual ~stream_impl_t() = default;

    unsigned flags() const { return flags_; }

    bool is_profiling_enabled() const {
        return (flags() & dnnl::impl::stream_flags::profiling);
    }

    bool is_verbose_profiler_enabled() const;

    // Returns true if the stream can be synchronized (blocked on) around
    // primitive execution without breaking its execution model.
    virtual bool is_syncable() const { return true; }

    virtual status_t init_verbose_profiler(engine_kind_t);

protected:
    bool use_verbose_profiler_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t)
    unsigned flags_;
};

} // namespace impl
} // namespace dnnl

#endif
