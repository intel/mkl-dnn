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

#include "common/stream_impl.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {

bool stream_impl_t::is_verbose_profiler_enabled() const {
    return get_verbose(verbose_t::exec_profile) && use_verbose_profiler_;
}

status_t stream_impl_t::init_verbose_profiler(engine_kind_t) {
    use_verbose_profiler_ = false;
    return status::success;
}

} // namespace impl
} // namespace dnnl
