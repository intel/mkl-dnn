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

#include "common/engine_impl.hpp"

#ifdef ONEDNN_BUILD_GRAPH
#include "graph/interface/allocator.hpp"
#endif

namespace dnnl {
namespace impl {

engine_impl_t::engine_impl_t(
        engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
    : kind_(kind)
    , runtime_kind_(runtime_kind)
    , index_(index)
#ifdef ONEDNN_BUILD_GRAPH
    , allocator_(utils::make_unique<graph::allocator_t>())
#endif
{
}

engine_impl_t::~engine_impl_t() = default;

#ifdef ONEDNN_BUILD_GRAPH
void engine_impl_t::set_allocator(graph::allocator_t *alloc) {
    *allocator_ = *alloc;
}
#endif

} // namespace impl
} // namespace dnnl
