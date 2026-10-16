/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"

#include "engine.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_desc_iterator.hpp"
#include "primitive_iface.hpp"
#include "reorder_primitive_desc_iface.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {

status_t primitive_desc_create(primitive_desc_iface_t **primitive_desc_iface,
        engine_t *engine, const op_desc_t *op_desc,
        const primitive_desc_iface_t *hint_fwd_pd, const primitive_attr_t *attr,
        engine_t *src_engine, engine_t *dst_engine) {
    using namespace primitive_kind;

    if (!primitive_desc_iface) return invalid_arguments;

    const bool known_primitive_kind = utils::one_of(op_desc->primitive_kind,
            batch_normalization, binary, convolution, deconvolution, eltwise,
            gemm, group_normalization, inner_product, layer_normalization, lrn,
            matmul, pooling, prelu, reduction, reorder, resampling, rnn, sdpa,
            shuffle, softmax);
    if (!known_primitive_kind) return invalid_arguments;

    std::unique_ptr<primitive_desc_iface_t> pd_iface;
    // Reorder is the only kind that dispatches over two engines. It relies on a
    // dedicated iface subclass to thread `src_engine` and `dst_engine` to both
    // the iterator and the created primitive.
    if (op_desc->primitive_kind == reorder) {
        pd_iface = utils::make_unique<reorder_primitive_desc_iface_t>(
                engine, op_desc, attr, src_engine, dst_engine);
    } else {
        pd_iface = utils::make_unique<primitive_desc_iface_t>(engine, op_desc,
                attr, hint_fwd_pd ? hint_fwd_pd->impl().get() : nullptr);
    }
    if (pd_iface == nullptr) return out_of_memory;
    CHECK(pd_iface->init());

    *primitive_desc_iface = pd_iface.release();

    return success;
}

} // namespace impl
} // namespace dnnl

dnnl_primitive_desc::dnnl_primitive_desc(
        const std::shared_ptr<primitive_desc_t> &pd, const engine_t *engine)
    : pd_(pd), engine_(engine) {}

dnnl_primitive_desc::dnnl_primitive_desc(const engine_t *engine,
        const op_desc_t *op_desc, const primitive_attr_t *attr,
        const primitive_desc_t *hint_fwd_pd, const engine_t *src_engine,
        const engine_t *dst_engine) {

    pd_iterator_ = utils::make_unique<primitive_desc_iterator_t>(engine,
            op_desc, attr, hint_fwd_pd, /* skip_idx = */ -1, src_engine,
            dst_engine);
}

status_t dnnl_primitive_desc::init() {
    if (!pd_iterator_) return status::out_of_memory;
    if (!pd_iterator_->is_initialized()) return out_of_memory;

    ++(*pd_iterator_);
    if (*pd_iterator_ == pd_iterator_->end())
        return pd_iterator_->no_impl_status();

    pd_ = *(*pd_iterator_);
    engine_ = pd_iterator_->engine();

    return success;
}

status_t dnnl_primitive_desc::next_impl() {
    if (!pd_iterator_) return status::last_impl_reached;
    ++(*pd_iterator_);
    if (*pd_iterator_ == pd_iterator_->end()) return last_impl_reached;
    pd_ = *(*pd_iterator_);
    return status::success;
}

status_t dnnl_primitive_desc::create_primitive_iface(
        std::pair<primitive_iface_t *, cache_state_t> &primitive_iface,
        const cache_blob_t &cache_blob) const {
    // Step 1: create impl::primitive_t or get it from primitive cache
    std::pair<std::shared_ptr<primitive_t>, cache_state_t> p;
    // Top level primitive can be fetched from the primitive cache since it's
    // fetching it faster.
    constexpr bool force_create_from_blob = false;
    auto status = impl()->create_primitive(
            p, engine(), cache_blob, force_create_from_blob);
    if (status != status::success) return status;
    // Step 2: create primitive_iface_t, init and return it to user
    primitive_iface_t *p_iface = nullptr;
    CHECK(safe_ptr_assign(p_iface, new primitive_iface_t(p.first, engine())));
    status = p_iface->init();
    if (status != status::success) {
        p_iface->release();
        return status;
    }
    primitive_iface = std::make_pair(p_iface, p.second);
    return status::success;
}

const std::shared_ptr<primitive_desc_t> &dnnl_primitive_desc::impl() const {
    return pd_;
}

dnnl::impl::engine_t *dnnl_primitive_desc::engine() const {
    // This method is passed to primitive_create which is not yet converted to
    // `const engine_t *`. This `const_cast` will be automatically removed once
    // it changes.
    return const_cast<dnnl::impl::engine_t *>(engine_);
}
const dnnl::impl::primitive_attr_t *dnnl_primitive_desc::attr() const {
    return impl()->attr();
}

const char *dnnl_primitive_desc::info() const {
    return impl()->info(engine_);
}

std::string dnnl_primitive_desc::info_with_runtime_dims(
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *bia_md, const memory_desc_t *dst_md) const {
    return impl()->info_with_runtime_dims(
            engine_, src_md, wei_md, bia_md, dst_md);
}

dnnl::impl::engine_t *dnnl_primitive_desc::src_engine() const {
    return engine();
}
dnnl::impl::engine_t *dnnl_primitive_desc::dst_engine() const {
    return engine();
}

dnnl::impl::engine_t *dnnl_primitive_desc::scratchpad_engine() const {
    return engine();
}

status_t dnnl_primitive_desc::query(query_t what, int idx, void *result) const {
    auto status = status::success;
    switch (what) {
        case query::engine: *(engine_t **)result = engine(); break;
        case query::cache_blob_id_size_s64:
            *(dim_t *)result
                    = (dim_t)impl()->get_cache_blob_id(engine()).size();
            break;
        case query::cache_blob_id:
            *(const uint8_t **)result
                    = impl()->get_cache_blob_id(engine()).empty()
                    ? nullptr
                    : impl()->get_cache_blob_id(engine()).data();
            break;

        default: status = impl()->query(what, idx, result);
    }
    return status;
}

status_t dnnl_primitive_desc_get_attr(
        const primitive_desc_iface_t *primitive_desc_iface,
        const primitive_attr_t **attr) {
    if (utils::any_null(primitive_desc_iface, attr)) return invalid_arguments;

    *attr = primitive_desc_iface->attr();
    return success;
}

status_t dnnl_primitive_desc_clone(
        primitive_desc_iface_t **primitive_desc_iface,
        const primitive_desc_iface_t *existing_primitive_desc_iface) {
    if (utils::any_null(primitive_desc_iface, existing_primitive_desc_iface))
        return invalid_arguments;

    return safe_ptr_assign(*primitive_desc_iface,
            new primitive_desc_iface_t(existing_primitive_desc_iface->impl(),
                    existing_primitive_desc_iface->engine()));
}

status_t dnnl_primitive_desc_destroy(
        primitive_desc_iface_t *primitive_desc_iface) {
    delete primitive_desc_iface;
    return success;
}

status_t dnnl_primitive_desc_next_impl(
        primitive_desc_iface_t *primitive_desc_iface) {
    if (!primitive_desc_iface) return invalid_arguments;
    return primitive_desc_iface->next_impl();
}
