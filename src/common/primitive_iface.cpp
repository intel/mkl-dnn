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

#include <string>

#include "c_types_map.hpp"
#include "engine.hpp"

#include "ittnotify.hpp"

#include "cache_hit_types.hpp"
#include "primitive.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_exec_types.hpp"
#include "primitive_iface.hpp"
#include "profiler.hpp"
#include "reorder_primitive_desc_iface.hpp"
#include "scratchpad_debug.hpp"
#include "stack_checker.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const exec_args_t &args) {
    for (const auto &arg : args) {
        if (arg.second.is_const()) continue;
        auto *mem = arg.second.mem();
        void *p;
        mem->get_data_handle(&p);
        size_t s = memory_desc_wrapper(*mem->md()).size();
        msan_unpoison(p, s);
    }
}
} // namespace

namespace dnnl {
namespace impl {

status_t primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface,
        const cache_blob_t &cache_blob = cache_blob_t()) {

    std::pair<primitive_iface_t *, cache_state_t> p_iface;

    const bool enable_itt = itt::get_itt(itt::__itt_task_level_low);
    if (enable_itt) {
        itt::primitive_task_start(
                primitive_desc_iface->impl()->kind(), VERBOSE_create);
    }

    if (get_verbose(verbose_t::create_profile,
                prim_kind2_comp_kind(primitive_desc_iface->impl()->kind()))) {
        double start_ms = get_msec();
        CHECK(primitive_desc_iface->create_primitive_iface(
                p_iface, cache_blob));
        double duration_ms = get_msec() - start_ms;

        // Since the primitive_hit has a higher fetching priority,
        // it shouldn't be overrided by a persistent_hit.
        if (cache_blob && p_iface.second != cache_state_t::primitive_hit)
            p_iface.second = cache_state_t::persistent_hit;
        const char *str = cache_state2str(p_iface.second);

        VPROF(start_ms, primitive, create, str, p_iface.first->pd()->info(),
                duration_ms);
    } else {
        CHECK(primitive_desc_iface->create_primitive_iface(
                p_iface, cache_blob));
    }

    if (enable_itt) {
        // Before metadata is added to the ITT task, the `info()` string is
        // checked for initialization.
        // An empty info_ string for cache states other than `cache_state_t::miss`
        // implies kernel or nested primitive operations.
        // In that case, the metadata addition is skipped to avoid additional
        // overheads.
        const char *pd_info = p_iface.first->pd()->impl()->info(
                primitive_desc_iface->engine(), false);
        if ((p_iface.second == cache_state_t::miss) || (pd_info && *pd_info)) {
#if defined(DNNL_ENABLE_ITT_TASKS)
            const __itt_id task_id
                    = itt::make_itt_id(p_iface.first->pd()->info(), get_msec());
            itt::primitive_add_metadata_and_id(
                    p_iface.first->pd()->info(), VERBOSE_create, &task_id);
#endif
        }

        itt::primitive_task_end(VERBOSE_create);
    }

    return safe_ptr_assign((*primitive_iface), p_iface.first);
}

status_t primitive_execute(
        const primitive_iface_t *primitive_iface, exec_ctx_t &ctx) {
    auto stream = ctx.stream();
    status_t status = success;
    auto pd = primitive_iface->pd();

    const bool enable_itt = itt::get_itt(itt::__itt_task_level_low);
    if (enable_itt) {
        itt::primitive_task_start(pd->impl()->kind(), VERBOSE_exec);

        // Before metadata is added to the ITT task, the `info()` string is
        // checked for initialization.
        // An empty info_ string implies kernel or nested primitive operations,
        // in which case the metadata addition is skipped to avoid additional
        // overheads.
        const auto *pd_info
                = pd->impl()->info(primitive_iface->engine(), false);
        if (pd_info && *pd_info) {
#if defined(DNNL_ENABLE_ITT_TASKS)
            // ITT task IDs are uniquely assigned using pd->info() and
            // timestamps - the timestamps helps distinguish between different
            // instances of the same configuration
            const __itt_id task_id = itt::make_itt_id(pd_info, get_msec());
            itt::primitive_add_metadata_and_id(pd_info, VERBOSE_exec, &task_id);
#endif
        }
    }

    if (get_verbose(verbose_t::exec_profile,
                prim_kind2_comp_kind(pd->impl()->kind()))) {
        std::string pd_info;

        if (pd->impl()->has_runtime_dims_or_strides()) {
            // Take out mds from `ctx` here to avoid primitive_desc dependency
            // on `exec_ctx_t` type.
            // TODO: invariant arg names for training?
            const auto pd_src_md = pd->impl()->invariant_src_md();
            const auto src_md = ctx.memory_mdw(DNNL_ARG_SRC, pd_src_md).md_;
            const auto pd_wei_md = pd->impl()->invariant_wei_md();
            const auto wei_md = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd_wei_md).md_;
            const auto pd_bia_md = pd->impl()->invariant_bia_md();
            const auto bia_md = ctx.memory_mdw(DNNL_ARG_BIAS, pd_bia_md).md_;
            const auto pd_dst_md = pd->impl()->invariant_dst_md();
            const auto dst_md = ctx.memory_mdw(DNNL_ARG_DST, pd_dst_md).md_;
            pd_info = pd->info_with_runtime_dims(
                    src_md, wei_md, bia_md, dst_md);
        } else {
            pd_info = pd->info();
        }

        // Block around execution to measure host-side timing, unless the runtime
        // supports non-blocking device-measured profiling (GPU runtimes and
        // async threadpool CPU).
        const bool block_on_wait = !stream->is_verbose_profiler_enabled()
                && stream->is_syncable();

        if (block_on_wait) stream->wait();
        double start_ms = get_msec();
        status = stream->enqueue_primitive(primitive_iface, ctx);
        if (block_on_wait) stream->wait();

        if (stream->is_verbose_profiler_enabled()) {
            CHECK(stream->run_verbose_profiler(pd_info, start_ms, 0));
        } else {
            double duration_ms = get_msec() - start_ms;
            VPROF(start_ms, primitive, exec, VERBOSE_profile, pd_info.c_str(),
                    duration_ms);
        }
    } else {
        status = stream->enqueue_primitive(primitive_iface, ctx);
    }

    if (enable_itt) itt::primitive_task_end(VERBOSE_exec);

    if (msan_enabled) unpoison_outputs(ctx.args());

    return status;
}

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;

#ifdef DNNL_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("dnnl_primitive_create");
    bool is_wino = std::string(primitive_desc_iface->info()).find("wino")
            != std::string::npos;

    if (!is_wino) {
        const cache_blob_t dummy;
        return sc.check(dnnl::impl::primitive_create, primitive_iface,
                primitive_desc_iface, std::ref(dummy));
    }
#endif
    return dnnl::impl::primitive_create(primitive_iface, primitive_desc_iface);
}

status_t dnnl_primitive_create_from_cache_blob(
        primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface, size_t size,
        const uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, primitive_desc_iface, cache_blob)
            || size == 0) {
        return invalid_arguments;
    }
    if (!primitive_desc_iface->engine()->is_cache_blob_supported()) {
        return status::unimplemented;
    }

    cache_blob_t cb(const_cast<uint8_t *>(cache_blob), size);
    return dnnl::impl::primitive_create(
            primitive_iface, primitive_desc_iface, cb);
}

status_t dnnl_primitive_execute(const primitive_iface_t *primitive_iface,
        stream_t *stream, int nargs, const dnnl_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, c_args, args);
    if (status != status::success) return status;

    stream->before_exec_hook();

    exec_ctx_t ctx(stream, std::move(args));
#ifdef DNNL_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("dnnl_primitive_execute");
    const auto *pd_iface = primitive_iface->pd();
    bool is_wino
            = std::string(pd_iface->info()).find("wino") != std::string::npos;
    if (!is_wino) {
        status = sc.check(
                dnnl::impl::primitive_execute, primitive_iface, std::ref(ctx));
    }
#else
    status = dnnl::impl::primitive_execute(primitive_iface, ctx);
#endif
    stream->after_exec_hook();

    return status;
}

status_t dnnl_primitive_get_primitive_desc(
        const primitive_iface_t *primitive_iface,
        const primitive_desc_iface_t **primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return safe_ptr_assign(*primitive_desc_iface, primitive_iface->pd());
}

status_t dnnl_primitive_get_cache_blob(const primitive_iface_t *primitive_iface,
        size_t *size, uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, size)) {
        return status::invalid_arguments;
    }

    if (!primitive_iface->engine()->is_cache_blob_supported()) {
        return status::unimplemented;
    }

    if (!cache_blob) {
        size_t sz = 0;
        CHECK(primitive_iface->get_cache_blob_size(&sz));
        (*size) = sz;
        return status::success;
    }

    cache_blob_t cb(cache_blob, *size);
    return primitive_iface->get_cache_blob(cb);
}

status_t dnnl_primitive_destroy(primitive_iface_t *primitive_iface) {
    if (primitive_iface != nullptr) primitive_iface->release();
    return success;
}

// primitive_iface_t implementation
dnnl_primitive::dnnl_primitive(
        const std::shared_ptr<primitive_t> &primitive, engine_t *engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<primitive_desc_iface_t>(
              primitive_->pd(), engine)) {}

// reorder specialization
dnnl_primitive::dnnl_primitive(const std::shared_ptr<primitive_t> &primitive,
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<reorder_primitive_desc_iface_t>(
              primitive_->pd(), engine, src_engine, dst_engine)) {}

dnnl_primitive::~dnnl_primitive() {
    if (scratchpad_debug::is_protect_scratchpad() && scratchpad_ != nullptr
            && scratchpad_->get_memory_storage() != nullptr) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        scratchpad_debug::unprotect_scratchpad_buffer(
                scratchpad_->get_memory_storage(), registry);
    }
}

status_t dnnl_primitive::init() {
    const size_t scratchpad_size
            = primitive_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        bool use_global_scratchpad = scratchpad_debug::is_protect_scratchpad()
                ? false
                : primitive_->use_global_scratchpad();
        auto *scratchpad_ptr = create_scratchpad(
                pd_->engine(), scratchpad_size, use_global_scratchpad);
        if (scratchpad_ptr == nullptr) return out_of_memory;
        if (scratchpad_ptr->get_memory_storage() == nullptr) {
            delete scratchpad_ptr;
            return out_of_memory;
        }

        if (scratchpad_debug::is_protect_scratchpad()) {
            scratchpad_debug::protect_scratchpad_buffer(
                    scratchpad_ptr->get_memory_storage(), registry);
        }
        scratchpad_.reset(scratchpad_ptr);
        if (scratchpad_ptr->size() < scratchpad_size) return out_of_memory;
    }
    return primitive_->create_resource(pd()->engine(), resource_mapper_);
}

engine_t *dnnl_primitive::engine() const {
    return pd_->engine();
}

const primitive_desc_iface_t *dnnl_primitive::pd() const {
    return pd_.get();
}

status_t dnnl_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_->pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(DNNL_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                                        : nullptr;
    } else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    // Obtain a scratchpad memory storage host ptr from the context.
    // `require_host_ptr = true` guarantees a nullptr will be returned if the
    // usage model doesn't assume memory mapping.
    // It's the only pointer that a grantor object requires to provide to
    // sub-storages for nested primitives.
    const void *mapped_mem_storage_ptr
            = ctx.host_ptr(mem_storage, /* require_host_ptr = */ true);

    auto *scratchpad_grantor
            = primitive_->pd()->scratchpad_registry().create_grantor(
                    mem_storage, mapped_mem_storage_ptr);
    ctx.set_scratchpad_grantor(scratchpad_grantor);
    ctx.set_resource_mapper(&resource_mapper_);

    auto status = primitive_->execute(ctx);
    return status;
}

status_t dnnl_primitive::get_cache_blob_size(size_t *size) const {
    return primitive_->get_cache_blob_size(engine(), size);
}

status_t dnnl_primitive::get_cache_blob(cache_blob_t cache_blob) const {
    return primitive_->get_cache_blob(engine(), cache_blob);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
