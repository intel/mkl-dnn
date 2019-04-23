/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <algorithm>

#include "sycl_memory_storage.hpp"

#include "common/guard_manager.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace sycl {

struct map_tag;
struct use_host_ptr_tag;

sycl_memory_storage_t::sycl_memory_storage_t(
        engine_t *engine, unsigned flags, size_t size, void *handle)
    : memory_storage_t(engine) {
    // Do not allocate memory if one of these is true:
    // 1) size is 0
    // 2) handle is nullptr and flags have use_backend_ptr
    if ((size == 0) || (!handle && (flags & memory_flags_t::alloc) == 0))
        return;

    if (flags & memory_flags_t::alloc) {
#if MKLDNN_ENABLE_SYCL_VPTR
        vptr_ = mkldnn::sycl_malloc(size);
        is_owned_ = true;
#else
        buffer_.reset(new untyped_sycl_buffer_t(data_type::u8, size));
#endif
    } else if (flags & memory_flags_t::use_backend_ptr) {
#if MKLDNN_ENABLE_SYCL_VPTR
        assert(mkldnn::is_sycl_vptr(handle));
        vptr_ = handle;
        is_owned_ = false;
#else
        auto *untyped_buf = static_cast<untyped_sycl_buffer_t *>(handle);
        auto &buf = untyped_buf->sycl_buffer<uint8_t>();
        buffer_.reset(new untyped_sycl_buffer_t(buf));
#endif
    } else if (flags & memory_flags_t::use_host_ptr) {
#if MKLDNN_ENABLE_SYCL_VPTR
        vptr_ = mkldnn::sycl_malloc(size);
        is_owned_ = true;

        auto buf = mkldnn::get_sycl_buffer(vptr_);
        {
            auto acc = buf.get_access<cl::sycl::access::mode::write>();
            uint8_t *handle_u8 = static_cast<uint8_t *>(handle);
            for (size_t i = 0; i < size; i++)
                acc[i] = handle_u8[i];
        }

        is_write_host_back_ = true;

        auto &guard_manager = guard_manager_t<use_host_ptr_tag>::instance();
        guard_manager.enter(this, [=]() {
            auto buf = mkldnn::get_sycl_buffer(vptr_);
            auto acc = buf.get_access<cl::sycl::access::mode::read>();
            uint8_t *handle_u8 = static_cast<uint8_t *>(handle);
            for (size_t i = 0; i < size; i++)
                handle_u8[i] = acc[i];
        });
#else
        buffer_.reset(new untyped_sycl_buffer_t(handle, data_type::u8, size));
#endif
    }
}

#if MKLDNN_ENABLE_SYCL_VPTR
sycl_memory_storage_t::~sycl_memory_storage_t() {
    if (is_write_host_back_) {
        auto &guard_manager = guard_manager_t<use_host_ptr_tag>::instance();
        guard_manager.exit(this);
    }
    if (is_owned_ && vptr_) {
        mkldnn::sycl_free(vptr_);
    }
}
#endif

status_t sycl_memory_storage_t::map_data(void **mapped_ptr) const {
#if MKLDNN_ENABLE_SYCL_VPTR
    if (!vptr_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#else
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }
#endif

    auto &guard_manager = guard_manager_t<map_tag>::instance();

#if MKLDNN_ENABLE_SYCL_VPTR
    auto buf = mkldnn::get_sycl_buffer(vptr_);

    auto acc = buf.get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };
#else
    std::function<void()> unmap_callback;
    *mapped_ptr = buffer_->map_data<cl::sycl::access::mode::read_write>(
            [&](std::function<void()> f) { unmap_callback = f; });

#endif
    return guard_manager.enter(this, unmap_callback);
}

status_t sycl_memory_storage_t::unmap_data(void *mapped_ptr) const {
    if (!mapped_ptr)
        return status::success;

    auto &guard_manager = guard_manager_t<map_tag>::instance();
    return guard_manager.exit(this);
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn
