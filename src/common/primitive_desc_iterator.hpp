/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_DESC_ITERATOR_HPP
#define COMMON_PRIMITIVE_DESC_ITERATOR_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "primitive_attr.hpp"
#include "primitive_cache.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;

struct primitive_desc_iterator_t {
    primitive_desc_iterator_t(const engine_t *engine, const op_desc_t *op_desc,
            const primitive_attr_t *attr, const primitive_desc_t *hint_fwd_pd,
            int skip_idx = -1, const engine_t *src_engine = nullptr,
            const engine_t *dst_engine = nullptr)
        : idx_(-1)
        , engine_(engine)
        , op_desc_(op_desc->clone())
        , attr_(attr ? *attr : primitive_attr_t())
        , hint_fwd_pd_(hint_fwd_pd)
        , impl_list_(engine_->get_implementation_list(op_desc_.get()))
        , last_idx_(0)
        , skip_idx_(skip_idx)
        , offset_(-1)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine) {

        while (impl_list_[last_idx_])
            ++last_idx_;
    }

    const engine_t *engine() const { return engine_; }

    bool operator==(const primitive_desc_iterator_t &rhs) const {
        return idx_ == rhs.idx_ && engine_ == rhs.engine_;
    }
    bool operator!=(const primitive_desc_iterator_t &rhs) const {
        return !operator==(rhs);
    }

    primitive_desc_iterator_t end() const {
        return primitive_desc_iterator_t(engine_, last_idx_);
    }

    primitive_desc_iterator_t &operator++() {
        // Quick return to preserve state of the iterator that reached the end.
        // The state is equal to the state of the iterator that end() returns.
        if (idx_ == last_idx_) return *this;

        offset_++;
        pd_.reset();

        std::vector<dnnl::impl::memory_desc_t> hint_mds;
        if (hint_fwd_pd_) hint_mds = hint_fwd_pd_->hint_mds(true /* is_hint */);
        primitive_hashing::key_t key(
                engine_, op_desc_.get(), &attr_, offset_, hint_mds, skip_idx_);

        pd_ = primitive_cache().get_pd(key);
        if (pd_) {
            has_found_impl_ = true;
            return *this;
        }

        while (++idx_ != last_idx_) {
            if (idx_ == skip_idx_) continue;
            primitive_desc_t *candidate_pd = nullptr;
            auto s = impl_list_[idx_](&candidate_pd, op_desc_.get(), &attr_,
                    engine_, hint_fwd_pd_, offset_, skip_idx_, src_engine_,
                    dst_engine_);
            if (s == status::success) {
                pd_.reset(candidate_pd);
                has_found_impl_ = true;
                break;
            }
            if (s == status::unsupported_data_type) has_unsupported_dt_ = true;
        }
        return *this;
    }

    const std::shared_ptr<primitive_desc_t> &operator*() const { return pd_; }

    const primitive_attr_t &attr() const { return attr_; }

    // The iterator is only usable if the attributes it copied were allocated
    // successfully; delegate the check to the copied attributes.
    bool is_initialized() const { return attr_.is_initialized(); }

    // Status to report when no suitable implementation was found. Unsupported
    // data type is a platform-wide limitation, so a single implementation
    // reporting it is enough. However, if any implementation was created and
    // then rejected by the caller (e.g. when creating a nested primitive), the
    // data types are supported and the reason is something else.
    status_t no_impl_status() const {
        return has_unsupported_dt_ && !has_found_impl_
                ? status::unsupported_data_type
                : status::unimplemented;
    }

protected:
    int idx_;
    const engine_t *engine_;
    std::shared_ptr<primitive_desc_t> pd_;
    std::unique_ptr<op_desc_t> op_desc_;
    const primitive_attr_t attr_;
    const primitive_desc_t *hint_fwd_pd_;
    const impl_list_item_t *impl_list_;
    int last_idx_;
    int skip_idx_;
    int offset_;
    const engine_t *src_engine_;
    const engine_t *dst_engine_;
    bool has_unsupported_dt_ = false;
    bool has_found_impl_ = false;

private:
    primitive_desc_iterator_t(const engine_t *engine, int last_idx)
        : idx_(last_idx)
        , engine_(engine)
        , hint_fwd_pd_(nullptr)
        , impl_list_(nullptr)
        , last_idx_(last_idx)
        , skip_idx_(-1)
        , offset_(-1)
        , src_engine_(engine)
        , dst_engine_(engine) {}

    primitive_desc_iterator_t(primitive_desc_iterator_t &&other)
        : idx_(other.idx_)
        , engine_(other.engine_)
        , pd_(std::move(other.pd_))
        , op_desc_(std::move(other.op_desc_))
        , attr_(other.attr_)
        , hint_fwd_pd_(other.hint_fwd_pd_)
        , impl_list_(other.impl_list_)
        , skip_idx_(other.skip_idx_)
        , offset_(other.offset_)
        , src_engine_(other.src_engine_)
        , dst_engine_(other.dst_engine_)
        , has_unsupported_dt_(other.has_unsupported_dt_)
        , has_found_impl_(other.has_found_impl_) {}

    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_desc_iterator_t);
};

} // namespace impl
} // namespace dnnl

#endif
