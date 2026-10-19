/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_BANK_CONFLICT_ALLOCATION_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_BANK_CONFLICT_ALLOCATION_HPP

#include "dsl/ir/codegen/reg_buf.hpp"
#include "dsl/ir/codegen/register_allocator.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Stores a set of register buffers that are allocated based on their usages
// to avoid bank and bundle conflicts.
class bank_conflict_allocation_t {
public:
    bank_conflict_allocation_t() = default;
    bank_conflict_allocation_t(reg_allocator_t &ra) : refs_(1), ra_(&ra) {}

    bool is_empty() const { return !ra_; }

    int refs() const { return refs_; }

    void retain() {
        dsl_assert(refs_ > 0);
        refs_++;
    }

    void release(const expr_t &buf) {
        dsl_assert(refs_ > 0);
        refs_--;
        auto it = buf_map_.find(buf);
        dsl_assert(it != buf_map_.end()) << "Buffer not found: " << buf;
        it->second.release(*ra_);
        buf_map_.erase(it);
    }

    const reg_buf_t &get_reg_buf(const expr_t &buf) const {
        return buf_map_.at(buf);
    }

    void set_reg_buf(const expr_t &buf, const reg_buf_t &reg_buf) {
        auto ret = buf_map_.emplace(buf, reg_buf);
        reg_buf.claim(*ra_);
        dsl_assert(ret.second) << "Buffer already exists: " << buf;
    }

    static bank_conflict_allocation_t create(
            reg_allocator_t &ra, const bank_conflict_attr_t &_attr);

private:
    int refs_ = 0;
    reg_allocator_t *ra_ = nullptr;
    object_map_t<expr_t, reg_buf_t> buf_map_;
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
