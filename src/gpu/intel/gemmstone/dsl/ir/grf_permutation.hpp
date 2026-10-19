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

#ifndef GEMMSTONE_DSL_IR_GRF_PERMUTATION_HPP
#define GEMMSTONE_DSL_IR_GRF_PERMUTATION_HPP

#include <array>

#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/../../dsl/utils/utils.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Helper class to permute registers. Used to permute registers after applying
// dpas -> dpasw transformation.
class grf_permutation_t {
public:
    grf_permutation_t() { permutation_.fill(-1); }

    int map(int off) const {
        dsl_assert(off >= 0 && off < max_regs);
        if (permutation_[off] == -1) return off;
        return permutation_[off];
    }

    bool is_empty() const { return is_empty_; }

    void set_permute(int old_off, int new_off) {
        dsl_assert(old_off >= 0 && old_off < max_regs);
        if (old_off == new_off || new_off == -1) return;
        is_empty_ = false;
        dsl_assert(one_of(permutation_[old_off], {-1, new_off}))
                << "Already assigned to a different offset.";
        permutation_[old_off] = new_off;
    }

    bool operator==(const grf_permutation_t &other) const {
        for (int i = 0; i < max_regs; i++) {
            if (permutation_[i] != other.permutation_[i]) return false;
        }
        return true;
    }

    bool operator!=(const grf_permutation_t &other) const {
        return !operator==(other);
    }

private:
    static const int max_regs = 256;

    std::array<int, max_regs> permutation_ = {};
    bool is_empty_ = true;
};

// Allocation attribute specifying permutation for a GRF buffer.
class grf_permute_attr_t : public alloc_attr_impl_t,
                           public object::info_t<grf_permute_attr_t> {
public:
    static alloc_attr_t make(
            const std::shared_ptr<grf_permutation_t> &grf_perm) {
        return alloc_attr_t(new grf_permute_attr_t(grf_perm));
    }

    bool is_equal(const object::impl_t &obj) const override {
        return this == &obj;
    }

    size_t compute_hash() const override { return 0; }

    std::shared_ptr<grf_permutation_t> grf_perm;

private:
    grf_permute_attr_t(const std::shared_ptr<grf_permutation_t> &grf_perm)
        : alloc_attr_impl_t(get_info()), grf_perm(grf_perm) {}
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
