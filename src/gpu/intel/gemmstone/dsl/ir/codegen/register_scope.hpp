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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_REGISTER_SCOPE_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_REGISTER_SCOPE_HPP

#include "gemmstone/../../dsl/ir/codegen/ngen_helpers.hpp"
#include "gemmstone/../../dsl/ir/codegen/reg_buf.hpp"
#include "ngen.hpp"
#include "ngen_register_allocator.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Maintains scoped allocations which are automatically released when the scope
// is destructed.
class ngen_register_scope_t {
public:
    ngen_register_scope_t(reg_allocator_t &ra) : ra_(ra) {}

    ngen_register_scope_t(const ngen_register_scope_t &) = delete;

    ngen_register_scope_t(ngen_register_scope_t &&other)
        : ra_(other.ra_)
        , grf_ranges_(std::move(other.grf_ranges_))
        , subregisters_(std::move(other.subregisters_)) {}

    reg_allocator_t &register_allocator() { return ra_; }

    ngen::HW hw() const { return ra_.hardware(); }

    ~ngen_register_scope_t() { clear(); }

    void clear() {
        for (auto &r : grf_ranges_)
            ra_.safeRelease(r);
        for (auto &s : subregisters_)
            ra_.safeRelease(s);
        for (auto &f : flags_)
            ra_.safeRelease(f);
        grf_ranges_.clear();
        subregisters_.clear();
        flags_.clear();
    }

    ngen::GRFRange find_grf_range(int base, int byte_offset) const {
        if (byte_offset != 0) return ngen::GRFRange();
        for (auto &r : grf_ranges_)
            if (r.getBase() == base) return r;
        return ngen::GRFRange();
    }

    ngen::Subregister find_sub(int base, int byte_offset) const {
        for (auto &s : subregisters_)
            if (s.getBase() == base && s.getByteOffset() == byte_offset)
                return s;
        return ngen::Subregister();
    }

    ngen::GRFRange try_alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_.try_alloc_range(regs, base_bundle);
        if (!ret.isInvalid()) grf_ranges_.push_back(ret);
        return ret;
    }

    ngen::GRFRange alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_.alloc_range(regs, base_bundle);
        grf_ranges_.push_back(ret);
        return ret;
    }

    reg_buf_t alloc_reg_buf(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto range = ra_.alloc_range(regs, base_bundle);
        grf_ranges_.push_back(range);
        return reg_buf_t(ra_.hardware(), range);
    }

    reg_buf_data_t alloc_reg_buf_data(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        return alloc_reg_buf(regs, base_bundle);
    }

    reg_buf_data_t alloc_reg_data(const type_t &type, int stride = 1,
            ngen::Bundle bundle = ngen::Bundle()) {
        if (type.is_scalar()) {
            auto sub = alloc_sub(to_ngen(type), bundle);
            return reg_buf_data_t(hw(), sub);
        }

        int grf_size = ngen::GRF::bytes(hw());
        int regs = div_up(
                type.with_elems(type.elems() * stride).size(), grf_size);
        auto buf = alloc_reg_buf(regs, bundle);
        reg_buf_data_t rbd(buf);
        return rbd.format(0, type.elems(), stride, to_ngen(type.scalar()));
    }

    ngen::GRF alloc(ngen::Bundle bundle = ngen::Bundle()) {
        auto range = ra_.alloc_range(1, bundle);
        grf_ranges_.push_back(range);
        return range[0];
    }

    ngen::Subregister alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra_.alloc_sub(type, bundle);
        subregisters_.push_back(ret);
        return ret;
    }

    ngen::FlagRegister try_alloc_flag(int elems) {
        auto ret = ra_.try_alloc_flag(/*sub=*/elems <= 16);
        if (!ret.isInvalid()) flags_.push_back(ret);
        return ret;
    }

    ngen::FlagRegister alloc_flag(int elems) {
        auto ret = ra_.alloc_flag(/*sub=*/elems <= 16);
        flags_.push_back(ret);
        return ret;
    }

    void claim(const ngen::GRFRange &range) {
        ra_.claim(range);
        grf_ranges_.push_back(range);
    }

    void claim(const ngen::Subregister &sub) {
        ra_.claim(sub);
        subregisters_.push_back(sub);
    }

private:
    reg_allocator_t &ra_;

    std::vector<ngen::GRFRange> grf_ranges_;
    std::vector<ngen::Subregister> subregisters_;
    std::vector<ngen::FlagRegister> flags_;
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
