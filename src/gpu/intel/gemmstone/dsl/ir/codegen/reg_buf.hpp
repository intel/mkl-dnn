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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_REG_BUF_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_REG_BUF_HPP

#include <vector>
#include <unordered_set>

#include "gemmstone/../../dsl/ir/codegen/register_allocator.hpp"
#include "gemmstone/../../dsl/ir/grf_permutation.hpp"
#include "gemmstone/../../dsl/utils/utils.hpp"
#include "gemmstone/dsl/ir.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Represents a register buffer allocated in blocks.
class reg_buf_t : public stringify_t<reg_buf_t> {
public:
    reg_buf_t() = default;

    reg_buf_t(ngen::HW hw, const ngen::GRFRange &range)
        : hw_(hw)
        , block_regs_(range.getLen())
        , block_bases_({range.getBase()}) {}

    reg_buf_t(ngen::HW hw, const ngen::GRF &grf)
        : hw_(hw), block_regs_(1), block_bases_({grf.getBase()}) {}

    reg_buf_t(ngen::HW hw, int block_regs, const std::vector<int> &block_bases)
        : hw_(hw), block_regs_(block_regs), block_bases_(block_bases) {}

    bool is_empty() const { return block_bases_.empty(); }

    ngen::HW hw() const { return hw_; }

    bool with_permute() const { return !grf_perm_.is_empty(); }

    int base(int reg_idx, bool apply_permute = true) const {
        if (apply_permute && !grf_perm_.is_empty())
            reg_idx = grf_perm_.map(reg_idx);
        dsl_assert(reg_idx >= 0 && reg_idx < regs())
                << "Invalid index: " << reg_idx;
        int block_idx = reg_idx / block_regs_;
        return block_bases_[block_idx] + (reg_idx % block_regs_);
    }

    int blocks() const { return int(block_bases_.size()); }

    int block_regs() const { return block_regs_; }

    int regs() const { return blocks() * block_regs(); }

    void set_grf_permutation(const grf_permutation_t &grf_perm) {
#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
        // Check that it's a valid permutation.
        std::unordered_set<int> seen;
        for (int i = 0; i < regs(); i++) {
            int i_mapped = grf_perm.map(i);
            dsl_assert(i_mapped >= 0 && i_mapped < regs());
            seen.insert(i_mapped);
        }
        dsl_assert(int(seen.size()) == regs()) << "Invalid permutation.";
#endif
        grf_perm_ = grf_perm;
    }

    bool operator==(const reg_buf_t &other) const {
        if (hw() != other.hw()) return false;
        if (block_regs() != other.block_regs()) return false;
        if (blocks() != other.blocks()) return false;
        for (int i = 0; i < blocks(); i++) {
            if (block_bases_[i] != other.block_bases_[i]) return false;
        }
        if (grf_perm_ != other.grf_perm_) return false;
        return true;
    }

    void claim(reg_allocator_t &ra) const {
        for (int i = 0; i < blocks(); i++) {
            ngen::GRFRange range(block_bases_[i], block_regs_);
            ra.claim(range);
        }
    }

    void release(reg_allocator_t &ra) const {
        for (int i = 0; i < blocks(); i++) {
            ngen::GRFRange range(block_bases_[i], block_regs_);
            ra.safeRelease(range);
        }
    }

    std::string str() const {
        if (is_empty()) return "(empty)";
        ostringstream_t oss;
        bool is_first = true;
        if (with_permute()) oss << "[permuted] ";
        for (int base : block_bases_) {
            if (!is_first) oss << "; ";
            oss << "r" << base;
            if (block_regs_ > 1) oss << " - r" << base + block_regs_ - 1;
            is_first = false;
        }
        return oss.str();
    }

private:
    ngen::HW hw_ = ngen::HW::Unknown;
    int block_regs_ = 0;
    std::vector<int> block_bases_;
    grf_permutation_t grf_perm_;
};

// ngen::RegData wrapper attached to a register buffer.
class reg_buf_data_t {
public:
    reg_buf_data_t() = default;

    reg_buf_data_t(const reg_buf_t &reg_buf)
        : reg_buf_(std::make_shared<reg_buf_t>(reg_buf))
        , rd_(ngen::GRF(reg_buf_->base(0))) {}

    reg_buf_data_t(const reg_buf_t &reg_buf, const ngen::RegData &rd)
        : reg_buf_(std::make_shared<reg_buf_t>(reg_buf)), rd_(rd) {}

    reg_buf_data_t(ngen::HW hw, const ngen::GRF &grf)
        : reg_buf_(std::make_shared<reg_buf_t>(hw, grf)), rd_(grf) {}

    reg_buf_data_t(ngen::HW hw, const ngen::Subregister &sub)
        : reg_buf_(std::make_shared<reg_buf_t>(
                  hw, ngen::GRFRange(sub.getBase(), 1)))
        , rd_(sub) {}

    const reg_buf_t &reg_buf() const { return *reg_buf_; }

    bool is_empty() const { return !reg_buf_; }

    bool with_permute() const { return reg_buf_->with_permute(); }

    ngen::HW hw() const { return reg_buf_->hw(); }

    ngen::DataType type() const { return rd_.getType(); }

    int base() const { return rd_.getBase(); }

    int byte_offset() const { return rd_.getByteOffset(); }
    int bit_offset() const { return offset() * rd_.getBits(); }

    int offset() const { return rd_.getOffset(); }

    int hs() const { return rd_.getHS(); }

    const ngen::RegData &reg_data() const { return rd_; }

    operator ngen::RegData() const { return rd_; }

    void set_grf_permutation(const grf_permutation_t &grf_perm) {
        reg_buf_->set_grf_permutation(grf_perm);
    }

    bool check_bounds(int off, int elems, ngen::DataType type,
            bool verify_access = true, bool check_dense = false) const {
        dsl_assert(off >= 0 && elems >= 0);
        if (elems == 0) return true;

        const int grf_bits = ngen::GRF::bytes(hw()) << 3;
        const int type_bits = ngen::getBits(type);
        int first_bit = bit_offset() + off * type_bits;
        int last_bit = first_bit + elems * type_bits - 1;
        int beg_off = first_bit / grf_bits;
        int end_off = last_bit / grf_bits;

        if (get_grf_buf_index() + end_off >= reg_buf_->regs()) {
            if (verify_access) {
                dsl_error() << "Invalid access";
                throw std::runtime_error("Internal error");
            }
            return false;
        }

        if (check_dense) {
            int base0 = get_grf_base(beg_off);
            for (int i = beg_off + 1; i <= end_off; ++i) {
                if (get_grf_base(i) != base0 + i) return false;
            }
        }
        return true;
    }

    bool is_dense(int bytes) const {
        return check_bounds(0, bytes, ngen::DataType::ub,
                /*verify_access=*/false, /*check_dense=*/true);
    }

    bool operator==(const reg_buf_data_t &other) const {
        return (*reg_buf_ == *other.reg_buf_) && (rd_ == other.rd_);
    }

    bool operator!=(const reg_buf_data_t &other) const {
        return !operator==(other);
    }

    // Retype register region while preserving data.
    reg_buf_data_t reinterpret(ngen::DataType new_type) const {
        int new_size = ngen::getBits(new_type);
        int old_size = ngen::getBits(type());
        if (new_size == old_size) {
            auto ret = *this;
            ret.rd_.setType(new_type);
            return ret;
        } else if (new_size < old_size) {
            dsl_assert(rd_.getHS() <= 1) << "Can't reinterpret strided data to "
                                            "differently sized data type.";
            return format(0, rd_.getWidth() * old_size / new_size, 1, new_type);
        } else {
            dsl_error() << "Can't reinterpret to larger data type.";
        }
        return reg_buf_data_t();
    }

    // Format register region to parameters regardless of data.
    reg_buf_data_t format(int offset, int width = 1, int hstride = 0,
            ngen::DataType type = ngen::DataType::invalid) const {
        if (type == ngen::DataType::invalid) type = rd_.getType();
        const auto grf_bits = ngen::GRF::bytes(hw()) << 3;
        const auto type_bits = ngen::getBits(type);

        auto off_bits = bit_offset() + offset * type_bits;
        auto new_base = off_bits / grf_bits;
        auto new_off = off_bits % grf_bits;

        dsl_assert(new_off % type_bits == 0);

        if (width == 1) {
            hstride = 0;
        } else if (hstride == 0) {
            dsl_assert(width == 1);
        } else {
            const int max_width = 32 * 8 / (hstride * type_bits);
            width = std::min({width, max_width, 16});
        }
        int vstride = width * hstride;

        int region = (width - 1) * hstride + 1;
        check_bounds(offset, region, type);

        auto ret = *this;
        auto grf = get_grf(new_base).retype(type);
        ret.rd_ = grf[new_off / type_bits](vstride, width, hstride);
        return ret;
    }

    reg_buf_data_t format(int offset, ngen::DataType type) const {
        return format(offset, 1, 1, type);
    }

    reg_buf_data_t format(ngen::DataType type) const { return format(0, type); }

    ngen::Subregister subregister(int offset, int width, int stride,
            ngen::DataType type = ngen::DataType::invalid) const {
        auto rd = format(offset * stride, width, stride, type).reg_data();
        return {rd, rd.getOffset(), rd.getType()};
    }

    ngen::Subregister subregister(
            int offset, ngen::DataType type = ngen::DataType::invalid) const {
        return subregister(offset, 1, 1, type);
    }

    ngen::Subregister subregister(ngen::DataType type) const {
        return subregister(0, type);
    }

    reg_buf_data_t unpermute() const {
        int idx = get_grf_buf_index();
        int base = reg_buf_->base(idx, /*apply_permute=*/false);

        auto ret = *this;
        ret.rd_.setBase(base);
        return ret;
    }

private:
    ngen::GRF get_grf(int off_regs) const {
        return ngen::GRF(get_grf_base(off_regs));
    }

    int get_grf_base(int off_regs) const {
        int idx = get_grf_buf_index();
        return reg_buf_->base(idx + off_regs);
    }

    int get_grf_buf_index() const {
        if (reg_buf_->blocks() == 1 && !reg_buf_->with_permute()) {
            return rd_.getBase() - reg_buf_->base(0);
        }
        for (int i = 0; i < reg_buf_->regs(); i++) {
            if (reg_buf_->base(i) == rd_.getBase()) return i;
        }
        stub();
        return -1;
    }

    std::shared_ptr<reg_buf_t> reg_buf_;
    ngen::RegData rd_;
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
