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

#ifndef GEMMSTONE_GENERATOR_PIECES_GRF_MULTIRANGE_HPP
#define GEMMSTONE_GENERATOR_PIECES_GRF_MULTIRANGE_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/type.hpp"

GEMMSTONE_NAMESPACE_START

// GRFMultirange represents a sequence of GRF registers, not necessarily contiguous.
// It is a generalization of nGEN's GRFRange, which represents a contiguous range of registers.
struct GRFMultirange {
    std::vector<ngen::GRFRange> ranges;

    GRFMultirange() {}
    GRFMultirange(ngen::GRFRange range) : ranges{1, range} {}

    ngen::GRF operator[](int idx) const { return lookup(idx); }

    ngen::GRF lookup(int idx, int *consecutive = nullptr) const {
        for (auto &r : ranges) {
            if (idx < r.getLen()) {
                if (consecutive) *consecutive = r.getLen() - idx;
                if (r.isInvalid()) return ngen::GRF();
                return ngen::GRF(r[idx].getBase());
            }
            idx -= r.getLen();
        }
        stub("Index out of bounds");
    }

    ngen::Subregister sub(int log2GRFBytes, int offset, ngen::DataType type, int *consecutive = nullptr) const {
        // u3 is packed (8 elements/3 bytes) with no valid getLog2Bits();
        // its `offset` is a byte offset (see RegisterBlock::find()), so use
        // byte granularity like ub instead of the formula below.
        const int lg2Len = (type == Type::ngen_u3())
                ? log2GRFBytes
                : log2GRFBytes + 3 - ngen::getLog2Bits(type);
        const int roffset = offset >> lg2Len;
        offset -= (roffset << lg2Len);
        int rconsecutive;
        auto reg = lookup(roffset, &rconsecutive);
        if (consecutive)
            *consecutive = (rconsecutive << lg2Len) - offset;
        return reg.sub(offset, type);
    }

    ngen::Subregister sub(ngen::HW hw, int offset, ngen::DataType type, int *consecutive = nullptr) const {
        return sub(ngen::GRF::log2Bytes(hw), offset, type, consecutive);
    }

    GRFMultirange subrange(int start, int count) const {
        GRFMultirange result;
        for (auto &r : ranges) {
            if (start < r.getLen()) {
                auto got = std::min(count, r.getLen() - start);
                result.ranges.push_back(ngen::GRFRange{r.getBase() + start, got});
                count -= got;
                start = 0;
                if (count <= 0) break;
            } else
                start -= r.getLen();
        }
        return result;
    }

    bool contiguous(int start, int count) const {
        for (auto &r : ranges) {
            if (start < r.getLen())
                return (start + count) <= r.getLen();
            start -= r.getLen();
        }
        return false;
    }

    void append(ngen::GRF r) {
        append(r-r);
    }

    void append(ngen::GRFRange r) {
        if (!ranges.empty()) {
            auto &rend = ranges.back();
            if (rend.getBase() + rend.getLen() == r.getBase()) {
                rend = ngen::GRFRange(rend.getBase(), rend.getLen() + r.getLen());
                return;
            }
        }
        ranges.push_back(r);
    }

    void append(const GRFMultirange &r) {
        for (auto &rr : r.ranges)
            append(rr);
    }

    int getLen() const {
        int len = 0;
        for (auto &r : ranges) len += r.getLen();
        return len;
    }

    bool empty() const {
        for (auto &r: ranges)
            if (r.getLen() > 0)
                return false;
        return true;
    }
    void clear()       { ranges.clear(); }
};

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
