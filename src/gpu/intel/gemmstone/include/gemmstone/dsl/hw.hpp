/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_HW_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_HW_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

// NOLINTBEGIN(readability-identifier-naming)
namespace ngen {
enum class Core;
using HW = Core;
enum class ProductFamily : int;
struct Product;
} // namespace ngen
// NOLINTEND(readability-identifier-naming)

GEMMSTONE_NAMESPACE_START
namespace dsl {

namespace hw {
enum class attr_t {
    none = 0,
    large_grf = 1,
    systolic = 2,
    atomic_fp64 = 4,
    efficient_64bit = 8
};
constexpr attr_t operator&(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr attr_t operator|(attr_t a, attr_t b) {
    return static_cast<attr_t>(
            static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr attr_t operator~(attr_t a) {
    return static_cast<attr_t>(~static_cast<uint32_t>(a));
}
constexpr bool any(attr_t a) {
    return a != static_cast<attr_t>(0);
}

inline attr_t &operator|=(attr_t &a, attr_t b) {
    return a = a | b;
}
inline attr_t &operator&=(attr_t &a, attr_t b) {
    return a = a & b;
}

} // namespace hw

// Provides access to HW configuration which includes non-configurable
// properties.
class hw_t : public stringify_t<hw_t> {
public:
    using attr_t = hw::attr_t;
    hw_t() = default;
    explicit hw_t(const ngen::Product &product, int eu_count, int max_wg_size,
            size_t l3_cache_size, attr_t attr);
    explicit hw_t(const ngen::ProductFamily &family);
    hw_t(const hw_t &);
    hw_t operator=(const hw_t &other) {
        hw_t tmp(other);
        std::swap(product_, tmp.product_);
        std::swap(hw_, tmp.hw_);
        std::swap(eu_count_, tmp.eu_count_);
        std::swap(max_wg_size_, tmp.max_wg_size_);
        std::swap(l3_cache_size_, tmp.l3_cache_size_);
        std::swap(attr_, tmp.attr_);
        return *this;
    }

    const ngen::Product &product() const;
    ngen::ProductFamily family() const;
    int stepping() const;
    ngen::HW ngen_hw() const { return hw_; }
    operator ngen::HW() const { return hw_; }
    attr_t attr() const { return attr_; }

    bool has_fp64_atomic_support() const {
        return any(attr_ & attr_t::atomic_fp64);
    }
    int eu_count() const { return eu_count_; }
    int large_grf_support() const { return any(attr_ & attr_t::large_grf); }
    int grf_size() const;
    int systolic_support() const { return any(attr_ & attr_t::systolic); }
    size_t l3_cache_size() const { return l3_cache_size_; }
    bool efficient_64_bit() const {
        return any(attr_ & attr_t::efficient_64bit);
    }

    size_t max_tg_size(int regs, int simd) const;

    // Number of EUs per Xe core (maps to dual subslice on XeHPG).
    int eus_per_core() const;
    int threads_per_eu(int regs = 128) const;
    int cache_line_size() const;

    std::string str() const;

    bool operator<(ngen::HW rhs) const { return hw_ < rhs; }
    bool operator>(ngen::HW rhs) const { return hw_ > rhs; }
    bool operator<=(ngen::HW rhs) const { return hw_ <= rhs; }
    bool operator>=(ngen::HW rhs) const { return hw_ >= rhs; }
    bool operator==(ngen::HW rhs) const { return hw_ == rhs; }
    bool operator!=(ngen::HW rhs) const { return hw_ != rhs; }
    bool operator==(const hw_t &other) const;

protected:
    // use product_t as an opaque handle to ngen::Product to avoid ngen dependency here
    using product_t = const ngen::Product *;
    std::unique_ptr<ngen::Product> product_;

private:
    size_t max_wg_size(int regs = 128) const {
        // max_wg_size_ implicitly assumes 128 GRF/thread - other GRF modes will vary
        size_t thread_per_eu = threads_per_eu(regs);
        size_t base_thread_per_eu = threads_per_eu(128);
        return max_wg_size_ * thread_per_eu / base_thread_per_eu;
    }

    ngen::HW hw_ = {};
    int eu_count_ = 0;
    size_t max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;
    attr_t attr_ = attr_t::none;
};

} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
