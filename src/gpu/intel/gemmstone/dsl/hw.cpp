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

#include "gemmstone/dsl/hw.hpp"
#include "dsl/utils/utils.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

hw_t::hw_t(const ngen::Product &product, int eu_count, int max_wg_size,
        size_t l3_cache_size, attr_t attr)
    : product_(make_unique<ngen::Product>(product))
    , hw_(ngen::getCore(product.family))
    , eu_count_(eu_count)
    , max_wg_size_(max_wg_size)
    , l3_cache_size_(l3_cache_size)
    , attr_(attr) {}

hw_t::hw_t(const hw_t &other)
    : product_(other.product_ ? make_unique<ngen::Product>(*other.product_)
                              : nullptr)
    , hw_(other.hw_)
    , eu_count_(other.eu_count_)
    , max_wg_size_(other.max_wg_size_)
    , l3_cache_size_(other.l3_cache_size_)
    , attr_(other.attr_) {}

static hw_t get_default_hardware(ngen::ProductFamily family) {
    using namespace ngen;
    // TODO: stepping should be set to stepping of released hardware
    ngen::Product product {family, 0, getPlatformType(family)};
    // EU count and L3 cache size are unused for kernel generation; set to 0.
    switch (family) {
        case ProductFamily::GenericXeLP:
            return hw_t(product, 0, 512, 0, hw::attr_t::atomic_fp64);
        case ProductFamily::GenericXeHPG:
        case ProductFamily::DG2:
            return hw_t(product, 0, 1024, 0,
                    hw::attr_t::systolic | hw::attr_t::atomic_fp64);
        case ProductFamily::MTL:
        case ProductFamily::ARL:
        case ProductFamily::GenericXeHP:
            return hw_t(product, 0, 1024, 0, hw::attr_t::systolic);
        case ProductFamily::GenericXeHPC:
        case ProductFamily::PVC:
            return hw_t(product, 0, 1024, 0,
                    hw::attr_t::large_grf | hw::attr_t::systolic
                            | hw::attr_t::atomic_fp64);
        case ProductFamily::PVCVG:
            return hw_t(product, 0, 1024, 0,
                    hw::attr_t::large_grf | hw::attr_t::atomic_fp64);
        case ProductFamily::GenericXe2:
        case ProductFamily::BMG:
        case ProductFamily::LNL:
        case ProductFamily::GenericXe3:
        case ProductFamily::GenericXe3p:
        case ProductFamily::NVLP:
        case ProductFamily::CRI:
            return hw_t(product, 0, 1024, 0,
                    hw::attr_t::large_grf | hw::attr_t::systolic
                            | hw::attr_t::atomic_fp64);
        default: break;
    }
    throw std::runtime_error("Unknown GPU product.");
}

hw_t::hw_t(const ngen::ProductFamily &family)
    : hw_t(get_default_hardware(family)) {}

bool hw_t::operator==(const hw_t &other) const {
    return hw_ == other.hw_ && eu_count_ == other.eu_count_
            && max_wg_size_ == other.max_wg_size_
            && l3_cache_size_ == other.l3_cache_size_ && attr_ == other.attr_
            && (product_ == other.product_
                    || (product_ && other.product_
                            && *product_ == *other.product_));
}

const ngen::Product &hw_t::product() const {
    if (!product_)
        throw std::runtime_error("Product information not available.");
    return *product_;
}

ngen::ProductFamily hw_t::family() const {
    return product().family;
}

int hw_t::stepping() const {
    return product().stepping;
}

int hw_t::grf_size() const {
    return ngen::GRF::bytes(hw_);
}

size_t hw_t::max_tg_size(int regs, int simd) const {
    size_t wg_size = max_wg_size(regs);
    size_t eu_based_tg_size
            = eus_per_core() * rounddown_pow2(threads_per_eu(regs));
    size_t wg_based_tg_size = wg_size / simd;
    return std::min(eu_based_tg_size, wg_based_tg_size);
}
int hw_t::eus_per_core() const {
    switch (hw_) {
        case ngen::HW::XeHPG: return 16;
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3:
        case ngen::HW::Xe3p: return 8;
        default: stub(); return 8;
    }
}

namespace {
int grf_per_eu(const ngen::Product &product) {
    ngen::HW hw = ngen::getCore(product.family);
    switch (hw) {
        case ngen::HW::XeLP: return 896;
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3:
        case ngen::HW::Xe3p:
            return product.family == ngen::ProductFamily::CRI ? 2048 : 1024;
        default: stub(); return 1024;
    }
}

int max_threads_per_eu(const ngen::Product product) {
    auto family = product.family;
    switch (getCore(family)) {
        case ngen::HW::XeLP: return 7;
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return 8;
        case ngen::HW::Xe3p:
            return family == ngen::ProductFamily::NVLP ? 10 : 8;
        default: stub();
    }
    return 8;
}
} // namespace

int hw_t::threads_per_eu(int regs) const {
    // CRI has an exception: 256 GRF/thread is restricted to 4 threads/EU due to lack of accumulators
    if (product().family == ngen::ProductFamily::CRI && regs == 256) return 4;

    int max_threads = max_threads_per_eu(product());
    return std::min(max_threads, grf_per_eu(product()) / regs);
}

int hw_t::cache_line_size() const {
    switch (hw_) {
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3:
        case ngen::HW::Xe3p: return 64;
        default: stub();
    }
    return 0;
}

std::string hw_t::str() const {
    ostringstream_t oss;
    oss << to_string(hw_);
    oss << ", stepping: " << stepping();
    oss << ", EUs: " << eu_count();
    return oss.str();
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
