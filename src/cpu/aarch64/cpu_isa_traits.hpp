/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2023, 2025-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "dnnl_types.h"

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#if !defined(_WIN32)
#define XBYAK_USE_MMAP_ALLOCATOR
#endif

#include "xbyak_aarch64/xbyak_aarch64.h"
#include "xbyak_aarch64/xbyak_aarch64_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

/* The following enum is temporal implementation.
   It should be made in dnnl_types.h,
   but an RFC is requird to modify dnnl_types.h.
   The following values are used with
   static_cast<dnnl_cpu_isa_t>, the same values
   defined in dnnl_cpu_isa_t are temporaly used. */
/// CPU instruction set flags
enum {
    /// AARCH64 Advanced SIMD & floating-point
    dnnl_cpu_isa_asimd = 0x1,
    /// AARCH64 SVE 128 bits
    dnnl_cpu_isa_sve_128 = 0x3,
    /// AARCH64 SVE 256 bits
    dnnl_cpu_isa_sve_256 = 0x7,
    /// AARCH64 SVE 512 bits
    dnnl_cpu_isa_sve_512 = 0x27,
};

enum cpu_isa_bit_t : unsigned {
    asimd_bit = 1u << 0,
    sve_bit = 1u << 1,
    sve_128_bit = 1u << 2,
    sve_256_bit = 1u << 3,
    sve_512_bit = 1u << 4,
    sme_bit = 1u << 8,
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    asimd = asimd_bit,
    sve = sve_bit | asimd,
    sve_128 = sve_128_bit | sve,
    sve_256 = sve_256_bit | sve_128,
    sve_512 = sve_512_bit | sve_256,
    sme = sme_bit,
    isa_all = ~0u,
};

enum class cpu_isa_cmp_t {
    // List of infix comparison relations between two cpu_isa_t
    // where we take isa_1 and isa_2 to be two cpu_isa_t instances.

    // isa_1 SUBSET isa_2 if all feature flags supported by isa_1
    // are supported by isa_2 as well (equality allowed)
    SUBSET,

    // isa_1 SUPERSET isa_2 if all feature flags supported by isa_2
    // are supported by isa_1 as well (equality allowed)
    SUPERSET,

    // Few more options that (depending upon need) can be enabled in future

    // 1. PROPER_SUBSET: isa_1 SUBSET isa_2 and isa_1 != isa_2
    // 2. PROPER_SUPERSET: isa_1 SUPERSET isa_2 and isa_1 != isa_2
};

const char *get_isa_info();

cpu_isa_t get_max_cpu_isa();
cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

// Reduce any vector length specific cpu_isa_t to vector length agnostic (VLA) sve
constexpr cpu_isa_t to_vla_sve(const cpu_isa_t isa) {
    return (cpu_isa_t)(isa & sve);
}

static inline bool compare_isa(
        cpu_isa_t isa_1, cpu_isa_cmp_t cmp, cpu_isa_t isa_2) {
    // By default, comparison between ISA ignores ISA specific hints
    unsigned mask_1 = static_cast<unsigned>(isa_1);
    unsigned mask_2 = static_cast<unsigned>(isa_2);
    unsigned mask_common = mask_1 & mask_2;

    switch (cmp) {
        case cpu_isa_cmp_t::SUBSET: return mask_1 == mask_common;
        case cpu_isa_cmp_t::SUPERSET: return mask_2 == mask_common;
        default: assert(!"unsupported comparison of isa"); return false;
    }
}

static inline bool is_subset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUBSET, isa_2);
}

static inline bool is_superset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUPERSET, isa_2);
}

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for sve2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_default;
    static constexpr const char *user_option_env = "default";
};

template <>
struct cpu_isa_traits<asimd> {
    using TReg = Xbyak_aarch64::VReg;
    using TRegB = Xbyak_aarch64::VReg16B;
    using TRegH = Xbyak_aarch64::VReg8H;
    using TRegS = Xbyak_aarch64::VReg4S;
    using TRegD = Xbyak_aarch64::VReg2D;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_asimd);
    static constexpr const char *user_option_env = "advanced_simd";
};

template <>
struct cpu_isa_traits<sve> {
    using TReg = Xbyak_aarch64::ZReg;
    using TRegB = Xbyak_aarch64::ZRegB;
    using TRegH = Xbyak_aarch64::ZRegH;
    using TRegS = Xbyak_aarch64::ZRegS;
    using TRegD = Xbyak_aarch64::ZRegD;
    static constexpr int n_vregs = 32;
};

#define CPU_ISA_SVE(bits, shift) \
    template <> \
    struct cpu_isa_traits<sve_##bits> : public cpu_isa_traits<sve> { \
        static constexpr int vlen_shift = shift; \
        static constexpr int vlen = (bits) / 8; \
        static constexpr dnnl_cpu_isa_t user_option_val \
                = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_##bits); \
        static constexpr const char *user_option_env = "sve_ ## bits"; \
    };

CPU_ISA_SVE(128, 4)
CPU_ISA_SVE(256, 5)
CPU_ISA_SVE(512, 6)
#undef CPU_ISA_SVE

inline const Xbyak_aarch64::util::Cpu &cpu() {
    const static Xbyak_aarch64::util::Cpu cpu_;
    return cpu_;
}

namespace {

inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak_aarch64::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case asimd: return cpu().has(XBYAK_AARCH64_HWCAP_ADVSIMD);
        case sve: return cpu().has(XBYAK_AARCH64_HWCAP_SVE);
        case sve_128:
            return cpu().has(XBYAK_AARCH64_HWCAP_SVE)
                    && cpu().getSveLen() >= SVE_128;
        case sve_256:
            return cpu().has(XBYAK_AARCH64_HWCAP_SVE)
                    && cpu().getSveLen() >= SVE_256;
        case sve_512:
            return cpu().has(XBYAK_AARCH64_HWCAP_SVE)
                    && cpu().getSveLen() >= SVE_512;
        case sme: return cpu().has(XBYAK_AARCH64_HWCAP_SME);
        case isa_undef: return true;
        case isa_all: return false;
    }
    return false;
}

// SVE length in bytes
inline uint64_t get_sve_length() {
    return cpu().getSveLen();
}

// SME length in bytes
inline uint64_t get_sme_length() {
    return cpu().getSmeLen();
}

// SVE length in element type
inline uint64_t get_sve_length(data_type_t data_type) {
    const size_t dt_size = types::data_type_size(data_type);
    assert(dt_size > 0);
    return get_sve_length() / dt_size;
}

// SME length in element type
inline uint64_t get_sme_length(data_type_t data_type) {
    const size_t dt_size = types::data_type_size(data_type);
    assert(dt_size > 0);
    return get_sme_length() / dt_size;
}

inline int isa_max_vlen(cpu_isa_t isa) {
    if (isa == sve_512)
        return cpu_isa_traits<sve_512>::vlen;
    else if (isa == sve_256)
        return cpu_isa_traits<sve_256>::vlen;
    else if (isa == sve_128)
        return cpu_isa_traits<sve_128>::vlen;
    else if (isa == sve)
        return get_sve_length();
    else if (isa == sme)
        return get_sme_length();
    else if (isa == asimd)
        return cpu_isa_traits<asimd>::vlen;
    else
        return 0;
}

inline bool mayiuse_atomic() {
    using namespace Xbyak_aarch64::util;
    return cpu().isAtomicSupported();
}

inline bool isa_has_s8s8(cpu_isa_t isa) {
    return is_superset(isa, sve_128);
}

inline bool mayiuse_bf16() {
    using namespace Xbyak_aarch64::util;
    return cpu().isBf16Supported();
}

inline bool mayiuse_sve_i8mm() {
    using namespace Xbyak_aarch64::util;
    return mayiuse(sve) && cpu().has(XBYAK_AARCH64_HWCAP_SVEI8MM);
}

inline bool mayiuse_f16() {
    using namespace Xbyak_aarch64::util;
    return cpu().isF16Supported();
}

inline bool mayiuse_fhm() {
    using namespace Xbyak_aarch64::util;
    return cpu().isFhmSupported();
}

inline int isa_num_vregs(cpu_isa_t isa) {
    if (isa == sve_512)
        return cpu_isa_traits<sve_512>::n_vregs;
    else if (isa == sve_256)
        return cpu_isa_traits<sve_256>::n_vregs;
    else if (isa == sve_128)
        return cpu_isa_traits<sve_128>::n_vregs;
    else if (isa == sve)
        return cpu_isa_traits<sve>::n_vregs;
    else if (isa == asimd)
        return cpu_isa_traits<asimd>::n_vregs;
    else
        return 0;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_undef ? prefix STRINGIFY(any) : \
    ((isa) == asimd ? prefix STRINGIFY(asimd) : \
    ((isa) == sve ? prefix STRINGIFY(sve) : \
    ((isa) == sve_128 ? prefix STRINGIFY(sve_128) : \
    ((isa) == sve_256 ? prefix STRINGIFY(sve_256) : \
    ((isa) == sve_512 ? prefix STRINGIFY(sve_512) : \
    ((isa) == sme ? prefix STRINGIFY(sme) : \
    prefix suffix_if_any)))))))
/* clang-format on */

inline size_t data_type_vnni_granularity(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f32:
        case s32: return size_t(1);
        case f16:
        case bf16: return size_t(2);
        case s8:
        case u8: return size_t(4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return size_t(0); /* should not be reachable */
}

template <cpu_isa_t isa>
inline size_t data_type_vnni_simd_elems(data_type_t data_type) {
    const size_t dt_size = types::data_type_size(data_type);
    assert(dt_size > 0);
    return cpu_isa_traits<isa>::vlen / dt_size;
}

// Maximum number of elements of a given type in a SIMD (SVE/Neon) vector for a
// given ISA. Note that if cpu_isa_t is just sve (not sve_vl) then the value is
// determined at runtime (unlike the others, which can be determiend at compile time)
inline size_t simd_elems(data_type_t dt, cpu_isa_t cpu_isa) {
    switch (cpu_isa) {
        case sve_512: return data_type_vnni_simd_elems<sve_512>(dt);
        case sve_256: return data_type_vnni_simd_elems<sve_256>(dt);
        case sve_128:
        case asimd: return data_type_vnni_simd_elems<sve_128>(dt);
        case sve: return get_sve_length(dt);
        case sme: return get_sme_length(dt);
        default: {
            // If this ISA does implement SIMD, then you need to add support for
            // it in this function. If not, then you need to check earlier in
            // the implementation and don't call this function.
            assert(!"simd_elems does not support this cpu_isa_t.");
            return 0;
        }
    }
}

inline size_t simd_bytes(cpu_isa_t isa) {
    return simd_elems(data_type::s8, isa);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
