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

#ifndef CPU_X64_CPU_ISA_TRAITS_HPP
#define CPU_X64_CPU_ISA_TRAITS_HPP

#include <functional>
#include <type_traits>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#if !defined(XBYAK64)
#define XBYAK64
#endif

#if !defined(XBYAK_NO_OP_NAMES)
#define XBYAK_NO_OP_NAMES
#endif

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#if !defined(XBYAK_USE_MMAP_ALLOCATOR)
#define XBYAK_USE_MMAP_ALLOCATOR
#endif

#if defined(NDEBUG) && !defined(XBYAK_NO_EXCEPTION)
#define XBYAK_NO_EXCEPTION
#endif

// Several JIT kernels intentionally or unintentionally use size-mismatched memory operands
// (e.g. zword[]-annotated addresses with 64-bit GPR instructions, or dword[] fields loaded into
// Reg64 registers). These are architecturally correct on x86-64 but trigger this check. Until all
// call sites are updated to use properly-sized operands or stripped annotations, set this to 0 to
// suppress the false-positive errors.
#if !defined(XBYAK_STRICT_CHECK_MEM_REG_SIZE)
#define XBYAK_STRICT_CHECK_MEM_REG_SIZE 0
#endif

#include "common/compiler_workarounds.hpp"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Maximum number of features + hints that can be specified via bits
static constexpr int cpu_isa_total_bits = sizeof(unsigned) * 8;

enum cpu_isa_bit_t : unsigned {
    // Fill in features from least significant bit to most significant bit
    avx10_version_bit_start = 0,
    avx10_version_bit_end = 3,
    xmm_bit = 1u << 4,
    ymm_bit = 1u << 5,
    zmm_bit = 1u << 6,
    amx_tile_bit = 1u << 7,

    // re-using xmm, ymm and zmm bits for avx2 and avx512.
    evex_core_bit = 1u << 8,
    avx2_bit = 1u << 9,
    vex_vnni_bit = 1u << 10,
    vex_vnni_2_bit = 1u << 11,
    evex_core_vnni_bit = 1u << 12,
    evex_core_bf16_bit = 1u << 13,
    evex_core_fp16_bit = 1u << 14,
    amx_int8_bit = 1u << 15,
    amx_bf16_bit = 1u << 16,
    amx_fp16_bit = 1u << 17,
    amx_2_bit = 1u << 18,
    ace_bit = 1u << 19,
    avx10_2_aux_bit = 1u << 20,

    // Fill in hints from most significant bit to least significant bit
    prefer_ymm_bit = 1u << (cpu_isa_total_bits - 1),

    avx10_version_bits
    = ((1 << (avx10_version_bit_end - avx10_version_bit_start + 1)) - 1)
            << avx10_version_bit_start,
    avx10_bit = avx2_bit | vex_vnni_bit | evex_core_bit | evex_core_vnni_bit
            | evex_core_bf16_bit | evex_core_fp16_bit,
    avx10_1_bit = (1 << avx10_version_bit_start) | avx10_bit,
    avx10_2_bit = (2 << avx10_version_bit_start) | avx10_bit | vex_vnni_2_bit,
};

dnnl_cpu_isa_hints_t DNNL_API get_cpu_isa_hints(bool soft = false);
status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints);

namespace cpu_isa_hints_utils {
/* hints_1 | hints_2 | ... | hints_n where hints_i are hint specific
   bits declared inside the cpu_isa_bit_t */
static constexpr unsigned hints_mask = prefer_ymm_bit;

inline unsigned cvt2mask(dnnl_cpu_isa_hints_t hints) {
    switch (hints) {
        case dnnl_cpu_isa_no_hints: return 0;
        case dnnl_cpu_isa_prefer_ymm: return prefer_ymm_bit;
    }
    assert(!"Unexpected CPU ISA hint");
    return 0;
}

inline bool is_hints_bit_set(cpu_isa_bit_t hint_bit, bool soft) {
    const dnnl_cpu_isa_hints_t hints = get_cpu_isa_hints(soft);
    const unsigned cur_hints_mask = cpu_isa_hints_utils::cvt2mask(hints);
    return (cur_hints_mask & hint_bit) == hint_bit;
}
} // namespace cpu_isa_hints_utils

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    avx2 = avx2_bit | ymm_bit | xmm_bit,
    avx2_vnni = vex_vnni_bit | avx2,
    avx2_vnni_2 = avx2_vnni | vex_vnni_2_bit,
    avx512_core = evex_core_bit | zmm_bit | avx2,
    avx512_core_vnni = evex_core_vnni_bit | avx512_core,
    avx512_core_bf16 = evex_core_bf16_bit | avx512_core_vnni,
    avx512_core_bf16_ymm = prefer_ymm_bit | avx512_core_bf16,
    amx_tile = amx_tile_bit,
    amx_int8 = amx_int8_bit | amx_tile,
    amx_bf16 = amx_bf16_bit | amx_tile,
    amx_fp16 = amx_fp16_bit | amx_tile,
    avx10_1_512 = avx10_1_bit | zmm_bit | ymm_bit | xmm_bit,
    avx512_core_fp16 = avx10_1_512,
    avx10_1_512_amx = avx10_1_512 | amx_int8 | amx_bf16,
    avx512_core_amx = avx10_1_512_amx,
    avx10_1_512_amx_fp16 = avx10_1_512_amx | amx_fp16,
    avx512_core_amx_fp16 = avx10_1_512_amx_fp16,
    avx10_2 = avx10_2_bit | xmm_bit | ymm_bit | zmm_bit,
    avx10_2_512 = avx10_2,
    avx10_2_aux = avx10_2 | avx10_2_aux_bit,
    avx10_2_amx_2
    = avx10_2 | amx_tile | amx_int8 | amx_bf16 | amx_fp16 | amx_2_bit,
    avx10_2_512_amx_2 = avx10_2_amx_2,
    avx10_2_ace = avx10_2_aux | amx_tile | ace_bit,
    // NOTES: 1. isa_all by default has no isa specific hints
    //        2. avx10_2_ace is under preview support and turned off by
    //           default. It is enabled only when the max CPU ISA is explicitly
    //           set to avx10_2_ace.
    //        3. avx10_2_aux is masked out for the same reason: AVX10.2-aux is
    //           only enumerated on parts that also expose ACE, so it shares
    //           the preview gate. It is re-enabled implicitly when the max CPU
    //           ISA is set to avx10_2_ace, which contains avx10_2_aux_bit.
    isa_all
    = ~0u & ~ace_bit & ~avx10_2_aux_bit & ~cpu_isa_hints_utils::hints_mask,
};

std::string isa2str(cpu_isa_t isa);

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

static inline uint32_t get_avx10_version(cpu_isa_t isa) {
    return (static_cast<uint32_t>(isa)
                   & static_cast<uint32_t>(avx10_version_bits))
            >> avx10_version_bit_start;
}

static inline bool compare_isa(
        cpu_isa_t isa_1, cpu_isa_cmp_t cmp, cpu_isa_t isa_2) {
    assert(isa_1 != isa_all);
    assert(isa_2 != isa_all);
    // Comparison with `isa_all` is illegal.
    if (utils::one_of(isa_all, isa_1, isa_2)) return false;

    const unsigned isa1_avx10_v = get_avx10_version(isa_1);
    const unsigned isa2_avx10_v = get_avx10_version(isa_2);
    // By default, comparison between ISA ignores ISA specific hints
    const unsigned isa1_non_converged_avx10
            = isa_1 & ~cpu_isa_hints_utils::hints_mask & ~avx10_version_bits;
    const unsigned isa2_non_converged_avx10
            = isa_2 & ~cpu_isa_hints_utils::hints_mask & ~avx10_version_bits;
    const unsigned common_non_converged_avx10
            = isa1_non_converged_avx10 & isa2_non_converged_avx10;
    switch (cmp) {
        case cpu_isa_cmp_t::SUBSET:
            return isa1_avx10_v <= isa2_avx10_v
                    && (isa1_non_converged_avx10 == common_non_converged_avx10);
        case cpu_isa_cmp_t::SUPERSET:
            return isa1_avx10_v >= isa2_avx10_v
                    && (isa2_non_converged_avx10 == common_non_converged_avx10);
        default: assert(!"unsupported comparison of isa"); return false;
    }
}

static inline bool is_subset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUBSET, isa_2);
}

static inline bool is_superset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUPERSET, isa_2);
}

template <typename Vmm>
struct vreg_traits_t {};

template <>
struct vreg_traits_t<Xbyak::Zmm> {
    using Vmm_lower_t = Xbyak::Ymm;
    static constexpr size_t vlen = 64;
};

template <>
struct vreg_traits_t<Xbyak::Ymm> {
    using Vmm_lower_t = Xbyak::Xmm;
    static constexpr size_t vlen = 32;
};

template <>
struct vreg_traits_t<Xbyak::Xmm> {
    using Vmm_lower_t = Xbyak::Xmm;
    static constexpr size_t vlen = 16;
};

template <cpu_isa_t>
struct cpu_isa_traits_t {}; /* ::vlen -> 32 (for avx2) */

// pack struct so it can fit into a single 64-byte cache line
#pragma pack(push, 1)
struct palette_config_t {
    static constexpr int max_size = 16;
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[max_size];
    uint8_t rows[max_size];
};
#pragma pack(pop)

// AMX/ACE tile palette identifiers.
// Palette 0 selects the INIT state, palette 1 the AMX TMUL tile configuration
// and palette 2 the ACE configuration with fixed 16x64B tiles.
enum : uint8_t {
    amx_palette_init = 0,
    amx_palette_tmul = 1,
    amx_palette_ace = 2,
};

template <>
struct cpu_isa_traits_t<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_default;
    static constexpr const char *user_option_env = "default";
};

template <>
struct cpu_isa_traits_t<avx2> {
    using Vmm = Xbyak::Ymm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = vreg_traits_t<Vmm>::vlen;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "avx2";
};

template <>
struct cpu_isa_traits_t<avx2_vnni> : public cpu_isa_traits_t<avx2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2_vnni;
    static constexpr const char *user_option_env = "avx2_vnni";
};

template <>
struct cpu_isa_traits_t<avx2_vnni_2> : public cpu_isa_traits_t<avx2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2_vnni_2;
    static constexpr const char *user_option_env = "avx2_vnni_2";
};

template <>
struct cpu_isa_traits_t<avx512_core> {
    using Vmm = Xbyak::Zmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = vreg_traits_t<Vmm>::vlen;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "avx512_core";
};

template <>
struct cpu_isa_traits_t<avx512_core_vnni>
    : public cpu_isa_traits_t<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "avx512_core_vnni";
};

template <>
struct cpu_isa_traits_t<avx512_core_bf16>
    : public cpu_isa_traits_t<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "avx512_core_bf16";
};

template <>
struct cpu_isa_traits_t<avx10_1_512_amx> {
    using Vmm = Xbyak::Zmm;
    static constexpr int vlen = vreg_traits_t<Vmm>::vlen;
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx10_1_512_amx;
    static constexpr const char *user_option_env = "avx10_1_512_amx";
};

template <>
struct cpu_isa_traits_t<avx10_1_512> : public cpu_isa_traits_t<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx10_1_512;
    static constexpr const char *user_option_env = "avx10_1_512";
};

template <>
struct cpu_isa_traits_t<avx10_1_512_amx_fp16> {
    using Vmm = Xbyak::Zmm;
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx10_1_512_amx_fp16;
    static constexpr const char *user_option_env = "avx10_1_512_amx_fp16";
};

template <>
struct cpu_isa_traits_t<avx10_2> : public cpu_isa_traits_t<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx10_2;
    static constexpr const char *user_option_env = "avx10_2";
};

template <>
struct cpu_isa_traits_t<avx10_2_amx_2> : public cpu_isa_traits_t<avx10_2> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx10_2_amx_2;
    static constexpr const char *user_option_env = "avx10_2_amx_2";
};

template <>
struct cpu_isa_traits_t<avx10_2_ace> : public cpu_isa_traits_t<avx10_2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx10_2_ace;
    static constexpr const char *user_option_env = "avx10_2_ace";
};

inline const Xbyak::util::Cpu &cpu() {
    const static Xbyak::util::Cpu cpu_;
    return cpu_;
}

namespace apx {

inline bool is_available() {
    const bool xsave_supported = cpu().has(Xbyak::util::Cpu::tOSXSAVE);
    if (!xsave_supported) return false;

    const uint64_t xcr0_features = Xbyak::util::Cpu::getXfeature();
    return ((xcr0_features >> 19) & 1) == 1;
}

} // namespace apx

namespace amx {

// Return the target palette for AMX instructions. Currently this is `0` if AMX
// instructions are not supported, and `1` if they are.
int get_target_palette();

int get_max_tiles(int palette);
int get_max_column_bytes(int palette);
int get_max_rows(int palette);
bool DNNL_API is_available();

// Helper function to get the maximum safe palette size for bounds checking
inline int get_max_palette_size() {
    const int max_tiles = get_max_tiles(get_target_palette());
    constexpr int max_palette_size = palette_config_t::max_size;
    return nstl::min(max_tiles, max_palette_size);
}

} // namespace amx

namespace ace {

// ACE adds the SCALEDATA (Block Scale Register) XSAVE state component on top
// of the AMX TILECFG/TILEDATA components. ACE instructions require
// XCR0[20,18:17] == 0b111, so AMX state alone is not sufficient.
bool DNNL_API is_available();

} // namespace ace

namespace {

inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak::util;

    unsigned cpu_isa_mask = x64::get_max_cpu_isa_mask(soft);
    unsigned cpu_isa_no_hints
            = cpu_isa & ~cpu_isa_hints_utils::hints_mask & ~avx10_version_bits;
    const auto cpu_avx10_version
            = get_avx10_version(static_cast<cpu_isa_t>(cpu_isa_mask));
    const auto isa_avx10_version = get_avx10_version(cpu_isa);

    if ((cpu_isa_mask & cpu_isa_no_hints) != cpu_isa_no_hints) return false;
    // AVX10 version N+1 will include all the features and capabilities
    // included in version N, and not vice versa.
    if (cpu_avx10_version < isa_avx10_version) return false;

    switch (cpu_isa) {
        case avx2: REG_AVX2_ISA(return cpu().has(Cpu::tAVX2));
        case avx2_vnni:
            REG_AVX2_ISA(
                    return mayiuse(avx2, soft) && cpu().has(Cpu::tAVX_VNNI));
        case avx2_vnni_2:
            REG_AVX2_ISA(return mayiuse(avx2_vnni, soft)
                    && cpu().has(Cpu::tAVX_VNNI_INT8)
                    && cpu().has(Cpu::tAVX_NE_CONVERT));
        case avx512_core:
            REG_AVX512_ISA(return cpu().has(Cpu::tAVX512F)
                    && cpu().has(Cpu::tAVX512BW) && cpu().has(Cpu::tAVX512VL)
                    && cpu().has(Cpu::tAVX512DQ));
        case avx512_core_vnni:
            REG_AVX512_ISA(return cpu().has(Cpu::tAVX512F)
                    && cpu().has(Cpu::tAVX512BW) && cpu().has(Cpu::tAVX512VL)
                    && cpu().has(Cpu::tAVX512DQ)
                    && cpu().has(Cpu::tAVX512_VNNI));
        case avx512_core_bf16:
            REG_AVX512_ISA(return mayiuse(avx512_core_vnni, soft)
                    && cpu().has(Cpu::tAVX512_BF16));
        case avx512_core_bf16_ymm:
            REG_AVX512_ISA(return mayiuse(avx512_core_bf16, soft)
                    && cpu_isa_hints_utils::is_hints_bit_set(
                            prefer_ymm_bit, soft));
        case avx512_core_fp16:
            REG_AVX512_ISA(return cpu().has(Cpu::tAVX512_FP16)
                    && mayiuse(avx512_core_bf16, soft)
                    && mayiuse(avx2_vnni, soft));
        case avx10_2:
            REG_AVX512_ISA(return cpu().getAVX10version() >= 2
                    && cpu().has(Cpu::tAVX512F) && mayiuse(avx2_vnni_2, soft)
                    && cpu().has(Cpu::tAPX_F) && cpu().has(Cpu::tMOVRS)
                    && x64::apx::is_available());
        case avx10_2_aux:
            REG_AVX512_ISA(return mayiuse(avx10_2, soft)
                    && cpu().has(Cpu::tAVX10_V2_AUX));
        case amx_tile:
            REG_AMX_ISA(return cpu().has(Cpu::tAMX_TILE)
                    && x64::amx::is_available());
        case amx_int8:
            REG_AMX_ISA(return mayiuse(amx_tile, soft)
                    && cpu().has(Cpu::tAMX_INT8));
        case amx_bf16:
            REG_AMX_ISA(return mayiuse(amx_tile, soft)
                    && cpu().has(Cpu::tAMX_BF16));
        case amx_fp16:
            REG_AMX_ISA(return mayiuse(amx_tile, soft)
                    && cpu().has(Cpu::tAMX_FP16));
        case avx512_core_amx:
            REG_AMX_ISA(return mayiuse(amx_int8, soft)
                    && mayiuse(amx_bf16, soft)
                    && mayiuse(avx512_core_fp16, soft));
        case avx512_core_amx_fp16:
            REG_AMX_ISA(return mayiuse(avx512_core_amx, soft)
                    && mayiuse(amx_fp16, soft));
        case avx10_2_amx_2:
            REG_AMX_ISA(return mayiuse(avx10_2, soft) && mayiuse(amx_tile, soft)
                    && cpu().has(Cpu::tAMX_AVX512) && cpu().has(Cpu::tAMX_MOVRS)
                    && cpu().has(Cpu::tAMX_FP8));
        case avx10_2_ace:
            // ACE v1 detection:
            //   1. AVX10.2
            //   2. AVX10_V2_AUX
            //   3. ACE
            //   4. ACE_VSN >= 1 (and MAX_PALETTE >= 2)
            //   5. XCR0[20,18:17] = 0b111  (ace::is_available)
            //   6. XCR0[7:5] = 0b111       (implied by the AVX10 checks)
            //   7. CR4.OSXSAVE = 1         (implied by the XCR0 queries)
            REG_AMX_ISA(return mayiuse(avx10_2_aux, soft)
                    && mayiuse(amx_tile, soft) && cpu().has(Cpu::tACE)
                    && cpu().getACEversion() >= 1
                    && cpu().getMaxPalette() >= amx_palette_ace
                    && x64::ace::is_available());
        case isa_all: return false;
        case isa_undef: return true;
    }
    return false;
}

inline bool isa_has_int8_vnni(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_vnni) || is_superset(isa, avx2_vnni);
}

inline bool isa_has_s8s8(cpu_isa_t isa) {
    return is_superset(isa, avx2_vnni_2) || is_superset(isa, avx10_2)
            || is_superset(isa, amx_int8);
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_bf16);
}

inline bool isa_has_f16(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_fp16)
            || is_superset(isa, avx10_1_512_amx_fp16);
}

inline bool isa_has_masks(cpu_isa_t isa) {
    return is_superset(isa, avx512_core);
}

// True when matrix multiplication accumulates into tile registers, through
// TMUL, through the ACE outer products, or through both. This is "has tiles",
// not "can emit TMUL": the latter is false on an ACE-only part.
inline bool isa_has_tile_accumulators(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_amx) || is_superset(isa, avx10_2_ace);
}

// Check if the ISA has saturating conversion support
inline bool isa_has_sat_cvt(cpu_isa_t isa, data_type_t dt) {
    return utils::one_of(dt, data_type::u8, data_type::s8)
            && is_superset(isa, avx10_2);
}

inline int isa_max_vlen(cpu_isa_t isa) {
    const bool is_avx512 = is_superset(isa, avx512_core);
    const bool is_avx2 = is_superset(isa, avx2);

    assert(utils::one_of(true, is_avx512, is_avx2));
    MAYBE_UNUSED(is_avx2);

    if (is_avx512)
        return cpu_isa_traits_t<avx512_core>::vlen;
    else
        return cpu_isa_traits_t<avx2>::vlen;
}

inline int isa_num_vregs(cpu_isa_t isa) {
    const bool is_avx512 = is_superset(isa, avx512_core);
    const bool is_avx2 = is_superset(isa, avx2);

    assert(utils::one_of(true, is_avx512, is_avx2));
    MAYBE_UNUSED(is_avx2);

    if (is_avx512)
        return cpu_isa_traits_t<avx512_core>::n_vregs;
    else
        return cpu_isa_traits_t<avx2>::n_vregs;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_undef ? prefix STRINGIFY(undef) : \
    (isa) == avx2 ? prefix STRINGIFY(avx2) : \
    (isa) == avx2_vnni ? prefix STRINGIFY(avx2_vnni) : \
    (isa) == avx2_vnni_2 ? prefix STRINGIFY(avx2_vnni_2) : \
    (isa) == avx512_core ? prefix STRINGIFY(avx512_core) : \
    (isa) == avx512_core_vnni ? prefix STRINGIFY(avx512_core_vnni) : \
    (isa) == avx512_core_bf16 ? prefix STRINGIFY(avx512_core_bf16) : \
    (isa) == avx10_1_512 ? prefix STRINGIFY(avx10_1_512) : \
    (isa) == avx10_1_512_amx ? prefix STRINGIFY(avx10_1_512_amx) : \
    (isa) == avx10_1_512_amx_fp16 ? prefix STRINGIFY(avx10_1_512_amx_fp16) : \
    (isa) == avx10_2 ? prefix STRINGIFY(avx10_2) : \
    (isa) == avx10_2_amx_2 ? prefix STRINGIFY(avx10_2_amx_2) : \
    (isa) == avx10_2_ace ? prefix STRINGIFY(avx10_2_ace) : \
    prefix suffix_if_any)
/* clang-format on */

// Used to get the Multiply-Add Compute (MAC) datatype for primitive support on
// CPU ISAs with non-native instruction support.
inline data_type_t get_mac_emu_data_type(const data_type_t data_type,
        const cpu_isa_t isa, const bool req_emulation = true) {
    if (!req_emulation) return data_type;
    using namespace data_type;
    switch (data_type) {
        case bf16:
            if (utils::one_of(isa, avx2, avx2_vnni_2, avx512_core)) return f32;
            break;
        case f16:
            if (utils::one_of(
                        isa, avx2, avx2_vnni_2, avx512_core, avx512_core_fp16))
                return f32;
            break;
        case f8_e4m3:
        case f8_e5m2:
            if (isa == avx10_2) return f16;
            break;
        default: return data_type;
    }
    return data_type;
}

inline size_t data_type_vnni_granularity(const data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f32:
        case s32: return size_t(1);
        case f16:
        case bf16: return size_t(2);
        case f8_e5m2:
        case f8_e4m3:
        case s8:
        case u8: return size_t(4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return size_t(0); /* should not be reachable */
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
