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

#ifndef GPU_INTEL_SDPA_CONFIGS_HPP
#define GPU_INTEL_SDPA_CONFIGS_HPP

#include <iostream>

#include "common/c_types_map.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/sdpa/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

namespace micro = gemmstone::microkernel;

enum class property : int {
    none = 0x0,
    second_token = 0x1,
    quantized = 0x2,
    integrated = 0x4,
    fma = 0x8,
    f32 = 0x10,
    f16_accumulate = 0x20,
};

property operator|(property a, property b);
property operator&(property a, property b);
property operator^(property a, property b);
property &operator|=(property &a, property b);
property &operator&=(property &a, property b);
property &operator^=(property &a, property b);

struct config_query_t {
    static constexpr int any = -1;
    compute::gpu_arch_t arch;
    int head_size;
    int seq_len = any;
    property prop = property::none;

    config_query_t(compute::gpu_arch_t arch_, int head_size_,
            int seq_len_ = any, property property_ = property::none)
        : arch(arch_)
        , head_size(head_size_)
        , seq_len(seq_len_)
        , prop(property_) {}
};

struct config_criteria_t {
    static constexpr int any = -1;
    compute::gpu_arch_t arch;
    int head_size;
    int seq_len = any;
    property prop = property::none;
    config_criteria_t(compute::gpu_arch_t a, int hs)
        : arch(a), head_size(hs), seq_len(any), prop(property::none) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, int sq)
        : arch(a), head_size(hs), seq_len(sq), prop(property::none) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, property prop)
        : arch(a), head_size(hs), seq_len(any), prop(prop) {}
    config_criteria_t(compute::gpu_arch_t a, int hs, int sq, property prop)
        : arch(a), head_size(hs), seq_len(sq), prop(prop) {}
};

struct fwd_config_record_t {
    config_criteria_t criteria;
    fwd_config_t config;
};

struct bwd_config_record_t {
    config_criteria_t criteria;
    bwd_config_t config;
};

// Common criteria matching: returns true if query matches key criteria
bool criteria_matches(
        const config_criteria_t &key, const config_query_t &query);

std::ostream &operator<<(std::ostream &s, const config_query_t &q);
std::ostream &operator<<(std::ostream &s, const config_criteria_t &c);
std::ostream &operator<<(std::ostream &s, const fwd_config_t &c);

bool operator==(const fwd_config_record_t &key, const config_query_t &query);
bool operator==(const bwd_config_record_t &key, const config_query_t &query);
bool operator<(const config_criteria_t &lhs, const config_criteria_t &rhs);
bool operator<(const fwd_config_record_t &lhs, const fwd_config_record_t &rhs);
bool operator<(const bwd_config_record_t &lhs, const bwd_config_record_t &rhs);

fwd_config_t *choose_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32, bool is_f16_accumulate);
bwd_config_t *choose_bwd_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t qry, dim_t seq, dim_t batch_heads, bool is_thin_q,
        bool is_quantized, bool is_integrated, bool is_fma, bool is_f32,
        bool is_causal);
dim_t round_up_seq_interval(dim_t seq, compute::gpu_arch_t arch);

dim_t nearest_conf_seq_interval(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32, bool is_f16_accumulate);

// serializable options for microkernel configuration
// follows reduced subset of structs from gemmstone that
// are used for ukernel generation
struct ukernel_serialized_opts_t
    : trivially_serializable_t<ukernel_serialized_opts_t> {

    ukernel_serialized_opts_t() = default;
    ukernel_serialized_opts_t(micro::GEMMOptions opts)
        : localA(opts.localA)
        , localB(opts.localB)
        , slmPtr(opts.slmPtr)
        , scaleA(opts.scaleA)
        , offsetA(opts.offsetA) {}
    bool localA, localB, slmPtr, scaleA, offsetA;
    uint8_t padding[3] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_opts_t);
static_assert(sizeof(ukernel_serialized_opts_t) == 8,
        "Expected sizeof(ukernel_serialized_opts_t) == 8");

struct ukernel_serialized_hwinfo_t
    : trivially_serializable_t<ukernel_serialized_hwinfo_t> {

    ukernel_serialized_hwinfo_t() = default;
    ukernel_serialized_hwinfo_t(micro::HWInformation &hwInfo)
        : gmdid(static_cast<uint32_t>(hwInfo.gmdid))
        , euCount(static_cast<uint32_t>(hwInfo.euCount))
        , systolicAvailable(hwInfo.systolicAvailable)
        , isEfficient64Bit(hwInfo.isEfficient64Bit) {}

    uint32_t gmdid;
    uint32_t euCount;
    bool systolicAvailable;
    bool isEfficient64Bit;
    uint8_t padding[6] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_hwinfo_t);
static_assert(sizeof(ukernel_serialized_hwinfo_t) == 16,
        "Expected sizeof(ukernel_serialized_hwinfo_t) == 16");

struct ukernel_serialized_sizes_t
    : trivially_serializable_t<ukernel_serialized_sizes_t> {

    ukernel_serialized_sizes_t() = default;
    ukernel_serialized_sizes_t(gemmstone::SizeParams &sizes)
        : batch(sizes.batch), m(sizes.m), n(sizes.n), k(sizes.k) {}
    int64_t batch = 0;
    int64_t m = 0, n = 0, k = 0;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_sizes_t);
static_assert(sizeof(ukernel_serialized_sizes_t) == 32,
        "Expected sizeof(ukernel_serialized_sizes_t) == 32");

struct ukernel_serialized_problem_t
    : trivially_serializable_t<ukernel_serialized_problem_t> {

    ukernel_serialized_problem_t() = default;
    ukernel_serialized_problem_t(gemmstone::GEMMProblem &problem)
        : Ta_ext(static_cast<uint32_t>(problem.Ta_ext))
        , Tb_ext(static_cast<uint32_t>(problem.Tb_ext))
        , Tc_ext(static_cast<uint32_t>(problem.Tc_ext))
        , Ta(static_cast<uint32_t>(problem.Ta))
        , Tb(static_cast<uint32_t>(problem.Tb))
        , Tc(static_cast<uint32_t>(problem.Tc))
        , Ts(static_cast<uint32_t>(problem.Ts))
        , A_layout(static_cast<int>(problem.A.layout))
        , B_layout(static_cast<int>(problem.B.layout))
        , C_layout(static_cast<int>(problem.C.layout))
        , A_tileR(problem.A.tileR)
        , B_tileR(problem.B.tileR)
        , A_tileC(problem.A.tileC)
        , B_tileC(problem.B.tileC)
        , A_crosspack(problem.A.crosspack)
        , B_crosspack(problem.B.crosspack)
        , A_alignment(problem.A.alignment)
        , A_scale_alignment(problem.A_scale.alignment)
        , AO_alignment(problem.AO.alignment)
        , B_alignment(problem.B.alignment)
        , asPtrDims(problem.asPtrDims)
        , aOffset(static_cast<int>(problem.aOffset))
        , Ta_scale(static_cast<uint32_t>(problem.Ta_scale))
        , A_scale_layout(static_cast<int>(problem.A_scale.layout))
        , Tao(static_cast<uint32_t>(problem.Tao))
        , AO_layout(static_cast<int>(problem.AO.layout))
        , aoPtrDims(problem.aoPtrDims)
        , aqGroupM(problem.aqGroupM)
        , aqGroupK(problem.aqGroupK) {}

    uint32_t Ta_ext, Tb_ext, Tc_ext;
    uint32_t Ta, Tb, Tc, Ts;

    int A_layout;
    int B_layout;
    int C_layout;

    uint16_t A_tileR, B_tileR;
    uint16_t A_tileC, B_tileC;
    uint8_t A_crosspack, B_crosspack;

    uint8_t A_alignment;
    uint8_t A_scale_alignment;
    uint8_t AO_alignment;
    uint8_t B_alignment;
    // trivially serializable classes require alignment to 8-byte boundaries
    // padding0 bumps class size from 54->56 bytes so uint8_t arguments
    // related to alignment can be grouped together rather than placed at the end of the struct
    uint8_t padding0[2] = {0};

    int asPtrDims;
    int aOffset;

    uint32_t Ta_scale;
    int A_scale_layout;
    uint32_t Tao;

    int AO_layout;
    int aoPtrDims;
    int aqGroupM;
    int aqGroupK;
    uint8_t padding1[4] = {0};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(ukernel_serialized_problem_t);
static_assert(sizeof(ukernel_serialized_problem_t) == 96,
        "Expected sizeof(ukernel_serialized_problem_t) == 96");

struct micro_fwd_ukernel_params_t
    : trivially_serializable_t<micro_fwd_ukernel_params_t> {
    int unroll_m_kq, unroll_n_kq;
    int unroll_m_vs, unroll_n_vs;
    int wg_m_kq, wg_n_kq;
    int wg_m_vs, wg_n_vs;

    ukernel_serialized_hwinfo_t hwinfo;
    ukernel_serialized_problem_t problem_kq;
    ukernel_serialized_problem_t problem_vs;
    ukernel_serialized_opts_t opts_kq;
    ukernel_serialized_opts_t opts_vs;
    ukernel_serialized_sizes_t sizes_kq, sizes_vs;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_fwd_ukernel_params_t);

struct micro_bwd_ukernel_params_t
    : trivially_serializable_t<micro_bwd_ukernel_params_t> {
    int unroll_m_BcBr, unroll_n_BcBr;
    int unroll_m_DBc, unroll_n_DBc;
    int unroll_m_DBr, unroll_n_DBr;

    int wg_m_BcBr, wg_n_BcBr;
    int wg_m_DBc, wg_n_DBc;
    int wg_m_DBr, wg_n_DBr;

    ukernel_serialized_hwinfo_t hwinfo;

    ukernel_serialized_problem_t problem_kq;
    ukernel_serialized_problem_t problem_vs;
    ukernel_serialized_problem_t problem_vtdA;
    ukernel_serialized_problem_t problem_ktq;
    ukernel_serialized_problem_t problem_qdSt;

    ukernel_serialized_opts_t opts_kq;
    ukernel_serialized_opts_t opts_vs;
    ukernel_serialized_opts_t opts_vtdA;
    ukernel_serialized_opts_t opts_ktq;
    ukernel_serialized_opts_t opts_qdSt;

    ukernel_serialized_sizes_t sizes_kq, sizes_vs;
    ukernel_serialized_sizes_t sizes_vtdA;
    ukernel_serialized_sizes_t sizes_ktq;
    ukernel_serialized_sizes_t sizes_qdSt;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_bwd_ukernel_params_t);

void deserialize_config_to_gemmstone(micro::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        micro::GEMMOptions &opts_kq, micro::GEMMOptions &opts_vs,
        gemmstone::SizeParams &sizes_kq, gemmstone::SizeParams &sizes_vs,
        const micro_fwd_ukernel_params_t &ukernel_config);

void deserialize_config_to_gemmstone(micro::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        gemmstone::GEMMProblem &problem_vtdA,
        gemmstone::GEMMProblem &problem_ktq,
        gemmstone::GEMMProblem &problem_qdSt, micro::GEMMOptions &opts_kq,
        micro::GEMMOptions &opts_vs, micro::GEMMOptions &opts_vtdA,
        micro::GEMMOptions &opts_ktq, micro::GEMMOptions &opts_qdSt,
        gemmstone::SizeParams &sizes_kq, gemmstone::SizeParams &sizes_vs,
        gemmstone::SizeParams &sizes_vtdA, gemmstone::SizeParams &sizes_ktq,
        gemmstone::SizeParams &sizes_qdSt,
        const micro_bwd_ukernel_params_t &ukernel_config);

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
