/*******************************************************************************
* Copyright 2026 Intel Corporation
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

// Device-free tests of the forward SDPA tile selector

#include "gtest/gtest.h"

#include "src/gpu/intel/sdpa/select.cpp"

namespace {

namespace sdpa = dnnl::impl::gpu::intel::sdpa;
using dnnl::impl::dim_t;
using dnnl::impl::gpu::intel::compute::gpu_arch_t;

// Arc A750: 28 Xe-cores of 16 EUs, 32-byte GRF, SIMD8 subgroups
sdpa::fwd_hw_t dg2_hw() {
    sdpa::fwd_hw_t hw;
    hw.arch = gpu_arch_t::xe_hpg;
    hw.eu_count = 448;
    hw.eus_per_subslice = 16;
    hw.threads_per_eu_128 = 8;
    hw.threads_per_eu_256 = 4;
    hw.grf_bytes = 32;
    hw.slm_per_wg = 64 * 1024;
    hw.slm_per_subslice = 128 * 1024;
    hw.max_wg_items_128 = 1024;
    hw.l3_bytes = 16u << 20;
    hw.subgroup_size = 8;
    hw.integrated = false;
    hw.systolic = true;
    return hw;
}

// Ponte Vecchio: 128 Xe-cores of 8 EUs, 64-byte GRF, SIMD16 subgroups
sdpa::fwd_hw_t pvc_hw() {
    sdpa::fwd_hw_t hw;
    hw.arch = gpu_arch_t::xe_hpc;
    hw.eu_count = 1024;
    hw.eus_per_subslice = 8;
    hw.threads_per_eu_128 = 8;
    hw.threads_per_eu_256 = 4;
    hw.grf_bytes = 64;
    hw.slm_per_wg = 128 * 1024;
    hw.slm_per_subslice = 128 * 1024;
    hw.max_wg_items_128 = 1024;
    hw.l3_bytes = 192u << 20;
    hw.subgroup_size = 16;
    hw.integrated = false;
    hw.systolic = true;
    return hw;
}

sdpa::fwd_problem_t problem(int d, dim_t keys, dim_t queries, dim_t bh) {
    sdpa::fwd_problem_t p;
    p.d_qk = p.d_v = d;
    p.d_max_kq = p.d_max_v = 32;
    while (p.d_max_kq < d)
        p.d_max_kq = p.d_max_v = p.d_max_kq * 2;
    p.keys = keys;
    p.queries = queries;
    p.batch_heads = bh;
    p.q_align = p.k_align = p.v_align = p.dst_align = 64;
    return p;
}

sdpa::fwd_config_t cfg(int a, int b, int c, int d, int e, int f, int g, int h) {
    return sdpa::fwd_config_t {a, b, c, d, e, f, g, h};
}

} // namespace

TEST(sdpa_select, shipped_configs_are_valid) {
    std::string reason;
    // {xe_hpg, 64} and {xe_hpg, 128, second_token} from configs.cpp
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(32, 16, 16, 16, 4, 8, 4, 8),
            problem(64, 512, 512, 8), dg2_hw(), &reason))
            << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(8, 16, 16, 8, 16, 1, 8, 2),
            problem(128, 1024, 1, 32), dg2_hw(), &reason))
            << reason;
    // {xe_hpc, 32} and {xe_hpc, 128}
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 64, 32, 16, 4, 2, 1, 8),
            problem(32, 1024, 1024, 64), pvc_hw(), &reason))
            << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 64, 32, 16, 16, 2, 4, 8),
            problem(128, 4096, 4096, 32), pvc_hw(), &reason))
            << reason;
}

TEST(sdpa_select, rejects_broken_configs) {
    const auto p = problem(64, 512, 512, 8);
    std::string reason;
    // q tiles differ: 16*8 vs 16*4
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 4, 8, 8, 4), p, dg2_hw(), &reason));
    EXPECT_NE(reason.find("tile N"), std::string::npos) << reason;
    // VS tile below the value head size: 16*2 < 64
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 4, 8, 2, 8), p, dg2_hw(), &reason));
    // Subgroup counts differ: 32 vs 31 (as in the xe_hpc 576 entry)
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(32, 16, 32, 16, 32, 1, 31, 1),
            problem(576, 1024, 1, 8), pvc_hw(), &reason));
    EXPECT_NE(reason.find("subgroup counts"), std::string::npos) << reason;
    // Unroll of 8 is not available at or above XeHPC
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(8, 16, 16, 8, 16, 1, 8, 2),
            problem(128, 1024, 1, 32), pvc_hw(), &reason));
    // A 32x32 KQ accumulator alone is 128 registers on a 32-byte GRF part;
    // gemmstone cannot generate it (observed on the A750)
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 16, 8, 2, 4, 4), p, dg2_hw(), &reason));
    EXPECT_NE(reason.find("register"), std::string::npos) << reason;
    // 64x64 never fits anywhere
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(64, 64, 64, 64, 4, 4, 4, 4),
            problem(256, 1024, 1024, 8), pvc_hw(), &reason));
    // Observed to fail generation on the A750: a 64-wide KQ N tile, and 32
    // subgroups along N
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(8, 64, 16, 16, 16, 1, 4, 4), p, dg2_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(8, 8, 64, 8, 1, 32, 1, 32), p, dg2_hw(), &reason));
    // Also failed on the A750: 64 subgroups, a 64-row VS tile with f16 V,
    // and a 32-row KQ tile with a single subgroup along N
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(8, 32, 16, 8, 32, 2, 8, 8),
            problem(128, 77, 64, 8), dg2_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(8, 16, 64, 8, 4, 1, 2, 2),
            problem(128, 32, 32, 16), dg2_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(32, 16, 8, 8, 32, 1, 16, 2),
            problem(128, 4096, 1, 32), dg2_hw(), &reason));
    // The quantized {xe_hpg, 256} entry keeps its 64-row VS tile with int8 V
    auto q8 = problem(256, 512, 512, 8);
    q8.v_bits = 8;
    q8.v_quantized = true;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 64, 8, 8, 4, 4, 8), q8, dg2_hw(), &reason))
            << reason;
    // A750 KPI sweeps: a 512-element KQ tile fails over a 32-column Q tile
    // with 2-byte aligned K rows and with 32 subgroups at D >= 256; the
    // 64-column and 16-subgroup variants build
    auto k77 = problem(64, 77, 4096, 10);
    k77.k_align = 2;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 4, 2, 4, 2), k77, dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 4, 4, 4, 4), k77, dg2_hw(), &reason))
            << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 16, 16, 4, 8, 4, 8), k77, dg2_hw(), &reason))
            << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 32, 32, 8, 32, 1, 8, 4),
            problem(256, 1024, 1024, 16), dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(32, 16, 32, 16, 8, 2, 8, 2),
            problem(256, 1024, 1024, 16), dg2_hw(), &reason))
            << reason;
    // Transposed K (head dimension contiguous): 512-element KQ tiles fail
    // the kernel build; the same tile builds with K rows contiguous
    auto tk = problem(128, 385, 1, 2);
    tk.transpose_k = true;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 8, 4, 8, 4), tk, dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 16, 16, 8, 4, 8, 4), tk, dg2_hw(), &reason))
            << reason;
    // 64-column Q tile over 16 subgroups builds (prefill sweep), over 32
    // subgroups only at D=64
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 16, 32, 16, 4, 4, 4, 4), tk, dg2_hw(), &reason))
            << reason;
    auto tk64 = problem(64, 2049, 1, 24);
    tk64.transpose_k = true;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 8, 8, 4, 4, 8), tk64, dg2_hw(), &reason))
            << reason;
    // The cooperative K load cannot split over 6 subgroups along N, in
    // either layout
    auto tk96 = problem(96, 1025, 1, 32);
    tk96.transpose_k = true;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 6, 4, 6), tk96, dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 8, 4, 8), tk96, dg2_hw(), &reason))
            << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(32, 16, 32, 16, 4, 6, 4, 6),
            problem(96, 4096, 4096, 64), dg2_hw(), &reason));
    tk.transpose_k = false;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 16, 16, 16, 8, 4, 8, 4), tk, dg2_hw(), &reason))
            << reason;
    // D=512: 32-subgroup work-groups fail the build with a 256-element KQ
    // tile and a 512-element VS tile, or with transposed K; the sweep
    // winners build
    auto d512 = problem(512, 385, 1, 2);
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 16, 2, 16, 2), d512, dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 8, 32, 1, 16, 2), d512, dg2_hw(), &reason))
            << reason;
    d512.transpose_k = true;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 8, 32, 8, 16, 2, 16, 2), d512, dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(8, 16, 32, 8, 32, 1, 16, 2), d512, dg2_hw(), &reason))
            << reason;
    // Small head sizes: a VS tile with one subgroup along M and several
    // along N fails preflight; two along M build
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 8, 32, 8, 1, 8, 1, 8),
            problem(32, 16, 16, 8), dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 8, 16, 8, 2, 4, 2, 4),
            problem(32, 16, 16, 8), dg2_hw(), &reason))
            << reason;
    // Odd work-group dimensions fail preflight unless the other one is 1;
    // the even cover for D=80 (8 x 10) builds
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 16, 16, 16, 5, 4, 5, 4),
            problem(80, 1024, 1024, 16), dg2_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(8, 8, 16, 8, 5, 2, 5, 2),
            problem(80, 1024, 1024, 16), dg2_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 8, 16, 8, 5, 1, 5, 1),
            problem(80, 1024, 1024, 16), dg2_hw(), &reason))
            << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(8, 16, 8, 16, 10, 2, 10, 2),
            problem(80, 1024, 1024, 16), dg2_hw(), &reason))
            << reason;
    // PVC D=512 sweep: a 16-column VS tile over 32 subgroups along M does
    // not generate with f16 V; the 32-column tile with the same work-group
    // does, and the shipped int8 entry of that shape is unaffected
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 16, 16, 16, 32, 1, 32, 1),
            problem(512, 512, 512, 8), pvc_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 32, 16, 32, 32, 1, 32, 1),
            problem(512, 512, 512, 8), pvc_hw(), &reason))
            << reason;
    // The same KQ N tile is fine on XeHPC, as in the shipped {xe_hpc, 64}
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(16, 64, 32, 16, 8, 2, 2, 8),
            problem(64, 1024, 1024, 16), pvc_hw(), &reason))
            << reason;
    // PVC K=77 landscape and D=512 decode: a 1024-element KQ tile with one
    // subgroup along N fails with 2-byte aligned K; two subgroups along N
    // build, as does the same tile with aligned K
    auto k77_pvc = problem(64, 77, 4096, 10);
    k77_pvc.k_align = 2;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(64, 16, 32, 16, 2, 1, 2, 1), k77_pvc, pvc_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(64, 16, 32, 16, 2, 2, 2, 2), k77_pvc, pvc_hw(), &reason))
            << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(64, 16, 32, 16, 4, 1, 4, 1),
            problem(128, 4096, 1, 32), pvc_hw(), &reason))
            << reason;
    auto k385 = problem(512, 385, 1, 2);
    k385.k_align = 2;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 32, 32, 1, 32, 1), k385, pvc_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(cfg(32, 32, 16, 32, 32, 1, 32, 1),
            problem(512, 512, 512, 8), pvc_hw(), &reason))
            << reason;
    // PVC builds non-power-of-two subgroup counts along M, but the f32 sweep
    // showed wrong results with 6, so they are rejected; along N they fail
    auto k1025 = problem(96, 1025, 1, 32);
    k1025.k_align = 2;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(64, 16, 32, 16, 6, 4, 6, 4), k1025, pvc_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 32, 32, 32, 4, 5, 4, 5),
            problem(80, 1024, 1024, 16), pvc_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(64, 16, 64, 16, 2, 6, 2, 6),
            problem(96, 2048, 2048, 32), pvc_hw(), &reason));
    // xe2/xe3 do not dispatch decode over 2-byte aligned f16 K rows; the
    // same config is fine with aligned K or on XeHPC
    auto xe2_hw = pvc_hw();
    xe2_hw.arch = gpu_arch_t::xe2;
    auto k385_dec = problem(128, 385, 1, 2);
    k385_dec.k_align = 2;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 16, 16, 2, 8, 4), k385_dec, xe2_hw, &reason));
    EXPECT_NE(reason.find("decode"), std::string::npos) << reason;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 16, 16, 2, 8, 4), k385_dec, pvc_hw(), &reason))
            << reason;
    k385_dec.k_align = 64;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 16, 16, 2, 8, 4), k385_dec, xe2_hw, &reason))
            << reason;
    // The PVC 1024-element rule for 2-byte aligned K does not apply on xe2
    auto k77_xe2 = problem(64, 77, 4096, 10);
    k77_xe2.k_align = 2;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 32, 32, 16, 8, 1, 4, 2), k77_xe2, xe2_hw, &reason))
            << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 32, 16, 8, 1, 4, 2), k77_xe2, pvc_hw(), &reason));
    // xe2 gives wrong results with a non-power-of-two subgroup count along
    // M; the power-of-two control and the same config on PVC are accepted
    const auto d80 = problem(80, 1024, 1024, 16);
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 32, 5, 2, 5, 2), d80, xe2_hw, &reason));
    EXPECT_NE(reason.find("power of two"), std::string::npos) << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(cfg(16, 16, 16, 16, 6, 1, 6, 1),
            problem(96, 1024, 1024, 32), xe2_hw, &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 32, 8, 2, 8, 2), d80, xe2_hw, &reason))
            << reason;
    auto xe3p_hw = xe2_hw;
    xe3p_hw.arch = gpu_arch_t::xe3p;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 32, 5, 2, 5, 2), d80, xe3p_hw, &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(32, 32, 16, 32, 8, 2, 8, 2), d80, xe3p_hw, &reason))
            << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 16, 16, 5, 2, 5, 2), d80, pvc_hw(), &reason));
    // PVC f32 (FMA): 64-row VS and 64-column KQ tiles fail to build
    auto f32 = problem(128, 2048, 2048, 32);
    f32.fma = f32.f32 = true;
    f32.q_bits = f32.k_bits = f32.v_bits = f32.dst_bits = 32;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 64, 16, 2, 2, 2, 2), f32, pvc_hw(), &reason));
    EXPECT_NE(reason.find("f32 FMA"), std::string::npos) << reason;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 64, 16, 32, 16, 1, 8, 2), f32, pvc_hw(), &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 8, 4, 8), f32, pvc_hw(), &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 4, 4, 4), f32, pvc_hw(), &reason))
            << reason;
    // 8 subgroups along N in VS alone fails too (D=64, K=77 landscape)
    auto f32_d64 = problem(64, 77, 77, 20);
    f32_d64.fma = f32_d64.f32 = true;
    f32_d64.q_bits = f32_d64.k_bits = f32_d64.v_bits = f32_d64.dst_bits = 32;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 32, 16, 16, 8, 4, 4, 8), f32_d64, pvc_hw(), &reason));
    // The shipped f32 decode config: 16-column VS over 32 subgroups along M
    // builds with 32-bit V
    auto f32_dec = problem(128, 4096, 1, 32);
    f32_dec.fma = f32_dec.f32 = true;
    f32_dec.q_bits = f32_dec.k_bits = f32_dec.v_bits = f32_dec.dst_bits = 32;
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 32, 1, 32, 1), f32_dec, pvc_hw(), &reason))
            << reason;
    // CRI f32 (FMA): KQ tiles wider than 16 columns do not generate
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 32, 16, 8, 2, 4, 4), f32, xe3p_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 64, 16, 2, 2, 2, 2), f32, xe3p_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 8, 4, 8), f32, xe3p_hw, &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(64, 16, 32, 16, 4, 4, 4, 4), f32, xe3p_hw, &reason))
            << reason;
    // BMG f32 (FMA): the PVC limits
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 64, 16, 2, 2, 2, 2), f32, xe2_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 64, 16, 32, 16, 1, 8, 2), f32, xe2_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 8, 4, 8), f32, xe2_hw, &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 4, 4, 4), f32, xe2_hw, &reason))
            << reason;
    // PTL f32 (FMA): the PVC limits
    auto xe3_hw = xe2_hw;
    xe3_hw.arch = gpu_arch_t::xe3;
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 64, 16, 2, 2, 2, 2), f32, xe3_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 64, 16, 32, 16, 1, 8, 2), f32, xe3_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 8, 4, 8), f32, xe3_hw, &reason));
    EXPECT_TRUE(sdpa::fwd_config_valid(
            cfg(16, 16, 32, 16, 4, 4, 4, 4), f32, xe3_hw, &reason))
            << reason;
    // CRI: 32x32 VS tile with one subgroup along M is wrong at D=32
    auto d32 = problem(32, 128, 128, 4096);
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(32, 32, 32, 32, 1, 2, 1, 2), d32, xe3p_hw, &reason));
    EXPECT_FALSE(sdpa::fwd_config_valid(
            cfg(16, 64, 32, 32, 2, 2, 1, 4), d32, xe3p_hw, &reason));
}

TEST(sdpa_select, slm_matches_kernel_layout) {
    // Q_slm = D_MAX * q_tile * 2, S_slm = kv_tile * q_tile * 2,
    // S_sum = q_tile * wg_m_kq * 4, S_max = q_tile * 4
    const auto p = problem(64, 512, 512, 8);
    const auto c = cfg(32, 16, 16, 16, 4, 8, 4, 8); // kv 128, q 128
    const int expected = 64 * 128 * 2 + 128 * 128 * 2 + 128 * 4 * 4 + 128 * 4;
    EXPECT_EQ(sdpa::fwd_slm_bytes(c, p), expected);
}

TEST(sdpa_select, grf_estimate_grows_with_tiles) {
    const auto p = problem(128, 1024, 1024, 8);
    const auto m = sdpa::fwd_model_coefs(gpu_arch_t::xe_hpc);
    EXPECT_EQ(sdpa::fwd_estimate_grfs(
                      cfg(16, 16, 16, 16, 8, 2, 8, 2), p, pvc_hw(), m),
            128);
    EXPECT_EQ(sdpa::fwd_estimate_grfs(
                      cfg(64, 64, 64, 64, 4, 4, 4, 4), p, pvc_hw(), m),
            256);
}

TEST(sdpa_select, enumeration_is_sorted_and_contains_shipped_config) {
    const auto hw = dg2_hw();
    const auto p = problem(64, 512, 512, 8);
    const auto ranked
            = sdpa::fwd_enumerate(p, hw, sdpa::fwd_model_coefs(hw.arch));
    ASSERT_GT(ranked.size(), 50u);
    for (size_t i = 1; i < ranked.size(); i++)
        EXPECT_LE(ranked[i - 1].cost_us, ranked[i].cost_us);
    const std::string shipped
            = sdpa::fwd_config_str(cfg(32, 16, 16, 16, 4, 8, 4, 8));
    bool found = false;
    for (const auto &c : ranked)
        found |= (sdpa::fwd_config_str(c.config) == shipped);
    EXPECT_TRUE(found);
    for (const auto &c : ranked) {
        EXPECT_GT(c.cost_us, 0.0);
        EXPECT_GE(c.wg_per_subslice, 1);
    }
}

TEST(sdpa_select, decode_prefers_narrow_q_tiles) {
    const auto hw = dg2_hw();
    const auto p = problem(128, 4096, 1, 32);
    const auto ranked
            = sdpa::fwd_enumerate(p, hw, sdpa::fwd_model_coefs(hw.arch));
    ASSERT_FALSE(ranked.empty());
    // A single query per head cannot use a wide q tile
    EXPECT_LE(ranked.front().q_tile, 32);
}

TEST(sdpa_select, key_and_config_strings) {
    const auto hw = dg2_hw();
    auto p = problem(64, 500, 70, 12);
    p.k_align = 2;
    const auto key = sdpa::fwd_make_key(p, hw);
    EXPECT_EQ(key.keys, 512);
    EXPECT_EQ(key.queries, 128);
    EXPECT_EQ(key.batch_heads, 16);
    EXPECT_EQ(key.k_align, 2);
    EXPECT_EQ(key.q_align, 64);
    const std::string s = key.str();
    EXPECT_NE(s.find("xe_hpg:d64x64:k512:q128:bh16"), std::string::npos) << s;
    EXPECT_EQ(s.find(' '), std::string::npos) << s;
    // The K layout is part of the key: the kernel differs per layout
    EXPECT_EQ(s.substr(s.size() - 8), ":tr0:tk0") << s;
    p.transpose_k = true;
    EXPECT_EQ(sdpa::fwd_make_key(p, hw).str().substr(s.size() - 4), ":tk1");

    sdpa::fwd_config_t c {};
    ASSERT_TRUE(sdpa::fwd_parse_config("32,16,16,16,4,8,4,8", c));
    EXPECT_EQ(sdpa::fwd_config_str(c), "32,16,16,16,4,8,4,8");
    EXPECT_FALSE(sdpa::fwd_parse_config("32,16,16", c));
}

TEST(sdpa_select, device_tagged_table_lines_take_precedence) {
    std::unordered_map<std::string, sdpa::fwd_config_t> t;
    sdpa::add_table_line(t, "K 16,16,16,16,8,4,8,4   # arch-wide");
    sdpa::add_table_line(t, "K:eu448 32,16,16,8,16,2,8,4   # A750 only");
    sdpa::add_table_line(t, "# comment only");
    sdpa::add_table_line(t, "broken 1,2,3");
    EXPECT_EQ(t.size(), 2u);

    sdpa::fwd_config_t c {};
    bool dev = false;
    ASSERT_TRUE(sdpa::lookup_in(t, "K", 448, c, &dev));
    EXPECT_TRUE(dev);
    EXPECT_EQ(sdpa::fwd_config_str(c), "32,16,16,8,16,2,8,4");
    ASSERT_TRUE(sdpa::lookup_in(t, "K", 512, c, &dev));
    EXPECT_FALSE(dev);
    EXPECT_EQ(sdpa::fwd_config_str(c), "16,16,16,16,8,4,8,4");
    EXPECT_FALSE(sdpa::lookup_in(t, "other", 448, c, &dev));
}

TEST(sdpa_select, hw_and_problem_strings_round_trip) {
    const auto hw = dg2_hw();
    sdpa::fwd_hw_t hw2;
    ASSERT_TRUE(sdpa::fwd_parse_hw(sdpa::fwd_hw_str(hw), hw2));
    EXPECT_EQ(sdpa::fwd_hw_str(hw2), sdpa::fwd_hw_str(hw));
    EXPECT_EQ(hw2.eu_count, 448);
    EXPECT_EQ(hw2.arch, gpu_arch_t::xe_hpg);

    auto p = problem(128, 777, 64, 8);
    p.mask = sdpa::fwd_mask_t::causal_bottom_right;
    p.k_align = 2;
    p.k_bits = 8;
    p.k_quantized = true;
    p.k_group_size = 32;
    p.training = true;
    sdpa::fwd_problem_t p2;
    ASSERT_TRUE(sdpa::fwd_parse_problem(sdpa::fwd_problem_str(p), p2));
    EXPECT_EQ(sdpa::fwd_problem_str(p2), sdpa::fwd_problem_str(p));
    EXPECT_EQ(p2.keys, 777);
    EXPECT_TRUE(p2.k_quantized);

    EXPECT_FALSE(sdpa::fwd_parse_hw("garbage", hw2));
    EXPECT_FALSE(sdpa::fwd_parse_problem("k=5", p2));
}

TEST(sdpa_select, coefficient_table_round_trips) {
    auto m = sdpa::fwd_model_coefs(gpu_arch_t::xe2);
    const auto &names = sdpa::fwd_model_coef_names();
    ASSERT_FALSE(names.empty());
    EXPECT_TRUE(sdpa::fwd_model_coef_set(m, "barrier_cycles", 123.0));
    EXPECT_FALSE(sdpa::fwd_model_coef_set(m, "no_such_coef", 1.0));
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == "barrier_cycles") {
            EXPECT_DOUBLE_EQ(sdpa::fwd_model_coef_get(m, (int)i), 123.0);
        }
    }
}
