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

// Unit tests for the IR-based online-softmax epilogue of the fused CPU SDPA
// kernel (src/graph/backend/dnnl/kernels/sdp_fused_softmax_ir.hpp). Each test
// builds one epilogue IR, JITs it through the x64 CPU IR pipeline, runs it and
// checks it against an independent scalar reference. The eltwise exp is a
// polynomial approximation, so denominator-dependent outputs use a relative
// tolerance; the running max is a plain max and stays exact.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/backend/dnnl/kernels/sdp_fused_softmax_ir.hpp"

namespace dnnl {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::cpu::x64::sdp_softmax_ir;

// Tests that require generating a kernel require AVX2.
#define SKIP_IF_NO_AVX2() \
    do { \
        if (!mayiuse(avx2)) GTEST_SKIP() << "IR emitter require AVX2"; \
    } while (0)

// Scalar reference for one online-softmax row update, matching one row of
// build_softmax_tile_ir (no select mask).
void ref_softmax_row(std::vector<float> &scores, float scale, float &m,
        float &l, float &old_coef) {
    const int w = (int)scores.size();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float m_old = m, l_old = l;
    float m_new = m_old;
    for (int j = 0; j < w; j++) {
        scores[j] *= scale;
        m_new = std::max(m_new, scores[j]);
    }
    // corr is 0 for the first tile (m_old == -inf), as in the fused kernel.
    const float corr = m_old == neg_inf ? 0.f : std::exp(m_old - m_new);
    float tile_sum = 0.f;
    for (int j = 0; j < w; j++) {
        const float e = std::exp(scores[j] - m_new);
        scores[j] = e;
        tile_sum += e;
    }
    const float l_new = l_old * corr + tile_sum;
    for (int j = 0; j < w; j++)
        scores[j] /= l_new;
    m = m_new;
    l = l_new;
    old_coef = (l_old * corr) / l_new;
}

// Scalar reference for one online-softmax row update with the select mask, as
// in build_softmax_tile_ir(..., has_select=true). A lane is kept when cond != 0
// (fusiable) or cond == 0 (non-fusiable); otherwise it takes `fill`.
void ref_softmax_row_masked(std::vector<float> &scores, float scale,
        const std::vector<uint8_t> &cond, float fill, bool fusiable, float &m,
        float &l, float &old_coef) {
    const int w = (int)scores.size();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float m_old = m, l_old = l;
    float m_new = m_old;
    for (int j = 0; j < w; j++) {
        float v = scores[j] * scale;
        const bool c = cond[j] != 0;
        const bool keep = fusiable ? c : !c;
        if (!keep) v = fill;
        scores[j] = v;
        m_new = std::max(m_new, v);
    }
    const float corr = m_old == neg_inf ? 0.f : std::exp(m_old - m_new);
    float tile_sum = 0.f;
    for (int j = 0; j < w; j++) {
        const float e = std::exp(scores[j] - m_new);
        scores[j] = e;
        tile_sum += e;
    }
    const float l_new = l_old * corr + tile_sum;
    for (int j = 0; j < w; j++)
        scores[j] /= l_new;
    m = m_new;
    l = l_new;
    old_coef = (l_old * corr) / l_new;
}

// Scalar reference for one accumulator renormalization row, matching one row of
// build_acc_renorm_ir.
void ref_acc_renorm(
        std::vector<float> &acc, const std::vector<float> &pv, float old_coef) {
    for (size_t d = 0; d < acc.size(); d++)
        acc[d] = old_coef * acc[d] + pv[d];
}

// Validates the online-softmax tile epilogue end to end: the JIT builder's row
// update must match the scalar reference across several tile widths. This
// proves the whole builder -> alloc -> emit -> run pipeline computes the real
// softmax math. The eltwise exp is a polynomial approximation, so
// denominator-dependent outputs use a relative tolerance; m_new is a plain max
// and stays exact.
TEST(SdpFusedSoftmaxIr, SoftmaxOnlineTileRow) {
    SKIP_IF_NO_AVX2();

    // Widths span pure tails (< simd_w), exact multiples, and multiples plus a
    // ragged tail, so the masked-tail path and its lane neutralization run.
    for (int w : {1, 5, 7, simd_w, 9, 15, 2 * simd_w, 17, 23, 4 * simd_w - 1,
                 4 * simd_w}) {
        softmax_ir_kernel_t kernel(build_softmax_tile_ir(1, w));
        ASSERT_EQ(kernel.create_kernel(), status::success) << "w=" << w;

        std::vector<float> scores(w), ref(w);
        for (int j = 0; j < w; j++) {
            scores[j] = (float)((j * 5) % 13) * 0.5f - 3.f;
            ref[j] = scores[j];
        }
        const float scale = 0.125f;
        // Finite running state (a later, non-first KV tile).
        float m = -1.5f, l = 2.0f;
        float ref_m = m, ref_l = l, ref_oc = 0.f;
        ref_softmax_row(ref, scale, ref_m, ref_l, ref_oc);

        float oc = -12345.f;
        softmax_row_args_t args {scores.data(), &scale, &m, &l, &oc};
        kernel(&args);

        EXPECT_NEAR(m, ref_m, 1e-5f * std::abs(ref_m) + 1e-6f) << "w=" << w;
        EXPECT_NEAR(l, ref_l, 1e-4f * std::abs(ref_l) + 1e-6f) << "w=" << w;
        EXPECT_NEAR(oc, ref_oc, 1e-4f * std::abs(ref_oc) + 1e-6f) << "w=" << w;
        for (int j = 0; j < w; j++)
            EXPECT_NEAR(scores[j], ref[j], 1e-4f * std::abs(ref[j]) + 1e-6f)
                    << "w=" << w << " j=" << j;
    }
}

// Validates the first-tile path (m_old == -inf, l_old == 0) and the running
// state carried across tiles: one per-row state is threaded through two
// successive KV tiles, matching two scalar-reference updates. The first tile
// exercises the -inf seed (the max reduction and exp both saturate correctly so
// corr == 0); the second tile then consumes the finite state it produced.
TEST(SdpFusedSoftmaxIr, SoftmaxOnlineFirstTile) {
    SKIP_IF_NO_AVX2();

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int w : {1, 5, 7, simd_w, 9, 15, 2 * simd_w, 17, 23, 4 * simd_w - 1,
                 4 * simd_w}) {
        softmax_ir_kernel_t kernel(build_softmax_tile_ir(1, w));
        ASSERT_EQ(kernel.create_kernel(), status::success) << "w=" << w;

        const float scale = 0.125f;
        // Two KV tiles of `w` scores, threaded through one running state that
        // starts at the first-tile seed.
        float m = neg_inf, l = 0.f, oc = -12345.f;
        float ref_m = neg_inf, ref_l = 0.f, ref_oc = -12345.f;

        for (int tile = 0; tile < 2; tile++) {
            std::vector<float> scores(w), ref(w);
            for (int j = 0; j < w; j++) {
                scores[j] = (float)((j * 7 + tile * 3) % 13) * 0.5f - 3.f;
                ref[j] = scores[j];
            }
            ref_softmax_row(ref, scale, ref_m, ref_l, ref_oc);

            softmax_row_args_t args {scores.data(), &scale, &m, &l, &oc};
            kernel(&args);

            EXPECT_NEAR(m, ref_m, 1e-5f * std::abs(ref_m) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            EXPECT_NEAR(l, ref_l, 1e-4f * std::abs(ref_l) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            EXPECT_NEAR(oc, ref_oc, 1e-4f * std::abs(ref_oc) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            for (int j = 0; j < w; j++)
                EXPECT_NEAR(scores[j], ref[j], 1e-4f * std::abs(ref[j]) + 1e-6f)
                        << "w=" << w << " tile=" << tile << " j=" << j;
        }
    }
}

// Validates the multi-row tile epilogue: one kernel processes seq_q score rows
// with a runtime loop, advancing the score and per-row state pointers each
// iteration. Every row carries its own finite running state and distinct data,
// so a wrong stride would bleed rows into each other. Each row must match an
// independent scalar-reference update.
TEST(SdpFusedSoftmaxIr, SoftmaxOnlineTileMultiRow) {
    SKIP_IF_NO_AVX2();

    const float scale = 0.125f;
    for (int seq_q : {2, 3, 5}) {
        for (int w : {1, 7, simd_w, 9, 17, 4 * simd_w - 1, 4 * simd_w}) {
            softmax_ir_kernel_t kernel(build_softmax_tile_ir(seq_q, w));
            ASSERT_EQ(kernel.create_kernel(), status::success)
                    << "seq_q=" << seq_q << " w=" << w;

            std::vector<float> scores((size_t)seq_q * w);
            std::vector<float> m(seq_q), l(seq_q), oc(seq_q, -12345.f);
            std::vector<std::vector<float>> ref(seq_q, std::vector<float>(w));
            std::vector<float> ref_m(seq_q), ref_l(seq_q), ref_oc(seq_q);
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < w; j++) {
                    const float v
                            = (float)(((i + 1) * j * 5 + i * 3) % 13) * 0.5f
                            - 3.f;
                    scores[(size_t)i * w + j] = v;
                    ref[i][j] = v;
                }
                // Distinct finite per-row running state.
                m[i] = -1.5f + 0.25f * i;
                l[i] = 2.0f + 0.5f * i;
                ref_m[i] = m[i];
                ref_l[i] = l[i];
                ref_oc[i] = -12345.f;
                ref_softmax_row(ref[i], scale, ref_m[i], ref_l[i], ref_oc[i]);
            }

            softmax_row_args_t args {
                    scores.data(), &scale, m.data(), l.data(), oc.data()};
            kernel(&args);

            for (int i = 0; i < seq_q; i++) {
                EXPECT_NEAR(m[i], ref_m[i], 1e-5f * std::abs(ref_m[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                EXPECT_NEAR(l[i], ref_l[i], 1e-4f * std::abs(ref_l[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                EXPECT_NEAR(
                        oc[i], ref_oc[i], 1e-4f * std::abs(ref_oc[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                for (int j = 0; j < w; j++)
                    EXPECT_NEAR(scores[(size_t)i * w + j], ref[i][j],
                            1e-4f * std::abs(ref[i][j]) + 1e-6f)
                            << "seq_q=" << seq_q << " w=" << w << " i=" << i
                            << " j=" << j;
            }
        }
    }
}

// Validates the select mask fused into pass 1 of the softmax tile epilogue:
// uint8 condition bytes choose between the scaled score and the fill scalar
// before the running max/denominator update, in both the fusiable (keep where
// cond != 0) and non-fusiable (keep where cond == 0) senses. Each row runs the
// full scale -> select -> softmax chain against an independent scalar reference
// over widths that exercise the ragged tail. The running state is finite (a
// later KV tile) so even a fully masked row keeps l_new > 0.
TEST(SdpFusedSoftmaxIr, SoftmaxOnlineTileSelect) {
    SKIP_IF_NO_AVX2();

    const float scale = 0.125f;
    const float fill = -30.f;
    for (bool fusiable : {false, true}) {
        for (int seq_q : {1, 2, 3}) {
            for (int w : {1, 5, 7, simd_w, 9, 17, 4 * simd_w - 1, 4 * simd_w}) {
                softmax_ir_kernel_t kernel(
                        build_softmax_tile_ir(seq_q, w, true, fusiable));
                ASSERT_EQ(kernel.create_kernel(), status::success)
                        << "fusiable=" << fusiable << " seq_q=" << seq_q
                        << " w=" << w;

                std::vector<float> scores((size_t)seq_q * w);
                std::vector<uint8_t> cond((size_t)seq_q * w);
                std::vector<float> m(seq_q), l(seq_q), oc(seq_q, -12345.f);
                std::vector<std::vector<float>> ref(
                        seq_q, std::vector<float>(w));
                std::vector<std::vector<uint8_t>> refc(
                        seq_q, std::vector<uint8_t>(w));
                std::vector<float> ref_m(seq_q), ref_l(seq_q), ref_oc(seq_q);
                for (int i = 0; i < seq_q; i++) {
                    for (int j = 0; j < w; j++) {
                        const float v
                                = (float)(((i + 1) * j * 5 + i * 3) % 13) * 0.5f
                                - 3.f;
                        scores[(size_t)i * w + j] = v;
                        ref[i][j] = v;
                        // Mix zeros and nonzero bytes, some > 127.
                        const uint8_t c = ((i + j) % 2 == 0)
                                ? 0
                                : (uint8_t)(50 + (i * 7 + j * 13) % 200);
                        cond[(size_t)i * w + j] = c;
                        refc[i][j] = c;
                    }
                    // Distinct finite per-row running state (a later KV tile).
                    m[i] = -1.5f + 0.25f * i;
                    l[i] = 2.0f + 0.5f * i;
                    ref_m[i] = m[i];
                    ref_l[i] = l[i];
                    ref_oc[i] = -12345.f;
                    ref_softmax_row_masked(ref[i], scale, refc[i], fill,
                            fusiable, ref_m[i], ref_l[i], ref_oc[i]);
                }

                softmax_row_args_t args {scores.data(), &scale, m.data(),
                        l.data(), oc.data(), cond.data(), &fill};
                kernel(&args);

                for (int i = 0; i < seq_q; i++) {
                    EXPECT_NEAR(
                            m[i], ref_m[i], 1e-5f * std::abs(ref_m[i]) + 1e-6f)
                            << "fusiable=" << fusiable << " seq_q=" << seq_q
                            << " w=" << w << " i=" << i;
                    EXPECT_NEAR(
                            l[i], ref_l[i], 1e-4f * std::abs(ref_l[i]) + 1e-6f)
                            << "fusiable=" << fusiable << " seq_q=" << seq_q
                            << " w=" << w << " i=" << i;
                    EXPECT_NEAR(oc[i], ref_oc[i],
                            1e-4f * std::abs(ref_oc[i]) + 1e-6f)
                            << "fusiable=" << fusiable << " seq_q=" << seq_q
                            << " w=" << w << " i=" << i;
                    for (int j = 0; j < w; j++)
                        EXPECT_NEAR(scores[(size_t)i * w + j], ref[i][j],
                                1e-4f * std::abs(ref[i][j]) + 1e-6f)
                                << "fusiable=" << fusiable << " seq_q=" << seq_q
                                << " w=" << w << " i=" << i << " j=" << j;
                }
            }
        }
    }
}

// Validates the accumulator renormalization tile epilogue: one kernel rescales
// seq_q accumulator rows (acc = old_coef*acc + pv) with a runtime loop,
// advancing the acc/pv/old_coef pointers each iteration. Every row carries a
// distinct old_coef and distinct data, so a wrong stride would bleed rows into
// each other. Each row must match an independent scalar-reference update.
TEST(SdpFusedSoftmaxIr, AccRenormTile) {
    SKIP_IF_NO_AVX2();

    for (int seq_q : {1, 2, 3, 5}) {
        // Head sizes span pure tails, exact multiples, and multiples plus a
        // ragged tail, so the masked-tail path runs.
        for (int hs : {1, 7, simd_w, 9, 17, 4 * simd_w - 1, 4 * simd_w}) {
            softmax_ir_kernel_t kernel(build_acc_renorm_ir(seq_q, hs));
            ASSERT_EQ(kernel.create_kernel(), status::success)
                    << "seq_q=" << seq_q << " hs=" << hs;

            std::vector<float> acc((size_t)seq_q * hs);
            std::vector<float> pv((size_t)seq_q * hs);
            std::vector<float> oc(seq_q);
            std::vector<std::vector<float>> ref(seq_q);
            for (int i = 0; i < seq_q; i++) {
                ref[i].resize(hs);
                for (int j = 0; j < hs; j++) {
                    const float a
                            = (float)(((i + 1) * j * 3 + i * 2) % 11) * 0.5f
                            - 2.f;
                    const float p
                            = (float)(((i + 2) * j * 7 + i) % 13) * 0.25f - 1.f;
                    acc[(size_t)i * hs + j] = a;
                    pv[(size_t)i * hs + j] = p;
                    ref[i][j] = a;
                }
                // Distinct per-row renorm coefficient.
                oc[i] = 0.3f + 0.2f * i;
                std::vector<float> pv_row(pv.begin() + (size_t)i * hs,
                        pv.begin() + (size_t)(i + 1) * hs);
                ref_acc_renorm(ref[i], pv_row, oc[i]);
            }

            acc_renorm_args_t args {acc.data(), pv.data(), oc.data()};
            kernel(&args);

            for (int i = 0; i < seq_q; i++)
                for (int j = 0; j < hs; j++)
                    EXPECT_NEAR(acc[(size_t)i * hs + j], ref[i][j],
                            1e-5f * std::abs(ref[i][j]) + 1e-6f)
                            << "seq_q=" << seq_q << " hs=" << hs << " i=" << i
                            << " j=" << j;
        }
    }
}

} // namespace dnnl
