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

#include "benchstone.hpp"
#include "benchstone_common.hpp"
#include "kernel_info.hpp"
#include "gemmstone/type.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>

static TestResult accuracyTestImpl(benchstone::Driver &bench, KernelInfo &info, bool doCrossingCheck)
{
    TestResult result;
    bool unchecked = false;

    // Some set sizes.
    int mt[2] = {13, 767};
    int nt[2] = {9, 715};
    int kt[2] = {7, 623};
    int bt[2] = {5, 2};

    if (bench.isSimulation() && info.isL3()) {
        // Reduce large size for level-3 kernels in simulation.
        mt[1] = 191;
        nt[1] = 143;
        kt[1] = 51;
    }

    if (info.isL3() && info.l3[0].triangular && info.l3[0].isHilbert())
        for (int t = 0; t < 2; t++)
            nt[t] = mt[t];

    auto addrOffsets = info.getAddrOffsets();

    // Do overrun/remainder checking.
    bench.OneFill = true;
    int mb, nb, kb = info.kbAlign();
    bool kRemHandling = info.kRemHandling();

    if (info.isL3()) {
        mb = info.l3[0].unroll[LoopM];
        nb = info.l3[0].unroll[LoopN];
    } else {
        mb = info.copy[0].unroll[LoopM];
        nb = info.copy[0].unroll[LoopN];
        if (!info.copy[0].inner())
            std::swap(mb, nb);
    }

    int emb = std::max(mb, 2);
    int enb = std::max(nb, 2);

    int mm_max = info.isCopy() ? emb : (emb - 1);
    int nn_max = info.isCopy() ? enb : (enb - 1);

    auto padW = [](char layout, int32_t &r, int32_t &c, Type T) {
        auto &w = (layout == 'T' ? c : r);
        if (w < 64 / T) w += 64 / T;
    };

    auto runTest = [&](uint32_t b, int32_t m, int32_t n, int32_t k,
                       int32_t lda = -1, int32_t ldb = -1, int32_t ldc = -1,
                       bool conja = false, bool conjb = false, bool unit = false,
                       char cOffType = 'F', bool aOffset = false, bool bOffset = false)
    {
        /* Increase sizes if needed to avoid simulator errors with small block 2D widths */
        if (bench.isSimulation() && info.isL3()) {
            auto l3Info = info.l3[0];
            if (l3Info.a2D) padW(l3Info.layoutA, m, k, l3Info.typeA());
            if (l3Info.b2D) padW(l3Info.layoutB, k, n, l3Info.typeB());
            if (l3Info.c2D) padW(l3Info.layoutC, m, n, l3Info.typeC());
        }
        if (getenv("ACC_TRACE")) {
            fprintf(stderr, "[ACC] b=%u m=%d n=%d k=%d lda=%d ldb=%d ldc=%d "
                            "conja=%d conjb=%d unit=%d cOff=%c aOff=%d bOff=%d\n",
                    b, m, n, k, lda, ldb, ldc, int(conja), int(conjb), int(unit),
                    cOffType, int(aOffset), int(bOffset));
            fflush(stderr);
        }
        result = bench.test(info, TestMode::Accuracy, b, m, n, k, lda, ldb, ldc,
                            conja, conjb, unit, cOffType, aOffset, bOffset);
        unchecked |= (result.accuracy == TestAccuracy::Unchecked);
    };

#define RUN_TEST(params, msg) \
    runTest params; \
    if (result.failed()) { \
        if (result.failingCase.empty()) result.failingCase = qsprintf msg; \
        return result; \
    }

    // m/n remainder tests.
    for (int mm = 1; mm <= mm_max; mm++) {
        for (int nn = 1; nn <= nn_max; nn++) {
            if (nn < (enb - 1) && mm < (emb - 1)) continue;

            auto kk = kb * 4;
            RUN_TEST((bt[0], mm, nn, kk),
                     ("m,n,k = %d,%d,%d", mm, nn, kk));
        }
    }

    // k remainder tests.
    if (info.isL3() && !info.l3[0].trxm() && kRemHandling) {
        int mm = emb, nn = enb;
        for (int kk = 1; kk <= info.l3[0].unroll[LoopK]; kk++) {
            RUN_TEST((bt[0], mm, nn, kk),
                     ("m,n,k = %d,%d,%d", mm, nn, kk));
        }

        mm = info.l3[0].wgTile(LoopM);
        nn = info.l3[0].wgTile(LoopN);

        for (int kk: {1, info.l3[0].unroll[LoopK] - 1}) {
            RUN_TEST((bt[0], mm, nn, kk),
                     ("m,n,k = %d,%d,%d", mm, nn, kk));
        }
    }

    // Random data, single tile.
    {
        bench.OneFill = false;
        int mm = emb - 1, nn = enb - 1, kk = kb * 4;
        RUN_TEST((bt[0], mm, nn, kk),
                 ("m,n,k = %d,%d,%d", mm, nn, kk));
    }

    // Random data, random sizes.
    if (!kRemHandling) for (int t = 0; t < 2; t++)
        kt[t] = align_up(kt[t], kb);

    for (bool beta0 : {false, true}) {
        if (beta0 && (info.isCopy() || info.l3[0].fixedBeta()))
            continue;
        bench.TestBeta0 = beta0;
        bench.TestNonKParallelGlobal = bench.TestBeta0 && info.l3[0].kParallel() && !info.l3[0].fusedBeta();
        for (bool onefill : {true, false}) {
            bench.OneFill = onefill;
            for (int t = 0; t < 2; t++) {
                RUN_TEST((bt[t], mt[t], nt[t], kt[t]),
                         ("%s%s m,n,k = %d,%d,%d", bench.OneFill ? "fixed data" : bench.TestBeta0 ? "random A/B, nan C" : "random data",
                                                   bench.TestBeta0 ? ", beta = 0," : "",
                                                   mt[t], nt[t], kt[t]));
            }
        }
    }

    bench.TestNonKParallelGlobal = bench.TestBeta0 = false;

    // Test conjugation and unit diagonals (trsm3).
    bench.OneFill = false;
    if (info.isL3()) {
        for (bool unit : {false, true}) {
            for (bool conja : {false, true}) {
                for (bool conjb : {false, true}) {
                    if ((conja || conjb) && !info.canConjAB(conja, conjb)) continue;
                    if (unit && !info.configurableUnit()) continue;
                    RUN_TEST((bt[0], mt[0], nt[0], kt[0], -1, -1, -1, conja, conjb, unit),
                             ("random data, conja = %d, conjb = %d, unit = %d; m,n,k = %d,%d,%d",
                                int(conja), int(conjb), int(unit), mt[0], nt[0], kt[0]));
                }
            }
        }
    }

    // Test in-place mode for out-of-place trsm.
    if (info.isL3() && info.l3[0].trsm3() && !info.l3[0].strongReadCheck) {
        bench.TestInPlace = true;
        for (int t = 0; t < 2; t++) {
            RUN_TEST((bt[t], mt[t], nt[t], kt[t]),
                     ("in place m,n,k = %d,%d,%d", mt[t], nt[t], kt[t]));
        }
        bench.TestInPlace = false;
    }

    // Test non-k-parallel mode for global k-parallel kernels (which allows beta).
    if (info.isL3() && info.l3[0].kParallel() && !info.l3[0].fusedBeta()) {
        bench.TestNonKParallelGlobal = true;
        for (int t = 0; t < 2; t++) {
            RUN_TEST((bt[t], mt[t], nt[t], kt[t]),
                     ("non k-parallel m,n,k = %d,%d,%d", mt[t], nt[t], kt[t]));
        }
        bench.TestNonKParallelGlobal = false;
    }

    // Test blocking for diag copy kernels to check diag handling.
    if (info.isCopy() && info.copy[0].nongeneral()) {
        auto OldNBlockKOverride = bench.NBlockOverrides[2];
        bench.NBlockOverrides[2] = 4;

        int mt[2] = {256, 767};
        int nt[2] = {256, 715};
        int kt[2] = {256, 623};

        for (bool onefill : {false, true}) {
            bench.OneFill = onefill;
            for (int t = 0; t < 2; t++) {
                RUN_TEST((bt[t], mt[t], nt[t], kt[t]),
                         ("%s with blocking m,n = %d,%d", bench.OneFill ? "fixed data" : "random data",
                                                          mt[t], nt[t]));
            }
        }

        bench.NBlockOverrides[2] = OldNBlockKOverride;
    }

    // Do odd leading dimension/offset/stride tests to check for alignment issues.
    // Do this with ones for HGEMM since the tolerances are so loose.
    if (info.isL3() && (info.l3[0].precisionC == 'H' || info.l3[0].precisionC == 'Q'))
        bench.OneFill = true;

    for (int dim = 0; dim < 3; dim++) {
        int bb = bt[1], mm = mt[1], nn = nt[1], kk = kb * 4;
        if (kRemHandling) kk += 13;

        bench.OddLD[dim] = true;
        RUN_TEST((bb, mm, nn, kk),
                 ("m,n,k = %d,%d,%d] with odd LD in dim %d", mm, nn, kk, dim));
        bench.OddLD[dim] = false;

        bool doOffset = true;
        if (info.isL3() && dim == 0) doOffset = !info.l3[0].copyA();
        if (info.isL3() && dim == 1) doOffset = !info.l3[0].copyB();
        if (info.isL3() && dim == 2) doOffset = !info.l3[0].copyC();

        if (doOffset) {
            bench.AddrOffsetOverride[dim] = addrOffsets[dim];
            RUN_TEST((bb, mm, nn, kk),
                     ("m,n,k = %d,%d,%d] with odd %c offset", mm, nn, kk, 'A' + dim));
            bench.AddrOffsetOverride[dim] = 0;
        }

        if (info.isBatched()) {
            bench.OddStride[dim] = true;
            RUN_TEST((bb, mm, nn, kk),
                     ("m,n,k = %d,%d,%d] with odd stride in dim %d", mm, nn, kk, dim));
            bench.OddStride[dim] = false;
        }
    }
    bench.OneFill = false;

    // Large ldc test to check for overflows.
    if (!bench.isSimulation()) {
        int kt = 64;
        if (!kRemHandling)
            kt = align_up(kt, kb);
        RUN_TEST((8, 64, 64, kt, 64, 64, 10000),
                 ("[large ldc]"));
    }

    // Test A/B offsets.
    if (info.isL3() && (info.l3[0].aOffset || info.l3[0].bOffset)) {
        for (int t = 0; t < 2; t++) {
            for (bool oddLDs : {false, true}) {
                for (bool aOffset : {false, true}) {
                    for (bool bOffset : {false, true}) {
                        if (!aOffset && !bOffset) continue;
                        if (aOffset && !info.l3[0].aOffset) continue;
                        if (bOffset && !info.l3[0].bOffset) continue;
                        bench.OddLD[0] = bench.OddLD[1] = oddLDs;
                        for (bool onefill : {true, false}) {
                            bench.OneFill = onefill;
                            RUN_TEST((bt[t], mt[t], nt[t], kt[t], -1, -1, -1, false, false, false, 'F', aOffset, bOffset),
                                     ("%s w/ %c%c offset; m,n,k = %d,%d,%d%s",
                                                              bench.OneFill ? "fixed data" : "random data",
                                                              aOffset ? 'A' : ' ', bOffset ? 'B' : ' ',
                                                              mt[t], nt[t], kt[t],
                                                              oddLDs ? ", odd lda/b" : ""));
                        }
                    }
                }
            }
        }
        bench.OddLD[0] = bench.OddLD[1] = false;
    }

    // Test C offsets.
    if (info.canCOffset()) {
        for (int t = 0; t < 2; t++) {
            for (bool abOffset : {true, false}) {
                bool aOffset = abOffset && info.l3[0].aOffset;
                bool bOffset = abOffset && info.l3[0].bOffset;
                if (abOffset && !aOffset && !bOffset) continue;
                for (char cOffType : {'F', 'C', 'R', 'M'}) {
                    if (cOffType == 'M' && !info.canCMatrixOffset()) continue;
                    for (bool onefill : {true, false}) {
                        bench.OneFill = onefill;
                        auto cOffString = (cOffType == 'M') ? "matrix" :
                                          (cOffType == 'C') ? "column" :
                                          (cOffType == 'R') ? "row" : "fixed";
                        auto abOffString = abOffset ? "A/B offsets, " : "";

                        RUN_TEST((bt[t], mt[t], nt[t], kt[t], -1, -1, -1, false, false, false, cOffType, aOffset, bOffset),
                                 ("%s w/ %sC %s offset; m,n,k = %d,%d,%d",
                                          bench.OneFill ? "fixed data" : "random data",
                                          abOffString, cOffString, mt[t], nt[t], kt[t]));
                    }
                }
            }
        }
    }

    // Test transa/transb combinations.
    if (info.isL3() && info.l3[0].fused && !bench.DoTransposeOverride) {
        bench.DoTransposeOverride = true;
        for (int t = 0; t < 2; t++) {
            for (bool transa : {false, true}) {
                for (bool transb : {false, true}) {
                    auto m = (mt[t] ^ 0x2A) + 1;
                    auto n = (nt[t] ^ 0x1B) + 1;
                    auto k = (kt[t] ^ 0x9) + 2;
                    if (!kRemHandling)
                        k = align_up(k, kb);
                    bench.TransposeOverride[0] = transa;
                    bench.TransposeOverride[1] = transb;
                    RUN_TEST((bt[t], m, n, k),
                             ("%c%c; m,n,k = %d,%d,%d", "NT"[transa], "NT"[transb], m, n, k));
                }
            }
        }
        bench.DoTransposeOverride = false;
    }

    // Test 4GB boundary crossings -- L0 only.
    #define CROSSING_TEST(flag, name) \
        flag = BUF_CROSS_4GB; \
        RUN_TEST((bt[1], mt[1], nt[1], kt[1]), \
                (name " crosses 4GB boundary")); \
        flag = 0;

    if ((bench.runtime() == benchstone::RuntimeBackend::LevelZero) && doCrossingCheck && !info.readCheck()) {
        bench.OneFill = true;
        if (info.isL3()) {
            CROSSING_TEST(bench.ExtraFlagsA, "A");
            CROSSING_TEST(bench.ExtraFlagsB, "B");
            CROSSING_TEST(bench.ExtraFlagsC, "C");
        } else {
            CROSSING_TEST(bench.ExtraFlagsS, "S");
            CROSSING_TEST(bench.ExtraFlagsD, "D");
        }
        bench.OneFill = false;
    }

#undef RUN_TEST
#undef CROSSING_TEST

    result.accuracy = unchecked ? TestAccuracy::Unchecked : TestAccuracy::OK;
    return result;
}

namespace benchstone {

TestResult Driver::runAccuracyTests(KernelInfo &info)
{
    bool savedQuiet = Quiet;
    Quiet = true;

    bool doCrossingCheck = false;
    if (auto env = getenv("CHECK_CROSS_4GB"))
        doCrossingCheck = atoi(env) != 0;

    auto result = accuracyTestImpl(*this, info, doCrossingCheck);
    Quiet = savedQuiet;
    return result;
}

} // namespace benchstone
