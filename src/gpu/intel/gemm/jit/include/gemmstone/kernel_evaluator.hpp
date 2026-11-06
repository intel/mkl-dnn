/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_EVALUATOR_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_EVALUATOR_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/strategy.hpp"

GEMMSTONE_NAMESPACE_START

struct SizeParams {
    int64_t batch = 0;
    int64_t m = 0, n = 0, k = 0;
};

struct EvaluateParams {
    SizeParams sizes;
    double alpha, beta;
    int euCount;
    ngen::Product product;
    int tileCount = 1;
    bool effective = false;
    bool cConvert = false;
    bool postOps = false;
    bool batch = false;
    bool deterministic = false;
    bool noCAtomics = false;
    int unrollRound[3] = {1, 1, 1}; // Unroll round-up applied after selection
    Type Tc_ext = Type::invalid;    // External C element type; Type::invalid = use catalog precision
};

struct DerivedEvaluateParams : public EvaluateParams {
    int64_t wgCountM, wgCountN, wgCountK;
    int64_t mPad, nPad;
    double threadCount;
    int threadsPerEU;
    int hwThreadCapacity;
    int hwThreadsPartialWave;
    int partialWaveCount;
    bool autoatomic;
};

struct EvaluateAuxOutput {
    int64_t k0 = 0;
    int wgK = 1;
    bool kParallel = false;
    bool kParallelVariable = false;
    bool disableAtomics = false;
};

DerivedEvaluateParams getDerivedParams(const kcatalog::Entry &e, const EvaluateParams &p);
double evaluate(const kcatalog::Entry &e, const EvaluateParams &p, EvaluateAuxOutput &aux);
double evaluate(const kcatalog::Entry &e, const DerivedEvaluateParams &p, EvaluateAuxOutput &aux);

void modifyStrategy(GEMMStrategy &strategy, const EvaluateAuxOutput &aux);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
