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


#include "gemmstone/generator.hpp"
#include "kernel_queries.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


// Get driver information from this strategy.
template <HW hw>
CommonDriverInfo Generator<hw>::driverInfo(GEMMProblem problem, const GEMMStrategy &strategy)
{
    CommonDriverInfo info;

    problem.autoTypeConversions(strategy.systolic);

    info.subgroupSize = strategy.subgroupSize;
    info.fusedLoop = strategy.fused ? strategy.fusedLoop : LoopNone;
    info.grfCount = strategy.GRFs;
    for (int d = 0; d < 3; d++) {
        info.loopOrder[d] = strategy.loopOrder[d];
        info.blocking[d] = strategy.blocking[d];
        info.blockingAlt[d] = strategy.blockingAlt[d];
        info.unroll[d] = strategy.unroll[d];
        info.wg[d] = strategy.wg[d];
    }
    info.unroll[LoopK] = strategy.kAlign(problem);
    info.wgExpand = (strategy.splitCopy ? 2 : 1) * strategy.wgPadFactor;
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN) ? LoopMNLinearNMK : LoopMNLinearMNK;
        info.loopOrder[1] = LoopNone;
    } else if (strategy.cWalkOrder == WalkOrder::NestedLinear) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN) ? LoopMNNestedLinearNMK : LoopMNNestedLinearMNK;
        info.loopOrder[1] = LoopNone;
    } else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN) ? LoopMNHilbertNMK : LoopMNHilbertMNK;
        info.loopOrder[1] = LoopNone;
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        info.loopOrder[0] = (info.loopOrder[0] == LoopN) ? LoopMNBoustrophedonNMK : LoopMNBoustrophedonMNK;
        info.loopOrder[1] = LoopNone;
    }
    if (strategy.persistent)
        info.loopOrder[0] = static_cast<LoopType>(info.loopOrder[0] | LoopPersistent);
    if (problem.batch == BatchMode::None && !strategy.kParallelLocal)
        info.loopOrder[2] = LoopNone;
    info.wgUpdate = strategy.getWGType(problem);
    info.flags = 0;
    if (strategy.remHandling[LoopK] != RemainderHandling::Ignore) info.flags |= FlagKRemainderHandling;
    if (strategy.kParallel)                                       info.flags |= FlagKParallel;
    if (strategy.kParallelLocal)                                  info.flags |= FlagKParallelLocal;
    if (strategy.kParallelVariable)                               info.flags |= FlagKParallelVariable;
    if (strategy.fuseBeta)                                        info.flags |= FlagFusedBeta;
    if (strategy.fuseBeta && strategy.altFusedBeta)               info.flags |= FlagAltFusedBeta;
    if (strategy.fusePostOps)                                     info.flags |= FlagFusedPostOps;
    if (strategy.needsTempC(problem))                             info.flags |= FlagTempC;
    if (strategy.zeroTempC)                                       info.flags |= FlagZeroTempC;
    if (useAutoAtomic(hw, problem, strategy, true))               info.flags |= FlagAutoAtomic;
    if (strategy.shrinkWGK)                                       info.flags |= FlagShrinkWGK;
    if (strategy.fixedWGK())                                      info.flags |= FlagFixedWGK;
    if (problem.alpha.pointer())                                  info.flags |= FlagAlphaPtr;
    if (problem.beta.pointer())                                   info.flags |= FlagBetaPtr;
    if (strategy.nondeterministic(problem))                       info.flags |= FlagNondeterministic;
    if (strategy.scramble[LoopM])                                 info.flags |= FlagScrambleM;
    if (strategy.scramble[LoopN])                                 info.flags |= FlagScrambleN;
    if (strategy.tlbWarmup)                                       info.flags |= FlagExtraWG;
    if (problem.needsSeparateASums())                             info.flags |= FlagSeparateASums;
    if (problem.needsSeparateBSums())                             info.flags |= FlagSeparateBSums;
    info.flags |= (strategy.fillGoal << FlagShiftFillGoal) & FlagMaskFillGoal;
    if (strategy.cInterleave)                                     info.flags |= FlagCInterleave;
    info.slm = int(gemmSLMSize(hw, problem, strategy));
    info.perKSLM = int(gemmPerKSLMSize(hw, problem, strategy));
    info.alignment[0] = problem.A.alignment;
    info.alignment[1] = problem.B.alignment;
    info.alignment[2] = problem.C.alignment;
    info.support4GB[0] = (strategy.A.base.getModel() == ModelA64);
    info.support4GB[1] = (strategy.B.base.getModel() == ModelA64);
    info.support4GB[2] = (strategy.C.base.getModel() == ModelA64);
    if (strategy.kParallel || strategy.kParallelVariable)
        info.blockingAlt[LoopK] = strategy.kPadding;
#ifdef STANDALONE
    info.problem = problem.toString() + ' ' + problem.scalarsToString();
    if (strategy.systolic)
        info.extra = "sys";
    if (strategy.tracking.enabled)
        info.extra += " track";
    if (strategy.strongReadCheck)
        info.extra += " src";
    if (strategy.relaxedAccumulation)
        info.extra += " rx";
    if (!strategy.optAlignAB2D) {
        if (isBlock2D(strategy.A.accessType))   /* rely on automatic downgrading for prefetches */
            info.extra += " a2d";
        if (isBlock2D(strategy.B.accessType))
            info.extra += " b2d";
    }
    if (isBlock2D(strategy.C.accessType) || (strategy.prefetchC && isBlock2D(strategy.C_prefetch.accessType)))
        info.extra += " c2d";
    if (strategy.A.base.getModel() == ModelSLM)
        info.extra += " alocal";
    if (strategy.B.base.getModel() == ModelSLM)
        info.extra += " blocal";
    if (strategy.C.base.getModel() == ModelSLM)
        info.extra += " clocal";
#endif

    return info;
}

#ifdef STANDALONE
#endif /* STANDALONE */

GEMMSTONE_NAMESPACE_END
