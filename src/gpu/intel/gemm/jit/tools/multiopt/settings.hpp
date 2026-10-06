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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_SETTINGS_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_SETTINGS_HPP

namespace multiopt {

struct Settings {
    static constexpr int defaultAgents = 8;                            // Default agent count.
    static constexpr int minSensitivitySamples = 5;                    // Minimum number of samples per dimension for calculating sensitivity.
    static constexpr float sensitivityCalcMultiplier = 1.2f;           // Timing constant for deciding when to update sensitivities.
    static constexpr double sensitivityRegularization = 1.0e-6;        // Ridge penalty for permanently coupled strategy dimensions.
    static constexpr int initialSVsPerPV = 10;                         // # of random legal SVs to benchmark for a new PV.
    static constexpr int seedGiveUpMultiplier = 5;                     // Give up seeding a PV after this many times initialSVsPerPV attempts.
    static constexpr double pollInterval = 0.05;                       // Re-check granularity (s) while waiting for a stalled generation.
    static constexpr int spreadEvery = 10000;                          // # of main loops between spreading steps.
    static constexpr int newPVEvery = 100;                             // # of main loops between introducing new PVs.
    static constexpr int maxSpreadSVs = 20;                            // Maximum number of SVs to spread per spread step.
    static constexpr int spreadBatchSize = 16;                         // Max # of PVs per spreading benchmark request.
    static constexpr int spreadNonadjacentBatch = 10;                  // Batch size for nonadjacent PVs to test when spreading an SV.
    static constexpr int maxUnsuccessfulNonadjacent = 15;              // During spreading, bail after this many failing nonadjacent searches.
    static constexpr float nstddevTopSV = 1.0f;                        // # of std deviations to use as tolerance for checking if SV might be optimal for a PV.
    static constexpr float agentSuboptimalityStepImprovement = 0.2f;   // Suboptimality improvement target when stepping.
    static constexpr float agentNeighborCheckThreshold = 0.95f;        // Only check neighboring PVs if within this fraction of best performance so far.
    static constexpr float agentCarefulCheckThreshold = 0.9f;          // Only carefully benchmark if within this fraction of best performance so far.
    static constexpr float agentProbabilitySwitchPVOnFailure = 0.01f;  // Probability of changing target PV after failed step (no kernel generation).
    static constexpr float agentSwitchPVMinProbability = 0.05f;        // Minimum probability of changing target PV after successful step.
    static constexpr float agentSwitchPVSuboptDecay = 2.0f;            // Decay constant for probability of changing target PV.
    static constexpr float agentAcceptSVSuboptDecay = 8.0f;            // Decay constant for probability of accepting target SV if it is not as good as previous SV.
    static constexpr float agentNewPVMinProbability = 0.25f;           // Minimum probability of accepting a new target PV.
    static constexpr float agentNewPVWeight = 2.0f;                    // Weight for selecting suboptimal target PVs. Higher values more heavily prefer suboptimal PVs.
    static constexpr float agentChooseTopSVSuboptDecay = 4.0f;         // Decay constant for probability of accepting the current best SV when targeting a new PV.
    static constexpr float agentProbTestUnknownSensitivity = 0.75f;    // Probability for agent to preferentially get more sensitivity if some sensitivity is unknown.
    static constexpr float agentRandomStepThresh = 4.0f;               // Suboptimality threshold above which agents can make random moves.
    static constexpr int agentMaxFailedAccepts = 2;                    // # of failed candidate steps before re-targeting.
    static constexpr float atticStart = 1.0f;                          // Starting value for attic space.
    static constexpr float atticMin = 0.1f;                            // Minimum value for attic space.
    static constexpr float atticDecay = 2000.0f;                       // Decay constant for attic space.
    static constexpr float extractorDefaultMinRelEff = 0.95f;          // Default minimum relative efficiency for extractor.
};

} /* namespace multiopt */

#endif
