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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_STRATEGY_PARSER_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_STRATEGY_PARSER_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"

#include <string>

GEMMSTONE_NAMESPACE_START

void parseStrategy(const std::string &str, const GEMMProblem &problem, GEMMStrategy &strategy);
// Deprecated: use the overload without hw (deduced from problem.product).
void parseStrategy(const std::string &str, ngen::HW hw, const GEMMProblem &problem, GEMMStrategy &strategy);

void adjustStrategy(ngen::HW hw, const GEMMProblem &problem, GEMMStrategy &strategy, const char *tags = nullptr);

const char *parseLayout(const char *s, MatrixAddressing &atype);
const char *parsePrecision(const char *s, Type &precision);
const char *parsePrecisions(const char *s, Type &precision1, Type &precision2);

std::string unparseStrategy(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
