/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_REF_HPP
#define GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_REF_HPP

#include "benchstone_matrix.hpp"

namespace ref {

void gemm(superscalar alpha, const Matrix &A, const Matrix &B, superscalar beta, Matrix &C,
          std::array<int, 2> aqGroup, std::array<int, 2> bqGroup, const Matrix &ao, const Matrix &bo, const Matrix &aScale, const Matrix &bScale);

void trsm(bool left, bool lower, bool unit, superscalar alpha, const Matrix &A, Matrix &B);

} /* namespace ref */

#endif
