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

#include "layout_utils.hpp"

using namespace ngen;
using std::vector;

GEMMSTONE_NAMESPACE_START

bool canDequantizeSubByteInt(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
    const RegisterLayout &layoutOffset, const RegisterLayout &layoutScale)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst.type();
    if (!Tsrc.isSubByteInt() || !one_of(Tdst, {Type::f16, Type::f32}))
        return false;

    if (layoutOffset.empty() || layoutScale.empty())
        if (layoutSrc.rows() < layoutDst.rows() || layoutSrc.cols() < layoutDst.cols())
            return false;

    return true;
}

GEMMSTONE_NAMESPACE_END
