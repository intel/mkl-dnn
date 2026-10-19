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


#include "hw_utils.hpp"
#include "remask.hpp"

GEMMSTONE_NAMESPACE_START

static bool needsRemask(Type T, bool column, const RegisterBlock &block,
                        const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks = false)
{
    if (!ignoreMasks)
        if (column ? !block.remainderC : !block.remainderR)
            return false;

    bool block2DRemask = isBlock2D(astrategy.accessType)
                      && ((block.colMajor ^ isTransposing(astrategy.accessType)) != column);

    int maskGranularity = block.ebytes;
    if (block.ebytes >= 16)
        maskGranularity = 4;
    if (block2DRemask)
        maskGranularity = std::max(maskGranularity, block2DWidthAlignment(T, block, atype, astrategy));
    if (ignoreMasks && !(block2DRemask && astrategy.address2D))
        maskGranularity = 256;

    return (T.paddedSize() < maskGranularity);
}

bool needsRemask(Type T, bool column, const RegisterLayout &layout,
                 const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks)
{
    for (auto &block: layout)
        if (needsRemask(T, column, block, atype, astrategy, ignoreMasks))
            return true;
    return false;
}

GEMMSTONE_NAMESPACE_END
