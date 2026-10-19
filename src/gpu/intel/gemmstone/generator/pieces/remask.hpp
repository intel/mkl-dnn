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


#ifndef GEMMSTONE_GENERATOR_PIECES_REMASK_HPP
#define GEMMSTONE_GENERATOR_PIECES_REMASK_HPP

#include "gemmstone/type.hpp"
#include "gemmstone/strategy.hpp"

#include "register_layout.hpp"


GEMMSTONE_NAMESPACE_START

// Check if a register block needs to be remasked to ensure out-of-bounds
//  entries are zero.
bool needsRemask(Type T, bool column, const RegisterLayout &layout,
                 const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks = false);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
