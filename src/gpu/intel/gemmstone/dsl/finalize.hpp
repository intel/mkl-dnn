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

#ifndef GEMMSTONE_DSL_FINALIZE_HPP
#define GEMMSTONE_DSL_FINALIZE_HPP

#include "gemmstone/dsl/decl.hpp"
#include "gemmstone/dsl/kernel.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

void finalize(kernel_t &kernel);

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
