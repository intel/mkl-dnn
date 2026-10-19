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

#ifndef GEMMSTONE_GENERATOR_DSL_BUILDER_HPP
#define GEMMSTONE_GENERATOR_DSL_BUILDER_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/dsl/dsl.hpp"

GEMMSTONE_NAMESPACE_START

struct generator_dsl_desc_t;

dsl::kernel_t make_kernel(const generator_dsl_desc_t &desc);

GEMMSTONE_NAMESPACE_END

#endif
