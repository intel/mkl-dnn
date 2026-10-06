/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_PRINT_HELPERS_HPP
#define GEMMSTONE_TOOLS_PRINT_HELPERS_HPP

#include <cstdint>

// Type conversions for printing with ostreams.
template <typename T> struct printable { using type = T; };
template <> struct printable<uint8_t> { using type = int; };
template <> struct printable<int8_t>  { using type = int; };

#endif /* header guard */
