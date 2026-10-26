/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_INTERFACE_ISA_ISA_TRAITS_HPP
#define RVJIT_INTERFACE_ISA_ISA_TRAITS_HPP

namespace rvjit {

/// What an ISA tells the interface about one of its native types
///
/// @details Left empty for any type no ISA describes. An ISA specializes it
///     for each native register type, declaring its `resource` class and
///     `name(native)`/`make(name)` conversions, and for every class derived
///     from its code generator, declaring that generator as `codegen_t`
template <typename Native, typename = void>
struct isa_traits {};

} // namespace rvjit

#endif
