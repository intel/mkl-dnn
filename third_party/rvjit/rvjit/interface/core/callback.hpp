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

#ifndef RVJIT_INTERFACE_CORE_CALLBACK_HPP
#define RVJIT_INTERFACE_CORE_CALLBACK_HPP

#include <functional>

#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

/// Parametrized callback to return code emission to a caller
template <typename... Args>
using callback_t = std::function<void(Args...)>;

/// Callback parametrized with a general-purpose register reference
using on_reg_t = callback_t<const reg_t &>;

} // namespace rvjit

#endif
