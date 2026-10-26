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

#ifndef CPU_RV64_RVJIT_RVJIT_VERBOSE_HPP
#define CPU_RV64_RVJIT_RVJIT_VERBOSE_HPP

#include "common/verbose.hpp"

namespace rvjit {

inline bool debug_enabled() {
    return dnnl::impl::get_verbose_dev_mode(dnnl::impl::verbose_t::debuginfo)
            > 1;
}

template <typename... str_args>
inline void debug_printf(const char *fmt_str, str_args... args) {
    dnnl::impl::verbose_printf(
            dnnl::impl::verbose_t::debuginfo, fmt_str, args...);
}

} // namespace rvjit

#endif
