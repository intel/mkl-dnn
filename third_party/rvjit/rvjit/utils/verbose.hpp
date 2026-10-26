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

#ifndef RVJIT_UTILS_VERBOSE_HPP
#define RVJIT_UTILS_VERBOSE_HPP

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace rvjit {

/// Whether DEBUG traces are printed, the default `RVJIT_DEBUG` back end
///
/// @note Enabled by `RVJIT_VERBOSE` set above 1; a host project replaces
///     this header by defining `RVJIT_VERBOSE_HEADER` (see debug_def.hpp)
inline bool debug_enabled() {
    static const bool enabled = [] {
        const char *env = std::getenv("RVJIT_VERBOSE");
        return env && std::atoi(env) > 1;
    }();
    return enabled;
}

/// Prints one DEBUG trace line to stderr
///
/// @param fmt The printf-style format of the trace
inline void debug_printf(const char *fmt, ...)
#if defined(__GNUC__)
        __attribute__((format(printf, 1, 2)))
#endif
        ;

inline void debug_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

} // namespace rvjit

#endif
