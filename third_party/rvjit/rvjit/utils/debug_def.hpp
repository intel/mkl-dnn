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

// Deliberately not include-guarded: every rvjit component/instruction/stream
// header includes this immediately after its own dependency includes, uses
// DEBUG/LOG_FAIL/CHECK/CHECK_OR_RET, then includes debug_undef.hpp right
// before its own closing #endif -- keeping each header exactly as
// standalone-includable, with none of these macros bleeding past that
// header's own span, as if it had hand-rolled the definitions itself

// A host project may define macros of the same names (oneDNN has CHECK):
// they are saved here and restored by debug_undef.hpp
#pragma push_macro("DEBUg")
#undef DEBUg
#pragma push_macro("DEBUG")
#undef DEBUG
#pragma push_macro("LOG_FAIL")
#undef LOG_FAIL
#pragma push_macro("CHECK")
#undef CHECK
#pragma push_macro("CHECK_OR_RET")
#undef CHECK_OR_RET
#pragma push_macro("LOG_FAIL_IF")
#undef LOG_FAIL_IF

#if defined(RVJIT_DEBUG)
// A host project supplies rvjit::debug_enabled() and rvjit::debug_printf()
// through its own header named by RVJIT_VERBOSE_HEADER
#if defined(RVJIT_VERBOSE_HEADER)
#include RVJIT_VERBOSE_HEADER
#else
#include "rvjit/utils/verbose.hpp"
#endif
#define DEBUg(...) \
    do { \
        if (::rvjit::debug_enabled()) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

// Logs "Failed <NAME>::<METHOD> due to <ERR>\n"
// NAME is an unqualified reference, so this only expands inside a member of a
// class declaring its own `static constexpr const char *NAME`
#define LOG_FAIL(METHOD, ERR, ...) \
    DEBUG({ \
        ::rvjit::debug_printf("Failed %s::%s due to " ERR "\n", NAME, \
                (METHOD), ##__VA_ARGS__); \
    })

#define CHECK(CFG, ERR, ...) \
    if (!(CFG)) { \
        LOG_FAIL(__func__, ERR, ##__VA_ARGS__); \
        return false; \
    }

#define CHECK_OR_RET(CFG, RET, ERR, ...) \
    if (!(CFG)) { \
        LOG_FAIL(__func__, ERR, ##__VA_ARGS__); \
        return RET; \
    }

// Unlike CHECK/CHECK_OR_RET, COND is the failure condition itself (not the
// guard that must hold), matching this macro's own "if failing" reading
#define LOG_FAIL_IF(COND, ERR, ...) \
    if (COND) { LOG_FAIL(__func__, ERR, ##__VA_ARGS__); }
