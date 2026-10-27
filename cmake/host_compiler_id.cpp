/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)

int main() {
#if defined(TRY_GNU)

#if (defined __GNUC__) && (!defined(__INTEL_COMPILER)) \
        && (!defined(__INTEL_LLVM_COMPILER)) && (!defined(__clang_major__))
#pragma message("host compiler version: " STRINGIFY(__GNUC__) "." STRINGIFY( \
        __GNUC_MINOR__))
    return 0;
#else
    breaks_on_purpose
#endif

#elif defined(TRY_CLANG)

#if (!defined(__INTEL_LLVM_COMPILER)) && (defined(__clang_major__))
#pragma message("host compiler version: " STRINGIFY( \
        __clang_major__) "." STRINGIFY(__clang_minor__))
    return 0;
#else
    breaks_on_purpose
#endif

#elif defined(TRY_MSVC)

#if (defined _MSC_VER) && (!defined(__INTEL_COMPILER)) \
        && (!defined(__INTEL_LLVM_COMPILER)) && (!defined(__clang__))
// _MSC_VER encodes the version as a single number, e.g. 1942 for 19.42.
#pragma message("host compiler version: " STRINGIFY(_MSC_VER))
    return 0;
#else
    breaks_on_purpose
#endif

#else
    breaks_on_purpose
#endif
}
