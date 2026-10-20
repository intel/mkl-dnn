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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_CONFIG_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_CONFIG_HPP

#if (defined(__has_include) && __has_include("gemmstone_config.hpp")) || defined(GEMMSTONE_CONFIG)
#include "gemmstone_config.hpp"
#else

namespace gemmstone {
enum class GEMMVerbose { DebugInfo };
#ifdef STANDALONE
int getVerbose(GEMMVerbose v);
#else
inline int getVerbose(GEMMVerbose v) { return 0; }
#endif
} // namespace gemmstone

#define GEMMSTONE_ASSERTIONS 1
#define GEMMSTONE_IR_EXTENSION_HANDLER nullptr

#define MICROKERNEL_INTERFACE

#if (__cplusplus >= 202002L || _MSVC_LANG >= 202002L)
#if __has_include(<version>)
#include <version>
#if __cpp_lib_source_location >= 201907L
#define GEMMSTONE_ENABLE_SOURCE_LOCATION true
#endif
#endif
#endif

#endif

#ifdef STANDALONE
#   if defined(FORMAT_ABBREV_)
#       define ASM_OUTPUT
#   elif defined(FORMAT_ABBREV_b)
#       define BINARY_OUTPUT
#   elif defined(FORMAT_ABBREV_e)
#       define ZEBIN_OUTPUT
#   elif defined(FORMAT_ABBREV_o)
#       define OPENCL_OUTPUT
#       ifndef CL_TARGET_OPENCL_VERSION
#           define CL_TARGET_OPENCL_VERSION 210
#       endif
#   endif
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
#   ifndef CL_TARGET_OPENCL_VERSION
#       define CL_TARGET_OPENCL_VERSION 210
#   endif
#endif

#ifndef GEMMSTONE_ASSERTIONS
#define GEMMSTONE_ASSERTIONS 0
#endif

#ifndef GEMMSTONE_NAMESPACE_START
#define GEMMSTONE_NAMESPACE gemmstone
#define GEMMSTONE_NAMESPACE_START namespace gemmstone {
#endif

#ifndef GEMMSTONE_NAMESPACE_END
#define GEMMSTONE_NAMESPACE_END }
#endif

#ifndef GEMMSTONE_DEPRECATED
#if __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
#define GEMMSTONE_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define GEMMSTONE_DEPRECATED(msg)
#endif
#endif

#endif /* header guard */
