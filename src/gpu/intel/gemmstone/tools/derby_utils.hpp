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

#ifndef GEMMSTONE_TOOLS_DERBY_UTILS_HPP
#define GEMMSTONE_TOOLS_DERBY_UTILS_HPP

#include "device_utils.hpp"

ngen::HW currentHW();
int eusPerSS();
int euCount();
int ssCount();
bool gen12();
int grfBytes();
int threadsPerSS(int grfs);
int maxWG();
int maxSLM();

bool isColMajor(char c);
bool isSubByte(char c);
bool isComplex(char c);
bool isInteger(char c);
bool isBlock2D(char c);

[[noreturn]] void fail(const char *message);
[[noreturn]] void fail(const std::string &message);

void testPipes();

static inline bool contains(const std::string &haystack, const std::string &needle)
{
    return haystack.find(needle) != std::string::npos;
}

static inline std::string lowercase(std::string s)
{
    for (auto &c: s)
        c = std::tolower(c);
    return s;
}

#endif
