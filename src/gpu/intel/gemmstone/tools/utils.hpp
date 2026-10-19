/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_UTILS_HPP
#define GEMMSTONE_TOOLS_UTILS_HPP

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vt100_codes.hpp"

namespace utils {

enum class KernelModifier {
    None, ReadCheck, Super, Fused, Batch,
    Offset, Scale, SumA, SumB, Bias, HPL
};

inline KernelModifier kernelModifier(std::string_view word)
{
    if (word.starts_with("rc")) return KernelModifier::ReadCheck;
    if (word == "super" || word == "superll") return KernelModifier::Super;
    if (word == "fused") return KernelModifier::Fused;
    if (word == "batchs" || word == "batchn" || word == "batchnv") return KernelModifier::Batch;
    if (word.starts_with("offset")) return KernelModifier::Offset;
    if (word.starts_with("scale")) return KernelModifier::Scale;
    if (word == "suma") return KernelModifier::SumA;
    if (word == "sumb") return KernelModifier::SumB;
    if (word == "bias") return KernelModifier::Bias;
    if (word == "hpl") return KernelModifier::HPL;
    return KernelModifier::None;
}

struct KernelType {
    std::string base;
    std::string modifiers;
};

template <typename Next>
KernelType readKernelType(Next next)
{
    KernelType result;
    for (;;) {
        std::string word = next();
        auto modifier = kernelModifier(word);
        if (modifier == KernelModifier::None) {
            result.base = std::move(word);
            break;
        }
        if (!result.modifiers.empty()) result.modifiers += ' ';
        result.modifiers += word;
    }
    return result;
}

template <typename T1, typename T2>
inline auto divUp(T1 x, T2 y) -> decltype((x + y - 1) / y)
{
    return (x + y - 1) / y;
}

template <typename T1, typename T2>
inline auto alignUp(T1 x, T2 y) -> decltype(divUp(x, y) * y)
{
    return divUp(x, y) * y;
}

inline std::vector<std::string> extractWords(const std::string &str)
{
    std::vector<std::string> words;
    std::string current;
    for (unsigned char c: str) {
        if (std::isspace(c)) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
        } else
            current += static_cast<char>(c);
    }
    if (!current.empty())
        words.push_back(std::move(current));
    return words;
}

inline void expect(std::istream &s, char c)
{
    char in = '\0';
    s >> std::ws;
    if (!(s >> in))
        throw std::runtime_error(std::string("expected '") + c + "', but found EOF");
    if (in != c)
        throw std::runtime_error(std::string("expected '") + c + "', but found '" + in + "'");
}

inline bool contains(const std::string &haystack, const std::string &needle)
{
    return haystack.find(needle) != std::string::npos;
}

inline bool contains(const std::string &haystack, char needle)
{
    return haystack.find(needle) != std::string::npos;
}

inline FILE * openFile(const char * file_name, const char * mode) {
    FILE * f = fopen(file_name, mode);
    if(f == nullptr)
        throw std::runtime_error(std::string("Failed to open file ") +
                                 file_name + ": " + strerror(errno));

    return f;
}

template <typename Container>
inline void readFile(Container &out, FILE *file)
{
    const int chunk = 65536;
    size_t read = 0;

    out.clear();
    do {
        auto osize = out.size();
        out.resize(osize + chunk);
        read = fread(&out[osize], 1, chunk, file);
        auto status = ferror(file);
        if (status)
            throw std::runtime_error(std::string("Failed to read file: ") + strerror(status));
        out.resize(osize + read);
    } while (read > 0);
}

template <typename T>
inline T readValueFromFile(const std::string &fileName, T defaultValue)
{
    std::ifstream s(fileName);
    if (!s) return defaultValue;

    T value = defaultValue;
    s >> value;

    if (!s) return defaultValue;
    return value;
}

inline int setenv(const char* name, const char* value, int overwrite) {
#ifdef _WIN32
    // Check if variable exists and we're not supposed to overwrite
    if (!overwrite) {
        const char* existing = std::getenv(name);
        if (existing != nullptr) {
            return 0; // Already set, and overwrite not requested
        }
    }
    return _putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite);
#endif
}

inline int unsetenv(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

inline int strcasecmp(const char* a, const char* b) {
#ifdef _WIN32
    return _stricmp(a, b);
#else
    return ::strcasecmp(a, b);
#endif
}

    
#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif

template <typename... Args>
inline void cprintf(FILE *f, const char *color, const char *fmt, Args&&... args)
{
    fputs(vt(color), f);
    fprintf(f, fmt, std::forward<Args>(args)...);
    fputs(vt(VT_RESET), f);
}

template <typename... Args>
inline void rprintf(const char *fmt, Args&&... args) { cprintf(stderr, VT_RED, fmt, std::forward<Args>(args)...); }

template <typename... Args>
inline void yprintf(const char *fmt, Args&&... args) { cprintf(stderr, VT_YELLOW, fmt, std::forward<Args>(args)...); }

template <typename... Args>
inline void gprintf(const char *fmt, Args&&... args) { cprintf(stdout, VT_GREEN, fmt, std::forward<Args>(args)...); }

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

} // namespace utils

#endif
