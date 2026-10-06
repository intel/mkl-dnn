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

#ifndef GEMMSTONE_TOOLS_KERNEL_MODIFIERS_HPP
#define GEMMSTONE_TOOLS_KERNEL_MODIFIERS_HPP

#include "gemmstone/strategy_parser.hpp"
#include "tools/utils.hpp"

#include <array>
#include <charconv>

namespace utils {

struct QuantizationModifier {
    bool enabled = false;
    bool precalc = false;
    int ptrDims = -1;
    std::array<int, 2> groupSize = {1, 1};
    gemmstone::Type T = gemmstone::Type::invalid;

    bool grouped() const { return enabled && ptrDims == 2; }
};

struct KernelModifiers {
    bool readCheck = false;
    bool readCheckA = true, readCheckB = true, readCheckC = true;
    bool superkernel = false, fused = false;
    gemmstone::BatchMode batch = gemmstone::BatchMode::None;
    QuantizationModifier aOffset, bOffset, cOffset, aScale, bScale;
    bool sumA = false, sumB = false, bias = false, hpl = false;

    KernelModifiers() = default;

    explicit KernelModifiers(const std::string &modifiers)
    {
        for (const auto &word : extractWords(modifiers))
            if (!parse(word))
                throw std::runtime_error("Unknown kernel modifier: " + word);
    }

    // Returns false for a base kernel type or another non-modifier word.
    // Malformed modifiers throw without changing the accumulated state.
    bool parse(const std::string &word)
    {
        auto modifier = kernelModifier(word);
        if (modifier == KernelModifier::None) return false;
        auto next = *this;
        switch (modifier) {
            case KernelModifier::ReadCheck:
                next.readCheck = true;
                next.readCheckA = word.size() == 2 || word.find('a', 2) != std::string::npos;
                next.readCheckB = word.size() == 2 || word.find('b', 2) != std::string::npos;
                next.readCheckC = word.size() == 2 || word.find('c', 2) != std::string::npos;
                if (word.find_first_not_of("abc", 2) != std::string::npos)
                    throw std::runtime_error("Invalid read check specification: " + word);
                break;
            case KernelModifier::Super: next.superkernel = true; break;
            case KernelModifier::Fused: next.fused = true; break;
            case KernelModifier::Batch:
                next.batch = word == "batchs" ? gemmstone::BatchMode::Strided
                           : word == "batchn" ? gemmstone::BatchMode::Nonstrided
                                             : gemmstone::BatchMode::Variable;
                break;
            case KernelModifier::Offset:
            case KernelModifier::Scale: {
                bool scale = modifier == KernelModifier::Scale;
                size_t prefix = scale ? 5 : 6;
                auto invalid = [&]() {
                    return std::runtime_error("Invalid " + word.substr(0, prefix) + " specification: " + word);
                };
                const char *cp = word.c_str() + prefix;
                if (!*cp) throw invalid();
                while (*cp) {
                    char matrix = *cp++;
                    if (matrix != 'a' && matrix != 'b' && (scale || matrix != 'c'))
                        throw invalid();
                    QuantizationModifier value;
                    value.enabled = true;
                    if (*cp == '[') {
                        ++cp;
                        if (!*cp) throw invalid();
                        value.precalc = std::isupper(static_cast<unsigned char>(*cp));
                        switch (std::tolower(static_cast<unsigned char>(*cp++))) {
                            case 's': value.ptrDims = -1; break;
                            case 'p': value.ptrDims = 0; break;
                            case 'v': value.ptrDims = 1; break;
                            case 'g': {
                                value.ptrDims = 2;
                                auto groupSize = [&]() {
                                    int size;
                                    auto result = std::from_chars(cp, word.c_str() + word.size(), size);
                                    if (result.ec != std::errc{} || size <= 0) throw invalid();
                                    cp = result.ptr;
                                    return size;
                                };
                                int first = groupSize();
                                if (*cp == 'x') {
                                    ++cp;
                                    value.groupSize = {first, groupSize()};
                                } else
                                    value.groupSize = matrix == 'a' ? std::array<int, 2>{1, first}
                                                                    : std::array<int, 2>{first, 1};
                                break;
                            }
                            default: throw invalid();
                        }
                        if (*cp == ',') {
                            ++cp;
                            if (!*cp) throw invalid();
                            cp = gemmstone::parsePrecision(cp, value.T);
                            if (value.T == gemmstone::Type::invalid) throw invalid();
                        }
                        if (*cp != ']') throw invalid();
                        ++cp;
                    }
                    auto &destination = scale ? (matrix == 'a' ? next.aScale : next.bScale)
                                              : (matrix == 'a' ? next.aOffset
                                               : matrix == 'b' ? next.bOffset : next.cOffset);
                    destination = value;
                }
                break;
            }
            case KernelModifier::SumA: next.sumA = true; break;
            case KernelModifier::SumB: next.sumB = true; break;
            case KernelModifier::Bias: next.bias = true; break;
            case KernelModifier::HPL: next.hpl = true; break;
            case KernelModifier::None: break;
        }
        *this = next;
        return true;
    }
};

} // namespace utils

#endif
