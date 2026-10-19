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

#ifndef GEMMSTONE_GENERATOR_PIECES_BFN_HPP
#define GEMMSTONE_GENERATOR_PIECES_BFN_HPP

#include <cstdint>

#include "internal/ngen_includes.hpp"

GEMMSTONE_NAMESPACE_START

struct BFN {
    ngen::Opcode op = ngen::Opcode::mov;
    uint8_t left = 0, right = 0;

    operator uint8_t() const;
    std::string str() const;

    bool operator==(const BFN& other) const {
        return (uint8_t)*this == (uint8_t)other;
    }

    static BFN nodes[256];

private:
    BFN() = default;
    BFN(uint8_t left) : BFN(ngen::Opcode::mov, left) {}
    BFN(ngen::Opcode op, uint8_t left, uint8_t right = 0)
        : op(op), left(left), right(right) {}

    BFN operator~() const;
    BFN operator&(const BFN& other) const { return {ngen::Opcode::and_, *this, other}; }
    BFN operator|(const BFN& other) const { return {ngen::Opcode::or_, *this, other}; }
    BFN operator^(const BFN& other) const { return {ngen::Opcode::xor_, *this, other}; }

    static BFN zeros;
    static BFN s0;
    static BFN s1;
    static BFN s2;
};

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
