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

#include "bfn.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

BFN::operator uint8_t() const {
    switch (op) {
        case ngen::Opcode::mov: return left;
        case ngen::Opcode::and_: return left & right;
        case ngen::Opcode::or_: return left | right;
        case ngen::Opcode::xor_: return left ^ right;
        default: stub();
    }
}

std::string BFN::str() const {
    if (op == ngen::Opcode::mov) {
        if (left == 0x00) return "0";
        if (left == 0xFF) return "1";
        if (left == 0x0F) return "~s2";
        if (left == 0x33) return "~s1";
        if (left == 0x55) return "~s0";
        if (left == 0xAA) return "s0";
        if (left == 0xCC) return "s1";
        if (left == 0xF0) return "s2";
        return "???";
    }
    std::string op_name = (op == ngen::Opcode::and_ ? "&" : op == ngen::Opcode::or_ ? "|" : "^");
    return nodes[left].str() + op_name + nodes[right].str();
}

BFN BFN::operator~() const {
    auto inv = [](uint8_t v) -> uint8_t { return ~v & 0xFF; };

    switch (op) {
        case ngen::Opcode::mov: return {ngen::Opcode::mov, inv(left)};
        case ngen::Opcode::and_: return {ngen::Opcode::or_, inv(left), inv(right)};
        case ngen::Opcode::or_: return {ngen::Opcode::and_, inv(left), inv(right)};
        case ngen::Opcode::xor_: return {ngen::Opcode::xor_, left, inv(right)};
        default: stub();
    }
}

BFN BFN::zeros = 0x00;
BFN BFN::s0 = 0xAA;
BFN BFN::s1 = 0xCC;
BFN BFN::s2 = 0xF0;

BFN BFN::nodes[256] = {
    zeros,                      // bfn.0x00
    (~s0&~s1)&~s2,              // bfn.0x01
    (s0&~s1)&~s2,               // bfn.0x02
    ~s1&~s2,                    // bfn.0x03
    (~s0&s1)&~s2,               // bfn.0x04
    ~s0&~s2,                    // bfn.0x05
    (s0^s1)&~s2,                // bfn.0x06
    (~s0|~s1)&~s2,              // bfn.0x07
    (s0&s1)&~s2,                // bfn.0x08
    (s0^~s1)&~s2,               // bfn.0x09
    s0&~s2,                     // bfn.0x0A
    (s0|~s1)&~s2,               // bfn.0x0B
    s1&~s2,                     // bfn.0x0C
    (~s0|s1)&~s2,               // bfn.0x0D
    (s0|s1)&~s2,                // bfn.0x0E
    ~s2,                        // bfn.0x0F
    (~s0&~s1)&s2,               // bfn.0x10
    ~s0&~s1,                    // bfn.0x11
    (s0^s2)&~s1,                // bfn.0x12
    (~s0|~s2)&~s1,              // bfn.0x13
    ~s0&(s1^s2),                // bfn.0x14
    ~s0&(~s1|~s2),              // bfn.0x15
    s0^((s0&s1)|(s1^s2)),       // bfn.0x16
    (~s0&~s1)|((~s0|~s1)&~s2),  // bfn.0x17
    (s0^s2)&(s1^s2),            // bfn.0x18
    s0^((s0&s2)|~s1),           // bfn.0x19
    s0^((s0|~s1)&s2),           // bfn.0x1A
    (s0&~s2)|(~s0&~s1),         // bfn.0x1B
    ((s0&s2)|s1)^s2,            // bfn.0x1C
    (~s0&~s1)|(s1&~s2),         // bfn.0x1D
    (s0|s1)^s2,                 // bfn.0x1E
    (~s0&~s1)|~s2,              // bfn.0x1F
    (s0&~s1)&s2,                // bfn.0x20
    (s0^~s2)&~s1,               // bfn.0x21
    s0&~s1,                     // bfn.0x22
    (s0|~s2)&~s1,               // bfn.0x23
    (s0^s1)&(s1^s2),            // bfn.0x24
    s0^((s0&s1)|~s2),           // bfn.0x25
    s0^((s0|~s2)&s1),           // bfn.0x26
    (s0&~s1)|(~s0&~s2),         // bfn.0x27
    s0&(s1^s2),                 // bfn.0x28
    s0^(((~s0&s1)|~s2)^s1),     // bfn.0x29
    s0&(~s1|~s2),               // bfn.0x2A
    (s0&~s1)|((s0|~s1)&~s2),    // bfn.0x2B
    ((~s0&s2)|s1)^s2,           // bfn.0x2C
    (~s0|s1)^s2,                // bfn.0x2D
    (s0&~s1)|(s1&~s2),          // bfn.0x2E
    (s0&~s1)|~s2,               // bfn.0x2F
    ~s1&s2,                     // bfn.0x30
    (~s0|s2)&~s1,               // bfn.0x31
    (s0|s2)&~s1,                // bfn.0x32
    ~s1,                        // bfn.0x33
    ((s0&s1)|s2)^s1,            // bfn.0x34
    (~s0&~s2)|(~s1&s2),         // bfn.0x35
    (s0|s2)^s1,                 // bfn.0x36
    (~s0&~s2)|~s1,              // bfn.0x37
    ((~s0&s1)|s2)^s1,           // bfn.0x38
    (~s0|s2)^s1,                // bfn.0x39
    (s0&~s2)|(~s1&s2),          // bfn.0x3A
    (s0&~s2)|~s1,               // bfn.0x3B
    s1^s2,                      // bfn.0x3C
    (~s0&~s1)|(s1^s2),          // bfn.0x3D
    (s0&~s1)|(s1^s2),           // bfn.0x3E
    ~s1|~s2,                    // bfn.0x3F
    (~s0&s1)&s2,                // bfn.0x40
    ~s0&(s1^~s2),               // bfn.0x41
    (s0^s1)&(s1^~s2),           // bfn.0x42
    ((s0&s1)|~s2)^s1,           // bfn.0x43
    ~s0&s1,                     // bfn.0x44
    ~s0&(s1|~s2),               // bfn.0x45
    s0^((s0&s2)|s1),            // bfn.0x46
    (~s0&s1)|(~s1&~s2),         // bfn.0x47
    (s0^s2)&s1,                 // bfn.0x48
    s0^((s0&s2)|(s1^~s2)),      // bfn.0x49
    s0^((s0|s1)&s2),            // bfn.0x4A
    (s0|~s1)^s2,                // bfn.0x4B
    (~s0|~s2)&s1,               // bfn.0x4C
    (~s0&s1)|((~s0|s1)&~s2),    // bfn.0x4D
    (s0&~s2)|(~s0&s1),          // bfn.0x4E
    (~s0&s1)|~s2,               // bfn.0x4F
    ~s0&s2,                     // bfn.0x50
    ~s0&(~s1|s2),               // bfn.0x51
    s0^((s0&s1)|s2),            // bfn.0x52
    (~s0&s2)|(~s1&~s2),         // bfn.0x53
    ~s0&(s1|s2),                // bfn.0x54
    ~s0,                        // bfn.0x55
    s0^(s1|s2),                 // bfn.0x56
    ~s0|(~s1&~s2),              // bfn.0x57
    s0^((s0&~s1)|s2),           // bfn.0x58
    s0^(~s1|s2),                // bfn.0x59
    s0^s2,                      // bfn.0x5A
    (s0^s2)|(~s0&~s1),          // bfn.0x5B
    (~s0&s2)|(s1&~s2),          // bfn.0x5C
    ~s0|(s1&~s2),               // bfn.0x5D
    (s0^s2)|(s1&~s2),           // bfn.0x5E
    ~s0|~s2,                    // bfn.0x5F
    (s0^s1)&s2,                 // bfn.0x60
    s0^((s0&s1)|(s1^~s2)),      // bfn.0x61
    s0^((s0|s2)&s1),            // bfn.0x62
    (s0|~s2)^s1,                // bfn.0x63
    s0^((s0&~s2)|s1),           // bfn.0x64
    s0^(s1|~s2),                // bfn.0x65
    s0^s1,                      // bfn.0x66
    (s0^s1)|(~s0&~s2),          // bfn.0x67
    s0^(((~s0&~s1)|s2)^~s1),    // bfn.0x68
    (s0^s1)^~s2,                // bfn.0x69
    s0^(s1&s2),                 // bfn.0x6A
    (s0^(s1&s2))|(~s1&~s2),     // bfn.0x6B
    (~s0|~s2)^~s1,              // bfn.0x6C
    (s0^(s1|~s2))|(s1&~s2),     // bfn.0x6D
    (s0^s1)|(s0&~s2),           // bfn.0x6E
    (s0^s1)|~s2,                // bfn.0x6F
    (~s0|~s1)&s2,               // bfn.0x70
    (~s0&s2)|((~s0|s2)&~s1),    // bfn.0x71
    (s0&~s1)|(~s0&s2),          // bfn.0x72
    (~s0&s2)|~s1,               // bfn.0x73
    (~s0&s1)|(~s1&s2),          // bfn.0x74
    ~s0|(~s1&s2),               // bfn.0x75
    (s0^s1)|(~s0&s2),           // bfn.0x76
    ~s0|~s1,                    // bfn.0x77
    (~s0|~s1)^~s2,              // bfn.0x78
    (s0^(~s1|s2))|(~s1&s2),     // bfn.0x79
    (s0^s2)|(s0&~s1),           // bfn.0x7A
    (s0^s2)|~s1,                // bfn.0x7B
    (~s0&s1)|(s1^s2),           // bfn.0x7C
    ~s0|(s1^s2),                // bfn.0x7D
    (s0^s1)|(s1^s2),            // bfn.0x7E
    (~s0|~s1)|~s2,              // bfn.0x7F
    (s0&s1)&s2,                 // bfn.0x80
    (s0^~s1)&(s1^~s2),          // bfn.0x81
    s0&(s1^~s2),                // bfn.0x82
    ((~s0&s1)|~s2)^s1,          // bfn.0x83
    (s0^~s2)&s1,                // bfn.0x84
    s0^((s0&~s1)|~s2),          // bfn.0x85
    s0^(((~s0&s2)|s1)^s2),      // bfn.0x86
    (~s0|~s1)^s2,               // bfn.0x87
    s0&s1,                      // bfn.0x88
    s0^((s0|~s2)&~s1),          // bfn.0x89
    s0&(s1|~s2),                // bfn.0x8A
    (s0&s1)|(~s1&~s2),          // bfn.0x8B
    (s0|~s2)&s1,                // bfn.0x8C
    (s0&s1)|(~s0&~s2),          // bfn.0x8D
    (s0&s1)|((s0|s1)&~s2),      // bfn.0x8E
    (s0&s1)|~s2,                // bfn.0x8F
    (s0^~s1)&s2,                // bfn.0x90
    s0^((s0&~s2)|~s1),          // bfn.0x91
    s0^(((~s0&s1)|s2)^s1),      // bfn.0x92
    (~s0|~s2)^s1,               // bfn.0x93
    s0^((s0&~s1)|(s1^s2)),      // bfn.0x94
    s0^(~s1|~s2),               // bfn.0x95
    (s0^s1)^s2,                 // bfn.0x96
    (s0^(~s1|~s2))|(~s1&~s2),   // bfn.0x97
    s0^((s0|s2)&~s1),           // bfn.0x98
    s0^~s1,                     // bfn.0x99
    s0^(~s1&s2),                // bfn.0x9A
    (s0^~s1)|(s0&~s2),          // bfn.0x9B
    (s0|~s2)^~s1,               // bfn.0x9C
    (s0^~s1)|(s1&~s2),          // bfn.0x9D
    (s0^(~s1&s2))|(s1&~s2),     // bfn.0x9E
    (s0^~s1)|~s2,               // bfn.0x9F
    s0&s2,                      // bfn.0xA0
    s0^((s0|~s1)&~s2),          // bfn.0xA1
    s0&(~s1|s2),                // bfn.0xA2
    (s0&s2)|(~s1&~s2),          // bfn.0xA3
    s0^((s0|s1)&~s2),           // bfn.0xA4
    s0^~s2,                     // bfn.0xA5
    s0^(s1&~s2),                // bfn.0xA6
    (s0^~s2)|(s0&~s1),          // bfn.0xA7
    s0&(s1|s2),                 // bfn.0xA8
    s0^(~s1&~s2),               // bfn.0xA9
    s0,                         // bfn.0xAA
    s0|(~s1&~s2),               // bfn.0xAB
    (s0&s2)|(s1&~s2),           // bfn.0xAC
    (s0^~s2)|(s0&s1),           // bfn.0xAD
    s0|(s1&~s2),                // bfn.0xAE
    s0|~s2,                     // bfn.0xAF
    (s0|~s1)&s2,                // bfn.0xB0
    (s0&s2)|(~s0&~s1),          // bfn.0xB1
    (s0&s2)|((s0|s2)&~s1),      // bfn.0xB2
    (s0&s2)|~s1,                // bfn.0xB3
    (s0|~s1)^~s2,               // bfn.0xB4
    (s0^~s2)|(~s0&~s1),         // bfn.0xB5
    (s0^(s1&~s2))|(~s1&s2),     // bfn.0xB6
    (s0^~s2)|~s1,               // bfn.0xB7
    (s0&s1)|(~s1&s2),           // bfn.0xB8
    (s0^~s1)|(s0&s2),           // bfn.0xB9
    s0|(~s1&s2),                // bfn.0xBA
    s0|~s1,                     // bfn.0xBB
    (s0&s1)|(s1^s2),            // bfn.0xBC
    (s0^~s1)|(s1^s2),           // bfn.0xBD
    s0|(s1^s2),                 // bfn.0xBE
    (s0|~s1)|~s2,               // bfn.0xBF
    s1&s2,                      // bfn.0xC0
    ((s0&~s1)|s2)^~s1,          // bfn.0xC1
    ((~s0&~s1)|s2)^~s1,         // bfn.0xC2
    s1^~s2,                     // bfn.0xC3
    (~s0|s2)&s1,                // bfn.0xC4
    (~s0&~s2)|(s1&s2),          // bfn.0xC5
    (~s0|s2)^~s1,               // bfn.0xC6
    (~s0&s1)|(s1^~s2),          // bfn.0xC7
    (s0|s2)&s1,                 // bfn.0xC8
    (s0|s2)^~s1,                // bfn.0xC9
    (s0&~s2)|(s1&s2),           // bfn.0xCA
    (s0&s1)|(s1^~s2),           // bfn.0xCB
    s1,                         // bfn.0xCC
    (~s0&~s2)|s1,               // bfn.0xCD
    (s0&~s2)|s1,                // bfn.0xCE
    s1|~s2,                     // bfn.0xCF
    (~s0|s1)&s2,                // bfn.0xD0
    (~s0&~s1)|(s1&s2),          // bfn.0xD1
    (~s0|s1)^~s2,               // bfn.0xD2
    (~s0&s2)|(s1^~s2),          // bfn.0xD3
    (~s0&s1)|((~s0|s1)&s2),     // bfn.0xD4
    ~s0|(s1&s2),                // bfn.0xD5
    (s0^(s1|s2))|(s1&s2),       // bfn.0xD6
    ~s0|(s1^~s2),               // bfn.0xD7
    (s0&s1)|(~s0&s2),           // bfn.0xD8
    (s0^~s1)|(s1&s2),           // bfn.0xD9
    (s0^s2)|(s0&s1),            // bfn.0xDA
    (s0^s2)|(s1^~s2),           // bfn.0xDB
    (~s0&s2)|s1,                // bfn.0xDC
    ~s0|s1,                     // bfn.0xDD
    (s0^s2)|s1,                 // bfn.0xDE
    (~s0|s1)|~s2,               // bfn.0xDF
    (s0|s1)&s2,                 // bfn.0xE0
    (s0|s1)^~s2,                // bfn.0xE1
    (s0&~s1)|(s1&s2),           // bfn.0xE2
    (s0&s2)|(s1^~s2),           // bfn.0xE3
    (s0&s2)|(~s0&s1),           // bfn.0xE4
    (s0^~s2)|(s1&s2),           // bfn.0xE5
    (s0^s1)|(s0&s2),            // bfn.0xE6
    (s0^s1)|(s1^~s2),           // bfn.0xE7
    (s0&s1)|((s0|s1)&s2),       // bfn.0xE8
    (s0^(~s1&~s2))|(s1&s2),     // bfn.0xE9
    s0|(s1&s2),                 // bfn.0xEA
    s0|(s1^~s2),                // bfn.0xEB
    (s0&s2)|s1,                 // bfn.0xEC
    (s0^~s2)|s1,                // bfn.0xED
    s0|s1,                      // bfn.0xEE
    (s0|s1)|~s2,                // bfn.0xEF
    s2,                         // bfn.0xF0
    (~s0&~s1)|s2,               // bfn.0xF1
    (s0&~s1)|s2,                // bfn.0xF2
    ~s1|s2,                     // bfn.0xF3
    (~s0&s1)|s2,                // bfn.0xF4
    ~s0|s2,                     // bfn.0xF5
    (s0^s1)|s2,                 // bfn.0xF6
    (~s0|~s1)|s2,               // bfn.0xF7
    (s0&s1)|s2,                 // bfn.0xF8
    (s0^~s1)|s2,                // bfn.0xF9
    s0|s2,                      // bfn.0xFA
    (s0|~s1)|s2,                // bfn.0xFB
    s1|s2,                      // bfn.0xFC
    (~s0|s1)|s2,                // bfn.0xFD
    (s0|s1)|s2,                 // bfn.0xFE
    ~zeros,                     // bfn.0xFF
};

GEMMSTONE_NAMESPACE_END
