/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_TYPE_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_TYPE_HPP

#include "gemmstone/config.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START


// Enum-like class for data types.
class Type {
public:
    /*
    * Bitfield type ID, consists of following flags/ranges:
    * // nibble 0: type meta-information
    * 0 - fp=1, int=0
    * 1 - signed=1, unsigned=0
    * 2-3 - reserved
    * // nibble 1: complexity meta-information
    * 4 - complex=1, real=0
    * 5 - split complex=1
    * 6-7 - reserved
    * // nibble 2-4: size meta-information
    * 8-11 - size of the block (units based on following flags)
    * 12-15 - number of values per block
    * 16 - block size in bits=0, block size in bytes=1
    * 17 - log2 block size=0, actual block size=1
    * 18-19 - reserved
    * // nibble 5-6: index in ngen mapping table
    * 20-27 - ngen reference
    * // nibble 7: vector meta-information
    * 28-31 - vector component number
    */
    enum _Type : uint32_t {
        invalid  = 0,
        f16      = 0x10011101,
        f32      = 0x10111201,
        f64      = 0x10211301,
        u3       = 0x02021300,
        u2       = 0x11001100,
        s2       = 0x11101102,
        u4       = 0x11201200,
        s4       = 0x11301202,
        u8       = 0x11411000,
        s8       = 0x11511002,
        u16      = 0x11611100,
        s16      = 0x11711102,
        u32      = 0x11811200,
        s32      = 0x11911202,
        u64      = 0x11A11300,
        s64      = 0x11B11302,
        f4_e2m1  = 0x10401201,
        nf4      = 0x10601201,
        f8_e8m0  = 0x10811001,
        bf8      = 0x10E11001,
        hf8      = 0x10F11001,
        bf16     = 0x10C11101,
        tf32     = 0x10D11201,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(invalid) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr bool isValid()          const { return val != Type::invalid; }
    constexpr Type real()             const { return *this; }
    constexpr bool isComplex()        const { return false; }
    constexpr int complexComponents() const { return 1; }
    constexpr int components()        const { return 1; }
    constexpr bool isFP()             const { return uint32_t(val) & 0x1; }
    constexpr bool isInteger()        const { return !isFP(); }
    constexpr bool isSubByteInt()     const { return isSubByte() && isInteger(); };
    constexpr bool is2()              const { return bits() == 2; }
    constexpr bool is3()              const { return val == Type::u3; }
    constexpr bool is4()              const { return bits() == 4; }
    constexpr bool isInt4()           const { return is4() && isInteger(); }
    constexpr bool isIntSubByte()     const { return isSubByteInt(); }
    constexpr bool isInt8()           const { return (val == Type::u8)  || (val == Type::s8);  }
    constexpr bool isInt16()          const { return (val == Type::u16) || (val == Type::s16); }
    constexpr bool isF8()             const {
        return (val == Type::bf8) || (val == Type::hf8) || (val == Type::f8_e8m0);
    }
    constexpr bool isF4()             const { return bits() == 4 && isFP(); }
    constexpr bool isSigned()         const { return (uint32_t(val) & 0x3) != 0x0; }
    constexpr int blockSize()         const { return (uint32_t(val) >> 12) & 0xF; }
    constexpr bool isBlocked()        const { return blockSize() > 1; }
    constexpr bool isLog2Size()       const { return !(uint32_t(val) & (1 << 17)); }
    constexpr bool isByteSize()       const { return uint32_t(val) & (1 << 16); }
    constexpr int bits()              const {
        return (isByteSize() ? 8 : 1) * (isLog2Size() ? (1 << ((uint32_t(val) >> 8) & 0xF)) : ((uint32_t(val) >> 8) & 0xF)); }
    constexpr int paddedSize()        const { return is3() ? 3 : (bits() + 7) / 8; }
    int log2Size()                    const {
        subByteCheck();
        auto temp = bits() / 8;
        int val = 0;
        while (temp > 1) {
            temp >>= 1;
            val += 1;
        }
        return val; }
    int size()                        const { subByteCheck(); blockCheck(); return paddedSize(); }
    constexpr bool isSubByte()        const { return isValid() && (bits() < 8); }
    constexpr int perByte()           const { return is2() ? 4 : is4() ? 2 : 1; }
    uint16_t logPerByte()             const {
        if (!isSubByte() || is3()) stub();
        return (perByte() == 4) ? 2 : 1;
    }
    void subByteCheck()               const { if (isSubByte()) stub(); }
    void blockCheck()                 const { if (isBlocked()) stub(); }

    constexpr Type arithmetic() const {
        return (val == tf32) ? Type(f32) : real();
    }
    constexpr Type asUnsigned() const {
        return static_cast<_Type>(uint32_t(val) & ~(isInteger() ? 0x100002 : 0));
    }
    constexpr Type asSigned() const {
        return static_cast<_Type>(uint32_t(val) | (isInteger() ? 0x100002 : 0));
    }
    constexpr Type baseType() const { return *this; }

    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(U a, Type t) {
        return (a * t.bits() + (a >= 0 ? 7 : -7)) / 8;
    }
    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(Type t, U a) { return a * t; }
    template <typename U>           friend U operator*=(U &a, Type t) { a = a * t; return a; }
    template <typename U> constexpr friend decltype(std::declval<U>()/1) operator/(U a, Type t) {
        return a * 8 / t.bits();
    }

    // Not valid nGEN DataTypes; for gemmstone internal use only
    static constexpr ngen::DataType ngen_nf4()  { return static_cast<ngen::DataType>(0x58); }
    static constexpr ngen::DataType ngen_e8m0() { return static_cast<ngen::DataType>(0x79); }
    // Pseudo type for packed u3 (8 elements/3 bytes) in the copy planner;
    // rewritten into ub shift/mask sequences by planTypeConversions().
    static constexpr ngen::DataType ngen_u3()   { return static_cast<ngen::DataType>(0x59); }

    ngen::DataType ngen() const
    {
        if (val == invalid) return ngen::DataType::invalid;
        uint32_t index = (uint32_t(val) >> 20) & 0xFF;
        using DT = ngen::DataType;
        auto none = DT::invalid;
        auto dt7 = none, dt28 = none, dt31 = none;
        static const DT table[64] = {
            DT::hf,      DT::f,       DT::df,      none,
            DT::e2m1,    DT::e3m0 ,   ngen_nf4(),  dt7,
            ngen_e8m0(), none,        none,        none,
            DT::bf,      DT::tf32,    DT::bf8,     DT::hf8,
            DT::u2,      DT::s2,      DT::u4,      DT::s4,
            DT::ub,      DT::b,       DT::uw,      DT::w,
            DT::ud,      DT::d,       DT::uq,      DT::q,
            dt28,        none,        none,        dt31,
            ngen_u3(),   none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none
        };
        // While 8 bits are allocated for ngen reference encoding,
        // the actual reference table is smaller and no type should
        // normally reference outside it
        if (index >= 64) {
            stub("Invalid ngen reference encoding in type ID");
        }
        return table[index];
    }

    bool isSubsetOf(Type T) const
    {
        if (*this == T) return true;
        if (val == invalid || T == invalid) return false;
        return (real().bits() < T.real().bits());
    }

    Type asSignedInt() const
    {
        switch (bits()) {
            case 8:  return Type::s8;
            case 16: return Type::s16;
            case 32: return Type::s32;
            default: return Type::invalid;
        }
    }
};

static_assert((-9 * Type(Type::s4) == -5) && (9 * Type(Type::s4) == 5), "Round away from zero is required non-integer type sizes");

inline char typeToChar(Type T)
{
    switch (T.baseType()) {
        case Type::bf8:     return 'Q';
        case Type::hf8:     return 'q';
        case Type::f16:     return 'H';
        case Type::f32:     return 'S';
        case Type::f64:     return 'D';
        case Type::u2:      return 'p';
        case Type::s2:      return 'P';
        case Type::u3:      return 'k';
        case Type::u4:      return 'f';
        case Type::s4:      return 'F';
        case Type::u8:      return 'o';
        case Type::s8:      return 'O';
        case Type::u16:     return 'w';
        case Type::s16:     return 'W';
        case Type::u32:     return 'i';
        case Type::s32:     return 'I';
        case Type::u64:     return 'l';
        case Type::s64:     return 'L';
        case Type::bf16:    return 'B';
        case Type::tf32:    return 'T';
        case Type::f4_e2m1: return 'E';
        case Type::nf4:     return 'N';
        case Type::f8_e8m0: return 'X';
        default:            return '?';
    }
}

inline Type charToType(char c)
{
    switch (c) {
        case 'Q': return Type::bf8;
        case 'q': return Type::hf8;
        case 'H': return Type::f16;
        case 'S': return Type::f32;
        case 'D': return Type::f64;
        case 'p': return Type::u2;
        case 'P': return Type::s2;
        case 'k': return Type::u3;
        case 'f': return Type::u4;
        case 'F': return Type::s4;
        case 'o': return Type::u8;
        case 'O': return Type::s8;
        case 'w': return Type::u16;
        case 'W': return Type::s16;
        case 'i': return Type::u32;
        case 'I': return Type::s32;
        case 'B': return Type::bf16;
        case 'T': return Type::tf32;
        case 'E': return Type::f4_e2m1;
        case 'N': return Type::nf4;
        case 'X': return Type::f8_e8m0;
        default:  return Type::invalid;
    }
}

/****************************/
/* Old names -- deprecated. */
static inline char precisionChar(Type T) { return typeToChar(T); }
static inline Type charPrecision(char c) { return charToType(c); }
/****************************/

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
