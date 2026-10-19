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


#ifndef GEMMSTONE_GENERATOR_PIECES_NGEN_OBJECT_HELPERS_HPP
#define GEMMSTONE_GENERATOR_PIECES_NGEN_OBJECT_HELPERS_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/type.hpp"
#include "gemmstone/problem.hpp"

GEMMSTONE_NAMESPACE_START

// DataType queries and helpers.
static inline bool isInt2(ngen::DataType dt) { return one_of(dt, {ngen::DataType::u2, ngen::DataType::s2});}
static inline bool isInt3(ngen::DataType dt) { return dt == Type::ngen_u3(); }
static inline bool isInt4(ngen::DataType dt) { return one_of(dt, {ngen::DataType::u4, ngen::DataType::s4});}
static inline bool isSubByteInt(ngen::DataType dt) { return isInt2(dt) || isInt4(dt); }
static inline bool isB(ngen::DataType dt) { return one_of(dt, {ngen::DataType::ub, ngen::DataType::b}); }
static inline bool isW(ngen::DataType dt) { return one_of(dt, {ngen::DataType::uw, ngen::DataType::w}); }
static inline bool isD(ngen::DataType dt) { return one_of(dt, {ngen::DataType::ud, ngen::DataType::d}); }
static inline bool isQ(ngen::DataType dt) { return one_of(dt, {ngen::DataType::uq, ngen::DataType::q}); }
static inline bool isFP8(ngen::DataType dt) { return one_of(dt, {ngen::DataType::bf8, ngen::DataType::hf8, Type::ngen_e8m0()}); }
static inline bool isFP4(ngen::DataType dt) { return one_of(dt, {ngen::DataType::e2m1, Type::ngen_nf4()}); }
static inline bool is4(ngen::DataType dt) { return isInt4(dt) || isFP4(dt); }

static inline bool isFP(ngen::DataType dt) {
    using namespace ngen;
    return one_of(dt, {DataType::bf8, DataType::hf8, DataType::bf, DataType::hf, DataType::tf32, DataType::f, DataType::df, Type::ngen_e8m0()});
}
static inline bool isInt(ngen::DataType dt) { return !isFP(dt); }

static inline ngen::DataType asSigned(ngen::DataType dt)
{
    using namespace ngen;
    switch (dt) {
        case DataType::u2: return DataType::s2;
        case DataType::u4: return DataType::s4;
        case DataType::ub: return DataType::b;
        case DataType::uw: return DataType::w;
        case DataType::ud: return DataType::d;
        case DataType::uq: return DataType::q;
        default: return dt;
    }
}

// u3's 3-bit width isn't a power of two, so ngen::DataType's log2-based
// getBits()/getLog2Bits() can't encode it; compute byte/element
// conversions directly instead.
static inline int elementsToBytes(int n, ngen::DataType dt) {
    if (dt == Type::ngen_u3()) return (n * 3 + 7) >> 3;
    return n * getBits(dt) >> 3;
}
static inline int bytesToElements(int b, ngen::DataType dt) {
    if (dt == Type::ngen_u3()) return (b * 8) / 3;
    return (b * 8) >> getLog2Bits(dt);
}

// Move subregister to another pipe.
void movePipes(ngen::Subregister &s, bool sizeCanChange = true);

// Move subregister to integer pipe.
void moveToIntPipe(ngen::Subregister &s);

// Move register region to integer pipe.
void moveToIntPipe(int esize, ngen::RegData &s);

// Measure distance in elements between two subregisters.
int elementDiff(ngen::HW hw, const ngen::RegData &r1, const ngen::RegData &r2);

// Cast a value to an nGEN immediate of the given type.
template <typename U>
ngen::Immediate cast(Type T, U val)
{
    switch (T) {
        case Type::f16:  return ngen::half((float)val);
        case Type::f32:  return float(val);
        case Type::f64:  return double(val);
        case Type::u8:   return  uint8_t(val);
        case Type::s8:   return   int8_t(val);
        case Type::u16:  return uint16_t(val);
        case Type::s16:  return  int16_t(val);
        case Type::u32:  return uint32_t(val);
        case Type::s32:  return  int32_t(val);
        case Type::u64:  return uint64_t(val);
        case Type::s64:  return  int64_t(val);
        case Type::bf16:
        case Type::tf32:
        default:         stub();
    }
}

static inline ngen::Immediate cast(Type T, Scalar val) {
    return cast(T, double(val));
}

// Comparison operators, for use with generic code.
/* check if these are still used */
constexpr bool operator==(const ngen::RegData &rd, int i)                    { return false; }
constexpr bool operator==(const ngen::RegData &rd, const ngen::Immediate &i) { return false; }
constexpr bool operator!=(const ngen::RegData &rd, int i)                    { return true; }
constexpr bool operator!=(const ngen::RegData &rd, const ngen::Immediate &i) { return true; }

/* Modify a cache policy to be L1-uncacheable */
ngen::CacheSettingsLSC makeL1Uncacheable(ngen::CacheSettingsLSC c);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
