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


#ifndef GEMMSTONE_GENERATOR_PIECES_HW_UTILS_HPP
#define GEMMSTONE_GENERATOR_PIECES_HW_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/strategy.hpp"

#include "register_layout.hpp"

GEMMSTONE_NAMESPACE_START

template <typename T>
static inline constexpr int elementsPerGRF(ngen::HW hw)
{
    return ngen::GRF::bytes(hw) / sizeof(T);
}

static inline constexpr int elementsPerGRF(ngen::HW hw, Type T)
{
    return ngen::GRF::bytes(hw) / T;
}

static inline constexpr int elementsPerGRF(ngen::HW hw, ngen::DataType dt)
{
    // u3 (Type::ngen_u3()) has a 3-bit element width, which is not a power of
    // two and thus has no valid getLog2Bits() encoding; compute directly via
    // division instead (mirroring Type::operator/ for sub-byte types).
    return (dt == Type::ngen_u3()) ? (ngen::GRF::bytes(hw) << 3) / 3
                                   : (ngen::GRF::bytes(hw) << 3) >> getLog2Bits(dt);
}

static inline bool canSwizzle(ngen::HW hw, ngen::DataType dt)
{
    using namespace ngen;

    if (hw < HW::XeHP) return true;
    if (hw >= HW::Xe3p) return false;

    switch (dt) {
        case DataType::b:
        case DataType::ub:
        case DataType::w:
        case DataType::uw:
        case DataType::d:
        case DataType::ud: return true;
        case DataType::q:
        case DataType::uq: return (hw >= HW::XeHPC);
        default:           return false;
    }
}

static inline bool canSwizzle(ngen::HW hw, Type T)
{
    return canSwizzle(hw, T.ngen());
}

static inline bool hasNativeAtomicAdd(ngen::HW hw, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    using namespace ngen;

    bool floatAtomics = (astrategy.base.getModel() == ModelA64);
    if (astrategy.newDP)
        floatAtomics |= (astrategy.base.getModel() != ModelSLM);
    if (hw >= HW::Xe3p) floatAtomics = true;

    if (T.isSubByteInt())
        return false;
    if (T.isInteger() && T.size() >= (astrategy.newDP ? 2 : 4))
        return true;
    else if (T == Type::f32)
        return floatAtomics && (hw >= HW::XeHP);
    else if (T == Type::f64)
        return floatAtomics && (hw >= HW::XeHPC);
    else if (T == Type::f16 || T == Type::bf16)
        return (hw >= HW::Xe3p);
    else
        return false;
}

static inline size_t slmCapacity(ngen::Product p)
{
    using namespace ngen;
    auto hw = ngen::getCore(p.family);
    switch (hw) {
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:      return 131072;
        case HW::Xe2:        return 131072;
        case HW::Xe3:        return 131072;
        case HW::Xe3p:       return (p.family >= ProductFamily::CRI) ? 393216 : 196608;
        default:
            return 0;
    }
}

static inline size_t maxSLMPerWG(ngen::Product p, int grfCount)
{
    auto slmMax = slmCapacity(p);
    auto hw = ngen::getCore(p.family);
    if (hw <= ngen::HW::XeHPG)
        slmMax = std::min<size_t>(slmMax, 65536);
    return slmMax;
}

static inline int threadsPerEU(ngen::Product p, const CommonStrategy &strategy)
{
    return p.threadsPerEU(strategy.GRFs);
}

static inline int eusPerSubslice(ngen::HW hw)
{
    using namespace ngen;
    switch (hw) {
        case HW::XeHPC:
        case HW::Xe2:
        case HW::Xe3:
        case HW::Xe3p:
            return 8;
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
            return 16;
        default:
            return 0;
    }
}

// Maximum SIMD width for a scattered message.
static inline int maxScatteredSIMD(ngen::HW hw, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP)
        return ngen::GRF::bytes(hw) >> 1;
    return 16;
}

// Minimum native SIMD width for a scattered message.
static inline int minScatteredSIMD(ngen::HW hw, const MatrixAddressingStrategy &astrategy)
{
    if (hw >= ngen::HW::XeHPC)
        return 16;
    return maxScatteredSIMD(hw, astrategy) >> 1;
}

// Minimum stride alignment for a block 2D message.
static inline int block2DMinAlignment(ngen::HW hw, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool asIfBlock2D = false)
{
    using namespace ngen;
    if (!isBlock2D(astrategy.accessType) && !asIfBlock2D) return 0;
    if (hw == HW::Xe2) return 16;
    if (hw == HW::Xe3) return 16;
    if (hw >= HW::Xe3p) return 4;
    return (isTransposing(astrategy.accessType) || astrategy.prefetch) ? 4 : 8;
}

// Minimum base address alignment for block 2D messages.
static inline int block2DBaseAlignment(ngen::HW hw, int stepping)
{
    using namespace ngen;
    if (hw >= HW::Xe3p) return 4;
    if (hw == HW::XeHPC && stepping < SteppingPVCXTB4)
        return 128;
    return 64;
}

// Width alignment for a block 2D message.
static inline int block2DWidthAlignment(Type T, const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    // Block 2D width must be DW-aligned, but generally use QW alignment for better performance for reads.
    return ((astrategy.noExtraPad || block.writable || atype.alignment % 8) ? 4 : 8);
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
