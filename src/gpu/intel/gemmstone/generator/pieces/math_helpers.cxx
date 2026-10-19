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


#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


// Scale then add: dst <- src0 + src1 * (numerator / denominator), rounding up.
// If exact = true, ensure src1 * num / denom is integral if src1 immediate.
template <HW hw>
void Generator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, int src0, const RegData &src1,
                              int numerator, int denominator, CommonState &state, bool exact)
{
    if (!is_zero_or_pow2(denominator)) stub();

    if (!is_zero_or_pow2(numerator)) {
        // Non-power-of-2 numerator (e.g. u3's 3-bit packing): use a real multiply
        // followed by a power-of-2 shift instead of the shift-only fast paths below.
        auto temp = state.ra.alloc_sub(src1.getType());
        mulConstant(mod, temp, src1, numerator);
        if (!exact) add(mod, temp, temp, denominator - 1);
        asr(mod, temp, temp, ilog2(denominator));
        (src0 != 0) ? add(mod, dst, temp, src0) : mov(mod, dst, temp);
        state.ra.safeRelease(temp);
        return;
    }

    if (numerator == denominator) {
        (src0 != 0)   ? add(mod, dst, src1, src0) :
        (src1 != dst) ? mov(mod, dst, src1)
                      : noop();
    } else if (numerator > denominator) {
        (src0 == 0) ? mulConstant(mod, dst, src1, numerator / denominator)
                    : mad(mod, dst, src0, src1, numerator / denominator);
    } else if ((numerator * 2) == denominator)
        avg(mod, dst, src1, src0 * 2);
    else {
        add(mod, dst, src1, ((src0 + 1) * denominator / numerator) - 1);
        asr(mod, dst, dst, ilog2(denominator) - ilog2(numerator));
    }
}

template <HW hw>
void Generator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1,
                              int numerator, int denominator, CommonState &state, bool exact)
{
    if (!is_zero_or_pow2(denominator)) stub();

    if (!is_zero_or_pow2(numerator)) {
        // Non-power-of-2 numerator (e.g. u3's 3-bit packing): use a real multiply
        // followed by a power-of-2 shift instead of the shift-only fast paths below.
        auto temp = state.ra.alloc_sub(src1.getType());
        mulConstant(mod, temp, src1, numerator);
        if (!exact) add(mod, temp, temp, denominator - 1);
        asr(mod, temp, temp, ilog2(denominator));
        add(mod, dst, temp, src0);
        state.ra.safeRelease(temp);
        return;
    }

    if (numerator == denominator)
        add(mod, dst, src1, src0);
    else if (numerator > denominator)
        mad(mod, dst, src0, src1, numerator / denominator);
    else {
        auto temp = state.ra.alloc_sub(src1.getType());
        if (exact)
            asr(mod, temp, src1, ilog2(denominator) - ilog2(numerator));
        else {
            add(mod, temp, src1, (denominator / numerator) - 1);
            asr(mod, temp, temp, ilog2(denominator) - ilog2(numerator));
        }
        add(mod, dst, temp, src0);
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
void Generator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, const RegData &src0, int src1,
                              int numerator, int denominator, CommonState &state, bool exact)
{
    if (!is_zero_or_pow2(denominator)) stub();
    if (exact && ((numerator * src1) % denominator))
        stub("Misaligned immediate value.");
    add(mod, dst, src0, (numerator * src1) / denominator);
}

template <HW hw>
template <typename S0, typename S1>
void Generator<hw>::addScaled(const InstructionModifier &mod, const RegData &dst, S0 src0, S1 src1,
                              Type T, CommonState &state, bool exact, int scale)
{
    addScaled(mod, dst, src0, src1, T.bits() * scale, 8, state, exact);
}

// Multiply by a constant, optimizing for power-of-2 constants.
template <HW hw>
template <typename DT>
void Generator<hw>::mulConstant(const InstructionModifier &mod, const RegData &dst, const RegData &src0, int32_t src1)
{
    if (src1 == 0)
        mov<DT>(mod, dst, uint16_t(0));
    else if (src1 == 1) {
        if (dst != src0) mov<DT>(mod, dst, src0);
    } else if (src1 == -1)
        mov<DT>(mod, dst, -src0);
    else if (is_zero_or_pow2(src1)) {
        shl<DT>(mod, dst, src0, uint16_t(ilog2(src1)));
    } else if (src1 >= 0x10000)
        mul<DT>(mod, dst, src0, uint32_t(src1));
    else if (src1 < -0x8000)
        mul<DT>(mod, dst, src0, int32_t(src1));
    else if (src1 > 0)
        mul<DT>(mod, dst, src0, uint16_t(src1));
    else
        mul<DT>(mod, dst, src0, int16_t(src1));
}

// Modulo by constant value.
template <HW hw>
template <typename DT>
void Generator<hw>::mod(const InstructionModifier &finalMod, const Subregister &dst, const Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state)
{
    if (finalMod.getExecSize() > 1) stub();
    if (is_zero_or_pow2(modulus))
        and_<DT>(finalMod, dst, src, modulus - 1);
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(finalMod, MathFunction::irem, dst, src, modulus);
    else {
        auto temp = dst;
        if (src == dst)
            temp = state.ra.alloc_sub<uint32_t>();
        alignDown<DT>(temp, src, modulus, strategy, state);
        add<DT>(finalMod, dst, src, -temp);
        if (src == dst)
            state.ra.safeRelease(temp);
    }
}

template <HW hw>
template <typename DT>
void Generator<hw>::mod(const Subregister &dst, const Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state)
{
    mod(1, dst, src, modulus, strategy, state);
}

// Return both (a % b) and a - (a % b).
template <HW hw>
template <typename DT>
void Generator<hw>::modExt(const Subregister &dstMod, const Subregister &dstMultiple, const Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state)
{
    if (is_zero_or_pow2(modulus)) {
        and_<DT>(1, dstMultiple, src, ~uint32_t(modulus - 1));
        and_<DT>(1, dstMod, src, modulus - 1);
    } else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP)) {
        math<DT>(1, MathFunction::irem, dstMod, src, modulus);
        add<DT>(1, dstMultiple, src, -dstMod);
    } else {
        alignDown<DT>(dstMultiple, src, modulus, strategy, state);
        add<DT>(1, dstMod, src, -dstMultiple);
    }
}

// Divide an unsigned value by a constant, rounding down.
template <HW hw>
template <typename DT>
void Generator<hw>::divDown(const ngen::Subregister &dst, const ngen::Subregister &src, uint32_t divisor, const CommonStrategy &strategy, CommonState &state)
{
    if (is_zero_or_pow2(divisor))
        shr<DT>(1, dst, src, ilog2(divisor));
    else if (strategy.emulate.emulate64 && (hw <= HW::Gen12LP))
        math<DT>(1, MathFunction::iqot, dst, src, divisor);
    else {
        // Replace integer division with multiplication by reciprocal + shift.
        // Valid for numerators <= 2^31.
        int shift = ngen::utils::bsr(divisor);
        auto recip32 = uint32_t(((uint64_t(0x100000000) << shift) + divisor - 1) / divisor);
        if (!strategy.emulate.emulate64_mul) {
            auto tmp = state.ra.alloc_sub<uint64_t>();
            mul(1, tmp, src, recip32);
            shr(1, dst, tmp.ud(1), shift);
            state.ra.safeRelease(tmp);
        } else {
            emul32High(1, dst, src, recip32);
            shr(1, dst, dst, shift);
        }
    }
}

// Divide an unsigned value by a constant, rounding up.
// Default algorithm: dst = (src / divisor) + (src % divisor > 0)
// Relax algorithm (can overflow): dst = (src + divisor - 1) / divisor
template <HW hw>
template <typename DT>
void Generator<hw>::divUp(const Subregister &dst, const Subregister &src, uint32_t divisor, const CommonStrategy &strategy, CommonState &state, bool relax)
{
    if (relax) {
        add<DT>(1, dst, src, divisor - 1);
        divDown<DT>(dst, dst, divisor, strategy, state);
    } else {
        auto flag = state.ra.alloc_flag();
        mod(1 | ne | flag, null.ud(0), src, divisor, strategy, state);
        divDown<DT>(dst, src, divisor, strategy, state);
        add(1 | flag, dst, dst, 1);
        state.ra.safeRelease(flag);
    }
}


// Align an unsigned value down to a multiple of align.
template <HW hw>
template <typename DT>
void Generator<hw>::alignDown(const InstructionModifier &mod, const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state)
{
    if (is_zero_or_pow2(align))
        and_<DT>(mod, dst, src, uint32_t(-align));
    else {
        divDown(dst, src, align, strategy, state);
        mul(mod, dst, dst, align);
    }
}

template <HW hw>
template <typename DT>
void Generator<hw>::alignDown(const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state)
{
    alignDown(1, dst, src, align, strategy, state);
}

// Align an unsigned value up to a multiple of align.
template <HW hw>
template <typename DT>
void Generator<hw>::alignUp(const Subregister &dst, const Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state)
{
    add<DT>(1, dst, src, uint16_t(align - 1));
    alignDown<DT>(dst, dst, align, strategy, state);
}

// Non-constant integer division.
// Requires an auxiliary constant: ceiling(2^(32 + s) / denom), where s = floor(log2(denom)).
template <HW hw>
template <typename DT>
void Generator<hw>::divDown(const Subregister &dst, const Subregister &src0, const Subregister &src1, const Subregister &src1Recip, const FlagRegister &flag, const CommonStrategy &strategy, CommonState &state)
{
    bool emulate = strategy.emulate.emulate64_mul;
    Subregister tmp;
    auto shift = state.ra.alloc_sub<uint32_t>();
    auto pop = state.ra.alloc_sub<uint32_t>();
    cbit(1, pop, src1);
    fbh(1, shift, src1);
    cmp(1 | gt | flag, pop, 1);
    add(1, shift, -shift, 31);
    if (emulate)
        emul32High(1 | flag, dst, src0, src1Recip);
    else {
        tmp = state.ra.alloc_sub<uint64_t>();
        mul(1 | flag, tmp, src0, src1Recip);
    }
    shr(1 | ~flag, dst, src0, shift);
    shr(1 | flag, dst, emulate ? dst : tmp.ud(1), shift);
    state.ra.safeRelease(shift);
    state.ra.safeRelease(pop);
    state.ra.safeRelease(tmp);
}

template <HW hw>
template <typename DT>
void Generator<hw>::divUp(const Subregister &dst, const Subregister &src0, const Subregister &src1, const Subregister &src1Recip, const FlagRegister &flag, const CommonStrategy &strategy, CommonState &state)
{
    auto adj = state.ra.alloc_sub<uint32_t>();
    eadd3(1, adj, src0, src1, -1);
    divDown(dst, adj, src1, src1Recip, flag, strategy, state);
    state.ra.safeRelease(adj);
}

// Unsigned integer division + modulo with no precomputation.
//
// Default algorithm (large = false):
//    num, denom < 2^16   OR  denom < 0x40, quotient < 2^16
// Expanded range (large = true):
//    denom <= 2^22, quotient < 2^16
template <HW hw>
void Generator<hw>::divMod(const Subregister &qot, const Subregister &rem,
                           const Subregister &num, const Subregister &denom,
                           const GEMMStrategy &strategy, CommonState &state, bool large)
{
    if (qot.getType() != DataType::ud || rem.getType() != DataType::ud) stub();

    auto numFP = state.ra.alloc_sub<float>();
    auto denomFP = state.ra.alloc_sub<float>();

    if (large) {
        or_(1, cr0[0], cr0[0], 0x20);               // round toward -inf
        mov(1, denomFP, denom);
        mov(1, numFP, -num);
        einv(1, denomFP, denomFP, strategy, state);
        add(1, denomFP.ud(), denomFP.ud(), 2);
        mul(1, qot.f(), -numFP, denomFP);
        mov(1, qot, qot.f());
        mad(1 | lt | f1[1], rem.d(), num, denom, -qot.uw());
        add(1 | f1[1], rem, rem, denom);
        add(1 | f1[1], qot, qot, -1);
        and_(1, cr0[0], cr0[0], ~0x30);
    } else {
        auto bias = state.ra.alloc_sub<float>();
        mov(1, denomFP, denom);
        mov(1, numFP, num);
        mov(1, bias, -0.499996185302734375f);       // -1/2 + 2^(-18)
        einv(1, denomFP, denomFP, strategy, state);
        add(1, denomFP.ud(), denomFP.ud(), 2);
        mad(1, qot.f(), bias, numFP, denomFP);
        mov(1, qot, qot.f());
        mad(1, rem, num, -qot.uw(), denom.uw());
        state.ra.safeRelease(bias);
    }

    state.ra.safeRelease(numFP);
    state.ra.safeRelease(denomFP);
}

GEMMSTONE_NAMESPACE_END
