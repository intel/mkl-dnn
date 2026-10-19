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
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


// Remove (sat) from an InstructionModifier.
static inline InstructionModifier unsaturated(InstructionModifier mod)
{
    if (mod.isSaturate())
        return mod ^ InstructionModifier::createSaturate();
    else
        return mod;
}

// Modify the signedness of an integer type.
static inline DataType withSignedness(DataType dt, bool signedType)
{
    switch (dt) {
        case DataType::b:
        case DataType::ub: return signedType ? DataType::b : DataType::ub;
        case DataType::w:
        case DataType::uw: return signedType ? DataType::w : DataType::uw;
        case DataType::d:
        case DataType::ud: return signedType ? DataType::d : DataType::ud;
        case DataType::q:
        case DataType::uq: return signedType ? DataType::q : DataType::uq;
        default: return dt;
    }
}

// Three-argument add.
template <HW hw>
template <typename DT, typename S0, typename S2>
void Generator<hw>::eadd3(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const S2 &src2, ngen::SourceLocation loc)
{
    if ((hw >= HW::XeHP) && !(dst.getOffset() & 1))
        add3<DT>(mod, dst, src0, src1, src2, loc);
    else {
        add<DT>(mod, dst, src1, src0, loc);
        add<DT>(mod, dst, dst, src2, loc);
    }
}

template <HW hw>
template <typename S0>
void Generator<hw>::ecsel(const InstructionModifier &mod, const InstructionModifier &cmod, const FlagRegister &flag,
                          const RegData &dst, const S0 &src0,
                          const RegData &src1, const RegData &src2, ngen::SourceLocation loc)
{
    if (dst.getByteOffset() & 7) {
        cmp(mod | cmod | flag, src2, 0, loc);
        sel(mod | ~flag, dst, src1, src0, loc);
    } else
        csel(mod | cmod | flag, dst, src0, src1, src2, loc);
};

template <HW hw>
template <typename DT>
void Generator<hw>::emov(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::RegData src0,
                         const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    EmulationImplementation::applyDefaultType<DT>(dst);
    EmulationImplementation::applyDefaultType<DT>(src0);

    if (dst.getType() == DataType::tf32 && src0.getType() == DataType::tf32) {
        dst.setType(DataType::f);
        src0.setType(DataType::f);
    }

    if (hw >= HW::XeHP && one_of(src0.getType(), {DataType::hf, DataType::f, DataType::bf})
            && src0.getType() == dst.getType()
            && ((src0.getHS() != dst.getHS()) || (src0.getOffset() != dst.getOffset()) || (src0.getHS() != 1 && getBytes(src0.getType()) == 2))) {
        moveToIntPipe(mod.getExecSize(), dst);
        moveToIntPipe(mod.getExecSize(), src0);
    }

    if (dst.getType() == DataType::f && src0.getType() == DataType::bf) {
        dst.setType(DataType::ud);
        src0.setType(DataType::uw);
        shl(mod, dst, src0, 16, loc);
    } else if (!strategy.systolicAvailable && dst.getType() == DataType::bf && src0.getType() == DataType::f) {
        // Emulated f32->bf16 RTNE conversion.
        auto flag = state.emulate.flag;
        if (!flag.isValid()) stub();
        dst.setType(DataType::uw);
        src0.setType(DataType::ud);
        add(mod, src0, src0, -0x8000, loc);
        and_(mod | nz | flag, null.ud(), src0, 0x1FFFF, loc);
        mov(mod, dst, EmulationImplementation::highWord(src0), loc);
        // add(mod, src0, src0, 0x8000);       // Preserve src0 -- if nondestructive mov -- not needed
        add(mod | flag, dst, dst, 1, loc);
    } else
        EmulationImplementation::emov(*this, mod, dst, src0, strategy.emulate, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    const bool is_xe3p = (hw == ngen::HW::Xe3p);
    const bool dstBf = dst.getType() == DataType::bf;
    const bool src1F = src1.getType() == DataType::f;
    // Xe3p has specific restrictions for mixed bf16/f32 mul.
    if (is_xe3p && (dstBf && src1F) && dst.getByteHS() != src1.getByteHS()){
        bool bcastSrc1 = src1.getHS() == 0 && src1.getVS() == 0;
        int tmp_elems = bcastSrc1 ? 1 : mod.getExecSize();
        auto tempRange = state.ra.alloc_range(div_up(tmp_elems, elementsPerGRF(hw, dst.getType())));
        auto tmp_mod = InstructionModifier(mod);
        tmp_mod.setExecSize(tmp_elems);
        auto temp = tempRange[0].sub(dst.getOffset(), dst.getType());
        mov(tmp_mod, temp(1), src1);
        mul(mod, dst, src0, temp(src1.getVS(), src1.getWidth(), src1.getHS()));
        state.ra.safeRelease(tempRange);
    } else {
        ngen::EmulationImplementation::emul<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc);
    }
}

template <HW hw>
template <typename DT>
void Generator<hw>::eadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1,
                         const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    if (dst.getType() == DataType::f && src0.getType() == DataType::f && src1.getType() == DataType::bf && src1.getHS() != 1) {
        GRF alloced, temp = state.emulate.temp[0];
        if (temp.isInvalid())
            temp = alloced = state.ra.alloc();

        auto src1UW = src1;
        src1UW.setType(DataType::uw);
        mov(mod, temp.uw(0)(1), src1UW, loc);
        add(mod, dst, src0, temp.bf(0)(1), loc);

        state.ra.safeRelease(alloced);
    } else {
        EmulationImplementation::eadd<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc);
    }
}

template <HW hw>
template <typename S0>
void Generator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, RegData src1, RegData src2,
                         const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    bool sub = false;
    if (src1.getNeg()) {
        src1 = -src1;
        sub = !sub;
    };
    if (src2.getNeg()) {
        src2 = -src2;
        sub = !sub;
    }
    emad(mod, dst, src0, src1, src2, strategy, state, sub, loc);
}

template <HW hw>
template <typename S0, typename S2>
void Generator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const S2 &src2,
                         const CommonStrategy &strategy, CommonState &state, bool sub, ngen::SourceLocation loc)
{
    auto dstType = dst.getType();
    if ((!sub && !(dst.getByteOffset() & 7) && !one_of(dstType, {DataType::q, DataType::uq}) && !one_of(src2.getType(), {DataType::d, DataType::ud}))
        || one_of(dstType, {DataType::hf, DataType::f, DataType::df})) {
        mad(mod, dst, src0, src1, src2, loc);
    } else {
        auto ttype = withSignedness(dst.getType(), isSigned(src1.getType()) || isSigned(src2.getType()));
        RegData temp;
        Subregister tempSub;
        RegisterRange tempRange;
        if (mod.getExecSize() == 1)
            temp = tempSub = state.ra.alloc_sub(ttype);
        else {
            tempRange = state.ra.alloc_range(div_up(mod.getExecSize(), elementsPerGRF(hw, ttype)));
            temp = tempRange[0].retype(ttype);
        }

        emul(unsaturated(mod), temp, src1, src2, strategy, state, loc);
        eadd(mod, dst, sub ? -temp : temp, src0, strategy, state, loc);

        state.ra.safeRelease(tempSub);
        state.ra.safeRelease(tempRange);
    }
}

template <HW hw>
template <typename S0>
void Generator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, const Immediate &src2,
                         const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    emad(mod, dst, src0, src1, src2, strategy, state, false, loc);
}

template <HW hw>
template <typename S0>
void Generator<hw>::emad(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, int32_t src2,
                         const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    auto dstType = dst.getType();
    if (src2 == 0)
        emov(mod, dst, src0, strategy, state, loc);
    else if (src2 == 1)
        eadd(mod, dst, src1, src0, strategy, state, loc);
    else if (!(dst.getByteOffset() & 7) && (src2 >= -0x8000 && src2 < 0x10000) && !one_of(dstType, {DataType::q, DataType::uq})) {
        mad(mod, dst, src0, src1, src2, loc);
    } else {
        auto ttype = (isSigned(src1.getType()) || src2 < 0) ? DataType::d : DataType::ud;
        Subregister tempScalar;
        RegisterRange tempGRFs;
        RegData temp;
        if (mod.getExecSize() == 1)
            temp = tempScalar = state.ra.alloc_sub(ttype);
        else {
            tempGRFs = state.ra.alloc_range(2);
            temp = tempGRFs[0].retype(ttype);
        }
        emulConstant(unsaturated(mod), temp, src1, src2, strategy, state, loc);
        eadd(mod, dst, temp, src0, strategy, state, loc);
        state.ra.safeRelease(tempScalar);
        state.ra.safeRelease(tempGRFs);
    }
}

template <HW hw>
template <typename S0>
void Generator<hw>::eaddScaled(const InstructionModifier &mod, const RegData &dst, const S0 &src0, const RegData &src1, Type src2,
                               const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    if (src2.is3()) {
        // u3 packs 8 elements into 3 bytes; byte offset = (elemIdx * 3) / 8.
        auto tmpRange = state.ra.alloc_range(2);
        auto tmp = tmpRange[0].retype(src1.getType());
        emulConstant(mod, tmp, src1, 3, strategy, state, loc);
        eshr(mod, tmp, tmp, 3, strategy, state, loc);
        eadd(mod, dst, tmp, src0, strategy, state, loc);
        state.ra.safeRelease(tmpRange);
    } else if (src2.isSubByte()) {
        auto tmpRange = state.ra.alloc_range(2);
        auto tmp = tmpRange[0].retype(src1.getType());
        eshr(mod, tmp, src1, src2.logPerByte(), strategy, state, loc);
        eadd(mod, dst, tmp, src0, strategy, state, loc);
        state.ra.safeRelease(tmpRange);
    } else
        emad(mod, dst, src0, src1, src2.size(), strategy, state, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emulConstant(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, Type src1,
                                 const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc)
{
    if (src1.isSubByte() && !src1.is3())
        eshr<DT>(mod, dst, src0, src1.logPerByte(), strategy, state, loc);
    else if (src1.is3()) {
        // u3 packs 8 elements into 3 bytes; byte size = (elemCount * 3) / 8.
        emulConstant<DT>(mod, dst, src0, 3, strategy, state, loc);
        eshr<DT>(mod, dst, dst, 3, strategy, state, loc);
    }
    else
        emulConstant<DT>(mod, dst, src0, src1.size(), strategy, state, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emath(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0,
                          const GEMMStrategy &strategy, CommonState &state, ngen::SourceLocation loc)
{
    if (hw == HW::XeHP && strategy.systolic && mod.getExecSize() <= 8) {
        // Workaround for DPAS + SIMD8 EM hang: use SIMD16 arithmetic.
        auto mod16 = mod;
        mod16.setExecSize(16);

        auto temp = state.ra.alloc_range(2);
        auto tt = temp[0].retype(src0.getType());

        mov(mod.getExecSize(), tt, src0, loc);
        math(mod16, fc, tt, tt, loc);
        mov(mod.getExecSize(), dst, tt, loc);

        state.ra.safeRelease(temp);
    } else
        math(mod, fc, dst, src0, loc);
}

template <HW hw>
void Generator<hw>::ejmpi(InstructionModifier mod, Label &dst, ngen::SourceLocation loc)
{
    if (hw >= HW::XeHPC && mod.getPredCtrl() == PredCtrl::anyv && !mod.isPredInv()) {
        mod.setPredCtrl(PredCtrl::Normal);
        jmpi(mod, dst, loc);
        auto flag = mod.getFlagReg();
        flag.setBase(flag.getBase() ^ 1);
        mod.setFlagReg(flag);
        jmpi(mod, dst, loc);
    } else
        jmpi(mod, dst, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emin_(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::Immediate &src1, CommonState &state, ngen::SourceLocation loc) {
        min_(mod, dst, src0, src1, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emin_(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, CommonState &state, ngen::SourceLocation loc) {
        min_(mod, dst, src0, src1, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emax_(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::Immediate &src1, CommonState &state, ngen::SourceLocation loc) {
        max_(mod, dst, src0, src1, loc);
}

template <HW hw>
template <typename DT>
void Generator<hw>::emax_(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, CommonState &state, ngen::SourceLocation loc) {
        max_(mod, dst, src0, src1, loc);
}

GEMMSTONE_NAMESPACE_END
