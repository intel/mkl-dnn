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


#include "copy_plan.hpp"
#include "bfn.hpp"
#include "internal/utils.hpp"
#include "ngen_object_helpers.hpp"

#include <algorithm>
#include <numeric>

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;


/****************/
/* Pseudo-types */
/****************/

static constexpr ngen::DataType ngen_b16_l2x() { return static_cast<ngen::DataType>(0x97); }   /* 16 bits with data in low 2 bits + possible junk */
static constexpr ngen::DataType ngen_b16_h2x() { return static_cast<ngen::DataType>(0x9F); }   /* 16 bits with data in high 2 bits + possible junk */
static constexpr ngen::DataType ngen_b16_l4x() { return static_cast<ngen::DataType>(0x95); }   /* 16 bits with data in low 4 bits + possible junk */
static constexpr ngen::DataType ngen_b16_h4x() { return static_cast<ngen::DataType>(0x9D); }   /* 16 bits with data in high 4 bits + possible junk */
static constexpr ngen::DataType ngen_b16()     { return static_cast<ngen::DataType>(0x90); }   /* 16 bits holding a smaller data type */

/********************/
/* Utility routines */
/********************/

static bool isBitwise(Opcode op) {
    return one_of(op, {Opcode::mov, Opcode::and_, Opcode::or_, Opcode::xor_});
}
static bool isBroadcast(const CopyOperand &op) {
    return (op.kind != op.GRF) || (op.stride == 0);
}

// Check if a CopyOperand spans multiple registers.
static bool multiGRF(HW hw, const CopyInstruction &i, const CopyOperand &op)
{
    if (op.kind != op.GRF) return false;
    return elementsToBytes(op.offset + op.stride * (i.simd - 1), op.type) >= GRF::bytes(hw);
}

// Check if two CopyOperands may overlap.
static bool mayOverlap(HW hw, const CopyInstruction &i, const CopyOperand &op1, const CopyOperand &op2)
{
    if (!op1 || !op2) return false;
    if (op1.kind != op2.kind) return false;
    if (op1.temp != op2.temp) return false;
    if (op1.temp && op1.value != op2.value) return false;

    if (op1.kind == CopyOperand::Flag) return op1.grf == op2.grf;
    if (op1.kind != CopyOperand::GRF) return false;

    int bs1 = op1.byteStride();
    int bs2 = op2.byteStride();
    int boStart1 = op1.absByteOffset(hw);
    int boStart2 = op2.absByteOffset(hw);
    int boEnd1 = boStart1 + bs1 * i.simd;
    int boEnd2 = boStart2 + bs2 * i.simd;

    if (boEnd2 <= boStart1 || boEnd1 <= boStart2) return false;
    if (bs1 != bs2) return true;

    int slotA = boStart1 % bs1;
    int slotB = boStart2 % bs1;
    auto dbytesA = getBytes(op1.type);
    auto dbytesB = getBytes(op2.type);

    if (slotA > slotB) {
        std::swap(slotA, slotB);
        std::swap(dbytesA, dbytesB);
    }

    if (slotB + dbytesB > bs1) return true;
    if (slotA + dbytesA > slotB) return true;
    return false;
}

// Return a DataType representing the potential numerical range of
//  a conversion from one data type to another.
static DataType conversionRange(DataType from, DataType to)
{
    return (getBytes(from) < getBytes(to)) ? from : to;
}

// Check if one data type (dt1) is a subset of another (dt2).
static bool isSubsetOf(DataType dt1, DataType dt2)
{
    if (dt1 == DataType::invalid || dt2 == DataType::invalid) return false;
    if (dt1 == dt2) return true;
    if (isFP(dt1) && isInt(dt2)) return false;
    if (isW(dt1) && dt2 == DataType::tf32) return false;
    if (isSubByteInt(dt1) && (isB(dt2) || dt2 == DataType::hf8)) return true;
    if (dt1 == DataType::s2 && dt2 == DataType::bf8) return true;
    if (dt1 == DataType::s4 && dt2 == DataType::bf8) return true;
    if (dt1 == DataType::e2m1 && isFP8(dt2)) return true;
    return getBytes(dt1) < getBytes(dt2);
}


/***********************/
/* CopyOperand methods */
/***********************/

CopyOperand::CopyOperand(RegData rd)
        : grf(rd.getBase()), offset(rd.getLogicalOffset()),
          stride(rd.getHS()), type(rd.getType()), kind(GRF),
          overwrite(false), overwriteStride(false), neg(rd.getNeg()), abs(rd.getAbs())
{
    if (rd.getAbs()) stub("Unsupported modifier");
    if (rd.getVS() != 0 || rd.getWidth() != 0)
        if (rd.getVS() != rd.getWidth() * stride)
            vs = rd.getVS(), width = rd.getWidth();
}

CopyOperand CopyOperand::operator-() const
{
    auto clone = *this;
    clone.neg = !clone.neg;
    return clone;
}

bool CopyOperand::operator==(const CopyOperand &op) const {
    bool ok = true;
    if (kind != op.kind) return false;
    if (kind == Null) return true;
    ok &= type == op.type
       && (!temp || value == op.value);
    if (kind == Immediate) return ok;
    ok &= temp   == op.temp
       && grf    == op.grf
       && offset == op.offset
       && neg    == op.neg;
    if (kind == Flag) return ok;
    ok &= stride == op.stride
       && vs     == op.vs
       && width  == op.width
       && abs    == op.abs;
    return ok;
}

// Convert a GRF CopyOperand to an nGEN object.
RegData CopyOperand::ngen() const
{
    if (kind == Null)
        return ngen::NullRegister().sub(offset, type)(stride);
    if (kind != GRF || temp) stub("Invalid operation");

    auto sub = ngen::GRF(grf).sub(offset, type);
    RegData rd;
    if (width)
        rd = sub(vs, width, stride);
    else
        rd = sub(stride);
    if (neg) rd = -rd;
    if (abs) rd = ngen::abs(rd);

    return rd;
}

// Convert an immediate CopyOperand to an nGEN object.
Immediate CopyOperand::ngenImmediate() const
{
    if (kind != Immediate) stub("Invalid operation");
    ngen::Immediate imm = value;
    imm.setType(type);
    return imm;
}

// Convert a flag CopyOperand to an nGEN object.
FlagRegister CopyOperand::ngenFlag() const
{
    if (kind != Flag || temp) stub("Invalid operation");
    auto flag = FlagRegister::createFromIndex(grf + (offset >> 4));
    flag.setType(type);
    if (neg) flag = ~flag;
    return flag;
}

/***************************/
/* CopyInstruction methods */
/***************************/

// Move an instruction to the integer pipe if possible.
void CopyInstruction::moveToIntegerPipe()
{
    auto &st = src0.type, &dt = dst.type;

    if (op != Opcode::mov) return;
    if (asSigned(st) != asSigned(dt)) return;
    if (src0.neg) return;
    if (sat) return;

    switch (getBits(st)) {
        case 2: st = dt = DataType::u2; break;
        case 4: st = dt = DataType::u4; break;
        case 8: st = dt = DataType::ub; break;
        case 16: st = dt = DataType::uw; break;
        case 32: st = dt = DataType::ud; break;
        case 64:
            if (src0.stride == 1 && dst.stride == 1) {
                st = dt = DataType::ud;
                simd *= 2;
                src0.offset *= 2;
                dst.offset *= 2;
            } else
                st = dt = DataType::uq;
            break;
        default: break;
    }
}

// Retrieve nGEN instruction modifiers for an instruction.
InstructionModifier CopyInstruction::ngenModifiers() const
{
    InstructionModifier mod = simd;
    mod |= cmod;
    if (flag) {
        mod |= flag.ngenFlag();
        mod |= InstructionModifier::createChanOff(flag.offset & 0xF);
    }
    if (atomic) mod |= ThreadCtrl::Atomic;
    if (sat) mod |= InstructionModifier::createSaturate();
    return mod;
}

/********************/
/* CopyPlan methods */
/********************/

// Run all transformation passes on a CopyPlan.
void CopyPlan::transform()
{
    distributePhases();
    planEarlySubByteIntUpconversions();
    split2DRegions();
    planEarlyInt3Upconversions();

    sort(SortType::Register);

    optimizeTranspose();
    optimizeIntegerDownconvert();
    optimizeZip();
    optimizeZipAdjacent();
    optimizeMoveToIntPipe();
    optimizeWidenIntegers();
    optimizeConcatenate(true);

    legalizeSIMD(true);
    planTypeConversions();
    planBFNEmulation();

    sort(SortType::Register);

    optimizeZip();
    optimizeZipAdjacent();
    optimizeWidenIntegers();
    optimizeConcatenate();

    legalizeSIMD();

    sort(SortType::Register);     /* for nicer temporary numbering; not required */

    legalizeRegions();
    legalizeNegation();
    legalizeSIMD();
    optimizeSaturate();

    sort(SortType::SourceOrder);

    optimizeZip(true);
    optimizeWriteCombine();
    optimizeWriteSpread();

    legalizeImmediateTypes();

    sort(SortType::PhaseOnly);

    legalizeShfl();

#ifndef STANDALONE
#if GEMMSTONE_ENABLE_COPY_PLAN_DUMP
    const auto verbose = getVerbose(GEMMVerbose::DebugInfo);
    if (verbose >= 170)
        dump(verbose >= 180);
#endif
#endif
}



/* Basic operations on copy plans. */
CopyInstruction &CopyPlan::append(CopyInstruction &&i)
{
    auto offset = insns.empty() ? 0 : insns.back().range.end + 1;
    i.range.start = offset;
    i.range.end = offset + i.simd - 1;
    insns.push_back(std::move(i));
    return insns.back();
}

CopyInstruction &CopyPlan::append(int phase, Opcode op, int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    bool sat = mod.isSaturate();

    InstructionModifier mmod{};
    if (sat) mmod |= InstructionModifier::createSaturate();

    if (mod.getAll() != mmod.getAll()) stub("Unsupported instruction modifiers");

    CopyInstruction i;
    i.op = op;
    i.simd = simd;
    i.dst = dst;
    i.src0 = src0;
    i.src1 = src1;
    i.src2 = src2;
    i.sat = sat;
    i.phase = phase;
    return append(std::move(i));
}

CopyInstruction &CopyPlan::append(int phase, Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(phase, op, simd, InstructionModifier{}, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::append(Opcode op, int simd, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(0, op, simd, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::append(Opcode op, int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, const CopyOperand &src1, const CopyOperand &src2)
{
    return append(0, op, simd, mod, dst, src0, src1, src2);
}

CopyInstruction &CopyPlan::appendDestructiveMov(int simd, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride)
{
    return appendDestructiveMov(simd, InstructionModifier{}, dst, src0, overwriteStride);
}

CopyInstruction &CopyPlan::appendDestructiveMov(int simd, InstructionModifier mod, const CopyOperand &dst, const CopyOperand &src0, bool overwriteStride)
{
    auto &i = append(Opcode::mov, simd, mod, dst, src0);
    i.src0.overwrite = true;
    i.src0.overwriteStride = overwriteStride;
    return i;
}

CopyOperand CopyPlan::newTemp(DataType type, int elems, int stride, int align, int offset)
{
    int grf = GRF::bytes(hw);
    auto bytes = elementsToBytes(elems * stride, type);

    if (align == 0) align = grf;
    int soffset = (offset / stride) * stride;

    if (soffset > 0 && soffset + bytes > grf)
        stub("Misaligned multi-GRF temporary");

    CopyOperand op{};
    op.grf = 0;
    op.type = type;
    op.offset = offset;
    op.stride = stride;
    op.kind = op.GRF;
    op.temp = true;
    op.value = temps.size();

    temps.emplace_back(bytes, align, offset);

    return op;
}

CopyOperand CopyPlan::newFlag(int bits)
{
    CopyOperand op{};
    op.grf = 0;
    op.kind = op.Flag;
    op.value = temps.size();
    op.temp = true;

    temps.push_back(CopyTemporary::createFlag(bits));

    return op;
}

CopyTemporary CopyTemporary::createFlag(int bits)
{
    CopyTemporary temp;
    if (bits > 32) stub();
    temp.bytes = (bits > 16) ? 4 : 2;
    temp.flag = true;
    return temp;
}

int CopyPlan::tempFlagBytes() const
{
    int bytes = 0;
    for (const auto &t: temps)
        if (t.flag)
            bytes += t.bytes;
    return bytes;
}

CopyOperand CopyPlan::getResource(CopyResource::Kind kind)
{
    CopyResource *res = nullptr;
    if (kind == CopyResource::Kind::null)
        return CopyOperand();
    for (auto &r: resources) if (r.kind == kind) {
        res = &r; break;
    }
    if (!res) {
        resources.push_back(kind);
        res = &resources.back();
    }
    if (!res->src) {
        std::array<uint8_t, 64> data;
        res->preinitialized = false;
        if (int n = res->getData(data))
            res->src = newTemp(DataType::ud, (n + 3) >> 2, 1);
    }
    return res->src;
}

// Split an instruction into two.
//   If sequenced is true (default), the two instructions depend on each other
//   and should be spaced apart.
// After splitting, mergeChanges must be applied to incorporate the new instruction.
CopyInstruction &CopyPlan::split(CopyInstruction &i, bool sequenced)
{
    newInsns.emplace_back(i);
    auto &clone = newInsns.back();

    if (sequenced) {
        if (i.spread == 0) stub("Too many splits");
        i.spread >>= 1;
        clone.spread >>= 1;
        i.phase -= i.spread;
        clone.phase += i.spread;
    }

    return clone;
}

// Split an instruction into n instructions.
// After splitting, mergeChanges must be applied to incorporate the new instruction(s).
template <int n>
std::array<CopyInstruction*, n> CopyPlan::splitMultiple(CopyInstruction &i)
{
    std::array<CopyInstruction*, n> result;

    i.phase -= i.spread;
    i.spread /= n;
    i.phase += i.spread;

    newInsns.reserve(newInsns.size() + n - 1);
    result[0] = &i;
    for (int j = 1; j < n; j++) {
        newInsns.emplace_back(i);
        result[j] = &newInsns.back();
        result[j]->phase += 2 * j * i.spread;
    }

    return result;
}

// Join two instructions.
// The second instruction will be marked for removal, but not removed until
//   a call to mergeChanges.
CopyInstruction &CopyPlan::join(CopyInstruction &i1, CopyInstruction &i2, int maxGap)
{
    // Reorder ranges to be adjacent, if possible, to reduce temporary usage.
    CopyInstruction *ifirst = nullptr, *ilast = nullptr;
    if (i1.range.end < i2.range.start)
        ifirst = &i1, ilast = &i2;
    else if (i2.range.end < i1.range.start)
        ifirst = &i2, ilast = &i1;

    if (ifirst && ilast) {
        bool gapTooLarge = (ifirst->range.end + maxGap + 1 < ilast->range.start);
        if (!freezeRange && trySwapRanges(ilast->range, ifirst->range.end + 1))
            gapTooLarge = false;
        if (gapTooLarge)
            return invalidInsn;
    }

    i1.range |= i2.range;
    i2.invalidate();

    return i1;
}

// Try to swap ranges so that the range [min0, max0] is moved to start at min1.
// Returns true if successful.
bool CopyPlan::trySwapRanges(const CopyRange &range, int min1)
{
    const auto &min0 = range.start;
    const auto &max0 = range.end;
    int max1 = min1 + max0 - min0;
    if (max0 >= min1 && max1 >= min0) return false;       /* ranges overlap */

    // Check validity of swap.
    auto moveOK = [](CopyInstruction &i, int minN, int maxN) -> bool {
        if (i.range.start < minN && i.range.end >= minN) return false;
        if (i.range.end > maxN && i.range.start <= maxN) return false;
        return true;
    };

    for (auto &i: insns) {
        if (!moveOK(i, min0, max0)) return false;
        if (!moveOK(i, min1, max1)) return false;
    }

    for (auto &i: newInsns) {
        if (!moveOK(i, min0, max0)) return false;
        if (!moveOK(i, min1, max1)) return false;
    }

    // Execute swap.
    int diff = min1 - min0;
    auto swap = [=](CopyInstruction &i) {
        if (i.range.start >= min0 && i.range.end <= max0)
            i.range.start += diff, i.range.end += diff;
        else if (i.range.start >= min1 && i.range.end <= max1)
            i.range.start -= diff, i.range.end -= diff;
    };

    for (auto &i: insns)    swap(i);
    for (auto &i: newInsns) swap(i);
    return true;
}

// Update all pending instruction insertions/removals.
void CopyPlan::mergeChanges()
{
    insns.insert(insns.end(), newInsns.begin(), newInsns.end());
    newInsns.clear();

    for (auto iter = insns.begin(); iter != insns.end(); ) {
        if (iter->isInvalid())
            iter = insns.erase(iter);
        else
            iter++;
    }
}

// Add an intermediate copy through the given type.
//   If stride != 0, require the given stride for the intermediate result.
//   If strideOff0 == true, require the intermediate result to have offset % stride = 0.
void CopyPlan::copyThrough(CopyInstruction &i, DataType type, int stride, bool strideOff0, bool movAfter)
{
    auto st = i.src0.type, dt = i.dst.type;
    auto sstride = i.src0.stride, dstride = i.dst.stride;
    auto ssize = getBits(st), dsize = getBits(dt), isize = getBits(type);

    auto &i0 = i, &i1 = split(i);

    auto inplaceSrc = (stride == 0) ? (ssize >= isize && i.src0.overwrite)
                                        || (ssize * sstride >= isize && i.src0.overwriteStride)
                                    : (ssize * sstride == isize * stride && i.src0.overwrite)
                                        && (isize <= ssize || i.src0.overwriteStride);
    auto inplaceDst = (stride == 0) ? (dsize >= isize)
                                        || (dsize * dstride >= isize && i.dst.overwriteStride)
                                    : (dsize * dstride == isize * stride)
                                        && (isize <= dsize || i.dst.overwriteStride);

    if (strideOff0) {
        inplaceSrc &= i.src0.stride > 0 && (i.src0.offset % i.src0.stride) == 0;
        inplaceDst &= i.dst.stride > 0  && (i.dst.offset  % i.dst.stride)  == 0;
    }

    if (i.src1) inplaceDst &= !mayOverlap(hw, i, i.src1, i.dst);
    if (i.src2) inplaceDst &= !mayOverlap(hw, i, i.src2, i.dst);

    if (inplaceSrc && inplaceDst)
        inplaceSrc = isFP(st) && !isFP(dt);     /* prioritize in-place on floating point types */

    if (inplaceSrc) {
        // Convert src0 in place
        i0.op = Opcode::mov;
        i0.dst = i0.src0;
        i0.dst.offset = (i0.dst.offset * ssize) / isize;
        i0.dst.stride = (i0.dst.stride * ssize) / isize;
        i0.flag = i0.src1 = i0.src2 = CopyOperand();
        i0.cmod = ConditionModifier::none;
        i1.src0 = i0.dst;
    } else if (inplaceDst) {
        // Convert dst in place
        i1.op = Opcode::mov;
        i1.src0 = i1.dst;
        i1.src0.offset = (i1.src0.offset * dsize) / isize;
        i1.src0.stride = (i1.src0.stride * dsize) / isize;
        i1.flag = i1.src1 = i1.src2 = CopyOperand();
        i1.cmod = ConditionModifier::none;
        i0.dst = i1.src0;
    } else {
        // No space for in-place conversion -- create temporary.
        if (stride == 0)
            stride = std::max(1, ssize * sstride / isize);
        i0.dst = newTemp(type, i.simd, stride);
        i1.src0 = i0.dst;
        i1.src0.overwriteStride = true;

        auto &im = movAfter ? i1 : i0;
        im.op = Opcode::mov;
        im.flag = im.src1 = im.src2 = CopyOperand();
        im.cmod = ConditionModifier::none;
    }
    i1.src0.overwrite = true;
    i0.dst.type = i1.src0.type = type;
    i0.moveToIntegerPipe();
    i1.moveToIntegerPipe();
    if (i0.op == Opcode::mov) {
        auto srange = i0.src0.range;
        if (srange == DataType::invalid)
            srange = i0.src0.type;
        i1.src0.range = conversionRange(srange, i0.dst.type);
    }

    auto needsSaturate = [](const CopyOperand &src, const CopyOperand &dst) {
        auto st = src.range == DataType::invalid ? src.type : src.range;
        auto dt = dst.type;
        if (!isInt(st) || !isInt(dt)) return false;
        return !isSubsetOf(st, dt);
    };

    auto setNullDstRegioning = [](CopyOperand &dst, const CopyOperand &src) {
        if (!dst.isNull()) return;
        dst.stride = src.width == 1 ? src.vs : src.stride;
        dst.offset = src.offset;
        dst.width = dst.vs = 0;
        dst.type = src.type;
    };

    setNullDstRegioning(i0.dst, i0.src0);
    setNullDstRegioning(i1.dst, i1.src0);

    if (needsSaturate(i0.src0, i0.dst))
        i0.sat = true;
    if (needsSaturate(i1.src0, i1.dst))
        i1.sat = true;
}

// Adjust stride on src0.
void CopyPlan::restrideSrc0(CopyInstruction &i, int stride, bool strideOff0)
{
    copyThrough(i, i.src0.type, stride, strideOff0);
}

// Adjust stride on dst.
void CopyPlan::restrideDst(CopyInstruction &i, int stride, bool strideOff0)
{
    copyThrough(i, i.dst.type, stride, strideOff0, true);
}

// Change src0/1/2 region.
void CopyPlan::repositionSrc(CopyInstruction &i, int n, int stride, int offset)
{
    if (n < 0 || n > 2) stub();
    auto CopyInstruction::* src = (n == 0) ? &CopyInstruction::src0 :
                                  (n == 1) ? &CopyInstruction::src1 :
                                             &CopyInstruction::src2;
    auto &op = i.*src;
    auto type = op.type;
    if (type == DataType::v) type = DataType::w;
    if (type == DataType::uv) type = DataType::uw;
    auto bytes = getBytes(type);
    auto abs = op.abs;

    bool inplaceDst = i.dst
                   && stride * bytes == i.dst.byteStride()
                   && offset * bytes == i.dst.byteOffset()
                   && (bytes <= getBytes(i.dst.type) || i.dst.overwriteStride);

    if (n != 0) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src0);
    if (n != 1) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src1);
    if (n != 2) inplaceDst &= !mayOverlap(hw, i, i.dst, i.src2);

    auto &i0 = i, &i1 = split(i);

    i0.op = Opcode::mov;
    i0.src0 = i0.*src;
    i0.src0.abs = false;
    if (inplaceDst) {
        i0.dst.type = type;
        i0.dst.stride = stride;
        i0.dst.offset = offset;
    } else {
        i0.dst = newTemp(type, i.simd, stride, 0, offset);
        i0.dst.overwriteStride = true;
    }
    i0.dst.neg = i0.src0.neg;
    i0.src1 = i0.src2 = i0.flag = CopyOperand{};

    i1.*src = i0.dst;
    (i1.*src).overwrite = true;
    (i1.*src).abs = abs;

    if (!i1.dst) {
        i1.dst.stride = i1.src0.width == 1 ? i1.src0.vs : i1.src0.stride;
        i1.dst.offset = i1.src0.offset;
        i1.dst.width = i1.dst.vs = 0;
        i1.dst.type = i1.src0.type;
    }

    i0.cmod = ConditionModifier::none;
    i0.moveToIntegerPipe();
}

// Change dst region.
void CopyPlan::repositionDst(CopyInstruction &i, int stride, int offset)
{
    auto &i0 = i, &i1 = split(i);

    i0.dst = newTemp(i.dst.type, i.simd, stride, 0, offset);

    i1.op = Opcode::mov;
    i1.src0 = i0.dst;
    i1.src0.overwrite = true;
    i1.src0.overwriteStride = true;
    i1.src1 = i1.src2 = i1.flag = CopyOperand{};
    i1.cmod = ConditionModifier::none;
    i1.moveToIntegerPipe();
}

// Pass to split 2D regioned instructions into 1D regions.
void CopyPlan::split2DRegions()
{
    auto is2D = [](const CopyOperand &op) { return op.vs || op.width; };

    for (auto &i: insns) {
        if ((is2D(i.dst) && !is4(i.dst.type) && !isInt2(i.dst.type)) || is2D(i.src1) || is2D(i.src2))
            stub("Unsupported 2D region");
        if (is2D(i.src0)) {
            if (i.dst.stride > 4)
                continue;
            if (i.flag) stub("Unsupported predication");
            int w = i.src0.width, vs = i.src0.vs, hs = i.src0.stride;
            bool splitH = (w * w >= i.simd || (hw == ngen::HW::Xe3p && i.dst.stride * w >= 8));
            int nsplit = splitH ? (i.simd / w) : w;
            if (nsplit == 0) stub("Source width exceeds SIMD");
            i.simd /= nsplit;
            i.src0.stride = splitH ? hs : vs;
            i.src0.vs = i.src0.width = 0;
            i.src0.overwriteStride = false;
            for (int isplit = 1; isplit < nsplit; isplit++) {
                newInsns.emplace_back(i);
                auto &inew = newInsns.back();
                inew.src0.offset += (splitH ? vs : hs) * isplit;
                for (auto *op: {&inew.dst, &inew.src1, &inew.src2}) {
                    if (op->kind == CopyOperand::GRF) {
                        op->offset += op->stride * (splitH ? w : 1) * isplit;
                        if (!splitH) op->stride *= w;
                    }
                }
            }
            if (!splitH) for (auto *op: {&i.dst, &i.src1, &i.src2})
                op->stride *= w;
        }
    }

    mergeChanges();
}

// Pass to spread phases through phase space.
// Instructions with the same phase are assumed to be logically independent.
void CopyPlan::distributePhases()
{
    uint16_t nphase = 0;
    for (const auto &i: insns)
        nphase = std::max(nphase, i.phase);
    nphase++;

    uint16_t spread = 0x8000 / nphase;
    for (auto &i: insns) {
        i.spread = spread;
        i.phase = (2*i.phase + 1) * spread;
    }
}

// Pass to legalize type conversions.
void CopyPlan::planTypeConversions()
{
    bool rerun = false;
    bool rerunZip = false;

    for (auto &i: insns) {
        if (i.op != Opcode::mov) continue;
        if (i.dst == i.src0) i.invalidate();
        if (i.isInvalid()) continue;

        auto &st = i.src0.type, &dt = i.dst.type;
        auto &srange = i.src0.range;

        if (asSigned(st) == asSigned(dt) && st != dt && !i.sat)
            dt = st;
        if (st == dt)
            i.moveToIntegerPipe();

        if (hw == ngen::HW::Xe3p && is4(st) && one_of(getBits(dt), {8, 16}))
            if (planShflUpconvertXe3p(i))
                continue;

        if (isInt2(st) && one_of(dt, {ngen_b16_h2x(), ngen_b16_l2x()}))
            planSubByteShifts(i);
        else if (is4(st) && one_of(dt, {ngen_b16_h4x(), ngen_b16_l4x()}))
            planSubByteShifts(i);
        else if (isInt4(st) && isInt4(dt) && st != dt) {
            copyThrough(i, DataType::w);
            rerun = true;
        } else if (isInt2(st) && isInt2(dt)) {
            copyThrough(i, DataType::w);
            rerun = true;
        } else if (isInt2(st) && isInt(dt)) {
            // Zip optimizations for int2-to-byte upconversions create illegal
            // instructions, use a word intermediate
            if (isB(dt))
                copyThrough(i, isSigned(st) ? DataType::w : DataType::uw);
            else
                planInt2Upconversion(i);
            rerun = true;
        } else if (isInt(st) && isInt2(dt)) {
            planInt2Downconversion(i);
            rerun = true;
        } else if (isInt4(st) && isInt(dt)) {
            planInt4Upconversion(i);
            rerun = true;
        } else if (isInt(st) && isInt4(dt)) {
            planInt4Downconversion(i);
            rerun = true;
        } else if (isSubByteInt(st) && one_of(dt, {DataType::hf, DataType::bf})) {
            if (bfArithmeticOK(i))
                copyThrough(i, isInt2(st) ? ngen_b16_l2x() : ngen_b16_l4x());
            else
                copyThrough(i, isSigned(st) ? DataType::b : DataType::ub);
            rerunZip = true;
        } else if (one_of(st, {ngen_b16_l2x(), ngen_b16_l4x()}) && one_of(dt, { DataType::hf, DataType::bf }))
            planSubByteIntToF16(i);
        else if (st == DataType::hf && dt == DataType::e2m1) {
            planEmulatedHFToF4(i);
            rerun = true;
        } else if (isFP4(dt)) {
            copyThrough(i, DataType::hf);
            rerun = true;
        } else if (isSubByteInt(st) && isFP(dt)) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (isFP(st) && isSubByteInt(dt)) {
            copyThrough(i, DataType::w);
            rerun = true;
        } else if (is4(dt))
            stub("Unsupported move to 4-bit type");
        else if (isInt2(dt))
            stub("Unsupported move to 2-bit type");
        else if (isInt3(dt))
            stub("Unsupported move to 3-bit type");
        else if (isB(st) && getBytes(dt) == 8)
            copyThrough(i, DataType::w);
        else if (getBytes(st) == 8 && isB(dt))
            copyThrough(i, DataType::w);
        else if (st == DataType::hf && dt == DataType::df)
            copyThrough(i, DataType::f);
        else if (st == DataType::df && dt == DataType::hf)
            copyThrough(i, DataType::f, 1);
        else if (st == DataType::hf && isQ(dt))
            copyThrough(i, DataType::d);
        else if (isQ(st) && dt == DataType::hf)
            copyThrough(i, DataType::d);
        else if (isB(st) && dt == DataType::hf)
            planInt8ToHF(i);
        else if (isB(st) && dt == DataType::bf) {
            planInt8ToBF(i);
            rerun = true;
        } else if (isB(st) && dt == DataType::f && needsByteFloatWorkaround()) {
            copyThrough(i, DataType::d);
            rerun = true;
        } else if (st == DataType::f && dt == DataType::tf32) {
            if (hw < HW::XeHPC)
                stub("No emulation for tf32 rounding");
        } else if (st != DataType::tf32 && dt == DataType::tf32) {
            if (isSubsetOf(st, dt))
                dt = DataType::f;
            else
                copyThrough(i, DataType::f);
            rerun = true;
        } else if (st == DataType::tf32) {
            st = DataType::f;
            if (dt == DataType::tf32)
                dt = DataType::f;
            rerun = true;
        } else if (st == DataType::bf && dt == DataType::f) {
            i.op = Opcode::shl;
            i.dst.type = DataType::ud;
            i.src0.type = DataType::uw;
            i.src1 = 16;
        } else if (st == DataType::f && dt == DataType::bf) {
            if (isSubsetOf(i.src0.range, dt)) {
                i.op = Opcode::mov;
                i.src0.type = i.dst.type = DataType::uw;
                i.src0.offset *= 2;
                i.src0.stride *= 2;
                i.src0.offset++;
            } else if (!systolicAvailable)
                planEmulatedHalveFloat(i);
        } else if (st == DataType::bf8 && dt == DataType::hf) {
            i.op = Opcode::shl;
            i.dst.type = DataType::uw;
            i.src0.type = DataType::ub;
            i.src1 = 8;
        } else if (st == DataType::hf && dt == DataType::bf8) {
            if (isSubsetOf(i.src0.range, dt)) {
                i.op = Opcode::mov;
                i.src0.type = i.dst.type = DataType::ub;
                i.src0.offset *= 2;
                i.src0.stride *= 2;
                i.src0.offset++;
            } else if (hw < HW::XeHPC) {
                if (i.dst.stride == 1) {
                    restrideDst(i, 2);
                    rerun = true;
                } else
                    planEmulatedHalveFloat(i);
            }
        } else if (st == DataType::bf8 && dt == DataType::bf) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (st == DataType::hf8 && dt == DataType::hf) {
            if (hw < HW::Xe3) {
                planUnpack8To16High(i);
                rerunZip = true;
            }
        } else if (st == DataType::hf8 && dt == DataType::bf && hw < HW::Xe3) {
            bfArithmeticOK(i) ? planUnpack8To16High(i)
                              : copyThrough(i, DataType::hf, 1);
            rerunZip = true;
        } else if (st == ngen_b16() && srange == DataType::hf8 && dt == DataType::hf)
            planEmulatedHF8ToHF(i);
        else if (st == ngen_b16() && srange == DataType::hf8 && dt == DataType::bf)
            planEmulatedHF8ToBF(i);
        else if (st == DataType::hf && dt == DataType::hf8) {
            if (hw < HW::Xe3)
                planEmulatedHFToHF8(i);
        } else if (st == Type::ngen_e8m0() && dt == DataType::f) {
            planE8M0ToF(i);
            rerun = true;
        } else if (st == Type::ngen_e8m0()) {
            copyThrough(i, DataType::f);
            rerun = true;
        } else if (st != dt && (isFP8(st) || isFP8(dt))) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (st == DataType::e2m1 && one_of(dt, {DataType::hf, DataType::bf})) {
            if (dt == DataType::bf && !bfArithmeticOK(i))
                copyThrough(i, DataType::hf);
            else
                copyThrough(i, ngen_b16_h4x());
            rerun = true;
        } else if (st == ngen_b16_h4x() && dt == DataType::hf)
            planEmulatedF4ToHF(i);
        else if (st == ngen_b16_h4x() && dt == DataType::bf)
            planEmulatedF4ToBF(i);
        else if (st == Type::ngen_nf4() && dt == DataType::hf) {
            planUnpack4To16(i);
            rerunZip = true;
        } else if (st == ngen_b16() && srange == Type::ngen_nf4() && dt == DataType::hf)
            planEmulatedNF4ToHF(i);
        else if (isFP4(st)) {
            copyThrough(i, DataType::hf, 1);
            rerun = true;
        } else if (st == DataType::bf && dt != DataType::bf) {
            copyThrough(i, DataType::f);
            rerun = true;
        } else if (st != DataType::bf && dt == DataType::bf) {
            copyThrough(i, DataType::f);
            rerun = true;
        } else for (auto t: {st, dt}) {
            if (one_of(t, {Type::ngen_e8m0(), Type::ngen_nf4(), ngen_b16_l2x(), ngen_b16_h2x(), ngen_b16_l4x(), ngen_b16_h4x(), ngen_b16()}))
                stub("Unsupported data type conversion");
        }
    }

    mergeChanges();
    if (rerun || rerunZip) {
        if (rerunZip) {
            sort(SortType::Register);
            optimizeZip();
            legalizeSIMD(true);
        }
        planTypeConversions();
    }
}

// Upconvert 4-bit types to 8/16 bits using shfl.idx4.
bool CopyPlan::planShflUpconvertXe3p(CopyInstruction &i)
{
    // Cases handled:     (with 16-bit upconversion; 8-bit similar)
    // 1a)
    //     mov  y:uw<1>    x:u4<8;2,1>      -->  shfl.idx4  y.0:ud<1>  lut:ud  x.(n/2):ub<4>
    // 1b)
    //     mov  y.0:uw<2>  x.n:u4<8>        -->  same as 1a
    //     mov  y.1:uw<2>  x.(n+1):u4<8>
    // 2)
    //     mov  y:uw<1>    x:u4<1>          -->  mov  y:ub<4>  x:ub<1>
    //                                           mov  y:uw<1>  x:u4<8;2,1> (--> case 1a)
    // 3)
    //     mov  y:uw<n>    x:u4<1>          -->  mov  t:uw<1>  x:u4<1>     (--> case 2 (<1>), 1a (<8;2,1>))
    //                      OR <8;2,1>           mov  y:uw<n>  t:uw<1>
    //
    // If dst is integral, only use shfl.idx4 in case 1 and only when src and dst have valid offsets for shfl.idx4.

    auto st = i.src0.type, dt = i.dst.type;
    const bool _16 = (getBytes(dt) == 2);
    const auto minElems = _16 ? 32 : 64;  // minimum 4-bit elements per shfl.idx4 instruction

    bool laneAligned = (i.src0.vs == 8 && i.src0.width * getBytes(dt) == 4 && i.src0.stride == 1);
    if ((i.src0.vs || i.src0.width) && !laneAligned)
        return false;       /* unsupported 2D region */
    if (!laneAligned && i.src0.stride != 1)
        return false;       /* expect stride 1 */
    if (i.src0.offset & (_16 ? 1 : 3))
        return false;       /* unaligned input */

    auto x = i.src0, y = i.dst;
    const bool copySrc = !laneAligned || x.byteOffset() >= 4;
    const bool copyDst = (y.stride != 1 || y.offset != 0 || i.simd % minElems);

    if (isInt(dt) && (copySrc || copyDst))
        return false;       /* use normal sequence */

    auto lut = getResource(CopyResource::makeShflLUT(st, dt));
    if (!lut)
        return false;       /* no LUT available */
    lut.type = DataType::ud;
    lut.stride = 0;         /* will be fixed up later */


    x.offset >>= (_16 ? 1 : 2);
    x.type = (_16 ? DataType::ub : DataType::uw);

    int orig_simd  = i.simd;
    if (copyDst) {
        // Round up SIMD to ensure a valid shfl.
        i.simd = round_up(i.simd, minElems);
        y = newTemp(dt, i.simd, 1);
    }

    i.simd /= (_16 ? 2 : 4);
    auto ie = splitMultiple<3>(i);

    if (copySrc) {
        ie[0]->op = Opcode::mov;
        ie[0]->src0 = x;
        x = y;
        x.type = (_16 ? DataType::ub : DataType::uw);
        x.stride = (_16 ? 4 : 2);
        ie[0]->dst = x;
    } else
        ie[0]->invalidate();

    ie[1]->op = Opcode::shfl;
    ie[1]->dst = y;
    ie[1]->dst.type = DataType::ud;
    ie[1]->src0 = lut;
    ie[1]->src1 = x;

    if (copyDst) {
        ie[2]->op = Opcode::mov;
        ie[2]->src0 = y;
        ie[2]->simd = orig_simd;
    } else
        ie[2]->invalidate();

    return true;
}

void CopyPlan::legalizeShfl()
{
    for (auto &i: insns) {
        if (i.op != Opcode::shfl) continue;
        if (!one_of(i.simd, {16, 32})) stub();
        if (i.simd == 32) {
            i.src0.stride = 1;
            i.src0.width = 16;
            i.src0.vs = 0;
        }else{
            i.src0.stride = 0;
            i.src0.width = 1;
            i.src0.vs = 1;
        }
    }
}

// Unpack 4-bit src type into 16 bits (zero extended), used in many conversion sequences.
void CopyPlan::planUnpack4To16(CopyInstruction &i)
{
    auto st = i.src0.type;
    auto &i0 = i, &i1 = split(i);
    i0.src0.type = DataType::u4;
    i0.dst.type = DataType::uw;
    i1.src0 = i0.dst;
    i1.src0.type = ngen_b16();
    i1.src0.range = st;
}

// Unpack 8-bit src type into the high 8 bits of a 16-bit slot.
void CopyPlan::planUnpack8To16High(CopyInstruction &i)
{
    auto st = i.src0.type;
    auto &i0 = i, &i1 = split(i);
    i0.op = Opcode::shl;
    i0.src0.type = DataType::ub;
    i0.src1 = 8;
    i0.dst.type = DataType::uw;
    i1.src0 = i0.dst;
    i1.src0.type = ngen_b16();
    i1.src0.range = st;
}

// Emulate bfn instructions on pre-Gen12HP architectures.
void CopyPlan::planBFNEmulation()
{
    if (hw > HW::Gen12LP) return;

    const CopyOperand zeros = ngen::Immediate::w(0);

    auto negate = [](CopyOperand op) {
        if (op.kind != CopyOperand::Immediate) {
            op.neg = !op.neg;
            return op;
        }

        const auto bits = getBits(op.type);
        const uint64_t mask_base = ((bits & 63) != 0);
        const uint64_t mask = (mask_base << (bits & 63)) - 1;
        op.value = ~op.value & mask;
        return op;
    };

    auto operand = [&](uint8_t id, const CopyOperand &s0, const CopyOperand &s1, const CopyOperand &s2) {
        if (id == 0x00) return zeros;
        if (id == 0x0F) return negate(s2);
        if (id == 0x33) return negate(s1);
        if (id == 0x55) return negate(s0);
        if (id == 0xAA) return s0;
        if (id == 0xCC) return s1;
        if (id == 0xF0) return s2;
        if (id == 0xFF) return negate(zeros);
        return CopyOperand();
    };

    auto fixup = [&](CopyInstruction &i) {
        i.src2 = CopyOperand();
        bool imm0 = (i.src0.kind == CopyOperand::Immediate);
        bool imm1 = (i.src1.kind == CopyOperand::Immediate);
        if (imm0 && !imm1) std::swap(i.src0, i.src1);
        if (!imm0 || !imm1) return;
        if (i.op == Opcode::and_) i.src0.value &= i.src1.value;
        else if (i.op == Opcode::or_) i.src0.value |= i.src1.value;
        else if (i.op == Opcode::xor_) i.src0.value ^= i.src1.value;
        else return;
        i.op = Opcode::mov;
        i.src1 = CopyOperand();
    };

    bool rerun = false;

    for (auto &i: insns) {
        if (i.op != Opcode::bfn) continue;

        auto dst = i.dst, src0 = i.src0, src1 = i.src1, src2 = i.src2;
        const auto &bfn = BFN::nodes[i.ctrl];

        if (bfn.op == Opcode::mov) {
            i.op = Opcode::mov;
            i.src0 = operand(bfn, i.src0, i.src1, i.src2);
            i.src1 = i.src2 = CopyOperand();
            continue;
        }

        // TODO: improve emulation of immediate (sub)trees.

        const auto &left = BFN::nodes[bfn.left];
        const auto &right = BFN::nodes[bfn.right];

        if (left.op == Opcode::mov && right.op == Opcode::mov) {
            i.op = bfn.op;
            i.src0 = operand(left, src0, src1, src2);
            i.src1 = operand(right, src0, src1, src2);
            fixup(i);
            continue;
        }

        rerun = true;
        if (left.op == Opcode::mov || right.op == Opcode::mov) {
            uint8_t nested = left.op == Opcode::mov ? right : left;
            auto s1 = operand(nested ^ left ^ right, src0, src1, src2);

            auto tmp = dst;
            if (dst == s1 || dst == negate(s1))
                tmp = newTemp(dst.type, i.simd, dst.stride);
            auto &i1 = i, &i2 = split(i);
            i1.ctrl = nested;
            i1.dst = tmp;

            i2.op = bfn.op;
            i2.src0 = tmp;
            i2.src1 = s1;
            fixup(i2);
            continue;
        }

        auto tmp = newTemp(dst.type, i.simd, dst.stride);
        auto ie = splitMultiple<3>(i);
        ie[0]->ctrl = left;
        ie[0]->dst = tmp;
        ie[1]->ctrl = right;
        ie[2]->op = bfn.op;
        ie[2]->src0 = dst;
        ie[2]->src1 = tmp;
        fixup(*ie[2]);
    }
    mergeChanges();
    if (rerun)
        planBFNEmulation();
}

// {b,ub}->hf sequence.
void CopyPlan::planInt8ToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) return;

    auto &i0 = i, &i1 = split(i);

    bool s8 = (i.src0.type == DataType::b);
    uint16_t bias = s8 ? 0x6480 : 0x6400;    // s8: 1024+128 / u8: 1024

    // Copy to low 8 bits and add hf bias term.
    i0.op = s8 ? Opcode::xor_ : Opcode::or_;
    i0.src0.type = DataType::ub;
    i0.src1 = bias;
    i0.dst.type = DataType::uw;

    // Undo bias in hf arithmetic.
    i1.op = Opcode::add;
    i1.src0 = i1.dst;
    i1.src1 = Immediate::hf(0x8000 | bias);
}

bool CopyPlan::bfArithmeticOK(const CopyInstruction &i) const
{
    return systolicAvailable && (hw > HW::XeHPG || i.simd > 1);
}

CopyOperand CopyPlan::bfImmediate(uint16_t bits, bool ternary)
{
    if (ternary) {
        auto kind = CopyResource::makeConstant32(uint32_t(bits) << 16);
        auto val = getResource(kind);
        val.stride = 0;
        val.type = DataType::f;
        return val;
    } else {
        auto imm = Immediate::ud(uint32_t(bits) << 16);
        imm.setType(DataType::f);
        return imm;
    }
};

// {b,ub}->bf sequence.
void CopyPlan::planInt8ToBF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod() || hw == ngen::HW::Xe3p || !bfArithmeticOK(i)) {
        copyThrough(i, DataType::f);
        return;
    }

    auto ie = splitMultiple<3>(i);

    bool s8 = (i.src0.type == DataType::b);

    // Copy to u16, shifting by 128 if signed.
    if (s8) {
        ie[0]->op = Opcode::xor_;
        ie[0]->src0.type = DataType::ub;
        ie[0]->src1 = 0x80;
    }
    ie[0]->dst.type = DataType::uw;

    // Reinterpret as denormal bf16 value, and scale to normal range,
    //  simultaneously undoing any shift.
    // This cannot be done in a single multiply. Start with one bf*f multiply (2 cycles).
    if (s8) {
        ie[1]->op = Opcode::mad;
        ie[1]->src0 = bfImmediate(0xBF00, true);
        ie[1]->src1 = ie[1]->dst;
        ie[1]->src2 = bfImmediate(0x7E00, true);
    } else {
        ie[1]->op = Opcode::mul;
        ie[1]->src0 = ie[1]->dst;
        ie[1]->src1 = bfImmediate(0x7E00, false);
    }

    // Complete scaling with a faster hf multiply (1 cycle).
    ie[2]->op = Opcode::mul;
    ie[2]->dst.type = DataType::hf;
    ie[2]->src0 = ie[2]->dst;
    ie[2]->src1 = Immediate::hf(0x4000);
}

void CopyPlan::legalizeBfImmediate(CopyInstruction &i1)
{
    if (i1.src1.kind != CopyOperand::Immediate) return;
    auto temp = newTemp(DataType::uw, i1.simd, 1);
    auto &i2 = split(i1);

    i1.op = Opcode::mov;
    i1.dst = temp;
    i1.src0 = Immediate::uw((uint16_t)(i1.src1.value >> 16));
    i1.src0.type = DataType::uw;
    i1.flag = i1.src1 = i1.src2 = CopyOperand{};
    i1.cmod = ConditionModifier::none;

    i2.src1 = temp;
    i2.src1.type = DataType::bf;
}

// s2/u2/s4/u4 -> hf/bf sequence.
void CopyPlan::planSubByteIntToF16(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    // Incoming int2/4 data x has been shifted into the low 2/4-bits of src0;
    //   there may be junk in other bits.
    //
    // Use bfn to create 2^m + x (+ 2/8 if x is s2/s4) as an hf/bf number, where
    //   m = # mantissa bits.
    // Then subtract the 2^m (+ 2/8) bias in hf/bf arithmetic.

    auto &i0 = i, &i1 = split(i);

    bool hf = (i.dst.type == DataType::hf);
    auto bits = getBits(i.src0.range);
    uint16_t bias_offset = isSigned(i.src0.range) ? 1 << (bits - 1) : 0;
    uint16_t bit_mask = (1 << bits) - 1;

    uint16_t bias = (hf ? 0x6400 : 0x4300) | bias_offset;

    auto yUW = i.dst;
    yUW.type = DataType::uw;

    i0.op = Opcode::bfn;
    i0.ctrl = 0x6A;             // src0 ^ (src1 & src2)
    i0.src0 = bias;
    i0.src1 = i0.dst = yUW;
    i0.src2 = bit_mask;

    i1.op = Opcode::add;
    i1.src0 = i1.dst;
    i1.src1 = hf ? CopyOperand(Immediate::hf(bias | 0x8000))
                 : bfImmediate(bias | 0x8000, false);
}

// Emulated f->bf or hf->bf8 sequence.
void CopyPlan::planEmulatedHalveFloat(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<4>(i);

    bool toBF = (i.src0.type == DataType::f);
    if (!toBF && i.src0.type != DataType::hf) stub();

    auto T_large = toBF ? DataType::ud : DataType::uw;
    auto T_small = toBF ? DataType::uw : DataType::ub;

    auto esrc0 = i.src0;
    if (esrc0.overwrite && !multiGRF(hw, i, i.src0))
        esrc0.type = T_large;
    else
        esrc0 = newTemp(T_large, i.simd, 1);

    // Emulation sequence for mov y:bf x:f:
    //   add            x:ud x:ud -0x8000
    //   and (nz)fN.N   x:ud       0x1FFFF
    //   mov            y:uw       x_hi:uw
    //   (fN.N) add     y:uw       x_hi:uw     1
    //
    // hf->bf8 is similar but half as wide.

    ie[0]->op = Opcode::add;
    ie[0]->src0.type = T_large;
    ie[0]->src1 = toBF ? -0x8000 : -0x80;
    ie[0]->dst = esrc0;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = esrc0;
    ie[1]->src1 = toBF ? 0x1FFFF : 0x1FF;
    ie[1]->dst = CopyOperand();
    ie[1]->dst.type = T_large;
    ie[1]->cmod = ConditionModifier::nz;
    ie[1]->flag = newFlag(ie[1]->simd);

    ie[2]->op = Opcode::mov;
    ie[2]->src0 = esrc0;
    ie[2]->src0.type = ie[2]->dst.type = T_small;
    ie[2]->src0.stride *= 2;
    ie[2]->src0.offset++;

    ie[3]->op = Opcode::add;
    ie[3]->src0 = esrc0;
    ie[3]->src0.type = ie[3]->dst.type = T_small;
    ie[3]->src0.stride *= 2;
    ie[3]->src0.offset++;
    ie[3]->src1 = 1;
    ie[3]->flag = ie[1]->flag;
}

// Pass to perform early int2/4 upconversion transformations before 2D
//   regions are split into 1D regions.
void CopyPlan::planEarlySubByteIntUpconversions()
{
    for (auto &i : insns) {
        if (i.op != Opcode::mov || !isB(i.dst.type) || !isInt(i.src0.type) || i.src0.stride != 1) continue;
        auto perByte = bytesToElements(1, i.src0.type);
        auto strideCutoff = isSigned(i.src0.type) ? perByte : 2 * perByte;
        if (perByte < 2 || i.src0.width != perByte || i.dst.stride < strideCutoff) continue;
        is4(i.src0.type) ? planInt4Upconversion(i) : planInt2Upconversion(i);
    }

    mergeChanges();
}

// Rewrite int2 -> int upconversion using byte operations.
// May need to be run twice.
//
// Example input:
//    mov (16)   r0.0<1>:ub   r1.0<1>:u2
// Output:
//    shr (4)    r0.0<4>:ub   r1.0<1>:ub   0:uw
//    and (4)    r0.0<4>:ub   r0.0<1>:ub   0x3:uw
//    shr (4)    r0.1<4>:ub   r1.0<1>:ub   2:uw
//    and (4)    r0.1<4>:ub   r0.1<4>:ub   0x3:uw
//    shr (4)    r0.2<4>:ub   r1.0<1>:ub   4:uw
//    and (4)    r0.2<4>:ub   r0.2<4>:ub   0x3:uw
//    shr (4)    r0.3<4>:ub   r1.0<1>:ub   6:uw
//    and (4)    r0.3<4>:ub   r0.3<4>:ub   0x3:uw
//
void CopyPlan::planInt2Upconversion(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    i.sat = false;

    // TODO: once a shfl instruction is available for 2-bit values, enable this
    /*
    if (hw >= HW::Xe3p && getBytes(i.dst.type) <= 2)
        if (planShflUpconvertXe3p(i))
            return;
    */

    bool s2 = (i.src0.type == DataType::s2);

    if (i.src0.stride < 4 && i.simd * i.src0.stride >= 4) {
        auto& i0 = i;
        int factor = 4 / i.src0.stride;
        i0.dst.stride *= factor;
        i0.src0.stride *= factor;
        i0.simd /= factor;
        split(i, false);
        i0.dst.offset += i0.dst.stride / factor;
        i0.src0.offset += i0.src0.stride / factor;
        if (factor == 4) {
            split(i, false);
            i0.dst.offset += i0.dst.stride / factor;
            i0.src0.offset += i0.src0.stride / factor;
            split(i, false);
            i0.dst.offset += i0.dst.stride / factor;
            i0.src0.offset += i0.src0.stride / factor;
        }
    } else {
        int offset = (i.src0.offset % 4);
        i.src0.stride /= 4;
        i.src0.offset /= 4;
        i.src0.type = DataType::ub;

        if (s2) {
            auto &i0 = i, &i1 = split(i);
            auto rshift = getBytes(i0.dst.type) * 8 - 2;
            auto lshift = rshift - 2 * offset;
            i0.op = Opcode::shl;
            i0.src1 = lshift;

            i1.src0 = i0.dst;
            i1.op = Opcode::asr;
            i1.src0.type = asSigned(i1.src0.type);
            i1.src1 = rshift;
        } else {
            auto &i0 = i, &i1 = split(i);
            auto rshift = 2 * offset;
            i0.op = Opcode::shr;
            i0.src1 = Immediate::uw(rshift);

            i1.src0 = i0.dst;
            i1.op = Opcode::and_;
            i1.src1 = 0x3;
        }
    }
}

// Lower u3 copies before generic passes interpret their register regions.
void CopyPlan::planEarlyInt3Upconversions()
{
    for (auto &i : insns)
        if (i.src0.type == Type::ngen_u3()) {
            if (i.op != Opcode::mov) stub("Unsupported u3 copy operation.");
            planInt3Upconvert(i);
        }
    mergeChanges();
}

// Rewrite u3 -> int upconversion using byte operations.
//
// Eight u3 values are packed into 3 bytes (24 bits), laid out as three
// "rows" of source bytes (row pitch and column pitch come from
// CopyInstruction::u3Regioning). Lane k occupies bits [3k, 3k+3) of the
// 24-bit group:
//   - lanes 0,1,3,4,6,7 lie entirely within one byte: a single shr by the
//     lane's bit offset extracts them.
//   - lanes 2 and 5 straddle a byte boundary: the low bits come from a shr of
//     the lower byte, the high bits from a shl of the next byte, and the two
//     parts are combined with or_.
// Lanes 0/1 and 6/7 share a source row, so when the destination stride is 2
// they are merged into a single shr with a packed pair of shift immediates.
// Each lane is written to its (possibly interleaved) destination position.
// Finally, one and_ with 0x7 clears the leftover high bits of all unpacked
// elements at once, and, for non-integer (hf/bf) destinations, a converting
// mov turns the masked uw values into the final type.
//
// Example (dst stride 1, integer dst, simplified):
//    shr  dst[0]  row0  0
//    shr  dst[1]  row0  3
//    shr  tmp     row0  6;  shl  tmpHi  row1  2;  or_  dst[2]  tmp  tmpHi
//    ...
//    and  dst[0-7]  dst[0-7]  0x7
//
void CopyPlan::planInt3Upconvert(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");

    struct U3Lane {
        uint8_t byte, shift;        // low (or only) source row and shift amount
        uint8_t hiByte, hiShift;    // high source row and shift amount, if straddling
        bool straddle;
    };
    static const U3Lane lanes[8] = {
        {0, 0, 0, 0, false},
        {0, 3, 0, 0, false},
        {0, 6, 1, 2, true},
        {1, 1, 0, 0, false},
        {1, 4, 0, 0, false},
        {1, 7, 2, 1, true},
        {2, 2, 0, 0, false},
        {2, 5, 0, 0, false},
    };

    int n = i.simd;
    if (n <= 0) stub("u3 upconversion requires a positive SIMD width.");
    int grfBytes = GRF::bytes(hw);

    // Integer destinations can be written directly; non-integer (hf/bf)
    // destinations go through a raw-uw reinterpretation + converting mov.
    bool directWrite = isInt(i.dst.type);
    // Preserve the caller-requested dst stride (e.g. crosspack-2).
    CopyOperand finalDst = i.dst;

    CopyOperand srcBase = i.src0;
    srcBase.type = DataType::ub;

    if (i.u3Regioning.columnPitchBytes <= 1 || i.u3Regioning.rowPitchBytes <= 0)
        stub("u3 upconversion requires row-spread regioning data.");

    int nLocal = n;
    int colPitchBytes = i.u3Regioning.columnPitchBytes;
    int byteStep = 1;
    int rowStrideBytes = i.u3Regioning.rowPitchBytes;

    // Add a byte offset, handling column-quad overflow (wraps to the
    // next row) and overflow into subsequent GRF registers.
    auto addByteOffset = [&](CopyOperand op, int bytes) {
        if ((op.offset % colPitchBytes + bytes) >= colPitchBytes)
            op.offset = op.offset - (op.offset % colPitchBytes)
                    + (op.offset % colPitchBytes + bytes) % colPitchBytes
                    + rowStrideBytes;
        else
            op.offset += bytes;

        int grfOffset = op.offset / grfBytes;
        op.grf += grfOffset;
        op.offset -= grfOffset * grfBytes;
        return op;
    };

    // Lane pairs (0,1)/(6,7) share a row and can merge into one shr when
    // dstStride == 2; otherwise all 8 lanes are processed individually.
    constexpr int laneOps = 12;
    // Trailing and_ mask + optional converting mov need 2 more slots.
    constexpr int extraOps = 2;
    constexpr int totalOps = laneOps + extraOps;

    // split()/splitMultiple() push into `newInsns`, so reserve first.
    int nOpsNeeded = totalOps;
    newInsns.reserve(newInsns.size() + nOpsNeeded - 1);

    auto setOp = [&](CopyInstruction *ci, Opcode op, int simd, const CopyOperand &d, const CopyOperand &s0, const CopyOperand &s1) {
        ci->op = op;
        ci->simd = simd;
        ci->dst = d;
        ci->src0 = s0;
        ci->src1 = s1;
        ci->src2 = CopyOperand();
        ci->sat = false;
        ci->cmod = ConditionModifier::none;
    };

    int type_size = getBytes(finalDst.type);
    int dstStride = finalDst.stride;
    switch (type_size) {
        case 1:
        case 2:
        case 4: break;
        default: stub("Unsupported dst type size for u3 upconversion.");
    }

    // Row-spread u3 sources are unpacked in a single pass: columns are
    // stride-4 (crosspack-4) and rows are 1 byte apart.
    CopyOperand rows[3];
    rows[0] = srcBase;
    rows[1] = addByteOffset(srcBase, byteStep);
    rows[2] = addByteOffset(srcBase, 2 * byteStep);
    for (auto &r : rows) r.stride = colPitchBytes;

    // Straddling lanes need lo/hi scratch for their shift/or sequence.
    auto tmp = newTemp(DataType::uw, nLocal, 2);
    auto tmpHi = newTemp(DataType::uw, nLocal, 2);

    std::vector<CopyInstruction*> allOps;
    {
        auto ops = splitMultiple<totalOps>(i);
        allOps.assign(ops.begin(), ops.end());
    }

    int next = 0;

    // Compute lane's destination region, interleaved with its pair
    // partner's at stride dstStride (crosspack).
    auto computeFinalDstLane = [&](int lane) {
        int pairIdx = lane / dstStride;
        int parity  = lane % dstStride;
        auto fd = finalDst;
        fd.stride = dstStride;
        fd.offset += pairIdx * nLocal * fd.stride + parity;
        int grfOffset = (fd.offset * type_size) / grfBytes;
        fd.grf += grfOffset;
        fd.offset -= ((grfOffset * grfBytes) / type_size);
        if (!directWrite) fd.type = DataType::uw;
        return fd;
    };

    // Element-index gap between two GRF-normalized lane destinations.
    auto elemIndex = [&](const CopyOperand &op) {
        return (int64_t) op.grf * (grfBytes / type_size) + op.offset;
    };

    // Merge two non-straddling lanes sharing a source row into one
    // SIMD-2*nLocal shr using a packed shift immediate.
    auto mergeRowPair = [&](int laneA, int laneB, CopyInstruction *shrOp) {
        const auto &LA = lanes[laneA];
        const auto &LB = lanes[laneB];
        const auto &rowLo = rows[LA.byte];

        auto dstA = computeFinalDstLane(laneA);
        auto dstB = computeFinalDstLane(laneB);
        int hstride = int(elemIndex(dstB) - elemIndex(dstA));

        auto dstMerged = dstA;
        dstMerged.width = 2;
        dstMerged.stride = hstride;
        dstMerged.vs = dstStride;

        auto rowDup = rowLo;
        rowDup.width = 2;
        rowDup.vs = rowLo.stride;
        rowDup.stride = 0;

        CopyOperand shiftLo(int(LA.shift)), shiftHi(int(LB.shift));
        shiftLo.type = shiftHi.type = DataType::uw;
        CopyOperand packedShift;
        packedShift = zipImmediates(shiftLo, shiftHi, 1);
        if (!packedShift) stub("Failed to pack u3 shift immediates.");

        setOp(shrOp, Opcode::shr, 2 * nLocal, dstMerged, rowDup, packedShift);
    };

    // Lanes 0&1/6&7 merge into one shr each when dstStride == 2; the
    // middle pair and straddling lanes are always processed individually.
    bool canMergePairs = (dstStride == 2);
    if (canMergePairs) {
        static const int mergePairs[2][2] = {{0, 1}, {6, 7}};
        for (auto &pr : mergePairs)
            mergeRowPair(pr[0], pr[1], allOps[next++]);
    }

    // Straddling lanes (2, 5) use a shr/shl/or_ sequence; non-merged
    // lanes use a single shr.
    std::vector<int> soloLanes = {2, 3, 4, 5};
    if (!canMergePairs)
        soloLanes = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int lane : soloLanes) {
        const auto &L = lanes[lane];
        auto finalDstLane = computeFinalDstLane(lane);
        const auto &rowLo = rows[L.byte];

        if (!L.straddle) {
            setOp(allOps[next++], Opcode::shr, nLocal, finalDstLane, rowLo, CopyOperand(int(L.shift)));
        } else {
            const auto &rowHi = rows[L.hiByte];
            setOp(allOps[next++], Opcode::shr, nLocal, tmp, rowLo, CopyOperand(int(L.shift)));
            setOp(allOps[next++], Opcode::shl, nLocal, tmpHi, rowHi, CopyOperand(int(L.hiShift)));
            setOp(allOps[next++], Opcode::or_, nLocal, finalDstLane, tmp, tmpHi);
        }
    }

    // Invalidate unused trailing lane slots (fewer ops needed when pair
    // merging was used).
    while (next < laneOps)
        allOps[next++]->invalidate();

    // Mask all unpacked elements together, then convert if necessary.
    int totalElems = nLocal * 8;
    auto maskDst = finalDst;
    maskDst.stride = 1;
    if (!directWrite) maskDst.type = DataType::uw;
    int maskBytes = getBytes(maskDst.type);
    int maskElems = totalElems;
    uint32_t mask = 7;
    if (maskDst.absByteOffset(hw) % 4 == 0
            && (totalElems * maskBytes) % 4 == 0) {
        maskDst.type = DataType::ud;
        maskDst.offset = finalDst.offset * getBytes(finalDst.type) / 4;
        maskElems = totalElems * getBytes(finalDst.type) / 4;
        if (maskBytes == 1)
            mask = 0x07070707;
        else if (maskBytes == 2)
            mask = 0x00070007;
    }
    setOp(allOps[next++], Opcode::and_, maskElems, maskDst, maskDst,
            CopyOperand(Immediate::ud(mask)));

    // Convert masked uw values in place when dst isn't an integer type.
    if (!directWrite) {
        auto realDst = finalDst;
        realDst.stride = 1;
        auto rawSrc = realDst;
        rawSrc.type = DataType::uw;
        setOp(allOps[next++], Opcode::mov, totalElems, realDst, rawSrc, CopyOperand());
    } else
        allOps[next++]->invalidate();
}

// Rewrite int4 -> int upconversion using byte operations.
// May need to be run twice.
//
// Example input:
//    mov (16)   r0.0<1>:ub   r1.0<1>:u4
// Output:
//    and (8)    r0.0<2>:ub   r1.0<1>:ub   0xF:uw
//    shr (8)    r0.1<2>:ub   r1.0<1>:ub   4:uw
//
void CopyPlan::planInt4Upconversion(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    i.sat = false;

    if (hw >= HW::Xe3p && one_of(getBits(i.dst.type), {8, 16}))
        if (planShflUpconvertXe3p(i))
            return;

    bool s4 = (i.src0.type == DataType::s4);

    if (i.src0.stride == 1 && i.simd > 1) {
        // Split into high and low nybble conversions first, if needed.
        // If dst stride is too large, copy through uw<1>.
        //   This path allows 2D regions.
        if (i.dst.stride >= (s4 ? 2 : 4)) {
            auto ie = splitMultiple<3>(i);
            auto t = newTemp(DataType::uw, i.simd, 1);
            ie[0]->simd /= 2;
            ie[0]->op = s4 ? Opcode::shl : Opcode::and_;
            ie[0]->src0.type = DataType::ub;
            ie[0]->src0.offset /= 2;
            if (ie[0]->src0.width > 1) {
                ie[0]->src0.width /= 2;
                if (ie[0]->src0.width == 1) {
                    ie[0]->src0.stride = ie[0]->src0.vs / 2;
                    ie[0]->src0.vs = ie[0]->src0.width = 0;
                }
            }
            ie[0]->src1 = s4 ? 4 : 0xF;
            ie[0]->dst = t;
            ie[0]->dst.stride *= 2;

            ie[1]->simd /= 2;
            ie[1]->op = s4 ? Opcode::mov : Opcode::shr;
            ie[1]->src0 = ie[0]->src0;
            if (!s4) ie[1]->src1 = 4;
            ie[1]->dst = ie[0]->dst;
            ie[1]->dst.offset++;

            ie[2]->op = s4 ? Opcode::asr : Opcode::mov;
            ie[2]->src0 = t;
            ie[2]->src0.type = s4 ? DataType::b : DataType::ub;
            ie[2]->src0.stride *= 2;
            ie[2]->src0.offset *= 2;
            if (s4) ie[2]->src1 = 4;
        } else {
            auto &i0 = i;
            i0.dst.stride *= 2;
            i0.src0.stride *= 2;
            i0.simd /= 2;
            split(i, true);
            i0.dst.offset += i0.dst.stride / 2;
            i0.src0.offset += i0.src0.stride / 2;
        }
    } else {
        bool even = (i.src0.offset % 2 == 0);
        if (i.dst.stride > 4) stub("Unsupported stride.");
        i.src0.stride /= 2;
        i.src0.offset /= 2;
        if (getBits(i.dst.type) < 8) {
            i.dst.type = DataType::ub;
            i.dst.stride /= 2;
            i.dst.offset /= 2;
        }

        if (even) {
            // Low nybbles
            if (s4) {
                auto &i0 = i, &i1 = split(i);
                if (getBytes(i0.dst.type) == 1)
                    i0.dst = newTemp(DataType::uw, i0.simd, (i0.src0.stride > 2) ? 2 : 1);
                i1.src0 = i0.dst;
                auto shift = getBytes(i0.dst.type) * 8 - 4;

                i0.op = Opcode::shl;
                i0.src0.type = DataType::ub;
                i0.src1 = shift;

                i1.op = Opcode::asr;
                i1.src0.type = asSigned(i1.src0.type);
                i1.src1 = shift;
            } else {
                i.op = Opcode::and_;
                i.src0.type = DataType::ub;
                i.src1 = 0xF;
            }
        } else {
            // High nybble
            auto tmp = newTemp(i.dst.type, i.simd, i.dst.stride, 1, 0);
            auto s0BO = i.src0.offset * getBytes(i.src0.type);
            auto dBO = i.dst.offset * getBytes(i.dst.type);
            if (hw == ngen::HW::Xe3p && (s0BO != dBO)) {
                auto ie = splitMultiple<2>(i);

                // High nybble
                ie[0]->op = s4 ? Opcode::asr : Opcode::shr;
                ie[0]->src0.type = s4 ? DataType::b : DataType::ub;
                ie[0]->src1 = 4;
                ie[0]->dst = tmp;

                ie[1]->op = Opcode::mov;
                ie[1]->src0 = tmp;

            } else {
                i.op = s4 ? Opcode::asr : Opcode::shr;
                i.src0.type = s4 ? DataType::b : DataType::ub;
                i.src1 = 4;
            }
        }
    }
}

// Shift sub-byte data into the high or low bits of a 16-bit channel.
// Bit width is read directly from i.src0.type.
//
// Example input (2-bit, dense packing):
//    mov (16)   r0.0<1>:b16_h2x  r1.0<1>:u2
// Output:
//    shl (4)    r0.0<4>:uw       r1.0<1>:ub     14:uw
//    shl (4)    r0.1<4>:uw       r1.0<1>:ub     12:uw
//    shl (4)    r0.2<4>:uw       r1.0<1>:ub     10:uw
//    shl (4)    r0.3<4>:uw       r1.0<1>:ub     8:uw
//
// Example input (4-bit, dense packing):
//    mov (16)   r0.0<1>:b16_h4x  r1.0<1>:u4
// Output:
//    shl (8)    r0.0<2>:uw       r1.0<1>:ub     12:uw
//    shl (8)    r0.1<2>:uw       r1.0<1>:ub     8:uw
//
void CopyPlan::planSubByteShifts(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    i.sat = false;

    const int bits = getBits(i.src0.type);
    const int perByte = 8 / bits;
    bool high = one_of(i.dst.type, {ngen_b16_h2x(), ngen_b16_h4x()});

    const auto sstride = i.src0.stride;
    const auto dstride = i.dst.stride;
    const auto stride_adjustment = div_up(perByte, sstride);

    i.op = high ? Opcode::shl : Opcode::shr;
    i.dst.stride *= stride_adjustment;
    i.dst.type = DataType::uw;
    i.src0.stride = stride_adjustment * sstride / perByte;
    i.src0.type = DataType::ub;
    i.simd /= stride_adjustment;

    for (int k = 0; k * sstride < perByte; k++) {
        auto &i1 = split(i, false);
        i1.dst.offset += k * dstride;
        i1.src0.offset += k * sstride;
        auto p = i1.src0.offset % perByte;
        i1.src0.offset /= perByte;
        i1.src1 = high ? (16 - bits * (p + 1)) : (p * bits);
    }
    i.invalidate();
}

void CopyPlan::planInt2Downconversion(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    int simd = i.simd;

    auto st = i.src0.type, dt = i.dst.type;
    bool s2 = (dt == DataType::s2);
    if (!one_of(dt, {DataType::s2, DataType::u2})) stub();
    if (isD(st) || isQ(st)) {
        copyThrough(i, (isSigned(st) && s2) ? DataType::w : DataType::uw, 1);
        return;
    }
    auto ddst = CopyOperand(i.dst);
    int tmp_elems = ddst.stride > 4 ? simd * 2 : simd;
    auto tmp = newTemp(DataType::uw, tmp_elems, 1);

    if (i.sat) {
        auto ie = splitMultiple<3>(i);
        auto ssrc = i.src0;
        if (ssrc.overwrite && isW(st)) {
            tmp = ssrc;
            tmp.type = DataType::uw;
        }
        for (int i = 0; i < 3; ++i)
            ie[i]->sat = false;

        ie[0]->op = Opcode::sel;
        ie[0]->cmod = ConditionModifier::lt;
        ie[0]->dst = tmp;
        ie[0]->src0 = ssrc;
        ie[0]->src1 = s2 ? 1 : 3;
        ie[0]->sat = isSigned(ssrc.type) && !s2;

        if (isSigned(ssrc.type) && s2) {
            ie[0]->dst.type = tmp.type = asSigned(tmp.type);

            ie[1]->op = Opcode::sel;
            ie[1]->cmod = ConditionModifier::ge;
            ie[1]->dst = tmp;
            ie[1]->src0 = tmp;
            ie[1]->src1 = -2;
        } else
            ie[1]->invalidate();

        ie[2]->op = Opcode::mov;
        ie[2]->dst = ddst;
        ie[2]->src0 = tmp;
        ie[2]->src0.range = dt;
        return;
    }

    auto ie = splitMultiple<10>(i);
    auto osrc = i.src0;
    auto stmp = newTemp(DataType::uw, simd, 1);
    auto dtmp = newTemp(DataType::uw, simd/4, 1);
    auto sstmp = newTemp(DataType::uw, simd/2 * ddst.stride, 1);

    ie[0]->op = Opcode::mov;
    ie[0]->dst = stmp;
    ie[0]->src0 = osrc;

    if (simd >= 4 && ddst.stride == 1) {
        // Dense destination, use bfn to pack values in bytes
        ie[1]->op = Opcode::mov;
        ie[1]->simd = simd/4;
        ie[1]->dst = dtmp;
        ie[1]->src0 = stmp;
        ie[1]->src0.stride *= 4;

        ie[2]->op = Opcode::shl;
        ie[2]->simd = simd/4;
        ie[2]->dst = sstmp;
        ie[2]->dst.stride *= 2;
        ie[2]->src0 = stmp;
        ie[2]->src0.offset += 1;
        ie[2]->src0.stride *= 4;
        ie[2]->src1 = Immediate::uw(0x2);

        ie[3]->op = Opcode::bfn;
        ie[3]->ctrl = 0xAC;
        ie[3]->simd = simd/4;
        ie[3]->dst = dtmp;
        ie[3]->src0 = dtmp;
        ie[3]->src1 = sstmp;
        ie[3]->src1.stride *= 2;
        ie[3]->src2 = Immediate::uw(0x03);

        ie[4]->op = Opcode::shl;
        ie[4]->simd = simd/4;
        ie[4]->dst = sstmp;
        ie[4]->dst.stride *= 2;
        ie[4]->src0 = stmp;
        ie[4]->src0.offset += 2;
        ie[4]->src0.stride *= 4;
        ie[4]->src1 = Immediate::uw(0x4);

        ie[5]->op = Opcode::bfn;
        ie[5]->ctrl = 0xAC;
        ie[5]->simd = simd/4;
        ie[5]->dst = dtmp;
        ie[5]->src0 = dtmp;
        ie[5]->src1 = sstmp;
        ie[5]->src1.stride *= 2;
        ie[5]->src2 = Immediate::uw(0x0F);

        ie[6]->op = Opcode::shl;
        ie[6]->simd = simd/4;
        ie[6]->dst = sstmp;
        ie[6]->dst.stride *= 2;
        ie[6]->src0 = stmp;
        ie[6]->src0.offset += 3;
        ie[6]->src0.stride *= 4;
        ie[6]->src1 = Immediate::uw(0x6);

        ie[7]->op = Opcode::bfn;
        ie[7]->ctrl = 0xAC;
        ie[7]->simd = simd/4;
        ie[7]->dst = dtmp;
        ie[7]->src0 = dtmp;
        ie[7]->src1 = sstmp;
        ie[7]->src1.stride *= 2;
        ie[7]->src2 = Immediate::uw(0x3F);

        ie[8]->op = Opcode::mov;
        ie[8]->simd = simd/4;
        ie[8]->dst = tmp;
        ie[8]->dst.type = DataType::ub;
        ie[8]->dst.stride = 1;
        ie[8]->src0 = dtmp;
        ie[8]->src0.stride *= 2;
        ie[8]->src0.type = DataType::ub;

        ie[9]->op = Opcode::mov;
        ie[9]->simd = simd/4;
        ie[9]->dst = ddst;
        ie[9]->dst.type = DataType::ub;
        if (ddst.vs != 0)
            ie[9]->dst.stride = ddst.vs / ddst.width;
        ie[9]->dst.offset /= 4;
        ie[9]->src0 = tmp;
        ie[9]->src0.stride = 1;
        ie[9]->src0.type = DataType::ub;
    } else if (ddst.stride >= 2) {
        // Use multiple bfn instructions to insert
        // values into the destination.
        auto offset = (ddst.offset % 8) * 2;
        int simd_factor = std::max(1, 8 / ddst.stride);
        int new_ins = std::min(simd, simd_factor);

        for (int j = 0; j < new_ins; ++j) {
            ie[2*j + 1]->op = Opcode::shl;
            ie[2*j + 1]->simd = std::max(simd/simd_factor, 1);
            ie[2*j + 1]->dst = sstmp;
            ie[2*j + 1]->dst.stride *= 2;
            ie[2*j + 1]->src0 = stmp;
            ie[2*j + 1]->src0.offset += j * ie[3]->src0.stride;
            ie[2*j + 1]->src0.stride *= simd_factor;
            ie[2*j + 1]->src1 = Immediate::uw((offset + 2*ddst.stride*j) % 16);

            const auto stride = std::max(ddst.stride / 8, 1);
            ie[2*j + 2]->op = Opcode::bfn;
            ie[2*j + 2]->simd = std::max(simd/simd_factor, 1);
            ie[2*j + 2]->ctrl = 0xCA;
            ie[2*j + 2]->dst = ddst;
            ie[2*j + 2]->dst.type = DataType::uw;
            ie[2*j + 2]->dst.stride = stride;
            ie[2*j + 2]->dst.offset /= 8;
            ie[2*j + 2]->src0 = ddst;
            ie[2*j + 2]->src0.type = DataType::uw;
            ie[2*j + 2]->src0.stride = stride;
            ie[2*j + 2]->src0.offset /= 8;
            if (offset + 2*ddst.stride*j >= 16) {
                ie[2*j + 2]->dst.offset += 1;
                ie[2*j + 2]->src0.offset += 1;
            }
            ie[2*j + 2]->src1 = sstmp;
            ie[2*j + 2]->src1.stride *= 2;
            ie[2*j + 2]->src2 = Immediate::uw(0x3 << ((offset + 2*ddst.stride*j) % 16));
        }

        for (int j = new_ins * 2 + 1; j < 10; ++j) {
            ie[j]->invalidate();
        }
    } else {
        stub("Unsupported int2 downconversion configuration");
    }
}

void CopyPlan::planInt4Downconversion(CopyInstruction &i)
{
    if (i.src0.neg || i.hasCMod()) stub("Unsupported modifier");
    int simd = i.simd;

    auto st = i.src0.type, dt = i.dst.type;
    bool s4 = (dt == DataType::s4);
    if (!one_of(dt, {DataType::s4, DataType::u4})) stub();
    if (isD(st) || isQ(st)) {
        copyThrough(i, (isSigned(st) && s4) ? DataType::w : DataType::uw, 1);
        return;
    }
    auto ddst = CopyOperand(i.dst);
    auto ssrc = CopyOperand(i.src0);
    int tmp_elems = ddst.stride > 4 ? simd * 2 : simd;
    auto tmp = newTemp(DataType::uw, tmp_elems, 1);

    int sStride = ssrc.stride * getBytes(ssrc.type) * 2;
    int dStride = ddst.stride / (getBytes(ssrc.type) * 2);
    if (i.sat) {
        auto ie = splitMultiple<3>(i);
        auto ssrc = i.src0;
        if (ssrc.overwrite && isW(st)) {
            tmp = ssrc;
            tmp.type = DataType::uw;
        }
        for (int i = 0; i < 3; ++i)
            ie[i]->sat = false;

        ie[0]->op = Opcode::sel;
        ie[0]->cmod = ConditionModifier::lt;
        ie[0]->dst = tmp;
        ie[0]->src0 = ssrc;
        ie[0]->src1 = s4 ? 7 : 15;
        ie[0]->sat = isSigned(ssrc.type) && !s4;

        if (isSigned(ssrc.type) && s4) {
            ie[0]->dst.type = tmp.type = asSigned(tmp.type);

            ie[1]->op = Opcode::sel;
            ie[1]->cmod = ConditionModifier::ge;
            ie[1]->dst = tmp;
            ie[1]->src0 = tmp;
            ie[1]->src1 = -8;
        } else
            ie[1]->invalidate();

        ie[2]->op = Opcode::mov;
        ie[2]->dst = ddst;
        ie[2]->src0 = tmp;
        ie[2]->src0.range = dt;
        return;
    }

    auto ie = splitMultiple<5>(i);
    auto osrc = i.src0;
    auto stmp = newTemp(DataType::uw, simd, 1);

    ie[0]->op = Opcode::mov;
    ie[0]->dst = stmp;
    ie[0]->src0 = osrc;

    if (isW(osrc.type) && osrc.stride == 1 && osrc.overwrite) {
        stmp = osrc;
        ie[0]->invalidate();
    }

    // Special case for expanding 4-bit values already at least byte aligned.
    if (ddst.stride >= sStride && sStride > 1 && ssrc.type == DataType::ub && simd >= 4) {
        int dst_mask = 0x0;
        int mask_granularity = std::min<int>(4, ddst.stride);
        switch (mask_granularity) {
            case 2:
                dst_mask = 0x0f0f << ((ddst.offset % 2) * 4);
                break;
            case 4:
                dst_mask = 0x000f << ((ddst.offset % 4) * 4);
                break;
            default: stub();
        }

        if (ddst.stride > sStride) {
            ie[0]->op = Opcode::mov;
            ie[0]->simd = simd;
            ie[0]->dst = tmp;
            ie[0]->dst.type = DataType::ud;
            ie[0]->dst.stride = 1;
            ie[0]->src0 = Immediate::d(0);

            int tmp_off = (ddst.offset / 4) * 2;
            ie[1]->op = Opcode::mov;
            ie[1]->simd = simd;
            ie[1]->dst = tmp;
            ie[1]->dst.type = ssrc.type;
            ie[1]->dst.stride = dStride;
            ie[1]->dst.offset = tmp_off;
            ie[1]->src0 = ssrc;

            tmp.type = ssrc.type;
            tmp.stride = ssrc.stride;
            ssrc = tmp;
        } else {
            ie[0]->invalidate();
            ie[1]->invalidate();
        }

        int tmp_off = ddst.offset / 4;
        if (ddst.offset % mask_granularity) {
            ie[2]->op = Opcode::shl;
            ie[2]->simd = simd;
            ie[2]->dst = ssrc;
            ie[2]->dst.type = DataType::uw;
            ie[2]->dst.stride = 2;
            ie[2]->dst.offset = tmp_off;
            ie[2]->src0 = ssrc;
            ie[2]->src0.type = DataType::uw;
            ie[2]->src0.stride = 2;
            ie[2]->src0.offset = tmp_off;
            ie[2]->src1 = Immediate::uw(0x4 * (ddst.offset % mask_granularity));
        } else {
            ie[2]->invalidate();
        }

        ie[3]->op = Opcode::bfn;
        ie[3]->ctrl = 0xCA;
        ie[3]->simd = simd;
        ie[3]->dst = ddst;
        ie[3]->dst.type = DataType::uw;
        ie[3]->dst.stride = ssrc.stride;
        ie[3]->dst.offset = ddst.offset / 4;
        ie[3]->src0 = ddst;
        ie[3]->src0.type = DataType::uw;
        ie[3]->src0.stride = ssrc.stride;
        ie[3]->src0.offset = ddst.offset / 4;
        ie[3]->src1 = ssrc;
        ie[3]->src1.type = DataType::uw;
        ie[3]->src1.stride = ssrc.stride;
        ie[3]->src1.offset = tmp_off;
        ie[3]->src2 = Immediate::uw(dst_mask);

        ie[4]->invalidate();
    } else if (simd > 1 && ddst.stride == 1) {
        if (hw < HW::Xe3p) {
            ie[1]->op = Opcode::shl;
            ie[1]->simd = simd / 2;
            ie[1]->dst = stmp;
            ie[1]->dst.offset += 1;
            ie[1]->dst.stride *= 2;
            ie[1]->src0 = stmp;
            ie[1]->src0.offset += 1;
            ie[1]->src0.stride *= 2;
            ie[1]->src1 = Immediate::uw(0x4);
        } else {
            // Single-instruction shift + alignment
            // Note: clobbers even words in tmp
            ie[1]->op = Opcode::shr;
            ie[1]->simd = simd / 2;
            ie[1]->dst = tmp;
            ie[1]->dst.type = DataType::uw;
            ie[1]->dst.stride = stmp.stride * 2;
            ie[1]->dst.offset = stmp.offset;
            ie[1]->src0 = stmp;
            ie[1]->src0.type = DataType::ud;
            ie[1]->src0.offset /= 2;
            ie[1]->src1 = Immediate::uw(0xC);
        }

        ie[2]->op = Opcode::bfn;
        // Note: Xe3p path has junk in low 4 bits of src1.
        ie[2]->ctrl = hw < HW::Xe3p ? 0xEC : 0xAC;
        ie[2]->simd = simd / 2;
        ie[2]->dst = hw < HW::Xe3p ? stmp : tmp;
        ie[2]->dst.stride *= 2;
        ie[2]->dst.offset = stmp.offset;
        ie[2]->src0 = stmp;
        ie[2]->src0.stride *= 2;
        ie[2]->src1 = ie[1]->dst;
        ie[2]->src2 = Immediate::uw(0x0F);

        if (simd > 2) {
            ie[3]->op = Opcode::mov;
            ie[3]->simd = simd / 2;
            ie[3]->dst = stmp;
            ie[3]->dst.type = DataType::ub;
            ie[3]->dst.stride = 1;
            ie[3]->src0 = ie[2]->dst;
            ie[3]->src0.stride *= 2;
            ie[3]->src0.offset *= 2;
            ie[3]->src0.type = DataType::ub;

            ie[4]->op = Opcode::mov;
            ie[4]->simd = simd / 2;
            ie[4]->dst = ddst;
            ie[4]->dst.type = DataType::ub;
            if (ddst.vs != 0)
                ie[4]->dst.stride = ddst.vs / ddst.width;
            ie[4]->dst.offset /= 2;
            ie[4]->src0 = ie[3]->dst;
        } else {
            ie[3]->op = Opcode::mov;
            ie[3]->simd = simd / 2;
            ie[3]->dst = ddst;
            ie[3]->dst.type = DataType::ub;
            ie[3]->dst.stride = 1;
            ie[3]->dst.offset /= 2;
            ie[3]->src0 = ie[2]->dst;
            ie[3]->src0.stride *= 2;
            ie[3]->src0.type = DataType::ub;

            ie[4]->invalidate();
        }
    } else {
        const auto stride = ddst.stride / 2;
        const auto offset = ddst.offset & 1;
        const auto mask = (uint16_t)(0xF << (4 * offset));

        ie[1]->op = Opcode::mov;
        ie[1]->dst = tmp;
        ie[1]->src0 = ddst;
        ie[1]->src0.type = DataType::ub;
        ie[1]->src0.stride = stride;
        ie[1]->src0.offset /= 2;

        if (offset) {
            ie[2]->op = Opcode::shl;
            ie[2]->simd = simd;
            ie[2]->dst = stmp;
            ie[2]->dst.stride = 1;
            ie[2]->src0 = stmp;
            ie[2]->src0.stride = 1;
            ie[2]->src1 = 4;
        } else {
            ie[2]->invalidate();
        }

        ie[3]->op = Opcode::bfn;
        ie[3]->ctrl = 0xCA;
        ie[3]->simd = simd;
        ie[3]->dst = tmp;
        ie[3]->src0 = stmp;
        ie[3]->src1 = tmp;
        ie[3]->src2 = 0xFFFF ^ mask;

        ie[4]->op = Opcode::mov;
        ie[4]->simd = simd;
        ie[4]->dst = ddst;
        ie[4]->dst.type = DataType::ub;
        ie[4]->dst.stride = stride;
        ie[4]->dst.offset /= 2;
        ie[4]->src0 = tmp;
        ie[4]->src0.type = DataType::ub;
        ie[4]->src0.stride *= 2;
    }
}

// e8m0->f conversion.
void CopyPlan::planE8M0ToF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    // Emulation sequence for mov y:f x:e8m0:
    //   shl         y.hi:uw  x:ub     7
    //   add (sat)   y.lo:uw  x:ub     -254
    //   sel (ge)    y.hi:uw  y.hi:uw  0x40
    // Note:
    //  * 0x00:e8m0 -> 0x00400000:f
    //  * 0xFF:e8m0 -> 0x7F800001:f (nan)

    i.dst.type = DataType::uw;
    i.dst.offset *= 2;
    i.dst.stride *= 2;
    i.src0.type = DataType::ub;

    auto ie = splitMultiple<3>(i);

    ie[0]->op = Opcode::shl;
    ie[0]->dst.offset += 1;
    ie[0]->src1 = 7;

    ie[1]->op = Opcode::add;
    ie[1]->src1 = -254;
    ie[1]->sat = true;

    ie[2]->op = Opcode::sel;
    ie[2]->cmod = ConditionModifier::ge;
    ie[2]->dst.offset += 1;
    ie[2]->src0 = ie[2]->dst;
    ie[2]->src1 = 0x40;
}

// Emulation sequence for hf8->hf conversion.
void CopyPlan::planEmulatedHF8ToHF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    // Emulation sequence for mov y:hf x:hf8:
    // shl          y:uw    x:ub    8               /* already done */
    // asr          y:w     y:w     1
    // and          y:uw    y:uw    0xBFFF
    // mul          y:hf    y:hf    0x7880          /* overflow 0x7F -> inf */
    // mul          y:hf    y:hf    0x1F1C          /* scale to correct hf values */
    // mad          y:hf    y:hf    y:hf    0:hf    /* convert inf -> nan */

    auto ie = splitMultiple<5>(i);

    auto y = i.dst, yUW = y, yW = y;
    yUW.type = DataType::uw;
    yW.type = DataType::w;

    ie[0]->op = Opcode::asr;
    ie[0]->src0 = ie[0]->dst = yW;
    ie[0]->src1 = 1;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = ie[1]->dst = yUW;
    ie[1]->src1 = 0xBFFF;

    ie[2]->op = Opcode::mul;
    ie[2]->src0 = ie[2]->dst = y;
    ie[2]->src1 = Immediate::hf(0x7880);

    ie[3]->op = Opcode::mul;
    ie[3]->src0 = ie[3]->dst = y;
    ie[3]->src1 = Immediate::hf(0x1F1C);

    ie[4]->op = Opcode::mad;
    ie[4]->src0 = ie[4]->src1 = ie[4]->dst = y;
    ie[4]->src2 = Immediate::hf(0);
}

// Emulation sequence for hf8->bf conversion.
void CopyPlan::planEmulatedHF8ToBF(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    if (!bfArithmeticOK(i)) stub();

    // Emulation sequence for mov y:bf x:hf8:
    // shl          y:uw    x:ub      8               /* already done */
    // asr          y:w     y:w       4
    // and          y:uw    y:uw      0x87FF
    // cmp (ge)f0   null:hf (abs)y:hf 0x07F0:hf       /* NaN check */
    // mul          y:bf    y:bf      0x7B80:bf
    // (f0) or      y:uw    y:uw      0x7FFF

    auto ie = splitMultiple<5>(i);

    auto y = i.dst, yUW = y, yW = y, yHF = y;
    yUW.type = DataType::uw;
    yW.type = DataType::w;
    yHF.type = DataType::hf;
    y.type = DataType::bf;

    auto f = newFlag(i.simd);

    ie[0]->op = Opcode::asr;
    ie[0]->src0 = ie[0]->dst = yW;
    ie[0]->src1 = 4;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = ie[1]->dst = yUW;
    ie[1]->src1 = 0x87FF;

    ie[2]->op = Opcode::cmp;
    ie[2]->src0 = abs(yHF);
    ie[2]->src1 = Immediate::hf(0x07F0);
    ie[2]->dst = CopyOperand();
    ie[2]->dst.stride = yHF.stride;
    ie[2]->dst.type = DataType::hf;
    ie[2]->cmod = ConditionModifier::ge;
    ie[2]->flag = f;

    ie[3]->op = Opcode::mul;
    ie[3]->src0 = ie[3]->dst = y;
    ie[3]->src1 = bfImmediate(0x7B80, false);

    ie[4]->op = Opcode::or_;
    ie[4]->src0 = ie[4]->dst = yUW;
    ie[4]->src1 = 0x7FFF;
    ie[4]->flag = f;
}

// Emulation sequence for e2m1->hf conversion.
void CopyPlan::planEmulatedF4ToHF(CopyInstruction &i)
{
    // Emulation sequence for mov y:hf x:e2m1:
    //   shl                 y:uw    x:u4    12                /* y may have junk in lower bits */
    //   asr                 y:w     y:w     3
    //   and                 y:uw    y:uw    0x8E00
    //   mul                 y:hf    y:hf    16384:hf

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<3>(i);

    auto y = i.dst, yUW = y, yW = y;
    yUW.type = DataType::uw;
    yW.type = DataType::w;

    ie[0]->op = Opcode::asr;
    ie[0]->src0 = ie[0]->dst = yW;
    ie[0]->src1 = 3;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = ie[1]->dst = yUW;
    ie[1]->src1 = 0x8E00;

    ie[2]->op = Opcode::mul;
    ie[2]->src0 = ie[2]->dst = y;
    ie[2]->src1 = Immediate::hf(0x7400);
}

// Emulation sequence for e2m1->bf conversion.
void CopyPlan::planEmulatedF4ToBF(CopyInstruction &i)
{
    // Emulation sequence for mov y:bf x:e2m1:
    //   shl                 y:uw    x:u4    12                /* y may have junk in lower bits */
    //   asr                 y:w     y:w     6
    //   and                 y:uw    y:uw    0x81C0
    //   mul                 y:bf    y:bf    0x7E80:bf

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");
    if (!bfArithmeticOK(i)) stub();      /* need bf/f arithmetic */

    auto ie = splitMultiple<3>(i);

    auto y = i.dst, yUW = y, yW = y;
    yUW.type = DataType::uw;
    yW.type = DataType::w;

    ie[0]->op = Opcode::asr;
    ie[0]->src0 = ie[0]->dst = yW;
    ie[0]->src1 = 6;

    ie[1]->op = Opcode::and_;
    ie[1]->src0 = ie[1]->dst = yUW;
    ie[1]->src1 = 0x81C0;

    ie[2]->op = Opcode::mul;
    ie[2]->src0 = ie[2]->dst = y;
    ie[2]->src1 = bfImmediate(0x7E80, false);
}

// Emulation sequence for nf4->hf conversion.
void CopyPlan::planEmulatedNF4ToHF(CopyInstruction &i)
{
    // nf4->hf conversion.
    //
    // After scaling and shifting, the conversion is essentially
    //   applying the inverse error function.
    //
    // The fast approximation here is adapted from the
    //   central region approximation in:
    //         "Approximating the erfinv function," Mike Giles,
    //         https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto y = i.dst;
    auto ie = splitMultiple<8>(i);

    auto t0 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);
    auto t1 = newTemp(DataType::hf, i.simd, y.stride, 0, y.offset);
    auto f = newFlag(i.simd);

    // On entry, data has been zero-extended into 16 bits.

    // 1. Reinterpret as denormal f16 and scale and shift to [-1, 1].
    ie[0]->op = Opcode::mad;
    ie[0]->src0 = Immediate::hf(0x9700);    // -7 * 2^(-12)
    ie[0]->src1 = y;
    ie[0]->src2 = Immediate::hf(0x6C00);    // 2^12
    ie[0]->dst = t0;
    ie[0]->cmod = ConditionModifier::lt;
    ie[0]->flag = f;

    // 1a. Scale positive half of range to [0, 1].
    ie[1]->op = Opcode::mul;
    ie[1]->src0 = t0;
    ie[1]->src1 = Immediate::hf(0x6000);    // 2^9
    ie[1]->dst = y;

    // 1b. Scale negative half of range to [-1, 0].
    ie[2]->op = Opcode::mad;
    ie[2]->src0 = Immediate::hf(0x8BFF);    // ~2^(-12)
    ie[2]->src1 = t0;
    ie[2]->src2 = Immediate::hf(0x6092);    // ~(2^12 / 7)
    ie[2]->dst = y;
    ie[2]->flag = f;

    // 2. erfinv approximation, with special handling for +/-1
    //     y*(c0 + c1*w * c2*w^2), where w = log2(1 - y*y).
    ie[3]->op = Opcode::mad;
    ie[3]->src0 = Immediate::hf(0x3BFF);    // ~1
    ie[3]->src1 = y;
    ie[3]->src2 = -y;
    ie[3]->dst = t0;                        // w
    ie[3]->cmod = ConditionModifier::gt;
    ie[3]->flag = f;

    ie[4]->op = Opcode::math;
    ie[4]->ctrl = static_cast<uint8_t>(MathFunction::log);
    ie[4]->dst = ie[4]->src0 = t0;

    ie[5]->op = Opcode::mad;
    ie[5]->src0 = Immediate::hf(0x2EA3);    // c1
    ie[5]->src1 = t0;
    ie[5]->src2 = Immediate::hf(0x1DA1);    // c2
    ie[5]->dst = t1;

    ie[6]->op = Opcode::mad;
    ie[6]->src0 = Immediate::hf(0x3912);    // c0
    ie[6]->src1 = t0;
    ie[6]->src2 = -t1;
    ie[6]->dst = t1;

    ie[7]->op = Opcode::mul;
    ie[7]->src0 = ie[7]->dst = y;
    ie[7]->src1 = t1;
    ie[7]->flag = f;
}

// Emulation sequence for hf->hf8 conversion.
void CopyPlan::planEmulatedHFToHF8(CopyInstruction &i)
{
    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto ie = splitMultiple<11>(i);

    // mad (lt)f0   t1:hf   0x8008:hf  (abs)x:hf  0x0200:hf  /* hf8 denormal check */
    // mul          t0:hf   (abs)x:hf  0x1C00:hf             /* adjust exponent */
    // (f0) mad     t0:hf   0x0400:hf  t1:hf      0x5800:hf  /* manual rounding for denormals */
    // cmp (lt)f1   null    (abs)x:hf  0x5FC0:hf             /* overflow/nan check */
    // add          t0:uw   t0:uw      -0x40                 /* round */
    // and (nz)f0   null    t0:uw      0xFF
    // shl          t0:uw   t0:uw      1                     /* move to high byte */
    // (f0) add     t0:uw   t0:uw      0x100
    // (~f1) mov    t0:uw   0x7F00
    // bfn.0xCA     t0:uw   x:uw       t0:uw      0x7FFF     /* copy sign */
    // mov          y:ub    t0_hi:ub

    auto x = i.src0;
    auto t0 = newTemp(DataType::hf, i.simd, x.stride);
    auto t1 = newTemp(DataType::hf, i.simd, x.stride);
    auto t0UW = t0, t1UW = t1;
    t0UW.type = t1UW.type = DataType::uw;
    auto f0 = newFlag(i.simd);
    auto f1 = newFlag(i.simd);

    ie[0]->op = Opcode::mad;
    ie[0]->dst = t1;
    ie[0]->src0 = Immediate::hf(0x8008);
    ie[0]->src1 = abs(x);
    ie[0]->src2 = Immediate::hf(0x0200);
    ie[0]->cmod = ConditionModifier::lt;
    ie[0]->flag = f0;

    ie[1]->op = Opcode::mul;
    ie[1]->dst = t0;
    ie[1]->src0 = abs(x);
    ie[1]->src1 = Immediate::hf(0x1C00);

    ie[2]->op = Opcode::mad;
    ie[2]->dst = t0;
    ie[2]->src0 = Immediate::hf(0x0400);
    ie[2]->src1 = t1;
    ie[2]->src2 = Immediate::hf(0x5800);
    ie[2]->flag = f0;

    ie[3]->op = Opcode::cmp;
    ie[3]->dst = CopyOperand();
    ie[3]->dst.type = DataType::hf;
    ie[3]->dst.stride = x.stride;
    ie[3]->dst.offset = x.offset;
    ie[3]->src0 = abs(x);
    ie[3]->src1 = Immediate::hf(0x5FC0);
    ie[3]->cmod = ConditionModifier::lt;
    ie[3]->flag = f1;

    ie[4]->op = Opcode::add;
    ie[4]->dst = ie[4]->src0 = t0UW;
    ie[4]->src1 = -0x40;

    ie[5]->op = Opcode::and_;
    ie[5]->dst = CopyOperand();
    ie[5]->dst.type = DataType::uw;
    ie[5]->src0 = t0UW;
    ie[5]->src1 = 0x0FF;
    ie[5]->cmod = ConditionModifier::nz;
    ie[5]->flag = f0;

    ie[6]->op = Opcode::shl;
    ie[6]->dst = ie[6]->src0 = t0UW;
    ie[6]->src1 = 1;

    ie[7]->op = Opcode::add;
    ie[7]->dst = ie[7]->src0 = t0UW;
    ie[7]->src1 = 0x100;
    ie[7]->flag = f0;

    ie[8]->op = Opcode::mov;
    ie[8]->dst = t0UW;
    ie[8]->src0 = 0x7F00;
    ie[8]->flag = f1;
    ie[8]->flag.neg = true;

    ie[9]->op = Opcode::bfn;
    ie[9]->dst = ie[9]->src1 = t0UW;
    ie[9]->src0 = x;
    ie[9]->src0.type = DataType::uw;
    ie[9]->src2 = 0x8000;
    ie[9]->ctrl = 0xEC;  // (s0 & s2) | s1

    ie[10]->op = Opcode::mov;
    ie[10]->src0 = t0;
    ie[10]->src0.type = ie[10]->dst.type = DataType::ub;
    ie[10]->src0.stride *= 2;
    ie[10]->src0.offset++;
}

// hf->e2m1 sequences.
void CopyPlan::planEmulatedHFToF4(CopyInstruction &i)
{
    // Emulation sequence for mov y:e2m1 x:hf
    //
    //        mad (lt)f0   t1:hf   0x8004:hf  (abs)x:hf  0x2:hf                    /* denormal check */
    //        sel (lt)     t0:hf   (abs)x:hf  0x4600:hf                            /* clamp */
    //        mul          t0:hf   t0:hf      0x400:hf                             /* adjust exponent */
    //   (f0) mad          t0:hf   0x800:hf   t1:hf      0x6000:hf                 /* manual denormal rounding */
    //        add          t0:uw   t0:uw      -0x100                               /* RTNE */
    //        and (nz)f0   null    t0:uw      0x3ff
    //   (f0) add          t0:uw   t0:uw      0x200
    //        shl          t0:uw   t0:uw      3                                    /* shift exponent field */
    //        bfn.0xCA     t0:uw   x:uw       t0:uw      0x7FFF                    /* copy sign */
    //        shr          t0:uw   t0:uw      12                                   /* move to lowest nybble */
    //        mov          y:u4    t0:uw                                           /* pack nybbles */

    if (i.src0.neg || i.sat || i.hasCMod()) stub("Unsupported modifier");

    auto x = i.src0;
    auto y = i.dst;

    if (x.stride > 1) {
        // Pack inputs to SIMD1 if they are strided for better efficiency.
        repositionSrc(i, 0, 1, 0);
        return;
    }

    if (hw >= HW::Xe3p) {
        auto t0 = newTemp(DataType::hf, i.simd / 2, 1);
        auto t1 = newTemp(DataType::hf, i.simd / 2, 1);
        auto ie = splitMultiple<5>(i);
        int simd = i.simd;
        int dstStride = y.stride;
        bool needPack = (y.stride > 1 || y.width > 2);

        ie[0]->op = Opcode::mov;
        ie[0]->simd = simd / 4;
        ie[0]->dst = t0;
        ie[0]->dst.stride = 1;
        ie[0]->dst.type = DataType::ud;
        ie[0]->src0 = x;
        ie[0]->src0.type = DataType::ud;
        ie[0]->src0.stride = 2;

        ie[1]->op = Opcode::mov;
        ie[1]->simd = simd / 4;
        ie[1]->dst = t1;
        ie[1]->dst.type = DataType::ud;
        ie[1]->dst.stride = 1;
        ie[1]->src0 = x;
        ie[1]->src0.type = DataType::ud;
        ie[1]->src0.offset = 1;
        ie[1]->src0.stride = 2;

        ie[2]->op = Opcode::dnscl;
        ie[2]->simd = simd / 4;
        ie[2]->dst = t0;
        ie[2]->dst.type = y.type;
        ie[2]->dst.stride = 8;
        ie[2]->src0 = t0;
        ie[2]->src0.type = x.type;
        ie[2]->src0.stride = 2;
        ie[2]->src1 = t1;
        ie[2]->src1.type = x.type;
        ie[2]->src1.stride = 2;
        ie[2]->src2.type = DataType::ud;

        ie[3]->op = Opcode::mov;
        ie[3]->simd = simd / 2;
        ie[3]->dst = needPack ? t0 : y;
        ie[3]->dst.type = DataType::ub;
        ie[3]->dst.offset = needPack ? 0 : y.offset / 2;
        ie[3]->dst.stride = needPack ? 1 : std::max(y.vs / 2, 1);
        ie[3]->src0 = t0;
        ie[3]->src0.type = DataType::ub;
        ie[3]->src0.stride = 2;

        if (needPack) {
            ie[4]->op = Opcode::mov;
            ie[4]->dst = y;
            ie[4]->dst.type = DataType::u4;
            ie[4]->dst.stride = dstStride;
            ie[4]->src0 = t0;
            ie[4]->src0.type = DataType::u4;
            ie[4]->src0.stride = 1;
        } else {
            ie[4]->invalidate();
        }

    } else
    {
        auto ie = splitMultiple<11>(i);


        auto t0 = newTemp(DataType::hf, i.simd, 1);
        auto t1 = newTemp(DataType::hf, i.simd, 1);
        auto t0UW = t0, t1UW = t1;
        t0UW.type = t1UW.type = DataType::uw;

        auto flag = newFlag(i.simd);

        // Clamp and round.
        ie[0]->op = Opcode::mad;
        ie[0]->cmod = ConditionModifier::lt;
        ie[0]->flag = flag;
        ie[0]->dst = t1;
        ie[0]->src0 = Immediate::hf(0x8004);
        ie[0]->src1 = abs(x);
        ie[0]->src2 = Immediate::hf(0x0002);

        ie[1]->op = Opcode::sel;
        ie[1]->cmod = ConditionModifier::lt;
        ie[1]->dst = t0;
        ie[1]->src0.abs = true;
        ie[1]->src1 = Immediate::hf(0x4600);

        ie[2]->op = Opcode::mul;
        ie[2]->src0 = ie[2]->dst = t0;
        ie[2]->src1 = Immediate::hf(0x0400);

        ie[3]->op = Opcode::mad;
        ie[3]->flag = flag;
        ie[3]->dst = t0;
        ie[3]->src0 = Immediate::hf(0x0800);
        ie[3]->src1 = t1;
        ie[3]->src2 = Immediate::hf(0x6000);

        ie[4]->op = Opcode::add;
        ie[4]->src0 = ie[4]->dst = t0UW;
        ie[4]->src1 = Immediate::w(-0x0100);

        ie[5]->op = Opcode::and_;
        ie[5]->flag = flag;
        ie[5]->cmod = ConditionModifier::nz;
        ie[5]->dst = CopyOperand();
        ie[5]->dst.type = DataType::uw;
        ie[5]->src0 = t0UW;
        ie[5]->src1 = Immediate::uw(0x03FF);

        ie[6]->op = Opcode::add;
        ie[6]->flag = flag;
        ie[6]->src0 = ie[6]->dst = t0UW;
        ie[6]->src1 = Immediate::uw(0x0200);

        ie[7]->op = Opcode::shl;
        ie[7]->src0 = ie[7]->dst = t0UW;
        ie[7]->src1 = Immediate::uw(3);

        // Restore sign.
        ie[8]->op = Opcode::bfn;
        ie[8]->src0 = ie[8]->dst = t0UW;
        ie[8]->src1 = x;
        ie[8]->src1.type = DataType::uw;
        ie[8]->src2 = 0x8000;
        ie[8]->ctrl = 0xCA;

        // Pack into bytes.
        ie[9]->op = Opcode::shr;
        ie[9]->src0 = ie[9]->dst = t0UW;
        ie[9]->src1 = Immediate::uw(12);

        ie[10]->op = Opcode::mov;
        ie[10]->dst = y;
        ie[10]->dst.type = DataType::u4;
        ie[10]->src0 = t0UW;
    }
}

// Check that no types smaller than a byte are present.
void CopyPlan::checkNoSubbytes()
{
    for (auto &i: insns)
        if ((is4(i.dst.type) && i.op != Opcode::dnscl) || is4(i.src0.type) || is4(i.src1.type) || is4(i.src2.type)
            || isInt2(i.dst.type) || isInt2(i.src0.type) || isInt2(i.src1.type) || isInt2(i.src2.type)
            || isInt3(i.dst.type) || isInt3(i.src0.type) || isInt3(i.src1.type) || isInt3(i.src2.type))
            stub("Unexpected sub-byte type");
}

// Pass to legalize SIMD lengths.
// If initial = true, does not perform complete legalization,
//   only SIMD32 limits for complex conversion sequences.
void CopyPlan::legalizeSIMD(bool initial)
{
    int grf = GRF::bytes(hw);
    int simdOff = 0;
    bool splitting = false;
    bool rerun = false;

    auto forceSIMD1 = [&](const CopyInstruction &i) {
        // Workaround for packed byte mov to odd-offset dst.
        // Limiting SIMD avoids using a temporary to re-align data.
        if (hw < HW::XeHPC) return false;
        if (i.op != Opcode::mov) return false;
        if (!isB(i.dst.type) || !isB(i.src0.type)) return false;
        if (i.dst.stride != 1) return false;
        if ((i.dst.offset & 1) == 0) return false;
        return true;
    };

    if (!initial)
        checkNoSubbytes();

    // Basic rule: maximum of 2 registers per operand.
    auto opSimdMax = [&] (const CopyOperand &op, bool src2 = false) {
        if (op.kind != CopyOperand::GRF || op.stride == 0) return 64;
        int nregs = (!src2 || ((op.offset == 0) && (op.stride == 1))) ? 2 : 1;
        int remaining = (bytesToElements(nregs * grf, op.type) - (op.offset + 1)) / op.stride + 1;
        return rounddown_pow2(remaining);
    };

    auto ninsn = insns.size();
    for (size_t n = 0; n < ninsn; ) {
        auto &i = insns[n];

        int simdMax = 32;

        if (!initial) {
            simdMax = std::min({simdMax, opSimdMax(i.dst), opSimdMax(i.src0), opSimdMax(i.src1), opSimdMax(i.src2, true)});

            // Special handling for mixed mode (f16/bf16 with f32) instructions.
            bool hasF  = one_of(DataType::f,  {i.dst.type, i.src0.type, i.src1.type, i.src2.type});
            bool hasHF = one_of(DataType::hf, {i.dst.type, i.src0.type, i.src1.type, i.src2.type});
            bool hasBF = one_of(DataType::bf, {i.dst.type, i.src0.type, i.src1.type, i.src2.type});
            bool dstHF = (i.dst.type == DataType::hf);
            bool bfException = (i.op == Opcode::mov && i.dst.type == DataType::bf && i.dst.stride == 2);
            bool mathHF = (i.op == Opcode::math && i.dst.type == DataType::hf);
            bool hfByteConvert = i.op == Opcode::mov && hasHF && (isB(i.dst.type) || isB(i.src0.type));

            if ((hasF && ((hasBF && !bfException) || (hasHF && hw <= HW::XeLP) || dstHF)) || mathHF)
                simdMax = std::min(simdMax, grf >> 2);
            if (hfByteConvert)
                // Uses a stride-4 byte intermediate => 4 * (grf >> 1) = 2 * grf limit
                simdMax = std::min(simdMax, grf >> 1);
        }

        if (initial) {
            bool skip = isInt(i.dst.type) && isInt(i.src0.type) && !i.sat;
            skip |= isSubByteInt(i.dst.type) || isSubByteInt(i.src0.type)
                || isSubByteInt(i.src1.type) || isSubByteInt(i.src2.type);
            if (skip) {
                n++; continue;
            }
        }

        // Fracture instruction into legal SIMD lengths.
        const int simd1 = std::min<int>(rounddown_pow2(i.simd), simdMax);
        int simd0 = simd1;

        if (!initial && forceSIMD1(i))
            simd0 = 1;

        int minSimd0 = 1;
        for (auto *op : {&i.src0, &i.src1}) {
            if (op->kind != CopyOperand::GRF) continue;
            if (op->width)
                minSimd0 = std::max<int>(op->width, minSimd0);
        }

        if (simd0 < minSimd0 && minSimd0 < simd1) {
            rerun = true;
            simd0 = minSimd0;
        }

        if (simd0 < i.simd || splitting) {
            auto &isplit = split(i, false);
            auto length = i.range.end - i.range.start + 1;
            auto simdOrig = simdOff + i.simd;

            auto startOff = (simdOff * length) / simdOrig;
            auto endOff = ((simdOff + simd0) * length - 1) / simdOrig;
            isplit.simd = simd0;
            isplit.range.start = i.range.start + startOff;
            isplit.range.end = i.range.start + endOff;

            auto advance = [grf](CopyOperand &op, int n) {
                if (op.kind == CopyOperand::Flag)
                    op.offset += n;
                if (op.kind != CopyOperand::GRF) return;
                int ne = bytesToElements(grf, op.type);
                if (op.width) {
                    op.offset += (n / op.width) * op.vs;
                    n %= op.width;
                }
                op.offset += n * op.stride;
                int grfOffset = op.offset / ne;
                op.grf += grfOffset;
                op.offset -= grfOffset * ne;
            };

            i.simd -= simd0;
            advance(i.dst, simd0);
            advance(i.src0, simd0);
            advance(i.src1, simd0);
            advance(i.src2, simd0);
            advance(i.flag, simd0);
            splitting = (i.simd > 0);
            simdOff += simd0;
        } else {
            simdOff = 0;
            n++;    /* done with this instruction */
        }
    }

    mergeChanges();

    if (rerun)
        legalizeSIMD(initial);
}

// Check if an operand is a legal packed bfloat16 region.
inline bool legalPackedBF(HW hw, const CopyOperand &op)
{
    if (op.kind != op.GRF) return true;

    if (op.stride == 0 && op.type == DataType::f) return true;

    int align = GRF::bytes(hw) / 4;
    return (op.stride == 1 && (op.offset & (align - 1)) == 0);
}

inline bool isCommutative(Opcode op) {
    switch (op) {
        case Opcode::add:
        case Opcode::mul:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
             return true;
        default: return false;
    }
}

void CopyPlan::planEmulatedSIMD1(CopyInstruction &i)
{
    // Convert SIMD1 instruction to SIMD2.
    // Used for native hf8 <-> hf conversions where SIMD1 is not allowed.
    auto ie = splitMultiple<2>(i);
    auto temp = newTemp(i.dst.type, 2, 1);

    ie[0]->dst = temp;
    ie[0]->src0.stride = 1;
    ie[0]->simd = 2;

    ie[1]->op = Opcode::mov;
    ie[1]->src0 = temp;
    ie[1]->moveToIntegerPipe();
}

// Pass to legalize regions.
void CopyPlan::legalizeRegions()
{
    bool rerun = false;

    checkNoSubbytes();

    for (auto &i: insns) {
        auto s0t = i.src0.type;
        auto s1t = i.src1.type;
        auto s2t = i.src2.type;
        auto dt = i.dst.type;

        if (!i.dst && (hw < ngen::HW::XeHPC || i.op != Opcode::cmp)) continue;
        if (i.dst.width > 1 && i.dst.stride == 0) stub("Illegal dst stride");
        if (isFP4(dt)) continue;

        /* Check for special packed conversion cases */
        if (i.op == Opcode::mov && ((s0t == DataType::hf && isFP8(dt))
                                  || (dt == DataType::hf && isFP8(s0t)))) {
            // hf <-> bf8/hf8: src0/dst must be packed unit stride, zero offset
            if (i.src0.offset != 0 || i.src0.stride != 1) {
                repositionSrc(i, 0, 1, 0);
                rerun = true;
            } else if (i.simd == 1) {
                planEmulatedSIMD1(i);
                rerun = true;
            } else if (i.dst.offset != 0 || i.dst.stride != 1)
                repositionDst(i, 1, 0);
            continue;
        }

        if (one_of(DataType::bf, {dt, s0t, s1t}) && one_of(DataType::f, {dt, s0t, s1t, s2t})) {
            // bf/f mixed mode: dst may be packed unit stride; src must be packed unit stride.
            if (!systolicAvailable) stub("Unsupported bf16 arithmetic instruction");
            bool dstOK = (legalPackedBF(hw, i.dst) || (i.dst.stride == 2 && i.dst.offset < 2));
            bool ok = dstOK && legalPackedBF(hw, i.src0) && legalPackedBF(hw, i.src1);

            if (!ok) {
                if (!dstOK)                     repositionDst(i, 1, 0);
                if (!legalPackedBF(hw, i.src0)) repositionSrc(i, 0, 1, 0);
                if (!legalPackedBF(hw, i.src1)) repositionSrc(i, 1, 1, 0);
                rerun = true;
            }
            continue;
        }

        if (i.op == Opcode::mov) {
            if (dt == DataType::hf || s0t == DataType::hf) {
                if (dt == DataType::f || s0t == DataType::f) {
                    // hf/f mixed mode: src/dst may be packed unit stride
                    if (i.dst.stride == 1 && i.src0.stride == 1) {
                        int dstBO  = (i.dst.offset  * 4) & (GRF::bytes(hw) - 1);
                        int src0BO = (i.src0.offset * 4) & (GRF::bytes(hw) - 1);
                        if (dt == DataType::f && dstBO == src0BO)
                            continue;
                        // Spec says this should also be dstBO == src0BO, but HW
                        // doesn't behave that way. Half-GRF alignment seems fine.
                        if (dt == DataType::hf && dstBO == 0 && src0BO == 0)
                            continue;
                    }
                }
            }

            if (isB(s0t) && isB(dt) && s0t != dt && i.sat) {
                if (i.simd == 1) {
                    copyThrough(i, DataType::w, 1);
                    rerun = true;
                    continue;
                } else if (i.dst.stride == 1) {
                    restrideDst(i, 2);
                    rerun = true;
                    continue;
                }
            }
        }

        bool hfIntConvert = (dt  == DataType::hf && isInt(s0t))
                         || (s0t == DataType::hf && isInt(dt));
        hfIntConvert &= (i.op == Opcode::mov);

        /* Check destination stride against execution channels */
        int channelSize = 1;
        for (auto &op: {i.dst, i.src0, i.src1, i.src2})
            if (op.kind == op.GRF)
                channelSize = std::max(channelSize, getBytes(op.type));

        if (channelSize == 1 && i.op != Opcode::mov)
            channelSize = 2;
        if (hfIntConvert)
            channelSize = 4;

        int dstMinStride = channelSize >> getLog2Bytes(dt);
        bool doRestrideDst = (i.dst.stride < dstMinStride);

        /* Check destination offset */
        int channelOffset = (i.dst.offset * getBytes(dt)) & (channelSize - 1);
        int maxChanOff = std::max(4 / getBytes(dt), 1);
        if (getBytes(dt) == 1 && hw < HW::XeHPC)
            maxChanOff = 2;     /* special case: pre-PVC only allows .{0,1}:b */
        if (hfIntConvert)
            maxChanOff = 1;     /* special case: integer<->hf only allows .0:hf */

        bool badChanOff = (channelOffset >= maxChanOff);
        doRestrideDst |= badChanOff;

        /* For illegal dst, copy through temporary dst */
        if (doRestrideDst) {
            if (i.simd == 1)
                i.dst.stride = dstMinStride;
            else {
                restrideDst(i, dstMinStride, badChanOff);
                rerun = true;
                continue;
            }
        }

        int dstBO  = i.dst.byteOffset();
        int src0BO = i.src0.byteOffset();
        int src1BO = i.src1.byteOffset();
        int src2BO = i.src2.byteOffset();
        int dstBS  = i.dst.byteStride();
        int src0BS = i.src0.byteStride();
        int src1BS = i.src1.byteStride();
        int src2BS = i.src2.byteStride();

        bool nullDst = i.dst.isNull();
        if (nullDst) {
            i.dst.offset = src0BO / getBytes(dt);
            dstBO = src0BO, dstBS = src0BS;
        }

        /* Check for swizzling */
        bool canSwizzle = true, splitQWMov = false;
        if (hw >= HW::XeHP) {
            if (isQ(dt) || isQ(s0t) || isQ(s1t)) {
                if (i.op == Opcode::mov)
                    splitQWMov = true;
                else
                    canSwizzle = false;
            }
            if (isFP(dt))
                canSwizzle = false;
        }
        if (hw >= HW::Xe3p) {
            auto isFlat = [&] (const CopyOperand &op) {
                if (!op) return true;
                if (isBroadcast(op)) return true;
                auto bo = op.byteOffset();
                auto bs = op.byteStride();
                return (bo == dstBO) && (bs == dstBS);
            };

            if (!isFlat(i.src1)) {
                if (isCommutative(i.op) && i.src0.kind == CopyOperand::GRF && isFlat(i.src0)) {
                    std::swap(i.src0, i.src1);
                    std::swap(src0BO, src1BO);
                    std::swap(src0BS, src1BS);
                } else
                    canSwizzle = false;
            }

            if (hfIntConvert)
                canSwizzle = false;
        }

        if (!canSwizzle) {
            bool strict = isFP(dt) || (hw >= HW::Xe3p && i.op != Opcode::mov);
            int dboMask = GRF::bytes(hw) - (strict ? 1 : 4);

            auto matchesDstBO = [=](int bo) -> bool {
                return (dstBO & dboMask) == (bo & dboMask);
            };

            auto doRepositionSrc = [&](int n, DataType st) -> bool {
                int stride = dstBS >> getLog2Bytes(st);
                int offset = dstBO >> getLog2Bytes(st);
                if (stride * getBytes(st) != dstBS || offset * getBytes(st) != dstBO)
                    return false;
                repositionSrc(i, n, stride, offset);
                return true;
            };

            /* Check src0 */
            if (i.src0 && !isBroadcast(i.src0)) {
                if (!matchesDstBO(src0BO)) {
                    if (!doRepositionSrc(0, s0t)) {
                        int stride = src0BS >> getLog2Bytes(dt);
                        int offset = src0BO >> getLog2Bytes(dt);
                        if (stride * getBytes(dt) != src0BS || offset * getBytes(dt) != src0BO)
                            stub("Cannot legalize src0/dst regions");
                        repositionDst(i, stride, offset);
                    }
                    continue;
                } else if (src0BS < dstBS) {
                    restrideSrc0(i, dstBS >> getLog2Bytes(s0t));
                    rerun = true;
                }
                else if (src0BS > dstBS)
                    restrideDst(i, src0BS >> getLog2Bytes(dt));
            }

            /* Check src1 */
            if (i.src1 && !isBroadcast(i.src1) && (!matchesDstBO(src1BO) || dstBS != src1BS)) {
                if (!doRepositionSrc(1, s1t))
                    stub("Cannot legalize src1 region");
                continue;
            }
            /* Check src2 */
            if (i.src2 && !isBroadcast(i.src2) && (!matchesDstBO(src2BO) || dstBS != src2BS)) {
                if (!doRepositionSrc(2, s2t))
                    stub("Cannot legalize src2 region");
                continue;
            }
        }

        /* PVC limitations on packing multiple execution channels into a DWord */
        if (canSwizzle && hw >= HW::XeHPC && channelSize < 4 && !nullDst && i.dst.stride * getBytes(dt) < 4 && !isBroadcast(i.src0)) {
            int d0s = i.dst.stride;
            int d0o = i.dst.offset;
            int s0s = i.src0.stride;
            int s0o = i.src0.offset;
            bool strideOK = true, offsetOK = true;

            if (!isW(dt)  && !isB(dt))  stub();
            if (!isW(s0t) && !isB(s0t)) stub();

            if (i.simd == 1) {}
            else if (isW(s0t)) {
                strideOK &= (s0s <= 2);
                if (s0s == 2) {
                    if (isW(dt)) {
                        offsetOK &= (s0o / 2 == d0o % 16);
                        d0o = (s0o / 2) % 16;
                    } else {
                        offsetOK &= (s0o == d0o % 32);
                        d0o = s0o % 32;
                    }
                }
            } else {
                if (isW(dt) || d0s > 1)
                    s0s /= 2;
                if (isW(dt))
                    d0o *= 2;
                strideOK &= (s0s <= 4);
                if (s0s >= 2) {
                    offsetOK &= (d0o % (64 / s0s) == s0o / s0s);
                    auto saveD0O = d0o;
                    d0o = s0o / s0s;
                    if (isW(dt)) {
                        if (d0o & 1)
                            s0o = (saveD0O % (64 / s0s)) * s0s; /* move src rather than dst */
                        else
                            d0o /= 2;
                    }
                }
            }

            if (!strideOK) {
                int istride = 4 / getBytes(dt);
                (i.src0.byteStride() < i.dst.byteStride()) ? restrideSrc0(i, istride)
                                                           : restrideDst(i, istride);
                continue;
            }

            if (!offsetOK) {
                (s0o != i.src0.offset) ? repositionSrc(i, 0, i.src0.stride, s0o)
                                       : repositionDst(i,    i.dst.stride,  d0o);
            }
        }

        /* Split unaligned QWord moves into DWords */
        if (splitQWMov && (dstBO != src0BO || dstBS != src0BS)) {
            if (!isQ(dt) || !isQ(s0t)) stub();
            i.dst.type = i.src0.type = DataType::ud;
            i.dst.stride *= 2;
            i.dst.offset *= 2;
            i.src0.stride *= 2;
            i.src0.offset *= 2;
            if (i.dst.stride == 2) {
                /* Use 2D regioned src */
                i.simd *= 2;
                i.dst.stride = 1;
                i.src0.vs = i.src0.stride;
                i.src0.stride = 1;
                i.src0.width = 2;
            } else {
                auto &i1 = split(i);
                i1.dst.offset++;
                i1.src0.offset++;
            }
        }
    }

    for (auto &i: insns) {
        bool use2D = false;
        auto width = i.simd;
        for (auto &op: {i.src0, i.src1, i.src2}) {
            if (op.width != 0 || op.vs != 0) continue;
            if (op.kind != CopyOperand::GRF || op.stride == 0) continue;
            auto size = getBytes(op.type);
            auto remaining = (GRF::bytes(hw) - size * (1 + op.offset)) / (size * op.stride) + 1;
            if (i.simd <= remaining) continue;
            use2D = true;
            remaining |= width;
            width = remaining & -remaining;  // pow2 GCD
            width = std::min(width, 32 / (size * op.stride));
        }

        auto set2DRegion = [&](CopyOperand& op) {
            if (op.kind != CopyOperand::GRF || op.stride == 0) return;
            op.vs = width * op.stride;
            op.width = width;
            if (width == 1) op.stride = 0;
        };
        if (use2D) {
            set2DRegion(i.src0);
            set2DRegion(i.src1);
            set2DRegion(i.src2);
            rerun = true;
        }
    }

    mergeChanges();
    if (rerun)
        legalizeRegions();
}

// Pass to legalize negation use.
void CopyPlan::legalizeNegation()
{
    for (auto &i: insns) if (i.dst.neg) {
        i.src0.neg = !i.src0.neg;
        i.src1.neg = !i.src1.neg;
        i.src2.neg = !i.src2.neg;
    }
}

// Pass to legalize immediate types.
void CopyPlan::legalizeImmediateTypes()
{
    for (auto &i: insns) {
        int srcN = -1;
        for (auto *op: {&i.src0, &i.src1, &i.src2}) {
            srcN++;
            if (op->kind != CopyOperand::Immediate)
                continue;
            if (one_of(op->type, {DataType::ub, DataType::u4}))
                op->type = DataType::uw;
            else if (one_of(op->type, {DataType::b, DataType::s4}))
                op->type = DataType::w;
            else if (hw == ngen::HW::Xe3p && i.op != Opcode::mov && op->type == DataType::f && i.dst.type == DataType::bf)
                legalizeBfImmediate(i);
            else if (one_of(op->type, {DataType::v, DataType::uv})) {
                // Destination must be 128 bit-aligned for vector immediates.
                if ((i.dst.offset * getBits(i.dst.type)) % 128 != 0)
                    repositionSrc(i, srcN, i.dst.stride, 0);
            }
        }
    }
    mergeChanges();
    legalizeRegions();
}

// Pass to sort instructions by phase and dst.
void CopyPlan::sort(SortType type)
{
    auto sortOrder = [type](const CopyInstruction &i) {
        switch (type) {
            case SortType::PhaseOnly:
                return std::make_tuple(i.phase, 0, 0);
            case SortType::SourceOrder:
                return std::make_tuple(i.phase, int(i.range.start), int(i.range.end));
            case SortType::Register:
            default:
                auto &op = i.dst.temp ? i.src0 : i.dst;
                return std::make_tuple(i.phase, int(op.grf), op.byteOffset());
        };
    };

    std::stable_sort(insns.begin(), insns.end(), [=](const CopyInstruction &i1, const CopyInstruction &i2) {
        return sortOrder(i1) < sortOrder(i2);
    });
}

// Optimization pass: Convert long chains of small-SIMD equal-strided movs
// to a few larger-SIMD movs.
// Example input:
//    mov (2) r3.0:ub<1>      r3.0:ub<2>
//    mov (2) r3.4:ub<1>      r3.4:ub<2>
//    mov (2) r3.8:ub<1>      r3.8:ub<2>
//    mov (2) r3.12:ub<1>     r3.12:ub<2>
//    mov (2) r3.16:ub<1>     r3.16:ub<2>
//    mov (2) r3.20:ub<1>     r3.20:ub<2>
//    mov (2) r3.24:ub<1>     r3.24:ub<2>
//    mov (2) r3.28:ub<1>     r3.28:ub<2>
// Output:
//    mov (8) r3.0:ub<4>      r3.0:ub<4>  /* removed */
//    mov (8) r3.1:ub<4>      r3.2:ub<4>
//
void CopyPlan::optimizeTranspose()
{
    const auto grf = ngen::GRF::bytes(hw);
    auto transposable = [grf](const CopyOperand &o1, const CopyOperand &o2, int offset) {
        if (o1.kind != CopyOperand::GRF || o2.kind != CopyOperand::GRF) return false;
        if (o1.type != o2.type || o1.stride != o2.stride) return false;
        if (o1.temp != o2.temp) return false;
        if (o1.temp && o1.value != o2.value) return false;
        if (o1.neg != o2.neg) return false;
        if (o1.abs != o2.abs) return false;

        auto grf_diff = o2.grf - o1.grf;
        auto dist = bytesToElements(grf, o1.type) * grf_diff + (o2.offset - o1.offset);
        return dist == offset;
    };

    auto ninsn = insns.size();
    for (size_t n1 = 0; n1 < ninsn; n1++) {
        auto &i1 = insns[n1];
        if (i1.op != Opcode::mov || i1.simd == 0) continue;
        const int8_t dstride = bytesToElements(4, i1.dst.type);
        const int8_t sstride = bytesToElements(4, i1.src0.type);

        int simd = 1;
        size_t n2 = n1 + 1;
        auto end = i1.range.end + 1;
        for (; n2 < ninsn; n2++, simd++) {
            auto &i2 = insns[n2];
            if (i1.phase != i2.phase) break;
            if (i2.op != Opcode::mov) break;
            if (i1.simd != i2.simd) break;
            if (i1.sat != i2.sat) break;
            if (!transposable(i1.src0, i2.src0, dstride * simd)) break;
            if (!transposable(i1.dst, i2.dst, sstride * simd)) break;
            if (i2.range.start != end) break;  // ensure joins are valid
            end = i2.range.end + 1;
        }

        size_t n = n1 + 1;
        n1 = n2 - 1;

        if (i1.simd >= simd) continue;

        for (; n < n2; n++)
            join(i1, insns[n]);

        auto simd0 = i1.simd;
        i1.simd = simd;
        auto soffset = i1.src0.stride;
        auto doffset = i1.dst.stride;
        i1.src0.stride = sstride;
        i1.dst.stride = dstride;
        for (int i = 1; i < simd0; ++i) {
            auto &i2 = split(i1, false);
            i2.src0.offset += i * soffset;
            i2.dst.offset += i * doffset;
        }
    }

    mergeChanges();
}

// Optimization pass: zip together interleaved operations.
// Requires a sorted plan.
//
// Example input:
//    mov (8)  r0.0<4>:uw   r10.0<2>:uw
//    mov (8)  r0.2<4>:uw   r10.1<2>:uw
// Output:
//    mov (16) r0.0<2>:uw   r10.0<1>:uw
//
// If zip2DSrc0 is true, then look for opportunities to use 2D regions
//   for src0:
//
// Example input:
//    mov (8)  r0.0<2>:uw   r10.0<4>:ub
//    mov (8)  r0.1<2>:uw   r10.1<4>:ub
// Output:
//    mov (16) r0.0<1>:uw   r10.0<4;2,1>:ub
//
void CopyPlan::optimizeZip(bool zip2DSrc0)
{
    bool didZip2D = false;
    bool didStridedVecZip = false;

    auto ninsn = insns.size();
    for (size_t n1 = 0; n1 < ninsn; n1++) {
        for (size_t n2 = n1 + 1; n2 < ninsn; n2++) {
            auto &i1 = insns[n1];
            auto &i2 = insns[n2];

            if (i1.op != i2.op || i1.phase != i2.phase || i1.dst.grf != i2.dst.grf || i1.flag) break;
            if (i1.simd != i2.simd) continue;

            auto zippable = [&](const CopyOperand &o1, const CopyOperand &o2, bool zip2D = false, bool zipImm = false) {
                if (o1.kind != o2.kind) return false;
                if (o1.kind == CopyOperand::Immediate) return (o1.value == o2.value || zipImm);
                if (o1.kind != CopyOperand::GRF) return true;
                if (o1.type != o2.type || o1.stride != o2.stride || o1.grf != o2.grf) return false;
                if (o1.temp != o2.temp) return false;
                if (o1.temp && o1.value != o2.value) return false;
                if (o1.vs != o2.vs || o1.width != o2.width) return false;
                if (o1.width && (!zip2D || o1.stride != 2 * (o2.offset - o1.offset))) return false;
                if (o1.neg != o2.neg) return false;
                if (o1.abs != o2.abs) return false;
                if (!is_zero_or_pow2(o2.offset - o1.offset)) return false;
                bool can1D = (o1.width == 0) && ((o1.stride & 1) == 0)
                          && (o1.offset + (o1.stride >> 1) == o2.offset);
                if (!can1D) {
                    unsigned od = (o2.offset - o1.offset);
                    if (od * getBytes(o1.type) >= 4) return false;
                }
                return can1D != zip2D;
            };

            bool zip = zippable(i1.dst, i2.dst) && zippable(i1.src0, i2.src0, zip2DSrc0);
            if (i1.src1) zip = zip && zippable(i1.src1, i2.src1, false, zip2DSrc0);
            if (i1.src2) zip = zip && zippable(i1.src2, i2.src2);

            CopyOperand zippedSrc1;
            if (zip && i1.src1.kind == CopyOperand::Immediate && i1.src1.value != i2.src1.value) {
                const auto stride = i1.dst.stride >> 1;
                const auto ok = ((stride * getBits(i1.dst.type)) % 16 == 0);
                zippedSrc1 = zipImmediates(i1.src1, i2.src1, ok ? stride : 0);
                zip = zip && zippedSrc1;
            }

            if (zip) {
                if (auto &i = join(i1, i2)) {
                    i.simd *= 2;
                    i.dst.stride /= 2;
                    i.src1.stride /= 2;
                    i.src2.stride /= 2;
                    if (!zip2DSrc0) {
                        i.src0.stride /= 2;
                        std::swap(i1, i2);      /* move joined entry to end for further processing */
                    } else {
                        if (!i.src0.width) {
                            // transform 1D to equivalent 2D regioning
                            // <N> -> <N;1,0>
                            i.src0.width = 1;
                            i.src0.vs = i.src0.stride;
                            i.src0.stride = 0;
                        }
                        i.src0.width *= 2;
                        i.src0.stride += i2.src0.offset - i1.src0.offset;
                        didZip2D = true;
                    }
                    if (zippedSrc1) {
                        i.src1 = zippedSrc1;
                        didStridedVecZip = (i.dst.stride > 1) && one_of(i.src1.type, {DataType::v, DataType::uv});
                    }
                    break;
                }
            }
        }
    }

    mergeChanges();

    if (didStridedVecZip)
        optimizeZip(zip2DSrc0);  /* may be able to zip more vector immediates */
    if (didZip2D)
        legalizeSIMD();     /* 2D zipping comes late in the pipeline */
}

// Zip small immediates, converting small values to a vector immediate, and
// other cases to <0;2,1>-regioned resource access.
CopyOperand CopyPlan::zipImmediates(const CopyOperand &o1, const CopyOperand &o2, uint8_t stride)
{
    if (o1.type != o2.type)
        return CopyOperand{};

    if (o1.type == DataType::uv || o1.type == DataType::v) {
        if (stride <= 0 || stride >= 8)
            return CopyOperand{};
        // alternate vector entries for stride = 1, alternate consecutive
        // pairs of vector entries for stride = 2, etc.
        uint32_t mask = (1 << (4 * stride)) - 1;
        for (auto bits = 8 * stride; bits < 32; bits <<= 1)
            mask |= mask << bits;
        auto value = ((uint32_t)o1.value & mask) | ((uint32_t)o2.value & ~mask);
        return o1.type == DataType::uv ? Immediate::uv(value) : Immediate::v(value);
    }

    if (!isW(o1.type))
        return CopyOperand{};

    auto v1 = (uint16_t)o1.value;
    auto v2 = (uint16_t)o2.value;

    if (stride > 0 && stride < 8) {
        auto pattern = (v1 & 0xF) | ((v2 & 0xF) << (4 * stride));
        for (uint8_t i = 1; i < stride; i <<= 1)
            pattern |= pattern << (4 * i);
        auto bits = 8 * stride;
        for (; bits < 32; bits <<= 1)
            pattern |= pattern << bits;
        if (v1 <= 0xF && v2 <= 0xF)
            return ngen::Immediate::uv(pattern);
        else if (o1.type == DataType::w && (uint16_t)(v1 + 8) <= 0xF && (uint16_t)(v2 + 8) <= 0xF)
            return ngen::Immediate::v(pattern);
    }

    if (hw >= HW::Xe3p)
        return CopyOperand{};

    auto rkind = CopyResource::makeConstant32((v2 << 16) | v1);
    auto op = getResource(rkind);
    op.vs = 0;
    op.width = 2;
    op.stride = 1;
    op.type = o1.type;
    return op;
}

// Make an integer operand twice as wide.
static void widen(CopyOperand &op, bool zipping = false)
{
    switch (op.kind) {
        case CopyOperand::GRF:
            op.offset /= 2;
            if (zipping)
                op.stride /= 2;
            break;
        case CopyOperand::Immediate:
            op.value |= op.value << getBits(op.type);
            break;
        case CopyOperand::Flag: stub();
        case CopyOperand::Null: return;
    }

    if (isInt2(op.type)) op.type = DataType::u4;
    else if (isInt4(op.type)) op.type = DataType::ub;
    else if (isB(op.type)) op.type = DataType::uw;
    else if (isW(op.type)) op.type = DataType::ud;
    else stub();
    op.range = op.type;
}

// Check if an integer operand can be widened.
static bool widenable(const CopyOperand &op, bool zipping = false)
{
    if (op.kind == CopyOperand::Flag) return false;
    if (op.kind != CopyOperand::GRF) return true;
    if (isFP(op.type) || getBytes(op.type) >= 4) return false;
    if (zipping && (op.stride & 1)) return false;
    if (!zipping && (op.stride != 1)) return false;
    if (op.offset & 1) return false;
    return true;
}

// Optimization pass: join adjacent integer operations into larger ones.
// Requires a sorted plan.
//
// Example input:
//    or (16)   r0.0<4>:uw    r10.0<4>:uw   0x1111:uw
//    or (16)   r0.1<4>:uw    r10.1<4>:uw   0x2222:uw
// Output:
//    or (16)   r0.0<2>:ud    r10.0<2>:ud   0x22221111:ud
//
void CopyPlan::optimizeZipAdjacent()
{
    bool changed = false;

    auto ninsn = insns.size();
    for (size_t n2 = 1; n2 < ninsn; n2++) {
        auto &i1 = insns[n2 - 1];
        auto &i2 = insns[n2];

        if (i1.isInvalid() || i1.op != i2.op || i1.simd != i2.simd || i1.phase != i2.phase || i1.dst.grf != i2.dst.grf) continue;
        if (i1.flag || i2.flag || i1.sat || i2.sat) continue;
        if (!isBitwise(i1.op)) continue;

        auto zippable = [](const CopyOperand &o1, const CopyOperand &o2) {
            if (o1.kind != o2.kind) return false;
            if (o1.kind == CopyOperand::Immediate) return (o1.value == o2.value);
            if (o1.kind != CopyOperand::GRF) return true;
            if (o1.type != o2.type || o1.stride != o2.stride || o1.grf != o2.grf) return false;
            if (o1.temp != o2.temp) return false;
            if (o1.temp && o1.value != o2.value) return false;
            if (!widenable(o1, true)) return false;
            if (o1.neg != o2.neg) return false;
            if (o1.abs != o2.abs) return false;
            return (o1.offset + 1 == o2.offset);
        };

        bool zip = zippable(i1.dst, i2.dst) && zippable(i1.src0, i2.src0)
                && asSigned(i1.dst.type) == asSigned(i1.src0.type);
        if (i1.src1)
            zip = zip && zippable(i1.src1, i2.src1) && (asSigned(i1.src1.type) == asSigned(i1.dst.type));
        if (i1.src2)
            zip = zip && zippable(i1.src2, i2.src2) && (asSigned(i1.src2.type) == asSigned(i1.dst.type));

        if (zip) {
            if (auto &i = join(i1, i2)) {
                widen(i.dst, true);
                if (i.src0) widen(i.src0, true);
                if (i.src1) widen(i.src1, true);
                if (i.src2) widen(i.src2, true);
                changed = true;
                break;
            }
        }
    }

    if (changed) {
        mergeChanges();
        optimizeZipAdjacent();
    }
}

// Optimization pass: use larger integer types for contiguous operands if possible.
//
// Example input:
//    mov (16)  r0.0<1>:ub   r1.0<1>:ub
// Output:
//    mov (4)   r0.0<1>:ud   r1.0<1>:ud
//
void CopyPlan::optimizeWidenIntegers()
{
    for (auto &i: insns) {
        if (!isBitwise(i.op) || i.flag || i.sat || i.src2) continue;

        while (true) {
            bool doWiden = widenable(i.dst) && widenable(i.src0)
                        && asSigned(i.dst.type) == asSigned(i.src0.type)
                        && !i.src0.neg && !i.src0.abs && i.simd % 2 == 0;

            for (auto op: {&i.src1, &i.src2}) if (*op) {
                doWiden = doWiden && widenable(*op)
                        && (asSigned(op->type) == asSigned(i.dst.type));
            }

            if (!doWiden) break;

            i.simd /= 2;
            widen(i.dst);
            widen(i.src0);
            widen(i.src1);
        }
    }
}

// Optimization pass: concatenate instructions.
//   The instructions may overlap partially or completely.
//   On the initial pass (initial = true), there is no limit on the SIMD width.
//   Otherwise, do not concatenate beyond SIMD32, or two registers.
//
// Example input:
//    mov (8)  r0.0<1>:uw  r10.0<1>:uw
//    mov (8)  r0.8<1>:uw  r10.8<1>:uw
// Output:
//    mov (16) r0.0<1>:uw  r10.0<1>:uw
//
void CopyPlan::optimizeConcatenate(bool initial)
{
    const auto grf = ngen::GRF::bytes(hw);
    auto ninsn = insns.size();
    for (size_t n1 = 0; n1 < ninsn; n1++) {
        for (size_t n2 = n1 + 1; n2 < ninsn; n2++) {
            auto &i1 = insns[n1];
            auto &i2 = insns[n2];

            if (i1.op != i2.op || i1.phase != i2.phase || i1.flag || i2.flag) break;

            int simd1 = i1.simd; // arbitrary
            auto joinable = [&](const CopyOperand &o1, const CopyOperand &o2, bool *outTooFar = nullptr) {
                if (o1.kind != o2.kind) return false;
                if (o1.kind == CopyOperand::Null) return true;
                if (o1.type != o2.type || o1.stride != o2.stride) return false;
                if (o1.kind == CopyOperand::Immediate) return (o1.value == o2.value);
                if (o1.temp != o2.temp) return false;
                if (o1.temp && (o1.value != o2.value)) return false;
                if (o1.neg != o2.neg) return false;
                if (o1.abs != o2.abs) return false;
                auto lead = bytesToElements((o2.grf - o1.grf) * grf, o1.type) + (int)o2.offset - (int)o1.offset;
                auto breadth = o1.stride * i1.simd;
                auto elems = o1.stride ? lead / o1.stride : 0;
                if (outTooFar) {
                    *outTooFar = (lead > breadth);
                    simd1 = elems;
                }
                return (elems == simd1) && ((lead & (o1.stride - 1)) == 0) && (lead >= 0) && (lead <= breadth);
            };

            bool tooFar = false;
            bool doJoin = joinable(i1.dst, i2.dst, &tooFar) && joinable(i1.src0, i2.src0)
                       && joinable(i1.src1, i2.src1) && joinable(i1.src2, i2.src2);

            int simd = std::max(simd1 + i2.simd, i1.simd);
            if (!initial) {
                doJoin &= (simd <= 32);
                doJoin &= (simd * getBytes(i1.dst.type) <= 2 * GRF::bytes(hw));
            }

            if (tooFar) break;

            if (doJoin) {
                if (simd == i1.simd)
                    // i1 completely overlaps i2; remove it.
                    i2.invalidate();
                else if (auto &i = join(i1, i2))
                    i.simd = simd;
            }
        }
    }

    mergeChanges();
}

// Optimization pass: enable write combining for byte writes (XeHPC+).
// Requires a sorted plan.
//
// Example input:
//    mov (8)  r0.0<4>:ub  r10.0<1>:ub
//    mov (8)  r0.1<4>:ub  r20.0<1>:ub
//    mov (8)  r0.2<4>:ub  r30.0<1>:ub
//    mov (8)  r0.3<4>:ub  r40.0<1>:ub
// Output:
//    mov (8)  r0.0<4>:ub  r10.0<1>:ub  {Atomic}
//    mov (8)  r0.1<4>:ub  r20.0<1>:ub  {Atomic}
//    mov (8)  r0.2<4>:ub  r30.0<1>:ub  {Atomic}
//    mov (8)  r0.3<4>:ub  r40.0<1>:ub
//
void CopyPlan::optimizeWriteCombine()
{
    auto ninsn = insns.size();

    if (hw < HW::XeHPC) return;

    for (size_t n = 0; n + 1 < ninsn; ) {
        auto &i0 = insns[n];

        auto canWC = [](HW hw, CopyInstruction &i) {
            auto st = i.src0.type;
            if (i.op != Opcode::mov || i.flag) return false;
            if (!isB(i.dst.type)) return false;
            if (!(isB(st) || isW(st) || isD(st) || st == DataType::f)) return false;
            if (multiGRF(hw, i, i.dst)) return false;
            return !mayOverlap(hw, i, i.dst, i.src0);
        };

        if (!canWC(hw, i0)) {
            n++; continue;
        }

        auto range = i0.range;
        size_t n1;
        for (n1 = n + 1; n1 < ninsn; n1++) {
            auto &i1 = insns[n1];
            if (!canWC(hw, i1)) break;
            if (i1.dst.grf != i0.dst.grf) break;
            if (i1.dst.temp ^ i0.dst.temp) break;
            if (i1.dst.temp && i0.dst.temp && i1.dst.value != i0.dst.value) break;
            if (i1.dst.offset + n != i0.dst.offset + n1) break;
            if (i0.dst.offset / 4 != i1.dst.offset / 4) break;
            range |= i1.range;
        }

        auto length = int(rounddown_pow2(n1 - n));
        for (n1 = n; n1 + 1 < n + length; n1++)
            insns[n1].atomic = true;
        for (n1 = n; n1 < n + length; n1++)
            insns[n1].range = range;

        n += length;
    }
}

// Optimization pass: spread writes to bytes in the same word (XeHPC+).
//   This reduces false WAW dependencies between the instructions.
// Requires a sorted plan.
//
// Example input:
//    shr (8)  r0.0<4>:ub  r10.0<1>:ub  4:uw
//    shr (8)  r0.1<4>:ub  r20.0<1>:ub  4:uw    // would be {@1}
//    shr (8)  r0.2<4>:ub  r30.0<1>:ub  4:uw
//    shr (8)  r0.3<4>:ub  r40.0<1>:ub  4:uw    // would be {@1}
// Output:
//    shr (8)  r0.0<4>:ub  r10.0<1>:ub  4:uw
//    shr (8)  r0.2<4>:ub  r30.0<1>:ub  4:uw
//  ...
//    shr (8)  r0.1<4>:ub  r20.0<1>:ub  4:uw
//    shr (8)  r0.3<4>:ub  r40.0<1>:ub  4:uw
//
void CopyPlan::optimizeWriteSpread()
{
    if (hw < HW::XeHPC) return;

    auto ninsn = insns.size();
    for (size_t n = 1; n < ninsn; n++) {
        auto &iprev = insns[n - 1];
        auto &i = insns[n];

        if (isB(i.dst.type) && i.dst.stride > 1 && i.dst.offset & 1 && !iprev.atomic) {
            (void) split(i, false);
            i.invalidate();
        }
    }

    mergeChanges();
}

// Optimization pass: reduce excess source bits in integer downconversions.
//
// Example input:
//    mov (8)  r0.0<1>:ub  r10.0<1>:ud
// Output:
//    mov (8)  r0.0<1>:ub  r10.0<4>:ub
//
void CopyPlan::optimizeIntegerDownconvert()
{
    for (auto &i: insns) {
        if (i.op != Opcode::mov || i.sat) continue;
        if (!isInt(i.dst.type) || !isInt(i.src0.type)) continue;
        if (isSubByteInt(i.dst.type) || isSubByteInt(i.src0.type)) continue;

        int expand = getBytes(i.src0.type) / getBytes(i.dst.type);
        if (expand > 1) {
            i.src0.type = i.dst.type;
            i.src0.offset *= expand;
            i.src0.stride *= expand;
        }
    }
}

// Optimization/cleanup pass: remove unneeded saturation modifiers.
void CopyPlan::optimizeSaturate()
{
    for (auto &i: insns) {
        if (i.op != Opcode::mov) continue;
        i.sat &= !isSubsetOf(i.src0.type, i.dst.type);
    }
}

// Optimization pass: move floating point moves to integer pipe where possible.
void CopyPlan::optimizeMoveToIntPipe()
{
    for (auto &i: insns) {
        if (i.op != Opcode::mov) continue;
        if (i.src0.neg || i.dst.neg) continue;
        if (i.src0.type != i.dst.type) continue;
        i.moveToIntegerPipe();
    }
}

struct AllocationManager {
    ngen::HW hw;
    const CopyPlan::GRFAllocator &grfAllocator;
    const CopyPlan::FlagAllocator &flagAllocator;

    AllocationManager(ngen::HW hw, const CopyPlan::GRFAllocator &grfAllocator, const CopyPlan::FlagAllocator &flagAllocator)
        : hw(hw), grfAllocator(grfAllocator), flagAllocator(flagAllocator) {}
    ~AllocationManager() {
        for (auto &alloc : grfAllocations) grfAllocator(0, alloc.range);
        for (auto &alloc : flagAllocations) flagAllocator(0, alloc.flag);
        grfAllocations.clear();
        flagAllocations.clear();
    }

    void reserve(size_t size) {
        grfAllocations.reserve(size);
        flagAllocations.reserve(size);
    }

    bool allocate(CopyTemporary &temp) {
        if (temp.flag)
            return allocateFlag(temp);
        return allocateRange(temp);
    }

    void release(int end) {
        release(end, grfAllocations);
        release(end, flagAllocations);
    }

private:
    template <typename AllocationType>
    void release(int end, std::vector<AllocationType> &allocs) {
        for (size_t i = 0; i < allocs.size();) {
            auto &alloc = allocs[i];
            if (alloc.end < end) {
                dealloc(alloc);
                std::swap(allocs[i], allocs[allocs.size() - 1]);
                allocs.pop_back();
            } else
                ++i;
        }
    }

protected:
    struct GRFAllocation {
        GRFRange range;
        int end;
    };

    struct FlagAllocation {
        FlagRegister flag;
        int end;
    };

    void dealloc(GRFAllocation &alloc) { grfAllocator(0, alloc.range); }
    void dealloc(FlagAllocation &alloc) { flagAllocator(0, alloc.flag); }

    bool allocateFlag(CopyTemporary &temp) {
        FlagRegister flag;
        flagAllocator(temp.bytes, flag);
        if (!flag.isValid()) return false;
        temp.assignment = flag.index();
        flagAllocations.push_back({flag, temp.range.end});
        return true;
    }

    bool allocateRange(CopyTemporary &temp) {
        GRFRange range;
        auto grfs = div_up(temp.bytes, GRF::bytes(hw));
        grfAllocator(grfs, range);
        if (!range.isValid()) return false;
        temp.assignment = range.getBase();
        grfAllocations.push_back({range, temp.range.end});
        return true;
    }

    std::vector<GRFAllocation> grfAllocations;
    std::vector<FlagAllocation> flagAllocations;
};

// Materialize temporary GRF and flag registers in a copy plan, replacing
//   them by physical GRF and flag registers.
// Instructions will be reordered as needed if there are not enough temporary
//   resources to give each temporary a distinct physical resource.
void CopyPlan::materializeTemps(const GRFAllocator &grfAllocator, const FlagAllocator &flagAllocator)
{
    std::vector<CopyInstruction> sortedInsns;
    AllocationManager manager(hw, grfAllocator, flagAllocator);
    uint16_t minPhaseTemp = 0xFFFF, maxPhaseTemp = 0x0;

    sortedInsns.reserve(insns.size());
    manager.reserve(temps.size());

    /* Round up instruction usage by each temporary */
    for (const auto &i: insns) {
        bool haveTemp = false;
        for (auto o: {&i.dst, &i.src0, &i.src1, &i.src2, &i.flag}) if (*o && o->temp) {
            temps[o->value].usedBy(i);
            haveTemp = true;
        }
        if (haveTemp) {
            minPhaseTemp = std::min(minPhaseTemp, i.phase);
            maxPhaseTemp = std::max(maxPhaseTemp, i.phase);
        }
    }

    // No temporaries used
    if (minPhaseTemp > maxPhaseTemp) return;

    /* Check which instruction groups must be issued together */
    auto groupInstructions = [&](std::vector<bool> &joined, CopyRange &range) {
        range = {};
        joined.assign(joined.size(), false);

        for (auto &i: insns) {
            if (i.phase < minPhaseTemp || i.phase > maxPhaseTemp)
                continue;
            for (auto j = i.range.start; j < i.range.end; j++)
                joined[j] = true;
            range |= i.range;
        }
    };

    CopyRange range0;
    for (auto &i : insns)
        range0 |= i.range;

    CopyRange range;
    std::vector<bool> joined(range0.end + 1);
    groupInstructions(joined, range);

    /* Sort instructions and temporaries by element index */

    auto cmp = [&](int i, int j) {
        const auto &ri = temps[i].range;
        const auto &rj = temps[j].range;
        if (ri != rj) return ri < rj;
        return i < j;
    };

    std::vector<int> rangeOrder(temps.size());
    std::iota(rangeOrder.begin(), rangeOrder.end(), 0);
    std::sort(rangeOrder.begin(), rangeOrder.end(), cmp);

    auto emit = [&](const CopyRange &range, uint16_t minPhase, uint16_t maxPhase) {
        bool emitted = false;
        for (const auto &i: insns) {
            if (i.phase < minPhase || i.phase > maxPhase)
                continue;
            if (i.range.start < range.start || i.range.end > range.end)
                continue;
            sortedInsns.push_back(i);
            emitted = true;
        }
        return emitted;
    };

    /* Issue instructions up to first temporary */
    if (minPhaseTemp > 0x0000)
        emit(range0, 0x0000, minPhaseTemp - 1);

    auto order = rangeOrder.begin();
    const auto end = rangeOrder.end();
    while (order != end) {
        for (; order != end; ++order) {
            auto &temp = temps[*order];
            // Don't allocate unused temporaries.
            if (!temp.range) continue;
            if (!manager.allocate(temp)) {
                range.end = temp.range.start - 1; break;
            }
        }

        /* Back off to the nearest instruction group boundary */
        while (range.end >= range.start && joined[range.end])
            range.end--;
        if (range.end < range.start) {
            bool emitted = false;
            if (order != end) {
                auto &temp = temps[*order];
                if (temp.phaseMin > minPhaseTemp) {
                    emitted = emit(range0, minPhaseTemp, temp.phaseMin - 1);
                    minPhaseTemp = temp.phaseMin;
                }
            }
            if (!emitted)
                throw out_of_registers_exception();
            groupInstructions(joined, range);
        }

        /* Issue instructions for this batch of instruction groups */
        emit(range, minPhaseTemp, maxPhaseTemp);

        manager.release(range.end + 1);
        range.start = range.end + 1;
        range.end = range0.end;
    }

    /* Issue any remaining instructions */
    if (range.start <= range0.end)
        emit(range, minPhaseTemp, maxPhaseTemp);
    if (maxPhaseTemp < 0xFFFF)
        emit(range0, maxPhaseTemp + 1, 0xFFFF);

    std::swap(insns, sortedInsns);

    /* Update operands with assignments */
    for (auto &i: insns) {
        for (auto o: {&i.dst, &i.src0, &i.src1, &i.src2, &i.flag}) {
            if (o->temp) {
                o->temp = false;
                o->grf += temps[o->value].assignment;
            }
        }
    }

    /* Update resources with assignments */
    for (auto &r: resources) if (r.src.temp) {
        r.src.temp = false;
        r.src.grf += temps[r.src.value].assignment;
    }

    temps.clear();
}

CopyResource::Kind CopyResource::makeConstant32(uint32_t c)
{
    return static_cast<Kind>(constantBase | c);
}

int CopyResource::getData(std::array<uint8_t, 64> &data) const
{
    if (kind & constantBase) {
        std::memcpy(&data[0], &kind, sizeof(uint32_t));
        return sizeof(uint32_t);
    }

    DataType st, dt;
    if (decodeShflLUT(st, dt)) {
#define LUT16(ST, DT, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, VA, VB, VC, VD, VE, VF)  \
        if (st == DataType::ST && dt == DataType::DT) {                                \
            static const uint16_t table[32] = {V0, V0, V1, V1, V2, V2, V3, V3,         \
                                            V4, V4, V5, V5, V6, V6, V7, V7,            \
                                            V8, V8, V9, V9, VA, VA, VB, VB,            \
                                            VC, VC, VD, VD, VE, VE, VF, VF};           \
            std::memcpy(&data[0], table, sizeof(table));                               \
            return sizeof(table);                                                      \
        }

        LUT16(e2m1, hf, 0x0, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600, 0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600)
        LUT16(e2m1, bf, 0x0, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0, 0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0)

        LUT16(u4, hf, 0x0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80)
        LUT16(s4, hf, 0x0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0xc800, 0xc700, 0xc600, 0xc500, 0xc400, 0xc200, 0xc000, 0xbc00)
        LUT16(u4, bf, 0x0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170)
        LUT16(s4, bf, 0x0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0xc100, 0xc0e0, 0xc0c0, 0xc0a0, 0xc080, 0xc040, 0xc000, 0xbf80)
#undef LUT16
    }

    return 0;
}

CopyResource::Kind CopyResource::makeShflLUT(DataType from, DataType to)
{
    return static_cast<Kind>(shflLUTBase | static_cast<uint32_t>(from) | (static_cast<uint32_t>(to) << 8));
}

bool CopyResource::decodeShflLUT(DataType &from, DataType &to) const
{
    if (kind & shflLUTBase) {
        from = static_cast<DataType>(kind);
        to   = static_cast<DataType>(kind >> 8);
        return true;
    }
    return false;
}

#if GEMMSTONE_ENABLE_COPY_PLAN_DUMP
int CopyPlan::cycleCount() const
{
    int count = 0;
    for (const auto &i: insns)
        count += (multiGRF(hw, i, i.dst) || multiGRF(hw, i, i.src0) || multiGRF(hw, i, i.src1) || multiGRF(hw, i, i.src2)) ? 2 : 1;
    return count;
}

void CopyPlan::dump(std::ostream &os, int n, bool sortInfo) const
{
    for (int i = 0; i < std::min(n, int(insns.size())); ++i) {
        insns[i].dump(os, *this, sortInfo);
        os << std::endl;
    }
}

void CopyInstruction::dump(std::ostream &os, const CopyPlan &plan, bool sortInfo) const
{
    if (flag && cmod == ConditionModifier::none) {
        os << '(';
        flag.dump(os);
        os << ")\t";
    }

    if (op == Opcode::shfl)
        os << "shfl.idx4";
    else if (op == Opcode::dnscl)
        os << "dnscl";
    else
        os << getMnemonic(op, HW::Gen9);
    switch (op) {
        case Opcode::bfn:  os << ".(" << BFN::nodes[ctrl].str() << ')';  break;
        case Opcode::math: os << '.' << static_cast<MathFunction>(ctrl); break;
        default: break;
    }

    os << " (" << simd << ")\t";
    if (sat) os << "(sat) ";
    if (cmod != ConditionModifier::none) {
        os << '(' << cmod << ')';
        flag.dump(os);
        os << ' ';
    }
    dst.dump(os);
    os << '\t';
    src0.dump(os);
    if (src1) {
        os << '\t';
        src1.dump(os);
        if (src2 || op == Opcode::dnscl) {
            os << '\t';
            src2.dump(os);
        }
    }
    if (atomic)
        os << "\t{Atomic}";

    if (sortInfo)
        os << "\t\t(phase = " << phase << ", range = " << range.str() << ')';
}

void CopyOperand::dump(std::ostream &os) const
{
    auto outType = [&](DataType dt) {
        if (dt == Type::ngen_nf4())       os << "nf4";
        else if (dt == Type::ngen_e8m0()) os << "e8m0";
        else if (dt == ngen_b16_l4x())    os << "b16_l4x";
        else if (dt == ngen_b16_h4x())    os << "b16_h4x";
        else if (dt == ngen_b16_l2x())    os << "b16_l2x";
        else if (dt == ngen_b16_h2x())    os << "b16_h2x";
        else if (dt == ngen_b16())        os << "b16";
        else                              os << dt;
    };

    if (neg) os << '-';
    if (abs) os << "(abs)";
    switch (kind) {
        case Null: os << "null:" << type; break;
        case GRF:
            if (temp) {
                os << 't' << value;
                if (grf) os << '+' << grf;
            } else
                os << 'r' << grf;
            os << '.' << int(offset) << ':';
            outType(type);
            if (range != DataType::invalid && range != type) {
                os << '[';
                outType(range);
                os << ']';
            }
            os << '<';
            if (vs || width)
                os << int(vs) << ';' << int(width) << ',';
            os << int(stride) << '>';
            break;
        case Flag:
            if (temp)
                os << 't' << value;
            else
                os << 'f' << (grf >> 1) << '.' << (grf & 1);
            if (offset)
                os << '+' << int(offset);
            break;
        case Immediate:
            LabelManager man;
            ngenImmediate().outputText(os, PrintDetail::full, man);
            break;
    }
    if (stride > 1 && overwriteStride) os << "!!";
    else if (overwrite)                os << '!';
}
#endif /* GEMMSTONE_ENABLE_COPY_PLAN_DUMP */

GEMMSTONE_NAMESPACE_END
