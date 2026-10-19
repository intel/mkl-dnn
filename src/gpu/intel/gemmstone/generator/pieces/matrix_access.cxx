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
#include "layout_utils.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


enum class AccessClass {Read, Write, Atomic};

static DataSizeLSC getDataSizeLSC(int ebytes, bool pad32);
static DataSpecLSC getDataSpecLSC(AccessType access, const RegisterBlock &block);
static DataSpecLSC getDataSpecLSC(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                  const RegisterBlock &block, AccessClass aclass, bool atomic = false);
static inline GRFDisp getAddress(GRF r, const RegisterBlock &block, const MatrixAddressingStrategy &astrategy);

// Output code for prefetching a matrix chunk (XeHPG+).
template <HW hw>
void Generator<hw>::prefetchMatrix(const RegisterLayout &layout, const vector<RegisterRange> &addrs, const CommonStrategy &strategy, CommonState &state)
{
    for (int l = 0; l < layout.blocks(); l++) {
        prepareSeriesRegisterBlockMasking(layout, state, l);
        bool chain = strategy.chainPrefetches && (l + 1) < layout.blocks();
        if (chain) {
            auto &cur = layout[l], &next = layout[l + 1];
            chain &= next.isLoadBlock() && (cur.rowMask == next.rowMask) && (cur.colMask == next.colMask);
        }
        loadMatrixBlock(null, layout[l], layout.addressing(), layout.addressingStrategy(), addrs[l], strategy, state, layout.type().real().ngen(), false, true, chain);
    }

    finishRegisterBlockMasking(state);
}

// Output code for loading a matrix chunk into registers.
template <HW hw>
void Generator<hw>::loadMatrix(const GRFMultirange &dest, const RegisterLayout &layout, const vector<RegisterRange> &addrs,
                               const CommonStrategy &strategy, CommonState &state, bool readCheck)
{
    if (layout.empty()) return;

    auto &astrategy = layout.addressingStrategy();
    if (astrategy.prefetch && astrategy.newDP) {
        prefetchMatrix(layout, addrs, strategy, state);
        return;
    }

    if (strategy.readSuppressionWA && (layout.hasFlags() || !getDefaultNoMask()))
        doReadSuppressionWA(strategy, state);

    for (int l = 0; l < layout.blocks(); l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], dest);
        prepareSeriesRegisterBlockMasking(layout, state, l);
        loadMatrixBlock(dest[offsetReg], layout[l], layout.addressing(), astrategy, addrs[l], strategy, state, layout.type().real().ngen(), readCheck, true);
    }

    finishRegisterBlockMasking(state);
}

// Output code for loading a single matrix block into registers.
template <HW hw>
void Generator<hw>::loadMatrixBlock(const Register &dest, const RegisterBlock &block, const MatrixAddressing &atype,
                                    const MatrixAddressingStrategy &astrategy, const RegisterRange &addr,
                                    const CommonStrategy &strategy, CommonState &state, DataType T,
                                    bool readCheck, bool series, bool chain)
{
    InstructionModifier maskMod;
    InstructionModifier mod = block.simdSize;

    // Zero SIMD size blocks are filled as part of another load. Skip them.
    if (!block.isLoadBlock())
        return;

    // Prepare masking.
    FlagRegister flag;
    auto mask = registerBlockMasking(block, state, &flag);
    maskMod |= mask;
    mod |= mask;

    // Apply {Atomic} or look up preassigned token.
    if (chain)
        mod |= Atomic;
    else for (auto &entry: state.tokenMap) {
        if (entry.first == dest.getBase() || entry.first == addr.getBase()) {
            mod |= SBID(entry.second);
            break;
        }
    }

    if (astrategy.newDP) switch (block.implAccessType(atype, astrategy)) {
        case AccessType::Block:
        case AccessType::Scattered:
        case AccessType::ChannelScattered: {
            auto spec = getDataSpecLSC(atype, astrategy, block, AccessClass::Read, astrategy.atomic);
            if (hw >= HW::Xe3p && !astrategy.atomic) spec |= Overfetch;
            if (astrategy.atomic && hw >= HW::Xe2) {
                atomic(AtomicOp::load, mod, dest, spec, astrategy.base, getAddress(addr, block, astrategy));
            } else if (block.descAssigned) {
                MessageDescriptor desc;
                ExtendedMessageDescriptor exdesc;
                encodeLoadDescriptors(hw, desc, exdesc, block.simdSize, r0, spec, astrategy.base, null);
                send(mod, static_cast<SharedFunction>(block.sfid), dest, addr, null, exdesc.all, a0[0]);
            } else
                load(mod, dest, spec, astrategy.base, getAddress(addr, block, astrategy));
            break;
        }
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            int w = 0, h = 0, count = 0;
            block.getBlock2DWH(w, h, count, atype, astrategy.prefetch);
            auto spec = block_2d(getDataSizeLSC(block.ebytes, false), w, h, count) | astrategy.cachingR;
            if (astrategy.accessType == AccessType::Block2DTranspose) spec |= transpose;
            if (astrategy.accessType == AccessType::Block2DVNNI)      spec |= vnni;
            if(hw >= HW::Xe3p) spec |= Overfetch;
            load(mod, dest, spec, astrategy.base, getAddress(addr, block, astrategy));
            break;
        }
        default: stub();
    } else if (block.descAssigned)
        send(mod, static_cast<SharedFunction>(block.sfid), dest, addr, null, block.sfid, a0[0]);
    else switch (block.implAccessType(atype, astrategy)) {
        case AccessType::ChannelScattered: {
            static const ChannelMask cmasks[4] = {ChannelMask::r, ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
            if (block.ebytes != 4) stub();
            load(mod, dest, surface_dword(cmasks[block.count - 1]), astrategy.base, addr);
            break;
        }
        case AccessType::Scattered:
            if (block.ebytes == 8)
                load(mod, dest, scattered_qword(block.count), astrategy.base, addr);
            else if (block.ebytes == 4)
                load(mod, dest, scattered_dword(block.count), astrategy.base, addr);
            else if (block.ebytes == 1)
                load(mod, dest, scattered_byte(block.count), astrategy.base, addr);
            else
                hw_unsupported();
            break;
        case AccessType::Block:
            if (block.ebytes == 32)
                load(mod, dest, block_hword(block.count), astrategy.base, addr);
            else if (block.ebytes == 16 && !block.extra)
                load(mod, dest, block_oword(block.count), astrategy.base, addr);
            else if (block.ebytes == 16)
                load(mod, dest, aligned_block_oword(block.count), astrategy.base, addr);
            else
                hw_unsupported();
#ifdef STANDALONE
            if (readCheck && (astrategy.base.getModel() == ModelBTS)) {
                auto init = (strategy.strongReadCheck ? ~uint32_t(0) : uint32_t(0));
                if (flag.isValid())
                    mov<uint32_t>(block.simdSize | ~maskMod, dest, init);
                if (block.simdSize <= 2)
                    mov<uint32_t>(2, dest[2](1), init);
                if (block.simdSize <= 1)
                    mov<uint32_t>(1, dest[1], init);
            }
#endif
            break;
        default: stub();
    }

    if (series) {
        if (flag.isValid())
            state.raVFlag.unlock(flag);     /* was locked during preload */
    } else
        finishRegisterBlockMasking(state);
}

// Output code for storing a matrix chunk from registers.
template <HW hw>
void Generator<hw>::storeMatrix(const GRFMultirange &src, const RegisterLayout &layout,
                                const vector<RegisterRange> &addrs, const CommonStrategy &strategy, CommonState &state)
{
    for (int l = 0; l < layout.blocks(); l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], src);
        prepareSeriesRegisterBlockMasking(layout, state, l);
        storeMatrixBlock(src[offsetReg], layout[l], layout.addressing(), layout.addressingStrategy(), addrs[l], strategy, state, true);
    }

    finishRegisterBlockMasking(state);
}

// Output code for storing a matrix block from registers.
template <HW hw>
void Generator<hw>::storeMatrixBlock(const GRF &src, const RegisterBlock &block,
                                     const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                     const RegisterRange &addr, const CommonStrategy &strategy, CommonState &state, bool series)
{
    InstructionModifier mod = block.simdSize;

    // Zero SIMD size blocks are filled as part of another store. Skip them.
    if (!block.isLoadBlock())
        return;

    // Get mask to apply, if any.
    FlagRegister flag;
    mod |= registerBlockMasking(block, state, &flag);

    // Look up preassigned token.
    for (auto &entry: state.tokenMap) {
        if (entry.first == src.getBase()) {
            mod |= SBID(entry.second);
            break;
        }
    }

    if (block.descAssigned)
        send(mod, static_cast<SharedFunction>(block.sfid), null, addr, src, a0.ud(1), a0.ud(0));
    else if (astrategy.newDP) switch (block.implAccessType(atype, astrategy)) {
        case AccessType::Block:
        case AccessType::Scattered:
        case AccessType::ChannelScattered: {
            auto spec = getDataSpecLSC(atype, astrategy, block, AccessClass::Write);
            store(mod, spec, astrategy.base, getAddress(addr, block, astrategy), src);
            break;
        }
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            int w = 0, h = 0, count = 0;
            block.getBlock2DWH(w, h, count, atype, astrategy.prefetch);
            auto spec = block_2d(getDataSizeLSC(block.ebytes, false), w, h, count) | astrategy.cachingW;
            if (astrategy.accessType == AccessType::Block2DTranspose) spec |= transpose;
            if (astrategy.accessType == AccessType::Block2DVNNI)      spec |= vnni;
            store(mod, spec, astrategy.base, getAddress(addr, block, astrategy), src);
            break;
        }
        default: stub();
    } else switch (block.implAccessType(atype, astrategy)) {
        case AccessType::ChannelScattered: {
            static const ChannelMask cmasks[4] = {ChannelMask::r, ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
            if (block.ebytes != 4) stub();
            store(mod, surface_dword(cmasks[block.count - 1]), astrategy.base, addr, src);
            break;
        }
        case AccessType::Scattered:
            if (block.ebytes == 8)
                store(mod, scattered_qword(block.count), astrategy.base, addr, src);
            else if (block.ebytes == 4)
                store(mod, scattered_dword(block.count), astrategy.base, addr, src);
            else if (block.ebytes == 1)
                store(mod, scattered_byte(block.count), astrategy.base, addr, src);
            else
                hw_unsupported();
            break;
        case AccessType::Block:
            if (block.ebytes == 32)
                store(mod, block_hword(block.count), astrategy.base, addr, src);
            else if (block.ebytes == 16 && !block.extra)
                store(mod, block_oword(block.count), astrategy.base, addr, src);
            else
                hw_unsupported();
            break;
        default: stub();
    }

    if (series) {
        if (flag.isValid())
            state.raVFlag.unlock(flag);     /* was locked during preload */
    } else
        finishRegisterBlockMasking(state);
}

// Atomic addition of a matrix in registers.
template <HW hw>
void Generator<hw>::atomicAddMatrix(const GRFMultirange &src, const RegisterLayout &layout, const vector<RegisterRange> &addrs,
                                    const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    if (strategy.readSuppressionWA && (layout.hasFlags() || !getDefaultNoMask()))
        doReadSuppressionWA(strategy, state);

    for (int l = 0; l < layout.blocks(); l++) {
        auto offsetReg = contiguityCheck(hw, layout[l], src);
        prepareSeriesRegisterBlockMasking(layout, state, l);
        atomicAddMatrixBlock(layout.type(), src[offsetReg], layout[l], layout.addressing(), layout.addressingStrategy(), addrs[l], problem, strategy, state, true);
    }
}

template <HW hw>
void Generator<hw>::atomicAddMatrixBlock(Type T, const GRF &src, const RegisterBlock &block, const MatrixAddressing &atype,
                                         const MatrixAddressingStrategy &astrategy, const RegisterRange &addr,
                                         const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state, bool series)
{
    InstructionModifier maskMod;

    if (!block.isLoadBlock())   return;
    if (block.descAssigned)     stub();

    FlagRegister flag;
    maskMod |= registerBlockMasking(block, state, &flag);

    // SIMD16 A64 atomics are emulated with 2x SIMD8.
    bool a64 = (astrategy.base.getModel() == ModelA64);
    int hsize = a64 ? 2 : 1;
    int simd = block.simdSize;
    if (!astrategy.newDP && a64) simd = std::min(simd, 8);
    if (hw >= HW::XeHPC && block.ebytes < 8 && block.simdSize == 16 && simd == 8) stub();    // Can't split data GRFs.
    auto nreg = block.nregs();
    auto nregReal = (nreg * simd) / block.simdSize;

    auto specLSC = D32;
    if (astrategy.newDP) specLSC = getDataSpecLSC(atype, astrategy, block, AccessClass::Atomic);

    switch (block.implAccessType(atype, astrategy)) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
            if (hasNativeAtomicAdd(hw, T.real(), atype, astrategy)) {
                auto curSrc = src;
                for (int eoff = 0, hoff = 0; eoff < block.simdSize; eoff += simd, hoff += hsize, curSrc += nregReal) {
                    auto mod = simd | maskMod | ExecutionOffset(eoff);
                    if (block.ebytes * block.count != T.real().size()) stub();
                    if (astrategy.newDP) {
                        auto op = T.isFP() ? (hw >= HW::Xe3p && T == Type::bf16) ? AtomicOp::bfadd : AtomicOp::fadd
                                           : AtomicOp::add;
                        atomic(op, mod, specLSC, astrategy.base, getAddress(addr[hoff], block, astrategy), curSrc);
                    } else switch (T.real()) {
                        case Type::f32: atomic(AtomicOp::fadd, mod, scattered_dword(), astrategy.base, addr[hoff], curSrc); break;
                        case Type::f64: atomic(AtomicOp::fadd_64b, mod, scattered_qword(), astrategy.base, addr[hoff], curSrc); break;
                        case Type::u64:
                        case Type::s64: atomic(AtomicOp::add, mod, scattered_qword(), astrategy.base, addr[hoff], curSrc); break;
                        case Type::u32:
                        case Type::s32: atomic(AtomicOp::add, mod, scattered_dword(), astrategy.base, addr[hoff], curSrc); break;
                        case Type::u16:
                        case Type::s16: atomic(AtomicOp::add, mod, scattered_word(), astrategy.base, addr[hoff], curSrc); break;
                        default: stub();
                    }
                }
            } else {
                // Emulated atomic addition with a compare-and-swap loop.
                auto rOldNew = state.eatomicAddRegs[0];
                auto rSave = state.eatomicAddRegs[1];
                auto rOld = rOldNew[0];
                auto rNew = rOldNew[nregReal];
                auto flagToDo = getPhysicalFlag(state.vflagEAtomicAdd, state);
                auto ebytes = block.ebytes;
                if (ebytes == 1) ebytes = block.count;

                if (block.simdSize > 16) stub();    // Need 32 channels.
                if (astrategy.newDP)
                    load(block.simdSize | maskMod, rOld, specLSC, astrategy.base, getAddress(addr, block, astrategy));
                else if (astrategy.base.getModel() == ModelA64) {
                    if (ebytes == 2)
                        load(block.simdSize | maskMod, rOld, scattered_byte(2), astrategy.base, addr);
                    else if (ebytes == 4)
                        load(block.simdSize | maskMod, rOld, scattered_dword(), astrategy.base, addr);
                    else if (ebytes == 8)
                        load(block.simdSize | maskMod, rOld, scattered_qword(), astrategy.base, addr);
                } else {
                    if (ebytes == 2)
                        load(block.simdSize | maskMod, rOld, scattered_byte(2), astrategy.base, addr);
                    else if (ebytes == 4)
                        load(block.simdSize | maskMod, rOld, surface_dword(ChannelMask::r), astrategy.base, addr);
                    else if (ebytes == 8)
                        stub();         // needs cmpwr2
                }
                Label labelMask;

                // Save off high half of data when emulating SIMD16.
                if (block.simdSize > simd)
                    mov<uint32_t>(nregReal * 8, rOld.advance(nreg), rOld.advance(nregReal));

                if (flag.isValid()) {
                    if_(16 | flag, labelMask);
                    setDefaultNoMask(false);
                }

                and_(1 | NoMask, flagToDo, ce0, uint16_t((1 << block.simdSize) - 1));

                auto curSrc = src;

                for (int eoff = 0, hoff = 0; eoff < block.simdSize; eoff += simd, hoff += hsize) {
                    auto eoMod = ExecutionOffset(eoff);

                    Label labelCmpXchgLoop;
                    mark(labelCmpXchgLoop);

                    auto dt = T.ngen();
                    auto hs = std::max(1, 4 / ebytes);
                    add(int(simd * ebytes / T.real()) | eoMod | NoMask, rNew.retype(dt)[0](hs), rOld.retype(dt)[0](hs), curSrc.retype(dt)[0](hs));
                    mov<uint32_t>((simd * hs * ebytes / 4) | eoMod | NoMask, rSave, rOld);

                    auto atomicMod = simd | flagToDo | eoMod;
                    auto cmpMod = simd | flagToDo | ne | flagToDo | eoMod;

                    if (astrategy.newDP)
                        atomic(AtomicOp::cmpwr, atomicMod, rOld, specLSC, astrategy.base, getAddress(addr[hoff], block, astrategy), rOld);
                    else switch (ebytes) {
                        case 2: atomic(AtomicOp::cmpwr, atomicMod, rOld, scattered_word(),  astrategy.base, addr[hoff], rOld); break;
                        case 4: atomic(AtomicOp::cmpwr, atomicMod, rOld, scattered_dword(), astrategy.base, addr[hoff], rOld); break;
                        case 8: atomic(AtomicOp::cmpwr, atomicMod, rOld, scattered_qword(), astrategy.base, addr[hoff], rOld); break;
                        default: stub();
                    }

                    if (ebytes == 2)
                        cmp<uint16_t>(cmpMod, rSave[0][0](2), rOld[0](2));
                    else if (ebytes == 4)
                        cmp<uint32_t>(cmpMod, rSave, rOld);
                    else if (ebytes == 8) {
                        if (strategy.emulate.emulate64) {
                            cmp<uint32_t>(simd | ne | flagToDo | eoMod, rSave[0][0](2), rOld[0](2));
                            cmp<uint32_t>(simd | ~flagToDo | ne | flagToDo | eoMod, rSave[0][1](2), rOld[1](2));
                        } else
                            cmp<uint64_t>(cmpMod, rSave, rOld);
                    } else
                        stub();

                    (hw >= HW::XeHPC)        ? simtDoWhileLoop(16 | flagToDo |    any, labelCmpXchgLoop) :
                    strategy.fused           ? simtDoWhileLoop(16 | flagToDo | any16h, labelCmpXchgLoop) :
                    (eoff == 0 && simd == 8) ?            jmpi(1  | flagToDo |  any8h, labelCmpXchgLoop)
                                             :            jmpi(1  | flagToDo | any16h, labelCmpXchgLoop);

                    rOld += 2 * nregReal;
                    rNew += 2 * nregReal;
                    curSrc += nregReal;
                }

                if (flag.isValid()) {
                    mark(labelMask);
                    setDefaultNoMask(true);
                    endif(16);
                }
            }
            break;
        default: hw_unsupported();
    }

    if (series) {
        if (flag.isValid())
            state.raVFlag.unlock(flag);     /* was locked during preload */
    } else
        finishRegisterBlockMasking(state);
}

// Setup/teardown for descriptor handling code.
template <HW hw>
void Generator<hw>::setupTeardownLoadStoreDesc(bool setup, const RegisterLayout &layout,
                                               const CommonStrategy &strategy, CommonState &state)
{
    if (strategy.emulate.emulateDWxDW) {
        auto nconstants = (hw >= HW::XeHPG) ? 3 : 2;

        for (int s = 0; s < nconstants; s++) {
            auto &constant = state.lsDescConstant[s];
            if (setup) {
                if (constant.isInvalid()) {
                    constant = state.ra.alloc_sub<uint32_t>();
                    mov(1, constant, uint32_t(0x00100040 << s));
                }
            } else
                state.ra.safeRelease(constant);
        }
    }
}

// Output code for loading address register(s) with load/store message descriptors for remainders.
template <HW hw>
void Generator<hw>::loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block, Subregister count,
                                             const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                             const CommonStrategy &strategy, CommonState &state, bool clamp, int offset)
{
    if (!block.descRemR && !block.descRemC) return;

    MessageDescriptor descLoad; // a0.0:ud
    MessageDescriptor descStore; // a0.2 (a0.0 if no loads)
    ExtendedMessageDescriptor exdescLoad;
    ExtendedMessageDescriptor exdescStore; // a0.1

    Subregister t1 = state.ra.alloc_sub<uint32_t>();
    Subregister t2 = state.ra.alloc_sub<uint32_t>();
    Subregister t3;

    int crosspack = block.crosspack;

    offset += crosspack - 1;

    if (offset) {
        t3 = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, t3, count, offset);
        count = t3;
    }
    if (clamp) {
        if (block.descRemR == block.descRemC) stub();
        uint16_t maxCount = block.descRemR ? block.rowFragment : block.colFragment;
        if (t3.isInvalid()) t3 = state.ra.alloc_sub<uint32_t>();
        min_(1 | sat, t3, count, maxCount);
        count = t3;
    }
    if (crosspack > 1) {
        if (t3.isInvalid()) t3 = state.ra.alloc_sub<uint32_t>();
        shr(1, t3, count, ilog2(crosspack));
        count = t3;
    }

    if (astrategy.newDP) switch (astrategy.accessType) {
        case AccessType::ChannelScattered:
        case AccessType::Scattered:
        {
            bool channel = (astrategy.accessType == AccessType::ChannelScattered);

            encodeLoadDescriptors(hw, descLoad, exdescLoad, block.simdSize, r0, getDataSpecLSC(atype, astrategy, block, AccessClass::Read), astrategy.base, null);
            encodeStoreDescriptors(hw, descStore, exdescStore, block.simdSize, getDataSpecLSC(atype, astrategy, block, AccessClass::Write), astrategy.base, null);
            descLoad.cmask.cmask = 0;   // also vectSize
            descStore.cmask.cmask = 0;
            exdescStore.parts.extMessageLen = 0;
            descLoad.parts.responseLen = 0;

            uint32_t underlyingSIMD = std::max<uint32_t>(block.simdSize, uint32_t(maxScatteredSIMD(hw, astrategy)) >> 1);
            int log2GRFs = ilog2(underlyingSIMD * block.ebytes) - GRF::log2Bytes(hw);
            int log2Components = int(block.splitComplex);

            if (channel) mov(1, t2, 0x1000 << log2Components);
            mul(1, t1, state.lsDescConstant[log2GRFs + log2Components], count.uw());
            channel ? shl(1, t2, t2, count)
                    : shl(1, t2, count, 12 + log2Components);
            if (store)
                or_(1, a0.ud(1), t1.uw(0), exdescStore.all);
            add(1, t1.uw(0), t2, -0x1000);
            if (load)
                or_(1, a0.ud(0), t1, descLoad.all);
            if (store)
                or_(1, a0.ud(load ? 2 : 0), t1.uw(0), descStore.all);
            break;
        }
        default: hw_unsupported();
    } else switch (astrategy.accessType) {
        case AccessType::ChannelScattered:
        {
            encodeLoadDescriptors(hw, descLoad, exdescLoad, block.simdSize, r0, surface_dword(ChannelMask::rgba), astrategy.base, null);
            encodeStoreDescriptors(hw, descStore, exdescStore, block.simdSize, surface_dword(ChannelMask::rgba), astrategy.base, null);
            descLoad.surface.cmask = 0;              //
            descStore.surface.cmask = 0;             // Fields to fill in.
            exdescStore.parts.extMessageLen = 0;     //
            descLoad.parts.responseLen = 0;

            int log2Components = int(block.splitComplex);
            int shift = int(block.simdSize == 16) + log2Components;
            auto bitmask = uint16_t(0x0F00 << log2Components);

            if (strategy.emulate.emulateDWxDW)
                mul(1, t1, state.lsDescConstant[shift], count.uw());
            else
                mul(1, t1, count, uint32_t(0x00100040) << shift);
            mov(1, t2, bitmask);
            if (store)
                or_(1, a0.ud(1), t1.uw(0), exdescStore.all);
            shl(1, t2, t2, count);
            and_(1, t1.uw(0), t2, bitmask);
            if (load)
                or_(1, a0.ud(0), t1, descLoad.all);
            if (store)
                or_(1, a0.ud(load ? 2 : 0), t1.uw(0), descStore.all);
            break;
        }
        default: hw_unsupported();
    }

    state.ra.safeRelease(t1);
    state.ra.safeRelease(t2);
    state.ra.safeRelease(t3);
    block.sfid = exdescLoad.all;
}

// Start a double-masked section.
template <HW hw>
void Generator<hw>::startDoubleMask(VirtualFlag vflag, CommonState &state)
{
    finishRegisterBlockMasking(state);

    auto pflag = getPhysicalFlag(vflag, state);
    int simd = pflag.getBytes() * 8;

    state.blockEMask = vflag;
    if_(simd | pflag, state.blockDone);
    setDefaultNoMask(false);
}

template <HW hw>
void Generator<hw>::prepareSeriesRegisterBlockDoubleMasking(const RegisterLayout &layout, CommonState &state, int start)
{
    if (start + 1 >= layout.blocks())
        return;

    auto &block0 = layout[start];
    auto &block1 = layout[start + 1];

    if (!(block0.flag[0] && block0.flag[1]))
        return;

    if (state.blockEMask)
        for (const auto &flag: block0.flag)
            if (flag == state.blockEMask)
                return;

    for (int idx = 0; idx <= 1; idx++) {
        if (block0.flag[idx] && block0.flag[idx] == block1.flag[idx]) {
            startDoubleMask(block0.flag[idx], state);
            return;
        }
    }
}

template <HW hw>
void Generator<hw>::prepareSeriesRegisterBlockMasking(const RegisterLayout &layout, CommonState &state, int start)
{
    prepareSeriesRegisterBlockDoubleMasking(layout, state, start);

    if (state.vflagsEnabled()) {
        for (int startPreload = start; startPreload < layout.blocks(); startPreload++) {
            auto &block = layout[startPreload];

            if (!block.isLoadBlock()) continue;

            bool plFlag[2];
            for (int i = 0; i <= 1; i++)
                plFlag[i] = block.flag[i] && (block.flag[i] != state.blockEMask);

            if (plFlag[0] && plFlag[1]) break;      /* reached end of double masking block */
            if (!plFlag[0] && !plFlag[1]) continue;

            auto &flag = block.flag[plFlag[0] ? 0 : 1];
            if (!state.raVFlag.canLock(flag.n)) break;
            state.raVFlag.lock(getPhysicalFlag(flag, state), true);
        }
    }
}

template <HW hw>
InstructionModifier Generator<hw>::registerBlockMasking(const RegisterBlock &block, CommonState &state, FlagRegister *outFlag)
{
    InstructionModifier mod;

    // Remove any masks that have already been applied via `if`.
    auto flags = block.flag;
    for (auto &flag: flags)
        if (flag && flag == state.blockEMask)
            flag.clear();

    int idx = -1;
    if (flags[0] && flags[1]) {
        // Simultaneous row and column masking: one will be applied via `if`.
        // Since we don't know which is better here, we pick one.
        startDoubleMask(flags[1], state);
        idx = 0;
    } else if (flags[0])
        idx = 0;
    else if (flags[1])
        idx = 1;

    if (idx >= 0) {
        auto pflag = getPhysicalFlag(flags[idx], state);
        if (outFlag) *outFlag = pflag;
        if (block.flagInvert)
            mod |= ~pflag;
        else
            mod |= pflag;
        if (block.simdSize > 1) {
            if (hw >= HW::XeHPC) {
                if (block.flagAll) mod |= all;
                if (block.flagAny) mod |= any;
            } else if (block.flagAll)
                mod |= (block.simdSize > 8) ? all16h : all8h;
            else if (block.flagAny)
                mod |= (block.simdSize > 8) ? any16h : any8h;
        }
    } else if (outFlag)
        *outFlag = FlagRegister();

    return mod;
}

template <HW hw>
void Generator<hw>::finishRegisterBlockMasking(CommonState &state)
{
    if (state.blockEMask) {
        setDefaultNoMask(true);
        mark(state.blockDone);                      state.blockDone = Label{};
        endif(state.blockEMask.getBytes() * 8);     state.blockEMask.clear();
        state.wipeActiveVFlags();
    }
}

static DataSizeLSC getDataSizeLSC(int ebytes, bool pad32)
{
    switch (ebytes) {
        case 8: return DataSizeLSC::D64;
        case 4: return DataSizeLSC::D32;
        case 2: return pad32 ? DataSizeLSC::D16U32 : DataSizeLSC::D16;
        case 1: return pad32 ? DataSizeLSC::D8U32  : DataSizeLSC::D8;
    }
    stub("Invalid data size");
}

static DataSpecLSC getDataSpecLSC(AccessType access, const RegisterBlock &block)
{
    DataSpecLSC D32{DataSizeLSC::D32};
    DataSpecLSC D64{DataSizeLSC::D64};
    auto T = DataSpecLSC::createTranspose();

    switch (access) {
        case AccessType::ChannelScattered: {
            static const ChannelMask cmasks[4] = {ChannelMask::r, ChannelMask::rg, ChannelMask::rgb, ChannelMask::rgba};
            if (block.ebytes != 4) hw_unsupported();
            return D32 | cmasks[block.count - 1];
        }
        case AccessType::Scattered:
            if (block.ebytes == 8) return D64(block.count);
            if (block.ebytes == 4) return D32(block.count);
            if (block.ebytes == 2) { if (block.count != 1) hw_unsupported(); return DataSizeLSC::D16U32; }
            if (block.ebytes == 1) return getDataSizeLSC(block.count, true);
            hw_unsupported();
        case AccessType::Block:
            if (block.ebytes == 8) return D64(block.count) | T;
            if (block.ebytes == 4) return D32(block.count) | T;
            hw_unsupported();
        default: stub();
    }
}

static DataSpecLSC getDataSpecLSC(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                  const RegisterBlock &block, AccessClass aclass, bool atomic)
{
    auto caching = (aclass == AccessClass::Read) ? astrategy.cachingR : astrategy.cachingW;
    if (aclass == AccessClass::Atomic || atomic)
        caching = makeL1Uncacheable(caching);
    return getDataSpecLSC(block.implAccessType(atype, astrategy), block) | caching;
}

static inline GRFDisp getAddress(GRF r, const RegisterBlock &block, const MatrixAddressingStrategy &astrategy)
{
    if (isBlock2D(astrategy.accessType))
        return r + block.offset2D();
    else
        return r + block.offsetAddr;
}

GEMMSTONE_NAMESPACE_END
