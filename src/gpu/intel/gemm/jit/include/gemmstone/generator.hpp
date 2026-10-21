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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_GENERATOR_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_GENERATOR_HPP

#define BINARY_OUTPUT

#include <array>
#include <bitset>
#include <complex>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

#include "gemmstone/config.hpp"
#include "internal/ngen_includes.hpp"

#include "gemmstone/type.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "generator/pieces/copy_plan.hpp"
#include "generator/pieces/register_layout.hpp"
#include "generator/pieces/state.hpp"
#include "ngen_emulation.hpp"


GEMMSTONE_NAMESPACE_START

// Macro configuration
#ifndef GENERATOR_BASE

#define GENERATOR_SUPER(hw) ngen::OpenCLCodeGenerator<hw>
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#define FORWARD(hw) NGEN_FORWARD_SYCL(hw)
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define FORWARD(hw) NGEN_FORWARD_OPENCL(hw)
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#define FORWARD(hw) NGEN_FORWARD_LEVEL_ZERO(hw)
#endif
#define GENERATOR_DEBUGINFO {__FILE__, __LINE__}

#define GENERATOR_BASE(hw) GENERATOR_SUPER(hw)
#endif

template <ngen::HW hw> class Generator : public GENERATOR_BASE(hw) {
public:
    using super = GENERATOR_SUPER(hw);

    Generator(ngen::Product product) : GENERATOR_BASE(hw)(product, GENERATOR_DEBUGINFO) {}

    FORWARD(hw)

    // Kernel generation entrypoints.
    void gemm(GEMMProblem problem, GEMMStrategy strategy, const ngen::InterfaceHandler &interface_);
    void gemmMicrokernel(GEMMProblem problem, GEMMStrategy strategy, const ngen::InterfaceHandler &interface_);
    microkernel::Package gemmMicrokernelPackage(const GEMMProblem &problem, const GEMMStrategy &strategy, const ngen::InterfaceHandler &interface_, const microkernel::Protocol &protocol, uint32_t gmdid, bool transposeC = false);

    // Driver information retrieval.
    static CommonDriverInfo driverInfo(GEMMProblem problem, const GEMMStrategy &strategy);

protected:
    ngen::InterfaceHandler &interface = super::interface_;

    std::exception_ptr lastException;
    GRFMultirange outputCRange;
    RegisterLayout outputCLayout;
    microkernel::ClobberSet knownClobbers;

    using Injector = PostOpsProblem::Injector<hw>;
    std::unique_ptr<Injector> postOpInjector;

    class status_stream {
    protected:
        char cc;
        std::stringstream line;
        bool lineStart = true;

        Generator<hw> &parent;

        friend class Generator<hw>;

    public:
        status_stream(Generator<hw> &parent_, int color = 1) : cc(color + '0'), parent(parent_) {}

        static constexpr struct Endl {} endl{};

        template <typename T>
        status_stream &operator<<(const T &obj) {
            return *this;
        }

        status_stream &operator<<(const Endl &e) {
            return *this;
        }
    } status{*this};

    enum class HintType     {Bank0, Bank1, TempComp0, TempComp1, LongTerm, LongTerm0, LongTerm1, R0Info, A0, A0Broadcast, A1, A1Broadcast, B0, B0Broadcast, B1, B1Broadcast, C, C1, CLoad, S, D, SAddr, DAddr};
    enum class StdCRemType  {Ignore, Mask, Descriptor};
    enum class COperation   {Load, Update, UpdateStore, Store};
    enum class KLoop {
        GEMM,
    };
    enum class KBarrierType {Normal, Signal, Wait};

    friend std::ostream &operator<<(std::ostream &s, StdCRemType rt) {
        const char *names[3] = {"ignore", "mask", "custom descriptor"};
        return (s << names[static_cast<int>(rt) % 3]);
    }

    // The generator method definitions are split between .cxx files
    //   in the generator/pieces folder.

    // address_setup.cxx
    template <typename BO> void setupAddr(Type T, const ngen::GRFRange &addr, const BO &ptr, const RegisterBlock &block, const ngen::Subregister &ld, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state, const Address2DParams &params = {}, LDMultiples ldMultiples = {});
    template <typename BO> void setupAddr(const std::vector<ngen::GRFRange> &addr, const BO &ptr, const RegisterLayout &layout, const ngen::Subregister &ld, const CommonStrategy &strategy, CommonState &state, const Address2DParams &params = {}, const LDMultiples &ldMultiples = {}, int start = 0);

    void offsetAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc, const RegisterBlock &blockDst, const RegisterBlock &blockSrc, int offsetFixed, int offsetLD, const ngen::Subregister &ld, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state, const LDMultiples &ldMultiples = {});
    void setupAddrRel(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc, const RegisterBlock &blockDst, const RegisterBlock &blockSrc, const RegisterLayout &layout, const ngen::Subregister &ld, const CommonStrategy &strategy, CommonState &state, const LDMultiples &ldMultiples = {});

    template <typename I, typename Ir, typename Ic> void incAddrShifted(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc, I inc, Ir incR, Ic incC, const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic> void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);
    template <typename I> void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic> void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc, I inc, Ir incR, Ic incC, const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic> void incAddr(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);
    template <typename I> void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc, I inc, const RegisterBlock &blockDst, const RegisterBlock &blockSrc, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state);
    template <typename I> void incAddr(const std::vector<ngen::GRFRange> &addr, I inc, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);
    template <typename A, typename I, typename Ir, typename Ic> void incDecAddr(const A &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state, bool decrement);
    template <typename A, typename I> void incDecAddr(const A &addr, I inc, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state, bool decrement);
    void incAddrK(const std::vector<ngen::GRFRange> &addr, bool column, int k, const SubregisterPair &ld, const LDIncrements &incs, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);

    void setAddrRemainder(Type T, const ngen::GRFRange &addr, const RegisterBlock &block, const ngen::Subregister &remR, const ngen::Subregister &remC, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state);
    void setAddrRemainder(const std::vector<ngen::GRFRange> &addr, const RegisterLayout &layout, const ngen::Subregister &remR, const ngen::Subregister &remC, const CommonStrategy &strategy, CommonState &state);

    ngen::Subregister startShift(const MultishiftSubregister &ptr, int shift, CommonState &state);
    SubregisterPair startShift(const SubregisterPair &ptr, int shift, CommonState &state);
    template <typename BO> typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value, BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO> typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value, BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO, typename BI> typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);
    template <typename BO, typename BI> typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);
    void doneShift(const SubregisterPair &ptr, const SubregisterPair &ptrShifted, int shift, CommonState &state);

    // asm_helpers.cxx
    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip) { goto12(mod, jip, jip); }
    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip, ngen::Label &uip, bool branchCtrl = false);

    template <typename DT = void> void mulConstant(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1);

    void cmp0(const ngen::InstructionModifier &mod, ngen::RegData src0);
    void syncall();

    void wrdepRanges(const GRFMultirange &rr)               { for (auto &r : rr.ranges) wrdep(r); }
    void wrdepRanges(const std::vector<GRFMultirange> &rrs) { for (auto &rr : rrs) wrdepRanges(rr); }

    void simtDoWhileLoop(const ngen::InstructionModifier &mod, ngen::Label &dest);

    void activeThreadBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info, const CommonStrategy &strategy);
    void activeThreadBarrierSignal(const ngen::GRF &temp, const ngen::GRF &r0_info, const CommonStrategy &strategy);
    void slmBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info, const CommonStrategy &strategy);
    void globalMemFence(const ngen::GRF &temp, const ngen::GRF &r0_info, const CommonStrategy &strategy);
    void globalMemBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info, const CommonStrategy &strategy);

    void pause(const CommonStrategy &strategy);

    void doReadSuppressionWA(const CommonStrategy &strategy, CommonState &state);

    // atomic_fusions.cxx
    void gemmStoreZeroC(GEMMProblem problem, GEMMStrategy strategy, GEMMState state, bool initialZeroing = true);
    void gemmFusedBetaPOInit(const ngen::Subregister &groupID, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmFusedBetaScale(GEMMProblem problem, GEMMStrategy strategy, GEMMState &state);
    void gemmFusedBetaNotifyCompletion(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmFusedBetaWaitCompletion(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool gemmFusedPostOpsFinalize(ngen::Label &labelLateExit, GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmRedirectToTempC(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    // c_update.cxx
    bool gemmAccessC(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool gemmUpdateC(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmUpdateCDispatch(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmAccessSums(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void updateC(const GRFMultirange &C_acc, const GRFMultirange &C_accSwap, const GRFMultirange &C_load, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void updateCLayout(const RegisterLayout &layoutExt, const ngen::GRFRange (&C_addr0)[2], const RegisterBlock &C_block0, COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool doStdCRemainder(RegisterLayout &layoutExt, RegisterLayout &layoutExtUnmasked, bool inside, bool columns[2], StdCRemType remTypes[2], bool fragments[2], bool fragPositives[2], int fragSizes[2], const ngen::GRFRange (&C_addr0)[2], const ngen::GRFRange (&C_addr0Unmasked)[2], COperation op, std::vector<MaskAssignment> &masks, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState state, RegisterBlock *C_block0 = nullptr, RegisterBlock *C_blockUnmasked0 = nullptr);
    void doAlternateCRemainder(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void convert(const GRFMultirange &range, Type Told, Type Tnew, const CommonStrategy &strategy, CommonState &state);
    bool gemmConvertC(Type Tnew, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmDotReduce(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmKReduce(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmPrefetchC(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void setupCAddr0(ngen::GRFRange (&C_addr0)[2], ngen::GRFRange (&C_addr0Unmasked)[2], const RegisterLayout &C_layout, const RegisterLayout &C_layoutUnmasked, int C_count, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, const Address2DParams *params = nullptr);

    // common.cxx
    void initInterface(const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);
    void initState(const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);

    void prologue(const CommonStrategy &strategy, int internalSIMD = 16);
    void prologue(const GEMMStrategy &strategy, GEMMState &state);
    void epilogue(const CommonStrategy &strategy, CommonState &state);

    void padding();

    void allocVFlagStorage(const CommonStrategy &strategy, CommonState &state, bool saveCurrent = true);
    void deallocVFlagStorage(CommonState &state, bool saveCurrent = true);
    ngen::FlagRegister getPhysicalFlag(VirtualFlag vflag, CommonState &state);

    ngen::Subregister copySubregister(const ngen::Subregister &reg, CommonState &state, ngen::Bundle hint = ngen::Bundle(ngen::Bundle::any, 0));
    void duplicateScalar(SubregisterPair &val, CommonState &state);
    void deduplicateScalar(SubregisterPair &val, CommonState &state);
    MultishiftSubregister multishift(const ngen::Subregister &reg, unsigned shifts, const CommonStrategy &strategy, CommonState &state, ngen::Bundle hint = ngen::Bundle());

    void getFusedID(int scale, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);

    void moveR0(const CommonStrategy &strategy, CommonState &state);
    void moveR0(const GEMMStrategy &strategy, GEMMState &state);
    template <typename F> inline void useR0(CommonState &state, F f);
    template <typename F> inline void useTempAndR0(CommonState &state, F f);

    void removeSG(const CommonProblem &problem, const CommonStrategy &strategy, const CommonState &state);
    void reorderFusedEUs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void zeroMatrix(const GRFMultirange &r, const CommonStrategy &strategy);

    ngen::GRF loadScalars(Type T, const std::vector<ngen::Subregister> &src, const CommonStrategy &strategy, CommonState &state);
    ngen::GRFRange loadVector(Type Tsrc, Type Tdst, ngen::Subregister ptr, int n, ngen::Subregister rem, const CommonStrategy &strategy, CommonState &state);

    void broadcastToWG(ngen::FlagRegister leaderFlag, ngen::GRF value, const CommonStrategy &strategy, CommonState &state, int slmOffset = 0);

    void extendIndexVec(int n, CommonState &state);
    ngen::Subregister accessIndexVec(int n, CommonState &state);

    LDMultiples createLDMultiples(bool a64, int nmultiples, const ngen::Subregister &ld, const CommonStrategy &strategy, CommonState &state);
    ngen::Subregister findLDMultiple(const LDMultiples &multiples, bool a64, int n, const CommonStrategy &strategy, CommonState &state);

    void calcIncrement(LDIncrements &increments, SubregisterPair &base, int scale, const CommonStrategy &strategy, CommonState &state, bool avoidConflicts = true);
    SubregisterPair lookupIncrement(const LDIncrements &increments, const SubregisterPair &base, int scale, const CommonStrategy &strategy, CommonState &state, bool *release = nullptr);

    // copy.cxx
    friend struct CopyInstruction;
    friend struct CopyResource;
    void copyRegisterBlock(Type Ts, Type Td, const RegisterBlock &blockSrc, const RegisterBlock &blockDst, const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false);
    void copyRegisters(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false, bool s4Shift = true);
    void copyRegisters(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC, bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false, bool s4Shift = true);
    void copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false, bool s4Shift = true);
    void copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC, bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false, bool s4Shift = true);
    void copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC, const Scalar &alpha, const SubregisterPair &alpha_real, const SubregisterPair &alpha_imag, bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc = false, bool s4Shift = true);
    void copyExecute(CopyPlan &&plan, CommonState &state);
    void overlappedCopy(const GRFMultirange &src, const GRFMultirange &dst, CommonState &state);

    // emulation.cxx
    friend struct ngen::EmulationImplementation;
    template <typename DT = void> void emov(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::RegData src0,   const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename DT = void> void emov(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::Immediate src0, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {})                                              { ngen::EmulationImplementation::emov<DT>(*this, mod, dst, src0, strategy.emulate, loc); }
    template <typename DT = void> void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename DT = void> void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, ngen::Immediate src1,      const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {}) { ngen::EmulationImplementation::eadd<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc); }
    template <typename DT = void> void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename DT = void> void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, ngen::Immediate src1,      const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {}) { ngen::EmulationImplementation::emul<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc); }
    template <typename DT = void> void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {})                           { ngen::EmulationImplementation::eshl<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc); }
    template <typename DT = void> void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst, ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {})                           { ngen::EmulationImplementation::eshr<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc); }
    template <typename DT = void> void emulConstant(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1, const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {})      { ngen::EmulationImplementation::emulConstant<DT>(*this, mod, dst, src0, src1, strategy.emulate, state.emulate, loc); }
    template <typename DT = void> void emulConstant(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, Type src1, const CommonStrategy &strategy, const CommonState &state, ngen::SourceLocation loc = {});
    template <typename S1> void emul32High(const ngen::InstructionModifier &mod, const ngen::RegData &dstHi, const ngen::RegData &src0, const S1 &src1, ngen::SourceLocation loc = {})                                                                     { ngen::EmulationImplementation::emul32High(*this, mod, dstHi, src0, src1, loc); }

    template <typename S0, typename S2> void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, const S2 &src2, const CommonStrategy &strategy, CommonState &state, bool sub, ngen::SourceLocation loc = {});
    template <typename S0> void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, const ngen::Immediate &src2, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename S0> void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, ngen::RegData src1, ngen::RegData src2, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename S0> void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, int32_t src2, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename S0> void eaddScaled(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, Type src2, const CommonStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename DT = void, typename S0, typename S2> void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, const S2 &src2, ngen::SourceLocation loc = {});
    template <typename S0> void ecsel(const ngen::InstructionModifier &mod, const ngen::InstructionModifier &cmod, const ngen::FlagRegister &flag, const ngen::RegData &dst, const S0 &src0, const ngen::RegData &src1, const ngen::RegData &src2, ngen::SourceLocation loc = {});

    template <typename DT = void> void emath(const ngen::InstructionModifier &mod, ngen::MathFunction fc, const ngen::RegData &dst, const ngen::RegData &src0, const GEMMStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {});
    template <typename DT = void> void einv(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const GEMMStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {}) { emath<DT>(mod, ngen::MathFunction::inv, dst, src0, strategy, state, loc); }
    template <typename DT = void> void esqt(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const GEMMStrategy &strategy, CommonState &state, ngen::SourceLocation loc = {}) { emath<DT>(mod, ngen::MathFunction::sqt, dst, src0, strategy, state, loc); }

    void ejmpi(ngen::InstructionModifier mod, ngen::Label &dst, ngen::SourceLocation loc = {});

    // gemm.cxx
    void gemm(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmSubkernel(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state);
    bool gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);
    bool gemmBodyInternal(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool gemmKLoop(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateC(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool wgRemCheck(const GEMMProblem &problem, const GEMMStrategy &strategy);
    template <typename Problem> bool mnRemainderHandling(LoopType loop, Problem &problem, GEMMStrategy &strategy, GEMMState &state, bool (Generator<hw>::*func)(Problem, GEMMStrategy, GEMMState));
    template <typename Problem> bool mnJointSplitRemainderHandling(Problem &problem, GEMMStrategy &strategy, GEMMState &state, bool (Generator<hw>::*func)(Problem, GEMMStrategy, GEMMState));
    bool gemmMEdge(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);
    void gemmOOBExit(ngen::Label &target, const GEMMStrategy &strategy, GEMMState &state);

    void makeSLMBaseRelative(ngen::Subregister addr, const GEMMState &state);

    // gemm_setup.cxx
    void gemmInitInterface(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK = false);
    void gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK = false);

    bool gemmAccumulateCSetup(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAccumulateCTeardown(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmAllocateTokens(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmABPrefetchAddrSetup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA = true, bool doB = true);

    void gemmOffsetAm(const ngen::Subregister &i, const ngen::Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetAk(int h, const ngen::Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetAk(const ngen::Subregister &h, const ngen::Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBk(int h, const ngen::Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBk(const ngen::Subregister &h, const ngen::Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBn(const ngen::Subregister &j, const ngen::Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmFoldOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmRestoreOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmReverseLoops(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmScaleInputs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcWGRemainders(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmGetBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmReleaseBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmOffsetABC(bool initial, ngen::Subregister i0, ngen::Subregister j0, ngen::Subregister h0, ngen::Subregister i0p, ngen::Subregister j0p, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA = true, bool doB = true, bool doC = true, bool doBinary = false);
    void gemmOffsetBatchABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmSetupABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmCacheLDABMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA = true, bool doB = true);
    void gemmCacheLDCMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool prefetch = false);

    void gemmDowngradeAccess(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    CoopSplit effCoopSplitA(const GEMMProblem &problem, const GEMMStrategy &strategy);
    CoopSplit effCoopSplitB(const GEMMProblem &problem, const GEMMStrategy &strategy);

    void gemmFreeIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA = true, bool doB = true);
    void gemmCalcIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int ka_load = 0, int kb_load = 0, bool doA = true, bool doB = true);
    void gemmCalcQuantizationIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    ngen::Subregister gemmMNLinearID(const GEMMStrategy &strategy, GEMMState &state);

    void gemmApplyWorkshareOffset(bool isA, ngen::Subregister &base, ngen::Subregister alias, Address2DParams &params2D, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, int r, int c, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcWorkshareAOffset(ngen::Subregister &off, ngen::Subregister &offR, ngen::Subregister &offC, const MatrixAddressing &A, const MatrixAddressingStrategy &A_strategy, int ma, int ka, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcWorkshareBOffset(ngen::Subregister &off, ngen::Subregister &offR, ngen::Subregister &offC, const MatrixAddressing &B, const MatrixAddressingStrategy &B_strategy, int kb, int nb, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // k_loop.cxx
    void gemmAIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, int ka_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int ha = 0);
    void gemmAIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const MultishiftSubregister &ka_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int ha = 0);
    void gemmAIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &ka_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int ha = 0);
    template <typename I> void gemmAIncrement(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, I ka_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int ha = 0, int h = 0);
    void gemmALoad(const GRFMultirange &regs, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    template <typename I> void gemmALoadInc(const GRFMultirange &regs, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, I ka_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmBIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, int kb_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int hb = 0);
    void gemmBIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const MultishiftSubregister &kb_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int hb = 0);
    void gemmBIncrementInternal(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &kb_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int hb = 0);
    template <typename I> void gemmBIncrement(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, I kb_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int hb = 0, int h = 0);
    void gemmBLoad(const GRFMultirange &regs, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    template <typename I> void gemmBLoadInc(const GRFMultirange &regs, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, I kb_inc, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    template <bool doA> void gemmAiBiRemLoadInc(int h, bool incremental, bool incrementalCopy, bool keepAddrTogether, bool willRemask, const ngen::Subregister &kSLMX, const GRFMultirange &Xi_regs, const RegisterLayout &Xi_layout, const std::vector<ngen::GRFRange> &Xi_addrs, const std::vector<RegisterLayout> &Xi_layoutK, const std::vector<std::vector<ngen::GRFRange>> &Xi_addrsK, const GRFMultirange &Xo_regs, const RegisterLayout &Xo_layout, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmSLMRemask(bool remaskA, bool remaskB, GRFMultirange &Ao_regs, GRFMultirange &Bo_regs, int kOffset, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void kLoopActivateABRemainder(bool active, bool doA, bool doB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int kOffset = 0);
    void kLoopActivateSLMRemainder(bool active, bool preactivate, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int kOffset = 0);

    void kLoop(KLoop type, const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool kLoopSingle(KLoop type, const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    // k_loop_setup.cxx
    bool gemmPrepMaskedAB(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmCalcKLoopBarrierCount(ngen::Subregister &count, const ngen::Subregister &k, int cooldown, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcKSLM(const ngen::Subregister &kSLM, const ngen::Subregister &lid, int kgran, int kdiv, int krep, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, ngen::Subregister kBase = ngen::Subregister());
    void gemmCalcKSLMA(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, ngen::Subregister kBase = ngen::Subregister());
    void gemmCalcKSLMB(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, ngen::Subregister kBase = ngen::Subregister());

    void kLoopAllocBarrierHeader(GEMMState &state);
    ngen::GRF kLoopGetBarrierHeader(const GEMMStrategy &strategy, GEMMState &state);

    bool kLoopSetup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void kLoopTeardown(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // l3_prefetch.cxx
    void gemmInitL3Prefetch(bool nextWave, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmWarmupL3Prefetch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmScheduleL3Prefetches(void *ls, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmScheduleL3PrefetchIncs(void *ls, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool allowDelay = true);
    void gemmTeardownL3Prefetch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // layout_setup.cxx
    bool tryAddRemainder(Type T, RegisterBlock &block, bool remainderR, bool remainderC, RemainderOptions remOpts, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);
    bool tryAddRemainder(RegisterLayout &layout, bool remainderR, bool remainderC, RemainderOptions remOpts);
    void addRemainder(RegisterLayout &layout, bool remainderR, bool remainderC, RemainderOptions remOpts);
    void addRemainder(RegisterLayout &layout, std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &ld, bool remainderR, bool remainderC, RemainderOptions remOpts, const CommonStrategy &strategy, CommonState &state, int dataRegs = -1);
    void updateBlock2DSizes(ngen::GRF addr, const RegisterBlock &dst, const RegisterBlock &src, const MatrixAddressing &atype, bool prefetch);
    void adjustSubblockAddrs(const RegisterLayout &sublayout, const std::vector<ngen::GRFRange> &subaddrs, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const CommonStrategy &strategy, const CommonState &state);

    // masks.cxx
    bool assignMasks(RegisterLayout &layout, LoopType rloop, LoopType cloop, std::vector<MaskAssignment> &assignments, const CommonStrategy &strategy, CommonState &state, bool retryVirtual = false, const std::vector<MaskAssignment> *existing = nullptr);
    void loadMask(MaskAssignment assignment, ngen::Subregister index, const CommonStrategy &strategy, CommonState &state, int offset = 0);
    void loadMasks(const std::vector<MaskAssignment> &assignments, ngen::Subregister (&indices)[3], const CommonStrategy &strategy, CommonState &state, int start = 0);
    void loadMasks(const std::vector<MaskAssignment> &assignments, ngen::Subregister (&indices)[3], int (&offsets)[3], const CommonStrategy &strategy, CommonState &state, int start = 0);

    // math_helpers.cxx
    void addScaled(const ngen::InstructionModifier &mod, const ngen::RegData &dst, int src0, const ngen::RegData &src1, int numerator, int denominator, CommonState &state, bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, int numerator, int denominator,  CommonState &state, bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod, const ngen::RegData &dst, const ngen::RegData &src0, int src1, int numerator, int denominator,  CommonState &state, bool exact = false);
    template <typename S0, typename S1> void addScaled(const ngen::InstructionModifier &mod, const ngen::RegData &dst, S0 src0, S1 src1, Type T, CommonState &state, bool exact = false, int scale = 1);

    template <typename DT = void> void mod(const ngen::InstructionModifier &finalMod, const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void mod(const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void modExt(const ngen::Subregister &dstMod, const ngen::Subregister &dstMultiple, const ngen::Subregister &src, uint16_t modulus, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void alignDown(const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void alignDown(const ngen::InstructionModifier &mod, const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void alignUp(const ngen::Subregister &dst, const ngen::Subregister &src, uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void divDown(const ngen::Subregister &dst, const ngen::Subregister &src, uint32_t divisor, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void divUp(const ngen::Subregister &dst, const ngen::Subregister &src, uint32_t divisor, const CommonStrategy &strategy, CommonState &state, bool relax = false);
    template <typename DT = void> void divDown(const ngen::Subregister &dst, const ngen::Subregister &src0, const ngen::Subregister &src1, const ngen::Subregister &src1Recip, const ngen::FlagRegister &flag, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void> void divUp(const ngen::Subregister &dst, const ngen::Subregister &src0, const ngen::Subregister &src1, const ngen::Subregister &src1Recip, const ngen::FlagRegister &flag, const CommonStrategy &strategy, CommonState &state);
    void divMod(const ngen::Subregister &qot, const ngen::Subregister &rem, const ngen::Subregister &num, const ngen::Subregister &denom, const GEMMStrategy &strategy, CommonState &state, bool large = false);

    // matrix_access.cxx
    void setupTeardownLoadStoreDesc(bool setup, const RegisterLayout &layout, const CommonStrategy &strategy, CommonState &state);
    void loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block, ngen::Subregister count, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const CommonStrategy &strategy, CommonState &state, bool clamp = false, int offset = 0);
    void startDoubleMask(VirtualFlag vflag, CommonState &state);
    void prepareSeriesRegisterBlockDoubleMasking(const RegisterLayout &layout, CommonState &state, int start);
    void prepareSeriesRegisterBlockMasking(const RegisterLayout &layout, CommonState &state, int start);
    ngen::InstructionModifier registerBlockMasking(const RegisterBlock &block, CommonState &state, ngen::FlagRegister *outFlag = nullptr);
    void finishRegisterBlockMasking(CommonState &state);

    void loadMatrixBlock(const ngen::Register &dest, const RegisterBlock &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const ngen::GRFRange &addr, const CommonStrategy &strategy, CommonState &state, bool readCheck = false, bool series = false);
    void loadMatrix(const GRFMultirange &dest, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const CommonStrategy &strategy, CommonState &state, bool readCheck = false);
    void prefetchMatrix(const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const CommonStrategy &strategy, CommonState &state);
    void storeMatrixBlock(const ngen::GRF &src, const RegisterBlock &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const ngen::GRFRange &addr, const CommonStrategy &strategy, CommonState &state, bool series = false);
    void storeMatrix(const GRFMultirange &src, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const CommonStrategy &strategy, CommonState &state);
    void atomicAddMatrixBlock(Type T, const ngen::GRF &src, const RegisterBlock &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const ngen::GRFRange &addr, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state, bool series = false);
    void atomicAddMatrix(const GRFMultirange &src, const RegisterLayout &layout, const std::vector<ngen::GRFRange> &addrs, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);

    // matrix_multiply.cxx
    void innerProductFMA(int h, int ha, int hb, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout, const GRFMultirange &A_regs, const GRFMultirange &B_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void outerProductFMA(int h, int ha, int hb, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout, const GRFMultirange &A_regs, const GRFMultirange &B_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void outerProductSystolic(int h, int ha, int hb, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout, const GRFMultirange &A_regs, const GRFMultirange &B_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void outerProduct(int h, int ha, int hb, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout, const GRFMultirange &A_regs, const GRFMultirange &B_regs, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void setupTeardownAccumulateSumSystolic(bool setup, Type Tother, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void outerProductRepackC(int x0, int xr0, int nx, int h, bool rem, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void applyLateABOffset(bool isA, int h, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int x0 = 0, int xr0 = 0, int nx = -1);

    // monolithic_k_loop_dpasw.cxx
    bool sysgemmAccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void sysgemmKLoop(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void sysgemmKLoop4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool oddB);
    void sysgemmStoreSignal(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool forceFence = false);
    void sysgemmCopyLoad(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool useC = false);
    void sysgemmCopyLoad4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool loadB, int useC = 0, ngen::RegData flagLoadB = ngen::RegData());
    void sysgemmCopyStore(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool first = false);
    void sysgemmCopyStore4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool storeB, int useC = 0, int useC_B = 0);
    void sysgemmMultiply(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool lastMultiply = false);
    void sysgemmMultiply4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool firstMultiply = false, ngen::RegData flagWaitLoad = ngen::RegData(), ngen::RegData flagSignal = ngen::RegData(), ngen::Label *labelDone = nullptr);
    void sysgemmMultiplyChunk(const GEMMProblem &problem, const GEMMStrategy &strategy, bool first, int ao, int i0, bool waitB, bool prepB, const ngen::InstructionModifier &swsb0 = ngen::InstructionModifier(), const ngen::InstructionModifier &swsbEnd = ngen::InstructionModifier());
    void sysgemmBarrierPrep(const ngen::InstructionModifier &swsb, const ngen::GRF &header);
    void sysgemmReorderLocalIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    bool sysgemm2AccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2KLoopCompute(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2KLoopCopy(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2Multiply(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool cooldown = false, ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(), ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX32(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool cooldown = false, ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(), ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX48(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool cooldown = false, ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(), ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyChunkX32(const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA, bool odd);
    void sysgemm2MultiplyChunkX48(const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA);

    // post_ops.cxx
    void gemmAlphaScale(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool cxCombine = true);
    void gemmBetaScale(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void binaryOp(BinaryOp op, int simd, const ngen::RegData &dst, const ngen::RegData &src0, const ngen::RegData &src1, CommonState &state);
    void gemmScalarBinaryOpC(BinaryOp op, Type Tco, const GRFMultirange &offsets, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmScalarBinaryOpC(BinaryOp op, Type Tco, const ngen::Subregister &scalar, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmVectorBinaryOpC(BinaryOp op, bool column, const GRFMultirange &offsets, const ngen::Subregister &scale, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, Type Tco = Type::invalid, RegisterLayout CO_layout = RegisterLayout(), int y0 = -1, int y1 = -1);
    void gemmRank1UpdateC(const GRFMultirange &r, const GRFMultirange &c, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcABOffsetAddrs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool gemmLoadABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmApplyABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool gemmBinaryOpC(BinaryOp op, bool row, bool column, Type Tco, MatrixAddressing CO, MatrixAddressingStrategy CO_strategy, ngen::Subregister base, ngen::Subregister ld, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    bool gemmApplyCOffsetDispatch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    void gemmApplyPostOps(size_t poMin, size_t poMax, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmApplyMXScale(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmLoadBinaryOpArgs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // quantization.cxx
    bool gemmMake2DQuantizationLayouts(bool isA, const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmRepack2DQuantizationData(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmRepack2DQuantizationData(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmRepack2DOffsetData(Type Text, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool offsetAfterScale);
    void dequantizeSubByteIntShift(Type Tsrc, GRFMultirange src, const CommonStrategy &strategy);
    void dequantizeSubByteInt(bool doA, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const RegisterLayout &layoutOffset, const RegisterLayout &layoutScale, const GRFMultirange &src, const GRFMultirange &dst, const GRFMultirange &offset, const GRFMultirange &scale, int offR, int offC, int h, int kab_load, int kq_load, const GEMMProblem *problem, const CommonStrategy &strategy, CommonState &state, bool signedShift = true);
    void gemmDequantizeOperation(bool doA, Type T, Type Tq, BinaryOp op, const RegisterLayout &layout, const RegisterLayout &qlayout, const GRFMultirange &regs, const GRFMultirange &qregs, int h, int kab_load, int kq_load, const GEMMProblem &problem, const CommonStrategy &strategy, CommonState &state, const RegisterLayout *olayout = nullptr, const GRFMultirange *oregs = nullptr);
    void gemmDequantizeAB(bool doA, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst, const GRFMultirange &src, const GRFMultirange &dst, int h, int kab_load, int kab_repack, int kq_load, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool s4Shift = true);
    // register_allocation.cxx
    ngen::Bundle getHint(HintType type);
    ngen::Bundle getHint(HintType type, const CommonStrategy &strategy);
    ngen::Bundle getHint(HintType type, const GEMMStrategy &strategy);

    void gemmAllocRegs(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocAoBoRegs(const GEMMStrategy &strategy, GEMMState &state);

    // remask.cxx
    void setupTeardownRemask(Type T, int index, bool setup, int nq, ngen::Subregister remQ, const CommonStrategy &strategy, CommonState &state, int fixedOffQ = 0, const ngen::Subregister &variableOffQ = ngen::Subregister());
    void remaskLayout(int index, bool column, const RegisterLayout &layout, const GRFMultirange &regs, const CommonStrategy &strategy, CommonState &state, int offset = 0);
    void remaskLayoutSingle(int index, bool column, int nq, ngen::Subregister remQ, const RegisterLayout &layout, const GRFMultirange &regs, const CommonStrategy &strategy, CommonState &state, int fixedOffQ = 0, const ngen::Subregister &variableOffQ = ngen::Subregister(), int maskOff = 0);

    // row_column_sums.cxx
    void accumulateSum(bool column, const GRFMultirange &srcRegs, const RegisterLayout &srcLayout, const GRFMultirange &dstRegs, const RegisterLayout &dstLayout, const CommonStrategy &strategy, CommonState &state, int q0 = -1, int q1 = -1);
    void makeSumLayout(bool column, const RegisterLayout &srcLayout, Type Tdst, RegisterLayout &dstLayout, const CommonStrategy &strategy, CommonState &state, bool systolicSum = false);
    void horizontalAdd(bool column, const GRFMultirange &regs, RegisterLayout &layout, CommonState &state);
    bool gemmFinalizeSums(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // state_utils.cxx
    void saveMNLocalIDs(const GEMMStrategy &strategy, GEMMState &state);
    void saveKLocalIDSize(const GEMMStrategy &strategy, GEMMState &state);
    void releaseSavedMNLocalIDs(GEMMState &state);

    // stream_k.cxx
    void gemmStreamKPrepareSlice2(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmStreamKSetup(ngen::Label &lKVPhaseDone, ngen::Label &lKernelDone, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // tlb_warmup.cxx
    void tlbWarmup(ngen::AddressBase base, const ngen::Subregister &ptr, const ngen::Subregister &bytes, const ngen::Subregister &lid, int whose, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);
    void tlbWarmup(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const ngen::Subregister &base, const ngen::Subregister &r, const ngen::Subregister &c, const ngen::Subregister &ld, const ngen::Subregister &lid, int whose, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state);
    void gemmTLBWarmup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

    // walk_orders.cxx
    void gemmLinearOrder(const ngen::Subregister &groupIDMN, const ngen::Subregister &groupIDM, const ngen::Subregister &groupIDN, const ngen::Subregister &aLeader, const ngen::Subregister &bLeader, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmSimpleLinearOrder(const ngen::Subregister &groupIDMN, const ngen::Subregister &groupIDM, const ngen::Subregister &groupIDN, const ngen::Subregister &aLeader, const ngen::Subregister &bLeader, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmNestedLinearOrder(const ngen::Subregister &groupIDMN, const ngen::Subregister &groupIDM, const ngen::Subregister &groupIDN, const ngen::Subregister &aLeader, const ngen::Subregister &bLeader, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmHilbertlikeOrder(const ngen::Subregister &groupIDMN, const ngen::Subregister &groupIDM, const ngen::Subregister &groupIDN, const ngen::Subregister &aLeader, const ngen::Subregister &bLeader, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmBoustrophedonOrder(const ngen::Subregister &groupIDMN, const ngen::Subregister &groupIDM, const ngen::Subregister &groupIDN, const ngen::Subregister &aLeader, const ngen::Subregister &bLeader, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmReorderGlobalIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
    void gemmReorderLocalIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);
};

#define MOCK_BARRIERS

template <ngen::HW hw> using gemm_kernel_generator_t = Generator<hw>;

#include "internal/generator_inline.hxx"
GEMMSTONE_NAMESPACE_END

#endif /* header guard */
