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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_PROBLEM_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_PROBLEM_HPP

#include "gemmstone/config.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"
#include "gemmstone/type.hpp"

GEMMSTONE_NAMESPACE_START

// Matrix layouts in memory.
enum class MatrixLayout : uint8_t {
    N = 0,  Nontranspose = N,
    T = 1,  Transpose = T,
    Pc = 2, PackedColumns = Pc,
    Pr = 3, PackedRows = Pr
};

static inline bool isPacked(MatrixLayout l) {
    return (l == MatrixLayout::PackedRows) || (l == MatrixLayout::PackedColumns);
}

static inline bool isColMajor(MatrixLayout l) {
    return (l == MatrixLayout::N || l == MatrixLayout::Pc);
}

static inline bool isRowMajor(MatrixLayout l) {
    return (l == MatrixLayout::T || l == MatrixLayout::Pr);
}

static inline MatrixLayout transposeLayout(MatrixLayout l) {
    return static_cast<MatrixLayout>(static_cast<uint8_t>(l) ^ 0x1);
}

inline char layoutChar(MatrixLayout layout)
{
    switch (layout) {
        case MatrixLayout::N:  return 'N';
        case MatrixLayout::T:  return 'T';
        case MatrixLayout::Pc: return 'A';
        case MatrixLayout::Pr: return 'B';
        default:               return '?';
    }
}

inline MatrixLayout charLayout(char c) {
    switch (c) {
        case 'A': return MatrixLayout::Pc;
        case 'B': return MatrixLayout::Pr;
        case 'N': return MatrixLayout::N;
        case 'T': return MatrixLayout::T;
        case '?': return MatrixLayout::N; // Either N/T; used for fused GEMM.
        default:
            throw std::runtime_error("Unknown matrix layout requested.");
    }
}

struct MatrixAddressing {
    MatrixLayout layout = MatrixLayout::N; // Layout type (N/T/Pr/Pc)
    uint8_t pad[3] = {};
    uint32_t packSize = 0;          // # of elements in a packed row/column for packed layouts.
    uint16_t tileR = 0, tileC = 0;  // Tiling (0 if none) for packed layouts.
    uint8_t panelLength = 0;        // Length of the panel for packed layouts = #cols/rows for Pc/Pr respectively.
    uint8_t crosspack = 1;          // Crosspack for packed layouts.
    uint8_t alignment = 1;          // Alignment for all addresses, offsets, and leading dimensions.
    bool needA64 = false;

    void setAlignment(int align) { alignment = static_cast<uint8_t>(sanitizeAlign(align)); }
    int defaultAlignment(Type T) const {
        return sanitizeAlign((isPacked(layout) ? (packSize * crosspack) : 1) * T);
    }

    void transpose() {
        layout = transposeLayout(layout);
        std::swap(tileR, tileC);
    }

    bool isColMajor() const {
        return ::GEMMSTONE_NAMESPACE::isColMajor(layout);
    }

    bool isRowMajor() const {
        return ::GEMMSTONE_NAMESPACE::isRowMajor(layout);
    }

private:
    static int sanitizeAlign(int align) { return std::min(128, largest_pow2_divisor(align)); }
};

// Information on a scalar argument (e.g. alpha/beta).
class Scalar {
public:
    enum ScalarType {Fixed, Variable, Pointer, RealPointer};

private:
    int value;
    ScalarType type;

public:
    Scalar() : Scalar(Variable) {}
    explicit Scalar(ScalarType type_) : value(0), type(type_) {}
    explicit Scalar(int value_) : value(value_), type(Fixed) {}

    Scalar &operator=(int value_) { type = Fixed; value = value_; return *this; }
    Scalar &operator=(ScalarType type_) { type = type_; value = 0; return *this; }

    template <typename U> bool operator==(U value_) const   { return fixed() && (value == value_); }
    bool operator==(ScalarType type_) const                 { return (type == type_); }
    template <typename U> bool operator!=(U value_) const   { return !operator==(value_); }

    operator int() const {
        if (!fixed()) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }
    operator double() const { return int(*this); }

    bool fixed() const { return (type == Fixed); }
    bool pointer() const { return (type == Pointer) || (type == RealPointer); }
    ScalarType getType() const { return type; }
};

// Types of updates for GEMM kernels.
enum class UpdateType {
    Full,
};

// A/B offset mode.
enum class ABOffset {
    None,       // No A/B offsets.
    Calc,       // Calculate A/B row/column sums in kernel.
    Load,       // Use precalculated row/column sums.
};

// C offset mode.
enum class COffset {
    None,       // No C offsets.
    Post,       // C offset after all other updates.
    Pre,        // C offset before all other updates (bias).
};

// Batch mode.
enum class BatchMode {
    None, Strided, Nonstrided, Variable
};

// Binary operations.
enum class BinaryOp {
    Add, Sub, Mul, Div,
    Min, Max,
    Prelu,
    ScaleSub,   /* internal use only */
    MulSub      /* internal use only */
};

// Problem parameters shared between kernel types.
struct CommonProblem {
    bool nonuniformWGs = false;                 // Support nonuniform workgroups?
    bool gtpinSupport = false;                  // Support GT-Pin?
};

// GEMM kernel problem description.
struct GEMMProblem : public CommonProblem {
    Type Ta, Tb, Tc, Ts;                            // Types for A/B/C/scalars in registers.
    Type Ta_ext, Tb_ext, Tc_ext;                    // Types for A/B/C data in memory.
    Type Tao, Tbo, Tco;                             // Types for A/B/C offsets.
    Type Ta_scale, Tb_scale, Tc_scale;              // Types for A/B/C scales.
    Type Tag, Tbg;                                  // Types for A/B group sums.
    Type Tacc_min;                                  // Type for minimum C accumulation precision.

    Scalar alpha, beta;                             // Scaling factors for A*B and C, respectively.
    MatrixAddressing A, B, C;                       // Addressing information for A/B/C matrices.
    MatrixAddressing AO, BO, CO;                    // Addressing information for A/B/C offsets (if 2D).
    MatrixAddressing A_scale, B_scale, C_scale;     // Addressing information for A/B scales (if 2D).
    MatrixAddressing Ag, Bg;                        // Addressing information for A/B group sums.

    bool checkBeta0 = true;                         // If true, check for beta = 0 and handle specially.
    ABOffset aOffset = ABOffset::None;              // A/B offset modes.
    ABOffset bOffset = ABOffset::None;              //
    bool aOffsetAfterScale = false;                 // If true, A/B offsets are applied after the scales:
    bool bOffsetAfterScale = false;                 //   x = q * scale - offset (floating-point offsets).
    int aoPtrDims = -1, boPtrDims = -1;             // A/B offset dimensionality (-1: none; 0: scalar; 1: vector, 2: matrix)
    int coPtrDims = -1;                             // C offset dimensionality (-1: none or hostscalar; 0 - others)
    int asPtrDims = -1, bsPtrDims = -1, csPtrDims = -1;           // A/B scale dimensionality (-1: none; 0: scalar; 1: vector, 2: matrix)
    int aqGroupM = 0, aqGroupK = 0;                 // Group sizes for A quantization parameters (offsets and scales)
    int bqGroupN = 0, bqGroupK = 0;                 // Group sizes for B quantization parameters (offsets and scales)
    int cqGroupM = 0, cqGroupN = 0;                 // Group sizes for C quantization parameters (offsets and scales)
    COffset cOffset = COffset::None;                // C offset mode.
    BatchMode batch = BatchMode::None;              // Batch mode.
    int batchDims = 0;                              // # of batch dimensions (strided batch only).
    bool sumA = false, sumB = false;                // If true, calculate A row sums/B column sums and store in CO.
    bool hasGroupSumsA = false;
    bool hasGroupSumsB = false;
    bool bdpasEnabled = false;                             // bdpas enabled for problem.
    bool cMXScale = false;
    MatrixAddressing sroundSeed;
    PostOpsProblem postOps;                         // Fused post operations to apply
    ngen::Product product;

    // The following data is derived from the postOps and does not need
    //   to be considered for equality/hashing purposes.
    std::vector<MatrixAddressing> binary;                   // Binary postop data
    std::vector<Type> Tbinary;                              // Binary types

    bool hasPostOp() const { return !postOps.empty(); }
    bool hasNonSum1PostOp() const {
        for (const auto &e : postOps.ops)
            if (!e.is_sum()) return true;
        return false;
    }
    bool hasBinaryPostOp() const {
        for (auto &e : postOps.ops)
            if (e.is_binary()) return true;
        return false;
    }
    bool hasSum1PostOpAtEnd() const {
        return !postOps.empty() && postOps.ops.back().is_sum();
    }
    void removeFinalSumPostOp() {
        if (hasSum1PostOpAtEnd())
            postOps.ops.pop_back();
    }
    bool binaryPostProcess() const { return postOps.cStochasticRound || hasCScale(); }


    bool beta0() const   { return (beta  ==  0); }
    bool beta1() const   { return (beta  ==  1); }
    bool alpha1() const  { return (alpha ==  1); }
    bool alphaM1() const { return (alpha == -1); }

    bool needsTsConvert() const {
        if (!(alpha1() || alphaM1())) return true;
        if (!(beta0() || beta1())) return true;
        if (beta1() && !Tc_ext.isSubsetOf(Tc)) return true;
        if ((Tc == Type::s32 || Tc == Type::u32) && Tc_ext == Type::bf16) return true;
        if (hasNonSum1PostOp()) return true;
        return false;
    }

    bool isIGEMM() const {
        return (Ta.real().isInt8() && Tb.real().isInt8() && Tc.real().paddedSize() == 4);
    }

    bool gemmt() const { return false; }
    bool backward() const { return false; }

    bool hasAScalePtr() const { return (asPtrDims > -1); }
    bool hasBScalePtr() const { return (bsPtrDims > -1); }
    bool hasCScale() const { return (csPtrDims > 0); }
    bool hasCMXScale() const { return cMXScale; }
    bool hasAOffsetPtr() const { return (aoPtrDims > -1); }
    bool hasBOffsetPtr() const { return (boPtrDims > -1); }
    bool hasCOffsetPtr() const { return (coPtrDims > -1); }
    bool aOffsetHostScalar() const {return aoPtrDims == -1 && aOffset == ABOffset::Calc; }
    bool bOffsetHostScalar() const {return boPtrDims == -1 && bOffset == ABOffset::Calc; }
    bool cOffsetHostScalar() const {return coPtrDims == -1 && cOffset != COffset::None; }

    bool aScale2D() const { return (asPtrDims >= 2); }
    bool bScale2D() const { return (bsPtrDims >= 2); }
    bool aOffset2D() const { return (aoPtrDims >= 2); }
    bool bOffset2D() const { return (boPtrDims >= 2); }

    bool quantized2DA() const { return hasGroupSumsB || aOffset2D() || aScale2D(); }
    bool quantized2DB() const { return hasGroupSumsA || bOffset2D() || bScale2D(); }

    bool earlyDequantizeA() const { return (aOffset == ABOffset::Calc && earlyDequantizableOffset(Ta_ext, Tao, Ta)) || (aScale2D() && (Ta_scale.isSubsetOf(Ta) || (Ta.isFP() && !Ta.isF4() && !Ta.isF8()))); }
    bool earlyDequantizeB() const { return (bOffset == ABOffset::Calc && earlyDequantizableOffset(Tb_ext, Tbo, Tb)) || (bScale2D() && (Tb_scale.isSubsetOf(Tb) || (Tb.isFP() && !Tb.isF4() && !Tb.isF8()))); }

    bool needsASums() const { return sumA || (bOffset == ABOffset::Calc && !earlyDequantizeB() && !quantized2DB()); }
    bool needsBSums() const { return sumB || (aOffset == ABOffset::Calc && !earlyDequantizeA() && !quantized2DA()); }

    bool nativeBDPAS() const {
        return ((Ta.isF4() || Ta.isF8() || Ta == Type::f16 || Ta == Type::bf16) &&
                (Tb.isF4() || Tb.isF8() || Tb == Type::f16 || Tb == Type::bf16) && 
        ngen::getCore(product.family) >= ngen::HW::Xe3p);
    }
    bool forceLateQuant(int minOPCount) const {
        bool fp4_fp8_dpas = ((Ta.isF8() && Tb.isF8()) || (Ta.isF4() && Tb.isF4())) && nativeBDPAS();
        return fp4_fp8_dpas && ((aScale2D() && !preferBDPAS() && aqGroupK % minOPCount == 0)
            || (bScale2D() && !preferBDPAS() && bqGroupK % minOPCount == 0));
    }
    bool forceUpconvertQuant() const {
        // Cover cases where scale group < ksys by upconverting, using normal dpas and scale routines.
        return nativeBDPAS() && Ta.isF4() && Tb.isF4() && ((aScale2D() && !preferBDPAS() && aqGroupK % 64 != 0)
            || (bScale2D() && !preferBDPAS() && bqGroupK % 64 != 0));
    }
    bool preferBDPAS() const {
        bool useBDPAS = (bdpasEnabled && nativeBDPAS() && (aScale2D() || bScale2D()));
        if (aScale2D()) useBDPAS &= (Ta_scale == Type::f8_e8m0) && (aqGroupK % 32 == 0);
        if (bScale2D()) useBDPAS &= (Tb_scale == Type::f8_e8m0) && (bqGroupK % 32 == 0);
        return useBDPAS;
    }
    bool validLateScaleGroups() const {
        return ((aScale2D() && (aqGroupK <=1 || aqGroupK % 32 == 0))
            || (bScale2D() && (bqGroupK <=1 || bqGroupK % 32 == 0)));
    }

    bool needsAGroupSums() const { return (bOffset == ABOffset::Calc && quantized2DB() && !earlyDequantizableOffset(Tb_ext, Tbo, Tb)); }
    bool needsBGroupSums() const { return (aOffset == ABOffset::Calc && quantized2DA() && !earlyDequantizableOffset(Ta_ext, Tao, Ta)); }

    bool usesCOPtr() const { return (hasCOffsetPtr() && cOffset != COffset::None) || sumA || sumB; }
    bool allowMatrixOffset() const { return (cOffset == COffset::Pre); }

    // Data type of dst registers for dpas/mad/dp4a.
    Type Tc_compute() const {
        if (Ta.isInteger() && Tb.isInteger() && Tc == Type::f32)
            return Type::s32;
        else if (Ta.isFP() && Tb.isFP() && Tc == Type::s32)
            return Type::f32;
        else
            return Tc;
    }

    bool isLegalAAlignment(int align, int unrollM) const { return (A.layout != MatrixLayout::N) || ((unrollM * Ta) % align == 0); }
    bool isLegalBAlignment(int align, int unrollN) const { return (B.layout != MatrixLayout::T) || ((unrollN * Tb) % align == 0); }

    inline void autoTypeConversions(bool systolicAvailable);
    void transpose();

    std::string toString() const;
    std::string scalarsToString() const;

    static bool earlyDequantizableOffset(Type T_ext, Type To, Type T) {
        return To.asSigned().isSubsetOf(T) && (To.bits() < T.bits() || T_ext.bits() < T.bits());
    }

    /* Serialization for kernel cache. */
    void serialize(SerializationStream &s) const
    {
        s.append(Ta, Tb, Tc, Ts);
        s.append(Ta_ext, Tb_ext, Tc_ext);
        s.append(Tao, Tbo, Tco);
        s.append(Ta_scale, Tb_scale, Tc_scale);
        s.append(alpha);
        s.append(beta);
        s.append(A, B, C);
        s.append(AO, BO, CO);
        s.append(A_scale, B_scale, C_scale);
        s.append(checkBeta0);
        s.append(bdpasEnabled);
        s.append(aOffset, bOffset);
        s.append(aOffsetAfterScale, bOffsetAfterScale);
        s.append(aoPtrDims, boPtrDims, coPtrDims);
        s.append(asPtrDims, bsPtrDims, csPtrDims);
        s.append(aqGroupM, aqGroupK);
        s.append(bqGroupN, bqGroupK);
        s.append(cqGroupM, cqGroupN);
        s.append(cOffset);
        s.append(batch);
        s.append(batchDims);
        s.append(sumA, sumB);
        s.append(sroundSeed);
        s.append(postOps);
        s.append(product);
    }
};


// Apply automatic internal type conversions to a problem.
void GEMMProblem::autoTypeConversions(bool systolicAvailable)
{
    using namespace ngen;
    auto hw = getCore(product.family);
    // Weights decompression
    if ((Ta.isInt8() || Ta.isSubByteInt()) && Tb.isFP() && Tc.isFP()) Ta = Tb;
    if ((Tb.isInt8() || Tb.isSubByteInt()) && Ta.isFP() && Tc.isFP()) Tb = Ta;

    if (Ta.isFP() && Tb.isFP() && Tc.isFP()) {
        if (Ta.bits() < Tb.bits()) Ta = Tb;
        if (Tb.bits() < Ta.bits()) Tb = Ta;
    }

    if (Ta == Ta_ext.asSigned()) Ta = Ta_ext;
    if (Tb == Tb_ext.asSigned()) Tb = Tb_ext;
    if (hw < HW::Xe3p || !systolicAvailable)
    {
        if (Ta.isF8()) Ta = Type::f16;
        if (Tb.isF8()) Tb = Type::f16;
    }
    if (hw < HW::Xe3p ||  product.family < ProductFamily::CRI || !systolicAvailable || forceUpconvertQuant())
    {
        if (product.family == ProductFamily::NVLP && (!forceUpconvertQuant() || validLateScaleGroups())){
            if (Ta.isF4()) Ta = Type::bf8;
            if (Tb.isF4()) Tb = Type::bf8;
        } else {
            if (Ta.isF4()) Ta = Type::f16;
            if (Tb.isF4()) Tb = Type::f16;
        }
    }

    if (!systolicAvailable && Tc == Type::f32) {
        if (Ta == Type::f16) Ta = Type::f32;
        if (Tb == Type::f16) Tb = Type::f32;
    }

    if (hw < HW::XeHP || (hw > HW::XeHP && !systolicAvailable)) {
        if (Ta == Type::bf16) Ta = Type::f32;
        if (Tb == Type::bf16) Tb = Type::f32;
    }
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
