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

#include <stdexcept>

#include "gpu/intel/gemm/jit/gen_kernel.hpp"

#include "common/c_types_map.hpp"
#include "common/impl_registration.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gemmstone/../../generator/pieces/compute_utils.hpp"
#include "gemmstone/../../generator/pieces/ngen_object_helpers.hpp"
#include "gemmstone/../../generator_dsl/builder.hpp"
#include "gemmstone/../../generator_dsl/kernel_desc.hpp"
#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/generator.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/gemm/jit/gen_kernel_db.hpp"
#include "gpu/intel/gemm/jit/pd.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "gpu/intel/logging.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

using namespace gemmstone;
using namespace intel::jit;

namespace {
std::string ms_hint(double score, int eu_count, int freq_mhz) {
    if (eu_count <= 0 || freq_mhz <= 0) return "";
    char b[40];
    snprintf(b, sizeof(b), " [%.3fms]", score / (eu_count * freq_mhz * 1e3));
    return b;
}

SelectionObserver make_entry_observer(int eu_count, int freq_mhz) {
    return [eu_count, freq_mhz](const kcatalog::Entry *entry, double score,
                   EvaluateAuxOutput) {
        gpu_debug() << "[" << lpad(entry - catalog().entries, 4)
                    << "] consider score:" << score
                    << ms_hint(score, eu_count, freq_mhz) << " "
                    << entry->str();
    };
}
} // anonymous namespace

bool enable_generator_dsl() {
    static const bool ret
            = gpu_utils::dev_getenv("enable_generator_dsl", false);
    return ret;
}

status_t gen_desc_t::create_generator(
        const intel::engine_t &engine, compute::kernel_t &kernel) const {
    gen_kernel_t kd(*this);
    return engine.create_kernel(&kernel, &kd);
}

compute::scalar_type_t gen_desc_t::scalar_type() const {
    switch (problem_.Ts) {
        case Type::u2: return compute::scalar_type_t::_uint2;
        case Type::s4: return compute::scalar_type_t::_int4;
        case Type::u4: return compute::scalar_type_t::_uint4;
        case Type::s8: return compute::scalar_type_t::_char;
        case Type::u8: return compute::scalar_type_t::_uchar;
        case Type::s16: return compute::scalar_type_t::_short;
        case Type::u16: return compute::scalar_type_t::_ushort;
        case Type::s32: return compute::scalar_type_t::_int;
        case Type::u32: return compute::scalar_type_t::_uint;
        case Type::s64: return compute::scalar_type_t::_long;
        case Type::u64: return compute::scalar_type_t::_ulong;
        case Type::f4_e2m1: return compute::scalar_type_t::_f4_e2m1;
        case Type::bf8: return compute::scalar_type_t::_bfloat8;
        case Type::hf8: return compute::scalar_type_t::_hfloat8;
        case Type::bf16: return compute::scalar_type_t::_bfloat16;
        case Type::f16: return compute::scalar_type_t::_half;
        case Type::f32: return compute::scalar_type_t::_float;
        case Type::f64: return compute::scalar_type_t::_double;
        default: return compute::scalar_type_t::undef;
    }
}

#ifdef DNNL_DEV_MODE
static gemmstone::Scalar stringToScalar(const std::string &val) {
    using namespace gemmstone;
    return val[0] == '-' ? Scalar(Scalar::Variable) : Scalar(std::stoi(val));
}

status_t gen_desc_t::apply_kernel_override(
        const std::string &ovr_strategy, const char *tags) {
    try {
        // Overrides problem data types; incompatible precisions lose accuracy.
        entry_ = nullptr;
        const GEMMProblem pd = problem_;
        const Scalar pd_alpha = problem_.alpha, pd_beta = problem_.beta;
        stringstream_t ss(ovr_strategy);
        std::string val;
        // Skip tokens before `gemm`; the pd sets them.
        while (ss >> val && val != "gemm") {}
        if (val != "gemm") throw std::runtime_error("expected 'gemm' prefix");
        ss >> val;
        const char *pstr = val.c_str();
        // Cannot modify external data types; f32 data may be read as tf32.
        auto ext_ok = [](Type &t, Type ovr) {
            bool f32 = utils::one_of(t, Type::f32, Type::tf32)
                    && utils::one_of(ovr, Type::f32, Type::tf32);
            if (t != ovr && !f32) return false;
            t = ovr;
            return true;
        };
        Type ext_dt;
        pstr = parsePrecisions(pstr, ext_dt, problem_.Ta);
        if (!ext_ok(problem_.Ta_ext, ext_dt))
            throw std::runtime_error("invalid external A data type");
        pstr = parsePrecisions(pstr, ext_dt, problem_.Tb);
        if (!ext_ok(problem_.Tb_ext, ext_dt))
            throw std::runtime_error("invalid external B data type");
        if (*pstr == '[') {
            pstr = parsePrecisions(pstr, problem_.Tc, ext_dt);
            if (ext_dt != problem_.Tc_ext)
                throw std::runtime_error("invalid external C data type");
        } else {
            pstr = parsePrecision(pstr, problem_.Tc);
        }
        ss >> val;
        pstr = val.c_str();
        pstr = parseLayout(pstr, problem_.A);
        pstr = parseLayout(pstr, problem_.B);
        pstr = parseLayout(pstr, problem_.C);
        if (problem_.A.layout != pd.A.layout || problem_.B.layout != pd.B.layout
                || problem_.C.layout != pd.C.layout)
            throw std::runtime_error("layout mismatch");

        if (problem_.A.alignment == 0)
            problem_.A.setAlignment(
                    problem_.A.defaultAlignment(problem_.Ta_ext));
        if (problem_.B.alignment == 0)
            problem_.B.setAlignment(
                    problem_.B.defaultAlignment(problem_.Tb_ext));
        if (problem_.C.alignment == 0)
            problem_.C.setAlignment(
                    problem_.C.defaultAlignment(problem_.Tc_ext));

        if (pd.A.alignment % problem_.A.alignment
                || pd.B.alignment % problem_.B.alignment
                || pd.C.alignment % problem_.C.alignment)
            throw std::runtime_error("alignment too strong");

        strategy_ = GEMMStrategy(hw_, stepping_);
        ss >> strategy_.unroll[LoopM];
        ss >> strategy_.unroll[LoopN];

        ss >> val;
        problem_.alpha = stringToScalar(val);
        ss >> val;
        problem_.beta = stringToScalar(val);
        auto scalar_ok = [](const Scalar &want, const Scalar &ovr) {
            return !ovr.fixed() || want == int(ovr);
        };
        if (!scalar_ok(pd_alpha, problem_.alpha)
                || !scalar_ok(pd_beta, problem_.beta))
            throw std::runtime_error("alpha/beta mismatch");

        if (!ss) throw std::runtime_error("truncated kernel string");
        std::string rest = ss.str().substr(ss.tellg());
        parseStrategy(rest, problem_, strategy_);

        // Later k-chunks accumulate (catalog parity).
        auto block_k = strategy_.blocking[LoopK];
        if (block_k <= 0) block_k = strategy_.defaultBlockK(hw_, problem_);
        if (k_ > block_k && !problem_.beta1()) problem_.beta = Scalar();

        if (strategy_.kParallel)
            throw std::runtime_error("unknown k0 for global k-parallel");
        aux_params_ = EvaluateAuxOutput();
        aux_params_.kParallelVariable = strategy_.kParallelVariable;
        if (strategy_.kParallelLocal) {
            if (strategy_.wg[LoopK] < 1 || strategy_.unroll[LoopK] < 1)
                throw std::runtime_error("invalid k-parallel-local geometry");
            aux_params_.k0
                    = utils::rnd_up(utils::div_up(k_, strategy_.wg[LoopK]),
                            strategy_.unroll[LoopK]);
            aux_params_.wgK = std::max(1,
                    std::min(strategy_.wg[LoopK],
                            int(utils::div_up(k_, aux_params_.k0))));
        }
    } catch (const std::exception &e) {
        reject_reason_ = e.what();
        return status::unimplemented;
    }
    return status::success;
}
#endif

status_t gen_desc_t::finalize(const char *tags) {
    reject_reason_.clear();
    auto reject = [&](const std::string &why) {
        reject_reason_ = why;
        return status::unimplemented;
    };
    problem_.CO.setAlignment(problem_.Tco.paddedSize());

#ifdef DNNL_DEV_MODE
    if (!kernel_override_.empty()) {
        CHECK(apply_kernel_override(kernel_override_, tags));
    } else {
#endif
        // Update problem alignments to match catalog entry.
        // Do not raise an alignment above the actual buffer alignment: catalog
        // entries may accept lower alignments than driverInfo.alignment via
        // unaligned access fallbacks, and overstating the alignment can cause
        // illegal accesses (e.g. 2D block loads on a sub-DWord pitch).
        if (!isPacked(problem_.A.layout)
                && problem_.Ta_ext.paddedSize() >= problem_.Ta.paddedSize()) {
            problem_.A.setAlignment(std::max(problem_.Ta_ext.paddedSize(),
                    std::min(entry_->driverInfo.alignment[0],
                            int(problem_.A.alignment))));
        }

        if (!isPacked(problem_.B.layout)
                && problem_.Tb_ext.paddedSize() >= problem_.Tb.paddedSize()) {
            problem_.B.setAlignment(std::max(problem_.Tb_ext.paddedSize(),
                    std::min(entry_->driverInfo.alignment[1],
                            int(problem_.B.alignment))));
        }

        if (!isPacked(problem_.C.layout)) {
            problem_.C.setAlignment(std::max(problem_.Tc_ext.paddedSize(),
                    entry_->restrictions.alignment[2]));
        }

        // Parse strategy string.
        strategy_ = GEMMStrategy(hw_, stepping_);
        strategy_.unroll[LoopM] = entry_->driverInfo.unroll[LoopM];
        strategy_.unroll[LoopN] = entry_->driverInfo.unroll[LoopN];
        parseStrategy(entry_->strategy, problem_, strategy_);
#ifdef DNNL_DEV_MODE
    }
#endif
    modifyStrategy(strategy_, aux_params_);
    if (l1_flush_wa_)
        strategy_.C.cachingW = makeL1Uncacheable(strategy_.C.cachingW);
    strategy_.panelCheck
            |= (isPacked(problem_.A.layout) || isPacked(problem_.B.layout));

    if (enable_generator_dsl()) { fixup_dsl_strategy(strategy_); }

    // Align k slice size and quantization group size
    if (strategy_.kParallelLocal) {
        if (problem_.quantized2DA())
            aux_params_.k0 = utils::rnd_up(aux_params_.k0, problem_.aqGroupK);
        if (problem_.quantized2DB())
            aux_params_.k0 = utils::rnd_up(aux_params_.k0, problem_.bqGroupK);
    }

    if (hw_ == ngen::HW::Xe2 || hw_ == ngen::HW::Xe3) {
        // Use XeHPC register banking on Xe2/Xe3, in order
        // to successfully reuse XeHPC strategies.
        if (strategy_.raHW == hw_) strategy_.raHW = ngen::HW::XeHPC;

        // Bump up alignments to 16 bytes for block 2D if available.
        bool block_2d_a = false, block_2d_b = false;
        for (auto c = tags; *c; c++) {
            block_2d_a |= (*c == kcatalog::ReqBlock2DA);
            block_2d_b |= (*c == kcatalog::ReqBlock2DB);
        }
        if (block_2d_a && strategy_.legalAAlignment(problem_, 16))
            problem_.A.setAlignment(nstl::max<int>(problem_.A.alignment, 16));
        if (block_2d_b && strategy_.legalBAlignment(problem_, 16))
            problem_.B.setAlignment(nstl::max<int>(problem_.B.alignment, 16));
    }

    if (hw_ == ngen::HW::Xe3p) {
        // Use XeHPC banking if reusing XeHPC strategies (legacy mode)
        if (!efficient_64b_ && strategy_.raHW == hw_)
            strategy_.raHW = ngen::HW::XeHPC;

        // Disable named barriers to avoid simulator errors, allow fallback to pvc strategies.
        strategy_.namedBarriers[0] = 0;
        strategy_.namedBarriers[1] = 0;
    }

    // Disable global k parallelization if it wouldn't be used.
    if (strategy_.kParallel && k_ >= 0) {
        auto k_min = aux_params_.k0 * aux_params_.wgK;
        if (k_ <= k_min) {
            strategy_.kParallel = false;
            strategy_.C.atomic = false;
            strategy_.CO.atomic = false;
        }
    }

    // Always use variable beta for global k-parallel kernels.
    if (strategy_.kParallel && !strategy_.fuseBeta) problem_.beta = Scalar();

    // Omit periodic barriers when k is small.
    if (strategy_.barrierFreq > 0 && k_ >= 0 && k_ < 2 * strategy_.barrierFreq)
        strategy_.barrierFreq = 0;

    // Correct GRF count in following calculations for fixed systolic kernels.
    if (strategy_.fixedSystolic) strategy_.GRFs = 256;

    // Disable linear ordering and persistent threads if the GEMM doesn't fill the GPU.
    if (m_ >= 0 && n_ >= 0 && eu_count_ >= 0) {
        int wg_tile_m = strategy_.wg[LoopM] * strategy_.unroll[LoopM];
        int wg_tile_n = strategy_.wg[LoopN] * strategy_.unroll[LoopN];
        if (wg_tile_m > 0 && wg_tile_n > 0) {
            dim_t m_tiles = dim_t(utils::div_up(m_, wg_tile_m));
            dim_t n_tiles = dim_t(utils::div_up(n_, wg_tile_n));
            dim_t thread_per_tg = strategy_.wg[LoopM] * strategy_.wg[LoopN];
            if (!strategy_.kParallelVariable)
                thread_per_tg *= std::max(strategy_.wg[LoopK], 1);
            dim_t thread_gpu = eu_count_
                    * compute::device_info_t::threads_per_eu(
                            product_, strategy_.GRFs);
            dim_t tiles_gpu = thread_gpu / thread_per_tg;

            bool use_linear = (m_tiles * n_tiles <= tiles_gpu);
            bool use_linear_m = (m_tiles * m_tiles <= 2 * tiles_gpu);
            bool use_linear_n = (n_tiles * n_tiles <= 2 * tiles_gpu);

            if (strategy_.fused)
                if (strategy_.wg[LoopM] % 2 || strategy_.wg[LoopN] % 2)
                    use_linear_m = use_linear_n = false; /* cannot swap */

            if (use_linear) {
                if (strategy_.kParallelVariable)
                    strategy_.cWalkOrder = WalkOrder::SimpleLinear;
                else if (strategy_.kParallel
                        && (strategy_.fuseBeta || strategy_.fusePostOps)) {
                    strategy_.persistent = false;
                    strategy_.cWalkOrder = WalkOrder::SimpleLinear;
                } else {
                    strategy_.persistent = false;
                    strategy_.cWalkOrder = WalkOrder::HW2D;
                    strategy_.blocking[LoopM] = 16777216;
                    strategy_.blocking[LoopN] = 16777216;
                }
            } else if (use_linear_m || use_linear_n) {
                if (use_linear_n && !use_linear_m) {
                    strategy_.loopOrder[0] = LoopN;
                    strategy_.loopOrder[1] = LoopM;
                } else if (use_linear_m && !use_linear_n) {
                    strategy_.loopOrder[0] = LoopM;
                    strategy_.loopOrder[1] = LoopN;
                }
                strategy_.cWalkOrder = WalkOrder::SimpleLinear;
            }
        }
    }

    strategy_.relaxedAccumulation |= relaxed_acc_;
    strategy_.systolicAvailable &= !disable_systolic_;
    problem_.autoTypeConversions(strategy_.systolicAvailable);
    adjustStrategy(hw_, problem_, strategy_, tags);
    try {
        strategy_.preflight(hw_, problem_);
    } catch (const std::exception &e) { return reject(e.what()); } catch (...) {
        return reject("preflight failed");
    }

    // Override: fixed beta would be reapplied on later k-chunks.
    if (!entry_ && k_ > strategy_.blocking[LoopK] && problem_.beta.fixed()
            && !problem_.beta1())
        return reject("fixed beta over k-chunks");

    // Check for legal 2D quantization group size.
    if (problem_.aOffset2D() || problem_.aScale2D())
        if (problem_.aqGroupK % strategy_.aqGroupKGranularity())
            return reject("A quantization group");
    if (problem_.bOffset2D() || problem_.bScale2D())
        if (problem_.bqGroupK % strategy_.bqGroupKGranularity())
            return reject("B quantization group");
    if (problem_.aScale2D()
            && problem_.aqGroupK % minOuterProductCount(problem_, strategy_)
                    != 0) {
        if (!problem_.Ta.isF4() || !problem_.Tb.isF4())
            return reject("quantization group vs outer product");
    }
    if (problem_.bScale2D()
            && problem_.bqGroupK % minOuterProductCount(problem_, strategy_)
                    != 0) {
        if (!problem_.Ta.isF4() || !problem_.Tb.isF4())
            return reject("quantization group vs outer product");
    }

    // If the M/N group size is equal to M or N, align up to a multiple of unroll size
    // XXX: Increase group size to a large value before aligning to increase reusability
    // TODO: Refactor M/N groups/thread setting to preserve MN group count.
    constexpr int perMNGroupSize = 1 << 24;
    if (problem_.aqGroupM == m_
            && ((!problem_.hasGroupSumsA && !problem_.preferBDPAS())
                    || m_ > 1)) {
        problem_.aqGroupM = std::max(problem_.aqGroupM, perMNGroupSize);
        problem_.aqGroupM
                = utils::rnd_up(problem_.aqGroupM, strategy_.unroll[LoopM]);
    }
    if (problem_.bqGroupN == n_
            && ((!problem_.hasGroupSumsB && !problem_.preferBDPAS())
                    || n_ > 1)) {
        problem_.bqGroupN = std::max(problem_.bqGroupN, perMNGroupSize);
        problem_.bqGroupN
                = utils::rnd_up(problem_.bqGroupN, strategy_.unroll[LoopN]);
    }

    strategy_.kInterleaveChunk
            = std::min(strategy_.kInterleaveChunk, (int)aux_params_.k0);
    if (strategy_.kInterleave) aux_params_.wgK = strategy_.wg[LoopK];
    if (aux_params_.wgK > strategy_.wg[LoopK])
        aux_params_.wgK = strategy_.wg[LoopK];
    update_driver_info();

    return status::success;
}

void gen_desc_t::update_driver_info() {
#define ARCH_DISPATCH(arch) \
    case ngen::HW::arch: \
        driver_info_ = gemm_kernel_generator_t<ngen::HW::arch>::driverInfo( \
                problem_, strategy_); \
        break;

    switch (hw_) {
        REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
        REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
        REG_XE2_ISA(ARCH_DISPATCH(Xe2))
        REG_XE3_ISA(ARCH_DISPATCH(Xe3))
        REG_XE3P_ISA(ARCH_DISPATCH(Xe3p))
        default:
            assert(!"Unsupported architecture");
            driver_info_ = entry_->driverInfo;
            break;
    }
#undef ARCH_DISPATCH
}

std::vector<const gemmstone::kcatalog::Entry *>
gen_nocopy_desc_t::select_kernel(const compute::device_info_t &dev_info,
        compute_mode mode, const gemmstone::GEMMProblem &problem, float alpha,
        float beta, dim_t m, dim_t n, dim_t k, dim_t lda, dim_t ldb, dim_t ldc,
        dim_t batch) {
    using namespace ngen;
    using namespace kcatalog;

    product_ = dev_info.product();
    hw_ = getCore(product_.family);
    arch_ = convert_ngen_arch_to_dnnl(hw_);
    stepping_ = dev_info.stepping_id();
    problem_.product = product_;
    l1_flush_wa_ = dev_info.has_l1_flush_bug();
    m_ = into<int>(m);
    n_ = into<int>(n);
    k_ = into<int>(k);
    eu_count_ = dev_info.eu_count();
    disable_systolic_ = !dev_info.mayiuse_systolic();
    relaxed_acc_ = mode & mode_relaxed_acc;

    // Select a kernel from the catalog.
    std::vector<MatchParams> match_params;
    MatchParams base(hw_, dev_info.mayiuse_systolic(), product_, problem);
    /* Reuse PVC strategies for legacy mode on Xe3p */
    if (hw_ == ngen::HW::Xe3p && !efficient_64b_)
        base.selector.hw = kcatalog::HWTagXeHPC;

    // By default gemmstone assumes that the accumulation type must be at least
    // as wide as the output type. For oneDNN this restriction is not needed.
    base.precisionCExt = '\0';

    base.sizes.m = m;
    base.sizes.n = n;
    base.sizes.k = k;
    base.sizes.batch = batch;
    base.stepping = dev_info.stepping_id();
    base.ignoreCase = true;

    bool can_2d_a = (lda * problem.Ta_ext <= 16777216);
    bool can_2d_b = (ldb * problem.Tb_ext <= 16777216);
    bool can_2d_c = (ldc * problem.Tc_ext <= 16777216);

    // Xe2 requires stronger alignment for block 2D.
    if (arch_ == compute::gpu_arch_t::xe2
            || arch_ == compute::gpu_arch_t::xe3) {
        can_2d_a &= (problem.A.alignment % 16 == 0);
        can_2d_b &= (problem.B.alignment % 16 == 0);
        can_2d_c &= (problem.C.alignment % 16 == 0);
    }

    auto tags = const_cast<char *>(base.tags);
    while (*tags)
        tags++;
    if (problem.A.needA64 || problem.B.needA64 || problem.C.needA64)
        *tags++ = kcatalog::ReqBatchN;
    if (can_2d_a) *tags++ = kcatalog::ReqBlock2DA;
    if (can_2d_b) *tags++ = kcatalog::ReqBlock2DB;
    if (can_2d_c) *tags++ = kcatalog::ReqBlock2DC;

    // Modify base params, used for mandatory conversion.
    auto mod_match = [&](MatchParams &params, bool has_mode,
                             const char *(*match)(Type)) {
        if (!has_mode) return;
        if (match(problem.Ta)) {
            params.selector.precisions[0] = match(problem.Ta);
        }
        if (match(problem.Tb)) {
            params.selector.precisions[1] = match(problem.Tb);
        }
    };

    // Workaround limited attribute support with int8 dynamic quant,
    // upconvert to f16.
    mod_match(base,
            (((problem.asPtrDims >= 2 || problem.bsPtrDims >= 2)
                     || problem.boPtrDims > -1)
                    && problem.aoPtrDims > -1 && problem.Ta_ext.isInt8()
                    && problem.Tb_ext.isInt8() && problem.Tc.isFP()
                    && !problem.hasGroupSumsA && !problem.hasGroupSumsB),
            [](Type dt) -> const char * {
        if (dt.isInt8()) return "[OH]";
        return nullptr;
    });

    match_params.push_back(base);

    bool fpmath_tf32 = mode & mode_tf32;
    bool fpmath_bf16 = mode & mode_bf16x1;
    bool fpmath_f16 = mode & mode_f16x1;

    auto add_matches = [&](MatchParams start, const char *(*match)(Type)) {
        if (match(problem.Ta_ext)) {
            match_params.push_back(start);
            match_params.back().selector.precisions[0] = match(problem.Ta_ext);
        }
        if (match(problem.Tb_ext)) {
            match_params.push_back(start);
            match_params.back().selector.precisions[1] = match(problem.Tb_ext);
        }
        if (match(problem.Ta_ext) && match(problem.Tb_ext)) {
            match_params.push_back(start);
            match_params.back().selector.precisions[0] = match(problem.Ta_ext);
            match_params.back().selector.precisions[1] = match(problem.Tb_ext);
        }
    };

    auto add_mode_matches = [&](bool has_mode, const char *(*match)(Type)) {
        if (!has_mode) return;
        add_matches(base, match);
    };

    add_mode_matches(fpmath_tf32, [](Type dt) -> const char * {
        if (dt == Type::f32) { return "T"; }
        return nullptr;
    });

    add_mode_matches(fpmath_bf16, [](Type dt) -> const char * {
        if (dt == Type::f32) { return "[SB]"; }
        if (dt.isInt8() || dt.isSubByteInt()) return "[OB]";
        if (dt.isF4()) return "F";
        return nullptr;
    });

    add_mode_matches(fpmath_f16, [](Type dt) -> const char * {
        if (dt == Type::f32) { return "[SH]"; }
        if (dt.isInt8() || dt.isSubByteInt()) return "[OH]";
        if (dt.isF4()) return "F";
        return nullptr;
    });

    add_mode_matches(!(fpmath_f16 || fpmath_bf16), [](Type dt) -> const char * {
        if (dt.bits() == 2) return "[PO]";
        if (dt.bits() == 4) return "[FO]";
        if (dt.is3()) return "[FO]";
        return nullptr;
    });

    // Add allowed variants of each valid kernel.
    // Should be used after all valid kernels are added to match_params
    auto add_variants = [&](const char *(*match)(Type)) {
        size_t npatterns = match_params.size();
        for (size_t i = 0; i < npatterns; i++) {
            add_matches(match_params[i], match);
        }
    };

    add_variants([](Type dt) -> const char * {
        // fp16 -> bf16
        if (dt == Type::bf16) return "H";
        return nullptr;
    });

    // Allow cases with integer acc to reuse strategies with equal size float acc.
    // Prioritize float acc strategies as theyre better optimized.
    if (problem.Tc == Type::s32) {
        size_t npatterns = match_params.size();
        std::vector<MatchParams> float_strats;
        for (size_t i = 0; i < npatterns; ++i) {
            auto start = match_params[i];
            if (!std::string("I").compare(start.selector.precisions[2])) {
                float_strats.push_back(start);
                float_strats.back().selector.precisions[2] = "S";
            }
        }
        match_params.insert(
                match_params.begin(), float_strats.begin(), float_strats.end());
    }

    eval_params_.sizes = base.sizes;
    eval_params_.alpha = alpha;
    eval_params_.beta = beta;
    eval_params_.product = product_;
    eval_params_.postOps = !problem.postOps.empty();
    eval_params_.cConvert = (problem.Tc != problem.Tc_ext);
    eval_params_.euCount = dev_info.eu_count();
    eval_params_.batch = (problem.batchDims > 0);
    eval_params_.deterministic = (mode & mode_deterministic);
    eval_params_.Tc_ext = problem.Tc_ext;

    SelectionObserver observer = make_entry_observer(
            dev_info.eu_count(), dev_info.max_clock_mhz());
    tags_ = match_params[0].tags;
    Ts_ = problem.Ts;
    beta_ = problem.beta;
    return select(catalog(), static_cast<int>(match_params.size()),
            match_params.data(), eval_params_, aux_params_, &observer);
}

status_t gen_nocopy_desc_t::finalize() {
    // Update A/B/C types from entry.
    Type Ta_new, Ta_ext_new, Tb_new, Tb_ext_new, Tc_new;
    parsePrecisions(entry_->selector.precisions[0], Ta_ext_new, Ta_new);
    parsePrecisions(entry_->selector.precisions[1], Tb_ext_new, Tb_new);
    Tc_new = charToType(entry_->selector.precisions[2][0]);

    auto update_type = [](Type &T, Type T_new) {
        if (T.isF8() && T_new.isF8()) return;
        if (T.isFP() && T.bits() == 16 && T_new.isFP() && T_new.bits() == 16)
            return;
        if (T.isF4() && (T_new.bits() == 4)) return;
        T = T.isSigned() ? T_new.asSigned() : T_new.asUnsigned();
    };
    update_type(problem_.Ta, Ta_new);
    update_type(problem_.Tb, Tb_new);
    if (!(problem_.Tc == Type::s32 && Tc_new == Type::f32))
        update_type(problem_.Tc, Tc_new);

    // If the kernel uses tf32 types, interpret the buffer as tf32 without
    // converting from fp32. This eliminates a rounding step, but improves performance
    auto use_tf32 = [](Type &T, const Type &newT) {
        if (newT == Type::tf32) {
            gpu_assert(T == Type::f32)
                    << "Unexpected use of tf32 for non-fp32 type";
            T = Type::tf32;
        }
    };
    use_tf32(problem_.Ta_ext, Ta_ext_new);
    use_tf32(problem_.Tb_ext, Tb_ext_new);
    problem_.Ts = Ts_;

    if (problem_.Ts == Type::invalid) problem_.Ts = problem_.Tc;

    auto block_k = entry_->driverInfo.blocking[LoopK];
    problem_.beta = beta_;
    // An override applies its own k blocking.
    if (kernel_override_.empty() && block_k > 0 && k_ > block_k
            && eval_params_.beta != 1.0f)
        problem_.beta = Scalar();
    evaluate(*entry_, eval_params_, aux_params_);
    return gen_desc_t::finalize(tags_.c_str());
}

status_t gen_xe_systolic_kernel_desc_t::select_kernel(
        const compute::device_info_t &dev_info, int batch_dims, bool packed_c,
        bool trans_co, bool a_offset, bool b_offset, bool c_offset, bool bias,
        float alpha, float beta, data_type_t a_type, data_type_t b_type,
        data_type_t c_type, data_type_t ao_type, data_type_t bo_type,
        data_type_t co_type, data_type_t acc_type, dim_t m, dim_t n, dim_t k,
        dim_t batch, int unroll_m, int unroll_n, bool alt,
        gpu_post_ops_t &&post_ops) {
    using namespace ngen;
    using namespace kcatalog;

    product_ = dev_info.product();
    hw_ = getCore(product_.family);
    arch_ = convert_ngen_arch_to_dnnl(hw_);
    stepping_ = dev_info.stepping_id();
    problem_.product = product_;
    l1_flush_wa_ = dev_info.has_l1_flush_bug();
    m_ = into<int>(m);
    n_ = into<int>(n);
    k_ = into<int>(k);
    eu_count_ = dev_info.eu_count();

    if (!utils::one_of(hw_, HW::XeHP, HW::XeHPG, HW::XeHPC, HW::Xe2, HW::Xe3,
                HW::Xe3p))
        return status::unimplemented;

    bool xehpc = (hw_ >= HW::XeHPC);

    auto osys = xehpc ? 16 : 8;
    auto ksys = int(32 / types::data_type_size(a_type));
    auto csys = int(4 / types::data_type_size(a_type));

    problem_.Ta = problem_.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem_.Tb = problem_.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem_.Ts = Type::f32;
    problem_.Tao = convert_dnnl_to_kernel_type(ao_type);
    problem_.Tbo = convert_dnnl_to_kernel_type(bo_type);
    problem_.Tco = convert_dnnl_to_kernel_type(co_type);
    problem_.A.layout = MatrixLayout::PackedColumns;
    problem_.B.layout = MatrixLayout::PackedRows;
    problem_.C.layout = MatrixLayout::N;
    problem_.A.crosspack = csys;
    problem_.B.crosspack = ksys;
    problem_.C.crosspack = 1;
    problem_.A.packSize = unroll_m;
    problem_.B.packSize = unroll_n;
    problem_.C.packSize = 0;
    if (osys < unroll_m) {
        problem_.A.tileR = osys;
        problem_.A.tileC = ksys;
    }
    problem_.A.setAlignment(32);
    problem_.B.setAlignment(32);
    problem_.C.setAlignment(int(types::data_type_size(c_type)));
    if (packed_c) problem_.C = problem_.B;
    if (batch_dims > 0) {
        problem_.batch = BatchMode::Strided;
        problem_.batchDims = batch_dims;
    }
    if (a_offset) {
        problem_.aOffset = ABOffset::Load;
        problem_.aoPtrDims = 0;
    }
    if (b_offset) {
        problem_.bOffset = ABOffset::Load;
        problem_.boPtrDims = 0;
    }
    if (alpha == 1.0f) problem_.alpha = (int)alpha;
    if (beta == 0.0f || beta == 1.0f) problem_.beta = (int)beta;

    auto status = transfer_post_ops(problem_, std::move(post_ops));
    if (status != status::success) return status;

    if (c_offset) problem_.cOffset = COffset::Post;

    if (bias) {
        if (problem_.cOffset != COffset::None) return status::unimplemented;
        problem_.cOffset = COffset::Pre;
        problem_.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
    }

    if (problem_.cOffset != COffset::None) {
        problem_.CO.crosspack = 1;
        problem_.CO.alignment = problem_.C.alignment;
    }

    // Find it in the catalog.
    MatchParams match_params(hw_, true, product_, problem_);

    // By default gemmstone assumes that the accumulation type must be at least
    // as wide as the output type. For oneDNN this restriction is not needed.
    match_params.precisionCExt = '\0';

    match_params.sizes.m = m;
    match_params.sizes.n = n;
    match_params.sizes.k = k;
    match_params.sizes.batch = batch;

    StrategyRequirement reqs[2] = {StrategyRequirement::UnrollM == unroll_m,
            StrategyRequirement::UnrollN == unroll_n};
    match_params.extraReqs = reqs;
    match_params.nExtraReqs = 2;

    auto tags = const_cast<char *>(match_params.tags);
    while (*tags)
        tags++;

    *tags++ = kcatalog::ReqSystolic;
    if (alt) *tags++ = kcatalog::ReqCustom1;

    EvaluateParams eval_params;

    eval_params.sizes = match_params.sizes;
    eval_params.alpha = alpha;
    eval_params.beta = beta;
    eval_params.product = product_;
    eval_params.euCount = dev_info.eu_count();
    eval_params.postOps = !problem_.postOps.empty();
    eval_params.cConvert = (acc_type != c_type);
    eval_params.batch = (batch_dims > 0);
    eval_params.Tc_ext = problem_.Tc_ext;

    SelectionObserver observer = make_entry_observer(
            dev_info.eu_count(), dev_info.max_clock_mhz());

    auto entries = select(
            catalog(), match_params, eval_params, aux_params_, &observer);

    if (entries.size() < 1) return status::unimplemented;
    entry_ = entries[0];
    return finalize(match_params.tags);
}

void gen_xe_systolic_kernel_desc_t::choose_unrolls(compute::gpu_arch_t arch,
        int eu_count, data_type_t a_type, data_type_t b_type,
        data_type_t c_type, dim_t m, dim_t n, dim_t k, dim_t batch,
        int &unroll_m, int &unroll_n, bool &alt) {

    using namespace data_type;

    alt = false;

    switch (arch) {
        case compute::gpu_arch_t::xe_hp:
        case compute::gpu_arch_t::xe_hpg:
            if (unroll_m == 0) unroll_m = 32;
            if (unroll_n == 0) unroll_n = (m * n >= 6144 * eu_count) ? 48 : 32;

            if (unroll_n == 48) alt = (m * n >= 13824 * eu_count);
            break;
        case compute::gpu_arch_t::xe_hpc:
        case compute::gpu_arch_t::xe2:
        case compute::gpu_arch_t::xe3:
        case compute::gpu_arch_t::xe3p:
            if (utils::one_of(a_type, f16, bf16)) {
                if (unroll_m != 0)
                    unroll_n = (unroll_m > 16) ? 32 : 16;
                else if (unroll_n != 0)
                    unroll_m = (unroll_n > 16) ? 64 : 16;
                else if (m * n < 4096 * eu_count)
                    unroll_m = unroll_n = 16;
                else {
                    unroll_m = 64;
                    unroll_n = 32;
                }
            } else {
                unroll_m = 64;
                unroll_n = 32;
            }
            break;
        default: assert(!"Unsupported architecture.");
    }
}

void gen_kernel_t::init_interface() {
    using namespace ngen;

    auto &problem = *desc()->problem();
    auto &strategy = *desc()->strategy();

    interface_ = NEOInterfaceHandler {desc()->hw_};
    auto s_type_ngen = problem.Ts.ngen();

    auto a_access = strategy.A.getGlobalAccessType();
    auto b_access = strategy.B.getGlobalAccessType();
    auto c_access = strategy.C.getGlobalAccessType();
    auto ao_access = strategy.AO.getGlobalAccessType();
    auto bo_access = strategy.BO.getGlobalAccessType();
    auto co_access = strategy.CO.getGlobalAccessType();
    auto as_access = strategy.A_scale.getGlobalAccessType();
    auto bs_access = strategy.B_scale.getGlobalAccessType();
    auto ag_access = strategy.Ag.getGlobalAccessType();
    auto bg_access = strategy.Bg.getGlobalAccessType();

    interface_.newArgument("A", ExternalArgumentType::GlobalPtr, a_access);
    interface_.newArgument("B", ExternalArgumentType::GlobalPtr, b_access);
    interface_.newArgument("C", ExternalArgumentType::GlobalPtr, c_access);
    interface_.newArgument("offset_A", DataType::q);
    interface_.newArgument("offset_B", DataType::q);
    interface_.newArgument("offset_C", DataType::q);
    interface_.newArgument("lda", DataType::d);
    interface_.newArgument("ldb", DataType::d);
    interface_.newArgument("ldc", DataType::d);
    interface_.newArgument("m", DataType::d);
    interface_.newArgument("n", DataType::d);
    interface_.newArgument("k", DataType::d);
    interface_.newArgument("alpha_real", s_type_ngen);
    interface_.newArgument("beta_real", s_type_ngen);
    if (problem.aoPtrDims >= 0)
        interface_.newArgument(
                "ao_ptr", ExternalArgumentType::GlobalPtr, ao_access);
    if (problem.boPtrDims >= 0)
        interface_.newArgument(
                "bo_ptr", ExternalArgumentType::GlobalPtr, bo_access);
    if (problem.aOffsetHostScalar()) interface_.newArgument("ao", DataType::w);
    if (problem.bOffsetHostScalar()) interface_.newArgument("bo", DataType::w);
    if (problem.aScale2D())
        interface_.newArgument(
                "a_scale_ptr", ExternalArgumentType::GlobalPtr, as_access);
    if (problem.bScale2D())
        interface_.newArgument(
                "b_scale_ptr", ExternalArgumentType::GlobalPtr, bs_access);
    if (problem.hasCMXScale())
        interface_.newArgument(
                "c_scale_ptr", ExternalArgumentType::GlobalPtr, c_access);
    if (problem.needsAGroupSums())
        interface_.newArgument(
                "ag_ptr", ExternalArgumentType::GlobalPtr, ag_access);
    if (problem.needsBGroupSums())
        interface_.newArgument(
                "bg_ptr", ExternalArgumentType::GlobalPtr, bg_access);
    if (problem.aOffset2D() || problem.aScale2D()
            || problem.needsAGroupSums()) {
        interface_.newArgument("ldaq", DataType::d);
    }
    if (problem.bOffset2D() || problem.bScale2D()
            || problem.needsBGroupSums()) {
        interface_.newArgument("ldbq", DataType::d);
    }

    if (problem.hasCMXScale()) interface_.newArgument("ldcq", DataType::d);
    if (problem.usesCOPtr()) {
        interface_.newArgument(
                "co_ptr", ExternalArgumentType::GlobalPtr, co_access);
        interface_.newArgument("offset_CO", DataType::q);
        if (problem.cOffset == COffset::Pre)
            interface_.newArgument("ldco", DataType::d);
    } else if (problem.cOffsetHostScalar()) {
        interface_.newArgument("co", DataType::w);
    }
    if (problem.postOps.cStochasticRound) {
        interface_.newArgument("sround_seed", ExternalArgumentType::GlobalPtr);
    }

    if (strategy.needsTempC(problem))
        interface_.newArgument(
                "temp_C", ExternalArgumentType::GlobalPtr, c_access);
    interface_.newArgument("flags", DataType::ud);
    if ((strategy.kParallel || strategy.kParallelLocal)
            && !strategy.kParallelVariable)
        interface_.newArgument("k0", DataType::d);
    for (size_t i = 0; i < problem.postOps.len(); i++) {
        if (!problem.postOps[i].is_binary()) continue;
        auto bname = "binary" + std::to_string(i);
        interface_.newArgument(bname, ExternalArgumentType::GlobalPtr,
                strategy.binary[i].getGlobalAccessType());
        interface_.newArgument("offset_" + bname, DataType::q);
        if (problem.postOps.binaryRow[i] && problem.postOps.binaryCol[i])
            interface_.newArgument("ld" + bname, DataType::d);
    }
    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            interface_.newArgument("stride_A" + std::to_string(i), DataType::q);
            interface_.newArgument("stride_B" + std::to_string(i), DataType::q);
            interface_.newArgument("stride_C" + std::to_string(i), DataType::q);
            if (problem.hasAScalePtr()) {
                interface_.newArgument(
                        "scale_stride_A" + std::to_string(i), DataType::d);
            }
            if (problem.hasBScalePtr()) {
                interface_.newArgument(
                        "scale_stride_B" + std::to_string(i), DataType::d);
            }
            if (problem.hasCMXScale()) {
                interface_.newArgument(
                        "scale_stride_C" + std::to_string(i), DataType::q);
            }
            if (problem.hasAOffsetPtr()) {
                interface_.newArgument(
                        "offset_stride_A" + std::to_string(i), DataType::d);
            }
            if (problem.hasBOffsetPtr()) {
                interface_.newArgument(
                        "offset_stride_B" + std::to_string(i), DataType::d);
            }
            if (problem.needsAGroupSums()) {
                interface_.newArgument(
                        "group_sums_stride_A" + std::to_string(i), DataType::d);
            }
            if (problem.needsBGroupSums()) {
                interface_.newArgument(
                        "group_sums_stride_B" + std::to_string(i), DataType::d);
            }
        }
        for (size_t i = 0; i < problem.postOps.len(); i++) {
            if (problem.postOps[i].is_binary()
                    && problem.postOps.binaryBatch[i]) {
                for (int b = 0; b < problem.batchDims; b++) {
                    interface_.newArgument("stride" + std::to_string(b)
                                    + "binary" + std::to_string(i),
                            DataType::q);
                }
            }
        }
        for (int i = 0; i < problem.batchDims - 1; i++) {
            interface_.newArgument(
                    "batch_size" + std::to_string(i), DataType::ud);
            if (enable_generator_dsl()) {
                interface_.newArgument(
                        "batch_magic" + std::to_string(i), DataType::uq);
            } else {
                interface_.newArgument(
                        "recip_batch_size" + std::to_string(i), DataType::ud);
            }
        }
    }
    if (strategy.fuseBeta || strategy.fusePostOps)
        interface_.newArgument("status", ExternalArgumentType::GlobalPtr,
                GlobalAccessType::Stateless);
    if (strategy.fuseBeta && strategy.kParallel)
        interface_.newArgument("group_count_k", DataType::ud);
    if (strategy.linearOrder()) {
        interface_.newArgument("group_count_m", DataType::ud);
        interface_.newArgument("group_count_n", DataType::ud);
    }
    if (strategy.cWalkOrder == WalkOrder::SimpleLinear)
        interface_.newArgument("group_count_recip", DataType::ud);
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        interface_.newArgument("hilbert_vd", DataType::ud);
        interface_.newArgument("hilbert_uvd_recip", DataType::ud);
        interface_.newArgument("hilbert_bail", DataType::ud);
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        interface_.newArgument("bslice", DataType::d);
        interface_.newArgument("bthresh", DataType::d);
    }
    if (strategy.kParallelVariable) {
        interface_.newArgument("k0", DataType::ud);
        interface_.newArgument("kv_config", DataType::ud);
        interface_.newArgument("k_recip", DataType::ud);
    }
    if (strategy.persistent)
        interface_.newArgument("group_stride", DataType::ud);
    if (strategy.variableSLM())
        interface_.newArgument("local_mem", ExternalArgumentType::LocalPtr);
    if (problem.aoPtrDims >= 1)
        interface_.newArgument("offset_AO", DataType::q);
    if (problem.boPtrDims >= 1)
        interface_.newArgument("offset_BO", DataType::q);
    if (problem.aScale2D())
        interface_.newArgument("offset_A_scale", DataType::q);
    if (problem.bScale2D())
        interface_.newArgument("offset_B_scale", DataType::q);
    if (problem.needsAGroupSums())
        interface_.newArgument("offset_Ag", DataType::q);
    if (problem.needsBGroupSums())
        interface_.newArgument("offset_Bg", DataType::q);

    if (desc()->hw_ >= HW::XeHPG) interface_.allowArgumentRearrangement(false);
    interface_.externalName(kernel_name());
    interface_.setEfficient64Bit(desc_.efficient_64b_);
}

dsl::kernel_t get_dsl_kernel(const GEMMProblem &problem,
        const GEMMStrategy &strategy, const ngen::InterfaceHandler &iface,
        const dsl::hw_t &hw, int m, int n, int k) {
    auto gemm_desc
            = gemmstone::generator_dsl_desc_t(problem, strategy, iface, hw);
    if (gpu_utils::dev_getenv("generator_dsl_specialize", false)) {
        auto &opt = gemm_desc.options;
        if (n != -1) opt.assume(gemm_desc.kernel_iface().find_arg("m") == m);
        if (m != -1) opt.assume(gemm_desc.kernel_iface().find_arg("n") == n);
        if (k != -1) opt.assume(gemm_desc.kernel_iface().find_arg("k") == k);
    }
    return make_kernel(gemm_desc);
}

std::string dump_kernel(ngen::HW hw, const gemmstone::GEMMProblem &problem,
        const gemmstone::GEMMStrategy &strategy) {
    auto pstr = problem.toString();
    auto astr = problem.scalarsToString();
    auto sstr = unparseStrategy(hw, problem, strategy);
    if (!astr.empty()) astr += ' ';
    return pstr + ' ' + std::to_string(strategy.unroll[LoopM]) + ' '
            + std::to_string(strategy.unroll[LoopN]) + ' ' + astr + sstr;
}

status_t gen_kernel_t::get_kernel(
        compute::kernel_t &kernel, const intel::engine_t *engine) {
    init_interface();
    maybe_print_verbose();

    if (enable_generator_dsl()) {
        auto k = get_dsl_kernel(*desc()->problem(), *desc()->strategy(),
                interface_, make_ir_hw(engine), desc()->m_, desc()->n_,
                desc()->k_);
        if (k.body.is_empty()) return status::runtime_error;
        return engine->create_kernel(kernel, k);
    }

#define ARCH_DISPATCH(arch) \
    case ngen::HW::arch: { \
        gemm_kernel_generator_t<ngen::HW::arch> generator(desc()->product_); \
        generator.setStepping(desc()->stepping_); \
        generator.gemm(*desc()->problem(), *desc()->strategy(), interface_); \
        return generator.get_kernel(kernel, engine); \
        break; \
    }

    try {
        switch (desc()->hw_) {
            REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
            REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
            REG_XE2_ISA(ARCH_DISPATCH(Xe2))
            REG_XE3_ISA(ARCH_DISPATCH(Xe3))
            REG_XE3P_ISA(ARCH_DISPATCH(Xe3p))
            default: assert(!"Unsupported architecture"); break;
        }
    } catch (const std::runtime_error &err) {
        auto str
                = dump_kernel(desc()->hw_, desc()->problem_, desc()->strategy_);
        if (!desc()->has_entry())
            VINFO(primitive, create, dispatch, gemm, "%s,%s,%s", "jit::gemm",
                    err.what(), str.c_str());
        else // Print kernel generation errors only in debug mode
            VDEBUGINFO(1, primitive, gpu, "%s,%s,%s", "jit::gemm", err.what(),
                    str.c_str());
    }
#undef ARCH_DISPATCH

    return status::runtime_error;
}

std::string gen_desc_t::str() const {
    return dump_kernel(hw_, problem_, strategy_);
}

void gen_kernel_t::maybe_print_verbose() {
    gpu_debug() << "kernel:"
                << dump_kernel(
                           desc()->hw_, desc()->problem_, desc()->strategy_);
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
