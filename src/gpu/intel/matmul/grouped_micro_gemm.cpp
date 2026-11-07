/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/matmul/grouped_micro_gemm.hpp"
#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/host_scalars.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

namespace {

// grouped_micro_gemm cross-thread argument bytes, plus headroom.
constexpr int host_argument_bytes = 268;

// Groups the sparse (gemv) dispatch searches per pass for a token's group.
// Fixing it keeps the group count out of the kernel cache key.
constexpr int sparse_groups_per_pass = 128;

gemmstone::microkernel::HWInformation get_hw_info(
        const compute::device_info_t &dev_info) {
    gemmstone::microkernel::HWInformation hw_info;
    hw_info.euCount = dev_info.eu_count();
    hw_info.gmdid = dev_info.ip_version();
    hw_info.systolicAvailable = dev_info.mayiuse_systolic();
    hw_info.isEfficient64Bit = dev_info.is_efficient_64bit();
    return hw_info;
}

gemmstone::microkernel::HostPayload get_host_payload(
        const compute::device_info_t &dev_info) {
    return {dev_info.min_subgroup_size(), host_argument_bytes};
}

// Interface options of a selected microkernel. Selection only accepts
// candidates whose scales and offsets match the options, so those follow from
// the candidate's problem.
gemmstone::microkernel::GEMMOptions get_ukernel_options(
        const grouped_ocl_params_t &conf,
        const gemmstone::GEMMProblem &problem) {
    gemmstone::microkernel::GEMMOptions opts;
    opts.scaleA = problem.aScale2D();
    opts.offsetA = problem.hasAOffsetPtr() || problem.aOffsetHostScalar();
    opts.offsetAHostScalar = problem.aOffsetHostScalar();
    opts.scaleB = problem.bScale2D();
    opts.offsetB = problem.hasBOffsetPtr() || problem.bOffsetHostScalar();
    opts.offsetBHostScalar = problem.bOffsetHostScalar();
    opts.slmPtr = true;
    opts.kParallelLocal = conf.is_gemv;
    return opts;
}

// Adjusts every candidate strategy, replacing it with the GRPGEMM_USTRATEGY
// override when set.
gemmstone::microkernel::StrategyAdjuster get_strategy_adjuster(
        const compute::device_info_t &dev_info,
        const gemmstone::GEMMProblem &problem) {
    using namespace gemmstone;

    auto hw = getCore(dev_info.product().family);
    auto stepping = dev_info.ip_version() & 0xFF;

    return [=](GEMMStrategy &strat) {
        std::string newStrat;
        strat.dpasw |= strat.fused;
        newStrat = gpu_utils::dev_getenv("GRPGEMM_USTRATEGY", newStrat);
        if (!newStrat.empty()) {
            // Example: 16 16 1 0 aT32 aM32 aB wg 2x4 sys
            printf("GRPGEMM_USTRATEGY: %s\n", newStrat.c_str());
            strat = GEMMStrategy(hw, stepping);
            std::stringstream ss(newStrat);
            ss >> strat.unroll[0];
            ss >> strat.unroll[1];
            float a, b;
            ss >> a;
            ss >> b;
            Scalar alpha((int)a), beta((int)b);
            std::string strategyString;
            std::getline(ss >> std::ws, strategyString);
            parseStrategy(strategyString, problem, strat);
            adjustStrategy(hw, problem, strat);
        }
    };
}

// Strategy requirements used when selection without requirements fails.
// TODO: Find a better way to select wg_n; it is currently derived from the
// tokens per group bucket.
std::vector<gemmstone::StrategyRequirement> fallback_reqs(
        const compute::device_info_t &dev_info,
        const gemmstone::GEMMProblem &problem,
        const gemmstone::microkernel::GEMMOptions &opts, dim_t avg_m) {
    using namespace gemmstone;

    const int sg_size = dev_info.min_subgroup_size();
    auto product = dev_info.product();
    // u3 weights are computed as u4, so their unrolls follow u4
    const Type ta_ext = problem.Ta_ext.is3() ? Type(Type::u4) : problem.Ta_ext;

    // TODO: These values should be based on the eu_count
    dim_t m_unroll = sg_size;

    dim_t n_unroll = std::max<dim_t>(2, utils::rnd_up_pow2(avg_m));
    dim_t min_n_unroll = 1;
    dim_t max_n_unroll = 0;
    dim_t max_wg_n = 4;
    dim_t min_wg_n = 1;

    switch (dev_info.gpu_arch()) {
        case compute::gpu_arch_t::xe_lp:
        case compute::gpu_arch_t::xe_hp:
            max_n_unroll = (problem.aqGroupK > 64
                                   && problem.A.layout == MatrixLayout::T)
                            || (ta_ext.isF8() && opts.scaleA && opts.scaleB)
                            || (problem.Ta_scale == Type::f8_e8m0
                                    || problem.Tb_scale == Type::f8_e8m0)
                    ? 8
                    : 16;
            break;
        case compute::gpu_arch_t::xe_hpg: {
            bool is_xelpg = (product.family == ngen::ProductFamily::ARL
                    || product.family == ngen::ProductFamily::MTL);
            max_n_unroll = (ta_ext.bits() <= 8 && ta_ext.isInteger())
                    ? sg_size * ta_ext
                    : 16;
            if (is_xelpg && !dev_info.mayiuse_systolic()
                    && ta_ext.bits() <= 8) {
                min_n_unroll = (opts.scaleA || opts.scaleB) ? sg_size : 4;
            }
            if (!dev_info.mayiuse_systolic()) max_wg_n = 2;
            if (ta_ext.bits() <= 8) min_wg_n = 2;
        } break;
        case compute::gpu_arch_t::xe_hpc: max_n_unroll = 32; break;
        default:
            if (problem.Tb != Type::f32) {
                m_unroll = std::max<dim_t>(sg_size, sg_size / ta_ext);
            }
            max_n_unroll = 32;
    }

    const dim_t unroll_n
            = std::max(min_n_unroll, std::min(n_unroll, max_n_unroll));
    const dim_t wg_n = utils::rnd_up_pow2(
            std::max(min_wg_n, std::min(avg_m / unroll_n, max_wg_n)));

    return {StrategyRequirement::UnrollM == m_unroll,
            StrategyRequirement::UnrollN == unroll_n,
            StrategyRequirement::WGM == 2, StrategyRequirement::WGN == wg_n};
}

// The k_axis kernel names its inputs after the microkernel operand roles,
// since transc decides which tensor is which. Define the types by role too,
// so the kernel never has to assume that src and wei share one.
void init_kernel_ctx_k_axis(
        const grouped_micro_params_t &conf, compute::kernel_ctx_t &kernel_ctx) {
    const data_type_t a_dt = conf.transc ? conf.src_dt : conf.wei_dt;
    const data_type_t b_dt = conf.transc ? conf.wei_dt : conf.src_dt;

    def_data_type(kernel_ctx, a_dt, "A");
    def_data_type(kernel_ctx, b_dt, "B");
    kernel_ctx.define_int(
            "A_ELEMS_PER_BYTE", types::bytes_to_elements(a_dt, 1));
    kernel_ctx.define_int(
            "B_ELEMS_PER_BYTE", types::bytes_to_elements(b_dt, 1));
}

// The m_axis kernel additionally carries quantization, bias, GEMV k-slicing
// and post-ops.
void init_kernel_ctx_m_axis(
        const grouped_micro_params_t &conf, compute::kernel_ctx_t &kernel_ctx) {
    const auto &problem = conf.ukernel.problem;
    const data_type_t ugemm_result_dt = problem.Tc == gemmstone::Type::s32
            ? data_type::s32
            : data_type::f32;

    def_data_type(kernel_ctx, conf.src_dt, "SRC");
    def_data_type(kernel_ctx, conf.wei_dt, "WEI");
    def_data_type(kernel_ctx, ugemm_result_dt, "UGEMM_RESULT");

    kernel_ctx.define_int(
            "SRC_ELEMS_PER_BYTE", types::bytes_to_elements(conf.src_dt, 1));
    if (!conf.int32_offset) kernel_ctx.use_int32_offset(false);

    // Only the quantization data types are consumed by the kernel. The rest of
    // quantization_t::define_macros() (e.g. the scale count) depends on the
    // grouped dimension and would defeat kernel caching.
    def_data_type(kernel_ctx, conf.src_scale_dt, "SRC_SCALES");
    def_data_type(kernel_ctx, conf.src_zp_dt, "SRC_ZP");
    def_data_type(kernel_ctx, conf.wei_scale_dt, "WEI_SCALES");
    def_data_type(kernel_ctx, conf.wei_zp_dt, "WEI_ZP");

    kernel_ctx.define_int("WITH_SRC_SCALES", conf.with_src_scale);
    kernel_ctx.define_int("WITH_WEI_SCALES", conf.with_wei_scale);
    kernel_ctx.define_int("WITH_SRC_ZP", conf.with_src_zp);
    kernel_ctx.define_int("WITH_WEI_ZP", conf.with_wei_zp);
    kernel_ctx.define_int("WEI_ZP_COMMON", conf.wei_zp_common);
    kernel_ctx.define_int("WEI_ZP_HOST_SCALAR", conf.wei_zp_host_scalar);
    if (conf.with_src_scale || conf.with_src_zp) {
        kernel_ctx.define_int("SRC_GROUP_SIZE", conf.src_group_k);
    }
    if (conf.with_wei_scale || conf.with_wei_zp) {
        kernel_ctx.define_int("WEI_K_GROUP_SIZE", conf.wei_group_k);
        kernel_ctx.define_int("WEI_N_GROUP_SIZE", conf.wei_group_n);
    }

    kernel_ctx.define_int("SRC_SCALES_GROUPED", problem.bScale2D());
    kernel_ctx.define_int("WEI_SCALES_GROUPED", problem.aScale2D());

    if (conf.with_src_zp) {
        kernel_ctx.define_int("SRC_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(conf.src_zp_dt, 1));
    }
    if (conf.with_wei_zp) {
        kernel_ctx.define_int("WEI_ZP_ELEMS_PER_BYTE",
                types::bytes_to_elements(conf.wei_zp_dt, 1));
    }

    auto with_po = [&](po_kind_t kind) {
        return find_po_in_chain(conf.po_chain.data(), kind) != -1;
    };
    def_data_type(kernel_ctx, conf.bia_dt, "BIA");
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("K_PARALLEL_LOCAL", conf.is_gemv);
    kernel_ctx.define_int("WITH_SPARSE_GROUPS", conf.is_gemv);
    kernel_ctx.define_int("SPARSE_GROUPS_PER_PASS", sparse_groups_per_pass);
    kernel_ctx.define_int("WITH_POST_OP", conf.with_post_op);
    kernel_ctx.define_int("WITH_BINARY_GROUPED_SCALE",
            with_po(po_kind_t::binary_grouped_scale));
    kernel_ctx.define_int(
            "WITH_BINARY_DENSE_SCALE", with_po(po_kind_t::binary_dense_scale));
    kernel_ctx.define_int(
            "WITH_BINARY_NVFP4_SCALE", with_po(po_kind_t::binary_nvfp4_scale));
    kernel_ctx.add_custom_header("grouped_post_ops.h",
            generate_post_ops_microgemm_header(
                    conf.post_ops, conf.po_chain.data()));
    def_data_type(kernel_ctx,
            binary_scale_dt(conf.post_ops, conf.po_chain.data(),
                    po_kind_t::binary_grouped_scale),
            "BINARY_SCALE_GROUPED");
    def_data_type(kernel_ctx,
            binary_scale_dt(conf.post_ops, conf.po_chain.data(),
                    po_kind_t::binary_dense_scale),
            "BINARY_SCALE_DENSE");
}

} // namespace

status_t grouped_micro_params_t::create_generator(
        const intel::engine_t &engine, compute::kernel_bundle_t &bundle) const {
    const compute::device_info_t &dev_info = *engine.device_info();

    gemmstone::microkernel::Package package;
    grouped_ukernel_config_t config;
    compute::kernel_ctx_t kernel_ctx;
    // Exceptions must not escape into the kernel cache
    try {
        package = gemmstone::microkernel::generateGEMM(
                get_ukernel_options(*this, ukernel.problem),
                get_host_payload(dev_info), get_hw_info(dev_info), ukernel);
        config.wg_tile_m = package.getSetting("wg_tile_m");
        config.wg_tile_n = package.getSetting("wg_tile_n");
        config.sg_per_wg_m = package.getSetting("sg_per_wg_m");
        config.sg_per_wg_n = package.getSetting("sg_per_wg_n");
        config.sg_per_wg_k = package.getSetting("sg_per_wg_k");
        CHECK(compute::validate_microkernel(package, "grouped_gemm"));
        CHECK(get_kernel_ctx(kernel_ctx, dev_info, package));
    } catch (const std::exception &ex) {
        VDEBUGINFO(4, primitive, matmul,
                "grouped_gemm microkernel generation failure: %s", ex.what());
        return status::runtime_error;
    }
    std::vector<compute::kernel_t> kernels;
    CHECK(engine.create_kernels(&kernels, get_kernel_names(), kernel_ctx));
    bundle = compute::kernel_bundle_t(
            std::move(kernels), get_kernel_names(), config);
    return status::success;
}

status_t grouped_micro_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx,
        const compute::device_info_t &dev_info,
        const gemmstone::microkernel::Package &package) const {
    /* Generate microkernel shims */
    const int sg_size = dev_info.min_subgroup_size();
    kernel_ctx.define_int("SUBGROUP_SIZE", sg_size);

    compute::microkernel_shims_t shims(
            kernel_ctx, sg_size, dev_info.gpu_arch());
    shims.add("gemm_grouped.h", "grouped", package);
    shims.require_grfs(ukernel.strategy.GRFs);
    shims.finalize();

    // Definitions shared by the m_axis and k_axis kernels
    kernel_ctx.set_data_type(dst_dt);
    def_data_type(kernel_ctx, dst_dt, "DST");
    kernel_ctx.define_int("WITH_SLM", package.getSetting("slm_size") > 0);
    kernel_ctx.add_option("-cl-std=CL3.0");

    switch (axis) {
        case grouped_axis_t::m_axis:
            init_kernel_ctx_m_axis(*this, kernel_ctx);
            break;
        case grouped_axis_t::k_axis:
            init_kernel_ctx_k_axis(*this, kernel_ctx);
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

template <size_t N>
void calc_group_sizes(std::array<int, N> &dims, const quant_entry_t &entry,
        const memory_desc_wrapper &desc) {
    memory_desc_t md;
    entry.get_md(md, *desc.md_);
    std::transform(desc.dims(), desc.dims() + dims.size(), md.dims, begin(dims),
            [](dim_t d, dim_t d2) -> int {
        return static_cast<int>(d2 == 0 ? 1 : d / d2);
    });
}

std::array<int, 2> grouped_micro_gemm_t::pd_t::src_group_sizes() const {
    std::array<int, 2> sizes = {0, 0};
    memory_desc_wrapper src_d(src_md());
    if (src_quant().with_scale()) {
        calc_group_sizes(sizes, attr()->scales_.get(DNNL_ARG_SRC), src_d);
    } else if (src_quant().with_zp()) {
        calc_group_sizes(sizes, attr()->zero_points_.get(DNNL_ARG_SRC), src_d);
    }
    return sizes;
}

status_t grouped_micro_gemm_t::pd_t::init_k_axis(const impl::engine_t *engine) {
    using namespace data_type;

    kernel_name_ = "grouped_gemm:micro:k_axis";

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    const data_type_t src_dt = src_d.data_type();

    VDISPATCH_MATMUL(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    // src is col-major, wei is row-major
    const dims_t &src_strides = src_d.strides();
    const dims_t &wei_strides = wei_d.strides();
    VDISPATCH_MATMUL(src_strides[0] == 1 && src_strides[1] == src_d.dims()[0],
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(wei_strides[1] == 1 && wei_strides[0] == wei_d.dims()[1],
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_MATMUL(!dst_d.is_sparse_desc() && !dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    if (dst_d.format_any())
        CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

    VDISPATCH_MATMUL(dst_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_MATMUL(
            utils::everyone_is(src_dt, wei_d.data_type(), dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(src_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t grouped_micro_gemm_t::pd_t::init_m_axis(const impl::engine_t *engine) {
    using namespace data_type;

    kernel_name_ = "grouped_gemm:micro:m_axis";

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    data_type_t src_dt = src_d.data_type();
    data_type_t wei_dt = wei_d.data_type();
    data_type_t dst_dt = dst_d.data_type();

    // Weights should be dense
    VDISPATCH_MATMUL(!wei_d.is_sparse_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    VDISPATCH_MATMUL_SC(attr_.post_ops_.set_default_formats(dst_md(0)),
            VERBOSE_UNSUPPORTED_POSTOP);

    // Resolve format_any to plain dense
    if (wei_d.format_any())
        CHECK(memory_desc_init_by_strides(weights_md_, nullptr));

    VDISPATCH_MATMUL(wei_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_MATMUL(utils::one_of(src_dt, f32, f16, bf16, u8, s8, f8_e5m2,
                             f8_e4m3, f4_e2m1),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(wei_dt, f32, f16, bf16, u8, s8, s4, u4, u3,
                             f8_e5m2, f8_e4m3, f4_e2m1),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(IMPLICATION(wei_dt == u3, src_dt == f16),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // u3 weights are only loaded with transposing 2D block messages, which
    // require K-contiguous weights with an aligned pitch
    if (wei_dt == u3) {
        const auto *dev_info = utils::downcast<const intel::engine_t *>(engine)
                                       ->device_info();
        VDISPATCH_MATMUL(dev_info->gpu_arch() >= compute::gpu_arch_t::xe_hpc,
                VERBOSE_UNSUPPORTED_DEVICE_FEATURE,
                "u3 weights requires 2d block loads");
        VDISPATCH_MATMUL(wei_d.matches_tag(format_tag::acb),
                VERBOSE_UNSUPPORTED_TAG_S, "u3 weights");
        auto hw = getCore(dev_info->product().family);
        const dim_t ld_bytes = types::elements_to_bytes(
                wei_dt, gemm_desc_t::get_ld(*weights_md(0)));
        const int pitch_alignment = jit::block_2d_pitch_alignment(hw);
        VDISPATCH_MATMUL(ld_bytes % pitch_alignment == 0,
                VERBOSE_UNSUPPORTED_MEM_STRIDE
                ": u3 weights pitch (%d bytes) is not %d-byte aligned",
                (int)ld_bytes, pitch_alignment);
    }

    const bool wei_subbyte = utils::one_of(wei_dt, s4, u4);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (N() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 2);
    VDISPATCH_MATMUL(IMPLICATION(wei_dt == u3, (K() % 8) == 0), VERBOSE_BAD_DIM,
            "weights", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_dt == u3, (N() % 8) == 0), VERBOSE_BAD_DIM,
            "weights", 2);

    // Check for supported quantization schemes
    if (src_quant().with_scale()) {
        VDISPATCH_MATMUL(utils::one_of(src_quant().scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": src scales dt(%s)",
                dnnl_dt2str(src_quant().scale_dt()));
    }

    if (src_quant().with_zp()) {
        const int src_zp_mask = src_quant().zp_mask();
        // Only per-row or per-column zero points supported for src
        VDISPATCH_MATMUL(utils::one_of(src_zp_mask, src_qmask_M(),
                                 src_qmask_M() | src_qmask_K(), 0),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points mask(%d)",
                src_zp_mask);
        VDISPATCH_MATMUL(utils::one_of(src_quant().zp_dt(), u8, s8),
                VERBOSE_UNSUPPORTED_ZP_CFG ": src zero points dt(%s)",
                dnnl_dt2str(src_quant().zp_dt()));
    }

    if (src_quant().with_scale() && src_quant().with_zp()) {
        const int src_scale_mask = src_quant().scale_mask();
        const int src_zp_mask = src_quant().zp_mask();
        VDISPATCH_MATMUL(src_scale_mask == src_zp_mask,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": src scale(%d) and zp(%d) mask must match",
                src_scale_mask, src_zp_mask);
    }

    if (wei_quant().with_scale()) {
        const int wei_mask = wei_quant().scale_mask();
        VDISPATCH_MATMUL(
                utils::one_of(wei_mask, 7, 5), VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(utils::one_of(wei_quant().scale_dt(), f32, f16, bf16,
                                 f8_e5m2, f8_e4m3, e8m0, f4_e2m1),
                VERBOSE_UNSUPPORTED_SCALES_CFG ": wei scales dt(%s)",
                dnnl_dt2str(wei_quant().scale_dt()));
    }

    if (wei_quant().with_zp()) {
        const int wei_zp_mask = wei_quant().zp_mask();
        // Common zero points are supported only for u3 weights.
        VDISPATCH_MATMUL(utils::one_of(wei_zp_mask, 7, 5, 0),
                VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points mask(%d)",
                wei_zp_mask);
        VDISPATCH_MATMUL(IMPLICATION(wei_zp_mask == 0, wei_dt == u3),
                VERBOSE_UNSUPPORTED_ZP_CFG
                ": common wei zero points require u3 weights");
        if (attr()->zero_points_.get(DNNL_ARG_WEIGHTS).is_host_scalar()) {
            VDISPATCH_MATMUL(utils::one_of(wei_quant().zp_dt(), u8, s8),
                    VERBOSE_UNSUPPORTED_ZP_CFG
                    ": host scalar wei zero points dt(%s)",
                    dnnl_dt2str(wei_quant().zp_dt()));
        } else if (wei_zp_mask == 5) {
            VDISPATCH_MATMUL(
                    utils::one_of(wei_quant().zp_dt(), s32, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points dt(%s)",
                    dnnl_dt2str(wei_quant().zp_dt()));
        } else {
            VDISPATCH_MATMUL(utils::one_of(wei_quant().zp_dt(), u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_ZP_CFG ": wei zero points dt(%s)",
                    dnnl_dt2str(wei_quant().zp_dt()));
        }
    }

    if (wei_quant().with_scale() && wei_quant().with_zp()) {
        const int wei_scale_mask = wei_quant().scale_mask();
        const int wei_zp_mask = wei_quant().zp_mask();
        VDISPATCH_MATMUL(wei_scale_mask == wei_zp_mask || wei_zp_mask == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG
                ": wei scale(%d) and zp(%d) mask must match",
                wei_scale_mask, wei_zp_mask);
    }

    // The microkernel applies scales and zero points with a single group
    // shape per operand. Common wei zero points have no groups.
    if (src_quant().with_scale() && src_quant().with_zp()) {
        std::array<int, 2> src_zp_group_sizes;
        calc_group_sizes(src_zp_group_sizes,
                attr()->zero_points_.get(DNNL_ARG_SRC), src_d);
        VDISPATCH_MATMUL(src_zp_group_sizes == src_group_sizes(),
                VERBOSE_UNSUPPORTED_ZP_CFG
                ": src scale and zp groups must match");
    }
    if (wei_quant().with_scale() && wei_quant().with_zp()
            && wei_quant().zp_mask() != 0) {
        std::array<int, 3> wei_scale_group_sizes;
        std::array<int, 3> wei_zp_group_sizes;
        calc_group_sizes(wei_scale_group_sizes,
                attr()->scales_.get(DNNL_ARG_WEIGHTS), wei_d);
        calc_group_sizes(wei_zp_group_sizes,
                attr()->zero_points_.get(DNNL_ARG_WEIGHTS), wei_d);
        VDISPATCH_MATMUL(wei_zp_group_sizes == wei_scale_group_sizes,
                VERBOSE_UNSUPPORTED_ZP_CFG
                ": wei scale and zp groups must match");
    }

    return status::success;
}

namespace {

// Derives the microkernel problem from the pd and the kernel parameters.
void init_ukernel_problem(const grouped_micro_gemm_t::pd_t &pd,
        const grouped_ocl_params_t &conf,
        const compute::device_info_t &dev_info, gemmstone::GEMMProblem &problem,
        gemmstone::microkernel::GEMMOptions &opts,
        gemmstone::SizeParams &sizes) {
    using namespace gemmstone;
    using namespace gemmstone::microkernel;
    using gemm::jit::convert_dnnl_to_kernel_type;

    problem.product = dev_info.product();
    problem.Ta_ext = convert_dnnl_to_kernel_type(conf.wei_dt);
    problem.Tb_ext = convert_dnnl_to_kernel_type(conf.src_dt);
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;

    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;

    data_type_t adt = conf.wei_dt;
    data_type_t bdt = conf.src_dt;
    dim_t lda = 0, ldb = 0;
    switch (conf.axis) {
        case grouped_axis_t::m_axis:
            // src is the B operand: [total_M, K] row-major, contiguous along k
            // (layout N). gemm_desc_t helpers read the blocking desc, which is
            // only populated for the dense 3D weights of this workload.
            sizes.m = pd.N();
            sizes.n = conf.is_gemv ? 1 : 32;
            sizes.k = pd.K();
            lda = gemm_desc_t::get_ld(*pd.weights_md(0));
            ldb = pd.K();
            problem.A.layout
                    = gemm_desc_t::get_trans(*pd.weights_md(0)) == dnnl_trans
                    ? MatrixLayout::T
                    : MatrixLayout::N;
            problem.B.layout = MatrixLayout::N;
            break;
        case grouped_axis_t::k_axis:
            // The C tile is col-major, so the operand order follows the
            // dst transposition:
            //   notrans (N contiguous): A = wei [total_K, N] -> tile is N x M
            //   trans   (M contiguous): A = src [M, total_K] -> tile is M x N
            // Only the leading dimensions and extents swap. Both operands are
            // k-major, hence A layout N and B layout T either way.
            sizes.m = lda = pd.transc() ? pd.M() : pd.N();
            sizes.n = ldb = pd.transc() ? pd.N() : pd.M();
            // Use the average k size to avoid unnecessary unrolls, rounded
            // up so that varying token counts share a kernel.
            sizes.k = utils::rnd_up_pow2(utils::div_up(pd.K(), pd.ngroups()));
            if (conf.transc) std::swap(adt, bdt);
            problem.A.layout = MatrixLayout::N;
            problem.B.layout = MatrixLayout::T;
            break;
        default: assert(!"unexpected grouped axis");
    }

    problem.C.layout = MatrixLayout::N;

    // Set Alignments
    problem.A.setAlignment(alignmentForLD(
            static_cast<int>(types::elements_to_bytes(adt, lda))));
    problem.B.setAlignment(alignmentForLD(
            static_cast<int>(types::elements_to_bytes(bdt, ldb))));
    problem.C.setAlignment(problem.Tc.size());

    // Scales grouped along N need the microkernel even when a single group
    // covers K: the per-column scales applied after the GEMM can't express
    // them.
    opts.scaleA = conf.with_wei_scale
            && (conf.wei_group_k < sizes.k || conf.wei_group_n > 1);
    opts.offsetA = conf.with_wei_zp;
    opts.offsetAHostScalar = opts.offsetA && conf.wei_zp_host_scalar;
    opts.scaleB = conf.with_src_scale && conf.src_group_k < sizes.k;
    opts.offsetB = conf.with_src_zp;
    opts.slmPtr = true;
    opts.kParallelLocal = conf.is_gemv;

    if (opts.scaleA) {
        problem.Ta_scale = convert_dnnl_to_kernel_type(conf.wei_scale_dt);
        problem.A_scale.setAlignment(alignmentForLD(
                static_cast<int>(types::data_type_size(conf.wei_scale_dt))));
        problem.A_scale.layout = MatrixLayout::N;
        problem.asPtrDims = 2;
    }

    if (opts.offsetA) {
        problem.Tao = convert_dnnl_to_kernel_type(conf.wei_zp_dt);
        problem.AO.setAlignment(
                static_cast<int>(types::data_type_size(conf.wei_zp_dt)));
        problem.AO.layout = MatrixLayout::N;
        const bool wei_zp_1d = pd.wei_quant().zp_mask() == 5
                && pd.attr()->zero_points_.has_default_groups(DNNL_ARG_WEIGHTS);
        problem.aoPtrDims = opts.offsetAHostScalar ? -1
                : conf.wei_zp_common               ? 0
                : wei_zp_1d                        ? 1
                                                   : 2;
        problem.aOffset = ABOffset::Calc;
    }

    if (opts.scaleB) {
        problem.Tb_scale = convert_dnnl_to_kernel_type(conf.src_scale_dt);
        problem.B_scale.setAlignment(
                static_cast<int>(types::data_type_size(conf.src_scale_dt)));
        problem.bsPtrDims = pd.src_quant().scale_mask()
                        == (pd.src_qmask_M() | pd.src_qmask_K())
                ? 2
                : 1;
        problem.B_scale.layout
                = problem.bsPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }
    if (opts.offsetB) {
        problem.Tbo = convert_dnnl_to_kernel_type(conf.src_zp_dt);
        problem.BO.setAlignment(
                static_cast<int>(types::data_type_size(conf.src_zp_dt)));
        problem.boPtrDims = pd.src_quant().zp_mask()
                        == (pd.src_qmask_M() | pd.src_qmask_K())
                ? 2
                : 1;
        problem.bOffset = ABOffset::Calc;
        problem.BO.layout
                = problem.boPtrDims > 1 ? MatrixLayout::N : MatrixLayout::T;
    }

    if (opts.scaleA || opts.offsetA) {
        problem.aqGroupM = conf.wei_group_n;
        problem.aqGroupK = utils::rnd_up_pow2(conf.wei_group_k);
    }

    if (opts.scaleB || opts.offsetB) {
        problem.bqGroupN = pd.src_group_sizes()[0];
        problem.bqGroupK = utils::rnd_up_pow2(conf.src_group_k);
    }

    // Mixed s8/s4 DPAS support:
    // - Xe3p: Not supported, require s4->s8 upconversion
    // - pre-Xe3p: supported, but only when s4 matrix doesn't have zero points
    bool has_s8s4_dpas = dev_info.gpu_arch() != compute::gpu_arch_t::xe3p;
    if (problem.Ta_ext.isSubByteInt() && problem.Tb_ext.isInt8()) {
        bool s8s4_dpas_ok = (problem.Ta_ext.bits() == 4) && has_s8s4_dpas
                && !opts.offsetA;
        if (!s8s4_dpas_ok) problem.Ta = Type::s8;
    }
    if (problem.Tb_ext.isSubByteInt() && problem.Ta_ext.isInt8()) {
        bool s8s4_dpas_ok = (problem.Tb_ext.bits() == 4) && has_s8s4_dpas
                && !opts.offsetB;
        if (!s8s4_dpas_ok) problem.Tb = Type::s8;
    }

    if (problem.Ta.is3()) problem.Ta = Type::u4;
    // When both A and B are integers and group sums are needed, we
    // can avoid using group sums by converting one of the inputs to
    // f16/bf16.
    if (problem.Ta.isInteger() && problem.Tb.isInteger()
            && ((problem.needsAGroupSums() || problem.needsBGroupSums())
                    || ((opts.scaleA || opts.scaleB)
                            && dev_info.gpu_arch()
                                    < compute::gpu_arch_t::xe_hpc))) {
        Type ctype = Type::f16;
        if (utils::one_of(Type::bf16, problem.Ta_scale, problem.Tb_scale,
                    convert_dnnl_to_kernel_type(conf.dst_dt)))
            ctype = Type::bf16;
        if (problem.Ta_ext.bits() < problem.Tb_ext.bits()) {
            problem.Ta = ctype;
        } else {
            problem.Tb = ctype;
        }
    } else if (problem.Ta.isInteger() && problem.Tb.isInteger() && !opts.scaleA
            && !opts.scaleB) {
        problem.Tc = problem.Ts = problem.Tc_ext = Type::s32;
    }
}

} // namespace

// Collects the kernel parameters but the microkernel into conf.
status_t grouped_micro_gemm_t::pd_t::init_conf(
        grouped_micro_params_t &conf) const {
    conf.axis = axis();
    conf.src_dt = src_md()->data_type;
    conf.wei_dt = weights_md(0)->data_type;
    conf.dst_dt = dst_md()->data_type;

    switch (conf.axis) {
        case grouped_axis_t::m_axis: {
            conf.with_src_scale = src_quant().with_scale();
            conf.with_src_zp = src_quant().with_zp();
            conf.with_wei_scale = wei_quant().with_scale();
            conf.with_wei_zp = wei_quant().with_zp();
            conf.src_scale_dt = src_quant().scale_dt();
            conf.src_zp_dt = src_quant().zp_dt();
            conf.wei_scale_dt = wei_quant().scale_dt();
            conf.wei_zp_dt = wei_quant().zp_dt();
            conf.src_group_k = src_group_sizes()[1];

            std::array<int, 3> wei_group_sizes = {0, 0, 0};
            memory_desc_wrapper wei_d(weights_md(0));
            if (wei_quant().with_scale()) {
                calc_group_sizes(wei_group_sizes,
                        attr()->scales_.get(DNNL_ARG_WEIGHTS), wei_d);
            } else if (wei_quant().with_zp()) {
                calc_group_sizes(wei_group_sizes,
                        attr()->zero_points_.get(DNNL_ARG_WEIGHTS), wei_d);
            }
            conf.wei_group_k = wei_group_sizes[1];
            conf.wei_group_n = wei_group_sizes[2];
            conf.wei_zp_common
                    = wei_quant().with_zp() && wei_quant().zp_mask() == 0;
            conf.wei_zp_host_scalar = wei_quant().with_zp()
                    && attr()->zero_points_.get(DNNL_ARG_WEIGHTS)
                               .is_host_scalar();
            conf.int32_offset = wei_d.strides()[0] * ngroups()
                    < std::numeric_limits<int>::max();

            conf.with_bias = with_bias();
            conf.bia_dt = weights_md(1)->data_type;

            conf.is_gemv = M() < ngroups();
            conf.with_post_op = !attr()->post_ops_.has_default_values();
            if (conf.with_post_op) {
                const memory_desc_wrapper dst_d(dst_md());
                CHECK(check_post_op_chain(
                        *attr(), dst_d, ngroups(), conf.po_chain.data()));
                VDISPATCH_MATMUL_SC(gpu_post_ops_t::make(conf.post_ops,
                                            attr()->post_ops_, dst_d),
                        VERBOSE_UNSUPPORTED_POSTOP);
            }
        } break;
        case grouped_axis_t::k_axis: conf.transc = transc(); break;
        default: return status::unimplemented;
    }

    return status::success;
}

// Tries the microkernel candidates for the kernel parameters in order of
// preference, first without strategy requirements and then with the fallback
// requirements. The microkernel is generated on a kernel cache miss; a
// candidate that fails to generate or build moves on to the next one.
bool grouped_micro_gemm_t::select_ukernel(impl::engine_t *engine,
        grouped_micro_params_t &conf, compute::kernel_bundle_t &bundle) {
    using namespace gemmstone;
    using namespace gemmstone::microkernel;

    const auto &dev_info
            = *utils::downcast<const intel::engine_t *>(engine)->device_info();

    GEMMProblem problem;
    GEMMOptions opts;
    SizeParams sizes;
    init_ukernel_problem(*pd(), conf, dev_info, problem, opts, sizes);

    const HWInformation hw_info = get_hw_info(dev_info);
    const HostPayload host = get_host_payload(dev_info);
    const StrategyAdjuster adjuster = get_strategy_adjuster(dev_info, problem);

    // Returns true once the kernels of a candidate are created.
    auto try_select = [&](const std::vector<StrategyRequirement> &reqs) {
        try {
            GEMMSelector selector(
                    opts, host, hw_info, sizes, problem, reqs, adjuster);
            GEMMCandidate candidate;
            while (selector.next(candidate)) {
                conf.ukernel = std::move(candidate);
                if (create_kernels(
                            engine, bundle, conf.get_kernel_names(), conf)
                        == status::success)
                    return true;
            }
        } catch (const std::runtime_error &ex) {
            VDEBUGINFO(4, primitive, matmul,
                    "grouped_gemm microkernel selection failure: %s",
                    ex.what());
        }
        return false;
    };

    // Extent of the microkernel's n dimension: tokens per group for the
    // m_axis workload, rounded up to a power of two so that token counts
    // select alike; for k_axis it follows the operand order.
    const dim_t avg_m = conf.axis == grouped_axis_t::k_axis
            ? sizes.n
            : utils::rnd_up_pow2(pd()->M() / pd()->ngroups());

    // Select without strategy requirements first, then with the fallback
    // requirements.
    return try_select({})
            || try_select(fallback_reqs(dev_info, problem, opts, avg_m));
}

status_t grouped_micro_gemm_t::pd_t::init(const impl::engine_t *engine) {
    assert(engine->kind() == engine_kind::gpu);
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");
    if (dev_info->ip_version() == 0) return status::unimplemented;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper dst_d(dst_md());

    // Grouped src is common to both workloads, its variable dim selects which
    // axis is partitioned (grouped_matmul_desc_init() checked the others)
    VDISPATCH_MATMUL(src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    switch (src_d.sparse_desc().grouped_desc.variable_dim_idx) {
        case 0: CHECK(init_m_axis(engine)); break;
        case 1: CHECK(init_k_axis(engine)); break;
        default: return status::unimplemented;
    }
    bool with_post_op = !attr()->post_ops_.has_default_values();
    if (with_post_op) {
        std::array<po_kind_t, 3> po_chain
                = {{po_kind_t::none, po_kind_t::none, po_kind_t::none}};
        VDISPATCH_MATMUL_SC(
                check_post_op_chain(*attr(), dst_d, ngroups(), po_chain.data()),
                VERBOSE_UNSUPPORTED_POSTOP "post-op chain");
    }

    return status::success;
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    sg_size_ = dev_info->min_subgroup_size();
    grouped_micro_params_t conf_;
    CHECK(pd()->init_conf(conf_));
    std::copy(std::begin(conf_.po_chain), std::end(conf_.po_chain),
            std::begin(po_chain));

    const auto &kernel_names = conf_.get_kernel_names();
    compute::kernel_bundle_t bundle;
    if (cache_blob()) {
        // The launch configuration precedes the kernels in the cache blob
        CHECK(cache_blob().get_value(
                (uint8_t *)&ukernel_config_, sizeof(ukernel_config_)));
        VCHECK_KERNEL(create_kernels(engine, bundle, kernel_names, conf_),
                VERBOSE_KERNEL_CREATION_FAIL, kernel_names[0]);
    } else {
        if (!select_ukernel(engine, conf_, bundle)) {
            VERROR(primitive, gpu, "%s,%s", pd()->info(engine),
                    "no grouped_gemm microkernel could be generated or built");
            return status::runtime_error;
        }
        ukernel_config_ = bundle.info<grouped_ukernel_config_t>();
    }

    return bundle.get_kernels(kernel_names, kernel_);
}

status_t grouped_micro_gemm_t::execute_k_axis(const exec_ctx_t &ctx) const {
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = ctx.input(DNNL_ARG_WEIGHTS)->md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const dim_t M = dst_md->dims[1];
    const dim_t N = dst_md->dims[2];

    const dim_t ldsrc = memory_desc_wrapper(src_md).strides()[1];
    const dim_t ldwei = memory_desc_wrapper(wei_md).strides()[0];
    const dim_t ldc = gemm_desc_t::get_ld(*dst_md);

    // Assign the microkernel operands so that the C tile stores contiguously:
    // its m dimension has to be the contiguous dst dimension. This must match
    // the layouts and extents init_microkernels() described to the generator.
    //   notrans: a = wei, tile is N x M, ldc = N
    //   trans:   a = src, tile is M x N, ldc = M
    const bool trans_c = pd()->transc();
    const memory_storage_t &a_data = trans_c ? src_data : wei_data;
    const memory_storage_t &b_data = trans_c ? wei_data : src_data;
    const dim_t lda = trans_c ? ldsrc : ldwei;
    const dim_t ldb = trans_c ? ldwei : ldsrc;
    const dim_t m = trans_c ? M : N;
    const dim_t n = trans_c ? N : M;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(a_data);
    arg_list.append(lda);
    arg_list.append(b_data);
    arg_list.append(ldb);
    arg_list.append(dst_data);
    arg_list.append(ldc);
    arg_list.append(src_offsets);
    arg_list.append(m);
    arg_list.append(n);

    const size_t wg_tile_m = ukernel_config_.wg_tile_m;
    const size_t wg_tile_n = ukernel_config_.wg_tile_n;

    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= sg_size_ * ukernel_config_.sg_per_wg_m;
    lws[1] *= ukernel_config_.sg_per_wg_n;
    lws[2] *= ukernel_config_.sg_per_wg_k;

    compute::range_t gws = lws;
    gws[0] *= utils::div_up(m, wg_tile_m);
    gws[1] *= utils::div_up(n, wg_tile_n);
    gws[2] *= pd()->ngroups();

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

status_t grouped_micro_gemm_t::execute(const exec_ctx_t &ctx) const {
    switch (pd()->axis()) {
        case grouped_axis_t::m_axis: return execute_m_axis(ctx);
        case grouped_axis_t::k_axis: return execute_k_axis(ctx);
        default:
            assert(!"unknown grouped workload");
            return status::invalid_arguments;
    }
}

status_t grouped_micro_gemm_t::execute_m_axis(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto src_quant = pd()->src_quant();
    const auto wei_quant = pd()->wei_quant();
    const auto &src_scales = src_quant.scales(ctx);
    const auto &src_zero_points = src_quant.zero_points(ctx);
    const auto &wei_scales = wei_quant.scales(ctx);
    const auto &wei_zero_points = wei_quant.zero_points(ctx);

    const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = pd()->weights_md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const bool with_src_scales = src_quant.with_scale();
    const bool with_src_zero_points = src_quant.with_zp();
    const bool with_wei_scales = wei_quant.with_scale();
    const bool with_wei_zero_points = wei_quant.with_zp();

    dim_t ldsrcq = 0;
    dim_t ldweiq = 0;

    if (with_src_scales || with_src_zero_points) {
        const memory_desc_t *src_quant_md = with_src_scales
                ? ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldsrcq = static_cast<int>(
                src_quant_md->format_desc.blocking.strides[0]);
    }
    if (with_wei_scales || with_wei_zero_points) {
        const memory_desc_t *wei_quant_md = with_wei_scales
                ? ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldweiq = static_cast<int>(
                wei_quant_md->format_desc.blocking.strides[1]);
    }
    dim_t m_all = dst_md->dims[dst_md->ndims - 2];
    dim_t n = dst_md->dims[dst_md->ndims - 1];
    dim_t k = src_md->dims[src_md->ndims - 1];

    dim_t ldsrc = src_md->dims[src_md->ndims - 1];
    dim_t lddst = dst_md->dims[dst_md->ndims - 1];
    // In units of WEI_DATA_T: sub-byte weights are addressed in bytes
    const data_type_t wei_dt = wei_md->data_type;
    dim_t wei_batch_stride
            = wei_md->format_desc.blocking.strides[wei_md->ndims - 3];
    if (utils::one_of(wei_dt, data_type::f4_e2m1, data_type::s4, data_type::u4,
                data_type::u3))
        wei_batch_stride = types::elements_to_bytes(wei_dt, wei_batch_stride);
    // Must match the A layout chosen by get_trans() in init_ukernel_problem():
    // with N == 1 the stride of the size-1 dimension is arbitrary.
    dim_t ldwei = gemm_desc_t::get_ld(*wei_md);

    compute::kernel_arg_list_t arg_list;

    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(src_offsets);
    arg_list.append(src_scales);
    arg_list.append(src_zero_points);
    arg_list.append(ldsrcq);
    arg_list.append(wei_data);
    arg_list.append(wei_scales);
    if (pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS).is_host_scalar()) {
        int value = 0;
        CHECK(gemm::maybe_get_host_scalar_value(wei_zero_points, value));
        arg_list.append(static_cast<int16_t>(value));
    } else {
        arg_list.append(wei_zero_points);
    }
    arg_list.append(ldweiq);
    arg_list.append(wei_batch_stride);
    arg_list.append(ldwei);
    arg_list.append(dst_data);
    arg_list.append(lddst);
    arg_list.append(dst_offsets);
    arg_list.append(into<int>(n));
    arg_list.append(into<int>(k));
    arg_list.append(into<int>(pd()->ngroups()));
    arg_list.append(bias_data);

    const memory_storage_t *grouped_scale = &memory_storage_t::empty_storage();
    const memory_storage_t *dense_scale = &memory_storage_t::empty_storage();
    const memory_storage_t *nvfp4_scale = &memory_storage_t::empty_storage();
    if (!pd()->attr()->post_ops_.has_default_values()) {
        for (int i = 0; i < pd()->attr()->post_ops_.len(); ++i) {
            auto &e = pd()->attr()->post_ops_.entry_[i];
            if (!e.is_binary()) continue;
            const int po_arg
                    = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
            if (po_chain[i] == po_kind_t::binary_grouped_scale) {
                grouped_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
                dense_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_nvfp4_scale) {
                nvfp4_scale = &CTX_IN_STORAGE(po_arg, 0);
            }
        }
    }

    arg_list.append(*grouped_scale);
    arg_list.append(*dense_scale);
    arg_list.append(*nvfp4_scale);

    size_t sg_per_wg_m = ukernel_config_.sg_per_wg_m;
    size_t sg_per_wg_n = ukernel_config_.sg_per_wg_n;
    size_t sg_per_wg_k = ukernel_config_.sg_per_wg_k;
    size_t wg_tile_m = ukernel_config_.wg_tile_m;
    size_t wg_tile_n = ukernel_config_.wg_tile_n;

    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= sg_size_;

    lws[0] *= sg_per_wg_m;
    lws[1] *= sg_per_wg_n;
    lws[2] *= sg_per_wg_k;

    bool is_gemv = pd()->M() < pd()->ngroups();
    dim_t m_dispatch = m_all;
    const int32_t *max_var_dim
            = CTX_IN_MEM(const int32_t *, DNNL_ARG_HINT_MAX_GROUP_SIZE);
    if (max_var_dim && *max_var_dim > 0 && *max_var_dim <= m_all)
        m_dispatch = *max_var_dim;

    compute::range_t gws = lws;
    // Swap wg_tile_[mn]_ for col-major vs row-major representations
    gws[0] *= utils::div_up(n, wg_tile_m);
    // Sparse (gemv) dispatch derives the M offset from the flat token index
    // in dim 2, so dim 1 only needs a single work-group.
    gws[1] *= is_gemv ? 1 : utils::div_up(m_dispatch, wg_tile_n);
    gws[2] *= is_gemv ? m_all : pd()->ngroups();

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
