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

#ifndef GPU_INTEL_GEMM_JIT_HPP
#define GPU_INTEL_GEMM_JIT_HPP

#include <assert.h>
#include <limits>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/zero_pool.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/gemm/jit/gen_kernel_db.hpp"
#include "gpu/intel/gemm/jit/pd.hpp"
#include "gpu/intel/gemm/primitive.hpp"
#include "gpu/intel/gemm/utils.hpp"
#include "gpu/intel/logging.hpp"
#include "gpu/intel/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

// The zero-buffer scratchpad path is prepared when the zero pool is disabled,
// or (under SYCL) as a fallback while recording a graph.
inline bool prepare_zero_buffer_scratchpad() {
#ifdef DNNL_WITH_SYCL
    return true;
#else
    return !use_zero_pool();
#endif
}

struct zero_fill_params_t
    : public trivially_serializable_t<zero_fill_params_t> {
    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        return engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names {"gemm_zero_fill"};
        return names;
    }

    compute::kernel_ctx_t get_kernel_ctx() const {
        compute::kernel_ctx_t kernel_ctx;
        // Match the main kernel's options (GRF mode, thread arbitration) to
        // avoid hardware state reconfiguration between back-to-back kernel
        // dispatches.
        if (grf_256) kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");
        if (no_subgroup_ifp) kernel_ctx.add_option("-cl-no-subgroup-ifp");
        return kernel_ctx;
    }

    bool grf_256 = false;
    bool no_subgroup_ifp = false;
    uint8_t pad[6] = {};
};

struct gen_t : public primitive_t {
    struct pd_t : public jit::pd_t {
        using jit::pd_t::pd_t;
        using kernel_desc_t = jit::gen_nocopy_desc_t;

        DECLARE_COMMON_PD_T("jit:gemm:any", gen_t);

        status_t init(const impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using namespace alg_kind;
            using smask_t = primitive_attr_t::skip_mask_t;
            using arch_t = compute::gpu_arch_t;

            assert(engine->kind() == engine_kind::gpu);
            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);

            CHECK(set_default_formats(false));

            dev_info_ = intel_engine->device_info();
            arch_ = dev_info_->gpu_arch();

            CHECK(jit::pd_t::init(engine, arch_));

            const auto d = desc();
            auto m = desc()->m();
            auto n = desc()->n();

            // Basic implementation attr support:
            auto attr_skip_mask = smask_t::post_ops | smask_t::fpmath_mode
                    | smask_t::accumulation_mode | smask_t::rounding_mode
                    | smask_t::scales | smask_t::scales_data_type
                    | smask_t::scales_groups | smask_t::precomputed_reductions
                    | smask_t::zero_points | smask_t::zero_points_data_type
                    | smask_t::zero_points_groups | smask_t::post_ops_inplace
                    | smask_t::gpu_attr;
            VDISPATCH_GEMM(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(!attr()->gpu_attr_
                            || utils::downcast<const gpu_primitive_attr_t *>(
                                    attr()->gpu_attr_.get())
                                       ->has_only_kernel_override(),
                    VERBOSE_UNSUPPORTED_ATTR);

            // static_sbzp: weights decompression with grouped scales and
            // zero-points (per-channel scales are applied after the GEMM).
            if (a_quant.zp_after_scale || b_quant.zp_after_scale) {
                VDISPATCH_GEMM(wei_decomp_ && !b_quant.zp_after_scale,
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_GEMM(
                        a_zp_2d() && a_scales_2d(), VERBOSE_UNSUPPORTED_ZP_CFG);
            }

            VDISPATCH_GEMM(
                    !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k(),
                            d->lda(), d->ldb(), d->ldc(), d->batch()),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            // If m = 1, swap A/B to use more efficient n = 1 kernels if possible.
            bool check_lda = ((d->transa() == dnnl_notrans && d->lda() == 1)
                    || (d->transa() == dnnl_trans));
            swap_ab_ = (d->m() == 1 && d->ldc() == 1 && check_lda)
                    || d->transc() == dnnl_trans;

            // Mixed-width kernels exist only with the narrower type in A,
            // swapping would put it in B, where the catalog has no coverage.
            swap_ab_ &= types::data_type_bits(d->a_type())
                    >= types::data_type_bits(d->b_type());

            // No kernels with transposed C, if swap_ab is disabled (e.g. due
            // to wei_decomp_) - this case cannot be handled.
            VDISPATCH_GEMM(IMPLICATION(d->transc() == dnnl_trans, swap_ab_),
                    VERBOSE_UNSUPPORTED_TAG);

            if (swap_ab_) {
                // Do not use transposed B when it is unnecessary
                if (!transa_ && m == 1) {
                    transa_ = true;
                    lda_ = d->k();
                }
            }

            // Pad leading dimensions in case of a single row/column.
            if ((d->k() == 1 && !trans_a()) || (m == 1 && trans_a())) {
                lda_ = utils::rnd_up(lda_, 16);
            }

            if ((n == 1 && !trans_b()) || (d->k() == 1 && trans_b())) {
                ldb_ = utils::rnd_up(ldb_, 16);
            }

            if (swap_ab_) std::swap(m, n);

            // Check parameters.
            if (utils::one_of(d->c_type(), s32, f16, bf16, f32, u8, s8)
                    && utils::one_of(d->a_type(), u8, s8, u4, s4, u2, u3)) {
                VDISPATCH_GEMM(
                        (utils::one_of(d->b_type(), u8, s8) || wei_decomp_),
                        VERBOSE_UNSUPPORTED_DT);

                VDISPATCH_GEMM(IMPLICATION(utils::one_of(d->c_type(), f32, s8,
                                                   u8, f16, bf16),
                                       arch_ >= arch_t::xe_hp),
                        VERBOSE_ISA_DT_MISMATCH);
            } else if (utils::one_of(d->a_type(), f16, bf16)) {
                VDISPATCH_GEMM(d->b_type() == d->a_type(),
                        VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_GEMM(utils::one_of(d->c_type(), d->a_type(), f32,
                                       f8_e5m2, f8_e4m3),
                        VERBOSE_INCONSISTENT_DT, "a", "c");
                VDISPATCH_GEMM(utils::one_of(d->acc_type, d->a_type(), f32),
                        VERBOSE_INCONSISTENT_DT, "a", "acc");
            } else if (!wei_decomp_) {
                VDISPATCH_GEMM(utils::one_of(d->a_type(), f64, f32, f16, bf16,
                                       f8_e5m2, f8_e4m3, f4_e2m1),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(
                        (d->b_type() == d->a_type()
                                || (utils::one_of(d->a_type(), f8_e5m2, f8_e4m3)
                                        && utils::one_of(
                                                d->b_type(), f8_e5m2, f8_e4m3))
                                || (d->a_type() == f4_e2m1
                                        && d->b_type() == f4_e2m1)),
                        VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_GEMM(utils::one_of(d->acc_type, d->a_type(), f32),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_GEMM(IMPLICATION(utils::one_of(f64, d->a_type(),
                                                   d->b_type()),
                                       dev_info_->has_native(f64)),
                        VERBOSE_UNSUPPORTED_DT);
            }

            VDISPATCH_GEMM(!has_blocks(), VERBOSE_BLOCKING_FAIL, "");
            VDISPATCH_GEMM(
                    batch_dims() <= 4, VERBOSE_BAD_DIM, "batch", batch_dims());
            VDISPATCH_GEMM(intel_engine->mayiuse_ngen_kernels(),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");

            // Do not use `with_bias()` as the bias operation may have been
            // moved into a post-op.
            bool with_bias = d->bias_type() != data_type::undef;
            VDISPATCH_GEMM(utils::one_of(d->bias_type(), data_type::undef, f64,
                                   f32, bf16, f16, f8_e5m2, f8_e4m3)
                            && (d->bias_desc.ndims <= 6) && d->bias_mask() < 8,
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_GEMM(
                    IMPLICATION(with_bias,
                            (d->c_type() != f64 || d->bias_type() == f64)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_GEMM(
                    IMPLICATION(with_sum_ab(),
                            !with_bias
                                    && (attr()->zero_points_.has_default_values(
                                            DNNL_ARG_DST))),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_GEMM(attr()->post_ops_.check_sum_consistency(d->c_type(),
                                   utils::one_of(d->a_type(), s8, u8)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            auto c_kernel_type
                    = jit::convert_dnnl_to_kernel_type(desc_.c_desc.data_type);
            for (int i = 0; i < desc_.c_desc.ndims; i++) {
                auto c_stride = desc_.c_desc.format_desc.blocking.strides[i];
                VDISPATCH_GEMM(
                        IMPLICATION(c_kernel_type.isSubByte(),
                                c_stride == 1
                                        || c_stride % c_kernel_type.perByte()
                                                == 0),
                        VERBOSE_SHAPE_RESTRICTION);
            }

            bool with_binary = (post_ops_.find(binary) != -1)
                    || (post_ops_.find(prelu) != -1);
            bool with_eltwise = (post_ops_.find(eltwise) != -1);

            // Check GPU architecture.
            bool arch_ok = utils::one_of(arch_, arch_t::xe_lp, arch_t::xe_hp,
                    arch_t::xe_hpg, arch_t::xe_hpc, arch_t::xe2, arch_t::xe3);
            arch_ok |= (arch_ >= arch_t::xe3p);

            VDISPATCH_GEMM(arch_ok, VERBOSE_UNSUPPORTED_ARCH, "gpu");
            VDISPATCH_GEMM(IMPLICATION(with_binary, arch_ >= arch_t::xe_hp),
                    VERBOSE_UNSUPPORTED_ARCH, "gpu");

            // Grouped scales break pre-XeHPG kernels due to increased register pressure
            bool A_grouped
                    = 1 < a_quant.group_k && a_quant.group_k < desc()->k();
            bool B_grouped
                    = 1 < b_quant.group_k && b_quant.group_k < desc()->k();
            VDISPATCH_GEMM(IMPLICATION(arch_ == compute::gpu_arch_t::xe_lp,
                                   !(A_grouped || B_grouped)),
                    VERBOSE_UNSUPPORTED_FEATURE, "grouped scales");

            // Size checks for fused reduction kernels.
            if (with_sum_ab()) {
                auto mnk = d->m() * d->n() * d->k();
                if (arch_ == arch_t::xe_hpc && d->a_type() == f32)
                    VDISPATCH_GEMM(
                            (mnk <= 256 * 1024 * 1024), VERBOSE_LARGE_SHAPES);
            }

            // Handle special compute modes.
            kernel_desc_t::compute_mode mode = kernel_desc_t::mode_default;

            if (attr()->mayiconvert(f32, tf32))
                set_mode(mode, kernel_desc_t::mode_tf32);
            if (attr()->mayiconvert(f32, bf16))
                set_mode(mode, kernel_desc_t::mode_bf16x1);
            if (attr()->mayiconvert(f32, f16))
                set_mode(mode, kernel_desc_t::mode_f16x1);
            if (attr()->mayiconvert(f32, f32))
                set_mode(mode, kernel_desc_t::mode_strict);
            if (attr()->deterministic_)
                set_mode(mode, kernel_desc_t::mode_deterministic);
            if (attr()->acc_mode_ == accumulation_mode::relaxed)
                set_mode(mode, kernel_desc_t::mode_relaxed_acc);

            if (wei_decomp_) { set_mode(mode, kernel_desc_t::mode_w_decomp); }

            // GEMM kernels down convert the following parameters to
            // int/uint32_t
            VDISPATCH_GEMM(std::max({m, n, d->k(), d->batch()})
                            <= std::numeric_limits<int32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);
            VDISPATCH_GEMM(
                    std::max({ld(DNNL_ARG_A), ld(DNNL_ARG_B), ld(DNNL_ARG_C)})
                            <= std::numeric_limits<uint32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);

            gemmstone::GEMMProblem problem;
            CHECK(init_GEMMProblem(problem, intel_engine));

            VDISPATCH_GEMM(IMPLICATION(problem.Tc == gemmstone::Type::f64,
                                   !with_eltwise && !with_binary),
                    VERBOSE_UNSUPPORTED_POSTOP);

            if (arch_ >= arch_t::xe3p)
                kernel_desc_.set_efficient_64b(dev_info_->is_efficient_64bit());

            bool kernel_success = false;
            // Kernel override: bare int = Nth candidate after sort (rank);
            // else strategy string.
            int index = -1;
            if (is_dev_mode() && attr()->gpu_attr_) {
                auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
                        attr()->gpu_attr_.get());
                // GEMM_KERNEL would replace the attribute override.
                bool has_env
                        = !gpu_utils::dev_getenv("GEMM_KERNEL", std::string())
                                   .empty();
                VDISPATCH_GEMM(!has_env || gpu_attr->kernel_override().empty(),
                        "kernel override conflicts with GEMM_KERNEL");
                index = kernel_rank(attr());
                if (index < 0)
                    kernel_desc_.set_kernel_override(
                            gpu_attr->kernel_override());
            }
            auto lda = ld(DNNL_ARG_A);
            auto ldb = ld(DNNL_ARG_B);
            if (swap_ab_) std::swap(lda, ldb);
            auto entries = kernel_desc_.select_kernel(*dev_info_, mode, problem,
                    alpha(), beta(), m, n, d->k(), lda, ldb, d->ldc(),
                    d->batch());
            kernel_count_ = (int)entries.size();

            if (index >= 0)
                VDISPATCH_GEMM(index < (int)entries.size(),
                        "kernel index out of range");

            for (size_t entry_idx = 0; entry_idx < entries.size();
                    entry_idx++) {
                auto *entry = entries[entry_idx];
                int g = (int)(entry - jit::catalog().entries);
                if (index >= 0 && (int)entry_idx != index) continue;
                auto skip = [&](const char *why) {
                    gpu_debug() << "[" << lpad(g, 4) << "] skip " << entry_idx
                                << "/" << entries.size() << " " << entry->str()
                                << "," << why;
                };
                kernel_desc_.set_entry(entry);
                kernel_desc_.set_problem(problem);
                auto status = kernel_desc_.finalize();
                // select_kernel can return a strategy that failed in the finalize call
                bool valid = status == status::success;
                if (!valid) skip("Strategy finalization failed.");
                // Global k-parallel kernels don't support post-ops or non-f32/s32
                //   accumulation unless fusion is enabled.
                if (kernel_desc_.driver_info()->kParallel()
                        && !kernel_desc_.driver_info()->fusedPostOps()) {
                    bool po_valid = !non_scale_po_
                            && !(with_sum_ && with_c_scales())
                            && utils::one_of(d->c_type(), f32, s32);
                    if (!po_valid) skip("Invalid post op.");
                    valid &= po_valid;
                }
                // An in-place binary post-op reads C as its right-hand side,
                // thus C must still hold the data it came in with by the time
                // post-ops run. Kernels splitting k between workgroups write
                // partial sums into C - or zero it out beforehand - long before
                // that, leaving the post-op nothing to read.
                if (post_ops_.has_inplace_binary()) {
                    bool c_intact = !kernel_desc_.driver_info()->kParallel()
                            && !kernel_desc_.driver_info()->kParallelVariable();
                    if (!c_intact) skip("In-place post op over k-parallel C.");
                    valid &= c_intact;
                }
                // Limited post-op support for low-precision accumulation.
                if (kernel_desc_.problem()->Tc.size() < 4) {
                    bool need_x32_acc = with_binary
                            || !IMPLICATION(with_sum_, sum_at_begin_);
                    valid &= !need_x32_acc;
                    if (need_x32_acc) skip("Invalid post op.");
                }
                // Ensure kernel can be run deterministically if required.
                if (attr()->deterministic_) {
                    bool deterministic
                            = !kernel_desc_.driver_info()->nondeterministic();
                    valid &= deterministic;
                    if (!deterministic) skip("Non deterministic kernel.");
                }

                if (valid) {
                    auto try_create = [&]() {
                        compute::kernel_t kernel;
                        const auto *intel_engine
                                = utils::downcast<const intel::engine_t *>(
                                        engine);
                        auto key = std::make_shared<
                                trivial_key_container_t<dnnl::impl::gpu::intel::
                                                gemm::jit::gen_nocopy_desc_t>>(
                                kernel_desc_, intel_engine->engine_id());
                        cache_state_t kernel_cache_status;
                        auto verbose
                                = get_verbose(verbose_t::create_profile) >= 1;
                        double start_ms = 0;
                        if (verbose) start_ms = get_msec();
                        status = get_cached_kernels<typename trivial_key_t<
                                dnnl::impl::gpu::intel::gemm::jit::
                                        gen_nocopy_desc_t>::value_type>(
                                std::move(key), intel_engine, kernel,
                                kernel_cache_status);
                        if (verbose && status == status::success) {
                            double duration_ms = get_msec() - start_ms;
                            const char *str
                                    = cache_state2str(kernel_cache_status);
                            VPROF(start_ms, primitive, create, str,
                                    info(engine), duration_ms);
                        }
                        return status;
                    };
                    status = try_create();
                    if (status == status::success) {
                        kernel_success = true;
                        if (kernel_desc_.has_entry())
                            gpu_info() << "[" << lpad(g, 4) << "] use "
                                       << entry_idx << "/" << entries.size()
                                       << (index >= 0 ? " pin" : "") << " "
                                       << kernel_desc_.str();
                        else
                            gpu_info()
                                    << "[override] use " << kernel_desc_.str();
                        break;
                    }
                }
                // Overrides do not depend on the entry.
                if (!kernel_desc_.has_entry()) break;
            }

            VDISPATCH_GEMM(
                    kernel_success, "matching kernel not found in catalog");

            init_scratchpad();

            return status::success;
        }

        status_t query(query_t what, int idx, void *result) const override {
            switch ((int)what) {
                case (int)query::preferred_gpu_grf_per_thread: {
                    *(int *)result = kernel_desc_.driver_info()->grfCount;
                    break;
                }
                case (int)query::gpu_kernel_count: {
                    *(int *)result = kernel_count_;
                    break;
                }
                default: return gemm::pd_t::query(what, idx, result);
            }
            return status::success;
        }

        status_t set_default_formats(bool no_transpose_c) {
            using namespace data_type;
            using namespace format_tag;
            using arch_t = compute::gpu_arch_t;

            auto d = desc();

            auto m = d->m();
            auto n = d->n();
            auto k = d->k();
            auto a_t = (utils::one_of(d->a_type(), s4, u4, u2)) ? s8
                                                                : d->a_type();
            auto b_t = (utils::one_of(d->b_type(), s4, u4, u2)) ? s8
                                                                : d->b_type();
            auto c_t = d->c_type();

            bool is_f16 = utils::everyone_is(f16, a_t, b_t, c_t);
            bool is_bf16 = utils::everyone_is(bf16, a_t, b_t, c_t);
            bool is_xe_hp_plus = arch_ >= arch_t::xe_hp;

            // Rename memory descriptors following column major format.
            auto &a_desc = desc_.b_desc;
            auto &b_desc = desc_.a_desc;
            auto &c_desc = desc_.c_desc;

            memory_desc_wrapper a_mdw(&a_desc);
            memory_desc_wrapper b_mdw(&b_desc);
            memory_desc_wrapper c_mdw(&c_desc);

            bool a_any = a_mdw.format_any();
            bool b_any = b_mdw.format_any();
            bool c_any = c_mdw.format_any();

            if (!a_any && !is_md_gemm_compatible_plain_format(&a_desc))
                return status::unimplemented;
            if (!b_any && !is_md_gemm_compatible_plain_format(&b_desc))
                return status::unimplemented;
            if (!c_any
                    && !is_md_gemm_compatible_plain_format(
                            &c_desc, no_transpose_c))
                return status::unimplemented;

            bool is_a_trans = (desc()->transa() == dnnl_trans);
            bool is_b_trans = (desc()->transb() == dnnl_trans);

            auto lda = is_a_trans ? m : k;
            auto ldb = is_b_trans ? k : n;

            auto is_aligned = [](dim_t ld, data_type_t dt, int byte) {
                return types::elements_to_bytes(dt, ld) % byte == 0;
            };

            bool a_4B_aligned = is_aligned(lda, a_t, 4);
            bool b_4B_aligned = is_aligned(ldb, b_t, 4);
            bool ab_4B_aligned = a_4B_aligned && b_4B_aligned;

            bool a_tn_4B_aligned = is_aligned(k, a_t, 4);
            bool b_tn_4B_aligned = is_aligned(k, b_t, 4);
            bool ab_tn_4B_aligned = a_tn_4B_aligned && b_tn_4B_aligned;

            bool use_tn = (m <= 32 || n <= 32) && !ab_4B_aligned
                    && ab_tn_4B_aligned;

            bool batch = d->is_batched();

            auto dotrans = batch ? acb : ba;
            auto notrans = batch ? abc : ab;

            auto cache_line_align_md = [&](memory_desc_t &md) {
                dnnl::impl::dims_t dims;
                dnnl::impl::utils::array_copy(dims, md.dims, md.ndims);

                auto kernel_type
                        = jit::convert_dnnl_to_kernel_type(md.data_type);
                size_t stride = [&](dim_t dim) {
                    auto stride = dim * kernel_type;

                    // Prefer cache line aligned sizes
                    if (stride > 32) {
                        stride = utils::rnd_up(stride, 64);
                        // Avoid conflicts in 8-way associative cache
                        if (stride % 256 == 0) stride += 64;
                        return stride / kernel_type;
                    }

                    // Optimal stride for data loading, determined by restrictions
                    // on loads.
                    int load_alignment = arch_ > arch_t::xe2 ? 16 : 4;
                    if (stride > load_alignment / 2)
                        return utils::rnd_up(stride, load_alignment)
                                / kernel_type;

                    // Limit padding for small dimensions
                    return utils::rnd_up_pow2(stride) / kernel_type;
                }(md.dims[md.ndims - 1]);

                dnnl::impl::dims_t strides;
                strides[md.ndims - 1] = 1;
                strides[md.ndims - 2] = stride;
                for (int i = md.ndims - 3; i >= 0; i--)
                    strides[i] = strides[i + 1] * dims[i + 1];

                CHECK(memory_desc_init_by_strides(
                        md, md.ndims, dims, md.data_type, strides));
                return status::success;
            };
            // u3 requires the natural Block2DTranspose layout;
            // cache_line_align_md assumes a whole-byte type.
            if (a_any && a_desc.data_type != u3)
                CHECK(cache_line_align_md(a_desc));
            if (b_any && b_desc.data_type != u3)
                CHECK(cache_line_align_md(b_desc));

            if ((is_f16 || is_bf16) && is_xe_hp_plus && use_tn) {
                if (a_any && b_any) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                } else if (a_any && !is_b_trans) {
                    CHECK(memory_desc_init_by_tag(a_desc, dotrans));
                } else if (b_any && is_a_trans) {
                    CHECK(memory_desc_init_by_tag(b_desc, notrans));
                }
            }

            return gemm::pd_t::set_default_formats() ? status::success
                                                     : status::unimplemented;
        }

        void init_scratchpad() {
            using namespace gemmstone;
            const auto *info = kernel_desc()->driver_info();
            if (info->needsTempC()) {
                auto scratchpad = scratchpad_registry().registrar();

                int temp_c_sz = nstl::max(
                        (int)types::data_type_size(desc()->c_type()), 4);
                int temp_c_elems = info->wgTile(LoopM) * info->wgTile(LoopN);
                if (with_sum_ab())
                    temp_c_elems += nstl::max(
                            info->wgTile(LoopM), info->wgTile(LoopN));
                temp_c_elems = utils::rnd_up(temp_c_elems, 64);
                temp_c_elems *= max_k_sliced_groups();

                scratchpad.book(memory_tracking::names::key_gemm_accumulator,
                        temp_c_elems, temp_c_sz, 64, 65536);
            }
            if (prepare_zero_buffer_scratchpad()
                    && (info->fusedBeta() || info->fusedPostOps())) {
                auto scratchpad = scratchpad_registry().registrar();
                int zg_cl = 0;
                if (info->fusedBeta()) zg_cl++;
                if (info->fusedPostOps()) zg_cl++;
                size_t zero_buffer_bytes = max_k_sliced_groups() * 64 * zg_cl;
                scratchpad.book(memory_tracking::names::key_gemm_zero_buffer,
                        zero_buffer_bytes, 1, 64, 65536);
            }
        }

        const jit::gen_nocopy_desc_t *kernel_desc() const {
            return &kernel_desc_;
        }

        int max_k_sliced_groups() const {
            const auto *info = kernel_desc()->driver_info();

            auto groups = dev_info_->hw_threads(info->grfCount)
                    / (info->wg[gemmstone::LoopM] * info->wg[gemmstone::LoopN]);
            if (info->kParallelVariable()) groups *= 2;

            return groups;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;

        const compute::device_info_t *dev_info_ = nullptr;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        kernel_desc_t kernel_desc_;
        int kernel_count_ = 0;
    };

    gen_t(const pd_t *apd) : primitive_t(apd) {}

    ~gen_t() override {
        if (zero_pool_) release_zero_pool(zero_pool_);
    }

    status_t init(impl::engine_t *engine) override {
        return init_nocopy(engine);
    }

    status_t init_nocopy(impl::engine_t *engine) {
        using namespace data_type;
        auto kd = pd()->kernel_desc();

        CHECK(create_kernel(engine, nocopy_kernel_, "gemm_kernel", *kd));

        scalar_type_ = kd->scalar_type();
        const auto *info = nocopy_info();

        if (need_zero_buffer()) {
            int zg_cl = 0;
            if (info->fusedBeta()) zg_cl++;
            if (info->fusedPostOps()) zg_cl++;

            zero_buffer_bytes_ = pd()->max_k_sliced_groups() * 64 * zg_cl;

            if (use_zero_pool()) {
                auto zg_max = pd()->dev_info_->hw_threads();
                zero_pool_chunk_size_ = zg_max * 2 * 2 * 64;

                auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
                CHECK(lookup_zero_pool(
                        intel_engine, zero_pool_chunk_size_, &zero_pool_));

                nocopy_kernel_.save_output_events();
            }
            if (prepare_zero_buffer_scratchpad()) {
                zero_fill_params_t params;
                params.grf_256 = (info->grfCount == 256);
                // IFP on (default) forces round-robin thread arbitration.
                // Disable it on a clear mismatch with the main kernel to
                // avoid a thread arbitration switch between dispatches.
                params.no_subgroup_ifp = (kd->strategy()->arbitrationMode
                        != ngen::ThreadArbitrationMode::RoundRobin);
                CHECK(create_kernel(
                        engine, zero_fill_kernel_, "gemm_zero_fill", params));
            }
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t launch_nocopy(const exec_ctx_t &ctx, intel::stream_t *s,
            zero_pool_t *zero_pool, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            const memory_storage_t *ao, const memory_storage_t *bo,
            int16_t ao_host_scalar, int16_t bo_host_scalar,
            const memory_storage_t *a_scales, const memory_storage_t *b_scales,
            const memory_storage_t *c_scales, const memory_storage_t *ag,
            const memory_storage_t *bg, const memory_storage_t &co,
            int16_t co_host_scalar, const memory_storage_t *c_temp,
            const memory_storage_t *zero_buf_scratchpad,
            const memory_storage_t *sround_seed, int po_count,
            const memory_storage_t **po_src, int64_t offset_a, int64_t offset_b,
            int64_t offset_c, int64_t offset_ao, int64_t offset_bo,
            int64_t offset_aq, int64_t offset_bq, int64_t offset_co,
            int64_t *offset_po_src, int32_t lda, int32_t ldb, int32_t ldc,
            int32_t ldaq, int32_t ldbq, int32_t m, int32_t n, int32_t k,
            int32_t k0, float alpha, float beta, int32_t cmask,
            bool last_k_block, bool swap_ab, bool disable_hilbert) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    const gemmstone::CommonDriverInfo *nocopy_info() const {
        return pd()->kernel_desc()->driver_info();
    }

    bool need_zero_buffer() const {
        return nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps();
    }

    compute::kernel_t nocopy_kernel_;
    compute::kernel_t zero_fill_kernel_;
    compute::scalar_type_t scalar_type_;
    zero_pool_t *zero_pool_ = nullptr;
    size_t zero_buffer_bytes_ = 0;
    size_t zero_pool_chunk_size_ = 0;
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
