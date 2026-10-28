/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "common/dnnl_thread.hpp"

#include "cpu/x64/jit_uni_reorder_direct_copy.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

using namespace Xbyak;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Vmm>
struct direct_copy_kernel_t
    : public jit_uni_reorder_direct_copy_t::kernel_base_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(direct_copy_kernel_t)

    direct_copy_kernel_t(const reorder_pd_t *pd, cpu_isa_t isa)
        : jit_uni_reorder_direct_copy_t::kernel_base_t(pd)
        , jit_generator_t(jit_name(), isa)
        , isa_(isa)
        , src_dt_(pd_->src_md()->data_type)
        , dst_dt_(pd_->dst_md()->data_type) {
        assert(!utils::one_of(isa_, isa_undef, isa_all));

        const memory_desc_wrapper src_d(pd_->src_md());

        tail_size_ = static_cast<int>(src_d.nelems() % simd_w_);

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(
                simd_w_, tail_size_, tail_opmask_idx_, tail_vmm_idx_, reg_tmp_);
        io::io_emu_bf16_conf_t io_bf16_conf(emu_zmm_1_idx_, emu_zmm_2_idx_,
                emu_zmm_3_idx_, reg_tmp_, emu_zmm_4_idx_);
        io::io_emu_fp8_conf_t io_fp8_conf(emu_zmm_1_idx_, emu_zmm_2_idx_,
                emu_zmm_3_idx_, emu_zmm_4_idx_, emu_zmm_5_idx_,
                emu_kmask_aux_idx_, reg_tmp_);
        io::io_saturation_conf_t io_saturation_conf(
                zero_idx_, saturation_ubound_idx_, reg_tmp_);
        // e8m0 conversions reuse the fp8 emulation scratch (they never
        // coexist): a scratch vector register, a scratch opmask and a gpr.
        io::io_emu_e8m0_conf_t io_e8m0_conf(
                emu_zmm_1_idx_, emu_kmask_aux_idx_, reg_tmp_);

        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, isa_, {src_dt_, dst_dt_},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_dt_, io_saturation_conf}}, utils::nullopt, io_fp8_conf,
                io_e8m0_conf);
    }

    static constexpr int vlen_ = vreg_traits_t<Vmm>::vlen;
    static constexpr int simd_w_ = vlen_ / sizeof(float);
    static constexpr int unroll_12_ = 12;
    static constexpr int unroll_4_ = 4;

    void operator()(
            const void *src, void *dst, dim_t work_amount) const override {
        ker_args_t args;
        args.src = src;
        args.dst = dst;
        args.work_amount = work_amount;
        jit_generator_t::operator()(&args);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

    Address src_ptr(int offt = 0) { return ptr[reg_src + offt]; }

    Address dst_ptr(int offt = 0) { return ptr[reg_dst + offt]; }

    Vmm vmm_src(int idx) const {
        // Incorporate `+ 1` here as `0th` index is used for tail on AVX2.
        return Vmm(idx + 1);
    }

    void copy(const int unroll, const bool tail) {
        const int src_dt_size
                = static_cast<int>(types::data_type_size(src_dt_));
        const int dst_dt_size
                = static_cast<int>(types::data_type_size(dst_dt_));

        // Copy two simdw at once in vectorized loop first when `ne_convert`
        // instructions are available for xf16.
        if (isa_ == avx2_vnni_2
                && (utils::one_of(src_dt_, data_type::bf16, data_type::f16))
                && (unroll % 2 == 0)) {
            for (int i = 0; i < unroll / 2; i++) {
                const Vmm &vmm_src_even = vmm_src(2 * i);
                const Vmm &vmm_src_odd = vmm_src(2 * i + 1);
                Vmm vmm_tmp(vmm_tmp_idx_);
                io_[src_dt_]->load_two_simdw_xf16(
                        src_ptr(2 * i * src_dt_size * simd_w_), vmm_src_even,
                        vmm_src_odd);
                io_[src_dt_]->merge_interleaved_to_plain(
                        vmm_src_even, vmm_src_odd, vmm_tmp);
                io_[dst_dt_]->store(vmm_src_even,
                        dst_ptr(2 * i * dst_dt_size * simd_w_), tail);
                io_[dst_dt_]->store(vmm_src_odd,
                        dst_ptr((2 * i + 1) * dst_dt_size * simd_w_), tail);
            }
        } else {
            for (int i = 0; i < unroll; i++) {
                io_[src_dt_]->load(
                        src_ptr(i * src_dt_size * simd_w_), vmm_src(i), tail);
            }
            for (int i = 0; i < unroll; i++) {
                io_[dst_dt_]->store(
                        vmm_src(i), dst_ptr(i * dst_dt_size * simd_w_), tail);
            }
        }

        if (tail) return;

        add(reg_src, unroll * src_dt_size * simd_w_);
        add(reg_dst, unroll * dst_dt_size * simd_w_);
        sub(reg_work_amount, unroll * simd_w_);
    }

    void generate() override {
        preamble();

        if (tail_size_) io_.prepare_tail_mask();
        if (is_bf16()) io_.init_bf16();
        io_.init_saturate_f32({dst_dt_});

        Reg64 param = abi_param1;
#define PARAM_OFF(x) offsetof(ker_args_t, x)
        mov(reg_src, ptr[param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[param + PARAM_OFF(dst)]);
        mov(reg_work_amount, ptr[param + PARAM_OFF(work_amount)]);
#undef PARAM_OFF

        Label unroll_12_start, unroll_4_start, full_vector_start, tail_start,
                end;

        L(unroll_12_start);
        {
            cmp(reg_work_amount, unroll_12_ * simd_w_);
            jl(unroll_4_start, T_NEAR);

            copy(unroll_12_, false);

            jmp(unroll_12_start, T_NEAR);
        }

        L(unroll_4_start);
        {
            cmp(reg_work_amount, unroll_4_ * simd_w_);
            jl(full_vector_start, T_NEAR);

            copy(unroll_4_, false);

            jmp(unroll_4_start, T_NEAR);
        }

        L(full_vector_start);
        {
            cmp(reg_work_amount, simd_w_);
            jl(tail_start, T_NEAR);

            copy(1, false);

            jmp(full_vector_start, T_NEAR);
        }

        L(tail_start);
        {
            cmp(reg_work_amount, 0);
            jle(end, T_NEAR);

            copy(1, true);
        }
        L(end);

        postamble();

        if (is_f8()) io_.prepare_table_fp8();
    }

private:
    struct ker_args_t {
        const void *src;
        void *dst;
        dim_t work_amount;
    };

    bool is_bf16() const {
        return utils::one_of(data_type::bf16, src_dt_, dst_dt_);
    }

    bool is_f16() const {
        return utils::one_of(data_type::f16, src_dt_, dst_dt_);
    }

    bool is_f8() const {
        return utils::one_of(data_type::f8_e4m3, src_dt_, dst_dt_)
                || utils::one_of(data_type::f8_e5m2, src_dt_, dst_dt_);
    }

    cpu_isa_t isa_;
    data_type_t src_dt_, dst_dt_;
    io::jit_io_multi_dt_helper_t<Vmm> io_;
    int tail_size_;

    const Reg64 reg_tmp_ = rax;
    const Reg64 reg_src = r8;
    const Reg64 reg_dst = r9;
    const Reg64 reg_work_amount = r10;

    const int tail_opmask_idx_ = 1;

    const int tail_vmm_idx_ = 0;
    // Indices from 1 to 12 are occupied in the unrolled body.
    const int vmm_tmp_idx_ = 13;
    const int zero_idx_ = 14;
    const int saturation_ubound_idx_ = 15;
    const int emu_zmm_1_idx_ = 27;
    const int emu_zmm_2_idx_ = 28;
    const int emu_zmm_3_idx_ = 29;
    const int emu_zmm_4_idx_ = 30;
    const int emu_zmm_5_idx_ = 31;
    const int emu_kmask_aux_idx_ = 2;
};

status_t jit_uni_reorder_direct_copy_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace data_type;

    VDISPATCH_REORDER(is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_REORDER(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
    isa_ = get_max_cpu_isa();

    const auto src_dt = src_md()->data_type;
    const auto dst_dt = dst_md()->data_type;
    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper dst_d(dst_md());
    const auto blocks_size = src_d.blk_size();

    VDISPATCH_REORDER(!src_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER(!dst_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER(
            src_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);
    VDISPATCH_REORDER(
            dst_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);

    // Note: io_helper has an implicit conversion to f32 which is incorrect for
    // s32->s32. Disabling it for now.
    const bool is_s32 = utils::everyone_is(s32, src_dt, dst_dt);
    VDISPATCH_REORDER(!is_s32, VERBOSE_UNSUPPORTED_DT);

    const bool is_f64 = utils::one_of(f64, src_dt, dst_dt);
    VDISPATCH_REORDER(!is_f64, VERBOSE_UNSUPPORTED_DT);

    const bool is_s64 = utils::one_of(s64, src_dt, dst_dt);
    VDISPATCH_REORDER(!is_s64, VERBOSE_UNSUPPORTED_DT);

    // Sub-byte data types require nibble-packed addressing which the direct
    // copy kernel doesn't support.
    const bool is_subbyte = types::data_type_bits(src_dt) < 8
            || types::data_type_bits(dst_dt) < 8;
    VDISPATCH_REORDER(!is_subbyte, VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_REORDER(IMPLICATION(utils::one_of(bf16, src_dt, dst_dt),
                              mayiuse(avx512_core) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_REORDER(
            IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                    mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_REORDER(
            IMPLICATION(utils::one_of(bf16, src_dt, dst_dt) && blocks_size < 8,
                    mayiuse(avx512_core_bf16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_REORDER(
            IMPLICATION(utils::one_of(f16, src_dt, dst_dt) && blocks_size < 8,
                    mayiuse(avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);

    const bool is_f8 = utils::one_of(f8_e4m3, src_dt, dst_dt)
            || utils::one_of(f8_e5m2, src_dt, dst_dt);
    VDISPATCH_REORDER(IMPLICATION(is_f8, mayiuse(avx512_core_amx)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_REORDER(IMPLICATION(is_f8,
                              !utils::one_of(src_dt, u8, s8)
                                      && !utils::one_of(dst_dt, u8, s8)),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // Gate e8m0 behind same ISA as FP8 data types.
    const bool is_e8m0 = utils::one_of(e8m0, src_dt, dst_dt);
    VDISPATCH_REORDER(IMPLICATION(is_e8m0, mayiuse(avx512_core_amx)),
            VERBOSE_ISA_DT_MISMATCH);

    // Direct copy operates only on identical formats.
    VDISPATCH_REORDER(src_d.similar_to(dst_d, true, false, 0),
            VERBOSE_TENSOR_FORMAT_MISMATCH, "src", "dst");

    // Direct copy doesn't operate on plain non-dense cases, though it can
    // simply extend nelems to contain wholes in strides but it would be
    // inefficient.
    // TODO: enable smart plain non-dense case support.
    VDISPATCH_REORDER(IMPLICATION(src_d.is_plain(), src_d.is_dense()),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    VDISPATCH_REORDER(src_d.extra().flags == dst_d.extra().flags,
            VERBOSE_UNSUPPORTED_MD_FLAG, "src or dst");

    VDISPATCH_REORDER(IMPLICATION(src_d.extra().flags > 0UL,
                              src_d.additional_buffer_size()
                                      == dst_d.additional_buffer_size()),
            VERBOSE_UNSUPPORTED_MD_FLAG, "src or dst");

    VDISPATCH_REORDER(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER(IMPLICATION(!src_d.is_plain(), blocks_size >= 8),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    return status::success;
}

jit_uni_reorder_direct_copy_t::kernel_base_t *
jit_uni_reorder_direct_copy_t::kernel_base_t::create(
        const reorder_pd_t *pd, cpu_isa_t isa) {
    // Reorder must support blocked formats such as aBx8b.
    // These variables will help to dispatch smaller blocks into proper kernels.
    const bool has_blocks = !memory_desc_wrapper(pd->src_md()).is_plain();
    const auto blocks_size = memory_desc_wrapper(pd->src_md()).blk_size();

    if (is_superset(isa, avx512_core)
            && IMPLICATION(has_blocks, blocks_size >= 16)) {
        return new direct_copy_kernel_t<Zmm>(pd, isa);
    } else if (is_superset(isa, avx2)
            && IMPLICATION(has_blocks, blocks_size >= 8)) {
        return new direct_copy_kernel_t<Ymm>(pd, isa);
    } else {
        assert(!"unexpected");
    }
    return nullptr;
}

status_t jit_uni_reorder_direct_copy_t::init(engine_t *engine) {
    const auto isa = pd()->isa_;
    CHECK(safe_ptr_assign(kernel_, kernel_base_t::create(pd(), isa)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_direct_copy_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto src_dt_size = src_d.data_type_size();
    const auto dst_dt_size = dst_d.data_type_size();
    const auto nelems = src_d.nelems(true);
    const int simd_w = isa_max_vlen(pd()->isa_) / sizeof(float);

    // Spawning threads costs more than copying a small buffer: up to 32 KiB,
    // a single thread is faster even with cold caches.
    constexpr dim_t max_seq_bytes = 32 * 1024;
    const dim_t nbytes
            = nelems * static_cast<dim_t>(nstl::max(src_dt_size, dst_dt_size));
    const int nthr = nbytes <= max_seq_bytes ? 1 : 0;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        (*kernel_)(in + (start + src_d.offset0()) * src_dt_size,
                out + (start + dst_d.offset0()) * dst_dt_size, end - start);
    });

    if (src_d.is_additional_buffer()) {
        // Verified in pd_t::init();
        assert(src_d.extra().flags == dst_d.extra().flags);

        const auto additional_size = src_d.additional_buffer_size();
        const auto data_size = src_d.size(/* index = */ 0,
                /* include_additional_size = */ false);
        std::memcpy(out + data_size * dst_dt_size, in + data_size * src_dt_size,
                additional_size);
    }

    return status::success;
}

template struct direct_copy_kernel_t<Zmm>;
template struct direct_copy_kernel_t<Ymm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
