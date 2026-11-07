/*******************************************************************************
* Copyright 2017 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
* Copyright 2022, 2025-2026 Arm Ltd. and affiliates
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_uni_eltwise.hpp"

#include <cstdint>

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_t : public jit_generator_t {
    jit_uni_eltwise_kernel_t(const eltwise_pd_t *pd) : pd_(pd) {}

    void operator()(jit_args_t *p) { jit_generator_t::operator()(p); }

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const {
        return pd_->use_dst() ? pd_->dst_md()->data_type
                              : pd_->src_md()->data_type;
    }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    bool is_f16() const { return data_type() == data_type::f16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
    // Simple operations can be done in F16 without conversion to/from F32
    bool can_compute_as_f16() const {
        const auto &desc = *pd_->desc();
        return pd_->is_fwd()
                && (((is_f16() || is_bf16()) // if F16 or BF16 data type
                            // ABS and non-leaky ReLU depend only on sign bit
                            // Therefore, BF16 values can also be computed as F16
                            && ((desc.alg_kind == alg_kind::eltwise_relu
                                        && desc.alpha == 0.0f)
                                    || desc.alg_kind == alg_kind::eltwise_abs))
                        || (is_f16() // only if F16 data type
                                && utils::one_of(desc.alg_kind,
                                        alg_kind::eltwise_clip,
                                        alg_kind::eltwise_clip_v2,
                                        alg_kind::eltwise_square,
                                        alg_kind::eltwise_sqrt)));
    }
};

// jit kernels
namespace {

template <cpu_isa_t isa>
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd) : jit_uni_eltwise_kernel_t(pd) {}

    void generate() override {
        auto reg_src = alloc_reg<XReg>();
        auto reg_dst = alloc_reg<XReg>();
        auto reg_injector_table = alloc_reg<XReg>();
        auto reg_diff_dst = alloc_reg<XReg>();
        auto reg_work_amount = alloc_reg<XReg>();
        auto injector_mask = alloc_reg<PReg>();

        const auto &desc = *pd_->desc();
        const bool is_fwd = pd_->is_fwd();
        jit_uni_eltwise_injector_t<isa> eltwise_injector(this, reg_pool(),
                desc.alg_kind, desc.alpha, desc.beta, 1.f, reg_injector_table,
                injector_mask, is_fwd, pd_->use_dst(),
                can_compute_as_f16() ? data_type::f16 : data_type::f32);

        // Note: load type may not the same as compute type
        const auto simd_elems_per_load = simd_elems(data_type(), isa);

        preamble();
        XReg param = param1;
        add_imm(X_TMP_0, param, GET_OFF(src), X_TMP_1);
        ldr(reg_src, ptr(X_TMP_0));
        add_imm(X_TMP_0, param, GET_OFF(dst), X_TMP_1);
        ldr(reg_dst, ptr(X_TMP_0));
        if (!is_fwd) {
            add_imm(X_TMP_0, param, GET_OFF(diff_dst), X_TMP_1);
            ldr(reg_diff_dst, ptr(X_TMP_0));
        }
        add_imm(X_TMP_0, param, GET_OFF(work_amount), X_TMP_1);
        ldr(reg_work_amount, ptr(X_TMP_0));
        eltwise_injector.load_table_addr();
        Label vectorized_unrolled_loop_start, vectorized_loop_check,
                vectorized_loop_start, remainder_loop_start, remainder_loop_end;
        static constexpr int unroll_factor = 4;
        const auto unrolled_simd_elems = unroll_factor * simd_elems_per_load;
        const auto unrolled_simd_bytes = unroll_factor * simd_bytes(isa);
        if (is_fwd) {
            auto src_vmms = alloc_regs<TReg, unroll_factor>();

            cmp(reg_work_amount, unrolled_simd_elems);
            b(LT, vectorized_loop_check);
            L(vectorized_unrolled_loop_start);
            injector_utils::vmm_index_set_t vmm_idxs;
            // Unrolled load
            for (int i = 0; i < unroll_factor; ++i)
                load_vector(src_vmms[i].get().s, reg_src, i);

            if (can_compute_as_f16()) {
                for (int i = 0; i < unroll_factor; ++i)
                    vmm_idxs.emplace(src_vmms[i].getIdx());
                eltwise_injector.compute_vector_range(vmm_idxs);
            } else if (is_bf16()) {
                auto aux_vmms = alloc_regs<TReg, unroll_factor>();
                for (int i = 0; i < unroll_factor; ++i) {
                    unpack_bf16(src_vmms[i], aux_vmms[i]);
                    vmm_idxs.emplace(src_vmms[i].getIdx());
                    vmm_idxs.emplace(aux_vmms[i].getIdx());
                }
                eltwise_injector.compute_vector_range(vmm_idxs);
                for (int i = 0; i < unroll_factor; ++i)
                    pack_bf16(src_vmms[i], aux_vmms[i]);
            } else if (is_f16()) {
                auto aux_vmms = alloc_regs<TReg, unroll_factor>();
                for (int i = 0; i < unroll_factor; ++i) {
                    unpack_fp16(src_vmms[i], aux_vmms[i]);
                    vmm_idxs.emplace(src_vmms[i].getIdx());
                    vmm_idxs.emplace(aux_vmms[i].getIdx());
                }
                eltwise_injector.compute_vector_range(vmm_idxs);
                for (int i = 0; i < unroll_factor; ++i)
                    pack_fp16(src_vmms[i], aux_vmms[i]);
            } else { // F32
                for (int i = 0; i < unroll_factor; ++i)
                    vmm_idxs.emplace(src_vmms[i].getIdx());
                eltwise_injector.compute_vector_range(vmm_idxs);
            }
            // Unrolled store
            for (int i = 0; i < unroll_factor; ++i)
                store_vector(reg_dst, src_vmms[i].get().s, i);

            add_imm(reg_src, reg_src, unrolled_simd_bytes, X_TMP_0);
            add_imm(reg_dst, reg_dst, unrolled_simd_bytes, X_TMP_0);
            sub_imm(reg_work_amount, reg_work_amount, unrolled_simd_elems,
                    X_TMP_0);
            cmp(reg_work_amount, unrolled_simd_elems);
            b(GE, vectorized_unrolled_loop_start);
        }
        L(vectorized_loop_check);
        cmp(reg_work_amount, simd_elems_per_load);
        b(LT, remainder_loop_start);
        L(vectorized_loop_start);

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.

        auto vmm_src = alloc_reg<TReg>();
        VReg4S xmm_src {vmm_src.getIdx()};
        VReg8H v_bf16 {vmm_src.getIdx()};
        VReg8H v_f16 {vmm_src.getIdx()};

        load_vector(vmm_src.get().s, reg_src);
        if (can_compute_as_f16()) {
            // For F16-computable algorithms, we keep the data in 16-bit format
            // throughout the computation, avoiding the need for conversion to/from F32.
            eltwise_injector.compute_vector(vmm_src.getIdx());
        } else if (is_bf16()) {
            // Convert BF16 input to F32, apply eltwise op, then convert back to BF16:
            // - unpack BF16 to F32 by zero-extending
            // - compute eltwise alg in F32
            // - down convert back to BF16 using bfcvt, and pack result
            auto tmp0 = alloc_reg<TReg>();
            unpack_bf16(vmm_src, tmp0);
            eltwise_injector.compute_vector_range(
                    {vmm_src.getIdx(), tmp0.getIdx()});
            pack_bf16(vmm_src, tmp0);
        } else if (is_f16()) {
            // Convert F16 to F32, apply eltwise op, then convert back to F16:
            // - upcast F16 to F32 using fcvt
            // - compute eltwise alg in F32
            // - downcast F32 back to F16 using fcvt, and pack result
            auto tmp0 = alloc_reg<TReg>();
            unpack_fp16(vmm_src, tmp0);
            eltwise_injector.compute_vector_range(
                    {vmm_src.getIdx(), tmp0.getIdx()});
            pack_fp16(vmm_src, tmp0);
        } else { // F32
            eltwise_injector.compute_vector(vmm_src.getIdx());
            if (!is_fwd) {
                auto vmm_diff_dst = alloc_reg<TReg>();
                load_vector(vmm_diff_dst.get().s, reg_diff_dst);
                fmul(vmm_src.get().s, vmm_src.get().s, vmm_diff_dst.get().s);
            }
        }

        store_vector(reg_dst, vmm_src.get().s);
        // Update pointers for the next iteration
        // Note: we use X_TMP_0 as a temporary register to avoid conflicts with
        // other registers.
        add_imm(reg_src, reg_src, simd_bytes(isa), X_TMP_0);
        add_imm(reg_dst, reg_dst, simd_bytes(isa), X_TMP_0);
        if (!is_fwd)
            add_imm(reg_diff_dst, reg_diff_dst, simd_bytes(isa), X_TMP_0);

        sub_imm(reg_work_amount, reg_work_amount, simd_elems_per_load, X_TMP_0);
        cmp(reg_work_amount, simd_elems_per_load);
        b(GE, vectorized_loop_start);

        // tail processing
        L(remainder_loop_start);

        cmp(reg_work_amount, 0);
        b(LE, remainder_loop_end);

        if (can_compute_as_f16()) {
            ld1(v_f16[0], ptr(reg_src));
            eltwise_injector.compute_vector(vmm_src.getIdx());
        } else if (is_bf16()) {
            ld1(v_bf16[0], ptr(reg_src));
            auto tmp0 = alloc_reg<TReg>();
            unpack_bf16(vmm_src, tmp0);
            eltwise_injector.compute_vector(vmm_src.getIdx());
            pack_bf16(vmm_src, tmp0);
        } else if (is_f16()) {
            ld1(v_f16[0], ptr(reg_src));
            auto tmp0 = alloc_reg<TReg>();
            unpack_fp16(vmm_src, tmp0);
            eltwise_injector.compute_vector(vmm_src.getIdx());
            pack_fp16(vmm_src, tmp0);
        } else {
            ld1(xmm_src[0], ptr(reg_src));
            eltwise_injector.compute_vector(xmm_src.getIdx());
            if (!is_fwd) {
                auto vmm_diff_dst = alloc_reg<TReg>();
                ld1(VReg4S(vmm_diff_dst.getIdx())[0], ptr(reg_diff_dst));
                fmul(TRegS(vmm_src.getIdx()), TRegS(vmm_src.getIdx()),
                        TRegS(vmm_diff_dst.getIdx()));
            }
        }

        if (is_bf16()) {
            st1(v_bf16[0], ptr(reg_dst));
        } else if (is_f16()) {
            st1(v_f16[0], ptr(reg_dst));
        } else {
            st1(xmm_src[0], ptr(reg_dst));
        }

        add_imm(reg_src, reg_src, dtype_size(), X_TMP_0);
        add_imm(reg_dst, reg_dst, dtype_size(), X_TMP_0);
        if (!is_fwd) add_imm(reg_diff_dst, reg_diff_dst, dtype_size(), X_TMP_0);
        subs(reg_work_amount, reg_work_amount, 1);

        b(remainder_loop_start);

        L(remainder_loop_end);

        postamble();

        eltwise_injector.prepare_table();
    }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    /**
     * @brief Load a vector from memory into a SIMD/SVE register.
     *
     * - **ASIMD (NEON)**: Loads a full-width fixed SIMD vector from the specified address.
     * - **SVE**:
     *   - Uses a scalable vector length.
     *   - Utilizes a predicate register (P register) to control active lanes.
     *   - Defaults to `PTRUE`, i.e., all lanes are active by default.
     *
     * @param dst  Destination SIMD/SVE register to receive the data.
     * @param addr Source memory address (base register) with aligned vector data.
     */
    void load_vector(const TRegS &dst, const XReg addr, int vl_offset = 0);

    /**
     * @brief Store a SIMD/SVE vector register to memory.
     *
     * - **ASIMD (NEON)**: Stores the full-width fixed SIMD vector to the specified address.
     * - **SVE**:
     *   - Uses a scalable vector length.
     *   - Utilizes a predicate register (P register) to control active lanes.
     *   - Defaults to `PTRUE`, i.e., all lanes are active by default.
     *
     * @param addr Destination memory address (base register) where data will be stored.
     * @param src  Source SIMD/SVE register providing the data.
     */
    void store_vector(const XReg &addr, const TRegS src, int vl_offset = 0);

    /**
     * @brief Unpack two BFloat16 values from a single register into two target registers.
     *
     * ASIMD (NEON) version: unpack in low/high half order.
     * SVE version: unpack in even/odd element order.
     *
     * @param v0 Destination register for first (low/even) half.
     * @param v1 Destination register for second (high/odd) half.
     */
    void unpack_bf16(const TReg &v0, const TReg &v1);

    /**
     * @brief Pack two BFloat16 values from two registers into a single register.
     *
     * ASIMD (NEON) version: pack from low/high half order.
     * SVE version: pack from even/odd element order.
     *
     * @param v0 Source register providing first (low/even) half.
     * @param v1 Source register providing second (high/odd) half.
     */
    void pack_bf16(const TReg &v0, const TReg &v1);

    /**
     * @brief Unpack two IEEE‑754 FP16 values from a single register into two target registers.
     *
     * ASIMD (NEON) version: unpack in low/high half order.
     * SVE version: unpack in even/odd element order.
     *
     * @param v0 Destination register for first (low/even) half.
     * @param v1 Destination register for second (high/odd) half.
     */
    void unpack_fp16(const TReg &v0, const TReg &v1);

    /**
     * @brief Pack two IEEE‑754 FP16 values from two registers into a single register.
     *
     * ASIMD (NEON) version: pack from low/high half order.
     * SVE version: pack from even/odd element order.
     *
     * @param v0 Source register providing first (low/even) half.
     * @param v1 Source register providing second (high/odd) half.
     */
    void pack_fp16(const TReg &v0, const TReg &v1);
};

// Template specializations for load_vector
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::load_vector(
        const TRegS &dst, const XReg addr, int vl_offset) {
    if (vl_offset == 0) {
        ld1(dst, ptr(addr));
    } else {
        add_imm(X_TMP_0, addr,
                vl_offset * static_cast<int32_t>(simd_bytes(cpu_isa_t::asimd)),
                X_TMP_1);
        ld1(dst, ptr(X_TMP_0));
    }
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::load_vector(
        const TRegS &dst, const XReg addr, int vl_offset) {
    ld1w(dst, P_ALL_ONE / T_z, ptr(addr, vl_offset, MUL_VL));
}

// Template specializations for store_vector
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::store_vector(
        const XReg &addr, const TRegS src, int vl_offset) {
    if (vl_offset == 0) {
        st1(src, ptr(addr));
    } else {
        add_imm(X_TMP_0, addr,
                vl_offset * static_cast<int32_t>(simd_bytes(cpu_isa_t::asimd)),
                X_TMP_1);
        st1(src, ptr(X_TMP_0));
    }
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::store_vector(
        const XReg &addr, const TRegS src, int vl_offset) {
    st1w(src, P_ALL_ONE / T_z, ptr(addr, vl_offset, MUL_VL));
}

// Template specializations for unpack_bf16
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::unpack_bf16(
        const TReg &v0, const TReg &v1) {
    auto tmp = alloc_reg<TReg>();
    movi(tmp.get().s, 0x0);
    zip2(VReg8H(v1.getIdx()), VReg8H(tmp.getIdx()), VReg8H(v0.getIdx()));
    zip1(VReg8H(v0.getIdx()), VReg8H(tmp.getIdx()), VReg8H(v0.getIdx()));
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::unpack_bf16(
        const TReg &v0, const TReg &v1) {
    mov(v1.s, P_ALL_ONE, v0.s);
    lsl(v0.s, v0.s, 16);
    and_(v1.s, 0xFFFF0000);
}

// Template specializations for pack_bf16
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::pack_bf16(
        const TReg &v0, const TReg &v1) {
    uzp2(v0.h, v0.h, v1.h);
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::pack_bf16(
        const TReg &v0, const TReg &v1) {
    bfcvt(v0.h, P_ALL_ONE, v0.s);
    bfcvtnt(v0.h, P_ALL_ONE, v1.s);
}

// Template specializations for unpack_fp16
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::unpack_fp16(
        const TReg &v0, const TReg &v1) {
    mov(VReg16B(v1.getIdx()), VReg16B(v0.getIdx()));
    fcvtl(v0.s, VReg4H(v0.getIdx())); // low 4 float16 to float32
    fcvtl2(v1.s, VReg8H(v1.getIdx())); // high 4 float16 to float32
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::unpack_fp16(
        const TReg &v0, const TReg &v1) {
    mov(v1.s, P_ALL_ONE, v0.s);
    fcvt(v0.s, P_ALL_ONE, v0.h);
    lsr(v1.s, v1.s, 16);
    fcvt(v1.s, P_ALL_ONE, v1.h);
}

// Template specializations for pack_fp16
template <>
inline void jit_uni_kernel_t<cpu_isa_t::asimd>::pack_fp16(
        const TReg &v0, const TReg &v1) {
    fcvtn(VReg4H(v0.getIdx()), v0.s);
    fcvtn2(VReg8H(v0.getIdx()), v1.s);
}

template <>
inline void jit_uni_kernel_t<cpu_isa_t::sve>::pack_fp16(
        const TReg &v0, const TReg &v1) {
    fcvt(v0.h, P_ALL_ONE, v0.s);
    // Next three lines could be replaced by fcvtnt(vmm_src.h, P_ALL_ONE, tmp0.s)
    // Not currently implemented in xbyak
    fcvt(v1.h, P_ALL_ONE, v1.s);
    lsl(v1.s, v1.s, 16);
    orr(v0.h, P_ALL_ONE, v1.h);
}
} // namespace

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());

    VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_ELTWISE((src_md()->data_type == dst_md()->data_type),
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_ELTWISE(utils::one_of(src_md()->data_type, data_type::f32,
                              data_type::bf16, data_type::f16),
            VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_ELTWISE(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "data");
    VDISPATCH_ELTWISE(src_d.is_dense(true), VERBOSE_NONTRIVIAL_STRIDE);

    VDISPATCH_ELTWISE(eltwise_injector::is_supported(isa, desc_.alg_kind),
            VERBOSE_BAD_ALGORITHM);
    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!src_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_ELTWISE(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_ELTWISE(src_d == memory_desc_wrapper(dst_md()),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    // Number of elements in a cacheline. We don't want threads to share
    const int cacheline_elems = 64 / data_d.data_type_size();

    const data_type_t src_dt = pd()->src_md()->data_type;
    const auto offset_bytes
            = types::elements_to_bytes(src_dt, data_d.offset0());

    src += offset_bytes;
    dst += offset_bytes;

    parallel(0, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(
                utils::div_up(nelems, cacheline_elems), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cacheline_elems);
        end = nstl::min(nelems, end * cacheline_elems);
        if (start == end) return;

        jit_args_t args;
        args.src = src + types::elements_to_bytes(src_dt, start);
        args.dst = dst + types::elements_to_bytes(src_dt, start);
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(data_md());
    const char *data_tensor_name = use_dst() ? "dst" : "src";

    VDISPATCH_ELTWISE(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_ELTWISE(
            utils::everyone_is(data_md()->data_type, diff_src_md()->data_type,
                    diff_dst_md()->data_type),
            VERBOSE_INCONSISTENT_DT, data_tensor_name, "diff_src");
    VDISPATCH_ELTWISE(((use_dst() ? dst_md()->data_type : src_md()->data_type)
                              == data_type::f32),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_ELTWISE(
            !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, data_tensor_name);
    VDISPATCH_ELTWISE(set_default_formats_common(),
            VERBOSE_TENSOR_FORMAT_MISMATCH, "diff_src/diff_dst",
            data_tensor_name);
    VDISPATCH_ELTWISE(data_d.is_dense(true), VERBOSE_NONTRIVIAL_STRIDE);
    VDISPATCH_ELTWISE(eltwise_injector::is_supported(isa, desc_.alg_kind),
            VERBOSE_BAD_ALGORITHM);

    // refer to a comment in jit_uni_kernel why this is needed
    VDISPATCH_ELTWISE(IMPLICATION(!data_d.is_dense(), is_zero_preserved()),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_ELTWISE(data_d == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, data_tensor_name, "diff_dst");
    VDISPATCH_ELTWISE(memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

    VDISPATCH_ELTWISE(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {

    auto src = pd()->use_dst() ? CTX_IN_MEM(const uint8_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const uint8_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    // Number of elements in a cacheline. We don't want threads to share
    const int cacheline_elems = 64 / data_d.data_type_size();

    const data_type_t data_dt = pd()->use_dst() ? pd()->dst_md()->data_type
                                                : pd()->src_md()->data_type;
    const auto data_off_bytes
            = types::elements_to_bytes(data_dt, data_d.offset0());
    const auto diff_off_bytes
            = types::elements_to_bytes(data_dt, diff_data_d.offset0());

    src += data_off_bytes;
    diff_dst += diff_off_bytes;
    diff_src += diff_off_bytes;

    parallel(0, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(
                utils::div_up(nelems, cacheline_elems), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cacheline_elems);
        end = nstl::min(nelems, end * cacheline_elems);
        if (start == end) return;

        jit_args_t args;
        args.src = src + types::elements_to_bytes(data_dt, start);
        args.dst = diff_src + types::elements_to_bytes(data_dt, start);
        args.diff_dst = diff_dst + types::elements_to_bytes(data_dt, start);
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template struct jit_uni_eltwise_fwd_t<asimd>;
template struct jit_uni_eltwise_fwd_t<sve>;
template struct jit_uni_eltwise_bwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
