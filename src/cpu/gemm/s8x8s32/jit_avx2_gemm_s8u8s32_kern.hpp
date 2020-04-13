/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef JIT_AVX2_GEMM_S8U8S32_KERN_HPP
#define JIT_AVX2_GEMM_S8U8S32_KERN_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class jit_avx2_gemm_s8u8s32_kern : public jit_generator {
public:
    jit_avx2_gemm_s8u8s32_kern(
            bool beta_zero, bool enable_offset_c, bool enable_offset_r);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_gemm_s8u8s32_kern);

protected:
    bool beta_zero_;
    bool enable_offset_c_, enable_offset_r_;

    void prefetch_a(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_b(const Xbyak::Address &src) { prefetcht0(src); }
    void prefetch_c(const Xbyak::Address &src) { prefetchw(src); }
    void prefetch_x(const Xbyak::Address &src) { prefetcht1(src); }

    void c_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void c_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);

    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
            const Xbyak::Xmm &src2);
    void kernel_loop(int unroll_m, int unroll_n, bool cfetch);
    void remainder_kernel(int unroll_m, int unroll_n, int unroll_k, int bwidth);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak::Label *&outerloop_label);

    void generate();

private:
    static const int IGEMM_UNROLL_M_ = 16;
    static const int IGEMM_UNROLL_N_ = 4;

    static const int isize_ = 2;
    static const int size_ = 4;

    // Prefetch configuration
    static const int prefetch_size_a_ = 256;
    static const int prefetch_size_b_ = 256;

    static const int offset_a_ = 32, offset_b_ = 32;
    static const int max_unroll_m_ = 16, max_unroll_n_ = 4;

    // Integer register assignments
    Xbyak::Reg64 M_, N_, K_, A_, B_, C_, LDC_, I_, J_, LoopCount_;
    Xbyak::Reg64 AO_, BO_, CO1_, CO2_, AA_;

    // Vector register assignments
    Xbyak::Ymm dp_scratch_, ones_, a_regs_[max_unroll_m_ >> 3], b_regs_[2];
    Xbyak::Ymm c_regs_[max_unroll_m_ >> 3][max_unroll_n_];

    // Stack variable assignments
    int stack_alloc_size_;
    Xbyak::Address arg_a_, arg_b_, arg_c_, arg_ldc_, arg_coffset_c_,
            arg_coffset_r_;
    Xbyak::Address coffset_cx_, coffset_cy_, coffset_rx_, coffset_ry_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // JIT_AVX2_GEMM_S8U8S32_KERN_HPP
