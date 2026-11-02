/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_FP8_CVT_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_FP8_CVT_KERNEL_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Defined in jit_avx512_core_fp8cvt.hpp. Forward-declared here so this
// header does not need to pull in the fp8 conversion injectors.
struct fp8_conversion_base_t;

// Converts contiguous fp8 (e5m2/e4m3) weights to f16 without changing layout.
struct jit_brgemm_conv_fp8_cvt_kernel_t : public jit_generator_t {
    struct cfg_t {
        cpu_isa_t isa {isa_undef};
        data_type_t src_dt {
                data_type_t::dnnl_data_type_undef}; // f8_e5m2/f8_e4m3
        data_type_t dst_dt {data_type_t::dnnl_data_type_undef}; // f16
    };

    struct ctx_t {
        const char *src {nullptr};
        char *dst {nullptr};
        size_t nelems {0};
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_conv_fp8_cvt_kernel_t)

    using reg64_t = Xbyak::Reg64;

    jit_brgemm_conv_fp8_cvt_kernel_t(
            const cfg_t &ajcp, const char *name = jit_name());
    ~jit_brgemm_conv_fp8_cvt_kernel_t() override;

private:
    cfg_t wjcp;

    std::unique_ptr<fp8_conversion_base_t> fp8_cvt_;

    const dim_t src_dsz = types::data_type_size(wjcp.src_dt);
    const dim_t dst_dsz = types::data_type_size(wjcp.dst_dt);
    const dim_t vec_elems = 32;
    const reg64_t reg_src = rax;
    const reg64_t reg_dst = rbx;
    const reg64_t reg_cnt = r10;
    const reg64_t reg_tmp = rdx;

    // dedicated scratch register required by the fp8 conversion injectors
    const reg64_t reg64_cvt_aux = r11;

    const Xbyak::Opmask ktail_mask = k1;
    const Xbyak::Opmask kmask_aux = k2;

    const Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(16);
    const Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(17);
    const Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(18);
    const Xbyak::Xmm xmm_aux4 = Xbyak::Xmm(19);
    const Xbyak::Xmm xmm_aux5 = Xbyak::Xmm(20);
    const Xbyak::Zmm zmm_result = Xbyak::Zmm(2);

    void generate() override;
};

// Source and weights both retain their original element order.
struct jit_brgemm_conv_src_copy_kernel_t
    : public jit_brgemm_conv_fp8_cvt_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_conv_src_copy_kernel_t)

    jit_brgemm_conv_src_copy_kernel_t(const cfg_t &ajcp)
        : jit_brgemm_conv_fp8_cvt_kernel_t(ajcp, jit_name()) {}
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
