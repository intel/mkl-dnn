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

#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_brgemm_conv_fp8_cvt_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

jit_brgemm_conv_fp8_cvt_kernel_t::jit_brgemm_conv_fp8_cvt_kernel_t(
        const cfg_t &ajcp, const char *name)
    : jit_generator_t(name, ajcp.isa), wjcp(ajcp) {
    assert(utils::one_of(wjcp.src_dt, data_type::f8_e5m2, data_type::f8_e4m3));
    assert(wjcp.dst_dt == data_type::f16);
    if (wjcp.src_dt == data_type::f8_e5m2)
        fp8_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(
                this, xmm_aux1, xmm_aux2, xmm_aux3, kmask_aux, reg64_cvt_aux);
    else
        fp8_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this, xmm_aux1,
                xmm_aux2, xmm_aux3, xmm_aux4, xmm_aux5, reg64_cvt_aux);
}

jit_brgemm_conv_fp8_cvt_kernel_t::~jit_brgemm_conv_fp8_cvt_kernel_t() = default;

void jit_brgemm_conv_fp8_cvt_kernel_t::generate() {
    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    mov(reg_cnt, ptr[param1 + GET_OFF(nelems)]);

    Label loop, tail, finish;
    L(loop);
    cmp(reg_cnt, vec_elems);
    jb(tail, T_NEAR);
    fp8_cvt_->vcvt_f8_to_f16(zmm_result, ptr[reg_src]);
    vmovdqu16(ptr[reg_dst], zmm_result);
    add(reg_src, vec_elems * src_dsz);
    add(reg_dst, vec_elems * dst_dsz);
    sub(reg_cnt, vec_elems);
    jmp(loop, T_NEAR);

    L(tail);
    test(reg_cnt, reg_cnt);
    jz(finish, T_NEAR);
    mov(reg_tmp, 1);
    mov(rcx, reg_cnt);
    shl(reg_tmp, cl);
    dec(reg_tmp);
    kmovd(ktail_mask, reg_tmp.cvt32());
    const Ymm ymm_result(zmm_result.getIdx());
    vmovdqu8(ymm_result | ktail_mask | T_z, ptr[reg_src]);
    fp8_cvt_->vcvt_f8_to_f16(zmm_result, ymm_result);
    vmovdqu16(ptr[reg_dst] | ktail_mask, zmm_result);

    L(finish);

    postamble();

    fp8_cvt_->prepare_table();
}

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
