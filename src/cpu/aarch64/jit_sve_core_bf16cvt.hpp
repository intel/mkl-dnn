/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_SVE_CORE_BF16CVT_HPP
#define CPU_AARCH64_JIT_SVE_CORE_BF16CVT_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/bfloat16.hpp"
#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace bf16_support {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t nelems;
};
} // namespace bf16_support

#define GET_OFF(field) offsetof(bf16_support::jit_call_t, field)

struct bf16_emulation_t {
    using ZReg_t = const Xbyak_aarch64::ZReg;
    using ZRegS_t = const Xbyak_aarch64::ZRegS;
    using PReg_t = const Xbyak_aarch64::PReg;
    using XReg_t = const Xbyak_aarch64::XReg;

    bf16_emulation_t(jit_generator_t *host, ZReg_t one, ZReg_t even,
            ZReg_t qnan_bit, XReg_t scratch, ZReg_t tr0, ZReg_t tr1,
            PReg_t p_all, PReg_t p_tmp)
        : host_(host)
        , one_(one)
        , even_(even)
        , qnan_bit_(qnan_bit)
        , scratch_(scratch)
        , tr0_(tr0)
        , tr1_(tr1)
        , p_all_(p_all)
        , p_tmp_(p_tmp) {
        assert(p_all_.getIdx() != p_tmp_.getIdx());
    }

    bf16_emulation_t(jit_generator_t *host, ZReg_t one, ZReg_t even,
            ZReg_t qnan_bit, XReg_t scratch, ZReg_t tr0, PReg_t p_all,
            PReg_t p_tmp)
        : bf16_emulation_t(
                  host, one, even, qnan_bit, scratch, tr0, tr0, p_all, p_tmp) {}

    void vdpbf16ps(ZRegS_t &acc, ZRegS_t wei, ZRegS_t inp) {
        using namespace Xbyak_aarch64;
        const uint32_t acc_idx = acc.getIdx();
        const uint32_t wei_idx = wei.getIdx();
        const uint32_t inp_idx = inp.getIdx();
        const uint32_t t0 = tr0_.getIdx();
        const uint32_t t1 = tr1_.getIdx();

        assert(t0 != t1 && "vdpbf16ps requires tr0_ != tr1_");
        assert(t0 != acc_idx && t1 != acc_idx);
        assert(t0 != wei_idx && t0 != inp_idx);
        assert(t1 != wei_idx && t1 != inp_idx);
        assert(acc_idx != wei_idx && acc_idx != inp_idx);

        host_->asr(ZRegS(t0), ZRegS(wei_idx), 16);
        host_->lsl(ZRegS(t0), ZRegS(t0), 16);

        host_->asr(ZRegS(t1), ZRegS(inp_idx), 16);
        host_->lsl(ZRegS(t1), ZRegS(t1), 16);

        host_->fmla(ZRegS(acc_idx), p_all_ / T_m, ZRegS(t0), ZRegS(t1));

        host_->lsl(ZRegS(t0), ZRegS(wei_idx), 16);
        host_->lsl(ZRegS(t1), ZRegS(inp_idx), 16);

        host_->fmla(ZRegS(acc_idx), p_all_ / T_m, ZRegS(t0), ZRegS(t1));
    }

    void vcvtneps2bf16(
            const Xbyak_aarch64::ZRegS &dst, const Xbyak_aarch64::ZRegS &src) {
        using namespace Xbyak_aarch64;
        const uint32_t d = dst.getIdx();
        const uint32_t s = src.getIdx();
        const uint32_t t0 = tr0_.getIdx();
        const uint32_t one_idx = one_.getIdx();
        const uint32_t even_idx = even_.getIdx();
        const uint32_t qnan_idx = qnan_bit_.getIdx();

        assert(d != t0 && d != one_idx && d != even_idx && d != qnan_idx);
        assert(s != t0);

        assert(t0 != one_idx && t0 != even_idx && t0 != qnan_idx);

        host_->fcmne(PRegS(p_tmp_.getIdx()), p_all_ / T_z, ZRegS(s), ZRegS(s));

        host_->lsr(ZRegS(t0), ZRegS(s), 16);
        host_->and_(ZRegD(t0), ZRegD(t0), ZRegD(one_idx));

        host_->add(ZRegS(t0), ZRegS(t0), ZRegS(even_idx));

        host_->add(ZRegS(t0), ZRegS(s), ZRegS(t0));
        host_->orr(ZRegD(d), ZRegD(s), ZRegD(qnan_idx));

        host_->sel(ZRegS(d), PReg(p_tmp_.getIdx()), ZRegS(d), ZRegS(t0));

        host_->lsr(ZRegS(d), ZRegS(d), 16);
    }

    void vcvtneps2bf16(const Xbyak_aarch64::ZRegS &z_inout) {
        vcvtneps2bf16(z_inout, z_inout);
    }

    void init_vcvtneps2bf16() {
        using namespace Xbyak_aarch64;
        const WReg w_scratch(scratch_.getIdx());

        host_->mov_imm(w_scratch, 0x1u);
        host_->dup(ZRegS(one_.getIdx()), w_scratch);

        host_->mov_imm(w_scratch, 0x7FFFu);
        host_->dup(ZRegS(even_.getIdx()), w_scratch);

        host_->mov_imm(w_scratch, 0x00400000u);
        host_->dup(ZRegS(qnan_bit_.getIdx()), w_scratch);
    }

    static cpu_isa_t get_isa() { return sve; }

private:
    jit_generator_t *const host_;
    ZReg_t one_;
    ZReg_t even_;
    ZReg_t qnan_bit_;
    XReg_t scratch_;
    ZReg_t tr0_;
    ZReg_t tr1_;
    PReg_t p_all_;
    PReg_t p_tmp_;
};

struct jit_sve_core_add_cvt_ps_to_bf16_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_core_add_cvt_ps_to_bf16)

    jit_sve_core_add_cvt_ps_to_bf16_t() : jit_generator_t() {
        assert(mayiuse(sve) && "SVE require for add_cvt_ps_to_bf16 kernel");

        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this, z_one, z_even,
                z_qnan_bit, x_scratch, z_tr0, z_tr1, P_ALL_ONE, p_nan_tmp);
        UNUSED_STATUS(create_kernel());
    }

    ~jit_sve_core_add_cvt_ps_to_bf16_t() override = default;

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_sve_core_add_cvt_ps_to_bf16_t)

    void generate() override {
        using namespace Xbyak_aarch64;
        preamble();

        ldr(x_inp, ptr(abi_param1, static_cast<int32_t>(GET_OFF(inp))));
        ldr(x_add, ptr(abi_param1, static_cast<int32_t>(GET_OFF(add))));
        ldr(x_out, ptr(abi_param1, static_cast<int32_t>(GET_OFF(out))));
        ldr(x_nelems, ptr(abi_param1, static_cast<int32_t>(GET_OFF(nelems))));

        bf16_emu_->init_vcvtneps2bf16();

        cntw(x_step);
        mov(x_cnt, xzr);

        Label loop_start, loop_end;
        L(loop_start);

        whilelo(PRegS(p_loop.getIdx()), x_cnt, x_nelems);
        b(EQ, loop_end);

        ld1w(ZRegS(z_fp32_inp.getIdx()), p_loop / T_z, ptr(x_inp));
        ld1w(ZRegS(z_fp32_add.getIdx()), p_loop / T_z, ptr(x_add));

        fadd(ZRegS(z_fp32_inp.getIdx()), ZRegS(z_fp32_inp.getIdx()),
                ZRegS(z_fp32_add.getIdx()));

        bf16_emu_->vcvtneps2bf16(ZRegS(z_fp32_inp.getIdx()));

        st1h(ZRegS(z_fp32_inp.getIdx()), p_loop, ptr(x_out));

        addvl(x_inp, x_inp, 1);
        addvl(x_add, x_add, 1);

        add(x_out, x_out, x_step, ShMod::LSL, 1);

        add(x_cnt, x_cnt, x_step);
        b(loop_start);

        L(loop_end);
        postamble();
    }

    void operator()(bf16_support::jit_call_t *params) const {
        jit_generator_t::operator()(params);
        msan_unpoison(params->out, params->nelems * sizeof(bfloat16_t));
    }

private:
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    Xbyak_aarch64::ZReg z_fp32_inp = Xbyak_aarch64::ZReg(0);
    Xbyak_aarch64::ZReg z_fp32_add = Xbyak_aarch64::ZReg(1);

    Xbyak_aarch64::ZReg z_tr0 = Xbyak_aarch64::ZReg(2);
    Xbyak_aarch64::ZReg z_tr1 = Xbyak_aarch64::ZReg(3);
    Xbyak_aarch64::ZReg z_one = Xbyak_aarch64::ZReg(4);
    Xbyak_aarch64::ZReg z_even = Xbyak_aarch64::ZReg(5);
    Xbyak_aarch64::ZReg z_qnan_bit = Xbyak_aarch64::ZReg(6);

    Xbyak_aarch64::PReg p_loop = Xbyak_aarch64::PReg(1);
    Xbyak_aarch64::PReg p_nan_tmp = Xbyak_aarch64::PReg(2);

    const Xbyak_aarch64::XReg x_inp = Xbyak_aarch64::XReg(9);
    const Xbyak_aarch64::XReg x_add = Xbyak_aarch64::XReg(10);
    const Xbyak_aarch64::XReg x_out = Xbyak_aarch64::XReg(11);
    const Xbyak_aarch64::XReg x_nelems = Xbyak_aarch64::XReg(12);
    const Xbyak_aarch64::XReg x_step = Xbyak_aarch64::XReg(13);
    const Xbyak_aarch64::XReg x_cnt = Xbyak_aarch64::XReg(14);
    const Xbyak_aarch64::XReg x_scratch = Xbyak_aarch64::XReg(15);
};

#undef GET_OFF

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
