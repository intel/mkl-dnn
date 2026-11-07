/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2026 Arm Ltd. and affiliates
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

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/reorder/jit_uni_reorder_kernel.hpp"

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace tr {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {

    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32_t::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0: return new jit_uni_reorder_kernel_f32_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

/* kernel */
void jit_uni_reorder_kernel_f32_t::operator()(const call_param_t *c) const {
    jit_generator_t::operator()(c);
}

void jit_uni_reorder_kernel_f32_t::operator()(
        const tail_call_param_t *c) const {
    jit_generator_t::operator()(c);
}

status_t jit_uni_reorder_kernel_f32_t::create_kernel() {
    return jit_generator_t::create_kernel();
}

#define PARAM(x) \
    abi_param1, \
            prb_.is_tail_present ? offsetof(tail_call_param_t, base_params) \
                    + offsetof(call_param_t, x) \
                                 : offsetof(call_param_t, x)
#define TAIL_PARAM(x) abi_param1, offsetof(tail_call_param_t, x)

bool jit_uni_reorder_kernel_f32_t::simple_impl_desc_init(
        const prb_t &prb, simple_impl_desc_t *desc) {
    const int ndims = prb.ndims;

    int ndims_full_unroll = 0;
    int len_last_dim_unroll = 1;
    int tail_len_unroll = 0;
    int len_unroll = 1;

    // It is responsible for finding as many values
    // as kernel can unroll. If tail is present then
    // kernel will unroll only last node (possible improvement).
    // If there is no tail kernel can unroll a few nodes without any loops etc.
    // ndims_full_unroll - how many nodes will be unrolled
    // len_last_dim_unroll - what piece of last unrolled node will be unrolled
    if (prb.is_tail_present) {
        ndims_full_unroll = 1;
        len_unroll = prb.nodes[0].n;
        tail_len_unroll = prb.nodes[0].is_zero_pad_needed
                ? 0
                : static_cast<int>(prb.nodes[0].tail_size);
    } else {
        for (int d = 0; d < ndims; ++d) {
            const auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }
    }

    if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max) return false;

    if (desc) {
        desc->ndims_full_unroll = ndims_full_unroll;
        desc->len_last_dim_unroll = len_last_dim_unroll;
        desc->tail_len_unroll = tail_len_unroll;
        desc->len_unroll = len_unroll;
    }

    return true;
}

bool jit_uni_reorder_kernel_f32_t::applicable(const prb_t &p) {
    using namespace data_type;

    bool bf16_ok = (mayiuse_bf16() && (p.itype == bf16) && (p.otype == bf16)
                           && !interim_f32_needed(p, false) && p.beta == 0.f)
            || (p.itype != bf16 && p.otype != bf16)
            || (p.itype == f32 && p.otype == bf16 && mayiuse_bf16()
                    && p.beta == 0.f)
            || (p.itype == bf16 && p.otype == f32 && mayiuse_bf16()
                    && p.beta == 0.f);

    bool is_f16 = (p.itype == f16 || p.otype == f16);
    bool f16_ok = (p.itype == f32 && p.otype == f16 && p.beta == 0.f)
            || (p.itype == f16 && p.otype == f32 && p.beta == 0.f)
            || (p.itype == f16 && p.otype == f16 && p.beta == 0.f);

    bool ok = true && p.ndims > 0
            && utils::one_of(p.itype, f32, f16, bf16, s32, data_type::s8, u8)
            && utils::one_of(p.otype, f32, f16, bf16, s32, data_type::s8, u8)
            && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
            && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
            && simple_impl_desc_init(p, nullptr) && prb_has_small_strides(p)
            && bf16_ok && IMPLICATION(is_f16, f16_ok);

    return ok;
}

XReg jit_uni_reorder_kernel_f32_t::o_addr(
        int o_off, bool with_type_multiplier) {
    if (o_off) {
        add_imm(X_DEFAULT_ADDR, x_ptr_out_off,
                o_off * (with_type_multiplier ? otype_sz_ : 1), X_TMP);
        return X_DEFAULT_ADDR;
    }

    return x_ptr_out_off;
}

XReg jit_uni_reorder_kernel_f32_t::src_s_addr(int s_off) {
    if (s_off) {
        add_imm(X_DEFAULT_ADDR, x_ptr_src_scale_off, s_off * stype_sz_, X_TMP);
        return X_DEFAULT_ADDR;
    } else {
        return x_ptr_src_scale_off;
    }
}

XReg jit_uni_reorder_kernel_f32_t::dst_s_addr(int s_off) {
    if (s_off) {
        add_imm(X_DEFAULT_ADDR, x_ptr_dst_scale_off, s_off * stype_sz_, X_TMP);
        return X_DEFAULT_ADDR;
    } else {
        return x_ptr_dst_scale_off;
    }
}

XReg jit_uni_reorder_kernel_f32_t::c_addr(int c_off) {
    if (c_off) {
        add_imm(X_DEFAULT_ADDR, x_ptr_comp_off, c_off * sizeof(int32_t), X_TMP);
        return X_DEFAULT_ADDR;
    }

    return x_ptr_comp_off;
}

XReg jit_uni_reorder_kernel_f32_t::data_chunk_addr(int node_id) {
    add_imm(X_DEFAULT_ADDR, abi_param1,
            offsetof(tail_call_param_t, curr_data_chunks)
                    + sizeof(int64_t) * (node_id),
            X_TMP);
    return X_DEFAULT_ADDR;
}

void jit_uni_reorder_kernel_f32_t::step(int off, int prev_i_off, int prev_o_off,
        int prev_s_off, int prev_c_off, int &i_off, int &o_off, int &s_off,
        int &c_off, int step_size) {
    i_off = prev_i_off;
    o_off = prev_o_off;
    s_off = prev_s_off;
    c_off = prev_c_off;

    if (off == 0) return;

    int start_dim = 0, dims_prod = 1;
    for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
        dims_prod *= prb_.n(start_dim);
    assert(start_dim < prb_.ndims);
    off /= step_size;

    for (int dim_id = start_dim; dim_id < prb_.ndims; ++dim_id) {
        i_off += prb_.is(dim_id);
        o_off += prb_.os(dim_id);
        s_off += prb_.ss(dim_id);
        c_off += prb_.cs(dim_id);

        if (off % prb_.n(dim_id)) break;

        i_off += -prb_.n(dim_id) * prb_.is(dim_id);
        o_off += -prb_.n(dim_id) * prb_.os(dim_id);
        s_off += -prb_.n(dim_id) * prb_.ss(dim_id);
        c_off += -prb_.n(dim_id) * prb_.cs(dim_id);

        off /= prb_.n(dim_id);

        if (off == 0) break; /* FIXME: is it really required? */
    }
}

void jit_uni_reorder_kernel_f32_t::step(int off, int prev_i_off, int prev_o_off,
        int &i_off, int &o_off, int step_size) {
    int dummy = 0;
    step(off, prev_i_off, prev_o_off, dummy, dummy, i_off, o_off, dummy, dummy,
            step_size);
}

bool jit_uni_reorder_kernel_f32_t::can_do_tr4x8() {
    using namespace data_type;

    // The kernel is specialised for f32 -> bf16 reorders.
    //
    // This process relies on swapping the two innermost dimensions.
    // Therefore, the input stride in the second node and output stride in
    // first node have to be equal to 1.
    return mayiuse(sve_256) && prb_.ndims >= 2
            && (prb_.itype == f32 && prb_.otype == bf16) && prb_.n(0) == 4
            && prb_.n(1) == 8 && utils::everyone_is(1, prb_.os(0), prb_.is(1))
            && !prb_.is_tail_present
            && prb_.src_scale_type == scale_type_t::NONE
            && prb_.dst_scale_type == scale_type_t::NONE && prb_.beta == 0.f
            && !compensation_needed_;
}

bool jit_uni_reorder_kernel_f32_t::process_unroll_tr4x8(
        const int ndims, const int len) {
    if (!can_do_tr4x8()) return false;

    const int step_size = prb_.n(0) * prb_.n(1);
    int i_off = 0, o_off = 0;
    for (int off = 0; off < len; off += step_size) {
        step(off, i_off, o_off, i_off, o_off, step_size);
        tr4x8_sve256(i_off, o_off);
    }

    return true;
}

void jit_uni_reorder_kernel_f32_t::tr4x8_sve256(int i_off, int o_off) {
    using namespace data_type;

    auto z0 = ZRegS(0);
    auto z1 = ZRegS(1);
    auto z2 = ZRegS(2);
    auto z3 = ZRegS(3);

    assert(temp_xregs.size() >= 4);
    auto x_tmp_0 = temp_xregs[0];
    auto x_tmp_1 = temp_xregs[1];
    auto x_tmp_2 = temp_xregs[2];
    auto x_tmp_3 = temp_xregs[3];

    // Load
    auto in_ptr_diff = itype_sz_ * prb_.is(0);
    add_imm(x_tmp_0, XReg(x_ptr_in_off), itype_sz_ * i_off, X_DEFAULT_ADDR);
    add_imm(x_tmp_1, x_tmp_0, 1 * in_ptr_diff, X_DEFAULT_ADDR);
    add_imm(x_tmp_2, x_tmp_0, 2 * in_ptr_diff, X_DEFAULT_ADDR);
    add_imm(x_tmp_3, x_tmp_0, 3 * in_ptr_diff, X_DEFAULT_ADDR);

    ld1w(z0, P_ALL_ONE, ptr(x_tmp_0));
    ld1w(z1, P_ALL_ONE, ptr(x_tmp_1));
    ld1w(z2, P_ALL_ONE, ptr(x_tmp_2));
    ld1w(z3, P_ALL_ONE, ptr(x_tmp_3));

    // Transpose
    auto z4 = ZReg(4);
    auto z5 = ZReg(5);
    auto z6 = ZReg(6);
    auto z7 = ZReg(7);

    // Interleaving two vectors containing rows of a tile is the same as
    // transposing pairs of elements.
    //
    // If you start with:
    //  vec0:  0     1     2     3     4     5     6     7
    //  vec1:  8     9    10    11    12    13    14    15
    //  vec2: 16    17    18    19    20    21    22    23
    //  vec3: 24    25    26    27    28    29    30    31
    //
    // Then after two zips you have:
    //  vec4 = zip1(vec0, vec2):
    //  vec4: 0    16    1    17    2     18     3    19
    //  vec5 = zip1(vec1, vec3):
    //  vec5: 8    24    9    25    10    26    11    27
    //
    // Notice that if you convert and interleave these then you are done. That's
    // what the subsequent bfcvt-bfcvtnt block of instructions does.
    zip1(z4.s, z0, z2);
    zip1(z5.s, z1, z3);
    zip2(z6.s, z0, z2);
    zip2(z7.s, z1, z3);

    // bfcvt converts one f32 vector to bf16 but leaves 0s in every alternate
    // position within the destination vector (dst1). bfcvtnt then converts the
    // second f32 vector to bf16 while filling in the zeroed spots left by bfcvt
    // within dst1.
    //
    // With the two vectors from above:
    //  vec4: 0    16    1    17    2     18     3    19
    //  vec5: 8    24    9    25    10    26    11    27
    //
    //  vec4 = bfcvt(vec4)
    //  vec4: 0     0    16    0     1     0    ...
    //              ^----------^-----------^
    //                         zeroed gaps left by bfcvt
    //
    //  Now convert vec5 and fill the gaps in vec4 with a single instruction
    //  (storing the result in vec4):
    //  vec4: 0     8    16    24     1     9    ...
    //
    //  Which contains the first 4 transposed columns of the original tile as
    //  required.
    bfcvt(z4.h, P_ALL_ONE / T_z, z4.s);
    bfcvtnt(z4.h, P_ALL_ONE / T_m, z5.s);
    bfcvt(z6.h, P_ALL_ONE / T_z, z6.s);
    bfcvtnt(z6.h, P_ALL_ONE / T_m, z7.s);

    // Store
    auto out_ptr_diff = get_sve_length();
    add_imm(x_tmp_0, XReg(x_ptr_out_off), otype_sz_ * o_off, X_DEFAULT_ADDR);
    add_imm(x_tmp_1, x_tmp_0, out_ptr_diff, X_DEFAULT_ADDR);

    st1h(z4.h, P_ALL_ONE, ptr(x_tmp_0));
    st1h(z6.h, P_ALL_ONE, ptr(x_tmp_1));
}

void jit_uni_reorder_kernel_f32_t::tr8x8_sve256(int i_off, int o_off) {
    using namespace data_type;

    const auto cvt2ps
            = [=](const int startIdx, const int regNum, data_type_t idt) {
        switch (idt) {
            case f32:
                /* do nothing */
                break;
            case f16: cvt_v_f16_f32(startIdx, regNum); break;
            case s32: cvt_z_s32_f32(startIdx, regNum); break;
            case bf16: cvt_v_bf16_fp32(startIdx, regNum); break;
            case data_type::s8:
                cvt_z_s8_s32(startIdx, regNum);
                cvt_z_s32_f32(startIdx, regNum);
                break;
            case u8:
                cvt_z_u8_s32(startIdx, regNum);
                cvt_z_s32_f32(startIdx, regNum);
                break;
            default: assert(!"unreachable");
        }
    };

    const auto cvt2odt = [=](const int startIdx, const int regNum,
                                 data_type_t odt, data_type_t idt) {
        switch (odt) {
            case s32:
                if (idt == f32)
                    cvt_z_f32_s32(startIdx, regNum);
                else if (idt == data_type::s8)
                    cvt_z_s8_s32(startIdx, regNum);
                else if (idt == u8)
                    cvt_z_u8_s32(startIdx, regNum);
                break;
            case data_type::s8:
                if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                if (utils::one_of(idt, f32, s32))
                    cvt_z_s32_s8(startIdx, regNum);
                if (idt == u8) cvt_z_u8_s8(startIdx, regNum);
                break;
            case data_type::bf16:
                if (idt == f32) cvt_v_f32_bf16(startIdx, regNum);
                break;
            case data_type::f16:
                if (idt == f32) cvt_v_f32_f16(startIdx, regNum);
                break;
            case u8:
                if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                if (utils::one_of(idt, f32, s32))
                    cvt_z_s32_u8(startIdx, regNum);
                if (idt == data_type::s8) cvt_z_s8_u8(startIdx, regNum);
                break;
            default: assert(!"unreachable");
        }
    };

    const int unroll = 8;

    const bool interim_f32
            = (prb_.itype != f32) || utils::one_of(f32, prb_.itype, prb_.otype);

    const bool need_saturation
            = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                    && interim_f32);
    const uint64_t sveLen = get_sve_length();

    PReg p_size(DUMMY_IDX);
    switch (unroll * itype_sz_) {
        case 32: p_size = p_lsb_256; break;
        case 16: p_size = p_lsb_128; break;
        case 8: p_size = p_lsb_64; break;
        default: assert(!"unreachable");
    }

    const int node_0_input_stride = prb_.is(0);
    add_imm(X_TMP_0, XReg(x_ptr_in_off), itype_sz_ * i_off, X_DEFAULT_ADDR);
    for (int i = 1; i < unroll / 2; i++)
        add_imm(temp_xregs[i], temp_xregs[i - 1],
                itype_sz_ * node_0_input_stride, X_DEFAULT_ADDR);
    for (uint32_t i = 0; i < unroll / 2; i++)
        ld1w(ZRegS {i}, p_size / T_z, ptr(temp_xregs[i]));
    for (int i = 0; i < unroll / 2; i++)
        add_imm(temp_xregs[i], temp_xregs[(i + 3) % 4],
                itype_sz_ * node_0_input_stride, X_DEFAULT_ADDR);
    for (uint32_t i = 0; i < unroll / 2; i++)
        ld1w(ZRegS {4 + i}, p_size / T_z, ptr(temp_xregs[i]));

    if (interim_f32) cvt2ps(0, unroll, prb_.itype);

#if 0
        /* Debug code to forcedly set test pattern. */
        index(z0.s, 0, 1);
        mov(z0.s, P_NOT_256/T_m, 0);
        mov(z_tmp_vec[0].s, 16);
        for(uint32_t i=1; i<8; i++) {
          add(ZRegS{i}, ZRegS{i-1}, z_tmp_vec[0].s);
          mov(ZRegS{i}, P_NOT_256/T_m, 0);
        }
#endif

    ptrue(p_tmp0.s, VL4);
    /* 1st turn */
    for (uint32_t i = 0; i < unroll / 2; i++) {
        trn1(z_tmp_vec[i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
        trn2(z_tmp_vec[unroll / 2 + i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
    }

    /* 2nd turn */
    trn1(z4.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
    trn1(z5.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
    trn2(z6.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
    trn2(z7.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
    trn1(z_tmp_vec[0].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
    trn1(z_tmp_vec[1].d, z_tmp_vec[6].d, z_tmp_vec[7].d);
    trn2(z_tmp_vec[2].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
    trn2(z_tmp_vec[3].d, z_tmp_vec[6].d, z_tmp_vec[7].d);

    /* 3rd turn */
    for (uint32_t i = 0; i < unroll / 2; i++) {
        mov(ZRegD {i}, ZRegD {unroll / 2 + i});
        mov(z_tmp_vec[unroll / 2 + i].d, z_tmp_vec[i].d);
    }

    /* 4th turn */
    for (uint32_t i = 0; i < unroll / 2; i++) {
        ZRegB z {unroll / 2 + i};
        ZRegB z_tmp = z_tmp_vec[unroll / 2 + i].b;
        /* Move bit 0-127 to 128-255. */
        ext(z, z, 16);
        /* Move bit 128-255 to 0-127. */
        ext(z_tmp, z_tmp, sveLen - 16);
    }

    /* 5th turn */
    for (uint32_t i = 0; i < unroll / 2; i++) {
        ZRegS z0 {i};
        ZRegS z1 {unroll / 2 + i};
        sel(z0, p_tmp0.s, z0, z_tmp_vec[unroll / 2 + i].s);
        sel(z1, p_tmp0, z1, z_tmp_vec[i].s);
    }

    if (need_saturation) {
        init_saturate_f32(ymm_zero_, ymm_saturation_ubound_, X_TMP_0,
                interim_f32 ? f32 : prb_.itype, prb_.otype);
        for (int i = 0; i < unroll; i++)
            saturate_f32(ZRegS(i), ymm_zero_, ymm_saturation_ubound_,
                    prb_.otype, P_ALL_ONE);
    }

    if (prb_.otype != f32)
        cvt2odt(0, unroll, prb_.otype, interim_f32 ? f32 : prb_.itype);

    const int node_1_output_stride = prb_.os(1);

    switch (unroll * otype_sz_) {
        case 32: p_size = p_lsb_256; break;
        case 16: p_size = p_lsb_128; break;
        case 8: p_size = p_lsb_64; break;
        default: assert(!"unreachable");
    }

    add_imm(X_TMP_0, XReg(x_ptr_out_off), otype_sz_ * o_off, X_DEFAULT_ADDR);
    for (int i = 1; i < unroll / 2; i++)
        add_imm(temp_xregs[i], temp_xregs[i - 1],
                otype_sz_ * node_1_output_stride, X_DEFAULT_ADDR);
    for (uint32_t i = 0; i < 4; i++)
        st1w(ZRegS {i}, p_size / T_z, ptr(temp_xregs[i]));
    for (int i = 0; i < unroll / 2; i++)
        add_imm(temp_xregs[i], temp_xregs[(i + 3) % 4],
                otype_sz_ * node_1_output_stride, X_DEFAULT_ADDR);

    for (uint32_t i = 0; i < unroll / 2; i++)
        st1w(ZRegS {4 + i}, p_size / T_z, ptr(temp_xregs[i]));
}

bool jit_uni_reorder_kernel_f32_t::can_do_tr8x8() {
    using namespace data_type;

    static constexpr int desirable_node_size = 8;
    static constexpr int desirable_stride = 1;

    // This process relies on swapping the two innermost dimensions.
    // Therefore, the input stride in the second node and output stride in
    // first node have to be equal to 1.
    return mayiuse(sve_256) && prb_.ndims >= 2
            && ((utils::one_of(prb_.itype, u8, data_type::s8, s32, f32)
                    && utils::one_of(prb_.otype, u8, data_type::s8, s32, f32)))
            && utils::everyone_is(desirable_node_size, prb_.n(0), prb_.n(1))
            && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(1))
            && !prb_.is_tail_present
            && prb_.src_scale_type == scale_type_t::NONE
            && prb_.dst_scale_type == scale_type_t::NONE && prb_.beta == 0.f
            && !compensation_needed_;
}

bool jit_uni_reorder_kernel_f32_t::process_unroll_tr8x8(
        const int ndims, const int len) {
    if (!can_do_tr8x8()) return false;

    const int step_size = prb_.n(0) * prb_.n(1);
    int i_off = 0, o_off = 0;
    for (int off = 0; off < len; off += step_size) {
        step(off, i_off, o_off, i_off, o_off, step_size);
        tr8x8_sve256(i_off, o_off);
    }

    return true;
}

template <cpu_isa_t isa>
bool jit_uni_reorder_kernel_f32_t::process_direct_copy(
        const int ndims, const int len) {
    using namespace data_type;

    static constexpr int desirable_stride = 1;
    using TRegS =
            typename utils::conditional<isa == asimd, VReg4S, ZRegS>::type;
    const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz_;

    // TODO: support tail_processing for direct copy

    const bool do_src_zp = prb_.req_src_zp;
    const bool do_dst_zp = prb_.req_dst_zp;
    const bool zp_applicable = IMPLICATION(
            (do_src_zp || do_dst_zp), utils::one_of(prb_.itype, s32, f32));
    const bool can_do = true && mayiuse(isa) && compensation_needed_ == false
            && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(0))
            && (false || (prb_.itype == prb_.otype ? zp_applicable : false)
                    || (prb_.itype == s32 && prb_.otype == f32)
                    || (prb_.itype == f32 && prb_.otype == s32))
            && len % simd_w == 0 && prb_.n(0) % len == 0
            && !prb_.is_tail_present
            && prb_.src_scale_type == scale_type_t::NONE
            && prb_.dst_scale_type == scale_type_t::NONE && prb_.beta == 0.f;
    if (!can_do) return false;

    static constexpr int vmm_zp_last_idx = 15;
    const auto vmm_src_zp
            = TRegS(do_dst_zp ? vmm_zp_last_idx - 1 : vmm_zp_last_idx);
    if (do_src_zp) {
        uni_ld1rw(vmm_src_zp, PARAM(src_zp));
        uni_scvtf(vmm_src_zp, vmm_src_zp);
    }
    const auto vmm_dst_zp = TRegS(vmm_zp_last_idx);
    if (do_dst_zp) {
        uni_ld1rw(vmm_dst_zp, PARAM(dst_zp));
        uni_scvtf(vmm_dst_zp, vmm_dst_zp);
    }

    const auto apply_zp_ps = [&](const TRegS vmm) {
        if (do_src_zp) fsub(vmm, vmm, vmm_src_zp);
        if (do_dst_zp) fadd(vmm, vmm, vmm_dst_zp);
    };

    for (int off = 0; off < len;) {
        // TODO: we need extra reg for proper saturation if otype == s32
        int unroll = nstl::min(16 - (prb_.otype == s32), (len - off) / simd_w);
        unroll = (do_src_zp || do_dst_zp)
                ? nstl::min(unroll, 16 - do_src_zp - do_dst_zp)
                : unroll;

        int ur = 0;
        int tmp_ur = 0;
        while (ur < unroll) {
            size_t count = 0;
            const int vlen = cpu_isa_traits<isa>::vlen;

            do {
                add_imm(temp_xregs[count++], x_ptr_in_off,
                        (off + ur * simd_w) * itype_sz_, X_DEFAULT_ADDR);
                ur++;
            } while (ur < unroll && count < temp_xregs.size());

            for (size_t i = 0; i < count; i++) {
                if (vlen == 64 || vlen == 32)
                    ld1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                            ptr(temp_xregs[i]));
                else if (vlen == 16)
                    ldr(QReg(tmp_ur + i), ptr(temp_xregs[i]));
                else
                    assert(!"unreachable");
            }
            tmp_ur += count;
        }

        if (prb_.itype != prb_.otype) {
            for (int ur = 0; ur < unroll; ++ur) {
                TRegS r(ur);
                if (prb_.itype == s32 && prb_.otype == f32) {
                    uni_scvtf(r, r);
                    apply_zp_ps(r);
                } else if (prb_.itype == f32 && prb_.otype == s32) {
                    apply_zp_ps(r);
                    uni_frinti(r, r);
                    uni_fcvtzs(r, r);
                } else
                    assert(!"unreachable");
            }
        } else if (do_src_zp || do_dst_zp) {
            for (int ur = 0; ur < unroll; ++ur) {
                const auto vmm = TRegS(ur);
                if (prb_.otype == f32) {
                    apply_zp_ps(vmm);
                } else if (prb_.otype == s32) {
                    uni_scvtf(vmm, vmm);
                    apply_zp_ps(vmm);
                    uni_frinti(vmm, vmm);
                    uni_fcvtzs(vmm, vmm);
                }
            }
        }

        ur = 0;
        tmp_ur = 0;
        while (ur < unroll) {
            size_t count = 0;
            const int vlen = cpu_isa_traits<isa>::vlen;

            do {
                add_imm(temp_xregs[count++], x_ptr_out_off,
                        (off + ur * simd_w) * otype_sz_, X_DEFAULT_ADDR);
                ur++;
            } while (ur < unroll && count < temp_xregs.size());

            for (size_t i = 0; i < count; i++) {
                if (vlen == 64 || vlen == 32)
                    st1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                            ptr(temp_xregs[i]));
                else if (vlen == 16)
                    str(QReg(tmp_ur + i), ptr(temp_xregs[i]));
                else
                    assert(!"unreachable");
            }
            tmp_ur += count;
        }

        off += unroll * simd_w;
    }

    return true;
}

void jit_uni_reorder_kernel_f32_t::process_unroll_generic_step(int reg_unroll,
        const int *i_off, const int *o_off, const int *s_off, const int *c_off,
        const int *zero_padding, const bool tail_processing) {
    using namespace data_type;

    auto cvt2ps = [=](const int startIdx, const int regNum, data_type_t idt) {
        switch (idt) {
            case f32:
                /* do nothing */
                break;
            case s32: cvt_v_s32_f32(startIdx, regNum); break;
            case bf16: cvt_v_bf16_fp32(startIdx, regNum); break;
            case f16: cvt_v_f16_f32(startIdx, regNum); break;
            case data_type::s8:
                cvt_v_s8_s32(startIdx, regNum);
                cvt_v_s32_f32(startIdx, regNum);
                break;
            case u8:
                cvt_v_u8_s32(startIdx, regNum);
                cvt_v_s32_f32(startIdx, regNum);
                break;
            default: assert(!"unreachable");
        }
    };

    auto cvt2odt = [=](const int startIdx, const int regNum, data_type_t odt,
                           data_type_t idt) {
        switch (odt) {
            case f32:
                if (idt == bf16) cvt_v_bf16_fp32(startIdx, regNum);
                if (idt == f16) cvt_v_f16_f32(startIdx, regNum);
                break;
            case s32:
                if (idt == f32)
                    cvt_v_f32_s32(startIdx, regNum);
                else if (idt == data_type::s8)
                    cvt_v_s8_s32(startIdx, regNum);
                else if (idt == u8)
                    cvt_v_u8_s32(startIdx, regNum);
                break;
            case data_type::s8:
                if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                if (idt == f32 || idt == s32) cvt_v_s32_s8(startIdx, regNum);
                if (idt == u8) { cvt_v_u8_s8(startIdx, regNum); }
                break;
            case u8:
                if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                if (idt == f32 || idt == s32) cvt_v_s32_u8(startIdx, regNum);
                if (idt == data_type::s8) cvt_v_s8_u8(startIdx, regNum);
                break;
            case bf16:
                if (idt == f32) cvt_v_f32_bf16(startIdx, regNum);
                break;
            case f16:
                if (idt == f32) cvt_v_f32_f16(startIdx, regNum);
                break;
            default: assert(!"unreachable");
        }
    };

    auto load_bytes_addr = [=](const int ur, const int r) {
        add_imm(temp_xregs[r], x_ptr_in_off, i_off[ur + r] * itype_sz_,
                X_DEFAULT_ADDR);
    };
    auto load_bytes = [=](const int ur, int size, int r) {
        switch (size) {
            case 4: ld1(VReg4S(ur)[r], ptr(temp_xregs[r])); break;
            case 2: ld1(VReg8H(ur)[r], ptr(temp_xregs[r])); break;
            case 1: ld1(VReg16B(ur)[r], ptr(temp_xregs[r])); break;
            default: assert(!"unreachable");
        }
    };

    auto store = [=](const XReg &addr, const VReg ymm, int size) {
        const uint32_t xmm = ymm.getIdx();
        switch (size) {
            case 16: str(QReg(xmm), ptr(addr)); break;
            case 8: str(DReg(xmm), ptr(addr)); break;
            case 4: str(SReg(xmm), ptr(addr)); break;
            case 2: str(HReg(xmm), ptr(addr)); break;
            case 1: str(BReg(xmm), ptr(addr)); break;
            default: assert(!"unreachable");
        }
    };

    /* check whether loading 4 values at once is possible */
    static constexpr int xmm_vlen = 4;
    bool can_load_xmm = reg_unroll % xmm_vlen == 0;
    int registers_total = reg_unroll / 4;
    for (int reg = 0; reg < registers_total; reg++) {
        for (int ur = 1 + (reg * 4); ur < ((reg + 1) * 4); ur++)
            if (i_off[ur] != i_off[ur - 1] + 1) {
                can_load_xmm = false;
                break;
            }
    }
    const int load_step = can_load_xmm ? xmm_vlen : 1;

    /* check whether storing 4 values at once is possible */
    bool can_store_xmm = reg_unroll % xmm_vlen == 0;
    for (int ur = 1; ur < reg_unroll; ++ur)
        if (o_off[ur] != o_off[ur - 1] + 1) {
            can_store_xmm = false;
            break;
        }
    const int ur_step = can_store_xmm ? 4 : 1;
    const int load_tail_step
            = !can_load_xmm && can_store_xmm ? ur_step : load_step;

    const bool interim_f32 = interim_f32_needed(prb_, compensation_needed_);

    const bool need_saturation
            = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                    && interim_f32);

    std::vector<int> store_masks;
    if (tail_processing) {
        for (int ur = 0; ur < reg_unroll; ur += load_tail_step) {
            uni_clear(VReg(ur));
            store_masks.push_back(0);
            for (int r = 0; r < load_tail_step; ++r) {
                if (zero_padding[ur + r] == 0) {
                    store_masks.back() += 1 << r;
                    load_bytes_addr(ur, r);
                }
            }

            for (int r = 0; r < load_tail_step; ++r)
                if (zero_padding[ur + r] == 0) load_bytes(ur, itype_sz_, r);
        }
    } else {
        if (!can_load_xmm && can_store_xmm) {
            assert(ur_step == xmm_vlen);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                for (int r = 0; r < ur_step; ++r) {
                    load_bytes_addr(ur, r);
                }
                for (int r = 0; r < ur_step; ++r)
                    load_bytes(ur, itype_sz_, r);
            }
        } else {
            int ur = 0;
            int tmp_ur = 0;
            while (ur < reg_unroll) {
                size_t count = 0;

                do {
                    add_imm(temp_xregs[count++], x_ptr_in_off,
                            i_off[ur] * itype_sz_, X_DEFAULT_ADDR);
                    ur += load_step;
                } while (ur < reg_unroll && count < temp_xregs.size());

                for (size_t i = 0; i < count; i++) {

                    switch (load_step * itype_sz_) {
                        case 16: ldr(QReg(tmp_ur), ptr(temp_xregs[i])); break;
                        case 8: ldr(DReg(tmp_ur), ptr(temp_xregs[i])); break;
                        case 4: ldr(SReg(tmp_ur), ptr(temp_xregs[i])); break;
                        case 2: ldr(HReg(tmp_ur), ptr(temp_xregs[i])); break;
                        case 1: ldr(BReg(tmp_ur), ptr(temp_xregs[i])); break;
                        default: assert(!"unreachable");
                    }
                    tmp_ur += load_step;
                }
            }
        }
    }

    /* xmm[:] <-- (f32)xmm[:] */
    if (interim_f32) {
        const int cvt_step = nstl::max(load_step, ur_step);
        for (int ur = 0; ur < reg_unroll; ur += cvt_step)
            cvt2ps(ur, 1, prb_.itype);
    }

    if (can_load_xmm && !can_store_xmm) {
        // transposition on the fly
        const bool fast_return = prb_.src_scale_type != scale_type_t::MANY
                && prb_.dst_scale_type != scale_type_t::MANY && prb_.beta == 0.f
                && !prb_.req_src_zp && !prb_.req_dst_zp
                && !compensation_needed_;
        if (fast_return) {
            if (prb_.src_scale_type == scale_type_t::COMMON)
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    fmul(VReg4S(ur), VReg4S(ur), xmm_src_scales_);
            if (prb_.dst_scale_type == scale_type_t::COMMON)
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    fmul(VReg4S(ur), VReg4S(ur), xmm_dst_scales_);
            if (prb_.otype != f32) {
                init_saturate_f32(xmm_zero_, xmm_saturation_ubound_, X_TMP_0,
                        interim_f32 ? f32 : prb_.itype, prb_.otype);
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    if (need_saturation)
                        saturate_f32(VReg4S(ur), xmm_zero_,
                                xmm_saturation_ubound_, prb_.otype, P_ALL_ONE);
                }

                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    cvt2odt(ur, 1, prb_.otype, interim_f32 ? f32 : prb_.itype);
            }
            for (int ur = 0; ur < reg_unroll; ur += load_step) {
                for (int r = 0; r < load_step; ++r) {
                    add_imm(temp_xregs[r], x_ptr_out_off,
                            o_off[ur + r] * otype_sz_, X_DEFAULT_ADDR);
                }

                for (int r = 0; r < load_step; ++r) {
                    if (otype_sz_ == 4)
                        st1(VReg4S(ur)[r], ptr(temp_xregs[r]));
                    else if (otype_sz_ == 2)
                        st1(VReg8H(ur)[r], ptr(temp_xregs[r]));
                    else
                        st1(VReg16B(ur)[r], ptr(temp_xregs[r]));
                }
            }
            return;
        }

        /* scatter elements of xmm into 4 xmms */
        if (itype_sz_ == 4 || interim_f32) {
            for (int ur = 0; ur < reg_unroll; ur += load_step)
                for (int r = 1; r < load_step; ++r) {
                    VReg4S v(ur);
                    VReg4S v_r(ur + r);
                    dup(VReg16B(ur + r), VReg16B(ur)[0]);
                    ins(VReg4S(ur + r)[0], VReg4S(ur)[r]);
                }
        } else {
            for (int ur = 0; ur < reg_unroll; ur += load_step)
                for (int r = 1; r < load_step; ++r)
                    ext(VReg16B(ur + r), VReg16B(ur), VReg16B(ur),
                            itype_sz_ * r);
        }
    }

    /* src zero point application */
    if (prb_.req_src_zp) {
        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            const auto xmm = VReg4S(ur);
            if (interim_f32)
                fsub(xmm, xmm, xmm_src_zp_);
            else
                sub(xmm, xmm, xmm_src_zp_);
        }
    }

    /* scale and beta processing */
    if (can_store_xmm) {
        const auto apply_scales
                = [&](const VReg4S &vreg_scales, scale_arg_t scale_arg,
                          scale_type_t scale_type) {
            if (scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
            } else if (scale_type == scale_type_t::MANY) {
                enum class scale_load_type_t { bcast, load, gather };
                const uint32_t idx = vreg_scales.getIdx();

                uni_clear(VReg(idx));
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    scale_load_type_t scale_load_type
                            = scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast
                            && !tail_processing) {
                        if (scale_arg == scale_arg_t::SRC)
                            ld1r(vreg_scales, ptr(src_s_addr(s_off[ur])));
                        else
                            ld1r(vreg_scales, ptr(dst_s_addr(s_off[ur])));
                        fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load
                            && !tail_processing) {
                        if (scale_arg == scale_arg_t::SRC)
                            ldr(QReg {idx}, ptr(src_s_addr(s_off[ur])));
                        else
                            ldr(QReg {idx}, ptr(dst_s_addr(s_off[ur])));

                        fmul(VReg4S(ur), VReg4S(ur), VReg4S {idx});
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r)
                        if (zero_padding[r] == 0 || !tail_processing) {
                            if (scale_arg == scale_arg_t::SRC)
                                mov(temp_xregs[r - ur], src_s_addr(s_off[r]));
                            else
                                mov(temp_xregs[r - ur], dst_s_addr(s_off[r]));
                        }
                    for (int r = ur; r < ur + ur_step; ++r)
                        if (zero_padding[r] == 0 || !tail_processing)
                            ld1(vreg_scales[r - ur], ptr(temp_xregs[r - ur]));
                    fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                }
            }
        };
        /* xmm <-- src_scales * xmm[:] */
        apply_scales(xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

        /* xmm[:] <-- beta * dst + xmm[:] */
        assert(prb_.beta == 0.f || prb_.beta == 1.f);
        if (prb_.beta == 1.f) {
            int ur = 0;
            int tmp_ur = 0;

            while (ur < reg_unroll) {
                size_t count = 0;

                do {
                    add_imm(temp_xregs[count++], x_ptr_out_off,
                            o_off[ur] * otype_sz_, X_DEFAULT_ADDR);
                    ur += ur_step;
                } while (ur < reg_unroll && count < temp_xregs.size());

                assert(count <= z_tmp_vec_size);
                /* Firstly, data is loaded. */
                for (size_t i = 0; i < count; i++) {

                    if (prb_.otype == f32 || prb_.otype == s32) {
                        ldr(QReg(tmp_vec_idx[i]), ptr(temp_xregs[i])); // bug
                    } else if (prb_.otype == data_type::s8
                            || prb_.otype == u8) {
                        ldr(SReg(tmp_vec_idx[i]), ptr(temp_xregs[i])); // bug
                    } else
                        assert(!"unreachable");
                }

                /* Secondly, it is added. */
                if (prb_.otype == f32) {
                    for (size_t i = 0; i < count; i++) {
                        VReg4S v(tmp_ur);
                        fadd(v, v, VReg4S(tmp_vec_idx[i]));
                        tmp_ur += ur_step;
                    }
                } else {
                    for (size_t i = 0; i < count; i++) {
                        /* cvt2ps() generate successive instructions
                               which have save destination operand,
                               but out of order can be expected. */
                        cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                    }
                    for (size_t i = 0; i < count; i++) {
                        VReg4S v(tmp_ur);
                        fadd(v, v, VReg4S(tmp_vec_idx[i]));
                        tmp_ur += ur_step;
                    }
                }
            }
        }

        /* dst <-- dst_scales * xmm[:] */
        apply_scales(xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
    } else {
        const auto apply_scales
                = [&](const VReg4S &vreg_scales, scale_arg_t scale_arg,
                          scale_type_t scale_type) {
            if (scale_type == scale_type_t::COMMON) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
            } else if (scale_type == scale_type_t::MANY) {
                std::vector<uint32_t> idx_list;
                std::vector<int> offt_list;
                std::vector<uint32_t> vec_reg;
                std::vector<XReg> addr_reg;
                const size_t max_cnt_per_loop
                        = std::min(tmp_vec_idx.size(), temp_xregs.size());
                size_t cnt = 0; // valid unroll steps count

                // 1. Listing up valid steps
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (zero_padding[ur] == 0 || !tail_processing) {
                        idx_list.push_back(ur);
                        offt_list.push_back(s_off[ur]);
                        vec_reg.push_back(tmp_vec_idx[cnt % max_cnt_per_loop]);
                        if (s_off[ur])
                            addr_reg.push_back(
                                    temp_xregs[cnt % max_cnt_per_loop]);
                        else
                            addr_reg.push_back(scale_arg == scale_arg_t::SRC
                                            ? x_ptr_src_scale_off
                                            : x_ptr_dst_scale_off);
                        cnt++;
                    }
                }
                /* 2. Generate instructions considering instruction order.
		       If cnt > max_cnt_per_loop, the following instruction sets are
		       generated several times.
		       add x?, ..., add x? for calculating address
		       ldr s?, ..., ldr s? for loading data
		       fmul v?, ..., fmul v? for scaling */
                for (size_t ur = 0; ur < cnt;) {
                    // Calculating address
                    for (size_t i = ur; i < cnt && i - ur < max_cnt_per_loop;
                            i++)
                        add_imm(addr_reg[i],
                                scale_arg == scale_arg_t::SRC
                                        ? x_ptr_src_scale_off
                                        : x_ptr_dst_scale_off,
                                offt_list[i] * stype_sz_, X_TMP);
                    // Loading data
                    for (size_t i = ur; i < cnt && i - ur < max_cnt_per_loop;
                            i++)
                        ldr(SReg(vec_reg[i]), ptr(addr_reg[i]));
                    // Scaling
                    for (size_t i = ur; i < cnt && i - ur < max_cnt_per_loop;
                            i++) {
                        VReg4S v(idx_list[i]);
                        fmul(v, v, VReg4S(vec_reg[i]));
                    }
                    ur += std::min(cnt, max_cnt_per_loop);
                }
            }
        };

        /* xmm[0] <-- src_scales * xmm[0] */
        apply_scales(xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

        /* xmm[0] <-- beta * dst + xmm[0] */
        assert(prb_.beta == 0.f || prb_.beta == 1.f);
        if (prb_.beta == 1.f) {
            int ur = 0;
            int tmp_ur = 0;
            while (ur < reg_unroll) {
                size_t count = 0;

                do {
                    add_imm(temp_xregs[count++], x_ptr_out_off,
                            o_off[ur] * otype_sz_, X_DEFAULT_ADDR);
                    ur += ur_step;
                } while (ur < reg_unroll && count < (temp_xregs.size() / 2));

                assert(static_cast<size_t>(count) <= z_tmp_vec.size());

                if (prb_.otype == f32) {
                    /* addss: dest[31:0] <- src1[31:0] + src2[31:0]
                         dset[MAXVL-1:32] (Unmodified) */
                    for (size_t i = 0; i < count; i++) {
                        ld1(VReg4S(z_tmp_vec[i].getIdx())[0],
                                ptr(temp_xregs[i]));
                    }
                    for (size_t i = 0; i < count; i++) {
                        SReg s {tmp_vec_idx[i]};
                        fadd(s, s, SReg(tmp_ur + ur_step * i));
                    }
                    for (size_t i = 0; i < count; i++) {
                        mov(VReg4S(tmp_ur)[0], VReg4S(tmp_vec_idx[i])[0]);
                        tmp_ur += ur_step;
                    }
                } else {
                    for (size_t i = 0; i < count; i++) {
                        if (prb_.otype == s32) {
                            ldr(SReg(tmp_vec_idx[i]), ptr(temp_xregs[i]));
                        } else if (utils::one_of(
                                           prb_.otype, data_type::s8, u8)) {
                            ldr(BReg(tmp_vec_idx[i]), ptr(temp_xregs[i]));
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                    }
                    for (size_t i = 0; i < count; i++) {
                        VReg4S v(tmp_ur);
                        fadd(v, v, VReg4S(tmp_vec_idx[i]));
                        tmp_ur += ur_step;
                    }
                }
            }
        }

        /* dst <-- dst_scales * xmm[0] */
        apply_scales(xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
    }

    /* dst zero point application */
    if (prb_.req_dst_zp) {
        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            const auto xmm = VReg4S(ur);
            if (interim_f32)
                fadd(xmm, xmm, xmm_dst_zp_);
            else
                add(xmm, xmm, xmm_dst_zp_);
        }
    }

    /* adjust scale application */
    if (prb_.scale_adjust != 1.f) {
        dup(xmm_tmp_, reg_scale_adjust_);
        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            fmul(VReg4S(ur), VReg4S(ur), xmm_tmp_);
        }
    }

    if (need_saturation) {
        init_saturate_f32(xmm_zero_, xmm_saturation_ubound_, X_TMP_0, f32,
                prb_.otype, compensation_needed_);
        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            saturate_f32(VReg4S(ur), xmm_zero_, xmm_saturation_ubound_,
                    prb_.otype, P_ALL_ONE, compensation_needed_);
        }

        // reset back xmm_zero_ if needed.
        if (compensation_needed_ && (prb_.req_src_zp || prb_.req_dst_zp))
            uni_clear(VReg(xmm_zero_.getIdx()));
    }

    if (compensation_needed_) {
        uint32_t xmm_id = 0;
        const auto get_temp_xmm = [&] {
            const Xbyak_aarch64::VReg temp {tmp_vec_idx[xmm_id]};

            xmm_id = (xmm_id + 1) % tmp_vec_idx.size();

            return temp;
        };
        if (can_store_xmm) {
            enum class comp_load_type_t { bcast, load, gather };

            for (int ur = 0; ur < reg_unroll; ur += ur_step) {

                bool all_ip_padding_one = true;
                bool all_ip_padding_zero = true;
                for (int r = ur; r < ur + ur_step; r++) {
                    if (zero_padding[r] != 1)
                        all_ip_padding_one = false;
                    else
                        all_ip_padding_zero = false;
                }
                if (all_ip_padding_one) continue;

                comp_load_type_t comp_load_type = comp_load_type_t::bcast;

                for (int r = ur + 1; r < ur + ur_step; ++r)
                    if (c_off[r] != c_off[r - 1] + 0) {
                        comp_load_type = comp_load_type_t::load;
                        break;
                    }

                if (comp_load_type == comp_load_type_t::bcast
                        && all_ip_padding_zero) {
                    frinti(xmm_compensation, VReg4S(ur));
                    fcvtzs(xmm_compensation, xmm_compensation);
                    addv(SReg(xmm_compensation.getIdx()), xmm_compensation);
                    addv(SReg(xmm_compensation.getIdx()), xmm_compensation);
                    const auto comp_addr = c_addr(c_off[ur]);
                    const auto xmm_tmp_ = get_temp_xmm().s4;
                    ldr(SReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                    add(xmm_tmp_, xmm_tmp_, xmm_compensation);
                    str(SReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                    continue;
                }

                if (comp_load_type == comp_load_type_t::load)
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (c_off[r] != c_off[r - 1] + 1) {
                            comp_load_type = comp_load_type_t::gather;
                            break;
                        }

                if (comp_load_type == comp_load_type_t::load
                        && all_ip_padding_zero) {
                    const auto xmm_reorder_result = VReg4S(ur);
                    const auto comp_addr = c_addr(c_off[ur]);
                    frinti(xmm_compensation, xmm_reorder_result);
                    fcvtzs(xmm_compensation, xmm_compensation);
                    const auto xmm_tmp_ = get_temp_xmm().s4;
                    ldr(QReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                    add(xmm_compensation, xmm_compensation, xmm_tmp_);
                    str(QReg(xmm_compensation.getIdx()), ptr(comp_addr));
                    continue;
                }

                frinti(xmm_compensation, VReg4S(ur));
                fcvtzs(xmm_compensation, xmm_compensation);
                for (int r = ur; r < ur + ur_step; ++r) {
                    if (zero_padding[r] == 0 || !tail_processing) {
                        mov(W_TMP_0, xmm_compensation[r % 4]);
                        const auto comp_addr = c_addr(c_off[r]);
                        ldr(W_TMP_1, ptr(comp_addr));
                        add(W_TMP_0, W_TMP_0, W_TMP_1);
                        str(W_TMP_0, ptr(comp_addr));
                    }
                }
            }
        } else {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                if (zero_padding[ur] == 0 || !tail_processing) {
                    const auto comp_addr = c_addr(c_off[ur]);
                    frinti(xmm_compensation, VReg4S(ur));
                    fcvtzs(xmm_compensation, xmm_compensation);
                    const auto xmm_tmp_ = get_temp_xmm().s4;
                    ld1(xmm_tmp_, ptr(comp_addr));
                    add(xmm_compensation, xmm_compensation, xmm_tmp_);
                    st1(VReg(xmm_compensation.getIdx()).s[0], ptr(comp_addr));
                }
            }
        }
    }

    for (int ur = 0; ur < reg_unroll; ur += ur_step) {
        if (prb_.req_src_zp || prb_.req_dst_zp) {
            const bool use_store_masks = !store_masks.empty();
            if (use_store_masks) {
                const auto mask = (~store_masks[ur / ur_step]) & 0xF;
                switch (mask) {
                    case 0x0:
                        /* Do nothing */
                        break;
                    case 0x1: ins(VReg4S(ur)[0], xmm_zero_[0]); break;
                    case 0x2: ins(VReg4S(ur)[1], xmm_zero_[1]); break;
                    case 0x3:
                        ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                        break;
                    case 0x4: ins(VReg4S(ur)[2], xmm_zero_[2]); break;
                    case 0x5:
                        ins(VReg4S(ur)[0], xmm_zero_[0]);
                        ins(VReg4S(ur)[2], xmm_zero_[2]);
                        break;
                    case 0x6:
                        ins(VReg4S(ur)[1], xmm_zero_[1]);
                        ins(VReg4S(ur)[2], xmm_zero_[2]);
                        break;
                    case 0x7:
                        ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                        ins(VReg4S(ur)[2], xmm_zero_[2]);
                        break;
                    case 0x8: ins(VReg4S(ur)[3], xmm_zero_[3]); break;
                    case 0x9:
                        ins(VReg4S(ur)[0], xmm_zero_[0]);
                        ins(VReg4S(ur)[3], xmm_zero_[3]);
                        break;
                    case 0xa:
                        ins(VReg4S(ur)[1], xmm_zero_[1]);
                        ins(VReg4S(ur)[3], xmm_zero_[3]);
                        break;
                    case 0xb:
                        ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                        ins(VReg4S(ur)[3], xmm_zero_[3]);
                        break;
                    case 0xc:
                        ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                        break;
                    case 0xd:
                        ins(VReg4S(ur)[0], xmm_zero_[0]);
                        ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                        break;
                    case 0xe:
                        ins(VReg4S(ur)[1], xmm_zero_[1]);
                        ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                        break;
                    case 0xf: movi(VReg16B(ur), 0); break;
                    default: assert(!"unreachable");
                }
            }
        }
        if (prb_.otype != f32)
            cvt2odt(ur, 1, prb_.otype, interim_f32 ? f32 : prb_.itype);

        store(o_addr(o_off[ur]), VReg(ur), ur_step * otype_sz_);
    }
}

bool jit_uni_reorder_kernel_f32_t::interim_f32_needed(
        const prb_t &prb, bool compensation_needed) {
    using namespace data_type;
    bool ret = utils::one_of(f32, prb.itype, prb.otype)
            || prb.src_scale_type != scale_type_t::NONE
            || prb.dst_scale_type != scale_type_t::NONE || prb.beta != 0.f
            || ((prb.req_src_zp || prb.req_dst_zp)
                            ? !(prb.itype == s32 && prb.otype == s32)
                            : false)
            || (prb.itype != f32 && compensation_needed)
            || prb.scale_adjust != 1.f;
    return ret;
}

void jit_uni_reorder_kernel_f32_t::process_unroll_generic(
        const int ndims, int len, const bool tail_processing) {
    assert(IMPLICATION(prb_.nodes[0].tail_size > 0,
            len == static_cast<int>(prb_.nodes[0].n)
                    || len == static_cast<int>(prb_.nodes[0].tail_size)));

    const int blk = 8;

    int i_off[2 * blk] = {0};
    int o_off[2 * blk] = {0};
    int s_off[2 * blk] = {0};
    int c_off[2 * blk] = {0};

    int curr = 0; // will switch between 0 and 1

    const bool interim_f32 = interim_f32_needed(prb_, compensation_needed_);

    if (prb_.req_src_zp) {
        add_imm(X_DEFAULT_ADDR, PARAM(src_zp), X_TMP_0);
        ld1r(xmm_src_zp_, ptr(X_DEFAULT_ADDR));
        if (interim_f32) scvtf(xmm_src_zp_, xmm_src_zp_);
    }
    if (prb_.req_dst_zp) {
        add_imm(X_DEFAULT_ADDR, PARAM(dst_zp), X_TMP_0);
        ld1r(xmm_dst_zp_, ptr(X_DEFAULT_ADDR));
        if (interim_f32) scvtf(xmm_dst_zp_, xmm_dst_zp_);
    }

    for (int off = 0; off < len; off += blk) {
        const int reg_unroll = nstl::min(off + blk, len) - off;
        int zero_padding[blk] = {0};
        const auto curr_blk = curr * blk;

        /* compute offsets and tail*/
        for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
            const int ur_c = curr_blk + ur;
            const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
            const bool is_tail
                    = off + ur >= static_cast<int>(prb_.nodes[0].tail_size);
            step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p], c_off[ur_p],
                    i_off[ur_c], o_off[ur_c], s_off[ur_c], c_off[ur_c]);
            if (tail_processing && is_tail) zero_padding[ur] = 1;
        }

        process_unroll_generic_step(reg_unroll, i_off + curr_blk,
                o_off + curr_blk, s_off + curr_blk, c_off + curr_blk,
                zero_padding, tail_processing);

        curr = 1 - curr;
    }
}

void jit_uni_reorder_kernel_f32_t::compute_ker(
        const int ndims, const int len_unroll, const bool tail_processing) {
    bool optimized = false;
    optimized = optimized || process_direct_copy<sve_256>(ndims, len_unroll)
            || process_direct_copy<asimd>(ndims, len_unroll)
            || process_unroll_tr8x8(ndims, len_unroll)
            || process_unroll_tr4x8(ndims, len_unroll);

    if (!optimized) process_unroll_generic(ndims, len_unroll, tail_processing);
}

void jit_uni_reorder_kernel_f32_t::loop_begin(Label &l, XReg reg_cnt, int len) {
    mov(reg_cnt, len);
    L(l);
}

void jit_uni_reorder_kernel_f32_t::check_if_this_is_last_chunk(
        const XReg reg_curr_chunk, int node_id) {
    // Chunks are backwards numered i.e:
    // [0] -> [node_size]
    // [1] -> [node_size - 1]
    // ...
    // [node_size - 1] -> [1]

    // It is done like this, because it is easier to decrement counter
    // and check if it is equal to zero than increment and check
    // if it is equal to node_size.
    static constexpr int64_t last_chunk = 1;
    cmp(reg_curr_chunk, last_chunk);
}

void jit_uni_reorder_kernel_f32_t::zero_dst_memory(const int bytes_to_zeroing) {
    static constexpr int num_of_bytes_in_xmm = 128 / 8;

    const int xmms_to_zeroing
            = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).quot;
    const int tail_to_zeroing
            = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).rem;

    movi(xmm_tmp_, 0);

    if (xmms_to_zeroing > 0) {
        Label loop;

        mov(X_TMP_4, xmms_to_zeroing);
        L(loop);
        str(QReg(xmm_tmp_.getIdx()), ptr(o_addr(0)));
        add_imm(reg_off_out_, reg_off_out_, num_of_bytes_in_xmm, X_TMP_0);
        add_imm(x_ptr_out_off, x_ptr_out_off, num_of_bytes_in_xmm, X_TMP_0);
        subs(X_TMP_4, X_TMP_4, 1);
        b(NE, loop);
    }

    if (tail_to_zeroing) mov_imm(W_TMP_4, 0);
    for (int i = 0; i < tail_to_zeroing; i++)
        strb(W_TMP_4, ptr(o_addr(i, false)));

    // Restore dst offset to initial value
    if (xmms_to_zeroing > 0) {
        sub_imm(reg_off_out_, reg_off_out_,
                num_of_bytes_in_xmm * xmms_to_zeroing, X_TMP_0);
        sub_imm(x_ptr_out_off, x_ptr_out_off,
                num_of_bytes_in_xmm * xmms_to_zeroing, X_TMP_0);
    }
}

void jit_uni_reorder_kernel_f32_t::finalize_tail_loop(int i_step, int o_step,
        int s_step, int c_step, const int curr_node_id) {
    static constexpr int empty_chunk_info = -1;

    mov(X_TMP_0, empty_chunk_info);
    str(X_TMP_0, ptr(data_chunk_addr(curr_node_id)));

    const int padded_area
            = prb_.nodes[curr_node_id].n - prb_.nodes[curr_node_id].tail_size;

    if (prb_.nodes[curr_node_id].is_zero_pad_needed) {
        int num_of_zero_padded_values = padded_area;
        for (int i = curr_node_id - 1; i >= 0; i--) {
            num_of_zero_padded_values *= prb_.nodes[i].n;
        }

        const int bytes_to_zeroing = num_of_zero_padded_values * otype_sz_;
        zero_dst_memory(bytes_to_zeroing);
    }

    // This function is called by loop_end. At the end
    // of loop_end is section that is responsible for
    // restoring offset values. Restoring is based on
    // len value which is equal to prb.nodes[x].n.
    // If fill_zero_padded_area is called then it means
    // offsets were shifted prb.nodes[x].tail_size times.
    // Therefore, this function has to shift offsets by
    // zero pad area.
    add_imm(reg_off_in_, reg_off_in_, padded_area * i_step * itype_sz_,
            X_TMP_0);
    add_imm(reg_off_out_, reg_off_out_, padded_area * o_step * otype_sz_,
            X_TMP_0);
    add_imm(x_ptr_in_off, x_ptr_in_off, padded_area * i_step * itype_sz_,
            X_TMP_0);
    add_imm(x_ptr_out_off, x_ptr_out_off, padded_area * o_step * otype_sz_,
            X_TMP_0);
    if (prb_.src_scale_type == scale_type_t::MANY)
        add_imm(x_ptr_src_scale_off, x_ptr_src_scale_off,
                padded_area * s_step * stype_sz_, X_TMP_0);
    if (prb_.dst_scale_type == scale_type_t::MANY)
        add_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off,
                padded_area * s_step * stype_sz_, X_TMP_0);

    if (compensation_needed_) {
        add_imm(reg_off_comp_, reg_off_comp_,
                padded_area * c_step * sizeof(int32_t), X_TMP_0);
        add_imm(x_ptr_comp_off, x_ptr_comp_off,
                padded_area * c_step * sizeof(int32_t), X_TMP_0);
    }
}

void jit_uni_reorder_kernel_f32_t::loop_end(Label &l, XReg reg_cnt, int len,
        int i_step, int o_step, int s_step, int c_step,
        const int curr_node_id) {
    add_imm(reg_off_in_, reg_off_in_, i_step * itype_sz_, X_TMP_0);
    add_imm(reg_off_out_, reg_off_out_, o_step * otype_sz_, X_TMP_0);
    add_imm(x_ptr_in_off, x_ptr_in_off, i_step * itype_sz_, X_TMP_0);
    add_imm(x_ptr_out_off, x_ptr_out_off, o_step * otype_sz_, X_TMP_0);

    if (prb_.src_scale_type == scale_type_t::MANY)
        add_imm(x_ptr_src_scale_off, x_ptr_src_scale_off, s_step * stype_sz_,
                X_TMP_0);
    if (prb_.dst_scale_type == scale_type_t::MANY)
        add_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off, s_step * stype_sz_,
                X_TMP_0);

    if (compensation_needed_) {
        add_imm(reg_off_comp_, reg_off_comp_, c_step * sizeof(int32_t),
                X_TMP_0);
        add_imm(x_ptr_comp_off, x_ptr_comp_off, c_step * sizeof(int32_t),
                X_TMP_0);
    }

    subs(reg_cnt, reg_cnt, 1);
    b(NE, l);

    if (prb_.tail(curr_node_id) != 0) {
        Label if_end;

        // On the stack should be an information if node
        // was processed with tail or not.
        ldr(X_TMP_0, post_ptr(sp, 16));

        cmp(X_TMP_0, with_tail_info_);
        b(NE, if_end);
        finalize_tail_loop(i_step, o_step, s_step, c_step, curr_node_id);
        L(if_end);
    }

    // Restore offset to initial values. It means before
    // loop execution.
    sub_imm(reg_off_in_, reg_off_in_, len * i_step * itype_sz_, X_TMP_0);
    sub_imm(reg_off_out_, reg_off_out_, len * o_step * otype_sz_, X_TMP_0);
    sub_imm(x_ptr_in_off, x_ptr_in_off, len * i_step * itype_sz_, X_TMP_0);
    sub_imm(x_ptr_out_off, x_ptr_out_off, len * o_step * otype_sz_, X_TMP_0);

    if (prb_.src_scale_type == scale_type_t::MANY)
        sub_imm(x_ptr_src_scale_off, x_ptr_src_scale_off,
                len * s_step * stype_sz_, X_TMP_0);
    if (prb_.dst_scale_type == scale_type_t::MANY)
        sub_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off,
                len * s_step * stype_sz_, X_TMP_0);
    if (compensation_needed_) {
        sub_imm(reg_off_comp_, reg_off_comp_, len * c_step * sizeof(int32_t),
                X_TMP_0);
        sub_imm(x_ptr_comp_off, x_ptr_comp_off, len * c_step * sizeof(int32_t),
                X_TMP_0);
    }
}

void jit_uni_reorder_kernel_f32_t::compute_blk_ker(
        const simple_impl_desc_t &desc) {
    static constexpr bool with_tail_processing = true;
    Label no_last_chunk, end_label;
    int omp_ndims = prb_.full_ndims - prb_.ndims;

    if (prb_.nodes[0].tail_size > 0) {
        if (!prb_.nodes[0].is_parent_empty()) {
            const int parent_node_id = prb_.nodes[0].parent_node_id;
            ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
            check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
            b(NE, no_last_chunk);
        }

        const int len_unroll = desc.tail_len_unroll > 0 ? desc.tail_len_unroll
                                                        : desc.len_unroll;
        compute_ker(omp_ndims, len_unroll, with_tail_processing);
        b(end_label);
    }

    L(no_last_chunk);
    compute_ker(omp_ndims, desc.len_unroll, !with_tail_processing);
    L(end_label);
}

void jit_uni_reorder_kernel_f32_t::create_loops(const simple_impl_desc_t &desc,
        const std::array<const XReg, 3> &reg_cnt, int jit_loop) {
    assert(jit_loop <= ndims_jit_loop_max);

    if (jit_loop > 0) {
        const int nfu = desc.ndims_full_unroll;
        const int unroll_factor = jit_loop == 1 ? desc.len_last_dim_unroll : 1;
        const int curr_node_id = nfu + (jit_loop - 1);
        const int parent_node_id = prb_.nodes[curr_node_id].parent_node_id;
        const int tail_size = prb_.tail(curr_node_id) / unroll_factor;
        const int node_size = prb_.n(curr_node_id) / unroll_factor;
        const XReg reg_loop_cnt = reg_cnt[jit_loop - 1];
        const bool curr_node_has_tail = prb_.tail(curr_node_id) != 0;
        Label loop, if_no_tail, if_end;

        if (curr_node_has_tail) {
            const size_t reg_bytes = utils::rnd_up(X_TMP_0.getBit() / 8, 16);
            if (prb_.nodes[curr_node_id].is_parent_empty()) {
                mov(reg_loop_cnt, tail_size);
                // Put info that node is being processed with tail.
                mov(X_TMP_0, with_tail_info_);
                str(X_TMP_0, pre_ptr(sp, -static_cast<int>(reg_bytes)));
            } else {
                ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
                check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
                b(NE, if_no_tail);
                mov(reg_loop_cnt, tail_size);
                // Put info that node is being processed with tail.
                mov(X_TMP_0, with_tail_info_);
                str(X_TMP_0, pre_ptr(sp, -static_cast<int>(reg_bytes)));
                b(if_end);

                L(if_no_tail);
                mov(reg_loop_cnt, node_size);
                // Put info that node is being processed without tail.
                mov(X_TMP_0, without_tail_info_);
                str(X_TMP_0, pre_ptr(sp, -static_cast<int>(reg_bytes)));
                L(if_end);
            }
        }

        if (prb_.is_tail_in_one_of_child_nodes(curr_node_id)) {
            if (!curr_node_has_tail) {
                mov(reg_loop_cnt, node_size);
                str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
            }
            L(loop);
            if (!prb_.nodes[curr_node_id].is_parent_empty()) {
                Label if_no_tail_in_child_node;
                ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
                check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
                b(NE, if_no_tail_in_child_node);
                str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
                L(if_no_tail_in_child_node);
            } else {
                str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
            }
        } else if (curr_node_has_tail) {
            L(loop);
        } else {
            loop_begin(loop, reg_loop_cnt, node_size);
        }

        create_loops(desc, reg_cnt, jit_loop - 1);

        loop_end(loop, reg_loop_cnt, node_size,
                prb_.is(curr_node_id) * unroll_factor,
                prb_.os(curr_node_id) * unroll_factor,
                prb_.ss(curr_node_id) * unroll_factor,
                prb_.cs(curr_node_id) * unroll_factor, curr_node_id);
    } else {
        compute_blk_ker(desc);
    }
}

bool jit_uni_reorder_kernel_f32_t::simple_impl() {
    simple_impl_desc_t d;
    if (!simple_impl_desc_init(prb_, &d)) return false;

    eor(reg_off_in_, reg_off_in_, reg_off_in_);
    eor(reg_off_out_, reg_off_out_, reg_off_out_);

    if (prb_.src_scale_type == scale_type_t::MANY)
        mov(x_ptr_src_scale_off, reg_ptr_src_scales_);
    if (prb_.dst_scale_type == scale_type_t::MANY)
        mov(x_ptr_dst_scale_off, reg_ptr_dst_scales_);

    if (compensation_needed_) eor(reg_off_comp_, reg_off_comp_, reg_off_comp_);

    std::array<const XReg, 3> reg_cnt({{x15, x14, x13}});

    const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
    create_loops(d, reg_cnt, n_jit_loops);

    return true;
}

void jit_uni_reorder_kernel_f32_t::impl() {
    if (simple_impl()) return;
    assert(!"no implementation available");
}

#define UNROLL_INST(inst, reg, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) { \
        reg tmp(i); \
        inst(__VA_ARGS__); \
    }
#define UNROLL_INST2(inst, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) \
        inst(__VA_ARGS__);

void jit_uni_reorder_kernel_f32_t::cvt_z_s32_f32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(scvtf, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s32_f32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(scvtf, VReg4S, tmp, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_f32_s32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(frinti, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
    UNROLL_INST(fcvtzs, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_f32_s32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(frinti, VReg4S, tmp, tmp);
    UNROLL_INST(fcvtzs, VReg4S, tmp, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_f32_bf16(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(bfcvtn, VReg4H(i), VReg4S(i));
}

void jit_uni_reorder_kernel_f32_t::cvt_v_bf16_fp32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(shll, VReg4S(i), VReg4H(i), 16);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_f16_f32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(fcvtl, VReg4S(i), VReg4H(i));
}

void jit_uni_reorder_kernel_f32_t::cvt_v_f32_f16(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(fcvtn, VReg4H(i), VReg4S(i));
}

void jit_uni_reorder_kernel_f32_t::cvt_z_s8_s32(
        const size_t startIdx, const size_t regNum) {
    cvt_z_b_s(startIdx, regNum);
    UNROLL_INST(sxtb, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s8_s32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(sxtl, VReg, tmp.h8, tmp.b8);
    UNROLL_INST(sxtl, VReg, tmp.s4, tmp.h4);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_s8_f32(
        const size_t startIdx, const size_t regNum) {
    cvt_z_b_s(startIdx, regNum);
    cvt_z_s32_f32(startIdx, regNum);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s8_f32(
        const size_t startIdx, const size_t regNum) {
    cvt_v_b_s(startIdx, regNum);
    cvt_v_s32_f32(startIdx, regNum);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_b_s(
        const size_t startIdx, const size_t regNum) {
    assert(z_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < z_tmp7.getIdx());

    dup(z_tmp7.b, 0);
    UNROLL_INST(zip1, ZRegB, tmp, tmp, z_tmp7.b);
    UNROLL_INST(zip1, ZRegH, tmp, tmp, z_tmp7.h);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_b_s(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 0);
    dup(v_tmp7.b16, W_TMP_0);
    UNROLL_INST(zip1, VReg16B, tmp, tmp, v_tmp7.b16);
    UNROLL_INST(zip1, VReg8H, tmp, tmp, v_tmp7.h8);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_u8_s32(
        const size_t startIdx, const size_t regNum) {
    cvt_z_b_s(startIdx, regNum);
    UNROLL_INST(uxtb, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_u8_s32(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST(uxtl, VReg, tmp.h8, tmp.b8);
    UNROLL_INST(uxtl, VReg, tmp.s4, tmp.h4);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_s32_s8(
        const size_t startIdx, const size_t regNum) {
    assert(z_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < z_tmp7.getIdx());

    dup(z_tmp7.s, 0);
    UNROLL_INST2(smin, ZRegS(i), 127);
    UNROLL_INST2(smax, ZRegS(i), -128);
    UNROLL_INST(uzp1, ZRegH, tmp, tmp, z_tmp7.h);
    UNROLL_INST(uzp1, ZRegB, tmp, tmp, z_tmp7.b);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s32_s8(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 127);
    dup(v_tmp7.s4, W_TMP_0);
    mov_imm(W_TMP_0, -128);
    UNROLL_INST2(smin, VReg4S(i), VReg4S(i), v_tmp7.s4);
    dup(v_tmp7.s4, W_TMP_0);
    UNROLL_INST2(smax, VReg4S(i), VReg4S(i), v_tmp7.s4);
    mov_imm(W_TMP_0, 0);
    dup(v_tmp7.s4, W_TMP_0);
    UNROLL_INST(uzp1, VReg8H, tmp, tmp, v_tmp7.h8);
    UNROLL_INST(uzp1, VReg16B, tmp, tmp, v_tmp7.b16);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_u8_s8(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(umin, ZRegB(i), 127);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_u8_s8(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 127);
    dup(v_tmp7.b16, W_TMP_0);
    UNROLL_INST(umin, VReg16B, tmp, tmp, v_tmp7.b16);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_u32_u8(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(umin, ZRegS(i), 255);
    UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
    UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_u32_u8(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 255);
    dup(v_tmp7.s4, W_TMP_0);
    UNROLL_INST(umin, VReg4S, tmp, tmp, v_tmp7.s4);
    UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
    UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_s32_u8(
        const size_t startIdx, const size_t regNum) {
    assert(z_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < z_tmp7.getIdx());

    dupm(z_tmp7.s, 255);
    UNROLL_INST2(smax, ZRegS(i), 0);
    UNROLL_INST2(smin, ZRegS(i), P_ALL_ONE / T_m, z_tmp7.s);
    UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
    UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
    UNROLL_INST2(mov, ZRegB(i), P_NOT_128 / T_m, 0);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s32_u8(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 0);
    dup(v_tmp7.s4, W_TMP_0);
    mov_imm(W_TMP_0, 255);
    UNROLL_INST(smax, VReg4S, tmp, tmp, v_tmp7.s4);
    dup(v_tmp7.s4, W_TMP_0);
    UNROLL_INST(smin, VReg4S, tmp, tmp, v_tmp7.s4);
    UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
    UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
}

void jit_uni_reorder_kernel_f32_t::cvt_z_s8_u8(
        const size_t startIdx, const size_t regNum) {
    UNROLL_INST2(smax, ZRegB(i), 0);
}

void jit_uni_reorder_kernel_f32_t::cvt_v_s8_u8(
        const size_t startIdx, const size_t regNum) {
    assert(v_tmp7.getIdx() < startIdx
            || startIdx + regNum - 1 < v_tmp7.getIdx());

    mov_imm(W_TMP_0, 0);
    dup(v_tmp7.b16, W_TMP_0);
    UNROLL_INST(smax, VReg16B, tmp, tmp, v_tmp7.b16);
}
#undef UNROLL_INST
#undef UNROLL_INST

jit_uni_reorder_kernel_f32_t::jit_uni_reorder_kernel_f32_t(const desc_t &desc)
    : kernel_t(desc)
    , itype_sz_(data_type_size(prb_.itype))
    , otype_sz_(data_type_size(prb_.otype))
    , stype_sz_(sizeof(float)) {}

void jit_uni_reorder_kernel_f32_t::generate() {
    using namespace Xbyak_aarch64::util;
    Label end_of_kernel;

    preamble();

    if (prb_.src_scale_type == scale_type_t::COMMON) {
        add_imm(X_DEFAULT_ADDR, PARAM(src_scales), X_TMP_1);
        ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
        ld1r(xmm_src_scales_, ptr(X_TMP_0));
    } else if (prb_.src_scale_type == scale_type_t::MANY) {
        add_imm(X_DEFAULT_ADDR, PARAM(src_scales), X_TMP_0);
        ldr(reg_ptr_src_scales_, ptr(X_DEFAULT_ADDR));
    }

    if (prb_.dst_scale_type == scale_type_t::COMMON) {
        add_imm(X_DEFAULT_ADDR, PARAM(dst_scales), X_TMP_1);
        ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
        ld1r(xmm_dst_scales_, ptr(X_TMP_0));
    } else if (prb_.dst_scale_type == scale_type_t::MANY) {
        add_imm(X_DEFAULT_ADDR, PARAM(dst_scales), X_TMP_0);
        ldr(reg_ptr_dst_scales_, ptr(X_DEFAULT_ADDR));
    }

    if (compensation_needed_) {
        add_imm(X_DEFAULT_ADDR, PARAM(compensation_scratch), X_TMP_0);
        ldr(reg_ptr_comp_, ptr(X_DEFAULT_ADDR));
    }
    if (prb_.scale_adjust == 0.5f) { mov(reg_scale_adjust_, 0x3f000000); }
    add_imm(X_TMP_0, PARAM(in), X_TMP_2);
    add_imm(X_TMP_1, PARAM(out), X_TMP_2);
    ldr(reg_ptr_in_, ptr(X_TMP_0));
    ldr(reg_ptr_out_, ptr(X_TMP_1));

    if (mayiuse(sve)) { /* SVE is available. */
        ptrue(p_lsb_256.b, VL32);
        ptrue(p_lsb_128.b, VL16);
        ptrue(p_lsb_64.b, VL8);
    }

    bool is_tail_in_drv_dims = false;
    for (int i = prb_.ndims; i < prb_.full_ndims; i++)
        if (prb_.nodes[i].tail_size > 0) {
            is_tail_in_drv_dims = true;
            break;
        }

    if (is_tail_in_drv_dims) {
        Label reorder_kernel;
        add_imm(X_DEFAULT_ADDR, TAIL_PARAM(skip_kernel_execution), X_TMP_0);
        ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
        cmp(X_TMP_0, static_cast<int64_t>(true));
        b(EQ, end_of_kernel);

        add_imm(X_DEFAULT_ADDR, TAIL_PARAM(zeroing_data), X_TMP_0);
        ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
        cmp(X_TMP_0, static_cast<int64_t>(false));
        b(EQ, reorder_kernel);
        // If zeroing data is set then all dst memory
        // will be zeroed and nothing more will be done.
        int bytes_to_zeroing = otype_sz_;
        for (int i = 0; i < prb_.ndims; i++) {
            bytes_to_zeroing *= prb_.nodes[i].n;
        }
        eor(reg_off_out_, reg_off_out_, reg_off_out_);
        mov(x_ptr_out_off, reg_ptr_out_);
        zero_dst_memory(bytes_to_zeroing);
        b(end_of_kernel);
        L(reorder_kernel);
    }

    if (can_do_tr8x8()) {
        dup(ymm_zero_, 0);
    } else {
        movi(xmm_zero_, 0);
    }

    impl();

    L(end_of_kernel);
    postamble();
}

#undef TAIL_PARAM
#undef PARAM
} //namespace tr
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
