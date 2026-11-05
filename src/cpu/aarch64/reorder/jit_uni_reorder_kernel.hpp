/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_REORDER_JIT_UNI_REORDER_KERNEL_HPP
#define CPU_AARCH64_REORDER_JIT_UNI_REORDER_KERNEL_HPP

#include <cassert>

#include "common/c_types_map.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace tr {
struct call_param_t {
    const void *in = nullptr;
    void *out = nullptr;
    const float *src_scales = nullptr;
    const float *dst_scales = nullptr;
    int32_t src_zp = 0;
    int32_t dst_zp = 0;
    int32_t *compensation_scratch = nullptr;
};

// The additional structure is needed because
// using a data structure with tail processing
// data for non-tail cases reduces kernel
// performance. This is because there is too
// much data that has to be transferred to the kernel.
struct tail_call_param_t {
    call_param_t base_params;
    int64_t curr_data_chunks[DNNL_MAX_NDIMS] = {-1};
    int64_t zeroing_data = static_cast<int64_t>(false);
    int64_t skip_kernel_execution = static_cast<int64_t>(false);
};

struct kernel_t {
    struct desc_t {
        int id;
        prb_t prb;
    };

    kernel_t(const desc_t &desc)
        : desc_(desc)
        , compensation_needed_(
                  desc.prb.req_s8s8_comp || desc.prb.req_asymmetric_comp) {}
    virtual void operator()(const call_param_t *c) const = 0;
    virtual void operator()(const tail_call_param_t *c) const = 0;
    virtual status_t create_kernel() = 0;
    virtual ~kernel_t() = default;

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- transposition problem (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(
            desc_t &desc, const prb_t &prb, int ndims_ker_max = 0);

    /** creates kernel for the problem described in desc */
    static kernel_t *create(const desc_t &desc);

    /** Minimal reasonable/desirable kernel size.
    * The constant might be used to determine how a problem should be split
    * between kernel and threading driver. */
    static constexpr size_t ker_prb_size_min = 64;

protected:
    const desc_t desc_;
    const prb_t &prb_ = desc_.prb;
    bool compensation_needed_ = false;
};

/* kernel */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    using XReg = Xbyak_aarch64::XReg;
    using WReg = Xbyak_aarch64::WReg;
    using ZReg = Xbyak_aarch64::ZReg;
    using ZRegS = Xbyak_aarch64::ZRegS;
    using VReg = Xbyak_aarch64::VReg;
    using VReg4S = Xbyak_aarch64::VReg4S;
    using PReg = Xbyak_aarch64::PReg;

    void operator()(const call_param_t *c) const override;
    void operator()(const tail_call_param_t *c) const override;

    status_t create_kernel() override;

    enum class scale_arg_t { NONE, SRC, DST };

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 0;
        int tail_len_unroll = 0;
        int len_unroll = 0;
    };

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc);

    static bool applicable(const prb_t &p);

    XReg o_addr(int o_off, bool with_type_multiplier = true);

    XReg src_s_addr(int s_off);

    XReg dst_s_addr(int s_off);

    XReg c_addr(int c_off);

    XReg data_chunk_addr(int node_id);

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int prev_c_off, int &i_off, int &o_off, int &s_off, int &c_off,
            int step_size = 1);

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1);

    bool can_do_tr4x8();
    bool process_unroll_tr4x8(const int ndims, const int len);
    void tr4x8_sve256(int i_off, int o_off);

    void tr8x8_sve256(int i_off, int o_off);

    bool can_do_tr8x8();

    bool process_unroll_tr8x8(const int ndims, const int len);

    template <cpu_isa_t isa>
    bool process_direct_copy(const int ndims, const int len);

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off, const int *c_off,
            const int *zero_padding, const bool tail_processing);

    static bool interim_f32_needed(const prb_t &prb, bool compensation_needed);

    void process_unroll_generic(
            const int ndims, int len, const bool tail_processing);

    void compute_ker(
            const int ndims, const int len_unroll, const bool tail_processing);

    void loop_begin(Xbyak_aarch64::Label &l, XReg reg_cnt, int len);

    void check_if_this_is_last_chunk(const XReg reg_curr_chunk, int node_id);

    void zero_dst_memory(const int bytes_to_zeroing);

    void finalize_tail_loop(int i_step, int o_step, int s_step, int c_step,
            const int curr_node_id);

    void loop_end(Xbyak_aarch64::Label &l, XReg reg_cnt, int len, int i_step,
            int o_step, int s_step, int c_step, const int curr_node_id);

    void compute_blk_ker(const simple_impl_desc_t &desc);

    void create_loops(const simple_impl_desc_t &desc,
            const std::array<const XReg, 3> &reg_cnt, int jit_loop);

    bool simple_impl();

    void impl();

    void cvt_z_s32_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_f32(const size_t startIdx, const size_t regNum);
    void cvt_z_f32_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_bf16(const size_t startIdx, const size_t regNum);
    void cvt_v_bf16_fp32(const size_t startIdx, const size_t regNum);
    void cvt_v_f16_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_f32_f16(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_s32(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_f32(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_f32(const size_t startIdx, const size_t regNum);
    void cvt_z_b_s(const size_t startIdx, const size_t regNum);
    void cvt_v_b_s(const size_t startIdx, const size_t regNum);
    void cvt_z_u8_s32(const size_t startIdx, const size_t regNum);
    void cvt_v_u8_s32(const size_t startIdx, const size_t regNum);
    void cvt_z_s32_s8(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_s8(const size_t startIdx, const size_t regNum);
    void cvt_z_u8_s8(const size_t startIdx, const size_t regNum);
    void cvt_v_u8_s8(const size_t startIdx, const size_t regNum);
    void cvt_z_u32_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_u32_u8(const size_t startIdx, const size_t regNum);
    void cvt_z_s32_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_s32_u8(const size_t startIdx, const size_t regNum);
    void cvt_z_s8_u8(const size_t startIdx, const size_t regNum);
    void cvt_v_s8_u8(const size_t startIdx, const size_t regNum);

    jit_uni_reorder_kernel_f32_t(const desc_t &desc);

    void generate() override;

    ~jit_uni_reorder_kernel_f32_t() override = default;

private:
    static constexpr int64_t with_tail_info_ = static_cast<int64_t>(true);
    static constexpr int64_t without_tail_info_ = static_cast<int64_t>(false);

    int itype_sz_;
    int otype_sz_;
    int stype_sz_;

    const XReg reg_ptr_in_ = x6;
    const XReg reg_ptr_out_ = x2;
    const XReg reg_ptr_src_scales_ = x1;
    const XReg reg_ptr_dst_scales_ = x12;
    const XReg reg_ptr_comp_ = x3;
    const WReg reg_scale_adjust_ = w5;

    const XReg reg_off_in_ = x8;
    const XReg reg_off_out_ = x9;
    const XReg reg_off_comp_ = x11;

    /* X_TMP is required to set address to
     x_tmp_vec(X_TMP_0 - X_TMP_4). */
    XReg X_TMP = x20;

    VReg4S xmm_src_scales_ = v15.s;
    VReg4S xmm_dst_scales_ = v11.s;
    VReg4S xmm_zero_ = v14.s;
    ZRegS ymm_zero_ = z14.s;
    VReg4S xmm_tmp_ = v12.s;
    const VReg4S xmm_src_zp_ = v9.s;
    const VReg4S xmm_dst_zp_ = v10.s;
    const VReg4S xmm_compensation = v8.s;
    VReg4S xmm_saturation_ubound_ = v12.s;
    ZRegS ymm_saturation_ubound_ = z12.s;

    /* Note: x22 - x28 are already used as temporal registgers
       in jit_generator.hpp.
       x_ptr_(in|out|scale|comp)_off keeps (base + offset) address. */
    XReg x_ptr_in_off = reg_ptr_in_;
    XReg x_ptr_out_off = reg_ptr_out_;
    XReg x_ptr_comp_off = reg_ptr_comp_;
    XReg x_ptr_src_scale_off = x19;
    XReg x_ptr_dst_scale_off = x29;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_lsb_256 = p7;
    PReg p_lsb_128 = p6;
    PReg p_lsb_64 = p4;
    PReg p_tmp0 = p5;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

/* TODO: add trans_t class */

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    using XReg = Xbyak_aarch64::XReg;
    using ZRegS = Xbyak_aarch64::ZRegS;
    using ZReg = Xbyak_aarch64::ZReg;
    using PReg = Xbyak_aarch64::PReg;
    using VReg = Xbyak_aarch64::VReg;

    static bool applicable(const prb_t &p);

    jit_single_blk_kernel_t(const prb_t &prb);

    void generate() override;

    void gen_loadu(const ZRegS ymm, const XReg &addr, int size);

    void gen_storeu(const XReg &addr, const ZRegS ymm, int size);

    void gen_maskloadu(
            const ZRegS ymm, const XReg &addr, const PReg mask, int size);

    void gen_maskstoreu(
            const XReg &addr, const ZRegS ymm, const PReg mask, int size);

    // Register allocation xmm0~11
    void gen_transpose_8x8();

    void gen_transpose_4x4();

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask);

    void gen_tr4x4(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);
    void gen_ker4x4(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail);

    void gen_ker16x16_in_8x8(
            int i_off, int o_off, int input_stride, int output_stride);

    // tail can be 1 ~ 16, using sve2 for now
    void gen_ker16x16_in_8x8(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

    void gen_ker32x32_in_16x16(
            int i_off, int o_off, int input_stride, int output_stride);

    void gen_ker32x32_in_16x16(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

    void gen_ker64x64_in_32x32(
            int i_off, int o_off, int input_stride, int output_stride);

    void gen_ker64x64_in_32x32(int i_off, int o_off, int input_stride,
            int output_stride, int in_tail, int out_tail);

private:
    // 6 ~ 12
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble();

    void postamble();

    const tr::prb_t &prb_;

    int itype_sz_;
    int otype_sz_;
    int block_sz;

    XReg reg_ptr_in_ = abi_param1;
    XReg reg_ptr_out_ = abi_param2;
    XReg reg_ptr_tail = abi_param3;

    /* Because the callee-saved registers are not restored blk_reorder,
     the temporary registers (x9-x15) must be assigned.
     Must be selected from the temporary registers (x9-x15). */
    XReg x_addr = x10;
    XReg x_tmp_0 = x11;
    XReg x_tmp_1 = x12;

    /* Avoid P_TMP(p7) in jit_generator.hpp. */
    PReg p_lsb_256 = p6;
    PReg p_mask = p5;
    PReg p_tmp1 = p4;
    PReg p_tmp2 = p3;

    ZRegS ymm_tmp = z0.s;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

} // namespace tr
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
