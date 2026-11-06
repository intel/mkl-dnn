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

#include <assert.h>
#include <numeric>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) >= 5) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
constexpr static bool is_windows = true;
#else
constexpr static bool is_windows = false;
#endif

using namespace Xbyak;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

static bool prb_has_small_strides(const prb_t &prb) {
    constexpr ptrdiff_t max_stride = (1LL << 31) - 1;
    for (int d = 0; d < prb.ndims; ++d) {
        const ptrdiff_t cms = max_stride / prb.nodes[d].n;
        const bool small_strides = true
                && prb.nodes[d].is < cms / (int)data_type_size(prb.itype)
                && prb.nodes[d].os < cms / (int)data_type_size(prb.otype);
        if (!small_strides) return false;
    }
    return true;
}

bool prb_has_huge_prime_number(const prb_t &prb) {
    for (int d = 0; d < prb.ndims; ++d) {
        auto n = prb.nodes[d].n;
        if (n >= INT_MAX && math::is_prime(n)) return true;
    }
    return false;
}

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    void operator()(const call_param_t *c) const override {
        jit_generator_t::operator()(c);
    }
    void operator()(const tail_call_param_t *c) const override {
        jit_generator_t::operator()(c);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

    enum class scale_arg_t { NONE, SRC, DST };

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll = 0;
        dim_t len_last_dim_unroll = 0;
        dim_t tail_len_unroll = 0;
        dim_t len_unroll = 0;
    };

#define PARAM(x) \
    prb_.is_tail_present \
            ? ptr[abi_param1 + offsetof(tail_call_param_t, base_params) \
                      + offsetof(call_param_t, x)] \
            : ptr[abi_param1 + offsetof(call_param_t, x)]
#define TAIL_PARAM(x) ptr[abi_param1 + offsetof(tail_call_param_t, x)]

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        dim_t len_last_dim_unroll = 1;
        dim_t tail_len_unroll = 0;
        dim_t len_unroll = 1;

        // It is responsible for finding as many values as kernel can unroll.
        // If tail is present, then kernel will unroll only last node.
        // If there is no tail, kernel can unroll few nodes without any loops.
        // `ndims_full_unroll` - how many nodes will be unrolled
        // `len_last_dim_unroll` - which piece of the last unrolled node will
        // be unrolled.
        if (prb.is_tail_present) {
            ndims_full_unroll = 1;
            len_unroll = prb.nodes[0].n;
            tail_len_unroll = prb.nodes[0].is_zero_pad_needed
                    ? 0
                    : prb.nodes[0].tail_size;
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

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims > 0
                && utils::one_of(
                        p.itype, f32, bf16, f16, s32, f8_e5m2, f8_e4m3, s8, u8)
                && utils::one_of(
                        p.otype, f32, bf16, f16, s32, f8_e5m2, f8_e4m3, s8, u8)
                && IMPLICATION(
                        utils::one_of(p.itype, bf16, f16, f8_e5m2, f8_e4m3),
                        utils::one_of(p.otype, s8, u8, f32, bf16, f16, f8_e5m2,
                                f8_e4m3))
                && IMPLICATION(
                        utils::one_of(p.otype, bf16, f16, f8_e5m2, f8_e4m3),
                        utils::one_of(p.itype, s8, u8, f32, bf16, f16, f8_e5m2,
                                f8_e4m3))
                && IMPLICATION(utils::one_of(p.itype, f8_e5m2, f8_e4m3)
                                || utils::one_of(p.otype, f8_e5m2, f8_e4m3),
                        !utils::one_of(p.itype, u8, s8)
                                && !utils::one_of(p.otype, u8, s8))
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr) && mayiuse(avx2)
                && IMPLICATION(utils::one_of(bf16, p.itype, p.otype),
                        mayiuse(avx512_core) || mayiuse(avx2_vnni_2))
                && IMPLICATION(utils::one_of(f16, p.itype, p.otype),
                        mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2))
                && IMPLICATION(utils::one_of(f8_e5m2, p.itype, p.otype)
                                || utils::one_of(f8_e4m3, p.itype, p.otype),
                        (mayiuse(avx512_core_fp16) || mayiuse(avx10_2)))
                && prb_has_small_strides(p) && !prb_has_huge_prime_number(p);
        return ok;
    }

    static bool is_f8_supported(cpu_isa_t isa) {
        return is_superset(isa, avx512_core_fp16) || is_superset(isa, avx10_2);
    }

    Address i_addr(dim_t i_off) {
        return ptr[reg_ptr_in_ + reg_off_in_ + i_off * itype_sz_];
    }

    Address o_addr(dim_t o_off, bool with_type_multiplier = true) {
        if (with_type_multiplier)
            return ptr[reg_ptr_out_ + reg_off_out_ + o_off * otype_sz_];
        else
            return ptr[reg_ptr_out_ + reg_off_out_ + o_off];
    }

    Address src_s_addr(dim_t s_off) {
        return ptr[reg_ptr_src_scales_ + reg_off_scale_ + s_off * stype_sz_];
    }

    Address dst_s_addr(dim_t s_off) {
        return ptr[reg_ptr_dst_scales_ + reg_off_scale_ + s_off * stype_sz_];
    }

    Address c_addr(dim_t c_off) {
        return ptr[reg_ptr_comp_ + reg_off_comp_ + c_off * sizeof(int32_t)];
    }

    Address data_chunk_addr(int node_id) {
        return ptr[abi_param1 + offsetof(tail_call_param_t, curr_data_chunks)
                + sizeof(int64_t) * (node_id)];
    }

    void step(dim_t off, dim_t prev_i_off, dim_t prev_o_off, dim_t prev_s_off,
            dim_t prev_c_off, dim_t &i_off, dim_t &o_off, dim_t &s_off,
            dim_t &c_off, dim_t step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;
        c_off = prev_c_off;

        if (off == 0) return;

        int start_dim = 0;
        dim_t dims_prod = 1;
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

            i_off -= prb_.n(dim_id) * prb_.is(dim_id);
            o_off -= prb_.n(dim_id) * prb_.os(dim_id);
            s_off -= prb_.n(dim_id) * prb_.ss(dim_id);
            c_off -= prb_.n(dim_id) * prb_.cs(dim_id);

            off /= prb_.n(dim_id);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(dim_t off, dim_t prev_i_off, dim_t prev_o_off, dim_t &i_off,
            dim_t &o_off, dim_t step_size = 1) {
        dim_t dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, dummy, i_off, o_off, dummy,
                dummy, step_size);
    }

    void tr8x8_avx2(dim_t i_off, dim_t o_off) {
        using namespace data_type;

        const auto cvt2ps
                = [this](const Ymm dst, const Operand &src, data_type_t idt) {
            switch (idt) {
                case f32:
                    if (src.isMEM() || src.getIdx() != dst.getIdx())
                        vmovups(dst, src);
                    break;
                case bf16:
                    vpmovzxwd(dst, src);
                    vpslld(dst, dst, 0x10);
                    break;
                case f16:
                    if (is_superset(isa_, avx512_core_fp16)) {
                        if (src.isMEM())
                            vcvtph2psx(dst, src);
                        else
                            vcvtph2psx(dst, Xmm(src.getIdx()));
                    } else if (is_superset(isa_, avx2_vnni_2)) {
                        if (src.isMEM())
                            vcvtph2ps(dst, src);
                        else
                            vcvtph2ps(dst, Xmm(src.getIdx()));
                    } else
                        assert(!"invalid isa");
                    break;
                case s32: vcvtdq2ps(dst, src); break;
                case f8_e5m2:
                    if (f8_e5m2_cvt_ && is_f8_supported(isa_))
                        f8_e5m2_cvt_->vcvt_f8_to_f32(Zmm(dst.getIdx()), src);
                    else
                        assert(!"invalid isa or fp8 emulation not "
                                "available");
                    break;
                case f8_e4m3:
                    if (f8_e4m3_cvt_ && is_f8_supported(isa_))
                        f8_e4m3_cvt_->vcvt_f8_to_f32(Zmm(dst.getIdx()), src);
                    else
                        assert(!"invalid isa or fp8 emulation not "
                                "available");
                    break;
                case s8:
                    vpmovsxbd(dst, src);
                    vcvtdq2ps(dst, dst);
                    break;
                case u8:
                    vpmovzxbd(dst, src);
                    vcvtdq2ps(dst, dst);
                    break;
                default: assert(!"unreachable");
            }
        };

        const auto cvt2odt = [this, cvt2ps](const Ymm ymm, data_type_t odt,
                                     data_type_t idt, bool need_saturation) {
            const Xmm xmm = Xmm(ymm.getIdx());
            if (need_saturation && isa_has_sat_cvt(isa_, odt)) {
                switch (odt) {
                    case s8:
                    case u8: vpmovusdb(xmm, ymm); break;
                    case s32: break; // nothing to do
                    default: assert(!"unsupported data type");
                }
                return;
            }

            switch (odt) {
                case bf16:
                    if (utils::one_of(
                                idt, f32, f16, f8_e5m2, f8_e4m3, s8, u8)) {
                        if (!utils::one_of(idt, f32, f8_e5m2, f8_e4m3))
                            cvt2ps(ymm, ymm, idt);
                        if (utils::one_of(idt, f8_e5m2, f8_e4m3)) {
                            if (is_f8_supported(isa_)) {
                                if (idt == f8_e5m2 && f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f32(
                                            Zmm(ymm.getIdx()), ymm);
                                else if (idt == f8_e4m3 && f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f32(
                                            Zmm(ymm.getIdx()), ymm);
                                else
                                    assert(!"fp8 emulation not available");
                            } else
                                assert(!"invalid isa for fp8 emulation");
                        }
                        if (is_superset(isa_, avx2_vnni_2)) {
                            vcvtneps2bf16(
                                    Xmm(ymm.getIdx()), ymm, Xbyak::VexEncoding);
                        } else if (mayiuse(avx512_core_bf16)) {
                            vcvtneps2bf16(Xmm(ymm.getIdx()), ymm);
                        } else {
                            bf16_emu_->vcvtneps2bf16(
                                    Ymm(ymm.getIdx()), Zmm(ymm.getIdx()));
                        }
                    }
                    break;
                case f16:
                    if (utils::one_of(
                                idt, f32, bf16, f8_e5m2, f8_e4m3, s8, u8)) {
                        if (!utils::one_of(idt, f32, f8_e5m2, f8_e4m3))
                            cvt2ps(ymm, ymm, idt);
                        if (utils::one_of(idt, f8_e5m2, f8_e4m3)) {
                            if (is_f8_supported(isa_)) {
                                if (idt == f8_e5m2 && f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f16(ymm, ymm);
                                else if (idt == f8_e4m3 && f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f16(ymm, ymm);
                                else
                                    assert(!"fp8 emulation not available");
                            } else
                                assert(!"invalid isa for fp8 emulation");
                        } else
                            vcvtps2ph(Xmm(ymm.getIdx()), ymm, _op_mxcsr);
                    }
                    break;
                case s32:
                    if (idt == f32)
                        vcvtps2dq(ymm, ymm);
                    else if (idt == s8)
                        vpmovsxbd(ymm, ymm);
                    else if (idt == u8)
                        vpmovzxbd(ymm, ymm);
                    break;
                case f8_e5m2:
                    if (utils::one_of(idt, f32, bf16, f16, f8_e4m3)) {
                        if (is_f8_supported(isa_)) {
                            if (idt == f8_e4m3) {
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f16(ymm, ymm);
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f16_to_f8(ymm, ymm);
                            } else {
                                if (idt != f32) cvt2ps(ymm, ymm, idt);
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f32_to_f8(
                                            Xmm(ymm.getIdx()),
                                            Zmm(ymm.getIdx()));
                            }
                        } else
                            assert(!"invalid isa or fp8 emulation not "
                                    "available");
                        break;
                    }
                case f8_e4m3:
                    if (utils::one_of(idt, f32, bf16, f16, f8_e5m2)) {
                        if (is_f8_supported(isa_)) {
                            if (idt == f8_e5m2) {
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f16(ymm, ymm);
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f16_to_f8(ymm, ymm);
                            } else {
                                if (idt != f32) cvt2ps(ymm, ymm, idt);
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f32_to_f8(
                                            Xmm(ymm.getIdx()),
                                            Zmm(ymm.getIdx()));
                            }
                        } else
                            assert(!"invalid isa or fp8 emulation not "
                                    "available");
                        break;
                    }
                case s8:
                    if (utils::one_of(idt, bf16, f16)) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16, f16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmovsdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero_);
                            vpermq(ymm, ymm, 0x58);
                            vpacksswb(ymm, ymm, ymm_zero_);
                        }
                    }
                    if (idt == u8) vpminub(ymm, ymm, ymm_8x127b_);
                    break;
                case u8:
                    if (utils::one_of(idt, bf16, f16)) cvt2ps(ymm, ymm, idt);
                    if (utils::one_of(idt, f32, bf16, f16)) vcvtps2dq(ymm, ymm);
                    if (utils::one_of(idt, bf16, f16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmaxsd(ymm, ymm, ymm_zero_);
                            vpmovusdb(xmm, ymm);
                        } else {
                            vpackssdw(ymm, ymm, ymm_zero_);
                            vpermq(ymm, ymm, 0x58);
                            vpackuswb(ymm, ymm, ymm_zero_);
                        }
                    }
                    if (idt == s8) vpmaxsb(ymm, ymm, ymm_zero_);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load = [this](const Ymm ymm, const Address &addr, int size) {
            const Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(ymm, addr); break;
                case 16: vmovups(xmm, addr); break;
                case 8: vmovsd(xmm, addr); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [this](const Address &addr, const Ymm ymm, int size) {
            const Xmm xmm = Xmm(ymm.getIdx());
            switch (size) {
                case 32: vmovups(addr, ymm); break;
                case 16: vmovups(addr, xmm); break;
                case 8: vmovsd(addr, xmm); break;
                default: assert(!"unreachable");
            }
        };

        const int unroll = 8;

        const bool interim_f32 = (prb_.itype != f32)
                || utils::one_of(f32, prb_.itype, prb_.otype);

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        for (int i = 0; i < unroll; i++) {
            const ptrdiff_t node_0_input_stride = prb_.is(0);
            load(Ymm(i), i_addr(i_off + i * node_0_input_stride),
                    unroll * itype_sz_);

            if (interim_f32) cvt2ps(Ymm(i), Ymm(i), prb_.itype);
        }

        for (int i = 0; i < unroll / 2; i++) {
            vunpcklps(Ymm(unroll + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < unroll / 2; i++) {
            const int j = i % 2 == 0 ? unroll + i : i - 1;
            vshufps(Ymm(unroll / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(unroll / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < unroll / 2; i++)
            vperm2f128(Ymm(i), Ymm(unroll / 2 + i), Ymm(unroll + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = unroll / 2; i < unroll; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(unroll / 2 + i), uquad);

        const bool use_sat_cvt
                = need_saturation && isa_has_sat_cvt(isa_, prb_.otype);
        if (need_saturation) {
            init_saturate_f32(ymm_zero_, ymm_saturation_ubound_, reg_tmp_, f32,
                    prb_.otype, false, use_sat_cvt);
            for (int i = 0; i < unroll; i++)
                saturate_cvt_f32(Ymm(i), ymm_zero_, ymm_saturation_ubound_,
                        prb_.otype, false, use_sat_cvt);
        }

        for (int i = 0; i < unroll; i++) {
            const ptrdiff_t node_1_output_stride = prb_.os(1);
            if (prb_.otype != f32)
                cvt2odt(Ymm(i), prb_.otype,
                        need_saturation       ? s32
                                : interim_f32 ? f32
                                              : prb_.itype,
                        use_sat_cvt);
            store(o_addr(o_off + i * node_1_output_stride), Ymm(i),
                    unroll * otype_sz_);
        }
    }

    bool can_do_tr8x8() {
        using namespace data_type;

        static constexpr dim_t desirable_node_size = 8;
        static constexpr ptrdiff_t desirable_stride = 1;

        // This process relies on swapping the two innermost dimensions.
        // Therefore, the input stride in the second node and output stride in
        // first node have to be equal to 1.
        return mayiuse(avx2) && prb_.ndims >= 2
                && ((utils::one_of(prb_.itype, u8, s8, f8_e5m2, f8_e4m3, s32,
                             f32, bf16, f16)
                        && utils::one_of(prb_.otype, u8, s8, f8_e5m2, f8_e4m3,
                                s32, f32, bf16, f16)))
                && utils::everyone_is(desirable_node_size, prb_.n(0), prb_.n(1))
                && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(1))
                && !prb_.is_tail_present
                && prb_.src_scale_type == scale_type_t::NONE
                && prb_.dst_scale_type == scale_type_t::NONE && prb_.beta == 0.f
                && !compensation_needed_;
    }

    bool process_unroll_tr8x8(const int ndims, const dim_t len) {
        if (!can_do_tr8x8()) return false;

        const dim_t step_size = prb_.n(0) * prb_.n(1);
        dim_t i_off = 0, o_off = 0;
        for (dim_t off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const dim_t *i_off,
            const dim_t *o_off, const dim_t *s_off, const dim_t *c_off,
            const int *zero_padding, const bool tail_processing) {
        using namespace data_type;

        const auto cvt2ps
                = [this](const Xmm dst, const Operand &src, data_type_t idt) {
            Xmm dst_pure = Xmm(dst.getIdx());
            switch (idt) {
                case f32:
                    if (src.isMEM() || src.getIdx() != dst.getIdx())
                        uni_vmovups(dst, src);
                    break;
                case bf16:
                    if (mayiuse(avx2)) {
                        vpmovzxwd(dst, src);
                        vpslld(dst, dst, 0x10);
                        break;
                    } else
                        assert(!"unreachable!");
                case f16: vcvtph2ps(dst, src); break;
                case s32: uni_vcvtdq2ps(dst, src); break;
                case f8_e5m2:
                    if (f8_e5m2_cvt_ && is_f8_supported(isa_))
                        f8_e5m2_cvt_->vcvt_f8_to_f32(Zmm(dst.getIdx()), src);
                    else
                        assert(!"invalid isa or fp8 emulation not "
                                "available");
                    break;
                case f8_e4m3:
                    if (f8_e4m3_cvt_ && is_f8_supported(isa_))
                        f8_e4m3_cvt_->vcvt_f8_to_f32(Zmm(dst.getIdx()), src);
                    else
                        assert(!"invalid isa or fp8 emulation not "
                                "available");
                    break;
                case s8:
                    uni_vpmovsxbd(dst, src);
                    uni_vcvtdq2ps(dst_pure, dst);
                    break;
                case u8:
                    uni_vpmovzxbd(dst, src);
                    uni_vcvtdq2ps(dst_pure, dst);
                    break;
                default: assert(!"unreachable");
            }
        };

        const auto cvt2odt = [this, cvt2ps](const Xmm xmm, data_type_t odt,
                                     data_type_t idt, bool need_saturation) {
            if (need_saturation && !compensation_needed_
                    && isa_has_sat_cvt(isa_, prb_.otype)) {
                switch (prb_.otype) {
                    case s8:
                    case u8: vpmovusdb(xmm, xmm); break;
                    case s32: break; // nothing to do
                    default: assert(!"unsupported data type");
                }
                return;
            }
            switch (odt) {
                case bf16:
                    if (!mayiuse(avx2)) assert(!"unreachable");
                    if (utils::one_of(
                                idt, f32, f16, f8_e5m2, f8_e4m3, s8, u8)) {
                        if (!utils::one_of(idt, f32, f8_e5m2, f8_e4m3))
                            cvt2ps(xmm, xmm, idt);
                        if (utils::one_of(idt, f8_e5m2, f8_e4m3)) {
                            if (is_f8_supported(isa_)) {
                                if (idt == f8_e5m2 && f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f32(
                                            Zmm(xmm.getIdx()), xmm);
                                else if (idt == f8_e4m3 && f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f32(
                                            Zmm(xmm.getIdx()), xmm);
                                else
                                    assert(!"fp8 emulation not available");
                            } else
                                assert(!"invalid isa for fp8 emulation");
                        }
                        if (is_superset(isa_, avx2_vnni_2)) {
                            vcvtneps2bf16(xmm, xmm, Xbyak::VexEncoding);
                        } else if (mayiuse(avx512_core_bf16)) {
                            vcvtneps2bf16(xmm, xmm);
                        } else {
                            bf16_emu_->vcvtneps2bf16(
                                    Ymm(xmm.getIdx()), Zmm(xmm.getIdx()));
                        }
                    }
                    break;
                case f16:
                    if (!mayiuse(avx2)) assert(!"unreachable");
                    if (utils::one_of(
                                idt, f32, bf16, f8_e5m2, f8_e4m3, s8, u8)) {
                        if (!utils::one_of(idt, f32, f8_e5m2, f8_e4m3))
                            cvt2ps(xmm, xmm, idt);
                        if (utils::one_of(idt, f8_e5m2, f8_e4m3)) {
                            if (is_f8_supported(isa_)) {
                                if (idt == f8_e5m2 && f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f16(xmm, xmm);
                                else if (idt == f8_e4m3 && f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f16(xmm, xmm);
                                else
                                    assert(!"fp8 emulation not available");
                            } else
                                assert(!"invalid isa for fp8 emulation");
                        } else
                            vcvtps2ph(xmm, xmm, _op_mxcsr);
                    }
                    break;
                case s32:
                    if (idt == f32)
                        uni_vcvtps2dq(xmm, xmm);
                    else if (idt == s8)
                        uni_vpmovsxbd(xmm, xmm);
                    else if (idt == u8)
                        uni_vpmovzxbd(xmm, xmm);
                    break;
                case f8_e5m2:
                    if (utils::one_of(idt, f32, bf16, f16, f8_e4m3)) {
                        if (is_f8_supported(isa_)) {
                            if (idt == f8_e4m3) {
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f8_to_f16(xmm, xmm);
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f16_to_f8(xmm, xmm);
                            } else {
                                if (idt != f32) cvt2ps(xmm, xmm, idt);
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f32_to_f8(
                                            xmm, Zmm(xmm.getIdx()));
                            }
                        } else
                            assert(!"invalid isa or fp8 emulation not "
                                    "available");
                    }
                    break;
                case f8_e4m3:
                    if (utils::one_of(idt, f32, bf16, f16, f8_e5m2)) {
                        if (is_f8_supported(isa_)) {
                            if (idt == f8_e5m2) {
                                if (f8_e5m2_cvt_)
                                    f8_e5m2_cvt_->vcvt_f8_to_f16(xmm, xmm);
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f16_to_f8(xmm, xmm);
                            } else {
                                if (idt != f32) cvt2ps(xmm, xmm, idt);
                                if (f8_e4m3_cvt_)
                                    f8_e4m3_cvt_->vcvt_f32_to_f8(
                                            xmm, Zmm(xmm.getIdx()));
                            }
                        } else
                            assert(!"invalid isa or fp8 emulation not "
                                    "available");
                    }
                    break;
                case s8:
                    if (utils::one_of(idt, bf16, f16)) cvt2ps(xmm, xmm, idt);
                    if (utils::one_of(idt, f32, bf16, f16))
                        uni_vcvtps2dq(xmm, xmm);
                    if (utils::one_of(idt, bf16, f16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmovsdb(xmm, xmm);
                        } else {
                            uni_vpackssdw(xmm, xmm, xmm_zero_);
                            uni_vpacksswb(xmm, xmm, xmm_zero_);
                        }
                    }
                    if (idt == u8) uni_vpminub(xmm, xmm, xmm_4x127b_);
                    break;
                case u8:
                    if (utils::one_of(idt, bf16, f16)) cvt2ps(xmm, xmm, idt);
                    if (utils::one_of(idt, f32, bf16, f16))
                        uni_vcvtps2dq(xmm, xmm);
                    if (utils::one_of(idt, bf16, f16, f32, s32)) {
                        if (mayiuse(avx512_core)) {
                            vpmaxsd(xmm, xmm, xmm_zero_);
                            vpmovusdb(xmm, xmm);
                        } else {
                            uni_vpackssdw(xmm, xmm, xmm_zero_);
                            uni_vpackuswb(xmm, xmm, xmm_zero_);
                        }
                    }
                    if (idt == s8) uni_vpmaxsb(xmm, xmm, xmm_zero_);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load = [this](const Xmm xmm, const Address &addr, int size) {
            switch (size) {
                case 16: uni_vmovups(xmm, addr); break;
                case 8: uni_vmovsd(xmm, addr); break;
                case 4: uni_vmovss(xmm, addr); break;
                case 2: uni_vpinsrw(xmm, xmm, addr, 0x0); break;
                case 1: uni_vpinsrb(xmm, xmm, addr, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        auto load_bytes = [this](const Xmm xmm, const Address &addr, int size,
                                  int imm) {
            switch (size) {
                case 4: uni_vpinsrd(xmm, xmm, addr, imm); break;
                case 2: uni_vpinsrw(xmm, xmm, addr, imm); break;
                case 1: uni_vpinsrb(xmm, xmm, addr, imm); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [this](const Address &addr, const Xmm xmm, int size) {
            switch (size) {
                case 16: uni_vmovups(addr, xmm); break;
                case 8: uni_vmovsd(addr, xmm); break;
                case 4: uni_vmovss(addr, xmm); break;
                case 2: uni_vpextrw(addr, xmm, 0x0); break;
                case 1: uni_vpextrb(addr, xmm, 0x0); break;
                default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        static constexpr int xmm_vlen = 4;
        bool can_load_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1) {
                can_load_xmm = false;
                break;
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

        const bool interim_f32 = interim_f32_needed();

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, s8, s32) && interim_f32);

        std::vector<int> store_masks;
        if (tail_processing) {
            for (int ur = 0; ur < reg_unroll; ur += load_tail_step) {
                uni_vpxor(Xmm(ur), Xmm(ur), Xmm(ur));
                store_masks.push_back(0);
                for (int r = 0; r < load_tail_step; ++r) {
                    if (zero_padding[ur + r] == 0) {
                        store_masks.back() += 1 << r;
                        load_bytes(
                                Xmm(ur), i_addr(i_off[ur + r]), itype_sz_, r);
                    }
                }
            }
        } else {
            if (!can_load_xmm && can_store_xmm) {
                assert(ur_step == xmm_vlen);
                /* load with stride */
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    for (int r = 0; r < ur_step; ++r) {
                        load_bytes(
                                Xmm(ur), i_addr(i_off[ur + r]), itype_sz_, r);
                    }
                }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    load(Xmm(ur), i_addr(i_off[ur]), load_step * itype_sz_);
                }
            }
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(Xmm(ur), Xmm(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            // transposition on the fly
            const bool fast_return = prb_.src_scale_type != scale_type_t::MANY
                    && prb_.dst_scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f && !prb_.req_src_zp && !prb_.req_dst_zp
                    && !compensation_needed_;
            if (fast_return) {
                const bool use_sat_cvt
                        = need_saturation && isa_has_sat_cvt(isa_, prb_.otype);
                if (prb_.src_scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        uni_vmulps(Xmm(ur), Xmm(ur), xmm_src_scales_);
                if (prb_.dst_scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        uni_vmulps(Xmm(ur), Xmm(ur), xmm_dst_scales_);
                if (prb_.otype != f32) {
                    init_saturate_f32(xmm_zero_, xmm_saturation_ubound_,
                            reg_tmp_, interim_f32 ? f32 : prb_.itype,
                            prb_.otype, false, use_sat_cvt);
                    for (int ur = 0; ur < reg_unroll; ur += load_step) {
                        if (need_saturation)
                            saturate_cvt_f32(Xmm(ur), xmm_zero_,
                                    xmm_saturation_ubound_, prb_.otype, false,
                                    use_sat_cvt);
                        cvt2odt(Xmm(ur), prb_.otype,
                                need_saturation       ? s32
                                        : interim_f32 ? f32
                                                      : prb_.itype,
                                use_sat_cvt);
                    }
                }
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz_ == 4)
                            uni_vpextrd(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else if (otype_sz_ == 2)
                            uni_vpextrw(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else
                            uni_vpextrb(o_addr(o_off[ur + r]), Xmm(ur), r);
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz_ == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        uni_vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur),
                                static_cast<uint8_t>(r));
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        if (mayiuse(avx2))
                            vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur),
                                    static_cast<uint8_t>(itype_sz_ * r));
                        else {
                            movups(Xmm(ur + r), Xmm(ur));
                            palignr(Xmm(ur + r), Xmm(ur), itype_sz_ * r);
                        }
                    }
            }
        }

        /* src zero point application */
        if (prb_.req_src_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = Xmm(ur);
                if (interim_f32)
                    uni_vsubps(xmm, xmm, xmm_src_zp_);
                else
                    uni_vpsubd(xmm, xmm, xmm_src_zp_);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            const auto apply_scales
                    = [&](const Xmm &vreg_scales, scale_arg_t scale_arg,
                              scale_type_t scale_type) {
                if (scale_type == scale_type_t::COMMON) {
                    for (int ur = 0; ur < reg_unroll; ur += ur_step)
                        uni_vmulps(Xmm(ur), Xmm(ur), vreg_scales);
                } else if (scale_type == scale_type_t::MANY) {
                    enum class scale_load_type_t { bcast, load, gather };

                    uni_vpxor(vreg_scales, vreg_scales, vreg_scales);
                    for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                        scale_load_type_t scale_load_type
                                = scale_load_type_t::bcast; // the best case

                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (s_off[r] != s_off[r - 1] + 0)
                                scale_load_type = scale_load_type_t::load;

                        if (scale_load_type == scale_load_type_t::bcast
                                && !tail_processing) {
                            uni_vbroadcastss(vreg_scales,
                                    scale_arg == scale_arg_t::SRC
                                            ? src_s_addr(s_off[ur])
                                            : dst_s_addr(s_off[ur]));
                            uni_vmulps(Xmm(ur), Xmm(ur), vreg_scales);
                            continue;
                        }

                        // bcast doesn't work, the next try -- load
                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (s_off[r] != s_off[r - 1] + 1)
                                scale_load_type = scale_load_type_t::gather;

                        if (scale_load_type == scale_load_type_t::load
                                && !tail_processing) {
                            uni_vmovups(vreg_scales,
                                    scale_arg == scale_arg_t::SRC
                                            ? src_s_addr(s_off[ur])
                                            : dst_s_addr(s_off[ur]));
                            uni_vmulps(Xmm(ur), Xmm(ur), vreg_scales);
                            continue;
                        }

                        // load doesn't work as well
                        // so gather the scale factors one by one
                        for (int r = ur; r < ur + ur_step; ++r) {
                            if (zero_padding[r] == 0 || !tail_processing)
                                uni_vpinsrd(vreg_scales, vreg_scales,
                                        scale_arg == scale_arg_t::SRC
                                                ? src_s_addr(s_off[r])
                                                : dst_s_addr(s_off[r]),
                                        r - ur);
                        }
                        uni_vmulps(Xmm(ur), Xmm(ur), vreg_scales);
                    }
                }
            };
            /* xmm <-- src_scales * xmm[:] */
            apply_scales(
                    xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

            /* xmm[:] <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        /* non VEX instructions do not support unaligned
                         * memory for instructions other than movups. */
                        if (mayiuse(avx2)) {
                            vaddps(Xmm(ur), o_addr(o_off[ur]));
                        } else {
                            /* register xmm(1) is unused */
                            movups(Xmm(1), o_addr(o_off[ur]));
                            addps(Xmm(ur), Xmm(1));
                        }
                    } else if (utils::one_of(prb_.otype, f8_e5m2, f8_e4m3)) {
                        /* Unlike every other data type, whose memory read
                         * width follows the `xmm` destination, the fp8
                         * `vcvt_f8_to_f32` routine is fixed 16-wide: it always
                         * reads a full 16-byte vector regardless of the number
                         * of requested elements, which over-reads an unpadded
                         * dst buffer here (ur_step == xmm_vlen). Stage only the
                         * valid bytes into a register and convert from the
                         * register, so the routine's full-width read hits the
                         * register instead of memory. This mirrors the scalar
                         * (ur_step == 1) path above, which stages via vpinsrb
                         * before converting. */
                        load(Xmm(1), o_addr(o_off[ur]), ur_step * otype_sz_);
                        cvt2ps(Xmm(1), Xmm(1), prb_.otype);
                        uni_vaddps(Xmm(ur), Xmm(ur), Xmm(1));
                    } else {
                        cvt2ps(Xmm(1), o_addr(o_off[ur]), prb_.otype);
                        uni_vaddps(Xmm(ur), Xmm(ur), Xmm(1));
                    }
                }
            }

            /* dst <-- dst_scales * xmm[:] */
            apply_scales(
                    xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
        } else {
            const auto apply_scales
                    = [&](const Xmm &vreg_scales, scale_arg_t scale_arg,
                              scale_type_t scale_type) {
                if (scale_type == scale_type_t::COMMON) {
                    for (int ur = 0; ur < reg_unroll; ur += ur_step)
                        uni_vmulss(Xmm(ur), Xmm(ur), vreg_scales);
                } else if (scale_type == scale_type_t::MANY) {
                    for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                        if (zero_padding[ur] == 0 || !tail_processing)
                            uni_vmulss(Xmm(ur), Xmm(ur),
                                    scale_arg == scale_arg_t::SRC
                                            ? src_s_addr(s_off[ur])
                                            : dst_s_addr(s_off[ur]));
                    }
                }
            };

            /* xmm[0] <-- src_scales * xmm[0] */
            apply_scales(
                    xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

            /* xmm[0] <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (prb_.otype == f32) {
                        uni_vaddss(Xmm(ur), Xmm(ur), o_addr(o_off[ur]));
                    } else {
                        if (prb_.otype == s32) {
                            uni_vmovss(xmm_tmp_, o_addr(o_off[ur]));
                        } else if (utils::one_of(prb_.otype, s8, u8, f8_e5m2,
                                           f8_e4m3)) {
                            uni_vpinsrb(
                                    xmm_tmp_, xmm_tmp_, o_addr(o_off[ur]), 0x0);
                        } else if (utils::one_of(prb_.otype, bf16, f16)) {
                            uni_vpinsrw(
                                    xmm_tmp_, xmm_tmp_, o_addr(o_off[ur]), 0x0);
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp_, xmm_tmp_, prb_.otype);
                        uni_vaddps(Xmm(ur), Xmm(ur), xmm_tmp_);
                    }
                }
            }

            /* dst <-- dst_scales * xmm[0] */
            apply_scales(
                    xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
        }

        /* dst zero point application */
        if (prb_.req_dst_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = Xmm(ur);
                if (interim_f32)
                    uni_vaddps(xmm, xmm, xmm_dst_zp_);
                else
                    uni_vpaddd(xmm, xmm, xmm_dst_zp_);
            }
        }

        /* adjust scale application */
        if (prb_.scale_adjust != 1.f) {
            uni_vmovd(xmm_tmp_, reg_scale_adjust_);
            uni_vpshufd(xmm_tmp_, xmm_tmp_, 0x0);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                uni_vmulps(Xmm(ur), Xmm(ur), xmm_tmp_);
            }
        }

        const bool use_sat_cvt = need_saturation && !compensation_needed_
                && isa_has_sat_cvt(isa_, prb_.otype);
        if (need_saturation) {
            init_saturate_f32(xmm_zero_, xmm_saturation_ubound_, reg_tmp_, f32,
                    prb_.otype, compensation_needed_, use_sat_cvt);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                if (compensation_needed_)
                    saturate_f32(Xmm(ur), xmm_zero_, xmm_saturation_ubound_,
                            prb_.otype, compensation_needed_);
                else
                    saturate_cvt_f32(Xmm(ur), xmm_zero_, xmm_saturation_ubound_,
                            prb_.otype, compensation_needed_, use_sat_cvt);
            }

            // reset back xmm_zero_ if needed.
            if (compensation_needed_ && (prb_.req_src_zp || prb_.req_dst_zp))
                uni_vxorps(xmm_zero_, xmm_zero_, xmm_zero_);
        }

        if (compensation_needed_) {
            const bool mayiuse_avx2 = mayiuse(avx2);
            const auto uni_vpaddd_wrapper
                    = [&](const Xmm xmm, const Address &addr) {
                if (mayiuse_avx2)
                    vpaddd(xmm, xmm, addr);
                else {
                    //isas < avx2 demand paddd instruction addr to be aligned
                    assert(xmm.getIdx() != xmm_tmp_.getIdx());
                    uni_vmovups(xmm_tmp_, addr);
                    paddd(xmm, xmm_tmp_);
                }
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
                        // xmm_compensation is used for reduction.
                        uni_vcvtps2dq(xmm_compensation, Xmm(ur));
                        uni_vphaddd(xmm_compensation, xmm_compensation,
                                xmm_compensation);
                        uni_vphaddd(xmm_compensation, xmm_compensation,
                                xmm_compensation);
                        const auto comp_addr = c_addr(c_off[ur]);
                        uni_vmovss(xmm_tmp_, comp_addr);
                        uni_vpaddd(xmm_tmp_, xmm_tmp_, xmm_compensation);
                        uni_vmovss(comp_addr, xmm_tmp_);
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
                        const auto comp_addr = c_addr(c_off[ur]);
                        uni_vcvtps2dq(xmm_compensation, Xmm(ur));
                        uni_vpaddd_wrapper(xmm_compensation, comp_addr);
                        uni_vmovups(comp_addr, xmm_compensation);
                        continue;
                    }

                    uni_vcvtps2dq(xmm_compensation, Xmm(ur));
                    for (int r = ur; r < ur + ur_step; ++r) {
                        if (zero_padding[r] == 0 || !tail_processing) {
                            uni_vshufps(xmm_tmp_, xmm_compensation,
                                    xmm_compensation, static_cast<uint8_t>(r));
                            const Reg32 reg_tmp_32 = reg_tmp_.cvt32();
                            uni_vmovd(reg_tmp_32, xmm_tmp_);
                            const auto comp_addr = c_addr(c_off[r]);
                            add(comp_addr, reg_tmp_32);
                        }
                    }
                }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (zero_padding[ur] == 0 || !tail_processing) {
                        const auto comp_addr = c_addr(c_off[ur]);
                        uni_vcvtps2dq(xmm_compensation, Xmm(ur));
                        uni_vpaddd_wrapper(xmm_compensation, comp_addr);
                        uni_vmovss(comp_addr, xmm_compensation);
                    }
                }
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.req_src_zp || prb_.req_dst_zp) {
                const bool use_store_masks = !store_masks.empty();
                if (use_store_masks) {
                    const auto mask = ~store_masks[ur / ur_step];
                    uni_vblendps(Xmm(ur), Xmm(ur), xmm_zero_, mask);
                }
            }
            if (prb_.otype != f32)
                cvt2odt(Xmm(ur), prb_.otype,
                        need_saturation && !compensation_needed_ ? s32
                                : interim_f32                    ? f32
                                                                 : prb_.itype,
                        use_sat_cvt);

            store(o_addr(o_off[ur]), Xmm(ur), ur_step * otype_sz_);
        }
    }

    bool interim_f32_needed() {
        using namespace data_type;

        return utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.src_scale_type != scale_type_t::NONE
                || prb_.dst_scale_type != scale_type_t::NONE || prb_.beta != 0.f
                || ((prb_.req_src_zp || prb_.req_dst_zp)
                                ? !(prb_.itype == s32 && prb_.otype == s32)
                                : false)
                || (prb_.itype != f32 && compensation_needed_)
                || prb_.scale_adjust != 1.f;
    }

    void process_unroll_generic(
            const int ndims, dim_t len, const bool tail_processing) {
        assert(IMPLICATION(prb_.nodes[0].tail_size > 0,
                len == prb_.nodes[0].n || len == prb_.nodes[0].tail_size));

        const int blk = 8;

        dim_t i_off[2 * blk] = {0};
        dim_t o_off[2 * blk] = {0};
        dim_t s_off[2 * blk] = {0};
        dim_t c_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        const bool interim_f32 = interim_f32_needed();

        if (prb_.req_src_zp) {
            uni_vbroadcastss(xmm_src_zp_, PARAM(src_zp));
            if (interim_f32) uni_vcvtdq2ps(xmm_src_zp_, xmm_src_zp_);
        }
        if (prb_.req_dst_zp) {
            uni_vbroadcastss(xmm_dst_zp_, PARAM(dst_zp));
            if (interim_f32) uni_vcvtdq2ps(xmm_dst_zp_, xmm_dst_zp_);
        }

        for (dim_t off = 0; off < len; off += blk) {
            const int reg_unroll
                    = static_cast<int>(nstl::min<dim_t>(off + blk, len) - off);
            int zero_padding[blk] = {0};
            const auto curr_blk = curr * blk;

            /* compute offsets and tail*/
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr_blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                const bool is_tail = off + ur >= prb_.nodes[0].tail_size;
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        c_off[ur_p], i_off[ur_c], o_off[ur_c], s_off[ur_c],
                        c_off[ur_c]);
                if (tail_processing && is_tail) zero_padding[ur] = 1;
            }

            process_unroll_generic_step(reg_unroll, i_off + curr_blk,
                    o_off + curr_blk, s_off + curr_blk, c_off + curr_blk,
                    zero_padding, tail_processing);

            curr = 1 - curr;
        }
    }

    void compute_ker(const int ndims, const dim_t len_unroll,
            const bool tail_processing) {
        bool optimized = process_unroll_tr8x8(ndims, len_unroll);
        if (!optimized)
            process_unroll_generic(ndims, len_unroll, tail_processing);
    }

    void loop_begin(Label &l, Reg64 reg_cnt, dim_t len) {
        mov(reg_cnt, len);
        L(l);
    }

    void check_if_this_is_last_chunk(const Reg64 reg_curr_chunk, int node_id) {
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

    void zero_dst_memory(const dim_t bytes_to_zeroing) {
        static constexpr int num_of_bytes_in_xmm = 128 / 8;

        const dim_t xmms_to_zeroing = bytes_to_zeroing / num_of_bytes_in_xmm;
        const int tail_to_zeroing
                = static_cast<int>(bytes_to_zeroing % num_of_bytes_in_xmm);

        uni_vpxor(xmm_tmp_, xmm_tmp_, xmm_tmp_);

        if (xmms_to_zeroing > 0) {
            Label loop;

            mov(reg_tmp_, xmms_to_zeroing);
            L(loop);
            uni_vmovups(o_addr(0), xmm_tmp_);
            add(reg_off_out_, num_of_bytes_in_xmm);
            dec(reg_tmp_);
            jnz(loop);
        }

        for (int i = 0; i < tail_to_zeroing; i++)
            uni_vpextrb(o_addr(i, false), xmm_tmp_, 0);

        // Restore dst offset to initial value
        if (xmms_to_zeroing > 0)
            sub(reg_off_out_, num_of_bytes_in_xmm * xmms_to_zeroing);
    }

    void finalize_tail_loop(dim_t i_step, dim_t o_step, dim_t s_step,
            dim_t c_step, const int curr_node_id) {
        static constexpr int empty_chunk_info = -1;

        mov(reg_tmp_, empty_chunk_info);
        mov(data_chunk_addr(curr_node_id), reg_tmp_);

        const dim_t padded_area = prb_.nodes[curr_node_id].n
                - prb_.nodes[curr_node_id].tail_size;

        if (prb_.nodes[curr_node_id].is_zero_pad_needed) {
            dim_t num_of_zero_padded_values = padded_area;
            for (int i = curr_node_id - 1; i >= 0; i--) {
                num_of_zero_padded_values *= prb_.nodes[i].n;
            }

            const dim_t bytes_to_zeroing
                    = num_of_zero_padded_values * otype_sz_;
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
        add(reg_off_in_, padded_area * i_step * itype_sz_);
        add(reg_off_out_, padded_area * o_step * otype_sz_);
        if (prb_.src_scale_type == scale_type_t::MANY
                || prb_.dst_scale_type == scale_type_t::MANY)
            add(reg_off_scale_, padded_area * s_step * stype_sz_);
        if (compensation_needed_)
            add(reg_off_comp_, padded_area * c_step * ctype_sz_);
    }

    void loop_end(Label &l, const Reg64 reg_cnt, dim_t len, dim_t i_step,
            dim_t o_step, dim_t s_step, dim_t c_step, const int curr_node_id) {
        add(reg_off_in_, i_step * itype_sz_);
        add(reg_off_out_, o_step * otype_sz_);
        if (prb_.src_scale_type == scale_type_t::MANY
                || prb_.dst_scale_type == scale_type_t::MANY)
            add(reg_off_scale_, s_step * stype_sz_);
        if (compensation_needed_) add(reg_off_comp_, c_step * ctype_sz_);

        dec(reg_cnt);
        jnz(l);

        if (prb_.tail(curr_node_id) != 0) {
            Label if_end;

            // On the stack should be an information if node
            // was processed with tail or not.
            pop(reg_tmp_);

            cmp(reg_tmp_, with_tail_info_);
            jne(if_end, T_NEAR);
            finalize_tail_loop(i_step, o_step, s_step, c_step, curr_node_id);
            L(if_end);
        }

        // Restore offset to initial values. It means before
        // loop execution.
        sub(reg_off_in_, len * i_step * itype_sz_);
        sub(reg_off_out_, len * o_step * otype_sz_);
        if (prb_.src_scale_type == scale_type_t::MANY
                || prb_.dst_scale_type == scale_type_t::MANY)
            sub(reg_off_scale_, len * s_step * stype_sz_);
        if (compensation_needed_) sub(reg_off_comp_, len * c_step * ctype_sz_);
    }

    void compute_blk_ker(const simple_impl_desc_t &desc) {
        static constexpr bool with_tail_processing = true;
        Label no_last_chunk, end_label;
        int omp_ndims = prb_.full_ndims - prb_.ndims;

        if (prb_.nodes[0].tail_size > 0) {
            if (!prb_.nodes[0].is_parent_empty()) {
                const int parent_node_id = prb_.nodes[0].parent_node_id;
                mov(reg_tmp_, data_chunk_addr(parent_node_id));
                check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                jne(no_last_chunk, T_NEAR);
            }

            const dim_t len_unroll = desc.tail_len_unroll > 0
                    ? desc.tail_len_unroll
                    : desc.len_unroll;
            compute_ker(omp_ndims, len_unroll, with_tail_processing);
            jmp(end_label, T_NEAR);
        }

        L(no_last_chunk);
        compute_ker(omp_ndims, desc.len_unroll, !with_tail_processing);
        L(end_label);
    }

    void create_loops(const simple_impl_desc_t &desc,
            const std::array<const Reg64, 3> &reg_cnt, int jit_loop) {
        assert(jit_loop <= ndims_jit_loop_max);

        if (jit_loop > 0) {
            const int nfu = desc.ndims_full_unroll;
            const dim_t unroll_factor
                    = jit_loop == 1 ? desc.len_last_dim_unroll : 1;
            const int curr_node_id = nfu + (jit_loop - 1);
            const int parent_node_id = prb_.nodes[curr_node_id].parent_node_id;
            const dim_t tail_size = prb_.tail(curr_node_id) / unroll_factor;
            const dim_t node_size = prb_.n(curr_node_id) / unroll_factor;
            const Reg64 reg_loop_cnt = reg_cnt[jit_loop - 1];
            const bool curr_node_has_tail = prb_.tail(curr_node_id) != 0;
            Label loop, if_no_tail, if_end;

            if (curr_node_has_tail) {
                if (prb_.nodes[curr_node_id].is_parent_empty()) {
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(reg_tmp_, with_tail_info_);
                    push(reg_tmp_);
                } else {
                    mov(reg_tmp_, data_chunk_addr(parent_node_id));
                    check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                    jne(if_no_tail, T_NEAR);
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(reg_tmp_, with_tail_info_);
                    push(reg_tmp_);
                    jmp(if_end, T_NEAR);

                    L(if_no_tail);
                    mov(reg_loop_cnt, node_size);
                    // Put info that node is being processed without tail.
                    mov(reg_tmp_, without_tail_info_);
                    push(reg_tmp_);
                    L(if_end);
                }
            }

            if (prb_.is_tail_in_one_of_child_nodes(curr_node_id)) {
                if (!curr_node_has_tail) {
                    mov(reg_loop_cnt, node_size);
                    mov(data_chunk_addr(curr_node_id), reg_loop_cnt);
                }
                L(loop);
                if (!prb_.nodes[curr_node_id].is_parent_empty()) {
                    Label if_no_tail_in_child_node;
                    mov(reg_tmp_, data_chunk_addr(parent_node_id));
                    check_if_this_is_last_chunk(reg_tmp_, parent_node_id);
                    jne(if_no_tail_in_child_node, T_NEAR);
                    mov(data_chunk_addr(curr_node_id), reg_loop_cnt);
                    L(if_no_tail_in_child_node);
                } else {
                    mov(data_chunk_addr(curr_node_id), reg_loop_cnt);
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

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        xor_(reg_off_in_, reg_off_in_);
        xor_(reg_off_out_, reg_off_out_);
        if (prb_.src_scale_type == scale_type_t::MANY
                || prb_.dst_scale_type == scale_type_t::MANY)
            xor_(reg_off_scale_, reg_off_scale_);
        if (compensation_needed_) xor_(reg_off_comp_, reg_off_comp_);

        std::array<const Reg64, 3> reg_cnt({{r15, r14, r13}});

        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        create_loops(d, reg_cnt, n_jit_loops);

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32_t(const desc_t &desc)
        : kernel_t(desc)
        , jit_generator_t(jit_name())
        , isa_(get_max_cpu_isa())
        , bf16_emu_(nullptr)
        , f8_e5m2_cvt_(nullptr)
        , f8_e4m3_cvt_(nullptr) {
        assert(!utils::one_of(isa_, isa_undef, isa_all));
        itype_sz_ = static_cast<int>(data_type_size(prb_.itype));
        otype_sz_ = static_cast<int>(data_type_size(prb_.otype));
        stype_sz_ = sizeof(float);
        if (prb_.otype == data_type::bf16 && !mayiuse(avx512_core_bf16)
                && !mayiuse(avx2_vnni_2)) {
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                    bf16_emu_scratch_, bf16_emu_reserv_4_);
        }
        if ((utils::one_of(prb_.otype, data_type::f8_e5m2, data_type::f8_e4m3)
                    || utils::one_of(
                            prb_.itype, data_type::f8_e5m2, data_type::f8_e4m3))
                && is_f8_supported(isa_)) {
            const auto create_fp8_cvt = [&](const data_type_t &dtype) {
                switch (dtype) {
                    case data_type::f8_e5m2:
                        f8_e5m2_cvt_
                                = utils::make_unique<fp8_conversion_e5m2_t>(
                                        this, fp8_emu_reserv_1_,
                                        fp8_emu_reserv_2_, fp8_emu_reserv_3_,
                                        fp8_cvt_kmask_aux_, fp8_cvt_scratch_);
                        break;
                    case data_type::f8_e4m3:
                        f8_e4m3_cvt_
                                = utils::make_unique<fp8_conversion_e4m3_t>(
                                        this, fp8_emu_reserv_1_,
                                        fp8_emu_reserv_2_, fp8_emu_reserv_3_,
                                        fp8_emu_reserv_4_, fp8_emu_reserv_5_,
                                        fp8_cvt_scratch_);
                        break;
                    default: assert(!"Unreachable.");
                }
            };
            if (utils::one_of(
                        prb_.otype, data_type::f8_e5m2, data_type::f8_e4m3))
                create_fp8_cvt(prb_.otype);
            if (utils::one_of(
                        prb_.itype, data_type::f8_e5m2, data_type::f8_e4m3))
                create_fp8_cvt(prb_.itype);
        }
    }

    void generate() override {
        Label end_of_kernel;

        preamble();

        if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();

        if (prb_.src_scale_type == scale_type_t::COMMON) {
            auto reg_ptr_src_scales__tmp = reg_ptr_in_;
            mov(reg_ptr_src_scales__tmp, PARAM(src_scales));
            uni_vbroadcastss(xmm_src_scales_, ptr[reg_ptr_src_scales__tmp]);
        } else if (prb_.src_scale_type == scale_type_t::MANY) {
            mov(reg_ptr_src_scales_, PARAM(src_scales));
        }

        if (prb_.dst_scale_type == scale_type_t::COMMON) {
            auto reg_ptr_dst_scales__tmp = reg_ptr_in_;
            mov(reg_ptr_dst_scales__tmp, PARAM(dst_scales));
            uni_vbroadcastss(xmm_dst_scales_, ptr[reg_ptr_dst_scales__tmp]);
        } else if (prb_.dst_scale_type == scale_type_t::MANY) {
            mov(reg_ptr_dst_scales_, PARAM(dst_scales));
        }

        if (compensation_needed_)
            mov(reg_ptr_comp_, PARAM(compensation_scratch));
        if (prb_.scale_adjust == 0.5f) { mov(reg_scale_adjust_, 0x3f000000); }
        mov(reg_ptr_in_, PARAM(in));
        mov(reg_ptr_out_, PARAM(out));

        bool is_tail_in_drv_dims = false;
        for (int i = prb_.ndims; i < prb_.full_ndims; i++)
            if (prb_.nodes[i].tail_size > 0) {
                is_tail_in_drv_dims = true;
                break;
            }

        if (is_tail_in_drv_dims) {
            Label reorder_kernel;

            mov(reg_tmp_, TAIL_PARAM(skip_kernel_execution));
            cmp(reg_tmp_, static_cast<int64_t>(true));
            je(end_of_kernel, T_NEAR);

            mov(reg_tmp_, TAIL_PARAM(zeroing_data));
            cmp(reg_tmp_, static_cast<int64_t>(false));
            je(reorder_kernel, T_NEAR);
            // If zeroing data is set then all dst memory
            // will be zeroed and nothing more will be done.
            dim_t bytes_to_zeroing = otype_sz_;
            for (int i = 0; i < prb_.ndims; i++) {
                bytes_to_zeroing *= prb_.nodes[i].n;
            }
            xor_(reg_off_out_, reg_off_out_);
            zero_dst_memory(bytes_to_zeroing);
            jmp(end_of_kernel, T_NEAR);
            L(reorder_kernel);
        }

        if (can_do_tr8x8()) {
            vxorps(ymm_zero_, ymm_zero_, ymm_zero_);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp_, 0x7f7f7f7f7f7f7f7f);
                uni_vmovq(Xmm(ymm_8x127b_.getIdx()), reg_tmp_);
            }
        } else {
            uni_vxorps(xmm_zero_, xmm_zero_, xmm_zero_);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
                mov(reg_tmp_.cvt32(), 0x7f7f7f7f);
                movd(xmm_4x127b_, reg_tmp_.cvt32());
            }
        }

        impl();

        L(end_of_kernel);
        postamble();

        const bool is_fp8_itype = utils::one_of(
                prb_.itype, data_type::f8_e5m2, data_type::f8_e4m3);
        const bool is_fp8_otype = utils::one_of(
                prb_.otype, data_type::f8_e5m2, data_type::f8_e4m3);
        if (is_fp8_itype || is_fp8_otype) {
            if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
            if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
        }
    }

    ~jit_uni_reorder_kernel_f32_t() override = default;

#undef TAIL_PARAM
#undef PARAM

private:
    static constexpr int64_t with_tail_info_ = static_cast<int64_t>(true);
    static constexpr int64_t without_tail_info_ = static_cast<int64_t>(false);

    int itype_sz_;
    int otype_sz_;
    int stype_sz_;
    static constexpr int ctype_sz_ = sizeof(int32_t);

    const cpu_isa_t isa_;

    const Reg64 reg_ptr_in_ = rsi;
    const Reg64 reg_ptr_out_ = rdx;
    const Reg64 reg_ptr_src_scales_ = abi_not_param1;
    const Reg64 reg_ptr_dst_scales_ = r12;
    const Reg64 reg_ptr_comp_ = rbx;
    const Reg32 &reg_scale_adjust_ = ebp;

    const Reg64 reg_off_in_ = r8;
    const Reg64 reg_off_out_ = r9;
    const Reg64 reg_off_scale_ = r10;
    const Reg64 reg_off_comp_ = r11;
    // r13-r15 are reserved for creating loops over compute kernels...

    const Reg64 reg_tmp_ = rax;

    const Xmm xmm_src_scales_ = xmm15;
    const Xmm xmm_dst_scales_ = xmm11;
    const Xmm xmm_zero_ = xmm14;
    const Xmm xmm_4x127b_ = xmm13; // TODO: unite with ymm_zero_
    const Ymm ymm_zero_ = ymm14;
    const Ymm ymm_8x127b_ = ymm13;
    const Xmm xmm_tmp_ = xmm12;
    const Xmm xmm_src_zp_ = xmm9;
    const Xmm xmm_dst_zp_ = xmm10;
    const Xmm xmm_compensation = xmm8;
    const Xmm xmm_saturation_ubound_ = xmm12;
    const Ymm ymm_saturation_ubound_ = ymm12;

    const int bf16_emu_zmm_1_idx_ = 16;
    const int bf16_emu_zmm_2_idx_ = 17;
    const int bf16_emu_zmm_3_idx_ = 18;
    const int bf16_emu_zmm_4_idx_ = 19;

    // for fp8 conversion via emulation only.
    const int fp8_emu_zmm_1_idx_ = 16;
    const int fp8_emu_zmm_2_idx_ = 17;
    const int fp8_emu_zmm_3_idx_ = 18;
    const int fp8_emu_zmm_4_idx_ = 19;
    const int fp8_emu_zmm_5_idx_ = 20;

    const int fp8_cvt_kmask_aux_idx_ = 1;
    /* bf16 support on SKX */
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    const Zmm bf16_emu_reserv_1_ = Zmm(bf16_emu_zmm_1_idx_);
    const Zmm bf16_emu_reserv_2_ = Zmm(bf16_emu_zmm_2_idx_);
    const Reg64 bf16_emu_scratch_ = reg_tmp_;
    const Zmm bf16_emu_reserv_3_ = Zmm(bf16_emu_zmm_3_idx_);
    const Zmm bf16_emu_reserv_4_ = Zmm(bf16_emu_zmm_4_idx_);
    /* fp8 support on SPR */
    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;
    /* fp8 emulation only*/
    const Zmm fp8_emu_reserv_1_ = Zmm(fp8_emu_zmm_1_idx_);
    const Zmm fp8_emu_reserv_2_ = Zmm(fp8_emu_zmm_2_idx_);
    const Zmm fp8_emu_reserv_3_ = Zmm(fp8_emu_zmm_3_idx_);
    const Zmm fp8_emu_reserv_4_ = Zmm(fp8_emu_zmm_4_idx_);
    const Zmm fp8_emu_reserv_5_ = Zmm(fp8_emu_zmm_5_idx_);

    const Opmask fp8_cvt_kmask_aux_ = Opmask(fp8_cvt_kmask_aux_idx_);
    const Reg64 fp8_cvt_scratch_ = bf16_emu_scratch_;
};

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims >= 2 && mayiuse(avx2)
                && p.src_scale_type == scale_type_t::NONE
                && p.dst_scale_type == scale_type_t::NONE
                && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
                && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f
                && prb_has_small_strides(p);
        if (!ok) return false;

        int64_t n0 = p.nodes[0].n;
        auto i0 = p.nodes[0].is;
        auto o0 = p.nodes[0].os;
        int64_t n1 = p.nodes[1].n;
        auto i1 = p.nodes[1].is;
        auto o1 = p.nodes[1].os;

        /*
         * for a transpose of plain to 8c case, nodes would be like:
         *     n    is   os
         *     m    1    8
         *     8    m    1
         * or
         *     8    m    1
         *     m    1    8
         */
        ok = (utils::one_of(n0, 8, 16) || utils::one_of(n1, 8, 16))
                && ((i0 == 1 && o1 == 1 && n0 == i1 && o0 == n1)
                        || (o0 == 1 && i1 == 1 && n0 == o1 && i0 == n1));
        if (!ok) return false;

        // Do not handle transpose of dimensions other than last 2
        for (int i = 2; i < p.ndims; ++i) {
            if (p.nodes[i].is != p.nodes[i].os) {
                ok = false;
                break;
            }
        }

        return ok;
    }

    jit_single_blk_kernel_t(const tr::prb_t &prb)
        : jit_generator_t(jit_name())
        , prb_(prb)
        , itype_sz_(static_cast<int>(data_type_size(prb_.itype)))
        , otype_sz_(static_cast<int>(data_type_size(prb_.otype)))
        , block_sz(static_cast<int>(prb.nodes[0].n)) {}

    void generate() override {
        const int input_stride = static_cast<int>(
                prb_.nodes[0].is != 1 ? prb_.nodes[0].is : prb_.nodes[1].is);
        const int output_stride = static_cast<int>(
                prb_.nodes[0].os != 1 ? prb_.nodes[0].os : prb_.nodes[1].os);

        Label tail_processing;

        preamble();

        cmp(reg_ptr_tail, true);
        je(tail_processing, T_NEAR);

        if (block_sz == 8) {
            gen_ker8x8(0, 0, input_stride, output_stride, 8, 8);
            block_sz = 8;
        } else if (block_sz == 16) {
            gen_ker16x16_in_8x8(input_stride, output_stride);
            block_sz = 16;
        } else {
            assert(!"unimplemented");
        }

        postamble();

        L(tail_processing);

        if (block_sz == 8) {
            auto i_tail = input_stride % 8 != 0 ? input_stride % 8 : 8;
            auto o_tail = output_stride % 8 != 0 ? output_stride % 8 : 8;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 8 ? o_tail : i_tail;
                gen_setmask(t_mask);
                gen_ker8x8(0, 0, input_stride, output_stride, i_tail, o_tail);
            }
        } else if (block_sz == 16) {
            auto i_tail = input_stride % 16 != 0 ? input_stride % 16 : 16;
            auto o_tail = output_stride % 16 != 0 ? output_stride % 16 : 16;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 16 ? o_tail : i_tail;
                t_mask %= 8;
                if (t_mask != 0) gen_setmask(t_mask);
                gen_ker16x16_in_8x8(
                        input_stride, output_stride, i_tail, o_tail);
            }
        } else {
            assert(!"unimplemented");
        }

        postamble();
    }

    void gen_loadu(const Ymm ymm, const Address &addr, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(ymm, addr); break;
            case 16: vmovups(xmm, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_storeu(const Address &addr, const Ymm ymm, int size) {
        Xmm xmm(ymm.getIdx());
        switch (size) {
            case 32: vmovups(addr, ymm); break;
            case 16: vmovups(addr, xmm); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskloadu(
            const Ymm ymm, const Address &addr, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(ymm, mask, addr); break;
            case 16: vmaskmovps(xmm, mask128, addr); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskstoreu(
            const Address &addr, const Ymm ymm, const Ymm mask, int size) {
        Xmm xmm(ymm.getIdx());
        Xmm mask128(mask.getIdx());
        switch (size) {
            case 32: vmaskmovps(addr, mask, ymm); break;
            case 16: vmaskmovps(addr, mask128, xmm); break;
            default: assert(!"unreachable");
        }
    }

    // Register allocation xmm0~11
    void gen_transpose_8x8() {
        constexpr int lane = 8;
        for (int i = 0; i < lane / 2; i++) {
            vunpcklps(Ymm(lane + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < lane / 2; i++) {
            int j = i % 2 == 0 ? lane + i : i - 1;
            vshufps(Ymm(lane / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(lane / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < lane / 2; i++)
            vperm2f128(Ymm(i), Ymm(lane / 2 + i), Ymm(lane + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = lane / 2; i < lane; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(lane / 2 + i), uquad);
    }

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask) {
        // all 0, all 1
        vxorps(ymm_tmp, ymm_tmp, ymm_tmp);
        vpcmpeqd(ymm_mask, ymm_mask, ymm_mask);
        // shift by mask to have tail nelems in ymm_mask
        const uint8_t in_mask = static_cast<uint8_t>(0xFF << mask);
        vpblendd(ymm_mask, ymm_mask, ymm_tmp, in_mask);
    }

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        constexpr int lane = 8;

        if (in_tail == 0 || out_tail == 0) return;

        for (int i = 0; i < out_tail; ++i) {
            if (in_tail != lane) {
                gen_maskloadu(Ymm(i),
                        ptr[reg_ptr_in_ + i_off + i * input_stride * itype_sz_],
                        ymm_mask, lane * itype_sz_);
            } else {
                gen_loadu(Ymm(i),
                        ptr[reg_ptr_in_ + i_off + i * input_stride * itype_sz_],
                        lane * itype_sz_);
            }
        }

        gen_transpose_8x8();

        for (int i = 0; i < in_tail; ++i) {
            if (out_tail == lane) {
                gen_storeu(ptr[reg_ptr_out_ + o_off
                                   + i * output_stride * otype_sz_],
                        Ymm(i), lane * otype_sz_);
            } else {
                gen_maskstoreu(ptr[reg_ptr_out_ + o_off
                                       + i * output_stride * otype_sz_],
                        Ymm(i), ymm_mask, lane * otype_sz_);
            }
        }
    }

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
    }

    void gen_ker16x16_in_8x8(int input_stride, int output_stride) {
        const auto lane = 16;
        const auto sub_lane = lane / 2;
        gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(sub_lane * itype_sz_, output_stride * sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8((input_stride * sub_lane + sub_lane) * itype_sz_,
                (output_stride * sub_lane + sub_lane) * otype_sz_, input_stride,
                output_stride, sub_lane, sub_lane);
    }

    // tail can be 1 ~ 16, using avx2 for now
    void gen_ker16x16_in_8x8(
            int input_stride, int output_stride, int in_tail, int out_tail) {
        constexpr auto lane = 16;
        constexpr auto sub_lane = lane / 2;
        auto tail = in_tail != lane ? in_tail : out_tail;

        const auto l_tail = tail < sub_lane ? tail : sub_lane;
        const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

        if (tail == in_tail) {
            gen_tr8x8(0, 0, input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * otype_sz_, input_stride,
                    output_stride, u_tail, sub_lane);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, u_tail, sub_lane);
        } else {
            gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, l_tail);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, sub_lane, u_tail);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * itype_sz_, input_stride,
                    output_stride, sub_lane, l_tail);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, sub_lane, u_tail);
        }
    }

private:
    // 6 ~ 12
    constexpr static int xmm_save_for_windows = is_windows ? 7 : 0;
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble() {
        if (is_windows) {
            sub(rsp, xmm_save_for_windows * xmm_width);
            for (int i = 0; i < xmm_save_for_windows; ++i) {
                uni_vmovdqu(ptr[rsp + i * xmm_width],
                        Xbyak::Xmm(xmm_save_start_from + i));
            }
        }
    }

    void postamble() {
        if (is_windows) {
            for (int i = 0; i < xmm_save_for_windows; ++i)
                uni_vmovdqu(Xbyak::Xmm(xmm_save_start_from + i),
                        ptr[rsp + i * xmm_width]);
            add(rsp, xmm_save_for_windows * xmm_width);
        }
        uni_vzeroupper();
        ret();
    }

    const prb_t &prb_;

    int itype_sz_;
    int otype_sz_;
    int block_sz;

    Reg64 reg_ptr_in_ = abi_param1;
    Reg64 reg_ptr_out_ = abi_param2;
    // Windows bool is 1-byte in register
    Reg8 reg_ptr_tail = is_windows ? r8b : dl;

    Ymm ymm_mask = ymm12;
    Ymm ymm_tmp = ymm0;
};

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

} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    static constexpr int num_elems_thr = 16;
    const bool stride_cache_friendly
            = ((prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > num_elems_thr)
                      || (prb.ndims > 1 && prb.nodes[1].is % num_elems_thr == 0
                              && prb.nodes[1].n > num_elems_thr))
            && !prb.is_tail_present;

    // performance improvement for shapes with large inner-most dimension
    const size_t L1_cache_sz
            = size_t(3) * platform::get_per_core_cache_size(1) / 4;
    const size_t itype_sz_ = data_type_size(prb.itype);
    const size_t inner_block_sz = prb.nodes[0].n * itype_sz_;
    const bool requires_inner_blocking = inner_block_sz > L1_cache_sz
            // 'is_tail_present' is not supported for cache_blocking when
            // asymmetric_comp is executed.
            && IMPLICATION(prb.req_asymmetric_comp, !prb.is_tail_present);

    const bool cache_blocking_needed
            = stride_cache_friendly || requires_inner_blocking;
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed)
            prb_node_split(prb, unit_input_stride_idx, num_elems_thr);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        if (unit_input_stride_idx != move_location)
            prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        if (split_needed) {
            prb_node_split(prb, 0, num_elems_thr);
            prb_node_move(prb, 1, 2);

            // Update node information
            prb_node_dependency(prb);

            // heuristics - looping over the unrolled dims should maximize reuse
            // of the already cached data; observation is choosing the smallest
            // dim from the remaining (from 2 up to ndims) gives good results
            constexpr int new_position = 2;
            const auto dim_beg_it = std::begin(prb.nodes);
            const auto dim_two_it = dim_beg_it + new_position;
            const auto dim_last_it = dim_beg_it + prb.ndims;
            const auto min_n_node_it = std::min_element(dim_two_it, dim_last_it,
                    [](const tr::node_t &lhs, const tr::node_t &rhs) {
                return lhs.n < rhs.n;
            });
            const int min_idx = static_cast<int>(
                    std::distance(dim_beg_it, min_n_node_it));
            // check if min_idx node is parent of node with tail processing which
            // is currently unsupported (i.e. tail processing can only be handled
            // at the inner-most dimension)
            bool inner_block_has_tail = false;
            for (int idx = min_idx - 1; idx >= new_position; idx--) {
                if (prb.nodes[idx].parent_node_id == min_idx) {
                    inner_block_has_tail = true;
                    break;
                }
            }

            if (min_idx > new_position && (!inner_block_has_tail))
                prb_node_move(prb, min_idx, new_position);
        }
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(
        tr::prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t size_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        size_total *= prb.nodes[d].n;

    // The general expression for size_drv_thr can be written as
    // size_drv_min = C0 + FC * (nthr > 1 ? 1 : 0) + VC * (nthr - 1)
    // where FC and VC are fixed and variable costs respectively.
    // Though for now, the below heuristic seems to be good enough
    const size_t size_drv_thr = (nthr > 1) ? 16 * nthr : 1;

    /* size_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t size_drv_min
            = nstl::min<size_t>(size_drv_thr, utils::div_up(size_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * size_ker_cur -- product of the dimension processed by a kernel
     * size_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t size_drv_cur = 1;
    for (; kdims > 1 && size_drv_cur < size_drv_min; --kdims)
        size_drv_cur *= prb.nodes[kdims - 1].n;

    size_t size_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        size_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that size_drv_cur >= size_drv_min.
     *
     * It might happen that for chosen kdims the size_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase size_ker_cur. */
    const bool want_borrow_ker_from_drv = kdims < prb.ndims
            && size_ker_cur < tr::ker_prb_size_min
            && size_drv_cur > size_drv_min;
    if (want_borrow_ker_from_drv) {
        /* size_want_borrow is the minimal size, so that:
         *  o) size_ker_cur * size_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     size_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal size_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        dim_t size_want_borrow
                = utils::div_up<dim_t>(tr::ker_prb_size_min, size_ker_cur);
        for (; prb.nodes[kdims].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, size_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the size_drv_cur is too small (less than size_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * size_drv_cur. */
    const bool want_borrow_drv_from_ker = size_ker_cur > tr::ker_prb_size_min
            && size_drv_cur < size_drv_min;

    VDEBUGINFO(5, primitive, reorder,
            "size_drv_thr=%zu size_drv_min=%zu size_drv_cur=%zu "
            "tr::ker_prb_size_min=%zu want_borrow_ker_from_drv=%d "
            "want_borrow_drv_from_ker=%d",
            size_drv_thr, size_drv_min, size_drv_cur, tr::ker_prb_size_min,
            want_borrow_ker_from_drv, want_borrow_drv_from_ker);

    if (want_borrow_drv_from_ker) {
        dim_t size_want_borrow
                = utils::div_up<dim_t>(size_drv_min, size_drv_cur);
        for (; prb.nodes[kdims - 1].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / size_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            verbose_printf(
                    verbose_t::debuginfo, "split: %s\n", prb_dump(prb).c_str());
            verbose_printf(verbose_t::debuginfo, "ndims_ker_max = %d\n",
                    ndims_ker_max);
        });
    }
}

status_t jit_uni_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md(), *dst_md(), attr());
    if (prb_init_status != status::success) return prb_init_status;

    prb_block_for_cache(prb);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "cache: %s\n", prb_dump(prb).c_str());
    });

    int ndims_ker_max {};
    int nthr = dnnl_get_max_threads();
    prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

    if (prb.is_tail_present) prb_node_dependency(prb);

    tr::kernel_t::desc_t ker_desc;
    status_t ker_init_status
            = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
    if (ker_init_status != status::success) return ker_init_status;

    const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
    VDISPATCH_REORDER(ndims_driver <= jit_uni_reorder_t::ndims_driver_max,
            VERBOSE_BAD_NDIMS, "driver", ndims_driver);

    DEBUG({
        verbose_printf(verbose_t::debuginfo, "ker  : %s\n",
                prb_dump(ker_desc.prb).c_str());
    });

    nthr_ = nthr;
    prb_ = prb;
    with_groups_ = prb.compensation_mask == tr::prb_t::comp_mask_with_groups;

    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    ker_desc_ = ker_desc;

    CHECK(init_scratchpad());

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();

    const bool compensation_needed
            = prb_.req_s8s8_comp || prb_.req_asymmetric_comp;
    if (compensation_needed) {
        const memory_desc_wrapper od(dst_md());
        const auto G = with_groups_ ? od.padded_dims()[0] : 1;
        const auto N = od.padded_dims()[with_groups_ ? 1 : 0];
        static constexpr int cache_line_size = 16;
        const auto wspace_per_thr_size
                = utils::rnd_up(G * N, cache_line_size) * sizeof(int32_t);

        const auto compensation_reduce_size = wspace_per_thr_size * nthr_;

        // Every thread gets its own scratchpad space for each N.
        scratchpad.template book<int32_t>(
                memory_tracking::names::key_reorder_space,
                compensation_reduce_size);
    }

    if (!attr()->scales_.has_default_values(DNNL_ARG_DST)) {
        const memory_desc_wrapper input_d(src_md());
        int mask = attr()->scales_.get_mask(DNNL_ARG_DST);
        get_D_values(input_d, mask, nullptr, &D_mask_, nullptr);
        // Every thread must handle scales inside a parallel task.
        const auto dst_scales_scratch_size = D_mask_ * nthr_;
        scratchpad.template book<float>(
                memory_tracking::names::key_reorder_precomputed_dst_scales,
                dst_scales_scratch_size);
    }

    return status::success;
}

void jit_uni_reorder_t::omp_driver_0d(int off, const char *in, char *out,
        const void *src_scales, const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;

    tr::call_param_t base_params;
    base_params.in = in;
    base_params.out = out;
    if (prb.src_scale_type != tr::scale_type_t::NONE)
        base_params.src_scales = src_scales;
    if (prb.dst_scale_type != tr::scale_type_t::NONE)
        base_params.dst_scales = dst_scales;
    base_params.src_zp = src_zp;
    base_params.dst_zp = dst_zp;
    base_params.compensation_scratch = compensation_scratch;

    if (prb.is_tail_present) {
        tr::tail_call_param_t tail_params;
        tail_params.base_params = base_params;

        static constexpr int omp_ndims = 0;
        fill_curr_data_chunks(prb, off, nullptr, omp_ndims, tail_params);

        (*kernel_)(&tail_params);
    } else {
        (*kernel_)(&base_params);
    }
}

void jit_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, ns[0].n, [&](dim_t d0) {
        tr::call_param_t base_params;
        base_params.in = in + d0 * ns[0].is * data_type_size(prb.itype);
        base_params.out = out + d0 * ns[0].os * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales
                    = static_cast<const float *>(src_scales) + d0 * ns[0].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales
                    = static_cast<const float *>(dst_scales) + d0 * ns[0].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 1;
            const dim_t omp_data_chunks[omp_ndims] = {d0};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, ns[1].n, ns[0].n, [&](dim_t d1, dim_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is) * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os) * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch
                = compensation_scratch + d0 * ns[0].cs + d1 * ns[1].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 2;
            const dim_t omp_data_chunks[omp_ndims] = {d0, d1};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, ns[2].n, ns[1].n, ns[0].n,
            [&](dim_t d2, dim_t d1, dim_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                        * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                        * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                + d1 * ns[1].cs + d2 * ns[2].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 3;
            const dim_t omp_data_chunks[omp_ndims] = {d0, d1, d2};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
        const char *in, char *out, const void *src_scales,
        const void *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, ns[3].n, ns[2].n, ns[1].n, ns[0].n,
            [&](dim_t d3, dim_t d2, dim_t d1, dim_t d0) {
        tr::call_param_t base_params;
        base_params.in = in
                + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                          + d3 * ns[3].is)
                        * data_type_size(prb.itype);
        base_params.out = out
                + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                          + d3 * ns[3].os)
                        * data_type_size(prb.otype);
        if (prb.src_scale_type != tr::scale_type_t::NONE)
            base_params.src_scales = static_cast<const float *>(src_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                    + d3 * ns[3].ss;
        if (prb.dst_scale_type != tr::scale_type_t::NONE)
            base_params.dst_scales = static_cast<const float *>(dst_scales)
                    + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                    + d3 * ns[3].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs
                + d1 * ns[1].cs + d2 * ns[2].cs + d3 * ns[3].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 4;
            const dim_t omp_data_chunks[omp_ndims] = {d0, d1, d2, d3};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::reduce_compensation(char *out,
        const int32_t *compensation_reduce_scratch, const int nthr,
        const dim_t wspace_per_thr_size) const {

    const memory_desc_wrapper od(pd()->dst_md());
    const size_t offset = od.size() - od.additional_buffer_size();

    static constexpr auto comp_dt_size = sizeof(int32_t);
    static constexpr int32_t comp_s8s8_shift = 128;

    // Note: We do not need to explicitly zero-out compensation buffer, as the
    // per_thread buffers are already zeroed out in the padded area.
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    const auto GN = G * N;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const size_t zp_offset
            = offset + (pd()->prb_.req_s8s8_comp ? GN * comp_dt_size : 0);

    parallel_nd(GN, [=](dim_t idx) {
        int32_t acc = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            acc -= compensation_reduce_scratch[ithr * wspace_per_thr_size
                    + idx];
        }
        if (req_s8s8_comp) {
            int32_t *out_comp = reinterpret_cast<int32_t *>(&out[offset]);
            out_comp[idx] = comp_s8s8_shift * acc;
        }
        if (req_asymmetric_comp) {
            int32_t *out_asym_comp
                    = reinterpret_cast<int32_t *>(&out[zp_offset]);
            out_asym_comp[idx] = acc;
        }
    });
}

void jit_uni_reorder_t::fill_curr_data_chunks(const tr::prb_t &prb,
        const int off, const dim_t *omp_data_chunks, const int omp_ndims,
        tr::tail_call_param_t &c) const {
    // Chunks are backwards numered i.e:
    // [0] -> [node_size]
    // [1] -> [node_size - 1]
    // ...
    // [node_size - 1] -> [1]

    // It is done like this, because it is easier to decrement counter
    // and check if it is equal to zero than increment and check
    // if it is equal to node_size in jit kernel.

    static constexpr int64_t empty_chunk_info = -1;
    static constexpr int64_t last_chunk = 1;

    for (int curr_node_id = prb.ndims - 1; curr_node_id >= 0; curr_node_id--) {
        const int parent_node_id = prb.nodes[curr_node_id].parent_node_id;
        const bool is_drv_processing_this_node
                = curr_node_id >= off && curr_node_id <= off + omp_ndims - 1;
        const bool is_tail_processing
                = prb.is_tail_in_one_of_child_nodes(curr_node_id)
                || prb.nodes[curr_node_id].tail_size > 0;

        if (is_drv_processing_this_node && is_tail_processing) {
            const int inner_idx = curr_node_id - off;
            assert(inner_idx < omp_ndims);
            const int64_t node_size = prb.nodes[curr_node_id].tail_size > 0
                    ? prb.nodes[curr_node_id].tail_size
                    : prb.nodes[curr_node_id].n;
            const int64_t data_chunk = node_size - omp_data_chunks[inner_idx];

            if (!prb.nodes[curr_node_id].is_parent_empty()) {
                const bool is_parent_chunk_last
                        = c.curr_data_chunks[parent_node_id] == last_chunk;
                c.curr_data_chunks[curr_node_id]
                        = is_parent_chunk_last ? data_chunk : empty_chunk_info;
                c.zeroing_data = static_cast<int64_t>(
                        is_parent_chunk_last && data_chunk <= 0);
            } else {
                c.curr_data_chunks[curr_node_id] = data_chunk;
                c.zeroing_data = static_cast<int64_t>(data_chunk <= 0);
            }
            c.skip_kernel_execution = static_cast<int64_t>(c.zeroing_data
                    && !prb.nodes[curr_node_id].is_zero_pad_needed);
            if (c.zeroing_data || c.skip_kernel_execution) break;
        } else
            c.curr_data_chunks[curr_node_id] = empty_chunk_info;
    }
}

status_t jit_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
    out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

    DEBUG({
        verbose_printf(verbose_t::debuginfo, "prb  : %s\n",
                tr::prb_dump(pd()->prb_).c_str());
    });
    DEBUG({
        verbose_printf(verbose_t::debuginfo, "ker  : %s\n",
                tr::prb_dump(pd()->ker_desc_.prb).c_str());
    });

    int ndims = pd()->prb_.ndims;
    int ndims_ker = pd()->ker_desc_.prb.ndims;
    int ndims_level = ndims - ndims_ker;

    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims_level <= ndims_driver_max);

    int32_t *compensation_reduce_scratch = scratchpad.template get<int32_t>(
            memory_tracking::names::key_reorder_space);

    const memory_desc_wrapper od(pd()->dst_md());
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size = utils::rnd_up(G * N, cache_line_size);
    const auto wspace_per_thr_bytes = wspace_per_thr_size * sizeof(int32_t);

    const int nthr_par = ndims_level == 0 ? 1 : pd()->nthr_;
    parallel(nthr_par, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        int32_t *compensation_scratch = nullptr;
        if (req_compensation) {
            if (ndims_level == 0)
                compensation_scratch = compensation_reduce_scratch;
            else
                compensation_scratch = &compensation_reduce_scratch[ithr
                        * wspace_per_thr_size];
            std::memset(compensation_scratch, 0, wspace_per_thr_bytes);
        }

        float *dst_scales_inv_ptr = nullptr;
        if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_DST)) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            const auto dst_scales_scratch_size_ithr = pd()->D_mask_;
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(memory_tracking::names::
                                      key_reorder_precomputed_dst_scales)
                    + ithr * dst_scales_scratch_size_ithr;
            for (int i = 0; i < dst_scales_scratch_size_ithr; i++) {
                dst_scales_inv_ptr[i] = 1.f / dst_scales_ptr[i];
            }
        }

        auto src_zp = src_zero_points ? src_zero_points[0] : 0;
        auto dst_zp = dst_zero_points ? dst_zero_points[0] : 0;

        switch (ndims_level) {
            case 0:
                omp_driver_0d(ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 1:
                omp_driver_1d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 2:
                omp_driver_2d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 3:
                omp_driver_3d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            case 4:
                omp_driver_4d(ithr, nthr, ndims_ker, in, out, src_scales,
                        dst_scales_inv_ptr, src_zp, dst_zp,
                        compensation_scratch);
                break;
            default: assert(!"unimplemented");
        }
    });

    //reduction of intermediate compensation results to the final output
    if (req_compensation) {
        reduce_compensation(out, compensation_reduce_scratch, nthr_par,
                wspace_per_thr_size);
    }

    return status::success;
}

status_t jit_blk_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md(), *dst_md(), attr());
    if (prb_init_status != status::success) return prb_init_status;
    // only uni_reorder supports tail processing now
    // TODO: Add tail processing support in blk_reorder
    VDISPATCH_REORDER(!prb.is_tail_present, "tail processing is not supported");

    prb_tile_normalize(prb);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "tile : %s\n", prb_dump(prb).c_str());
    });

    VDISPATCH_REORDER(tr::jit_single_blk_kernel_t::applicable(prb),
            "applicable() call failed");

    prb_ = prb;
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    return status::success;
}

void jit_blk_reorder_t::pd_t::prb_tile_normalize(tr::prb_t &p) {
    if (!utils::one_of(p.nodes[0].n, 8, 16)
            && utils::one_of(p.nodes[1].n, 8, 16)) {
        nstl::swap(p.nodes[0], p.nodes[1]);
    }
}

jit_blk_reorder_t::jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}
jit_blk_reorder_t::~jit_blk_reorder_t() = default;

status_t jit_blk_reorder_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
    return kernel_->create_kernel();
}

status_t jit_blk_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

    // kernel handle 2-dimension tiles, a tail is possible
    auto &prb = this->pd()->prb_;
    dim_t BH = 1;
    for (int i = 2; i < prb.ndims; ++i) {
        BH *= prb.nodes[i].n;
    }

    auto block_sz = prb.n(0);
    auto n1 = prb.n(1);
    auto i1 = prb.is(1);
    auto o1 = prb.os(1);
    auto FL = (n1 + block_sz - 1) / block_sz;
    auto bh_stride = BH == 1 ? 0 : prb.is(2);

    auto itype_sz_ = data_type_size(pd()->prb_.itype);
    auto otype_sz_ = data_type_size(pd()->prb_.otype);

    parallel_nd(BH, FL, [= COMPAT_THIS_CAPTURE](dim_t bh, dim_t fl) {
        auto fl_b = fl * block_sz;
        auto bh_b = bh_stride * bh;
        auto *i = in + (bh_b + fl_b * i1) * itype_sz_;
        auto *o = out + (bh_b + fl_b * o1) * otype_sz_;
        (*kernel_)(i, o, n1 - fl_b < block_sz);
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
