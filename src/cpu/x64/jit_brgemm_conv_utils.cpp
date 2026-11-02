/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/scale_utils.hpp"
#include "cpu/x64/brgemm/brgemm_utils.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

#include <set>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace {
bool allow_perf_heuristics(const jit_brgemm_conv_conf_t &jcp) {
    // Disable performance heuristics for plain weights as there are no other
    // optimized implementations.
    if (jcp.wei_plain) return false;
    // Disable performance heuristics for f16, fp8, f32 with xf16 weights
    // as there are no other optimized implementations.
    if (jcp.wei_dt == f16) return false;
    if (one_of(jcp.wei_dt, f8_e5m2, f8_e4m3)) return false;
    if (one_of(true, jcp.is_f32_f16, jcp.is_f32_bf16)) return false;
    return true;
}
} // namespace

namespace brgemm_convolution_utils {

struct brg_blocking_t : public jit_brgemm_conv_conf_t {
    struct array_in_loop_t {
        dim_t itersize;
        float repeatn;
        float overlap;
        template <typename T, typename U = float>
        void set(dim_t iter_s, T rpt, U ovlp = 1.f) {
            itersize = iter_s;
            repeatn = static_cast<float>(rpt);
            overlap = static_cast<float>(ovlp);
        }
    };

    struct loop_t {
        array_in_loop_t src;
        array_in_loop_t wei;
        array_in_loop_t dst;
    };

    brg_blocking_t() {
        // TODO: This is a broken form of initialization for a base class.
        // Either set default values in a base class, or provide a proper
        // default ctor, or take a `jit_brgemm_conv_conf_t` object to initialize
        // a base class object.
        jit_brgemm_conv_conf_t *base
                = static_cast<jit_brgemm_conv_conf_t *>(this);
        *base = jit_brgemm_conv_conf_t();
        init();
    }
    brg_blocking_t(const jit_brgemm_conv_conf_t &jcp)
        : jit_brgemm_conv_conf_t(jcp) {
        init();
    }
    void init() {
        ur = 0;
        ur_block = 0;
        ur_block_tail = 0;
        eff = 0.f;
        nb_kd = 0;
        nb_kh = 0;
        nb_kw = 0;
        sp = 0;
        sp_block = 0;
        nb_sp = 0;
        eff = 0;
        // TODO: remove workaround once constructor is fixed
        max_regs = isa == isa_undef ? 0 : isa_num_vregs(isa);
    }

    dim_t ur, ur_block, ur_block_tail, adj_ocblock;
    dim_t nb_kd, nb_kh, nb_kw;
    int max_regs;
    float eff;
    static unsigned L1;
    static unsigned L2;
    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // TODO: Improve memory access estimates
    static float L1_k;
    static float L2_k;
    static float L3_k;
    static float mem_k;
    static constexpr int bench_iterations = 1;

    dim_t sp, sp_block, nb_sp;

    void get_from_jcp(const jit_brgemm_conv_conf_t &jcp) { *this = jcp; }
    void save_to_jcp(jit_brgemm_conv_conf_t &jcp) const { jcp = *this; }

    status_t estimate_brgemm_ur();
    status_t get_brgemm_ur(
            const primitive_attr_t *attr, const memory_desc_t &dst_md);

    float io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
            bool is_broadcast, bool is_shared) const;

    float io_k(const loop_t loop, const array_in_loop_t arr, float pk,
            bool is_broadcast, bool is_shared) const;

    void select_ic_block();

    void update_blocks();
    bool fast_check_oc_block() const;
    float est_eff();
    void iterate_ker_block(brg_blocking_t &best_brgb, dim_t kd_block,
            dim_t kh_block, bool maybe_use_buffer, dim_t max_ow_block_thr);
    status_t calc_blocks();

    bool fast_check_oc_block_1x1() const;
    float est_eff_1x1();
    void calc_blocks_1x1();

    // utils
    static dim_t get_inp_size(dim_t max_src_size, dim_t dst_size, dim_t k,
            dim_t stride, dim_t dilate) {
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, stride,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    static float squeeze_val(float eff, float koeff) {
        if (koeff <= 0) return 1;
        if (koeff == 1) return eff;
        const auto k = 1.f / koeff;
        return (k > 1.f) ? (k - 1 + eff) / k : eff * koeff;
    }

    static int estimate_ur(cpu_isa_t isa, dim_t oc_block) {
        if (one_of(isa, avx2, avx2_vnni, avx2_vnni_2)) {
            switch (oc_block) {
                case 32: return 3;
                case 24: return 4;
                case 16: return 6;
                default: return 14;
            }
        } else {
            switch (oc_block) {
                case 64: return 6;
                case 48: return 9;
                case 32: return 14;
                default: return 28;
            }
        }
    }

    dim_t inp_w(dim_t out_w, dim_t ker_w) const {
        return get_inp_size(iw, out_w, ker_w, stride_w, dilate_w);
    }

    dim_t rnd_simd(dim_t val) const { return rnd_up(val, simd_w); }

    dim_t rnd_inp_simd(dim_t out_w, dim_t ker_w, dim_t vic) const {
        const auto vsp = inp_w(out_w, ker_w);
        return ((stride_w == 1 && vic >= ic) ? rnd_up(vsp * vic, simd_w)
                                             : vsp * rnd_up(vic, simd_w));
    }
    dim_t grid_coverage(
            dim_t nb, dim_t x, dim_t nx, dim_t xb, dim_t y, dim_t yb);
    static constexpr int MAXNLOOPS = 32;
    loop_t loop[MAXNLOOPS];
};

dim_t get_comp_ow_size(const jit_brgemm_conv_conf_t &jcp) {
    if (jcp.exec_type == exec_vpad) return 1;

    const auto IW = jcp.iw;
    const auto DW = jcp.dilate_w + 1;
    const auto KW = jcp.kw;
    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    dim_t comp_ow_size = 0;

    for (int ow = 0; ow < jcp.ow;) {
        const auto iiw = ow * SW - LP;
        const auto kw_s = div_up(nstl::max<dim_t>(0, -iiw), DW);
        const auto kw_f = KW
                - div_up(nstl::max<dim_t>(0, iiw - IW + (KW - 1) * DW + 1), DW);
        int ow_e = ow;
        while (ow_e < jcp.ow) {
            const auto iiw_e = ow_e * SW - LP;
            const auto cur_kw_s = div_up(nstl::max<dim_t>(0, -iiw_e), DW);
            const auto cur_kw_f = KW
                    - div_up(
                            nstl::max<dim_t>(0, iiw_e - IW + (KW - 1) * DW + 1),
                            DW);
            if (cur_kw_s != kw_s || cur_kw_f != kw_f) break;
            if (ow_e - ow < jcp.ow_block) comp_ow_size++;
            ow_e++;
        }
        ow = ow_e;
    }

    return comp_ow_size;
}

bool is_any_eligible(const jit_brgemm_conv_conf_t &jcp) {
    return (jcp.prop_kind == prop_kind::forward_inference || jcp.wei_plain
            || one_of(jcp.wei_dt, data_type::s8, data_type::f16,
                    data_type::f8_e5m2, data_type::f8_e4m3)
            || one_of(jcp.isa, avx2_vnni_2) || is_amx(jcp.isa));
}

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value,
        bool any_eligible) {

    if (mdw.format_kind() == format_kind::any) {
        if (any_eligible) {
            CHECK(memory_desc_init_by_tag(md, tag_value));
            tag = tag_value;
        } else {
            tag = format_tag::undef;
        }
    } else {
        tag = mdw.mb_stride_relaxed_match(tag_value);
    }

    VDISPATCH_CONV_IC(tag == tag_value, VERBOSE_UNSUPPORTED_TAG);

    return status::success;
}

bool is_amx(cpu_isa_t isa) {
    // Tile accumulators drive blocking and kernel choice; ACE uses them too.
    return isa_has_tile_accumulators(isa);
}

bool uses_batch_elements(
        brgemm_batch_kind_t brg_type, conv_brgemm_exec_type_t exec_type) {
    // Batch elements are required except for fixed-stride batches and virtual padding.
    return IMPLICATION(brg_type == brgemm_strd, exec_type == exec_vpad);
}

bool post_ops_ok(jit_brgemm_conv_conf_t &jcp, primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(jcp.isa,
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            false /*sum_requires_zp_zero*/, true /*sum_requires_same_params*/,
            {broadcasting_strategy_t::per_oc, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::no_broadcast,
                    broadcasting_strategy_t::spatial}));
}

bool is_groups_ok(jit_brgemm_conv_conf_t &jcp) {
    // Grouped convs are allowed where direct convs reject them.
    // This is limited to int8/bf16 groups with IC/OC multiples of 4.
    // TODO: drop this once the small-IC restriction is removed.
    return jcp.ngroups > 1
            && IMPLICATION(one_of(jcp.src_dt, u8, s8, bf16),
                    jcp.ic % 4 == 0 && jcp.oc % 4 == 0);
}

status_t pick_tags(jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md) {
    format_tag_t src_tag, dst_tag, wei_tag;
    dst_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const bool is_1d = jcp.ndims == 3;
    const bool is_2d = jcp.ndims == 4;
    const bool is_3d = jcp.ndims == 5;

#define BRGEMM_WEITAG(_OC_, _DIMS_, _IC_, _RDP_, _OCB_, _VNNIB_) \
    { \
        wei_tag = with_groups ? g##_OC_##_DIMS_##_IC_##_RDP_##_OCB_##_VNNIB_ \
                              : _OC_##_DIMS_##_IC_##_RDP_##_OCB_##_VNNIB_; \
    }

#define BRGEMM_WEITAG_OC_RDP_OCB(_OC_, _RDP_, _OCB_) \
    { \
        if (is_3d) { \
            switch (jcp.vnni_block) { \
                case 1: BRGEMM_WEITAG(_OC_, dhw, i, , _OCB_, ) break; \
                case 2: BRGEMM_WEITAG(_OC_, dhw, I, _RDP_, _OCB_, 2i) break; \
                case 4: BRGEMM_WEITAG(_OC_, dhw, I, _RDP_, _OCB_, 4i) break; \
                default: return status::unimplemented; \
            } \
        } else if (is_1d) { \
            switch (jcp.vnni_block) { \
                case 1: BRGEMM_WEITAG(_OC_, w, i, , _OCB_, ) break; \
                case 2: BRGEMM_WEITAG(_OC_, w, I, _RDP_, _OCB_, 2i) break; \
                case 4: BRGEMM_WEITAG(_OC_, w, I, _RDP_, _OCB_, 4i) break; \
                default: return status::unimplemented; \
            } \
        } else { \
            assert(is_2d); \
            UNUSED(is_2d); \
            switch (jcp.vnni_block) { \
                case 1: BRGEMM_WEITAG(_OC_, hw, i, , _OCB_, ) break; \
                case 2: BRGEMM_WEITAG(_OC_, hw, I, _RDP_, _OCB_, 2i) break; \
                case 4: BRGEMM_WEITAG(_OC_, hw, I, _RDP_, _OCB_, 4i) break; \
                default: return status::unimplemented; \
            } \
        } \
    }

#define BRGEMM_WEITAG_OCB(_OCB_) \
    { \
        if (jcp.is_rd_padded_to_block) \
            BRGEMM_WEITAG_OC_RDP_OCB(O, 16i, _OCB_) \
        else \
            BRGEMM_WEITAG_OC_RDP_OCB(O, , _OCB_) \
    }

    if (jcp.wei_plain) {
        jcp.LDB = jcp.oc_without_padding;
        // Note: non-f32 datatypes are currently unsupported
        assert(jcp.vnni_block == 1);
        if (is_3d) {
            wei_tag = with_groups ? dhwigo : dhwio;
        } else if (is_1d) {
            wei_tag = with_groups ? wigo : wio;
        } else if (is_2d) {
            wei_tag = with_groups ? hwigo : hwio;
        } else {
            return status::unimplemented;
        }
    } else {
        if (jcp.is_relo() && jcp.relo_conv_weights) {
            if (jcp.is_relo_whi()) {
                if (is_1d)
                    BRGEMM_WEITAG(O, w, i, , 16o, )
                else if (is_2d)
                    BRGEMM_WEITAG(O, wh, i, , 16o, )
                else {
                    assert(!"3d not supported by relo whi");
                    return status::unimplemented;
                }
            } else if (jcp.is_relo_wi()) {
                if (is_1d)
                    BRGEMM_WEITAG(O, w, i, , 16o, )
                else if (is_2d)
                    BRGEMM_WEITAG(O, hw, i, , 16o, )
                else
                    BRGEMM_WEITAG(O, dhw, i, , 16o, )
            }
        } else {
            jcp.LDB = jcp.oc_block;
            switch (jcp.oc_block) {
                case 64: BRGEMM_WEITAG_OCB(64o) break;
                case 48: BRGEMM_WEITAG_OCB(48o) break;
                case 32: BRGEMM_WEITAG_OCB(32o) break;
                case 24: BRGEMM_WEITAG_OC_RDP_OCB(O, , 24o) break;
                case 16: BRGEMM_WEITAG_OCB(16o) break;
                case 8: BRGEMM_WEITAG_OC_RDP_OCB(O, , 8o) break;
                default: return status::unimplemented;
            }
        }
    }

#undef BRGEMM_WEITAG_OCB
#undef BRGEMM_WEITAG_OC_RDP_OCB
#undef BRGEMM_WEITAG

    src_tag = dst_tag;

    const bool any_eligible = is_any_eligible(jcp);
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));
    CHECK(init_tag(jcp.dst_tag, dst_md, dst_d, dst_tag, any_eligible));
    CHECK(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag, true));

    return status::success;
}

unsigned brg_blocking_t::L1;
unsigned brg_blocking_t::L2;
float brg_blocking_t::L1_k;
float brg_blocking_t::L2_k;
float brg_blocking_t::L3_k;
float brg_blocking_t::mem_k;

float brg_blocking_t::io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
        bool is_broadcast, bool is_shared) const {
    if (n < 1) return 0;
    if (n == 1) return pk;
    const auto amount = src * src_dsz + wei * wei_dsz + dst * dst_dsz
            + (use_buffer ? dst * acc_dsz : 0);
    const auto amount_L1 = is_broadcast ? src * src_dsz : amount;
    const auto k = is_broadcast
            ? ((amount_L1 < L1) ? L1_k
                                : ((amount < L2) ? L2_k
                                                 : (is_shared ? L3_k : mem_k)))
            : ((amount < L2) ? L2_k : (is_shared ? L3_k : mem_k));
    const auto cost = pk + k * (n - 1);
    return cost / n;
}

float brg_blocking_t::io_k(const loop_t loop, const array_in_loop_t arr,
        float pk, bool is_broadcast, bool is_shared) const {
    return io_k(loop.src.itersize, loop.wei.itersize, loop.dst.itersize,
            arr.repeatn * arr.overlap, pk, is_broadcast, is_shared);
}

void brg_blocking_t::select_ic_block() {
    if (is_1x1 && is_amx(isa)) {
        // TODO: merge with non-1x1 code block below
        const int ic_padded_block = 16 * vnni_block;

        MAYBE_UNUSED(ic_padded_block);
        // Note: bf32 requires ic_block be less than 64, otherwise it results
        // in incorrect output.
        ic_block = is_bf32 && (!is_rtus) ? nstl::min<dim_t>(64, ic) : ic;
        nb_ic = utils::div_up(ic, ic_block); // trivially 1 for now
        inp_ic_block = ic_block;
        return;
    }
    auto nb_simd = utils::div_up(ic, simd_w);
    auto max_simd_blocks = nstl::min<dim_t>(5 * simd_w, nb_simd);
    const auto nb_icb_eff_threshold = 0.5f;
    const auto padded_rd
            = vnni_block * (is_rd_padded_to_block ? acc_simd_w : 1);
    if (is_amx(isa)) {
        const auto kw_koef = is_relo() ? kw : 1;
        const auto kh_koef = is_relo_whi() ? kh : 1;
        if (kd * kh * ic * src_dsz > 8 * 1024) {
            // For huge ic try to split it by equal ic_blocks
            const auto max_ic_block
                    = rnd_up(div_up(1024, kd * kh * src_dsz), vnni_block);
            const auto min_ic_block = rnd_up(simd_w / 2, vnni_block);
            ic_block = ic;
            for (int iic_block = max_ic_block; iic_block >= min_ic_block;
                    iic_block -= vnni_block) {
                if (ic % iic_block == 0) {
                    ic_block = iic_block;
                    break;
                }
            }
        } else if (ic * kw_koef * kh_koef <= simd_w) {
            // this is current requirement from brgemm kernel
            ic_block = rnd_up(ic, vnni_block);
        } else if (is_bf32) {
            ic_block = simd_w;
        } else {
            if (exec_type == exec_trans) {
                // TODO: double check calculation of ic_block here:
                // for example for ic == 48
                dim_t simd_blocks = 1;
                for (dim_t nb_icb = max_simd_blocks; nb_icb >= 1; nb_icb--) {
                    auto nb_icb_eff = static_cast<float>(nb_simd)
                            / static_cast<float>(rnd_up(nb_simd, nb_icb));
                    if (nb_icb_eff >= nb_icb_eff_threshold) {
                        simd_blocks = nb_icb;
                        break;
                    }
                }
                ic_block = simd_blocks * simd_w;
            } else
                ic_block = simd_w;
        }
    } else {
        const auto est_ur = sp_block > 0
                ? nstl::min<dim_t>(sp_block, estimate_ur(isa, oc_block))
                : estimate_ur(isa, oc_block);
        const auto inp_ur = is_os_blocking ? est_ur : inp_w(est_ur, kw_block);

        if (kw_block > 1) {
            // try to fit src into L1
            const auto inp_per_ic = static_cast<unsigned int>(inp_ur) * src_dsz;
            max_simd_blocks = utils::saturate<dim_t>(
                    1, max_simd_blocks, L1 / (inp_per_ic * simd_w));
        }
        // try to fit all batch for ur into L2
        const bool adjust = wei_plain && math::is_pow2(oc)
                && utils::everyone_is(1, kd_block, kh_block, kw_block);
        const dim_t adj_oc_block = adjust ? oc : oc_block; // due to aliasing
        const auto wei_per_ic = static_cast<unsigned int>(kd_block) * kh_block
                * kw_block * adj_oc_block * wei_dsz;
        const auto inp_per_ic = static_cast<unsigned int>(kd_block) * kh_block
                * inp_ur * src_dsz;
        const auto out_size
                = static_cast<unsigned int>(ur) * oc_block * dst_dsz;

        max_simd_blocks = utils::saturate<dim_t>(1, max_simd_blocks,
                (L2 - out_size) / ((wei_per_ic + inp_per_ic) * simd_w));

        // use nb_simd as max_simd_blocks for some shapes on avx2
        // TODO: optimize max_simd_blocks
        if (isa == avx2 && nb_simd <= 256 && (wei_per_ic + inp_per_ic < 1400))
            max_simd_blocks = nb_simd;

        auto simd_blocks = 1;
        for (int nb_icb
                = static_cast<int>(nstl::min<dim_t>(max_simd_blocks, nb_simd));
                nb_icb >= 1; nb_icb--) {
            auto nb_icb_eff = static_cast<float>(nb_simd)
                    / static_cast<float>(rnd_up(nb_simd, nb_icb));
            if (nb_icb_eff >= nb_icb_eff_threshold) {
                simd_blocks = nb_icb;
                break;
            }
        }

        ic_block = nstl::min<dim_t>(
                exec_type == exec_trans ? rnd_up(ic, padded_rd) : ic,
                simd_blocks * simd_w);
    }

    if (is_relo()) {
        inp_ic_block = ic;
        if (ic_block < inp_ic_block) ic_block = inp_ic_block;
    } else
        inp_ic_block = ic_block;

    nb_ic = utils::div_up(ic, ic_block);
}

status_t brg_blocking_t::estimate_brgemm_ur() {
    // Simple simulation of brgemm_desc init
    if (sp_block <= 0) return status::invalid_arguments;
    const auto kh_koef = is_relo_whi() ? kh : 1;
    if (is_rtus)
        LDA = is_reduced_rtus ? ic_without_padding : inp_ic_block;
    else {
        const size_t ic_stride = exec_type == exec_trans
                ? inp_ic_block
                : ngroups * ic_without_padding;
        LDA = kh_koef * stride_w * ic_stride;
    }
    bool reduce_kw = ow == 1 && !is_reduced_rtus;
    if (reduce_kw) { LDA *= ext_kw; }

    LDB = wei_plain ? oc_without_padding : oc_block;
    LDC = use_buffer ? oc_block : oc_without_padding;

    // Configure matrix sizes
    // for amx if ic_block != ic then we use exec_trans so K is ic_block
    const auto padded_rd
            = vnni_block * (is_rd_padded_to_block ? acc_simd_w : 1);

    icp = rnd_up(ic, padded_rd);
    M = brgM = sp >= sp_block ? sp_block : 0;
    M_tail = brgM_tail = sp % sp_block;
    if (is_os_blocking) {
        if (!is_1x1) M_tail = (oh * ow) % sp_block;
        const auto adj_sh = is_relo_whi() ? 1 : stride_h;
        oskip = reduce_kw ? 0 : ((ext_kw - 1) / stride_w) * adj_sh;
        oskip += (adj_sh - 1) * ow;

        brgM = M + oskip * (div_up(M, ow) - 1);
        brgM_tail = M_tail + oskip * div_up(M_tail, ow);

        // round up brgM and brgM_tail to help brgemm kernels use max amx_h as
        // bd_block and to avoid bd blocks with all zeros in the mask
        if (use_M_mask == 2) {
            int ibrgM = 0;
            const auto adj_ow = ow_block + oskip;
            while (ibrgM < brgM) {
                if (ibrgM % adj_ow < ow_block)
                    ibrgM += amx_h;
                else
                    ibrgM++;
            }
            brgM = ibrgM;
            const auto start_M_tail_in_ow = rnd_dn(oh * ow, sp_block) % ow;
            ibrgM = 0;
            while (ibrgM < brgM_tail) {
                if ((ibrgM + start_M_tail_in_ow) % adj_ow < ow_block)
                    ibrgM += amx_h;
                else
                    ibrgM++;
            }
            brgM_tail = ibrgM;
        } else {
            brgM = rnd_up(brgM, amx_h);
            brgM_tail = rnd_up(brgM_tail, amx_h);
        }
    }

    N = oc >= oc_block ? oc_block : 0;
    N_tail = oc % oc_block;
    if (is_relo()) {
        K = (ic >= ic_block) ? rnd_up(kh_koef * kw * inp_ic_block, vnni_block)
                             : 0;
        if (vnni_block > 1 && K > simd_w) K = rnd_up(K, simd_w);

        K_tail = rnd_up(kh_koef * kw
                        * (!is_bf32 ? inp_ic_block
                                    : rnd_up(ic % ic_block, vnni_block)),
                vnni_block);
        if (vnni_block > 1 && K_tail > simd_w) K_tail = rnd_up(K_tail, simd_w);
    } else {
        K = kh_koef * (ic >= ic_block ? ic_block : 0);
        const auto ic_ceil
                = exec_type == exec_trans && ic_block % simd_w == 0 && !is_bf32
                ? simd_w
                : vnni_block;
        K_tail = kh_koef
                * (exec_type == exec_trans ? rnd_up(ic % ic_block, ic_ceil)
                                           : (ic % ic_block));
    }

    const auto vK = K > 0 ? K : K_tail;
    const auto vM = M > 0 ? M : M_tail;
    const auto vN = N > 0 ? N : N_tail;

    const float alpha = 1.0;
    const float beta = 0.0;
    brgemm_desc_t brg;
    CHECK(brgemm_utils::init_brgemm_conf(&brg, isa, brgemm_addr, src_dt, wei_dt,
            brgemm_row_major, alpha, beta, LDA, LDB, LDC, vM, vN, vK, nullptr,
            is_bf32));
    if (exec_type == exec_vpad) {
        brg.zp_type_a = src_zero_point ? brgemm_broadcast_t::per_tensor
                                       : brgemm_broadcast_t::none;
    }
    brgemm_attr_t brgattr;
    brgattr.use_uker = use_uker;
    brgattr.max_bs = max_batch;
    max_vpad = exec_type == exec_vpad
            ? static_cast<int>(nstl::max(l_pad, r_pad))
            : 0;
    brgattr.max_top_vpad = max_vpad;
    brgattr.max_bottom_vpad = max_vpad;
    brgattr.use_ace = is_ace;
    brgattr.use_uker = use_uker;
    CHECK(brgemm_desc_set_attr(&brg, brgattr));
    CHECK(brgemm_utils::brgemm_blocking(&brg));
    // AMX kernel may fall back to VMM, in which case bd_block2 == 0
    ur = brg.bd_block * nstl::max<dim_t>(1, brg.bd_block2);
    ur_block = brg.bd_block;
    adj_ocblock = nstl::max<dim_t>(
            1, (brg.ldb2 != 0 ? brg.ld_block2 : brg.ldb2_tail));
    if (((is_1x1 && is_amx(isa)) || max_vpad > 0) && M > 0 && M_tail > 0) {
        brgemm_desc_t brg_sp_tail;

        CHECK(brgemm_utils::init_brgemm_conf(&brg_sp_tail, isa, brgemm_addr,
                src_dt, wei_dt, brgemm_row_major, alpha, beta, LDA, LDB, LDC,
                M_tail, vN, vK, nullptr, is_bf32));
        if (exec_type == exec_vpad) {
            brg_sp_tail.zp_type_a = src_zero_point
                    ? brgemm_broadcast_t::per_tensor
                    : brgemm_broadcast_t::none;
        }
        brgattr.use_ace = is_ace;
        brgattr.use_uker = use_uker;
        CHECK(brgemm_desc_set_attr(&brg_sp_tail, brgattr));
        CHECK(brgemm_utils::brgemm_blocking(&brg_sp_tail));
        ur_block_tail = brg_sp_tail.bd_block;
    } else {
        ur_block_tail = 0;
    }
    return status::success;
}

status_t brg_blocking_t::get_brgemm_ur(
        const primitive_attr_t *attr, const memory_desc_t &dst_md) {
    // Detailed simulation of brgemm convolution init
    if (sp_block <= 0 || ic_block <= 0 || oc_block <= 0)
        return status::invalid_arguments;
    CHECK(estimate_brgemm_ur());

    LDD = oc_without_padding;

    // The code below will verify that for selected `brgemm_ur` all brgemm
    // descriptors will be initialized successfully. If they do, stick to this
    // `brgemm_ur`, otherwise, a new estimation will come.
    //
    // This set will filter only needed M for validating brgemm descriptors.
    //
    // Note: the same approach for `jcp_.exec_type == exec_base` to filter out
    // M and extra brgemm descriptors is used in convolution implementation in
    // `pd_t::init`. Maybe it's possible to align them, or make them relying on
    // the same set of unique M values.
    std::set<dim_t> unique_vM;
    if (utils::one_of(exec_type, exec_trans, exec_vpad) || is_1x1) {
        // Up to two M values for these exec_types.
        unique_vM.emplace(M);
        if (M_tail) unique_vM.emplace(M_tail);
    } else {
        // For `exec_base` exec_type need to compute all relevant M values.
        for (int ow = 0; ow < this->ow;
                ow += static_cast<int>(this->ow_block)) {
            int kw_s = 0;
            int kw_f = 0;
            int dummy; // `kw_full_s` and `kw_full_f` are not needed here.
            brgemm_convolution_utils::get_kw_range(
                    /* jcp */ *this, ow, kw_s, dummy, dummy, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                int ow_s = 0;
                int ow_f = 0;
                brgemm_convolution_utils::get_ow_range(
                        /* jcp */ *this, ow, kw, ow_s, ow_f);
                const auto M = ow_f - ow_s;
                if (M <= 0) continue;
                unique_vM.emplace(M);
            }
        }
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float beta_init = 0.0f;

    for_(auto vM : unique_vM)
    for_(int i_init = 0; i_init < 2; i_init++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vN = (i_N) ? N_tail : N;
        auto vK = (i_K) ? K_tail : K;
        if (vN == 0 || vK == 0) continue;
        brgemm_desc_t brg;
        brgemm_strides_t brg_strides;
        brg_strides.stride_a
                = ngroups * ic_without_padding * (dilate_w + 1) * src_dsz;
        // weights are padded by oc_block and last_ic_block
        brg_strides.stride_b
                = rnd_up(ic, vnni_block) * rnd_up(oc, oc_block) * wei_dsz;
        const auto strides_ptr
                = (brg_type == brgemm_strd) ? &brg_strides : nullptr;
        CHECK(brgemm_utils::init_brgemm_conf(&brg, isa, brg_type, src_dt,
                wei_dt, brgemm_row_major, alpha, vbeta, LDA, LDB, LDC, vM, vN,
                vK, strides_ptr, is_bf32));

        brgemm_attr_t brgattr;
        brgattr.use_uker = use_uker;
        brgattr.max_bs = max_batch;
        max_vpad = exec_type == exec_vpad
                ? static_cast<int>(nstl::max(l_pad, r_pad))
                : 0;
        brgattr.max_top_vpad = max_vpad;
        brgattr.max_bottom_vpad = max_vpad;
        brgattr.fpmath_mode = attr->fpmath_.mode_;
        brgattr.use_ace = is_ace;
        brgattr.use_uker = use_uker;
        CHECK(brgemm_desc_set_attr(&brg, brgattr));

        brg.with_sum = with_sum;
        CHECK(brgemm_desc_set_postops(&brg, attr, &dst_md, LDD, bia_dt));
        CHECK(brgemm_utils::brgemm_blocking(&brg));
    }

    return status::success;
}

void brg_blocking_t::update_blocks() {
    if (sp_block <= 0
            || utils::one_of(0, od_block, oh_block, ic_block, oc_block,
                    kd_block, kh_block, kw_block, os_block, ow_block))
        return;

    nb_od = div_up(od, od_block);
    nb_oh = div_up(oh, oh_block);
    nb_ic = div_up(ic, ic_block);
    nb_oc = div_up(oc, oc_block);
    nb_kd = div_up(kd, kd_block);
    nb_kh = div_up(kh, kh_block);
    nb_kw = div_up(kw, kw_block);
    nb_ow = div_up(ow, ow_block);
    if (is_os_blocking) {
        nb_os = div_up(os, os_block);
        sp = os;
        sp_block = os_block;
        nb_sp = nb_os;
    } else {
        sp = ow;
        sp_block = ow_block;
        nb_sp = nb_ow;
        iw_block = get_inp_size(iwp, ow_block, kw, stride_w, dilate_w);
    }
}

bool brg_blocking_t::fast_check_oc_block() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, acc_simd_w);
    auto res = false;
    if (oc_block == 64) {
        res = one_of(src_dt, u8, s8)
                || (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz < 192 * 4);
    } else if (oc_block == 48) {
        const bool big_spatial
                = id * ih * iw > 81 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz <= 384 * 4
                && big_spatial);
    } else
        res = true;

    return res;
}

dim_t brg_blocking_t::grid_coverage(
        dim_t nb, dim_t x, dim_t nx, dim_t xb, dim_t y, dim_t yb) {
    /*
* Calculates the total area covered by a given number of blocks in a 2D grid.
* This function estimates the number of operations required for a given blocking
* configuration in a convolution by computing the area covered by rectangles
* (blocks) in a grid. It accounts for blocks that may not fully fit into the
* grid at the edges.
* Parameters:
* nb  The total number of blocks.
* x   The size of the grid in the x dimension.
* nx  The number of grid elements in the x dimension.
* xb  The size of each block in the x dimension.
* y   The size of the grid in the y dimension.
* yb  The size of each block in the y dimension.
* Returns:
* The total area covered by the rectangles formed by the blocks.
*/
    const auto nb_x = div_up(x, xb);
    const auto nb_y = div_up(y, yb);
    // Compute how many full y blocks and x blocks are covered
    const auto blocks_per_nx = nx * nb_x;
    const auto yb_last = nb / blocks_per_nx;
    const auto xb_last = nb % blocks_per_nx;
    const auto yb_finish = yb_last % nb_y;
    const auto xb_finish = xb_last % nb_x;

    // Calculate the end positions for the last partial blocks
    const auto x_end = xb_finish * xb;
    const auto y1_end = yb_finish * yb;
    const auto y2_end = nstl::min(y, y1_end + yb);

    // Number of full rectangles
    const auto n_rect = nx * (yb_last / nb_y);
    // Area covered by full y blocks
    const auto y_tail = x * nx * y1_end;
    // Area covered by partial x and y blocks
    const auto x_tail = (x * (xb_last / nb_x) + x_end) * (y2_end - y1_end);

    // Area of a full rectangle
    const auto rect_square = x * y;
    // Total area covered
    const auto res = n_rect * rect_square + y_tail + x_tail;
    return res;
}

float brg_blocking_t::est_eff() {
    const auto N_regs = static_cast<float>(adj_ocblock);
    const auto M_regs = is_amx(isa) ? div_up(ur, amx_h) : ur;
    const auto tot_regs = is_amx(isa) ? brgemm_desc_t::AMX_TILES_NUM : max_regs;
    const auto brgemm_microkernel_eff = (N_regs * static_cast<float>(M_regs))
            / ((N_regs + static_cast<float>(M_regs))
                    * static_cast<float>(tot_regs));

    const auto ur_eff = static_cast<float>(sp_block)
            / static_cast<float>(rnd_up(sp_block, ur));
    const auto brgemm_eff = squeeze_val(static_cast<float>(ur)
                    * (2.f
                            - nstl::min(1.9f,
                                    static_cast<float>(ur)
                                            / static_cast<float>(sp_block)))
                    / 64,
            0.5f);

    const dim_t sp_amount = static_cast<dim_t>(nb_od) * nb_oh * nb_sp;
    const auto work_amount = sp_amount * mb * ngroups * nb_oc;
    const auto sp_eff = (static_cast<float>(sp)
            / static_cast<float>(rnd_up(sp, sp_block)));

    const auto oc_block_eff
            = static_cast<float>(oc) / static_cast<float>(rnd_up(oc, oc_block));

    const auto job = div_up(work_amount, static_cast<dim_t>(nthr));

    // the thread #0 is one of the most loaded threads
    // so we use it to estimate the max_job
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, 0, start, end);
    const auto thread_job = end - start;
    const dim_t max_job = (loop_order == loop_ndhwgc)
            ? grid_coverage(thread_job, oc, ngroups, oc_block, sp, sp_block)
            : grid_coverage(thread_job, sp, static_cast<dim_t>(nb_od) * nb_oh,
                      sp_block, oc, oc_block);
    const dim_t sum_job = static_cast<dim_t>(mb) * od * oh * ow * ngroups * oc;

    const float job_eff = max_job == 0
            ? 1.f
            : static_cast<float>(sum_job) / static_cast<float>(max_job * nthr);

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;

    // -- brgemm kernel: loop by simd_w  --
    l++;
    const auto inp_ur = inp_w(ur, kw_block);
    loop[l].src.set(inp_ur * simd_w, 1, acc_simd_w);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by kw in kw_block  --
    l++;
    auto src_is = rnd_inp_simd(ur, kw_block, ic_blocking_size);
    loop[l].src.set(src_is, 1, kw_block);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_blocking_size, 1);

    // -- brgemm kernel: loop by batch (grouped by kw_block) in ur  --
    l++;
    loop[l].src.set(src_is, 1);
    loop[l].dst.set(0, 1);
    dim_t wei_is = static_cast<dim_t>(kw_block) * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);
    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(kd_block * kh_block * src_is, 1);
    loop[l].dst.set(ur * oc_block, 1);
    wei_is = static_cast<dim_t>(kd_block) * kh_block * kw_block
            * oc_blocking_size;
    loop[l].wei.set(wei_is, nb_ur);

    // -- harness: loop by k_blocks in ks --
    l++;
    loop[l].src.set(kd_block * kh_block
                    * rnd_inp_simd(sp_block, kw_block, ic_blocking_size),
            1);
    loop[l].dst.set(sp_block * oc_block, nb_kd * nb_kh * nb_kw);
    loop[l].wei.set(wei_is, 1);

    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(kd * kh * rnd_inp_simd(sp_block, kw, ic_blocking_size), 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    wei_is = static_cast<dim_t>(kd) * kh * kw * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr
            = nstl::min(static_cast<dim_t>(nb_oc), div_up(job, dim_oc));
    const auto oc_thr = nstl::min(static_cast<dim_t>(oc), nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr
            = nstl::min(static_cast<dim_t>(nb_sp), div_up(job, dim_sp));
    const auto sp_thr = nstl::min(static_cast<dim_t>(sp), nb_sp_thr * sp_block);

    dim_t nb_oh_thr {1}, oh_thr {1}, nb_od_thr {1}, od_thr {1};
    if (!is_os_blocking) {
        const auto dim_oh = nb_sp * dim_sp;
        nb_oh_thr = nstl::min(static_cast<dim_t>(nb_oh), div_up(job, dim_oh));
        oh_thr = nstl::min(oh, nb_oh_thr * oh_block);

        const auto dim_od = nb_oh * dim_oh;
        nb_od_thr = nstl::min(static_cast<dim_t>(nb_od), div_up(job, dim_od));
        od_thr = nstl::min(od, nb_od_thr * od_block);
    }

    src_is = kd * kh * rnd_inp_simd(sp_block, kw, ic);

    dim_t wei_op = static_cast<dim_t>(kd) * kh * kw * adj_ocblock * ic;
    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(src_is, nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = static_cast<dim_t>(kd) * kh * kw * oc_block * ic;
        wei_op = kd * kh * kw * nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(src_is, 1);
    const auto rnd_oc_for_sp = simd_w
            * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : adj_ocblock);
    loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    // oh_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by oh_blocks --
    l++;
    src_is = kd * kh * rnd_inp_simd(sp_thr, kw, ic);
    loop[l].src.set(oh_block * src_is, 1);
    loop[l].dst.set(sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
    // od_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by od_blocks --
    l++;
    loop[l].src.set(od_block * oh_thr * src_is, 1);
    loop[l].dst.set(oh_thr * sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_od_thr);

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * src_is, nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(kd * kh * kw * oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(
            static_cast<dim_t>(mb), div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * src_is, 1);
    loop[l].dst.set(od_thr * oh_thr * sp_thr * nsimd_oc_thr * simd_w, 1);
    loop[l].wei.set(
            static_cast<dim_t>(kd) * kh * kw * nsimd_oc_thr * simd_w * ic,
            mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * od_thr * oh_thr * sp_thr
            * kd * kh * kw * ic;
    const auto dst_op = static_cast<dim_t>(mb_thr) * od_thr * oh_thr * sp_thr
            * nsimd_oc_thr;
    wei_op = kd * kh * kw * nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true,
                loop_order == loop_ndhwgc ? false : true);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false,
                loop_order == loop_ndhwgc ? true : false);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (static_cast<float>(src_op) * src_rp) / iterations;
    const auto dst_ops = (static_cast<float>(dst_op) * dst_rp) / iterations;
    const auto wei_ops = (static_cast<float>(wei_op) * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost = 1000.f * static_cast<float>(job)
            * static_cast<float>(ic_chunks) * static_cast<float>(nb_kd)
            * static_cast<float>(nb_kh) * static_cast<float>(nb_kw);

    // Avoid huge batch sizes if possible (ie prefer to block on kd/kh/kw).
    const float gemm_batch_bytes = sizeof(brgemm_batch_element_t)
            * static_cast<float>(gemm_batch_size);
    const float batch_eff = uses_batch_elements(brg_type, exec_type)
            ? nstl::min(1.f, static_cast<float>(L2) / (gemm_batch_bytes))
            : 1.f;

    const auto cache_eff = static_cast<float>(mb * od * oh * sp * ic * oc)
            / (static_cast<float>(nthr)
                    * (src_cost + dst_cost + wei_cost + call_kernel_cost));
    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff * batch_eff;
    return res_eff;
}

void brg_blocking_t::iterate_ker_block(brg_blocking_t &best_brgb,
        dim_t kd_block_, dim_t kh_block_, bool maybe_use_buffer,
        dim_t max_ow_block_thr) {

    unsigned est_k_amount = static_cast<unsigned>(ic * oc_block * wei_dsz);

    kd_block = kd_block_;
    kh_block = kh_block_;
    if (one_of(exec_type, exec_vpad, exec_trans)) {
        kw_block = kw;
        kd_block_pad = kd_block;
        kh_block_pad = kh_block;
        kw_block_pad = kw_block;
    } else {
        kw_block = (est_k_amount * kw < L2) ? kw : 1;
        kd_block_pad = kh_block >= kd ? kd : 1;
        kh_block_pad = kw_block >= kh ? kh : 1;
        kw_block_pad = kw;
    }
    gemm_batch_size = nb_ic_blocking
            * nstl::max(kd_block * kh_block * kw_block,
                    kd_block_pad * kh_block_pad * kw_block_pad);

    sp_block = -1;
    select_ic_block();

    if (exec_type == exec_vpad || is_relo_whi()) {
        od_block = 1;
        oh_block = 1;
    } else if (exec_type == exec_trans) {
        const auto ic_size = is_bf32 ? ic : ic_block;
        // `ic_block` can't be used in calculation because it's always 16 or 64
        // for bf32, and it's too low to have good oh_block size.
        const auto w_block_size
                = 2 * src_dsz * ic_size * iwp + dst_dsz * ow * oc_block;
        const auto other_size = wei_dsz * kd * kh * kw * ic_size * oc_block
                + acc_dsz * 2 * amx_h * oc_block;
        const auto L2_available = nstl::min<size_t>(
                div_up(L2, 2), other_size > L2 ? 0 : L2 - other_size);
        if (idp * ihp * w_block_size > L2_available) {
            od_block = utils::saturate<dim_t>(
                    1, od, L2_available / (ihp * w_block_size));
            if (od_block == 1)
                oh_block = utils::saturate<dim_t>(
                        1, oh, L2_available / w_block_size);
            else
                oh_block = oh;
        } else {
            od_block = 1;
            oh_block = oh;
        }
        if (is_amx(isa)) {
            // try to fit into L1
            bool L1_fit_res = false;
            auto cur_od_block = od_block;
            auto cur_oh_block = oh_block;
            const auto src_w_block_size
                    = src_dsz * ic * iwp + dst_dsz * ow * oc_block;
            if (src_w_block_size < L1) {
                cur_od_block = utils::saturate<dim_t>(
                        1, od, L1 / (ihp * src_w_block_size));
                if (cur_od_block == 1)
                    cur_oh_block = utils::saturate<dim_t>(
                            1, oh, L1 / src_w_block_size);
            }
            for (; cur_od_block > 1; cur_od_block--) {
                const auto sp_size = cur_od_block * cur_oh_block * iwp;
                if ((static_cast<float>(od)
                            / static_cast<float>(rnd_up(od, cur_od_block)))
                                > 0.9f
                        && static_cast<float>(sp_size)
                                        / static_cast<float>(rnd_up(sp, amx_h))
                                > 0.8f) {
                    L1_fit_res = true;
                    break;
                }
            }
            if (cur_od_block == 1) {
                auto tmp_oh_block = cur_oh_block;
                while (tmp_oh_block >= 1) {
                    const auto sp_size = tmp_oh_block * iwp;
                    if ((static_cast<float>(oh)
                                / static_cast<float>(rnd_up(oh, tmp_oh_block)))
                                    > 0.9f
                            && sp_size > 128) {
                        L1_fit_res = true;
                        cur_oh_block = tmp_oh_block;
                        break;
                    }
                    tmp_oh_block--;
                }
            }
            if (L1_fit_res) {
                od_block = cur_od_block;
                oh_block = cur_oh_block;
            }
        }

        // limit oh_block to have good threading
        const auto thr_oc_block = div_up(
                nthr, mb * div_up((oc > 32 ? ngroups : 1) * oc, oc_block));
        const auto thr_od_block = div_up(od, thr_oc_block);
        const auto thr_oh_block
                = div_up(oh, thr_oc_block * div_up(od, thr_od_block));
        od_block = nstl::min(od_block, thr_od_block);
        oh_block = nstl::min(oh_block, thr_oh_block);
    } else {
        od_block = 1;
        oh_block = 1;
    }

    // --- Select ow_block ----
    const auto max_ow_block_L2 = ow;
    auto start_ow_block = nstl::min<dim_t>(max_ow_block_thr, max_ow_block_L2);

    sp = ow * (is_os_blocking ? oh : 1);
    const auto start_sp_block = is_os_blocking ? ow : start_ow_block;
    auto prev_spb = dim_t(0);
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        if (is_os_blocking && spb != ow) continue;
        prev_spb = spb;
        ow_block = spb;
        sp_block = ow_block * (is_os_blocking ? oh_block : 1);

        select_ic_block();

        use_buffer = maybe_use_buffer
                && (ic_block * nb_ic_blocking < ic || kd_block != kd
                        || kh_block != kh || kw_block != kw
                        || kd_block_pad != kd || kh_block_pad != kh
                        || kw_block_pad != kw);
        if (exec_type == exec_base)
            use_buffer = use_buffer || (maybe_use_buffer && iwp != iw);

        const status_t st = estimate_brgemm_ur();
        if (st != status::success) continue;
        os_block = sp_block = ow_block * (is_os_blocking ? oh_block : 1);
        update_blocks();

        eff = est_eff();

        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
}

status_t brg_blocking_t::calc_blocks() {
    sp = ow * (is_os_blocking ? oh : 1);

    nb_ic_blocking = 1;
    // --- Select kernel blocking ---
    // if dst_dt != acc_dt and we need to store intermediate
    // results then we need the out buffer
    const auto maybe_use_buffer = (dst_dt != acc_dt || with_sum);

    std::vector<dim_t> kd_blocks(1), kh_blocks(1);
    kd_blocks[0] = kd;
    kh_blocks[0] = kh;
    if (kd != 1) {
        kd_blocks.resize(2);
        kd_blocks[1] = 1;
    }
    if (kh != 1) {
        kh_blocks.resize(2);
        kh_blocks[1] = 1;
    }

    const auto thr_eff_threshold = 0.9f;
    const auto max_ow_block_thr = utils::saturate<dim_t>(1, ow,
            static_cast<dim_t>(
                    ceil(static_cast<float>(mb * ngroups * nb_oc * os)
                            / (thr_eff_threshold * static_cast<float>(nthr)))));

    ow_block = os_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;
    for (const auto &kd_block : kd_blocks) {
        for (const auto &kh_block : kh_blocks) {
            iterate_ker_block(best_brgb, kd_block, kh_block, maybe_use_buffer,
                    max_ow_block_thr);
        }
    }
    *this = best_brgb;
    VDISPATCH_CONV_IC(IMPLICATION(!is_os_blocking, sp_block > 0),
            VERBOSE_BLOCKING_FAIL, "bad blocking parameters");

    if (is_os_blocking) {
        ow_block = ow;
        os_block = ow_block * oh_block;
        sp_block = os_block;
        ow_tail = 0;
    } else {
        ow_block = os_block = sp_block;
        os_block = sp_block = ow_block;
        ow_tail = ow % ow_block;
    }
    update_blocks();
    return status::success;
}

bool brg_blocking_t::fast_check_oc_block_1x1() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    if (is_1x1 && is_amx(isa)) return true;
    const auto rnd_oc = rnd_up(oc, acc_simd_w);
    auto res = false;
    if (oc_block == 64) {
        const auto big_spatial
                = od * oh * ow >= 64 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && big_spatial);
    } else if (oc_block == 48) {
        const auto oc_block_eff = static_cast<float>(oc)
                / static_cast<float>(rnd_up(oc, oc_block));
        res = (oc_block_eff >= 0.95f);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff_1x1() {

    auto calc_ave_blk = [&](dim_t dim, dim_t block, bool use_ave) -> float {
        if (block <= 0) return 0.f;
        const dim_t nb = dim / block;
        // only consider 2x2 tile blocking
        // TODO: ACE selects bd_block2/ld_block2 by search (ld_block2 can
        // reach 6 or 8), so this estimate does not model ACE blocking
        // correctly.
        constexpr int max_nb = 2;
        const dim_t block2 = nstl::min<dim_t>(max_nb, nb);
        if (block2 == 0) return 0.f;
        const dim_t nb2 = nb / block2;
        const dim_t nb2_tail = nb % block2;
        if (!use_ave) return static_cast<float>(block2);
        return (float(nb2) * static_cast<float>(block2)
                       + static_cast<float>(nb2_tail))
                / static_cast<float>(div_up(nb, block2));
    };
    const bool use_ocb_ave = true;
    const auto ocb_ave = calc_ave_blk(oc_block, acc_simd_w, use_ocb_ave);
    const bool use_spb_ave = false;
    const auto spb_ave = calc_ave_blk(sp_block, ur_block, use_spb_ave);
    const auto M_n_sp_blks = ur_block > 0 ? nstl::max(M, M_tail) / ur_block : 0;
    const auto M_tail_n_sp_blks
            = ur_block_tail > 0 ? M_tail / ur_block_tail : 0;
    const auto n_sp_blks = nstl::max<dim_t>(1, M_n_sp_blks + M_tail_n_sp_blks);

    // heuristic for maskrcnn workaround: use old blocking for some convolutions
    // TODO: remove this condition
    const bool maskrcnn_cond = (ic == 1024 && oc == 2048)
            || (ic == 1024 && oc == 512) || (ic == 256 && oc == 1024)
            || (ic == 512 && oc == 1024) || (ic == 512 && oc == 2048);
    const auto amx_fac = maskrcnn_cond
            ? (static_cast<float>(div_up(M + M_tail, 16))
                      / static_cast<float>(n_sp_blks))
            : (static_cast<float>(div_up(M + M_tail, 16))
                      / static_cast<float>(n_sp_blks));

    const auto brgemm_microkernel_eff = is_amx(isa)
            ? amx_fac * (static_cast<float>(ocb_ave) * spb_ave)
                    / (ocb_ave + spb_ave)
            : (static_cast<float>(adj_ocblock) * static_cast<float>(ur))
                    / static_cast<float>((ur + adj_ocblock) * max_regs);
    const auto ur_eff = static_cast<float>(sp_block)
            / static_cast<float>(rnd_up(sp_block, ur));

    // heuristic sp_block: for reduced rtus, prioritize a smaller sp_block
    const auto heur_sp_block = is_reduced_rtus && sp_block != 0
            ? 1.f / static_cast<float>(sp_block)
            : static_cast<float>(sp_block);
    const auto brgemm_eff = squeeze_val(static_cast<float>(ur)
                    * (2.f
                            - nstl::min(1.9f,
                                    static_cast<float>(ur) / heur_sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = is_os_blocking ? div_up(nb_os, nb_os_blocking)
                                          : nb_od * nb_oh * nb_sp;
    const auto work_amount
            = static_cast<dim_t>(mb) * ngroups * nb_oc * sp_amount;

    const auto sp_eff
            = static_cast<float>(sp) / static_cast<float>(rnd_up(sp, sp_block));
    const auto oc_block_eff
            = static_cast<float>(oc) / static_cast<float>(rnd_up(oc, oc_block));

    const auto job = div_up(work_amount, nthr);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr
            = nstl::min(static_cast<dim_t>(nb_oc), div_up(job, dim_oc));
    const auto oc_thr = nstl::min(static_cast<dim_t>(oc), nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr
            = nstl::min(static_cast<dim_t>(nb_sp), div_up(job, dim_sp));
    const auto sp_thr = nstl::min(static_cast<dim_t>(sp), nb_sp_thr * sp_block);

    dim_t nb_oh_thr {1}, oh_thr {1}, nb_od_thr {1}, od_thr {1};
    if (!is_os_blocking) {
        const auto dim_oh = nb_sp * dim_sp;
        nb_oh_thr = nstl::min(static_cast<dim_t>(nb_oh), div_up(job, dim_oh));
        oh_thr = nstl::min(static_cast<dim_t>(oh), nb_oh_thr * oh_block);

        const auto dim_od = nb_oh * dim_oh;
        nb_od_thr = nstl::min(static_cast<dim_t>(nb_od), div_up(job, dim_od));
        od_thr = nstl::min(static_cast<dim_t>(od), nb_od_thr * od_block);
    }

    // the thread #0 is one of the most loaded threads
    // so we use it to estimate the max_job
    dim_t start {0}, end {0};
    balance211(work_amount, nthr, 0, start, end);
    const auto thread_job = end - start;
    dim_t max_job {0};
    if (is_os_blocking) {
        max_job = (loop_order == loop_ndhwgc)
                ? grid_coverage(thread_job, oc, ngroups, oc_block, os,
                          static_cast<dim_t>(nb_os_blocking) * sp_block)
                : grid_coverage(thread_job, os, 1,
                          static_cast<dim_t>(nb_os_blocking) * sp_block, oc,
                          oc_block);
    } else {
        max_job = (loop_order == loop_ndhwgc)
                ? grid_coverage(thread_job, oc, ngroups, oc_block, sp, sp_block)
                : grid_coverage(thread_job, sp, static_cast<dim_t>(od) * oh,
                          sp_block, oc, oc_block);
    }

    const dim_t sum_job = static_cast<dim_t>(mb) * od * oh * ow * ngroups * oc;

    const float job_eff = max_job == 0
            ? 1.f
            : static_cast<float>(sum_job) / static_cast<float>(max_job * nthr);

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;
    // -- brgemm kernel: loop by simd_w  --
    l++;
    loop[l].src.set(ur * simd_w, 1, acc_simd_w);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    const auto nb_sp_no_tail = sp / sp_block;
    const auto sp_block_tail = sp % sp_block;
    const auto nb_ur_average
            = (nb_sp_no_tail * nb_ur + div_up(sp_block_tail, ur)) / nb_sp;
    loop[l].src.set(ur * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(ur * oc_block, 1);
    loop[l].wei.set(oc_blocking_size, is_amx(isa) ? nb_ur_average : nb_ur);
    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(sp_block * ic_blocking_size, 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    auto wei_is = oc_blocking_size;
    auto wei_op = adj_ocblock * ic;
    loop[l].wei.set(wei_is, 1);

    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(sp_block * rnd_simd(ic), nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = oc_block * ic;
        wei_op = nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    const auto rnd_oc_for_sp = simd_w
            * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : adj_ocblock);
    if (is_os_blocking) {
        // -- harness: loop by os_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    } else {
        // -- harness: loop by sp_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
        // -- harness: loop by oh_blocks --
        l++;
        loop[l].src.set(oh_block * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(oh_block * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
        // -- harness: loop by od_blocks --
        l++;
        loop[l].src.set(
                od_block * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(od_block * oh_thr * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_od_thr);
    }

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * rnd_simd(sp_thr * ic_blocking_size),
                nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(
            static_cast<dim_t>(mb), div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(nsimd_oc_thr * simd_w * od_thr * oh_thr * sp_thr, 1);
    loop[l].wei.set(nsimd_oc_thr * ic * simd_w, mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * od_thr * oh_thr * sp_thr
            * ic_blocking_size;
    const auto dst_op = static_cast<dim_t>(mb_thr) * nsimd_oc_thr * od_thr
            * oh_thr * sp_thr;
    wei_op = nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true, false);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false, true);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (static_cast<float>(src_op) * src_rp) / iterations;
    const auto dst_ops = (static_cast<float>(dst_op) * dst_rp) / iterations;
    const auto wei_ops = (static_cast<float>(wei_op) * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost
            = 1000.f * static_cast<float>(job) * static_cast<float>(ic_chunks);

    const auto up_sp_size = is_os_blocking ? 1 : od * oh;

    const auto cache_eff = static_cast<float>(mb * up_sp_size * sp * ic * oc)
            / (static_cast<float>(nthr)
                    * (src_cost + dst_cost + wei_cost + call_kernel_cost));

    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::calc_blocks_1x1() {
    const bool is_os_blocking_ok
            = utils::everyone_is(1, stride_d, stride_h) && iw % stride_w == 0;
    const bool is_ic_zero_padded = ic != ic_without_padding;
    is_rtus = is_ic_zero_padded || (!is_os_blocking_ok && is_amx(isa));
    const bool is_int8_convolution = everyone_is(true, one_of(src_dt, u8, s8),
            wei_dt == s8, one_of(dst_dt, f32, s32, s8, u8, bf16));
    // reduced_rtus shrinks the RTUS copy buffer for the last OS/IC block.
    // It assumes non-grouped layout and AMX tile geometry; ACE loads A via ZMMs,
    // so the AMX assumption is invalid and we must fall back to regular RTUS.
    is_reduced_rtus = is_rtus && is_int8_convolution && ic > ic_without_padding
            && everyone_is(1, stride_d, stride_h, stride_w) && ngroups == 1
            && !is_ace;

    if (is_os_blocking_ok || is_rtus) {
        sp = os;
        is_os_blocking = true;
    } else {
        sp = ow;
        is_os_blocking = false;
    }

    od_block = 1;
    oh_block = 1;
    kd_block = kh_block = kw_block = 1;
    kd_block_pad = kh_block_pad = kw_block_pad = 1;
    nb_ic_blocking = 1;

    const auto thr_eff_threshold = 0.9f;

    const auto max_sp_block_L2 = os;
    // TODO: nb_os_blocking is currently hard-coded to 1.
    nb_os_blocking = 1;
    dim_t start_sp_block = 0;

    if (is_os_blocking) {
        ow_block = 0;

        const auto max_os_block_thr
                = (src_dsz * ic >= 1024 && src_dsz * ic < 4096)
                ? nstl::max<dim_t>(nstl::min<dim_t>(16, os),
                          div_up(os, div_up(nthr, mb * div_up(oc, oc_block))))
                : nstl::max<dim_t>(div_up(2048, oc_block),
                          div_up(mb * ngroups * os, nthr));
        const auto max_os_block_L2 = max_sp_block_L2;

        auto max_os_block_aliasing = 1000000 / nthr;
        if ((oc_without_padding * os * dst_dsz) % P4K == 0) {
            max_os_block_aliasing /= 1;
            for (auto cur_oc = oc_without_padding;
                    max_os_block_aliasing * dst_dsz > 400 && cur_oc % 2 == 0
                    && cur_oc * os * dst_dsz >= P4K;
                    cur_oc /= 2) {
                max_os_block_aliasing /= 2;
            }
            max_os_block_aliasing += max_os_block_aliasing % 2 ? 0 : 1;
        }
        max_os_block_aliasing
                = nstl::min(div_up(1001, dst_dsz), max_os_block_aliasing);

        start_sp_block = utils::saturate<dim_t>(1, os,
                nstl::min<dim_t>(
                        nstl::min<dim_t>(max_os_block_thr, max_os_block_L2),
                        max_os_block_aliasing));

    } else {
        os_block = 0;

        const auto max_ow_block_thr = utils::saturate<dim_t>(1, ow,
                static_cast<dim_t>(ceil(
                        static_cast<float>(mb * ngroups * nb_oc * os)
                        / (thr_eff_threshold * static_cast<float>(nthr)))));
        const auto max_ow_block_L2 = max_sp_block_L2;

        start_sp_block = utils::saturate<dim_t>(
                1, ow, nstl::min<dim_t>(max_ow_block_thr, max_ow_block_L2));
    }
    os_block = ow_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;

    auto prev_spb = dim_t(0);
    for (auto ns = 1; ns <= sp; ns++) {
        auto spb = div_up(sp, ns);
        if (is_amx(isa)) {
            dim_t min_dis = 16;
            dim_t best_w = 16;
            const auto max_tile_width = nstl::min<dim_t>(16, sp);
            const auto min_tile_width = utils::saturate<dim_t>(1, 11, sp / 2);
            if (spb < min_tile_width) break;
            for (auto w = max_tile_width; w >= min_tile_width; w--) {
                const auto dis = nstl::additive_inverse_modulo(spb, w);
                if (dis < min_dis) {
                    min_dis = dis;
                    best_w = w;
                }
            }
            spb = nstl::min(sp, rnd_dn(spb, best_w));
            if (spb == prev_spb) continue;
        }
        if (spb == prev_spb || spb > start_sp_block) continue;
        prev_spb = spb;
        sp_block = spb;
        os_block = ow_block = spb;
        select_ic_block();
        const status_t st = estimate_brgemm_ur();
        if (st != status::success) continue;
        update_blocks();

        use_buffer = (dst_dt != acc_dt || with_sum)
                && (ic_block * nb_ic_blocking < ic || is_reduced_rtus);

        eff = est_eff_1x1();
        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
    *this = best_brgb;
    os_block = ow_block = sp_block;
    update_blocks();
}

brgemm_broadcast_t get_zp_type(const primitive_attr_t &attr, int arg) {
    return attr.zero_points_.has_default_values(arg)
            ? brgemm_broadcast_t::none
            : brgemm_broadcast_t::per_tensor;
}
status_t init_jcp(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    // take L1 as 7/8 of the real size for L1
    brg_blocking_t::L1 = (platform::get_per_core_cache_size(1) * 7) / 8;
    // take L2 as 3/4 of the real size for L2
    brg_blocking_t::L2 = (platform::get_per_core_cache_size(2) * 3) / 4;

    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // TODO: Improve memory access estimates
    brg_blocking_t::L1_k = 1.f;
    brg_blocking_t::L2_k = 2.3f;
    brg_blocking_t::L3_k = 17.f;
    // TODO: At the moment, we are primarily evaluating the fit of the data into
    // the L1/L2. Need to take into account the difference between the L3 and
    // memory.
    brg_blocking_t::mem_k = 17.f;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    // Per-tensor spatial product, weights nelems, and per-dim
    // strides/paddings/dilations must fit in int.
    // TODO: change data type of jcp fields to size_t
    VDISPATCH_CONV_IC(!has_large_size_relaxed(cd, src_d, weights_d, dst_d),
            VERBOSE_BAD_PARAM, "large size is not supported");

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.isa = isa;

    if (is_amx(isa)) {
        const int target_palette = amx::get_target_palette();
        VDISPATCH_CONV_IC(!(amx::get_max_tiles(target_palette) != 8
                                  || amx::get_max_rows(target_palette) != 16),
                VERBOSE_BLOCKING_FAIL,
                "amx: incorrect number of tiles/rows available");
    }

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc_without_padding = dst_d.dims()[1];
    jcp.oc = jcp.oc_without_padding / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.os = jcp.od * jcp.oh * jcp.ow;

    jcp.ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, jcp.ext_kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, jcp.ext_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.ext_kw);

    jcp.is_1x1 = jcp.f_pad <= 0 && jcp.back_pad <= 0 && jcp.t_pad <= 0
            && jcp.b_pad <= 0 && jcp.l_pad <= 0 && jcp.r_pad <= 0
            && utils::everyone_is(1, jcp.kd, jcp.kh, jcp.kw);

    jcp.with_bias = bias_md.format_kind != format_kind::undef;

    jcp.src_dt = src_md.data_type;
    jcp.dst_dt = dst_md.data_type;
    jcp.wei_dt = weights_md.data_type;
    jcp.bia_dt = jcp.with_bias ? bias_md.data_type : data_type::undef;

    if (one_of(jcp.src_dt, u8, s8)) {
        jcp.acc_dt = s32;
    } else if (one_of(jcp.src_dt, f32, bf16, f16, f8_e5m2, f8_e4m3)) {
        jcp.acc_dt = f32;
    } else
        return status::unimplemented;

    jcp.is_fp8 = one_of(jcp.src_dt, f8_e5m2, f8_e4m3)
            && one_of(jcp.wei_dt, f8_e5m2, f8_e4m3);
    jcp.is_fp8_convert
            = jcp.is_fp8 && one_of(isa, avx10_1_512_amx_fp16, avx10_2);
    jcp.is_f32_f16
            = everyone_is(f32, jcp.src_dt, jcp.dst_dt) && jcp.wei_dt == f16;
    jcp.is_f32_bf16
            = everyone_is(f32, jcp.src_dt, jcp.dst_dt) && jcp.wei_dt == bf16;
    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);
    jcp.acc_dsz = types::data_type_size(jcp.acc_dt);
    jcp.bia_dsz = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.simd_w = static_cast<int>(isa_max_vlen(isa) / jcp.src_dsz);
    jcp.acc_simd_w = static_cast<int>(isa_max_vlen(isa) / jcp.acc_dsz);
    jcp.is_bf32 = everyone_is(f32, jcp.src_dt, jcp.wei_dt)
            && one_of(attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any)
            && isa == avx512_core_amx;
    jcp.wei_plain = everyone_is(true, jcp.wei_dt == data_type::f32,
            is_superset(isa, avx512_core), weights_d.is_plain());
    jcp.req_fp8_convert_wsp = jcp.is_fp8_convert && !is_amx(isa);
    if (jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    const bool is_bwd_w = jcp.prop_kind == prop_kind::backward_weights;

    const auto vnni_dt = is_bwd_w                                  ? jcp.dst_dt
            : utils::one_of(true, jcp.is_f32_bf16, jcp.is_f32_f16) ? jcp.src_dt
                                                                   : jcp.wei_dt;
    jcp.is_ace = is_superset(isa, avx10_2_ace)
            && brgemm_utils::ace_dt_ok(
                    jcp.src_dt, is_bwd_w ? jcp.dst_dt : jcp.wei_dt);

    const bool req_emulation = utils::one_of(isa, avx10_1_512, avx10_2);
    const data_type_t vnni_block_dt
            = get_mac_emu_data_type(vnni_dt, isa, req_emulation);
    jcp.vnni_block
            = static_cast<int>(data_type_vnni_granularity(vnni_block_dt));

    if (one_of(jcp.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
            && jcp.ngroups == 1 && jcp.dilate_w == 0 && jcp.kw > 1
            && jcp.stride_w > 1 && jcp.l_pad <= 0 && jcp.r_pad <= 0
            && jcp.ic % jcp.vnni_block == 0
            && IMPLICATION(jcp.ic > jcp.simd_w, jcp.ic % jcp.simd_w == 0)) {
        // such convolutions are equivalent to
        // [iw / k][kw / k][stride_w / k][ic * k]
        // Considering that the layout of weights (e.g. AcdB16b16a2b for bf16
        // where 'b' is 'ic') for old and new ic should be equivalent.
        // Therefore we require
        // IMPLICATION(jcp.ic > jcp.simd_w, jcp.ic % jcp.simd_w == 0)
        // TODO: check if it may go to kw lowering
        const bool pure_1d = (jcp.mb == 1 && jcp.id == 1 && jcp.ih == 1);
        auto w_koef_max = nstl::min(jcp.kw, nstl::min(jcp.stride_w, jcp.iw));
        for (int i = 1; i <= w_koef_max; i++) {
            if (IMPLICATION(!pure_1d, jcp.iw % i == 0)
                    && IMPLICATION(jcp.ic * i > jcp.simd_w,
                            (jcp.ic * i) % jcp.simd_w == 0)
                    && jcp.iw % i == 0 && jcp.kw % i == 0
                    && jcp.stride_w % i == 0)
                jcp.trans_dim_koef = i;
        }
        if (jcp.trans_dim_koef > 1) {
            jcp.ic_without_padding *= jcp.trans_dim_koef;
            jcp.ic *= jcp.trans_dim_koef;
            jcp.iw /= jcp.trans_dim_koef;
            jcp.kw /= jcp.trans_dim_koef;
            jcp.stride_w /= jcp.trans_dim_koef;
            jcp.ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
            jcp.r_pad = calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.ext_kw);
        }
    }

    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    // dst scales is not supported for fp8 with f32/xf16 dst
    if (!dst_scales.has_default_values())
        VDISPATCH_CONV_IC(
                IMPLICATION(jcp.is_fp8, one_of(jcp.dst_dt, f8_e5m2, f8_e4m3)),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

    // TODO: optimize depthwise convolutions (for now direct approach is faster)
    const bool is_depthwise
            = with_groups && jcp.ngroups > 1 && everyone_is(1, jcp.ic, jcp.oc);
    if (is_depthwise)
        VDISPATCH_CONV_IC(!allow_perf_heuristics(jcp),
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "no optimization for depthwise convolution");

    // TODO: optimize grouped convolutions with small ic for non-amx kernels
    const bool is_grouped_small_ic
            = jcp.prop_kind != prop_kind::backward_weights && with_groups
            && jcp.ngroups > 1
            && jcp.ic <= jcp.acc_simd_w
            // Enable the shapes not supported in direct convs
            && is_groups_ok(jcp);
    const bool isa_has_small_group_perf = is_amx(isa) || jcp.isa == avx2;
    if (is_grouped_small_ic && !isa_has_small_group_perf)
        VDISPATCH_CONV_IC(!allow_perf_heuristics(jcp),
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "no optimization for grouped convolutions with small ic");

    // Dispatch the shapes to VNNI for better performance
    // TODO: optimize the perf of 3d shape with small ic and large spatial
    const auto max_small_shapes_sz = jcp.is_1x1
            ? static_cast<int32_t>(brg_blocking_t::L1) / 2
            : static_cast<int32_t>(brg_blocking_t::L1);
    const auto is_small_shape = is_amx(jcp.isa) && jcp.os <= 4 && jcp.ic <= 512
            && jcp.mb * jcp.ngroups * jcp.ic * jcp.oc <= max_small_shapes_sz;
    const auto is_3d_small_ic = is_amx(jcp.isa) && jcp.ndims == 5
            && jcp.ic * jcp.oc <= 32 && jcp.od >= 128 && jcp.oh >= 128
            && jcp.ow >= 128;
    if (one_of(jcp.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
            && (is_small_shape || is_3d_small_ic) && !jcp.is_fp8)
        VDISPATCH_CONV_IC(!allow_perf_heuristics(jcp),
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "no optimization for fwd-prop and 3d shapes / small ic");

    const bool is_signed_input = jcp.src_dt == s8;
    jcp.s8s8_compensation_required = is_signed_input && !isa_has_s8s8(jcp.isa);
    jcp.has_int8_vnni = isa_has_int8_vnni(jcp.isa);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.wei_dt == s8,
                    mayiuse(avx512_core)
                            || one_of(jcp.isa, avx2_vnni, avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.wei_dt == bf16 && !jcp.is_f32_bf16,
                    mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.wei_dt == f16 && !jcp.is_f32_f16,
                    mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(one_of(jcp.wei_dt, f8_e5m2, f8_e4m3),
                    mayiuse(avx512_core_amx_fp16) || mayiuse(avx10_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.wei_dt == f8_e5m2,
                    mayiuse(avx512_core_amx_fp16) || mayiuse(avx10_2)),
            VERBOSE_ISA_DT_MISMATCH);
    const bool is_f32
            = utils::everyone_is(f32, jcp.src_dt, jcp.wei_dt, jcp.dst_dt);
    VDISPATCH_CONV_IC(
            IMPLICATION(is_f32, one_of(isa, avx512_core, avx2) || jcp.is_bf32),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.is_f32_f16, one_of(isa, avx512_core, avx2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.is_f32_bf16, one_of(isa, avx512_core, avx2)),
            VERBOSE_ISA_DT_MISMATCH);

    jcp.amx_h = 16;
    jcp.amx_w = static_cast<int>(
            64 / (jcp.is_bf32 ? types::data_type_size(bf16) : jcp.src_dsz));

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind = p.find(primitive_kind::binary);
    const int prelu_ind = p.find(primitive_kind::prelu);
    jcp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    const auto &zp = attr.zero_points_;
    jcp.src_zero_point
            = get_zp_type(attr, DNNL_ARG_SRC) != brgemm_broadcast_t::none;
    jcp.dst_zero_point
            = get_zp_type(attr, DNNL_ARG_DST) != brgemm_broadcast_t::none;

    VDISPATCH_CONV_IC(IMPLICATION(jcp.src_zero_point || jcp.dst_zero_point,
                              utils::one_of(jcp.src_dt, s8, u8)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.src_zero_point, zp.get_mask(DNNL_ARG_SRC) == 0),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    VDISPATCH_CONV_IC(
            IMPLICATION(jcp.dst_zero_point, zp.get_mask(DNNL_ARG_DST) == 0),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    jcp.nthr = nthreads;
    jcp.copy_block_only = false;
    jcp.amx_tile_load_xx = false;
    jcp.use_M_mask = 0;
    jcp.is_os_blocking = false;
    jcp.oskip = 0;
    jcp.use_uker = false;
    jcp.use_interleave_stores = false;
    jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf_default;
    jcp.brgemm_bd_loop_innermost = false;

    if (!jcp.wei_plain) {
        // fast check data layout before spending time for blocking selection
        format_tag_t src_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);
        const bool any_eligible = is_any_eligible(jcp);
        CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));
        CHECK(init_tag(jcp.dst_tag, dst_md, dst_d, src_tag, any_eligible));
    }

    CHECK(attr.set_default_formats(&dst_md));

    VDISPATCH_CONV_IC(
            post_ops_ok(jcp, attr, dst_d), VERBOSE_UNSUPPORTED_POSTOP);

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    const auto rd_padded_block = jcp.simd_w;
    const auto kw_koef = jcp.is_relo() ? jcp.kw : 1;
    const auto kh_koef = jcp.is_relo_whi() ? jcp.kh : 1;

    jcp.is_rd_padded_to_block = !jcp.is_1x1
            && one_of(jcp.wei_dt, bf16, f16, s8, f8_e5m2, f8_e4m3)
            && jcp.ic * kw_koef * kh_koef > rd_padded_block && is_amx(isa);

    jcp.idp = jcp.id + jcp.f_pad + jcp.back_pad;
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    return status::success;
}

void adjust_nthr(jit_brgemm_conv_conf_t &jcp, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d) {
    /* adjust the thread decomposition
     * to improve the perf for small size problem
     * the threshold 8192 is empirical */
    static constexpr size_t threshold = 8 * 1024; // 8 KB per tensor
    const bool in_small = src_d.size() < threshold;
    const bool out_small = dst_d.size() < threshold;
    if (in_small && out_small && jcp.ngroups < jcp.nthr
            && jcp.nb_oc < jcp.nthr) {
        int nthr = static_cast<int>(nstl::max<dim_t>(jcp.ngroups, jcp.nb_oc));
        jcp.nthr = nstl::min(jcp.nthr, nthr);
    }
}

static void init_fp8_conversion_conf(jit_brgemm_conv_conf_t &jcp) {
    jcp.preconvert_fp8 = jcp.is_fp8_convert;
    if (!jcp.preconvert_fp8) return;

    // Keep public FP8 descriptors; blocking and addressing use FP16 buffers.
    jcp.src_dt = jcp.wei_dt = f16;
    jcp.src_dsz = jcp.wei_dsz = types::data_type_size(f16);
    jcp.simd_w = static_cast<int>(isa_max_vlen(jcp.isa) / jcp.src_dsz);
    jcp.vnni_block = data_type_vnni_granularity(f16);
    jcp.amx_w = static_cast<int>(64 / jcp.src_dsz);
    jcp.req_fp8_convert_wsp = false;
    jcp.is_rd_padded_to_block
            = !jcp.is_1x1 && jcp.ic > jcp.simd_w && is_amx(jcp.isa);
}

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    init_fp8_conversion_conf(jcp);

    const bool is_int8_convolution = everyone_is(true,
            (jcp.src_dt == u8 || jcp.src_dt == s8), jcp.wei_dt == s8,
            one_of(jcp.dst_dt, f32, s32, s8, u8, bf16));

    if (jcp.is_1x1)
        VDISPATCH_CONV_IC(!allow_perf_heuristics(jcp),
                VERBOSE_IMPL_HEURISTIC_FAIL,
                "no optimization for 1x1 convolution");
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    // TODO: check these restrictions
    if (is_amx(isa) && !jcp.is_fp8) {
        // disabled for two convolutions from ssd_resnet34
        if ((jcp.ic == jcp.oc) && (jcp.ic == 128 || jcp.ic == 256)
                && (jcp.oh == jcp.ow) && (jcp.oh == 150))
            VDISPATCH_CONV_IC(!allow_perf_heuristics(jcp),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "no optimization for current convolution dimensions from "
                    "ssd_resnet34");

        VDISPATCH_CONV_IC(!(jcp.f_pad >= jcp.ext_kd || jcp.t_pad >= jcp.ext_kh
                                  || jcp.r_pad >= jcp.ext_kw),
                VERBOSE_UNSUPPORTED_PAD_FEATURE,
                "padding mismatch with extended filter size");
    }

    using namespace data_type;
    // ======================= blocking =================================

    auto bcast_amount = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw
            * jcp.src_dsz * jcp.ic;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.kd * jcp.kh * jcp.kw
            * jcp.wei_dsz * jcp.ic;

    jcp.loop_order
            = (one_of(isa, avx2, avx2_vnni, avx2_vnni_2) && jcp.mb > jcp.nthr
                      && bcast_amount > brg_blocking_t::L2
                      && wei_amount > brg_blocking_t::L2)
            ? loop_gcndhw
            : ((bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc);
    jcp.brgemm_kernel_loop_order
            = brgemm_kernel_loop_order_t::brgemm_lo_default;

    const int min_oc_block = jcp.acc_simd_w;

    dim_t selected_ur = 0;
    MAYBE_UNUSED(selected_ur);

    auto try_exec_type = [&]() {
        brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
        best_brgb.oc_block = min_oc_block;
        const dim_t est_amx_job = div_up(jcp.mb * div_up(jcp.os, 4 * 16)
                        * jcp.ngroups * div_up(jcp.oc, 4 * 16),
                jcp.nthr);
        const bool small_amx_job = est_amx_job < 64 || jcp.oc < 256;
        auto start_ocb
                = (is_amx(isa) && jcp.is_os_blocking && small_amx_job) ? 2 : 4;
        if (one_of(isa, avx2, avx2_vnni, avx2_vnni_2)
                && jcp.loop_order == loop_gcndhw)
            start_ocb = 2;
        if (one_of(isa, avx2, avx2_vnni, avx2_vnni_2)
                && jcp.oh * jcp.ow >= 150 * 150)
            start_ocb = 2;

        start_ocb = static_cast<int>(
                nstl::min<dim_t>(div_up(jcp.oc, jcp.acc_simd_w), start_ocb));

        auto finish_ocb = 1;
        for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
            brg_blocking_t cur_brgb = zero<decltype(best_brgb)>();
            cur_brgb.get_from_jcp(jcp);
            cur_brgb.oc_block = ocb * jcp.acc_simd_w;
            cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);
            if (!cur_brgb.fast_check_oc_block()) continue;

            const status_t blocking_ok = cur_brgb.calc_blocks();
            if (blocking_ok != status::success) continue;

            const status_t st = cur_brgb.get_brgemm_ur(&attr, dst_md);
            if (st != status::success) continue;
            cur_brgb.eff = cur_brgb.est_eff();
            if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
        }
        if (best_brgb.oc_block == 0 || best_brgb.ic_block == 0
                || best_brgb.ow_block == 0)
            return false;
        best_brgb.save_to_jcp(jcp);
        selected_ur = best_brgb.ur;
        return true;
    };

    //-----------------------------------------------------------------------

    jcp.exec_type = exec_base;

    bool try_exec_vpad = false;
    bool try_exec_trans = false;
    bool try_exec_base = true;

    // TODO: this logic seems not taking dilation into which can avoid pure
    // kernel-in-pad cases.
    if (!is_amx(isa)
            && div_up(nstl::max<dim_t>(0, jcp.l_pad), jcp.stride_w) < jcp.kw
            && div_up(nstl::max<dim_t>(0, jcp.r_pad), jcp.stride_w) < jcp.kw) {
        try_exec_vpad = true;
    }

    const auto rd_padded_block = jcp.simd_w;
    // TODO: remove this restriction
    if (is_amx(isa)) {
        const auto w_padding = jcp.l_pad > 0 || jcp.r_pad > 0;
        try_exec_base = !w_padding
                && IMPLICATION(
                        jcp.ic <= rd_padded_block, jcp.ic % jcp.vnni_block == 0)
                && IMPLICATION(
                        jcp.ic > rd_padded_block, jcp.ic % rd_padded_block == 0)
                && jcp.ow > 50 /*TODO: reinvestigate this heuristic */;
        try_exec_trans = !try_exec_base;
    }
    // Try to use os_blocking for cases with ow and kw == 1
    // TODO: maybe extend this approach for other cases with small kw and ow
    if (is_superset(isa, avx512_core) && jcp.od == 1 && jcp.kw == 1
            && jcp.ow == 1
            && IMPLICATION(jcp.s8s8_compensation_required,
                    jcp.t_pad == 0 && jcp.b_pad == 0)) {
        try_exec_vpad = false;
        try_exec_trans = true;
    }

    // Determine if a relo should be applied and which one
    // required for use of VPERMB instruction in weights copy kernel
    const bool relo_supported_isa = IMPLICATION(
            is_int8_convolution, cpu().has(Xbyak::util::Cpu::tAVX512_VBMI));
    const bool relo_reasonable_isa = is_superset(isa, avx512_core);

    // try_relo_wi
    bool try_relo_wi = false;
    bool relo_conv_weights_wi = true;
    const auto rd_wi = jcp.kw * jcp.ic;
    const auto rnd_rd_wi = (float)rnd_up(rd_wi, jcp.simd_w);
    // TODO: enable relo for fp8 with f16 vnni block later
    if (!jcp.wei_plain && !(jcp.is_fp8 && jcp.vnni_block == 2)
            && relo_supported_isa && relo_reasonable_isa) {
        if (jcp.vnni_block == 1 /* For f32 weights are in needed layout */
                || (jcp.ic % jcp.vnni_block == 0
                        && IMPLICATION(
                                rd_wi > jcp.simd_w, rd_wi % jcp.simd_w == 0)))
            relo_conv_weights_wi = false;
        //TODO: support all 3d cases
        const bool relo_supported_shape = jcp.trans_dim_koef == 1
                && IMPLICATION(jcp.id > 1, relo_conv_weights_wi == false)
                && !cd.use_inversion && jcp.dilate_w == 0;

        const auto rnd_kwic = (float)jcp.kw * (float)rnd_up(jcp.ic, jcp.simd_w);
        const auto src_per_ic = (float)jcp.src_dsz * (float)jcp.mb
                * (float)jcp.id * (float)jcp.ih * (float)jcp.iw;
        const auto wei_per_ic = (float)jcp.wei_dsz * (float)jcp.oc
                * (float)jcp.kd * (float)jcp.kh * (float)jcp.kw;
        bool perf_relo = false;
        if (is_amx(jcp.isa)) {
            if (jcp.ic < jcp.simd_w / 2
                    || (rd_wi > jcp.simd_w && rnd_rd_wi / rnd_kwic < 0.5f
                            && IMPLICATION(relo_conv_weights_wi,
                                    wei_per_ic / src_per_ic <= 4)))
                perf_relo = true;
        } else {
            if (one_of(jcp.wei_dt, f32, s8)) {
                if (jcp.ic == 1) perf_relo = true;
            } else {
                if (jcp.ic < jcp.vnni_block) perf_relo = true;
            }
        }
        perf_relo = perf_relo && jcp.kw > 1;

        try_relo_wi = (relo_supported_shape && perf_relo && relo_supported_isa);
        if (try_relo_wi) try_exec_trans = true;
    }

    // try_relo_whi
    bool try_relo_whi = false;
    bool relo_conv_weights_whi = true;
    if (!jcp.wei_plain && relo_supported_isa && relo_reasonable_isa
            && !jcp.is_fp8) {
        const dim_t rd_whi = jcp.kh * jcp.kw * jcp.ic;
        //TODO: support fp8
        if (jcp.ic % jcp.vnni_block == 0
                && IMPLICATION(rd_whi > jcp.simd_w, rd_whi % jcp.simd_w == 0)
                && one_of(1, jcp.kh, jcp.kw))
            relo_conv_weights_whi = false;
        //TODO: support 3d cases
        const bool relo_supported_shape
                = everyone_is(0, jcp.dilate_h, jcp.dilate_w)
                && jcp.trans_dim_koef == 1 && jcp.ndims < 5 && !cd.use_inversion
                && IMPLICATION(jcp.s8s8_compensation_required,
                        everyone_is(0, jcp.t_pad, jcp.b_pad));

        const float rnd_rd_whi = (float)rnd_up(rd_whi, jcp.simd_w);
        const auto rnd_khkwic = (float)jcp.kh * (float)jcp.kw
                * (float)rnd_up(jcp.ic, jcp.simd_w);
        const auto src_per_ic = (float)jcp.src_dsz * (float)jcp.mb
                * (float)jcp.id * (float)jcp.ih * (float)jcp.iw;
        const auto wei_per_ic = (float)jcp.wei_dsz * (float)jcp.oc
                * (float)jcp.kd * (float)jcp.kh * (float)jcp.kw;
        // Heuristics for determining relo_whi makes sense for performance
        bool perf_relo = false;
        if (is_amx(jcp.isa)) {
            if ((jcp.ic < jcp.simd_w / 2
                        || (rd_whi > jcp.simd_w
                                && rnd_rd_whi / rnd_khkwic < 0.5f
                                && IMPLICATION(relo_conv_weights_whi,
                                        wei_per_ic / src_per_ic <= 4)))
                    && (float)rd_whi / rnd_rd_whi > (float)rd_wi / rnd_rd_wi
                    && (jcp.ow > 48 || jcp.ow % 16 == 0)
                    && IMPLICATION(
                            try_relo_wi, (float)rd_whi / rnd_rd_whi > 0.7f))
                perf_relo = true;
        } else {
            if (one_of(jcp.wei_dt, f32, s8)) {
                if (jcp.ic == 1 && jcp.ow > 4) perf_relo = true;
            } else {
                if (jcp.ic < jcp.vnni_block && jcp.ow > 4) perf_relo = true;
            }
        }
        perf_relo = perf_relo && jcp.kh > 1;

        try_relo_whi
                = (relo_supported_shape && perf_relo && relo_supported_isa);
        if (try_relo_whi) try_exec_trans = true;
    }

    bool must_exec_vpad = false;

    // TODO: in future use (kd/kh/kw) and (kd/kh/kw)_pad blocks for more
    // precise calculation of jcp.max_batch
    jcp.max_batch = jcp.kd * jcp.kh * jcp.kw;

    bool try_exec_type_res = false;

    if (try_exec_type_res == false && try_exec_trans) {
        jcp.exec_type = exec_trans;
        if (try_relo_whi) {
            jcp.relo_type = conv_brgemm_relo_type_t::whi;
            jcp.max_batch = jcp.kd;
            jcp.relo_conv_weights = relo_conv_weights_whi;
            jcp.copy_input = true; // we have to duplicate input
        } else if (try_relo_wi) {
            jcp.relo_type = conv_brgemm_relo_type_t::wi;
            jcp.max_batch = jcp.kd * jcp.kh;
            jcp.relo_conv_weights = relo_conv_weights_wi;
        }
        if (jcp.is_relo()) jcp.copy_block_only = true;

        // try loop_ndhwgc always for exec_trans
        jcp.loop_order = loop_ndhwgc;

        // we read input block only once for loop_ndhwgc, so we don't need to
        // keep it memory
        if (jcp.loop_order == loop_ndhwgc) { jcp.copy_block_only = true; }

        const auto rd_ksize = (jcp.is_relo() ? jcp.kw : 1)
                * (jcp.is_relo_whi() ? jcp.kh : 1);
        jcp.is_rd_padded_to_block
                = one_of(jcp.wei_dt, bf16, f16, s8, f8_e5m2, f8_e4m3)
                && IMPLICATION(jcp.wei_dt == f16, isa != avx10_1_512)
                && jcp.ic * rd_ksize > rd_padded_block;

        // Disable os blocking to avoid using large buffer
        // The value is empirical
        jcp.is_os_blocking = jcp.f_pad < jcp.kd && jcp.back_pad < jcp.kd
                && jcp.t_pad < jcp.kh && jcp.b_pad < jcp.kh
                && jcp.r_pad < jcp.kw && jcp.l_pad < jcp.kw && jcp.iwp < 10800;

        if (is_amx(isa)
                && IMPLICATION(!jcp.is_relo(),
                        /* heuristic */ jcp.ow < 256)) {
            jcp.use_M_mask = jcp.is_os_blocking ? 2 : 0;
            jcp.use_uker = true;
            jcp.use_interleave_stores = jcp.use_uker;
            jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
            // assuming 2x2 decomposition in amx brgemm kernel
            // and overlap of input by kw
            const auto bd_blocking = 2 * jcp.amx_h;
            const auto ld_blocking = 2 * 16;
            const auto A_ds
                    = jcp.src_dsz * bd_blocking * jcp.ic * jcp.kd * jcp.kh;
            const auto B_ds = jcp.wei_dsz * ld_blocking * jcp.ic * jcp.kd
                    * jcp.kh * jcp.kw;
            const auto C_ds = jcp.acc_dsz * bd_blocking * ld_blocking;
            if (A_ds + B_ds + C_ds > brg_blocking_t::L1)
                jcp.amx_tile_load_xx = true;
        }
        if (!jcp.use_uker) {
            // M_mask is not supported in non-uker so os_blocking possible for
            // shapes restricted by some ow/kw/stride_w/stride_h
            jcp.is_os_blocking = (jcp.is_os_blocking && jcp.stride_h == 1
                    && (jcp.ow == 1 || jcp.ext_kw <= jcp.stride_w));
        }

        try_exec_type_res = try_exec_type();
    }
    if (try_exec_type_res == false && try_exec_vpad) {
        jcp.exec_type = exec_vpad;
        try_exec_type_res = try_exec_type();
        // to avoid case when both top and bottom virtual padding are non-zero
        // TODO: remove this restriction
        const auto iw_block = (jcp.ow_block - 1) * jcp.stride_w + 1;
        if (!must_exec_vpad && (iw_block > jcp.iw)) try_exec_type_res = false;
    }
    if (try_exec_base && try_exec_type_res == false) {
        jcp.exec_type = exec_base;
        if (is_amx(isa) && jcp.ow < (8 * 1024)) {
            jcp.use_uker = true;
            jcp.use_interleave_stores = jcp.use_uker;
            jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
        }

        try_exec_type_res = try_exec_type();
    }

    if (try_exec_type_res == false) return status::unimplemented;

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    adjust_nthr(jcp, src_d, dst_d);
#endif

    // ============ end blocking ===========================================
    jcp.brg_type
            = (jcp.use_uker && one_of(jcp.exec_type, exec_base, exec_trans))
            ? brgemm_static_offs
            : brgemm_addr; // TODO: Choose right type of BRGEMM

    assert(IMPLICATION(!jcp.copy_input, !jcp.copy_block_only));

    VDISPATCH_CONV_IC(
            !(jcp.ow_block == 0 || jcp.ic_block == 0 || jcp.oc_block == 0),
            VERBOSE_BLOCKING_FAIL, "bad blocking dimensions");

    // Dispatch the shape to VNNI for better performance on AMX
    const bool is_int8_small_ic = jcp.oc == 32 && jcp.ic < jcp.simd_w / 2
            && is_int8_convolution && is_amx(jcp.isa)
            && everyone_is(640, jcp.oh, jcp.ow, jcp.ih, jcp.iw)
            && everyone_is(3, jcp.kh, jcp.kw);
    VDISPATCH_CONV_IC(!is_int8_small_ic, VERBOSE_IMPL_HEURISTIC_FAIL,
            "Dispatch the shape that has small ic/oc to VNNI");

    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, P4K), sc_size);

    if (!jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    jcp.buffer_size = static_cast<dim_t>(jcp.LDC) * jcp.M;

    jcp.nb_od = div_up(jcp.od, jcp.od_block);
    jcp.nb_oh = div_up(jcp.oh, jcp.oh_block);

    if (jcp.exec_type == exec_trans) {
        // TODO: this is rough estimation of buffer for transpose input
        dim_t ds = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.idp, jcp.od_block, jcp.kd,
                           jcp.stride_d, jcp.dilate_d)
                          + nstl::max<dim_t>(0, jcp.f_pad)
                          + nstl::max<dim_t>(0, jcp.back_pad))
                : jcp.idp;
        dim_t hs = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.ihp, jcp.oh_block, jcp.kh,
                           jcp.stride_h, jcp.dilate_h)
                          + nstl::max<dim_t>(0, jcp.t_pad)
                          + nstl::max<dim_t>(0, jcp.b_pad))
                : jcp.ihp;
        if (jcp.is_os_blocking)
            hs = div_up(rnd_up(hs * jcp.iwp, jcp.brgM), jcp.iwp)
                    + jcp.kh * (jcp.dilate_h + 1);

        jcp.inp_buffer_size = (jcp.is_relo_whi() ? jcp.kh : 1) * ds * hs
                * jcp.iwp * jcp.ngroups * jcp.nb_ic * jcp.LDA;

        if (jcp.is_relo_whi())
            jcp.inp_buffer_size +=
                    // pbuffer pointer shifts each oh step for reduced-lowering
                    (jcp.oh - 1) * jcp.oh_block * jcp.stride_h * jcp.LDA
                    // extra cacheline due to pbuffer writing full Zmm
                    + jcp.LDA;

        jcp.inp_buffer_size = rnd_up(jcp.inp_buffer_size, P4K);

        jcp.inp_buffer_mask_size = rnd_up(static_cast<dim_t>(jcp.nb_od)
                        * jcp.nb_oh * jcp.nb_ow * jcp.ngroups * jcp.nb_ic,
                P4K);
    }

    if (jcp.s8s8_compensation_required) {
        weights_md.extra.flags = 0 | memory_extra_flags::compensation_conv_s8s8;
        weights_md.extra.compensation_mask = with_groups ? 0x3 : 0x1;
        if (!jcp.has_int8_vnni) {
            weights_md.extra.flags |= memory_extra_flags::scale_adjust;
            weights_md.extra.scale_adjust = 0.5f;
        }
    }
    jcp.scale_adjust_factor
            = (jcp.s8s8_compensation_required && !jcp.has_int8_vnni)
            ? 1 / weights_md.extra.scale_adjust
            : 1.0f;
    if (jcp.src_zero_point) {
        weights_md.extra.flags
                |= memory_extra_flags::compensation_conv_asymmetric_src;
        weights_md.extra.asymm_compensation_mask = with_groups ? 0x3 : 0x1;
    }

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    jcp.with_src_scales = !src_scales.has_default_values();
    jcp.with_wei_scales = !wei_scales.has_default_values()
            || jcp.scale_adjust_factor != 1.0f;
    jcp.is_oc_scale = wei_scales.get_mask() > 0;
    jcp.with_dst_scales = !dst_scales.has_default_values();

    const bool compensation_w_padding
            = (jcp.s8s8_compensation_required || jcp.src_zero_point)
            && !everyone_is(0, jcp.t_pad, jcp.back_pad, jcp.f_pad, jcp.b_pad,
                    jcp.l_pad, jcp.r_pad);
    // estimate the number of kernel range combination for compensation
    const auto kd_cnt = 1 + utils::div_up(abs(jcp.f_pad), jcp.dilate_d + 1)
            + utils::div_up(abs(jcp.back_pad), jcp.dilate_d + 1);
    const auto kh_cnt = 1 + utils::div_up(abs(jcp.t_pad), jcp.dilate_h + 1)
            + utils::div_up(abs(jcp.b_pad), jcp.dilate_h + 1);
    jcp.ker_ranges_size = jcp.exec_type == exec_trans ? kd_cnt
                    * nstl::min(
                            jcp.oh, rnd_up(jcp.oh_block + kh_cnt, jcp.oh_block))
                                                      : kd_cnt * kh_cnt;
    jcp.comp_ow_size = get_comp_ow_size(jcp);
    jcp.comp_a_buffer_size = jcp.ngroups * jcp.nb_oc * jcp.ker_ranges_size
            * jcp.comp_ow_size * jcp.oc_block;
    jcp.s8s8_comp_buffer_size = jcp.comp_a_buffer_size;

    // For padding shapes, we calculate the comp along with the computation
    // inside brgemm kernel when output size is small to get optimal perf
    // For shapes with large ow we calculate the comp inside brgemm kernel too
    // because current implementation of brgemm_comp_pad kernel unrolled by ow
    // so not optimal for large ow.
    // Otherwise we calculate the comp using brgemm_comp_pad kernel
    const auto output_sz = static_cast<dim_t>(jcp.mb) * jcp.ngroups * jcp.oc
            * jcp.od * jcp.oh * jcp.ow;
    // TODO: revise below condition to avoid limitation for big ow
    const auto shape_for_brgemm_kernel
            = (output_sz <= 8192 && jcp.oc < 512) || jcp.ow > 128;
    const auto is_relo = jcp.is_relo() && jcp.relo_conv_weights;
    jcp.req_brg_comp_pad = compensation_w_padding && jcp.exec_type != exec_trans
            && IMPLICATION(!is_relo, shape_for_brgemm_kernel)
            && IMPLICATION(
                    jcp.exec_type == exec_vpad, jcp.comp_a_buffer_size > 1024);
    jcp.req_cal_comp_pad = compensation_w_padding && !jcp.req_brg_comp_pad
            && IMPLICATION(jcp.exec_type == exec_vpad,
                    jcp.t_pad > 0 || jcp.b_pad > 0 || jcp.f_pad > 0
                            || jcp.back_pad > 0);

    // enable ununroll_bd_loop for big shapes to reduce kernel sizes
    jcp.ununroll_bd_loop
            = static_cast<dim_t>(jcp.M) * jcp.N * (jcp.is_bf32 ? 1 : 2)
            > 8 * 1024;

    jcp.fp8_convert_wsp_size = static_cast<dim_t>(sizeof(float16_t))
            * isa_max_vlen(jcp.isa)
            * nstl::max(
                    static_cast<dim_t>(isa_num_vregs(jcp.isa)), jcp.ow_block);

    VDISPATCH_CONV_IC(IMPLICATION(jcp.is_bf32, jcp.use_uker),
            "cannot use unrolled kernel for current datatype configuration");

    return status::success;
}

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    init_fp8_conversion_conf(jcp);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    VDISPATCH_CONV_IC(jcp.is_1x1, VERBOSE_BAD_FLAGS);

    using namespace data_type;
    // ===================== blocking =================================

    auto bcast_amount
            = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    if (is_amx(isa)) {
        // round up ic if needed
        const dim_t n_vnni_blocks = utils::div_up(jcp.ic, jcp.vnni_block);
        const dim_t ic_block = nstl::min<dim_t>(jcp.acc_simd_w, n_vnni_blocks)
                * jcp.vnni_block;

        jcp.extendable_k = jcp.ic > jcp.simd_w && jcp.ic % jcp.simd_w;

        const bool do_zeropad = !jcp.is_bf32 && !jcp.extendable_k
                && (jcp.ic % jcp.vnni_block != 0 || jcp.ic > ic_block);
        if (do_zeropad) jcp.ic = utils::rnd_up(jcp.ic, ic_block);
        const auto ic_padded_block = jcp.simd_w;
        jcp.is_rd_padded_to_block = jcp.ic > ic_padded_block && !jcp.is_bf32;

        // try to choose optimal loop order
        // TODO: incorporate loop order into smart blocking selection
        auto wei_size = (size_t)jcp.oc * jcp.ic * jcp.wei_dsz;
        auto max_size = 0.75f * static_cast<float>(brg_blocking_t::L2);
        const dim_t os = static_cast<dim_t>(jcp.od) * jcp.oh * jcp.ow;
        const dim_t os_cutoff = 400; // approximate and empiric
        const bool use_loop_ngcdhw = max_size < static_cast<float>(wei_size)
                || (jcp.mb == 1 && os < os_cutoff);
        jcp.loop_order = use_loop_ngcdhw ? loop_ngcdhw : loop_ndhwgc;

        // The blocking search below dispatches on both flags, so for ACE they
        // have to be set before the search rather than after it.
        if (jcp.is_ace) {
            jcp.use_uker = brgemm_utils::ace_prefer_uker();
            // The unrolled kernel already disables interleaved stores on ACE,
            // setting it here keeps the blocking search on the same value.
            jcp.use_interleave_stores = false;
        }
    }

    const auto min_oc_block = jcp.acc_simd_w;

    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // max_batch is 1 for 1x1 convolutions
    jcp.max_batch = 1;

    brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
    best_brgb.oc_block = min_oc_block;
    dim_t start_ocb = 4;
    start_ocb = nstl::min<dim_t>(div_up(jcp.oc, jcp.acc_simd_w), start_ocb);

    dim_t finish_ocb = 1;

    const bool is_os_blocking_ok
            = utils::everyone_is(1, jcp.stride_d, jcp.stride_h)
            && jcp.iw % jcp.stride_w == 0;
    if (jcp.wei_plain && is_os_blocking_ok) {
        start_ocb = div_up(jcp.oc, jcp.acc_simd_w);
    }

    for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
        brg_blocking_t cur_brgb = zero<decltype(cur_brgb)>();
        cur_brgb.get_from_jcp(jcp);
        cur_brgb.oc_block = ocb * min_oc_block;
        cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);

        if (!cur_brgb.fast_check_oc_block_1x1()) continue;

        cur_brgb.calc_blocks_1x1();
        const status_t st = cur_brgb.get_brgemm_ur(&attr, dst_md);
        if (st != status::success) continue;
        cur_brgb.eff = cur_brgb.est_eff_1x1();
        if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
    }
    best_brgb.save_to_jcp(jcp);

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    adjust_nthr(jcp, src_d, dst_d);
#endif

    // =============== end blocking =================================

    jcp.brg_stride_a = jcp.ic_block * jcp.src_dsz;
    jcp.brg_stride_b = jcp.ic_block * jcp.oc_without_padding * jcp.wei_dsz;

    VDISPATCH_CONV_IC(!(jcp.ic_block == 0 || jcp.oc_block == 0),
            VERBOSE_BLOCKING_FAIL, "bad blocking dimensions");

    // Configure matrix sizes

    if (best_brgb.is_os_blocking) {
        VDISPATCH_CONV_IC(jcp.os_block != 0, VERBOSE_BLOCKING_FAIL,
                "bad blocking dimensions");
        jcp.M = jcp.brgM = jcp.os_block;
        jcp.M_tail = jcp.brgM_tail = jcp.os % jcp.os_block;
    } else {
        VDISPATCH_CONV_IC(jcp.ow_block != 0, VERBOSE_BLOCKING_FAIL,
                "bad blocking dimensions");
        jcp.M = jcp.brgM = jcp.ow_block;
        jcp.M_tail = jcp.brgM_tail = jcp.ow % jcp.ow_block;
    }

    jcp.K = jcp.ic >= jcp.ic_block ? jcp.ic_block : 0;
    jcp.N = jcp.oc >= jcp.oc_block ? jcp.oc_block : 0;
    jcp.N_tail = jcp.oc % jcp.oc_block;
    jcp.K_tail = jcp.ic % jcp.ic_block;

    jcp.gemm_batch_size = jcp.nb_ic_blocking;
    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, P4K), sc_size);

    if (is_amx(isa) && !jcp.is_ace) {
        // heuristic for small mb
        const bool is_small_mb = jcp.nthr > 1 && jcp.mb == 1
                && jcp.ic * jcp.oh <= 28 * 1024 && jcp.oc * jcp.oh <= 14 * 1024;
        MAYBE_UNUSED(is_small_mb);
        // non-unrolled kernel does not support bf32, only dispatch unrolled
        // kernel for now
        jcp.use_uker = jcp.is_bf32 || !is_small_mb;
        jcp.use_interleave_stores = jcp.use_uker;
    }

    // TODO: heuristic to dispatch BF32 BRGeMM
    // The following condition checks for shapes where down-convert execution
    // in brgemm fails
    VDISPATCH_CONV_IC(!(jcp.is_bf32 && jcp.ic < 64 && jcp.ic % 32 != 0),
            "invalid datatype or number of input channels");

    if (jcp.use_uker)
        jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
    if (!jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));
    CHECK(attr.set_default_formats(&dst_md));

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    // no inp buffer or brgemm_vpad for 1x1
    constexpr int align_size = platform::get_cache_line_size();
    jcp.exec_type = jcp.is_rtus ? exec_trans : exec_base;

    /* calculate the ic block size for reduced_rtus as:
     *
     * [------src_input_data--------|*****rtus_buffer*****00000000000000]
     * ^ -- directly into brgemm -- ^ -- reduced_rtus -- ^ zero-padding ^ic_sz
     **/
    constexpr int int8_outer_vnni_block = 16; // currently, only for AMX
    const int ic_padded_block = int8_outer_vnni_block * jcp.vnni_block;
    const dim_t ic_vnni_block = jcp.ic_without_padding / ic_padded_block;
    const dim_t effective_rtus_ic_block
            = jcp.ic_without_padding - (ic_vnni_block * ic_padded_block);
    const dim_t rtus_padded_ic_size
            = rnd_up(effective_rtus_ic_block, ic_padded_block);
    // used as K dimension for BRGeMM:
    jcp.rtus_ic_size = jcp.is_reduced_rtus ? effective_rtus_ic_block : 1;

    // ic_size + padding, used for BRGeMM LDA:
    jcp.rtus_padded_ic_size = jcp.is_reduced_rtus ? rtus_padded_ic_size : 1;

    const size_t rtus_buffer_size = jcp.is_reduced_rtus
            ? jcp.rtus_padded_ic_size * jcp.os_block
            : static_cast<size_t>(jcp.LDA) * jcp.os;
    jcp.inp_buffer_size
            = jcp.is_rtus ? rnd_up(rtus_buffer_size, align_size) : 0;

    const size_t rtus_mask_size = !jcp.is_reduced_rtus
            ? div_up(jcp.nb_ic, jcp.nb_ic_blocking) * jcp.nb_os
            : 1; // only uses 1 block, so only 1 mask needed
    jcp.inp_buffer_mask_size
            = jcp.is_rtus ? rnd_up(rtus_mask_size, align_size) : 0;
    jcp.buffer_size = jcp.LDC * jcp.M;

    if (jcp.s8s8_compensation_required) {
        weights_md.extra.flags = 0 | memory_extra_flags::compensation_conv_s8s8;
        weights_md.extra.compensation_mask = with_groups ? 0x3 : 0x1;
        if (!jcp.has_int8_vnni) {
            weights_md.extra.flags |= memory_extra_flags::scale_adjust;
            weights_md.extra.scale_adjust = 0.5f;
        }
    }
    jcp.scale_adjust_factor
            = (jcp.s8s8_compensation_required && !jcp.has_int8_vnni)
            ? 1 / weights_md.extra.scale_adjust
            : 1.0f;
    if (jcp.src_zero_point) {
        weights_md.extra.flags
                |= memory_extra_flags::compensation_conv_asymmetric_src;
        weights_md.extra.asymm_compensation_mask = with_groups ? 0x3 : 0x1;
    }
    jcp.req_cal_comp_pad = false;
    jcp.comp_ow_size = 1;
    jcp.s8s8_comp_buffer_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    jcp.comp_a_buffer_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    jcp.with_src_scales = !src_scales.has_default_values();
    jcp.with_wei_scales = !wei_scales.has_default_values()
            || jcp.scale_adjust_factor != 1.0f;
    jcp.is_oc_scale = wei_scales.get_mask() > 0;
    jcp.with_dst_scales = !dst_scales.has_default_values();

    // enable ununroll_bd_loop for big shapes to reduce kernel sizes
    jcp.ununroll_bd_loop
            = static_cast<dim_t>(jcp.M) * jcp.N * (jcp.is_bf32 ? 1 : 2)
            > 8 * 1024;

    jcp.fp8_convert_wsp_size = static_cast<dim_t>(sizeof(float16_t))
            * isa_max_vlen(jcp.isa)
            * nstl::max(
                    static_cast<dim_t>(isa_num_vregs(jcp.isa)), jcp.ow_block);

    return status::success;
}

void set_amx_wsp_per_thread(jit_brgemm_conv_conf_t &jcp) {
    // ensure buffers for individual threads do not lie on same page and also
    // they are not contiguous.
    jcp.amx_buf_size_per_thread
            = utils::rnd_up(jcp.amx_buf_size_per_thread + 1, P4K);
}

status_t init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp, const memory_desc_t &src_md,
        const memory_desc_t &weights_md, const memory_desc_t &dst_md) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    if (jcp.preconvert_fp8) {
        scratchpad.book(key_conv_fp8_src, src_d.nelems(true),
                types::data_type_size(f16), 0, P4K);
        scratchpad.book(key_conv_fp8_weights, weights_d.nelems(true),
                types::data_type_size(f16), 0, P4K);
    }

    if (uses_batch_elements(jcp.brg_type, jcp.exec_type)) {
        scratchpad.book(key_brgemm_primitive_batch,
                static_cast<size_t>(jcp.nthr) * jcp.adjusted_batch_size,
                sizeof(brgemm_batch_element_t), 64, P4K);
    }
    if (jcp.exec_type == exec_trans) {
        size_t inp_buffer_size
                = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_size;
        scratchpad.book(key_conv_brgemm_inp_buffer, inp_buffer_size,
                jcp.src_dsz, 0, P4K);
        size_t inp_buffer_mask_size
                = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_mask_size;
        scratchpad.book(key_conv_brgemm_inp_buffer_mask, inp_buffer_mask_size,
                sizeof(uint8_t), 0, P4K);
    }
    if (jcp.is_relo()) {
        auto rd = jcp.kw * jcp.ic;
        auto rd_koef = jcp.kh;
        if (jcp.is_relo_whi()) {
            rd = jcp.kw * jcp.kh * jcp.ic;
            rd_koef = 1;
        }

        const auto padded_rd
                = jcp.vnni_block * (jcp.is_rd_padded_to_block ? 16 : 1);

        const auto non_oc_size = rd_koef * rnd_up(rd, padded_rd);

        const auto wei_buffer_size = rnd_up(
                (size_t)jcp.ngroups * jcp.nb_oc * jcp.oc_block * non_oc_size,
                1024);
        scratchpad.book(
                key_conv_amx_wei_buffer, wei_buffer_size, jcp.wei_dsz, 0, P4K);
    }

    if (jcp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer, jcp.nthr * jcp.buffer_size,
                jcp.acc_dsz, 0, P4K);
    }
    if (is_amx(jcp.isa)) {
        scratchpad.book(key_conv_amx_tile_buffer,
                jcp.nthr * jcp.amx_buf_size_per_thread, sizeof(char), 0, P4K);
    }
    if (jcp.s8s8_compensation_required && jcp.req_cal_comp_pad) {
        scratchpad.book(key_brgemm_primitive_buffer_comp,
                jcp.s8s8_comp_buffer_size, sizeof(int32_t), 0, P4K);
    }

    if (jcp.src_zero_point && jcp.req_cal_comp_pad) {
        scratchpad.book(key_brgemm_primitive_zp_comp_a, jcp.comp_a_buffer_size,
                sizeof(int32_t), 0, P4K);
    }

    if (jcp.with_dst_scales) {
        // See brgemm_types.hpp comment for `with_dst_scales`.
        scratchpad.book(key_conv_dst_scales,
                static_cast<size_t>(jcp.nthr) * sizeof(float), P4K);
    }

    // Check scratchpad size to avoid allocating huge buffers
    if (jcp.exec_type == exec_trans) {
        constexpr size_t scratchpad_limit_by_absolute_value = (size_t)32
                << 30; // 32Gb - TODO: may it's too large?
        const size_t scratchpad_limit_by_tensor_sizes = (size_t)64 * jcp.nthr
                * (src_d.size() + weights_d.size() + dst_d.size());
        const size_t scratchpad_limit
                = nstl::min(scratchpad_limit_by_absolute_value,
                        scratchpad_limit_by_tensor_sizes);
        if (scratchpad.size() > scratchpad_limit) return status::unimplemented;
    }

    if (jcp.req_fp8_convert_wsp) {
        scratchpad.book(key_brgemm_primitive_fp8_convert_wsp,
                static_cast<size_t>(jcp.nthr) * jcp.fp8_convert_wsp_size,
                sizeof(float16_t), 0, P4K);
    }

    return status::success;
}

void balance_bwd_w(jit_brgemm_conv_conf_t &jcp) {

    const auto os_chunks = jcp.nthr_mb_work;
    const auto oc_chunks = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const auto ic_chunks = div_up(jcp.nb_ic, jcp.nb_ic_blocking);

    auto calc_mem_cost = [&jcp, os_chunks, oc_chunks, ic_chunks](int nthr_mb,
                                 int nthr_g, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
            * optimizer tries to minimize memory consumption. few notes:
            *  (n1) if weights tensor size is less than source and destination
            *       tensors we apply the ratio of the source and destination
            *       tensor sizes to weights one as compensation coefficient to
            *       avoid parallelization across batch size only, otherwise we
            *       apply additional coefficient to source component based on
            *       performance measurements
            *  (n2) use scales based on output vs input channels ratio for
            *       source and destination components to improve threading
            *       balance across input and output channels */

        const dim_t src_type_size = 2;
        const dim_t wei_type_size = 4;
        const dim_t acc_type_size = wei_type_size;

        const auto wei_ks = jcp.kh * jcp.kw * jcp.kd;

        const auto src_spatial = (dim_t)jcp.mb * jcp.id * jcp.ih * jcp.tr_iw;
        const auto dst_spatial = (dim_t)jcp.mb * jcp.od * jcp.oh * jcp.tr_ow;

        dim_t src_size = src_spatial * jcp.ic * src_type_size;
        dim_t dst_size = dst_spatial * jcp.oc * src_type_size;
        dim_t wei_size = (dim_t)jcp.oc * jcp.ic * wei_ks * wei_type_size;

        float wei_compensation_scale = 0.5f
                * static_cast<float>(dst_size + src_size)
                / static_cast<float>(wei_size);
        float oi_channels_ratio
                = (float)(oc_chunks) / static_cast<float>(ic_chunks);

        auto get_src_coef = [=]() {
            float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
            if (wei_compensation_scale < 1.0f) src_coef *= 4.0f;
            return src_coef;
        };

        auto get_dst_coef
                = [=]() { return nstl::max(oi_channels_ratio, 1.0f); };

        auto get_wei_coef
                = [=]() { return nstl::max(wei_compensation_scale, 1.0f); };

        const float src_coef = get_src_coef();
        const float dst_coef = get_dst_coef();
        const float wei_coef = get_wei_coef();

        const auto thr_mb = div_up(os_chunks, nthr_mb);
        const auto nb_oc_job = jcp.oc_block * jcp.nb_oc_blocking;
        const auto nb_ic_job = jcp.ic_block * jcp.nb_ic_blocking;

        const auto src_chunk = src_spatial / os_chunks;
        const auto dst_chunk = dst_spatial / os_chunks;

        const auto thr_g = div_up(jcp.ngroups, nthr_g);
        const auto thr_ic_b = div_up(ic_chunks, nthr_ic_b);
        const auto thr_src_sp = thr_mb * src_chunk / jcp.stride_d / jcp.stride_h
                / jcp.stride_w;
        const auto thr_dst_sp = thr_mb * dst_chunk;
        const auto thr_ic_amount = thr_ic_b * nb_ic_job;

        const auto thr_oc_b = div_up(oc_chunks, nb_oc_job * nthr_oc_b);

        const auto thr_oc_amount = thr_oc_b * nb_oc_job;
        float src_v = src_type_size * src_coef * static_cast<float>(thr_g)
                * static_cast<float>(thr_ic_amount)
                * static_cast<float>(thr_src_sp);
        float dst_v = src_type_size * dst_coef * static_cast<float>(thr_g)
                * static_cast<float>(thr_oc_amount)
                * static_cast<float>(thr_dst_sp);
        float wei_v = acc_type_size * wei_coef * static_cast<float>(thr_g)
                * static_cast<float>(thr_oc_amount)
                * static_cast<float>(thr_ic_amount)
                * static_cast<float>(wei_ks);

        return src_v + dst_v + wei_v;
    };

    auto balance = [&jcp, calc_mem_cost, oc_chunks](int &nthr_, int &nthr_mb_,
                           int &nthr_g_, int &nthr_oc_b_, int &nthr_ic_b_) {
        nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

        if (jcp.nthr < jcp.ngroups) {
            /* simplification... fortunately it doesn't hurt much */
            nthr_ = nthr_g_ = jcp.nthr;
            return;
        }

        nthr_g_ = static_cast<int>(jcp.ngroups);
        const int nthr = jcp.nthr / nthr_g_;

        float best_mem_cost
                = calc_mem_cost(nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_);

        /* find the best thread distribution with lowest memory cost */

        const int nthr_mb_max = nstl::min(nthr, jcp.nthr_mb_work);
        for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
            const int nthr_par = nthr / nthr_mb;
            const int nthr_oc_b_max = static_cast<int>(nstl::min<dim_t>(
                    nthr_par, oc_chunks)); // Amount of nb_oc_blocks
            for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
                int nthr_ic_b = static_cast<int>(
                        nstl::min<dim_t>(nthr_par / nthr_oc_b,
                                (jcp.nb_ic / jcp.nb_ic_blocking)));

                float mem_cost
                        = calc_mem_cost(nthr_mb, nthr_g_, nthr_oc_b, nthr_ic_b);
                if (mem_cost <= best_mem_cost) {
                    best_mem_cost = mem_cost;
                    nthr_mb_ = nthr_mb;
                    nthr_oc_b_ = nthr_oc_b;
                    nthr_ic_b_ = nthr_ic_b;
                }
            }
        }

        if (nthr_mb_ > nthr / 2 && nthr_mb_ < nthr)
            nthr_mb_ = nstl::min(jcp.nthr_mb_work, nthr);
        nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

        assert(nthr_ <= jcp.nthr);
    };

    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    balance(nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);

    // empiric balancing for some shapes
    const auto sps = (jcp.ih * jcp.iw);
    bool neat_1x1
            = everyone_is(1, jcp.id, jcp.kh, jcp.kw, jcp.ngroups, jcp.stride_h);
    if (neat_1x1 && jcp.nthr >= 28 && jcp.mb >= jcp.nthr) {
        const bool more_oc = (jcp.ic < jcp.oc);
        if (sps >= 56 * 56 && jcp.ic >= 64 && jcp.oc >= 64) {
            nthr_mb = jcp.nthr;
            nthr_oc_b = 1;
        } else if (sps >= 28 * 28 && jcp.ic >= 128 && jcp.oc >= 128) {
            nthr_mb = jcp.nthr / 4;
            nthr_oc_b = more_oc ? jcp.nthr / nthr_mb : 1;
        } else if (sps >= 14 * 14 && jcp.ic >= 256 && jcp.oc >= 256) {
            nthr_mb = div_up(jcp.nthr, 8);
            nthr_oc_b = more_oc ? jcp.nthr / nthr_mb : 1;
        } else if (sps >= 7 * 7 && jcp.ic >= 512 && jcp.oc >= 512) {
            nthr_mb = div_up(jcp.nthr, 14);
            nthr_oc_b = more_oc ? jcp.nthr / nthr_mb : 1;
        }
        nthr_ic_b = jcp.nthr / (nthr_mb * nthr_oc_b);
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    } else if (is_amx(jcp.isa)
            && jcp.nthr <= static_cast<int>(platform::get_num_cores())
            && jcp.mb <= jcp.nthr / 2 && jcp.oc >= 64 && jcp.ic >= 64
            && jcp.ngroups == 1) {
        // This heuristic is intended for usual convolutions if the minibatch
        // is much less than the number of threads: it tries to divide the
        // total amount of work into more-less 4-dimensional (by mb, g, oc, ic)
        // "cubic" pieces
        enum bwd_w_dims { g, ic, oc, sp };
        constexpr int nd = 4;
        // Keep maximum values for each dimension as a map
        std::map<bwd_w_dims, dim_t> maxv;
        maxv.emplace(bwd_w_dims::g, jcp.ngroups);
        maxv.emplace(bwd_w_dims::ic, div_up(jcp.nb_ic, 2));
        maxv.emplace(bwd_w_dims::oc, div_up(jcp.nb_oc, 2));
        maxv.emplace(bwd_w_dims::sp, jcp.mb * jcp.od * jcp.oh);

        // Keep dimension values as a vector
        std::vector<std::pair<double, bwd_w_dims>> dv;
        const auto ks = jcp.kd * jcp.kh * jcp.kw;
        double v = (jcp.ngroups > 1)
                ? static_cast<double>(jcp.ic) * static_cast<double>(jcp.oc)
                        * static_cast<double>(jcp.ngroups)
                        * static_cast<double>(jcp.ngroups)
                        * static_cast<double>(ks)
                : 1;
        dv.emplace_back(v, bwd_w_dims::g);
        v = 5 * static_cast<double>(div_up(jcp.ic, jcp.amx_h))
                * static_cast<double>(ks);
        dv.emplace_back(v, bwd_w_dims::ic);
        v = 3 * static_cast<double>(div_up(jcp.oc, jcp.amx_h))
                * static_cast<double>(ks);
        dv.emplace_back(v, bwd_w_dims::oc);
        v = static_cast<double>(
                div_up(jcp.mb * jcp.od * jcp.oh * jcp.ow, jcp.amx_w));
        dv.emplace_back(v, bwd_w_dims::sp);
        // Estimate the size of "cubic" piece
        double xd = 1;
        for (int j = 0; j < nd; j++)
            xd *= dv[j].first;
        xd = pow(xd / jcp.nthr, 1.f / nd);
        // Adjust piece to fit into dimensions
        std::sort(dv.begin(), dv.end());
        double tot_v = 1;
        for (int i = 0; i < nd; i++) {
            auto &dvf = dv[i].first;
            const auto &dvs = dv[i].second;
            const auto maxvf = static_cast<double>(maxv[dvs]);
            if (dvf < xd) {
                v = 1;
                xd = 1;
                for (int j = i + 1; j < nd; j++)
                    xd *= dv[j].first;
                xd = pow(xd / jcp.nthr, 1.f / static_cast<float>(nd - i - 1));
            } else {
                v = nstl::min(dvf / xd, maxvf);
            }
            tot_v *= v;
            dvf = v;
        }
        std::sort(dv.begin(), dv.end());

        // Normalize dimension values so product should be ~= nthr
        double knorm = pow(jcp.nthr / tot_v, 1.f / nd);
        tot_v = 1;
        for (int i = 0; i < nd; i++) {
            auto &dvf = dv[i].first;
            auto &dvs = dv[i].second;
            const auto maxvf = static_cast<double>(maxv[dvs]);
            const auto new_dvf = dvf * knorm;
            dvf = utils::saturate(1., maxvf, new_dvf);
            knorm *= pow(new_dvf / dvf, 1.f / static_cast<float>(nd - i - 1));
            tot_v *= dvf;
        }
        std::sort(dv.begin(), dv.end());
        knorm = jcp.nthr / tot_v;
        for (int i = 0; i < nd; i++) {
            auto &dvf = dv[i].first;
            auto &dvs = dv[i].second;
            const auto maxvf = static_cast<double>(maxv[dvs]);
            const auto new_dvf = dvf * knorm;
            dvf = utils::saturate(1., maxvf, new_dvf);
            knorm = new_dvf / dvf;
        }
        std::sort(dv.begin(), dv.end());

        // Selecting the number of threads for every dimension closest to what
        // we defined before
        auto calc_diff
                = [&](const std::vector<std::pair<int, bwd_w_dims>> &cv) {
            auto tot_n = 1;
            double res = 1;
            for (int i = 0; i < nd; i++) {
                const auto nvf = dv[i].first;
                const auto n = cv[i].first;
                const auto v = maxv[cv[i].second];
                const auto disb = nvf * static_cast<double>(rnd_up(v, n))
                        / static_cast<double>(v);
                const auto nf = static_cast<double>(n);
                const auto var = ((nf > nvf) ? (nf / nvf) : (nvf / nf));
                tot_n *= n;
                res *= disb * var;
            }
            const auto thr_disb = static_cast<double>(jcp.nthr) / tot_n;
            return res * thr_disb;
        };

        // nv: vector to keep result of selection
        std::vector<std::pair<int, bwd_w_dims>> nv;
        // Initial vector and estimation
        for (int i = 0; i < nd; i++) {
            const auto dvf = dv[i].first;
            const auto dvs = dv[i].second;
            const auto maxvf = static_cast<int>(maxv[dvs]);
            nv.emplace_back(
                    utils::saturate(1, maxvf, static_cast<int>(dvf + 0.5f)),
                    dvs);
        }
        nv[nd - 1].first = jcp.nthr / (nv[0].first * nv[1].first * nv[2].first);
        double best_diff = calc_diff(nv);

        // Iterate through all combinations of numbers
        std::vector<std::pair<int, bwd_w_dims>> cv = nv;
        const auto n0_max = jcp.nthr;
        for (int n0 = 1; n0 <= n0_max; n0++) {
            if (n0 > maxv[dv[0].second]) continue;
            cv[0].first = n0;
            const auto n1_max = n0_max / n0;
            for (int n1 = 1; n1 <= n1_max; n1++) {
                if (n1 > maxv[dv[1].second]) continue;
                cv[1].first = n1;
                const auto n2_max = n1_max / n1;
                for (int n2 = 1; n2 <= n2_max; n2++) {
                    if (n2 > maxv[dv[2].second]) continue;
                    cv[2].first = n2;
                    const auto n3_max = n2_max / n2;
                    for (int n3 = n3_max; n3 >= 1; n3--) {
                        if (n3 > maxv[dv[3].second]) continue;
                        cv[3].first = n3;
                        const auto tot_n = n0 * n1 * n2 * n3;
                        const auto cdiff = calc_diff(cv);
                        if (cdiff < best_diff && tot_n <= jcp.nthr) {
                            best_diff = cdiff;
                            nv = cv;
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < nd; i++) {
            const auto &nvf = nv[i].first;
            const auto &nvs = nv[i].second;
            if (nvs == bwd_w_dims::g)
                nthr_g = nvf;
            else if (nvs == bwd_w_dims::ic)
                nthr_ic_b = nvf;
            else if (nvs == bwd_w_dims::oc)
                nthr_oc_b = nvf;
            else if (nvs == bwd_w_dims::sp)
                nthr_mb = nvf;
        }
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    } else if (jcp.ngroups == 1 && (jcp.oc > 2048 || jcp.ic > 2048)) {
        const bool more_oc = (jcp.ic < jcp.oc);
        if (more_oc) {
            nthr_oc_b = div_up(jcp.nthr, 8);
            nthr_mb = div_up(jcp.nthr / nthr_oc_b, 2);
            nthr_ic_b = jcp.nthr / (nthr_mb * nthr_oc_b);
        } else {
            nthr_ic_b = div_up(jcp.nthr, 8);
            nthr_mb = div_up(jcp.nthr / nthr_ic_b, 2);
            nthr_oc_b = jcp.nthr / (nthr_mb * nthr_ic_b);
        }
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    } else if (jcp.kw > 100 && jcp.id == 1 && jcp.ih == 1) {
        nthr_g = static_cast<int>(nstl::min<dim_t>(jcp.nthr, jcp.ngroups));
        nthr_oc_b = static_cast<int>(
                nstl::min<dim_t>(jcp.nthr / nthr_g, div_up(jcp.nb_oc, 2)));
        nthr_ic_b = static_cast<int>(nstl::min<dim_t>(
                jcp.nthr / (nthr_g * nthr_oc_b), div_up(jcp.nb_ic, 2)));
        nthr_mb = jcp.nthr / (nthr_g * nthr_oc_b * nthr_ic_b);
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    }

    jcp.nthr = nthr;
    jcp.nthr_mb = nthr_mb;
    jcp.nthr_g = nthr_g;
    jcp.nthr_oc_b = nthr_oc_b;
    jcp.nthr_ic_b = nthr_ic_b;
}

status_t init_conf_bwd_w(jit_brgemm_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
        memory_desc_t &diff_dst_md, primitive_attr_t &attr, int nthreads) {

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);

    const bool is_f16 = src_d.data_type() == data_type::f16;

    const auto is_fp8 = one_of(src_d.data_type(), f8_e5m2, f8_e4m3)
            && one_of(diff_weights_d.data_type(), f32, f16, f8_e5m2, f8_e4m3)
            && one_of(diff_dst_d.data_type(), f8_e5m2, f8_e4m3);

    // The ACE kernels compute bf16 and int8 only, so every other datatype has
    // to keep its existing AMX flavor. This mirrors the jcp.is_ace condition in
    // init_jcp().
    const bool is_ace_dt = brgemm_utils::ace_dt_ok(
            src_d.data_type(), diff_dst_d.data_type());

    jcp.isa = (is_ace_dt && mayiuse(avx10_2_ace))
            ? avx10_2_ace
            : (is_fp8 ? (mayiuse(avx10_2_amx_2) ? avx10_2_amx_2
                                                : avx512_core_amx_fp16)
                      : (is_f16 ? avx512_core_amx_fp16 : avx512_core_amx));

    // disabling verbose dispatch messages for unsupported isa for better readability
    if (!mayiuse(jcp.isa)) return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    CHECK(init_jcp(jcp, jcp.isa, cd, src_md, diff_weights_md, diff_dst_md,
            diff_bias_md, attr, nthreads));

    jcp.max_batch = jcp.od * jcp.oh;
    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM
    jcp.use_uker = true;
    jcp.var_bs = true;

    // Process some 1x1 convolutions with small iw as 1d (h=1, w = h*w)
    // convolutions to make brgemm K dimension bigger for better utilization of
    // AMX tiles
    bool neat_1x1_2d = (everyone_is(
                                1, jcp.kh, jcp.kw, jcp.stride_h, jcp.stride_w)
            && everyone_is(0, jcp.t_pad, jcp.b_pad, jcp.l_pad, jcp.r_pad));
    bool make_1d = neat_1x1_2d && jcp.iw <= 28;
    if (make_1d) {
        jcp.iw *= jcp.ih;
        jcp.ih = 1;
        jcp.ow *= jcp.oh;
        jcp.oh = 1;
        jcp.max_batch = jcp.od;
    }
    // TODO: sometimes we can call brgemm kernel with bs = 0 to do initialization
    // review this condition
    if (jcp.max_batch == 1
            && everyone_is(0, jcp.f_pad, jcp.back_pad, jcp.t_pad, jcp.b_pad))
        jcp.var_bs = false;

    jcp.typesize_in = static_cast<int>(jcp.src_dsz);
    jcp.typesize_out = sizeof(float);

    bool ok = true
            // general condition to simplify dilations
            && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
            && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
            && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
            // special condition to simplify dilations in compute_oh_loop_common
            && IMPLICATION(jcp.dilate_h != 0, jcp.ext_kh <= jcp.ih);
    VDISPATCH_CONV_IC(ok, "dilation / stride values do not match");

    jcp.transform_to_vnni = diff_weights_d.data_type() != data_type::f32;

    /* XXX: no support for padding when dilation_d > 0 */
    VDISPATCH_CONV_IC(IMPLICATION(jcp.dilate_d > 0,
                              everyone_is(0, jcp.back_pad, jcp.f_pad)),
            VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "no support for padding when dilation_d > 0");

    const bool is_depthwise = true && with_groups && jcp.ngroups > 1
            && everyone_is(1, jcp.ic, jcp.oc);
    // TODO: add support of DW convolution
    VDISPATCH_CONV_IC(!is_depthwise, VERBOSE_UNSUPPORTED_FEATURE,
            "no support for depthwise convolution");

    const int dat_format_tag = ndims - 3;
    format_tag_t dat_tag_nspc = utils::pick(dat_format_tag, format_tag::nwc,
            format_tag::nhwc, format_tag::ndhwc);
    format_tag_t dat_tag_opt = dat_tag_nspc;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag_opt));
        jcp.src_tag = dat_tag_opt;
    } else
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag_opt);
    VDISPATCH_CONV_IC(
            one_of(jcp.src_tag, dat_tag_opt), VERBOSE_UNSUPPORTED_TAG_S, "src");

    const bool is_nspc = jcp.src_tag == dat_tag_nspc;
    VDISPATCH_CONV_IC(is_nspc, VERBOSE_UNSUPPORTED_TAG_S, "src");

    if (diff_dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, jcp.src_tag));
        jcp.dst_tag = jcp.src_tag;
    } else
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(jcp.src_tag);
    VDISPATCH_CONV_IC(jcp.dst_tag == jcp.src_tag, VERBOSE_UNSUPPORTED_TAG);

    jcp.wei_dt = diff_weights_d.data_type();

    const int wei_format_tag = 2 * ndims - 6 + with_groups;
    format_tag_t wei_tag;
    if (jcp.transform_to_vnni) {
        if (one_of(jcp.wei_dt, f8_e5m2, f8_e4m3))
            wei_tag = pick(wei_format_tag, format_tag::OIw16i16o4i,
                    format_tag::gOIw16i16o4i, format_tag::OIhw16i16o4i,
                    format_tag::gOIhw16i16o4i, format_tag::OIdhw16i16o4i,
                    format_tag::gOIdhw16i16o4i);
        else
            wei_tag = pick(wei_format_tag, format_tag::OIw16i16o2i,
                    format_tag::gOIw16i16o2i, format_tag::OIhw16i16o2i,
                    format_tag::gOIhw16i16o2i, format_tag::OIdhw16i16o2i,
                    format_tag::gOIdhw16i16o2i);
    } else {
        wei_tag = pick(wei_format_tag, format_tag::OIw16i16o,
                format_tag::gOIw16i16o, format_tag::OIhw16i16o,
                format_tag::gOIhw16i16o, format_tag::OIdhw16i16o,
                format_tag::gOIdhw16i16o);
    }
    if (diff_weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
        VDISPATCH_CONV_IC(
                jcp.wei_tag == wei_tag, VERBOSE_UNSUPPORTED_TAG_S, "weights");
    }

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const dim_t max_pad_h = jcp.ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < jcp.ext_kw
            && jcp.r_pad < jcp.ext_kw && jcp.t_pad <= max_pad_h
            && jcp.b_pad <= max_pad_h && jcp.f_pad < jcp.ext_kd
            && jcp.back_pad < jcp.ext_kd;
    VDISPATCH_CONV_IC(boundaries_ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "padding size unsupported (overflow)");

    jcp.ic_block = 16;
    jcp.oc_block = 16;

    jcp.nb_ic = utils::div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = utils::div_up(jcp.oc, jcp.oc_block);

    jcp.ic_tail = static_cast<int>(jcp.ic % jcp.ic_block);
    jcp.oc_tail = static_cast<int>(jcp.oc % jcp.oc_block);

    jcp.nb_oc_blocking = (jcp.nb_oc > 1) ? 2 : 1;
    jcp.nb_ic_blocking = (jcp.nb_ic > 1) ? 2 : 1;

    const bool is_2d = (ndims == 4);
    const bool is_3d = (ndims == 5);

    // TODO: Find more shapes (especially 3D with large spatials) for which
    // local transposition will be beneficial. Furthermore, for TBB threads
    // more shapes can potentially benefit from spatial blocking
    dim_t optimal_blk_size = is_3d ? jcp.od : is_2d ? jcp.oh : jcp.ow;

    jcp.global_transpose = dnnl_thr_syncable();
    jcp.spatial_blk_size = optimal_blk_size;

    const int tr_round = 32; // To load full tile register
    dim_t tr_pad = rnd_up(nstl::max(jcp.l_pad, jcp.r_pad + 1), tr_round);
    jcp.tr_iw = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad, jcp.stride_w),
                        tr_round)
            * jcp.stride_w;

    // TODO: xf16 or fp8 training is supported only
    const auto rnd_val = jcp.vnni_block;
    jcp.tr_src_num_guard_elems = tr_pad; // upper bound
    jcp.tr_ow = rnd_up(jcp.ow, rnd_val);
    if (jcp.tr_ow > tr_round) {
        // we may increase tr_ow to have better bd_block in brgemm kernel
        dim_t best_bdb = jcp.tr_ow / rnd_val;
        dim_t best_tr_ow = jcp.tr_ow;
        for (dim_t tr_ow = jcp.tr_ow; tr_ow <= rnd_up(jcp.tr_ow, tr_round);
                tr_ow += rnd_val) {
            for (int i = tr_round; i > 0; i -= rnd_val) {
                if (tr_ow % i == 0) {
                    const auto cbdb = tr_ow / i;
                    if (cbdb < best_bdb) {
                        best_bdb = cbdb;
                        best_tr_ow = tr_ow;
                    }
                    break;
                }
            }
        }
        jcp.tr_ow = best_tr_ow;
    }

    bool args_ok = true && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_TAG);

    jcp.harness = ndims == 5 ? harness_3d_reduction : harness_2d_reduction;

    if (!one_of(jcp.harness, harness_2d_reduction, harness_3d_reduction)) {
        return status::unimplemented;
    }

    switch (jcp.harness) {
        case harness_2d_reduction:
            jcp.nthr_mb_work = static_cast<int>(jcp.mb * jcp.oh);
            break;
        case harness_3d_reduction:
            jcp.nthr_mb_work = static_cast<int>(jcp.mb * jcp.od);
            break;
        default:
            assert(!"Invalid harness");
            jcp.nthr_mb_work = static_cast<int>(jcp.mb);
    }

    balance_bwd_w(jcp);

    if (one_of(jcp.harness, harness_2d_reduction, harness_3d_reduction)) {
        jcp.K = jcp.tr_ow;
    }
    jcp.K_tail = 0;

    jcp.M = jcp.ic <= 16 ? jcp.ic : jcp.ic_block * jcp.nb_ic_blocking;
    // assumption that jcp.nb_ic_blocking is always 2
    if (jcp.nb_ic % jcp.nthr_ic_b == 0
            && (jcp.nb_ic / jcp.nthr_ic_b) % jcp.nb_ic_blocking == 0)
        jcp.M_tail = 0;
    else
        jcp.M_tail = jcp.ic_block;

    jcp.N = jcp.oc_block * jcp.nb_oc_blocking;
    // assumption that jcp.nb_oc_blocking is always 2
    if (jcp.nb_oc % jcp.nthr_oc_b == 0
            && (jcp.nb_oc / jcp.nthr_oc_b) % jcp.nb_oc_blocking == 0)
        jcp.N_tail = 0;
    else
        jcp.N_tail = jcp.oc_block;

    // for convolutions with big spatial: transpose only chunk
    // (oc_block * nb_oc_blocking) of diff_dst on each iteration by oc blocks
    // for better cache utilization
    // the equal number of channel blocks per thread is required to use this
    // approach to avoid hangs
    bool tr_ocb_chunk_allowed = (jcp.nb_oc % jcp.nthr_oc_b == 0);
    jcp.tr_ocb_chunk = tr_ocb_chunk_allowed && (jcp.oh * jcp.ow > 38 * 38);
    jcp.tr_icb_chunk = false;

    const dim_t irow_size = static_cast<dim_t>(jcp.src_dsz) * jcp.tr_iw
            * jcp.ic_block * div_up(jcp.nb_ic, jcp.nthr_ic_b)
            * 2 /*we have real and transposed input */;
    const dim_t orow_size = static_cast<dim_t>(jcp.dst_dsz) * jcp.tr_ow
            * jcp.oc_block * div_up(jcp.nb_oc, jcp.nthr_oc_b)
            * 2 /*we have real and transposed diff_dst*/;
    dim_t oh_block_limit = static_cast<dim_t>(nstl::max(1.f,
            nstl::max(0.f,
                    0.8f * static_cast<float>(brg_blocking_t::L2)
                            - static_cast<float>(jcp.kh)
                                    * static_cast<float>(irow_size))
                    / static_cast<float>(irow_size + orow_size)));
    // try to split oh by equal oh blocks
    oh_block_limit = div_up(jcp.oh, div_up(jcp.oh, oh_block_limit));
    jcp.oh_block = utils::saturate<dim_t>(1, jcp.oh, oh_block_limit);
    jcp.ih_block = nstl::min<dim_t>(jcp.ih,
            jcp.stride_h
                    * brg_blocking_t::get_inp_size(jcp.ih, jcp.oh_block, jcp.kh,
                            jcp.stride_h, jcp.dilate_h));

    // try to find tr_ic_block to have enough src transpose work to distribute
    // among nthr_oc_b
    jcp.tr_ic_block = jcp.ic_block;
    if (jcp.ic <= jcp.ic_block) {
        for (dim_t itr_icb = jcp.ic_block; itr_icb > 1; itr_icb--) {
            if (jcp.ic_block % itr_icb != 0) continue;
            const auto icb_per_thr_ic_b = div_up(jcp.nb_ic, jcp.nthr_ic_b);
            const auto ic_per_thr_ic_b
                    = nstl::min<dim_t>(jcp.ic, icb_per_thr_ic_b * jcp.ic_block);
            const auto ic_block_per_thr_ic_b
                    = nstl::min<dim_t>(jcp.ic, jcp.ic_block);
            if (ic_block_per_thr_ic_b % itr_icb != 0) continue;
            const auto tr_icb_per_thr = div_up(ic_per_thr_ic_b, itr_icb);
            const auto sp_per_thr_mb
                    = div_up(jcp.id * jcp.ih_block, jcp.nthr_mb);
            if (tr_icb_per_thr * sp_per_thr_mb < jcp.nthr_oc_b)
                jcp.tr_ic_block = itr_icb;
        }
    }

    jcp.nb_tr_ic = utils::div_up(jcp.ic, jcp.tr_ic_block);
    jcp.tr_ic_tail = jcp.ic % jcp.tr_ic_block;

    // TODO: Optimize memory allocation when threaded on height and depth
    jcp.tr_src_buf_count = jcp.global_transpose
            ? jcp.nthr_mb * jcp.nb_ic * jcp.ngroups
            : jcp.nthr;
    jcp.tr_diff_dst_buf_count = jcp.global_transpose
            ? jcp.nthr_mb * jcp.nb_oc * jcp.ngroups
            : jcp.nthr;
    jcp.tr_src_block_size = static_cast<size_t>(jcp.tr_iw) * jcp.ic_block
            * jcp.ih_block * jcp.id;
    jcp.tr_diff_dst_block_size = static_cast<size_t>(jcp.tr_ow) * jcp.oc_block
            * jcp.oh_block * jcp.od;

    jcp.tr_src_buf_size = jcp.tr_src_block_size
            * (jcp.global_transpose ? 1 : jcp.nb_ic_blocking);
    jcp.tr_diff_dst_buf_size = jcp.tr_diff_dst_block_size
            * (jcp.global_transpose ? 1 : jcp.nb_oc_blocking);

    const dim_t iframe_size = irow_size * jcp.id;
    const dim_t oframe_size = orow_size * jcp.od;
    dim_t od_block_limit = static_cast<dim_t>(nstl::max(1.f,
            nstl::max(0.f,
                    0.8f * static_cast<float>(brg_blocking_t::L2)
                            - static_cast<float>(jcp.kd)
                                    * static_cast<float>(iframe_size))
                    / static_cast<float>(iframe_size + oframe_size)));
    // try to split od by equal od blocks
    od_block_limit = div_up(jcp.od, div_up(jcp.od, od_block_limit));
    jcp.od_block = utils::saturate<dim_t>(1, jcp.od, od_block_limit);

    jcp.use_interleave_stores = false;
    jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
    jcp.amx_tile_load_xx = false;

    if (one_of(jcp.harness, harness_2d_reduction, harness_3d_reduction)) {
        jcp.LDA = jcp.tr_iw;
        jcp.LDB = jcp.oc_block;
        jcp.LDC = jcp.LDD = jcp.oc_block;
    }

    jcp.gemm_batch_size = jcp.max_batch;
    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, P4K), sc_size);

    return status::success;
}

status_t init_scratchpad_bwd_w(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    // XXX: See the comment about tr_iw and guarding elements in
    // jit_avx512_core_amx_bwd_weights_kernel_t::init_conf()
    const size_t tr_src_size = jcp.tr_src_buf_count * jcp.tr_src_buf_size
            + jcp.tr_src_num_guard_elems;
    scratchpad.book(key_conv_tr_src, tr_src_size, jcp.src_dsz);

    /* prepare synchronization contexts */
    if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
        const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx, tr_src_bctx_size);
    }

    // The tr_ow <= tr_iw, so we need some guarding at the end of diff_dst
    // TODO: update this guarding:
    // (jcp.tr_diff_dst_buf_size + jcp.tr_iw * jcp.oc_block)
    const auto tr_diff_dst_size
            = jcp.tr_diff_dst_buf_count * jcp.tr_diff_dst_buf_size
            + static_cast<size_t>(jcp.tr_iw) * jcp.oc_block;

    const size_t min_align = 64;
    scratchpad.book(
            key_conv_tr_diff_dst, tr_diff_dst_size, jcp.src_dsz, min_align);

    /* prepare synchronization contexts */
    if (jcp.global_transpose && jcp.nthr_ic_b > 1) {
        const size_t tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_tr_diff_dst_bctx, tr_diff_dst_bctx_size);
    }

    if (IMPLICATION(jcp.nthr_mb == 1,
                (jcp.with_bias && jcp.bia_dt != data_type::f32)
                        || jcp.wei_dt != data_type::f32)) {
        const size_t wei_size = static_cast<size_t>(jcp.ngroups) * jcp.nb_oc
                * jcp.oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw
                * jcp.kd;
        const size_t bia_size
                = jcp.with_bias * jcp.ngroups * jcp.nb_oc * jcp.oc_block;

        const int num_wei_buffers
                = jcp.wei_dt != data_type::f32 ? jcp.nthr_mb : jcp.nthr_mb - 1;
        const int num_bia_buffers = jcp.with_bias
                ? (jcp.bia_dt != data_type::f32 ? jcp.nthr_mb : jcp.nthr_mb - 1)
                : 0;

        const size_t wei_bia_reduction_size
                = wei_size * num_wei_buffers + bia_size * num_bia_buffers;

        scratchpad.book<float>(
                key_conv_wei_bia_reduction, wei_bia_reduction_size);

        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);
    }

    if (jcp.with_bias
            && ((jcp.oc % jcp.oc_block != 0) && jcp.bia_dt == data_type::f32)) {
        scratchpad.book(key_conv_padded_bias,
                jcp.ngroups * jcp.nb_oc * jcp.oc_block, jcp.bia_dsz);
    }
    scratchpad.book(key_conv_amx_tilecfg, 1, 64); // 1 whole cacheline

    constexpr size_t scratchpad_limit_by_absolute_value = (size_t)32
            << 30; // 32Gb - TODO: may it's too large?
    const size_t scratchpad_limit_by_tensor_sizes = (size_t)64 * jcp.nthr
            * (src_d.size() + diff_weights_d.size() + diff_dst_d.size());
    const size_t scratchpad_limit
            = nstl::min(scratchpad_limit_by_absolute_value,
                    scratchpad_limit_by_tensor_sizes);

    scratchpad.book(key_brgemm_primitive_batch,
            static_cast<size_t>(jcp.nthr) * jcp.adjusted_batch_size,
            sizeof(brgemm_batch_element_t), 64, P4K);

    scratchpad.book(
            key_conv_amx_tile_buffer, jcp.nthr * 2 * P4K, sizeof(char), 0, P4K);

    VDISPATCH_CONV_IC(
            scratchpad.size() <= scratchpad_limit, VERBOSE_SCRATCHPAD_LIMIT);
    return status::success;
}

// Sets `ow_s` and `ow_f` values based on given `jcp`.
void get_ow_range(const jit_brgemm_conv_conf_t &jcp, int ow, int kw, int &ow_s,
        int &ow_f) {
    // This function is used for exec_base only

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;

    const auto IW = jcp.iw;
    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;

    const auto iiw = ow * SW - LP;
    auto iw_lp = iiw + kw * DW;
    const auto iw_rp = iw_lp + (M - 1) * SW - IW + 1;
    ow_s = ow;

    int ker_idx = 0;
    if (iw_lp < 0) {
        iw_lp = nstl::abs(iw_lp);
        ker_idx += static_cast<int>(div_up(iw_lp, SW));
        ow_s += ker_idx;
    }
    if (iw_rp > 0) ker_idx += static_cast<int>(div_up(iw_rp, SW));
    ow_f = static_cast<int>(nstl::max<dim_t>(ow_s, ow_s + (M - ker_idx)));

    ow_s = static_cast<int>(nstl::min<dim_t>(ow_s, ow + M));
    ow_f = static_cast<int>(nstl::min<dim_t>(ow_f, ow + M));
}

// Sets pairs of `kw_s`-`kw_f` and `kw_full_s`-`kw_full_f` based on given `jcp`.
void get_kw_range(const jit_brgemm_conv_conf_t &jcp, int ow, int &kw_s,
        int &kw_full_s, int &kw_full_f, int &kw_f) {
    // This function is used for exec_base only
    // TODO: calculate these values instead direct loop by kw

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;
    kw_s = kw_full_s = kw_full_f = kw_f = -1;
    for (int kw = 0; kw < jcp.kw; kw++) {
        int ow_s {0}, ow_f {0};
        brgemm_convolution_utils::get_ow_range(jcp, ow, kw, ow_s, ow_f);
        if (ow_s < ow_f) {
            if (kw_s == -1) kw_s = kw;
            kw_f = kw + 1;
            if (ow_f - ow_s == M) {
                if (kw_full_s == -1) kw_full_s = kw;
                kw_full_f = kw + 1;
            }
        }
    }
    if (kw_f == -1) {
        kw_s = 0;
        kw_f = 0;
    }
    if (kw_full_f == -1) kw_full_s = kw_full_f = kw_f;
}

} // namespace brgemm_convolution_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
