/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024-2026 FUJITSU LIMITED
* Copyright 2024-2026 Arm Ltd. and affiliates
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
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_brgemm_conv_mmla_pack_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_conv_utils.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

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
    return true;
}
} // namespace

namespace brgemm_convolution_utils {

bool is_any_eligible(const jit_brgemm_conv_conf_t &jcp) {
    return (jcp.prop_kind == prop_kind::forward_inference || jcp.wei_plain
            || one_of(jcp.wei_dt, data_type::s8, data_type::f16)
            || one_of(jcp.isa, sve_512, sve_256, sve_128));
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
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

bool uses_batch_elements(
        brgemm_batch_kind_t brg_type, conv_brgemm_exec_type_t exec_type) {
    // Batch elements are required for all batch kinds except fixed strides.
    // Batch elements are also required for virtual padding.
    return IMPLICATION(brg_type == brgemm_strd, exec_type == exec_vpad);
}

int mmla_bd_block(int oc_block, int M) {
    constexpr int max_acc_regs = 24;
    constexpr int ld_block = 8; // s32 lanes per vector on sve_256
    const int ld_block2 = div_up(oc_block, ld_block);
    // one register pair (2 rows) per ld block; at most 12 rows per bd block
    const int max_bd = nstl::min(12, 2 * (max_acc_regs / (2 * ld_block2)));
    return nstl::max(1, nstl::min(M, max_bd));
}

void set_mmla_brgattr(
        const jit_brgemm_conv_conf_t &jcp, int M, brgemm_attr_t &brgattr) {
    brgattr.use_mmla = true;
    brgattr.use_mmla_packed_a = true;
    // weights are stored in oc_block-wide panels; the N-tail kernels must use
    // the same panel stride, so the panel width is always passed as the hint
    brgattr.hint_ld_block2 = div_up(jcp.oc_block, jcp.acc_simd_w);
    brgattr.hint_bd_block = mmla_bd_block(jcp.oc_block, M);
    brgattr.max_top_vpad = 0;
    brgattr.max_bottom_vpad = 0;
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
                    broadcasting_strategy_t::no_broadcast}));
}

bool is_groups_ok(jit_brgemm_conv_conf_t &jcp) {
    // Enable grouped convs for the shapes not supported in direct convs
    // direct approach only supports int8/bf16 grouped conv
    // when channels per groups is at least multiple of 4
    // and bf16 grouped conv with layout nxc on jit_bf16 impl
    // TODO: remove this condition after the restriction on small ic is removed
    return jcp.ngroups > 1
            && IMPLICATION(one_of(jcp.src_dt, u8, s8, bf16),
                    jcp.ic % 4 == 0 && jcp.oc % 4 == 0);
}

static bool int8_large_spatial_ok(
        const jit_brgemm_conv_conf_t &jcp, int oc_block, int ow_block) {
    return IMPLICATION(jcp.ic <= 128,
                   jcp.od * jcp.oh < 100 || jcp.ic * oc_block * ow_block > 8192)
            && !(jcp.oc == 1024
                    && utils::everyone_is(1, jcp.od, jcp.oh, jcp.kd, jcp.kh)
                    && jcp.ow >= 595 && jcp.kw <= 5);
}

static constexpr int mmla_max_oc_block = 24;

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

    const int vnni_granularity
            = jcp.use_mmla ? 8 : data_type_vnni_granularity(jcp.wei_dt);

    const bool is_1d = jcp.ndims == 3;
    const bool is_2d = jcp.ndims == 4;
    const bool is_3d = jcp.ndims == 5;

    if (jcp.wei_plain) {
        jcp.LDB = jcp.oc_without_padding;
        if (is_3d) {
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? dhwigo : dhwio; break;
                case 2: wei_tag = with_groups ? gdhwIo2i : dhwIo2i; break;
                case 4: wei_tag = with_groups ? gdhwIo4i : dhwIo4i; break;
                default: return status::unimplemented;
            }
        } else if (is_1d) {
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? wigo : wio; break;
                case 2: wei_tag = with_groups ? gwIo2i : wIo2i; break;
                case 4: wei_tag = with_groups ? gwIo4i : wIo4i; break;
                default: return status::unimplemented;
            }
        } else {
            assert(is_2d);
            UNUSED(is_2d);
            switch (vnni_granularity) {
                case 1: wei_tag = with_groups ? hwigo : hwio; break;
                case 2: wei_tag = with_groups ? ghwIo2i : hwIo2i; break;
                case 4: wei_tag = with_groups ? ghwIo4i : hwIo4i; break;
                default: return status::unimplemented;
            }
        }
    } else {
        jcp.LDB = jcp.oc_block;
        if (jcp.oc_block == 64) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi64o : Odhwi64o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i64o2i
                                                  : OdhwI16i64o2i;
                        else
                            wei_tag = with_groups ? gOdhwI64o2i : OdhwI64o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i64o4i
                                                  : OdhwI16i64o4i;
                        else
                            wei_tag = with_groups ? gOdhwI64o4i : OdhwI64o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi64o : Owi64o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i64o2i : OwI16i64o2i;
                        else
                            wei_tag = with_groups ? gOwI64o2i : OwI64o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i64o4i : OwI16i64o4i;
                        else
                            wei_tag = with_groups ? gOwI64o4i : OwI64o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi64o : Ohwi64o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i64o2i
                                                  : OhwI16i64o2i;
                        else
                            wei_tag = with_groups ? gOhwI64o2i : OhwI64o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i64o4i
                                                  : OhwI16i64o4i;
                        else
                            wei_tag = with_groups ? gOhwI64o4i : OhwI64o4i;
                        break;
                    default: return status::unimplemented;
                }
            }
        } else if (jcp.oc_block == 48) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi48o : Odhwi48o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i48o2i
                                                  : OdhwI16i48o2i;
                        else
                            wei_tag = with_groups ? gOdhwI48o2i : OdhwI48o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i48o4i
                                                  : OdhwI16i48o4i;
                        else
                            wei_tag = with_groups ? gOdhwI48o4i : OdhwI48o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi48o : Owi48o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i48o2i : OwI16i48o2i;
                        else
                            wei_tag = with_groups ? gOwI48o2i : OwI48o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i48o4i : OwI16i48o4i;
                        else
                            wei_tag = with_groups ? gOwI48o4i : OwI48o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi48o : Ohwi48o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i48o2i
                                                  : OhwI16i48o2i;
                        else
                            wei_tag = with_groups ? gOhwI48o2i : OhwI48o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i48o4i
                                                  : OhwI16i48o4i;
                        else
                            wei_tag = with_groups ? gOhwI48o4i : OhwI48o4i;
                        break;
                    default: return status::unimplemented;
                }
            }
        } else if (jcp.oc_block == 32) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi32o : Odhwi32o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i32o2i
                                                  : OdhwI16i32o2i;
                        else
                            wei_tag = with_groups ? gOdhwI32o2i : OdhwI32o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i32o4i
                                                  : OdhwI16i32o4i;
                        else
                            wei_tag = with_groups ? gOdhwI32o4i : OdhwI32o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi32o : Owi32o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i32o2i : OwI16i32o2i;
                        else
                            wei_tag = with_groups ? gOwI32o2i : OwI32o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i32o4i : OwI16i32o4i;
                        else
                            wei_tag = with_groups ? gOwI32o4i : OwI32o4i;
                        break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi32o : Ohwi32o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i32o2i
                                                  : OhwI16i32o2i;
                        else
                            wei_tag = with_groups ? gOhwI32o2i : OhwI32o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i32o4i
                                                  : OhwI16i32o4i;
                        else
                            wei_tag = with_groups ? gOhwI32o4i : OhwI32o4i;
                        break;
                    default: return status::unimplemented;
                }
            }
        } else if (jcp.oc_block == 24) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi24o : Odhwi24o; break;
                    case 2:
                        wei_tag = with_groups ? gOdhwI24o2i : OdhwI24o2i;
                        break;
                    case 4:
                        wei_tag = with_groups ? gOdhwI24o4i : OdhwI24o4i;
                        break;
                    case 8:
                        wei_tag = with_groups ? gOdhwI24o8i : OdhwI24o8i;
                        break;
                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi24o : Owi24o; break;
                    case 2: wei_tag = with_groups ? gOwI24o2i : OwI24o2i; break;
                    case 4: wei_tag = with_groups ? gOwI24o4i : OwI24o4i; break;
                    case 8: wei_tag = with_groups ? gOwI24o8i : OwI24o8i; break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi24o : Ohwi24o; break;
                    case 2:
                        wei_tag = with_groups ? gOhwI24o2i : OhwI24o2i;
                        break;
                    case 4:
                        wei_tag = with_groups ? gOhwI24o4i : OhwI24o4i;
                        break;
                    case 8:
                        wei_tag = with_groups ? gOhwI24o8i : OhwI24o8i;
                        break;
                    default: return status::unimplemented;
                }
            }
        } else if (jcp.oc_block == 16) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi16o : Odhwi16o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i16o2i
                                                  : OdhwI16i16o2i;
                        else
                            wei_tag = with_groups ? gOdhwI16o2i : OdhwI16o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOdhwI16i16o4i
                                                  : OdhwI16i16o4i;
                        else
                            wei_tag = with_groups ? gOdhwI16o4i : OdhwI16o4i;
                        break;
                    case 8:
                        assert(!jcp.is_ic_padded);
                        wei_tag = with_groups ? gOdhwI16o8i : OdhwI16o8i;
                        break;

                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi16o : Owi16o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i16o2i : OwI16i16o2i;
                        else
                            wei_tag = with_groups ? gOwI16o2i : OwI16o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOwI16i16o4i : OwI16i16o4i;
                        else
                            wei_tag = with_groups ? gOwI16o4i : OwI16o4i;
                        break;
                    case 8:
                        assert(!jcp.is_ic_padded);
                        wei_tag = with_groups ? gOwI16o8i : OwI16o8i;
                        break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi16o : Ohwi16o; break;
                    case 2:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i16o2i
                                                  : OhwI16i16o2i;
                        else
                            wei_tag = with_groups ? gOhwI16o2i : OhwI16o2i;
                        break;
                    case 4:
                        if (jcp.is_ic_padded)
                            wei_tag = with_groups ? gOhwI16i16o4i
                                                  : OhwI16i16o4i;
                        else
                            wei_tag = with_groups ? gOhwI16o4i : OhwI16o4i;
                        break;
                    case 8:
                        assert(!jcp.is_ic_padded);
                        wei_tag = with_groups ? gOhwI16o8i : OhwI16o8i;
                        break;
                    default: return status::unimplemented;
                }
            }
        } else if (jcp.oc_block == 8) {
            if (is_3d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOdhwi8o : Odhwi8o; break;
                    case 2:
                        wei_tag = with_groups ? gOdhwI8o2i : OdhwI8o2i;
                        break;
                    case 4:
                        wei_tag = with_groups ? gOdhwI8o4i : OdhwI8o4i;
                        break;
                    case 8:
                        wei_tag = with_groups ? gOdhwI8o8i : OdhwI8o8i;
                        break;
                    default: return status::unimplemented;
                }
            } else if (is_1d) {
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOwi8o : Owi8o; break;
                    case 2: wei_tag = with_groups ? gOwI8o2i : OwI8o2i; break;
                    case 4: wei_tag = with_groups ? gOwI8o4i : OwI8o4i; break;
                    case 8: wei_tag = with_groups ? gOwI8o8i : OwI8o8i; break;
                    default: return status::unimplemented;
                }
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                switch (vnni_granularity) {
                    case 1: wei_tag = with_groups ? gOhwi8o : Ohwi8o; break;
                    case 2: wei_tag = with_groups ? gOhwI8o2i : OhwI8o2i; break;
                    case 4: wei_tag = with_groups ? gOhwI8o4i : OhwI8o4i; break;
                    case 8: wei_tag = with_groups ? gOhwI8o8i : OhwI8o8i; break;
                    default: return status::unimplemented;
                }
            }
        } else {
            return status::unimplemented;
        }
    }

    src_tag = dst_tag;

    const bool any_eligible = is_any_eligible(jcp);
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));
    CHECK(init_tag(jcp.dst_tag, dst_md, dst_d, dst_tag, any_eligible));
    CHECK(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag, true));

    return status::success;
}

struct brg_blocking_t : public jit_brgemm_conv_conf_t {
    struct array_in_loop_t {
        dim_t itersize;
        float repeatn;
        float overlap;
        void set(dim_t iter_s, float rpt, float ovlp = 1.f) {
            itersize = iter_s;
            repeatn = rpt;
            overlap = ovlp;
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
        mmla_choice_fixed = false;
        eff = 0.f;
        nb_kd = 0;
        nb_kh = 0;
        nb_kw = 0;
        sp = 0;
        sp_block = 0;
        nb_sp = 0;
        eff = 0;
        max_regs = isa_num_vregs(isa);
        bcast_simd = acc_simd_w;
    }

    int ur, ur_block, ur_block_tail;
    int nb_kd, nb_kh, nb_kw;
    bool mmla_choice_fixed;
    int max_regs;
    int bcast_simd;
    float eff;
    static unsigned L1;
    static unsigned L2;
    static unsigned L3;
    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // TODO: Improve memory access estimates
    static constexpr float L1_k = 1.f;
    static constexpr float L2_k = 3.f;
    static constexpr float L3_k = 15.f;
    // TODO: At the moment, we are primarily evaluating the fit of the data into
    // the L1/L2. Need to take into account the difference between the L3 and
    // memory.
    static constexpr float mem_k = 15.f;
    static constexpr int bench_iterations = 1;

    int sp, sp_block, nb_sp;
    static thread_local int last_ic_block_size;

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
    void iterate_ker_block(brg_blocking_t &best_brgb, int kd_block,
            int kh_block, bool maybe_use_buffer, int max_ow_block_thr);
    bool mmla_pack_fits() const;
    bool mmla_threads_ok() const;
    status_t calc_blocks();

    bool fast_check_oc_block_1x1() const;
    float est_eff_1x1();
    void calc_blocks_1x1();

    // utils
    static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        auto adj_str = nstl::min(k, stride);
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, adj_str,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    static float squeeze_val(float eff, float koeff) {
        if (koeff <= 0) return 1;
        if (koeff == 1) return eff;
        const auto k = 1.f / koeff;
        return (k > 1.f) ? (k - 1 + eff) / k : eff * koeff;
    }

    static int estimate_ur(int oc_block) {
        const auto est_ur = (oc_block == 64)
                ? 6
                : ((oc_block == 48) ? 9 : ((oc_block == 32) ? 14 : 28));
        return est_ur;
    }

    int inp_w(int out_w, int ker_w) const {
        return get_inp_size(iw, out_w, ker_w, stride_w, dilate_w);
    }

    int rnd_simd(int val) const { return rnd_up(val, simd_w); }

    int rnd_inp_simd(int out_w, int ker_w, int vic) const {
        const auto vsp = inp_w(out_w, ker_w);
        return ((stride_w == 1 && vic >= ic) ? rnd_up(vsp * vic, simd_w)
                                             : vsp * rnd_up(vic, simd_w));
    }

    static constexpr int MAXNLOOPS = 32;
    loop_t loop[MAXNLOOPS];
};

unsigned brg_blocking_t::L1;
unsigned brg_blocking_t::L2;
unsigned brg_blocking_t::L3;
thread_local int brg_blocking_t::last_ic_block_size;

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
    auto nb_simd = utils::div_up(ic, simd_w);
    auto max_simd_blocks = nstl::min(5 * simd_w, nb_simd);
    const auto nb_icb_eff_threshold = 0.5f;
    const auto padded_ic = last_ic_block_size * (is_ic_padded ? acc_simd_w : 1);

    const auto est_ur = sp_block > 0
            ? nstl::min(sp_block, estimate_ur(oc_block))
            : estimate_ur(oc_block);
    const auto inp_ur = is_os_blocking ? est_ur : inp_w(est_ur, kw_block);

    if (kw_block > 1) {
        // try to fit src into L1
        const auto inp_per_ic = static_cast<unsigned int>(inp_ur) * src_dsz;
        max_simd_blocks = saturate(1, max_simd_blocks,
                static_cast<int>(L1 / (inp_per_ic * simd_w)));
    }
    // try to fit all batch for ur into L2
    const bool adjust = wei_plain && math::is_pow2(oc)
            && utils::everyone_is(1, kd_block, kh_block, kw_block);
    const int adj_oc_block = adjust ? oc : oc_block; // due to aliasing
    const auto wei_per_ic = static_cast<unsigned int>(kd_block) * kh_block
            * kw_block * adj_oc_block * wei_dsz;
    const auto inp_per_ic
            = static_cast<unsigned int>(kd_block) * kh_block * inp_ur * src_dsz;
    const auto out_size = static_cast<unsigned int>(ur) * oc_block * dst_dsz;

    max_simd_blocks = saturate(1, max_simd_blocks,
            static_cast<int>(
                    (L2 - out_size) / ((wei_per_ic + inp_per_ic) * simd_w)));

    auto simd_blocks = 1;
    for (int nb_icb = nstl::min(max_simd_blocks, nb_simd); nb_icb >= 1;
            nb_icb--) {
        auto nb_icb_eff = static_cast<float>(nb_simd) / rnd_up(nb_simd, nb_icb);
        if (nb_icb_eff >= nb_icb_eff_threshold) {
            simd_blocks = nb_icb;
            break;
        }
    }

    ic_block = nstl::min((exec_type == exec_trans) ? rnd_up(ic, padded_ic) : ic,
            simd_blocks * simd_w);

    nb_ic = utils::div_up(ic, ic_block);
}

status_t brg_blocking_t::estimate_brgemm_ur() {
    // Simple simulation of brgemm_desc init
    if (sp_block <= 0) return status::invalid_arguments;
    LDA = is_rtus
            ? (ic_block)
            : (kh_sets > 1 ? kh_sets : 1) * (kw_sets > 1 ? kw_sets : stride_w)
                    * (exec_type == exec_trans ? ic_block
                                               : ngroups * ic_without_padding);
    LDB = wei_plain ? oc_without_padding : oc_block;
    LDC = use_buffer ? oc_block : oc_without_padding;

    const auto padded_ic = last_ic_block_size * (is_ic_padded ? acc_simd_w : 1);

    icp = rnd_up(ic, padded_ic);
    M = brgM = sp >= sp_block ? sp_block : 0;
    M_tail = brgM_tail = sp % sp_block;
    if (is_os_blocking) {
        if (!is_1x1) M_tail = (oh * ow) % sp_block;
        oskip = ((ext_kw - 1) / stride_w) * stride_h + (stride_h - 1) * ow;

        brgM = M + oskip * (div_up(M, ow) - 1);
        brgM_tail = M_tail + oskip * div_up(M_tail, ow);
    }

    N = oc >= oc_block ? oc_block : 0;
    N_tail = oc % oc_block;

    K = kh_sets * kw_sets * (ic >= ic_block ? ic_block : 0);
    K_tail = kh_sets * kw_sets
            * (exec_type == exec_trans && (!is_bf32)
                            ? ic_block
                            : rnd_up(ic % ic_block, last_ic_block_size));

    const auto vK = K > 0 ? K : K_tail;
    const auto vM = M > 0 ? M : M_tail;
    const auto vN = N > 0 ? N : N_tail;

    const float alpha = 1.0;
    const float beta = 0.0;
    brgemm_desc_t brg;
    CHECK(brgemm_utils::init_brgemm_conf(&brg, isa, brgemm_addr, src_dt, wei_dt,
            brgemm_row_major, alpha, beta, LDA, LDB, LDC, vM, vN, vK, nullptr,
            is_bf32));
    CHECK(brgemm_utils::brgemm_blocking(&brg));
    ur = use_mmla ? mmla_bd_block(oc_block, vM) : brg.bd_block;
    ur_block = ur;
    ur_block_tail = 0;

    return status::success;
}

status_t brg_blocking_t::get_brgemm_ur(
        const primitive_attr_t *attr, const memory_desc_t &dst_md) {
    // Detailed simulation of brgemm convolution init
    if (sp_block <= 0 || ic_block <= 0 || oc_block <= 0)
        return status::invalid_arguments;
    CHECK(estimate_brgemm_ur());

    LDD = oc_without_padding;

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    for (int i = 0; i < M; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if ((utils::one_of(exec_type, exec_trans, exec_vpad) || is_1x1)
                && vM != M && vM != M_tail)
            continue;
        for (int i_init = 0; i_init < 2; i_init++) {
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? beta_init : beta;
                auto vN = (i_N) ? N_tail : N;
                auto vK = (i_K) ? K_tail : K;
                if (vN == 0 || vK == 0) continue;
                brgemm_desc_t brg;
                brgemm_strides_t brg_strides;
                brg_strides.stride_a = ngroups * ic_without_padding
                        * (dilate_w + 1) * src_dsz;
                // weights are padded by oc_block and last_ic_block
                brg_strides.stride_b = rnd_up(ic, last_ic_block_size)
                        * rnd_up(oc, oc_block) * wei_dsz;
                const auto strides_ptr
                        = (brg_type == brgemm_strd) ? &brg_strides : nullptr;
                CHECK(brgemm_utils::init_brgemm_conf(&brg, isa, brg_type,
                        src_dt, wei_dt, brgemm_row_major, alpha, vbeta, LDA,
                        LDB, LDC, vM, vN, vK, strides_ptr, is_bf32));
                CHECK(brgemm_utils::brgemm_blocking(&brg));

                brgemm_attr_t brgattr;
                brgattr.max_bs = max_batch;
                max_vpad = exec_type == exec_vpad ? nstl::max(l_pad, r_pad) : 0;
                brgattr.max_top_vpad = max_vpad;
                brgattr.max_bottom_vpad = max_vpad;
                brgattr.fpmath_mode = attr->fpmath_.mode_;
                if (use_mmla) set_mmla_brgattr(*this, vM, brgattr);
                CHECK(brgemm_desc_set_attr(&brg, brgattr));

                brg.with_sum = with_sum;
                CHECK(brgemm_desc_set_postops(
                        &brg, attr, &dst_md, LDD, bia_dt));
            }
        }
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

float brg_blocking_t::est_eff() {
    const auto ocblock = oc_block / acc_simd_w;

    const auto brgemm_microkernel_eff
            = (static_cast<float>(ocblock) * ur) / ((ur + ocblock) * max_regs);

    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;
    const auto sp_eff = (static_cast<float>(sp) / rnd_up(sp, sp_block));

    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);

    const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);

        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc)
                nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                        ngroups, ocb, nb_oc);
            else if (loop_order == loop_ngcdhw)
                nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp, od,
                        ohp, oh, spb, nb_sp);

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                const int spp = spb * sp_block;
                sp_sz = nstl::min(sp - spp, sp_block);
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc)
                    nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                            ngroups, ocb, nb_oc);
                else if (loop_order == loop_ngcdhw)
                    nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                            ohp, oh, spb, nb_sp);
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }
        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);

    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;

    // -- brgemm kernel: loop by simd_w  --
    l++;
    const auto inp_ur = inp_w(ur, kw_block);
    loop[l].src.set(inp_ur * simd_w, 1, bcast_simd);
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
    auto wei_is = kw_block * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);
    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(kd_block * kh_block * src_is, 1);
    loop[l].dst.set(ur * oc_block, 1);
    wei_is = kd_block * kh_block * kw_block * oc_blocking_size;
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
    wei_is = kd * kh * kw * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    int nb_oh_thr {1}, oh_thr {1}, nb_od_thr {1}, od_thr {1};
    if (!is_os_blocking) {
        const auto dim_oh = nb_sp * dim_sp;
        nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
        oh_thr = nstl::min(oh, nb_oh_thr * oh_block);

        const auto dim_od = nb_oh * dim_oh;
        nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
        od_thr = nstl::min(od, nb_od_thr * od_block);
    }

    src_is = kd * kh * rnd_inp_simd(sp_block, kw, ic);

    auto wei_op = kd * kh * kw * ocblock * ic;
    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(src_is, nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = kd * kh * kw * oc_block * ic;
        wei_op = kd * kh * kw * nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(src_is, 1);
    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
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
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * src_is, 1);
    loop[l].dst.set(od_thr * oh_thr * sp_thr * nsimd_oc_thr * simd_w, 1);
    loop[l].wei.set(kd * kh * kw * nsimd_oc_thr * simd_w * ic, mb_thr);

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
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost
            = 1000.f * job * ic_chunks * nb_kd * nb_kh * nb_kw;

    // Avoid huge batch sizes if possible (ie prefer to block on kd/kh/kw).
    const float gemm_batch_bytes
            = sizeof(brgemm_batch_element_t) * gemm_batch_size;
    const float batch_eff = uses_batch_elements(brg_type, exec_type)
            ? nstl::min(1.f, L2 / (gemm_batch_bytes))
            : 1.f;

    const auto cache_eff = (static_cast<dim_t>(mb) * od * oh * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));
    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff * batch_eff;
    return res_eff;
}

// ow_block choice of the int8 mmla path. The packed source of a call is
// reused for every oc block a thread computes for the same output row, and
// every weight byte loaded by the kernel is reused by all M rows, so the
// widest row block is preferred, subject to:
//  - the packed source of one call fits in half of L2, and
//  - with the (row block, oc block) jobs split over the threads, each thread
//    gets enough oc blocks of a row to reuse its pack (min(nb_oc, 4) jobs)
//    and the jobs are balanced (>= 80%).
// Candidates are visited from the widest down; see iterate_ker_block().
bool brg_blocking_t::mmla_pack_fits() const {
    const dim_t taps = static_cast<dim_t>(kd_block) * kh_block * kw_block;
    const dim_t pack_bytes = static_cast<dim_t>(rnd_up(ow_block, 2))
            * rnd_up(ic_block, 8) * src_dsz * taps;
    return ow_block <= 2 || pack_bytes <= L2 / 2;
}

bool brg_blocking_t::mmla_threads_ok() const {
    const dim_t jobs
            = static_cast<dim_t>(mb) * ngroups * od * oh * nb_ow * nb_oc;
    const dim_t per_thr_min = nstl::min(nb_oc, 4);
    const bool reuse_ok = jobs >= static_cast<dim_t>(nthr) * per_thr_min;
    const bool balance_ok
            = static_cast<float>(jobs) / rnd_up(jobs, static_cast<dim_t>(nthr))
            >= 0.8f;
    return reuse_ok && balance_ok;
}

void brg_blocking_t::iterate_ker_block(brg_blocking_t &best_brgb, int kd_block_,
        int kh_block_, bool maybe_use_buffer, int max_ow_block_thr) {
    if (best_brgb.mmla_choice_fixed) return;
    unsigned est_k_amount = ic * oc_block * wei_dsz;

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

    if (exec_type == exec_vpad) {
        od_block = 1;
        oh_block = 1;
    } else if (exec_type == exec_trans) {
        const auto w_block_size
                = 2 * src_dsz * ic_block * iwp + dst_dsz * ow * oc_block;
        const auto other_size = wei_dsz * kd * kh * kw * ic_block * oc_block;
        const auto L2_available = nstl::min(static_cast<size_t>(div_up(L2, 2)),
                other_size > L2 ? 0 : L2 - other_size);
        if (idp * ihp * w_block_size > L2_available) {
            od_block = utils::saturate(
                    1, od, int(L2_available / (ihp * w_block_size)));
            if (od_block == 1)
                oh_block = utils::saturate(
                        1, oh, int(L2_available / (w_block_size)));
            else
                oh_block = oh;
        } else {
            od_block = 1;
            oh_block = oh;
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
    auto start_ow_block = nstl::min(max_ow_block_thr, max_ow_block_L2);

    sp = ow;
    const auto start_sp_block = is_os_blocking ? ow : start_ow_block;
    auto prev_spb = 0;
    // mmla: widest candidate whose pack fits in L2 (fallback choice)
    const bool mmla_full_kernel = use_mmla && kd_block == kd && kh_block == kh;
    constexpr int mmla_min_rows = 16;
    bool mmla_have_fit = false;
    brg_blocking_t mmla_first_fit;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        if (is_os_blocking && spb != ow) continue;
        prev_spb = spb;
        ow_block = spb;
        sp_block = ow_block;

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
        os_block = sp_block = ow_block;
        update_blocks();

        eff = est_eff();

        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;

        if (mmla_full_kernel) {
            const bool fits = mmla_pack_fits();
            if (fits && !mmla_have_fit) {
                mmla_first_fit = *this;
                mmla_have_fit = true;
            }
            if (fits && mmla_threads_ok()) {
                best_brgb = *this;
                best_brgb.mmla_choice_fixed = true;
                return;
            }
            // splitting rows below mmla_min_rows does not pay off: keep the
            // widest block that fits instead
            if (ow_block <= mmla_min_rows && mmla_have_fit) break;
        }
    }
    if (mmla_full_kernel && mmla_have_fit) {
        best_brgb = mmla_first_fit;
        best_brgb.mmla_choice_fixed = true;
    }
}

status_t brg_blocking_t::calc_blocks() {
    sp = ow;

    nb_ic_blocking = 1;
    // --- Select kernel blocking ---
    // if dst_dt != acc_dt and we need to store intermediate
    // results then we need the out buffer
    const auto maybe_use_buffer = (dst_dt != acc_dt || with_sum);

    std::vector<int> kd_blocks(1), kh_blocks(1);
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
    const auto max_ow_block_thr = utils::saturate(1, ow,
            static_cast<int>(ceil(
                    mb * ngroups * nb_oc * os / (thr_eff_threshold * nthr))));

    ow_block = os_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;
    for (const auto &kd_block : kd_blocks) {
        for (const auto &kh_block : kh_blocks) {
            iterate_ker_block(best_brgb, kd_block, kh_block, maybe_use_buffer,
                    max_ow_block_thr);
        }
    }
    *this = best_brgb;
    if (!IMPLICATION(!is_os_blocking, sp_block > 0))
        return status::unimplemented;

    if (is_os_blocking) {
        ow_block = ow;
        os_block = ow * oh_block;
        sp_block = os_block;
        ow_tail = 0;
    } else {
        ow_block = os_block = sp_block;
        ow_tail = ow % ow_block;
    }
    update_blocks();
    return status::success;
}

bool brg_blocking_t::fast_check_oc_block_1x1() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, acc_simd_w);
    auto res = false;
    if (oc_block == 64) {
        const auto big_spatial
                = od * oh * ow >= 64 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && big_spatial);
    } else if (oc_block == 48) {
        const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);
        res = (oc_block_eff >= 0.95f);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff_1x1() {
    const auto ocblock = oc_block / acc_simd_w;

    // TODO: remove this condition

    const auto brgemm_microkernel_eff
            = (static_cast<float>(ocblock) * ur) / ((ur + ocblock) * max_regs);
    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = is_os_blocking ? div_up(nb_os, nb_os_blocking)
                                          : nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;

    const auto sp_eff = static_cast<float>(sp) / rnd_up(sp, sp_block);
    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);
    const auto oc_block_eff = static_cast<float>(oc) / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    int nb_oh_thr {1}, oh_thr {1}, nb_od_thr {1}, od_thr {1};
    if (!is_os_blocking) {
        const auto dim_oh = nb_sp * dim_sp;
        nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
        oh_thr = nstl::min(oh, nb_oh_thr * oh_block);

        const auto dim_od = nb_oh * dim_oh;
        nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
        od_thr = nstl::min(od, nb_od_thr * od_block);
    }

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);
        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, oss {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc) {
                if (is_os_blocking)
                    nd_iterator_init(start, n, mb, oss, sp_amount, g, ngroups,
                            ocb, nb_oc);
                else
                    nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp,
                            g, ngroups, ocb, nb_oc);
            } else if (loop_order == loop_ngcdhw) {
                if (is_os_blocking)
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, oss,
                            sp_amount);
                else
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp,
                            od, ohp, oh, spb, nb_sp);
            }

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                if (is_os_blocking) {
                    const auto osb_start = oss * nb_os_blocking;
                    const auto osb_range
                            = nstl::min(nb_os - osb_start, nb_os_blocking);
                    for (int osb = 0; osb < osb_range; osb++) {
                        const int osp = (osb_start + osb) * sp_block;
                        sp_sz = nstl::min(os - osp, sp_block);
                    }
                } else {
                    const int spp = spb * sp_block;
                    sp_sz = nstl::min(sp - spp, sp_block);
                }
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc) {
                    if (is_os_blocking)
                        nd_iterator_step(
                                n, mb, oss, sp_amount, g, ngroups, ocb, nb_oc);
                    else
                        nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                                ngroups, ocb, nb_oc);
                } else if (loop_order == loop_ngcdhw) {
                    if (is_os_blocking)
                        nd_iterator_step(
                                n, mb, g, ngroups, ocb, nb_oc, oss, sp_amount);
                    else
                        nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                                ohp, oh, spb, nb_sp);
                }
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }

        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);
    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;
    // -- brgemm kernel: loop by simd_w  --
    l++;
    loop[l].src.set(ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(ur * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(ur * oc_block, 1);
    loop[l].wei.set(oc_blocking_size, nb_ur);
    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(sp_block * ic_blocking_size, 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    auto wei_is = oc_blocking_size;
    auto wei_op = ocblock * ic;
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

    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
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
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
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
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost = 1000.f * job * ic_chunks;

    const auto up_sp_size = is_os_blocking ? 1 : od * oh;

    const auto cache_eff = (static_cast<dim_t>(mb) * up_sp_size * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));

    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::calc_blocks_1x1() {
    const bool is_os_blocking_ok
            = utils::everyone_is(1, stride_d, stride_h) && iw % stride_w == 0;
    const bool is_ic_zero_padded = ic != ic_without_padding;
    is_rtus = is_ic_zero_padded;
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
    // TODO: nb_os_blocking always is 1 for now. Update this code
    nb_os_blocking = 1;
    int start_sp_block = 0;

    if (is_os_blocking) {
        ow_block = 0;

        const auto max_os_block_thr
                = (src_dsz * ic >= 1024 && src_dsz * ic < 4096)
                ? nstl::max(nstl::min(16, os),
                          div_up(os, div_up(nthr, mb * div_up(oc, oc_block))))
                : nstl::max(div_up(2048, oc_block),
                          static_cast<int>(div_up(mb * ngroups * os, nthr)));
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

        start_sp_block = utils::saturate(1, os,
                nstl::min(nstl::min(max_os_block_thr, max_os_block_L2),
                        max_os_block_aliasing));

    } else {
        os_block = 0;

        const auto max_ow_block_thr = utils::saturate(1, ow,
                static_cast<int>(ceil(mb * ngroups * nb_oc * os
                        / (thr_eff_threshold * nthr))));
        const auto max_ow_block_L2 = max_sp_block_L2;

        start_sp_block = utils::saturate(
                1, ow, nstl::min(max_ow_block_thr, max_ow_block_L2));
    }
    os_block = ow_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;

    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        prev_spb = spb;
        os_block = ow_block = sp_block = spb;
        select_ic_block();
        const status_t st = estimate_brgemm_ur();
        if (st != status::success) continue;
        update_blocks();

        use_buffer = (dst_dt != acc_dt || with_sum)
                && (ic_block * nb_ic_blocking < ic);

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
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads,
        bool allow_mmla) {
    using namespace prop_kind;

    brg_blocking_t::L1 = platform::get_per_core_cache_size(1);
    brg_blocking_t::L2 = platform::get_per_core_cache_size(2);
    brg_blocking_t::L3 = platform::get_per_core_cache_size(2);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = utils::zero<decltype(jcp)>();
    jcp.isa = isa;
    if (!is_superset(jcp.isa, sve_128)) return status::unimplemented;

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
    } else if (one_of(jcp.src_dt, f32, bf16, f16)) {
        jcp.acc_dt = f32;
    } else
        return status::unimplemented;

    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);
    jcp.acc_dsz = types::data_type_size(jcp.acc_dt);
    jcp.bia_dsz = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.simd_w = isa_max_vlen(isa) / jcp.src_dsz;
    jcp.acc_simd_w = isa_max_vlen(isa) / jcp.acc_dsz;
    jcp.is_bf32 = false;
    jcp.wei_plain = everyone_is(true, jcp.wei_dt == data_type::f32,
            is_superset(isa, sve_512), weights_d.is_plain());
    if (jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    const bool is_f32
            = jcp.src_dt == data_type::f32 && jcp.wei_dt == data_type::f32;

    const bool is_fwd = one_of(jcp.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);

    if (is_fwd && jcp.ngroups == 1 && jcp.dilate_w == 0 && jcp.kw > 1
            && jcp.stride_w > 1 && jcp.l_pad <= 0 && jcp.r_pad <= 0 && is_f32) {
        // such convolutions are equivalent to
        // [iw / k][kw / k][stride_w / k][ic * k]
        // Folding is limited to f32 because int8/bf16 dot-product instructions
        // (sdot/bfdot) require 4-byte contiguity in the K-dimension. Folding
        // requires interleaving weights across spatial positions (KW) into
        // these VNNI-style 4-byte blocks, which is not supported by current
        // AArch64 reorders.
        const bool pure_1d = (jcp.mb == 1 && jcp.id == 1 && jcp.ih == 1);
        int w_koef = 1;
        auto w_koef_max = nstl::min(jcp.kw, nstl::min(jcp.stride_w, jcp.iw));
        for (int i = 1; i <= w_koef_max; i++) {
            if (IMPLICATION(!pure_1d, jcp.iw % i == 0)
                    && IMPLICATION(jcp.ic * i > jcp.simd_w,
                            (jcp.ic * i) % jcp.simd_w == 0)
                    && jcp.kw % i == 0 && jcp.stride_w % i == 0)
                w_koef = i;
        }
        if (w_koef > 1) {
            jcp.ic_without_padding *= w_koef;
            jcp.ic *= w_koef;
            jcp.iw /= w_koef;
            jcp.kw /= w_koef;
            jcp.stride_w /= w_koef;
            jcp.ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
            jcp.r_pad = calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.ext_kw);
        }
    }

    // TODO: optimize depthwise convolutions (for now direct approach is faster)
    const bool is_depthwise
            = with_groups && jcp.ngroups > 1 && everyone_is(1, jcp.ic, jcp.oc);
    if (is_depthwise)
        if (allow_perf_heuristics(jcp)) return status::unimplemented;

    // TODO: optimize grouped convolutions with small ic
    const bool is_grouped_small_ic
            = jcp.prop_kind != prop_kind::backward_weights && with_groups
            && jcp.ngroups > 1
            && jcp.ic <= jcp.acc_simd_w
            // Enable the shapes not supported in direct convs
            && IMPLICATION(with_groups, is_groups_ok(jcp));
    if (is_grouped_small_ic)
        if (allow_perf_heuristics(jcp)) return status::unimplemented;

    // TODO: optimize the perf of 3d shape with small ic and large spatial

    const bool is_signed_input = jcp.src_dt == s8;
    jcp.s8s8_compensation_required = is_signed_input && !isa_has_s8s8(jcp.isa);
    jcp.has_int8_vnni = isa_has_s8s8(jcp.isa);

    const bool vpad_ok = div_up(nstl::max(0, jcp.l_pad), jcp.stride_w) < jcp.kw
            && div_up(nstl::max(0, jcp.r_pad), jcp.stride_w) < jcp.kw;

    const bool mmla_shape_ok = jcp.ow >= 4 && jcp.ic > 4 && jcp.oc >= 8;

    jcp.use_mmla = allow_mmla && is_fwd && isa == sve_256
            && one_of(jcp.src_dt, u8, s8) && jcp.wei_dt == s8
            && jcp.ngroups == 1 && !jcp.is_1x1 && !jcp.wei_plain
            && !jcp.s8s8_compensation_required && vpad_ok
            && IMPLICATION(allow_perf_heuristics(jcp),
                    mmla_shape_ok
                            && int8_large_spatial_ok(
                                    jcp, mmla_max_oc_block, jcp.ow));

    brg_blocking_t::last_ic_block_size
            = jcp.use_mmla ? 8 : data_type_vnni_granularity(jcp.wei_dt);

    const bool is_bf16
            = jcp.src_dt == data_type::bf16 && jcp.wei_dt == data_type::bf16;
    if (!IMPLICATION(is_bf16, mayiuse_bf16())) return status::unimplemented;

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
    jcp.kh_sets = 1;
    jcp.kw_sets = 1;
    jcp.copy_block_only = false;
    jcp.use_M_mask = 0;
    jcp.is_os_blocking = false;
    jcp.oskip = 0;
    jcp.use_uker = false;
    jcp.use_interleave_stores = false;
    jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf_default;
    jcp.brgemm_bd_loop_innermost = false;

    if (!jcp.wei_plain && jcp.prop_kind != prop_kind::backward_weights) {
        // fast check data layout before spending time for blocking selection
        format_tag_t dat_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);
        CHECK(init_tag(
                jcp.src_tag, src_md, src_d, dat_tag, is_any_eligible(jcp)));
        CHECK(init_tag(
                jcp.dst_tag, dst_md, dst_d, dat_tag, is_any_eligible(jcp)));
    }
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    jcp.idp = jcp.id + jcp.f_pad + jcp.back_pad;
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    CHECK(attr.set_default_formats(&dst_md));

    return status::success;
}

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr,
            nthreads, true));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    using namespace data_type;
    // ======================= blocking =================================

    auto bcast_amount
            = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.kd * jcp.kh * jcp.kw
            * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    if (jcp.use_mmla) jcp.loop_order = loop_ndhwgc;

    const int min_oc_block = jcp.acc_simd_w;

    int selected_ur = 0;
    MAYBE_UNUSED(selected_ur);

    auto try_exec_type = [&]() {
        brg_blocking_t best_brgb = utils::zero<decltype(best_brgb)>();
        best_brgb.oc_block = min_oc_block;
        auto start_ocb = 4;
        start_ocb = nstl::min(div_up(jcp.oc, jcp.acc_simd_w), start_ocb);
        // Start at oc_block=16 for sve_128 so that we always consider at least one case
        // This is tied to the early break in the below loop.
        // TODO: remove this heuristic
        if (isa == sve_128)
            start_ocb = nstl::max(start_ocb, div_up(16, jcp.acc_simd_w));

        if (jcp.use_mmla)
            start_ocb = nstl::min(div_up(jcp.oc, jcp.acc_simd_w),
                    mmla_max_oc_block / jcp.acc_simd_w);

        auto finish_ocb = 1;
        for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
            brg_blocking_t cur_brgb = utils::zero<decltype(best_brgb)>();
            cur_brgb.get_from_jcp(jcp);
            cur_brgb.oc_block = ocb * jcp.acc_simd_w;
            cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);
            if (!cur_brgb.fast_check_oc_block()) continue;
            // eff heuristics seem to be wrong for sve_128, <16 is always worse
            if (isa == sve_128 && cur_brgb.oc_block < 16) break;

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

    if (div_up(nstl::max(0, jcp.l_pad), jcp.stride_w) < jcp.kw
            && div_up(nstl::max(0, jcp.r_pad), jcp.stride_w) < jcp.kw) {
        try_exec_vpad = true;
    }

    const auto ic_padded_block
            = jcp.acc_simd_w * brg_blocking_t::last_ic_block_size;

    bool must_exec_vpad = false;

    // TODO: in future use (kd/kh/kw) and (kd/kh/kw)_pad blocks for more
    // precise calculation of jcp.max_batch
    jcp.max_batch = jcp.kd * jcp.kh * jcp.kw;

    bool try_exec_type_res = false;

    if (try_exec_vpad) {
        jcp.exec_type = exec_vpad;
        try_exec_type_res = try_exec_type();
        // to avoid case when both top and bottom virtual padding are non-zero
        // TODO: remove this restriction
        const auto iw_block = (jcp.ow_block - 1) * jcp.stride_w + 1;
        if (!must_exec_vpad && !jcp.use_mmla && (iw_block > jcp.iw))
            try_exec_type_res = false;
    }

    if (jcp.use_mmla && try_exec_type_res == false)
        return status::unimplemented;

    if (try_exec_type_res == false && try_exec_trans) {
        jcp.exec_type = exec_trans;

        // try loop_ndhwgc always for exec_trans
        jcp.loop_order = loop_ndhwgc;

        // we read input block only once for loop_ndhwgc, so we don't need to
        // keep it memory
        if (jcp.loop_order == loop_ndhwgc) { jcp.copy_block_only = true; }

        jcp.is_ic_padded = one_of(jcp.wei_dt, bf16, f16, s8)
                && jcp.ic * jcp.kw_sets > ic_padded_block;

        try_exec_type_res = try_exec_type();
    }
    if (try_exec_base && try_exec_type_res == false) {
        jcp.exec_type = exec_base;
        try_exec_type_res = try_exec_type();
    }

    if (try_exec_type_res == false) return status::unimplemented;

    // ============ end blocking ===========================================

    jcp.brg_type = (jcp.use_uker && jcp.exec_type == exec_trans)
            ? brgemm_static_offs
            : brgemm_addr; // TODO: Choose right type of BRGEMM

    if (jcp.ow_block == 0 || jcp.ic_block == 0 || jcp.oc_block == 0)
        return status::unimplemented;

    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, P4K), sc_size);

    if (!jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    if (!post_ops_ok(jcp, attr, dst_d)) return status::unimplemented;

    jcp.buffer_size = jcp.LDC * jcp.M;

    jcp.nb_od = div_up(jcp.od, jcp.od_block);
    jcp.nb_oh = div_up(jcp.oh, jcp.oh_block);

    if (jcp.exec_type == exec_trans) {
        // TODO: this is rough estimation of buffer for transpose input
        dim_t ds = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.idp, jcp.od_block, jcp.kd,
                           jcp.stride_d, jcp.dilate_d)
                          + nstl::max(0, jcp.f_pad)
                          + nstl::max(0, jcp.back_pad))
                : jcp.idp;
        dim_t hs = jcp.copy_block_only
                ? (brg_blocking_t::get_inp_size(jcp.ihp, jcp.oh_block, jcp.kh,
                           jcp.stride_h, jcp.dilate_h)
                          + nstl::max(0, jcp.t_pad) + nstl::max(0, jcp.b_pad))
                : jcp.ihp;
        if (jcp.is_os_blocking)
            hs = div_up(rnd_up(hs * jcp.iwp, jcp.brgM), jcp.iwp);

        jcp.inp_buffer_size = rnd_up(
                ds * hs * jcp.iwp * jcp.ngroups * jcp.nb_ic * jcp.LDA, P4K);

        jcp.inp_buffer_mask_size = rnd_up(static_cast<dim_t>(jcp.nb_od)
                        * jcp.nb_oh * jcp.nb_ow * jcp.ngroups * jcp.nb_ic,
                P4K);
    }

    const bool with_pad = jcp.f_pad > 0 || jcp.back_pad > 0 || jcp.t_pad > 0
            || jcp.b_pad > 0;

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
    jcp.with_scales = !src_scales.has_default_values()
            || !wei_scales.has_default_values()
            || jcp.scale_adjust_factor != 1.0f;
    jcp.is_oc_scale = wei_scales.get_mask() > 0;

    // disables the shape with small ic but large spatial
    // or specific large spatial shapes for int8 conv
    // new
    const auto is_ok_large_spatial = jcp.use_mmla
            ? int8_large_spatial_ok(jcp, mmla_max_oc_block, jcp.ow)
            : int8_large_spatial_ok(jcp, jcp.oc_block, jcp.ow_block);

    if (one_of(jcp.src_dt, u8, s8) && !is_ok_large_spatial)
        if (allow_perf_heuristics(jcp)) return status::unimplemented;

    // For padding shapes, we calculate the comp along with the computation
    // inside brgemm kernel when output size is small to get optimal perf
    // Or we calculate the comp using brgemm_coomp_pad kernel
    const auto output_sz = static_cast<dim_t>(jcp.mb) * jcp.ngroups * jcp.oc
            * jcp.od * jcp.oh * jcp.ow;
    const auto comp_with_pads
            = (jcp.src_zero_point || jcp.s8s8_compensation_required)
            && IMPLICATION(jcp.exec_type == exec_vpad, with_pad);
    jcp.req_brg_comp_pad = comp_with_pads && output_sz <= 8192 && jcp.oc < 512;
    jcp.req_cal_comp_pad = comp_with_pads && !jcp.req_brg_comp_pad;

    if (jcp.use_mmla) {
        /// padded taps are packed as zp rows, so full-filter comp applies
        jcp.req_brg_comp_pad = false;
        jcp.req_cal_comp_pad = false;

        using namespace jit_brgemm_conv_mmla_pack_kernel;
        const int rows = nstl::max(jcp.M, jcp.M_tail);
        jcp.mmla_pack_buffer_size = rnd_up(jcp.max_batch
                        * jit_brgemm_conv_mmla_pack_kernel_t::tap_size(
                                rows, jcp.ic_block),
                P4K);
        jcp.mmla_pad_row_size = rnd_up(jcp.ic_block, 64);
    }

    // estimate the number of kernel range combination for compensation
    const auto kd_cnt = 1 + utils::div_up(abs(jcp.f_pad), jcp.dilate_d + 1)
            + utils::div_up(abs(jcp.back_pad), jcp.dilate_d + 1);
    const auto kh_cnt = 1 + utils::div_up(abs(jcp.t_pad), jcp.dilate_h + 1)
            + utils::div_up(abs(jcp.b_pad), jcp.dilate_h + 1);
    const auto kw_cnt
            = (1
                      + (utils::div_up(abs(jcp.l_pad), jcp.dilate_w + 1)
                              + utils::div_up(
                                      abs(jcp.r_pad), jcp.dilate_w + 1)))
            * 2;

    jcp.ker_ranges_size = jcp.exec_type == exec_base ? kd_cnt * kh_cnt * kw_cnt
                                                     : kd_cnt * kh_cnt;
    jcp.comp_a_buffer_size
            = jcp.ngroups * jcp.nb_oc * jcp.ker_ranges_size * jcp.oc_block;
    jcp.s8s8_comp_buffer_size = jcp.comp_a_buffer_size;

    // enable ununroll_bd_loop for big shapes to reduce kernel sizes
    jcp.ununroll_bd_loop
            = static_cast<dim_t>(jcp.M) * jcp.N * (jcp.is_bf32 ? 1 : 2)
            > 8 * 1024;

    if (!IMPLICATION(jcp.is_bf32, jcp.use_uker)) return status::unimplemented;

    return status::success;
}

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr,
            nthreads, false));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (!jcp.is_1x1) return status::unimplemented;

    using namespace data_type;
    // ===================== blocking =================================

    auto bcast_amount
            = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.oc) * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    const auto min_oc_block = jcp.acc_simd_w;

    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // max_batch is 1 for 1x1 convolutions
    jcp.max_batch = 1;

    brg_blocking_t best_brgb = utils::zero<decltype(best_brgb)>();
    best_brgb.oc_block = min_oc_block;
    auto start_ocb = 4;
    start_ocb = nstl::min(div_up(jcp.oc, jcp.acc_simd_w), start_ocb);

    auto finish_ocb = 1;

    const bool is_os_blocking_ok
            = utils::everyone_is(1, jcp.stride_d, jcp.stride_h)
            && jcp.iw % jcp.stride_w == 0;
    if (jcp.wei_plain && is_os_blocking_ok) {
        start_ocb = div_up(jcp.oc, jcp.acc_simd_w);
    }

    for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
        brg_blocking_t cur_brgb = utils::zero<decltype(cur_brgb)>();
        cur_brgb.get_from_jcp(jcp);
        cur_brgb.oc_block = ocb * min_oc_block;
        cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);

        // eff heuristics seem to be wrong for sve_128, <16 is always worse
        if (isa == sve_128 && cur_brgb.oc_block < 16) break;

        if (!cur_brgb.fast_check_oc_block_1x1()) continue;

        cur_brgb.calc_blocks_1x1();
        const status_t st = cur_brgb.get_brgemm_ur(&attr, dst_md);
        if (st != status::success) continue;
        cur_brgb.eff = cur_brgb.est_eff_1x1();
        if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
    }
    best_brgb.save_to_jcp(jcp);

    // =============== end blocking =================================
    jcp.brg_stride_a = jcp.ic_block * jcp.src_dsz;
    jcp.brg_stride_b = jcp.ic_block * jcp.oc_without_padding * jcp.wei_dsz;

    if (jcp.ic_block == 0 || jcp.oc_block == 0) return status::unimplemented;

    // Configure matrix sizes

    if (best_brgb.is_os_blocking) {
        if (jcp.os_block == 0) return status::unimplemented;
        jcp.M = jcp.brgM = jcp.os_block;
        jcp.M_tail = jcp.brgM_tail = jcp.os % jcp.os_block;
    } else {
        if (jcp.ow_block == 0) return status::unimplemented;
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

    // TODO: heuristic to dispatch BF32 BRGeMM
    // The following condition checks for shapes where down-convert execution
    // in brgemm fails
    if (jcp.is_bf32 && jcp.ic < 64 && jcp.ic % 32 != 0)
        return status::unimplemented;

    if (jcp.use_uker)
        jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf1;
    if (!jcp.wei_plain)
        CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    if (!post_ops_ok(jcp, attr, dst_d)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    // no inp buffer or brgemm_vpad for 1x1
    constexpr int align_size = platform::get_cache_line_size();
    jcp.exec_type = jcp.is_rtus ? exec_trans : exec_base;
    jcp.inp_buffer_size
            = jcp.is_rtus ? rnd_up(jcp.LDA * jcp.os, align_size) : 0;
    jcp.inp_buffer_mask_size = jcp.is_rtus
            ? rnd_up(div_up(jcp.nb_ic, jcp.nb_ic_blocking) * jcp.nb_os,
                      align_size)
            : 0;
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
    jcp.s8s8_comp_buffer_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    jcp.comp_a_buffer_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    jcp.with_scales = !src_scales.has_default_values()
            || !wei_scales.has_default_values()
            || jcp.scale_adjust_factor != 1.0f;
    jcp.is_oc_scale = wei_scales.get_mask() > 0;

    // enable ununroll_bd_loop for big shapes to reduce kernel sizes
    jcp.ununroll_bd_loop
            = static_cast<dim_t>(jcp.M) * jcp.N * (jcp.is_bf32 ? 1 : 2)
            > 8 * 1024;

    return status::success;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp) {
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
    if (jcp.use_mmla) {
        // exec_vpad only, so the exec_trans input buffer key is free
        assert(jcp.exec_type == exec_vpad);
        scratchpad.book(key_conv_brgemm_inp_buffer,
                static_cast<size_t>(jcp.nthr)
                        * (jcp.mmla_pack_buffer_size + jcp.mmla_pad_row_size),
                sizeof(char), 0, P4K);
        if (jcp.src_zero_point) {
            // used only when the runtime src zero point does not fit the src
            // data type (see mmla_zp_correction); the padded-compensation
            // keys are free since the mmla path never uses them
            assert(!jcp.req_cal_comp_pad && !jcp.s8s8_compensation_required);
            const size_t ks = static_cast<size_t>(jcp.kd) * jcp.kh * jcp.kw;
            scratchpad.book(key_brgemm_primitive_zp_comp_a,
                    jcp.nb_oc * ks * jcp.oc_block, sizeof(int32_t), 0, P4K);
            const size_t rows = nstl::max(jcp.M, jcp.M_tail);
            scratchpad.book(key_brgemm_primitive_buffer_comp,
                    static_cast<size_t>(jcp.nthr) * rows * jcp.LDC,
                    sizeof(int32_t), 0, P4K);
        }
    }
    if (jcp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer, jcp.nthr * jcp.buffer_size,
                jcp.acc_dsz, 0, P4K);
    }
    if (jcp.s8s8_compensation_required && jcp.req_cal_comp_pad) {
        scratchpad.book(key_brgemm_primitive_buffer_comp,
                jcp.s8s8_comp_buffer_size, sizeof(int32_t), 0, P4K);
    }
    if (jcp.src_zero_point && jcp.req_cal_comp_pad) {
        scratchpad.book(key_brgemm_primitive_zp_comp_a, jcp.comp_a_buffer_size,
                sizeof(int32_t), 0, P4K);
    }
}

void balance_bwd_w(jit_brgemm_conv_conf_t &jcp) {

    const auto os_chunks = jcp.nthr_mb_work;
    const auto oc_chunks = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const auto ic_chunks = div_up(jcp.nb_ic, jcp.nb_ic_blocking);

    auto calc_mem_cost
            = [=](int nthr_mb, int nthr_g, int nthr_oc_b, int nthr_ic_b) {
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

        float wei_compensation_scale = 0.5f * (dst_size + src_size) / wei_size;
        float oi_channels_ratio = (float)(oc_chunks) / ic_chunks;

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
        float src_v
                = src_type_size * src_coef * thr_g * thr_ic_amount * thr_src_sp;
        float dst_v
                = src_type_size * dst_coef * thr_g * thr_oc_amount * thr_dst_sp;
        float wei_v = acc_type_size * wei_coef * thr_g * thr_oc_amount
                * thr_ic_amount * wei_ks;

        return src_v + dst_v + wei_v;
    };

    auto balance = [=](int &nthr_, int &nthr_mb_, int &nthr_g_, int &nthr_oc_b_,
                           int &nthr_ic_b_) {
        nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

        if (jcp.nthr < jcp.ngroups) {
            /* simplification... fortunately it doesn't hurt much */
            nthr_ = nthr_g_ = jcp.nthr;
            return;
        }

        nthr_g_ = jcp.ngroups;
        const int nthr = jcp.nthr / nthr_g_;

        float best_mem_cost
                = calc_mem_cost(nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_);

        /* find the best thread distribution with lowest memory cost */

        const int nthr_mb_max = nstl::min(nthr, jcp.nthr_mb_work);
        for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
            const int nthr_par = nthr / nthr_mb;
            const int nthr_oc_b_max = nstl::min(nthr_par,
                    oc_chunks); // Amount of nb_oc_blocks
            for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
                int nthr_ic_b = nstl::min(
                        nthr_par / nthr_oc_b, (jcp.nb_ic / jcp.nb_ic_blocking));

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
        nthr_g = nstl::min(jcp.nthr, jcp.ngroups);
        nthr_oc_b = nstl::min(jcp.nthr / nthr_g, div_up(jcp.nb_oc, 2));
        nthr_ic_b = nstl::min(
                jcp.nthr / (nthr_g * nthr_oc_b), div_up(jcp.nb_ic, 2));
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

    return status::unimplemented;

    return status::success;
}

status_t init_scratchpad_bwd_w(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

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
            + jcp.tr_iw * jcp.oc_block;

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
        const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
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

    if (scratchpad.size() > scratchpad_limit)
        return status::unimplemented;
    else
        return status::success;
}

} // namespace brgemm_convolution_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
