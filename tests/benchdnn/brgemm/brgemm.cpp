/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#include <float.h>
#include <functional>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// TODO: refactor the driver to avoid using extra flags of a memory descriptor.
#include "src/common/memory_desc.hpp"

#include "tests/test_isa_common.hpp"

#include "utils/dnnl_query.hpp"
#include "utils/parallel.hpp"
#include "utils/parser.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "brgemm/brgemm.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
// Need these macro independently of API.
#if defined(DNNL_X64) && DNNL_X64 == 1
#define brg_x64
#elif defined(DNNL_AARCH64) && DNNL_AARCH64 == 1
#define brg_aarch64
#endif

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#if defined(DNNL_X64) && DNNL_X64 == 1
#define namespace_impl dnnl::impl::cpu::x64
#elif defined(DNNL_AARCH64) && DNNL_AARCH64 == 1
#define namespace_impl dnnl::impl::cpu::aarch64
#endif

#if defined(brg_x64) || defined(brg_aarch64)
template <>
struct dnnl_api_traits<namespace_impl::brgemm_kernel_t *> {
    static void destroy(namespace_impl::brgemm_kernel_t *t) {
        DNN_SAFE_V(namespace_impl::brgemm_kernel_destroy(t));
    }
};
#endif

#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)

template <>
struct dnnl_api_traits<dnnl_brgemm_t> {
    static void destroy(dnnl_brgemm_t t) { DNN_SAFE_V(dnnl_brgemm_destroy(t)); }
};

template <>
struct dnnl_api_traits<dnnl_transform_t> {
    static void destroy(dnnl_transform_t t) {
        DNN_SAFE_V(dnnl_transform_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_ukernel_attr_params_t> {
    static void destroy(dnnl_ukernel_attr_params_t t) {
        DNN_SAFE_V(dnnl_ukernel_attr_params_destroy(t));
    }
};

#endif

#endif // DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE

namespace brgemm {

#if defined(brg_x64) || defined(brg_aarch64)

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
/// Initializes BRGEMM attributes from an input string.
///
/// @param brgattr Output BRGEMM attributes.
/// @param str Input string of values in the format: KEY:VALUE[+KEY:VALUE[...]].
///     `KEY` follows exact name of the brgemm_attr_t object members and their
///     `VALUE` follow the member type. enum and boolean types are treated as
///     integers.
///
dnnl_status_t brgemm_attr_init(
        namespace_impl::brgemm_attr_t *brgattr, const prb_t *prb) {
    using namespace namespace_impl;

    // `max_bs` is handled directly through the driver interface.
    brgattr->max_bs = prb->batch_size;

    // `fpmath_mode` is handled directly through the driver interface.
    brgattr->fpmath_mode = prb->attr.fpmath_mode.mode;

    const auto &str = prb->brgemm_attr;
    if (str.empty()) return dnnl_success;

    size_t entry_pos = 0;
    while (entry_pos != std::string::npos) {
        auto key_value_str = parser::get_substr(str, entry_pos, '+');
        size_t value_pos = 0;
        auto key_str = parser::get_substr(key_value_str, value_pos, ':');
        auto value_str = parser::get_substr(key_value_str, value_pos, '\0');

#define PROCESS_SETTING_KEY_VAL(setting, key) \
    if (key_str.compare(STRINGIFY(key)) == 0) \
        brgattr->setting = std::stoi(value_str);

#define PROCESS_KEY_VAL(setting) PROCESS_SETTING_KEY_VAL(setting, setting)

        // TODO: `max_top_vpad` and `max_bottom_vpad` do not affect anything in
        // the kernel call and reference computation so far since
        // batch_element_t struct is not adjusted to incorporate different pad
        // values.
        // PROCESS_KEY_VAL(max_top_vpad);
        // PROCESS_KEY_VAL(max_bottom_vpad);
        PROCESS_KEY_VAL(hint_expected_A_size);
        PROCESS_KEY_VAL(hint_expected_B_size);
        PROCESS_KEY_VAL(hint_expected_C_size);
        PROCESS_KEY_VAL(wary_A_k_tail_read);
        PROCESS_KEY_VAL(extendable_k);
        PROCESS_KEY_VAL(generate_skip_accumulation);
        // TODO: `bd_mask` can't be passed to the kernel at this moment, that's
        // why `bd_mask_level` has to stay `0` for now until it's enabled.
        // PROCESS_KEY_VAL(bd_mask_level);
        PROCESS_KEY_VAL(use_uker);
        PROCESS_KEY_VAL(var_bs);
        PROCESS_KEY_VAL(use_interleave_stores);
#if defined(brg_x64)
        // ACE is an x64-only compute path.
        PROCESS_KEY_VAL(use_ace);
#elif defined(brg_aarch64)
        PROCESS_KEY_VAL(use_mmla);
        PROCESS_KEY_VAL(use_mmla_packed_a);
#endif
        PROCESS_KEY_VAL(b_is_vnni);
        PROCESS_KEY_VAL(postops_only);
        PROCESS_KEY_VAL(hint_bd_block);
        PROCESS_KEY_VAL(hint_bd_block2);
        PROCESS_KEY_VAL(hint_ld_block);
        PROCESS_KEY_VAL(hint_ld_block2);

        PROCESS_SETTING_KEY_VAL(hint_prfA.dist1, hint_prfA_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfA.dist2, hint_prfA_dist2);
        PROCESS_SETTING_KEY_VAL(hint_prfB.dist1, hint_prfB_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfB.dist2, hint_prfB_dist2);
        PROCESS_SETTING_KEY_VAL(hint_prfC.dist1, hint_prfC_dist1);
        PROCESS_SETTING_KEY_VAL(hint_prfC.dist2, hint_prfC_dist2);

#undef PROCESS_SETTING_KEY_VAL
#undef PROCESS_KEY_VAL

        if (key_str.find(STRINGIFY(hint_innermost_loop)) != std::string::npos)
            brgattr->hint_innermost_loop
                    = static_cast<brgemm_kernel_innermost_loop_t>(
                            std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_loop_order)) != std::string::npos)
            brgattr->hint_loop_order = static_cast<brgemm_kernel_loop_order_t>(
                    std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_prefetching)) != std::string::npos)
            brgattr->hint_prefetching
                    = static_cast<brgemm_kernel_prefetching_t>(
                            std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_load_nt_A)) != std::string::npos)
            brgattr->hint_load_nt_A = static_cast<brgemm_kernel_hint_nt_t>(
                    std::stoi(value_str));
        if (key_str.find(STRINGIFY(hint_load_nt_B)) != std::string::npos)
            brgattr->hint_load_nt_B = static_cast<brgemm_kernel_hint_nt_t>(
                    std::stoi(value_str));
    }

    return dnnl_success;
}

#if defined(brg_aarch64)
std::string mmla_wei_tag(int ld_block, int ld_block2, dnnl_data_type_t wei_dt) {
    const std::string n_blk = std::to_string(ld_block * ld_block2) + "b";
    if (dnnl_data_type_size(wei_dt) == 1) return "BA" + n_blk + "8a";
    return "BA2a" + n_blk + "4a";
}

std::string mmla_src_tag(dnnl_data_type_t src_dt) {
    return dnnl_data_type_size(src_dt) == 1 ? "AB2a8b" : "AB2a4b";
}
#endif

namespace_impl::brgemm_batch_kind_t str2batch_kind(const std::string &str) {
    if (str == "addr")
        return namespace_impl::brgemm_batch_kind_t::brgemm_addr;
    else if (str == "offs")
        return namespace_impl::brgemm_batch_kind_t::brgemm_offs;
    assert(!"Unsupported batch kind value");
    return namespace_impl::brgemm_batch_kind_t::brgemm_batch_kind_undef;
}
#endif

int fill_data(data_kind_t kind, int exec_arg, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {

    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->k;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_f32_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            if (!is_one) {
                mem_fp.set_f32_elem(idx, 0.f);
                continue;
            }
            float val = gen(int_seed);
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

// An object to pass information between different modules of the flow.
struct kernel_args_t {
    kernel_args_t(const prb_t *prb)
        :
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
        brgemm_kernel_(nullptr)
#if defined(brg_x64)
        , palette()
        , need_tile_config_(false)
#elif defined(brg_aarch64)
        , use_mmla_(false)
        , use_mmla_packed_a_(false)
#endif
        , is_b_data_layout_vnni_(false)
        , original_wei_md_size_(0)
#else
        brgemm_(nullptr)
        , transform_(nullptr)
        , need_pack_(0)
#endif
        , scratchpad_size_(0)
        , generate_skip_accumulation_(false)
        , prb_(prb) {
    }

    // Output members
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    namespace_impl::brgemm_kernel_t *brgemm_kernel_;
#if defined(brg_x64)
    char palette[/*dnnl::impl::cpu::x64::AMX_PALETTE_SIZE = */ 64];
    bool need_tile_config_;
#elif defined(brg_aarch64)
    std::string mmla_wei_tag_;
    bool use_mmla_;
    bool use_mmla_packed_a_;
#endif
    bool is_b_data_layout_vnni_;
    size_t original_wei_md_size_;
#else
    dnnl_brgemm_t brgemm_;
    dnnl_transform_t transform_;
    int need_pack_; // `int` to match C API
#endif
    size_t scratchpad_size_;
    bool generate_skip_accumulation_;

    // Input members
    const prb_t *prb_;
};

std::string prepare_src_format_string(const kernel_args_t &kernel_args) {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(brg_aarch64)
    if (kernel_args.use_mmla_ && kernel_args.use_mmla_packed_a_)
        return mmla_src_tag(kernel_args.prb_->src_dt());
#else
    MAYBE_UNUSED(kernel_args);
#endif
    return "";
}

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
std::string prepare_wei_format_string(
        const prb_t *prb, const kernel_args_t &kernel_args) {
#if defined(brg_aarch64)
    if (kernel_args.use_mmla_) return kernel_args.mmla_wei_tag_;
#endif
    // The weights data type affects the last inner block (for VNNI-friendliness).
    // LDB affects the choice of B block.
    std::string wtag("BA16a");
    wtag += std::to_string(prb->get_ldb()) + "b";
    if (kernel_args.is_b_data_layout_vnni_) {
        switch (prb->wei_dt()) {
            case dnnl_f32: break;
            case dnnl_f16:
            case dnnl_bf16: wtag += "2a"; break;
            case dnnl_f8_e5m2:
            case dnnl_f8_e4m3:
            case dnnl_u8:
            case dnnl_s8: wtag += "4a"; break;
            default: assert(!"unsupported data type");
        }
    }

    return wtag;
}
#endif

int init_kernel(kernel_args_t &kernel_args, res_t *res) {
    const prb_t *prb = kernel_args.prb_;

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    using namespace namespace_impl;

    // Supports only address model for now as only affects the way memory is
    // passed to `brgemm_batch_element_t` object.
    brgemm_batch_kind_t batch_kind = str2batch_kind(prb->batch_kind);
    brgemm_layout_t layout = brgemm_layout_t::brgemm_row_major;

    // Pass `isa_undef` for now since internal work with it or rather isa bits
    // than isa values directly which causes misalignment between public enum
    // and internal values.
    // TODO: re-consider enabling isa values.
    const auto isa_undef = cpu_isa_t::isa_undef;

    brgemm_desc_t brgemm_desc;

    // Create BRGeMM descriptor, analogous to primitive descriptor creation
    const auto status_init = brgemm_desc_init(&brgemm_desc, isa_undef,
            batch_kind, prb->src_dt(), prb->wei_dt(), false /* transA */,
            false /* transB */, layout, prb->alpha, prb->beta, prb->get_lda(),
            prb->get_ldb(), prb->get_ldc(), prb->m, prb->n, prb->k,
            nullptr /* strides */);
    SAFE(check_dnnl_status(status_init, prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    const auto &wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        attr_args.prepare_quant(
                prb->attr, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, 2);
    }
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE(create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims),
            WARN);
    dims_t dst_strides = {prb->get_ldd(), 1};
    auto dst_md = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims.data(), prb->dst_dt(), "", dst_strides);

    SAFE(check_dnnl_status(brgemm_desc_set_postops(&brgemm_desc, dnnl_attr,
                                   dst_md, prb->get_ldd(), prb->bia_dt),
                 prb, res),
            WARN);
    if (res->state == SKIPPED) return OK;

    brgemm_attr_t brgemm_attr;
    DNN_SAFE(brgemm_attr_init(&brgemm_attr, prb), WARN);
#if defined(brg_aarch64)
    if (brgemm_attr.use_mmla && prb->batch_size > 1) {
        SAFE(check_dnnl_status(dnnl_unimplemented, prb, res), WARN);
        return OK;
    }
    // The reorder into the int8 mmla weights layout cannot compute the src
    // zero-point compensation yet.
    if (brgemm_attr.use_mmla
            && !prb->attr.zero_points.is_def(DNNL_ARG_SRC)) {
        SAFE(check_dnnl_status(dnnl_unimplemented, prb, res), WARN);
        return OK;
    }
#endif
    SAFE(check_dnnl_status(
                 brgemm_desc_set_attr(&brgemm_desc, brgemm_attr), prb, res),
            WARN);
    if (res->state == SKIPPED) return OK;

    SAFE(check_dnnl_status(brgemm_desc_finalize(&brgemm_desc), prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    kernel_args.generate_skip_accumulation_
            = brgemm_attr.generate_skip_accumulation;
#if defined(brg_aarch64)
    kernel_args.use_mmla_ = brgemm_desc.brgattr.use_mmla;
    if (kernel_args.use_mmla_) {
        kernel_args.mmla_wei_tag_
                = mmla_wei_tag(brgemm_desc.ld_block, brgemm_desc.ld_block2, prb->wei_dt());
    }
    kernel_args.use_mmla_packed_a_ = brgemm_desc.brgattr.use_mmla_packed_a;
#endif

    // Create BRGeMM kernel, analogous to primitive creation.
    // ctx_init can here be used to select core type on hetero ISA with TBB.
    brgemm_kernel_t **brgemm_kernel_addr = &kernel_args.brgemm_kernel_;
    // The lambda performs the return type conversion (dnnl_status_t -> int)
    // required by create_in_thr_ctx.
    std::function<int()> brgemm_kernel_create_fn = [&] {
        DNN_SAFE(brgemm_kernel_create(brgemm_kernel_addr, brgemm_desc), WARN);
        return OK;
    };
    SAFE(create_in_thr_ctx(prb->ctx_init, brgemm_kernel_create_fn), WARN);

#if defined(brg_x64)
    // Palette configuration is required here to have `kernel_args`
    // initialization consoidated in a single place.
    const auto init_tiles_st
            = brgemm_init_tiles(brgemm_desc, kernel_args.palette);
    if (init_tiles_st == dnnl_success) { kernel_args.need_tile_config_ = true; }
#endif

    kernel_args.is_b_data_layout_vnni_ = brgemm_desc.is_b_data_layout_vnni();
    kernel_args.scratchpad_size_ = brgemm_desc.get_wsp_buffer_size();
#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)
    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->dst_dims.data());
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE(create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims),
            WARN);
    auto dnnl_post_ops = query_post_ops(dnnl_attr);

    dnnl_status_t st = dnnl_success;
    auto &brgemm = kernel_args.brgemm_;
    // brgemm ukernel returns unimplemented for int8 and bf16
    // on aarch64 at the moment, for now we'll treat those cases
    // as SKIPPED rather than UNTESTED FAILED
    st = dnnl_brgemm_create(&brgemm, prb->m, prb->n, prb->k, prb->batch_size,
            prb->get_lda(), prb->get_ldb(), prb->get_ldc(), prb->src_dt(),
            prb->wei_dt(), prb->acc_dt());
#if defined(brg_aarch64)
    if (st == dnnl_unimplemented) {
        SAFE(check_dnnl_status(st, prb, res), WARN);
        if (res->state == SKIPPED) return OK;
    }
#endif
    DNN_SAFE(st, WARN);

    // Only `beta` equal to `0.f` and `1.f` works.
    DNN_SAFE(dnnl_brgemm_set_add_C(brgemm, static_cast<int>(prb->beta)), WARN);
    DNN_SAFE(dnnl_brgemm_set_post_ops(
                     brgemm, prb->get_ldd(), prb->dst_dt(), dnnl_post_ops),
            WARN);
    if (!prb->attr.scales.is_def(DNNL_ARG_SRC)) {
        DNN_SAFE(dnnl_brgemm_set_A_scales(
                         brgemm, prb->attr.scales.get_mask(DNNL_ARG_SRC)),
                WARN);
    }
    if (!prb->attr.scales.is_def(DNNL_ARG_WEIGHTS)) {
        DNN_SAFE(dnnl_brgemm_set_B_scales(
                         brgemm, prb->attr.scales.get_mask(DNNL_ARG_WEIGHTS)),
                WARN);
    }
    if (!prb->attr.scales.is_def(DNNL_ARG_DST)) {
        DNN_SAFE(dnnl_brgemm_set_D_scales(
                         brgemm, prb->attr.scales.get_mask(DNNL_ARG_DST)),
                WARN);
    }
    // This call is responsible whether the final configuration is supported
    // or not.
    st = dnnl_brgemm_finalize(brgemm);
    SAFE(check_dnnl_status(st, prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    dnnl_pack_type_t pack_type = dnnl_pack_type_undef;
    DNN_SAFE(dnnl_brgemm_get_B_pack_type(
                     &pack_type, prb->src_dt(), prb->wei_dt()),
            WARN);
    kernel_args.need_pack_ = pack_type == dnnl_pack_type_pack32;

    DNN_SAFE(dnnl_brgemm_generate(brgemm), WARN);
    DNN_SAFE(dnnl_brgemm_get_scratchpad_size(
                     brgemm, &kernel_args.scratchpad_size_),
            WARN);

    if (kernel_args.need_pack_) {
        // Create a memory desc based on user inputs and query strides to use
        // them in a pack routine.
        const dnnl_dims_t wei_dims = {prb->k * prb->batch_size, prb->n};
        auto wei_md = dnn_mem_t::init_md(prb->ndims, wei_dims, prb->wei_dt(),
                prb->wtag, prb->strides[STRIDES_WEI]);
        const auto &wei_strides = query_md_strides(wei_md);
        assert(query_md_ndims(wei_md) == 2);

        auto &transform = kernel_args.transform_;
        // Choose `no_trans` for cases when K = 1 as less memory is required.
        auto in_pack_type = wei_strides[1] > wei_strides[0]
                ? dnnl_pack_type_trans
                : dnnl_pack_type_no_trans;
        // One of strides implicitly equals to `1`.
        auto in_ld = MAX2(wei_strides[0], wei_strides[1]);
        st = dnnl_transform_create(&transform, prb->k * prb->batch_size, prb->n,
                in_pack_type, in_ld, prb->get_ldb(), prb->wei_dt(),
                prb->wei_dt());
        SAFE(check_dnnl_status(st, prb, res), WARN);
        if (res->state == SKIPPED) return OK;

        DNN_SAFE(dnnl_transform_generate(transform), WARN);
    }

    // Unneeded from API perspective, it's needed for reference.
    kernel_args.generate_skip_accumulation_ = false;
#endif
    return OK;
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const auto dt = prb->get_dt(kind);
    const float trh = dt == dnnl_f32 ? 1e-6f : epsilon_dt(dt);
    cmp.set_threshold(trh);
    cmp.set_zero_trust_percent(90.f); // TODO: why so bad filling?
}

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
// A special wrapper needed to match internal benchdnn infrastructure.
dnnl_status_t brgemm_kernel_execute_postops_wrapper(
        const namespace_impl::brgemm_kernel_t *brgemm_kernel,
        const std::string &batch_kind, int batch_size, const void *src_ptr,
        const void *wei_ptr,
        const namespace_impl::brgemm_batch_element_t *batch_element,
        void *acc_ptr, void *dst_ptr,
        const namespace_impl::brgemm_post_ops_data_t &post_ops_data,
        void *scratchpad_ptr, const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {

    if (batch_kind == "addr") {
        brgemm_kernel_execute_postops(brgemm_kernel, batch_size, batch_element,
                acc_ptr, dst_ptr, post_ops_data, scratchpad_ptr);
    } else if (batch_kind == "offs") {
        brgemm_kernel_execute_postops(brgemm_kernel, batch_size, src_ptr,
                wei_ptr, batch_element, acc_ptr, dst_ptr, post_ops_data,
                scratchpad_ptr);
    }
    return dnnl_success;
}
#else
// A special wrapper needed to match internal benchdnn infrastructure.
dnnl_status_t brgemm_kernel_execute_postops_wrapper(const_dnnl_brgemm_t brgemm,
        const bool use_dst_as_acc, const void *src_ptr,
        const void *wei_packed_ptr, const std::vector<dnnl_dim_t> &offsets,
        void *acc_ptr, void *dst_ptr, void *scratchpad_ptr,
        const_dnnl_ukernel_attr_params_t attr_params,
        const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {

    dnnl_status_t st = dnnl_runtime_error;
    if (use_dst_as_acc) {
        st = dnnl_brgemm_execute(brgemm, src_ptr, wei_packed_ptr,
                offsets.data(), dst_ptr, scratchpad_ptr);
    } else {
        st = dnnl_brgemm_execute_postops(brgemm, src_ptr, wei_packed_ptr,
                offsets.data(), acc_ptr, dst_ptr, scratchpad_ptr, attr_params);
    }
    return st;
}
#endif

// `init_memory_args` is responsible for:
// * Constructing all necessary `dnn_mem_t` objects needed by the brgemm kernel
//   for the main operation and attributes.
// * Stashing them with a proper exec_arg ID in a `mem_map` object.
// See a common version of `init_memory_args` comment for more details.
void init_memory_args(
        dnn_mem_map_t &mem_map, const prb_t *prb, kernel_args_t &kernel_args) {
    // Fuse batch size into K dimension which follows the library usage of the
    // kernel batch size setting.
    const dnnl_dims_t src_dims = {prb->m, prb->k * prb->batch_size};
    const dnnl_dims_t wei_dims = {prb->k * prb->batch_size, prb->n};

    const auto stag = prepare_src_format_string(kernel_args);
    const dims_t src_strides
            = stag.empty() ? dims_t {prb->get_lda(), 1} : dims_t {};
    dims_t dst_strides = {prb->get_ldd(), 1};
    dims_t acc_strides = prb->use_dst_as_acc() ? dst_strides : dims_t();

    auto src_md = dnn_mem_t::init_md(
            prb->ndims, src_dims, prb->src_dt(), stag, src_strides);

    auto dst_md = dnn_mem_t::init_md(
            prb->ndims, prb->dst_dims.data(), prb->dst_dt(), "", dst_strides);

    // Same as dst_md but with a pre-defined data type according to doc.
    auto acc_md = dnn_mem_t::init_md(prb->ndims, prb->dst_dims.data(),
            prb->acc_dt(), tag::abx, acc_strides);

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    // Create weights memory descriptor with kernel-friendly format.
    // Note: LDB is not passed here. This is because it's super difficult to
    // incorporate stride on top of blocking - oneDNN API doesn't provide any
    // calls to support both options together. Submemory descriptor, which is
    // the only one who can create such memory desc, can't return the size of
    // memory. Thus, it requires two memories and we need to pass a memory
    // handle from bigger one (where LDB is an actual dim value) to smaller, but
    // there's some reorder bug resulting in an error.
    const auto wtag = prepare_wei_format_string(prb, kernel_args);
    BENCHDNN_PRINT(6, "wtag: %s\n", wtag.c_str());

    auto wei_md = dnn_mem_t::init_md(prb->ndims, wei_dims, prb->wei_dt(), wtag);
    kernel_args.original_wei_md_size_ = dnnl_memory_desc_get_size(wei_md);

    // Prepare and assign extra for wei_md when s8s8 compensation, or source
    // zero point reduction values are needed.
    dnnl::impl::memory_extra_desc_t wei_md_extra {};
    wei_md_extra.flags = dnnl::impl::memory_extra_flags::none;
    bool need_s8s8_comp = prb->get_dt(SRC) == dnnl_s8 && prb->get_dt(WEI) == dnnl_s8;
    #if !defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(brg_aarch64)
        // SVE int8 (sdot/smmla) handles s8:s8 natively; the mmla weight layout
        // has no reorder with s8s8 compensation, so do not request it there.
        if (kernel_args.use_mmla_) need_s8s8_comp = false;
    #endif
    if (need_s8s8_comp) {
        wei_md_extra.flags
                |= dnnl::impl::memory_extra_flags::compensation_conv_s8s8;
        wei_md_extra.compensation_mask = 2; // N dimension
    }
    static_cast<dnnl_memory_desc_t>(wei_md)->extra = wei_md_extra;

    const bool need_src_comp = !prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if (need_src_comp) {
        wei_md_extra.flags |= dnnl::impl::memory_extra_flags::
                compensation_conv_asymmetric_src;
        wei_md_extra.asymm_compensation_mask = 2; // N dimension
    }
    static_cast<dnnl_memory_desc_t>(wei_md)->extra = wei_md_extra;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bia_md {};
    if (prb->bia_dt != dnnl_data_type_undef) {
        const dnnl_dims_t bia_dims = {1, prb->n};
        bia_md = dnn_mem_t::init_md(
                prb->ndims, bia_dims, prb->bia_dt, tag::abx);
    }
#else
    auto wei_md = dnn_mem_t::init_md(prb->ndims, wei_dims, prb->wei_dt(),
            prb->wtag, prb->strides[STRIDES_WEI]);
    const auto &wei_strides = query_md_strides(wei_md);

    // Note: packing routine transforms a plain user tensor into an internal
    // blocking format with various JIT kernels depending on user inputs.
    // While some kernels working fine with less memory, some don't, such as
    // transposed `ba` format.
    //
    // To supply enough memory for the transformation, the following logic
    // adjusts memory amount based on `simd_w` and `dt_multiplier` same way
    // what physical padding would do.
    //
    // `dt_multiplier` is required to form a 32-bit element which is a basic
    // unit of BRGeMM computations. There's the only outlier - f16 on ISAs where
    // packing is not expected, it acts as f32 there.
    const int dt_multiplier
            = prb->wei_dt() == dnnl_f16 && !kernel_args.need_pack_
            ? 1
            : static_cast<int>(4 / dnnl_data_type_size(prb->wei_dt()));

    int multiplier = 1;
    if (kernel_args.need_pack_) {
        multiplier = dt_multiplier;
        // Note (impl detail): transposed kernel wants 64 bytes for K dim.
        if (wei_strides[0] < wei_strides[1]) {
            // Though `simd_w` is not necessarily `16` on all ISAs, it's for
            // simplicity.
            constexpr int simd_w = 16;
            multiplier *= simd_w;
        }
    }

    const dnnl_dim_t k_rounded = multiplier * div_up(prb->k, multiplier);
    const dnnl_dims_t wei_packed_dims = {k_rounded * prb->batch_size, prb->n};
    dims_t wei_packed_strides = {prb->get_ldb(), 1};
    auto wei_packed_md = dnn_mem_t::init_md(
            prb->ndims, wei_packed_dims, prb->wei_dt(), "", wei_packed_strides);
#endif

    dnnl_dim_t scratchpad_size
            = static_cast<dnnl_dim_t>(kernel_args.scratchpad_size_);
    int ndims = scratchpad_size ? 1 : 0;
    dnnl_data_type_t dt = scratchpad_size ? dnnl_u8 : dnnl_data_type_undef;
    dnnl_dims_t scratchpad_dims = {scratchpad_size};
    auto scratchpad_md
            = dnn_mem_t::init_md(ndims, scratchpad_dims, dt, tag::abx);

    const auto &test_engine = get_test_engine();

    mem_map.emplace(
            DNNL_ARG_SRC, dnn_mem_t(src_md, test_engine, /* prefill = */ true));
    mem_map.emplace(DNNL_ARG_WEIGHTS,
            dnn_mem_t(wei_md, test_engine, /* prefill = */ true));
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    if (prb->bia_dt != dnnl_data_type_undef) {
        // Need condition to extract bias pointer based on presence in map
        // condition.
        mem_map.emplace(DNNL_ARG_BIAS,
                dnn_mem_t(bia_md, test_engine, /* prefill = */ true));
    }
#else
    mem_map.emplace(DNNL_ARG_WEIGHTS_1,
            dnn_mem_t(wei_packed_md, test_engine, /* prefill = */ true));
#endif
    mem_map.emplace(DNNL_ARG_DST_1,
            dnn_mem_t(acc_md, test_engine, /* prefill = */ true));
    mem_map.emplace(
            DNNL_ARG_DST, dnn_mem_t(dst_md, test_engine, /* prefill = */ true));
    if (scratchpad_size > 0) {
        // Need condition to extract scratchpad pointer based on presence in map
        // condition.
        mem_map.emplace(DNNL_ARG_SCRATCHPAD,
                dnn_mem_t(scratchpad_md, test_engine, /* prefill = */ true));
    }

    // Binary post-op.
    const auto &po = prb->attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (!e.is_binary_kind()) continue;

        int po_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;
        const auto &b = e.binary;
        int ndims = 2;
        dims_t dims = prb->dst_dims;

        const int mask = b.mask_input == attr_t::mask_input_t::mask
                ? b.mask
                : attr_t::policy2mask(po_arg, b.policy, ndims, dnnl_matmul);

        switch (mask) {
            case 0: dims = {1, 1}; break;
            case 1: dims = {dims[0], 1}; break;
            case 2: dims = {1, dims[1]}; break;
            // Masks can be bigger than values above depending on the policy.
            default: break;
        }

        auto po_md
                = dnn_mem_t::init_md(ndims, dims.data(), b.src1_dt, tag::abx);
        mem_map.emplace(
                po_arg, dnn_mem_t(po_md, test_engine, /* prefill = */ true));
    }

    if (!prb->attr.scales.is_def()) {
        const auto &sc = prb->attr.scales;
        static const std::vector<int> supported_args {
                DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        for (const auto &exec_arg : supported_args) {
            if (sc.is_def(exec_arg)) continue;

            const int exec_sc_arg = DNNL_ARG_ATTR_SCALES | exec_arg;
            dims_t dims = {};
            int64_t ndims = 1;
            const auto mask = sc.get_mask(
                    exec_arg, dnnl_matmul, 2, /* has_groups = */ false);

            if (mask > 0) {
                const auto &md = mem_map.at(exec_arg).md_;
                dims = md2dims(md, mask, false);
                ndims = static_cast<int>(dims.size());
            } else {
                dims = {1};
                ndims = 1;
            }
            const auto dt = sc.get(exec_arg).dt;
            auto scales_md
                    = dnn_mem_t::init_md(ndims, dims.data(), dt, tag::abx);
            mem_map.emplace(exec_sc_arg,
                    dnn_mem_t(scales_md, test_engine, /* prefill = */ true));
        }
    }

    if (!prb->attr.zero_points.is_def()) {
        const auto &zp = prb->attr.zero_points;
        static const std::vector<int> supported_args {
                DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        for (const auto &exec_arg : supported_args) {
            if (zp.is_def(exec_arg)) continue;

            const int exec_zp_arg = DNNL_ARG_ATTR_ZERO_POINTS | exec_arg;
            dims_t dims = {};
            int64_t ndims = 1;
            const auto mask
                    = zp.get_mask(exec_arg, dnnl_matmul, /* ndims = */ 2);

            if (mask > 0) {
                const auto &md = mem_map.at(exec_arg).md_;
                dims = md2dims(md, mask, false);
                ndims = static_cast<int>(dims.size());
            } else {
                dims = {1};
                ndims = 1;
            }
            const auto dt = zp.get(exec_arg).dt;
            auto zp_md = dnn_mem_t::init_md(ndims, dims.data(), dt, tag::abx);
            mem_map.emplace(exec_zp_arg,
                    dnn_mem_t(zp_md, test_engine, /* prefill = */ true));
        }
    }
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, const kernel_args_t &kernel_args, res_t *res) {

    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    const bool need_fill_acc
            = (prb->beta != 0) || kernel_args.generate_skip_accumulation_;

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }

        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST: {
                const auto &po = prb->attr.post_ops;
                const int sum_idx = po.find(attr_t::post_ops_t::SUM);
                if (sum_idx >= 0) {
                    SAFE(fill_data(DST, exec_arg, prb, cfg, mem, ref_mem, res),
                            WARN);
                }
            } break;
            case DNNL_ARG_DST_1: {
                if (need_fill_acc) {
                    SAFE(fill_data(DST, exec_arg, prb, cfg, mem, ref_mem, res),
                            WARN);
                }
            } break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }
    }

    if (need_fill_acc) {
        // Beta requires same values for reference and the kernel.
        if (prb->use_dst_as_acc()) {
            auto &acc_fp = ref_mem_map.at(DNNL_ARG_DST_1);
            auto &dst_fp = ref_mem_map.at(DNNL_ARG_DST);
            auto &dst_dt = mem_map.at(DNNL_ARG_DST);

            SAFE(dst_fp.reorder(acc_fp, res), WARN);
            SAFE(dst_dt.reorder(dst_fp, res), WARN);
        }
    }

    // A hack to pass brgemm attributes to reference execution since some
    // members change the computation flow for correctness validation.
    dnnl_dims_t dims = {1};
    auto workspace_md = dnn_mem_t::init_md(1, dims, dnnl_u8, tag::abx);
    ref_mem_map.emplace(DNNL_ARG_WORKSPACE,
            dnn_mem_t(workspace_md, ref_engine, /* prefill = */ false,
                    {false, (void *)&kernel_args.generate_skip_accumulation_}));
    ref_mem_map.at(DNNL_ARG_WORKSPACE).map();

    return OK;
}

int scales_post_processing(dnn_mem_map_t &mem_map) {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    // dst scales are applied inversed in the brgemm kernel. However, for x64
    // it's enough to have a single value, though we keep v16 just for the sake
    // of a single code path with aarch64.
    const bool has_dst_scale
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto replace_mem_to_v16
            = [&](dnnl_data_type_t dt, int exec_arg, float val) {
        dims_t dims = {16};
        auto new_md = dnn_mem_t::init_md(1, dims.data(), dt, tag::abx);
        dnn_mem_t new_m(new_md, get_test_engine(), /* prefill = */ true);
        if (!new_m.is_mapped()) new_m.map();
        for (int64_t i = 0; i < new_m.nelems(); i++) {
            new_m.set_elem(i, val);
        }
        mem_map[DNNL_ARG_ATTR_SCALES | exec_arg] = std::move(new_m);
    };

    if (has_dst_scale) {
        const auto &dst_scales_m
                = mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
        assert(dst_scales_m.nelems() == 1);
        // Create a v16 dst scales memory and bcast inversed dst value there.
        replace_mem_to_v16(dst_scales_m.dt(), DNNL_ARG_DST,
                1.f / dst_scales_m.get_elem(0));
    }

#if defined(brg_aarch64)
    // Internal API has specific implementation details w.r.t. scales.
    // If any of source or weights scales present in the descriptor, then the
    // kernel expects to get a vector of 16 float values (v16) of "fused" scale
    // values (src * wei) under the pointer passed in brgemm_post_ops_data_t
    // struct.
    // However, if weights scales is per channel, then the kernel expects just
    // `N` values, even if `N < 16`.
    // Same applies for a destination scale. Due to it's handled separately from
    // source and weights, and must be a single value, it must be a v16 memory.
    //
    // To smoothly take care of this detail, the code below will **always**
    // update WEIGHTS scale (even if they are not present) with a proper memory
    // of 16 or N elements, depending on the case.
    // It will update destination memory to contain 16 elements as well.

    const bool has_src_scale
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const bool has_wei_scale
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);

    if (has_wei_scale) {
        const auto &wei_scales_m
                = mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
        // First, update the values...
        if (has_src_scale) {
            const auto &src_scales_m
                    = mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
            assert(src_scales_m.nelems() == 1);
            const float src_val = src_scales_m.get_elem(0);
            for (int64_t i = 0; i < wei_scales_m.nelems(); i++) {
                float val = wei_scales_m.get_elem(i) * src_val;
                wei_scales_m.set_elem(i, val);
            }
        }
        // Second, update memory for a single scale.
        if (wei_scales_m.nelems() == 1) {
            replace_mem_to_v16(wei_scales_m.dt(), DNNL_ARG_WEIGHTS,
                    wei_scales_m.get_elem(0));
        }
    } else if (has_src_scale) {
        const auto &src_scales_m
                = mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        assert(src_scales_m.nelems() == 1);
        // Create a v16 weights scales memory and put src value there.
        replace_mem_to_v16(
                src_scales_m.dt(), DNNL_ARG_WEIGHTS, src_scales_m.get_elem(0));
    }

#endif
#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)
    // ukernel API takes split pointers for scales, no need to update them on
    // user level.
#endif
    return OK;
}

int binary_post_op_preprocessing(
        std::vector<const void *> &binary_po_v, const dnn_mem_map_t &mem_map) {
    // Preprocessing must happen in two stages:
    // 1. Collect all arguments values and sort them.
    // 2. Insert memory pointers correspondent to arguments in order to satisfy
    //    the kernel expectations.
    std::set<int> arg_vals;
    for (const auto &map_entry : mem_map) {
        const int exec_arg = map_entry.first;

        const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
        const bool is_post_ops_arg = (exec_arg & post_ops_range);
        if (!is_post_ops_arg) continue;

        arg_vals.insert(exec_arg);
    }

    binary_po_v.reserve(arg_vals.size());
    for (const auto &set_entry : arg_vals) {
        void *handle = mem_map.at(set_entry).get_mapped_pointer<void>();
        binary_po_v.push_back(handle);
    }

    return OK;
}

int init_hw_config(const kernel_args_t &kernel_args) {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#if defined(brg_x64)
    if (kernel_args.need_tile_config_) {
        DNN_SAFE(namespace_impl::amx_tile_configure(kernel_args.palette), WARN);
    }
#endif
#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)
    DNN_SAFE(dnnl_brgemm_set_hw_context(kernel_args.brgemm_), WARN);
#endif
    return OK;
}

int release_hw_config(const kernel_args_t &kernel_args) {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#if defined(brg_x64)
    if (kernel_args.need_tile_config_) {
        DNN_SAFE(namespace_impl::amx_tile_release(), WARN);
    }
#endif
#endif
    return OK;
}

// `release_hw_config` and `brgemm_finalize` are doing the same thing -
// releasing the hw resources they allocated. The difference is in the
// implementation side - ukernel has lazy initialization and would reset the
// state per `set_hw_config` call while internal API doesn't - it just sets
// the new state unconditionally. Because of laziness, the test wants to ensure
// resetting is correct and, thus, releasing is done once - at the end of the
// suite, while for internal API it is done after each case.
int brgemm_finalize() {
#if defined(DNNL_EXPERIMENTAL_UKERNEL)
    DNN_SAFE(dnnl_brgemm_release_hw_context(), WARN);
#endif
    return OK;
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    skip_start(res);
    if (res->state == SKIPPED) return OK;

    // Need this here as brgemm has no primitive creation step
    prb->skip_invalid(res);
    if (res->state == SKIPPED) return OK;

    kernel_args_t kernel_args(prb);
    SAFE(init_kernel(kernel_args, res), WARN);
    if (res->state == SKIPPED) return OK;
    if (bench_mode == bench_mode_t::init) return res->state = INITIALIZED, OK;

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    auto brgemm_kernel = make_benchdnn_dnnl_wrapper(kernel_args.brgemm_kernel_);
#else
    auto brgemm = make_benchdnn_dnnl_wrapper(kernel_args.brgemm_);
    auto transform = make_benchdnn_dnnl_wrapper(kernel_args.transform_);
#endif

    dnn_mem_map_t mem_map, ref_mem_map;
    init_memory_args(mem_map, prb, kernel_args);
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prb, kernel_args, res),
            WARN));

    // "Library" args are needed to get dst for comparison.
    // "Reference" are used as usual.
    args_t args(mem_map), ref_args(ref_mem_map);

    // The implementation memory must be mapped to setup point arguments for
    // brgemm implementation call. This assumes that mapping is effectively a
    // no-op on the target device.
    for (auto &kv : mem_map) {
        if (!kv.second.is_mapped()) kv.second.map();
    }

    const char *src_ptr = (const char *)mem_map.at(DNNL_ARG_SRC);
    const char *wei_ptr = (const char *)mem_map.at(DNNL_ARG_WEIGHTS);
    char *acc_ptr = (char *)mem_map.at(DNNL_ARG_DST_1);
    char *dst_ptr = (char *)mem_map.at(DNNL_ARG_DST);
    if (prb->use_dst_as_acc()) acc_ptr = dst_ptr;

    SAFE(scales_post_processing(mem_map), WARN);

    std::vector<const void *> binary_po_v;
    SAFE(binary_post_op_preprocessing(binary_po_v, mem_map), WARN);

#if defined(brg_x64)
    const void *src_scales_ptr
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC)
            ? (const void *)mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC)
            : nullptr;
    const void *wei_scales_ptr
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
            ? (const void *)mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
            : nullptr;
#endif
    const float *dst_scales_ptr
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
            ? (const float *)mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
            : nullptr;

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    std::vector<namespace_impl::brgemm_batch_element_t> v_batch_element(
            prb->batch_size);
    for (size_t i = 0; i < v_batch_element.size(); i++) {
        if (prb->batch_kind == "addr") {
            v_batch_element[i].ptr.A
                    = src_ptr + i * prb->get_src_batch_offset();
            v_batch_element[i].ptr.B
                    = wei_ptr + i * prb->get_wei_batch_offset();
        } else if (prb->batch_kind == "offs") {
            v_batch_element[i].offset.A = i * prb->get_src_batch_offset();
            v_batch_element[i].offset.B = i * prb->get_wei_batch_offset();
        }
    }

    // For internal API, scales are combined. See `scales_post_processing`.
#if defined(brg_aarch64)
    const float *scales_ptr
            = mem_map.count(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
            ? (const float *)mem_map.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
            : nullptr;
#endif
    const int32_t *dst_zp_ptr
            = mem_map.count(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
            ? (const int32_t *)mem_map.at(
                      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
            : nullptr;
    int32_t zp_a_val = mem_map.count(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC)
            ? *(const int32_t *)mem_map.at(
                      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC)
            : 0;
    const char *bia_dt_ptr = mem_map.count(DNNL_ARG_BIAS)
            ? (const char *)mem_map.at(DNNL_ARG_BIAS)
            : nullptr;

    const auto &wei_md = mem_map.at(DNNL_ARG_WEIGHTS).md_;
    const auto &wei_md_extra = static_cast<dnnl_memory_desc_t>(wei_md)->extra;
    // This relies on an internal knowledge of how compensation is implemented.
    const size_t wei_offset_s8s8 = kernel_args.original_wei_md_size_;
    const size_t wei_offset_zp = wei_offset_s8s8
            + ((wei_md_extra.flags
                       & dnnl::impl::memory_extra_flags::compensation_conv_s8s8)
                            ? prb->get_ldb() * sizeof(int32_t)
                            : 0);
    char *src_comp_ptr = const_cast<char *>(wei_ptr) + wei_offset_zp;

#if defined(brg_x64)
    namespace_impl::brgemm_post_ops_data_t post_ops_data(
            /* bias */ bia_dt_ptr,
            /* binary_post_ops_rhs */ binary_po_v.data(),
            /* oc_logical_off */ 0, /* dst_row_logical_off */ 0,
            // TODO: though the field is called `data_C_ptr_`, this is a
            // misleading name since actually dst_ptr must be used there to
            // have binary injector working for per_tensor policy.
            /* data_C_ptr_ */ dst_ptr, /* first_mb_matrix_addr_off */ 0,
            /* a_zp_compensations */ src_comp_ptr,
            /* b_zp_compensations */ nullptr,
            /* c_zp_values */ dst_zp_ptr,
            /* skip_accumulation */
            kernel_args.generate_skip_accumulation_,
            /* zp_a_val */ zp_a_val,
            /* do_only_comp */ false,
            /* do_only_zp_a_val */ false,
            /* src_scales */ src_scales_ptr,
            /* wei_scales */ wei_scales_ptr,
            /* dst_scales */ dst_scales_ptr);
#elif defined(brg_aarch64)
    namespace_impl::brgemm_post_ops_data_t post_ops_data(
            /* bias */ bia_dt_ptr,
            /* scales */ scales_ptr,
            /* binary_post_ops_rhs */ binary_po_v.data(),
            /* oc_logical_off */ 0, /* dst_row_logical_off */ 0,
            // TODO: though the field is called `data_C_ptr_`, this is a
            // misleading name since actually dst_ptr must be used there to
            // have binary injector working for per_tensor policy.
            /* data_C_ptr_ */ dst_ptr, /* first_mb_matrix_addr_off */ 0,
            /* a_zp_compensations */ src_comp_ptr,
            /* b_zp_compensations */ nullptr,
            /* c_zp_values */ dst_zp_ptr,
            /* skip_accumulation */
            kernel_args.generate_skip_accumulation_,
            /* zp_a_val */ zp_a_val,
            /* do_only_comp */ false,
            /* do_only_zp_a_val */ false,
            /* dst_scales */ dst_scales_ptr);
#endif

    // Note: hardware lacking native s8s8 support expects compensation buffer
    // passed through a scratchpad argument in postops execution call.
    const bool has_scratchpad = mem_map.count(DNNL_ARG_SCRATCHPAD);
    const bool need_hidden_compensation = !has_scratchpad
            && prb->get_dt(SRC) == dnnl_s8 && prb->get_dt(WEI) == dnnl_s8;
    char *scratchpad_ptr = need_hidden_compensation
            ? (const_cast<char *>(wei_ptr) + wei_offset_s8s8)
            : has_scratchpad ? (char *)mem_map.at(DNNL_ARG_SCRATCHPAD)
                             : nullptr;

#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)
    char *wei_packed_ptr = (char *)mem_map.at(DNNL_ARG_WEIGHTS_1);

    char *scratchpad_ptr = mem_map.count(DNNL_ARG_SCRATCHPAD)
            ? (char *)mem_map.at(DNNL_ARG_SCRATCHPAD)
            : nullptr;

    if (kernel_args.need_pack_) {
        DNN_SAFE(dnnl_transform_execute(transform, wei_ptr, wei_packed_ptr),
                WARN);
    } else {
        const auto &wei_dt = mem_map.at(DNNL_ARG_WEIGHTS);
        auto &wei_packed_dt = mem_map.at(DNNL_ARG_WEIGHTS_1);
        SAFE(wei_packed_dt.reorder(wei_dt, res), WARN);
    }

    std::vector<dnnl_dim_t> offsets(2 * prb->batch_size);
    for (dnnl_dim_t i = 0; i < prb->batch_size; i++) {
        offsets[2 * i + 0] = i * prb->get_src_batch_offset();
        offsets[2 * i + 1] = i * prb->get_wei_batch_offset();
    }

    dnnl_ukernel_attr_params_t attr_params_ptr;
    DNN_SAFE(dnnl_ukernel_attr_params_create(&attr_params_ptr), WARN);
    auto attr_params = make_benchdnn_dnnl_wrapper(attr_params_ptr);
    DNN_SAFE(dnnl_ukernel_attr_params_set_post_ops_args(
                     attr_params, binary_po_v.data()),
            WARN);
#if defined(brg_x64)
    DNN_SAFE(dnnl_ukernel_attr_params_set_A_scales(attr_params, src_scales_ptr),
            WARN);
    DNN_SAFE(dnnl_ukernel_attr_params_set_B_scales(attr_params, wei_scales_ptr),
            WARN);
#endif
    DNN_SAFE(dnnl_ukernel_attr_params_set_D_scales(attr_params, dst_scales_ptr),
            WARN);
#endif

    SAFE(init_hw_config(kernel_args), WARN);

    if (has_bench_mode_bit(mode_bit_t::exec)
            && !has_bench_mode_bit(mode_bit_t::perf)) {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
        if (prb->batch_kind == "addr") {
            brgemm_kernel_execute_postops(brgemm_kernel, prb->batch_size,
                    v_batch_element.data(), acc_ptr, dst_ptr, post_ops_data,
                    scratchpad_ptr);
        } else if (prb->batch_kind == "offs") {
            brgemm_kernel_execute_postops(brgemm_kernel, prb->batch_size,
                    src_ptr, wei_ptr, v_batch_element.data(), acc_ptr, dst_ptr,
                    post_ops_data, scratchpad_ptr);
        }
#else /* !defined(DNNL_EXPERIMENTAL_UKERNEL) */
        // `prb->use_dst_as_acc()=true` will make `dst_ptr=acc_ptr` and rest should
        // be handled by API.
        DNN_SAFE(dnnl_brgemm_execute_postops(brgemm, src_ptr, wei_packed_ptr,
                         offsets.data(), acc_ptr, dst_ptr, scratchpad_ptr,
                         attr_params),
                WARN);
#endif
        res->state = EXECUTED;
        check_correctness(prb, {DST}, args, ref_args, compute_ref, setup_cmp,
                res, prb->dir);
    }

    // Create a bind to match internals to run performance measurements.
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
    perf_function_t perf_func = std::bind(brgemm_kernel_execute_postops_wrapper,
            kernel_args.brgemm_kernel_, prb->batch_kind, prb->batch_size,
            src_ptr, wei_ptr, v_batch_element.data(), acc_ptr, dst_ptr,
            post_ops_data, scratchpad_ptr, std::placeholders::_1,
            std::placeholders::_2);
#else // !defined(DNNL_EXPERIMENTAL_UKERNEL)
    perf_function_t perf_func = std::bind(brgemm_kernel_execute_postops_wrapper,
            kernel_args.brgemm_, prb->use_dst_as_acc(), src_ptr, wei_packed_ptr,
            offsets, acc_ptr, dst_ptr, scratchpad_ptr, attr_params_ptr,
            std::placeholders::_1, std::placeholders::_2);
#endif

    measure_perf(prb->ctx_exe, res, perf_func, args);

    SAFE(release_hw_config(kernel_args), WARN);

    return OK;
}

#else

// For builadability of non-x64 configuration.
int brgemm_finalize() {
    return OK;
}

int doit(const prb_t *prb, res_t *res) {
    return OK;
}

#endif

} // namespace brgemm
