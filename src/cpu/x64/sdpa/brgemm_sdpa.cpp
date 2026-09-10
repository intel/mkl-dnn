/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "common/host_scalar_memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/sdpa/brgemm_sdpa.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Fills bp_ directly from this pd's memory descriptors (sdpa_desc_t), the
// same way GPU's micro_fwd_t::pd_t::init() fills its own params struct from
// desc() -- NOT by reusing the graph pattern-matching / subgraph-parsing
// logic that the fused BRGEMM *graph kernel* (sdp_fused_brgemm_blocked_kernel_t)
// uses to build the very same kind of struct from a lowered partition.
status_t brgemm_sdpa_fwd_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace status;

    // The pd stays kernel-free: it only derives the plain compute params and
    // sizes the scratchpad (via the driver's JIT-free configure()); the BRGEMM
    // kernels are compiled later in brgemm_sdpa_fwd_t::init(engine).
    UNUSED(engine);
    VDISPATCH_SDPA(is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_SDPA(desc()->prop_kind == prop_kind::forward_inference,
            "training-forward (softmax stats output) is not supported");

    const memory_desc_wrapper qry_mdw(desc()->qry_md());
    const memory_desc_wrapper key_mdw(desc()->key_md());
    const memory_desc_wrapper val_mdw(desc()->val_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    VDISPATCH_SDPA(utils::everyone_is(sdpa_pd_t::ndims, qry_mdw.ndims(),
                           key_mdw.ndims(), val_mdw.ndims(), dst_mdw.ndims()),
            VERBOSE_SHAPE_RESTRICTION
            ": qry(%d) key(%d) val(%d) and dst(%d) must be 4d",
            qry_mdw.ndims(), key_mdw.ndims(), val_mdw.ndims(), dst_mdw.ndims());
    VDISPATCH_SDPA(
            utils::everyone_is(true, qry_mdw.is_plain(), key_mdw.is_plain(),
                    val_mdw.is_plain(), dst_mdw.is_plain()),
            VERBOSE_UNSUPPORTED_TAG);

    const data_type_t dt = qry_mdw.data_type();
    VDISPATCH_SDPA(utils::one_of(dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_SDPA(key_mdw.data_type() == dt && val_mdw.data_type() == dt
                    && dst_mdw.data_type() == dt,
            "query, key, value and dst data types must match");

    VDISPATCH_SDPA(kq_acc_dt() == f32 && vs_acc_dt() == f32,
            "only f32 accumulation is currently supported");
    VDISPATCH_SDPA(!with_key_scales() && !with_value_scales() && !with_key_zp()
                    && !with_value_zp(),
            "quantized key/value tensors are not yet supported");
    VDISPATCH_SDPA(!with_causal_mask(),
            "an implicit causal mask is not yet supported");
    VDISPATCH_SDPA(IMPLICATION(with_attn_mask(),
                           desc()->mask_type == attn_mask_type::buffer),
            "only an explicit buffer attention mask is supported");
    VDISPATCH_SDPA(
            utils::one_of(desc()->softmax_alg, alg_kind::softmax_accurate,
                    alg_kind::softmax_accurate_inf_as_zero),
            "unsupported softmax algorithm");
    VDISPATCH_SDPA(desc()->num_kv_heads() > 0
                    && desc()->num_q_heads() % desc()->num_kv_heads() == 0,
            "the number of query heads must be a multiple of the number of "
            "key/value heads");

    if (with_attn_mask()) {
        const memory_desc_wrapper mask_mdw(desc()->attn_mask_md());
        VDISPATCH_SDPA(mask_mdw.ndims() == sdpa_pd_t::ndims,
                VERBOSE_SHAPE_RESTRICTION ": attn_mask(%d) must be 4d",
                mask_mdw.ndims());
        VDISPATCH_SDPA(mask_mdw.is_plain(), VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_SDPA(utils::one_of(mask_mdw.dims()[mask_q_index],
                               desc()->queries(), 1),
                VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
        VDISPATCH_SDPA(mask_mdw.dims()[mask_k_index] == desc()->keys(),
                VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
        VDISPATCH_SDPA(
                mask_mdw.data_type() == dt || mask_mdw.data_type() == f32,
                "the attention mask data type must match qry/dst or be f32");
    }

    if (with_attn_scale()) {
        const memory_desc_wrapper scale_mdw(desc()->scale_md());
        VDISPATCH_SDPA(scale_mdw.data_type() == f32,
                "only an f32 attention scale is currently supported");
    }

    VDISPATCH_SDPA(
            sdpa_fwd_pd_t::set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    // sdp_fused_driver_t never materializes the full [seq_q x seq_kv] score
    // matrix, but is f32-only and has no attention-mask post-op yet;
    // sdp_blocked_driver_t covers bf16/f16 and an additive mask. Pick fused
    // whenever the shapes/dtypes allow it, unless overridden for
    // debugging/perf comparisons via ONEDNN_SDPA_IMPL={blocked,fused,auto}
    // (auto, the default, is the capability-based choice above).
    const bool fused_capable = dt == f32 && !with_attn_mask();
    const std::string forced = getenv_string_user("SDPA_IMPL");
    VDISPATCH_SDPA(utils::one_of(forced, std::string(), std::string("auto"),
                           std::string("blocked"), std::string("fused")),
            "unknown ONEDNN_SDPA_IMPL value '%s'", forced.c_str());
    if (forced == "fused") {
        VDISPATCH_SDPA(fused_capable,
                "the fused SDPA driver was forced via ONEDNN_SDPA_IMPL but "
                "does not support this data type or attention mask");
        kind_ = sdpa_driver_kind_t::fused;
    } else if (forced == "blocked") {
        kind_ = sdpa_driver_kind_t::blocked;
    } else {
        kind_ = fused_capable ? sdpa_driver_kind_t::fused
                              : sdpa_driver_kind_t::blocked;
    }

    auto scratchpad = scratchpad_registry().registrar();

    if (kind_ == sdpa_driver_kind_t::fused) {
        fp_ = sdp_fused_params_t();
        fp_.ndims = sdpa_pd_t::ndims;
        fp_.batch = desc()->batch();
        fp_.num_head_q = desc()->num_q_heads();
        fp_.group_head = desc()->num_q_heads() / desc()->num_kv_heads();
        fp_.seq_q = desc()->queries();
        fp_.seq_kv = desc()->keys();
        fp_.head_size_qk = desc()->head_size();
        fp_.head_size_v = desc()->values();
        fp_.q_strides.assign(
                qry_mdw.strides(), qry_mdw.strides() + sdpa_pd_t::ndims);
        fp_.k_strides.assign(
                key_mdw.strides(), key_mdw.strides() + sdpa_pd_t::ndims);
        fp_.v_strides.assign(
                val_mdw.strides(), val_mdw.strides() + sdpa_pd_t::ndims);
        fp_.o_strides.assign(
                dst_mdw.strides(), dst_mdw.strides() + sdpa_pd_t::ndims);
        fp_.has_select = false;
        fp_.select_fusiable = false;

        // Size the scratchpad from a throwaway driver: configure() runs only
        // the JIT-free arithmetic (KV tiling, per-thread scratch), so the pd
        // never compiles a kernel.
        sdp_fused_driver_t sizer;
        CHECK(sizer.configure(fp_));
        nthr_ = sizer.nthr();
        scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
                sizer.scratch_total(nthr_), 1, 64);
        return status::success;
    }

    bp_ = sdp_blocked_params_t();
    bp_.ndims = sdpa_pd_t::ndims;
    bp_.batch = desc()->batch();
    bp_.num_head_q = desc()->num_q_heads();
    bp_.group_head = desc()->num_q_heads() / desc()->num_kv_heads();
    bp_.seq_q = desc()->queries();
    bp_.seq_kv = desc()->keys();
    bp_.head_size_qk = desc()->head_size();
    bp_.head_size_v = desc()->values();
    bp_.mm_dt = dt;
    bp_.out_dt = dt;
    bp_.q_strides.assign(
            qry_mdw.strides(), qry_mdw.strides() + sdpa_pd_t::ndims);
    bp_.k_strides.assign(
            key_mdw.strides(), key_mdw.strides() + sdpa_pd_t::ndims);
    bp_.v_strides.assign(
            val_mdw.strides(), val_mdw.strides() + sdpa_pd_t::ndims);
    bp_.o_strides.assign(
            dst_mdw.strides(), dst_mdw.strides() + sdpa_pd_t::ndims);
    // key_md()'s logical dim order is fixed by the pd contract to
    // [batch, kv_heads, head_size, keys] (keys() reads the *last* dim), i.e.
    // head_size is already the row axis and keys the inner axis -- the
    // "non-transposed" orientation the driver expects.
    bp_.mm1_transpose_b = false;
    bp_.has_select = false;
    bp_.select_fusiable = false;
    bp_.softmax_inf_as_zero
            = desc()->softmax_alg == alg_kind::softmax_accurate_inf_as_zero;

    // mm1 post-op chain: scale (binary-mul, scalar rhs) then the additive
    // attention mask (binary-add, tensor rhs); soft-cap will be appended here
    // once the pd gains support for it.
    if (with_attn_scale()) {
        sdp_mm1_post_op_t sc;
        sc.alg = alg_kind::binary_mul;
        sc.is_binary = true;
        sc.rhs_is_scalar = true;
        sc.rhs_dt = f32;
        bp_.mm1_post_ops.push_back(sc);
    }
    if (with_attn_mask()) {
        const memory_desc_wrapper mask_mdw(desc()->attn_mask_md());
        sdp_mm1_post_op_t mk;
        mk.alg = alg_kind::binary_add;
        mk.is_binary = true;
        mk.rhs_is_scalar = false;
        mk.rhs_dt = mask_mdw.data_type();
        mk.rhs_dims.assign(mask_mdw.dims(), mask_mdw.dims() + sdpa_pd_t::ndims);
        mk.rhs_strides.assign(
                mask_mdw.strides(), mask_mdw.strides() + sdpa_pd_t::ndims);
        bp_.mm1_post_ops.push_back(mk);
    }

    // Size the scratchpad from a throwaway driver: configure() runs only the
    // JIT-free arithmetic + the AMX palette/wsp read from the finalized BRGEMM
    // descriptors, so the pd never compiles a kernel.
    sdp_blocked_driver_t sizer;
    CHECK(sizer.configure(bp_));
    nthr_ = sizer.nthr();
    scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
            sizer.scratch_total(nthr_), 1, 64);

    return status::success;
}

status_t brgemm_sdpa_fwd_t::init(engine_t *engine) {
    if (pd()->driver_kind() == sdpa_driver_kind_t::fused) {
        fused_driver_ = std::make_shared<sdp_fused_driver_t>();
        return fused_driver_->init(pd()->fused_params(), engine);
    }
    blocked_driver_ = std::make_shared<sdp_blocked_driver_t>();
    return blocked_driver_->init(pd()->blocked_params(), engine);
}

status_t brgemm_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto *scratch = ctx.get_scratchpad_grantor().get<char>(
            memory_tracking::names::key_sdpa_brgemm_buffer);

    auto *q = CTX_IN_MEM(const void *, DNNL_ARG_QUERIES);
    auto *k = CTX_IN_MEM(const void *, DNNL_ARG_KEYS);
    auto *v = CTX_IN_MEM(const void *, DNNL_ARG_VALUES);
    auto *out = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    // The scale is applied as a multiply; invert_scale means the user stored a
    // divisor, so multiply by its reciprocal (matches the graph kernel). The
    // local outlives the driver call (blocked passes &scale_val by pointer).
    // The graph backend delivers the scale as a host scalar (kept on the host);
    // a regular tensor scale is read from its device buffer instead.
    float scale_val = 1.0f;
    if (pd()->with_attn_scale()) {
        if (pd()->with_host_scale()) {
            const auto &scale_storage = CTX_IN_STORAGE(DNNL_ARG_SCALE);
            const auto *host_storage
                    = utils::downcast<const host_scalar_memory_storage_t *>(
                            &scale_storage);
            CHECK(host_storage->get_scalar_value(
                    &scale_val, sizeof(scale_val)));
        } else {
            scale_val = *CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
        }
        if (pd()->desc()->invert_scale) scale_val = 1.0f / scale_val;
    }

    if (pd()->driver_kind() == sdpa_driver_kind_t::fused) {
        sdp_fused_run_args_t args;
        args.q = q;
        args.k = k;
        args.v = v;
        args.cond = nullptr;
        args.out = out;
        args.scale = scale_val;
        args.fill = 0.0f;
        return fused_driver_->execute(args, scratch, pd()->nthr());
    }

    sdp_blocked_run_args_t args;
    args.q = q;
    args.k = k;
    args.v = v;
    args.cond = nullptr;
    args.out = out;
    args.fill = 0.0f;
    // rhs base pointers for the mm1 binary post-ops, in the same order they
    // were appended to bp_.mm1_post_ops: scale (scalar) then attention mask.
    if (pd()->with_attn_scale()) args.mm1_post_op_rhs.push_back(&scale_val);
    if (pd()->with_attn_mask())
        args.mm1_post_op_rhs.push_back(
                CTX_IN_MEM(const void *, DNNL_ARG_ATTN_MASK));
    return blocked_driver_->execute(args, scratch, pd()->nthr());
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
