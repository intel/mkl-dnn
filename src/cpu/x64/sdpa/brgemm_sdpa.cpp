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

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/sdpa/brgemm_sdpa.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Fills bp_ directly from this pd's memory descriptors (sdpa_desc_t), the
// same way GPU's micro_fwd_t::pd_t::init() fills its own params struct from
// desc() -- not from any graph pattern-matching / subgraph-parsing logic.
status_t brgemm_sdpa_fwd_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace status;

    // The pd stays kernel-free: it only derives the compute conf and sizes the
    // scratchpad (via the JIT-free configure()); the BRGEMM kernels are compiled
    // later in brgemm_sdpa_fwd_t::init(engine).
    UNUSED(engine);
    VDISPATCH_SDPA(is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_SDPA(desc()->prop_kind == prop_kind::forward_inference,
            "training-forward (softmax stats output) is not supported");

    // Both strategies JIT a vectorized softmax epilogue and run BRGEMM, which
    // need at least AVX2.
    VDISPATCH_SDPA(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);

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

    VDISPATCH_SDPA(qry_mdw.strides()[sdpa_pd_t::ndims - 1] == 1,
            "the query head-size axis must be contiguous");
    VDISPATCH_SDPA(key_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
                    || key_mdw.strides()[sdpa_pd_t::ndims - 2] == 1,
            "one key matrix axis must be contiguous");
    VDISPATCH_SDPA(val_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
                    || val_mdw.strides()[sdpa_pd_t::ndims - 2] == 1,
            "one value matrix axis must be contiguous");

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
                           with_buffer_mask() || with_select_mask()),
            "only explicit buffer or select attention masks are supported");
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
        if (with_select_mask()) {
            VDISPATCH_SDPA(utils::one_of(mask_mdw.data_type(), s8, u8),
                    "the select condition data type must be s8 or u8");
            VDISPATCH_SDPA(mask_mdw.dims()[mask_k_index] == 1
                            || mask_mdw.strides()[mask_k_index] == 1,
                    "the select condition key axis must be contiguous");

            const memory_desc_wrapper fill_mdw(desc()->fill_md());
            VDISPATCH_SDPA(fill_mdw.ndims() == 1 && fill_mdw.dims()[0] == 1
                            && utils::one_of(
                                    fill_mdw.data_type(), f32, bf16, f16),
                    "the select attention mask fill must be a scalar f32, "
                    "bf16, or f16");
        } else
            VDISPATCH_SDPA(
                    mask_mdw.data_type() == dt || mask_mdw.data_type() == f32,
                    "the attention mask data type must match qry/dst or be "
                    "f32");
    }

    if (with_attn_scale()) {
        const memory_desc_wrapper scale_mdw(desc()->scale_md());
        VDISPATCH_SDPA(scale_mdw.ndims() == 1 && scale_mdw.dims()[0] == 1
                        && utils::one_of(scale_mdw.data_type(), f32, bf16, f16),
                "the attention scale must be a scalar f32, bf16, or f16");
    }

    VDISPATCH_SDPA(
            sdpa_fwd_pd_t::set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    // The online-softmax strategy never materializes the full [seq_q x seq_kv]
    // score matrix, but is f32-only and has no attention-mask post-op yet; the
    // full-softmax strategy covers bf16/f16 and an additive mask. Pick
    // online-softmax whenever the shapes/dtypes allow it, unless overridden for
    // debugging/perf comparisons via ONEDNN_SDPA_IMPL={online_softmax,full_softmax,auto}
    // (auto, the default, is the capability-based choice above).
    // The online mm1 reads K in place (no transpose), so its keys axis must be
    // unit-stride; full-softmax materialises a transpose for a strided keys
    // axis, so route those cases to full rather than rejecting them.
    const bool online_capable = dt == f32 && !with_buffer_mask()
            && key_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
            && val_mdw.strides()[sdpa_pd_t::ndims - 1] == 1;
    const std::string forced = getenv_string_user("SDPA_IMPL");
    VDISPATCH_SDPA(
            utils::one_of(forced, std::string(), std::string("auto"),
                    std::string("full_softmax"), std::string("online_softmax")),
            "unknown ONEDNN_SDPA_IMPL value '%s'", forced.c_str());
    if (forced == "online_softmax") {
        VDISPATCH_SDPA(online_capable,
                "the online-softmax SDPA strategy was forced via "
                "ONEDNN_SDPA_IMPL but "
                "does not support this data type or attention mask");
        kind_ = sdpa_impl_kind_t::online_softmax;
    } else if (forced == "full_softmax") {
        kind_ = sdpa_impl_kind_t::full_softmax;
    } else {
        kind_ = online_capable ? sdpa_impl_kind_t::online_softmax
                               : sdpa_impl_kind_t::full_softmax;
    }

    auto scratchpad = scratchpad_registry().registrar();

    if (kind_ == sdpa_impl_kind_t::online_softmax) {
        auto &fp_ = online_conf_.params;
        fp_ = sdpa_online_softmax_params_t();
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
        fp_.has_select = with_select_mask();
        fp_.invert_select = desc()->invert_select;
        if (with_select_mask()) {
            const memory_desc_wrapper cond_mdw(desc()->attn_mask_md());
            fp_.cond_strides.assign(
                    cond_mdw.strides(), cond_mdw.strides() + sdpa_pd_t::ndims);
            fp_.cond_dims.assign(
                    cond_mdw.dims(), cond_mdw.dims() + sdpa_pd_t::ndims);
        }

        // Derive the tiling + per-thread scratch layout once (JIT-free), then
        // size the scratchpad from it; the primitive compiles its kernels from
        // the same conf, so no work is repeated and the pd stays kernel-free.
        VDISPATCH_SDPA_SC(sdpa_online_softmax::configure(fp_, online_conf_),
                "sdpa_online_softmax configuration failed");
        nthr_ = online_conf_.nthr;
        scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
                online_conf_.scratch_total(nthr_), 1, 64);
        return status::success;
    }

    auto &bp_ = full_conf_.params;
    bp_ = sdpa_full_softmax_params_t();
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
    bp_.has_select = with_select_mask();
    bp_.invert_select = desc()->invert_select;
    if (with_select_mask()) {
        const memory_desc_wrapper cond_mdw(desc()->attn_mask_md());
        bp_.cond_strides.assign(
                cond_mdw.strides(), cond_mdw.strides() + sdpa_pd_t::ndims);
        bp_.cond_dims.assign(
                cond_mdw.dims(), cond_mdw.dims() + sdpa_pd_t::ndims);
    }
    bp_.softmax_inf_as_zero
            = desc()->softmax_alg == alg_kind::softmax_accurate_inf_as_zero;

    if (with_attn_scale()) {
        sdpa_mm1_post_op_t sc;
        sc.alg = alg_kind::binary_mul;
        sc.is_binary = true;
        sc.rhs_is_scalar = true;
        sc.rhs_dt = f32;
        bp_.mm1_post_ops.push_back(sc);
    }
    if (with_buffer_mask()) {
        const memory_desc_wrapper mask_mdw(desc()->attn_mask_md());
        sdpa_mm1_post_op_t mk;
        mk.alg = alg_kind::binary_add;
        mk.is_binary = true;
        mk.rhs_is_scalar = false;
        mk.rhs_dt = mask_mdw.data_type();
        mk.rhs_dims.assign(mask_mdw.dims(), mask_mdw.dims() + sdpa_pd_t::ndims);
        mk.rhs_strides.assign(
                mask_mdw.strides(), mask_mdw.strides() + sdpa_pd_t::ndims);
        bp_.mm1_post_ops.push_back(mk);
    }

    // Derive the query/KV blocking, per-thread + global scratch layout, and the
    // finalized BRGEMM descriptors once (JIT-free); size the scratchpad from the
    // conf the primitive later compiles its kernels from, so no work is repeated
    // and the pd stays kernel-free.
    VDISPATCH_SDPA_SC(
            sdpa_full_softmax::configure(bp_, full_conf_, full_descs_),
            "sdpa_full_softmax configuration failed");
    nthr_ = full_conf_.nthr;
    scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
            full_conf_.scratch_total(nthr_), 1, 64);

    return status::success;
}

status_t brgemm_sdpa_fwd_t::init(engine_t *engine) {
    const auto isa = mayiuse(avx512_core) ? avx512_core : avx2;

    if (pd()->impl_kind() == sdpa_impl_kind_t::online_softmax) {
        return sdpa_online_softmax::create_kernels(
                isa, pd()->online_softmax_conf(), engine, online_kernels_);
    } else {
        return sdpa_full_softmax::create_kernels(isa, pd()->full_softmax_conf(),
                pd()->full_softmax_descs(), engine, full_kernels_);
    }
    assert(!"Unknown SDPA implementation kind");
    return status::unimplemented;
}

namespace {
status_t get_scalar_value(
        const exec_ctx_t &ctx, const memory_desc_t &md, int arg, float &value) {
    if (md.data_type == data_type::f32) {
        value = *CTX_IN_MEM(const float *, arg);
    } else if (md.data_type == data_type::bf16) {
        value = *CTX_IN_MEM(const bfloat16_t *, arg);
    } else if (md.data_type == data_type::f16) {
        value = *CTX_IN_MEM(const float16_t *, arg);
    } else {
        return status::unimplemented;
    }
    return status::success;
}

status_t get_scalar_value(
        const exec_ctx_t &ctx, const sdpa_pd_t &pd, int arg, float &value) {
    const auto &storage = CTX_IN_STORAGE(arg);
    if (storage.is_host_scalar()) {
        const auto *host_storage
                = utils::downcast<const host_scalar_memory_storage_t *>(
                        &storage);
        return host_storage->get_scalar_value(&value, sizeof(value));
    } else {
        const auto &md = *pd.arg_md(arg);
        return get_scalar_value(ctx, md, arg, value);
    }
}
} // namespace

status_t brgemm_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto *scratch = ctx.get_scratchpad_grantor().get<char>(
            memory_tracking::names::key_sdpa_brgemm_buffer);

    auto *q = CTX_IN_MEM(const void *, DNNL_ARG_QUERIES);
    auto *k = CTX_IN_MEM(const void *, DNNL_ARG_KEYS);
    auto *v = CTX_IN_MEM(const void *, DNNL_ARG_VALUES);
    auto *out = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    // The scale is applied as a multiply; invert_scale means the user stored a
    // divisor, so multiply by its reciprocal. The local outlives the execute
    // call (the full-softmax path passes &scale_val by pointer). A host scalar scale is read
    // from host storage; a regular tensor scale from its device buffer.
    float scale_val = 1.0f;
    if (pd()->with_attn_scale()) {
        CHECK(get_scalar_value(ctx, *pd(), DNNL_ARG_SCALE, scale_val));
        if (pd()->desc()->invert_scale) scale_val = 1.0f / scale_val;
    }

    // For a select mask the condition tensor is delivered through the attn-mask
    // arg and the scalar fill through its own arg. The fill can arrive either as
    // a host scalar or as a runtime f32 buffer, mirroring the scale handling.
    const void *cond = nullptr;
    float fill_val = 0.0f;
    if (pd()->with_select_mask()) {
        cond = CTX_IN_MEM(const void *, DNNL_ARG_ATTN_MASK);
        CHECK(get_scalar_value(ctx, *pd(), DNNL_ARG_ATTN_MASK_FILL, fill_val));
    }

    if (pd()->impl_kind() == sdpa_impl_kind_t::online_softmax) {
        sdpa_online_softmax_run_args_t args;
        args.q = q;
        args.k = k;
        args.v = v;
        args.cond = cond;
        args.out = out;
        args.scale = scale_val;
        args.fill = fill_val;
        return sdpa_online_softmax::execute(pd()->online_softmax_conf(),
                online_kernels_, args, scratch, pd()->nthr());
    }

    sdpa_full_softmax_run_args_t args;
    args.q = q;
    args.k = k;
    args.v = v;
    args.cond = cond;
    args.out = out;
    args.fill = fill_val;
    // rhs base pointers for the mm1 binary post-ops, in the same order they
    // were appended to bp_.mm1_post_ops: scale (scalar) then attention mask.
    if (pd()->with_attn_scale()) args.mm1_post_op_rhs.push_back(&scale_val);
    if (pd()->with_buffer_mask())
        args.mm1_post_op_rhs.push_back(
                CTX_IN_MEM(const void *, DNNL_ARG_ATTN_MASK));
    return sdpa_full_softmax::execute(pd()->full_softmax_conf(), full_kernels_,
            args, scratch, pd()->nthr());
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
