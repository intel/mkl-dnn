/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "ref_primitive.hpp"
#include "setting_handler.hpp"

#define DEFINE_REF_PRIM_HELPER_FUNCS(driver) \
    namespace driver { \
    void init_memory_args_native(dnn_mem_map_t &, const base_prb_t *, \
            const graph::deserialized_op_t &, const engine_t &) {} \
    int execute(const base_prb_t *, const args_t &, res_t *res) { \
        res->state = UNIMPLEMENTED; \
        return FAIL; \
    } \
    } // namespace driver

// template to generate empty memory post processing and execute functions
// This is used for drivers that do not require any memory post processing
// or specialized execution logic.
DEFINE_REF_PRIM_HELPER_FUNCS(binary);
DEFINE_REF_PRIM_HELPER_FUNCS(bnorm);
DEFINE_REF_PRIM_HELPER_FUNCS(concat);
DEFINE_REF_PRIM_HELPER_FUNCS(conv);
DEFINE_REF_PRIM_HELPER_FUNCS(deconv);
DEFINE_REF_PRIM_HELPER_FUNCS(eltwise);
DEFINE_REF_PRIM_HELPER_FUNCS(gnorm);
DEFINE_REF_PRIM_HELPER_FUNCS(lnorm);
DEFINE_REF_PRIM_HELPER_FUNCS(matmul);
DEFINE_REF_PRIM_HELPER_FUNCS(pool);
DEFINE_REF_PRIM_HELPER_FUNCS(prelu);
DEFINE_REF_PRIM_HELPER_FUNCS(reduction);
DEFINE_REF_PRIM_HELPER_FUNCS(reorder);
DEFINE_REF_PRIM_HELPER_FUNCS(resampling);
// DEFINE_REF_PRIM_HELPER_FUNCS(softmax); // Has defined body below.

namespace softmax {

void init_memory_args_native(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        const graph::deserialized_op_t &base_op_ref, const engine_t &ref_eng) {
    const auto *prb = ::softmax::prb_t::from(base_prb);
    if (base_op_ref.out_lts_.size() != 2) {
        assert(!"softmax should have two outputs to run into "
                "init_memory_args_native");
    }

    mem_map.emplace(DNNL_ARG_SRC,
            dnn_mem_t(prb->ndims, prb->dims.data(), dnnl_f32, prb->stag,
                    ref_eng, /* prefill = */ true));

    mem_map.emplace(DNNL_ARG_DST,
            dnn_mem_t(prb->ndims, prb->dims.data(), dnnl_f32, prb->dtag,
                    ref_eng, /* prefill = */ true));

    dims_t stats_shapes = base_op_ref.out_lts_[1].shape_;
    dims_t stats_strides = base_op_ref.out_lts_[1].stride_;
    mem_map.emplace(DNNL_ARG_DST_1,
            dnn_mem_t(prb->ndims, stats_shapes.data(), dnnl_f32,
                    stats_strides.data(), ref_eng, /* prefill = */ true));
}

int execute(const base_prb_t *base_prb, const args_t &args, res_t *res) {
    const auto *prb = ::softmax::prb_t::from(base_prb);
    assert(args.find(DNNL_ARG_DST_1).size() != 0);
    // in order to run with ::softmax::compute_ref, we need to
    // construct all memory with abx tag and f32 dt
    args_t ref_args;
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    dnn_mem_t src_f32_abx(src, dnnl_f32, tag::abx, src.engine());
    SAFE_V(src_f32_abx.reorder(src, res));
    ref_args.set(DNNL_ARG_SRC, src_f32_abx);

    dnn_mem_t &dst = const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST));
    dnn_mem_t dst_f32_abx(dst, dnnl_f32, tag::abx, dst.engine());
    SAFE_V(dst_f32_abx.reorder(dst, res));
    ref_args.set(DNNL_ARG_DST, dst_f32_abx);

    dnn_mem_t &stats = const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST_1));
    dnn_mem_t stats_f32_abx(stats, dnnl_f32, tag::abx, stats.engine());
    SAFE_V(stats_f32_abx.reorder(stats, res));
    ref_args.set(DNNL_ARG_DST_1, stats_f32_abx);

    ::softmax::compute_ref(prb, prb->dir, ref_args);

    // restore original memory format after ::softmax::compute_ref
    SAFE_V(dst.reorder(dst_f32_abx, res));
    SAFE_V(stats.reorder(stats_f32_abx, res));

    res->state = EXECUTED;
    return OK;
}
} // namespace softmax

namespace graph {

ref_primitive_t::ref_primitive_t(const deserialized_op_t &op)
    : op_(op), kind_(opstr2kind(op_.kind_)), driver_(op_.opkind2driver()) {

    static const ::std::unordered_set<::std::string> special_backward_op = {
            // bnorm backward
            "BatchNormTrainingBackward",
            // eltwise backward
            "AbsBackward",
            "ClampBackward",
            "EluBackward",
            "GELUBackward",
            "HardSigmoidBackward",
            "HardSwishBackward",
            "MishBackward",
            "ReLUBackward",
            "SigmoidBackward",
            "SoftPlusBackward",
            "SqrtBackward",
            "TanhBackward",
            // pool backward
            "AvgPoolBackward",
            "MaxPoolBackward",
    };
    is_special_backward_op_
            = special_backward_op.find(op_.kind_) != special_backward_op.end();
}

int ref_primitive_t::init_prb(res_t *res) {
#define CASE_INIT_PRB(driver) \
    case dnnl_driver_t::driver: { \
        ::driver::settings_t setting \
                = get_setting<::driver::settings_t>(op_, res); \
        if (res->state == INVALID_ARGUMENTS) return FAIL; \
        setting.finalize(); \
        prb_ = std::make_shared<::driver::prb_t>(setting); \
        setup_cmp_func_ = ::driver::setup_cmp; \
        execute_func_ = ::driver::execute; \
        init_ref_memory_args_func_ = ::driver::init_ref_memory_args; \
        init_memory_args_native_func_ = ::driver::init_memory_args_native; \
        init_pd_func_ = ::driver::init_pd; \
        break; \
    }

    switch (driver_) {
        CASE_INIT_PRB(custom);
        CASE_INIT_PRB(binary);
        CASE_INIT_PRB(bnorm);
        CASE_INIT_PRB(concat);
        CASE_INIT_PRB(conv);
        CASE_INIT_PRB(deconv);
        CASE_INIT_PRB(eltwise);
        CASE_INIT_PRB(gnorm);
        CASE_INIT_PRB(lnorm);
        CASE_INIT_PRB(matmul);
        CASE_INIT_PRB(pool);
        CASE_INIT_PRB(prelu);
        CASE_INIT_PRB(reduction);
        CASE_INIT_PRB(reorder);
        CASE_INIT_PRB(resampling);
        CASE_INIT_PRB(softmax);
        default: {
            SAFE_V(FAIL);
            break;
        }
    }

    return OK;
}

int ref_primitive_t::init_prim(
        const engine_t &ref_eng, res_t *res, bool force_override) {
    // The custom driver does not contain a primitive, skip this step.
    if (driver_ == dnnl_driver_t::custom) {
        ::custom::collect_mem_size(res->mem_size_args, prb_.get());
        return OK;
    }

    const bool is_quant_or_dequant = kind_ == dnnl::graph::op::kind::Dequantize
            || kind_ == dnnl::graph::op::kind::Quantize
            || kind_ == dnnl::graph::op::kind::DynamicDequantize
            || kind_ == dnnl::graph::op::kind::DynamicQuantize;
    // (De-)Quantize op is built on reorder which expects int8 dt for
    // zero-points attribute. Thus, skip them for forcing.
    const bool force_f32_prim_dt = !force_override && !is_quant_or_dequant;

    const base_prb_t *prb = prb_.get();
    dnn_mem_map_t ref_mems;
    if (is_special_backward_op_) {
        SAFE(create_primitive(fwd_prim_, ref_eng, init_pd_func_, prb, res,
                     FLAG_FWD, nullptr, prb->dir & FLAG_BWD, nullptr,
                     force_f32_prim_dt, /*is_graph_ref=*/true),
                WARN);
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
        SAFE(::init_memory_args(mems_, prb, fwd_prim_, res,
                     /*override_dir_with_fwd=*/true, ref_eng),
                WARN);
        SAFE(init_ref_memory_args_func_(
                     ref_mems, mems_, fwd_prim_, prb, res, nullptr),
                WARN);
        args_ = args_t(mems_);
        SAFE(execute_and_wait(fwd_prim_, args_, res), WARN);
    }
    SAFE(create_primitive(prim_, ref_eng, init_pd_func_, prb, res, prb->dir,
                 is_special_backward_op_ ? query_pd(fwd_prim_) : nullptr, false,
                 nullptr, force_f32_prim_dt, /*is_graph_ref=*/true),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED
            || res->state == DEFERRED)
        return OK;

    return OK;
}

void ref_primitive_t::init_memory_args(const engine_t &ref_eng, res_t *res) {
    if (!prb_) return;

    if (prim_) {
        SAFE_V(::init_memory_args(mems_, prb_.get(), prim_, res,
                /*override_dir_with_fwd=*/false, ref_eng));
    } else {
        // handle an empty primitive through a driver's reference
        init_memory_args_native_func_(mems_, prb_.get(), op_, ref_eng);
    }
}

int ref_primitive_t::init_ref_memory_args(const engine_t &ref_eng, res_t *res) {
    dnn_mem_map_t ref_mems;
    if (prb_) {
        SAFE(init_ref_memory_args_func_(
                     ref_mems, mems_, prim_, prb_.get(), res, nullptr),
                WARN);
        args_ = args_t(mems_);
    }
    return OK;
}

int ref_primitive_t::execute_prim(res_t *res) const {
    if (prim_) {
        SAFE(execute_and_wait(prim_, args_, res), WARN);
    } else if (prb_) {
        SAFE(execute_func_(prb_.get(), args_, res), WARN);
    }
    return OK;
}

void ref_primitive_t::check_correctness(const args_t &args, bool has_eltwise,
        bool has_nans, res_t *res, float p2p_threshold) const {

    static const std::unordered_map<size_t, data_kind_t>
            dnnl_arg_2_data_kind_map {
                    {DNNL_ARG_SRC, SRC},
                    {DNNL_ARG_WEIGHTS_0, WEI},
                    {DNNL_ARG_DIFF_WEIGHTS_0, WEI},
                    {DNNL_ARG_BIAS, BIA},
                    {DNNL_ARG_DIFF_BIAS, BIA},
                    {DNNL_ARG_DST, DST},
                    {DNNL_ARG_DST_1, SDPA_STATS},
                    {DNNL_ARG_DIFF_SRC_0, DST},
                    {DNNL_ARG_SRC_1, SRC_1},
                    {DNNL_ARG_MEAN, MEAN},
                    {DNNL_ARG_VARIANCE, VAR},
                    {DNNL_ARG_SCALE, SC},
                    {DNNL_ARG_DIFF_SCALE, SC},
                    {DNNL_ARG_SHIFT, SH},
                    {DNNL_ARG_DIFF_SHIFT, SH},
            };

    // args is the result from graph side
    // args_ is the reference result under this context
    // only check the arg contained in args, compare args with args_
    for (int i = 0; i < args.size(); i++) {
        check_zero_padding(args.dnn_mem(i), args.arg(i), res);
        check_buffer_overwrite(args.dnn_mem(i), args.arg(i), res);

        const auto arg = args.arg(i);
        const auto &mem_dt = args.find(arg);
        const auto &mem_fp = args_.find(arg);

        auto it = dnnl_arg_2_data_kind_map.find(arg);
        if (it == dnnl_arg_2_data_kind_map.end()) {
            BENCHDNN_PRINT(1, "Output arg %d is unsupported!\n", arg);
            res->state = UNIMPLEMENTED;
            return;
        }
        compare::compare_t cmp;
        setup_cmp_func_(cmp, prb_.get(), it->second, args_);
        const attr_t &attr = prb_->attr;

        cmp.set_data_kind(it->second);
        cmp.set_has_eltwise_post_op(has_eltwise);
        cmp.set_op_output_has_nans(has_nans);
        // `cmp` object has internal knowledge on when this check must be
        // enabled.
        cmp.set_allow_norm_check(true);
        // Partition-level override, e.g. for rounding inside a fusion
        if (p2p_threshold > 0.f) cmp.set_threshold(p2p_threshold);
        // TODO: there's an open question with how to determine the threshold
        // and what the criteria to use. Unless a partition says it is some
        // complex fusion (such as SDP) with a specific data type, setting such
        // unconditional threshold is potentially unsafe.
        //
        // So far, the issue only with pure bf16 patterns, and here's why:
        // * f32 supposed to be exact on both ends as computations repeat each
        //   other on both ends.
        // * int8 softmax's output are integer values which in turn makes second
        //   matmul's output precise.
        // * bf16 softmax's output contains irregular floating-point values that
        //   potentially get accumulated in a different order on each end, and
        //   it leads to an output mismatch. Different underlying
        //   implementations can add more to that.
        //
        // Note: the following threshold is obtained from actual runs on
        // different hardware.
        cmp.set_threshold_norm(3e-3f);
        dnn_mem_t mem_fp_abx(mem_fp, dnnl_f32, tag::abx, ::get_cpu_engine());
        // Clear previous output stats.
        auto cur_res_state = res->state;
        res->reset_stats(cur_res_state);
        cmp.compare(mem_fp_abx, mem_dt, attr, res);
    }
}

int ref_primitive_t::displace_scales() const {
    // Runtime data for scales attribute is supported for quantization ops only.
    if (op_.kind_ != "Dequantize" && op_.kind_ != "Quantize") return OK;

    const auto it_attr_scales = op_.attrs_.find("scales");
    const bool has_scales = it_attr_scales != op_.attrs_.end();
    if (!has_scales) return OK;

    int arg = DNNL_ARG_UNDEF;
    bool scales_found = false;
    for (auto it = mems_.begin(); it != mems_.end(); it++) {
        const int cur_arg = (*it).first;
        const bool is_scales_arg = (cur_arg & DNNL_ARG_ATTR_SCALES);
        if (!is_scales_arg) continue;
        // Protection from the cases when somehow scales are applied to more
        // than a single argument (which is unexpected).
        if (scales_found) {
            assert(!"scales are applied to more than a single arg");
            return FAIL;
        }
        scales_found = true;
        arg = cur_arg;
    }

    // No correspondent memory was found. Nothing to update.
    if (arg == DNNL_ARG_UNDEF) return OK;

    // Updating values.
    const auto &mem = mems_.at(arg);
    const auto &f32_vector = it_attr_scales->second.f32_vector_;
    for (size_t i = 0; i < f32_vector.size(); i++) {
        mem.set_elem(i, f32_vector[i]);
    }

    return OK;
}

dnnl_data_type_t ref_primitive_t::get_lt_dt(size_t id) const {
    for (size_t i = 0; i < op_.in_lts_.size(); i++) {
        if (op_.in_lts_[i].id_ == id)
            return str2dt(op_.in_lts_[i].data_type_.c_str());
    }
    for (size_t i = 0; i < op_.out_lts_.size(); i++) {
        if (op_.out_lts_[i].id_ == id)
            return str2dt(op_.out_lts_[i].data_type_.c_str());
    }
    assert(!"id not found");
    return dnnl_data_type_undef;
}

} // namespace graph
