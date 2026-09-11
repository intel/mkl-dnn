/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#include "graph/backend/dnnl/executables/sdpa.hpp"

#include "common/sdpa_test_iface.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using ltw = logical_tensor_wrapper_t;

status_t create_sdpa_pd(std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> &pd,
        const std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        const fpmath_t &fpmath) {
    const bool with_scale = op->get_attr<bool>(op_attr::with_scale);
    const bool is_training = op->get_attr<bool>(op_attr::is_training);
    const auto mask_type = static_cast<attn_mask_type_t>(
            op->get_attr<int64_t>(op_attr::mask_type));
    const bool with_dropout = op->get_attr<bool>(op_attr::with_dropout);

    auto md_q = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto md_k = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto md_v = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
    auto md_dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    auto md_scale = dnnl::memory::desc();
    size_t idx = 3;
    if (with_scale)
        md_scale = make_dnnl_memory_desc(op->get_input_logical_tensor(idx++));

    dnnl::memory::desc md_mask;
    const bool with_explicit_mask = mask_type == attn_mask_type::buffer;
    if (with_explicit_mask)
        md_mask = make_dnnl_memory_desc(op->get_input_logical_tensor(idx++));

    dnnl::primitive_attr attr, qk_attr, vs_attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(fpmath.mode_));
    if (with_dropout) {
        dnnl::memory::desc dropout_mask_desc;
        auto prop_type
                = ltw(op->get_input_logical_tensor(idx++)).property_type();
        attr.set_dropout(dropout_mask_desc, dnnl::memory::data_type::s64,
                /*use_offset*/ true,
                /*use_host_scalars*/ prop_type == property_type::host_scalar);
    }

    const bool is_invert_scale = op->has_attr(op_attr::is_invert_scale)
            ? op->get_attr<bool>(op_attr::is_invert_scale)
            : false;

    if (op->has_attr(op_attr::fusion_info)) {
        const auto &sdpa_fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        qk_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::QK);
        vs_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::VS);
    }

    // Set accumulation mode: the two attributes are requested for
    // dnnl_sdpa, so we can get them directly without calling has_attr().
    qk_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::qk_acc_mode)));
    vs_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::vs_acc_mode)));

    dim_t kv_head_number = op->get_input_logical_tensor(1).dims[1];

    const std::string &softmax_mode = op->get_attr<std::string>(op_attr::mode);
    const alg_kind_t softmax_alg = softmax_mode == "inf_as_zero"
            ? alg_kind::softmax_accurate_inf_as_zero
            : alg_kind::softmax_accurate;

    const auto prop
            = is_training ? dnnl_forward_training : dnnl_forward_inference;

    dnnl::memory::desc md_stats;
    if (is_training)
        md_stats = make_dnnl_memory_desc(op->get_output_logical_tensor(2));

    dnnl_primitive_desc_t raw_pd = nullptr;
    auto ret = sdpa_primitive_desc_create(&raw_pd, p_engine.get(), md_q.get(),
            md_k.get(), md_v.get(), md_dst.get(), md_mask.get(), md_scale.get(),
            is_invert_scale, kv_head_number, mask_type,
            static_cast<dnnl_alg_kind_t>(softmax_alg), prop, attr.get(),
            qk_attr.get(), vs_attr.get(), md_stats.get());

    if (raw_pd && ret == dnnl_success) {
        pd.reset(raw_pd);
        return status::success;
    }
    return status::unimplemented;
}

sdpa_executable_t::sdpa_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout)
    : with_scale_(op->get_attr<bool>(op_attr::with_scale))
    , is_training_(op->get_attr<bool>(op_attr::is_training))
    , mask_type_(static_cast<attn_mask_type_t>(
              op->get_attr<int64_t>(op_attr::mask_type)))
    , with_dropout_(op->get_attr<bool>(op_attr::with_dropout)) {
    UNUSED(pd_cache);
    UNUSED(use_block_layout);

    is_invert_scale_ = op->has_attr(op_attr::is_invert_scale)
            ? op->get_attr<bool>(op_attr::is_invert_scale)
            : false;
    with_explicit_mask_ = mask_type_ == attn_mask_type::buffer;

    if (create_sdpa_pd(pd_, op, p_engine, fpmath) != status::success) return;

    dnnl_primitive_t prim = nullptr;
    auto ret = dnnl_primitive_create(&prim, pd_.get());
    if (prim && ret == dnnl_success) { prim_.reset(prim); }
}

void sdpa_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    auto ret = dnnl_primitive_execute(prim_.get(), stream.get(),
            static_cast<int>(c_args.size()), c_args.data());
    dnnl::error::wrap_c_api(ret, "could not execute sdpa primitive on cpu");
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> sdpa_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    sycl::event return_event;
    auto ret = dnnl_sycl_interop_primitive_execute(prim_.get(), stream.get(),
            c_args.size(), c_args.data(), &deps, &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute sdpa primitive with sycl runtime");

    return return_event;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t sdpa_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    std::vector<cl_event> raw_deps(deps.begin(), deps.end());
    const cl_event *c_deps = raw_deps.empty() ? nullptr : raw_deps.data();

    cl_event return_event = nullptr;
    auto ret = dnnl_ocl_interop_primitive_execute(prim_.get(), stream.get(),
            static_cast<int>(c_args.size()), c_args.data(), c_deps,
            static_cast<int>(raw_deps.size()), &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute sdpa primitive with ocl runtime");

    return ocl_event_t(return_event);
}
#endif

arg_indices_t sdpa_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    // inputs: query, key, value
    size_t idx = 0;
    args.insert({DNNL_ARG_QUERIES, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_KEYS, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_VALUES, {indices_t::type_t::input, idx++}});
    // Optional inputs: scale, mask
    if (op->get_attr<bool>(op_attr::with_scale)) {
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
    }
    if (op->get_attr<int64_t>(op_attr::mask_type)
            == static_cast<int64_t>(attn_mask_type::buffer)) {
        args.insert({DNNL_ARG_ATTN_MASK, {indices_t::type_t::input, idx++}});
    }

    if (op->get_attr<bool>(op_attr::with_dropout)) {
        args.insert({DNNL_ARG_ATTR_DROPOUT_SEED,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_OFFSET,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_PROBABILITY,
                {indices_t::type_t::input, idx++}});
    }

    const auto &sdpa_fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_KEYS)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_KEYS)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_VALUES)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_VALUES)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES,
                {indices_t::type_t::input, idx++}});
    }

    // outputs
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    if (op->get_attr<bool>(op_attr::is_training)) {
        args.insert({DNNL_ARG_WORKSPACE, {indices_t::type_t::output, 2}});
    }
    return args;
}

sdpa_bwd_executable_t::sdpa_bwd_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout)
    : with_scale_(op->get_attr<bool>(op_attr::with_scale))
    , mask_type_(static_cast<attn_mask_type_t>(
              op->get_attr<int64_t>(op_attr::mask_type)))
    , is_invert_scale_(op->has_attr(op_attr::is_invert_scale)
                      ? op->get_attr<bool>(op_attr::is_invert_scale)
                      : false)
    , with_explicit_mask_(mask_type_ == attn_mask_type::buffer)
    , with_dropout_(op->get_attr<bool>(op_attr::with_dropout)) {
    // Op inputs: Q(0) K(1) V(2) dst/O(3) stats(4) diff_dst/dO(5) [scale(6)] [mask(7)]
    // Op outputs: diff_q(0) diff_k(1) diff_v(2) scratchpad(3) [diff_mask/dS(4)]
    auto md_q = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto md_k = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto md_v = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
    auto md_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(3));
    // stats at input 4 is consumed as DNNL_ARG_WORKSPACE at execute time
    auto md_diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(5));
    auto md_diff_q = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    auto md_diff_k = make_dnnl_memory_desc(op->get_output_logical_tensor(1));
    auto md_diff_v = make_dnnl_memory_desc(op->get_output_logical_tensor(2));

    // Optional scale, attn_mask, dS (diff_mask)
    dnnl::memory::desc md_scale, md_attn_mask, md_dS;
    size_t idx = 6;
    if (with_scale_) {
        md_scale = make_dnnl_memory_desc(op->get_input_logical_tensor(idx++));
    }
    if (with_explicit_mask_) {
        md_attn_mask
                = make_dnnl_memory_desc(op->get_input_logical_tensor(idx++));
        if (op->num_outputs() > 4) {
            md_dS = make_dnnl_memory_desc(op->get_output_logical_tensor(4));
        }
    }

    // Fusion info and attributes (if any)
    const auto &sdpa_fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    dnnl::primitive_attr attr, qk_attr, vs_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        qk_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::QK);
        vs_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::VS);
    }
    // Set accumulation mode: the two attributes are requested for
    // dnnl_sdpa, so we can get them directly without calling has_attr().
    qk_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::qk_acc_mode)));
    vs_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::vs_acc_mode)));
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(fpmath.mode_));

    if (with_dropout_) {
        dnnl::memory::desc dropout_mask_desc;
        auto prop_type
                = ltw(op->get_input_logical_tensor(idx++)).property_type();
        attr.set_dropout(dropout_mask_desc, dnnl::memory::data_type::s64,
                /*use_offset*/ true,
                /*use_host_scalars*/ prop_type == property_type::host_scalar);
    }

    dim_t kv_head_number = op->get_input_logical_tensor(1).dims[1];
    const alg_kind_t softmax_alg = alg_kind::softmax_accurate_inf_as_zero;

    // create hint_fwd pd
    dnnl_primitive_desc_t hint_pd = nullptr;
    auto ret = sdpa_primitive_desc_create(&hint_pd, p_engine.get(), md_q.get(),
            md_k.get(), md_v.get(), md_dst.get(), md_attn_mask.get(),
            md_scale.get(), is_invert_scale_, kv_head_number, mask_type_,
            static_cast<dnnl_alg_kind_t>(softmax_alg), dnnl_forward_training,
            attr.get(), qk_attr.get(), vs_attr.get());

    if (hint_pd && ret == dnnl_success) {
        hint_pd_.reset(hint_pd);
    } else {
        return;
    }

    dnnl_primitive_desc_t pd = nullptr;
    ret = sdpa_primitive_desc_create(&pd, p_engine.get(), md_q.get(),
            md_k.get(), md_v.get(), md_dst.get(), md_attn_mask.get(),
            md_scale.get(), md_diff_q.get(), md_diff_k.get(), md_diff_v.get(),
            md_diff_dst.get(), md_dS.get(), is_invert_scale_, kv_head_number,
            mask_type_, static_cast<dnnl_alg_kind_t>(softmax_alg), attr.get(),
            hint_pd_.get());

    if (pd && ret == dnnl_success) {
        pd_.reset(pd);
    } else {
        return;
    }

    dnnl_primitive_t prim = nullptr;
    ret = dnnl_primitive_create(&prim, pd_.get());
    if (prim && ret == dnnl_success) { prim_.reset(prim); }
}

void sdpa_bwd_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    auto ret = dnnl_primitive_execute(prim_.get(), stream.get(),
            static_cast<int>(c_args.size()), c_args.data());
    dnnl::error::wrap_c_api(
            ret, "could not execute sdpa backward primitive on cpu");
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> sdpa_bwd_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    sycl::event return_event;
    auto ret = dnnl_sycl_interop_primitive_execute(prim_.get(), stream.get(),
            c_args.size(), c_args.data(), &deps, &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute sdpa backward with sycl runtime");
    return return_event;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t sdpa_bwd_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    std::vector<cl_event> raw_deps(deps.begin(), deps.end());
    const cl_event *c_deps = raw_deps.empty() ? nullptr : raw_deps.data();
    cl_event return_event = nullptr;
    auto ret = dnnl_ocl_interop_primitive_execute(prim_.get(), stream.get(),
            static_cast<int>(c_args.size()), c_args.data(), c_deps,
            static_cast<int>(raw_deps.size()), &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute sdpa backward with ocl runtime");
    return ocl_event_t(return_event);
}
#endif

arg_indices_t sdpa_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    // inputs: Q, K, V, dst(O), stats(workspace), diff_dst(dO)
    size_t idx = 0;
    args.insert({DNNL_ARG_QUERIES, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_KEYS, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_VALUES, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_DST, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_WORKSPACE, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, idx++}});
    // optional: scale, mask
    if (op->get_attr<bool>(op_attr::with_scale)) {
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
    }
    if (op->get_attr<int64_t>(op_attr::mask_type)
            == static_cast<int64_t>(attn_mask_type::buffer)) {
        args.insert({DNNL_ARG_ATTN_MASK, {indices_t::type_t::input, idx++}});
    }

    if (op->get_attr<bool>(op_attr::with_dropout)) {
        args.insert({DNNL_ARG_ATTR_DROPOUT_SEED,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_OFFSET,
                {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_ATTR_DROPOUT_PROBABILITY,
                {indices_t::type_t::input, idx++}});
    }
    // outputs: diff_q, diff_k, diff_v, scratchpad, [dS]
    args.insert({DNNL_ARG_DIFF_QUERIES, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_DIFF_KEYS, {indices_t::type_t::output, 1}});
    args.insert({DNNL_ARG_DIFF_VALUES, {indices_t::type_t::output, 2}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 3}});
    if (op->num_outputs() > 4) {
        args.insert({DNNL_ARG_DS, {indices_t::type_t::output, 4}});
    }
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
