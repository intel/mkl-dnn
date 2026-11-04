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

#include "graph/backend/dnnl/executables/softmax.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

softmax_executable_t::desc_t softmax_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::softmax_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    int64_t axis = op->get_attr<int64_t>(op_attr::axis);
    if (axis < 0) { axis += src.get_ndims(); }

    dnnl::algorithm algo = dnnl::algorithm::undef;
    if (op->get_kind() == op_kind::_softmax) {
        const auto mode = op->get_attr<std::string>(op_attr::mode);
        algo = mode == "inf_as_zero"
                ? static_cast<dnnl::algorithm>(
                          dnnl::impl::alg_kind::softmax_accurate_inf_as_zero)
                : dnnl::algorithm::softmax_accurate;
    } else if (op->get_kind() == op_kind::_logsoftmax) {
        algo = dnnl::algorithm::softmax_log;
    } else {
        assert(!"unexpected op kind");
    }

    dnnl::softmax_forward::primitive_desc pd;
    pd = dnnl::softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algo, src, dst,
            static_cast<int>(axis), prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

softmax_bwd_executable_t::desc_t softmax_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::softmax_backward::primitive_desc>(pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    diff_dst = to_format_any(diff_dst);

    auto diff_src_lt = op->get_output_logical_tensor(0);
    auto diff_src = make_dnnl_memory_desc(diff_src_lt);

    const auto rank = op->get_output_logical_tensor(0).ndims;
    const auto res = utils::try_reverse_axis(
            op->get_attr<int64_t>(op_attr::axis), rank);
    assertm(res.first, "Incorrect axis value.");
    const auto axis = res.second;

    auto dst_lt = op->get_input_logical_tensor(1);
    auto dst = make_dnnl_memory_desc(dst_lt);

    // construct src with layout information from dst and data type information
    // from diff_src.
    auto src_lt = dst_lt;
    src_lt.data_type = diff_src_lt.data_type;
    auto src = make_dnnl_memory_desc(src_lt);

    const dnnl::algorithm algo = op->get_kind() == op_kind::_logsoftmax_bwd
            ? dnnl::algorithm::softmax_log
            : dnnl::algorithm::softmax_accurate;

    auto hint_fwd_pd = dnnl::softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_training, algo, src, dst, static_cast<int>(axis),
            prm_attr);

    auto pd = dnnl::softmax_backward::primitive_desc(p_engine, algo, diff_src,
            diff_dst, dst, static_cast<int>(axis), hint_fwd_pd, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t softmax_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

arg_indices_t softmax_bwd_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);

    arg_indices_t args;
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_DST, {indices_t::type_t::input, 1}});
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
