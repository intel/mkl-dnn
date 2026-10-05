/*******************************************************************************
 * Copyright 2022 Intel Corporation
 * Copyright 2026 Arm Ltd. and affiliates
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

#include <memory>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/layout_propagator.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

#include "graph/interface/shape_infer.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
#define VCHECK_LAYOUT_PROPAGATOR(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, layout_propagator, (cond), status, msg, \
            ##__VA_ARGS__);

using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

status_t insert_reorder_before(op_ptr &op, size_t offset,
        const dnnl::memory::desc &opt_mdesc, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr in_val = op->get_input_value(offset);
    const logical_tensor_t &in_lt = in_val->get_logical_tensor();
    // just return if real input layout is the same as optimal layout or
    // input layout type is ANY
    if (make_dnnl_memory_desc(in_lt) == opt_mdesc || ltw(in_lt).is_any())
        return status;

    // create reorder op, connect it to graph and add it's scratchpad output
    auto reorder_op = std::make_shared<op_t>(op_kind::_reorder);
    rewriter.insert_op_before(reorder_op, op, offset);
    auto scratchpad_val = insert_empty_scratchpad(reorder_op);
    // set optimal layout to reorder's output
    auto reorder_out_val = reorder_op->get_output_value(0);
    status = fill_layout_info(reorder_out_val, opt_mdesc);
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder output");
    // fill shape info
    reorder_out_val->set_data_type(ltw(in_lt).data_type());
    reorder_out_val->set_dims(ltw(in_lt).vdims());

    // set layout info for scratchpad output
    const auto &pd = reorder_executable_t::create_desc(
            reorder_op, p_engine, pd_cache, fpmath, use_block_layout);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t insert_reorder_after(op_ptr &op, size_t offset,
        const dnnl::memory::desc &opt_mdesc, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr out_val = op->get_output_value(offset);
    const logical_tensor_t &out_lt = out_val->get_logical_tensor();
    // just return if real output layout is the same as optimal layout or
    // output layout type is ANY
    if (make_dnnl_memory_desc(out_lt) == opt_mdesc || ltw(out_lt).is_any())
        return status;

    // create reorder op, connect it to graph and add it's scratchpad output
    auto reorder_op = std::make_shared<op_t>(op_kind::_reorder);
    rewriter.insert_op_after(reorder_op, op, offset);
    auto scratchpad_val = insert_empty_scratchpad(reorder_op);
    // set optimal layout to reorder's input
    auto reorder_in_val = reorder_op->get_input_value(0);
    status = fill_layout_info(reorder_in_val, opt_mdesc);
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder input");
    // fill shape info
    reorder_in_val->set_data_type(ltw(out_lt).data_type());
    reorder_in_val->set_dims(ltw(out_lt).vdims());

    // set layout info for scratchpad output
    const auto &pd = reorder_executable_t::create_desc(
            reorder_op, p_engine, pd_cache, fpmath, use_block_layout);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_conv(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    // always create pd using any format
    const auto &pd = conv_fwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    // insert reorders for conv's inputs
    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv src");
    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv weights");

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(op, 2, pd.bias_desc(), p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder before conv bias");
    }

    if (!op->has_attr(op_attr::fusion_info)) {
        fusion_info_t fusion_info;
        op->set_attr<fusion_info_t>(op_attr::fusion_info, fusion_info);
    }
    const fusion_info_t &fusion_info
            = op->get_attr<fusion_info_t>(op_attr::fusion_info);

    if (fusion_info.has_post_dw_conv()) {
        const auto &dw_conv = fusion_info.get_post_dw_conv();
        auto dw_idx = dw_conv->get_unfused_input_indices();
        value_ptr dw_wei = op->get_input_value(dw_idx[0]);
        value_ptr dw_bias = nullptr;
        if (dw_conv->get_unfused_input_indices().size() > 1) {
            dw_bias = op->get_input_value(dw_idx[1]);
        }

        const auto &dw_wei_opt_mdesc = pd.query_md(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        insert_reorder_before(op, dw_idx[0], dw_wei_opt_mdesc, p_engine,
                pd_cache, fpmath, use_block_layout, rewriter);
        status = fill_layout_info(dw_wei, dw_wei_opt_mdesc);

        if (dw_conv->get_unfused_input_indices().size() > 1) {
            const auto &dw_bias_opt_mdesc = pd.query_md(query::exec_arg_md,
                    DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
            insert_reorder_before(op, dw_idx[1], dw_bias_opt_mdesc, p_engine,
                    pd_cache, fpmath, use_block_layout, rewriter);
            status = fill_layout_info(dw_bias, dw_bias_opt_mdesc);
        }
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder before post_dw_conv");
    }

    if (fusion_info.has_post_binary()) {
        const auto &post_ops = fusion_info.get_post_ops();
        for (size_t i = 0; i < post_ops.size(); ++i) {
            if (!post_ops[i]->is_post_binary()) continue;
            const auto &binary = post_ops[i];
            std::vector<size_t> binary_idx
                    = binary->get_unfused_input_indices();
            if (binary_idx.empty()) continue;

            value_ptr binary_unfused_src = op->get_input_value(binary_idx[0]);
            const auto &binary_unfused_src_opt_mdesc
                    = pd.query_md(query::exec_arg_md,
                            DNNL_ARG_SRC_1
                                    | DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                            static_cast<int>(i)));
            insert_reorder_before(op, binary_idx[0],
                    binary_unfused_src_opt_mdesc, p_engine, pd_cache, fpmath,
                    use_block_layout, rewriter);
            status = fill_layout_info(
                    binary_unfused_src, binary_unfused_src_opt_mdesc);

            VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                    "failed to fill layout info for reorder before "
                    "conv post_binary");
        }
    }
    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after conv dst");

    // fill scratchpads dimensions and data type to scratchpad value_t
    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = deconv_fwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    // insert reorders for deconv's inputs
    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv src");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv weights");

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(op, 2, pd.bias_desc(), p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder before deconv bias");
    }

    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        const auto &post_ops = fusion_info.get_post_ops();
        for (size_t i = 0; i < post_ops.size(); ++i) {
            if (!post_ops[i]->is_post_binary()) continue;
            const auto &binary = post_ops[i];
            std::vector<size_t> binary_idx
                    = binary->get_unfused_input_indices();
            if (binary_idx.empty()) continue;

            value_ptr binary_unfused_src = op->get_input_value(binary_idx[0]);
            const auto &binary_unfused_src_opt_mdesc
                    = pd.query_md(query::exec_arg_md,
                            DNNL_ARG_SRC_1
                                    | DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                            static_cast<int>(i)));
            insert_reorder_before(op, binary_idx[0],
                    binary_unfused_src_opt_mdesc, p_engine, pd_cache, fpmath,
                    use_block_layout, rewriter);
            status = fill_layout_info(
                    binary_unfused_src, binary_unfused_src_opt_mdesc);

            VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                    "failed to fill layout info for reorder before "
                    "deconv post_binary");
        }
    }

    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after deconv dst");

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv_bwd_data(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    // always create pd using any format
    const auto &pd = deconv_bwd_data_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    // insert reorders for inputs
    insert_reorder_before(op, 0, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status_t status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv_bwd_data "
            "diff_dst");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv_bwd_data "
            "weights");

    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after deconv_bwd_data "
            "diff_src");

    // fill scratchpads dimensions and data type to scratchpad value_t
    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv_bwd_weights(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = deconv_bwd_weights_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv_bwd_weights "
            "src");

    insert_reorder_before(op, 1, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before deconv_bwd_weights "
            "diff_dst");

    insert_reorder_after(op, 0, pd.diff_weights_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr diff_weights = op->get_output_value(0);
    status = fill_layout_info(diff_weights, pd.diff_weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after deconv_bwd_weights "
            "diff_weights");

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_eltwise(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    // When input's layout is specified (opaque or strided),
    // we can propagate it to output.
    const auto &pd = eltwise_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after eltwise dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_eltwise_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = eltwise_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    // to hit an optimized kernel, input/output of forward and both diff_dst and
    // diff_src should use the same memory format. Primitive is created based on
    // a backward data and here we are conditionally aligning forward data
    // format.
    auto opt_desc = (op->has_attr(op_attr::use_dst)
                            && op->get_attr<bool>(op_attr::use_dst))
            ? pd.dst_desc()
            : pd.src_desc();
    insert_reorder_before(op, 0, opt_desc, p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr data = op->get_input_value(0);
    status_t status = fill_layout_info(data, opt_desc);
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before eltwise_bwd "
            "inputs 0");

    insert_reorder_before(op, 1, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, opt_desc);
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before eltwise_bwd "
            "diff_dst");
    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after eltwise_bwd "
            "diff_src");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_binary(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    using ltw = logical_tensor_wrapper_t;
    status_t status = status::success;

    // if with zero dimension, the binary op will take no effect, we just
    // complete the layout propagation process by using dummy dst md.
    if (ltw(op->get_input_logical_tensor(0)).has_zero_dim()
            || ltw(op->get_input_logical_tensor(1)).has_zero_dim()) {
        value_ptr dst = op->get_output_value(0);
        status = fill_layout_info(dst,
                to_ncx_format(
                        make_dnnl_memory_desc(dst->get_logical_tensor())));
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for binary logical tensor with 0 "
                "dims");

        return fill_layout_info(op->get_output_value(1), dnnl::memory::desc {});
    }

    const auto &pd = binary_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after binary dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_concat(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = concat_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        insert_reorder_before(op, i, pd.src_desc(static_cast<int>(i)), p_engine,
                pd_cache, fpmath, use_block_layout, rewriter);
        status = fill_layout_info(
                op->get_input_value(i), pd.src_desc(static_cast<int>(i)));
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder before concat "
                "src");
    }

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    status = fill_layout_info(op->get_output_value(0), pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after concat dst");

    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_shuffle(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = shuffle_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);

    VCHECK_LAYOUT_PROPAGATOR(!ltw(src->get_logical_tensor()).is_any(),
            status::invalid_arguments, "shuffle's src layout can't be any");

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after shuffle dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_matmul(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    using ltw = logical_tensor_wrapper_t;
    status_t status = status::success;

    // if with zero dimension, the matmul op will take no effect, we just
    // complete the layout propagation process by using dummy dst md.
    if (ltw(op->get_input_logical_tensor(0)).has_zero_dim()
            || ltw(op->get_input_logical_tensor(1)).has_zero_dim()) {
        value_ptr dst = op->get_output_value(0);
        status = fill_layout_info(dst,
                to_ncx_format(
                        make_dnnl_memory_desc(dst->get_logical_tensor())));
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for matmul 0 dims dst");

        return fill_layout_info(op->get_output_value(1), dnnl::memory::desc {});
    }

    const auto &pd = matmul_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    // insert reorders for matmul's inputs
    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before matmul src 0");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before matmul src 1");

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(op, 2, pd.bias_desc(), p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder before matmul bias");
    }
    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after matmul dst");

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_pool(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = pool_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after pool dst");

    // make scratchpad as pool's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after pool scratchpad");

    if (op->has_attr(op_attr::is_training)
            && op->get_attr<bool>(op_attr::is_training)) {
        value_ptr workspace_val = op->get_output_value(2);
        const memory::desc &ws_md = pd.workspace_desc();
        workspace_val->set_dims(ws_md.get_dims());
        workspace_val->set_data_type(
                static_cast<data_type_t>(ws_md.get_data_type()));
        status = fill_layout_info(workspace_val, ws_md);
    }
    return status;
}

status_t layout_propagator_for_pool_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = pool_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status_t status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before pool_bwd "
            "diff_dst");

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after pool_bwd diff_src");

    // make scratchpad as pool's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_batchnorm(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = batchnorm_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before batchnorm src");

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after batchnorm dst");

    if (op->get_attr<bool>(op_attr::is_training)) {
        value_ptr running_mean = op->get_output_value(1);
        value_ptr running_variance = op->get_output_value(2);
        value_ptr batch_mean = op->get_output_value(3);
        value_ptr batch_variance = op->get_output_value(4);

        status = fill_layout_info(running_mean, pd.mean_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm running_mean");
        status = fill_layout_info(running_variance, pd.variance_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm running_variance");
        status = fill_layout_info(batch_mean, pd.mean_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm batch_mean");
        status = fill_layout_info(batch_variance, pd.variance_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm batch_variance");
    }

    size_t scratchpad_index = op->num_outputs() - 1;

    // if batchnorm's prop_kind is forward_training and fused with ReLU, it will
    // have a workspace output
    if (op->has_attr(op_attr::fuse_relu)
            && op->get_attr<bool>(op_attr::fuse_relu)) {
        scratchpad_index = op->num_outputs() - 2;
        value_ptr workspace_val = op->get_output_value(op->num_outputs() - 1);
        status = fill_layout_info(workspace_val, pd.workspace_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm workspace");
    }

    value_ptr scratchpad_val = op->get_output_value(scratchpad_index);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());

    return status;
}

status_t layout_propagator_for_batchnorm_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = batchnorm_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before batchnorm_bwd "
            "src");

    insert_reorder_before(op, 1, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before batchnorm_bwd "
            "diff_dst");

    insert_reorder_before(op, 2, pd.mean_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr mean = op->get_input_value(2);
    status = fill_layout_info(mean, pd.mean_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before batchnorm_bwd "
            "mean");

    insert_reorder_before(op, 3, pd.variance_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr var = op->get_input_value(3);
    status = fill_layout_info(var, pd.variance_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before batchnorm_bwd "
            "virance");

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after batchnorm_bwd "
            "diff_src");

    if (op->num_outputs() > 2) {
        value_ptr diff_gamma = op->get_output_value(1);
        value_ptr diff_beta = op->get_output_value(2);

        status = fill_layout_info(diff_gamma, pd.diff_weights_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm_bwd diff_gamma");
        status = fill_layout_info(diff_beta, pd.diff_weights_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for batchnorm_bwd diff_beta");
    }

    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_prelu(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = prelu_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before prelu src");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before prelu weights");

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after prelu dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    // make scratchpad as prelu's last output
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_prelu_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = prelu_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before prelu_bwd src");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before prelu_bwd "
            "weights");

    value_ptr diff_dst = op->get_input_value(2);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for prelu_bwd diff_dst");

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after prelu_bwd "
            "diff_src");

    insert_reorder_after(op, 1, pd.diff_weights_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr diff_wei = op->get_output_value(1);
    status = fill_layout_info(diff_wei, pd.diff_weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after prelu_bwd "
            "diff_weights");

    value_ptr scratchpad_val = op->get_output_value(2);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_layernorm(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = layernorm_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before layernorm src");

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after layernorm dst");

    if (op->num_outputs() > 2) {
        // keep_stats is true
        value_ptr mean = op->get_output_value(1);
        value_ptr variance = op->get_output_value(2);
        status = fill_layout_info(mean, pd.mean_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for layernorm mean");
        status = fill_layout_info(variance, pd.variance_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for layernorm variance");
    }

    // scratchpad is layernorm's last output
    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_layernorm_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = layernorm_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    size_t in_index {0};
    insert_reorder_before(op, in_index, pd.src_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr src = op->get_input_value(in_index++);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before layernorm_bwd "
            "src");

    insert_reorder_before(op, in_index, pd.diff_dst_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(in_index++);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before layernorm_bwd "
            "diff_dst");

    insert_reorder_before(op, in_index, pd.mean_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr mean = op->get_input_value(in_index++);
    status = fill_layout_info(mean, pd.mean_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before layernorm_bwd "
            "mean");

    insert_reorder_before(op, in_index, pd.variance_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr var = op->get_input_value(in_index++);
    status = fill_layout_info(var, pd.variance_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before layernorm_bwd "
            "variance");

    size_t out_index {0};
    insert_reorder_after(op, out_index, pd.diff_src_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(out_index++);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after layernorm_bwd "
            "diff_src");

    const bool use_affine = op->get_attr<bool>(op_attr::use_affine);
    if (use_affine) {
        const auto &diff_scale_opt_mdesc
                = pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SCALE);
        insert_reorder_after(op, out_index, diff_scale_opt_mdesc, p_engine,
                pd_cache, fpmath, use_block_layout, rewriter);
        value_ptr diff_scale = op->get_output_value(out_index++);
        status = fill_layout_info(diff_scale, diff_scale_opt_mdesc);
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder after layernorm_bwd "
                "diff_scale");

        const auto &diff_shift_opt_mdesc
                = pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SHIFT);
        insert_reorder_after(op, out_index, diff_shift_opt_mdesc, p_engine,
                pd_cache, fpmath, use_block_layout, rewriter);
        value_ptr diff_shift = op->get_output_value(out_index++);
        status = fill_layout_info(diff_shift, diff_shift_opt_mdesc);
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder after layernorm_bwd "
                "diff_shift");
    }
    auto scratchpad_val = op->get_output_value(op->num_outputs() - 1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_permute(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);

    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    auto perm = op->get_attr<std::vector<int64_t>>(op_attr::permutation);
    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);

        auto int32_perm = dnnl_impl::utils::cast_to_int32(perm);
        dnnl::memory::desc out_md = in_md.permute_axes(int32_perm);
        status = fill_layout_info(dst, out_md);
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        std::vector<int32_t> inverse_perm(perm.size(), -1);
        for (size_t i = 0; i < perm.size(); i++) {
            inverse_perm[static_cast<size_t>(perm[i])]
                    = static_cast<int32_t>(i);
        }
        dnnl::memory::desc in_md = out_md.permute_axes(inverse_perm);
        status = fill_layout_info(src, in_md);
    } else if (!ltw(in_lt).is_any() && !ltw(out_lt).is_any()) {
        // case `conv (opaque) -> permute -> output (strided)` or
        // case `input (strided) -> permute -> conv (opaque)`
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        std::vector<int32_t> inverse_perm(perm.size(), -1);
        for (size_t i = 0; i < perm.size(); i++) {
            inverse_perm[static_cast<size_t>(perm[i])]
                    = static_cast<int32_t>(i);
        }
        dnnl::memory::desc tmp_in_md = out_md.permute_axes(inverse_perm);

        // if the input md derived from output md is different from the real
        // input mem desc, just insert a reorder before the op
        if (make_dnnl_memory_desc(in_lt) != tmp_in_md)
            insert_reorder_before(op, 0, tmp_in_md, p_engine, pd_cache, fpmath,
                    use_block_layout, rewriter);
    }
    return status;
}

status_t layout_propagator_for_to_group(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);

    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md;
        auto groups = op->get_attr<int64_t>(op_attr::groups);
        // avoid dividing by zero at below.
        if (groups == 0) return status::invalid_shape;
        if (op->has_attr(op_attr::is_convtranspose)
                && op->get_attr<bool>(op_attr::is_convtranspose)) {
            auto permuted_weight = transpose(in_md, 0, 1);
            auto permuted_group_weight = to_grouped(permuted_weight, groups);
            out_md = transpose(permuted_group_weight, 1, 2);
        } else {
            out_md = to_grouped(in_md, groups);
        }
        status = fill_layout_info(dst, out_md);
    }
    return status;
}

status_t layout_propagator_for_from_group(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto get_dst_md
            = [](const dnnl::memory::desc &src_md,
                      bool is_convtranspose) -> dnnl::memory::desc {
        if (is_convtranspose) {
            auto permuted_dst = transpose(src_md, 1, 2);
            auto permuted_dst_no_groups = from_grouped(permuted_dst);
            return !permuted_dst_no_groups
                    ? permuted_dst_no_groups
                    : transpose(permuted_dst_no_groups, 0, 1);
        } else {
            return from_grouped(src_md);
        }
    };
    const auto get_strides = [](const dnnl::memory::desc &src_md,
                                     bool is_convtranspose) -> dims {
        if (is_convtranspose) {
            // chain of (transpose -> from_grouped -> transpose) requires
            // such permuted strides, otherwise reshape will fail
            auto strides
                    = get_dense_strides(transpose(src_md, 0, 1).get_dims());
            std::swap(strides[0], strides[1]);
            return strides;
        } else {
            return get_dense_strides(src_md.get_dims());
        }
    };

    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);
    auto src_lt = src->get_logical_tensor();
    auto dst_lt = dst->get_logical_tensor();

    if (ltw(src_lt).is_any()) return status;

    const bool is_convtranspose = op->has_attr(op_attr::is_convtranspose)
            ? op->get_attr<bool>(op_attr::is_convtranspose)
            : false;
    const auto src_md = make_dnnl_memory_desc(src_lt);
    dnnl::memory::desc inferred_dst_md = get_dst_md(src_md, is_convtranspose);
    // from_grouped uses the 'allow_empty' option when reshaping, so if reshape
    // will not succeed (e.g. padding exists inside dims we want to join),
    // inferred_dst_md will be an empty memory descriptor.
    if (!inferred_dst_md) {
        dnnl::memory::desc strided_dst_md(src_md.get_dims(),
                src_md.get_data_type(), get_strides(src_md, is_convtranspose));
        insert_reorder_before(op, 0, strided_dst_md, p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
        inferred_dst_md = get_dst_md(strided_dst_md, is_convtranspose);
    }

    if (ltw(dst_lt).is_any()) {
        status = fill_layout_info(dst, inferred_dst_md);
    } else {
        insert_reorder_after(op, 0, inferred_dst_md, p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
    }
    return status;
}

static status_t layout_propagator_for_reshape_like_ops(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter, const dnnl::memory::dims &target_dims) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    VCHECK_LAYOUT_PROPAGATOR(!ltw(in_lt).is_any(), status::invalid_arguments,
            "input layout must be specified for reshape_like");

    if (target_dims.empty()) {
        // scalar, just set empty strides to make the dst strided
        dst->set_strides({});
        return status::success;
    }

    if (ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md
                = in_md.reshape(target_dims, /* allow empty */ true);

        // out_md will be empty if the in_md is not reshape-able. We need to
        // reorder the in_md first and then reshape the reordered reshape-able
        // md.
        if (!out_md) {
            dnnl::memory::desc reshapable_md(in_md.get_dims(),
                    in_md.get_data_type(), get_ncx_format(in_md.get_ndims()));
            insert_reorder_before(op, 0, reshapable_md, p_engine, pd_cache,
                    fpmath, use_block_layout, rewriter);
            out_md = reshapable_md.reshape(target_dims);
        }

        status = fill_layout_info(dst, out_md);
    } else if (ltw(out_lt).is_strided()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        // check if the out_md is reshape-able
        dnnl::memory::desc expected_in_md
                = out_md.reshape(in_md.get_dims(), /* allow empty */ true);
        if (expected_in_md) {
            // If the out_md is reshape-able, the expected_in_md must be
            // reshape-able too. Then we just need to check if the real in_md
            // has same layout as the expected_in_md, and insert only one
            // possible reorder if needed.
            if (expected_in_md != in_md) {
                insert_reorder_before(op, 0, expected_in_md, p_engine, pd_cache,
                        fpmath, use_block_layout, rewriter);
            }
            // finally, we have a chain of: in_md -> (optional reorder) ->
            // expected_in_md -> reshape -> out_md
        } else {
            // Check if the in_md is reshape-able.
            dnnl::memory::desc reshaped_in_md
                    = in_md.reshape(target_dims, /* allow empty */ true);
            if (!reshaped_in_md) {
                dnnl::memory::desc reshapable_md(in_md.get_dims(),
                        in_md.get_data_type(),
                        get_ncx_format(in_md.get_ndims()));
                insert_reorder_before(op, 0, reshapable_md, p_engine, pd_cache,
                        fpmath, use_block_layout, rewriter);
                reshaped_in_md = reshapable_md.reshape(target_dims);
            }
            // If the reshaped_in_md is not same as the specified out_md, we
            // insert reorder
            if (reshaped_in_md != out_md) {
                insert_reorder_after(op, 0, reshaped_in_md, p_engine, pd_cache,
                        fpmath, use_block_layout, rewriter);
            }
            // finally, we have a chain of: in_md -> (optional reorder) ->
            // reshapable_md -> reshape -> reshaped_in_md -> (optional reorder)
            // -> out_md. The optional reorder will only occurs when both in_md
            // and out_md are not reshapable.
        }
    }
    return status;
}

status_t layout_propagator_for_reshape(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    auto out_lt = op->get_output_value(0)->get_logical_tensor();
    auto target_dims = ltw(out_lt).vdims();
    status_t status = layout_propagator_for_reshape_like_ops(op, p_engine,
            pd_cache, fpmath, use_block_layout, rewriter, target_dims);
    return status;
}

status_t layout_propagator_for_transpose(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    VCHECK_LAYOUT_PROPAGATOR(!ltw(in_lt).is_any(), status::invalid_arguments,
            "layout of transpose src can't be any layout");

    std::vector<int64_t> order
            = op->get_attr<std::vector<int64_t>>(op_attr::order);
    // if order < 0, convert it to positive order
    if (!order.empty()) {
        for (int64_t &axis : order) {
            if (axis < 0) axis += ltw(in_lt).ndims();
        }
    } else {
        // FIXME(xx) handle this case
        VCHECK_LAYOUT_PROPAGATOR(false, status::unimplemented,
                "transpose with empty order is not supported");
    }

    /// The order in spec op is used as:
    /// for (i = 0; i < ndims(); i++)
    ///     new_shape[i] = org_shape[order[i]];
    ///
    /// The axes for permute_axes function is used as:
    /// for (i = 0; i < ndims(); i++)
    ///     new_shape[axes[i]] = org_shape[i];
    ///
    /// So, we need to convert the order to axes
    std::vector<int> axes(order.size(), -1);
    for (size_t i = 0; i < order.size(); i++) {
        size_t new_shape_idx = i;
        size_t org_shape_idx = order[i];
        axes[org_shape_idx] = static_cast<int>(new_shape_idx);
    }

    // calculate the expected transposed layout by permuting the md
    dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
    dnnl::memory::desc expected_out_md = in_md.permute_axes(axes);
    if (ltw(out_lt).is_any()) {
        status = fill_layout_info(dst, expected_out_md);
    } else {
        // if the output layout is specified, we need to check if it's matched
        // with the expected out layout. If not, we should insert a reorder op
        // to convert the transposed layout to the specified one.
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        if (expected_out_md != out_md) {
            insert_reorder_after(op, 0, expected_out_md, p_engine, pd_cache,
                    fpmath, use_block_layout, rewriter);
        }
    }
    return status;
}

status_t layout_propagator_for_unsqueeze(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        // 'out_lt' shape should be known at this stage
        status = fill_layout_info(dst, in_md.reshape(ltw(out_lt).vdims()));
    }
    return status;
}

status_t layout_propagator_for_squeeze(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    auto out_lt = op->get_output_value(0)->get_logical_tensor();
    auto target_dims = ltw(out_lt).vdims();
    status_t status = layout_propagator_for_reshape_like_ops(op, p_engine,
            pd_cache, fpmath, use_block_layout, rewriter, target_dims);
    return status;
}

status_t layout_propagator_for_reorder(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        VCHECK_LAYOUT_PROPAGATOR(!op->has_attr(op_attr::change_layout)
                        || !op->get_attr<bool>(op_attr::change_layout),
                status::invalid_arguments,
                "layout of dnnl_reorder input and output must be known "
                "if it changes layout");
        // for logical tensor with opaque layout, make_dnnl_memory_desc
        // cannot help manually modify the logical tensor. The previously
        // created memory will be quired according to logic tensor id.
        // So data_type of quired memory may be not same as the logical tensor.
        // We used to be able to change the data type manually in oneDNN
        // v2.7, but we can't modify the data type in this way anymore
        // since v3.0.
        auto out_md = make_dnnl_memory_desc(in_lt);
        if (in_lt.data_type != out_lt.data_type) {
            auto format_tag = md2fmt_tag_str(out_md.get());
            const auto &dims = out_md.get_dims();
            dnnl_memory_desc_t tmp_md;
            dnnl_memory_desc_create_with_string_tag(&tmp_md,
                    static_cast<int>(dims.size()), dims.data(),
                    static_cast<dnnl_data_type_t>(out_lt.data_type),
                    format_tag.data());
            status = fill_layout_info(dst, tmp_md);
        } else {
            status = fill_layout_info(dst, out_md);
        }
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        VCHECK_LAYOUT_PROPAGATOR(!op->has_attr(op_attr::change_layout)
                        || !op->get_attr<bool>(op_attr::change_layout),
                status::invalid_arguments,
                "layout of dnnl_reorder input and output must be known "
                "if it changes layout");
        out_lt.data_type = ltw(in_lt).data_type();
        auto in_md = make_dnnl_memory_desc(out_lt);
        status = fill_layout_info(src, in_md);
    }
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder");

    // set layout info for scratchpad output
    if (op->num_outputs() == 1) { insert_empty_scratchpad(op); }

    const auto &pd = reorder_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_mul_scales(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    return layout_propagator_for_reorder(
            op, p_engine, pd_cache, fpmath, use_block_layout, rewriter);
}

status_t layout_propagator_for_bn_folding(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    // skip the scratchpad
    for (size_t i = 0; i < op->num_outputs() - 1; i++) {
        auto in_lt = op->get_input_logical_tensor(i);
        auto out_lt = op->get_output_logical_tensor(i);
        if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
            dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
            auto dst = op->get_output_value(i);
            status = fill_layout_info(dst, in_md);
            VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                    "failed to fill layout info for bn_folding dst");
        }
    }

    auto pd = bn_folding_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);
    // scratchpad is bn_folding's last inputs
    auto val = op->get_output_value(2);
    status = fill_layout_info(val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_conv_bwd_data(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = conv_bwd_data_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status_t status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv_bwd_data "
            "diff_dst");

    insert_reorder_before(op, 1, pd.weights_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv_bwd_data "
            "weights");

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after conv_bwd_data "
            "diff_src");

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_conv_bwd_weights(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = conv_bwd_weights_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr src = op->get_input_value(0);
    status_t status = fill_layout_info(src, pd.src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv_bwd_weights "
            "src");

    insert_reorder_before(op, 1, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before conv_bwd_weights "
            "diff_dst");

    insert_reorder_after(op, 0, pd.diff_weights_desc(), p_engine, pd_cache,
            fpmath, use_block_layout, rewriter);
    value_ptr diff_weights = op->get_output_value(0);
    status = fill_layout_info(diff_weights, pd.diff_weights_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after conv_bwd_weights "
            "diff_weights");

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_resampling(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = resampling_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after interpolate "
            "dst");

    // make scratchpad as resampling's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_resampling_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = resampling_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status_t status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after interpolate_bwd "
            "diff_src");

    auto scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_sum(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    value_ptr dst = op->get_output_value(0);
    bool input_has_any_format = false;
    for (const auto &in_val : op->get_input_values()) {
        if (ltw(in_val->get_logical_tensor()).is_any()) {
            input_has_any_format = true;
            break;
        }
    }

    MAYBE_UNUSED(input_has_any_format);
    assertm(!input_has_any_format,
            "input format of sum primitive cannot be any");
    VCHECK_LAYOUT_PROPAGATOR(!input_has_any_format, status::invalid_arguments,
            "input format of sum primitive cannot be any ");

    const auto &pd = sum_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    if (ltw(dst->get_logical_tensor()).is_any()) {
        insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
        dst = op->get_output_value(0);
        status_t status = fill_layout_info(dst, pd.dst_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for reorder after sum dst");
    }

    // scratchpad is dnnl_sum's last output
    value_ptr scratchpad_val = op->get_output_values().back();
    status_t status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_softmax(op_ptr &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    value_ptr src = op->get_input_value(0);
    VCHECK_LAYOUT_PROPAGATOR(!ltw(src->get_logical_tensor()).is_any(),
            status::invalid_arguments,
            "layout of softmax/logsoftmax src can't be any");

    const auto &pd = softmax_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after softmax dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_softmax_bwd(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    value_ptr dst = op->get_input_value(1);
    VCHECK_LAYOUT_PROPAGATOR(!ltw(dst->get_logical_tensor()).is_any(),
            status::invalid_arguments,
            "layout of softmax/logsoftmax bwd dst can't be any");

    const auto &pd = softmax_bwd_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_before(op, 0, pd.diff_dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status_t status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder before softmax_bwd "
            "diff_dst");

    insert_reorder_after(op, 0, pd.diff_src_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after softmax_bwd "
            "diff_src");

    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_reduction(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    value_ptr src = op->get_input_value(0);
    VCHECK_LAYOUT_PROPAGATOR(!ltw(src->get_logical_tensor()).is_any(),
            status::invalid_arguments, "layout of reduction src can't be any");
    const auto &pd = reduction_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after reduction dst");

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_constant_filler(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    return status::success;
}

status_t layout_propagator_for_sub_zps(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    assertm(false,
            "dnnl_sub_zps op is only for fusion purpose, we shouldn't do "
            "layout propagation for it");
    return status::invalid_graph_op;
}

status_t layout_propagator_for_add_zps(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    assertm(false,
            "dnnl_add_zps op is only for fusion purpose, we shouldn't do "
            "layout propagation for it");
    return status::invalid_graph_op;
}

status_t layout_propagator_for_gen_index(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    auto src_md = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    if (!is_plain(src_md)) {
        src_md = dnnl::memory::desc(src_md.get_dims(), src_md.get_data_type(),
                dnnl::memory::format_tag::abcd);
        insert_reorder_before(op, 0, src_md, p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
    }
    value_ptr dst_val = op->get_output_value(0);
    status_t status = fill_layout_info(dst_val, src_md);
    return status;
}

status_t layout_propagator_for_identity(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    logical_tensor_t dst_lt = op->get_input_logical_tensor(0);
    VCHECK_LAYOUT_PROPAGATOR(!ltw(dst_lt).is_any(), status::invalid_arguments,
            "layout of identity dst can't be any");
    auto dst_md = make_dnnl_memory_desc(dst_lt);
    auto src_md = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    if (src_md != dst_md) {
        insert_reorder_before(op, 0, dst_md, p_engine, pd_cache, fpmath,
                use_block_layout, rewriter);
    }
    return status::success;
}

status_t layout_propagator_for_groupnorm(op_ptr &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    const auto &pd = groupnorm_executable_t::create_desc(
            op, p_engine, pd_cache, fpmath, use_block_layout);

    insert_reorder_after(op, 0, pd.dst_desc(), p_engine, pd_cache, fpmath,
            use_block_layout, rewriter);
    value_ptr dst = op->get_output_value(0);
    status_t status = fill_layout_info(dst, pd.dst_desc());
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for reorder after groupnorm dst");

    if (op->num_outputs() > 2) {
        // keep_stats is true
        value_ptr mean = op->get_output_value(1);
        value_ptr variance = op->get_output_value(2);
        status = fill_layout_info(mean, pd.mean_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for groupnorm mean");
        status = fill_layout_info(variance, pd.variance_desc());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for groupnorm variance");
    }

    // scratchpad is groupnorm's last output
    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_mask(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    auto src_md = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    value_ptr dst_val = op->get_output_value(0);
    status_t status = fill_layout_info(dst_val, src_md);
    return status;
}

status_t layout_propagator_for_sdpa(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(pd_cache);
    UNUSED(use_block_layout);
    UNUSED(rewriter);

    size_t output_idx = 0;
    value_ptr dst_val = op->get_output_value(output_idx++);
    const logical_tensor_t &out_lt = dst_val->get_logical_tensor();

    dnnl::memory::desc expected_md;
    if (ltw(out_lt).is_any()) {
        // For GQA, we need to check the layout of the dnnl_reshape output
        // following dnnl_sdpa, which is given by the user.
        if (!dst_val->get_consumers().empty()) {
            const auto &consumer_op = dst_val->get_consumers()[0].get_op();
            const logical_tensor_t &consumer_out
                    = consumer_op.get_output_logical_tensor(0);
            if (consumer_op.get_kind() == op_kind::_reshape
                    && ltw(consumer_out).ndims() == 5
                    && ltw(consumer_out).is_strided()) {
                const auto &ori_strides = ltw(consumer_out).vstrides();
                std::vector<dim_t> strides = {ori_strides[0], ori_strides[2],
                        ori_strides[3], ori_strides[4]};
                expected_md = {ltw(out_lt).vdims(),
                        static_cast<dnnl::memory::data_type>(
                                ltw(out_lt).data_type()),
                        strides};
            } else {
                // Set default output layout format for sdpa as acbd if user
                // doesn't specify the layout since no reorder will be required.
                expected_md = {ltw(out_lt).vdims(),
                        static_cast<dnnl::memory::data_type>(
                                ltw(out_lt).data_type()),
                        dnnl::memory::format_tag::acbd};
            }
        } else {
            expected_md = {ltw(out_lt).vdims(),
                    static_cast<dnnl::memory::data_type>(
                            ltw(out_lt).data_type()),
                    dnnl::memory::format_tag::acbd};
        }
    } else {
        expected_md = make_dnnl_memory_desc(out_lt);
    }
    status_t status = fill_layout_info(dst_val, expected_md);

    // Size the scratchpad output from the sdpa primitive descriptor (built here
    // and discarded) so memory planning reserves a real buffer before
    // compile_ops creates the executable. GPU micro-SDPA needs no user
    // scratchpad, but the CPU BRGEMM primitive does.
    value_ptr scratchpad_val = op->get_output_value(output_idx++);
    memory::desc scratchpad_desc;
    std::unique_ptr<dnnl_primitive_desc, pd_deleter_t> sdpa_pd;
    if (create_sdpa_pd(sdpa_pd, op, p_engine, fpmath) == status::success) {
        const_dnnl_memory_desc_t c_scratchpad_md = dnnl_primitive_desc_query_md(
                sdpa_pd.get(), dnnl_query_scratchpad_md, 0);
        dnnl_memory_desc_t cloned_md = nullptr;
        dnnl_memory_desc_clone(&cloned_md, c_scratchpad_md);
        scratchpad_desc = memory::desc(cloned_md);
    }
    status = fill_layout_info(scratchpad_val, scratchpad_desc);

    if (op->get_attr<bool>(op_attr::is_training)) {
        value_ptr stats_val = op->get_output_value(output_idx);
        const logical_tensor_t &stats_lt = stats_val->get_logical_tensor();
        const auto stats_dims = ltw(stats_lt).vdims();
        const auto dense_strides = get_dense_strides(stats_dims);
        auto stats_strides = dense_strides;
        if (!stats_val->get_consumers().empty()) {
            const auto &consumer_op = stats_val->get_consumers()[0].get_op();
            const logical_tensor_t &consumer_out
                    = consumer_op.get_output_logical_tensor(0);
            if (consumer_op.get_kind() == op_kind::_reshape
                    && ltw(consumer_out).ndims() == 5
                    && ltw(consumer_out).is_strided()
                    && stats_dims.size() == 4) {
                const auto &ori_strides = ltw(consumer_out).vstrides();
                stats_strides = {ori_strides[0], ori_strides[2], ori_strides[3],
                        ori_strides[4]};
            }
        }
        const dnnl::memory::desc stats_md {stats_dims,
                static_cast<dnnl::memory::data_type>(ltw(stats_lt).data_type()),
                stats_strides};
        status = fill_layout_info(stats_val, stats_md);
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "sdpa: failed to fill layout info for the softmax stats");
    }
    return status;
}

status_t layout_propagator_for_sdpa_bwd(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    UNUSED(pd_cache);
    UNUSED(use_block_layout);
    UNUSED(rewriter);

    // Helper: derive a memory desc for a diff output from the corresponding
    // forward input logical tensor. If the input layout is already fixed, reuse
    // it; otherwise fall back to the canonical acbd format used by sdpa.
    auto get_md_for_diff = [](const logical_tensor_t &lt) {
        if (!ltw(lt).is_any()) return make_dnnl_memory_desc(lt);
        return dnnl::memory::desc {ltw(lt).vdims(),
                static_cast<dnnl::memory::data_type>(ltw(lt).data_type()),
                dnnl::memory::format_tag::acbd};
    };

    status_t status = status::success;
    size_t output_idx = 0;

    // diff_query (output 0): propagate layout from query (input 0)
    value_ptr diff_query_val = op->get_output_value(output_idx++);
    const logical_tensor_t &query_lt = op->get_input_logical_tensor(0);
    status = fill_layout_info(diff_query_val, get_md_for_diff(query_lt));
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for sdpa_bwd diff_query");

    // diff_key (output 1): propagate layout from key (input 1)
    value_ptr diff_key_val = op->get_output_value(output_idx++);
    const logical_tensor_t &key_lt = op->get_input_logical_tensor(1);
    status = fill_layout_info(diff_key_val, get_md_for_diff(key_lt));
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for sdpa_bwd diff_key");

    // diff_value (output 2): propagate layout from value (input 2)
    value_ptr diff_value_val = op->get_output_value(output_idx++);
    const logical_tensor_t &value_lt = op->get_input_logical_tensor(2);
    status = fill_layout_info(diff_value_val, get_md_for_diff(value_lt));
    VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
            "failed to fill layout info for sdpa_bwd diff_value");

    const auto mask_type = static_cast<attn_mask_type_t>(
            op->get_attr<int64_t>(op_attr::mask_type));

    VCHECK_LAYOUT_PROPAGATOR(mask_type != attn_mask_type::select,
            status::unimplemented, "sdpa_bwd does not support select masks");

    // scratchpad (output 3): create the pd to get the real scratchpad size
    {
        const bool with_scale = op->get_attr<bool>(op_attr::with_scale);
        const bool is_invert_scale = op->has_attr(op_attr::is_invert_scale)
                ? op->get_attr<bool>(op_attr::is_invert_scale)
                : false;
        const bool with_explicit_mask = mask_type == attn_mask_type::buffer;

        auto md_q = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
        auto md_k = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
        auto md_v = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
        auto md_dst = make_dnnl_memory_desc(op->get_input_logical_tensor(3));
        auto md_diff_dst
                = make_dnnl_memory_desc(op->get_input_logical_tensor(5));
        auto md_diff_q = get_md_for_diff(op->get_input_logical_tensor(0));
        auto md_diff_k = get_md_for_diff(op->get_input_logical_tensor(1));
        auto md_diff_v = get_md_for_diff(op->get_input_logical_tensor(2));

        dnnl::memory::desc md_scale, md_attn_mask, md_dS;
        size_t idx = 6;
        if (with_scale)
            md_scale = make_dnnl_memory_desc(
                    op->get_input_logical_tensor(idx++));
        if (with_explicit_mask) {
            md_attn_mask = make_dnnl_memory_desc(
                    op->get_input_logical_tensor(idx++));
            if (op->num_outputs() > 4)
                md_dS = make_dnnl_memory_desc(op->get_output_logical_tensor(4));
        }

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
        qk_attr.set_accumulation_mode(str2accumulation_mode(
                op->get_attr<std::string>(op_attr::qk_acc_mode)));
        vs_attr.set_accumulation_mode(str2accumulation_mode(
                op->get_attr<std::string>(op_attr::vs_acc_mode)));
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        attr.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(fpmath.mode_));

        dim_t kv_head_number = op->get_input_logical_tensor(1).dims[1];
        const alg_kind_t softmax_alg = alg_kind::softmax_accurate_inf_as_zero;

        std::shared_ptr<primitive_desc_t> hint_fwd_pd;
        status = create_sdpa_pd(hint_fwd_pd, p_engine.get(), md_q.get(),
                md_k.get(), md_v.get(), md_dst.get(), md_attn_mask.get(),
                md_scale.get(), is_invert_scale, kv_head_number, mask_type,
                softmax_alg, /* fill_md = */ nullptr,
                /* invert_select = */ false, impl::prop_kind::forward_training,
                attr.get(), qk_attr.get(), vs_attr.get());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to create hint fwd pd for sdpa_bwd scratchpad");

        std::shared_ptr<primitive_desc_t> sdpa_bwd_pd;
        status = create_sdpa_pd(sdpa_bwd_pd, p_engine.get(), md_q.get(),
                md_k.get(), md_v.get(), md_dst.get(), md_attn_mask.get(),
                md_scale.get(), md_diff_q.get(), md_diff_k.get(),
                md_diff_v.get(), md_diff_dst.get(), md_dS.get(),
                is_invert_scale, kv_head_number, mask_type, softmax_alg,
                attr.get(), hint_fwd_pd.get(), qk_attr.get(), vs_attr.get());
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to create pd for sdpa_bwd scratchpad");

        value_ptr scratchpad_val = op->get_output_value(output_idx++);
        dnnl_memory_desc_t cloned_md = nullptr;
        dnnl_memory_desc_clone(&cloned_md, sdpa_bwd_pd->scratchpad_md());
        dnnl::memory::desc scratchpad_desc;
        scratchpad_desc.reset(cloned_md);
        status = fill_layout_info(scratchpad_val, scratchpad_desc);
    }

    // diff_mask (output 4, optional)
    if (op->num_outputs() > output_idx) {
        value_ptr diff_mask_val = op->get_output_value(output_idx);
        const logical_tensor_t &diff_mask_lt
                = diff_mask_val->get_logical_tensor();
        status = fill_layout_info(
                diff_mask_val, make_dnnl_memory_desc(diff_mask_lt));
        VCHECK_LAYOUT_PROPAGATOR(status == status::success, status,
                "failed to fill layout info for sdpa_bwd diff_mask");
    }

    return status;
}

status_t layout_propagator_for_host_scalar(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout,
        subgraph_rewriter_t &rewriter) {
    // no need to do layout propagation for host scalar
    // as its output is always strided
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    UNUSED(rewriter);
    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
