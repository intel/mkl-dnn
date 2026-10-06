/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/backend/dnnl/kernels/sdp_decomp_config.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#include "common/dnnl_thread.hpp"

#define VCHECK_SDP_DECOMP(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_decomp_kernel_t, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bool sdp_decomp_config_t::initial_check(const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    // The order of input logical tensors in inputs is not certain, we need
    // to record the input offset in a certain order of ops.
    CHECK_BOOL(record_input_offset(sg, inputs));
    dims src1_user_dims = ltw(inputs[graph_inport[mm1_src]]).vdims();
    ndims = src1_user_dims.size();
    VCHECK_SDP_DECOMP(ndims == 4 || ndims == 5, false,
            "Input dims should be 4 or 5, but got %zu", src1_user_dims.size());
    VCHECK_SDP_DECOMP(
            outputs.size() == 1, false, "does not support multiple outputs");

    // Initialize SDP input dimension according to the src of mm1
    int index = 0;
    batch_size = src1_user_dims[index++];
    num_head_q = src1_user_dims[index++];
    if (ndims == 5) { num_head_q *= src1_user_dims[index++]; }
    seq_len_q = src1_user_dims[index++];
    head_size_qk = src1_user_dims[index++];

    dims wei1_user_dims = ltw(inputs[graph_inport[mm1_wei]]).vdims();
    dims wei2_user_dims = ltw(inputs[graph_inport[mm2_wei]]).vdims();
    num_head_kv = wei1_user_dims[1];
    VCHECK_SDP_DECOMP(num_head_kv == wei2_user_dims[1], false,
            "kv head number mismatch, kv head number: %ld, wei1: %ld, wei2: "
            "%ld",
            static_cast<long int>(num_head_kv),
            static_cast<long int>(wei1_user_dims[1]),
            static_cast<long int>(wei2_user_dims[1]));

    // Check batch size compatibility.

    VCHECK_SDP_DECOMP(
            batch_size == wei1_user_dims[0] && batch_size == wei2_user_dims[0],
            false, "Batch size mismatch, batch_size: %ld, wei1: %ld, wei2: %ld",
            static_cast<long int>(batch_size),
            static_cast<long int>(wei1_user_dims[0]),
            static_cast<long int>(wei2_user_dims[0]));

    head_size_v = wei2_user_dims.back();
    // Check scale size
    if (graph_inport[mm1_scale] != -1) {
        auto scale_sz = ltw(inputs[graph_inport[mm1_scale]]).nelems();
        VCHECK_SDP_DECOMP(scale_sz == 1, false,
                "Only supports single scale value, but got %ld",
                static_cast<long int>(scale_sz));
    }

    // Check soft-capping size
    if (graph_inport[mm1_soft_capping] != -1) {
        auto scale_sz = ltw(inputs[graph_inport[mm1_soft_capping]]).nelems();
        VCHECK_SDP_DECOMP(scale_sz == 1, false,
                "Only supports single scale value for soft-capping, but got "
                "%ld",
                static_cast<long int>(scale_sz));
    }

    VCHECK_SDP_DECOMP(ltw(inputs[graph_inport[mm1_wei]]).data_type()
                    == ltw(inputs[graph_inport[mm2_wei]]).data_type(),
            false,
            "Key and value should have the same data type. But got key:%s, "
            "value:%s",
            dnnl_dt2str(ltw(inputs[graph_inport[mm1_wei]]).data_type()),
            dnnl_dt2str(ltw(inputs[graph_inport[mm2_wei]]).data_type()));

    // Initialize nthr with max threads num
    nthr = dnnl_get_max_threads();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
// RATIO is an empirical value used to determine the numerical relationship
// between batch_size, num_head_q and thread number to determine whether to use
// decompose kernel. The key to the decompose kernel is that we do parallel in
// the batch_size and num_head_q dimensions. Therefore, if the batch_size or
// num_head_q is too small, it will cause many idle threads and affect
// efficiency which may even worse than the original sequential kernel. Here we
// set this ratio based on the experimental value to ensure that users do not
// have any regression when using the decompose kernel.
// TODO: Refine the inequation based on the relationship of cache size and sdp
// memory footprint requirements.
#define RATIO 2
    VCHECK_SDP_DECOMP(batch_size * num_head_q > RATIO * nthr, false,
            "Doesn't meet condition for decompose: Batch size * num_head_q "
            "should be larger than ratio * nthr, but got batch_size %ld, "
            "num_head_q %ld, ration %d , nthr %d",
            static_cast<long int>(batch_size),
            static_cast<long int>(num_head_q), RATIO, nthr);
#endif
    return true;
}

template <bool quantized, memory::data_type dt>
impl::status_t sdp_decomp_config_t::construct_params(
        std::shared_ptr<subgraph_t> &sg, registry_t &sdp_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs) {

    // Record the ops inside of SDP pattern for later usage
    CHECK(record_sdp_ops(sg, quantized));
    const dim_t last_dim = ndims - 1, second_last_dim = ndims - 2;

    // Update SDPA input params. Sequence length for query and key/value are
    // NOT always same.
    const auto &lt_wei = sdp_op[1]->get_input_logical_tensor(1);
    const ltw ltw_wei(lt_wei);
    seq_len_kv = ltw_wei.vdims()[last_dim];

    // Acquire the data type from input param for later primitive creation.
    // The src and wei dt of both quantized sdp and float sdp are the same.
    memory::data_type dt_src_user = static_cast<memory::data_type>(
            ltw(inputs[graph_inport[mm1_src]]).data_type());
    memory::data_type dt_wei_user = static_cast<memory::data_type>(
            ltw(inputs[graph_inport[mm1_wei]]).data_type());
    memory::data_type dt_wei
            = (quantized && dt_wei_user == memory::data_type::u8)
            ? memory::data_type::s8
            : dt_wei_user;
    memory::data_type dt_inter = quantized
            ? dt
            : static_cast<memory::data_type>(
                      ltw(sdp_op[1]->get_output_logical_tensor(0)).data_type());

    ////////////////////////////////////////////////////////////////////////
    ////////////// Start Creating primitives ///////////////////////////////
    ////////////////////////////////////////////////////////////////////////
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    // TODO: Here we create primitive with single thread, no exact reason,
    // pending on primitive investigation and fix
    omp_set_num_threads(1);
#endif
    // intermediate md used to create primitives
    memory::desc sub_src1_md, sub_wei1_user_md, sub_mm1_src_md, sub_mm1_wei_md,
            sub_mm1_dst_md, sub_softmax_dst_md, sub_wei2_user_md,
            sub_mm2_wei_md, sub_mm2_dst_md, sub_dst_user_md, sub_select_cond_md,
            sub_select_src_md;
    std::vector<memory::desc> sub_mm1_post_md, sub_softmax_post_md;

    // per-head source view for first matmul
    dims sub_src1_dims = {seq_len_q, head_size_qk};
    src1_strides = ltw(inputs[graph_inport[mm1_src]]).vstrides();
    sub_src1_md = memory::desc(sub_src1_dims, dt_src_user,
            {src1_strides[second_last_dim], src1_strides[last_dim]});
    auto sub_src1_d_md
            = memory::desc(sub_src1_dims, dt_src_user, format_tag::ab);

    // per-head: reorder u8->s8 wei for first matmul
    dims sub_wei1_dims = {head_size_qk, seq_len_kv};
    auto wei_md = make_dnnl_memory_desc(sdp_op[1]->get_input_logical_tensor(1));
    wei1_strides = wei_md.get_strides();
    sub_wei1_user_md = memory::desc(sub_wei1_dims, dt_wei_user,
            {wei1_strides[second_last_dim], wei1_strides[last_dim]});
    auto sub_wei1_d_md = memory::desc(sub_wei1_dims, dt_wei, format_tag::ba);
    dnnl::primitive_attr sub_reorder1_attr = make_primitive_attr(sdp_op[0]);

    // first matmul
    // create first matmul primitive attr
    dnnl::primitive_attr sub_matmul1_attr = make_primitive_attr(sdp_op[1]);
    dims sub_mm1_wei_dims = {head_size_qk, seq_len_kv};
    dims sub_mm1_dst_dims = {seq_len_q, seq_len_kv};

    sub_mm1_dst_md = memory::desc(sub_mm1_dst_dims, dt_inter, format_tag::ab);
    dnnl::post_ops dnnl_pops;
    auto mm1_ori_dnnl_pops = sub_matmul1_attr.get_post_ops();
    auto make_sub_md = [&](const dnnl::impl::memory_desc_t &ori_desc,
                               dim_t second_last_dim,
                               dim_t last_dim) -> dnnl::memory::desc {
        auto post_shape = ori_desc.dims;
        auto post_stride = ori_desc.format_desc.blocking.strides;
        auto post_dt = static_cast<dnnl::memory::data_type>(ori_desc.data_type);
        dims post_stride_dims
                = {post_stride[second_last_dim], post_stride[last_dim]};
        return dnnl::memory::desc(
                {post_shape[second_last_dim], post_shape[last_dim]}, post_dt,
                post_stride_dims);
    };
    for (int i = 0; i < mm1_ori_dnnl_pops.get()->len(); i++) {
        if (mm1_ori_dnnl_pops.get()->entry_[i].is_binary()) {
            auto alg = static_cast<algorithm>(
                    mm1_ori_dnnl_pops.get()->entry_[i].binary.alg);
            if (alg == algorithm::binary_select) {
                const dnnl::impl::memory_desc_t &src1_desc
                        = mm1_ori_dnnl_pops.get()
                                  ->entry_[i]
                                  .binary.user_src1_desc;
                auto select_sub_src1_md
                        = make_sub_md(src1_desc, second_last_dim, last_dim);
                sub_mm1_post_md.emplace_back(select_sub_src1_md);
                const dnnl::impl::memory_desc_t &cond_desc
                        = mm1_ori_dnnl_pops.get()
                                  ->entry_[i]
                                  .binary.user_src2_desc;
                auto select_sub_cond_md
                        = make_sub_md(cond_desc, second_last_dim, last_dim);
                sub_mm1_post_md.emplace_back(select_sub_cond_md);
                dnnl_pops.append_binary(
                        alg, select_sub_src1_md, select_sub_cond_md);
            } else {
                const dnnl::impl::memory_desc_t &ori_desc
                        = mm1_ori_dnnl_pops.get()
                                  ->entry_[i]
                                  .binary.user_src1_desc;
                auto new_sub_md
                        = make_sub_md(ori_desc, second_last_dim, last_dim);
                sub_mm1_post_md.emplace_back(new_sub_md);
                dnnl_pops.append_binary(alg, new_sub_md);
            }
        } else if (mm1_ori_dnnl_pops.get()->entry_[i].is_eltwise()) {
            auto alg = static_cast<algorithm>(
                    mm1_ori_dnnl_pops.get()->entry_[i].eltwise.alg);
            auto alpha = mm1_ori_dnnl_pops.get()->entry_[i].eltwise.alpha;
            auto beta = mm1_ori_dnnl_pops.get()->entry_[i].eltwise.beta;
            dnnl_pops.append_eltwise(alg, alpha, beta);
        }
    }
    sub_matmul1_attr.set_post_ops(dnnl_pops);
    CHECK(sub_mm1_prim.init(p_engine, sub_src1_md, sub_src1_d_md,
            primitive_attr {}, sub_wei1_user_md, sub_wei1_d_md,
            sub_reorder1_attr, sub_mm1_dst_md, sub_matmul1_attr));
    const auto &query_plan = sub_mm1_prim.query_plan();
    const auto &key_plan = sub_mm1_prim.key_plan();
    sub_mm1_src_md = query_plan.selected_md;
    sub_mm1_wei_md = key_plan.selected_md;
    CHECK(sub_reorder0.init(
            p_engine, sub_src1_md, sub_mm1_src_md, primitive_attr {}));
    CHECK(sub_reorder1.init(
            p_engine, sub_wei1_user_md, sub_mm1_wei_md, sub_reorder1_attr));

    //select
    if (has_select && !select_fusiable) {
        dnnl::primitive_attr sub_select_attr = make_primitive_attr(sdp_op[5]);
        auto select_cond_lt = sdp_op[5]->get_input_logical_tensor(2);
        auto select_cond_ltw = ltw(select_cond_lt);
        auto select_src0_lt = sdp_op[5]->get_input_logical_tensor(0);
        auto select_src0_ltw = ltw(select_src0_lt);
        sub_select_cond_md = memory::desc(
                {select_cond_ltw.vdims()[second_last_dim],
                        select_cond_ltw.vdims()[last_dim]},
                static_cast<memory::data_type>(select_cond_ltw.data_type()),
                {select_cond_ltw.vstrides()[second_last_dim],
                        select_cond_ltw.vstrides()[last_dim]});
        sub_select_src_md = memory::desc(
                {select_src0_ltw.vdims()[second_last_dim],
                        select_src0_ltw.vdims()[last_dim]},
                static_cast<memory::data_type>(select_src0_ltw.data_type()),
                {select_src0_ltw.vstrides()[second_last_dim],
                        select_src0_ltw.vstrides()[last_dim]});
        auto sub_select_pd = binary::primitive_desc(p_engine,
                algorithm::binary_select, sub_select_src_md, sub_mm1_dst_md,
                sub_select_cond_md, sub_mm1_dst_md, sub_select_attr);
        sub_select_prim = binary(sub_select_pd);
    }

    // softmax
    // create softmax primitive attr
    dnnl::primitive_attr sub_softmax_attr = make_primitive_attr(sdp_op[2]);

    dnnl_pops = {};
    auto softmax_ori_dnnl_pops = sub_softmax_attr.get_post_ops();
    for (int i = 0; i < softmax_ori_dnnl_pops.get()->len(); i++) {
        const auto alg = static_cast<algorithm>(
                softmax_ori_dnnl_pops.get()->entry_[i].binary.alg);
        const dnnl::impl::memory_desc_t &ori_desc
                = softmax_ori_dnnl_pops.get()->entry_[i].binary.user_src1_desc;
        auto post_shape = ori_desc.dims;
        auto post_stride = ori_desc.format_desc.blocking.strides;
        auto post_dt = static_cast<memory::data_type>(ori_desc.data_type);
        auto new_sub_md = memory::desc(
                {post_shape[second_last_dim], post_shape[last_dim]}, post_dt,
                {post_stride[second_last_dim], post_stride[last_dim]});
        sub_softmax_post_md.emplace_back(new_sub_md);
        dnnl_pops.append_binary(alg, new_sub_md);
    }
    sub_softmax_attr.set_post_ops(dnnl_pops);

    sub_softmax_dst_md
            = memory::desc(sub_mm1_dst_dims, dt_src_user, format_tag::ab);
    const auto mode = sdp_op[2]->get_attr<std::string>(op_attr::mode);
    const dnnl::algorithm algo = mode == "inf_as_zero"
            ? static_cast<dnnl::algorithm>(
                      dnnl::impl::alg_kind::softmax_accurate_inf_as_zero)
            : dnnl::algorithm::softmax_accurate;
    auto sub_softmax_pd = softmax_forward::primitive_desc(p_engine,
            prop_kind::forward_inference, algo, sub_mm1_dst_md,
            sub_softmax_dst_md, sub_mm1_dst_md.get_ndims() - 1,
            sub_softmax_attr);
    sub_softmax_prim = softmax_forward(sub_softmax_pd);

    // reorder u8->s8 wei for second matmul
    dims sub_wei2_dims = {seq_len_kv, head_size_v};
    wei2_strides = ltw(inputs[graph_inport[mm2_wei]]).vstrides();
    sub_wei2_user_md = memory::desc(sub_wei2_dims, dt_wei_user,
            {wei2_strides[second_last_dim], wei2_strides[last_dim]});
    auto sub_wei2_d_md = memory::desc(sub_wei2_dims, dt_wei, format_tag::ab);
    dnnl::primitive_attr sub_reorder2_attr = make_primitive_attr(sdp_op[3]);

    // second matmul
    // create second matmul primitive attr
    dnnl::primitive_attr sub_matmul2_attr = make_primitive_attr(sdp_op[4]);
    dims sub_mm2_src_dims = {seq_len_q, seq_len_kv};
    dims sub_mm2_wei_dims = {seq_len_kv, head_size_v};
    dims sub_mm2_dst_dims = {seq_len_q, head_size_v};
    auto sub_mm2_src_md
            = memory::desc(sub_mm2_src_dims, dt_src_user, format_tag::ab);
    auto out_lt = sdp_op[4]->get_output_logical_tensor(0);
    dst_strides = ltw(out_lt).vstrides();
    sub_dst_user_md = memory::desc(sub_mm2_dst_dims, dt_src_user,
            {dst_strides[second_last_dim], dst_strides[last_dim]});
    auto sub_dst_dense_md
            = memory::desc(sub_mm2_dst_dims, dt_src_user, format_tag::ab);
    CHECK(sub_mm2_prim.init(p_engine, sub_mm2_src_md, sub_wei2_user_md,
            sub_wei2_d_md, sub_reorder2_attr, sub_dst_user_md, sub_dst_dense_md,
            primitive_attr {}, sub_matmul2_attr));
    const auto &value_plan = sub_mm2_prim.value_plan();
    const auto &output_plan = sub_mm2_prim.output_plan();
    sub_mm2_wei_md = value_plan.selected_md;
    sub_mm2_dst_md = output_plan.selected_md;
    CHECK(sub_reorder2.init(
            p_engine, sub_wei2_user_md, sub_mm2_wei_md, sub_reorder2_attr));
    CHECK(sub_reorder3.init(
            p_engine, sub_mm2_dst_md, sub_dst_user_md, primitive_attr {}));
    ////////////////////////////////////////////////////////////////////////
    /////////////// End Creating primitives ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    /////////////// Start Constructing exec args ///////////////////////////
    ////////////////////////////////////////////////////////////////////////
    memory::desc max_scratchpad_md, sub_max_src1_src2_md, sub_max_dst1_wei2_md;
    size_t max_scratchpad_size = 0;
    // all the scratchpads required by the primitives.
    const std::vector<memory::desc> scratchpads {sub_reorder0.scratchpad_desc(),
            sub_reorder1.scratchpad_desc(), sub_mm1_prim.scratchpad_desc(),
            sub_softmax_pd.scratchpad_desc(), sub_reorder2.scratchpad_desc(),
            sub_mm2_prim.scratchpad_desc(), sub_reorder3.scratchpad_desc()};

    for (auto &sp : scratchpads) {
        const size_t size = sp.get_size();
        if (size > max_scratchpad_size) {
            max_scratchpad_size = size;
            max_scratchpad_md = sp;
        }
    }

    auto sub_src1_size = sub_mm1_src_md.get_size();
    auto sub_src2_size = sub_softmax_dst_md.get_size();
    sub_max_src1_src2_md
            = !sub_reorder0.is_alias() && sub_src1_size > sub_src2_size
            ? sub_mm1_src_md
            : sub_softmax_dst_md;

    auto sub_dst1_size = sub_mm1_dst_md.get_size();
    auto sub_wei2_size = sub_mm2_wei_md.get_size();
    sub_max_dst1_wei2_md
            = !sub_reorder2.is_alias() && sub_wei2_size > sub_dst1_size
            ? sub_mm2_wei_md
            : sub_mm1_dst_md;

    // Initialize memory object with empty buffer
    sub_max_src1_src2 = memory(sub_max_src1_src2_md, p_engine, nullptr);
    sub_max_dst1_wei2 = memory(sub_max_dst1_wei2_md, p_engine, nullptr);
    // reorder0
    sub_src1 = memory(sub_src1_md, p_engine, nullptr);
    // reorder1: 2d strided u8 -> 2d ba s8
    sub_wei1_user = memory(sub_wei1_user_md, p_engine, nullptr);
    // mm1
    sub_mm1_src = memory(sub_mm1_src_md, p_engine, nullptr);
    sub_mm1_wei = memory(sub_mm1_wei_md, p_engine, nullptr);
    sub_mm1_dst = memory(sub_mm1_dst_md, p_engine, nullptr);

    for (size_t i = 0; i < sub_mm1_post_md.size(); i++) {
        sub_mm1_post_mem.emplace_back(sub_mm1_post_md[i], p_engine, nullptr);
    }
    //select
    if (has_select && !select_fusiable) {
        sub_select_cond = memory(sub_select_cond_md, p_engine, nullptr);
        sub_select_src = memory(sub_select_src_md, p_engine, nullptr);
        sub_select_dst = memory(sub_mm1_dst_md, p_engine, nullptr);
    }
    // softmax
    sub_softmax_dst = memory(sub_softmax_dst_md, p_engine, nullptr);
    for (int i = 0; i < (int)sub_softmax_post_md.size(); i++) {
        sub_softmax_post_mem.emplace_back(sub_softmax_post_md[i], p_engine);
        auto alg = static_cast<algorithm>(
                softmax_ori_dnnl_pops.get()->entry_[i].binary.alg);
        if (alg == dnnl::algorithm::binary_mul) {
            float *ptr = reinterpret_cast<float *>(
                    sub_softmax_post_mem[i].get_data_handle());
            ptr[0] = get_attr_value<float, float>(
                    sdp_op[2], i + 1, op_attr::scales);
        }
        if (alg == dnnl::algorithm::binary_add) {
            int *ptr = reinterpret_cast<int *>(
                    sub_softmax_post_mem[i].get_data_handle());
            ptr[0] = get_attr_value<int64_t, int32_t>(
                    sdp_op[2], i + 1, op_attr::zps);
        }
    }
    // reorder2
    sub_wei2_user = memory(sub_wei2_user_md, p_engine, nullptr);
    // mm2
    sub_mm2_wei = memory(sub_mm2_wei_md, p_engine, nullptr);
    sub_mm2_dst = memory(sub_mm2_dst_md, p_engine, nullptr);
    // reorder3
    sub_dst_user = memory(sub_dst_user_md, p_engine, nullptr);
    // scratchpad, each thread will have a largest scratchpad.
    sub_scratchpad = memory(max_scratchpad_md, p_engine, nullptr);

    sub_reorder0_args = {{DNNL_ARG_SRC, sub_src1}, {DNNL_ARG_DST, sub_mm1_src},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder1_args = {{DNNL_ARG_SRC, sub_wei1_user},
            {DNNL_ARG_DST, sub_mm1_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm1_args = {{DNNL_ARG_SRC, sub_mm1_src},
            {DNNL_ARG_WEIGHTS, sub_mm1_wei}, {DNNL_ARG_DST, sub_mm1_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    int index = 0;
    for (int i = 0; i < mm1_ori_dnnl_pops.get()->len(); i++) {
        if (mm1_ori_dnnl_pops.get()->entry_[i].is_binary()) {
            if (static_cast<algorithm>(
                        mm1_ori_dnnl_pops.get()->entry_[i].binary.alg)
                    == algorithm::binary_select) {
                sub_mm1_args.insert(
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                                sub_mm1_post_mem[index++]});
                sub_mm1_args.insert(
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_2,
                                sub_mm1_post_mem[index++]});
            } else {
                sub_mm1_args.insert(
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                                sub_mm1_post_mem[index++]});
            }
        }
    }

    if (has_select && !select_fusiable) {
        sub_select_args = {{DNNL_ARG_SRC_0, sub_select_src},
                {DNNL_ARG_SRC_1, sub_mm1_dst},
                {DNNL_ARG_SRC_2, sub_select_cond},
                {DNNL_ARG_DST, sub_select_dst},
                {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};
    }

    sub_softmax_args
            = {{DNNL_ARG_SRC,
                       (has_select && !select_fusiable) ? sub_select_dst
                                                        : sub_mm1_dst},
                    {DNNL_ARG_DST, sub_softmax_dst},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    for (int i = 0; i < (int)sub_softmax_post_mem.size(); i++) {
        sub_softmax_args.insert(
                {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1,
                        sub_softmax_post_mem[i]});
    }

    sub_reorder2_args = {{DNNL_ARG_SRC, sub_wei2_user},
            {DNNL_ARG_DST, sub_mm2_wei}, {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_mm2_args = {{DNNL_ARG_SRC, sub_softmax_dst},
            {DNNL_ARG_WEIGHTS, sub_mm2_wei}, {DNNL_ARG_DST, sub_mm2_dst},
            {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    sub_reorder3_args
            = {{DNNL_ARG_SRC, sub_mm2_dst}, {DNNL_ARG_DST, sub_dst_user},
                    {DNNL_ARG_SCRATCHPAD, sub_scratchpad}};

    // add scales and zps for mm1, softmax, mm2
    prepare_sdp_scales_zps(sdp_op[0], 1, sub_reorder1_args, p_engine);
    prepare_sdp_scales_zps(sdp_op[1], 2, sub_mm1_args, p_engine);
    prepare_sdp_scales_zps(sdp_op[2], 1, sub_softmax_args, p_engine);
    prepare_sdp_scales_zps(sdp_op[3], 1, sub_reorder2_args, p_engine);
    prepare_sdp_scales_zps(sdp_op[4], 2, sub_mm2_args, p_engine);
    ////////////////////////////////////////////////////////////////////////
    /////////////// End Constructing exec args /////////////////////////////
    ////////////////////////////////////////////////////////////////////////

    // memory planing for buffer sharing
    memory_planning(sdp_registry);
    // TODO: remove this when primitive new API ready
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_num_threads(nthr);
#endif
    return status::success;
}

op_ptr sdp_decomp_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

impl::status_t sdp_decomp_config_t::record_input_offset(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    auto find_graph_inport = [&](std::shared_ptr<value_t> val) {
        // for quantized matmul, it has producer such as add_zp,sub_zp,mul_scale.
        if (val->get_consumers()[0].get_op().get_kind()
                        == graph::op_kind::MatMul
                || (val->has_producer()
                        && val->get_producer().get_kind()
                                == graph::op_kind::StaticReshape)) {
            while (val->has_producer()) {
                val = val->get_producer().get_input_value(0);
            }
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };
    //mul2 is for soft-capping
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr, add = nullptr,
           select = nullptr, mul2 = nullptr;
    const std::unordered_set<graph::op_kind_t> post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        VCHECK_SDP_DECOMP(op_kind != graph::op_kind::GenIndex,
                status::unimplemented, "Not support implicit causal mask");
        VCHECK_SDP_DECOMP(op_kind != graph::op_kind::DynamicDequantize,
                status::unimplemented,
                "Decomposed kernel does not support dynamic quantization");
        // both mm1 and mm2 are found.
        if (mm1 && mm2) break;
        if (op_kind != graph::op_kind::MatMul) continue;

        auto post_op = get_post_op(cur_op);
        if (post_op && post_op_kind.count(post_op->get_kind())) {
            // find mm1
            mm1 = cur_op;
            // TODO(xxx): Currently, p2 is not supported by decomp kernel.
            // p1: [matmul] --> [scale] --> [select] --> [mask] --> ...
            // p2: [matmul] --> [select] --> [scale] --> [mask] --> ...
            VCHECK_SDP_DECOMP(
                    mm1->get_attr<bool>(op_attr::transpose_a) == false,
                    status::unimplemented,
                    "Not support matmul 1 transpose_a is true");
            VCHECK_SDP_DECOMP(post_op->get_kind() != graph::op_kind::Select,
                    status::unimplemented,
                    "Not support select between matmul1 and scale");
            // find scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                has_scale = true;
                scale = post_op;
                post_op = get_post_op(post_op);
            }
            if (post_op && post_op->get_kind() == graph::op_kind::Tanh) {
                has_soft_capping = true;
                post_op = get_post_op(post_op);
                VCHECK_SDP_DECOMP(
                        post_op->get_kind() == graph::op_kind::Multiply,
                        status::unimplemented,
                        "Soft-capping must have tanh+multiply");
                mul2 = post_op;
                post_op = get_post_op(post_op);
            }

            if (post_op) {
                // find mask
                if (post_op->get_kind() == graph::op_kind::Add) {
                    add = std::move(post_op);
                    has_attention_mask = true;
                } else if (post_op->get_kind() == graph::op_kind::Select) {
                    // mm1 -> scale -> select -> ...
                    select = std::move(post_op);
                    has_select = true;
                }
            }
        } else {
            // find mm2
            mm2 = cur_op;
        }
    }
    VCHECK_SDP_DECOMP(mm1 != nullptr && mm2 != nullptr, status::invalid_graph,
            "Failed to find matmul1 or matmul2");
    int src1_id = find_graph_inport(mm1->get_input_value(0));
    graph_inport.emplace_back(src1_id);
    int wei1_id = find_graph_inport(mm1->get_input_value(1));
    graph_inport.emplace_back(wei1_id);
    int wei2_id = find_graph_inport(mm2->get_input_value(1));
    graph_inport.emplace_back(wei2_id);

    // for scale and add op. The input order is uncertain.
    if (has_scale) {
        int scale_id = find_graph_inport(scale->get_input_value(1));
        if (scale_id == -1)
            scale_id = find_graph_inport(scale->get_input_value(0));
        graph_inport.emplace_back(scale_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
    }
    if (has_soft_capping) {
        int mul2_id = find_graph_inport(mul2->get_input_value(1));
        if (mul2_id == -1)
            mul2_id = find_graph_inport(mul2->get_input_value(0));
        graph_inport.emplace_back(mul2_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
    }
    if (has_attention_mask) {
        int add_id = find_graph_inport(add->get_input_value(1));
        if (add_id == -1) add_id = find_graph_inport(add->get_input_value(0));
        graph_inport.emplace_back(add_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
    }

    if (has_select) {
        // Note: Currently, decompose kernel supports the following 2 pattern.
        // But only the select in pattern2 can be fused into preceding op where
        // select's input1 comes from the output of preceding op and it's input2
        // is an external graph input. Therefore, pattern2 usually have good
        // performance against pattern1.
        ///                             |                     |
        ///                         Preceding_op         Preceding_op
        ///                             |                     |
        ///    input0     input1     input2       input0    input1     input2
        ///  (condition) (graph in)  (inter)    (condition) (inter)  (graph in)
        ///          \      |        /                \      |        /
        ///            \    |      /                    \    |       /
        ///               Select                           Select
        ///                 |                                |
        ///               output                           output
        ///                 |                                 |
        ///             Succeeding_op                    Succeeding_op
        ///              (pattern1)                      (pattern2)
        int sel_cond_id = find_graph_inport(select->get_input_value(0));
        int sel_src0_id = find_graph_inport(select->get_input_value(1));
        int sel_src1_id = find_graph_inport(select->get_input_value(2));
        VCHECK_SDP_DECOMP(
                (sel_src0_id != -1 || sel_src1_id != -1) && sel_cond_id != -1,
                status::invalid_graph, "failed to find graph inport");
        if (sel_src1_id != -1) select_fusiable = true;
        graph_inport.emplace_back(sel_cond_id);
        if (sel_src0_id != -1)
            graph_inport.emplace_back(sel_src0_id);
        else
            graph_inport.emplace_back(sel_src1_id);
    } else {
        //placeholder
        graph_inport.emplace_back(-1);
        graph_inport.emplace_back(-1);
    }
    return status::success;
}

impl::status_t sdp_decomp_config_t::record_sdp_ops(
        std::shared_ptr<subgraph_t> &sg, bool is_quantize) {
    const auto get_wei_pre_op = [](const op_ptr &op) -> op_ptr {
        auto in_val = op->get_input_value(1);
        if (in_val->has_producer()) {
            auto *producer = &in_val->get_producer();
            if (producer->get_kind() == op_kind::_permute) {
                in_val = producer->get_input_value(0);
                if (in_val->has_producer())
                    producer = &in_val->get_producer();
                else
                    return nullptr;
            }
            if (producer == nullptr
                    || producer->get_kind() != op_kind::_reorder)
                return nullptr;
            return producer->shared_from_this();
        } else
            return nullptr;
    };

    for (const auto &cur_op : sg->get_ops()) {
        if (!cur_op || cur_op->get_kind() != op_kind::_matmul) continue;
        auto post_op = get_post_op(cur_op);
        op_ptr select;
        if (has_select && !select_fusiable) {
            if (!post_op || post_op->get_kind() != op_kind::_binary
                    || post_op->get_attr<int64_t>(op_attr::alg_kind)
                            != alg_kind::binary_select)
                continue;
            select = post_op;
            post_op = get_post_op(select);
        }

        if (!post_op || post_op->get_kind() != op_kind::_softmax) continue;
        auto ppost_op = get_post_op(post_op);
        VCHECK_SDP_DECOMP(ppost_op != nullptr, status::invalid_graph,
                "Failed to find post post op for matmul");

        op_ptr reorder1;
        op_ptr reorder2;
        if (is_quantize) {
            reorder1 = get_wei_pre_op(cur_op);
            reorder2 = get_wei_pre_op(ppost_op);
        }

        this->sdp_op = {reorder1, cur_op, post_op, reorder2, ppost_op, select};
        break;
    }
    return status::success;
}

void sdp_decomp_config_t::memory_planning(registry_t &sdp_registry) {
    // Registry is used to do the memory planning for sdp decomposition
    // algorithm. We reused some internal memory to reduce the memory
    // footprint for better cache hit. And here the key in registar of each
    // memory is planned in a specific order.
    registrar_t temporary_registrar = sdp_registry.registrar();

    // Here we initialize the map based on certain memory reuse logic. Those
    // memories(mds) who share the same buffer have the same registar key in
    // this map. So if we want to change the memory reuse logic, we need to
    // change the value of map here.
    mem_key_map = {{sub_max_src1_src2.get(), 0}, {sub_max_dst1_wei2.get(), 2},
            {sub_softmax_dst.get(), 0}, {sub_scratchpad.get(), 4}};
    if (!sub_reorder1.is_alias()) mem_key_map[sub_mm1_wei.get()] = 1;
    if (!sub_reorder3.is_alias()) mem_key_map[sub_mm2_dst.get()] = 3;

    temporary_registrar.book(mem_key_map[sub_max_src1_src2.get()],
            sub_max_src1_src2.get_desc().get_size());
    if (!sub_reorder1.is_alias())
        temporary_registrar.book(mem_key_map[sub_mm1_wei.get()],
                sub_mm1_wei.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_max_dst1_wei2.get()],
            sub_max_dst1_wei2.get_desc().get_size());
    if (!sub_reorder3.is_alias())
        temporary_registrar.book(mem_key_map[sub_mm2_dst.get()],
                sub_mm2_dst.get_desc().get_size());
    if (has_select && !select_fusiable)
        temporary_registrar.book(mem_key_map[sub_select_dst.get()],
                sub_select_dst.get_desc().get_size());
    temporary_registrar.book(mem_key_map[sub_scratchpad.get()],
            sub_scratchpad.get_desc().get_size());
}

impl::status_t sdp_decomp_config_t::prepare_sdp_scales_zps(
        std::shared_ptr<op_t> &op, int index,
        std::unordered_map<int, memory> &args, const dnnl::engine &p_engine) {
    const auto dt_scale = memory::data_type::f32,
               dt_zp = memory::data_type::s32;
    // scale zp order:
    // 1. src scale, wei scale
    // 2. src zp, wei zp
    // 3. dst scale, dst zp
    if (op && op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        if (fusion_info.with_runtime_scales(true, 0)) {
            memory::desc sub_src_scale_md
                    = memory::desc({1}, dt_scale, format_tag::x);
            memory sub_src_scale = memory(sub_src_scale_md, p_engine);
            float *src_scale_val_ptr = reinterpret_cast<float *>(
                    sub_src_scale.get_data_handle());
            src_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);

            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sub_src_scale});
        }
        if (fusion_info.with_runtime_scales(true, 1)) {
            memory::desc sub_wei_scale_md
                    = memory::desc({1}, dt_scale, format_tag::x);
            memory sub_wei_scale = memory(sub_wei_scale_md, p_engine);
            float *wei_scale_val_ptr = reinterpret_cast<float *>(
                    sub_wei_scale.get_data_handle());
            wei_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);
            args.insert(
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sub_wei_scale});
        }

        // src_zp and wei_zp
        if (fusion_info.with_runtime_zero_points(true, 0)) {
            memory::desc sub_src_zp_md
                    = memory::desc({1}, dt_zp, format_tag::x);
            memory sub_src_zp = memory(sub_src_zp_md, p_engine);
            int *src_zp_val_ptr
                    = reinterpret_cast<int *>(sub_src_zp.get_data_handle());
            src_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, sub_src_zp});
        }
        if (fusion_info.with_runtime_zero_points(true, 1)) {
            memory::desc sub_wei_zp_md
                    = memory::desc({1}, dt_zp, format_tag::x);
            memory sub_wei_zp = memory(sub_wei_zp_md, p_engine);
            int *wei_zp_val_ptr
                    = reinterpret_cast<int *>(sub_wei_zp.get_data_handle());
            wei_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert(
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, sub_wei_zp});
        }

        // dst scale, dst zp
        if (fusion_info.with_runtime_scales(false, 0)) {
            memory::desc sub_dst_scale_md
                    = memory::desc({1}, dt_scale, format_tag::x);
            memory sub_dst_scale = memory(sub_dst_scale_md, p_engine);
            float *dst_scale_val_ptr = reinterpret_cast<float *>(
                    sub_dst_scale.get_data_handle());
            dst_scale_val_ptr[0] = get_attr_value<float, float>(
                    op, index++, op_attr::scales);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sub_dst_scale});
        }
        if (fusion_info.with_runtime_zero_points(false, 0)) {
            memory::desc sub_dst_zp_md
                    = memory::desc({1}, dt_zp, format_tag::x);
            memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
            int *dst_zp_val_ptr
                    = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
            dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
        }
    }
    if (op && op->get_kind() == op_kind::_reorder) {
        if (op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            memory::desc sub_dst_zp_md
                    = memory::desc({1}, dt_zp, format_tag::x);
            memory sub_dst_zp = memory(sub_dst_zp_md, p_engine);
            int *dst_zp_val_ptr
                    = reinterpret_cast<int *>(sub_dst_zp.get_data_handle());
            dst_zp_val_ptr[0] = get_attr_value<int64_t, int32_t>(
                    op, index++, op_attr::zps);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, sub_dst_zp});
        }
    }
    return status::success;
}

dnnl::primitive_attr sdp_decomp_config_t::make_primitive_attr(
        std::shared_ptr<op_t> &op) {
    dnnl::primitive_attr attr;
    if (op && op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    if (op && op->get_kind() == op_kind::_reorder) {
        int mask = 0;
        if (op->has_attr(op_attr::mask)) {
            mask = static_cast<int>(op->get_attr<int64_t>(op_attr::mask));
        }

        if (op->has_attr(op_attr::with_runtime_dst_zps)
                && op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            // runtime dst zps
            attr.set_zero_points_mask(DNNL_ARG_TO, mask);
        } else if (op->has_attr(op_attr::dst_zps)) {
            assertm(false, "only support runtime dst zero points.\n");
        }
    }
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    return attr;
}

template status_t
sdp_decomp_config_t::construct_params<false, dnnl::memory::data_type::f32>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);
template status_t
sdp_decomp_config_t::construct_params<true, dnnl::memory::data_type::f32>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);
template status_t
sdp_decomp_config_t::construct_params<true, dnnl::memory::data_type::bf16>(
        std::shared_ptr<subgraph_t> &sg, registry_t &mqa_registry,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
