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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info) {
    dnnl::primitive_attr attr;
    std::vector<int64_t> default_groups;

    if (fusion_info.dst_scales_) {
        const op_t *dst_scales_op = fusion_info.dst_scales_->get_op();
        VCHECK_FUSION_INFO(fusion_info.with_runtime_scales(false, 0), attr,
                "failed to set scales for %s since primitive only supports "
                "runtime dst scales",
                op->get_name().c_str());
        int mask = 0;
        dnnl::memory::data_type dt = dnnl::memory::data_type::undef;

        mask = get_quant_mask(dst_scales_op);
        dt = dst_scales_op->has_attr(op_attr::data_type)
                ? static_cast<dnnl::memory::data_type>(
                          dst_scales_op->get_attr<int64_t>(op_attr::data_type))
                : dnnl::memory::data_type::f32;
        attr.set_scales(DNNL_ARG_DST, mask, default_groups, dt);
    }

    // convert input scales
    if (!fusion_info.input_scales_.empty()) {
        for (const auto &in_scales : fusion_info.input_scales_) {
            size_t in_scales_indices = in_scales.first;
            const op_t *in_scales_op = in_scales.second->get_op();
            VCHECK_FUSION_INFO(
                    fusion_info.with_runtime_scales(true, in_scales_indices),
                    attr,
                    "failed to set scales for %s since primitive only supports "
                    "runtime src scales",
                    op->get_name().c_str());
            int mask = 0;
            if (in_scales_op->has_attr(op_attr::mask)) {
                const auto scales_data_type
                        = in_scales_op->has_attr(op_attr::data_type)
                        ? in_scales_op->get_attr<int64_t>(op_attr::data_type)
                        : dnnl_f32;
                const bool is_per_group = utils::has_group_shape(in_scales_op);
                if (is_per_group) {
                    // oneDNN only supports weights-decompressed matmul
                    if (in_scales_indices != 1
                            || op->get_kind() != op_kind::_matmul)
                        continue;
                    const auto &group_shape
                            = in_scales_op->get_attr<std::vector<int64_t>>(
                                    op_attr::group_shape);

                    // Currently oneDNN only supports grouped scales and zps on
                    // last two dimensions.
                    std::vector<int64_t> groups(
                            group_shape.end() - 2, group_shape.end());
                    mask = get_quant_mask(in_scales_op);
                    attr.set_scales(DNNL_ARG_WEIGHTS, mask, groups,
                            static_cast<dnnl::memory::data_type>(
                                    scales_data_type));
                } else {
                    mask = get_quant_mask(in_scales_op);
                    // Conv weights: remap per-OC mask to the expected
                    // primitive mask (1 without groups, 3 with groups).
                    if (impl::utils::one_of(op->get_kind(),
                                op_kind::_convolution, op_kind::_convtranspose)
                            && in_scales_indices == 1) {
                        bool with_groups = false;
                        if (op->get_input_value(1)->has_producer()
                                && op->get_input_op(1)->get_kind()
                                        == op_kind::_to_group) {
                            const auto &to_group = op->get_input_op(1);
                            if (to_group->get_attr<int64_t>(op_attr::groups)
                                    > 1) {
                                with_groups = true;
                            }
                        }
                        const int expected_oc_mask = with_groups ? 3 : 1;
                        if (mask != 0 && mask != expected_oc_mask)
                            mask = expected_oc_mask;
                    }
                    attr.set_scales(in_scales_indices == 0 ? DNNL_ARG_SRC
                                                           : DNNL_ARG_WEIGHTS,
                            mask, default_groups,
                            static_cast<dnnl::memory::data_type>(
                                    scales_data_type));
                }
            }
        }
    }

    // convert input zps
    if (!fusion_info.input_zps_.empty()) {
        for (const auto &in_zps : fusion_info.input_zps_) {
            size_t in_zps_indices = in_zps.first;
            const op_t *in_zps_op = in_zps.second->get_op();
            VCHECK_FUSION_INFO(
                    fusion_info.with_runtime_zero_points(true, in_zps_indices),
                    attr,
                    "failed to set zero points for %s since primitive only "
                    "supports runtime src zero points",
                    op->get_name().c_str());

            if (in_zps_op->has_attr(op_attr::mask)) {
                const auto zps_data_type
                        = in_zps_op->has_attr(op_attr::data_type)
                        ? in_zps_op->get_attr<int64_t>(op_attr::data_type)
                        : dnnl_s32;
                const bool is_per_group = utils::has_group_shape(in_zps_op);
                if (is_per_group) {
                    // oneDNN only supports weights-decompressed matmul
                    if (in_zps_indices != 1
                            || op->get_kind() != op_kind::_matmul)
                        break;
                    const auto &group_shape
                            = in_zps_op->get_attr<std::vector<int64_t>>(
                                    op_attr::group_shape);

                    // Currently oneDNN only supports grouped scales and zps on
                    // last two dimensions.
                    std::vector<int64_t> groups(
                            group_shape.end() - 2, group_shape.end());
                    int mask = get_quant_mask(in_zps_op);

                    attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, groups,
                            static_cast<dnnl::memory::data_type>(
                                    zps_data_type));

                } else {
                    int mask = get_quant_mask(in_zps_op);
                    attr.set_zero_points(in_zps_indices == 0 ? DNNL_ARG_SRC
                                                             : DNNL_ARG_WEIGHTS,
                            mask, default_groups,
                            static_cast<dnnl::memory::data_type>(
                                    zps_data_type));
                }
            }
        }
    }

    // convert output zps
    if (fusion_info.output_zps_) {
        const op_t *output_zps_op = fusion_info.output_zps_->get_op();
        const auto zps_data_type = output_zps_op->has_attr(op_attr::data_type)
                ? output_zps_op->get_attr<int64_t>(op_attr::data_type)
                : dnnl_s32;
        VCHECK_FUSION_INFO(fusion_info.with_runtime_zero_points(false, 0), attr,
                "failed to set zero points for %s since primitive only "
                "supports runtime dst zero points",
                op->get_name().c_str());
        int mask = 0;
        attr.set_zero_points(DNNL_ARG_DST, mask, default_groups,
                static_cast<dnnl::memory::data_type>(zps_data_type));
    }

    if (fusion_info.dropout_) {
        // TODO: output mask is not enabled yet.
        const auto prop_type = ltw(
                fusion_info.dropout_->get_op()->get_input_logical_tensor(1))
                                       .property_type();
        memory::desc mask_desc;
        attr.set_dropout(mask_desc, /*seed_dt*/ memory::data_type::s64,
                /*use_offset*/ true,
                /*use_host_scalars*/ prop_type == property_type::host_scalar);
    }

    // convert post ops
    dnnl::post_ops dnnl_pops;
    for (auto &pop : fusion_info.get_post_ops()) {
        const op_t *fused_op = pop->get_op();
        const auto fused_op_kind = fused_op->get_kind();
        if (fused_op_kind == op_kind::_eltwise) {
            float alpha = 0.f;
            float beta = 0.f;
            if (fused_op->has_attr(op_attr::alpha)) {
                alpha = fused_op->get_attr<float>(op_attr::alpha);
            }
            if (fused_op->has_attr(op_attr::beta)) {
                beta = fused_op->get_attr<float>(op_attr::beta);
            }
            const auto alg = static_cast<dnnl::algorithm>(
                    fused_op->get_attr<int64_t>(op_attr::alg_kind));
            dnnl_pops.append_eltwise(alg, alpha, beta);
        } else if (fused_op_kind == op_kind::_binary) {
            const auto alg = static_cast<dnnl::algorithm>(
                    fused_op->get_attr<int64_t>(op_attr::alg_kind));
            const auto &extra_inputs = pop->get_unfused_input_indices();
            float scale = pop->get_scale();
            int32_t zp = pop->get_zp();
            const auto psrc_val = op->get_input_value(extra_inputs[0]);
            const auto psrc = psrc_val->get_logical_tensor();
            const auto dst = op->get_output_logical_tensor(0);
            // check if can use post-sum, otherwise use binary post ops
            // algorithm should be binary_add
            bool is_post_sum = alg == dnnl::algorithm::binary_add;
            // base_op should not be eltwise, pool, or softmax.
            is_post_sum = is_post_sum
                    && !impl::utils::one_of(op->get_kind(), op_kind::_eltwise,
                            op_kind::_pool, op_kind::_softmax,
                            op_kind::_logsoftmax);
            // only support one post-sum
            is_post_sum = is_post_sum
                    && !(op->has_attr(op_attr::with_sum)
                            && op->get_attr<bool>(op_attr::with_sum));
            // post src and dst should have the same shape
            is_post_sum = is_post_sum
                    && logical_tensor_wrapper_t(dst).vdims()
                            == logical_tensor_wrapper_t(psrc).vdims();
            // sum post-op cannot be a part of post-op chain in dw post-op case
            is_post_sum = is_post_sum && !fusion_info.has_post_dw_conv();
            // dst should have equal or larger memory size than post src
            is_post_sum = is_post_sum
                    && (psrc.data_type == dst.data_type
                            || impl::utils::one_of(psrc.data_type,
                                    impl::data_type::u8, impl::data_type::s8));
            // post src should not have alias. Here the condition of alias_ins == 1
            // is to disable the inplace option for src = main_op(src) + src
            const auto get_external_id
                    = [](const std::shared_ptr<value_t> &val) {
                auto tmp_val = val;
                while (tmp_val->has_producer()) {
                    size_t lt_id = tmp_val->get_logical_tensor().id;
                    // check if lt_id is already a external id
                    if (lt_id != std::numeric_limits<size_t>::max())
                        return lt_id;

                    const op_t &prod_op = tmp_val->get_producer();
                    // ops like Dnnl_constant_scales doesn't have external input
                    // return a internal id
                    if (prod_op.num_inputs() == 0) return lt_id;
                    tmp_val = prod_op.get_input_value(0);
                }
                return tmp_val->get_logical_tensor().id;
            };
            size_t alias_ins = 0;
            size_t psrc_lt_id = get_external_id(psrc_val);
            for (size_t op_in_idx = 0; op_in_idx < op->num_inputs();
                    ++op_in_idx) {
                size_t op_in_lt_id
                        = get_external_id(op->get_input_value(op_in_idx));
                if (op_in_lt_id == psrc_lt_id) alias_ins++;
            }
            is_post_sum = is_post_sum && alias_ins == 1;
            if (is_post_sum) {
                pop->set_post_sum();
                op->set_attr<bool>(op_attr::with_sum, true);
                dnnl::memory::data_type sum_dt = dnnl::memory::data_type::undef;
                if (psrc.data_type == impl::data_type::s8
                        && dst.data_type == impl::data_type::u8) {
                    sum_dt = dnnl::memory::data_type::s8;
                }
                dnnl_pops.append_sum(scale, zp, sum_dt);
            } else if (alg == dnnl::algorithm::binary_select) {
                // post-binary
                VCHECK_FUSION_INFO(
                        extra_inputs.size() == 2 && scale == 1.f && zp == 0,
                        attr,
                        "%s post-binary_select only has 2 extra inputs and "
                        "doesn't support input scale and zp",
                        op->get_name().c_str());
                auto md1 = make_dnnl_memory_desc(psrc);
                auto psrc_cond = op->get_input_value(extra_inputs[1])
                                         ->get_logical_tensor();
                auto md2 = make_dnnl_memory_desc(psrc_cond);
                dnnl_pops.append_binary(alg, md1, md2);
            } else {
                // post-binary
                VCHECK_FUSION_INFO(
                        extra_inputs.size() == 1 && scale == 1.f && zp == 0,
                        attr,
                        "%s post-binary only has 1 extra input and doesn't "
                        "support "
                        "input scale and zp",
                        op->get_name().c_str());
                auto md = make_dnnl_memory_desc(psrc);
                if (impl::utils::one_of(op->get_kind(), op_kind::_convolution,
                            op_kind::_convtranspose))
                    md = to_format_any(md);
                dnnl_pops.append_binary(alg, md);
            }
        } else if (fused_op_kind == op_kind::_convolution) {
            const auto &extra_input_indices = pop->get_unfused_input_indices();

            auto get_dnn_dt = [](const std::shared_ptr<value_t> &val) {
                using ltw = logical_tensor_wrapper_t;
                auto graph_dt = ltw(val->get_logical_tensor()).data_type();
                return static_cast<dnnl::memory::data_type>(graph_dt);
            };

            const size_t wei_idx = extra_input_indices[0];
            auto wei_value = op->get_input_value(wei_idx);
            const auto wei_dt = get_dnn_dt(wei_value);
            const auto dst_dt = get_dnn_dt(op->get_output_value(0));
            auto bia_dt = dnnl::memory::data_type::undef;
            const int64_t ks = wei_value->get_logical_tensor().dims[3];
            const int64_t stride = fused_op->get_attr<std::vector<int64_t>>(
                    op_attr::strides)[0];
            const int64_t pad_l = fused_op->get_attr<std::vector<int64_t>>(
                    op_attr::pads_begin)[0];
            if (extra_input_indices.size() > 1) {
                const size_t bias_idx = extra_input_indices[1];
                auto bias_value = op->get_input_value(bias_idx);
                bia_dt = get_dnn_dt(bias_value);
                dnnl_pops.append_dw(wei_dt, bia_dt, dst_dt, ks, stride, pad_l);
            } else {
                dnnl_pops.append_dw(wei_dt, bia_dt, dst_dt, ks, stride, pad_l);
            }
        } else {
            // not reachable
        }
    }
    attr.set_post_ops(dnnl_pops);

    return attr;
}

dnnl::primitive_attr make_dnnl_sdpa_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info,
        const attr_type_t attr_type) {
    dnnl::primitive_attr attr;
    std::vector<int64_t> default_groups;

    const static std::unordered_map<size_t, size_t> arg_map = {
            {DNNL_ARG_QUERIES, DNNL_ARG_SRC},
            {DNNL_ARG_KEYS, DNNL_ARG_WEIGHTS},
            {DNNL_ARG_VALUES, DNNL_ARG_WEIGHTS},
            {DNNL_ARG_PROBABILITIES, DNNL_ARG_SRC},
    };

    // convert input scales
    if (!fusion_info.input_scales_.empty()) {

        for (const auto &in_scales : fusion_info.input_scales_) {
            size_t in_scales_indices = in_scales.first;
            if (attr_type == attr_type_t::QK) {
                if (in_scales_indices != DNNL_ARG_QUERIES
                        && in_scales_indices != DNNL_ARG_KEYS) {
                    continue;
                }
            } else if (attr_type == attr_type_t::VS) {
                if (in_scales_indices != DNNL_ARG_VALUES
                        && in_scales_indices != DNNL_ARG_PROBABILITIES) {
                    continue;
                }
            }
            const op_t *in_scales_op = in_scales.second->get_op();
            VCHECK_FUSION_INFO(
                    fusion_info.with_runtime_scales(true, in_scales_indices),
                    attr,
                    "failed to set scales for %s since primitive only supports "
                    "runtime src scales",
                    op->get_name().c_str());
            int mask = 0;
            // fuse_sdpa only fuses host-scalar P scales
            const bool is_host_scalar
                    = in_scales_indices == DNNL_ARG_PROBABILITIES;
            if (in_scales_op->has_attr(op_attr::mask)) {
                const auto scales_data_type
                        = in_scales_op->has_attr(op_attr::data_type)
                        ? in_scales_op->get_attr<int64_t>(op_attr::data_type)
                        : dnnl_f32;
                const bool is_per_group = utils::has_group_shape(in_scales_op);
                if (is_per_group) {
                    // oneDNN only supports weights-decompressed matmul or sdpa
                    if (arg_map.at(in_scales_indices) != DNNL_ARG_WEIGHTS)
                        continue;
                    const auto &group_shape
                            = in_scales_op->get_attr<std::vector<int64_t>>(
                                    op_attr::group_shape);

                    // Currently oneDNN only supports grouped scales and zps on
                    // last two dimensions.
                    std::vector<int64_t> groups(
                            group_shape.end() - 2, group_shape.end());
                    mask = get_quant_mask(in_scales_op);
                    attr.set_scales(DNNL_ARG_WEIGHTS, mask, groups,
                            static_cast<dnnl::memory::data_type>(
                                    scales_data_type),
                            is_host_scalar);
                } else {
                    mask = get_quant_mask(in_scales_op);
                    attr.set_scales(
                            static_cast<int>(arg_map.at(in_scales_indices)),
                            mask, default_groups,
                            static_cast<dnnl::memory::data_type>(
                                    scales_data_type),
                            is_host_scalar);
                }
            }
        }
    }

    // The softmax dequantization scale multiplies the VS matmul result
    if (attr_type == attr_type_t::VS
            && fusion_info.with_runtime_scales(false, 0)) {
        attr.set_scales(DNNL_ARG_DST, 0, default_groups,
                dnnl::memory::data_type::f32, /* is_on_host = */ true);
    }

    // convert input zps
    if (!fusion_info.input_zps_.empty()) {
        for (const auto &in_zps : fusion_info.input_zps_) {
            size_t in_zps_indices = in_zps.first;
            if (attr_type == attr_type_t::QK) {
                if (in_zps_indices != DNNL_ARG_QUERIES
                        && in_zps_indices != DNNL_ARG_KEYS) {
                    continue;
                }
            } else if (attr_type == attr_type_t::VS) {
                if (in_zps_indices != DNNL_ARG_VALUES) { continue; }
            }
            const op_t *in_zps_op = in_zps.second->get_op();
            VCHECK_FUSION_INFO(
                    fusion_info.with_runtime_zero_points(true, in_zps_indices),
                    attr,
                    "failed to set zero points for %s since primitive only "
                    "supports runtime src zero points",
                    op->get_name().c_str());

            if (in_zps_op->has_attr(op_attr::mask)) {
                const auto zps_data_type
                        = in_zps_op->has_attr(op_attr::data_type)
                        ? in_zps_op->get_attr<int64_t>(op_attr::data_type)
                        : dnnl_s32;
                const bool is_per_group = utils::has_group_shape(in_zps_op);
                if (is_per_group) {
                    // oneDNN only supports weights-decompressed matmul
                    if (arg_map.at(in_zps_indices) != DNNL_ARG_WEIGHTS) break;
                    const auto &group_shape
                            = in_zps_op->get_attr<std::vector<int64_t>>(
                                    op_attr::group_shape);

                    // Currently oneDNN only supports grouped scales and zps on
                    // last two dimensions.
                    std::vector<int64_t> groups(
                            group_shape.end() - 2, group_shape.end());
                    int mask = get_quant_mask(in_zps_op);

                    attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, groups,
                            static_cast<dnnl::memory::data_type>(
                                    zps_data_type));

                } else {
                    int mask = get_quant_mask(in_zps_op);
                    attr.set_zero_points(
                            static_cast<int>(arg_map.at(in_zps_indices)), mask,
                            default_groups,
                            static_cast<dnnl::memory::data_type>(
                                    zps_data_type));
                }
            }
        }
    }
    return attr;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
