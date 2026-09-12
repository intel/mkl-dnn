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

#ifndef COMMON_SDPA_PD_HPP
#define COMMON_SDPA_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive_desc.hpp"
#include "common/sdpa_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VDISPATCH_SDPA(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, sdpa, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_SDPA_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, sdpa, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

// NOLINTBEGIN(google-default-arguments)

struct sdpa_fwd_pd_t;

struct sdpa_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::sdpa;

    static constexpr int mask_q_index = 2;
    static constexpr int mask_k_index = 3;
    static constexpr int ndims = 4;

    const sdpa_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool with_attn_scale() const {
        return (desc()->scale_md()->data_type != data_type::undef);
    }

    bool with_host_scale() const {
        return (desc()->scale_md()->format_kind == format_kind::host_scalar);
    }

    bool with_attn_mask() const {
        return (desc()->attn_mask_md()->data_type != data_type::undef);
    }

    /// Returns the accumulation data type of the KQ matmul
    data_type_t kq_acc_dt() const { return desc()->kq_acc_dt; }

    /// Returns the accumulation data type of the VS matmul
    data_type_t vs_acc_dt() const { return desc()->vs_acc_dt; }

    /// If true, the attention mask is a causal mask
    bool with_causal_mask() const {
        return desc_.mask_type == attn_mask_type::top_left
                || desc_.mask_type == attn_mask_type::bottom_right;
    }

    /// If true, dequantize the Q tensor using scaling in the KQ matmul
    bool with_query_scales() const {
        return (!desc()->q_scales.has_default_values());
    }

    /// If true, dequantize the K tensor using scaling in the KQ matmul
    bool with_key_scales() const {
        return (!desc()->kq_scales.has_default_values());
    }

    /// If true, dequantize the V tensor using scaling in the VS matmul
    bool with_value_scales() const {
        return (!desc()->vs_scales.has_default_values());
    }

    /// If true, dequantize the K tensor with zero points in the KQ matmul
    bool with_key_zp() const {
        return (!desc()->kq_zero_points.has_default_values());
    }

    /// If true, dequantize the V tensor with zero points in the VS matmul
    bool with_value_zp() const {
        return (!desc()->vs_zero_points.has_default_values());
    }

    /// Returns the data type of the scales tensor for the Q tensor
    data_type_t query_scales_dt() const {
        return desc()->q_scales.get_data_type();
    }

    /// Returns the data type of the scales tensor for the KQ matmul
    data_type_t key_scales_dt() const {
        return desc()->kq_scales.get_data_type();
    }

    /// Returns the data type of the zero points tensor for the KQ matmul
    data_type_t key_zp_dt() const {
        return desc()->kq_zero_points.get_data_type();
    }

    /// Returns the data type of the scales tensor for the VS matmul
    data_type_t value_scales_dt() const {
        return desc()->vs_scales.get_data_type();
    }

    /// Returns the data type of the zero points tensor for the VS matmul
    data_type_t value_zp_dt() const {
        return desc()->vs_zero_points.get_data_type();
    }

    // Number of Q elements covered by one Q scale
    int query_group_size() const {
        if (!with_query_scales()) return 0;
        return group_size(desc()->q_scales, *desc()->qry_md());
    }

    // Returns the group size for the quantization parameters for the KQ matmul
    int key_group_size() const {
        int out = 0;
        if (with_key_scales())
            out = group_size(desc()->kq_scales, *desc()->key_md());
        else if (with_key_zp()) {
            out = group_size(desc()->kq_zero_points, *desc()->key_md());
        }
        return out;
    }

    // Returns the group size for the quantization parameters for the VS matmul
    int value_group_size() const {
        int out = 0;
        if (with_value_scales())
            out = group_size(desc()->vs_scales, *desc()->val_md());
        else if (with_value_zp()) {
            out = group_size(desc()->vs_zero_points, *desc()->val_md());
        }
        return out;
    }

protected:
    sdpa_desc_t desc_;
    const sdpa_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t ws_md_;

    sdpa_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const sdpa_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<sdpa_desc_t>(adesc))
        , hint_fwd_pd_(hint_fwd_pd) {}

    status_t init_default_ws() {
        dims_t d;
        d[0] = desc()->batch() * desc()->num_q_heads()
                * desc()->queries(); // (logsumexp) per query

        return memory_desc_init_by_tag(
                ws_md_, 1, d, data_type::f32, format_tag::a);
    }

    bool set_default_format(memory_desc_t *md) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) return false;

        return true;
    }

private:
    static int group_size(
            const quant_entry_t &scales, const memory_desc_t &desc) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        const auto mask = scales.get_mask();
        if (scales.has_default_groups()) {
            for (int idx : mask_iterator(mask)) {
                out /= desc.dims[idx];
            }
        } else {
            for (int idx : mask_iterator(mask)) {
                if (idx < 2) {
                    out /= desc.dims[idx];
                } else {
                    out /= (desc.dims[idx] / scales.get_group(idx - 2));
                }
            }
        }
        return static_cast<int>(out);
    }
};

struct sdpa_fwd_pd_t : public sdpa_pd_t {
    using base_class = sdpa_fwd_pd_t;
    using hint_class = sdpa_fwd_pd_t;

    arg_usage_t arg_usage(int arg) const override {
        // TODO: this is broken for cases when the user passes quantization
        // memories unconditionally but the primitive desc is not set up for
        // quantization.
        if (utils::one_of(arg, DNNL_ARG_QUERIES, DNNL_ARG_KEYS, DNNL_ARG_VALUES,
                    DNNL_ARG_ATTN_MASK, DNNL_ARG_SCALE,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_QUERIES,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE)
            return !types::is_zero_md(workspace_md()) ? arg_usage_t::output
                                                      : arg_usage_t::unused;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_QUERIES: return src_md(0);
            case DNNL_ARG_KEYS: return src_md(1);
            case DNNL_ARG_VALUES: return src_md(2);
            case DNNL_ARG_ATTN_MASK: return src_md(3);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.q_desc;
            case 1: return &desc_.k_desc;
            case 2: return &desc_.v_desc;
            case 3: return &desc_.attn_mask_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.dst_desc : &glob_zero_md;
    }
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
    }

    int n_inputs() const override {
        return 3 + int(with_attn_mask()) + int(with_attn_scale());
    }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }

protected:
    sdpa_fwd_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : sdpa_pd_t(adesc, attr, hint_fwd_pd) {}

    bool set_default_formats() {
        bool ok = true;

        for (auto md : {&desc_.q_desc, &desc_.k_desc, &desc_.v_desc,
                     &desc_.dst_desc}) {
            ok = ok && set_default_format(md);
        }

        auto status = attr_.post_ops_.set_default_formats(&desc_.dst_desc);
        ok = ok && (status == status::success);

        return ok;
    }
};

struct sdpa_bwd_pd_t : public sdpa_pd_t {
    using base_class = sdpa_bwd_pd_t;
    using hint_class = sdpa_fwd_pd_t;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_QUERIES, DNNL_ARG_KEYS, DNNL_ARG_VALUES,
                    DNNL_ARG_DST, DNNL_ARG_DIFF_DST, DNNL_ARG_ATTN_MASK,
                    DNNL_ARG_SCALE))
            return arg_usage_t::input;

        if (utils::one_of(arg, DNNL_ARG_DIFF_QUERIES, DNNL_ARG_DIFF_KEYS,
                    DNNL_ARG_DIFF_VALUES))
            return arg_usage_t::output;

        if (arg == DNNL_ARG_DS)
            return with_dS() ? arg_usage_t::output : arg_usage_t::unused;

        if (arg == DNNL_ARG_WORKSPACE) return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_QUERIES: return src_md(0);
            case DNNL_ARG_KEYS: return src_md(1);
            case DNNL_ARG_VALUES: return src_md(2);
            case DNNL_ARG_ATTN_MASK: return src_md(3);
            case DNNL_ARG_DST: return src_md(4);
            case DNNL_ARG_DIFF_DST: return src_md(5);
            case DNNL_ARG_DIFF_QUERIES: return dst_md(0, user_input);
            case DNNL_ARG_DIFF_KEYS: return dst_md(1, user_input);
            case DNNL_ARG_DIFF_VALUES: return dst_md(2, user_input);
            case DNNL_ARG_DS: return dst_md(3, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.q_desc;
            case 1: return &desc_.k_desc;
            case 2: return &desc_.v_desc;
            case 3: return &desc_.attn_mask_desc;
            case 4: return &desc_.dst_desc;
            case 5: return &desc_.diff_dst_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.diff_q_desc;
            case 1: return &desc_.diff_k_desc;
            case 2: return &desc_.diff_v_desc;
            case 3: return &desc_.dS_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *workspace_md(int index = 0) const override {
        return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_
                                                         : &glob_zero_md;
    }

    bool with_dS() const {
        return (desc_.dS_desc.data_type != data_type::undef);
    }

    int n_inputs() const override {
        // Q, K, V, O, dO
        return 5 + int(with_attn_mask()) + int(with_attn_scale())
                + int(!types::is_zero_md(workspace_md()));
    }
    int n_outputs() const override { return 3 + int(with_dS()); }

    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.diff_dst_desc : &glob_zero_md;
    }

protected:
    sdpa_bwd_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : sdpa_pd_t(adesc, attr, hint_fwd_pd) {}

    bool set_default_formats() {
        bool ok = true;

        for (auto md : {&desc_.q_desc, &desc_.k_desc, &desc_.v_desc,
                     &desc_.dst_desc, &desc_.diff_dst_desc, &desc_.diff_q_desc,
                     &desc_.diff_k_desc, &desc_.diff_v_desc}) {
            ok = ok && set_default_format(md);
        }

        auto status = attr_.post_ops_.set_default_formats(&desc_.dst_desc);
        ok = ok && (status == status::success);

        return ok;
    }
};

// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif
