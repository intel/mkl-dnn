/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef COMMON_MATMUL_PD_HPP
#define COMMON_MATMUL_PD_HPP

#include <assert.h>
#include <map>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

#define VDISPATCH_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_MATMUL_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, matmul, f, msg, ##__VA_ARGS__);

#define VDISPATCH_MATMUL_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {

status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *reduce_desc, matmul_reduce_kind_t reduce_kind);

status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc);

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bias_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr,
        const memory_desc_t *reduce_md = nullptr,
        matmul_reduce_kind_t reduce_kind = matmul_reduce_kind::undef,
        data_type_t acc_dt = data_type::undef);

// NOLINTBEGIN(google-default-arguments)
struct matmul_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::matmul;

    using base_class = matmul_pd_t;
    using hint_class = matmul_pd_t;

    const matmul_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        const bool input = utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS);
        if (input) return arg_usage_t::input;

        if (arg == DNNL_ARG_BIAS)
            return with_bias() ? arg_usage_t::input : arg_usage_t::unused;

        if (arg == DNNL_ARG_REDUCE)
            return with_reduce() ? arg_usage_t::output : arg_usage_t::unused;
        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
        if (arg == DNNL_ARG_HINT_MAX_GROUP_SIZE)
            return memory_desc_wrapper(src_md()).is_grouped_desc()
                    ? arg_usage_t::input
                    : arg_usage_t::unused;
#endif

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_BIAS: return weights_md(1);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_REDUCE: return reduce_md(0);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *weights_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->weights_desc : &weights_md_;
        if (index == 1) return user_input ? &desc()->bias_desc : &bias_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    const memory_desc_t *reduce_md(
            int index = 0, bool user_input = false) const {
        if (index == 0) return user_input ? &desc()->reduce_desc : &reduce_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + with_bias() + n_binary_po_inputs() + n_prelu_po_inputs();
    }
    int n_outputs() const override { return 1 + with_reduce(); }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(src_md(0)).has_zero_dim()
                || memory_desc_wrapper(weights_md(0)).has_zero_dim()
                || memory_desc_wrapper(dst_md(0)).has_zero_dim();
    }

    int ndims() const { return dst_md_.ndims; }

    dim_t ldc() const {
        return memory_desc_wrapper(dst_md(0))
                .blocking_desc()
                .strides[ndims() - 2];
    }

    bool with_bias() const { return bias_md_.ndims != 0; }
    bool with_reduce() const { return reduce_md_.ndims != 0; }

    matmul_reduce_kind_t reduce_kind() const { return desc_.reduce_kind; }

    bool batched() const { return ndims() > 2; }

    dim_t batch() const {
        return utils::array_product(dst_md_.dims, ndims() - 2);
    }
    dim_t M() const { return dst_md_.dims[ndims() - 2]; }
    dim_t N() const { return dst_md_.dims[ndims() - 1]; }
    dim_t K() const { return src_md_.dims[src_md_.ndims - 1]; }

    bool is_bias_1xN() const {
        if (!with_bias()) return false;

        const auto &dims = weights_md(1)->dims;
        const int n_dims = ndims();
        for (int i = 0; i < n_dims - 1; ++i) {
            if (dims[i] != 1) return false;
        }

        return dims[n_dims - 1] == N();
    }

    // Quantization mask frequently used for scales and zero points
    int src_qmask_M() const {
        const int src_ndims = src_md(0)->ndims;
        assert(src_ndims >= 2);
        return 1 << (src_ndims - 2);
    }

    int src_qmask_K() const {
        const int src_ndims = src_md(0)->ndims;
        assert(src_ndims >= 2);
        return 1 << (src_ndims - 1);
    }

    int wei_qmask_N() const {
        const int wei_ndims = weights_md(0)->ndims;
        assert(wei_ndims >= 2);
        return 1 << (wei_ndims - 1);
    }

    int wei_qmask_K() const {
        const int wei_ndims = weights_md(0)->ndims;
        assert(wei_ndims >= 2);
        return 1 << (wei_ndims - 2);
    }

    int full_tensor_mask() const { return (1 << ndims()) - 1; }

    int dst_qmask_N() const { return wei_qmask_N(); }

    int dst_qmask_M() const { return src_qmask_M(); }

    static constexpr int any_mask = -1;

    virtual status_t attr_scales_ok(const engine_t *engine,
            const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
            const std::vector<int> &supported_qmodes
            = {quantization_mode::static_sazp},
            const std::map<int, std::vector<int>> &extra_masks = {}) const {
        const auto &scales = attr()->scales_;
        if (scales.has_default_values()) return status::success;

        const auto extra_mask_ok = [&](int arg, int mask) {
            const auto it = extra_masks.find(arg);
            if (it != extra_masks.end()) {
                for (const auto &extra_mask : it->second)
                    if (extra_mask == any_mask || mask == extra_mask)
                        return true;
            }
            return false;
        };

        VDISPATCH_MATMUL(scales.has_default_values(supported_args),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        for (int arg : supported_args) {
            if (scales.has_default_values(arg)) { continue; }

            // Fold-left to check if quantization mode is supported
            bool is_qmode_supported = false;
            for (auto &qmode : supported_qmodes) {
                is_qmode_supported = is_qmode_supported
                        || (scales.get(arg).get_quantization_mode() == qmode);
            }
            VDISPATCH_MATMUL(
                    is_qmode_supported, VERBOSE_UNSUPPORTED_SCALES_CFG);

            const auto &mask = scales.get_mask(arg);
            if (arg == DNNL_ARG_WEIGHTS) {
                // Mask over K dim is allowed for fp types or weights decompression only.
                if (types::is_integral_dt(weights_md(0)->data_type)) {
                    const bool is_decompression
                            = utils::one_of(weights_md(0)->data_type,
                                      data_type::s8, data_type::u8,
                                      data_type::s4, data_type::u4,
                                      data_type::u2, data_type::u3)
                            && IMPLICATION(
                                    !types::is_integral_dt(src_md()->data_type),
                                    attr()->fpmath_.apply_to_int_);
                    VDISPATCH_MATMUL(IMPLICATION((mask & wei_qmask_K()),
                                             is_decompression),
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            } else if (arg == DNNL_ARG_SRC) {
                // Masks supported across all implementations. Implementation
                // specific masks can be passed through `extra_masks`.
                const bool any_mask_ok = extra_mask_ok(arg, any_mask);
                VDISPATCH_MATMUL(any_mask_ok
                                || utils::one_of(mask, 0, src_qmask_K(),
                                        src_qmask_M() + src_qmask_K(),
                                        full_tensor_mask())
                                || extra_mask_ok(arg, mask),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        IMPLICATION(!any_mask_ok && (mask & src_qmask_K()),
                                !scales.get(arg).has_default_groups()),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        IMPLICATION(!scales.get(arg).has_default_groups(),
                                scales.get_group(arg, 0) == 1),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        IMPLICATION(!any_mask_ok && mask == src_qmask_M(),
                                scales.get(arg).has_default_groups()),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            } else if (arg == DNNL_ARG_DST) {
                // Masks supported across all implementations. Implementation
                // specific masks can be passed through `extra_masks`.
                VDISPATCH_MATMUL(utils::one_of(mask, 0, dst_qmask_N(),
                                         dst_qmask_M() + dst_qmask_N(),
                                         full_tensor_mask())
                                || extra_mask_ok(arg, mask),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            } else {
                assert(!"Unsupported arg");
            }
        }
        return status::success;
    }

    virtual status_t attr_zero_points_ok(const engine_t *engine,
            const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
            const std::vector<int> &supported_qmodes
            = {quantization_mode::static_sazp},
            const std::map<int, std::vector<int>> &extra_masks = {}) const {
        const auto &zp = attr()->zero_points_;
        if (zp.has_default_values()) return status::success;

        const auto extra_mask_ok = [&](int arg, int mask) {
            const auto it = extra_masks.find(arg);
            if (it != extra_masks.end()) {
                for (const auto &extra_mask : it->second)
                    if (extra_mask == any_mask || mask == extra_mask)
                        return true;
            }
            return false;
        };

        const auto broadcast_mask = [](const memory_desc_t &md) {
            int mask = 0;
            for (int d = 0; d < md.ndims; ++d)
                if (md.dims[d] == 1) mask |= (1 << d);
            return mask;
        };

        VDISPATCH_MATMUL(zp.has_default_values(supported_args),
                VERBOSE_UNSUPPORTED_ZP_CFG);
        for (int arg : supported_args) {
            if (zp.has_default_values(arg)) { continue; }

            // Fold-left to check if quantization mode is supported
            bool is_qmode_supported = false;
            for (auto &qmode : supported_qmodes) {
                is_qmode_supported = is_qmode_supported
                        || (zp.get(arg).get_quantization_mode() == qmode);
            }
            VDISPATCH_MATMUL(is_qmode_supported, VERBOSE_UNSUPPORTED_ZP_CFG);

            if (arg == DNNL_ARG_WEIGHTS) {
                if (!zp.get(arg).has_default_groups()) {
                    const auto gK = zp.get_group(arg, -2);
                    const auto gN = zp.get_group(arg, -1);

                    // Only one non-unit group is supported.
                    VDISPATCH_MATMUL(utils::one_of(1, gK, gN),
                            VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            } else if (arg == DNNL_ARG_SRC) {
                const int bcast_mask = broadcast_mask(*src_md());
                const auto &mask = zp.get_mask(arg) & ~bcast_mask;

                VDISPATCH_MATMUL(
                        utils::one_of(mask, 0, src_qmask_K() & ~bcast_mask,
                                (src_qmask_M() + src_qmask_K()) & ~bcast_mask)
                                || extra_mask_ok(arg, mask),
                        VERBOSE_UNSUPPORTED_ZP_CFG);

                if (!zp.get(DNNL_ARG_SRC).has_default_groups()) {
                    const auto gM = zp.get_group(DNNL_ARG_SRC, -2);
                    VDISPATCH_MATMUL(gM == 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            } else if (arg == DNNL_ARG_DST) {
                const int bcast_mask = broadcast_mask(*dst_md());
                const auto &mask = zp.get_mask(arg) & ~bcast_mask;

                VDISPATCH_MATMUL(
                        utils::one_of(mask, 0, dst_qmask_N() & ~bcast_mask)
                                || extra_mask_ok(arg, mask),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_MATMUL(zp.get(arg).has_default_groups(),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
            } else {
                assert(!"Unsupported arg");
            }
        }

        return status::success;
    }

protected:
    matmul_desc_t desc_;

    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;
    memory_desc_t reduce_md_;

    matmul_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const matmul_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<matmul_desc_t>(adesc))
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc)
        , reduce_md_(desc_.reduce_desc) {}

    // Sets format to `abx` for all `format_kind::any` tensors.
    bool set_default_formats() {
        for (auto md :
                {&src_md_, &weights_md_, &bias_md_, &dst_md_, &reduce_md_}) {
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) {
                if (mdw.has_runtime_dims_or_strides()) return false;
                status_t status = status::success;
                // Any data types that don't fit a single byte require special
                // strides to form a valid operation.
                // `memory_desc_init_by_strides` with `strides=nullptr` will
                // set straight `abx` leading to odd stride for subbyte types
                // which is illegal in reorder API.
                const auto last_dim = mdw.dims()[mdw.ndims() - 1];
                if (mdw.sub_byte_data_type_multiplier() > 1
                        && last_dim % mdw.sub_byte_data_type_multiplier()
                                != 0) {
                    dims_t rounded_dims;
                    utils::array_copy(rounded_dims, mdw.dims(), mdw.ndims());
                    rounded_dims[mdw.ndims() - 1] = utils::rnd_up(
                            last_dim, mdw.sub_byte_data_type_multiplier());

                    dims_t rounded_strides;
                    rounded_strides[mdw.ndims() - 1] = 1;
                    for (dim_t d = mdw.ndims() - 2; d >= 0; d--)
                        rounded_strides[d]
                                = rounded_strides[d + 1] * rounded_dims[d + 1];
                    status = memory_desc_init_by_strides(*md, rounded_strides);
                } else {
                    status = memory_desc_init_by_strides(*md, nullptr);
                }
                if (status != status::success) return false;
            }
        }

        return true;
    }

    // All implementations that do not support sparse inputs/outputs should
    // call this function.
    bool is_dense_format_kind() {
        return impl::is_dense_format_kind(
                {&src_md_, &weights_md_, &bias_md_, &dst_md_, &reduce_md_});
    }
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif
