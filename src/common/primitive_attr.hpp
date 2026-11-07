/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ATTR_HPP
#define COMMON_PRIMITIVE_ATTR_HPP

#include <map>
#include <initializer_list>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "primitive_attr_quant.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

// Use `...` for `msg` and additional variables used in msg
#define VCHECK_ATTR(cond, ...) \
    VCONDCHECK(primitive, create, check, primitive, (cond), \
            status::invalid_arguments, __VA_ARGS__)

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr();

struct rnn_data_qparams_t {
    rnn_data_qparams_t() : scale_(1.f), shift_(0.f) {}
    bool has_default_values() const { return (scale_ == 1. && shift_ == 0.); }
    bool defined() const {
        return !is_runtime_value(scale_) && !is_runtime_value(shift_);
    }

    status_t set(float scale, float shift) {
        scale_ = scale;
        shift_ = shift;
        return status::success;
    }

    bool operator==(const rnn_data_qparams_t &rhs) const {
        using namespace utils;
        return equal_with_nan(scale_, rhs.scale_)
                && equal_with_nan(shift_, rhs.shift_);
    }

    float scale_;
    float shift_;
};

struct rnn_tparams_t {
    rnn_tparams_t()
        : test_mode_(false), scales_(nullptr), ngates_(0), cscale_(0.0f) {}

    ~rnn_tparams_t() {
        test_mode_ = false;
        if (scales_ != nullptr) impl::free(scales_);
        scales_ = nullptr;
        ngates_ = 0;
        cscale_ = 0.0f;
    }

    bool operator==(const rnn_tparams_t &rhs) const {
        using namespace utils;

        bool ret = test_mode_ == rhs.test_mode_ && ngates_ == rhs.ngates_
                && equal_with_nan(cscale_, rhs.cscale_);

        if (!ret) return ret;

        if (scales_) {
            if (std::memcmp(scales_, rhs.scales_, sizeof(float) * ngates_))
                return false;
        }
        return true;
    }

    bool has_default_values() const {
        return (test_mode_ == false && scales_ == nullptr && ngates_ == 0
                && cscale_ == 0.0f);
    }

    status_t set(bool mode, dim_t ngates, const float *scales, float cscale) {
        test_mode_ = mode;
        ngates_ = ngates;
        scales_ = nullptr;
        if (scales != nullptr) {
            scales_ = (float *)impl::malloc(ngates_ * sizeof(*scales_), 64);
            if (scales_ == nullptr) return status::out_of_memory;
            utils::array_copy(scales_, scales, ngates_);
        }

        cscale_ = cscale;

        return status::success;
    }

    // copy_from() functions are used for each attribute member instead of
    // operator= in order to return a status.
    // TODO: consider replacing copy_from() functions with copy-constructors and
    // std::move, since there are only a few places in the library that actually
    // use them.
    status_t copy_from(const rnn_tparams_t &other) {
        return set(
                other.test_mode_, other.ngates_, other.scales_, other.cscale_);
    }

    bool test_mode_; /* we could also use scale_ == nullptr as a test to check test_mode*/
    float *scales_;
    dim_t ngates_; /* ngates is equel to the number of scales */
    float cscale_; /* =0.0f if no c state */

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(rnn_tparams_t);
};

// Note: keep for RNN quantization
struct rnn_create_time_scales_t {
    rnn_create_time_scales_t() : count_(1), mask_(0), scales_(scales_buf_) {
        set_single_scale(1.f);
    }

    ~rnn_create_time_scales_t() { cleanup(); }

    bool operator==(const rnn_create_time_scales_t &rhs) const {
        bool ret = count_ == rhs.count_ && mask_ == rhs.mask_
                && !utils::any_null(scales_, rhs.scales_)
                && defined() == rhs.defined()
                && IMPLICATION(defined(),
                        !std::memcmp(
                                scales_, rhs.scales_, sizeof(float) * count_));
        return ret;
    }

    bool has_default_values() const {
        for (dim_t c = 0; c < count_; ++c) {
            if (scales_[c] != 1.) return false;
        }
        return true;
    }

    bool defined() const { return !is_runtime_value(scales_[0]); }

    void set_single_scale(float single_scale);
    status_t set(dim_t count, int mask, const float *scales);
    status_t set(float single_scale) {
        set_single_scale(single_scale);
        return status::success;
    }

    status_t copy_from(const rnn_create_time_scales_t &other) {
        return set(other.count_, other.mask_, other.scales_);
    }

    dim_t count_;
    int mask_;
    float *scales_;

private:
    enum { scales_buf_size = 16 };
    float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr) impl::free(scales_);

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(rnn_create_time_scales_t);
};

struct dropout_t {
    dropout_t() = default;

    bool has_default_values() const {
        return types::is_zero_md(&user_dropout_desc_)
                && (seed_dt_ == data_type::undef) && !use_offset_
                && !use_host_scalars_;
    }

    bool operator==(const dropout_t &rhs) const {
        return (user_dropout_desc_ == rhs.user_dropout_desc_)
                && (seed_dt_ == rhs.seed_dt_)
                && (use_offset_ == rhs.use_offset_)
                && (use_host_scalars_ == rhs.use_host_scalars_);
    }

    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    status_t set_default_formats(const memory_desc_t *dst_md);

    bool has_output_mask() const {
        return !dnnl::impl::types::is_zero_md(&user_dropout_desc_);
    }

    dnnl::impl::memory_desc_t dropout_desc_;
    dnnl::impl::memory_desc_t user_dropout_desc_;
    dnnl::impl::data_type_t seed_dt_ = data_type::undef;
    bool use_offset_ = false;
    bool use_host_scalars_ = false;
};

struct rnd_mode_t {
    rnd_mode_t() = default;

    bool has_default_values() const { return rounding_modes_map_.empty(); }
    bool has_default_values(int arg) const { return get(arg) == default_mode; }

    rounding_mode_t get(int arg) const {
        auto r = rounding_modes_map_.find(arg);
        if (r == rounding_modes_map_.end()) return default_mode;
        return rounding_modes_map_.at(arg);
    }

    dnnl_status_t set(int arg, dnnl_rounding_mode_t rm) {
        if (!check(arg, rm)) return status::invalid_arguments;
        if (rm != default_mode) rounding_modes_map_[arg] = rm;
        return status::success;
    }

    bool operator==(const rnd_mode_t &rhs) const {
        bool res = rounding_modes_map_.size() == rhs.rounding_modes_map_.size();
        if (!res) return false;
        for (const auto &e : rounding_modes_map_)
            if (e.second != rhs.get(e.first)) return false;
        return true;
    }

    std::unordered_map<int, rounding_mode_t> rounding_modes_map_;

private:
    const static rounding_mode_t default_mode = rounding_mode::environment;

    bool check(int arg, dnnl_rounding_mode_t rm) const {
        // Only gradients and matmul dst for gradient computation
        // can use non-default rounding mode.
        return IMPLICATION(rm != default_mode,
                utils::one_of(arg, DNNL_ARG_DST, DNNL_ARG_DIFF_SRC,
                        DNNL_ARG_DIFF_WEIGHTS));
    }

    bool check() const {
        for (auto e : rounding_modes_map_)
            if (!check(e.first, e.second)) return false;
        return true;
    }
};

struct serialization_stream_t;

struct primitive_attr_item_t {
    virtual std::unique_ptr<primitive_attr_item_t> clone() const = 0;
    virtual bool has_default_values() const = 0;
    virtual bool is_equal(const primitive_attr_item_t &other) const = 0;
    virtual size_t get_hash() const = 0;
    virtual void serialize(serialization_stream_t &stream) const = 0;
    virtual ~primitive_attr_item_t() = default;
};

struct fpmath_t {
    fpmath_t(dnnl_fpmath_mode_t mode = fpmath_mode::strict,
            bool apply_to_int = false)
        : mode_(mode), apply_to_int_(apply_to_int) {}

    bool operator==(const fpmath_t &rhs) const {
        return mode_ == rhs.mode_ && apply_to_int_ == rhs.apply_to_int_;
    }

    dnnl::impl::fpmath_mode_t mode_;
    bool apply_to_int_;
};

} // namespace impl
} // namespace dnnl

struct dnnl_post_ops {
    struct entry_t {
        // Initialize binary to avoid uninitialized memory in the union.
        entry_t() : kind(dnnl::impl::primitive_kind::undefined), binary {} {}

        entry_t(const entry_t &other) = default;

        entry_t &operator=(const entry_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            *this = entry_t(other);
            return *this;
        }
        entry_t &operator=(entry_t &&other) = default;

        struct sum_t {
            float scale;
            int32_t zero_point;
            dnnl::impl::data_type_t dt;
        };

        struct eltwise_t {
            dnnl::impl::alg_kind_t alg;
            float scale, alpha, beta;
        };

        struct depthwise_conv_t {
            dnnl::impl::dim_t kernel;
            dnnl::impl::dim_t stride;
            dnnl::impl::dim_t padding;
            dnnl::impl::data_type_t wei_dt;
            dnnl::impl::data_type_t bias_dt;
            dnnl::impl::data_type_t dst_dt;
        };

        struct binary_t {
            dnnl::impl::alg_kind_t alg;
            // This is an unmodifiable user copy of attributes which is used in
            // caching mechanism. Not to be used internally.
            dnnl::impl::memory_desc_t user_src1_desc, user_src2_desc;

            // This is a modifiable copy of memory desc. It changes format kind
            // and tag of md in case user passed format_kind::any. To be used
            // everywhere internally.
            dnnl::impl::memory_desc_t src1_desc, src2_desc;
        };

        struct prelu_t {
            int mask;
        };

        dnnl::impl::primitive_kind_t kind
                = dnnl::impl::primitive_kind::undefined;
        union {
            sum_t sum;
            eltwise_t eltwise;
            depthwise_conv_t depthwise_conv;
            binary_t binary;
            prelu_t prelu;
        };

        bool is_eltwise(bool require_scale_one = false) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::eltwise
                    && IMPLICATION(require_scale_one, eltwise.scale == 1.f);
        }

        bool is_relu(bool require_scale_one = true,
                bool require_nslope_zero = true) const {
            using namespace dnnl::impl;
            return is_eltwise(require_scale_one)
                    && eltwise.alg == alg_kind::eltwise_relu
                    && IMPLICATION(require_nslope_zero, eltwise.alpha == 0.f);
        }

        bool is_sum(bool require_scale_one = true,
                bool require_zp_zero = true) const {
            using namespace dnnl::impl;
            return kind == primitive_kind::sum
                    && IMPLICATION(require_scale_one, sum.scale == 1.f)
                    && IMPLICATION(require_zp_zero, sum.zero_point == 0);
        }

        bool is_convolution() const {
            using namespace dnnl::impl;
            return kind == primitive_kind::convolution;
        }

        bool is_binary() const {
            return kind == dnnl::impl::primitive_kind::binary;
        }

        bool is_prelu() const {
            return kind == dnnl::impl::primitive_kind::prelu;
        }

        bool is_like_binary() const { return is_binary() || is_prelu(); }

        bool is_binary_with_ternary_op() const {
            return is_binary()
                    && (binary.alg == dnnl::impl::alg_kind::binary_select);
        }

        bool is_inplace_binary() const {
            return is_binary()
                    && dnnl::impl::utils::one_of(binary.alg,
                            dnnl::impl::alg_kind::binary_mul_inplace);
        }

        dnnl::impl::status_t validate_binary(
                dnnl::impl::engine_kind_t engine_kind,
                const dnnl::impl::memory_desc_t *dst_desc) const;

        bool operator==(const entry_t &rhs) const {
            using namespace dnnl::impl;
            using namespace dnnl::impl::utils;
            if (kind != rhs.kind) { return false; }
            bool ret = true;
            switch (kind) {
                case primitive_kind::eltwise:
                    ret = eltwise.alg == rhs.eltwise.alg
                            && equal_with_nan(eltwise.scale, rhs.eltwise.scale)
                            && equal_with_nan(eltwise.alpha, rhs.eltwise.alpha)
                            && equal_with_nan(eltwise.beta, rhs.eltwise.beta);
                    break;
                case primitive_kind::sum:
                    ret = equal_with_nan(sum.scale, rhs.sum.scale)
                            && sum.zero_point == rhs.sum.zero_point
                            && sum.dt == rhs.sum.dt;
                    break;
                case primitive_kind::convolution:
                    // Depthwise Only
                    ret = depthwise_conv.kernel == rhs.depthwise_conv.kernel
                            && depthwise_conv.stride
                                    == rhs.depthwise_conv.stride
                            && depthwise_conv.padding
                                    == rhs.depthwise_conv.padding
                            && depthwise_conv.wei_dt
                                    == rhs.depthwise_conv.wei_dt
                            && depthwise_conv.bias_dt
                                    == rhs.depthwise_conv.bias_dt
                            && depthwise_conv.dst_dt
                                    == rhs.depthwise_conv.dst_dt;
                    break;
                case primitive_kind::binary:
                    ret = binary.alg == rhs.binary.alg
                            && binary.user_src1_desc
                                    == rhs.binary.user_src1_desc
                            && IMPLICATION(is_binary_with_ternary_op(),
                                    binary.user_src2_desc
                                            == rhs.binary.user_src2_desc);
                    break;
                case primitive_kind::prelu:
                    ret = prelu.mask == rhs.prelu.mask;
                    break;
                default: assert(!"unsupported post_op");
            }
            return ret;
        }

        bool operator!=(const entry_t &rhs) const {
            return !this->operator==(rhs);
        }
    };

    dnnl_post_ops() = default;
    ~dnnl_post_ops() = default;

    dnnl::impl::status_t append_sum(float scale, int32_t zero_point = 0,
            dnnl::impl::data_type_t dt = dnnl_data_type_undef);
    dnnl::impl::status_t append_eltwise(
            float scale, dnnl::impl::alg_kind_t alg, float alpha, float beta);
    dnnl::impl::status_t append_dw(dnnl::impl::data_type_t wei_dt,
            dnnl::impl::data_type_t bias_dt, dnnl::impl::data_type_t dst_dt,
            dnnl::impl::dim_t kernel_size, dnnl::impl::dim_t stride_size,
            dnnl::impl::dim_t padding_l_size);
    dnnl::impl::status_t append_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc,
            const dnnl::impl::memory_desc_t *user_src2_desc = nullptr);
    dnnl::impl::status_t append_prelu(int mask);

    dnnl::impl::status_t prepend_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc,
            const dnnl::impl::memory_desc_t *user_src2_desc = nullptr);

    int find(dnnl::impl::primitive_kind_t kind, int start = 0,
            int stop = -1) const {
        if (stop == -1) stop = len();
        stop = dnnl::impl::nstl::min(stop, len());
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) return idx;
        return -1;
    }

    dnnl::impl::data_type_t get_sum_dt(
            const dnnl::impl::data_type_t dst_dt, int sum_ind = -1) const {
        if (sum_ind == -1) sum_ind = find(dnnl::impl::primitive_kind::sum);
        if (sum_ind == -1) return dst_dt;
        const auto sum_dt = entry_[sum_ind].sum.dt;
        if (sum_dt != dnnl::impl::data_type::undef) return sum_dt;
        return dst_dt;
    }

    int len() const { return (int)entry_.size(); }
    bool has_default_values(
            const std::vector<dnnl::impl::primitive_kind_t> &skip_pk
            = {}) const {
        if (len() == 0) return true;

        for (const auto &e : entry_) {
            bool skip = false;
            for (const auto &pk : skip_pk)
                if (e.kind == pk) {
                    skip = true;
                    break;
                }
            if (skip) continue;
            return false;
        }
        return true;
    }

    bool has_inplace_binary() const {
        for (auto &e : entry_)
            if (e.is_inplace_binary()) return true;
        return false;
    }

    dnnl::impl::status_t set_default_formats(
            const dnnl::impl::memory_desc_t *dst_md);

    bool check_sum_consistency(const dnnl::impl::data_type_t dst_dt,
            const bool is_int8,
            const bool diverse_sum_dt_allowed = false) const;

    bool sum_with_default_dt(
            dnnl::impl::data_type_t dst_dt = dnnl_data_type_undef) const {
        int sum_ind = find(dnnl::impl::primitive_kind::sum);
        return sum_ind == -1 || entry_[sum_ind].sum.dt == dnnl_data_type_undef
                || entry_[sum_ind].sum.dt == dst_dt;
    }

    dnnl::impl::status_t validate_binary(dnnl::impl::engine_kind_t engine_kind,
            const dnnl::impl::memory_desc_t *dst_desc) const;

    bool contain(dnnl::impl::primitive_kind_t kind, int index) const {
        return find(kind, index, index + 1) == index;
    }

    bool operator==(const dnnl_post_ops &rhs) const {
        bool ret = len() == rhs.len();
        for (int i = 0; i < len(); ++i)
            ret = ret && entry_[i] == rhs.entry_[i];
        return ret;
    }

    std::vector<entry_t> entry_;

    // Since binary post op accepts no more than 32 memory arguments by
    // design, we limit the amount of post-ops to 32.
    static constexpr int post_ops_limit = 32;

private:
    dnnl::impl::status_t validate_binary(dnnl::impl::alg_kind_t alg,
            const dnnl::impl::memory_desc_t *user_src1_desc,
            const dnnl::impl::memory_desc_t *user_src2_desc) const;

    bool check_sum_consistent_dt(const dnnl::impl::data_type_t dst_dt,
            const bool diverse_sum_dt_allowed = false) const;

    bool check_sum_consistent_quantization(
            const dnnl::impl::data_type_t dst_dt, const bool is_int8) const;
};

struct dnnl_primitive_attr {
    dnnl_primitive_attr()
        : scratchpad_mode_(dnnl::impl::scratchpad_mode::library)
        , fpmath_(dnnl::impl::get_fpmath_mode(), false)
        , acc_mode_(dnnl::impl::accumulation_mode::strict)
        , deterministic_(false) {}

    ~dnnl_primitive_attr() = default;

    dnnl_primitive_attr *clone() const {
        return new dnnl_primitive_attr(*this);
    }

    dnnl_primitive_attr(const dnnl_primitive_attr &other) {
        if (copy_from(other) != dnnl::impl::status::success)
            is_initialized_ = false;
    }

    dnnl::impl::status_t copy_from(const dnnl_primitive_attr &other) {
        using namespace dnnl::impl;

        scales_ = other.scales_;
        zero_points_ = other.zero_points_;
        precomputed_reductions_ = other.precomputed_reductions_;
        rounding_mode_ = other.rounding_mode_;
        scratchpad_mode_ = other.scratchpad_mode_;
        fpmath_ = other.fpmath_;
        acc_mode_ = other.acc_mode_;
        deterministic_ = other.deterministic_;
        post_ops_ = other.post_ops_;
        rnn_data_qparams_ = other.rnn_data_qparams_;
        CHECK(rnn_weights_qparams_.copy_from(other.rnn_weights_qparams_));
        CHECK(rnn_weights_projection_qparams_.copy_from(
                other.rnn_weights_projection_qparams_));
        CHECK(rnn_tparams_.copy_from(other.rnn_tparams_));
        if (other.gpu_attr_) gpu_attr_ = other.gpu_attr_->clone();
        dropout_ = other.dropout_;

        return status::success;
    }

    /**
     * This function returns a copy of the attributes with the parameters reset
     * to the user-provided ones.
     *
     * This can be helpful in situations where an implementation has already
     * modified the attributes via `set_default_formats`, but later needs the
     * original attributes to decide whether a fallback is possible.
     *
     * For example, brgemm matmul must ensure that a gemm primitive descriptor
     * can be created before making a final decision on fallback. At that point,
     * the attributes may have already been modified. It is not always trivial
     * to rearrange the initialization order of the implementation configuration
     * to reach the decision point before the attributes are modified.
     * Therefore, it is more robust to have the ability to query the original
     * attributes.
     */
    dnnl::impl::status_t copy_from_and_reset(const dnnl_primitive_attr &other) {
        CHECK(copy_from(other));

        // Restore user provided parameters for the binary post-op.
        auto &entries = post_ops_.entry_;
        for (int idx = 0; idx < post_ops_.len(); ++idx) {
            if (!post_ops_.contain(dnnl::impl::primitive_kind::binary, idx))
                continue;
            entries[idx].binary.src1_desc = entries[idx].binary.user_src1_desc;
            entries[idx].binary.src2_desc = entries[idx].binary.user_src2_desc;
        }

        // Restore user provided parameters for the dropout attribute.
        dropout_.dropout_desc_ = dropout_.user_dropout_desc_;

        return dnnl::impl::status::success;
    }

    bool is_initialized() const { return is_initialized_; }

    enum class skip_mask_t : unsigned {
        none = 0,
        scales = 1u << 1,
        scales_groups = (unsigned)scales | (1u << 2),
        scales_data_type = (unsigned)scales | (1u << 3),
        zero_points = 1u << 4,
        zero_points_groups = (unsigned)zero_points | (1u << 5),
        zero_points_data_type = (unsigned)zero_points | (1u << 6),
        post_ops = 1u << 7,
        sum_dt = 1u << 8,
        rnn_data_qparams = 1u << 9,
        rnn_weights_qparams = 1u << 10,
        rnn_tparams = 1u << 11,
        rnn_weights_projection_qparams = 1u << 12,
        gpu_attr = 1u << 13,
        accumulation_mode = 1u << 14,
        fpmath_mode = 1u << 15,
        dropout = 1u << 16,
        rounding_mode = 1u << 17,
        precomputed_reductions = 1u << 18,
        post_ops_inplace = (unsigned)post_ops | (1u << 19),
    };

    /** Returns true if the attributes have default values.
     *
     * @note The scratchpad_mode_ is not take into account */
    bool has_default_values(skip_mask_t mask = skip_mask_t::none,
            dnnl::impl::data_type_t dst_dt = dnnl_data_type_undef) const;

    /** Returns true if the attributes are fully defined. */
    bool defined(skip_mask_t mask = skip_mask_t::none) const;

    bool operator==(const dnnl_primitive_attr &rhs) const {
        bool ret = scratchpad_mode_ == rhs.scratchpad_mode_
                && fpmath_ == rhs.fpmath_ && acc_mode_ == rhs.acc_mode_
                && deterministic_ == rhs.deterministic_
                && scales_ == rhs.scales_ && zero_points_ == rhs.zero_points_
                && precomputed_reductions_ == rhs.precomputed_reductions_
                && post_ops_ == rhs.post_ops_
                && rnn_data_qparams_ == rhs.rnn_data_qparams_
                && rnn_weights_qparams_ == rhs.rnn_weights_qparams_
                && rnn_weights_projection_qparams_
                        == rhs.rnn_weights_projection_qparams_
                && rnn_tparams_ == rhs.rnn_tparams_
                && ((gpu_attr_ && rhs.gpu_attr_
                            && gpu_attr_->is_equal(*rhs.gpu_attr_))
                        || (!gpu_attr_ && !rhs.gpu_attr_))
                && dropout_ == rhs.dropout_
                && rounding_mode_ == rhs.rounding_mode_;
        return ret;
    }

    dnnl::impl::status_t set_fpmath_mode(
            dnnl::impl::fpmath_mode_t fpmath_mode, bool apply_to_int = false);
    dnnl::impl::status_t set_accumulation_mode(
            dnnl::impl::accumulation_mode_t am);
    dnnl::impl::status_t set_dropout(
            const dnnl::impl::memory_desc_t *dropout_desc,
            dnnl::impl::data_type_t seed_dt, bool user_offset,
            bool use_host_scalars);
    dnnl::impl::status_t set_scratchpad_mode(
            dnnl::impl::scratchpad_mode_t scratchpad_mode);
    dnnl::impl::status_t set_post_ops(const dnnl::impl::post_ops_t &post_ops);
    dnnl::impl::status_t set_gpu_attr(
            const dnnl::impl::primitive_attr_item_t &gpu_attr);
    dnnl::impl::status_t set_default_formats(
            const dnnl::impl::memory_desc_t *dst_md);

    /* Auxiliary functions */

    bool mayiconvert(dnnl::impl::data_type_t dt_from,
            dnnl::impl::data_type_t dt_to) const {

        auto mayidownconvert = [](dnnl::impl::fpmath_mode_t fpmath_mode,
                                       dnnl::impl::data_type_t dt_from,
                                       dnnl::impl::data_type_t dt_to) -> bool {
            using namespace dnnl::impl;

            bool is_compat = is_fpsubtype(dt_to, dt_from);
            auto can_downconvert = [&]() {
                switch (fpmath_mode) {
                    case fpmath_mode::strict: return dt_from == dt_to;
                    case fpmath_mode::any: return true;
                    case fpmath_mode::bf16:
                        return is_fpsubtype(data_type::bf16, dt_to);
                    case fpmath_mode::f16:
                        return is_fpsubtype(data_type::f16, dt_to);
                    case fpmath_mode::tf32:
                        return is_fpsubtype(data_type::tf32, dt_to);
                    default: return false;
                }
            };
            return is_compat && can_downconvert();
        };

        if (dnnl::impl::types::is_integral_dt(dt_from)) {
            // integer inputs can be converted only:
            // - if apply_to_int_fpmath_ is enabled, and
            // - to an fp type compatible with fpmath mode
            // `dt_from` = `f32` to override `is_compat` check.
            return fpmath_.apply_to_int_
                    && mayidownconvert(
                            fpmath_.mode_, dnnl::impl::data_type::f32, dt_to);
        } else {
            // fp inputs can be converted only:
            // - if target datatype is bigger
            // - or if fpmath mode allows the conversion
            return dnnl::impl::is_fpsubtype(dt_from, dt_to)
                    || mayidownconvert(fpmath_.mode_, dt_from, dt_to);
        }
    }

    // NOTE: make sure that the types below have overloaded comparison operator
    dnnl::impl::scales_t scales_;
    dnnl::impl::zero_points_t zero_points_;
    dnnl::impl::precomputed_reductions_t precomputed_reductions_;
    dnnl::impl::scratchpad_mode_t scratchpad_mode_;
    dnnl::impl::fpmath_t fpmath_;
    dnnl::impl::accumulation_mode_t acc_mode_;
    bool deterministic_;
    dnnl::impl::post_ops_t post_ops_;
    dnnl::impl::rnn_data_qparams_t rnn_data_qparams_;
    dnnl::impl::rnn_create_time_scales_t rnn_weights_qparams_;
    dnnl::impl::rnn_create_time_scales_t rnn_weights_projection_qparams_;
    dnnl::impl::rnn_tparams_t rnn_tparams_;
    dnnl::impl::dropout_t dropout_;
    dnnl::impl::rnd_mode_t rounding_mode_;

    std::unique_ptr<dnnl::impl::primitive_attr_item_t> gpu_attr_;

    // Set to `false` when copying from another attribute fails, so that
    // `status::out_of_memory` is reported.
    bool is_initialized_ = true;

    dnnl_primitive_attr &operator=(const dnnl_primitive_attr &other) = delete;
};

inline dnnl_primitive_attr::skip_mask_t operator|(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator&(
        dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t &operator|=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs | rhs;
    return lhs;
}
inline dnnl_primitive_attr::skip_mask_t &operator&=(
        dnnl_primitive_attr::skip_mask_t &lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    lhs = lhs & rhs;
    return lhs;
}
inline bool operator!=(dnnl_primitive_attr::skip_mask_t lhs,
        dnnl_primitive_attr::skip_mask_t rhs) {
    return (static_cast<unsigned>(lhs) != static_cast<unsigned>(rhs));
}
inline dnnl_primitive_attr::skip_mask_t operator~(
        dnnl_primitive_attr::skip_mask_t rhs) {
    return static_cast<dnnl_primitive_attr::skip_mask_t>(
            ~static_cast<unsigned>(rhs));
}

#endif
