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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "primitive_attr.hpp"
#include "primitive_hashing.hpp"
#include "primitive_serialization.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "verbose.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr() {
    static const primitive_attr_t default_attr_instance;
    return default_attr_instance;
}

void rnn_create_time_scales_t::set_single_scale(float scale) {
    count_ = 1;
    mask_ = 0;
    scales_ = scales_buf_;
    if (is_runtime_value(scale)) {
        scales_[0] = scale;
    } else {
        utils::array_set(scales_, scale, scales_buf_size);
    }
}

status_t rnn_create_time_scales_t::set(
        dim_t count, int mask, const float *scales) {
    cleanup();

    count_ = count;
    mask_ = mask;

    if (is_runtime_value(*scales)) {
        scales_ = scales_buf_;
        scales_[0] = *scales;
    } else if (count_ == 1) {
        set_single_scale(scales[0]);
    } else {
        scales_ = (float *)impl::malloc(count_ * sizeof(*scales_), 64);
        if (scales_ == nullptr) return status::out_of_memory;

        for (dim_t c = 0; c < count_; ++c)
            scales_[c] = scales[c];
    }

    return status::success;
}

status_t dropout_t::set_default_formats(const memory_desc_t *dst_md) {
    auto is_any_or_undef = [](format_kind_t kind) {
        return one_of(kind, dnnl_format_kind_any, dnnl_format_kind_undef);
    };
    const bool dst_ok = dst_md && !is_any_or_undef(dst_md->format_kind);
    if (dst_ok && is_any_or_undef(dropout_desc_.format_kind)) {
        const memory_desc_wrapper dst_mdw(dst_md);
        CHECK(memory_desc_init_by_blocking_desc(
                dropout_desc_, dst_mdw.blocking_desc()));
    }
    return (dst_ok) ? status::success : status::invalid_arguments;
}

size_t dropout_t::get_hash() const {
    size_t seed = 0;
    seed = hash_combine(
            seed, primitive_hashing::get_md_hash(user_dropout_desc_));
    seed = hash_combine(seed, seed_dt_);
    seed = hash_combine(seed, use_offset_);
    seed = hash_combine(seed, use_host_scalars_);
    return seed;
}

void dropout_t::serialize(serialization_stream_t &sstream) const {
    dnnl::impl::serialize(sstream, user_dropout_desc_);
    sstream.append(seed_dt_);
    sstream.append(use_offset_);
    sstream.append(use_host_scalars_);
}
} // namespace impl
} // namespace dnnl

bool primitive_attr_t::has_default_values(dnnl_primitive_attr::skip_mask_t mask,
        dnnl::impl::data_type_t dst_dt) const {
    using smask_t = skip_mask_t;
    bool ok = true;

#define CHECK_ARG(x) ok = ok && (x)
#define CHECK_MASK(mask_name, mask_field) \
    CHECK_ARG(IMPLICATION( \
            (bool)(~mask & (mask_name)), (mask_field).has_default_values()))
    CHECK_MASK(smask_t::scales, scales_);
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::scales_groups),
            scales_.has_default_groups()));
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::scales_data_type),
            scales_.has_default_data_type()));
    CHECK_MASK(smask_t::zero_points, zero_points_);
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::zero_points_groups),
            zero_points_.has_default_groups()));
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::zero_points_data_type),
            zero_points_.has_default_data_type()));
    CHECK_MASK(smask_t::precomputed_reductions, precomputed_reductions_);
    CHECK_MASK(smask_t::post_ops, post_ops_);
    CHECK_MASK(smask_t::rnn_data_qparams, rnn_data_qparams_);
    CHECK_MASK(smask_t::rnn_weights_qparams, rnn_weights_qparams_);
    CHECK_MASK(smask_t::rnn_weights_projection_qparams,
            rnn_weights_projection_qparams_);
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::sum_dt),
            post_ops_.sum_with_default_dt(dst_dt)));
    bool gpu_attr_ok = IMPLICATION((bool)(~mask & smask_t::gpu_attr),
            !gpu_attr_ || gpu_attr_->has_default_values());
    CHECK_ARG(gpu_attr_ok);
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::accumulation_mode),
            utils::one_of(acc_mode_, dnnl::impl::accumulation_mode::strict,
                    dnnl::impl::accumulation_mode::relaxed,
                    dnnl::impl::accumulation_mode::any)));
    CHECK_ARG(IMPLICATION(
            (bool)(~mask & smask_t::dropout), dropout_.has_default_values()));
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::rounding_mode),
            rounding_mode_.has_default_values()));
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::post_ops_inplace),
            !post_ops_.has_inplace_binary()));
    CHECK_ARG(this->defined(smask_t::none));
    bool fpmath_mode_ok = IMPLICATION(
            (bool)(~mask & smask_t::fpmath_mode) && fpmath_.apply_to_int_,
            fpmath_.mode_ == fpmath_mode::strict);
    CHECK_ARG(fpmath_mode_ok);
    return ok;
#undef CHECK_MASK
#undef CHECK_ARG
}

bool primitive_attr_t::defined(dnnl_primitive_attr::skip_mask_t mask) const {
    using smask_t = skip_mask_t;
    bool ok = true;
#define CHECK_ARG(x) ok = ok && (x)
#define CHECK_MASK(mask_name, mask_field) \
    CHECK_ARG(IMPLICATION((bool)(~mask & (mask_name)), (mask_field).defined()))
    CHECK_MASK(smask_t::rnn_data_qparams, rnn_data_qparams_);
    CHECK_MASK(smask_t::rnn_weights_qparams, rnn_weights_qparams_);
    CHECK_MASK(smask_t::rnn_weights_projection_qparams,
            rnn_weights_projection_qparams_);
    return ok;
#undef CHECK_MASK
#undef CHECK_ARG
}

status_t post_ops_t::append_sum(
        float scale, int32_t zero_point, data_type_t dt) {
    if (is_runtime_value(scale)) return invalid_arguments;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::sum;
    e.sum.scale = scale;
    e.sum.zero_point = zero_point;
    e.sum.dt = dt;
    return success;
}

status_t post_ops_t::append_eltwise(
        float scale, alg_kind_t alg, float alpha, float beta) {
    if (!math::is_eltwise_ok(data_type::f32, alg, alpha, beta))
        return invalid_arguments;
    if (is_runtime_value(scale)) return invalid_arguments;
    if (is_runtime_value(alpha)) return invalid_arguments;
    if (is_runtime_value(beta)) return invalid_arguments;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::eltwise;
    e.eltwise.scale = scale;
    e.eltwise.alg = alg;
    e.eltwise.alpha = alpha;
    e.eltwise.beta = beta;
    return success;
}

status_t post_ops_t::append_dw(data_type_t wei_dt, data_type_t bias_dt,
        data_type_t dst_dt, dim_t kernel_size, dim_t stride_size,
        dim_t padding_l_size) {
    if (len() == post_ops_limit) return out_of_memory;
    bool ok = wei_dt != data_type::undef && dst_dt != data_type::undef;
    if (!ok) return invalid_arguments;

    ok = ok && kernel_size > 0 && stride_size > 0;
    if (!ok) return invalid_arguments;

    // Avoiding cases when kernel in pad area
    ok = ok && (padding_l_size + 1) <= kernel_size;
    if (!ok) return invalid_arguments;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::convolution;
    auto &d = e.depthwise_conv;
    d.kernel = kernel_size;
    d.stride = stride_size;
    d.padding = padding_l_size;
    d.wei_dt = wei_dt;
    d.bias_dt = bias_dt;
    d.dst_dt = dst_dt;

    return success;
}

status_t post_ops_t::validate_binary(alg_kind_t alg,
        const memory_desc_t *user_src1_desc,
        const memory_desc_t *user_src2_desc) const {

    if (len() == post_ops_limit) return out_of_memory;
    using namespace alg_kind;
    bool alg_ok = one_of(alg, binary_add, binary_mul, binary_max, binary_min,
            binary_div, binary_sub, binary_ge, binary_gt, binary_le, binary_lt,
            binary_eq, binary_ne, binary_select, binary_mul_inplace);
    bool is_ternary_op = (alg == binary_select);

    VCHECK_ATTR(alg_ok, VERBOSE_BAD_ALGORITHM);
    CHECK(memory_desc_sanity_check(*user_src1_desc));

    // A sub-byte src1 with `any` format gets its strides resolved from
    // `dst_md` in `set_default_formats`, which inherits dst strides verbatim
    // and cannot guarantee the storage-unit (byte) alignment sub-byte types
    // require. If a concrete format is passed, it gets validated using memory
    // descriptor creation public API which enforces proper strides.
    VCHECK_ATTR(IMPLICATION(user_src1_desc->format_kind == format_kind::any,
                        types::data_type_bits(user_src1_desc->data_type) >= 8),
            VERBOSE_UNSUPPORTED_DT);

    // Additional check to restrict run-time dimension usage until supported.
    for (int d = 0; d < user_src1_desc->ndims; ++d) {
        VCHECK_ATTR(!is_runtime_value(user_src1_desc->dims[d]),
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    }

    // Additional checks if the algorithm involves ternary inputs
    if (is_ternary_op) {
        CHECK(memory_desc_sanity_check(*user_src2_desc));
        VCHECK_ATTR(
                IMPLICATION(user_src2_desc->format_kind == format_kind::any,
                        types::data_type_bits(user_src2_desc->data_type) >= 8),
                VERBOSE_UNSUPPORTED_DT);
        // The conditional input is s8 only, as for the binary primitive.
        VCHECK_ATTR(user_src2_desc->data_type == data_type::s8,
                VERBOSE_INVALID_DATATYPE, "bin_po src2");
        for (int d = 0; d < user_src2_desc->ndims; ++d) {
            VCHECK_ATTR(!is_runtime_value(user_src2_desc->dims[d]),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        }
    }

    return success;
}

status_t post_ops_t::append_binary(alg_kind_t alg,
        const memory_desc_t *user_src1_desc,
        const memory_desc_t *user_src2_desc) {
    CHECK(validate_binary(alg, user_src1_desc, user_src2_desc));

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::binary;
    e.binary.alg = alg;

    e.binary.user_src1_desc = *user_src1_desc;
    e.binary.src1_desc = *user_src1_desc;

    if (user_src2_desc) {
        e.binary.user_src2_desc = *user_src2_desc;
        e.binary.src2_desc = *user_src2_desc;
    }
    return success;
}

status_t post_ops_t::prepend_binary(alg_kind_t alg,
        const memory_desc_t *user_src1_desc,
        const memory_desc_t *user_src2_desc) {
    CHECK(validate_binary(alg, user_src1_desc, user_src2_desc));

    entry_.emplace(entry_.begin());
    auto &e = entry_[0];
    e.kind = primitive_kind::binary;
    e.binary.alg = alg;

    e.binary.user_src1_desc = *user_src1_desc;
    e.binary.src1_desc = *user_src1_desc;

    if (alg == alg_kind::binary_select) {
        e.binary.user_src2_desc = *user_src2_desc;
        e.binary.src2_desc = *user_src2_desc;
    }
    return success;
}

status_t post_ops_t::append_prelu(int mask) {
    if (len() == post_ops_limit) return out_of_memory;

    auto it_entry = entry_.emplace(entry_.end());
    it_entry->kind = primitive_kind::prelu;
    it_entry->prelu.mask = mask;

    return success;
}

status_t post_ops_t::set_default_formats(const memory_desc_t *dst_md) {
    for (int idx = 0; idx < len(); ++idx) {
        if (!contain(primitive_kind::binary, idx)) continue;

        auto &src1_md = entry_[idx].binary.src1_desc;
        const memory_desc_wrapper src1_mdw(src1_md);

        const memory_desc_wrapper dst_mdw(dst_md);

        // 1D tensors should be plain abx.
        if (src1_mdw.format_any()) {
            assert(!dst_mdw.format_any());

            // Note: sub-byte src1 is rejected by the binary post-op API. To
            // support it here, strides must be computed from `dst_md` format
            // with storage-unit (byte) alignment rather than inherited
            // verbatim, which `memory_desc_init_by_blocking_desc` does not do.
            if (src1_mdw.count_non_unit_dims(1))
                CHECK(memory_desc_init_by_strides(src1_md, nullptr));
            else
                CHECK(memory_desc_init_by_blocking_desc(
                        src1_md, dst_mdw.blocking_desc()));
        }

        auto &src2_md = entry_[idx].binary.src2_desc;
        const memory_desc_wrapper src2_mdw(src2_md);

        if (entry_[idx].is_binary_with_ternary_op() && src2_mdw.format_any()) {
            assert(!dst_mdw.format_any());

            if (src1_mdw.count_non_unit_dims(1))
                CHECK(memory_desc_init_by_strides(src2_md, nullptr));
            else
                CHECK(memory_desc_init_by_blocking_desc(
                        src2_md, dst_mdw.blocking_desc()));
        }
    }

    return status::success;
}

bool post_ops_t::check_sum_consistent_dt(const data_type_t dst_dt,
        const bool diverse_sum_dt_is_supported) const {
    int sum_ind = find(dnnl::impl::primitive_kind::sum);
    if (sum_ind == -1) return true;
    const auto sum_dt = entry_[sum_ind].sum.dt;

    // sum dt and dst dt must have the same size
    const bool compatible_dt_size = IMPLICATION(
            !utils::one_of(dnnl_data_type_undef, sum_dt, dst_dt),
            types::data_type_size(dst_dt) == types::data_type_size(sum_dt));
    if (!compatible_dt_size) return false;
    if (diverse_sum_dt_is_supported) return true;

    bool ok = true;
    while ((sum_ind = find(dnnl::impl::primitive_kind::sum, sum_ind + 1)) != -1)
        ok = ok && entry_[sum_ind].sum.dt == sum_dt;
    return ok;
}

bool post_ops_t::check_sum_consistent_quantization(
        const data_type_t dst_dt, const bool is_int8) const {
    using namespace data_type;
    using namespace primitive_kind;
    bool ok = true;
    int sum_ind = -1;
    while ((sum_ind = find(sum, sum_ind + 1)) != -1) {
        const auto &sum_e = entry_[sum_ind].sum;
        // validate interface requirements
        ok = ok && IMPLICATION(!is_int8, sum_e.zero_point == 0)
                && IMPLICATION(sum_e.zero_point != 0,
                        one_of(get_sum_dt(dst_dt, sum_ind), s8, u8, s32));
    }
    return ok;
}

bool post_ops_t::check_sum_consistency(const data_type_t dst_dt,
        const bool is_int8, const bool diverse_sum_dt_is_supported) const {

    return check_sum_consistent_dt(dst_dt, diverse_sum_dt_is_supported)
            && check_sum_consistent_quantization(dst_dt, is_int8);
}

status_t post_ops_t::entry_t::validate_binary(
        engine_kind_t engine_kind, const memory_desc_t *dst_md) const {
    if (!is_binary()) return status::success;

    VCHECK_ATTR(dst_md->ndims == binary.user_src1_desc.ndims,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "dst", "bin_po src1",
            dst_md->ndims, binary.user_src1_desc.ndims);

    VCHECK_ATTR(IMPLICATION(engine_kind == dnnl_cpu,
                        binary.user_src1_desc.data_type != data_type::f64),
            VERBOSE_INVALID_DATATYPE, "bin_po src1");

    if (is_binary_with_ternary_op()) {
        VCHECK_ATTR(dst_md->ndims == binary.user_src2_desc.ndims,
                VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "dst", "bin_po src2",
                dst_md->ndims, binary.user_src2_desc.ndims);

        VCHECK_ATTR(IMPLICATION(engine_kind == dnnl_cpu,
                            binary.user_src2_desc.data_type != data_type::f64),
                VERBOSE_INVALID_DATATYPE, "bin_po src2");
    }

    if (is_inplace_binary()) {
        const memory_desc_wrapper dst_d(dst_md);
        const memory_desc_wrapper src1_d(binary.user_src1_desc);
        VCHECK_ATTR(dst_d.data_type() == src1_d.data_type(),
                VERBOSE_INCONSISTENT_DT, "dst", "bin_po src1");
        VCHECK_ATTR(!dst_d.has_runtime_dims_or_strides(),
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        VCHECK_ATTR(
                utils::array_cmp(dst_d.dims(), src1_d.dims(), src1_d.ndims()),
                VERBOSE_INCONSISTENT_DIM, "dst", -1, "bin_po src1", -1);
        VCHECK_ATTR(IMPLICATION(!src1_d.format_any(),
                            !dst_d.format_any() && dst_d.similar_to(src1_d)),
                VERBOSE_INCONSISTENT_MDS, "dst", "bin_po src1");
    }
    return status::success;
}

status_t post_ops_t::validate_binary(
        engine_kind_t engine_kind, const memory_desc_t *dst_md) const {
    for (const auto &e : entry_) {
        CHECK(e.validate_binary(engine_kind, dst_md));
    }

    return status::success;
}

status_t primitive_attr_t::set_dropout(const memory_desc_t *user_dropout_desc,
        data_type_t seed_dt, bool use_offset, bool use_host_scalars) {
    VCHECK_ATTR(user_dropout_desc, VERBOSE_NULL_ARG);
    VCHECK_ATTR(utils::one_of(user_dropout_desc->data_type, data_type::u8,
                        data_type::undef),
            VERBOSE_INVALID_DATATYPE, "dropout");
    VCHECK_ATTR(utils::one_of(seed_dt, data_type::s32, data_type::s64),
            VERBOSE_INVALID_DATATYPE, "dropout");

    dropout_.user_dropout_desc_ = *user_dropout_desc;
    dropout_.dropout_desc_ = *user_dropout_desc;
    dropout_.seed_dt_ = seed_dt;
    dropout_.use_offset_ = use_offset;
    dropout_.use_host_scalars_ = use_host_scalars;
    return status::success;
}

status_t primitive_attr_t::set_fpmath_mode(
        fpmath_mode_t fpmath_mode, bool apply_to_int) {
    auto st = check_fpmath_mode(fpmath_mode);
    if (st == success) {
        fpmath_.mode_ = fpmath_mode;
        fpmath_.apply_to_int_ = apply_to_int;
    }
    return st;
}

status_t primitive_attr_t::set_accumulation_mode(accumulation_mode_t am) {
    VCONDCHECK(primitive, create, check, attr,
            utils::one_of(am, accumulation_mode::strict,
                    accumulation_mode::relaxed, accumulation_mode::any,
                    accumulation_mode::s32, accumulation_mode::f32,
                    accumulation_mode::f16),
            invalid_arguments, VERBOSE_INVALID_ACC_MODE,
            dnnl_accumulation_mode2str(am));
    acc_mode_ = am;
    return success;
}

status_t primitive_attr_t::set_scratchpad_mode(
        scratchpad_mode_t scratchpad_mode) {
    const bool ok = one_of(
            scratchpad_mode, scratchpad_mode::library, scratchpad_mode::user);
    if (!ok) return invalid_arguments;

    scratchpad_mode_ = scratchpad_mode;
    return success;
}

status_t primitive_attr_t::set_post_ops(const post_ops_t &post_ops) {
    post_ops_ = post_ops;
    return status::success;
}

status_t primitive_attr_t::set_default_formats(const memory_desc_t *dst_md) {
    CHECK(post_ops_.set_default_formats(dst_md));
    CHECK(dropout_.set_default_formats(dst_md));
    return status::success;
}

status_t primitive_attr_t::set_gpu_attr(const primitive_attr_item_t &gpu_attr) {
    gpu_attr_ = gpu_attr.clone();
    return status::success;
}

/* Public C API */

status_t dnnl_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr) return invalid_arguments;

    return safe_ptr_assign(*attr, new dnnl_primitive_attr);
}

status_t dnnl_primitive_attr_clone(
        primitive_attr_t **attr, const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr)) return invalid_arguments;

    auto new_attr = utils::make_unique<primitive_attr_t>(*existing_attr);
    if (!new_attr->is_initialized()) return out_of_memory;

    return safe_ptr_assign(*attr, new_attr.release());
}

status_t dnnl_primitive_attr_destroy(primitive_attr_t *attr) {
    delete attr;

    return success;
}

status_t dnnl_primitive_attr_get_dropout(
        const primitive_attr_t *attr, const memory_desc_t **user_dropout_desc) {
    if (any_null(attr)) return invalid_arguments;
    if (user_dropout_desc)
        *user_dropout_desc = &attr->dropout_.user_dropout_desc_;
    return success;
}

status_t dnnl_primitive_attr_get_dropout_v2(const primitive_attr_t *attr,
        const memory_desc_t **user_dropout_desc, data_type_t *seed_dt,
        int *use_offset, int *use_host_scalars) {
    if (any_null(attr)) return invalid_arguments;
    if (user_dropout_desc)
        *user_dropout_desc = &attr->dropout_.user_dropout_desc_;
    if (seed_dt) *seed_dt = attr->dropout_.seed_dt_;
    if (use_offset) *use_offset = attr->dropout_.use_offset_;
    if (use_host_scalars) *use_host_scalars = attr->dropout_.use_host_scalars_;
    return success;
}

status_t dnnl_primitive_attr_set_dropout(
        primitive_attr_t *attr, const memory_desc_t *user_dropout_desc) {
    return dnnl_primitive_attr_set_dropout_v2(
            attr, user_dropout_desc, dnnl_s32, false, false);
}

status_t dnnl_primitive_attr_set_dropout_v2(primitive_attr_t *attr,
        const memory_desc_t *user_dropout_desc, data_type_t seed_dt,
        int use_offset, int use_host_scalars) {
    if (any_null(attr)) return invalid_arguments;
    return attr->set_dropout(
            user_dropout_desc, seed_dt, use_offset, use_host_scalars);
}

status_t dnnl_primitive_attr_get_fpmath_mode(
        const primitive_attr_t *attr, fpmath_mode_t *mode) {
    if (any_null(attr, mode)) return invalid_arguments;
    *mode = attr->fpmath_.mode_;
    return success;
}

status_t dnnl_primitive_attr_set_fpmath_mode(
        primitive_attr_t *attr, fpmath_mode_t mode) {
    if (any_null(attr)) return invalid_arguments;
    return attr->set_fpmath_mode(mode, false);
}

status_t dnnl_primitive_attr_get_fpmath_mode_v2(
        const primitive_attr_t *attr, fpmath_mode_t *mode, int *apply_to_int) {
    if (!attr) return invalid_arguments;
    if (mode) *mode = attr->fpmath_.mode_;
    if (apply_to_int) *apply_to_int = attr->fpmath_.apply_to_int_;
    return success;
}

status_t dnnl_primitive_attr_set_fpmath_mode_v2(
        primitive_attr_t *attr, fpmath_mode_t mode, int apply_to_int_fpmath) {
    if (any_null(attr)) return invalid_arguments;
    return attr->set_fpmath_mode(mode, apply_to_int_fpmath);
}

status_t dnnl_primitive_attr_get_accumulation_mode(
        const primitive_attr_t *attr, accumulation_mode_t *am) {
    if (any_null(attr, am)) return invalid_arguments;
    *am = attr->acc_mode_;
    return success;
}

status_t dnnl_primitive_attr_set_accumulation_mode(
        primitive_attr_t *attr, accumulation_mode_t am) {
    if (any_null(attr)) return invalid_arguments;
    return attr->set_accumulation_mode(am);
}

status_t dnnl_primitive_attr_get_deterministic(
        const primitive_attr_t *attr, int *d) {
    if (any_null(attr, d)) return invalid_arguments;
    *d = attr->deterministic_;
    return success;
}

status_t dnnl_primitive_attr_set_deterministic(primitive_attr_t *attr, int d) {
    if (any_null(attr)) return invalid_arguments;
    attr->deterministic_ = d;
    return success;
}

status_t dnnl_primitive_attr_get_scratchpad_mode(
        const primitive_attr_t *attr, scratchpad_mode_t *scratchpad_mode) {
    if (any_null(attr, scratchpad_mode)) return invalid_arguments;

    *scratchpad_mode = attr->scratchpad_mode_;

    return success;
}

status_t dnnl_primitive_attr_set_scratchpad_mode(
        primitive_attr_t *attr, scratchpad_mode_t scratchpad_mode) {
    if (any_null(attr)) return invalid_arguments;

    return attr->set_scratchpad_mode(scratchpad_mode);
}

status_t dnnl_primitive_attr_set_scales_mask(
        primitive_attr_t *attr, int arg, int mask) {
    VCHECK_ATTR(attr, VERBOSE_NULL_ARG);
    VCHECK_ATTR(mask >= 0, VERBOSE_BAD_PARAM, "mask");
    VCHECK_ATTR(arg >= 0, VERBOSE_BAD_PARAM, "arg");
    return attr->scales_.set(arg, mask);
}

status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr, int arg,
        int mask, int group_ndims, const dims_t group_dims,
        data_type_t data_type) {
    return dnnl_primitive_attr_set_scales_v3(attr, arg, mask, group_ndims,
            group_dims, data_type, 0, quantization_mode::static_sazp);
}

status_t dnnl_primitive_attr_set_scales_v2(primitive_attr_t *attr, int arg,
        int mask, int group_ndims, const dims_t group_dims,
        data_type_t data_type, int is_on_host) {
    return dnnl_primitive_attr_set_scales_v3(attr, arg, mask, group_ndims,
            group_dims, data_type, is_on_host, quantization_mode::static_sazp);
}

status_t dnnl_primitive_attr_set_scales_v3(primitive_attr_t *attr, int arg,
        int mask, int group_ndims, const dims_t group_dims,
        data_type_t data_type, int is_on_host, quantization_mode_t qmode) {
    using namespace data_type;
    VCHECK_ATTR(attr, VERBOSE_NULL_ARG);
    VCHECK_ATTR(arg >= 0, VERBOSE_BAD_PARAM, "arg");
    VCHECK_ATTR(utils::one_of(qmode, quantization_mode::static_sazp,
                        quantization_mode::static_sbzp,
                        quantization_mode::dynamic_mx,
                        quantization_mode::dynamic_fp),
            VERBOSE_BAD_PARAM, "qmode");
    VCHECK_ATTR(utils::one_of(data_type, f32, bf16, f16, e8m0, f8_e4m3),
            VERBOSE_INVALID_DATATYPE, "scales");
    // A single low-precision scale is not expected as they work per group.
    VCHECK_ATTR(IMPLICATION(utils::one_of(data_type, e8m0, f8_e4m3), mask > 0),
            VERBOSE_INVALID_DATATYPE, "scales");
    VCHECK_ATTR(
            IMPLICATION(group_ndims, validate_dims(group_ndims, group_dims)),
            VERBOSE_BAD_PARAM, "group_dims");
    VCHECK_ATTR(utils::one_of(group_ndims, 0, 2), VERBOSE_BAD_PARAM,
            "group_ndims must be 0 or 2");
    if (is_on_host) { // only single value host-side scale is supported
        VCHECK_ATTR(mask == 0, VERBOSE_BAD_PARAM, "mask");
        VCHECK_ATTR(group_ndims == 0, VERBOSE_BAD_PARAM, "group_ndims");
    } else {
        VCHECK_ATTR(mask >= 0, VERBOSE_BAD_PARAM, "mask");
        VCHECK_ATTR(group_ndims >= 0, VERBOSE_BAD_PARAM, "group_ndims");
    }
    VCHECK_ATTR(IMPLICATION(group_ndims > 0, mask > 0), VERBOSE_BAD_PARAM,
            "mask incompatible with group_dims");
    return attr->scales_.set(
            arg, mask, data_type, group_ndims, group_dims, is_on_host, qmode);
}

status_t dnnl_primitive_attr_set_zero_points_mask(
        primitive_attr_t *attr, int arg, int mask) {
    VCHECK_ATTR(attr, VERBOSE_NULL_ARG);
    VCHECK_ATTR(mask >= 0, VERBOSE_BAD_PARAM, "mask");
    return attr->zero_points_.set(arg, mask);
}

status_t dnnl_primitive_attr_set_zero_points(dnnl_primitive_attr_t attr,
        int arg, int mask, int group_ndims, const dnnl_dims_t group_dims,
        dnnl_data_type_t data_type) {
    return dnnl_primitive_attr_set_zero_points_v2(
            attr, arg, mask, group_ndims, group_dims, data_type, 0);
}

status_t dnnl_primitive_attr_set_zero_points_v2(dnnl_primitive_attr_t attr,
        int arg, int mask, int group_ndims, const dnnl_dims_t group_dims,
        dnnl_data_type_t data_type, int is_on_host) {
    return dnnl_primitive_attr_set_zero_points_v3(attr, arg, mask, group_ndims,
            group_dims, data_type, is_on_host, quantization_mode::static_sazp);
}

status_t dnnl_primitive_attr_set_zero_points_v3(dnnl_primitive_attr_t attr,
        int arg, int mask, int group_ndims, const dnnl_dims_t group_dims,
        dnnl_data_type_t data_type, int is_on_host,
        dnnl_quantization_mode_t qmode) {
    using namespace data_type;

    VCHECK_ATTR(attr, VERBOSE_NULL_ARG);
    VCHECK_ATTR(arg >= 0, VERBOSE_BAD_PARAM, "arg");
    VCHECK_ATTR(utils::one_of(qmode, quantization_mode::static_sazp,
                        quantization_mode::static_sbzp),
            VERBOSE_BAD_PARAM, "qmode");
    // static_sbzp zero-points are scaled values of a floating-point type.
    const bool sbzp = qmode == quantization_mode::static_sbzp;
    VCHECK_ATTR(sbzp ? utils::one_of(data_type, bf16, f16)
                     : utils::one_of(data_type, s32, s8, u8, s4, u4, u2),
            VERBOSE_INVALID_DATATYPE, "zero points");
    VCHECK_ATTR(IMPLICATION(utils::one_of(data_type, s4, u4, u2), mask > 0),
            VERBOSE_BAD_PARAM, "mask with sub-byte data type");
    VCHECK_ATTR(IMPLICATION(!utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS,
                                    DNNL_ARG_WEIGHTS_1, DNNL_ARG_WEIGHTS_2),
                        data_type == s32 && group_ndims == 0),
            VERBOSE_INVALID_DATATYPE, "zero points");
    VCHECK_ATTR(
            IMPLICATION(group_ndims, validate_dims(group_ndims, group_dims)),
            VERBOSE_BAD_PARAM, "group_dims");
    VCHECK_ATTR(utils::one_of(group_ndims, 0, 2), VERBOSE_BAD_PARAM,
            "group_ndims must be 0 or 2");

    if (is_on_host) { // host-side zero point is only supported as single scalar value
        VCHECK_ATTR(mask == 0, VERBOSE_BAD_PARAM, "mask");
        VCHECK_ATTR(group_ndims == 0, VERBOSE_BAD_PARAM, "group_ndims");
        return attr->zero_points_.set(arg, 0, data_type, 0, {}, true, qmode);
    } else {
        VCHECK_ATTR(mask >= 0, VERBOSE_BAD_PARAM, "mask");
        VCHECK_ATTR(group_ndims >= 0, VERBOSE_BAD_PARAM, "group_ndims");
        VCHECK_ATTR(IMPLICATION(group_ndims > 0, mask > 0), VERBOSE_BAD_PARAM,
                "mask incompatible with group_dims");
        return attr->zero_points_.set(
                arg, mask, data_type, group_ndims, group_dims, false, qmode);
    }
}

status_t dnnl_primitive_attr_set_precomputed_reductions(
        dnnl_primitive_attr_t attr, int arg, int mask, int group_ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type) {
    using namespace data_type;
    VCHECK_ATTR(attr, VERBOSE_NULL_ARG);
    VCHECK_ATTR(mask >= 0, VERBOSE_BAD_PARAM, "mask");
    VCHECK_ATTR(utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS),
            VERBOSE_BAD_PARAM, "arg");
    VCHECK_ATTR(group_ndims >= 0, VERBOSE_BAD_PARAM, "group_ndims");
    VCHECK_ATTR(utils::one_of(data_type, s32), VERBOSE_INVALID_DATATYPE,
            "precomputed reductions");
    VCHECK_ATTR(
            IMPLICATION(group_ndims, validate_dims(group_ndims, group_dims)),
            VERBOSE_BAD_PARAM, "group_dims");
    VCHECK_ATTR(utils::one_of(group_ndims, 0, 2), VERBOSE_BAD_PARAM,
            "group_ndims must be 0 or 2");
    VCHECK_ATTR(IMPLICATION(group_ndims > 0, mask > 0), VERBOSE_BAD_PARAM,
            "mask incompatible with group_dims");

    return attr->precomputed_reductions_.set(
            arg, mask, data_type, group_ndims, group_dims);
}

status_t dnnl_primitive_attr_get_rounding(
        primitive_attr_t *attr, int arg, dnnl_rounding_mode_t *mode) {
    if (any_null(attr, mode)) return invalid_arguments;
    *mode = attr->rounding_mode_.get(arg);
    return success;
}

status_t dnnl_primitive_attr_set_rounding(
        primitive_attr_t *attr, int arg, dnnl_rounding_mode_t mode) {
    if (attr == nullptr) return invalid_arguments;
    return attr->rounding_mode_.set(arg, mode);
}

status_t dnnl_primitive_attr_get_post_ops(
        const primitive_attr_t *attr, const post_ops_t **post_ops) {
    if (any_null(attr, post_ops)) return invalid_arguments;

    *post_ops = &attr->post_ops_;
    return success;
}

status_t dnnl_primitive_attr_set_post_ops(
        primitive_attr_t *attr, const post_ops_t *post_ops) {
    if (any_null(attr, post_ops)) return invalid_arguments;

    return attr->set_post_ops(*post_ops);
}

status_t dnnl_post_ops_create(post_ops_t **post_ops) {
    if (post_ops == nullptr) return invalid_arguments;

    return safe_ptr_assign(*post_ops, new dnnl_post_ops);
}

status_t dnnl_post_ops_clone(
        post_ops_t **post_ops, const post_ops_t *existing_post_ops) {
    if (any_null(post_ops, existing_post_ops)) return invalid_arguments;

    auto new_post_ops = utils::make_unique<post_ops_t>(*existing_post_ops);
    if (!new_post_ops) return out_of_memory;

    return safe_ptr_assign(*post_ops, new_post_ops.release());
}

status_t dnnl_post_ops_destroy(post_ops_t *post_ops) {
    delete post_ops;

    return success;
}

int dnnl_post_ops_len(const post_ops_t *post_ops) {
    if (post_ops) return post_ops->len();

    return 0;
}

primitive_kind_t dnnl_post_ops_get_kind(const post_ops_t *post_ops, int index) {
    bool ok = post_ops && 0 <= index && index < post_ops->len();
    if (!ok) return primitive_kind::undefined;

    return post_ops->entry_[index].kind;
}

status_t dnnl_post_ops_append_sum(
        post_ops_t *post_ops, float scale, int32_t zero_point, data_type_t dt) {
    if (post_ops == nullptr) return invalid_arguments;
    if (post_ops->len() >= post_ops_t::post_ops_limit) return out_of_memory;

    return post_ops->append_sum(scale, zero_point, dt);
}

namespace {
status_t simple_get_params_check(
        const post_ops_t *post_ops, int index, primitive_kind_t kind) {
    VCHECK_ATTR(post_ops, VERBOSE_NULL_ARG);
    VCHECK_ATTR(index >= 0, VERBOSE_BAD_PARAM, "index");
    VCHECK_ATTR(index < post_ops->len(), VERBOSE_BAD_PARAM, "index");
    VCHECK_ATTR(
            post_ops->entry_[index].kind == kind, VERBOSE_BAD_PARAM, "kind");
    return status::success;
}
} // namespace

status_t dnnl_post_ops_get_params_sum(const post_ops_t *post_ops, int index,
        float *scale, int32_t *zero_point, data_type_t *dt) {
    CHECK(simple_get_params_check(post_ops, index, primitive_kind::sum));

    if (scale) *scale = post_ops->entry_[index].sum.scale;
    if (zero_point) *zero_point = post_ops->entry_[index].sum.zero_point;
    if (dt) *dt = post_ops->entry_[index].sum.dt;
    return success;
}

status_t dnnl_post_ops_append_eltwise(
        post_ops_t *post_ops, alg_kind_t kind, float alpha, float beta) {
    if (post_ops == nullptr) return invalid_arguments;
    if (post_ops->len() >= post_ops_t::post_ops_limit) return out_of_memory;

    return post_ops->append_eltwise(1.0f, kind, alpha, beta);
}

status_t dnnl_post_ops_get_params_eltwise(const post_ops_t *post_ops, int index,
        alg_kind_t *alg, float *alpha, float *beta) {
    CHECK(simple_get_params_check(post_ops, index, primitive_kind::eltwise));

    const auto &e = post_ops->entry_[index].eltwise;
    if (alg) *alg = e.alg;
    if (alpha) *alpha = e.alpha;
    if (beta) *beta = e.beta;

    return success;
}

status_t dnnl_post_ops_append_dw(post_ops_t *post_ops, data_type_t wei_dt,
        data_type_t bias_dt, data_type_t dst_dt, dim_t kernel_size,
        dim_t stride_size, dim_t padding_l_size) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_dw(
            wei_dt, bias_dt, dst_dt, kernel_size, stride_size, padding_l_size);
}

status_t dnnl_post_ops_get_params_dw(const post_ops_t *post_ops, int index,
        data_type_t *wei_dt, data_type_t *bias_dt, data_type_t *dst_dt,
        dim_t *kernel, dim_t *stride, dim_t *padding) {
    CHECK(simple_get_params_check(
            post_ops, index, primitive_kind::convolution));

    const auto &d = post_ops->entry_[index].depthwise_conv;
    if (wei_dt) *wei_dt = d.wei_dt;
    if (bias_dt) *bias_dt = d.bias_dt;
    if (dst_dt) *dst_dt = d.dst_dt;
    if (kernel) *kernel = d.kernel;
    if (stride) *stride = d.stride;
    if (padding) *padding = d.padding;

    return success;
}

status_t dnnl_post_ops_append_binary(post_ops_t *post_ops, alg_kind_t alg_kind,
        const memory_desc_t *user_src1_desc) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_binary(alg_kind, user_src1_desc);
}

status_t dnnl_post_ops_append_binary_v2(post_ops_t *post_ops,
        alg_kind_t alg_kind, const memory_desc_t *user_src1_desc,
        const memory_desc_t *user_src2_desc) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_binary(alg_kind, user_src1_desc, user_src2_desc);
}

status_t dnnl_post_ops_get_params_binary(const post_ops_t *post_ops, int index,
        alg_kind_t *alg_kind, const memory_desc_t **user_src1_desc) {
    CHECK(simple_get_params_check(post_ops, index, primitive_kind::binary));

    const auto &b = post_ops->entry_[index].binary;
    if (alg_kind) *alg_kind = b.alg;
    if (user_src1_desc) *user_src1_desc = &b.user_src1_desc;

    return success;
}

status_t dnnl_post_ops_get_params_binary_v2(const post_ops_t *post_ops,
        int index, alg_kind_t *alg_kind, const memory_desc_t **user_src1_desc,
        const memory_desc_t **user_src2_desc) {
    CHECK(simple_get_params_check(post_ops, index, primitive_kind::binary));

    const auto &b = post_ops->entry_[index].binary;
    if (alg_kind) *alg_kind = b.alg;
    if (user_src1_desc) *user_src1_desc = &b.user_src1_desc;
    if (user_src2_desc) *user_src2_desc = &b.user_src2_desc;

    return success;
}

status_t dnnl_post_ops_append_prelu(post_ops_t *post_ops, int mask) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_prelu(mask);
}

status_t dnnl_post_ops_get_params_prelu(
        const post_ops_t *post_ops, int index, int *mask) {
    CHECK(simple_get_params_check(post_ops, index, primitive_kind::prelu));

    const auto &prelu_entry = post_ops->entry_[index].prelu;
    if (mask) *mask = prelu_entry.mask;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_data_qparams(
        primitive_attr_t *attr, const float scale, const float shift) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_data_qparams_.set(scale, shift);
}

status_t dnnl_primitive_attr_get_rnn_data_qparams(
        const primitive_attr_t *attr, float *scale, float *shift) {
    if (attr == nullptr) return invalid_arguments;

    const auto qparams = attr->rnn_data_qparams_;
    if (scale) *scale = qparams.scale_;
    if (shift) *shift = qparams.shift_;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_weights_qparams(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->rnn_weights_qparams_.set(count, mask, scales);
}

status_t dnnl_primitive_attr_get_rnn_weights_qparams(
        const primitive_attr_t *attr, dim_t *count, int *mask,
        const float **scales) {
    if (attr == nullptr) return invalid_arguments;

    const auto &qparams = attr->rnn_weights_qparams_;
    if (count) *count = qparams.count_;
    if (mask) *mask = qparams.mask_;
    if (scales) *scales = qparams.scales_;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_weights_projection_qparams(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->rnn_weights_projection_qparams_.set(count, mask, scales);
}

status_t dnnl_primitive_attr_get_rnn_weights_projection_qparams(
        const primitive_attr_t *attr, dim_t *count, int *mask,
        const float **scales) {
    if (attr == nullptr) return invalid_arguments;

    const auto &qparams = attr->rnn_weights_projection_qparams_;
    if (count) *count = qparams.count_;
    if (mask) *mask = qparams.mask_;
    if (scales) *scales = qparams.scales_;

    return success;
}

status_t DNNL_API dnnl_primitive_attr_set_rnn_tparams(
        dnnl_primitive_attr_t attr, bool mode, dim_t ngates,
        const float *scales, float cscale) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_tparams_.set(mode, ngates, scales, cscale);
}
