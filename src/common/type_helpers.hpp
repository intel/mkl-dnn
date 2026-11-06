/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef COMMON_TYPE_HELPERS_HPP
#define COMMON_TYPE_HELPERS_HPP

#include <algorithm>
#include <assert.h>
#include <math.h>

#include "oneapi/dnnl/dnnl.h"

#include "bit_cast.hpp"
#include "c_types_map.hpp"
#include "dnnl_traits.hpp"
#include "gemm_types.hpp"
#include "memory_desc.hpp"
#include "nstl.hpp"
#include "opdesc.hpp"
#include "sdpa_types.hpp"
#include "utils.hpp"
#include "verbose.hpp"

namespace dnnl {
namespace impl {

// Global zero memory descriptor. Mostly used for queries to return
extern memory_desc_t DNNL_API glob_zero_md;

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(base_type *&lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs = rhs;
    return status::success;
}

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(std::unique_ptr<base_type> &lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs.reset(rhs);
    return status::success;
}

template <typename base_type, typename base_type_deleter, typename derived_type>
status_t safe_ptr_assign(
        std::unique_ptr<base_type, base_type_deleter> &lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs.reset(rhs);
    return status::success;
}

template <typename T, typename U>
struct is_subset { // NOLINT(readability-identifier-naming)
    static constexpr bool value = false;
};
template <typename T>
struct is_subset<T, T> {
    static constexpr bool value = true;
};
template <typename T>
struct is_subset<T,
        typename utils::enable_if<nstl::is_integral<T>::value, float>::type> {
    static constexpr bool value = true;
};
#define ISSPEC(t1, t2) \
    template <> \
    struct is_subset<t1, t2> { \
        static constexpr bool value = true; \
    }
ISSPEC(int16_t, int32_t);
ISSPEC(int8_t, int32_t);
ISSPEC(uint8_t, int32_t);
ISSPEC(int8_t, int16_t);
ISSPEC(uint8_t, int16_t);
#undef ISSPEC

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs);

namespace types {

inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch ((int)data_type) {
        case f4_e2m1: return sizeof(prec_traits_t<f4_e2m1>::type);
        case e8m0: return sizeof(prec_traits_t<e8m0>::type);
        case f8_e5m2: return sizeof(prec_traits_t<f8_e5m2>::type);
        case f8_e4m3: return sizeof(prec_traits_t<f8_e4m3>::type);
        case f16: return sizeof(prec_traits_t<f16>::type);
        case bf16: return sizeof(prec_traits_t<bf16>::type);
        case tf32: // the tf32 type is an f32
        case f32: return sizeof(prec_traits_t<f32>::type);
        case f64: return sizeof(prec_traits_t<f64>::type);
        case s64: return sizeof(prec_traits_t<s64>::type);
        case s32: return sizeof(prec_traits_t<s32>::type);
        case s8: return sizeof(prec_traits_t<s8>::type);
        case u8: return sizeof(prec_traits_t<u8>::type);
        case s4: return sizeof(prec_traits_t<s4>::type);
        case u4: return sizeof(prec_traits_t<u4>::type);
        case u2: return sizeof(prec_traits_t<u2>::type);
        case u3: return sizeof(prec_traits_t<u3>::type);
        case boolean: return sizeof(prec_traits_t<boolean>::type);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return (size_t)-1; /* not supposed to be reachable */
}

inline int data_type_bits(data_type_t data_type) {
    using namespace data_type;

    switch ((int)data_type) {
        case s64:
        case f64: return 64;
        case tf32:
        case s32:
        case f32: return 32;
        case bf16:
        case f16: return 16;
        case e8m0:
        case f8_e5m2:
        case f8_e4m3:
        case s8:
        case u8: return 8;
        case f4_e2m1:
        case s4:
        case u4: return 4;
        case u3: return 3;
        case u2: return 2;
        case boolean: return 1;
        default: assert(!"unsupported data type"); return 0;
    }
}

inline size_t elements_to_bytes(data_type_t data_type, size_t count) {
    using namespace data_type;
    switch ((int)data_type) {
        // sub-bytes data types handling
        case f4_e2m1:
        case s4:
        case u4:
        case u2:
        case u3: {
            constexpr size_t byte = 8;
            const size_t dt_bits = data_type_bits(data_type);

            // number of bits holding whole number of values,
            // that ends on byte boundary (e.g., 8 for u2/u4 and 24 for u3)
            size_t storage_unit_bits = byte;
            while (storage_unit_bits % dt_bits != 0)
                storage_unit_bits += byte;

            return utils::rnd_up(count * dt_bits, storage_unit_bits) / 8;
        }
        // Note: boolean is 1 bit in data_type_bits(),
        // but is stored as a whole byte, so goes to default
        default: return data_type_size(data_type) * count;
    }
}

inline size_t bytes_to_elements(data_type_t data_type, size_t bytes) {
    using namespace data_type;
    switch ((int)data_type) {
        case f4_e2m1:
        case s4:
        case u4:
        case u2:
        case u3:
            assert(bytes * 8 % data_type_bits(data_type) == 0);
            return bytes * 8 / data_type_bits(data_type);
        default: return utils::div_up(bytes, data_type_size(data_type));
    }
}

template <typename T>
inline T min_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::min())
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(f64);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

template <typename T>
inline T max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::max())
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        case f64: return nstl::numeric_limits<T>::max();
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

// This is a hack to comply with a big comment below.
template <>
inline float max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<float>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::max())
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        // INT_MAX is not representable in float. The nearest float to it is
        // INT_MAX + 1 = 2^31 (0x4f000000). Regular conversion instructions such
        // as `cvtps2dq` or `cvtss2si` will convert this number to INT_MIN
        // making the result negative. We on purpose choose the previous float
        // number (0x4effffff) to return leaving the output close to INT_MAX but
        // still positive. In addition, we adjust validation of this approach.
        // The main concern against `real` saturation is performance, which
        // likely to drop (but it was not proved). The only drawback of current
        // approach is saturating on some integer values before it should happen
        // in the reality.
        case s32: return 2147483520.f;
        case f64: return nstl::numeric_limits<float>::max();
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return 0.f; /* not supposed to be reachable */
#undef CASE
}

template <typename T>
inline T lowest_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::lowest())
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        case f64: return nstl::numeric_limits<T>::lowest();
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

template <typename T>
inline T digits(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::digits)
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(f64);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

template <typename T>
inline T epsilon_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>( \
                nstl::numeric_limits<prec_traits_t<x>::type>::epsilon())
    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(f64);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

inline float round_to_dt(data_type_t data_type, float val) {
    using namespace data_type;

#define CASE(x) \
    case x: \
        return static_cast<float>( \
                static_cast<typename prec_traits_t<x>::type>(val))

    switch (data_type) {
        CASE(f4_e2m1);
        CASE(e8m0);
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(f64);
        CASE(s64);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        CASE(u2);
        CASE(u3);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }

    return 0.0f; /* not supposed to be reachable */
#undef CASE
}

inline format_kind_t format_tag_to_kind(format_tag_t tag) {
    switch (tag) {
        case format_tag::undef: return format_kind::undef;
        case format_tag::any: return format_kind::any;
        case format_tag::last: return format_kind::undef;
        default: return format_kind::blocked;
    }

    assert(!"unreachable");
    return format_kind::undef;
}

inline bool memory_extra_desc_is_equal(
        const memory_extra_desc_t &lhs, const memory_extra_desc_t &rhs) {
    using namespace memory_extra_flags;
    return lhs.flags == rhs.flags
            && IMPLICATION(lhs.flags & compensation_conv_s8s8,
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION(lhs.flags & rnn_u8s8_compensation,
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION(lhs.flags & scale_adjust,
                    lhs.scale_adjust == rhs.scale_adjust)
            && IMPLICATION(lhs.flags & compensation_conv_asymmetric_src,
                    lhs.asymm_compensation_mask == rhs.asymm_compensation_mask)
            && IMPLICATION(lhs.flags & compensation_gpu_conv_asymmetric_src,
                    (lhs.dst_size == rhs.dst_size)
                            && utils::array_cmp(lhs.idhw, rhs.idhw, 3)
                            && utils::array_cmp(lhs.odhw, rhs.odhw, 3)
                            && utils::array_cmp(lhs.pdhw, rhs.pdhw, 3)
                            && utils::array_cmp(lhs.ddhw, rhs.ddhw, 3));
}

inline bool blocking_desc_is_equal(const memory_desc_t &lhs_md,
        const memory_desc_t &rhs_md, bool ignore_strides = false) {
    using dnnl::impl::utils::array_cmp;

    auto is_sparse_packed_desc = [](const memory_desc_t &md) {
        return md.format_kind == format_kind::sparse
                && md.format_desc.sparse_desc.encoding
                == sparse_encoding::packed;
    };

    const bool lhs_is_sparse_packed_desc = is_sparse_packed_desc(lhs_md);
    const bool rhs_is_sparse_packed_desc = is_sparse_packed_desc(rhs_md);

    if (lhs_md.format_kind != format_kind::blocked
            && !lhs_is_sparse_packed_desc)
        return false;
    if (rhs_md.format_kind != format_kind::blocked
            && !rhs_is_sparse_packed_desc)
        return false;

    const auto &lhs = lhs_md.format_kind == format_kind::sparse
            ? lhs_md.format_desc.sparse_desc.packed_desc
            : lhs_md.format_desc.blocking;
    const auto &rhs = rhs_md.format_kind == format_kind::sparse
            ? rhs_md.format_desc.sparse_desc.packed_desc
            : rhs_md.format_desc.blocking;
    bool equal = lhs.inner_nblks == rhs.inner_nblks
            && array_cmp(lhs.inner_blks, rhs.inner_blks, lhs.inner_nblks)
            && array_cmp(lhs.inner_idxs, rhs.inner_idxs, lhs.inner_nblks);

    // Check the strides.
    // Note: for dimensions of size `1` the stride doesn't really matter
    if (ignore_strides) return equal;

    for (int d = 0; d < lhs_md.ndims; ++d) {
        if (lhs_md.dims[d] == 1 && lhs_md.padded_dims[d] == 1) continue;
        equal = equal && lhs.strides[d] == rhs.strides[d];
    }

    return equal;
}

inline bool wino_desc_is_equal(const wino_desc_t &lhs, const wino_desc_t &rhs) {
    return lhs.wino_format == rhs.wino_format && lhs.alpha == rhs.alpha
            && lhs.ic == rhs.ic && lhs.oc == rhs.oc
            && lhs.ic_block == rhs.ic_block && lhs.oc_block == rhs.oc_block
            && lhs.ic2_block == rhs.ic2_block && lhs.oc2_block == rhs.oc2_block
            && lhs.r == rhs.r;
}
inline bool cublaslt_blocked_desc_is_equal(const cublaslt_blocked_desc_t &lhs,
        const cublaslt_blocked_desc_t &rhs) {
    return lhs.cublaslt_format == rhs.cublaslt_format && lhs.size == rhs.size;
}

inline bool rnn_packed_desc_is_equal(
        const rnn_packed_desc_t &lhs, const rnn_packed_desc_t &rhs) {
    bool ok = true && lhs.format == rhs.format && lhs.ldb == rhs.ldb
            && lhs.n_parts == rhs.n_parts
            && lhs.offset_compensation == rhs.offset_compensation
            && lhs.size == rhs.size && lhs.n == rhs.n;
    if (!ok) return false;

    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.parts[i] == rhs.parts[i];
    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.part_pack_size[i] == rhs.part_pack_size[i];
    return ok;
}

inline bool sparse_desc_is_equal(
        const sparse_desc_t &lhs, const sparse_desc_t &rhs) {
    bool ok = lhs.encoding == rhs.encoding && lhs.nnz == rhs.nnz;
    if (!ok) return false;

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    if (lhs.encoding == sparse_encoding::grouped) {
        ok = ok && lhs.grouped_desc.group_count == rhs.grouped_desc.group_count
                && lhs.grouped_desc.variable_dim_idx
                        == rhs.grouped_desc.variable_dim_idx;
        if (!ok) return false;
    }
#endif

    for (int i = 0; i < sparse_desc_t::max_metadata_types; i++)
        ok = ok && lhs.metadata_types[i] == rhs.metadata_types[i];

    return ok;
}

inline memory_desc_t zero_md() {
    auto zero = memory_desc_t();
    return zero;
}

inline bool is_zero_md(const memory_desc_t *md) {
    return md == nullptr || *md == zero_md();
}

inline data_type_t default_accum_data_type(
        data_type_t src_dt, data_type_t dst_dt, bool strict = true) {
    using namespace utils;
    using namespace data_type;

    // we allow to use f32 accumulation type only when the
    // accumulation chain is small. Otherwise, strict should be set to
    // true
    if (one_of(src_dt, s8, u8, u4, s4, u2, u3) && (dst_dt != f32 || strict))
        return s32;

    if (one_of(f4_e2m1, src_dt, dst_dt)) return f32;
    if (one_of(f8_e5m2, src_dt, dst_dt)) return f32;
    if (one_of(f8_e4m3, src_dt, dst_dt)) return f32;
    if (one_of(f16, src_dt, dst_dt)) return f32;
    if (one_of(bf16, src_dt, dst_dt)) return f32;
    if (one_of(f32, src_dt, dst_dt)) return f32;
    if (one_of(f64, src_dt, dst_dt)) return f64;
    if (one_of(s32, src_dt, dst_dt)) return s32;

    if (one_of(s8, src_dt, dst_dt) || one_of(u8, src_dt, dst_dt)
            || one_of(s4, src_dt, dst_dt) || one_of(u4, src_dt, dst_dt)
            || one_of(u2, src_dt, dst_dt) || one_of(u3, src_dt, dst_dt))
        return s32;

    return data_type::undef;
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt, prop_kind_t prop_kind) {
    using namespace utils;
    using namespace data_type;
    using namespace prop_kind;

    /* prop_kind doesn't matter */
    if (everyone_is(f32, src_dt, wei_dt)) return f32;
    if (everyone_is(f64, src_dt, wei_dt)) return f64;

    if (one_of(prop_kind, forward_training, forward_inference)) {
        if (one_of(src_dt, u8, s8) && one_of(wei_dt, u8, s8, s4, u4, u2, u3))
            return s32;
        if (one_of(f16, src_dt, wei_dt)) return f32;
        // weights decompression
        if (one_of(src_dt, bf16, f32) && one_of(wei_dt, u8, s8, s4, u4, u2, u3))
            return f32;
    } else if (prop_kind == backward_data) {
        if (one_of(src_dt, f32, s32, s8, u8) && wei_dt == s8
                && one_of(dst_dt, s8, u8, s32))
            return s32;
        if (one_of(f16, dst_dt, wei_dt)) return f32;
        if (everyone_is(f32, dst_dt, wei_dt) && one_of(src_dt, s8, u8))
            return f32;
    }

    if (one_of(f4_e2m1, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(f8_e5m2, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(f8_e4m3, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(bf16, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(f16, src_dt, wei_dt, dst_dt)) return f32;

    return data_type::undef;
}

inline bool is_integral_dt(data_type_t dt) {
    using namespace data_type;
    switch ((int)dt) {
        case s64:
        case s32:
        case s8:
        case u8:
        case u4:
        case s4:
        case u2:
        case u3: return true;
        case f4_e2m1:
        case e8m0:
        case f8_e5m2:
        case f8_e4m3:
        case f16:
        case bf16:
        case tf32:
        case f32:
        case f64:
        case boolean: return false;
        case data_type::undef:
        default: assert(!"unknown data_type"); return false;
    }
}

template <typename data_t>
inline void cvt_from_float(data_t *out, const float *inp, size_t nelems)
        = delete;

template <typename data_t>
inline void cvt_to_float(float *out, const data_t *inp, size_t nelems) = delete;

template <>
inline void cvt_from_float<float>(float *out, const float *inp, size_t nelems) {
    // This operation should be avoided as it does nothing useful
    assert(!"unexpected");
    for (size_t i = 0; i < nelems; i++)
        out[i] = inp[i];
}

template <>
inline void cvt_to_float<float>(float *out, const float *inp, size_t nelems) {
    // This operation should be avoided as it does nothing useful
    assert(!"unexpected");
    for (size_t i = 0; i < nelems; i++)
        out[i] = inp[i];
}

template <>
inline void cvt_from_float<bfloat16_t>(
        bfloat16_t *out, const float *inp, size_t nelems) {
    cvt_float_to_bfloat16(out, inp, nelems);
}

template <>
inline void cvt_to_float<bfloat16_t>(
        float *out, const bfloat16_t *inp, size_t nelems) {
    cvt_bfloat16_to_float(out, inp, nelems);
}

template <>
inline void cvt_from_float<float16_t>(
        float16_t *out, const float *inp, size_t nelems) {
    cvt_float_to_float16(out, inp, nelems);
}

template <>
inline void cvt_to_float<float16_t>(
        float *out, const float16_t *inp, size_t nelems) {
    cvt_float16_to_float(out, inp, nelems);
}

template <>
inline void cvt_to_float<float8_e5m2_t>(
        float *out, const float8_e5m2_t *inp, size_t nelems) {
    cvt_f8_e5m2_to_float(out, inp, nelems);
}

template <>
inline void cvt_from_float<float8_e5m2_t>(
        float8_e5m2_t *out, const float *inp, size_t nelems) {
    cvt_float_to_f8_e5m2(out, inp, nelems);
}

template <>
inline void cvt_to_float<float8_e4m3_t>(
        float *out, const float8_e4m3_t *inp, size_t nelems) {
    cvt_f8_e4m3_to_float(out, inp, nelems);
}

template <>
inline void cvt_from_float<float8_e4m3_t>(
        float8_e4m3_t *out, const float *inp, size_t nelems) {
    cvt_float_to_f8_e4m3(out, inp, nelems);
}

inline void cvt_from_float(
        data_type_t dt, void *out, const float *inp, size_t nelems) {
    switch (dt) {
        case data_type::bf16:
            cvt_from_float((bfloat16_t *)out, inp, nelems);
            break;
        case data_type::f16:
            cvt_from_float((float16_t *)out, inp, nelems);
            break;
        case data_type::f8_e5m2:
            cvt_from_float((float8_e5m2_t *)out, inp, nelems);
            break;
        case data_type::f8_e4m3:
            cvt_from_float((float8_e4m3_t *)out, inp, nelems);
            break;
        default: assert(!"unimplemented");
    }
}

inline void cvt_to_float(
        data_type_t dt, float *out, const void *inp, size_t nelems) {
    switch (dt) {
        case data_type::bf16:
            cvt_to_float(out, (const bfloat16_t *)inp, nelems);
            break;
        case data_type::f16:
            cvt_to_float(out, (const float16_t *)inp, nelems);
            break;
        case data_type::f8_e5m2:
            cvt_to_float(out, (const float8_e5m2_t *)inp, nelems);
            break;
        case data_type::f8_e4m3:
            cvt_to_float(out, (const float8_e4m3_t *)inp, nelems);
            break;
        default: assert(!"unimplemented");
    }
}

} // namespace types

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    using namespace dnnl::impl::utils;
    // quick path for zero_mds
    if (utils::everyone_is(0, lhs.ndims, rhs.ndims)) return true;

    bool base_equal = true && lhs.ndims == rhs.ndims
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && lhs.data_type == rhs.data_type
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0 && lhs.format_kind == rhs.format_kind;
    if (!base_equal) return false;
    if (!types::memory_extra_desc_is_equal(lhs.extra, rhs.extra)) return false;
    if (lhs.format_kind == format_kind::blocked)
        return types::blocking_desc_is_equal(lhs, rhs);
    else if (lhs.format_kind == format_kind::wino)
        return types::wino_desc_is_equal(
                lhs.format_desc.wino_desc, rhs.format_desc.wino_desc);
    else if (lhs.format_kind == format_kind::rnn_packed)
        return types::rnn_packed_desc_is_equal(lhs.format_desc.rnn_packed_desc,
                rhs.format_desc.rnn_packed_desc);
    else if (lhs.format_kind == format_kind::sparse)
        return types::sparse_desc_is_equal(
                lhs.format_desc.sparse_desc, rhs.format_desc.sparse_desc);
    else if (lhs.format_kind == format_kind::zen_packed)
        return lhs.format_desc.zen_packed_desc.size
                == rhs.format_desc.zen_packed_desc.size
                && lhs.format_desc.zen_packed_desc.per_slice_size
                == rhs.format_desc.zen_packed_desc.per_slice_size
                && lhs.format_desc.zen_packed_desc.gemm_src_dt
                == rhs.format_desc.zen_packed_desc.gemm_src_dt
                && lhs.format_desc.zen_packed_desc.weights_transposed
                == rhs.format_desc.zen_packed_desc.weights_transposed;
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

// Comparison operators for descriptors
#define COMPARE_DESC_MEMBERS(m) lhs.m == rhs.m
#define COMPARE_DESC_ARRAY_MEMBERS(m, s) utils::array_cmp(lhs.m, rhs.m, s)
#define DEREF_AND_COMPARE_DESC_MEMBERS(m) *lhs.m == *rhs.m
#define COMPARE_FLOAT_DESC_MEMBERS(m) utils::equal_with_nan(lhs.m, rhs.m)
#define COMPARE_FLOAT_DESC_ARRAY_MEMBERS(m, s) \
    !std::memcmp(lhs.m, rhs.m, sizeof(float) * (s))

// clang-format off
inline bool operator==(const batch_normalization_desc_t &lhs,
        const batch_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(batch_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
    return ret;
}

inline bool operator==(const binary_desc_t &lhs, const binary_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc[0])
            && COMPARE_DESC_MEMBERS(src_desc[1])
            && COMPARE_DESC_MEMBERS(dst_desc);

    // For ternary operators like select, the additional input for conditional
    // select must also be compared
    if(utils::one_of(alg_kind::binary_select, lhs.alg_kind, rhs.alg_kind))
        ret = ret && COMPARE_DESC_MEMBERS(src_desc[2]);

    return ret;
}

inline bool operator==(const concat_desc_t &lhs, const concat_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && DEREF_AND_COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n)
            && COMPARE_DESC_MEMBERS(concat_dimension);

    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = *lhs.src_mds[i] == *rhs.src_mds[i];
        if (!ret) break;
    }
    return ret;
}

// This function can only be used to compare the opdescs in the primitive cache.
// For comparing the opdescs outside the primitive cache please use the regular
// comparison operator (==).
inline bool compare_conv_opdesc(const convolution_desc_t &lhs, const convolution_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilates, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type)
            && COMPARE_DESC_MEMBERS(use_inversion);

      // The `alg_kind` can be `auto` only if this function is called for the
      // primitive descriptor cache scenario. In this case, we ignore `alg_kind`
      // and rely on `pd_iterator_offset` to fetch the first suitable
      // implementation.
      //
      // Background: when a convolution primitive descriptor is created for
      // the algorithm `auto` we overwrite `alg_kind` field in `op_desc` when
      // store it in the primitive descriptor. Because of that, the `op_desc`
      // stored in the primitive descriptor is different from the one user
      // passed to oneDNN API. Because of the difference the requested
      // primitive descriptor cannot be found in the cache if we compare
      // `alg_kind`.
      if (!utils::one_of(alg_kind::convolution_auto, lhs.alg_kind, rhs.alg_kind))
          ret = ret && COMPARE_DESC_MEMBERS(alg_kind);

    return ret;
}

inline bool operator==(
        const convolution_desc_t &lhs, const convolution_desc_t &rhs) {
        if (!(COMPARE_DESC_MEMBERS(alg_kind))) return false;
        return compare_conv_opdesc(lhs, rhs);
}

inline bool operator==(const eltwise_desc_t &lhs, const eltwise_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const gemm_desc_t &lhs, const gemm_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(a_desc)
            && COMPARE_DESC_MEMBERS(b_desc)
            && COMPARE_DESC_MEMBERS(c_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(acc_type)
            && COMPARE_DESC_MEMBERS(sum_ab)
            && COMPARE_DESC_MEMBERS(sum_ab_type);
    return ret;
}

inline bool operator==(
        const group_normalization_desc_t &lhs, const group_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_DESC_MEMBERS(groups)
            && COMPARE_FLOAT_DESC_MEMBERS(group_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
     return ret;
}

inline bool operator==(
        const inner_product_desc_t &lhs, const inner_product_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(
        const layer_normalization_desc_t &lhs, const layer_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(layer_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
     return ret;
}

inline bool operator==(const lrn_desc_t &lhs, const lrn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(local_size)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_beta)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_k);
    return ret;
}

inline bool operator==(const matmul_desc_t &lhs, const matmul_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(reduce_desc)
            && COMPARE_DESC_MEMBERS(reduce_kind)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(
        const pooling_desc_t &lhs, const pooling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(kernel, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilation, DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
     return ret;
}

inline bool operator==(const prelu_desc_t &lhs, const prelu_desc_t &rhs) {
    const bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc);
    return ret;
}

inline bool operator==(
        const reduction_desc_t &lhs, const reduction_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(p)
            && COMPARE_FLOAT_DESC_MEMBERS(eps);
    return ret;
}

inline bool operator==(const reorder_desc_t &lhs, const reorder_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(src_engine_kind)
            && COMPARE_DESC_MEMBERS(dst_engine_kind)
            && COMPARE_DESC_MEMBERS(is_cross_engine);
    return ret;
}

inline bool operator==(
        const resampling_desc_t &lhs, const resampling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_FLOAT_DESC_ARRAY_MEMBERS(factors, DNNL_MAX_NDIMS);
    return ret;
}

inline bool operator==(const rnn_desc_t &lhs, const rnn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(cell_kind)
            && COMPARE_DESC_MEMBERS(direction)
            && COMPARE_DESC_MEMBERS(src_layer_desc)
            && COMPARE_DESC_MEMBERS(src_iter_desc)
            && COMPARE_DESC_MEMBERS(src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_layer_desc)
            && COMPARE_DESC_MEMBERS(weights_iter_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_layer_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(weights_projection_desc)
            && COMPARE_DESC_MEMBERS(diff_src_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_projection_desc)
            && COMPARE_DESC_MEMBERS(flags)
            && COMPARE_DESC_MEMBERS(activation_kind)
            && COMPARE_FLOAT_DESC_MEMBERS(alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const shuffle_desc_t &lhs, const shuffle_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(axis)
            && COMPARE_DESC_MEMBERS(group_size);
    return ret;
}

inline bool operator==(
        const softmax_desc_t &lhs, const softmax_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(softmax_axis);
     return ret;
}

inline bool operator==(const sum_desc_t &lhs, const sum_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && DEREF_AND_COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n);
    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = *lhs.src_mds[i] == *rhs.src_mds[i];
        if (!ret) break;
    }
    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = ret && COMPARE_FLOAT_DESC_MEMBERS(scales[i]);
        if (!ret) break;
    }

    return ret;
}

inline bool operator==(const zero_pad_desc_t &lhs, const zero_pad_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind);
    return ret;
}

inline bool operator==(const sdpa_desc_t &lhs, const sdpa_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(q_desc)
            && COMPARE_DESC_MEMBERS(k_desc)
            && COMPARE_DESC_MEMBERS(v_desc)
            && COMPARE_DESC_MEMBERS(kq_scales)
            && COMPARE_DESC_MEMBERS(kq_zero_points)
            && COMPARE_DESC_MEMBERS(vs_scales)
            && COMPARE_DESC_MEMBERS(vs_zero_points)
            && COMPARE_DESC_MEMBERS(dS_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(stats_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(diff_q_desc)
            && COMPARE_DESC_MEMBERS(diff_k_desc)
            && COMPARE_DESC_MEMBERS(diff_v_desc)
            && COMPARE_DESC_MEMBERS(attn_mask_desc)
            && COMPARE_DESC_MEMBERS(scale_desc)
            && COMPARE_DESC_MEMBERS(fill_desc)
            && COMPARE_DESC_MEMBERS(kq_acc_dt)
            && COMPARE_DESC_MEMBERS(vs_acc_dt)
            && COMPARE_DESC_MEMBERS(invert_scale)
            && COMPARE_DESC_MEMBERS(invert_select)
            && COMPARE_DESC_MEMBERS(kv_head_number)
            && COMPARE_DESC_MEMBERS(mask_type)
            && COMPARE_DESC_MEMBERS(softmax_alg);
    return ret;
}

// clang-format on

#undef COMPARE_DESC_MEMBERS
#undef COMPARE_DESC_ARRAY_MEMBERS
#undef DEREF_AND_COMPARE_DESC_MEMBERS
#undef COMPARE_FLOAT_DESC_MEMBERS
#undef COMPARE_FLOAT_DESC_ARRAY_MEMBERS

inline bool is_dense_format_kind(
        const std::vector<const memory_desc_t *> &mds) {
    for (const auto *md : mds)
        if (md->format_kind == format_kind::sparse) return false;
    return true;
}

inline memory_desc_t cvt_blocked2sparse_packed(
        const memory_desc_t &blocked_md, dim_t nnz) {
    if (blocked_md.format_kind != format_kind::blocked) return glob_zero_md;

    auto sparse_packed_md = blocked_md;
    sparse_packed_md.format_kind = format_kind::sparse;
    sparse_packed_md.format_desc.sparse_desc.encoding = sparse_encoding::packed;
    sparse_packed_md.format_desc.sparse_desc.nnz = nnz;
    sparse_packed_md.format_desc.sparse_desc.packed_desc
            = blocked_md.format_desc.blocking;
    return sparse_packed_md;
}

inline memory_desc_t cvt_sparse_packed2blocked(
        const memory_desc_t &sparse_packed_md) {
    if (sparse_packed_md.format_kind != format_kind::sparse
            || sparse_packed_md.format_desc.sparse_desc.encoding
                    != sparse_encoding::packed)
        return glob_zero_md;

    const blocking_desc_t &blk_desc
            = sparse_packed_md.format_desc.sparse_desc.packed_desc;
    auto blocked_md = sparse_packed_md;
    blocked_md.format_desc.blocking = blk_desc;
    blocked_md.format_kind = format_kind::blocked;
    return blocked_md;
}

inline status_t memory_desc_init_by_strides(
        memory_desc_t &md, const dims_t strides) {
    return memory_desc_init_by_strides(
            md, md.ndims, md.dims, md.data_type, strides);
}

inline status_t memory_desc_init_by_tag(
        memory_desc_t &md, format_tag_t tag, const dims_t strides = nullptr) {

    const bool is_sparse = md.format_kind == format_kind::sparse;
    auto md_tmp = memory_desc_t();

    CHECK(memory_desc_init_by_tag(
            md_tmp, md.ndims, md.dims, md.data_type, tag));

    CHECK(memory_desc_strides_check(md_tmp, strides));

    if (is_sparse) {
        if (md.format_desc.sparse_desc.encoding != sparse_encoding::packed
                || md.offset0 != 0)
            return status::invalid_arguments;
        md = cvt_blocked2sparse_packed(md_tmp, md.format_desc.sparse_desc.nnz);
    } else {
        md = md_tmp;
    }

    if (strides == nullptr) return status::success;

    for (int d = 0; d < md.ndims; ++d) {
        if (is_sparse)
            md.format_desc.sparse_desc.packed_desc.strides[d] = strides[d];
        else
            md.format_desc.blocking.strides[d] = strides[d];
    }

    return status::success;
}

/** inits memory descriptor based on logical dimensions kept in @p md, and the
 * blocking structure @p blk.
 *
 * @note blk.strides represent the order only (from smaller to bigger)
 *
 * TODO: move md related functions to one single place
 */
inline status_t memory_desc_init_by_blocking_desc(
        memory_desc_t &md, const blocking_desc_t &blk) {
    dims_t blocks = {0};
    utils::array_set(blocks, 1, md.ndims);
    dim_t block_size = 1;
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
    }

    for (int d = 0; d < md.ndims; ++d) {
        md.padded_dims[d] = utils::rnd_up(md.dims[d], blocks[d]);
        md.padded_offsets[d] = 0;
    }
    md.offset0 = 0;

    md.format_kind = format_kind::blocked;
    auto &mblk = md.format_desc.blocking;
    mblk = blk;

    const int ndims = nstl::min(DNNL_MAX_NDIMS, md.ndims); // make GCC 5 happy
    utils::array_copy(mblk.strides, blk.strides, ndims);

    dims_t ou_blocks = {0};
    utils::array_copy(ou_blocks, md.padded_dims, ndims);

    int perm[DNNL_MAX_NDIMS];
    for (int d = 0; d < ndims; ++d) {
        perm[d] = d;
        ou_blocks[d] /= blocks[d];
    }

    utils::simultaneous_sort(
            mblk.strides, ou_blocks, perm, ndims, [](stride_t a, stride_t b) {
        static_assert(runtime_value_for<stride_t>() < 0,
                "negative value is expected");
        if (any_runtime_value(a, b))
            return runtime_value_for<stride_t>(); // negative: preserves order
        return b - a;
    });

    dim_t stride = block_size;
    for (int _d = ndims - 1; _d >= 0; --_d) {
        const int d = perm[_d];
        md.format_desc.blocking.strides[d] = stride;
        if (md.padded_dims[d] != 0) { // Keep same stride for zero dim
            stride *= md.padded_dims[d] / blocks[d];
        }
    }

    md.extra = utils::zero<memory_extra_desc_t>();

    return status::success;
}

/** inits memory descriptor @p md based on another one memory descriptor
 * @p md_base and given @p data_type.
 * Essentially: { md = md_base; md.dt = data_type; } */
inline status_t memory_desc_init_by_md_and_dt(memory_desc_t &md,
        const memory_desc_t &md_base, data_type_t data_type) {
    if (&md != &md_base) md = md_base;
    md.data_type = data_type;
    return status::success;
}

/** returns true if memory desc @p md corresponds to the given format tag.
 * Assumes a dense structure such as that returned by memory_desc_init_by_tag().
 * Strides must match those returned by memory_desc_init_by_tag(), with one
 * exception: the strides of unit dimensions are ignored in order to align with
 * memory descriptor equality comparisons and hashing,
 * the strides of unit dimensions are ignored.
 * When strides are empty the dense structure is assumed (e.g., the one that
 * memory_desc_init_by_tag() returns).
 * When strides are not empty, standard strides check is overrided, and
 * additional rules are applied:
 * Strides might contain `0` value, indicating the stride must match the one
 * that memory_desc_init_by_tag() returns.
 * Strides might contain `-1` values, that would be ignored during the
 * comparison. For instance, this can be used if a stride along minibatch
 * doesn't matter. */
inline bool memory_desc_matches_tag(const memory_desc_t &md, format_tag_t tag,
        const dims_t strides = nullptr) {
    if (md.format_kind != format_kind::sparse) {
        if (md.format_kind != types::format_tag_to_kind(tag)) return false;
    }

    memory_desc_t md_gold;
    status_t status = memory_desc_init_by_tag(
            md_gold, md.ndims, md.dims, md.data_type, tag);
    if (status != status::success) return false;
    bool equal = types::blocking_desc_is_equal(
            md, md_gold, /* ignore_strides = */ (bool)strides);
    if (!strides || !equal) return equal;

    const auto &blk = md.format_desc.blocking;
    const auto &blk_gold = md_gold.format_desc.blocking;
    for (int d = 0; d < md.ndims; ++d) {
        dim_t stride = strides[d];
        if (stride == -1) continue;
        if (stride == 0) stride = blk_gold.strides[d];
        if (blk.strides[d] != stride) return false;
    }
    return true;
}

/** returns matching tag (or undef if match is not found)
 * XXX: This is a workaround that eventually should go away! */
template <typename... Tags>
format_tag_t memory_desc_matches_one_of_tag(
        const memory_desc_t &md, Tags... tags) {
    for (const auto tag : {tags...}) {
        if (memory_desc_matches_tag(md, tag)) return tag;
    }
    return format_tag::undef;
}
inline bool any_memory_desc_host_scalar(const memory_desc_t *md) {
    return md != nullptr && md->format_kind == format_kind::host_scalar;
}

template <typename... Args>
inline bool any_memory_desc_host_scalar(const memory_desc_t *md, Args... mds) {
    if (md != nullptr && md->format_kind == format_kind::host_scalar)
        return true;
    return any_memory_desc_host_scalar(mds...);
}

} // namespace impl
} // namespace dnnl

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
