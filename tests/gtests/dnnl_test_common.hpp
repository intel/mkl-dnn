/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef DNNL_TEST_COMMON_HPP
#define DNNL_TEST_COMMON_HPP

#ifdef _WIN32
#include <windows.h> // GetEnvironmentVariable
#endif

#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdint.h>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "gtest/gtest.h"

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL || defined(DNNL_WITH_SYCL)
#include "dnnl_test_common_ocl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "dnnl_test_common_ze.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

// Don't move it higher than library public headers
#include "dnnl_test_macros.hpp"

#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/memory_desc_wrapper.hpp"
#include "src/common/nstl.hpp"
#include "src/common/primitive_cache_test_api.hpp"
#include "tests/gtests/test_malloc.hpp"

// Explicit dependency on dnnl::impl::parallel calls.
#include "tests/test_thread.hpp"
#include "tests/test_thread_decl.hpp"

#include "src/cpu/platform.hpp"

#define for_ for

using dnnl::impl::bfloat16_t;
using dnnl::impl::float16_t;
using dnnl::impl::float8_e4m3_t;
using dnnl::impl::float8_e5m2_t;

#ifdef DNNL_ENABLE_MEM_DEBUG
#define DNNL_CHECK(f) \
    do { \
        dnnl_status_t s = (f); \
        dnnl::error::wrap_c_api(s, dnnl_status2str(s)); \
    } while (0)
#else
#define DNNL_CHECK(f) \
    do { \
        dnnl_status_t s = (f); \
        ASSERT_EQ(s, dnnl_success); \
    } while (0)
#endif

// XXX: Using EXPECT_NE in 'if' statement raises a warning when GCC compiler is
// used: suggest explicit braces to avoid ambiguous 'else'
#define GTEST_EXPECT_NE(val1, val2) \
    do { \
        EXPECT_NE(val1, val2); \
    } while (0)

using memory = dnnl::memory;

bool is_current_test_failed();

#ifdef DNNL_TEST_WITH_ENGINE_PARAM
dnnl::engine::kind get_test_engine_kind();
dnnl::engine get_test_engine();
#endif

inline uint32_t get_vendor_id(const std::string &vendor) {
    if (vendor == "nvidia") {
        return 0x10DE;
    } else if (vendor == "amd") {
        return 0x1002;
    } else if (vendor == "intel") {
        return 0x8086;
    } else {
        return 0x0;
    }
}

inline bool is_nvidia_gpu(const dnnl::engine &eng) {
#ifdef DNNL_WITH_SYCL
    if (eng.get_kind() != dnnl::engine::kind::gpu) return false;
    const uint32_t nvidia_vendor_id = get_vendor_id("nvidia");
    const auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == nvidia_vendor_id;
#endif
    return false;
}

inline bool is_amd_gpu(const dnnl::engine &eng) {
#ifdef DNNL_WITH_SYCL
    if (eng.get_kind() != dnnl::engine::kind::gpu) return false;
    const uint32_t amd_vendor_id = get_vendor_id("amd");
    const auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == amd_vendor_id;
#endif
    return false;
}

inline bool is_sycl_engine(dnnl::engine::kind eng_kind) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (eng_kind == dnnl::engine::kind::cpu) return true;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (eng_kind == dnnl::engine::kind::gpu) return true;
#endif
    return false;
}

inline bool unsupported_data_type(
        memory::data_type dt, const dnnl::engine &eng) {
    bool supported = true; // optimism

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    dnnl::engine::kind kind = eng.get_kind();
    if (kind == dnnl::engine::kind::cpu)
        supported = dnnl::impl::cpu::platform::has_data_type_support(
                memory::convert_to_c(dt));
#endif

#if defined(DNNL_SYCL_CUDA) || defined(DNNL_SYCL_HIP)
    if (is_nvidia_gpu(eng) || is_amd_gpu(eng)) {
        switch (dt) {
            case memory::data_type::f32: return false;
            case memory::data_type::f16: return false;
            case memory::data_type::s8: return false;
            case memory::data_type::undef: return false;
            default: return true;
        }
    }
#endif
    return !supported;
}

#ifdef DNNL_TEST_WITH_ENGINE_PARAM
inline bool unsupported_data_type(memory::data_type dt) {
    return unsupported_data_type(dt, get_test_engine());
}

template <typename... Rest>
inline bool unsupported_data_type(
        memory::data_type first_dt, Rest... rest_dts) {
    bool rval = unsupported_data_type(first_dt, get_test_engine());
    if (rval) return rval;
    return unsupported_data_type(rest_dts...);
}
#endif

inline bool unsupported_prop_kind(
        dnnl::prop_kind pk, memory::data_type dt, const dnnl::engine &eng) {
    bool supported = !unsupported_data_type(dt, eng);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    dnnl::engine::kind kind = eng.get_kind();
    if (supported && kind == dnnl::engine::kind::cpu)
        supported = IMPLICATION(pk != dnnl::prop_kind::forward_inference,
                dnnl::impl::cpu::platform::has_training_support(
                        memory::convert_to_c(dt)));
#endif
    return !supported;
}

#ifdef DNNL_TEST_WITH_ENGINE_PARAM
inline bool unsupported_prop_kind(dnnl::prop_kind pk, memory::data_type dt) {
    return unsupported_prop_kind(pk, dt, get_test_engine());
}

template <typename... Rest>
inline bool unsupported_prop_kind(
        dnnl::prop_kind pk, memory::data_type first_dt, Rest... rest_dts) {
    bool rval = unsupported_prop_kind(pk, first_dt, get_test_engine());
    if (rval) return rval;
    return unsupported_prop_kind(pk, rest_dts...);
}
#endif

template <typename data_t>
struct data_traits_t {};
template <>
struct data_traits_t<float16_t> {
    static const auto data_type = memory::data_type::f16;

    using uint_type = uint16_t;
};
template <>
struct data_traits_t<bfloat16_t> {
    static const auto data_type = memory::data_type::bf16;

    using uint_type = uint16_t;
};
template <>
struct data_traits_t<float> {
    static const auto data_type = memory::data_type::f32;

    using uint_type = uint32_t;
};
template <>
struct data_traits_t<uint8_t> {
    static const auto data_type = memory::data_type::u8;

    using uint_type = uint8_t;
};
template <>
struct data_traits_t<int8_t> {
    static const auto data_type = memory::data_type::s8;

    using uint_type = uint8_t;
};
template <>
struct data_traits_t<int32_t> {
    static const auto data_type = memory::data_type::s32;

    using uint_type = uint32_t;
};

template <typename T>
inline void assert_eq(T a, T b);
template <>
inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
inline int mxcsr_cvt(float f) {
    return _mm_cvtss_si32(_mm_load_ss(&f));
}
#else
inline int mxcsr_cvt(float f) {
    return (int)nearbyintf(f);
}
#endif

template <typename data_t>
data_t out_round(float x) {
    return (data_t)mxcsr_cvt(x);
}
template <>
inline float out_round<float>(float x) {
    return x;
}

template <typename data_t, typename out_t>
out_t saturate(const out_t &x) {
    out_t v = x;
    if (v <= std::numeric_limits<data_t>::min())
        v = std::numeric_limits<data_t>::min();
    if (v > std::numeric_limits<data_t>::max())
        v = std::numeric_limits<data_t>::max();
    return v;
}

inline memory::dim right_padding(memory::dim i, memory::dim o, memory::dim k,
        memory::dim p, memory::dim s, memory::dim d = 0) {
    return (o - 1) * s + (k - 1) * (d + 1) - (p + i - 1);
}

template <typename data_t>
struct acc_t {
    using type = data_t;
};
template <>
struct acc_t<int8_t> {
    using type = int;
};
template <>
struct acc_t<uint8_t> {
    using type = int;
};

// Smart pointer for map/unmap operations with unique_ptr semantics.
//
// Note: this abstraction works only under assumption that execution post this
// object construction is synchronous. However, this assumption doesn't hold for
// asynchronous threadpool runtime (ATR); it also requires any object being
// copyable when captured by a parallel-like call but this class restricts a
// copy.
// To overcome this limitation a trick is used which, in turn, relies on another
// assumption that ATR doesn't require real mapping and freely uses a bare
// pointer provided by this abstraction when accessing it.
// Thus, runtimes relying on mapping will execute in synchronous mode and use
// a mapped pointer with this abstraction unmapping it at its destruction, and
// other runtimes just using a bare pointer.
//
// If the latter assumption will no longer hold, there's no other way but to do
// honest manual mapping and unmapping and passing a pointer inside parallel
// calls to satisfy all runtimes requirements.
template <typename T>
struct mapped_ptr_t {
    using nonconst_type = typename std::remove_cv<T>::type;

    mapped_ptr_t(std::nullptr_t) : mem_(nullptr), ptr_(nullptr) {}
    mapped_ptr_t(const memory *mem)
        : mem_(mem), ptr_(mem->map_data<nonconst_type>()) {}
    mapped_ptr_t(mapped_ptr_t &&other) : mem_(other.mem_), ptr_(other.ptr_) {
        other.mem_ = nullptr;
        other.ptr_ = nullptr;
    }

    mapped_ptr_t(const mapped_ptr_t &) = delete;
    mapped_ptr_t &operator=(const mapped_ptr_t &) = delete;

    ~mapped_ptr_t() {
        if (mem_ && ptr_) mem_->unmap_data(ptr_);
    }

    operator T *() { return ptr_; }
    operator const T *() const { return ptr_; }
    operator bool() const { return ptr_ != nullptr; }

private:
    const memory *mem_;
    nonconst_type *ptr_;
};

template <typename T>
mapped_ptr_t<T> map_memory(const memory &mem) {
    return mapped_ptr_t<T>(&mem);
}

// check_zero_tail - check on zero or set to zero padded memory
template <typename data_t>
void check_zero_tail(int set_zero_flag, const memory &src) {

    auto src_data = map_memory<data_t>(src);

    const memory::desc src_d = src.get_desc();
    const int ndims = src_d.get_ndims();
    const auto dims = src_d.get_dims();
    const auto pdims = src_d.get_padded_dims();
    const dnnl::impl::memory_desc_wrapper mdw(src_d.get());

    memory::dim idx[DNNL_MAX_NDIMS] = {}, str[DNNL_MAX_NDIMS] = {};
    memory::dim nelems = 1;
    int tail_flag = 0;
    for (int i = 0; i < ndims; ++i) {
        if (dims[ndims - i - 1] != pdims[ndims - i - 1]) tail_flag = 1;
        nelems *= pdims[ndims - i - 1];
        idx[i] = 0;
        str[i] = (i == 0) ? 1 : str[i - 1] * pdims[ndims - i];
    }
    if (tail_flag == 0) return;

    for (memory::dim i = 0; i < nelems; ++i) {
        memory::dim off = 0;
        bool flag = false;
        for (int j = 0; j < ndims; ++j) {
            off += idx[j] * str[j];
            if (idx[j] >= dims[ndims - j - 1]) flag = true;
        }
        if (flag == true) {
            memory::dim blk_off = mdw.off_l(off, true);
            if (set_zero_flag) {
                src_data[blk_off] = 0.0;
            } else {
                ASSERT_EQ(src_data[blk_off], 0.0)
                        << " blk_off = " << blk_off << "off = " << off;
            }
        }
        /*Update idx*/
        for (int j = 0; j < ndims; ++j) {
            idx[j]++;
            if (idx[j] < pdims[ndims - j - 1]) break;
            idx[j] = 0;
        }
    }
}

inline memory::desc create_md(const memory::dims &dims,
        memory::data_type data_type, memory::format_tag fmt_tag) {
    return memory::desc(dims, data_type, fmt_tag);
}

inline void synchronize_threadpool(dnnl::engine::kind akind) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (akind == dnnl::engine::kind::cpu) {
        auto *tp = dnnl::testing::get_threadpool();
        if (tp) tp->wait();
    }
#endif
}

template <typename data_t>
static inline data_t set_value(
        memory::dim index, data_t mean, data_t deviation, double sparsity) {
    if (data_traits_t<data_t>::data_type == memory::data_type::f16
            || data_traits_t<data_t>::data_type == memory::data_type::bf16) {
        return data_t(set_value<float>(index, mean, deviation, sparsity));
    } else if (data_traits_t<data_t>::data_type == memory::data_type::f32) {
        const memory::dim group_size = (memory::dim)(1. / sparsity);
        const memory::dim group = index / group_size;
        const memory::dim in_group = index % group_size;
        const bool fill = in_group == ((group % 1637) % group_size);
        return fill ? static_cast<data_t>(
                              mean + deviation * sinf(float(index % 37)))
                    : data_t {0};
    } else if (data_traits_t<data_t>::data_type == memory::data_type::s32
            || data_traits_t<data_t>::data_type == memory::data_type::s8) {
        return data_t(index * 13 % 21 - 10);
    } else if (data_traits_t<data_t>::data_type == memory::data_type::u8) {
        return data_t(index * 13 % 17);
    }
    assert(!"not expected");
    return data_t(0);
}

template <typename data_t>
void fill_data(const memory::dim nelems, data_t *data, data_t mean,
        data_t deviation, double sparsity = 1.) {
    dnnl::impl::parallel_nd(nelems, [=](memory::dim n) {
        data[n] = set_value<data_t>(n, mean, deviation, sparsity);
    });
}

template <typename data_t>
void fill_data(const memory::dim nelems, const memory &mem, data_t mean,
        data_t deviation, double sparsity = 1.) {
    auto data_ptr = map_memory<data_t>(mem);
    fill_data<data_t>(nelems, data_ptr, mean, deviation, sparsity);
    synchronize_threadpool(mem.get_engine().get_kind());
}

inline void fill_data(memory::data_type dt, const memory &mem, float mean,
        float deviation, double sparsity = 1.) {
    const auto dt_size = memory::data_type_size(dt);
    if (dt_size <= 0) {
        assert(!"unexpected data type");
        return;
    }

    memory::dim nelems
            = static_cast<memory::dim>(mem.get_desc().get_size() / dt_size);
    switch (dt) {
        case memory::data_type::f32:
            fill_data<float>(nelems, mem, mean, deviation, sparsity);
            break;
        case memory::data_type::bf16:
            fill_data<bfloat16_t>(nelems, mem, mean, deviation, sparsity);
            break;
        case memory::data_type::f16:
            fill_data<float16_t>(nelems, mem, mean, deviation, sparsity);
            break;
        case memory::data_type::s32:
            fill_data<int>(nelems, mem, mean, deviation, sparsity);
            break;
        case memory::data_type::s8:
            fill_data<int8_t>(nelems, mem, mean, deviation, sparsity);
            break;
        case memory::data_type::u8:
            fill_data<uint8_t>(nelems, mem, mean, deviation, sparsity);
            break;
        default: assert(!"unsupported data type"); break;
    }
    synchronize_threadpool(mem.get_engine().get_kind());
}

template <typename data_t>
void fill_data(const memory::dim nelems, data_t *data, double sparsity = 1.,
        bool init_negs = false) {
    dnnl::impl::parallel_nd(nelems, [=](memory::dim n) {
        data[n] = set_value<data_t>(n, data_t(1), data_t(0.2f), sparsity);

        if (init_negs && n % 4 == 0)
            data[n] = static_cast<data_t>(
                    -data[n]); // weird for unsigned types!
    });
}

template <typename data_t>
void fill_data(const memory::dim nelems, const memory &mem,
        double sparsity = 1., bool init_negs = false) {
    auto data_ptr = map_memory<data_t>(mem);
#if !defined(TEST_DNNL_DPCPP_BUFFER)
    // Workaround for PVC/ATSM gtest hangs:
    // On GPU USM memory (i915), concurrent first write from many threads into a
    // freshly mapped buffer can hit a page-fault race.
    // To avoid this race, wa touches every page sequentially from a single
    // thread before what serialized the faults.
    // The time overhead is close to the noise level.
    if (data_ptr && nelems > 0
            && mem.get_engine().get_kind() == dnnl::engine::kind::gpu) {
        const size_t page = 4096;
        const size_t nbytes = (size_t)nelems * sizeof(data_t);
        volatile char *p = reinterpret_cast<volatile char *>(
                static_cast<data_t *>(data_ptr));
        for (size_t off = 0; off < nbytes; off += page)
            p[off] = 0;
        p[nbytes - 1] = 0; // touch last page too
    }
#endif
    fill_data<data_t>(nelems, data_ptr, sparsity, init_negs);
    synchronize_threadpool(mem.get_engine().get_kind());
}

template <typename data_t>
void remove_zeroes(const memory &mem) {
    size_t nelems = mem.get_desc().get_size() / sizeof(data_t);
    auto mem_data = map_memory<data_t>(mem);
    data_t *data_ptr = mem_data;
    dnnl::impl::parallel_nd(nelems, [=](memory::dim n) {
        if (data_ptr[n] == data_t(0)) data_ptr[n] += data_t(1);
    });
}

template <typename data_t>
void compare_data(
        const memory &ref, const memory &dst, data_t threshold = (data_t)1e-4) {
    using data_type = memory::data_type;

    ASSERT_TRUE(data_traits_t<data_t>::data_type == data_type::f32
            || data_traits_t<data_t>::data_type == data_type::f16
            || data_traits_t<data_t>::data_type == data_type::bf16
            || data_traits_t<data_t>::data_type == data_type::s32
            || data_traits_t<data_t>::data_type == data_type::s8);

    /* Note: size_t incompatible with MSVC++ */
    auto ref_desc = ref.get_desc();
    auto dst_desc = dst.get_desc();
    const dnnl::impl::memory_desc_wrapper mdw_ref(ref_desc.get());
    const dnnl::impl::memory_desc_wrapper mdw_dst(dst_desc.get());

    ASSERT_TRUE(ref_desc.get_ndims() == dst_desc.get_ndims());

    auto ndims = ref_desc.get_ndims();

    for (auto d = 0; d < ndims; ++d) {
        ASSERT_TRUE(ref_desc.get_dims()[d] == dst_desc.get_dims()[d]);
    }

    auto dims = ref_desc.get_dims();

    memory::dim num = 1;
    for (auto d = 0; d < ndims; ++d) {
        num *= dims[d];
    }

    auto ref_mapped = map_memory<data_t>(ref);
    data_t *ref_data = ref_mapped;
    auto dst_mapped = map_memory<data_t>(dst);
    data_t *dst_data = dst_mapped;

    dnnl::impl::parallel_nd(num, [=](memory::dim i) {
        if (is_current_test_failed()) return;

        data_t ref = ref_data[mdw_ref.off_l(i, true)];
        data_t got = dst_data[mdw_dst.off_l(i, true)];

        if (data_traits_t<data_t>::data_type == data_type::f32
                || data_traits_t<data_t>::data_type == data_type::f16
                || data_traits_t<data_t>::data_type == data_type::bf16) {
            const float threshold_f32 = static_cast<float>(threshold);
            const float ref_f32 = static_cast<float>(ref);
            const float got_f32 = static_cast<float>(got);
            const float diff_f32
                    = (got_f32 == ref_f32) ? 0.0f : got_f32 - ref_f32;
            const float e = (std::abs(ref_f32) > threshold_f32)
                    ? diff_f32 / ref_f32
                    : diff_f32;
            ASSERT_NEAR(e, 0.0, threshold_f32)
                    << "Index: " << i << " Total: " << num;
        } else {
            ASSERT_EQ(ref, got) << "Index: " << i << " Total: " << num;
        }
    });

    synchronize_threadpool(dst.get_engine().get_kind());
}

inline const char *query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    return str;
}

inline dnnl_status_t get_conv_impl_status(
        const_dnnl_primitive_desc_t pd, const char *match_str) {
    const char *conv_str = query_impl_info(pd);

    if (strstr(conv_str, match_str) != nullptr)
        return dnnl_status_t::dnnl_success;
    return dnnl_status_t::dnnl_unimplemented;
}

struct test_convolution_sizes_t {
    test_convolution_sizes_t(memory::dim mb, memory::dim ng, memory::dim ic,
            memory::dim ih, memory::dim iw, memory::dim oc, memory::dim oh,
            memory::dim ow, memory::dim kh, memory::dim kw, memory::dim padh,
            memory::dim padw, memory::dim strh, memory::dim strw,
            memory::dim dilh = 0, memory::dim dilw = 0)
        : mb(mb)
        , ng(ng)
        , ic(ic)
        , ih(ih)
        , iw(iw)
        , oc(oc)
        , oh(oh)
        , ow(ow)
        , kh(kh)
        , kw(kw)
        , padh(padh)
        , padw(padw)
        , strh(strh)
        , strw(strw)
        , dilh(dilh)
        , dilw(dilw) {}
    memory::dim mb;
    memory::dim ng;
    memory::dim ic, ih, iw;
    memory::dim oc, oh, ow;
    memory::dim kh, kw;
    memory::dim padh, padw;
    memory::dim strh, strw;
    memory::dim dilh, dilw;
};

struct test_convolution_attr_t {
    struct scale_t {
        enum policy_t { NONE = 0, COMMON };

        bool is_def() const { return policy != NONE; }

        scale_t(float s, policy_t p = NONE) : policy(p), scale(s) {}

        policy_t policy;
        float scale;
    };

    void dnnl_attr_recreate() {
        dnnl_attr = dnnl::primitive_attr();
        if (src_scale.is_def()) {
            const int mask = 0;
            dnnl_attr.set_scales_mask(DNNL_ARG_SRC, mask);
        }
        if (wei_scale.is_def()) {
            const int mask = 0;
            dnnl_attr.set_scales_mask(DNNL_ARG_WEIGHTS, mask);
        }
        if (dst_scale.is_def()) {
            const int mask = 0;
            dnnl_attr.set_scales_mask(DNNL_ARG_DST, mask);
        }
    }

    test_convolution_attr_t(
            float s, scale_t::policy_t p = scale_t::policy_t::NONE)
        : src_scale(s, p), wei_scale(s, p), dst_scale(s, p) {}

    test_convolution_attr_t() : test_convolution_attr_t(1.f) {}

    scale_t src_scale;
    scale_t wei_scale;
    scale_t dst_scale;
    dnnl::primitive_attr dnnl_attr;
};

struct test_convolution_formats_t {
    memory::format_tag src_format;
    memory::format_tag weights_format;
    memory::format_tag bias_format;
    memory::format_tag dst_format;
};

struct test_convolution_params_t {
    dnnl::algorithm aalgorithm;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

struct test_convolution_eltwise_params_t {
    const dnnl::algorithm alg;
    dnnl::algorithm aalgorithm;
    const float eltwise_alpha;
    const float eltwise_beta;
    test_convolution_formats_t formats;
    test_convolution_attr_t attr;
    test_convolution_sizes_t sizes;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename F>
bool catch_expected_failures(const F &f, bool expect_to_fail,
        dnnl_status_t expected_status, bool ignore_unimplemented = false,
        const char *filename = __FILE__, int64_t line_num = __LINE__) {
    try {
        f();
    } catch (const dnnl::error &e) {
        // Rethrow the exception if it is not expected or the error status did
        // not match.
        if (!(expect_to_fail) || e.status != (expected_status)) {
            // Ignore unimplemented and unsupported data type
            if (ignore_unimplemented
                    && (e.status == dnnl_unimplemented
                            || e.status == dnnl_unsupported_data_type)) {
                // Print the reason but do not treat as error
                std::cout
                        << "(" << filename << ":" << line_num << ") "
                        << "[  UNIMPL  ] "
                        << (e.status == dnnl_unsupported_data_type
                                           ? "Data type not supported on this "
                                             "platform"
                                           : "Implementation not found")
                        << std::endl;
                reset_failed_malloc_counter();
                return true;
            } else if (test_out_of_memory()
                    && (e.status == dnnl_out_of_memory
                            || e.status == dnnl_unimplemented)) {
                // Restart if error thrown due to a malloc failed intentionally,
                // and increment malloc counter.
                // TODO: This should be valid only for `dnnl_out_of_memory`
                // error. Currently a failed malloc inside
                // gemm_pack_storage_shell_t ctor makes it unable to use the
                // reference RNN impl, and the iterator produces an
                // `dnnl_unimplemented` error.
                increment_failed_malloc_counter();
                return catch_expected_failures(f, expect_to_fail,
                        expected_status, ignore_unimplemented);
            } else {
                if (expect_to_fail && (e.status != expected_status))
                    std::cout << "(" << filename << ":" << line_num << ") "
                              << "Expect failure status mismatch: expect("
                              << dnnl_status2str(expected_status) << ") get("
                              << dnnl_status2str(e.status)
                              << "). Re-throwing...\n";
                throw e;
            }
        }
        // Return normally if the failure is expected. Reset failed malloc
        // counter to zero before performing a new test.
        if (expect_to_fail) {
            reset_failed_malloc_counter();
            return true;
        }
    }

    // Throw an exception if the failure is expected but did not happen
    if (expect_to_fail) {
        std::cout << "(" << filename << ":" << line_num << ") "
                  << "Expect failure with status("
                  << dnnl_status2str(expected_status) << "), "
                  << "but operation succeed. Throwing an exception...\n";
        throw std::exception();
    }

    // Reset failed malloc counter to zero before performing a new test.
    reset_failed_malloc_counter();
    return false;
}

namespace test {
inline dnnl::memory make_memory(
        const dnnl::memory::desc &md, const dnnl::engine &eng) {

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        return dnnl::memory(md, eng);
    }
#endif

#if defined(TEST_DNNL_OCL_USM)
    return dnnl::ocl_interop::make_memory(
            md, eng, dnnl::ocl_interop::memory_kind::usm);
#elif defined(TEST_DNNL_DPCPP_BUFFER)
    return dnnl::sycl_interop::make_memory(
            md, eng, dnnl::sycl_interop::memory_kind::buffer);
#else
    return dnnl::memory(md, eng);
#endif
}

inline dnnl::memory make_memory(
        const dnnl::memory::desc &md, const dnnl::engine &eng, void *handle) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        return dnnl::memory(md, eng, handle);
    }
#endif

#if defined(TEST_DNNL_OCL_USM)
    return dnnl::ocl_interop::make_memory(
            md, eng, dnnl::ocl_interop::memory_kind::usm, handle);
#elif defined(TEST_DNNL_DPCPP_BUFFER)
    return dnnl::sycl_interop::make_memory(
            md, eng, dnnl::sycl_interop::memory_kind::buffer, handle);
#else
    return dnnl::memory(md, eng, handle);
#endif
}
} // namespace test

#define TEST_MALLOC_OFFSET 8
static char *test_malloc(size_t size) {
    void *ptr;
    const size_t align = 64;
    const size_t padded_size = TEST_MALLOC_OFFSET + size;
#ifdef _WIN32
    ptr = _aligned_malloc(padded_size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, padded_size);
#endif /* _WIN32 */
    return rc == 0 ? (char *)ptr + TEST_MALLOC_OFFSET : nullptr;
}

static void test_free(char *ptr) {
    char *base_ptr = ptr - TEST_MALLOC_OFFSET;
#ifdef _WIN32
    _aligned_free(base_ptr);
#else
    return ::free(base_ptr);
#endif /* _WIN32 */
}
#undef TEST_MALLOC_OFFSET

//NOLINTNEXTLINE(readability-identifier-naming)
class test_memory {
public:
    test_memory(const memory::desc &d, const dnnl::engine &e)
        : size_(d.get_size()) {
        bool is_cpu_native = (e.get_kind() == dnnl::engine::kind::cpu)
                && DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL;

        if (is_cpu_native) {
            data_.reset(test_malloc(size_), test_free);
            mem_ = test::make_memory(d, e, data_.get());
        } else {
            mem_ = test::make_memory(d, e);
        }
        // Fill with a magic number to catch possible uninitialized access
        mapped_ptr_t<char> ptr(&mem_);
        if (ptr) memset(ptr, 0xFF, size_);
    }

    size_t get_size() const { return size_; }
    const memory &get() const { return mem_; }

    operator bool() const { return mem_.get(true) != nullptr; }

private:
    memory mem_;
    std::shared_ptr<char> data_;
    size_t size_;
};

template <typename T>
mapped_ptr_t<T> map_memory(const test_memory &mem) {
    return mapped_ptr_t<T>(&mem.get());
}

inline std::string to_string(dnnl_engine_kind_t engine_kind) {
    dnnl::impl::stringstream_t ss;
    if (engine_kind == dnnl_cpu)
        ss << "cpu";
    else if (engine_kind == dnnl_gpu)
        ss << "gpu";
    else
        ss << "unknown";

    return ss.str();
}

inline std::string to_string(dnnl_stream_flags_t stream_flags) {
    dnnl::impl::stringstream_t ss;
    if (stream_flags & dnnl_stream_default_flags)
        ss << "default";
    else if (stream_flags & dnnl_stream_in_order)
        ss << "in_order";
    else if (stream_flags & dnnl_stream_out_of_order)
        ss << "out_of_order";

    return ss.str();
}

// testing all available C++ primitive descriptor constructors
struct allows_attr_t {
    bool po_sum = false;
    bool po_eltwise = false;
    bool po_binary = false;
    bool po_prelu = false;
    bool zp = false;
    bool scales = false;
    int scales_arg = DNNL_ARG_SRC;
};

using engine = dnnl::engine;
// forward
template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr(const engine &eng, const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr;
    EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_po_sum(const engine &eng, bool supports_po_sum,
        const prim_params_t &...prim_params) {
    dnnl::post_ops ops_sum;
    ops_sum.append_sum(1.1f);
    dnnl::primitive_attr attr_po_sum;
    attr_po_sum.set_post_ops(ops_sum);
    if (supports_po_sum)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_po_sum));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_po_sum));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_po_eltwise(const engine &eng, bool supports_po_eltwise,
        const prim_params_t &...prim_params) {
    dnnl::post_ops ops_eltwise;
    ops_eltwise.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    dnnl::primitive_attr attr_po_eltwise;
    attr_po_eltwise.set_post_ops(ops_eltwise);
    if (supports_po_eltwise)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_po_eltwise));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_po_eltwise));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_po_binary(const engine &eng, bool supports_po_binary,
        const pd_t &pd, const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr_po_binary_good, attr_po_binary_bad;
    dnnl::post_ops ops_binary_good, ops_binary_bad;

    int dst_ndims = pd.dst_desc().get_ndims();
    dnnl::memory::dims dims_good(dst_ndims, 1);
    dnnl::memory::dims dims_bad(
            dst_ndims > 1 ? dst_ndims - 1 : dst_ndims + 1, 1);

    dnnl::memory::desc src1_desc_good(
            dims_good, memory::data_type::f32, /* strides */ dims_good);
    dnnl::memory::desc src1_desc_bad(
            dims_bad, memory::data_type::s8, /* strides */ dims_bad);

    ops_binary_good.append_binary(dnnl::algorithm::binary_mul, src1_desc_good);
    ops_binary_bad.append_binary(dnnl::algorithm::binary_mul, src1_desc_bad);

    attr_po_binary_good.set_post_ops(ops_binary_good);
    attr_po_binary_bad.set_post_ops(ops_binary_bad);

    if (!supports_po_binary) {
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_po_binary_good));
        return;
    }

    EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_po_binary_bad));
    EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_po_binary_good));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_po_prelu(const engine &eng, bool supports_po_prelu,
        const prim_params_t &...prim_params) {
    dnnl::post_ops ops_prelu;
    ops_prelu.append_prelu(0);
    dnnl::primitive_attr attr_po_prelu;
    attr_po_prelu.set_post_ops(ops_prelu);
    if (supports_po_prelu)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_po_prelu));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_po_prelu));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_zp(const engine &eng, bool supports_zero_point,
        const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr_zp;
    attr_zp.set_zero_points_mask(DNNL_ARG_SRC, 0);
    if (supports_zero_point)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_zp));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_zp));
}

// Note: `arg` is needed to specify `DNNL_ARG_MULTIPLE_SRC` as a value for
// supported scale for concat and sum.
template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_attr_scales(const engine &eng, bool supports_scales, int arg,
        const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr_scales;
    attr_scales.set_scales_mask(arg, 0);

    if (supports_scales) { // Currently only used with binary ops
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., attr_scales));
    } else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., attr_scales));
}

template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_allow_empty(
        const pd_t &pd, const prim_params_t &...prim_params) {
    bool allow_empty = true;
    pd_t new_pd {};
    dnnl::primitive_attr unsupported_attr;
    // Assumption is that mask "10" is a legit mask for scales
    // from API perspective.
    unsupported_attr.set_scales_mask(DNNL_ARG_SRC, 10);
    ASSERT_NO_THROW(new_pd = pd_t(pd.get_engine(), prim_params...,
                            unsupported_attr, allow_empty));
    ASSERT_FALSE(new_pd);
}

// Note: requires a valid primitive descriptor!
template <typename pd_t, typename... prim_params_t>
void test_fwd_pd_constructors(const pd_t &pd, const allows_attr_t &aa,
        const prim_params_t &...prim_params) {
    auto test_pd = pd_t();
    auto eng = pd.get_engine();
    // ctor from C pd, should not throw
    ASSERT_NO_THROW(test_pd = pd_t(pd.get()));
    // ctor w/ empty attr, should not throw
    test_fwd_pd_attr<pd_t>(eng, prim_params...);
    // following ctors w/ attrs may throw based on pd support
    test_fwd_pd_attr_po_sum<pd_t>(eng, aa.po_sum, prim_params...);
    test_fwd_pd_attr_po_eltwise<pd_t>(eng, aa.po_eltwise, prim_params...);
    test_fwd_pd_attr_po_binary<pd_t>(eng, aa.po_binary, pd, prim_params...);
    test_fwd_pd_attr_po_prelu<pd_t>(eng, aa.po_prelu, prim_params...);
    test_fwd_pd_attr_zp<pd_t>(eng, aa.zp, prim_params...);
    test_fwd_pd_attr_scales<pd_t>(
            eng, aa.scales, aa.scales_arg, prim_params...);
    // check allow empty, should not throw
    test_fwd_pd_allow_empty<pd_t>(test_pd, prim_params...);
}

// backward: has hint
template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr(const engine &eng, const hint_pd_t &hint,
        const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr;
    EXPECT_NO_THROW(pd_t pd(eng, prim_params..., hint, attr));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr_po_sum(const engine &eng, const hint_pd_t &hint,
        bool supports_po_sum, const prim_params_t &...prim_params) {
    dnnl::post_ops ops_sum;
    ops_sum.append_sum(1.1f);
    dnnl::primitive_attr attr_po_sum;
    attr_po_sum.set_post_ops(ops_sum);
    if (supports_po_sum)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., hint, attr_po_sum));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., hint, attr_po_sum));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr_po_eltwise(const engine &eng, const hint_pd_t &hint,
        bool supports_po_eltwise, const prim_params_t &...prim_params) {
    dnnl::post_ops ops_eltwise;
    ops_eltwise.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    dnnl::primitive_attr attr_po_eltwise;
    attr_po_eltwise.set_post_ops(ops_eltwise);
    if (supports_po_eltwise)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., hint, attr_po_eltwise));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., hint, attr_po_eltwise));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr_po_binary(const engine &eng, const hint_pd_t &hint,
        bool supports_po_binary, const prim_params_t &...prim_params) {
    dnnl::post_ops ops_binary;
    dnnl::memory::desc src1_desc(
            {16}, memory::data_type::s8, memory::format_tag::x);
    ops_binary.append_binary(dnnl::algorithm::binary_mul, src1_desc);
    dnnl::primitive_attr attr_po_binary;
    attr_po_binary.set_post_ops(ops_binary);
    if (supports_po_binary)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., hint, attr_po_binary));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., hint, attr_po_binary));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr_zp(const engine &eng, const hint_pd_t &hint,
        bool supports_zero_point, const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr_zp;
    attr_zp.set_zero_points_mask(DNNL_ARG_SRC, 0);
    if (supports_zero_point)
        EXPECT_NO_THROW(pd_t pd(eng, prim_params..., hint, attr_zp));
    else
        EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., hint, attr_zp));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_attr_scales(const engine &eng, const hint_pd_t &hint,
        bool supports_scales, const prim_params_t &...prim_params) {
    dnnl::primitive_attr attr_scales;
    attr_scales.set_scales_mask(DNNL_ARG_SRC, 0);
    EXPECT_ANY_THROW(pd_t pd(eng, prim_params..., hint, attr_scales));
}

template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_allow_empty(const pd_t &pd, const hint_pd_t &hint,
        const prim_params_t &...prim_params) {
    bool allow_empty = true;
    pd_t new_pd {};
    dnnl::primitive_attr unsupported_attr;
    // Assumption is that mask "10" is a legit mask for scales
    // from API perspective.
    unsupported_attr.set_scales_mask(DNNL_ARG_SRC, 10);
    ASSERT_NO_THROW(new_pd = pd_t(pd.get_engine(), prim_params..., hint,
                            unsupported_attr, allow_empty));
    ASSERT_FALSE(new_pd);
}

// Note: requires a valid primitive descriptor!
template <typename pd_t, typename hint_pd_t, typename... prim_params_t>
void test_bwd_pd_constructors(const pd_t &pd, const hint_pd_t &hint_pd,
        const allows_attr_t &aa, const prim_params_t &...prim_params) {
    auto test_pd = pd_t();
    auto eng = pd.get_engine();
    // ctor from C pd, should not throw
    ASSERT_NO_THROW(test_pd = pd_t(pd.get()));
    // ctor w/ empty attr, should not throw
    test_bwd_pd_attr<pd_t>(eng, hint_pd, prim_params...);
    // following ctors w/ attrs may throw based on pd support
    test_bwd_pd_attr_po_sum<pd_t>(eng, hint_pd, aa.po_sum, prim_params...);
    test_bwd_pd_attr_po_eltwise<pd_t>(
            eng, hint_pd, aa.po_eltwise, prim_params...);
    test_bwd_pd_attr_po_binary<pd_t>(
            eng, hint_pd, aa.po_binary, prim_params...);
    test_bwd_pd_attr_zp<pd_t>(eng, hint_pd, aa.zp, prim_params...);
    test_bwd_pd_attr_scales<pd_t>(eng, hint_pd, aa.scales, prim_params...);
    // check allow empty, should not throw
    test_bwd_pd_allow_empty<pd_t>(test_pd, hint_pd, prim_params...);
}

inline dnnl::stream make_stream(const dnnl::engine &engine,
        dnnl::stream::flags flags = dnnl::stream::flags::default_flags) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (engine.get_kind() == dnnl::engine::kind::cpu)
        return dnnl::threadpool_interop::make_stream(
                engine, dnnl::testing::get_threadpool());
#endif
    return dnnl::stream(engine, flags);
}

inline int get_primitive_cache_size() {
    int result = 0;
    auto status = dnnl_test_get_primitive_cache_size(&result);
    if (status != dnnl::impl::status::success) return -1;
    return result;
}

// This is a local copy of dnnl::impl::getenv.
// Copying to avoid exposure of internal symbol from the library.
inline int gtest_getenv(const char *name, char *buffer, int buffer_size) {
    if (name == nullptr || buffer_size < 0
            || (buffer == nullptr && buffer_size > 0))
        return INT_MIN;

    int result = 0;
    int term_zero_idx = 0;
    size_t value_length = 0;

#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, buffer, buffer_size);
#else
    const char *value = ::getenv(name);
    value_length = value == nullptr ? 0 : strlen(value);
#endif

    if (value_length > INT_MAX)
        result = INT_MIN;
    else {
        int int_value_length = (int)value_length;
        if (int_value_length >= buffer_size) {
            result = -int_value_length;
        } else {
            term_zero_idx = int_value_length;
            result = int_value_length;
#ifndef _WIN32
            if (value) strncpy(buffer, value, buffer_size - 1);
#endif
        }
    }

    if (buffer != nullptr) buffer[term_zero_idx] = '\0';
    return result;
}

#endif
