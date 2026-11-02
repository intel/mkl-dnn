/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <cmath>
#include <vector>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "tests/test_isa_common.hpp"

#include "src/common/bit_cast.hpp"

#if DNNL_X64
#include "src/cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "src/cpu/x64/jit_brgemm_conv_fp8_cvt_kernel.hpp"
#endif

using dnnl::impl::utils::bit_cast;

namespace dnnl {

TEST(test_ref_float8_conversions, f8_e5m2_to_f32) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e5m2),
            "Engine does not support this data type.");
    // check all 256 f8_e5m2 values
    impl::parallel_nd(0xff, [=](uint16_t u16) {
        // Convert f8_e5m2 to f32 and back, expecting bitwise identical values.
        // Exception is sNaN, where the convention is to set the quiet bit:
        // * +sNaN: 0x7d -> 0x7f
        // * -sNaN: 0xfd -> 0xff
        // f8_e5m2 encoding: seeeeemm
        //                         |_-> quiet bit is msb of mantissa
        uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e5m2_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));
        float x32(x8);
        float8_e5m2_t y8(x32);

        // if x8 is an sNaN the conversion sets the quiet bit (msb of mantissa)
        const bool is_x8_snan = impl::utils::one_of(u8, 0x7d, 0xfd);
        const uint8_t y8_expect = is_x8_snan ? u8 | 0x02 : u8;

        ASSERT_EQ(y8_expect, bit_cast<uint8_t>(y8));
    });
}

TEST(test_ref_float8_conversions, f8_e4m3_to_f32) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e4m3),
            "Engine does not support this data type.");
    // check all 256 f8_e4m3 values
    impl::parallel_nd(0xff, [=](uint16_t u16) {
        uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e4m3_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));

        // Convert f8_e4m3 to f32 and back, expecting bitwise identical values.
        // Note: f8_e4m3 does not have sNaN values, so no need to set quiet bit
        float x32(x8);
        float8_e4m3_t y8(x32);
        const uint8_t y8_expect = u8;
        ASSERT_EQ(y8_expect, bit_cast<uint8_t>(y8))
                << std::hex << std::endl
                << "u8 = " << static_cast<uint32_t>(u8) << std::endl
                << "x8.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(x8)) << std::endl
                << "y8.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(y8)) << std::endl
                << "y8_expect = " << static_cast<uint32_t>(y8_expect)
                << std::endl
                << std::dec;
    });
}

TEST(test_ref_float8_conversions, f8_e5m2_to_bf16) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e5m2),
            "Engine does not support this data type.");
    // check all 256 f8_e5m2 values
    impl::parallel_nd(0xff, [&](uint16_t u16) {
        // convert f8_e5m2 to bf16 and back again,
        // expecting bitwise idendical values except for sNaN,
        // where the convention is to set the quiet bit:
        // * +sNaN: 0x7d -> 0x7f
        // * -sNaN: 0xfd -> 0xff
        // f8_e5m2 encoding: seeeeemm
        //                         |_-> quiet bit is msb of mantissa
        const uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e5m2_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));
        bfloat16_t x16 = x8;
        float8_e5m2_t y8 = x16;

        // if x8 is an sNaN the conversion sets the quiet bit (msb of mantissa)
        const bool is_x8_snan = impl::utils::one_of(u8, 0x7d, 0xfd);
        const uint8_t y8_expect = is_x8_snan ? u8 | 0x02 : u8;

        ASSERT_EQ(y8_expect, bit_cast<uint8_t>(y8));
    });
}

TEST(test_ref_float8_conversions, f8_e4m3_to_bf16) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e4m3),
            "Engine does not support this data type.");
    // check all 256 f8_e4m3 values
    impl::parallel_nd(0xff, [&](uint16_t u16) {
        const uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e4m3_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));

        // convert f8_e4m3 to bf16 and back again,
        // expecting bitwise idendical values.
        // Note: f8_e4m3 does not have sNaN values, so no need to set quiet bit
        bfloat16_t x16 = x8;
        float8_e4m3_t y8 = x16;
        const uint8_t y8_expect = u8;
        ASSERT_EQ(y8_expect, bit_cast<uint8_t>(y8))
                << std::hex << std::endl
                << "u8 = " << static_cast<uint32_t>(u8) << std::endl
                << "x8.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(x8)) << std::endl
                << "y8.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(y8)) << std::endl
                << "y8_expect = " << static_cast<uint32_t>(y8_expect)
                << std::endl
                << std::dec;
    });
}

TEST(test_ref_float8_conversions, f32_to_f8_e4m3) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e4m3),
            "Engine does not support this data type.");
    // check all 2^32 f32 values
    impl::parallel_nd(0x100000000, [=](int64_t s64) {
        uint32_t u32 = static_cast<uint32_t>(s64);
        float x32 = bit_cast<float>(u32);
        ASSERT_EQ(u32, bit_cast<uint32_t>(x32));

        float16_t x16(x32);
        float8_e4m3_t x8_from_x16(x16);
        float8_e4m3_t x8_from_x32(x32);

        // check for double rounding
        ASSERT_EQ(x8_from_x16.raw_bits_, x8_from_x32.raw_bits_)
                << std::hex << std::endl
                << "x32 (raw bits) = " << bit_cast<uint32_t>(x32) << std::endl
                << "x16 (raw bits) = " << bit_cast<uint16_t>(x16) << std::endl
                << "x8_from_x16 (raw bits) = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(x8_from_x16))
                << std::endl
                << "x8_from_x32 (raw_bits) = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(x8_from_x32))
                << std::endl
                << std::dec;
    });
}

#if DNNL_X64
template <typename kernel_t>
void test_conv_fp8_to_f16() {
    using namespace impl;
    using namespace impl::cpu::x64;
    SKIP_IF(!impl::cpu::x64::mayiuse(avx512_core_fp16),
            "Engine does not support this ISA.");

    for (auto dt : {data_type::f8_e5m2, data_type::f8_e4m3}) {
        typename kernel_t::cfg_t cfg;
        cfg.isa = avx512_core_fp16;
        cfg.src_dt = dt;
        cfg.dst_dt = data_type::f16;
        kernel_t kernel(cfg);
        ASSERT_EQ(status::success, kernel.create_kernel());

        for (size_t count = 0; count <= 257; ++count) {
            std::vector<char> src(count);
            constexpr uint16_t sentinel = 0x1234;
            std::vector<uint16_t> dst(count + 2, sentinel);
            for (size_t i = 0; i < count; ++i)
                src[i] = static_cast<char>((i + count) % 256);
            typename kernel_t::ctx_t p;
            p.src = src.data();
            p.dst = reinterpret_cast<char *>(dst.data() + 1);
            p.nelems = count;
            kernel(&p);
            EXPECT_EQ(sentinel, dst.front());
            EXPECT_EQ(sentinel, dst.back());
            for (size_t i = 0; i < count; ++i) {
                const auto bits = static_cast<uint8_t>(src[i]);
                const float expected_f32 = dt == data_type::f8_e5m2
                        ? static_cast<float>(float8_e5m2_t(bits, true))
                        : static_cast<float>(float8_e4m3_t(bits, true));
                const float16_t expected(expected_f32);
                EXPECT_EQ(expected.raw, dst[i + 1])
                        << "count=" << count << " index=" << i;
            }
        }
    }
}

TEST(test_jit_float8_conversions, conv_weights_to_f16) {
    test_conv_fp8_to_f16<impl::cpu::x64::jit_brgemm_conv_fp8_cvt_kernel_t>();
}

TEST(test_jit_float8_conversions, conv_src_to_f16) {
    test_conv_fp8_to_f16<impl::cpu::x64::jit_brgemm_conv_src_copy_kernel_t>();
}

TEST(test_jit_float8_conversions, f8_e5m2_to_f32) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e5m2),
            "Engine does not support this data type.");

    // check all 2^8 fp8 values
    impl::parallel_nd(0xff, [=](uint16_t u16) {
        // convert from f8_e5m2 to f32 using ref and jit converters
        uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e5m2_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));

        float16_t ref_x16 = x8;
        float16_t jit_x16;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f8_e5m2_to_f16(&jit_x16, &x8));
        // expect bitwise identical results
        ASSERT_EQ(bit_cast<uint16_t>(ref_x16), bit_cast<uint16_t>(jit_x16))
                << std::hex << std::endl
                << "x8.raw_bits_ = " << static_cast<uint32_t>(u8) << std::endl
                << "ref_x16.raw_bits_ = " << bit_cast<uint16_t>(ref_x16)
                << std::endl
                << "jit_x16.raw_bits_ = " << bit_cast<uint16_t>(jit_x16)
                << std::endl
                << std::dec;

        float ref_x32 = static_cast<float>(x8);
        float jit_x32;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f8_e5m2_to_f32(&jit_x32, &x8));

        // expect bitwise identical results
        ASSERT_EQ(bit_cast<uint32_t>(ref_x32), bit_cast<uint32_t>(jit_x32))
                << std::hex << std::endl
                << "x8.raw_bits_ = " << static_cast<uint32_t>(u8) << std::endl
                << "ref_x32.raw_bits_ = " << bit_cast<uint32_t>(ref_x32)
                << std::endl
                << "jit_x32.raw_bits_ = " << bit_cast<uint32_t>(jit_x32)
                << std::endl
                << std::dec;
    });
}

TEST(test_jit_float8_conversions, f8_e4m3_to_f32) {
    SKIP_IF(!impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e4m3),
            "Engine does not support this data type.");

    // check all 2^8 fp8 values
    impl::parallel_nd(0xff, [=](uint16_t u16) {
        // convert from f8_e4m3 to f32 using ref and jit converters
        uint8_t u8 = static_cast<uint8_t>(u16);
        constexpr bool is_bitcast = true;
        float8_e4m3_t x8(u8, is_bitcast);
        ASSERT_EQ(u8, x8.raw_bits_);
        ASSERT_EQ(u8, bit_cast<uint8_t>(x8));

        float16_t ref_x16 = x8;
        float16_t jit_x16;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f8_e4m3_to_f16(&jit_x16, &x8));
        // expect bitwise identical results
        ASSERT_EQ(bit_cast<uint16_t>(ref_x16), bit_cast<uint16_t>(jit_x16))
                << std::hex << std::endl
                << "x8.raw_bits_ = " << static_cast<uint32_t>(u8) << std::endl
                << "ref_x16.raw_bits_ = " << bit_cast<uint16_t>(ref_x16)
                << std::endl
                << "jit_x16.raw_bits_ = " << bit_cast<uint16_t>(jit_x16)
                << std::endl
                << std::dec;

        float ref_x32 = static_cast<float>(x8);
        float jit_x32;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f8_e4m3_to_f32(&jit_x32, &x8));
        // expect bitwise identical results
        ASSERT_EQ(bit_cast<uint32_t>(ref_x32), bit_cast<uint32_t>(jit_x32))
                << std::hex << std::endl
                << "x8.raw_bits_ = " << static_cast<uint32_t>(u8) << std::endl
                << "ref_x32.raw_bits_ = " << bit_cast<uint32_t>(ref_x32)
                << std::endl
                << "jit_x32.raw_bits_ = " << bit_cast<uint32_t>(jit_x32)
                << std::endl
                << std::dec;
    });
}

TEST(test_jit_float8_conversions, f16_to_fp8) {
    const bool is_fp8_supported = impl::utils::everyone_is(true,
            impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e5m2),
            impl::cpu::platform::has_data_type_support(
                    impl::data_type::f8_e4m3));
    SKIP_IF(!is_fp8_supported, "Engine does not support this data type.");

    // check all 2^16 f16 values
    impl::parallel_nd(0xffff, [=](uint16_t u32) {
        uint16_t u16 = static_cast<uint16_t>(u32);
        constexpr bool is_bitcast = true;
        auto x16 = float16_t(u16, is_bitcast);
        ASSERT_EQ(u16, x16.raw);
        ASSERT_EQ(u16, bit_cast<uint16_t>(x16));

        // convert from f16 to f8_e5m2 using ref and jit converters
        float8_e5m2_t ref_x8_e5m2 = x16;
        float8_e5m2_t jit_x8_e5m2;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f16_to_f8_e5m2(&jit_x8_e5m2, &x16));
        // expect bitwise identical results
        ASSERT_EQ(
                bit_cast<uint8_t>(ref_x8_e5m2), bit_cast<uint8_t>(jit_x8_e5m2))
                << std::hex << std::endl
                << "x16.raw_bits_ = " << static_cast<uint32_t>(u16) << std::endl
                << "ref_x8_e5m2.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(ref_x8_e5m2))
                << std::endl
                << "jit_x8_e5m2.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(jit_x8_e5m2))
                << std::endl
                << std::dec;

        // convert from f16 to f8_e4m3 using ref and jit converters
        float8_e4m3_t ref_x8_e4m3 = x16;
        float8_e4m3_t jit_x8_e4m3;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f16_to_f8_e4m3(&jit_x8_e4m3, &x16));
        // expect bitwise identical results
        ASSERT_EQ(
                bit_cast<uint8_t>(ref_x8_e4m3), bit_cast<uint8_t>(jit_x8_e4m3))
                << std::hex << std::endl
                << "x16.raw_bits_ = " << static_cast<uint32_t>(u16) << std::endl
                << "ref_x8_e4m3.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(ref_x8_e4m3))
                << std::endl
                << "jit_x8_e4m3.raw_bits_ = "
                << static_cast<uint32_t>(bit_cast<uint8_t>(jit_x8_e4m3))
                << std::endl
                << std::dec;
    });
}

TEST(test_f16_conversions, f16_to_f32) {
    SKIP_IF(!mayiuse(impl::cpu::x64::avx512_core_fp16),
            "Engine does not support this ISA.");

    // check all 2^16 f16 values
    impl::parallel_nd(0xffff, [=](uint16_t u32) {
        // convert from f16 to f32 using ref and jit converters
        uint16_t u16 = static_cast<uint16_t>(u32);
        constexpr bool is_bitcast = true;
        auto x16 = float16_t(u16, is_bitcast);
        ASSERT_EQ(u16, x16.raw);
        ASSERT_EQ(u16, bit_cast<uint16_t>(x16));
        float ref_x32 = static_cast<float>(x16);
        float jit_x32;
        // can only fail if data type not supported
        ASSERT_TRUE(impl::cpu::x64::try_cvt_f16_to_f32(&jit_x32, &x16));

        // expect bitwise identical results
        ASSERT_EQ(bit_cast<uint32_t>(ref_x32), bit_cast<uint32_t>(jit_x32))
                << std::hex << std::endl
                << "x16.raw_bits_ = " << static_cast<uint32_t>(u16) << std::endl
                << "ref_x32.raw_bits_ = " << bit_cast<uint32_t>(ref_x32)
                << std::endl
                << "jit_x32.raw_bits_ = " << bit_cast<uint32_t>(jit_x32)
                << std::endl
                << std::dec;
    });
}
#endif // DNNL_X64

} // namespace dnnl
