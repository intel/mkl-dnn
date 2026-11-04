/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <memory>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

static constexpr float epsilon = 1e-5f;

struct test_lnorm_params_t {
    memory::format_tag src_tag;
    memory::format_tag stat_tag;
    memory::format_tag diff_src_tag;
    memory::data_type src_dt;
    memory::data_type dst_dt;
    memory::data_type diff_src_dt;
    memory::dims dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename T>
void fill(const memory &m) {
    auto numElements = m.get_desc().get_size() / sizeof(T);
    fill_data<T>(numElements, m);
}

class lnorm_test_t : public ::testing::TestWithParam<test_lnorm_params_t> {
private:
    std::shared_ptr<test_memory> src, dst, diff_src, diff_dst;
    memory weights, bias, diff_weights, diff_bias, mean, variance;

    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> dst_md;
    std::shared_ptr<memory::desc> stat_d;
    std::shared_ptr<memory::desc> diff_src_md;

    layer_normalization_forward::primitive_desc lnorm_fwd_pd;
    layer_normalization_backward::primitive_desc lnorm_bwd_pd;

    test_lnorm_params_t p;
    engine eng;
    stream strm;

protected:
    void SetUp() override {
        SKIP_IF_CUDA(true, "Layer normalization not supported by CUDA.");
        SKIP_IF_HIP(true, "Layer normalization not supported by HIP.");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        SKIP_IF(unsupported_data_type(p.src_dt)
                        || unsupported_data_type(p.dst_dt),
                "Engine does not support this data type.");
        if (p.diff_src_dt != memory::data_type::undef) {
            SKIP_IF(unsupported_data_type(p.diff_src_dt),
                    "Engine does not support this data type.");
        }

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        eng = get_test_engine();
        strm = make_stream(eng);

        src_md = std::make_shared<memory::desc>(p.dims, p.src_dt, p.src_tag);
        dst_md = std::make_shared<memory::desc>(p.dims, p.dst_dt, p.src_tag);
        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        stat_d = std::make_shared<memory::desc>(
                stat_dims, memory::data_type::f32, p.stat_tag);

        auto training = prop_kind::forward_training;
        auto inference = prop_kind::forward_inference;

        using flags = normalization_flags;
        Forward(training);
        Forward(training, flags::use_global_stats);
        Forward(training, flags::use_scale);
        Forward(training, flags::use_shift);
        Forward(training, flags::use_scale | flags::use_shift);
        Forward(training,
                flags::use_scale | flags::use_shift | flags::use_global_stats);

        // RMS normalization on GPU is currently limited to Intel GPUs only
        // todo: remove the vendor check once supported
        if (!is_nvidia_gpu(eng) && !is_amd_gpu(eng)) {
            Forward(training, flags::rms_norm);
            Forward(training,
                    flags::rms_norm | flags::use_scale | flags::use_shift);
            Forward(training,
                    flags::rms_norm | flags::use_scale | flags::use_shift
                            | flags::use_global_stats);
        }

        Forward(inference);
        Forward(inference, flags::use_global_stats);
        Forward(inference, flags::use_scale | flags::use_shift);

        // RMS normalization on GPU is currently limited to Intel GPUs only
        // todo: remove the vendor check once supported
        if (!is_nvidia_gpu(eng) && !is_amd_gpu(eng)) {
            Forward(inference, flags::rms_norm);
            Forward(inference,
                    flags::rms_norm | flags::use_scale | flags::use_shift);
            Forward(inference,
                    flags::rms_norm | flags::use_scale | flags::use_shift
                            | flags::use_global_stats);
        }

        if (!impl::utils::one_of(p.dst_dt, memory::data_type::f16,
                    memory::data_type::s8, memory::data_type::u8)) {
            diff_src_md = std::make_shared<memory::desc>(
                    p.dims, p.diff_src_dt, p.diff_src_tag);

            Backward(prop_kind::backward_data);
            Backward(prop_kind::backward_data, flags::use_global_stats);
            Backward(prop_kind::backward, flags::use_scale);
            Backward(prop_kind::backward, flags::use_shift);
            Backward(prop_kind::backward, flags::use_scale | flags::use_shift);
            Backward(prop_kind::backward,
                    flags::use_scale | flags::use_shift
                            | flags::use_global_stats);

            // RMS normalization on GPU is currently limited to Intel GPUs only
            // todo: remove the vendor check once supported
            if (!is_nvidia_gpu(eng) && !is_amd_gpu(eng)) {
                Backward(prop_kind::backward, flags::rms_norm);
                Backward(prop_kind::backward,
                        flags::rms_norm | flags::use_scale | flags::use_shift);
                Backward(prop_kind::backward,
                        flags::rms_norm | flags::use_scale | flags::use_shift
                                | flags::use_global_stats);
            }
        }
    }

    void Forward(prop_kind pk,
            normalization_flags flags = normalization_flags::none) {
        // layer normalization specific types and values
        using pd_t = layer_normalization_forward::primitive_desc;

        lnorm_fwd_pd = pd_t(eng, pk, *src_md, *dst_md, *stat_d, epsilon, flags);
        lnorm_fwd_pd = pd_t(eng, pk, *src_md, *dst_md, *stat_d,
                memory::data_type::f32, epsilon, flags);
        lnorm_fwd_pd
                = pd_t(lnorm_fwd_pd.get()); // test construction from a C pd
        bool is_int8 = impl::utils::one_of(src_md->get_data_type(),
                               memory::data_type::s8, memory::data_type::u8)
                || impl::utils::one_of(dst_md->get_data_type(),
                        memory::data_type::s8, memory::data_type::u8);

        allows_attr_t aa {};
        aa.po_eltwise = true;
        aa.po_binary = true;

        if (is_int8) aa.scales = true;

        // test all pd ctors
        test_fwd_pd_constructors<pd_t>(lnorm_fwd_pd, aa, pk, *src_md, *dst_md,
                *stat_d, epsilon, flags);

        fwd_iface_test_stat_any(pk, flags);

        bool useScale = (bool)(flags & normalization_flags::use_scale);
        bool useShift = (bool)(flags & normalization_flags::use_shift);
        bool useGlobalStats
                = (bool)(flags & normalization_flags::use_global_stats);
        bool skipMean = (bool)(flags & normalization_flags::rms_norm);
        bool isTraining = pk == prop_kind::forward_training;

        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lnorm_fwd_pd.src_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == lnorm_fwd_pd.dst_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN)
                == lnorm_fwd_pd.mean_desc());
        ASSERT_TRUE(lnorm_fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == lnorm_fwd_pd.variance_desc());
        if (p.src_tag != memory::format_tag::any) {
            ASSERT_TRUE(*src_md == lnorm_fwd_pd.src_desc());
        }

        ASSERT_EQ(lnorm_fwd_pd.get_prop_kind(), pk);
        ASSERT_EQ(lnorm_fwd_pd.get_epsilon(), epsilon);
        ASSERT_EQ(lnorm_fwd_pd.get_flags(), flags);

        src = std::make_shared<test_memory>(lnorm_fwd_pd.src_desc(), eng);
        dst = std::make_shared<test_memory>(lnorm_fwd_pd.dst_desc(), eng);

        if (useScale) {
            auto weights_md = lnorm_fwd_pd.weights_desc();
            weights = test::make_memory(weights_md, eng);
            ASSERT_TRUE(weights_md.get_data_type() == memory::data_type::f32);
        }
        if (useShift) {
            auto weights_md = lnorm_fwd_pd.weights_desc();
            bias = test::make_memory(weights_md, eng);
            ASSERT_TRUE(weights_md.get_data_type() == memory::data_type::f32);
        }
        if (isTraining || useGlobalStats) {
            mean = test::make_memory(*stat_d, eng);
            variance = test::make_memory(*stat_d, eng);
        }

        fill<float>(src->get());
        fill<float>(dst->get());
        if (useScale) fill<float>(weights);
        if (useShift) fill<float>(bias);
        if (useGlobalStats) {
            fill<float>(mean);
            fill<float>(variance);
        }

        execlnormFwd(isTraining, useGlobalStats, useScale, useShift, skipMean);
    }

    void Backward(prop_kind pk,
            normalization_flags flags = normalization_flags::none) {
        bwd_iface_test_stat_any(pk, flags);

        bool useScale = (bool)(flags & normalization_flags::use_scale);
        bool useShift = (bool)(flags & normalization_flags::use_shift);
        bool skipMean = (bool)(flags & normalization_flags::rms_norm);

        lnorm_fwd_pd = layer_normalization_forward::primitive_desc(eng,
                prop_kind::forward_training, *src_md, *dst_md, *stat_d, epsilon,
                flags);

        lnorm_bwd_pd = layer_normalization_backward::primitive_desc(eng, pk,
                *diff_src_md, *dst_md, *src_md, *stat_d, epsilon, flags,
                lnorm_fwd_pd);

        lnorm_bwd_pd = layer_normalization_backward::primitive_desc(eng, pk,
                *diff_src_md, *dst_md, *src_md, *stat_d, memory::data_type::f32,
                memory::data_type::f32, epsilon, flags, lnorm_fwd_pd);

        lnorm_bwd_pd = layer_normalization_backward::primitive_desc(
                lnorm_bwd_pd.get()); // test construction from a C pd

        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lnorm_bwd_pd.src_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == lnorm_bwd_pd.diff_src_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == lnorm_bwd_pd.diff_dst_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_MEAN)
                == lnorm_bwd_pd.mean_desc());
        ASSERT_TRUE(lnorm_bwd_pd.query_md(query::exec_arg_md, DNNL_ARG_VARIANCE)
                == lnorm_bwd_pd.variance_desc());
        if (p.diff_src_tag != memory::format_tag::any) {
            ASSERT_TRUE(*diff_src_md == lnorm_bwd_pd.diff_src_desc());
        }

        ASSERT_EQ(lnorm_bwd_pd.get_prop_kind(), pk);
        ASSERT_EQ(lnorm_bwd_pd.get_epsilon(), epsilon);
        ASSERT_EQ(lnorm_bwd_pd.get_flags(), flags);

        diff_src = std::make_shared<test_memory>(
                lnorm_bwd_pd.diff_src_desc(), eng);
        diff_dst = std::make_shared<test_memory>(
                lnorm_bwd_pd.diff_dst_desc(), eng);

        if (useScale) {
            auto weights_md = lnorm_bwd_pd.weights_desc();
            weights = test::make_memory(weights_md, eng);
            ASSERT_TRUE(weights_md.get_data_type() == memory::data_type::f32);
        }
        if (useShift) {
            auto weights_md = lnorm_bwd_pd.weights_desc();
            bias = test::make_memory(weights_md, eng);
            ASSERT_TRUE(weights_md.get_data_type() == memory::data_type::f32);
        }
        if (useScale) {
            auto diff_weights_md = lnorm_bwd_pd.diff_weights_desc();
            diff_weights = test::make_memory(diff_weights_md, eng);
            ASSERT_TRUE(
                    diff_weights_md.get_data_type() == memory::data_type::f32);
        }
        if (useShift) {
            auto diff_weights_md = lnorm_bwd_pd.diff_weights_desc();
            diff_bias = test::make_memory(diff_weights_md, eng);
            ASSERT_TRUE(
                    diff_weights_md.get_data_type() == memory::data_type::f32);
        }
        mean = test::make_memory(*stat_d, eng);
        variance = test::make_memory(*stat_d, eng);

        if (useScale) fill<float>(weights);
        if (useShift) fill<float>(bias);
        fill<float>(diff_src->get());
        fill<float>(diff_dst->get());
        fill<float>(mean);
        fill<float>(variance);

        execlnormBwd(useScale, useShift, skipMean, pk);
    }

    void execlnormFwd(bool isTraining, bool useGlobalStats, bool useScale,
            bool useShift, bool skipMean) {
        std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, src->get()},
                {DNNL_ARG_DST, dst->get()},
        };

        if (useScale) args.insert({DNNL_ARG_SCALE, weights});
        if (useShift) args.insert({DNNL_ARG_SHIFT, bias});

        if (isTraining || useGlobalStats) {
            if (!skipMean) args.insert({DNNL_ARG_MEAN, mean});
            args.insert({DNNL_ARG_VARIANCE, variance});
        }

        EXPECT_ANY_THROW(layer_normalization_forward(lnorm_fwd_pd, {}));
        layer_normalization_forward prim(lnorm_fwd_pd);
        prim.execute(strm, args);
        strm.wait();
    }

    void execlnormBwd(
            bool useScale, bool useShift, bool skipMean, prop_kind pk) {
        std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, src->get()},
                {DNNL_ARG_DIFF_DST, dst->get()},
                {DNNL_ARG_VARIANCE, variance},
                {DNNL_ARG_DIFF_SRC, diff_src->get()},
        };

        if (!skipMean) { args.insert({DNNL_ARG_MEAN, mean}); }

        if (useScale) {
            args.insert({DNNL_ARG_SCALE, weights});
            if (pk == prop_kind::backward)
                args.insert({DNNL_ARG_DIFF_SCALE, diff_weights});
        }

        if (useShift) {
            args.insert({DNNL_ARG_SHIFT, bias});
            if (pk == prop_kind::backward)
                args.insert({DNNL_ARG_DIFF_SHIFT, diff_bias});
        }

        EXPECT_ANY_THROW(layer_normalization_backward(lnorm_bwd_pd, {}));
        layer_normalization_backward prim(lnorm_bwd_pd);
        prim.execute(strm, args);
        strm.wait();
    }

    void fwd_iface_test_stat_any(prop_kind pk, normalization_flags flags) {
        // non stats if inference w/o use global stats
        if (pk == prop_kind::forward_inference
                && !(bool)(flags & normalization_flags::use_global_stats))
            return;

        using tag = memory::format_tag;

        tag expect_stat_tag = derive_stat_tag();
        if (expect_stat_tag == tag::undef) return; // optimism

        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        memory::desc expect_stat_md(
                stat_dims, memory::data_type::f32, expect_stat_tag);

        // no stat_md provided at all
        {
            layer_normalization_forward::primitive_desc fwd_pd(
                    eng, pk, *src_md, *dst_md, epsilon, flags);

            EXPECT_EQ(fwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(fwd_pd.variance_desc(), expect_stat_md);
        }

        // stat_md with format_tag::any
        {
            memory::desc any_stat_md(
                    stat_dims, memory::data_type::f32, tag::any);
            layer_normalization_forward::primitive_desc fwd_pd(
                    eng, pk, *src_md, *dst_md, any_stat_md, epsilon, flags);

            EXPECT_EQ(fwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(fwd_pd.variance_desc(), expect_stat_md);
        }
    }

    void bwd_iface_test_stat_any(prop_kind pk, normalization_flags flags) {
        using tag = memory::format_tag;

        tag expect_stat_tag = derive_stat_tag();
        if (expect_stat_tag == tag::undef) return; // optimism

        memory::dims stat_dims(p.dims.begin(), p.dims.end() - 1);
        memory::desc expect_stat_md(
                stat_dims, memory::data_type::f32, expect_stat_tag);

        layer_normalization_forward::primitive_desc fwd_pd(eng,
                prop_kind::forward_training, *src_md, *dst_md, epsilon, flags);

        // stat_md with format_tag::any
        {
            memory::desc any_stat_md(
                    stat_dims, memory::data_type::f32, tag::any);
            layer_normalization_backward::primitive_desc bwd_pd(eng, pk,
                    *diff_src_md, *dst_md, *src_md, any_stat_md, epsilon, flags,
                    fwd_pd);

            EXPECT_EQ(bwd_pd.mean_desc(), expect_stat_md);
            EXPECT_EQ(bwd_pd.variance_desc(), expect_stat_md);
        }
    }

private:
    memory::format_tag derive_stat_tag() const {
        using tag = memory::format_tag;
        tag expect_stat_tag = tag::undef;

        // TODO: add more cases and test cases
        // XXX: currently test only simple cases like `abc`, `acb`. Extend,
        //      if possible, to blocked formats too.
        switch (p.src_tag) {
            case tag::abc: expect_stat_tag = tag::ab; break;
            case tag::bac: expect_stat_tag = tag::ba; break;
            default: break;
        }

        return expect_stat_tag;
    }
};

#define EXPAND_FORMATS(src, stat, diff_src) \
    memory::format_tag::src, memory::format_tag::stat, \
            memory::format_tag::diff_src

#define EXPAND_DTS(src, dst, diff_src) \
    memory::data_type::src, memory::data_type::dst, memory::data_type::diff_src

#define TAGS_NC EXPAND_FORMATS(ab, a, ab)
#define TAGS_TNC EXPAND_FORMATS(abc, ab, abc)
#define TAGS_cTNC EXPAND_FORMATS(abc, ba, abc)
#define TAGS_NTC EXPAND_FORMATS(bac, ba, bac)
#define TAGS_LDSNC EXPAND_FORMATS(abcde, abcd, abcde)
#define TAGS_cLDSNC EXPAND_FORMATS(abcde, acdb, abcde)

#define LNORM_TEST_CASE(...) \
    test_lnorm_params_t { \
        __VA_ARGS__, false, dnnl_success \
    }

static auto expected_failure_cases = []() {
    // clang-format off
    return ::testing::Values(
        // Negative dimension
        test_lnorm_params_t {TAGS_NC, EXPAND_DTS(f32, f32, f32), {-1, 10}, true, dnnl_invalid_arguments},
        // Undef data type
        test_lnorm_params_t {TAGS_NC, EXPAND_DTS(undef, f32, f32), {1, 10}, true, dnnl_invalid_arguments},
        // Only `any` tags
        test_lnorm_params_t {EXPAND_FORMATS(any, any, any), EXPAND_DTS(f32, f32, f32), {1, 10}, true, dnnl_invalid_arguments}
    );
    // clang-format on
};

static auto zero_dim_cases
        = [](memory::data_type src_dt, memory::data_type dst_dt,
                  memory::data_type diff_src_dt) {
    // clang-format off
    return ::testing::Values(
        LNORM_TEST_CASE(TAGS_NC, src_dt, dst_dt, diff_src_dt, {0, 100}),
        LNORM_TEST_CASE(TAGS_TNC, src_dt, dst_dt, diff_src_dt, {6, 0, 8}),
        LNORM_TEST_CASE(TAGS_NTC, src_dt, dst_dt, diff_src_dt, {6, 32, 0}),
        LNORM_TEST_CASE(TAGS_LDSNC, src_dt, dst_dt, diff_src_dt, {6, 2, 2, 32, 0})
    );
    // clang-format on
};

static auto simple_cases
        = [](memory::data_type src_dt, memory::data_type dst_dt,
                  memory::data_type diff_src_dt) {
    // clang-format off
    return ::testing::Values(
        LNORM_TEST_CASE(TAGS_NC, src_dt, dst_dt, diff_src_dt, {1, 100}),
        LNORM_TEST_CASE(TAGS_NC, src_dt, dst_dt, diff_src_dt, {20, 8}),
        LNORM_TEST_CASE(TAGS_NC, src_dt, dst_dt, diff_src_dt, {2, 10}),
        LNORM_TEST_CASE(TAGS_TNC, src_dt, dst_dt, diff_src_dt, {6, 32, 8}),
        LNORM_TEST_CASE(TAGS_TNC, src_dt, dst_dt, diff_src_dt, {2, 8, 16}),
        LNORM_TEST_CASE(TAGS_TNC, src_dt, dst_dt, diff_src_dt, {2, 10, 4}),
        LNORM_TEST_CASE(TAGS_cTNC, src_dt, dst_dt, diff_src_dt, {6, 32, 8}),
        LNORM_TEST_CASE(TAGS_cTNC, src_dt, dst_dt, diff_src_dt, {2, 8, 16}),
        LNORM_TEST_CASE(TAGS_cTNC, src_dt, dst_dt, diff_src_dt, {2, 10, 4}),
        LNORM_TEST_CASE(TAGS_NTC, src_dt, dst_dt, diff_src_dt, {64, 32, 8}),
        LNORM_TEST_CASE(TAGS_NTC, src_dt, dst_dt, diff_src_dt, {12, 8, 16}),
        LNORM_TEST_CASE(TAGS_NTC, src_dt, dst_dt, diff_src_dt, {32, 10, 4}),
        LNORM_TEST_CASE(TAGS_LDSNC, src_dt, dst_dt, diff_src_dt, {6, 2, 2, 32, 8}),
        LNORM_TEST_CASE(TAGS_LDSNC, src_dt, dst_dt, diff_src_dt, {2, 2, 2, 8, 16}),
        LNORM_TEST_CASE(TAGS_LDSNC, src_dt, dst_dt, diff_src_dt, {2, 2, 2, 10, 4}),
        LNORM_TEST_CASE(TAGS_cLDSNC, src_dt, dst_dt, diff_src_dt, {6, 2, 2, 32, 8}),
        LNORM_TEST_CASE(TAGS_cLDSNC, src_dt, dst_dt, diff_src_dt, {2, 2, 2, 8, 16}),
        LNORM_TEST_CASE(TAGS_cLDSNC, src_dt, dst_dt, diff_src_dt, {2, 2, 2, 10, 4})
    );
    // clang-format on
};

TEST_P(lnorm_test_t, TestsLnormV2) {}

#define INST_TEST_CASE(name, ...) \
    INSTANTIATE_TEST_SUITE_P(name, lnorm_test_t, simple_cases(__VA_ARGS__));

#define CPU_INST_TEST_CASE(name, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(name, lnorm_test_t, simple_cases(__VA_ARGS__));

#define GPU_INST_TEST_CASE(name, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P(name, lnorm_test_t, simple_cases(__VA_ARGS__));

INSTANTIATE_TEST_SUITE_P(LnormEF, lnorm_test_t, expected_failure_cases());
INSTANTIATE_TEST_SUITE_P(
        LnormZeroDim, lnorm_test_t, zero_dim_cases(EXPAND_DTS(f32, f32, f32)));

INST_TEST_CASE(LnormSimpleF32, EXPAND_DTS(f32, f32, f32))
INST_TEST_CASE(LnormSimpleBF16, EXPAND_DTS(bf16, bf16, bf16))
INST_TEST_CASE(LnormSimpleF16, EXPAND_DTS(f16, f16, undef))
CPU_INST_TEST_CASE(LnormSimpleF32BF16, EXPAND_DTS(f32, bf16, f32))
CPU_INST_TEST_CASE(LnormSimpleBF16F32, EXPAND_DTS(bf16, f32, bf16))
CPU_INST_TEST_CASE(LnormSimpleF32S8, EXPAND_DTS(f32, s8, undef))
CPU_INST_TEST_CASE(LnormSimpleBF16U8, EXPAND_DTS(bf16, u8, undef))

} // namespace dnnl
