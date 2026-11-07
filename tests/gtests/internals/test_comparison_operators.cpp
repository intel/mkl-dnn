/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "tests/gtests/dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

#include "common/opdesc.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {

namespace {
bool compare(const dnnl::primitive_attr &lhs, const dnnl::primitive_attr &rhs) {
    return *lhs.get() == *rhs.get();
}

bool self_compare(const dnnl::primitive_attr &attr) {
    return *attr.get() == *attr.get();
}

template <typename T>
bool self_compare(const T &desc) {
    return dnnl::impl::operator==(desc, desc);
}

} // namespace

#define TEST_SELF_COMPARISON(v) ASSERT_EQ(true, self_compare(v))

class comparison_operators_t : public ::testing::Test {};

TEST(comparison_operators_t, TestAttrScales) {
    dnnl::primitive_attr default_attr;

    const std::vector<int> supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    for (auto arg : supported_args) {
        dnnl::primitive_attr attr;
        attr.set_scales_mask(arg, 0);
        ASSERT_EQ(compare(default_attr, attr), false);
    }
}

TEST(comparison_operators_t, TestAttrZeroPoints) {
    dnnl::primitive_attr default_attr;

    const std::vector<int> supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    for (auto arg : supported_args) {
        dnnl::primitive_attr attr;
        attr.set_zero_points_mask(arg, 0);
        ASSERT_EQ(compare(default_attr, attr), false);
    }
}

TEST(comparison_operators_t, TestAttrDataQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_data_qparams(1.5f, NAN);
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestAttrWeightsQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_weights_qparams(0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_rnn_weights_qparams(1 << 1, {1.5f, NAN, 3.5f});
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(
        comparison_operators_t, TestAttrWeightsProjectionQparams) {
    dnnl::primitive_attr attr;

    attr.set_rnn_weights_projection_qparams(0, {NAN});
    TEST_SELF_COMPARISON(attr);

    attr.set_rnn_weights_projection_qparams(1 << 1, {1.5f, NAN, 3.5f});
    TEST_SELF_COMPARISON(attr);
}

TEST(comparison_operators_t, TestSumPostOp) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    ops.append_sum(NAN);
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestDepthwisePostOp) {
    dnnl::primitive_attr attr;
    dnnl::post_ops ops;

    ops.append_dw(memory::data_type::s8, memory::data_type::f32,
            memory::data_type::u8, 3, 1, 1);
    attr.set_post_ops(ops);
    TEST_SELF_COMPARISON(attr);
}

HANDLE_EXCEPTIONS_FOR_TEST(comparison_operators_t, TestBinarySelectPostOp) {
    const memory::dims dims = {2, 3};
    memory::desc src1_md(dims, memory::data_type::f32, memory::format_tag::ab);
    memory::desc src2_full_md(
            dims, memory::data_type::s8, memory::format_tag::ab);
    memory::desc src2_bcast_md(
            {2, 1}, memory::data_type::s8, memory::format_tag::ab);

    auto make_attr = [&](const memory::desc &src2_md) {
        post_ops ops;
        ops.append_binary(algorithm::binary_select, src1_md, src2_md);
        primitive_attr attr;
        attr.set_post_ops(ops);
        return attr;
    };

    // post-ops with binary_select must not compare equal when their
    // third operand descriptor differs (full vs broadcast dims).
    ASSERT_EQ(
            compare(make_attr(src2_full_md), make_attr(src2_bcast_md)), false);
}

TEST(comparison_operators_t, TestBatchNormDesc) {
    auto bnorm_desc = dnnl::impl::batch_normalization_desc_t();
    bnorm_desc.batch_norm_epsilon = NAN;
    TEST_SELF_COMPARISON(bnorm_desc);
}

TEST(comparison_operators_t, TestEltwiseDesc) {
    auto eltwise_desc = dnnl::impl::eltwise_desc_t();
    eltwise_desc.alpha = NAN;
    TEST_SELF_COMPARISON(eltwise_desc);
}

TEST(comparison_operators_t, TestLayerNormDesc) {
    auto lnorm_desc = dnnl::impl::layer_normalization_desc_t();
    lnorm_desc.layer_norm_epsilon = NAN;
    TEST_SELF_COMPARISON(lnorm_desc);
}

TEST(comparison_operators_t, TestLRNDesc) {
    auto lrn_desc = dnnl::impl::lrn_desc_t();
    lrn_desc.lrn_alpha = NAN;
    TEST_SELF_COMPARISON(lrn_desc);
}

TEST(comparison_operators_t, TestReductionDesc) {
    auto reduction_desc = dnnl::impl::reduction_desc_t();
    reduction_desc.p = NAN;
    TEST_SELF_COMPARISON(reduction_desc);
}

TEST(comparison_operators_t, TestResamplingDesc) {
    auto resampling_desc = dnnl::impl::resampling_desc_t();
    resampling_desc.factors[0] = NAN;
    TEST_SELF_COMPARISON(resampling_desc);
}

TEST(comparison_operators_t, TestRNNDesc) {
    auto rnn_desc = dnnl::impl::rnn_desc_t();
    rnn_desc.alpha = NAN;
    TEST_SELF_COMPARISON(rnn_desc);
}

TEST(comparison_operators_t, TestSumDesc) {
    float scales[2] = {NAN, 2.5f};
    dnnl::impl::memory_desc_t md {};
    dnnl_memory_desc_t mds[2] = {&md, &md};

    dnnl::impl::sum_desc_t sum_desc(
            dnnl::impl::primitive_kind::sum, &md, 2, scales, mds);
    TEST_SELF_COMPARISON(sum_desc);
}

} // namespace dnnl
