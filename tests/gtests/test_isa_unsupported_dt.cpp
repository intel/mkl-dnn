/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "tests/test_isa_common.hpp"

namespace dnnl {

namespace {

using dt = memory::data_type;
using tag = memory::format_tag;

// The maximal ISA can be set only once and before any query of it, so cap it
// once for all the tests in this file. AVX2 lacks support for `bf16`.
bool cap_max_cpu_isa_to_avx2() {
    static const bool capped = mayiuse(cpu_isa::avx2, /* soft = */ true)
            && set_max_cpu_isa(cpu_isa::avx2) == status::success;
    return capped;
}

template <typename F>
dnnl_status_t get_status(const F &f) {
    try {
        f();
    } catch (const error &e) { return e.status; }
    return dnnl_success;
}

} // namespace

class unsupported_dt_test_t : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF(!cap_max_cpu_isa_to_avx2(),
                "Maximal CPU ISA can't be capped to AVX2.");
    }

    engine eng_ {engine::kind::cpu, 0};
};

TEST_F(unsupported_dt_test_t, TestMatmul) {
    const memory::dims dims = {16, 16};
    auto md = [&](dt data_type) {
        return memory::desc(dims, data_type, tag::ab);
    };

    ASSERT_EQ(get_status([&]() {
        matmul::primitive_desc(eng_, md(dt::f32), md(dt::f32), md(dt::f32));
    }),
            dnnl_success);
    ASSERT_EQ(get_status([&]() {
        matmul::primitive_desc(eng_, md(dt::bf16), md(dt::bf16), md(dt::bf16));
    }),
            dnnl_unsupported_data_type);
}

TEST_F(unsupported_dt_test_t, TestBinaryPostOp) {
    const memory::dims dims = {16, 16};
    auto md = [&](dt data_type) {
        return memory::desc(dims, data_type, tag::ab);
    };

    post_ops po;
    po.append_binary(algorithm::binary_add, md(dt::bf16));
    primitive_attr attr;
    attr.set_post_ops(po);

    ASSERT_EQ(get_status([&]() {
        matmul::primitive_desc(
                eng_, md(dt::f32), md(dt::f32), md(dt::f32), attr);
    }),
            dnnl_unsupported_data_type);
}

// Deconvolution is implemented through a nested convolution, which must pass
// the status through.
TEST_F(unsupported_dt_test_t, TestNestedDeconvolution) {
    auto md = [&](const memory::dims &dims) {
        return memory::desc(dims, dt::bf16, tag::any);
    };

    ASSERT_EQ(get_status([&]() {
        deconvolution_forward::primitive_desc(eng_,
                prop_kind::forward_inference, algorithm::deconvolution_direct,
                md({1, 16, 8, 8}), md({16, 16, 3, 3}), md({1, 16, 10, 10}),
                {1, 1}, {0, 0}, {0, 0});
    }),
            dnnl_unsupported_data_type);
}

// `f64` is not supported on any CPU, so it is not a hardware limitation.
TEST_F(unsupported_dt_test_t, TestF64IsUnimplemented) {
    const memory::desc md({16, 16}, dt::f64, tag::ab);

    ASSERT_EQ(get_status([&]() {
        eltwise_forward::primitive_desc(eng_, prop_kind::forward_inference,
                algorithm::eltwise_relu, md, md, 0.f);
    }),
            dnnl_unimplemented);
}

} // namespace dnnl
