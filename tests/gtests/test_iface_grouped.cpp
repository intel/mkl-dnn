/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "dnnl_test_common.hpp"
#include "dnnl_test_macros.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using dt = memory::data_type;

class iface_grouped_test_t : public ::testing::Test {};

TEST(iface_grouped_test_t, TestGroupedMDCreation) {
    const int ngroups = 3;
    const int K = 256;

    // variable_dim_idx = 0
    ASSERT_NO_THROW(memory::desc::grouped({9, K}, dt::f32, 0, ngroups));

    ASSERT_NO_THROW(
            memory::desc::grouped({4, K}, dt::f16, 0, ngroups, dt::s32));

    // variable_dim_idx = 1
    ASSERT_NO_THROW(memory::desc::grouped({K, 9}, dt::f32, 1, ngroups));
}

TEST(iface_grouped_test_t, TestGroupedMDInvalidArgs) {
    const int ngroups = 2;
    const int M = 100, K = 256;

    // 3D is not supported
    EXPECT_THROW(memory::desc::grouped({4, K, 10}, dt::f32, 0, ngroups),
            dnnl::error);

    // Invalid group count: 0, negative
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, 0, 0), dnnl::error);
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, 0, -1), dnnl::error);

    // Invalid variable_dim_idx: out of range for a 2D tensor
    EXPECT_THROW(
            memory::desc::grouped({4, K}, dt::f32, -1, ngroups), dnnl::error);
    EXPECT_THROW(
            memory::desc::grouped({4, K}, dt::f32, 2, ngroups), dnnl::error);

    // Runtime dimensions not supported
    EXPECT_THROW(memory::desc::grouped(
                         {DNNL_RUNTIME_DIM_VAL, K}, dt::f32, 0, ngroups),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped(
                         {M, DNNL_RUNTIME_DIM_VAL}, dt::f32, 0, ngroups),
            dnnl::error);

    // Zero dimensions not allowed
    EXPECT_THROW(
            memory::desc::grouped({0, K}, dt::f32, 0, ngroups), dnnl::error);
    EXPECT_THROW(
            memory::desc::grouped({M, 0}, dt::f32, 0, ngroups), dnnl::error);

    // Valid offsets type: s32; invalid: f32
    ASSERT_NO_THROW(
            memory::desc::grouped({M, K}, dt::f32, 0, ngroups, dt::s32));
    EXPECT_THROW(memory::desc::grouped({M, K}, dt::f32, 0, ngroups, dt::f32),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMDQueries) {
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const memory::dims dims = {total_tokens, K};

    {
        const memory::data_type data_type = dt::f32;
        const memory::data_type offsets_dt = dt::s32;

        memory::desc md;
        ASSERT_NO_THROW(md = memory::desc::grouped(
                                dims, data_type, 0, ngroups, offsets_dt));

        // Basic queries
        ASSERT_EQ(md.get_dims(), dims);
        ASSERT_EQ(md.get_data_type(), data_type);
        ASSERT_EQ(md.get_data_type(0), data_type);
        ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);
        ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::grouped);
        ASSERT_EQ(md.get_data_type(1), offsets_dt);

        // Query nnz
        memory::dim nnz = md.get_nnz();
        ASSERT_EQ(nnz, total_tokens * K);

        // Query strides
        // Note, that this return empty vector since strides are not supported
        memory::dims strides = md.get_strides();
        ASSERT_TRUE(strides.empty());
    }

    {
        // variable_dim_idx = 1, dims = [K, total_tokens]
        const memory::dims dims_t = {K, total_tokens};
        memory::desc md;
        ASSERT_NO_THROW(md
                = memory::desc::grouped(dims_t, dt::f32, 1, ngroups, dt::s32));

        ASSERT_EQ(md.get_dims(), dims_t);
        ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::grouped);
        ASSERT_EQ(md.get_nnz(), K * total_tokens);
        ASSERT_EQ(md.get_size(0),
                K * total_tokens * memory::data_type_size(dt::f32));
        ASSERT_EQ(md.get_size(1), ngroups * memory::data_type_size(dt::s32));
    }
}

TEST(iface_grouped_test_t, TestGroupedMDComparison) {
    const int ngroups = 2;
    const int K = 256;

    memory::desc md1, md2;

    // equal descriptors
    ASSERT_NO_THROW(md1 = memory::desc::grouped({4, K}, dt::f32, 0, ngroups));
    ASSERT_NO_THROW(md2 = memory::desc::grouped({4, K}, dt::f32, 0, ngroups));
    ASSERT_EQ(md1, md2);

    // different data types
    ASSERT_NO_THROW(md1 = memory::desc::grouped({4, K}, dt::f32, 0, ngroups));
    ASSERT_NO_THROW(md2 = memory::desc::grouped({4, K}, dt::f16, 0, ngroups));
    ASSERT_NE(md1, md2);

    // different ngroups
    ASSERT_NO_THROW(md1 = memory::desc::grouped({4, K}, dt::f32, 0, ngroups));
    ASSERT_NO_THROW(md2 = memory::desc::grouped({4, K}, dt::f32, 0, 3));
    ASSERT_NE(md1, md2);
}

TEST(iface_grouped_test_t, TestGroupedMDSize) {
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped(
                            {total_tokens, K}, dt::f32, 0, ngroups, dt::s32));

    // Size of values buffer (buffer 0): total_tokens * K elements
    size_t ref_values_size = total_tokens * K * memory::data_type_size(dt::f32);
    ASSERT_EQ(md.get_size(0), ref_values_size);

    // Size of offsets buffer (buffer 1): ngroups
    size_t ref_offsets_size = ngroups * memory::data_type_size(dt::s32);
    ASSERT_EQ(md.get_size(1), ref_offsets_size);
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestGroupedMemoryCreation) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    int variable_dim_idx = 0;

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            variable_dim_idx, ngroups));

    memory mem;
    mem = memory(md, eng);

    // user provided buffers (2 buffers: values and offsets)
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);
        offsets = {0, 1, 4, 9};

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }

    // same, but skipping one group
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);
        offsets = {0, 1, 1, 9}; // skip group 2

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_grouped_test_t, TestGroupedMemorySetGetDataHandles) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    int variable_dim_idx = 0;

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            variable_dim_idx, ngroups));

    memory mem = memory(md, eng);

    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);

        ASSERT_NO_THROW(mem.set_data_handle(values.data(), 0));
        ASSERT_NO_THROW(mem.set_data_handle(offsets.data(), 1));

        ASSERT_EQ(mem.get_data_handle(0), values.data());
        ASSERT_EQ(mem.get_data_handle(1), offsets.data());
    }
}

TEST(iface_grouped_test_t, TestGroupedMemoryMapUnmap) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int K = 4;
    const int total_tokens = 3;
    int variable_dim_idx = 0;

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            variable_dim_idx, ngroups, dt::s32));

    const int total_elements = total_tokens * K;
    std::vector<float> values(total_elements);
    for (int i = 0; i < total_elements; i++)
        values[i] = static_cast<float>(i) * 0.5f;

    std::vector<int32_t> offsets = {1, 3};
    ASSERT_EQ(offsets[ngroups - 1], total_tokens);

    memory mem(md, eng, {values.data(), offsets.data()});

    float *mapped_values = nullptr;
    int32_t *mapped_offsets = nullptr;

    ASSERT_NO_THROW(mapped_values = mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_offsets = mem.map_data<int32_t>(1));

    for (size_t i = 0; i < values.size(); i++)
        ASSERT_EQ(values[i], mapped_values[i]);

    for (size_t i = 0; i < offsets.size(); i++)
        ASSERT_EQ(offsets[i], mapped_offsets[i]);

    ASSERT_NO_THROW(mem.unmap_data(mapped_values, 0));
    ASSERT_NO_THROW(mem.unmap_data(mapped_offsets, 1));
}

TEST(c_api_grouped_md, TestGroupedMDQueries) {
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const int total_elements = total_tokens * K;

    dnnl_memory_desc_t md = nullptr;
    dnnl_dims_t dims = {total_tokens, K};

    DNNL_CHECK(dnnl_memory_desc_create_with_grouped_encoding(
            &md, 2, dims, dnnl_f32, 0, ngroups, dnnl_s32));
    ASSERT_NE(md, nullptr);

    // Query all properties
    int ndims = -1;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims));
    EXPECT_EQ(ndims, 2);

    dnnl_data_type_t dtype = dnnl_data_type_undef;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_data_type, &dtype));
    EXPECT_EQ(dtype, dnnl_f32);

    dnnl_format_kind_t fmt_kind = dnnl_format_kind_undef;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_format_kind, &fmt_kind));
    EXPECT_EQ(fmt_kind, dnnl_format_kind_sparse);

    dnnl_sparse_encoding_t encoding = dnnl_sparse_encoding_undef;
    DNNL_CHECK(dnnl_memory_desc_query_v2(
            md, dnnl_query_sparse_encoding, 0, &encoding));
    EXPECT_EQ(encoding, dnnl_grouped);

    dnnl_dim_t nnz = -1;
    DNNL_CHECK(dnnl_memory_desc_query_v2(md, dnnl_query_nnz_s64, 0, &nnz));
    EXPECT_EQ(nnz, total_elements);

    size_t values_size = dnnl_memory_desc_get_size_v2(md, 0);
    EXPECT_EQ(values_size, total_elements * sizeof(float));

    size_t offsets_size = dnnl_memory_desc_get_size_v2(md, 1);
    EXPECT_EQ(offsets_size, (ngroups) * sizeof(int32_t));

    DNNL_CHECK(dnnl_memory_desc_destroy(md));
}

TEST(c_api_grouped_md, TestGroupedMDInvalidArgs) {
    dnnl_memory_desc_t md = nullptr;
    const int K = 256;
    dnnl_dims_t dims = {4, K};

    // 3D is not supported
    dnnl_dims_t dims_3d = {4, K, 10};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 3, dims_3d, dnnl_f32, 0, 2, dnnl_s32),
            dnnl_unimplemented);
    EXPECT_EQ(md, nullptr);

    // variable_dim_idx out of range
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 2, dims, dnnl_f32, -1, 2, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 2, dims, dnnl_f32, 2, 2, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // invalid group count
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 2, dims, dnnl_f32, 0, 0, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 2, dims, dnnl_f32, 0, -1, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // null pointer md
    dnnl_status_t status = dnnl_memory_desc_create_with_grouped_encoding(
            nullptr, 2, dims, dnnl_f32, 0, 2, dnnl_s32);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

TEST(iface_grouped_test_t, TestGroupedMatmulValidation) {
    engine eng = get_test_engine();
    const int ngroups = 3;
    const int M = 100, K = 256, N = 512;

    auto src_md = memory::desc::grouped({M, K}, dt::f32, 0, ngroups);
    auto dst_md = memory::desc::grouped({M, N}, dt::f32, 0, ngroups);
    auto wei_md
            = memory::desc({ngroups, K, N}, dt::f32, memory::format_tag::abc);

    // Invalid: 2D weights
    auto wei_2d = memory::desc({K, N}, dt::f32, memory::format_tag::ab);
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_md, wei_2d, dst_md), dnnl::error);

    // Invalid: mismatched ngroups
    auto dst_4groups = memory::desc::grouped({M, N}, dt::f32, 0, 4);
    auto wei_4groups
            = memory::desc({4, K, N}, dt::f32, memory::format_tag::abc);

    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_4groups),
            dnnl::error);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_4groups, dst_md),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMatmulPatterns) {
    engine eng = get_test_engine();

    const int ngroups = 2;
    const int M = 4, K = 8, N = 6;
    const auto abc = memory::format_tag::abc;

    // Shapes to mix and match:

    // 2D grouped, variable idx = 0
    auto src_grouped_var0 = memory::desc::grouped({M, K}, dt::f32, 0, ngroups);
    auto wei_grouped_var0 = memory::desc::grouped({K, N}, dt::f32, 0, ngroups);
    auto dst_grouped_var0 = memory::desc::grouped({M, N}, dt::f32, 0, ngroups);

    // 2D grouped, variable idx = 1
    auto src_grouped_var1 = memory::desc::grouped({M, K}, dt::f32, 1, ngroups);
    auto wei_grouped_var1 = memory::desc::grouped({K, N}, dt::f32, 1, ngroups);

    // Dense 3D shapes
    auto src_dense_3d = memory::desc({ngroups, M / ngroups, K}, dt::f32, abc);
    auto wei_dense_3d = memory::desc({ngroups, K, N}, dt::f32, abc);
    auto dst_dense_3d = memory::desc({ngroups, M, N}, dt::f32, abc);

    // Valid cases:

    // grouped(varM) * dense 3D -> grouped(varM) (MoE forward)
    EXPECT_NO_THROW(matmul::primitive_desc(
            eng, src_grouped_var0, wei_dense_3d, dst_grouped_var0));

    // grouped(varK) * grouped(varK) -> dense 3D (MoE backward)
    EXPECT_NO_THROW(matmul::primitive_desc(
            eng, src_grouped_var1, wei_grouped_var0, dst_dense_3d));

    // Unsupported cases:

    // grouped(varM) * dense 3D -> dense 3D
    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_grouped_var0, wei_dense_3d, dst_dense_3d),
            dnnl::error);

    // dense 3D * dense 3D -> grouped
    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_dense_3d, wei_dense_3d, dst_grouped_var0),
            dnnl::error);

    // grouped(varM) * grouped(varK) -> dense 3D
    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_grouped_var0, wei_grouped_var0, dst_dense_3d),
            dnnl::error);

    // grouped(varK) * grouped(varN) -> dense 3D
    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_grouped_var1, wei_grouped_var1, dst_dense_3d),
            dnnl::error);
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestMaxGroupMHint) {
    engine eng = get_test_engine();
    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    auto hint_md = memory::desc::host_scalar(dt::s32);
    memory hint_mem(hint_md, int32_t(3));

    // Grouped matmul: hint is accepted
    {
        const int ngroups = 2;
        const int M = 6, K = 4, N = 4;

        auto src_md = memory::desc::grouped({M, K}, dt::f32, 0, ngroups);
        auto wei_md = memory::desc(
                {ngroups, K, N}, dt::f32, memory::format_tag::abc);
        auto dst_md = memory::desc::grouped({M, N}, dt::f32, 0, ngroups);

        matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
        matmul prim(pd);
        stream strm(eng);

        std::vector<float> src_data(M * K, 1.f);
        std::vector<float> wei_data(ngroups * K * N, 1.f);
        std::vector<float> dst_data(M * N, 0.f);
        std::vector<int32_t> offsets = {3, 6};

        memory src_mem(src_md, eng, {src_data.data(), offsets.data()});
        memory wei_mem(wei_md, eng, wei_data.data());
        memory dst_mem(dst_md, eng, {dst_data.data(), offsets.data()});

        EXPECT_NO_THROW(prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}));
        strm.wait();
    }

    // Non-grouped matmul
    {
        const int M = 4, K = 8, N = 6;
        auto src_md = memory::desc({M, K}, dt::f32, memory::format_tag::ab);
        auto wei_md = memory::desc({K, N}, dt::f32, memory::format_tag::ab);
        auto dst_md = memory::desc({M, N}, dt::f32, memory::format_tag::ab);

        matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
        matmul prim(pd);
        stream strm(eng);

        std::vector<float> src_data(M * K, 1.f);
        std::vector<float> wei_data(K * N, 1.f);
        std::vector<float> dst_data(M * N, 0.f);
        memory src_mem(src_md, eng, src_data.data());
        memory wei_mem(wei_md, eng, wei_data.data());
        memory dst_mem(dst_md, eng, dst_data.data());

        // Hint is unused and ignored
        EXPECT_NO_THROW(prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}));
        strm.wait();

        // Hint is passed as input instead of wei, so should be rejected
        EXPECT_THROW(prim.execute(strm,
                             {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                                     {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}),
                dnnl::error);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestBinaryPostOpPDCreation) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    auto src_md = memory::desc::grouped({6, 4}, dt::f32, 0, ngroups);
    auto wei_md
            = memory::desc({ngroups, 4, 4}, dt::f32, memory::format_tag::abc);
    auto dst_md = memory::desc::grouped({6, 4}, dt::f32, 0, ngroups);
    auto bin_md = memory::desc::grouped({6, 4}, dt::f32, 0, ngroups);

    post_ops po;
    po.append_binary(algorithm::binary_mul, bin_md);
    primitive_attr attr;
    attr.set_post_ops(po);

    ASSERT_NO_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));

    // Mismatched: 3 groups vs 2
    auto mismatched_bin_md = memory::desc::grouped({6, 4}, dt::f32, 0, 3);

    post_ops mismatched_po;
    mismatched_po.append_binary(algorithm::binary_mul, mismatched_bin_md);
    primitive_attr mismatched_attr;
    mismatched_attr.set_post_ops(mismatched_po);

    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_md, wei_md, dst_md, mismatched_attr),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestPostOps) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    auto src_md = memory::desc::grouped({6, 4}, dt::f32, 0, ngroups);
    auto wei_md
            = memory::desc({ngroups, 4, 4}, dt::f32, memory::format_tag::abc);
    auto dst_md = memory::desc::grouped({6, 4}, dt::f32, 0, ngroups);
    auto bin_md = memory::desc({6, 4}, dt::f32, memory::format_tag::ab);

    // Any eltwise algorithm is accepted as an activation post-op
    for (auto alg : {algorithm::eltwise_relu, algorithm::eltwise_tanh,
                 algorithm::eltwise_elu, algorithm::eltwise_square,
                 algorithm::eltwise_abs, algorithm::eltwise_sqrt,
                 algorithm::eltwise_linear, algorithm::eltwise_soft_relu,
                 algorithm::eltwise_logistic, algorithm::eltwise_exp,
                 algorithm::eltwise_gelu_tanh, algorithm::eltwise_gelu_erf,
                 algorithm::eltwise_log, algorithm::eltwise_clip,
                 algorithm::eltwise_clip_v2, algorithm::eltwise_pow,
                 algorithm::eltwise_round, algorithm::eltwise_mish,
                 algorithm::eltwise_hardswish,
                 algorithm::eltwise_hardsigmoid}) {
        post_ops po;
        po.append_eltwise(alg, 0.f, 1.f);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // eltwise_swish + binary_mul is accepted
    {
        post_ops po;
        po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
        po.append_binary(algorithm::binary_mul, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // binary_add is not supported for grouped matmul
    {
        post_ops po;
        po.append_binary(algorithm::binary_add, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }

    // sum is not supported
    {
        post_ops po;
        po.append_sum(1.0f);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_grouped_test_t, TestBinaryPostOpRejectsMismatchedOffsets) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int total_m = 6;
    const int K = 4;
    const int N = 4;

    auto src_md = memory::desc::grouped({total_m, K}, dt::f32, 0, ngroups);
    auto wei_md
            = memory::desc({ngroups, K, N}, dt::f32, memory::format_tag::abc);
    auto dst_md = memory::desc::grouped({total_m, N}, dt::f32, 0, ngroups);
    auto bin_md = memory::desc::grouped({total_m, N}, dt::f32, 0, ngroups);

    post_ops po;
    po.append_binary(algorithm::binary_mul, bin_md);
    primitive_attr attr;
    attr.set_post_ops(po);

    matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr);
    matmul prim(pd);
    stream strm(eng);

    std::vector<float> src_data(total_m * K, 1.f);
    std::vector<float> wei_data(ngroups * K * N, 1.f);
    std::vector<float> dst_data(total_m * N, 0.f);
    std::vector<float> bin_data(total_m * N, 1.f);
    std::vector<int32_t> src_offsets = {3, 6};
    std::vector<int32_t> dst_offsets = {3, 6};
    std::vector<int32_t> bin_offsets = {2, 6};

    memory src_mem(src_md, eng, {src_data.data(), src_offsets.data()});
    memory wei_mem(wei_md, eng, wei_data.data());
    memory dst_mem(dst_md, eng, {dst_data.data(), dst_offsets.data()});
    memory bin_mem(bin_md, eng, {bin_data.data(), bin_offsets.data()});

    EXPECT_THROW(
            prim.execute(strm,
                    {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                            {DNNL_ARG_DST, dst_mem},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    bin_mem}}),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestBinaryPostOpDenseShapes) {
    engine eng = get_test_engine();

    const int ngroups = 4;
    const int total_m = 32;
    const int K = 64;
    const int N = 32;

    auto src_md = memory::desc::grouped({total_m, K}, dt::f32, 0, ngroups);
    auto wei_md
            = memory::desc({ngroups, K, N}, dt::f32, memory::format_tag::abc);
    auto dst_md = memory::desc::grouped({total_m, N}, dt::f32, 0, ngroups);

    // Scalar [1, 1] binary post-op should be rejected
    {
        auto scalar_md = memory::desc({1, 1}, dt::f32, memory::format_tag::ab);
        post_ops po;
        po.append_binary(algorithm::binary_mul, scalar_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }

    // Per-group [G, 1] binary post-op is supported
    {
        auto per_group_md
                = memory::desc({ngroups, 1}, dt::f32, memory::format_tag::ab);
        post_ops po;
        po.append_binary(algorithm::binary_mul, per_group_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // Per-row [total_M, 1] binary post-op is supported
    {
        auto per_row_md
                = memory::desc({total_m, 1}, dt::f32, memory::format_tag::ab);
        post_ops po;
        po.append_binary(algorithm::binary_mul, per_row_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }
}

} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
