/*******************************************************************************
* Copyright 2022 Intel Corporation
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

namespace dnnl {

using dt = memory::data_type;

class iface_sparse_test_t : public ::testing::Test {};

TEST(iface_sparse_test_t, TestSparseMDCreation) {
    const int nnz = 12;
    memory::desc md;
    // CSR.
    ASSERT_NO_THROW(
            md = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));
    // Packed.
    ASSERT_NO_THROW(md = memory::desc::packed({64, 128}, dt::f32, nnz));
}

TEST(iface_sparse_test_t, TestSparseMDComparison) {
    const int nnz = 12;
    memory::desc md1;
    memory::desc md2;

    // CSR.

    // Different index data types.
    ASSERT_NO_THROW(
            md1 = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    ASSERT_NO_THROW(
            md2 = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s8, dt::s32));
    ASSERT_NE(md1, md2);

    // Different pointer data types.
    ASSERT_NO_THROW(
            md1 = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    ASSERT_NO_THROW(
            md2 = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s8));
    ASSERT_NE(md1, md2);

    // Different nnz.
    ASSERT_NO_THROW(
            md1 = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    ASSERT_NO_THROW(md2
            = memory::desc::csr({64, 128}, dt::f32, nnz + 1, dt::s32, dt::s32));
    ASSERT_NE(md1, md2);

    // COO.

    // Different value data types.
    ASSERT_NO_THROW(md1 = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));
    ASSERT_NO_THROW(md2 = memory::desc::coo({64, 128}, dt::f16, nnz, dt::s32));
    ASSERT_NE(md1, md2);

    // Different index data types.
    ASSERT_NO_THROW(md1 = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));
    ASSERT_NO_THROW(md2 = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s8));
    ASSERT_NE(md1, md2);

    // Different nnz.
    ASSERT_NO_THROW(md1 = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));
    ASSERT_NO_THROW(
            md2 = memory::desc::coo({64, 128}, dt::f32, nnz + 1, dt::s32));
    ASSERT_NE(md1, md2);

    // Packed.

    // Equal memory descriptors.
    ASSERT_NO_THROW(md1 = memory::desc::packed({64, 128}, dt::f32, nnz));
    ASSERT_NO_THROW(md2 = memory::desc::packed({64, 128}, dt::f32, nnz));
    ASSERT_EQ(md1, md2);

    // Different nnz.
    ASSERT_NO_THROW(md1 = memory::desc::packed({64, 128}, dt::f32, nnz));
    ASSERT_NO_THROW(md2 = memory::desc::packed({64, 128}, dt::f32, nnz + 1));
    ASSERT_NE(md1, md2);
}

TEST(iface_sparse_test_t, TestSparseMDQueries) {
    const int nnz = 12;
    const auto indices_dt = dt::s8;
    const auto pointers_dt = dt::s32;
    const memory::dims dims = {64, 128};
    const memory::data_type data_type = dt::f32;

    // The C API is used for the out-of-range checks below because the C++
    // wrapper maps any error to `undef`, which is indistinguishable from an
    // out-of-bounds read that happens to return zero.
    auto query_dt = [](const memory::desc &md, int index) {
        dnnl_data_type_t qdt;
        return dnnl_memory_desc_query_v2(
                md.get(), dnnl_query_data_type, index, &qdt);
    };

    memory::desc md;

    // CSR.
    ASSERT_NO_THROW(md
            = memory::desc::csr(dims, data_type, nnz, indices_dt, pointers_dt));
    ASSERT_EQ(md.get_dims(), dims);
    ASSERT_EQ(md.get_data_type(), data_type);
    ASSERT_EQ(md.get_data_type(0), data_type);
    ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);

    ASSERT_EQ(md.get_nnz(), nnz);
    ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::csr);
    ASSERT_EQ(md.get_data_type(1), indices_dt);
    ASSERT_EQ(md.get_data_type(2), pointers_dt);
    // Indices past the metadata types are rejected instead of reading out of
    // bounds.
    ASSERT_EQ(query_dt(md, 3), dnnl_invalid_arguments);
    ASSERT_EQ(query_dt(md, 1 << 20), dnnl_invalid_arguments);
    ASSERT_EQ(query_dt(md, -1), dnnl_invalid_arguments);

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo(dims, data_type, nnz, indices_dt));
    ASSERT_EQ(md.get_dims(), dims);
    ASSERT_EQ(md.get_data_type(), data_type);
    ASSERT_EQ(md.get_data_type(0), data_type);
    ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);

    ASSERT_EQ(md.get_nnz(), nnz);
    ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::coo);
    ASSERT_EQ(md.get_data_type(1), indices_dt);
    ASSERT_EQ(md.get_data_type(2), indices_dt);
    ASSERT_EQ(query_dt(md, -1), dnnl_invalid_arguments);

    // Packed.
    ASSERT_NO_THROW(md = memory::desc::packed(dims, data_type, nnz));
    ASSERT_EQ(md.get_dims(), dims);
    ASSERT_EQ(md.get_data_type(), data_type);
    ASSERT_EQ(md.get_data_type(0), data_type);
    ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);

    ASSERT_EQ(md.get_nnz(), nnz);
    ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::packed);
    ASSERT_EQ(query_dt(md, 3), dnnl_invalid_arguments);
    ASSERT_EQ(query_dt(md, 1 << 20), dnnl_invalid_arguments);
    ASSERT_EQ(query_dt(md, -1), dnnl_invalid_arguments);
    ASSERT_EQ(query_dt(md, std::numeric_limits<int>::min()),
            dnnl_invalid_arguments);
}

TEST(iface_sparse_test_t, TestSparseMDSize) {
    const int nnz = 12;
    memory::desc md;

    // CSR.
    ASSERT_NO_THROW(
            md = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    // Size of values.
    size_t exp_values_size = nnz * memory::data_type_size(md.get_data_type());
    // Default.
    ASSERT_EQ(md.get_size(), exp_values_size);
    // Explicit.
    ASSERT_EQ(md.get_size(0), exp_values_size);

    // Size of indices.
    size_t exp_indices_size = nnz * memory::data_type_size(md.get_data_type(1));
    ASSERT_EQ(md.get_size(1), exp_indices_size);

    // Size of  pointers.
    size_t exp_pointers_size = (md.get_dims()[0] + 1)
            * memory::data_type_size(md.get_data_type(2));
    ASSERT_EQ(md.get_size(2), exp_pointers_size);

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));
    // Size of values.
    exp_values_size = nnz * memory::data_type_size(md.get_data_type());
    // Default.
    ASSERT_EQ(md.get_size(), exp_values_size);
    // Explicit.
    ASSERT_EQ(md.get_size(0), exp_values_size);

    // Size of indices.
    exp_indices_size = nnz * memory::data_type_size(md.get_data_type(1));
    ASSERT_EQ(md.get_size(1), exp_indices_size);
    ASSERT_EQ(md.get_size(2), exp_indices_size);

    // Packed.

    // The user-created memory descriptor for packed encoding cannot
    // be queried for sizes.
    ASSERT_NO_THROW(md = memory::desc::packed({64, 128}, dt::f32, nnz));
    // Size of values.
    // Default.
    ASSERT_EQ(md.get_size(), 0u);
    // Explicit.
    ASSERT_EQ(md.get_size(0), 0u);

    // Size of offsets.
    ASSERT_EQ(md.get_size(1), 0u);

    // Size of bitmask.
    ASSERT_EQ(md.get_size(2), 0u);
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_sparse_test_t, TestSparseMemoryCreation) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (eng.get_kind() == engine::kind::gpu
            || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int nnz = 12;
    memory::desc md;

    // CSR.
    ASSERT_NO_THROW(
            md = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    memory mem;

    // Default memory constructor.
    mem = memory(md, eng);

    // User provided buffers.
    {
        std::vector<float> values(1);
        std::vector<int> indices(1);
        std::vector<int> pointers(1);
        EXPECT_NO_THROW(
                mem = memory(md, eng,
                        {values.data(), indices.data(), pointers.data()}));
    }

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));

    // Default memory constructor.
    mem = memory(md, eng);
    // User provided buffers.
    {
        std::vector<float> values(1);
        std::vector<int> row_indices(1);
        std::vector<int> col_indices(1);

        EXPECT_NO_THROW(mem = memory(md, eng,
                                {values.data(), row_indices.data(),
                                        col_indices.data()}));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_sparse_test_t, TestSparseMemorySetGetDataHandles) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (eng.get_kind() == engine::kind::gpu
            || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int nnz = 12;

    // CSR.
    memory::desc md;
    ASSERT_NO_THROW(
            md = memory::desc::csr({64, 128}, dt::f32, nnz, dt::s32, dt::s32));
    memory mem;

    int nhandles = 3;
    // Default memory constructor.
    mem = memory(md, eng);

    {
        for (int i = 0; i < nhandles; i++) {
            void *h = mem.get_data_handle(i);
            ASSERT_NE(h, nullptr);
        }

        // Creating a memory object without underlying buffers.
        for (int i = 0; i < nhandles; i++) {
            EXPECT_NO_THROW(mem.set_data_handle(DNNL_MEMORY_NONE, i));
        }

        for (int i = 0; i < nhandles; i++) {
            void *h = mem.get_data_handle(i);
            ASSERT_EQ(h, nullptr);
        }
    }

    // User provided buffers.
    {
        std::vector<float> values(1);
        std::vector<int> indices(1);
        std::vector<int> pointers(1);

        ASSERT_NO_THROW(mem.set_data_handle(values.data(), 0));
        ASSERT_NO_THROW(mem.set_data_handle(indices.data(), 1));
        ASSERT_NO_THROW(mem.set_data_handle(pointers.data(), 2));

        ASSERT_EQ(mem.get_data_handle(0), values.data());
        ASSERT_EQ(mem.get_data_handle(1), indices.data());
        ASSERT_EQ(mem.get_data_handle(2), pointers.data());
    }

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, dt::f32, nnz, dt::s32));

    // Default memory constructor.
    mem = memory(md, eng);

    {
        for (int i = 0; i < nhandles; i++) {
            void *h = mem.get_data_handle(i);
            ASSERT_NE(h, nullptr);
        }

        // Creating a memory object without underlying buffers.
        for (int i = 0; i < nhandles; i++) {
            EXPECT_NO_THROW(mem.set_data_handle(DNNL_MEMORY_NONE, i));
        }

        for (int i = 0; i < nhandles; i++) {
            void *h = mem.get_data_handle(i);
            ASSERT_EQ(h, nullptr);
        }
    }

    // User provided buffers.
    {
        std::vector<float> values(1);
        std::vector<int> row_indices(1);
        std::vector<int> col_indices(1);

        ASSERT_NO_THROW(mem.set_data_handle(values.data(), 0));
        ASSERT_NO_THROW(mem.set_data_handle(row_indices.data(), 1));
        ASSERT_NO_THROW(mem.set_data_handle(col_indices.data(), 2));

        ASSERT_EQ(mem.get_data_handle(0), values.data());
        ASSERT_EQ(mem.get_data_handle(1), row_indices.data());
        ASSERT_EQ(mem.get_data_handle(2), col_indices.data());
    }
}

TEST(iface_sparse_test_t, TestSparseMemoryMapUnmap) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (eng.get_kind() == engine::kind::gpu
            || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int nnz = 2;

    // CSR.
    memory::desc md;
    ASSERT_NO_THROW(
            md = memory::desc::csr({2, 2}, dt::f32, nnz, dt::s32, dt::s32));

    std::vector<float> values = {1.5, 2.5};
    std::vector<int> indices = {0, 1};
    std::vector<int> pointers = {0, 1, 2};

    memory mem(md, eng, {values.data(), indices.data(), pointers.data()});

    float *mapped_values = nullptr;
    int *mapped_indices = nullptr;
    int *mapped_pointers = nullptr;

    ASSERT_NO_THROW(mapped_values = mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_indices = mem.map_data<int>(1));
    ASSERT_NO_THROW(mapped_pointers = mem.map_data<int>(2));

    for (size_t i = 0; i < values.size(); i++)
        ASSERT_EQ(values[i], mapped_values[i]);

    for (size_t i = 0; i < indices.size(); i++)
        ASSERT_EQ(indices[i], mapped_indices[i]);

    for (size_t i = 0; i < pointers.size(); i++)
        ASSERT_EQ(pointers[i], mapped_pointers[i]);

    ASSERT_NO_THROW(mem.unmap_data(mapped_values, 0));
    ASSERT_NO_THROW(mem.unmap_data(mapped_indices, 1));
    ASSERT_NO_THROW(mem.unmap_data(mapped_pointers, 2));

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({2, 2}, dt::f32, nnz, dt::s32));

    std::vector<float> coo_values = {1.5, 2.5};
    std::vector<int> row_indices = {0, 1};
    std::vector<int> col_indices = {0, 1};

    memory coo_mem(md, eng,
            {coo_values.data(), row_indices.data(), col_indices.data()});

    float *mapped_coo_values = nullptr;
    int *mapped_row_indices = nullptr;
    int *mapped_col_indices = nullptr;

    ASSERT_NO_THROW(mapped_coo_values = coo_mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_row_indices = coo_mem.map_data<int>(1));
    ASSERT_NO_THROW(mapped_col_indices = coo_mem.map_data<int>(2));

    for (size_t i = 0; i < coo_values.size(); i++)
        ASSERT_EQ(values[i], mapped_values[i]);

    for (size_t i = 0; i < row_indices.size(); i++)
        ASSERT_EQ(row_indices[i], mapped_row_indices[i]);

    for (size_t i = 0; i < col_indices.size(); i++)
        ASSERT_EQ(col_indices[i], mapped_col_indices[i]);

    ASSERT_NO_THROW(mem.unmap_data(mapped_coo_values, 0));
    ASSERT_NO_THROW(mem.unmap_data(mapped_row_indices, 1));
    ASSERT_NO_THROW(mem.unmap_data(mapped_col_indices, 2));
}

} // namespace dnnl
