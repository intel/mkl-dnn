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

#include "test_gemm_data_preparation.hpp"
#include "test_gemm_params.hpp"
#include "test_gemm_validation.hpp"
#include "gtest/gtest.h"

#include "dnnl_test_common.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#include "tests/test_isa_common.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace dnnl {

struct brgemm_params_t : test_params_t {
    impl::data_type_t dt_a;
    impl::data_type_t dt_b;
    impl::cpu::x64::brgemm_batch_kind_t batch_kind;
    impl::cpu::x64::brgemm_layout_t layout;

    impl::cpu::x64::brgemm_attr_t attrs;

    int bs;
};

class params_creator_t {
public:
    std::vector<brgemm_params_t> create_simple_brgemm_params() {
        params = {};

        transpose_ = {'n'};
        sizes_and_leading_dims_[0]
                = {{1, 4}, {3, 3}, {3, 8}, {30, 30}, {64, 64}, {31, 61}};
        sizes_and_leading_dims_[1]
                = {{1, 4}, {2, 6}, {2, 5}, {20, 20}, {64, 64}, {21, 51}};
        sizes_and_leading_dims_[2]
                = {{1, 4}, {1, 3}, {1, 2}, {10, 20}, {64, 64}, {11, 81}};

        alpha_values_ = {1.0f, 2.0f, 2.5f};
        beta_values_ = {0.0f, 1.0f, 2.0f};

        amx_dts_ = {{dnnl_f32, dnnl_f32}};
        dts_ = {{dnnl_f32, dnnl_f32}};

        put_params();

        sizes_and_leading_dims_[0] = {{4, 4}, {8, 12}, {64, 1024}};
        sizes_and_leading_dims_[1] = {{4, 4}, {16, 32}, {128, 512}};
        sizes_and_leading_dims_[2] = {{4, 4}, {12, 56}, {16, 256}};

        amx_dts_ = {
                {dnnl_bf16, dnnl_bf16}, {dnnl_u8, dnnl_u8}, {dnnl_s8, dnnl_s8}};
        dts_ = {{dnnl_bf16, dnnl_bf16}, {dnnl_u8, dnnl_s8}};

        put_params();

        return params;
    }

private:
    void put_params() {
        for_(auto tr : transpose_)
        for_(size_t i = 0; i < sizes_and_leading_dims_[0].size(); i++)
        for_(auto alpha : alpha_values_)
        for_(auto beta : beta_values_)
        for (const auto &dt : is_amx ? amx_dts_ : dts_) {
            brgemm_params_t param = {};
            param.transA = tr;
            param.transB = 'n';
            param.M = sizes_and_leading_dims_[0][i].first;
            param.lda = sizes_and_leading_dims_[0][i].second;
            param.N = sizes_and_leading_dims_[1][i].first;
            param.ldb = sizes_and_leading_dims_[1][i].second;
            param.K = sizes_and_leading_dims_[2][i].first;
            param.ldc = sizes_and_leading_dims_[2][i].second;
            param.alpha = alpha;
            param.beta = beta;
            param.dt_a = dt.first;
            param.dt_b = dt.second;
            param.batch_kind = impl::cpu::x64::brgemm_addr;
            param.layout = impl::cpu::x64::brgemm_row_major;
            param.bs = 1;
            param.attrs.max_bs = 1;
            param.attrs.max_top_vpad = 0;
            param.attrs.max_bottom_vpad = 0;
            param.expect_to_fail = false;
            param.expected_status = dnnl_success;

            params.emplace_back(param);
        }
    }

    const bool is_amx = dnnl::mayiuse(cpu_isa::avx512_core_amx);

    std::vector<char> transpose_;
    std::vector<std::pair<int64_t, int64_t>> sizes_and_leading_dims_[3];
    std::vector<float> alpha_values_;
    std::vector<float> beta_values_;
    std::vector<std::pair<impl::data_type_t, impl::data_type_t>> amx_dts_;
    std::vector<std::pair<impl::data_type_t, impl::data_type_t>> dts_;

    std::vector<brgemm_params_t> params;
};

class brgemm_test_t : public ::testing::TestWithParam<brgemm_params_t> {
protected:
    void SetUp() override {
        const auto &p = GetParam();

        SKIP_IF(engine::get_count(engine::kind::cpu) == 0,
                "Brgemm requires cpu.");
        eng_ = std::make_shared<engine>(engine::kind::cpu, 0);

        SKIP_IF(!impl::cpu::platform::has_data_type_support(p.dt_a),
                "Engine does not support this data type.");

        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status, true);
    }

    void Test() {
        const auto &p = ::testing::TestWithParam<brgemm_params_t>::GetParam();
        run_proper_test(p);
    }

private:
    template <typename b_dt>
    void reorder_B(const brgemm_params_t &p, const mapped_ptr_t<b_dt> &b_mem,
            mapped_ptr_t<b_dt> &b_mem_reordered) const {
        static constexpr int k_pack = 4 / sizeof(b_dt);

        const b_dt *b_mem_data = b_mem;
        b_dt *b_mem_reordered_data = b_mem_reordered;
        dnnl::impl::parallel_nd(p.K, p.N, [=](int64_t k, int64_t n) {
            size_t b_off = k * p.ldb + n;
            size_t b_reordered_off
                    = (k / k_pack) * p.ldb * k_pack + n * k_pack + k % k_pack;
            b_mem_reordered_data[b_reordered_off] = b_mem_data[b_off];
        });
    }

    template <typename b_dt>
    mapped_ptr_t<b_dt> get_B_mem(const brgemm_params_t &p) {
        mapped_ptr_t<b_dt> B = map_memory<b_dt>(*gemm_data_.b_mem);

        static constexpr int k_pack = 4 / sizeof(b_dt);
        if (k_pack > 1) {
            size_t sizeA, sizeB, sizeC;
            get_matrix_size(p, sizeA, sizeB, sizeC);

            b_mem_reordered_ = std::make_shared<test_memory>(
                    get_matrix_md<b_dt>(sizeB, p.off.b), *eng_);
            auto B_reordered = map_memory<b_dt>(*b_mem_reordered_);

            reorder_B(p, B, B_reordered);

            return B_reordered;
        }

        return B;
    }

    template <typename a_dt, typename b_dt, typename c_dt>
    dnnl_status_t run_brgemm(const brgemm_params_t &p) {
        using namespace dnnl::impl::cpu;
        using namespace dnnl::impl::cpu::x64;

        mapped_ptr_t<a_dt> A = map_memory<a_dt>(*gemm_data_.a_mem);
        mapped_ptr_t<b_dt> B = get_B_mem<b_dt>(p);
        mapped_ptr_t<c_dt> C = map_memory<c_dt>(*gemm_data_.c_mem);

        //initialize brgemm kernel
        char palette[64];
        char tile_buffer[1024];
        x64::brgemm_desc_t desc;
        auto res = brgemm_desc_init(&desc, x64::cpu_isa_t::isa_undef,
                p.batch_kind, p.dt_a, p.dt_b, p.tr_a(), p.tr_b(), p.layout,
                p.alpha, p.beta, p.lda, p.ldb, p.ldc, p.M, p.N, p.K);
        if (res != dnnl_success) return res;

        if (!desc.is_tmm) {
            res = brgemm_desc_set_attr(&desc, p.attrs);
            if (res != dnnl_success) return res;
        }

        res = brgemm_desc_finalize(&desc);
        if (res != dnnl_success) return res;

        if (desc.is_tmm) {
            res = brgemm_init_tiles(desc, palette);
            if (res != dnnl_success) return res;
        }

        x64::brgemm_kernel_t *_t_ptr;
        res = brgemm_kernel_create(&_t_ptr, desc);
        if (res != dnnl_success) return res;

        x64::brgemm_batch_element_t batch_element;
        batch_element.ptr.A = A;
        batch_element.ptr.B = B;
        batch_element.vvpad.top = 0;
        batch_element.vvpad.bottom = 0;
        if (desc.is_tmm) {
            res = amx_tile_configure(palette);
            if (res != dnnl_success) return res;
        }
        brgemm_kernel_execute(_t_ptr, p.bs, &batch_element, C,
                desc.is_tmm ? tile_buffer : nullptr);

        res = brgemm_kernel_destroy(_t_ptr);
        if (res != dnnl_success) return res;

        if (desc.is_tmm) {
            res = amx_tile_release();
            if (res != dnnl_success) return res;
        }

        return res;
    }

    template <typename a_dt, typename b_dt, typename c_dt>
    void test_brgemm(const brgemm_params_t &p) {
        gemm_data_ = {};
        prepare_data_for_gemm_testing<a_dt, b_dt, c_dt>(p, gemm_data_, *eng_);

        dnnl_status_t status = run_brgemm<a_dt, b_dt, c_dt>(p);

        if (status == dnnl_success) {
            validate<a_dt, b_dt, c_dt>(p, gemm_data_);
        }

        if (status != dnnl_success)
            throw error(status, "oneDNN brgemm returned error");
    }

    void run_proper_test(const brgemm_params_t &p) {
        using namespace impl::cpu::x64;

        if (dnnl::mayiuse(cpu_isa::avx512_core_amx)) {
            if (p.dt_a == dnnl_f32 && p.dt_b == dnnl_f32)
                test_brgemm<float, float, float>(p);
            else if (p.dt_a == dnnl_bf16 && p.dt_b == dnnl_bf16)
                test_brgemm<bfloat16_t, bfloat16_t, float>(p);
            else if (p.dt_a == dnnl_s8 && p.dt_b == dnnl_s8)
                test_brgemm<int8_t, int8_t, int32_t>(p);
            else if (p.dt_a == dnnl_u8 && p.dt_b == dnnl_u8)
                test_brgemm<uint8_t, uint8_t, int32_t>(p);
            else
                throw error(dnnl_unimplemented, "Brgemm unimplemented.");
        } else {
            if (p.dt_a == dnnl_f32 && p.dt_b == dnnl_f32)
                test_brgemm<float, float, float>(p);
            else if (p.dt_a == dnnl_bf16 && p.dt_b == dnnl_bf16)
                test_brgemm<bfloat16_t, bfloat16_t, float>(p);
            else if (p.dt_a == dnnl_u8 && p.dt_b == dnnl_s8) {
                assert(p.layout == brgemm_layout_t::brgemm_row_major);
                test_brgemm<uint8_t, int8_t, int32_t>(p);
            } else if (p.dt_a == dnnl_s8 && p.dt_b == dnnl_u8) {
                assert(p.layout == brgemm_layout_t::brgemm_col_major);
                test_brgemm<int8_t, uint8_t, int32_t>(p);
            } else
                throw error(dnnl_unimplemented, "Brgemm unimplemented.");
        }
    }

    std::shared_ptr<engine> eng_;
    test_gemm_data_t gemm_data_;
    std::shared_ptr<test_memory> b_mem_reordered_;
};

TEST_P(brgemm_test_t, TestsBRGEMM) {}
INSTANTIATE_TEST_SUITE_P(TestBRGEMMSimple, brgemm_test_t,
        ::testing::ValuesIn(params_creator_t().create_simple_brgemm_params()));

namespace {

using namespace impl::cpu::x64;
using impl::dim_t;

impl::status_t init_roll_desc(brgemm_desc_t &desc, impl::data_type_t dt_a,
        impl::data_type_t dt_b, float alpha, dim_t M, dim_t N, dim_t K,
        dim_t LDA, dim_t LDB, dim_t LDC, const brgemm_attr_t &attr) {
    CHECK(brgemm_desc_init(&desc, isa_undef, brgemm_addr, dt_a, dt_b, false,
            false, brgemm_row_major, alpha, 0.f, LDA, LDB, LDC, M, N, K));
    CHECK(brgemm_desc_set_attr(&desc, attr));
    return brgemm_desc_finalize(&desc);
}

// Runs the kernel on small integers, so the reference is exact. A row masked
// by bd_mask is not stored, any other row goes to the row equal to the number
// of stored rows before it. Output elements that must stay untouched are
// checked against a sentinel. With to_D the kernel runs with post-ops and the
// output is D, otherwise it is C.
template <typename a_t, typename b_t, typename out_t>
void check_roll_kernel(const brgemm_desc_t &desc, dim_t max_bs,
        const std::vector<dim_t> &batch_sizes, out_t *out, bool to_D = false) {
    ASSERT_EQ(desc.beta, 0.f);
    ASSERT_TRUE(desc.is_b_data_layout_vnni());
    const dim_t M = desc.bcast_dim, N = desc.load_dim, K = desc.reduce_dim;
    const dim_t k_pack = 4 / sizeof(b_t);
    const dim_t K_padded = impl::utils::rnd_up(K, k_pack);

    std::vector<std::vector<a_t>> A(max_bs);
    std::vector<std::vector<b_t>> B(max_bs);
    std::vector<brgemm_batch_element_t> batch(max_bs);
    for (dim_t b = 0; b < max_bs; b++) {
        A[b].resize(M * desc.LDA);
        for (dim_t i = 0; i < M * desc.LDA; i++) {
            const dim_t m = i / desc.LDA, k = i % desc.LDA;
            A[b][i] = static_cast<a_t>(
                    static_cast<float>((m * 3 + k * 7 + b) % 5));
        }
        B[b].resize(K_padded * desc.LDB, static_cast<b_t>(0.f));
        for (dim_t k = 0; k < K; k++)
            for (dim_t n = 0; n < N; n++)
                B[b][(k / k_pack) * desc.LDB * k_pack + n * k_pack + k % k_pack]
                        = static_cast<b_t>(static_cast<float>(
                                (k * 2 + n * 3 + b) % 5 - 2));
        batch[b].ptr.A = A[b].data();
        batch[b].ptr.B = B[b].data();
        batch[b].vvpad.top = 0;
        batch[b].vvpad.bottom = 0;
    }

    const auto out_off = [&](dim_t m, dim_t n) {
        if (to_D) return m * desc.LDD + n;
        return m * desc.LDC2_M + (n / desc.LDC) * desc.LDC2_N + n % desc.LDC;
    };
    const auto is_stored = [&](dim_t m) {
        return desc.brgattr.bd_mask_level == 0 || desc.brgattr.bd_mask[m];
    };
    const out_t sentinel = static_cast<out_t>(-777.f);
    std::vector<char> C(to_D ? M * desc.LDC * desc.typesize_C : 0);

    brgemm_kernel_t *kernel = nullptr;
    ASSERT_EQ(brgemm_kernel_create(&kernel, desc), impl::status::success);
    char palette[AMX_PALETTE_SIZE] = {};
    const bool use_tiles
            = brgemm_init_tiles(desc, palette) == impl::status::success;
    std::vector<char> wsp(
            impl::nstl::max<dim_t>(desc.get_wsp_buffer_size(), 1));

    for (const dim_t bs : batch_sizes) {
        for (dim_t m = 0; m < M; m++)
            for (dim_t n = 0; n < N; n++)
                out[out_off(m, n)] = sentinel;

        if (use_tiles) amx_tile_configure(palette);
        if (to_D)
            brgemm_kernel_execute_postops(kernel, bs, batch.data(), C.data(),
                    out, brgemm_post_ops_data_t(), wsp.data());
        else
            brgemm_kernel_execute(kernel, bs, batch.data(), out, wsp.data());
        if (use_tiles) amx_tile_release();

        dim_t m_out = 0;
        for (dim_t m = 0; m < M; m++) {
            if (!is_stored(m)) continue;
            for (dim_t n = 0; n < N; n++) {
                float acc = 0.f;
                for (dim_t b = 0; b < bs; b++)
                    for (dim_t k = 0; k < K; k++)
                        acc += static_cast<float>(A[b][m * desc.LDA + k])
                                * static_cast<float>(
                                        B[b][(k / k_pack) * desc.LDB * k_pack
                                                + n * k_pack + k % k_pack]);
                const out_t ref = static_cast<out_t>(desc.alpha * acc);
                ASSERT_EQ(static_cast<float>(out[out_off(m_out, n)]),
                        static_cast<float>(ref))
                        << "bs=" << bs << " m=" << m << " n=" << n;
            }
            m_out++;
        }
        for (dim_t m = m_out; m < M; m++)
            for (dim_t n = 0; n < N; n++)
                ASSERT_EQ(static_cast<float>(out[out_off(m, n)]),
                        static_cast<float>(sentinel))
                        << "bs=" << bs << " m=" << m << " n=" << n;
    }

    brgemm_kernel_destroy(kernel);
}

} // namespace

TEST(brgemm_roll_loops_test, BdMaskDisablesFirstRowOfIteration) {
    SKIP_IF(!dnnl::mayiuse(cpu_isa::avx512_core_amx), "AMX is not supported");

    // Level 2 walks the mask, the first two rows are skipped by blocking.
    const dim_t M = 66, N = 32, K = 32;
    std::vector<char> mask(M, 1);
    mask[0] = mask[1] = 0;

    brgemm_attr_t attr;
    attr.use_uker = true;
    attr.max_bs = 1;
    attr.bd_mask_level = 2;
    attr.bd_mask = mask.data();
    attr.hint_bd_block2 = 1;
    attr.hint_roll_bd_loop = true;

    brgemm_desc_t desc;
    ASSERT_EQ(init_roll_desc(desc, impl::data_type::bf16, impl::data_type::bf16,
                      1.f, M, N, K, K, N, N, attr),
            impl::status::success);
    ASSERT_TRUE(desc.brgattr.use_uker);

    std::vector<float> C(M * N);
    check_roll_kernel<bfloat16_t, bfloat16_t, float>(desc, 1, {1}, C.data());
}

TEST(brgemm_roll_loops_test, BdStepBeyondInt32) {
    SKIP_IF(!dnnl::mayiuse(cpu_isa::avx512_core_amx), "AMX is not supported");
#if !defined(__linux__)
    SKIP_IF(true, "Sparse D buffer needs Linux mmap");
#else
    // Rows of D are 64 MiB apart, so the D pointer step of a rolled bd loop
    // over 32 rows does not fit into imm32. Interleaved stores apply the bd
    // shifts at other points, so run both store modes.
    const dim_t M = 96, N = 32, K = 32, LDD = dim_t(1) << 25;

    for (const bool ils : {false, true}) {
        SCOPED_TRACE(ils ? "interleaved stores" : "no interleaved stores");
        brgemm_attr_t attr;
        attr.use_uker = true;
        attr.max_bs = 1;
        attr.hint_bd_block2 = 2;
        attr.hint_ld_block2 = 2;
        attr.hint_roll_bd_loop = true;
        attr.use_interleave_stores = ils;

        brgemm_desc_t desc;
        ASSERT_EQ(brgemm_desc_init(&desc, isa_undef, brgemm_addr,
                          impl::data_type::bf16, impl::data_type::bf16, false,
                          false, brgemm_row_major, 1.f, 0.f, K, N, N, M, N, K),
                impl::status::success);
        dnnl::primitive_attr pattr;
        const memory::desc dst_md(
                {M, N}, memory::data_type::bf16, memory::dims {LDD, 1});
        ASSERT_EQ(
                brgemm_desc_set_postops(&desc, pattr.get(), dst_md.get(), LDD),
                impl::status::success);
        ASSERT_EQ(brgemm_desc_set_attr(&desc, attr), impl::status::success);
        ASSERT_EQ(brgemm_desc_finalize(&desc), impl::status::success);
        ASSERT_TRUE(desc.brgattr.use_uker);
        ASSERT_EQ(desc.interleave_tilestores_, ils);

        const size_t D_size = ((M - 1) * LDD + N) * sizeof(bfloat16_t);
        void *D = mmap(nullptr, D_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        SKIP_IF(D == MAP_FAILED, "Cannot reserve the sparse D buffer");
        check_roll_kernel<bfloat16_t, bfloat16_t, bfloat16_t>(
                desc, 1, {1}, static_cast<bfloat16_t *>(D), true);
        munmap(D, D_size);
    }
#endif
}

TEST(brgemm_roll_loops_test, LdStepBeyondInt32) {
    SKIP_IF(!dnnl::mayiuse(cpu_isa::avx512_core_amx), "AMX is not supported");
#if !defined(__linux__)
    SKIP_IF(true, "Sparse C buffer needs Linux mmap");
#else
    // Each ld block of C is a separate panel 2 GiB + 4 KiB apart, so the C
    // pointer step of a rolled ld loop does not fit into imm32. With the bd
    // loop innermost over two bd blocks, the first ld iteration is not split
    // out of the rolled body, so the body itself has no C displacement.
    const dim_t M = 32, N = 64, K = 32, LDC = 16;
    const dim_t LDC2_N = (dim_t(1) << 29) + 1024;

    brgemm_attr_t attr;
    attr.use_uker = true;
    attr.max_bs = 1;
    attr.LDC2_N = LDC2_N;
    attr.hint_innermost_loop = brgemm_bd_loop_innermost;
    attr.hint_bd_block2 = 1;
    attr.hint_ld_block2 = 1;
    attr.hint_roll_ld_loop = true;
    attr.use_interleave_stores = false;

    brgemm_desc_t desc;
    ASSERT_EQ(init_roll_desc(desc, impl::data_type::bf16, impl::data_type::bf16,
                      1.f, M, N, K, K, N, LDC, attr),
            impl::status::success);
    ASSERT_TRUE(desc.brgattr.use_uker);
    ASSERT_EQ(desc.LDC2_N, LDC2_N);
    ASSERT_FALSE(desc.save_transform_A());

    const size_t C_size
            = ((N / LDC - 1) * LDC2_N + M * desc.LDC2_M) * sizeof(float);
    void *C = mmap(nullptr, C_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    SKIP_IF(C == MAP_FAILED, "Cannot reserve the sparse C buffer");
    check_roll_kernel<bfloat16_t, bfloat16_t, float>(
            desc, 1, {1}, static_cast<float *>(C));
    munmap(C, C_size);
#endif
}

TEST(brgemm_roll_loops_test, AceVarBsKeepsUker) {
    SKIP_IF(!dnnl::mayiuse(avx10_2_ace), "ACE is not supported");
    const dim_t M = 2048, N = 128, K = 64, max_bs = 3;

    brgemm_attr_t attr;
    attr.use_ace = true;
    attr.use_uker = true;
    attr.enable_size_driven_roll = true;
    attr.max_bs = max_bs;
    attr.hint_innermost_loop = brgemm_bd_loop_innermost;

    // The bd loop is innermost and cannot roll, so the ACE uker stays too
    // big and falls back to the base kernel, unless the batch size is only
    // known at run time.
    brgemm_desc_t desc;
    ASSERT_EQ(init_roll_desc(desc, impl::data_type::u8, impl::data_type::s8,
                      1.f, M, N, K, K, N, N, attr),
            impl::status::success);
    ASSERT_FALSE(desc.brgattr.use_uker);

    attr.var_bs = true;
    ASSERT_EQ(init_roll_desc(desc, impl::data_type::u8, impl::data_type::s8,
                      1.f, M, N, K, K, N, N, attr),
            impl::status::success);
    ASSERT_TRUE(desc.brgattr.use_uker);

    std::vector<int32_t> C(M * N);
    check_roll_kernel<uint8_t, int8_t, int32_t>(
            desc, max_bs, {0, 1, 2, 3}, C.data());
}

TEST(brgemm_roll_loops_test, AceLargeLddKeepsUker) {
    SKIP_IF(!dnnl::mayiuse(avx10_2_ace), "ACE is not supported");
    // The ACE uker of AceVarBsKeepsUker is too big and falls back to the
    // base kernel, unless the D stride does not fit into int32 for it.
    const dim_t M = 2048, N = 128, K = 64;

    brgemm_attr_t attr;
    attr.use_ace = true;
    attr.use_uker = true;
    attr.enable_size_driven_roll = true;
    attr.max_bs = 3;
    attr.hint_innermost_loop = brgemm_bd_loop_innermost;

    for (const dim_t LDD : {N, dim_t(8388608)}) {
        brgemm_desc_t desc;
        ASSERT_EQ(brgemm_desc_init(&desc, isa_undef, brgemm_addr,
                          impl::data_type::u8, impl::data_type::s8, false,
                          false, brgemm_row_major, 1.f, 0.f, K, N, N, M, N, K),
                impl::status::success);
        dnnl::primitive_attr pattr;
        const memory::desc dst_md(
                {M, N}, memory::data_type::s32, memory::dims {LDD, 1});
        ASSERT_EQ(
                brgemm_desc_set_postops(&desc, pattr.get(), dst_md.get(), LDD),
                impl::status::success);
        ASSERT_EQ(brgemm_desc_set_attr(&desc, attr), impl::status::success);
        ASSERT_EQ(brgemm_desc_finalize(&desc), impl::status::success);
        ASSERT_EQ(desc.brgattr.use_uker, LDD != N) << "LDD=" << LDD;
    }
}

} // namespace dnnl
