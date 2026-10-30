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

#ifndef GPU_INTEL_GEMM_JIT_GEN_KERNEL_HPP
#define GPU_INTEL_GEMM_JIT_GEN_KERNEL_HPP

#include <string>

#include "common/c_types_map.hpp"
#include "gemmstone/driver_info.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/type.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/gemm/jit/pd.hpp"
#include "gpu/intel/jit/generator_base.hpp"
#include "gpu/intel/kernel_cache.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

bool enable_generator_dsl();

static inline gemmstone::Type convert_dnnl_to_kernel_type(data_type_t type) {
    using gemmstone::Type;
    switch (type) {
        default: assert(!"Unknown type");
        case data_type::f64: return Type::f64;
        case data_type::f32: return Type::f32;
        case data_type::f16: return Type::f16;
        case data_type::bf16: return Type::bf16;
        case data_type::f8_e5m2: return Type::bf8;
        case data_type::f8_e4m3: return Type::hf8;
        case data_type::e8m0: return Type::f8_e8m0;
        case data_type::f4_e2m1: return Type::f4_e2m1;
        case data_type::s32: return Type::s32;
        case data_type::u8: return Type::u8;
        case data_type::s8: return Type::s8;
        case data_type::u3: return Type::u3;
        case data_type::u4: return Type::u4;
        case data_type::s4: return Type::s4;
        case data_type::u2: return Type::u2;
        case data_type::undef: return Type::invalid;
    }
}

struct gen_desc_t {
    friend struct gen_kernel_t;

    const gemmstone::GEMMProblem *problem() const { return &problem_; }
    const gemmstone::GEMMStrategy *strategy() const { return &strategy_; }
    std::string str() const;

    const gemmstone::CommonDriverInfo *driver_info() const {
        return &driver_info_;
    }
    const gemmstone::EvaluateAuxOutput *aux_params() const {
        return &aux_params_;
    }

    compute::scalar_type_t scalar_type() const;

    status_t create_generator(
            const intel::engine_t &engine, compute::kernel_t &kernel) const;

    serialization_stream_t serialize() const {
        return serialization_stream_t(problem_, strategy_);
    }
    compute::gpu_arch_t arch() const { return arch_; }

    bool has_entry() const { return entry_ != nullptr; }

    const gemmstone::kcatalog::Entry &entry() const {
        assert(entry_ != nullptr);
        return *entry_;
    }

    void set_entry(const gemmstone::kcatalog::Entry *entry) { entry_ = entry; }
    void set_problem(const gemmstone::GEMMProblem &problem) {
        problem_ = problem;
    }

    void set_efficient_64b(bool efficient_64b) {
        efficient_64b_ = efficient_64b;
    }

    void set_kernel_override(const std::string &kernel) {
        kernel_override_ = kernel;
    }
    // Why the override does not run.
    const std::string &reject_reason() const { return reject_reason_; }

protected:
    compute::gpu_arch_t arch_;
    ngen::HW hw_ = ngen::HW::Unknown;
    ngen::Product product_;
    int stepping_ = 0;
    bool l1_flush_wa_ = false;
    gemmstone::GEMMProblem problem_ = {};
    gemmstone::GEMMStrategy strategy_;
    const gemmstone::kcatalog::Entry *entry_ = nullptr;
    gemmstone::EvaluateAuxOutput aux_params_;
    gemmstone::CommonDriverInfo driver_info_;

    bool efficient_64b_ = false;

    std::string kernel_override_;
    std::string reject_reason_;

    /* optional information to fine-tune kernel */
    int m_ = -1, n_ = -1, k_ = -1;
    int eu_count_ = -1;
    bool disable_systolic_ = false;
    bool relaxed_acc_ = false;

    status_t finalize(const char *tags);
    status_t apply_kernel_override(
            const std::string &ovr_strategy, const char *tags);
    void update_driver_info();
};

struct gen_nocopy_desc_t : public gen_desc_t {
    enum compute_mode {
        mode_default = 0,
        mode_tf32 = 0x1,
        mode_bf16x1 = 0x2,
        mode_f16x1 = 0x4,
        mode_w_decomp = 0x8,
        mode_relaxed_acc = 0x10,
        mode_strict = 0x20,
        mode_deterministic = 0x8000
    };

    friend void set_mode(compute_mode &mode, compute_mode flag) {
        mode = static_cast<compute_mode>(mode | flag);
    }

    std::vector<const gemmstone::kcatalog::Entry *> select_kernel(
            const compute::device_info_t &dev_info, compute_mode mode,
            const gemmstone::GEMMProblem &problem, float alpha, float beta,
            dim_t m, dim_t n, dim_t k, dim_t lda, dim_t ldb, dim_t ldc,
            dim_t batch);

    status_t finalize();

private:
    std::string tags_;
    gemmstone::EvaluateParams eval_params_;
    gemmstone::Type Ts_;
    gemmstone::Scalar beta_;
};

struct gen_xe_systolic_kernel_desc_t : public gen_desc_t {
    status_t select_kernel(const compute::device_info_t &dev_info,
            int batch_dims, bool packed_c, bool trans_co, bool a_offset,
            bool b_offset, bool c_offset, bool bias, float alpha, float beta,
            data_type_t a_type, data_type_t b_type, data_type_t c_type,
            data_type_t ao_type, data_type_t bo_type, data_type_t co_type,
            data_type_t acc_type, dim_t m, dim_t n, dim_t k, dim_t batch,
            int unroll_m, int unroll_n, bool alt, gpu_post_ops_t &&post_ops);

    static void choose_unrolls(compute::gpu_arch_t arch, int eu_count,
            data_type_t a_type, data_type_t b_type, data_type_t c_type, dim_t m,
            dim_t n, dim_t k, dim_t batch, int &unroll_m, int &unroll_n,
            bool &alt);

    static int min_block_k(data_type_t a_type) { return 2048; }
};

std::string dump_kernel(ngen::HW hw, const gemmstone::GEMMProblem &problem,
        const gemmstone::GEMMStrategy &strategy);

struct gen_kernel_t : public intel::jit::generator_base_t {

    explicit gen_kernel_t(const gen_desc_t &desc) : desc_(desc) {}

    const char *kernel_name() const override { return "gemm_kernel"; }
    status_t get_kernel(
            compute::kernel_t &kernel, const intel::engine_t *engine) override;

    const gen_desc_t *desc() const { return &desc_; }

protected:
    const gen_desc_t &desc_;
    ngen::NEOInterfaceHandler interface_ {ngen::HW::Unknown};

    void init_interface();
    void maybe_print_verbose();
};

} // namespace jit
} // namespace gemm

template <>
struct key_validator_t<gemm::jit::gen_desc_t> {
    static bool is_valid(const gemm::jit::gen_desc_t &) { return true; }
};

template <>
struct key_validator_t<gemm::jit::gen_nocopy_desc_t> {
    static bool is_valid(const gemm::jit::gen_nocopy_desc_t &derived) {
        return key_validator_t<gemm::jit::gen_desc_t>::is_valid(derived);
    }
};

template <>
struct key_validator_t<gemm::jit::gen_xe_systolic_kernel_desc_t> {
    static bool is_valid(
            const gemm::jit::gen_xe_systolic_kernel_desc_t &derived) {
        return key_validator_t<gemm::jit::gen_desc_t>::is_valid(derived);
    }
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
