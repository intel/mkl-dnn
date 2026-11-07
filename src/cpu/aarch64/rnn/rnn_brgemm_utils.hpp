/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2026 FUJITSU LIMITED
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_RNN_RNN_BRGEMM_UTILS_HPP
#define CPU_AARCH64_RNN_RNN_BRGEMM_UTILS_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {
struct rnn_conf_t;
} // namespace rnn_utils

namespace aarch64 {
namespace matmul {
struct jit_brgemm_matmul_copy_b_t;
} // namespace matmul

namespace rnn_brgemm_utils {

using brgemm_ker_ptr_t = std::unique_ptr<brgemm_kernel_t>;

using scratch_gates_reorder_ker_ptr_t
        = std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>;

struct rnn_brgemm_base_t {

    static void init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
            memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
            dim_t gemm_acc_align);

    static constexpr dim_t num_base_kernels_ = 3;
    static constexpr dim_t num_proj_kernels_ = 4;
    static constexpr dim_t num_vanilla_gru_iter_part2_kernels_ = 4;
};

template <prop_kind_t aprop>
struct rnn_brgemm_t;

template <>
struct rnn_brgemm_t<prop_kind::forward> : public rnn_brgemm_base_t {

    using rnn_brgemm_base_t::init_scratchpad;

    static status_t configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
            alg_kind_t cell_kind, dim_t src_layer_type_size,
            dim_t scratch_type_size);

    status_t init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
            data_type_t src_type, data_type_t weights_type);

    brgemm_desc_t desc_layer_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_b1_[num_base_kernels_];

    brgemm_desc_t desc_layer_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_iter_N_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_layer_K1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_layer_NK1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_iter_K2_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_iter_NK2_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_layermerged_b0_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_N_tail_b0_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_K1_tail_b1_[num_base_kernels_];
    brgemm_desc_t desc_layermerged_NK1_tail_b1_[num_base_kernels_];

    brgemm_desc_t desc_proj_b0_[num_proj_kernels_];
    brgemm_desc_t desc_proj_N_tail_b0_[num_proj_kernels_];
    brgemm_desc_t desc_proj_N_tail_b1_[num_proj_kernels_];

    // Set of brgemm descriptor for 2nd part of iteration gemm in vanulla GRU cell
    brgemm_desc_t desc_iter_p2_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t desc_iter_p2_N_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t desc_iter_p2_K2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_desc_t
            desc_iter_p2_NK2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];

    brgemm_ker_ptr_t kernel_layer_b0_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_iter_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_layer_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_N_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_layer_K1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layer_NK1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_K2_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_iter_NK2_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_layermerged_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_N_tail_b0_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_K1_tail_b1_[num_base_kernels_];
    brgemm_ker_ptr_t kernel_layermerged_NK1_tail_b1_[num_base_kernels_];

    brgemm_ker_ptr_t kernel_proj_b0_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_N_tail_b0_[num_proj_kernels_];
    brgemm_ker_ptr_t kernel_proj_N_tail_b1_[num_proj_kernels_];

    // Set of brgemm kernels for 2nd part of iteration gemm in vanilla GRU cell
    brgemm_ker_ptr_t kernel_iter_p2_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_N_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_K2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
    brgemm_ker_ptr_t
            kernel_iter_p2_NK2_tail_b1_[num_vanilla_gru_iter_part2_kernels_];
};

struct rnn_diff_src_brgemm_t {
    brgemm_desc_t desc_iter_layer_beta0_;
    brgemm_desc_t desc_layer_N_tail_beta0_;
    brgemm_desc_t desc_iter_N_tail_beta0_;
    brgemm_desc_t desc_iter_layer_K_tail_beta1_;
    brgemm_desc_t desc_layer_NK_tail_beta1_;
    brgemm_desc_t desc_iter_NK_tail_beta1_;

    brgemm_ker_ptr_t kernel_iter_layer_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_N_tail_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_N_tail_beta0_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_layer_K_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_NK_tail_beta1_ = nullptr;
};

struct rnn_diff_wei_brgemm_t {
    brgemm_desc_t desc_iter_beta1_;
    brgemm_desc_t desc_layer_beta1_;

    brgemm_desc_t desc_iter_N_tail_beta1_;
    brgemm_desc_t desc_layer_N_tail_beta1_;

    brgemm_desc_t desc_iter_NK_tail_beta1_;
    brgemm_desc_t desc_layer_NK_tail_beta1_;

    brgemm_desc_t desc_iter_K_tail_beta1_;
    brgemm_desc_t desc_layer_K_tail_beta1_;

    brgemm_ker_ptr_t kernel_iter_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_N_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_NK_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_iter_K_tail_beta1_ = nullptr;
    brgemm_ker_ptr_t kernel_layer_K_tail_beta1_ = nullptr;

    scratch_gates_reorder_ker_ptr_t scratch_gates_reorder_kernel_;
};

template <>
struct rnn_brgemm_t<prop_kind::backward> : public rnn_brgemm_base_t {
public:
    static void init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
            memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
            dim_t gemm_acc_align);

    static status_t configure_brgemm(cpu::rnn_utils::rnn_conf_t &rnn,
            alg_kind_t cell_kind, dim_t src_layer_type_size,
            dim_t scratch_type_size);

    status_t init_kernels(const cpu::rnn_utils::rnn_conf_t &rnn,
            data_type_t src_type, data_type_t weights_type);

    rnn_diff_src_brgemm_t diff_src_;
    rnn_diff_wei_brgemm_t diff_wei_;

private:
    static void configure_brgemm_peephole(cpu::rnn_utils::rnn_conf_t &rnn);
    status_t init_peephole_kernels(const cpu::rnn_utils::rnn_conf_t &rnn);
};

} // namespace rnn_brgemm_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_RNN_RNN_BRGEMM_UTILS_HPP
