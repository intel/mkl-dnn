/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2022-2024 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_AARCH64_INJECTORS_JIT_UNI_BINARY_INJECTOR_HPP

#include <functional>
#include <map>
#include <utility>
#include <unordered_set>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/binary_injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace binary_injector {
using dnnl::impl::cpu::binary_injector_utils::prepare_binary_args;

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops);

bool any_binary_postop_rhs_non_scalar_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen,
        const bcast_set_t &supported_strategy_set);

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);
bool any_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> &predicate);
bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set,
        const std::function<bool(const memory_desc_wrapper &)> &predicate);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_orig_offset - offset 0 to destination tensor
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations).
 * @param tail_opmask - register with loaded by user mask, used in sve for load with
 * tail handling.
 * @param tail_size - size of processed tail in elements.
 * @param use_exact_tail_scalar_bcast - in case of scalar broadcast user can disable
 * loading data with tail, usually bcast through entire vector is faster (usually 1 instruction)
 * vs. broadcasting limited by tail size (potentially several instructions). In case
 * when user during storing ignores values from vmm above tail size, setting this option to
 * false can result in better performance.
 */
struct rhs_arg_static_params_t {
    rhs_arg_static_params_t(std::size_t abi_param_offset,
            std::size_t dst_orig_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
            bool use_exact_tail_scalar_bcast);

    bool is_opmask_set() const noexcept { return true; }
    bool is_dst_orig_set() const noexcept { return true; }

    std::size_t abi_param_offset;
    std::size_t dst_orig_offset;
    memory_desc_wrapper dst_d;
    std::size_t tail_size;
    Xbyak_aarch64::PReg tail_opmask;
    bool use_exact_tail_scalar_bcast;
    Xbyak_aarch64::XReg reg_tail_size {0};
    bool is_tail;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator.
 * @param bcast_set_t supported_strategy_set - set allowing disabling particular
 * bcast strategies
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak_aarch64::XReg &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params);
    static_params_t(const Xbyak_aarch64::XReg &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params);

    Xbyak_aarch64::XReg param1;
    const bcast_set_t supported_strategy_set;
    rhs_arg_static_params_t rhs_arg_static_params;
};

/*
 * Mode of data load with tail for rhs:
 * STATIC - load based on given integer.
 * DYNAMIC - load based on opmask or 64-bit register.
 * DEFAULT - DYNAMIC for sve512, STATIC for others.
 */

enum class tail_lode_mode_t { STATIC, DYNAMIC, DEFAULT };

struct rhs_address_t {
    Xbyak_aarch64::XReg base_;
    int64_t offt_ = 0;
    bool isBroadcast_ = false;
    uint32_t bits_ = 0;

    rhs_address_t(const Xbyak_aarch64::XReg &base, const int64_t offt = 0,
            bool isBroadcast = false, const uint32_t bits = 0)
        : base_(base), offt_(offt), isBroadcast_(isBroadcast), bits_(bits) {}

    bool operator==(const rhs_address_t &rhs) const {
        return getBit() == rhs.getBit()
                && getBase().getIdx() == rhs.getBase().getIdx()
                && isBroadcast_ == rhs.isBroadcast();
    }
    bool operator!=(const rhs_address_t &rhs) const { return !operator==(rhs); }

    Xbyak_aarch64::XReg getBase() const { return base_; }
    uint32_t getBit() const { return bits_; }
    bool isBroadcast() const { return isBroadcast_; }
};

struct rhs_operand_t {

    bool isAddress_ = false;
    uint32_t idx_;
    rhs_address_t address_;

    bool operator==(const rhs_operand_t &rhs) const {
        if (isAddress_) {
            if (isAddress_ == rhs.isAddress_ && address_ == rhs.address_)
                return true;
            else
                return false;
        } else {
            if (isAddress_ == rhs.isAddress_ && idx_ == rhs.idx_)
                return true;
            else
                return false;
        }
    }

    bool operator!=(const rhs_operand_t &rhs) const { return !operator==(rhs); }

    bool isBroadcast() const { return isAddress_ && address_.isBroadcast(); }
};

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call.
 * Contains configurable std::maps where key is vmm index and value is
 * offset in elements. The offset value identifies tensor slice in particular
 * vmm. This is utilized by broadcasting mechanism. Offset, depending on the
 * implementation particular kernels, can be passed as value (usually during
 * unrolling), inside operand, under memory address.
 *
 * @param vmm_idx_to_out_addr - vmm mapped to address of destination tensor with offset,
 * used to calculate offset in no_broadcast strategy, but also in other strategies whose
 * calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_reg - vmm mapped to register containing address of destination
 * with offset, used to calculate offset in no_broadcast strategy, but also in other
 * strategies whose calculations are based on no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_addr - vmm mapped to offset in elements stored under
 * memory address intended to use in no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_addr - vmm mapped to offset in elements stored under
 * memory address intended to use in no_broadcast strategy.
 * @param vmm_idx_to_out_elem_off_val - vmm mapped to offset in elements passed as raw
 * value intended to use in no_broadcast strategy
 * @param vmm_idx_to_out_off_oprnd - vmm mapped to offset in elements inside operand
 * intended to use in no_broadcast strategy
 * @param vmm_idx_to_oc_elem_off_addr - vmm mapped to output channel offset in elements
 * stored under memory address intended to use in per_oc broadcast strategies.
 * @param vmm_idx_to_oc_elem_off_val - vmm mapped to  output channel offset in elements
 * passed as raw value intended to use in per_oc broadcast strategies.
 * @param vmm_idx_to_oc_off_oprnd - vmm mapped to output channel offset in elements inside
 * operand intended to use in per_oc broadcast strategies.
 * @param vmm_idx_to_sp_elem_off_addr - vmm mapped to proper output spatial offset in
 * elements stored under memory address intended to use in per_mb_spatial strategies.
 * @param vmm_idx_to_sp_elem_off_val - vmm mapped to proper output spatial offset in
 * elements passed as raw value intended to use in per_mb_spatial strategies.
 * @param vmm_idx_to_sp_off_oprnd - vmm mapped to proper output spatial offset in 
 * elements inside operand intended to use in per_mb_spatial strategies.
 * @param vmm_idx_to_mb_w_elem_off_addr - vmm mapped to proper output last dim
 * per first dim offset in elements stored under memory address intended to use
 * in per_mb_w strategies.
 * @param vmm_idx_to_mb_w_elem_off_val - vmm mapped to proper output last dim
 * per first dim offset in elements passed as raw value intended to use in
 * per_mb_w strategies.
 * @param vmm_idx_to_mb_w_off_oprnd - vmm mapped to proper output last dim
 * per first dim offset in elements inside operand intended to use in per_mb_w
 * strategies.
 * @param vmm_idx_to_w_elem_off_addr - vmm mapped to proper output last dim
 * offset in elements stored under memory address intended to use in per_w strategy.
 * @param vmm_idx_to_w_elem_off_val - vmm mapped to proper output last dim
 * offset in elements passed as raw value intended to use in per_w strategy.
 * @param vmm_idx_to_w_off_oprnd - vmm mapped to proper output last dim offset
 * in elements inside operand intended to use in per_w strategy.
 * @param vmm_tail_idx - vmm indices that contains data don't fill the whole vector (tail).
 * @param is_dynamic_tail_load - determines whether to load with tail in
 * runtime (based on the value from reg_tail_size or opmask) or based on given
 * integer.
 */

struct rhs_arg_dynamic_params_t {
    std::map<int, rhs_address_t> vmm_idx_to_out_addr;
    std::map<int, Xbyak_aarch64::XReg> vmm_idx_to_out_reg;

    std::map<int, rhs_address_t> vmm_idx_to_out_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_out_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_out_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_oc_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_oc_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_oc_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_sp_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_sp_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_sp_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_mb_w_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_mb_w_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_mb_w_off_oprnd;

    std::map<int, rhs_address_t> vmm_idx_to_w_elem_off_addr;
    std::map<int, size_t> vmm_idx_to_w_elem_off_val;
    std::map<int, rhs_operand_t> vmm_idx_to_w_off_oprnd;

    std::unordered_set<int> vmm_tail_idx_;
    tail_lode_mode_t tail_load_mode = tail_lode_mode_t::DEFAULT;
};

/*
 * Checks if src1 data type is supported by binary injector.
 */
bool is_data_supported(data_type_t data_type);

/*
 * Checks if broadcast of src1 is supported by binary injector.
 */
bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Checks if binary injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, alg_kind_t alg,
        const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Main mechanism responsible for injecting binary postops supporting various
 * isa: sve
 * types: f32, s32, u8, s8.
 */
template <cpu_isa_t isa>
class jit_uni_binary_injector_t {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;
    using Vmm = typename cpu_isa_traits<isa>::TReg;

public:
    jit_uni_binary_injector_t(jit_generator_t *host,
            Xbyak_aarch64::RegPoolManager &reg_pool,
            const static_params_t &static_params);

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. Function loads appropriate
     * slice of rhs tensor for computations based on internally determined
     * broadcast strategy and information about stored data in particular vmm
     * described inside rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes. Function loads
     * appropriate slice of rhs tensor for computations based on internally
     * determined broadcast strategy and information about stored data in particular
     * vmm described inside rhs_arg_params.
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular vmm described inside rhs_arg_params.
     */
    void compute_vector(size_t idx, std::size_t rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

private:
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Vmm(vmm_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns
     * ptr to it.
     */
    rhs_address_t prepare_rhs_arg_addr(std::size_t vmm_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy,
            const Xbyak_aarch64::XReg &rhs_addr) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, Vmm dst,
            const rhs_address_t &rhs_addr, bool with_tail,
            const tail_lode_mode_t tail_load_mode) const;

    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_offset_from_operand(
            const std::map<int, rhs_operand_t> &vmm_idx_to_elem_addr_off,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void append_offset_under_mem_addr(
            const std::map<int, rhs_address_t> &vmm_idx_to_elem_addr_off,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void append_value_offset(
            const std::map<int, size_t> &vmm_idx_to_elem_val_off, int vmm_idx,
            const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;

    void append_no_broadcast_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void calculate_no_broadcast(rhs_address_t addr, std::size_t offset,
            const Xbyak_aarch64::XReg &out_reg) const;

    void append_oc_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void calculate_oc_ncsp(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_oc_blocked(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_oc_nspc(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_oc_cspn(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;

    void append_mb_sp_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_sp_ncsp(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_sp_blocked(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_sp_nspc(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_sp_cspn(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;

    void append_mb_w_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void calculate_mb_w_ncsp(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_w_blocked(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_w_nspc(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_mb_w_cspn(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;

    void append_w_offset(
            const std::map<int, rhs_address_t> &vmm_idx_to_out_addr,
            const std::map<int, Xbyak_aarch64::XReg> &vmm_idx_to_out_reg,
            const std::map<int, size_t> &vmm_idx_to_out_elem_off_val,
            int vmm_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;
    void calculate_w_ncsp(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_w_blocked(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_w_nspc(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;
    void calculate_w_cspn(const dim_t *strides,
            const Xbyak_aarch64::XReg &offset,
            const Xbyak_aarch64::XReg &result) const;

    void compute_cmp_mask(const Xbyak_aarch64::PReg &cmp_dst,
            const Xbyak_aarch64::PReg &mask, const Xbyak_aarch64::ZReg &cmp_src,
            const Xbyak_aarch64::ZReg &cmp_src2,
            alg_kind_t cmp_predicate) const;

    template <typename T>
    void fdiv(const Vmm &dst, const Xbyak_aarch64::PReg &mask, const Vmm &lhs,
            const T &rhs) const;
    template <typename T>
    void fdiv(const Vmm &dst, const Vmm &lhs, const T &rhs) const;

    void execute_cmp_binary(const Vmm &dst, const Xbyak_aarch64::PReg &mask,
            const Vmm &lhs, const Vmm &rhs, alg_kind_t cmp_predicate) const;
    template <typename T>
    void execute_binary(alg_kind_t binary_alg, const Vmm &dst,
            const Xbyak_aarch64::PReg &mask, const Vmm &lhs, const T &rhs,
            const Vmm &rhs_scratch) const;
    /*
     * Used in scalar broadcast strategy, broadcasting single value of given
     * data type over entire vector Vmm register.
     */
    void execute_broadcast(const data_type_t &data_type, const Vmm &tmp_reg,
            const rhs_address_t &rhs_addr,
            const tail_lode_mode_t tail_load_mode, bool with_tail) const;
    void load_rhs(const data_type_t &data_type, const Vmm &tmp_reg,
            const rhs_address_t &rhs_addr,
            const tail_lode_mode_t tail_load_mode, bool with_tail) const;
    void execute_broadcast_tail_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_reg, const rhs_address_t &rhs_addr) const;
    void load_rhs_tail_dynamically_with_opmask(const data_type_t &data_type,
            const Vmm &tmp_vmm, const rhs_address_t &rhs_addr) const;
    void load_rhs_tail_statically(const data_type_t &data_type,
            const Vmm &tmp_vmm, const rhs_address_t &rhs_addr) const;
    void execute_broadcast_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const rhs_address_t &rhs_addr) const;
    void execute_broadcast_s8u8_no_tail(const data_type_t &data_type,
            const Vmm &tmp_vmm, const rhs_address_t &rhs_addr) const;
    void load_rhs_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const rhs_address_t &rhs_addr) const;
    void load_rhs_i8_no_tail(const data_type_t &data_type, const Vmm &tmp_reg,
            const rhs_address_t &rhs_addr) const;
    /*
     * Returns pair consisting of flag indication preservation is needed for vmm
     * index in second member that should be used as temporary vmm inside inject
     * binary.
     */
    /*
     * Used in isa != sve512 where m32bcst is not supported, replaces ptr_b
     * with ptr.
     */
    rhs_address_t remove_bcast_bit(const rhs_address_t &rhs_addr) const;

    jit_generator_t *host_;
    mutable rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak_aarch64::XReg param1_;
    const bcast_set_t supported_strategy_set_;
    Xbyak_aarch64::RegPoolManager &reg_pool_;

    static constexpr int sizeof_reg64 = 8;
    /*
     * When using benchdnn zmalloc_protect doesn't guarantee that tensor memory
     * address is 64 byte aligned, which can cause segmentation fault.
     */
    static constexpr bool binary_op_with_unaligned_mem_operand_allowed_ = true;
};

} // namespace binary_injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
