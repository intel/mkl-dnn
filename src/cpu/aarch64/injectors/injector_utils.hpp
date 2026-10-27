/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_INJECTORS_INJECTOR_UTILS_HPP
#define CPU_AARCH64_INJECTORS_INJECTOR_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;

// Run an injector which needs scratch_count vector registers. The caller must
// claim every active result in reg_pool before entering this helper.
//
// If scratch is short, exchange equal-sized result subsets through the stack.
// Other live vectors may be post-op inputs and cannot be spill candidates.
template <cpu_isa_t isa, typename Compute>
void compute_vector_range_with_spill(jit_generator_t *host,
        Xbyak_aarch64::RegPoolManager &reg_pool,
        const vmm_index_set_t &vmm_idxs, size_t scratch_count,
        const Compute &compute) {
    using TReg = typename cpu_isa_traits<isa>::TReg;

    if (vmm_idxs.empty()) return;
    assert(std::all_of(vmm_idxs.begin(), vmm_idxs.end(), [&](size_t idx) {
        return reg_pool.vec_idx_in_use(static_cast<int>(idx));
    }));

    const size_t free_count = reg_pool.get_all_free_vec().size()
            + reg_pool.get_all_preserved_vec().size();

    // Happy path: enough free registers for injector scratch.
    if (free_count >= scratch_count) {
        compute(vmm_idxs);
        return;
    }

    const size_t spill_count = scratch_count - free_count;

    // Each spilled result needs a resident result to exchange with.
    assert(spill_count <= vmm_idxs.size() / 2);

    const auto spill = [&](const vmm_index_set_t &idxs) {
        size_t slot = 0;
        for (const auto idx : idxs) {
            const TReg reg(static_cast<int>(idx));
            host->uni_store_vmm_stack(reg, host->sp, slot++);
            reg_pool.free(reg);
        }
    };
    const auto restore = [&](const vmm_index_set_t &idxs) {
        size_t slot = 0;
        for (const auto idx : idxs) {
            const TReg reg(static_cast<int>(idx));
            reg_pool.alloc<TReg>(reg.getIdx());
            host->uni_load_vmm_stack(reg, host->sp, slot++);
        }
    };

    vmm_index_set_t spilled_idxs;
    vmm_index_set_t resident_idxs(vmm_idxs);
    auto spill_it = resident_idxs.begin();
    for (size_t i = 0; i < spill_count; ++i) {
        spilled_idxs.insert(*spill_it);
        spill_it = resident_idxs.erase(spill_it);
    }

    host->uni_reserve_vmm_stack<TReg>(spill_count);
    spill(spilled_idxs);
    compute(resident_idxs);

    vmm_index_set_t swap_idxs;
    auto swap_it = resident_idxs.begin();
    for (size_t i = 0; i < spill_count; ++i, ++swap_it)
        swap_idxs.insert(*swap_it);

    restore(spilled_idxs);
    spill(swap_idxs);
    compute(spilled_idxs);
    restore(swap_idxs);
    host->uni_release_vmm_stack<TReg>(spill_count);
}

enum class layout_t { ncsp, c_blocked, nspc, cspn, unsupported };

inline layout_t get_layout_type(const memory_desc_wrapper &dst_d) {
    const auto strides = dst_d.blocking_desc().strides;
    if (!dst_d.is_plain()) return layout_t::c_blocked;
    if (strides[0] >= strides[1]
            && IMPLICATION(dst_d.ndims() >= 3, strides[1] >= strides[2]))
        return layout_t::ncsp;
    if (strides[1] == 1) return layout_t::nspc;
    if (strides[0] == 1) return layout_t::cspn;
    return layout_t::unsupported;
}

/*
 * Scope guard for general purpose register and vector registers preservation.
 * Pushes registers to stack during construction and pops during destruction.
 */
template <cpu_isa_t isa>
class register_preserve_guard_t {

public:
    register_preserve_guard_t(jit_generator_t *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    register_preserve_guard_t(register_preserve_guard_t &&other) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&other)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();
    size_t calc_vmm_to_preserve_size_bytes(
            const std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
            const;
    size_t stack_space_occupied() const;

private:
    jit_generator_t *host_;
    std::vector<Xbyak_aarch64::XReg> gpr_regs_;
    std::vector<Xbyak_aarch64::VReg> vec_regs_;
    size_t vmm_to_preserve_size_bytes_;
};

template <cpu_isa_t isa>
class conditional_register_preserve_guard_t
    : public register_preserve_guard_t<to_vla_sve(isa)> {
public:
    conditional_register_preserve_guard_t(bool condition_to_be_met,
            jit_generator_t *host,
            std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
            std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve = {});
    DNNL_DISALLOW_COPY_AND_ASSIGN(conditional_register_preserve_guard_t);
};

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
