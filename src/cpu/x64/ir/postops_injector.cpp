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

#include <cassert>

#include "common/memory_desc_wrapper.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename Vmm>
using injector_t = injector::jit_uni_postops_injector_t<Vmm>;

namespace {

// Create an injector and return it type-erased. The ISA it generates for comes
// from the host generator.
template <typename Vmm>
std::shared_ptr<void> create_injector(jit_generator_t &gen,
        const post_ops_t &post_ops, const binary_injector::static_params_t &bsp,
        const eltwise_injector::static_params_t &esp) {
    return std::make_shared<injector_t<Vmm>>(
            &gen, post_ops, bsp, esp, /* inject_sum = */ true);
}

template <typename Vmm>
injector_t<Vmm> *cast2tgt(void *injector) {
    return static_cast<injector_t<Vmm> *>(injector);
}

} // namespace

postops_injector_t::postops_injector_t(jit_generator_t &gen, cpu_isa_t isa,
        const post_ops_t &post_ops, const memory_desc_t &dst_md,
        const bcast_set_t &bcast_set, const Xbyak::Reg64 &param_reg,
        int rhs_arg_offset, dim_t dst_orig_off, int tail_elems,
        int eltwise_opmask, int binary_tail_opmask)
    : is_zmm_(is_superset(isa, avx512_core))
    , tail_elems_(tail_elems)
    , binary_tail_opmask_(binary_tail_opmask) {
    const memory_desc_wrapper dst_d(dst_md);

    // The eltwise injector overwrites its mask and the binary injector reads
    // its own across the same call, so sharing one register would destroy the
    // tail pattern.
    JIT_ASSERT(eltwise_opmask != binary_tail_opmask
            && "postops_injector: eltwise and binary opmask must differ");
    // `init()` builds the tail pattern with a 64-bit shift.
    JIT_ASSERT(tail_elems < 64 && "postops_injector: tail is too wide");

    // The injector preserves the state of the gpr and vec registers it borrows.
    static constexpr bool preserve_gpr = true;
    static constexpr bool preserve_vmm = true;
    static constexpr bool use_exact_tail_scalar_bcast = false;

    const size_t rhs_dt_helper_vmm_idx = 0;

    // Right-hand-side argument config, used by binary post-ops and, for the
    // load path only, by sum. Eltwise does not use it.
    // `rhs_arg_offset` locates the argument pointer array in the parameter
    // struct. `dst_orig_off` locates the destination origin, used to turn an
    // accumulator address into its destination position. `tail_elems` is the
    // element count a partial load reads, as an integer on avx2 and through
    // `binary_tail_opmask` on avx512.
    const binary_injector::rhs_arg_static_params_t rhs_sp {
            rhs_dt_helper_vmm_idx, gen.r14, gen.r15, gen.r13, preserve_gpr,
            preserve_vmm, rhs_arg_offset, dst_orig_off, dst_d, tail_elems,
            Xbyak::Opmask(binary_tail_opmask), use_exact_tail_scalar_bcast};

    const binary_injector::static_params_t bsp {param_reg, bcast_set, rhs_sp};

    // Everything but the mask keeps the eltwise injector's own defaults. The
    // mask has to be named because its default, `k1`, is a register the IR
    // allocator hands out.
    const eltwise_injector::static_params_t esp {/*save_state=*/true,
            /*p_table=*/Xbyak::Reg64(Xbyak::Operand::RAX),
            Xbyak::Opmask(eltwise_opmask)};

    injector_ = is_zmm_ ? create_injector<Xbyak::Zmm>(gen, post_ops, bsp, esp)
                        : create_injector<Xbyak::Ymm>(gen, post_ops, bsp, esp);
    assert(injector_ && "ir post-ops injector creation failed");

    with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;
    with_sum_ = post_ops.find(primitive_kind::sum) != -1;
    // Prelu is injected through the binary injector and needs the same
    // right-hand-side arguments, so treat it like a binary post-op. The sum
    // reads the previous destination value through those same arguments.
    needs_rhs_args_ = post_ops.find(primitive_kind::binary) != -1
            || post_ops.find(primitive_kind::prelu) != -1 || with_sum_;
}

void postops_injector_t::init(jit_generator_t &gen) const {
    // The tail opmask is read only by the binary injector. An eltwise-only
    // chain never reaches it, and on avx2* the tail is an integer inside the
    // injector, so neither case has anything to set up.
    if (!is_zmm_ || !needs_rhs_args_ || tail_elems_ <= 0) return;

    // `kxnorq` sets all 64 bits and `kshiftrq` keeps the low `tail_elems_` of
    // them. Building the mask in the k-register file needs no gpr.
    const Xbyak::Opmask k(binary_tail_opmask_);
    gen.kxnorq(k, k, k);
    gen.kshiftrq(k, k, (uint8_t)(64 - tail_elems_));
}

void postops_injector_t::inject(const std::vector<int> &acc_phys, int base_phys,
        const std::vector<dim_t> &out_byte_off, int n_elems) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (int idx : acc_phys)
        vmm_idxs.insert(idx);

    if (!needs_rhs_args_) {
        // Chains without a binary, prelu or sum post-op have nothing to address
        // relative to the destination.
        if (is_zmm_)
            cast2tgt<Xbyak::Zmm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        else
            cast2tgt<Xbyak::Ymm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        return;
    }

    // An accumulator holding the tail count has fewer valid elements than a
    // full vector, so a right-hand-side load has to stop at that count to stay
    // in bounds. `init()` has put that pattern in the tail opmask.
    const bool is_tail = tail_elems_ > 0 && n_elems == tail_elems_;

    // Map each accumulator to its destination address. A binary post-op uses it
    // to locate the corresponding right-hand-side slice, and the sum operation
    // uses it to access the previous destination value.
    binary_injector::rhs_arg_dynamic_params_t rhs_args;
    const Xbyak::Reg64 out_reg(base_phys);
    for (size_t i = 0; i < acc_phys.size(); i++) {
        rhs_args.vmm_idx_to_out_reg.emplace(acc_phys[i], out_reg);
        rhs_args.vmm_idx_to_out_elem_off_val.emplace(
                acc_phys[i], (size_t)out_byte_off[i]);
        if (is_tail) rhs_args.vmm_tail_idx_.emplace(acc_phys[i]);
    }

    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())
                ->compute_vector_range(vmm_idxs, rhs_args);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())
                ->compute_vector_range(vmm_idxs, rhs_args);
}

void postops_injector_t::maybe_prepare_table() {
    // The constant table is needed for eltwise and sum.
    if (!with_eltwise_ && !with_sum_) return;

    const bool generate = true;
    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())->prepare_table(generate);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())->prepare_table(generate);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
