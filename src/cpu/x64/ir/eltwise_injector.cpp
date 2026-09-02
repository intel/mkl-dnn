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

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/ir/eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename Vmm>
using injector_t = jit_uni_eltwise_injector_t<Vmm>;

namespace {

// Create an injector and return it type-erased. `save_state = true` makes the
// injector preserve on the stack every register it borrows, so it does not
// participate in the IR register allocation.
template <typename Vmm>
std::shared_ptr<void> create_injector(jit_generator_t &gen, alg_kind_t alg,
        float alpha, float beta, float scale) {
    return std::make_shared<injector_t<Vmm>>(&gen, alg, alpha, beta, scale,
            data_type::f32, /* save_state = */ true);
}

template <typename Vmm>
injector_t<Vmm> *cast2tgt(void *injector) {
    return static_cast<injector_t<Vmm> *>(injector);
}

} // namespace

eltwise_injector_t::eltwise_injector_t(jit_generator_t &gen, cpu_isa_t isa,
        alg_kind_t alg, float alpha, float beta, float scale)
    : is_zmm_(is_superset(isa, avx512_core)) {
    injector_ = is_zmm_
            ? create_injector<Xbyak::Zmm>(gen, alg, alpha, beta, scale)
            : create_injector<Xbyak::Ymm>(gen, alg, alpha, beta, scale);
    assert(injector_ && "ir eltwise injector creation failed");
}

void eltwise_injector_t::apply(int vec_phys) {
    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())->compute_vector(vec_phys);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())->compute_vector(vec_phys);
}

void eltwise_injector_t::prepare_table() {
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
