/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#include <algorithm>
#include <cstddef>
#include <functional>

#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"

#include "rvjit/implementation/rvv/linalg/tuning.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/linalg/matmul/postop/chain.hpp"
#include "rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

namespace matmul = ::rvjit::linalg::matmul;

// Latency, in cycles, of the cache the vector loads read from.
constexpr int cache_latency = 4;

// Largest dimension bound handed to rvjit, which sizes them as int.
constexpr dim_t max_dim = 1 << 30;

rvjit::type_t to_rvjit(data_type_t dt) {
    switch (dt) {
        case data_type::f32: return rvjit::type_t::f32;
        case data_type::bf16: return rvjit::type_t::bf16;
        case data_type::f16: return rvjit::type_t::f16;
        case data_type::s8: return rvjit::type_t::s8;
        case data_type::u8: return rvjit::type_t::u8;
        case data_type::s32: return rvjit::type_t::s32;
        default: return rvjit::type_t::none;
    }
}

rvjit::type_parameter_t to_rvjit(const rvjit_type_t &t) {
    const rvjit::type_t mem = to_rvjit(t.mem);
    return rvjit::type_parameter_t(
            mem, t.reg == data_type::undef ? mem : to_rvjit(t.reg));
}

// Construct the rvjit VPU model based on the host's characteristics
rvjit::rvv::vpu_t make_model() {
    rvjit::rvv::vpu_t model;
    model.from_params(static_cast<int>(get_platform_vlen()),
            static_cast<int>(platform::get_per_core_cache_size(1)),
            platform::get_cache_line_size(), cache_latency);
    return model;
}

// a0 holds the arguments of the whole kernel, and the default list leaves it
// out of the pool
rvjit::rvv::buffer::architecture_t make_architecture() {
    return rvjit::rvv::buffer::architecture_t::default_list();
}

rvjit::source_t from_args(ptrdiff_t off) {
    return rvjit::source_t(
            rvjit::pointer_t(Xbyak_riscv::a0, static_cast<int>(off)));
}

rvjit::parameter_t to_parameter(const rvjit_value_t &v) {
    if (v.is_literal()) return rvjit::parameter_t(static_cast<int>(v.value));
    return rvjit::parameter_t(from_args(v.arg));
}

rvjit::size_parameter_t to_size(const rvjit_value_t &v) {
    if (v.is_literal())
        return rvjit::size_parameter_t(static_cast<int>(v.value));
    rvjit::size_parameter_t size(from_args(v.arg));
    if (v.value > 0)
        size.as_less_eq(
                static_cast<int>(utils::saturate<dim_t>(1, max_dim, v.value)));
    return size;
}

rvjit::scalar_parameter_t to_scalar(double value, ptrdiff_t arg) {
    if (arg == rvjit_value_t::literal_value)
        return rvjit::scalar_parameter_t(value);
    return rvjit::scalar_parameter_t(from_args(arg));
}

// The matmul plan `d` stands for.
void make_plan(const rvjit_matmul_desc_t &d, matmul::plan_t &p) {
    p.dta = to_rvjit(d.a);
    p.dtb = to_rvjit(d.b);
    p.dtc = to_rvjit(d.c);
    p.dtepi = to_rvjit(d.epilogue);
    p.dtbias = to_rvjit(d.bias);
    p.bias_kind = d.bias_kind == rvjit_bias_kind_t::scalar
            ? matmul::bias_kind_t::scalar
            : matmul::bias_kind_t::per_m;
    p.is_transa = d.trans_a;
    p.is_transb = d.trans_b;

    p.m = to_size(d.m);
    p.n = to_size(d.n);
    p.k = to_size(d.k);
    p.lda = to_parameter(d.lda);
    p.ldb = to_parameter(d.ldb);
    p.ldc = to_parameter(d.ldc);

    p.ptra = rvjit::parameter_t(from_args(static_cast<ptrdiff_t>(d.ptr_a)));
    p.ptrb = rvjit::parameter_t(from_args(static_cast<ptrdiff_t>(d.ptr_b)));
    p.ptrc = rvjit::parameter_t(from_args(static_cast<ptrdiff_t>(d.ptr_c)));

    p.flags = 0;
    if (d.features & rvjit_alpha) {
        p.flags |= matmul::with_alpha;
        p.alpha = to_scalar(d.alpha, d.alpha_arg);
    }
    if (d.features & rvjit_beta) {
        p.flags |= matmul::with_beta;
        p.beta = to_scalar(d.beta, d.beta_arg);
    }
    if (d.features & rvjit_bias) {
        p.flags |= matmul::with_bias;
        p.bias = rvjit::parameter_t(from_args(d.ptr_bias));
    }
    if (d.rt_flags != rvjit_value_t::literal_value)
        p.rt_flags = rvjit::parameter_t(from_args(d.rt_flags));
}

} // namespace

status_t rvjit_matmul_tune(
        const rvjit_matmul_desc_t &desc, rvjit_matmul_blocks_t &blocks) {
    matmul::plan_t plan;
    make_plan(desc, plan);

    const rvjit::rvv::vpu_t model = make_model();
    rvjit::rvv::linalg::matmul::tuning_t tuning(model);
    if (desc.lmul > 0) tuning.set_lmul(rvjit::rvv::lmul_for(desc.lmul));
    rvjit::rvv::buffer::architecture_t arch = make_architecture();
    rvjit::rvv::buffer::buffer_pool_t pool(arch);
    if (!pool.configure_vectors(tuning.vector_layout(plan)))
        return status::unimplemented;

    const auto epilogue = [&](int mb) {
        matmul::needs_t needs;
        matmul::chain_needs(plan, mb, 1, needs);
        return needs;
    };
    rvjit::rvv::linalg::matmul::tuned_t tuned;
    if (!tuning(plan, pool, epilogue, tuned)) return status::unimplemented;

    blocks.m_unroll = tuned.mb;
    blocks.m_grain = tuned.m_grain;
    blocks.n_unroll = tuned.nb;
    blocks.n_grain = tuned.n_grain;
    blocks.k_unroll = tuned.kb;
    blocks.k_grain = tuned.k_grain;
    return status::success;
}

uint64_t rvjit_runtime_flags(bool alpha, bool beta, bool bias) {
    int flags = 0;
    if (alpha) flags |= matmul::with_alpha;
    if (beta) flags |= matmul::with_beta;
    if (bias) flags |= matmul::with_bias;
    return static_cast<uint64_t>(flags);
}

struct jit_rvjit_matmul_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvjit_matmul_t)

    jit_rvjit_matmul_t(const rvjit_matmul_desc_t &desc,
            const rvjit_matmul_blocks_t &blocks)
        : jit_generator_t("rv64_matmul_kernel_rvjit")
        , desc_(desc)
        , blocks_(blocks) {}

    status_t create_kernel() override {
        const status_t st = jit_generator_t::create_kernel();
        return generated_ ? st : status::runtime_error;
    }

protected:
    void generate() override {
        const rvjit::rvv::vpu_t model = make_model();
        rvjit::rvv::base_programming_model_t pm(
                *this, model, make_architecture());
        rvjit::rvv::linalg::matmul::impl_t impl(pm);
        if (desc_.lmul > 0) impl.set_lmul(rvjit::rvv::lmul_for(desc_.lmul));
        matmul::matmul_t backend(impl);

        // The tuned blocks, honored as given in granules
        matmul::plan_t plan;
        make_plan(desc_, plan);
        plan.blocks.mb = blocks_.m_unroll;
        plan.blocks.nb = blocks_.n_unroll;
        plan.blocks.kb = blocks_.k_unroll;

        generated_ = backend.init(plan);
        if (!generated_) {
            ret();
            return;
        }

        pm.stack.preserve();
        backend();
        pm.stack.restore();
        ret();
    }

private:
    rvjit_matmul_desc_t desc_;
    rvjit_matmul_blocks_t blocks_;
    bool generated_ = true;
};

rvjit_matmul_kernel_t::rvjit_matmul_kernel_t(
        const rvjit_matmul_desc_t &desc, const rvjit_matmul_blocks_t &blocks)
    : jit_(new jit_rvjit_matmul_t(desc, blocks)) {}

rvjit_matmul_kernel_t::~rvjit_matmul_kernel_t() {
    delete jit_;
}

status_t rvjit_matmul_kernel_t::create_kernel() {
    return jit_->create_kernel();
}

void rvjit_matmul_kernel_t::operator()(const void *args) const {
    (*jit_)(args);
}

status_t rvjit_matmul_kernel_create(const rvjit_matmul_desc_t &desc,
        const rvjit_matmul_blocks_t &blocks,
        std::unique_ptr<rvjit_matmul_kernel_t> &kernel) {
    kernel.reset(new rvjit_matmul_kernel_t(desc, blocks));
    return kernel->create_kernel();
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
