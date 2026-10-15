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

#ifndef CPU_X64_MATMUL_POSTOPS_ESTIMATOR_HPP
#define CPU_X64_MATMUL_POSTOPS_ESTIMATOR_HPP

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

// The postops estimator JIT-compiles post-operation code for a single
// cache line result produced by the GEMM. The size of the generated code
// serves as a metric to help blocking heuristics account for the impact
// of post-operations on execution performance. Note: the generated code
// is not executed.

class postops_estimator_t : public jit_generator_t {
public:
    static status_t estimate_insts_per_cacheline(memory_desc_t &dst_md,
            primitive_attr_t &attr, int &estimated_vec_insts) {
        postops_estimator_t post_ops_gen(dst_md, attr);
        status_t res = post_ops_gen.check_status();
        if (res != status::success) return res;
        post_ops_gen.generate();
        res = post_ops_gen.check_status();
        if (res != status::success) return res;

        size_t code_size = post_ops_gen.getSize();
        const size_t estimated_vec_code_size
                = code_size > mean_none_vec_code_bytes
                ? code_size - mean_none_vec_code_bytes
                : 0;
        estimated_vec_insts = static_cast<int>(
                estimated_vec_code_size / mean_vec_inst_bytes);
        return status::success;
    }

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<Xbyak::Zmm>;
    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;
    std::unique_ptr<po_injector_t> postops_injector_;
    static constexpr size_t mean_none_vec_code_bytes = 8;
    static constexpr size_t mean_vec_inst_bytes = 7;

    postops_estimator_t(memory_desc_t &dst_md, primitive_attr_t &attr)
        : jit_generator_t("dummy_generator",
                  impl::cpu::x64::cpu_isa_t::avx512_core_amx) {
        auto dsc = memory_desc_wrapper(dst_md);

        // Binary post-ops with fp8 src1 require converters. Their lookup
        // tables are not emitted since the generated code is never executed.
        f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(this,
                Xbyak::Zmm(31), Xbyak::Zmm(30), Xbyak::Zmm(29),
                Xbyak::Opmask(2), this->r12);
        f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this,
                Xbyak::Zmm(31), Xbyak::Zmm(30), Xbyak::Zmm(29), Xbyak::Zmm(28),
                Xbyak::Zmm(27), this->r12);

        const dnnl::impl::cpu::x64::binary_injector::rhs_arg_static_params_t
                rhs_sp(Xbyak::Zmm(1).getIdx(), this->r14, this->r15, this->r13,
                        false, false, static_cast<size_t>(0),
                        static_cast<size_t>(0), dsc, static_cast<size_t>(0),
                        Xbyak::Opmask(0), false);

        const dnnl::impl::cpu::x64::binary_injector::static_params_t bsp(
                this->param1,
                dnnl::impl::cpu::x64::binary_injector::
                        get_all_strategies_supported_by_injector(),
                rhs_sp, f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get());

        dnnl::impl::cpu::x64::eltwise_injector::static_params_t esp;
        esp.preserve_vmm = false;
        esp.preserve_p_table = false;

        postops_injector_ = utils::make_unique<po_injector_t>(this,
                attr.post_ops_, bsp, esp,
                // The estimate leaves sum out, as it always has.
                /* inject_sum = */ false);
    }

    const char *name() const final {
        assert(!"postops_estimator_t name function is not expected to be "
                "called. The jitted code is not to be registered or used");
        return nullptr;
    }
    const char *source_file() const final {
        assert(!"postops_estimator_t source_file function is not expected to be"
                " called. The jitted code is not to be registered or used");
        return nullptr;
    }

    status_t check_status() {
        if (postops_injector_ == nullptr) return status::out_of_memory;
        int err_code = Xbyak::GetError();
        if (err_code == Xbyak::ERR_CANT_ALLOC) return status::out_of_memory;
        if (err_code != Xbyak::ERR_NONE) return status::runtime_error;
        return status::success;
    }

    void generate() final { postops_injector_->compute_vector(0); }
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
