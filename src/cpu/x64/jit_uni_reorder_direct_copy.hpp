/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_REORDER_DIRECT_COPY_HPP
#define CPU_X64_JIT_UNI_REORDER_DIRECT_COPY_HPP

#include "common/c_types_map.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_reorder_direct_copy_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T(
                "jit_direct_copy:uni", jit_uni_reorder_direct_copy_t);

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

        cpu_isa_t isa_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    struct kernel_base_t {
        virtual void operator()(
                const void *src, void *dst, dim_t work_amount) const
                = 0;
        static kernel_base_t *create(const reorder_pd_t *pd, cpu_isa_t isa);
        virtual status_t create_kernel() = 0;
        virtual ~kernel_base_t() = default;

    protected:
        kernel_base_t(const reorder_pd_t *pd) : pd_(pd) {}

        // `pd_` is needed to access its members (such as `attr()`) in
        // `generate()` call.
        const reorder_pd_t *pd_;
    };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<kernel_base_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
