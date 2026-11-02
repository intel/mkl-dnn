/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2020-2023, 2026 Arm Ltd. and affiliates
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

#ifndef CPU_CPU_ENGINE_HPP
#define CPU_CPU_ENGINE_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/engine_id.hpp"
#include "common/impl_list_item.hpp"
#include "cpu/cpu_engine_impl.hpp"

#include "cpu/platform.hpp"

#if DNNL_AARCH64 && defined(DNNL_AARCH64_USE_ACL)
#include "cpu/aarch64/acl_thread.hpp"
#endif

#define CPU_INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>()),
#define CPU_INSTANCE_X64(...) DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AVX2(...) REG_AVX2_ISA(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AVX512(...) REG_AVX512_ISA(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AMX(...) REG_AMX_ISA(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64(...) DNNL_AARCH64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64_ACL(...) \
    DNNL_AARCH64_ACL_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64_KAI(...) \
    DNNL_AARCH64_KAI_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_X64_ZEN(...) DNNL_X64_ZEN(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_RV64(...) DNNL_RV64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_PPC64(...) DNNL_PPC64_ONLY(CPU_INSTANCE(__VA_ARGS__))

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
#define CPU_INSTANCE_GROUPED(...) CPU_INSTANCE(__VA_ARGS__)
#define CPU_INSTANCE_X64_ZEN_GROUPED(...) CPU_INSTANCE_X64_ZEN(__VA_ARGS__)
#else
#define CPU_INSTANCE_GROUPED(...)
#define CPU_INSTANCE_X64_ZEN_GROUPED(...)
#endif

namespace dnnl {
namespace impl {
namespace cpu {

#define DECLARE_IMPL_LIST(kind) \
    const impl_list_item_t *get_##kind##_impl_list(const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(group_normalization);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling);
DECLARE_IMPL_LIST(prelu);
DECLARE_IMPL_LIST(reduction);
DECLARE_IMPL_LIST(reorder);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);

#undef DECLARE_IMPL_LIST

class cpu_engine_impl_list_t {
public:
    static const impl_list_item_t *get_concat_implementation_list();
    static const impl_list_item_t *get_sum_implementation_list();

    static const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) {
        static const impl_list_item_t empty_list[] = {nullptr};

        // clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        return get_##kind##_impl_list((const kind##_desc_t *)desc);
        switch ((int)desc->primitive_kind) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(group_normalization);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(matmul);
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            case primitive_kind::sdpa: return empty_list;
            default: assert(!"unknown primitive kind"); return empty_list;
        }
#undef CASE
    }
    // clang-format on
};

class cpu_engine_t : public engine_t {
public:
    cpu_engine_t(impl::engine_impl_t *engine_impl) : engine_t(engine_impl) {}

    /* implementation part */

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(
            stream_t **stream, impl::stream_impl_t *stream_impl) override;

    const impl_list_item_t *get_concat_implementation_list() const override {
        return cpu_engine_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return cpu_engine_impl_list_t::get_sum_implementation_list();
    }
    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        return cpu_engine_impl_list_t::get_implementation_list(desc);
    }

protected:
    ~cpu_engine_t() override = default;
};

class cpu_engine_factory_t : public engine_factory_t {
public:
    size_t count() const override { return 1; }
    status_t engine_create(engine_t **engine, size_t index) const override {
        assert(index == 0);
        // Use engine_impl_t to ensure CPU-specific stream creation.
        *engine = new cpu_engine_t(new cpu::engine_impl_t(
                engine_kind::cpu, get_cpu_native_runtime(), 0));

#if DNNL_AARCH64 && defined(DNNL_AARCH64_USE_ACL)
        dnnl::impl::cpu::aarch64::acl_thread_utils::set_acl_threading();
#endif
        return status::success;
    }
};

engine_t *get_service_engine();

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
