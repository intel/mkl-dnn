/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_PRIMITIVE_ATTR_HPP
#define GPU_INTEL_PRIMITIVE_ATTR_HPP

#include <string>

#include "common/primitive_attr.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_primitive_attr_t : public primitive_attr_item_t {
    gpu_primitive_attr_t(int grf_per_thread = 0, bool use_dpas = false,
            std::string kernel_override = {})
        : grf_per_thread_(grf_per_thread)
        , use_dpas_(use_dpas)
        , kernel_override_(std::move(kernel_override)) {}

    std::unique_ptr<primitive_attr_item_t> clone() const override {
        return utils::make_unique<gpu_primitive_attr_t>(
                grf_per_thread_, use_dpas_, kernel_override_);
    }

    bool has_default_values() const override {
        return grf_per_thread_ == 0 && !use_dpas_ && kernel_override_.empty();
    }

    bool has_only_kernel_override() const {
        return grf_per_thread_ == 0 && !use_dpas_;
    }

    bool is_equal(const primitive_attr_item_t &other) const override {
        auto *other_ptr = utils::downcast<const gpu_primitive_attr_t *>(&other);
        return grf_per_thread_ == other_ptr->grf_per_thread_
                && use_dpas_ == other_ptr->use_dpas_
                && kernel_override_ == other_ptr->kernel_override_;
    }

    size_t get_hash() const override {
        size_t seed = 0;
        seed = hash_combine(seed, grf_per_thread_);
        seed = hash_combine(seed, use_dpas_);
        seed = hash_combine(seed, std::hash<std::string> {}(kernel_override_));
        return seed;
    }

    void serialize(serialization_stream_t &stream) const override {
        stream.append(grf_per_thread_);
        stream.append(use_dpas_);
        stream.append_array(kernel_override_.size(), kernel_override_.c_str());
    }

    int grf_per_thread() const { return grf_per_thread_; }
    bool use_dpas() const { return use_dpas_; }
    void set_use_dpas(bool value) { use_dpas_ = value; }

    const std::string &kernel_override() const { return kernel_override_; }
    void set_kernel_override(const std::string &value) {
        kernel_override_ = value;
    }

private:
    int grf_per_thread_;
    bool use_dpas_ = false;
    std::string kernel_override_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_set_kernel_override(
        dnnl_primitive_attr_t attr, const char *kernel);

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_override(
        const_dnnl_primitive_attr_t attr, const char **kernel);

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_gemm_kernel(
        const_dnnl_primitive_desc_t pd, const char **kernel);

#endif
