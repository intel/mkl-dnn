/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "common/memory.hpp"

#include "xpu/sycl/memory_storage.hpp"

#include "gemmstone/dsl/runtime.hpp"
#include "gemmstone/microkernel/fuser.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/jit/generator_base.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/device_info.hpp"
#include "gpu/intel/sycl/engine.hpp"
#include "gpu/intel/sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    std::unique_ptr<intel::sycl::engine_t, engine_deleter_t> e(
            (new intel::sycl::engine_t(dev, ctx, index)));
    if (!e) return status::out_of_memory;

    CHECK(e->init());
    *engine = e.release();

    return status::success;
}

status_t engine_t::create_stream(
        impl::stream_t **stream, impl::stream_impl_t *stream_impl) {
    return gpu::intel::sycl::stream_t::create_stream(stream, this, stream_impl);
}

status_t engine_t::create_kernel(gpu::intel::compute::kernel_t *kernel,
        gpu::intel::jit::generator_base_t *jitter) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }
    return jitter->get_kernel(*kernel, this);
}

status_t engine_t::create_kernel(compute::kernel_t &kernel,
        const gemmstone::dsl::kernel_t &kernel_dsl) const {
    return interop_kernel_t::make(kernel,
            gemmstone::dsl::make_kernel(
                    kernel_dsl, impl()->context(), impl()->device()),
            {});
}

#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
status_t engine_t::create_kernels_from_cache_blob(
        const cache_blob_t &cache_blob,
        std::vector<gpu::intel::compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    kernels = std::vector<compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!kernel_names[i] && kernel_names.size() > 1) continue;
        std::string kernel_name(kernel_names[i] ? kernel_names[i] : "");

        const uint8_t *binary_data = nullptr;
        size_t binary_size = 0;
        CHECK(cache_blob.get_binary(&binary_data, &binary_size));

        xpu::binary_t binary(binary_data, binary_data + binary_size);
        CHECK(create_kernel_from_binary(kernels[i], binary, kernel_names[i],
                gpu::intel::compute::program_src_t()));
    }

    return status::success;
}

status_t engine_t::create_kernels(
        std::vector<gpu::intel::compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::compute::kernel_ctx_t &kernel_ctx) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }
    namespace syclex = ::sycl::ext::oneapi::experimental;

    auto device
            = utils::downcast<const impl::xpu::sycl::engine_impl_t *>(impl())
                      ->device();
    VERROR_ENGINE(device.ext_oneapi_can_build(syclex::source_language::opencl),
            status::runtime_error,
            "SYCL implementation does not support OpenCL kernel compiler "
            "extension - make sure that SYCL and OCLOC are correctly "
            "installed");

    const char *source = nullptr;
    for (size_t i = 0; source == nullptr && i < kernel_names.size(); i++)
        source = get_kernel_source(kernel_names[i]);
    VERROR_ENGINE(source, status::runtime_error,
            "No OpenCL source was found for kernel");

    stringstream_t pp_code;
    CHECK(compute::preprocess_headers(pp_code, source, kernel_ctx));
    std::string code_str = pp_code.str();

    std::string build_options = kernel_ctx.options();
    build_options += " " + device_info()->get_cl_ext_options();

    gpu::intel::compute::program_src_t src(code_str);
    if (src) { build_options += " -g -s " + std::string(src.name()); }

    compute::debugdump_processed_source(
            code_str, build_options, device_info()->get_cl_ext_options());

    auto kb_src = syclex::create_kernel_bundle_from_source(
            context(), syclex::source_language::opencl, code_str);
    auto kb_exe = syclex::build(
            kb_src, syclex::properties {syclex::build_options(build_options)});
    *kernels = std::vector<compute::kernel_t>(kernel_names.size());

    if (kernel_ctx.has_custom_headers()
            && gemmstone::microkernel::hasMicrokernels(code_str.c_str())) {
        xpu::binary_t binary;
        CHECK(get_kernel_bundle_binary(this, kb_exe, binary));
        CHECK(compute::fuse_microkernels(
                binary, code_str.c_str(), device_info()->grf_size()));

        std::vector<std::unique_ptr<::sycl::kernel>> sycl_kernels;
        CHECK(compat::make_kernels(sycl_kernels, kernel_names, this, binary));
        for (size_t i = 0; i < kernel_names.size(); ++i) {
            if (!sycl_kernels[i]) continue;
            CHECK(interop_kernel_t::make((*kernels)[i], *sycl_kernels[i], src));
        }
        return status::success;
    }

    for (size_t i = 0; i < kernel_names.size(); ++i) {
        if (!kernel_names[i]) continue;

        CHECK(interop_kernel_t::make((*kernels)[i],
                kb_exe.ext_oneapi_get_kernel(kernel_names[i]), src));
    }

    return status::success;
}
#else
status_t engine_t::convert_to_sycl(
        std::vector<gpu::intel::compute::kernel_t> &kernels,
        const std::vector<gpu::intel::compute::kernel_t> &ocl_kernels,
        const std::vector<const char *> &kernel_names,
        gpu::intel::ocl::engine_t *ocl_engine) const {
    kernels = std::vector<gpu::intel::compute::kernel_t>(kernel_names.size());
    for (size_t i = 0; i < ocl_kernels.size(); ++i) {
        if (!ocl_kernels[i]) continue;
        auto *k = utils::downcast<gpu::intel::ocl::kernel_t *>(
                ocl_kernels[i].impl());
        xpu::binary_t binary;
        CHECK(k->get_binary(ocl_engine, binary));
        CHECK(create_kernel_from_binary(
                kernels[i], binary, kernel_names[i], k->src()));
    }
    return status::success;
}

status_t engine_t::create_kernels_from_cache_blob(
        const cache_blob_t &cache_blob,
        std::vector<gpu::intel::compute::kernel_t> &kernels,
        const std::vector<const char *> &kernel_names) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t> ocl_engine;
    auto status = gpu::intel::sycl::create_ocl_engine(&ocl_engine, this);
    if (status != status::success) return status;

    std::vector<gpu::intel::compute::kernel_t> ocl_kernels;
    CHECK(ocl_engine->create_kernels_from_cache_blob(
            cache_blob, ocl_kernels, kernel_names));
    CHECK(convert_to_sycl(
            kernels, ocl_kernels, kernel_names, ocl_engine.get()));
    return status::success;
}

status_t engine_t::convert_to_sycl(
        std::vector<gpu::intel::compute::kernel_t> &kernels, cl_program program,
        const gpu::intel::compute::program_src_t &program_src,
        const std::vector<const char *> &kernel_names,
        gpu::intel::ocl::engine_t *ocl_engine) const {
    kernels = std::vector<gpu::intel::compute::kernel_t>(kernel_names.size());
    xpu::binary_t binary;
    CHECK(ocl::get_ocl_program_binary(program, ocl_engine->device(), binary));

    std::vector<std::unique_ptr<::sycl::kernel>> sycl_kernels;
    CHECK(gpu::intel::sycl::compat::make_kernels(
            sycl_kernels, kernel_names, this, binary));

    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (!sycl_kernels[i]) continue;
        CHECK(interop_kernel_t::make(
                kernels[i], *sycl_kernels[i], program_src));
    }
    return status::success;
}

status_t engine_t::create_kernels(
        std::vector<gpu::intel::compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::compute::kernel_ctx_t &kernel_ctx) const {
    if (kind() != engine_kind::gpu) {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t> ocl_engine;
    CHECK(gpu::intel::sycl::create_ocl_engine(&ocl_engine, this));

    xpu::ocl::wrapper_t<cl_program> ocl_program;
    gpu::intel::compute::program_src_t src;
    CHECK(ocl_engine->create_program(
            ocl_program, src, kernel_names, kernel_ctx));
    CHECK(convert_to_sycl(
            *kernels, ocl_program, src, kernel_names, ocl_engine.get()));
    return status::success;
}
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

status_t engine_t::init_device_info() {
    device_info_ = std::make_shared<gpu::intel::sycl::device_info_t>();
    CHECK(device_info_->init(this));
    return status::success;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
