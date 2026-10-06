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

#include <numeric>

#include "examples/utils.hpp"
#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/dsl/kernel.hpp"
#include "gemmstone/dsl/runtime.hpp"
#include "gemmstone/runtime.hpp"

using namespace gemmstone;
using namespace gemmstone::examples;

dsl::kernel_t generate_kernel(dsl::kernel::options_t options) {
    using namespace dsl;

    // Define kernel interface which is composed of the kernel name and the list
    // of input arguments.
    kernel::iface_t iface("vector_scale");
    expr_t buffer = iface.register_global("buffer", f32);
    expr_t size = iface.register_arg("size", u32);
    expr_t alpha = iface.register_arg("alpha", f32);

    // Start kernel generation. The DSL relies on an thread local context
    // structure, so we cannot generate multiple kernels simultaneously.
    declare_kernel(iface, options);

    auto offset = let("global_id", global_id(0, u32));
    auto is_inbounds = offset < size;
    auto b = def("b", load(buffer, offset, is_inbounds));
    b *= alpha;
    store(buffer, offset, b, is_inbounds);

    // End kernel generation and obtain IR representation that can be lowered to
    // an runtime object.
    kernel_t kernel = end_kernel();
    return kernel;
}

int main(int argc, char *argv[]) {
    const char *target = argc > 1 ? argv[1] : "l0";

    // Generate kernel build options. For simplicity, this example only depends
    // on the target hardware.
    gemmstone::dsl::kernel::options_t options(get_hardware(target));

    // Generate IR kernel representation
    gemmstone::dsl::kernel_t kernel = generate_kernel(options);

    handle_hardware_target(target, kernel);

    // Create kernel arguments on host
    uint32_t N = 16;
    float alpha = 1.5f;
    auto input = Array<float>::arange(1.0f, N + 1.0f);
    auto buffer = input;
    uint32_t local = uint32_t(options.simd());
    uint32_t global = (N / local + ((N % local) != 0)) * local;

    // Dispatch to a helper function to create a runtime kernel and execute it
    execute(target, kernel, NDRange(global, local), buffer, N, alpha);

    // Print results
    input.print("Input");
    std::cout << "Alpha:  " << alpha << std::endl;
    buffer.print("Result");

    assert_close(buffer, alpha * input);
}
