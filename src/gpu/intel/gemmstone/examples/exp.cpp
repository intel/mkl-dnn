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

#include "examples/utils.hpp"
#include "tools/types/fp16.hpp"
#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/dsl/kernel.hpp"

using namespace gemmstone;
using namespace gemmstone::examples;

dsl::kernel_t generate_kernel(dsl::type_t input_type, dsl::type_t output_type,
        dsl::kernel::options_t options) {
    using namespace dsl;

    // Define kernel interface which is composed of the kernel name and the list
    // of input arguments.
    kernel::iface_t iface("block_texp");
    iface.register_global("input", input_type);
    iface.register_global("output", output_type);
    iface.register_arg("elems", s32);

    // Start kernel generation. The DSL relies on an thread local context
    // structure, so we cannot generate multiple kernels simultaneously.
    declare_kernel(iface, options);

    expr_t input = iface.find_arg("input");
    expr_t output = iface.find_arg("output");
    expr_t elems = iface.find_arg("elems");

    // col index (x-dimension) and row index (y-dimension)
    auto offset = let("global", group_id(0) * local_size(0) + local_id(0));

    // Bounds check: ensure both row and col are within the matrix dimensions
    // (pointer arithmetic already accounts for element size)
    auto is_inbounds = offset < elems;

    auto b = def("b", load(input, offset, is_inbounds));
    auto output_reg = exp(b, output_type);
    store(output, offset, output_reg, is_inbounds);

    // End kernel generation and obtain IR representation that can be lowered to
    // an runtime object.
    kernel_t kernel = end_kernel();
    return kernel;
}

template <typename InputHostType, typename OutputHostType>
void run_texp(const char *runtime, gemmstone::dsl::type_t input_dtype,
        gemmstone::dsl::type_t output_dtype, uint32_t N, uint32_t M) {
    // Generate kernel build options. For simplicity, this example only depends
    // on the target hardware.
    gemmstone::dsl::kernel::options_t options(get_hardware(runtime));

    // Generate IR kernel representation
    gemmstone::dsl::kernel_t kernel
            = generate_kernel(input_dtype, output_dtype, options);

    handle_hardware_target("l0", kernel);

    // row major 2d matrix stored in 1d buffer
    auto input = Array<InputHostType>(N * M); // strides = {M, 1}
    // fill data
    for (uint32_t i = 0; i < N * M; i++) {
        input[i] = InputHostType(i / 32.0);
    }

    auto output = Array<OutputHostType>(N * M);
    uint32_t local = M * N;
    uint32_t global = ((M * N) / local + (((M * N) % local) != 0)) * local;

    auto range = NDRange(global, local);

    // Dispatch to a helper function to create a runtime kernel and execute it
    uint32_t elems = M * N;
    execute(runtime, kernel, range, input, output, elems);

    // Print results
    std::cout << "Input data type:  " << input_dtype << std::endl;
    std::cout << "Output data type: " << output_dtype << std::endl;
    input.print("Calculating: exp");
    output.print("Result");
}

int main(int argc, char *argv[]) {
    const char *runtime = argc > 1 ? argv[1] : "l0";
    uint32_t N = 4;
    uint32_t M = 32;

    using namespace gemmstone::dsl;

    // --- Input: f32 ---
    run_texp<float, float>(runtime, f32, f32, N, M);
    // run_texp<float,    half>    (runtime, f32,  f16,  N, M); // Not Supported yet.

    // --- Input: f16 ---
    run_texp<half, half>(runtime, f16, f16, N, M);
}
