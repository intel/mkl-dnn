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

#ifndef CPU_X64_UTILS_JIT_REGOPS_HPP
#define CPU_X64_UTILS_JIT_REGOPS_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace regops {

/**
 * Horizontally sums packed fp32 values in a 256-bit YMM register,
 * using workspace as temporary storage.
 * @param code JIT code generator instance.
 * @param src Source YMM register.
 * @param workspace Temporary YMM register.
 */
void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace);

/**
 * Horizontally sums packed fp32 values in a 512-bit ZMM register,
 * using workspace as temporary storage.
 * @param code JIT code generator instance.
 * @param src Source ZMM register.
 * @param workspace Temporary ZMM register.
 */
void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace);

/**
 * Horizontally reduces packed fp32 values in a 128-bit XMM register to their
 * maximum, placing the result in every lane. @param workspace is temporary.
 */
void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Xmm src, Xbyak::Xmm workspace);

/**
 * Horizontally reduces packed fp32 values in a 256-bit YMM register to their
 * maximum, using workspace as temporary storage.
 */
void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace);

/**
 * Horizontally reduces packed fp32 values in a 512-bit ZMM register to their
 * maximum, using workspace as temporary storage.
 */
void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace);

} // namespace regops
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
