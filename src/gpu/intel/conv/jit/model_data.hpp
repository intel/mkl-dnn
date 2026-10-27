/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_MODEL_DATA_HPP
#define GPU_INTEL_CONV_JIT_MODEL_DATA_HPP

#include <cstddef>
#include <cstdint>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

const uint64_t *get_model_xehpc_dw_data(size_t &size);
const uint64_t *get_model_xehpc_common_data(size_t &size);
const uint64_t *get_model_xehpg_dw_data(size_t &size);
const uint64_t *get_model_xehpg_common_data(size_t &size);

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
