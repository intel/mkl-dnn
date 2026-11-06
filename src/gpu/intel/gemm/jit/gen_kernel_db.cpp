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

#include "gpu/intel/gemm/jit/gen_kernel_db.hpp"

namespace gemmstone {
#define _CATALOG_ gemm_catalog
#include "selector/db/kernel.db"
#undef _CATALOG_
#define _CATALOG_ gemm_lut_catalog
#include "selector/db/lut.db"
#undef _CATALOG_
} // namespace gemmstone

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

gemmstone::kcatalog::Catalog catalog() {
    return gemmstone::gemm_catalog;
}

gemmstone::kcatalog::Catalog lut_catalog() {
    return gemmstone::gemm_lut_catalog;
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
