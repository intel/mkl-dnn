/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_IMPLEMENTATION_RVV_LINALG_LINALG_HPP
#define RVJIT_IMPLEMENTATION_RVV_LINALG_LINALG_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/linalg/matmul.hpp"
#include "rvjit/interface/linalg/linalg.hpp"

namespace rvjit {
namespace rvv {
namespace linalg {

/// RVV specialization of `linalg::linalg_t`
struct linalg_t : public rvjit::linalg::linalg_t {
    static constexpr const char *NAME = "rvjit::rvv::linalg";

    /// Creates the RVV linear-algebra components
    ///
    /// @param pm The programming model the components are built around
    explicit linalg_t(rvjit::rvv::programming_model_t &pm)
        : impl_(pm), matmul_(impl_) {}

    rvjit::linalg::matmul::matmul_t &matmul() override { return matmul_; }

    /// The RVV implementation, for methods (e.g. `tuning()`) outside the
    /// `rvjit::linalg::matmul::matmul_t` contract
    rvv::linalg::matmul::impl_t &rvv_impl() { return impl_; }

private:
    rvv::linalg::matmul::impl_t impl_;
    rvjit::linalg::matmul::matmul_t matmul_;
};

} // namespace linalg
} // namespace rvv
} // namespace rvjit

#endif
