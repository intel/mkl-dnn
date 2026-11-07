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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_TUNED_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_TUNED_HPP

namespace rvjit {
namespace linalg {
namespace matmul {

/// What an implementation decides about a plan, whatever its ISA
///
/// @details The dispatch widths are in granules, whose size along each
///     dimension is the grain. The grains decide the shape of the
///     multiply-accumulate: a vector ISA has a wide M grain and unit N and K
///     grains, whereas another may take several elements along any
struct tuned_t {
    /// The dispatch widths: M granules, N granules and the main K unroll
    int mb = 1, nb = 1, kb = 1;

    /// The K unroll under an N-tail tile
    int ktb = 1;

    /// Elements one granule covers along M, N and K
    int m_grain = 1, n_grain = 1, k_grain = 1;
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#endif
