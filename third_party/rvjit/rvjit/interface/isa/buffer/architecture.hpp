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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_ARCHITECTURE_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_ARCHITECTURE_HPP

#include "rvjit/interface/isa/buffer/buffer.hpp"

namespace rvjit {
namespace buffer {

/// The registers of one class a pool draws from, and the order it draws them
struct buffer_file_t {
    buffer_file_t() = default;

    /// Creates a register file
    ///
    /// @param regs The registers
    /// @param order The allocation order of the registers
    explicit buffer_file_t(buffer_t regs, const order_t &order = {})
        : file_(regs), order_(order) {}

    /// Removes `excl` from the registers
    ///
    /// @param excl The registers removed
    buffer_file_t &without(const buffer_t &excl = {}) {
        file_ = file_.filter(excl);
        return *this;
    }

    /// The registers
    const buffer_t &file() const { return file_; }

    /// Allocation order of the registers, unlisted ones following ascending
    const order_t &order() const { return order_; }

private:
    buffer_t file_;
    order_t order_;
};

/// The integer, float and vector register file sources for a `buffer_pool_t`
struct architecture_t {
    buffer_file_t integer;
    buffer_file_t floating;
    buffer_file_t vector;
};

} // namespace buffer
} // namespace rvjit

#endif
