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

#ifndef RVJIT_INTERFACE_ISA_BUFFERED_HPP
#define RVJIT_INTERFACE_ISA_BUFFERED_HPP

#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/resource/materializable.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// A value materialized into integer buffers from an optional origin
///
/// @note Supplying or withdrawing buffers is the one mutation after creation
struct buffered_t : public materializable_t {
    using buffer_t = buffer::buffer_t;

    buffer_t supplied() const override { return buffer_; }

    /// Number of buffers still pending for this object
    virtual int buffer_needed() const { return 0; }

    /// Whether any buffer resource is still pending
    bool needs_buffer() const { return buffer_needed() > 0; }

    /// Whether any buffer resource is supplied
    bool has_buffer() const { return !buffer_.is_empty(); }

    demand_t demand() const override {
        return demand_t::of(resource_t::integers, buffer_needed());
    }

    bool is_valid() const override { return !needs_buffer(); }

    /// The origin the value materializes from
    const source_t &source() const { return source_; }

    /// Replaces the object's buffer with a single register
    ///
    /// @param r The register supplied
    ///
    /// @note A zero register withdraws the buffer
    void set_buffer(const reg_t &r) {
        buffer_ = is_not_zero(r) ? buffer_t({r}) : buffer_t();
    }

    /// Supplies additional buffer resources
    ///
    /// @param b The registers supplied
    ///
    /// @pre `b.size()` must not exceed `buffer_needed()`
    void set_buffer(const buffer_t &b) { buffer_ = buffer_ | b; }

    /// Withdraws every buffer supplied to this object
    void clear_buffer() { buffer_ = {}; }

protected:
    /// The first buffer register, or zero when none is held
    reg_t reg_() const {
        return buffer_.is_empty() ? reg_t {} : reg_t(buffer_.id(0));
    }

    buffer_t buffer_; // Supplied resources
    source_t source_ = source_t::none(); // Optional origin to materialize from
};

} // namespace rvjit

#endif
