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

#ifndef RVJIT_INTERFACE_STREAM_STREAM_IMPL_HPP
#define RVJIT_INTERFACE_STREAM_STREAM_IMPL_HPP

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {

struct ref_t;

/// Describes and emits one instruction stream over an ND-dimensional space
///
/// @details Emission happens in ascending nested order
///
/// @note Streams change in place through `resize`
///
/// @pre Iterations are only correct when driven in ascending nested order
struct stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream";

    virtual ~stream_impl_t() = default;

    /// Number of axes of the stream
    virtual int ndims() const {
        LOG_FAIL(__func__, "unimplemented");
        return 0;
    }

    /// Size of the `d`-th axis of the stream
    ///
    /// @param d The axis
    virtual int size(int d) const {
        (void)d;
        LOG_FAIL(__func__, "unimplemented");
        return 0;
    }

    /// Whether or not the stream is ready to be emitted
    virtual bool is_valid() const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Narrows or widens the axes of the stream to `shape`
    ///
    /// @param shape The size of each axis
    ///
    /// @pre The resources backing the stream cover the `shape`
    virtual bool resize(const index_t &shape) {
        (void)shape;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// The register at position `idx`
    ///
    /// @param idx The position
    ///
    /// @note Streams that do not resolve to registers do not override this
    virtual name_t get(const index_t &idx) const {
        (void)idx;
        LOG_FAIL(__func__, "unimplemented");
        return name_t();
    }

    /// The registers backing the stream, if any
    virtual const buffer::buffer_t &buffer() const {
        LOG_FAIL(__func__, "unimplemented");
        static const buffer::buffer_t none;
        return none;
    }

    /// Emits the code for position `idx`
    ///
    /// @param idx The position
    ///
    /// @note Emits nothing if `idx` lies outside the shape of the stream
    void iteration(const index_t &idx) {
        if (in_shape_(idx)) produce_(idx);
    }

    /// Emits the initialization of position `idx`
    ///
    /// @param idx The position
    ///
    /// @note Emits nothing if `idx` lies outside the shape of the stream
    void init_iteration(const index_t &idx) {
        if (in_shape_(idx)) init_(idx);
    }

    /// A reference to this stream usable by the combinators
    ///
    /// @param init Whether the reference drives the init phase
    ref_t ref(bool init = false);

    /// Emits the code for every position iteratively
    ///
    /// @note A stream may emit its positions in another order than ascending
    ///     when its own code is independent of it, e.g. to stagger a producer
    ///     and a consumer instruction
    virtual void emit() {
        index_t idx {};
        sweep_(false, 0, ndims(), idx);
    }

    /// Emits the initialization of every position
    void emit_init() {
        index_t idx {};
        sweep_(true, 0, ndims(), idx);
    }

protected:
    /// Emits the production of in-shape position `idx`
    virtual void produce_(const index_t &idx) {
        (void)idx;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Emits the initialization of in-shape position `idx`
    virtual void init_(const index_t &idx) {
        (void)idx;
        LOG_FAIL(__func__, "unimplemented");
    }

private:
    bool in_shape_(const index_t &idx) const {
        const int nd = ndims();
        for (int d = 0; d < nd; ++d)
            if (idx[d] < 0 || idx[d] >= size(d)) return false;
        return true;
    }

    void sweep_(bool init, int d, int nd, index_t &idx) {
        if (d == nd) {
            init ? init_iteration(idx) : iteration(idx);
            return;
        }
        for (idx[d] = 0; idx[d] < size(d); ++idx[d])
            sweep_(init, d + 1, nd, idx);
        idx[d] = 0;
    }
};

/// A stream of any `stream_impl_t`, in its production or init phase
///
/// @details Lets combinators drive streams owned by different components
///
/// @note Non-owning; the owner must outlive every reference to it
struct ref_t {
    ref_t() = default;

    /// Creates a reference to `stream`
    ///
    /// @param stream The referenced stream
    /// @param init Whether the reference drives the init phase
    ref_t(stream_impl_t &stream, bool init = false)
        : stream_(&stream), init_(init) {}

    /// This stream in its init phase
    ref_t init() const { return ref_t(*stream_, true); }

    /// This stream in its production phase
    ref_t produce() const { return ref_t(*stream_, false); }

    /// Whether the referenced stream is ready to be emitted
    bool is_valid() const { return stream_ && stream_->is_valid(); }

    /// Number of axes of the referenced stream
    int ndims() const { return stream_->ndims(); }

    /// Size of the `d`-th axis of the referenced stream
    ///
    /// @param d The axis
    int size(int d) const { return stream_->size(d); }

    /// Narrows or widens the axes of the referenced stream to `shape`
    ///
    /// @param shape The size of each axis
    bool resize(const index_t &shape) { return stream_->resize(shape); }

    /// The register of the referenced stream at position `idx`
    ///
    /// @param idx The position
    name_t get(const index_t &idx) const { return stream_->get(idx); }

    /// The registers backing the referenced stream
    const buffer::buffer_t &buffer() const { return stream_->buffer(); }

    /// Emits position `idx` in this reference's phase
    ///
    /// @param idx The position
    void iteration(const index_t &idx) const {
        init_ ? stream_->init_iteration(idx) : stream_->iteration(idx);
    }

    /// Emits every position in this reference's phase
    void emit() const { init_ ? stream_->emit_init() : stream_->emit(); }

private:
    stream_impl_t *stream_ = nullptr;
    bool init_ = false;
};

inline ref_t stream_impl_t::ref(bool init) {
    return ref_t(*this, init);
}

} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
