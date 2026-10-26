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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_MEMORY_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_MEMORY_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace memory {

/// Memory operation direction
enum class dir_t { load, store };

/// Vector memory addressing mode
enum class vec_addr_t {
    unit, // Consecutive elements from the base
    stride, // Elements a register-held byte stride apart
    index, // Elements at the byte offsets a vector register holds
};

/// Generic memory operation description
struct op_t : public typed_t {
    using operand_t = rvjit::operand_t;

    operand_t data;
    pointer_t addr;
    dir_t dir = dir_t::load;
    name_t args = no_name;
    vec_addr_t vector_mode = vec_addr_t::unit;
    predicate_t predicate = predicate_t::unmasked;

    op_t() = default;

    /// A load or store of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dir The direction of the operation
    /// @param dt The data type
    op_t(rvjit::form_t form, dir_t dir, type_t dt) : dir(dir) {
        data.form = form;
        data.set_type(dt);
    }

    /// A load of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    static op_t load(rvjit::form_t form, type_t dt) {
        return op_t(form, dir_t::load, dt);
    }

    /// A store of `dt` data in the given register form
    ///
    /// @param form The storage category of the data
    /// @param dt The data type
    static op_t store(rvjit::form_t form, type_t dt) {
        return op_t(form, dir_t::store, dt);
    }

    bool is_scalar() const override { return data.is_scalar(); }
    bool is_vector() const override { return data.is_vector(); }
    bool is_int() const override { return data.is_int(); }
    bool is_float() const override { return data.is_float(); }
    bool is_signed() const override { return data.is_signed(); }

    /// Sets the data to the scalar form
    op_t &scalar() {
        data.form = rvjit::form_t::scalar;
        vector_mode = vec_addr_t::unit;
        return *this;
    }

    /// Sets the data to the vector form with unit-stride addressing
    ///
    /// @param vm The vector mask policy
    op_t &vector(predicate_t vm = predicate_t::unmasked) {
        data.form = rvjit::form_t::vector;
        vector_mode = vec_addr_t::unit;
        predicate = vm;
        return *this;
    }

    /// Sets the vector addressing to a register-held byte stride
    ///
    /// @param stride_reg The register holding the stride
    op_t &strided(const reg_t &stride_reg) {
        args = reg_name(stride_reg);
        vector_mode = vec_addr_t::stride;
        return *this;
    }

    /// Sets the vector addressing to the byte offsets of a vector register
    ///
    /// @param index_reg The vector register holding the offsets
    op_t &indexed(const vreg_t &index_reg) {
        args = reg_name(index_reg);
        vector_mode = vec_addr_t::index;
        return *this;
    }

    /// Sets the direction to load
    op_t &load() {
        dir = dir_t::load;
        return *this;
    }

    /// Sets the direction to store
    op_t &store() {
        dir = dir_t::store;
        return *this;
    }

    /// Sets the data type
    ///
    /// @param dt The data type
    op_t &as_type(type_t dt) {
        data.set_type(dt);
        return *this;
    }
};

} // namespace memory
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Memory expressions
struct memory_t {
    static constexpr const char *NAME = "rvjit::operation::memory";

    virtual ~memory_t() = default;

    /// Whether `load` can emit `op`, ignoring the operand's register name
    ///
    /// @param op The operation checked
    virtual bool supports_load(const memory::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `store` can emit `op`, ignoring the operand's register name
    ///
    /// @param op The operation checked
    virtual bool supports_store(const memory::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Loads data from memory and stores into a buffer
    ///
    /// @param op The operation, loading `addr` into `data`
    ///
    /// @note Returns false when the expression fails to emit code
    ///
    /// @pre Caller must supply operand combinations supported by the ISA
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool load(const memory::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Stores data from a buffer into memory
    ///
    /// @param op The operation, storing `data` into `addr`
    ///
    /// @note Returns false when the expression fails to emit code
    ///
    /// @pre Caller must supply operand combinations supported by the ISA
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool store(const memory::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// Assembles a `memory_t` from expression implementations
///
/// @details The expression must be constructible from `emitter_t`
template <typename LoadOp, typename StoreOp>
struct memory_compose_t : public memory_t {
    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit memory_compose_t(emitter_t e) : load_(e), store_(e) {}

    bool supports_load(const memory::op_t &op) const override {
        return load_.supports(op);
    }
    bool supports_store(const memory::op_t &op) const override {
        return store_.supports(op);
    }
    bool load(const memory::op_t &op) const override { return load_(op); }
    bool store(const memory::op_t &op) const override { return store_(op); }

protected:
    LoadOp load_;
    StoreOp store_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
