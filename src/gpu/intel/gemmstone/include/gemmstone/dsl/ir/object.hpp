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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_IR_OBJECT_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_IR_OBJECT_HPP

#include <cstdint>
#include <string>

#include "gemmstone/dsl/type.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class object_t;
class expr_impl_t;
class stmt_impl_t;
class ir_mutator_t;
class ir_visitor_t;

namespace object {
// Base class for all IR objects. Implemented as an intrusive pointer, with
// the reference counter stored inside the object.
class impl_t : public stringify_t<impl_t> {
public:
    impl_t(const impl_t &) = delete;
    impl_t &operator=(const impl_t &) = delete;

    virtual ~impl_t() = default;

    // Provides equality semantics.
    virtual bool is_equal(const impl_t &obj) const = 0;

    // Memoized, the hash is computed at most once per object: IR objects are
    // immutable once created by design.
    size_t get_hash() const {
        if (!hash_) {
            size_t h = compute_hash();
            hash_ = h ? h : 1; // 0 marks "not computed yet"
        }
        return hash_;
    }

    bool is_expr() const { return info().is_expr; }
    bool is_stmt() const { return info().is_stmt; }

    // Downcasts the object to the IR type, returns a reference. The IR type
    // must match the real IR type.
    // N.B.: this can potentially be dangerous if applied to non-const objects,
    //       since assigning a different value to the source object might make
    //       the reference dangling due to the destruction of the former object;
    //       please only call this method on non-const objects if absolutely
    //       necessary, and please don't add a non-const variant of the method!
    template <typename T>
    const T &as() const {
        gemm_assert(is<T>());
        return *as_ptr<T>(); // fails on incorrect casts even in Release
    }

    // Downcasts the object to the IR type, returns a pointer. If the IR type
    // doesn't match the real IR type, returns nullptr.
    // N.B.: this can potentially be dangerous if applied to non-const objects,
    //       since assigning a different value to the source object might make
    //       the reference dangling due to the destruction of the former object;
    //       please only call this method on non-const objects if absolutely
    //       necessary, and please don't add a non-const variant of the method!
    template <typename T>
    const T *as_ptr() const {
        return (is<T>()) ? (const T *)this : nullptr;
    }

    // Returns true if T matches the real IR type.
    template <typename T>
    bool is() const {
        return info() == T::get_info();
    }

    virtual std::string str() const;

    virtual object_t _mutate(ir_mutator_t &mutator) const;
    virtual void _visit(ir_visitor_t &visitor) const;

    virtual size_t compute_hash() const = 0;

protected:
    friend class gemmstone::dsl::ir::object_t;
    template <typename T>
    friend struct info_t;

    void retain() { ref_count_.increment(); }
    void release() {
        if (ref_count_.decrement() == 0) { delete this; }
    }

    struct info_t {
        constexpr info_t(const void *uid, bool is_expr, bool is_stmt)
            : uid(uid), is_expr(is_expr), is_stmt(is_stmt) {}

        const void *uid = nullptr;
        bool is_expr = false;
        bool is_stmt = false;

        bool operator==(const info_t &other) const { return uid == other.uid; }
        bool operator!=(const info_t &other) const {
            return !operator==(other);
        }
    };

    // std::type_info objects may differ between TUs. Deduplicate them in a
    // single TU to ensure uniqueness. This is used to improve performance of
    // object equality checks, as std::type_info comparisons may be relatively
    // slow.
    static const void *get_uid(const std::type_info &);

    impl_t(info_t info) : info_(info) {};

private:
    // Reference counter for IR objects.
    class ref_count_t {
    public:
        ref_count_t() : value_(0) {}
        ref_count_t(const ref_count_t &) = delete;
        ref_count_t &operator=(const ref_count_t &) = delete;
        ~ref_count_t() = default;

        uint32_t increment() { return ++value_; }
        uint32_t decrement() { return --value_; }

    private:
        uint32_t value_;
    };

    // Type information.
    const info_t &info() const { return info_; }

    ref_count_t ref_count_;
    info_t info_;
    mutable size_t hash_ = 0;
};

template <typename T>
struct info_t {
    using self_type = T;

protected:
    friend class impl_t;
    static impl_t::info_t get_info() {
        static const void *uid = impl_t::get_uid(typeid(T));
        return impl_t::info_t(static_cast<const void *>(uid),
                is_expr_t<T>::value, is_stmt_t<T>::value);
    }

private:
    template <typename U, typename = void>
    struct is_expr_t {
        static const bool value = false;
    };

    template <typename U>
    struct is_expr_t<U,
            typename std::enable_if<
                    std::is_base_of<expr_impl_t, U>::value>::type> {
        static const bool value = true;
    };

    template <typename U, typename = void>
    struct is_stmt_t {
        static const bool value = false;
    };

    template <typename U>
    struct is_stmt_t<U,
            typename std::enable_if<
                    std::is_base_of<stmt_impl_t, U>::value>::type> {
        static const bool value = true;
    };
};
} // namespace object

// Base wrapper for IR objects.
class object_t : public stringify_t<object_t> {
public:
    using impl_t = object::impl_t;
    object_t(impl_t *impl = nullptr) : impl_(impl) { retain(impl_); }
    object_t(const impl_t &impl) : object_t(const_cast<impl_t *>(&impl)) {}
    object_t(const impl_t *impl) : object_t(const_cast<impl_t *>(impl)) {}
    object_t(const object_t &obj) : object_t(obj.impl()) {}
    object_t(object_t &&obj) : impl_(obj.impl_) { obj.impl_ = nullptr; }

    ~object_t() { release(impl_); }

    object_t &operator=(const object_t &other) {
        if (&other == this) return *this;
        auto *other_impl = other.impl();
        retain(other_impl);
        release(impl_);
        impl_ = other_impl;
        return *this;
    }

    object_t &operator=(object_t &&other) {
        std::swap(impl_, other.impl_);
        return *this;
    }

    impl_t *impl() const { return impl_; }

    bool is_empty() const { return !impl_; }

    explicit operator bool() const { return !is_empty(); }

    template <typename T>
    const T &as() const {
        gemm_assert(impl_);
        return impl_->as<T>();
    }

    template <typename T>
    const T *as_ptr() const {
        if (!impl_) return nullptr;
        return impl_->as_ptr<T>();
    }

    template <typename T>
    bool is() const {
        if (is_empty()) return false;
        return impl_->is<T>();
    }

    // Comparison with identity semantics.
    bool is_same(const object_t &other) const { return impl_ == other.impl(); }

    // Comparison with equality semantics.
    bool is_equal(const object_t &other) const {
        if (is_empty() || other.is_empty())
            return is_empty() == other.is_empty();
        if (impl_ == other.impl()) return true;

        return impl_->is_equal(*other.impl());
    }

    size_t get_hash() const {
        if (is_empty()) return 0;
        return impl()->get_hash();
    }

    bool is_expr() const { return impl_ && impl_->is_expr(); }
    bool is_stmt() const { return impl_ && impl_->is_stmt(); }

    std::string str() const {
        if (is_empty()) return "(nil)";
        return impl()->str();
    }

private:
    static void retain(impl_t *impl) {
        if (impl) impl->retain();
    }

    static void release(impl_t *impl) {
        if (impl) impl->release();
    }

    impl_t *impl_;
};

// Wrapper for IR expression objects.
class expr_t : public object_t {
public:
    using object_t::object_t;

    expr_t() = default;
    expr_t(const object_t &obj) : object_t(obj) {}
    expr_t(object_t &&obj) : object_t(obj) {}
    expr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    expr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    explicit expr_t(bool v);
    expr_t(float v);
    expr_t(double v);
    expr_t(int16_t v);
    expr_t(int32_t v);
    expr_t(int64_t v);
    expr_t(uint16_t v);
    expr_t(uint32_t v);
    expr_t(uint64_t v);

    const type_t &type() const;

#define DECLARE_BINARY_ASSIGN_OPERATOR(op) \
    expr_t &operator op##=(const expr_t & rhs);

    DECLARE_BINARY_ASSIGN_OPERATOR(+)
    DECLARE_BINARY_ASSIGN_OPERATOR(-)
    DECLARE_BINARY_ASSIGN_OPERATOR(*)
    DECLARE_BINARY_ASSIGN_OPERATOR(/)
    DECLARE_BINARY_ASSIGN_OPERATOR(%)
    DECLARE_BINARY_ASSIGN_OPERATOR(&)
    DECLARE_BINARY_ASSIGN_OPERATOR(^)

#undef DECLARE_BINARY_ASSIGN_OPERATOR

    // Depending on the base expression type:
    // - If the base expression is of a pointer type: returns a pointer shifted
    //   by `off` elements relative to the base pointer.
    // - If the base expression is a shuffle expression: returns the `off`-th
    //   component of this shuffle. `off` msut be a constant.
    expr_t operator[](const expr_t &off) const;

    // Returns a pointer type expression pointing to this variable. The base
    // expression must be of var_t or ref_t type.
    expr_t ptr(const expr_t &off = expr_t(0)) const;

    template <typename T>
    bool is() const {
        return object_t::is<T>();
    }

    bool is(int value) const;
};

bool is_const(const expr_t &e);
bool to_bool(const expr_t &e);
expr_t operator-(const expr_t &a);

#define DECLARE_BINARY_OPERATOR(op) \
    expr_t operator op(const expr_t &a, const expr_t &b)

DECLARE_BINARY_OPERATOR(+);
DECLARE_BINARY_OPERATOR(-);
DECLARE_BINARY_OPERATOR(*);
DECLARE_BINARY_OPERATOR(/);
DECLARE_BINARY_OPERATOR(%);
DECLARE_BINARY_OPERATOR(<<);
DECLARE_BINARY_OPERATOR(>>);

DECLARE_BINARY_OPERATOR(==);
DECLARE_BINARY_OPERATOR(!=);
DECLARE_BINARY_OPERATOR(>);
DECLARE_BINARY_OPERATOR(>=);
DECLARE_BINARY_OPERATOR(<);
DECLARE_BINARY_OPERATOR(<=);

DECLARE_BINARY_OPERATOR(&);
DECLARE_BINARY_OPERATOR(|);
DECLARE_BINARY_OPERATOR(^);

#undef DECLARE_BINARY_OPERATOR

// Wrapper for IR statement objects.
class stmt_t : public object_t {
public:
    using object_t::object_t;

    stmt_t() = default;
    stmt_t(const object_t &obj) : object_t(obj) {}
    stmt_t(object_t &&obj) : object_t(std::move(obj)) {}
    stmt_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    stmt_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    stmt_t append(const stmt_t &s) const;
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
