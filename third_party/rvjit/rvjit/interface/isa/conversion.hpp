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

#ifndef RVJIT_INTERFACE_ISA_CONVERSION_HPP
#define RVJIT_INTERFACE_ISA_CONVERSION_HPP

#include <functional>

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/parameter.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace conversion {

/// The arithmetic expression a conversion step emits
enum class kind_t {
    extend, // An integer 2x, 4x or 8x as wide, by sign or zeros
    widen, // A type 2x as wide
    convert, // Between an integer and a float of one width
    narrow, // A type half as wide, saturating integers
};

/// Switches the configuration to a type's element width, keeping the vector
/// length, for the steps run at that width (`step_t::config()`); false when
/// no configuration fits the width
using configure_t = std::function<bool(type_t)>;

/// One step of a conversion, from a type into another
struct step_t {
    kind_t kind = kind_t::convert;
    type_t from = type_t::none;
    type_t to = type_t::none;

    /// The type whose element width configures the step on vector forms
    ///
    /// @note A widening reads at its source's width; every other step runs at
    ///     its destination's
    type_t config() const { return kind == kind_t::widen ? from : to; }
};

} // namespace conversion

/// The steps converting a value between two types
///
/// @details Pure data, planned against an arithmetic dialect. A widening
///     conversion changes width at most once, in its first step, and then
///     converts in place; a narrowing one converts in place first and then
///     halves its width in place, step by step. So a widening reads the
///     narrow value from the highest part of the wide register group and a
///     narrowing leaves it in the lowest part
///
/// @note An empty conversion converts nothing: the types match
struct conversion_t {
    static constexpr const char *NAME = "rvjit::conversion";
    static constexpr int CAPACITY = 4;

    conversion_t() = default;

    /// Plans the steps converting `from` into `to`
    ///
    /// @param from The type converted
    /// @param to The type produced
    /// @param form The storage category of the operands
    /// @param arith The dialect every step must be supported by
    ///
    /// @note Returns false, leaving the conversion empty, when no chain of
    ///     supported steps converts the pair
    bool plan(type_t from, type_t to, form_t form,
            const operation::arithmetic_t &arith) {
        clear();
        form_ = form;
        if (from == to) return true;
        // Two types of one width and family, e.g. s8 and u8, need a step too
        const bool ok = plan_(from, to) && n_ > 0 && supported_(arith);
        if (!ok) clear();
        CHECK(ok, "no conversion");
        return true;
    }

    /// Plans the steps loading a value: from memory into the registers
    ///
    /// @param t The value's types
    /// @param form The storage category of the operands
    /// @param arith The dialect every step must be supported by
    bool plan_load(const type_parameter_t &t, form_t form,
            const operation::arithmetic_t &arith) {
        return plan(t.mem, t.reg, form, arith);
    }

    /// Plans the steps storing a value: from the registers into memory
    ///
    /// @param t The value's types
    /// @param form The storage category of the operands
    /// @param arith The dialect every step must be supported by
    bool plan_store(const type_parameter_t &t, form_t form,
            const operation::arithmetic_t &arith) {
        return plan(t.reg, t.mem, form, arith);
    }

    /// Forgets every step
    void clear() { n_ = 0; }

    /// The number of steps
    int size() const { return n_; }

    /// Whether there is nothing to convert
    bool empty() const { return n_ == 0; }

    /// The step `i`
    ///
    /// @param i The step's index
    const conversion::step_t &step(int i) const { return steps_[i]; }

    /// The type converted
    type_t from() const { return n_ ? steps_[0].from : type_t::none; }

    /// The type produced
    type_t to() const { return n_ ? steps_[n_ - 1].to : type_t::none; }

    /// How many times wider the widest type is than the narrowest one
    int ratio() const {
        if (!n_) return 1;
        const int a = sewb(from()), b = sewb(to());
        return a > b ? a / b : b / a;
    }

    /// The operation of step `i`, with the registers `dst` and `src`
    ///
    /// @param i The step's index
    /// @param dst The destination register
    /// @param src The source register
    arithmetic::op_t op(int i, name_t dst, name_t src) const {
        const conversion::step_t &s = steps_[i];
        return arithmetic::op_t::convert(s.to, s.from, form_).names(dst, src);
    }

private:
    conversion::step_t steps_[CAPACITY];
    int n_ = 0;
    form_t form_ = form_t::vector;

    void push_(conversion::kind_t k, type_t from, type_t to) {
        conversion::step_t &s = steps_[n_++];
        s.kind = k;
        s.from = from;
        s.to = to;
    }

    // The integer type of `width` bytes and the signedness asked for
    static type_t int_of_(int width, bool sign) {
        switch (width) {
            case 1: return sign ? type_t::s8 : type_t::u8;
            case 2: return sign ? type_t::s16 : type_t::u16;
            case 4: return sign ? type_t::s32 : type_t::u32;
            case 8: return sign ? type_t::s64 : type_t::none;
            default: return type_t::none;
        }
    }

    bool plan_(type_t from, type_t to) {
        const int wf = sewb(from), wt = sewb(to);
        CHECK(wf > 0 && wt > 0, "a conversion of an unsized type");
        const bool fi = is_int(from), ti = is_int(to);

        // Floats change width by two at most
        if (!fi && !ti) {
            CHECK(wf == 2 * wt || wt == 2 * wf, "a float pair beyond 2x");
            push_(wt > wf ? conversion::kind_t::widen
                          : conversion::kind_t::narrow,
                    from, to);
            return true;
        }

        // Integers grow once, by a widening when doubling, as it reads at the
        // narrow width, else by an extension; they shrink by halves
        if (fi && ti) {
            if (wt > wf) {
                push_(wt == 2 * wf ? conversion::kind_t::widen
                                   : conversion::kind_t::extend,
                        from, to);
                return true;
            }
            const bool sign = is_signed_int(from);
            CHECK(sign == is_signed_int(to) || wt == wf,
                    "a narrowing that changes signedness");
            return narrow_ints_(from, to);
        }

        // An integer reaches the float's width first, then converts
        if (fi) {
            CHECK(wt >= wf, "an integer wider than its float");
            const type_t at = int_of_(wt, is_signed_int(from));
            CHECK(at != type_t::none, "no integer of the float's width");
            if (at != from) push_(conversion::kind_t::extend, from, at);
            push_(conversion::kind_t::convert, at, to);
            return true;
        }

        // A float converts at its width, then halves down to the integer's
        CHECK(wf >= wt, "an integer wider than the float converted");
        const type_t at = int_of_(wf, is_signed_int(to));
        CHECK(at != type_t::none, "no integer of the float's width");
        push_(conversion::kind_t::convert, from, at);
        return narrow_ints_(at, to);
    }

    bool narrow_ints_(type_t from, type_t to) {
        const bool sign = is_signed_int(to);
        type_t cur = from;
        while (sewb(cur) > sewb(to)) {
            const type_t next = int_of_(sewb(cur) / 2, sign);
            CHECK(n_ < CAPACITY, "too many steps");
            push_(conversion::kind_t::narrow, cur, next);
            cur = next;
        }
        return true;
    }

    bool supported_(const operation::arithmetic_t &arith) const {
        for (int i = 0; i < n_; ++i) {
            const arithmetic::op_t o = op(i, 0, 0);
            bool ok = false;
            switch (steps_[i].kind) {
                case conversion::kind_t::extend:
                    ok = arith.supports_extend(o);
                    break;
                case conversion::kind_t::widen:
                    ok = arith.supports_widen(o);
                    break;
                case conversion::kind_t::convert:
                    ok = arith.supports_convert(o);
                    break;
                default: ok = arith.supports_narrow(o); break;
            }
            CHECK(ok, "an unsupported step");
        }
        return true;
    }
};

} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
